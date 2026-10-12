import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math
import random
import pickle
from collections import Counter
import wandb
from huggingface_hub import login, HfApi

# --- Vocabulary ---
class WordVocab:
    def __init__(self, plain_file, max_vocab_size=10000):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3, '<MASK>': 4}
        
        # 1. Count word frequencies
        word_counts = Counter()
        with open(plain_file, 'r') as f:
            for line in f:
                word_counts.update(line.strip().split())
                
        # 2. Keep only the most common words
        most_common = word_counts.most_common(max_vocab_size - len(self.word2idx))
        
        for word, _ in most_common:
            self.word2idx[word] = len(self.word2idx)
            
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def encode(self, text):
        return [self.word2idx.get(w, self.word2idx['<UNK>']) for w in text.split()]

# --- Dataset ---
class DominionLanguageDataset(Dataset):
    def __init__(self, plain_file, vocab, task='NWP', max_len=128, mask_prob=0.15):
        self.vocab = vocab
        self.task = task
        self.max_len = max_len
        self.mask_prob = mask_prob
        
        with open(plain_file, 'r') as f:
            self.lines = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self): 
        return len(self.lines)

    def __getitem__(self, idx):
        tokens = self.vocab.encode(self.lines[idx])
        
        # Truncate to max length
        tokens = tokens[:self.max_len]
        
        if self.task == 'NWP':
            # Next Word Prediction: Input = seq[:-1], Target = seq[1:]
            input_seq = torch.tensor([self.vocab.word2idx['<SOS>']] + tokens)
            target_seq = torch.tensor(tokens + [self.vocab.word2idx['<EOS>']])
            return input_seq[:self.max_len], target_seq[:self.max_len]
            
        elif self.task == 'MLM':
            # Masked Language Modeling: randomly mask 15% of tokens
            input_seq = tokens.copy()
            target_seq = [-100] * len(tokens) # -100 is ignored by CrossEntropyLoss
            
            for i in range(len(tokens)):
                if random.random() < self.mask_prob:
                    input_seq[i] = self.vocab.word2idx['<MASK>']
                    target_seq[i] = tokens[i] # Only calculate loss on masked tokens
                    
            return torch.tensor(input_seq), torch.tensor(target_seq)

def lm_collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_pad = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_pad = pad_sequence(targets, batch_first=True, padding_value=-100) 
    return inputs_pad, targets_pad

# --- SSM Model ---
class SimplifiedSSMLayer(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        self.A = nn.Parameter(torch.rand(d_model, d_state))
        self.B = nn.Parameter(torch.rand(d_model, d_state))
        self.C = nn.Parameter(torch.rand(d_model, d_state))
        self.D = nn.Parameter(torch.rand(d_model))
        
    def forward(self, x):
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.d_model, self.d_state, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            xt = x[:, t, :] 
            h = self.A * h + self.B * xt.unsqueeze(-1)
            yt = (self.C * h).sum(dim=-1) + self.D * xt
            outputs.append(yt)
            
        return torch.stack(outputs, dim=1)

class SSM_NWP(nn.Module):
    def __init__(self, vocab_size, d_model=256, d_state=256, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.ssm = SimplifiedSSMLayer(d_model, d_state)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        ssm_out = self.ssm(emb)
        ssm_out = self.dropout(ssm_out)
        return self.fc_out(ssm_out)

# --- Bi-LSTM MLM Model ---
class ManualLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.w_ih = nn.Linear(input_size, 4 * hidden_size)
        self.w_hh = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, states):
        h, c = states
        gates = self.w_ih(x) + self.w_hh(h)
        i, f, g, o = gates.chunk(4, dim=1)
        
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next

class CustomBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell_forward = ManualLSTMCell(input_size, hidden_size)
        self.cell_backward = ManualLSTMCell(input_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        device = x.device

        h_f = torch.zeros(batch_size, self.hidden_size, device=device)
        c_f = torch.zeros(batch_size, self.hidden_size, device=device)
        h_b = torch.zeros(batch_size, self.hidden_size, device=device)
        c_b = torch.zeros(batch_size, self.hidden_size, device=device)

        outputs_forward = []
        outputs_backward = []

        for t in range(seq_len):
            h_f, c_f = self.cell_forward(x[:, t, :], (h_f, c_f))
            outputs_forward.append(h_f)

        for t in range(seq_len - 1, -1, -1):
            h_b, c_b = self.cell_backward(x[:, t, :], (h_b, c_b))
            outputs_backward.insert(0, h_b)

        outputs_forward = torch.stack(outputs_forward, dim=1)
        outputs_backward = torch.stack(outputs_backward, dim=1)
        outputs = torch.cat((outputs_forward, outputs_backward), dim=2)
        
        return outputs

class BiLSTM_MLM(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.bilstm = CustomBiLSTM(emb_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out = self.bilstm(emb) 
        lstm_out = self.dropout(lstm_out)
        return self.fc_out(lstm_out)

# --- Training & Evaluation ---
def train_and_evaluate_custom(model, train_loader, val_loader, epochs=20, lr=0.001, project_name="dominion-lm"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    wandb.init(project=project_name, config={"epochs": epochs, "lr": lr, "batch_size": 64})
    wandb.watch(model, log="all", log_freq=10)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = amp.GradScaler()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            with amp.autocast():
                outputs = model(inputs) 
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                wandb.log({"train_loss": loss.item()})
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with amp.autocast():
                    outputs = model(inputs)
                    outputs = outputs.view(-1, outputs.size(-1))
                    targets = targets.view(-1)
                    val_loss += criterion(outputs, targets).item()
                
        avg_val_loss = val_loss / len(val_loader)
        perplexity = math.exp(avg_val_loss)
        
        print(f"--- Epoch {epoch+1} Complete | Val Loss: {avg_val_loss:.4f} | Perplexity: {perplexity:.2f} ---")
        wandb.log({"val_loss": avg_val_loss, "perplexity": perplexity, "epoch": epoch + 1})
    
    wandb.finish()
    return model

def final_evaluation(model, test_loader, task_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    for module in model.modules():
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()
        module._backward_hooks.clear()
    
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
    avg_loss = total_loss / len(test_loader)
    final_perplexity = math.exp(avg_loss)
    return final_perplexity

# --- Inference ---
def predict_next_word_topk(model, vocab, text, device, k=5):
    model.eval()
    tokens = [vocab.word2idx['<SOS>']] + vocab.encode(text)
    input_tensor = torch.tensor([tokens]).to(device) 
    
    with torch.no_grad():
        outputs = model(input_tensor) 
        last_step_logits = outputs[0, -1, :]
        last_step_logits[vocab.word2idx['<UNK>']] = -float('Inf')
        probs = F.softmax(last_step_logits, dim=0)
        top_probs, top_indices = torch.topk(probs, k)
        choice = torch.multinomial(top_probs, 1).item()
        predicted_idx = top_indices[choice].item()
        
    return vocab.idx2word.get(predicted_idx, '<UNK>')

def predict_masked_word_topk(model, vocab, text, device, k=5):
    model.eval()
    tokens = vocab.encode(text)
    mask_idx = tokens.index(vocab.word2idx['<MASK>'])
    input_tensor = torch.tensor([tokens]).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor) 
        mask_step_logits = outputs[0, mask_idx, :]
        mask_step_logits[vocab.word2idx['<UNK>']] = -float('Inf')
        probs = F.softmax(mask_step_logits, dim=0)
        top_probs, top_indices = torch.topk(probs, k)
        choice = torch.multinomial(top_probs, 1).item()
        predicted_idx = top_indices[choice].item()
        
    return vocab.idx2word.get(predicted_idx, '<UNK>')

def generate_full_transmission(model, vocab, prompt, device, max_words=30, k=5):
    model.eval()
    current_text = prompt
    generated_words = []
    
    for _ in range(max_words):
        next_word = predict_next_word_topk(model, vocab, current_text, device, k)
        if next_word == '<EOS>':
            break
        generated_words.append(next_word)
        current_text += " " + next_word
        
    return prompt + " " + " ".join(generated_words)
