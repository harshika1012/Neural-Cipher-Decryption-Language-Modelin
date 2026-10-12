# # import os
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from torch.utils.data import Dataset, DataLoader
# # from torch.nn.utils.rnn import pad_sequence
# # from tqdm import tqdm
# # import Levenshtein
# # import wandb
# # from huggingface_hub import login, HfApi

# # # --- Preprocessing & Dataset ---
# # def parse_cipher(line):
# #     # Rule: '9' -> space (let's use a unique int, e.g., 99)
# #     # Other chars are 2-digit numbers
# #     tokens = []
# #     i = 0
# #     while i < len(line.strip()):
# #         if line[i] == '9':
# #             tokens.append(99)
# #             i += 1
# #         else:
# #             tokens.append(int(line[i:i+2]))
# #             i += 2
# #     # Shift by +2 to reserve 0 for <PAD> and 1 for <SOS>
# #     return [t + 2 for t in tokens]

# # class CipherDataset(Dataset):
# #     def __init__(self, plain_file, cipher_file, max_len=128):
# #         with open(plain_file, 'r') as f: self.plain_lines = f.readlines()
# #         with open(cipher_file, 'r') as f: self.cipher_lines = f.readlines()
        
# #         # Build Char Vocab for Plain Text
# #         self.chars = sorted(list(set("".join(self.plain_lines) + "\n")))
# #         self.char2idx = {ch: i + 2 for i, ch in enumerate(self.chars)}
# #         self.char2idx['<PAD>'] = 0
# #         self.char2idx['<SOS>'] = 1
# #         self.char2idx['<EOS>'] = len(self.char2idx)
# #         self.idx2char = {v: k for k, v in self.char2idx.items()}
# #         self.max_len = max_len

# #     def __len__(self): return len(self.plain_lines)

# #     def __getitem__(self, idx):
# #         # 1. Parse the cipher text into integer tokens
# #         c_raw = parse_cipher(self.cipher_lines[idx])
        
# #         # 2. Map plain text characters to indices
# #         p_text = self.plain_lines[idx].strip()
# #         p_raw = [self.char2idx['<SOS>']] + [self.char2idx[ch] for ch in p_text] + [self.char2idx['<EOS>']]
        
# #         # 3. Apply MAX_LEN truncation
# #         c_tokens = torch.tensor(c_raw[:self.max_len])
# #         p_tokens = torch.tensor(p_raw[:self.max_len])
    
# #         return c_tokens, p_tokens

# # def collate_fn(batch):
# #     c_side, p_side = zip(*batch)
# #     c_pad = pad_sequence(c_side, batch_first=True, padding_value=0)
# #     p_pad = pad_sequence(p_side, batch_first=True, padding_value=0)
# #     return c_pad, p_pad

# # # --- Models ---
# # class ManualRNNCell(nn.Module):
# #     def __init__(self, input_size, hidden_size):
# #         super().__init__()
# #         self.xh = nn.Linear(input_size, hidden_size)
# #         self.hh = nn.Linear(hidden_size, hidden_size)

# #     def forward(self, x, h):
# #         return torch.tanh(self.xh(x) + self.hh(h))

# # class ManualLSTMCell(nn.Module):
# #     def __init__(self, input_size, hidden_size):
# #         super().__init__()
# #         self.hidden_size = hidden_size
# #         self.w_ih = nn.Linear(input_size, 4 * hidden_size)
# #         self.w_hh = nn.Linear(hidden_size, 4 * hidden_size)
        
# #         # Initialize the forget gate bias to 1.0
# #         with torch.no_grad():
# #             # (i, f, g, o), 'f' is the second chunk
# #             start, end = hidden_size, 2 * hidden_size
# #             self.w_ih.bias[start:end].fill_(1.0)
# #             self.w_hh.bias[start:end].fill_(1.0)

# #     def forward(self, x, states):
# #         h, c = states
# #         # Compute all gate inputs in two parallel linear ops
# #         gates = self.w_ih(x) + self.w_hh(h)
        
# #         i, f, g, o = gates.chunk(4, dim=1)
        
# #         c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
# #         h_next = torch.sigmoid(o) * torch.tanh(c_next)
# #         return h_next, c_next
# # class Attention(nn.Module):
# #     def __init__(self, hid_dim):
# #         super().__init__()
# #         # 1. Added learned projections (The Luong Dot-Product Upgrade)
# #         self.W_q = nn.Linear(hid_dim, hid_dim)
# #         self.W_k = nn.Linear(hid_dim, hid_dim)
# #         self.W_v = nn.Linear(hid_dim, hid_dim)
# #         self.scale = hid_dim ** -0.5

# #     def forward(self, dec_hidden, enc_outputs, mask=None):
# #         # dec_hidden: (batch, hid_dim) -> (batch, 1, hid_dim)
# #         q = self.W_q(dec_hidden).unsqueeze(1)
        
# #         # enc_outputs: (batch, src_len, hid_dim)
# #         k = self.W_k(enc_outputs)
# #         v = self.W_v(enc_outputs)

# #         # 2. Scaled Dot-Product
# #         # scores: (batch, 1, src_len)
# #         scores = torch.bmm(q, k.transpose(1, 2)) * self.scale

# #         # 3. Apply Mask before softmax
# #         if mask is not None:
# #             # mask is (batch, src_len), needs to be (batch, 1, src_len)
# #             scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e10)

# #         attention_weights = torch.softmax(scores, dim=-1)

# #         # Return the weights AND the projected values
# #         return attention_weights.squeeze(1), v
    
# # class Attentionprev(nn.Module):
# #     def __init__(self, hid_dim):
# #         super().__init__()
# #         self.w = nn.Linear(hid_dim * 2, hid_dim)
# #         self.v = nn.Linear(hid_dim, 1, bias=False)

# #     def forward(self, dec_hidden, enc_outputs, mask=None):
# #         src_len = enc_outputs.shape[1]
# #         dec_hidden_rep = dec_hidden.unsqueeze(1).repeat(1, src_len, 1)
        
# #         energy = torch.tanh(self.w(torch.cat((dec_hidden_rep, enc_outputs), dim=2)))
# #         attention = self.v(energy).squeeze(2) # (batch_size, src_len)
        
# #         if mask is not None:
# #             attention = attention.masked_fill(mask == 0, -1e10)
            
# #         return torch.softmax(attention, dim=1)

# # class Seq2Seq(nn.Module):
# #     def __init__(self, input_dim, output_dim, emb_dim, hid_dim, mode='LSTM'):
# #         super().__init__()
# #         self.mode = mode
# #         self.hid_dim = hid_dim
        
# #         self.embedding = nn.Embedding(input_dim, emb_dim)
# #         self.out_embedding = nn.Embedding(output_dim, emb_dim)
# #         self.attention = Attention(hid_dim)
        
# #         if mode == 'RNN':
# #             self.enc_cell = ManualRNNCell(emb_dim, hid_dim)
# #             self.dec_cell = ManualRNNCell(emb_dim + hid_dim, hid_dim)
# #         else:
# #             self.enc_cell = ManualLSTMCell(emb_dim, hid_dim)
# #             self.dec_cell = ManualLSTMCell(emb_dim + hid_dim, hid_dim)
            
# #         self.fc_out = nn.Linear(hid_dim * 2 + emb_dim, output_dim)

# #     def forward(self, src, trg, teacher_forcing_ratio=0.5):
# #         batch_size, src_len = src.shape
# #         trg_len = trg.shape[1]
# #         outputs = torch.zeros(trg_len, batch_size, self.fc_out.out_features).to(src.device)
        
# #         # --- ENCODER ---
# #         embedded = self.embedding(src)
# #         h = torch.zeros(batch_size, self.hid_dim).to(src.device)
# #         c = torch.zeros(batch_size, self.hid_dim).to(src.device) if self.mode == 'LSTM' else None
        
# #         enc_outputs = torch.zeros(batch_size, src_len, self.hid_dim).to(src.device)
        
# #         for t in range(src_len):
# #             if self.mode == 'RNN': h = self.enc_cell(embedded[:, t, :], h)
# #             else: h, c = self.enc_cell(embedded[:, t, :], (h, c))
# #             enc_outputs[:, t, :] = h
            
# #         # --- DECODER ---
# #         src_mask = (src != 0) 
# #         input_token = trg[:, 0]
        
# #         for t in range(1, trg_len):
# #             emb_dec = self.out_embedding(input_token)
            
# #             a = self.attention(h, enc_outputs, mask=src_mask) 
# #             a = a.unsqueeze(1)
            
# #             context = torch.bmm(a, enc_outputs).squeeze(1) # (batch, hid_dim)
            
# #             dec_input = torch.cat((emb_dec, context), dim=1)
            
# #             if self.mode == 'RNN': h = self.dec_cell(dec_input, h)
# #             else: h, c = self.dec_cell(dec_input, (h, c))
            
# #             output = self.fc_out(torch.cat((h, context, emb_dec), dim=1))
# #             outputs[t] = output
            
# #             top1 = output.argmax(1)
# #             input_token = trg[:, t] if torch.rand(1) < teacher_forcing_ratio else top1
            
# #         return outputs.permute(1, 0, 2)

# # # --- Metrics ---
# # def calculate_metrics(preds, targets, idx2char):
# #     char_acc, word_acc, total_lev = 0, 0, 0
# #     batch_size = preds.size(0)
    
# #     for i in range(batch_size):
# #         # 1. Convert to lists of strings
# #         p_raw = [idx2char[idx.item()] for idx in preds[i]]
# #         t_raw = [idx2char[idx.item()] for idx in targets[i]]
        
# #         # 2. Cut off everything after <EOS>
# #         def truncate_at_eos(char_list):
# #             if '<EOS>' in char_list:
# #                 return char_list[:char_list.index('<EOS>')]
# #             return char_list
            
# #         p_clean = truncate_at_eos(p_raw)
# #         t_clean = truncate_at_eos(t_raw)
        
# #         # 3. Join into strings, ignoring PAD and SOS
# #         p_str = "".join([c for c in p_clean if c not in ['<PAD>', '<SOS>']])
# #         t_str = "".join([c for c in t_clean if c not in ['<PAD>', '<SOS>']])
        
# #         # Char Accuracy
# #         char_acc += sum(1 for a, b in zip(p_str, t_str) if a == b) / max(len(t_str), 1)
        
# #         # Word Accuracy
# #         p_words = p_str.split()
# #         t_words = t_str.split()
# #         correct_words = sum(1 for pw, tw in zip(p_words, t_words) if pw == tw)
# #         word_acc += correct_words / max(len(t_words), 1)
        
# #         # Levenshtein Distance
# #         total_lev += Levenshtein.distance(p_str, t_str)
        
# #     return char_acc/batch_size, word_acc/batch_size, total_lev/batch_size

# # def log_samples(model, loader, idx2char, device, num_samples=5):
# #     model.eval()
# #     samples = []

# #     with torch.no_grad():
# #         for src, trg in loader:
# #             src, trg = src.to(device), trg.to(device)
# #             output = model(src, trg, teacher_forcing_ratio=0)
# #             preds = output.argmax(2)

# #             for i in range(src.size(0)):
# #                 if len(samples) >= num_samples:
# #                     wandb.log({"samples": samples})
# #                     return

# #                 p_str = "".join([idx2char[idx.item()] for idx in preds[i] if idx.item() > 2])
# #                 t_str = "".join([idx2char[idx.item()] for idx in trg[i] if idx.item() > 2])

# #                 samples.append({
# #                     "target": t_str,
# #                     "prediction": p_str
# #                 })

# # # --- Training / Evaluation ---
# # def train_epoch(model, loader, optimizer, criterion, device, teacher_forcing_ratio=0.5):
# #     model.train()
# #     epoch_loss = 0
# #     progress_bar = tqdm(loader, desc="Training", leave=False)
    
# #     for src, trg in progress_bar:
# #         src, trg = src.to(device), trg.to(device)
# #         optimizer.zero_grad()
        
# #         # The forward method takes teacher_forcing_ratio
# #         output = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)
# #         output_dim = output.shape[-1]
        
# #         output = output[:, 1:].reshape(-1, output_dim)
# #         trg = trg[:, 1:].reshape(-1)
        
# #         loss = criterion(output, trg)
# #         loss.backward()
        
# #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# #         optimizer.step()
# #         epoch_loss += loss.item()
        
# #         progress_bar.set_postfix({'batch_loss': f"{loss.item():.4f}"})
        
# #     return epoch_loss / len(loader)

# # def evaluate(model, loader, criterion, device, idx2char):
# #     model.eval()
# #     epoch_loss = 0
# #     all_char_acc, all_word_acc, all_lev = 0, 0, 0
    
# #     with torch.no_grad():
# #         for src, trg in loader:
# #             src, trg = src.to(device), trg.to(device)
# #             output = model(src, trg, teacher_forcing_ratio=0) 
            
# #             output_dim = output.shape[-1]
# #             loss = criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))
# #             epoch_loss += loss.item()
            
# #             preds = output.argmax(2)
# #             c_acc, w_acc, lev = calculate_metrics(preds, trg, idx2char)
# #             all_char_acc += c_acc
# #             all_word_acc += w_acc
# #             all_lev += lev
            
# #     n = len(loader)
# #     return epoch_loss/n, all_char_acc/n, all_word_acc/n, all_lev/n

# # def print_samples(model, loader, idx2char, device, num_samples=5):
# #     model.eval()
# #     samples_shown = 0
    
# #     with torch.no_grad():
# #         for src, trg in loader:
# #             src, trg = src.to(device), trg.to(device)
# #             output = model(src, trg, teacher_forcing_ratio=0)
# #             preds = output.argmax(2)
            
# #             for i in range(src.size(0)):
# #                 if samples_shown >= num_samples: return
                
# #                 p_str = "".join([idx2char[idx.item()] for idx in preds[i] if idx.item() > 2])
# #                 t_str = "".join([idx2char[idx.item()] for idx in trg[i] if idx.item() > 2])
                
# #                 print(f"Sample {samples_shown + 1}:")
# #                 print(f"  Target: {t_str}")
# #                 print(f"  Pred:   {p_str}")
# #                 print("-" * 20)
# #                 samples_shown += 1


# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torch.nn.utils.rnn import pad_sequence
# from tqdm import tqdm
# import Levenshtein
# import wandb
# from huggingface_hub import login, HfApi

# # --- Preprocessing & Dataset ---
# def parse_cipher(line):
#     # Rule: '9' -> space (let's use a unique int, e.g., 99)
#     # Other chars are 2-digit numbers
#     tokens = []
#     i = 0
#     while i < len(line.strip()):
#         if line[i] == '9':
#             tokens.append(99)
#             i += 1
#         else:
#             tokens.append(int(line[i:i+2]))
#             i += 2
#     # Shift by +2 to reserve 0 for <PAD> and 1 for <SOS>
#     return [t + 2 for t in tokens]

# class CipherDataset(Dataset):
#     def __init__(self, plain_file, cipher_file, max_len=128):
#         with open(plain_file, 'r') as f: self.plain_lines = f.readlines()
#         with open(cipher_file, 'r') as f: self.cipher_lines = f.readlines()
        
#         # Build Char Vocab for Plain Text
#         self.chars = sorted(list(set("".join(self.plain_lines) + "\n")))
#         self.char2idx = {ch: i + 2 for i, ch in enumerate(self.chars)}
#         self.char2idx['<PAD>'] = 0
#         self.char2idx['<SOS>'] = 1
#         self.char2idx['<EOS>'] = len(self.char2idx)
#         self.idx2char = {v: k for k, v in self.char2idx.items()}
#         self.max_len = max_len

#     def __len__(self): return len(self.plain_lines)

#     def __getitem__(self, idx):
#         # 1. Parse the cipher text into integer tokens
#         c_raw = parse_cipher(self.cipher_lines[idx])
        
#         # 2. Map plain text characters to indices
#         p_text = self.plain_lines[idx].strip()
#         p_raw = [self.char2idx['<SOS>']] + [self.char2idx[ch] for ch in p_text] + [self.char2idx['<EOS>']]
        
#         # 3. Apply MAX_LEN truncation
#         c_tokens = torch.tensor(c_raw[:self.max_len])
#         p_tokens = torch.tensor(p_raw[:self.max_len])
    
#         return c_tokens, p_tokens

# def collate_fn(batch):
#     c_side, p_side = zip(*batch)
#     c_pad = pad_sequence(c_side, batch_first=True, padding_value=0)
#     p_pad = pad_sequence(p_side, batch_first=True, padding_value=0)
#     return c_pad, p_pad

# # --- Models ---
# class ManualRNNCell(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super().__init__()
#         self.xh = nn.Linear(input_size, hidden_size)
#         self.hh = nn.Linear(hidden_size, hidden_size)

#     def forward(self, x, h):
#         return torch.tanh(self.xh(x) + self.hh(h))

# class ManualLSTMCell(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.w_ih = nn.Linear(input_size, 4 * hidden_size)
#         self.w_hh = nn.Linear(hidden_size, 4 * hidden_size)
        
#         # Initialize the forget gate bias to 1.0
#         with torch.no_grad():
#             # (i, f, g, o), 'f' is the second chunk
#             start, end = hidden_size, 2 * hidden_size
#             self.w_ih.bias[start:end].fill_(1.0)
#             self.w_hh.bias[start:end].fill_(1.0)

#     def forward(self, x, states):
#         h, c = states
#         # Compute all gate inputs in two parallel linear ops
#         gates = self.w_ih(x) + self.w_hh(h)
        
#         i, f, g, o = gates.chunk(4, dim=1)
        
#         c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
#         h_next = torch.sigmoid(o) * torch.tanh(c_next)
#         return h_next, c_next

# class Attention(nn.Module):
#     def __init__(self, hid_dim):
#         super().__init__()
#         self.W_q = nn.Linear(hid_dim, hid_dim)
#         self.W_k = nn.Linear(hid_dim, hid_dim)
#         self.W_v = nn.Linear(hid_dim, hid_dim)

#     def forward(self, dec_hidden, enc_outputs, mask=None):
#         # Bahdanau-style but using QKV projections
#         query = self.W_q(dec_hidden).unsqueeze(1) # (batch, 1, hid)
#         keys = self.W_k(enc_outputs)              # (batch, seq, hid)
        
#         # Calculate attention scores (energy)
#         energy = torch.bmm(query, keys.permute(0, 2, 1)).squeeze(1) # (batch, seq)
        
#         if mask is not None:
#             energy = energy.masked_fill(mask == 0, -1e10)
            
#         return torch.softmax(energy, dim=1)

# class Seq2Seq(nn.Module):
#     def __init__(self, input_dim, output_dim, emb_dim, hid_dim, mode='LSTM'):
#         super().__init__()
#         self.mode = mode
#         self.hid_dim = hid_dim
        
#         self.embedding = nn.Embedding(input_dim, emb_dim)
#         self.out_embedding = nn.Embedding(output_dim, emb_dim)
#         self.attention = Attention(hid_dim)
        
#         if mode == 'RNN':
#             self.enc_cell = ManualRNNCell(emb_dim, hid_dim)
#             self.dec_cell = ManualRNNCell(emb_dim + hid_dim, hid_dim)
#         else:
#             self.enc_cell = ManualLSTMCell(emb_dim, hid_dim)
#             self.dec_cell = ManualLSTMCell(emb_dim + hid_dim, hid_dim)
            
#         self.fc_out = nn.Linear(hid_dim * 2 + emb_dim, output_dim)

#     def forward(self, src, trg, teacher_forcing_ratio=0.5):
#         batch_size, src_len = src.shape
#         trg_len = trg.shape[1]
#         outputs = torch.zeros(trg_len, batch_size, self.fc_out.out_features).to(src.device)
        
#         # --- ENCODER ---
#         embedded = self.embedding(src)
#         h = torch.zeros(batch_size, self.hid_dim).to(src.device)
#         c = torch.zeros(batch_size, self.hid_dim).to(src.device) if self.mode == 'LSTM' else None
        
#         enc_outputs = torch.zeros(batch_size, src_len, self.hid_dim).to(src.device)
        
#         for t in range(src_len):
#             if self.mode == 'RNN': h = self.enc_cell(embedded[:, t, :], h)
#             else: h, c = self.enc_cell(embedded[:, t, :], (h, c))
#             enc_outputs[:, t, :] = h
            
#         # --- DECODER ---
#         src_mask = (src != 0) 
#         input_token = trg[:, 0]
        
#         for t in range(1, trg_len):
#             emb_dec = self.out_embedding(input_token)
            
#             a = self.attention(h, enc_outputs, mask=src_mask) 
#             a = a.unsqueeze(1)
            
#             # Apply W_v projection to encoder outputs before weighted sum
#             values = self.attention.W_v(enc_outputs)
#             context = torch.bmm(a, values).squeeze(1) # (batch, hid_dim)
            
#             dec_input = torch.cat((emb_dec, context), dim=1)
            
#             if self.mode == 'RNN': h = self.dec_cell(dec_input, h)
#             else: h, c = self.dec_cell(dec_input, (h, c))
            
#             output = self.fc_out(torch.cat((h, context, emb_dec), dim=1))
#             outputs[t] = output
            
#             top1 = output.argmax(1)
#             input_token = trg[:, t] if torch.rand(1) < teacher_forcing_ratio else top1
            
#         return outputs.permute(1, 0, 2)

# # --- Metrics ---
# def calculate_metrics(preds, targets, idx2char):
#     char_acc, word_acc, total_lev = 0, 0, 0
#     batch_size = preds.size(0)
    
#     for i in range(batch_size):
#         # 1. Convert to lists of strings
#         p_raw = [idx2char[idx.item()] for idx in preds[i]]
#         t_raw = [idx2char[idx.item()] for idx in targets[i]]
        
#         # 2. Cut off everything after <EOS>
#         def truncate_at_eos(char_list):
#             if '<EOS>' in char_list:
#                 return char_list[:char_list.index('<EOS>')]
#             return char_list
            
#         p_clean = truncate_at_eos(p_raw)
#         t_clean = truncate_at_eos(t_raw)
        
#         # 3. Join into strings, ignoring PAD and SOS
#         p_str = "".join([c for c in p_clean if c not in ['<PAD>', '<SOS>']])
#         t_str = "".join([c for c in t_clean if c not in ['<PAD>', '<SOS>']])
        
#         # Char Accuracy
#         char_acc += sum(1 for a, b in zip(p_str, t_str) if a == b) / max(len(t_str), 1)
        
#         # Word Accuracy
#         p_words = p_str.split()
#         t_words = t_str.split()
#         correct_words = sum(1 for pw, tw in zip(p_words, t_words) if pw == tw)
#         word_acc += correct_words / max(len(t_words), 1)
        
#         # Levenshtein Distance
#         total_lev += Levenshtein.distance(p_str, t_str)
        
#     return char_acc/batch_size, word_acc/batch_size, total_lev/batch_size

# def log_samples(model, loader, idx2char, device, num_samples=5):
#     model.eval()
#     samples = []

#     with torch.no_grad():
#         for src, trg in loader:
#             src, trg = src.to(device), trg.to(device)
#             output = model(src, trg, teacher_forcing_ratio=0)
#             preds = output.argmax(2)

#             for i in range(src.size(0)):
#                 if len(samples) >= num_samples:
#                     wandb.log({"samples": samples})
#                     return

#                 p_str = "".join([idx2char[idx.item()] for idx in preds[i] if idx.item() > 2])
#                 t_str = "".join([idx2char[idx.item()] for idx in trg[i] if idx.item() > 2])

#                 samples.append({
#                     "target": t_str,
#                     "prediction": p_str
#                 })

# # --- Training / Evaluation ---
# def train_epoch(model, loader, optimizer, criterion, device, teacher_forcing_ratio=0.5):
#     model.train()
#     epoch_loss = 0
#     progress_bar = tqdm(loader, desc="Training", leave=False)
    
#     for src, trg in progress_bar:
#         src, trg = src.to(device), trg.to(device)
#         optimizer.zero_grad()
        
#         # The forward method takes teacher_forcing_ratio
#         output = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)
#         output_dim = output.shape[-1]
        
#         output = output[:, 1:].reshape(-1, output_dim)
#         trg = trg[:, 1:].reshape(-1)
        
#         loss = criterion(output, trg)
#         loss.backward()
        
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
#         epoch_loss += loss.item()
        
#         progress_bar.set_postfix({'batch_loss': f"{loss.item():.4f}"})
        
#     return epoch_loss / len(loader)

# def evaluate(model, loader, criterion, device, idx2char):
#     model.eval()
#     epoch_loss = 0
#     all_char_acc, all_word_acc, all_lev = 0, 0, 0
    
#     with torch.no_grad():
#         for src, trg in loader:
#             src, trg = src.to(device), trg.to(device)
#             output = model(src, trg, teacher_forcing_ratio=0) 
            
#             output_dim = output.shape[-1]
#             loss = criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))
#             epoch_loss += loss.item()
            
#             preds = output.argmax(2)
#             c_acc, w_acc, lev = calculate_metrics(preds, trg, idx2char)
#             all_char_acc += c_acc
#             all_word_acc += w_acc
#             all_lev += lev
            
#     n = len(loader)
#     return epoch_loss/n, all_char_acc/n, all_word_acc/n, all_lev/n

# def print_samples(model, loader, idx2char, device, num_samples=5):
#     model.eval()
#     samples_shown = 0
    
#     with torch.no_grad():
#         for src, trg in loader:
#             src, trg = src.to(device), trg.to(device)
#             output = model(src, trg, teacher_forcing_ratio=0)
#             preds = output.argmax(2)
            
#             for i in range(src.size(0)):
#                 if samples_shown >= num_samples: return
                
#                 p_str = "".join([idx2char[idx.item()] for idx in preds[i] if idx.item() > 2])
#                 t_str = "".join([idx2char[idx.item()] for idx in trg[i] if idx.item() > 2])
                
#                 print(f"Sample {samples_shown + 1}:")
#                 print(f"  Target: {t_str}")
#                 print(f"  Pred:   {p_str}")
#                 print("-" * 20)
#                 samples_shown += 1

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import Levenshtein
import wandb
from huggingface_hub import login, HfApi

# --- Preprocessing & Dataset ---
def parse_cipher(line):
    # Rule: '9' -> space (let's use a unique int, e.g., 99)
    # Other chars are 2-digit numbers
    tokens = []
    i = 0
    while i < len(line.strip()):
        if line[i] == '9':
            tokens.append(99)
            i += 1
        else:
            tokens.append(int(line[i:i+2]))
            i += 2
    # Shift by +2 to reserve 0 for <PAD> and 1 for <SOS>
    return [t + 2 for t in tokens]

class CipherDataset(Dataset):
    def __init__(self, plain_file, cipher_file, max_len=128):
        with open(plain_file, 'r') as f: self.plain_lines = f.readlines()
        with open(cipher_file, 'r') as f: self.cipher_lines = f.readlines()
        
        # Build Char Vocab for Plain Text
        self.chars = sorted(list(set("".join(self.plain_lines) + "\n")))
        self.char2idx = {ch: i + 2 for i, ch in enumerate(self.chars)}
        self.char2idx['<PAD>'] = 0
        self.char2idx['<SOS>'] = 1
        self.char2idx['<EOS>'] = len(self.char2idx)
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.max_len = max_len

    def __len__(self): return len(self.plain_lines)

    def __getitem__(self, idx):
        # 1. Parse the cipher text into integer tokens
        c_raw = parse_cipher(self.cipher_lines[idx])
        
        # 2. Map plain text characters to indices
        p_text = self.plain_lines[idx].strip()
        p_raw = [self.char2idx['<SOS>']] + [self.char2idx[ch] for ch in p_text] + [self.char2idx['<EOS>']]
        
        # 3. Apply MAX_LEN truncation
        c_tokens = torch.tensor(c_raw[:self.max_len])
        p_tokens = torch.tensor(p_raw[:self.max_len])
    
        return c_tokens, p_tokens

def collate_fn(batch):
    c_side, p_side = zip(*batch)
    c_pad = pad_sequence(c_side, batch_first=True, padding_value=0)
    p_pad = pad_sequence(p_side, batch_first=True, padding_value=0)
    return c_pad, p_pad

# --- Models ---
class ManualRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.xh = nn.Linear(input_size, hidden_size)
        self.hh = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h):
        return torch.tanh(self.xh(x) + self.hh(h))

class ManualLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.w_ih = nn.Linear(input_size, 4 * hidden_size)
        self.w_hh = nn.Linear(hidden_size, 4 * hidden_size)
        
        # Initialize the forget gate bias to 1.0
        with torch.no_grad():
            # (i, f, g, o), 'f' is the second chunk
            start, end = hidden_size, 2 * hidden_size
            self.w_ih.bias[start:end].fill_(1.0)
            self.w_hh.bias[start:end].fill_(1.0)

    def forward(self, x, states):
        h, c = states
        # Compute all gate inputs in two parallel linear ops
        gates = self.w_ih(x) + self.w_hh(h)
        
        i, f, g, o = gates.chunk(4, dim=1)
        
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.w = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs, mask=None):
        src_len = enc_outputs.shape[1]
        dec_hidden_rep = dec_hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        energy = torch.tanh(self.w(torch.cat((dec_hidden_rep, enc_outputs), dim=2)))
        attention = self.v(energy).squeeze(2) # (batch_size, src_len)
        
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
            
        return torch.softmax(attention, dim=1)

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, hid_dim, mode='LSTM'):
        super().__init__()
        self.mode = mode
        self.hid_dim = hid_dim
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.out_embedding = nn.Embedding(output_dim, emb_dim)
        self.attention = Attention(hid_dim)
        
        if mode == 'RNN':
            self.enc_cell = ManualRNNCell(emb_dim, hid_dim)
            self.dec_cell = ManualRNNCell(emb_dim + hid_dim, hid_dim)
        else:
            self.enc_cell = ManualLSTMCell(emb_dim, hid_dim)
            self.dec_cell = ManualLSTMCell(emb_dim + hid_dim, hid_dim)
            
        self.fc_out = nn.Linear(hid_dim * 2 + emb_dim, output_dim)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, src_len = src.shape
        trg_len = trg.shape[1]
        outputs = torch.zeros(trg_len, batch_size, self.fc_out.out_features).to(src.device)
        
        # --- ENCODER ---
        embedded = self.embedding(src)
        h = torch.zeros(batch_size, self.hid_dim).to(src.device)
        c = torch.zeros(batch_size, self.hid_dim).to(src.device) if self.mode == 'LSTM' else None
        
        enc_outputs = torch.zeros(batch_size, src_len, self.hid_dim).to(src.device)
        
        for t in range(src_len):
            if self.mode == 'RNN': h = self.enc_cell(embedded[:, t, :], h)
            else: h, c = self.enc_cell(embedded[:, t, :], (h, c))
            enc_outputs[:, t, :] = h
            
        # --- DECODER ---
        src_mask = (src != 0) 
        input_token = trg[:, 0]
        
        for t in range(1, trg_len):
            emb_dec = self.out_embedding(input_token)
            
            a = self.attention(h, enc_outputs, mask=src_mask) 
            a = a.unsqueeze(1)
            
            context = torch.bmm(a, enc_outputs).squeeze(1) # (batch, hid_dim)
            
            dec_input = torch.cat((emb_dec, context), dim=1)
            
            if self.mode == 'RNN': h = self.dec_cell(dec_input, h)
            else: h, c = self.dec_cell(dec_input, (h, c))
            
            output = self.fc_out(torch.cat((h, context, emb_dec), dim=1))
            outputs[t] = output
            
            top1 = output.argmax(1)
            input_token = trg[:, t] if torch.rand(1) < teacher_forcing_ratio else top1
            
        return outputs.permute(1, 0, 2)

# --- Metrics ---
def calculate_metrics(preds, targets, idx2char):
    char_acc, word_acc, total_lev = 0, 0, 0
    batch_size = preds.size(0)
    
    for i in range(batch_size):
        # 1. Convert to lists of strings
        p_raw = [idx2char[idx.item()] for idx in preds[i]]
        t_raw = [idx2char[idx.item()] for idx in targets[i]]
        
        # 2. Cut off everything after <EOS>
        def truncate_at_eos(char_list):
            if '<EOS>' in char_list:
                return char_list[:char_list.index('<EOS>')]
            return char_list
            
        p_clean = truncate_at_eos(p_raw)
        t_clean = truncate_at_eos(t_raw)
        
        # 3. Join into strings, ignoring PAD and SOS
        p_str = "".join([c for c in p_clean if c not in ['<PAD>', '<SOS>']])
        t_str = "".join([c for c in t_clean if c not in ['<PAD>', '<SOS>']])
        
        # Char Accuracy
        char_acc += sum(1 for a, b in zip(p_str, t_str) if a == b) / max(len(t_str), 1)
        
        # Word Accuracy
        p_words = p_str.split()
        t_words = t_str.split()
        correct_words = sum(1 for pw, tw in zip(p_words, t_words) if pw == tw)
        word_acc += correct_words / max(len(t_words), 1)
        
        # Levenshtein Distance
        total_lev += Levenshtein.distance(p_str, t_str)
        
    return char_acc/batch_size, word_acc/batch_size, total_lev/batch_size

def log_samples(model, loader, idx2char, device, num_samples=5):
    model.eval()
    samples = []

    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0)
            preds = output.argmax(2)

            for i in range(src.size(0)):
                if len(samples) >= num_samples:
                    wandb.log({"samples": samples})
                    return

                p_str = "".join([idx2char[idx.item()] for idx in preds[i] if idx.item() > 2])
                t_str = "".join([idx2char[idx.item()] for idx in trg[i] if idx.item() > 2])

                samples.append({
                    "target": t_str,
                    "prediction": p_str
                })

# --- Training / Evaluation ---
def train_epoch(model, loader, optimizer, criterion, device, teacher_forcing_ratio=0.5):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(loader, desc="Training", leave=False)
    
    for src, trg in progress_bar:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        
        # The forward method takes teacher_forcing_ratio
        output = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)
        output_dim = output.shape[-1]
        
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
        
        progress_bar.set_postfix({'batch_loss': f"{loss.item():.4f}"})
        
    return epoch_loss / len(loader)

def evaluate(model, loader, criterion, device, idx2char):
    model.eval()
    epoch_loss = 0
    all_char_acc, all_word_acc, all_lev = 0, 0, 0
    
    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0) 
            
            output_dim = output.shape[-1]
            loss = criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))
            epoch_loss += loss.item()
            
            preds = output.argmax(2)
            c_acc, w_acc, lev = calculate_metrics(preds, trg, idx2char)
            all_char_acc += c_acc
            all_word_acc += w_acc
            all_lev += lev
            
    n = len(loader)
    return epoch_loss/n, all_char_acc/n, all_word_acc/n, all_lev/n

def print_samples(model, loader, idx2char, device, num_samples=5):
    model.eval()
    samples_shown = 0
    
    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0)
            preds = output.argmax(2)
            
            for i in range(src.size(0)):
                if samples_shown >= num_samples: return
                
                p_str = "".join([idx2char[idx.item()] for idx in preds[i] if idx.item() > 2])
                t_str = "".join([idx2char[idx.item()] for idx in trg[i] if idx.item() > 2])
                
                print(f"Sample {samples_shown + 1}:")
                print(f"  Target: {t_str}")
                print(f"  Pred:   {p_str}")
                print("-" * 20)
                samples_shown += 1