# import torch
# import Levenshtein
# import evaluate
# from huggingface_hub import hf_hub_download
# import pickle
# from collections import Counter

# # --- Metrics ---
# def calc_accuracy_metrics(pred_str, target_str):
#     """Calculates Character Accuracy, Word Accuracy, and Levenshtein Distance."""
#     lev_dist = Levenshtein.distance(pred_str, target_str)
#     char_acc = sum(1 for a, b in zip(pred_str, target_str) if a == b) / max(len(target_str), 1)
#     p_words, t_words = pred_str.split(), target_str.split()
#     word_acc = sum(1 for pw, tw in zip(p_words, t_words) if pw == tw) / max(len(t_words), 1)
#     return char_acc, word_acc, lev_dist

# # --- Vocab Helpers ---
# class WordVocab:
#     def __init__(self, plain_file=None, max_vocab_size=10000):
#         self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3, '<MASK>': 4}
#         self.idx2word = {}
#         self.vocab_size = 5
#         if plain_file:
#             word_counts = Counter()
#             with open(plain_file, 'r') as f:
#                 for line in f:
#                     word_counts.update(line.strip().split())
#             most_common = word_counts.most_common(max_vocab_size - len(self.word2idx))
#             for word, _ in most_common:
#                 self.word2idx[word] = len(self.word2idx)
#             self.idx2word = {v: k for k, v in self.word2idx.items()}
#             self.vocab_size = len(self.word2idx)

#     def encode(self, text):
#         return [self.word2idx.get(w, self.word2idx['<UNK>']) for w in text.split()]

# class CharVocabProcessor:
#     def __init__(self, vocab_dict):
#         self.char2idx = vocab_dict['char2idx']
#         self.idx2char = vocab_dict['idx2char']
        
#     def encode_cipher(self, line):
#         tokens = []
#         i = 0
#         while i < len(line.strip()):
#             if line[i] == '9':
#                 tokens.append(99)
#                 i += 1
#             else:
#                 tokens.append(int(line[i:i+2]))
#                 i += 2
#         raw_tokens = [t + 2 for t in tokens]
#         return torch.tensor(raw_tokens[:128]).unsqueeze(0)

#     def decode_plain(self, output_tensor):
#         p_tokens = []
#         for idx in output_tensor:
#             char = self.idx2char.get(idx.item(), '')
#             if char == '<EOS>': break
#             if char not in ['<PAD>', '<SOS>']: p_tokens.append(char)
#         return "".join(p_tokens)

# # --- Inference & Correction Pipeline ---
# def get_base_decryption(cipher_text, decrypt_model, char_vocab, device):
#     inputs = char_vocab.encode_cipher(cipher_text).to(device)
#     dummy_trg = torch.zeros(1, 128, dtype=torch.long).to(device)
#     dummy_trg[0, 0] = char_vocab.char2idx.get('<SOS>', 1)
#     with torch.no_grad():
#         outputs = decrypt_model(inputs, dummy_trg, teacher_forcing_ratio=0) 
#     preds = outputs.argmax(2)[0] 
#     return char_vocab.decode_plain(preds)

# def correct_with_ssm_nwpp(noisy_text, ssm_model, word_vocab, eng_dict, device):
#     words = noisy_text.split()
#     corrected_words = []
#     context_idxs = [word_vocab.word2idx.get('<SOS>', 2)]
#     for w in words:
#         if w not in eng_dict and len(w) > 0 and len(context_idxs) > 0:
#             input_tensor = torch.tensor([context_idxs]).to(device)
#             with torch.no_grad():
#                 logits = ssm_model(input_tensor) 
#                 next_word_idx = logits[0, -1, :].argmax().item()
#                 predicted_word = word_vocab.idx2word.get(next_word_idx, w)
#                 if predicted_word not in ['<PAD>', '<UNK>', '<SOS>', '<EOS>', '<MASK>']:
#                     corrected_words.append(predicted_word)
#                     context_idxs.append(next_word_idx)
#                 else:
#                     corrected_words.append(w)
#                     context_idxs.append(word_vocab.word2idx.get(w, 1))
#         else:
#             corrected_words.append(w)
#             context_idxs.append(word_vocab.word2idx.get(w, 1))
#     return " ".join(corrected_words)

# def correct_with_bilstm_mlmp(noisy_text, bilstm_model, word_vocab, eng_dict, device):
#     words = noisy_text.split()
#     input_idxs = []
#     mask_positions = []
#     for i, w in enumerate(words):
#         if w not in eng_dict and len(w) > 0:
#             input_idxs.append(word_vocab.word2idx.get('<MASK>', 4))
#             mask_positions.append(i)
#         else:
#             input_idxs.append(word_vocab.word2idx.get(w, 1)) 
#     if not mask_positions: return noisy_text
#     input_tensor = torch.tensor([input_idxs]).to(device)
#     with torch.no_grad():
#         logits = bilstm_model(input_tensor)
#     corrected_words = list(words)
#     for pos in mask_positions:
#         pred_idx = logits[0, pos, :].argmax().item()
#         pred_word = word_vocab.idx2word.get(pred_idx, words[pos])
#         if pred_word not in ['<PAD>', '<UNK>', '<SOS>', '<EOS>', '<MASK>']:
#             corrected_words[pos] = pred_word
#     return " ".join(corrected_words)

# def correct_with_ssm_nwp_ppp(noisy_text, ssm_model, word_vocab, eng_dict, device):
#     words = noisy_text.split()
#     corrected_words = []
#     context_idxs = [word_vocab.word2idx.get('<SOS>', 2)]
#     for w in words:
#         clean_w = ''.join(e for e in w if e.isalnum()).lower()
#         if w not in eng_dict and clean_w not in eng_dict and len(clean_w) > 0:
#             input_tensor = torch.tensor([context_idxs]).to(device)
#             with torch.no_grad():
#                 logits = ssm_model(input_tensor) 
#                 top_k_probs, top_k_idx = torch.topk(logits[0, -1, :], k=50)
#                 best_word = w
#                 best_score = float('inf')
#                 for prob, idx in zip(top_k_probs, top_k_idx):
#                     candidate = word_vocab.idx2word.get(idx.item(), "")
#                     if candidate in ['<PAD>', '<UNK>', '<SOS>', '<EOS>', '<MASK>']:
#                         continue
#                     edit_dist = Levenshtein.distance(candidate, w)
#                     score = edit_dist - (0.05 * prob.item()) 
#                     if score < best_score:
#                         best_score = score
#                         best_word = candidate
#                 corrected_words.append(best_word)
#                 context_idxs.append(word_vocab.word2idx.get(best_word, 1))
#         else:
#             corrected_words.append(w)
#             context_idxs.append(word_vocab.word2idx.get(w, 1))
#     return " ".join(corrected_words)

# def correct_with_bilstm_mlmppp(noisy_text, bilstm_model, word_vocab, eng_dict, device):
#     words = noisy_text.split()
#     input_idxs = []
#     mask_positions = []
#     for i, w in enumerate(words):
#         clean_w = ''.join(e for e in w if e.isalnum()).lower()
#         if w not in eng_dict and clean_w not in eng_dict and len(clean_w) > 0:
#             input_idxs.append(word_vocab.word2idx.get('<MASK>', 4))
#             mask_positions.append(i)
#         else:
#             input_idxs.append(word_vocab.word2idx.get(w, word_vocab.word2idx.get('<UNK>', 1))) 
#     if not mask_positions: return noisy_text
#     input_tensor = torch.tensor([input_idxs]).to(device)
#     with torch.no_grad():
#         logits = bilstm_model(input_tensor)
#     corrected_words = list(words)
#     for pos in mask_positions:
#         original_noisy_word = words[pos]
#         top_k_probs, top_k_idx = torch.topk(logits[0, pos, :], k=50)
#         best_word = original_noisy_word
#         best_score = float('inf')
#         for prob, idx in zip(top_k_probs, top_k_idx):
#             candidate = word_vocab.idx2word.get(idx.item(), "")
#             if candidate in ['<PAD>', '<UNK>', '<SOS>', '<EOS>', '<MASK>']:
#                 continue
#             edit_dist = Levenshtein.distance(candidate, original_noisy_word)
#             score = edit_dist - (0.05 * prob.item())
#             if score < best_score:
#                 best_score = score
#                 best_word = candidate
#         corrected_words[pos] = best_word
#     return " ".join(corrected_words)

# def correct_with_ssm_nwp(noisy_text, ssm_model, word_vocab, eng_dict, device):
#     words = noisy_text.split()
#     corrected_words = []
#     context_idxs = [word_vocab.word2idx.get('<SOS>', 2)]
#     for w in words:
#         clean_w = ''.join(e for e in w if e.isalnum()).lower()
#         if w not in eng_dict and clean_w not in eng_dict and len(clean_w) > 0:
#             input_tensor = torch.tensor([context_idxs]).to(device)
#             with torch.no_grad():
#                 logits = ssm_model(input_tensor) 
#                 top_k_probs, top_k_idx = torch.topk(logits[0, -1, :], k=500)
#                 best_word = w
#                 best_score = float('inf')
#                 for prob, idx in zip(top_k_probs, top_k_idx):
#                     candidate = word_vocab.idx2word.get(idx.item(), "")
#                     if candidate in ['<PAD>', '<UNK>', '<SOS>', '<EOS>', '<MASK>']:
#                         continue
#                     if len(candidate) != len(w): continue
#                     edit_dist = Levenshtein.distance(candidate, w)
#                     score = edit_dist - (0.01 * prob.item()) 
#                     if score < best_score:
#                         best_score = score
#                         best_word = candidate
#                 corrected_words.append(best_word)
#                 context_idxs.append(word_vocab.word2idx.get(best_word, 1))
#         else:
#             corrected_words.append(w)
#             context_idxs.append(word_vocab.word2idx.get(w, 1))
#     return " ".join(corrected_words)

# def correct_with_bilstm_mlm(noisy_text, bilstm_model, word_vocab, eng_dict, device):
#     words = noisy_text.split()
#     input_idxs = []
#     mask_positions = []
#     for i, w in enumerate(words):
#         clean_w = ''.join(e for e in w if e.isalnum()).lower()
#         if w not in eng_dict and clean_w not in eng_dict and len(clean_w) > 0:
#             input_idxs.append(word_vocab.word2idx.get('<MASK>', 4))
#             mask_positions.append(i)
#         else:
#             input_idxs.append(word_vocab.word2idx.get(w, word_vocab.word2idx.get('<UNK>', 1))) 
#     if not mask_positions: return noisy_text
#     input_tensor = torch.tensor([input_idxs]).to(device)
#     with torch.no_grad():
#         logits = bilstm_model(input_tensor)
#     corrected_words = list(words)
#     for pos in mask_positions:
#         original_noisy_word = words[pos]
#         top_k_probs, top_k_idx = torch.topk(logits[0, pos, :], k=500)
#         best_word = original_noisy_word
#         best_score = float('inf')
#         for prob, idx in zip(top_k_probs, top_k_idx):
#             candidate = word_vocab.idx2word.get(idx.item(), "")
#             if candidate in ['<PAD>', '<UNK>', '<SOS>', '<EOS>', '<MASK>']:
#                 continue
#             if len(candidate) != len(original_noisy_word): continue
#             edit_dist = Levenshtein.distance(candidate, original_noisy_word)
#             score = edit_dist - (0.01 * prob.item())
#             if score < best_score:
#                 best_score = score
#                 best_word = candidate
#         corrected_words[pos] = best_word
#     return " ".join(corrected_words)

import torch
import torch.nn as nn
import Levenshtein
import evaluate
from huggingface_hub import hf_hub_download
import pickle
from collections import Counter

# --- Specialized Models for Task 3 (matching uploaded weights) ---
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
        with torch.no_grad():
            start, end = hidden_size, 2 * hidden_size
            self.w_ih.bias[start:end].fill_(1.0)
            self.w_hh.bias[start:end].fill_(1.0)
    def forward(self, x, states):
        h, c = states
        gates = self.w_ih(x) + self.w_hh(h)
        i, f, g, o = gates.chunk(4, dim=1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.W_q = nn.Linear(hid_dim, hid_dim)
        self.W_k = nn.Linear(hid_dim, hid_dim)
        self.W_v = nn.Linear(hid_dim, hid_dim)

    def forward(self, dec_hidden, enc_outputs, mask=None):
        query = self.W_q(dec_hidden).unsqueeze(1)
        keys = self.W_k(enc_outputs)
        energy = torch.bmm(query, keys.permute(0, 2, 1)).squeeze(1)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        return torch.softmax(energy, dim=1)

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
        embedded = self.embedding(src)
        h = torch.zeros(batch_size, self.hid_dim).to(src.device)
        c = torch.zeros(batch_size, self.hid_dim).to(src.device) if self.mode == 'LSTM' else None
        enc_outputs = torch.zeros(batch_size, src_len, self.hid_dim).to(src.device)
        for t in range(src_len):
            if self.mode == 'RNN': h = self.enc_cell(embedded[:, t, :], h)
            else: h, c = self.enc_cell(embedded[:, t, :], (h, c))
            enc_outputs[:, t, :] = h
        src_mask = (src != 0) 
        input_token = trg[:, 0]
        for t in range(1, trg_len):
            emb_dec = self.out_embedding(input_token)
            a = self.attention(h, enc_outputs, mask=src_mask) 
            a = a.unsqueeze(1)
            values = self.attention.W_v(enc_outputs)
            context = torch.bmm(a, values).squeeze(1)
            dec_input = torch.cat((emb_dec, context), dim=1)
            if self.mode == 'RNN': h = self.dec_cell(dec_input, h)
            else: h, c = self.dec_cell(dec_input, (h, c))
            output = self.fc_out(torch.cat((h, context, emb_dec), dim=1))
            outputs[t] = output
            top1 = output.argmax(1)
            input_token = trg[:, t] if torch.rand(1) < teacher_forcing_ratio else top1
        return outputs.permute(1, 0, 2)

# --- Metrics ---
def calc_accuracy_metrics(pred_str, target_str):
    """Calculates Character Accuracy, Word Accuracy, and Levenshtein Distance."""
    lev_dist = Levenshtein.distance(pred_str, target_str)
    char_acc = sum(1 for a, b in zip(pred_str, target_str) if a == b) / max(len(target_str), 1)
    p_words, t_words = pred_str.split(), target_str.split()
    word_acc = sum(1 for pw, tw in zip(p_words, t_words) if pw == tw) / max(len(t_words), 1)
    return char_acc, word_acc, lev_dist

# --- Vocab Helpers ---
class WordVocab:
    def __init__(self, plain_file=None, max_vocab_size=10000):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3, '<MASK>': 4}
        self.idx2word = {}
        self.vocab_size = 5
        if plain_file:
            word_counts = Counter()
            with open(plain_file, 'r') as f:
                for line in f:
                    word_counts.update(line.strip().split())
            most_common = word_counts.most_common(max_vocab_size - len(self.word2idx))
            for word, _ in most_common:
                self.word2idx[word] = len(self.word2idx)
            self.idx2word = {v: k for k, v in self.word2idx.items()}
            self.vocab_size = len(self.word2idx)

    def encode(self, text):
        return [self.word2idx.get(w, self.word2idx['<UNK>']) for w in text.split()]

class CharVocabProcessor:
    def __init__(self, vocab_dict):
        self.char2idx = vocab_dict['char2idx']
        self.idx2char = vocab_dict['idx2char']
        
    def encode_cipher(self, line):
        tokens = []
        i = 0
        while i < len(line.strip()):
            if line[i] == '9':
                tokens.append(99)
                i += 1
            else:
                tokens.append(int(line[i:i+2]))
                i += 2
        raw_tokens = [t + 2 for t in tokens]
        return torch.tensor(raw_tokens[:128]).unsqueeze(0)

    def decode_plain(self, output_tensor):
        p_tokens = []
        for idx in output_tensor:
            char = self.idx2char.get(idx.item(), '')
            if char == '<EOS>': break
            if char not in ['<PAD>', '<SOS>']: p_tokens.append(char)
        return "".join(p_tokens)

# --- Inference & Correction Pipeline ---
def get_base_decryption(cipher_text, decrypt_model, char_vocab, device):
    inputs = char_vocab.encode_cipher(cipher_text).to(device)
    dummy_trg = torch.zeros(1, 128, dtype=torch.long).to(device)
    dummy_trg[0, 0] = char_vocab.char2idx.get('<SOS>', 1)
    with torch.no_grad():
        outputs = decrypt_model(inputs, dummy_trg, teacher_forcing_ratio=0) 
    preds = outputs.argmax(2)[0] 
    return char_vocab.decode_plain(preds)

def correct_with_ssm_nwpp(noisy_text, ssm_model, word_vocab, eng_dict, device):
    words = noisy_text.split()
    corrected_words = []
    context_idxs = [word_vocab.word2idx.get('<SOS>', 2)]
    for w in words:
        if w not in eng_dict and len(w) > 0 and len(context_idxs) > 0:
            input_tensor = torch.tensor([context_idxs]).to(device)
            with torch.no_grad():
                logits = ssm_model(input_tensor) 
                next_word_idx = logits[0, -1, :].argmax().item()
                predicted_word = word_vocab.idx2word.get(next_word_idx, w)
                if predicted_word not in ['<PAD>', '<UNK>', '<SOS>', '<EOS>', '<MASK>']:
                    corrected_words.append(predicted_word)
                    context_idxs.append(next_word_idx)
                else:
                    corrected_words.append(w)
                    context_idxs.append(word_vocab.word2idx.get(w, 1))
        else:
            corrected_words.append(w)
            context_idxs.append(word_vocab.word2idx.get(w, 1))
    return " ".join(corrected_words)

def correct_with_bilstm_mlmp(noisy_text, bilstm_model, word_vocab, eng_dict, device):
    words = noisy_text.split()
    input_idxs = []
    mask_positions = []
    for i, w in enumerate(words):
        if w not in eng_dict and len(w) > 0:
            input_idxs.append(word_vocab.word2idx.get('<MASK>', 4))
            mask_positions.append(i)
        else:
            input_idxs.append(word_vocab.word2idx.get(w, 1)) 
    if not mask_positions: return noisy_text
    input_tensor = torch.tensor([input_idxs]).to(device)
    with torch.no_grad():
        logits = bilstm_model(input_tensor)
    corrected_words = list(words)
    for pos in mask_positions:
        pred_idx = logits[0, pos, :].argmax().item()
        pred_word = word_vocab.idx2word.get(pred_idx, words[pos])
        if pred_word not in ['<PAD>', '<UNK>', '<SOS>', '<EOS>', '<MASK>']:
            corrected_words[pos] = pred_word
    return " ".join(corrected_words)

def correct_with_ssm_nwp_ppp(noisy_text, ssm_model, word_vocab, eng_dict, device):
    words = noisy_text.split()
    corrected_words = []
    context_idxs = [word_vocab.word2idx.get('<SOS>', 2)]
    for w in words:
        clean_w = ''.join(e for e in w if e.isalnum()).lower()
        if w not in eng_dict and clean_w not in eng_dict and len(clean_w) > 0:
            input_tensor = torch.tensor([context_idxs]).to(device)
            with torch.no_grad():
                logits = ssm_model(input_tensor) 
                top_k_probs, top_k_idx = torch.topk(logits[0, -1, :], k=50)
                best_word = w
                best_score = float('inf')
                for prob, idx in zip(top_k_probs, top_k_idx):
                    candidate = word_vocab.idx2word.get(idx.item(), "")
                    if candidate in ['<PAD>', '<UNK>', '<SOS>', '<EOS>', '<MASK>']:
                        continue
                    edit_dist = Levenshtein.distance(candidate, w)
                    score = edit_dist - (0.05 * prob.item()) 
                    if score < best_score:
                        best_score = score
                        best_word = candidate
                corrected_words.append(best_word)
                context_idxs.append(word_vocab.word2idx.get(best_word, 1))
        else:
            corrected_words.append(w)
            context_idxs.append(word_vocab.word2idx.get(w, 1))
    return " ".join(corrected_words)

def correct_with_bilstm_mlmppp(noisy_text, bilstm_model, word_vocab, eng_dict, device):
    words = noisy_text.split()
    input_idxs = []
    mask_positions = []
    for i, w in enumerate(words):
        clean_w = ''.join(e for e in w if e.isalnum()).lower()
        if w not in eng_dict and clean_w not in eng_dict and len(clean_w) > 0:
            input_idxs.append(word_vocab.word2idx.get('<MASK>', 4))
            mask_positions.append(i)
        else:
            input_idxs.append(word_vocab.word2idx.get(w, word_vocab.word2idx.get('<UNK>', 1))) 
    if not mask_positions: return noisy_text
    input_tensor = torch.tensor([input_idxs]).to(device)
    with torch.no_grad():
        logits = bilstm_model(input_tensor)
    corrected_words = list(words)
    for pos in mask_positions:
        original_noisy_word = words[pos]
        top_k_probs, top_k_idx = torch.topk(logits[0, pos, :], k=50)
        best_word = original_noisy_word
        best_score = float('inf')
        for prob, idx in zip(top_k_probs, top_k_idx):
            candidate = word_vocab.idx2word.get(idx.item(), "")
            if candidate in ['<PAD>', '<UNK>', '<SOS>', '<EOS>', '<MASK>']:
                continue
            edit_dist = Levenshtein.distance(candidate, original_noisy_word)
            score = edit_dist - (0.05 * prob.item())
            if score < best_score:
                best_score = score
                best_word = candidate
        corrected_words[pos] = best_word
    return " ".join(corrected_words)

def correct_with_ssm_nwp(noisy_text, ssm_model, word_vocab, eng_dict, device):
    words = noisy_text.split()
    corrected_words = []
    context_idxs = [word_vocab.word2idx.get('<SOS>', 2)]
    for w in words:
        clean_w = ''.join(e for e in w if e.isalnum()).lower()
        if w not in eng_dict and clean_w not in eng_dict and len(clean_w) > 0:
            input_tensor = torch.tensor([context_idxs]).to(device)
            with torch.no_grad():
                logits = ssm_model(input_tensor) 
                top_k_probs, top_k_idx = torch.topk(logits[0, -1, :], k=500)
                best_word = w
                best_score = float('inf')
                for prob, idx in zip(top_k_probs, top_k_idx):
                    candidate = word_vocab.idx2word.get(idx.item(), "")
                    if candidate in ['<PAD>', '<UNK>', '<SOS>', '<EOS>', '<MASK>']:
                        continue
                    if len(candidate) != len(w): continue
                    edit_dist = Levenshtein.distance(candidate, w)
                    score = edit_dist - (0.01 * prob.item()) 
                    if score < best_score:
                        best_score = score
                        best_word = candidate
                corrected_words.append(best_word)
                context_idxs.append(word_vocab.word2idx.get(best_word, 1))
        else:
            corrected_words.append(w)
            context_idxs.append(word_vocab.word2idx.get(w, 1))
    return " ".join(corrected_words)

def correct_with_bilstm_mlm(noisy_text, bilstm_model, word_vocab, eng_dict, device):
    words = noisy_text.split()
    input_idxs = []
    mask_positions = []
    for i, w in enumerate(words):
        clean_w = ''.join(e for e in w if e.isalnum()).lower()
        if w not in eng_dict and clean_w not in eng_dict and len(clean_w) > 0:
            input_idxs.append(word_vocab.word2idx.get('<MASK>', 4))
            mask_positions.append(i)
        else:
            input_idxs.append(word_vocab.word2idx.get(w, word_vocab.word2idx.get('<UNK>', 1))) 
    if not mask_positions: return noisy_text
    input_tensor = torch.tensor([input_idxs]).to(device)
    with torch.no_grad():
        logits = bilstm_model(input_tensor)
    corrected_words = list(words)
    for pos in mask_positions:
        original_noisy_word = words[pos]
        top_k_probs, top_k_idx = torch.topk(logits[0, pos, :], k=500)
        best_word = original_noisy_word
        best_score = float('inf')
        for prob, idx in zip(top_k_probs, top_k_idx):
            candidate = word_vocab.idx2word.get(idx.item(), "")
            if candidate in ['<PAD>', '<UNK>', '<SOS>', '<EOS>', '<MASK>']:
                continue
            if len(candidate) != len(original_noisy_word): continue
            edit_dist = Levenshtein.distance(candidate, original_noisy_word)
            score = edit_dist - (0.01 * prob.item())
            if score < best_score:
                best_score = score
                best_word = candidate
        corrected_words[pos] = best_word
    return " ".join(corrected_words)