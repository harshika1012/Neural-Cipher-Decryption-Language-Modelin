# import os
# import torch
# import pickle
# import pandas as pd
# import evaluate
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from huggingface_hub import hf_hub_download
# import yaml
# import sys
# from src.task1.common import Seq2Seq
# from src.task2.common import SSM_NWP, BiLSTM_MLM
# from .utils import (
#     CharVocabProcessor, 
#     WordVocab,
#     calc_accuracy_metrics, 
#     get_base_decryption, 
#     correct_with_ssm_nwp, 
#     correct_with_bilstm_mlm
# )
# from src.utils.logging import Logger

# # Fix for AttributeError: module '__main__' has no attribute 'WordVocab'
# # This happens because the object was pickled in a script where WordVocab was defined at the top level.
# sys.modules['__main__'].WordVocab = WordVocab

# def main(config_path: str, mode: str = "evaluate") -> None:
#     # Load config
#     with open(config_path, 'r') as f:
#         config = yaml.safe_load(f)
    
#     # output_dir from config or default
#     output_dir = os.path.dirname(config['output']['metrics_csv']) or "outputs/task3"
#     os.makedirs(output_dir, exist_ok=True)
    
#     # --- REDIRECT OUTPUT TO FILE ---
#     log_file = os.path.join(output_dir, f"output_{mode}.txt")
#     sys.stdout = Logger(log_file)
#     print(f"--- STARTING RUN ({mode}) ---")
    
#     device = torch.device('cuda' if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
#     print(f"Using device: {device}")
    
#     # 1. Download Assets
#     print("Downloading assets from Hugging Face...")
#     REPO_P1 = config['repos']['p1']
#     REPO_P2 = config['repos']['p2']
    
#     char_vocab_path = hf_hub_download(repo_id=REPO_P1, filename=config['files']['char_vocab'])
#     with open(char_vocab_path, 'rb') as f: 
#         char_vocab = CharVocabProcessor(pickle.load(f))

#     word_vocab_path = hf_hub_download(repo_id=REPO_P2, filename=config['files']['word_vocab'])
#     with open(word_vocab_path, 'rb') as f: 
#         word_vocab = pickle.load(f)
    
#     english_dict = set(word_vocab.word2idx.keys())
    
#     # 2. Initialize Models
#     print("Initializing Models...")
#     lstm_path = hf_hub_download(repo_id=REPO_P1, filename=config['files']['lstm_model'])
#     lstm_decrypt = Seq2Seq(input_dim=200, output_dim=len(char_vocab.char2idx), emb_dim=128, hid_dim=256, mode='LSTM').to(device)
#     lstm_decrypt.load_state_dict(torch.load(lstm_path, map_location=device))
#     lstm_decrypt.eval()

#     ssm_path = hf_hub_download(repo_id=REPO_P2, filename=config['files']['ssm_model'])
#     ssm_nwp = SSM_NWP(word_vocab.vocab_size).to(device)
#     ssm_nwp.load_state_dict(torch.load(ssm_path, map_location=device))
#     ssm_nwp.eval()

#     bilstm_path = hf_hub_download(repo_id=REPO_P2, filename=config['files']['bilstm_model'])
#     bilstm_mlm = BiLSTM_MLM(word_vocab.vocab_size).to(device)
#     bilstm_mlm.load_state_dict(torch.load(bilstm_path, map_location=device))
#     bilstm_mlm.eval()

#     # 3. Metrics Setup
#     bleu_metric = evaluate.load("sacrebleu")
#     rouge_metric = evaluate.load("rouge")

#     # 4. Load Ground Truth
#     with open(config['data']['plain_file'], 'r') as f:
#         ground_truths = [line.strip() for line in f.readlines()][:config['data']['num_samples']]

#     results_data = []
#     error_analysis = []

#     # 5. Execute Experiments
#     for noise in config['data']['noise_levels']:
#         cipher_file = f"{config['data']['noisy_cipher_prefix']}{noise}.txt"
#         print(f"\nEvaluating Noise Level: {cipher_file}")
        
#         if not os.path.exists(cipher_file):
#             print(f"Skipping {cipher_file} (Not Found)")
#             continue
            
#         with open(cipher_file, 'r') as f:
#             ciphers = [line.strip() for line in f.readlines()][:config['data']['num_samples']]

#         metrics = {"Base": [], "SSM": [], "BiLSTM": []}
#         preds = {"Base": [], "SSM": [], "BiLSTM": []}

#         for cipher, truth in tqdm(zip(ciphers, ground_truths), total=len(ciphers)):
#             # Decryption Alone
#             base_text = get_base_decryption(cipher, lstm_decrypt, char_vocab, device)
#             preds["Base"].append(base_text)
#             metrics["Base"].append(calc_accuracy_metrics(base_text, truth))
            
#             # SSM Correction
#             ssm_text = correct_with_ssm_nwp(base_text, ssm_nwp, word_vocab, english_dict, device)
#             preds["SSM"].append(ssm_text)
#             metrics["SSM"].append(calc_accuracy_metrics(ssm_text, truth))
            
#             # Bi-LSTM Correction
#             bilstm_text = correct_with_bilstm_mlm(base_text, bilstm_mlm, word_vocab, english_dict, device)
#             preds["BiLSTM"].append(bilstm_text)
#             metrics["BiLSTM"].append(calc_accuracy_metrics(bilstm_text, truth))
            
#             error_analysis.append({
#                 "Noise_Level": noise,
#                 "Target": truth,
#                 "Decrypted_Alone": base_text,
#                 "SSM_Corrected": ssm_text,
#                 "BiLSTM_Corrected": bilstm_text
#             })

#         for app_key, app_name in [("Base", "Decryption Alone"), ("SSM", "Decryption + SSM"), ("BiLSTM", "Decryption + BiLSTM")]:
#             avg_char_acc = sum(m[0] for m in metrics[app_key]) / len(metrics[app_key])
#             avg_word_acc = sum(m[1] for m in metrics[app_key]) / len(metrics[app_key])
#             avg_lev = sum(m[2] for m in metrics[app_key]) / len(metrics[app_key])
            
#             bleu = bleu_metric.compute(predictions=preds[app_key], references=[[t] for t in ground_truths])['score']
#             rouge = rouge_metric.compute(predictions=preds[app_key], references=ground_truths)['rougeL']

#             results_data.append({
#                 "Noise_Level": noise,
#                 "Approach": app_name,
#                 "Char_Acc": avg_char_acc,
#                 "Word_Acc": avg_word_acc,
#                 "Levenshtein": avg_lev,
#                 "BLEU": bleu,
#                 "ROUGE-L": rouge
#             })

#     # 6. Results & Visualization
#     metrics_df = pd.DataFrame(results_data)
#     error_df = pd.DataFrame(error_analysis)

#     error_df.to_csv(config['output']['error_csv'], index=False)
#     metrics_df.to_csv(config['output']['metrics_csv'], index=False)

#     print("\n" + "="*50)
#     print("FINAL DOMINION PERFORMANCE PIPELINE RESULTS")
#     print("="*50)
#     print(metrics_df.to_string(index=False))

#     # Plotting
#     plt.figure(figsize=(18, 5))
#     plt.subplot(1, 3, 1)
#     for approach in metrics_df['Approach'].unique():
#         subset = metrics_df[metrics_df['Approach'] == approach]
#         plt.plot(subset['Noise_Level'], subset['Levenshtein'], marker='o', label=approach)
#     plt.title("Levenshtein Distance (Lower is Better)")
#     plt.xlabel("Noise Level")
#     plt.legend()
#     plt.grid(True)

#     plt.subplot(1, 3, 2)
#     for approach in metrics_df['Approach'].unique():
#         subset = metrics_df[metrics_df['Approach'] == approach]
#         plt.plot(subset['Noise_Level'], subset['BLEU'], marker='x', label=approach)
#     plt.title("BLEU Score (Higher is Better)")
#     plt.xlabel("Noise Level")
#     plt.legend()
#     plt.grid(True)

#     plt.subplot(1, 3, 3)
#     for approach in metrics_df['Approach'].unique():
#         subset = metrics_df[metrics_df['Approach'] == approach]
#         plt.plot(subset['Noise_Level'], subset['Word_Acc'], marker='s', label=approach)
#     plt.title("Word Accuracy (Higher is Better)")
#     plt.xlabel("Noise Level")
#     plt.legend()
#     plt.grid(True)

#     plt.tight_layout()
#     plt.savefig(config['output']['charts'])
#     print(f"Charts saved to {config['output']['charts']}")
#     plt.show()

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, required=True)
#     parser.add_argument("--mode", type=str, default="evaluate")
#     args = parser.parse_args()
#     main(args.config, args.mode)

import os
import torch
import pickle
import pandas as pd
import evaluate
from tqdm import tqdm
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
import yaml
import sys
import sys
from src.task2.common import SSM_NWP, BiLSTM_MLM
from .utils import (
    CharVocabProcessor, 
    WordVocab,
    Seq2Seq,
    calc_accuracy_metrics, 
    get_base_decryption, 
    correct_with_ssm_nwp, 
    correct_with_bilstm_mlm
)
from src.utils.logging import Logger

# Fix for AttributeError: module '__main__' has no attribute 'WordVocab'
# This happens because the object was pickled in a script where WordVocab was defined at the top level.
sys.modules['__main__'].WordVocab = WordVocab

def main(config_path: str, mode: str = "evaluate") -> None:
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # output_dir from config or default
    output_dir = os.path.dirname(config['output']['metrics_csv']) or "outputs/task3"
    os.makedirs(output_dir, exist_ok=True)
    
    # --- REDIRECT OUTPUT TO FILE ---
    log_file = os.path.join(output_dir, f"output_{mode}.txt")
    sys.stdout = Logger(log_file)
    print(f"--- STARTING RUN ({mode}) ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Download Assets
    print("Downloading assets from Hugging Face...")
    REPO_P1 = config['repos']['p1']
    REPO_P2 = config['repos']['p2']
    
    char_vocab_path = hf_hub_download(repo_id=REPO_P1, filename=config['files']['char_vocab'])
    with open(char_vocab_path, 'rb') as f: 
        char_vocab = CharVocabProcessor(pickle.load(f))

    word_vocab_path = hf_hub_download(repo_id=REPO_P2, filename=config['files']['word_vocab'])
    with open(word_vocab_path, 'rb') as f: 
        word_vocab = pickle.load(f)
    
    english_dict = set(word_vocab.word2idx.keys())
    
    # 2. Initialize Models
    print("Initializing Models...")
    lstm_path = hf_hub_download(repo_id=REPO_P1, filename=config['files']['lstm_model'])
    lstm_decrypt = Seq2Seq(input_dim=200, output_dim=len(char_vocab.char2idx), emb_dim=128, hid_dim=256, mode='LSTM').to(device)
    lstm_decrypt.load_state_dict(torch.load(lstm_path, map_location=device))
    lstm_decrypt.eval()

    ssm_path = hf_hub_download(repo_id=REPO_P2, filename=config['files']['ssm_model'])
    ssm_nwp = SSM_NWP(word_vocab.vocab_size).to(device)
    ssm_nwp.load_state_dict(torch.load(ssm_path, map_location=device))
    ssm_nwp.eval()

    bilstm_path = hf_hub_download(repo_id=REPO_P2, filename=config['files']['bilstm_model'])
    bilstm_mlm = BiLSTM_MLM(word_vocab.vocab_size).to(device)
    bilstm_mlm.load_state_dict(torch.load(bilstm_path, map_location=device))
    bilstm_mlm.eval()

    # 3. Metrics Setup
    bleu_metric = evaluate.load("sacrebleu")
    rouge_metric = evaluate.load("rouge")

    # 4. Load Ground Truth
    with open(config['data']['plain_file'], 'r') as f:
        ground_truths = [line.strip() for line in f.readlines()][:config['data']['num_samples']]

    results_data = []
    error_analysis = []

    # 5. Execute Experiments
    for noise in config['data']['noise_levels']:
        cipher_file = f"{config['data']['noisy_cipher_prefix']}{noise}.txt"
        print(f"\nEvaluating Noise Level: {cipher_file}")
        
        if not os.path.exists(cipher_file):
            print(f"Skipping {cipher_file} (Not Found)")
            continue
            
        with open(cipher_file, 'r') as f:
            ciphers = [line.strip() for line in f.readlines()][:config['data']['num_samples']]

        metrics = {"Base": [], "SSM": [], "BiLSTM": []}
        preds = {"Base": [], "SSM": [], "BiLSTM": []}

        for cipher, truth in tqdm(zip(ciphers, ground_truths), total=len(ciphers)):
            # Decryption Alone
            base_text = get_base_decryption(cipher, lstm_decrypt, char_vocab, device)
            preds["Base"].append(base_text)
            metrics["Base"].append(calc_accuracy_metrics(base_text, truth))
            
            # SSM Correction
            ssm_text = correct_with_ssm_nwp(base_text, ssm_nwp, word_vocab, english_dict, device)
            preds["SSM"].append(ssm_text)
            metrics["SSM"].append(calc_accuracy_metrics(ssm_text, truth))
            
            # Bi-LSTM Correction
            bilstm_text = correct_with_bilstm_mlm(base_text, bilstm_mlm, word_vocab, english_dict, device)
            preds["BiLSTM"].append(bilstm_text)
            metrics["BiLSTM"].append(calc_accuracy_metrics(bilstm_text, truth))
            
            error_analysis.append({
                "Noise_Level": noise,
                "Target": truth,
                "Decrypted_Alone": base_text,
                "SSM_Corrected": ssm_text,
                "BiLSTM_Corrected": bilstm_text
            })

        for app_key, app_name in [("Base", "Decryption Alone"), ("SSM", "Decryption + SSM"), ("BiLSTM", "Decryption + BiLSTM")]:
            avg_char_acc = sum(m[0] for m in metrics[app_key]) / len(metrics[app_key])
            avg_word_acc = sum(m[1] for m in metrics[app_key]) / len(metrics[app_key])
            avg_lev = sum(m[2] for m in metrics[app_key]) / len(metrics[app_key])
            
            bleu = bleu_metric.compute(predictions=preds[app_key], references=[[t] for t in ground_truths])['score']
            rouge = rouge_metric.compute(predictions=preds[app_key], references=ground_truths)['rougeL']

            results_data.append({
                "Noise_Level": noise,
                "Approach": app_name,
                "Char_Acc": avg_char_acc,
                "Word_Acc": avg_word_acc,
                "Levenshtein": avg_lev,
                "BLEU": bleu,
                "ROUGE-L": rouge
            })

    # 6. Results & Visualization
    metrics_df = pd.DataFrame(results_data)
    error_df = pd.DataFrame(error_analysis)

    error_df.to_csv(config['output']['error_csv'], index=False)
    metrics_df.to_csv(config['output']['metrics_csv'], index=False)

    print("\n" + "="*50)
    print("FINAL DOMINION PERFORMANCE PIPELINE RESULTS")
    print("="*50)
    print(metrics_df.to_string(index=False))

    # Plotting
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    for approach in metrics_df['Approach'].unique():
        subset = metrics_df[metrics_df['Approach'] == approach]
        plt.plot(subset['Noise_Level'], subset['Levenshtein'], marker='o', label=approach)
    plt.title("Levenshtein Distance (Lower is Better)")
    plt.xlabel("Noise Level")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    for approach in metrics_df['Approach'].unique():
        subset = metrics_df[metrics_df['Approach'] == approach]
        plt.plot(subset['Noise_Level'], subset['BLEU'], marker='x', label=approach)
    plt.title("BLEU Score (Higher is Better)")
    plt.xlabel("Noise Level")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    for approach in metrics_df['Approach'].unique():
        subset = metrics_df[metrics_df['Approach'] == approach]
        plt.plot(subset['Noise_Level'], subset['Word_Acc'], marker='s', label=approach)
    plt.title("Word Accuracy (Higher is Better)")
    plt.xlabel("Noise Level")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(config['output']['charts'])
    print(f"Charts saved to {config['output']['charts']}")
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, default="evaluate")
    args = parser.parse_args()
    main(args.config, args.mode)