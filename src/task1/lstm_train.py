import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import wandb
import os
import pickle
import sys
from huggingface_hub import login, HfApi
from .common import CipherDataset, Seq2Seq, collate_fn, train_epoch, evaluate, print_samples
from src.utils.logging import Logger

def main(config_path: str, mode: str = "evaluate") -> None:
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    torch.manual_seed(config.get('seed', 42))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Dataset
    dataset = CipherDataset(
        config['data']['plain_file'], 
        config['data']['cipher_file'], 
        max_len=config['model'].get('max_len', 128)
    )
    
    
    # Split
    train_size = int(config['data']['train_split'] * len(dataset))
    val_size = int(config['data']['val_split'] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_db, val_db, test_db = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_db, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_db, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_db, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    # Model (LSTM Mode)
    model = Seq2Seq(
        input_dim=config['model']['input_dim'],
        output_dim=len(dataset.char2idx),
        emb_dim=config['model']['emb_dim'],
        hid_dim=config['model']['hid_dim'],
        mode=config['model'].get('mode', 'LSTM')
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    
    output_dir = config.get('output', {}).get('dir', 'outputs/task1_lstm')
    os.makedirs(output_dir, exist_ok=True)
    
    # --- REDIRECT OUTPUT TO FILE ---
    log_file = os.path.join(output_dir, f"output_{mode}.txt")
    sys.stdout = Logger(log_file)
    print(f"--- STARTING RUN ({mode}) ---")
    
    model_path = os.path.join(output_dir, 'model.pt')
    
    if mode in ['train', 'both']:
        if config['logging'].get('use_wandb', False):
            # Using the keys provided in the user request
            wandb.login(key="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")
            login(token="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")
            wandb.init(project=config['logging']['project'], config=config)
            
        print(f"Starting Training ({config['model'].get('mode', 'LSTM')})...")
        for epoch in range(config['training']['epochs']):
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, device, 
                teacher_forcing_ratio=config['training']['teacher_forcing_ratio']
            )
            val_loss, char_acc, word_acc, lev = evaluate(model, val_loader, criterion, device, dataset.idx2char)
            
            print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}')
            print(f'\tChar Acc: {char_acc*100:.2f}% | Levenshtein: {lev:.2f}')
            
            if config['logging'].get('use_wandb', False):
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "char_acc": char_acc,
                    "word_acc": word_acc,
                    "levenshtein": lev
                })
        
        # Save model
        os.makedirs(output_dir, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'char2idx': dataset.char2idx,
        }, model_path)
        print(f"Model saved to {model_path}")

        # --- SAVING AND UPLOADING TO HUGGING FACE ---
        print("\n" + "="*30)
        print("SAVING AND UPLOADING TO HUGGING FACE")
        print("="*30)

        batch_size = config['training']['batch_size']
        lr = config['training']['lr']
        n_epochs = config['training']['epochs']
        mode_name = config['model'].get('mode', 'lstm').lower()

        # 1. Save Model Weights Locally
        hf_model_name = f"{mode_name}_b{batch_size}_lr{lr}_epoch{n_epochs}.pt"
        hf_model_path = os.path.join(output_dir, hf_model_name)
        torch.save(model.state_dict(), hf_model_path)
        print(f"Saved model locally: {hf_model_path}")

        # 2. Save Vocabulary Locally
        vocab_data = {
            'char2idx': dataset.char2idx,
            'idx2char': dataset.idx2char
        }
        vocab_name = f"char_vocab_{mode_name}_b{batch_size}_lr{lr}_epoch{n_epochs}.pkl"
        hf_vocab_path = os.path.join(output_dir, vocab_name)

        with open(hf_vocab_path, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"Saved vocabulary locally: {hf_vocab_path}")

        # 3. Connect to Hugging Face
        repo_id = config['output'].get('repo_id', "Krishiv03/seq2seq-cipher-model")
        login(token="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") 
        api = HfApi()

        # Create repo just in case it doesn't exist yet
        api.create_repo(repo_id=repo_id, exist_ok=True)

        # 4. Upload Model Weights
        print(f"Uploading model {hf_model_name} to Hub...")
        api.upload_file(
            path_or_fileobj=hf_model_path,
            path_in_repo=hf_model_name,
            repo_id=repo_id
        )

        # 5. Upload Vocabulary
        print(f"Uploading vocabulary {vocab_name} to Hub...")
        api.upload_file(
            path_or_fileobj=hf_vocab_path,
            path_in_repo=vocab_name,
            repo_id=repo_id
        )

        print("Upload Complete! Your model and vocabulary are now safely packaged together.")

    if mode in ['evaluate', 'both']:
        print("Running Evaluation...")
        if mode == 'evaluate':
            if not os.path.exists(model_path):
                print(f"Error: Model not found at {model_path}. Please train first.")
                return
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
        test_loss, test_char_acc, test_word_acc, test_lev = evaluate(
            model, test_loader, criterion, device, dataset.idx2char
        )
        
        print("="*30)
        print("FINAL TEST SET PERFORMANCE")
        print("="*30)
        print(f"Test Loss:          {test_loss:.4f}")
        print(f"Character Accuracy: {test_char_acc * 100:.2f}%")
        print(f"Word Accuracy:      {test_word_acc * 100:.2f}%")
        print(f"Avg Levenshtein:    {test_lev:.2f}")
        print("="*30)
        
        print_samples(model, test_loader, dataset.idx2char, device)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, default="evaluate")
    args = parser.parse_args()
    main(args.config, args.mode)
