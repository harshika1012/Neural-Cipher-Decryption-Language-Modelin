import yaml
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import wandb
from huggingface_hub import login, HfApi
import os
import pickle
import sys
from .common import WordVocab, DominionLanguageDataset, SSM_NWP, lm_collate_fn, train_and_evaluate_custom, final_evaluation
from src.utils.logging import Logger

def main(config_path: str, mode: str = "evaluate") -> None:
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    torch.manual_seed(config.get('seed', 42))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Vocab & Dataset
    vocab = WordVocab(config['data']['plain_file'], max_vocab_size=config['model']['vocab_size'])
    dataset = DominionLanguageDataset(
        config['data']['plain_file'], 
        vocab, 
        task='NWP', 
        max_len=config['data']['max_len']
    )
    
    # Split
    train_size = int(config['data']['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_db, val_db = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_db, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=lm_collate_fn)
    val_loader = DataLoader(val_db, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=lm_collate_fn)
    
    # Model
    model = SSM_NWP(
        vocab_size=vocab.vocab_size,
        d_model=config['model']['d_model'],
        d_state=config['model']['d_state'],
        dropout=config['model']['dropout']
    ).to(device)
    
    output_dir = config.get('output', {}).get('dir', 'outputs/task2_ssm')
    os.makedirs(output_dir, exist_ok=True)
    
    # --- REDIRECT OUTPUT TO FILE ---
    log_file = os.path.join(output_dir, f"output_{mode}.txt")
    sys.stdout = Logger(log_file)
    print(f"--- STARTING RUN ({mode}) ---")
    
    model_path = os.path.join(output_dir, "ssm_dominion_nwp.pth")
    vocab_path = os.path.join(output_dir, "dominion_vocab.pkl")
    
    if mode in ['train', 'both']:
        if config['logging'].get('use_wandb', False):
            wandb.login(key="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")
            login(token="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")
            
        print("Starting Training (SSM NWP)...")
        train_and_evaluate_custom(
            model, train_loader, val_loader, 
            epochs=config['training']['epochs'], 
            lr=config['training']['lr'],
            project_name=config['logging']['project']
        )
        
        # Save model and vocab
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
        print(f"Model saved to {model_path}")

        # Upload to Hugging Face if config provided
        if config['output'].get('repo_id'):
            api = HfApi()
            repo_id = config['output']['repo_id']
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=f"models/ssm_dominion_nwp.pth",
                repo_id=repo_id,
                repo_type="model"
            )
            api.upload_file(
                path_or_fileobj=vocab_path,
                path_in_repo=f"models/dominion_vocab.pkl",
                repo_id=repo_id,
                repo_type="model"
            )
            print("Upload Complete!")

    if mode in ['evaluate', 'both']:
        print("Running Evaluation (SSM NWP)...")
        if mode == 'evaluate':
            if not os.path.exists(model_path):
                print(f"Error: Model not found at {model_path}. Please train first.")
                return
            model.load_state_dict(torch.load(model_path, map_location=device))
            
        perplexity = final_evaluation(model, val_loader, "SSM (NWP)")
        print("-" * 30)
        print(f"Model: SSM (Next Word Prediction)")
        print(f"Final Perplexity: {perplexity:.2f}")
        print("-" * 30)
        
        # Sample prediction
        from .common import predict_next_word_topk, generate_full_transmission
        prompt = "the collective forces will"
        pred = predict_next_word_topk(model, vocab, prompt, device)
        print(f"Prompt: {prompt}")
        print(f"Next word prediction: {pred}")
        
        print("\nGenerated Transmission:")
        gen = generate_full_transmission(model, vocab, prompt, device)
        print(gen)
        print("-" * 30)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, default="evaluate")
    args = parser.parse_args()
    main(args.config, args.mode)
