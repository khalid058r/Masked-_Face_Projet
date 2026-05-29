import argparse
from pathlib import Path
import pandas as pd
import torch

from src.data.indexer import build_index
from src.training.trainer import train_model, train_gan

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for Masked Face Super Resolution")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--task", type=str, choices=["inpainting", "sr"], default="inpainting", help="Task to train: inpainting or sr")
    parser.add_argument("--model_type", type=str, choices=["unet", "gan"], default="unet", help="Architecture to train")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Configuration
    config = {
        'lr_size': 64,
        'hr_size': 128,
        'batch_size': args.batch_size,
        'num_workers': 2,
        'epochs': args.epochs,
        'lr': args.lr,
        'betas': (0.5, 0.999),
        'base_ch': 32,
        'lambda_perc': 0.1,
        'use_amp': torch.cuda.is_available(),
        'val_subset': 500
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Starting Training ---")
    print(f"Task: {args.task}")
    print(f"Model Type: {args.model_type.upper()}")
    print(f"Device: {device}")
    
    # 2. Build Index
    data_path = Path(args.data_dir)
    print(f"Building index from {data_path}...")
    index_df = build_index(data_path, parts=['part1', 'part2'])
    print(f"Found {len(index_df)} images.")
    
    if len(index_df) == 0:
        print("Error: No images found. Please check your --data_dir path.")
        return
        
    # 3. Train
    print("Starting training loop...")
    out_dir = Path("checkpoints")
    out_dir.mkdir(exist_ok=True)
    
    if args.model_type == "gan":
        generator, discriminator, history = train_gan(task=args.task, cfg=config, index_df=index_df, device=device)
        save_path = out_dir / f"gan_{args.task}_epochs_{args.epochs}.pt"
        torch.save({'model': generator.state_dict(), 'discriminator': discriminator.state_dict()}, save_path)
    else:
        model, history = train_model(task=args.task, cfg=config, index_df=index_df, device=device)
        save_path = out_dir / f"unet_{args.task}_epochs_{args.epochs}.pt"
        torch.save({'model': model.state_dict()}, save_path)
        
    print(f"Model successfully saved to {save_path}")

if __name__ == "__main__":
    main()
