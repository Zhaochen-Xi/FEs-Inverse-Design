import argparse
import torch
import os
import json
from model import CVAE
from utils import LoadDataset, get_train_test_loader, train_cvae

def main():
    parser = argparse.ArgumentParser(description="Train a CVAE")
    
    # System args
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    
    # Model args
    parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--img_size', type=int, default=128, help='Input image size (width/height)')
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimension of latent space')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of conditional classes')
    
    # Training args
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train/test split ratio')
    
    # Paths (REQUIRED args ensure no hardcoded paths exist)
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--save_path', type=str, default='./checkpoints', help='Directory to save models')
    parser.add_argument('--save_logs', action='store_true', help='Save training logs')

    args = parser.parse_args()

    # Device setup
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Model Init
    model = CVAE(
        in_channels=args.in_channels, 
        num_classes=args.num_classes,
        latent_dims=args.latent_dim,
        img_size=args.img_size,
        hidden_dims=None
    )

    print(f"Model initialized: In({args.in_channels}), Latent({args.latent_dim}), ImgSize({args.img_size})")

    # Data Loading
    # NOTE: Ensure your LoadDataset class in utils.py implements the correct extraction logic for your data
    try:
        ori_data = LoadDataset(root_dir=args.data_path)
        train_loader, test_loader = get_train_test_loader(
            ori_data, 
            batch_size=args.batch_size, 
            train_ratio=args.train_ratio, 
            num_workers=args.num_workers
        )
        print(f"Data loaded. Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")
    except NotImplementedError:
        print("ERROR: You must implement the '_process_data' method in utils.py to parse your specific data format.")
        return

    # Training
    train_loss, test_loss = train_cvae(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        save_dir=args.save_path
    )

    # Logging
    if args.save_logs:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        log_path = os.path.join(args.save_path, "losses.json")
        with open(log_path, "w") as f:
            json.dump({"train_loss": train_loss, "test_loss": test_loss}, f, indent=4)
        print(f"Logs saved to {log_path}")

if __name__ == '__main__':
    main()