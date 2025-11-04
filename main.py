

import sys
import os
import argparse
from data_maker.data_provider import Data_provider_levir, Data_provider_SYSU, Data_provider_WHU
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import random
import numpy as np
from method.Model import MambaCSSMUnet
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d
from utils.metrics.ev import Evaluator
from utils.loss.L import lovasz_softmax
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Change Detection Training Script')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['levir', 'sysu', 'whu'],
                        help='Dataset to use: levir, sysu, or whu')
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to training data (for WHU: main data directory)')
    parser.add_argument('--test_path', type=str, default=None,
                        help='Path to test data (not used for WHU dataset)')
    parser.add_argument('--val_path', type=str, default=None,
                        help='Path to validation data (not used for WHU dataset)')
    
    # WHU-CD specific arguments
    parser.add_argument('--train_txt', type=str, default=None,
                        help='Text file for WHU-CD training data (required for WHU dataset)')
    parser.add_argument('--test_txt', type=str, default=None,
                        help='Text file for WHU-CD test data (required for WHU dataset)')
    parser.add_argument('--val_txt', type=str, default=None,
                        help='Text file for WHU-CD validation data (required for WHU dataset)')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--step_size', type=int, default=10,
                        help='Step size for learning rate scheduler (default: 10)')
    
    # Model saving
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints (default: ./checkpoints)')
    parser.add_argument('--model_name', type=str, default='best_model.pth',
                        help='Name for saved model file (default: best_model.pth)')
    
    # Other settings
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    
    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_data_provider(dataset_name):
    """Return the appropriate data provider class based on dataset name"""
    providers = {
        'levir': Data_provider_levir,
        'sysu': Data_provider_SYSU,
        'whu': Data_provider_WHU
    }
    return providers[dataset_name]


def seed_worker(worker_id):
    worker_seed = 42
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(model, data, loss_ce, opt, device, train_list):
    model.train()
    size = len(data.dataset)

    for b, (pre, post, target) in enumerate(data):
        pre, post, target = pre.to(device), post.to(device), target.to(device)

        y_pred = model(pre, post)

        loss = loss_ce(y_pred, target) + lovasz_softmax(F.softmax(y_pred, dim=1), target, ignore=255)    

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_list.append(loss.item())

        print(f"loss:{loss.item():.4f} [{b * len(pre)} | {size}]")


def test(model, data, loss_ce, device, evaluator, val_list):
    model.eval()
    size = len(data.dataset)
    num_batch = len(data)
    test_loss = 0

    evaluator.reset()

    with torch.no_grad():
        for pre, post, target in data:
            pre, post, target = pre.to(device), post.to(device), target.to(device)

            y_pred = model(pre, post) 
            test_loss += loss_ce(y_pred, target).item()
            output_clf = y_pred.data.cpu().numpy()
            output_clf = np.argmax(output_clf, axis=1)
            labels_clf = target.cpu().numpy()

            evaluator.add_batch(labels_clf, output_clf)

        test_loss /= num_batch
        val_list.append(test_loss)
        print(f"Validation Loss: {test_loss:.4f}")
        print(f"IoU: {evaluator.Intersection_over_Union()}")
        print(f"Confusion Matrix:\n{evaluator.confusion_matrix}")
        return np.array(evaluator.Intersection_over_Union()).mean()


def main():
    args = parse_args()
    
    # Validate dataset requirements
    if args.dataset == 'whu':
        if not all([args.train_txt, args.test_txt, args.val_txt]):
            print("Error: WHU dataset requires --train_txt, --test_txt, and --val_txt arguments")
            sys.exit(1)
    else:
        if not all([args.test_path, args.val_path]):
            print(f"Error: {args.dataset.upper()} dataset requires --train_path, --test_path, and --val_path arguments")
            sys.exit(1)
    
    # Set seed
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataset
    print(f"\nLoading {args.dataset.upper()} dataset...")
    DataProvider = get_data_provider(args.dataset)
    
    if args.dataset == 'whu':
        # WHU uses single data path with different text files
        train_ds = DataProvider(args.train_path, args.train_txt)
        test_ds = DataProvider(args.train_path, args.test_txt)
        val_ds = DataProvider(args.train_path, args.val_txt)
    else:
        # LEVIR and SYSU use separate paths
        train_ds = DataProvider(args.train_path)
        test_ds = DataProvider(args.test_path)
        val_ds = DataProvider(args.val_path)
    
    # Create data loaders
    train_dl = DataLoader(dataset=train_ds, batch_size=args.batch_size, 
                         shuffle=True, num_workers=args.num_workers, 
                         worker_init_fn=seed_worker)
    val_dl = DataLoader(dataset=val_ds, batch_size=args.batch_size, 
                       shuffle=False, num_workers=1, 
                       worker_init_fn=seed_worker)
    test_dl = DataLoader(dataset=test_ds, batch_size=args.batch_size, 
                        shuffle=False, num_workers=1, 
                        worker_init_fn=seed_worker)
    
    # Initialize model
    print("\nInitializing model...")
    model = MambaCSSMUnet().to(device)
    
    # Define loss and optimizer
    loss_ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=args.step_size)
    
    # Training setup
    train_list = []
    val_list = []
    evaluator = Evaluator(num_class=2)
    best_val_iou = 0.0
    best_model_weight = None
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*60)
    
    for e in range(args.epochs):
        print(f"\nEpoch: {e+1}/{args.epochs}")
        t1 = time.time()
        
        train(model, train_dl, loss_ce, opt, device, train_list)
        
        val_iou = test(model, val_dl, loss_ce, device, evaluator, val_list)
        
        if val_iou > best_val_iou:
            print(f"✓ Best model updated! IoU improved from {best_val_iou:.4f} to {val_iou:.4f}")
            best_val_iou = val_iou
            best_model_weight = copy.deepcopy(model.state_dict())
            
            # Save best model
            save_path = os.path.join(args.save_dir, args.model_name)
            torch.save(best_model_weight, save_path)
            print(f"Model saved to {save_path}")
        
        scheduler.step()
        print(f"Learning Rate: {scheduler.get_last_lr()}")
        
        t2 = time.time()
        print(f"Epoch Time: {t2 - t1:.2f} seconds")
        print("-"*60)
    
    print("\n" + "="*60)
    print(f"Training completed! Best IoU: {best_val_iou:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()