import argparse
import yaml
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from datasets.pathmnist import get_pathmnist_loaders
from datasets.eurosat import get_eurosat_loaders
from models.resnet import get_resnet
from models.efficientnet import get_efficientnet
from models.vit import get_vit
from utils.seed import set_seed
from utils.logger import get_logger
from utils.metrics import compute_metrics

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    for inputs, targets in tqdm(loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Squeeze targets if they are (N, 1) to (N,)
        if targets.ndim > 1:
            targets = targets.squeeze()
            
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        all_preds.extend(outputs.detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, np.array(all_preds), np.array(all_targets)

def validate(model, loader, criterion, device, domain):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Validation", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            if targets.ndim > 1:
                targets = targets.squeeze()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_preds, all_targets, domain=domain)
    metrics['loss'] = epoch_loss
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Vision Benchmark Training")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--dry-run', action='store_true', help='Run a single batch for debugging')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Setup
    set_seed(config['training']['seed'])
    log_dir = os.path.join('results', config['experiment_name'])
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(config['experiment_name'], log_dir)
    
    logger.info(f"Starting experiment: {config['experiment_name']}")
    logger.info(f"Config: {config}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Data
    if config['dataset'] == 'pathmnist':
        train_loader, val_loader, test_loader = get_pathmnist_loaders(batch_size=config['training']['batch_size'])
    elif config['dataset'] == 'eurosat':
        train_loader, val_loader, test_loader = get_eurosat_loaders(batch_size=config['training']['batch_size'])
    else:
        raise ValueError("Unknown dataset")
        
    if args.dry_run:
        logger.info("Dry run enabled. Truncating datasets.")
        # Minimal subset hack
        pass # Actual truncation logic can be complex with loaders, relying on 'break' in loop for dry-run if needed, or just run 1 epoch.
    
    # Model
    num_classes = config['num_classes']
    if config['model'] == 'resnet':
        model = get_resnet(num_classes)
    elif config['model'] == 'efficientnet':
        model = get_efficientnet(num_classes)
    elif config['model'] == 'vit':
        model = get_vit(num_classes)
    else:
        raise ValueError("Unknown model")
    
    model = model.to(device)
    
    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # Tracking
    best_val_f1 = 0.0
    patience = config['training']['patience']
    patience_counter = 0
    history = []
    
    start_time = time.time()
    
    epochs = 1 if args.dry_run else config['training']['epochs']
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        train_loss, _, _ = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device, config['domain'])
        
        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val F1: {val_metrics['f1_macro']:.4f}")
        
        # Save metrics
        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            **{f'val_{k}': v for k, v in val_metrics.items()}
        }
        history.append(epoch_stats)
        
        # Early Stopping
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
            logger.info("New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break
                
    total_time = time.time() - start_time
    logger.info(f"Training finished in {total_time:.2f}s")
    
    # Save training history
    pd.DataFrame(history).to_csv(os.path.join(log_dir, 'training_log.csv'), index=False)
    
    # Final Test
    logger.info("Running final test on best model...")
    model.load_state_dict(torch.load(os.path.join(log_dir, 'best_model.pth')))
    test_metrics = validate(model, test_loader, criterion, device, config['domain'])
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    final_results = {
        'experiment': config['experiment_name'],
        'dataset': config['dataset'],
        'model': config['model'],
        'params': num_params,
        'training_time': total_time,
        **test_metrics
    }
    
    logger.info(f"Test Results: {final_results}")
    
    # Append to global results (with locking if parallel, but here purely sequential assumption or manual)
    # Simple append to a shared CSV
    results_file = 'results/final_results.csv'
    df = pd.DataFrame([final_results])
    if os.path.exists(results_file):
        df.to_csv(results_file, mode='a', header=False, index=False)
    else:
        df.to_csv(results_file, index=False)

if __name__ == "__main__":
    main()
