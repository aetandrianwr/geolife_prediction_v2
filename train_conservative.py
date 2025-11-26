"""
Conservative training: Use original proven architecture with advanced training techniques.

This uses the baseline attention_model.py (which achieved 37.95%) but with:
- Focal loss
- Mixup augmentation  
- Better learning rate schedule
- Snapshot ensembles

Usage:
    python train_conservative.py --config configs/conservative_v1.yml
"""

import os
import sys
import random
import argparse
import pickle
import numpy as np
import torch
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import get_dataloaders
from src.models.attention_model import LocationPredictionModel  # Original baseline model!
from src.utils.advanced_trainer import AdvancedTrainer
from src.utils.config import get_config, save_config, print_config
from src.utils.logger import setup_logger, log_experiment_info


def set_seed(seed: int, deterministic: bool = True):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Conservative training with proven baseline')
    parser.add_argument('--config', type=str, default='configs/conservative_v1.yml')
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    config = get_config(args.config, args)
    set_seed(config.experiment['seed'], config.experiment.get('deterministic', True))
    
    # Setup directories
    checkpoint_dir = Path(config.experiment['checkpoint_dir']) / config.experiment['name']
    log_dir = Path(config.experiment['log_dir'])
    result_dir = Path(config.experiment['result_dir'])
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    log_file = log_dir / f"{config.experiment['name']}_train.log"
    logger = setup_logger('conservative', str(log_file))
    
    logger.info("="*80)
    logger.info("CONSERVATIVE TRAINING: BASELINE ARCHITECTURE + ADVANCED TECHNIQUES")
    logger.info("="*80)
    logger.info("Using ORIGINAL attention_model.py (proven 37.95% baseline)")
    logger.info("Enhanced with: Focal loss, Mixup, Better LR schedule, Snapshots")
    logger.info("="*80)
    
    print_config(config)
    
    # Save config
    save_config(config, str(checkpoint_dir / 'config.yml'))
    
    # Set device
    device = config.experiment['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading dataset...")
    train_loader, val_loader, test_loader, dataset_info = get_dataloaders(
        data_dir=config.data['data_dir'],
        batch_size=config.training['batch_size'],
        max_len=config.data['max_len'],
        num_workers=config.data.get('num_workers', 0)
    )
    logger.info(f"Dataset: {dataset_info['num_train']} train, {dataset_info['num_val']} val, {dataset_info['num_test']} test")
    
    # Create BASELINE model (not advanced!)
    logger.info("Creating model (BASELINE ARCHITECTURE)...")
    model = LocationPredictionModel(
        num_locations=dataset_info['num_locations'],
        num_users=dataset_info['num_users'],
        d_model=config.model['d_model'],
        d_inner=config.model['d_inner'],
        n_layers=config.model['n_layers'],
        n_head=config.model['n_head'],
        d_k=config.model['d_k'],
        d_v=config.model['d_v'],
        dropout=config.model['dropout'],
        max_len=config.model['max_len']
    )
    
    # Check parameters
    num_params = model.count_parameters()
    if num_params >= 500000:
        logger.error(f"Model has {num_params:,} parameters (EXCEEDS budget!)")
        return None
    else:
        logger.info(f"✓ Model: {num_params:,} parameters ({num_params/500000*100:.1f}% of budget)")
    
    log_experiment_info(logger, config, model, dataset_info)
    
    logger.info("\nTRAINING ENHANCEMENTS:")
    logger.info(f"  Focal loss (gamma={config.training.get('gamma', 2.0)})")
    logger.info(f"  Mixup augmentation (alpha={config.training.get('mixup_alpha', 0.2)})")
    logger.info(f"  Larger batch size: {config.training['batch_size']}")
    logger.info(f"  Warmup epochs: {config.training.get('warmup_epochs', 5)}")
    logger.info(f"  Snapshot ensemble: {config.training.get('snapshot_ensemble', False)}")
    
    # Create advanced trainer
    trainer = AdvancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        num_locations=dataset_info['num_locations'],
        learning_rate=config.training['learning_rate'],
        weight_decay=config.training['weight_decay'],
        loss_type=config.training.get('loss_type', 'focal'),
        gamma=config.training.get('gamma', 2.0),
        label_smoothing=config.training.get('label_smoothing', 0.1),
        max_epochs=config.training['max_epochs'],
        patience=config.training['patience'],
        warmup_epochs=config.training.get('warmup_epochs', 5),
        checkpoint_dir=str(checkpoint_dir),
        log_interval=config.training.get('log_interval', 50),
        use_mixup=config.training.get('use_mixup', False),
        mixup_alpha=config.training.get('mixup_alpha', 0.2),
        use_auxiliary=False,  # Baseline doesn't support auxiliary
        snapshot_ensemble=config.training.get('snapshot_ensemble', False),
        snapshot_interval=config.training.get('snapshot_interval', 10),
        logger=logger
    )
    
    # Train
    logger.info("\n" + "="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80)
    
    test_metrics = trainer.train()
    
    # Assessment
    logger.info("\n" + "="*80)
    logger.info("GOAL ASSESSMENT")
    logger.info("="*80)
    if test_metrics['acc@1'] >= 50.0:
        logger.info(f"🎉 EXCELLENT! {test_metrics['acc@1']:.2f}% (Target: >50%)")
    elif test_metrics['acc@1'] >= 40.0:
        logger.info(f"✓ SUCCESS! {test_metrics['acc@1']:.2f}% (Minimum: 40%)")
        logger.info(f"  Gap to 50%: {50.0 - test_metrics['acc@1']:.2f}%")
    else:
        logger.info(f"⚠ BELOW TARGET: {test_metrics['acc@1']:.2f}% (Need: 40%)")
        logger.info(f"  Gap: {40.0 - test_metrics['acc@1']:.2f}%")
    logger.info("="*80)
    
    # Save results
    result_file = result_dir / f'{config.experiment["name"]}_results.txt'
    with open(result_file, 'w') as f:
        f.write(f"CONSERVATIVE TRAINING RESULTS\n")
        f.write(f"="*80 + "\n\n")
        f.write(f"Architecture: Baseline (attention_model.py)\n")
        f.write(f"Enhancements: Focal loss + Mixup + Advanced LR\n\n")
        f.write(f"Model: {num_params:,} parameters\n")
        f.write(f"Training: {config.training.get('loss_type', 'focal')} loss\n")
        f.write(f"Mixup: {config.training.get('use_mixup', False)}\n\n")
        f.write(f"Test Acc@1:  {test_metrics['acc@1']:.2f}%\n")
        f.write(f"Test Acc@5:  {test_metrics['acc@5']:.2f}%\n")
        f.write(f"Test Acc@10: {test_metrics['acc@10']:.2f}%\n")
        f.write(f"Test MRR:    {test_metrics['mrr']:.2f}%\n")
        f.write(f"Test NDCG:   {test_metrics['ndcg']:.2f}%\n\n")
        f.write(f"Best Val Acc@1: {trainer.best_val_acc:.2f}%\n")
        f.write(f"Val-Test Gap: {trainer.best_val_acc - test_metrics['acc@1']:.2f}%\n")
    
    logger.info(f"\nResults: {result_file}")
    
    return test_metrics


if __name__ == '__main__':
    main()
