"""
Advanced training script for next-location prediction with state-of-the-art techniques.

Features:
- Focal loss and class-balanced loss
- Hierarchical attention
- Cyclic temporal encoding
- Multi-task learning
- Mixup augmentation
- Snapshot ensembles
- Advanced optimization with warmup

Usage:
    python train_advanced.py --config configs/advanced_v1.yml
    python train_advanced.py --config configs/advanced_v2.yml --seed 123
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import get_dataloaders
from src.models.advanced_model import AdvancedLocationPredictionModel
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


def get_class_distribution(data_dir):
    """Calculate class distribution for class-balanced loss."""
    with open(f'{data_dir}/geolife_transformer_7_train.pk', 'rb') as f:
        train_data = pickle.load(f)
    
    # Count targets
    targets = [sample['Y'] for sample in train_data]
    target_counts = Counter(targets)
    
    # Create array of samples per class (assume 1187 classes, indexed from 1)
    num_classes = 1187
    samples_per_class = np.ones(num_classes)  # Initialize with 1 to avoid division by zero
    
    for loc_id, count in target_counts.items():
        if 0 < loc_id < num_classes:
            samples_per_class[loc_id] = count
    
    return samples_per_class


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train advanced next-location prediction model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file
    parser.add_argument('--config', type=str, default='configs/advanced_v1.yml',
                       help='Path to config file')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=None)
    parser.add_argument('--d_inner', type=int, default=None)
    parser.add_argument('--n_layers', type=int, default=None)
    parser.add_argument('--n_head', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--use_hierarchical', action='store_true', default=None)
    parser.add_argument('--use_auxiliary', action='store_true', default=None)
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--loss_type', type=str, default=None,
                       choices=['focal', 'class_balanced', 'multitask', 'ce'])
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--max_epochs', type=int, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--use_mixup', action='store_true', default=None)
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default=None)
    
    # Experiment arguments
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--experiment_name', type=str, default=None)
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = get_config(args.config, args)
    
    # Set random seed
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
    logger = setup_logger('geolife_advanced', str(log_file))
    
    # Print configuration
    print_config(config)
    logger.info("="*80)
    logger.info("ADVANCED TRAINING WITH STATE-OF-THE-ART TECHNIQUES")
    logger.info("="*80)
    
    # Save configuration
    config_save_path = checkpoint_dir / 'config.yml'
    save_config(config, str(config_save_path))
    logger.info(f"Configuration saved to: {config_save_path}")
    
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
    logger.info(f"Dataset loaded: {dataset_info['num_train']} train, "
                f"{dataset_info['num_val']} val, {dataset_info['num_test']} test samples")
    
    # Get class distribution for class-balanced loss
    samples_per_class = None
    if config.training.get('loss_type') == 'class_balanced':
        logger.info("Calculating class distribution for class-balanced loss...")
        samples_per_class = get_class_distribution(config.data['data_dir'])
        logger.info(f"Class distribution calculated: {len(samples_per_class)} classes")
    
    # Create model
    logger.info("Creating advanced model...")
    logger.info(f"  Architecture: Hierarchical={'Yes' if config.model.get('use_hierarchical', True) else 'No'}, "
                f"Auxiliary={'Yes' if config.model.get('use_auxiliary', False) else 'No'}")
    
    model = AdvancedLocationPredictionModel(
        num_locations=dataset_info['num_locations'],
        num_users=dataset_info['num_users'],
        d_model=config.model['d_model'],
        d_inner=config.model['d_inner'],
        n_layers=config.model['n_layers'],
        n_head=config.model['n_head'],
        d_k=config.model['d_k'],
        d_v=config.model['d_v'],
        dropout=config.model['dropout'],
        max_len=config.model['max_len'],
        use_hierarchical=config.model.get('use_hierarchical', True),
        use_auxiliary=config.model.get('use_auxiliary', False)
    )
    
    # Check parameter budget
    num_params = model.count_parameters()
    if num_params >= 500000:
        logger.error(f"Model has {num_params:,} parameters (EXCEEDS 500K budget!)")
        logger.error("Aborting training. Please reduce model size.")
        return None
    else:
        logger.info(f"✓ Model has {num_params:,} parameters ({num_params/500000*100:.1f}% of 500K budget)")
    
    # Log experiment info
    log_experiment_info(logger, config, model, dataset_info)
    
    # Log training techniques
    logger.info("\nTRAINING TECHNIQUES:")
    logger.info(f"  Loss function: {config.training.get('loss_type', 'focal')}")
    if config.training.get('loss_type') in ['focal', 'multitask']:
        logger.info(f"  Focal gamma: {config.training.get('gamma', 2.0)}")
    logger.info(f"  Label smoothing: {config.training.get('label_smoothing', 0.1)}")
    logger.info(f"  Mixup augmentation: {'Yes' if config.training.get('use_mixup', False) else 'No'}")
    if config.training.get('use_mixup', False):
        logger.info(f"  Mixup alpha: {config.training.get('mixup_alpha', 0.2)}")
    logger.info(f"  Warmup epochs: {config.training.get('warmup_epochs', 5)}")
    logger.info(f"  Snapshot ensemble: {'Yes' if config.training.get('snapshot_ensemble', False) else 'No'}")
    
    # Create trainer
    logger.info("\nInitializing advanced trainer...")
    trainer = AdvancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        num_locations=dataset_info['num_locations'],
        samples_per_class=samples_per_class,
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
        use_auxiliary=config.training.get('use_auxiliary', False),
        aux_weight=config.training.get('aux_weight', 0.3),
        snapshot_ensemble=config.training.get('snapshot_ensemble', False),
        snapshot_interval=config.training.get('snapshot_interval', 10),
        logger=logger
    )
    
    # Train
    logger.info("")
    logger.info("="*80)
    logger.info("STARTING ADVANCED TRAINING")
    logger.info("="*80)
    
    test_metrics = trainer.train()
    
    # Check if we achieved the goal
    logger.info("\n" + "="*80)
    logger.info("GOAL ASSESSMENT")
    logger.info("="*80)
    if test_metrics['acc@1'] >= 50.0:
        logger.info(f"🎉 EXCELLENT! Achieved {test_metrics['acc@1']:.2f}% (Target: >50%)")
    elif test_metrics['acc@1'] >= 40.0:
        logger.info(f"✓ SUCCESS! Achieved {test_metrics['acc@1']:.2f}% (Minimum: 40%)")
        logger.info(f"  Still {50.0 - test_metrics['acc@1']:.2f}% away from 50% target")
    else:
        logger.info(f"⚠ BELOW TARGET: {test_metrics['acc@1']:.2f}% (Minimum: 40%)")
        logger.info(f"  Need {40.0 - test_metrics['acc@1']:.2f}% more to reach minimum")
    logger.info("="*80)
    
    # Save results
    result_file = result_dir / f'{config.experiment["name"]}_results.txt'
    with open(result_file, 'w') as f:
        f.write(f"ADVANCED MODEL RESULTS\n")
        f.write(f"Experiment: {config.experiment['name']}\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Seed: {config.experiment['seed']}\n")
        f.write("="*80 + "\n\n")
        
        f.write("MODEL:\n")
        f.write(f"  d_model: {config.model['d_model']}\n")
        f.write(f"  n_layers: {config.model['n_layers']}\n")
        f.write(f"  n_head: {config.model['n_head']}\n")
        f.write(f"  dropout: {config.model['dropout']}\n")
        f.write(f"  Parameters: {num_params:,}\n")
        f.write(f"  Hierarchical attention: {config.model.get('use_hierarchical', True)}\n")
        f.write(f"  Auxiliary task: {config.model.get('use_auxiliary', False)}\n\n")
        
        f.write("TRAINING:\n")
        f.write(f"  Loss: {config.training.get('loss_type', 'focal')}\n")
        f.write(f"  Mixup: {config.training.get('use_mixup', False)}\n")
        f.write(f"  Snapshot ensemble: {config.training.get('snapshot_ensemble', False)}\n\n")
        
        f.write("RESULTS:\n")
        f.write(f"  Test Acc@1:  {test_metrics['acc@1']:.2f}%\n")
        f.write(f"  Test Acc@5:  {test_metrics['acc@5']:.2f}%\n")
        f.write(f"  Test Acc@10: {test_metrics['acc@10']:.2f}%\n")
        f.write(f"  Test MRR:    {test_metrics['mrr']:.2f}%\n")
        f.write(f"  Test NDCG:   {test_metrics['ndcg']:.2f}%\n\n")
        
        f.write(f"  Best Val Acc@1: {trainer.best_val_acc:.2f}%\n")
        f.write(f"  Best Epoch: {trainer.best_epoch + 1}\n")
        f.write(f"  Val-Test Gap: {trainer.best_val_acc - test_metrics['acc@1']:.2f}%\n")
    
    logger.info(f"\nResults saved to: {result_file}")
    logger.info(f"Checkpoint saved to: {checkpoint_dir / 'best_model.pt'}")
    
    return test_metrics


if __name__ == '__main__':
    main()
