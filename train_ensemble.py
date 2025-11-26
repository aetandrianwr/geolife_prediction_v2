"""
Simple Ensemble Strategy: Train baseline model with different seeds.

Key insight: Advanced techniques (focal loss, mixup) HURT generalization.
The original baseline (model_v2) with simple training is BEST.

Strategy:
1. Train EXACT baseline architecture (88d, 4L)
2. Use ORIGINAL training (standard CE loss, no mixup)
3. Train with 5 different random seeds
4. Ensemble predictions (average logits)
5. Expect: Each model ~38% test, ensemble ~40-42%

Usage:
    python train_ensemble.py
"""

import os
import sys
import random
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import get_dataloaders
from src.models.attention_model import LocationPredictionModel
from src.utils.trainer import Trainer
from src.utils.config import get_config
from src.utils.logger import setup_logger
from src.utils.metrics import calculate_correct_total_prediction, get_performance_dict


def set_seed(seed: int):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_single_model(seed, device, data_dir, logger):
    """Train a single model with given seed."""
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING MODEL WITH SEED {seed}")
    logger.info(f"{'='*80}")
    
    set_seed(seed)
    
    # Load data
    train_loader, val_loader, test_loader, dataset_info = get_dataloaders(
        data_dir=data_dir,
        batch_size=64,  # Original baseline
        max_len=50,
        num_workers=0
    )
    
    # Create EXACT baseline model
    model = LocationPredictionModel(
        num_locations=dataset_info['num_locations'],
        num_users=dataset_info['num_users'],
        d_model=88,  # Exact baseline
        d_inner=176,
        n_layers=4,
        n_head=8,
        d_k=11,
        d_v=11,
        dropout=0.15,  # Original baseline dropout
        max_len=50
    )
    
    num_params = model.count_parameters()
    logger.info(f"Model: {num_params:,} parameters")
    
    # Create checkpoint directory
    checkpoint_dir = Path('checkpoints') / f'ensemble_seed_{seed}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Train with ORIGINAL settings (no focal loss, no mixup!)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        num_locations=dataset_info['num_locations'],
        learning_rate=0.0005,  # Original baseline
        weight_decay=0.0001,   # Original baseline
        label_smoothing=0.1,   # Original baseline
        max_epochs=200,
        patience=30,
        checkpoint_dir=str(checkpoint_dir),
        log_interval=50,
        logger=logger
    )
    
    # Train
    test_metrics = trainer.train()
    
    logger.info(f"\nSeed {seed} Results:")
    logger.info(f"  Val Acc@1:  {trainer.best_val_acc:.2f}%")
    logger.info(f"  Test Acc@1: {test_metrics['acc@1']:.2f}%")
    logger.info(f"  Gap: {trainer.best_val_acc - test_metrics['acc@1']:.2f}%")
    
    # Return model path and metrics
    return {
        'seed': seed,
        'model_path': str(checkpoint_dir / 'best_model.pt'),
        'val_acc': trainer.best_val_acc,
        'test_acc': test_metrics['acc@1'],
        'test_metrics': test_metrics
    }


def ensemble_evaluate(model_paths, device, data_dir, logger):
    """Evaluate ensemble of models."""
    logger.info(f"\n{'='*80}")
    logger.info("ENSEMBLE EVALUATION")
    logger.info(f"{'='*80}")
    logger.info(f"Ensembling {len(model_paths)} models")
    
    # Load data
    _, _, test_loader, dataset_info = get_dataloaders(
        data_dir=data_dir,
        batch_size=64,
        max_len=50,
        num_workers=0
    )
    
    # Load all models
    models = []
    for path in model_paths:
        model = LocationPredictionModel(
            num_locations=dataset_info['num_locations'],
            num_users=dataset_info['num_users'],
            d_model=88,
            d_inner=176,
            n_layers=4,
            n_head=8,
            d_k=11,
            d_v=11,
            dropout=0.15,
            max_len=50
        )
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        models.append(model)
    
    logger.info(f"Loaded {len(models)} models")
    
    # Ensemble prediction
    all_ensemble_logits = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Get predictions from all models
            logits_list = []
            for model in models:
                logits = model(batch)
                logits_list.append(logits)
            
            # Average logits (better than averaging probabilities)
            ensemble_logits = torch.stack(logits_list).mean(dim=0)
            
            all_ensemble_logits.append(ensemble_logits.cpu())
            all_targets.append(batch['target'].cpu())
    
    # Concatenate all results
    all_ensemble_logits = torch.cat(all_ensemble_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    metrics_array, _, _ = calculate_correct_total_prediction(all_ensemble_logits, all_targets)
    metrics_dict = {
        'correct@1': metrics_array[0],
        'correct@3': metrics_array[1],
        'correct@5': metrics_array[2],
        'correct@10': metrics_array[3],
        'rr': metrics_array[4],
        'ndcg': metrics_array[5],
        'f1': 0.0,
        'total': metrics_array[6]
    }
    performance = get_performance_dict(metrics_dict)
    
    logger.info("\n" + "="*80)
    logger.info("ENSEMBLE TEST RESULTS:")
    logger.info(f"  Test Acc@1:  {performance['acc@1']:.2f}%")
    logger.info(f"  Test Acc@5:  {performance['acc@5']:.2f}%")
    logger.info(f"  Test Acc@10: {performance['acc@10']:.2f}%")
    logger.info(f"  Test MRR:    {performance['mrr']:.2f}%")
    logger.info(f"  Test NDCG:   {performance['ndcg']:.2f}%")
    logger.info("="*80)
    
    # Goal assessment
    logger.info("\n" + "="*80)
    logger.info("GOAL ASSESSMENT")
    logger.info("="*80)
    if performance['acc@1'] >= 50.0:
        logger.info(f"🎉 EXCELLENT! {performance['acc@1']:.2f}% (Target: >50%)")
    elif performance['acc@1'] >= 40.0:
        logger.info(f"✓ SUCCESS! {performance['acc@1']:.2f}% (Minimum: 40%)")
        logger.info(f"  Gap to 50%: {50.0 - performance['acc@1']:.2f}%")
    else:
        logger.info(f"⚠ BELOW TARGET: {performance['acc@1']:.2f}% (Need: 40%)")
        logger.info(f"  Gap: {40.0 - performance['acc@1']:.2f}%")
    logger.info("="*80)
    
    return performance


def main():
    """Main ensemble training."""
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = '/content/geolife_prediction_v2/data/geolife'
    
    # Setup logging
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    logger = setup_logger('ensemble', str(log_dir / 'ensemble_training.log'))
    
    logger.info("="*80)
    logger.info("SIMPLE ENSEMBLE STRATEGY")
    logger.info("="*80)
    logger.info("Training EXACT baseline model with 5 different seeds")
    logger.info("NO focal loss, NO mixup, NO fancy tricks")
    logger.info("Just simple, proven baseline + ensemble")
    logger.info("="*80)
    
    # Train models with different seeds
    seeds = [42, 123, 456, 789, 2024]
    results = []
    
    for seed in seeds:
        result = train_single_model(seed, device, data_dir, logger)
        results.append(result)
    
    # Print individual results
    logger.info("\n" + "="*80)
    logger.info("INDIVIDUAL MODEL RESULTS:")
    logger.info("="*80)
    for r in results:
        logger.info(f"Seed {r['seed']:4d}: Val {r['val_acc']:.2f}% → Test {r['test_acc']:.2f}% (Gap: {r['val_acc']-r['test_acc']:.2f}%)")
    
    avg_test = np.mean([r['test_acc'] for r in results])
    logger.info(f"\nAverage Test Acc@1: {avg_test:.2f}%")
    
    # Ensemble evaluation
    model_paths = [r['model_path'] for r in results]
    ensemble_perf = ensemble_evaluate(model_paths, device, data_dir, logger)
    
    # Save results
    result_dir = Path('results')
    result_dir.mkdir(exist_ok=True)
    with open(result_dir / 'ensemble_results.txt', 'w') as f:
        f.write("SIMPLE ENSEMBLE RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write("Individual models:\n")
        for r in results:
            f.write(f"  Seed {r['seed']}: {r['test_acc']:.2f}% test\n")
        f.write(f"\nAverage: {avg_test:.2f}%\n")
        f.write(f"\nEnsemble: {ensemble_perf['acc@1']:.2f}% test\n")
        f.write(f"Improvement: +{ensemble_perf['acc@1'] - avg_test:.2f}%\n")
    
    logger.info(f"\nResults saved to: results/ensemble_results.txt")
    
    return ensemble_perf


if __name__ == '__main__':
    main()
