"""
Fast Ensemble: Train 3 more baseline models quickly + ensemble with existing best.

Reuse existing model_v2 (seed=42, 37.95% test) + train 3 more with different seeds.
Expected: 4-model ensemble → ~40%+ test accuracy

Usage:
    python train_fast_ensemble.py
"""

import os
import sys
import random
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import get_dataloaders
from src.models.attention_model import LocationPredictionModel
from src.utils.trainer import Trainer
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
    """Train baseline model with given seed."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training model with seed {seed}")
    logger.info(f"{'='*60}")
    
    set_seed(seed)
    
    # Load data
    train_loader, val_loader, test_loader, dataset_info = get_dataloaders(
        data_dir=data_dir,
        batch_size=64,
        max_len=50,
        num_workers=0
    )
    
    # Baseline model
    model = LocationPredictionModel(
        num_locations=1187,
        num_users=46,
        d_model=88,
        d_inner=176,
        n_layers=4,
        n_head=8,
        d_k=11,
        d_v=11,
        dropout=0.15,
        max_len=50
    )
    
    logger.info(f"Parameters: {model.count_parameters():,}")
    
    checkpoint_dir = Path('checkpoints') / f'fast_ensemble_seed_{seed}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        num_locations=1187,
        learning_rate=0.0005,
        weight_decay=0.0001,
        label_smoothing=0.1,
        max_epochs=200,
        patience=30,
        checkpoint_dir=str(checkpoint_dir),
        log_interval=100,  # Less logging for speed
        logger=logger
    )
    
    test_metrics = trainer.train()
    
    logger.info(f"Seed {seed}: Val {trainer.best_val_acc:.2f}% → Test {test_metrics['acc@1']:.2f}%")
    
    return str(checkpoint_dir / 'best_model.pt'), test_metrics['acc@1']


@torch.no_grad()
def ensemble_evaluate(model_paths, device, data_dir, logger):
    """Evaluate ensemble."""
    logger.info(f"\n{'='*60}")
    logger.info(f"ENSEMBLE EVALUATION ({len(model_paths)} models)")
    logger.info(f"{'='*60}")
    
    _, _, test_loader, _ = get_dataloaders(
        data_dir=data_dir,
        batch_size=64,
        max_len=50,
        num_workers=0
    )
    
    # Load models
    models = []
    for i, path in enumerate(model_paths):
        logger.info(f"Loading model {i+1}/{len(model_paths)}: {Path(path).parent.name}")
        model = LocationPredictionModel(
            num_locations=1187,
            num_users=46,
            d_model=88,
            d_inner=176,
            n_layers=4,
            n_head=8,
            d_k=11,
            d_v=11,
            dropout=0.15,
            max_len=50
        ).to(device)
        
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
    
    # Ensemble prediction
    all_logits = []
    all_targets = []
    
    for batch in tqdm(test_loader, desc='Ensemble eval'):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # Average logits from all models
        logits_list = [model(batch) for model in models]
        ensemble_logits = torch.stack(logits_list).mean(dim=0)
        
        all_logits.append(ensemble_logits.cpu())
        all_targets.append(batch['target'].cpu())
    
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    
    # Calculate metrics
    metrics_array, _, _ = calculate_correct_total_prediction(all_logits, all_targets)
    perf = get_performance_dict({
        'correct@1': metrics_array[0],
        'correct@3': metrics_array[1],
        'correct@5': metrics_array[2],
        'correct@10': metrics_array[3],
        'rr': metrics_array[4],
        'ndcg': metrics_array[5],
        'f1': 0.0,
        'total': metrics_array[6]
    })
    
    logger.info("\n" + "="*60)
    logger.info("ENSEMBLE RESULTS:")
    logger.info(f"  Test Acc@1:  {perf['acc@1']:.2f}%")
    logger.info(f"  Test Acc@5:  {perf['acc@5']:.2f}%")
    logger.info(f"  Test Acc@10: {perf['acc@10']:.2f}%")
    logger.info(f"  Test MRR:    {perf['mrr']:.2f}%")
    logger.info("="*60)
    
    if perf['acc@1'] >= 50.0:
        logger.info(f"🎉 EXCELLENT! {perf['acc@1']:.2f}% (Target: >50%)")
    elif perf['acc@1'] >= 40.0:
        logger.info(f"✓ SUCCESS! {perf['acc@1']:.2f}% (Minimum: 40%)")
    else:
        logger.info(f"⚠ Below target: {perf['acc@1']:.2f}% (Need: 40%)")
        logger.info(f"  Gap: {40.0 - perf['acc@1']:.2f}%")
    
    return perf


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = '/content/geolife_prediction_v2/data/geolife'
    
    logger = setup_logger('fast_ensemble', 'logs/fast_ensemble.log')
    
    logger.info("="*60)
    logger.info("FAST ENSEMBLE STRATEGY")
    logger.info("="*60)
    logger.info("Reuse existing model_v2 (37.95% test)")
    logger.info("Train 3 more models with different seeds")
    logger.info("Ensemble all 4 models")
    logger.info("="*60)
    
    # Use existing best model
    existing_model = 'checkpoints/model_v2/best_model.pt'
    if not Path(existing_model).exists():
        logger.error(f"Existing model not found: {existing_model}")
        return
    
    logger.info(f"\n✓ Using existing model: {existing_model}")
    logger.info("  (seed=42, Test Acc@1=37.95%)")
    
    model_paths = [existing_model]
    
    # Train 3 new models
    new_seeds = [123, 456, 789]
    for seed in new_seeds:
        path, acc = train_single_model(seed, device, data_dir, logger)
        model_paths.append(path)
        logger.info(f"✓ Trained seed {seed}: {acc:.2f}% test")
    
    # Ensemble evaluation
    ensemble_perf = ensemble_evaluate(model_paths, device, data_dir, logger)
    
    # Save results
    with open('results/fast_ensemble_results.txt', 'w') as f:
        f.write("FAST ENSEMBLE RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write("Models:\n")
        f.write(f"  1. model_v2 (seed=42): 37.95%\n")
        for i, seed in enumerate(new_seeds, 2):
            f.write(f"  {i}. seed={seed}: trained\n")
        f.write(f"\nEnsemble Test Acc@1: {ensemble_perf['acc@1']:.2f}%\n")
    
    logger.info("\n✓ Results saved to: results/fast_ensemble_results.txt")
    
    return ensemble_perf


if __name__ == '__main__':
    main()
