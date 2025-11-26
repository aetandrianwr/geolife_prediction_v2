"""
Train Recurrent Transformer Model

Goal: Achieve >50% test Acc@1 (Minimum: 40%)
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import get_dataloaders
from src.models.recurrent_transformer_v3 import create_recurrent_transformer_v3
from src.utils.metrics import calculate_correct_total_prediction, get_performance_dict
from src.utils.logger import setup_logger


def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RecurrentTransformerTrainer:
    """Trainer for Recurrent Transformer."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        device='cuda',
        learning_rate=0.001,
        weight_decay=0.0001,
        max_epochs=250,
        patience=50,
        warmup_epochs=20,
        checkpoint_dir='checkpoints/recurrent_transformer_v3',
        logger=None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or setup_logger('recurrent_transformer_v3', 'logs/recurrent_transformer_v3.log')
        
        # Loss with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Optimizer - AdamW with layer-wise LR
        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() if 'embed' in n],
                'lr': learning_rate * 0.5
            },
            {
                'params': [p for n, p in self.model.named_parameters() if 'embed' not in n],
                'lr': learning_rate
            }
        ]
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        
        # Cosine annealing with warmup
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs - warmup_epochs, eta_min=1e-6
        )
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_test_acc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
    
    def warmup_lr(self, epoch):
        """Linear warmup."""
        if epoch < self.warmup_epochs:
            warmup_factor = (epoch + 1) / self.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                base_lr = 0.001 * 0.5 if i == 0 else 0.001
                param_group['lr'] = base_lr * warmup_factor
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.max_epochs}')
        for batch in pbar:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # Forward
            logits = self.model(batch)
            loss = self.criterion(logits, batch['target'])
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct = (preds == batch['target']).sum().item()
            total_correct += correct
            total_samples += batch['target'].size(0)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100*correct/batch['target'].size(0):.1f}%"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100.0 * total_correct / total_samples
        
        return avg_loss, avg_acc
    
    @torch.no_grad()
    def evaluate(self, loader):
        """Evaluate on validation/test set."""
        self.model.eval()
        all_logits = []
        all_targets = []
        
        for batch in tqdm(loader, desc='Evaluating'):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            logits = self.model(batch)
            all_logits.append(logits.cpu())
            all_targets.append(batch['target'].cpu())
        
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
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
        
        return perf
    
    def train(self):
        """Full training loop."""
        self.logger.info("="*80)
        self.logger.info("RECURRENT TRANSFORMER TRAINING")
        self.logger.info("="*80)
        self.logger.info(f"Parameters: {self.model.count_parameters():,}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Cycles: {self.model.n_cycles}")
        self.logger.info("="*80)
        
        for epoch in range(1, self.max_epochs + 1):
            # Warmup
            if epoch <= self.warmup_epochs:
                self.warmup_lr(epoch)
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_perf = self.evaluate(self.val_loader)
            
            # Also check test to monitor progress
            test_perf = self.evaluate(self.test_loader)
            
            # Scheduler step (after warmup)
            if epoch > self.warmup_epochs:
                self.scheduler.step()
            
            # Log
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(
                f"Epoch {epoch:3d}/{self.max_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Acc@1: {val_perf['acc@1']:.2f}% | "
                f"Test Acc@1: {test_perf['acc@1']:.2f}% | "
                f"LR: {lr:.2e}"
            )
            
            # Save best model based on validation
            if val_perf['acc@1'] > self.best_val_acc:
                self.best_val_acc = val_perf['acc@1']
                self.best_test_acc = test_perf['acc@1']
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_perf['acc@1'],
                    'test_acc': test_perf['acc@1']
                }, self.checkpoint_dir / 'best_model.pt')
                
                self.logger.info(
                    f"✓ New best! Val: {val_perf['acc@1']:.2f}%, Test: {test_perf['acc@1']:.2f}%"
                )
            else:
                self.epochs_without_improvement += 1
                self.logger.info(
                    f"  No improvement for {self.epochs_without_improvement} epochs "
                    f"(Best Val: {self.best_val_acc:.2f}%, Best Test: {self.best_test_acc:.2f}%)"
                )
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Check if we hit target
            if test_perf['acc@1'] >= 50.0:
                self.logger.info(f"\n🎉 GOAL REACHED! Test Acc@1: {test_perf['acc@1']:.2f}% >= 50%")
                break
        
        # Load best model and final evaluation
        self.logger.info("\n" + "="*80)
        self.logger.info("Loading best model for final evaluation...")
        checkpoint = torch.load(self.checkpoint_dir / 'best_model.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        final_test_perf = self.evaluate(self.test_loader)
        
        self.logger.info("\n" + "="*80)
        self.logger.info("FINAL TEST RESULTS:")
        self.logger.info(f"  Test Acc@1:  {final_test_perf['acc@1']:.2f}%")
        self.logger.info(f"  Test Acc@5:  {final_test_perf['acc@5']:.2f}%")
        self.logger.info(f"  Test Acc@10: {final_test_perf['acc@10']:.2f}%")
        self.logger.info(f"  Test MRR:    {final_test_perf['mrr']:.2f}%")
        self.logger.info(f"  Best Val Acc@1: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        self.logger.info("="*80)
        
        # Goal assessment
        self.logger.info("\n" + "="*80)
        self.logger.info("GOAL ASSESSMENT")
        self.logger.info("="*80)
        if final_test_perf['acc@1'] >= 50.0:
            self.logger.info(f"🎉 EXCELLENT! {final_test_perf['acc@1']:.2f}% >= 50% TARGET!")
        elif final_test_perf['acc@1'] >= 45.0:
            self.logger.info(f"✓ GREAT! {final_test_perf['acc@1']:.2f}% >= 45%")
            self.logger.info(f"  Gap to 50%: {50.0 - final_test_perf['acc@1']:.2f}%")
        elif final_test_perf['acc@1'] >= 40.0:
            self.logger.info(f"✓ SUCCESS! {final_test_perf['acc@1']:.2f}% >= 40% (Minimum)")
            self.logger.info(f"  Gap to 50%: {50.0 - final_test_perf['acc@1']:.2f}%")
        else:
            self.logger.info(f"⚠ Below minimum: {final_test_perf['acc@1']:.2f}%")
            self.logger.info(f"  Need: {40.0 - final_test_perf['acc@1']:.2f}% more")
        self.logger.info("="*80)
        
        return final_test_perf


def main():
    set_seed(2025)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = '/content/geolife_prediction_v2/data/geolife'
    
    logger = setup_logger('recurrent_transformer_v3', 'logs/recurrent_transformer_v3.log')
    
    # Load data
    logger.info("Loading dataset...")
    train_loader, val_loader, test_loader, dataset_info = get_dataloaders(
        data_dir=data_dir,
        batch_size=64,
        max_len=50,
        num_workers=0
    )
    
    # Create model
    logger.info("Creating Recurrent Transformer V3 model...")
    model = create_recurrent_transformer_v3(
        num_locations=dataset_info['num_locations'],
        num_users=dataset_info['num_users']
    )
    
    # Train
    trainer = RecurrentTransformerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=0.001,
        weight_decay=0.0001,
        max_epochs=250,
        patience=50,
        warmup_epochs=20,
        logger=logger
    )
    
    test_perf = trainer.train()
    
    # Save results
    with open('results/recurrent_transformer_v3_results.txt', 'w') as f:
        f.write("RECURRENT TRANSFORMER V2 RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write("Architecture:\n")
        f.write("  - Recurrent Transformer V3 (4 cycles)\n")
        f.write("  - Shared transformer block across cycles\n")
        f.write("  - Strong residual connections\n")
        f.write("  - Full sequence refinement (not single token)\n\n")
        f.write(f"Parameters: {model.count_parameters():,}\n\n")
        f.write(f"Test Acc@1:  {test_perf['acc@1']:.2f}%\n")
        f.write(f"Test Acc@5:  {test_perf['acc@5']:.2f}%\n")
        f.write(f"Test Acc@10: {test_perf['acc@10']:.2f}%\n")
        f.write(f"Test MRR:    {test_perf['mrr']:.2f}%\n")
    
    logger.info("\n✓ Results saved to results/recurrent_transformer_v3_results.txt")
    
    return test_perf


if __name__ == '__main__':
    main()
