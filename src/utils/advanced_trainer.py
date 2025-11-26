"""
Advanced training utilities with support for:
- Focal loss and class-balanced loss
- Multi-task learning
- Snapshot ensembles
- Advanced optimization strategies
- Mixup augmentation
"""

import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict

from .metrics import calculate_correct_total_prediction, get_performance_dict
from .losses import get_loss_function


class AdvancedTrainer:
    """Advanced trainer with state-of-the-art training techniques."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        num_locations,
        samples_per_class=None,
        learning_rate=5e-4,
        weight_decay=1e-4,
        loss_type='focal',
        gamma=2.0,
        label_smoothing=0.1,
        max_epochs=200,
        patience=30,
        warmup_epochs=5,
        checkpoint_dir='checkpoints',
        log_interval=50,
        use_mixup=False,
        mixup_alpha=0.2,
        use_auxiliary=False,
        aux_weight=0.3,
        snapshot_ensemble=False,
        snapshot_interval=10,
        logger=None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_locations = num_locations
        
        # Loss function
        self.loss_type = loss_type
        self.use_auxiliary = use_auxiliary
        
        if loss_type == 'multitask' and use_auxiliary:
            self.criterion = get_loss_function(
                'multitask',
                num_classes=num_locations,
                gamma=gamma,
                label_smoothing=label_smoothing,
                aux_weight=aux_weight
            )
        elif loss_type == 'class_balanced' and samples_per_class is not None:
            self.criterion = get_loss_function(
                'class_balanced',
                num_classes=num_locations,
                samples_per_class=samples_per_class,
                gamma=gamma
            )
        else:
            self.criterion = get_loss_function(
                loss_type,
                num_classes=num_locations,
                gamma=gamma,
                label_smoothing=label_smoothing
            )
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler with warmup
        self.warmup_epochs = warmup_epochs
        self.warmup_scheduler = None
        if warmup_epochs > 0:
            # Linear warmup
            warmup_factor = 1.0 / warmup_epochs
            def warmup_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) * warmup_factor
                return 1.0
            self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=warmup_lambda
            )
        
        # Main scheduler: Cosine annealing with warm restarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-7
        )
        
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.logger = logger
        
        # Mixup augmentation
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        
        # Snapshot ensemble
        self.snapshot_ensemble = snapshot_ensemble
        self.snapshot_interval = snapshot_interval
        self.snapshots = []
        
        # Tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.train_history = defaultdict(list)
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def mixup_data(self, batch, alpha=0.2):
        """
        Apply mixup augmentation to batch.
        
        Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = batch['locations'].size(0)
        index = torch.randperm(batch_size, device=self.device)
        
        # Mix inputs
        mixed_batch = {}
        for key in ['locations', 'users', 'weekdays', 'start_mins']:
            if key in batch:
                mixed_batch[key] = lam * batch[key] + (1 - lam) * batch[key][index]
                # Round to integers for embeddings
                mixed_batch[key] = mixed_batch[key].round().long()
        
        # Keep other fields
        mixed_batch['mask'] = batch['mask']
        mixed_batch['target'] = batch['target']
        
        # Return mixed batch and mixing parameters
        return mixed_batch, batch['target'][index], lam, index
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_results = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.max_epochs}')
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # Apply mixup if enabled
            if self.use_mixup and np.random.random() < 0.5:
                mixed_batch, targets_b, lam, index = self.mixup_data(batch, self.mixup_alpha)
                
                # Forward pass
                if self.use_auxiliary:
                    logits, aux_logits = self.model(mixed_batch, return_auxiliary=True)
                    # Mixup loss
                    if self.loss_type == 'multitask':
                        loss_a, _, _ = self.criterion(logits, batch['target'], aux_logits, mixed_batch['locations'])
                        loss_b, _, _ = self.criterion(logits, targets_b, aux_logits, mixed_batch['locations'][index])
                    else:
                        loss_a = self.criterion(logits, batch['target'])
                        loss_b = self.criterion(logits, targets_b)
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    logits = self.model(mixed_batch)
                    loss_a = self.criterion(logits, batch['target'])
                    loss_b = self.criterion(logits, targets_b)
                    loss = lam * loss_a + (1 - lam) * loss_b
            else:
                # Standard forward pass
                if self.use_auxiliary:
                    logits, aux_logits = self.model(batch, return_auxiliary=True)
                    if self.loss_type == 'multitask':
                        loss, main_loss, aux_loss = self.criterion(
                            logits, batch['target'], aux_logits, batch['locations']
                        )
                    else:
                        loss = self.criterion(logits, batch['target'])
                else:
                    logits = self.model(batch)
                    loss = self.criterion(logits, batch['target'])
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Track predictions for metrics
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                all_results.append({
                    'predictions': preds.cpu().numpy(),
                    'targets': batch['target'].cpu().numpy()
                })
            
            # Update progress bar
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        
        # Calculate accuracy
        all_preds = np.concatenate([r['predictions'] for r in all_results])
        all_targets = np.concatenate([r['targets'] for r in all_results])
        acc = (all_preds == all_targets).mean() * 100
        
        return avg_loss, acc
    
    @torch.no_grad()
    def evaluate(self, data_loader, desc='Evaluating'):
        """Evaluate model on given data loader."""
        self.model.eval()
        total_loss = 0
        all_logits = []
        all_targets = []
        
        for batch in tqdm(data_loader, desc=desc):
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # Forward pass
            if self.use_auxiliary:
                logits, _ = self.model(batch, return_auxiliary=True)
            else:
                logits = self.model(batch)
            
            # Calculate loss
            if self.loss_type == 'multitask':
                loss, _, _ = self.criterion(logits, batch['target'])
            else:
                loss = self.criterion(logits, batch['target'])
            
            total_loss += loss.item()
            
            all_logits.append(logits.cpu())
            all_targets.append(batch['target'].cpu())
        
        # Concatenate all results
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        metrics_array, true_y, top1 = calculate_correct_total_prediction(all_logits, all_targets)
        
        # Convert array to dict
        metrics_dict = {
            'correct@1': metrics_array[0],
            'correct@3': metrics_array[1],
            'correct@5': metrics_array[2],
            'correct@10': metrics_array[3],
            'rr': metrics_array[4],
            'ndcg': metrics_array[5],
            'f1': 0.0,  # Not calculated
            'total': metrics_array[6]
        }
        
        performance = get_performance_dict(metrics_dict)
        
        avg_loss = total_loss / len(data_loader)
        performance['loss'] = avg_loss
        
        return performance
    
    def save_snapshot(self, epoch, val_metrics):
        """Save snapshot for ensemble."""
        snapshot_path = os.path.join(
            self.checkpoint_dir,
            f'snapshot_epoch_{epoch+1}_acc_{val_metrics["acc@1"]:.2f}.pt'
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': copy.deepcopy(self.model.state_dict()),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metrics': val_metrics
        }, snapshot_path)
        
        self.snapshots.append({
            'path': snapshot_path,
            'epoch': epoch,
            'val_acc': val_metrics['acc@1']
        })
        
        if self.logger:
            self.logger.info(f"Snapshot saved: {snapshot_path}")
    
    def train(self):
        """Main training loop."""
        if self.logger:
            self.logger.info("Starting training...")
        
        for epoch in range(self.max_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Warmup scheduler
            if self.warmup_scheduler and epoch < self.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                # Main scheduler
                self.scheduler.step()
            
            # Validate
            val_metrics = self.evaluate(self.val_loader, desc='Validation')
            
            epoch_time = time.time() - epoch_start
            
            # Log
            if self.logger:
                self.logger.info(
                    f"Epoch {epoch+1}/{self.max_epochs} | "
                    f"Time: {epoch_time:.1f}s | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val Acc@1: {val_metrics['acc@1']:.2f}% | "
                    f"Val Acc@5: {val_metrics['acc@5']:.2f}% | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )
            
            # Save history
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_metrics['loss'])
            self.train_history['val_acc'].append(val_metrics['acc@1'])
            
            # Check improvement
            if val_metrics['acc@1'] > self.best_val_acc:
                self.best_val_acc = val_metrics['acc@1']
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save best model
                best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'train_history': self.train_history
                }, best_path)
                
                if self.logger:
                    self.logger.info(f"✓ New best model! Val Acc@1: {val_metrics['acc@1']:.2f}%")
            else:
                self.patience_counter += 1
                
                if self.logger:
                    self.logger.info(
                        f"  No improvement for {self.patience_counter} epochs "
                        f"(Best: {self.best_val_acc:.2f}% at epoch {self.best_epoch+1})"
                    )
            
            # Snapshot ensemble
            if self.snapshot_ensemble and (epoch + 1) % self.snapshot_interval == 0:
                self.save_snapshot(epoch, val_metrics)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                if self.logger:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model and evaluate on test set
        if self.logger:
            self.logger.info("\nLoading best model for final evaluation...")
        
        best_checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'best_model.pt'))
        self.model.load_state_dict(best_checkpoint['model_state_dict'])
        
        test_metrics = self.evaluate(self.test_loader, desc='Test Set')
        
        if self.logger:
            self.logger.info("\n" + "="*80)
            self.logger.info("FINAL TEST RESULTS:")
            self.logger.info(f"  Test Acc@1:  {test_metrics['acc@1']:.2f}%")
            self.logger.info(f"  Test Acc@5:  {test_metrics['acc@5']:.2f}%")
            self.logger.info(f"  Test Acc@10: {test_metrics['acc@10']:.2f}%")
            self.logger.info(f"  Test MRR:    {test_metrics['mrr']:.2f}%")
            self.logger.info(f"  Test NDCG:   {test_metrics['ndcg']:.2f}%")
            self.logger.info(f"  Best Val Acc@1: {self.best_val_acc:.2f}% (Epoch {self.best_epoch+1})")
            self.logger.info("="*80)
        
        return test_metrics
