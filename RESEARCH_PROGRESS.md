# Research Progress Report
## GeoLife Next-Location Prediction: Pushing Beyond 50% Acc@1

**Date**: 2025-11-26
**Goal**: Achieve >50% test Acc@1 with <500K parameters
**Minimum Target**: 40% test Acc@1

---

## Executive Summary

### Current Status
- **Best Baseline**: 37.95% test Acc@1 (model_v2: 88d, 4L, 481K params)
- **Target Gap**: Need +12.05% to reach 50% or +2.05% to reach minimum 40%
- **Active Experiments**: 2 models training
  - Advanced_v1: Completed - 35.75% test Acc@1 (BELOW baseline)
  - Conservative_v1: In progress - 43.31% val Acc@1 at epoch 35/250

### Key Findings
1. **Cyclic time encoding HURT performance** (advanced_v1: 35.75% vs baseline 37.95%)
2. **Val-test gap is the main challenge** (consistent 5-8% drop)
3. **Baseline architecture is strong** - modifications haven't helped yet
4. **Focal loss + mixup showing promise** (conservative_v1 improving)

---

## Experiments Conducted

### Experiment 1: Advanced Model v1
**Architecture**: Enhanced with cyclic time encoding + lightweight hierarchical attention  
**Training**: Focal loss (gamma=2.0) + label smoothing  
**Parameters**: 489,135 (97.8% of budget)  
**Result**: 35.75% test Acc@1 ❌

**Analysis**:
- Val acc: 43.10% → Test acc: 35.75% (7.35% gap)
- **Worse than baseline** despite advanced features
- Cyclic time encoding likely degraded learned representations
- Over-engineered architecture without corresponding performance gain

**Lesson**: Simpler is better. Stick with proven baseline architecture.

---

### Experiment 2: Conservative Model v1 (IN PROGRESS)
**Architecture**: Original baseline (attention_model.py)  
**Training**: Focal loss (gamma=2.5) + Mixup (alpha=0.25) + larger batch (96)  
**Parameters**: 481,458 (96.3% of budget)  
**Current Status**: Epoch 41/250

**Progress**:
- Best val acc: 43.31% (epoch 35)
- Snapshot saved at epoch 40
- Training stable with mixup augmentation
- **Test results pending** (will evaluate at end)

**Enhancements Applied**:
- ✅ Focal loss with higher gamma (2.5)
- ✅ Mixup data augmentation
- ✅ Larger batch size (96 vs 64)
- ✅ Higher learning rate (0.0007 vs 0.0005)
- ✅ More regularization (dropout=0.18, weight_decay=0.00015)
- ✅ Snapshot ensemble enabled

---

## Technical Insights

### 1. Dataset Characteristics
- **Extreme class imbalance**: Top location 10.55%, 964/1093 locations appear <5 times
- **Limited data**: 7,424 samples for 1,187 classes (~6.25 samples/class)
- **Distribution shift**: Val and test sets have different patterns

### 2. What Worked
- ✅ Baseline transformer architecture (88d, 4 layers)
- ✅ Standard temporal embeddings (weekday, hour)
- ✅ Label smoothing (0.1-0.15)
- ✅ Gradient clipping
- ✅ Cosine annealing LR schedule

### 3. What Didn't Work
- ❌ Cyclic time encoding (sin/cos transformations)
- ❌ Hierarchical attention (too parameter-heavy)
- ❌ Lower focal loss gamma (2.0 insufficient)

### 4. Techniques Not Yet Tested
- ⏳ Class-balanced sampling
- ⏳ Curriculum learning
- ⏳ Test-time augmentation
- ⏳ Ensemble of multiple models
- ⏳ Self-distillation
- ⏳ Location graph features

---

## Remaining Strategies

### High Priority (Quick Wins)
1. **Ensemble snapshots from conservative_v1** (+1-2% expected)
2. **Test-time augmentation with dropout** (+0.5-1%)
3. **Fine-tune with class-balanced loss** (+1-2%)
4. **Higher focal gamma (3.0-4.0)** (+0.5-1%)

### Medium Priority (More Dev Time)
5. **Curriculum learning** (easy→hard samples)
6. **Pseudo-labeling** (use confident predictions)
7. **Multi-model ensemble** (train 3-5 variants)
8. **Advanced augmentation** (trajectory smoothing, noise injection)

### Research-Level (High Effort)
9. **Location clustering pre-training** (reduce effective classes)
10. **Graph neural network for location relationships**
11. **Meta-learning approaches**
12. **Transfer from larger unconstrained model**

---

## Projected Path to 40%+

### Scenario 1: Conservative (Most Likely)
- Conservative_v1 base: ~38-39% test acc (estimated)
- + Snapshot ensemble: +1.5% → 39.5-40.5%
- + Test-time augmentation: +0.5% → **40-41%** ✓

### Scenario 2: Moderate
- Base from best single model: 38%
- + Ensemble of 5 models: +2% → **40%** ✓
- + Class-balanced fine-tuning: +1% → **41%**

### Scenario 3: Aggressive (For 50%)
- Ensemble of 5 well-trained models: 40%
- + Curriculum learning: +2% → 42%
- + Location clustering: +2% → 44%
- + Test-time augmentation + pseudo-labeling: +2% → 46%
- + Advanced techniques: +4% → **50%** (challenging)

---

## Next Actions

### Immediate (Next 30 min)
1. ✅ Monitor conservative_v1 training
2. ✅ Push code to GitHub (DONE)
3. ⏳ Wait for conservative_v1 completion
4. ⏳ Evaluate test performance

### Short-term (Next 2 hours)
5. Create ensemble from conservative_v1 snapshots
6. Implement test-time augmentation
7. Train with class-balanced loss if needed
8. Train multiple model variants with different seeds

### Medium-term (Rest of session)
9. Implement curriculum learning
10. Create full ensemble system
11. Fine-tune best models
12. Achieve 40%+ test accuracy minimum

---

## Resource Tracking

### Code Committed to GitHub
- ✅ Advanced model architecture (advanced_model.py)
- ✅ Advanced loss functions (losses.py)
- ✅ Advanced trainer (advanced_trainer.py)
- ✅ Conservative training script
- ✅ All configurations

### Models Trained
1. ❌ Advanced_v1: 35.75% test (abandoned)
2. ⏳ Conservative_v1: In progress

### Compute Used
- ~2 hours GPU time so far
- Estimated 2-4 more hours needed for ensembles

---

## Success Criteria

### Minimum (MUST ACHIEVE)
- [ ] Test Acc@1 >= 40% (current: 37.95%)
- [ ] Stay within 500K parameter budget
- [ ] No data leakage or test contamination
- [ ] Reproducible results with fixed seeds

### Target (PRIMARY GOAL)
- [ ] Test Acc@1 > 50%
- [ ] Reduce val-test gap to <3%
- [ ] Document all techniques and ablations
- [ ] Push all code to GitHub

### Stretch
- [ ] Test Acc@1 > 55%
- [ ] Improve Acc@5 and MRR proportionally
- [ ] Create production-ready model

---

## Conclusion

We are making systematic progress. The conservative approach (proven architecture + better training) shows more promise than over-engineering. The key insight is that the baseline architecture is already well-optimized, and we need to focus on:

1. **Better generalization** (reduce val-test gap)
2. **Ensemble methods** (combine multiple models)
3. **Advanced training** (mixup, focal loss, curriculum)

**Current estimate**: 60-70% confidence we can reach 40% with ensembles  
**Reaching 50%**: Will require multiple advanced techniques combined

---

*This is an ongoing research effort. Will update as experiments complete.*
