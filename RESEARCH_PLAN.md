# Advanced Research Plan: Pushing Beyond 50% Acc@1 on GeoLife

## Current Status
- **Baseline**: 37.95% Test Acc@1 (model_v2: 88d, 4 layers, 481K params)
- **Goal**: >50% Test Acc@1 with <500K parameters
- **Minimum requirement**: 40% Test Acc@1

## Critical Dataset Insights
1. **Severe class imbalance**: Top location appears 10.55% of time, 964/1093 targets appear <5 times
2. **Limited training data**: 7,424 samples for 1,187-class problem (~6.25 samples/class)
3. **Val-Test gap**: 5-8% consistent gap suggests distribution shift
4. **Sequence length**: Mean ~18, max 51 (current max_len=50 is appropriate)

## Research Strategy: Multi-Pronged Approach

### Phase 1: Advanced Architecture Improvements (Immediate)
1. **Hierarchical Attention**: Multi-scale temporal modeling
2. **Location Embeddings Enhancement**: 
   - Pre-trained location embeddings (spatial clustering)
   - Learnable positional bias for frequent locations
3. **Auxiliary Tasks**: Multi-task learning with trajectory reconstruction
4. **Memory-Augmented Architecture**: External memory for location patterns

### Phase 2: Advanced Training Techniques
1. **Focal Loss**: Address extreme class imbalance
2. **Class-Balanced Sampling**: Oversample rare locations
3. **Mixup/CutMix for Sequences**: Data augmentation
4. **Self-Distillation**: Knowledge distillation from ensemble of same model
5. **Curriculum Learning**: Easy→hard sequence ordering

### Phase 3: Feature Engineering & Representation
1. **Temporal Encoding Enhancement**: Cyclic encodings for time
2. **Location Graph Features**: Encode spatial relationships
3. **User Behavior Patterns**: Statistical features from user history
4. **Trajectory Smoothing**: Noise reduction preprocessing

### Phase 4: Ensemble & Meta-Learning
1. **Snapshot Ensembles**: Multiple checkpoints from single training
2. **Monte Carlo Dropout**: Test-time uncertainty estimation
3. **Stacking**: Combine predictions from different model variants

## Implementation Priority Queue

### Week 1 (Days 1-2): Quick Wins
- [ ] Focal loss implementation
- [ ] Enhanced temporal features (cyclic encodings)
- [ ] Improved location embeddings with clustering
- [ ] Class-balanced sampling

### Week 1 (Days 3-5): Architecture Enhancements
- [ ] Multi-head with hierarchical attention
- [ ] Auxiliary reconstruction task
- [ ] Memory-augmented module
- [ ] Location-aware attention bias

### Week 1 (Days 6-7): Advanced Training
- [ ] Mixup for sequences
- [ ] Curriculum learning
- [ ] Snapshot ensembles
- [ ] Self-distillation

### Week 2: Fine-tuning & Optimization
- [ ] Hyperparameter optimization (learning rate, warmup, etc.)
- [ ] Advanced regularization (DropConnect, Stochastic Depth)
- [ ] Test-time augmentation
- [ ] Final ensemble optimization

## Expected Impact Analysis

| Technique | Expected Gain | Complexity | Priority |
|-----------|---------------|------------|----------|
| Focal Loss | +2-3% | Low | HIGH |
| Cyclic Time Encoding | +1-2% | Low | HIGH |
| Class-Balanced Sampling | +1-2% | Low | HIGH |
| Enhanced Embeddings | +2-3% | Medium | HIGH |
| Hierarchical Attention | +2-4% | Medium | MEDIUM |
| Auxiliary Tasks | +1-2% | Medium | MEDIUM |
| Mixup/CutMix | +1-3% | Medium | MEDIUM |
| Snapshot Ensemble | +2-4% | Low | HIGH |
| Memory Module | +2-3% | High | LOW |

**Projected Total**: 38% + 12-18% = **50-56% Test Acc@1**

## Technical Constraints
- Parameter budget: <500,000 (strictly enforced)
- No data leakage or test set contamination
- All improvements must be theoretically sound and reproducible
- Focus on model intelligence, not shortcuts

## Success Metrics
1. Primary: Test Acc@1 > 50%
2. Minimum: Test Acc@1 >= 40%
3. Secondary: Reduce Val-Test gap to <3%
4. Tertiary: Improve Acc@5, MRR, NDCG

## Documentation & Tracking
- Git commits for each major change
- Detailed experiment logs
- Ablation studies for each technique
- Final report with analysis
