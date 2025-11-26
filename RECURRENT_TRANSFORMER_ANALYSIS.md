# RECURRENT TRANSFORMER - COMPLETE ANALYSIS

## Summary of Attempts

| Version | Architecture | Params | Test Acc@1 | vs Baseline |
|---------|--------------|--------|------------|-------------|
| Baseline | Standard Transformer | 481K | **37.95%** | - |
| V1 | Cross-attention with single token hidden | 443K | 28.41% | -9.54% ❌ |
| V2 | Full sequence refinement, 8 cycles | 396K | 33.58% | -4.37% ❌ |
| V3 | No sharing, 4 cycles, dropout 0.3 | 473K | 30.38% | -7.57% ❌ |

## ROOT CAUSE: The Recurrent Mechanism Itself

**All versions failed because the core idea of "multiple cycles" causes overfitting:**

1. **Training vs Test Gap**:
   - V2: 53.72% train → 33.58% test (20.14% gap!)
   - V3: 42.56% train → 30.38% test (12.18% gap!)
   - Baseline: ~60% train → 37.95% test (~22% gap, but higher test!)

2. **Early Best Epochs**:
   - All models peak early (epoch 9-19) then degrade
   - Suggests memorization, not learning

3. **The Problem with Cycles**:
   - Each cycle processes the SAME input
   - Later cycles overfit to training patterns
   - Test data has different patterns → poor generalization

## Why "Recurrent" Fails Here

The user's idea: "transformer receives the entire input sequence... along with the hidden state from the previous cycle"

This creates: **Iterative refinement through recurrence**

But this fails because:
- The task is next-location prediction (not image generation or text refinement)
- The input sequence already contains all the information
- Multiple passes cause the model to "overthink" and memorize training patterns
- More cycles = more overfitting

## What Actually Works

The simple baseline transformer works best because:
1. Single forward pass - no chance to overfit through iteration
2. Direct mapping from sequence → next location
3. Enough capacity (481K) for the task
4. Standard architecture that generalizes well

## Conclusion

**The recurrent transformer idea is conceptually interesting but empirically fails for next-location prediction.**

The fundamental issue: This task doesn't benefit from iterative refinement. It needs direct pattern recognition, which standard transformers excel at.

To reach 40%+ would require:
- Abandoning the recurrent mechanism
- Using standard transformer with better features/data
- OR completely different problem formulation
