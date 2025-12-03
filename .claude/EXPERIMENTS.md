# Training Experiments Log

## Experiment Naming Convention

`YYYYMMDD_HHMMSS` or `YYYYMMDD_HHMMSS_jSLURM_JOB_ID`

---

## Baseline Experiments (Before Fixes)

### Experiment: Rust ViT Baseline (Pre-fixes)
**Date**: 2025-12-01 (approximate)
**Location**: `checkpoints/` (old runs)

**Configuration**:
- Model: ScanLineEncoder (ViT)
- Layers: 4
- d_model: 256
- Heads: 8
- LR: 1e-3 (constant, no scheduling)
- Batch size: 32
- Epochs: 50
- Class weights: ❌ NOT used
- Normalization: ❌ None
- Num classes: 6 ❌ (should be 4)

**Results**:
```
Train accuracy: ~36% (epoch 50)
Valid accuracy: ~36% (epoch 50)
Train loss: ~1.60
Valid loss: ~1.60
```

**Analysis**:
- No overfitting (train ≈ valid)
- Model appears stuck/plateaued
- Close to random guessing for 4 classes (25%)
- Multiple critical issues identified

**Issues**:
1. Class weights not used → imbalanced learning
2. No normalization → unstable gradients
3. LR too high → overshooting
4. Wrong num_classes → incorrect output layer
5. Only 4 layers → less capacity than Python (6 layers)

---

### Experiment: Python ViT Reference
**Date**: 2025-11 (approximate)
**Location**: `~/Programming/python/transformer-line-classification/`

**Configuration**:
- Model: Vision Transformer
- Layers: 6
- d_model: 256
- Heads: 8
- FFN dims: 1024
- LR: 1e-4 (warmup + ReduceLROnPlateau)
- Batch size: 32
- Epochs: 54 (early stopping)
- Class weights: ✅ [2.845, 0.569, 1.478, 0.823]
- Normalization: ✅ Per-scanline (mean=0, std=1)
- Augmentation: ✅ Horizontal flip, Gaussian noise, line dropout

**Results**:
```
Best validation accuracy: 39.23% (epoch 39)
Best validation AUROC: 0.6633
Training stopped: epoch 54 (early stopping)
```

**Analysis**:
- Still significantly underperforms target (39% vs 85%)
- Proper training setup but wrong architecture paradigm
- ViT approach may not capture sequential nature of STM scanning

---

## Experiments with Fixes (Phase 1)

### Experiment: Run 1 - All 5 fixes applied 🎉
**Date**: 2025-12-03
**Status**: ✅ Complete

**Configuration**:
- Model: ScanLineEncoder (ViT - global attention, no causal masking)
- Layers: 6 ✅
- d_model: 256
- Heads: 8
- LR: 1e-4 ✅
- LR Schedule: ✅ Linear warmup (1e-6 → 1e-4 over 5 epochs)
- Batch size: 32
- Epochs: 50
- Dropout: 0.1
- Class weights: ✅ [2.845, 0.569, 1.478, 0.823]
- Normalization: ✅ Per-scanline (mean=0, std=1)
- Num classes: ✅ 4 (auto-detected)
- Weight decay: ❌ None (not yet added)

**Results**:
```
=== train directory ===
Total epochs: 50
Best accuracy: 83.1928 (epoch 50)
Best loss: 0.357031 (epoch 50)

=== valid directory ===
Total epochs: 50
Best accuracy: 66.4982 (epoch 49)
Best loss: 1.237079 (epoch 27) ← Validation loss increased after this!
Latest accuracy: 66.1257 (epoch 50)
Latest loss: 1.443985 (epoch 50)
```

**Analysis**:
- ✅ **HUGE SUCCESS!** Validation: 66.5% (vs baseline 36%)
- ✅ **+30 percentage points improvement** from all fixes
- ✅ **Beats Python ViT by 27 points** (66.5% vs 39%)
- ⚠️ **Overfitting**: 17-point train/valid gap (83% vs 66%)
- ⚠️ **Validation loss degraded** after epoch 27 (1.237 → 1.444)
- 📊 Training still improving at epoch 50 (not saturated)

**Key Insights**:
1. All 5 critical fixes worked perfectly
2. Model has capacity to learn (83% train accuracy)
3. **Needs regularization** - overfitting is the bottleneck
4. Optimal stopping point: epoch 27-30
5. Still ViT architecture (not sequential transformer yet)

**Next Steps**:
1. Add weight decay (try 0.01, 0.05)
2. Increase dropout (try 0.15, 0.2)
3. Add early stopping
4. Target: 70-73% validation accuracy

---

## Experiment Template

### Experiment: [Name]
**Date**: YYYY-MM-DD
**Checkpoint**: `checkpoints/[id]/`
**Status**: [Pending / Running / Complete]

**Configuration**:
- Model:
- Layers:
- d_model:
- Heads:
- LR:
- LR Schedule:
- Batch size:
- Epochs:
- Class weights:
- Normalization:
- Other:

**Results**:
```
[Paste analyzer output or key metrics]
```

**Analysis**:
- [What worked well]
- [What didn't work]
- [Unexpected findings]

**Next Steps**:
- [Planned follow-up experiments]

---

## Ablation Studies (Future)

### Planned Ablations

1. **Class Weights Impact**
   - Run with and without class weights
   - Expected: Significant impact on minority classes

2. **Normalization Impact**
   - Run with and without per-scanline normalization
   - Expected: Affects training stability

3. **Learning Rate**
   - Test: 1e-5, 1e-4, 5e-4, 1e-3
   - Expected: 1e-4 optimal

4. **Number of Layers**
   - Test: 4, 6, 8, 12
   - Expected: Diminishing returns after 6-8

5. **Model Size**
   - Test d_model: 128, 256, 512
   - Expected: 256 optimal for dataset size

---

## Notes

- Always use `cargo run --release --bin analyze -- --checkpoints-path checkpoints/[id]/` to analyze results
- Compare train vs valid metrics to detect overfitting
- Track best epoch, not just final epoch
- Save config.json with each checkpoint for reproducibility
