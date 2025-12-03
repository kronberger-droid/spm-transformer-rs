# Progress Tracker

Last updated: 2025-12-02

## Current Phase: Baseline ViT Fixes

### Completed ✅

#### Fix #1: Class Weights Implementation
**Status**: ✅ Complete
**Changes**:
- Added `num_classes` and `class_weights` fields to `STMDataset` struct
- Computed class weights once at data load time (inverse frequency)
- Prints class distribution on load
- Stored in model and passed to `CrossEntropyLoss`
- Removed `use_class_weights` CLI flag (always enabled)

**Files Modified**:
- `src/data.rs`: Added `compute_num_classes()` and `compute_weights()` helpers
- `src/model.rs`: Added `class_weights: Option<Vec<f32>>` field, updated `init()` signature
- `src/main.rs`: Pass weights from dataset to model

**Expected Impact**: High - critical for handling 43.9% class imbalance

---

#### Fix #2: Per-Scanline Normalization
**Status**: ✅ Complete
**Changes**:
- Added `normalize_per_scanline()` method in `STMDataset`
- Normalizes each scanline to mean=0, std=1
- Applied once at data load time
- Uses epsilon=1e-8 to prevent division by zero

**Files Modified**:
- `src/data.rs`: Added normalization in `from_npz()`

**Expected Impact**: High - stabilizes training, removes brightness variations

---

#### Fix #3: Learning Rate + Scheduling
**Status**: ✅ Complete
**Changes**:
- Lowered base LR from 1e-3 to 1e-4 (10× reduction)
- Added `LinearLrScheduler` with 5-epoch warmup
- Warmup: 1e-6 → 1e-4 over first 5 epochs
- Added `warmup_epochs` CLI arg (default: 5)
- Fixed borrow checker issue (store `train_dataset_len` before move)

**Files Modified**:
- `src/main.rs`: Added LR scheduler setup, updated imports

**Expected Impact**: High - prevents overshooting, enables fine-tuning

---

#### Fix #4: Number of Classes
**Status**: ✅ Complete (via Fix #1)
**Changes**:
- Model now infers `num_classes` from dataset
- No longer hardcoded to 6

**Expected Impact**: Critical - correctness issue fixed

---

### In Progress 🔄

None - all critical fixes completed!

---

#### Fix #5: Increase Transformer Layers
**Status**: ✅ Complete
**Changes**: Updated default from 4 to 6 layers in `src/main.rs`
**Impact**: Medium - more model capacity

---

### First Training Results 🎉

**Validation Accuracy**: 66.5% (epoch 49)
**Training Accuracy**: 83.2% (epoch 50)

**Analysis**:
- ✅ **Massive improvement**: 36% → 66.5% (+30 points!)
- ✅ **Beats Python baseline**: 39% → 66.5% (+27 points!)
- ⚠️ **Overfitting detected**: 17-point train/valid gap
- ⚠️ **Validation loss increased** after epoch 27 (best: 1.237 @ epoch 27, latest: 1.444 @ epoch 50)

**Conclusion**: All 5 fixes worked! Model is learning well but overfitting. Need regularization.

---

### Next Steps 📋

**Phase 2a: Address Overfitting** (Target: 70-73% validation)
1. Add weight decay to optimizer (0.01-0.05)
2. Tune dropout (try 0.15, 0.2)
3. Add early stopping (stop at epoch ~27-30)
4. Add ReduceLROnPlateau after warmup

**Phase 2b: If Still < 75%**
5. Add online data augmentation
6. Increase model capacity (d_model=512, layers=8)
7. Longer training with early stopping

**Phase 3: Sequential Transformer** (If ViT plateaus < 80%)
- Implement causal masking
- Add CNN tokenization per scanline
- Target: 85%+ validation accuracy (match Gordon et al. 2020)

---

## Future Work

### Phase 2: Incremental Improvements (if ViT shows promise)
- Add weight decay to Adam optimizer
- Add gradient clipping
- Implement early stopping
- Add data augmentation during training

### Phase 3: Sequential Transformer Architecture (if ViT plateaus)
See `ARCHITECTURE.md` for detailed plan.

---

## Experiments Log

See `EXPERIMENTS.md` for detailed results.

### Baseline (Before Fixes)
- **Rust ViT**: 36% accuracy, 1.60 loss (epoch 50)
- **Python ViT**: 39% accuracy, 0.66 AUROC (epoch 39)

### With All 5 Fixes ✅
- **Run 1**: 66.5% validation accuracy (epoch 49), 83.2% train accuracy
- **Improvement**: +30 points over baseline, +27 points over Python
- **Issue**: Overfitting (17-point gap)

---

## Known Issues

None currently blocking.

---

## Notes

- Removed `TrainingConfig` struct (unused, warnings only)
- `is_empty()` method unused (warnings only)
- All critical paths compile and pass `cargo check`
