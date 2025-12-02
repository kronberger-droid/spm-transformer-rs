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

#### Fix #5: Increase Transformer Layers
**Status**: 🔄 Ready to implement
**Required Change**:
```rust
// src/main.rs, line ~61
#[arg(long, default_value_t = 6)]  // Change from 4 to 6
num_layers: usize,
```

**Expected Impact**: Medium - more model capacity

**Blockers**: None

---

### Next Steps 📋

1. **Immediate**: Increase layers to 6
2. **Test Training**: Run 5-10 epoch experiment with all fixes
3. **Evaluate Results**:
   - If ~50-60% accuracy → proceed to architectural changes
   - If <45% accuracy → debug
   - If >60% accuracy → continue incremental improvements

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

### With Fixes
- **Run 1**: TBD

---

## Known Issues

None currently blocking.

---

## Notes

- Removed `TrainingConfig` struct (unused, warnings only)
- `is_empty()` method unused (warnings only)
- All critical paths compile and pass `cargo check`
