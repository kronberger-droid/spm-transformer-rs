# Progress Tracker

Last updated: 2025-12-03

## Current Phase: Phase 3 - Sequential Transformer Architecture

**Decision Rationale**: Phase 2 hyperparameter sweep revealed ViT architecture has hit ceiling at ~67% validation accuracy. Need 85% to match Gordon et al. 2020. Sequential transformer with causal masking is required.

---

## Phase 2: Address Overfitting ✅ COMPLETE

### Completed ✅

#### AdamW Optimizer + Weight Decay
**Status**: ✅ Complete (2025-12-03)
**Changes**:
- Replaced Adam with AdamW optimizer
- Added `weight_decay` CLI argument (env: `WEIGHT_DECAY`, default: 0.01)
- Tested with weight_decay=0.01

**Files Modified**:
- `src/main.rs`: Changed `AdamConfig` to `AdamWConfig`, added weight_decay parameter

**Result**: No improvement over baseline (66.5% → 66.1% valid accuracy)

---

#### Environment Variable Support for All Hyperparameters
**Status**: ✅ Complete (2025-12-03)
**Changes**:
- Added `env` attribute to all CLI arguments
- Enables clean hyperparameter sweeps via environment variables
- Environment variables: `LEARNING_RATE`, `WARMUP_EPOCHS`, `BATCH_SIZE`, `NUM_EPOCHS`, `WEIGHT_DECAY`, `DROPOUT`, `D_MODEL`, `NUM_HEADS`, `NUM_LAYERS`, `TRAIN_RATIO`, `VAL_RATIO`

**Files Modified**:
- `src/main.rs`: Added `env` to all `#[arg()]` attributes

**Impact**: Clean sweep scripts, better experiment tracking

---

#### Early Stopping Implementation
**Status**: ✅ Complete (2025-12-03)
**Changes**:
- Implemented `MetricEarlyStoppingStrategy` monitoring validation loss
- Added `early_stopping_patience` CLI argument (env: `EARLY_STOPPING_PATIENCE`, default: 10)
- Stops training when validation loss doesn't improve for N epochs

**Files Modified**:
- `src/main.rs`: Added early stopping imports and configuration

**Expected Impact**: Saves 30-60% compute, catches best model automatically

---

#### Hyperparameter Sweep (8 Experiments)
**Status**: ✅ Complete (2025-12-03)
**Sweep Grid**:
- Weight decay: [0.0, 0.01, 0.05, 0.1]
- Dropout: [0.1, 0.15, 0.2]
- 8 configurations tested in parallel

**Files Created**:
- `slurm/sweep.nu`: SLURM job array script for parallel hyperparameter testing

**Results Summary** (see EXPERIMENTS.md for full details):

| Config | WD | Drop | Best Valid | Train/Valid Gap | Best Epoch |
|--------|-----|------|------------|----------------|------------|
| no_wd (best) | 0.0 | 0.1 | 66.74% | 18.99 | 48 |
| baseline | 0.01 | 0.1 | 66.06% | 18.14 | 44 |
| wd05_drop10 | 0.05 | 0.1 | 65.59% | 18.42 | 48 |
| wd10_drop15 | 0.1 | 0.15 | 65.07% | 14.21 | 47 |
| wd01_drop15 | 0.01 | 0.15 | 64.86% | 14.91 | 47 |
| wd05_drop15 | 0.05 | 0.15 | 64.51% | 14.52 | 50 |
| wd01_drop20 | 0.01 | 0.2 | 63.25% | 11.52 | 49 |
| wd05_drop20 | 0.05 | 0.2 | 60.54% | 13.66 | 46 |

**Key Findings**:
1. **ViT ceiling: ~66-67% validation accuracy** - Cannot reach 85% target
2. **No weight decay performs best** - Regularization reduces capacity more than overfitting
3. **Strong regularization is counterproductive** - Higher dropout/weight_decay → lower accuracy
4. **Architecture is the bottleneck** - All configs cluster around 60-67%, 19 points below target
5. **Validation degradation persists** - All configs show loss increase after epoch 20-48

**Conclusion**: ViT with global attention is fundamentally wrong for sequential scanline data. Moving to Phase 3 (Sequential Transformer) is required to reach 85% target.

---

## Phase 1: Baseline ViT Fixes ✅ COMPLETE

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

## Phase 3: Sequential Transformer Architecture (CURRENT)

### Status: Planning & Design

**Goal**: Achieve 85%+ validation accuracy to match Gordon et al. 2020

**See `ARCHITECTURE.md` and `TODOS.md` for detailed implementation plan.**

---

## Experiments Log

**See `EXPERIMENTS.md` for full detailed results and analysis.**

### Phase 0: Baseline (Before Fixes)
- **Rust ViT**: 36% accuracy, 1.60 loss (epoch 50)
- **Python ViT**: 39% accuracy, 0.66 AUROC (epoch 39)

### Phase 1: With All 5 Fixes ✅
- **Run 1**: 66.5% validation accuracy (epoch 49), 83.2% train accuracy
- **Improvement**: +30 points over baseline, +27 points over Python
- **Issue**: Overfitting (17-point gap)

### Phase 2: Hyperparameter Sweep (8 Experiments) ✅
- **Best**: 66.74% validation (no weight decay, dropout=0.1)
- **Range**: 60.5% - 66.7% across all configs
- **Conclusion**: ViT ceiling at ~67%, cannot reach 85% target

---

## Known Issues

None currently blocking.

---

## Notes

- Removed `TrainingConfig` struct (unused, warnings only)
- `is_empty()` method unused (warnings only)
- All critical paths compile and pass `cargo check`
