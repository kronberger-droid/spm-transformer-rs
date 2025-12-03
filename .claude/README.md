# Project Documentation

This folder contains project documentation, progress tracking, and planning documents.

## Files

- **`PROGRESS.md`** - Current status, completed work, and what's next
- **`ARCHITECTURE.md`** - Detailed architectural plans and comparisons
- **`EXPERIMENTS.md`** - Training experiments and results
- **`TODOS.md`** - Active task list and blockers
- **`REFERENCE.md`** - Target paper details and benchmark metrics

## Quick Status

**Goal**: Replicate Gordon et al. 2020 performance (96.0% AUROC, 84.7% balanced accuracy)

**Phase 1: Critical Fixes** ✅ COMPLETE
- ✅ All 5 fixes implemented and tested
- ✅ **Result**: 66.5% validation accuracy (was 36% baseline)
- ✅ **+30 percentage points improvement!**
- ✅ Beats Python ViT by 27 points (66.5% vs 39%)

**Phase 2: Address Overfitting** 🔄 CURRENT
- **Problem**: Train 83%, Valid 66% (17-point gap)
- **Next**: Add weight decay, tune dropout, early stopping
- **Target**: 70-73% validation accuracy

**Latest Results** (2025-12-03):
- Validation: 66.5% accuracy, 1.24 loss (best @ epoch 27)
- Training: 83.2% accuracy, 0.36 loss (epoch 50)
- Overfitting after epoch 27 → need regularization
