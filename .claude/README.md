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

**Current Status**: Implementing critical fixes to baseline ViT model
- ✅ Class weights implemented
- ✅ Per-scanline normalization implemented
- ✅ Learning rate lowered + warmup scheduler added
- ✅ Number of classes fixed (4 instead of 6)
- 🔄 Next: Increase layers to 6, then test training

**Latest Results**: TBD (pending first training run with fixes)
