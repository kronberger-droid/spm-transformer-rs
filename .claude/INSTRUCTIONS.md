# Instructions for Claude Code

This project uses structured documentation in the `.claude/` folder. **Always read this folder first** when starting a new session.

## Quick Start

1. **Read `.claude/README.md`** - Project overview and current status
2. **Read `.claude/TODOS.md`** - Active tasks and priorities
3. **Read `.claude/PROGRESS.md`** - What's been completed
4. **Read `.claude/EXPERIMENTS.md`** - Training results and analysis

## Current Context (Last updated: 2025-12-03)

**Project**: STM tip state classification using deep learning
**Goal**: 96% AUROC, 85% balanced accuracy (Gordon et al. 2020)

**Current Phase**: Phase 2 - Address Overfitting
**Current Status**:
- Phase 1 complete: 66.5% validation (was 36% baseline) ✅
- Overfitting detected: Train 83%, Valid 66% (17-point gap)
- Architecture: ViT (Vision Transformer, global attention)

**Next Priority**: Add weight decay to reduce overfitting

## Important Files

**Source Code**:
- `src/model.rs` - ScanLineEncoder (ViT architecture)
- `src/data.rs` - Dataset with class weights & normalization
- `src/main.rs` - Training loop with LR scheduler

**Training**:
- `slurm/build.nu` - Build binary with CUDA
- `slurm/train.nu` - Training job script
- Use `cargo run --bin analyze` to analyze checkpoints

**Documentation**:
- All `.claude/*.md` files contain project knowledge
- Update these files as progress is made

## Key Decisions & Context

1. **Architecture**: Currently ViT (no causal masking). Sequential transformer with causal masking planned for Phase 3 if ViT plateaus <80%.

2. **5 Critical Fixes Applied** (Phase 1):
   - Class weights for 43.9% imbalance
   - Per-scanline normalization (mean=0, std=1)
   - LR 1e-4 with 5-epoch warmup
   - Auto-detect 4 classes (not hardcoded 6)
   - 6 transformer layers (was 4)

3. **Overfitting Pattern**: Validation loss best at epoch 27 (1.237), then degraded to 1.444 by epoch 50. Early stopping would help.

4. **Build Workflow**:
   - Build once: `sbatch slurm/build.nu`
   - Train many: `sbatch slurm/train.nu`

## Reminders

- **Always check `.claude/` first** when resuming work
- Update documentation as you make progress
- Results in this project are stored in SLURM logs and checkpoints
- Architecture is still ViT (not sequential yet)
