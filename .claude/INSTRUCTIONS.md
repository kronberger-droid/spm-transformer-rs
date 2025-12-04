# Instructions for Claude Code

This project uses structured documentation in the `.claude/` folder. **Always read this folder first** when starting a new session.

## Quick Start

1. **Read `.claude/README.md`** - Project overview and current status
2. **Read `.claude/TODOS.md`** - Active tasks and priorities
3. **Read `.claude/PROGRESS.md`** - What's been completed
4. **Read `.claude/EXPERIMENTS.md`** - Training results and analysis

## Current Context (Last updated: 2025-12-03 Evening)

**Project**: STM tip state classification using deep learning
**Goal**: 96% AUROC, 85% balanced accuracy (Gordon et al. 2020)

**Current Phase**: Phase 3 - Sequential Transformer Architecture
**Current Status**:
- Phase 1 complete: 66.5% validation (was 36% baseline) ✅
- Phase 2 complete: ViT optimized to 66.74% ceiling (8-config sweep) ✅
- **Critical finding**: ViT cannot reach 85% target (ceiling at ~67%, 19 points short)
- Architecture: Moving from ViT → Sequential Transformer with causal masking

**Next Priority**: Design and implement sequential transformer with CNN tokenization per scanline

## Important Files

**Source Code**:
- `src/model.rs` - ScanLineEncoder (ViT architecture)
- `src/data.rs` - Dataset with class weights & normalization
- `src/main.rs` - Training loop with LR scheduler

**Training**:
- `slurm/build.nu` - Build binary with CUDA
- `slurm/train.nu` - Training job script (single run)
- `slurm/sweep.nu` - Hyperparameter sweep (8 parallel jobs)
- Use `cargo run --bin analyze -- --checkpoints-path <path>` to analyze checkpoints

**Documentation**:
- All `.claude/*.md` files contain project knowledge
- Update these files as progress is made

## Key Decisions & Context

1. **Architecture Decision (Phase 3)**:
   - **Phase 1-2**: ViT (Vision Transformer, global attention)
   - **ViT Result**: Optimized to 66.74% validation ceiling (cannot reach 85% target)
   - **Phase 3 Decision**: Moving to Sequential Transformer with causal masking
   - **Rationale**: STM scans are inherently sequential (scanline-by-scanline). ViT's global attention doesn't match data structure.

2. **Phase 1: 5 Critical Fixes** (36% → 66.5% validation):
   - Class weights for imbalanced classes (6 classes in dataset)
   - Per-scanline normalization (mean=0, std=1)
   - LR 1e-4 with 5-epoch warmup
   - Auto-detect num_classes from data
   - 6 transformer layers (was 4)

3. **Phase 2: Hyperparameter Sweep Results**:
   - 8 configs tested: weight_decay [0.0, 0.01, 0.05, 0.1] × dropout [0.1, 0.15, 0.2]
   - **Best**: No weight decay, dropout=0.1 → 66.74% validation
   - **Finding**: Regularization reduces capacity more than it helps overfitting
   - **Conclusion**: ViT ceiling at ~67%, need architectural change

4. **Build Workflow**:
   - Build once: `sbatch slurm/build.nu`
   - Single run: `sbatch slurm/train.nu`
   - Sweep: `sbatch slurm/sweep.nu` (8 parallel jobs)

## Reminders

- **Always check `.claude/` first** when resuming work
- Update documentation as you make progress
- Results in this project are stored in SLURM logs and checkpoints
- **Phase 2 complete**: ViT optimized, now moving to Sequential Transformer (Phase 3)
- All hyperparameters now support environment variables (see `src/main.rs` Args struct)
- Early stopping implemented (default patience: 10 epochs)
