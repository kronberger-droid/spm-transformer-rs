# STM Tip State Classification

Rust implementation of a deep learning model for classifying STM (Scanning Tunneling Microscopy) tip states from scan line images.

## Goal

Replicate the performance of **Gordon et al. 2020** ("Embedding Human Heuristics in Machine Learning for Probe Microscopy"):
- **Target AUROC**: 0.960
- **Target Balanced Accuracy**: 0.847

## Quick Start

```bash
# Build
cargo build --release

# Train (local)
cargo run --release -- --data-path data/processed_data.npz --num-epochs 10

# Train (cluster with CUDA)
cargo build --release --features cuda
sbatch slurm/train.nu

# Analyze results
cargo run --release --bin analyze -- --checkpoints-path checkpoints/[run_id]/
```

## Project Structure

```
.
├── src/
│   ├── main.rs          # Training entry point
│   ├── model.rs         # ScanLineEncoder (ViT) architecture
│   ├── data.rs          # Dataset loading and preprocessing
│   └── training.rs      # Training configuration
├── .claude/             # 📚 Project documentation
│   ├── README.md        # Documentation overview
│   ├── PROGRESS.md      # Current status and completed work
│   ├── ARCHITECTURE.md  # Architecture plans and comparisons
│   ├── EXPERIMENTS.md   # Training experiment logs
│   ├── TODOS.md         # Active tasks and blockers
│   └── REFERENCE.md     # Target paper and benchmarks
├── checkpoints/         # Training checkpoints
├── data/               # Dataset (NPZ format)
└── slurm/              # Cluster job scripts
```

## Documentation

**See `.claude/` folder for detailed documentation:**

- **[.claude/README.md](.claude/README.md)** - Start here for project overview
- **[.claude/PROGRESS.md](.claude/PROGRESS.md)** - Current status and completed fixes
- **[.claude/ARCHITECTURE.md](.claude/ARCHITECTURE.md)** - Architecture designs and plans
- **[.claude/EXPERIMENTS.md](.claude/EXPERIMENTS.md)** - Experiment logs and results
- **[.claude/TODOS.md](.claude/TODOS.md)** - Active tasks and next steps
- **[.claude/REFERENCE.md](.claude/REFERENCE.md)** - Target paper details

## Current Status

**Phase**: Baseline ViT with critical fixes

**Recent Progress**:
- ✅ Implemented class weights for imbalanced dataset
- ✅ Added per-scanline normalization (mean=0, std=1)
- ✅ Lowered learning rate (1e-3 → 1e-4) with warmup scheduler
- ✅ Fixed number of classes (auto-detected from data)
- 🔄 Next: Increase layers to 6, test training

**Latest Results**: Pending first training run with fixes

## Model Architecture

**Current**: Vision Transformer (ViT)
- Scanline-as-token approach (128 scanlines → 128 tokens)
- 6 transformer encoder layers
- 8 attention heads, 256 embedding dims
- Class weights, per-scanline normalization
- Linear LR warmup (5 epochs)

**Planned**: Sequential Encoder-Only Transformer (see `.claude/ARCHITECTURE.md`)

## Dataset

- **Format**: 128×128 STM images (128 scanlines × 128 pixels/line)
- **Classes**: 4 tip states
  1. Individual atoms (8.79%)
  2. Asymmetries/Dimers (43.90%) ← majority
  3. Rows (16.91%)
  4. Defects (30.39%)
- **Class weights**: [2.845, 0.569, 1.478, 0.823]

## Requirements

- Rust 1.70+
- Burn ML framework
- CUDA (optional, for GPU training)

See `flake.nix` for complete development environment.

## License

[Add license here]

## References

Gordon, O. M., et al. (2020). "Embedding Human Heuristics in Machine Learning for Probe Microscopy." *[Journal]*. AUROC: 0.960, Balanced Accuracy: 0.847.
