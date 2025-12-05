#!/usr/bin/env nu
#
#SBATCH --job-name=spm-rust-transformer
#SBATCH --output=/share/rusty-tip/logs/train_rust_%j.out
#SBATCH --error=/share/rusty-tip/logs/train_rust_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=a100:1
#SBATCH --partition=GPU-a100

# =================
# Environment Setup
# =================

# Host paths
let code_dir = $"($env.HOME)/rust/spm-transformer-rs"
let container = "/share/rusty-tip/apptainer/stm-transformer.sif"
let data_dir = "/share/rusty-tip/data"

# Container paths (used inside apptainer after bind mounts)
let container_data = "/data"
let container_app = "/app"

# Environment variables passed to container (must use container paths)
$env.DATA_PATH = $"($container_data)/processed_data.npz"
$env.CHECKPOINT_BASE_DIR = $"($container_data)/checkpoints"

# Training hyperparameters (via environment variables)
$env.NUM_EPOCHS = "50"
$env.BATCH_SIZE = "32"
$env.LEARNING_RATE = "0.0001"
$env.WARMUP_EPOCHS = "5"
$env.WEIGHT_DECAY = "0.01"
$env.DROPOUT = "0.1"

# Model selection
$env.MODEL = "vit"  # Options: "vit" or "sequential"

# Model architecture
$env.D_MODEL = "256"
$env.NUM_HEADS = "8"
$env.NUM_LAYERS = "6"

# Enter project directory
cd $code_dir

# Print useful information
print $"
Job Information:
  Job ID: ($env.SLURM_JOB_ID)
  Job Name: ($env.SLURM_JOB_NAME)
  Node: ($env.SLURM_NODELIST)
  Working Directory: ($env.PWD)
  Start Time: (date now | format date '%Y-%m-%d %H:%M:%S')

System Information:
  CPUs allocated: ($env.SLURM_CPUS_PER_TASK)
  Memory allocated: ($env.SLURM_MEM_PER_NODE)MB
  GPU: ($env.CUDA_VISIBLE_DEVICES)
"

# ===========
# Training
# ===========

# Check binary exists (should be built with slurm/build.nu first)
if not (($"($code_dir)/target/release/spm-transformer" | path exists)) {
  print "Error: Binary not found!"
  print "Please run: sbatch slurm/build.nu first"
  exit 1
}

print $"\nStarting training with configuration:
  Model type: ($env.MODEL)
  Epochs: ($env.NUM_EPOCHS)
  Batch size: ($env.BATCH_SIZE)
  Learning rate: ($env.LEARNING_RATE)
  Warmup epochs: ($env.WARMUP_EPOCHS)
  Weight decay: ($env.WEIGHT_DECAY)
  Dropout: ($env.DROPOUT)
  Architecture: d_model=($env.D_MODEL), heads=($env.NUM_HEADS), layers=($env.NUM_LAYERS)
  Data: ($env.DATA_PATH)
  Checkpoint base: ($env.CHECKPOINT_BASE_DIR)
"

# Build apptainer command
let args = [
  # Apptainer options
  "exec" "--nv"

  # Bind mounts (host:container)
  "--bind" $"($code_dir):($container_app)"
  "--bind" $"($data_dir):($container_data)"

  # Environment variables - all hyperparameters passed via env
  "--env" $"DATA_PATH=($env.DATA_PATH)"
  "--env" $"CHECKPOINT_BASE_DIR=($env.CHECKPOINT_BASE_DIR)"
  "--env" $"SLURM_JOB_ID=($env.SLURM_JOB_ID)"
  "--env" $"MODEL=($env.MODEL)"
  "--env" $"NUM_EPOCHS=($env.NUM_EPOCHS)"
  "--env" $"BATCH_SIZE=($env.BATCH_SIZE)"
  "--env" $"LEARNING_RATE=($env.LEARNING_RATE)"
  "--env" $"WARMUP_EPOCHS=($env.WARMUP_EPOCHS)"
  "--env" $"WEIGHT_DECAY=($env.WEIGHT_DECAY)"
  "--env" $"DROPOUT=($env.DROPOUT)"
  "--env" $"D_MODEL=($env.D_MODEL)"
  "--env" $"NUM_HEADS=($env.NUM_HEADS)"
  "--env" $"NUM_LAYERS=($env.NUM_LAYERS)"
  "--env" $"NUM_WORKERS=($env.SLURM_CPUS_PER_TASK)"

  # Container and binary (no CLI args needed - all via env)
  $container
  $"($container_app)/target/release/spm-transformer"
]

# Run training
^apptainer ...$args

# Capture exit code
let exit_code = $env.LAST_EXIT_CODE

# Print completion status
print ""
if $exit_code == 0 {
  print $"
✓ Training completed successfully!
  Checkpoints saved to: ($data_dir)/checkpoints/<timestamp>_j($env.SLURM_JOB_ID)
  Job ID: ($env.SLURM_JOB_ID)
"
} else {
  print $"✗ Training failed with exit code: ($exit_code)"
}

exit $exit_code
