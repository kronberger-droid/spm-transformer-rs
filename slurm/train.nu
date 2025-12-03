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

# Training hyperparameters
let epochs = 50
let batch_size = 32
let lr = 0.0001
let warmup_epochs = 5

# Model architecture
let d_model = 256
let num_heads = 8
let num_layers = 6

print $"\nStarting training with configuration:
  Epochs: ($epochs)
  Batch size: ($batch_size)
  Learning rate: ($lr)
  Model: d_model=($d_model), heads=($num_heads), layers=($num_layers)
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

  # Environment variables
  "--env" $"DATA_PATH=($env.DATA_PATH)"
  "--env" $"CHECKPOINT_BASE_DIR=($env.CHECKPOINT_BASE_DIR)"
  "--env" $"SLURM_JOB_ID=($env.SLURM_JOB_ID)"

  # Container and binary
  $container
  $"($container_app)/target/release/spm-transformer"

  # Training arguments
  "--learning-rate" $"($lr)"
  "--batch-size" $"($batch_size)"
  "--num-epochs" $"($epochs)"
  "--warmup-epochs" $"($warmup_epochs)"
  "--d-model" $"($d_model)"
  "--num-heads" $"($num_heads)"
  "--num-layers" $"($num_layers)"
  "--num-workers" $"($env.SLURM_CPUS_PER_TASK)"
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
