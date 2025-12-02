#!/usr/bin/env nu
#
#SBATCH --job-name=stm-rust-transformer
#SBATCH --output=/share/rusty-tip/logs/train_rust_%j.out
#SBATCH --error=/share/rusty-tip/logs/train_rust_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=a40:1
#SBATCH --partition=GPU-a40

# =================
# Environment Setup
# =================

# Directories
let code_dir = $"($env.HOME)/rust/spm-transformer-rs"
let container = "/share/rusty-tip/apptainer/stm-transformer.sif"
let data_dir = "/share/rusty-tip/data"

# Set environment variables for Rust to use
$env.DATA_PATH = $"($data_dir)/processed_data.npz"
$env.CHECKPOINT_BASE_DIR = $"($data_dir)/checkpoints"

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

# Print Rust and CUDA info from container
print "Build Environment (inside container):"
^apptainer exec $container rustc --version
^apptainer exec $container cargo --version

# ===========
# Build Phase
# ===========

print "\nBuilding with CUDA support inside container..."
^apptainer exec --nv --bind $"($code_dir):/app" $container sh -c "cd /app && 
cargo build --release --features cuda"

if $env.LAST_EXIT_CODE != 0 {
  print "Build failed!"
  exit $env.LAST_EXIT_CODE
}

# ===========
# Training
# ===========

let epochs = 50
let batch_size = 32
# Learning rate
let lr = 0.001
# Model Dimension
let d_model = 256
# Attention heads per layer
let num_heads = 8
# Attention layers in model (defaults to 6 only for reference)
let num_layers = 6

print $"\nStarting training with configuration:
  Epochs: ($epochs)
  Batch size: ($batch_size)
  Learning rate: ($lr)
  Model: d_model=($d_model), heads=($num_heads), layers=($num_layers)
  Data: ($env.DATA_PATH)
  Checkpoint base: ($env.CHECKPOINT_BASE_DIR)
"

# Training arguments (without the apptainer command itself)
let args = [
  "exec" "--nv"
  "--bind" $"($code_dir):/app"
  "--bind" $"($data_dir):/data"
  "--env" $"DATA_PATH=($env.DATA_PATH)"
  "--env" $"CHECKPOINT_BASE_DIR=($env.CHECKPOINT_BASE_DIR)"
  "--env" $"SLURM_JOB_ID=($env.SLURM_JOB_ID)"
  $container
  "/app/target/release/stm-transformer"
  "--learning-rate" $"($lr)"
  "--batch-size" $"($batch_size)"
  "--num-epochs" $"($epochs)"
  "--d-model" $"($d_model)"
  "--num-heads" $"($num_heads)"
  "--num-layers" $"($num_layers)"
  "--num-workers" $"($env.SLURM_CPUS_PER_TASK)"
  "--use-class-weights"
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
  Checkpoints saved to: ($env.CHECKPOINT_BASE_DIR)/<timestamp>_j($env.SLURM_JOB_ID)
  Job ID: ($env.SLURM_JOB_ID)
"
} else {
  print $"✗ Training failed with exit code: ($exit_code)"
}

exit $exit_code
