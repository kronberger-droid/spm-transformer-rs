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
let code_dir = $"($env.HOME)/Programming/rust/spm-transformer"
let data_dir = "/share/rusty-tip"
let data_path = $"($data_dir)/processed_data.npz"
let output_dir = $"($data_dir)/checkpoints/($env.SLURM_JOB_ID)"

# Create output directory
mkdir $output_dir

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

# Print Rust and CUDA info
print "Build Environment:"
^rustc --version
^cargo --version
^nvcc --version | head -n 1

# ===========
# Build Phase
# ===========

print "\nBuilding with CUDA support..."
^cargo build --release --features cuda

if $env.LAST_EXIT_CODE != 0 {
  print "Build failed!"
  exit $env.LAST_EXIT_CODE
}

# ===========
# Training
# ===========

let epochs = 50
let batch_size = 32
let lr = 0.001
let d_model = 256
let num_heads = 8
let num_layers = 4

print $"\nStarting training with configuration:
Epochs: ($epochs)
Batch size: ($batch_size)
Learning rate: ($lr)
Model: d_model=($d_model), heads=($num_heads), layers=($num_layers)
Data: ($data_path)
Output: ($output_dir)
"

let cmd = [
  "./target/release/stm-transformer"
  "--learning-rate" $"($lr)"
  "--batch-size" $"($batch_size)"
  "--num-epochs" $"($epochs)"
  "--d-model" $"($d_model)"
  "--num-heads" $"($num_heads)"
  "--num-layers" $"($num_layers)"
  "--data-path" $data_path
  "--checkpoint-dir" $output_dir
  "--num-workers" $"($env.SLURM_CPUS_PER_TASK)"
  "--use-class-weights"
]

# Run training
...$cmd

# Capture exit code
let exit_code = $env.LAST_EXIT_CODE

# Print completion status
print ""
if $exit_code == 0 {
  print $"
✓ Training completed successfully!
Checkpoints saved to: ($output_dir)
Job ID: ($env.SLURM_JOB_ID)
"
} else {
  print $"✗ Training failed with exit code: ($exit_code)"
}
