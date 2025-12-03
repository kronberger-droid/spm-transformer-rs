#!/usr/bin/env nu
#
#SBATCH --job-name=spm-transformer-build
#SBATCH --output=/share/rusty-tip/logs/build_rust_%j.out
#SBATCH --error=/share/rusty-tip/logs/build_rust_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
# No GPU needed for build

# =================
# Environment Setup
# =================

let code_dir = $"($env.HOME)/rust/spm-transformer-rs"
let container = "/share/rusty-tip/apptainer/stm-transformer.sif"

cd $code_dir

print $"
Build Job Information:
  Job ID: ($env.SLURM_JOB_ID)
  Node: ($env.SLURM_NODELIST)
  Working Directory: ($env.PWD)
  Start Time: (date now | format date '%Y-%m-%d %H:%M:%S')
"

# ===========
# Build Phase
# ===========

print "Cleaning previous build artifacts..."
^apptainer exec --bind $"($code_dir):/app" $container sh -c "cd /app && cargo clean"

print "Building with CUDA support..."
^apptainer exec --nv --bind $"($code_dir):/app" $container sh -c "cd /app && cargo build --release --features cuda"

let exit_code = $env.LAST_EXIT_CODE

if $exit_code == 0 {
  print $"\n✓ Build successful!
  Binary: ($code_dir)/target/release/spm-transformer
  Ready for training jobs.
"
} else {
  print $"\n✗ Build failed with exit code: ($exit_code)"
}

exit $exit_code
