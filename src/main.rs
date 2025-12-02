mod data;
mod model;

use std::{fs::File, path::PathBuf};

use burn::{
    backend::Autodiff,
    data::dataloader::DataLoaderBuilder,
    lr_scheduler::linear::LinearLrSchedulerConfig,
    optim::AdamConfig,
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder, LearningStrategy,
    },
};
use chrono::Local;
use clap::Parser;
use serde::Serialize;

use crate::{
    data::{STMBatcher, STMDataset},
    model::ScanLineEncoderConfig,
};

// Backend selection: CUDA for cluster, NdArray for local testing
#[cfg(feature = "cuda")]
type MyBackend = burn::backend::Cuda;

#[cfg(not(feature = "cuda"))]
type MyBackend = burn::backend::NdArray;

// Training backend (with autodiff)
type MyAutodiffBackend = Autodiff<MyBackend>;

#[derive(Parser, Debug, Serialize)]
#[command(version, about, long_about = None)]
struct Args {
    // Training Hyperparameters
    #[arg(short, long, default_value_t = 1e-4)]
    learning_rate: f64,

    #[arg(long, default_value_t = 5)]
    warmup_epochs: usize,

    #[arg(short, long, default_value_t = 32)]
    batch_size: usize,

    #[arg(short, long, default_value_t = 10)]
    num_epochs: usize,

    #[arg(short, long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value_t = 4)]
    num_workers: usize,

    // Model Architecture
    #[arg(long, default_value_t = 256)]
    d_model: usize,

    #[arg(long, default_value_t = 8)]
    num_heads: usize,

    #[arg(long, default_value_t = 6)]
    num_layers: usize,

    #[arg(long, default_value_t = 0.1)]
    dropout: f64,

    // Data
    #[arg(long, env = "DATA_PATH", default_value_t = String::from("data/processed_data.npz"))]
    data_path: String,

    #[arg(long, default_value_t = 0.7)]
    train_ratio: f32,

    #[arg(long, default_value_t = 0.15)]
    val_ratio: f32,

    // Checkpoints and Logging
    #[arg(long, env = "CHECKPOINT_BASE_DIR", default_value_t = String::from("./checkpoints"))]
    checkpoint_base_dir: String,
}

fn create_checkpoint_dir(base_dir: &str) -> Result<String, std::io::Error> {
    // Generate timestamp: YYYYMMDD_HHMMSS
    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();

    // Check for SLURM_JOB_ID environment variable
    let job_suffix = std::env::var("SLURM_JOB_ID")
        .map(|id| format!("_j{}", id))
        .unwrap_or_default();

    // Combine: timestamp_jJOBID or just timestamp
    let dir_name = format!("{}{}", timestamp, job_suffix);
    let checkpoint_dir = PathBuf::from(base_dir).join(dir_name);

    // Create the directory
    std::fs::create_dir_all(&checkpoint_dir)?;

    Ok(checkpoint_dir.to_string_lossy().to_string())
}

fn main() {
    // Parse command line arguments
    let args = Args::parse();

    // Auto-generate checkpoint directory
    let checkpoint_dir = create_checkpoint_dir(&args.checkpoint_base_dir)
        .expect("Failed to create checkpoint directory");

    // Save configuration as JSON
    let config_path = PathBuf::from(&checkpoint_dir).join("config.json");
    let config_file =
        File::create(&config_path).expect("Failed to create config.json");
    serde_json::to_writer_pretty(config_file, &args)
        .expect("Failed to write config.json");

    // Set up device
    let device = Default::default();

    println!("Configuration:");
    println!(
        "  Learning rate: {} (warmup: {} epochs)",
        args.learning_rate, args.warmup_epochs
    );
    println!("  Batch size: {}", args.batch_size);
    println!("  Epochs: {}", args.num_epochs);
    println!(
        "  Model: d_model={}, heads={}, layers={}",
        args.d_model, args.num_heads, args.num_layers
    );
    println!("  Data: {}", args.data_path);
    println!("  Checkpoints: {}", checkpoint_dir);
    println!("  Config saved: {}", config_path.display());

    println!("\nLoading dataset...");
    // Load data with plain backend (no autodiff needed for data)
    let (train_dataset, val_dataset, test_dataset) =
        STMDataset::<MyBackend>::train_val_test_split(
            &args.data_path,
            &device,
            args.train_ratio,
            args.val_ratio,
            Some(args.seed),
        )
        .expect("Failed to load dataset");

    println!("Train: {} samples", train_dataset.len());
    println!("Val:   {} samples", val_dataset.len());
    println!("Test:  {} samples", test_dataset.len());

    let num_classes = train_dataset.num_classes;
    let class_weights = train_dataset.class_weights.clone();
    let train_dataset_len = train_dataset.len(); // Store length before move

    println!("\nUsing {num_classes} classes with weights: {class_weights:?}");

    // Create model config
    let model_config = ScanLineEncoderConfig::new(
        128, // pixels_per_line (fixed by data format)
        args.d_model,
        args.num_heads,
        args.num_layers,
        128,         // max_lines (fixed by data format)
        num_classes, // from dataset
    );

    // Set dropout in config
    let model_config = model_config.with_dropout(args.dropout);

    println!("\nInitializing model...");
    println!("  d_model: {}", model_config.d_model);
    println!("  num_heads: {}", model_config.num_heads);
    println!("  num_layers: {}", model_config.num_layers);
    println!("  dropout: {}", model_config.dropout);

    // Train batcher uses autodiff (for gradients), valid uses plain backend
    let batcher_train = STMBatcher::<MyAutodiffBackend>::new(device.clone());
    let batcher_valid = STMBatcher::<MyBackend>::new(device.clone());

    println!("\nCreating dataloaders...");
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(args.batch_size)
        .shuffle(args.seed)
        .num_workers(args.num_workers)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(args.batch_size)
        .shuffle(args.seed)
        .num_workers(args.num_workers)
        .build(val_dataset);

    // Model uses autodiff backend for training
    let model =
        model_config.init::<MyAutodiffBackend>(&device, Some(class_weights));

    // Calculate steps per epoch (using stored length)
    let steps_per_epoch =
        (train_dataset_len + args.batch_size - 1) / args.batch_size;
    let warmup_steps = args.warmup_epochs * steps_per_epoch;

    println!("Setting up learner...");

    let scheduler =
        LinearLrSchedulerConfig::new(1e-6, args.learning_rate, warmup_steps)
            .init()
            .expect("Failed to create LR scheduler");

    let learner = LearnerBuilder::new(&checkpoint_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(burn::record::CompactRecorder::new())
        .learning_strategy(LearningStrategy::SingleDevice(device.clone()))
        .num_epochs(args.num_epochs)
        .build(model, AdamConfig::new().init(), scheduler);

    println!("\nStarting training...");
    let _trained_model = learner.fit(dataloader_train, dataloader_valid);

    println!("\n✓ Training complete!");
    println!("Model saved to {}", checkpoint_dir);
}
