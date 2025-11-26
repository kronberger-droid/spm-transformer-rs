mod data;
mod model;
mod training;

use burn::{
    backend::Autodiff,
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder, LearningStrategy,
    },
};
use clap::Parser;

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

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    // Training Hyperparameters
    #[arg(short, long, default_value_t = 1e-3)]
    learning_rate: f64,

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

    #[arg(long, default_value_t = 4)]
    num_layers: usize,

    #[arg(long, default_value_t = 0.1)]
    dropout: f64,

    // Data
    #[arg(long, default_value_t = String::from("data/processed_data.npz"))]
    data_path: String,

    #[arg(long, default_value_t = 0.7)]
    train_ratio: f32,

    #[arg(long, default_value_t = 0.15)]
    val_ratio: f32,

    // Checkpoints and Logging
    #[arg(long, default_value_t = String::from("./checkpoints"))]
    checkpoint_dir: String,

    // Class Weights
    #[arg(long, default_value_t = true)]
    use_class_weights: bool,
}

fn main() {
    // Parse command line arguments
    let args = Args::parse();

    // Set up device
    let device = Default::default();

    println!("Configuration:");
    println!("  Learning rate: {}", args.learning_rate);
    println!("  Batch size: {}", args.batch_size);
    println!("  Epochs: {}", args.num_epochs);
    println!("  Model: d_model={}, heads={}, layers={}", args.d_model, args.num_heads, args.num_layers);
    println!("  Data: {}", args.data_path);
    println!("  Checkpoints: {}", args.checkpoint_dir);
    println!("  Weighted loss: {}", args.use_class_weights);

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

    // Create model config
    let model_config = ScanLineEncoderConfig::new(
        128,              // pixels_per_line (fixed by data format)
        args.d_model,
        args.num_heads,
        args.num_layers,
        128,              // max_lines (fixed by data format)
        6,                // num_classes (fixed by dataset)
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
    let model = model_config.init::<MyAutodiffBackend>(&device);

    println!("Setting up learner...");
    let learner = LearnerBuilder::new(&args.checkpoint_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(burn::record::CompactRecorder::new())
        .learning_strategy(LearningStrategy::SingleDevice(device.clone()))
        .num_epochs(args.num_epochs)
        .build(model, AdamConfig::new().init(), args.learning_rate);

    println!("\nStarting training...");
    let _trained_model = learner.fit(dataloader_train, dataloader_valid);

    println!("\n✓ Training complete!");
    println!("Model saved to {}", args.checkpoint_dir);
}
