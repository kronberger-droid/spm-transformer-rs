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

use crate::{
    data::{STMBatcher, STMDataset},
    model::ScanLineEncoderConfig,
    training::TrainingConfig,
};

// Backend selection: CUDA for cluster, NdArray for local testing
#[cfg(feature = "cuda")]
type MyBackend = burn::backend::LibTorch<f32>;

#[cfg(not(feature = "cuda"))]
type MyBackend = burn::backend::NdArray;

// Training backend (with autodiff)
type MyAutodiffBackend = Autodiff<MyBackend>;

fn main() {
    // Set up device
    let device = Default::default();

    let dataset_path = "data/processed_data.npz";

    let train_config = TrainingConfig::new();

    println!("Loading dataset...");
    // Load data with plain backend (no autodiff needed for data)
    let (train_dataset, val_dataset, test_dataset) =
        STMDataset::<MyBackend>::train_val_test_split(
            dataset_path,
            &device,
            0.7,  // 70% train
            0.15, // 15% val
            Some(train_config.seed),
        )
        .expect("Failed to load dataset");

    println!("Train: {} samples", train_dataset.len());
    println!("Val:   {} samples", val_dataset.len());
    println!("Test:  {} samples", test_dataset.len());

    // Create model config
    let model_config = ScanLineEncoderConfig::new(
        128, // pixels_per_line
        256, // d_model
        8,   // num_heads
        4,   // num_layers
        128, // max_lines
        6,   // num_classes
    );

    println!("\nInitializing model...");
    println!("  d_model: {}", model_config.d_model);
    println!("  num_heads: {}", model_config.num_heads);
    println!("  num_layers: {}", model_config.num_layers);

    // Train batcher uses autodiff (for gradients), valid uses plain backend
    let batcher_train = STMBatcher::<MyAutodiffBackend>::new(device.clone());
    let batcher_valid = STMBatcher::<MyBackend>::new(device.clone());

    println!("\nCreating dataloaders...");
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(train_config.batch_size)
        .shuffle(train_config.seed)
        .num_workers(4)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(train_config.batch_size)
        .shuffle(train_config.seed)
        .num_workers(4)
        .build(val_dataset);

    // Model uses autodiff backend for training
    let model = model_config.init::<MyAutodiffBackend>(&device);

    println!("Setting up learner...");
    let learner = LearnerBuilder::new("./checkpoints")
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(burn::record::CompactRecorder::new())
        .learning_strategy(LearningStrategy::SingleDevice(device.clone()))
        .num_epochs(train_config.num_epochs)
        // .summary()
        .build(model, AdamConfig::new().init(), train_config.learning_rate);

    println!("\nStarting training...");
    let _trained_model = learner.fit(dataloader_train, dataloader_valid);

    println!("\n✓ Training complete!");
    println!("Model saved to ./checkpoints");
}
