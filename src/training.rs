use burn::config::Config;

#[derive(Config, Debug)]
pub struct TrainingConfig {
    #[config(default = 32)]
    pub batch_size: usize,

    #[config(default = 10)]
    pub num_epochs: usize,

    #[config(default = 1e-3)]
    pub learning_rate: f64,

    #[config(default = 42)]
    pub seed: u64,
}
