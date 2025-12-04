use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv1d, Conv1dConfig},
        pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig},
        Dropout, DropoutConfig,
    },
    prelude::Backend,
    tensor::activation::relu,
    Tensor,
};

#[derive(Config, Debug)]
pub struct CnnTokenizerConfig {
    pub input_size: usize,  // pixels_per_line
    pub output_size: usize, // d_model
    pub dropout: f64,
}

impl CnnTokenizerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CnnTokenizer<B> {
        // Conv layers progressively increase channels
        // Input: 1 channel (grayscale scanline)
        // Hidden: 128 channels
        // Hidden: 256 channels
        // Output: d_model channels
        let conv1 = Conv1dConfig::new(1, 128, 5)
            .with_padding(burn::nn::PaddingConfig1d::Same)
            .init(device);

        let conv2 = Conv1dConfig::new(128, 256, 3)
            .with_padding(burn::nn::PaddingConfig1d::Same)
            .init(device);

        let conv3 = Conv1dConfig::new(256, self.output_size, 3)
            .with_padding(burn::nn::PaddingConfig1d::Same)
            .init(device);

        let pool = AdaptiveAvgPool1dConfig::new(1).init();
        let dropout = DropoutConfig::new(self.dropout).init();

        CnnTokenizer {
            conv1,
            conv2,
            conv3,
            pool,
            dropout,
        }
    }
}

#[derive(Module, Debug)]
pub struct CnnTokenizer<B: Backend> {
    conv1: Conv1d<B>,
    conv2: Conv1d<B>,
    conv3: Conv1d<B>,
    pool: AdaptiveAvgPool1d,
    dropout: Dropout,
}

impl<B: Backend> CnnTokenizer<B> {
    /// Process each scanline through CNN to get embeddings
    ///
    /// Input: [batch, num_scanlines, pixels_per_line]
    /// Ouput: [batch, num_scanlines, output_size]
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, num_scanlines, pixels] = input.dims();

        // Reshape to process all scanlines independently
        // [batch, num_scanlines, pixels] -> [batch * num_scanlines, 1, pixels]
        let x: Tensor<B, 3> = input
            .reshape([batch_size * num_scanlines, pixels])
            .unsqueeze_dim(1);

        // Apply conv layers with ReLU activation
        let x = self.conv1.forward(x);
        let x = relu(x);
        let x = self.dropout.forward(x);

        let x = self.conv2.forward(x);
        let x = relu(x);
        let x = self.dropout.forward(x);

        let x = self.conv3.forward(x);
        let x = relu(x);

        // Global average pooling
        let x = self.pool.forward(x);
        let [bs_ns, out_size, _] = x.dims();
        let x = x.reshape([bs_ns, out_size]);

        x.reshape([batch_size, num_scanlines, out_size])
    }
}
