use burn::{
    config::Config,
    module::{Module, Param},
    nn::{
        loss::CrossEntropyLossConfig,
        transformer::{
            TransformerEncoder, TransformerEncoderConfig,
            TransformerEncoderInput,
        },
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm,
        LayerNormConfig, Linear, LinearConfig,
    },
    tensor::{
        backend::{AutodiffBackend, Backend},
        Distribution, Int, Tensor,
    },
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::{
    data::STMBatch,
    tokenizer::{CnnTokenizer, CnnTokenizerConfig},
};

#[derive(Config, Debug)]
pub struct ScanLineEncoderConfig {
    /// Number of pixels per scanline
    pub pixels_per_line: usize,
    /// Model embedding dimension
    pub d_model: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Maximum number of scanlines
    pub max_lines: usize,
    /// Number of output classes
    pub num_classes: usize,
    /// Dropout rate
    #[config(default = 0.1)]
    pub dropout: f64,
}

#[derive(Module, Debug)]
pub struct ScanLineEncoder<B: Backend> {
    cnn_tokenizer: CnnTokenizer<B>,
    pos_embedding: Embedding<B>,
    cls_token: Param<Tensor<B, 1>>,
    transformer: TransformerEncoder<B>,
    norm: LayerNorm<B>,
    classifier: Linear<B>,
    dropout: Dropout,
    class_weights: Option<Vec<f32>>,
}

impl ScanLineEncoderConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
        class_weights: Option<Vec<f32>>,
    ) -> ScanLineEncoder<B> {
        let seq_length = self.max_lines + 1;

        ScanLineEncoder {
            cnn_tokenizer: CnnTokenizerConfig::new(
                self.pixels_per_line,
                self.d_model,
                self.dropout,
            )
            .init(device),
            pos_embedding: EmbeddingConfig::new(seq_length, self.d_model)
                .init(device),
            cls_token: Param::from_tensor(Tensor::random(
                [self.d_model],
                Distribution::Uniform(-0.02, 0.02),
                device,
            )),
            transformer: TransformerEncoderConfig::new(
                self.d_model,
                self.d_model * 4,
                self.num_heads,
                self.num_layers,
            )
            .with_dropout(self.dropout)
            .with_norm_first(true)
            .init(device),
            norm: LayerNormConfig::new(self.d_model).init(device),
            classifier: LinearConfig::new(self.d_model, self.num_classes)
                .init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            class_weights,
        }
    }
}

impl<B: Backend> ScanLineEncoder<B> {
    pub fn forward(&self, scans: Tensor<B, 3>) -> Tensor<B, 2> {
        // Input: [batch_size, num_lines, pixels_per_line], e.g., [32, 128, 128]
        let [batch_size, num_lines, _] = scans.dims();

        // Tokenize each scan line using CNN: [batch, num_lines, pixels_per_line] -> [batch, num_lines, d_model]
        let embedded = self.cnn_tokenizer.forward(scans);
        let device = embedded.device();

        // Prepare CLS token for each batch item: [d_model] -> [batch, 1, d_model]
        let cls_tokens = self
            .cls_token
            .val()
            .reshape([1, 1, -1])
            .repeat_dim(0, batch_size);

        // Prepend CLS token to sequence: [batch, num_lines+1, d_model]
        let x = Tensor::cat(vec![cls_tokens, embedded], 1);

        // Add positional embeddings to preserve scan line order
        let positions = Tensor::arange(0..(num_lines + 1) as i64, &device)
            .reshape([1, num_lines + 1])
            .repeat_dim(0, batch_size);
        let pos_embedded = self.pos_embedding.forward(positions);
        let x = x + pos_embedded;

        // Apply dropout for regularization
        let x = self.dropout.forward(x);

        let input = TransformerEncoderInput::new(x);
        let x = self.transformer.forward(input);

        // Extract CLS token for classification: [batch, num_lines+1, d_model] -> [batch, d_model]
        let cls = x.slice([0..batch_size, 0..1]).squeeze();

        // Final classification: [batch, d_model] -> [batch, num_classes]
        let cls = self.norm.forward(cls);
        self.classifier.forward(cls)
    }

    pub fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
        class_weights: Option<Vec<f32>>,
    ) -> ClassificationOutput<B> {
        // Forward pass through model
        let output = self.forward(images); // [batch, num_classes]

        // Compute cross-entropy loss
        let loss = CrossEntropyLossConfig::new()
            .with_weights(class_weights)
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<STMBatch<B>, ClassificationOutput<B>>
    for ScanLineEncoder<B>
{
    fn step(&self, item: STMBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(
            item.images,
            item.targets,
            self.class_weights.clone(),
        );

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<STMBatch<B>, ClassificationOutput<B>>
    for ScanLineEncoder<B>
{
    fn step(&self, item: STMBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(
            item.images,
            item.targets,
            self.class_weights.clone(),
        )
    }
}
