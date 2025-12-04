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
        Bool, Distribution, Int, Tensor,
    },
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::{
    data::STMBatch,
    tokenizer::{CnnTokenizer, CnnTokenizerConfig},
    utils::create_causal_mask,
};

/// Configuration for Sequential Transformer with causal masking.
#[derive(Config, Debug)]
pub struct SequentialScanLineEncoderConfig {
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

/// Sequential Transformer Encoder with CNN tokenization and causal masking.
///
/// Key differences from ViT:
/// - Uses CNN tokenizer instead of Linear for scanline embedding
/// - Applies causal masking so scanline i only sees 0..=i (no future peeking)
/// - Pre-computes causal mask once since sequence length is fixed (128+1=129)
#[derive(Module, Debug)]
pub struct SequentialScanLineEncoder<B: Backend> {
    cnn_tokenizer: CnnTokenizer<B>,
    pos_embedding: Embedding<B>,
    cls_token: Param<Tensor<B, 1>>,
    transformer: TransformerEncoder<B>,
    norm: LayerNorm<B>,
    classifier: Linear<B>,
    dropout: Dropout,
    class_weights: Option<Vec<f32>>,
    /// Pre-computed causal mask: [1, seq_length, seq_length]
    /// Broadcasts over batch dimension automatically
    causal_mask: Tensor<B, 3, Bool>,
}

impl SequentialScanLineEncoderConfig {
    /// Initialize a new sequential encoder.
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
        class_weights: Option<Vec<f32>>,
    ) -> SequentialScanLineEncoder<B> {
        let seq_length = self.max_lines + 1; // 128 scanlines + 1 CLS token = 129

        SequentialScanLineEncoder {
            // CNN tokenizer: [batch, 128, 128] → [batch, 128, d_model]
            cnn_tokenizer: CnnTokenizerConfig::new(
                self.pixels_per_line,
                self.d_model,
                self.dropout,
            )
            .init(device),

            // Positional embeddings for seq_length positions
            pos_embedding: EmbeddingConfig::new(seq_length, self.d_model)
                .init(device),

            // CLS token initialization
            cls_token: Param::from_tensor(Tensor::random(
                [self.d_model],
                Distribution::Uniform(-0.02, 0.02),
                device,
            )),

            // Transformer encoder with same config as ViT
            transformer: TransformerEncoderConfig::new(
                self.d_model,
                self.d_model * 4,
                self.num_heads,
                self.num_layers,
            )
            .with_dropout(self.dropout)
            .with_norm_first(true)
            .init(device),

            // Layer norm and classifier
            norm: LayerNormConfig::new(self.d_model).init(device),
            classifier: LinearConfig::new(self.d_model, self.num_classes)
                .init(device),
            dropout: DropoutConfig::new(self.dropout).init(),

            // Pre-compute causal mask: [1, 129, 129] lower triangular
            causal_mask: create_causal_mask(seq_length, device),

            class_weights,
        }
    }
}

impl<B: Backend> SequentialScanLineEncoder<B> {
    /// Forward pass with causal attention masking.
    ///
    /// # Arguments
    /// * `images` - Input tensor of shape [batch, num_scanlines, pixels_per_line]
    ///
    /// # Returns
    /// Logits tensor of shape [batch, num_classes]
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        // Input: [batch, num_scanlines, pixels_per_line], e.g., [32, 128, 128]
        let [batch_size, num_lines, _] = images.dims();

        // 1. Tokenize scanlines using CNN: [batch, 128, 128] → [batch, 128, d_model]
        let embedded = self.cnn_tokenizer.forward(images);
        let device = embedded.device();

        // 2. Prepend CLS token: [batch, 128, d_model] → [batch, 129, d_model]
        let cls_tokens = self
            .cls_token
            .val()
            .reshape([1, 1, -1])
            .repeat_dim(0, batch_size);
        let x = Tensor::cat(vec![cls_tokens, embedded], 1);

        // 3. Add positional embeddings
        let positions = Tensor::arange(0..(num_lines + 1) as i64, &device)
            .reshape([1, num_lines + 1])
            .repeat_dim(0, batch_size);
        let pos_embedded = self.pos_embedding.forward(positions);
        let x = x + pos_embedded;

        // 4. Apply dropout
        let x = self.dropout.forward(x);

        // 5. Create TransformerEncoderInput with causal mask
        // The causal_mask is [1, 129, 129] and broadcasts over batch dimension
        let input =
            TransformerEncoderInput::new(x).mask_attn(self.causal_mask.clone());

        // 6. Apply transformer
        let x = self.transformer.forward(input);

        // 7. Extract CLS token and classify
        let cls = x.slice([0..batch_size, 0..1]).squeeze();
        let cls = self.norm.forward(cls);
        self.classifier.forward(cls)
    }

    /// Forward pass for classification with loss computation.
    pub fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
        class_weights: Option<Vec<f32>>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .with_weights(class_weights)
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<STMBatch<B>, ClassificationOutput<B>>
    for SequentialScanLineEncoder<B>
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
    for SequentialScanLineEncoder<B>
{
    fn step(&self, item: STMBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(
            item.images,
            item.targets,
            self.class_weights.clone(),
        )
    }
}
