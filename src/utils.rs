use burn::tensor::{backend::Backend, Bool, Tensor};

/// Generate a causal attention mask for autoregressive/sequential processing
/// Position i can only attend to position 0..=i (lower triangular)
pub fn create_causal_mask<B: Backend>(
    seq_length: usize,
    device: &B::Device,
) -> Tensor<B, 3, Bool> {
    // Zero vector entries
    let mut mask_data = vec![false; seq_length * seq_length];

    // Create lower triangular matrix
    for i in 0..seq_length {
        for j in 0..=i {
            mask_data[i * seq_length + j] = true;
        }
    }

    // Shape: [seq_length, seq_length]
    let mask = Tensor::<B, 2, Bool>::from_data(
        burn::tensor::TensorData::new(mask_data, [seq_length, seq_length]),
        device,
    );

    // Expand to [1, seq_length, seq_length]
    mask.unsqueeze_dim(0)
}
