// Model module - contains all neural network architectures

pub mod vit;
pub mod sequential;

// Re-export main types for convenience
pub use vit::{ScanLineEncoder, ScanLineEncoderConfig};
pub use sequential::{SequentialScanLineEncoder, SequentialScanLineEncoderConfig};
