use std::fs::File;

use anyhow::{bail, Context, Result};
use burn::{
    data::dataset::Dataset,
    prelude::*,
    tensor::{backend::Backend, Int, Tensor, TensorData},
};
use ndarray_npy::NpzReader;

pub struct STMDataset<B: Backend> {
    pub images: Tensor<B, 3>,      // [num_samples, 128, 128]
    pub labels: Tensor<B, 1, Int>, // [num_samples]
}

impl<B: Backend> STMDataset<B> {
    pub fn from_npz(path: &str, device: &B::Device) -> Result<Self> {
        // open file and load images
        let file =
            File::open(path).context(format!("failed to open {}", path))?;
        let mut npz = NpzReader::new(file)?;

        // Create Images Tensor
        let images_nd: ndarray::Array3<f32> = npz.by_name("images")?;

        let [n_samples, height, width] = images_nd.shape() else {
            bail!("Expected 3D image array, got shape {:?}", images_nd.shape());
        };

        if *height != 128 || *width != 128 {
            bail!("Expected 128x128 images, got {}x{}", height, width);
        }

        let shape = images_nd.shape().to_vec();
        let images_data =
            TensorData::new(images_nd.into_raw_vec_and_offset().0, shape);

        let images = Tensor::from_data(images_data, device);

        // Create Labels Tensor
        let labels_nd: ndarray::Array1<i64> = npz.by_name("labels")?;
        let shape = labels_nd.shape().to_vec();
        let labels_data =
            TensorData::new(labels_nd.into_raw_vec_and_offset().0, shape);

        let labels = Tensor::<B, 1, Int>::from_data(
            labels_data.convert::<i64>(),
            device,
        );

        Ok(STMDataset { images, labels })
    }

    pub fn len(&self) -> usize {
        self.labels.dims()[0]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn train_val_test_split(
        path: &str,
        device: &B::Device,
        train_ratio: f32,
        val_ratio: f32,
        seed: Option<u64>,
    ) -> Result<(Self, Self, Self)> {
        let full = Self::from_npz(path, device)?;
        let n = full.len();

        let mut indices: Vec<usize> = (0..n).collect();

        if let Some(seed) = seed {
            use rand::{seq::SliceRandom, SeedableRng};
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            indices.shuffle(&mut rng);
        }

        let indices_i64: Vec<i64> = indices.iter().map(|&i| i as i64).collect();

        let indices_tensor = Tensor::<B, 1, Int>::from_data(
            TensorData::from(indices_i64.as_slice()),
            device,
        );

        let images_shuffled = full.images.select(0, indices_tensor.clone());
        let labels_shuffled = full.labels.select(0, indices_tensor);

        let train_end = (n as f32 * train_ratio) as usize;
        let val_end = train_end + (n as f32 * val_ratio) as usize;

        Ok((
            Self {
                images: images_shuffled.clone().slice(0..train_end),
                labels: labels_shuffled.clone().slice(0..train_end),
            },
            Self {
                images: images_shuffled.clone().slice(train_end..val_end),
                labels: labels_shuffled.clone().slice(train_end..val_end),
            },
            Self {
                images: images_shuffled.slice(val_end..n),
                labels: labels_shuffled.slice(val_end..n),
            },
        ))
    }
}

impl<B: Backend> Dataset<STMItem> for STMDataset<B> {
    fn get(&self, index: usize) -> Option<STMItem> {
        if index >= self.len() {
            return None;
        }

        let image_tensor = self.images.clone().slice(index..index + 1);
        let label_tensor = self.labels.clone().slice(index..index + 1);

        let image_data = image_tensor.into_data();
        let label_data = label_tensor.into_data();

        let image = image_data
            .to_vec::<f32>()
            .expect("There should be an image available");

        let label_vec = label_data
            .to_vec::<i64>()
            .expect("There should be an label available");

        let label = label_vec[0];

        Some(STMItem { image, label })
    }

    fn len(&self) -> usize {
        self.labels.dims()[0]
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn iter(&self) -> burn::data::dataset::DatasetIterator<'_, STMItem>
    where
        Self: Sized,
    {
        todo!()
    }
}

#[derive(Clone, Debug)]
pub struct STMBatch<B: Backend> {
    pub images: Tensor<B, 3>,       // [batch_size, 128, 128]
    pub targets: Tensor<B, 1, Int>, // [batch_size]
}

#[derive(Clone, Debug)]
pub struct STMItem {
    pub image: Vec<f32>,
    pub label: i64,
}

use burn::data::dataloader::batcher::Batcher;

#[derive(Clone)]
pub struct STMBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> STMBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<B, STMItem, STMBatch<B>> for STMBatcher<B> {
    fn batch(&self, items: Vec<STMItem>, device: &B::Device) -> STMBatch<B> {
        let batch_size = items.len();

        // Collect images into flat Vector
        let images_vec: Vec<f32> = items
            .iter()
            .flat_map(|item| item.image.iter().copied())
            .collect();

        // Collect all labels into flat Vector
        let labels_vec: Vec<i64> =
            items.iter().map(|item| item.label).collect();

        // Convert to tensors
        let images = Tensor::from_data(
            TensorData::new(images_vec, [batch_size, 128, 128]),
            device,
        );
        let targets = Tensor::from_data(
            TensorData::new(labels_vec, [batch_size]),
            device,
        );

        STMBatch { images, targets }
    }
}
