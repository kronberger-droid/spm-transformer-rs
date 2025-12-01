use std::fs::File;

use anyhow::Result;
use ndarray::Axis;
use ndarray_npy::NpzReader;
use plotly::{HeatMap, Plot};
use rustfft::{num_complex::Complex, FftPlanner};

fn main() -> Result<()> {
    // Load dataset
    let path = "data/data_32bit_6_class.npz";
    let file = File::open(path).expect("failed to open data file");
    let mut npz = NpzReader::new(file)?;

    let image_index = 0;
    let row_index = 50;

    // Get some row of some image
    let images_nd: ndarray::Array3<f32> = npz.by_name("images")?;
    // let labels_nd: ndarray::Array1<f32> = npz.by_name("labels")?;
    let image = get_nth_image(&images_nd, image_index)
        .expect("No image for this index");

    let z: Vec<Vec<f32>> = image.outer_iter().map(|row| row.to_vec()).collect();

    let trace = HeatMap::new_z(z);
    let mut plot = Plot::new();
    plot.add_trace(trace);

    // Make the plot square (you still need to pick a size)
    let size = 800; // Adjust this one value to change the square size
    let layout = plotly::Layout::new()
        .width(size)
        .height(size)
        .auto_size(false);

    plot.set_layout(layout);
    plot.show();

    let row = get_nth_row(&image, row_index).expect("No row at this index");
    // Setup FFT
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(128);
    let mut buffer: Vec<Complex<f32>> =
        row.iter().map(|&x| Complex { re: x, im: 0.0 }).collect();
    fft.process(&mut buffer);

    // Calculate magnitudes
    let magnitudes: Vec<f32> = buffer.iter().map(|c| c.norm()).collect();

    // Plot the FFT magnitudes (skip first 10 low frequencies to avoid DC peak)
    let skip_low_freq = 10;
    let frequencies: Vec<usize> = (skip_low_freq..magnitudes.len()).collect();
    let filtered_magnitudes: Vec<f32> =
        magnitudes.iter().skip(skip_low_freq).copied().collect();
    let trace = plotly::Scatter::new(frequencies, filtered_magnitudes)
        .name("FFT Magnitude");

    let mut fft_plot = Plot::new();
    fft_plot.add_trace(trace);

    let fft_layout = plotly::Layout::new()
        .title("FFT Magnitude Spectrum")
        .x_axis(plotly::layout::Axis::new().title("Frequency Bin"))
        .y_axis(plotly::layout::Axis::new().title("Magnitude"))
        .width(1200)
        .height(600);

    fft_plot.set_layout(fft_layout);
    fft_plot.show();

    Ok(())
}

fn get_nth_image(
    images: &ndarray::Array3<f32>,
    index: usize,
) -> Option<ndarray::Array2<f32>> {
    if index < images.shape()[0] {
        Some(images.index_axis(Axis(0), index).to_owned())
    } else {
        None
    }
}

fn get_nth_row(image: &ndarray::Array2<f32>, row: usize) -> Option<Vec<f32>> {
    if row < image.shape()[0] {
        Some(image.index_axis(Axis(0), row).to_owned().to_vec())
    } else {
        None
    }
}
