use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    checkpoints_path: String,
}

#[derive(Debug, Default)]
struct EpochMetrics {
    accuracy: Option<f64>,
    loss: Option<f64>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let checkpoints_path: PathBuf = args.checkpoints_path.parse()?;
    let dir_entries = checkpoints_path.read_dir()?;

    for entry in dir_entries {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            if let Some(name) = entry.file_name().to_str() {
                match name {
                    "train" | "valid" => {
                        println!("\n=== {} directory ===", name);
                        analyze_directory(&entry.path())?;
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(())
}

fn parse_csv_average(content: &str) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0;

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }

        if let Some((value_str, _batch_size)) = line.split_once(',') {
            if let Ok(value) = value_str.trim().parse::<f64>() {
                sum += value;
                count += 1;
            }
        }
    }

    if count > 0 {
        Some(sum / count as f64)
    } else {
        None
    }
}

fn analyze_directory(path: &Path) -> Result<()> {
    let mut epochs: BTreeMap<u32, EpochMetrics> = BTreeMap::new();

    // Read all epoch directories
    for entry in path.read_dir()? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            if let Some(dir_name) = entry.file_name().to_str() {
                if let Some(epoch_num) = dir_name.strip_prefix("epoch-") {
                    if let Ok(epoch) = epoch_num.parse::<u32>() {
                        let epoch_path = entry.path();
                        let mut metrics = EpochMetrics::default();

                        // Read Accuracy.log (CSV format: value,batch_size)
                        let acc_path = epoch_path.join("Accuracy.log");
                        if acc_path.exists() {
                            if let Ok(content) = fs::read_to_string(&acc_path) {
                                metrics.accuracy = parse_csv_average(&content);
                            }
                        }

                        // Read Loss.log (CSV format: value,batch_size)
                        let loss_path = epoch_path.join("Loss.log");
                        if loss_path.exists() {
                            if let Ok(content) = fs::read_to_string(&loss_path) {
                                metrics.loss = parse_csv_average(&content);
                            }
                        }

                        epochs.insert(epoch, metrics);
                    }
                }
            }
        }
    }

    println!("Total epochs: {}", epochs.len());

    if epochs.is_empty() {
        println!("No epoch data found.");
        return Ok(());
    }

    // Find best metrics
    let best_acc = epochs.iter()
        .filter_map(|(epoch, m)| m.accuracy.map(|acc| (*epoch, acc)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let best_loss = epochs.iter()
        .filter_map(|(epoch, m)| m.loss.map(|loss| (*epoch, loss)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    if let Some((epoch, acc)) = best_acc {
        println!("Best accuracy: {:.4} (epoch {})", acc, epoch);
    }

    if let Some((epoch, loss)) = best_loss {
        println!("Best loss: {:.6} (epoch {})", loss, epoch);
    }

    // Show latest epoch metrics
    if let Some((&last_epoch, metrics)) = epochs.iter().last() {
        println!("\nLatest epoch ({}):", last_epoch);
        if let Some(acc) = metrics.accuracy {
            println!("  Accuracy: {:.4}", acc);
        }
        if let Some(loss) = metrics.loss {
            println!("  Loss: {:.6}", loss);
        }
    }

    Ok(())
}
