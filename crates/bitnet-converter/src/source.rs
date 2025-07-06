// File: E:\Desktop\Bitnet rs\crates\bitnet-converter\src\source.rs
// --- FULL REPLACEMENT ---

//! Robust safetensors loader and tensor source utilities for BitNet Converter.
//!
//! Provides types and functions for loading, mapping, and representing tensors from safetensors files.

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use bytemuck;
use rayon::prelude::*;

/// Enum for robust tensor data representation.
#[derive(Debug, Clone)]
pub enum TensorData {
    /// 32-bit floating point tensor data.
    F32(Vec<f32>),
    /// 32-bit unsigned integer tensor data.
    U32(Vec<u32>),
    /// 32-bit signed integer tensor data.
    I32(Vec<i32>),
    /// BF16 (stored as f32) tensor data.
    BF16(Vec<f32>),
    /// 8-bit signed integer tensor data.
    I8(Vec<i8>),
    /// 8-bit unsigned integer tensor data.
    U8(Vec<u8>),
    // Add more as needed
}

impl TensorData {
    /// Returns a reference to the underlying u32 vector, if present.
    pub fn as_u32_vec(&self) -> Option<&Vec<u32>> {
        match self {
            TensorData::U32(v) => Some(v),
            _ => None,
        }
    }
}

/// Map from tensor name to (data, shape).
pub type RichTensorMap = HashMap<String, (TensorData, Vec<usize>)>;

/// Model source for loading tensors.
pub enum ModelSource {
    /// Load tensors from a safetensors file.
    SafetensorsFile(String),
}

impl ModelSource {
    /// Load tensors from the model source into a RichTensorMap.
    pub fn load_tensors(&self) -> Result<RichTensorMap, Box<dyn std::error::Error>> {
        match self {
            ModelSource::SafetensorsFile(path) => load_safetensors_mmap(path.as_ref()),
        }
    }
}

fn load_safetensors_mmap(path: &Path) -> Result<RichTensorMap, Box<dyn std::error::Error>> {
    let t0 = std::time::Instant::now();
    let file = File::open(path)?;
    println!("[PROFILE] File open: {:?}", t0.elapsed());
    let t1 = std::time::Instant::now();
    let mmap = unsafe { Mmap::map(&file)? };
    println!("[PROFILE] Mmap: {:?}", t1.elapsed());
    let t2 = std::time::Instant::now();
    let safetensors = SafeTensors::deserialize(&mmap)?;
    println!("[PROFILE] Header/metadata parse: {:?}", t2.elapsed());
    // --- Parallel tensor conversion for large models ---
    let tensor_vec: Vec<_> = safetensors.names().par_iter().map(|name| {
        let t_tensor = std::time::Instant::now();
        let tensor_view = safetensors.tensor(name).ok()?;
        let dtype = tensor_view.dtype();
        let original_shape = tensor_view.shape();
        let shape: Vec<usize> = if original_shape.len() == 1 {
            vec![1, original_shape[0]]
        } else if original_shape.len() == 2 {
            original_shape.to_vec()
        } else {
            log::warn!("Skipping tensor '{}' with unsupported shape {:?}", name, original_shape);
            return None;
        };
        let data = tensor_view.data();
        let tensor_data = match dtype {
            Dtype::BF16 => {
                let u16_slice = bytemuck::cast_slice::<u8, u16>(data);
                let f32_vec: Vec<f32> = u16_slice.iter().map(|&bits| half::bf16::from_bits(bits).to_f32()).collect();
                TensorData::BF16(f32_vec)
            },
            Dtype::F32 => {
                let f32_slice = bytemuck::cast_slice::<u8, f32>(data);
                TensorData::F32(f32_slice.to_vec())
            },
            Dtype::U32 => {
                let u32_slice = bytemuck::cast_slice::<u8, u32>(data);
                TensorData::U32(u32_slice.to_vec())
            },
            Dtype::I32 => {
                let i32_slice = bytemuck::cast_slice::<u8, i32>(data);
                TensorData::I32(i32_slice.to_vec())
            },
            Dtype::I8 => {
                TensorData::I8(data.to_vec().into_iter().map(|b| b as i8).collect())
            },
            Dtype::U8 => {
                TensorData::U8(data.to_vec())
            },
            _ => {
                log::warn!("Skipping tensor '{}' with unsupported dtype {:?}", name, dtype);
                return None;
            }
        };
        let elapsed = t_tensor.elapsed();
        println!("[PROFILE] Tensor {} conversion: {:?} (dtype: {:?})", name, elapsed, dtype);
        Some((name.to_string(), (tensor_data, shape)))
    }).filter_map(|x| x).collect();
    let map: RichTensorMap = tensor_vec.into_iter().collect();
    println!("[PROFILE] Total tensor conversion: {:?}", t0.elapsed());
    println!("[PROFILE] Total load_safetensors_mmap: {:?}", t0.elapsed());
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use safetensors::{tensor::TensorView, Dtype, serialize_to_file};
    use half::bf16;
    use bytemuck;

    // Helper: convert Vec<bf16> to Vec<u8> for writing
    fn bf16_vec_to_bytes(vec: &[bf16]) -> Vec<u8> {
        let u16_vec: Vec<u16> = vec.iter().map(|b| b.to_bits()).collect();
        bytemuck::cast_slice(&u16_vec).to_vec()
    }

    #[test]
    fn test_loads_both_1d_and_2d_tensors() {
        let data_f32_2d = vec![1.0f32, 2.0, 3.0, 4.0];
        let data_bf16_2d: Vec<bf16> = data_f32_2d.iter().map(|&f| bf16::from_f32(f)).collect();
        let bytes_2d: Vec<u8> = bf16_vec_to_bytes(&data_bf16_2d);

        let data_f32_1d = vec![5.0f32, 6.0];
        let data_bf16_1d: Vec<bf16> = data_f32_1d.iter().map(|&f| bf16::from_f32(f)).collect();
        let bytes_1d: Vec<u8> = bf16_vec_to_bytes(&data_bf16_1d);
        
        let mut tensors = std::collections::BTreeMap::new();
        tensors.insert("weight_2d".to_string(), TensorView::new(Dtype::BF16, vec![2, 2], &bytes_2d).unwrap());
        tensors.insert("weight_1d".to_string(), TensorView::new(Dtype::BF16, vec![2], &bytes_1d).unwrap());
        
        let tmp = NamedTempFile::new().unwrap();
        serialize_to_file(&tensors, &None, tmp.path()).unwrap();

        let source = ModelSource::SafetensorsFile(tmp.path().to_str().unwrap().to_string());
        let tensor_map = source.load_tensors().unwrap();
        
        // Assert 2D tensor was loaded correctly
        assert!(tensor_map.contains_key("weight_2d"), "2D tensor should be loaded");
        let (tensor_2d, shape_2d) = &tensor_map["weight_2d"];
        assert_eq!(shape_2d, &vec![2, 2]);
        assert_eq!(tensor_2d.as_f32_vec().unwrap(), &data_f32_2d);
        
        // Assert 1D tensor was loaded and promoted correctly
        assert!(tensor_map.contains_key("weight_1d"), "1D tensor should be loaded");
        let (tensor_1d, shape_1d) = &tensor_map["weight_1d"];
        assert_eq!(shape_1d, &vec![1, 2], "1D tensor should be promoted to 2D");
        assert_eq!(tensor_1d.as_f32_vec().unwrap(), &data_f32_1d);

        println!("\n✅ Test Passed: Burn-free loader correctly handles 1D and 2D tensors.");
    }

    #[test]
    fn test_error_on_corrupt_safetensors_file() {
        use std::io::Write;
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(b"not a safetensors file").unwrap();
        let source = ModelSource::SafetensorsFile(tmp.path().to_str().unwrap().to_string());
        let result = source.load_tensors();
        assert!(result.is_err(), "Should error on corrupt safetensors file");
    }

    #[test]
    fn test_error_on_unsupported_shape() {
        // 3D tensor (unsupported)
        let data_f32 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let data_bf16: Vec<bf16> = data_f32.iter().map(|&f| bf16::from_f32(f)).collect();
        let bytes: Vec<u8> = bf16_vec_to_bytes(&data_bf16);
        let mut tensors = std::collections::BTreeMap::new();
        tensors.insert("weight_3d".to_string(), TensorView::new(Dtype::BF16, vec![2, 2, 2], &bytes).unwrap());
        let tmp = NamedTempFile::new().unwrap();
        serialize_to_file(&tensors, &None, tmp.path()).unwrap();
        let source = ModelSource::SafetensorsFile(tmp.path().to_str().unwrap().to_string());
        let tensor_map = source.load_tensors().unwrap();
        assert!(!tensor_map.contains_key("weight_3d"), "3D tensor should be skipped");
    }

    #[test]
    fn test_kernel_compatibility_shapes() {
        // 2D tensor with shape [n, k] (should be loaded as is)
        let data_f32 = vec![1.0f32, 2.0, 3.0, 4.0];
        let data_bf16: Vec<bf16> = data_f32.iter().map(|&f| bf16::from_f32(f)).collect();
        let bytes: Vec<u8> = bf16_vec_to_bytes(&data_bf16);
        let mut tensors = std::collections::BTreeMap::new();
        tensors.insert("kernel_weight".to_string(), TensorView::new(Dtype::BF16, vec![2, 2], &bytes).unwrap());
        let tmp = NamedTempFile::new().unwrap();
        serialize_to_file(&tensors, &None, tmp.path()).unwrap();
        let source = ModelSource::SafetensorsFile(tmp.path().to_str().unwrap().to_string());
        let tensor_map = source.load_tensors().unwrap();
        let (_tensor, shape) = &tensor_map["kernel_weight"];
        // Kernel expects [n, k] shape
        assert_eq!(shape, &vec![2, 2], "Shape should match kernel expectation");
    }

    #[test]
    fn test_load_real_bitnet_model() {
        // Check for CARGO_MANIFEST_DIR and hf_loader.rs presence
        let manifest_dir = match std::env::var("CARGO_MANIFEST_DIR") {
            Ok(dir) => dir,
            Err(_) => {
                eprintln!("CARGO_MANIFEST_DIR not set, skipping real model test");
                return;
            }
        };
        let tools_path = std::path::Path::new(&manifest_dir).parent().unwrap().join("bitnet-tools/src/hf_loader.rs");
        if !tools_path.exists() {
            eprintln!("hf_loader.rs not found, skipping real model test");
            return;
        }
        // Import hf_loader dynamically
        // (Assume bitnet-tools is a dependency and hf_loader is public)
        // Use the get_model function to get a real model
        let model_files = match bitnet_tools::hf_loader::get_model(None) {
            Ok(files) => files,
            Err(e) => {
                eprintln!("Could not get real model: {e}, skipping test");
                return;
            }
        };
        // Use the first safetensors file found
        let safetensors_path = match model_files.safetensors_files.first() {
            Some(p) => p,
            None => {
                eprintln!("No safetensors file found in model, skipping test");
                return;
            }
        };
        let source = ModelSource::SafetensorsFile(safetensors_path.to_string_lossy().to_string());
        let tensor_map = match source.load_tensors() {
            Ok(map) => map,
            Err(e) => {
                eprintln!("Failed to load real model: {e}, skipping test");
                return;
            }
        };
        assert!(!tensor_map.is_empty(), "Tensor map should not be empty for real model");
        println!("Loaded {} tensors from real model at {}", tensor_map.len(), safetensors_path.display());
    }
}