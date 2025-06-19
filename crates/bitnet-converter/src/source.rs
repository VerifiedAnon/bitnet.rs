// File: E:\Desktop\Bitnet rs\crates\bitnet-converter\src\source.rs
// --- FULL REPLACEMENT ---

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use bytemuck;
use half::bf16;

/// Map from tensor name to (data, shape)
pub type TensorMap = HashMap<String, (Vec<f32>, Vec<usize>)>;

pub enum ModelSource {
    SafetensorsFile(String),
}

impl ModelSource {
    pub fn load_tensors(&self) -> Result<TensorMap, Box<dyn std::error::Error>> {
        match self {
            ModelSource::SafetensorsFile(path) => load_safetensors_mmap(path.as_ref()),
        }
    }
}

fn load_safetensors_mmap(path: &Path) -> Result<TensorMap, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let safetensors = SafeTensors::deserialize(&mmap)?;
    let mut map = TensorMap::new();

    for name in safetensors.names() {
        let tensor_view = safetensors.tensor(name)?;

        if tensor_view.dtype() != Dtype::BF16 {
            continue;
        }

        let original_shape = tensor_view.shape();
        let shape: Vec<usize> = if original_shape.len() == 1 {
            // Promote 1D tensors to [1, N] for consistency
            vec![1, original_shape[0]]
        } else if original_shape.len() == 2 {
            original_shape.to_vec()
        } else {
            log::warn!("Skipping tensor '{}' with unsupported shape {:?}", name, original_shape);
            continue;
        };

        let bf16_bytes = tensor_view.data();
        // SAFETY: BF16 is stored as u16 in safetensors, so we can cast &[u8] to &[u16]
        let u16_slice = bytemuck::cast_slice::<u8, u16>(bf16_bytes);
        let f32_vec: Vec<f32> = u16_slice.iter().map(|&bits| bf16::from_bits(bits).to_f32()).collect();

        map.insert(name.to_string(), (f32_vec, shape));
    }
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
        assert_eq!(tensor_2d, &data_f32_2d);
        
        // Assert 1D tensor was loaded and promoted correctly
        assert!(tensor_map.contains_key("weight_1d"), "1D tensor should be loaded");
        let (tensor_1d, shape_1d) = &tensor_map["weight_1d"];
        assert_eq!(shape_1d, &vec![1, 2], "1D tensor should be promoted to 2D");
        assert_eq!(tensor_1d, &data_f32_1d);

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
}