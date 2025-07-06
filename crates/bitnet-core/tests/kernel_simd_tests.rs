//! Validation tests for the CPU SIMD kernels.
//!
//! This suite ensures that the `unsafe` SIMD implementations for AVX2 and NEON
//! produce bit-for-bit identical results to the safe, scalar reference implementation.
//! This is crucial for preventing correctness regressions and data corruption.

use bitnet_core::kernels::cpu;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

fn random_data(len: usize, min: i8, max: i8, seed: u64) -> Vec<i8> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.random_range(min..=max)).collect()
}

#[test]
fn test_qgemm_kernels_rigorous() {
    let configs = [
        (1, 16, 1),   // Small, aligned
        (2, 32, 2),   // Small, AVX2/NEON block
        (4, 64, 4),   // Medium, aligned
        (3, 70, 5),   // Unaligned in_features
        (8, 128, 8),  // Large, aligned
        (8, 70, 5),   // Large, unaligned
    ];
    let seeds = [42, 99, 123, 2024];
    for &(batch_size, in_features, out_features) in &configs {
        for &seed in &seeds {
            // Random
            let activations = random_data(batch_size * in_features, -127, 127, seed as u64);
            let weights = random_data(out_features * in_features, -1, 1, seed as u64 + 1);
            let activation_scales: Vec<f32> = (0..batch_size).map(|i| 1.0 + i as f32 * 0.01).collect();
            let weight_scales: Vec<f32> = (0..out_features).map(|i| 1.0 + i as f32 * 0.01).collect();
            let weight_rows: Vec<Vec<i8>> = weights.chunks(in_features).map(|c| c.to_vec()).collect();
            let (packed_weights, _) = bitnet_core::kernels::pack_ternary_weights(&weight_rows).unwrap();
            // All zeros
            let zeros = vec![0i8; batch_size * in_features];
            // All ones
            let ones = vec![1i8; batch_size * in_features];
            // Test cases
            let cases = vec![
                ("random", activations.clone()),
                ("zeros", zeros.clone()),
                ("ones", ones.clone()),
            ];
            for (label, acts) in cases {
                // Scalar reference
                let scalar = cpu::scalar::qgemm_lut_scalar(
                    &acts, &packed_weights, &activation_scales, &weight_scales,
                    batch_size, in_features, out_features
                ).unwrap();
                // Dispatcher (should match scalar)
                let dispatch = cpu::execute(
                    &acts, &packed_weights, &activation_scales, &weight_scales,
                    batch_size, in_features, out_features
                ).unwrap();
                assert_eq!(scalar, dispatch, "Dispatch mismatch: {} bs={} in={} out={} seed={}", label, batch_size, in_features, out_features, seed);
                // AVX2 (if available)
                #[cfg(target_arch = "x86_64")]
                if is_x86_feature_detected!("avx2") {
                    let avx2 = unsafe {
                        cpu::x86::qgemm_lut_avx2(
                            &acts, &packed_weights, &activation_scales, &weight_scales,
                            batch_size, in_features, out_features
                        ).unwrap()
                    };
                    assert_eq!(scalar, avx2, "AVX2 mismatch: {} bs={} in={} out={} seed={}", label, batch_size, in_features, out_features, seed);
                }
                // NEON (if available)
                #[cfg(target_arch = "aarch64")]
                if is_aarch64_feature_detected!("neon") {
                    let neon = unsafe {
                        cpu::arm::qgemm_lut_neon(
                            &acts, &packed_weights, &activation_scales, &weight_scales,
                            batch_size, in_features, out_features
                        ).unwrap()
                    };
                    assert_eq!(scalar, neon, "NEON mismatch: {} bs={} in={} out={} seed={}", label, batch_size, in_features, out_features, seed);
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2_debug {
    use super::*;
    use bitnet_core::kernels::cpu;
    use std::arch::is_x86_feature_detected;

    fn unpack_weights_scalar(packed: &[u32], in_features: usize) -> Vec<i8> {
        let mut weights = vec![0i8; in_features];
        let packed_per_row = (in_features + 15) / 16;
        for k in 0..in_features {
            let packed_idx = k / 16;
            let bit_idx = (k % 16) * 2;
            let bits = ((packed[packed_idx] >> bit_idx) & 0b11) as u8;
            weights[k] = match bits {
                1 => 1,
                2 => -1,
                _ => 0,
            };
        }
        weights
    }

    #[test]
    fn test_weight_unpacking_matches_scalar() {
        let in_features = 33;
        let out_features = 1;
        let weights = super::random_data(out_features * in_features, -1, 1, 123);
        let weight_rows: Vec<Vec<i8>> = weights.chunks(in_features).map(|c| c.to_vec()).collect();
        let (packed_weights, _) = bitnet_core::kernels::pack_ternary_weights(&weight_rows).unwrap();
        let scalar_unpacked = unpack_weights_scalar(&packed_weights, in_features);
        // Simulate AVX2 block unpacking for k=0..in_features
        let mut avx2_unpacked = vec![0i8; in_features];
        let mut k = 0;
        while k + 31 < in_features {
            let mut block = [0i8; 32];
            for i in 0..2 {
                let packed = packed_weights[(k / 16) + i];
                for j in 0..16 {
                    let idx = i * 16 + j;
                    if k + idx >= in_features { break; }
                    let bits = ((packed >> (j * 2)) & 0b11) as u8;
                    block[idx] = match bits {
                        1 => 1,
                        2 => -1,
                        _ => 0,
                    };
                }
            }
            for i in 0..32 {
                if k + i < in_features {
                    avx2_unpacked[k + i] = block[i];
                }
            }
            k += 32;
        }
        // Scalar tail
        for kk in k..in_features {
            let packed_idx = kk / 16;
            let bit_idx = (kk % 16) * 2;
            let bits = ((packed_weights[packed_idx] >> bit_idx) & 0b11) as u8;
            avx2_unpacked[kk] = match bits {
                1 => 1,
                2 => -1,
                _ => 0,
            };
        }
        assert_eq!(scalar_unpacked, avx2_unpacked, "Weight unpacking mismatch between scalar and AVX2 logic");
    }

    fn run_avx2_vs_scalar(batch_size: usize, in_features: usize, out_features: usize, seed: u64, label: &str) {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("[SKIP] AVX2 not available");
            return;
        }
        let activations = super::random_data(batch_size * in_features, -127, 127, seed);
        let weights = super::random_data(out_features * in_features, -1, 1, seed + 1);
        let activation_scales: Vec<f32> = (0..batch_size).map(|i| 1.0 + i as f32 * 0.01).collect();
        let weight_scales: Vec<f32> = (0..out_features).map(|i| 1.0 + i as f32 * 0.01).collect();
        let weight_rows: Vec<Vec<i8>> = weights.chunks(in_features).map(|c| c.to_vec()).collect();
        let (packed_weights, _) = bitnet_core::kernels::pack_ternary_weights(&weight_rows).unwrap();
        let scalar = cpu::scalar::qgemm_lut_scalar(
            &activations, &packed_weights, &activation_scales, &weight_scales,
            batch_size, in_features, out_features
        ).unwrap();
        let avx2 = unsafe {
            cpu::x86::qgemm_lut_avx2(
                &activations, &packed_weights, &activation_scales, &weight_scales,
                batch_size, in_features, out_features
            ).unwrap()
        };
        if scalar != avx2 {
            eprintln!("[FAIL] AVX2 mismatch: {} bs={} in={} out={} seed={}", label, batch_size, in_features, out_features, seed);
            eprintln!("scalar: {:?}", scalar);
            eprintln!("avx2:   {:?}", avx2);
        }
        assert_eq!(scalar, avx2, "AVX2 mismatch: {} bs={} in={} out={} seed={}", label, batch_size, in_features, out_features, seed);
    }

    #[test]
    fn test_avx2_block_aligned() {
        run_avx2_vs_scalar(1, 32, 1, 42, "block_aligned");
        run_avx2_vs_scalar(2, 64, 2, 43, "block_aligned");
    }
    #[test]
    fn test_avx2_unaligned() {
        run_avx2_vs_scalar(1, 33, 1, 44, "unaligned");
        run_avx2_vs_scalar(2, 70, 2, 45, "unaligned");
    }
    #[test]
    fn test_avx2_all_zeros() {
        let batch_size = 2;
        let in_features = 32;
        let out_features = 2;
        let activations = vec![0i8; batch_size * in_features];
        let weights = vec![0i8; out_features * in_features];
        let activation_scales: Vec<f32> = (0..batch_size).map(|i| 1.0 + i as f32 * 0.01).collect();
        let weight_scales: Vec<f32> = (0..out_features).map(|i| 1.0 + i as f32 * 0.01).collect();
        let weight_rows: Vec<Vec<i8>> = weights.chunks(in_features).map(|c| c.to_vec()).collect();
        let (packed_weights, _) = bitnet_core::kernels::pack_ternary_weights(&weight_rows).unwrap();
        let scalar = cpu::scalar::qgemm_lut_scalar(
            &activations, &packed_weights, &activation_scales, &weight_scales,
            batch_size, in_features, out_features
        ).unwrap();
        let avx2 = unsafe {
            cpu::x86::qgemm_lut_avx2(
                &activations, &packed_weights, &activation_scales, &weight_scales,
                batch_size, in_features, out_features
            ).unwrap()
        };
        assert_eq!(scalar, avx2, "AVX2 all-zeros");
    }
    #[test]
    fn test_avx2_all_ones() {
        let batch_size = 2;
        let in_features = 32;
        let out_features = 2;
        let activations = vec![1i8; batch_size * in_features];
        let weights = vec![1i8; out_features * in_features];
        let activation_scales: Vec<f32> = (0..batch_size).map(|i| 1.0 + i as f32 * 0.01).collect();
        let weight_scales: Vec<f32> = (0..out_features).map(|i| 1.0 + i as f32 * 0.01).collect();
        let weight_rows: Vec<Vec<i8>> = weights.chunks(in_features).map(|c| c.to_vec()).collect();
        let (packed_weights, _) = bitnet_core::kernels::pack_ternary_weights(&weight_rows).unwrap();
        let scalar = cpu::scalar::qgemm_lut_scalar(
            &activations, &packed_weights, &activation_scales, &weight_scales,
            batch_size, in_features, out_features
        ).unwrap();
        let avx2 = unsafe {
            cpu::x86::qgemm_lut_avx2(
                &activations, &packed_weights, &activation_scales, &weight_scales,
                batch_size, in_features, out_features
            ).unwrap()
        };
        assert_eq!(scalar, avx2, "AVX2 all-ones");
    }
} 