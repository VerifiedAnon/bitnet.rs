use std::fs::File;
use std::io::{Read, Write};
use tempfile::{NamedTempFile, tempdir};
use bincode::serde::{encode_to_vec, decode_from_slice};
use bincode::config::standard;

use bitnet_converter::packer::{
    BitNetModelRecord, TransformerBlockRecord, AttentionRecord, FeedForwardRecord, BitLinearRecord, RmsNormRecord, EmbeddingRecord, ModelMetadata
};

// Test 1: Serialize and deserialize a full BitNetModelRecord
#[test]
fn test_full_model_serialization() {
    // Create dummy model
    let embedding = EmbeddingRecord { weight: vec![1.0, 2.0, 3.0, 4.0], shape: vec![2, 2] };
    let norm = RmsNormRecord { weight: vec![0.1, 0.2], shape: vec![2] };
    let lm_head = EmbeddingRecord { weight: vec![5.0, 6.0, 7.0, 8.0], shape: vec![2, 2] };
    let block = TransformerBlockRecord {
        attention: AttentionRecord {
            wqkv: BitLinearRecord {
                packed_weights: vec![123, 456],
                weight_scales: vec![0.5, 0.6],
                in_features: 2,
                out_features: 2,
            },
            o_proj: BitLinearRecord {
                packed_weights: vec![789, 1011],
                weight_scales: vec![0.7, 0.8],
                in_features: 2,
                out_features: 2,
            },
        },
        feed_forward: FeedForwardRecord {
            w13: BitLinearRecord {
                packed_weights: vec![111, 222],
                weight_scales: vec![0.9, 1.0],
                in_features: 2,
                out_features: 2,
            },
            w2: BitLinearRecord {
                packed_weights: vec![333, 444],
                weight_scales: vec![1.1, 1.2],
                in_features: 2,
                out_features: 2,
            },
        },
        attention_norm: RmsNormRecord { weight: vec![0.3, 0.4], shape: vec![2] },
        ffn_norm: RmsNormRecord { weight: vec![0.5, 0.6], shape: vec![2] },
    };
    let model = BitNetModelRecord {
        embedding,
        blocks: vec![block],
        norm,
        lm_head,
        metadata: ModelMetadata {
            num_layers: 1,
            vocab_size: 2,
            hidden_size: 2,
            conversion_timestamp: 0,
        },
    };
    // Serialize to file
    let file = NamedTempFile::new().unwrap();
    let path = file.path();
    let mut f = File::create(path).unwrap();
    let bytes = encode_to_vec(&model, standard()).unwrap();
    f.write_all(&bytes).unwrap();
    // Deserialize from file
    let mut f = File::open(path).unwrap();
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    let (loaded, _): (BitNetModelRecord, usize) = decode_from_slice(&buf, standard()).unwrap();
    // Check equality (basic fields)
    assert_eq!(loaded.embedding.weight, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(loaded.blocks[0].attention.wqkv.packed_weights, vec![123, 456]);
    assert_eq!(loaded.lm_head.weight, vec![5.0, 6.0, 7.0, 8.0]);
    println!("Test 1 Passed: Full model serialization/deserialization works.");
}

// Test 2: Serialize and deserialize a single TransformerBlockRecord (streaming)
#[test]
fn test_block_streaming_serialization() {
    let block = TransformerBlockRecord {
        attention: AttentionRecord {
            wqkv: BitLinearRecord {
                packed_weights: vec![1, 2],
                weight_scales: vec![0.1, 0.2],
                in_features: 2,
                out_features: 2,
            },
            o_proj: BitLinearRecord {
                packed_weights: vec![3, 4],
                weight_scales: vec![0.3, 0.4],
                in_features: 2,
                out_features: 2,
            },
        },
        feed_forward: FeedForwardRecord {
            w13: BitLinearRecord {
                packed_weights: vec![5, 6],
                weight_scales: vec![0.5, 0.6],
                in_features: 2,
                out_features: 2,
            },
            w2: BitLinearRecord {
                packed_weights: vec![7, 8],
                weight_scales: vec![0.7, 0.8],
                in_features: 2,
                out_features: 2,
            },
        },
        attention_norm: RmsNormRecord { weight: vec![0.9, 1.0], shape: vec![2] },
        ffn_norm: RmsNormRecord { weight: vec![1.1, 1.2], shape: vec![2] },
    };
    let file = NamedTempFile::new().unwrap();
    let path = file.path();
    let mut f = File::create(path).unwrap();
    let bytes = encode_to_vec(&block, standard()).unwrap();
    f.write_all(&bytes).unwrap();
    let mut f = File::open(path).unwrap();
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    let (loaded, _): (TransformerBlockRecord, usize) = decode_from_slice(&buf, standard()).unwrap();
    assert_eq!(loaded.attention.wqkv.packed_weights, vec![1, 2]);
    assert_eq!(loaded.feed_forward.w2.packed_weights, vec![7, 8]);
    println!("Test 2 Passed: Block streaming serialization/deserialization works.");
}

// Test 3: Save each block of a model to a separate file, then load and reassemble
#[test]
fn test_streaming_multiple_blocks() {
    let dir = tempdir().unwrap();
    let mut blocks = Vec::new();
    for i in 0..3 {
        let block = TransformerBlockRecord {
            attention: AttentionRecord {
                wqkv: BitLinearRecord {
                    packed_weights: vec![i, i+1],
                    weight_scales: vec![i as f32, (i+1) as f32],
                    in_features: 2,
                    out_features: 2,
                },
                o_proj: BitLinearRecord {
                    packed_weights: vec![i+2, i+3],
                    weight_scales: vec![(i+2) as f32, (i+3) as f32],
                    in_features: 2,
                    out_features: 2,
                },
            },
            feed_forward: FeedForwardRecord {
                w13: BitLinearRecord {
                    packed_weights: vec![i+4, i+5],
                    weight_scales: vec![(i+4) as f32, (i+5) as f32],
                    in_features: 2,
                    out_features: 2,
                },
                w2: BitLinearRecord {
                    packed_weights: vec![i+6, i+7],
                    weight_scales: vec![(i+6) as f32, (i+7) as f32],
                    in_features: 2,
                    out_features: 2,
                },
            },
            attention_norm: RmsNormRecord { weight: vec![i as f32], shape: vec![1] },
            ffn_norm: RmsNormRecord { weight: vec![(i+1) as f32], shape: vec![1] },
        };
        let block_path = dir.path().join(format!("block_{}.bin", i));
        let mut f = File::create(&block_path).unwrap();
        let bytes = encode_to_vec(&block, standard()).unwrap();
        f.write_all(&bytes).unwrap();
        blocks.push(block_path);
    }
    // Load and reassemble
    let mut loaded_blocks = Vec::new();
    for block_path in &blocks {
        let mut f = File::open(block_path).unwrap();
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).unwrap();
        let (block, _): (TransformerBlockRecord, usize) = decode_from_slice(&buf, standard()).unwrap();
        loaded_blocks.push(block);
    }
    assert_eq!(loaded_blocks.len(), 3);
    assert_eq!(loaded_blocks[1].attention.wqkv.packed_weights, vec![1, 2]);
    assert_eq!(loaded_blocks[2].feed_forward.w2.packed_weights, vec![8, 9]);
    println!("Test 3 Passed: Streaming multiple blocks as separate files works.");
}

