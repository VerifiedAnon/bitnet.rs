# ATTENTION_GPU_TESTS Test Report

> Generated on: 2025-07-01 16:20:10

## Test Results

| No. | Test Name | Status | Time Taken | Error Message |
|:---:|:----------|:------:|:----------:|:-------------|
|  1 | Attention Cross Device Consistency                 | ✅ Pass |   1.15 sec |             |
|  2 | Attention Edge Cases                               | ✅ Pass |  523.00 ms |             |
|  3 | Attention Error Handling                           | ✅ Pass |  420.00 ms |             |
|  4 | Attention GPU Correctness                          | ✅ Pass |  718.00 ms |             |
|  5 | Attention GPU Correctness All Shapes               | ✅ Pass |  532.00 ms |             |
|  6 | Attention Memory Safety                            | ✅ Pass |  433.00 ms |             |
|  7 | Attention Performance                              | ✅ Pass |  435.00 ms |             |
|  8 | Attention Shader Compilation                       | ✅ Pass |   2.16 sec |             |

<details>
<summary>📝 View Full Log Dump</summary>

```
[2025-07-01, 16:20:01.878] -> Running test_attention_gpu_correctness...
[2025-07-01, 16:20:02.452] -> Device: NVIDIA GeForce RTX 2070 SUPER (Backend: Vulkan, Type: DiscreteGpu)
[2025-07-01, 16:20:02.452] -> Limits: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_binding_array_elements_per_shader_stage: 0, max_binding_array_sampler_elements_per_shader_stage: 0, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 268435456, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-07-01, 16:20:02.542] -> Testing shape: batch=1, heads=2, seq_len=4, head_dim=8
[2025-07-01, 16:20:02.555] -> CPU output sample: [0.2544937, -0.7783191, 0.10527706, -0.34261632], GPU output sample: [0.2544937, -0.7783191, 0.10527706, -0.34261632]
[2025-07-01, 16:20:02.556] -> Testing shape: batch=2, heads=4, seq_len=8, head_dim=16
[2025-07-01, 16:20:02.559] -> CPU output sample: [0.035744905, 0.89135313, 0.20639348, 0.27892184], GPU output sample: [0.035744905, 0.89135313, 0.20639348, 0.27892184]
[2025-07-01, 16:20:02.559] -> Testing shape: batch=1, heads=1, seq_len=16, head_dim=32
[2025-07-01, 16:20:02.561] -> CPU output sample: [0.92878485, -0.17276359, -0.6558993, 0.61100173], GPU output sample: [0.92878485, -0.17276359, -0.6558993, 0.61100173]
[2025-07-01, 16:20:02.561] -> Testing shape: batch=3, heads=2, seq_len=6, head_dim=4
[2025-07-01, 16:20:02.563] -> CPU output sample: [-0.7463832, -0.66046405, -0.7713177, 0.45896006], GPU output sample: [-0.7463832, -0.66046405, -0.7713177, 0.45896006]
[2025-07-01, 16:20:02.563] -> [PASS] Attention GPU Correctness passed.
[2025-07-01, 16:20:02.667] -> Device: NVIDIA GeForce RTX 2070 SUPER (Backend: Vulkan, Type: DiscreteGpu)
[2025-07-01, 16:20:02.667] -> Limits: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_binding_array_elements_per_shader_stage: 0, max_binding_array_sampler_elements_per_shader_stage: 0, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 268435456, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-07-01, 16:20:02.721] -> [PASS] Backend Backends(VULKAN) compiled successfully
[2025-07-01, 16:20:03.169] -> Device: NVIDIA GeForce RTX 2070 SUPER (Backend: Dx12, Type: DiscreteGpu)
[2025-07-01, 16:20:03.169] -> Limits: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_binding_array_elements_per_shader_stage: 0, max_binding_array_sampler_elements_per_shader_stage: 0, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 268435456, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-07-01, 16:20:03.810] -> [PASS] Backend Backends(DX12) compiled successfully
[2025-07-01, 16:20:03.876] -> Device: NVIDIA GeForce RTX 2070 SUPER/PCIe/SSE2 (Backend: Gl, Type: Other)
[2025-07-01, 16:20:03.876] -> Limits: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_binding_array_elements_per_shader_stage: 0, max_binding_array_sampler_elements_per_shader_stage: 0, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 268435456, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-07-01, 16:20:04.753] -> [PASS] Backend Backends(GL) compiled successfully
[2025-07-01, 16:20:04.756] -> [PASS] Attention Shader Compilation passed.
[2025-07-01, 16:20:05.109] -> Device: NVIDIA GeForce RTX 2070 SUPER (Backend: Vulkan, Type: DiscreteGpu)
[2025-07-01, 16:20:05.109] -> Limits: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_binding_array_elements_per_shader_stage: 0, max_binding_array_sampler_elements_per_shader_stage: 0, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 268435456, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-07-01, 16:20:05.115] -> Testing shape: batch=1, heads=2, seq_len=4, head_dim=8, pattern=random
[2025-07-01, 16:20:05.129] -> CPU output sample: [0.2544937, -0.7783191, 0.10527706, -0.34261632], GPU output sample: [0.2544937, -0.7783191, 0.10527706, -0.34261632]
[2025-07-01, 16:20:05.129] -> Testing shape: batch=1, heads=2, seq_len=4, head_dim=8, pattern=all_zero
[2025-07-01, 16:20:05.131] -> CPU output sample: [0.0, 0.0, 0.0, 0.0], GPU output sample: [0.0, 0.0, 0.0, 0.0]
[2025-07-01, 16:20:05.131] -> Testing shape: batch=1, heads=2, seq_len=4, head_dim=8, pattern=all_one
[2025-07-01, 16:20:05.133] -> CPU output sample: [1.0, 1.0, 1.0, 1.0], GPU output sample: [1.0, 1.0, 1.0, 1.0]
[2025-07-01, 16:20:05.133] -> Testing shape: batch=2, heads=4, seq_len=8, head_dim=16, pattern=random
[2025-07-01, 16:20:05.136] -> CPU output sample: [-0.54744935, 0.8403597, 0.95048094, -0.31572366], GPU output sample: [-0.54744935, 0.8403597, 0.95048094, -0.31572366]
[2025-07-01, 16:20:05.137] -> Testing shape: batch=2, heads=4, seq_len=8, head_dim=16, pattern=all_zero
[2025-07-01, 16:20:05.139] -> CPU output sample: [0.0, 0.0, 0.0, 0.0], GPU output sample: [0.0, 0.0, 0.0, 0.0]
[2025-07-01, 16:20:05.139] -> Testing shape: batch=2, heads=4, seq_len=8, head_dim=16, pattern=all_one
[2025-07-01, 16:20:05.141] -> CPU output sample: [1.0, 1.0, 1.0, 1.0], GPU output sample: [1.0, 1.0, 1.0, 1.0]
[2025-07-01, 16:20:05.141] -> Testing shape: batch=1, heads=1, seq_len=16, head_dim=32, pattern=random
[2025-07-01, 16:20:05.144] -> CPU output sample: [0.3077283, -0.3268659, -0.17309237, -0.19555545], GPU output sample: [0.3077283, -0.3268659, -0.17309237, -0.19555545]
[2025-07-01, 16:20:05.144] -> Testing shape: batch=1, heads=1, seq_len=16, head_dim=32, pattern=all_zero
[2025-07-01, 16:20:05.146] -> CPU output sample: [0.0, 0.0, 0.0, 0.0], GPU output sample: [0.0, 0.0, 0.0, 0.0]
[2025-07-01, 16:20:05.146] -> Testing shape: batch=1, heads=1, seq_len=16, head_dim=32, pattern=all_one
[2025-07-01, 16:20:05.148] -> CPU output sample: [1.0, 1.0, 1.0, 1.0], GPU output sample: [1.0, 1.0, 1.0, 1.0]
[2025-07-01, 16:20:05.148] -> Testing shape: batch=3, heads=2, seq_len=6, head_dim=4, pattern=random
[2025-07-01, 16:20:05.149] -> CPU output sample: [-0.7184231, 0.37015247, 0.5461695, -0.05128503], GPU output sample: [-0.7184231, 0.37015247, 0.5461695, -0.05128503]
[2025-07-01, 16:20:05.150] -> Testing shape: batch=3, heads=2, seq_len=6, head_dim=4, pattern=all_zero
[2025-07-01, 16:20:05.152] -> CPU output sample: [0.0, 0.0, 0.0, 0.0], GPU output sample: [0.0, 0.0, 0.0, 0.0]
[2025-07-01, 16:20:05.152] -> Testing shape: batch=3, heads=2, seq_len=6, head_dim=4, pattern=all_one
[2025-07-01, 16:20:05.154] -> CPU output sample: [1.0, 1.0, 1.0, 1.0], GPU output sample: [1.0, 1.0, 1.0, 1.0]
[2025-07-01, 16:20:05.154] -> Testing shape: batch=1, heads=8, seq_len=1, head_dim=8, pattern=random
[2025-07-01, 16:20:05.156] -> CPU output sample: [-0.52921677, -0.24482274, -0.5652766, 0.053202152], GPU output sample: [-0.52921677, -0.24482274, -0.5652766, 0.053202152]
[2025-07-01, 16:20:05.156] -> Testing shape: batch=1, heads=8, seq_len=1, head_dim=8, pattern=all_zero
[2025-07-01, 16:20:05.157] -> CPU output sample: [0.0, 0.0, 0.0, 0.0], GPU output sample: [0.0, 0.0, 0.0, 0.0]
[2025-07-01, 16:20:05.157] -> Testing shape: batch=1, heads=8, seq_len=1, head_dim=8, pattern=all_one
[2025-07-01, 16:20:05.159] -> CPU output sample: [1.0, 1.0, 1.0, 1.0], GPU output sample: [1.0, 1.0, 1.0, 1.0]
[2025-07-01, 16:20:05.159] -> Testing shape: batch=2, heads=2, seq_len=32, head_dim=8, pattern=random
[2025-07-01, 16:20:05.163] -> CPU output sample: [-0.18887901, -0.89565444, -0.5502329, -0.40584397], GPU output sample: [-0.18887901, -0.89565444, -0.5502329, -0.40584397]
[2025-07-01, 16:20:05.163] -> Testing shape: batch=2, heads=2, seq_len=32, head_dim=8, pattern=all_zero
[2025-07-01, 16:20:05.166] -> CPU output sample: [0.0, 0.0, 0.0, 0.0], GPU output sample: [0.0, 0.0, 0.0, 0.0]
[2025-07-01, 16:20:05.167] -> Testing shape: batch=2, heads=2, seq_len=32, head_dim=8, pattern=all_one
[2025-07-01, 16:20:05.170] -> CPU output sample: [1.0, 1.0, 1.0, 1.0], GPU output sample: [1.0, 1.0, 1.0, 1.0]
[2025-07-01, 16:20:05.171] -> Testing shape: batch=1, heads=4, seq_len=64, head_dim=16, pattern=random
[2025-07-01, 16:20:05.187] -> CPU output sample: [-0.77418804, 0.39468122, 0.9562905, 0.06734681], GPU output sample: [-0.77418804, 0.39468122, 0.9562905, 0.06734681]
[2025-07-01, 16:20:05.187] -> Testing shape: batch=1, heads=4, seq_len=64, head_dim=16, pattern=all_zero
[2025-07-01, 16:20:05.200] -> CPU output sample: [0.0, 0.0, 0.0, 0.0], GPU output sample: [0.0, 0.0, 0.0, 0.0]
[2025-07-01, 16:20:05.200] -> Testing shape: batch=1, heads=4, seq_len=64, head_dim=16, pattern=all_one
[2025-07-01, 16:20:05.213] -> CPU output sample: [1.0, 1.0, 1.0, 1.0], GPU output sample: [1.0, 1.0, 1.0, 1.0]
[2025-07-01, 16:20:05.214] -> Testing shape: batch=2, heads=1, seq_len=128, head_dim=4, pattern=random
[2025-07-01, 16:20:05.224] -> CPU output sample: [-0.24976277, -0.28497434, -0.76331186, -0.14553022], GPU output sample: [-0.24976277, -0.28497434, -0.76331186, -0.14553022]
[2025-07-01, 16:20:05.224] -> Testing shape: batch=2, heads=1, seq_len=128, head_dim=4, pattern=all_zero
[2025-07-01, 16:20:05.234] -> CPU output sample: [0.0, 0.0, 0.0, 0.0], GPU output sample: [0.0, 0.0, 0.0, 0.0]
[2025-07-01, 16:20:05.234] -> Testing shape: batch=2, heads=1, seq_len=128, head_dim=4, pattern=all_one
[2025-07-01, 16:20:05.244] -> CPU output sample: [1.0, 1.0, 1.0, 1.0], GPU output sample: [1.0, 1.0, 1.0, 1.0]
[2025-07-01, 16:20:05.245] -> [PASS] Attention GPU Correctness All Shapes passed.
[2025-07-01, 16:20:05.639] -> Device: NVIDIA GeForce RTX 2070 SUPER (Backend: Vulkan, Type: DiscreteGpu)
[2025-07-01, 16:20:05.639] -> Limits: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_binding_array_elements_per_shader_stage: 0, max_binding_array_sampler_elements_per_shader_stage: 0, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 268435456, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-07-01, 16:20:05.646] -> Testing shape: batch=1, heads=1, seq_len=1, head_dim=1
[2025-07-01, 16:20:05.660] -> CPU output sample: [0.46811962], GPU output sample: [0.46811962]
[2025-07-01, 16:20:05.660] -> Testing shape: batch=1, heads=1, seq_len=2, head_dim=1
[2025-07-01, 16:20:05.662] -> CPU output sample: [-0.17120576, -0.67247736], GPU output sample: [-0.17120576, -0.67247736]
[2025-07-01, 16:20:05.662] -> Testing shape: batch=1, heads=1, seq_len=512, head_dim=8
[2025-07-01, 16:20:05.765] -> CPU output sample: [-0.6651964, -0.4274261, -0.60916066, 0.6251788], GPU output sample: [-0.6651964, -0.4274261, -0.60916066, 0.6251788]
[2025-07-01, 16:20:05.766] -> Testing shape: batch=1, heads=8, seq_len=4, head_dim=8
[2025-07-01, 16:20:05.769] -> CPU output sample: [0.5229726, -0.81027365, 0.038002014, 0.5505018], GPU output sample: [0.5229726, -0.81027365, 0.038002014, 0.5505018]
[2025-07-01, 16:20:05.769] -> Testing shape: batch=2, heads=2, seq_len=3, head_dim=5
[2025-07-01, 16:20:05.770] -> CPU output sample: [0.3065393, 0.5816591, 0.045762062, 0.38640118], GPU output sample: [0.3065393, 0.5816591, 0.045762062, 0.38640118]
[2025-07-01, 16:20:05.770] -> Testing shape: batch=1, heads=2, seq_len=7, head_dim=3
[2025-07-01, 16:20:05.772] -> CPU output sample: [-0.4954939, 0.31177783, 0.9220915, 0.8460772], GPU output sample: [-0.4954939, 0.31177783, 0.9220915, 0.8460772]
[2025-07-01, 16:20:05.772] -> [PASS] Attention Edge Cases passed.
[2025-07-01, 16:20:06.163] -> Device: NVIDIA GeForce RTX 2070 SUPER (Backend: Vulkan, Type: DiscreteGpu)
[2025-07-01, 16:20:06.164] -> Limits: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_binding_array_elements_per_shader_stage: 0, max_binding_array_sampler_elements_per_shader_stage: 0, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 268435456, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-07-01, 16:20:06.196] -> Testing shape: batch=1, heads=2, seq_len=8, head_dim=64
[2025-07-01, 16:20:06.198] -> Testing shape: batch=1, heads=2, seq_len=16, head_dim=64
[2025-07-01, 16:20:06.200] -> Testing shape: batch=1, heads=2, seq_len=24, head_dim=64
[2025-07-01, 16:20:06.203] -> Testing shape: batch=1, heads=2, seq_len=32, head_dim=64
[2025-07-01, 16:20:06.205] -> Testing shape: batch=1, heads=2, seq_len=40, head_dim=64
[2025-07-01, 16:20:06.207] -> [PASS] Attention Memory Safety passed.
[2025-07-01, 16:20:06.319] -> Device: NVIDIA GeForce RTX 2070 SUPER (Backend: Vulkan, Type: DiscreteGpu)
[2025-07-01, 16:20:06.319] -> Limits: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_binding_array_elements_per_shader_stage: 0, max_binding_array_sampler_elements_per_shader_stage: 0, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 268435456, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-07-01, 16:20:06.771] -> Device: NVIDIA GeForce RTX 2070 SUPER (Backend: Dx12, Type: DiscreteGpu)
[2025-07-01, 16:20:06.771] -> Limits: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_binding_array_elements_per_shader_stage: 0, max_binding_array_sampler_elements_per_shader_stage: 0, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 268435456, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-07-01, 16:20:07.384] -> Device: NVIDIA GeForce RTX 2070 SUPER/PCIe/SSE2 (Backend: Gl, Type: Other)
[2025-07-01, 16:20:07.385] -> Limits: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_binding_array_elements_per_shader_stage: 0, max_binding_array_sampler_elements_per_shader_stage: 0, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 268435456, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-07-01, 16:20:07.401] -> CPU output sample: [0.3, 0.3, 0.3, 0.3], GPU output sample: [0.3, 0.3, 0.3, 0.3]
[2025-07-01, 16:20:07.401] -> CPU output sample: [0.3, 0.3, 0.3, 0.3], GPU output sample: [0.3, 0.3, 0.3, 0.3]
[2025-07-01, 16:20:07.401] -> [PASS] Attention Cross Device Consistency passed.
[2025-07-01, 16:20:07.749] -> Device: NVIDIA GeForce RTX 2070 SUPER (Backend: Vulkan, Type: DiscreteGpu)
[2025-07-01, 16:20:07.749] -> Limits: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_binding_array_elements_per_shader_stage: 0, max_binding_array_sampler_elements_per_shader_stage: 0, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 268435456, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-07-01, 16:20:07.770] -> [WARN] Invalid input shape did not error as expected
[2025-07-01, 16:20:07.770] -> Result: Ok([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
[2025-07-01, 16:20:08.175] -> Device: NVIDIA GeForce RTX 2070 SUPER (Backend: Vulkan, Type: DiscreteGpu)
[2025-07-01, 16:20:08.175] -> Limits: Limits { max_texture_dimension_1d: 8192, max_texture_dimension_2d: 8192, max_texture_dimension_3d: 2048, max_texture_array_layers: 256, max_bind_groups: 4, max_bindings_per_bind_group: 1000, max_dynamic_uniform_buffers_per_pipeline_layout: 8, max_dynamic_storage_buffers_per_pipeline_layout: 4, max_sampled_textures_per_shader_stage: 16, max_samplers_per_shader_stage: 16, max_storage_buffers_per_shader_stage: 8, max_storage_textures_per_shader_stage: 4, max_uniform_buffers_per_shader_stage: 12, max_binding_array_elements_per_shader_stage: 0, max_binding_array_sampler_elements_per_shader_stage: 0, max_uniform_buffer_binding_size: 65536, max_storage_buffer_binding_size: 134217728, max_vertex_buffers: 8, max_buffer_size: 268435456, max_vertex_attributes: 16, max_vertex_buffer_array_stride: 2048, min_uniform_buffer_offset_alignment: 256, min_storage_buffer_offset_alignment: 256, max_inter_stage_shader_components: 60, max_color_attachments: 8, max_color_attachment_bytes_per_sample: 32, max_compute_workgroup_storage_size: 16384, max_compute_invocations_per_workgroup: 256, max_compute_workgroup_size_x: 256, max_compute_workgroup_size_y: 256, max_compute_workgroup_size_z: 64, max_compute_workgroups_per_dimension: 65535, min_subgroup_size: 0, max_subgroup_size: 0, max_push_constant_size: 0, max_non_sampler_bindings: 1000000 }
[2025-07-01, 16:20:08.181] -> Testing shape: batch=1, heads=2, seq_len=128, head_dim=32
[2025-07-01, 16:20:08.197] -> Perf: batch=1, heads=2, seq_len=128, head_dim=32, time=16.1738ms
[2025-07-01, 16:20:08.198] -> Testing shape: batch=2, heads=4, seq_len=64, head_dim=64
[2025-07-01, 16:20:08.205] -> Perf: batch=2, heads=4, seq_len=64, head_dim=64, time=6.1693ms
[2025-07-01, 16:20:08.205] -> Testing shape: batch=4, heads=8, seq_len=32, head_dim=128
[2025-07-01, 16:20:08.215] -> Perf: batch=4, heads=8, seq_len=32, head_dim=128, time=7.8284ms
[2025-07-01, 16:20:08.215] -> [PASS] Attention Performance passed.
[2025-07-01, 16:20:01.877] -> STARTING KERNEL TEST SUITE
```

</details>


## Summary

### Test Statistics

- **Total Tests:** 8
- **Passed:** 8
- **Failed:** 0

### Timing Information

- **Total Time:** 6.37 sec
- **Average Time:** 796.00 ms

### Status

✅ All tests passed successfully!

---

_Report generated by BitNet Test Framework_
