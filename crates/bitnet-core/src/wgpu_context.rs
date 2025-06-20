// File: crates/bitnet-core/src/wgpu_context.rs
// --- NEW FILE ---

//! WGPU context management for BitNet operations.
//!
//! This module provides a shared WGPU context that can be used across
//! different components of BitNet. It handles device initialization,
//! adapter selection, and queue management.

use crate::error::BitNetError;
use std::sync::Arc;

/// A shared WGPU context for GPU operations.
///
/// # Fields
///
/// * `device` - The WGPU device for executing compute operations.
/// * `queue` - The command queue for submitting GPU commands.
#[derive(Clone)]
pub struct WgpuContext {
    /// The WGPU device, wrapped in Arc for thread-safe sharing.
    pub device: Arc<wgpu::Device>,
    /// The WGPU command queue, wrapped in Arc for thread-safe sharing.
    pub queue: Arc<wgpu::Queue>,
}

impl WgpuContext {
    /// Creates a new WGPU context.
    ///
    /// This function will:
    /// 1. Create a WGPU instance.
    /// 2. Request a high-performance adapter.
    /// 3. Create a device and queue.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No suitable adapter is found (`BitNetError::NoSuitableAdapter`).
    /// - Device creation fails (`BitNetError::RequestDeviceError`).
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use bitnet_core::error::BitNetError;
    /// # use bitnet_core::wgpu_context::WgpuContext;
    /// # async fn example() -> Result<(), BitNetError> {
    /// let context = WgpuContext::new().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn new() -> Result<Self, BitNetError> {
        let instance = wgpu::Instance::default();
        
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|_| BitNetError::NoSuitableAdapter)?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_wgpu_context_creation() {
        // This test validates that the WgpuContext can be created.
        // It's designed to be robust in environments with and without a GPU.
        let context_result = WgpuContext::new().await;

        match context_result {
            Ok(context) => {
                // Success case: A device and queue were created.
                println!("Successfully created WGPU context.");
                // Simple smoke test to ensure the device is responsive.
                let _limits = context.device.limits();
                println!("Device limits: {:?}", _limits);
                assert!(true);
            }
            Err(BitNetError::NoSuitableAdapter) => {
                // This is an expected and valid outcome in environments without a GPU (e.g., some CI runners).
                // We treat this as a pass, not a failure.
                println!("Test passed: No suitable GPU adapter found, which is an expected outcome in some environments.");
                assert!(true);
            }
            Err(e) => {
                // Any other error is unexpected and should fail the test.
                panic!("An unexpected error occurred during WGPU context creation: {:?}", e);
            }
        }
    }
}