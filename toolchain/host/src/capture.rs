//! `wrela-host --capture <frame> <out.gputrace>` ([`crate::Options::capture`]): one frame's GPU
//! work captured to a Metal GPU trace, which Xcode opens to replay the frame with its
//! performance counters (each pipeline's registers, occupancy and limiters, which Metal's API
//! doesn't report). macOS only.
//!
//! The capture is Metal's `MTLCaptureManager`, scoped to the host's device (reached through
//! `Device::as_hal`) with an `MTLCaptureDescriptor` whose destination is
//! `MTLCaptureDestinationGPUTraceDocument` (Apple's "Capturing a Metal workload
//! programmatically", docs/citations.md). Metal captures to a file only in a process that starts with
//! `MTL_CAPTURE_ENABLED=1` in its environment (a command-line binary has no Info.plist to say
//! `MetalCaptureEnabled`); `wrela-host` sets it for itself when `--capture` is given.
//!
//! Metal captures the command buffers made after the capture starts and committed before it
//! stops: [`crate::gpu`] submits what it has recorded before starting, and submits the frame's
//! work and waits for it before stopping.
//!
//! The `unsafe` calls are `as_hal`, which only reads the device; Metal's shared capture
//! manager, which is a singleton any thread may use; and the descriptor's capture object, which
//! must be a device, command queue or capture scope: it's the device.
#![allow(unsafe_code)]

use std::path::Path;

/// The environment variable Metal reads, as the process starts, to capture to a file.
pub const ENABLE: &str = "MTL_CAPTURE_ENABLED";

/// Starts capturing `device`'s work to `path`, a `.gputrace` that doesn't exist yet.
pub(crate) fn start(device: &wgpu::Device, path: &Path) -> Result<(), String> {
    if path.extension().is_none_or(|e| e != "gputrace") {
        return Err(format!("a capture's file ends in .gputrace, not {}", path.display()));
    }
    if path.exists() {
        return Err(format!("{} exists: Metal won't write over it", path.display()));
    }
    let path = std::path::absolute(path).map_err(|e| format!("{}: {e}", path.display()))?;
    #[cfg(target_vendor = "apple")]
    {
        apple::start(device, &path)
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (device, path);
        Err("a frame capture is Metal's: macOS only".into())
    }
}

/// Stops the capture, which writes its file.
pub(crate) fn stop() {
    #[cfg(target_vendor = "apple")]
    apple::stop();
}

#[cfg(target_vendor = "apple")]
mod apple {
    use super::ENABLE;
    use objc2::rc::autoreleasepool;
    use objc2_foundation::{NSString, NSURL};
    use objc2_metal::{MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager};
    use std::path::Path;

    pub(super) fn start(device: &wgpu::Device, path: &Path) -> Result<(), String> {
        // SAFETY: the raw device is only named as what to capture; nothing wgpu made is touched.
        let raw = match unsafe { device.as_hal::<wgpu::hal::api::Metal>() } {
            Some(hal) => hal.raw_device().clone(),
            None => return Err("a frame capture is Metal's; this device isn't Metal".into()),
        };
        autoreleasepool(|_| {
            // SAFETY: the shared capture manager is a process-wide singleton.
            let manager = unsafe { MTLCaptureManager::sharedCaptureManager() };
            if !manager.supportsDestination(MTLCaptureDestination::GPUTraceDocument) {
                return Err(format!(
                    "Metal won't capture to a file in this process: start it with {ENABLE}=1 in its environment (wrela-host sets it unless the environment already has {ENABLE})"
                ));
            }
            let descriptor = MTLCaptureDescriptor::new();
            // SAFETY: the capture object is a Metal device, as Metal requires.
            unsafe { descriptor.setCaptureObject(Some(raw.as_ref())) };
            descriptor.setDestination(MTLCaptureDestination::GPUTraceDocument);
            let url = NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()));
            descriptor.setOutputURL(Some(&url));
            manager
                .startCaptureWithDescriptor_error(&descriptor)
                .map_err(|e| format!("Metal couldn't start the capture: {e}"))
        })
    }

    pub(super) fn stop() {
        autoreleasepool(|_| {
            // SAFETY: as in `start`.
            let manager = unsafe { MTLCaptureManager::sharedCaptureManager() };
            if manager.isCapturing() {
                manager.stopCapture();
            }
        });
    }
}
