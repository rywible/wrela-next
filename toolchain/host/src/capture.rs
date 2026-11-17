//! `wrela-host --capture <frame> <out.gputrace>` ([`crate::Options::capture`]): one frame's GPU
//! work captured to a Metal GPU trace, which Xcode opens to replay the frame with its
//! performance counters (each pipeline's registers, occupancy and limiters, which Metal's API
//! doesn't report). macOS only.
//!
//! The capture is Metal's `MTLCaptureManager`, scoped to the host's device
//! (`wrela_gpu::Device::start_capture`) with an `MTLCaptureDescriptor` whose destination is
//! `MTLCaptureDestinationGPUTraceDocument` (Apple's "Capturing a Metal workload
//! programmatically", docs/citations.md). Metal captures to a file only in a process that starts
//! with `MTL_CAPTURE_ENABLED=1` in its environment (a command-line binary has no Info.plist to
//! say `MetalCaptureEnabled`); `wrela-host` sets it for itself when `--capture` is given.
//!
//! Metal captures the command buffers made after the capture starts and committed before it
//! stops: [`crate::gpu`] submits what it has recorded before starting, and submits the frame's
//! work and waits for it before stopping.

use std::path::Path;

/// The environment variable Metal reads, as the process starts, to capture to a file.
pub const ENABLE: &str = "MTL_CAPTURE_ENABLED";

/// Starts capturing `device`'s work to `path`, a `.gputrace` that doesn't exist yet.
pub(crate) fn start(device: &wrela_gpu::Device, path: &Path) -> Result<(), String> {
    if path.extension().is_none_or(|e| e != "gputrace") {
        return Err(format!("a capture's file ends in .gputrace, not {}", path.display()));
    }
    if path.exists() {
        return Err(format!("{} exists: Metal won't write over it", path.display()));
    }
    let path = std::path::absolute(path).map_err(|e| format!("{}: {e}", path.display()))?;
    #[cfg(target_vendor = "apple")]
    {
        device.start_capture(&path)
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (device, path);
        Err("a frame capture is Metal's: macOS only".into())
    }
}

/// Stops the capture, which writes its file.
pub(crate) fn stop(device: &wrela_gpu::Device) {
    #[cfg(target_vendor = "apple")]
    device.stop_capture();
    #[cfg(not(target_vendor = "apple"))]
    let _ = device;
}
