//! The GPU through Vulkan 1.3 (ash), for every machine but Apple's: RADV on the Linux machine.
//!
//! What wgpu decided for the host is decided here, and kept simple until a measurement says
//! otherwise:
//!
//! - **Submissions** are counted by one timeline semaphore: submission `n` signals `n`.
//! - **Memory**: ranges of shared blocks of device memory (`Heap`), by memory type and kind
//!   (buffers, images), from the first memory type that fits (device-local for the GPU's own,
//!   host-visible and coherent for what the CPU maps). Every resource starts zeroed, as WebGPU's
//!   do: a fill or a clear staged before the next submission's own work. Staged writes go through
//!   a belt of host-visible buffers reused once their submission is done.
//! - **Images** rest in a home layout by their use (`GENERAL` where a kernel writes one, else
//!   read-only optimal), and a pass or a copy moves what it uses out and back: in `GENERAL` (or
//!   made `MUTABLE_FORMAT`) RADV keeps no compression of a target, which cost the opening 10%.
//! - **Hazards**: a dispatch or a copy waits on a full memory barrier only when it reads or
//!   writes what a command since the last barrier wrote, or writes what one read (each
//!   encoder's sets of handles); a pass begins and ends with one.
//! - **Bindings**: set 0, each layout entry numbered by its place among the layout's entries
//!   the table doesn't hold (`wrela_abi::vulkan::descriptor`), a uniform at a dynamic offset as
//!   `UNIFORM_BUFFER_DYNAMIC`. The table (`wrela_abi::table`): one descriptor set a device,
//!   arrays of every view as a sampled image and, where it can be one, a storage image, and of
//!   every sampler, each at its index, written when it's made (update-after-bind, partially
//!   bound); set 1 of a pipeline that reaches it, its indices pushed as push constants.
//! - **Validation**: `WRELA_VULKAN_VALIDATION=1` turns on Khronos's layer, and every message it
//!   reports is an error the host reads after the frame (`take_errors`).

use crate::*;
use ash::vk;
use std::ffi::{CStr, CString, c_void};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

// ---- the device ---------------------------------------------------------------------------------

struct Shared {
    entry: ash::Entry,
    /// Whether it can present to a window (the surface and swapchain extensions are on).
    swapchain: bool,
    /// Whether a present can carry an id and be waited for (`VK_KHR_present_id`, `_wait`).
    present_wait: bool,
    instance: ash::Instance,
    debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    physical: vk::PhysicalDevice,
    raw: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    memory: vk::PhysicalDeviceMemoryProperties,
    info: AdapterInfo,
    limits: Limits,
    features: Features,
    timestamp_period: f32,
    /// The timeline semaphore: submission `n` signals `n`.
    timeline: vk::Semaphore,
    /// The pool of each command buffer's two timestamps, its start and its end, at `2 * (n %
    /// CLOCK_KEPT)` for the `n`th buffer begun, and the mask of a timestamp's valid bits
    /// ([`Device::gpu_ms`]).
    clock: Option<(vk::QueryPool, u64)>,
    /// The command buffers begun so far (each its slot in the clock's pool).
    clock_next: AtomicU64,
    /// Each recent submission's first and last timed command buffers, newest last.
    clock_of: Mutex<std::collections::VecDeque<(SubmissionIndex, u64, u64)>>,
    submitted: AtomicU64,
    /// Staged writes, recorded before the next submission's own work.
    pending: Mutex<Vec<Pending>>,
    /// Command pools and buffers of submissions not yet seen done, with what they hold alive.
    in_flight: Mutex<Vec<(SubmissionIndex, vk::CommandPool, Vec<Keep>)>>,
    /// Errors: the validation layer's messages, a lost device.
    errors: Arc<Mutex<Vec<String>>>,
    /// Pipelines and layouts made for each bind group layout, freed with the device.
    descriptor_pools: Mutex<Vec<vk::DescriptorPool>>,
    /// The table of views and samplers.
    table: Table,
    /// Host-visible buffers staged writes are copied into, reused once their submission is done.
    belt: Mutex<Belt<Arc<RawBuffer>>>,
    /// Device memory in blocks, by memory type and kind (buffers, images), each resource a range
    /// of one (`Heap`).
    heap: Mutex<Heap>,
}

// SAFETY: Vulkan's handles may be used from any thread when externally synchronized; the queue
// and the pending list are behind mutexes, and the host records on one thread at a time.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

/// The device's table (`wrela_abi::table`): its one descriptor set (set
/// `wrela_abi::table::TABLE_SET` of a pipeline that reaches it) and its free indices.
struct Table {
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    free: IndexPool,
}

impl Table {
    /// The table of `len` views and samplers and its set.
    ///
    /// # Safety
    /// `d` is a live device with descriptor indexing's update-after-bind and partially bound
    /// features on.
    unsafe fn new(d: &ash::Device, len: u32) -> std::result::Result<Table, vk::Result> {
        let kinds = [
            (wrela_abi::table::SAMPLED, vk::DescriptorType::SAMPLED_IMAGE),
            (wrela_abi::table::STORAGE, vk::DescriptorType::STORAGE_IMAGE),
            (wrela_abi::table::SAMPLERS, vk::DescriptorType::SAMPLER),
        ];
        let bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = kinds
            .iter()
            .map(|&(binding, ty)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(ty)
                    .descriptor_count(len)
                    .stage_flags(vk::ShaderStageFlags::ALL)
            })
            .collect();
        let flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND
            | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
            | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING; 3];
        let mut binding_flags =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
        // SAFETY: the caller's.
        unsafe {
            let set_layout = d.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                    .bindings(&bindings)
                    .push_next(&mut binding_flags),
                None,
            )?;
            let sizes: Vec<vk::DescriptorPoolSize> = kinds
                .iter()
                .map(|&(_, ty)| vk::DescriptorPoolSize { ty, descriptor_count: len })
                .collect();
            let pool = d.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )?;
            let layouts = [set_layout];
            let set = d.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )?[0];
            Ok(Table { set_layout, pool, set, free: IndexPool::new(len) })
        }
    }
}

/// What a submission holds alive until it's done: what its commands use (held, never read).
#[allow(dead_code)]
enum Keep {
    Buffer(Arc<RawBuffer>),
    Group(Arc<Group>),
    Pipeline(Arc<RawPipeline>),
    Held(Held),
    Queries(Arc<RawQueries>),
    Image(Arc<RawImage>),
}

enum Pending {
    /// Bytes from a staging buffer (at `at`) into a buffer.
    Buffer { staging: Arc<RawBuffer>, at: u64, dst: Arc<RawBuffer>, offset: u64, len: u64 },
    /// Zeros over a whole buffer.
    Fill { dst: Arc<RawBuffer> },
    /// Texels from a staging buffer into a level and layer of an image.
    Texture {
        staging: Arc<RawBuffer>,
        at: u64,
        bytes_per_row: u32,
        rows_per_image: u32,
        dst: Arc<RawImage>,
        level: u32,
        layer: u32,
        origin: [u32; 3],
        extent: [u32; 3],
    },
    /// An image zeroed (every level and layer), and put in its home layout: cleared, or, where
    /// its format is compressed (which Vulkan clears not, VUID-vkCmdClearColorImage-image-00007),
    /// each level copied from `zeros`, a staging buffer's zeroed bytes at `at` as many as its
    /// first level holds (`extent`, its width, height and depth).
    Clear { dst: Arc<RawImage>, zeros: Option<(Arc<RawBuffer>, u64, vk::Extent3D)> },
}

impl Pending {
    /// What it reads (a staging buffer), if anything, and what it writes, by handle, for the
    /// hazards between staged writes.
    fn uses(&self) -> (Option<u64>, u64) {
        use vk::Handle;
        match self {
            Pending::Buffer { staging, dst, .. } => {
                (Some(staging.buffer.as_raw()), dst.buffer.as_raw())
            }
            Pending::Fill { dst } => (None, dst.buffer.as_raw()),
            Pending::Texture { staging, dst, .. } => {
                (Some(staging.buffer.as_raw()), dst.image.as_raw())
            }
            Pending::Clear { dst, zeros } => {
                (zeros.as_ref().map(|z| z.0.buffer.as_raw()), dst.image.as_raw())
            }
        }
    }
}

/// The GPU: a Vulkan device and its one queue.
#[derive(Clone)]
pub struct Device(Arc<Owner>);

/// What the device's handles (and the encoders and surfaces made of them) hold: the device. Its
/// resources hold the device too (`Arc<Shared>`), and the device holds some of them (staged
/// writes, the belt, what submissions keep), so once the last handle goes those are let go
/// here, after the GPU is idle: else they would keep the device, and themselves, alive for good.
struct Owner(Arc<Shared>);

impl std::ops::Deref for Owner {
    type Target = Shared;

    fn deref(&self) -> &Shared {
        &self.0
    }
}

impl Drop for Owner {
    fn drop(&mut self) {
        let s = &self.0;
        // SAFETY: once the device is idle nothing in flight uses what submissions kept.
        unsafe {
            let _ = s.raw.device_wait_idle();
        }
        drop(std::mem::take(&mut *lock(&s.pending)));
        let flights = std::mem::take(&mut *lock(&s.in_flight));
        for (_, pool, keep) in flights {
            if pool != vk::CommandPool::null() {
                // SAFETY: the device is idle.
                unsafe { s.raw.destroy_command_pool(pool, None) };
            }
            drop(keep);
        }
        drop(std::mem::take(&mut *lock(&s.belt)));
    }
}

unsafe extern "system" fn on_message(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _kind: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user: *mut c_void,
) -> vk::Bool32 {
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
        || severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING)
    {
        // SAFETY: the layer passes its message, and `user` is the errors list it was given,
        // which the device keeps alive while the messenger exists.
        let message = unsafe {
            let d = &*data;
            if d.p_message.is_null() {
                String::from("(no message)")
            } else {
                CStr::from_ptr(d.p_message).to_string_lossy().into_owned()
            }
        };
        let errors = unsafe { &*(user as *const Mutex<Vec<String>>) };
        lock(errors).push(format!("Vulkan validation: {message}"));
    }
    vk::FALSE
}

impl Device {
    /// The machine's GPU: a discrete one first, then an integrated one; a GPU on the CPU only if
    /// `allow_software`.
    pub fn open(options: &DeviceOptions) -> Result<Device> {
        // SAFETY: every call below is Vulkan's, with structures built here and kept alive past
        // the calls that read them.
        unsafe {
            let entry = ash::Entry::load().map_err(|e| Error(format!("no Vulkan loader: {e}")))?;
            let validate = std::env::var("WRELA_VULKAN_VALIDATION").is_ok_and(|v| v == "1");
            let app_name = CString::new("wrela").expect("a name");
            let app = vk::ApplicationInfo::default()
                .application_name(&app_name)
                .engine_name(&app_name)
                .api_version(vk::API_VERSION_1_3);
            let layer = CString::new("VK_LAYER_KHRONOS_validation").expect("a name");
            let layers: Vec<*const i8> = if validate { vec![layer.as_ptr()] } else { Vec::new() };
            let mut extensions: Vec<*const i8> =
                if validate { vec![ash::ext::debug_utils::NAME.as_ptr()] } else { Vec::new() };
            // A window's surface, wherever the loader has its kind (a headless run uses none).
            let have = entry.enumerate_instance_extension_properties(None).unwrap_or_default();
            let has = |name: &CStr| have.iter().any(|e| e.extension_name_as_c_str() == Ok(name));
            let surfaces = [
                ash::khr::surface::NAME,
                ash::khr::wayland_surface::NAME,
                ash::khr::xlib_surface::NAME,
                ash::khr::xcb_surface::NAME,
                ash::ext::swapchain_colorspace::NAME,
            ];
            let windowed = has(ash::khr::surface::NAME);
            if windowed {
                extensions.extend(surfaces.iter().filter(|n| has(n)).map(|n| n.as_ptr()));
            }
            let instance = entry
                .create_instance(
                    &vk::InstanceCreateInfo::default()
                        .application_info(&app)
                        .enabled_layer_names(&layers)
                        .enabled_extension_names(&extensions),
                    None,
                )
                .map_err(|e| Error(format!("Vulkan made no instance: {e}")))?;
            let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
            let debug = if validate {
                let du = ash::ext::debug_utils::Instance::new(&entry, &instance);
                let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                    .message_severity(
                        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                            | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                    )
                    .message_type(
                        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                    )
                    .pfn_user_callback(Some(on_message))
                    .user_data(Arc::as_ptr(&errors) as *mut c_void);
                let m = du
                    .create_debug_utils_messenger(&info, None)
                    .map_err(|e| Error(format!("no validation messenger: {e}")))?;
                Some((du, m))
            } else {
                None
            };
            let devices = instance
                .enumerate_physical_devices()
                .map_err(|e| Error(format!("Vulkan lists no GPU: {e}")))?;
            let rank = |p: &vk::PhysicalDevice| match instance
                .get_physical_device_properties(*p)
                .device_type
            {
                vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                vk::PhysicalDeviceType::CPU if options.allow_software => 3,
                _ => 9,
            };
            let mut found: Vec<String> = Vec::new();
            for p in &devices {
                let props = instance.get_physical_device_properties(*p);
                found.push(cstr(&props.device_name));
            }
            let physical =
                devices.iter().copied().filter(|p| rank(p) < 9).min_by_key(rank).ok_or_else(
                    || Error(format!("no hardware GPU through Vulkan: found {}", found.join(", "))),
                )?;
            let props = instance.get_physical_device_properties(physical);
            if props.api_version < vk::API_VERSION_1_3 {
                return Err(Error(format!(
                    "{} has Vulkan {}.{}; the host needs 1.3",
                    cstr(&props.device_name),
                    vk::api_version_major(props.api_version),
                    vk::api_version_minor(props.api_version)
                )));
            }
            let families = instance.get_physical_device_queue_family_properties(physical);
            let queue_family = families
                .iter()
                .position(|f| {
                    f.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                })
                .ok_or("the GPU has no queue that both draws and computes")?
                as u32;
            let valid_bits = families[queue_family as usize].timestamp_valid_bits;
            let timestamps_valid = valid_bits > 0;

            // What it has.
            let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
            let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut all = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut f11)
                .push_next(&mut f12)
                .push_next(&mut f13);
            instance.get_physical_device_features2(physical, &mut all);
            let base = all.features;
            let (f11, f12, f13) = (f11, f12, f13);
            for (have, what) in [
                (f12.timeline_semaphore, "timeline semaphores"),
                (f13.synchronization2, "synchronization2"),
                (f13.dynamic_rendering, "dynamic rendering"),
                (f12.runtime_descriptor_array, "runtime descriptor arrays"),
                (f12.descriptor_binding_partially_bound, "partially bound descriptors"),
                (
                    f12.descriptor_binding_update_unused_while_pending,
                    "descriptors updated while pending",
                ),
                (
                    f12.descriptor_binding_sampled_image_update_after_bind,
                    "sampled images updated after bind",
                ),
                (
                    f12.descriptor_binding_storage_image_update_after_bind,
                    "storage images updated after bind",
                ),
                (base.shader_sampled_image_array_dynamic_indexing, "sampled images indexed"),
                (base.shader_storage_image_array_dynamic_indexing, "storage images indexed"),
                // What keeps a kernel's buffer indexes inside their buffers (the SPIR-V doesn't
                // clamp them: `wrela_shader::spirv`, language.md §11.3).
                (base.robust_buffer_access, "robust buffer access"),
            ] {
                if have == vk::FALSE {
                    return Err(Error(format!("the GPU lacks {what}, which the host needs")));
                }
            }
            let mut sub = vk::PhysicalDeviceSubgroupProperties::default();
            let mut sub13 = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
            let mut driver = vk::PhysicalDeviceDriverProperties::default();
            let mut indexing = vk::PhysicalDeviceDescriptorIndexingProperties::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut sub)
                .push_next(&mut sub13)
                .push_next(&mut driver)
                .push_next(&mut indexing);
            instance.get_physical_device_properties2(physical, &mut props2);
            let l = props.limits;

            // The features WebGPU names that it has.
            let f16 = f12.shader_float16 == vk::TRUE
                && f11.storage_buffer16_bit_access == vk::TRUE
                && f11.uniform_and_storage_buffer16_bit_access == vk::TRUE;
            let subgroups = sub.supported_stages.contains(vk::ShaderStageFlags::COMPUTE)
                && sub.supported_operations.contains(
                    vk::SubgroupFeatureFlags::BASIC
                        | vk::SubgroupFeatureFlags::VOTE
                        | vk::SubgroupFeatureFlags::ARITHMETIC
                        | vk::SubgroupFeatureFlags::BALLOT
                        | vk::SubgroupFeatureFlags::SHUFFLE,
                );
            let format_has = |f: vk::Format, want: vk::FormatFeatureFlags| {
                instance
                    .get_physical_device_format_properties(physical, f)
                    .optimal_tiling_features
                    .contains(want)
            };
            let mut features: Vec<&'static str> = Vec::new();
            if f16 {
                features.push("shader-f16");
            }
            if subgroups {
                features.push("subgroups");
            }
            if format_has(
                vk::Format::R32_SFLOAT,
                vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR,
            ) {
                features.push("float32-filterable");
            }
            if format_has(vk::Format::R32_SFLOAT, vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND) {
                features.push("float32-blendable");
            }
            if format_has(
                vk::Format::B10G11R11_UFLOAT_PACK32,
                vk::FormatFeatureFlags::COLOR_ATTACHMENT,
            ) {
                features.push("rg11b10ufloat-renderable");
            }
            features.push("indirect-first-instance");
            if base.multi_draw_indirect == vk::TRUE {
                features.push("chromium-experimental-multi-draw-indirect");
            }
            if base.texture_compression_bc == vk::TRUE {
                features.push("texture-compression-bc");
            }
            if base.texture_compression_etc2 == vk::TRUE {
                features.push("texture-compression-etc2");
            }
            if base.texture_compression_astc_ldr == vk::TRUE {
                features.push("texture-compression-astc");
            }
            if timestamps_valid && l.timestamp_compute_and_graphics == vk::TRUE {
                features.push("timestamp-query");
            }
            let features = options.features_of(features);

            // The device, with what it has of what the host uses.
            let priorities = [1.0f32];
            let queues = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family)
                .queue_priorities(&priorities)];
            let mut on11 = vk::PhysicalDeviceVulkan11Features::default()
                .storage_buffer16_bit_access(f16)
                .uniform_and_storage_buffer16_bit_access(f16);
            let mut on12 = vk::PhysicalDeviceVulkan12Features::default()
                .timeline_semaphore(true)
                .shader_float16(f16)
                .host_query_reset(f12.host_query_reset == vk::TRUE)
                .shader_int8(f12.shader_int8 == vk::TRUE)
                .runtime_descriptor_array(true)
                .descriptor_binding_partially_bound(true)
                .descriptor_binding_update_unused_while_pending(true)
                .descriptor_binding_sampled_image_update_after_bind(true)
                .descriptor_binding_storage_image_update_after_bind(true);
            let mut on13 = vk::PhysicalDeviceVulkan13Features::default()
                .synchronization2(true)
                .dynamic_rendering(true)
                .maintenance4(f13.maintenance4 == vk::TRUE)
                .subgroup_size_control(f13.subgroup_size_control == vk::TRUE)
                .shader_demote_to_helper_invocation(
                    f13.shader_demote_to_helper_invocation == vk::TRUE,
                );
            let on = vk::PhysicalDeviceFeatures::default()
                .robust_buffer_access(true)
                .geometry_shader(base.geometry_shader == vk::TRUE)
                .multi_draw_indirect(base.multi_draw_indirect == vk::TRUE)
                .draw_indirect_first_instance(base.draw_indirect_first_instance == vk::TRUE)
                .sampler_anisotropy(base.sampler_anisotropy == vk::TRUE)
                .texture_compression_bc(base.texture_compression_bc == vk::TRUE)
                .texture_compression_etc2(base.texture_compression_etc2 == vk::TRUE)
                .texture_compression_astc_ldr(base.texture_compression_astc_ldr == vk::TRUE)
                .fragment_stores_and_atomics(base.fragment_stores_and_atomics == vk::TRUE)
                .shader_storage_image_write_without_format(
                    base.shader_storage_image_write_without_format == vk::TRUE,
                )
                .shader_storage_image_extended_formats(
                    base.shader_storage_image_extended_formats == vk::TRUE,
                )
                .independent_blend(base.independent_blend == vk::TRUE)
                .depth_clamp(base.depth_clamp == vk::TRUE)
                .depth_bias_clamp(base.depth_bias_clamp == vk::TRUE)
                .shader_sampled_image_array_dynamic_indexing(true)
                .shader_storage_image_array_dynamic_indexing(true);
            let device_exts =
                instance.enumerate_device_extension_properties(physical).unwrap_or_default();
            let device_has =
                |n: &CStr| device_exts.iter().any(|e| e.extension_name_as_c_str() == Ok(n));
            let swapchain = windowed && device_has(ash::khr::swapchain::NAME);
            // When a frame reached the display: a present's id, and a wait for it.
            let present_wait = swapchain
                && device_has(ash::khr::present_id::NAME)
                && device_has(ash::khr::present_wait::NAME);
            let mut device_extensions: Vec<*const i8> =
                if swapchain { vec![ash::khr::swapchain::NAME.as_ptr()] } else { Vec::new() };
            if present_wait {
                device_extensions.push(ash::khr::present_id::NAME.as_ptr());
                device_extensions.push(ash::khr::present_wait::NAME.as_ptr());
            }
            let mut on_id = vk::PhysicalDevicePresentIdFeaturesKHR::default().present_id(true);
            let mut on_wait =
                vk::PhysicalDevicePresentWaitFeaturesKHR::default().present_wait(true);
            let mut info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queues)
                .enabled_extension_names(&device_extensions)
                .enabled_features(&on)
                .push_next(&mut on11)
                .push_next(&mut on12)
                .push_next(&mut on13);
            if present_wait {
                info = info.push_next(&mut on_id).push_next(&mut on_wait);
            }
            let raw = instance
                .create_device(physical, &info, None)
                .map_err(|e| Error(format!("Vulkan made no device: {e}")))?;
            let queue = raw.get_device_queue(queue_family, 0);
            let mut timeline_type = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let timeline = raw
                .create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(&mut timeline_type),
                    None,
                )
                .map_err(|e| Error(format!("Vulkan made no semaphore: {e}")))?;
            // Two timestamps a command buffer, its start and its end (`Device::gpu_ms`).
            let clock = if timestamps_valid {
                let pool = raw
                    .create_query_pool(
                        &vk::QueryPoolCreateInfo::default()
                            .query_type(vk::QueryType::TIMESTAMP)
                            .query_count(2 * CLOCK_KEPT as u32),
                        None,
                    )
                    .map_err(|e| Error(format!("Vulkan made no query pool: {e}")))?;
                let mask = if valid_bits >= 64 { u64::MAX } else { (1u64 << valid_bits) - 1 };
                Some((pool, mask))
            } else {
                None
            };
            let memory = instance.get_physical_device_memory_properties(physical);
            let device_type = match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => DeviceType::DiscreteGpu,
                vk::PhysicalDeviceType::INTEGRATED_GPU => DeviceType::IntegratedGpu,
                vk::PhysicalDeviceType::VIRTUAL_GPU => DeviceType::VirtualGpu,
                vk::PhysicalDeviceType::CPU => DeviceType::Cpu,
                _ => DeviceType::Other,
            };
            let info = AdapterInfo {
                name: cstr(&props.device_name),
                backend: Backend::Vulkan,
                device_type,
                vendor: props.vendor_id,
                device: props.device_id,
                driver: cstr(&driver.driver_name),
                driver_info: cstr(&driver.driver_info),
                subgroup_min_size: if sub13.min_subgroup_size > 0 {
                    sub13.min_subgroup_size
                } else {
                    sub.subgroup_size
                },
                subgroup_max_size: sub13.max_subgroup_size.max(sub.subgroup_size),
            };
            let table_len = TABLE_LEN
                .min(indexing.max_descriptor_set_update_after_bind_sampled_images)
                .min(indexing.max_descriptor_set_update_after_bind_storage_images)
                .min(indexing.max_descriptor_set_update_after_bind_samplers)
                .min(indexing.max_per_stage_descriptor_update_after_bind_sampled_images)
                .min(indexing.max_per_stage_descriptor_update_after_bind_storage_images)
                .min(indexing.max_per_stage_descriptor_update_after_bind_samplers)
                .min(indexing.max_per_stage_update_after_bind_resources / 3);
            let table = Table::new(&raw, table_len)
                .map_err(|e| Error(format!("Vulkan made no table: {e}")))?;
            let storage_max = u64::from(l.max_storage_buffer_range).min(u64::from(u32::MAX)) & !3;
            let limits = Limits {
                max_texture_dimension_2d: l.max_image_dimension2_d,
                max_texture_dimension_3d: l.max_image_dimension3_d,
                max_texture_array_layers: l.max_image_array_layers,
                max_buffer_size: storage_max.max(u64::from(l.max_storage_buffer_range)),
                max_storage_buffer_binding_size: storage_max,
                max_uniform_buffer_binding_size: u64::from(l.max_uniform_buffer_range),
                max_storage_buffers_per_shader_stage: l.max_per_stage_descriptor_storage_buffers,
                max_sampled_textures_per_shader_stage: l.max_per_stage_descriptor_sampled_images,
                max_samplers_per_shader_stage: l.max_per_stage_descriptor_samplers,
                max_storage_textures_per_shader_stage: l.max_per_stage_descriptor_storage_images,
                max_color_attachments: l.max_color_attachments.min(8),
                max_color_attachment_bytes_per_sample: 64,
                min_uniform_buffer_offset_alignment: (l.min_uniform_buffer_offset_alignment as u32)
                    .max(256),
                min_storage_buffer_offset_alignment: (l.min_storage_buffer_offset_alignment as u32)
                    .max(32),
                max_compute_workgroup_storage_size: l.max_compute_shared_memory_size,
                max_compute_invocations_per_workgroup: l.max_compute_work_group_invocations,
                max_compute_workgroup_size: l.max_compute_work_group_size,
                max_compute_workgroups_per_dimension: l.max_compute_work_group_count[0]
                    .min(l.max_compute_work_group_count[1])
                    .min(l.max_compute_work_group_count[2]),
            };
            Ok(Device(Arc::new(Owner(Arc::new(Shared {
                entry,
                swapchain,
                present_wait,
                instance,
                debug,
                physical,
                raw,
                queue,
                queue_family,
                memory,
                info,
                limits,
                features,
                timestamp_period: l.timestamp_period,
                timeline,
                clock,
                clock_next: AtomicU64::new(0),
                clock_of: Mutex::new(std::collections::VecDeque::new()),
                submitted: AtomicU64::new(0),
                pending: Mutex::new(Vec::new()),
                in_flight: Mutex::new(Vec::new()),
                errors,
                descriptor_pools: Mutex::new(Vec::new()),
                table,
                belt: Mutex::new(Belt::default()),
                heap: Mutex::new(Heap::default()),
            })))))
        }
    }

    /// The device, as its resources hold it.
    fn shared(&self) -> Arc<Shared> {
        self.0.0.clone()
    }

    pub fn info(&self) -> &AdapterInfo {
        &self.0.info
    }

    pub fn limits(&self) -> &Limits {
        &self.0.limits
    }

    pub fn features(&self) -> &Features {
        &self.0.features
    }

    /// Nanoseconds a timestamp tick.
    pub fn timestamp_period(&self) -> f32 {
        self.0.timestamp_period
    }

    pub fn report(&self) -> String {
        let i = &self.0.info;
        let mut s = String::new();
        s.push_str(&format!("driver {} ({})\n", i.driver, i.driver_info));
        s.push_str(&format!("subgroups {} to {}\n", i.subgroup_min_size, i.subgroup_max_size));
        for (k, h) in self.0.memory.memory_heaps_as_slice().iter().enumerate() {
            s.push_str(&format!("heap {k}: {} bytes, {:?}\n", h.size, h.flags));
        }
        s
    }

    fn raw(&self) -> &ash::Device {
        &self.0.raw
    }

    /// The first memory type of `bits` with `want`.
    fn memory_type(&self, bits: u32, want: vk::MemoryPropertyFlags) -> Option<u32> {
        let m = &self.0.memory;
        (0..m.memory_type_count).find(|&i| {
            bits & (1 << i) != 0 && m.memory_types[i as usize].property_flags.contains(want)
        })
    }

    /// A range of device memory of type `ty` for a resource with requirements `req` (an image's
    /// pool apart from buffers', so the buffer-image granularity rule can't apply): from a block
    /// of its pool, a new one twice the last (from [`FIRST_BLOCK`] to [`BLOCK`]) where none has
    /// room, or one of its own if it's more than half of [`BLOCK`].
    fn allocate(&self, req: vk::MemoryRequirements, ty: u32, image: bool) -> Allocation {
        let d = self.raw();
        let host = self.0.memory.memory_types[ty as usize]
            .property_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
        // SAFETY: memory allocated, and mapped whole where it's host-visible, here.
        let fresh = |size: u64| unsafe {
            let memory = d
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default().allocation_size(size).memory_type_index(ty),
                    None,
                )
                .expect("Vulkan allocated no memory");
            let mapped = if host {
                d.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                    .expect("Vulkan mapped no memory") as *mut u8
            } else {
                std::ptr::null_mut()
            };
            (memory, mapped)
        };
        if req.size > BLOCK / 2 {
            let (memory, mapped) = fresh(req.size);
            return Allocation { memory, offset: 0, size: req.size, mapped, pool: None };
        }
        let key = (ty, image);
        let mut heap = lock(&self.0.heap);
        let blocks = heap.pools.entry(key).or_default();
        for b in blocks.iter_mut() {
            if let Some(at) = b.take(req.size, req.alignment.max(1)) {
                let mapped = if b.mapped.is_null() {
                    std::ptr::null_mut()
                } else {
                    // SAFETY: inside the block's mapping.
                    unsafe { b.mapped.add(at as usize) }
                };
                return Allocation {
                    memory: b.memory,
                    offset: at,
                    size: req.size,
                    mapped,
                    pool: Some(key),
                };
            }
        }
        // Small first, so a device that holds little reserves little (a test suite runs many).
        let last = blocks.iter().map(|b| b.size).max().unwrap_or(FIRST_BLOCK / 2);
        let size = (last * 2).clamp(FIRST_BLOCK, BLOCK).max(req.size.next_power_of_two());
        let (memory, mapped) = fresh(size);
        let mut b = Block { memory, size, mapped, free: vec![(0, size)] };
        let at = b.take(req.size, req.alignment.max(1)).expect("a fresh block holds it");
        let mapped_at = if mapped.is_null() {
            std::ptr::null_mut()
        } else {
            // SAFETY: inside the block's mapping.
            unsafe { mapped.add(at as usize) }
        };
        blocks.push(b);
        Allocation { memory, offset: at, size: req.size, mapped: mapped_at, pool: Some(key) }
    }

    fn raw_buffer(&self, size: u64, mappable: bool, usage: vk::BufferUsageFlags) -> Arc<RawBuffer> {
        let size = size.max(4).next_multiple_of(4);
        // SAFETY: a buffer, its memory and its mapping, made and bound here.
        unsafe {
            let d = self.raw();
            let buffer = d
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(size)
                        .usage(usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .expect("Vulkan made no buffer");
            let req = d.get_buffer_memory_requirements(buffer);
            let want = if mappable {
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
            } else {
                vk::MemoryPropertyFlags::DEVICE_LOCAL
            };
            let ty = self
                .memory_type(req.memory_type_bits, want)
                .or_else(|| {
                    self.memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::empty())
                })
                .expect("no memory type for a buffer");
            let memory = self.allocate(req, ty, false);
            d.bind_buffer_memory(buffer, memory.memory, memory.offset)
                .expect("Vulkan bound no memory");
            let mapped = if mappable {
                assert!(!memory.mapped.is_null(), "a mappable buffer's memory isn't host-visible");
                memory.mapped
            } else {
                std::ptr::null_mut()
            };
            Arc::new(RawBuffer { device: self.shared(), buffer, memory, size, mapped })
        }
    }

    pub fn create_buffer(&self, desc: &BufferDesc<'_>) -> Buffer {
        let usage = if desc.mappable {
            vk::BufferUsageFlags::TRANSFER_DST
        } else {
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::UNIFORM_BUFFER
                | vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::INDIRECT_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
        };
        let raw = self.raw_buffer(desc.size, desc.mappable, usage);
        if desc.mappable {
            // SAFETY: the mapping is the buffer's whole size.
            unsafe { std::ptr::write_bytes(raw.mapped, 0, raw.size as usize) };
        } else {
            lock(&self.0.pending).push(Pending::Fill { dst: raw.clone() });
        }
        Buffer { raw, size: desc.size, mappable: desc.mappable }
    }

    pub fn create_texture(&self, desc: &TextureDesc<'_>) -> Texture {
        let layers = desc.depth_or_layers.max(1);
        let levels = desc.levels.max(1);
        let format = vk_format(desc.format);
        let depth = desc.format == TextureFormat::Depth32Float;
        let mut usage = vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST;
        if desc.usage.storage {
            usage |= vk::ImageUsageFlags::STORAGE;
        }
        if desc.usage.render {
            usage |= if depth {
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
            } else {
                vk::ImageUsageFlags::COLOR_ATTACHMENT
            };
        }
        let (ty, extent, array) = match desc.dimension {
            TextureDimension::D3 => (
                vk::ImageType::TYPE_3D,
                vk::Extent3D { width: desc.width, height: desc.height, depth: layers },
                1,
            ),
            TextureDimension::D2 => (
                vk::ImageType::TYPE_2D,
                vk::Extent3D { width: desc.width, height: desc.height, depth: 1 },
                layers,
            ),
        };
        let cube =
            desc.dimension == TextureDimension::D2 && layers % 6 == 0 && desc.width == desc.height;
        // SAFETY: an image and its memory, made and bound here.
        let raw = unsafe {
            let d = self.raw();
            let image = d
                .create_image(
                    &vk::ImageCreateInfo::default()
                        // Never MUTABLE_FORMAT: every view keeps its image's format, and the
                        // flag would cost a target its compression (RADV drops DCC).
                        .flags(if cube {
                            vk::ImageCreateFlags::CUBE_COMPATIBLE
                        } else {
                            vk::ImageCreateFlags::empty()
                        })
                        .image_type(ty)
                        .format(format)
                        .extent(extent)
                        .mip_levels(levels)
                        .array_layers(array)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    None,
                )
                .expect("Vulkan made no image");
            let req = d.get_image_memory_requirements(image);
            let ty = self
                .memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
                .or_else(|| {
                    self.memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::empty())
                })
                .expect("no memory type for an image");
            let memory = self.allocate(req, ty, true);
            d.bind_image_memory(image, memory.memory, memory.offset)
                .expect("Vulkan bound no memory");
            Arc::new(RawImage {
                device: self.shared(),
                image,
                memory,
                format: desc.format,
                aspect: if depth {
                    vk::ImageAspectFlags::DEPTH
                } else {
                    vk::ImageAspectFlags::COLOR
                },
                swapchain: false,
                usage,
                home: if desc.usage.storage {
                    vk::ImageLayout::GENERAL
                } else if depth {
                    vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
                } else {
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                },
                levels,
                layers: array,
            })
        };
        // A compressed format is zeroed by copies, from zeros enough for its first level.
        let zeros = desc.format.is_compressed().then(|| {
            let bytes = desc.format.bytes_of(extent.width, extent.height)
                * u64::from(extent.depth)
                * u64::from(array);
            let zeros = vec![0u8; bytes as usize];
            let (staging, at) = self.staging(&zeros, u64::from(COPY_BYTES_PER_ROW_ALIGNMENT));
            (staging, at, extent)
        });
        lock(&self.0.pending).push(Pending::Clear { dst: raw.clone(), zeros });
        Texture {
            raw,
            format: desc.format,
            width: desc.width,
            height: desc.height,
            depth_or_layers: layers,
            levels,
            dimension: desc.dimension,
        }
    }

    pub fn create_sampler(&self, desc: &SamplerDesc) -> Sampler {
        let filter = if desc.linear { vk::Filter::LINEAR } else { vk::Filter::NEAREST };
        let mode = if desc.repeat {
            vk::SamplerAddressMode::REPEAT
        } else {
            vk::SamplerAddressMode::CLAMP_TO_EDGE
        };
        let mut info = vk::SamplerCreateInfo::default()
            .mag_filter(filter)
            .min_filter(filter)
            .mipmap_mode(if desc.mip_linear {
                vk::SamplerMipmapMode::LINEAR
            } else {
                vk::SamplerMipmapMode::NEAREST
            })
            .address_mode_u(mode)
            .address_mode_v(mode)
            .address_mode_w(mode)
            .min_lod(0.0)
            .max_lod(32.0);
        if desc.anisotropy > 1 {
            info = info.anisotropy_enable(true).max_anisotropy(f32::from(desc.anisotropy));
        }
        if let Some(c) = desc.compare {
            info = info.compare_enable(true).compare_op(compare_op(c));
        }
        // SAFETY: a sampler.
        let raw =
            unsafe { self.raw().create_sampler(&info, None).expect("Vulkan made no sampler") };
        let index = self.0.table.free.take();
        let image = [vk::DescriptorImageInfo {
            sampler: raw,
            image_view: vk::ImageView::null(),
            image_layout: vk::ImageLayout::UNDEFINED,
        }];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.0.table.set)
            .dst_binding(wrela_abi::table::SAMPLERS)
            .dst_array_element(index)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .image_info(&image);
        // SAFETY: as a view's (`Texture::create_view`).
        unsafe { self.raw().update_descriptor_sets(&[write], &[]) };
        Sampler { raw: Arc::new(RawSampler { device: self.shared(), raw, index }) }
    }

    pub fn create_query_set(&self, count: u32) -> Result<QuerySet> {
        if !self.0.features.has("timestamp-query") {
            return Err("this GPU writes no timestamps".into());
        }
        // SAFETY: a query pool.
        let pool = unsafe {
            self.raw()
                .create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(count),
                    None,
                )
                .map_err(|e| Error(format!("Vulkan made no query pool: {e}")))?
        };
        Ok(QuerySet { raw: Arc::new(RawQueries { device: self.shared(), pool, count }) })
    }

    pub fn create_bind_group_layout(&self, entries: &[LayoutEntry]) -> BindGroupLayout {
        // What the device's table holds isn't in the set: its index is in the block, pushed.
        let block = entries.iter().filter_map(|e| e.index_at).map(|at| at + 4).max().unwrap_or(0);
        let mut bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = Vec::new();
        let mut numbers = Vec::with_capacity(entries.len());
        let mut sizes: Vec<vk::DescriptorPoolSize> = Vec::new();
        for e in entries {
            if e.index_at.is_some() {
                numbers.push(None);
                continue;
            }
            let number = bindings.len() as u32;
            let ty = descriptor_type(e.ty);
            numbers.push(Some(number));
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(number)
                    .descriptor_type(ty)
                    .descriptor_count(1)
                    .stage_flags(stages(e.visibility)),
            );
            match sizes.iter_mut().find(|s| s.ty == ty) {
                Some(s) => s.descriptor_count += 1,
                None => sizes.push(vk::DescriptorPoolSize { ty, descriptor_count: 1 }),
            }
        }
        // SAFETY: a set layout and a pipeline layout over it.
        unsafe {
            let d = self.raw();
            let set = d
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .expect("Vulkan made no set layout");
            let with_table = [set, self.0.table.set_layout];
            let ranges = [vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::ALL,
                offset: 0,
                size: block,
            }];
            let info = if block == 0 {
                vk::PipelineLayoutCreateInfo::default().set_layouts(&with_table[..1])
            } else {
                vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&with_table)
                    .push_constant_ranges(&ranges)
            };
            let pipeline =
                d.create_pipeline_layout(&info, None).expect("Vulkan made no pipeline layout");
            BindGroupLayout(Arc::new(Layout {
                device: self.shared(),
                entries: entries.to_vec(),
                numbers,
                sizes,
                set,
                pipeline,
                block: block as usize,
            }))
        }
    }

    pub fn create_bind_group(
        &self,
        layout: &BindGroupLayout,
        entries: &[BindGroupEntry<'_>],
    ) -> BindGroup {
        let l = &layout.0;
        // SAFETY: a pool of one set, the set, and its descriptors written from live resources.
        unsafe {
            let d = self.raw();
            let pool = if l.sizes.is_empty() {
                vk::DescriptorPool::null()
            } else {
                d.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&l.sizes),
                    None,
                )
                .expect("Vulkan made no descriptor pool")
            };
            let set = if pool == vk::DescriptorPool::null() {
                vk::DescriptorSet::null()
            } else {
                let layouts = [l.set];
                d.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&layouts),
                )
                .expect("Vulkan allocated no descriptor set")[0]
            };
            let mut buffers: Vec<[vk::DescriptorBufferInfo; 1]> = Vec::new();
            let mut images: Vec<[vk::DescriptorImageInfo; 1]> = Vec::new();
            let mut writes: Vec<(u32, vk::DescriptorType, bool, usize)> = Vec::new();
            let mut held = Vec::new();
            let (mut read, mut written) = (Vec::new(), Vec::new());
            // What it binds by its handle, read or (where a shader may write it) written.
            let mut uses = |handle: u64, writes: bool| {
                if writes { written.push(handle) } else { read.push(handle) }
            };
            let mut block = vec![0u8; l.block];
            for e in entries {
                let Some(k) = l.entries.iter().position(|x| x.binding == e.binding) else {
                    continue;
                };
                if let Some(at) = l.entries[k].index_at {
                    let index = match &e.resource {
                        Resource::View(v) => {
                            held.push(Held::View(v.raw.clone()));
                            uses(
                                vk::Handle::as_raw(v.raw.image.image),
                                matches!(l.entries[k].ty, BindingType::StorageTexture { .. }),
                            );
                            v.raw.index
                        }
                        Resource::Sampler(s) => {
                            held.push(Held::Sampler(s.raw.clone()));
                            s.raw.index
                        }
                        Resource::Buffer { .. } => panic!("a buffer isn't in the table"),
                    };
                    let at = at as usize;
                    block[at..at + 4].copy_from_slice(&index.to_le_bytes());
                    continue;
                }
                let k_set = l.numbers[k].expect("an entry of the set");
                let ty = descriptor_type(l.entries[k].ty);
                match &e.resource {
                    Resource::Buffer { buffer, offset, size } => {
                        let range = match (l.entries[k].ty, size) {
                            // A dynamic uniform's range is its block, at offsets the pass gives.
                            (BindingType::Uniform { dynamic: true, size }, _)
                            | (BindingType::Storage { dynamic: true, size, .. }, _) => size,
                            (_, Some(s)) => *s,
                            (_, None) => vk::WHOLE_SIZE,
                        };
                        buffers.push([vk::DescriptorBufferInfo {
                            buffer: buffer.raw.buffer,
                            offset: *offset,
                            range,
                        }]);
                        writes.push((k_set, ty, true, buffers.len() - 1));
                        held.push(Held::Buffer(buffer.raw.clone()));
                        uses(
                            vk::Handle::as_raw(buffer.raw.buffer),
                            matches!(
                                l.entries[k].ty,
                                BindingType::Storage { read_only: false, .. }
                            ),
                        );
                    }
                    Resource::View(v) => {
                        images.push([vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: v.raw.view,
                            image_layout: v.raw.image.home,
                        }]);
                        writes.push((k_set, ty, false, images.len() - 1));
                        held.push(Held::View(v.raw.clone()));
                        uses(
                            vk::Handle::as_raw(v.raw.image.image),
                            matches!(l.entries[k].ty, BindingType::StorageTexture { .. }),
                        );
                    }
                    Resource::Sampler(s) => {
                        images.push([vk::DescriptorImageInfo {
                            sampler: s.raw.raw,
                            image_view: vk::ImageView::null(),
                            image_layout: vk::ImageLayout::UNDEFINED,
                        }]);
                        writes.push((k_set, ty, false, images.len() - 1));
                        held.push(Held::Sampler(s.raw.clone()));
                    }
                }
            }
            let vk_writes: Vec<vk::WriteDescriptorSet<'_>> = writes
                .iter()
                .map(|&(binding, ty, buffer, at)| {
                    let w = vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(binding)
                        .descriptor_type(ty);
                    if buffer { w.buffer_info(&buffers[at]) } else { w.image_info(&images[at]) }
                })
                .collect();
            if !vk_writes.is_empty() {
                d.update_descriptor_sets(&vk_writes, &[]);
            }
            BindGroup(Arc::new(Group {
                device: self.shared(),
                layout: layout.0.clone(),
                pool,
                set,
                _held: held,
                reads: read,
                writes: written,
                block,
            }))
        }
    }

    /// A shader module of SPIR-V words (little-endian bytes): whole words, a header's at least,
    /// from its magic number.
    pub fn create_shader_module(&self, label: &str, source: &[u8]) -> Result<ShaderModule> {
        if source.len() % 4 != 0 || source.len() < 20 {
            return Err(Error(format!(
                "{label}: it isn't SPIR-V: {} bytes, not whole words of a header and more",
                source.len()
            )));
        }
        let words: Vec<u32> =
            source.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        if words[0] != 0x0723_0203 {
            return Err(Error(format!("{label}: it isn't SPIR-V: no magic number")));
        }
        // SAFETY: a module of the words.
        let raw = unsafe {
            self.raw()
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
                .map_err(|e| Error(format!("{label}: Vulkan made no shader module: {e}")))?
        };
        Ok(ShaderModule { raw: Arc::new(RawModule { device: self.shared(), raw }) })
    }

    pub fn create_compute_pipeline(
        &self,
        desc: &ComputePipelineDesc<'_>,
    ) -> Result<ComputePipeline> {
        let name = CString::new(desc.entry).map_err(|_| "an entry point's name with a NUL")?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(desc.module.raw.raw)
            .name(&name);
        let info =
            vk::ComputePipelineCreateInfo::default().stage(stage).layout(desc.layout.0.pipeline);
        // SAFETY: a pipeline from a live module and layout.
        let raw = unsafe {
            self.raw().create_compute_pipelines(vk::PipelineCache::null(), &[info], None).map_err(
                |(_, e)| Error(format!("{}: Vulkan made no compute pipeline: {e}", desc.label)),
            )?[0]
        };
        Ok(ComputePipeline {
            raw: Arc::new(RawPipeline { device: self.shared(), raw }),
            layout: desc.layout.0.clone(),
        })
    }

    pub fn create_render_pipeline(&self, desc: &RenderPipelineDesc<'_>) -> Result<RenderPipeline> {
        let vs = CString::new(desc.vertex_entry).map_err(|_| "an entry point's name with a NUL")?;
        let fs =
            CString::new(desc.fragment_entry).map_err(|_| "an entry point's name with a NUL")?;
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(desc.module.raw.raw)
                .name(&vs),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(desc.module.raw.raw)
                .name(&fs),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport =
            vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let mut raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(match desc.cull {
                Cull::None => vk::CullModeFlags::NONE,
                Cull::Back => vk::CullModeFlags::BACK,
                Cull::Front => vk::CullModeFlags::FRONT,
            })
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        if let Some(d) = &desc.depth
            && (d.bias.constant != 0 || d.bias.slope_scale != 0.0)
        {
            raster = raster
                .depth_bias_enable(true)
                .depth_bias_constant_factor(d.bias.constant as f32)
                .depth_bias_slope_factor(d.bias.slope_scale)
                .depth_bias_clamp(d.bias.clamp);
        }
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_state = match &desc.depth {
            Some(d) => vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true)
                .depth_write_enable(d.state.write)
                .depth_compare_op(compare_op(d.state.compare)),
            None => vk::PipelineDepthStencilStateCreateInfo::default(),
        };
        let attachments: Vec<vk::PipelineColorBlendAttachmentState> =
            desc.colors.iter().map(|c| blend_state(c.blend)).collect();
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
        let dynamic = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic);
        let formats: Vec<vk::Format> = desc.colors.iter().map(|c| vk_format(c.format)).collect();
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&formats)
            .depth_attachment_format(
                desc.depth.as_ref().map_or(vk::Format::UNDEFINED, |d| vk_format(d.format)),
            );
        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_state)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(desc.layout.0.pipeline)
            .push_next(&mut rendering);
        // SAFETY: a pipeline from a live module and layout, its state built above.
        let raw = unsafe {
            self.raw().create_graphics_pipelines(vk::PipelineCache::null(), &[info], None).map_err(
                |(_, e)| Error(format!("{}: Vulkan made no render pipeline: {e}", desc.label)),
            )?[0]
        };
        Ok(RenderPipeline {
            raw: Arc::new(RawPipeline { device: self.shared(), raw }),
            layout: desc.layout.0.clone(),
        })
    }

    /// A command encoder (Vulkan's command buffers carry no label here).
    pub fn create_command_encoder(&self, _label: &str) -> CommandEncoder {
        CommandEncoder::new(self.clone())
    }

    /// Writes `data` at `offset` of `buffer` before the next submission's own work.
    pub fn write_buffer(&self, buffer: &Buffer, offset: u64, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        let (staging, at) = self.staging(data, COPY_BUFFER_ALIGNMENT);
        lock(&self.0.pending).push(Pending::Buffer {
            staging,
            at,
            dst: buffer.raw.clone(),
            offset,
            len: data.len() as u64,
        });
    }

    /// Writes texels to a level and layer of `texture` before the next submission's own work:
    /// `data` holds `rows_per_image` rows (of blocks) of `bytes_per_row` each, a layer.
    pub fn write_texture(
        &self,
        at: TextureCopy<'_>,
        data: &[u8],
        bytes_per_row: u32,
        rows_per_image: u32,
        extent: [u32; 3],
    ) {
        if data.is_empty() {
            return;
        }
        let (staging, staged_at) = self.staging(data, u64::from(COPY_BYTES_PER_ROW_ALIGNMENT));
        let (layer, z) = at.texture.dimension.split(at.origin[2], 0);
        lock(&self.0.pending).push(Pending::Texture {
            staging,
            at: staged_at,
            bytes_per_row,
            rows_per_image,
            dst: at.texture.raw.clone(),
            level: at.level,
            layer,
            origin: [at.origin[0], at.origin[1], z],
            extent,
        });
    }

    /// `data` copied into the staging belt at an offset aligned to `align`: the buffer and the
    /// offset, which the next submission copies from.
    fn staging(&self, data: &[u8], align: u64) -> (Arc<RawBuffer>, u64) {
        lock(&self.0.belt).stage(data, align, |len| {
            self.raw_buffer(len, true, vk::BufferUsageFlags::TRANSFER_SRC)
        })
    }

    /// Records the staged writes, then submits them and `buffers`, in order: their submission's
    /// index.
    pub fn submit(&self, buffers: Vec<CommandBuffer>) -> SubmissionIndex {
        self.submit_waiting(buffers, None)
    }

    /// [`Device::submit`], first waiting on a binary semaphore and signalling another as it ends
    /// (a frame's image acquired, and rendered).
    fn submit_waiting(
        &self,
        buffers: Vec<CommandBuffer>,
        frame: Option<(vk::Semaphore, vk::Semaphore)>,
    ) -> SubmissionIndex {
        let pending = std::mem::take(&mut *lock(&self.0.pending));
        let index = self.0.submitted.fetch_add(1, Ordering::AcqRel) + 1;
        let done = self.completed();
        lock(&self.0.belt).turn(index, done);
        let mut keep: Vec<Keep> = Vec::new();
        let mut pools = Vec::new();
        let mut cbs = Vec::new();
        let mut timed: Vec<u64> = Vec::new();
        if !pending.is_empty() {
            let mut enc = CommandEncoder::new(self.clone());
            for w in &pending {
                enc.record_pending(w, &mut keep);
            }
            let cb = enc.finish();
            timed.extend(cb.clock);
            pools.push(cb.pool);
            cbs.push(cb.raw);
            keep.extend(cb.keep);
        }
        for b in buffers {
            timed.extend(b.clock);
            pools.push(b.pool);
            cbs.push(b.raw);
            keep.extend(b.keep);
        }
        if let (Some(&first), Some(&last)) = (timed.first(), timed.last()) {
            let mut c = lock(&self.0.clock_of);
            c.push_back((index, first, last));
            while c.len() > CLOCK_KEPT {
                c.pop_front();
            }
        }
        let signal = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.0.timeline)
            .value(index)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let cb_infos: Vec<vk::CommandBufferSubmitInfo<'_>> = cbs
            .iter()
            .map(|&cb| vk::CommandBufferSubmitInfo::default().command_buffer(cb))
            .collect();
        let mut signals = signal.to_vec();
        let mut waits = Vec::new();
        if let Some((acquired, rendered)) = frame {
            waits.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(acquired)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            );
            signals.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(rendered)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            );
        }
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&waits)
            .command_buffer_infos(&cb_infos)
            .signal_semaphore_infos(&signals);
        // SAFETY: recorded command buffers, submitted once, kept until the semaphore passes
        // `index`.
        let r = unsafe { self.raw().queue_submit2(self.0.queue, &[submit], vk::Fence::null()) };
        if let Err(e) = r {
            lock(&self.0.errors).push(format!("Vulkan refused a submission: {e}"));
            // Nothing will signal `index`: once every submission before it is done, the host
            // does, so a wait for it ends and what it held is let go (its command buffers never
            // ran). A lost device fails the wait, and is reported so.
            let sems = [self.0.timeline];
            let before = [index - 1];
            let info = vk::SemaphoreWaitInfo::default().semaphores(&sems).values(&before);
            // SAFETY: waiting on, then signalling, the timeline semaphore past every value
            // signalled or to be signalled (this is the latest submission).
            let signalled = unsafe {
                self.raw().wait_semaphores(&info, u64::MAX).and_then(|()| {
                    self.raw().signal_semaphore(
                        &vk::SemaphoreSignalInfo::default().semaphore(self.0.timeline).value(index),
                    )
                })
            };
            if let Err(e) = signalled {
                lock(&self.0.errors).push(format!("Vulkan lost the device: {e}"));
            }
        }
        let mut f = lock(&self.0.in_flight);
        // Its pools, the last holding what it keeps (none: an entry of its own, never one of a
        // submission before it).
        let mut entries: Vec<_> = pools.into_iter().map(|p| (index, p, Vec::new())).collect();
        match entries.last_mut() {
            Some(last) => last.2 = keep,
            None => entries.push((index, vk::CommandPool::null(), keep)),
        }
        f.extend(entries);
        drop(f);
        self.reclaim();
        index
    }

    /// Frees what submissions now done held.
    fn reclaim(&self) {
        let done = self.completed();
        let freed: Vec<(SubmissionIndex, vk::CommandPool, Vec<Keep>)> = {
            let mut f = lock(&self.0.in_flight);
            let (gone, kept): (Vec<_>, Vec<_>) = f.drain(..).partition(|(i, _, _)| *i <= done);
            *f = kept;
            gone
        };
        for (_, pool, _keep) in freed {
            if pool != vk::CommandPool::null() {
                // SAFETY: its submission is done.
                unsafe { self.raw().destroy_command_pool(pool, None) };
            }
        }
    }

    /// Whether it times its submissions ([`Device::gpu_ms`]): where its queue writes timestamps.
    pub fn has_clock(&self) -> bool {
        self.0.clock.is_some()
    }

    /// How long submission `index` held the GPU (ms): from its first command buffer's start,
    /// stamped once everything before it is done, to its last one's end; once it's done, while
    /// those buffers' slots aren't yet others' (the last [`CLOCK_KEPT`] buffers begun).
    pub fn gpu_ms(&self, index: SubmissionIndex) -> Option<f64> {
        let (pool, mask) = self.0.clock?;
        if index > self.completed() {
            return None;
        }
        let (first, last) = span(&lock(&self.0.clock_of), index)?;
        if first + CLOCK_KEPT as u64 <= self.0.clock_next.load(Ordering::Acquire) {
            return None;
        }
        let start = self.clock_stamp(pool, clock_slot(first, false))?;
        let end = self.clock_stamp(pool, clock_slot(last, true))?;
        Some(self.ticks_ms(end.wrapping_sub(start) & mask))
    }

    /// How long the GPU had none of this device's work before submission `index` began (ms): from
    /// the end of the submission before it to its start, on the GPU's clock (0 where they
    /// overlap); once it's done, while both are among the last [`CLOCK_KEPT`] stamped.
    pub fn idle_before_ms(&self, index: SubmissionIndex) -> Option<f64> {
        let (pool, mask) = self.0.clock?;
        if index > self.completed() || index == 0 {
            return None;
        }
        let (before, (first, _)) = {
            let c = lock(&self.0.clock_of);
            let k = span(&c, index)?;
            (span(&c, index - 1)?, k)
        };
        if before.0 + CLOCK_KEPT as u64 <= self.0.clock_next.load(Ordering::Acquire) {
            return None;
        }
        let end = self.clock_stamp(pool, clock_slot(before.1, true))?;
        let start = self.clock_stamp(pool, clock_slot(first, false))?;
        let ticks = start.wrapping_sub(end) & mask;
        Some(self.ticks_ms(if ticks > mask / 2 { 0 } else { ticks }))
    }

    /// `ticks` of the GPU's clock in ms.
    fn ticks_ms(&self, ticks: u64) -> f64 {
        ticks as f64 * f64::from(self.0.timestamp_period) * 1e-6
    }

    /// One query of the device's clock pool, read without waiting: one not yet written isn't
    /// ready, and gives nothing.
    fn clock_stamp(&self, pool: vk::QueryPool, query: u32) -> Option<u64> {
        let mut stamp = [0u64; 1];
        // SAFETY: one query of the pool, read without waiting.
        unsafe {
            self.raw().get_query_pool_results(
                pool,
                query,
                &mut stamp,
                vk::QueryResultFlags::TYPE_64,
            )
        }
        .ok()?;
        Some(stamp[0])
    }

    /// The submissions done so far: every one up to this index.
    pub fn completed(&self) -> SubmissionIndex {
        // SAFETY: the timeline semaphore's value.
        match unsafe { self.raw().get_semaphore_counter_value(self.0.timeline) } {
            Ok(v) => v,
            Err(e) => {
                lock(&self.0.errors).push(format!("Vulkan lost the device: {e}"));
                self.submitted()
            }
        }
    }

    pub fn submitted(&self) -> SubmissionIndex {
        self.0.submitted.load(Ordering::Acquire)
    }

    /// Waits for submission `index` (and every one before it) to be done.
    pub fn wait(&self, index: SubmissionIndex) {
        let index = index.min(self.submitted());
        let sems = [self.0.timeline];
        let values = [index];
        let info = vk::SemaphoreWaitInfo::default().semaphores(&sems).values(&values);
        // SAFETY: waiting on the timeline semaphore.
        if let Err(e) = unsafe { self.raw().wait_semaphores(&info, u64::MAX) } {
            lock(&self.0.errors).push(format!("Vulkan lost the device: {e}"));
        }
        self.reclaim();
    }

    /// Submits the staged writes, if any, and waits for every submission to be done.
    pub fn wait_idle(&self) {
        if !lock(&self.0.pending).is_empty() {
            self.submit(Vec::new());
        }
        self.wait(self.submitted());
    }

    /// The errors reported since the last take.
    pub fn take_errors(&self) -> Vec<String> {
        std::mem::take(&mut *lock(&self.0.errors))
    }

    /// Reports the device lost, as a real loss is reported (`completed`, `wait`): what the
    /// host's handling of a loss is tested by (`WRELA_GPU_LOSE_AT`).
    pub fn lose(&self) {
        lock(&self.0.errors).push("Vulkan lost the device: simulated (WRELA_GPU_LOSE_AT)".into());
    }
}

/// Submission `index`'s first and last timed command buffers, while `clock_of` has it.
fn span(
    clock_of: &std::collections::VecDeque<(SubmissionIndex, u64, u64)>,
    index: SubmissionIndex,
) -> Option<(u64, u64)> {
    let k = clock_of.binary_search_by_key(&index, |(i, ..)| *i).ok()?;
    Some((clock_of[k].1, clock_of[k].2))
}

impl Drop for Shared {
    fn drop(&mut self) {
        // SAFETY: the device is idle before anything of it is destroyed; what's left of each
        // submission is freed, then the device and the instance.
        unsafe {
            let _ = self.raw.device_wait_idle();
            for (_, pool, _) in lock(&self.in_flight).drain(..) {
                if pool != vk::CommandPool::null() {
                    self.raw.destroy_command_pool(pool, None);
                }
            }
            lock(&self.pending).clear();
            if let Some((pool, _)) = self.clock {
                self.raw.destroy_query_pool(pool, None);
            }
            for p in lock(&self.descriptor_pools).drain(..) {
                self.raw.destroy_descriptor_pool(p, None);
            }
            for blocks in lock(&self.heap).pools.values_mut() {
                for b in blocks.drain(..) {
                    self.raw.free_memory(b.memory, None);
                }
            }
            self.raw.destroy_descriptor_pool(self.table.pool, None);
            self.raw.destroy_descriptor_set_layout(self.table.set_layout, None);
            self.raw.destroy_semaphore(self.timeline, None);
            self.raw.destroy_device(None);
            if let Some((du, m)) = self.debug.take() {
                du.destroy_debug_utils_messenger(m, None);
            }
            self.instance.destroy_instance(None);
        }
        let _ = (self.physical, self.queue_family);
    }
}

/// The bytes of device memory a block holds: resources take ranges of it, so a device makes few
/// allocations however many resources a program has (Vulkan allows as few as 4,096 at once, and
/// each costs the kernel a call and, host-visible, a mapping). 64 MiB: a 1080p RGBA32F target
/// (33 MB) fits one.
const BLOCK: u64 = 64 << 20;

/// A pool's first block: each after it twice the last, to [`BLOCK`].
const FIRST_BLOCK: u64 = 2 << 20;

/// A resource's device memory: a range of a block (`pool` its pool), or an allocation of its own.
struct Allocation {
    memory: vk::DeviceMemory,
    offset: u64,
    size: u64,
    /// Where the host sees it, if its memory is host-visible.
    mapped: *mut u8,
    pool: Option<(u32, bool)>,
}

impl Allocation {
    const NONE: Allocation = Allocation {
        memory: vk::DeviceMemory::null(),
        offset: 0,
        size: 0,
        mapped: std::ptr::null_mut(),
        pool: None,
    };
}

/// The device's blocks, by memory type and kind (images apart from buffers).
#[derive(Default)]
struct Heap {
    pools: std::collections::HashMap<(u32, bool), Vec<Block>>,
}

struct Block {
    memory: vk::DeviceMemory,
    size: u64,
    mapped: *mut u8,
    /// Its free ranges, by offset, none touching another.
    free: Vec<(u64, u64)>,
}

// SAFETY: a block's mapping is written only through the ranges handed out, each by the one
// resource that holds it.
unsafe impl Send for Block {}

impl Block {
    /// The offset of `size` bytes aligned to `align`, taken from the first free range they fit.
    fn take(&mut self, size: u64, align: u64) -> Option<u64> {
        for k in 0..self.free.len() {
            let (start, len) = self.free[k];
            let at = start.next_multiple_of(align);
            let end = start + len;
            if at + size > end {
                continue;
            }
            self.free.remove(k);
            if at + size < end {
                self.free.insert(k, (at + size, end - at - size));
            }
            if at > start {
                self.free.insert(k, (start, at - start));
            }
            return Some(at);
        }
        None
    }

    /// Gives `size` bytes at `at` back, merged with free neighbours.
    fn give(&mut self, at: u64, size: u64) {
        let k = self.free.partition_point(|&(s, _)| s < at);
        self.free.insert(k, (at, size));
        if k + 1 < self.free.len() && self.free[k].0 + self.free[k].1 == self.free[k + 1].0 {
            self.free[k].1 += self.free[k + 1].1;
            self.free.remove(k + 1);
        }
        if k > 0 && self.free[k - 1].0 + self.free[k - 1].1 == self.free[k].0 {
            self.free[k - 1].1 += self.free[k].1;
            self.free.remove(k);
        }
    }

    fn empty(&self) -> bool {
        self.free.len() == 1 && self.free[0] == (0, self.size)
    }
}

impl Shared {
    /// Gives a resource's memory back: its range to its block (a block left empty, past one a
    /// pool, freed), or its own allocation freed.
    fn release(&self, a: &Allocation) {
        if a.memory == vk::DeviceMemory::null() {
            return;
        }
        let Some(key) = a.pool else {
            // SAFETY: nothing in flight holds the resource whose memory this is.
            unsafe { self.raw.free_memory(a.memory, None) };
            return;
        };
        let mut heap = lock(&self.heap);
        let Some(blocks) = heap.pools.get_mut(&key) else { return };
        let Some(k) = blocks.iter().position(|b| b.memory == a.memory) else { return };
        blocks[k].give(a.offset, a.size);
        if blocks[k].empty() && blocks.len() > 1 {
            let b = blocks.swap_remove(k);
            // SAFETY: no range of it is held.
            unsafe { self.raw.free_memory(b.memory, None) };
        }
    }
}

#[cfg(test)]
mod heap_tests {
    use super::*;

    #[test]
    fn a_block_gives_aligned_ranges_and_merges_what_comes_back() {
        let mut b = Block {
            memory: vk::DeviceMemory::null(),
            size: BLOCK,
            mapped: std::ptr::null_mut(),
            free: vec![(0, BLOCK)],
        };
        let x = b.take(100, 1).unwrap();
        let y = b.take(1000, 256).unwrap();
        let z = b.take(10, 4).unwrap();
        assert_eq!((x, y, z), (0, 256, 100));
        b.give(y, 1000);
        b.give(x, 100);
        b.give(z, 10);
        assert!(b.empty(), "{:?}", b.free);
        assert!(b.take(BLOCK + 1, 1).is_none());
    }
}

fn cstr(chars: &[std::ffi::c_char]) -> String {
    // SAFETY: Vulkan's fixed-size names end in a NUL.
    unsafe { CStr::from_ptr(chars.as_ptr()) }.to_string_lossy().into_owned()
}

// ---- resources ------------------------------------------------------------------------------------

struct RawBuffer {
    device: Arc<Shared>,
    buffer: vk::Buffer,
    memory: Allocation,
    size: u64,
    /// The host's view of a mappable buffer's memory; null otherwise.
    mapped: *mut u8,
}

// SAFETY: the mapping is only read or written by the host's one recording thread.
unsafe impl Send for RawBuffer {}
unsafe impl Sync for RawBuffer {}

impl StagingBuffer for Arc<RawBuffer> {
    fn size(&self) -> u64 {
        self.size
    }

    unsafe fn write(&self, at: u64, data: &[u8]) {
        // SAFETY: the caller's; a staging buffer is mapped whole.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.mapped.add(at as usize), data.len())
        };
    }
}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        // SAFETY: nothing in flight holds it (submissions keep their resources alive).
        unsafe {
            self.device.raw.destroy_buffer(self.buffer, None);
            self.device.release(&self.memory);
        }
    }
}

pub struct Buffer {
    raw: Arc<RawBuffer>,
    size: u64,
    mappable: bool,
}

impl Buffer {
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Whether it and `other` are the same buffer (clones of one).
    pub fn same(&self, other: &Buffer) -> bool {
        Arc::ptr_eq(&self.raw, &other.raw)
    }

    /// A mappable buffer's bytes `offset..offset + len`, read in place: its submission must be
    /// done.
    pub fn read(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        if !self.mappable {
            return Err("the buffer isn't mappable".into());
        }
        if offset + len > self.raw.size {
            return Err("a read past the buffer's end".into());
        }
        // SAFETY: the mapping is coherent and holds the range.
        Ok(unsafe {
            std::slice::from_raw_parts(self.raw.mapped.add(offset as usize), len as usize)
        }
        .to_vec())
    }
}

impl Clone for Buffer {
    fn clone(&self) -> Buffer {
        Buffer { raw: self.raw.clone(), size: self.size, mappable: self.mappable }
    }
}

struct RawImage {
    device: Arc<Shared>,
    image: vk::Image,
    /// Its memory; a swapchain's image has none of ours (`Allocation::NONE`).
    memory: Allocation,
    format: TextureFormat,
    aspect: vk::ImageAspectFlags,
    levels: u32,
    layers: u32,
    /// The layout it rests in between commands, by its use: `GENERAL` where a kernel writes it,
    /// else read-only optimal. A command that uses it otherwise (a pass drawing into it, a
    /// copy) moves what it uses out and back, so a compressed target stays compressed
    /// (RADV keeps no DCC or HTILE in `GENERAL`).
    home: vk::ImageLayout,
    /// A swapchain's image: not ours to free, and drawn whole each frame, so a pass takes it
    /// from `UNDEFINED`.
    swapchain: bool,
    /// What it may be used as: a view of it is in the table's arrays it can be in.
    usage: vk::ImageUsageFlags,
}

impl RawImage {
    /// Every level and layer.
    fn all(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: self.aspect,
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        }
    }

    /// One level's layers `layer..layer + count`.
    fn part(&self, level: u32, layer: u32, count: u32) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: self.aspect,
            base_mip_level: level,
            level_count: 1,
            base_array_layer: layer,
            layer_count: count,
        }
    }
}

/// The pipeline stage and access masks an image in `layout` uses: source masks wait on writes,
/// destination masks make writes visible to the next use (no `ALL_COMMANDS` stall across unrelated
/// stages).
fn layout_sync(layout: vk::ImageLayout, src: bool) -> (vk::PipelineStageFlags2, vk::AccessFlags2) {
    match layout {
        // Its contents dropped, but the move still waits on everything before (no access): the
        // stage chains it to a frame's acquisition and to the last barrier, so it can't land
        // while the display or an earlier pass still uses the image (`NONE` would order it after
        // nothing).
        vk::ImageLayout::UNDEFINED => {
            (vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::NONE)
        }
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => (
            vk::PipelineStageFlags2::COPY | vk::PipelineStageFlags2::CLEAR,
            vk::AccessFlags2::TRANSFER_WRITE,
        ),
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (
            vk::PipelineStageFlags2::COPY,
            if src { vk::AccessFlags2::NONE } else { vk::AccessFlags2::TRANSFER_READ },
        ),
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            if src {
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
            } else {
                vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
            },
        ),
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => (
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            if src {
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
            } else {
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
            },
        ),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
            vk::PipelineStageFlags2::VERTEX_SHADER
                | vk::PipelineStageFlags2::FRAGMENT_SHADER
                | vk::PipelineStageFlags2::COMPUTE_SHADER,
            if src { vk::AccessFlags2::NONE } else { vk::AccessFlags2::SHADER_READ },
        ),
        vk::ImageLayout::GENERAL => (
            vk::PipelineStageFlags2::VERTEX_SHADER
                | vk::PipelineStageFlags2::FRAGMENT_SHADER
                | vk::PipelineStageFlags2::COMPUTE_SHADER,
            if src {
                vk::AccessFlags2::SHADER_WRITE
            } else {
                vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE
            },
        ),
        vk::ImageLayout::PRESENT_SRC_KHR => (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE),
        _ => (
            vk::PipelineStageFlags2::ALL_COMMANDS,
            if src {
                vk::AccessFlags2::MEMORY_WRITE
            } else {
                vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE
            },
        ),
    }
}

/// `range` of `image` moved from layout `from` to `to`, with stage and access masks matched to
/// each layout.
fn transition(
    d: &ash::Device,
    cb: vk::CommandBuffer,
    image: vk::Image,
    range: vk::ImageSubresourceRange,
    from: vk::ImageLayout,
    to: vk::ImageLayout,
) {
    if from == to {
        return;
    }
    let (src_stage, src_access) = layout_sync(from, true);
    let (dst_stage, dst_access) = layout_sync(to, false);
    let barrier = [vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .old_layout(from)
        .new_layout(to)
        .image(image)
        .subresource_range(range)];
    // SAFETY: a layout transition in a recording command buffer.
    unsafe {
        d.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&barrier))
    };
}

impl Drop for RawImage {
    fn drop(&mut self) {
        if self.swapchain {
            return;
        }
        // SAFETY: nothing in flight holds it.
        unsafe {
            self.device.raw.destroy_image(self.image, None);
            self.device.release(&self.memory);
        }
    }
}

#[derive(Clone)]
pub struct Texture {
    raw: Arc<RawImage>,
    format: TextureFormat,
    width: u32,
    height: u32,
    depth_or_layers: u32,
    levels: u32,
    dimension: TextureDimension,
}

impl Texture {
    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn format(&self) -> TextureFormat {
        self.format
    }

    pub fn levels(&self) -> u32 {
        self.levels
    }

    pub fn depth_or_layers(&self) -> u32 {
        self.depth_or_layers
    }

    pub fn create_view(&self, desc: &ViewDesc) -> TextureView {
        let ty = match desc.dimension {
            ViewDimension::D2 => vk::ImageViewType::TYPE_2D,
            ViewDimension::D2Array => vk::ImageViewType::TYPE_2D_ARRAY,
            ViewDimension::Cube => vk::ImageViewType::CUBE,
            ViewDimension::D3 => vk::ImageViewType::TYPE_3D,
        };
        let range = vk::ImageSubresourceRange {
            aspect_mask: self.raw.aspect,
            base_mip_level: desc.base_level,
            level_count: desc.level_count.unwrap_or(self.raw.levels - desc.base_level),
            base_array_layer: desc.base_layer,
            layer_count: desc.layer_count.unwrap_or(self.raw.layers - desc.base_layer),
        };
        // SAFETY: a view of a live image.
        let view = unsafe {
            self.raw
                .device
                .raw
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(self.raw.image)
                        .view_type(ty)
                        .format(vk_format(self.raw.format))
                        .subresource_range(range),
                    None,
                )
                .expect("Vulkan made no image view")
        };
        let level = desc.base_level;
        // Its index in the table: a sampled image there, and a storage image where it can be
        // one (a kernel may write it: one level, not a cube).
        let device = &self.raw.device;
        let index = device.table.free.take();
        let usage = self.raw.usage;
        let mut writes = Vec::new();
        let sampled = [vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view: view,
            image_layout: self.raw.home,
        }];
        let storage = [vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view: view,
            image_layout: vk::ImageLayout::GENERAL,
        }];
        if usage.contains(vk::ImageUsageFlags::SAMPLED) {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(device.table.set)
                    .dst_binding(wrela_abi::table::SAMPLED)
                    .dst_array_element(index)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&sampled),
            );
        }
        if usage.contains(vk::ImageUsageFlags::STORAGE)
            && range.level_count == 1
            && ty != vk::ImageViewType::CUBE
        {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(device.table.set)
                    .dst_binding(wrela_abi::table::STORAGE)
                    .dst_array_element(index)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&storage),
            );
        }
        // SAFETY: descriptors of a live view, at an index no command in flight reaches
        // (update-after-bind, update-unused-while-pending).
        unsafe { device.raw.update_descriptor_sets(&writes, &[]) };
        TextureView {
            raw: Arc::new(RawView {
                image: self.raw.clone(),
                view,
                range,
                width: (self.width >> level).max(1),
                height: (self.height >> level).max(1),
                index,
            }),
        }
    }
}

struct RawView {
    image: Arc<RawImage>,
    view: vk::ImageView,
    /// Its levels and layers.
    range: vk::ImageSubresourceRange,
    width: u32,
    height: u32,
    /// Its index in the device's table ([`NO_INDEX`] for a swapchain's image, only drawn into).
    index: u32,
}

/// A view's index where it has none in the table.
const NO_INDEX: u32 = u32::MAX;

impl Drop for RawView {
    fn drop(&mut self) {
        // SAFETY: nothing in flight holds it.
        unsafe { self.image.device.raw.destroy_image_view(self.view, None) };
        if self.index != NO_INDEX {
            self.image.device.table.free.give_back(self.index);
        }
    }
}

#[derive(Clone)]
pub struct TextureView {
    raw: Arc<RawView>,
}

struct RawSampler {
    device: Arc<Shared>,
    raw: vk::Sampler,
    /// Its index in the device's table.
    index: u32,
}

impl Drop for RawSampler {
    fn drop(&mut self) {
        // SAFETY: nothing in flight holds it.
        unsafe { self.device.raw.destroy_sampler(self.raw, None) };
        self.device.table.free.give_back(self.index);
    }
}

#[derive(Clone)]
pub struct Sampler {
    raw: Arc<RawSampler>,
}

struct RawQueries {
    device: Arc<Shared>,
    pool: vk::QueryPool,
    count: u32,
}

impl Drop for RawQueries {
    fn drop(&mut self) {
        // SAFETY: nothing in flight holds it.
        unsafe { self.device.raw.destroy_query_pool(self.pool, None) };
    }
}

#[derive(Clone)]
pub struct QuerySet {
    raw: Arc<RawQueries>,
}

impl QuerySet {
    pub fn count(&self) -> u32 {
        self.raw.count
    }
}

struct RawModule {
    device: Arc<Shared>,
    raw: vk::ShaderModule,
}

impl Drop for RawModule {
    fn drop(&mut self) {
        // SAFETY: pipelines made of it don't need it.
        unsafe { self.device.raw.destroy_shader_module(self.raw, None) };
    }
}

#[derive(Clone)]
pub struct ShaderModule {
    raw: Arc<RawModule>,
}

struct RawPipeline {
    device: Arc<Shared>,
    raw: vk::Pipeline,
}

impl Drop for RawPipeline {
    fn drop(&mut self) {
        // SAFETY: nothing in flight holds it.
        unsafe { self.device.raw.destroy_pipeline(self.raw, None) };
    }
}

#[derive(Clone)]
pub struct ComputePipeline {
    raw: Arc<RawPipeline>,
    layout: Arc<Layout>,
}

#[derive(Clone)]
pub struct RenderPipeline {
    raw: Arc<RawPipeline>,
    layout: Arc<Layout>,
}

struct Layout {
    device: Arc<Shared>,
    entries: Vec<LayoutEntry>,
    /// Each entry's binding in the set (none for what the table holds).
    numbers: Vec<Option<u32>>,
    /// The descriptors of each type a set of it holds, for a group's pool.
    sizes: Vec<vk::DescriptorPoolSize>,
    set: vk::DescriptorSetLayout,
    /// Its set, then (where it has a block) the table's, and the block's push constants.
    pipeline: vk::PipelineLayout,
    /// The bytes of its block of indices in the device's table.
    block: usize,
}

impl Drop for Layout {
    fn drop(&mut self) {
        // SAFETY: nothing in flight holds it.
        unsafe {
            self.device.raw.destroy_pipeline_layout(self.pipeline, None);
            self.device.raw.destroy_descriptor_set_layout(self.set, None);
        }
    }
}

#[derive(Clone)]
pub struct BindGroupLayout(Arc<Layout>);

/// A resource a bind group or a command holds alive (never read but to keep it).
#[allow(dead_code)]
enum Held {
    Buffer(Arc<RawBuffer>),
    Image(Arc<RawImage>),
    View(Arc<RawView>),
    Sampler(Arc<RawSampler>),
}

struct Group {
    device: Arc<Shared>,
    layout: Arc<Layout>,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    /// The resources it binds, alive while it is.
    _held: Vec<Held>,
    /// What it binds by its handle (a buffer's or an image's) that shaders only read, and what
    /// they may write: what a command that sets it uses, for the encoder's hazards.
    reads: Vec<u64>,
    writes: Vec<u64>,
    /// Its block: the index in the device's table of each view and sampler the table holds.
    block: Vec<u8>,
}

impl Drop for Group {
    fn drop(&mut self) {
        if self.pool != vk::DescriptorPool::null() {
            // SAFETY: nothing in flight holds it: a pass holds its groups until it's done.
            unsafe { self.device.raw.destroy_descriptor_pool(self.pool, None) };
        }
    }
}

#[derive(Clone)]
pub struct BindGroup(Arc<Group>);

// ---- recording ------------------------------------------------------------------------------------

pub struct CommandEncoder {
    device: Device,
    pool: vk::CommandPool,
    raw: vk::CommandBuffer,
    /// Its slot in the device's clock (`Device::gpu_ms`), if it has one.
    clock: Option<u64>,
    /// What its commands use (groups, pipelines, resources), alive until its submission is done.
    keep: Vec<Keep>,
    /// What its commands since the last full barrier used.
    hazards: Hazards,
}

/// The resources commands since the last full barrier wrote, and read, by handle: whether the
/// next command must wait on a full barrier ([`CommandEncoder::hazard`]).
#[derive(Default)]
struct Hazards {
    written: std::collections::HashSet<u64>,
    read: std::collections::HashSet<u64>,
}

impl Hazards {
    /// Notes a command that reads `reads` and writes `writes`: whether a full barrier must come
    /// before it (it reads or writes what one since the last barrier wrote, or writes what one
    /// read), after which only its own uses are noted.
    fn note(&mut self, reads: &[u64], writes: &[u64]) -> bool {
        let conflict = reads.iter().any(|r| self.written.contains(r))
            || writes.iter().any(|w| self.written.contains(w) || self.read.contains(w));
        if conflict {
            self.clear();
        }
        self.read.extend(reads);
        self.written.extend(writes);
        conflict
    }

    /// Nothing noted: a full barrier was recorded.
    fn clear(&mut self) {
        self.written.clear();
        self.read.clear();
    }
}

#[cfg(test)]
mod hazard_tests {
    use super::*;

    /// Which of `writes` (each a staged write's staging read, if any, and its destination) wait
    /// on a barrier, as `record_pending` notes them.
    fn barriers(writes: &[(Option<u64>, u64)]) -> Vec<bool> {
        let mut h = Hazards::default();
        writes.iter().map(|(read, dst)| h.note(read.as_slice(), &[*dst])).collect()
    }

    #[test]
    fn staged_writes_wait_only_on_what_they_touch() {
        let (staging, a, b, image, other) = (1, 2, 3, 4, 5);
        // A program's start: its resources zeroed, then written from one staging buffer.
        let got = barriers(&[
            (None, a),              // fill a
            (None, b),              // fill b
            (None, image),          // clear the image
            (Some(staging), a),     // write a: after its fill
            (Some(staging), b),     // write b: its fill is behind that barrier
            (Some(staging), image), // write the image: its clear is too
            (None, other),          // clear another image
        ]);
        assert_eq!(got, [false, false, false, true, false, false, false]);
    }

    #[test]
    fn writes_into_one_resource_wait_on_each_other() {
        let (staging, a) = (1, 2);
        // `writes_apply_in_recorded_order`'s writes, one submission's worth: each waits on the
        // one before, so they land in order.
        let got =
            barriers(&[(None, a), (Some(staging), a), (Some(staging), a), (Some(staging), a)]);
        assert_eq!(got, [false, true, true, true]);
    }

    #[test]
    fn a_write_into_what_was_read_waits() {
        let mut h = Hazards::default();
        assert!(!h.note(&[7], &[8]));
        assert!(h.note(&[], &[7]), "a write after a read of it");
        assert!(!h.note(&[9], &[10]));
        assert!(h.note(&[7], &[]), "a read after a write of it");
    }
}

pub struct CommandBuffer {
    pool: vk::CommandPool,
    raw: vk::CommandBuffer,
    keep: Vec<Keep>,
    /// Its slot in the device's clock, if it times its work.
    clock: Option<u64>,
}

/// A timestamp of `stage` into query `query` of `pool`, reset before.
fn stamp(
    d: &ash::Device,
    cb: vk::CommandBuffer,
    stage: vk::PipelineStageFlags2,
    pool: vk::QueryPool,
    query: u32,
) {
    // SAFETY: a timestamp write in a recording command buffer, into a live pool's reset query.
    unsafe { d.cmd_write_timestamp2(cb, stage, pool, query) };
}

/// The query of the device's clock pool that holds the `seq`th command buffer begun's start, or
/// its `end`: two a buffer, the last [`CLOCK_KEPT`] buffers' in turn.
fn clock_slot(seq: u64, end: bool) -> u32 {
    2 * (seq % CLOCK_KEPT as u64) as u32 + u32::from(end)
}

/// Everything before, done and visible to everything after: the full barrier between commands.
fn full_barrier(d: &ash::Device, cb: vk::CommandBuffer) {
    let barrier = [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)];
    // SAFETY: a barrier in a recording command buffer.
    unsafe {
        d.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&barrier))
    };
}

impl CommandEncoder {
    fn new(device: Device) -> CommandEncoder {
        // SAFETY: a pool of one command buffer, begun for one submission.
        unsafe {
            let d = device.raw();
            let pool = d
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(device.0.queue_family)
                        .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                    None,
                )
                .expect("Vulkan made no command pool");
            let raw = d
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .expect("Vulkan allocated no command buffer")[0];
            d.begin_command_buffer(
                raw,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .expect("Vulkan began no command buffer");
            // What the submissions before did, this one sees.
            full_barrier(d, raw);
            // Its slot in the clock's pool, reset, and its start once everything before is done.
            let clock = device.0.clock.map(|(queries, _)| {
                let seq = device.0.clock_next.fetch_add(1, Ordering::AcqRel);
                let slot = clock_slot(seq, false);
                d.cmd_reset_query_pool(raw, queries, slot, 2);
                stamp(d, raw, vk::PipelineStageFlags2::ALL_COMMANDS, queries, slot);
                seq
            });
            CommandEncoder {
                device,
                pool,
                raw,
                clock,
                keep: Vec::new(),
                hazards: Hazards::default(),
            }
        }
    }

    fn d(&self) -> &ash::Device {
        &self.device.0.raw
    }

    /// Keeps `h` alive until the submission is done.
    fn hold(&mut self, h: Held) {
        self.keep.push(Keep::Held(h));
    }

    /// Before a command that reads `reads` and writes `writes`: a barrier, if it reads or
    /// writes what a command since the last one wrote, or writes what one read; then notes
    /// its own.
    fn hazard(&mut self, reads: &[u64], writes: &[u64]) {
        if self.hazards.note(reads, writes) {
            full_barrier(self.d(), self.raw);
        }
    }

    /// The full barrier, and nothing pending after it.
    fn barrier_all(&mut self) {
        full_barrier(self.d(), self.raw);
        self.hazards.clear();
    }

    /// One staged write, after a barrier only where it uses what one before it wrote (a write
    /// into what was just filled or cleared, two writes into one resource). That is all the
    /// ordering the staged writes need: their staging buffers are written by the host before
    /// the submission and only read here; their layout moves are their own image barriers; and
    /// whatever comes after them, in this submission or a later one, is recorded by another
    /// encoder, whose command buffer begins with a full barrier ([`CommandEncoder::new`]), as
    /// this one's does for what came before.
    fn record_pending(&mut self, w: &Pending, keep: &mut Vec<Keep>) {
        let (read, written) = w.uses();
        self.hazard(read.as_slice(), &[written]);
        let cb = self.raw;
        // SAFETY: copies, fills and clears in a recording command buffer, of live resources the
        // submission keeps.
        unsafe {
            let d = &self.device.0.raw;
            match w {
                Pending::Buffer { staging, at, dst, offset, len } => {
                    let region =
                        [vk::BufferCopy { src_offset: *at, dst_offset: *offset, size: *len }];
                    d.cmd_copy_buffer(cb, staging.buffer, dst.buffer, &region);
                    keep.push(Keep::Buffer(staging.clone()));
                    keep.push(Keep::Buffer(dst.clone()));
                }
                Pending::Fill { dst } => {
                    d.cmd_fill_buffer(cb, dst.buffer, 0, vk::WHOLE_SIZE, 0);
                    keep.push(Keep::Buffer(dst.clone()));
                }
                Pending::Texture {
                    staging,
                    at,
                    bytes_per_row,
                    rows_per_image,
                    dst,
                    level,
                    layer,
                    origin,
                    extent,
                } => {
                    let region = [vk::BufferImageCopy {
                        buffer_offset: *at,
                        buffer_row_length: texels_per_row(dst.format, *bytes_per_row),
                        buffer_image_height: rows_per_image * dst.format.block(),
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: dst.aspect,
                            mip_level: *level,
                            base_array_layer: *layer,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D {
                            x: origin[0] as i32,
                            y: origin[1] as i32,
                            z: origin[2] as i32,
                        },
                        image_extent: vk::Extent3D {
                            width: extent[0],
                            height: extent[1],
                            depth: extent[2],
                        },
                    }];
                    let part = dst.part(*level, *layer, 1);
                    let to = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                    transition(d, cb, dst.image, part, dst.home, to);
                    d.cmd_copy_buffer_to_image(cb, staging.buffer, dst.image, to, &region);
                    transition(d, cb, dst.image, part, to, dst.home);
                    keep.push(Keep::Buffer(staging.clone()));
                    keep.push(Keep::Image(dst.clone()));
                }
                Pending::Clear { dst, zeros } => {
                    keep.push(Keep::Image(dst.clone()));
                    // From nothing to a copy's target, zeroed, then home.
                    let range = dst.all();
                    let to = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                    transition(d, cb, dst.image, range, vk::ImageLayout::UNDEFINED, to);
                    if let Some((staging, at, extent)) = zeros {
                        // Each level whole, every layer, from the same zeros (tightly packed).
                        let regions: Vec<vk::BufferImageCopy> = (0..dst.levels)
                            .map(|level| vk::BufferImageCopy {
                                buffer_offset: *at,
                                buffer_row_length: 0,
                                buffer_image_height: 0,
                                image_subresource: vk::ImageSubresourceLayers {
                                    aspect_mask: dst.aspect,
                                    mip_level: level,
                                    base_array_layer: 0,
                                    layer_count: dst.layers,
                                },
                                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                                image_extent: vk::Extent3D {
                                    width: (extent.width >> level).max(1),
                                    height: (extent.height >> level).max(1),
                                    depth: (extent.depth >> level).max(1),
                                },
                            })
                            .collect();
                        d.cmd_copy_buffer_to_image(cb, staging.buffer, dst.image, to, &regions);
                        keep.push(Keep::Buffer(staging.clone()));
                    } else if dst.aspect == vk::ImageAspectFlags::DEPTH {
                        d.cmd_clear_depth_stencil_image(
                            cb,
                            dst.image,
                            to,
                            &vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
                            &[range],
                        );
                    } else {
                        d.cmd_clear_color_image(
                            cb,
                            dst.image,
                            to,
                            &vk::ClearColorValue { uint32: [0; 4] },
                            &[range],
                        );
                    }
                    transition(d, cb, dst.image, range, to, dst.home);
                }
            }
        }
    }

    pub fn copy_buffer_to_buffer(
        &mut self,
        src: &Buffer,
        src_offset: u64,
        dst: &Buffer,
        dst_offset: u64,
        size: u64,
    ) {
        // An empty copy is nothing (WebGPU allows one; Vulkan doesn't).
        if size == 0 {
            return;
        }
        let region = [vk::BufferCopy { src_offset, dst_offset, size }];
        self.hazard(&[vk::Handle::as_raw(src.raw.buffer)], &[vk::Handle::as_raw(dst.raw.buffer)]);
        // SAFETY: a copy between live buffers the encoder keeps.
        unsafe { self.d().cmd_copy_buffer(self.raw, src.raw.buffer, dst.raw.buffer, &region) };
        self.hold(Held::Buffer(src.raw.clone()));
        self.hold(Held::Buffer(dst.raw.clone()));
    }

    fn buffer_image(
        b: &BufferCopy<'_>,
        t: &TextureCopy<'_>,
        extent: [u32; 3],
    ) -> vk::BufferImageCopy {
        let format = t.texture.raw.format;
        let (layer, z) = t.texture.dimension.split(t.origin[2], 0);
        let (layers, depth) = t.texture.dimension.split(extent[2], 1);
        vk::BufferImageCopy {
            buffer_offset: b.offset,
            buffer_row_length: texels_per_row(format, b.bytes_per_row),
            buffer_image_height: b.rows_per_image * format.block(),
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: t.texture.raw.aspect,
                mip_level: t.level,
                base_array_layer: layer,
                layer_count: layers,
            },
            image_offset: vk::Offset3D {
                x: t.origin[0] as i32,
                y: t.origin[1] as i32,
                z: z as i32,
            },
            image_extent: vk::Extent3D { width: extent[0], height: extent[1], depth },
        }
    }

    pub fn copy_buffer_to_texture(
        &mut self,
        src: BufferCopy<'_>,
        dst: TextureCopy<'_>,
        extent: [u32; 3],
    ) {
        if extent.contains(&0) {
            return;
        }
        let region = [Self::buffer_image(&src, &dst, extent)];
        self.hazard(
            &[vk::Handle::as_raw(src.buffer.raw.buffer)],
            &[vk::Handle::as_raw(dst.texture.raw.image)],
        );
        let image = &dst.texture.raw;
        let sub = region[0].image_subresource;
        let part = image.part(sub.mip_level, sub.base_array_layer, sub.layer_count);
        let to = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        transition(self.d(), self.raw, image.image, part, image.home, to);
        // SAFETY: a copy between live resources the encoder keeps.
        unsafe {
            self.d().cmd_copy_buffer_to_image(
                self.raw,
                src.buffer.raw.buffer,
                image.image,
                to,
                &region,
            )
        };
        transition(self.d(), self.raw, image.image, part, to, image.home);
        self.hold(Held::Buffer(src.buffer.raw.clone()));
        self.hold(Held::Image(image.clone()));
    }

    pub fn copy_texture_to_buffer(
        &mut self,
        src: TextureCopy<'_>,
        dst: BufferCopy<'_>,
        extent: [u32; 3],
    ) {
        if extent.contains(&0) {
            return;
        }
        let region = [Self::buffer_image(&dst, &src, extent)];
        self.hazard(
            &[vk::Handle::as_raw(src.texture.raw.image)],
            &[vk::Handle::as_raw(dst.buffer.raw.buffer)],
        );
        let image = &src.texture.raw;
        let sub = region[0].image_subresource;
        let part = image.part(sub.mip_level, sub.base_array_layer, sub.layer_count);
        let from = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        transition(self.d(), self.raw, image.image, part, image.home, from);
        // SAFETY: a copy between live resources the encoder keeps.
        unsafe {
            self.d().cmd_copy_image_to_buffer(
                self.raw,
                image.image,
                from,
                dst.buffer.raw.buffer,
                &region,
            )
        };
        transition(self.d(), self.raw, image.image, part, from, image.home);
        self.hold(Held::Buffer(dst.buffer.raw.clone()));
        self.hold(Held::Image(image.clone()));
    }

    pub fn copy_texture_to_texture(
        &mut self,
        src: TextureCopy<'_>,
        dst: TextureCopy<'_>,
        extent: [u32; 3],
    ) {
        if extent.contains(&0) {
            return;
        }
        let (sl, sz) = src.texture.dimension.split(src.origin[2], 0);
        let (dl, dz) = dst.texture.dimension.split(dst.origin[2], 0);
        let (n, depth) = src.texture.dimension.split(extent[2], 1);
        let region = [vk::ImageCopy {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask: src.texture.raw.aspect,
                mip_level: src.level,
                base_array_layer: sl,
                layer_count: n,
            },
            src_offset: vk::Offset3D {
                x: src.origin[0] as i32,
                y: src.origin[1] as i32,
                z: sz as i32,
            },
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: dst.texture.raw.aspect,
                mip_level: dst.level,
                base_array_layer: dl,
                layer_count: n,
            },
            dst_offset: vk::Offset3D {
                x: dst.origin[0] as i32,
                y: dst.origin[1] as i32,
                z: dz as i32,
            },
            extent: vk::Extent3D { width: extent[0], height: extent[1], depth },
        }];
        self.hazard(
            &[vk::Handle::as_raw(src.texture.raw.image)],
            &[vk::Handle::as_raw(dst.texture.raw.image)],
        );
        let (a, b) = (&src.texture.raw, &dst.texture.raw);
        let pa = a.part(src.level, sl, n);
        let pb = b.part(dst.level, dl, n);
        // (One image to another place of itself: the general layout for both.)
        let same = a.image == b.image;
        let (from, to) = if same {
            (vk::ImageLayout::GENERAL, vk::ImageLayout::GENERAL)
        } else {
            (vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        };
        let (d, cb) = (self.d(), self.raw);
        transition(d, cb, a.image, pa, a.home, from);
        transition(d, cb, b.image, pb, b.home, to);
        // SAFETY: a copy between live images.
        unsafe { d.cmd_copy_image(cb, a.image, from, b.image, to, &region) };
        transition(d, cb, a.image, pa, from, a.home);
        transition(d, cb, b.image, pb, to, b.home);
        let (a, b) = (a.clone(), b.clone());
        self.hold(Held::Image(a));
        self.hold(Held::Image(b));
    }

    /// Timestamps `first..first + count` as 64-bit ticks into `dst` at `offset`.
    pub fn resolve_query_set(
        &mut self,
        queries: &QuerySet,
        first: u32,
        count: u32,
        dst: &Buffer,
        offset: u64,
    ) {
        self.barrier_all();
        // SAFETY: results of written queries, into a live buffer. Without `WAIT`: the barrier
        // before has every timestamp recorded before it written, and one never written leaves
        // its slot as it was, where `WAIT` would hold the GPU until it's lost.
        unsafe {
            self.d().cmd_copy_query_pool_results(
                self.raw,
                queries.raw.pool,
                first,
                count,
                dst.raw.buffer,
                offset,
                8,
                vk::QueryResultFlags::TYPE_64,
            )
        };
        self.hold(Held::Buffer(dst.raw.clone()));
        self.keep.push(Keep::Queries(queries.raw.clone()));
        self.barrier_all();
    }

    /// A pass's timestamps reset, and its begin written.
    fn reset(&mut self, t: Option<PassTimestamps<'_>>) {
        if let Some(t) = t {
            let pool = t.queries.raw.pool;
            // SAFETY: resets outside any pass.
            unsafe {
                self.d().cmd_reset_query_pool(self.raw, pool, t.begin, 1);
                self.d().cmd_reset_query_pool(self.raw, pool, t.end, 1);
            }
            stamp(self.d(), self.raw, vk::PipelineStageFlags2::TOP_OF_PIPE, pool, t.begin);
        }
    }

    /// A pass's end timestamp, if it's timed, written as it ends; its queries kept.
    fn end_pass(&mut self, end: Option<(Arc<RawQueries>, u32)>) {
        if let Some((q, end)) = end {
            stamp(self.d(), self.raw, vk::PipelineStageFlags2::ALL_COMMANDS, q.pool, end);
            self.keep.push(Keep::Queries(q));
        }
    }

    /// Whether a timed compute pass is open for an untimed one to join: never here. A timed pass
    /// writes its own stamps around its dispatch, with no encoder to switch, so each is timed
    /// alone (Metal's joins them: an encoder each cost its frames).
    pub fn timed_compute_open(&self) -> bool {
        false
    }

    pub fn begin_compute_pass(
        &mut self,
        label: &str,
        timestamps: Option<PassTimestamps<'_>>,
    ) -> ComputePass<'_> {
        let _ = label;
        self.reset(timestamps);
        let end = timestamps.map(|t| (t.queries.raw.clone(), t.end));
        ComputePass { encoder: self, layout: None, end, bound: None }
    }

    /// A pass's attachment `v` moved out of its home into `to`, the layout drawing takes, to go
    /// home as the pass ends (`targets`), and kept until the submission is done. A swapchain's
    /// image comes from `UNDEFINED`: it's drawn whole each frame.
    fn attach(
        &mut self,
        targets: &mut Vec<(Arc<RawView>, vk::ImageLayout)>,
        to: vk::ImageLayout,
        v: &Arc<RawView>,
    ) {
        let from = if v.image.swapchain { vk::ImageLayout::UNDEFINED } else { v.image.home };
        transition(self.d(), self.raw, v.image.image, v.range, from, to);
        targets.push((v.clone(), to));
        self.hold(Held::View(v.clone()));
    }

    pub fn begin_render_pass(&mut self, desc: &RenderPassDesc<'_>) -> RenderPass<'_> {
        self.barrier_all();
        self.reset(desc.timestamps);
        // Its targets, out of their homes into the layouts drawing takes.
        let mut targets: Vec<(Arc<RawView>, vk::ImageLayout)> = Vec::new();
        for c in desc.colors {
            self.attach(&mut targets, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, &c.view.raw);
        }
        if let Some(d) = &desc.depth {
            self.attach(&mut targets, vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL, &d.view.raw);
        }
        let colors: Vec<vk::RenderingAttachmentInfo<'_>> = desc
            .colors
            .iter()
            .map(|c| {
                let a = vk::RenderingAttachmentInfo::default()
                    .image_view(c.view.raw.view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .store_op(vk::AttachmentStoreOp::STORE);
                match c.load {
                    LoadOp::Load => a.load_op(vk::AttachmentLoadOp::LOAD),
                    LoadOp::Clear(v) => {
                        a.load_op(vk::AttachmentLoadOp::CLEAR).clear_value(vk::ClearValue {
                            color: vk::ClearColorValue { float32: v.map(|x| x as f32) },
                        })
                    }
                }
            })
            .collect();
        let depth = desc.depth.as_ref().map(|d| {
            let a = vk::RenderingAttachmentInfo::default()
                .image_view(d.view.raw.view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .store_op(vk::AttachmentStoreOp::STORE);
            match d.load {
                LoadOp::Load => a.load_op(vk::AttachmentLoadOp::LOAD),
                LoadOp::Clear(v) => {
                    a.load_op(vk::AttachmentLoadOp::CLEAR).clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue { depth: v, stencil: 0 },
                    })
                }
            }
        });
        let (w, h) = desc
            .colors
            .first()
            .map(|c| (c.view.raw.width, c.view.raw.height))
            .or_else(|| desc.depth.as_ref().map(|d| (d.view.raw.width, d.view.raw.height)))
            .unwrap_or((1, 1));
        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width: w, height: h },
        };
        let mut info = vk::RenderingInfo::default()
            .render_area(area)
            .layer_count(1)
            .color_attachments(&colors);
        if let Some(d) = &depth {
            info = info.depth_attachment(d);
        }
        // SAFETY: rendering to live views in GENERAL; the viewport and scissor cover the target
        // until the pass sets its own.
        unsafe {
            let d = self.d();
            d.cmd_begin_rendering(self.raw, &info);
            d.cmd_set_viewport(self.raw, 0, &[viewport(0.0, 0.0, w as f32, h as f32, 0.0, 1.0)]);
            d.cmd_set_scissor(self.raw, 0, &[area]);
        }
        let end = desc.timestamps.map(|t| (t.queries.raw.clone(), t.end));
        RenderPass { encoder: self, layout: None, index: None, end, targets }
    }

    pub fn finish(mut self) -> CommandBuffer {
        if let (Some(seq), Some((queries, _))) = (self.clock, self.device.0.clock) {
            // Its end, after everything recorded in it.
            let slot = clock_slot(seq, true);
            stamp(self.d(), self.raw, vk::PipelineStageFlags2::ALL_COMMANDS, queries, slot);
        }
        // What it wrote, the host sees once its submission is done (a mappable buffer read in
        // place, `Buffer::read`): waiting for the submission alone makes no write visible to
        // the host.
        let to_host = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)];
        // SAFETY: a barrier in a recording command buffer.
        unsafe {
            self.d().cmd_pipeline_barrier2(
                self.raw,
                &vk::DependencyInfo::default().memory_barriers(&to_host),
            )
        };
        // SAFETY: the recording ends.
        unsafe { self.d().end_command_buffer(self.raw).expect("Vulkan ended no command buffer") };
        let keep = std::mem::take(&mut self.keep);
        let pool = std::mem::replace(&mut self.pool, vk::CommandPool::null());
        CommandBuffer { pool, raw: self.raw, keep, clock: self.clock }
    }
}

impl Drop for CommandEncoder {
    fn drop(&mut self) {
        if self.pool != vk::CommandPool::null() {
            // SAFETY: never submitted: nothing runs it.
            unsafe { self.device.0.raw.destroy_command_pool(self.pool, None) };
        }
    }
}

pub struct ComputePass<'a> {
    encoder: &'a mut CommandEncoder,
    layout: Option<vk::PipelineLayout>,
    end: Option<(Arc<RawQueries>, u32)>,
    /// The bind group set: what the next dispatch uses.
    bound: Option<Arc<Group>>,
}

impl ComputePass<'_> {
    pub fn set_pipeline(&mut self, p: &ComputePipeline) {
        // SAFETY: a live pipeline the encoder keeps.
        unsafe {
            self.encoder.d().cmd_bind_pipeline(
                self.encoder.raw,
                vk::PipelineBindPoint::COMPUTE,
                p.raw.raw,
            )
        };
        self.layout = Some(p.layout.pipeline);
        self.encoder.keep.push(Keep::Pipeline(p.raw.clone()));
    }

    pub fn set_bind_group(&mut self, group: &BindGroup, offsets: &[u32]) {
        let layout = self.layout.unwrap_or(group.0.layout.pipeline);
        bind_group(self.encoder, vk::PipelineBindPoint::COMPUTE, layout, &group.0, offsets);
        self.encoder.keep.push(Keep::Group(group.0.clone()));
        self.bound = Some(group.0.clone());
    }

    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        let (reads, writes) = uses(self.bound.as_deref());
        self.encoder.hazard(reads, writes);
        // SAFETY: a dispatch of the bound pipeline, after any barrier its uses need.
        unsafe { self.encoder.d().cmd_dispatch(self.encoder.raw, x, y, z) };
    }

    pub fn dispatch_indirect(&mut self, buffer: &Buffer, offset: u64) {
        let (bound, writes) = uses(self.bound.as_deref());
        let reads: Vec<u64> = std::iter::once(vk::Handle::as_raw(buffer.raw.buffer))
            .chain(bound.iter().copied())
            .collect();
        self.encoder.hazard(&reads, writes);
        // SAFETY: an indirect dispatch of the bound pipeline, from a live buffer.
        unsafe {
            self.encoder.d().cmd_dispatch_indirect(self.encoder.raw, buffer.raw.buffer, offset)
        };
        self.encoder.hold(Held::Buffer(buffer.raw.clone()));
    }
}

/// What a dispatch under bound group `g` reads, and writes.
fn uses(g: Option<&Group>) -> (&[u64], &[u64]) {
    g.map_or((&[], &[]), |g| (&g.reads, &g.writes))
}

impl Drop for ComputePass<'_> {
    fn drop(&mut self) {
        self.encoder.end_pass(self.end.take());
    }
}

pub struct RenderPass<'a> {
    encoder: &'a mut CommandEncoder,
    /// Its targets, and the layouts they're in while it draws: home again as it ends.
    targets: Vec<(Arc<RawView>, vk::ImageLayout)>,
    layout: Option<vk::PipelineLayout>,
    /// The index buffer and its offset, for the indexed draws after it.
    index: Option<(vk::Buffer, u64)>,
    end: Option<(Arc<RawQueries>, u32)>,
}

impl RenderPass<'_> {
    pub fn set_pipeline(&mut self, p: &RenderPipeline) {
        // SAFETY: a live pipeline the encoder keeps.
        unsafe {
            self.encoder.d().cmd_bind_pipeline(
                self.encoder.raw,
                vk::PipelineBindPoint::GRAPHICS,
                p.raw.raw,
            )
        };
        self.layout = Some(p.layout.pipeline);
        self.encoder.keep.push(Keep::Pipeline(p.raw.clone()));
    }

    pub fn set_bind_group(&mut self, group: &BindGroup, offsets: &[u32]) {
        let layout = self.layout.unwrap_or(group.0.layout.pipeline);
        bind_group(self.encoder, vk::PipelineBindPoint::GRAPHICS, layout, &group.0, offsets);
        self.encoder.keep.push(Keep::Group(group.0.clone()));
    }

    pub fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, min: f32, max: f32) {
        // SAFETY: dynamic state.
        unsafe {
            self.encoder.d().cmd_set_viewport(
                self.encoder.raw,
                0,
                &[viewport(x, y, w, h, min, max)],
            )
        };
    }

    pub fn set_scissor_rect(&mut self, x: u32, y: u32, w: u32, h: u32) {
        // SAFETY: dynamic state.
        unsafe {
            self.encoder.d().cmd_set_scissor(
                self.encoder.raw,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: x as i32, y: y as i32 },
                    extent: vk::Extent2D { width: w, height: h },
                }],
            )
        };
    }

    pub fn set_index_buffer(&mut self, buffer: &Buffer, offset: u64) {
        self.index = Some((buffer.raw.buffer, offset));
        self.encoder.hold(Held::Buffer(buffer.raw.clone()));
        // SAFETY: a live buffer of u32 indices.
        unsafe {
            self.encoder.d().cmd_bind_index_buffer(
                self.encoder.raw,
                buffer.raw.buffer,
                offset,
                vk::IndexType::UINT32,
            )
        };
    }

    pub fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) {
        // SAFETY: a draw of the bound pipeline.
        unsafe {
            self.encoder.d().cmd_draw(
                self.encoder.raw,
                vertices.len() as u32,
                instances.len() as u32,
                vertices.start,
                instances.start,
            )
        };
    }

    pub fn draw_indexed(
        &mut self,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) {
        let _ = self.index.expect("an index buffer is set");
        // SAFETY: an indexed draw of the bound pipeline.
        unsafe {
            self.encoder.d().cmd_draw_indexed(
                self.encoder.raw,
                indices.len() as u32,
                instances.len() as u32,
                indices.start,
                base_vertex,
                instances.start,
            )
        };
    }

    pub fn draw_indirect(&mut self, buffer: &Buffer, offset: u64) {
        self.multi_draw_indirect(buffer, offset, 1);
    }

    pub fn draw_indexed_indirect(&mut self, buffer: &Buffer, offset: u64) {
        self.multi_draw_indexed_indirect(buffer, offset, 1);
    }

    /// `count` indirect draws, their arguments 16 bytes apart from `offset`.
    pub fn multi_draw_indirect(&mut self, buffer: &Buffer, offset: u64, count: u32) {
        self.encoder.hold(Held::Buffer(buffer.raw.clone()));
        // SAFETY: indirect draws from a live buffer.
        unsafe {
            self.encoder.d().cmd_draw_indirect(
                self.encoder.raw,
                buffer.raw.buffer,
                offset,
                count,
                16,
            )
        };
    }

    /// `count` indexed indirect draws, their arguments 20 bytes apart from `offset`.
    pub fn multi_draw_indexed_indirect(&mut self, buffer: &Buffer, offset: u64, count: u32) {
        let _ = self.index.expect("an index buffer is set");
        self.encoder.hold(Held::Buffer(buffer.raw.clone()));
        // SAFETY: indexed indirect draws from a live buffer.
        unsafe {
            self.encoder.d().cmd_draw_indexed_indirect(
                self.encoder.raw,
                buffer.raw.buffer,
                offset,
                count,
                20,
            )
        };
    }
}

impl Drop for RenderPass<'_> {
    fn drop(&mut self) {
        // SAFETY: the pass ends; what it wrote, the next command sees.
        unsafe { self.encoder.d().cmd_end_rendering(self.encoder.raw) };
        self.encoder.end_pass(self.end.take());
        for (v, from) in std::mem::take(&mut self.targets) {
            transition(
                self.encoder.d(),
                self.encoder.raw,
                v.image.image,
                v.range,
                from,
                v.image.home,
            );
        }
        self.encoder.barrier_all();
    }
}

// ---- conversions ----------------------------------------------------------------------------------

/// Sets group `g` at `point` for the commands after it, under pipeline layout `layout`: its set
/// (set 0, at `offsets`) and, where it has a block, the device's table (set
/// `wrela_abi::table::TABLE_SET`) and the block as push constants.
fn bind_group(
    encoder: &CommandEncoder,
    point: vk::PipelineBindPoint,
    layout: vk::PipelineLayout,
    g: &Group,
    offsets: &[u32],
) {
    let d = encoder.d();
    let table = g.device.table.set;
    // SAFETY: live sets the encoder keeps (the group, and the device's table), and a block of the
    // size of the layout's push constants.
    unsafe {
        match (g.set != vk::DescriptorSet::null(), g.block.is_empty()) {
            (true, true) => {
                d.cmd_bind_descriptor_sets(encoder.raw, point, layout, 0, &[g.set], offsets)
            }
            (true, false) => {
                d.cmd_bind_descriptor_sets(encoder.raw, point, layout, 0, &[g.set, table], offsets)
            }
            (false, false) => d.cmd_bind_descriptor_sets(
                encoder.raw,
                point,
                layout,
                wrela_abi::table::TABLE_SET,
                &[table],
                &[],
            ),
            (false, true) => {}
        }
        if !g.block.is_empty() {
            d.cmd_push_constants(encoder.raw, layout, vk::ShaderStageFlags::ALL, 0, &g.block);
        }
    }
}

/// A WebGPU viewport as Vulkan's: flipped, a negative height from its bottom edge, as wgpu-hal
/// does, since WebGPU's clip space has y up and Vulkan's down (and the compiler's SPIR-V writes
/// WebGPU's positions).
fn viewport(x: f32, y: f32, w: f32, h: f32, min: f32, max: f32) -> vk::Viewport {
    vk::Viewport { x, y: y + h, width: w, height: -h, min_depth: min, max_depth: max }
}

pub fn vk_format(f: TextureFormat) -> vk::Format {
    match f {
        TextureFormat::Rgba8 => vk::Format::R8G8B8A8_UNORM,
        TextureFormat::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
        TextureFormat::Depth32Float => vk::Format::D32_SFLOAT,
        TextureFormat::R16Float => vk::Format::R16_SFLOAT,
        TextureFormat::Rg16Float => vk::Format::R16G16_SFLOAT,
        TextureFormat::R32Float => vk::Format::R32_SFLOAT,
        TextureFormat::R32Uint => vk::Format::R32_UINT,
        TextureFormat::R8 => vk::Format::R8_UNORM,
        TextureFormat::Rg8 => vk::Format::R8G8_UNORM,
        TextureFormat::Rgb10a2 => vk::Format::A2B10G10R10_UNORM_PACK32,
        TextureFormat::Rg32Float => vk::Format::R32G32_SFLOAT,
        TextureFormat::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
        TextureFormat::Rg32Uint => vk::Format::R32G32_UINT,
        TextureFormat::Rg11b10Float => vk::Format::B10G11R11_UFLOAT_PACK32,
        TextureFormat::Bc1 => vk::Format::BC1_RGBA_UNORM_BLOCK,
        TextureFormat::Bc4 => vk::Format::BC4_UNORM_BLOCK,
        TextureFormat::Bc5 => vk::Format::BC5_UNORM_BLOCK,
        TextureFormat::Bc7 => vk::Format::BC7_UNORM_BLOCK,
        TextureFormat::Etc2Rgb8 => vk::Format::ETC2_R8G8B8_UNORM_BLOCK,
        TextureFormat::Astc4x4 => vk::Format::ASTC_4X4_UNORM_BLOCK,
    }
}

/// A copy's row of `bytes_per_row` bytes of `format`, in texels (whole blocks of a compressed
/// format), as `VkBufferImageCopy` takes its length.
fn texels_per_row(format: TextureFormat, bytes_per_row: u32) -> u32 {
    bytes_per_row / format.bytes_per_texel() * format.block()
}

fn compare_op(c: Compare) -> vk::CompareOp {
    match c {
        Compare::Never => vk::CompareOp::NEVER,
        Compare::Less => vk::CompareOp::LESS,
        Compare::Equal => vk::CompareOp::EQUAL,
        Compare::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        Compare::Greater => vk::CompareOp::GREATER,
        Compare::NotEqual => vk::CompareOp::NOT_EQUAL,
        Compare::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
        Compare::Always => vk::CompareOp::ALWAYS,
    }
}

fn descriptor_type(t: BindingType) -> vk::DescriptorType {
    match t {
        BindingType::Uniform { dynamic: true, .. } => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
        BindingType::Uniform { dynamic: false, .. } => vk::DescriptorType::UNIFORM_BUFFER,
        BindingType::Storage { dynamic: true, .. } => vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
        BindingType::Storage { dynamic: false, .. } => vk::DescriptorType::STORAGE_BUFFER,
        BindingType::Texture { .. } => vk::DescriptorType::SAMPLED_IMAGE,
        BindingType::StorageTexture { .. } => vk::DescriptorType::STORAGE_IMAGE,
        BindingType::Sampler { .. } => vk::DescriptorType::SAMPLER,
    }
}

fn stages(s: Stages) -> vk::ShaderStageFlags {
    let mut f = vk::ShaderStageFlags::empty();
    if s.vertex {
        f |= vk::ShaderStageFlags::VERTEX;
    }
    if s.fragment {
        f |= vk::ShaderStageFlags::FRAGMENT;
    }
    if s.compute {
        f |= vk::ShaderStageFlags::COMPUTE;
    }
    f
}

fn blend_state(b: Blend) -> vk::PipelineColorBlendAttachmentState {
    let state = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA);
    let Some((c, a)) = b.components() else { return state };
    state
        .blend_enable(true)
        .src_color_blend_factor(blend_factor(c.0))
        .dst_color_blend_factor(blend_factor(c.1))
        .color_blend_op(blend_op(c.2))
        .src_alpha_blend_factor(blend_factor(a.0))
        .dst_alpha_blend_factor(blend_factor(a.1))
        .alpha_blend_op(blend_op(a.2))
}

fn blend_factor(name: &str) -> vk::BlendFactor {
    match name {
        "zero" => vk::BlendFactor::ZERO,
        "one" => vk::BlendFactor::ONE,
        "src-alpha" => vk::BlendFactor::SRC_ALPHA,
        "one-minus-src-alpha" => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        "dst" => vk::BlendFactor::DST_COLOR,
        "dst-alpha" => vk::BlendFactor::DST_ALPHA,
        other => unreachable!("Blend::components names no factor `{other}`"),
    }
}

fn blend_op(name: &str) -> vk::BlendOp {
    match name {
        "add" => vk::BlendOp::ADD,
        "min" => vk::BlendOp::MIN,
        "max" => vk::BlendOp::MAX,
        other => unreachable!("Blend::components names no operation `{other}`"),
    }
}

// ---- a window's surface -------------------------------------------------------------------------

/// A window's surface and its swapchain. Its frames are `R8G8B8A8_UNORM`, the screen's format,
/// so presenting a frame is drawing the screen into it.
pub struct Surface {
    device: Device,
    surface: vk::SurfaceKHR,
    surface_fns: ash::khr::surface::Instance,
    swapchain_fns: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    images: Vec<Arc<RawView>>,
    /// Semaphores an acquisition signals, used in turn, each with the submission that last
    /// waited on it (reused once that's done).
    acquired: Vec<(vk::Semaphore, SubmissionIndex)>,
    next: usize,
    /// One a submission signals for each image, which its presentation waits on.
    rendered: Vec<vk::Semaphore>,
    /// How its frames are shown, asked (`set_space`) and done: `(hdr, p3)`.
    asked: (bool, bool),
    done: (bool, bool),
    /// The last configuration, which a change of space configures again.
    config: Option<SurfaceConfig>,
    /// When each frame presented reached the display, by its id, until taken; the next id; and
    /// the thread that waits for each present (where the device has `present_wait`).
    presented: Arc<Mutex<Vec<(u64, std::time::Instant)>>>,
    next_id: u64,
    waiter: Option<Waiter>,
    /// Held around each call that touches the swapchain from either thread: Vulkan requires host
    /// access to it to be externally synchronized, `vkWaitForPresentKHR`'s included
    /// (docs/citations.md). The waiter takes it a millisecond at a time, so presenting never
    /// waits on it for longer.
    chain: Arc<Mutex<()>>,
    /// The window, alive while its surface is.
    _window: Arc<dyn Window>,
}

/// What a surface's waiter thread is asked: to wait for a present (by its swapchain and id), or
/// to say it has waited for every one asked before.
enum Wait {
    Present(vk::SwapchainKHR, u64),
    Drained(std::sync::mpsc::Sender<()>),
}

/// The thread that waits for each present to reach the display and notes when.
struct Waiter {
    asks: std::sync::mpsc::Sender<Wait>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Waiter {
    fn new(
        device: &Device,
        presented: Arc<Mutex<Vec<(u64, std::time::Instant)>>>,
        held: Arc<Mutex<()>>,
    ) -> Waiter {
        let fns = ash::khr::present_wait::Device::new(&device.0.instance, &device.0.raw);
        let (asks, waits) = std::sync::mpsc::channel::<Wait>();
        let thread = std::thread::spawn(move || {
            for w in waits {
                match w {
                    Wait::Present(chain, id) => {
                        // At most a second in all, in waits of a millisecond, each under the
                        // swapchain's lock (`Surface::chain`) and none longer: the main thread's
                        // acquisitions and presents take the lock between them.
                        let began = std::time::Instant::now();
                        loop {
                            let r = {
                                let _held = lock(&held);
                                // SAFETY: the swapchain lives until this thread says it's
                                // drained, and its host access is held.
                                unsafe { fns.wait_for_present(chain, id + 1, 1_000_000) }
                            };
                            match r {
                                Ok(()) => {
                                    lock(&presented).push((id, std::time::Instant::now()));
                                    break;
                                }
                                Err(vk::Result::TIMEOUT)
                                    if began.elapsed() < std::time::Duration::from_secs(1) => {}
                                Err(_) => break,
                            }
                        }
                    }
                    Wait::Drained(done) => {
                        let _ = done.send(());
                    }
                }
            }
        });
        Waiter { asks, thread: Some(thread) }
    }

    /// Returns once every wait asked before is done.
    fn drain(&self) {
        let (tx, rx) = std::sync::mpsc::channel();
        if self.asks.send(Wait::Drained(tx)).is_ok() {
            let _ = rx.recv();
        }
    }
}

impl Drop for Waiter {
    fn drop(&mut self) {
        let (tx, _) = std::sync::mpsc::channel();
        // Closing the channel ends the thread.
        self.asks = tx;
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// A frame of a surface to draw into, then [`Surface::present`].
pub struct SurfaceTexture {
    index: u32,
    acquired: vk::Semaphore,
    rendered: vk::Semaphore,
    slot: usize,
    pub texture: Texture,
    pub view: TextureView,
}

impl Device {
    /// A surface of `window` (Wayland, X11).
    pub fn create_surface(&self, window: Arc<dyn Window>) -> Result<Surface> {
        if !self.0.swapchain {
            return Err("this Vulkan has no surface or swapchain extension".into());
        }
        let display = window.display_handle().map_err(|e| Error(format!("the window: {e}")))?;
        let handle = window.window_handle().map_err(|e| Error(format!("the window: {e}")))?;
        // SAFETY: the handles are the window's, alive while the surface is (it keeps the window).
        let surface = unsafe {
            ash_window::create_surface(
                &self.0.entry,
                &self.0.instance,
                display.as_raw(),
                handle.as_raw(),
                None,
            )
        }
        .map_err(|e| Error(format!("Vulkan made no surface: {e}")))?;
        let surface_fns = ash::khr::surface::Instance::new(&self.0.entry, &self.0.instance);
        // SAFETY: a query of the surface.
        let ok = unsafe {
            surface_fns.get_physical_device_surface_support(
                self.0.physical,
                self.0.queue_family,
                surface,
            )
        }
        .unwrap_or(false);
        if !ok {
            // SAFETY: the surface was made above and nothing uses it.
            unsafe { surface_fns.destroy_surface(surface, None) };
            return Err("the GPU's queue can't present to this window".into());
        }
        let swapchain_fns = ash::khr::swapchain::Device::new(&self.0.instance, &self.0.raw);
        Ok(Surface {
            device: self.clone(),
            surface,
            surface_fns,
            swapchain_fns,
            swapchain: vk::SwapchainKHR::null(),
            images: Vec::new(),
            acquired: Vec::new(),
            next: 0,
            rendered: Vec::new(),
            asked: (false, false),
            done: (false, false),
            config: None,
            presented: Arc::new(Mutex::new(Vec::new())),
            next_id: 0,
            waiter: None,
            chain: Arc::new(Mutex::new(())),
            _window: window,
        })
    }

    /// Submits `buffers` for `frame`: after its image is acquired, signalling what its
    /// presentation waits on.
    pub fn submit_frame(
        &self,
        buffers: Vec<CommandBuffer>,
        frame: &SurfaceTexture,
    ) -> SubmissionIndex {
        self.submit_waiting(buffers, Some((frame.acquired, frame.rendered)))
    }
}

impl Surface {
    /// Its frames' format: half floats when it shows them HDR.
    pub fn format(&self) -> TextureFormat {
        if self.done.0 { TextureFormat::Rgba16Float } else { TextureFormat::Rgba8 }
    }

    /// Shows its frames as `hdr` (linear half floats in extended sRGB) or not, in Display P3
    /// (`p3`) or sRGB, where the surface offers the colour space: what it does, `(hdr, p3)`.
    pub fn set_space(&mut self, hdr: bool, p3: bool) -> (bool, bool) {
        self.asked = (hdr, p3);
        if let Some(c) = self.config
            && self.configure(&c).is_err()
        {
            self.asked = (false, false);
            let _ = self.configure(&c);
        }
        self.done
    }

    /// The display's headroom: Vulkan has no query of it, so 1.
    pub fn headroom(&self) -> f32 {
        1.0
    }

    /// Makes its swapchain as `c` says, on the device it was made of.
    pub fn configure(&mut self, c: &SurfaceConfig) -> Result<()> {
        self.config = Some(*c);
        let device = self.device.clone();
        let d = &device.0;
        // SAFETY: queries of the surface, then a swapchain of it (the old one retired once the
        // device is idle), and semaphores.
        unsafe {
            let caps = self
                .surface_fns
                .get_physical_device_surface_capabilities(d.physical, self.surface)
                .map_err(|e| Error(format!("the surface: {e}")))?;
            let formats = self
                .surface_fns
                .get_physical_device_surface_formats(d.physical, self.surface)
                .map_err(|e| Error(format!("the surface: {e}")))?;
            // The space asked for where the surface has it, else SDR in sRGB.
            let pair = |f: vk::Format, s: vk::ColorSpaceKHR| {
                formats.iter().find(|x| x.format == f && x.color_space == s).copied()
            };
            let hdr = self.asked.0.then(|| {
                pair(vk::Format::R16G16B16A16_SFLOAT, vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT)
            });
            let p3 = (self.asked.1 && !self.asked.0).then(|| {
                pair(vk::Format::R8G8B8A8_UNORM, vk::ColorSpaceKHR::DISPLAY_P3_NONLINEAR_EXT)
            });
            let (format, done) = match (hdr.flatten(), p3.flatten()) {
                (Some(f), _) => (f, (true, false)),
                (None, Some(f)) => (f, (false, true)),
                _ => (
                    formats
                        .iter()
                        .find(|f| {
                            f.format == vk::Format::R8G8B8A8_UNORM
                                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                        })
                        .copied()
                        .ok_or("the surface has no R8G8B8A8_UNORM format in sRGB")?,
                    (false, false),
                ),
            };
            self.done = done;
            // The frames' format, as the host names it (Vulkan's is `format.format`).
            let texel = self.format();
            let modes = self
                .surface_fns
                .get_physical_device_surface_present_modes(d.physical, self.surface)
                .map_err(|e| Error(format!("the surface: {e}")))?;
            let want = match c.present {
                PresentMode::Fifo => vk::PresentModeKHR::FIFO,
                PresentMode::FifoRelaxed => vk::PresentModeKHR::FIFO_RELAXED,
                PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
                PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
            };
            // FIFO is every surface's.
            let mode = if modes.contains(&want) { want } else { vk::PresentModeKHR::FIFO };
            let extent = if caps.current_extent.width == u32::MAX {
                vk::Extent2D { width: c.width, height: c.height }
            } else {
                caps.current_extent
            };
            let mut count = c.frames.clamp(2, 3).max(caps.min_image_count);
            if caps.max_image_count > 0 {
                count = count.min(caps.max_image_count);
            }
            let old = self.swapchain;
            let _ = d.raw.device_wait_idle();
            // Every wait for a present of the old swapchain done while it can still be waited
            // on: making the new one retires it (VUID-vkWaitForPresentKHR-swapchain-04997).
            if let Some(w) = &self.waiter {
                w.drain();
            }
            let swapchain = self
                .swapchain_fns
                .create_swapchain(
                    &vk::SwapchainCreateInfoKHR::default()
                        .surface(self.surface)
                        .min_image_count(count)
                        .image_format(format.format)
                        .image_color_space(format.color_space)
                        .image_extent(extent)
                        .image_array_layers(1)
                        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .pre_transform(caps.current_transform)
                        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                        .present_mode(mode)
                        .clipped(true)
                        .old_swapchain(old),
                    None,
                )
                .map_err(|e| Error(format!("Vulkan made no swapchain: {e}")))?;
            self.release();
            if old != vk::SwapchainKHR::null() {
                self.swapchain_fns.destroy_swapchain(old, None);
            }
            self.swapchain = swapchain;
            let images = self
                .swapchain_fns
                .get_swapchain_images(swapchain)
                .map_err(|e| Error(format!("the swapchain: {e}")))?;
            for image in images {
                let raw = Arc::new(RawImage {
                    device: device.shared(),
                    image,
                    memory: Allocation::NONE,
                    format: texel,
                    aspect: vk::ImageAspectFlags::COLOR,
                    levels: 1,
                    layers: 1,
                    home: vk::ImageLayout::PRESENT_SRC_KHR,
                    swapchain: true,
                    usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                });
                let range = raw.part(0, 0, 1);
                let view = d
                    .raw
                    .create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(format.format)
                            .subresource_range(range),
                        None,
                    )
                    .map_err(|e| Error(format!("Vulkan made no view: {e}")))?;
                self.images.push(Arc::new(RawView {
                    image: raw,
                    view,
                    range,
                    width: extent.width,
                    height: extent.height,
                    index: NO_INDEX,
                }));
                self.rendered.push(
                    d.raw
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .map_err(|e| Error(format!("Vulkan made no semaphore: {e}")))?,
                );
            }
            for _ in 0..self.images.len() + 1 {
                let sem = d
                    .raw
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .map_err(|e| Error(format!("Vulkan made no semaphore: {e}")))?;
                self.acquired.push((sem, 0));
            }
        }
        Ok(())
    }

    /// The swapchain's images, views and semaphores let go (the device idle).
    fn release(&mut self) {
        let d = &self.device.0.raw;
        // SAFETY: the device is idle; nothing uses them.
        unsafe {
            let _ = d.device_wait_idle();
            for (s, _) in self.acquired.drain(..) {
                d.destroy_semaphore(s, None);
            }
            for s in self.rendered.drain(..) {
                d.destroy_semaphore(s, None);
            }
        }
        self.images.clear();
        self.next = 0;
    }

    /// The next frame to draw into (waits for an image the display is done with). An error
    /// when the window changed and the surface wants configuring again.
    pub fn acquire(&mut self) -> Result<SurfaceTexture> {
        let slot = self.next % self.acquired.len();
        self.next += 1;
        let (sem, used) = self.acquired[slot];
        // The semaphore's last wait must be done before it's signalled again.
        self.device.wait(used);
        let (index, _) = {
            let _held = lock(&self.chain);
            // SAFETY: an acquisition signalling an unsignalled semaphore, the swapchain's host
            // access held.
            unsafe {
                self.swapchain_fns.acquire_next_image(
                    self.swapchain,
                    u64::MAX,
                    sem,
                    vk::Fence::null(),
                )
            }
        }
        .map_err(|e| Error(format!("the swapchain: {e}")))?;
        let view = self.images[index as usize].clone();
        let texture = Texture {
            raw: view.image.clone(),
            format: self.format(),
            width: view.width,
            height: view.height,
            depth_or_layers: 1,
            levels: 1,
            dimension: TextureDimension::D2,
        };
        Ok(SurfaceTexture {
            index,
            acquired: sem,
            rendered: self.rendered[index as usize],
            slot,
            texture,
            view: TextureView { raw: view },
        })
    }

    /// Shows `frame` once the submission for it ([`Device::submit_frame`]) is done: its id,
    /// which [`Surface::take_presented`] gives with the time it reached the display (where the
    /// device can say: `present_wait`).
    pub fn present(&mut self, frame: SurfaceTexture) -> Result<u64> {
        self.acquired[frame.slot].1 = self.device.submitted();
        let id = self.next_id;
        self.next_id += 1;
        let waits = [frame.rendered];
        let chains = [self.swapchain];
        let indices = [frame.index];
        // Present ids count from 1, and only up.
        let ids = [id + 1];
        let mut present_id = vk::PresentIdKHR::default().present_ids(&ids);
        let mut info = vk::PresentInfoKHR::default()
            .wait_semaphores(&waits)
            .swapchains(&chains)
            .image_indices(&indices);
        if self.device.0.present_wait {
            info = info.push_next(&mut present_id);
        }
        let presented = {
            let _held = lock(&self.chain);
            // SAFETY: the image was acquired and its rendering submitted, signalling `rendered`;
            // the swapchain's host access held.
            unsafe { self.swapchain_fns.queue_present(self.device.0.queue, &info) }
        };
        match presented {
            Ok(_) => {}
            // The window changed under it (a resize between acquiring and presenting): the
            // frame isn't shown, and the swapchain is made again as configured, as an
            // acquisition's out-of-date error has the host do. The present's wait on
            // `rendered` still runs; configuring waits for the device to be idle.
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                drop(frame);
                if let Some(c) = self.config {
                    self.configure(&c)?;
                }
                return Ok(id);
            }
            Err(e) => return Err(Error(format!("the swapchain: {e}"))),
        }
        if self.device.0.present_wait {
            let presented = self.presented.clone();
            let held = self.chain.clone();
            let w = self.waiter.get_or_insert_with(|| Waiter::new(&self.device, presented, held));
            let _ = w.asks.send(Wait::Present(self.swapchain, id));
        }
        Ok(id)
    }

    /// The frames shown since the last take: each id and when it reached the display.
    pub fn take_presented(&mut self) -> Vec<(u64, std::time::Instant)> {
        std::mem::take(&mut *lock(&self.presented))
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        if let Some(w) = self.waiter.take() {
            w.drain();
        }
        self.release();
        // SAFETY: the device is idle; the swapchain and then its surface go.
        unsafe {
            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_fns.destroy_swapchain(self.swapchain, None);
            }
            self.surface_fns.destroy_surface(self.surface, None);
        }
    }
}
