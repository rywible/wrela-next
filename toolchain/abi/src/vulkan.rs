//! How a pipeline's bindings reach a Vulkan shader: the one rule the compiler writes its SPIR-V
//! by and the host binds by.
//!
//! The host makes each pipeline's bind group layout from [`crate::metal::layout`] (bind group 0,
//! its only one, each binding in ascending order) and hands wgpu each shader as finished SPIR-V
//! (`create_shader_module_passthrough`). wgpu 30's Vulkan backend doesn't give a binding its own
//! number as a Vulkan binding: it numbers a layout's descriptors 0, 1, 2, .. in the order of its
//! entries (`wgpu-hal` 30, `vulkan/device.rs`, `create_bind_group_layout`), which wgpu-core has
//! sorted by binding (`wgpu-core` 30, `device/bgl.rs`, `EntryMap::from_entries`), and rewrites a
//! shader it translates to match (`create_pipeline_layout`'s `binding_map`). A passthrough shader
//! isn't rewritten, so its SPIR-V names each binding by that number itself: [`descriptor`].

use crate::manifest::Pipeline;

/// Set 0's binding in Vulkan of the pipeline's binding `binding`: its place among the bindings
/// of the pipeline's layout that aren't in the device's table ([`crate::table`]), in ascending
/// order. `None` if the layout has no such binding, or the table holds it.
pub fn descriptor(p: &Pipeline, binding: u32) -> Option<u32> {
    let at =
        crate::metal::layout(p).iter().filter(|e| !e.table).position(|e| e.binding == binding)?;
    u32::try_from(at).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A layout's bindings are numbered in order, with the gaps closed: the uniform block, the
    /// buffers, and a debug build's flag after them.
    #[test]
    fn descriptors_are_places_in_the_layout() {
        let mut p = crate::vectors::sample_manifest().pipelines.remove(0);
        p.debug_flag = Some(7);
        assert_eq!(p.uniform.as_ref().map(|u| u.binding), Some(0));
        assert_eq!(p.bindings.iter().map(|b| b.binding).collect::<Vec<_>>(), [1]);
        assert_eq!(descriptor(&p, 0), Some(0));
        assert_eq!(descriptor(&p, 1), Some(1));
        assert_eq!(descriptor(&p, 7), Some(2));
        assert_eq!(descriptor(&p, 3), None);
    }
}
