//! Building: the back ends, and the build's files (vision.md, "What ships and how it runs"):
//! `game.wasm`, one Metal shading language (MSL) module per pipeline, and `manifest.json`.

use wrela_diag::{Diagnostic, SourceMap};
use wrela_lower::{Lowered, PipelineKey, PipelineKind};

#[derive(Debug)]
pub struct BuildOutput {
    /// Each file of the build, by its path relative to the output directory, in a fixed order.
    pub files: Vec<(String, Vec<u8>)>,
    pub diagnostics: Vec<Diagnostic>,
    /// Where each WASM offset is in the source: for a frame test's failure (`frames`).
    pub lines: Vec<(u32, Option<wrela_diag::Span>)>,
}

/// A pipeline's MSL at most, in bytes: what Metal's compiler takes while the scene loads. It was
/// the browser's budget for WGSL (256 KiB, spike 13), and Metal's compiler crashed on kernels
/// near it (the atoms spike's cook, 160-200 KiB of WGSL). A shader's MSL is larger than its WGSL:
/// the fighters' placing kernel, which fitted as WGSL and compiles and runs, is 262 KiB of MSL,
/// so the budget as MSL is 288 KiB. Not yet measured against Metal's compile time.
pub const MAX_SHADER_BYTES: usize = 288 * 1024;

/// A pipeline's weight at most (§18.11): its entry points' statements with every call written
/// out, what predicts how long a browser's shader compiler takes over it (the opening's far pass
/// weighed 270,830 and took three quarters of its cold compile, Nov 2026).
pub const MAX_WEIGHT: u64 = 300_000;

/// Maximum per-invocation local frame / scratch memory (`kept` bytes) permitted in a shipped build
/// (compile-time register-spill guard matching the GPU host safety limit).
pub const MAX_KEPT_BYTES: u32 = crate::wcet::MAX_KEPT_BYTES;

/// How a build writes its pipelines' shaders: for which machine, and with which run-time checks.
#[derive(Clone, Copy, Debug)]
pub struct Shaders {
    pub target: wrela_abi::Target,
    pub checks: wrela_shader::Checks,
}

impl Shaders {
    /// For the machine the compiler runs on, checked.
    pub const HERE: Shaders =
        Shaders { target: wrela_abi::Target::HERE, checks: wrela_shader::Checks::CHECKED };
}

/// Runs the back ends over a lowered program. `sources` names the locations a trap reports.
/// With `budgets`, a shipped build's, it's E0707 to have a pipeline whose MSL is over
/// `MAX_SHADER_BYTES`. `conditional[i]` says pipeline `i` is reached only under runtime
/// conditions (every site of it is): the manifest puts it in tier 1, which a host builds second
/// (§18.9); a pipeline past the slice's end is tier 0.
#[allow(clippy::too_many_arguments)]
pub fn emit(
    l: &Lowered,
    sources: &SourceMap,
    simd: bool,
    budgets: bool,
    conditional: &[bool],
    relaxed: bool,
    set: &[(String, String)],
    shaders: Shaders,
) -> BuildOutput {
    let options = wrela_wasm::Options { simd, relaxed, ..Default::default() };
    emit_in(l, sources, options, budgets, Some(shaders), conditional, set)
}

/// A server's or a tool's build's files (language.md §15.3): its module, at [`SERVER_WASM`] or
/// [`TOOL_WASM`]. It has no pipelines. A server's
/// memory is its own (not shared: it runs on one thread, and its host caps the memory); a
/// tool's is imported and shared, as a client's is, so its host's helpers run `std::par`'s
/// work on it, and may grow to 4 GiB.
pub fn emit_headless(
    l: &Lowered,
    sources: &SourceMap,
    simd: bool,
    role: wrela_sema::defs::Role,
) -> BuildOutput {
    let tool = role == wrela_sema::defs::Role::Tool;
    let module = if tool { TOOL_WASM } else { SERVER_WASM };
    let max_pages =
        if tool { wrela_abi::memory::TOOL_MAX_PAGES } else { wrela_abi::memory::MAX_PAGES };
    let options = wrela_wasm::Options { simd, shared_memory: tool, max_pages, relaxed: false };
    let mut out = emit_in(l, sources, options, false, None, &[], &[]);
    out.files.retain(|(path, _)| path == "game.wasm");
    for (path, _) in &mut out.files {
        *path = module.to_string();
    }
    out
}

/// Where a relaxed-SIMD build's second module is, beside `game.wasm` (`build_relaxed`).
pub const RELAXED_WASM: &str = "game.relaxed.wasm";

/// Where a server build's module is, in the build's directory.
pub const SERVER_WASM: &str = "server/game.wasm";

/// Where a tool build's module is, in the build's directory.
pub const TOOL_WASM: &str = "tool/tool.wasm";

/// Puts the program's pair id ([`wrela_abi::pair_id`]) into each of its builds' modules, as
/// the custom section [`wrela_abi::PAIR_SECTION`]: `client` and `server` are their bytes, and
/// `relaxed` the client's relaxed-SIMD module's, which is the same program.
pub fn pair(
    client: Option<&mut Vec<u8>>,
    server: Option<&mut Vec<u8>>,
    relaxed: Option<&mut Vec<u8>>,
) {
    let id = wrela_abi::pair_id(
        client.as_deref().map(|w| w.as_slice()),
        server.as_deref().map(|w| w.as_slice()),
    );
    for wasm in [client, server, relaxed].into_iter().flatten() {
        let bytes = std::mem::take(wasm);
        *wasm = wrela_wasm::with_custom_section(bytes, wrela_abi::PAIR_SECTION, &id.to_le_bytes());
    }
}

/// A CPU frame test's files: its WASM and the manifest the host uses to check its commands.
/// There are no shader or browser files; `CpuBuild`'s explicit CPU-only loaders read it.
pub(crate) fn emit_for_cpu(l: &Lowered, sources: &SourceMap, simd: bool) -> BuildOutput {
    emit_in(l, sources, wrela_wasm::Options { simd, ..Default::default() }, false, None, &[], &[])
}

/// Where each of a module's code offsets came from (`wrela_wasm::Emitted::lines`).
type CodeLines = Vec<(u32, Option<wrela_diag::Span>)>;

/// The program's module with `options`: its bytes, its trap locations in a section of their
/// own; the locations; and its interface.
fn module(
    l: &Lowered,
    sources: &SourceMap,
    options: wrela_wasm::Options,
) -> Result<(Vec<u8>, CodeLines, wrela_abi::manifest::WasmAbi), String> {
    let out = wrela_wasm::emit_with(&l.cpu, options)
        .map_err(|e| format!("the WASM back end failed: {e}"))?;
    let section = code_lines_section(&out.lines, sources);
    let wasm = wrela_wasm::with_custom_section(out.wasm, wrela_abi::lines::SECTION, &section);
    let abi = wrela_wasm::interface(&wasm)
        .map_err(|e| format!("the WASM back end's module doesn't read: {e}"))?;
    if options.relaxed {
        wrela_wasm::check_relaxed_only_in(&wasm, wrela_abi::EXPORT_FRAME)?;
    }
    Ok((wasm, out.lines, abi))
}

fn emit_in(
    l: &Lowered,
    sources: &SourceMap,
    options: wrela_wasm::Options,
    budgets: bool,
    shaders: Option<Shaders>,
    conditional: &[bool],
    set: &[(String, String)],
) -> BuildOutput {
    let mut diagnostics = Vec::new();
    let mut costly_said = std::collections::HashSet::new();
    let mut copies_said = std::collections::HashSet::new();
    let mut files = Vec::new();
    let mut code_lines = Vec::new();
    let mut wasm_abi = None;
    let mut relaxed = None;
    if options.relaxed {
        // The presentation code with relaxed SIMD, beside the standard module.
        match module(l, sources, options) {
            Ok((wasm, _, abi)) => relaxed = Some((wasm, abi)),
            Err(e) => diagnostics.push(Diagnostic::internal(format!("its relaxed module: {e}"))),
        }
    }
    match module(l, sources, wrela_wasm::Options { relaxed: false, ..options }) {
        Ok((wasm, lines, abi)) => {
            code_lines = lines;
            wasm_abi = Some(abi);
            files.push(("game.wasm".to_string(), wasm));
        }
        Err(e) => diagnostics.push(Diagnostic::internal(e)),
    }
    let mut manifest = wrela_abi::Manifest::new("game.wasm");
    let target = shaders.map_or(wrela_abi::Target::HERE, |s| s.target);
    manifest.target = target;
    manifest.wasm_abi = wasm_abi;
    manifest.set = set.iter().cloned().collect();
    if let Some((wasm, abi)) = relaxed {
        files.push((RELAXED_WASM.to_string(), wasm));
        manifest.wasm_relaxed = Some(RELAXED_WASM.to_string());
        manifest.wasm_relaxed_abi = Some(abi);
    }
    for (i, p) in l.pipelines.iter().enumerate() {
        let shader = format!("pipeline_{i}.{}", wrela_shader::extension(target));
        // The CPU host checks stages and bindings without looking up shader entry points: its
        // build names them as the IR does, and a full build as the MSL writer does, which may
        // rename them.
        let names: Vec<String> = p.module.entry_points.iter().map(|e| e.name.clone()).collect();
        let stage = match (&p.kind, &p.key, names.as_slice()) {
            (PipelineKind::Compute { workgroup_size }, _, [entry]) => {
                wrela_abi::manifest::Stage::Compute {
                    entry: entry.clone(),
                    workgroup_size: *workgroup_size,
                }
            }
            (
                PipelineKind::Render { output },
                PipelineKey::Render { state, .. },
                [vertex, fragment],
            ) => wrela_abi::manifest::Stage::Render {
                vertex_entry: vertex.clone(),
                fragment_entry: fragment.clone(),
                blends: wrela_lower::blends_of(state, output),
                cull: state.cull,
                depth_bias: wrela_abi::manifest::DepthBias {
                    constant: state.bias_constant,
                    slope_scale: f32::from_bits(state.bias_slope),
                    clamp: f32::from_bits(state.bias_clamp),
                },
                depth: state.depth,
                writes_depth: output.writes_depth(),
                outputs: output
                    .colours
                    .iter()
                    .map(|(c, _)| {
                        if c.uint {
                            wrela_abi::manifest::Output::Uint
                        } else {
                            wrela_abi::manifest::Output::Float
                        }
                    })
                    .collect(),
                targets: p.targets.clone(),
            },
            _ => {
                diagnostics.push(Diagnostic::internal(format!(
                    "the shader for `{}` has the wrong entry points",
                    p.name
                )));
                continue;
            }
        };
        let mut pipeline = wrela_abi::manifest::Pipeline {
            name: p.name.clone(),
            shader,
            stage,
            uniform: p.uniform.as_ref().map(|u| wrela_abi::manifest::UniformBlock {
                binding: u.binding,
                size: u.size,
                space: if u.storage {
                    wrela_abi::manifest::UniformSpace::Storage
                } else {
                    wrela_abi::manifest::UniformSpace::Uniform
                },
            }),
            bindings: p.bindings.clone(),
            debug_flag: p.debug_flag,
            requires: p.module.requires(),
            conditional: conditional.get(i).copied().unwrap_or(false),
            standin: p.standin,
            sizes: None,
            kept: p.module.kept_bytes(),
            work: p.work.clone(),
            unbounded: p
                .unbounded
                .iter()
                .map(|s| s.map(|s| sources.file(s.file).location(s.start)))
                .collect(),
        };
        let shader = if let Some(s) = shaders {
            match wrela_shader::emit_shader(&p.module, &mut pipeline, s.target, s.checks) {
                Ok(m) => Some(m),
                Err(e) => {
                    diagnostics.push(Diagnostic::internal(format!(
                        "the {} back end failed for `{}`: {e}",
                        target.name(),
                        p.name
                    )));
                    continue;
                }
            }
        } else {
            None
        };
        if let Some(written) = &shader {
            match (&mut pipeline.stage, written.entry_points.as_slice()) {
                (wrela_abi::manifest::Stage::Compute { entry, .. }, [e]) => *entry = e.clone(),
                (
                    wrela_abi::manifest::Stage::Render { vertex_entry, fragment_entry, .. },
                    [v, f],
                ) => {
                    *vertex_entry = v.clone();
                    *fragment_entry = f.clone();
                }
                _ => {
                    diagnostics.push(Diagnostic::internal(format!(
                        "the shader for `{}` has the wrong entry points",
                        p.name
                    )));
                    continue;
                }
            }
        }
        if budgets
            && target == wrela_abi::Target::Metal
            && let Some(written) = &shader
            && written.bytes.len() > MAX_SHADER_BYTES
            && let Some(&at) = p.sites.first()
        {
            diagnostics.push(over_budget(
                at,
                format!(
                    "`{}`'s MSL is {} KiB, over the {} KiB a pipeline's shader may be",
                    p.name,
                    written.bytes.len().div_ceil(1024),
                    MAX_SHADER_BYTES / 1024
                ),
                "Metal compiles each pipeline's shader while the scene loads",
                "move work that its own pass can do there, or keep a big function a call (`ir::opt`)",
            ));
        }
        if budgets
            && pipeline.kept > MAX_KEPT_BYTES
            && let Some(&at) = p.sites.first()
        {
            diagnostics.push(over_budget(
                at,
                format!(
                    "`{}` keeps {} B of local state per invocation, over the {} B compile-time spill guard",
                    p.name, pipeline.kept, MAX_KEPT_BYTES
                ),
                "a kernel's local arrays and stack live in per-thread scratch memory; exceeding 16 KiB causes catastrophic register spills and GPU stalls",
                "shrink local arrays, move persistent state to a storage buffer, or split the pass; `wrela pipelines` reports each pipeline's kept bytes",
            ));
        }
        if budgets
            && p.weight > MAX_WEIGHT
            && let Some(&at) = p.sites.first()
        {
            diagnostics.push(over_budget(
                at,
                format!(
                    "`{}` weighs {}, over the {} a pipeline may weigh",
                    p.name, p.weight, MAX_WEIGHT
                ),
                "a pipeline's weight is its statements with every call written out, as a shader compiler copies a function into each place it's called; its compile time grows faster than its weight (§18.11)",
                "call a big function from fewer places (once in a loop, not once a step), or move work its own pass can do there; `wrela pipelines` shows each pipeline's weight",
            ));
        }
        if budgets {
            for w in &p.same {
                let Some(at) = w.at else { continue };
                diagnostics.push(same_work(at, w));
            }
            for c in &p.costly {
                // Once a loop, however many pipelines run it.
                if costly_said.insert(c.at) {
                    diagnostics.push(costly_loop(c));
                }
            }
            for c in &p.copies {
                // Once a place, however many pipelines copy there.
                if let Some(at) = c.at
                    && copies_said.insert(at)
                {
                    diagnostics.push(copied(at, c));
                }
            }
        }
        if let Some(written) = shader {
            files.push((pipeline.shader.clone(), written.bytes));
        }
        manifest.pipelines.push(pipeline);
    }
    if let Err(e) = manifest.validate() {
        diagnostics.push(Diagnostic::internal(e.to_string()));
    }
    files.push(("manifest.json".to_string(), manifest.to_json().into_bytes()));
    BuildOutput { files, diagnostics, lines: code_lines }
}

/// Where each part of the code came from, as `file:line:column`, for trap messages: the lines
/// section of a module whose offsets are `lines`.
fn code_lines_section(lines: &[(u32, Option<wrela_diag::Span>)], sources: &SourceMap) -> Vec<u8> {
    let entries: Vec<(u32, Option<String>)> = lines
        .iter()
        .map(|&(offset, span)| (offset, span.map(|s| sources.file(s.file).location(s.start))))
        .collect();
    wrela_abi::lines::Lines::new(entries).encode()
}

/// States `game.wasm`'s interface in `manifest.json` again, after something rewrote the module
/// (`wasm-opt` may drop an import nothing calls).
pub fn restate_abi(files: &mut [(String, Vec<u8>)]) -> Result<(), String> {
    let wasm = files.iter().find(|(f, _)| f == "game.wasm").ok_or("the build has no game.wasm")?;
    let abi = wrela_wasm::interface(&wasm.1)?;
    let relaxed = files
        .iter()
        .find(|(f, _)| f == RELAXED_WASM)
        .map(|(_, w)| wrela_wasm::interface(w))
        .transpose()?;
    let Some(manifest) = files.iter_mut().find(|(f, _)| f == "manifest.json") else {
        return Ok(());
    };
    let text = std::str::from_utf8(&manifest.1).map_err(|_| "manifest.json isn't UTF-8")?;
    let mut m = wrela_abi::Manifest::parse(text).map_err(|e| e.to_string())?;
    m.wasm_abi = Some(abi);
    if relaxed.is_some() {
        m.wasm_relaxed_abi = relaxed;
    }
    manifest.1 = m.to_json().into_bytes();
    Ok(())
}

/// E0713: a shipped build's GPU code copying a large struct or array of read-only data into
/// each invocation (`wrela_ir::opt::read_only_copies`), at the copy.
fn copied(at: wrela_diag::Span, c: &wrela_ir::opt::Copied) -> Diagnostic {
    let from = if c.uniform { "the uniform block" } else { "a read-only buffer" };
    Diagnostic::new(
        wrela_diag::codes::E0713,
        at,
        format!(
            "every invocation copies {} ({} bytes) from {from} into its own memory here",
            c.what, c.bytes
        ),
    )
    .with_note("the compiler reads a struct passed by value where it lies, unless the code needs a copy: a part of it changed and the whole then read again (a change at an index only the invocation knows, say). GPU compilers keep such a copy in slow memory; a 964-byte one made a pass 11 to 20 times slower (§18.13)")
    .with_help("change nothing in it: pass what would change as an argument of its own (`form.sample_of(p, mask)` rather than `form.only(mask).sample(p)`)")
}

/// E0712: a shipped build's GPU code computing, in every invocation, `work` of a value the same
/// for all of them (§18.13), at the value.
fn same_work(at: wrela_diag::Span, w: &wrela_ir::same::SameWork) -> Diagnostic {
    let work = match w.work {
        Some(w) => format!("{w} work"),
        None => "work with no bound".to_string(),
    };
    let mut path = format!("`{}`", w.entry);
    for f in &w.path {
        path.push_str(&format!(" -> `{f}`"));
    }
    Diagnostic::new(
        wrela_diag::codes::E0712,
        at,
        format!("every invocation computes this again, the same for all of them: {work}"),
    )
    .with_note(format!("in the invocations of entry point {path}"))
    .with_note("it's computed only from what the command gives every invocation alike: its uniform block, the pipeline's constants, a buffer or a texture read at a same place (§18.13)")
    .with_help("compute it once, on the CPU or in a kernel of one invocation (`@compute(1)`), and pass the result in")
}

/// W0014: a shipped build's GPU loop that every invocation runs, that the compiler couldn't move
/// to an earlier rate and that costs (`wrela_ir::rates`, language.md §18.13), at the loop.
fn costly_loop(c: &wrela_ir::rates::Costly) -> Diagnostic {
    use wrela_ir::rates::Why;
    // The count's passes include the last, which only tests and leaves: a `for` over 0..4 makes
    // 5. Said as the body's.
    let passes = match c.passes {
        Some(n) => format!("{} passes", n.saturating_sub(1)),
        None => "passes with no bound the count is sure of".to_string(),
    };
    let work = match c.work {
        Some(w) => format!("{w} work"),
        None => "work with no bound".to_string(),
    };
    let each = c.each;
    let (message, note, help) = match c.why {
        Why::Constants => (
            format!(
                "every {each} runs this loop of constants again, to the same result: {passes}, {work}"
            ),
            format!(
                "every value it computes is a constant, but running it at the build took more than the {} steps the build runs a loop for (§18.13)",
                wrela_ir::FOLD_FUEL
            ),
            "compute it as a `const` (`const SUM: f32 = sum()`), with `@fuel(..)` on the `const` if it needs more than the build's default; or compute less".to_string(),
        ),
        Why::SameCount => (
            format!(
                "every {each} runs this loop to a count the command gives, the same for all of them: up to {passes}, {work}"
            ),
            "its count is the same for the whole command (read from its uniform block or a buffer), so every invocation tests it and carries the loop's code".to_string(),
            "take the count as a `const` parameter of the entry point (`lights: const u32`), so each count a command has is a pipeline of its own and one with none has no loop (the engine's listers draw so); or hold the function to a `@budget(work: n)` if the loop is meant".to_string(),
        ),
        Why::Heavy => {
            let callee = c.callee.as_deref().unwrap_or("this function");
            let times = if c.times == 1 { "once".to_string() } else { format!("{} times", c.times) };
            // (Counted before flattening, a call may weigh more than its whole invocation does after,
            // where the build did some of it.)
            let of = match (c.of, c.work) {
                (Some(all), Some(w)) if w <= all => format!(" of its {all}"),
                _ => String::new(),
            };
            (
                format!("every {each} calls `{callee}` here {times}: {work}{of}"),
                "what reads the invocation's own place runs at every invocation: the compiler moves to the build only what reads constants alone (§18.13). Work that depends only on what never changes (a look on a fixed solid's own points) is the engine's to bake".to_string(),
                format!("compute it at a lower rate (a table, a cook, a constant), or compute less; if it's meant, hold `{callee}`, or a function that calls it, to a `@budget(work: n)`"),
            )
        }
    };
    Diagnostic::new(wrela_diag::codes::W0014, c.at, message).with_note(note).with_help(help)
}

/// E0707: a shipped build over one of its budgets, at the command that records what's over.
fn over_budget(at: wrela_diag::Span, message: String, note: &str, help: &str) -> Diagnostic {
    Diagnostic::new(wrela_diag::codes::E0707, at, message).with_note(note).with_help(help)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CPU_FRAMES: &str = r#"use std::gpu::{
    ClipPosition, GlobalId, GpuBuffer, Slots, VertexIndex, begin_screen_pass, buffer,
    dispatch, draw,
}

@compute(64)
fn fill(out: mut Slots<f32>, id: GlobalId, value: f32) {
    let values = [value, value + 1.0]
    out[id] = values[id.x % 2]
}

@vertex
fn cover(v: VertexIndex) -> ClipPosition {
    let x = if v.index == 1 { 3.0 } else { -1.0 }
    let y = if v.index == 2 { 3.0 } else { -1.0 }
    ClipPosition { position: vec4(x, y, 0.0, 1.0) }
}

@fragment
fn colour() -> vec4 {
    vec4(1.0)
}

pub fn frame(time: f32, width: u32, height: u32) {
    var out: GpuBuffer<f32> = buffer(64)
    dispatch(fill.bind(out: mut out, value: time), groups: 1)
    let screen = begin_screen_pass(clear: vec4(0.0))
    draw(screen, cover, colour, vertices: 3)
    screen.present()
}

@test(frames: 2)
fn records_commands() {}
"#;

    fn package(name: &str, source: &str) -> std::path::PathBuf {
        let root =
            std::env::temp_dir().join(format!("wrela-cpu-build-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("package directory");
        std::fs::write(root.join("main.wrela"), source).expect("source");
        root
    }

    fn file<'a>(out: &'a BuildOutput, name: &str) -> &'a [u8] {
        &out.files.iter().find(|(path, _)| path == name).expect("built file").1
    }

    #[test]
    fn cpu_frame_build_keeps_wasm_and_pipeline_interfaces() {
        let root = package("interfaces", CPU_FRAMES);
        crate::with_checked(&root, |checked, sources| {
            let roots = wrela_lower::Roots::of(checked);
            for debug in [false, true] {
                let data = wrela_lower::BuildData { debug, ..Default::default() };
                let (full, errors) = wrela_lower::lower(checked, &roots, &data, true);
                assert!(!wrela_diag::has_errors(&errors), "{errors:?}");
                let (cpu, errors) = wrela_lower::lower_for_cpu(checked, &roots, &data);
                assert!(!wrela_diag::has_errors(&errors), "{errors:?}");
                assert!(cpu.pipelines.iter().all(|p| p.debug_flag.is_none()));
                if debug {
                    assert!(full.pipelines.iter().any(|p| p.debug_flag.is_some()));
                }
                let full = emit(&full, sources, true, false, &[], false, &[], Shaders::HERE);
                let cpu = emit_for_cpu(&cpu, sources, true);
                assert!(!wrela_diag::has_errors(&full.diagnostics), "{:?}", full.diagnostics);
                assert!(!wrela_diag::has_errors(&cpu.diagnostics), "{:?}", cpu.diagnostics);
                assert_eq!(file(&full, "game.wasm"), file(&cpu, "game.wasm"));
                assert_eq!(cpu.files.len(), 2, "only WASM and the command manifest");
                let interfaces = |out: &BuildOutput| {
                    let mut value: serde_json::Value =
                        serde_json::from_slice(file(out, "manifest.json")).expect("manifest JSON");
                    let pipelines = value["pipelines"].as_array_mut().expect("pipelines");
                    assert_eq!(pipelines.len(), 2, "compute and render commands are checked");
                    for pipeline in pipelines {
                        let pipeline = pipeline.as_object_mut().expect("pipeline");
                        // Naga may rename the entry points. The CPU host neither resolves them
                        // nor runs a shader's debug checks; its own debug checks remain in WASM.
                        // Nor does it read buffers' sizes, which a shader is given, or what a
                        // GPU's invocation keeps (the GPU's build is optimized more).
                        let gpu_only =
                            ["entry", "vertex_entry", "fragment_entry", "debug_flag", "sizes", "kept"];
                        for key in gpu_only {
                            pipeline.remove(key);
                        }
                    }
                    value
                };
                assert_eq!(interfaces(&full), interfaces(&cpu));
                // A draw on the screen draws into the screen's target, which says it's the
                // screen: a host may draw the screen in its own format (the browser's canvas).
                let manifest: serde_json::Value =
                    serde_json::from_slice(file(&full, "manifest.json")).expect("manifest JSON");
                let pipelines = manifest["pipelines"].as_array().expect("pipelines");
                let render = pipelines.iter().find(|p| p["kind"] == "render").expect("render");
                assert_eq!(
                    render["targets"],
                    serde_json::json!([{ "colors": ["rgba8unorm"], "depth": false, "screen": true }])
                );
            }
        })
        .expect("checked package");
        let ran = crate::test(&root, None);
        assert!(ran.passed(), "{:?}, {:?}", ran.diagnostics, ran.results);
        assert_eq!(ran.results.len(), 1);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn cpu_frame_build_still_checks_gpu_control_flow() {
        let mut source =
            include_str!("../../tests/conformance/gpu_uniformity_reject.wrela").to_string();
        source.push_str("\n@test(frames: 1)\nfn checks_shaders() {}\n");
        let root = package("uniformity", &source);
        let out = crate::test(&root, None);
        assert!(out.results.is_empty(), "invalid shader prevents the frame test");
        assert!(
            out.diagnostics.iter().any(|d| d.code == wrela_diag::codes::E0608),
            "{:?}",
            out.diagnostics
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn cpu_frame_build_keeps_files_a_program_fetches() {
        let root = package(
            "fetch",
            r#"use std::io::{Pending, fetch}

pub struct State {
    page: Pending<u8>,
    length: u32,
}

pub fn init() -> State {
    State { page: fetch("manifest.json"), length: 0 }
}

pub fn frame(state: mut State, time: f32, width: u32, height: u32) {
    if let Some(Ok(bytes)) = state.page.poll() {
        state.length = bytes.len()
    }
}

@test(frames: 2)
fn fetches_the_builds_manifest(state: State) {
    assert(state.length > 0)
}
"#,
        );
        let out = crate::test(&root, None);
        assert!(out.passed(), "{:?}, {:?}", out.diagnostics, out.results);
        assert_eq!(out.results.len(), 1);
        let _ = std::fs::remove_dir_all(root);
    }
}
