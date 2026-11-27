//! Persistent content-addressed cache of optimized GPU pipelines and emitted shaders.
//!
//! Each pipeline's pre-optimization IR module, target, shader check flags, exact/hoist flags,
//! interface bindings, compiler binary digest, and (for `@strict` pipelines) `lib/std` source
//! digest uniquely determine its post-optimization analysis (`work`, `weight`, `same`, `costly`,
//! `copies`, `hoisted`, `uncounted_most`) and its emitted Metal/SPIR-V shader.
//!
//! Files live in `$WRELA_PIPELINE_CACHE/<version>` (or `$HOME/.wrela/pipelines/<version>`),
//! written atomically via `<key>.<pid>-<nanos>-<seq>.tmp` and renamed to `<key>.wpv`. Set
//! `WRELA_PIPELINE_CACHE=off` to disable. Entries unused for 7 days are swept at most once a day.

use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use wrela_abi::manifest::{
    BindingKind, BindingStage, BufferSizes, Cull, DepthBias, DepthState, Pipeline, ResourceBinding,
    Stage, UniformBlock, UniformSpace,
};
use wrela_abi::stream::TextureFormat;
use wrela_diag::{FileId, SourceMap, Span};
use wrela_ir as ir;
use wrela_lower::{
    CachedPipeline, PipelineCache, PipelineCacheKeyInput, PipelineKind, PipelineOut,
    PreEmittedShader, UniformOut,
};

const MAGIC: &[u8; 8] = b"wrelap\x00\x01";
const MAGIC_WORK: &[u8; 8] = b"wrelaw\x00\x01";
const VERSION: &str = env!("CARGO_PKG_VERSION");
const KEY_BYTES: usize = 64;
const CHECKSUM_BYTES: usize = 32;
const KEEP_FOR: std::time::Duration = std::time::Duration::from_secs(7 * 24 * 3600);
const SWEPT: &str = ".swept";

static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Persistent pipeline cache for one compilation.
pub struct DiskPipelineCache {
    dir: PathBuf,
    exe_hash: [u8; 32],
    std_hash: [u8; 32],
    target: wrela_abi::Target,
    checks: wrela_shader::Checks,
}

impl DiskPipelineCache {
    /// Opens the cache for this compile, or returns `None` when disabled (`WRELA_PIPELINE_CACHE=off`,
    /// `WRELA_DUMP_IR` set, no cache directory available, or current executable unreadable).
    pub fn open(
        sources: &SourceMap,
        target: wrela_abi::Target,
        checks: wrela_shader::Checks,
    ) -> Option<DiskPipelineCache> {
        if std::env::var_os("WRELA_DUMP_IR").is_some() {
            return None;
        }
        let dir = cache_dir()?;
        let exe_hash = exe_sha256()?;
        let std_hash = std_sources_sha256(sources);
        clean_if_due(&dir);
        Some(DiskPipelineCache { dir, exe_hash, std_hash, target, checks })
    }
}

fn cache_dir() -> Option<PathBuf> {
    let named = std::env::var_os("WRELA_PIPELINE_CACHE").filter(|d| !d.is_empty());
    if named.as_deref() == Some(std::ffi::OsStr::new("off")) {
        return None;
    }
    let base = match named {
        Some(d) => PathBuf::from(d),
        None => {
            let home = std::env::var_os("HOME").filter(|d| !d.is_empty())?;
            PathBuf::from(home).join(".wrela").join("pipelines")
        }
    };
    let dir = base.join(VERSION);
    crate::ensure_no_symlink_path(&base, &dir).ok()?;
    Some(dir)
}

fn exe_sha256() -> Option<[u8; 32]> {
    static HASH: OnceLock<Option<[u8; 32]>> = OnceLock::new();
    *HASH.get_or_init(|| {
        let exe = std::env::current_exe().ok()?;
        let bytes = std::fs::read(exe).ok()?;
        Some(Sha256::digest(&bytes).into())
    })
}

fn std_sources_sha256(sources: &SourceMap) -> [u8; 32] {
    let mut h = Sha256::new();
    for (id, file) in sources.files() {
        if file.name.starts_with("<std::") {
            h.update(id.0.to_le_bytes());
            h.update((file.name.len() as u32).to_le_bytes());
            h.update(file.name.as_bytes());
            h.update((file.text.len() as u32).to_le_bytes());
            h.update(file.text.as_bytes());
        }
    }
    h.finalize().into()
}

impl PipelineCache for DiskPipelineCache {
    fn key(&self, input: &PipelineCacheKeyInput<'_>) -> Option<String> {
        let mut h = Sha256::new();
        h.update(MAGIC);
        h.update(self.exe_hash);
        if input.module.strict {
            h.update([1u8]);
            h.update(self.std_hash);
        } else {
            h.update([0u8]);
        }
        h.update([match input.target {
            wrela_abi::Target::Metal => 0u8,
            wrela_abi::Target::Vulkan => 1u8,
        }]);
        h.update([
            u8::from(self.checks.bounds),
            u8::from(self.checks.loops),
            u8::from(self.checks.int_div),
            u8::from(self.checks.exact),
            u8::from(input.exact),
            u8::from(input.hoist),
        ]);
        match input.kind {
            PipelineKind::Compute { workgroup_size } => {
                h.update([0u8]);
                for &x in workgroup_size {
                    h.update(x.to_le_bytes());
                }
            }
            PipelineKind::Render { .. } => {
                h.update([1u8]);
            }
        }
        match input.uniform {
            Some(u) => {
                h.update([1u8]);
                h.update(u.binding.to_le_bytes());
                h.update(u.size.to_le_bytes());
                h.update([u8::from(u.storage)]);
            }
            None => h.update([0u8]),
        }
        h.update((input.bindings.len() as u32).to_le_bytes());
        for b in input.bindings {
            h.update(b.binding.to_le_bytes());
            h.update([binding_kind_tag(b.kind), binding_stage_tag(b.stage)]);
            match b.format {
                Some(f) => {
                    h.update([1u8]);
                    h.update((f as u32).to_le_bytes());
                }
                None => h.update([0u8]),
            }
        }
        ir::fingerprint::feed_module(input.module, &mut |bytes| h.update(bytes));
        let mut out = String::with_capacity(KEY_BYTES);
        for b in h.finalize() {
            let _ = write!(out, "{b:02x}");
        }
        Some(out)
    }

    fn get(&self, key: &str) -> Option<CachedPipeline> {
        let path = self.dir.join(format!("{key}.wpv"));
        crate::ensure_no_symlink_path(&self.dir, &path).ok()?;
        let bytes = std::fs::read(&path).ok()?;
        let entry = decode(key, &bytes)?;
        let _ = std::fs::File::options()
            .append(true)
            .open(&path)
            .and_then(|f| f.set_modified(std::time::SystemTime::now()));
        Some(entry)
    }

    fn emit_and_put(
        &self,
        key: &str,
        out: &PipelineOut,
        uncounted_most: Option<u64>,
    ) -> Option<Result<PreEmittedShader, String>> {
        let stage = match (&out.kind, out.module.entry_points.as_slice()) {
            (PipelineKind::Compute { workgroup_size }, [e]) => {
                Stage::Compute { entry: e.name.clone(), workgroup_size: *workgroup_size }
            }
            (PipelineKind::Render { .. }, [v, f]) => Stage::Render {
                vertex_entry: v.name.clone(),
                fragment_entry: f.name.clone(),
                blends: Vec::new(),
                cull: Cull::None,
                depth_bias: DepthBias { constant: 0, slope_scale: 0.0, clamp: 0.0 },
                depth: DepthState::default(),
                writes_depth: false,
                outputs: Vec::new(),
                targets: Vec::new(),
            },
            _ => return None,
        };
        let mut pipeline = Pipeline {
            name: out.name.clone(),
            shader: String::new(),
            stage,
            uniform: out.uniform.as_ref().map(|u| UniformBlock {
                binding: u.binding,
                size: u.size,
                space: if u.storage { UniformSpace::Storage } else { UniformSpace::Uniform },
            }),
            bindings: out.bindings.clone(),
            debug_flag: None,
            requires: Vec::new(),
            conditional: false,
            strict: out.module.strict,
            standin: None,
            sizes: None,
            kept: 0,
            work: Vec::new(),
            unbounded: Vec::new(),
        };
        let written =
            match wrela_shader::emit_shader(&out.module, &mut pipeline, self.target, self.checks) {
                Ok(w) => w,
                Err(e) => {
                    return Some(Err(format!(
                        "the {} back end failed for `{}`: {e}",
                        self.target.name(),
                        out.name
                    )));
                }
            };
        let pre = PreEmittedShader {
            bytes: written.bytes,
            entry_points: written.entry_points,
            requires: out.module.requires(),
            kept: out.module.kept_bytes(),
            sizes: pipeline.sizes,
        };
        let cached = CachedPipeline {
            uniform: out.uniform.clone(),
            bindings: out.bindings.clone(),
            work: out.work.clone(),
            unbounded: out.unbounded.clone(),
            weight: out.weight,
            same: out.same.clone(),
            costly: out.costly.clone(),
            copies: out.copies.clone(),
            hoisted: out.hoisted.clone(),
            hoisted_types: (!out.hoisted.is_empty()).then(|| out.module.types.clone()),
            uncounted_most,
            shader: pre.clone(),
        };
        if let Some(bytes) = encode(key, &cached) {
            write_atomic(&self.dir, key, "wpv", &bytes);
        }
        Some(Ok(pre))
    }

    fn work_key(
        &self,
        module: &ir::Module,
        id: ir::FuncId,
        strict: bool,
        divides_by_std: bool,
    ) -> Option<String> {
        let mut h = Sha256::new();
        h.update(MAGIC_WORK);
        h.update(self.exe_hash);
        h.update([u8::from(strict), u8::from(strict && divides_by_std)]);
        if strict {
            h.update(self.std_hash);
        }
        h.update(id.0.to_le_bytes());
        ir::fingerprint::feed_module(module, &mut |bytes| h.update(bytes));
        let mut out = String::with_capacity(KEY_BYTES);
        for b in h.finalize() {
            let _ = write!(out, "{b:02x}");
        }
        Some(out)
    }

    fn get_work(&self, key: &str) -> Option<u32> {
        let path = self.dir.join(format!("{key}.wwk"));
        crate::ensure_no_symlink_path(&self.dir, &path).ok()?;
        let bytes = std::fs::read(&path).ok()?;
        let work = decode_work(key, &bytes)?;
        let _ = std::fs::File::options()
            .append(true)
            .open(&path)
            .and_then(|f| f.set_modified(std::time::SystemTime::now()));
        Some(work)
    }

    fn put_work(&self, key: &str, work: u32) {
        if let Some(bytes) = encode_work(key, work) {
            write_atomic(&self.dir, key, "wwk", &bytes);
        }
    }
}

fn encode_work(key: &str, work: u32) -> Option<Vec<u8>> {
    if !is_key(key) {
        return None;
    }
    let mut out = Vec::with_capacity(MAGIC_WORK.len() + KEY_BYTES + 4 + CHECKSUM_BYTES);
    out.extend_from_slice(MAGIC_WORK);
    out.extend_from_slice(key.as_bytes());
    out.extend_from_slice(&work.to_le_bytes());
    let checksum = Sha256::digest(&out);
    out.extend_from_slice(&checksum);
    Some(out)
}

fn decode_work(key: &str, bytes: &[u8]) -> Option<u32> {
    if !is_key(key) {
        return None;
    }
    let checksum_at = bytes.len().checked_sub(CHECKSUM_BYTES)?;
    let (body, stored) = bytes.split_at(checksum_at);
    let actual: [u8; CHECKSUM_BYTES] = Sha256::digest(body).into();
    if actual.as_slice() != stored {
        return None;
    }
    let rest = body.strip_prefix(MAGIC_WORK.as_slice())?.strip_prefix(key.as_bytes())?;
    let arr: [u8; 4] = rest.try_into().ok()?;
    Some(u32::from_le_bytes(arr))
}

fn write_atomic(dir: &Path, key: &str, ext: &str, bytes: &[u8]) {
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp = dir.join(format!("{key}.{}-{nanos}-{seq}.tmp", std::process::id()));
    let dst = dir.join(format!("{key}.{ext}"));
    if crate::ensure_no_symlink_path(dir, &tmp).is_err()
        || crate::ensure_no_symlink_path(dir, &dst).is_err()
    {
        return;
    }
    if std::fs::write(&tmp, bytes).is_ok() && std::fs::rename(&tmp, &dst).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

fn clean_if_due(dir: &Path) {
    let marker = dir.join(SWEPT);
    let day = std::time::Duration::from_secs(24 * 3600);
    let fresh = std::fs::metadata(&marker)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.elapsed().ok())
        .is_some_and(|age| age < day);
    if fresh {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let _ = std::fs::write(&marker, b"");
    let now = std::time::SystemTime::now();
    for e in entries.flatten() {
        if !e.file_type().is_ok_and(|t| t.is_file()) {
            continue;
        }
        let name = e.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(prefix) = name.get(..KEY_BYTES) else { continue };
        if !is_key(prefix) {
            continue;
        }
        let rest = &name[KEY_BYTES..];
        let temporary = if rest == ".wpv" || rest == ".wwk" {
            false
        } else if rest.starts_with('.') && rest.ends_with(".tmp") {
            true
        } else {
            continue;
        };
        let age = e
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| now.duration_since(t).ok())
            .unwrap_or_default();
        if (!temporary && age > KEEP_FOR) || (temporary && age.as_secs() > 3600) {
            let _ = std::fs::remove_file(e.path());
        }
    }
}

fn is_key(s: &str) -> bool {
    s.len() == KEY_BYTES && s.bytes().all(|b| b.is_ascii_hexdigit())
}

// ---- binary codec -------------------------------------------------------------------------------

fn encode(key: &str, c: &CachedPipeline) -> Option<Vec<u8>> {
    if !is_key(key) {
        return None;
    }
    let mut out = Vec::with_capacity(2048 + c.shader.bytes.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(key.as_bytes());
    put_opt(&mut out, c.uniform.as_ref(), |o, u| {
        put_u32(o, u.binding);
        put_u32(o, u.size);
        o.push(u8::from(u.storage));
    });
    put_slice(&mut out, &c.bindings, |o, b| {
        put_u32(o, b.binding);
        o.push(binding_kind_tag(b.kind));
        o.push(binding_stage_tag(b.stage));
        put_opt(o, b.format.as_ref(), |o, f| put_u32(o, *f as u32));
    });
    put_slice(&mut out, &c.work, |o, w| put_opt(o, w.as_ref(), |o, &x| put_u64(o, x)));
    put_slice(&mut out, &c.unbounded, |o, s| put_opt(o, s.as_ref(), |o, &sp| put_span(o, sp)));
    put_u64(&mut out, c.weight);
    put_slice(&mut out, &c.same, |o, s| {
        put_str(o, &s.function);
        put_str(o, &s.entry);
        put_slice(o, &s.path, |o, p| put_str(o, p));
        put_opt(o, s.at.as_ref(), |o, &sp| put_span(o, sp));
        put_opt(o, s.work.as_ref(), |o, &x| put_u64(o, x));
    });
    put_slice(&mut out, &c.costly, |o, k| {
        put_span(o, k.at);
        o.push(match k.why {
            ir::rates::Why::Constants => 0,
            ir::rates::Why::SameCount => 1,
            ir::rates::Why::Heavy => 2,
        });
        put_opt(o, k.passes.as_ref(), |o, &x| put_u32(o, x));
        put_opt(o, k.work.as_ref(), |o, &x| put_u64(o, x));
        o.push(match k.each {
            "pixel" => 0,
            "vertex" => 1,
            "invocation" => 2,
            _ => 3,
        });
        put_opt(o, k.callee.as_ref(), |o, s| put_str(o, s));
        put_u64(o, k.times);
        put_opt(o, k.of.as_ref(), |o, &x| put_u64(o, x));
    });
    put_slice(&mut out, &c.copies, |o, cp| {
        put_str(o, &cp.function);
        put_opt(o, cp.at.as_ref(), |o, &sp| put_span(o, sp));
        put_u32(o, cp.bytes);
        put_str(o, &cp.what);
        o.push(u8::from(cp.uniform));
    });
    put_u32(&mut out, c.hoisted.len() as u32);
    for h in &c.hoisted {
        put_u32(&mut out, h.field);
        put_u32(&mut out, h.ty.0);
        put_opt(&mut out, h.count.as_ref(), |o, &(a, b)| {
            put_u32(o, a);
            put_u32(o, b);
        });
        put_recipe(&mut out, &h.recipe)?;
    }
    put_opt(&mut out, c.hoisted_types.as_ref(), |o, types| {
        let defs: Vec<&ir::TypeDef> = types.iter().map(|(_, d)| d).collect();
        put_u32(o, defs.len() as u32);
        for d in defs {
            put_type_def(o, d);
        }
    });
    put_opt(&mut out, c.uncounted_most.as_ref(), |o, &x| put_u64(o, x));
    put_slice(&mut out, &c.shader.entry_points, |o, s| put_str(o, s));
    put_slice(&mut out, &c.shader.requires, |o, s| put_str(o, s));
    put_u32(&mut out, c.shader.kept);
    put_opt(&mut out, c.shader.sizes.as_ref(), |o, sz| {
        put_u32(o, sz.binding);
        put_slice(o, &sz.buffers, |o, &b| put_u32(o, b));
    });
    put_bytes(&mut out, &c.shader.bytes);
    let checksum = Sha256::digest(&out);
    out.extend_from_slice(&checksum);
    Some(out)
}

fn decode(key: &str, bytes: &[u8]) -> Option<CachedPipeline> {
    if !is_key(key) {
        return None;
    }
    let checksum_at = bytes.len().checked_sub(CHECKSUM_BYTES)?;
    let (body, stored) = bytes.split_at(checksum_at);
    let actual: [u8; CHECKSUM_BYTES] = Sha256::digest(body).into();
    if actual.as_slice() != stored {
        return None;
    }
    let rest = body.strip_prefix(MAGIC.as_slice())?.strip_prefix(key.as_bytes())?;
    let mut r = Reader { bytes: rest, at: 0 };
    let uniform =
        r.opt(|r| Some(UniformOut { binding: r.u32()?, size: r.u32()?, storage: r.bool()? }))?;
    let bindings = r.vec(|r| {
        Some(ResourceBinding {
            binding: r.u32()?,
            kind: binding_kind_from(r.u8()?)?,
            stage: binding_stage_from(r.u8()?)?,
            format: r.opt(|r| TextureFormat::from_u32(r.u32()?))?,
        })
    })?;
    let work = r.vec(|r| r.opt(|r| r.u64()))?;
    let unbounded = r.vec(|r| r.opt(|r| r.span()))?;
    let weight = r.u64()?;
    let same = r.vec(|r| {
        Some(ir::same::SameWork {
            function: r.str()?,
            entry: r.str()?,
            path: r.vec(|r| r.str())?,
            at: r.opt(|r| r.span())?,
            work: r.opt(|r| r.u64())?,
        })
    })?;
    let costly = r.vec(|r| {
        let at = r.span()?;
        let why = match r.u8()? {
            0 => ir::rates::Why::Constants,
            1 => ir::rates::Why::SameCount,
            2 => ir::rates::Why::Heavy,
            _ => return None,
        };
        let passes = r.opt(|r| r.u32())?;
        let work = r.opt(|r| r.u64())?;
        let each = match r.u8()? {
            0 => "pixel",
            1 => "vertex",
            2 => "invocation",
            _ => return None,
        };
        let callee = r.opt(|r| r.str())?;
        let times = r.u64()?;
        let of = r.opt(|r| r.u64())?;
        Some(ir::rates::Costly { at, why, passes, work, each, callee, times, of })
    })?;
    let copies = r.vec(|r| {
        Some(ir::opt::Copied {
            function: r.str()?,
            at: r.opt(|r| r.span())?,
            bytes: r.u32()?,
            what: r.str()?,
            uniform: r.bool()?,
        })
    })?;
    let hoisted = r.vec(|r| {
        let field = r.u32()?;
        let ty = ir::TypeId(r.u32()?);
        let count = r.opt(|r| Some((r.u32()?, r.u32()?)))?;
        let recipe = r.recipe()?;
        Some(ir::hoist::Hoisted { field, ty, recipe, count })
    })?;
    let hoisted_types = r.opt(|r| {
        let defs = r.vec(|r| r.type_def())?;
        ir::Types::from_defs(defs)
    })?;
    let uncounted_most = r.opt(|r| r.u64())?;
    let entry_points = r.vec(|r| r.str())?;
    let requires = r.vec(|r| r.str())?;
    let kept = r.u32()?;
    let sizes = r.opt(|r| Some(BufferSizes { binding: r.u32()?, buffers: r.vec(|r| r.u32())? }))?;
    let shader_bytes = r.bytes()?;
    if r.at != r.bytes.len() {
        return None;
    }
    Some(CachedPipeline {
        uniform,
        bindings,
        work,
        unbounded,
        weight,
        same,
        costly,
        copies,
        hoisted,
        hoisted_types,
        uncounted_most,
        shader: PreEmittedShader { bytes: shader_bytes, entry_points, requires, kept, sizes },
    })
}

fn binding_kind_tag(k: BindingKind) -> u8 {
    match k {
        BindingKind::Read => 0,
        BindingKind::ReadWrite => 1,
        BindingKind::Texture => 2,
        BindingKind::DepthTexture => 3,
        BindingKind::Sampler => 4,
        BindingKind::ComparisonSampler => 5,
        BindingKind::NonFilteringSampler => 6,
        BindingKind::StorageTexture => 7,
        BindingKind::Texture3d => 8,
        BindingKind::StorageTexture3d => 9,
        BindingKind::TextureArray => 10,
        BindingKind::TextureCube => 11,
        BindingKind::DepthTextureArray => 12,
        BindingKind::StorageTextureArray => 13,
    }
}

fn binding_kind_from(t: u8) -> Option<BindingKind> {
    Some(match t {
        0 => BindingKind::Read,
        1 => BindingKind::ReadWrite,
        2 => BindingKind::Texture,
        3 => BindingKind::DepthTexture,
        4 => BindingKind::Sampler,
        5 => BindingKind::ComparisonSampler,
        6 => BindingKind::NonFilteringSampler,
        7 => BindingKind::StorageTexture,
        8 => BindingKind::Texture3d,
        9 => BindingKind::StorageTexture3d,
        10 => BindingKind::TextureArray,
        11 => BindingKind::TextureCube,
        12 => BindingKind::DepthTextureArray,
        13 => BindingKind::StorageTextureArray,
        _ => return None,
    })
}

fn binding_stage_tag(s: BindingStage) -> u8 {
    match s {
        BindingStage::Both => 0,
        BindingStage::Vertex => 1,
        BindingStage::Fragment => 2,
    }
}

fn binding_stage_from(t: u8) -> Option<BindingStage> {
    Some(match t {
        0 => BindingStage::Both,
        1 => BindingStage::Vertex,
        2 => BindingStage::Fragment,
        _ => return None,
    })
}

fn put_u32(out: &mut Vec<u8>, x: u32) {
    out.extend_from_slice(&x.to_le_bytes());
}
fn put_i32(out: &mut Vec<u8>, x: i32) {
    out.extend_from_slice(&x.to_le_bytes());
}
fn put_u64(out: &mut Vec<u8>, x: u64) {
    out.extend_from_slice(&x.to_le_bytes());
}
fn put_i64(out: &mut Vec<u8>, x: i64) {
    out.extend_from_slice(&x.to_le_bytes());
}
fn put_str(out: &mut Vec<u8>, s: &str) {
    put_u32(out, s.len() as u32);
    out.extend_from_slice(s.as_bytes());
}
fn put_bytes(out: &mut Vec<u8>, bs: &[u8]) {
    put_u32(out, bs.len() as u32);
    out.extend_from_slice(bs);
}
fn put_span(out: &mut Vec<u8>, s: Span) {
    put_u32(out, s.file.0);
    put_u32(out, s.start);
    put_u32(out, s.end);
}
fn put_opt<T>(out: &mut Vec<u8>, v: Option<&T>, f: impl FnOnce(&mut Vec<u8>, &T)) {
    match v {
        Some(x) => {
            out.push(1);
            f(out, x);
        }
        None => out.push(0),
    }
}
fn put_slice<T>(out: &mut Vec<u8>, xs: &[T], mut f: impl FnMut(&mut Vec<u8>, &T)) {
    put_u32(out, xs.len() as u32);
    for x in xs {
        f(out, x);
    }
}

fn scalar_tag(s: ir::Scalar) -> u8 {
    match s {
        ir::Scalar::Bool => 0,
        ir::Scalar::I8 => 1,
        ir::Scalar::U8 => 2,
        ir::Scalar::I16 => 3,
        ir::Scalar::U16 => 4,
        ir::Scalar::I32 => 5,
        ir::Scalar::U32 => 6,
        ir::Scalar::I64 => 7,
        ir::Scalar::U64 => 8,
        ir::Scalar::F32 => 9,
        ir::Scalar::F64 => 10,
        ir::Scalar::F16 => 11,
    }
}

fn scalar_from(t: u8) -> Option<ir::Scalar> {
    Some(match t {
        0 => ir::Scalar::Bool,
        1 => ir::Scalar::I8,
        2 => ir::Scalar::U8,
        3 => ir::Scalar::I16,
        4 => ir::Scalar::U16,
        5 => ir::Scalar::I32,
        6 => ir::Scalar::U32,
        7 => ir::Scalar::I64,
        8 => ir::Scalar::U64,
        9 => ir::Scalar::F32,
        10 => ir::Scalar::F64,
        11 => ir::Scalar::F16,
        _ => return None,
    })
}

fn put_type_def(out: &mut Vec<u8>, d: &ir::TypeDef) {
    match d {
        ir::TypeDef::Scalar(s) => {
            out.push(0);
            out.push(scalar_tag(*s));
        }
        ir::TypeDef::Vector(s, n) => {
            out.push(1);
            out.push(scalar_tag(*s));
            out.push(*n);
        }
        ir::TypeDef::Matrix(c, r) => {
            out.push(2);
            out.push(*c);
            out.push(*r);
        }
        ir::TypeDef::Struct { name, fields } => {
            out.push(3);
            put_str(out, name);
            put_slice(out, fields, |o, (n, t)| {
                put_str(o, n);
                put_u32(o, t.0);
            });
        }
        ir::TypeDef::Enum { name, variants, tags } => {
            out.push(4);
            put_str(out, name);
            put_slice(out, variants, |o, (n, p)| {
                put_str(o, n);
                put_opt(o, p.as_ref(), |o, t| put_u32(o, t.0));
            });
            put_slice(out, tags, |o, &t| put_u32(o, t));
        }
        ir::TypeDef::Array(e, n) => {
            out.push(5);
            put_u32(out, e.0);
            put_u32(out, *n);
        }
        ir::TypeDef::RuntimeArray(e) => {
            out.push(6);
            put_u32(out, e.0);
        }
        ir::TypeDef::Run(e) => {
            out.push(7);
            put_u32(out, e.0);
        }
        ir::TypeDef::Ptr(e) => {
            out.push(8);
            put_u32(out, e.0);
        }
        ir::TypeDef::Atomic(s) => {
            out.push(9);
            out.push(scalar_tag(*s));
        }
    }
}

fn put_const(out: &mut Vec<u8>, c: &ir::Const) {
    match c {
        ir::Const::Bool(b) => {
            out.push(0);
            out.push(u8::from(*b));
        }
        ir::Const::I32(x) => {
            out.push(1);
            put_i32(out, *x);
        }
        ir::Const::U32(x) => {
            out.push(2);
            put_u32(out, *x);
        }
        ir::Const::I64(x) => {
            out.push(3);
            put_i64(out, *x);
        }
        ir::Const::U64(x) => {
            out.push(4);
            put_u64(out, *x);
        }
        ir::Const::Small(s, x) => {
            out.push(5);
            out.push(scalar_tag(*s));
            put_i64(out, *x);
        }
        ir::Const::F32(x) => {
            out.push(6);
            put_u32(out, x.to_bits());
        }
        ir::Const::F64(x) => {
            out.push(7);
            put_u64(out, x.to_bits());
        }
        ir::Const::F16(x) => {
            out.push(8);
            put_u32(out, x.to_bits());
        }
    }
}

fn put_place(out: &mut Vec<u8>, p: &ir::Place) -> Option<()> {
    match p.root {
        ir::PlaceRoot::Local(l) => {
            out.push(0);
            put_u32(out, l.0);
        }
        ir::PlaceRoot::Param(i) => {
            out.push(1);
            put_u32(out, i);
        }
        ir::PlaceRoot::Resource(r) => {
            out.push(2);
            put_u32(out, r.0);
        }
        ir::PlaceRoot::Ptr(v) => {
            out.push(3);
            put_u32(out, v.0);
        }
        ir::PlaceRoot::Data(d) => {
            out.push(4);
            put_u32(out, d.0);
        }
    }
    put_slice(out, &p.path, |o, proj| match proj {
        ir::Proj::Field(k) => {
            o.push(0);
            put_u32(o, *k);
        }
        ir::Proj::Comp(c) => {
            o.push(1);
            o.push(*c);
        }
        ir::Proj::Index(v) => {
            o.push(2);
            put_u32(o, v.0);
        }
    });
    Some(())
}

fn binop_tag(op: ir::BinOp) -> u8 {
    match op {
        ir::BinOp::Add => 0,
        ir::BinOp::Sub => 1,
        ir::BinOp::Mul => 2,
        ir::BinOp::Div => 3,
        ir::BinOp::Rem => 4,
        ir::BinOp::WrappingAdd => 5,
        ir::BinOp::WrappingSub => 6,
        ir::BinOp::WrappingMul => 7,
        ir::BinOp::Shl => 8,
        ir::BinOp::Shr => 9,
        ir::BinOp::BitAnd => 10,
        ir::BinOp::BitOr => 11,
        ir::BinOp::BitXor => 12,
        ir::BinOp::And => 13,
        ir::BinOp::Or => 14,
        ir::BinOp::Eq => 15,
        ir::BinOp::Ne => 16,
        ir::BinOp::Lt => 17,
        ir::BinOp::Le => 18,
        ir::BinOp::Gt => 19,
        ir::BinOp::Ge => 20,
    }
}

fn binop_from(t: u8) -> Option<ir::BinOp> {
    Some(match t {
        0 => ir::BinOp::Add,
        1 => ir::BinOp::Sub,
        2 => ir::BinOp::Mul,
        3 => ir::BinOp::Div,
        4 => ir::BinOp::Rem,
        5 => ir::BinOp::WrappingAdd,
        6 => ir::BinOp::WrappingSub,
        7 => ir::BinOp::WrappingMul,
        8 => ir::BinOp::Shl,
        9 => ir::BinOp::Shr,
        10 => ir::BinOp::BitAnd,
        11 => ir::BinOp::BitOr,
        12 => ir::BinOp::BitXor,
        13 => ir::BinOp::And,
        14 => ir::BinOp::Or,
        15 => ir::BinOp::Eq,
        16 => ir::BinOp::Ne,
        17 => ir::BinOp::Lt,
        18 => ir::BinOp::Le,
        19 => ir::BinOp::Gt,
        20 => ir::BinOp::Ge,
        _ => return None,
    })
}

fn builtin_tag(b: ir::Builtin) -> u8 {
    match b {
        ir::Builtin::Sqrt => 0,
        ir::Builtin::InverseSqrt => 1,
        ir::Builtin::Sin => 2,
        ir::Builtin::Cos => 3,
        ir::Builtin::Tan => 4,
        ir::Builtin::Asin => 5,
        ir::Builtin::Acos => 6,
        ir::Builtin::Atan => 7,
        ir::Builtin::Atan2 => 8,
        ir::Builtin::Exp => 9,
        ir::Builtin::Exp2 => 10,
        ir::Builtin::Log => 11,
        ir::Builtin::Log2 => 12,
        ir::Builtin::Sinh => 13,
        ir::Builtin::Cosh => 14,
        ir::Builtin::Tanh => 15,
        ir::Builtin::Pow => 16,
        ir::Builtin::Floor => 17,
        ir::Builtin::Ceil => 18,
        ir::Builtin::Round => 19,
        ir::Builtin::Trunc => 20,
        ir::Builtin::Fract => 21,
        ir::Builtin::Abs => 22,
        ir::Builtin::Sign => 23,
        ir::Builtin::Min => 24,
        ir::Builtin::Max => 25,
        ir::Builtin::Clamp => 26,
        ir::Builtin::Saturate => 27,
        ir::Builtin::Mix => 28,
        ir::Builtin::Step => 29,
        ir::Builtin::Smoothstep => 30,
        ir::Builtin::Length => 31,
        ir::Builtin::Distance => 32,
        ir::Builtin::Dot => 33,
        ir::Builtin::Cross => 34,
        ir::Builtin::Normalize => 35,
        ir::Builtin::Dpdx => 36,
        ir::Builtin::Dpdy => 37,
        ir::Builtin::Fwidth => 38,
        ir::Builtin::AllEqual => 39,
        ir::Builtin::CountOnes => 40,
        ir::Builtin::LeadingZeros => 41,
        ir::Builtin::TrailingZeros => 42,
    }
}

fn builtin_from(t: u8) -> Option<ir::Builtin> {
    Some(match t {
        0 => ir::Builtin::Sqrt,
        1 => ir::Builtin::InverseSqrt,
        2 => ir::Builtin::Sin,
        3 => ir::Builtin::Cos,
        4 => ir::Builtin::Tan,
        5 => ir::Builtin::Asin,
        6 => ir::Builtin::Acos,
        7 => ir::Builtin::Atan,
        8 => ir::Builtin::Atan2,
        9 => ir::Builtin::Exp,
        10 => ir::Builtin::Exp2,
        11 => ir::Builtin::Log,
        12 => ir::Builtin::Log2,
        13 => ir::Builtin::Sinh,
        14 => ir::Builtin::Cosh,
        15 => ir::Builtin::Tanh,
        16 => ir::Builtin::Pow,
        17 => ir::Builtin::Floor,
        18 => ir::Builtin::Ceil,
        19 => ir::Builtin::Round,
        20 => ir::Builtin::Trunc,
        21 => ir::Builtin::Fract,
        22 => ir::Builtin::Abs,
        23 => ir::Builtin::Sign,
        24 => ir::Builtin::Min,
        25 => ir::Builtin::Max,
        26 => ir::Builtin::Clamp,
        27 => ir::Builtin::Saturate,
        28 => ir::Builtin::Mix,
        29 => ir::Builtin::Step,
        30 => ir::Builtin::Smoothstep,
        31 => ir::Builtin::Length,
        32 => ir::Builtin::Distance,
        33 => ir::Builtin::Dot,
        34 => ir::Builtin::Cross,
        35 => ir::Builtin::Normalize,
        36 => ir::Builtin::Dpdx,
        37 => ir::Builtin::Dpdy,
        38 => ir::Builtin::Fwidth,
        39 => ir::Builtin::AllEqual,
        40 => ir::Builtin::CountOnes,
        41 => ir::Builtin::LeadingZeros,
        42 => ir::Builtin::TrailingZeros,
        _ => return None,
    })
}

fn put_recipe_expr(out: &mut Vec<u8>, e: &ir::Expr) -> Option<()> {
    match e {
        ir::Expr::Const(c) => {
            out.push(0);
            put_const(out, c);
        }
        ir::Expr::Zero(t) => {
            out.push(1);
            put_u32(out, t.0);
        }
        ir::Expr::Load(p) => {
            out.push(2);
            put_place(out, p)?;
        }
        ir::Expr::Unary(op, v) => {
            out.push(3);
            out.push(match op {
                ir::UnOp::Neg => 0,
                ir::UnOp::Not => 1,
            });
            put_u32(out, v.0);
        }
        ir::Expr::Binary(op, a, b) => {
            out.push(4);
            out.push(binop_tag(*op));
            put_u32(out, a.0);
            put_u32(out, b.0);
        }
        ir::Expr::Builtin(b, xs) => {
            out.push(5);
            out.push(builtin_tag(*b));
            put_slice(out, xs, |o, v| put_u32(o, v.0));
        }
        ir::Expr::Construct(t, xs) => {
            out.push(6);
            put_u32(out, t.0);
            put_slice(out, xs, |o, v| put_u32(o, v.0));
        }
        ir::Expr::Variant(t, k, p) => {
            out.push(7);
            put_u32(out, t.0);
            put_u32(out, *k);
            put_opt(out, p.as_ref(), |o, v| put_u32(o, v.0));
        }
        ir::Expr::Extract(v, k) => {
            out.push(8);
            put_u32(out, v.0);
            put_u32(out, *k);
        }
        ir::Expr::ExtractDyn(v, i) => {
            out.push(9);
            put_u32(out, v.0);
            put_u32(out, i.0);
        }
        ir::Expr::Splat(v, n) => {
            out.push(10);
            put_u32(out, v.0);
            out.push(*n);
        }
        ir::Expr::Swizzle(v, cs) => {
            out.push(11);
            put_u32(out, v.0);
            put_bytes(out, cs);
        }
        ir::Expr::Convert(v, s) => {
            out.push(12);
            put_u32(out, v.0);
            out.push(scalar_tag(*s));
        }
        ir::Expr::Bitcast(v, s) => {
            out.push(13);
            put_u32(out, v.0);
            out.push(scalar_tag(*s));
        }
        ir::Expr::Select { cond, if_true, if_false } => {
            out.push(14);
            put_u32(out, cond.0);
            put_u32(out, if_true.0);
            put_u32(out, if_false.0);
        }
        ir::Expr::Param(i) => {
            out.push(15);
            put_u32(out, *i);
        }
        _ => return None,
    }
    Some(())
}

fn put_recipe(out: &mut Vec<u8>, f: &ir::Function) -> Option<()> {
    put_str(out, &f.name);
    put_slice(out, &f.params, |o, p| {
        put_str(o, &p.name);
        put_u32(o, p.ty.0);
        o.push(u8::from(p.by_ref));
        o.push(u8::from(p.mutable));
    });
    put_opt(out, f.ret.as_ref(), |o, t| put_u32(o, t.0));
    put_slice(out, &f.values, |o, t| put_u32(o, t.0));
    put_u32(out, f.body.len() as u32);
    for s in &f.body {
        match s {
            ir::Stmt::Let(v, e) => {
                out.push(0);
                put_u32(out, v.0);
                put_recipe_expr(out, e)?;
            }
            ir::Stmt::Return(v) => {
                out.push(1);
                put_opt(out, v.as_ref(), |o, v| put_u32(o, v.0));
            }
            ir::Stmt::At(span) => {
                out.push(2);
                put_span(out, *span);
            }
            _ => return None,
        }
    }
    Some(())
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.at.checked_add(n)?;
        let slice = self.bytes.get(self.at..end)?;
        self.at = end;
        Some(slice)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn bool(&mut self) -> Option<bool> {
        match self.u8()? {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn i32(&mut self) -> Option<i32> {
        Some(i32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn i64(&mut self) -> Option<i64> {
        Some(i64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn bytes(&mut self) -> Option<Vec<u8>> {
        let n = self.u32()? as usize;
        Some(self.take(n)?.to_vec())
    }
    fn str(&mut self) -> Option<String> {
        let b = self.bytes()?;
        String::from_utf8(b).ok()
    }
    fn span(&mut self) -> Option<Span> {
        Some(Span { file: FileId(self.u32()?), start: self.u32()?, end: self.u32()? })
    }
    fn opt<T>(&mut self, f: impl FnOnce(&mut Self) -> Option<T>) -> Option<Option<T>> {
        match self.u8()? {
            0 => Some(None),
            1 => Some(Some(f(self)?)),
            _ => None,
        }
    }
    fn vec<T>(&mut self, mut f: impl FnMut(&mut Self) -> Option<T>) -> Option<Vec<T>> {
        let n = self.u32()? as usize;
        let mut out = Vec::with_capacity(n.min(4096));
        for _ in 0..n {
            out.push(f(self)?);
        }
        Some(out)
    }
    fn type_def(&mut self) -> Option<ir::TypeDef> {
        Some(match self.u8()? {
            0 => ir::TypeDef::Scalar(scalar_from(self.u8()?)?),
            1 => ir::TypeDef::Vector(scalar_from(self.u8()?)?, self.u8()?),
            2 => ir::TypeDef::Matrix(self.u8()?, self.u8()?),
            3 => ir::TypeDef::Struct {
                name: self.str()?,
                fields: self.vec(|r| Some((r.str()?, ir::TypeId(r.u32()?))))?,
            },
            4 => ir::TypeDef::Enum {
                name: self.str()?,
                variants: self.vec(|r| Some((r.str()?, r.opt(|r| Some(ir::TypeId(r.u32()?)))?)))?,
                tags: self.vec(|r| r.u32())?,
            },
            5 => ir::TypeDef::Array(ir::TypeId(self.u32()?), self.u32()?),
            6 => ir::TypeDef::RuntimeArray(ir::TypeId(self.u32()?)),
            7 => ir::TypeDef::Run(ir::TypeId(self.u32()?)),
            8 => ir::TypeDef::Ptr(ir::TypeId(self.u32()?)),
            9 => ir::TypeDef::Atomic(scalar_from(self.u8()?)?),
            _ => return None,
        })
    }
    fn const_(&mut self) -> Option<ir::Const> {
        Some(match self.u8()? {
            0 => ir::Const::Bool(self.bool()?),
            1 => ir::Const::I32(self.i32()?),
            2 => ir::Const::U32(self.u32()?),
            3 => ir::Const::I64(self.i64()?),
            4 => ir::Const::U64(self.u64()?),
            5 => ir::Const::Small(scalar_from(self.u8()?)?, self.i64()?),
            6 => ir::Const::F32(f32::from_bits(self.u32()?)),
            7 => ir::Const::F64(f64::from_bits(self.u64()?)),
            8 => ir::Const::F16(f32::from_bits(self.u32()?)),
            _ => return None,
        })
    }
    fn place(&mut self) -> Option<ir::Place> {
        let root = match self.u8()? {
            0 => ir::PlaceRoot::Local(ir::LocalId(self.u32()?)),
            1 => ir::PlaceRoot::Param(self.u32()?),
            2 => ir::PlaceRoot::Resource(ir::ResourceId(self.u32()?)),
            3 => ir::PlaceRoot::Ptr(ir::ValueId(self.u32()?)),
            4 => ir::PlaceRoot::Data(ir::DataId(self.u32()?)),
            _ => return None,
        };
        let path = self.vec(|r| {
            Some(match r.u8()? {
                0 => ir::Proj::Field(r.u32()?),
                1 => ir::Proj::Comp(r.u8()?),
                2 => ir::Proj::Index(ir::ValueId(r.u32()?)),
                _ => return None,
            })
        })?;
        Some(ir::Place { root, path })
    }
    fn recipe_expr(&mut self) -> Option<ir::Expr> {
        Some(match self.u8()? {
            0 => ir::Expr::Const(self.const_()?),
            1 => ir::Expr::Zero(ir::TypeId(self.u32()?)),
            2 => ir::Expr::Load(self.place()?),
            3 => {
                let op = match self.u8()? {
                    0 => ir::UnOp::Neg,
                    1 => ir::UnOp::Not,
                    _ => return None,
                };
                ir::Expr::Unary(op, ir::ValueId(self.u32()?))
            }
            4 => ir::Expr::Binary(
                binop_from(self.u8()?)?,
                ir::ValueId(self.u32()?),
                ir::ValueId(self.u32()?),
            ),
            5 => ir::Expr::Builtin(
                builtin_from(self.u8()?)?,
                self.vec(|r| Some(ir::ValueId(r.u32()?)))?,
            ),
            6 => ir::Expr::Construct(
                ir::TypeId(self.u32()?),
                self.vec(|r| Some(ir::ValueId(r.u32()?)))?,
            ),
            7 => ir::Expr::Variant(
                ir::TypeId(self.u32()?),
                self.u32()?,
                self.opt(|r| Some(ir::ValueId(r.u32()?)))?,
            ),
            8 => ir::Expr::Extract(ir::ValueId(self.u32()?), self.u32()?),
            9 => ir::Expr::ExtractDyn(ir::ValueId(self.u32()?), ir::ValueId(self.u32()?)),
            10 => ir::Expr::Splat(ir::ValueId(self.u32()?), self.u8()?),
            11 => ir::Expr::Swizzle(ir::ValueId(self.u32()?), self.bytes()?),
            12 => ir::Expr::Convert(ir::ValueId(self.u32()?), scalar_from(self.u8()?)?),
            13 => ir::Expr::Bitcast(ir::ValueId(self.u32()?), scalar_from(self.u8()?)?),
            14 => ir::Expr::Select {
                cond: ir::ValueId(self.u32()?),
                if_true: ir::ValueId(self.u32()?),
                if_false: ir::ValueId(self.u32()?),
            },
            15 => ir::Expr::Param(self.u32()?),
            _ => return None,
        })
    }
    fn recipe(&mut self) -> Option<ir::Function> {
        let name = self.str()?;
        let params = self.vec(|r| {
            Some(ir::Param {
                name: r.str()?,
                ty: ir::TypeId(r.u32()?),
                by_ref: r.bool()?,
                mutable: r.bool()?,
            })
        })?;
        let ret = self.opt(|r| Some(ir::TypeId(r.u32()?)))?;
        let values = self.vec(|r| Some(ir::TypeId(r.u32()?)))?;
        let body = self.vec(|r| {
            Some(match r.u8()? {
                0 => ir::Stmt::Let(ir::ValueId(r.u32()?), r.recipe_expr()?),
                1 => ir::Stmt::Return(r.opt(|r| Some(ir::ValueId(r.u32()?)))?),
                2 => ir::Stmt::At(r.span()?),
                _ => return None,
            })
        })?;
        Some(ir::Function { name, params, ret, values, body, ..ir::Function::default() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_cached_pipeline_preserves_float_bits_and_hoisted_recipes() {
        let mut types = ir::Types::default();
        let f32_ty = types.f32();
        let u32_ty = types.u32();
        let vec4_ty = types.vector(4);
        let arr_ty = types.intern(ir::TypeDef::Array(vec4_ty, 4));
        let mut recipe =
            ir::Function::new("hoist", vec![ir::Param::value("k", u32_ty)], Some(f32_ty));
        let mut body = Vec::new();
        let neg_zero = recipe.let_(&mut body, f32_ty, ir::Expr::Const(ir::Const::F32(-0.0)));
        body.push(ir::Stmt::Return(Some(neg_zero)));
        recipe.body = body;
        let cached = CachedPipeline {
            uniform: Some(UniformOut { binding: 0, size: 64, storage: false }),
            bindings: vec![ResourceBinding {
                binding: 1,
                kind: BindingKind::Read,
                stage: BindingStage::Both,
                format: None,
            }],
            work: vec![Some(1234), None],
            unbounded: vec![None, Some(Span { file: FileId(2), start: 10, end: 25 })],
            weight: 999,
            same: vec![ir::same::SameWork {
                function: "f".into(),
                entry: "main".into(),
                path: vec!["a".into()],
                at: Some(Span { file: FileId(1), start: 3, end: 9 }),
                work: Some(300),
            }],
            costly: vec![ir::rates::Costly {
                at: Span { file: FileId(1), start: 20, end: 40 },
                why: ir::rates::Why::Heavy,
                passes: Some(8),
                work: Some(2000),
                each: "pixel",
                callee: Some("noise".into()),
                times: 4,
                of: Some(4000),
            }],
            copies: vec![ir::opt::Copied {
                function: "g".into(),
                at: None,
                bytes: 512,
                what: "Big".into(),
                uniform: true,
            }],
            hoisted: vec![ir::hoist::Hoisted { field: 3, ty: arr_ty, recipe, count: Some((0, 4)) }],
            hoisted_types: Some(types),
            uncounted_most: Some(50000),
            shader: PreEmittedShader {
                bytes: b"metal shader bytes".to_vec(),
                entry_points: vec!["vs_e".into(), "fs_e".into()],
                requires: vec!["shader-f16".into()],
                kept: 256,
                sizes: Some(BufferSizes { binding: 2, buffers: vec![1] }),
            },
        };
        let key = "0".repeat(KEY_BYTES);
        let bytes = encode(&key, &cached).expect("encodes");
        let got = decode(&key, &bytes).expect("decodes");
        assert_eq!(got.uniform, cached.uniform);
        assert_eq!(got.bindings, cached.bindings);
        assert_eq!(got.work, cached.work);
        assert_eq!(got.unbounded, cached.unbounded);
        assert_eq!(got.weight, cached.weight);
        assert_eq!(got.same, cached.same);
        assert_eq!(got.costly, cached.costly);
        assert_eq!(got.copies, cached.copies);
        assert_eq!(got.uncounted_most, cached.uncounted_most);
        assert_eq!(got.shader.bytes, cached.shader.bytes);
        assert_eq!(got.shader.entry_points, cached.shader.entry_points);
        assert_eq!(got.shader.requires, cached.shader.requires);
        assert_eq!(got.shader.kept, cached.shader.kept);
        assert_eq!(got.shader.sizes, cached.shader.sizes);
        let ir::Stmt::Let(_, ir::Expr::Const(ir::Const::F32(x))) = &got.hoisted[0].recipe.body[0]
        else {
            panic!("expected f32 const");
        };
        assert_eq!(x.to_bits(), (-0.0f32).to_bits());
        assert!(decode(&"1".repeat(KEY_BYTES), &bytes).is_none());
    }

    #[test]
    fn fingerprint_distinguishes_signed_zero() {
        let mut m1 = ir::Module::default();
        let f32_ty = m1.types.f32();
        let mut f1 = ir::Function::new("f", vec![], Some(f32_ty));
        let mut b1 = Vec::new();
        let v1 = f1.let_(&mut b1, f32_ty, ir::Expr::Const(ir::Const::F32(0.0)));
        b1.push(ir::Stmt::Return(Some(v1)));
        f1.body = b1;
        m1.add_function(f1);

        let mut m2 = m1.clone();
        m2.functions[0].body[0] =
            ir::Stmt::Let(ir::ValueId(0), ir::Expr::Const(ir::Const::F32(-0.0)));

        let hash = |m: &ir::Module| {
            let mut h = Sha256::new();
            ir::fingerprint::feed_module(m, &mut |b| h.update(b));
            h.finalize()
        };
        assert_ne!(hash(&m1), hash(&m2));
    }

    #[test]
    fn roundtrip_work_count_entry() {
        let key = "a".repeat(KEY_BYTES);
        let bytes = encode_work(&key, 42_123).expect("encodes");
        assert_eq!(decode_work(&key, &bytes), Some(42_123));
        assert_eq!(decode_work(&"b".repeat(KEY_BYTES), &bytes), None);
    }
}
