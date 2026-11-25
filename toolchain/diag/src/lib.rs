//! Diagnostics for the wrela compiler: a registry of codes, source files and spans, and
//! rendering for people (`render`) and for tools (`json`, D-021).

pub mod codes;
mod diagnostic;
pub mod json;
pub mod render;
mod source;

pub use codes::Code;
pub use diagnostic::{
    Diagnostic, Edit, Fix, Label, Severity, apply_edits, has_errors, sort_and_dedup,
};
pub use source::{FileId, LineCol, SourceFile, SourceMap, Span};

/// The checkout this program runs from, if it runs from one: the repository whose own
/// `toolchain/target` holds the running binary (a tool's, a test's), found by its `lib/std`.
/// Such a compiler reads std's sources and the spec from the checkout each time it starts, so an
/// edit to either needs no build of the compiler (tools/built). A compiler anywhere else has
/// none, and uses the text it was built with. `WRELA_CHECKOUT` names the checkout instead, or,
/// set to nothing, says there is none.
pub fn checkout() -> Option<&'static std::path::Path> {
    use std::path::PathBuf;
    static ROOT: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    ROOT.get_or_init(|| {
        if let Some(named) = std::env::var_os("WRELA_CHECKOUT") {
            return (!named.is_empty()).then(|| PathBuf::from(named));
        }
        let exe = std::env::current_exe().ok()?.canonicalize().ok()?;
        exe.ancestors().find_map(|dir| {
            let toolchain = dir.parent()?;
            if dir.file_name()? != "target" || toolchain.file_name()? != "toolchain" {
                return None;
            }
            let root = toolchain.parent()?;
            root.join("lib/std/prelude.wrela").is_file().then(|| root.to_path_buf())
        })
    })
    .as_deref()
}

/// The plural ending of a noun counting `n` things: `"s"`, or `""` when `n` is 1.
pub fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}
