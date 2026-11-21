//! The platforms' own offline tools, run on a shader the compiler wrote: Apple's `metal`
//! compiler on MSL (`xcrun metal -c -std=metal4.0 -Werror`), Khronos's `spirv-val` on SPIR-V
//! (`--target-env vulkan1.3`). A writer's output that either rejects is a compiler bug, and the
//! test suite runs every shader it builds through them ([`check`]), where the machine has them.
//!
//! Each shader is checked once: a passing one leaves a mark named by its bytes' hash in a cache
//! directory (`WRELA_SHADER_CHECKS`, else the system's temporary directory), so a test run
//! compiles only what changed.

use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::process::Command;

/// Which tool checks a shader, by its file's extension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tool {
    /// Apple's `metal`, through `xcrun`: MSL 4.0, warnings errors.
    Metal,
    /// `spirv-val` for Vulkan 1.3.
    SpirvVal,
}

impl Tool {
    /// The target whose shaders the tool checks.
    pub fn target(self) -> wrela_abi::Target {
        match self {
            Tool::Metal => wrela_abi::Target::Metal,
            Tool::SpirvVal => wrela_abi::Target::Vulkan,
        }
    }

    /// The tool for a shader file named `name`, if it's a shader ([`crate::extension`]).
    pub fn for_file(name: &str) -> Option<Tool> {
        let (_, ext) = name.rsplit_once('.')?;
        [Tool::Metal, Tool::SpirvVal].into_iter().find(|t| crate::extension(t.target()) == ext)
    }

    /// Whether this machine has the tool (asked once a process).
    pub fn available(self) -> bool {
        static METAL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static SPIRV_VAL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let (cell, probe): (_, fn() -> std::io::Result<std::process::Output>) = match self {
            Tool::Metal => (&METAL, || Command::new("xcrun").args(["-f", "metal"]).output()),
            Tool::SpirvVal => (&SPIRV_VAL, || Command::new("spirv-val").arg("--version").output()),
        };
        *cell.get_or_init(|| probe().is_ok_and(|o| o.status.success()))
    }
}

fn cache_dir() -> PathBuf {
    std::env::var_os("WRELA_SHADER_CHECKS")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("wrela-shader-checks"))
}

/// Checks `bytes` (a shader for `tool`) with the tool: `Ok(true)` it passed, `Ok(false)` the
/// machine hasn't the tool, `Err` what the tool said.
pub fn check(tool: Tool, bytes: &[u8]) -> Result<bool, String> {
    let hash = Sha256::digest(bytes);
    let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
    let dir = cache_dir();
    let mark = dir.join(format!("{hex}.ok"));
    if mark.exists() {
        return Ok(true);
    }
    if !tool.available() {
        return Ok(false);
    }
    let _ = std::fs::create_dir_all(&dir);
    let ext = crate::extension(tool.target());
    let file = dir.join(format!("{hex}-{}.{ext}", std::process::id()));
    std::fs::write(&file, bytes).map_err(|e| format!("{}: {e}", file.display()))?;
    let out = match tool {
        Tool::Metal => Command::new("xcrun")
            .args(["metal", "-c", "-std=metal4.0", "-Werror", "-o"])
            .arg(file.with_extension("air"))
            .arg(&file)
            .output(),
        Tool::SpirvVal => {
            Command::new("spirv-val").args(["--target-env", "vulkan1.3"]).arg(&file).output()
        }
    };
    let _ = std::fs::remove_file(&file);
    let _ = std::fs::remove_file(file.with_extension("air"));
    let out = out.map_err(|e| format!("running {tool:?}: {e}"))?;
    if !out.status.success() {
        let mut said = String::from_utf8_lossy(&out.stderr).into_owned();
        said.push_str(&String::from_utf8_lossy(&out.stdout));
        return Err(said);
    }
    let _ = std::fs::write(&mark, b"");
    Ok(true)
}

/// Checks every shader file among `files` (name, bytes): the first failure, naming its file.
pub fn check_all<'a>(files: impl IntoIterator<Item = (&'a str, &'a [u8])>) -> Result<(), String> {
    for (name, bytes) in files {
        if let Some(tool) = Tool::for_file(name) {
            check(tool, bytes).map_err(|e| format!("{name} fails {tool:?}:\n{e}"))?;
        }
    }
    Ok(())
}
