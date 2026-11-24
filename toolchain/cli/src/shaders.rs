//! `wrela shaders <build-dir> [--jobs N]`: every shader of a build through its platform's own
//! offline tool, as the test suite checks every shader it builds (`wrela_shader::offline`):
//! each `.metal` compiled by Apple's `metal`, each `.spv` validated by `spirv-val`, N at a time
//! (the machine's cores unless said). A shader that passed before is passed from the checks'
//! cache (`WRELA_SHADER_CHECKS`).
//!
//! Exit status: 0 when every shader passes; 1 when one fails (each failure printed); 2 for a
//! usage error, a directory with no shaders, or a shader whose tool this machine hasn't.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use wrela_shader::offline::{self, Tool};

pub fn run(args: &[String]) -> ExitCode {
    let mut dir = None;
    let mut jobs = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--jobs" => match it.next().and_then(|n| n.parse().ok()).filter(|&n| n > 0) {
                Some(n) => jobs = n,
                None => {
                    eprintln!("error: `--jobs` takes a number of shaders checked at a time");
                    return crate::usage();
                }
            },
            _ if a.starts_with('-') || dir.is_some() => return crate::unknown(a),
            _ => dir = Some(PathBuf::from(a)),
        }
    }
    let Some(dir) = dir else { return crate::usage() };
    let entries = match std::fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(e) => {
            eprintln!("error: {}: {e}", dir.display());
            return ExitCode::from(2);
        }
    };
    // The build's shaders (its own files, not its directories'), by name.
    let mut shaders: Vec<(String, Tool)> = entries
        .filter_map(|e| {
            let name = e.ok()?.file_name().into_string().ok()?;
            let tool = Tool::for_file(&name)?;
            Some((name, tool))
        })
        .collect();
    shaders.sort_by(|a, b| a.0.cmp(&b.0));
    if shaders.is_empty() {
        eprintln!("error: {} has no shaders", dir.display());
        return ExitCode::from(2);
    }
    let mut missing: Vec<&str> =
        shaders.iter().filter(|(_, t)| !t.available()).map(|(_, t)| tool_name(*t)).collect();
    missing.sort();
    missing.dedup();
    if !missing.is_empty() {
        eprintln!("error: this machine hasn't {}", missing.join(", "));
        return ExitCode::from(2);
    }
    // Each distinct shader is checked once.
    let mut by_bytes: BTreeMap<(bool, Vec<u8>), Vec<usize>> = BTreeMap::new();
    for (i, (name, tool)) in shaders.iter().enumerate() {
        let bytes = match std::fs::read(dir.join(name)) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("error: {name}: {e}");
                return ExitCode::from(2);
            }
        };
        by_bytes.entry((*tool == Tool::Metal, bytes)).or_default().push(i);
    }
    let distinct: Vec<(&[u8], Tool, &[usize])> = by_bytes
        .iter()
        .map(|((_, bytes), at)| (bytes.as_slice(), shaders[at[0]].1, at.as_slice()))
        .collect();
    let said: Mutex<Vec<Option<String>>> = Mutex::new(vec![None; shaders.len()]);
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..jobs.min(distinct.len()) {
            s.spawn(|| {
                while let Some(&(bytes, tool, at)) =
                    distinct.get(next.fetch_add(1, Ordering::Relaxed))
                {
                    let failed = match offline::check(tool, bytes) {
                        Ok(true) => continue,
                        Ok(false) => format!("this machine hasn't {}", tool_name(tool)),
                        Err(e) => e,
                    };
                    let mut said = said.lock().unwrap_or_else(|p| p.into_inner());
                    for &i in at {
                        said[i] = Some(failed.clone());
                    }
                }
            });
        }
    });
    let said = said.into_inner().unwrap_or_else(|p| p.into_inner());
    let mut failed = 0;
    for ((name, _), why) in shaders.iter().zip(&said) {
        if let Some(why) = why {
            failed += 1;
            println!("FAILED {name}\n{why}");
        }
    }
    let mut largest_bytes = 0usize;
    let mut largest_name = "";
    for (name, _) in &shaders {
        if let Ok(meta) = std::fs::metadata(dir.join(name)) {
            let sz = meta.len() as usize;
            if sz > largest_bytes {
                largest_bytes = sz;
                largest_name = name.as_str();
            }
        }
    }
    println!(
        "shaders: {} of {} pass (largest: {} KiB `{}`)",
        shaders.len() - failed,
        shaders.len(),
        largest_bytes.div_ceil(1024),
        largest_name,
    );
    if failed > 0 { ExitCode::from(1) } else { ExitCode::SUCCESS }
}

/// The tool, as a person would look for it.
fn tool_name(tool: Tool) -> &'static str {
    match tool {
        Tool::Metal => "xcrun metal",
        Tool::SpirvVal => "spirv-val",
    }
}
