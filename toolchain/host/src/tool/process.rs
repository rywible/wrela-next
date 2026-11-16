//! The tool host's process helpers: the lock around starting a process, the sweep of what a
//! killed host left, and (Linux) the reaping of a program's process group.

use std::path::Path;
use std::sync::Mutex;

/// Held while this process makes a pipe or starts a process: the tool host's programs and driven
/// builds take it. On macOS a new pipe's ends become close-on-exec only a moment after they're
/// made (there's no `pipe2`), so a process started on another thread in that moment would keep
/// them open, and a process waiting for its pipe's end would wait on it. Taken around the making
/// of pipes and each start alone, never a wait.
pub(crate) fn spawning() -> std::sync::MutexGuard<'static, ()> {
    static SPAWNING: Mutex<()> = Mutex::new(());
    SPAWNING.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Whether process `pid` still runs (as anyone's).
fn alive(pid: i32) -> bool {
    rustix::process::Pid::from_raw(pid).is_some_and(|p| {
        matches!(rustix::process::test_kill_process(p), Ok(()) | Err(rustix::io::Errno::PERM))
    })
}

/// Removes each directory of `base` named `<prefix><pid>` or `<prefix><pid>-...` whose process
/// has died: what a killed process left.
pub(crate) fn sweep(base: &Path, prefix: &str) {
    let Ok(entries) = std::fs::read_dir(base) else { return };
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(rest) = name.strip_prefix(prefix) else { continue };
        let Ok(pid) = rest.split('-').next().unwrap_or("").parse::<i32>() else { continue };
        let ours = pid == std::process::id() as i32;
        // A link isn't followed: only a directory of a dead process's goes.
        let dir = e.file_type().is_ok_and(|t| t.is_dir());
        if !ours && dir && !alive(pid) {
            let _ = std::fs::remove_dir_all(e.path());
        }
    }
}

/// Reaps every exited process of group `group` that is this process's child (Linux): whether
/// none is left, exited or running. The tool host's `run` reaps its programs' groups with it.
#[cfg(target_os = "linux")]
pub(crate) fn reap_group(group: rustix::process::Pid) -> bool {
    loop {
        match rustix::process::waitpgid(group, rustix::process::WaitOptions::NOHANG) {
            // One reaped: look again.
            Ok(Some(_)) => {}
            // Some still run.
            Ok(None) => return false,
            // None of ours is in the group.
            Err(_) => return true,
        }
    }
}
