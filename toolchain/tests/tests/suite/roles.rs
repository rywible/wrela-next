//! Roles (language.md §15.3): a program that exports `frame` has a client build, one that
//! exports `serve` a server build, one that exports `tool` a tool build, and one with several
//! has each, lowered from its own entries; the client and server modules share one pair id.
//! What only one role may run, reached from another role's entries, is E0708 with the call
//! chain.

use std::path::{Path, PathBuf};

/// A package in a scratch directory of its own, with `main.wrela` holding `main`.
fn package(name: &str, main: &str) -> PathBuf {
    let dir = super::scratch(&format!("roles-{name}"));
    super::write_package(&dir, &name.replace('-', "_"), main);
    dir
}

/// The built files, by path.
fn files(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let built = wrela_driver::build(dir);
    assert!(
        !built.has_errors(),
        "{}",
        wrela_diag::render::render_all(&built.sources, &built.diagnostics)
    );
    built.files
}

fn file<'a>(files: &'a [(String, Vec<u8>)], path: &str) -> &'a [u8] {
    &files.iter().find(|(p, _)| p == path).unwrap_or_else(|| panic!("no {path}")).1
}

/// The module's exports, and whether it imports its memory shared.
fn exports_and_memory(wasm: &[u8]) -> (Vec<String>, bool) {
    let engine =
        wasmtime::Engine::new(wasmtime::Config::new().wasm_threads(true).shared_memory(true))
            .expect("an engine");
    let module = wasmtime::Module::new(&engine, wasm).expect("a module");
    let exports = module.exports().map(|e| e.name().to_string()).collect();
    let shared = module
        .imports()
        .any(|i| matches!(i.ty(), wasmtime::ExternType::Memory(m) if m.is_shared()));
    (exports, shared)
}

/// The rendered errors of checking `main`.
fn errors(name: &str, main: &str) -> (Vec<String>, String) {
    super::checked(&package(name, main))
}

const BOTH: &str = r#"
struct Room: Clone {
    ticks: u32,
}

/// Code both builds run.
fn shared<T: Copy>(x: T) -> T {
    x
}

pub fn init() -> u32 {
    shared(1)
}

pub fn frame(state: mut u32, time: f32, width: u32, height: u32) {
    state = shared(state) + 1
}

pub fn serve_init() -> Room {
    Room { ticks: shared(0) }
}

pub fn serve(room: mut Room, time: f32) {
    room.ticks = shared(room.ticks) + 1
}
"#;

#[test]
fn a_program_with_both_roles_builds_a_module_for_each() {
    let files = files(&package("both", BOTH));
    let client = file(&files, "game.wasm");
    let server = file(&files, "server/game.wasm");
    // Each lowered from its own entries: the client's exports aren't the server's.
    let (client_exports, client_shared) = exports_and_memory(client);
    let (server_exports, server_shared) = exports_and_memory(server);
    assert!(client_exports.contains(&"frame".into()) && client_exports.contains(&"init".into()));
    assert!(!client_exports.iter().any(|e| e.starts_with("serve")), "{client_exports:?}");
    assert!(
        server_exports.contains(&"serve".into()) && server_exports.contains(&"serve_init".into())
    );
    assert!(!server_exports.iter().any(|e| e == "frame" || e == "init"), "{server_exports:?}");
    // The client's memory is shared with its workers; the server's is its own.
    assert!(client_shared && !server_shared);
    // One pair id in both: the hash of both modules without it.
    let pair = wrela_abi::read_pair(client).expect("the client's pair id");
    assert_eq!(wrela_abi::read_pair(server), Some(pair));
    assert!(files.iter().any(|(p, _)| p == "manifest.json" || p == "index.html"));
}

#[test]
fn a_server_only_program_builds_no_client() {
    let main = "pub fn serve(time: f32) {}\n";
    let files = files(&package("server-only", main));
    let paths: Vec<&str> = files.iter().map(|(p, _)| p.as_str()).collect();
    assert_eq!(paths, ["server/game.wasm"]);
    assert!(wrela_abi::read_pair(&files[0].1).is_some());
}

#[test]
fn a_client_only_program_builds_as_before() {
    let main = "pub fn frame(time: f32, width: u32, height: u32) {}\n";
    let files = files(&package("client-only", main));
    assert!(!files.iter().any(|(p, _)| p.starts_with("server/")));
    assert!(wrela_abi::read_pair(file(&files, "game.wasm")).is_some());
}

#[test]
fn a_server_function_reached_from_the_client_is_e0708_with_its_chain() {
    let main = r#"
@server
fn keep(x: u32) -> u32 {
    x
}

fn helper(x: u32) -> u32 {
    keep(x)
}

pub fn frame(time: f32, width: u32, height: u32) {
    let _ = helper(width)
}

pub fn serve(time: f32) {
    let _ = keep(3)
}
"#;
    let (codes, text) = errors("server-from-client", main);
    assert_eq!(codes, ["E0708"], "{text}");
    assert!(text.contains("runs only in a server build"), "{text}");
    assert!(text.contains("`frame` → `helper`"), "{text}");
}

/// An associated constant only the server's build may read is E0708 read from the client's,
/// though its value would fold (it stays a call, which checks the role), and through another
/// constant that reads it.
#[test]
fn a_server_associated_constant_read_from_the_client_is_e0708() {
    let main = r#"
struct Keys {}

impl Keys {
    @server
    const SECRET: u32 = 4
    const TWICE: u32 = Keys::SECRET * 2
}

pub fn frame(time: f32, width: u32, height: u32) {
    let _ = Keys::SECRET + width
    let _ = Keys::TWICE + width
}

pub fn serve(time: f32) {
    let _ = Keys::SECRET
}
"#;
    let (codes, text) = errors("server-constant-from-client", main);
    assert_eq!(codes, ["E0708", "E0708"], "{text}");
}

#[test]
fn client_code_reached_from_the_server_is_e0708_through_generics_and_closures() {
    let main = r#"
@client
fn show(x: u32) -> u32 {
    x
}

fn apply<F: fn(u32) -> u32>(f: F, x: u32) -> u32 {
    f(x)
}

pub fn frame(time: f32, width: u32, height: u32) {
    let _ = show(1)
}

pub fn serve(time: f32) {
    let _ = apply(|x| show(x), 2)
}
"#;
    let (codes, text) = errors("client-from-server", main);
    assert_eq!(codes, ["E0708"], "{text}");
    assert!(text.contains("runs only in a client build"), "{text}");
    assert!(text.contains("`serve` → `apply`"), "{text}");
}

#[test]
fn std_client_functions_and_gpu_work_are_e0708_in_a_server() {
    let main = r#"
use std::gpu::{GlobalId, Slots, buffer, dispatch}
use std::tick::{Ticked, start}

@compute(64, 1, 1)
fn fill(id: GlobalId, out: mut Slots<u32>) {
    out[id] = 1
}

@deterministic
fn step(s: mut u32, t: Ticked) {
    s += 1
}

@deterministic
fn hash(s: u32) -> u64 {
    u64(s)
}

fn ticking() {
    start(0, hz: 30, step: step, hash: hash)
}

pub fn serve(time: f32) {
    var out = buffer::<u32>(64)
    dispatch(fill.bind(mut out), over: 64)
    ticking()
}
"#;
    let (codes, text) = errors("std-from-server", main);
    assert_eq!(codes, ["E0708", "E0708"], "{text}");
    assert!(text.contains("GPU work in the server build's code"), "{text}");
    assert!(text.contains("`start` runs only in a client build"), "{text}");
    assert!(text.contains("`serve` → `ticking`"), "{text}");
}

#[test]
fn the_server_entries_signatures_are_checked() {
    let main = r#"
pub fn serve_init(seed: u32) -> u32 {
    seed
}

pub fn serve(state: mut u32, time: f32, extra: u32) {}
"#;
    let (codes, text) = errors("server-signatures", main);
    assert_eq!(codes, ["E0703", "E0703"], "{text}");
    assert!(text.contains("`serve_init`'s signature is `serve_init() -> State`"), "{text}");
    assert!(text.contains("`serve`'s signature is `serve(time: f32)`"), "{text}");
}

#[test]
fn a_function_has_one_role() {
    let main = r#"
@client
@server
fn both() {}

@server
@tool
fn also() {}

pub fn frame(time: f32, width: u32, height: u32) {}
"#;
    let (codes, text) = errors("both-roles", main);
    assert_eq!(codes, ["E0204", "E0204"], "{text}");
    assert!(text.contains("it's `@client` already, so not `@server`"), "{text}");
    assert!(text.contains("it's `@server` already, so not `@tool`"), "{text}");
}

// ---- the tool role ----------------------------------------------------------------------------

/// A program with all three roles: what each build runs, and what only one role may.
const THREE: &str = r#"
/// Code every build runs.
fn shared(x: u32) -> u32 {
    x + 1
}

@tool
fn tool_only(x: u32) -> u32 {
    shared(x)
}

pub fn frame(time: f32, width: u32, height: u32) {
    let _ = shared(width)
}

pub fn serve(time: f32) {
    let _ = shared(2)
}

pub fn tool() -> u32 {
    shared(tool_only(0))
}
"#;

/// The maximum pages of the memory a module defines, if it defines one.
fn memory_max(wasm: &[u8]) -> Option<u64> {
    let engine = wasmtime::Engine::new(wasmtime::Config::new().wasm_threads(true)).expect("engine");
    let module = wasmtime::Module::new(&engine, wasm).expect("a module");
    module.exports().find_map(|e| match e.ty() {
        wasmtime::ExternType::Memory(m) => m.maximum(),
        _ => None,
    })
}

#[test]
fn a_tool_build_has_only_the_tools_code() {
    let files = files(&package("three", THREE));
    let tool = file(&files, "tool/tool.wasm");
    let (tool_exports, tool_shared) = exports_and_memory(tool);
    let mut sorted = tool_exports.clone();
    sorted.sort();
    assert_eq!(sorted, ["memory", "tool"], "{tool_exports:?}");
    // Its memory shared with its helpers (std::par), with room to grow to 4 GiB less a page.
    assert!(tool_shared);
    assert_eq!(memory_max(tool), Some(65535));
    // The other builds have none of the tool's exports, and the tool isn't paired with them.
    let (client_exports, _) = exports_and_memory(file(&files, "game.wasm"));
    let (server_exports, _) = exports_and_memory(file(&files, "server/game.wasm"));
    for exports in [&client_exports, &server_exports] {
        assert!(!exports.iter().any(|e| e.starts_with("tool")), "{exports:?}");
    }
    assert!(wrela_abi::read_pair(tool).is_none());
    assert_eq!(
        wrela_abi::read_pair(file(&files, "game.wasm")),
        wrela_abi::read_pair(file(&files, "server/game.wasm"))
    );
}

#[test]
fn a_tool_only_program_builds_only_the_tool() {
    let main = "pub fn tool() -> u32 {\n    0\n}\n";
    let files = files(&package("tool-only", main));
    let paths: Vec<&str> = files.iter().map(|(p, _)| p.as_str()).collect();
    assert_eq!(paths, ["tool/tool.wasm"]);
}

#[test]
fn tool_code_reached_from_a_client_or_a_server_is_e0708_with_its_chain() {
    let main = r#"
use std::tool::files

@tool
fn keep(x: u32) -> u32 {
    x
}

fn helper(x: u32) -> u32 {
    keep(x)
}

pub fn frame(time: f32, width: u32, height: u32) {
    let _ = helper(width)
}

pub fn serve(time: f32) {
    let _ = files::read("out/a.txt")
}

pub fn tool() -> u32 {
    keep(0)
}
"#;
    let (codes, text) = errors("tool-from-others", main);
    assert_eq!(codes, ["E0708", "E0708"], "{text}");
    assert!(text.contains("`keep` runs only in a tool build, and this is the client build's code"));
    assert!(text.contains("`frame` → `helper`"), "{text}");
    assert!(text.contains("`read` runs only in a tool build, and this is the server build's"));
}

#[test]
fn client_server_and_gpu_code_reached_from_a_tool_is_e0708() {
    let main = r#"
use std::gpu::{GlobalId, Slots, buffer, dispatch}
use std::host

@compute(64, 1, 1)
fn fill(id: GlobalId, out: mut Slots<u32>) {
    out[id] = 1
}

@server
fn persist(x: u32) -> u32 {
    x
}

@client
fn show(x: u32) -> u32 {
    x
}

fn work() -> u32 {
    var out = buffer::<u32>(64)
    dispatch(fill.bind(mut out), over: 64)
    persist(1) + show(2)
}

pub fn tool() -> u32 {
    let _ = host::copy("x")
    work()
}
"#;
    let (codes, text) = errors("others-from-tool", main);
    assert_eq!(codes, ["E0708", "E0708", "E0708", "E0708"], "{text}");
    assert!(text.contains("GPU work in the tool build's code: a tool has no GPU"), "{text}");
    assert!(text.contains("`persist` runs only in a server build, and this is the tool build's"));
    assert!(text.contains("`show` runs only in a client build, and this is the tool build's"));
    assert!(text.contains("`copy` runs only in a client build"), "{text}");
    assert!(text.contains("`tool` → `work`"), "{text}");
}

#[test]
fn the_tool_entrys_signature_is_checked() {
    for (name, main) in [
        ("tool-signature-params", "pub fn tool(state: mut u32, extra: u32) -> u32 {\n    0\n}\n"),
        ("tool-signature-result", "pub fn tool() -> f32 {\n    0.0\n}\n"),
    ] {
        let (codes, text) = errors(name, main);
        assert_eq!(codes, ["E0703"], "{text}");
        assert!(text.contains("`tool`'s signature is `tool() -> u32`"), "{text}");
    }
}

#[test]
fn a_client_or_a_server_that_drops_tool_tasks_is_e0708() {
    // A tool task's drop waits for it in the tool host: no other role's build may reach it,
    // also through a container's destructor.
    let client = r#"
pub struct State {
    tasks: Vec<std::tool::Task<u32>>,
}

pub fn init() -> State {
    State { tasks: Vec::new() }
}

pub fn frame(state: mut State, time: f32, width: u32, height: u32) {
    state.tasks.clear()
}
"#;
    let server = r#"
pub struct Room {
    tasks: Vec<std::tool::Task<u32>>,
}

pub fn serve_init() -> Room {
    Room { tasks: Vec::new() }
}

pub fn serve(room: mut Room, time: f32) {
    room.tasks.clear()
}
"#;
    for (name, main, role) in
        [("client-drops-tasks", client, "client"), ("server-drops-tasks", server, "server")]
    {
        let (codes, text) = errors(name, main);
        assert!(!codes.is_empty() && codes.iter().all(|c| c == "E0708"), "{text}");
        assert!(
            text.contains(&format!("runs only in a tool build, and this is the {role} build's"))
        );
    }
}

#[test]
fn dropping_what_holds_no_tool_task_stays_deterministic() {
    // A type argument that names a task, where nothing holds one: a label's, a handle's.
    let main = r#"
use std::arena::Handle

struct Label<T> {
    name: String,
}

struct Holder {
    at: Handle<std::tool::Task<u32>>,
}

@deterministic
fn drop_label(l: take Label<std::tool::Task<u32>>) -> u32 {
    let _ = take l
    0
}

@deterministic
fn drop_holder(h: take Holder) -> u32 {
    let _ = take h
    0
}

pub fn tool() -> u32 {
    drop_label(Label { name: String::from("a") })
}
"#;
    let (codes, text) = errors("drops-no-tool-task", main);
    assert!(!codes.contains(&"E0600".to_string()), "{text}");
    // A `Vec` of tasks holds them: dropping it waits for each.
    let main = "@deterministic\nfn drop_all(tasks: take Vec<std::tool::Task<u32>>) -> u32 {\n    let _ = take tasks\n    0\n}\n\npub fn tool() -> u32 {\n    0\n}\n";
    let (codes, text) = errors("drops-tool-tasks", main);
    assert!(codes.contains(&"E0600".to_string()), "{text}");
}

#[test]
fn joining_or_dropping_a_tool_task_isnt_deterministic() {
    for (name, body) in [
        ("joins-a-tool-task", "take task.join()"),
        (
            "drops-a-tool-task",
            "let _ = take task
    0",
        ),
    ] {
        let main = format!(
            "@deterministic\nfn finish(task: take std::tool::Task<u32>) -> u32 {{\n    {body}\n}}\n\npub fn tool() -> u32 {{\n    0\n}}\n"
        );
        let (codes, text) = errors(name, &main);
        assert!(codes.contains(&"E0600".to_string()), "{name}: {text}");
    }
}
