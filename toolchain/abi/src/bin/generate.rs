//! Writes lib/std/abi.wrela from the ABI's definitions, and toolchain/abi/vectors.json, the
//! test vectors.

fn main() -> std::io::Result<()> {
    for (path, text) in wrela_abi::generated_files() {
        std::fs::write(&path, text)?;
        println!("wrote {}", path.display());
    }
    Ok(())
}
