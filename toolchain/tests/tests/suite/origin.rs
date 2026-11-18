//! A dispatch with an origin (`gpu.origin`, toolchain/tests/origin): a 3D texture filled in twelve
//! uneven parts, each `over:` its size from its `origin:`, is the texture filled whole, to the
//! bit, and the CPU's; a row of `Slots` filled in three parts likewise. On both machines.
//! `cargo test -p wrela-tests --test suite origin:: -- --ignored`.

use wrela_host::{Host, Value};
use wrela_tests::{one_u32, page};

#[test]
#[ignore = "needs a GPU"]
fn a_volume_filled_in_parts_is_the_volume_filled_whole() {
    let (dir, _) = page("toolchain/tests/origin", "origin-native");
    let mut host = Host::load(&dir).expect("load");
    let times = crate::frame_times(4);
    host.run_frames(&times, 16, 16).expect("run");
    let (volume, row) = (20 * 13 * 9, 1000);
    for (k, n) in [(0, volume), (1, volume), (2, row), (3, row)] {
        assert_eq!(one_u32(&mut host, "arrived", &[Value::I32(k)]), n, "read-back {k} arrived");
    }
    assert_eq!(one_u32(&mut host, "matched", &[Value::I32(1)]), volume, "the volume's parts");
    assert_eq!(one_u32(&mut host, "matched", &[Value::I32(3)]), row, "the row's parts");
}
