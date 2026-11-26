# Plan: the engine makes its decisions at build

Status: in progress. Phase 0 is done on the M4 (its Radeon half is open). The plan agreed with
the code at commit `1c78609` (2026-11-25) when it was written; each phase's state is at its
"Done when".

## 1. Purpose

Today the engine makes many decisions while the game runs. This plan moves those decisions to the
build.

- The engine stays in wrela, in `lib/engine`.
- The game stays a set of fields.
- The compiler gives facts about each field.
- At build, the engine uses the facts to write the code and the data for each kind.
- The released game contains only that code and that data.

The plan has a second aim. The proofs about GPU code must be true for the released build. Today
they are fully true only for a build with `--exact`.

## 2. Read this first

1. Read `AGENTS.md`. Its rules apply to all work in this plan.
2. Read sections 3 to 6 of this plan. They give the terms, the places in the code, the commands
   and the rules.
3. Do the phases in the order of section 7.
4. In a phase, do the steps in their order. Each step is one change or a small number of changes.
5. A phase is done only when each line of its "Done when" list is true.
6. When you learn something that makes a step wrong, correct the step in this file in the same
   change.

The plan gives the source of each number: a commit, a tool, a file or the design notes. The
design notes are not in the repository now. To read them, use
`git show ca13c8e:docs/design.md`. A number from the design notes is older than the code.
Measure it again before you use it.

## 3. Terms

Each term has one meaning in this plan.

| Term | Meaning |
| --- | --- |
| Kind | One type of thing in the world, for example a pillar or a creature. A kind is one field |
| Part | One member of a field that is a union of fields |
| Eye | The camera |
| Kernel | One program that the GPU runs |
| Pass | One draw or one dispatch that has a label. A frame is a list of passes |
| Extraction | The work that makes a mesh from a field |
| Batch | The mesh of one kind and its buffers (`MeshBatch`) |
| Settled | A batch is settled when its mesh is complete and the engine knows that it is complete (`MeshBatch::settled`) |
| Exact cell | A cell where the mesh is not sufficient. The engine draws the box of the cell and finds the surface of the field for each pixel |
| Certificate | The data that shows that a mesh agrees with its field |
| Bound | A limit that the compiler proves for a value, for all inputs in a box |
| Steepness | The maximum rate at which the distance of a field changes for each metre (the Lipschitz constant) |
| Facts | The properties of a kind that the build proves (`FieldFacts`): its steepness, the channels that it can use, the work of its functions |
| Tape | The list of the operations of a field, which `std::derive::tape` gives |
| Work count | The maximum work of one invocation of a kernel, which the compiler counts. It is not a time |
| Span | The cost of one frame: the time on the clock of the GPU from the start of the first pass to the end of the last pass |
| Strict kernel | A kernel that the back end compiles with no contraction and with no fast math |
| Geometry kernel | A kernel whose output is a mesh, a certificate or a decision about where matter is. A kernel whose output is only a colour is not a geometry kernel |
| Cook | The data that the engine calculates one time for a kind at build and then keeps |
| Change class | The fastest rate at which one part of a field can change |
| Proof debt | The work that a pass does only because a certificate or a fact is missing |
| Target | A measured result that a phase must get before the phase is done |
| Gate | `tools/gate`: the layers, the tests, the hole audit, the judge of the sun and the span of four scenes |
| Hole audit | `tools/audit`: it compares the picture of each kind with its field and counts holes |
| Judge, reference | `lib/engine/judge` and `lib/engine/reference`: slow code that finds the correct result from the field |

## 4. Where the code is

| Subject | Place |
| --- | --- |
| Shader back end, exact mode | `toolchain/shader/src/lib.rs`, `metal.rs`, `spirv.rs` |
| Bounds for GPU code | `toolchain/ir/src/derive/interval.rs`, `affine.rs` |
| Work count | `toolchain/ir/src/work.rs` |
| Dispatch guard (the 50 ms rule) | `toolchain/host/src/step.rs` |
| What a program can ask about a field | `lib/std/derive.wrela` |
| `FieldFacts` | `lib/std/field.wrela` |
| The build proves the facts | `lib/engine/matter/verify.wrela` (`prove_field_facts`) |
| Certificates of cells | `lib/engine/matter/certify.wrela` |
| Extraction: control, batches | `lib/engine/mesh/extract.wrela` |
| Extraction: grids for the eye | `lib/engine/mesh/grid.wrela` |
| Extraction: kernels | `lib/engine/mesh/dc.wrela`, `emit.wrela` |
| Exact cells, proofs for a discard | `lib/engine/mesh/proof.wrela`, `shade.wrela` |
| Exact rays of the sun | `lib/engine/mesh/sunlight.wrela` |
| The passes of a frame | `lib/engine/mesh/draw.wrela`, `world_draw.wrela` |
| Certificate report, facts report | `game/world/certificates.wrela` |
| The scenes | `game/scenes/main.wrela` (`SCENE`, `VIEW`) |
| The gate and its lines | `tools/gate`, `tools/gate.json` |

Parts of the specification (`docs/spec/language.md`) to read:

| Section | Subject | Read before |
| --- | --- | --- |
| §11.3 | What each target does with floats, exact mode | Phases 0 and 1 |
| §12 | `@deterministic` | Phase 1 |
| §13.1, §13.3 | Constants at build, fuel, `embed` | Phases 3 and 4 |
| §14 | Bounds, steepness, known limits of the intervals | Phases 2 and 5 |
| §18.9, §18.11 | Pipelines, work and weight | Phase 4 |

## 5. How to measure

Use these commands. Do not use a different measure of a frame.

| To get | Command |
| --- | --- |
| A build of one scene | `tools/wrela build game/scenes -o DIR --set SCENE=10` |
| A first run of new GPU work | `tools/gpu DIR --size 320x180 --frames 2 --timestamps --serial` |
| The span of a scene, and the time of each pass | `tools/bench DIR --size 1920x1080 --frames 30-89` |
| Two builds compared | `tools/bench DIR --size 1920x1080 --frames 30-89 --against DIR2` |
| The spans of the gate only | `tools/gate --only frames` |
| The full gate | `tools/gate` (add `--full` for a change to extraction or to the passes of the eye) |
| The hole audit, small | `tools/audit --size 160x90 --eyes 12` |
| The certificate report and the facts report | `tools/wrela test game/world` |
| Each kernel and the code that dispatches it | `tools/wrela pipelines game/scenes --set SCENE=10` |
| Each constant that the build calculated | `tools/wrela build game/scenes --report consts` |
| The same command on the Radeon | `tools/radeon COMMAND` (commit and push first) |

- The span is the median of frames 30 to 89 at 1920x1080. The gate uses the same numbers, as
  their mean over three placements of the world's lattice of cells (`PLACE`), and the sun's
  judge over four: where the cells fall moves a span or a count past its slack on its own (cells
  1% larger: SCENE 13 VIEW 0's hard leaks 754 to 882, SCENE 13's span 18.20 to 19.30 ms; the
  lattice moved a few mm: 677 to 816, 19.2 to 20.5 ms), so a change to the mesh is held to a mean
  of that draw, not to one draw of it.
- A paired `tools/bench --against` of two builds of one placement measures a change that keeps
  the mesh (a pass, the host); a change that moves the mesh is measured over placements.
- On Metal a timed span times a label's dispatches in a row as one run (2026-11-26): an encoder
  stamped for each dispatch had cost the frame what the game's frame doesn't pay. SCENE 10's span
  24.5 to 21.0 ms against 20.5 ms a frame untimed (steady throughput, 600 frames less 200); the
  gate's lines were written again: SCENE 10/11/13/14 20.87/16.52/18.97/21.95 ms.
- `tools/bench` also runs each pass alone and prints a table. The times in the table do not add
  up to the span. Use the table to compare one pass before and after a change.
- Put each measurement in the commit message: before and after, same machine, same run.

## 6. Rules for all phases

1. Keep the engine in `lib/engine`. Do not move engine code into `toolchain/`.
2. Add only general mechanisms to the toolchain. A mechanism is general if it knows fields, but
   knows no pass of the engine and no thing of the game.
3. Keep the judges and the reference independent. They must ask the field of the author
   directly. They must not read the cook. They must not use a decision of the build.
4. Make each build decision from a count or from a proof. Do not make a decision from a measured
   time. A decision from a time changed with the state of the GPU (design notes, "What is
   measured, and what is not").
5. When a new method replaces an old method, remove the old method in the same change.
6. Work on each phase until each line of its "Done when" list is true. If a result is not good,
   find the cause and use a different method. Do not stop, and do not leave the problem open.
7. Measure on the two machines. Pass the gate on the two machines before you call a phase done.

## 7. Order of the phases

| Phase | Result | Uses the result of |
| --- | --- | --- |
| 0 | Three measurements that show where the time is | Nothing |
| 1 | Strict math in the geometry kernels | Phase 0 |
| 2 | The compiler gives the change class of each part of a field | Nothing |
| 3 | The build makes the cook on the GPU and keeps it | Phases 0, 1 and 2 |
| 4 | The engine makes its decisions and the frame schedule at build | Phases 2 and 3 |
| 5 | Proofs remove work, and each frame is in the bar | Phases 1 and 4 |

Reasons for this order:

- Phase 1 is before phase 3 because the cook keeps certificates. A certificate is only as true
  as the math below it.
- Phase 2 is before phase 3 because the cook contains only the parts that do not change.
- Phase 4 is after phase 3 because it changes the most code. It also uses the results of all
  the phases before it.
- Phase 5 is last because it uses the cost of each pass, which phase 4 gives.

Phases 1 and 2 are independent. Do phase 1 first.

## 8. Phase 0: three measurements

Purpose: get three numbers that the other phases use.

Do not change what the engine draws in this phase. A temporary option in the toolchain is
permitted for 0.1.

### 0.1 The cost of strict math

Question: how much slower is a frame when only the geometry kernels are strict?

What is known:

- Strictness is a property of the full build today (`Checks::exact` in
  `toolchain/shader/src/lib.rs`, which `wrela build --exact` sets). No kernel can be strict
  alone.
- With no contraction, the crowd lost a maximum of 4.3% of a frame on the M4
  (`toolchain/shader/src/lib.rs`).
- With safe math, an older frame lost 91% on the M4 (`toolchain/shader/src/metal.rs`).
- No person measured strict math on the engine as it is today.
- Measured (phase 0, M4, commit after `587dc54`; `tools/bench --against`, 1080p, frames 30 to 89,
  three pairs). The geometry kernels strict:
  - SCENE 10: span 24.633 to 25.339 ms (+2.9%, noise 0.284 ms). Passes alone: `mesh_extract`
    3.044 to 3.521 ms (+15.7%), `mesh_certify` +33.5%, `mesh_skin` +23.0%, `mesh_sun_exact`
    +18.9%; every other pass within 2%.
  - SCENE 11: span 16.646 to 16.594 ms (-0.3%, inside the noise). `mesh_sun_exact` +8.4%.
  - Safe math alone (contraction left on), SCENE 10: span +1.0%, inside the noise;
    `mesh_extract` +13.1%, `mesh_certify` +28.1%, `mesh_sun_exact` +16.4%. No contraction alone
    (relaxed math): span -1.1%, inside the noise; `mesh_extract` +3.7%, `mesh_certify` +11.7%,
    `mesh_sun_exact` +2.8%. Most of the cost of a strict pass is safe math, not the loss of
    contraction.
  - With the fragment kernels of the exact cells strict too (step 8: `exact_box_vs`,
    `exact_box_one_fs`, `vis_cut_vs`, `vis_cut_one_fs`, `vis_cut_many_fs`,
    `vis_deformed_cut_vs`, `vis_deformed_cut_one_fs`): SCENE 10 span 24.791 to 25.550 ms
    (+3.1%, noise 0.550), `mesh_exact` +27.8%; SCENE 11 16.614 to 16.768 ms (+0.9%, inside the
    noise), `mesh_exact` +22.7%.
  - The Radeon is not measured yet: it was in use by another project for the whole of phase 0.

Steps:

1. Add a temporary build option that takes a list of kernel names. The back end compiles those
   kernels strict and the other kernels relaxed.
2. Make the list of the geometry kernels. Use `tools/wrela pipelines` to find the names. Include
   the compute kernels of extraction, of the certificates and of the exact rays of the sun. Put
   the list in this file. The list (SCENES 10, 11, 13 and 14):
   - Extraction (`lib/engine/mesh/extract.wrela`, `dc.wrela`, `emit.wrela`): `prep_slices`,
     `prep_chunk`, `zero_chunk_counts`, `reset_dirty_bricks`, `cull_bricks`,
     `cull_bricks_rest`, `eval_corners`, `eval_corners_rest`, `classify_cells`, `solve_verts`,
     `solve_verts_rest`, `prep_creases`, `refine_creases`, `refine_creases_rest`,
     `count_brick_tris`, `count_brick_tris_lod`, `write_brick_meta`, `emit_tris`,
     `emit_tris_lod`, `scan_brick_groups`, `scan_brick_local`, `scan_group_sums`,
     `record_seam_start`, `note_seams`, `reset_seam_seen`, `box_seam_cells`,
     `solve_seam_smooth`, `refine_seam_creases`, `inherit_seam_corners`, `finish_seam_tris`,
     `finish_draw`, `finish_draw_patch`, `set_instances`, `deform_skinned_verts`, `keep_verts`,
     `mark_first`, `prep_from`.
   - Certificates and exact cells: `dilate_outline`, `filter_thin_cells`, `note_untested`,
     `certify_outline`, `note_uncertified`, `set_exact_args`.
   - The sun (`lib/engine/mesh/sunlight.wrela`): `sun_cert_level0`, `sun_cert_reduce`,
     `sun_fit_reduce`, `sun_fit_set`, `sun_gather`, `sun_slices`. (`sun_exact_one` was in this
     list; phase 1 found that its output is a pixel's light and made it relaxed.)
   - Not geometry kernels: the swarm's (`swarm_*`: a simulation of grains, for the picture), the
     radiance, sky and shading kernels, the culls of instances and the Hi-Z.
3. Build SCENE 10 and SCENE 11 two times from the same commit: one time relaxed, one time with
   the list strict.
4. Do the first run of each strict build at a small size (section 5). Read the times.
5. Compare the two builds of each scene with `tools/bench ... --against` on the M4 and on the
   Radeon.
6. Record the difference of the span in percent, and the difference of each pass.
7. If a span is more than 5% slower, find the cause. Build with only no contraction, then with
   only safe math. Record which one costs the time, and in which kernels.
8. Do steps 3 to 6 again with the fragment kernels of the exact cells in the list. They also
   find where matter is. Record this result separately.

Result: the cost of strict geometry kernels in percent, for each machine, and its cause. Phase 1
has the target (5%) and removes the temporary option.

The temporary option is the environment of `wrela build`: `WRELA_STRICT_KERNELS` (a comma list
of entry functions; a pipeline any of whose entries it names is written strict) and
`WRELA_STRICT_PARTS` (`contract` or `safe` for half of strict mode, on Metal).

### 0.2 The part of extraction that the cook can save

Question: how much of the extraction time is work that needs only the field, and not the eye?

What is known:

- The mesh of a batch depends on the eye. The engine makes a set of grids for each view
  (`build_lod_grids_view` in `lib/engine/mesh/grid.wrela`). It extracts again when the eye moves
  one step or turns. The key of an extraction contains the position and the direction of the
  eye (`lib/engine/mesh/extract.wrela`).
- Thus a cook cannot contain "the mesh of a kind". It can contain only the results that are the
  same for each eye.
- The first frame of SCENE 10 was 225 ms on the M4. A frame after that was approximately 21 ms.
  In the serial run, extraction was 43 ms (design notes, 2026-11-24). These numbers are older
  than commits `233fd5d` and `5a6dfd6`. After those commits, an extraction that a frame cuts
  short continues in the next frame.
- Measured (phase 0, M4, 1080p, `--timestamps --serial`, so each dispatch alone; each batch's
  passes labelled with its index for the run, an instrumentation not kept):
  - SCENE 10. Settled at frame 2: the pillars (9.8 ms of extraction), the brazier stands
    (4.8 ms), the rival orb (0.18 ms: a rigid mesh moved), the glass barrage (0.25 ms) and the
    two creatures (2.5 and 2.2 ms, then 0.1 ms of skinning a frame). The courtyard settled at
    frame 4 (30.2 ms). The flames and the parry arc never settle: their fields read the time,
    so they are extracted again each frame (2.8 and 4.3 ms a frame).
  - SCENE 13. The room (9.0 ms), the desk (17.4 ms) and the book (4.6 ms) settled at frame 2.
  - `std::derive::reads_only_input` is false for the distance of every kind of the scenes: each
    distance reads the values of its own struct, which the derivation counts as captures. So the
    question "does it read the time" has no answer today; phase 2 gives it one. By the code, the
    kinds that read no time and no state are the courtyard, the pillars, the brazier stands, the
    rival orb's shape (its placement moves), the citadel, the marble column (until it is
    edited), the stack, the stage, and the room, the desk and the book.
  - The parts of the extraction of the kinds that read no time, to the frame each settles, by
    kernel. Work that needs only the field and a cell (`eval_corners*`, `classify_cells`,
    `solve_verts*`, `prep_creases`, `refine_creases*`, `certify_outline`, `dilate_outline`,
    `keep_verts`): SCENE 10 15.8 ms of 42.3, SCENE 13 21.2 ms of 31.1. Work that needs the eye
    (`cull_bricks*`, the seams' kernels, the selection of exact cells): SCENE 10 12.6 ms,
    SCENE 13 4.7 ms. The rest (counting and writing triangles, scans, preparation): SCENE 10
    13.9 ms, SCENE 13 5.2 ms. Triangles are written per brick, so the triangles of a brick
    whose cells the cook holds are field-only work too, except at seams.
  - SCENE 13 `VIEW 3` (`game/scenes/main.wrela`: the eye moves from the room's eye to the
    carving's and the book's, 120 frames each way, 240 frames): the room was extracted again 74
    times, the desk 52 times, the book 70 times. Of the bricks they asked for after their first
    extraction (before the GPU culls them: 15.1 million, 24.7 million and 3.9 million), the
    room had asked for 51.7% before at the same cell size and place, the desk 40.8%, the book
    0%: its one grid's cell changes with the eye's distance at every step, and the origin of a
    grid is `aabb.lo - 2h`.

Steps:

1. Measure the extraction time of each kind in SCENE 10 and SCENE 13. Start at the first frame.
   Stop at the frame in which each batch is settled. Record the number of that frame.
2. Record which kinds have a field that reads no time and no state
   (`std::derive::reads_only_input`).
3. For those kinds, divide the extraction time into two parts:
   - Work that needs only the field and a cell: the values at the corners of a cell, the vertex
     of a cell, the certificate of a cell.
   - Work that needs the eye: the selection of the grids, the cull, the seams between grids, the
     exact cells.
4. Move the eye through SCENE 13 (`VIEW` 0, 1 and 2). Count how many times extraction runs again
   and how many cells it calculates again that it calculated before.

Result: two times. The first is the time that a cook can save in the first frames. The second
is the time that it can save when the eye moves. Phase 3 step 1 uses this result.

The result (M4, serial times): in the first frames, the field-only work of the kinds that read no
time, 15.8 ms of SCENE 10's 42.3 ms of their extraction and 21.2 ms of SCENE 13's 31.1 ms, and the
triangles of the bricks away from seams beside it. When the eye moves, nothing today: a cell is in
the same place for two eyes only where they chose the same root cell (the book: never), so a cook
of the eye's cells needs grids whose cells are the same for every eye first (phase 3 step 1).

### 0.3 The time to build one kind

Question: how long is the build from a changed field to a kind that the engine can draw?

What is known:

- The build of one scene takes 13 to 19 s on the M4. It uses one thread. Each scene build
  compiles `lib/std` and the engine again (`tools/gate` prints the time of its builds, commit
  `298e357`).
- The tests of `game/world`, which contain the certificate report, take 9 s on the M4 (commit
  `1c78609`).
- `wrela build` prints the time of its parts (`took:`, phase 0): reading the sources, checking,
  the constants and proofs, lowering, the CPU's modules, the shaders. `tools/wrela` prints the
  time of the field contracts that it proves before a build of a package under `game/`.
- Measured (phase 0, M4): SCENE 10 after a change to one field of `game/world` (the courtyard's
  curb radius) took 18.9 s. The field contracts (`wrela test game/world are_sound`, five tests on
  several threads) took 9 s. The build took 10.4 s: reading 0.1 s, checking 0.2 s, constants and
  proofs 0.2 s (one constant computed again), lowering 8.5 s, the CPU's module 0.2 s, the
  shaders (319 pipelines) 1.0 s. The same build with nothing changed takes 10.4 s. Lowering is
  the build's time: it instantiates, inlines and counts the work of every pipeline.

Steps:

1. Make `wrela build` print the time of each of its parts: the front end, the constants and
   proofs, the shaders. This is a general mechanism, so it stays.
2. Change one field in `game/world`.
3. Build SCENE 10. Record the time of each part.

Result: the time of a build after a change to one kind, in its parts. It shows how many kinds a
build can cook before the build is too slow for an agent.

The result: 18.9 s, of which 9 s are the contracts and 8.5 s lowering. Nothing in it grows with
the kinds that a cook would cover but the contracts. A cook that adds at most 5 s (phase 3's
target) keeps a changed kind under 25 s.

### Done when

- The three results are in the commit messages and in the "What is known" lists of this file.
- The list of geometry kernels is in this file.
- The gate passes on the two machines with the temporary option not used.

State (2026-11-25): done on the M4 (the full gate passed: audit small passes, full fails only on
the crowd's named debt, sun 13/0/1/2 hard leaks 754/1343/316, spans SCENE 10/11/13/14
24.46/16.55/18.32/21.58 ms). The Radeon's measurements of 0.1 and its gate are open: it was in
use by another project; they are to be run with the rest of the plan's Radeon work.

## 9. Phase 1: strict math in the geometry kernels

State (2026-11-26): steps 1 to 12 are done and measured on the M4; the Radeon is not measured.
Step 10 for the engine's kernels is `judge::bits`: the hole audit, for each thing it audits, runs
the strict corner kernel (`eval_corners`) over 64 bricks spread through the thing's bound, every
part live, and computes each of the 32,768 corners on the CPU from the same inputs; it says the
hash, and a corner a bit off stops the run, naming it and both values (so the gate's audit
fails). Its first run found one: the CSG-cut column, 10 corners an ulp off. `smax` was
`-smin(-a, -b, k)`, and the M4's shader compiler folds those negations into a sum and fuses it
into one rounding in a strict kernel, `fp contract(off)` or not (a probe: 6,192 of 262,144;
`smin` itself, and a sum or difference of a product written as such, none). `smax` is now written
as that sum, the same bits on the CPU (negation is exact, rounding to nearest symmetric), and
every thing the small audit reaches (15) has the CPU's bits at every corner. Step 11 is `std::strict`: a strict kernel computes std's strict functions, a
division, a root and a remainder at 64 fixed hard arguments (132 µs once, on the M4), held to the
bits the CPU computed from the same code at build (a constant); the engine's sight asks at its
start, and a GPU that differs stops the program, naming the first result (with the kernel's
argument moved by an ulp: "654 of 1152 results differ, the first sin at 0.5"). SCENE 10 is +3.3% over phase 0, paired (below), inside the
target of 5%. The gate on the M4: the audit and the sun's judge pass unchanged; spans SCENE
10/11/13/14 24.87/17.02/18.43/22.02 ms against lines 25.41/17.34/18.93/22.25.

Where SCENE 10's strict cost is (M4, frames 30 to 33 at 1080p, each dispatch alone, GPU time
summed over the four frames; the same build relaxed beside it): extraction 21.4 ms strict, 12.8
relaxed. Of the 8.6 ms: std's transcendentals 3.0 ms (strict with the driver's: 18.4 ms);
`lower_strict`'s writing of the driver's built-ins as the CPU computes them 2.5 ms (strict without
it: 18.9 ms; a vector's `normalize` is a division a component, as the CPU's is, and a safe-math
division is costly), and safe math the rest. Skinning 1.84 ms strict, 0.79 relaxed, 0.91 without
`lower_strict`. The certificates 2.11 ms strict, 0.67 relaxed. These are the kinds re-extracted
every frame (the flames and the parry arc read the time): phase 2's classes and phase 3's cook
remove most of the extraction of the rest. Next: a `normalize` of one division (on the CPU and
in a strict kernel alike, so their bits still agree), cheaper std transcendentals for |x| < 6000
(their error limits measured again), then measure again.

### Where the strict cost is, and the decision it leads to (2026-11-25)

Paired on the M4 (`tools/bench --against`, same machine state), SCENE 10 strict is 29.86 ms
against the same build relaxed 26.56 ms (+12.4%, noise 1.66 ms; the machine's load moved every
span by up to 2.4 ms during the day, so only a paired run measures this). SCENES 11, 13 and 14 are
within their lines. SCENE 10's strict GPU time over frames 30 to 33, each dispatch alone:
extraction 20.2 ms (12.8 relaxed), skinning 1.24 (0.79), certificates 2.08 (0.67). Taken off one
at a time: std's transcendentals 3.1 ms of the extraction; the strict target's derived intervals
1.2 (and 0.95 of the certificates: a NaN guard on each widened bound); `lower_strict`'s built-ins
1.2. Not the cost: safe math itself (relaxed math with reassociation and contraction off and
precise divisions measured the same and passed every bit test), the long reduction of a large
sine's argument (0.5), constant folding, the size limit of turning branches into selects.

The cost falls on the kinds extracted again every frame (the flames and the parry arc read the
time; the bodies are skinned): their meshes live one frame, are never cooked and never leave the
machine. What needs a strict kernel's portable bits is a mesh that persists: a cooked cell
(phase 3) and a settled kind's cells beside it. So step 3 becomes: a kind that changes every frame
is extracted by the same kernels written exact (no contraction, safe math, the driver's
transcendentals with their WGSL limits in the proofs), and a kind that settles by the strict ones.
Which is which is phase 2's class (a part of class `Frame`), so phase 2 comes before this phase's
span target; until then the measurement above stands as the distance from the target.

### Transient kernels: measured and removed (2026-11-26)

`dispatch(..., transient: true)` wrote a `@strict` kernel exact with the driver's built-ins, and a
batch whose field changed the frame before dispatched so. Each pass got faster (SCENE 10, passes
alone: extraction 4.87 to 3.45 ms, certificates 0.53 to 0.23, exact cells 0.97 to 1.36), but the
frame got slower: paired against phase 0 on one host, SCENE 10 was 31.0 ms with it and 26.3
without (phase 0 24.5 to 24.6, steady), the GPU idle longer after the sun's certificates (1.0 to
2.2 ms), and its runs varied by up to 8 ms. A paired run is the only measure here: the same build
read 24.7 to 28.6 ms in single runs. The mechanism, its pipelines (452 against 319) and the
suggestion to make it a specialization constant are gone with it.

Where the rest was (2026-11-26): not the strict kernels. With every `@strict` taken off the
engine, SCENE 10 still read 25.65 ms against phase 0's 24.56, its runs 24.2 to 29.5; with the
sun's alone off, 26.51 against 24.53; with the driver's transcendentals, or the GPU target's
derived intervals, no nearer. It was the timer of a timed run: its 512 queries held 256 timed
passes, and a frame of more waited for the GPU in its middle to read them, the GPU then idle while
the CPU recorded what followed, inside the span. Phase 0's frame has 309 timed passes and waited
after the eye's shadow map, where the next piece is short; today's 269 waited after the sun's
classification, before the exact sun rays and the shading (1.7 to 2.8 ms idle a frame). The timer
now grows when a frame fills it, as the uniform ring does (`a_timer_a_frame_fills_grows_to_hold_the_frame`).
Paired with it, five runs: SCENE 10 24.72 ms [24.65, 24.82] against phase 0's 23.92 [23.77, 23.99],
+3.3% (noise 0.23), inside the target of 5%. The spans of the gate's lines were measured with
the old timer, so each scene's line holds more slack than it says.

### What is wrong today

- The proofs make each GPU result wider by two times the error limit that WGSL gives for the
  operation (`toolchain/ir/src/derive/interval.rs`).
- The default Metal build uses relaxed math. Relaxed math can change the order of the
  arithmetic. The error limit of one operation does not include that change.
- WGSL gives an error limit for `sin` and `cos` only in [-π, π]. The proofs assume the same
  limit outside that range. No specification says this.
- Vulkan permits an error of 2.5 ulp in a division (`docs/spec/language.md`, §11.3). Thus two
  GPUs can give different bits for the same division.

### The idea

Some operations give the same bits on each GPU when the kernel is strict: a float add, subtract
and multiply, a comparison, a conversion, and the other operations that §11.3 lists. These are
the safe operations. If a function uses only safe operations, the same source gives the same
bits on the CPU and on each GPU. Then a test of the function on the CPU is also true for each
GPU.

### Steps

1. **Add strictness as a property of a function.** Make it an attribute of the same type as
   `@deterministic`. The compiler must refuse a strict function that uses, itself or through a
   function it calls, an operation whose result a driver defines (the `driver` effect, step 9).
   (A function a strict function calls is compiled strict inside it, so it need not be marked:
   the effect, not a mark, is what the compiler checks. `toolchain/sema/src/effects.rs`.)
2. **Compile a kernel strict if its entry function is strict.** Use no contraction on the two
   targets. Use safe math on Metal. Remove the temporary option of phase 0.1.
3. **Make the geometry kernels strict.** Use the list from phase 0.1. The shade kernels stay
   relaxed. Their results are only for the picture.
4. **Do a test of the hardware operations on each machine.** Put the test with the GPU tests of
   `toolchain/host`, which `tools/check` runs.
   - Square root: all 2³² inputs.
   - Reciprocal: all 2³² inputs.
   - Division: a set of difficult inputs and a large random set with a fixed seed.
   - Divide each test into dispatches of 50 ms or less.
   - Compare each result with the result of the CPU. The CPU rounds each of these correctly.
5. **If a GPU gives a different result, write that operation in software for strict
   functions.** Use only safe operations. Record the cost in the frame. An operation that
   passed step 4 on the two machines is also a safe operation.
6. **Write each math function that a strict function calls one time in `lib/std`, in `f32`
   only.** The compiler error of step 1 gives the list. It contains at least `sin`, `cos`,
   `exp`, `log`, `pow` and `atan`.
   - Use only safe operations.
   - Compile the same source for the CPU and for the GPU.
   - Strict functions must call these functions. They must not call the functions of the driver.
7. **Find the error limit of each function of step 6.**
   - For a function of one argument, run it on the CPU for all 2³² inputs. Compare each result
     with a result that has more precision. Record the maximum error in ulp.
   - For a function of two arguments, calculate the limit from the limits of the functions
     that it calls.
   - Keep the test in the toolchain tests. If it is too slow for each run, give it a build
     option and record the last result in the code.
8. **Change the proof code to use the limits of step 7 for strict functions.** A proof about a
   strict function widens a result by the limit of step 7, one time. Remove the assumption about
   `sin` and `cos` outside [-π, π].
9. **Refuse operations that the driver defines in a strict function.** The compiler must give
   an error for a texture read that uses the hardware filter, and for a derivative. To blend
   texels in a strict function, read each texel and calculate the blend in the function.
10. **Add a bit comparison to the judge.** For each strict kernel, the judge runs the entry
    function on the CPU with the same inputs. It compares the hash of the CPU output with the
    hash of the GPU output. A difference is a failure of the gate.
11. **Add a self-test that the host runs at start.** It runs each strict math function on the
    GPU with a small set of inputs and compares the results with constants. The host stops if
    one is different. This step is necessary before a release. It is not necessary before phase
    3.

12. **Keep each strict kernel within the 50 ms rule.** std's code for a transcendental or a root
    costs far more work than the driver's (`toolchain/tests`' `strict_builtins_cost_what_std_s_code_does`:
    a root 70, a sine 302, against 4), so a heavy field's crease kernel came to 4.06M work an
    invocation (SCENE 13's desk), past the most one invocation may do on the M4 (2.44M:
    `std::gpu::limits().lane_work`, `LANE_WORK_LEAST`). A crease cell's solve is now resumable
    (`crease_steps`), and each kind takes as many of its 64 steps a dispatch as its strict count
    of a step allows (`crease_part`), a constant of the build: the heaviest crease kernels are
    1.21M (SCENE 13) and 1.37M (SCENE 14).

### What was learned

- On 2026-11-25 a SCENE 13 run held the M4's GPU until the window server's watchdog fired. A
  pass (`range_simplify`) took a loop counter's range past the loop's exit test for its range at
  the test, so it erased the test. Only a strict module runs that pass after its calls are
  merged into loops, so 14 strict kernels had loops with no exit, and the strict recount dropped
  the "no bound" result. All three are repaired: the range, the recount (a strict kernel's work
  is that of the code that runs), and a check (a pass that erases a bounded loop's exit is an
  internal error). A dispatch that runs cannot be stopped, and a host the timeout kills leaves it
  on the GPU until a restart: `tools/gpu` now refuses to start on a busy GPU.
- The M4's plain root differs from the correctly rounded one by an ulp, its `precise::sqrt`
  doesn't (all 2³² arguments): a strict root on Metal is the GPU's. std's correction for a target
  whose root isn't rounded (`std::math::sqrt_from`) is the root's bits plus or minus one, from one
  wrapping square, with no branch, checked for every positive normal argument from its root and
  both neighbours on the CPU.
- A count of a kernel's code in a module of its own is not the kernel's: the kernel inlines it
  and turns small branches into selects differently, and reads its data from buffers (35% to 55%
  under some SCENE 14 kernels' bounds). A program reads a kernel's own bound with
  `std::gpu::kernel_work(kernel.bind(...))`, and crease dispatches are sliced by it.
- `sun_exact_one` is not strict: what it decides is a pixel's light, not where matter is (step
  3's list corrected), and a strict march at every listed pixel cost SCENE 13 2.3 ms.
- A strict kernel's bound is the greater of its source's count and its flattened code's: the
  flattened count of `cull_bricks` came to 2,755 against 164,544 as written, and the kernel
  takes 4.4 ms for 8,192 invocations. Why the flattened count loses that work is open.
- `glass_shard` has no steepness (`lip inf` in the facts report). It entered the report at
  `22446ef`, and the `std::field` of that commit gives it none too: this is not a loss of
  phase 1. Phase 5 finds its cause.
- A scene build lowers in more time: SCENE 10's lowering is 13.1 s (8.5 s at phase 0): std's code
  in each strict kernel.

### Done when

- Each strict kernel gives the same bits on the M4, on the Radeon and on the CPU (step 10).
- `interval.rs` uses no error limit of WGSL for an operation in a strict function.
- The certificate report gives no kind fewer certified cells than before phase 1. The table in
  section 13 gives the counts at `1c78609`.
- No span of the gate is more than 5% slower, on the two machines.

If a function of step 6 makes a kernel too slow, make the function faster. Do not go back to
the function of the driver.

### What cannot be the same

The rasteriser selects the triangle that covers a pixel. Each driver does this with its own
precision. Thus two GPUs can give different pictures at the edge of a triangle. The judges must
continue to use a tolerance for pictures. The bit comparison is only for the data that the
geometry kernels write.

## 10. Phase 2: the change class of each part of a field

State (2026-11-25): done on the M4 (the Radeon is open, as for every phase). What was learned
corrects steps 1 to 3 and 7:

- A part's class is not in its field's code. Phase 0 found that every kind's distance reads its
  own data (`reads_only_input` false for all), and whether that data moves is decided where the
  game builds the value (`arena_pillars()` against `arena_flames(t)`), which the engine receives
  as a value; the language has no channel from a caller's expression to the callee (a parameter's
  default is a constant of the build). So what the toolchain gives is the parts' structure, and
  the class is measured: `std::field::Surface::part_state` writes what part `i` is made of (std's
  leaves themselves, its unions the operand holding it, its warps, displacements, mirrors and
  materials their own data), `part_bound` its bound, and `engine::mesh::extract::part_classes`
  compares two moments. A surface std can't see into says so, and its whole hash decides.
- The report is `game/world`'s `part_classes_report` (`tools/wrela test game/world`): the flames,
  the parry arc, the rival orb, the glass shards and a posed body are `frame`; the courtyard,
  pillars, brazier stands, citadel, marble column, stack, stage, table (4 parts) and grimoire (6)
  are `never`. Step 5's test is `lib/engine`'s
  `a_part_that_reads_the_time_is_frame_and_the_rest_never`.
- Step 6: a batch keeps each part's state and box (`part_states`); a field whose state moved in
  some parts only is patched, as an edit is, in the bricks of those parts' old and new boxes.
  SCENE 15 (a slab, three blocks and a ball circling them): 128 bricks culled and 47,104 corners
  evaluated a frame, against 9,856 and 2,621,440 for the whole field; extraction's GPU time over
  frames 30 to 39 1.9 ms against 23.8; the picture as the whole field's.
- Step 7: no kind's parts are `never` by proof (each reads its data), so the hash stays: it is how
  the class is measured, a hash a part a frame on the CPU.
- No span is slower (M4 gate: SCENE 10/11/13/14 29.13/16.87/18.39/21.72 ms, as before; SCENE 10's
  excess is phase 1's).


Purpose: the compiler tells the engine how fast each part of a field can change.

### What exists today

- `std::derive::reads_only_input` answers one question for a full field: does the field read
  only its input?
- The engine compares one hash of the full field (`field.state_hash` in
  `lib/engine/mesh/extract.wrela`). If one part of a kind changes, the engine extracts all parts
  of that batch again.
- An edit is the exception. The engine extracts again only the bricks that the edit touches
  (`Edited<M>`, `try_patch_dirty`).

### The classes

Start with these four classes. The list is in order, from slow to fast.

| Class | A part has this class when |
| --- | --- |
| Never | It reads only its input point and constants |
| State | It reads a value of the game that changes at a tick or less frequently |
| Frame | It reads the time, or a value that changes each frame |
| Eye | It reads the position or the direction of the eye |

A part that reads values of more than one class gets the fastest of those classes.

### Steps

1. **Calculate the class of each operation on the tape of a field.** An operation gets the
   fastest class of the values that it reads.
2. **Calculate the class of each part of a field** from the classes of its operations.
3. **Give the classes to programs through `std::derive`.** Code that runs at build must be able
   to read them.
4. **Add a report to `wrela`.** Make it a command of the same type as `wrela pipelines`. For
   each kind, it lists the parts and the class of each part.
5. **Add a test.** One field has a part that reads the time and a part that does not. The test
   makes sure that the first part is "Frame" and the second part is "Never".
6. **Use the classes in extraction.** When a part of class "State" or "Frame" changes, extract
   again only the bricks that the bound of that part touches. Use the method that an edit uses
   today.
7. **Remove the run-time check that a class replaces.** If each part of a kind is "Never",
   remove the comparison of the hash for that kind in the same change.

### Done when

- The report is correct for each kind in SCENE 10, SCENE 11 and SCENE 13. The test of step 5
  passes.
- In a scene with one part that moves, extraction calculates only the bricks that the part
  touches. Show the count of bricks before and after.
- No span of the gate is slower, on the two machines.

## 11. Phase 3: the cook

State (2026-11-26): in progress. Step 1's grids are built and tested on branch
`engine-plan-ladder`, and not merged: they fail the gate's sun line of SCENE 13 VIEW 0 (below).
Every grid's cell is on one ladder, the largest of `2^k` times 1, 1.25, 1.5 or 1.75 m no larger
than the cell asked (`ladder_cell`, at most 20% finer, never coarser), and its origin is a whole
number of its bricks, so `origin + n · cell` is exact and a corner's bits are its cell's and
place's whatever eye chose the grid. The world's own placement of the lattice stands 0.137 mm off
the ladder's points, which hold every round coordinate: where two solids meet on such a plane
(the stage's wall on its floor at `y = 0`) their union is 0 on the face they share, and corners
there meshed a sheet of outside between them (448 edges of four triangles). Its test,
`two_eyes_put_a_cell_at_the_same_bits`: three eyes' grids round a room share 4,689,500 corners,
two single grids asked for different cells 1,828,265, and no corner's bits differ.

What moving the cells showed, each a fault the old placement had hidden (the general ones are on
this branch):

- The gate's numbers move past their slack with where the cells fall (§5): it holds the sun's
  judge and the spans as means over placements of the lattice.
- A kind's first extraction was capped at 16,384 bricks (from when a frame held 4,096
  dispatches), so a kind with more drew its first frame with holes (the cut column at cells 20%
  finer: 8,981); a cold extraction takes what the frame's budget of dispatches affords.
- A chunk's thin-cell candidates passed its 8 slices (the pillars at 3.7 mm: 68,917 in one chunk,
  past 65,536), so the batch was meshed again every frame and never settled: 16 slices.
- The audit's march took a fixed 65,536 rays a frame: the grimoire's frames 414 to 571 ms, and
  with frames queued one was undone 1.5 s after its submission, which stops a run. A frame's
  march is now what 4 dispatches' most work holds at its field's step cost.
- The audit made each kind's sight again at the world's own placement: it keeps the scene's.
- The crowd's outline at 160x90 fails its budget at three of four placements on the old grids (5,
  4 and 5 holes against 2.9) and passed at placement 0 only: on the ladder 3, 4, 3 and 2.
- A noise-weathered rock's mesh folds about 1 triangle in 10,000 at most placements where a
  crease cell's solution lies past its cell (its clamped vertex 0.09 cells inside the solid); a
  smooth sphere across a 2:1 seam, 5 in 220,018 triangles over twelve placements. Each test had
  passed at its one placement; on the ladder's branch they hold several to a rate. Open.

Why the ladder waits: the sun's judge moves with the mesh far past its slack, whatever moves it
(cells 1% larger: SCENE 13 VIEW 0's hard leaks 754 to 882), and on the ladder VIEW 0 is worse at
every placement (845 against 773 over four, cells 8% finer there), VIEW 1 better (1,037 against
1,358). The cause is in the sun: `sun_classify_fs` proves a pixel lit or dark from the map taking
the meshes' error as half a pixel (`HOLE_PX`), which only outline cells are certified to; inside
the outline a mesh may stand off its field by much of a cell (at a crease more), so where the map
proves light the field blocks, and which pixels those are moves with the tessellation. The sun's
proofs need the mesh's own bound there (phase 5's proof debt), and then the ladder is judged
again. Rounded to the nearest step instead, VIEW 1 was the worse (1,553 against 1,358) and
SCENE 10's span the better (23.45 against 24.38 ms; rounded down 26.68: its flames and arc are
meshed again every frame, at cells 9% finer).

Purpose: the build calculates, one time, the results of extraction that do not depend on the
eye. The game loads them and does not calculate them again.

### What exists today

- The mesh of a batch depends on the eye (phase 0.2). Thus the cook cannot be "one mesh for each
  kind".
- The grids of one batch share one origin, and each grid uses whole numbers of cells from that
  origin (`lib/engine/mesh/grid.wrela`). Thus a cell can have the same position for two eyes.
- The build keeps the constants that it calculated in a cache on the machine
  (`wrela build --report consts`). A build can also put a file of its package into the program
  (`embed`, `docs/spec/language.md` §13.3).
- `tools/gate` keeps each result under a key that it makes from the contents of files
  (`~/.wrela/gate/`).
- Two earlier methods were removed. Read their commits before you start.
  - A pool of bricks that the engine kept between frames (added in `d6a9616`, removed in
    `b7edd49`). It kept a mesh that was cut short and called it settled.
  - `BakedField`, which kept a field as a volume (removed in `00225aa`). Nothing used it.

### What the cook contains

The cook contains the results for one cell that are the same for each eye:

- the values of the field at the corners of the cell;
- the vertex of the cell;
- the certificate of the cell.

It contains them for each brick of a kind, at a fixed set of cell sizes. While the game runs,
the engine selects the bricks and the cell size for the eye, as it does today. It makes only the
seams and the exact cells live.

The cook is not a second engine. The same GPU kernels make it. Thus a cooked cell and a live
cell have the same bits, and they join with no crack.

### Steps

1. **Make sure that the list above is correct.** Use the result of phase 0.2. For each item,
   show with a test that two different eyes get the same bits for the same cell. If the grids
   of two eyes do not put a cell in the same position, change the grids first. That change must
   pass the gate alone.
2. **Divide extraction into two stages** in the engine: a stage that needs only the field and a
   brick, and a stage that needs the eye. The picture must not change. Pass the gate.
3. **Add a cook run.** It is a run through `tools/gpu` with no window. It cooks each kind whose
   parts are all of class "Never". For such a kind, it runs the first stage for each brick at
   each cell size. It writes the results to a file. Obey the 50 ms rule: divide the work into
   dispatches.
4. **Give each cook a key.** The key is the hash of the field, the hash of the engine and the
   name of the machine. Make the hash of the engine from the contents of its files, as
   `tools/gate` does. Do not use the commit.
5. **Keep the cook between builds,** out of the repository, in the same manner as the results
   of the gate. A build makes a cook again only when its key changes.
6. **Load the cook when the game starts.** For each cell that the cook holds, the engine skips
   the first stage.
7. **Do a check of each certificate on the CPU when the cook loads.** The check must share no
   code with the extraction kernels, apart from the field. If a cook does not pass, refuse it,
   say which kind failed, and extract that kind live.
8. **Remove the name of the machine from the key** after phase 1 shows the same bits on the two
   machines.
9. **Before a release, decide where the cook is made.** If phase 1 gives the same bits on the
   GPU of each player, the cook comes with the game. If it does not, make the cook on the
   machine of the player at the first start. Do not make it while the player plays.

### Done when

- A cooked scene passes the hole audit with the same counts as the live scene, on the same
  machine.
- The first frame in which each batch is settled has a smaller number than with live
  extraction, on the two machines. The sum of the spans to that frame is also smaller.
- When the eye moves through SCENE 13, the engine calculates no cell again that the cook holds.
- The cook of one changed kind adds not more than 5 seconds to the build.
- No span of the gate is slower, on the two machines.

## 12. Phase 4: decisions and the frame schedule at build

State (2026-11-26): step 1's list is below; step 7 is done: `wrela pipelines` names each kernel's
kind by its type arguments (`eval_corners::<ParryArc>`) and the engine function at each of its
sites (`extract.wrela:2873:17 in `MeshBatch::run_extract``). Step 6 is done: a dispatch whose
size the build knows (its group counts constants, or arithmetic of them) and whose bound times
its invocations is past 50 ms on either target, at the host's arithmetic, is E0714 (no engine
scene has one). Step 5 has its first model: `tools/cost <build>` prices each pass of a frame by
the compiler's bounds at the host's arithmetic (a dispatch's bound times its invocations, a
covering draw's fragment bound times the pixels) beside the serial run's time of each. SCENE 10,
frame 40, 1080p, M4: 628.5 ms modelled against 13.8 ms measured over the priced passes (45.6
times), extraction 67.6 times (its indirect dispatches booked at their stated most), direct light
85.6, the sun's certificates 0.8; mesh draws are not priced yet. The model now counts what a
pass does in a usual frame: `wrela build --tally` makes every invocation add up the work it does
(`ir::tally`), and the host prints each pipeline's sum a frame; `tools/cost <build> --tally
<counting build>` prices a pass by that. SCENE 10, the same frame: 63.6 ms against 17.9 (3.5
times); the dispatched passes within 0.3 to 2.5 times (extraction 1.6, the certificates 0.6, the
sun's exact rays 0.9), the full-screen fragment passes 4 to 9 times over (direct light 6.0, the
refraction 8.6, the tone 5.8): the time a unit was measured on a dependent chain of sines, and a
fragment's work runs its independent operations side by side. Then the model became counts times
units: a counting build also counts each pipeline's invocations and its subgroups' streams (what
each subgroup ran, each stretch once for its lanes: a pass whose lanes diverge costs that, not
its lanes' sum), and its run reads back each indirect draw's vertices and instances; `tools/cost`
prices a pass as units times (compute work, compute stream, fragment work, fragment stream,
fragments, vertices, commands), the units this machine's, fitted (`--fit`, least squares, none
negative) on some scenes and held to the others' gate spans. M4, frame 40 at 1080p, fitted on
all four: SCENE 10 -0.9%, 11 -15.0%, 13 -52.9%, 14 -56.2%; fitted on SCENE 10 alone, 10 -3.2%,
11 -15.1%. What it can't price is SCENE 13's and 14's exact pass: SCENE 14's candlestick's exact
cells (2,873 boxes, 525,603 fragments, a stream of 4.1e6 at about 190 work a fragment, so about
0.1 ms of arithmetic) take 10.3 ms of its 22, while the table's (25,223 fragments, a stream
eleven times its lanes' sum) take 0.19. Not the code after a discard (a fragment shader that
returns right after it is slower, 15.8 ms: the M4 doesn't run a discarded fragment on), nor the
arithmetic the counts see: the draw writes depth (a `WithDepthBehind`) and discards, over boxes
many deep, so its fragments wait on each other at each pixel. That is phase 5's largest item
(it is most of the proof debt counted below), and the model's next term. What is measured of it
(SCENE 14, frame 40, 1080p, M4): the candlestick's draw has the same counts with its march
compiled in or cut out of the shader (work 9.8e7 / 1.1e8, fragments 525,732 / 510,777, stream
4.1e6 / 3.9e6), and takes 10.3 ms or 1.4: almost none of its fragments runs the march, yet each
pays for the march being in the shader, about seven times over. Not its depth write (10.4 ms
with none), not the code after its discards (returning there: 15.8 ms), not a lost early depth
test (with no discard in the shader, every empty fragment dropped by writing the farthest depth,
518,499 fragments still run, 11.8 ms; with a floor at the shader's discard margin drawn under
it first, 519,479, 9.9 ms): those fragments are in front of the mesh, and they march: a
candidates' pass (a shader with no field that keeps only the fragments whose ray can still meet
a surface in front of the mesh, the rest turned away by the early depth test) kept 525,147 of
them, 10.8 ms. Each marches cheaply (about 190 work: a cell's live mask prunes the field to a
part or two), so the time is the shader holding the whole field while a fragment runs a part of
it, some 20 ns a fragment, and no count of work or stream sees it. Corrected by two more measures: the
counting build itself spends 19.3 ms there (so the march runs), and a variant that never
discards (every lane counted, none a helper) tallies 5.9e8 work for 11.8 ms, with no divergence
(its stream times 32 is its work), where `mesh_vis_resolve` does 5.0e9 in 2.3: the exact shader's
operations cost some forty times the screen passes' a counted unit, and a lane a discard makes a
helper (kept running for the shading's derivatives) runs the march uncounted. The counts are
right on a probe (a loop of a call, 10 and 100 passes: 406 work an invocation where 400 is
meant), its kept bytes (524) and its text (15% smaller) don't change without the march, and
neither its depth write nor an early depth test is the cause. Next: tally work by kind of
operation (arithmetic, transcendental, a uniform's or storage's read, a texture's), so the
model fits a unit to each and shows which the kit's profile code (`segment_of`, `crosses`,
`arc_of`: 60% of the candlestick shader's bound) spends on. Found on the way: a rigid thing's exact cells are tested in the
frame it was meshed in (`exact_box_one_fs` brackets the world ray by the cell's box as meshed,
where `exact_box_vs` draws it moved by `xforms[0]`), right while it is still, wrong once it
moves without being meshed again. The
target is 20% of each gate scene's span, on the two machines. Step 2's scene types are done: whether a scene has a swarm, a heightfield (opaque,
transmissive) or a medium, whether its look blooms and how many batches its world draws are
constants of their types, so SCENE 14 (no swarm, a look without bloom) has 159 pipelines where it
had 165. Step 2 has its first decision: whether a kind can be
skinned is `Surface::MAY_SKIN`, an associated constant (true for `Skinned` and the wrappers that
forward its hash, false otherwise), and the extraction's, the pool's and every draw's skinned or
deformed branch is `M::MAY_SKIN && …` (the world's, `MeshWorld::MAY_SKIN`, for the shadows), so
it folds: SCENE 14's build has 165 pipelines where it had 206 (its 17 skinned and deformed draws
and the 24 rest-pose extraction kernels of its six unskinned kinds gone), SCENE 10's 247 where it
had 320 (the creatures keep theirs); a build alone 38.6 to 23.6 s and 13.0 to 10.8 s, the host's
load (a second run) 5.1 to 5.2 s and 2.1 to 0.8 s, the first frame's CPU the same, the spans
20.87/16.52/18.97/21.95 to 20.88/16.36/18.86/21.74 ms. A kind's `FieldFacts` was the first carrier and didn't fold
on the CPU for every kind (a `Dressed` thing's facts are a call the inliner leaves), so the
constant is the kind's own. A skinned field whose kind says it can't be stops where it's meshed
(`held_unskinned`). Step 4: `wrela pipelines <package> --schedule` writes the frame's schedule from the build's CPU
code after inlining and folding (`ir::schedule`): each pass in order, the pipelines it records,
which the build couldn't settle (`?`) and which a loop records (`*`). SCENE 14's is 95 passes
(`mesh_sky`, then a `mesh_extract` and a `mesh_certify` for each of its kinds and rings, the
sdf, shadows, radiance, the eye's passes, the sun's, the light's, refraction, TSR, the tone),
recording 159 pipelines; a pass of a type the scene hasn't (the swarm's, the bloom's, a skinned
draw) is in neither. The rest of step 2's list, from the build's CPU code (`WRELA_DUMP_IR=cpu`,
SCENE 14): `edit_base_hash`, `field_facts`, `skin_shape_hash`, `lipschitz` and `steep` leave no
function behind, so the kind settles them at build; `rigid_key` and `part_states` stay, one of each
a kind, because they hash the value (a rigid thing's shape, each part's state) to see what moved,
which is the game's state; the cells' tuning (group 2) reads the value's stated steepness and
bound, and the slices (group 3) read the machine (`limits().step_work`) beside the build's counts,
so both stay at run time. Step 3 needs no change: the eye's decisions are the eye's.

### The decisions made at run time (step 1)

Read from `extract.wrela`, `world_draw.wrela` and `draw.wrela` at `f2b0742`: about 220 places
that choose a path, a pass, a capacity, a cell or a skip. By what each depends on:

- The eye (about 90): the cells (`cell_for_one`, `h_near`, `h_top`, `force_direct`), the rings
  and whether they have seams (`build_lod_grids_view`, `has_seams`, frustum culling), the exact
  cells (`use_exact`, `has_exact`, `eye_reach`), a cut (`lod_cut`), the near slots of a pool, the
  Hi-Z passes, the screen's size. These stay at run time (step 3).
- The state of the game (about 50): a field's hash moved (steady skips, rigid deltas, patches of
  edits and of parts), what is still (`still_frames`: the static SDF, shadow map and radiance),
  the swarm, a pool's size and visible things. These stay.
- What the GPU read back (about 45): every capacity's growth and its overflow, the hints that
  size chunks and slices (`chunk_limit`, `crease_slices`, `thin_slices`, `cert_slices`,
  `seam_slices`), `has_opaque` / `has_trans` / `has_exact`, `settled`, the budget's shortness,
  `cell_scale`. These stay: they are counts of this run.
- The kind alone, made at run time (step 2's list):
  1. Which path a field takes, asked of it on the CPU each frame: skinned
     (`skin_shape_hash`, and the `is_skinned` / `is_deformed` latches every draw, shadow and
     refraction then branches on; done: `Surface::MAY_SKIN`), rigid (`rigid_shape_hash`, `has_constant_channels`,
     `has_local_channels`: `rigid_key`), edited (`edit_base_hash().is_some()`: the patch
     headroom), parts (`parts()`, `part_state`: per-part patching), a pool's first thing skinned.
  2. The tuning of its cells from its stated steepness: `steep()` (`lipschitz(1) > 1.15`) and
     `lipschitz(1) > 2.5`, which set pixels a cell, the LOD's `k_px`, its area budget and
     `root_min_ratio`, and `cell_for_one`'s multiplier; `size_floor(r)` from its bound;
     `skin_bone_count`.
  3. Its kernels' slices, from counts the build already has, worked out each extraction:
     `crease_step_cost_of` / `_rest_of`, `crease_part`, `crease_passes`, `crease_slice`,
     `seam_crease_slice` and `surf_slice` (from `kernel_work`), `keep_slice_of`, `uses_kept`,
     and the dispatches a chunk makes. (They read the machine too: `limits().step_work`.)
  4. Its facts, asked again at each draw: `field_facts().may_transmit`, which the shaders fold,
     in more than ten CPU branches of the draws and the patch.
  5. The scene's types, asked each frame: `batch_count()`, `height.active()`,
     `medium.active()`, the look's bloom strength, and `finish_poll`'s readback length (always
     14). Done: `MeshWorld::BATCHES`, `Height::ACTIVE` / `HAS_OPAQUE` / `HAS_TRANSMISSIVE`,
     `Medium::ACTIVE`, `Swarm::ACTIVE` and `Look::BLOOMS` are constants of the types (the
     methods are gone), the swarm's simulation, shadows, radiance and draw are built only for a
     swarm, and the readback is read as its 14 words; the bloom's strength stays the look's
     value.

Purpose: the engine becomes a program that the build runs. It writes the kernels, the tables and
the frame schedule for this game.

### What exists today

- Constants run at build in wasmtime with a fuel limit (`docs/spec/language.md`, §13.1).
- `prove_field_facts` runs at build and gives the facts of a kind. The facts remove code paths
  that the kind does not use. That change made the pass `exact_box` two times faster: 1.71 to
  0.88 ms on the M4, 4.00 to 1.95 ms on the Radeon (commit `40512f2`).
- The compiler gives the work count of each kernel. The manifest of a build contains it.
- The host knows the maximum time of one unit of work on each machine
  (`toolchain/host/src/step.rs`). Before a dispatch runs, the host multiplies the work count,
  the number of invocations and that time. It refuses a dispatch that can be longer than 50 ms.
- `wrela pipelines` lists each kernel, its type arguments and the code that dispatches it.
- A batch finds some properties of its kind while the game runs: if the kind has a skin and if
  its mesh changes shape (`is_skinned` and `is_deformed` in `lib/engine/mesh/extract.wrela`). A
  pool finds while the game runs if its things share one mesh
  (`lib/engine/mesh/world_draw.wrela`).
- The frame is code that tests conditions each frame and then runs or skips a pass
  (`lib/engine/mesh/draw.wrela`).

One thing was removed. The toolchain had a model of the cost of a frame
(`toolchain/driver/src/wcet.rs`). Commit `00225aa` removed it because no person compared it with
a measured pass. The cost model of step 5 must not have the same fault.

### Steps

1. **Make a list of the decisions that the engine makes while the game runs.** Read
   `extract.wrela`, `world_draw.wrela` and `draw.wrela`. For each decision, record what it
   depends on: the kind, the state of the game, or the eye. Put the list in this file.
2. **Move each decision that depends only on the kind to the build.** Do one decision in one
   change:
   - Write engine code that runs at build and makes the decision from facts and counts.
   - Keep the result as a constant of the kind.
   - Remove the run-time code for that decision.
   - Pass the gate.
3. **Leave a decision that depends on the eye at run time.** Examples are `has_exact` and
   `has_seams`.
4. **Write the frame schedule at build.** The schedule is the list of the passes that a scene
   uses, in order. A pass that no kind of the scene uses is not in the schedule and has no
   pipeline.
5. **Calculate the cost of each pass at build,** for the M4 and for the Radeon. Add a command
   that prints the cost of each pass and the sum. In the same change, add a check that compares
   each calculated cost with the time of that pass from `tools/bench`.
   - Start with the work count multiplied by the number of invocations and by the time of one
     unit of work.
   - The work count is a maximum and the time of one unit is a maximum. Thus the first result
     is much larger than the measured time. To get the target, count the work that a pass does
     in a usual frame, not its maximum.
6. **Make the 50 ms rule a build error where the build knows the number of invocations.** Use
   the same arithmetic as the host, so that the build and the host agree. The message names the
   kernel. The host continues to refuse a dispatch whose size is known only at run time.
7. **Show the source of each kernel.** `wrela pipelines` must name the engine function and the
   kind that made the kernel.

### Done when

- The list of step 1 has no decision that depends only on the kind and is made at run time.
- The calculated cost of a frame differs from the measured span by not more than 20%. This is
  true on the two machines, for each scene of the gate.
- The pipeline count of each scene of the gate is not larger than before phase 4.
- The time from the start of the host to the first frame is not longer than before phase 4.
- The build of one scene is not more than 20% slower than before phase 4.
- No span of the gate is slower, on the two machines.

The calculated cost of a frame is a report in this phase. It is not a build error. Measure the
pipeline count, the time to the first frame and the build time before and after each step.

## 13. Phase 5: make the frame faster with proofs

State (2026-11-26): the loop has begun on the M4. The gate's spans now (timed as the game's frame
is, §5): SCENE 10/11/13/14 20.87/16.52/18.97/21.95 ms; SCENE 11 is within the bar.

The largest proof debt, counted: SCENE 14's frame (1080p, frame 60, each pass alone) spends 10.7 ms
in the exact cells (`mesh_exact`) and 3.5 ms in the sun's exact rays, of about 22. Its table's
batch (3.7 mm cells) gives 4,959 outline cells a certificate's test and 4,638 exact cells, 1,397 of
them thin cells; the candlestick's (4.8 mm) 2,508 and 2,873, 365 thin. So 65% and 99% of their
outline cells fail the sheet certificate. Not for want of a stronger certificate: the affine one in
place of the pruned one (`is_heavy_field`) failed as many. On the CPU, at a surface point with the
field's own normal, it certifies 6 of 300 cells of each (41 at a slab a cell wide). The cause: the
affine derivation falls back to the plain interval (no slopes) for the kit's turned and moulded
shapes (`kit::shapes::outline::Profile`), so a certificate sees a box, never a sheet. A straight
segment's offset alone kept its slopes; the profile's code lost them where it:

1. compares two conditions (`crosses`' `(a.y > p.y) != (b.y > p.y)`): no rule for `==` and `!=`
   of `bool`s; one now (sure where each side is);
2. calls a function that returns a condition (`crosses` itself, any predicate): a derived call
   returned only forms of floats; it now returns a condition's sureness;
3. carries a struct of floats and conditions through its loop (`Signed2`, `Near2`): an active
   local or result must be floats alone, or a `bool`. Now a function (or a callee) that can't be
   derived is derived as its flattened copy (`ir::opt::flattened_copy`: its calls written in, its
   counted loops written out, its small structs split, tidied as a GPU module is) before its
   interval is taken. The loop's sentinel (`none()`, 1e30) had been in every pass's form, so
   with the loop kept the form had no bound. A turned profile of four straight segments: slopes
   (2.4, -0.55) mm a half cell and an error of 0.74 mm at a 4.8 mm cell, where it had none.
4. A profile of bowed segments derives with a bound, a loose one: four points, two bowed, an
   error of 15 mm at a 4.8 mm cell. A bowed segment alone is tight (its arc's offset 0.34 mm),
   but with the rest the profile's nearest segment is unsure: a neighbouring arc's test of
   whether the box is within its span (`on_arc`) is unsure, and its distance to its whole circle,
   which may pass near the box, joins the choice. Next.
5. And the forge's kinds are heavy (`is_heavy_field`: the table's distance 10,750 work, its form
   240,820), so their kernels certify by pruning alone, which can't certify a sheet: the affine
   certificate for a heavy kind is the other half of SCENE 14's exact cells.

`WRELA_DUMP_IR=affine` says why a field's affine form falls to its interval.

A form a field gets now where it fell to its interval can be wider than that interval (a clamp's,
a mix's), and the facts' proofs read some ranges from the form alone: the pillars' and the
citadel's channels went unproven (their shaders kept the range guards) until each range the facts
read is the form's intersected with the interval (`std::derive::affine`), never wider than
before. The Lipschitz derivation given the flattened copy too moved no kind's steepness (the
citadel and the glass shard have none, the pillars 12,039, the marble column 5,720) and ran the
kit's soundness test out of fuel, so it keeps its own fallback: their causes are elsewhere, next
on this list.

Purpose: find the work that the engine does only because a proof is missing, and remove it. Then
decrease the cost that no proof can remove.

This phase is a loop. It has no fixed list of steps.

### What is known

- The spans of the gate at 1080p, in ms (`tools/gate.json`, commit `1c78609`). The bar is
  16.6 ms.

  | Scene | M4 | Radeon |
  | --- | --- | --- |
  | 10 | 24.4 | 56.6 |
  | 11 | 16.7 | 46.2 |
  | 13 | 18.2 | 26.9 |
  | 14 | 21.4 | 29.0 |

- A proof made single passes two times faster (`exact_box`, the exact rays of the sun). Those
  passes are approximately 2 ms of a 17 ms frame (design notes, 2026-11-24, SCENE 10). Thus a
  faster small pass is not sufficient. The large results come from proofs that remove a full
  class of work.
- Many cells have no certificate. The table gives the certified cells of each kind at three
  cell sizes (`tools/wrela test game/world`, commit `1c78609`).

  | Kind | 1 cm | 2.7 cm | 8 cm |
  | --- | --- | --- | --- |
  | `arena_brazier_stands` | 39 of 42 | 20 of 42 | 3 of 42 |
  | `arena_courtyard` | 100 of 100 | 100 of 100 | 100 of 100 |
  | `arena_flame` | 58 of 60 | 47 of 60 | 10 of 60 |
  | `arena_pillars` | 97 of 100 | 93 of 100 | 32 of 100 |
  | `citadel` | 100 of 100 | 99 of 100 | 99 of 100 |
  | `creature` | 20 of 25 | 9 of 25 | 0 of 25 |
  | `glass_shard` | 98 of 100 | 49 of 100 | 6 of 100 |
  | `marble_column` | 99 of 100 | 98 of 100 | 40 of 100 |
  | `parry_arc` | 56 of 60 | 51 of 60 | 14 of 60 |
  | `rival_orb` | 60 of 60 | 60 of 60 | 2 of 60 |
  | `stack` | 100 of 100 | 99 of 100 | 63 of 100 |
  | `stage` | 100 of 100 | 99 of 100 | 98 of 100 |

- Some facts are not proven (the same command, the lines of `field_facts_report`).
  - The compiler gives no steepness for the citadel. A kind with no steepness uses the slower
    solver in its exact cells (`lib/engine/mesh/shade.wrela`).
  - The steepness of `arena_pillars` is 12,039 and that of `marble_column` is 5,720. The other
    kinds are below 2.
  - The channels of 4 of 9 kinds are not proven: `arena_courtyard`, `creature`, `marble_column`
    and `parry_arc`. Their shaders keep the NaN guards and the range guards.
- Some counts of proof debt exist. None is shown as a time.
  - The judge of the sun counts the fields whose exact rays a frame did not trace (`untraced`
    in `tools/gate`). The count is 0 on the two machines at commits `605e1ad` and `1c78609`.
  - The hole audit counts the eyes that are not settled.
  - Each batch counts its exact cells (`exact_cells` in `lib/engine/mesh/extract.wrela`).

### The loop

1. **Count the proof debt.** For each pass, count the items that use a slow path because a
   certificate or a fact is missing. An item is a pixel, a cell, a ray or a brick. Start from
   the counts that exist.
2. **Record the cause.** For each count, record the kind and the operation that the compiler
   was not able to bound.
3. **Show the proof debt as a time.** Add it to the cost command of phase 4 step 5. For each
   machine, the command gives the milliseconds of each item.
4. **Repair the largest item.** Make the bound tighter, or add a new type of certificate.
5. **Measure again on the two machines.** Pass the gate. Then do step 1 again.

Do the loop until each line of "Done when" is true.

### First items for speed

Do the first two items first. Their causes are in the code today. The order of the other items
comes from step 3 of the loop.

- **Facts that are not proven.** Find the operation that gives the citadel no steepness. Find
  the operations that give `arena_pillars` and `marble_column` a large steepness. Find the
  channel that is not proven in each of the 4 kinds. Find the cost of each in the frame before
  you repair it.
- **Certificates at 8 cm.** The things of the parry and the creature have the fewest:
  `glass_shard` 6 of 100, `rival_orb` 2 of 60, `parry_arc` 14 of 60, `creature` 0 of 25.
- **Detail by distance.** A certificate shows that a coarse mesh is not more than one pixel
  from the field at a given distance. Then the engine draws a far thing with the coarse mesh.
- **A mesh that stays correct for some frames.** A proof gives the maximum rate at which a field
  changes. Then the engine keeps the mesh of a thing that moves for more than one frame.
- **Rewrites that the compiler proves equal.** The compiler finds a form of a field that costs
  less. It proves that the two forms are equal in a tolerance. Then one shape, written by two
  different methods, has one cost.
- **Small items.** Remove an index clamp or a NaN guard that a proof makes unnecessary. Use half
  floats where a proof shows that they are safe. Do these only when the work is small.

### First items for expression

- **Spell pairs.** A parry makes one field from two fields while the game runs. At build, prove
  the bound, the steepness and the cost of that field for all values of its parameters.
- **New operators.** Add a warp, a twist or a displacement to the kit when the compiler can
  bound its steepness. A field with a loose bound is correct but slow. A tighter proof makes it
  fast. Add an operator only in the change in which a field of the game uses it. Commit
  `00225aa` removed four forms of the kit that nothing used. `Twisted` was one of them.
- **The time of a parry.** Calculate from the fields the time interval in which one spell
  touches the other. The client and the server must get the same interval.

### Rules for proofs

1. A proof search in the build must have a limit in steps, not in seconds. It must give the same
   result on each machine and on each run.
2. If the search finds no proof in its limit, the build uses the slow path, which is always
   correct. The build reports the item as proof debt.
3. A long search is permitted out of the build. An agent or a tool does the search and writes a
   certificate. The build only does a check of the certificate. The check must be fast and must
   give the same result each time.
4. Do not add a general solver to the build. Its time changes much with a small change of its
   input, and its limit is a time.
5. Do not write a proof about a driver or about the rasteriser. The project does not control
   them.
6. The judges and the reference must continue to find a wrong proof (rule 3 of section 6).

### The cost that no proof removes

The cost of a shade pass is a cost for each pixel. A proof does not decrease it.

1. Find the passes with the largest cost in the cost command of phase 4 step 5.
2. If a pass has no proof debt, make the pass faster by a change to its method.
3. Measure before and after on the two machines.

### Done when

- The proof debt is less than 5% of the span, for each scene of the gate, on the two machines.
- The span of each scene of the gate is not more than 16.6 ms on the M4.
- No span of the gate is larger on the Radeon than before phase 5.
- The judge of the sun and the hole audit give results that are not worse.

## 14. Work that is not in this plan

| Idea | Why it is not here | Look again when |
| --- | --- | --- |
| Move the engine code into the Rust toolchain | The engine changes the most, and a toolchain build is the slowest edit. The engine also loses the checks that wrela gives it | Do not look again |
| A compiler on the server for spells that players write | It is necessary only if players write new fields. The design of spells is not decided | Phase 4 is done and phase 0.3 gives a build time |
| A compiler in the client | It adds an attack surface and many pipelines | A cost table for the GPU of the player is necessary |
| A general solver in the build | Its time is not stable, so two machines can get different builds. A search with a limit in steps, and a certificate that the build only examines, do the same job (phase 5) | Do not look again |
| A search that finds the frame schedule | The result is difficult to explain, and a time measurement is not stable | The cost of phase 4 step 5 is in its target |
| A server target and a wire format from the same fields | It is a second back end | The rules of the game need the field of a spell on the server |
| Cost proofs for each pair of spells | The spells do not exist | The first two spells that answer each other exist |

## 15. Questions that are open

Each question has the step that must answer it. Write the answer here when you have it.

1. Which part of safe math on Metal caused the loss of 91%? (phase 0.1, step 7) Answer, for the
   geometry kernels of today's SCENE 10 on the M4: safe math is most of the cost of strict
   kernels (`mesh_extract` +13.1% with safe math alone, +3.7% with no contraction alone, +15.7%
   with both), and the frame pays 1 to 3% of its span for it. The old frame's 91% was not
   measured again: that frame is gone.
2. Are the fragment kernels of the exact cells geometry kernels? They find where matter is, but
   the rasteriser selects their pixels. (phase 0.1, step 8) Answer: their cost strict is small
   (SCENE 10 +3.1% against +2.9% without them; `mesh_exact` +27.8%), so they can be. What they
   decide is a discard and a depth, and the rasteriser's choice of pixels is not the same on
   two GPUs, so the bit comparison of phase 1 step 10 cannot hold them. Phase 1 makes them
   strict, so their arithmetic is the CPU's, and leaves them out of the bit comparison.
3. Does each GPU round division and square root correctly for all inputs? (phase 1, step 4)
   Answer for the M4: no GPU computes subnormals in a strict kernel; the M4 takes a subnormal
   operand as a zero and gives a zero for any result whose exact value is below 2⁻¹²⁶ (before
   rounding). With that model (`strict_model` in `toolchain/host/src/gpu/strict_ops.rs`), its
   division, reciprocal, add, subtract and multiply give the CPU's bits for every input tested
   (1 / x for all 2³² inputs; the others for 16,781,060 pairs). Its square root does too, as
   `metal::precise::sqrt`, which the Metal writer uses in an exact module, for all 2³² inputs
   (2026-11-25); the plain `metal::sqrt` differs in 605,737,168 by an ulp. So on Metal a strict
   root is the GPU's (`wrela_abi::Target::rounds_sqrt`); where a target's isn't, std's correction
   (`std::math::sqrt_rounded`, `sqrt_from`) sets it right. The Radeon is not tested yet.
4. What error limit does WGSL give to `sin` and `cos`? The code uses 2⁻¹¹. Read the
   specification of WGSL before phase 1 step 8.
5. Do the grids of two eyes put a cell in the same position? The grids of one batch share one
   origin. It is not known if the origin is the same for each eye. (phase 3, step 1) Answer: no,
   before phase 3: a root's cell followed the eye's distance and its origin was `aabb.lo - 2h`.
   On branch `engine-plan-ladder`, yes: cells on a ladder, origins on whole bricks (phase 3's
   state).
6. How does the game get the cook: from a file that the host loads, or from `embed`? `embed`
   needs the file in the package, and GPU code cannot hold its bytes. (phase 3, step 6)
7. How large is the cook of one scene? (phase 3, step 3) Answer (2026-11-26, on branch
   `engine-plan-ladder`: `game/world`'s `cook_census_report`, `engine::mesh::cook::cook_census`:
   the bricks that may hold a kind's surface at each cell of the ladder, 512 corners and a flat surface's 72 vertices a brick,
   3.2 KiB): a whole kind at every cell an eye asks for is far too large. The bricks grow about
   four times a halving: the courtyard 5,065 bricks (15 MiB) at 62.5 mm, 23,649 (72 MiB) at
   31 mm, 255,480 (779 MiB) at 7.8 mm; the pillars 406,199 (1.2 GiB) at 3.9 mm; the stage 281,783
   (859 MiB) at 7.8 mm; the table 248,808 (759 MiB) at 1 mm; the citadel 274,010 (836 MiB) at
   62.5 mm already; the grimoire 12,420 (37 MiB) at 1 mm. So a cook holds each kind's coarse
   cells, to a budget of bytes, and the fine cells near an eye stay live: step 3 cooks "each brick
   at each cell size" only down to the cell where its kind's bricks pass the budget.

## 16. How to keep this plan

This file is the plan. Keep it in `docs/plans/`.

- When a phase is done, write "Done", the date and the commit at its heading.
- When a measurement of this plan has a result, write the result in the "What is known" list of
  its phase.
- If a step or a target changes, change it here in the same change.
- When all phases are done, remove this file.
