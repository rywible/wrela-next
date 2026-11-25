# Plan: the engine makes its decisions at build

Status: proposal, 2026-11-25. No phase is started.

This plan contains no new measurement. Each number comes from the design notes, from
`docs/spec/language.md` or from a comment in the code. The plan gives the source of each number.

The design notes are not in the repository now. To read them, use
`git show ca13c8e:docs/design.md`.

## 1. Purpose

Today the engine makes many decisions while the game runs. This plan moves those decisions to the
build.

- The engine stays in wrela, in `lib/engine`.
- The game stays a set of fields.
- The compiler gives facts about each field.
- At build, the engine uses the facts to write the code and the data for each kind.
- The released game contains only that code and that data.

The plan has one more aim. The proofs about GPU code must be true for the build that we release.
Today they are fully true only for an `--exact` build.

## 2. Terms

| Term | Meaning |
| --- | --- |
| Kind | One type of thing in the world, for example a pillar or a creature |
| Kernel | One program that the GPU runs |
| Strict kernel | A kernel that the back end compiles with no contraction and with no fast math |
| Geometry kernel | A kernel that finds where matter is: extraction, the certificate pass, the exact rays |
| Certificate | The data that shows that a mesh agrees with its field |
| Cook | The data that the engine calculates one time for a kind and then keeps |
| Change class | The fastest rate at which one part of a field can change |
| Target | A measured result that a phase must get before the phase is done |

## 3. Rules for all phases

1. Keep the engine in `lib/engine`. Do not move engine code into `toolchain/`.
2. Add only general mechanisms to the toolchain. A mechanism is general if it knows fields, but
   knows no engine pass and no thing of the game.
3. Keep `lib/engine/reference` and `lib/engine/judge` independent. They must ask the field of the
   author directly. They must not read the cook. They must not use a decision of the build.
4. Make each build decision from a count or from a proof. Do not make a decision from a measured
   time. A decision from a time changed with the state of the GPU (design notes, "What is
   measured, and what is not").
5. When a new method replaces an old method, remove the old method in the same change.
6. Work on each phase until each line of its "Done when" list is true. If a result is not good,
   find the cause and use a different method. Do not stop, and do not leave the problem open.
   Put each measurement in the commit message.
7. Obey the rules in `AGENTS.md`. Start each GPU run with `tools/gpu`. Keep each dispatch below
   approximately 50 ms. Measure on the two machines. Pass `tools/gate` on the two machines before
   you call a phase done.

## 4. Order of the phases

| Phase | Result | Uses the result of |
| --- | --- | --- |
| 0 | Three measurements that tell us where the time is | Nothing |
| 1 | Strict math in the geometry kernels | Phase 0 |
| 2 | The compiler gives the change class of each part of a field | Nothing |
| 3 | The build makes the cook on the GPU and keeps it | Phases 1 and 2 |
| 4 | The engine makes its decisions and the frame schedule at build | Phases 2 and 3 |

Reasons for this order:

- Phase 1 is first because a certificate is only as true as the math below it. Phase 3 keeps
  certificates.
- Phase 2 is before phase 3 because the cook contains only the parts that do not change.
- Phase 4 is last because it changes the most code. It also gets the most from the other phases.

Phases 1 and 2 are independent. Do phase 1 first.

## 5. Phase 0: three measurements

Write no new engine code in this phase. Each measurement answers one question.

### 0.1 The cost of strict math

Question: how much time does the frame lose when only the geometry kernels are strict?

Known numbers:

- With no contraction, the crowd lost a maximum of 4.3% of a frame on the M4
  (`toolchain/shader/src/lib.rs`).
- With safe math, an older frame lost 91% on the M4 (`toolchain/shader/src/metal.rs`).
- No person measured strict math on the engine as it is today. The specification says that an
  exact build is not for a time measurement.

Steps:

1. Let the shader back end make one kernel strict while the other kernels stay relaxed.
2. Make the geometry kernels strict.
3. Build SCENE 10 and SCENE 11.
4. Do a first run with `--frames 2 --timestamps --serial` at a small size. Read the times.
5. Measure frame 11 at 1080p on the M4 and on the Radeon. Measure the same commit with relaxed
   kernels.

Target: the frame is not more than 5% slower on the two machines.

If the frame is slower than the target, find the part of safe math that costs the time. Then
write the geometry kernels so that relaxed math can change nothing in them, and measure again.
Continue until the result is in the target.

### 0.2 The parts of the first frame

Question: how much of the first frame is extraction of matter that does not change?

Known numbers: the first frame of SCENE 10 is 225 ms on the M4. A frame after that is
approximately 21 ms. In the serial run, extraction is 43 ms (design notes, 2026-11-24).

Steps:

1. Measure the extraction time of each kind in the first frame of SCENE 10 and SCENE 13.
2. Record which kinds have a field that reads no time and no state.
3. Calculate the part of the extraction time that those kinds use.

This number tells us how much time the cook can save in the first frame.

### 0.3 The time to build one kind

Question: how long is the build from a changed field to a kind that the engine can draw?

Steps:

1. Change one field in `game/world`.
2. Measure the time of the build. Record the time of the proofs and the time of the shaders
   separately.

This number tells us how many kinds a build can cook before the build is too
slow for an agent.

## 6. Phase 1: strict math in the geometry kernels

Purpose: make the proofs true for the released build, and make the geometry kernels give the same
bits on each GPU.

The problem today:

- The proofs make each GPU result wider by two times the error limit of WGSL
  (`toolchain/ir/src/derive/interval.rs`).
- The default Metal build uses relaxed math. Relaxed math can change the order of the arithmetic.
  The error limit of one operation does not include that change.
- The code assumes that `sin` and `cos` have the same error outside [-π, π] as inside. No
  specification says this.

### Steps

1. **Add strictness as a property of a function.** Model it on `@deterministic`. The compiler must
   refuse a strict function that calls a function that is not strict.
2. **Compile a kernel strict if its entry function is strict.** Use no contraction on the two
   targets. Use safe math on Metal.
3. **Make the geometry kernels strict.** The shade kernels stay relaxed. Their results are only
   for the picture.
4. **Do a test of the hardware operations on each machine.**
   - Square root: all 2³² inputs.
   - Reciprocal: all 2³² inputs.
   - Division: a set of difficult inputs and a large random set.
   - Divide each test into dispatches of 50 ms or less.
   - Compare each result with the CPU result.
5. **If a GPU gives a different result, correct the operation in software for strict kernels.**
   Record the cost.
6. **Write `sin`, `cos`, `exp`, `log`, `pow` and `atan` one time in `lib/std`, in `f32` only.**
   - Use only add, subtract, multiply and the operations that step 4 showed correct.
   - Compile the same source for the CPU and for the GPU.
   - Give each function a proven error limit.
   - Strict functions must use these functions. They must not use the functions of the driver.
7. **Change the proof code to use the new error limits.** Remove the assumption about `sin` and
   `cos` outside [-π, π].
8. **Remove operations that the driver defines from strict kernels.**
   - Do not use the hardware texture filter. Read the texels and calculate the blend in the kernel.
   - Do not use derivatives.
   - Calculate the attributes of a pixel from the number of its triangle.
9. **Add a bit comparison to the judge.** For a strict kernel, the judge compares the hash of the
   GPU output with the hash of the CPU output.
10. **Add a self-test that the host runs at start.** It runs a small set of inputs with known
    results. This step is necessary before a release, not before phase 3.

### Done when

- Each strict kernel gives the same bits on the M4, on the Radeon and on the CPU.
- No proof uses an error limit that the build does not obey.
- The certificate report (`game/world/certificates.wrela`, which `tools/check` shows) gives no
  kind fewer certified cells than before phase 1.
- The frame is not more than 5% slower on the two machines (the same target as 0.1).

If a function from step 6 makes a kernel too slow, make the function faster. Do not go back to
the function of the driver.

### What we cannot make the same

The rasteriser selects the triangle that covers a pixel. Each driver does this with its own
precision. Thus two GPUs can give different pictures at the edge of a triangle. The judges must
continue to use a tolerance for pictures. The bit comparison is only for the data that the
geometry kernels write.

## 7. Phase 2: the change class of each part of a field

Purpose: the compiler tells the engine when each part of a field can change.

Today `std::derive::reads_only_input` answers one question for a full field: does the field read
only its input? Phase 2 answers the question for each part.

### Steps

1. **Define the change classes.** Start with four: never, when state changes, each frame, with
   the eye.
2. **Calculate the class of each member of a union and of each operation on the tape.** A part
   gets the fastest class of the values that it reads.
3. **Give the classes to programs through `std::derive`.** Build-time code must be able to read
   them.
4. **Add a report to `wrela`.** For each kind, it lists the parts and their classes.
5. **Use the classes in the engine.** Extract a part of class "never" one time only.
6. **Remove the run-time check that the class replaces.** Today the engine finds a field that did
   not change from its `StateHash`. If the class makes that check unnecessary for a kind, remove
   the check for that kind in the same change.

### Done when

- The report is correct for each kind in SCENE 10, SCENE 11 and SCENE 13. Do a test with a field
  that changes and make sure that the class is not "never".
- No frame time is slower.

## 8. Phase 3: the cook

Purpose: the engine calculates the mesh and the certificate of matter that does not change one
time, at build, and keeps them.

The cook is not a second engine. It is the work that the engine does in the first frames, done
before the game runs. The same GPU kernels make it. Thus a cooked mesh and a live mesh join with
no crack between them.

### Steps

1. **Add a cook run.** It is a run through `tools/gpu` with no window. It lets extraction become
   stable and then writes the meshes and the certificates to a file.
2. **Give each cook a key.** The key is the hash of the field, the hash of the engine and the
   name of the machine.
3. **Keep the cook between builds.** A build makes a cook again only when its key changes.
4. **Load the cook when the game starts.** The engine does no extraction for a kind that has a
   cook.
5. **Do a check of each certificate on the CPU when the cook loads.** Use code that the cook run
   does not use. Refuse a cook that does not pass. Then extract the kind live.
6. **Cook each kind at a fixed set of cell sizes.** The eye is not known at build. The engine
   selects the size for the eye while the game runs, as it does today.
7. **Remove the name of the machine from the key** after phase 1 shows the same bits on the two
   machines.
8. **Before a release, make the cook on the machine of the player** at the first start. Do not
   make it while the player plays.

### Done when

- A cooked scene passes the hole audit with the same counts as the live scene, on the same
  machine.
- The first stable frame comes sooner than with live extraction, on the two machines.
- The cook of one changed kind adds not more than 5 seconds to the build.

## 9. Phase 4: decisions and the frame schedule at build

Purpose: the engine becomes a program that the build runs. It writes the kernels, the tables and
the frame schedule for this game.

Much of this exists. Constants run at build in wasmtime with a fuel limit. `FieldFacts` already
removes code paths that a kind does not use. That change made `exact_box` two times faster: 1.71
to 0.88 ms on the M4, 4.00 to 1.95 ms on the Radeon (design notes, 2026-11-24).

### Steps

1. **Move one decision at a time from run time to build.** For each decision:
   - Write engine code that runs at build and makes the decision from facts and counts.
   - Keep the result as a constant of the kind.
   - Remove the run-time code for that decision in the same change.
2. **Start with the decision of how to draw each kind:** closed form, the full field, the live
   parts, or cells of one size.
3. **Write the frame schedule at build.** The schedule is the list of passes that a scene uses,
   in order. A pass that no kind uses is not in the schedule.
4. **Calculate the cost of the full frame at build** from the work counts of the compiler. Show
   it for each machine line in `tools/gate.json`.
5. **Make the 50 ms rule a build error.** If the calculated time of one dispatch is above the
   limit, the build fails. The message names the kernel.
6. **Show the source of each kernel.** `wrela explain` must name the engine function and the
   field that made the kernel.

### Done when

- No scene makes a draw decision while the game runs that the build was able to make.
- The calculated frame cost is in 20% of the measured cost on the two machines, for each scene
  in the gate.
- The pipeline count and the load time are not larger than before phase 4.
- A build after an engine change is not slower than the checks that the build replaces.

Until the calculated cost is that accurate, show it as a report and make the cost model better.
Make it a build error when it is accurate. Measure the pipeline count, the load time and the
build time before and after each step.

## 10. Work that is not in this plan

| Idea | Why it is not here | Look again when |
| --- | --- | --- |
| Move the engine code into the Rust toolchain | The engine changes the most, and a toolchain build is the slowest edit. The engine also loses the checks that wrela gives it | Do not look again |
| A compiler on the server for spells that players write | It is necessary only if players write new fields. The design of spells is not decided | Phase 4 is done and 0.3 gives a build time |
| A compiler in the client | It adds an attack surface and many pipelines | A cost table for the GPU of the player is necessary |
| A search that finds the frame schedule | The result is difficult to explain, and a time measurement is not stable | Phase 4 step 4 is accurate |
| A server target and a wire format from the same fields | It is a second back end | The rules of the game need the field of a spell on the server |
| Cost proofs for each pair of spells | The spells do not exist | The first two spells that answer each other exist |

## 11. Questions that are open

Each question has the phase that must answer it.

1. Can a build-time constant hold some megabytes of cook data? (phase 3, step 1)
2. Can the capture or the hand-over of the host carry mesh buffers? (phase 3, step 1)
3. Which part of safe math on Metal caused the loss of 91%? (phase 0.1)
4. Does the Radeon round division and square root correctly for all inputs? (phase 1, step 4)
5. What error limit does WGSL give to `sin` and `cos`? The value 2⁻¹¹ is from memory. Read the
   specification before step 7 of phase 1.

## 12. How to keep this plan

This file is the plan. Keep it in `docs/plans/`.

- When a phase is done, write "Done" and the date at its heading. Put the measurements in the
  commit message.
- If a step or a target changes, change it here in the same change.
- When all phases are done, remove this file.
