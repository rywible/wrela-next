# Plan: one body declaration and one movement solver

Status: approved by the owner on 2026-11-25, with a limit of 300 compute units. Phase 0 is in
work. Section 5 has a table that shows what is done.

This plan contains no new measurement. Each fact about the repository comes from the code at
`342c2e6`. Each fact about an outside source comes from the page in section 15. Where a fact is
not verified at its source, the plan says so.

A target with the words "first value" is an estimate. Phase 3 sets its final value from the
calibration. After that, a target changes only with a reason in the commit message.

## 1. Purpose

Agents cannot write movement that looks alive. Movement of a body comes from physics and from
recorded movement, not from curves that an author writes. This plan gives the engine the means to
make movement.

- The game says what a body is: its segments, its joints and its freedom.
- The game says what a body must do: a small number of goals, each with a time.
- The engine makes the pose for each tick.
- A net does the fast work. Physics teaches the net and measures the result.
- The released game contains the weights and their record. It contains no capture file.

The plan has two parts. Phases 1 to 3 change how the engine describes a body and how it measures
movement. They are useful with no net. Phases 4 to 6 use Colab to make the weights.

### What the earlier attempt showed

The branch `train_character_animation` (2026-11-22, not merged) made a net for movement. It
shows three faults that this plan must prevent.

- The net learned from one walk cycle. It could only play that cycle again.
- The inputs of the net named a book and a table of one scene.
- The judge measured slide, penetration and joint limits. A body that does not move passes all
  three.

Do not merge that branch. Do not copy its files.

### What is in the repository today

| Fact | Where |
| --- | --- |
| The kit knows only the humanoid: 55 bone names, and a pose of 64 free rotations | `lib/kit/motion/humanoid.wrela` |
| A second skeleton with 57 bones, and a hand-written table that maps the first to it | `game/world/skeleton.wrela` (`HAKURO_BONE`, `on_rig`) |
| The creature uses 16 of those bones, with hand-written tables from each part to its bones | `game/world/creature.wrela` (`CREATURE_RIG_BONES`, `PART_BONES_START`, `PART_BONES_END`) |
| The skin has a fixed limit of 24 bones | `lib/std/field.wrela` (`MAX_SKIN_BONES`) |
| The arm twist is a special case that names four bones | `lib/kit/motion/humanoid.wrela` (`SPREAD`, `TWIST_STEPS`) |
| No joint has a type, an axis, a limit, a mass or a strength | All of the above |
| No field states a physical density. A substance states only how matter answers light | `lib/engine/matter/substance.wrela` |
| The only movement is one walk cycle, played in place | `game/rules/walk.wrela`, `lib/kit/motion/cycle.wrela`, `lib/kit/motion/playback.wrela` |
| The engine has no solver for bodies with joints | `lib/` |

## 2. Terms

| Term | Meaning |
| --- | --- |
| Body | A tree of segments |
| Segment | One rigid part of a body: a field with a density |
| Joint | The connection of a segment to its parent |
| Degree of freedom | One number that a joint can change |
| Range field | A function over the degrees of freedom of a joint. It is negative in the range of the joint and positive out of the range |
| Strength | The largest torque that a degree of freedom can give |
| Pose | One value for each degree of freedom of a body, and the place of its root |
| Goal | One thing that a body must do, with the tick at which it must be true |
| Solver | The part of the engine that makes a pose for each tick |
| Net | A function with weights |
| Weights | The numbers of a net. A training run finds them |
| Capture | The recorded movement of a real person or animal |
| Retarget | To change a capture so that a different body can play it |
| Simulator | An outside program that calculates the physics of many bodies on a GPU |
| Tracker | A net that moves a body in the simulator so that the body follows a capture |
| Filtered capture | The movement that the tracker made when it followed a capture |
| Training run | A run on Colab that finds weights |
| Judge | The engine code that measures movement |
| Root residual | The force at the root that a movement needs and that contact with the world cannot give |
| Provenance record | A file that names the data, the code, the seed and the machine that made one file of weights |

## 3. Rules for all phases

1. Keep the layers. The game states bodies and goals. The engine holds the solver and the judge.
   The toolchain gets only general mechanisms.
2. Do not put the name of a body, a scene or a prop in the solver, the judge or the inputs of a
   net. An input must have a physical meaning for all bodies.
3. Keep the judge independent. It reads only the poses, the body declaration and the field of the
   world. It does not read the solver, the tracker or the weights.
4. Use a source for a training run only if `docs/citations.md` has its row and its rights permit
   the use. Add the row in the same change that first uses the source.
5. Give each file of weights a provenance record. Keep the two files together in git.
6. Keep capture files and the output of training runs in a directory that git ignores. No file
   in `game/` or `lib/` reads them.
7. Write each tool that reads the body declaration in wrela. Do not read a `.wrela` file with a
   script.
8. When a new method replaces an old method, remove the old method in the same change.
9. Work on each phase until each line of its "Done when" list is true. If a result is not good,
   find the cause and use a different method. Put each measurement in the commit message.
10. Obey the rules in `AGENTS.md`. Start each GPU run with `tools/gpu`. Measure on the two
    machines. Pass `tools/gate` on the two machines before you call a phase done.

### Rules for Colab

An agent starts each run with the `colab` command. No person starts a run. Section 5 (0.4 and
0.5) gives the GPU type, the faults of Colab and the run tool.

1. Use the L4 GPU with MuJoCo Warp. Start each run with the run tool of 0.5, not with
   `colab run`.
2. Do the first run of new work at a small size. Read the times before a larger run.
3. Write a checkpoint not less than one time each 10 minutes. A run must continue from its last
   checkpoint after the session stops.
4. Set the seed of each run and write it in the provenance record.
5. Put no secret in a script.
6. Keep each script in git. Keep the output out of git, but not the weights.
7. Read `colab usage` before and after each run. Put the compute units that the run used in the
   commit message.
8. Do not use more compute units than the limit that the owner gave: 300 for this plan. Write
   each run in the table of units in 0.5. Stop each session when its run is done.
9. Before a run, estimate its compute units from the measured rate. If the estimate is more than
   the units that remain in the limit, do not start the run. Tell the owner.

## 4. Order of the phases

| Phase | Result | Uses the result of | Uses Colab |
| --- | --- | --- | --- |
| 0 | Two decisions, the rights of each source and two measurements | Nothing | A small run |
| 1 | One declaration for each body, with its freedom (done on the Linux machine, 2026-11-25) | Nothing | No |
| 2 | Capture with clean rights, on our bodies | Phases 0 and 1 | No |
| 3 | A judge that gives numbers for movement | Phases 1 and 2 | No |
| 4 | A tracker, and filtered capture | Phases 2 and 3 | Yes |
| 5 | The solver in the engine | Phases 3 and 4 | Yes |
| 6 | Bodies and ground that have no capture | Phase 5 | Yes |

Reasons for this order:

- Phase 1 is first because all other phases read the body declaration. The judge needs the mass
  and the axis of each joint. The net takes its inputs and its outputs from the degrees of
  freedom.
- Phase 3 is before all training runs because the judge decides if a training run is good. The
  earlier attempt had no such judge.
- Phase 4 is before phase 5 because the tracker makes the data that the solver learns from.
- Phase 6 is last because it has the largest risk. It also uses each part that the phases before
  it make.

Phases 1 to 3 are useful if the owner stops the plan after them.

## 5. Phase 0: decisions, rights and two measurements

Write no engine code in this phase.

### 0.1 Two decisions of the owner

The owner made these decisions on 2026-11-25. Write them in `docs/citations.md` in this phase.

1. **The rule for what ships.** `docs/citations.md` says that no generative model makes
   something that ships. Weights from our own training runs can ship.
2. **A commercial release.** The game must be sold. The repository has the MIT licence, and the
   rights of each source must agree with it.

### 0.2 The rights of each source

The decisions of 0.1 give this rule. A source is sufficient only if its terms permit commercial
use and changed copies. A duty to give credit is permitted.

- Permitted: the public domain, CC0, MIT, BSD, Apache 2.0 and CC BY.
- Not permitted: terms that say "non-commercial", "no derivatives", "share-alike" or "research
  only".

One point is not certain. No lawyer has examined it. A CC BY source stays CC BY. Its duty to
give credit goes with the weights, and the MIT licence does not replace it. The repository does
this today for the walk of ACCAD.

A source with terms of its own that limit what a person can do with the data is not sufficient.
The owner decided this for the CMU set on 2026-11-25: its terms do not permit a person to sell
the data again, "even in converted form".

Done on 2026-11-25. Three searches read the terms of each source at the source. The result is
the register `docs/plans/movement-sources.md`. It is the list of sources for phase 2. A source
gets its row in `docs/citations.md` when a phase first uses it.

What the register shows:

| Movement | Sources that the rule permits | Effect on this plan |
| --- | --- | --- |
| Walk and run in many styles | 100STYLE and ACCAD | The base of phase 4 |
| Arms, hands and fingers | A CC0 archive of 2,710 sign movements, and two other sets with fingers | Fingers come into phase 2 |
| Strikes and fast actions | A CC0 set of 3,229 karate punches and kicks. A small parkour set | The nearest capture to a cast and a parry. No source has a sword or a staff |
| Gestures, idle, crouch | 417 clips with the MIT licence | Phase 2 |
| Stairs, slopes, rough ground, obstacles | More than 10 sets, most with measured ground forces | Phase 6 has capture for ground that is not flat |
| Pushes and recovery | 8 sets | Phase 4 can teach the tracker to stay up |
| Measured ground forces | Most laboratory sets | Phase 3 can compare the judge with a measurement |
| A dog, a horse or a cat as a skeleton | None | Phase 6 uses published numbers, as planned |
| Rats and birds | 3 rat sets and 5 bird sets | A check of the method of phase 6 on a real animal |

The CMU set is refused by the decision of the owner. Its terms do not permit a person to sell
the data again, "even in converted form".

### 0.3 The cost of a net on the CPU

Question: how many bodies can the solver move in one tick?

1. Write a net with random weights in wrela. It has three layers.
2. Make three sizes: 256, 512 and 1024 units in each layer.
3. Measure the time for one evaluation on the two machines, as WASM with strict floats.
4. Write the times in the commit message. They answer question 3 of section 14.

Result: the number of bodies at the full rate for a CPU budget of 1 ms for each tick.

Done on the Linux machine on 2026-11-25 (`tools/movement/netcost`, run with
`tools/wrela tool tools/movement/netcost`). The net has 256 inputs, three layers of n units with
ELU, and 64 outputs. Its weights are in rows of four (`vec4`), and four rows are summed at a time.
The median of 21 rounds of 20 evaluations, three runs, AMD Ryzen 5 5500U, one thread:

| n | Weights | One evaluation | Bodies in 1 ms |
| --- | --- | --- | --- |
| 128 | 73,728 | 31 µs | 31 |
| 256 | 212,992 | 86 µs | 11.6 |
| 512 | 688,128 | 254 µs | 3.9 |
| 1024 | 2,424,832 | 1,110 µs | 0.9 |

The simplest loop (one row at a time) took twice as long: 171, 520 and 1,829 µs for 256, 512
and 1024. The M4 is not measured: the owner gave this work to the Linux machine alone. Measure it
there before phase 5 sets the rates.

What it means for phase 5: one thread moves approximately 11 bodies at the full rate with a net
of 256, and 31 with a net of 128. Bodies far from the eye must get a pose at a lower rate (step
10), or the solver must use the workers (`std::par`).

### 0.4 The speed of the simulator on Colab

Question: is Colab sufficient for phase 4, and which accelerator is best?

Done on 2026-11-25, but the two lines of "Not measured" below.

Each run stepped the humanoid that comes with the benchmarks of MuJoCo Warp (27 degrees of
freedom, a step of 0.005 s, the Newton solver with 100 iterations) for 1000 steps, with no net.
The GPUs ran MuJoCo Warp 4.0.1. The TPU ran MJX with JAX 0.11.2, because MuJoCo Warp runs on
NVIDIA GPUs only. The scripts are `tools/movement/sim_speed.py` and
`tools/movement/sim_speed_jax.py`.

Steps in one second, for each number of bodies:

| Accelerator | Units in one hour | 256 | 1,024 | 4,096 | 16,384 | Steps for one unit |
| --- | --- | --- | --- | --- | --- | --- |
| L4 | 2.33 | 560,146 | 1,650,991 | 2,514,086 | 2,111,676 | 3.9 billion |
| T4 | 1.24 | 297,491 | 762,114 | 1,077,946 | 1,121,625 | 3.3 billion |
| A100 | 8.16 | 367,642 | 1,234,368 | 2,829,722 | 4,211,051 | 1.9 billion |
| TPU v5e | 3.26 | Not measured | 306,412 | 254,734 | 204,152 | 0.34 billion |

**Decision of the owner on 2026-11-25: use the L4, with MuJoCo Warp.**

- The L4 gives the most steps for one unit. It is 2.2 times as fast as the T4.
- Use approximately 4,096 bodies. The L4 was slower with 16,384.
- The T4 is the second selection, if Colab has no L4.
- The A100 is 1.7 times as fast as the L4 and uses 3.5 times the units. Use it only if a run
  must end in one session.
- Do not use a TPU. With the same model, the v5e was 8 times slower than the L4 and used more
  units. It did not have sufficient memory for 65,536 bodies.

The licence of MuJoCo Warp 4.0.1 and of Warp 1.19.0 is Apache 2.0 (read in the data of each
package). Add their rows to `docs/citations.md` in the change that first uses them.

What the limit gives: 300 units are approximately 129 hours of an L4, or 1.2 million million
steps with no net.

Estimate for the later phases. It is not a measurement. The numbers of samples come from memory
of published work, and a net in the loop makes each step slower.

| Phase | Assumption | Hours of an L4 | Units |
| --- | --- | --- | --- |
| 4 | 5 to 20 thousand million control steps, at 250,000 in one second | 6 to 22 | 13 to 52 |
| 5 | Learning from the filtered capture, with no simulator in the loop | 4 to 9 | 10 to 20 |
| 6 | Two bodies and ground that is not flat, each as phase 4 | 21 to 43 | 50 to 100 |

The first small run of phase 4 measures the control steps in one second with the net. Calculate
this table again from that measurement before a large run.

Not measured:

- How long a session stays alive.
- A run that stops and continues from its checkpoint. The run tool of 0.5 must show this.

Not measured, and not necessary: the TPU v6e (the run did not start before it was stopped), and
JAX on the L4. The v5e ran a different, light model at 1.7 million steps in one second, but that
model sets the solver to 1 iteration, so the number does not compare with the table.

### 0.5 The faults of Colab, and the run tool

The measurements of 0.4 were short, and they found these faults. A long training run will find
them again. The log of the `colab` command is `~/.config/colab-cli/colab.log`.

| Fault | What occurred on 2026-11-25 | What the run tool must do |
| --- | --- | --- |
| The connection is lost | An A100 session became ready. Then `colab run` gave "Connection was lost" and stopped the session. The second try was good | Try again, with a longer wait each time. Count the tries |
| A session does not start | `colab run --tpu V6E1` showed "Creating session" and then gave no more output for minutes | Stop a start that takes more than a set time. Try again |
| Colab refuses a session | With three sessions in work, the fourth got "Allocation refused (precondition failed)": HTTP 412. The cause is not known: the number of sessions, or the capacity | Use one session at a time. Wait and try again |
| A session stays alive with no owner | The local command was stopped. One TPU session stayed alive and used 3.26 units in one hour until `colab stop` | Read `colab sessions` when the tool starts and when it ends. Stop each session that is not in use. Stop the session if the tool gets a signal |
| The run depends on this machine | `colab run` holds the connection. If the laptop sleeps or the network drops, the run is lost | Start the work in the session so that it continues with no connection. Connect again to read its state |
| No output looks like a failure | `colab run` stops a run after 30 s with no output. The first step of MuJoCo Warp compiles for 60 s, and an install can be silent | Print a line not less than one time each 20 s. Set `--timeout` for the longest silent step |
| Each new session starts cold | Each new VM installs the packages (15 s) and compiles the kernels (60 s) | Count this time in the estimate. Do not use many short sessions |
| The balance is late | `colab usage` showed the same balance after a run. It shows the rate only while a session is alive | Read the rate while the session is alive. Calculate the units of a run from the rate and the time |
| A run gives no result and no error | `sim_speed.py mjx` on an L4 printed the model and then no speed. The cause was not examined | Examine the full output of a run. A run with no result line is a failure |

The length of a session is not known. Thus a run must not depend on it.

The tests of the run tool on 2026-11-25 found three more:

| Fault | What occurred on 2026-11-25 | What the run tool does |
| --- | --- | --- |
| A start that was cut still gets a VM | A `colab new` cut at 3 s left an L4, then a T4, with no session name ("[?]" in `colab sessions`). `colab stop` cannot name it | After each failed start, wait 10 s, read Colab's assignments, and release each new one with no local record. `sweep --orphans` releases all such |
| One session of each accelerator | While an L4 with no name was alive, each start of an L4 got HTTP 412. The tool asked for a T4, then a second orphan held the T4 the same way | The release above. A run on the T4 does not give the same bits as one on the L4 |
| A command hangs | One `colab exec` in five hung for more than 60 s, and one launch hung for 120 s | Cut each poll at 60 s and try again. A launch that hangs stops the session and starts a new one |

#### Steps

1. **Write the run tool** `tools/movement/colab-run`. It is the only method that starts a
   training run. It does what the table says.
2. **Keep the checkpoints out of the VM.** Mount Google Drive in the session, or download each
   checkpoint to this machine. A checkpoint in the VM only is lost when the session ends.
3. **Make the run continue.** The tool starts a new session and gives it the last checkpoint.
   The training script starts from that checkpoint.
4. **Keep the table of units below.** The tool adds a row for each run.
5. **Examine the tool with a small run** on an L4: 10 minutes of steps with a checkpoint each
   minute.

#### Done when

Done on 2026-11-25. The tool is `tools/movement/colab-run`, with `tools/movement/runlib.py` in the
VM. Its test job is `tools/movement/run_test.py`: a deterministic training of a small net, 600
steps of 1 s, a checkpoint each minute, after an install of MuJoCo Warp and its first steps (the
cold start: 70 s).

- Done: the local tool is stopped in the middle of the small run (`kill -9` at 4 minutes). The
  run continues in the session, and `colab-run follow` connects to it again.
- Done: the session is stopped in the middle of the small run. The tool starts a new session,
  and the run continues from its last checkpoint (step 184) to the same end result: the SHA-256 of
  the weights of run `l4-c` is the same as that of run `l4-a`, which did not stop
  (`89a7ca28...`).
- Done: after each of these tests, `colab sessions` shows no session.
- Done: the tool stops a start that hangs, and tries again (each start cut at 3 s, then 6 s; the
  VM that a cut start gets is released).
- Done: a run does not depend on the length of a session: it continues in a new one. The longest
  session of the tests was 10 minutes, so the length itself is not measured.

#### The table of units

The limit is 300 units for this plan. The balance was 413.09 on 2026-11-25, before the first
run. Other work can use the same account, so use this table, not the balance, to know what this
plan used. The tool writes a row as the rate of the account times the time of each session: more
than the balance fell (1.59 written, 1.12 fallen for the tests of 0.5).

| Date | Run | Units |
| --- | --- | --- |
| 2026-11-25 | The measurements of 0.4: T4, L4, A100 and TPU v5e | 0.13 |
| 2026-11-25 | `smoke1` (CPU): smoke test of the run tool, done | 0.00 |
| 2026-11-25 | `smoke3` (CPU): smoke test: the same run with no stop, done | 0.00 |
| 2026-11-25 | `smoke2` (CPU): smoke test: a session stopped in the run, done | 0.01 |
| 2026-11-25 | `l4-a` (L4): the run tool's test: 600 steps of 1 s, no stop (the reference), done | 0.46 |
| 2026-11-25 | `l4-b` (L4, then T4): the run tool's test; a start cut at 3 s left an L4 and a T4 with no session name, which held the account's one of each, so the run lost its session and found none (the balance's fall, the two orphans included), failed | 0.43 |
| 2026-11-25 | `l4-c` (L4): the run tool's test: the tool killed, the session stopped, a start cut at 3 s, done | 0.69 |

#### A new machine

- Install the command: `uv tool install google-colab-cli`. The version of 2026-11-25 is 0.8.0.
- The owner signs in to Colab one time. Then `colab usage` shows the balance.
- Before work, `colab sessions` must show no session.

### What is done in phase 0

| Part | State on 2026-11-25 |
| --- | --- |
| 0.1 The decisions of the owner | Done, and written in `docs/citations.md` |
| 0.2 The rights of each source | Done: the register |
| 0.3 The cost of a net on the CPU | Done on the Linux machine. The M4 is not measured |
| 0.4 The speed of the simulator | Done. The length of a session is not measured; 0.5 shows that a run does not depend on it |
| 0.5 The run tool | Done on 2026-11-25 |

### Done when

- `docs/citations.md` contains the two decisions. The register contains the verified rights of
  each source. (The register is done.)
- The commit message gives the time of each net size on the two machines. (Done on the Linux
  machine. The M4 is not measured.)
- The plan gives the simulator speed and the compute units in one hour for each accelerator.
  (Done: 0.4.)
- This plan gives the estimated hours and compute units for phases 4, 5 and 6. (Done as an
  estimate: 0.4. The first run of phase 4 replaces it.)
- Each line of "Done when" of 0.5 is true. (Done.)
- The owner gave a limit of compute units for the plan. (Done: 300.)

## 6. Phase 1: one declaration for each body

Done on the Linux machine on 2026-11-25, but for the M4 and one target ("What is done in phase 1",
below).

Purpose: the game states each body one time. The engine calculates all other data about the body.

### Steps

1. **Give matter a physical density** in kg/m³, as a channel of a field. Keep it apart from
   `Substance`, which states only how matter answers light. Cite the source of each value.
2. **Add the body types to the kit.** A body is a tree of segments. A segment is a field. A
   joint connects a segment to its parent. The root has six degrees of freedom.
3. **Add the joint types.** Start with four: fixed, hinge (one degree of freedom), universal
   (two) and ball (three). Each joint has its place in the parent, its axes, its range field and
   the strength of each degree of freedom.
4. **Write the range field as a function.** For a hinge it is an interval. For a ball joint it is
   a function of the swing direction and the twist. The compiler gives its gradient.
5. **Make the pose a list of degrees of freedom.** The build knows the length of the list. A
   knee has one number, so it cannot bend to the side.
6. **Write one function from a pose to the place of each segment.** All engine code uses it.
7. **Calculate the mass, the centre of mass and the inertia of each segment at build.** Integrate
   the density across the field of the segment. Make a slow sum of samples as the reference.
8. **Calculate the skin from the declaration.** The engine finds the segment of each part. The
   limits on the number of bones come from the body, not from a constant.
9. **Make the twist a degree of freedom.** The roll of a forearm is a hinge along its length.
   One general rule spreads a twist along the skin of a segment.
10. **Declare the creature one time** with the new types. Calculate its palette from its pose.
11. **Change the walk cycle to degrees of freedom** with a tool, one time. The creature must
    still walk at the end of this phase. Phase 5 removes the cycle.
12. **Declare a second body with four legs and a tail.** Draw it at rest. This shows that the
    types are not only for a humanoid.
13. **Remove the old descriptions in the same change:** the bone names, `Rig`, `RigBone` and the
    old `Pose` in the kit, `HAKURO_BONE`, `on_rig`, `CREATURE_RIG_BONES`, `PART_BONES_START`,
    `PART_BONES_END`, `SPREAD` and `TWIST_STEPS`.

### Where it is

- `lib/engine/matter/density.wrela`: `Density`, a channel apart from `Substance`.
  `lib/kit/substances/densities.wrela`: tissue and wood, each with its source.
- `lib/engine/body/`: `joint` (the four kinds, the range field and its gradient, a kind's turn
  and its inverse), `body` (`Body`, `Pose`, `Span`, and `Body::places`, the one function from a
  pose to each segment's frame), `mass` (the fast integration and the slow grid), `skin` (the
  skin from the declaration, `Cone`), `sweep` (the sweep of each joint, and `cleared`, which cuts
  a range where the body's own flesh stops it), `fit` (a damped least-squares fit of a pose to
  aims, for phase 2's retarget).
- `std::field::Skinned<S, B, P>` takes its numbers of bones and parts from the body: no
  `MAX_SKIN_BONES` or `MAX_SKIN_PARTS` (question 4: yes, an array's length can be a body's
  constant).
- The creature: `game/world/creature_segments.wrela` (written once by `tools/movement/onetime`
  from the old rig), `game/world/creature.wrela`. Its walk: `game/rules/walk.wrela`, a Fourier
  series of each of its 55 numbers, played by `lib/kit/motion/cycle.wrela`.
- The hound: `game/world/hound.wrela`, four legs and a tail of three, drawn at rest in the hole
  audit (its 16th kind).

Decisions that the steps did not give:

- The body types are in the engine (`lib/engine/body`), not the kit: the judge is in the engine
  and reads them, and the kit depends on the engine.
- A universal joint is a flex about x and then a twist about y, the segment's length: an elbow's
  bend and its forearm's roll, a knee's bend and its shin's turn. A knee is universal, not a
  hinge: the old walk turns the shin about its length up to 0.6 rad, which a hinge can't do. It
  still can't bend to the side. An ankle is a ball joint (the ankle and the joints of the foot
  below it).
- The twist of a universal joint spreads along the skin of each part on its segment alone (a
  second bone: the segment's frame with the bind pose's twist). The twist of a ball joint turns
  its segment whole.
- A range is a person's (or a dog's) from a published table, then cut where the body's own flesh
  stops it: each number moved alone from the bind pose until a part that moves with it goes 1 mm
  into a part of the segments it hangs from. One limb against another (the thighs as a leg
  crosses) is a collision, which the judge measures; it does not cut a range.
- A segment with no part of its own (each collarbone) has no mass. Phase 4 must give such a
  segment to the simulator in a form it takes.

### Checks to build in this phase

- The fast mass calculation against the slow sum of samples.
- A sweep of each joint through its range. It finds where two segments that are not neighbours
  go into each other.
- The pose function against the old palette, bone by bone, before the old code is removed.
- The mass calculation against a published table, for a test body of human form. The register
  gives the tables that have open rights.

### Done when

- Done: each body has one declaration. A search for each removed name finds nothing (the bone
  names, `Rig`, `RigBone`, the old `Pose`, `HAKURO_BONE`, `on_rig`, `CREATURE_RIG_BONES`,
  `PART_BONES_START`, `PART_BONES_END`, `SPREAD`, `TWIST_STEPS`, and `MAX_SKIN_BONES`,
  `MAX_SKIN_PARTS`, `pack_part_bones`; `humanoid.wrela`, `playback.wrela`, `gaits.wrela`,
  `rig.wrela` and `skeleton.wrela` are removed).
- Not met, by the decision above: frames 0 and 1 of SCENE 0 and SCENE 2 are the same as before
  but at the creature's outline. The old walk bends each knee to the side, which a knee of the new
  body can't: each part's ends are where the old palette put them to 5.2 mm at most (the left
  knee; 2 mm or less elsewhere but the feet and knees). By `tools/same` against a second run of
  the old build, at 1920x1080 on the Linux machine: SCENE 0 frame 0 has 185 pixels over 8/255
  (0.009%; the noise 3), frame 1 has 480 (0.023%; the noise 0); SCENE 2 frame 0 has 574 (0.028%;
  the noise 0), frame 1 1,502 (0.072%; the noise 3); each on the outline of the legs, the ears and
  the wand. The two pictures are otherwise the same. Not measured on the M4.
- Done: the mass of each segment agrees with the slow reference to 1%, but for a segment of less
  than 1% of the body, to 1% and 0.05% of the body (`creature.wrela`'s test). The creature's neck
  is a wedge of 7.6 g between the head's part and the chest's bone; grids of 3, 2 and 1 mm weigh it
  at 7.2, 8.0 and 7.4 g. Each other segment agrees to 0.5% or better (the toes at 1 mm to 0.04%).
  The creature weighs 39.6 kg (its head 16.2 kg), the hound 35.8 kg.
- Done: no two segments that are not neighbours go more than 2 mm into each other in the range
  (1 mm at most, both bodies), as `sweep` measures them (above).
- Done on the Linux machine: the time of the skin pass is not more than before (`mesh_skin`,
  SCENE 0 at 1920x1080, frames 30 to 89, the median of two runs each: 709 and 711 µs before, 709
  and 713 after). Not measured on the M4.
- Done on the Linux machine: the body with four legs builds and passes the hole audit at rest (12
  eyes at 320x180: no holes, no extras, 4 pixels off by more than a pixel).
- `tools/gate` on the Linux machine: in the commit message of this phase. Not run on the M4.

Checks that the steps asked for, and where they are: the fast mass against the slow sum
(`each_segment_weighs_what_the_slow_sum_says`); the sweep (`no_joint_in_its_range_puts_a_part_into_another`,
`no_joint_of_the_hound_puts_a_part_into_another`); the pose function against the old palette,
part end by part end, over the 119 frames of the walk (`tools/movement/onetime`'s report, run at
the change's parent: 5.2 mm at most); the mass against de Leva's table for a body of human form
(`a_human_shaped_body_turns_as_the_table_says`); and the walk in its ranges
(`the_walk_stays_in_each_joints_range`).

## 7. Phase 2: capture with clean rights

Purpose: each source that phase 0 permits becomes poses of our bodies.

### Steps

1. **Write the import tool in wrela.** It reads capture files and writes poses of one body. The
   bone names of a capture format stay in this tool. They do not go into the kit. Take the
   sources from the register, the sets with the mark 2 first. Read the licence file in each
   download before use.
2. **Retarget by the degrees of freedom.** Fit the pose of our body to each frame of the capture.
   Keep each pose in the range fields.
3. **Scale the time for the size of the body.** A body with shorter legs moves with a shorter
   period. Scale time by the square root of the ratio of leg lengths. This keeps the Froude
   number the same.
4. **Keep the output in a directory that git ignores.** Write a record for each clip: the source,
   the file, its hash and the credit.
5. **Divide the clips into three sets:** training, test and calibration. Keep the division in
   git as a list of names.
6. **Write a table of what the data contains:** the hours of each type of movement. This table
   shows what the solver can learn and what it cannot.

### Done when

- Each permitted source is imported. The table of contents is in the commit message.
- Each frame of each clip is in the range fields. If a frame is not, find the cause: the range,
  the retarget or the capture.
- The mean distance between the joints of the capture and of our body is in the commit message,
  for each source.

## 8. Phase 3: the judge of movement

Purpose: a number says if a movement is correct and if it looks alive. No agent decides this by
eye.

The judge is in `lib/engine/judge`. It works for all bodies. It reads a sequence of poses, the
body declaration and the field of the world.

### What the judge measures

| Measure | Unit | Meaning |
| --- | --- | --- |
| Root residual | Share of the weight of the body | The movement needs a force that the world does not give |
| Torque above strength | Share of the strength | A joint is stronger than it can be |
| Slide in contact | mm/s | A part that touches the world moves along it |
| Penetration and float | mm | A part is in the world, or above it when it carries weight |
| Range | Value of the range field | A joint is out of its range |
| Goal error | mm and ticks | The body did not do the goal, or did it at the wrong tick |
| Gait numbers | No unit | The duty factor and the stride length against the speed, and the energy for each metre |
| Distance from capture | No unit | How far the statistics of the movement are from those of the capture |
| Response | ms | The time from a change of goal to the change of movement |

### Steps

1. **Write inverse dynamics on the CPU.** It gives the torque of each joint and the force at the
   root for a sequence of poses.
2. **Find the contact forces.** A part touches the world if its distance to the field of the
   world is near zero. Find the forces that make the root residual smallest. A contact can only
   push, and its force stays in the friction cone. Limit the number of steps of this search.
3. **Write each measure of the table.**
4. **Compare the judge with measured forces.** Use a source of the register that has force
   plates. Calculate the ground force of each step with the judge. Compare it with the measured
   force.
5. **Make the calibration set** from the calibration clips of phase 2. Add damaged copies of each
   clip: a faster time, a slide of the root, noise on the joints, a root that is too high and a
   body that does not move.
6. **Calibrate.** The judge must give a better result to each clip than to its damaged copies.
   More damage must give a worse result.
7. **Set the final value of each target** in this plan from the calibration.
8. **Add the judge to `tools/gate`** and its limits to `tools/gate.json`.
9. **Write a film of a movement to a file.** It shows one body from the side and from the front,
   with its contacts and its root residual drawn. Open no window. Phase 5 uses this for the one
   review by the owner.

The calibration needs no person. Thus the judge can be wrong in a way that only an eye finds. The
review in phase 5 is the check of that.

### Done when

- The ground force that the judge calculates agrees with the measured force. The mean error is
  not more than 10% of the weight of the body (first value).
- The judge gives the better result to the capture in not less than 95% of the pairs of a clip
  and a damaged copy (first value).
- The result gets worse with more damage, for each type of damage.
- The walk cycle of today, played in place on a body that moves, fails the slide measure.
- The body that does not move fails the goal error and the distance from capture.
- The judge measures one body for one second of movement in less than 50 ms on the two machines
  (first value).
- Each target in phases 4 to 6 has its final value.

## 9. Phase 4: the tracker

Purpose: a net moves our body in the simulator so that the body follows each capture. Physics
removes the faults of the retarget: slide, penetration and forces that cannot occur.

### Steps

1. **Write the export tool in wrela.** It writes the model of a body for the simulator from the
   body declaration. No person and no script writes the model by hand.
2. **Compare the two models.** The mass and the inertia of each segment in the simulator must
   agree with the engine.
3. **Compare the judge with the simulator.** Calculate inverse dynamics for three clips with
   each. This makes the simulator the slow reference for the judge.
4. **Write the training script.** It obeys the rules for Colab in section 3. The run tool of
   0.5 starts it, on an L4.
5. **Do the first run at a small size:** one clip, the walk, with 256 bodies. Read the times.
6. **Train the tracker.** The tracker reads the state of the body and the next poses of the
   capture. It gives a target for each degree of freedom. The reward is how near the body stays
   to the capture. A run stops a body that falls.
7. **Increase the set of clips** in steps. Find the clips that the tracker cannot follow. Give
   them more of the training run.
8. **Write the filtered capture.** For each clip, record the movement of the body in the
   simulator, with the contacts.
9. **Add the rows to `docs/citations.md`:** the simulator, the training library and the methods.

### Done when

- The tracker follows not less than 95% of the clips, test clips included, to the end with no
  fall (first value).
- The mean distance of a segment from its place in the capture is not more than 50 mm (first
  value).
- The simulator and the engine agree on each mass to 0.1%.
- The judge and the simulator agree on the torques of the three clips to 5% (first value).
- The filtered capture has less slide and a smaller root residual than the retargeted capture,
  by the judge.
- A stopped run continues from its checkpoint and gets the same result.
- The provenance record names the data, the commit, the seed, the simulator and the GPU.

## 10. Phase 5: the solver

Purpose: the engine makes the pose of each body for each tick from its goals. The creature walks
through the world with no cycle and no clip.

### Steps

1. **Add the goal types to the engine.** The first types are: a speed and a direction of the
   root, a direction to face, a part at a point at a tick, a part on a field at a tick and a
   point to look at. The rules of the game give the ticks.
2. **Write the evaluation of a net in wrela.** The build embeds the weights. Compare its output
   with the training library for 1000 inputs.
3. **Define the inputs for all bodies.** They are the pose and its rate, the goals in the frame
   of the body and samples of the field of the world at the parts that can touch it. The engine
   calculates the layout from the body declaration.
4. **Define the output.** It is the change of the pose and the contacts. Map each degree of
   freedom into its range, so that the net cannot give a pose out of range.
5. **Train on the filtered capture.** Make the goals of each example from the later frames of
   the same clip. Show some goals and hide the others.
6. **Train with the output of the net as its next input.** If not, small errors increase with
   time.
7. **Give the net a random input with a seed.** If not, the net gives the mean of many
   movements, and the mean does not look alive. Calculate the seed from the state, so that each
   client gets the same pose.
8. **Start with the smallest net of 0.3.** Use a larger net only if the judge shows that it is
   necessary.
9. **Correct the contacts after the net.** Put each part in contact on the field of the world.
   Keep each joint in its range.
10. **Set the rate by distance.** A body near the eye gets a pose at each tick. A body far from
    the eye gets a pose at a lower rate. The CPU budget of 0.3 sets the rates.
11. **Send goals on the network, not poses.** Each client calculates the same pose.
12. **Use the tracker as the slow reference.** For each new file of weights, the tracker follows
    the output of the solver in the simulator. Write the result in the provenance record.
13. **Show that the solver does not only play clips.** Search the output of the solver for each
    clip of the training set. The test fails if the solver gives a clip back.
14. **Move the creature with the solver in each scene.** Remove `lib/kit/motion/cycle.wrela`,
    `lib/kit/motion/playback.wrela`, `game/rules/walk.wrela` and `game/rules/gaits.wrela` in the
    same change. Correct the rows of `docs/citations.md`.

### The goal test

Make a fixed list of goals that are not in the training set: start, stop, walk, run, turn, step
to the side, put a hand at a point at a tick, and look at a point. The judge measures each one.

### Done when

- Each measure of the judge is in its target for each goal of the test.
- The error of a timed goal is 0 ticks and not more than 20 mm (first value).
- The tracker follows the output of the solver with no fall in not less than 95% of the goal
  tests (first value).
- The poses have the same bits on the two machines.
- The CPU time for each body is in the budget of 0.3.
- The solver gives back no clip of the training set.
- `tools/gate` passes on the two machines, with the frame times in their limits.

### The one review by the owner

All lines above need no person. When they are true, make 20 pairs of films with the tool of
phase 3 step 9. One film of each pair is capture and one is the solver. The owner selects the
film that looks more alive.

- The phase is done if the owner selects the capture in not more than 14 of the 20 pairs (first
  value).
- If not, the judge has a fault. Find what the eye saw and the judge did not. Add it to the
  judge as a measure, calibrate again and train again.

## 11. Phase 6: bodies and ground that have no capture

Purpose: a body with four legs moves correctly, and each body moves correctly on ground that is
not flat. The register has no capture of a dog, a horse or a cat that the rule permits. It has
human capture on stairs, slopes and rough ground, and it has capture of rats and birds.

This phase has the largest risk. A gait that a training run finds with no capture can look
wrong. Expect more than one method before the targets are met.

### Steps

1. **Add the chain joint:** a spine or a tail with a limit on its curvature.
2. **Calculate the strength of each degree of freedom** from the cross-section of its segment.
   Cite the source of the muscle stress.
3. **Write the objective for all bodies.** The body follows a speed and a direction and uses the
   least energy for each metre. It stays in its ranges and in its strengths. The objective has
   no term for one body.
4. **Set the targets before the first run.** Take them from published measurements of animals:
   the Froude number at which the gait changes, the duty factor against the speed and the energy
   for each metre against the mass. Cite each one.
5. **Train in the simulator** at a range of speeds, as in phase 4.
6. **Train each body on ground that is not flat.** Make the ground from fields: slopes, steps
   and rough ground. For the humanoid, the tracker follows the capture of the register on such
   ground.
7. **Examine the method on a real animal.** Train a body of a rat or of a bird with the
   objective of step 3 and no capture. Compare the result with the capture of that animal in
   the register. This shows how good the method is before the owner looks at it.
8. **Train the solver from this movement,** as in phase 5.

### Done when

- The body with four legs passes each physical measure of the judge for each goal of the test.
- Its gait numbers are in the published ranges.
- Its gait changes with speed at the published Froude number, in the published tolerance.
- The humanoid passes the goal test on a slope and on steps.
- `tools/gate` passes on the two machines.

No measure in this phase says that a body with no capture looks alive. Step 7 gives a part of
the answer. The rest is question 6.

## 12. Risks

| Risk | Effect | What the plan does |
| --- | --- | --- |
| No capture has a cast, a parry, a sword or a staff | The solver does these badly | The register has strikes, signs of the hands and gestures. Phase 2 step 6 shows the gap. Question 5 |
| Movement with no capture looks wrong | Phase 6 takes a long time | Phases 1 to 5 do not depend on it |
| A Colab session stops, or its connection is lost | A long run is lost, or a session uses units with no work | The run tool of 0.5, before the first run of phase 4 |
| A net is too slow for many bodies | Bodies far from the eye move at a low rate | 0.3 measures this before the design of the net |
| The judge does not agree with the eye | A training run gets a good number and a bad picture | The calibration of phase 3, then the one review by the owner in phase 5 |
| The compute units are not sufficient | A training run cannot end | The estimate of 0.4 is 73 to 172 of the 300 units. Rule 9 for Colab |
| A licence field in a repository is wrong | The game ships weights with rights that are not clean | Read the licence file in each download. Use the sets with the mark 2 first |

## 13. Work that is not in this plan

| Idea | Why it is not here | Look again when |
| --- | --- | --- |
| Physics of bodies while the game runs | The engine has no solver for bodies with joints. It costs much for each body | Phase 5 is done and a hit must push a body |
| One file of weights for all bodies | It is research, and its results are poor today | Phase 6 is done for three bodies |
| Muscles | A strength for each degree of freedom is sufficient to start | The gaits of phase 6 are not in their ranges |
| Faces | It is a different system | The eye is near a face |
| Cloth and hair on a body | It is a different system | Phase 5 is done |
| A capture that the owner records | The rule is that no person makes assets | The owner answers question 5 |
| A capture file in the released game | The rule is that the build reads no motion file | Do not look again |
| Movement from a sentence | The game gives goals, not sentences | Do not look again |

## 14. Questions that are open

Each question has the phase or the person that must answer it.

1. Answered on 2026-11-25: weights from our training runs can ship.
2. Answered on 2026-11-25: the game must be sold, and each source must agree with the MIT
   licence.
3. What size of net is in the CPU budget? (0.3) Partly answered on 2026-11-25: on the Linux
   machine, 11 bodies a millisecond with three layers of 256, 31 with 128.
4. Answered on 2026-11-25: yes. `Skinned<S, B, P>` takes its lengths from the body (phase 1,
   step 8).
5. Where does the movement of a cast and of a parry come from? The register has strikes, hand
   signs and gestures, but no cast. The possible sources are those sets with goals, clips made
   by hand as goals for the tracker, and a new capture. (owner and phase 2, step 6)
6. What number says that a body with no capture looks alive? (phase 6)
7. Answered on 2026-11-25: the CMU set is not used (0.2).
8. Answered on 2026-11-25: a share-alike source is not used (0.2).
9. Does the server run the solver, or only the rules and the root? (phase 5, step 11)
10. How large can a file of weights be in git? (phase 5, step 2)
11. Answered on 2026-11-25: the limit is 300 compute units for the plan.

## 15. Sources

The sources of movement, with the terms of each one as read on 2026-11-25, are in the register
`docs/plans/movement-sources.md`. The phase that first uses a source adds its row to
`docs/citations.md`.

The simulator: [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) and
[MuJoCo Warp](https://mujoco.readthedocs.io/en/stable/mjwarp). Phase 0.4 reads the licence of
the version that it selects.

Published methods that the phases will cite when they use them:

- Tracking of capture with physics: X. B. Peng et al., "DeepMimic" (2018); Z. Luo et al.,
  "Perpetual Humanoid Control" (2023).
- Goals as a partial view of later frames: C. Tessler et al., "MaskedMimic" (2024).
- A random input so that a net does not give the mean: H. Y. Ling et al., "Character Controllers
  Using Motion VAEs" (2020).
- Gaits and the Froude number: R. McN. Alexander and A. S. Jayes, "A dynamic similarity
  hypothesis for the gaits of quadrupedal mammals" (1983).

## 16. How to keep this plan

This file is the plan. Keep it in `docs/plans/`.

- When a phase is done, write "Done" and the date at its heading. Put the measurements in the
  commit message.
- If a step or a target changes, change it here in the same change.
- When all phases are done, remove this file.
