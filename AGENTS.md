# wrela-next agent instructions

Agents write all of this MMO: the toolchain, the engine and the game. No person writes code or
makes assets. Thus the game is only as good as the tools and the checks that agents have.

## The game

The design is open. The idea today is a spellcraft game about the parry: a spell that answers a
spell at the correct time and changes it. The world is open. The camera is behind the shoulder of
the caster.

Build toward this image. One caster throws lightning. The other throws sand into its path. The
lightning melts the sand into a branch of glass that hangs in the air, glows, and then breaks.

- Do not name or copy other games, films or books in code, documents, commit messages or image
  prompts.
- Make light, materials and motion physically correct first. Then add style.
- Lore goes in `docs/lore.md` (not written). Lore changes to fit the game.

## Three layers

| Layer     | Where        | Its job                                                         |
| --------- | ------------ | --------------------------------------------------------------- |
| Game      | `game/`      | What is in the world and what it does                           |
| Engine    | `lib/`       | How to draw, move, light and simulate things                    |
| Toolchain | `toolchain/` | The compiler and all work before the frame: types, costs, cooks |

Dependencies point down. A game file that imports `std::gpu` is an engine bug. The toolchain does
not know the things of this game.

## How to work

**Solve each problem one time, in the engine.** The engine is for this MMO only.

- If a scene needs a correction, repair the engine fault that all scenes have.
- Do not add a constant, a special case or a flag for one scene or one caller. A control must
  have a physical meaning in a general system.
- When a new method lands, remove the old method in the same change.
- The obvious code of an author must be correct and fast.

**All content is a field:** a function over a domain, such as a distance, a density or a height.

- The game says what a thing is. The engine selects how to draw it.
- Do not write meshes, textures or vertices. The engine makes them from the fields.
- A new type of thing is a new field or a new type of field, not a special asset path.

**Build the check with the system.**

- Change a judgement by eye into a number.
- Give each system the means to examine it: diagnostics, queries, screenshots to compare, and a
  slow reference for the fast path.
- If you made a guess that something works, build the missing check next.
- Put measurements in the commit message: before and after, same machine, same run.

**Work until it is good.**

- Work is done when the picture is correct, the measurement is in its target and the gate
  passes. It is not done before that.
- If a result is not good, find the cause and use a different method. Do not stop. Do not revert
  and leave the problem open.
- A difficult target is still the target. Say how far the work is from it. Then continue.

## Rules

1. **Cite each outside source in `docs/citations.md`:** what it is, its rights, and how and why
   we use it. If a change makes a line there wrong, correct the line in the same change.
2. **Start each GPU run with `tools/gpu`.** Do not run the host by a different method. Only one
   run can use the GPU at a time.
   - Keep each dispatch below approximately 50 ms, or draw the work as a render pass. You cannot
     stop a long dispatch, and two of them made the machine fail.
   - Do the first run of new work at a small size with `--frames 2 --timestamps --serial`. Read
     the times before a larger run.
   - Do not increase `WRELA_GPU_BEHIND`.
   - Do not open a full-screen or borderless window. One hung the Mac.
   - On the Mac, keep display sync on. Open a window only after the user says yes to that run.
     Measure windows first on the headless Weston of the Linux machine.
3. **Run one Rust build at a time.** The machine has 16 GB. Each cargo command uses
   `toolchain/target` and waits for other cargo commands.
4. **Use the two machines.** This Mac (the M4) uses Metal. The Linux machine
   (`ssh wrela-popos`) uses Vulkan with SPIR-V.
   - Measure a change to how the engine draws on the two machines before you call it done.
   - The Linux machine is a second place to work. If the GPU of the Mac is busy, do the run
     there.
   - `tools/radeon COMMAND` runs a command in the Linux checkout (`~/projects/wrela-next`) at
     the pushed head of your branch.
   - Move work only through GitHub: commit, push, then `git pull` there. Do not copy files.
   - The same rules apply there. Touch nothing else on that machine.
5. **Pass `tools/gate` on the two machines before you call a change to how the engine draws
   done.** It holds the hole audit, the judge of the sun and the 1080p frame time to
   `tools/gate.json`.
   - You can commit and push work in progress before the gate passes. Do not merge it to `main`.
   - Use `--only` or `--quick` while you work. Use the full gate before you call the work done.
     Add `--full` for a change to the mesher or to the passes of the eye.
   - Put the results of the gate in the message of the commit that completes the change.
   - A phase of a plan is done only when the work is in its target.
   - If the commit that you start from has a failure, find the commit that caused it. Read
     `tools/gate --seen A..B` first: it shows the results that the gate kept. Bisect only the
     commits that it did not see.
6. **Commit and push freely to `origin` (`github.com/rywible/wrela-next`). Do not open pull
   requests.** A commit message is one sentence that says what is now true, with no period at
   the end.
