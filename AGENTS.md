# wrela-next agent instructions

The goal is an amazing MMO. Agents write all of it: the toolchain, the engine, the content, the
whole game. No person hand-authors code or assets here, so the quality of the game is bounded by
the quality of the tools agents have and the loops that tell them whether their work is right.

## The game

We are still finding what the game looks and feels like. The working idea is a **parry-based
spellcraft game**, somewhere between Clair Obscur: Expedition 33 and Hogwarts Legacy. The
defining image is Dumbledore parrying Voldemort by turning the glass hurled at him into sand: a
parry is not only a block on a timing window but a spell that answers the incoming spell,
transforming it. Build toward that feeling, and treat everything else about the design as open.

**Visual style:** realistic first. Get light, materials and motion physically right, then
specialize: add style on top of a believable base, rather than starting stylized and faking
realism later.

**Lore** lives in `docs/lore.md` (not written yet). Lore is flexible. The first job is a great
game; lore can shape the game, but it bends to the game, not the other way round.

## Three layers

| Layer     | Where        | Its job                                                                                            |
| --------- | ------------ | -------------------------------------------------------------------------------------------------- |
| Game      | `game/`      | What exists in the world and what it does: fields, looks, rigs, moves, rules, spells               |
| Engine    | `lib/`       | How things are drawn, moved, lit and simulated, solved once for the whole game                     |
| Toolchain | `toolchain/` | The compiler and everything before the frame: types, effects, constants, cooks, costs, diagnostics |

Dependencies point down. The game never knows how anything is drawn; a game file that imports
`std::gpu` is a bug in the engine. The toolchain never knows this game's own things.

## The engine is where things get solved once

The engine is for this MMO, not for some other game that might exist later. But inside that
scope it is where we solve a problem once so that nobody has to think about it again.

- Build deep, reusable systems, not per-scene fixes. If a scene needs a tweak, ask what the
  engine got wrong that every scene would hit, and fix that.
- Do not accumulate one-off tuning: no magic constants for a single scene, no special cases
  keyed on a scene's name, no flags that one caller sets. If a knob is needed, it belongs to a
  general system with a physical meaning, and every scene gets the same behaviour from it.
- When a new way to do a job lands, the old one leaves in the same change. One way to do each
  job.
- An author's obvious code should be correct and fast because it calls the engine.

## All content is fields

Every piece of content is a field: a function over a domain (a distance from a point, a density
in space, a height over a plane, a colour in a direction). The game says _what_ a thing is as a
field; the engine decides how to draw it. Nobody authors meshes, textures or vertices; the
engine makes whatever it needs from the fields. A new kind of thing is a new field, or a new kind of field in the engine, not a
special asset path.

## Agents need great tools and great verification

Because agents author the whole game, tooling and verification are product work, not overhead.

- Turn what a person would judge by eye into a number wherever possible: a measurement, a
  comparison against a reference, a test that fails when it looks wrong.
- Every system ships with the means to check it: diagnostics, queries, screenshots that can be
  compared, references computed another way (on the CPU, by brute force) to check the fast path
  against.
- If you had to guess whether something works, the missing check is the next thing to build.
- Claims carry measurements: before and after, same machine, same run, in the commit message.
  Write down what would kill an idea before you measure it.

## Rules

1. **Cite your sources** in `docs/citations.md`: every outside paper, talk, codebase, crate or
   technique, with what it is, its rights, and how and why we use it. Fix a line there in the
   same change that makes it wrong.
2. **Every GPU launch goes through `tools/gpu`.** Never run the host any other way. One runs at
   a time. **No dispatch or pass may hold the GPU for long:** a compute dispatch that runs for
   seconds cannot be stopped, and twice that has taken the whole machine down. Heavy work is cut
   into parts of at most about 50 ms each, or drawn as a render pass. Run anything new first with
   `--frames 2 --timestamps --serial` at a small size and read its times before a long or large
   run. Don't raise `WRELA_GPU_BEHIND` to get a heavy kernel through.
   **Never take the display:** no full-screen or borderless-fullscreen window on either machine.
   A full-screen run with the GPU busy hung the Mac's GPU driver and WindowServer, and the
   machine had to be restarted. Windows stay ordinary windows.
3. **One Rust build at a time.** The machine has 16 GB. Every cargo command uses the one target
   directory, `toolchain/target`, and waits for any other cargo to finish.
4. **Two machines, two targets.** Metal on this Mac (the M4) and Vulkan with SPIR-V on the Linux
   machine (`ssh wrela-popos`). A change to how anything is drawn is measured on both before it
   is called done. On the Linux machine this repository lives in `~/projects/wrela-next`, sent
   from here, never edited there; the same rules hold there. Touch nothing else on it.
5. **Commit locally, never push.** A message is one sentence saying what is now true, with no
   trailing period.
