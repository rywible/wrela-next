# The MMO: Vision and Technical Requirements

This document captures the working direction for the MMO and the technical requirements that follow
from it. Part I is the game we want players to experience; Part II is what the toolchain, engine,
and server architecture must support for that game to exist.

---

# Part I: Vision

## Core Thesis

A small team—even one augmented by AI agents—cannot win a content-volume arms race against a
500-person studio, and mass-generated filler feels hollow. Instead, we build **deep, physically
honest systems** in a **high-craft world**, and let **human players** generate the culture,
inventions, fighting styles, guilds, rivalries, and legends.

The point of the game is **the discovery, craft, and mastery of magical power** in one living,
shared world.

- **We build:** the physical laws of matter, light, and magic; the parry-spellcraft combat feel;
  the crafting and spell-authoring tools; the foundational ocean world, warded trade ports, and
  anchor encounters/mysteries that set the bar.
- **Players build:** spells, counter-parries, custom wands, fighting schools, grimoires, frontier
  sanctuaries, trials, trade networks, and the world's history.

## 1. Emergent Spellcraft & Parry Combat

Magic is not "combine fire + ice to spawn a fireball with snowflakes." Spells are **real physical
manipulations of matter, motion, and energy** operating on the world's shared physical substrate:

- **Form:** shaping, carving, extruding, compressing, hollowing, shattering, and binding matter.
- **Substance & phase:** transmuting materials across space and time (`glass -> sand`,
  `water -> ice`, `stone -> molten`, `air -> dense mist`), where every substance has real physical
  properties (mass, hardness, brittleness, buoyancy, refractive index, heat capacity, conductivity,
  emission).
- **Motion & energy:** impulse, spin/vortex, pressure gradients, stasis (freezing kinetic energy and
  releasing it later), heat transfer, optical reflection and refraction.
- **Weaves (triggers & conditions):** binding operators to physical events—contact, velocity
  thresholds, parry timing windows, light intensity, or fracture.

**Parrying is spellcraft in real time.** The defining image is turning hurled glass into sand: a
parry is not a generic block on a timer, but a counter-operator applied to an incoming spell at the
moment of contact—granulating a lethal blade of glass into blinding sand, melting and spinning it
back as molten slag, locking shards in stasis mid-air as stepping stones, or raising a curved ice
lens to refract a beam.

Because every spell carries real physical trade-offs (mass, velocity, cast geometry, visibility,
recoil, environmental dependence) and can be physically read and counter-parried, **no single
"solved DPS formula" dominates forever.** A dominant spell creates a race to invent its counter.

## 2. One Seamless Shared World

There are **no servers, shards, or realm selection screens.** Everyone inhabits the same vast,
continuous ocean-and-archipelago world.

- **Sailing is how realms connect.** Crossing open water—dealing with real swell, wind, weather,
  storms, and rival crews—gives the world genuine geography and weight.
- **Distance creates culture and rumor.** Regional materials, distant trade hubs, and remote
  workshops mean hearing about a legendary wandmaker or a newly discovered fighting style in the
  northern storm sea gives you a reason to rig a skiff and sail there.

## 3. Warded Sanctuaries vs. the Wild Frontier

World deformation and combat are governed diagetically by **Wards**, not invisible game rules:

- **Warded Trade Ports (Safe Zones):** Anchored sanctuaries where terrain and architecture cannot be
  deformed and hostile magic is damped at the ward boundary. Ports are the social and economic
  heart of the game—markets, wandmakers' workshops, grimoire exchanges, taverns, and **consensual
  dueling circles** where players test inventions and spectate duels.
- **The Wild World (Open Sea, Wild Islands, Ancient Ruins):** Full combat and physical matter
  deformation (craters, ice walls, shattered pillars, redirected rivers). Untended scars naturally
  weather, melt, erode, and heal back toward the base world over time so the world stays habitable
  and visually cohesive.
- **Frontier & Guild Wards (Earned Permanence):** Players and guilds can anchor and maintain their
  own wards on wild islands to establish permanent academies, workshops, or strongholds—contestable
  and part of living player history.
- **Player-Authored Trials / Crucibles:** Dedicated spaces where players can build, publish, and
  remix magical challenges, boss duels, and arenas that surface socially by clear rate and community
  reputation.

## 4. Wandcraft as Physical Engineering

Crafting is a first-class pillar, not a stat-rolling menu. A wand is a **physical, geometric, and
material instrument** that shapes how a mage channels magic:

- **Materials** (storm-struck driftwood, volcanic obsidian, reef coral, hollowed bone, spun glass)
  govern thermal tolerance, arcane resonance, recoil damping, and stress limits.
- **Carved geometry** (length, taper, spiral flutes, forked tips, embedded focusing lenses) governs
  cast focus vs. wide sweep, parry responsiveness, and stability under complex high-work weaves.
- **Every handcrafted wand is visually and physically unique**, rendered in the hand at millimetre
  precision, and carries its maker's mark and provenance. Top duelists commission custom wands
  engineered specifically for their fighting style and signature weaves.

## 5. Living Legends & Player Specialness

Modern MMOs make every player "the Chosen One," which makes nobody special. We want the anime
feeling (*Hunter x Hunter*, *Shangri-La Frontier*, *One Piece*, pre-CU *Star Wars Galaxies*) where
**individual players genuinely matter to the entire shared world**—a single renowned wandmaker, the
first mage to stabilize a refraction weave, the navigator who charted an elusive storm-bound island.

To achieve real specialness without breaking the game through permanent stat inflation or hoarding:

1. **Specialness is unique capability, craft, or knowledge—never raw `+500%` stat power.** Finding
   an ancient grimoire or rare material principle unlocks a new physical technique, crafting method,
   or signature weave, not an unbeatable HP/damage multiplier.
2. **Living legacies (active resonance & succession).** Truly unique relics, titles, or ancient
   grimoires remain bound only while their holder actively wields, advances, or defends them. Idle
   legacies slip back into the world—or holders can intentionally pass them to an apprentice,
   creating real master-disciple lineages and schools.
3. **Organic discovery & social lineage.** Spells and wands carry provenance (*"Kael's Sand-Parry,
   remixed by Soren"*, *"Carved by Mira of Saltspire"*). Keep a secret and stay mysterious; use it
   in public duels, teach apprentices, or sell custom wands and earn fame across the world—knowing
   clever rivals can study your weaves, develop counters, or discover rival principles elsewhere in
   the vast sea.
4. **The 90 / 9 / 1 creator pyramid.**
   - **~90% (Duelists & Explorers):** Fun in the first 60 seconds using foundational spells, guild
     grimoires, and trending player weaves—zero math or authoring required.
   - **~9% (Tinkerers & Remixers):** Tweak existing spells and wand blueprints—bend trajectories,
     swap phase transitions, adjust parry triggers.
   - **~1% (Archmages & Master Crafters):** Invent from low-level physical primitives, shape the
     global meta, and gain lasting social influence when thousands adopt their creations.

---

# Part II: Technical Requirements

## 1. Unified Field Substrate (Single Source of Truth)

The entire vision depends on rendering, lighting, collision, spell interaction, and server
validation reading **the exact same mathematical description of matter**:

- **Content as typed fields:** Geometry (`Surface`), material/substance (`Field<Substance>`),
  volumetric density/emission, height/ocean fields, and spatial ward volumes are expressed as
  composable Wrela fields—never authored meshes, texture atlases, or separate collision hulls.
- **Exact contact & gradients:** Collision, parry-window intersection, deflection normals, and
  matter transformation query the same `distance` and compiler-derived `gradient` as the renderer.
- **Certified spatial queries:** Gameplay and authoring tools rely on derived interval arithmetic
  (`Surface::interval`, `certify_clear`, `certify_apart`) to guarantee non-interpenetration,
  containment (e.g., ward boundaries), and physical validity.
- **Cohesive visuals by construction:** Every player-invented spell, wand, and structure passes
  through the engine's unified lighting store and physical `Look` (atmosphere, tone, ink), preventing
  clashing visual styles.

## 2. Safe, Bounded, Syncable Player Creations

Allowing players to invent spells, weaves, and wands in a live MMO requires strict compiler-enforced
safety and compact representation:

- **Symbolic, data-driven representation:** A player spell or wand is a compact, serializable tree
  of bounded physical primitives and parameters (hundreds of bytes, not megabytes of assets), making
  network replication, grimoire storage, remixing, and lineage tracking trivial.
- **Compile-time / submission-time work budgets:** Using the toolchain's static cost model
  (`std::derive::work`, `@budget`, loop termination proofs, E0712/E0713), the runtime must reject
  unbounded or overly expensive player spells before they can enter the world.
- **Cost tied to physics and compute:** In-game mana/energy/stability costs must scale with both
  physical work (mass, velocity, volume displaced, phase-change energy) and static evaluation cost,
  closing "lag bomb" and infinite-DPS exploits at the root.
- **Deterministic simulation:** Core combat, sailing, and spell physics run deterministically on a
  fixed tick (shared between client prediction and server authority).

## 3. World Coordinates, Persistence & Entropy

- **Large-world precision:** World simulation coordinates must use `f64` (or cell-relative coordinates
  with integer cell indices + local `f32` offsets), translated to camera-relative `f32` before
  rendering so millimetre-scale wands remain jitter-free tens of kilometres out at sea.
- **Layered world field (Base + Scars + Anchored Wards):**
  - **Base field:** Procedural/parametric islands, seabed, and ocean (compact, immutable, identical
    on every client).
  - **Transient deformation layer (scars):** Bounded CSG/displacement/substance deltas stamped by
    combat, with deterministic physical decay/weathering back to the base field over time.
  - **Anchored persistent layer:** Warded ports (immutable) and guild-maintained frontier structures
    (persisted in authoritative storage, replicated by spatial interest).
- **Diagetic ward enforcement:** Ward volumes must be evaluated authoritatively on the server to
  clamp deformation and damp hostile operators, and exposed to the renderer for boundary visual
  feedback.

## 4. Single-World Networking & Spatial Meshing

- **Seamless single-world topology:** Players never choose a shard/realm. World state is organized
  around spatial cells so the backend can start as **one authoritative server** for the initial
  archipelago and scale horizontally via **dynamic server meshing** (handing entities smoothly
  across cell boundaries, ideally in low-density open ocean) without changing gameplay code.
- **Spatial interest management:** Clients receive only nearby entities, active spell fields, and
  local deformation deltas, with smooth LOD falloff for distant ships and landmarks.
- **Crowd resilience:** Trade ports must support high player density via aggressive interest
  prioritization and graceful crowd scaling (with optional interior soft-instancing only as an
  emergency overflow valve, never splitting the global economy, chat, or open world).
- **Authoritative combat & low-latency parrying:** Parry timing windows require tight input-to-
  photon latency (already tracked via `host/src/pace.rs` and `<= 2` frames in flight) paired with
  client prediction, lag-compensated parry evaluation, and authoritative server resolution.

## 5. Rendering Performance Architecture (1080p @ 60 fps on M4 Air / RTX 3060 Bar)

To render a realistic, dynamic, spell-deformable MMO world with hundreds of players, ocean swell,
thin 1 cm wands, and dynamic magic lighting within a **16.6 ms frame budget**—while keeping the
author contract ("fields only") intact—the engine and `backend` toolchain require:

### A. Compiler & GPU Backend (`toolchain/`)
1. **First-class counted `for` loops in `wrela_ir`:** Preserve loop bounds/steps from MIR through
   IR so `unroll`, `hoist`, `work`, MSL (`for`), and SPIR-V (`LoopControl::UNROLL`) stay unified
   with Metal/Vulkan parity.
2. **Lean `toolchain/gpu` execution:**
   - Triple-buffered `StorageModeShared` uniform/dynamic rings on Apple Silicon unified memory
     (eliminating per-frame staging copies and extra `queue writes` command buffers).
   - Concurrent compute encoders (`MTLDispatchTypeConcurrent`) instead of one encoder per dispatch.
   - Vulkan memory sub-allocation (avoiding `maxMemoryAllocationCount` limits), granular
     stage/access pipeline barriers (replacing `ALL_COMMANDS` stalls), and skipping redundant
     zero-fills on write-before-read render targets.
3. **Field combinator scaling (`lib/std/field.wrela`):**
   - `ctz` (`trailing_zeros`) bit-loops for homogeneous `UnionAll`/`SmoothUnionAll` masked asks so
     cost scales with *live parts*, not total parts.
   - Per-thing micro-BVH over part bounds for `O(log N)` root pruning and ray meeting, with tight
     per-brick lowering (`fold_lowering(live_count, k)` rather than `fold_lowering(N, k)`).
   - Avoid duplicate `cone_crossings` in `round_cone_meets` when `inner == grow`.
4. **Mixed precision & wavefront compaction:** Enable `f16` for normals, local brick offsets, shading,
   and lighting (halving VGPR pressure / doubling ALU throughput) while keeping world positions and
   distances in `f32`; use `subgroupBallot` worklists to compact long-tail grazing silhouette rays.

### B. Hybrid Hardware-Accelerated Field Realization (`lib/engine/`)
All strategies remain invisible to authors and selected automatically per thing/distance by the
build-time GPU measurer (`cd615c8`), held to the brute-force raytraced reference:

1. **Hardware 3D texture brick atlas (`Texture3d<R16Snorm>` / `R16Unorm`):** Replace manual 8-load
   `u32` storage-buffer trilinear reads with 1-instruction hardware TMU filtering; split normals out
   of the distance cache (computed at hit via 4-tap tetrahedron or stored separately for surface
   bricks); cache the current leaf brick in `Walk`; upgrade binary octrees to wider `4×4×4`
   64-trees with `u64` occupancy bitmasks.
2. **GPU-extracted view-adaptive brick meshlets + 1-step analytic snap:** For still and expensive
   noise-displaced fields (islands, stacks, architecture), extract surface-nets / dual-contouring
   quads in compute at brick allocation time and rasterize via GPU-driven indirect draws + Hi-Z.
   Optionally snap close-up/silhouette hits with 1 Newton step to exact field precision.
3. **Tight proxy cages for exact, thin, and smooth-blended moving matter:** Rasterize tight
   occupied-brick/part proxies with early-Z (`depth(less)` / reversed-Z `depth(greater)`) for 1 cm
   wands, closed-form `EXACT_MEETS` primitives, and dynamic spell/joint blends—while moving rigid
   parts on bones reuse cached bricks/meshlets in bone space ("move the question, not the matter").
4. **Compact Visibility Buffer + TBDR tile memory:** Compress G-buffer footprint (reconstructing
   world position and pixel footprint from depth instead of storing `Rgba32Float` position/ask
   targets) and keep intermediate passes memoryless in on-chip tile SRAM on Metal (and
   `VK_KHR_dynamic_rendering_local_read` on Vulkan).
5. **Fast, crawl-free antialiasing (replacing 9× edge supersampling):** Hardware 4× MSAA in tile
   memory for rasterized brick meshlets + single-ray analytic cone coverage (`nearest = d / width`)
   for raymarched silhouettes + Halton jittered temporal accumulation using **exact analytic motion
   vectors** and **field/depth validation**.
6. **Hardware inline ray tracing (`ray_query` / Metal `intersector`) + Unified Radiance Cache:**
   Populate a world TLAS/BLAS of occupied bricks/meshlets to eliminate `O(things)` full-screen
   shadow/eye draws, and drive unified GI, reflections, and dynamic spellcraft lighting through a
   sparse world-space brick radiance cache (zero cost when static, zero light leaks, fast updates
   for moving spells and sun).

## 6. Verification & Measurement Discipline

Every system above ships with automated numerical checks:

- **Visual parity:** Fast realizations (meshlets, 3D texture bricks, temporal AA, radiance cache)
  are measured on both Metal (M4) and Vulkan (RADV) against the brute-force raytraced reference for
  pixel error, outline error, crawl, thin-part retention (1 cm wand), and zero holes.
- **Gameplay & physics parity:** Collision, parry contact, ward clamping, and matter conservation
  have headless CPU/WASM deterministic simulation tests.
- **Budget enforcement:** Every frame share holds its target at 1080p/60 on the M4 Air (the proxy
  for the RTX 3060 Steam bar), with build-time cost tables keeping quality predictable across runs.

---

## Phased Path (Vertical Slice First)

Do not build 10,000-player server meshing before the core loop is electric:

1. **Slice A — The Living Archipelago (Single Authoritative Server):** One seamless ocean region
   with sailing, one Warded Trade Port (with a dueling circle), and 3–4 wild islands.
2. **Slice B — Physical Spellcraft & Parry Combat:** Core form/substance/motion operators (proving
   the *glass -> sand* parry end-to-end with real lighting, sound, physics, and bounded player
   remixing).
3. **Slice C — Wandcraft, Grimoires & Provenance:** Physical wand carving as fields, maker's marks,
   grimoire inscription/forking, and the first unique discoverable physical principles.
4. **Slice D — Scale Out:** Expand the continuous ocean, add dynamic spatial server meshing across
   ocean boundaries, frontier guild wards, and player-authored Crucibles.
