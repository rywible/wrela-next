# Register: sources of movement with open rights

Status: 2026-11-25. No source in this register is used yet. This file belongs to
`docs/plans/bodies-and-movement.md` (phase 0.2).

Three searches read the terms of each source on 2026-11-25. No file was downloaded, unless a row
says so. Thus the contents and the formats come from the description of each record.

## 1. The rule

The owner decided on 2026-11-25 that the game must be sold and that each source must agree with
the MIT licence of the repository.

- Permitted: the public domain, CC0, MIT, BSD, Apache 2.0, CC BY and ODC-By. A duty to give
  credit is permitted.
- Not permitted: terms that say "non-commercial", "no derivatives", "share-alike" or "research
  only", and terms of an owner that limit what a person can do with the data.
- Not permitted: movement that only a body model with non-commercial terms can read (SMPL,
  SMPL-X, SMAL).
- Not permitted: a set with an open licence that was made from a source that is not permitted.

The column "Read" shows how sure a row is:

| Mark | Meaning |
| --- | --- |
| 2 | A search read the licence at the source. A second read of the licence field of the repository gave the same result |
| 1 | A search read the licence at the source, one time |
| 0 | The licence was not read at the source. Read it before use |

Before a phase uses a source, it reads the licence file in the download, and it adds the row to
`docs/citations.md`.

## 2. What the search found

- **Human movement for a game:** 14 capture sets pass the rule. Three have fingers. One has real
  strikes of karate. None has a sword or a staff.
- **Human movement on ground that is not flat:** many sets pass the rule. They have stairs,
  ramps, slopes, rough ground and obstacles, and most have measured ground forces.
- **Pushes and recovery:** 8 sets pass the rule.
- **Animals:** no set with the skeleton of a dog, a horse or a cat in three dimensions passes the
  rule. Each one found is non-commercial, has no licence, or is given only on request. Rats and
  birds pass. For dogs and horses, published numbers and summary tables pass.

## 3. Human capture for game movement

| Source | Contents | Format | Fingers | Licence | Read |
| --- | --- | --- | --- | --- | --- |
| [100STYLE](https://ianxmason.github.io/100style) (2022) | More than 4 million frames at 60 frames a second: 100 styles of walk and run | BVH | No | CC BY 4.0 | 1 |
| ACCAD Open Motion Project | Walk, run and other movement; Male 2's martial arts (strikes, blocks, dodges, kicks) | BVH | No | CC BY 3.0 | Imported (2026-11-27: 299 clips), in `docs/citations.md` |
| [Studio Galt sign language archive](https://github.com/StudioGalt/Sign-Language-Mocap-Archive) (2020 to 2026) | 2,710 movements of the arms, hands and fingers: 2,142 signs, letters and numbers. 60 frames a second | FBX | Yes | CC0 1.0 | 2 |
| [Microsoft Rocketbox animations](https://github.com/microsoft/Microsoft-Rocketbox) (2022) | 417 clips: idle, gestures of talk, sit, crouch, wave, dance. In place, with no travel | FBX | Yes | MIT | 2 |
| [Kyokushin karate](https://springernature.figshare.com/articles/dataset/12315629) (2021) | 37 athletes, 3,229 punches and kicks, in the air and on a target | C3D markers. A skeleton fit is necessary | No | CC0 | 2 |
| [CeTI-Age-Kinematics](https://figshare.com/articles/dataset/26983645) (2025) | 32 persons, 30 daily tasks: reach, lift, walk, use of objects. 100 Hz | BVH | Yes | CC BY 4.0 | 2 |
| [Traditional Lanna dance](https://zenodo.org/records/24150940) (2026) | 24 dances, 14 with props. Approximately 3.4 hours at 60 frames a second | FBX | Yes | CC BY 4.0 | 2 |
| [Berkeley MHAD](https://tele-immersion.citris-uc.org/berkeley_mhad) (2013) | 12 persons, 11 actions: jump, bend, punch, wave, clap, throw, sit and stand | C3D and BVH | No | BSD 2-clause. Read in an archive copy, because the site gave an error | 1 |
| [Multi-activity optical and inertial set](https://figshare.com/articles/dataset/34742223) (2026) | 22 persons: walk, run, squat, box, yoga, dance, badminton, stairs, golf | CSV markers and joint angles | No | CC BY 4.0 | 2 |
| [Environment-aware motion matching](https://zenodo.org/records/17220516) (2025) | More than 50 minutes of movement near obstacles | Not given | Not known | CC BY 4.0. A different set of the same author has a conflict | 2 |
| [hhtools parkour](https://huggingface.co/datasets/YaojieShen/hhtools_parkour_mocap) (2026) | 143 clips, approximately 29 minutes: vaults, slopes, a beam, flips, with the mesh of each obstacle | Quaternions of 15 bones | No | Apache 2.0, only as a tag. The page does not say who made the capture | 1 |
| [AndyData-lab](https://zenodo.org/records/3254403) (2019) | 13 persons: reach, carry, pick, at different heights | Xsens files and C3D | No | CC BY 4.0 | 2 |
| [Industrial tasks and crafts](https://zenodo.org/records/6472155) (2021) | Assembly, work with tools, glass work, weaving | BVH | Not known | CC BY 4.0 | 2 |
| [DeepDance](https://zenodo.org/records/7950401) (2019) | 30 dancers, 540 takes of one minute. Use the movement only, not the music | Marker positions | No | CC BY 4.0 | 2 |
| [MPI EMBM](https://figshare.com/articles/dataset/1220428) (2014) | More than 1,440 clips: the body while a person tells a story with emotion | BVH | Not known | CC BY 4.0 | 2 |
| [Dyadic dance](https://doi.org/10.48557/UR2GBG) (2024) | Pairs that dance | CSV | No | CC BY 4.0 | 1 |

Smaller sets with CC BY 4.0 on Zenodo (read 1): records 7031672 and 7034917 (dance), 4288286
(acted scenes), 15232882 (seven emotions), 7821844 (one dancer), 16634663 (step and kneel).

### Movement that a person made by hand (not capture)

| Source | Contents | Licence | Read |
| --- | --- | --- | --- |
| Quaternius Universal Animation Library | Idle and sword clips | CC0 | In `docs/citations.md` |
| [Quaternius Universal Animation Library 2](https://quaternius.itch.io/universal-animation-library-2) | 42 clips in the free part | CC0 | 1 |
| [KayKit character animations](https://kaylousberg.itch.io/kaykit-character-animations) | 161 clips, which include melee, ranged and spell casts | CC0 | 1 |
| [Mesh2Motion](https://github.com/scottpetrovic/mesh2motion-app) | Rigs and clips | The README says CC0 for the art. The repository has no standard licence file | 1 |
| [Kenney animated characters](https://kenney.nl/assets/animated-characters-1) | 8 clips for each pack | CC0 | 1 |

These clips are not physics. Use them as goals for the tracker, not as truth.

## 4. Human movement with measured forces

All sets in this section come from laboratories. Most have the legs and the pelvis only. "GRF"
is the measured force of the ground.

### Ground that is not flat

| Source | Contents | Body | Licence | Read |
| --- | --- | --- | --- | --- |
| Camargo et al. 2021, Mendeley `fcgm3chfff`, `k9kvm5tn3f`, `jj3r5f9pnf` | 22 persons: treadmill, level ground, ramps, stairs. Approximately 20 hours, 10 with GRF | Legs | CC BY 4.0 | 1 |
| Reznick et al. 2021, figshare collection 5175254 | 10 persons: slopes of 0, 5 and 10 degrees, run, stairs, changes between them. GRF | Legs | CC0 | 2 |
| Carter et al. 2024, `researchdata.bath.ac.uk/1341` | 50 runners: speeds and slopes on a treadmill. Approximately 22 hours. GRF | Full | CC BY 4.0 | 1 |
| Gait120, figshare 27677016 (2025) | 120 men: level, stairs, slopes, sit and stand. GRF for level and sit only | Full | CC BY 4.0 | 2 |
| Lencioni et al. 2019, figshare collection 4494755 | 50 persons of age 6 to 72: speeds, toe and heel walk, stairs. GRF | Full | CC0 | 2 |
| Vielemeyer et al. 2026, figshare 29300888 | 13 persons on a ramp with force plates in it | Full | CC BY 4.0 | 2 |
| Dimitrov et al. 2023, figshare 22227337 | 10 persons: slopes, stairs, steps to the side. GRF | Legs | CC BY 4.0 | 2 |
| Scherpereel et al. 2023, `doi.org/10.35090/gatech/70296` | 12 persons: 20 tasks that do not repeat (jump, squat, lunge, cut) and 11 that repeat. GRF | Legs | CC BY 4.0 | 1 |
| Yang et al. 2026, Zenodo 21840566 | 43 children: walk, run, obstacles of 3 heights, jumps, hops. GRF | Full | CC BY 4.0 | 2 |
| Embry et al. 2018, IEEE DataPort `10.21227/gk32-e868` | 10 persons: 3 speeds, slopes from -10 to +10 degrees. GRF | Legs | CC BY | 1 |

### Real ground, with no force plates

| Source | Contents | Body | Licence | Read |
| --- | --- | --- | --- | --- |
| Sharma et al. 2023, figshare collection 6076607 and `10.25452/figshare.plus.21761465` | Xsens suit out of the laboratory: stairs, a course with obstacles, agility. Part 2 has 24 hours | Full | CC0 (part 1), CC BY 4.0 (part 2) | 2 |
| NEWBEE, Losing and Hasenjäger 2022, figshare collection 5758997 | 20 persons, 9 hours: three courses in a town with stairs, ramps and kerbs | Full | CC0 | 2 |
| Muller et al. 2024, Dryad `10.5061/dryad.r7sqv9sn2` | Walk on rocks, with the shape of the ground along the path | Full | CC0 | 1 |
| Brantley et al. 2018, figshare 5616109 | 10 persons: level, ramps, stairs | Full | CC BY 4.0 | 1 |

### Pushes, slips and recovery

| Source | Contents | Licence | Read |
| --- | --- | --- | --- |
| Moore et al. 2015, Zenodo 13030 | 15 persons on a treadmill with random changes of belt speed. Approximately 6 hours. GRF. Full body | CC0 | 2 |
| Chatagnon 2024, Zenodo 10512652 | 21 persons pushed at the shoulder from different directions. Full body. No GRF | CC BY 4.0 | 2 |
| Alizadeh Noghani et al. 2025, `doi.org/10.7274/28689221` | 10 persons: 14 balance tasks. GRF. Full body | CC BY 4.0 | 2 |
| Lorenz and van den Bogert 2023, Zenodo 10557913 | 10 persons: sudden jumps of belt speed | CC BY 4.0 | 1 |
| Leestma et al. 2023, `doi.org/10.35090/gatech/70293` | The ground moves below a person who walks | CC BY 4.0 | 1 |
| Wu et al. 2025, Dryad `10.5061/dryad.cnp5hqch3` | Pushes at the pelvis while a person walks | CC0 | 1 |
| Santos et al. 2017, figshare 4525082 | 49 persons: balance while they stand. GRF | CC0 | 1 |
| OpenCap, Uhlrich et al. 2023, `simtk.org/projects/opencap` | 10 persons: walk, squat, sit to stand, drop jump. GRF. Full body | Apache 2.0 | 1 |

### Level walk and run with forces

These sets pass the rule (read 1, but Schreiber and Fukuchi: read 2).

- Schreiber and Moissenet 2019, figshare 7734767: 50 persons, 5 speeds, full body. CC BY 4.0.
- Fukuchi et al. 2018, figshare 5722711 (walk), and 2017, figshare 4543435 (run). CC BY 4.0.
- Horst et al. 2019, Mendeley `svx74xcrjr`: 57 persons, full body. CC BY 4.0.
- van der Zee et al. 2022, figshare 16530939: speed, step length and step width change. CC BY 4.0.
- Van Criekinge et al. 2023, figshare collection 6503791: 138 persons of age 21 to 86. CC0.
- Hamner and Delp 2013, `simtk.org/projects/nmbl_running`: run at 2 to 5 m/s, full body. MIT.
- Hollinger et al. 2026, figshare 30225739: walk, run, jump, hop. CC BY 4.0.
- AddBiomechanics 1.0 (Werling et al. 2024): 273 persons and 71.5 hours on one skeleton, with
  masses and joint torques. It says CC BY 4.0 for all of it. Use only the nine parts whose own
  records pass the rule: approximately 59 hours. One part has no licence at its source.

## 5. Mass of the segments of a body

| Source | What it gives | Rights | Read |
| --- | --- | --- | --- |
| OpenSim full-body model of Rajagopal, `simtk.org/projects/full_body` | Mass, centre of mass and inertia of each segment of an adult | MIT | 1 |
| `github.com/BMClab/BMC`, folder `data` | The tables of de Leva (1996) and of Dempster as text files | CC BY 4.0 for the copy. The numbers come from de Leva | 1 |
| NASA RP-1024, Anthropometric Source Book (1978) | The inertia of the body and its segments | Public use permitted | 1 |
| AddBiomechanics | The scaled mass and inertia of each of 273 persons | CC BY 4.0, with the limit above | 1 |
| Jones et al. 2018, PLoS ONE 13:e0206037 | The segments of a German Shepherd dog: mass share and centre of mass | CC BY | 1 |
| Nauwelaerts et al. 2011, J. Anat. 218:500 | The limb segments of 38 horses | A paper. Use the numbers with a citation | 1 |

A measured number is a fact. Use it with a citation. Do not copy the table or the text of a
publisher.

## 6. Animals

### Capture and tables that pass the rule

| Source | Animal and movement | What it is | Licence | Read |
| --- | --- | --- | --- | --- |
| Aldarondo et al. 2024, `doi:10.7910/DVN/FB0MZT` | Rat, free movement. 69 GB | Joint angles on a body model with open code | ODC-By 1.0 | 1 |
| Rat 7M, Dunn et al. 2021, figshare 13739233 | Rat, free movement. Almost 7 million frames | Markers in three dimensions | CC BY 4.0 | 1 |
| Monsees et al. 2022, Dryad `10.5061/dryad.g4f4qrfsw` | Rat and mouse: gait and a jump across a gap | A fitted skeleton with joint angles | CC0 | 1 |
| Catavitello et al. 2015, PLoS ONE 10:e0133936, file S1 | 6 dogs: walk, trot, gallop | Points in two dimensions, from the side | CC BY | 1 |
| Charles et al. 2025, `doi:10.17638/datacat.liverpool.ac.uk/3038` | 27 dogs: trot. 242 trials | A table: speed, stride, duty factor, peak force. No paths | CC BY 4.0 | 1 |
| Gmel et al. 2024, Mendeley `c24tf9332k` | 425 horses: walk and trot | A table of 28 stride numbers | CC BY 4.0 | 1 |
| Wimberly et al. 2021, Dryad `10.5061/dryad.z08kprrd5` | 244 species: walk | A table: duty factor and limb phase | CC0 | 1 |
| Daley and Birn-Jeffery 2026, Dryad `10.5061/dryad.7m0cfxqc4` | Guinea fowl: walk to run. 1,678 strides | GRF and the path of the centre of mass and of the feet | CC0 | 1 |
| Daley et al. 2026, Dryad `10.5061/dryad.n02v6wxbv` | Guinea fowl: straight runs and turns of 90 degrees | GRF and paths | CC0 | 1 |
| Andrada et al. 2014, Dryad `10.5061/dryad.jh5h4` | Quail: walk and run | X-ray movement in three dimensions, with forces | CC0 | 1 |
| Rankin et al. 2016, Dryad `10.5061/dryad.fh3h6` | Ostrich: walk and run | A body model with gait data | CC0 | 1 |
| 3D-POP, `doi:10.17617/3.HPBBC7` | Pigeon on the ground | Markers of the head and the body | CC BY 4.0 | 1 |
| [Quaternius animated animal pack](https://quaternius.com/packs/ultimateanimatedanimals.html) | 12 animals with four legs | Made by hand, not capture | CC0 | 1 |

### Published numbers for gaits

Phase 6 takes its targets from these. Read each paper before its number becomes a target.
"Second" means that the number was read in a paper that quotes the first paper.

`M` is the mass in kg. `Fr` is the Froude number: the square of the speed, divided by gravity
and by the height of the hip.

| Quantity | Value | Citation | How read |
| --- | --- | --- | --- |
| Change from walk to trot | `Fr` approximately 0.5 | Alexander and Jayes 1983, J. Zool. 201:135 | Second |
| Change from trot to gallop | `Fr` between 2 and 3 | Alexander and Jayes 1983 | Second |
| Change from walk to trot, horse | `Fr` 0.35, with leg length as the height | Griffin et al. 2004, J. Exp. Biol. 207:4215 | Full paper |
| Stride length divided by hip height | Walk: 2.4 `Fr`^0.34. Run: 1.9 `Fr`^0.40 | Alexander and Jayes 1983 | Second |
| Duty factor of a fore limb | Walk: 0.52 `Fr`^-0.14. Run: 0.52 `Fr`^-0.28 | Alexander and Jayes 1983 | Second |
| Speed at the change from trot to gallop | 1.54 `M`^0.216 m/s | Heglund and Taylor 1988, J. Exp. Biol. 138:301 | Full paper |
| Stride frequency at that change | 4.19 `M`^-0.150 Hz | Heglund and Taylor 1988 | Full paper |
| Metabolic power for each kg | 10.7 `M`^-0.316 times the speed, plus 6.03 `M`^-0.303 W | Taylor, Heglund and Maloiy 1982, J. Exp. Biol. 97:1 | Abstract |
| Limb phase | Trot 50%, pace 0%. A walk has a duty factor above 0.5 | Hildebrand 1965, Science 150:701 | Second |
| Walk of a dog: duty factor and limb phase | 60% and 19% | Wimberly et al. 2021, calculated from the table | Data |
| Walk of a horse: duty factor and limb phase | 67% and 24% | Wimberly et al. 2021 | Data |
| Trot of a dog | 2.25 m/s, cycle 0.45 s, duty factor 0.49 fore and 0.44 hind | Charles et al. 2025, calculated from the table | Data |
| Share of weight on the fore limbs, dog | 63% in a walk | Griffin et al. 2004, J. Exp. Biol. 207:3545 | Abstract |
| Stiffness of the leg in a trot | 0.715 `M`^0.67 kN/m | Farley et al. 1993, J. Exp. Biol. 185:71 | Full paper |
| Largest stress of a muscle | 200 to 300 kPa. Median 187 kPa for muscles of legs | Rospars and Meyer-Vernet 2016, R. Soc. Open Sci. 3:160313 | Full paper |

## 7. Sources that need a decision

| Source | Contents | Problem |
| --- | --- | --- |
| [BEAT, version 1](https://huggingface.co/datasets/H-Liu1997/BEAT) | 76 hours of gestures of 30 persons who speak, as BVH with fingers | The only licence is an Apache 2.0 tag. A different page says non-commercial. Ask the authors in writing before use |
| MOSAIC (2026) | Human movement | CDLA-Permissive 2.0, but only a non-commercial body model can read it |
| UMONS-TAICHI | 2,200 samples of taijiquan | Zenodo says CC BY 4.0. The repository of the authors says non-commercial and share-alike |
| MocapFlow | Box, Muay Thai, dance | Two different licences, and it does not say how the movement was made |
| DeepMimic dog clips | Four clips of a dog | The repository is MIT. The source of the clips is not given, and later clips of the author are non-commercial |

Do not use these until the problem is solved.

## 8. Sources that are refused

Do not examine these again.

- **The owner decided:** the CMU motion capture database.
- **Non-commercial or research only:** AMASS and each set made from it, LAFAN1, Bandai Namco 1
  and 2, Motorica, ZeroEGGS, SFU, MoVi, Talking With Hands, Trinity, SnapMoGen, Nymeria,
  Motion-X, InterHuman, Inter-X, Human3.6M, the Virginia Tech natural motion set, UnderPressure,
  the Cyprus dance database, Seamless Interaction, ChingMu, the sword clips of Reallusion,
  the dog capture of AI4Animation and each set made from it, PFERD, RGBD-Dog, DigiDogs.
- **Share-alike or no derivatives:** HDM05, SparsePoser, the Vienna horse collection, the horse
  files of Hobbs et al. 2016.
- **Terms of an owner:** Mixamo, BONES-SEED, HUMOTO, Truebones Zoo, the Unity and Unreal samples.
- **Made from a source that is not permitted:** AIST++, PHUMA, BEAT2, Motion247, OmniMocap-X,
  MoCapAct.
- **No licence:** KIT, HuGaDB, GroundLink data, HuMoD, DogML, AcinoSet, OpenMonkeyStudio.
