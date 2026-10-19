# C-levelset-placement.md: where the level-set code belongs

Branch issue-449-modern-mesh-core @ e7080344. Paths are relative to `Code/Source/solver/` unless absolute. This was a read-only study; nothing was built or run. Cross-references:
- **B §x** = B-infrastructure-design.md;
- **A §x** = A-formulation-design.md;
- **Plan Rx** = Documentation/free_surface_architecture_refactoring_plan_20260904.md.

## 0. Answer to the user's question

The question was whether the level-set files and work should move out of Application into existing FE folders. The answer is **mostly yes, with two exceptions.**

| Category | Lines | Destination |
|---|---:|---|
| Physics-agnostic FE numerics in Application/Core | ≈7,550 of 13,116 (58%) | FE/LevelSet (new subfolders Extension/ and Maintenance/, as B proposes), FE/Systems, FE/Geometry, FE/Core |
| Input-schema translation | ≈5,050 (39%) | stays in Application (Translators/) |
| ApplicationDriver.cpp FE capability | ≈19k (B: 18.6k) | FE/LevelSet, FE/TimeStepping, FE/Systems |
| ApplicationDriver.cpp physics | ≈9k | Physics |
| Level-set code already in FE that is physics | 1,740 (LevelSetStaticCapillaryEquilibrium.*) plus the Young weights in LevelSetCurvatureProjection | Physics |

The ≈5,050 lines that stay are the translators, the alias tables and the legacy maintenance resolution. They change only when the FE/Core OptionSchema lands (DECISIONS.md, 2026-10-19).

The driver's ≈19k FE lines are composition of FE/LevelSet primitives: it calls about 40 distinct `svmp::FE::level_set::` functions.

## 1. Inventory

### 1.1 Application level-set files

Notes used in the "Application-only dependencies" column:
- "XML tree" means `Parameters` / `EquationParameters` from `Parameters.h`.
- "Physics input" means `svmp::Physics::ParameterMap` / `EquationModuleInput`.

| File (lines incl. .h) | What it does | Includes and calls | Application-only dependencies | Callers | Class | Verdict |
|---|---|---|---|---|---|---|
| Core/LevelSetPdeVelocityExtension (2,341+288 = 2,629) | Harmonic or least-squares-normal extension of a vector field to dry vertices. Canonical replicated assembly; LU/LDLT/MUMPS/MUMPS-serial factorization. `PdeVelocityExtensionCache`, keyed on mesh revisions and exact system content, with a collective reuse vote. Per-component solves on rank c. Owner-local constraint rows of type `FE::level_set::VelocityExtensionConstraintRow`. | Mesh, MeshComm, DistributedTopology, FE Backends (Eigen, MUMPS), Sparsity, MPI (12 calls); 8 `SVMP_PDE_EXTENSION_*` env names | **None** (it includes the map header, which moves with it) | Driver `updateLevelSetAdvectionVelocitiesFromState` 19520–19607. Translator: token check at 1956. Tests: test_LevelSetPdeVelocityExtension (19 tests), its MPI twin (6 tests). | FE capability | **Move** to FE/LevelSet/Extension/ (B §2.7) |
| Core/LevelSetVelocityExtensionMap (3,177+370 = 3,547) | Wall-compatible normal-band extension: graph, regression, condition and rank checks, wall projection, component labels, MPI layer sync (2012–3177). Map snapshot, revision and change report (1–482). JSON artifact writer (483–959). | Mesh, DistributedTopology, FE/LevelSet/LevelSetVelocityExtensionConstraint.h, MPI (13 calls) | **None** | Driver 19633, 19655; 15 using-declarations at 2860–2873; workflow tests (116 refs) | FE capability, plus an output writer | **Move** to FE/LevelSet/Extension/. Later, split the writer into a sibling `…Artifact` file (precedent: LevelSetConservativePhaseArtifact in FE). |
| Core/LevelSetCurvatureSamples (628+95 = 723) | Supplemental curvature samples from cut-volume and high-order interface rules; reference-to-physical mapping; insertion-ordered duplicate filter | FESystem, IMeshAccess, CutQuadrature, MappingFactory, LevelSetCurvatureProjection | **None** | Driver `collectLevelSetCurvatureSupplementalSamples` 17071–17291 (17125, 17243, 17273), reached from `projectLevelSetCurvatureFieldsFromState` 18541; tests (2 + 4) | FE capability | **Move** to FE/LevelSet/ next to LevelSetCurvatureProjection. Remove the verbatim duplicate of `createCellGeometryMapping` / `physicalCellPointAtReference` (CurvatureSamples.cpp 37–80 = driver 17027–17070). |
| Core/LevelSetMaintenanceHistory (45+28 = 73) | `copyFieldDofsIntoFeOrderedSolution`: copies one field's DOF slice | FESystem | None | **No production caller**; only 3 tests in test_LevelSetMaintenanceHistory.cpp (129–251) | generic FE/Systems helper | **Delete** with its 3 tests, or move to FE/Systems if kept. Not Maintenance (differs from B). |
| Core/LevelSetMaintenanceTransactionConsensus (399+38 = 437) | (a) exact all-rank agreement of a word sequence, with a padded MIN/MAX Allreduce (184–321); (b) commit/reject decision from the ledger's canonical words (28–183, 322–399) | MeshComm, MPI; **ApplicationDriver.h** for `LevelSetMaintenanceWorkLedger` | (b) only: the ledger type | (a) driver 4687, 25705, 25820, 26004, 26085, 26532; (b) driver 34306, 35436; tests (2 MPI) | (a) FE; (b) ledger (energy accounting) | **Split.** (a), ≈150 lines → FE/Systems/SystemCollectives (A/B name). (b) stays with the ledger: Application now, B's neutral rows or Physics later (B S7). |
| Core/LevelSetCutConfiguration (1,182+115 = 1,297) | `ActiveCutVolumeRequest` and 3 enums; XML → request parsing; request → FE options; policy hash; marker resolution; consumer validation | **Parameters.h, OopMpiLog.h**, FE/LevelSet, FE/Interfaces | XML tree in 115–190 and 347–999; oopCout at 1145 | Driver (17 parse calls, 37 type uses), SimulationBuilder 784 and 1416, LevelSetMaintenanceConfiguration, tests (34 tests) | mixed | **Split.** ≈400 lines → FE/LevelSet/LevelSetGeometry (the struct, 64–114, 316–346, 673–733, 1000–1101). ≈900 lines stay in Application (lexical 26–63, 115–315, 347–672, 734–999; the provenance messages in 1103–1179). |
| Core/LevelSetMaintenanceConfiguration (1,259+57 = 1,316) | Resolves legacy-maintenance and installation inputs into an immutable configuration (transport options, isovalue, open boundaries by face name, cut request, curvature/static-capillary options, input observations); JSON serializer | Translators/LevelSetConfigurationParsing.h; ResolvedMovingDomainConfiguration.h; FE option headers | Physics input; Translators; provenance | Driver 30865 (steady), 31370 (transient); tests (21 tests). The serializer (1107–1259) is called by **tests only**. | input translation | **Keep** in Application, moved to Translators/ (Core including Translators today is an upward layering smell). The output struct later splits into FE `LevelSetMaintenanceOptions` plus Application provenance (C3e). |
| Core/LevelSetEquationInputSnapshot.h (35) | Immutable snapshot of the equation input plus the legacy maintenance input | Physics/Core/EquationModuleInput.h | Physics input | EquationTranslator 450–490, LevelSetEquationTranslator, ResolvedMovingDomainConfiguration.h, MaintenanceConfiguration | input | **Keep** (Translators/) |
| Translators/LevelSetConfigurationParsing (382+223 = 605) | `LevelSetConfigurationReader` over ParameterMap, with installation versus legacy policy; alias tables; enum parsers | FE/LevelSet/LevelSetOptions.h, Physics ParameterMap | Physics input | LevelSetEquationTranslator, MaintenanceConfiguration | input schema | **Keep.** Becomes an FE/Core `OptionSchema<LevelSetTransportOptions>` plus an Application alias overlay (§6.5). |
| Translators/LevelSetEquationTranslator (2,363+91 = 2,454) | See §2.5 for the breakdown by line range | Physics PhysicsModule / EquationModuleInput / JITRuntimePolicy; Application snapshot and resolved config; FE LevelSetTransport, SpaceFactory, EntityDofMap; Mesh fields | Physics input; Application config | EquationTranslator 498–504, SimulationBuilder, the driver (resolved configs) | ≈2,160 schema; ≈293 FE logic | **Keep**, after moving about 293 lines of FE logic out (§2.5) |

Two further helpers are used only by the level-set code:
- **Core/NearestPointIndex.h** (192 lines, no includes beyond std): a nearest-point search used by the `nearest_interface_point` extension (driver 19728). It goes to FE/Geometry, not Extension as B has it, because it is a generic geometric search.
- **Core/OopMpiLog.h** (118 lines, depends only on FE/Core/Logger.h and MeshComm): its rank-aware `oopCout` is needed by any driver group that moves. It goes to FE/Core/RankLog.h, with a forwarding header, so log text stays byte-identical.

### 1.2 Level-set function groups in ApplicationDriver.cpp

**File structure:**
- 118–1296: `application::core` anonymous namespace;
- 1297–2356: exported ledger and energy code;
- 2358–30569: a global anonymous namespace (28,200 lines, 513 top-level declarations);
- 30571–37439: the `ApplicationDriver` methods. `runTransient` (31153–37118) alone is 5,966 lines.

**Dependency codes** in the table are per range, counted from the source:
- P = `Parameters`;
- S = `SimulationComponents` / `sim.` (used only as a bag of fe_system, primary_mesh, time_history, backend, linear_solver and snapshot caches);
- R = `LevelSetMaintenanceRequest` (it wraps the Application config handle);
- C = `ActiveCutVolumeRequest`;
- A = `LevelSetAdvectionVelocityRequest`;
- L = oop log calls;
- E = env reads;
- M = MPI calls;
- Lg = ledger types from ApplicationDriver.h.

**Hot** means the outer-pass branch's territory (§5.2).

| Range | Lines | Content | Deps | Called from | Class | Destination (B ref) |
|---|---:|---|---|---|---|---|
| 118–2356 | 2,239 | maintenance work ledger, accepted-step energy aggregation, energy candidates | Lg 105 | runTransient ledgers | Physics (≈1,800) + neutral rows | Physics FreeSurfaceEnergyAccount; neutral rows to FE/LevelSet/Maintenance (B §1.1, S7). Agree. |
| 2368–2570 | ≈200 | `VelocityExtensionSample`, `gatherVelocityExtensionSamples` (Allgatherv) | M | 19713 | FE | FE/LevelSet/Extension (C3c) |
| 2571–2875 | 305 | communicator helpers, ownership snapshot, boundary-intersection summary | M 31 | many | FE | FE/Systems/SystemCollectives |
| 3674–3760 | 87 | `LevelSetMaintenanceRuntimeState` (incl. the redistancing accumulator 3682–3685), `DeclarationBindings`, `Request` | R | driver-wide | FE state wrapping an Application config | FE/LevelSet/Maintenance, after the config split (C3e) |
| 3761–4702 | 942 | canonical maintenance-schedule words and collective consensus | R 14, S 6 | 30870, 30885, 31375, 31382, 32224, 33436, 34388 | FE | FE/LevelSet/Maintenance (B S9) |
| 4703–4725 | 23 | curvature projection cache structs | — | sync, accepted | FE | curvature service (C3b) |
| 4726–4775 | 50 | `LevelSetAdvectionVelocityRequest`, `AcceptedVelocityExtensionMapRecord` | A | extension | FE declaration + App face names | split: FE `FieldExtensionDeclaration` (B §2.7) plus App binding |
| 4776–4828 | 53 | `splitFaceNameList`, `strongZeroVelocityComponentMask` (4799–4828, a pure 30-line function) | — | extension | App binding | App FreeBoundaryBinding; later `WallTracePolicy` (B §6.7, not bitwise by construction) |
| 4830–5137 | 308 | `ActiveCutContextRefreshReport`, `TransientCutTopologyAttemptTracker` (topology epochs) | — | runTransient epoch hooks 32405–32464 | FE | LevelSetGeometryService / GeneratedState epochs. **B's row 4776–5180 labels this range as the extension mask.** |
| 5203–5527 | 325 | `levelSetAdvectionVelocityRequests(const Parameters&)`: re-parses the same keys as the translator (1940–2090), with inline alias lists | P, A | 30891, 31251 | App (duplicate parser) | App FreeBoundaryBinding, rebuilt from `ResolvedLevelSetEquationConfiguration` (R1 duplicate removal; characterize first) |
| 5528–6120 | 593 | side helpers, retained-rule pointers, region summaries, facet adjacency, interface-velocity sample markers and cells | C 4, A 5 | refresh, extension | FE | `ls::LevelSetGeometry` queries (A §2.7 #2–5) |
| 6121–6447 | 327 | partition-independent cut-topology fingerprints (collective) | C 4, M 7 | refresh, tracker | FE | LevelSetGeometryService (B S8). **Hot.** |
| 6448–6716 | 269 | sync-point policy predicates, tangent-policy summary, topology-change log | E 2 | `synchronize_state` | FE | GeneratedState policy (B S6) |
| 6717–7231 | 515 | wet-volume diagnostics and active-fluid output | L 6 | outputResults 37311–37338 | FE diagnostics + App output | B: FE/FreeBoundary/Diagnostics; the writers stay App |
| 7232–7397 | 166 | coupled level-set residual criteria from XML; coverage logs | P 2 | 30892, 31387 | App | SolverOptionResolution (B §5) |
| 7398–7989 | 592 | FE-ordered gather/scatter, reductions (≈360); pressure-update guard (231) | M 22, E 8 | everywhere | FE + Physics | SystemCollectives; Physics |
| 11061–11347 | 287 | maintenance cut-topology revision, functional values, algebraic revision | S 5, Lg 3 | ledger | FE revisions + Physics values | split (B S7, S9) |
| 12669–12810 | 142 | implicit-cut backend selection and qualification counts | — | refresh logs | FE | **LevelSetGeometryService** (B puts it in Maintenance) |
| 12811–13193 | 383 | volume options, maintenance targets, open boundaries (face → marker), bound-preserving candidates (12930), volume-correction accounting | S 15, R 5 | 33703 | FE (face names: App) | Maintenance (B S9) |
| 13194–14402 | 1,209 | accepted wall-contact constraints, wall-aware maintenance context | S 14 | maintenance | FE with a physics input (contact angles) | Maintenance; contact data via a Physics callback (B §1.1) |
| 14403–15224 | 822 | first-order generalized-alpha maintenance publication plan and algebra | M 7 only | 15413; tests (14 + 9 refs) | FE | **FE/TimeStepping/FieldRepairPublication** (B). Cold, so it can move now (C3a). |
| 15225–15555 | 331 | projection-reinitialization staging, transport coefficients, kinematic baseline | S, R | 16077, 27956 | FE | Maintenance stages |
| 15556–16639 | 1,084 | `applyLevelSetMaintenance`: reinit, volume correction, reconciliation, patch bounds, redistancing trigger (15880–16072: env `SVMP_REDISTANCING_TRIGGER_CELLS` at 15896, redistance-and-restore loop 15914–16002), generalized-alpha publication | S 29, R, Lg 5 (observer enum) | 35215 (post-accept transaction) | FE orchestration | `LevelSetMaintenancePipeline` (B §2.3). The restore loop goes to the existing FE/LevelSet/LevelSetInterfacePreservingRedistancing.* and the trigger becomes `RedistancingTrigger`. |
| 16640–16965 | 326 | volume-correction free-surface work logs | S 6 | maintenance | Physics | Physics (B S7) |
| 16966–17291 | 326 | state view, vertex-field evaluation, duplicated mapping helpers, `collectLevelSetCurvatureSupplementalSamples` | E 1 | 18541 | FE | FE/LevelSet curvature (C3b). **Hot.** |
| 17292–18752 | 1,461 | curvature signatures and caches, quadratic traction-rule validation, KAG traction binding (17604–17779, Physics), `projectLevelSetCurvatureFieldsFromState` (18349) | R 12, S 4 | 29226, 30954, 31013, 31073, 31678, 32379, 32479, 35841 | FE (+FS) | DerivedField participant in FE/LevelSet (B §7.1); KAG binding → Physics. **Hot.** |
| 18753–18945 | 193 | level-set mesh-field checkpoint, restore and sync | S 10, C 2 | refresh, transaction | FE | LevelSetGeometryService |
| 18946–20019 | 1,074 | `updateLevelSetAdvectionVelocitiesFromState` (18987), factorization env setting (18958–18985), **static PDE cache map (19520)** | S 5 (only fe_system and primary_mesh), A, E 2 | 30965, 32390 (sync), 20016 → 31023, 31085, 32489, 35853 | FE | FE/LevelSet/Extension participant (B S10). **Hot.** |
| 20020–20292 | 273 | accepted extension-map registry; `writeAcceptedVelocityExtensionMapArtifacts(params,…)` (20138: params is used only for the output folder) | P 4 | 31091, 35864 | FE registry + App writer | registry → Extension; output folder passed as a path |
| 20293–20528 | 236 | memory details, ghost-layer audit | E 5 | refresh | FE | LevelSetGeometryService / HaloDiagnostics |
| 20529–21976 | 1,448 | `refreshActiveCutIntegrationContextFromSolution`: **re-parses `activeCutVolumeRequests(params)` at 20539**, lifecycle build, snapshot, install | P, S 35 | 22886, 22924, 23090, 23157 | FE | LevelSetGeometryService (B S8). **Hot.** |
| 21977–23451 | 1,475 | refresh cache, signature, reinstall, resync, attempt-entry retention, restore reinstall (more re-parses at 22811, 22942, 23290) | P 7, S 60, C 8, E 8 | `refreshActiveCutIntegrationContextCached`: 14 call sites (30943…35828) | FE | same. **Hot.** |
| 23452–23517 | 66 | prescribed-coefficient checkpoint | — | transaction | FE | FE/Systems/StepTransaction (B) |
| 23518–23920 | 403 | `LevelSetMaintenanceGeometryTransaction` (`refresh(params,…)` at 23656, 23690 / 23716) | P 2, S 2, C 2 | conservative phase, maintenance | FE | StepTransaction + LevelSetGeometryService |
| 23921–24399 | 479 | post-accept topology evidence, rollback reasons, collective vector capture and assign (24007–24339) | M 10 | maintenance | FE | StepTransaction / SystemCollectives |
| 24400–28606 | 4,207 | conservative phase: graph, projection, init (24567), velocity sampling (24689), contact-protected nodes, contact-stage candidate (25129–25404, Physics), stage snapshots, artifacts (26473–26963), `reconcileConservativePhaseGeometry` (26965), `applyConservativePhaseCandidates` (27273–28572, 1,300 lines), rollback | P 7, S 29, R 16, L 12, M 17 | 31012, 31673, 33655, 34454 | FE (+ contact = Physics) | FE/LevelSet/ConservativePhaseWorkflow + LevelSetPhaseGeometryReconciliation (B S11, Plan R6) |
| 28607–28666 | 60 | `ZeroTimeDerivativeIntegrator`, constraint semantics key | — | static capillary | FE | FE/Systems |
| 28667–30569 | 1,903 | static-capillary certificate and initializer (29419) | P 2, S 23 | 30997, 31659 | Physics | Physics (B S5). Cold. |
| 30811–31152 | 342 | `runSteadyState`: maintenance setup 30864–30885, duplicate `synchronize_state` 30925–30972 | — | — | App + glue | keep ≈150 (B §1.1) |
| 31153–37118 | 5,966 | `runTransient` | — | — | App ≈350 + glue ≈5,600 | B §1.2 sub-table is accurate. Level-set wiring is listed below. |

Level-set wiring inside `runTransient`:
- maintenance setup: 31369–31387;
- outer fixed-point options: 31403–31592;
- initial cut refresh, curvature, conservative phase and static capillary: 31656–31692;
- `synchronize_state`: 32272–32404, which runs cut refresh → curvature → extension → semi-implicit reference;
- epoch hooks: 32405–32464;
- post-accept maintenance transaction: 34488–35578, with the redistancing check at 34511–34516;
- accepted refresh, curvature, extension and artifacts: 35828–35864;
- rejection rollback: 36854–37047.

File restart is not wired in the OOP driver (30707 prints "restart/continuation" as unsupported). "Restarts" in the driver are cut-topology nonlinear restarts (32414–32443).

### 1.3 Corrections to B's driver map (§1.1)

- **4776–5180.** The XML read of the wall direction is in 5203–5527, not in `strongZeroVelocityComponentMask`, which is a pure function at 4799–4828. 4830–5137 is cut-topology epoch machinery.
- **12669–12810.** These are cut-geometry backend statistics: LevelSetGeometryService, not Maintenance.
- **LevelSetMaintenanceHistory.** It has no production caller (§1.1).

## 2. Verdicts on the specific questions

### 2.1 PDE velocity extension, extension map and the MPI/MUMPS cache

- **Verdict: move whole** to FE/LevelSet/Extension/ (B §2.7). It belongs next to FE/LevelSet/LevelSetVelocityExtensionConstraint:
  - FE already owns the row type (`VelocityExtensionConstraintRow`, LevelSetVelocityExtensionConstraint.h 47–53), the kernel and its installation (LevelSetTransport.cpp 2031).
  - Application owns only the producer of the rows. That split is backwards.
- **The cache.** `PdeVelocityExtensionCache` (h 201–243, cpp 908–1096) is pure numerical reuse: it is keyed on mesh revisions plus bitwise system content, with collective votes. It uses only FE backends (FE/Backends/Eigen; FE/Backends/MUMPS/MumpsDistributedSolver, already in svfe with `FE_HAS_MUMPS` PUBLIC), MeshComm and MPI. **No new link dependencies.**
- **Cache ownership.** The only Application-side wart is ownership: a function-local `static std::map<std::string, PdeVelocityExtensionCache>` in the driver (19520–19524).
  - It outlives the FESystem.
  - In test_application it persists across tests, so hit and miss statistics depend on test order.
  - In C3c it becomes a member of the FieldExtension participant, one per target field.
  - This is bitwise for LuColamd, LdltAmd and MumpsSerial: reuse equals a fresh solve, per header 182–186.
  - For opt-in distributed Mumps, reuse matches only to round-off (header 81–88). With one run per process the lifetime is unchanged, so production is unaffected.
- **No split is needed.** The Application keeps what it already has: `pdeVelocityExtensionFactorizationSetting()` (driver 18958–18985, env → typed option) until B's S13 types the remaining `SVMP_PDE_EXTENSION_*` knobs (REPLICATED_SOLVES, CACHE_ENTRIES, SELF_CHECK, DUMP, DUMP_CALLS, FAIL_COMPONENT, MUMPS_DIAGNOSTICS). FE already reads env in many places, so moving them is not a new kind of dependency.
- **Do not rename in the move** (velocity → field, wall → trace). Renaming changes test names and log text and belongs to C3c. Keep the log prefix `[svMultiPhysics::Application] PDE velocity extension` unchanged; several generators reference it.
- **Mesh dependency.** Both files take `svmp::Mesh` / `MeshComm` directly. That is allowed in FE under `FE_WITH_MESH` (precedent: FE/MovingMesh, CMakeLists 1181–1191). Porting to `IMeshAccess` is optional later work.

### 2.2 Maintenance configuration, history, transaction consensus: an FE maintenance pipeline?

- **Yes, an FE `LevelSetMaintenancePipeline`** (B §2.3, FE/LevelSet/Maintenance/). It covers driver 3674–4702, 12811–16639 and part of 23452–24399.
  - **Its stages are already FE functions:** `shouldApply*`, `reconcileLevelSetWithKinematicFlux`, `redistanceLevelSetToZero{Polygon,Surface}`, `measureLevelSetZero*Change`, `levelSetDofCellClosure{,3D}` and `computeLevelSetCutCellVolume`.
  - **What it needs from Application** is only the resolved options and an observer for energy rows.
- **The configuration is not part of FE.** `LevelSetMaintenanceConfiguration` is legacy-input resolution: keep it in Application (Translators/). Its output `ResolvedLevelSetMaintenanceCompatibilityConfiguration` (h 17–38) mixes three things:
  - FE option structs: transport, curvature projection, cut request, isovalue;
  - a physics option: static capillary, which goes to Physics with S5;
  - provenance: input observations, installation handle, face names.
  
  In C3e, define `ls::LevelSetMaintenanceOptions`, containing transport, isovalue, open boundaries by **marker**, `LevelSetVolumeOptions`, curvature options plus cadence, and `RedistancingTrigger`. The Application binding fills it.
- **History:** delete (§1.1). It is not a maintenance concern.
- **Consensus: split** (§1.1). The word-agreement primitive is generic and used 6 times outside the ledger. The decision function follows the ledger.
- **The observer enum.** `LevelSetMaintenanceWorkSubstage` (ApplicationDriver.h 24–30) is the only Application type in the pipeline's callback signature (`LevelSetMaintenanceStageObserver`, driver 12924). Move it to FE (`ls::MaintenanceSubstage`) with an Application alias (C2e). The values do not change.

### 2.3 Cut configuration

**Split** (≈400 lines to FE, ≈900 stay).
- **`ActiveCutVolumeRequest` and its 3 enums** become the declaration struct of `ls::LevelSetGeometry` (A §2.7 #2–5, B §7.1), with `using ActiveCutVolumeRequest = …` kept in Application.
- **Policy hash.** `activeCutVolumeRequestPolicyKey` (1012–1080) hashes only numerical fields, not `origin`, `equation_type` or `geometry_tangent_policy_explicit`. The provenance fields can therefore leave the struct later without changing the key, which matters because the key is used in consensus and caches (20547–20550, 32268).
- **Pure functions that move:** 64–114 and 316–346; `generatedInterfaceOptionsForActiveCut` and `volumeOptionsForCutMaintenance` (673–733); 1000–1101, including marker resolution, which A also places in LevelSetGeometry.
- **Stays in Application:** XML parsing (347–672 and 823–999); lexical helpers; declaration merging (191–315); the `requireResolved…` / `validateEquationLevelCutVolumeConsumer` messages (1103–1179), which use `origin`/`equation_type` and oopCout.
- **Enum token parsers (734–822).** FE has precedent (`parseLevelSetCurvatureRecoveryMode`), but these become lexicons of the FE/Core OptionSchema. Leave them until then.
- **The duplicate parse.** A notes that NavierStokesRegister parses the same free-surface keys (A §1.4). The schema removes the duplication.

### 2.4 Curvature samples

**Move whole** to FE/LevelSet/LevelSetCurvatureSamples.{h,cpp} (flat, next to LevelSetCurvatureProjection). Fold the driver's verbatim copies of the two mapping helpers (17027–17070) into it later (C3b). Identical code, so bitwise. The driver wrapper `collectLevelSetCurvatureSupplementalSamples` (17071–17291) follows in C3b with the curvature participant.

### 2.5 Equation translator and parsing: schema versus equation logic

LevelSetEquationTranslator.cpp by range:

| Lines | n | Content | Verdict |
|---|---:|---|---|
| 46–344 | 299 | temporal-spatial inflow file reader | stays (file I/O, B §7.4.4) |
| 345–711 | 367 | JSON effective-configuration artifact | stays; later generated by the FE schema serializer |
| 713–742, 860–867 | 38 | `LevelSetTransportInputAdapter` (PhysicsModule glue calling `ls::installLevelSetTransport`) | stays (thin) |
| **743–859** | **117** | initial condition from a mesh vertex field (MeshFields → `projectMeshVertexValuesToFieldCoefficients` → vector view) | **move**: FE/Systems initialization helper (A §3: `initializeFromMeshVertexFields`; the NS module has a twin in 879–1533) |
| 869–1011 | 143 | parameter lookup helpers | stays (OptionSchema later) |
| **1012–1119** | **108** | install options; element type, order and basis inference from the mesh | **move** to a shared helper. A says Physics/Core. I prefer FE/Spaces, because the level-set translator is in Application and FE must not depend on Physics. |
| 1120–1232 | 113 | enum parsers, input observations | stays |
| 1233–1684 | 452 | `apply_level_set_params` (key → `LevelSetTransportOptions`) | stays (schema) |
| 1685–1776 | 92 | projected-curvature field preflight and registration | split: registration → FE/Systems FieldBinding (A #13); names stay |
| 1777–2137 | 361 | BC translation, `translate_level_set_transport_input`, factory | stays |
| **2138–2218** | **81** | field-source mapping, scalar transport space check, preflight and ensure field | **move** → FE/Systems FieldBinding (A #13) |
| 2219–2363 | 145 | public API | stays |

**Totals:** ≈2,160 lines stay, ≈293 lines move. LevelSetConfigurationParsing (605 lines) stays.

### 2.6 Level-set work that should leave FE

- **Static capillary.** FE/LevelSet/LevelSetStaticCapillaryEquilibrium.* (1,740 lines) and driver 28667–30569 go to Physics (Plan §4.1, B S5).
- **Young-wall weights.** These live in LevelSetCurvatureProjection (`LevelSetKinematicAreaGradientYoungWall`, bound by the driver at 17604–17779) and go behind a Physics callback (B §1.4, R5).

## 3. Dependency hygiene

### 3.1 Current state and what each move must break

- **No reverse dependencies exist today.** `grep '#include "Application/\|"Physics/'` over FE/ finds nothing. FE includes Mesh unconditionally in a few headers and under `SVMP_FE_WITH_MESH` elsewhere.
- **Moves that need no dependency breaking (C1):** CurvatureSamples, the extension map, the PDE extension, NearestPointIndex, consensus part (a).
- **Moves that need dependency breaking:**

| Dependency (where) | Fix | Bitwise? |
|---|---|---|
| `const Parameters&` in the refresh family: 20531, 22804, 22914, 22935, 23126, 23144, 23168, 23282, 23656, 23690, 25170, 26474, 26968, 27277, 28964, 29421, 20139 (14 signatures, 19 `activeCutVolumeRequests(params)` calls) | Parse once in run{Steady,Transient}. Pass `std::span<const ActiveCutVolumeRequest>`, plus a `std::filesystem::path` output folder for the two artifact writers (20159, 26507). | yes: the parse is deterministic. It also removes a per-pass XML re-parse. |
| `SimulationComponents&` in every moving group | Pass explicit `FESystem&`, `const svmp::Mesh&`, `TimeHistory&`, `LinearSolver&` and snapshot-cache references | yes (mechanical) |
| `ActiveCutVolumeRequest` (an Application header) | Move the struct to FE/LevelSet/LevelSetGeometry.h; Application alias | yes |
| `LevelSetMaintenanceRequest` → `LevelSetMaintenanceConfigurationHandle` (Application) | FE `LevelSetMaintenanceOptions`; Application builds it from the handle | yes, if the fields are copied verbatim |
| `oopCout` / `oopDiagnosticsCout` (OopMpiLog.h) | Move the header to FE/Core/RankLog.h; Application forwarding header | yes (text identical). Do not switch to `FE_LOG_*`: that changes the log format and some log tokens are script-parsed (A §5.4). |
| `LevelSetMaintenanceWorkSubstage`, `LevelSetMaintenanceWorkLedger` (ApplicationDriver.h) | Enum → FE with an alias. The ledger stays outside FE and attaches through the pipeline observer. | yes |
| Env reads in moving code (`SVMP_REDISTANCING_TRIGGER_CELLS` 15896, `SVMP_DISABLE_CUT_CONTEXT_REINSTALL` 32308, `SVMP_TRACE_LEVEL_SET_ADVECTION` 19005, …) | Typed options resolved in Application SolverOptionResolution (B S13), following the FE/Core OptionSchema decision | default path bitwise |
| Contact stages (`FreeSurfaceAcceptedContactStageState`) feeding the conservative phase and wall-aware maintenance | Builder callback, as is already done (`ConservativePhaseContactStageBuilder` 25137) | yes |

### 3.2 CMake

**Today:**
- All Application files build into `svapplication` (Application/CMakeLists.txt 7–25), which links `svfe` and `svmesh` PUBLIC and tinyxml PRIVATE.
- `test_application` and `test_application_mpi` compile the Application sources directly (Tests/CMakeLists.txt 11–45, 76–98) and get FE through svphysics → svfe.

**After the moves:**
- **FE/CMakeLists.txt.**
  - Add to `FE_LEVELSET_HEADERS` / `FE_LEVELSET_SOURCES` (943–988).
  - Gate the Mesh-API files (Extension/*) with `if(FE_WITH_MESH) list(APPEND …)`, as for MovingMesh (1181–1191).
  - Add the headers to test_LevelSetHeaders.cpp. Add them to the umbrella LevelSet/LevelSet.h only under `SVMP_FE_WITH_MESH`.
- **No new link dependencies.** Eigen (`FE_HAS_EIGEN`), MUMPS (1829–1833, PUBLIC), MPI, Threads, OpenMP (1881, PUBLIC) and svmesh (1796–1800, PUBLIC) are already on svfe.
- **Application CMake.** Remove each moved .cpp from all three lists: svapplication, test_application and test_application_mpi. A file left in a test list while it is also in libsvfe.a gives duplicate definitions; with static archives the linker may silently pick one.
- **FE tests.**
  - Add to `FE_LEVELSET_TEST_SOURCES` (2487–2511).
  - MPI tests are globbed (`Tests/Unit/LevelSet/test_*MPI*.cpp`, 3262) into `test_fe_levelset_mpi`, whose main is test_mainMPI.cpp, and registered at 2 ranks with TIMEOUT 120.
  - The PDE extension's FourRank* tests need `add_fe_mpi_test(test_fe_levelset_mpi 4 FILTER "LevelSetPdeVelocityExtensionMPI.FourRank*" TIMEOUT 600)`, replacing `Application_PdeVelocityExtension_MPI_4` (Tests/CMakeLists 180–188).
- **Warnings.** FE targets compile with `-Wall -Wextra -Wpedantic -Wold-style-cast -Wsign-conversion -Wshadow …` (165–200). Errors are off by default (`FE_WARNINGS_AS_ERRORS` OFF), so moved code builds but warns. **Do not fix warnings in a move commit.**

### 3.3 Why pure moves are bitwise by construction, and the caveats

**Why:**
- The source tokens are identical apart from namespace and include lines.
- Codegen flags are global: `-O3`, and `-ffp-contract=off` from Code/CMake/SimVascularBuildOptimization.cmake 37–39.
- OpenMP flags already reach svapplication through svfe's PUBLIC link.
- svfe's only extra codegen-relevant setting is the PRIVATE `SVMP_FE_ASSEMBLY_TIMING`, and no header uses it.
- D16, re-verified on arm64 (e7080344-ltopgo-mumps against plain, 9/9), shows that cross-TU inlining changes do not change bits.

**Caveats:**
1. Keep function-local statics function-local: the PDE cache map (19520) and the `warned` set (LevelSetCutConfiguration 1138).
2. A namespace change can change ADL. I found no colliding names (`globalMax`, `globalSum`, `allGatherV`, `ownsVertex`, …) in FE headers. Compile errors would expose any that appear.
3. LTO+PGO profiles are keyed by object path, so moved objects lose their profile. Results stay bitwise (D16), but timing comparisons need a re-trained profile.
4. Forwarding shims preserve all test call sites until shim removal (C1z).

## 4. Staged move plan

**Checks:**
- **K1:** full build (`FE_BUILD_TESTS`, `PHYSICS_BUILD_TESTS`, `APPLICATION_BUILD_TESTS`; MUMPS on and off where the extension is touched).
- **K2:** CTest, with no new failures against /scratch/users/zsexton/svmp-integration-arm64/logs/failed-e7080344.txt.
- **K3:** refset 9/9 bitwise. Submitted by the coordinator: `sbatch refset.sbatch <bin> <tag> e7080344`; tools/cmp_runs.py.
- **K4:** 2- and 4-rank parity. Run the refset decks at `-n 2` and `-n 4` with the parent and the candidate and compare with cmp_runs.py; refset.sbatch needs a ranks argument. Plus `Application_FreeSurfaceRankConsistency_MPI_4`.
- **K5:** B's refset-ext (S0) for maintenance and conservative-phase steps.
- **K6:** pytest, with source-contract files updated in the same commit.

**Gate before every step:** both of these must be empty for the step's paths:
- `git -C /scratch/users/zsexton/svmp-dev-outerpass/source diff --stat e7080344 -- <paths>`;
- the same command for svmp-dev-mpiscale.

### C1. Pure file moves with forwarding shims

| Step | Files and lines | Destination and namespace | Tests that move | CMake | Checks | Effort | Timing |
|---|---|---|---|---|---|---|---|
| C1a | LevelSetCurvatureSamples (723) | FE/LevelSet/, `svmp::FE::level_set`; shim at Application/Core/LevelSetCurvatureSamples.h (≈15 lines of using-declarations) | test_LevelSetCurvatureSamples (212, 2 tests) plus the 4 curvature tests of test_LevelSetMaintenanceHistory (252–768) → FE/Tests/Unit/LevelSet/test_LevelSetCurvatureSamples.cpp | FE +2 sources, +1 test; App −3 list entries | K1–K4. K3 matters: 4 drop cases use KAG curvature. | 0.5 d | now |
| C1b | LevelSetMaintenanceHistory (73) | delete (or FE/Systems) | its 3 tests: delete or move | App −3 entries | K1, K2 | 0.25 d | now (coordinator decision) |
| C1c | NearestPointIndex.h (192) | FE/Geometry/, `svmp::FE::geometry`; driver using-declarations 2364–2365 | test_NearestPointIndex (124, 5 tests) → FE/Tests/Unit/Geometry | FE +1 header, +1 test | K1, K2 | 0.25 d | now |
| C1d | Consensus (a): 184–321 plus 28–37 (≈150) | FE/Systems/SystemCollectives.{h,cpp} `collectiveCanonicalWordsAgree(words, comm, log_prefix)`; Application keeps the old name as a one-line forward (6 call sites unchanged) | new FE 2-rank test; the ledger tests stay | FE +2 | K1, K2 (Consensus_MPI_2/4), K4 | 0.5 d | now |
| C1e | LevelSetVelocityExtensionMap (3,547) + LevelSetPdeVelocityExtension (2,629) | FE/LevelSet/Extension/, `svmp::FE::level_set`; shims for the 2 headers (≈60 lines) keep the driver (73 refs), the translator (1) and the workflow tests (139) unchanged | test_LevelSetPdeVelocityExtension (1,026, 19 tests), MPI twin (794, 6 tests) → FE/Tests/Unit/LevelSet | FE: `FE_WITH_MESH`-gated +4; 4-rank entry; App −3 entries and remove the MPI_4 entry | K1 (MUMPS on and off), K2, K3 (8/9 cases run pde_harmonic monolithic), K4; plus a sessile run with `SVMP_PDE_EXTENSION_SELF_CHECK=1` and the ltopgo-mumps binary with `SVMP_PDE_EXTENSION_FACTORIZATION=mumps_serial`, bitwise against the parent in the same config | 1.5–2 d | **after the outer-pass branch (dev/outer-pass-reuse) merges**, or once it confirms it does not edit these files (the PDE cache is its likely target) |
| C1z | shim removal: change the namespaces in tests and driver | — | — | — | K1, K2 | 0.5 d | after C1e and the outer-pass merge |

C1 total: ≈7,400 lines moved, ≈3.5 days.

### C2. Enablers (breaking dependencies; all bitwise)

| Step | Change | Lines | Timing |
|---|---|---|---|
| C2a | Struct and pure functions of the cut configuration → FE/LevelSet/LevelSetGeometry.{h,cpp} (≈400), with an Application alias. Move the typed tests (generated options, policy key, marker) out of test_LevelSetCutConfiguration; the parsing tests stay. | ≈400 moved | **now** (LevelSetCutConfiguration.* is cold) |
| C2b | `Parameters` → resolved requests and output path in 14 driver signatures (§3.1) | ≈150 edited | **hot**: coordinate with the outer-pass branch (dev/outer-pass-reuse), which may already be removing the per-pass `activeCutVolumeRequests(params)` re-parse |
| C2c | `SimulationComponents&` → FE references in the moving groups | ≈300 edited | hot |
| C2d | OopMpiLog.h → FE/Core/RankLog.h + forwarding | 118 moved | now |
| C2e | `LevelSetMaintenanceWorkSubstage` → FE enum + alias | 10 | now |

Effort ≈4 days. Checks K1–K4.

### C3. Driver extractions (splits), mapped onto B's steps

| Step (B) | Ranges | Destination | Lines out of the driver | Extra checks | Effort | Timing |
|---|---|---|---:|---|---|---|
| C3a (part of S9) | 14380–15224, 24007–24339 | FE/TimeStepping/FieldRepairPublication, FE/Systems/SystemCollectives | ≈1,150 | the 23 white-box refs compile through using-declarations; add an FE unit test of the plan algebra | 1.5 d | **now** (post-accept, cold) |
| C3g (S5) | 28607–30569; FE/LevelSet/LevelSetStaticCapillaryEquilibrium.* (1,740) | Physics/Formulations/NavierStokes/FreeSurface/StaticCapillaryEquilibrium | ≈1,960 + 1,740 from FE | K2 (13 + 4 white-box tests), K5 static drop | 2–3 d | now |
| C3b (S6 participant) | 4703–4725, 16966–18752 except 17604–17779 (→ Physics) | FE/LevelSet curvature participant; dedup with CurvatureSamples | ≈1,600 | K3 (drop KAG ×4) | 1 wk | after the outer-pass merge |
| C3c (S10) | 2368–2570, 4726–4775, 18946–20292 | FE/LevelSet/Extension `ls::extendField` participant. The cache moves into it. The registry moves; the artifact writer is given a path. | ≈1,500 | K3, K4, MUMPS-serial, self-check | 1 wk | after the outer-pass merge |
| C3d (S8) | 4830–5137, 5528–6716, 12669–12810, 18753–18945, 20293–23451 | FE/LevelSet/LevelSetGeometryService (B §7.1) | ≈5,500 | K3, K4, the exhaustive 2-rank matrix, the incremental-rebuild self-check | 1–1.5 wk | after the outer-pass merge |
| C3e (S9) | 3674–4702, 12811–16639, 23452–24006, 24340–24399 | FE/LevelSet/Maintenance/ (pipeline and stages) and FE/Systems/StepTransaction. Redistance-and-restore loop 15914–16002 → existing FE/LevelSet/LevelSetInterfacePreservingRedistancing.*. Trigger env → `RedistancingTrigger`. | ≈6,000 | **K5 is mandatory** (all 9 refset cases have maintenance off), plus the consensus MPI tests | 1–1.5 wk | after S0; partly overlaps `on_step_accepted` |
| C3f (S11) | 24400–28606 except the contact-stage builder | FE/LevelSet/ConservativePhaseWorkflow + LevelSetPhaseGeometryReconciliation. The artifact writer uses the existing FE LevelSetConservativePhaseArtifact. | ≈3,900 | K5 conservative phase; **K6: update the source contracts in run_free_surface_wp6_conservative_phase_qualification.py and test_free_surface_wp6_…runner.py in the same commit** | 1 wk | after C3d and C3e |

C3 total: ≈21.6k lines (≈19.9k from the driver, 1,740 from FE), 5–6 developer-weeks, shared with B's S5 and S8–S11.

**Source-contract pytest files** must be updated as extractions land:
- tests/test_fs16_physical_instrumentation.py 63–90: requires the order refresh → curvature → wet diagnostics in the driver text;
- the wp6 runner and its test;
- run_free_surface_wp8_energy_qualification.py.

### C4. Application consolidation

- Move LevelSetCutConfiguration (parsing remainder), LevelSetMaintenanceConfiguration, LevelSetEquationInputSnapshot.h and ResolvedMovingDomainConfiguration.h into Application/Translators/. Cosmetic; bitwise.
- Rebuild `levelSetAdvectionVelocityRequests` (driver 5203–5527) from the resolved translator configuration in FreeBoundaryBinding (B §5). **Not bitwise by construction**: alias precedence may differ, so add characterization tests first, as R1 did.
- Move the translator's ≈293 lines of FE logic (§2.5) to FE/Systems and FE/Spaces.
- The OptionSchema migration of the alias tables follows DECISIONS.md.

Effort ≈1 week.

## 5. Interactions

### 5.1 With A and B

I adopt B's destinations and A/B §7 names:
- `ls::LevelSetGeometry` and `LevelSetGeometryService` (FE/LevelSet);
- `LevelSetMaintenancePipeline` (FE/LevelSet/Maintenance);
- `ls::extendField` / FieldExtension (FE/LevelSet/Extension);
- GeneratedStateRegistry and FieldRepairPublication (FE/TimeStepping);
- StepTransaction and SystemCollectives (FE/Systems);
- SolverOptionResolution and FreeBoundaryBinding (Application).

My C-steps are pre-steps or sub-steps of B's S5 and S8–S11:
- C1 and C2 lower the cost of S8–S10, because the files are already in FE when the services are built.
- C3a and C3g can run in parallel with B's S0–S5.

A's §2.3 statement `L.requireMaterialInterface(u_id)` relies on level-set kinematics being owned by FE/LevelSet. C3c and C3e deliver that.

### 5.2 With the parallel performance branches

- **Outer-pass branch** (svmp-dev-outerpass, dev/outer-pass-reuse; no commits yet). **Hot zones:** driver 4830–5137, 6121–6716, 16966–23451, 30925–30972 and 32261–32526; LevelSetPdeVelocityExtension.* (PDE cache); possibly FE/LevelSet/LevelSetInterfaceLifecycle.cpp and NewtonSolver `solveStep`. C1e, C2b, C2c and C3b–d wait for its merge.
- **MPI-scaling branch (dev/aggregation-ledger-local)** (svmp-dev-mpiscale, e28c86ce). It touches only FE/Constraints/SmallCutAggregationConstraint.cpp. No step of this plan edits that file. FE/CMakeLists.txt edits are in different blocks.
- **Order now:** C1a, C1c, C1d, C2a, C2d, C2e, C3a, C3g (and C1b once decided). After the outer-pass merge: C1e → C1z → C2b/C2c → C3b → C3c → C3d. After S0: C3e → C3f.

## 6. Points for the coordinator

2. **D7** (tracker line 285) still pauses R3–R12. The level-set moves are R6/R7. C1/C2 could be argued to be "targeted extractions", but that needs the user.
3. **Disagreements with B:**
   - (a) LevelSetMaintenanceHistory is dead: delete it, not Maintenance.
   - (b) Consensus splits into SystemCollectives plus a ledger decision.
   - (c) Driver 4830–5137 is cut-topology epochs (LevelSetGeometryService / GeneratedState), not FieldExtension.
   - (d) Driver 12669–12810 goes to LevelSetGeometryService.
   - (e) NearestPointIndex goes to FE/Geometry.
   - (f) The C1e file moves need not wait for all of S10, only for the outer-pass branch's confirmation that it does not edit those files.
   - (g) B's FE/FreeBoundary is a new top-level folder. If FreeBoundaryWorkflow composes only level-set services (B withdrew the fitted handle, §7.4.2), the existing FE/LevelSet would do. Keep FE/FreeBoundary only if it also composes FE/MovingMesh.
4. **Dead code** (user): `copyFieldDofsIntoFeOrderedSolution` (73 lines plus 3 tests) and `serializeLevelSetMaintenanceCompatibility` (≈150 lines, test-only). Delete or keep?
5. **OptionSchema overlay** (follow-up to DECISIONS.md). FE-side schemas for FE option structs should carry canonical keys, types, defaults and validators. The legacy XML spellings (for example the 6 `Transport_form` aliases in LevelSetConfigurationParsing.h 115–118) and the LegacyMaintenance policy should stay as an Application-registered alias overlay, so that FE stays input-format-agnostic. Please confirm with the user.
6. **Test isolation.** The function-static PDE cache (driver 19520) persists across tests in test_application. Note it now; it is fixed by C3c.
7. **Evidence gap.** The 9 refset cases cover the PDE extension, KAG curvature and one contact case, but no maintenance, conservative phase or static capillary. B's S0 is required before C3e and C3f.
8. **Question for the outer-pass branch (dev/outer-pass-reuse).** Will it touch LevelSetPdeVelocityExtension.cpp, LevelSetCurvatureSamples.cpp or LevelSetCutConfiguration.cpp, or remove the 19 `activeCutVolumeRequests(params)` re-parses? The answer decides whether C1e and C2b run before or after its merge.
9. **Refset runner.** refset.sbatch runs serial only. K4 needs a ranks parameter (or a sibling script) to run the decks at 2 and 4 ranks against parent runs.

### Critical files for implementation
- /home/users/zsexton/svMultiPhysics/Code/Source/solver/Application/Core/ApplicationDriver.cpp
- /home/users/zsexton/svMultiPhysics/Code/Source/solver/Application/Core/LevelSetPdeVelocityExtension.cpp (with LevelSetVelocityExtensionMap.cpp)
- /home/users/zsexton/svMultiPhysics/Code/Source/solver/Application/Core/LevelSetCutConfiguration.h
- /home/users/zsexton/svMultiPhysics/Code/Source/solver/FE/CMakeLists.txt (FE_LEVELSET lists 943–988, tests 2487–2545, MPI glob 3262–3285) and Application/CMakeLists.txt with Application/Tests/CMakeLists.txt
- /home/users/zsexton/svMultiPhysics/Code/Source/solver/FE/LevelSet/LevelSetVelocityExtensionConstraint.h
