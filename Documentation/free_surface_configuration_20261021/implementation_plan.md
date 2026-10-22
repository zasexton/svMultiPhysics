# Implementation plan for D47: reducing the free-surface configuration

2026-10-22. This plan implements `reduction_proposal.md` with the user's answers: Q1(a), Q2(b) gated on the fitted γ = 0 manifest, and Q3–Q13 as recommended. The step names P0–P6 follow proposal section 6.4.

**Common rules for every step:**

- **Worktree:** `/scratch/users/zsexton/svmp-dev-params/source`, one branch per step: `dev/fs-config-p0`, `-p1`, and so on.
- **Commits:** author and committer Zachary Sexton <zsexton@stanford.edu>, no trailers.
- **Names:** no new `SVMP_*` names.
- **Running:** no builds or runs outside Slurm. The coordinator submits every job.
- **Commit size:** each commit is one concern, includes its tests, and builds and passes on its own.

## 0. Verification used by every step

| Gate | What it compares | Tool |
|---|---|---|
| G0 static scan | No generator, deck, Python file or C++ test uses a spelling that the commit deletes. The deletion list is a file in the commit message's evidence (`p0_spelling_manifest.tsv`). | `scripts/p0_spelling_manifest.py`, `usage_counts.py` |
| G1 resolved configuration | `effective_configuration.json` of every corpus deck, old against new, compared leaf by leaf. Only the listed intended entries may differ (for P0: `input_deprecations`). | `r2_cmp_artifacts.py` (R2), plus an allow-list filter |
| G2 S0 manifests | Ordered kernel and constraint manifests of the 16 reference decks: 16/16 identical, 0 JIT object-cache misses. | `rf_manifest.sh`, `rf_cmp_manifests.sh` (R1), copied with the work directory changed |
| G3 reference sets | refset9 and refset-ext serial, against `e7080344`; at 2 and 4 ranks, against `pwd-92f1a2fc`. | `int_evidence.sbatch <bin> <tag> pwd-92f1a2fc` |
| G4 tests | CTest FE, Physics and Application with no new failures against integration 4. The edited tests are named in the commit. | the configure and CTest lines of `int_build_ctest.sbatch` |
| G5 (P2 onwards) | Exhaustive two-rank MPI matrix; 2/4-rank parity. | the rf_verify last-commit stage |

**The corpus for G1.**

- The 16 reference decks. Their artifacts come from the G3 runs at no extra cost.
- From P0.0 on, the resolve-only mode (section 1) extends G1 to:
  - the default-level deck of each of the 12 generators;
  - the capillary-rise generator;
  - the free-surface decks under `tests/` whose meshes are present.

  This needs setup only: no solve.

The job script is `/scratch/users/zsexton/svmp-dev-params/jobs/p_verify.sbatch` (4 tasks, 8 GB). It follows the structure of rf_verify:

1. For each commit: an incremental build, the named unit tests, refset9 and refset-ext serial, G2 against the first commit, and G1 for the 16 decks.
2. For the last commit: the resolve-only corpus (G1), CTest (G4), and `bash int_evidence.sbatch` for the parity runs (G3 at 2 and 4 ranks).

## 1. The steps

### P0: dead spellings, the initial-rate key, the certificate defect

**Can start as soon as the integration-4 hash arrives, except P0.4.** Base: integration 4 (expected `b74ea15c`).

| Commit | Content | Files | Possible conflicts |
|---|---|---|---|
| P0.0 | **Resolve-only mode.** A command-line flag `--resolve-only` makes `ApplicationDriver::run` stop right after `writeEffectiveConfigurationArtifact` (driver line about 25759 at `490f03aa`). Without the flag the run is unchanged. | `main.cpp`, `ApplicationDriver.{h,cpp}` (around `run` / `runWithParameters` only) | none (S11 and outer-pass touch other driver regions) |
| P0.1 | **Delete 455 accepted spellings** from the acceptance lists and from every parser's alias list, for free-surface items only; general items are untouched. 365 of them are used nowhere. 90 are used only by C++ tests, which are edited to the canonical spelling in the same commit. Breakdown: 301 CamelCase, 142 secondary spellings, and 12 unused canonical spellings of FIX, RETIRE or ALIAS items. Examples of the last group: `SUPG_velocity_epsilon`, `Kinematic_nitsche_symmetric`, `Curvature_projection_neighbor_rings`, `Cut_domain_type`. The aliases `Pressure` and `Value` leave the free-surface `External_pressure` list; no free-surface BC in the corpus uses them, and `Value` stays a standard BC key. | `Parameters.cpp` (both lists), `NavierStokesRegister.cpp`, `LevelSetConfigurationParsing.h`, `LevelSetEquationTranslator.cpp`, `LevelSetMaintenanceConfiguration.cpp`, `LevelSetCutConfiguration.cpp`, `EquationTranslator.cpp`, `ApplicationDriver.cpp` (`levelSetAdvectionVelocityRequests` lists only); tests `test_NavierStokesLegacyBCs`, `test_LevelSetCutConfiguration`, `test_EquationTranslator`, `test_ApplicationDriverLevelSetWorkflows{,MPI}`, `test_OpenVesselExamples`, `test_LevelSetMaintenanceConfiguration`, `test_LevelSetEquationTranslator`, two FE analysis tests | **MPI-scaling step B** (`cd80dc8c`, not integrated): it adds `Generated_cut_ghost_layers`/`GeneratedCutGhostLayers` to both lists and edits `LevelSetCutConfiguration.cpp`. Whichever lands second takes a trivial textual merge; ask that agent to add only the first spelling. **Partition (integration 4)** edits `Parameters.cpp`/`.h` in `MeshParameters`, a different section. **R1** does not touch these files (its diff `490f03aa..8e1521dd` is limited to the NS module directory and leaves `NavierStokesRegister.cpp` alone). |
| P0.2 | **Deprecation warnings** for the 11 spellings still used by a generator, deck or Python file. Each stays for one release, then becomes an error. They are `Generated_interface_quadrature_order`, `Advection_velocity_from_field`, `Wet_extension_advection_velocity_method`, `Allow_corner_linearized_geometry`, `Projected_curvature_field`, `Free_surface_curvature_field`, `FreeSurfacePhysicalModel`, `CurvatureFieldName`, `CurvatureField`, `ProjectedCurvatureField` and `FreeSurfaceCurvatureField`. The users are old MMS, SPHERIC and tank decks, the wet-extension options of the sessile, sloshing and capillary-rise generators, `run_validation_matrix.py` and the WP-10 capability runner, all of them frozen history. Each spelling prints one warning on rank 0 that names the canonical spelling. A `Parameters`-level registry records it, and the driver adds it to `effective_configuration.json` as a top-level `input_deprecations` array, written only when it is non-empty. | `Parameters.{h,cpp}`, `ApplicationDriver.cpp` (`writeEffectiveConfigurationArtifact`), one new unit test | as P0.1 |
| P0.3 | **New key `Generalized_alpha_initial_rate`** (`Consistent` \| `Zero`, default `Consistent`) in `GeneralSimulationParameters`. It replaces `SVMP_GENERALIZED_ALPHA_PDE_UDOT_INIT`, which stays a legacy alias for one release: it warns and is recorded in `input_deprecations`. If the variable and the key are both set and disagree, the run stops. | `Parameters.{h,cpp}`, `ApplicationDriver.cpp` (`generalizedAlphaPdeRateInitializationRequested` and its caller), a unit test | as P0.1 |
| P0.3b | **Capillary-rise generator.** `REQUIRED_ENVIRONMENT` is replaced by the key, and `test_capillary_rise_case.py` is updated. The frozen wp5 protocol JSON (v1) is left unchanged as history. A copy of `cases-ext/capillary_rise_L10` with the key and no `env.sh` is placed in `svmp-dev-params/decks/` for the evidence. Once the coordinator accepts it, it replaces the reference deck. | `tests/cases/fluid/open_vessel_free_surface/capillary_rise_2d.py`, `tests/test_capillary_rise_case.py` | none |
| P0.4 | **The static-capillary certificate defect. Waits for R1's split to be integrated.** The conservative-balance diagnostic operators are installed whenever the deck configures the static-capillary initialization (`Enable_static_capillary_equilibrium_initialization=true`), not only when `SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC` is set. The NS options get a flag that the Application binding sets from the resolved maintenance configuration; the environment name stays an observation switch. | `FreeSurface/FreeSurfaceDiagnosticOperators.{h,cpp}` (R1's new file), `IncompressibleNavierStokesVMSOptions`, the Application binding (`LevelSetMaintenanceConfiguration.cpp` or `SimulationBuilder.cpp`) | **R1** (`FreeSurfaceDiagnosticOperators` is its new file): rebase onto the integrated R1 |

**What P0.4 changes, and where.** The defect is that the Newton probe of the startup certificate (`evaluateStaticCapillaryPressureCertificate`, which sets `read_static_free_surface_environment_options=false`) reads its representability and KKT distances from these operators, and they exist only when the variable is set.

- **Runs with static-capillary initialization and without the variable.** Today the certificate's gates cannot pass, and the startup minimizer reports the certificate as failed. After the fix the operators exist and the gates are evaluated. The residual and Jacobian of the time steps are unchanged: the operators are observation-only, installed under their own operator tags. Their kernels do appear in that deck's kernel manifest.
- **Runs with the variable set** are unchanged.
- **Runs without static-capillary initialization** are unchanged. This covers every reference deck and every accepted benchmark.
- **Affected inputs.** Only the C++ level-set workflow tests and the two Python runners that use static-capillary initialization (`run_test05_velocity_growth_smoke.py` and `test_open_vessel_smoke_gates.py`). The commit message will name them after reading the post-R1 code.

**P0 evidence.**

- G0 for P0.1.
- G1: the 16 decks identical. Only the old `capillary_rise_L10` deck with `env.sh` gains one `input_deprecations` entry. The resolve-only corpus is identical apart from `input_deprecations` entries on the decks that use the 11 deprecated spellings.
- G2: 16/16 manifests identical.
- G3: refset9 9/9 and refset-ext 7/7 serial against `e7080344`; parity against `pwd-92f1a2fc`. The capillary-rise copy with the key must be bitwise identical to the `env.sh` baseline.
- G4: no new failures; the edited tests are listed.
- P0.4 is verified separately: G2 shows no change on the 16 decks, and a manifest of one static-capillary deck shows the added diagnostic kernels and no change to the residual kernels.

### P1 = S13a: option schema and full resolved manifest

- **P1.1 can start now.** It adds only new files: `FE/Core/OptionSchema.h` and `FE/Core/OptionSchema.cpp`, with the member pointer, ordered aliases, typed lexicon, default, validator, "implies" and "conflicts" rules, legacy-only acceptance, artifact key and documentation (formulation design §4). Unit tests come with it. Nothing in the code base uses it yet.
- **P1.2 waits for R1 to be integrated.** It extends the artifact to every field of `FreeSurfaceBoundary`, `LevelSetTransportOptions`, `ActiveCutVolumeRequest`, the maintenance and extension requests, the outer and Newton options, and the environment values that were read. Each field carries its provenance (explicit, default, derived or legacy alias). The new data goes under a `resolved` section, so the existing v1 keys are unchanged.
  - Files: `FreeSurfaceConfigurationArtifact.cpp` (R1's new file), the artifact part of `LevelSetEquationTranslator.cpp`, the driver writer.
  - Evidence: G1 on the v1 projection is identical; the v2 manifests of the corpus are recorded as the baseline; G2, G3, G4.
  - One commit per struct family.

### P2 = S13b: free-surface BC schema

**Waits for R1 and MPI-scaling step B to be integrated, and needs P1.**

- **Content.**
  - `FreeSurface/FreeSurfaceSchema.{h,cpp}` declares the BC schema of proposal 3.1, 3.2 and 3.6.
  - One parser replaces `append_free_surface_bc`, `append_free_surface_contact_line`, the enum parsers (register 2784–3116) and the BC branch of `activeCutVolumeRequestFromParameters`.
  - `Application/Translators/FreeSurfaceLegacyAliases.cpp` holds the overlay for the P0.2 spellings and the retired-value messages.
  - The string-typed geometry and tangent fields become enums.
  - The schema-1 legacy mode and its about 20 branches are deleted. `test_NavierStokesLegacyBCs` and `test_FittedFreeSurfaceALE`'s schema-1 cases become tests of the retirement messages.
- **Files.** `NavierStokesRegister.cpp`, `FreeSurfaceOptions.h`, `FreeSurfaceValidation.cpp` (R1), `LevelSetCutConfiguration.cpp` (MPI-scaling), `EquationTranslator.cpp`, and the tests.
- **Evidence.** G1 on the full v2 corpus with provenance removed, then G2–G5.
- **Commits.**
  1. schema and parser, keeping the old function bodies as thin calls;
  2. delete the old parser;
  3. delete the legacy mode;
  4. turn the strings into enums.

### P3 = S13c: level-set equation schema

**Waits for P1. Coordinate with whoever takes S9 and S10, since the driver's extension request belongs to S10.**

- **Content.** One schema for `LevelSetTransportOptions`, the maintenance configuration, the extension request, conservative phase (on S11's FE location) and static capillary. It replaces:
  - `apply_level_set_params`;
  - the second parse in `resolveLegacyLevelSetMaintenanceConfiguration`, which then consumes the resolved schema;
  - the parse inside `levelSetAdvectionVelocityRequests` (driver 5050–5374). The function itself stays where S10 will move it.
- **Files.** `LevelSetEquationTranslator.cpp`, `LevelSetConfigurationParsing.{h,cpp}`, `LevelSetMaintenanceConfiguration.cpp`, `ApplicationDriver.cpp` (parse lines only).
- **Possible conflicts.** S11 has finished. The outer-pass agent works in `StandardAssembler` and `AssemblyContext`, not here. No other agent is active in these files today.
- **Evidence.** As for P2.
- **Commits.** Transport, then maintenance, then extension, then conservative phase and static capillary. Each one deletes its old parse lines.

### P4: profiles, generators and decks

**Waits for P2 and P3.**

- **Content.**
  - `FreeSurface/FreeSurfaceProfiles.{h,cpp}` defines `Capillary@1`, `Wetting@1`, `Gravity_wave@1`, `Fitted_ALE@1` and `None`, with the values of proposal section 4.
  - The new keys `Free_surface_profile`, `Level_set_maintenance`, `Level_set_initial_condition` and `Kinematic_area_gradient_mass` on the BC.
  - The overlay warnings for the profile-folded keys, about 25 of them: the D9 set, the maintenance booleans, the KAG chain, SUPG, the ghost penalty and the geometry.
  - The 12 generators and `capillary_rise_2d.py` emit the profile and the physics, and keep `--explicit` for one cycle.
  - A script converts the `tests/` decks.
  - New reference decks go in `cases-v2` and `cases-ext-v2`, next to the old ones, which become the overlay test set.
- **Files.** The generators and their pytests, plus new Physics files. Possible conflict: benchmark agents editing a generator; coordinate per generator.
- **Evidence.**
  - G1 with the same binary: the `--explicit` deck and the profile deck must resolve identically once provenance is removed. This holds for every generator at every CLI variant, and for the regenerated reference decks against the old ones.
  - G2, G3 with the v2 decks against the existing baselines, and G4.
- **Commits.** Profiles; then one commit per generator family; then the decks.

### P5: environment names, family by family

The work is done family by family, each when its file is free:

| Family | Files | Wait for |
|---|---|---|
| Time integration and outer fixed point: the `Outer_fixed_point_*` and predictor environment twins deleted, the Aitken factors made constants, the key applied to steady solves, `TIMELOOP_*` deleted, the absorb fraction made a constant | `ApplicationDriver.cpp`, `TimeLoop.cpp` | can start after P0 |
| Newton: line search and pseudo-transient continuation become on/off keys, the rebuild period becomes a key, 11 constants, the static-certificate gates are derived | `NewtonSolver.cpp`, `Parameters` | S3 (Newton hooks), or before it to shrink what it moves |
| Level set: the redistancing trigger moves to the registry; the extension factorization becomes a key; the retention force and the per-step-only refresh retire | `ApplicationDriver.cpp`, `LevelSetPdeVelocityExtension.cpp`, `LevelSetCutConfiguration.cpp` | MPI-scaling step B (cut configuration) |
| Aggregation experiments retire | `SmallCutAggregationConstraint.cpp` | MPI-scaling agent's file: after step B integrates |
| Cut numerics and assembly: `MIN_GENERATED_CUT_VOLUME_FRACTION` becomes a constant; cut-volume fusion moves to the performance registry | `CutIntegrationContext.h`, `SystemAssembly.cpp`, `StandardAssembler.cpp` | outer-pass agent's file: after round 3 integrates |
| Registry: the FE DiagnosticRegistry (D42), with diagnostic, performance and test-hook kinds and one legacy environment adapter that records every use in `effective_configuration.json`; then the 114 names, family by family | `FE/Core/DiagnosticRegistry.{h,cpp}` (new), plus every reading file | the registry core can start now (new files); the conversions follow the families above |

**Evidence.**

- G1 and G2: the default path is unchanged, because no environment name is set in the reference runs apart from the capillary-rise one.
- For each name that becomes a key, an A/B pair: the run with the variable against the run with the key, bitwise, on a deck that exercises it.
- G3, G4.

**Commits:** one per family.

### P6a: retirements decided under Q2, Q5, Q8 and Q9

**Waits for R1 to be integrated (`FreeSurfaceLegacyTerms.cpp` holds the CurvatureTraction rows and the fitted Penalty/Nitsche rows) and for P2.**

| Commit | Content | Gate |
|---|---|---|
| Q2(b)-0 | Kernel manifest of the fitted γ = 0 sloshing deck, with `Automatic` resolved to SurfaceStress instead of CurvatureTraction | must be identical; otherwise fall back to Q2(a) and report |
| Q2(b) | Retire CurvatureTraction, the 11 least-squares curvature settings, `Curvature`, `Use_level_set_curvature` and `Allow_fitted_surface_stress`. About 25 C++ test uses move to SurfaceStress where they test contact physics, or retire with the route: 24 in `test_MovingDomainPhysics`, and one each in `test_SharpBoundaryOperatorMPI`, `test_FreeSurfaceSemiImplicitSurfaceTension`, `test_FreeSurfaceCutStability` and `test_OpenVesselExamples`. The commit lists each test. MPI-scaling's "every ghost layer for LevelSetQuadratic" fallback loses its trigger and is deleted with it. | G2 and G3 bitwise (no accepted benchmark uses the route); G4 with the listed tests |
| Q5(b) | Retire reinitialization, global volume correction, the algebraic wet extension (`LevelSetVelocityExtensionMap.cpp`) and discontinuity capturing. The bound-preserving limiter stays as one advanced switch with fixed tolerances. | G2, G3, G4; the old SPHERIC and MMS decks are marked in the tracker as needing the current protocol |
| Q8(b) | Take the implicit-cut backend keys out of the free-surface schema; FE keeps the capability. | G2, G3, G4 |
| Q9(a) | Retire the fitted Penalty and Nitsche kinematics, with `fitted_sloshing --kinematic-enforcement`. | G2, G3 (fitted_capwave), G4 |

Each retirement is one commit for the tests and one for the code, as R2 did for D42.

### P6b: default changes

Nothing is scheduled. Q1(b) (deriving the profile from the physics) and Q7(a) (a uniform cut order of 2) are deferred by decision. Each will be a separate commit with a new arm64 baseline, judged against the spread between rank counts.

## 2. What can start when

- **Now:**
  - P0.0–P0.3 on integration 4, as soon as the hash is pushed;
  - P1.1 (OptionSchema core, new files only);
  - P5 registry core (new files only).
- **After R1's split is integrated:** P0.4, P1.2, then P6a.
- **After MPI-scaling step B is integrated:** P2, the level-set family of P5, and the aggregation family of P5.
- **After the outer-pass round 3 is integrated:** the cut-numerics family of P5.
- **After P2 and P3:** P4.

## 3. P0 verify line (≤ 4 CPUs)

Once the hash arrives and the P0.0–P0.3 commits exist, I will send the exact line. It will have this form:

```
sbatch --export=NONE --ntasks=4 --cpus-per-task=1 --mem=8G --time=04:00:00 \
  /scratch/users/zsexton/svmp-dev-params/jobs/p_verify.sbatch -T \
  -p pwd-92f1a2fc -f <integration-4 sha8> <integration-4 sha8> <P0.0> <P0.1> <P0.2> <P0.3> <P0.3b>
```

- The first commit is the integration-4 tip itself. It provides the manifest and artifact reference and warms the JIT cache, as in rf_verify.
- At the last commit the script calls `bash /scratch/users/zsexton/svmp-integration-arm64/jobs/int_evidence.sbatch <bin> p0-<sha8> pwd-92f1a2fc`. Serial runs are compared with `e7080344`; the 2- and 4-rank runs with `pwd-92f1a2fc`.
- Results go to `/scratch/users/zsexton/svmp-dev-params/logs/summary.txt`, one line per stage.
