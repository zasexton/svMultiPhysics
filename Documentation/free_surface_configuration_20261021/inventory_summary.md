# Free-surface configuration inventory: summary

2026-10-21. Tree: `/scratch/users/zsexton/svmp-integration-arm64/int-src` at `490f03aa`, which is the main line `8784268a` plus 22 commits. The machine-readable table is `inventory.tsv`, with one row per distinct configuration item; its 27 columns are listed in section 7. The reduction design is in `reduction_proposal.md`. The work was read-only: no builds, solver runs or jobs.

## 1. Counts

| Layer | Distinct items | Accepted spellings | Used by a benchmark generator or reference deck | Value enters every run (`yes`) | Opt-in (`only-if-set`) | Dead |
|---|---:|---:|---:|---:|---:|---:|
| (a) `Free_surface` BC keys | 74 | 293 | 34 | 26 | 42 | 6 |
| (b) level-set equation, cut geometry, fluid-equation extensions | 156 | 458 | 56 | 34 | 108 | 5 |
| (c) mesh and partition | 5 | 5 | 1 | 1 | 4 | 0 |
| (d) `GeneralSimulationParameters` | 15 | 22 | 5 | 9 | 5 | 0 |
| (e) internal option fields | 56 | – | – | 26 | 22 | 6 |
| (f) `SVMP_*` environment names | 412 | 473 | 3 | 0 | 159 (+23 round-off only) | 4 not env vars |

How the counts relate to the coordinator's first count:

- **(a)** `append_free_surface_bc`, together with `append_free_surface_contact_line` and `free_surface_implementation_from_params`, reads 55 items under 223 spellings. The other 19 rows of layer (a) are keys that the BC acceptance list also takes:
  - the cut-request keys, which `LevelSetCutConfiguration::activeCutVolumeRequestFromParameters` reads from the same BC;
  - the mesh-motion BC keys;
  - the level-set inflow keys.

  `Parameters.cpp::is_oop_boundary_extension_parameter` has 293 entries: 291 distinct spellings (two are listed twice), or 153 after folding case and underscores.
- **(b)** `is_oop_equation_extension_parameter` accepts 526 spellings, 290 after folding. The parsers read them as 153 alias groups, and three more items are read but not accepted (section 2.4).
- 22 rows of (a) and (b) are not free-surface items: the mesh-motion and two generic fluid-equation families, marked `GENERAL`. That leaves **208 free-surface XML items under 689 spellings**.
- **(f)** 412 distinct names appear as string literals in production code at `490f03aa`; the main line has 440. The 28 names the integration tip dropped are the PSPG, VMS-override, probe, shape-tangent and symmetric-Nitsche names it retired. Of the 412:
  - 213 are read on a free-surface run path (driver, Newton, FE assembly and constraints, extension, mesh);
  - 199 belong to the FSILS, Eigen, PETSc, Trilinos and JIT backends;
  - 4 are not environment variables: two packet headers, a restart-file magic string and a compile-time macro.

## 2. Findings that drive the reduction

### 2.1 Most of the surface is never used

- **Spellings.** Of the 721 accepted XML spellings, 396 are used nowhere: not in a generator, a deck, a Python runner or test, or a C++ test. Only 95 are used by a benchmark generator or a reference deck.
- **Benchmarks use canonical spellings only.** Every one of those 95 is the canonical `Snake_case` spelling of its item. The only exceptions are generic words such as `Pressure` and `Field_name`, which other sections of the deck use as well.
- **Secondary spellings.** Of 521 secondary spellings, 127 appear anywhere. 98 of them appear only in C++ alias-acceptance tests: `test_NavierStokesLegacyBCs`, `test_LevelSetCutConfiguration`, `test_LevelSetMaintenanceConfiguration`, `test_EquationTranslator` and `test_ApplicationDriverLevelSetWorkflows`. About ten appear in old decks or runners, for example:
  - `Wet_extension_advection_velocity_method`
  - `Advection_velocity_from_field`
  - `Generated_interface_quadrature_order`
  - `Projected_curvature_field`
- **Environment.** 302 of the 412 names are set by no deck, runner or test. Only three are set by a reference deck's `env.sh`:
  - `SVMP_GENERALIZED_ALPHA_PDE_UDOT_INIT=0`, by capillary rise;
  - `SVMP_CONTACT_POINT_VELOCITY_TRACE`, by the strong Ren–E deck;
  - `SVMP_OOP_SOLVER_TRACE=0`.

### 2.2 The protocols restate the method in every deck

- **Key counts.** A reference deck carries 15 to 43 free-surface keys, with a median of 33. The 2D sessile protocol deck has 42: 17 on the level-set equation and 25 on the BC.
- **Identical values.** The 13 unfitted decks give the same value for almost every numerical setting. These values are the method, not case data:
  - `Active_domain_method=CutVolume`, `Generated_interface_geometry=LinearCorner`, `Geometry_tangent_policy=RefreshedFrozenQuadrature`;
  - `Interface_quadrature_order=2`, `Level_set_isovalue=0`;
  - `Enable_cut_cell_stabilization=true`, `Cut_cell_pressure_gradient_penalty=1.0`, `Use_cut_metadata_scale=false`, `Small_cut_aggregation=true`, `Enable_velocity_extension=false`;
  - `Enable_SUPG=true`, `SUPG_tau_scale=0.5`, `SUPG_transient_scale=2.0`;
  - `Enable_reinitialization=false`, `Enable_volume_correction=false`;
  - the nine advection-velocity keys of decision D9.
- **Protocol defaults differ from code defaults.** The code defaults are maintenance off, SUPG off and ghost penalty off. The generators default kinematic reconciliation on for every moving benchmark, plus sign-definite bounds and interface-preserving redistancing for contact lines. `Surface_tension_semi_implicit=NormalIncrement` is the default for the drops, waves and the sphere. So each generator writes the whole method out.
- **The nine-case reference set uses neither.** It enables none of the maintenance options, so it follows neither the code defaults nor the protocol defaults; see the `cases-ext` README.
- **One deck depends on the environment.** The capillary-rise protocol requires `SVMP_GENERALIZED_ALPHA_PDE_UDOT_INIT=0` (`capillary_rise_2d.py` `REQUIRED_ENVIRONMENT`), and no deck key exists for it. The tracker already says it should become one.

### 2.3 The same keys are parsed in up to five places

- **Duplicate parsers.**
  - 76 of the 230 items in (a) and (b) are read by two or more parsers, 12 of them by three or more.
  - `Level_set_field_name` is parsed in five files: NavierStokesRegister, LevelSetCutConfiguration, the level-set aliases table, the driver's `levelSetAdvectionVelocityRequests`, and MeshMotionRegister through `Field_name`.
  - Every transport, maintenance and conservative-phase key is read twice: `apply_level_set_params` (installation policy) and `resolveLegacyLevelSetMaintenanceConfiguration` (legacy policy).
  - The cut keys are read by both `append_free_surface_bc` and `activeCutVolumeRequestFromParameters`.
  - The default of `Small_cut_aggregation` (true) is written in three files, with comments asking that they be kept in sync.
- **Inconsistent value parsing.** On/off parsing differs between call sites:
  - in the parameter maps, `parse_bool_relaxed` maps unknown text to false while the strict lexicons throw;
  - about 18 environment switches treat `false` or `0` as on, or empty text as on.

### 2.4 Dead and unreachable configuration

**Read by a parser but rejected by `Parameters.cpp`** (so no deck can set them):

- `Free_surface_configuration_schema_version`
- `Enable_explicit_legacy_free_surface_configuration`, which is the explicit schema-1 legacy mode
- `Discontinuity_capturing_residual_epsilon`
- six level-set basis-type spellings

Consequence: about 20 "accepted only by the explicit schema-1 legacy mode" branches in `append_free_surface_bc` are reachable only from `test_NavierStokesLegacyBCs.cpp`.

**Accepted but only able to throw, or to restate the retired value:**

- `Enable_velocity_extension` and `Velocity_extension_diffusivity`, retired by D42; 13 decks restate `false`.
- `Use_current_geometry_curvature` (D42).
- `Cut_cell_velocity_gradient_penalty` and `Cut_cell_velocity_max_derivative_order`; the velocity ghost penalty was retired.
- `Use_level_set_curvature`: 11 decks write `false`, and no current capillary form reads it.
- `Conservative_phase_impermeable_normal_velocity_tolerance`: setting it throws.
- `Active_domain_method`: only the value `CutVolume` is left.

**Environment names that no longer act as their names say:**

- `SVMP_GENERATED_STATE_MAX_DISCONTINUITY_RESTARTS` only throws.
- `SVMP_ACTIVE_PRESSURE_UPDATE_*` is what remains of the retired active-pressure experiments. The driver still reads it every step, and two of the names can still reject a step.
- `SVMP_POSITIVE_AUX_DIRECT_COUPLING` is a no-op.
- `SVMP_FE_DISABLE_MONOLITHIC_COMPILED_DISPATCH` is redundant.

### 2.5 Environment names that change results and have no deck key

68 names on a free-surface run path change computed results when set and have no deck equivalent. 18 of them belong to the coupled 0D-outlet Newton path or the DOF and block layout, which is not free surface. The free-surface ones are:

| Family | Names | Proposal |
|---|---|---|
| Time integration | `GENERALIZED_ALPHA_PDE_UDOT_INIT` (used by the capillary-rise protocol), `TIMELOOP_LAST_STEP_ABSORB_FRACTION` | becomes a key; constant |
| Outer fixed point | `GENERATED_STATE_OUTER_DYNAMIC_RELAXATION{,_INITIAL,_MINIMUM,_MAXIMUM_FACTOR,_DENOMINATOR_RELATIVE_TOLERANCE}`, `GENERATED_STATE_OUTER_FIXED_POINT`, `CUT_REFRESH_PER_STEP_ONLY`, `SYNC_LINE_SEARCH_TRIALS`, `CURVATURE_REUSE_CACHE_ON_TRIAL_FAILURE`, `NO_MPC_STATE_DISTRIBUTE` | constants; delete the legacy names |
| Newton | `NEWTON_LINE_SEARCH{,_ALPHA_MIN,_C1,_SHRINK}`, 8 × `NEWTON_PTC*`, `NEWTON_ACCEPT_INEXACT_LINEAR`, `JACOBIAN_REBUILD_PERIOD` | 3 general Newton keys (on/off and the rebuild period); 11 constants |
| Level-set maintenance | `REDISTANCING_TRIGGER_CELLS`: fixed at 1 cell in code and called "diagnostic only", but it changes when redistancing fires | fixed; sensitivity override through the diagnostics registry |
| Cut numerics | `MIN_GENERATED_CUT_VOLUME_FRACTION` (1e-8, applied in every cut cell) | constant |
| Aggregation | `AGGREGATION_{ALLOW_UNAGGREGATED, LINEAR_EXTENSION, SLAVE_ALL_CUT, MAX_LINES}` | retire (experiments, used only by tests) |
| Static free-surface certificate | `[NS_]FREE_SURFACE_{PRESSURE_REPRESENTABILITY, CONSTANT_PRESSURE_KKT}_MAX_RELATIVE_DISTANCE`, `[NS_]FREE_SURFACE_STATIC_COMPATIBLE_PRESSURE_INITIALIZER` | derive from the static-capillary deck keys |
| Active pressure, assembly ghost policy, constraint history | `ACTIVE_PRESSURE_UPDATE_{REJECT_ON_TRIGGER,THRESHOLD_PA}`, `ASSEMBLY_GHOST_POLICY`, `DISTRIBUTE_CONSTRAINTS_INTO_HISTORY`, `ZERO_CONSTRAINED_RATES`, `FSILS_REJECT_FILL_BEYOND_HALO` | retire or delete |
| Fault injection | `AUDIT_WITHHOLD_GHOST_CUT_LAYERS`, `PDE_EXTENSION_FAIL_COMPONENT` | test-hook registry |
| Round-off only | `DISABLE_CUT_VOLUME_FUSION`, `DISABLE_VERTEX_FIELD_FAST_PATH`, `FE_{ENABLE,DISABLE}_MONOLITHIC_COMPILED_DISPATCH`, `PDE_EXTENSION_FACTORIZATION` (mumps, rank-dependent) | performance registry; the factorization becomes an opt-in key |

**A defect found on the way.** The deck-driven static-capillary certificate needs operators that are installed only when `SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC` is set. No deck key installs them, so without that variable the representability and KKT gates always fail. The level-set workflow tests set the variable by hand.

**Precedence between keys and environment names is inconsistent:**

- The environment wins over `Newton_*`, `Adaptive_time_loop_*` and `Assembly_threads`.
- The deck key wins over `Outer_fixed_point_*` and `Generalized_alpha_predictor`.
- The `Outer_fixed_point_*` keys apply only to transient runs; steady solves take environment-only defaults.

### 2.6 Linear-solver and JIT names

These 199 names are outside the free-surface reduction, but three findings matter for it:

- 98 of them are read on the default path.
- `SVMP_FSILS_BLOCKSCHUR_OUTER_CAP`: the NS `Max_iterations` is silently capped at 50, or 200 with coupled updates.
- `SVMP_FSILS_GMRES_REORTH`: when unset, vector GMRES reorthogonalizes selectively under MPI but not in serial, so serial and MPI runs follow different algorithms by default.

## 3. Per-layer notes

- **(a) BC.**
  - **Physics items:** `Implementation`, `Surface_tension`, `External_pressure`, `Active_domain`, and the contact-line model, angle, wall faces, slip length and mobility.
  - **Wall normal.** `Contact_line_wall_normal` is physics in form, but every deck writes the axis-aligned outward normal of a planar face, so it can be derived.
  - **Kept as advanced:** the 13 cut-quadrature and implicit-cut keys, which serve only the HighOrderImplicit geometry or are the derived quadrature orders, and the 5 aggregation guards, whose FE-contract defaults no current deck changes.
  - **Fixed by D35:** the fitted ALE kinematics (MeshNitsche, γ_N = 10).
- **(b) Level-set equation.**
  - **The D9 advection protocol** uses 9 keys, all with identical values in 12 decks.
  - **Accepted-step maintenance** is three parameter-free booleans.
  - **Legacy algorithms** are used by no accepted benchmark, only by the old SPHERIC and MMS decks and the qualification runners:
    - reinitialization, 8 keys;
    - global volume correction, 9 keys;
    - algebraic wet extension, 5 keys;
    - discontinuity capturing, 5 keys;
    - the bound-preserving limiter, 8 keys.
  - **The KAG curvature route** takes 7 keys besides the form to express one choice.
  - **The least-squares curvature projection** has 11 keys and serves only the CurvatureTraction form, which no accepted benchmark uses.
  - **Static capillary.** One switch, `Enable_static_capillary_equilibrium_initialization`, plus 23 minimizer settings that no deck sets.
  - **Conservative phase.** One switch plus 19 settings; two reference decks use 6 of them, mostly restating defaults.
- **(c) Mesh.**
  - `Ghost_layers` is derived as 12 for an aggregating free surface. The two two-fluid reference decks still set 8, and `static_sphere_3d` writes 12, which equals the derived value.
  - The `Partition_weighting` weights moved from 3/4/1 to 1/2/0 in `490f03aa`. They are measured constants, and no recorded run used them.
- **(d) General.**
  - `Max_cut_topology_restarts_per_step` and the three `Outer_fixed_point_*` keys are the free-surface items.
  - The `Newton_*` and adaptive-time keys are general and have environment twins.
- **(e) Internal.**
  - `FreeSurfaceBoundary` (34 fields), `FreeSurfaceCutCellStabilization` (5) and `FreeSurfaceContactLine` (7 distinct fields over 3 variants) hold 46 fields and 11 enums with 30 values.
  - **Not user-facing:**
    - `role`, the fake one-phase boundary of two-fluid (finding F02);
    - the string-typed `generated_interface_geometry` and `geometry_tangent_policy`, which duplicate the cut-request enums;
    - the single-value `FreeSurfaceActiveDomainMethod`;
    - `input_configuration_schema_version` and `explicit_legacy_configuration`;
    - `ActiveCutVolumeRequest::volume_retention`, which is derived from aggregation;
    - `LevelSetReinitializationOptions::preserve_band_width`, which is deprecated and has no key.
  - **Struct defaults versus resolved defaults.** `kinematic_enforcement` is `None` in the struct but resolves to MeshNitsche for fitted. `active_domain` is `None`, which unfitted rejects. `cut_cell_stabilization.enabled` is false while every deck sets true.
- **(f) Environment, by proposal.** For the 213 free-surface-run names: 88 move to the diagnostics registry, 24 to the performance registry and 2 to the test-hook registry. 30 are deleted because a key exists or they are legacy, 13 are retired experiments, 18 become constants, 10 are derived, 5 become keys, 19 are general and 4 are not environment variables.

## 4. Benchmark protocol matrix (reference decks)

`scripts/work/refdeck_matrix.tsv` lists every free-surface key of the 16 reference decks (9 in `cases`, 7 in `cases-ext`) with its value. `scripts/deck_matrix.py` regenerates it. The protocol constants of the generators are:

| Generator | Transport | Reconciliation | Bounds and redistancing | Semi-implicit | Capillary forms offered |
|---|---|---|---|---|---|
| static_drop_2d, static_sphere_3d | pde_harmonic monolithic | on | – | NormalIncrement | surface_stress, kag_consistent, kag_lumped |
| oscillating_drop_2d, capillary_wave_2d | pde_harmonic monolithic | on | – | NormalIncrement | same three |
| sessile_drop_2d, sessile_drop_3d | pde_harmonic monolithic | on | on | None | same three (2D) |
| capillary rise (open_vessel/capillary_rise_2d.py) | pde_harmonic monolithic | on | on | None | surface_stress; requires `SVMP_GENERALIZED_ALPHA_PDE_UDOT_INIT=0` |
| ren_e_2d | as sessile | on | on | None | law enforcement Strong (default) or Weak |
| linear_sloshing_2d | pde_harmonic monolithic | on | – | – (γ = 0) | – |
| tank_at_rest | coupled field | – | – | – | – |
| fitted_* | fitted ALE, MeshNitsche γ_N = 10, Tangential_mesh_policy Free | – | – | – | SurfaceStress with Allow_fitted_surface_stress |

## 5. Differences between the integration tip and the main line

- **Retired at the tip.** The main line `8784268a` still accepts these keys or values, and the tip has retired them:
  - SmoothedIndicator;
  - GeneratedCurvatureTraction;
  - `Use_current_geometry_curvature=true`;
  - `Enable_velocity_extension=true` and its diffusivity;
  - the symmetric-Nitsche energy diagnostic;
  - the unfitted shape tangents;
  - the inactive-velocity constraint switch;
  - the PSPG and VMS environment overrides and the free-surface probes.

  These amount to 28 environment names.
- **Partition weights.** The defaults of `Partition_weight_{active,inactive,cut}` are 3/4/1 on the main line and 1/2/0 at the tip.
- **Accepted spellings** are the same on both: 526 for the equation list, 293 for the boundary list.

## 6. How to regenerate

From `/scratch/users/zsexton/free-surface-params`:

```
python3 -I scripts/extract_param_lists.py <solver>/Parameters.cpp scripts/work
sort -u scripts/work/eq_ext_names.txt scripts/work/bc_ext_names.txt > scripts/work/accepted_names.txt
python3 -I scripts/extract_alias_groups.py <solver> scripts/work/accepted_names.txt scripts/work/alias_groups.tsv
python3 -I scripts/extract_env.py <solver> scripts/work/env_names.tsv
awk '{print $1"\txml\t"$1}' scripts/work/accepted_names.txt > scripts/work/names_xml_spellings.tsv
python3 -I scripts/usage_counts.py <repo> <reference-deck root> scripts/work/names_xml_spellings.tsv scripts/work/usage_xml_spellings.tsv
python3 -I scripts/build_inventory.py <repo> <reference-deck root>
python3 -I scripts/summary_stats.py inventory.tsv scripts/work/usage_xml_spellings.tsv
```

- **Hand-curated inputs:**
  - `scripts/inventory_spec_xml.py`: type, default, effect, category and proposal of each XML item;
  - `scripts/inventory_spec_internal.py`: the internal fields;
  - `scripts/work/env_class_{driver,newton,other,fsils}.tsv`: one row per environment name, written from its read site;
  - `scripts/env_proposals.py`: the proposal rules and the decisions that override them.
- **Usage counts** are numbers of files that contain a spelling, either as a delimited token (`<Name>`, `"Name"` or `'Name'`) or, for environment names, as a whole word. Generic spellings (`Pressure`, `Value`, `Penalty`, `Model`, `Field_name` and others) are left out of the usage count of their row, because other sections of a deck use the same words.

## 7. Columns of inventory.tsv

| Column | Content |
|---|---|
| `id` | `layer:canonical` |
| `canonical` | the canonical spelling |
| `aliases` | every accepted spelling, canonical first; for layer (e), the key that fills the field |
| `n_spellings` | number of accepted spellings |
| `layer`, `section`, `scope` | where the item belongs |
| `type`, `allowed`, `default` | its values |
| `parsed_at` | `file:function` of every parser that reads it |
| `controls` | what it changes |
| `category` | physics, numerics, solver-control, performance-only, diagnostic-output, legacy-compatibility, experiment, test-only, fault-injection, or naming (field and domain names) |
| `result_changing_default_path` | `yes`: its value, default included, enters every run of its scope; `only-if-set`; `round-off-if-set`; `no`; `no (bitwise by design)`; `dead` |
| `env_read_on_default_path`, `env_changes_results_when_set`, `deck_key_equivalent` | environment names only |
| `uses_generators`, `uses_ref_decks`, `uses_other_decks`, `uses_py`, `uses_cxx` | file counts |
| `generators`, `ref_decks` | the lists behind the first two counts |
| `proposal`, `target`, `note` | the reduction, see `reduction_proposal.md` section 3 |
