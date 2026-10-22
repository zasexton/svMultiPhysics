# Reducing the free-surface configuration to a small typed schema

2026-10-21. This proposal follows the inventory in `inventory.tsv` and `inventory_summary.md`, taken at integration tip `490f03aa`. It builds on decision D42:

- a typed option schema: `OptionSchema<T>` in FE/Core, with the physics schemas in Physics/Core;
- legacy XML spellings kept only as an alias overlay registered by the Application;
- step S13: typed configuration with provenance, plus one legacy environment adapter.

Two documents constrain it. `Documentation/free_surface_refactor_20261019/formulation_design.md` §4 says what the schema must carry and which per-key behaviours must be preserved. `infrastructure_design.md` §4 gives the staged migration and its verification ladder L1–L6.

## 0. Summary

**Before.**

- **XML.** 208 free-surface XML items under 689 accepted spellings, plus 22 general items. A reference deck carries 15 to 43 free-surface keys, with a median of 33. The keys are parsed in up to five places.
- **Environment.** 213 environment names are read on a free-surface run path. 68 of them change results when set and have no deck key. One accepted benchmark (capillary rise) depends on one of them.
- **Internal.** The internal option struct has 46 fields and 11 enums with 30 values.

**After.**

- **XML.** 23 user-facing keys, about 17 advanced keys and 10 two-fluid keys, each with one spelling.
- **Profiles.** One profile key states the method as a named protocol: `Capillary`, `Wetting`, `Gravity_wave` or `Fitted_ALE`.
- **Typical deck.** A deck writes 4 to 10 free-surface keys; the 2D sessile protocol needs 8 instead of 42.
- **Environment.** No environment name changes the results of a production run. Diagnostics, performance switches and test hooks go through one registry with one legacy adapter. That adapter records every use in `effective_configuration.json`.
- **Internal.** The internal struct shrinks to about 30 fields.

**Top changes:**

1. **Method profiles.** The protocol values that every generator writes out today are named, and a deck states physics plus a profile (section 4).
2. **Fixed or derived items.** 71 XML items become method constants and 28 are derived (sections 3 and 5).
3. **Retirements.** 47 XML items retire with algorithms no accepted benchmark uses, and 13 dead or unreachable items are deleted (section 5).
4. **One schema, one parser per key.** The duplicate parse paths go:
   - `append_free_surface_bc` together with `activeCutVolumeRequestFromParameters`;
   - `apply_level_set_params` together with `resolveLegacyLevelSetMaintenanceConfiguration` and the driver's `levelSetAdvectionVelocityRequests`.
5. **Environment.** Five names become keys: the initial generalized-alpha rate, the extension factorization, and three general Newton switches. 18 become constants, 10 are derived, 30 are deleted, 13 retire, and 114 move behind the registry (section 5).
6. **Identity proof.** Every legacy deck must resolve to the same `effective_configuration.json`, byte for byte, before any solver run is compared (section 6.3).

**Decisions needed.** Section 8 lists 13 open questions, each with options and a recommendation. The four that shape the design are:

- Q1: profiles only, or also changing the code defaults;
- Q2: whether the CurvatureTraction route stays;
- Q5: whether the legacy level-set algorithms retire;
- Q12: the environment policy.

## 1. What the inventory shows

1. **Little of the surface is used.** 396 of the 721 accepted XML spellings appear nowhere in the repository's decks, generators, runners or tests. The 95 spellings the benchmarks use are all the canonical `Snake_case` spellings. 98 of the 127 secondary spellings in use appear only in C++ tests that check alias acceptance.
2. **The benchmarks restate the method.** The 13 unfitted reference decks set the same values for about 30 numerical items:
   - the ghost-penalty coefficient 1.0;
   - SUPG 0.5 and 2.0;
   - LinearCorner, RefreshedFrozenQuadrature and interface order 2;
   - aggregation on;
   - the nine D9 advection keys;
   - reinitialization off and volume correction off.

   The generators carry protocol constants on top of these: reconciliation on for every moving case, bounds and redistancing on for contact lines, and NormalIncrement on for drops and waves. None of these is a per-case choice. Under rule P1 ("no numerical parameter chosen per case or per mesh") they are the method.
3. **Code defaults are not protocol defaults.** The struct defaults are:
   - `cut_cell_stabilization.enabled=false`;
   - `supg.enabled=false`;
   - maintenance off;
   - `active_domain=None`, which unfitted rejects;
   - `kinematic_enforcement=None`, which resolves to MeshNitsche for fitted.

   So every deck has to write the method, and every generator carries it twice: as constants and as XML.
4. **Several parsers per key.** 76 items are parsed by two or more parsers and 12 by three or more. Their defaults are kept "synchronized" by comments, for example the `Small_cut_aggregation` default true in three files.
5. **Dead and unreachable items.**
   - **The schema-1 legacy mode cannot be selected.** Its keys `Free_surface_configuration_schema_version` and `Enable_explicit_legacy_free_surface_configuration` are not accepted by `Parameters.cpp`, so about 20 "legacy-only" branches of `append_free_surface_bc` can be reached only from one C++ test.
   - **Seven accepted keys can only throw or restate a retired value.** They are listed in `inventory_summary.md` §2.4.
6. **The environment is a second, untyped configuration.**
   - **Result-changing names.** 68 names on the free-surface path change results and have no key. The capillary-rise protocol needs `SVMP_GENERALIZED_ALPHA_PDE_UDOT_INIT=0`.
   - **Precedence.** It is inconsistent: the environment wins for the `Newton_*`, adaptive-time and assembly-thread keys, and the key wins for the outer fixed point and the predictor.
   - **Missing operators.** The deck-driven static-capillary certificate silently needs `SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC` to install its operators.

## 2. Design rules

- **R1. Physics in the deck, method in the profile.** A deck states material, geometry and boundary physics, plus one named method profile. Numerical constants that every accepted benchmark shares, and that P1 forbids tuning, are not deck keys.
- **R2. Fixed or derived before configurable.** An item is user-facing only if a physical case needs a different value. Otherwise it is either:
  - **fixed**: a method constant, set by the profile; or
  - **derived**: computed from other inputs, such as field names, markers, the wall normal of a planar face, the KAG curvature chain, or retention from aggregation.
- **R3. One schema, one parser.** Each option struct has one `OptionSchema` (`formulation_design.md` §4). The register, the Application translators, the artifact writer and the documentation all read it.
- **R4. Exact results.** Moving from the old input to the new one is proven identical on the resolved configuration before any run (section 6.3). Every result-changing simplification is a separate step with its own baseline and a user decision.
- **R5. Nothing in the environment changes results silently.** Every result-changing control is a typed key, a constant or a test hook. Observation and bitwise performance switches go through one registry. The resolved configuration records any registry entry that can change round-off.
- **R6. One spelling.** The new schema has one spelling per key and case-insensitive enum values with one canonical token each. Legacy spellings live only in the Application alias overlay, for a bounded time.

## 3. Target schema

In the tables:

- "Profile" means the profile sets the value, and the key is an override.
- "Derived" means the value is computed and the key goes, except where an override is noted.

The names keep today's canonical spelling wherever one exists, so most kept keys need no deck change.

### 3.1 Physical model (`Free_surface` BC)

| Key | Type | Default | Why the user needs it |
|---|---|---|---|
| `Implementation` | `UnfittedLevelSet` \| `FittedALE` | required for a free surface (today it defaults to FittedALE) | the representation of the interface is a modelling choice |
| `Surface_tension` | real ≥ 0 | 0 | physics |
| `External_pressure` | real | 0 | physics (gas pressure) |
| `Active_domain` | `LevelSetNegative` \| `LevelSetPositive` | `LevelSetNegative` (today `None`, which unfitted rejects) | which side of φ is liquid |
| `Contact_line_model` | `None` \| `Pinned` \| `PrescribedAngle` \| `DynamicRenE` | `None` | wetting physics |
| `Contact_line_wall_face` | face names | – | the walls the contact line lives on |
| `Contact_angle_degrees` | real in (0, 180) | required with an angle model | θ_e |
| `Wall_slip_length` | real > 0 | none (no slip) | Navier slip; `Wall_slip_model` is derived from its presence (required for DynamicRenE) |
| `Contact_line_mobility` | real > 0 | required for DynamicRenE | Ren–E law |

### 3.2 Method (`Free_surface` BC)

| Key | Type | Default | Why |
|---|---|---|---|
| `Free_surface_profile` | `Capillary` \| `Wetting` \| `Gravity_wave` \| `Fitted_ALE` \| `None` | `None` (today's code defaults; see Q1) | names the protocol |
| `Surface_tension_form` | `SurfaceStress` \| `KinematicAreaGradientTraction` (\| `CurvatureTraction`, Q2) | `SurfaceStress` | the two capillary forms the benchmarks compare |
| `Kinematic_area_gradient_mass` | `Consistent` \| `Lumped` | `Consistent` | the one free choice of the KAG form; moved from the level-set equation |
| `Surface_tension_semi_implicit` | `None` \| `NormalIncrement` | profile | solver control; it moves results only within the outer tolerance |
| `Contact_line_law_enforcement` | `Weak` \| `Strong` | profile (Q6) | how the Ren–E law is imposed |

### 3.3 Level-set transport and maintenance (`level_set` equation)

| Key | Type | Default | Why |
|---|---|---|---|
| `Level_set_field_name` | name | `level_set` | naming; the BC and the cut request take it from here |
| `Level_set_initial_condition` | `Mesh_field` \| `Discrete_static_equilibrium` | `Mesh_field` if the mesh has a vertex field of that name | replaces `Level_set_source` and `Enable_static_capillary_equilibrium_initialization` |
| `Level_set_maintenance` | `None` \| `Reconcile` \| `Reconcile_and_redistance` | profile | replaces the three parameter-free accepted-step booleans; the redistancing trigger stays fixed at one cell (Q10) |
| `Advection_velocity_extension_method` | `pde_harmonic` \| `pde_normal` \| `fluid_velocity` | profile (`pde_harmonic`) | replaces the nine D9 keys; the coupling is fixed monolithic, and the field names and registration are derived |

### 3.4 Time and outer coupling (`GeneralSimulationParameters`)

| Key | Type | Default | Why |
|---|---|---|---|
| `Outer_fixed_point_max_passes` | int ≥ 1 | 30 | refresh budget per step. Applies to steady solves too, which now use environment-only defaults |
| `Generalized_alpha_initial_rate` | `Consistent` \| `Zero` | `Consistent` | replaces `SVMP_GENERALIZED_ALPHA_PDE_UDOT_INIT`; the capillary-rise protocol sets `Zero` |

### 3.5 Partition and diagnostics

| Key | Where | Type | Default | Why |
|---|---|---|---|---|
| `Partition_weighting` | `Add_mesh` | `none` \| `free_surface` | `none` | opt-in load balance; changes round-off as a rank count does (D45). The three weights become constants (1, 0, 2) |
| `Ghost_layers` | `Add_mesh` | int | derived (12 for an aggregating free surface) | override only |
| `Diagnostics` | `GeneralSimulationParameters` | list of registry names | empty | replaces about 90 observation environment names and the XML diagnostic keys: `Write_velocity_extension_maps`, `Conservative_phase_write_flux_artifacts` with its cadence, and the KAG finite-difference check |

That makes 23 user-facing keys. A unfitted deck writes the keys of 3.1 that its physics needs, plus `Free_surface_profile`, and adds overrides only when it deviates from the protocol.

### 3.6 Advanced keys

These are documented separately as numerics for verification and tests. They are not in the user guide.

| Key | Default | Note |
|---|---|---|
| `Cut_quadrature_order` | derived (P1: 2 for interface and volume; see Q7) | replaces `Interface_`, `Volume_` and `Cut_quadrature_order` with 18 spellings |
| `Small_cut_aggregation` | true | only tests switch it off |
| `Small_cut_aggregation_maximum_{root_path_length, reference_extrapolation_distance, absolute_coefficient, row_l1_norm}` | 8, 4, 16, 32 | FE contract defaults; no current deck changes them |
| `Small_cut_aggregation_rootless_fallback` | false | D33 fail-safe, used by C++ tests |
| `Contact_line_wall_normal` | derived from the planar wall face | override for curved walls |
| `Max_cut_topology_restarts_per_step` | the outer limit | 0 restores stop-and-reject |
| `Transport_form` | advective | |
| `Enable_bound_preserving_limiter` | false | only if Q5(b) |
| `Constant_velocity` | – | prescribed-velocity verification (MMS) |
| `Extension_factorization` | default (rank-independent) | replaces `SVMP_PDE_EXTENSION_FACTORIZATION`; `mumps` is opt-in and rank-dependent at round-off |
| `Tangential_mesh_policy`, `Prescribed_tangential_mesh_velocity` | profile `Fitted_ALE` | fitted only |
| `Prescribed_pressure_jump`, `Prescribed_viscous_traction_jump` (one vector key) | 0 | two-fluid verification |
| `Newton_line_search`, `Newton_pseudo_transient_continuation`, `Newton_jacobian_rebuild_period` | today's environment defaults | general Newton block; replaces 14 environment names, 11 of which become constants |

### 3.7 Two-fluid, conservative phase and fitted ALE

- **Two-fluid.** These keys stay as they are, because they are physics:
  - `Free_surface_physical_model`;
  - `Negative_phase_density`, `Negative_phase_dynamic_viscosity`, `Positive_phase_density`, `Positive_phase_dynamic_viscosity`;
  - `Two_fluid_surface_tension`;
  - `Material_interface_marker`.

  `Two_fluid_interface_nitsche_gamma` (24 in both decks) becomes a method constant (Q4). The interface-kinematic keys are derived from `Material_interface_marker`.
- **Conservative phase.** One switch, `Enable_conservative_phase_transport`. The field name and the liquid side are derived. The boundary-flux policy is derived from the open level-set boundaries. The tolerances are constants, and the flux artifacts are diagnostics.
- **Fitted ALE.** `Implementation=FittedALE` with `Free_surface_profile=Fitted_ALE` derives everything else:
  - on the fluid equation: `Enable_ALE`, the mesh velocity source and fields, and `Moving_mesh_tangent_path=SymbolicRequired`;
  - on the BC: `Kinematic_enforcement=MeshNitsche` with γ_N = 10, `Normal_kinematic_policy=MatchFluidNormalVelocity`, and `Tangential_mesh_policy=Free`.

  The `mesh_motion` equation keeps its own general schema.

### 3.8 Example: the 2D sessile protocol deck

The deck has 42 free-surface keys today (`cases-ext/sessile_protocol/solver.xml`). In the new form:

```xml
<Add_equation type="level_set">
  <!-- standard keys: Coupled, Min/Max_iterations, Tolerance, Module_options, LS -->
  <Level_set_field_name>phi</Level_set_field_name>
</Add_equation>
...
<Add_BC name="free_surface">
  <Type>Free_surface</Type>
  <Implementation>UnfittedLevelSet</Implementation>
  <Free_surface_profile>Wetting</Free_surface_profile>
  <Surface_tension>1</Surface_tension>
  <Contact_line_model>PrescribedAngle</Contact_line_model>
  <Contact_line_wall_face>wall_bottom</Contact_line_wall_face>
  <Contact_angle_degrees>60</Contact_angle_degrees>
  <Wall_slip_length>0.125</Wall_slip_length>
</Add_BC>
```

The other three cases:

- **Capillary-rise protocol:** the same plus `Generalized_alpha_initial_rate=Zero` in `GeneralSimulationParameters`.
- **KAG static-drop reference deck:** `Free_surface_profile=Capillary`, `Surface_tension_form=KinematicAreaGradientTraction`, `Kinematic_area_gradient_mass=Lumped`, and the two overrides that make it a nine-case regression deck: `Level_set_maintenance=None` and `Surface_tension_semi_implicit=None`.

## 4. Profiles

A profile is a named, versioned value set of the method items. It lives in Physics/Core, next to the free-surface schema. Values resolve in this order:

1. an explicit override key;
2. the profile;
3. a value derived from other inputs;
4. the code default.

`effective_configuration.json` records the provenance of every field, for example `profile:Wetting@1`, `explicit`, `derived:planar_face_normal` or `legacy_alias:Enable_SUPG`.

| Item | `Capillary` | `Wetting` | `Gravity_wave` | `Fitted_ALE` | `None` (today) |
|---|---|---|---|---|---|
| Benchmarks | static drop 2D, static sphere 3D, oscillating drop, capillary wave | sessile 2D/3D, capillary rise, Ren–E | linear sloshing, tank at rest (with overrides) | fitted capillary wave, fitted static drop, fitted sloshing | C++ tests, legacy decks |
| Advection velocity | pde_harmonic, monolithic | same | same | – | coupled field |
| SUPG (τ 0.5, transient 2.0) | on | on | on | – | off |
| Ghost penalty (γ = 1.0), aggregation and its guards | on | on | on | – | penalty off, aggregation on |
| Geometry: LinearCorner, RefreshedFrozenQuadrature, cut order 2 | yes | yes | yes | – | LinearCorner, RefreshedFrozen; order 2 in 2D and 1 in 3D |
| `Level_set_maintenance` | Reconcile | Reconcile_and_redistance | Reconcile | – | None |
| `Surface_tension_semi_implicit` | NormalIncrement | None | None (γ = 0) | None | None |
| Contact law | – | Strong for DynamicRenE (Q6) | – | – | Weak |
| Fitted kinematics | – | – | – | MeshNitsche, γ_N = 10, Tangential Free | MeshNitsche |
| Capillary form | SurfaceStress | SurfaceStress | – | SurfaceStress (Q2) | Automatic |

**The KAG chain is derived, not profiled.** `Surface_tension_form=KinematicAreaGradientTraction` implies the following, whatever the profile:

- `Enable_curvature_projection`;
- the curvature field name on both the BC and the equation;
- recovery mode `KinematicAreaGradient`;
- filter coefficient 0.0;
- cadence 1.

All three KAG reference decks set exactly these values.

**Values come from the protocol constants.** The profile values are taken from the generators' constants (`KINEMATIC_RECONCILIATION`, `SIGN_DEFINITE_PATCH_BOUNDS`, `INTERFACE_PRESERVING_REDISTANCING`, `DEFAULT_SEMI_IMPLICIT`, `DEFAULT_TRANSPORT`, `DEFAULT_LAW_ENFORCEMENT`), not from the nine-case reference decks. The reference decks are regression decks: they become `Capillary` (or `Gravity_wave` for the tank and sloshing decks) with explicit overrides that reproduce them exactly.

**Versioning.** A profile is versioned (`Wetting@1`). Changing a profile value creates a new version. Decks that want the old behaviour can name the old version, so accepted results stay reproducible after a protocol revision.

## 5. What goes

The `proposal` column of `inventory.tsv` gives the decision for every item. In summary:

| Action | XML (a–d) | Internal (e) | Environment, free-surface path | Examples |
|---|---:|---:|---:|---|
| FIX (method constant) | 71 | 12 | 18 | SUPG constants; ghost-penalty γ; isovalue 0; CutVolume; RefreshedFrozen; the 23 static-capillary minimizer settings; the KAG filter and cadence; conservative-phase tolerances; the fitted γ_N and normal policy; partition weights; outer relaxation pass, exit pass and Aitken factors; `MIN_GENERATED_CUT_VOLUME_FRACTION`; PTC and line-search constants |
| DERIVE | 28 | 10 | 10 | field names, domain id, markers; `Velocity_source`; `Auto_register_*`; `Generated_interface_geometry` from the φ order; wall normal; `Wall_slip_model`; the curvature-projection chain; interface-kinematic keys; conservative-phase side and flux policy; ALE fields; the static-certificate gates and their operators |
| ALIAS | 2 | – | – | `Contact_angle_radians`; `Cut_domain_type` |
| RETIRE | 47 | 10 | 13 | reinitialization (8), volume correction (9), wet extension (4), discontinuity capturing (4), least-squares curvature recovery (11), CurvatureTraction inputs (`Curvature`), legacy fitted kinematics (3), cut metadata scale (2), pressure-stabilization policy, backend-qualification gate, corner-linearized geometry, `Allow_full_domain_*`, `Allow_fitted_surface_stress`; environment: active-pressure remnants, aggregation experiments, retention force, per-step-only refresh, assembly ghost policy |
| DELETE (dead, unreachable, or duplicated by a key) | 13 | 7 | 30 (+4 not environment variables) | schema-1 legacy mode and schema version; `Use_level_set_curvature`; retired-only-throws keys; `Discontinuity_capturing_residual_epsilon`; basis-type spellings; environment twins of the `Newton_*`, `Adaptive_time_loop_*`, `Outer_fixed_point_*`, predictor and `Assembly_threads` keys |
| DIAG (diagnostics registry) | 4 | – | 88 | traces, dumps, self-checks, `REDISTANCING_TRIGGER_CELLS` (sensitivity only) |
| PERF (performance registry) | – | – | 24 | cache disables, fast-path disables, extension cache entries, replicated solves |
| TEST (test-hook registry) | – | – | 2 | `AUDIT_WITHHOLD_GHOST_CUT_LAYERS`, `PDE_EXTENSION_FAIL_COMPONENT` |
| KEY (environment name becomes a key) | – | – | 5 | initial rate, extension factorization, three Newton switches |
| KEEP, PROFILE, ADVANCED | 52 | 17 | – | sections 3.1–3.7 |
| GENERAL (outside this reduction) | 33 | – | 19 | mesh-motion equation and BC keys, momentum source, Coriolis, `Newton_*` keys; coupled 0D-outlet and DOF-layout environment names |

**Spellings.**

- **Delete now:** the 396 spellings used nowhere and every CamelCase spelling.
- **Keep for one release, with a warning:** about 25 legacy spellings that a deck, generator or runner still uses (section 6.1).
- **Delete the rest** with the alias-acceptance tests that exercise them.

**Internal structs.**

- **Removed:** `FreeSurfaceBoundaryRole`, which becomes a `VolumeDomain` argument (F02); the single-value `FreeSurfaceActiveDomainMethod`; `FreeSurfacePressureStabilizationPolicy`; the legacy kinematic fields; `allow_*`; `use_level_set_curvature`; and the cut-metadata scale.
- **Enums instead of strings:** `generated_interface_geometry` and `geometry_tangent_policy` become the FE enums.
- **Struct defaults** follow the `None` profile, so the struct itself documents the legacy behaviour.

## 6. Migration

### 6.1 Alias overlay

The overlay is registered by the Application, in `Application/Translators/FreeSurfaceLegacyAliases.cpp`, and consulted only when a key is not in the schema.

- **Renamed or folded keys** are accepted with a one-line deprecation message that names the replacement. For example:
  - "`Enable_kinematic_reconciliation` is deprecated; use `Level_set_maintenance=Reconcile` or a profile";
  - "`Curvature_projection_*` is derived from `Surface_tension_form`".

  The overlay translates them into schema fields, so the resolved configuration is identical.
- **Fixed items** are accepted only with their fixed value, with a warning, for example `Enable_cut_cell_stabilization=true` under a profile that sets it, or `SUPG_tau_scale=0.5`. Any other value is an error that names the profile.
- **Retired keys and values** are errors with the retirement record, for example "`Enable_volume_correction=true` was retired (decision D4x); use `Level_set_maintenance`". A value that restates the off state is accepted with a warning for one release.
- **Legacy spellings that survive for one release.** These are the spellings still used by the repository's generators, runners or old decks that are not yet migrated:
  - the D9 advection set (`Velocity_source`, `Velocity_field_name`, `Auto_register_velocity_field`, `Source_velocity_field_name`, `Advection_velocity_extension_coupling`, `Operator_tag`, `Level_set_source`);
  - the three maintenance booleans;
  - the KAG chain (6 keys);
  - `Enable_SUPG`, `SUPG_tau_scale`, `SUPG_transient_scale`, `Enable_cut_cell_stabilization`, `Cut_cell_pressure_gradient_penalty`, `Use_cut_metadata_scale`, `Small_cut_aggregation`;
  - `Active_domain_method`, `Active_domain_smoothing_width`, `Generated_interface_geometry`, `Geometry_tangent_policy`, `Interface_quadrature_order`, `Level_set_isovalue`, `Use_level_set_curvature`, `Enable_velocity_extension`, `Enable_reinitialization`, `Enable_volume_correction`, `Wall_slip_model`, `Contact_line_wall_normal`, `Generated_interface_domain_id`;
  - in old decks: `Advection_velocity_from_field`, `Wet_extension_advection_velocity_method`, `Generated_interface_quadrature_order`.

  The overlay ends when the in-repository decks and generators are migrated (6.2), and at the latest one release after the schema lands.
- **Values.** The current synonym tokens collapse to one canonical token per value, compared case- and underscore-insensitively. Examples of synonyms: `fitted`, `ale`, `levelset` and `embeddedlevelset` for `Implementation`; eleven spellings of SurfaceStress or KAG.
- **Environment.** One legacy environment adapter (S13) maps the old diagnostic, performance and test-hook names to registry entries for one release. It writes each use to the run log and to `effective_configuration.json`, and a result-changing entry stamps the run `non_reference`. The names that become keys or constants are rejected with the same kind of message `SVMP_GENERATED_STATE_MAX_DISCONTINUITY_RESTARTS` already gives.

### 6.2 Decks and generators

- **Generators.** Each generator emits `Free_surface_profile` and the physics keys. The method constants at the top of each file (`KINEMATIC_RECONCILIATION` and the others) become the profile definition and are removed from the generators.
  - Options that select a non-protocol variant stay, and emit an override: `--capillary-form`, `--surface-tension-semi-implicit`, `--kinematic-reconciliation off`, `--law-enforcement`.
  - Options that select a retired algorithm go, subject to Q5 and Q9: `--transport wet_extension` and `coupled`, `--reinitialization`, `--kinematic-enforcement Penalty`.
  - For one cycle each generator keeps an `--explicit` switch that writes today's long form, so the two forms can be compared (6.3).
- **Capillary rise.** `capillary_rise_2d.py` replaces `REQUIRED_ENVIRONMENT` with `Generalized_alpha_initial_rate=Zero`. The wp5 protocol JSON records the key.
- **Reference decks.** The nine-case and extended reference decks are regenerated in the new form, with the overrides of section 4. The old decks stay in `cases-legacy` as the overlay test set.
- **Other decks under `tests/`** (the SPHERIC, MMS and open-vessel validation decks) are converted by a script that applies the overlay mapping. Decks that use a retired algorithm keep the old form until Q5 is decided, and are marked in the tracker.
- **C++ tests.** Tests that build `ParameterMap`s with legacy spellings are switched to the canonical schema keys. Alias-acceptance tests become overlay tests: one table-driven test per overlay entry, plus one test per retired key that checks the message.

### 6.3 Proving result identity

The steps are ordered from cheapest to most expensive. A later step runs only if the earlier ones pass.

1. **Resolved-configuration identity, with no solve.**
   - Extend `effective_configuration.json` to a complete resolved manifest with provenance:
     - every field of `FreeSurfaceBoundary`, `LevelSetTransportOptions`, `ActiveCutVolumeRequest`, the maintenance and extension requests, and the outer and Newton options;
     - every environment value that was read;
     - the canonical order of the FE option structs.

     About 208 leaves are written today for the sessile deck.
   - Add a resolve-only mode that stops after setup.
   - For every deck in the corpus, the old and new inputs must give byte-identical manifests once the provenance fields are removed. The corpus is: the 16 reference decks; every generator at every CLI variant, in both the `--explicit` and the profile form; the 17 free-surface decks under `tests/`; and the C++ test `ParameterMap`s, through a unit-test hook. This step proves identity by construction; nothing has to be run.
2. **Kernel and constraint manifests (S0).** Ordered JIT kernel identities and the constraint plan must be identical for the same corpus. This catches a schema change that reorders term installation, for example a derived `Generated_interface_domain_id` that changes the generated marker keys.
3. **Bitwise runs.** The nine-case serial set (L2), refset-ext (L3), CTest (L4), the two-rank MPI matrix (L5), and 2- and 4-rank parity (L6), all against `e7080344` / `490f03aa`, using the tools of `perf-reference-arm64`.

Any item marked "result-changing" in Q1, Q2, Q4, Q5 and Q7 is not part of this migration. Each is a later, separate commit with its own baseline, judged against the round-off spread between rank counts (as for the partition weighting in D45).

### 6.4 Order relative to the refactor steps

| Step | Content | Bitwise | Collides with |
|---|---|---|---|
| P0 (now) | Delete the dead and unreachable XML items and spellings (section 5, DELETE, plus the never-used spellings). Delete the dead environment names (the `MAX_DISCONTINUITY_RESTARTS` guard after one release, `POSITIVE_AUX_DIRECT_COUPLING`, the not-environment entries). Retire the active-pressure remnants. Fix the static-certificate dependence on the diagnostic environment name (derive the operators). | yes (the certificate fix changes only runs that use the static initializer; no reference deck does) | small edits to `Parameters.cpp`, `NavierStokesRegister.cpp` and `LevelSetConfigurationParsing.h`; R1 touches the VMS module, not the register |
| P1 = S13a | `OptionSchema<T>` in FE/Core. The resolved manifest with provenance and the resolve-only mode. Record baseline manifests for the corpus. | yes (adds output only) | none |
| P2 = S13b | Free-surface BC schema (`FreeSurfaceBoundary` and contact lines). One parser replaces `append_free_surface_bc`, `append_free_surface_contact_line` and the BC branch of `activeCutVolumeRequestFromParameters`. Alias overlay. | yes (manifest identity) | the register part of R1, if the register split comes first; coordinate with the R1 owner |
| P3 = S13c | Level-set equation schema. One parser replaces `apply_level_set_params`, the re-parse in `resolveLegacyLevelSetMaintenanceConfiguration`, and the parse in the driver's `levelSetAdvectionVelocityRequests`. | yes | the outer-pass branch (driver 5050–5374 belongs to S10's extension code): start after it merges, or replace only the parse and leave the function to S10 |
| P4 | Profiles. Generator migration and regenerated reference decks. Converted `tests/` decks. | yes (manifest identity, then L2–L6) | none |
| P5 | Environment: the five new keys, constants, derivations, deletions. The diagnostics, performance and test-hook registry (D42's FE DiagnosticRegistry). The legacy environment adapter. | yes (defaults unchanged; registry entries default off) | NewtonSolver (S3) and the driver (S6–S9): do each family when its code moves, or before the move to shrink it |
| P6a | Retire the algorithms decided in Q2, Q5, Q8 and Q9, with their tests. | yes on all reference sets (no accepted benchmark uses them) | none; reduces what S9 and S10 must move |
| P6b | Default changes decided in Q1(b), Q4(b) and Q7(a), one per commit, each with a new arm64 baseline | **no**: each is result-changing and judged against the rank spread | after the refsets are regenerated in P4 |

S13 is "ongoing" in the infrastructure plan. P1–P3 are its core and can start now: P1 has no collisions; P2 needs only coordination on the register. P5 is the environment half of S13 and should follow the code moves (S3, S6–S10) family by family.

## 7. Numbers before and after

| Layer | Before | After |
|---|---|---|
| (a) `Free_surface` BC | 74 items, 293 spellings (55 items and 223 spellings in `append_free_surface_bc`) | 14 user-facing, 11 advanced, 1 level-set BC value; 26 keys with one spelling each (plus 5 general mesh-motion BC keys) |
| (b) level-set equation and fluid-equation extensions | 156 items, 458 spellings (139 free-surface specific) | 4 user-facing, 3 advanced, 10 two-fluid; 17 keys (plus 17 general) |
| (c) mesh and partition | 5 | 2 |
| (d) general simulation, free-surface specific | 4 (plus 11 general) | 3, including the new `Generalized_alpha_initial_rate` and `Diagnostics` (plus 14 general: 3 Newton keys from the environment) |
| Free-surface XML in total | 208 items under 689 spellings | 23 user-facing, about 17 advanced, 10 two-fluid: about 50 keys, one spelling each |
| Keys per reference deck | 15–43 (median 33) | 4–10 |
| (e) internal fields | 46 fields, 11 enums, 30 values (plus 10 related internals) | about 30 fields, 8 enums, about 20 values |
| (f) environment names on the free-surface path | 213 (68 result-changing without a key; 137 read on the default path) | 0 result-changing names. 114 registry entries (88 diagnostic, 24 performance, 2 test) behind one adapter. 19 general names left for the solver cleanup |
| (f) linear-solver and JIT names | 199 (83 result-changing without a key) | not in scope (recommend a separate pass under the same rules) |

## 8. Open questions

**Q1. Profiles and code defaults.**

- (a) Add profiles; keep today's code defaults as the `None` profile, which is the default; generators and decks name a profile.
- (b) As (a), then make the default profile follow from the physics: γ > 0 with contact lines gives `Wetting`, γ > 0 without them `Capillary`, γ = 0 `Gravity_wave`, and fitted ALE `Fitted_ALE`. The deck then needs no profile key.
- (c) No profiles: keep the method items as trimmed explicit keys.

**Recommendation: (a) now; (b) as a separate, result-changing step after P4.** Under (b) the C++ tests and any deck without a profile change results, so it needs a new baseline and a check against the rank spread.

**Q2. The CurvatureTraction route.** No accepted benchmark uses it: not unfitted, and not fitted with γ > 0.

- (a) Retire the unfitted CurvatureTraction, the least-squares curvature recovery (11 keys) and `Curvature`. Keep fitted `Automatic` → CurvatureTraction.
- (b) Retire it everywhere. `Automatic` becomes SurfaceStress on fitted too, and `Allow_fitted_surface_stress` goes (both fitted capillary benchmarks set it to true).
- (c) Keep it as an advanced verification route.

**Recommendation: (b), provided the kernel manifest shows that fitted γ = 0 decks (sloshing) are unchanged. Otherwise (a).**

**Q3. Make `Implementation` required and default `Active_domain` to `LevelSetNegative`?**

Today a free-surface BC without `Implementation` is FittedALE, and `Active_domain=None` makes unfitted fail.

- (a) Yes.
- (b) Keep the current defaults.

**Recommendation: (a).** No working deck changes: every unfitted deck writes both keys, and an omitted `Active_domain` fails today.

**Q4. The method constants whose code default differs from the protocol.** These are the ghost penalty (on, γ = 1), SUPG (on, 0.5 and 2.0), and the two-fluid interface Nitsche γ = 24.

- (a) Fixed by the profile: no deck key; the `None` profile keeps the code default.
- (b) Also change the code defaults to the protocol values.
- (c) Keep them as advanced keys.

**Recommendation: (a).** It is bitwise, and P1 already forbids tuning them.

**Q5. Legacy level-set algorithms.** These are PDE/projection reinitialization (8 keys), global volume correction (9), algebraic wet extension (5), discontinuity capturing (4) and the bound-preserving limiter (8). No accepted benchmark uses them; the old SPHERIC and MMS decks and the qualification runners do. The planned SPHERIC re-runs (tracker T1 items) would use the current protocol: PDE extension and reconciliation.

- (a) Retire all five.
- (b) Retire the first four; keep the bound-preserving limiter as one advanced switch with fixed tolerances until SPHERIC 02/05/10 are re-run.
- (c) Move all five under Advanced.

**Recommendation: (b).**

**Q6. Ren–E law enforcement.**

- (a) Make Strong the code default for DynamicRenE.
- (b) Keep Weak as the code default; the `Wetting` profile sets Strong.
- (c) Leave it to the deck.

**Recommendation: (b).** It is bitwise and matches the ren_e protocol default.

**Q7. Cut quadrature order.** Today the unset interface order is 2 in 2D and 1 in 3D.

- (a) One derived default of 2 for P1 in every dimension. This changes 3D decks that leave it unset, which is `tank3d_L8` among the reference decks.
- (b) Keep the asymmetry in the code; the profiles set 2.

**Recommendation: (b) now; (a) together with Q1(b).**

**Q8. HighOrderImplicit geometry and the implicit cut backends** (about 10 keys: backend, fallback, root tolerances, subdivision depth, qualification gate, corner-linearized geometry).

- (a) Keep them as an experimental advanced block in the free-surface schema.
- (b) Take them out of the free-surface schema, keeping the capability in FE (the equation-level cut domain uses it), until a free-surface benchmark qualifies it.

**Recommendation: (b).**

**Q9. Legacy fitted kinematics** (the Penalty and Nitsche enforcements and their 3 keys, legacy since D35).

- (a) Retire them, together with `fitted_sloshing --kinematic-enforcement`.
- (b) Keep them under Advanced.

**Recommendation: (a).**

**Q10. The redistancing trigger.**

- (a) Fixed at one cell. `SVMP_REDISTANCING_TRIGGER_CELLS` becomes a registry sensitivity entry that stamps the run `non_reference`.
- (b) A deck parameter.
- (c) Delete the override.

**Recommendation: (a).** The one-cell rule makes the number of calls independent of the time step, and P1 forbids tuning it.

**Q11. The outer fixed point and time keys.**

- (a) Keep only `Outer_fixed_point_max_passes`, applied to steady solves too. The relaxation start pass (12), the exit pass (13) and the Aitken factors become constants. `Max_cut_topology_restarts_per_step` becomes advanced. Add `Generalized_alpha_initial_rate` with default `Consistent`.
- (b) Keep all three outer keys.
- (c) Make the initial rate part of the `Wetting` profile. This changes the sessile results.

**Recommendation: (a).**

**Q12. Environment policy.**

- (a) No environment name may change a production result. Result-changing names become keys, constants or test hooks. Observation and bitwise performance switches go through one registry (`Diagnostics` in the deck, or a single `SVMP_REGISTRY` variable), with one legacy adapter for one release. Round-off-changing entries stamp the run.
- (b) As (a), but keep the performance switches as individual environment names.
- (c) Only convert the five keys; leave the rest.

**Recommendation: (a).** Apply the same rules to the 199 linear-solver and JIT names in a separate pass. That pass must also consider the hidden 50-iteration cap and the difference between serial and MPI reorthogonalization.

**Q13. Legacy spellings.**

- (a) Delete the 396 never-used spellings and all CamelCase spellings in P0. Keep about 25 used legacy spellings in the overlay with a warning for one release, then make them errors.
- (b) Keep every current spelling in the overlay, with warnings, for one release.
- (c) Keep the overlay indefinitely.

**Recommendation: (a).** These keys are new in this program and not in upstream releases, so no external deck depends on the unused spellings.

## 9. Risks

- **Derived names.** A derived `Generated_interface_domain_id` or `Interface_marker` must reproduce the generated marker keys exactly. Step 2 of 6.3 (the kernel and constraint manifests) is the check. If it fails, derive the old explicit id in the overlay for legacy decks only.
- **Derived wall normal.** A wall normal derived from a planar face must equal the deck literal bit for bit. Axis-aligned faces give exact unit vectors. A slanted face needs either an explicit override or a manifest check.
- **The KAG filter.** All KAG reference decks write a filter of 0.0, so fixing it at 0.0 reproduces them. A deck that uses consistent KAG and omits the filter gets 1.0 today and would change; the overlay should warn about it. The corpus has none.
- **Parsing quirks.** These must keep their old behaviour in the overlay, as per-key policies (formulation design §4):
  - `parse_bool_relaxed` maps unknown text to false;
  - the first alias defined wins;
  - "implies" defaults, such as a sub-option enabling cut stabilization.
- **Concurrent branches.** Coordination with the R1 and outer-pass branches is needed for P2 and P3. The steps are small and parse-only, so a short freeze on `NavierStokesRegister.cpp` and the two translators is enough.
