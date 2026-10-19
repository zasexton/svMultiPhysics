# A-formulation-design.md — Navier–Stokes VMS and free-surface formulation (physics side)

Branch `issue-449-modern-mesh-core` @ `e7080344`. Paths are relative to `Code/Source/solver/` unless they start with `Documentation/` or `tests/`. This was a read-only study: nothing was built or run. B-*.md files were absent when I wrote this, so every FE name here is a proposal for B to confirm or rename.

Class key used throughout:
- **a**: governing equations (weak form, VMS/PSPG, standard BCs).
- **b**: free-surface physics that should stay as a concise statement.
- **c**: physics-agnostic machinery that belongs in FE (or Application).
- **d**: dead, legacy, environment-gated or experimental code that could be retired.

## 0. Key facts

- The module grew from 284 lines (`0f8ee0d^`) to 11,030. The physics (a + b) is about 890 lines, 8% of the file. The other 92% is machinery, validation, diagnostics and message text.
- The 284-line pre-free-surface version is still essentially the bulk of the target. What has to be added is the free-surface statement, and that needs FE constructs that do not exist yet.
- The form language already has the measures needed: `.dI(m)`, `.dCutVolume(m, side)`, `dInterfaceBoundary(.., m)` (codimension two), `.dExteriorBoundary(ExteriorBoundaryMeasure::generatedActiveSubset(..))` and `cutAdjacentFacetIntegral(.., m, side)`. What is missing is **typed handles**: the physics module computes stable generated markers by hand, registers them, validates them, certifies the geometry with MPI, and orders constraints.
- Refactor R2 added `FE/Forms/IntegrationDomain.h::VolumeIntegrationDomain`, but only a test uses it. The module still uses its own `integrateOnActiveVolume` (6109–6122).

## 1. Inventory

### 1.1 `Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.cpp` (11,030 lines)

| Lines | n | Class | Content | Destination |
|---|---:|---|---|---|
| 1–85 | 85 | — | 37 includes, constructor | — |
| 86–168 | 83 | c | rank-failure log; `coordinateCutContextCallbackLocalPhase` (MPI_MIN 3-state agreement) | FE/Systems collectives |
| 169–403 | 235 | c | embedded-interface measure evidence (3 `MPI_Allreduce`); `CutVolumePressureAnchorMeasureGuard` GlobalKernel | FE/Constraints gauge anchor |
| 404–746 | 343 | d | 17 environment-hook readers, 34 names (§1.6) | retire, or Application diagnostics config |
| 747–878 | 132 | c | space compatibility; validate, ensure-unknown and ensure-prescribed field; distinct-name check | FE/Systems FieldBinding |
| 879–1533 | 655 | c (+a 13) | active-pressure domain; vertex-field views; `initializeStateFieldFromMeshVertexField`; active-support vertices; `hydrostaticPressureAt` (13 lines, a); node-pressure-constraint checks against the level-set side, with a log | FE/Systems init, FE/LevelSet support, FE/Constraints |
| 1535–1843 | 309 | c (+a ≈5) | `applyInitialConditions`: mesh-field init, hydrostatic loop, dry-side reference pressure, 2 large logs | FE/Systems init helpers |
| 1847–2388 | 542 | c | option predicates; contact-variant accessors; generated marker keys (interface, contact, active boundary); uniqueness; marker resolution | FE/LevelSet geometry handles; enums from the schema |
| 2389–2605 | 217 | c | discrete-functional (energy ledger) declaration that sends Young, Ren–E and viscosity data into FESystem (refactor-plan finding F04) | Physics FreeSurfaceEnergy plus a generic FE functional (R5) |
| 2607–2786 | 180 | c | wall-normal normalization; ScalarValue validators; owner-operator lookup; vec3 algebra | FE/Forms bc, FE/Systems, FE/Math |
| 2787–3325 | 539 | c | planar-wall certificate (MPI); contact covector mapping (MappingFactory); wall-normal and transversality checks; preflight | FE/Interfaces contact certificate |
| 3326–3790 | 465 | d | `logDynamicContactOperatorAngle`: re-derives contact angles from the vertex field, with 4 reductions, only to print a log line | delete, or FE/LevelSet diagnostics |
| 3792–3891 | 100 | b/c | wall-normal FormExpr; axis-aligned restriction (an FE limitation); strong normal-only wall Dirichlet required by D4 | b: `requireEssential`; c: FE normal-trace essential |
| 3893–4066 | 174 | c/d | token normalization of string-typed options; policy and curvature names for logs and artifact | enums plus schema |
| 4067–4120 | 54 | c | semi-implicit capability check | schema rule |
| 4121–4880 | 760 | c | `validateFreeSurfaceBoundary`: capability matrix with 85 throws | schema rules (table owned by Physics) |
| 4882–5127 | 246 | c | BC preflight that builds every BC once before mutating the system | dry-run mode of the BC manager |
| 5129–5157 | 29 | d | raw-curvature warning, unreachable because 4444 already throws for that case; Q1 level-set normal | delete / legacy |
| 5159–5340 | 182 | d | contact-line measure log (debug level only) | delete |
| 5342–5554 | 213 | a (≈45) + c | pressure-gradient facet-jump stabilization: about 45 lines of math, a 60-line log | keep the formula in Physics |
| 5556–5576 | 21 | c | `integrateOnFreeSurface` dispatch | FE/Forms domains |
| 5578–5752 | 175 | c/d | names, side mapping, experimental Hadamard cut-volume tangent, smoothed indicator (legacy) | FE/Forms; delete the experiments |
| 5754–6056 | 303 | b (≈110) + c | Young term, Ren–E friction, Navier slip; geometry callbacks; logs | keep the math |
| 6058–6152 | 95 | c | active volume domain and its integration | FE/Forms `VolumeDomain` |
| 6153–6315 | 163 | c/d | level-set and curvature field accessors; semi-implicit reference velocity; normal-speed and point-motion factors | FE/LevelSet; `lagged()` |
| 6317–6498 | 182 | d | experimental shape tangents: environment-gated, off by default, rejected with SurfaceStress and KAG | delete |
| 6500–6670 | 171 | c | extra trial fields, 4 logs | builder |
| 6672–6707 | 36 | b | pinned fitted contact line (strong `d_mesh = 0`) | keep |
| 6709–6799 | 91 | c | prescribed-angle geometry callbacks | FE/LevelSet |
| 6801–7300 | 500 | b (≈190) + c/d | `applyFreeSurfaceBoundary`: SurfaceStress, KAG and curvature traction; semi-implicit term; fitted Penalty and Nitsche; 2 environment probes (d); 7 out-parameter diagnostic forms; 10 logs | keep the math |
| 7302–7622 | 321 | b (≈110) + c | fitted mesh-row kinematics; tangential policy; 4 BC descriptors | math stays; declarations go to FE/Systems ALEBinding |
| 7624–8349 | 726 | c | JSON escaping; effective-configuration artifact v4 | schema serializer |
| 8351–8590 | 240 | c/d | `registerOn` part 1: schema-version checks, environment override resolution, 10 logs | schema; retire |
| 8591–9050 | 460 | c | field specs; ALE options; free-surface preflight (fitted ownership, KAG and semi-implicit spaces, generated Nitsche certification); field registration; energy-functional declaration | FieldBinding, ALEBinding preflight, schema |
| 9052–9254 | 203 | b/c | active-boundary marker lambda; gauge anchor plus guard; pending inactive-side and aggregation requests; node pressure constraints | ConstraintPlan |
| 9255–9318 | 64 | c | FormExpr fields; ALE binding; fitted normal declarations | — |
| **9319–9486** | **168** | **a** | **the PDE**: forcing, μ(γ̇), ALE convection, strong residual, Galerkin, τ_M/τ_C, SUPG, cross stress | keep |
| 9488–9810 | 323 | a (≈25) + d | PSPG split; cut-volume cap experiment; 4 boundary-PSPG experiments (environment, default 0) | keep the split; delete the rest |
| 9812–9958 | 147 | a (≈16) + c/d | active-volume integration; residual-work forms; conservative-balance forms; shape tangents | builder channels |
| 9960–10096 | 137 | a/b | BC manager; free-surface loop; standard and pressure BCs | keep |
| 10097–10228 | 132 | b/c | inactive-side constraints, aggregation, strong Ren–E contact point. **The order is load-bearing.** | ConstraintPlan |
| 10230–10322 | 93 | a/c | Nitsche; generated-trace bookkeeping; main `installFormulation` | builder |
| 10324–10554 | 231 | c | 7 free-surface residual-work channels as separate operators; boundary-energy declaration | builder channels |
| 10556–10656 | 101 | d | symmetric-Nitsche energy operators (environment) | diagnostics registry |
| **10658–10676** | 19 | **a** | **separate PSPG ∇p kernel group on the production operator** | builder group |
| 10678–10811 | 134 | d | conservative-balance diagnostics (environment) | diagnostics registry |
| 10813–10933 | 121 | d | pressure-row diagnostics (environment) | diagnostics registry |
| 10935–10963 | 29 | d | shape-tangent mixed bilinear | delete |
| 10965–11030 | 66 | c | fitted normal declarations; artifact | ALEBinding; schema |

**Totals (approximate):** a ≈ 370 (3%), b ≈ 520 (5%), c ≈ 7,900 (72%), d ≈ 2,200 (20%). The file has 258 throw sites, 74 `FE_LOG_*` calls and 11 `MPI_Allreduce` calls. About 1,060 lines of stream or concatenation code plus about 470 lines of string literals (about 14% of the file) exist only to build message text.

### 1.2 `IncompressibleNavierStokesVMSModule.h` (424 lines)

- **63–158 (96 lines):** three operator-name structs with 17 names, consumed by `FE/TimeStepping/NewtonSolver.cpp` (around lines 268, 14499 and 14776).
  - `FreeSurfaceConservativeBalanceDiagnosticOperators` and `SymmetricNitscheEnergyDiagnosticOperators` are d.
  - `FreeSurfaceResidualWorkOperatorSuffixes` is c.
- **160–163 and 180–185:** the qualification scope is d.
- **165–397 (233 lines, about 76 members):**
  - a: density, viscosity and its model, body force (constant, space-time and field), Coriolis, convection, ALE flags, VMS constants, BC vectors, Nitsche constants.
  - c: field names, `operator_tag`, auto-register flags, JIT policy, tangent path.
  - d: `explicit_legacy_configuration`, schema-1 mode, the 0.25 generated-Nitsche energy ratio (this one is a policy that belongs in FE).

### 1.3 `FreeSurface/FreeSurfaceOptions.h` (352 lines; extracted under R1 and now on mainline)

| Lines | Member | Class | Note |
|---|---|---|---|
| 24–27 | `FreeSurfaceImplementation` | b | keep |
| 38–41 | `FreeSurfaceBoundaryRole::InternalMaterialInterfaceVolume` | d | two-fluid uses it to choose a phase volume (finding F02); replace it with a `VolumeDomain` argument |
| 43–45 | `FreeSurfacePhysicalModel` (one value) | d | keep only in the schema and artifact |
| 47–56 | active domain; `CutVolume`, `SmoothedIndicator` | c / d | `SmoothedIndicator` is allowed only in schema-1 legacy mode (8643) |
| 58–82 | kinematic enforcement and policies | b | Penalty and Nitsche are legacy per D35 |
| 84–89 | pressure-stabilization policy | a/c | `DisabledForRefreshedFrozenHighOrder` is d, because high-order fails closed elsewhere |
| 107–113 | `SurfaceTensionForm` (5 values) | b | unfitted CurvatureTraction and GeneratedCurvatureTraction are d candidates; no benchmark generator uses them |
| 128–131 | semi-implicit | b | — |
| 141–194 | contact-line variant | b | keep |
| 196–210 | cut-cell stabilization | a/c | keep (fluid scaling) |
| 212–217 | `FreeSurfaceVelocityExtension` | d | `enabled = true` is rejected on every path (4294, 4465, 4528) |
| 229–242 | markers, level-set field, domain id, **string** geometry and tangent policy, isovalue | c | → FE `LevelSetGeneratedInterfaceOptions` with enums |
| 245–258 | side, method, smoothing width, `allow_full_domain` | c/d | — |
| 268–292 | `p_ext`, γ, form, semi-implicit, `allow_fitted_surface_stress` | b | — |
| 297–300 | curvature inputs | b/d | raw level-set curvature is rejected at 4444; pointwise fitted curvature was withdrawn by D5 |
| 303–321 | fitted kinematics and tangential settings | b | — |
| 323–344 | cut stabilization, contact lines, aggregation and guards | b/c | — |

### 1.4 `NavierStokesRegister.cpp` (5,835 lines)

| Lines | n | Class | Content | Target |
|---|---:|---|---|---|
| 47–542 | 496 | c/d | rejection of two-phase and jump scope; schema-version contract; physical-model resolution | schema forbidden-key and version policy (≈60 lines) |
| 543–703 | 161 | c | trim, parse and lookup helpers; trace environment variable. Copies of these helpers exist in 10 files. | `Physics/Core/ParameterParsing.h` |
| 704–1562 | 859 | c | `Temporal_and_spatial_values` reader with 17 MPI calls | Application/Translators (B) |
| 1563–1648 | 86 | c | element type and order inference; single-domain selection | Physics/Core shared |
| 1649–1818 | 170 | c | node-pressure CSV | Application reader |
| 1819–2076 | 258 | a-options | density, body force, hydrostatic init, viscosity models, ALE keys | schema |
| 2077–2783 | 707 | c | vec3 algebra; marker geometry and perimeter gather (22 MPI calls); parabolic inlet profile | FE/Geometry or Application (B) |
| 2784–3116 | 333 | c | free-surface enum parsers; `first_defined_*` | schema enum tables |
| 3117–3310 | 194 | c | two-fluid required and rejected parameters | schema |
| 3311–3538 | 228 | a/c | Coriolis; space-time source file | schema plus reader |
| 3539–3920 | 382 | b-options | contact-line parse (55 keys) | schema |
| 3921–4434 | 514 | b-options | `append_free_surface_bc` (163 keys including aliases, with "implies" and "legacy-only" rules) | schema |
| 4436–4524 | 89 | c | fitted contact capability | schema rule |
| 4525–5126 | 602 | a-options | standard BC translation (Dir, Neu, Trac, Robin, RCR, RCRCR, profiles, weak) | BC schema plus dispatch (≈250) |
| 5127–5409 | 283 | c | two-fluid translation | schema |
| 5410–5476 | 67 | — | `create_navier_stokes_from_input` | keep |
| 5477–5835 | 359 | c | two-fluid preflight; pre-registration of dependency fields for the level-set equation | FE/Systems declare-fields phase |

Other counts: 349 distinct quoted keys, 70 `first_defined_*` calls, 42 `get_defined_*` calls and 209 throws. The Application re-parses the same free-surface keys in `Application/Core/LevelSetCutConfiguration.cpp` (347–420) and `Application/Translators/EquationTranslator.cpp`.

### 1.5 Other files

- **`NavierStokesBCFactories.h` (620 lines):** a. The natural, Robin, essential, outflow, RCR and RCRCR builders are already in good shape. About 120 lines of generated-boundary Nitsche binding are c.
- **`IncompressibleTwoFluidModule.cpp` (1,044 lines):**
  - 378–453 build fake `FreeSurfaceBoundary` options (finding F02).
  - 454–588 are the artifact.
  - About 300 lines are validation.
  - `registerOn` (662–998) instantiates two one-phase modules plus the interface forms.
- **`IncompressibleTwoFluidInterface.cpp` (303 lines):** b, already written math-first. It is a good model.
- **Semi-implicit term (131 lines):** b, fine as is.

### 1.6 `SVMP_*` environment hooks

The module reads 34 names, which make 17 controls. The register reads 1 more. The header comments mention 2 of the same names.

| # | Control (aliases) | Read at | Effect | Kind | Outside consumers | Proposal |
|---|---|---|---|---|---|---|
| 1 | `SVMP_NS_ENABLE_VMS` / `SVMP_NS_DISABLE_VMS` | 461 | overrides `enable_vms` | residual | `test_SharpBoundaryOperatorMPI` | retire; the option already exists |
| 2 | `SVMP_NS_PSPG_PRESSURE_GRADIENT_SCALE` (+`SVMP_PSPG_…`) | 589 | scales PSPG ∇p | residual | none | retire |
| 3 | `…PSPG_NONPRESSURE_RESIDUAL_SCALE` (+alias) | 599 | scales PSPG R without ∇p | residual | none | retire |
| 4 | `…PSPG_PRESSURE_GRADIENT_CUT_VOLUME_SCALE_CAP` (+alias) | 609 | applies min(cap, 1/fraction) | residual | none | retire |
| 5 | `…PSPG_PRESSURE_GRADIENT_FORM` (+alias) | 698 | absolute or incremental (the default is incremental exactly when KAG is used) | residual | none | turn into a typed option and keep the default rule |
| 6 | `…PSPG_CONTINUITY_FULL_CELL_SUPPORT` (+alias) | 522 | PSPG integrated on full cells | residual | `test_MovingDomainPhysics` | retire |
| 7–10 | `…PSPG_BOUNDARY_{PRESSURE_GRADIENT, PRESSURE_FLUX, TANGENTIAL_PRESSURE_GRADIENT, TANGENTIAL_MOMENTUM_RESIDUAL}_SCALE` (+aliases) | 620–662 | add wall boundary forms | residual | `test_SharpBoundaryOperatorMPI` | retire |
| 11 | `…FREE_SURFACE_PRESSURE_REFERENCE_PROBE_PENALTY` (+alias) | 569 | adds `∫Γ c(p−p_ext)q` | residual | none | retire |
| 12 | `…FREE_SURFACE_TANGENTIAL_PRESSURE_GRADIENT_SCALE` (+alias) | 579 | adds `∫Γ c ∇Γp·∇Γq` | residual | none | retire |
| 13 | `SVMP_CONSTRAIN_INACTIVE_ACTIVE_DOMAIN_VELOCITY` | 709 | `0` drops the dry-side velocity constraints | constraint | none | retire |
| 14 | `SVMP_ENABLE/DISABLE_UNFITTED_LEVEL_SET_SHAPE_TANGENTS` | 727 | experimental φ-shape tangents | tangent | `test_MovingDomainPhysics` | retire together with 6317–6498 |
| 15 | `SVMP_NS_PRESSURE_ROW_CONTRIBUTION_DIAGNOSTIC` (+alias) | 475 | adds 12 diagnostic operators | observation | `NewtonSolver`, a cut-stability test, 2 audit scripts | diagnostics request |
| 16 | `SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC` (+3 aliases, including `…EQUILIBRIUM…`) | 490 | adds 7 operators | observation | `NewtonSolver`, 3 test files, 2 Python scripts | diagnostics request |
| 17 | `SVMP_NS_SYMMETRIC_NITSCHE_ENERGY_DIAGNOSTIC` | 513 | adds 4 operators | observation | 2 tests | diagnostics request |
| + | `SVMP_OOP_SOLVER_TRACE` (register, 557) | — | trace logging | observation | 14 files | resolve once in Application |

Breakdown: 25 names change the residual or constraints, 2 change the tangent, 7 only observe. No `tests/cases/fluid/free_surface_benchmarks` generator sets any of them.

## 2. Target top-level formulation

### 2.1 Rules

- Follow `Documentation/mesh_motion_math_first_formulation_guide.md`: FE exposes mathematical primitives and typed domains, never PDE wrappers. The physics module writes the weak form visibly.
- No `getenv`, MPI, JSON, string building, marker hashing or constraint ordering inside a formulation.
- **First phase: identical trees.** The refactored code must produce byte-identical FormExpr trees, in the same order and with the same association. Each new construct below lowers to exactly the primitive the current code uses.

### 2.2 The module (target shape; constructs marked `NEW` are proposals)

```cpp
// IncompressibleNavierStokesVMSModule.cpp  (target ≈ 470 lines)
namespace svmp::Physics::formulations::navier_stokes {
using namespace svmp::FE::forms;
namespace fs = FE::systems; namespace ls = FE::level_set; namespace cs = FE::constraints;

// Bulk residual, also called by IncompressibleTwoFluidModule on Ω⁻ and Ω⁺.
//   R_m = ρ(∂t u + (∇u)a + χ(∇·w)u − f) − ∇·(2μ ε(u)) + ∇p,   a = u − w,
//   u' = −τ_M R_m,  p' = −τ_C ∇·u  (static subscales: SUPG/PSPG/LSIC and the τ_B cross stress)
NavierStokesVMSForms navierStokesVMSResidual(const NSTerms& t, const VMSOptions& s)
{
    const auto& [u, p, v, q, rho, mu, f, dim] = t;
    const auto zero = zeroVector(dim);
    const auto w    = s.ale ? meshVelocity() : zero;
    const auto a    = s.convection ? (s.ale ? (u - meshVelocity()) : u) : zero;
    const auto chi  = s.moving_control_volume ? (div(w) * u) : zero;
    const auto R_np = rho * (dt(u) + grad(u) * a + chi - f)
                    - div(FormExpr::constant(2.0) * mu * sym(grad(u)));   // R_m without ∇p
    const auto R_m  = R_np + grad(p);

    const auto inertia  = rho * inner(dt(u), v);
    const auto mcv      = s.moving_control_volume ? rho * div(meshVelocity()) * inner(u, v)
                                                  : FormExpr::constant(0.0);
    const auto viscous  = FormExpr::constant(2.0) * mu * inner(sym(grad(u)), sym(grad(v)));
    const auto pressure = -p * div(v);
    const auto forcing  = -rho * inner(f, v);
    const auto convect  = rho * inner(grad(u) * a, v);

    NavierStokesVMSForms r;
    r.galerkin   = inertia + mcv + convect + viscous + pressure + forcing;
    r.momentum   = r.galerkin;
    r.continuity = q * div(u);
    r.channels   = {{Channel::Convection, mcv + convect}, {Channel::PressureWork, pressure}};   // NEW (named sub-forms)
    if (!s.vms) return r;

    const auto eps = FormExpr::constant(s.epsilon), dts = FormExpr::effectiveTimeStep();
    const auto ct_m = FormExpr::constant(s.ct_m), ct_c = FormExpr::constant(s.ct_c);
    const auto K   = transpose(Jinv()) * Jinv();                 // element metric J^{-T} J^{-1}
    const auto nu  = mu / rho;
    const auto kT  = FormExpr::constant(4.0) * (ct_m * ct_m) / (dts * dts);
    const auto tau_m = FormExpr::constant(1.0) /
        (rho * sqrt(kT + inner(a, K * a) + ct_c * doubleContraction(K, K) * (nu * nu) + eps));
    const auto tau_c = FormExpr::constant(1.0) / (tau_m * trace(K) + eps);
    const auto u_s   = -tau_m * R_m;
    const auto p_s   = -tau_c * div(u);
    const auto u_adv = s.convection ? (u + u_s - w) : a;
    const auto supg  = -rho * inner(grad(v) * u_adv, u_s);
    const auto cross = s.convection
        ? inner(grad(v) * u_s, (rho / sqrt(inner(u_s, K * u_s) + eps)) * (grad(u) * u_s))
        : FormExpr::constant(0.0);
    r.momentum = inertia + mcv + rho * inner(grad(u) * u_adv, v) + viscous
               + (-(p + p_s)) * div(v) + forcing + supg + cross;
    // PSPG, split: the ∇p part may use Δt ∂t p (incremental form, the default under KAG)
    // and is installed as its own kernel group.
    const auto one = FormExpr::constant(1.0);   // first phase keeps the unit scale factors (kernel identity)
    r.continuity = q * div(u) + one * inner(grad(q), tau_m * R_np);
    r.pspg_grad_p = one * one * inner(grad(q), tau_m * grad(s.incremental_pspg ? dts * dt(p) : p));
    return r;
}

void IncompressibleNavierStokesVMSModule::registerOn(fs::FESystem& system) const
{
    const auto& o = options_;
    navierStokesOptionSchema().preflight(o, system);                 // NEW: capability rules, no mutation

    fs::FieldBinding fields(system, "IncompressibleNavierStokesVMSModule");          // NEW
    const auto u_id = fields.unknown(o.velocity_field_name, velocity_space_);       // same order as today:
    const auto p_id = fields.unknown(o.pressure_field_name, pressure_space_);       // u, p, f_h, ū, then ALE
    const auto f_id = fields.prescribed(o.body_force_field_name, velocity_space_, o.auto_register_body_force_field);

    // Ω, or the liquid side Ω_h = {φ_h ≶ 0} of the generated interface Γ_h
    const auto liquid = liquidSide(system, o);        // optional<ls::CutSide>, from ls::LevelSetGeometry::bind  NEW
    const auto Omega  = liquid ? liquid->volume() : VolumeDomain::full();             // NEW (extends R2)

    cs::ConstraintPlan plan(system);                                                  // NEW: phased registration
    plan.vertexValues(p_id, o.node_pressure_constraints);
    const auto ale = fs::resolveALEBinding(system, aleOptions(o, velocity_space_));
    if (!system.hasOperator(o.operator_tag)) system.addOperator(o.operator_tag);

    const int dim = velocity_space_->value_dimension();
    const auto u = StateField(u_id, *velocity_space_, o.velocity_field_name);
    const auto p = StateField(p_id, *pressure_space_, o.pressure_field_name);
    const auto v = TestField(u_id, *velocity_space_, "v");
    const auto q = TestField(p_id, *pressure_space_, "q");
    const NSTerms t{u, p, v, q, FormExpr::constant(o.density), viscosity(o, u), bodyForce(o, dim, f_id, u), dim};

    const auto r = navierStokesVMSResidual(t, vmsOptions(o));
    FormExpr M = r.momentum.dx(Omega);                                                // NEW overload .dx(VolumeDomain)
    FormExpr C = r.continuity.dx(Omega);

    // ∂Ω conditions; with a liquid side they act on the wetted part ∂Ω ∩ Ω̄_h
    const auto dsOf = [&](int m) { return liquid ? liquid->boundary(m) : BoundaryDomain::fullPhysical(m); }; // NEW
    fs::BoundaryConditionManager bcs;
    bcs.install(o.velocity_dirichlet_weak, Factories::reserveMarker);
    for (const auto& b : o.free_surface)
        b.implementation == FreeSurfaceImplementation::UnfittedLevelSet
            ? addFreeSurface(M, C, plan, b, *liquid, t, o)                            // §2.3
            : addFittedFreeSurface(system, M, bcs, plan, b, ale, t, o);               // §2.4
    bcs.install(o.traction_neumann, [&](auto& b) { return Factories::toTractionBC(b, dim, dsOf); });
    bcs.install(o.traction_robin,   [&](auto& b) { return Factories::toTractionRobinBC(b, dim, dsOf); });
    bcs.install(o.pressure_outflow, [&](auto& b) { return Factories::toOutflowBC(b, u, t.rho, dsOf); });
    bcs.install(o.coupled_outflow_rcr,   [&](auto& b) { return Factories::toCoupledOutflowBC(b, system, u, t.rho, dsOf); });
    bcs.install(o.coupled_outflow_rcrcr, [&](auto& b) { return Factories::toCoupledOutflowBC(b, system, u, t.rho, dsOf); });
    bcs.install(o.velocity_dirichlet, [&](auto& b) { return Factories::toVelocityEssentialBC(b, dim, o.velocity_field_name); });
    plan.essentials(bcs, system, M, u, v, u_id);     // = bcs.applyAll, then the pressure BCs, then the later phases
    Factories::applyVelocityNitscheBCs(M, C, o, dim, u, p, v, q, t.mu, dsOf);

    fs::FormulationBuilder F(system, o.operator_tag, physicsInstallOptions(o.jit_policy));   // NEW
    F.symbolicTangent().recordViscosity(u_id, o.viscosity, o.viscosity_model).configure(ale);
    F.residual({u_id, p_id}, M + C);
    if (r.pspg_grad_p.isValid())
        F.group("navier_stokes_vms_pspg_pressure_gradient", {p_id}, r.pspg_grad_p.dx(Omega), {.extra_trial = {u_id}});
    F.channels(r.channels, Omega);       // installed only if an energy or work consumer declared interest
    F.install();                         // reproduces today's installFormulation call order
}
```

`viscosity` (12 lines), `bodyForce` (20), `aleOptions` and `vmsOptions` (15), `liquidSide` (10) and the two free-surface functions below fill out the line budget in §2.8.

### 2.3 Unfitted free surface (the statement of tracker §6.1)

```cpp
//  ∫Γ p_ext n·v + γ∫Γ (I − n⊗n):∇v − γ cosθe ∫CL v·m + ∫CL ξ (u·m)(v·m) + ∫W (μ/ℓs) u_t·v_t
//  u·n_w = 0 (strong);  φ transported by an extension of u;  small cut cells aggregated.
void addFreeSurface(FormExpr& M, FormExpr& C, cs::ConstraintPlan& plan, const FreeSurfaceBoundary& b,
                    const ls::CutSide& L, const NSTerms& t, const Options& o)
{
    const auto& [u, p, v, q, rho, mu, f, dim] = t;
    const auto n = L.outwardNormal();     // NEW: the rule normal, flipped for the positive side (= today's −normal())
    const auto Gamma = L.interface();     // NEW: InterfaceDomain → .dI(marker)
    const auto g = FormExpr::constant(b.surface_tension), p_ext = FormExpr::constant(b.external_pressure);
    L.requireMaterialInterface(t.u_id);   // NEW: φ's transport velocity is (an extension of) u. Kinematic condition.

    for (const auto& c : b.contact_lines) {                           // contact terms come first, as today
        const auto CL  = L.contactLine(c.wall_marker, c.wall_normal);  // NEW: Γh ∩ wall, certified planar
        const auto n_w = CL.wallNormal(), m = CL.conormal();           // m = safeNormalize(n − (n·n_w)n_w)
        if (c.isWeak())                                                // Young term (+ Ren–E friction ξ = 1/M)
            M = M + dInterfaceBoundary((c.friction(u, m) - g * FormExpr::constant(std::cos(c.theta_e))) * dot(v, m), CL); // NEW typed overload
        else
            plan.contactPointVelocity(t.u_id, CL, [=](Real cos_d) { return c.M * b.surface_tension * (std::cos(c.theta_e) - cos_d); });
        const auto ut = u - dot(u, n_w) * n_w, vt = v - dot(v, n_w) * n_w;
        M = M + (mu / FormExpr::constant(c.slip_length) * dot(ut, vt)).ds(L.wetted(c.wall_marker)); // NEW wetted wall W_h
        plan.requireEssential(normalTrace(u, n_w), c.wall_marker);    // D4 strong no-penetration (phase 1: check only)
    }
    if (b.form == Capillary::SurfaceStress)        // σn = −(p_ext + γκ)n in variational form
        M = M + (b.surface_tension == 0 ? p_ext * inner(n, v)
                 : p_ext * inner(n, v) + g * inner(FormExpr::identity(dim) - outer(n, n), grad(v))).dI(Gamma);
    else                                           // KAG: κh from FE/LevelSet curvature recovery
        M = M + (p_ext * inner(n, v)).dI(Gamma) + (g * L.recoveredCurvature() * inner(n, v)).dI(Gamma); // NEW handle
    if (b.semi_implicit)                           // γΔt ∫Γ ∇Γ((u − ū)·n)·∇Γ(v·n), ū frozen at the geometry refresh
        M = M + semiImplicitNormalIncrementIntegrand(g, u, lagged(u, fs::Refresh::GeneratedState), v, n).dI(Gamma); // NEW lagged()

    const auto h = avg(hNormal()), mu_gp = mu + rho * h * h / FormExpr::effectiveTimeStep();   // pressure facet jump
    C = C + (cutScale(b) * FormExpr::constant(b.pressure_jump_penalty) * FormExpr::constant(0.01) * (h * h * h)
             / (mu_gp + FormExpr::constant(o.stabilization_epsilon))
             * inner(jump(grad(pPSPG(b, p))), jump(grad(q)))).dS(L.cutFacets());             // NEW .dS(FacetSet)

    plan.inactiveSide({t.u_id, t.p_id}, L, 0.0);              // dry DOFs pinned
    plan.aggregate({t.u_id, t.p_id}, L, b.aggregation_guards); // AgFEM small-cut aggregation (constraint declaration)
    plan.anchorGauge(t.p_id, Gamma);                           // traction on |Γh| > 0 fixes the pressure constant
}
```

How each item of tracker §6.1 maps onto the sketch:

| §6.1 item | Target |
|---|---|
| Ω_h, `dCutVolume` | `.dx(L.volume())` |
| Γ_h with rule normal, outward from the liquid | `.dI(L.interface())`, `L.outwardNormal()` |
| SurfaceStress | `γ(I−n⊗n):∇v` |
| Young term `−γ cosθe ∫CL v·m` | `dInterfaceBoundary(.., CL)` |
| Ren–E weak law, or strong law `u·t_w = Mγ(cosθe − cosθ)` | `c.friction`, or `plan.contactPointVelocity` |
| Navier slip on the wetted wall | `.ds(L.wetted(wall))` |
| Strong no-penetration | `plan.requireEssential` |
| Pressure facet jump `γp·0.01·h³/(μ+ρh²/Δt)` | `.dS(L.cutFacets())` |
| Aggregation, inactive pressure pinned to 0 | `plan.aggregate`, `plan.inactiveSide` |
| Kinematics by level-set transport, with PDE extension (D9/D15), redistancing (D35), reconciliation (D14), patch bounds (D21) | owned by the level-set equation; the formulation states only `requireMaterialInterface(u)` |
| KAG | `L.recoveredCurvature()` |
| Semi-implicit term (D13) | `lagged(u)` |

Two orders from the current code are kept on purpose: contact terms before the traction, and SurfaceStress as one `.dI` of a sum. Also kept: the γ = 0 pressure-only branch and the two separate KAG integrals.

### 2.4 Fitted ALE

```cpp
// fluid rows: ∫Γ(t) p_ext n·v + γ∫ (I−n⊗n):∇v  (SurfaceStress opt-in) or ∫ (p_ext + γκ) n·v;
// mesh rows:  ∫Γ (γN/h_n)((ḋ − u)·n)(ψ·n)  (MeshNitsche, D35)  + tangential policy (Free / SmoothingOnly / Prescribed)
void addFittedFreeSurface(...)
{
    const auto Gm = BoundaryDomain::fullPhysical(b.boundary_marker);  // = ExteriorBoundaryMeasure::fullPhysical
    const auto n  = currentNormal();
    M = M + fittedTraction(b, n, v, dim).ds(Gm);                      // two branches, about 6 lines
    const auto d = ale.displacementState(), psi = ale.displacementTest();   // NEW accessors on ALEBinding
    ale.installRows(fs::FormulationBuilder::Rows{b.gammaN_over_h() * normalTrace(dt(d) - u, n) * normalTrace(psi, n)
                                                 + tangentialPolicyTerm(b, d, psi, n)}.ds(Gm), {.extra_trial = {t.u_id}});
    ale.declareNormalKinematicOwner(b.boundary_marker, normalTrace(u, n), b.requiresFluxConsistency());  // existing declarations
    ale.declareTangentialPolicy(b.boundary_marker, b.tangential_mesh_policy);
    if (b.pinned()) plan.essential(ale.displacement(), b.contact_marker, 0.0);
}
```

The mesh-row install must stay **before** the main `installFormulation`, as it is today (10031 against 10321).

### 2.5 Two-fluid

`IncompressibleTwoFluidModule` becomes a composition:
- `navierStokesVMSResidual` on `G.side(Negative).volume()` and on `G.side(Positive).volume()`;
- plus `buildIncompressibleTwoFluidInterfaceForms(..)` on `G.interface()`;
- plus `plan.aggregate` on each side.

This removes `FreeSurfaceBoundaryRole::InternalMaterialInterfaceVolume` and `makePhaseOptions` (378–453).

### 2.6 Acceptance test that the constructs are physics-agnostic

```cpp
const auto L = ls::LevelSetGeometry::bind(system, {.level_set_field = "phi", .domain_id = "inclusion"}).side(Side::Negative);
auto R = (k * inner(grad(T), grad(s)) - f * s).dx(L.volume()) + (alpha * (T - T_inf) * s).dI(L.interface());
cs::ConstraintPlan(system).aggregate({T_id}, L, {}).inactiveSide({T_id}, L, 0.0).commit();
```

This is a cut-cell Poisson problem with a Robin condition on an evolving inclusion. Refactor step R12 already asks for a test of this kind.

### 2.7 Constructs the form language lacks

| # | Construct | Proposed syntax | Lowers to (keeps trees identical) | FE home |
|---|---|---|---|---|
| 1 | Volume domain | `expr.dx(VolumeDomain)`; `VolumeDomain::full()` / `L.volume()` | `.dx()` / `.dCutVolume(m, side)` | FE/Forms/IntegrationDomain.h (extend `VolumeIntegrationDomain`) |
| 2 | Interface domain | `expr.dI(L.interface())` | `.dI(m)` (exists, untyped) | FE/Forms + FE/LevelSet |
| 3 | Contact-line measure | `dInterfaceBoundary(expr, ContactLineDomain)` (alias `.dL(CL)`) | `.dI(stable intersection marker)`, codimension two | FE/LevelSet/LevelSetGeometry |
| 4 | Wetted or clipped boundary | `L.boundary(m)`, `L.wetted(wall)` → `BoundaryDomain` | `dExteriorBoundary(generatedActiveSubset(m, key))` plus marker registration | FE/LevelSet |
| 5 | Cut-adjacent facet set | `expr.dS(L.cutFacets())` | `cutAdjacentFacetIntegral(expr, m, side)` | FE/Forms/CutCellForms.h |
| 6 | Oriented normal, wall normal, conormal | `L.outwardNormal()`, `CL.wallNormal()`, `CL.conormal()` | `±normal()`, `asVector(constants)`, `safeNormalize(n−(n·n_w)n_w)` | FE/LevelSet + FE/Interfaces certificate |
| 7 | Recovered curvature | `L.recoveredCurvature(policy)` | `discreteField(κ)`, with the sign flip on the positive side | FE/LevelSet/LevelSetCurvatureProjection |
| 8 | Lagged copy | `lagged(u, Refresh::GeneratedState)` | auto-registered prescribed field; replaces `Application/Core/FreeSurfaceSemiImplicitReference.cpp` | FE/Systems |
| 9 | Constraint declarations | `plan.vertexValues/essentials/requireEssential/inactiveSide/aggregate/contactPointVelocity/anchorGauge` | existing constraint classes, in today's order | FE/Constraints/ConstraintPlan.h |
| 10 | Kinematics through the level set | `L.requireMaterialInterface(u_id)` (phase 1, a check); later `G.declareKinematics({.velocity, .extension, .maintenance})` | validates against the `LevelSetTransportOptions` of the level-set equation | FE/LevelSet |
| 11 | Kernel groups, channels, diagnostics | `F.residual`, `F.group`, `F.channels`, `F.diagnostic` | today's `installFormulation` sequence | FE/Systems/FormulationBuilder.h |
| 12 | Named sub-expression | `tagged(Channel::SurfaceEnergy, e)` | removed before IR hashing | FE/Forms |
| 13 | Field binding | `FieldBinding::unknown/prescribed` | the logic at 747–878 | FE/Systems (generalizes `MeshDisplacementBinding`) |
| 14 | No-tangent marker (optional) | `frozen(phi)` | removed from the residual IR; no trial column | FE/Forms |
| 15 | Option schema | §4 | — | Physics/Core/OptionSchema.h |

### 2.8 Line budget for the target module

| Part | Lines |
|---|---:|
| Includes, constructor | 35 |
| Preamble (preflight, fields, domains) | 40 |
| μ and f helpers | 35 |
| Bulk VMS residual | 75 |
| Standard BCs and Nitsche | 35 |
| Unfitted free surface | 70 |
| Fitted free surface | 55 |
| Install | 30 |
| Initial conditions: mesh-field init, `p0 = p_ref + ρg·(x−x_ref)` on liquid support | 30 |
| Comments and blank lines | 65 |
| **Total** | **≈470** |

The header drops to about 200 lines once the schema-generated options and the diagnostic-name registry exist. `FreeSurfaceOptions.h` drops to about 180 lines.

## 3. Destinations for class (c) items

| Item (current lines) | FE destination | Proposed name |
|---|---|---|
| Phase agreement and reductions (86–168, 216–289, 2985–3022, 3644–3721) | FE/Systems | `SystemCollectives.h`: `agreeOnPhase`, `allReduce{Sum,Min,Max}(system, …)` on one communicator |
| Measure evidence and pressure-anchor guard plus callback (169–403, 9097–9161) | FE/Constraints | `InterfaceMeasureGaugeAnchor` via `ConstraintPlan::anchorGauge` |
| Field validation and registration (747–878, 8887–9039) | FE/Systems | `FieldBinding.h` |
| Level-set vertex views, active support, side tests (879–1349) | FE/LevelSet | `LevelSetGeometry::CutSide::supportVertices()`, `VertexFieldView` |
| Mesh-field and hydrostatic initial conditions (1056–1176, 1535–1843) | FE/Systems | `initializeFromMeshVertexFields`, `initializeVertexValues(field, fn, support, outside)` |
| Pressure node constraints checked against the level-set side (1364–1531) | FE/Constraints | `ConstraintPlan::vertexValues(..., requireInside(L))` |
| Generated marker keys, uniqueness, resolution (2146–2361, 9052–9063) | FE/LevelSet | `LevelSetGeometry.{h,cpp}` (`interface`, `contactLine`, `boundary`, `wetted`, `cutFacets`) |
| Energy-functional declaration (2389–2605) | Physics FreeSurface/FreeSurfaceEnergy, with a generic FE functional | R5 |
| ScalarValue validators, vec3 algebra, owner lookup (2607–2786) | FE/Forms bc; FE/Math; FE/Systems | `bc::require{Finite,Positive,Nonnegative}`, `FESystem::rowOwnerOperator` |
| Planar-wall, covector and transversality certification (2787–3325, 3792–3843) | FE/Interfaces | `ContactLineGeometryCertificate` (configured-normal alignment within 1e-8, sine ≥ 1e-6, checked at every refresh) |
| String tokens and policy names (3893–4066) | Physics/Core/OptionSchema | enums |
| Capability validation (4067–4880) | Physics/Core/OptionSchema | `FreeSurfaceCapabilityRules` (a physics table) |
| BC preflight (4882–5127) | FE/Systems | `BoundaryConditionManager::validateOnly` |
| Domain dispatch (5556–5669, 6058–6152) | FE/Forms | `IntegrationDomain.h` |
| Level-set and curvature accessors; semi-implicit reference (6153–6219, 8806–8828) | FE/LevelSet; FE/Systems | `CutSide::levelSet()`, `recoveredCurvature()`, `lagged()` |
| Extra trial fields (6500–6670) | FE/Systems | builder `couple()` / `frozen()` |
| Contact-geometry callbacks (6709–6799, 5954–5995) | FE/LevelSet | certificate hook inside `contactLine()` |
| Fitted descriptors and declarations (7302–7622, 9266–9310, 10965–11010) | FE/Systems | `ALEBinding::installRows`, `declareNormalKinematicOwner`, `declareTangentialPolicy` |
| JSON artifact (7624–8349) | Physics/Core/OptionSchema, with an Application writer | schema serializer |
| Fitted ownership preflight (8641–8795) | FE/Systems | `ALEBinding::validateNormalOwnership` |
| Generated Nitsche certification bookkeeping (8841–9009, 10254–10315) | FE/Systems | inside `FormulationBuilder` |
| Constraint order (9080–9232, 10097–10228) | FE/Constraints | `ConstraintPlan` phases |
| Residual-work channels (10324–10554) | FE/Systems | `FormulationBuilder::channels` (generic `ResidualChannel`) |
| Diagnostic operators (10556–10933) | FE/Systems | `DiagnosticRegistry`; `NewtonSolver` looks names up there instead of hard-coding them |

## 4. Options and registration

- **One declarative schema per option struct** (`Physics/Core/OptionSchema.h`). Each entry carries: member pointer, ordered aliases, type and lexicon, default, validator, "implies" and "conflicts" rules, legacy-only acceptance, artifact key and order, and documentation. Sketch:

```cpp
OptionSchema<FreeSurfaceBoundary>("Free_surface")
  .key(&FSB::active_domain, {"Active_domain","ActiveDomain","Free_surface_active_domain","FreeSurfaceActiveDomain"},
       Enum<FreeSurfaceActiveDomain>{{"LevelSetNegative",..},{"LevelSetPositive",..}})
  .key(&FSB::surface_tension, {"Surface_tension","SurfaceTension"}, Literal<Real>().nonnegative())
  .key(member(&FSB::small_cut_aggregation_guards, &Guards::maximum_root_path_length),
       {"Small_cut_aggregation_maximum_root_path_length","SmallCutAggregationMaximumRootPathLength"}, Positive<int>())
  .implies({"Cut_cell_pressure_gradient_penalty"}, &FSB::cut_cell_enabled, true, UnlessExplicitlyFalse)
  .retired({"Cut_cell_velocity_gradient_penalty", …}, "…replaced by small-cut aggregation", AcceptIn::LegacySchema1)
  .rule("SurfaceStress requires RefreshedFrozenQuadrature", …);
```

- **The same schema drives five things:** parsing in the register, `preflight` (replacing 4121–5127), the artifact (replacing 7624–8349), documentation, and the Application parsers (`LevelSetCutConfiguration`, `EquationTranslator`, and the unmerged `FreeSurfaceInputAdapter` from `ae11daea`).
- **Expected sizes:**
  - Register: about 450 lines (factory and BC dispatch) plus about 500 (NS schema), about 350 (free-surface schema), about 150 (two-fluid).
  - About 1,800 lines move to Application/FE: file readers (859), profile and marker geometry (707), node-pressure CSV (170), mesh inference (86).
  - About 2,500 lines are deleted.
- **Behavior to preserve** (refactor plan §7.3), as per-key policies rather than one global rule:
  - first-defined alias wins, except where duplicates are rejected today (schema-version and legacy keys);
  - `parse_bool_relaxed` maps unknown text to false, against the strict lexicon elsewhere;
  - full-consumption numeric parsing;
  - "implies" defaults;
  - legacy-only acceptance of unused sub-options.

## 5. Risks and gates

1. **Bitwise identity**, against the arm64 baseline (the plain e7080344 build on the nine reference decks) and 2- and 4-rank parity. All of the following must stay unchanged:
   - **Field and DOF order:** u, p, f_h, ū, then ALE fields after node constraints (9251, 9263).
   - **Expression trees:** term order and association, including `-p*div(v)` (no conversion to subtraction), the `constant(1.0)` PSPG factors, 0.01, 1e-12 and 1e-30.
   - **Install sequence:** fitted mesh rows, then the main residual, then the work channels, then the PSPG ∇p group on `equations` (in **all** Navier–Stokes runs), then the shape-tangent bilinear.
   - **Constraint order:** vertex pressure, pinned mesh-displacement strong condition, velocity essential, pressure essential, inactive u and p, aggregation of u and p, contact point.
   - **Markers and gauge:** stable marker keys, and the gauge source string used for de-duplication.
   - **Build flags:** `-ffp-contract=off` (D41); JIT FP contraction off (D18).
2. **JIT kernel identity.** `computeKernelCacheKey` mixes the domain, markers, side and the combined IR hash; `KernelIR::stableHash64` is insensitive to the order of commutative operands but not to association. Proposed gate: an ordered **kernel manifest** (operator, fields, domain, marker, side, IR hash, key) that is identical before and after. Tags (`tagged`, `frozen`) must be removed before hashing.
3. **Rank independence.** Moved collectives must run in the same order on every rank, with the MPI_MIN failure agreement and owned-only sums preserved. Preflight collectives currently run inside validation (2985–3022), so moving validation into a schema must keep them. Before unifying `dofHandler().mpiComm()` (3657) with `activeMpiCommunicator()`, prove that they are the same communicator.
4. **Contracts beyond the numbers:**
   - about 100 test assertions on message text;
   - log tokens parsed by scripts: `mesh_field_initialization` (5), `dynamic_contact_operator_angle` (3), `hydrostatic_initialization` and `pressure_gauge_check` (1 each);
   - artifact v4, written by ApplicationDriver and checked by 5 test files;
   - operator names used by `NewtonSolver` and ApplicationDriver.
5. **Process.** D7 pauses refactor packages R3–R12, and M4 runs are in flight on e7080344. Proposed order:
   1. P0: manifest dump tools.
   2. P1: move class (c) to FE with identical trees.
   3. P2: delete class (d); the default path stays bitwise identical, only opt-in modes change.
   4. P3: option schema.
   5. P4: builder, constraint plan and domains, reaching ≤500 lines.
   6. P5: separately-evidenced simplifications of the trees (drop the unit factors, merge integrals), which need a cold JIT cache and a value-identity check.

## 6. Points for the coordinator

1. **Saving this file:** this session was read-only, so the coordinator needs to save this text as `A-formulation-design.md`.
2. **Naming with B:** the FE names (`LevelSetGeometry`, `ConstraintPlan`, `FormulationBuilder`, `FieldBinding`, `SystemCollectives`, `ContactLineGeometryCertificate`, extended `IntegrationDomain`) are proposals because B's files were absent. Please have B confirm or rename them.
3. **Schema location:** I recommend Physics/Core, because modules own their schemas and Application reuses them. B may prefer Application.
4. **Lockstep change with NewtonSolver:** it reads `SVMP_NS_PRESSURE_ROW_CONTRIBUTION_DIAGNOSTIC` and `SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC` and hard-codes the `equations_diagnostic_ns_*` names.
5. **User decisions needed on retiring:** unfitted CurvatureTraction, GeneratedCurvatureTraction, SmoothedIndicator, schema-1 legacy mode, shape tangents, the PSPG and probe hooks, `FreeSurfaceVelocityExtension`, and fitted pointwise curvature. No benchmark generator uses any of them, but older `open_vessel_free_surface` decks do.
6. **No-penetration wall condition:** it remains a check against the user's `Dir` BC (`requireEssential`) until a declared constraint is shown to be bitwise identical.
7. **Scope of PSPG changes:** the separate PSPG ∇p group affects all Navier–Stokes runs, not only free-surface ones. Merging it is a numerical change.
8. **D7:** lifting it, or granting a scoped exception for the Navier–Stokes module on a branch, is a user decision.

### Critical files for implementation

- /home/users/zsexton/svMultiPhysics/Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.cpp
- /home/users/zsexton/svMultiPhysics/Code/Source/solver/Physics/Formulations/NavierStokes/NavierStokesRegister.cpp
- /home/users/zsexton/svMultiPhysics/Code/Source/solver/Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceOptions.h
- /home/users/zsexton/svMultiPhysics/Code/Source/solver/FE/Forms/IntegrationDomain.h (with FormExpr.h and CutCellForms.h)
- /home/users/zsexton/svMultiPhysics/Code/Source/solver/FE/Systems/FormsInstaller.h (and the new FormulationBuilder and ConstraintPlan next to FE/Constraints/SmallCutAggregationConstraint.h)
