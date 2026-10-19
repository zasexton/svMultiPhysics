# Free-surface refactor: FE and Application infrastructure design (2026-10-19)

## 0. Summary

**1. Where the code lives today.** Almost all free-surface orchestration sits in three places, plus scattered physics inside FE.
- **ApplicationDriver.cpp (37,439 lines):**
  - about 18,600 lines are physics-agnostic FE capability: cut-geometry lifecycle, the maintenance pipeline, velocity extension, conservative phase, history repair, transactions;
  - about 9,200 lines are free-surface physics: energy ledger, contact stages, static capillary initializer, free-surface diagnostics;
  - about 4,000 lines are generic run orchestration;
  - about 5,600 lines of free-surface glue sit inside `runTransient` (31153–37118, 5,966 lines in all).
- **NewtonSolver.cpp:** about 8,000 of its 20,473 lines (39%) are Navier–Stokes or free-surface specific: pressure-representability and KKT gates, the static pressure initializer, and 4,746 lines of active-pressure and PSPG diagnostics and matrix-modifying experiments.
- **Physics inside FE:**
  - FESystem.cpp 11,095–17,716 (about 6,600 lines) and FESystem.h 205–980 hold two-fluid, contact and energy records;
  - StandardAssembler.cpp about 100–2,720 holds direct-PSPG experiments.
- **Environment hooks:** 491 unique `SVMP_*` names outside tests (310 excluding FSILS). The driver has 70 (51 change results), NewtonSolver 97 (45), the NS module 34 (27).

**2. Target architecture.** Formulations declare FE services; lifecycle participants carry out the work. There are seven pieces:
- `EvolvingInterface`: a handle with measures, normal and snapshot, in level-set and fitted-boundary representations;
- `CutGeometryService`;
- `LevelSetMaintenancePipeline`, with ordered stages and a typed redistancing trigger;
- `GeneratedStateRegistry` plus `GeneratedStateFixedPoint`: the outer fixed point already in `NewtonSolver::solveStep` 12210–13594, lifted out;
- `StepParticipantList` and `StepTransaction`;
- `ConstraintPlan`, with stages Essential < InactiveSupport < Aggregation < PointLaw, which reproduces today's registration order;
- `FieldExtension`, which takes over the PDE extension and wall masks inherited from the source field's essential constraints.

Newton hooks take the physics out of FE TimeStepping. The elegance test works for two-fluid and for Marangoni with a bulk γ(T). It exposes two FE gaps: surface fields on generated interfaces (TraceFEM), and contact-point laws in 3D and on walls that are not axis-aligned.

**3. Form language.** Typed `Measure` objects extend the R2 `IntegrationDomain.h` and lower to today's `dCutVolume`, `dI`, `dExteriorBoundary` and `dS`, so the IR, JIT keys and results stay the same. The helpers `tangentialProjector` and `conormal` must expand to exactly the trees used today.
- Equivalent algebra is not bitwise. `Vocabulary::surfaceDivergence` builds `div(v) − (∇v n)·n`, a different tree from `(I − n⊗n):∇v`, so it compiles a different kernel.
- Kernels are grouped by (domain, marker, side) and hashed over the ordered list of terms. Term splitting and term order must therefore be preserved.
- A cheap check of kernel identity: zero JIT object-cache misses against a cache warmed by the parent build.

**4. Migration.** 14 steps, ordered by value per risk: S0 evidence (the 9-case set covers none of reconciliation, patch bounds, redistancing, NormalIncrement, strong Ren–E, conservative phase, static capillary, fitted ALE or two-fluid) → S1 Forms measures → S2 ConstraintPlan → S3 Newton hooks → S5 static capillary to Physics → S4 assembler hooks → S6 registries → S7 Physics participants → S9 maintenance → S10 extension → S8 cut service → S11 conservative phase → S12 FreeBoundaryWorkflow, with S13 typed configuration running throughout. Steps S0–S5 and S11 can run now. S6, S8 and S10 wait for the outer-pass branch's branch. No step edits SmallCutAggregationConstraint.cpp, so none conflicts with the MPI-scaling branch (dev/aggregation-ledger-local).

**5. Application.** Target ApplicationDriver.cpp of about 2,500 lines, with `runTransient` at 300 lines or fewer. Two new files: SolverOptionResolution (about 900 lines) and FreeBoundaryBinding (about 500). The free-surface helpers in Application/Core shrink from 12,286 to about 2,800 lines.

---

## 1. Where the free-surface machinery lives today (outside the NS formulation)

### 1.1 ApplicationDriver.cpp map

Classes: **FE** = physics-agnostic FE capability; **App** = generic run orchestration; **FS** = free-surface physics.

| Range | Lines | Content | Class | Destination |
|---|---:|---|---|---|
| 118–2356 | 2,239 | `LevelSetMaintenanceWorkLedger` rows and transactions; accepted-step energy aggregation, energy candidates, channel applicability | FS (about 1,800); ledger mechanics FE | Physics FreeSurfaceEnergyAccount; neutral rows to FE/LevelSet/Maintenance |
| 2358–3712 | 1,355 | Effective-config artifact, MPI gathers, ownership snapshots, memory and OpenMP; XML and env parsers for Newton, TimeLoop and the outer fixed point | App | Application/Core/SolverOptionResolution |
| 3697–4775 | ~1,080 | `LevelSetMaintenanceRequest`/runtime state; canonical maintenance-schedule hashing and collective consensus; `LevelSetAdvectionVelocityRequest` | FE | FE/LevelSet/Maintenance |
| 4776–5180 | 405 | `strongZeroVelocityComponentMask`: reads NS Dirichlet XML to build the extension's wall masks | FS glue | FE FieldExtension `WallTracePolicy` |
| 5181–6120 | 940 | Advection requests from XML (5203, 325 lines), side helpers, wet-volume names, interface-velocity sample cells, cut-adjacent facet sets | FE + App binding | FE/Interfaces queries; App binding |
| 6121–6716 | 596 | Partition-independent cut-topology fingerprints, sync-point policy helpers, topology-change log | FE | CutGeometryService, TimeStepping |
| 6717–7397 | 681 | Wet-volume diagnostics, active-fluid output, coupled level-set residual criteria, coverage logs | FE diagnostics | FE/FreeBoundary/Diagnostics |
| 7398–7989 | 592 | FE-ordered gather/scatter and reductions (~360); accepted pressure-update guard (231, can reject steps) | FE/App; FS | FE/Systems utilities; Physics |
| 7990–8882 | 893 | Dynamic-contact generalized-alpha operator-stage capture and consensus | FS | Physics ContactLineStageRecorder |
| 8883–11060 | 2,178 | Free-surface velocity evaluator, two-fluid stage diagnostics, contact stages, discrete functionals, energies, powers, backward-Euler kinetic work | FS | Physics FreeSurfaceEnergyAccount, TwoFluidStageRecorder |
| 11061–11681 | 621 | Maintenance and mesh-boundary revision fingerprints, functional values, master-bearing reimposition | mixed | FE revisions; Physics values |
| 11682–12668 | 987 | Maintenance and energy logs, accepted free-surface functional records, initial baseline | FS | Physics |
| 12669–12929 | 261 | Implicit-cut backend counts, maintenance targets, open boundaries | FE | FE/LevelSet/Maintenance |
| 12930–14462 | 1,533 | Bound-preserving candidates, volume-correction accounting, accepted wall constraints, wall-aware maintenance context | FE (wall angles are a physics input) | FE/LevelSet/Maintenance |
| 14463–15233 | 771 | First-order generalized-alpha maintenance publication (state and rate repair) | FE | FE/TimeStepping/FieldRepairPublication |
| 15234–16639 | 1,406 | Projection-reinit staging, transport coefficients, kinematic baseline; `applyLevelSetMaintenance` 15556–16639 | FE | Maintenance pipeline |
| 16640–16965 | 326 | Volume-correction free-surface work logs | FS | Physics |
| 16966–18348 | 1,383 | Vertex-field evaluation, geometry mapping, curvature supplemental samples, curvature-projection cache keys; KAG traction binding (17604, about 180 lines, FS) | FE (+FS) | FE/LevelSet curvature service |
| 18349–18945 | 597 | Curvature field projection; mesh-field checkpoint and restore; vertex-field sync | FE | FE/LevelSet, FE/Systems |
| 18946–20292 | 1,347 | `updateLevelSetAdvectionVelocitiesFromState` (18987, 1,022 lines), extension-map registry, artifacts | FE | FE/LevelSet/Extension |
| 20293–23462 | 3,170 | Cut-context lifecycle: `refreshActiveCutIntegrationContextFromSolution` (20529, 1,580 lines), refresh cache, reinstall and resync, attempt-entry retention, ghost-layer audit | FE | FE/Interfaces/CutGeometryService |
| 23463–24399 | 937 | Prescribed-coefficient capture and restore, post-accept topology evidence, collective vector capture and assign | FE transactions | FE/Systems/StepTransaction |
| 24400–28637 | 4,238 | Conservative phase: init, velocity sampling, stage snapshots, artifacts, reconciliation, `applyConservativePhaseCandidates` (27273, 1,300 lines), rollback | FE (+contact-protected nodes) | FE/LevelSet/ConservativePhaseWorkflow |
| 28638–30569 | 1,932 | Static capillary certificate, history publication, `initializeDiscreteStaticCapillaryEquilibrium` (29419, 1,150 lines) | FS | Physics (optional tool under D3) |
| 30571–30810 | 240 | `run`, `runWithParameters` | App | Keep |
| 30811–31152 | 342 | `runSteadyState`, including a duplicate `synchronize_state` at 30925–30972 | App + FS | Keep about 150 |
| 31153–37118 | 5,966 | `runTransient` (see 1.2) | App about 350; FS glue about 5,600 | Keep about 300 |
| 37119–37439 | 321 | `outputResults` | App | Keep |

**Totals:** FE about 18.6k, FS about 9.2k, App about 4.0k, `runTransient` glue about 5.6k.

**Churn:** the driver had 57 commits since 2026-09-20; it is the hot spot.

### 1.2 What `runTransient` does

| Range | Lines | Role |
|---|---:|---|
| 31153–31365 | 213 | Newton and TimeLoop options from XML and env, PTC, adaptivity (App) |
| 31367–31700 | 334 | Maintenance requests and consensus; outer fixed-point policy (31403–31592, 12 plain + 30 max passes, Aitken, cycle exit); semi-implicit reference; initial cut refresh; static capillary; conservative phase |
| 31701–32186 | 486 | Ledgers; phase, fitted-ALE, two-fluid, residual-work and contact declarations and their collective consistency |
| 32261–32526 | 266 | `synchronize_state` (32272–32404) runs G(u) in a fixed order: cut refresh → [constraint pass returns] → curvature projection → velocity extension → semi-implicit reference. Also epoch hooks (32405–32464) and `on_before_physics_solve` |
| 32527–33253 | 727 | `on_nonlinear_done`: generalized-alpha contact, fitted-ALE, two-fluid and residual-work stage captures |
| 33254–34385 | ~1,130 | Pending phase and contact state; `on_before_step_accept` (33433); discarded and ready candidates; `commit_ready` (34012) |
| 34386–36853 | 2,468 | `on_step_accepted`: post-accept maintenance transaction with rollback (34488–35578); accepted free-surface functionals (35579–35826, 35874–36784); extension-map artifacts; output |
| 36854–37047 | 194 | `on_step_rejected`: rollback of cut, phase, contact and ledgers |
| 37048–37118 | 71 | Loop run and report |

### 1.3 NewtonSolver.cpp map (20,473 lines)

| Range | Lines | Content | Class |
|---|---:|---|---|
| 67–1103 | 1,037 | Env readers for diagnostics, free-surface gates and initializer, active-pressure experiments, coupled-BC linear paths | mixed (about 600 NS/FS) |
| 1104–2252 | 1,149 | Tracing, post-tangent analysis, dumps, FD Jacobian check | generic diagnostics |
| 2253–2659 | 407 | Pressure-representability LSQR | FS |
| 2966–3377 | 412 | Constant-pressure cut-volume binding (KKT) | FS |
| 3378–4581 | 1,204 | Halo sync, component norms, field-residual and matrix-support diagnostics | generic |
| 4582–9327 | 4,746 | Pressure-row and direct-PSPG diagnostics; active-pressure support-rank guard and clamp; graph completion (the clamp and graph completion modify the matrix); update-support diagnostics | NS-specific |
| 9328–11376 | 2,049 | Rank-one and reduced-field (0D-coupled BC) updates, FSILS normalization | generic |
| 11379–12209 | 831 | Outer gate, constructors, static-pressure communicator check (about 180 FS) | generic + FS |
| 12210–13594 | 1,385 | `solveStep`: the generated-state outer fixed point (Aitken, epochs, cycle exit, certificate, rollback); deferred free-surface gates at 12739–12900 | FE capability |
| 13595–20471 | 6,877 | `solveStepFrozenExternalState`; see the breakdown below | generic + FS |

Breakdown of `solveStepFrozenExternalState`:
- free-surface diagnostics and gates: 14677–15728 (1,052 lines);
- static compatible-pressure initializer: 16100–16587 (488 lines);
- NS hooks inside the Newton loop: 18580–18610, 19744 and 19819.

### 1.4 Other physics inside FE, and the Application helpers

**Physics inside FE:**
- **FESystem.h 205–~980 and FESystem.cpp 11,095–17,716:** fitted-ALE stage measurements, mesh normal and tangential boundary histories with a fluid consumer binding, two-fluid stage telemetry, free-surface residual work and functionals.
- **StandardAssembler.cpp about 100–2,720:** 20 `SVMP_FE_CUT_VOLUME_DIRECT_PSPG_*` names. The `TOPOLOGY_POLICY` family modifies element matrices.
- **Free-surface physics types already in FE:**
  - Interfaces/FreeSurfaceGeometrySnapshot.h 273–680: Young, contact, energy, dissipation and power types;
  - Interfaces/IncompressibleTwoFluidDiagnostics.* (1,533 lines);
  - LevelSet/LevelSetStaticCapillaryEquilibrium.* (1,740 lines);
  - the Young weights computed inside LevelSet/LevelSetCurvatureProjection.

**Application/Core free-surface helpers (12,286 lines):**
- LevelSetPdeVelocityExtension: 2,629;
- LevelSetVelocityExtensionMap: 3,547;
- maintenance configuration, history and consensus: about 1,826;
- LevelSetCutConfiguration: 1,297;
- FreeSurfaceEnergyLedger: 1,233;
- LevelSetCurvatureSamples: 723;
- FreeSurfaceSemiImplicitReference: 385;
- ActiveDomainOutput: 374;
- NearestPointIndex: 192.

### 1.5 SVMP_* environment hooks

| File | Unique | Result-changing | Performance (bitwise by design) | Checks | Observation |
|---|---:|---:|---:|---:|---:|
| Application/Core/ApplicationDriver.cpp | 70 | 51 | 10 | 0 | 8 (+1 guard: `GENERATED_STATE_MAX_DISCONTINUITY_RESTARTS` throws) |
| FE/TimeStepping/NewtonSolver.cpp | 97 | 45 | 1 | 9 | 42 |
| Physics/.../IncompressibleNavierStokesVMSModule.cpp | 34 | 27 | 0 | 0 | 7 |
| FE/Assembly/StandardAssembler.cpp | 40 | ~14 | ~3 | ~3 | ~20 |
| FE/Systems/FESystem.cpp | 6 | 4 | 0 | 0 | 2 |
| FE/Constraints/SmallCutAggregationConstraint.cpp | 7 | 4 | 1 | 0 | 2 |
| Application/Core/LevelSetPdeVelocityExtension.cpp | 9 | 1 (fault injection) | 3 | 0 | 5 |
| FE/TimeStepping/TimeLoop.cpp | 3 | 2 | 0 | 0 | 1 |

**Result-changing names in the driver:**
- Outer fixed point (10): `GENERATED_STATE_OUTER_{FIXED_POINT, MAX_ITERATIONS, CONSTRAINT_CYCLE_EXIT_PASS}`, `GENERATED_STATE_OUTER_DYNAMIC_RELAXATION{,_START_PASS,_INITIAL_FACTOR,_MINIMUM_FACTOR,_MAXIMUM_FACTOR,_DENOMINATOR_RELATIVE_TOLERANCE}`, `CUT_REFRESH_PER_STEP_ONLY`.
- Newton (18): `NEWTON_LINE_SEARCH*` (6), `NEWTON_{ABS,REL}_TOLERANCE`, `NEWTON_PTC*` (8), `NEWTON_ACCEPT_INEXACT_LINEAR`, `JACOBIAN_REBUILD_PERIOD`.
- Time loop (12): `TIMELOOP_*` (9), `GENERALIZED_ALPHA_PREDICTOR{,_FIELDS}`, `GENERALIZED_ALPHA_PDE_UDOT_INIT`.
- Trial policy (3): `SYNC_LINE_SEARCH_TRIALS`, `CURVATURE_REUSE_CACHE_ON_TRIAL_FAILURE`, `NO_MPC_STATE_DISTRIBUTE`.
- Free-surface method (7): `REDISTANCING_TRIGGER_CELLS`, `PDE_EXTENSION_FACTORIZATION` (round-off), `ACTIVE_PRESSURE_UPDATE_{GUARD, REJECT_ON_TRIGGER, FAIL_ON_TRIGGER, THRESHOLD_PA, FIELD}`.
- Fault injection (1): `AUDIT_WITHHOLD_GHOST_CUT_LAYERS`.

**Result-changing names in NewtonSolver:**
- History and constraint distribution (2): `DISTRIBUTE_CONSTRAINTS_INTO_HISTORY`, `NO_MPC_STATE_DISTRIBUTE`.
- Free-surface gates and initializer (6, aliases included).
- Active-pressure guard, clamp and graph completion (24).
- Coupled-BC linear paths (13).

**Result-changing names in the NS module (27):** VMS on and off, 18 PSPG scale, form and support names, 4 free-surface probe names, the inactive-velocity clamp, and the shape tangents.

### 1.6 What the NS formulation calls into today

- **Measures:** `dCutVolume` through `integrateOnActiveVolume` (6109), `dI` through `integrateOnFreeSurface` (5556), `dInterfaceBoundary`, which is `dI` on a stable intersection marker, and `dExteriorBoundary(generatedActiveSubset)`.
- **The rule normal:** `FormExpr::normal()`, flipped for the positive side (1939).
- **`FESystem` calls:**
  - `registerGeneratedEmbeddedInterfaceMarker` (5 calls);
  - `addCutIntegrationContextUpdateCallback` (3);
  - `addSystemConstraint`, in a deliberate order at 10096–10232: inactive u and p, then aggregation of u and p, then the Ren–E point law;
  - `declareFreeSurface*` (energy) and `declareTwoFluid*`;
  - `declareMeshNormal/TangentialBoundary*`.
- **Energy channels:** `applyFreeSurfaceBoundary` (6801–7305) returns them through 7 optional mutable out-pointers.

---

## 2. Target FE architecture

**Principles:**
1. Formulations declare. Services own numerics. Lifecycle participants are composed by a generic registry.
2. Every service is reached through a handle, so the NS module never sees caches, transactions or MPI consensus.
3. Every move preserves call order, so the default path stays bitwise identical.

### 2.1 EvolvingInterface (FE/Interfaces/EvolvingInterface.h)

**Responsibilities:**
- interface identity: the field, isovalue, primary side, retention and walls;
- measures and geometric terminals;
- immutable snapshot access;
- two representations: a level-set cut and a fitted boundary (ALE), which removes the `ale_enabled` branching inside `integrateOnFreeSurface`.

**Moves here:** marker computation and registration from the NS module (2188–2400), and the request structs from Application/Core/LevelSetCutConfiguration.h.

```cpp
namespace svmp::FE::interfaces {
enum class InterfaceRepresentation : std::uint8_t { LevelSetCut, BoundaryFitted };
struct EvolvingInterfaceDeclaration {
  std::string name{"free_surface"};
  FieldId level_set{INVALID_FIELD_ID};          // LevelSetCut
  int boundary_marker{-1};                      // BoundaryFitted
  Real isovalue{0.0};
  forms::CutVolumeSide primary_side{forms::CutVolumeSide::Negative};
  RetainedSides retention{RetainedSides::Primary};   // Primary | Both
  std::vector<int> wall_markers{};              // generates interface/boundary intersections
  level_set::LevelSetGeneratedInterfaceOptions geometry{};
};
class EvolvingInterface {
public:
  virtual ~EvolvingInterface() = default;
  [[nodiscard]] virtual InterfaceRepresentation representation() const noexcept = 0;
  [[nodiscard]] virtual forms::Measure dOmega(forms::CutVolumeSide) const = 0;   // dCutVolume(m,s) | dx()
  [[nodiscard]] virtual forms::Measure dGamma() const = 0;                       // dI(m) | ds(m)
  [[nodiscard]] virtual forms::Measure dContactLine(int wall) const = 0;         // dI(intersection m); registers marker
  [[nodiscard]] virtual forms::Measure dWetted(int wall) const = 0;              // dExteriorBoundary(generatedActiveSubset)
  [[nodiscard]] virtual forms::Measure dCutFacets(forms::CutVolumeSide) const = 0; // dS(set, side)
  [[nodiscard]] virtual forms::FormExpr normal() const = 0;                      // outward from primary side
  virtual void moveWith(const KinematicDeclaration&) = 0;                        // level-set transport | mesh kinematics
  virtual void onGeometryRefresh(systems::CutIntegrationContextUpdateCallback) = 0;
  [[nodiscard]] virtual std::shared_ptr<const InterfaceGeometrySnapshot> snapshot() const = 0;
};
EvolvingInterface& declareEvolvingInterface(systems::FESystem&, EvolvingInterfaceDeclaration);
}
```

**Formulation:** `auto& G = interfaces::declareEvolvingInterface(system, fs.interface);`

### 2.2 CutGeometryService (FE/Interfaces/CutGeometryService.{h,cpp})

**Responsibilities:** one authoritative cut context and snapshot per interface, containing:
- the refresh, keyed on the vector signature;
- incremental rebuilds;
- reinstall and resync on restore;
- retention at attempt entry;
- a partition-independent topology fingerprint;
- the constraint-state rebuild when the geometry revision changes.

**Moves here:** driver 20293–23462, 6121–6716, and the query parts of 5181–6120 (about 4,700 lines).

```cpp
class CutGeometryService final : public timestepping::IGeneratedStateParticipant {
public:
  CutGeometryService(systems::FESystem&, std::vector<EvolvingInterface*>);
  GeneratedStateRole role() const noexcept override { return GeneratedStateRole::Geometry; }
  void synchronize(const systems::SystemStateView&, SyncPoint) override;
  std::optional<std::uint64_t> epoch() const override;   // cut-topology fingerprint
  CutRefreshReport refresh(const systems::SystemStateView&, std::string_view provenance);
};
```

### 2.3 Level-set evolution: transport and maintenance (FE/LevelSet/Maintenance/)

**Responsibilities:**
- the transport installation (the existing `installLevelSetTransport`);
- an ordered maintenance pipeline: kinematic reconciliation → patch bounds → redistancing with its trigger → reinitialization → volume correction (today's order in 15556–16639);
- publication that is consistent with the time scheme;
- neutral work rows, to which Physics energy observers attach.

**Moves here:** driver 3697–4775, 12669–16639, part of 18349–18945, plus Application LevelSetMaintenance{History,TransactionConsensus}. That is about 6,100 lines.

```cpp
namespace svmp::FE::level_set {
class ILevelSetMaintenanceStage {
public:
  virtual ~ILevelSetMaintenanceStage() = default;
  [[nodiscard]] virtual MaintenanceSubstage substage() const noexcept = 0; // Limiting, Reinitialization, ...
  [[nodiscard]] virtual bool due(const MaintenanceContext&, MaintenanceRuntime&) const = 0;
  virtual StageOutcome apply(const MaintenanceContext&, MaintenanceRuntime&, std::vector<Real>& phi) const = 0; // collective, no publication
};
struct RedistancingTrigger { Real interface_motion_cells{1.0}; };   // replaces SVMP_REDISTANCING_TRIGGER_CELLS
class LevelSetMaintenancePipeline final : public timestepping::ITransactionalParticipant {
public:
  LevelSetMaintenancePipeline& then(std::unique_ptr<ILevelSetMaintenanceStage>);
  MaintenanceCandidate stage(const MaintenanceContext&);
  void publish(MaintenanceCandidate&&, timestepping::TimeHistory&, const FieldRepairPublicationPolicy&);
  void rollback(MaintenanceCandidate&&);
  void addObserver(MaintenanceObserver);   // Physics energy rows
};
namespace stages { /* kinematicReconciliation(o), signDefinitePatchBounds(o),
  interfacePreservingRedistancing(o, RedistancingTrigger), projectionReinitialization(o, cadence), volumeCorrection(o) */ }
}
```

**Formulation, or the translator:** `G.moveWith({.velocity = w, .transport = opts, .maintenance = plan});`

### 2.4 Generated state and the outer fixed point (FE/TimeStepping/GeneratedState.{h,cpp})

**Responsibilities:**
- replace the driver lambdas `synchronize_state` (32272–32404, duplicated in steady at 30925–30972) and the epoch hooks (32405–32464) with an ordered registry;
- lift `solveStep` 12210–13594 into `GeneratedStateFixedPoint`, which takes typed options resolved once.

```cpp
enum class GeneratedStateRole : std::uint8_t { Geometry, DerivedField, Extension, Reference };
class IGeneratedStateParticipant {
public:
  virtual ~IGeneratedStateParticipant() = default;
  [[nodiscard]] virtual std::string_view name() const = 0;
  [[nodiscard]] virtual GeneratedStateRole role() const noexcept = 0;
  virtual void synchronize(const systems::SystemStateView&, SyncPoint) = 0;   // reproducible, reversible, collective
  [[nodiscard]] virtual std::optional<std::uint64_t> epoch() const { return std::nullopt; }
  virtual void acknowledgeEpoch(SyncPoint) {}
};
class GeneratedStateRegistry {
public:
  void add(std::shared_ptr<IGeneratedStateParticipant>);       // stable order: role, then registration
  void bind(NewtonOptions&, const GeneratedStatePolicy&) const;  // sync points, discontinuity, epoch revisit
};
struct GeneratedStateFixedPointOptions { int max_passes{30}; int plain_passes{12}; AitkenOptions aitken{};
  int epoch_restarts{-1}; int constraint_cycle_exit_pass{13}; bool residual_first_on_predicted_certificate{true}; };
class GeneratedStateFixedPoint {
public:
  GeneratedStateFixedPoint(GeneratedStateFixedPointOptions, const GeneratedStateRegistry&);
  NewtonReport solve(const NewtonSolver& inner, systems::TransientSystem&, backends::LinearSolver&,
                     double t, TimeHistory&, NewtonWorkspace&) const;
};
```

Registering any participant enables the outer fixed point; the policy comes from resolved configuration. FE/Coupling's `PartitionedCouplingPlan` (`RelaxedFixedPoint`, Aitken) could eventually execute this loop. That is not recommended now.

### 2.5 Step participants and transactions (FE/TimeStepping/StepParticipants.h, FE/Systems/StepTransaction)

```cpp
class IStepParticipant {
public:
  virtual ~IStepParticipant() = default;
  virtual void onStepStart(const TimeHistory&) {}
  virtual void afterNonlinear(const TimeHistory&, const NewtonReport&) {}
  virtual std::optional<StepRejectReason> beforeAccept(TimeHistory&, const NewtonReport&) { return {}; }
  virtual void candidateReady(TimeHistory&) {}  virtual void candidateDiscarded(TimeHistory&) {}
  virtual void commitReady(TimeHistory&) {}     virtual void accepted(TimeHistory&) {}
  virtual void rejected(const TimeHistory&, StepRejectReason, const NewtonReport&) {}
};
class ITransactionalParticipant : public IStepParticipant {
public:  // checkpoint / stage / validate / commit / rollback, collective consensus
  virtual void checkpoint() = 0; virtual void stage() = 0; virtual bool validate() = 0;
  virtual void commit() = 0; virtual void rollback() = 0;
};
class StepParticipantList { public: void add(std::shared_ptr<IStepParticipant>, int order);
  [[nodiscard]] TimeLoopCallbacks callbacks(TimeLoopCallbacks base = {}) const; };  // reverse order on discard/rollback
```

`MovingDomainOrchestrator::makeBeforePhysicsSolveCallback` is the existing precedent for an FE service that produces TimeLoop callbacks.

### 2.6 ConstraintPlan (FE/Constraints/ConstraintPlan.h)

```cpp
enum class ConstraintStage : std::uint8_t { Essential, InactiveSupport, Aggregation, PointLaw };
class ConstraintPlan { public: void declare(ConstraintStage, std::unique_ptr<ISystemConstraint>); };
std::unique_ptr<ISystemConstraint> inactiveSide(FieldId, const interfaces::EvolvingInterface&, forms::CutVolumeSide active);
std::unique_ptr<ISystemConstraint> aggregate(FieldId, const interfaces::EvolvingInterface&, forms::CutVolumeSide, SmallCutAggregationGuardOptions = {});
std::unique_ptr<ISystemConstraint> contactPointLaw(FieldId u, const interfaces::EvolvingInterface&, int wall, std::function<Real(Real cos_theta)> speed);
```

Constraints install by stage, then in declaration order. That reproduces the NS sequence at 10096–10232. The constructors of `SmallCutAggregationConstraint` and `ContactPointVelocityConstraint` do not change.

### 2.7 FieldExtension (FE/LevelSet/Extension/)

**Moves here:** driver 4776–5180 and 18946–20292; Application PDE extension, extension map and NearestPointIndex. That is about 8,100 lines.

```cpp
enum class FieldExtensionMethod : std::uint8_t { Harmonic, NormalLeastSquares, WallCompatibleNormal, NearestInterfacePoint };
enum class FieldExtensionCoupling : std::uint8_t { Lagged, FrozenRows };     // "prescribed" | "monolithic"
enum class WallTracePolicy : std::uint8_t { InheritHomogeneousEssential, None, Explicit };
struct FieldExtensionDeclaration {
  const interfaces::EvolvingInterface* interface; FieldId source; std::string target_name;
  forms::CutVolumeSide known_side; FieldExtensionMethod method{FieldExtensionMethod::Harmonic};
  FieldExtensionCoupling coupling{FieldExtensionCoupling::FrozenRows};
  WallTracePolicy walls{WallTracePolicy::InheritHomogeneousEssential};
  ExtensionFactorization factorization{ExtensionFactorization::LuColamd};  // replaces SVMP_PDE_EXTENSION_FACTORIZATION
  int band_layers{0}; };
FieldId extendField(systems::FESystem&, FieldExtensionDeclaration);   // registers a role=Extension participant
```

### 2.8 Newton hooks (FE/TimeStepping/NewtonHooks.h)

```cpp
struct NewtonHook { std::string name; bool modifies_system{false};
  std::function<void(const systems::FESystem&, backends::GenericMatrix&, backends::GenericVector&,
                     std::span<const GlobalIndex>, const NewtonIterationInfo&)> after_jacobian;
  std::function<void(const systems::FESystem&, const backends::GenericVector&, const NewtonIterationInfo&)> after_update;
  std::function<std::optional<NewtonVerdict>(const systems::FESystem&, const NewtonReport&)> acceptance_gate;
  std::function<void(systems::FESystem&, backends::GenericVector& u)> initial_guess; };
// NewtonOptions::hooks : std::vector<NewtonHook>   (empty by default)
```

**Moves to Physics** (Physics/Formulations/NavierStokes/Solver/): the pressure-representability and KKT gates, the static initializer, and the active-pressure and PSPG diagnostics. Alternatively, retire the experiments (see section 6).

### 2.9 Elegance test

- **Two-fluid.** One interface with retention Both; `dOmega(Negative)` and `dOmega(Positive)`; `dGamma` with jump and avg; aggregation per side. No FE change is needed.
- **Marangoni with γ(T) from a bulk field.** The variational term γ(I − n⊗n):∇v already contains ∇_Γγ. Only the Physics validation ("literal constant only") and the energy account change.
- **Surfactant-driven Marangoni, or an elastic membrane.** These need a surface field on Γ_h. **This is an FE gap.** Proposed: `declareSurfaceField(G, name, order)`, a TraceFEM bulk P1 field restricted to cut cells by an `outsideBand` constraint, with normal-gradient stabilization on cut cells.
- **Moving-contact membrane.** It also needs `ContactPointVelocityConstraint` generalized beyond 2D Triangle3 and axis-aligned walls (see the cpp at 178–280). **This is the second gap.**
- **Stefan problems.** FieldExtension needs a known set made of interface cells only, a small option.

---

## 3. Form-language extensions

### 3.1 Syntax, with the tracker §6.1 terms

```cpp
auto& G = interfaces::declareEvolvingInterface(system, fs.interface);
const auto n = G.normal();                               // rule normal, outward from liquid
const auto P = tangentialProjector(n);                   // = identity(dim) - outer(n,n)  (same tree as NS 7000)
R += bulk(u, p, v, q) * G.dOmega(liquid);
R += (p_ext * dot(n, v) + gamma * inner(P, grad(v))) * G.dGamma();      // γ∫_Γ(I−n⊗n):∇v (one term, as today)
for (const auto& cl : fs.contact_lines) {
  const auto m = conormal(n, cl.wall_normal);            // = safeNormalize(n - dot(n,nw)*nw)  (same tree as 5833)
  R += (xi(cl) * dot(u, m) - gamma * cos_e(cl)) * dot(v, m) * G.dContactLine(cl.wall);   // −γ cos θe ∫_CL v·m (+ Ren–E)
  R += mu / cl.slip * dot(tangentialProjection(u, cl.wall_normal),
                          tangentialProjection(v, cl.wall_normal)) * G.dWetted(cl.wall); // Navier slip
}
G.moveWith({.velocity = level_set::extendField(system, {&G, u_id, "LevelSetAdvectionVelocity", liquid})}); // kinematic
auto& C = system.constraintPlan();
C.declare(InactiveSupport, constraints::inactiveSide(p_id, G, liquid));
C.declare(Aggregation, constraints::aggregate(u_id, G, liquid, fs.guards));
C.declare(Aggregation, constraints::aggregate(p_id, G, liquid, fs.guards));
```

**`Measure` type.** `forms::Measure` (`cell`, `cutVolume`, `interface`, `exterior`, `boundary`, `facets`) has `operator*(FormExpr, Measure)`. It extends the R2 `IntegrationDomain.h`, which no consumer uses yet. Today `tangentialProjection(u, n) = u − inner(u, n) n`, the same tree as the NS form `u − dot(u, nw) nw`, because both `dot` and `inner` call `a.inner(b)`.

**Kinematic condition, fitted case.** `G.moveWith` installs a generic trace-Nitsche term on the Physics relation `dot(u − meshVelocity(), n)`. The 540 lines of MeshNitsche (7354–7896) become an FE `traceNitsche` plus a one-line relation.

### 3.2 Compiler and IR implications, and JIT kernel identity

1. **Sugar-only lowering.** Measures lower to the existing `dCutVolume`, `dI`, `dExteriorBoundary` and `dS`. The IR, the `IntegralDomain`, the markers (stable hashes) and the JIT keys are all unchanged.
2. **Kernel grouping.** Kernels are grouped by (domain, marker, side). The cache key hashes the ordered list of term IR hashes (JITCompiler.cpp 395–500). Therefore:
   - keep one integral where today has one (p_ext + γ is a single `dI` term);
   - keep the left-fold order of `momentum_form = momentum_form + term`;
   - do not swap trees for algebraically equal ones.
3. **Named energy channels.** A `ResidualBuilder::add(channel, term)` folds terms left in insertion order and records channels in a side table, not as IR nodes. It replaces the 7 out-pointers with the hash unchanged.
4. **Later, not bitwise:**
   - New terminals (`conormal()` and `wallNormal()` from the intersection rule's `boundary_normal` and `tangent`) need FormExprType, KernelIR and LLVMGen changes plus a bump of `kKernelArgsABILayoutRevision`. That re-keys every kernel: one cold compile, and the pinned D17 caches must be regenerated.
   - An explicit codimension-two `IntegralDomain` touches about 22 dispatch sites (FormCompiler 5, JITKernelWrapper 6, FormKernels 3, LLVMGen 2, JITCompiler 2, and others) and bumps `kKernelCacheKeySchemaVersion` from 10 to 11.

**Kernel-identity gate.** Warm a JIT object cache with the parent build. The candidate run, with `SVMP_JIT_CACHE_DIAGNOSTICS=1`, must report `object_cache_misses=0`. Add unit tests asserting `toString()` equality between the helper expansions and the current NS trees.

---

## 4. Staged migration plan

**Verification ladder:**
- L1: unit tests and IR/kernel identity;
- L2: the 9-case arm64 serial set (cmp_runs.py against runs/e7080344);
- L3: refset-ext (new decks, see S0);
- L4: CTest;
- L5: the exhaustive two-rank MPI matrix;
- L6: 2- and 4-rank parity.

| Order | Step | Line moves | Effort | Risk | Checks | Alongside performance branches |
|---|---|---|---|---|---|---|
| 1 | **S0** Evidence. Add refset-ext: 7 short decks for the sessile protocol (reconciliation, bounds, redistancing), static drop with NormalIncrement, strong Ren–E, capillary rise L10, fitted capillary wave, conservative phase, two-fluid hydrostatic. Also the kernel-identity gate and IR-tree tests. | 0 | 1–2 d | none | baseline at e7080344 | yes |
| 2 | **S1** Forms `Measure`, `tangentialProjector`, `conormal`, `ResidualBuilder` | +300 FE | 1–2 d | low | L1 | yes |
| 3 | **S2** ConstraintPlan; NS 10096–10232 switches to `declare()` | +250 FE / −60 NS | 1–2 d | low | L2, L4, L5 | yes (aggregation .cpp untouched) |
| 4 | **S3** Newton hooks; FS/NS code out of NewtonSolver.cpp | −8,000 FE → Physics or retired | 3–5 d | low–medium | L2, L4, L5 | yes (check with the outer-pass branch (dev/outer-pass-reuse) if it touches `solveStep`) |
| 5 | **S5** Static capillary initializer to Physics (driver 28638–30569; FE LevelSetStaticCapillaryEquilibrium.*) | −1,930 / −1,740 | 2–3 d | low | L4 | yes |
| 6 | **S4** Assembler post-processor hook; direct-PSPG experiments out of StandardAssembler | −2,600 FE | 2–3 d | low | L2, L4 | yes |
| 7 | **S6** GeneratedStateRegistry and StepParticipantList; adapters replace both `synchronize_state` lambdas and the epoch hooks | +450 FE / −250 | 2–3 d | medium | L2, L3, L5, L6 | **after the outer-pass branch (dev/outer-pass-reuse) merges** |
| 8 | **S7** Physics participants: energy account, contact stage, two-fluid stage; FESystem physics records to Physics | −9,000 driver, −7,400 FE, −1,233 App | 1.5–2 wk | medium | byte-identical energy logs, L2, L4, L5 | mostly (conflicts possible in `on_step_accepted`) |
| 9 | **S9** Maintenance pipeline and FieldRepairPublication | −5,600 driver, −480 App | 1–1.5 wk | medium | L3 (sessile, capillary rise), L4, consensus MPI | partly |
| 10 | **S10** FieldExtension | −1,750 driver, −6,370 App | 1 wk | medium | L2 (all 9 cases use pde_harmonic monolithic), extension MPI parity, mumps_serial | **after the outer-pass branch (dev/outer-pass-reuse) (PDE cache)** |
| 11 | **S8** CutGeometryService | −4,700 driver | 1 wk | medium–high | L2, L3, L5, L6, incremental self-check | **after the outer-pass branch (dev/outer-pass-reuse)** |
| 12 | **S11** Conservative-phase workflow | −4,240 driver | 1 wk | medium | L3, L4 | yes |
| 13 | **S12** FreeBoundaryWorkflow (FE/FreeBoundary) and EvolvingInterface lifecycle; `runTransient` reduced to composition | −4,500 driver; +1,200 FE, +500 App | 1–2 wk | medium | full ladder | after S6–S11 |
| cont. | **S13** Typed configuration with provenance for the result-changing env hooks; one legacy env adapter | −1,000 driver | ongoing | low | default-path bitwise | yes |

**Total effort:** about 9–11 developer-weeks. S0–S5 (about 2–3 weeks) can start now. A can write the concise formulation against S1/S2 plus an `EvolvingInterface` facade backed by today's markers, before any lifecycle code moves.

**Expected sizes after migration:**
- ApplicationDriver.cpp: about 2,500;
- NewtonSolver.cpp: about 11,200, plus GeneratedStateFixedPoint.cpp at about 1,250;
- FESystem.cpp: about 26,300 (FESystem.h about 2,850);
- StandardAssembler.cpp: about 21,500;
- Application/Core free-surface helpers: about 2,800.

---

## 5. Application layer

**ApplicationDriver keeps:**
- `run` and `runWithParameters`;
- the SimulationBuilder call;
- the steady/transient choice;
- construction of the TimeLoop and Newton solver;
- binding `system.lifecycle()` (the generated-state registry and step participants) into `NewtonOptions` and `TimeLoopCallbacks`;
- output (VTK, PVD, ActiveDomainOutput), the run summary and the effective-configuration artifact.

**Target `runTransient` (300 lines or fewer):**

```cpp
auto opts = resolveTimeLoopOptions(params, *sim.time_history);          // SolverOptionResolution
auto& life = sim.fe_system->lifecycle();
life.generatedState().bind(opts.newton, resolveGeneratedStatePolicy(params));
auto cb = life.steps().callbacks();
appendOutput(cb, [&](auto& h){ outputResults(sim, params, h.stepIndex(), h.time(), pvd); });
life.initialize(*sim.time_history);   // initial geometry, conservative phase, optional static capillary
const auto rep = TimeLoop(opts).run(transient, *sim.backend, *sim.linear_solver, *sim.time_history, cb);
```

**What moves:**
- FE capability → FE folders (section 2), composed by **FE/FreeBoundary/FreeBoundaryWorkflow**. This is the physics-agnostic free-boundary component. It sits in FE so that FE stays usable without the Application.
- Free-surface physics → Physics/Formulations/NavierStokes/FreeSurface/ participants and hooks.
- XML binding → Application/Core/FreeBoundaryBinding.cpp (about 500 lines). It reuses the translators.

**Lint target:** no free-surface, level-set, capillary or contact identifiers in ApplicationDriver.cpp outside output field names.

---

## 6. Points for the coordinator

1. **No A-files existed when I finished.** I propose these names for A to adopt: `EvolvingInterface`, `dOmega`/`dGamma`/`dContactLine`/`dWetted`/`dCutFacets`, `tangentialProjector`, `conormal`, `ResidualBuilder`, `ConstraintPlan` with `aggregate`/`inactiveSide`/`contactPointLaw`, `extendField`, `G.moveWith`.
2. **Anticipated disagreements with A:**
   - Who declares the kinematic condition. I propose the FE level-set evolution, declared through `G.moveWith(extendField(u))`, with the NS module only binding it.
   - Whether the energy ledger belongs in Application or Physics. I propose a Physics participant.
3. **Bitwise is a stronger constraint than algebraic equality.** The concise form must preserve tree shapes and term grouping. Decide whether to allow one later "canonicalization" step validated against tolerances.
4. **The 9-case reference set covers none of the protocol-default opt-ins.** S0 must come first.
5. **D7 still pauses R3–R12.** Starting this plan needs a user decision.
6. **Retire or move.** The active-pressure clamp and graph completion, the direct-PSPG local Schur and edge balance, and the pressure-reference and tangential probes are experiments. Under P1 and the earlier decisions, retiring them is cheaper than moving them (about 7,000 lines).
7. **One step is not bitwise by construction.** `WallTracePolicy::InheritHomogeneousEssential` must reproduce `strongZeroVelocityComponentMask`, which reads the XML `effective_direction`. Verify it on the 2-rank matrix.
8. **Concurrency.** S6, S8 and S10 touch the outer-pass branch's caches (driver 20293–23462, 32272–32404, 17309–18133, the PDE cache). No step edits SmallCutAggregationConstraint.cpp.
9. **Env count.** There are now 491 unique `SVMP_*` names, against about 174 in the tracker's audit.

### Critical files for implementation
- /home/users/zsexton/svMultiPhysics/Code/Source/solver/Application/Core/ApplicationDriver.cpp
- /home/users/zsexton/svMultiPhysics/Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp
- /home/users/zsexton/svMultiPhysics/Code/Source/solver/FE/Systems/FESystem.h
- /home/users/zsexton/svMultiPhysics/Code/Source/solver/FE/Forms/IntegrationDomain.h
- /home/users/zsexton/svMultiPhysics/Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.cpp



## 7. Alignment with A-formulation-design.md

I adopt A's form-language and handle names wherever the two documents cover the same thing. The B-only services keep their names.

### 7.1 Name mapping

| Concept | A's name (adopted) | B's earlier name | What sits behind it (B) |
|---|---|---|---|
| Level-set interface handle | `ls::LevelSetGeometry::bind(system, decl)`, `.side(s)` → `ls::CutSide` (FE/LevelSet/LevelSetGeometry.{h,cpp}) | `interfaces::EvolvingInterface` | `ls::LevelSetGeometryService`: B's CutGeometryService renamed. It owns the cut-context lifecycle (driver 20293–23462, 6121–6716) and is a generated-state participant |
| Volume | `expr.dx(VolumeDomain)`; `VolumeDomain::full()`, `L.volume()` | `dOmega(side)`, `forms::Measure` | extends `VolumeIntegrationDomain` (R2); lowers to `dx` or `dCutVolume` |
| Interface | `expr.dI(L.interface())` → `InterfaceDomain` | `dGamma()` | `dI(marker)` |
| Contact line | `.dL(CL)` / `dInterfaceBoundary(expr, CL)`; `L.contactLine(wall, n_w)` → `ContactLineDomain`; `CL.wallNormal()`, `CL.conormal()` | `dContactLine`, `conormal(n, n_w)` | `dI(stable intersection marker)`. `conormal()` must expand exactly to `safeNormalize(n − (n·n_w)n_w)` |
| Wetted wall | `.ds(L.wetted(wall))` → `BoundaryDomain` | `dWetted` | `dExteriorBoundary(generatedActiveSubset)` plus marker registration |
| Cut facets | `.dS(L.cutFacets())` | `dCutFacets` | `cutAdjacentFacetIntegral(m, side)` |
| Normal | `L.outwardNormal()` | `G.normal()` | rule normal, flipped for the positive side |
| Curvature | `L.recoveredCurvature(policy)` | curvature service | driver 16966–18945, moved to FE/LevelSet as a DerivedField participant |
| Kinematics | phase 1: `L.requireMaterialInterface(u_id)` (a check only); later `G.declareKinematics({.velocity, .extension, .maintenance})` | `G.moveWith` | payload: `ls::extendField` (FieldExtensionDeclaration) and `LevelSetMaintenancePipeline` |
| Constraints | `cs::ConstraintPlan` method API | `declare(stage, …)` | see 7.2 |
| Builder and channels | `fs::FormulationBuilder`, `tagged(Channel, e)` | `ResidualBuilder` | see 7.3 |
| Lagged copy | `lagged(u, fs::Refresh::GeneratedState)` | Reference-role participant | a GeneratedStateRegistry participant (S6). Until S6 lands it wraps the existing FreeSurfaceSemiImplicitReference |
| Collectives | `fs::SystemCollectives` | "FE/Systems utilities" | also absorbs the driver's gather and reduction helpers (7398–7989) |
| Contact certificate | `interfaces::ContactLineGeometryCertificate` | contact-point gap | also the place to lift the 2D, axis-aligned limit of ContactPointVelocityConstraint |
| Field binding | `fs::FieldBinding` | — | — |
| Diagnostics | `fs::DiagnosticRegistry` | `NewtonHook` | Physics registers Newton hooks and diagnostic operators through the registry; NewtonSolver looks operators up by name |

B-only services keep their names:
- GeneratedStateRegistry and GeneratedStateFixedPoint;
- StepParticipantList, ITransactionalParticipant and StepTransaction;
- LevelSetMaintenancePipeline and its stages;
- ls::extendField / FieldExtension, and FieldRepairPublication;
- FreeBoundaryWorkflow;
- SolverOptionResolution and FreeBoundaryBinding.

### 7.2 Constraint phases (revised to A's complete order)

The order is VertexValues → Essential → InactiveSupport → Aggregation → PointLaw, as follows:
1. **VertexValues:** vertex pressure.
2. **Essential**, in today's sequence: the pinned mesh-displacement strong condition, velocity essential, pressure essential.
3. **InactiveSupport:** u, then p.
4. **Aggregation:** u, then p.
5. **PointLaw:** the Ren–E contact point.

Within a phase, constraints keep declaration order. `commit()` issues one `addSystemConstraint` per entry, so the FESystem order is unchanged. Gauge anchors go to the GaugeRegistry, not to a constraint phase.

```cpp
namespace svmp::FE::constraints {
class ConstraintPlan {
public:
  explicit ConstraintPlan(systems::FESystem&);
  ConstraintPlan& vertexValues(FieldId, std::span<const VertexValue>, VertexValueSupport = {});
  ConstraintPlan& essentials(systems::BoundaryConditionManager&, systems::FESystem&, forms::FormExpr& M,
                             const forms::FormExpr& u, const forms::FormExpr& v, FieldId u_id);
  ConstraintPlan& requireEssential(const forms::FormExpr& trace, int boundary_marker);   // phase 1: check only
  ConstraintPlan& inactiveSide(std::initializer_list<FieldId>, const level_set::CutSide&, Real value = 0);
  ConstraintPlan& aggregate(std::initializer_list<FieldId>, const level_set::CutSide&, SmallCutAggregationGuardOptions = {});
  ConstraintPlan& contactPointVelocity(FieldId u, const level_set::ContactLineDomain&, std::function<Real(Real cos_theta)>);
  ConstraintPlan& anchorGauge(FieldId p, const forms::InterfaceDomain&);
  void commit();
};
}
```

The constructors of SmallCutAggregationConstraint and ContactPointVelocityConstraint do not change, so S2 still does not touch the file the MPI-scaling branch (dev/aggregation-ledger-local) is editing.

### 7.3 Form language

**A's §2.3 sketch is the agreed target statement.** It replaces B §3.1. The rules in B §3.2 still hold, with three changes:
- **Kernel identity.** Adopt A's ordered kernel manifest (operator, fields, domain, marker, side, IR hash, key) and constraint manifest as S0 tools. B's "zero JIT object-cache misses against a parent-warmed cache" remains the end-to-end confirmation.
- **Hash behaviour.** `KernelIR::stableHash64` ignores the order of commutative operands but not association. Swapping operands therefore keeps both the key and the bits, because IEEE addition and multiplication are exactly commutative. Re-associating changes both. A's list of trees to preserve (`-p*div(v)`, the `constant(1.0)` PSPG factors, the separate PSPG ∇p group, contact terms before the traction, SurfaceStress as one `.dI` of a sum) is the binding list.
- **`tagged()`.** It must be stripped in one place, `FormulationBuilder::install()`, before any expression reaches FormCompiler, SymbolicDifferentiation, AffineAnalysis or KernelIR. If any pass could see a tagged node, about 10 passes would have to learn to ignore it. The builder keeps the channel map.

### 7.4 Agreements and remaining disagreements (additions to B §6)

1. **Option schema location (disagreement).** A recommends Physics/Core. I recommend putting the generic header-only `OptionSchema<T>` template in FE/Core. FE numerical option structs need schemas too (LevelSetTransportOptions, LevelSetGeneratedInterfaceOptions, SmallCutAggregationGuardOptions, GeneratedStateFixedPointOptions, the extension declaration), and FE must not depend on Physics (refactor plan §2.2 rule 1). Physics/Core would then hold the NS, free-surface and two-fluid schemas. Application's SolverOptionResolution would use the same template, including the single legacy `SVMP_*` adapter. You or the user should decide.
2. **Representation-agnostic handle.** I withdraw the single fitted-or-unfitted EvolvingInterface for phase 1. The fitted path stays BoundaryDomain plus ALEBinding, as A proposes. Revisit only if a second physics needs both representations behind one statement.
3. **Kinematics ownership (agreed).** In phase 1 the level-set equation owns transport, extension and maintenance; the formulation only checks with `requireMaterialInterface`. `declareKinematics` arrives with S9, S10 and S12.
4. **Register items A assigned to B:**
   - The `Temporal_and_spatial_values` reader (NavierStokesRegister.cpp 704–1562) and the node-pressure CSV (1649–1818) go to Application/Translators, because they are file I/O.
   - The marker geometry and perimeter gathers (in 2077–2783) go to FE/Geometry `BoundaryGeometryQueries`, built on SystemCollectives.
   - The parabolic inlet profile stays a Physics BC factory that uses that query.
5. **Lockstep.** B's S3 (Newton hooks) and A's removal of the operator-name structs (header 63–158) and the diagnostic env readers must land as one change, with NewtonSolver resolving operators through DiagnosticRegistry.
6. **Communicator.** A flagged `dofHandler().mpiComm()` at NS 3657, which differs from `activeMpiCommunicator()` used everywhere else. SystemCollectives takes one accessor. Before switching, add a 2-rank test that asserts the two compare MPI_IDENT or MPI_CONGRUENT.
7. **One retirement decision.** Put A's class-d list (A §6.5) and B's retire option (the NewtonSolver active-pressure experiments and StandardAssembler's direct-PSPG experiments, together about 7,000 lines) to the user as a single list.
8. **Merged order:**
   1. S0 (A's P0 manifests plus B's refset-ext decks).
   2. S1 domain handles (A constructs 1–6).
   3. S2 ConstraintPlan.
   4. A's P1 moves of class-c code inside the NS module (FieldBinding, SystemCollectives, ContactLineGeometryCertificate, marker plumbing into LevelSetGeometry), in parallel with B's S3, S4 and S5.
   5. B's S6 registries. These make `lagged()` exact.
   6. A's FormulationBuilder with `tagged`, reaching the ≤500-line module.
   7. A's P3 OptionSchema, after decision 1.
   8. B's S7–S12.
   9. A/B P5: tree simplifications, kept separate with tolerance-based evidence and a cold JIT cache.

   The constraints on running alongside the outer-pass and MPI-scaling branches (B §4) do not change.
