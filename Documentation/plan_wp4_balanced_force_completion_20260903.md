# WP-4 Balanced-Force Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the balanced-force half of WP-4 by making the production total-energy-gradient traction quadrature-exact for affine P1 three-dimensional cuts, enforcing that contract, validating its discrete energy adjoint and restoring force, and freezing corrected qualification evidence.

**Architecture:** Keep the unfiltered `KinematicAreaGradientTraction` selected by the audit and retain the fixed-volume minimizer of the same discrete surface-plus-wall energy. Extend only planar `LinearCorner` polygon integration to degree two with an ordered triangle fan, then make the Application binding and runtime rules fail closed unless requested and achieved interface order are both at least two. Validate sampled curved geometry by convergence/GCI and minimized discrete geometry by the existing algebraic equilibrium gates; do not add pressure enrichment or project capillary force into the pressure range.

**Tech Stack:** C++20, GoogleTest, CMake 3.31, MPI/OpenMPI, Python 3.12 qualification runners, Slurm on Sherlock.

**Spec:** `Documentation/free_surface_boundary_unfitted_audit_20260720.md`, especially FSR-03, WP-4, and Q2.

## Global Constraints

- Work only in the isolated scratch worktree `/scratch/users/zsexton/wp4-application-regression-fixes-20260902`; never modify the shared checkout.
- Preserve order-zero and order-one planar-interface behavior byte-for-byte in numerical semantics; planar polygon requests above order two fail closed.
- The selected production method remains unfiltered `KinematicAreaGradientTraction` plus fixed-volume minimization of the same discrete surface-plus-wall energy.
- Do not add pressure enrichment and do not project capillary force into the discrete pressure-gradient range.
- Every production change follows a witnessed red/green test cycle, and each coherent reviewed commit is pushed immediately.
- Every commit has author and committer `Zachary Sexton <zsexton@stanford.edu>`.
- Before each commit and push, scan every added/staged byte and the proposed message for the repository-prohibited vocabulary; a nonempty scan blocks the commit.
- Put disposable builds, scripts, logs, caches, and runner output below `/scratch/users/zsexton`.
- At most four owned `amarsden` nodes and 40 GiB total may run concurrently. Every submitted job declares `--mail-user=zsexton@stanford.edu` and `--mail-type=BEGIN,END,FAIL` in the submitted script from its first submission.
- Preserve the inherited module environment for launched work; do not invoke `bash -lc` inside `srun`.
- Touch only jobs recorded in this plan's scratch ledger.
- Do not launch the qualification runner until the source is a clean frozen commit, all LFS objects are hydrated, caches/worktrees are fresh, and matrix plus runner hashes are final.
- Do not check WP-4, Q2, FSR-03, or FSR-04 from prerequisite or development evidence alone.

---

## Remaining WP4 critical path and delegation

Coordination addition dated 2026-09-05. This maps the remaining work to existing designs; it does not replace Tasks 1-4, dispatch workers, reserve compute, or grant qualification credit. The prescribed-angle stream below is separately required for full WP4/Q2 closure, beyond this plan's balanced-force goal. Initial states reflect the 18:42 UTC continuation checkpoint; the coordinator must reconcile current ownership, source/cache guards and accepted evidence before assigning anything.

The balanced-force critical path is **A -> B/C -> E -> F**: B and C can have independent preparation, but production derivative use requires their joint acceptance. D can progress independently where ownership permits; full WP4/Q2 closure additionally requires D and the complete audit matrix.

| Package | Initial state | Ownership boundary for the next packet | Deliverable and acceptance dependency |
| --- | --- | --- | --- |
| A. Arithmetic kernel and producer adapters | Kernel active; adapters blocked on accepted kernel | Existing kernel owner; then one owner for the cell/boundary producer files and their geometry tests | Finish the current report and scoped review, centrally integrate the kernel, then execute the existing adapter brief. Require real repeat-margin controls, unchanged geometry outputs and preserved unrelated failure observations. Do not replan or restart this work. |
| B. Remaining geometry and scalar certificates | Conditional designs available; implementation blocked on an explicit scoped packet and required A interfaces | FE interface producers and functional evaluation; serialize overlaps with A | Convert accepted normal/incidence results into tested bounds for the actual emitted support, normals, moments, weights and scalar quantities. Assess the auxiliary gradient separately where it affects area. Each slice must identify the derivative-eligibility requirement it establishes; root/dedup success alone is insufficient. |
| C. Authoritative derivative and kinematic binding | Ready for packet preparation; production use blocked on required B evidence | FE projection/lifecycle interfaces and the current Application candidate owner | Reuse the accepted binding design. Validate current coefficients, source, field mapping, cache reuse and collective order; reject stale/mismatched linkage. Demonstrate surface, wall and volume variation with the actual production mapping. Independent API/precondition work may overlap B only on disjoint, reserved files. |
| D. Prescribed-angle completion | Gap map and scaling design available; implementation requires the remaining explicit contracts | FE reinitialization/options plus current Application maintenance owners | Resolve only unanswered stage/anchor and wall-strip contracts. Follow the existing sequence: scale invariance and fixed point; independent wall schedule; stage consistency; curved 3D shared-DOF strip; production work/convergence evidence. Reconcile refactoring moves before naming writable files. |
| E. Corrected production pilot | Blocked on the relevant B/C repair and its regression evidence | Production evaluator/minimizer integration and a separately assigned immutable pilot lane | Establish the production-case milestone below before expanding balanced-force qualification. Preserve earlier failed runs and unchanged physical/minimizer gates. A diagnostic or an unavailable derivative is not a passing pilot. |
| F. Integrated validation and qualification | Regression preparation ready; qualification blocked on accepted pilot and a new freeze | Coordinator-owned source/matrix freeze; separately reserved FE, Physics and Application lanes; read-only evidence reviewer | Justified integrated regressions may overlap E once their frozen inputs, binaries and reservations permit. Reuse accepted runner work and correct remaining contract mismatches before the full immutable qualification matrix. Validate and archive raw evidence centrally. Balanced-force acceptance closes only its documented scope; WP4/Q2 and FSR-04 additionally require prescribed-angle and full audit acceptance. |

### Assignment and parallel-work rules

- Use at most three workers, including existing workers. Reuse an implementer for supported fixes; allow one writer per overlapping source area. A scientific worker resolves one genuinely unanswered premise on immutable inputs. A reviewer inspects the exact frozen patch and evidence; test execution requires an explicit justified lane. Workers do not spawn helpers.
- Every dispatch uses the packet in `/scratch/users/zsexton/wp4-delegation-briefs-20260905-Z6OoSG/README.md` and the matching role brief: precise deliverable/non-goals; source and participating-input identities including untracked inputs; exact writable/protected paths and named owner; accepted evidence and dependencies; commands, expected outcomes and timeouts; source/cache and scheduler reservations; allowed actions; report path and next acceptance check. A roadmap row alone is not an executable assignment.
- Name live owners and actual reservations in the existing `delegation-checkpoint.md`, not a second status ledger here. Use **ready**, **blocked**, **active** and **accepted**, with the blocking dependency or accepted evidence named. Preparation readiness does not authorize dependent production use.
- Parallel preparation is useful only with independent deliverables. Reviews may overlap other work using immutable review inputs; numerical lanes need separate caches and frozen participating sources. A separate worktree alone does not remove shared repository/cache conflicts. Preserve all current resource limits, inherited modules, protected files and submission-time mail requirements; no new allocation is implied by this roadmap.
- The coordinator alone handles integration, shared audit edits, staging, commits and pushes under existing identity/content rules. Resolve current paths against `free_surface_architecture_refactoring_plan_20260904.md`; historical source locations in design reports are not current write reservations.

### Required production-case milestone

Before expanding the physical campaign, a relevant tested repair must demonstrate that the actual production evaluator's energy/volume variations and returned model agree under the declared derivative and kinematic contract, with current candidate/source binding and rollback intact. Reuse the captured failure and its mapped coefficients as the regression target; do not repeat an unchanged diagnostic to recreate accepted evidence. Any directional comparison must distinguish a genuine same-branch sample from a topology/canonicalization change and account for its declared error bounds.

Then require the previously failing production minimization case to reach its unchanged convergence/publication gates. Preserve its input identity and report any scientifically necessary change explicitly before comparison. Helper tests, conservative rejection, a higher iteration cap, altered tolerances or a conditional theorem do not substitute for this outcome. At each accepted prerequisite, record which remaining failure or eligibility condition it removes; if a new obstruction appears, assign that specific question rather than restarting the completed investigation.

### Execution priorities for timely WP4 completion

User-requested coordination addition dated 2026-09-05. Apply these priorities to the remaining work without restarting active assignments, reopening accepted evidence, or replacing Tasks 1-4. They grant no additional source ownership, compute allocation, job authority or qualification credit. The existing physical method, protected files, resource ceilings and acceptance gates remain binding.

1. **Organize the next work around one corrected production case.** Work backward from the required production-case milestone above. Every new prerequisite packet must name the specific failure or eligibility gap it removes, the testable contract it delivers, and the next consumer that can use it. Prioritize the smallest complete geometry/scalar/derivative/kinematic repair that can make the captured failing case pass its unchanged convergence and publication gates before expanding the physical matrix. A correct rejection of the troublesome state is useful protection, but is not that repair; identify any further represented-geometry contract it leaves necessary. Do not rerun the physical case until the relevant repair and its focused regression evidence justify it. Full three-dimensional and prescribed-angle obligations remain required for closure.

2. **Advance genuinely independent dependencies concurrently.** Prepare C's accepted source/candidate binding and API preconditions while B's geometry/scalar work proceeds, using immutable inputs or disjoint explicitly reserved source and cache paths. Progress D's unanswered stage/anchor or wall-strip contracts where ownership permits. Each assignment must deliver a concrete result and follow the existing full task packet; preparation does not authorize dependent production use. Reconcile the current architecture owners first, keep one writer per overlapping source area, respect the three-worker limit, and leave a slot idle when no useful independent deliverable exists.

3. **Use stable integration and verification windows.** Coordinate incoming architecture changes with the main source owner and integrate only after every affected wrapper has completed and released its guards. Preserve the other contributor's history and notes. Keep each verification input set stable through its run and scoped review; do not merge changes merely because a numerical command has exited. Tracked documentation edits can also change a guarded status listing, so schedule them in the same safe windows. After integration, rerun only checks invalidated by actual changed dependencies, not accepted suites on unchanged inputs.

4. **Allocate compute to the measured ready workload.** When source/input identities and reservations permit, run justified FE, Physics and Application verification lanes concurrently with separate caches and explicit CPU, memory and walltime limits. Use current scheduler reservations and the existing total budget, not historical node counts or worker-slot availability. Increase compile parallelism only where memory and measured build time support it; extra nodes cannot remove a serial implementation or proof dependency. Preserve inherited modules and submission-time mail settings. Qualification still requires the accepted production pilot, clean frozen source, fresh qualification caches/worktrees and finalized matrix/input hashes.

5. **Measure elapsed time and minimize repeated coordination work.** At completed milestones, record time spent on implementation, scientific checks/review, compilation, queueing and numerical execution in the existing checkpoint/ledger. Use available timestamps; label estimates and overlapping intervals instead of summing concurrent work as wall time. State the current bottleneck, next exact acceptance result and why the next assignment addresses it. Reuse reviewed wrappers and accepted evidence rather than constructing a new harness or proof audit for each continuation; record the concrete changed input, failure or missing evidence that justifies any additional verification.

Keep continuation records concise and outside guarded inputs. Detailed logs and immutable source/evidence identities belong in the assigned artifacts; neither repeated summaries nor more completed prerequisite reports substitute for movement toward the production milestone.

At the next safe handoff, reconcile these priorities with live workers and guards, finish the active bounded deliverable, and continue the highest-leverage authorized work. Do not stop merely after writing another plan, reduce the remaining audit matrix, loosen a threshold, or equate prerequisite completion with WP4 qualification.

### Reuse and continuation index

Local coordination artifacts are under `.superpowers/sdd/plan_wp4_balanced_force_completion_20260903/` in the existing worktree; they are not published qualification evidence. Reuse:

- A: `task-4-producer-arithmetic-implementation-plan.md`, the current kernel assignment/report, and `task-4-producer-arithmetic-adapters-brief.md`.
- B: `task-4-producer-margin-design-report.md`, its accepted review, and `task-4-affine-normal-premise-report.md`. Their conditional results still require the named executable premises; do not reprove unchanged results.
- C: `task-4-authoritative-derivative-contract-design-report.md`, `task-4-authoritative-derivative-binding-seam-report.md` and `task-4-authoritative-derivative-eligibility-seam-report.md`.
- D: `task-4-prescribed-angle-gap-map.md` and `task-4-prescribed-scaling-design-report.md`; retain their explicit unresolved contracts and resolve moved owners before implementation.
- E/F: the audit's accepted terminal capture/replay/probe records, preserved rejected qualification, and Task 4's corrected-runner, freeze and acceptance steps. Repeat verification only for changed inputs, an actual failure, missing evidence or a named unresolved concern.

Maintain the existing concise checkpoint outside guarded paths after each phase: owner, exact source/evidence identity, live job/process route, active source/cache guards, blocking dependency and next action. Resume from that record without duplicate submissions. Keep detailed logs in their existing evidence directories and the public audit limited to accepted results and honest open gates.

## File Structure

- `Code/Source/solver/FE/Interfaces/LevelSetInterfaceDomain.h`: generate degree-two planar polygon rules while retaining stored centroid rules for orders zero and one.
- `Code/Source/solver/FE/LevelSet/LevelSetImplicitCutQuadratureBackend.cpp`: advertise the new three-dimensional `LinearCorner` interface-order ceiling.
- `Code/Source/solver/FE/Tests/Unit/Geometry/test_LevelSetInterfaceDomain.cpp`: prove triangular and quadrilateral polygon moments, weights, normals, legacy order one, and rejection above order two.
- `Code/Source/solver/FE/Tests/Unit/LevelSet/test_LevelSetInterfaceLifecycle.cpp`: prove backend capability and achieved-order publication for generated tetrahedral cuts.
- `Code/Source/solver/Application/Core/ApplicationDriver.cpp`: enforce requested and achieved order at total-energy-traction binding/runtime boundaries and bind the order into curvature-cache identity.
- `Code/Source/solver/Application/Tests/Unit/test_ApplicationDriverLevelSetWorkflows.cpp`: cover serial admission failures and production discrete-adjoint/minimized-equilibrium behavior.
- `Code/Source/solver/Application/Tests/Unit/test_ApplicationDriverLevelSetWorkflowsMPI.cpp`: cover distributed admission, achieved-order consensus, and minimized-equilibrium parity.
- `Code/Source/solver/Physics/Tests/Unit/test_MovingDomainPhysics.cpp`: contract the assembled traction with a nonconstant P1 velocity and prove the expected energy-variation/restoring sign.
- `tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py`: emit the required interface order for the selected traction and report the balance/restoring observables.
- `tests/cases/fluid/free_surface_wp4_balanced_capillary_matrix_v3.json`: corrected immutable matrix with fixed horizons and bounded, staged resource groups.
- `tests/cases/fluid/run_free_surface_wp4_balanced_capillary_matrix_v3.py`: fail-closed V3 expansion, execution, GCI, minimized-state, MPI, provenance, and disposition gates.
- `Documentation/free_surface_boundary_unfitted_audit_20260720.md`: record accepted evidence and change boxes only after the complete closure criteria pass.

### Task 1: Degree-Two Planar Polygon Interface Quadrature

**Files:**

- Modify: `Code/Source/solver/FE/Interfaces/LevelSetInterfaceDomain.h`
- Modify: `Code/Source/solver/FE/LevelSet/LevelSetImplicitCutQuadratureBackend.cpp`
- Test: `Code/Source/solver/FE/Tests/Unit/Geometry/test_LevelSetInterfaceDomain.cpp`
- Test: `Code/Source/solver/FE/Tests/Unit/LevelSet/test_LevelSetInterfaceLifecycle.cpp`

**Interfaces:**

- Consumes: ordered `CutInterfaceFragment::vertices`, each carrying physical and parent coordinates; `CutInterfaceDomainRequest::resolvedInterfaceQuadratureOrder()` and achieved-order metadata.
- Produces: a `CutQuadratureRule` with `3 * (vertices.size() - 2)` positive points for an order-two polygon, `exact_polynomial_order == 2`, matching policy/provenance order, and unchanged stored rules for orders zero and one.

- [ ] **Step 1: Add hand-derived failing moment tests**

Add a three-dimensional monomial helper that multiplies literal coordinate powers at rule points. Generate two unit-tetrahedron cuts through `appendLinearLevelSetCellCut3D`: signed values `{-1, 1, 1, 1}` for the triangle and `{-1, -1, 1, 1}` for the quadrilateral. Request interface order two and assert:

```cpp
ASSERT_EQ(triangle_rule.points.size(), 3u);
EXPECT_NEAR(integrateWeight(triangle_rule), std::sqrt(3.0) / 8.0, 1.0e-14);
EXPECT_NEAR(integrateMonomial3D(triangle_rule, 2, 0, 0),
            std::sqrt(3.0) / 192.0, 2.0e-15);
EXPECT_NEAR(integrateMonomial3D(triangle_rule, 1, 1, 0),
            std::sqrt(3.0) / 384.0, 2.0e-15);

ASSERT_EQ(quad_rule.points.size(), 6u);
EXPECT_NEAR(integrateWeight(quad_rule), std::sqrt(2.0) / 4.0, 1.0e-14);
EXPECT_NEAR(integrateMonomial3D(quad_rule, 2, 0, 0),
            std::sqrt(2.0) / 48.0, 2.0e-15);
EXPECT_NEAR(integrateMonomial3D(quad_rule, 1, 1, 0),
            std::sqrt(2.0) / 64.0, 2.0e-15);
EXPECT_NEAR(integrateMonomial3D(quad_rule, 0, 1, 1),
            std::sqrt(2.0) / 96.0, 2.0e-15);
```

For every point require positive weight and the generated unit normal. Rebuild the same quadrilateral with interface order one and require its single stored centroid point and original weight. Request order three from the order-two polygon and require `std::invalid_argument` even if achieved metadata is two.

- [ ] **Step 2: Witness the expected red result**

Configure or incrementally build a scratch FE cache against this worktree, then run:

```bash
/scratch/users/zsexton/wp4-planar-polygon-order2-dev/test_fe_geometry \
  --gtest_filter='LevelSetInterfaceDomain.PlanarPolygon*'
```

Expected: the order-two polygon request is rejected by the current order-one ceiling; order-one legacy assertions remain green.

- [ ] **Step 3: Implement the minimal triangle-fan rule**

In `CutInterfaceFragment`, classify a polygon with at least three ordered vertices and positive measure as supporting order two. For effective order two, triangulate `(v[0], v[i], v[i+1])`; on every nondegenerate triangle emit the standard three barycentric points

```cpp
constexpr std::array<std::array<Real, 3>, 3> barycentric{{
    {{Real{2.0} / Real{3.0}, Real{1.0} / Real{6.0}, Real{1.0} / Real{6.0}}},
    {{Real{1.0} / Real{6.0}, Real{2.0} / Real{3.0}, Real{1.0} / Real{6.0}}},
    {{Real{1.0} / Real{6.0}, Real{1.0} / Real{6.0}, Real{2.0} / Real{3.0}}},
}};
```

Interpolate both `point` and `parent_coordinate`, copy the fragment normal, and assign one third of that triangle's physical area. Normalize all emitted weights once by `fragment.measure / fan_measure` so the rule exactly retains the authoritative measure. Reject fewer than three vertices, a nonpositive/nonfinite fan measure, requested order above two, or a point count inconsistent with the fan. Keep the existing stored-point path and policy names unchanged for effective orders zero and one; use one new deterministic policy name only for the order-two fan.

- [ ] **Step 4: Advertise and test the actual backend ceiling**

Change only the `LinearCorner` three-dimensional maximum reported interface order from one to two. Add a lifecycle case requesting interface order two on a `Tetra4` cut and require requested, possible, achieved, verified, rule exactness, rule provenance, and total point count to report two consistently. Leave `SayeHyperrectangle` and P1 fallback behavior in other high-order backends unchanged.

- [ ] **Step 5: Run focused and adjacent FE tests**

Run the new moment/capability filters, then the complete geometry and lifecycle executables from the same binary set:

```bash
test_fe_geometry --gtest_filter='LevelSetInterfaceDomain.*'
test_fe_levelset --gtest_filter='LevelSetInterfaceLifecycle.*'
```

Expected: zero failures; only predeclared scheduled skips may remain in the full lifecycle executable.

- [ ] **Step 6: Review, commit, and push Task 1**

Inspect the entire diff and run the prohibited-vocabulary scan. Commit with the required author/committer identity and message `Add quadratic planar interface quadrature`, verify the recorded identity and tree, then push the current branch.

### Task 2: Total-Energy Traction Quadrature Admission

**Files:**

- Modify: `Code/Source/solver/Application/Core/ApplicationDriver.cpp`
- Test: `Code/Source/solver/Application/Tests/Unit/test_ApplicationDriverLevelSetWorkflows.cpp`
- Test: `Code/Source/solver/Application/Tests/Unit/test_ApplicationDriverLevelSetWorkflowsMPI.cpp`
- Modify: `tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py`
- Test: existing Python unit module covering the physical runner.

**Interfaces:**

- Consumes: `ActiveCutVolumeRequest::{quadrature_order,interface_quadrature_order}`, generated interface-rule provenance, and the total-energy traction declaration matched by `bindKinematicAreaGradientTractionMaintenance`.
- Produces: an early requested-order gate plus a runtime achieved-order gate; successful bindings guarantee every active production interface rule is degree two or higher.

- [ ] **Step 1: Add failing serial admission tests**

Extend the existing binding test with independent copies of its valid request:

```cpp
auto default_order = request;
default_order.volume_cut_request->interface_quadrature_order.reset();
default_order.volume_cut_request->quadrature_order.reset();
EXPECT_THROW(bind_one(default_order), std::runtime_error);

auto linear_order = request;
linear_order.volume_cut_request->interface_quadrature_order = 1;
EXPECT_THROW(bind_one(linear_order), std::runtime_error);

auto quadratic_order = request;
quadratic_order.volume_cut_request->interface_quadrature_order = 2;
EXPECT_NO_THROW(bind_one(quadratic_order));
```

Add a pure internal-rule validator test with a degree-one rule, inconsistent requested/achieved metadata, and a valid degree-two rule. Each malformed case must throw before projection; the valid rule must pass.

- [ ] **Step 2: Witness the serial red result**

Rebuild `test_application` and run only the binding/validator filters. Expected: default and order-one requests are currently admitted, so their assertions fail.

- [ ] **Step 3: Implement requested and achieved gates**

Resolve requested interface order as explicit interface order, else generic order, else the production default of one. In the binder, reject a matched total-energy-traction request below two before mutating the derived Young-wall or liquid-side options. At the curvature-projection context boundary, require each active rule for the matched marker to satisfy all of:

```cpp
rule.exact_polynomial_order >= 2
rule.policy.polynomial_order >= 2
rule.provenance.requested_quadrature_order >= 2
rule.provenance.achieved_quadrature_order >= 2
```

An empty generated marker or mixed-order rule set fails closed. Include all three optional order fields, including presence bits, in `curvatureProjectionCutRequestSignature` so cached projection identity cannot alias different order contracts.

- [ ] **Step 4: Update every selected-traction fixture**

Set `Interface_quadrature_order` to two in the serial and MPI production XML fixtures and set `.interface_quadrature_order = 2` in direct `ActiveCutVolumeRequest` construction. In the physical runner, emit that XML field automatically whenever positive surface tension selects `kinematic_area_gradient_traction`; reject an explicit conflicting lower value if a caller-facing override exists.

- [ ] **Step 5: Add and witness the MPI red/green path**

The two-rank collective-dispatch fixture must first fail without an explicit order and then pass with order two. Assert rank-consensus on the requested/achieved rule orders. Rebuild once, run the focused filter on exactly two ranks, and retain per-rank machine-readable records.

- [ ] **Step 6: Run focused Application and Python regressions**

Run the binding tests, static flat total-energy test, collective two-rank dispatch, physical-runner configuration tests, and any direct source-pin/envelope tests affected by the emitted XML. Require zero failures and no unexpected skip.

- [ ] **Step 7: Review, commit, and push Task 2**

Inspect the complete Task 2 range, scan staged bytes/message, commit as `Require quadratic total-energy traction rules` with the required identity, verify, and push.

### Task 3: Production Energy Adjoint, Minimized Equilibria, and Restoring Force

**Files:**

- Test: `Code/Source/solver/Physics/Tests/Unit/test_MovingDomainPhysics.cpp`
- Test: `Code/Source/solver/Application/Tests/Unit/test_ApplicationDriverLevelSetWorkflows.cpp`
- Test: `Code/Source/solver/Application/Tests/Unit/test_ApplicationDriverLevelSetWorkflowsMPI.cpp`
- Modify only if a witnessed test exposes a defect: the narrow owning production source.

**Interfaces:**

- Consumes: unfiltered kinematic-area-gradient derivative/curvature arrays, order-two generated rules, production traction assembly, fixed-volume static minimizer, and pressure/physical-potential certificate.
- Produces: direct evidence that production traction is the adjoint of the same discrete geometric energy, minimized curved states meet algebraic gates, distributed results agree, and a volume-preserving perturbation receives restoring work.

- [ ] **Step 1: Write a failing three-dimensional production adjoint test**

Use one affine `Tetra4` cut with signed nodal values `{-1, -1, 1, 1}`, a nonconstant nodal velocity, and the unfiltered recovered curvature. Define nodal level-set direction `delta_phi_i = -u_i dot grad(phi)` and compare the derivative-array contraction against the assembled capillary residual contracted with the identical velocity coefficients. Require the order-two quadrilateral rule and a scaled tolerance of `1e-12`; repeat with order one as a negative control and require a nonzero mismatch. This test catches replacing the quadratic fan by the centroid rule.

- [ ] **Step 2: Witness red, then make only evidence-driven fixes**

Run the focused Physics test before changing production assembly. If Task 1 and Task 2 are sufficient, the positive path may already pass; in that case preserve the independently failing order-one negative control as the witnessed regression mechanism. If another production defect appears, reduce it to a separate failing assertion before changing its owner.

- [ ] **Step 3: Add minimized curved production certificates**

Create compact two-dimensional circle/sessile and three-dimensional sphere/sessile fixtures that invoke the existing fixed-volume minimizer with `KinematicAreaGradientTraction`, interface order two, zero filter, and the same surface-plus-wall energy. For minimized states require the existing `1e-8` algebraic gates on projected gradient, volume, pressure-range distance, conservative balance, and production residual. For sampled analytic controls record errors but require convergence/GCI rather than exact equilibrium.

- [ ] **Step 4: Add five-angle and geometry transformations**

For sessile cases cover `30`, `60`, `90`, `120`, and `150` degrees. Across the focused set cover both liquid signs, translated subcell offsets, rotated walls, and the supported two- and three-dimensional simplex meshes. Every result labels initialization as sampled or minimized and reports pressure jump, pressure-space distance, force residual, parasitic capillary number, kinetic energy, volume, base radius, apex height, and angle.

- [ ] **Step 5: Add MPI parity**

Run one nontrivial minimized sessile case and one closed-surface case under two different two-rank ownership/numbering layouts. Require identical accepted/rejected disposition, topology/source revisions, requested/achieved order, certificate flags, and observables within their declared floating-point reductions.

- [ ] **Step 6: Add a moving restoring-force test**

Apply a small volume-orthogonal perturbation to a minimized state, evaluate both signs of the perturbation, and require positive second energy difference plus capillary work opposing the displacement. Advance a fixed short physical horizon, increasing the step count as the time step is refined; keep the step count equal among the positive, negative, and unperturbed runs at each refinement. Require initial acceleration/velocity to have the restoring sign and require the unperturbed minimized control to remain under the algebraic/parasitic gates.

- [ ] **Step 7: Run focused FE, Physics, and Application suites**

Run the complete kinematic-area-gradient curvature group, the new Physics adjoint/restoring group, serial minimized Application group, and two-rank MPI parity group. Retain exact commands, binary hashes, per-rank records, and all failure controls.

- [ ] **Step 8: Review, commit, and push Task 3**

Review the entire range against the audit, scan staged content/message, commit as `Validate balanced capillary energy response`, verify identity and evidence, and push.

### Task 4: Corrected V3 Matrix, Frozen Qualification, and Audit Closure

**Files:**

- Create: `tests/cases/fluid/free_surface_wp4_balanced_capillary_matrix_v3.json`
- Create: `tests/cases/fluid/run_free_surface_wp4_balanced_capillary_matrix_v3.py`
- Test: corresponding V3 runner unit module beside the existing matrix-runner tests.
- Modify: `Documentation/free_surface_boundary_unfitted_audit_20260720.md`
- Create after acceptance: one checksum-bound directory under `Documentation/qualification_logs/`.

**Interfaces:**

- Consumes: Tasks 1-3, the physical solver runner, the audit's predeclared candidate gates, and clean source/build provenance.
- Produces: immutable serial/MPI qualification evidence with explicit dispositions and, only if every WP-4 exit passes, formal FSR-03/WP-4 status changes attributable to balanced force.

- [ ] **Step 1: Write failing V3 schema/expansion tests**

Require nonzero finite-difference components; KAG interface order two in every selected-traction command; positive-scaling lanes that actually execute prescribed maintenance; fixed final physical horizon under time refinement; independently varied bulk-redistance cadence; bounded three-dimensional sizes derived from the 10 GiB/node envelope; exact test-set/resource union; and rejection of any unrecognized or missing axis.

- [ ] **Step 2: Implement V3 by correcting V2, not mutating it**

Retain V2 as historical evidence. Build V3 with separate focused algebra, sampled-convergence, minimized-equilibrium, restoring-motion, and MPI groups. Use at least `R/h = 8, 16, 32`, add 64 only for a declared nonasymptotic triplet, and compute observed order plus GCI from raw results. Freeze the finest-level gates before execution: pressure jump below one percent, angle below one degree, base radius/height below one percent, and parasitic capillary number below `1e-6`.

- [ ] **Step 3: Complete runner self-tests and a dry expansion**

Run all V3 unit tests, expand the full case set without numerical execution, and independently recompute counts, resources, arguments, fixed horizons, hashes, and expected artifacts. Any mismatch blocks freezing.

- [ ] **Step 4: Freeze a clean source commit and fresh inputs**

Commit and push the reviewed V3 inputs with the required identity. Create a detached, fully hydrated scratch clone at that exact commit, prove a clean tree and zero LFS pointers, make fresh build/cache/output directories, and record source tree, matrix, runner, physical-runner, compiler, MPI, and dependency hashes before submission.

- [ ] **Step 5: Submit bounded independent builds and qualification groups**

Use no more than four concurrent one-node jobs totaling 40 GiB on `amarsden`, with mail on begin/end/fail in every original script. Build FE, Physics, and Application independently, then launch only hash-bound groups with dependencies. Record every owned job in the scratch ledger before/at submission; leave all unrelated jobs untouched.

- [ ] **Step 6: Independently validate immutable evidence**

Rehash the exact file set; reconstruct test unions, numeric gates, GCI, minimized algebraic gates, MPI properties, dispositions, resources, execution routes, and scheduler envelope from raw artifacts. Preserve failed/inconclusive cases and never replace them without recording the original outcome.

- [ ] **Step 7: Archive, review, and push evidence**

Copy only byte-preserved accepted evidence into a new qualification-log directory, validate exact bytes and absence of symbolic links, scan all archive bytes, commit as `Record WP4 balanced-force qualification`, verify identity/ancestry, and push.

- [ ] **Step 8: Reconcile formal status conservatively**

If and only if all balanced-force requirements pass, update FSR-03 and the balanced-force portion of WP-4. Keep FSR-04, Q2, and WP-4 open until the separate prescribed-angle plan and complete Q2 evidence also pass. Record any remaining scope limits explicitly; scan, commit, verify, and push the audit update.

## Plan Self-Review

- Spec coverage: Task 1 closes the missing three-dimensional quadratic rule; Task 2 makes it mandatory; Task 3 covers derivative, minimized/sampled, MPI, five-angle, and moving-response evidence; Task 4 freezes thresholds, provenance, GCI, and formal status.
- Placeholder scan: every step names concrete files, behavior, commands or gates; no deferred placeholder remains.
- Type consistency: Task 1 produces order-two rule/provenance consumed by Task 2; Task 2 guarantees the rule contract consumed by Task 3; Tasks 1-3 are the frozen source consumed by Task 4.
