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
