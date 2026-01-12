# Constraints Module - Unit Test Checklist

This document lists additional unit tests that should be added to ensure the Constraints module conforms to established finite element literature and best practices.

**Current Test Coverage Summary:**
- 18 test files, ~3,200 lines
- Core functionality well-tested
- Gaps in: verification tests, parallel, coupled BCs, convergence studies

---

## Table of Contents

1. [AffineConstraints](#1-affineconstraints)
2. [DirichletBC](#2-dirichletbc)
3. [NeumannBC](#3-neumannbc)
4. [RobinBC](#4-robinbc)
5. [PeriodicBC](#5-periodicbc)
6. [MultiPointConstraint](#6-multipointconstraint)
7. [HangingNodeConstraint](#7-hangingnodeconstraint)
8. [GlobalConstraint](#8-globalconstraint)
9. [LagrangeMultiplier](#9-lagrangemultiplier)
10. [PenaltyMethod](#10-penaltymethod)
11. [ConstraintDistributor](#11-constraintdistributor)
12. [ConstraintTransform](#12-constrainttransform)
13. [ParallelConstraints](#13-parallelconstraints)
14. [CoupledBoundaryConditions](#14-coupledboundaryconditions)
15. [Integration Tests](#15-integration-tests)

---

## 1. AffineConstraints

### 1.1 [x] `TransitiveClosureLongChain`
- **Why Needed:** Current tests only verify chains of length 2. Longer chains may expose numerical accumulation errors or algorithm bugs.
- **What It Tests:** Transitive closure with chain length 5+: `u_0 = u_1 = u_2 = u_3 = u_4 = u_5`. Verify all slaves resolve to terminal master.

### 1.2 [x] `TransitiveClosureDiamondPattern`
- **Why Needed:** Diamond dependency patterns (A→B, A→C, B→D, C→D) are common in mesh refinement and can cause issues with duplicate resolution.
- **What It Tests:** Constraint `u_0 = 0.5*u_1 + 0.5*u_2` where `u_1 = u_3` and `u_2 = u_3`. Final result should be `u_0 = u_3`.

### 1.3 [x] `TransitiveClosureWeightAccumulation`
- **Why Needed:** Complex constraint chains with non-unity weights can accumulate floating-point errors.
- **What It Tests:** Chain with weights: `u_0 = 0.7*u_1`, `u_1 = 0.3*u_2 + 0.7*u_3`, verify final weights sum correctly and maintain precision to 1e-14.

### 1.4 [x] `CycleDetectionSelfReference`
- **Why Needed:** Self-referential constraints (`u_0 = u_0`) are degenerate and must be caught.
- **What It Tests:** Verify `ConstraintCycleException` is thrown for self-referencing constraint.

### 1.5 [x] `CycleDetectionLongCycle`
- **Why Needed:** Cycles longer than 2 may not be detected by simple pairwise checks.
- **What It Tests:** Detect cycle in chain: `u_0→u_1→u_2→u_3→u_0`.

### 1.6 [ ] `LargeScalePerformance`
- **Why Needed:** Assembly performance is critical; constraint operations should scale linearly.
- **What It Tests:** Create 100,000 constraints with varying masters. Verify `close()` completes in reasonable time and memory usage is bounded.

### 1.7 [x] `NearZeroWeightElimination`
- **Why Needed:** Weights below tolerance should be eliminated to prevent ill-conditioning.
- **What It Tests:** Add entry with weight 1e-16, verify it is eliminated after close(). Add entry with weight 1e-14, verify it is retained.

### 1.8 [x] `InhomogeneityOnlyUpdate`
- **Why Needed:** Time-dependent BCs require efficient inhomogeneity updates without full re-closure.
- **What It Tests:** After close(), call `updateInhomogeneity()` and verify constraints update correctly without recomputing structure.

### 1.9 [x] `CopyAndMoveSemantics`
- **Why Needed:** Ensure deep copy and move operations preserve constraint integrity.
- **What It Tests:** Copy closed AffineConstraints, modify original, verify copy unchanged. Test move constructor leaves source in valid empty state.

### 1.10 [x] `MergeConstraintSets`
- **Why Needed:** Multi-physics problems often require merging constraint sets from different sources.
- **What It Tests:** Merge two AffineConstraints objects, verify no duplicate slaves, correct closure of merged set.

---

## 2. DirichletBC

### 2.1 [ ] `PatchTestConstantField`
- **Why Needed:** Fundamental FE verification test (Hughes, "The Finite Element Method", Ch. 4). A mesh with Dirichlet BCs imposing a constant field must reproduce that constant exactly.
- **What It Tests:** On a simple mesh, impose u=C on all boundary nodes. Solve Laplace equation. Verify interior nodes have u=C to machine precision.

### 2.2 [ ] `PatchTestLinearField`
- **Why Needed:** Linear polynomial fields must be reproduced exactly by conforming elements with proper Dirichlet BCs.
- **What It Tests:** Impose u(x,y) = ax + by + c on boundary. Solve Laplace equation. Verify interior nodes match linear function to machine precision.

### 2.3 [ ] `ManufacturedSolutionVerification`
- **Why Needed:** Method of manufactured solutions (Roache, "Verification and Validation") provides rigorous BC accuracy verification.
- **What It Tests:** Use u_exact = sin(πx)sin(πy), compute source term, apply exact Dirichlet BCs, verify L2 error converges at expected rate with mesh refinement.

### 2.4 [x] `ComponentWiseDirichletVector`
- **Why Needed:** Vector fields (velocity, displacement) often require constraining individual components.
- **What It Tests:** For 3-component vector field, constrain only x-component. Verify y,z components remain unconstrained and solve correctly.

### 2.5 [x] `TimeDependentRampFunction`
- **Why Needed:** Transient simulations require smooth BC ramping to avoid numerical shocks.
- **What It Tests:** Apply g(t) = t for t∈[0,1], g(t)=1 for t>1. Verify constraint values update correctly at multiple time steps.

### 2.6 [x] `TimeDependentSinusoidalFunction`
- **Why Needed:** Oscillatory BCs are common in wave propagation and vibration problems.
- **What It Tests:** Apply g(t) = sin(ωt). Verify values at t=0, π/2ω, π/ω, 3π/2ω match expected values.

### 2.7 [ ] `HighOrderNodeDirichlet`
- **Why Needed:** Higher-order elements have interior edge/face nodes that may require special handling.
- **What It Tests:** P2 element with mid-edge nodes on Dirichlet boundary. Verify all boundary nodes (corner and mid-edge) correctly constrained.

### 2.8 [ ] `DirichletOnInternalInterface`
- **Why Needed:** Internal interfaces (e.g., crack faces, material interfaces) may have Dirichlet-type constraints.
- **What It Tests:** Mark internal face as Dirichlet. Verify nodes on that face are constrained while adjacent volume nodes are free.

### 2.9 [x] `ConflictingDirichletDetection`
- **Why Needed:** Two Dirichlet BCs on same DOF with different values is an error that must be caught.
- **What It Tests:** Apply u_5 = 1.0 and u_5 = 2.0. Verify exception or warning is raised.

### 2.10 [ ] `LiftingFunctionComputation`
- **Why Needed:** Some formulations require computing a lifting function that satisfies inhomogeneous BCs.
- **What It Tests:** Given Dirichlet data, verify `computeLiftingFunction()` returns vector satisfying all Dirichlet constraints.

---

## 3. NeumannBC

### 3.1 [ ] `ConstantFluxHeatConduction`
- **Why Needed:** Analytical solution exists for 1D heat conduction with constant flux BC.
- **What It Tests:** 1D bar with q=1 on right, T=0 on left. Verify linear temperature profile T(x) = qx/k.

### 3.2 [ ] `ParabolicProfilePoiseuille`
- **Why Needed:** Poiseuille flow provides analytical traction BC verification.
- **What It Tests:** Verify computed traction on pipe wall matches τ_wall = 4μU/R for Poiseuille flow.

### 3.3 [ ] `BalancedNeumannSingularSystem`
- **Why Needed:** Pure Neumann problem is singular; flux must be balanced for solvability.
- **What It Tests:** Verify error/warning when total Neumann flux is non-zero without pinned DOF or mean constraint.

### 3.4 [ ] `SpatiallyVaryingFlux`
- **Why Needed:** Non-uniform flux BCs are common (e.g., pressure distribution on surface).
- **What It Tests:** Apply q(x) = x² on boundary segment. Verify integrated flux ∫q dS matches analytical value.

### 3.5 [ ] `NormalVsTangentialTraction`
- **Why Needed:** Vector Neumann BCs (traction) have normal and tangential components with different physical meanings.
- **What It Tests:** Apply pure normal traction t = p·n. Verify tangential component is zero in resulting forces.

### 3.6 [ ] `TractionConsistencyWithStress`
- **Why Needed:** Traction BC t = σ·n must be consistent with internal stress field.
- **What It Tests:** Solve elasticity with known stress, verify boundary traction matches σ·n at quadrature points.

### 3.7 [ ] `FollowerLoadVsDeadLoad`
- **Why Needed:** Nonlinear problems distinguish between loads that follow deformation (follower) vs. fixed direction (dead).
- **What It Tests:** Verify NeumannBC correctly handles both modes with appropriate flags.

### 3.8 [x] `TimeDependentPulsatileFlux`
- **Why Needed:** Cardiovascular simulations require pulsatile flow BCs.
- **What It Tests:** Apply q(t) = Q_max · sin(2πt/T). Verify flux values at key time points in cardiac cycle.

---

## 4. RobinBC

### 4.1 [ ] `ConvectiveHeatTransferNewtonCooling`
- **Why Needed:** Robin BC α·u + β·∂u/∂n = g models Newton's law of cooling; analytical solutions exist.
- **What It Tests:** Fin with convective BC: h(T - T∞) = -k·∂T/∂n. Verify temperature profile matches analytical fin solution.

### 4.2 [ ] `AbsorbingBCWaveReflection`
- **Why Needed:** First-order absorbing BC ∂u/∂t + c·∂u/∂n = 0 should minimize reflections.
- **What It Tests:** Send Gaussian pulse toward absorbing boundary. Measure reflected wave amplitude. Verify reflection coefficient < 5% for normal incidence.

### 4.3 [ ] `ImpedanceBCAcoustics`
- **Why Needed:** Acoustic impedance BC relates pressure and velocity: p = Z·v_n.
- **What It Tests:** Apply impedance BC with known Z. Verify pressure/velocity ratio matches Z at boundary.

### 4.4 [ ] `ElasticFoundation`
- **Why Needed:** Winkler foundation models support as distributed springs: σ_n = k·u.
- **What It Tests:** Beam on elastic foundation. Verify deflection matches analytical Winkler beam solution.

### 4.5 [ ] `RobinToNeumannLimit`
- **Why Needed:** As α→0, Robin BC should reduce to Neumann BC.
- **What It Tests:** Set α=1e-15, β=1, g=q. Verify solution matches pure Neumann BC solution.

### 4.6 [ ] `RobinToDirichletLimit`
- **Why Needed:** As β→0, Robin BC should reduce to Dirichlet BC.
- **What It Tests:** Set α=1, β=1e-15, g=αu_D. Verify solution matches Dirichlet BC u=u_D.

### 4.7 [ ] `SpatiallyVaryingRobinCoefficients`
- **Why Needed:** Variable heat transfer coefficients occur at varying flow conditions.
- **What It Tests:** Apply h(x) = h_0(1 + x/L) along boundary. Verify correct evaluation at each quadrature point.

### 4.8 [x] `MatrixContributionSymmetry`
- **Why Needed:** Robin BC adds symmetric contribution to stiffness matrix (for self-adjoint problems).
- **What It Tests:** Verify `computeMatrixContribution()` produces symmetric local matrix: K_ij = K_ji.

---

## 5. PeriodicBC

### 5.1 [ ] `BlochWaveModeVerification`
- **Why Needed:** Periodic BCs are fundamental to Bloch wave analysis in phononic/photonic crystals.
- **What It Tests:** Unit cell with periodic BCs. Compute eigenfrequencies, verify against analytical dispersion relation for simple lattice.

### 5.2 [ ] `AntiPeriodicModeShapes`
- **Why Needed:** Anti-periodic BCs (u_L = -u_R) capture half-wavelength modes.
- **What It Tests:** 1D domain with anti-periodic BC. Verify odd mode shapes are captured: sin(π(2n+1)x/L).

### 5.3 [ ] `PeriodicWithPhaseShift`
- **Why Needed:** Floquet-Bloch analysis requires u(x+L) = e^{ikL}·u(x).
- **What It Tests:** Apply complex phase shift e^{iπ/4}. Verify constraint correctly relates real and imaginary parts.

### 5.4 [ ] `ThreeDimensionalPeriodicity`
- **Why Needed:** Crystal simulations require periodicity in all three directions.
- **What It Tests:** Cube with x, y, z periodicity. Verify 8 corner nodes reduce to 1 independent DOF, 12 edges reduce correctly.

### 5.5 [ ] `PeriodicCornerNodeConsistency`
- **Why Needed:** Corner nodes belong to multiple periodic faces; constraint consistency is critical.
- **What It Tests:** 2D square with x and y periodicity. All 4 corners should be constrained to single master.

### 5.6 [ ] `PeriodicHighOrderElementNodes`
- **Why Needed:** High-order elements have edge/face interior nodes requiring periodic constraints.
- **What It Tests:** Q2 element with periodic BCs. Verify edge midpoint nodes on opposite faces are correctly paired.

### 5.7 [x] `NonMatchingPeriodicMeshTolerance`
- **Why Needed:** Floating-point coordinates may not match exactly on opposite boundaries.
- **What It Tests:** Introduce 1e-10 coordinate perturbation. Verify matching still succeeds with appropriate tolerance.

### 5.8 [ ] `PeriodicWithRotationTransform`
- **Why Needed:** Rotational periodicity (e.g., turbomachinery sectors) requires transformation.
- **What It Tests:** 60° sector with rotational periodicity. Verify velocity vectors correctly transformed at periodic boundaries.

### 5.9 [x] `PeriodicConstraintChainResolution`
- **Why Needed:** Periodic constraints can create chains (u_A = u_B = u_C) that must resolve correctly.
- **What It Tests:** Three-domain periodic problem. Verify transitive closure selects consistent master DOF.

---

## 6. MultiPointConstraint

### 6.1 [ ] `RigidBodyConstraint`
- **Why Needed:** Rigid body motion constraints are common in structural mechanics.
- **What It Tests:** Constrain node B to move rigidly with node A: u_B = u_A + θ × (x_B - x_A). Verify correct displacement under rotation.

### 6.2 [ ] `TyingConstraintAcrossInterface`
- **Why Needed:** Non-conforming mesh interfaces require tying constraints.
- **What It Tests:** Two meshes with non-matching nodes at interface. Apply mortar-style tying. Verify displacement continuity.

### 6.3 [x] `AveragingConstraintDirichlet`
- **Why Needed:** Average Dirichlet BCs (mean value constraints) arise in incompressible flow.
- **What It Tests:** Constrain average of DOFs 1-10 to equal 5.0. Verify mean(u_1...u_10) = 5.0 after solve.

### 6.4 [ ] `DistributedLoadToMPC`
- **Why Needed:** Converting distributed loads to point constraints is sometimes needed.
- **What It Tests:** Verify weighted MPC correctly distributes load among nodes.

### 6.5 [ ] `OverconstrainedSystemDetection`
- **Why Needed:** More constraints than DOFs leads to singular or inconsistent systems.
- **What It Tests:** Add 3 linearly dependent constraints on 2 DOFs. Verify error is raised.

### 6.6 [ ] `RedundantConstraintElimination`
- **Why Needed:** Redundant constraints (same slave, same result) should be safely merged.
- **What It Tests:** Add u_0 = 0.5*u_1 + 0.5*u_2 twice. Verify single constraint after processing.

### 6.7 [x] `EquationFormSlaveSelection`
- **Why Needed:** Equation form `a*u_0 + b*u_1 + c*u_2 = d` requires selecting slave (pivot) wisely.
- **What It Tests:** Equation with coefficients [1e-10, 1.0, 0.5]. Verify slave is u_1 (largest coefficient), not u_0.

### 6.8 [ ] `IncompatibleConstraintDetection`
- **Why Needed:** Conflicting constraints (u_0 = 1.0 and u_0 = 2.0) must be detected.
- **What It Tests:** Add contradictory constraints. Verify exception raised with clear diagnostic.

---

## 7. HangingNodeConstraint

### 7.1 [x] `P1EdgeMidpointExact`
- **Why Needed:** Linear interpolation at midpoint must be exact for linear fields.
- **What It Tests:** Hanging node at edge midpoint. Impose linear field u(x)=x on parents. Verify u_hanging = 0.5*(u_0 + u_1) exactly.

### 7.2 [x] `P2EdgeThirdPoints`
- **Why Needed:** Quadratic elements have nodes at 1/3 and 2/3 positions.
- **What It Tests:** Verify P2 interpolation weights at ξ=1/3: N_0(1/3), N_1(1/3), N_2(1/3) are correct.

### 7.3 [x] `Q1FaceHangingNode`
- **Why Needed:** 3D hex mesh refinement creates face-interior hanging nodes.
- **What It Tests:** Face center hanging node on refined hex. Verify bilinear weights [0.25, 0.25, 0.25, 0.25].

### 7.4 [ ] `Q2FaceHangingNode`
- **Why Needed:** Quadratic hexes have 9 nodes per face; hanging nodes need biquadratic weights.
- **What It Tests:** Face center on Q2 element. Verify 9 biquadratic weights sum to 1 and match standard FE shape functions.

### 7.5 [ ] `TriangleFaceHangingNode`
- **Why Needed:** Tetrahedral mesh refinement creates triangular face hanging nodes.
- **What It Tests:** Centroid hanging node on P1 triangle face. Verify weights [1/3, 1/3, 1/3].

### 7.6 [x] `HangingNodeChainResolution`
- **Why Needed:** Multi-level refinement can create chains: hanging_1 depends on hanging_2 depends on real node.
- **What It Tests:** Two levels of refinement. Verify transitive closure resolves all hanging nodes to real nodes.

### 7.7 [x] `HangingNodePreservesPartitionOfUnity`
- **Why Needed:** Sum of shape functions must equal 1 (partition of unity) for consistency.
- **What It Tests:** For all hanging node weight sets, verify sum(weights) = 1.0 to machine precision.

### 7.8 [ ] `HangingNodeConvergenceRate`
- **Why Needed:** Hanging node constraints must not degrade convergence rate of the FE solution.
- **What It Tests:** Solve Poisson on adaptively refined mesh. Verify optimal convergence rate O(h^{p+1}) in L2 norm.

### 7.9 [ ] `HangingNodeFluxContinuity`
- **Why Needed:** Constrained approximations should still satisfy flux continuity in weak sense.
- **What It Tests:** Compute jump in normal flux across hanging node interface. Verify jump is O(h^p) or better.

### 7.10 [ ] `MixedOrderHangingNode`
- **Why Needed:** Transition between p=1 and p=2 elements requires careful constraint handling.
- **What It Tests:** Interface between P1 and P2 elements. Verify constraint correctly interpolates using P1 basis.

---

## 8. GlobalConstraint

### 8.1 [ ] `ZeroMeanPressureIncompressible`
- **Why Needed:** Incompressible Stokes flow has pressure defined up to a constant; zero mean is standard fix.
- **What It Tests:** Solve Stokes, apply zero-mean pressure. Verify ∫p dΩ = 0 and solution matches analytical.

### 8.2 [ ] `FixedMeanTemperature`
- **Why Needed:** Some thermal problems specify mean temperature rather than pointwise value.
- **What It Tests:** Constrain mean(T) = T_ref. Verify after solve that mean equals target.

### 8.3 [ ] `VolumeConservationConstraint`
- **Why Needed:** Incompressibility as global constraint: ∫div(u) dΩ = 0.
- **What It Tests:** Apply volume conservation. Verify volume change is zero for solved velocity field.

### 8.4 [ ] `NullspacePinningSymmetric`
- **Why Needed:** Symmetric problems may have zero-energy modes (rigid body) requiring pinning.
- **What It Tests:** Floating structure (no Dirichlet BCs). Pin 6 DOFs for rigid body modes. Verify unique solution.

### 8.5 [x] `WeightedMeanConstraint`
- **Why Needed:** Non-uniform weights arise from mass-weighted averages.
- **What It Tests:** Mass-weighted mean constraint. Verify Σ(m_i * u_i) / Σm_i = target.

### 8.6 [ ] `MultipleMeanConstraints`
- **Why Needed:** Multi-physics may require multiple global constraints simultaneously.
- **What It Tests:** Zero-mean pressure AND fixed-mean temperature. Verify both constraints satisfied.

### 8.7 [ ] `MeanConstraintIterativeSolverCompatibility`
- **Why Needed:** Lagrange multiplier formulation affects iterative solver convergence.
- **What It Tests:** Solve system with CG and global constraint. Verify convergence in reasonable iterations.

---

## 9. LagrangeMultiplier

### 9.1 [ ] `SaddlePointWellPosedness`
- **Why Needed:** Saddle-point system [A B^T; B 0] must satisfy inf-sup condition (Brezzi condition).
- **What It Tests:** Verify minimum eigenvalue of Schur complement B A^{-1} B^T is bounded away from zero.

### 9.2 [ ] `ConstraintForceAccuracy`
- **Why Needed:** Lagrange multiplier λ represents reaction force; accuracy is physically important.
- **What It Tests:** Cantilever with Dirichlet BC at support. Verify λ matches analytical reaction force.

### 9.3 [ ] `MultiplierVsPenaltyComparison`
- **Why Needed:** Both methods should give same solution; multiplier is exact while penalty is approximate.
- **What It Tests:** Same constraint via both methods. Verify solutions match as penalty → ∞.

### 9.4 [ ] `StabilizedSaddlePoint`
- **Why Needed:** Unstabilized saddle-point can have spurious pressure modes.
- **What It Tests:** Apply stabilization parameter. Verify solution unchanged but condition number improved.

### 9.5 [x] `MultiplierExtraction`
- **Why Needed:** Post-processing to extract λ values is needed for reaction forces.
- **What It Tests:** Solve saddle-point system. Extract λ vector. Verify dimensions and values.

### 9.6 [ ] `NestedIterativeSolver`
- **Why Needed:** Direct solve of saddle-point is expensive; iterative methods needed.
- **What It Tests:** Apply Schur complement preconditioned MINRES. Verify convergence.

### 9.7 [x] `TransposeOperatorSymmetry`
- **Why Needed:** B^T must be exactly transpose of B for symmetric formulation.
- **What It Tests:** Apply B and B^T to test vectors. Verify (Bx, y) = (x, B^T y).

---

## 10. PenaltyMethod

### 10.1 [ ] `PenaltyParameterConvergence`
- **Why Needed:** Error should decrease as penalty α increases: error ∝ 1/α.
- **What It Tests:** Sweep α from 1e3 to 1e9. Plot constraint violation vs α. Verify linear relationship on log-log scale.

### 10.2 [x] `OptimalPenaltySelection`
- **Why Needed:** Too small α = inaccurate; too large α = ill-conditioned.
- **What It Tests:** Use `computeOptimalPenalty()`. Verify selected α gives constraint error < tol without excessive condition number.

### 10.3 [ ] `ConditionNumberSensitivity`
- **Why Needed:** Penalty method degrades condition number as α increases.
- **What It Tests:** Measure condition number for α = 1e3, 1e6, 1e9. Verify cond(A + αB^TB) ≈ cond(A) * α.

### 10.4 [ ] `IterativeSolverConvergence`
- **Why Needed:** Large penalties cause slow iterative convergence.
- **What It Tests:** Solve with CG at various penalty values. Measure iteration count. Verify degradation pattern.

### 10.5 [x] `AdaptivePenaltyRefinement`
- **Why Needed:** Adaptive penalty can improve accuracy without excessive ill-conditioning.
- **What It Tests:** Apply `adaptPenalties()` based on residuals. Verify tight constraints get higher penalties.

### 10.6 [ ] `PenaltyVsElimination`
- **Why Needed:** Elimination (row/column modification) is exact; penalty is approximate.
- **What It Tests:** Same Dirichlet BC via penalty vs elimination. Compare solution accuracy.

### 10.7 [ ] `NitscheMethodComparison`
- **Why Needed:** Nitsche's method is variationally consistent penalty; should be more accurate.
- **What It Tests:** Compare standard penalty vs Nitsche for Dirichlet BC. Verify Nitsche achieves optimal convergence rate.

### 10.8 [ ] `MixedPenaltyValues`
- **Why Needed:** Different constraints may need different penalty values.
- **What It Tests:** Dirichlet α=1e6, MPC α=1e4. Verify both constraints satisfied to appropriate tolerances.

---

## 11. ConstraintDistributor

### 11.1 [x] `SymmetricEliminationPreservesSymmetry`
- **Why Needed:** Symmetric elimination must not break matrix symmetry.
- **What It Tests:** Apply Dirichlet to symmetric matrix. Verify result is symmetric: |A_ij - A_ji| < tol.

### 11.2 [ ] `NonSymmetricAssemblySupport`
- **Why Needed:** Advection-dominated problems have non-symmetric matrices.
- **What It Tests:** Apply constraints to non-symmetric matrix. Verify correct row/column elimination.

### 11.3 [ ] `BlockMatrixDistribution`
- **Why Needed:** Multi-physics problems use block matrices (e.g., [[K, G], [D, 0]]).
- **What It Tests:** Apply Dirichlet to velocity block only. Verify pressure block unchanged.

### 11.4 [ ] `SparsityPatternPreservation`
- **Why Needed:** Constraint distribution should not introduce new non-zeros unexpectedly.
- **What It Tests:** Track sparsity pattern before/after distribution. Verify fill-in is bounded.

### 11.5 [ ] `ElementAssemblyOrder Independence`
- **Why Needed:** Final matrix must not depend on element assembly order.
- **What It Tests:** Assemble elements in forward vs reverse order. Verify identical final matrix.

### 11.6 [ ] `ThreadSafeDistribution`
- **Why Needed:** Parallel assembly requires thread-safe constraint distribution.
- **What It Tests:** Parallel element loop with constraint distribution. Verify no race conditions, correct result.

### 11.7 [ ] `LocalCondensationVsGlobalElimination`
- **Why Needed:** Local condensation before assembly vs. global elimination should give same result.
- **What It Tests:** Compare `condenseLocal()` approach vs. post-assembly elimination.

---

## 12. ConstraintTransform

### 12.1 [x] `ProjectionIsIdempotent`
- **Why Needed:** Projector P should satisfy P² = P (idempotency is fundamental projector property).
- **What It Tests:** Apply projection twice: P(Pv) = Pv for arbitrary v.

### 12.2 [ ] `ReducedSystemConditionNumber`
- **Why Needed:** Reduced system should have better condition number than penalized system.
- **What It Tests:** Compare cond(P^T A P) vs cond(A + αB^TB). Verify reduced system is better conditioned.

### 12.3 [x] `ReductionRatioMetrics`
- **Why Needed:** DOF reduction provides computational savings; metrics help predict cost.
- **What It Tests:** Verify `getStats()` correctly reports n_full, n_reduced, reduction_ratio.

### 12.4 [ ] `ReducedOperatorSPD`
- **Why Needed:** If A is SPD and constraints are consistent, P^T A P should be SPD.
- **What It Tests:** Verify reduced operator is positive definite (all eigenvalues > 0).

### 12.5 [x] `RhsReductionAccuracy`
- **Why Needed:** Reduced RHS g = P^T(b - Ac) must be computed accurately.
- **What It Tests:** Compute reduced RHS, expand solution, verify original equation A*u = b - residual.

### 12.6 [ ] `CSRExportImport`
- **Why Needed:** Projection matrix in CSR format needed for sparse solvers.
- **What It Tests:** Export P to CSR, reimport, verify P*z gives same result as internal `applyProjection()`.

---

## 13. ParallelConstraints

### 13.1 [x] `GhostConstraintExchange`
- **Why Needed:** Constrained ghost DOFs need constraint data from owning rank.
- **What It Tests:** Two-rank partition with ghost node. Exchange constraints. Verify ghost has correct constraint.

### 13.2 [x] `OwnerWinsConflictResolution`
- **Why Needed:** When multiple ranks claim same DOF, owner's constraint should win.
- **What It Tests:** Conflict where rank 0 owns DOF, rank 1 has different constraint. Verify rank 0's constraint used.

### 13.3 [x] `SmallestRankConflictResolution`
- **Why Needed:** Alternative policy: smallest rank number wins (for determinism without ownership).
- **What It Tests:** Set policy to SmallestRank. Verify rank 0 wins regardless of ownership.

### 13.4 [x] `ConsistencyAcrossPartitions`
- **Why Needed:** All ranks must have identical constraints for shared DOFs.
- **What It Tests:** After makeConsistent(), verify constraints on shared DOFs match across all ranks.

### 13.5 [ ] `ParallelTransitiveClosure`
- **Why Needed:** Constraint chains may span partitions: u_A (rank 0) depends on u_B (rank 1) depends on u_C (rank 0).
- **What It Tests:** Cross-partition chain. Verify closure correctly resolves to terminal master.

### 13.6 [ ] `ParallelAssemblyConsistency`
- **Why Needed:** Distributed assembly with constraints must give same result as serial.
- **What It Tests:** Assemble same problem in serial and parallel. Verify identical global matrix/vector.

### 13.7 [ ] `ScalabilityTest`
- **Why Needed:** Communication overhead should not dominate at scale.
- **What It Tests:** Strong scaling test: measure `makeConsistent()` time from 1 to 64 ranks. Verify sublinear growth.

### 13.8 [ ] `NonBlockingExchange`
- **Why Needed:** Overlap communication with computation for performance.
- **What It Tests:** Verify `startAsyncExchange()` / `finishAsyncExchange()` API works correctly.

---

## 14. CoupledBoundaryConditions

### 14.1 [x] `CoupledNeumannBasicEvaluation`
- **Why Needed:** CoupledNeumannBC exists but has no dedicated tests.
- **What It Tests:** Create CoupledNeumannBC depending on auxiliary field. Verify flux evaluation with different auxiliary states.

### 14.2 [ ] `CoupledNeumannPartialDerivative`
- **Why Needed:** Newton methods need ∂flux/∂u for Jacobian assembly.
- **What It Tests:** Verify `evaluateDerivative()` returns correct partial derivative.

### 14.3 [x] `CoupledRobinBasicEvaluation`
- **Why Needed:** CoupledRobinBC exists but has no dedicated tests.
- **What It Tests:** Create CoupledRobinBC with state-dependent coefficients. Verify evaluation at different states.

### 14.4 [ ] `CoupledRobinLinearization`
- **Why Needed:** Nonlinear Robin BC α(u)u + β∂u/∂n = g(u) requires linearization.
- **What It Tests:** Verify linearized matrix contributions: ∂/∂u[α(u)u] evaluated correctly.

### 14.5 [ ] `FluidStructureInterfaceBC`
- **Why Needed:** FSI problems have coupled traction and displacement BCs at interface.
- **What It Tests:** Model problem with fluid traction = solid traction continuity. Verify coupling terms correct.

### 14.6 [ ] `ThermalContactResistance`
- **Why Needed:** Contact resistance R: q = (T_1 - T_2)/R is state-dependent Neumann BC.
- **What It Tests:** Two-body contact with resistance. Verify heat flux depends on temperature jump.

### 14.7 [x] `CoupledBCContextPropagation`
- **Why Needed:** CoupledBCContext carries auxiliary data needed by coupled BCs.
- **What It Tests:** Populate context with auxiliary field. Verify BC evaluations access context correctly.

---

## 15. Integration Tests

### 15.1 [ ] `FullPoissonWithMixedBCs`
- **Why Needed:** Verify all BC types work together in realistic problem.
- **What It Tests:** Poisson equation with Dirichlet, Neumann, and Robin BCs on different boundaries. Compare to analytical solution.

### 15.2 [ ] `ElasticityWithPeriodicAndDirichlet`
- **Why Needed:** Periodic unit cell with bottom fixed is common in material modeling.
- **What It Tests:** Elastic cube with bottom Dirichlet, top traction, x/y periodic. Verify effective modulus.

### 15.3 [ ] `TransientHeatWithTimeDependentBCs`
- **Why Needed:** Time-dependent BCs must update correctly through time-stepping.
- **What It Tests:** Heat equation with T(boundary, t) = sin(t). Verify solution tracks BC through multiple time steps.

### 15.4 [ ] `AdaptiveRefinementWithHangingNodes`
- **Why Needed:** Adaptive mesh refinement is a core use case for hanging nodes.
- **What It Tests:** Adaptively refine mesh based on error estimator. Verify convergence rate is optimal despite hanging nodes.

### 15.5 [ ] `IncompressibleFlowWithPressurePinning`
- **Why Needed:** Stokes flow needs pressure constraint for uniqueness.
- **What It Tests:** Lid-driven cavity with zero-mean pressure. Verify velocity and pressure fields.

### 15.6 [ ] `MultiPhysicsConstraintMerging`
- **Why Needed:** Coupled problems merge constraints from multiple physics modules.
- **What It Tests:** Thermal-structural problem. Merge thermal and structural constraints. Verify no conflicts.

### 15.7 [ ] `ContactConstraintsNonPenetration`
- **Why Needed:** Contact mechanics uses inequality constraints (gap ≥ 0).
- **What It Tests:** Two-body contact. Verify non-penetration constraint satisfied, contact pressure ≥ 0.

### 15.8 [ ] `ConstraintSerializationRoundTrip`
- **Why Needed:** Checkpoint/restart requires serializing constraint state.
- **What It Tests:** Save AffineConstraints to file. Reload. Verify identical constraints.

### 15.9 [ ] `BackwardCompatibilityWithLegacySolver`
- **Why Needed:** New constraint module should produce equivalent results to legacy implementation.
- **What It Tests:** Same problem with old and new constraint handling. Verify solution difference < tolerance.

---

## Priority Matrix

| Priority | Test Category | Rationale |
|----------|---------------|-----------|
| **P0 (Critical)** | CoupledBoundaryConditions (14.1-14.7) | No existing tests |
| **P0 (Critical)** | ParallelConstraints MPI (13.1-13.6) | Only serial tests exist |
| **P1 (High)** | Verification tests (2.1-2.3, 3.1, 4.1-4.2) | Literature conformance |
| **P1 (High)** | Penalty convergence (10.1-10.4) | Quantitative accuracy |
| **P1 (High)** | Lagrange accuracy (9.1-9.3) | Physical correctness |
| **P2 (Medium)** | Advanced periodic (5.1-5.5) | Specialized applications |
| **P2 (Medium)** | Hanging node convergence (7.8-7.10) | Adaptive methods |
| **P2 (Medium)** | Integration tests (15.1-15.9) | System-level validation |
| **P3 (Low)** | Edge cases and robustness | Defensive coding |
| **P3 (Low)** | Performance tests (1.6, 13.7) | Optimization phase |

---

## References

1. Hughes, T.J.R. "The Finite Element Method: Linear Static and Dynamic Finite Element Analysis" (2000) - Patch tests, Ch. 4
2. Brezzi, F. & Fortin, M. "Mixed and Hybrid Finite Element Methods" (1991) - Inf-sup condition, Ch. 2
3. Roache, P.J. "Verification and Validation in Computational Science and Engineering" (1998) - Manufactured solutions
4. deal.II documentation - AffineConstraints design patterns: https://dealii.org/developer/doxygen/deal.II/classAffineConstraints.html
5. Demkowicz, L. "Computing with hp-Adaptive Finite Elements" (2006) - Hanging node constraints
6. Zienkiewicz, O.C. & Taylor, R.L. "The Finite Element Method" (2005) - General FEM reference

---

*Document Version: 1.0*
*Last Updated: 2026-01-12*
*Total Tests Listed: 122*
