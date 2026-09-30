# Free-Surface Boundary Condition: Program Tracker

**Status date:** 2026-09-29.
**Reviewed source:** `origin/issue-449-modern-mesh-core` at `fc56527` (2026-09-21), together with the WP-4 scratch worktree and its checkpoints (see section 9).

This file is the **single authoritative record** for the free-surface work. It contains:

- the goal;
- the current state;
- a re-assessment of the approach;
- the remaining work;
- a log of what has been tried.

The documents it superseded were removed on 2026-09-29. Section 10.3 lists them, with a one-line summary and a command to retrieve each from Git history. Where this file and a remaining document disagree about status or next steps, this file wins.

Items marked **Assessment** or **Proposal** are judgments made during the 2026-09-29 review. Everything else is taken from the repository, the committed documents, or the recorded checkpoints, and a source is named for each.

---

## Contents

1. [Goal](#1-goal)
2. [Where we are](#2-where-we-are)
3. [Re-assessment: why WP-4 has not closed](#3-re-assessment-why-wp-4-has-not-closed)
4. [Decisions requested](#4-decisions-requested)
5. [Remaining work](#5-remaining-work)
6. [Formulation reference (current code)](#6-formulation-reference-current-code)
7. [Legacy tracking IDs and their disposition](#7-legacy-tracking-ids-and-their-disposition)
8. [Log of what has been tried](#8-log-of-what-has-been-tried)
9. [State inventory](#9-state-inventory)
10. [Document index](#10-document-index)
11. [Glossary](#11-glossary)
12. [How to update this file](#12-how-to-update-this-file)

---

## 1. Goal

### 1.1 Physical problem

We need a free-surface boundary condition for incompressible Navier–Stokes in the new OOP solver (FE/Mesh/Physics/Application). The liquid occupies `Ω(t)`, and its free surface `Γ(t)` meets solid walls along contact lines.

- **Bulk:** `ρ(∂t u + u·∇u) − div σ(u,p) = ρ g` and `div u = 0`, with `σ = −p I + 2μ ε(u)`.
- **Kinematic condition on Γ:** the normal velocity of the surface equals `u·n`.
- **Dynamic condition on Γ:** `σ n = −(p_ext + γ κ) n`.
  - `n` is the unit normal pointing out of the liquid, and `κ = div_Γ n`.
  - `γ` is a constant surface tension. Marangoni stresses are out of scope.
- **Walls:**
  - no penetration, `u·n_w = 0`;
  - optional Navier slip, `(μ/ℓ_s) u_t = −(σ n_w)_t`;
  - a contact line with either a static Young angle `θ_e` or a dynamic law (Ren–E: `V_CL = γ M (cos θ_e − cos θ_d)`).
- **Conservation:** liquid volume is conserved.
- **Two-fluid extension (later):** the exterior is a second incompressible fluid, and the dynamic condition becomes a stress jump.

### 1.2 Capability tiers and target cases

The target cases come from `moving_free_surface_validation_cases.md` and the audit's Q-gates.

| Tier | Capability | Target validation cases |
|---|---|---|
| **T1** | Gravity-dominated free surface (`γ = 0`) | Open tank at rest (hydrostatics); small-amplitude 2D sloshing (linear theory); SPHERIC Test 10 lateral sloshing; SPHERIC Test 05 wet-bed dam break (D18/D38); SPHERIC Test 02 dam break with obstacle; long-transient volume conservation |
| **T2** | Capillarity without contact lines | Static drop, 2D circle and 3D sphere (Laplace pressure, spurious currents); capillary wave (Prosperetti); oscillating drop (Lamb/Rayleigh) |
| **T3** | Wetting | Static sessile drop or meniscus at 30–150°; capillary rise (Gründing et al. reference, already prepared); dynamic contact line (Ren–E), advancing and receding |
| **T4** | Two-fluid | Static drop or bubble with both phases; Hysing rising bubble (cases 1 and 2) |
| **T5** | Violent, film and gas flows | D18/D38 to full horizon; films, jets and crowns; gas-sensitive impacts |

### 1.3 Two surface representations

- **Unfitted level set (CutFEM).** A fixed background mesh; a P1 level set; a generated `LinearCorner` interface; equal-order P1/P1 VMS/PSPG with small-cut aggregation. This is the primary path. It is the only path with qualified closures and the only one with contact-line models.
- **Fitted ALE.** A body-fitted free-surface boundary, coupled mesh displacement, and normal kinematics. It is suited to small or moderate deformations and is the classical well-balanced route for surface tension (Bänsch/Dziuk-type Laplace–Beltrami). Today it is only a low-level prerequisite.

---

## 2. Where we are

### 2.1 Capability status (2026-09-29)

| Capability | Path | Status | Best evidence | Blocking issue |
|---|---|---|---|---|
| Authoritative cut geometry (snapshot, measures, normals, contact rules) | Unfitted | **Qualified** (WP-2 v4) | `qualification_logs/free_surface_wp2_geometry_20260722_5cf65650` | Recovery and minimizer later built their own cut, which breaks the "one geometry" invariant (§3.2) |
| Velocity extension, no dry-side feedback | Unfitted | **Qualified** (WP-1) | `..._wp1_extension_20260720_398a2477` | Long-horizon D38 not shown |
| Sharp exterior boundary conditions | Unfitted, P1 `LinearCorner` | **Qualified** (WP-3 v6) | `..._wp3_sharp_boundary_v6_20260826_a73c77f4` | Higher order fails closed |
| Configuration containment | Both | **Qualified** (WP-0) | `..._wp0_configuration_20260720_ffef62d3` | — |
| Hydrostatics, flat interface, `γ` on, 90° walls | Unfitted | **Works**: exact to about 1e-11 over 960 2D/3D, 2-rank, layout and numbering cases | Audit WP-4 entries dated 08-25 to 08-27 | None. This is effectively done and does not need to grow further. |
| 2D linear sloshing | Unfitted | Worked on 2026-05-17 (interface L2 8.9e-5) | `linear_sloshing_2d/README.md` | Not re-run since the June stabilization change |
| SPHERIC 05 dam break | Unfitted | Early profile passes at t = 0.156 s (RMSE about 0.02 m) | 06-02 and 06-10 logs | D18 false wetting after 0.22 s; never run to full horizon |
| SPHERIC 10 sloshing and 02 dam break | Unfitted | **Fail.** Test 10 dies near 0.9 s of 8.35 s; Test 02 has MPa pressure spikes | 06-05/06-07 root-cause reports | Never re-run after the 06-10 to 06-12 aggregation and sparsity fixes |
| Static drop / Laplace pressure | Unfitted | **Physically reasonable but not accepted.** SurfaceStress closed drop (07-17): pressure error 3.15% → 0.32% → 0.0067% at n = 8/16/32; speed/γ about 1–2.6e-5. KAG sampled circle (09-04): after one step, 27% → 13% → 6.2% | `free_surface_level_set_review_20260713.md` L238; audit 09-04 entry | Gates demand algebraic equilibrium (§3.1) |
| Capillary wave | Unfitted | One scoped pass (n = 16: frequency error 1.2%, volume drift 9.3e-7) | Level-set review | No refinement study; raw data not archived |
| Static contact angle (sessile) | Unfitted | Partial. 07-17: all n = 32 rows pass per-mesh gates; the n = 16 90° and 120° rows fail the angle gate; refinement rates fail | Level-set review L16, L170 | Two owners of the angle (§3.5); P1 repair targets not representable |
| Dynamic contact line (Ren–E) | Unfitted | Pilot only. KAG has the correct sign with relative error 0.25–0.44. SurfaceStress fails advancing cases (error 128.66, and 17.38 with the wrong sign) | Audit L1819 (job 41286834) | No refined campaign |
| Capillary rise | Unfitted | Reference envelope and runner prepared; never run | `free_surface_wp5_capillary_rise_reference.json` | — |
| Conservative phase transport | Unfitted | Implemented; the 18-point matrix is frozen but not run | `level_set_conservative_phase_transport.md` | — |
| Cut stability (aggregation, pressure facet jump) | Unfitted | Implemented. Node-crossing slice passes; no uniform bound | `free_surface_wp7_combined_p1_method.md` | — |
| Discrete energy ledger | Unfitted | Narrow fixed-topology backward-Euler connector passes | WP-8 records | No physical energy campaign |
| Fitted ALE free surface | Fitted | Low-level prerequisite only (32 tests). SurfaceStress and contact are **rejected** on fitted boundaries | WP-9 record | SPHERIC 10 fitted run failed at step 12 (05-26) and was never re-run |
| Two-fluid core | Unfitted | Four stationary planar prerequisites pass (errors about 1e-16 to 1e-11) | WP-10 records | Static drop rejected on an exact-node topology event |

### 2.2 WP-4 as of 2026-09-21

WP-4 is titled "balanced capillary pressure, wall energy, and prescribed angle".

**Selected method.** The method was fixed on 2026-09-02. It has two parts:

- `KinematicAreaGradientTraction` (KAG): the curvature `κ_h` is recovered by solving `M κ = −∂(A_lg − Σ cos θ_w A_sl,w)/∂φ`, where `M_ij = ∫_Γh N_i N_j / |∇φ|` is an unstabilized trace mass.
- A fixed-volume minimizer of the same discrete energy over the nodal `φ`.

**The balance argument is correct in exact arithmetic.** At a constrained stationary point, `κ_h` is constant, so a constant pressure balances the force exactly. The code review on 2026-09-29 checked this algebra.

**Actual blockers.** These are recorded in `goal_wp4_conditioning_to_qualification_20260914.md` (removed; retrieve from commit `5b46da55`) and in the scratch checkpoint:

1. **The static minimizer does not converge.** The preserved 2D cap case, job `42294456`, ran 147 iterations and 2,071 evaluations. It ended with projected gradient `0.0220` and volume error `2.7e-7`; both were required to reach `1e-10`. It had reached the 128 topology-transition allowance and failed in `authoritative_equilibrated_consistent_mass_solve_failed`.
2. **The trace-mass solve is ill-conditioned.** The quotient condition number is about **3.4e14**. The best double-precision published `κ` for captured state "trial 39" reaches a scaled residual of `1.08e-10`, against a `1e-10` gate. The repair attempts were long-double recurrences, 80/120-digit oracles, about 33k "alternate rounding assessments", and finally storing curvature as a double-double pair (commit `fc56527`, 2026-09-21).
3. **Two geometry definitions disagree near vertex-touch configurations.**
   - The recovery and minimizer derivatives come from a strict cut that nudges nodes with `|φ|` below about 512ε to ±ε^(1/4).
   - The objective and the assembly use the snapshot's zero-band and pruned geometry.
   - On a Triangle3 witness the energy action is −0.7071 while the secant is 0.
   - The "authoritative derivative binding" overload is a stub that always returns `unverifiedAuthoritativeGeometry` (`LevelSetCurvatureProjection.cpp` about L5392–5421 at `fc56527`).
4. **The V3 qualification matrix cannot run.** Its status is `AWAITING_SCIENTIFIC_CONTRACTS`, with an empty contracts list. It holds 2,136 physical cases as frozen on 09-03, and 2,304 cases with 13,882 artifacts after the 09-14 edit. At up to 10 GiB and 4 h per case, with at most 4 nodes, that is weeks to months of wall time per campaign. Reruns are not allowed.
5. **Work is paused** at a weekly-usage floor. There are no live jobs. The latest checkpoint is `/scratch/users/zsexton/wp4-continuation-20260921-szBnNq/checkpoint.md`.

### 2.3 Effort metrics (for the re-assessment)

| Measure | Value | Source |
|---|---|---|
| Commits since 2026-07-15 | 406 | `git log` |
| Share of commits changing numerics | 57% (07-15 to 08-24) → 30% (since 08-25) → 19% (since 09-02); about 10 of 107 commits since 09-02 change the free-surface method | commit classification, 2026-09-29 |
| Line churn going to `Documentation/qualification_logs` | 58% (910k lines), versus 13% (197k lines) for production C++ | same |
| WP-4 dated checkpoint entries in the audit | 49 entries, about 164 KB (38% of the audit file) | audit |
| Closed work packages / closed Q-gates | 4 of 11 (WP-0 to WP-3) / 0 of 8 | audit |
| `ApplicationDriver.cpp` | 1,036 → **34,500** lines | refactor plan; tip |
| `NewtonSolver.cpp` | 19,515 lines, 110 environment-variable hooks | tip |
| Free-surface Python harness | 62.7k lines (84 files), plus 95.4k lines under `open_vessel_free_surface/` | tip |
| WP-4 code that defines the discretization vs. everything else | about 3–5k lines of discretization; more than 90% of the WP-4 code surface is diagnostics, certificates, provenance and harness | code review, 2026-09-29 |

---

## 3. Re-assessment: why WP-4 has not closed

The 2026-07-20 audit correctly identified real defects, and much of the infrastructure built since then is sound. The rigor also caught genuine bugs:

- the E ≠ u trace defect;
- the tetrahedral moment cancellation;
- the JIT cache-key bug;
- the sparsity/dropped-writes bug;
- the radius-propagation bug;
- the partition-dependent padding.

WP-4 stalled for the structural reasons below, not for lack of effort.

### 3.1 The acceptance gates require an exact discrete equilibrium that this discretization cannot reach on sampled data

The finest-level V3 gates are:

- parasitic capillary number `Ca = μ max|u|/γ ≤ 1e-6`;
- kinetic-energy proxy `≤ 1e-12`;
- pressure-jump error `≤ 1%` at R/h = 32;
- φ-scale invariance spread `≤ 1e-10`;
- static initializer tolerances `1e-10`.

These apply to *every* study, including sampled-analytic circles and the physical-scale lanes at R/h = 8.

**Assessment.** No published sharp unfitted P1 method reaches these on sampled curved interfaces:

- **Gross & Reusken** (2007; 2011 book, ch. 7): Laplace–Beltrami on piecewise-planar interfaces has an O(h^½)–O(h) force error. Spurious velocities shrink with h but never reach roundoff.
- **Popinet** and **Francois et al.**: machine-zero currents require either exact curvature or relaxation over viscous time scales.

The only way to satisfy the gates is to construct an exact discrete equilibrium. That requirement forced the chain of work in §3.2.

### 3.2 The exact-equilibrium route has three numerical obstacles

1. **The minimization is nonsmooth and degenerate.**
   - Over nodal φ on a fixed mesh, `E(φ)` is only piecewise smooth, with kinks wherever a vertex touches the interface or the topology changes.
   - Only the ratios of φ values near the interface matter, so the minimization also has scale null-directions.
   - The goal document itself concedes "no proof that a fixed-background P1 energy minimum always has a classical zero gradient".
   - Requiring `‖g_proj‖ ≤ 1e-10` on a nonsmooth problem is not a well-posed stopping rule.
2. **The trace mass is ill-conditioned.**
   - `M = ∫_Γh N_i N_j / |∇φ|` without stabilization is ill-conditioned whenever a basis function barely touches Γh. This is standard TraceFEM/CutFEM knowledge.
   - The measured condition number, about 3.4e14, makes a `1e-10` residual on the rounded published vector a finite-precision lottery.
   - The standard remedy is a normal-gradient or ghost-penalty stabilization, or the Helmholtz filter already in the code. It keeps constants in the kernel, so the "KKT ⇔ constant κ ⇔ balanced force" equivalence survives.
   - The static initializer hard-requires filter = 0 (`ApplicationDriver.cpp` about L26956–26976 at `fc56527`), and the method rules forbid "a force filter". The standard fix was therefore ruled out by policy, not by analysis.
3. **Two geometry definitions.** Value and derivative come from different cuts (§2.2, item 3), so line searches fail exactly at the symmetric vertex-touch states that the minimizer visits.

The integer interval arithmetic ("producer arithmetic", "dyadic bounds"), the coefficient pairs, and the binding and provenance work of 09-04 to 09-21 do not change the discretization. By the audit's own text, they certify which floating-point branch was taken.

### 3.3 The planned head-to-head comparison of capillary force routes was never run; a good result was set aside

The audit's AD-2 asked to "evaluate the alternatives on the same static and moving tests". Instead, a single route was fixed on 2026-09-02 because it promised exact balance.

The data that do exist point in different directions:

- **SurfaceStress** (Laplace–Beltrami on the generated interface, the current default) produced a closed drop with pressure-jump error 3.15% → 0.32% → 0.0067% at n = 8/16/32 (07-17). It was not accepted because speed/γ (about 1–2.6e-5) was nonmonotone and above a fixed 1e-5 gate.
  - Correction (2026-09-29): that speed figure was read after only 3 steps of 1 ms, in SI units for water, at R/h ≤ 9.6. It measured the start-up transient, not a relaxed spurious current, so it says little either way about spurious currents. The pressure-jump convergence stands.
- **KAG** on a sampled circle (09-04), measured after *one* step, gave 27% → 13% → 6.2%. This was with the analytic initial pressure; from zero pressure the result was about 114%.
- For **moving contact lines** the order reverses. At resolution 16, KAG gave correct-sign Ren–E speeds (error 0.25–0.44), while SurfaceStress failed advancing cases (errors 128.66, and 17.38 with the wrong sign).

These setups are not identical, so the numbers are indicative only. They still show that no single route is known to be best, and that the choice should be made on physical tests, not on algebraic identities.

### 3.4 The measurement protocol reports one-step transients

Static-drop pressure was read after a single step from sampled geometry, and it depended strongly on the initial pressure (10.7 versus 6.3 at R/h = 8). That is a transient, not an equilibrium.

**Proposal.** Use the standard protocol (Popinet 2009 "spurious currents"; Basilisk `spurious.c`):

- run for several viscous times `ρR²/μ`;
- report `max|u|(t)`;
- measure the pressure jump in the relaxed state;
- sweep the Laplace number and R/h.

### 3.5 The contact angle has two owners, and the P1 repair targets are not representable

For the prescribed angle, the code imposes the angle twice:

- geometrically, by an accepted-endpoint repair that resets each contact cell to an affine φ with the target angle;
- variationally, through the Young term in the momentum equation (SurfaceStress line force `−γ cos θ_e v·m`, or inside `κ_h` for KAG).

These disagree. The discrete stationary angle is not the repair's per-cell angle, so every repair moves the state off equilibrium and injects unaccounted work.

The team derived an irreducible worst-cell error of `atan(1/2)` for continuous-P1 repair targets on a bent contact line (audit about L1720). In addition:

- A no-slip wall with a moving contact line is the classical paradox.
- Weak (Nitsche) wall normal constraints cannot hold the wall-normal component of a static line force against a constant pressure.

**Assessment.** Pick one mechanism. The variational Young term, Navier slip on the wetted wall, a strong no-penetration constraint, and angle-*preserving* wall maintenance are the standard, energy-consistent choice (Gerbeau–Lelièvre 2009; Buscaglia–Ausas 2011).

### 3.6 Possible capillary time-step violation (to verify)

Both SurfaceStress and KAG are explicit in geometry within a Newton solve, with geometry refreshed by outer Picard passes. No semi-implicit Bänsch/Hysing term `γ Δt ∫ ∇_Γ u : ∇_Γ v` is present.

**Assessment (estimate, not verified).** The Brackbill-type limit `Δt ≲ sqrt(ρ h³ / (2π γ))` is about 6.6e-4 at R = 0.45, R/h = 32, ρ = γ = 1. The V3 matrix uses Δt = 1e-3 to 4e-3. Instability or poor outer-loop convergence at the finer levels would then be expected, independent of force balance.

### 3.7 Scope bundling and closure rules make partial progress invisible

WP-4 closes only when all of the following pass together:

- 2D and 3D;
- five angles, every wall rotation, both liquid signs, cut offsets;
- MPI layouts;
- GCI;
- energy adjoint, restoring force, φ-scale invariance;
- prescribed-angle scheduling;
- a frozen immutable campaign with reruns forbidden.

A harness failure forces a new matrix version (WP-3 needed V1–V6 for launcher and environment reasons after the numerics had passed). The flat case was expanded to 960 cases even though it is trivially balanced (zero curvature), while the curved case, the actual question, never passed. Each dated entry ends with "no checklist status changes".

### 3.8 Process overhead dominates

Evidence of the overhead:

- the metrics in §2.3;
- the audit's own remarks: "preparation and review, not the roughly one-second calculation, are the measured current bottleneck" (about L1752), and "provenance checks and wrapper overhead account for most of the 44-second phase despite its 0.08-second test time" (about L1768);
- the September operating rules: reciprocal hashes, source, index and cache guards, hash-only protected test files, 10-field worker packets, three-worker limits, byte-preserved archives.

These rules suit a publication-grade qualification record. They are a poor fit for a method that has not yet been chosen.

### 3.9 Tier-1 (gravity-dominated) work is unfinished, and it is the most application-relevant

- SPHERIC 10 and 02 fail.
- D18 shows false wetting after 0.22 s.
- Fitted-ALE sloshing failed at step 12.
- None of these was re-run after the June aggregation, sparsity and penalty fixes.

For open-vessel applications (large Bond number), T1 robustness matters more than 1e-6 spurious currents.

### 3.10 The parallel architecture refactor

The refactor has been dormant since 2026-09-05. R0–R2 were done: the baseline, the options and resolved configuration, and a thin FE integration-domain selector. The driver shrank only 1.8%.

It is not on the critical path. Its later packages (R5–R8) would move exactly the files WP-4 is editing.

---

## 4. Decisions requested

These are proposals. Each lists a recommended option and an alternative. Record the outcome here with a date.

| # | Decision | Recommended | Alternative |
|---|---|---|---|
| **D1** | Acceptance philosophy | **Adopted 2026-09-29.** Tiered capability goals (§1.2). Gates are literature-calibrated convergence and error bounds on physically relaxed states. Algebraic roundoff gates apply only to exactly representable states (flat, hydrostatic). | Keep the V3 gate contract. It needs an exactly-balanced construction and has no demonstrated path. |
| **D2** | Capillary force route (unfitted) | **Adopted 2026-09-29, under principle P1.** Run the missing AD-2 comparison, M2 in §5: SurfaceStress, KAG with filter or stabilized mass, and unfiltered KAG, on the same static-drop, capillary-wave, sessile and Ren–E tests, then select. Allow the filter or stabilization for KAG. | Continue unfiltered KAG with the double-double mass solve (the 09-14 goal, milestones 1–6). |
| **D3** | Static minimizer | **Adopted 2026-09-29.** Make it optional tooling. Reach equilibrium dynamically (viscous relaxation). Never gate physics on `‖g_proj‖ ≤ 1e-10`. If kept, it must use one geometry definition and a nonsmooth-aware stopping rule. | Keep it as the required initializer for all "minimized" lanes. |
| **D4** | Contact angle | **Adopted 2026-09-29 (single mechanism).** Variational Young term, Navier slip, strong no-penetration, and angle-preserving (scale-only) wall maintenance. Retire the repair-to-target as a production path, or keep it as an explicit alternative mode that is never combined with the Young term. | Keep both owners and finish the FSR-04 scheduling, strip and fixed-point work. |
| **D5** | Fitted ALE | **Adopted 2026-09-29.** Unlock fitted SurfaceStress behind a flag, use slip (not Dirichlet-0) wall mesh BCs, and run fitted 2D sloshing, then a static drop and capillary wave. This gives an independent, well-balanced reference. | Leave fitted capillarity rejected and focus on unfitted only. |
| **D6** | Process | **Adopted 2026-09-29.** Lightweight process (§12): a commit hash, a small benchmark script, a tolerance file, and a results row in this tracker. Raw outputs for accepted results go to group storage. Time-box investigations to about 3 working days before a method decision. Keep the author/committer identity rule, the commit-message vocabulary scan, and the job-mail settings. | Keep the frozen/immutable campaign process for every step. |
| **D7** | Architecture refactor | **Adopted 2026-09-29.** Pause R3–R12. Do targeted extractions only when a milestone touches that code. Revisit once T1–T3 are working. | Resume the refactor in parallel now. |
| **D8** | WP-4 September work in progress | **Adopted 2026-09-29.** Stop the conditioning/double-double line. Archive the dirty W worktree diff as a patch in `$GROUP_HOME`; do not integrate it. Keep committed `fc56527` as is. | Integrate the native pair implementation and continue milestones 1–3 of the 09-14 goal. |

### Recorded decisions

- **D1, 2026-09-29: the recommended option is adopted.**
  - Progress is judged against the capability tiers in §1.2.
  - Acceptance uses literature-calibrated convergence rates and error bounds, measured on physically relaxed states.
  - Roundoff-level gates apply only to exactly representable states: flat interfaces and hydrostatics.
  - **Rationale:** the V3 gates require an exact discrete equilibrium on sampled curved interfaces. No comparable published method meets them, and pursuing them stalled WP-4 (§3.1–§3.2).
  - **Consequences:**
    - The WP-4 V3 gate contract is no longer an acceptance standard and remains history only. This covers `tests/cases/fluid/free_surface_wp4_balanced_capillary_matrix_v3.json` and the W-only `wp4_qualification_gate_contract_20260906.md`.
    - The tolerances marked "proposal" in §5 are the working acceptance criteria. Each is confirmed or adjusted once, before its first use.
    - The Q0–Q7 texts remain reference material only (§7).
- **P1, 2026-09-29 (standing principle): prefer methods that need no additional parameter tuning.**
  - When candidates meet the same acceptance criteria, choose the one without a tunable numerical coefficient.
  - If a numerical parameter is unavoidable, fix its value once from analysis or dimensional scaling, document it in §6, and never tune it per case or per mesh.
  - Physical inputs (surface tension, contact angle, slip length, mobility) come from the case definition and are not tuning parameters.
  - Existing fixed constants (for example the 0.01 cut-pressure calibration and the aggregation guards) stay as they are; they are not re-tuned.
- **D2, 2026-09-29: the recommended comparison is adopted, under P1.**
  - The comparison runs the M2 static drop, the M3 capillary wave, and the M4 sessile and Ren–E cases, all with the same protocol.
  - The candidate set follows P1:
    - `SurfaceStress`, which is parameter-free;
    - a new parameter-free KAG variant with a lumped (row-sum) mass;
    - unfiltered consistent-mass KAG, as a reference only.
  - Filtered or stabilized KAG is a fallback only.
  - The unfiltered-KAG double-double line of the 09-14 goal is not continued as the method path. Recording D8 (archiving the W worktree diff) is the follow-up.
- **D3, 2026-09-29: the recommended option is adopted.**
  - Static equilibria are reached dynamically. Each case starts from the sampled analytic shape and runs until the flow has relaxed, typically 5–10 viscous times `ρR²/μ`.
  - Static-equilibrium results are always measured on the relaxed state.
  - No physical result is gated on minimizer convergence (`‖g_proj‖`, volume error, or the minimized-state certificate).
  - The static minimizer (`FE/LevelSet/LevelSetStaticCapillaryEquilibrium.*`, driven by `initializeDiscreteStaticCapillaryEquilibrium` in `ApplicationDriver.cpp`) remains optional tooling in two roles:
    - a best-effort initializer that shortens relaxation;
    - a small verification test that the balance equivalence (KKT ⇔ constant curvature ⇔ zero spurious flow) holds on meshes where the interface avoids mesh nodes.
  - Before either role is used, the minimizer must use one geometry definition for the energy and its derivatives, and a kink-aware stopping rule (M8).
  - The algebraic check of exact balance remains covered by the flat and hydrostatic tests.
  - **Rationale:**
    - D1 retired the minimized-state algebraic criteria as a standard.
    - The minimizer carries algorithm settings that P1 disfavors.
    - On the preserved 2D cap it stalled at `‖g_proj‖ = 0.022`, and the discrete energy may have no classical zero-gradient minimum.
- **D4, 2026-09-29: one contact-angle mechanism is adopted.**
  - The contact angle is imposed only through the variational Young term in the momentum equation:
    - with `SurfaceStress`, the line term `−γ cos θ_e ∫_CL v·m`;
    - with KAG, the wall-area gradient inside `κ_h`.
  - The contact line moves through Navier slip on the wetted wall, with the slip length taken from the case definition (P1). The Ren–E line friction is added for dynamic contact.
  - Contact walls use a strong no-penetration constraint, because weak (Nitsche) wall constraints cannot hold a static line force (§6.4).
  - Level-set wall maintenance preserves the current angle through the existing scale-only wall-aware mode.
  - The accepted-endpoint repair to the target angle is retired as a production path and is never combined with the Young term. Its code is removed once M4 validates the single mechanism.
  - The prescribed-angle completion stream of the 09-03 WP-4 plan (package D) is not pursued. That stream covered wall-repair scheduling, φ-scale invariance of the repair, the curved 3D wall strip, and the stage and fixed-point studies.
  - **Rationale:**
    - The two mechanisms impose different discrete angles, and each repair injects unaccounted work.
    - Continuous-P1 repair targets cannot represent a bent contact line (worst-cell error `atan(1/2)`).
    - The variational route is energy-consistent (Gerbeau–Lelièvre; Buscaglia–Ausas) and adds no numerical parameter (P1).
- **D5, 2026-09-29: the recommended option is adopted.**
  - The fitted ALE path is developed as an independent reference for the unfitted results.
  - Wall mesh boundaries slide along the wall: a normal-only (slip) mesh condition replaces the Dirichlet-0 pinning in the fitted decks.
  - Fitted `SurfaceStress` (Laplace–Beltrami on the moving boundary) is enabled behind an explicit flag. It is currently rejected as `fitted_surface_stress_current_frame_gradient_unqualified`.
  - Fitted `CurvatureTraction` with pointwise curvature is withdrawn from use, because that curvature is identically zero on P1 faces.
  - Order of work, as in M5:
    - fitted 2D gravity sloshing, including the 05-26 step-12 failure;
    - a static drop and a capillary wave;
    - a 2D contact point with the D4 Young term and slip.
  - Under P1, prefer the parameter-free fitted choices: the `Free` tangential policy and harmonic mesh motion. Any mesh-motion or kinematic-enforcement coefficient that remains (Nitsche or penalty) is fixed once from dimensional scaling, never tuned per case.
- **D6, 2026-09-29: the lightweight process is adopted (§12).**
  - Evidence for a result is:
    - the commit hash;
    - a short benchmark script with its tolerance file under `tests/cases/fluid/free_surface_benchmarks/`;
    - one results row in this tracker;
    - raw outputs in group storage for accepted results.
  - Investigations are time-boxed to about 3 working days before a method decision is recorded in §4.
  - Retired: the frozen/immutable campaign process (FROZEN_BEFORE_EXECUTION matrices, reciprocal hashes, no-rerun rules, a new version per harness failure), source, index and cache guards, hash-only protected files, multi-worker packet rules, and checkpoint prose in audit documents.
  - Kept: the author/committer identity (Zachary Sexton), the commit-message vocabulary scan, and the begin/end/fail job-mail settings.
  - Existing frozen matrices, runners and qualification records are left unchanged as history.
- **D7, 2026-09-29: the refactor is paused.**
  - Packages R3–R12 of `free_surface_architecture_refactoring_plan_20260904.md` are not scheduled.
  - Code is extracted only when a milestone already touches it, using that plan's target architecture as guidance.
  - Revisit once tiers T1–T3 are working.
- **D8, 2026-09-29: the September WP-4 conditioning work is stopped.**
  - The WP-4 worktree's uncommitted work is archived, not integrated, in `/home/groups/amarsden/zsexton/svMultiPhysics-archives/wp4-worktree-20260929/`:
    - 51 modified files as `tracked-changes.patch`, including the formerly protected curvature-projection test;
    - 5 untracked files as `untracked-files.tar.gz`;
    - the `.superpowers/` coordination history;
    - `README.txt` and `SHA256SUMS`.
  - Committed `fc565279` is unchanged.
  - The two suspended WP-4 plan documents were removed (§10.3).

---

## 5. Remaining work

The ordering assumes D1–D8 are accepted as recommended. If a decision goes the other way, adjust the affected milestone and note it here.

Tolerances marked "proposal" are the working acceptance criteria under D1. Confirm or adjust each once, before first use, with a one-line justification in this file.

### M0 — Decisions and housekeeping (about 1–2 days)

- [x] Record the D1–D8 outcomes in §4. All eight decisions and principle P1 were recorded on 2026-09-29.
- [x] Reconcile the home checkout (2026-09-29).
  - Its 53 modified files were committed. The Forms/JIT/Assembly/FESystem changes became `b9f59552` ("Add side-selected cut-adjacent facet integrals"). The refactoring-plan edits were already upstream and were dropped during the rebase.
  - The branch was rebased onto `fc56527`. The redundant local commit `09976e2` was skipped because it is patch-identical to `2768303`.
  - Syntax-only compilation of all 36 affected translation units passed with the full FE/Physics/Application configuration.
- [x] Export the W worktree's uncommitted diff to `/home/groups/amarsden/zsexton/svMultiPhysics-archives/wp4-worktree-20260929/` (2026-09-29, D8).
- [x] Retire the W worktree (2026-09-29). The archive checksums were verified, and `git worktree remove --force` removed `/scratch/users/zsexton/wp4-application-regression-fixes-20260902`. Four older WP-4 scratch worktrees remain attached to the same scratch repository; they expire with the scratch purge.
- [x] Archive the scratch evidence (2026-09-29).
  - The results-only set (28,665 JSON, XML, Markdown, log, CSV/TSV and script files, about 1.2 GB) from the free-surface scratch directories of 2026-08-18 to 09-21 is in one tarball on Oak (172 MB): `/oak/stanford/groups/amarsden/zsexton/svMultiPhysics-archives/free-surface-scratch-results-20260929/`, with `README.txt`, `file-list.txt` and `SHA256SUMS`.
  - Builds, binaries, source worktrees, caches and field output were left to the scratch purge (from about 2026-11-16).
  - The July level-set review outputs were no longer in scratch.
- [x] Remove the 26 superseded documents (§10.3) and repoint code comments and remaining documents to this tracker (2026-09-29).
- [x] Fix the stale statements listed in §10.4 (2026-09-29).
  - `NavierStokesFreeSurface.md` and `LevelSet.md` were corrected against the current code.
  - Both now also state that unfitted surface tension requires `Geometry_tangent_policy=RefreshedFrozenQuadrature`, which is not the active-cut default.
- [x] Create `tests/cases/fluid/free_surface_benchmarks/` (2026-09-29). Its README defines the layout (`generate_case.py`, `tolerances.json`, `verify.py`), the tolerance-file format, and the P1, D1 and D3 rules. It also lists the planned benchmarks.

### M1 — Tier-1 baseline on the current tip (unfitted, γ = 0)

- [ ] **Baseline build and C++ suites at `fef0d02f`.** Submitted 2026-09-29 as Slurm job `45961287`: 8 CPUs, 32 GB, 12 h.
  - Source: a clean worktree at `/scratch/users/zsexton/svmp-baseline-fef0d02f/source`, with all 955 LFS files present.
  - Configuration: the September GCC 12.4.0 / OpenMPI 4.1.2 / LLVM 17.0.6 / VTK 9.4.1 stack, plus Boost 1.90.0 headers.
  - Suites: CTest FE (32 tests), Physics (7) and Application (4).
  - Script and logs: `jobs/` and `logs/` in the same directory; `logs/summary.txt` has the per-suite exit codes.
  - Record the outcome here and in §8.
  - **Outcome, 2026-09-29: FAILED on a filesystem error, not a code error.** Configure passed. At 16:12 PDT (23:12 UTC) on `sh03-08n20`, at 43% of the build, the compiler got `Input/output error` closing `build/Source/solver/FE/CMakeFiles/svfe.dir/LevelSet/LevelSetStaticCapillaryEquilibrium.cpp.o.d`, which was left with 0 bytes.
  - This was the only I/O error in the log. The job was not resubmitted pending a user decision: resubmit, or report to SRCC first.
  - **Resubmitted as `45968323` (user decision).**
    - The build completed in 930 s, and **the FE suite passed 32/32**.
    - The job was then cancelled during the Physics suite because of the MPI launch problem below. Its logs are in `logs/run-45968323/`.
  - **Physics and Application suites:** tests-only job `45975100` (`jobs/tests_only.sbatch`, submitted with `sbatch --export=NONE`). CTest runs with the Slurm/PMI variables removed.
    - A `tests` symlink beside `build/` points at the source fixtures. Some Application tests search upward from their working directory for `tests/cases/fluid/open_vessel_free_surface`.
  - **MPI launch rule (found 2026-09-29).** Jobs submitted from inside the interactive `sh_dev` session inherit that session's `srun` PMI contact variables.
    - An MPI binary started without `mpiexec` in such a job contacts the interactive `srun`, which prints `PMK_KVS_Barrier task count inconsistent` in the user's terminal, and then hangs.
    - Verified in test jobs `45973957` and `45974084`: bare launches fail; `mpiexec -n 1`, `srun --mpi=pmix --exact`, and bare launches with the Slurm/PMI variables removed all work.
    - Rule: submit with `--export=NONE` and launch MPI programs through `mpiexec` (benchmarks README).

- [ ] **Tank at rest (2D and 3D).** Take a small subset of the existing hydrostatic matrix into CTest. The full 960-case matrix does not need to run routinely.
- [ ] **2D linear sloshing** at 3 meshes and 3 time steps. Compare frequency and damping with linear theory. Proposal: frequency error ≤ 1% at the finest mesh with observed convergence; volume drift ≤ 1e-4 over the run.
- [ ] **SPHERIC 05 D18/D38** to t = 0.3 s and then further. Compare profiles with experiment. Proposal: no regression from the June RMSE of about 0.02 m, and no false wetting.
- [ ] **Re-run SPHERIC 10 and 02** on the current tip. They were never re-run after the aggregation, sparsity and penalty fixes. Record whether the June pressure-spike and sliver-cut failures persist.
- [ ] **MMS traveling interface** refinement (the June matrix) as a regression.
- Reuse: the `open_vessel_free_surface` cases and verifiers; the WP-6 conservative transport (run its frozen 18-point matrix only if volume drift is the failure).

### M2 — Capillary force route comparison and static drop (unfitted, 2D then 3D)

- [x] **Benchmark written (2026-09-29):** `tests/cases/fluid/free_surface_benchmarks/static_drop_2d/`, with `generate_case.py`, `tolerances.json`, `verify.py`, a README and 10 synthetic-data tests.
  - Setup: box `[0,3]²`, R = 1, a fixed off-grid centre to avoid vertex touches, and ρ = γ = 1.
  - Laplace numbers 12 (primary) and 120, over 5 viscous times.
  - Time step `Δt ≤ sqrt(ρh³/(4πγ))`, the capillary limit for a one-sided free surface.
  - A dry run of all 24 cases validates against the parser.
  - A smoke run at R/h = 8 with `SurfaceStress` (Sept 4 binary) accepted 5 of 5 steps, with pressure-jump error 0.14% and area drift 1.5e-7.
  - Level-set advection uses the coupled velocity. Check this once against the wet-extension map at R/h = 8.
- [ ] **Per-step cost must come down before the refinement study (added 2026-09-29).**
  - The smoke run took 5.5 s per step on a 625-vertex 2D mesh. Each step needed 9–10 outer geometry passes against a cap of 12, driven by the 1e-10 absolute level-set gate, and wrote about 0.35 MB of log.
  - Extrapolated, R/h = 32 at La = 12 needs about 6 days, and R/h = 64 needs weeks to months. The same cost limits M1, M3 and M4.
  - Profile one R/h = 8 step, then reduce the unnecessary outer passes and per-step output. Any convergence gate must be scaled or derived rather than tuned (P1).
  - Target: the La = 12 study at R/h = 8/16/32 completes within about a day.
- [ ] **Protocol.**
  - Static drop in a box, fluid initially at rest. The Laplace number La = ργD/μ² is swept over 12 and 120; 1,200 and above are deferred until the per-step cost is reduced.
  - Start from the sampled analytic shape. No minimizer is required (D3).
  - R/h = 8, 16, 32, 64 in 2D.
  - Run to at least 5–10 viscous times `ρR²/μ` and report `max|u|(t)`.
  - At the end state report the pressure jump (mean interior minus exterior), volume and shape.
- [ ] **Candidates** (all without tunable parameters, per P1):
  - (a) **`SurfaceStress`** (the current default).
  - (b) **KAG with a lumped trace mass.** Set `κ_i = −g_i / m_i`, where `g` is the φ-gradient of the surface-plus-Young energy and `m_i` is the row sum of `M`.
    - The code already computes `m_i` as `lumped_kinematic_mass`, and it equals ± the discrete liquid-volume derivative (`FE/LevelSet/LevelSetCurvatureProjection.cpp` about L3321–3330).
    - At a discrete constrained stationary point, `g = μ ∂V/∂φ`, so `κ_i` is exactly constant. The exact balance by a constant pressure is therefore kept.
    - The ill-conditioned solve becomes a diagonal division.
    - Implementation: a mass-mode option in the curvature-projection options that bypasses the PCG/LSQR solve.
    - To check: noise in `κ_i` at nodes with very small `m_i`, and its effect on spurious currents.
    - **Implemented 2026-09-29 on local branch `dev/lumped-kag-mass`** (commits `0e990caf` and `04e911d1`; not yet merged). Worktree: `/scratch/users/zsexton/svmp-dev-lumped-kag/source`.
      - Option `Curvature_projection_kinematic_area_gradient_mass` = `Consistent` | `Lumped`. Lumped requires filter 0.
      - Tests: full `test_fe_levelset` 371 passed, 1 declared skip, 0 failed; `test_fe_levelset_mpi` 21/21, with lumped matching serial bitwise on 2 ranks; `test_application` 376/376.
      - Curvature quality on a sampled circle (n = 12/24/48/96):
        - The mass-weighted mean converges at second order, the same as consistent mode.
        - The trace error stays near 5.5% of 1/R and does not converge.
        - Nodal errors at vertices with a sliver of interface in their support grow to 23–347.
        - Filtered consistent mode (`c_l = 1`) does converge, with trace error 0.044 → 0.0022.
      - The M2 static drop will show whether the lumped noise shows up as spurious currents.
  - (c) **Unfiltered consistent-mass KAG**, for reference only.
  - **Fallback only if (a) and (b) both fail the D1 criteria:** KAG with the Helmholtz filter or a normal-gradient-stabilized mass. Its coefficient must be fixed once from dimensional scaling (P1), never tuned.
- [ ] **Time step.** Check Δt against the capillary constraint (§3.6). If needed, add a semi-implicit surface-tension term (Bänsch/Hysing) or use Δt below the limit.
- [x] **Acceptance** (working criteria; fixed in `static_drop_2d/tolerances.json` on 2026-09-29):
  - pressure-jump error ≤ 1% at R/h = 32 with observed order ≥ 1;
  - final `Ca_sp` strictly decreasing with h, with absolute values reported. There is no absolute limit (decided 2026-09-29), because the July speed/γ figure was a start-up transient in SI units, not a capillary number;
  - no growth of `max|u|` in time;
  - volume drift ≤ 1e-4.
- [ ] **Selection.** After the M2 static drop, the M3 capillary wave, and the M4 sessile and Ren–E comparisons, record the chosen default route and the reason in §4.
  - Among candidates meeting the D1 criteria, prefer the one without tunable parameters (P1).
  - Then run the 3D sphere at R/h = 8, 16, 32.
- Optional accuracy improvement if spurious currents dominate: the Gross–Reusken improved Laplace–Beltrami (the projection uses a recovered, smoother normal).

### M3 — Dynamic capillarity

- [ ] **Capillary wave** (Prosperetti) at λ/h = 16, 32, 64 and three time steps. Proposal: frequency error ≤ 2% and damping error ≤ 5% at λ/h = 32 (the 07-17 n = 16 run already had 1.2% frequency error).
- [ ] **Oscillating 2D drop**: Lamb frequency and viscous damping.

### M4 — Wetting (unfitted)

- [ ] **Implement or confirm the D4 configuration** (decided 2026-09-29):
  - Young term;
  - Navier slip on the wetted wall;
  - strong no-penetration;
  - scale-only wall maintenance. The dynamic-contact path already has this; enable it for `PrescribedAngle`, which today applies the repair to the target angle.
  - no repair-to-target.
- [ ] **2D sessile relaxation** at 60°, 90° and 120°. Start from a *non-equilibrium* shape (for example a 90° cap for a 60° target) and relax to equilibrium at R/h = 16, 32, 64. Proposal: angle error ≤ 2° at R/h = 32 and decreasing; base radius and apex height within 2%.
- [ ] **Capillary rise** against the prepared Gründing et al. envelope, using `free_surface_wp5_capillary_rise_reference.json` and the comparison runner.
- [ ] **Ren–E** advancing and receding: refine the 08-30 pilot at 3 meshes and 3 time steps, with slip length ratio ℓ_s/h = 2, 4, 8.
- [ ] **3D sessile** at one angle, then extend.

### M5 — Fitted ALE reference path

- [ ] Use slip (normal-only) mesh BCs on walls (D5). The current fitted SPHERIC 10 deck pins walls with Dirichlet-0.
- [ ] **2D fitted sloshing**, gravity only. Reproduce the 05-26 step-12 failure, fix it, then compare with linear theory.
- [ ] Allow fitted `SurfaceStress` behind a flag (today it is rejected as `fitted_surface_stress_current_frame_gradient_unqualified`). Then run the static drop and the capillary wave. Remove fitted `CurvatureTraction` with pointwise curvature from use: it is identically zero on P1 faces.
- [ ] **2D contact point** (a codimension-2 point) with Young term and slip. Compare with the static meniscus.

### M6 — Violent and long-horizon one-phase flows (T1/T5)

- [ ] SPHERIC 10 to its 7.3 s peak; SPHERIC 02; D18/D38 to the full horizon.
- These pull in the open WP-6 items (conservative transport, no global shift) and WP-7 items (sliver cuts in line-search trial states, pressure-row support).

### M7 — Two-fluid (T4)

- [ ] Handle exact-node topology events (step-size reduction or an event restart). This is the current static-drop blocker.
- [ ] Static drop with both phases, reusing the M2 protocol.
- [ ] Hysing rising bubble, cases 1 and 2.

### M8 — Optional research track (not on the capability critical path)

- [ ] Exact discrete energy law for moving interfaces (WP-8 physical campaigns).
- [ ] An energy-consistent KAG pulled back through the actual level-set transport operator.
- [ ] Discrete static minimizer with nonsmooth optimization.
- [ ] φ-scale invariance to roundoff.
- [ ] Publication-grade frozen qualification records for completed capabilities.

---

## 6. Formulation reference (current code)

Paths are relative to `Code/Source/solver/`, at `fc56527`.

### 6.1 Unfitted one-phase

**Active domain.**
- Liquid is `{φ < 0}`, or `{φ > 0}` if selected. Volume terms use `dCutVolume(marker, side)`; interface terms use `dI(marker)` on the generated `LinearCorner` polygon Γh.
- The generated normal comes from the same rule, oriented outward from the liquid.

**Spaces and stabilization.**
- Continuous P1/P1 with VMS/PSPG (incremental PSPG variant available).
- Small-cut aggregation (AgFEM-type affine constraints on u and p; guards: path ≤ 8, extrapolation ≤ 4, |coefficient| ≤ 16, row L1 ≤ 32). Rootless features are deleted.
- Pressure-gradient facet jump `γ_p · 0.01 · h³ / (μ + ρh²/Δt)` on cut-adjacent facets.
- The velocity ghost penalty was removed on 2026-06-12.
- Inactive pressure is pinned to 0; an optional PDE velocity extension acts on the dry side.
- Sources: `free_surface_wp7_combined_p1_method.md`; `Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceOptions.h`.

**Free-surface traction** (`IncompressibleNavierStokesVMSModule.cpp` about L6681–6863). Every route adds `∫_Γh p_ext n_h·v`.

| Form | Capillary term | Status |
|---|---|---|
| `SurfaceStress` (unfitted default) | `γ ∫_Γh (I − n_h⊗n_h) : ∇v`; wall line term `−γ cos θ_e ∫_CL v·m` | Production default |
| `KinematicAreaGradientTraction` | `γ ∫_Γh κ_h n_h·v`, with `κ_h` a prescribed P1 field from KAG recovery; no separate line force (Young is inside `κ_h`) | WP-4 selected route; requires interface quadrature order ≥ 2 in 3D |
| `GeneratedCurvatureTraction` | `∫ (p_ext + γκ) n_h·v`, with κ projected | Experimental |
| `CurvatureTraction` (legacy) | Same integrand, with `n = ∇φ/|∇φ|` | Legacy; fitted-only use today |

- Geometry is `RefreshedFrozenQuadrature`: the capillary terms contribute no geometry Jacobian, and coupling is through outer fixed-point refreshes.

**Curvature recovery** (`FE/LevelSet/LevelSetCurvatureProjection.*`) has three modes:
- `LevelSetQuadratic` (least-squares quadratic fit of φ);
- `GeneratedInterfacePatch` (fit to the interface point cloud);
- `KinematicAreaGradient`: analytic `∂A/∂φ` plus Young-wall `∂A_sl/∂φ`, the trace mass `M`, an optional Helmholtz filter `ℓ = c √(h R)`, and a PCG solve with an LSQR fallback.

**Pressure diagnostics** (`FE/TimeStepping/NewtonSolver.cpp`).
- LSQR distance of the capillary load to `range(G)`, and a constant-pressure KKT fit.
- They are diagnostic, an optional acceptance gate, or a one-shot pressure initial guess. Force projection is never applied.

**Static initializer** (`ApplicationDriver.cpp::initializeDiscreteStaticCapillaryEquilibrium`; `FE/LevelSet/LevelSetStaticCapillaryEquilibrium.*`).
- An SQP/L-BFGS method with an l1 merit function, working over nodal φ at fixed volume.
- It allows bounded topology-epoch transitions and ends with a KKT certificate.

**Contact lines.**
- `PrescribedAngle`: geometric wall repair at accepted endpoints (`FE/LevelSet/LevelSetReinitialization.*`), plus the route-dependent Young term, with optional slip.
- `DynamicRenE`: line friction `ξ u·m` plus the route-appropriate Young term, sharp Navier slip on the wetted wall, and a strong normal-only wall constraint.
- The literal level-set angle penalty was retired, and code asserts it is absent.

**Level set.**
- Transport: Galerkin + SUPG, optional discontinuity capturing, a bound gate with reject/retry.
- Reinitialization: projection-only and wall-aware. There is no Hamilton–Jacobi or fast-marching method.
- Optional conservative P1 indicator with FCT and reconciliation (backward Euler).
- Geometry is coupled by an outer fixed point that accepts a step only when a fresh pass needs zero Newton updates.

**Time and solvers.**
- Generalized-α (first order) or backward Euler.
- Newton with Armijo line search.
- Eigen direct (serial), or FSILS GMRES / NS block-Schur.

### 6.2 Fitted ALE

- The unknowns are the fluid velocity plus a coupled mesh displacement `d_m`.
- Normal kinematics `(ḋ_m − u)·n = 0` are enforced by penalty or Nitsche.
- Tangential mesh policy is `Free`, `SmoothingOnly` or `Prescribed`.
- Interior mesh motion is harmonic or pseudo-elastic.
- Allowed capillarity: `CurvatureTraction` with pointwise current-geometry curvature. This curvature is zero on P1 faces.
- Rejected: `SurfaceStress`, `PrescribedAngle` and `DynamicRenE`.
- Source: `free_surface_wp9_fitted_ale_architecture.md`.

### 6.3 Two-fluid

- Separate P1 velocity and pressure per phase; a sharp pressure jump.
- A weighted symmetric Nitsche interface.
- Surface tension as the area variation with `avg_c(v)`.
- Phase-local stabilization and aggregation.
- Fixed BlockSchur field order.
- Source: `free_surface_wp10_two_fluid_method.md`.

### 6.4 Known formulation issues (open)

- Two geometry definitions (the snapshot versus the recovery/minimizer strict cut). See §3.2.
- An unstabilized KAG trace mass. See §3.2.
- Two owners of the prescribed angle; P1 repair targets are not representable (§3.5). Resolved by D4 in favour of the Young term; the implementation is in M4.
- Explicit-in-geometry capillarity, with no semi-implicit term. See §3.6.
- Weak walls cannot hold a static line force; contact walls need strong normal constraints.
- Two liquid-volume measures coexist: the lumped nodal `M_q` from conservative transport and the sharp cut `V_h`. Reconciliation must preserve the target volume.
- Rootless-feature deletion silently removes liquid. It is recorded as an event, but it is not conservative.
- About 110 environment-variable hooks in `NewtonSolver.cpp`, and more in the NS module and driver (the local-commit audit counted 174 `SVMP_*` variables). Several change residuals.

---

## 7. Legacy tracking IDs and their disposition

The 2026-07-20 audit defines FSR findings, AD decisions, WP work packages and Q gates. They stay as reference IDs. **Progress is tracked by the M-milestones in §5.**

| ID | Topic | Legacy status | New home |
|---|---|---|---|
| WP-0 / FSR-10, 11, 12 | Configuration containment | Closed (FSR-10/11 fitted parts ambiguous) | M5 (fitted) |
| WP-1 / FSR-01, 02, 18 | Extension, dry feedback | Closed; long-horizon D38 not shown | M6 |
| WP-2 / FSR-13, 14, 15, 17 | Authoritative geometry | Closed; "one geometry" later violated by recovery and minimizer | M2 (single geometry definition) |
| WP-3 / FSR-16 | Sharp exterior BCs | Closed within P1 `LinearCorner` | — |
| WP-4 / FSR-03 | Balanced capillary force | Open | **M2** (route selection and static drop), M3 |
| WP-4, WP-5 / FSR-04 | Prescribed angle | Open | **M4** (under D4) |
| WP-5 / FSR-05 | Dynamic wetting, capillary rise | Open (pilot only) | M4 |
| WP-6 / FSR-06 | Conservative transport | Open (matrix frozen, not run) | M1, M6 |
| WP-7 / FSR-07 | Cut stability, inf-sup | Open (finite certificates only) | M1, M6 |
| WP-8 / FSR-09 | Discrete energy law | Open (prerequisites only) | M8 |
| WP-9 / FSR-10, 11 | Fitted ALE | Open (prerequisite only) | M5 |
| WP-10 / FSR-08 | Two-fluid, gas | Open (planar prerequisites) | M7 |
| AD-1 | Extension | De facto: bounded one-way map with convex fallback | — |
| AD-2 | Capillary force/pressure pair | De facto: unfiltered KAG (09-02). **Reopened** by D2. | M2 |
| AD-3 | Small-cut stability | De facto: aggregation + pressure facet jump | — |
| AD-4 | Conservative interface | Conservative P1 indicator + FCT + reconciliation | M1, M6 |
| AD-5 | Geometry/nonlinear coupling | Outer fixed point (no shape tangent) | M2 (time step), M8 |
| AD-6 | Physical capability staging | One-phase → two-fluid → gas | M7 |
| Q0–Q7 | Qualification gates | All open | Replaced by the per-milestone acceptance in §5. Q-gate text remains the reference for publication-grade studies (M8). |

---

## 8. Log of what has been tried

This section is compact and chronological. Detailed evidence is in the indexed documents (§10) and the checkpoints (§9).

**Dispositions:**
- **KEPT**: the change is in the code today.
- **DROPPED**: tried and removed.
- **OPEN**: unresolved.
- **SUPERSEDED**: replaced by a later approach.

### 8.1 Bring-up of the unfitted level set (2026-05-12 to 05-26)

- **05-12 to 05-16: foundations.**
  - Level-set services migrated from Physics to `FE/LevelSet` (phases 1–16). KEPT.
  - High-order implicit cut quadrature (`HighOrderImplicit`: Saye / HighOrderSubcell) added as opt-in. It is experimental; its production-completion plan has 104 open items.
- **05-17: linear 2D sloshing accepted.** Interface L2 8.9e-5, velocity relative error 1.85e-2, area error 1.3e-5.
- **05-18: "official" MMS record later invalidated.** A static source frozen at t = 0 had caused a −319 Pa offset.
- **05-22 to 05-26: qualification-log campaign.**
  - **High-order MMS.** A Saye fallback at depth 8 was fixed with midpoint hints. A pressure residual of 7.9e10 came from the pressure ghost penalty amplified by the metadata scale; fixed by policy.
  - **MMS pressure spaces.** Equal-order P1/P1 fails (relative RMS 0.15–0.51); Taylor–Hood P2/P1 passes (0.008–0.035).
  - **Square tank tilt.** Fails at step 167. The refined case fails at t = 0.173. The interface pressure error localizes at the contact-line endpoints (677 Pa vs 199 Pa). The verifier was weakened to "settling" and "core" profiles to pass: OPEN.
  - **Capillary sign (F1).** Fixed.
  - **D18/D38.** First profile at t = 0.156 passes (RMSE 0.022 / 0.018 m) at about 10 s per step.
  - The remediation outline (F1–F5, workstreams A–G) was written and its checklist ticked, but W2/W3 remained open. SUPERSEDED.
- **05-26: ordered sweep.** Cases 01, 02 and 04 pass. 03 (MMS stall), 05 (Test10 unfitted), 06 (Test10 **fitted ALE**, failing at step 12), 07 and 08 (D18/D38 FSILS residual) fail.

### 8.2 Open-vessel robustness (2026-06-02 to 06-12)

- **06-02 to 06-04: SPHERIC runs.**
  - **Test05** passes with cadence-1 volume correction (RMSE 0.0184 / 0.0158 m).
  - **Test10.** Every long run dies near 0.89–0.90 s: line-search *trial* cut fractions reach about 1e-8, while accepted states are clean. The accepted Sensor1 pressure is 667–1078 Pa against a 341 Pa reference. PTC, ρ_∞, pruning and metadata caps all fail.
  - **Test02.** The wall `Effective_direction` bug and the `nearest_interface_point` extension bug were fixed (KEPT). The front is still slow (0.43 of the reference speed), and MPa spikes appear at tiny cuts.
- **06-05 to 06-07: pressure-jump root-cause campaign (about 40 diagnostics).**
  - Jumps sit on full-wet boundary rows and are residual-consistent.
  - The cause is **not** the ghost penalty, the gauge, the rebuild, trace support, Δt or shape tangents.
  - Tried and DROPPED: clamps; PSPG scale 0 and 10; wall and flux PSPG variants; about 15 graph-completion and Schur modes.
  - KEPT: the trace-only support fix and an optional pre-commit pressure-update guard.
  - OPEN: "direct PSPG pressure-gradient support topology". Never re-examined after 06-12.
- **06-10: fixes and first passing MMS matrix.**
  - A D18/D38 step-0 regression came from the transient ghost-penalty scaling; the 0.01 calibration was KEPT.
  - D18 false wetting after 0.22 s at lid corners: OPEN.
  - The MMS refinement matrix passes for the first time (constraint history, reinitialization DOF binding, metadata aggregation, dry down-weighting, band-preserving reinitialization, trial synchronization). Interface pressure converges at second order.
- **06-11 to 06-12: stabilization overhaul.**
  - The eigenvalue-calibrated ghost penalty failed (coefficients 1e5–5e9) and was DROPPED.
  - Small-cut aggregation for u and p became the default (KEPT), and the velocity ghost penalty was deleted.
  - Found and fixed: cut volumes never contributed to sparsity and Eigen silently dropped writes; the MPC state distribution; a stale slave-DOF span; island pins.
  - Aggregation converges (h-order 1.73 to nx24). nx32 aborts in Saye on corner-degenerate cuts: OPEN.
- **Gap:** SPHERIC 10 and 02 were never re-run after 06-12.

### 8.3 Review and audit (2026-07-13 to 07-28)

- **07-13 to 07-17: level-set review (FS-01 to FS-16).**
  - Fixed: the contact residual on an unsolved operator; the codimension-2 frames; sign errors (`n·n_w = −cos θ`; capillary traction `−(p_ext + γκ)n`); acceptance of ambiguous intersections.
  - SurfaceStress became the default.
  - **SurfaceStress closed drop:** pressure error 3.15% / 0.32% / 0.0067% at n = 8/16/32; speed/γ 2.6e-5, 1.2e-5, 2.5e-5 (gate 1e-5). **Projected** curvature error worsens with refinement (0.30 → 0.35 → 1.53).
  - **Sessile (Q1 mesh):** all n = 32 rows pass per-mesh gates. The n = 16 90° and 120° angle errors are 6.09° and 6.49° (gate 5°). Refinement rates fail.
  - **Capillary wave n = 16:** frequency error 1.2%, profile error 1.5%, volume drift 9.26e-7 after the trace-support fix (a 402× reduction).
  - Force-balance probe: cosine −0.954, normalized imbalance 0.153 (became FSR-03).
- **07-20: master audit.**
  - Defines FSR-01 to FSR-18, AD-1 to AD-6, WP-0 to WP-10, Q0 to Q7, a closure rule (six artifacts per box) and a 12-rule acceptance protocol.
  - The local-commit architecture audit (ARC-01 to ARC-15) says "request architectural cleanup before merge". It was never dispositioned.
- **07-20 to 07-22: first closures.**
  - WP-0 PASS (24 tests); WP-1 PASS (53 tests + Enright and drop exits).
  - WP-2: v1 FAIL_METHOD → v2 PASS → v4 PASS (200 tests, 130 checks).
- **07-23 to 07-28:** WP-3 v2 low-level PASS; Nitsche coercivity v1 (108 cases); WP-9 and WP-10 boundary documents; Q0 harness.
- No commits between 07-29 and 08-17.

### 8.4 Prerequisites for the other work packages (2026-08-18 to 09-02)

- **08-18:**
  - Bulk import (809 files).
  - Nitsche v2: exact dyadic aggregate-trace certificate, max C = 1.39 against α = 12.
  - Discrete-energy method document: SurfaceStress + Young + multiplier pressure, with a constrained-minimization initializer.
- **08-24 to 08-26:**
  - Nitsche v3 accepted-state floor c* = 1/4.
  - WP-3 V3, V4 and V5 rejected for *harness* reasons (MPI discovery, `ldd` under RLIMIT_AS, inherited `SLURM_NPROCS`). V6 PASS closed WP-3 within P1 `LinearCorner`.
- **08-30:**
  - WP-5: non-endpoint maintenance publication; capillary-rise plumbing (33 tests); JIT cache-key bug fixed (KEPT).
  - Ren–E pilot: KAG has the correct sign (error 0.25–0.44); SurfaceStress advancing errors 128.66 and 17.38 (wrong sign).
  - WP-8 residual-work, rejected-attempt and complete-connector prerequisites.
  - WP-9 fitted-ALE prerequisite (32 tests).
- **08-31:** WP-6 prerequisite v2 (59 tests). The Enright 64³ point is INCONCLUSIVE_RESOLUTION.
- **09-01:**
  - WP-10 planar prerequisites all PASS: constant state; pressure jump (error 8.9e-16); viscous jump; hydrostatic with density ratio up to 1e4.
  - Two-fluid static drop rejected by an exact-node `CutTopologyChanged` event.
- **09-02:** WP-8 exact-node topology detection and rollback PASS (prerequisite).

### 8.5 WP-4 in detail (2026-08-25 to 09-21)

**Flat and hydrostatic balance (08-25 to 08-27). KEPT; effectively complete.**
- Physical flat equilibrium, serial: 12 cases (2 directions × 2 wall families × 2 signs × 3 offsets). KKT residual 2.2e-16.
- Two-rank MPI version; fixture ownership repairs.
- Gravity and fixed-gauge pressure certificate; direct-residual LSQR pressure correction; LSQR stationarity target tightened 1e-10 → 1e-12.
- Matrix expanded 24 → 60 (3D) → 384 (layouts, numberings, ownership) → **960** two-rank 2D/3D cases. Maximum residuals are about 5e-11 to 7.8e-16.
- Golub–Kahan local reorthogonalization kept residual refinement inside the LSQR budget.

**Curvature routes (08-28 to 08-30).**
- **`GeneratedCurvatureTraction` feasibility.**
  - Manufactured discrete contact geometry is exact to roundoff.
  - A sampled 90° cap at n = 8 gives angle error 12.6°, pressure error 8.3%, Ca 4.4e-3.
  - Status: experimental.
- **`GeneratedInterfacePatch` recovery.**
  - Circle curvature error about second order (2.38 → 0.025 over n = 8 to 64).
  - Sessile 90° pressure error 59.6% / 17.6% / 4.74% at n = 8/16/32; Ca 1.3e-3 to 4.4e-3. With 20 smoothing iterations Ca is 1.5e-4 to 3.0e-4, but nonmonotone.
  - Status: KEPT as an option.
- **KAG (variational kinematic area gradient)** with Young walls.
  - At 60° and 120°, mass-weighted RMS curvature errors are 0.064 / 0.038 / 0.017 and 0.037 / 0.019 / 0.0098 at res 32/64/128 (order about 0.8–1.1).
  - The distributed version matches serial bit for bit.
  - Status: KEPT.
- **08-29 to 08-30.**
  - KAG connected to momentum (`KinematicAreaGradientTraction`).
  - Static initializer uses exact functional derivatives.
  - V2 matrix frozen, then found defective and **never launched**: it requests zero FD components, its scaling lanes do not redistance, and its time refinement changes the horizon.
  - Preserved sphere at R/h = 2.1: pressure error 5.06%.

**Method freeze (09-02 to 09-03).**
- Contract review: keep **unfiltered** KAG plus the fixed-volume minimizer; no enrichment or projection. The identified gap was degree-2 planar polygon quadrature.
- Prescribed-angle review:
  - the penalty is retired and the Young term has one owner;
  - open items: scheduling tied to bulk redistancing, absolute tolerances breaking φ-scale invariance, a non-representable curved 3D strip, stage semantics, no fixed-point study.
- Plan written (`plan_wp4_balanced_force_completion_20260903.md`; removed, retrieve from commit `5b46da55`).
- Implemented: quadratic planar polygon quadrature, order admission, a production adjoint test (1e-12 on one tetrahedron), restoring-force and MPI-parity tests.
- V3 frozen: 2,136 cases.

**Qualification attempt and pilots (09-04).**
- The V3 exact job timed out after 4 h 15 min (a launch issue).
- The physical pilot failed:
  - a sampled circle hit the global 1e-8 gate (0.1397);
  - the radius declared as 0.2 reached the generator as 0.3 (bug fixed);
  - an unsupported FSILS direct request.
- Corrected sampled circles at R/h = 8/16/32:
  - from zero pressure, the pressure jump is 10.71 / 10.02 / 9.69 against the analytic 5;
  - with the analytic initial pressure it is 6.33 / 5.65 / 5.31 (6.2% at the finest level);
  - **measured after one step.**
- Discrete-minimum circle: 64 topology transitions exhausted; a cap-128 repeat failed at 104 transitions with `‖g_proj‖ = 0.0237`.
- A constant-pressure certificate false negative was found and fixed (`1fe82be`).
- Line-search trace instrumentation added.
- **Derivative mismatch** identified: the recovery uses a strict tie-broken cut, while the snapshot uses zero-band and pruned geometry (Triangle3 witness: action −0.7071 vs secant 0).

**Certification machinery (09-05 to 09-07).**
- Producer integer interval arithmetic, "dyadic bounds" and "compact integer brackets". This is analysis tooling that changes no geometry; a sphere test slowed from 0.067 s to 206 s, then to 18 s.
- Coefficient-classification policy.
- Prescribed-angle working-coefficient normalization (`d2b3326`).
- Derivative binding (`fc5e61b`; the overload is a stub).
- Captured-state C9 acceptance.
- **Minimizer job 42294456: FAIL** (see §2.2).

**Conditioning (09-14 to 09-21).**
- Goal document written.
- Quotient condition about 3.4e14.
- Rounding-search publication experiments failed on trial 39: best scaled residual 1.07e-10 to 1.22e-10 against 1e-10.
- User authorized a double-double curvature representation. `fc56527` stores paired prescribed coefficients; the native pair passes 19 private tests (not integrated).
- V3 edited on 09-14 (`AWAITING_SCIENTIFIC_CONTRACTS`).
- Work paused at the usage floor. OPEN, and recommended to stop (D8).

### 8.6 Architecture refactor (2026-09-04 to 09-05; dormant since)

- R0: baseline manifest, capability ledger, operator and lifecycle references.
- R1: options moved to `FreeSurfaceOptions.h`, level-set configuration resolver, maintenance configuration split (−899 driver lines), cut-option conversion.
- R2: thin `FE/Forms/IntegrationDomain.h` selector.
- About 25 failed steps, all in tooling or build-environment, none numerical.
- One unmerged commit exists on local branch `free-surface-architecture-20260904` (`ae11dae`, "Extract legacy free-surface face input adapter").
- The job ledger still lists job `42161975` as RUNNING; it completed on 2026-09-05.

### 8.7 Recurring lessons

- **Converged is not correct.** Residual-consistent but unphysical pressure appeared, with a jump-to-residual ratio of about 1e6. Keep physical guards.
- **Sliver cuts cause failures.** They appear mostly in line-search *trial* states, not in accepted states.
- **Scalar tuning never fixed a structural problem.** Every real gain came from finding a bug: dropped writes, sparsity supplied by an unrelated term, aliased spans, dead knobs, and index-numbering mismatches.
- **Regime-dependent regressions.** MMS at O(1) `ρh²/(μΔt)` hid failures in the water regime.
- **The wall and contact line are the persistent weak spot:** endpoint pressure, near-wall bands, wall-row PSPG support, false wetting.
- **Harness failures can consume more time than numerical failures** (WP-3 V3–V5, WP-4 V2/V3).
- **Gates that exceed what the method can achieve create unbounded work**, because every failure looks like a new prerequisite (WP-4 since 09-04).

---

## 9. State inventory

Recorded on 2026-09-29. Re-check before acting.

| Item | Location / identity | State |
|---|---|---|
| Authoritative branch | `origin/issue-449-modern-mesh-core` at `fc56527` (2026-09-21) | Clean; latest WP-4 commits |
| Home checkout | `/home/users/zsexton/svMultiPhysics`, branch `issue-449-modern-mesh-core` | Synchronized with origin on 2026-09-29. The 53 previously uncommitted files were committed (§5 M0), and this tracker plus the document cleanup were committed on top. Still untracked, and deliberately left out: the WP-10 static-drop matrix, runner and test (`tests/cases/fluid/free_surface_wp10_static_drop_matrix.json`, `tests/cases/fluid/run_free_surface_wp10_static_drop_qualification.py`, `tests/test_free_surface_wp10_static_drop_qualification.py`) and the generated `svmp_fe_jit_dumps_tests_basis_baking/` directory. |
| WP-4 worktree "W" | `/scratch/users/zsexton/wp4-application-regression-fixes-20260902` | **Removed 2026-09-29.** Its uncommitted work is archived in `/home/groups/amarsden/zsexton/svMultiPhysics-archives/wp4-worktree-20260929/` (D8). |
| Latest WP-4 checkpoint | `/scratch/users/zsexton/wp4-continuation-20260921-szBnNq/checkpoint.md`; `W/.superpowers/sdd/.../delegation-checkpoint.md` | Paused; no live jobs |
| Other run roots | `/scratch/users/zsexton/wp4-continuation-20260906`, `wp4-conditioning-20260914-jpcnJX`, `wp4-delegation-briefs-20260905-Z6OoSG`, `free-surface-refactor-20260904-905239de` | Historical |
| Scratch volume | About 380 `wp4-*` and about 740 `wp*` / `free-surface*` entries under `/scratch/users/zsexton` | **Subject to the 90-day purge** (oldest 2026-08-25, so from about 2026-11-23). Copy anything to keep to `$GROUP_HOME` (`/home/groups/amarsden`) or Oak (`/oak/stanford/groups/amarsden`). |
| Baseline worktree | `/scratch/users/zsexton/svmp-baseline-fef0d02f/` (`source/` is a detached worktree of the home repository at `fef0d02f`; `build/`, `logs/`, `jobs/`) | Build and test baseline for M1; job `45961287` |
| Scratch results archive | `/oak/stanford/groups/amarsden/zsexton/svMultiPhysics-archives/free-surface-scratch-results-20260929/` | 28,665 result files from the Aug–Sep free-surface scratch directories (172 MB compressed) |
| Sync backups | `/scratch/users/zsexton/sync-backup-20260929/` (patches of the pre-rebase commits, the obsolete untracked capability-ledger copy, syntax-check logs); local branch `pre-sync-backup-20260929` | Temporary; delete once the pushed state is confirmed |
| Local branches | `free-surface-architecture-20260904` (`ae11dae`, 1 commit not in origin); `wp3-v5-qualification`, `wp3-v6-qualification` (merged); `issue-604`, `performance/solver-efficiency-review` (unrelated) | — |
| Reference text | Gross & Reusken, *Numerical Methods for Two-phase Incompressible Flows* (2011): `/home/users/zsexton/wp4-references` (a PDF without an extension) | Kept outside Git |
| Tooling note | System git 1.8.3 cannot operate the W worktree; use `/share/software/user/open/git/2.45.1/bin`. LFS objects need `git-lfs` 2.4.0 from `/share/software/user/open/git-lfs/2.4.0/bin`. | — |

---

## 10. Document index

**Status keys:**
- **R**: current reference. Keep it, and keep it accurate.
- **C**: case definition. Keep the definitions; ignore any status text inside.
- **H**: historical. Superseded by this tracker and removed on 2026-09-29 (§10.3).
- **S**: suspended plan, pending the §4 decisions.

### 10.1 Method and contract references (R)

| Document | Content |
|---|---|
| `Code/Source/solver/FE/Docs/LevelSet.md` | FE level-set services and contracts, including the curvature recovery modes (corrected 2026-09-29) |
| `Code/Source/solver/Physics/Docs/NavierStokesFreeSurface.md` | Navier-Stokes surface-tension forms and contact-line terms as currently implemented (corrected 2026-09-29) |
| `Documentation/free_surface_discrete_energy_balance_method.md` | Discrete surface + Young energy; KKT; static initializer rationale |
| `Documentation/free_surface_wp5_contact_line_architecture.md` | Contact-line conventions, Ren–E, prescribed-angle split, dissipation |
| `Documentation/free_surface_wp7_combined_p1_method.md` | P1/P1 + aggregation + pressure facet jump; node-crossing results |
| `Documentation/free_surface_wp8_geometry_energy_architecture.md` | Outer fixed point (AD-5); energy ledger design |
| `Documentation/free_surface_wp9_fitted_ale_architecture.md` | Fitted-ALE contract |
| `Documentation/free_surface_wp10_two_fluid_method.md`, `free_surface_wp10_physical_capability_boundary.md` | Two-fluid method and capability boundary |
| `Documentation/level_set_conservative_phase_transport.md` | AD-4 conservative indicator and FCT |
| `Documentation/free_surface_wp3_wp7_symmetric_nitsche_coercivity_method_v3.md` | Current Nitsche floor (v1 and v2 are H) |
| `Documentation/free_surface_capability_ledger.md` | Implemented-vs-qualified inventory. Keep it in sync with §2.1, or fold it into §2.1. |
| `Documentation/qualification_logs/` | Frozen evidence records (20 directories) |
| `Documentation/free_surface_architecture_refactoring_plan_20260904.md` | Refactor target architecture (R0–R12); paused under D7 and used as guidance for targeted extractions. |
| `Documentation/mesh_motion_math_first_formulation_guide.md`, `plan_ale_mesh_motion_data_and_coupled_displacement.md`, `plan_mesh_motion_math_first_formulations.md`, `plan_moving_mesh_infrastructure.md` | Mesh-motion infrastructure (R, with open items relevant to M5) |
| `Documentation/plan_high_order_curved_implicit_level_set_quadrature.md`, `plan_high_order_implicit_geometry_completion.md` | High-order geometry. Experimental; 104 open items. |

### 10.2 Case definitions (C)

- `Documentation/moving_free_surface_validation_cases.md`: case definitions are useful; the long Test10/Test02 status logs are H.
- `tests/cases/fluid/open_vessel_free_surface/**/README.md`. The MMS README still describes the deleted velocity ghost penalty.
- `tests/cases/fluid/free_surface_wp5_capillary_rise_reference.json` and the comparison runner.

### 10.3 Removed (H) and suspended (S) documents

The 26 documents marked **H** were removed on 2026-09-29. Commit `fc565279` still contains every one of them; retrieve one with, for example:

```bash
git show fc565279:Documentation/free_surface_boundary_unfitted_audit_20260720.md
```

Frozen qualification records, matrices and runners that name these paths were left unchanged; they describe the historical inputs of their runs.

| Document (under `Documentation/` unless noted) | Key | One-line summary |
|---|---|---|
| `unfitted_level_set_free_surface_qualification_log_20260522.md` | H | May bring-up log (W2/W3/W4/W6 gates); §8.1 |
| `unfitted_level_set_free_surface_bc_remediation_outline_20260526.md` | H | F1–F5 and workstreams A–G; checklist ticked but gaps remained |
| `open_vessel_free_surface_remaining_test_case_issues_20260526.md` | H | 05-26 sweep triage; all boxes unchecked and stale |
| `open_vessel_free_surface_{active_pressure_support_rank_guard, cut_context_transition_audit, linear_pressure_cut_volume_patch, newton_pressure_residual_diagnostic, pressure_constraint_coverage_diagnostic, pressure_row_contribution_diagnostic, pressure_stabilization_contribution_audit, pressure_update_guard_diagnostic, supportfix_replay_audit, test02_test10_root_cause_report, vms_pressure_path_control}_20260605.md` (11 files) | H | Test02/Test10 pressure-jump diagnostics; §8.2 |
| `d18_d38_spheric_test05_validation_root_cause_20260610.md` | H | Transient ghost-penalty calibration; profile error classes |
| `plan_ghost_penalty_eigen_calibration_20260611.md` | H | Eigen penalty failed → aggregation; sparsity bug; §8.2 |
| `plan_aggregation_band_churn_phi_coupling_investigation_20260612.md` | H | MPC state distribution, stale span, island pins |
| `plan_level_set_fe_library_migration.md` | H | Migration complete (Phase 17 partly open → M1) |
| `free_surface_level_set_review_20260713.md` | H | FS-01 to FS-16 review; source of the 07-17 static and wave numbers; §8.3 |
| `free_surface_boundary_unfitted_audit_20260720.md` | H | Master audit, 2,386 lines. FSR, AD, WP and Q definitions remain reference IDs (§7); its 85 checkpoint entries are history (§8). |
| `free_surface_local_commit_architecture_audit_20260720.md` | H | ARC-01 to ARC-15; succeeded by the refactor plan |
| `free_surface_q0_harness_provenance_architecture.md` | H | Q0 harness; eight exits open |
| `free_surface_wp3_wp7_symmetric_nitsche_coercivity_method.md`, `..._v2.md` | H | Superseded by v3 |
| `free_surface_refactor_job_ledger_20260904.md` | H | Refactor jobs (it still listed job `42161975` as running; that job completed on 2026-09-05) |
| `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/linear_sloshing_2d/LEVEL_SET_FREE_SURFACE_SUPPORT_STATUS.md` | H | 05-17 support status; stale on stabilization |
| `plan_wp4_balanced_force_completion_20260903.md` | H (removed after D2, D3 and D8; retrieve from `5b46da55`) | Unfiltered-KAG completion plan; Tasks 1–3 done, Task 4 blocked |
| `goal_wp4_conditioning_to_qualification_20260914.md` | H (removed after D2, D3 and D8; retrieve from `5b46da55`) | Conditioning/double-double goal; stopped by D8 |
| `W/Documentation/wp4_qualification_gate_contract_20260906.md` (untracked, W only) | H | V3 gate contract; retired by D1; archived under D8 |

Both suspended plans were removed on 2026-09-29 after D2, D3 and D8 were adopted. Commit `5b46da55` is the last one that contains them.

### 10.4 Known stale statements

None outstanding. On 2026-09-29 the following were corrected in
`NavierStokesFreeSurface.md` and `LevelSet.md`:

- the default unfitted capillary form, which is `SurfaceStress`;
- the removed level-set contact-angle penalty;
- optional prescribed-angle slip;
- the sharp wetted-wall operator;
- the 3D `LinearCorner` interface order, which is 2;
- the capillary-curvature contract.

Add new entries here when a document is found to disagree with the code.

---

## 11. Glossary

| Term | Meaning |
|---|---|
| `LinearCorner` | Production cut backend: a piecewise-planar interface from a P1 level set on simplices |
| `CutVolume` | Active-domain integration over the cut liquid sub-volume |
| Snapshot | `FreeSurfaceGeometrySnapshot`: the single published cut geometry for assembly, constraints and diagnostics |
| SurfaceStress | Laplace–Beltrami capillary form `γ ∫ (I − n⊗n) : ∇v` on the generated interface |
| KAG | Kinematic area gradient: curvature recovered as `M⁻¹` times the φ-gradient of the discrete surface-plus-wall energy |
| Aggregation (AgFEM) | Small-cut DOFs constrained to extrapolations from well-supported root cells |
| Pressure facet jump | Ghost-penalty-type `[∇p]·[∇q]` term on cut-adjacent facets |
| Sampled vs minimized | Sampled: interface interpolated from an analytic shape. Minimized: nodal φ optimized to a discrete energy stationary point. |
| `Ca_sp` | Parasitic capillary number `μ max|u| / γ` |
| La | Laplace number `ρ γ D / μ²` |
| FSR / AD / WP / Q | Audit finding / architecture decision / work package / qualification gate (2026-07-20 audit) |
| M0–M8 | Milestones in this tracker (§5) |
| W | The WP-4 scratch worktree (§9) |

---

## 12. How to update this file

- **Status changes:** edit §2.1, tick the milestone box in §5, and add one dated bullet to §8 with the key numbers, the commit hash, and where the raw output lives. Two to five lines is the norm.
- **Decisions:** record them in §4 with a date and one sentence of rationale.
- **Tolerances:** set them before first use, in the milestone text. If a tolerance proves unrealistic, change it with a one-line justification rather than creating a new document or matrix version.
- **Evidence:**
  - a benchmark script and tolerance file under `tests/cases/fluid/free_surface_benchmarks/`;
  - the commit hash;
  - the results row here;
  - raw output for accepted milestones in group storage (not scratch).
  - Frozen, hash-bound qualification records are reserved for publication-grade studies (M8).
- **Time-box:** if an investigation passes about 3 working days without a physics-visible improvement, stop. Write the question and the options in §4 and decide before continuing.
- **Do not** create new status, plan or checkpoint documents for this work. Link detail documents from §10 if they are truly needed.
- **Commits:** keep the existing author/committer identity and the commit-message vocabulary scan used on this branch. Submitted jobs keep the begin/end/fail mail settings.
