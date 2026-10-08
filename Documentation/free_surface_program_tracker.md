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
- **D9, 2026-09-30: level-set transport uses a new PDE velocity extension.**
  - Moving-interface benchmarks (sloshing, static drop, capillary wave, sessile drop) will advect φ with an extension velocity from a new PDE-based extension.
  - The extension lives on a separate auxiliary field and is parameter-free (P1). It never enters the momentum rows; the retired same-field dry-domain diffusion (FSR-01) stays retired.
  - Until it exists, the M2, M3 and M4 benchmark runs wait. The existing algebraic wet extension (`wall_compatible_normal`) is the comparison baseline.
  - Reason: advecting φ with the coupled fluid velocity leaves dry vertices at zero velocity, and the resulting lag failed `linear_sloshing_2d` at L/h = 64.
  - **Implemented and merged 2026-09-30 (`6977c5ac`..`02da73ea`, new `Application/Core/LevelSetPdeVelocityExtension.{h,cpp}`).**
    - The known set K is the wet vertices plus every vertex of the retained interface cells, the same seed as `wall_compatible_normal`, so w = u on the interface.
    - On every other vertex w solves, with Dirichlet data from K, one of two parameter-free symmetric problems:
      - `pde_harmonic`: (∇w, ∇v) = 0;
      - `pde_normal`: ((n·∇)w, (n·∇)v) = 0 with n = ∇φ/|∇φ| per cell.
    - The extension covers all dry vertices (no band). A band edge brought back a velocity jump; a band remains only as a diagnostic option.
    - Wall-normal components are zero on dry wall vertices, using the fluid's strong wall conditions.
    - Each rank contributes its owned dry cells; every rank assembles the same system in global-ID order and solves it by sparse LU, so the result is independent of the partition.
    - Two couplings, selected by `Advection_velocity_extension_coupling`:
      - `prescribed` writes w to the separate prescribed advection field, as D9 specified; the transport sees it one outer pass late.
      - `monolithic` installs the same problem as frozen rows of the existing algebraic extension unknown; it never touches the momentum rows.
    - Both the method and the coupling must be named; there is no default. The run fails closed on a singular solve, a zero ∇φ (normal operator), non-simplex dry cells, or amplification above 16×.
    - Tests: 8 serial unit tests, 2-rank parity to 1e-13, driver selection tests and 48 Python tests. Application CTest 4/4 (job `46108807`).
    - Recommended variant: `pde_harmonic` with `monolithic` coupling. It is always well-posed (each dry value is a convex combination of its neighbours) and keeps the coupled-field outer-pass counts. `prescribed` coupling raises static-drop start-up steps to 9–11 outer passes against the 12-pass cap. The replicated dry-region solve may need a distributed solve for large 3D decks.
- **D10, 2026-09-30: space and time convergence are judged separately.** Spatial convergence is gated with the time-step error removed (a small fixed Δt or a converged-Δt reference), and a separate Δt study is run at a fixed mesh. Refining Δt with h let opposite-sign errors cancel.
- **D11, 2026-09-30: volume criteria gate the maximum deviation over the run**, including any reversible oscillation of the P1 area.
- **D12, 2026-09-30: sloshing damping error ≤ 5% at the finest level** (pass/fail), matching the capillary-wave criterion.
- **D14, 2026-10-05: kinematic reconciliation of the transported level set.**
  - The benchmark generators (sessile drop, static drop, linear sloshing, capillary wave) write `Enable_kinematic_reconciliation=true`; `--kinematic-reconciliation off` reproduces the earlier decks.
  - The solver default stays off until it has been tested on more physics.
  - Its fixed internal constants (halfway limit 0.5, at most 8 fixed-point iterations, the sign and cut-class guards) are accepted for now under P1.
- **D15, 2026-10-05: the sessile-drop protocol uses the PDE velocity extension** (harmonic, monolithic), as the other free-surface benchmarks do.
- **D13 result (2026-10-05; merged `b7d55fad`..`aa811c43`; opt-in `Surface_tension_semi_implicit=NormalIncrement`, default off and bitwise identical; FE 34/34, Physics 7/7, Application 4/4).** Validation is in the design note §9.
  - **M2 static drop:** La = 12 passes every gate at Δt = 0.04, 0.02 and 0.01. At R/h = 32 the pressure-jump error is 3.27e-5 (order 2.18), the same as the 2·Δt_B reference, with 618 instead of 4,000 steps at Δt = 0.02, about 6× fewer outer passes.
  - **First La = 120 pass**, at Δt = 0.01 (3,900 steps instead of 24,900), or at 0.02 with kinematic reconciliation, which removes the R/h = 8 volume failure.
  - **Capillary wave:** runs at 25–100 steps per period. Without the term it fails at step 0 for λ/h ≥ 32 at 100 steps per period. Observed time order is about 2 in frequency and 1.8–1.9 in damping. The damping gate at λ/h = 32 still fails (6–7%, a spatial error).
  - **Sessile drop:** contact-line behaviour is unchanged at 2–4× the protocol step.
  - **Energy:** does not increase, except for two start-up steps accepted on a frozen epoch after a topology cycle (relative 3.5e-7 and 7.6e-7).
  - **Risks:** frozen-epoch acceptance leaves the term non-zero; GMRES iterations rise to about 150 at the largest steps.
- **D19, 2026-10-05: new time-step protocols built on D13.** The lagged-increment term is on by default in the M2 and M3 generators.
  - M2: one fixed physical Δt for all levels, Δt = 0.02 at La = 12 and 0.01 at La = 120.
  - M3: 50 steps per inviscid period at every level.
  - Each protocol is checked against a Δt/2 run.
  - Criterion in `tolerances.json`: every gate passes in both studies, and the gated quantity changes by at most 0.1% when the step is halved (M2: pressure jump at the finest level; M3: frequency ≤ 0.2%, damping ≤ 1% at the finest level).
  - The old protocols stay reproducible through generator options.
  - **Implemented 2026-10-05 (`c5798e96`, `0fbe786c`; benchmark Python tests 114/114):**
    - New generator defaults, with the old protocols reproducible bitwise:
      - M2: `--dt-multiple 2|1 --surface-tension-semi-implicit None`;
      - M3: `--dt-rule capillary-limit --surface-tension-semi-implicit None`.
    - `verify.py` gates the Δt criterion between the Δt and Δt/2 groups at the finest common level, and prints "not evaluated" when only one Δt is given. M2 requires every level at Δt/2; M3 requires only λ/h = 64.
    - `fitted_capillary_wave_2d` is pinned to the capillary-limit rule, so it is unchanged.
    - **New protocol runs:** binary `aa811c43`, PDE transport plus kinematic reconciliation, `surface_stress`, outputs in `/scratch/users/zsexton/free-surface-benchmarks/protocol-2026-10-05/`:
      - job `46704017`: M2 La = 12, R/h = 8/16/32 at Δt and Δt/2, 4 ranks each;
      - job `46704018`: M2 La = 120, same layout;
      - job `46704019`: M3 λ/h = 16/32/64 at 50 steps per period plus λ/h = 64 at 100, serial.
    - **M3 result (job `46704019`, all 4 runs in about 40 min):**
      - frequency passes: 1.58e-2 / 3.94e-3 / 1.39e-3, order 1.75;
      - volume passes: ≤ 2.1e-6 with reconciliation;
      - the Δt criterion passes at λ/h = 64: frequency changes 9.8e-4 and damping 2.7e-3 when the step is halved;
      - **damping fails only at λ/h = 32** (7.1% > 5%). It converges at order 2.9 to 0.39% at λ/h = 64, so it is a spatial error at λ/h = 32.
    - **M2 rerun:** the first M2 jobs (`46704017`, `46704018`) failed at step 1 with the 4-rank maintenance consensus bug, because the binary `aa811c43` predates the MPI follow-up. They were resubmitted on `3bc60e4e` as jobs `46787442` (La = 12) and `46787451` (La = 120).
    - **M2 result under D19 (2026-10-06, binary `3bc60e4e`, 4 ranks per run, verified with `verify.py`): both Laplace numbers PASS every gate at Δt and Δt/2.**

      | La | Δt | pressure-jump error at R/h = 32 (order) | max volume drift | Δt criterion (pressure-jump change) |
      |---:|---:|---:|---:|---:|
      | 12 | 0.02 / 0.01 | 3.29e-5 (2.18) / 3.29e-5 (2.19) | 2.1e-9 | 1.2e-10 ≤ 1e-3 |
      | 120 | 0.01 / 0.005 | 3.00e-5 (2.08) / 3.01e-5 (2.10) | 8.1e-10 | 5.4e-8 ≤ 1e-3 |

      Velocity-growth ratios stay ≤ 0.99 and the parasitic capillary number decreases with R/h at both La. Wall time on 4 ranks: R/h = 32 took 2.7 h (La = 120, Δt) and 5.0 h (Δt/2). The 2D part of M2 is complete; the 3D static drop remains.
- **D20, 2026-10-06: the M3 damping gate applies at the finest level, λ/h = 64** (≤ 5%; it also applied at λ/h = 32 until now). This matches M2, which gates at its finest level, and the sloshing damping criterion (D12). The observed-order requirement over 16/32/64 stays.
  - The change was made after the first D19 protocol run (job `46704019`) had been seen. The damping error converges at observed order 2.9, from 7.1% at λ/h = 32 to 0.39% at 64, so the λ/h = 32 excess is a resolution error, not a method error. This is the one adjustment allowed for this tolerance (§5).
  - `capillary_wave_2d/tolerances.json` (`at_level` 64, the earlier value kept as `at_level_before_D20`), README table and two new synthetic tests.
  - **M3 passes under D19 + D20** (same runs, re-verified): frequency 3.94e-3 at λ/h = 32 (order 1.75), damping 3.95e-3 at λ/h = 64 (order 2.90), volume ≤ 2.1e-6, Δt criterion 9.8e-4 / 2.7e-3. The Δt/2 run at λ/h = 64 also passes (damping 6.6e-3, volume 1.3e-6).
- **D21, 2026-10-06: the sessile-drop protocol enables the sign-definite patch bounds** (`Enable_sign_definite_patch_bounds=true`, written by `sessile_drop_2d/generate_case.py`; `--sign-definite-patch-bounds off` reproduces the earlier decks). The static-drop, capillary-wave and sloshing generators keep them off; the solver default stays off.
  - Why: the P1 Galerkin transport step has no local maximum principle. Next to a contact line a wall vertex whose whole patch is in one phase drifts across the isovalue and creates a spurious wall spot (the "4 wall crossings" that made `verify.py` reject the full R/h = 16 runs, with or without reconciliation). The bound clamps such a node to the range of its previous patch values; nodes of cut cells never change, so the interface, contact line and liquid area are untouched. No parameter; partition-independent reductions.
  - Solver code merged with the MPI follow-up (`cc7cdaee`, `3bc60e4e`); generator default and README (`11c69cd9`, `1508d98f`).
  - **Validation (R/h = 16, `SurfaceStress`, PDE transport, reconciliation on, job `46685007`):** two wall crossings at all 101 outputs at 60° and 120°; final angle errors 1.745° and 0.660°; base and apex errors below 0.6%; area drift 8.7e-6 and 1.5e-5. The bounds act on at most 8 nodes per step (corrections ≤ h/330) and add no measurable time. With the option off the output is bitwise identical.
- **D22, 2026-10-06: faster turnaround for free-surface runs.** Measured cost split (sessile R/h = 16; capillary wave λ/h = 16 and 64):
  - 4.05 outer geometry passes per sessile step (4.8 for the capillary wave at λ/h = 64), with 57–61% of wall time outside the Newton solve (geometry regenerated on every pass, constraints, maintenance). Assembly is 23–28%. The linear solve is 7–10% at the coarse levels but 33% at λ/h = 64, where GMRES needs 378 iterations per Newton iteration (82 at λ/h = 16).
  - Sessile steps grow as h^-1.5 under the capillary limit (2,800 / 7,900 / 22,300 at R/h = 16 / 32 / 64).
  - More compute raises throughput but barely shortens one 2D run (4 ranks give 3.1× at R/h = 32; ghost layers and the replicated dry-region solve limit further gains), so the work goes into the solver and the protocol.
  - Approved work:
    1. A sessile protocol with the lagged term (D13) and one fixed physical Δt for all levels, as D19 did for M2/M3. The Δt criterion is fixed in `tolerances.json` before the decisive runs, and the protocol is adopted only if it passes at Δt and Δt/2 (R/h = 16, 32). Current-protocol runs at R/h = 16/32/64 serve as references.
    2. Fewer and cheaper outer geometry passes per step: bitwise identical where possible, otherwise opt-in and rank-independent.
    3. A rank-independent sparse direct solve option, then inexact Newton forcing terms, both opt-in.
  - Already queued: constraint build, post-merge hot spots, dry-cell records, LTO + PGO (D16), multithreaded assembly. Development checks use R/h = 16 runs and restarts from saved states; R/h = 64 is reserved for acceptance.
- **D23, 2026-10-06 (taken under the user's auto-approval window, 17:34–21:34): linear-solver and predictor options stay opt-in; the benchmark protocols are unchanged.**
  - The gathered sparse direct solve (`<LS type="Direct">` with FSILS) is opt-in. It is recommended for 2D development runs on 1–4 ranks when GMRES needs hundreds of iterations per Newton iteration (capillary wave λ/h = 64: −27% serial, −17% on 4 ranks; static drop R/h = 32 early transient: −40%). It is neutral for full M2 runs on 4 ranks and at 8 ranks, and not viable in 3D (sphere L8: 11× slower per solve, 7.2 GB on rank 0). Across rank counts it agrees to 1e-12, the same size as GMRES's own spread (7e-13). M2 La = 12 and M3 pass with it. The protocols keep GMRES so that gated results stay comparable with earlier runs.
  - Eisenstat–Walker forcing (`<Inexact_Newton_forcing>`) stays off. It raised Newton iterations by 70–90% and slowed every case, because most passes need only one Newton iteration.
  - The generalized-α rate-extrapolation predictor (`Generalized_alpha_predictor=RateExtrapolation`, fields `phi`) stays opt-in. It is 20–24% faster per step and gives the same results on 1, 2 and 4 ranks to 1e-14. It is as far from a 100× tighter-gate reference as the default is, and it moves the M2/M3/M4 gate metrics by at most 1.5e-6 relative. Making it the default would change every result within the nonlinear tolerance, so that waits for an explicit user decision.
  - Measured before the merge: commits that are bitwise identical on the default path cut a further 10–14% per step on top of 119818c3 (check-only passes assemble only the residual, redundant cut-context rebuilds removed).
- **D24, 2026-10-06: the fixed-step sessile protocol is not adopted** (outcome of the pre-registered check approved under D22; generator default unchanged). `sessile_drop_2d/generate_case.py` gains non-default options (`--dt-rule fixed`, `--dt-multiple`, `--dt-divisor`, `--surface-tension-semi-implicit`); default decks are bitwise identical. `verify.py` gates the pre-registered `time_step_criterion`: every M4 gate at dt and dt/2; final angle within 0.1°; base and apex within 0.1%; history angle within 0.5° and history base within 0.1%; two wall contact points at every output. Results (binary `4e604d10`, 4 ranks; README "Fixed-step validation"):
  - **R/h = 16: the contact angle is not time-converged, at any step tested, including the current protocol step.** The final angle moves 0.1–1.5° per halving from 2dt to dt/4 with no trend toward zero (60°: error 1.93 / 1.75 / 1.45 / 1.16°; 120°: 0.64 / 0.65 / 0.51 / 1.84°). Base and apex converge within about 0.15%. At T the contact lines still creep (capillary number 4e-3 to 9e-3), and when a partly pinned contact line crosses each wall vertex depends on dt, which sets the local circle-fit angle.
    - With the lagged term at the protocol step the result changes by at most 0.01° (final angle) and 0.12° (history), and mean outer passes fall from 4.05 to 3.4–3.7.
  - **R/h = 32 and 64: runs fail at every step tried, including the current protocol**, so no dt pair exists there. The failure modes are the same as in the current-protocol study:
    - outer-loop stagnation at the 12-pass cap: R/h = 32 120° at t = 7.9 in the reference run, and at t = 5.9–11.6 at other steps; R/h = 64 60° at t = 0.39;
    - the 4-rank aggregation owner disagreement (R/h = 32 60° at t = 0.267);
    - the rollback failure (R/h = 32 60° in the reference run at the same t = 0.267, and R/h = 64 120° on 8 ranks).
  - **Cost:** at fine levels GMRES iterations grow with the step: R/h = 64 120° needs 1,650 iterations per solve at the fixed step against 360 at the current protocol. A working fixed step would save at most about 3× there.
  - **Consequences for M4:**
    - The solver failures (debugging agent, branch `dev/fix-sessile-mpi-robustness`) block the resolution study under any protocol.
    - The angle criterion needs a protocol decision once the runs complete. Options, not decided: a longer run (for example 10 viscous times) so the end state is closer to rest; or an angle tolerance that allows for the dt and vertex-crossing scatter, since base and apex converge.
- **Reference study status (2026-10-06, binary `4e604d10`, current protocol):** R/h = 16 passes at 60° and 120°. Every R/h = 32 and 64 run failed:
  - R/h = 32 60°: rollback failure at step 172 (4 ranks);
  - R/h = 32 120°: outer cap at step 5096 (t = 7.9) after 7.75 h (4 ranks);
  - R/h = 64 60° and 120°: aggregation owner disagreement (8 ranks; 60° at setup, 120° after step 459).
- **D25, 2026-10-07: protocol for the 3D static sphere (M2, 3D part).** The user approved starting it and a larger memory budget, and left the four open 3D questions to the coordinator's recommendation:
  1. **Gating level R/h = 16**, with the observed order over R/h = 8/16. R/h = 32 is not affordable. Gated at R/h = 16: pressure-jump error ≤ 1% with order ≥ 1; parasitic capillary number decreasing from 8 to 16; no velocity growth; volume drift ≤ 1e-4.
  2. **Memory:** the dry-cell compaction is merged (R/h = 16 peaks at about 19 GB serially, against an estimated 49 GB before). Jobs may use up to a full node (at most 8000 MB per CPU).
  3. **Box:** keep the 3R cube that the benchmark and its tests were built for.
  4. **Wet-extension map output:** already opt-in (Perf 8); transport is the PDE extension.
  - **Time step:** the D19 static-drop protocol, with the lagged normal-increment term on and Δt = 0.02 at La = 12 for both levels. The Δt/2 check runs at R/h = 8 only. In 2D halving the step changed the pressure jump by 1e-10 (La = 12), and a Δt/2 run at R/h = 16 would cost days.
  - **Form:** `SurfaceStress` only, since both KAG forms fail M2 in 2D.
  - **Binary:** `svmultiphysics-39c88f74-ltopgo`, with multi-rank runs (FSILS).
- **D26, 2026-10-07 (user's auto-approval window): the sessile and MPI robustness fixes are accepted** (branch `dev/fix-sessile-mpi-robustness`).
  - **A, aggregation owner disagreement on 4 and 8 ranks.** The finalized-row check exempts a rank that sees a slave only deep in the halo. It recognized such a rank by a closed master being absent, but closure removes an absent Dirichlet wall master, so ranks that correctly carry no line were rejected.
    - The exemption now also covers a non-owning rank that lacks a master of the unclosed line. That is the same rule the installation and `ParallelConstraints::validateConsistency` use. Ranks that hold a line, own the slave, or see all masters are still compared, and every rank that assembles with the slave must still see every master.
  - **B, "rollback failure".** It was A raised in the entry synchronization of the outer fixed point. The rollback re-synchronized the same state and failed again. The failure is now reported as a step failure with both messages, and the TimeLoop restores the accepted state and retries with a smaller step or stops with the cause.
  - **C, FSILS GMRES deadlock.** Shared Dirichlet-face reductions returned early on ranks with an empty face part. Those ranks now take part with zeros.
  - Each fix has a regression test that fails before the fix.
  - **Evidence:** sessile R/h = 64 on 8 ranks ran 300 steps (60°, failed at setup before) and 650 steps (120°, failed at step 459). R/h = 32 60° on 4 ranks ran 400 steps (failed at 172). The capillary wave λ/h = 64 on 8 ranks completed all 200 steps (failed at 86). Reference set bitwise identical, 2- and 4-rank parity bitwise identical, CTest 88/88.
- **D27, 2026-10-07: failure D, outer-loop stagnation, is open.**
  - Seen at R/h = 32 (120° current protocol at t = 7.9; 60° at larger steps) and R/h = 64 (60° at t = 0.39). Steps hit the 12-pass cap while still contracting slowly: about 0.41 per pass, or about 0.8 with an oscillating sliver cut cell.
  - The existing opt-in Aitken relaxation took 7 passes on one such step, where a cap of 20 needed 13.
  - The outer-pass agent is evaluating a higher cap, relaxation, the predictor and a sliver fix. The sessile resolution study waits for it.
- **D28, 2026-10-07 (user's auto-approval window): outer-loop stagnation fixed with a higher pass limit and deferred Aitken relaxation; the default changes, bitwise on every step accepted within 12 passes.**
  - **Diagnosis:** the slow mode alternates sign (contraction λ ≈ −0.43 at R/h = 32 60°; −0.74 to −1.08 at R/h = 32 120°, where plain iteration slowly diverges). The driver is a sliver cut cell at the contact line whose active fraction alternates between passes (about 2e-5 to 4e-4).
  - **Change:** `Outer_fixed_point_max_passes` defaults to 30 (was 12) and `Outer_fixed_point_relaxation_start_pass` to 12 (0 = off, 1 = every update). The cut-topology restart limit follows the pass limit.
  - **Effect on earlier results:** steps that converge within 12 passes never reach a relaxed update, so every run that completed before is bitwise identical. Only steps that previously failed, or that adaptive runs previously retried with a smaller step, now converge at their own step.
  - **Evidence (4 ranks):**
    - every reproducible stall completes: R/h = 32 60° at 2dt, 200 steps, at most 13 passes; R/h = 32 120° restart, 150 steps, at most 19; R/h = 64 60° dt/2 restart, 60 steps;
    - 1, 2 and 4 ranks agree to 2.4e-13;
    - the result differs from plain 30-pass iteration by 4e-7 (velocity), against a tolerance-level spread of 1–5e-5;
    - reference set bitwise identical, CTest 88/88, `test_fe_timestepping` 284/284.
  - **Rejected:**
    - a higher cap alone: R/h = 32 120° still fails at 30 passes;
    - the predictor: R/h = 32 120° still fails;
    - a sliver prune threshold, which would change the discretization;
    - relaxing every update stays opt-in: it is faster (up to 37%) but moves the R/h = 32 120° result 10–50× more.
  - **Caveat:** in the late R/h = 32 120° regime, tolerance-level perturbations grow (phi differences of about 1.5e-3, and local velocity up to 10%). The M4 angle results there must be read with that sensitivity in mind.
- **D29, 2026-10-07 (user's auto-approval window; the M4 protocol question became blocking): the sessile resolution study runs to 10 viscous times.**
  - Settings: 200 outputs, the same cadence as before; the capillary-limit Δt is unchanged; binary `688a625a` (D28) with LTO + PGO.
  - An R/h = 16 Δt/2 pair at the longer run measures whether the contact angle settles once the contact lines have stopped creeping (D24: at 5 viscous times it moved 0.1–1.5° per halving).
  - If it still moves by more than about 0.5°, an angle tolerance that allows for the Δt and vertex-crossing scatter goes to the user.
  - The generator default (5 viscous times) changes only after the results.
  - Jobs `46899877` (R/h = 16 and 32, plus the R/h = 16 Δt/2 pair) and `46899878` (R/h = 64, 8 ranks each).
- **D30, 2026-10-07 (user): a second round of FE-infrastructure speed-ups (physics-agnostic), in three agents.**
  1. MPI scaling: per-rank phase profiles at 1–16 ranks; ghost depth computed instead of fixed at 8; batched collectives; no replicated serial work; cut-weighted partitioning only if it changes round-off alone. This includes the CTest case that is about 13× slower on 2 ranks than serially and sets the 25–45 min test cycle.
  2. Incremental cut, snapshot and constraint regeneration, and an assembler and sparsity pattern kept across passes and steps; bitwise. Then threads for per-cell geometry rebuilds, bitwise for any thread count.
  3. Quieter default logging: the held proposal `e2efc6cc` plus other unparsed per-pass lines, moved to DEBUG; the solution is unchanged. Then an opt-in scalable preconditioner whose results do not depend on the partition: a field split from the generic block layout with partition-independent inner solves.
  - Measured starting point (s/step at 1/2/4/8 ranks): static drop R/h = 32 14.9 / 8.7 / 5.6 / 3.3 (57% at 8); sessile R/h = 32 2.38 / 1.49 / – / 0.86 (35%); capillary wave λ/h = 64 61% at 4 ranks.
- **D32, 2026-10-07 (user): the 3D static-sphere R/h = 16 run (D25) launches only if its measured step time fits within the 7-day job limit.**
  - The PDE velocity extension's replicated dry-region factorization was 41% of a 3D R/h = 16 step. It is now one component per rank with a 3-entry cache, bitwise identical (`f6057e20`).
  - The step time is then measured with a short probe: about 10 steps on 16 ranks with `<Ghost_layers>12</Ghost_layers>` (D31).
  - The full run starts only if setup + 618 × the mean step time, plus 10% margin, fits within 7 days.
- **D29 result (2026-10-07, binary `688a625a`-ltopgo, 4 ranks): running to 10 viscous times does not settle the sessile drop; the M4 sessile gates fail and the trend is wrong.**

  | case | angle error (L/R mean) | base err | apex err | max dA/A | velocity growth | Ca at T |
  |---|---:|---:|---:|---:|---:|---:|
  | 60°, R/h = 16, Δt | 2.25° | 7.8e-3 | 1.1e-3 | 1.0e-5 | 2.11 | 1.3e-2 |
  | 60°, R/h = 16, Δt/2 | 2.92° | 3.7e-3 | 1.7e-3 | 1.3e-5 | 1.15 | 6.3e-3 |
  | 120°, R/h = 16, Δt | 2.55° | 9.3e-3 | 2.7e-3 | 1.4e-4 | 0.41 | 9.4e-3 |
  | 120°, R/h = 16, Δt/2 | 5.78° | 1.2e-2 | 2.7e-3 | 1.3e-4 | 0.89 | 9.0e-3 |
  | 120°, R/h = 32, Δt | 4.08° | 5.3e-2 | 2.0e-2 | 8.7e-4 | 0.47 | 1.2e-1 |

  - **Comparison with 5 viscous times:** at 5 viscous times R/h = 16 had angle errors of 1.75° (60°) and 0.66° (120°), with velocity growth 0.74 and 0.51. Running longer makes the angle error larger and, at 60°, makes the velocity grow.
  - **Δt sensitivity:** halving the step still moves the final angle by 0.66° (60°) and 3.2° (120°).
  - **Refinement:** at 120° the R/h = 32 errors are larger than the R/h = 16 errors, and the area drift (8.7e-4) exceeds the D11 limit.
  - **Failed runs:**
    - R/h = 32 60° stopped at step 12,373 (t = 19.2) with a small-cut aggregation `no_valid_root_proposal` (4 ranks).
    - R/h = 32 120° completed all 15,800 steps but reported "TimeLoop: max_steps exceeded". The accumulated time ends 6.6e-12 short of t_end, which is beyond `TimeLoop`'s fixed 1000·ε tolerance, so the post-loop check fails. This is a physics-agnostic end-time bug.
  - **Conclusion:** the drop does not reach the Young equilibrium, and the contact lines keep moving at Ca of about 1e-2. Together with the Ren–E result (the contact-point balance shares the uncompensated Young force with the viscous/slip stress in the contact element), this points to the D4 contact-line formulation, not to the time step or run length.
  - The R/h = 64 run (job `46899878`) is still running.
- **D33, 2026-10-07 (user): M4 next steps after the sessile and Ren–E results.**
  - **Sessile R/h = 64 cancelled.** Job `46899878` stopped at step 7,204 (60°, t = 3.96) and step 2,399 (120°, t = 1.32), because the R/h = 32 result already shows the wrong trend.
  - **After the weekly reset (2026-10-08), one agent analyses the discrete contact-point force balance:** why the D4 sessile drop does not reach the Young equilibrium, and why Ren–E speeds are below the law by an amount set by μ·M. It proposes a consistent fix.
  - **Three small fixes approved for after the reset:**
    1. The `TimeLoop` end-time tolerance should allow for round-off accumulated over the steps. Every run that ends normally today stays bitwise identical; runs that complete all steps within accumulated round-off of t_end no longer report "max_steps exceeded".
    2. Small-cut aggregation failures near walls (`no_valid_root_proposal` for nearly full cells, and isolated wet wall patches; sessile R/h = 32 60° and capillary rise level 10).
    3. A failed step's rollback must leave the system set up, so that the real cause is reported instead of "setup() has not been called".
  - **Choices the wetting agent took during the approval window (D-W1 to D-W8), confirmed on the coordinator's recommendation:**
    - **Ren–E criteria:** confirmed as pre-registered. The finest-level RMS law error limit is 10%; the run fails it (R/h = 32: 17% advancing, 35% receding).
    - **Capillary rise, accepted:**
      - no level-set inflow condition on the bottom;
      - the capillary step bound dt ≤ 0.7·√(ρh³/2πγ) as candidate-protocol revision 2;
      - the gathered direct solve (D23);
      - comparison over a partial time window until level 10 completes.
    - **Rate initialization off** (`SVMP_GENERALIZED_ALPHA_PDE_UDOT_INIT=0`): accepted, but the setting should become a deck key rather than an environment variable.
    - **Wider aggregation guards in the capillary-rise deck (6 and 12 instead of 4): provisional only.** Results that rely on them are flagged; after fix 2 the deck returns to the default guards and is rerun.
    - **Diagnostic jobs pinned to the protocol node:** accepted.
- **D16, 2026-10-05: production binaries will use LTO + PGO** (`SV_ENABLE_LTO=ON`, `SV_PGO=USE`). The outputs are bitwise identical.
  - Adoption waits until the current speed-up branches settle (post-merge hotspots, dry-cell records, constraint build), because they move the hot paths.
  - The profile is then trained on the tip and stored in group storage (not scratch, which is purged), and refreshed after hot-path changes.
  - **Adopted 2026-10-07 at `39c88f74`:** `svmp-bin/svmultiphysics-39c88f74-ltopgo` (job `46853003`, `jobs/build_ltopgo.sbatch`), profile in `/home/groups/amarsden/zsexton/svmp-pgo/39c88f74/`. Bitwise identical on the nine reference cases and on 2 and 4 ranks; CTest 87/87. A step takes 6–18% less time (job `46866398`): sessile R/h = 16 15% serial and 11% on 4 ranks, capillary wave λ/h = 32 and 64 18% and 16%, static drop R/h = 32 12%, 3D tank 6%, sphere proxy 9%. Training set and policy: `FE/Docs/BuildOptimization.md`.
- **D17, 2026-10-05: benchmark runs share one JIT object cache across Skylake and Milan nodes.** They set `SVMP_JIT_CPU=x86-64-v3` and a pinned `SVMP_CACHE_PROFILE` (`run_case.sbatch`, `run_case_mpi.sbatch`).
  - The study found this bitwise identical to the host target, and it saves 10–15 s of kernel compilation per cold run.
  - Speed-up bitwise checks keep the default target, so they stay comparable with the shared baseline.
- **D18, 2026-10-05: no round-off-changing JIT options.** JIT floating-point contraction and optimization level 1 stay off, matching the preference for exact, reproducible results.
- **D13, 2026-10-01: implement the lagged normal-increment surface-tension term** (`Documentation/free_surface_semi_implicit_surface_tension_design.md` §3.3).
  - Opt-in `Surface_tension_semi_implicit=NormalIncrement`, `SurfaceStress` first.
  - Δt_eff comes from the integrator, not a tuned parameter (P1). The term vanishes in every fresh residual, so the accepted solution is unchanged within the outer tolerance.
  - Purpose: remove Δt_B as the step-count driver for the relaxation and capillary benchmarks.
  - Any protocol change to a fixed physical Δt with Δt refinement waits for validation and a user decision.
  - Branch `dev/semi-implicit-surface-tension`.
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

- [x] **Baseline build and C++ suites at `fef0d02f`: all passed (2026-09-30).** Submitted 2026-09-29 as Slurm job `45961287`: 8 CPUs, 32 GB, 12 h.
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
  - **Physics and Application suites:** tests-only job `45975100` (`jobs/tests_only.sbatch`, submitted with `sbatch --export=NONE`). CTest runs with the Slurm/PMI variables removed. **Result: Physics 7/7 (2,129 s) and Application 4/4 (4,520 s) CTest entries passed.** Together with FE 32/32, the whole C++ baseline is green on `fef0d02f`.
    - A `tests` symlink beside `build/` points at the source fixtures. Some Application tests search upward from their working directory for `tests/cases/fluid/open_vessel_free_surface`.
  - **MPI launch rule (found 2026-09-29).** Jobs submitted from inside the interactive `sh_dev` session inherit that session's `srun` PMI contact variables.
    - An MPI binary started without `mpiexec` in such a job contacts the interactive `srun`, which prints `PMK_KVS_Barrier task count inconsistent` in the user's terminal, and then hangs.
    - Verified in test jobs `45973957` and `45974084`: bare launches fail; `mpiexec -n 1`, `srun --mpi=pmix --exact`, and bare launches with the Slurm/PMI variables removed all work.
    - Rule: submit with `--export=NONE` and launch MPI programs through `mpiexec` (benchmarks README).

- [x] **Tank at rest (2D and 3D): PASS (2026-09-30, job `46023412`).** Benchmark `tests/cases/fluid/free_surface_benchmarks/tank_at_rest/` (merged 2026-09-30).
  - Free-slip walls (strong zero normal velocity through `Effective_direction`). The fill height is off vertex rows.
  - max|u|/√(gH) ≤ 3.4e-13; pressure, interface and volume errors 0 to 5e-17. The gate is 1e-8 for this exactly representable state.
  - Newton makes no updates, so this checks balance, not the solve. A perturbed-start variant would exercise the solve.
- [ ] **Tank at rest (original plan item).** Take a small subset of the existing hydrostatic matrix into CTest. The full 960-case matrix does not need to run routinely.
- [ ] **2D linear sloshing: with the PDE extension, only the convergence-order part of the frequency criterion still fails (2026-09-30).** First run: job `46023412`. Benchmark `tests/cases/fluid/free_surface_benchmarks/linear_sloshing_2d/` (merged 2026-09-30).
  - Setup: free-slip walls; a reference from the exact viscous linear dispersion relation (ω = 1.7006, damping 9.52e-3); Δt = T0/(2L/h).
  - Frequency error −0.32% / +0.28% / +1.43% at L/h = 16/32/64 (limit 1% with convergence). Max area oscillation 5.5e-5 / 5.4e-5 / 6.5e-4 (limit 1e-4). Damping error 1.7% / 2.1% / 7.6%.
  - Diagnosed cause: with φ advected by the coupled fluid velocity and no velocity extension, dry vertices one row above the cut cells keep zero velocity and their φ lags. That lag grows with refinement (8% → 17% at half period) and drives a spurious cos(2kx) mode of 9.5% of the amplitude at L/h = 64.
  - With the existing wet-extension transport (the SPHERIC 05 option): frequency error −0.385% / −0.046% / +0.097%, area ≤ 4e-5. The frequency error is not strictly monotone, and the extension writes about 1 MB of map per step.
  - The Δt error alone is clean second order. Refining Δt with h lets opposite-sign time and space errors cancel, so gate the spatial study with the time error removed.
  - `static_drop_2d` used the same coupled transport, so M2 was exposed too.
  - **PDE extension (D9), job `46089180`.** The spatial study runs 128 steps per period at every level, with the time error removed via the Δt study at L/h = 32 (observed order 2.11) (D10).

    | Transport | Frequency error, L/h = 16 / 32 / 64 | Damping error at 64 (probe) | Max dA/A |
    |---|---|---:|---:|
    | `pde_harmonic` monolithic (protocol) | +0.019% / +0.062% / +0.074% | 3.3% | 3.2e-5 |
    | `pde_normal` monolithic | +0.016% / +0.054% / +0.059% | 3.1% | 2.6e-5 |
    | wet extension | +0.017% / +0.053% / +0.120% | 3.4% | 4.0e-5 |
    | coupled field | +0.081% / +0.383% / +1.455% | 7.6% | 6.5e-4 |

    - Protocol transport: the frequency limit (≤ 1%), damping (≤ 5%, D12) and volume (≤ 1e-4, D11) pass. The frequency error does not decrease (observed order −0.96), so the "observed convergence" part fails.
    - The errors are at the level of the measurement uncertainty: probe and modal fits differ by up to 0.018%, and the viscous frequency shift is 0.020%.
    - The modal-amplitude damping converges to the reference: γ/γ_ref = 1.028, 1.005, 0.999. The probe damping error is 3.3%.
    - The spurious cos(2kx) mode drops from 9.5% to 2.0% of the amplitude. Cost is 3.83 s/step at L/h = 64 with 3.08 outer passes per step.
    - Two-rank runs match serial to 1e-16 but need `Ghost_layers` = 3 (MPI defect 2).
    - **Open questions (for the user):**
      1. Frequency convergence: add an error floor below which the order test is skipped, gate on the modal fit, or accept the failure.
      2. Whether the damping gate should use the modal amplitude instead of the probe.
- [ ] **2D linear sloshing (original plan item)** at 3 meshes and 3 time steps. Compare frequency and damping with linear theory. Proposal: frequency error ≤ 1% at the finest mesh with observed convergence; volume drift ≤ 1e-4 over the run.
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
  - Level-set transport check (job `46089180`, La = 12, `surface_stress`, 2·Δt_B): the PDE extension does not change the relaxed drop. At R/h = 8 and 16 the pressure-jump error, `Ca_sp` and growth ratio agree with coupled transport to 2–3 digits, and it costs 15–18% more per step. The protocol transport is now `pde_harmonic_monolithic`, with Δt = 2·Δt_B at La = 12 and Δt_B at La = 120.
- [x] **M2 La = 12 study, done 2026-10-01: `surface_stress` PASSES every gate; both KAG forms fail.** Binary `35a81fd3` (`/scratch/users/zsexton/svmp-bin/svmultiphysics-35a81fd3`).
  - Three capillary forms (`surface_stress`, `kag_lumped`, `kag_consistent`) × R/h = 8, 16, 32, each as a serial single-rank job (MPI defects).
  - Jobs `46130396`, `46130397`, `46130399`, `46130400`, `46130403`, `46130406`, `46130407`, `46130408`, `46130409`.
  - Cases and output: `/scratch/users/zsexton/free-surface-benchmarks/static_drop_2d/35a81fd3/La12/<form>/L<level>`, with the job list in `jobs.txt`.
  - Expected: about 5 min at R/h = 8, about 1 h at 16, and about 10–15 h at 32 (longer on SKX nodes).
  - The first submission on `02da73ea` was held and cancelled when the face-sampling fix arrived. Job `46121663` then showed the fix leaves the unfitted static drop (R/h = 8, full run), sloshing (L/h = 16, full run) and capillary wave (100 steps) bitwise identical at the same cost.
  - **Results (complete; `surface_stress` at R/h = 32 took 25.2 h serial).**

    | Form | R/h | dp/(γ/R) − 1 | Ca_sp (final quarter) | growth | max dA/A | wall time |
    |---|---:|---:|---:|---:|---:|---:|
    | `surface_stress` | 8 | 6.76e-4 | 2.49e-4 | 0.741 | 1.1e-5 | 9 min |
    | `surface_stress` | 16 | 1.49e-4 | 1.32e-4 | 0.705 | 3.9e-7 | 1.5 h |
    | `kag_lumped` | 8 | −1.81e-4 | 1.48e-3 | **1.334** | 8.7e-5 | 18 min |
    | `kag_lumped` | 16 | 7.68e-5 | 3.90e-4 | 0.874 | 5.9e-6 | 2.3 h |
    | `kag_consistent` | 8 | 8.65e-4 | 5.52e-3 | 0.415 | 1.1e-4 | 21 min |
    | `kag_consistent` | 16 | **stopped at step 111** (t = 0.97) | | | | |
    | `surface_stress` | 32 | 3.27e-5 | 7.42e-5 | 0.792 | 5.5e-8 | 25.2 h |
    | `kag_lumped` | 32 | −1.42e-4 | 2.41e-4 | **1.264** | 7.7e-7 | 15.1 h |
    | `kag_consistent` | 32 | **stopped at step 1112** (t = 3.40), same outer-loop failure | | | | |

    - **`surface_stress` passes** (`verify.py`, `surface_stress.json` in the case directory):
      - pressure-jump error 6.8e-4 / 1.5e-4 / 3.3e-5, observed order 2.18 (limit 1% at R/h = 32, order ≥ 1);
      - `Ca_sp` strictly decreasing: 2.5e-4, 1.3e-4, 7.4e-5;
      - no velocity growth (0.74, 0.71, 0.79);
      - volume ≤ 1.1e-5.

      It is the only parameter-free form that passes, so the D2 route selection points to `SurfaceStress`. The formal selection in §4 waits for M3 and M4, as planned.
    - **`kag_lumped` fails.** The pressure-jump error is ≤ 1.4e-4, but it does not converge (observed order 0.17, with a sign change), and max|u| grows at R/h = 8 and 32 (growth 1.33 and 1.26). `Ca_sp` does decrease (1.5e-3, 3.9e-4, 2.4e-4) but stays 2–6× above `surface_stress`, and volume passes.
    - `kag_consistent` at R/h = 16: the outer geometry loop reached its 12-pass cap with the state change stalled at 6.8e-10 against a 1e-10 gate. The projected curvature had spikes up to 2.0e4 against a mean of 1.0 (RMS deviation 1.9). Without an adaptive step controller the run then aborted. This is evidence for D2: consistent-mass KAG curvature is oscillatory and its outer loop is not robust. Its R/h = 8 spurious currents are also 22× those of `surface_stress`.
- [x] **Per-step cost must come down before the refinement study (added 2026-09-29; done 2026-09-30).**
  - Merged commits `b5837011`, `09b46072`, `889c75f2` and `67b4395a`:
    - a reference-element metadata cache;
    - snapshot-currency checks once per content change;
    - reuse of the exact trial residual after an accepted line-search step;
    - a scaled outer gate, accepting when the fresh residual is at most max(absolute tolerance, relative tolerance × the step's first fresh residual), with no new parameter.
  - Results (SKX nodes, `surface_stress`, La = 12): R/h = 8 went from 3.41 to 1.08 s/step (7.7 → 4.0 outer passes), and R/h = 16 from 13.62 to 3.91 s/step. `kag_consistent`, which previously failed at step 0 by stalling at 1.1e-10, now completes.
  - The first three changes leave all output fields bitwise identical. The gate changes fields by at most 0.2% of the spurious max|u|, and benchmark metrics agree to 4–5 digits.
  - Integrated build and regression of `67b4395a`: job `46039315`.
  - Follow-up resolved 2026-09-30: JIT object-cache hardening.
    - Cause: temporary names had no host or process ID and files were opened with truncate, so colliding writers could produce a corrupt object that still loaded.
    - Now: host/PID/random temporary names with exclusive create, then rename; a checksummed cache-file format whose mismatches are rejected, deleted and recompiled; objects in an `objects-v2/` subdirectory; an `SVMP_JIT_CACHE_DIR` override.
    - `FE_LOG_LEVEL` and the other `FE_LOG_*` settings now take effect: they are read on first logger use, so the linker can no longer drop them.
    - FE CTest 34/34, including the new logger targets. The failing field-op test passed 20/20 repeats.
  - Integrated build and regression of `45bc5b09` (with the JIT and logger fixes), job `46076509`: FE 32/32, Physics 7/7 and Application 4/4 CTest entries passed. The stable solver binary for benchmark runs is `/scratch/users/zsexton/svmp-bin/svmultiphysics-45bc5b09`.
  - The smoke run took 5.5 s per step on a 625-vertex 2D mesh. Each step needed 9–10 outer geometry passes against a cap of 12, driven by the 1e-10 absolute level-set gate, and wrote about 0.35 MB of log.
  - Extrapolated, R/h = 32 at La = 12 needs about 6 days, and R/h = 64 needs weeks to months. The same cost limits M1, M3 and M4.
  - Profile one R/h = 8 step, then reduce the unnecessary outer passes and per-step output. Any convergence gate must be scaled or derived rather than tuned (P1).
  - Target: the La = 12 study at R/h = 8/16/32 completes within about a day.
- [x] **MPI check (2026-09-30; jobs `46084708`, `46086154`, `46089257`; results in `/scratch/users/zsexton/free-surface-benchmarks/mpi-check/`). Parallel runs are not usable yet: only 4 of 16 multi-rank runs completed.**
  - 2 ranks matched serial to solver tolerance for the static drop at R/h = 32 (speed-up 1.7×) and for sloshing (1.2×).
  - Defects found:
    1. Deadlock on 4+ ranks with a flat interface. `reimposeAcceptedMasterBearingState` (ApplicationDriver) skips the collective `history.updateGhosts()` based on the rank-local `hasMasterBearingLines()`.
    2. The default `<Ghost_layers>` of 0 fails small-cut aggregation (`incomplete_distributed_aggregation_halo`); 8 ranks fail even with 8 layers.
    3. The static-drop assembled system depends on the partition (the initial residual changes with rank count), while sloshing agrees. Capillary or aggregation assembly is suspected.
    4. Parallel Newton converges linearly: the Jacobian and residual are inconsistent in parallel.
    5. Rate initialization regularizes 0 empty rows in serial but thousands in parallel.
  - Consequence: M2 and M3 run serially as concurrent single-rank jobs, preferably on `-C CPU_GEN:MLN` nodes (about 2× faster than SKX).
  - Estimates: M2 La = 12 at R/h = 8/16/32 for three forms is about 24 h wall time (70 core-hours). M3 is about 3 h wall time per (form, transport) combination.
- [x] **MPI correctness fixes: done and merged 2026-10-01 (13 commits, tip `ac273512`).**
  - Static drop at R/h = 16 and 32 and sloshing at L/h = 32 and 64 complete on 1, 2, 4 and 8 ranks and match serial to round-off. Newton iteration counts equal serial (R/h = 32: 3.87 on every rank count, previously 7.47 on 2 ranks), and the rate initialization regularizes 0 rows on every rank count.
  - Fixes:
    1. Collective decisions in place of rank-local early returns around collectives: the master-bearing reimposition, the TimeLoop workspace and ghost exchanges, `FESystem` constraint refresh, wet-volume diagnostics, extension-map revisions and curvature-projection cache reuse.
    2. Ghost layers: 8 derived automatically for multi-rank aggregating decks when `<Ghost_layers>` is unset. This is a fixed geometric constant and is untested in 3D.
    3. Aggregation roots independent of the partition: only ranks that condense a slave must see its masters.
    4. Off-rank constraint fill exchanged into the distributed Jacobian pattern (FSILS had dropped entries), which restores quadratic Newton.
    5. Owned Dirichlet rows get their unit diagonal in `ParallelAssembler::finalize`.
    - No capillary-assembly bug was found.
  - Speed-up per step on a shared node: static drop R/h = 32 1.87× / 3.08× / 3.67× on 2/4/8 ranks; sloshing L/h = 64 1.52× / 2.02× / 2.30×.
  - Open: a few rank-local throws before collectives on error paths (NewtonSolver line search, `LevelSetVolume::build`, FSILS `dot`, PDE extension). No end-to-end multi-rank cut-case CTest yet.
  - Integrated check on `ac273512` (job `46205355`): serial outputs bitwise identical to the shared baseline on all 9 reference cases. FE 34/34, Physics 7/7, Application 4/4. With perf C, the Application suite now takes 1,824 s, down from 3,139 s.
- **Run policy since 2026-10-01:** long runs with FSILS linear algebra use 4 ranks on one node (`run_case_mpi.sbatch`). Bitwise comparisons between builds stay serial, and decks with Eigen linear algebra stay serial (benchmarks README).
- [x] **MPI follow-up (found and fixed 2026-10-05; branch `dev/mpi-followup`, merged locally as `017e397f`..`fd4a03c4`):**
  1. **Startup failure when a rank has no interface.** The capillary wave fails at step 0 on 4 and 8 ranks, and the sessile drop on 8 ranks, with "another communicator rank rejected embedded free-surface measure preflight" (`IncompressibleNavierStokesVMSModule.cpp`). Every failing run has a rank with no cut cells. Until this is fixed, the 4-rank policy does not cover those decks.
  2. **Default serial and parallel solves differ by up to 1.2e-7** (static drop R/h = 32), while 4 and 8 ranks agree to 1e-14. Serial needs 1.25–1.6× more GMRES iterations from the first solve, so the serial and parallel paths solve different systems or use different scaling.
  - **Fixes:**
    1. The interface marker is registered with an empty rule list on ranks without interface rules (`CutIntegrationContext::addGeneratedInterfaceDomain`). Before, those ranks threw in `assemble_on_marker` and desynchronized a collective. Rank-local failures are now logged before the collective that propagates them. Curvature projection also checks snapshot revisions only on ranks that have samples.
    2. FSILS row/column scaling gathered an int "continue" flag as `MPI_CXX_BOOL` (one byte per rank) into an int array, so parallel runs did 1 scaling sweep where serial did 2. The flag is now reduced as an int with `MPI_MAX`. The assembled systems were already identical.
    3. The maintenance consensus words included rank-local vector modification counters (`valueRevision()`). They are dropped; the content-binding algebraic revisions stay.
  - **Result:** serial vs 4 and 8 ranks agree to about 1e-13 with identical Newton, outer and GMRES counts, for the static drop R/h = 16 and 32, the sessile drop and the capillary wave. The capillary wave and sessile drop run on 4 and 8 ranks, and every maintenance transaction commits on 1/2/4/8 ranks.
  - Serial outputs are bitwise identical to the baseline. New end-to-end MPI test `Application_FreeSurfaceRankConsistency_MPI_4`.
  - **Open:**
    - fully dry ranks with small-cut aggregation turned off (the wet-volume diagnostic marker lookup);
    - `verify.py` `shape_rms_radial_deviation` reads partitioned output about 0.6% differently.
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
    - **Implemented and merged 2026-09-30** (commits `cd602301`, `1557d24d`, `41931bf2`).
      - Option `Curvature_projection_kinematic_area_gradient_mass` = `Consistent` | `Lumped`. With `Lumped` the filter coefficient defaults to 0, and an explicit nonzero value is rejected when the input is read.
      - After the rebase and the zero-filter default (job `46020877`): `test_fe_levelset` 371 passed + 1 declared skip; `test_fe_levelset_mpi` 21/21; binding 10/10 + MPI 1/1; `test_application` 379/379.
      - Before the rebase: lumped matched serial bitwise on 2 ranks.
      - Curvature quality on a sampled circle (n = 12/24/48/96):
        - The mass-weighted mean converges at second order, the same as consistent mode.
        - The trace error stays near 5.5% of 1/R and does not converge.
        - Nodal errors at vertices with a sliver of interface in their support grow to 23–347.
        - Filtered consistent mode (`c_l = 1`) does converge, with trace error 0.044 → 0.0022.
      - The M2 static drop will show whether the lumped noise shows up as spurious currents.
  - (c) **Unfiltered consistent-mass KAG**, for reference only.
  - **Fallback only if (a) and (b) both fail the D1 criteria:** KAG with the Helmholtz filter or a normal-gradient-stabilized mass. Its coefficient must be fixed once from dimensional scaling (P1), never tuned.
- [ ] **Time step.** Check Δt against the capillary constraint (§3.6). If needed, add a semi-implicit surface-tension term (Bänsch/Hysing) or use Δt below the limit.
  - Design note (2026-09-30): `Documentation/free_surface_semi_implicit_surface_tension_design.md`. It corrects §3.6: the outer geometry fixed point already makes accepted steps implicit in geometry. The capillary limit therefore appears as slow or divergent outer-loop convergence, contracting by about x/(1+b) per pass with x = Δt²ω² and b = 2νk²Δt, not as an unstable step.
  - At La = 12, viscosity may already permit 3–7 × the benchmark Δt.
  - Proposed, pending approval: a default-off lagged normal-increment term, `γ Δt_eff ∫ ∇_Γ((u − u_ref)·n_h)·∇_Γ(v·n_h)`. It is zero at convergence, so the accepted solution and the energy balance are unchanged, and it has no tunable parameter. The literal Bänsch/Hysing term is not recommended here, because it would double-count the step displacement. KAG stays explicit in geometry.
  - **Step-0 measurement done (2026-09-30; jobs `46075447` and `46076505`; results in `/scratch/users/zsexton/free-surface-benchmarks/static_drop_2d/step0/`).**
    - The per-pass contraction ρ fits A·r²/(1 + B·r) with r = Δt/Δt_B, within about 6%. The one-mode model has the right shape but is 4–12 × too pessimistic.
    - Steps accepted with default settings (12-pass cap): up to 2·Δt_B at La = 12, and up to Δt_B at La = 120 and 1,200.
    - The loop diverges (ρ = 1) at about 10–14·Δt_B for La = 12, 4–6·Δt_B for La = 120, and about 3·Δt_B for La = 1,200.
    - With γ = 0, a step takes 1–2 passes, so the extra passes come entirely from capillary geometry feedback.
    - **M2 time step at La = 12: 2·Δt_B**, which halves the step count. Use 4·Δt_B once the pass cap is about 25; the cap is a budget and does not change the converged state. Not beyond 4·Δt_B without the lagged-increment term, which mainly pays off at La ≥ 120.
    - A translating drop with γ = 0 stops at its first vertex crossing, confirming that M2 also depends on the vertex-crossing fix.
- [x] **Acceptance** (working criteria; fixed in `static_drop_2d/tolerances.json` on 2026-09-29):
  - pressure-jump error ≤ 1% at R/h = 32 with observed order ≥ 1;
  - final `Ca_sp` strictly decreasing with h, with absolute values reported. There is no absolute limit (decided 2026-09-29), because the July speed/γ figure was a start-up transient in SI units, not a capillary number;
  - no growth of `max|u|` in time;
  - volume drift ≤ 1e-4.
- [ ] **Selection.** After the M2 static drop, the M3 capillary wave, and the M4 sessile and Ren–E comparisons, record the chosen default route and the reason in §4.
  - Among candidates meeting the D1 criteria, prefer the one without tunable parameters (P1).
  - Then run the 3D sphere at R/h = 8, 16, 32.
- [ ] **3D readiness (2026-09-30; benchmark merged as `e687c6bd`, `12f30793`, `6350285e`).**
  - **Benchmark.** `tests/cases/fluid/free_surface_benchmarks/static_sphere_3d/` (16 Python tests), ready but unrun.
    - A sphere in the cube [0,3R]³, on an affine Tetra4 Kuhn mesh that is conforming and symmetric.
    - An irrational centre offset (min |φ|/h of 1.2e-3, 2.9e-4 and 6.5e-5 at R/h = 8, 16, 32).
    - Reference 2γ/R_eff, with R_eff from the exact P1 volume.
    - 2·Δt_B at La = 12, assumed from 2D (`--dt-multiple` to check).
    - PDE-extension transport by default; tolerances copied from `static_drop_2d`.
  - **Smoke run (job `46097410`, R/h = 8, pre-vertex-fix binary): every capillary form stopped at step 0.**
    1. The functional-consistency check `freeSurfaceFunctionalValueNear` (`FESystem.cpp`) allows 512 ulp. It compares the liquid volume summed per rule with the sum over about 0.3 M quadrature weights, and in 3D the rounding difference is larger. A rounding bound derived from the term count (or compensated sums) would fix it without changing the solution.
    2. The start-up transient moved φ across the nearest vertex, and the old binary aborted ("external-state discontinuity requires an adaptive step controller"). The merged vertex-crossing fix should cover this but is not yet checked in 3D.
  - **Cost (SKX, serial).**
    - R/h = 8 (82,944 cells): about 245 s per outer pass, so 16–25 min per step and 6–9 days per run (KAG several times more); peak RSS 6.1 GB.
    - R/h = 16: needs about 25 GB for snapshots, over the 16 GB allocation.
    - R/h = 32 (5.3 M cells): not feasible on one node.
    - Per pass at R/h = 8: setup cut context 122–153 s; aggregation constraints 30–33 s; Newton 65–69 s (3 iterations, 13.3 s per Jacobian, 9.5 s of it cut volumes); cut rebuild 120–142 s; KAG curvature projection about 400 s each, 2–3 per pass.
  - **Results-neutral speed-ups found by profiling** (raw data in `/scratch/users/zsexton/free-surface-benchmarks/profiling-3d/`):
    1. cache `MeshAccess::globalEntityIdsAvailable()` (scans every cell and face, called per cell, region or fragment; about 60 s per rebuild);
    2. make the KAG finite-difference diagnostic in `LevelSetCurvatureProjection.cpp` opt-in (131,760 strict cuts per projection; about 5 min each);
    3. index fragments by cell in `buildGeneratedActiveBoundaryDomain` (about 56 s per rebuild);
    4. hash the duplicate search in `collectLevelSetCurvatureSupplementalSamples` (about 30% of a KAG projection);
    5. a stable-id map in `buildFreeSurfaceGeometrySnapshot` (about 23 s per rebuild);
    6. stop the DOF-layout revision from invalidating the cut context every step (40% of a 3D tank step);
    7. skip aggregation when the topology is unchanged;
    8. classification-only records for fully dry cells (needed for R/h = 16 memory);
    9. basis tabulation and cache keys (25–35% of assembly).

    Items 1–5 bring a `SurfaceStress` pass to about 2 min, which is still 2–4 days per R/h = 8 run.
  - **FE performance work (started 2026-09-30).**
    - Every change must leave outputs bitwise identical against the shared reference set. The reference is job `46134332`, run with binary `35a81fd3`: plan `/scratch/users/zsexton/free-surface-benchmarks/perf-reference/jobs/reference.txt`, compared with `perf-reference/tools/cmp_runs.py`.
    - Running:

      | Branch | Speed-ups |
      |---|---|
      | `dev/perf-rebuild-path` | items 1, 3 and 5 above. **Merged 2026-10-01 (`14dd83e6`, `704cf54b`, `88b13550`), bitwise identical; FE 34/34, Physics 7/7, Application 4/4.** 3D sphere proxy R/h = 8: cut rebuild plus snapshot 125.9 → 31.1 s per rebuild (−75%), small-cut aggregation 16.4 → 2.3 s per call, step 0 2,509 → 1,867 s (−26%). 2D static drop R/h = 16: 149 → 128 s. Next quadratic hotspot: `LevelSetInterfaceDomain::twoSidedParentCellBindings` (fragments × regions, about 8–10 s of the remaining 34 s per 3D rebuild). |
      | `dev/perf-reuse-unchanged` | items 6 and 7. **Merged 2026-10-05 (`9fd3bd2f`, `1b0f4e1f`), bitwise identical in serial and on 2 and 4 ranks.** Cut-context signatures compare a content key of the level-set DOF layout instead of revision counters, and rollbacks keep a matching context. Small-cut aggregation is reused while its full input key is unchanged; the decision is collective. Steady step 16–22% faster in 2D (for example drop R/h = 16: 2.98 → 2.36 s) and 30–35% in the 3D tank (1/h = 16: 3.99 → 2.58 s). Aggregation full refreshes fall from about 700 to 2 per 2D run. Debug toggles (environment): `SVMP_RETAIN_RESTORED_CUT_CONTEXT=1` (opt-in, +1.8 GB on the sphere) and `SVMP_DISABLE_SMALL_CUT_AGGREGATION_REUSE=1`. |
      | `dev/perf-kernels` | item 2 and the cache-key and full-cell basis part of item 9. **Merged 2026-10-01 (`7cdc8842`, `371f521c`), bitwise identical on the whole reference set; FE 34/34, Physics 7/7, Application 4/4.** The finite-difference check is now opt-in (`Curvature_projection_kinematic_area_gradient_finite_difference_check`). 2D `kag_lumped` drop R/h = 8: 141.9 → 96.6 s (KAG's extra cost over `surface_stress` falls from +51% to +5%). 3D sphere proxy R/h = 8, step 0: 2,288 → 990 s; the check alone was 54% of the step. Per Jacobian: 14.0 → 12.1 s. |
      | `dev/perf-cut-integration-reuse` | reuse of cut-cell quadrature and basis data within a frozen geometry epoch. **Merged 2026-10-05 (`cc3d91f0`, `f38b10c8` as cherry-picked), bitwise identical.** Content-keyed cache in `FE/Assembly/CutVolumeEpochCache.h` with an 83% basis hit rate on the sphere: −13% of 3D cut-volume time, +121 MB peak RSS. Within noise in 2D after perf C. 3D sphere step with all merged speed-ups: about 540–735 s, against 2,000–3,300 s at the baseline. The cache rarely survives a time step, because `setup()` rebuilds the assembler. |
      | `dev/perf-linear-solver` | **Merged 2026-10-01 (13 commits, tip `4d12fa8e`); all options opt-in, defaults bitwise identical on the reference set.** New `<LS>` keys `<Right_preconditioner>` (`block-ilu0`, `simple`) and `<Preconditioner_reuse>`, plus an opt-in system dump and replay harness. Findings and results are listed below the table. |

    - **Linear-solver findings** (`dev/perf-linear-solver`):
      - On the base solver (FSILS GMRES with row/column scaling), linear solves are 11–33% of 2D Newton time and 7% in 3D. SpMV takes 55–68% of that and Gram–Schmidt 20–31%. Solves end only at restart boundaries, and 31% (2D) to 63% (3D) of the unknowns are Dirichlet identity rows.
      - None of the existing options helps: diagonal scaling, BiCGSTAB (frequent failures), NS/BlockSchur (very slow), Eigen ILUT and SparseLU (refactorized every solve).
      - Block ILU(0) as a right preconditioner, with break-even reuse (refresh when extra iterations reach the setup cost in iterations): 2D linear time 2.3–3.1× faster (iterations per solve about 5× fewer, for example 115 → 20), 3D 1.8×.
      - Newton, outer and inner counts are identical on all 7 FSILS 2D cases. Fields differ by ≤ 1.3e-8 of scale, and metrics by ≤ 1.1e-9 absolute.
      - SIMPLE: 1.6–2.2×.
      - An in-cycle unscaled-residual stopping estimate changes Newton counts, because the Newton and linear absolute tolerances are both 1e-10, so it stays opt-in.
      - `petsc/3.18.5` (module) looks usable through `-DFE_ENABLE_PETSC=ON`; `trilinos/12.12.1` is not.
      - **Rank check (2026-10-05; jobs `46633103`, `46634669`; `/scratch/users/zsexton/free-surface-benchmarks/ilu-rank-check/`):** keep block ILU(0) opt-in.
        - With ILU, 4 and 8 ranks differ from each other by up to about 8e-10 of the pressure scale and 5e-9 of the velocity scale, which is linear-solver tolerance. With the default, 4 and 8 ranks agree to about 1e-14.
        - ILU never changed Newton or outer counts across rank counts. In serial it changed Newton counts on the capillary wave (3–4 of 150 steps).
        - The per-step speed-up is only 1.03–1.18× (linear solve 2.0–3.5×), and it shrinks with ranks: iterations per solve grow 7–72% at 4–8 ranks.
        - Given the user's preference for rank-independent results, the default stays unchanged.
    - Started 2026-10-01 (base `904c65a5`, same bitwise gate):

      | Branch | Work |
      |---|---|
      | `dev/perf-pdeext-factorization` | keep the PDE extension's dry-region factorization and monolithic rows while the known set, walls and mesh are unchanged. **Done (`e4a08ec2`, `22c9485d`); merge held, see below.** Bitwise identical with a 99.4–99.8% hit rate; the extension's own cost falls about 89%. But the extension is only 1.6–1.9% of runtime, so the end-to-end gain is about 1–2%. The earlier "15–18% per step" figure compared whole steps with and without PDE transport, which includes the monolithic coupling's extra Newton work, not the extension solve itself. |
      | `dev/perf-functional-diagnostics` | **Merged 2026-10-05 (`84207885`, `d23e41e0`, `505e0cab`); bitwise identical; FE 34/34, Physics 7/7, Application 4/4.** Functional-record cost per step: 459 → 79 ms (3D tank 1/h = 16), 23.9 → 9.7 ms (2D drop R/h = 16); 60% of it was per-point allocation in the velocity evaluator, plus history regrowth every step. **Consistency checks** comparing sums of the same terms now use a derived rounding bound (γ_{n_a−1} + γ_{n_b−1})·Σ\|x\| from the term counts (`FE/Systems/FreeSurfaceFunctionalRounding.h`); checks without counts keep 512 ulp. **3D sphere R/h = 8 now runs:** step 0's mismatch is 1.89e-12, 4× the old bound and 0.5% of the derived one; 2 steps accepted at about 22 min each, vertex crossings handled. |
      | `dev/perf-log-output` | **Merged 2026-10-05, bitwise identical.** (1) Diagnostic lines are built only when `FE_LOG_LEVEL` prints them; the default output is unchanged, and at WARNING a step writes 16–26 kB instead of 94–115 kB. (2) `Write_velocity_extension_maps` (default false) makes the 1–5 MB per step wet-extension map files opt-in. (3) Hashed KAG duplicate-sample search, which was 50% of 3D projection time and is now about 0.1 s; a one-step sphere uses 28% less CPU time. A quieter default verbosity is proposed on `dev/perf-log-output-quiet-proposal` (`e2efc6cc`), pending user approval. Next 3D hotspot: `collectLevelSetCurvatureCutVolumeSupplementalSamples`, about 80% of projection time. |

    - Started 2026-10-05, after the merges: `dev/perf-hotspots-postmerge` and `dev/perf-dry-cell-records`.
      - `dev/perf-hotspots-postmerge` is the post-merge profiling pass. It takes in the perf C follow-ups, the remaining 3D KAG sample collection, `twoSidedParentCellBindings`, the duplicated boundary-partition validation and the formatting costs.
      - `dev/perf-dry-cell-records` stores classification-only records for fully dry cells, plus other lossless compaction, so that the 3D R/h = 16 case fits in memory.
    - Queued:
      - multithreaded assembly in `StandardAssembler`, after `dev/perf-hotspots-postmerge` merges (both edit the assembler broadly);
      - constraint-build cost (about 3 s per 3D tank step at 1/h = 16): started 2026-10-01 after the MPI merge, branch `dev/perf-constraint-build`.
    - **Build/JIT study: merged 2026-10-05 (`88a02419`, `f81bf270`, `0fea3646`). Everything is opt-in, and default builds and JIT cache keys are unchanged.**
      - **LTO+PGO** (`SV_ENABLE_LTO=ON`, `SV_PGO=GENERATE|USE`): bitwise identical; 8.5–11% faster on the 2D reference set (assembly −18%, cut volumes −19%), 3–4% on held-out cases, 1.6% on a sphere step (74% of it is geometry work that barely changes).
      - **Gains nothing:** LTO alone.
      - **Slower, and not bitwise:** `-march=x86-64-v3`, which also loses up to 4e-9 of field scale.
      - **JIT settings:** none worth changing. Opt-in overrides are `SVMP_JIT_CPU`, `SVMP_JIT_FP_CONTRACT` and `SVMP_JIT_OPT_LEVEL`.
      - **JIT cache:** keyed by CPU name and features, so there is no SKX/MLN mix-up. With `SVMP_JIT_CPU=x86-64-v3` plus a common `SVMP_CACHE_PROFILE`, one cache serves both node types and stays bitwise identical.
      - Documentation: `Code/Source/solver/FE/Docs/BuildOptimization.md`.
      - **Decided 2026-10-05 (D16–D18):** LTO+PGO once the speed-up branches settle; a shared JIT cache for benchmark runs; no round-off-changing JIT options.
    - Queued after the current merges: a fresh 2D/3D profiling pass of the post-merge step, attacking the next hotspots (results-neutral).
    - **Integrated check of the 2026-10-05 merges** (volume conservation, functional diagnostics, perf B, perf D, perf 8; tip `a4fb2de2`, jobs `46647477`/`46647480`/`46647482`): serial outputs bitwise identical on all 9 reference cases; FE 34/34, Physics 7/7, Application 4/4.
    - **Latent uninitialized-memory bug found (2026-10-01).** `activeCutContextMatchesRefreshCache`, the cut-context reuse decision, branches on a value from an uninitialized stack allocation in `buildFreeSurfaceGeometrySnapshot`. `validateFreeSurfaceGeometrySnapshotCurrentForMarker` reads uninitialized heap memory from `makeCutCellGeometryMapping`. Memcheck reports about 45k uninitialized reads on the tip.
      - It depends on code generation: with the PDE-cache branch, `MinimizedCircleSphereAndSessileCapsMeetProductionCertificates` fails when the Workflows suite runs in one process, apparently through stale cut-context reuse (residual 0.108 instead of 4.5e-17).
      - Fix in progress on `dev/fix-uninitialized-snapshot`, including a production-impact assessment.
      - The PDE-cache merge waits for the fix.
    - **Job efficiency (2026-10-01).** `seff` showed four causes of low CPU efficiency:
      - every serial benchmark job ran at exactly 50%, because `--mem=8G` exceeds amarsden's `MaxMemPerCPU=8000` MB and Slurm added a second, idle CPU;
      - build-and-test jobs at 11–12%, because CTest runs one entry at a time on 16 cores;
      - packed jobs at 18–30%, because their lanes are unbalanced;
      - agent build-then-run jobs at 21–44%, from the same two effects.

      Fixes: memory requests now stay at or below 8000 MB per CPU, and long runs use 4 ranks. Started `dev/test-suite-parallel`: finer CTest registration and `ctest -j` with MPI-aware `PROCESSORS`, plus separate build and test jobs.
    - Need user approval, because results change within solver tolerance: fewer outer passes (a better geometry predictor, or Jacobian reuse across passes). The lagged-increment term was approved as D13.
  - **Open questions (for the user):**
    1. 3D gating: R/h = 32 is not affordable, so gate at R/h = 16 with the order over 8/16, or report 3D without gating.
    2. For R/h = 16 memory: implement item 8, or allow a job larger than 16 GB.
    3. Keep the 3R box or move to 2.75R (23% fewer cells).
    4. Make the wet-extension map output opt-in; it is about 37 MB per step at R/h = 8.
- Optional accuracy improvement if spurious currents dominate: the Gross–Reusken improved Laplace–Beltrami (the projection uses a recovered, smoother normal).

### M3 — Dynamic capillarity

- [x] **Capillary-wave benchmark written (2026-09-30):** `tests/cases/fluid/free_surface_benchmarks/capillary_wave_2d/`.
  - Setup: half-wavelength domain with mirror side walls, a0 = 0.01λ, La = 3000 (ε = νk²/ω0 = 0.046).
  - Reference: Prosperetti's viscous initial-value solution (`prosperetti_reference.py`), checked against a Laplace-transform derivation to 1e-10 and against the inviscid and weak-damping limits. At this La, ω0 and 2νk² are off by 1.4% and 18%, so the comparison always uses the full viscous solution.
  - Metrics: amplitude from the exact cos(kx) coefficient of the P1 surface; frequency and damping from a damped-cosine fit; area as the maximum over the run (D11).
  - Protocol (D9, D10): `--transport` coupled / wet_extension / PDE extension, where the PDE extension becomes the default once it exists. One shared Δt at the λ/h = 64 capillary limit (2,900 steps) for the spatial study, plus a separate Δt/2, Δt/4 study at λ/h = 32.
  - Smoke run `46075460` (λ/h = 16, 0.1 period): amplitude within 7.5e-4 of the reference, area drift 4.9e-7 and growing.
- [x] **Capillary wave runs** (Prosperetti) at λ/h = 16, 32, 64 and three time steps. **Passed 2026-10-06 under the D19 protocol and D20** (`surface_stress`, PDE transport, reconciliation; job `46704019`; see D20 in §4). The earlier results below predate D13/D19. The PDE extension (D9) has merged, and `capillary_wave_2d/generate_case.py` now defaults to it (`3a9e3ba5`). Note: the wet extension writes about 1 MB of JSON map per step; make that output opt-in before long runs. Proposal: frequency error ≤ 2% and damping error ≤ 5% at λ/h = 32 (the 07-17 n = 16 run already had 1.2% frequency error).
  - **Submitted 2026-09-30 on binary `35a81fd3`:** `surface_stress` with `pde_extension` transport at λ/h = 16, 32 and 64 (shared Δt), plus Δt/2 and Δt/4 at λ/h = 32 (D10).
    - Jobs `46130411`, `46130412`, `46130432`, `46130438`, `46130568`.
    - Output in `/scratch/users/zsexton/free-surface-benchmarks/capillary_wave_2d/35a81fd3/pde_extension/surface_stress/`.
    - The KAG forms follow once M2 has compared the capillary routes.
  - **Result: all three gates fail** (verified with `verify.py`):

    | λ/h | frequency error | damping error | max dA/A |
    |---:|---:|---:|---:|
    | 16 | 0.80% | 26% | 1.1e-5 |
    | 32 | 0.66% | 7.8% | 1.3e-4 |
    | 64 | 0.23% | 0.06% | 7.3e-5 |

    - The frequency limit is met, but its observed order is 0.91 < 1.
    - Damping exceeds 5% at λ/h = 32.
    - The area deviation exceeds 1e-4 at λ/h = 32.
    - The Δt study at λ/h = 32 is not clean: damping error 7.8% / 9.3% / 12.1% at Δt, Δt/2, Δt/4, which grows as Δt shrinks.
    - The fitted capillary wave (M5) has 7–25× smaller frequency errors and 2–200× smaller area deviations.
    - Part of the frequency plateau at a0 = 0.01λ is a finite-amplitude effect that the linear reference does not contain. The fitted runs at a0 = 0.0025λ converge, with the amplitude dependence matching −0.10 to −0.16 (a0k)².
- [x] **Oscillating 2D drop**: Lamb frequency and viscous damping. **Passed 2026-10-07** (`tests/cases/fluid/free_surface_benchmarks/oscillating_drop_2d/`, approved 2026-10-07; no contact line).
  - Setup: mode n = 2 released from rest, r = R0(1 + ε cos 2θ) with ε = 0.01 (area-preserving), zero gravity, p_ext = 0, La = ργD/μ² = 800 (μ = 0.05, β/ω = 0.073, 4 periods damp the mode to 0.16), static-drop box mesh, R/h = 8, 16, 32.
  - Reference (`drop_reference.py`): exact linear viscous initial-value solution of the 2D drop (closed-form Laplace transform with the I_n Bessel ratio; residue sum over the complex pair and the real viscous modes). Checked against ω0² = n(n²−1)γ/(ρR³), the weak-viscosity expansion β = 2n(n−1)ν/R² (1 − (n−1)√(ε_ν/2)), ω = ω0(1 − √2 n(n−1)² ε_ν^1.5), the Stokes limit nγ/(2μR), Lamb's planar relation for large n, and an independent Chebyshev collocation of the stream-function equations (1e-9). At La = 800, ω0 and 2n(n−1)ν/R² are off by 1% and 13%. Finite-amplitude shift from the nonlinear inviscid drop (`finite_amplitude.py`): −0.770 ε², i.e. −7.7e-5.
  - Protocol, pre-registered in `tolerances.json` (`7e0e6867`) before the first run: `SurfaceStress`, PDE transport, reconciliation, lagged normal-increment term, 100 steps per inviscid period shared by all levels (50 would leave the time error, −0.13% on the capillary wave, above the expected R/h = 32 spatial error), Δt/2 at R/h = 32; frequency ≤ 2% and damping ≤ 5% at R/h = 32 with order ≥ 1 over 8/16/32, area ≤ 1e-4, Δt criterion 0.2% / 1%.
  - **Result** (job `46903541`, binary `4cbf2643-ltopgo`, 4 ranks at R/h ≥ 16; `verify.py` PASS on every criterion): frequency +2.70e-3 / +7.08e-4 / −1.37e-4 (order 2.15; about 1.7 after removing the time error estimated from Δt/2), damping +2.53e-2 / +4.29e-3 / +9.09e-4 (order 2.40), area ≤ 3.2e-6; Δt/2 at R/h = 32: 1.47e-4 and 1.75e-3; Δt criterion 2.8e-4 / 8.4e-4. 1 h 28 min for all six runs (R/h = 32: 62 min at Δt, 87 min at Δt/2). The 4-rank R/h = 32 runs log `off_rank_constraint_fill_outside_halo` (2–20 rejected columns, "increase <Ghost_layers>") at two times; a serial run of the same deck (job `46914683`, to step 72) matches to round-off before the first window and differs by 6.5e-9 in velocity and 5e-11 a0 in amplitude after it (within the nonlinear tolerance, decaying), so the gated metrics are unaffected but the 4-rank runs are not bitwise equal to serial there.

### M4 — Wetting (unfitted)

- [x] **D4 implemented and merged 2026-09-30** (commits `36b6966d`, `6ff1377d`, `fecba23d`, `846c745c`).
  - `PrescribedAngle` contact cells now use the angle-preserving wall maintenance: the kind `PreserveAcceptedAngle`, formerly `AcceptedDynamicAngle`.
  - The target-angle reset (`RepairToPrescribedAngle`) is fenced off: `requireAnglePreservingWallMaintenance` throws on every rank if it reaches production maintenance. The implementation stays for verification until M4.
  - Branch tests: `test_fe_levelset` 366 passed + 1 declared skip; MPI 19/19; Physics 7/7 (492 tests); `test_application` 377/377; `test_application_mpi` 32 tests.
  - Follow-up merged 2026-09-30 (`6ab328c8`): unfitted `PrescribedAngle` now requires Navier slip and the strong, normal-only, planar wall condition, as `DynamicRenE` already did. Without slip the setup fails with an explicit message. The rule covers axis-aligned planar walls only, and the orientation test covers ±z only (8 cases).
  - The historical runners `run_test05_velocity_growth_smoke.py` and `static_capillary_3d.py` now use slip, with slip length R/8 in the latter, so those decks differ from their earlier runs.
- [x] **Vertex crossings block moving-interface runs (found and fixed 2026-09-30).** In sessile smoke runs (R/h = 16), every configuration stops at or just after the first mesh-vertex crossing:
  - generalized-α: a cut-topology rejection, and bisection closes in on the crossing without passing it;
  - backward Euler with FSILS: `FsilsVector::dot: layout mismatch`;
  - backward Euler with Eigen and `SVMP_GENERATED_STATE_MAX_DISCONTINUITY_RESTARTS=4`: one crossing is accepted, then "Backward-Euler kinetic work does not bind ..." aborts the run.

  Before the crossing the physics looked right: 60° spreads (88.8° → 73.7°, base 0.627 → 0.656) and 120° recedes (91° → 103°).

  Fix merged 2026-09-30 (`cce6a0fd`..`6dd474d0`):
  - **Causes:**
    - Generalized-α rejected any topology change seen at the start of an attempt, and its final check required the endpoint topology to equal the stage topology, so bisection could never pass the crossing.
    - Some steps cycle A→B→A across a switching surface, which no step size removes.
    - `FsilsVector::dot` compared layout pointers, but small-cut aggregation rebuilds the layout object.
    - After a topology change the new constraints re-project the previous velocity, so the backward-Euler kinetic-work pairing threw.
  - **Changes:**
    - The initial canonicalization adopts the new epoch (not counted against the budget).
    - The generalized-α endpoint may start a new topology.
    - A revisited topology ends the step on that frozen epoch once the inner solve converges (`frozen_epoch_after_cycle=1`, `cut_topology_cycle action=accept_on_frozen_epoch`).
    - FSILS layouts are compared field by field.
    - An unpaired backward-Euler record is logged (`backward_euler_kinetic_work_binding status=unavailable`) and marks the energy history non-contiguous.
    - New key `GeneralSimulationParameters/Max_cut_topology_restarts_per_step`: the default is the outer iteration limit, and 0 restores stop-and-reject. The old environment variable applies only when the key is absent.
  - **Branch tests:** FE 32/32 (job `46089334`), Physics 7/7 and Application 4/4 (jobs `46089334`, `46099710`), pytest 72.
  - **Validation (job `46084684`, serial, R/h = 16, 300 steps to t = 1.31, 11% of T = 12.25):** all six configurations (60° and 120°; generalized-α + FSILS, backward Euler + FSILS, backward Euler + Eigen) reached 300 accepted steps with none rejected.
    - Topology changes per run: 48–138; cycles: 8–23 (3–8% of steps).
    - Backward Euler matches generalized-α within 0.3° and 0.1% in base; FSILS and Eigen give identical results.
    - A static drop without crossings is bit-identical to the pre-change build.
    - Cost: 0.77 s/step (60°) and 2.0 s/step (120°), so the full protocol takes about 0.6–1.6 h at R/h = 16, 6–13 h at 32 and 2.3–6 days at 64.
  - **Open questions (for the user):**
    1. Cycle acceptance: the inner solve meets the Newton tolerance, but there is no fresh zero-update check on a further topology.
    2. Whether to re-anchor the backward-Euler energy history after a topology change, since it is diagnostic only.
    3. Whether to remove `SVMP_GENERATED_STATE_MAX_DISCONTINUITY_RESTARTS`.
  - **Area drift (not addressed here):** the liquid area grows by 0.7–1.0% over 300 steps with coupled transport, about 100× the 1e-4 limit. With `--transport pde_extension` (harmonic, monolithic) the maximum drift falls to 2.45e-3 at 60° and 1.6e-3 at 120°, with similar angles, but that is still over the limit. The sessile protocol default is still coupled.
  - **Area drift fixed by kinematic reconciliation (merged 2026-10-05, `39804989`..`8cda8493`; opt-in `Enable_kinematic_reconciliation`, default off).**
    - **Attribution** (60°, 100 steps): 99.8% of the drift comes from transport. The Galerkin + SUPG residual satisfies the kinematic condition only in L2, and at a moving contact line the contact point runs ahead by O(h·Δt) per step, which does not converge away. Wall maintenance contributes 0 (it never ran), topology epochs 2.8e-6, measurement 0.
    - **Fix:** after each accepted step, a per-node correction on cut-cell vertices only makes the step's area change equal its trapezoidal kinematic interface flux. Guards: sign class kept, halfway limit, cut topology preserved.
    - No tunable parameter, no global shift, no volume target. Rank-independent reductions (2-rank tests pass).
    - **Full sessile runs** (R/h = 16, T = 12.25): maximum drift 6.1e-6 at 60° (was 2.6e-2) and 6.8e-5 at 120° (was 3.7e-3), so both pass D11. Angles and base radius are unchanged within 0.3%.
    - **Other cases:** static drop 1.0e-9, sloshing 4.6e-9, capillary wave 1.1e-11. With the option off, outputs are bit-identical. Cost +18–45% per step.
    - **Why not WP-6:** the conservative phase transport fails on this case, because contact-protected nodes block its local reconciliation.
    - **Decided 2026-10-05 (D14, D15):** enabled in the benchmark generators; solver default stays off; constants accepted for now; sessile transport is the PDE extension.
    - **Remaining issues:**
      - Residual drift comes from the generalized-α interface flux (about 4e-8·A per step at 120°).
      - Spurious near-zero wall vertices behind receding contact lines make `verify.py` reject the full runs (4 wall crossings), with or without the fix. **Resolved by the sign-definite patch bounds (D21, 2026-10-06).**
      - A pre-existing 2-rank `collective_consensus_rejection` with any post-accept maintenance option is handed to `dev/mpi-followup`.
- [x] **Configuration details for the D4 runs (all in place after `6ab328c8`):**
  - Young term;
  - Navier slip on the wetted wall;
  - strong no-penetration;
  - scale-only wall maintenance (`PreserveAcceptedAngle`), as on the dynamic-contact path;
  - no repair-to-target.
- [ ] **2D sessile relaxation** at 60°, 90° and 120°. **Protocol since 2026-10-06:** PDE transport (D15), reconciliation (D14) and patch bounds (D21). R/h = 16 passes every single-level check at 60° and 120° (D21 validation). The resolution study at R/h = 16, 32, 64 (60° and 120°; 4 ranks at 32, 8 at 64) runs on one binary that includes the FSILS layout-stamp fix (stale assembly slots after a layout rebuild, found by the uninitialized-read work), because sessile runs rebuild layouts at every topology change. Estimated from the R/h = 16 serial times (29 and 74 min on Milan): about 8 and 21 h serial at R/h = 32 and 6 and 15 days serial at R/h = 64. Benchmark scripts merged 2026-09-30: `tests/cases/fluid/free_surface_benchmarks/sessile_drop_2d/` (angles by local circle fit, within 0.13° at R/h = 16 on exact caps; slip length R/8; La = 12). Unblocked by the vertex-crossing fix. Generalized-α + FSILS is the single default configuration; `--transport coupled|wet_extension|pde_extension` selects the transport (`pde_extension` refused until D9 lands). The first 300 steps (above) show 60° at 59.5/59.3° with base error 2.8% and apex error 4.8%, still slowing down, and 120° at 115.3/118.6°, still receding (base error 13%). `verify.py` requires the last output at the end time. Start from a *non-equilibrium* shape (for example a 90° cap for a 60° target) and relax to equilibrium at R/h = 16, 32, 64. Proposal: angle error ≤ 2° at R/h = 32 and decreasing; base radius and apex height within 2%.
- [ ] **Capillary rise** against the prepared Gründing et al. envelope, using `free_surface_wp5_capillary_rise_reference.json` and the comparison runner.
- [ ] **Ren–E** advancing and receding: refine the 08-30 pilot at 3 meshes and 3 time steps, with slip length ratio ℓ_s/h = 2, 4, 8.
- [ ] **3D sessile** at one angle, then extend.

### M5 — Fitted ALE reference path

Merged 2026-09-30 (`f157011b`..`35a81fd3`, branch `dev/fitted-ale-m5`). Branch tests: FE 32/32, Physics 7/7 and Application 4/4 (job `46104940`); 77 benchmark Python tests. Output is in `/scratch/users/zsexton/free-surface-benchmarks/fitted_ale/`.

- [x] **Shared FE assembly bug fixed (`f157011b`).** `StandardAssembler::prepareContextFace` left the face rule in `cached_quad_rule_`, so every non-primary field on a boundary, interior or interface mesh face was evaluated at the canonical face points rather than the face-to-cell mapped points. Constants were exact; anything varying in space was wrong. In the fitted path it caused an 11% slow wave, linear Newton convergence and instability. The fix changes every face integral that uses a non-primary field (fitted kinematics, FSI interface, DG and other boundary terms). The unfitted static drop, sloshing and capillary wave are bitwise identical before and after the fix, at the same cost (job `46121663`).
- [x] Sliding mesh walls: a mesh `Dir` with `Value 0` and `Effective_direction` constrains only the selected components (`DirichletBC::active_components`), with the same input as a free-slip fluid wall.
- [x] **2D fitted sloshing passes D10–D12** (`fitted_sloshing_2d`, job `46104905`).
  - Setup: the same tank, mode, amplitude, viscosity and exact viscous reference as `linear_sloshing_2d`, on a liquid-only mesh.
  - Formulation: `Kinematic_enforcement=MeshNitsche` (the mesh row carries γ_N/h (w−u)·n (ψ·n); γ_N = 10 is fixed from the P1 trace inverse inequality), the `Free` tangential policy, and a harmonic operator on the mesh velocity (`Harmonic_quantity=velocity`). A displacement operator needs a Δt-scaled penalty whose flux defect accumulates into spurious damping (4× the reference at T0/64).
  - Frequency error (time error removed by Richardson extrapolation from the Δt study): −1.17e-3 / −2.67e-4 / −6.6e-6 at L/h = 16/32/64, least-squares order 3.7. The Δt study has order 2.11.
  - Damping error 6.6% / 0.37% / 1.1%; max dA/A 5.8e-7. Every step takes 2 Newton iterations; 0.93 s/step at L/h = 64.
  - The unfitted L/h = 64 errors are +1.5% (coupled transport) and +0.074% (PDE extension).
- [x] **Fitted `SurfaceStress`** is admitted behind `Allow_fitted_surface_stress=true`. It requires an explicit form, coupled ALE or a static mesh, a literal γ and no fitted contact model.
  - Static drop (`fitted_static_drop_2d`, job `46109392`): the relaxed pressure equals γ/(R cos(π/N)) of the inscribed polygon to 7 digits. The error against γ/R is 2.15e-3 / 5.36e-4 / 1.34e-4 at R/h = 8/16/32 (second order). Spurious μ|u|/γ peaks at 3.9e-8 and decays to roundoff; area is conserved to 1e-10.
  - Fitted `CurvatureTraction` with pointwise curvature now warns that it is zero on affine faces; it is not used.
- [x] **Fitted formulation fixes.** Coupled-displacement ALE is now assembled in the current configuration (the reference override had removed the gravity restoring force), and the surface Jacobian is no longer counted twice in `integrand*currentMeasure()`. The fluid equation must precede `mesh_motion`.
- [x] **The 05-26 step-12 failure is explained.**
  - At the old tip, the fitted SPHERIC 10 decks failed at step 0: unpreconditioned Eigen GMRES in 3D, and a Newton stall with FSILS in 2D.
  - With a direct solver the 3D tank "at rest" accelerated at about g/2 at the surface; that spurious growth was the historical failure.
  - After the frame, measure and face-sampling fixes the deck stays at rest to 1e-12 m/s.
  - Its legacy `Nitsche` mode replaces the normal dynamic condition and needs a pressure gauge, so it is not a free-surface model.
- [ ] **Open questions (for the user):**
  1. Confirm current-frame assembly for coupled ALE.
  2. Retire the legacy fitted `Penalty`/`Nitsche` kinematics for free surfaces, or keep them as legacy only, and make `MeshNitsche` the qualified default.
- [x] **Fitted 3D contact-line leak: root cause found and fixed (2026-09-30, merged `7698d5b7`).**
  - Cause: the legacy face files overlap. `generate_validation_meshes.py` classified faces by centroid with tolerance 0.35h, so `free_surface.vtp` also held all 88 top-row wall faces (12 overlapping file pairs).
  - A boundary face carries one label, which goes to the last face file listed (`MeshTranslator.cpp:595`, `:737`). The top-row wall faces therefore became free-surface faces with natural traction.
  - It is not a Dirichlet precedence problem: wall-normal velocity and displacement stay exactly 0 at every wall node, including shared edges, in a forced 3D run.
  - Fix: new decks with disjoint face sets. `MeshTranslator` and `generate_validation_meshes.py` now warn on overlapping face files; labels and outputs are unchanged.
- [x] **New `MeshNitsche` SPHERIC 10 decks** (`fitted_ale/..._meshnitsche`, 2D section 120×12 and 3D; written by `generate_spheric_test10_fitted_decks.py`; the legacy decks and their pinned tests are unchanged).
  - At rest over 1,000 steps: max|u| 7.9e-15 (3D) and 3.2e-14 m/s (2D), volume constant to roundoff.
  - 2D forced run (job `46155727`, full forcing table, no Coriolis because the solver supports it only in 3D): reached 1.514 s of 8.35 s.
    - Up to 1.40 s: volume change ≤ 4.6e-6, minimum angle ≥ 29.5°.
    - Sensor 1 first peak: 3.38 mbar at 0.925 s against 3.86 mbar measured at 0.953 s (RMS difference 0.18 mbar over 0–1.4 s).
    - It ends when the run-up at the left wall shears the wall cells (minimum angle 0.6° at 1.51 s) and Newton fails. The mesh-velocity operator has no restoring term, so this needs a mesh-quality policy.
  - Performance side fix (`d92af1cd`, Physics `NavierStokesRegister.cpp`): spacetime forcing tables now use a bucket grid and bisection, 16 → 2.0 s per step, bitwise identical.
- [ ] **Unfitted SPHERIC 02, 05 and 10 decks also have overlapping face sets.** In Test 05, for example, bottom faces get the free-slip front/back label. This may have affected the June results. Regenerate those decks; they have pinned tests.
- [ ] A mesh-quality policy for long or violent fitted runs: the mesh-velocity operator has no restoring term back to the reference mesh.
- [x] **Fitted capillary wave** (`fitted_capillary_wave_2d`, job `46129890`; same setup, reference and gates as `capillary_wave_2d`).

  | λ/h | frequency error | damping error | max dA/A |
  |---:|---:|---:|---:|
  | 16 | −4.1e-4 | +3.31% | 4.9e-6 |
  | 32 | −7.3e-4 | +0.51% | 1.4e-6 |
  | 64 | −4.3e-4 | −0.17% | 3.7e-7 |

  - Damping (order 2.14) and volume pass, and the frequency is within the 2% limit, but the frequency observed order is −0.04, so that gate fails.
  - The plateau is a finite-amplitude effect: at a0 = 0.0025λ the errors are +1.3e-4, −3.8e-5 and +2.6e-5.
  - Runs at 3.6–7.2× the capillary time-step limit are stable, because the geometry is solved inside Newton.
  - The Δt study converges at first order (error ≤ 1e-4), which is unexplained.
- [ ] **Open questions (for the user):**
  1. Capillary-wave frequency-order gate: lower the protocol amplitude to 0.0025λ, or use an amplitude-corrected reference. This affects unfitted M3 too.
  2. Overlapping face files: keep them as a warning or fail closed?
  3. Extend the Coriolis term to 2D, or run the forced SPHERIC case in 3D.
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
| `Documentation/free_surface_semi_implicit_surface_tension_design.md` | Relaxing the capillary time-step limit: outer-loop analysis and the proposed lagged normal-increment term (pending approval) |
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
