# Semi-implicit surface tension for the unfitted level-set free surface (design note)

**Status:** proposal, 2026-09-29; step-0 measurements added 2026-09-30 (§1.1). Approved as decision D13 on 2026-10-01. Implemented for `SurfaceStress` (`Surface_tension_semi_implicit=NormalIncrement`, default `None`) and validated on the static drop, capillary wave and sessile drop (§9, 2026-10-05).
**Base:** `origin/issue-449-modern-mesh-core` at `bfb3d53c`.
**Scope:** tracker §3.6, §5 M2 ("Time step" and the per-step cost item), §6.1. The design must respect D1–D8 and principle P1.
*Model* marks results of the single-mode analysis in §1. *(Unverified)* marks claims not yet checked against the original source.

## Summary

- **The limit here is a limit on outer-loop convergence,** not strictly an explicit stability limit. The outer fixed point iterates the frozen geometry to convergence, so an accepted step is already implicit in geometry. The capillary limit appears as a failure of that Picard loop to converge.
- **Recommended: a lagged-increment Bänsch term on the normal velocity** (eq. 4).
  - It is zero in every freshly refreshed residual, so the accepted solution, the acceptance test and the energy balance are unchanged.
  - It supplies the principal part of the missing geometry Jacobian.
- **Do not add the literal Bänsch/Hysing term.** It counts the step displacement twice.
- **Forms:** the term can be written with the current vocabulary. A reference-velocity field and its refresh hook are missing.
- **KAG stays explicit in geometry.**
- **No tunable parameter** is introduced.
- **Measured (§1.1).** The per-pass contraction follows the form of (1) but is 4–12 times smaller than the model predicts. At La = 12 the outer loop converges up to 10–14 times `Δt_B`, and `2Δt_B` works with the current settings. At La ≥ 120 the loop limit is 3–6 times `Δt_B`, so (4) matters most there.

## 1. The limit and the step counts it forces

A capillary wave on a free surface obeys `ω² = γk³/ρ`. Brackbill, Kothe and Zemach (JCP 100, 1992, 335–354) require `Δt < sqrt(ρ̄h³/(2πγ))` with `ρ̄ = (ρ₁+ρ₂)/2`. A one-sided surface has `ρ₂ = 0`, so

```text
Δt_B = sqrt(ρ h³ / (4π γ))        (the static_drop_2d rule; the tracker §3.6 form is sqrt(2) larger)
```

Viscosity relaxes this limit. Galusinski and Vigneaux (JCP 227, 2008, 6140–6164) give `Δt ≤ ½(c₂τ_μ + sqrt(c₂²τ_μ² + 4c₁τ_ρ²))`, with `τ_μ = μh/γ`, `τ_ρ = sqrt(ρh³/γ)` and empirical constants `c₁`, `c₂`.

**How the limit arises here.** The Newton unknown is the stage state `(u, p, φ)` at `t_{n+α_f}` (`TimeLoop.cpp`), and φ is advected by the coupled velocity inside Newton. The capillary term is assembled on `Γ_h(φ^(k))` held frozen, with no `dF/dφ`. `external_state_fixed_point` regenerates the geometry until a fresh refresh needs no Newton update, up to 12 passes.

**Model.** Take one mode with amplitude `a` and velocity `w`:

- the kinematics `a = a_n + Δt·w` are implicit;
- the viscous damping `2βw`, with `β = νk²`, is implicit;
- the force `−ω²a^(k)` comes from the previous pass.

The per-pass error factor is then

```text
ρ_c = x/(1+b),   x = Δt²ω²,   b = 2βΔt                       (1)
convergence iff  Δt < Δt_P = (β + sqrt(β² + ω²))/ω²            (2)
```

With truly lagged geometry the stability limit would be twice this (the Galusinski–Vigneaux form). At `Δt_B` and `k = π/h`, `x = π²/4` on every mesh, and `b = π^{3/2} μ sqrt(R/h)`.

The table uses the `static_drop_2d` settings: ρ = γ = R = 1, μ = sqrt(2/La), and T = 5 viscous times.

| La | R/h | Δt_B | steps at Δt_B | ρ_c at Δt_B (model) | Δt_P/Δt_B (model) | steps at Δt_P |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 8 | 1.25e-2 | 982 | 0.33 | 2.8 | 357 |
| 12 | 16 | 4.41e-3 | 2,779 | 0.24 | 3.8 | 733 |
| 12 | 32 | 1.56e-3 | 7,859 | 0.18 | 5.3 | 1,486 |
| 12 | 64 | 5.51e-4 | 22,229 | 0.13 | 7.4 | 2,994 |
| 120 | 32 | 1.56e-3 | 24,853 | 0.49 | 1.9 | 13,323 |
| 120 | 64 | 5.51e-4 | 70,294 | 0.37 | 2.5 | 28,193 |
| 1200 | 32 | 1.56e-3 | 78,591 | 1.08 (diverges) | 0.95 | — |

The benchmark README rounds the La = 12 row to 1,000 / 2,800 / 7,900 / 22,300 steps.

The R/h = 8 smoke run needed 9–10 passes per step. The measured contraction factor at that point is 0.065, not the model's 0.33 (§1.1).

**Model caveats:**

- it assumes a flat deep layer, a single mode, and the conservative damping `β = νk²`;
- it does not represent sliver cuts or P1 interpolation.

**Capillary wave.** At `Δt_B` a run needs `sqrt(2)(λ/h)^{3/2}` steps per period: 91, 256 and 724 at λ/h = 16, 32 and 64. At low Ohnesorge number viscosity does not relax this.

### 1.1 Measured contraction (step 0, 2026-09-30)

**Setup:**

- `static_drop_2d` with `surface_stress`, 10 steps per case;
- baseline binary `fef0d02f`, Slurm jobs `46075447` and `46076505`;
- runs and scripts under `/scratch/users/zsexton/free-surface-benchmarks/static_drop_2d/step0/`.

**Measurement.** `ρ` is the median ratio of successive fresh-pass residuals, skipping the first pass. Each cell of the table gives `ρ` and, in parentheses, the median passes per step.

- The default is at most 12 passes with an absolute gate of 1e-10.
- \* marks a case that failed on the 12-pass cap but was accepted 10/10 when rerun with `SVMP_GENERATED_STATE_OUTER_MAX_ITERATIONS=60`.
- "topo" means that step 0 was rejected with `CutTopologyChanged` while the loop was diverging.

| La | R/h | Δt/Δt_B = 0.5 | 1 | 2 | 4 | 8 | ρ = 1 at (fit) |
|---:|---:|---|---|---|---|---|---:|
| 12 | 8 | 0.024 (6) | 0.065 (8) | 0.160 (10) | 0.36 (16)\* | 0.79 (fails at 60) | 10 |
| 12 | 16 | 0.018 (6) | 0.047 (7) | 0.112 (9) | 0.25 (13)\* | 0.54 (26)\* | 14 |
| 120 | 8 | 0.034 (7) | 0.113 (9) | 0.34 (16)\* | 0.93 (fails at 60) | 2.1, topo | 4.4 |
| 120 | 16 | 0.027 (6) | 0.088 (9) | 0.25 (13)\* | 0.68 (40)\* | 1.5, topo | 5.7 |
| 1200 | 8 | 0.040 (7) | 0.149 (11) | 0.53 (29)\* | 1.6, topo | topo | 2.9 |
| 1200 | 16 | 0.033 (7) | 0.123 (10) | 0.42 (23)\* | 1.3 (cap) | 3.2, topo | 3.5 |

**Findings:**

- **The form of (1) is confirmed.** The fit `ρ = A r²/(1 + B r)`, with `r = Δt/Δt_B`, reproduces all 29 measured points, converging and diverging, to within about 6%. The fitted values are A = 0.13–0.18 against the model's 2.47, and B ≈ 0.26 times the model's b.
- **The model is 4–12 times too pessimistic.** Part of the gap is the generalized-α `Δt_eff = 0.533 Δt`, which the backward-Euler model ignores. The rest corresponds to a slowest mode of wavelength about 3h rather than 2h.
- **γ = 0 controls:**
  - A static drop needs 1 pass per step.
  - A drop in rigid translation (u = 0.1) needs 2 passes: the fresh residual drops by about 1e-9 in one pass. The observed 6–12 passes are therefore due entirely to capillary geometry feedback.
  - The translating drop fails at the first vertex crossing with "external-state discontinuity requires an adaptive step controller". `SVMP_GENERATED_STATE_MAX_DISCONTINUITY_RESTARTS=4` does not change this. **Any fixed-step run whose interface crosses a vertex ends there.** This is a risk for the multi-day M2 runs.
- **The La = 12 limit is ρ = 1 at 10–14 times `Δt_B`.** The practical optimum is lower, because passes grow as `log(gate)/log ρ`. Passes per unit simulated time are 7.5, 4.8, 3.5–4 and 3.3 or more at `r` = 1, 2, 4 and 8.
- **The semi-implicit term (4) matters mainly at La ≥ 120** (ρ = 1 at 3–6 times `Δt_B`) and for the capillary wave.

## 2. Methods in the literature

| Work | What changes | Reported time-step effect |
|---|---|---|
| Dziuk, Numer. Math. 58 (1991) 603–611 | Mean-curvature flow using `∫_{Γⁿ} ∇_Γx^{n+1} : ∇_Γv` | Unconditionally stable; the origin of the idea |
| Bänsch, Numer. Math. 88 (2001) 203–235 | Fitted surface with `x^{n+1} = xⁿ + Δt u^{n+1}`, which adds `γΔt∫_{Γⁿ} ∇_Γu^{n+1} : ∇_Γv` | Energy-norm stability proved for a space–time discretization; no factor quoted *(full text not reviewed)* |
| Hysing, IJNMF 51 (2006) 659–672 | The same term for a level set with finite elements and a regularized delta | Stable beyond the limit but not for arbitrary Δt (per Denner et al. 2022); factors *(unverified)* |
| Raessi, Bussmann, Mostaghimi, IJNMF 59 (2009) 1093–1110 | Hysing's term in finite volume/VOF | Limit exceeded "by at least a factor of 5" (abstract) |
| Sussman & Ohta, SIAM J. Sci. Comput. 31 (2009) 2447–2471 | Curvature from a volume-preserving mean-curvature-flow predictor | "At least three and sometimes five or more" speed-up at equal accuracy (abstract) |
| Zahedi, Kronbichler, Kreiss, IJNMF 69 (2012) 1433–1456 | Finite-element level set: spurious currents of sharp versus regularized forces | Addresses balance, not Δt *(no time-step content found)* |
| Denner & van Wachem, JCP 285 (2015) 24–40 | The limit reflects the need to sample capillary waves in time, whether surface tension is explicit or implicit | Without capillary waves, Δt can exceed the limit by orders of magnitude |
| Denner, Evrard, van Wachem, JCP 459 (2022) 111128 | VOF transport, momentum and continuity in one linearized system, with implicit CSF | Static drop at 50·Δt_σ; capillary wave accurate at 5·Δt_σ and stable at 50–100·Δt_σ; a Galusinski–Vigneaux-type limit remains |

Two related results:

- Popinet (Annu. Rev. Fluid Mech. 50, 2018, 49–75) relates semi-implicit terms to a surface viscosity of order `γΔt` *(quoted secondhand)*.
- Barrett, Garcke and Nürnberg (J. Sci. Comput. 63, 2015, 78–117) prove unconditional energy stability for fitted parametric interfaces.

**Takeaways:**

1. Semi-implicit terms stabilize by adding an O(Δt) surface viscosity.
2. The large gains come from coupling interface transport and force implicitly. The outer loop here already does this whenever it converges.
3. Capillary waves that are physical must still be resolved in time.

## 3. Application to `SurfaceStress`

### 3.1 The literal term and the time integrator

The residual is `γ∫_{Γ_h} P_h : ∇v`, with `P_h = I − n_h⊗n_h = ∇_Γ id`. Bänsch keeps the principal part of that residual on the displaced surface: `γ∫_{Γ_h} ∇_Γ(δx) : ∇_Γv`.

For first-order generalized-α (`GeneralizedAlpha.cpp`, with `a₀ = α_m/(γ_α α_f Δt)`), the kinematics `ẋ = u` give

```text
x_{n+α_f} − x_n = Δt_eff·u_{n+α_f} + α_f Δt (1 − γ_α/α_m) ẋ_n,   Δt_eff = γ_α α_f Δt/α_m = 1/a₀     (3)
```

- At ρ∞ = 0.5: `Δt_eff = 0.533 Δt`, and the history coefficient is `0.133 Δt`.
- For backward Euler: `Δt_eff = Δt`, with no history term.
- `Δt_eff` is exactly `FormExpr::effectiveTimeStep()`.

A literal term on `Γⁿ` would also need the interface velocity at `t_n`.

### 3.2 Why the literal term is wrong here

At acceptance, `Γ_h = Γ_h(φ_{n+α_f})` already contains the displacement, so adding `γΔt_eff∫∇_Γu : ∇_Γv` counts it twice. In the model:

- the fixed point becomes `(1+b+2x)w = …`, which is an O(Δt) surface viscosity;
- the per-pass factor becomes `x/(1+b+x)`, which tends to 1 as Δt grows;
- steady spurious currents (`u ≠ 0`) become Δt-dependent, which biases D1/D3.

### 3.3 Proposed term: a lagged normal increment

```text
R_SI(u; v) = γ Δt_eff ∫_{Γ_h^(k)} ∇_Γ((u − u_ref^(k))·n_h) · ∇_Γ(v·n_h) dΓ,
∇_Γ(w·n_h) = P_h (∇w)ᵀ n_h        (n_h is constant on each LinearCorner facet)       (4)
```

`u_ref^(k)` is the velocity of the projected iterate that generated `Γ_h^(k)`. It is overwritten at every refresh, so `R_SI ≡ 0` in every fresh residual.

- **Residual.** The accepted state satisfies the existing `R(y, G(y)) = 0` within the existing tolerances. The step result is the current scheme's, to within the outer tolerance.
- **Jacobian.** The term adds a constant, symmetric, positive-semidefinite `u–u` block, with no φ columns and no new sparsity. In the linearized stage transport, `δφ = −Δt_eff|∇φ| δu·n_h`, so the block is the Laplace–Beltrami part of the omitted `(dF/dφ)(dφ/du)`.
- **Model.** The per-pass factor becomes `|x_true − x_SI|/(1+b+x_SI)`. That is 0 for a single mode, for any Δt. For circle mode n, `x_SI/x_true = n²/(n²−1)` because κ² is lost on flat facets, so the factor is at most 1/4 for n ≥ 2.
- **Why the normal component only.** Equation (4) is Bänsch's identity with level-set kinematics `ẋ = (u·n)n`, and `(u·n)∇_Γn` vanishes on flat facets. The full `∇_Γu : ∇_Γv` also penalizes tangential velocity changes, which do not move a level-set interface. Used incrementally, it would slow tangential outer convergence by `x/(1+b+x)`. Keep the full form only as a test comparison: `inner(grad(u),grad(v)) − inner(grad(u)*n, grad(v)*n)`.
- **Time accuracy.** It is unaffected, because `R_SI` vanishes at convergence. An error in `Δt_eff` would change only the convergence rate. The integrator's stage weights on spatial terms apply to `R_SI` automatically.

### 3.4 Forms support

The term can be written with the current vocabulary:

```text
gu = transpose(grad(u - u_ref)) * n;   gu_t = gu - inner(gu, n) * n     (same for v)
(gamma * deltat_eff() * inner(gu_t, gv_t)).dI(bc.interface_marker),   n = generatedInterfaceOutwardNormal(bc)
```

Evidence that each ingredient already exists:

- **State-field gradients on `dI`:** two-fluid Nitsche (`sym(grad(u))*n`, `IncompressibleTwoFluidInterface.cpp` about L158–213) and the tangential pressure-gradient probe in `applyFreeSurfaceBoundary`.
- **The generated normal:** `FormExpr::normal()` on `dI`.
- **`effectiveTimeStep()`:** implemented in the interpreter and the JIT.
- **`discreteField`:** carries the KAG curvature.
- **Jacobian:** the existing linearization produces it; the term is linear in `u`.

**Missing:**

- the `u_ref` field and its refresh (application code);
- a vector overload of `surfaceGradient(f, n)`, which is scalar-only (optional);
- a JIT regression test for one-sided `InterfaceFace` terms.

## 4. Application to `KinematicAreaGradientTraction`

**KAG stays explicit in geometry.** An exact linearization `δκ_h = −M⁻¹(H_E δφ + δM κ_h)` would need three things:

1. **The nodal area-plus-Young Hessian `H_E`.** It is not implemented. It is nonsmooth at vertex touches and indefinite with φ-scale null directions, the obstacles that stalled the minimizer (§3.2).
2. **`M⁻¹`.** It is dense unless the mass is lumped.
3. **A traction/transport adjoint pairing.** That is the M8 energy-consistent KAG.

This is not feasible before M8.

Equation (4) can be added to KAG unchanged. It vanishes at convergence, so the balance property is preserved. However, KAG's response at sliver nodes (lumped nodal errors of 23–347 on 09-29) is far from `−γΔ_Γ`, so there is no model guarantee. Treat it as an experiment only.

## 5. Energy, outer loop, contact line, and P1

**Energy.**

- In the model, the converged scheme satisfies `E^{n+1} − Eⁿ + 2βΔt·w² + ½(Δw)² + ½ω²(Δa)² = 0`.
  - Energy decays unconditionally.
  - The capillary numerical dissipation `½γ∫|∇_Γ(Δt_eff·u·n)|²` per step is O(Δt) per unit time. This is the order of the literal schemes' surface viscosity.
- With (4) there is no extra term at convergence. The only numerical dissipation is the integrator's own, controlled by ρ∞.
- No discrete energy inequality is claimed, because Γ_h is regenerated from φ rather than moved in a Lagrangian way. Energy decay must be measured.
- The WP-8 ledger should report the work of `R_SI` separately. It should be at roundoff at acceptance.

**Outer loop.**

- The acceptance criterion is unchanged.
- Keep the optional delta-squared relaxation off while (4) is being assessed.

**Contact line.**

- Integrating (4) by parts linearizes the conormal pull inside `SurfaceStress`.
- The Young term `−γcosθ_e∫_CL v·m` stays explicit:
  - on a planar 2D wall it is a constant point force;
  - in 3D it scales with the line length, which is first order in k, so it creates no `h^{3/2}` limit.
- The Ren–E friction is already implicit.
- D4's single angle owner and the dissipation sign are untouched.

**P1.**

- The coefficient is `γ·Δt_eff`: a physical input times `1/a₀` from the integrator.
- The converged solution does not depend on the coefficient.
- Artificial-viscosity variants that scale the coefficient (for example Denner et al., Comput. Fluids 2017 *(unverified)*) are excluded.

## 6. Expected benefit

Once the loop converges independently of Δt, the remaining limits are:

- **Viscous:** none, because viscosity is implicit.
- **Transport CFL:** not binding for the static drop. The start-up speed is at most about 0.1 (estimate), which gives CFL ≤ 0.13 at R/h = 64 with Δt = 0.02. It binds for T1/T5 flows.
- **Accuracy:** the capillary time 1, the n = 2 mode period 2.57, and `t_μ` must be resolved. Δt = 0.02 gives 612, 1,936 and 6,124 steps at La = 12, 120 and 1200, for any R/h.
  - At La = 12 the gain over `Δt_B` is 1.6×, 4.5×, 13× and 36× at R/h = 8, 16, 32 and 64. Over a confirmed `Δt_P` it is 0.6×, 1.2×, 2.4× and 4.9×.
  - At La = 120 the gain is 13× at R/h = 32 and 36× at R/h = 64.
- **Capillary wave:** about 50 steps per period (to be confirmed by M3). Against 91, 256 and 724 steps, that is 1.8×, 5× and 14× at λ/h = 16, 32 and 64.
- **Unresolved mesh-scale capillary waves:** these are damped rather than resolved (Denner & van Wachem 2015). That is acceptable for static relaxation, but not where short waves are physical.
- **Per-step cost:** about 2 passes instead of about 10 in the model. The gain compounds with the step count *(unverified)*.

## 7. Implementation sketch

**New option:** `Surface_tension_semi_implicit` = `None` (default) | `NormalIncrement`.

**Files to change:**

1. **`FreeSurface/FreeSurfaceOptions.h`:** add the enum and the field.
2. **Parser:** parse the key in `NavierStokesRegister.cpp` (about L3828) and add it to the key list in `Parameters.cpp` (about L774).
3. **`IncompressibleNavierStokesVMSModule.cpp`:**
   - in `validateFreeSurfaceBoundary`, fail closed unless all of these hold: an unfitted interface; `SurfaceStress` or KAG; `RefreshedFrozenQuadrature`; a transient dt stencil; the outer fixed point enabled; coupled-field level-set velocity; literal γ > 0;
   - in `applyFreeSurfaceBoundary`, append (4) with a `diagnostic=` log line and a separate ledger work form;
   - add a field helper beside `freeSurfaceCurvatureField`.
4. **Reference field:** auto-register a prescribed vector field `ns_free_surface_semi_implicit_reference_velocity` in the velocity space.
   - Refresh it in the transient `synchronize_state` lambda (`ApplicationDriver.cpp` about L29482) at every projected or restored refresh, copying the velocity block and updating ghosts.
   - Put the helper in a new small translation unit (D7).
   - With frozen-map (wet-extension) transport, `u_ref` would have to be the previous advecting velocity. The first version rejects frozen-map transport.
5. **Documents:** `NavierStokesFreeSurface.md`, and tracker §6.1, §6.4 and M2.

**Tests:**

- with the option off, the residual and Jacobian are bitwise unchanged;
- with `u_ref = u`, the residual is unchanged;
- the Jacobian block is analytic for a planar interface in one `Triangle3`/`Tetra4` cell, symmetric positive semidefinite, with tangential and constant fields in its kernel, and it passes a finite-difference check;
- `Δt_eff` is 0.5333 Δt (generalized-α, ρ∞ = 0.5) and Δt (backward Euler);
- the validation rejections fire;
- 2-rank parity holds;
- an R/h = 8 run matches the run with the option off to within the outer tolerance, with pass counts reported.

**Benchmarks:**

- **Step 0 (current code, no new code):** R/h = 8 and 16; La = 12, 120 and 1200; `Δt/Δt_B` = 0.5, 1, 2, 4 and 8; 20 steps each. Record the passes per step and any failures, with a γ = 0 control. This resolves §3.6 and calibrates (1)–(2).
- **Static drop with the option:** La = 12 and 120; R/h = 8/16/32; Δt = 0.04/0.02/0.01, plus `Δt_B` reference runs. Apply the `tolerances.json` metrics. Proposed: pressure-jump difference ≤ 0.1% between Δt = 0.02 and 0.01, and `Ca_sp` reported for each Δt.
- **Energy:** `½ρ∫|u|² + γ|Γ_h|` must not increase at any Δt.
- **Capillary wave (M3):** λ/h = 16/32/64 at 100, 50 and 25 steps per period. Compare frequency and damping with Prosperetti; the observed order in Δt should be about 2.
- **KAG:** the same Δt scan with lumped KAG, as an experiment.

## 8. Risks, open questions, recommendation

**Risks:**

1. **At La = 12 the gain from (4) is limited.** The current loop already converges up to 10–14 times `Δt_B` (§1.1).
2. **Vertex crossings within large steps** make the fixed-point map nonsmooth, and the loop may cycle (`max_discontinuity_restarts`, step rejection).
3. **The approximation may be worse than the model** at sliver cuts, and under aggregation and SUPG.
4. **Conditioning** may suffer from interface stiffness `γΔt_eff` under FSILS GMRES/RCS.
5. **Numerical damping may bias `Ca_sp`.** Convergence in Δt must be shown.
6. **The contact-line mode at large Δt** has not been tested.
7. **No discrete energy inequality** holds.

**Open questions:**

- the definition of `u_ref` for frozen-map transport;
- a Δt-convergence criterion in `tolerances.json`;
- behaviour in 3D.

**Recommendation:**

1. **Step 0 is done (§1.1).** Use `Δt = 2Δt_B` for M2 at La = 12 now. Use `4Δt_B` once the outer gate is scaled or the pass cap is raised.
2. **If the loop fails near `Δt_B` at La ≥ 120 or at low Ohnesorge number, implement (4) for `SurfaceStress`.** Default the option to off and time-box the work to 3 days (D6).
3. **Implement neither the literal Bänsch/Hysing term nor a semi-implicit KAG operator.**
4. **Handle vertex crossings before the multi-day M2 runs.** A fixed-step run ends at the first cut-topology change (§1.1). Either add an adaptive step controller or monitor the smallest `|φ|/h`.

**Decisions needed:**

- accept the lagged-increment interpretation;
- adopt `Δt = 2Δt_B` (later `4Δt_B`) for M2 at La = 12;
- after validation, use a fixed physical Δt with a Δt refinement in M2 and M3;
- decide whether frozen-map support is in scope.

## 9. Implementation and validation (2026-10-01 to 2026-10-05)

### 9.1 What was implemented

- **Term.** Equation (4) with `n_h` the generated-rule normal and `dt_eff = 1/a0`
  (`FormExpr::effectiveTimeStep()`); Physics helper
  `FreeSurface/FreeSurfaceSemiImplicitSurfaceTension.{h,cpp}`, appended in
  `applyFreeSurfaceBoundary`. The velocity difference is formed before the
  projection, so the integrand is exactly zero where `u` and `u_ref` carry the
  same coefficients.
- **`u_ref`.** Prescribed field `ns_free_surface_semi_implicit_reference_velocity`
  in the velocity space, registered by `registerOn`. The application
  (`Application/Core/FreeSurfaceSemiImplicitReference.{h,cpp}`) copies the
  velocity coefficients into it at the projected outer fixed-point, projected
  endpoint and restored synchronization points and before each physical solve
  (three hooks in `ApplicationDriver.cpp`). The two fields share one DOF map,
  which is checked collectively once.
- **Transport.** The PDE extension equals the fluid velocity on all
  interface-cell vertices, so `u_ref` is well defined for `coupled_field` and
  both PDE couplings. The algebraic `wall_compatible_normal` and
  `nearest_interface_point` maps, plain prescribed or constant velocities,
  steady solves and runs without the outer fixed point fail closed.
- **Physics scope** (fails closed otherwise): exterior unfitted interface with an
  active side, `CutVolume`, `LinearCorner`, `RefreshedFrozenQuadrature` without
  shape tangents, literal `gamma > 0`, affine P1 `Triangle3`/`Tetra4` velocity,
  `SurfaceStress` (KAG admitted as an experiment).
- **Ledger.** The term enters no conservative or residual-work channel; it is
  zero at acceptance, except on frozen-epoch steps (§9.8).

Runs, logs and analysis: `/scratch/users/zsexton/svmp-dev-sist2/runs/`
(`smoke1-46231209`, `sist2-c1-46232162/analysis`, `recon-46645656`); campaign
at `9632beb7`. The runs of §9.3 to §9.6 predate decision D14 and ran without
kinematic reconciliation, like the M2 references; §9.7 adds one
reconciliation-on variant per case.

### 9.2 Unit and smoke tests

`test_FreeSurfaceSemiImplicitSurfaceTension` (6 tests, all pass):

- the velocity block for a planar interface in one `Tetra4` (horizontal and
  tilted planes, backward Euler and generalized-alpha) equals
  `gamma dt_eff |Gamma_K| n_c n_d (P grad N_a).(P grad N_b)` to 1e-12; it is
  symmetric positive semidefinite of rank 2, constant and tangential fields are
  in its kernel, nothing outside the velocity block changes, and
  `R_on - R_off = J_SI (u - u_ref)`;
- the full Jacobian passes a central finite-difference check;
- with `u_ref = u` the residual is bitwise unchanged;
- `dt_eff = dt` (backward Euler) and `8/15 dt = 0.5333 dt` (generalized-alpha,
  `rho_inf = 0.5`);
- with the option off nothing is registered and the configuration artifact is
  unchanged;
- each rejection fires before the system is modified.

Solver checks (jobs `46231209`, `46232162`):

- option off: VTU output and logs bitwise identical to the base binary;
- option on: the first fresh residual of a run is bitwise identical to the run
  without the term;
- 1 and 2 ranks: same outer passes at every step and fields within 1.4e-11.
- The application rejections (invalid token, outer fixed point disabled,
  `wall_compatible_normal` transport) end the run with their messages.

### 9.3 Static drop (`surface_stress`, PDE transport)

Every run completed. The pressure-jump error is `dp/(gamma/R_eff) - 1`.
Wall times are on 4 ranks at R/h = 32 and 16 and serial at R/h = 8, all on one
loaded 24-core node; the M2 R/h = 32 reference ran serially on another node.

| La | Δt | steps (any R/h) | Δp error, R/h = 8/16/32 | order | `Ca_sp`, R/h = 8/16/32 | max growth | max dA/A | passes at R/h = 32 (mean/max) | wall at R/h = 32 | verdict |
|---:|---|---:|---|---:|---|---:|---|---|---:|---|
| 12 | 0.04 | 309 | 6.74e-4 / 1.486e-4 / 3.270e-5 | 2.18 | 2.48e-4 / 1.30e-4 / 7.37e-5 | 0.79 | 1.2e-5 | 3.34 / 4 | 2,691 s | PASS |
| 12 | 0.02 | 618 | 6.76e-4 / 1.486e-4 / 3.273e-5 | 2.18 | 2.49e-4 / 1.30e-4 / 7.36e-5 | 0.79 | 1.1e-5 | 3.11 / 4 | 4,792 s | PASS |
| 12 | 0.01 | 1,236 | 6.84e-4 / 1.486e-4 / 3.273e-5 | 2.19 | 2.50e-4 / 1.31e-4 / 7.36e-5 | 0.79 | 8.3e-6 | 3.02 / 4 | 8,915 s | PASS |
| 12 | 2Δt_B (ref.) | 500 / 1,400 / 4,000 | 6.76e-4 / 1.487e-4 / 3.274e-5 | 2.18 | 2.49e-4 / 1.32e-4 / 7.42e-5 | 0.79 | 1.1e-5 | 2.96 / 6 | 90,694 s serial | PASS |
| 120 | 0.04 | 970 | 4.97e-4 / 1.39e-4 / 2.99e-5 | 2.03 | 1.50e-4 / 5.79e-5 / 2.59e-5 | 1.00 | 1.6e-4 (R/h = 8) | 3.16 / 5 | 8,543 s | volume fails at R/h = 8 |
| 120 | 0.02 | 1,938 | 5.10e-4 / 1.39e-4 / 2.99e-5 | 2.05 | 1.32e-4 / 5.63e-5 / 2.57e-5 | 1.00 | 1.2e-4 (R/h = 8) | 3.07 / 4 | 15,845 s | volume fails at R/h = 8 |
| 120 | 0.01 | 3,900 | 5.25e-4 / 1.40e-4 / 2.99e-5 | 2.07 | 1.09e-4 / 5.24e-5 / 2.56e-5 | 1.00 | 8.2e-5 | 2.82 / 4 | 28,845 s | PASS |
| 120 | Δt_B (ref.) | 3,200 / 8,800 / 24,900 | 5.21e-4 / 1.40e-4 / – | – | 1.16e-4 / 4.30e-5 / – | 1.007 (R/h = 16) | 9.2e-5 | 3.64 / 5 (first 500 steps) | about 239,000 s projected | growth fails at R/h = 16 |

- **Δt criterion.** `|Δp(0.02) - Δp(0.01)| / Δp(0.01)` is at most 5.9e-6 (La = 12)
  and 5.2e-6 (La = 120) over all levels; the 0.1% criterion holds with three
  orders of margin. `Ca_sp` changes by at most 1.2% between Δt at R/h = 32; at
  R/h = 8 and 16 and La = 120 it falls by up to 27% from Δt = 0.04 to 0.01
  (risk 5), and stays monotone in h at every Δt.
- **Volume at La = 120, R/h = 8.** The drift grows with Δt (8.2e-5, 1.2e-4,
  1.6e-4 at Δt = 0.01, 0.02, 0.04; 9.2e-5 at Δt_B). It is a level-set transport
  time error at the coarsest level; R/h = 16 and 32 stay below 2e-5.
- **Cost.** Passes per step stay at 2.8 to 3.4 on average (at most 5) for steps
  of up to 26 Δt_B, so the wall time follows the step count. At R/h = 32,
  measured in total outer passes (machine independent), Δt = 0.02 needs 6.2
  times fewer than the La = 12 reference and 15 times fewer than the projected
  La = 120 reference; Δt = 0.01 needs 3.2 and 8.2 times fewer. At the protocol
  step itself the term lowers the passes slightly (La = 12: 3.18 against 3.60 at
  R/h = 8, 3.01 against 3.16 at R/h = 16, maximum 4 against 6).
- **KAG (experiment, R/h = 8, La = 12).** `kag_lumped` with the term at Δt = 0.04
  and 0.02 reproduces the M2 KAG result at 2Δt_B (Δp error -1.8e-4, `Ca_sp`
  1.5e-3, growth 1.32, so it still fails growth) with 5.2 and 4.4 passes per
  step against 4.9. The term does not remove the KAG defects.

### 9.4 Energy

`E = (1/2) rho int |u|^2 + gamma |Gamma_h|` (plus the Young wall energy for the
sessile drop) from the per-step functional record:

- **Static drop and capillary wave.** `E` decreases at every accepted step at
  every Δt and level, except one step each at Δt = 0.04 and R/h = 32
  (La = 12: +2.2e-6, relative 3.5e-7; La = 120: +4.8e-6, relative 7.6e-7). Both
  are start-up steps accepted on a frozen epoch after a topology cycle (§9.8).
- **Sessile drop.** `E` increases on 69% of the steps by up to 2e-5 relative
  both with and without the term (identical histories). The run also loses
  2.9e-3 of its area, so the discrete energy balance is not closed there; this
  is independent of the term.

### 9.5 Capillary wave (λ/h = 16/32/64, PDE transport, La = 3000)

Errors are against the fitted Prosperetti solution. The protocol row is the M3
run at the shared step (Δt = 5.5e-4, 725 steps per period, term off).

| steps/period | Δt/Δt_B at λ/h = 16/32/64 | a0/λ | ω error, 16/32/64 | spatial order | β error, 16/32/64 | dA/A max at 32 | passes at 64 (mean/max) | wall at 64 |
|---:|---|---:|---|---:|---|---:|---|---:|
| 100 | 0.9 / 2.6 / 7.2 | 0.01 | 1.78e-2 / 7.33e-3 / 2.32e-3 | 1.47 | 0.216 / 0.068 / 2.3e-3 | 1.27e-4 | 4.34 / 6 | 4,072 s |
| 50 | 1.8 / 5.1 / 14.5 | 0.01 | 1.80e-2 / 6.47e-3 / 1.09e-3 | 2.02 | 0.216 / 0.068 / 5.9e-3 | 1.26e-4 | 4.90 / 6 | 2,283 s |
| 25 | 3.6 / 10.2 / 29 | 0.01 | 1.38e-2 / 1.80e-3 / 3.86e-3 | 0.92 | 0.214 / 0.057 / 1.86e-2 | 1.24e-4 | 5.28 / 7 | 1,483 s |
| 100 | as above | 0.0025 | 1.60e-2 / 5.59e-3 / 1.80e-3 | 1.58 | 0.220 / 0.072 / 7.0e-3 | 8.4e-6 | 3.19 / 5 | 2,675 s |
| 50 | as above | 0.0025 | 1.62e-2 / 4.73e-3 / 4.96e-4 | 2.51 | 0.220 / 0.072 / 3.4e-3 | 8.3e-6 | 3.42 / 6 | 1,567 s |
| 25 | as above | 0.0025 | 1.21e-2 / 9.9e-5 / 4.10e-3 | 0.78 | 0.218 / 0.063 / 1.01e-2 | 8.2e-6 | 3.93 / 5 | 997 s |
| 725 (protocol, off) | 0.12 / 0.35 / 1.0 | 0.01 | 8.02e-3 / 6.59e-3 / 2.28e-3 | 0.91 | 0.264 / 0.079 / 5.8e-4 | 1.29e-4 | 4.20 / 6 | 10,442 s (other node) |

- **Without the term** the run fails at the first step at 100 steps per period
  for λ/h = 32 (2.6 Δt_B) and 64 (7.2 Δt_B); at λ/h = 16 it runs with 4.0 and
  6.6 passes per step (100 and 50 steps per period) against 3.0 and 3.3 with
  the term.
- **Order in Δt.** At λ/h = 64 the frequency converges with order 2.0
  (a0 = 0.01) and 1.8 (a0 = 0.0025), the damping with 1.8 and 1.9; at λ/h = 32
  the frequency order is 2.4. The Richardson time error at λ/h = 64 and
  a0 = 0.01 is -0.04%, -0.16% and -0.66% in frequency and -0.15%, -0.5% and
  -1.8% in damping at 100, 50 and 25 steps per period. This is the
  generalized-alpha phase error, about `(ω dt)^2/12`.
- **Gates.** The frequency gate passes at 100 and 50 steps per period (order
  1.5 and 2.0; 1.6 and 2.5 at a0 = 0.0025) and fails at 25, where the time
  error at λ/h = 64 exceeds the spatial error. The damping error at λ/h = 32
  (6 to 7%) is spatial and fails the 5% gate as in the protocol run. The
  area gate fails at λ/h = 32 for a0 = 0.01, as in the protocol run, and
  passes at a0 = 0.0025 (at most 1.6e-5).
- **Amplitude.** At 50 steps per period the λ/h = 64 frequency error halves from
  1.1e-3 to 5.0e-4 at the smaller amplitude, consistent with a finite-amplitude
  plateau.

### 9.6 Sessile drop (60°, R/h = 16, two viscous times)

| step | term | passes (mean/max) | θ_L, θ_R | base, apex error | wall |
|---|---|---|---|---|---:|
| 2 × protocol (2.0 Δt_B) | on | 3.55 / 6 | 60.19°, 56.66° | 1.7e-3, 7.0e-3 | 774 s |
| 2 × protocol | off | 4.46 / 8 | 60.19°, 56.66° | 1.7e-3, 7.0e-3 | 879 s |
| 4 × protocol (4.0 Δt_B) | on | 4.06 / 6 | 60.20°, 56.69° | 1.7e-3, 7.0e-3 | 444 s |
| 4 × protocol | off | 5.74 / 9 | 60.20°, 56.69° | 1.7e-3, 7.0e-3 | 559 s |

The contact points behave identically with and without the term (risk 6 not
observed); the term lowers the passes by 20 to 30%.

### 9.7 With kinematic reconciliation (D14)

One variant per case with the term and `Enable_kinematic_reconciliation=true`
(job `46645656`, at `b716620c`):

| case | Δt | volume drift (off → on) | other metrics (off → on) | passes (mean/max) | energy increases |
|---|---|---|---|---|---|
| static drop La = 120, R/h = 8 | 0.04 | 1.6e-4 → 2.3e-9 | Δp error 4.97e-4 → 5.13e-4, growth 0.85 → 0.78 | 3.13 / 5 | 400 steps, at most 1.1e-7 |
| static drop La = 120, R/h = 8 | 0.02 | 1.2e-4 → 5.3e-10 | Δp error 5.10e-4 → 5.26e-4 | 3.05 / 4 | 460 steps, at most 2e-8 |
| static drop La = 120, R/h = 16 | 0.04 | 1.8e-5 → 5.6e-8 | growth 0.999 → 0.981 | 3.81 / 5 | 434 steps, at most 4e-8 |
| static drop La = 12, R/h = 8 | 0.04 | 1.2e-5 → 7.2e-10 | Δp error 6.74e-4 → 6.75e-4 | 3.39 / 4 | none |
| capillary wave λ/h = 32, 50 steps/period | P/50 | 1.26e-4 → 2.0e-8 | ω error 6.5e-3 → 3.9e-3, β error 0.068 → 0.071 | 3.60 / 4 | none |
| capillary wave λ/h = 64, 50 steps/period | P/50 | 6.7e-5 → 2.1e-6 | ω error 1.1e-3 → 1.4e-3, β error 5.9e-3 → 4.0e-3 | 4.78 / 6 | none |
| sessile drop 60°, 4 × protocol | 4.0 Δt_B | 2.9e-3 → 7.8e-5 | θ_L, θ_R 60.20°, 56.69° → 60.64°, 58.03° | 4.21 / 6 | 38 steps |

Reconciliation removes the La = 120, R/h = 8 volume failure at Δt = 0.04 and
0.02 and the capillary-wave area failure, without changing the passes per step.
The accepted-step area correction raises `gamma |Gamma_h|` slightly on some
static-drop steps (relative increase at most 2e-8).

### 9.8 Risks found

1. **Frozen-epoch acceptance.** When the outer loop cycles between two cut
   topologies, the step is accepted on a frozen epoch without a final refresh;
   `R_SI` is then not zero at acceptance, and the geometry is not
   self-consistent. This occurred in start-up steps at large Δt (static drop
   R/h = 32 at Δt = 0.04: 4 and 6 cycle steps) and coincides with the only two
   energy increases without reconciliation (§9.4).
2. **Vertex crossings** are frequent at large Δt (up to 89 restarts in 400
   capillary-wave steps at λ/h = 64) and are handled by the topology restarts;
   no run failed.
3. **Conditioning.** GMRES iterations per inner solve rise from about 52 to up
   to 150 at the largest steps; all linear solves converged.
4. **Δt bias.** `Ca_sp` at coarse levels and the La = 120, R/h = 8 volume drift
   depend on Δt; the gated pressure jump does not.

### 9.9 Recommendation (the protocol choice is the user's)

- **M2:** a fixed physical step for all R/h: Δt = 0.02 at La = 12 and
  Δt = 0.01 at La = 120, each checked against a run at twice or half the step
  (the 0.01 and 0.02 runs of §9.3 already serve). La = 120 at Δt = 0.02
  passes every gate except the R/h = 8 volume drift, which reconciliation
  (§9.7) removes; with reconciliation on, Δt = 0.02 is also adequate at
  La = 120. At R/h = 32 this replaces
  4,000 and 24,900 steps by 618 and 3,900.
- **M3:** 50 steps per inviscid period at every level, with a 100-step check.
  The time error at 50 steps per period is 0.16% in frequency and 0.5% in
  damping at λ/h = 64.
- **`tolerances.json` Δt criterion.** Every gate passes at both steps, and the
  gated quantity changes by at most 0.1% under halving: the pressure jump for
  M2, and frequency and damping (at most 0.2% and 1%) at the finest level for
  M3. `Ca_sp` is reported for each step.
