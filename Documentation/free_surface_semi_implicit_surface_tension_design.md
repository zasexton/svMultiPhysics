# Semi-implicit surface tension for the unfitted level-set free surface (design note)

**Status:** proposal, 2026-09-29. No code has been changed.
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
- **Measure first.** At La = 12 the model says viscosity already allows 3–7 times the current Δt.

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

The R/h = 8 smoke run needed 9–10 passes per step. That matches `ρ_c ≈ 0.33` with the absolute 1e-10 gate. It is consistent with the model but does not prove it.

**Model caveats:**

- it assumes a flat deep layer, a single mode, and the conservative damping `β = νk²`;
- it does not represent sliver cuts or P1 interpolation.

**Capillary wave.** At `Δt_B` a run needs `sqrt(2)(λ/h)^{3/2}` steps per period: 91, 256 and 724 at λ/h = 16, 32 and 64. At low Ohnesorge number viscosity does not relax this.

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

1. **The La = 12 premise** may not hold (§1).
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

1. **Run step 0 now.** It takes hours and needs no code.
2. **If the loop fails near `Δt_B` at La ≥ 120 or at low Ohnesorge number, implement (4) for `SurfaceStress`.** Default the option to off and time-box the work to 3 days (D6).
3. **Implement neither the literal Bänsch/Hysing term nor a semi-implicit KAG operator.**
4. **Keep M2 La = 12 at `Δt_B`,** or at a measured `Δt_P`, until (4) is validated.

**Decisions needed:**

- accept the lagged-increment interpretation;
- run step 0 before the M2 launch;
- after validation, use a fixed physical Δt with a Δt refinement in M2 and M3;
- decide whether frozen-map support is in scope.
