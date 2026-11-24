# Reusable boundary laws in Forms

The original review and pipe comparisons used `issue-449-modern-mesh-core` at `a41f0bec7`.

A physics developer should be able to write a boundary law once in the notation of its weak form. A problem author should be able to apply that law to several boundaries, each with its own data and initial conditions. The residual, Jacobian, and recorded outputs must all read the state belonging to that application.

Forms supplies fields, traces, integration domains, coefficients, quantities, and symbolic equations. The FE system binds those symbols to a problem and manages storage and assembly. Physics modules supply the constitutive equations, signs, parameter meanings, and admissible configurations. [Forms vocabulary](VOCABULARY.md#design-principles)

## Implementation status as of 2026-11-24

- **Wave 3 implements the instance binding on `issue-449-modern-mesh-core`.** Commit `be26f57c3` binds each state used by an inlined output to its instance and exposes the bound names for inspection. Commit `5ed4d69a2` also resolves each input port once, so a model with two inputs reads the two quantities it was given. Both landed in integration batch 12 (`88fdc80fd`).
- **Wave 4 owns the authoring interface.** Decision D76 assigns suggestions 2–5 below to that work, including the measured quantity's Jacobian, selectable trace enforcement, oriented geometry, and inspectable contributions. The C++ sketch illustrates what a module author could write. The public spelling is still a design choice. [Decision D76](../../../../../Documentation/free_surface_program_tracker.md#L1185) · [Forms vocabulary entry](VOCABULARY.md#planned-reusable-boundary-laws)

## What the outlet example reveals

The Navier–Stokes factory defines the resistance–capacitance–resistance (RCR) outlet model once. For outlet $i$, it measures $Q_i=\int_{\Gamma_i}u\cdot n_i\,ds$, deploys a model with its own resistances, capacitance, and initial capacitor pressure $X_i(0)$, and uses $P_i=X_i+R_{p,i}Q_i$ in the traction. [Model definition](../../Physics/Formulations/NavierStokes/NavierStokesBCFactories.h#L240) · [Per-outlet binding](../../Physics/Formulations/NavierStokes/NavierStokesBCFactories.h#L482)

At `a41f0bec7`, output expansion substituted the bound input and parameters and could leave a differential state at its model-local slot. Each RCR definition used slot zero for $X$. The assembled kernel read a single array containing all instances’ states, so both tractions could read the first capacitor pressure. The scalar histories still evaluated each model separately. [Output expansion](../Systems/FormsInstaller.cpp#L4893) · [State substitution](../Systems/FESystem.cpp#L25106) · [Assembly state layout](../Systems/FESystem.cpp#L24756)

| Check at `a41f0bec7` | Observation |
| --- | --- |
| Two compliant RCR outlets | Reversing only the BC declarations changed the mean fluid pressure at step 3 from **4.92649** to **1.98792**. The two recorded capacitor pressures were **4.91736** and **1.98641** in both orders. |
| Two RCRCR outlets, each with two capacitor states | Reversing the declarations changed the mean fluid pressure at step 1 from **4.98164** to **1.99631**. In a mixed RCR and RCRCR pair, changing the RCRCR's initial $P_1$ from 2 to 8 changed its reported outlet pressure by nearly 6. The fluid pressure changed by less than $1.2\times 10^{-12}$. |
| Prescribed pressure, strong velocity, and pure resistance ($C=0$) | Reversing two distinct entries reproduced the nodal fields in each case. |
| Weak velocity enforcement | The initial assembled load equaled the sum of the two individual loads in either order. The transient pipe solve did not converge in either order. |
| Two fitted free-surface patches | Reordering patches with exterior pressures 5 and 2 preserved the fields to roundoff. Exchanging the values moved the pressure response to the opposite patch. For unfitted surfaces, the Navier–Stokes module allows one surface to restrict pressure operations. [Active-domain rule](../../Physics/Options/NavierStokesOptionRules.cpp#L596) |

To repeat the first check, use the pipe case's no-slip wall and give each end face a Neumann RCR block. Set $(R_p,C,R_d,P_d,X_0)$ to $(12,0.01,90,0,5)$ on `lumen_inlet` and $(25,0.02,110,0,2)$ on `lumen_outlet`. Set `Write_scalar_history` to true and `Increment_in_saving_VTK_files` to 1. Run three steps of $0.005$ with generalized-alpha spectral radius $0.5$. Reverse the two `<Add_BC>` blocks, keeping each block's face name and values together, and repeat. Compare the saved fluid pressure and both scalar histories at $t=0.015$. [Starting pipe case](../../../../../tests/cases/fluid/pipe_RCR_3d/solver_new.xml)

**After the instance-binding fix**, the two-rank pipe runs gave a mean fluid pressure of **3.534290099** in both declaration orders at step 3. Matching all 3,107 vertices by global ID, the largest difference between orders was $8.9\times10^{-16}$ in pressure and $1.1\times10^{-16}$ in any velocity component. The capacitor pressures remained distinct, **4.90796** on `lumen_inlet` and **1.99113** on `lumen_outlet` in both orders.

## Proposed authoring interface

The following C++-like sketch shows the pressure contribution of a compliant RCR. The module supplies the velocity field `u` and its test function `v`. All API spellings in the sketch are illustrative.

```cpp
auto law = boundaryLaw();
auto G = law.boundary();
auto n = G.outwardNormal();
auto Q = law.quantity("Q", G.integrate(dot(u, n)));
auto X = law.scalarState("X");
auto [Rp, C, Rd, Pd] = law.parameters("Rp", "C", "Rd", "Pd");

law.equation(dt(X) == (Q - (X - Pd) / Rd) / C);
auto P = law.output("P_out", X + Rp * Q);
law.add(G.integrate(P * dot(n, v)));
```

The problem loader binds the same definition for each input entry. Here `X` denotes one scalar per application over the selected boundary. A surface field would express a spatially distributed unknown.

```cpp
auto left  = problem.bind(law, left_face,  left_params,  initial(X, left_X0));
auto right = problem.bind(law, right_face, right_params, initial(X, right_X0));
```

The same vocabulary can express a lumped thermal reservoir. A heat-transfer module would supply the heat flux integrand, the reservoir's evolution equation, and its contribution to the thermal residual. The Poisson module already provides another concrete client of state-dependent boundary outputs. [Poisson boundary flux](../../Physics/Formulations/Poisson/PoissonBCFactories.h#L113)

## Design suggestions

1. **Bind symbols to a law instance.** Every BC added to a problem is its own instance. The parameters that define it, and its states, histories and outputs, belong to that application. Several BCs rarely share a parameterization, so sharing must be explicit and never follow from equal values or model-local slots. Give a bound state or output a reference that identifies both the instance and its local symbol. Resolve that reference to storage when the system layout is known. Inlining an output should preserve those bindings. Keeping an instance-qualified output reference is a valid fallback. Two bindings create independent state histories even with identical input values. Reusing one handle in several terms explicitly shares the state. Extend the existing instance handle and auxiliary equation builder for this. [Instance handle](../Auxiliary/AuxiliaryBindings.h#L244) · [Auxiliary equation syntax](../Auxiliary/AuxiliaryModelDSL.h#L21)

2. **Give measurements and residual terms distinct roles.** In the sketch, $u\cdot n$ is a pointwise expression, $Q$ is one scalar integral for the whole patch, and the final integral contributes to the equation tested by $v$. A quantity consumed by an implicit law should carry its dependence on the current iterate so the Jacobian includes the derivative of the integral. Make a lagged input an explicit choice. Let recording refer to the same bound quantity or output at the requested time level. [Existing quantity abstraction](../Systems/FunctionalForms.h#L94)

3. **Let authors state a trace relation and select its enforcement.** A prescribed value could read `on(G).impose(trace(w) == g, strong())` for a scalar or a vector field. For Nitsche enforcement, let the module declare the operator's conormal flux, its adjoint contribution, and the penalty scale once. Forms can assign terms to equation rows from their test fields. This would let the velocity factory express the prescribed value directly while retaining the pressure-test contribution of the mixed operator. [Generic flux and enforcement](StandardBCs.h#L321) · [Current row adapter](../../Physics/Formulations/NavierStokes/NavierStokesBCFactories.h#L604) · [Field inference](../Systems/FormulationBuilder.h#L95)

4. **Take the normal, traces, and measure from the same domain.** Let an author select the current or reference configuration on a fitted surface and name the side supplying the normal on an interface. With an oriented domain, paired traces, and weights, an interface term can read `-inner(avg(traction, weights), jump(V))` integrated over that interface. The module defines the traction and weights. The installer can expand the test-field contributions. The two-fluid factory currently spells out eight integrals for its adjoint term alone. [Oriented interface normal](IntegrationDomain.h#L305) · [Paired traces](Vocabulary.h#L664) · [Expanded adjoint term](../../Physics/Formulations/NavierStokes/NavierStokesBCFactories.h#L1670)

5. **Make each installed term inspectable.** For the example above, asking for the right boundary's load should show $P_B=X_B+25Q_B$, identify the domain of $Q_B$, and name the test field receiving the load. The builder already supports `tag` and `tagged` for selecting work terms, and `installedRow` shows an operator row's terms and field dependencies. Extend that report to trace a selected boundary load back to its bound state and measured quantity. [Tagged contributions](../Systems/FormulationBuilder.h#L222) · [Installed row report](../Systems/SystemManifest.h#L126) · [Free-surface contributions](../../Physics/Formulations/NavierStokes/NavierStokesBCFactories.h#L1279)

## Regression coverage and follow-up

The wave-3 branch covers the first three checks through the existing factory API. Its Forms installer tests compare residuals, Jacobians, direct state columns, and per-instance outputs. Physics tests exercise a mixed RCR/RCRCR pair in Navier–Stokes and two Windkessel conditions in Poisson. The last check applies to the wave-4 authoring interface.

- **Reorder two instances.** Use distinct parameters and initial states on disjoint markers. Compare residuals, Jacobian blocks, and histories by field and instance after mapping any internal row or slot permutation.
- **Perturb one state at fixed FE fields.** For the generic output $y_i=z_i+r_iQ_i$, the direct derivative is $\partial y_i/\partial z_j=\delta_{ij}$. Check the assembled boundary loads and their Jacobian columns against a finite difference. This isolates binding errors from physical coupling through the solved fields.
- **Exercise both output paths.** Include a differential model, an explicitly eliminable algebraic model, and two different models whose first local state has slot zero. The original two-outlet factory test used $C=0$ on both outlets and exercised the algebraic path. [Original two-outlet test](../../Physics/Tests/Unit/test_NavierStokesOutletFactory.cpp#L3305)
- **Check the public notation against an explicit weak form.** Compare a prescribed trace, a mixed Nitsche row, and an oriented interface term with their hand-written integrals. Check the domain, signs, and test fields as well as the numerical residual.

Restart across reordered declarations is a separate case. The wave-3 test `AuxiliaryCheckpointRestoresBlocksByPosition` at `b1d3681eb` shows that restoring an in-memory checkpoint into a reordered system exchanges equal-sized state blocks. A restart intended to support such reordering needs to match blocks by instance identity.
