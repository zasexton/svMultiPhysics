# Reusable boundary laws in Forms

Draft design notes from reviewing `issue-449-modern-mesh-core` at `a41f0bec7`.

A physics developer should be able to write a boundary law once in the notation of its weak form. A problem author should be able to apply that law to several boundaries, each with its own data and initial conditions. The first change I would make is to preserve the identity of each application from the expression the developer writes through to the assembled residual and Jacobian.

Forms supplies fields, traces, integration domains, coefficients, quantities, and symbolic equations. The FE system binds those symbols to a problem and manages storage and assembly. Physics modules supply the constitutive equations, signs, parameter meanings, and admissible configurations. [Forms vocabulary](VOCABULARY.md#design-principles)

## What the outlet example reveals

The Navier–Stokes factory defines the resistance–capacitance–resistance (RCR) outlet model once. For outlet $i$, it measures $Q_i=\int_{\Gamma_i}u\cdot n_i\,ds$, deploys a model with its own resistances, capacitance, and initial capacitor pressure $X_i(0)$, and uses $P_i=X_i+R_{p,i}Q_i$ in the traction. [Model definition](../../Physics/Formulations/NavierStokes/NavierStokesBCFactories.h#L240) · [Per-outlet binding](../../Physics/Formulations/NavierStokes/NavierStokesBCFactories.h#L482)

That ownership is lost when Forms expands the output for assembly. It substitutes the bound input and parameters, but a differential state can retain its model-local slot. Each RCR definition uses slot zero for $X$. The assembled kernel reads a single array containing all instances’ states, so two tractions can both read the first capacitor pressure. The scalar histories still evaluate each model separately. [Output expansion](../Systems/FormsInstaller.cpp#L4893) · [State substitution](../Systems/FESystem.cpp#L25106) · [Assembly state layout](../Systems/FESystem.cpp#L24756)

| Branch check | Observation |
| --- | --- |
| Two compliant RCR outlets | Reversing only the BC declarations changed the mean fluid pressure at step 3 from **4.92649** to **1.98792**. The two recorded capacitor pressures were **4.91736** and **1.98641** in both orders. |
| Two RCRCR outlets, each with two capacitor states | Reversing the declarations changed the mean fluid pressure at step 1 from **4.98164** to **1.99631**. In a mixed RCR and RCRCR pair, changing the RCRCR's initial $P_1$ from 2 to 8 changed its reported outlet pressure by nearly 6. The fluid pressure changed by less than $1.2\times 10^{-12}$. |
| Prescribed pressure, strong velocity, and pure resistance ($C=0$) | Reversing two distinct entries reproduced the nodal fields in each case. |
| Weak velocity enforcement | The initial assembled load equaled the sum of the two individual loads in either order. The transient pipe solve did not converge in either order. |
| Two fitted free-surface patches | Reordering patches with exterior pressures 5 and 2 preserved the fields to roundoff. Exchanging the values moved the pressure response to the opposite patch. For unfitted surfaces, the Navier–Stokes module allows one surface to restrict pressure operations. [Active-domain rule](../../Physics/Options/NavierStokesOptionRules.cpp#L596) |

To repeat the first check, use the pipe case's no-slip wall and give each end face a Neumann RCR block. Set $(R_p,C,R_d,P_d,X_0)$ to $(12,0.01,90,0,5)$ on `lumen_inlet` and $(25,0.02,110,0,2)$ on `lumen_outlet`. Set `Write_scalar_history` to true and `Increment_in_saving_VTK_files` to 1. Run three steps of $0.005$ with generalized-alpha spectral radius $0.5$. Reverse the two `<Add_BC>` blocks, keeping each block's face name and values together, and repeat. Compare the saved fluid pressure and both scalar histories at $t=0.015$. [Starting pipe case](../../../../../tests/cases/fluid/pipe_RCR_3d/solver_new.xml)

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

3. **Let authors state a trace relation and select its enforcement.** A prescribed value could read `on(G).impose(trace(w) == g, strong())` for a scalar or a vector field. For Nitsche enforcement, let the module declare the operator's conormal flux, its adjoint contribution, and the penalty scale once. Forms can assign terms to equation rows from their test fields. This would let the velocity factory express the prescribed value directly while retaining the pressure-test contribution of the mixed operator. [Generic flux and enforcement](StandardBCs.h#L321) · [Current row adapter](../../Physics/Formulations/NavierStokes/NavierStokesBCFactories.h#L604) · [Field inference](../Systems/FormulationBuilder.h#L77)

4. **Take the normal, traces, and measure from the same domain.** Let an author select the current or reference configuration on a fitted surface and name the side supplying the normal on an interface. With an oriented domain, paired traces, and weights, an interface term can read `-inner(avg(traction, weights), jump(V))` integrated over that interface. The module defines the traction and weights. The installer can expand the test-field contributions. The two-fluid factory currently spells out eight integrals for its adjoint term alone. [Oriented interface normal](IntegrationDomain.h#L305) · [Paired traces](Vocabulary.h#L664) · [Expanded adjoint term](../../Physics/Formulations/NavierStokes/NavierStokesBCFactories.h#L1670)

5. **Make each installed term inspectable.** For the example above, asking for the right boundary's load should show $P_B=X_B+25Q_B$, identify the domain of $Q_B$, and name the test field receiving the load. Allow an author to name a weak contribution once and select it for a balance or diagnostic. This would also let free-surface pressure work and surface energy reuse the terms that actually entered assembly. [Current free-surface contributions](../../Physics/Formulations/NavierStokes/NavierStokesBCFactories.h#L1279)

## Suggested first change and regression tests

Start with instance-aware output expansion and an inspection or test hook for the bound expression. Both can be exercised through the current factory API. The shorter relation syntax and geometry conveniences can follow independently.

- **Reorder two instances.** Use distinct parameters and initial states on disjoint markers. Compare residuals, Jacobian blocks, and histories by field and instance after mapping any internal row or slot permutation.
- **Perturb one state at fixed FE fields.** For the generic output $y_i=z_i+r_iQ_i$, the direct derivative is $\partial y_i/\partial z_j=\delta_{ij}$. Check the assembled boundary loads and their Jacobian columns against a finite difference. This isolates binding errors from physical coupling through the solved fields.
- **Exercise both output paths.** Include a differential model, an explicitly eliminable algebraic model, and two different models whose first local state has slot zero. The existing two-outlet factory test uses $C=0$ on both outlets and exercises the algebraic path. [Two-outlet test](../../Physics/Tests/Unit/test_NavierStokesOutletFactory.cpp#L3305)
- **Check the public notation against an explicit weak form.** Compare a prescribed trace, a mixed Nitsche row, and an oriented interface term with their hand-written integrals. Check the domain, signs, and test fields as well as the numerical residual.
