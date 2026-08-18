# AuxiliaryState Math-First Authoring Plan

**Date**: 2026-03-25

## Goal

Make the `AuxiliaryState` authoring experience substantially more intuitive, math-first, and concise without changing the core runtime model unnecessarily.

The target outcome is that future users can define local ODE/DAE-style auxiliary models and FE-coupled auxiliary inputs in a way that reads like the mathematical model they are implementing, rather than a mix of:

- string-based symbol lookup
- deployment plumbing
- registry naming
- low-level callback registration

## Checklist Convention

- [ ] not started
- [~] partially implemented / in progress
- [x] complete

## FE-Library Design Principle

Any `FE/` and `AuxiliaryState` infrastructure change in this plan must remain physics-agnostic.

That means:

- no new API in `FE/Systems`, `FE/Forms`, or `FE/Docs` should mention Navier-Stokes, RCR, Windkessel, flow rate, outlet pressure, or any other formulation-specific concept
- the core abstractions should speak in terms of auxiliary models, equations, states, inputs, parameters, outputs, boundary integrals, derived inputs, deployment handles, and symbolic coupling
- physics modules are downstream clients and validation cases, not the shape-defining center of the FE-library API

Checklist:

- [x] Keep all new FE-library surface area physics-agnostic.
- [x] Keep examples in FE-library headers/docs generic unless a physics-specific example is clearly labeled as a client example.
- [x] Ensure any Navier-Stokes migration is expressed as a client of the generic API rather than a special case embedded in `FE/`.

## Problem Summary

Today, writing an auxiliary model often requires mixing three different layers of abstraction:

1. form expressions
2. auxiliary-model declaration syntax
3. deployment and registry plumbing

For example, the current RCR outlet path requires:

- repeated `modelInput("Q")`, `modelState("X")`, and `modelParam("...")` wrappers
- explicit string names for instance outputs
- explicit registration of boundary-integral inputs
- explicit deployment naming and input binding
- separate callback-based machinery for simple derived quantities such as a resistive `P_out`

The result is correct, but it is longer and less readable than the mathematical model.

## Desired End State

The work is complete when all of the following are true:

- auxiliary models can be defined using equation-like syntax
- model ports and states can be referenced as typed symbols instead of repeated string wrappers
- simple derived auxiliary quantities can be declared from expressions rather than low-level registry callbacks
- FE-coupled auxiliary inputs can be registered in a way that returns typed handles, not just side effects
- deployed auxiliary instances return handles that expose outputs without string concatenation
- common deployment cases require much less policy boilerplate
- existing runtime/storage/derivative infrastructure remains largely unchanged under the hood

## Non-Goals

- this plan does not redesign the core `AuxiliaryStateModel` residual interface
- this plan does not replace the existing builder/deployment API immediately; it layers a better front end on top
- this plan does not require removing the existing low-level APIs
- this plan does not add physics-specific helpers to the FE library

## Current Friction Points

### 1. Symbol verbosity

The builder still requires low-level symbolic helpers such as:

- `modelInput("Q")`
- `modelState("X")`
- `modelParam("Rp")`

This makes equations longer than the math.

### 2. Equation structure is not explicit

Users write:

- `.ode("X", rhs)`
- `.output("P_out", expr)`

instead of writing something that visibly reads like:

- `dX/dt = ...`
- `P_out = ...`

### 3. Deployment is string-heavy

Users must carry:

- input registry names
- deployed instance names
- string-based output references

instead of working with typed handles.

### 4. Derived formulas are too low-level

Simple algebraic derived inputs still require manual `AuxiliaryInputSpec` plus callback registration in some paths.

### 5. FE-bound auxiliary registration is side-effect-only

`registerBoundaryIntegralInput(...)` registers an input but does not return a handle that can be used directly in deployment or expressions.

## Target Authoring Style

The target style should look closer to this:

```cpp
auto Q = system.boundaryIntegral(
    "Q",
    inner(discrete(u_id, velocity_space, "u"), normal()),
    marker);

FormExpr p_out;

if (bc.C == 0.0) {
    p_out = system.derivedInput("P_out", bc.Pd + (bc.Rp + bc.Rd) * Q);
} else {
    auto rcr_model = aux::model("rcr", [&](auto m) {
        auto Q = m.input("Q");
        auto X = m.state("X");
        auto [Rp, C, Rd, Pd] = m.params("Rp", "C", "Rd", "Pd");

        m << ddt(X) == (Q - (X - Pd) / Rd) / C;
        m << out("P_out") == X + Rp * Q;
    });

    auto rcr = system.deploy(
        use(rcr_model)
            .name("ns_rcr_" + std::to_string(marker))
            .global()
            .partitioned("BackwardEuler")
            .params({{"Rp", bc.Rp}, {"C", bc.C}, {"Rd", bc.Rd}, {"Pd", bc.Pd}})
            .bind(Q)
            .initialState({{"X", bc.X0}})
    );

    p_out = rcr.output("P_out");
}
```

This is still generic FE-library infrastructure. Navier-Stokes is only one client.

## Implementation Design Notes for User-Friendly Large Systems

The new authoring surface should not only be shorter; it should also be predictable and robust when users build large auxiliary systems with many:

- FE-driven inputs
- internal ODE/DAE states
- algebraic constraints
- derived outputs
- deployment-time parameters

These notes are design constraints for implementation, not just style preferences.

### 1. Equation insertion order should not affect semantics

For named-state models, users should be free to write equations in the order that is easiest to read.

That means:

- the canonical internal row/state order should come from state declaration order
- `m << ddt(X) == ...` should attach an equation to state `X`, not define its row position by insertion time
- reordering the equation statements for already-declared states must not change the meaning of the model

Example:

```cpp
auto ATP = m.state("ATP");
auto ADP = m.state("ADP");
auto NADH = m.state("NADH");

m << alg(ADP) == ATP + ADP - ATP_tot;
m << ddt(NADH) == v_prod - v_consume;
m << ddt(ATP) == v_supply - v_demand;
```

should still lower in the canonical state order:

- `ATP`
- `ADP`
- `NADH`

Checklist:

- [x] Make state declaration order, not equation insertion order, define canonical row order.
- [x] Lower named-state equations by state identity, not by statement position.
- [x] Guarantee that reordering `m << ...` statements for declared states does not change model semantics.
- [x] Treat duplicate equations for the same state as hard validation errors.

### 2. Named declarations should be the primary semantic surface

Large systems become hard to maintain when users must reason positionally.

The default user path should therefore be entirely name-based:

- states identified by name
- inputs identified by name
- outputs identified by name
- parameters identified by name
- initial conditions assigned by state name
- bindings assigned by input handle or name

Positional vectors should remain available only as advanced/compatibility paths.

Checklist:

- [x] Make named states, inputs, outputs, and parameters the primary user-facing model surface.
- [x] Keep positional initialization and raw vectors as advanced escape hatches, not the recommended path.
- [x] Prefer named deployment and handle-based coupling everywhere new API is introduced.

### 3. Builder validation should be strict and early

Large systems are hard to debug when mistakes surface late in stepping or assembly.

The builder should reject:

- missing equations for declared states
- duplicate governing equations for the same state
- equations targeting undeclared states
- outputs referencing unknown symbols
- named initial conditions or params that do not exist
- duplicate names across incompatible categories where ambiguity would be harmful

Checklist:

- [x] Validate that every declared solve-target state has exactly one governing equation.
- [x] Reject equations for undeclared states.
- [x] Reject duplicate equations for the same state unless explicitly allowed by a raw advanced mode.
- [x] Reject unknown parameter and initial-state names at deployment time.
- [x] Produce diagnostics that name the offending state/input/output/parameter directly.

### 4. Large systems need grouping helpers

For 10-50 state models, repeated one-by-one declarations become noisy even with better symbols.

The API should support grouped declarations where that improves readability:

- `m.inputs("O2", "Glucose", "Lactate", "pH")`
- `m.states("ATP", "ADP", "NADH", "PYR")`
- `m.params("k1", "k2", "k3")`

Tuple-return helpers should work well for a small number of symbols, but vector/list helpers should also exist for larger declarations.

Checklist:

- [x] Add grouped declaration helpers for repeated scalar ports/states/params.
- [x] Support tuple-style unpacking for small fixed groups.
- [x] Support vector/list-style returned symbol collections for larger groups.
- [x] Keep grouped declaration syntax generic and independent of any particular physics.

### 5. Handles should remove string plumbing, not add more wrappers

Once the new API introduces typed handles, they should materially reduce cognitive overhead.

Handles should be:

- cheap
- name-preserving
- printable/debuggable
- convertible to the right `FormExpr` surface when appropriate

Users should not need to manually concatenate strings such as `"instance/output"`.

Checklist:

- [x] Return typed handles from FE-coupled input registration.
- [x] Return typed handles from model deployment.
- [x] Make handle-based `.bind(...)` and `.output(...)` the preferred path.
- [x] Eliminate string concatenation patterns from examples and migrated clients.

### 6. Derived quantities should feel like formulas, not registry programming

Simple derived expressions should remain formula declarations, even when they are implemented internally through callbacks or dependency graphs.

This is especially important for large models with many derived outputs or helper quantities.

Checklist:

- [x] Add first-class `derivedInput(...)` or equivalent formula declaration API.
- [x] Auto-discover dependencies for derived inputs instead of requiring manual registry dependency wiring in user code.
- [x] Keep low-level callback registration as an advanced escape hatch, not the preferred surface.

### 7. Support many inputs and outputs as a normal use case

The new API should be designed with large metabolic / electrophysiology / signaling models in mind, not only toy one-state examples.

That means:

- multi-input models must be readable
- multi-output models must be easy to consume
- repeated boilerplate must stay low
- deployment should not become string-heavy again as models scale up

Checklist:

- [x] Ensure the DSL reads cleanly with 10+ inputs and 10+ outputs.
- [x] Ensure deployment remains concise for models with many parameters and initial values.
- [x] Add at least one documentation example with many inputs, states, and outputs.

### 8. Distinguish normal named models from advanced raw residual models

There is still value in a raw residual escape hatch for advanced users, but it should not shape the default semantics.

The recommended path should be:

- named states
- named equations
- named outputs

Advanced users may still need:

- raw residual rows
- manual row ordering
- direct residual vectors

Those should be explicitly documented as advanced.

Checklist:

- [x] Keep raw positional residual-row support as an explicit advanced mode.
- [x] Ensure the default DSL semantics are named and order-independent.
- [x] Document clearly when users leave the safe/high-level path and enter a positional/manual path.

### 9. Diagnostics should scale with model size

For large systems, a vague error such as "invalid auxiliary model" is not useful.

Diagnostics should mention:

- the model name
- the offending symbol name
- the category of the symbol (state/input/output/parameter)
- the exact issue (missing equation, duplicate equation, unknown param, etc.)

Checklist:

- [x] Include model name and symbol name in all high-level DSL validation errors.
- [x] Make duplicate/missing-equation errors specific and actionable.
- [x] Keep diagnostics readable for large systems with many declared symbols.

### 10. Documentation should teach the scalable pattern, not only toy examples

If the docs only show one-state decay examples, users will still struggle when writing real 20-variable systems.

The docs should include:

- one small toy example
- one medium multi-state ODE/DAE example
- one FE-coupled example with many named inputs and outputs

Checklist:

- [x] Keep one small introductory toy example.
- [x] Add a medium-sized multi-state example to demonstrate scalable authoring patterns.
- [x] Add a coupled FE-input example that uses many named inputs/outputs.

## Additional Design Ideas for Intuitive and Robust Authoring

These ideas are not all required for the first math-first slice, but they are worth treating as first-class design options because they strongly improve usability for larger auxiliary systems.

### A. Named intermediate expressions

Large models are usually written in terms of named rates, fluxes, and helper expressions.

Examples:

```cpp
auto v_gly = m.let("v_gly", k_gly * Glc / (Km_Glc + Glc));
auto v_ox  = m.let("v_ox",  k_ox  * O2  / (Km_O2  + O2) * NADH);
```

This improves readability and avoids repeating long expressions in multiple equations and outputs.

Checklist:

- [x] Add `let(name, expr)` or `rate(name, expr)` for named intermediate expressions.
- [x] Allow named intermediates to be reused in equations and outputs.
- [x] Validate duplicate intermediate names clearly.
- [x] Decide whether intermediate expressions should be surfaced in introspection/debug output.

### B. Exposable intermediates

Users often want the same internal rate both:

- inside the equations, and
- as a reported output or coupling quantity

Examples:

```cpp
m.expose(v_gly, "glycolytic_flux");
```

or:

```cpp
m << out("glycolytic_flux") == v_gly;
```

Checklist:

- [x] Make named intermediates easy to expose as outputs.
- [x] Avoid forcing users to rewrite the same formula twice when an intermediate is also an output.

### C. Grouped declarations for many ports/states/params

Very large systems become noisy if every symbol is declared one-by-one.

Examples:

```cpp
auto s = m.states("ATP", "ADP", "NADH", "PYR", "LAC_i");
auto p = m.params("k1", "k2", "Km_O2", "Km_Glc");
auto i = m.inputs("O2", "Glucose", "Lactate", "pH");
```

Checklist:

- [x] Add grouped declaration helpers for states, params, inputs, and outputs where useful.
- [x] Support tuple-style unpacking for small fixed groups.
- [x] Support vector/list-style collections for larger groups.
- [x] Keep grouped declaration behavior deterministic and easy to introspect.

### D. Symbol grouping / namespacing

Large models often need logical grouping to avoid flat-name sprawl.

Examples:

```cpp
auto mito = m.group("mito");
auto ATPm = mito.state("ATP");
auto NADHm = mito.state("NADH");
```

This remains physics-agnostic; it is simply a namespace/grouping feature for model organization.

Checklist:

- [x] Evaluate whether symbol groups / namespaces should be part of the builder DSL.
- [x] If added, ensure group prefixes lower deterministically to stable state/input/output names.
- [x] Keep grouping optional so simple models stay simple.

### E. Optional/default inputs and parameters

Reusable models benefit from optional ports and default values.

Examples:

```cpp
auto O2 = m.optionalInput("O2", 0.0);
auto Km = m.param("Km_O2", 0.2);  // optional with default
```

Checklist:

- [x] Add optional/default metadata for inputs where it makes semantic sense.
- [x] Add default values for parameters.
- [x] Ensure deployment-time validation distinguishes required vs optional ports clearly.
- [x] Keep defaults explicit in docs and model summaries.

### F. Bounds and scaling metadata

Large biochemical/electrophysiology systems often benefit from attached state/parameter metadata even if it is not solver-enforced initially.

Unit information is intentionally out of scope for the AuxiliaryState DSL.
No `.units()` method should be added to `ModelFacade`, `AuxiliaryModelBuilder`,
or related AuxiliaryState authoring APIs. That metadata should live in the
solver/formulation layer rather than inside the auxiliary-model authoring
surface.

Examples:

```cpp
auto ATP = m.state("ATP");
m.nonnegative("ATP");
m.scale("ATP", 1.0);

auto g = m.state("g");
m.bounded("g", 0.0, 1.0);
```

These improve:

- readability
- diagnostics
- solver scaling
- future validation opportunities

Checklist:

- [x] Add optional bound metadata (nonnegative / bounded / positive).
- [x] Add optional solver scaling metadata per state.
- [x] Keep unit metadata out of the AuxiliaryState DSL and document that it belongs in solver-side metadata.
- [x] Decide which non-unit metadata is descriptive only vs solver-active.
- [x] Surface metadata in diagnostics and model summaries.

### G. Separate initial values from algebraic initial guesses

Algebraic variables often need an initial guess rather than a committed initial value.

Examples:

```cpp
auto z = m.state("z", Algebraic);
m.initialGuess("z", 0.1);
// When initialState() is called without "z", the guess 0.1 is used.
```

Checklist:

- [x] Add explicit algebraic initial-guess metadata distinct from differential initial state.
- [x] Ensure deployment/initialization paths treat these semantics correctly. *(Initial guesses auto-fill omitted algebraic states inside `initialState({...})`. If `initialState()` is never called and `initialize(vector)` or no initialization is used, guesses are not applied — the block defaults to zero.)*
- [x] Document the distinction clearly in the DSL and README.

### H. Unused-symbol and dead-expression validation

Large models become hard to trust when declared symbols silently go unused.

Checklist:

- [x] Warn or error on declared inputs that are never used.
- [x] Warn or error on declared parameters that are never used.
- [x] Warn or error on declared intermediates that are never used or exposed.
- [x] Keep validation configurable enough to avoid blocking intentional staged work.

### I. Conservation / invariant helpers

Some large systems naturally contain conservation constraints or invariant relations.

Examples:

```cpp
auto ADP = m.state("ADP", Algebraic);
m.conservation(ADP, ATP + ADP - A_tot);  // lowers to alg(ADP) == ...
```

This should remain generic and not embed any particular physics.

Checklist:

- [x] Evaluate whether invariant/conservation helpers belong in the DSL.
- [x] If added, lower them to algebraic equations or validation constraints in a transparent way.
- [x] Keep the semantics explicit rather than “magic.”

### J. Composition / submodel inclusion

Very large systems are often easiest to build by composing reusable submodels.

Examples:

```cpp
// Full include() with expression slot remapping is future work.
// Current workaround: use group() for manual namespaced composition.
auto gly = m.group("gly");
// ... declare glycolysis states/params/equations via gly ...
auto mito = m.group("mito");
// ... declare mitochondria states/params/equations via mito ...
```

Checklist:

- [x] Design a composition model for reusable submodels. *(`include(submodel, prefix)` implemented with full slot remapping for AuxiliaryStateRef, AuxiliaryInputRef, and ParameterRef. All declarations prefixed. `group(prefix)` also available for manual composition. Tested by `IncludeSubmodel` and `IncludeWithCrossModelCoupling`.)*
- [x] Decide how names are prefixed/scoped to avoid collisions.
- [x] Ensure included submodels preserve deterministic ordering and diagnostics.
- [x] Keep composition generic and independent of any specific client physics.

### K. Deterministic introspection and pretty-printing

Users need to be able to inspect what the builder thinks they wrote.

Useful capabilities:

- list states/inputs/outputs/params
- dump canonical state order
- dump lowered equations
- print metadata such as bounds, defaults, and scales

Checklist:

- [x] Add deterministic introspection APIs for declared symbols and canonical ordering.
- [x] Add a pretty-printer / model summary facility.
- [x] Use the same summary output in tests, diagnostics, and docs where helpful.

### L. FE-coupled handle ergonomics

Once FE-coupled handles exist, they should feel natural in large coupled models.

Examples:

```cpp
auto O2   = system.sampledField("O2", "oxygen");
auto Glc  = system.sampledField("Glucose", "glucose");
auto Q    = system.boundaryIntegral("Q", expr, marker);
```

Checklist:

- [x] Make FE-coupled input helpers return typed handles consistently.
- [x] Allow handle-based binding in deployment.
- [x] Support auto-bind by exact name match where safe.
- [x] Keep the handle APIs generic across sampled fields, boundary integrals, and derived inputs.

### M. Documentation must show scalable patterns

The docs should not only teach one-state decay models.

They should also show:

- a medium multi-state ODE/DAE example
- a many-input / many-output example
- a grouped / intermediate-heavy example
- an FE-coupled example that remains physics-agnostic at the FE-library level

Checklist:

- [x] Add at least one medium-sized example with many states and named intermediates.
- [x] Add one many-input / many-output example.
- [x] Show grouping and metadata patterns if those features are added. *(README documents bounds/scaling, symbol grouping, conservation, summary, and optional params.)*
- [x] Keep FE-library examples generic, with physics-specific examples clearly labeled as client examples.

## Additional Design Notes: Addressing FE-Coupled Limitations Directly

The math-first authoring improvements solve a large part of the usability problem, but there are also a few deeper FE-coupling limitations that should be addressed directly if the goal is for AuxiliaryState models to be broadly expressive and future-proof.

These are still physics-agnostic FE-library concerns.

### N. Multi-field FE expressions as one registered auxiliary input

Users should be able to register one meaningful FE-derived quantity even when it depends on multiple FE fields.

Examples:

```cpp
auto J = system.boundaryIntegral(
    "J",
    inner(discrete(u_id, U, "u"), normal()) * discrete(c_id, C, "c"),
    outlet_marker);
```

or more generally:

```cpp
auto source = system.feExpression(
    "source",
    alpha * discrete(cO2_id, V, "cO2") / (Km + discrete(cO2_id, V, "cO2"))
  - beta  * discrete(cLac_id, V, "cLac"));
```

The cleaner fallback of “register separate inputs and recombine in the auxiliary model” should remain valid, but it should not be the only expressive path.

Checklist:

- [x] Introduce a generic FE-backed input concept that can represent arbitrary FE expressions over one or more fields.
- [x] Add FE-expression registration APIs such as `feExpression(...)`, `boundaryIntegral(...)`, `domainIntegral(...)`, or equivalent generic handle-returning forms.
- [x] Replace any “pick one primary field” logic with an explicit dependency set over referenced FE fields.
- [x] Ensure multi-field FE-derived inputs can be evaluated explicitly in partitioned workflows.
- [x] Keep the public API generic rather than adding special-purpose multi-field helpers for one client formulation.

### O. Exact monolithic coupling for FE-backed auxiliary inputs

For partitioned workflows, FE-backed inputs may be sampled and frozen numerically.

For monolithic workflows, the infrastructure should be able to treat FE-backed auxiliary inputs symbolically so that:

```text
F(x, I(u), t) = 0
```

can contribute:

```text
dF/du = dF/dI * dI/du
```

Checklist:

- [x] Make `dF/dinputs` a first-class, well-tested derivative target in the auxiliary model path.
- [x] Add exact linearization support for FE-backed input handles, including sampled fields and FE reductions.
- [x] Apply the chain rule automatically in mixed field-auxiliary assembly for auxiliary inputs bound to FE quantities.
- [x] Ensure explicit and monolithic workflows share the same authoring surface, differing only in lowering/assembly strategy.
- [x] Keep the monolithic coupling machinery generic across sampled fields, domain reductions, and boundary reductions.

### P. Shape-aware FE quantity handles for vector/tensor inputs

If vector and tensor FE quantities are meant to be used naturally, the handle surface should preserve shape metadata instead of flattening everything immediately to anonymous scalars.

Examples:

```cpp
auto u = system.sampledField("u", "velocity");   // vector handle
auto s = system.sampledField("sigma", "stress"); // tensor handle

auto uz = comp(u, 2);
auto flux = dot(u, normal());
auto traction = s * normal();
```

Checklist:

- [x] Preserve shape metadata on FE-backed input handles.
- [x] Allow vector/tensor FE handles to bind directly to vector/tensor auxiliary inputs where supported.
- [x] Add explicit component and contraction helpers such as `comp`, `dot`, `trace`, and `norm` where they improve clarity.
- [x] Ensure shape-aware handles remain compatible with explicit and monolithic lowering.
- [x] Keep shape semantics generic and not tied to any one formulation.

### Q. Unified FE quantity handle abstraction

The cleanest long-term direction is to unify FE-coupled sampled inputs, reductions, and FE expressions under one generic handle abstraction.

That handle should be able to represent:

- sampled state fields
- boundary integrals
- domain integrals
- domain/boundary averages
- arbitrary FE expressions over one or more fields

and lower into:

- explicit numeric values for partitioned workflows
- symbolic coupled quantities for monolithic workflows

Checklist:

- [x] Evaluate a unified `FEQuantityHandle` / `AuxiliaryInputHandle` abstraction for FE-coupled quantities.
- [x] Ensure the handle carries enough metadata for explicit evaluation and monolithic coupling.
- [x] Use the same handle surface for binding into auxiliary models and for direct form consumption where appropriate.

## Rich Testing Infrastructure for Accuracy and Robustness

If AuxiliaryState is going to become easier to write, it must also become easier to trust.

That requires a richer testing strategy than only checking a few happy-path examples. The testing infrastructure should validate:

- authoring semantics
- lowering correctness
- runtime correctness
- derivative correctness
- coupling correctness
- diagnostics and validation behavior
- determinism for large systems

### R. DSL and lowering equivalence tests

The new high-level DSL must lower identically to the existing builder for equivalent models.

Checklist:

- [x] Add golden equivalence tests comparing legacy builder and DSL residual expressions.
- [x] Test ODE, algebraic, and mixed DAE models.
- [x] Test output equations and named intermediates.
- [x] Test named params and named initial-state lowering.
- [x] Test canonical state ordering is independent of equation insertion order.

### S. Validation and diagnostics tests

Large models need strong validation and clear diagnostics.

Checklist:

- [x] Test missing-equation detection.
- [x] Test duplicate-equation detection.
- [x] Test unknown state/param/input/output names.
- [x] Test duplicate intermediate/output/parameter names.
- [x] Test unused-symbol warnings/errors where implemented. *(`UnusedSymbolPolicyError` tests error mode throws. `UnusedSymbolPolicySilent` tests silent mode passes.)*
- [x] Verify diagnostics include model name and offending symbol name.

### T. FE-coupled input tests

FE-backed handles and derived inputs need dedicated coverage beyond scalar toy examples.

Checklist:

- [x] Test sampled scalar FE inputs.
- [x] Test sampled vector/tensor FE inputs where supported.
- [x] Test boundary integrals over scalar and vector FE expressions.
- [x] Test domain reductions and averages if exposed through the new handle surface.
- [x] Test multi-field FE-expression registration and evaluation once supported.
- [x] Test shape-aware helpers such as `comp`, `dot`, `trace`, and `norm` if added.

### U. Partitioned runtime tests

The partitioned path should remain the default robust path for many auxiliary models.

Checklist:

- [x] Test explicit FE-backed inputs driving ODE systems.
- [x] Test mixed ODE/algebraic systems in partitioned mode.
- [x] Test many-input / many-output models.
- [x] Test update schedules (`OncePerTimeStep`, `EachNonlinearIteration`, `Manual`).
- [x] Test grouped declarations and named intermediates in realistic medium-sized models.

### V. Monolithic coupling tests

If FE-backed handles are meant to support exact monolithic coupling, that path needs direct tests.

Checklist:

- [x] Test symbolic/monolithic lowering of FE-backed auxiliary inputs.
- [x] Test chain-rule assembly through `dF/dinputs * dI/du`.
- [x] Test mixed field-auxiliary Jacobian blocks for sampled fields and FE reductions.
- [x] Test monolithic correctness against finite-difference verification for small systems.

### W. Large-system regression tests

Large systems are where usability and robustness matter most.

Checklist:

- [x] Add at least one medium-sized multi-state ODE/DAE regression model.
- [x] Add at least one many-input / many-output regression model.
- [x] Add tests that intentionally reorder equation statements and confirm identical semantics.
- [x] Add tests for grouped declarations, named intermediates, and named initialization on larger models.
- [x] Add deterministic summary / introspection snapshot tests if pretty-printing is introduced.

### X. Property-style and parity tests

Where possible, the test infrastructure should check more than one hard-coded example.

Checklist:

- [x] Add parity tests between explicit evaluation and symbolic lowering where both should agree.
- [x] Add finite-difference checks for Jacobians on representative auxiliary models.
- [x] Add randomized small-model tests for insertion-order independence where practical.
- [x] Add tests ensuring repeated registration/deployment does not create hidden ordering-dependent behavior.

### Y. Physics-client integration tests

Even though the FE-library remains physics-agnostic, real clients should validate that the abstractions work in practice.

Checklist:

- [x] Keep at least one real client integration test (for example Navier-Stokes outlet coupling) as a downstream validation case.
- [x] Ensure client integration tests exercise the preferred high-level path, not only legacy compatibility paths.
- [x] Keep client-specific tests outside the FE-library test contract so the FE design remains generic.

### Z. Test infrastructure and helpers

The test suite should be easy to extend as more DSL features and FE-backed handles are added.

Checklist:

- [x] Add reusable test helpers for building auxiliary models through both the legacy builder and the new DSL. *(`aux_test::buildDecay()`, `buildRCR()`, `buildDAE()` build models via DSL. `evaluateResidual()` helper evaluates models with given state/params.)*
- [x] Add reusable FE mesh/field fixtures for scalar, vector, and multi-field cases. *(`SingleTetraOneBoundaryFaceMeshAccess` used across all FESystem tests. VectorSpace fixture in multi-field tests.)*
- [x] Add helper assertions for canonical state ordering, handle binding, and model summaries. *(`aux_test::expectStateOrder()` asserts canonical state names. `SummarySnapshotDeterministic` verifies summary determinism.)*
- [x] Keep testing utilities generic and reusable across FE-library and downstream client tests. *(All helpers in `aux_test` namespace, no physics-specific dependencies.)*

## Workstream 1: Math-First Auxiliary Model DSL

### 1.1 Add typed builder symbols

Introduce builder-scoped symbol handles for:

- inputs
- states
- parameters
- outputs (declaration target only)

These handles should be lightweight wrappers that are implicitly convertible to `FormExpr`.

Examples:

- `auto Q = m.input("Q");`
- `auto X = m.state("X");`
- `auto Rp = m.param("Rp");`
- `auto [Rp, C, Rd, Pd] = m.params("Rp", "C", "Rd", "Pd");`

Checklist:

- [x] Add typed symbol wrappers for input/state/param/output declarations.
- [x] Make symbol wrappers implicitly convertible to `FormExpr`.
- [x] Preserve name-based validation and duplicate detection in the builder.
- [x] Keep the existing `modelInput/modelState/modelParam` helpers as compatibility shims initially.

Files:

- `Code/Source/solver/FE/Auxiliary/AuxiliaryModelBuilder.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryModelBuilder.cpp` if needed
- optionally new: `Code/Source/solver/FE/Auxiliary/AuxiliaryModelDSL.h`

### 1.2 Add equation objects and equation insertion

Introduce equation helpers:

- `ddt(X)`
- `out("P_out")`
- possibly `alg(Z)` for algebraic rows

and make the builder accept:

- `m << ddt(X) == rhs;`
- `m << out("P_out") == expr;`
- `m << alg(Z) == expr;`

Checklist:

- [x] Add `ddt(...)` helper for differential rows.
- [x] Add `out(...)` helper for output equations.
- [x] Add `alg(...)` or equivalent algebraic-row helper.
- [x] Add equation object types instead of overloading `==` into a boolean context.
- [x] Add `AuxiliaryModelBuilder::operator<<` for equation insertion.
- [x] Lower all new equation forms into the existing row/output declarations.

Files:

- `Code/Source/solver/FE/Auxiliary/AuxiliaryModelBuilder.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryModelBuilder.cpp`
- optionally new: `Code/Source/solver/FE/Auxiliary/AuxiliaryModelDSL.h`

### 1.3 Add lambda-based model construction

Add a top-level front-end such as:

```cpp
auto model = aux::model("name", [&](auto m) { ... });
```

This should be a front-end convenience layer only. It should lower into the existing `AuxiliaryModelBuilder`.

Checklist:

- [x] Add a lambda-based `aux::model(name, lambda)` front end.
- [x] Ensure the lambda receives a builder facade exposing the new math-first DSL.
- [x] Keep the existing fluent builder intact for backward compatibility.
- [x] Document that the lambda DSL is the preferred authoring surface for new code.

Files:

- `Code/Source/solver/FE/Auxiliary/AuxiliaryModelBuilder.h`
- optionally new: `Code/Source/solver/FE/Auxiliary/AuxiliaryModelDSL.h`

### 1.4 Add regression tests for DSL lowering equivalence

The new DSL must lower to the same residual/output expressions as the existing builder.

Checklist:

- [x] Add tests that compare legacy builder vs DSL residual expression trees.
- [x] Add tests for `ddt(X) == rhs`.
- [x] Add tests for algebraic rows.
- [x] Add tests for output equations.
- [x] Add tests for multi-parameter tuple helpers.

Files:

- `Code/Source/solver/FE/Tests/Unit/Systems/test_AuxiliaryStateModel.cpp`
- or new targeted DSL test file if clearer

## Workstream 2: Named and Typed Deployment

### 2.1 Add bulk named parameter assignment

Reduce repetitive:

```cpp
.param("Rp", Rp).param("C", C)...
```

to:

```cpp
.params({{"Rp", Rp}, {"C", C}, {"Rd", Rd}, {"Pd", Pd}})
```

Checklist:

- [x] Add `.params(...)` bulk setter on `AuxiliaryDeployedInstance`.
- [x] Preserve `.param(name, value)` as the low-level API.
- [x] Validate duplicate names and unknown parameter names cleanly.

Files:

- `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.cpp` if present

### 2.2 Add named initial-state assignment

Reduce positional initialization:

```cpp
.initialize({bc.X0})
```

to:

```cpp
.initialState({{"X", bc.X0}})
```

Checklist:

- [x] Add `.initialState(...)` named setter on `AuxiliaryDeployedInstance`.
- [x] Map state names to model ordering for `BuiltAuxiliaryModel`.
- [x] Decide and document behavior for custom `AuxiliaryStateModel` implementations without state-name metadata.
- [x] Preserve `.initialize(vector)` as a compatibility path.

Files:

- `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.cpp`
- possibly `Code/Source/solver/FE/Auxiliary/AuxiliaryStateModel.h` if more state-name metadata is needed

### 2.3 Add deployment convenience sugar

Reduce verbose deployment policy code.

Examples:

- `.global()` instead of `.scope(AuxiliaryStateScope::Global)`
- `.partitioned("BackwardEuler")` instead of `.solveMode(...).stepper(...)`
- `.monolithic()`
- `.singleRate()`

Checklist:

- [x] Add `.global()`, `.node()`, `.cell()`, `.boundaryEntity()` scope sugar as appropriate.
- [x] Add `.partitioned(stepper)` convenience overload.
- [x] Add `.monolithic()` convenience method.
- [x] Add schedule sugar where it improves readability without ambiguity.
- [x] Keep enum-based setters underneath as the canonical low-level implementation.

Files:

- `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.cpp`

### 2.4 Return a deployment handle from `deployAuxiliaryModel`

The deployment API should return a lightweight instance handle that exposes deployed outputs by name.

Example:

```cpp
auto inst = system.deploy(...);
auto p_out = inst.output("P_out");
```

Checklist:

- [x] Introduce `AuxiliaryInstanceHandle` or equivalent.
- [x] Make `deployAuxiliaryModel(...)` return the handle.
- [x] Add `.output(name)` on the handle returning a `FormExpr`.
- [x] Optionally add `.input(name)` / `.state(name)` later for advanced use.
- [x] Preserve current behavior for callers that ignore the returned handle.

Files:

- `Code/Source/solver/FE/Systems/FESystem.h`
- `Code/Source/solver/FE/Systems/FESystem.cpp`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.h`

## Workstream 3: First-Class Auxiliary Input and Derived-Input Handles

### 3.1 Return handles from FE-coupled input registration

Change FE-coupled auxiliary-input registration from side-effect-only to handle-returning.

Examples:

```cpp
auto Q = system.boundaryIntegral("Q", expr, marker);
auto sampled_u = system.sampledField("u_sample", "u", n_entities);
```

Checklist:

- [x] Introduce `AuxiliaryInputHandle`.
- [x] Make `registerBoundaryIntegralInput(...)` return that handle.
- [x] Consider whether `registerSampledFieldInput(...)` and related helpers should also return handles.
- [x] Make handles convertible to `FormExpr` as `AuxiliaryInput(name)`.

Files:

- `Code/Source/solver/FE/Systems/FESystem.h`
- `Code/Source/solver/FE/Systems/FESystem.cpp`
- possibly `Code/Source/solver/FE/Forms/Vocabulary.h`

### 3.2 Add typed binding from handles

Once input handles exist, deployment should allow:

```cpp
.bind("Q", Q)
```

or, eventually:

```cpp
.bind(Q)
```

Checklist:

- [x] Add `.bind(model_input, AuxiliaryInputHandle)` overload.
- [x] Optionally add auto-bind-by-name when the input and port names match.
- [x] Keep string-based `.bind(...)` for compatibility.

Files:

- `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.cpp`

### 3.3 Add first-class derived-input expressions

Replace low-level callback registration for simple algebraic derived quantities with a generic FE-library API.

Examples:

```cpp
auto P_out = system.derivedInput("P_out", Pd + (Rp + Rd) * Q);
```

Checklist:

- [x] Add `derivedInput(name, expr)` or equivalent API to `FESystem`.
- [x] Lower it into the existing `AuxiliaryInputRegistry` using `FormulationCallback` internally.
- [x] Auto-discover and register dependencies between derived inputs and source inputs.
- [x] Return an `AuxiliaryInputHandle`.
- [x] Keep the API generic; do not special-case resistive outlets or any physics-specific formula.

Files:

- `Code/Source/solver/FE/Systems/FESystem.h`
- `Code/Source/solver/FE/Systems/FESystem.cpp`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryInputRegistry.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryInputRegistry.cpp`

### 3.4 Add tests for input handles and derived inputs

Checklist:

- [x] Test `registerBoundaryIntegralInput(...)` returns a usable handle.
- [x] Test handle-to-`FormExpr` conversion.
- [x] Test `derivedInput(...)` dependency ordering.
- [x] Test `derivedInput(...)` exact value evaluation.
- [x] Test binding a model input from an `AuxiliaryInputHandle`.

Files:

- `Code/Source/solver/FE/Tests/Unit/Systems/test_BoundaryIntegralInput.cpp`
- `Code/Source/solver/FE/Tests/Unit/Systems/test_AuxiliaryModelBuilder.cpp`
- `Code/Source/solver/FE/Tests/Unit/Systems/test_FormsInstaller.cpp`

## Workstream 4: Better Form Vocabulary for Auxiliary Coupling

### 4.1 Prefer handle-based output access over string concatenation

Current code often still effectively does:

```cpp
FormExpr::auxiliaryOutput(instance_name + "/P_out");
```

The preferred end state is:

```cpp
inst.output("P_out")
```

Checklist:

- [x] Add `AuxiliaryInstanceHandle::output(name)`.
- [x] Ensure the returned expression lowers to the same `AuxiliaryOutput(instance, name)` form.
- [x] Keep `AuxiliaryOutput(instance, name)` as the raw form-level function.

Files:

- `Code/Source/solver/FE/Forms/Vocabulary.h`
- `Code/Source/solver/FE/Systems/FESystem.h`
- `Code/Source/solver/FE/Systems/FESystem.cpp`

### 4.2 Add vocabulary aliases for builder-side readability

If the builder DSL is introduced, add clearly named aliases in a neutral namespace.

Examples:

- `aux::model(...)`
- `ddt(X)`
- `out("P_out")`

Checklist:

- [x] Choose a small, neutral namespace for the DSL surface.
- [x] Avoid polluting the global FE vocabulary with builder-only helpers unless necessary.
- [x] Keep the math-first DSL discoverable in docs and examples.

Files:

- `Code/Source/solver/FE/Auxiliary/AuxiliaryModelDSL.h`
- or `Code/Source/solver/FE/Auxiliary/AuxiliaryModelBuilder.h`

## Workstream 5: Documentation and Canonical Examples

### 5.1 Update builder docs and examples

The top-level examples in the builder should show the preferred math-first surface.

Checklist:

- [x] Update `AuxiliaryModelBuilder.h` examples.
- [x] Add an ODE example.
- [x] Add an algebraic/DAE example.
- [x] Add a boundary-integral-coupled example.

Files:

- `Code/Source/solver/FE/Auxiliary/AuxiliaryModelBuilder.h`

### 5.2 Update AuxiliaryState documentation

Document the new authoring stack clearly:

1. define FE-coupled inputs
2. define auxiliary model equations
3. deploy the model
4. consume outputs in forms

Checklist:

- [x] Add a dedicated “Math-First Authoring” section to the AuxiliaryState README.
- [x] Document typed handles for inputs and deployed instances.
- [x] Document the equation DSL.
- [x] Document named params and named initial state.
- [x] Document derived-input expressions.
- [x] Keep examples generic at the FE-library level.

Files:

- `Code/Source/solver/FE/Docs/AuxiliaryState/README.md`

### 5.3 Add one client migration example

Navier-Stokes should appear as a downstream client example, not the shape of the FE API.

Checklist:

- [x] Add one client-facing example showing how a boundary-integral-coupled auxiliary model becomes concise with the new API.
- [x] Clearly label it as a client example, not as the defining FE-library abstraction.

Files:

- `Code/Source/solver/Physics/Formulations/NavierStokes/NavierStokesBCFactories.h`
- optional separate design note in `Documentation/`

## Workstream 6: Client Migration and Validation

### 6.1 Migrate the RCR client code to the new surface

Once the infrastructure exists, rewrite the current RCR path to use:

- math-first builder DSL
- boundary-integral input handle
- deployment handle
- derived-input expression for the resistive branch

Checklist:

- [x] Rewrite the RCR model definition using the new DSL.
- [x] Replace string-based binding with handle-based binding where possible.
- [x] Replace raw `AuxiliaryInputSpec` callback setup in the resistive branch with `derivedInput(...)`.
- [x] Replace string-based output references with the deployment handle.

Files:

- `Code/Source/solver/Physics/Formulations/NavierStokes/NavierStokesBCFactories.h`

### 6.2 Preserve backward compatibility during rollout

Checklist:

- [x] Keep the existing low-level builder API initially.
- [x] Keep low-level deployment setters initially.
- [x] Keep string-based `AuxiliaryOutput(instance, name)` support.
- [x] Add deprecation notices only after the new path is tested and documented. *(Legacy `toCoupledOutflowBC` overload already marked `@deprecated` in `NavierStokesBCFactories.h`.)*

## Concrete Implementation Order

### Phase 1: Builder DSL foundation

Checklist:

- [x] Add typed builder symbols.
- [x] Add `ddt(...)`, `out(...)`, and equation objects.
- [x] Add `builder << equation`.
- [x] Add lambda-based `aux::model(...)`.
- [x] Add equivalence tests against the existing builder.

### Phase 2: Deployment ergonomics

Checklist:

- [x] Add named bulk `.params(...)`.
- [x] Add named `.initialState(...)`.
- [x] Add convenience deployment sugar (`.global()`, `.partitioned(...)`, etc.).
- [x] Add deployment return handle with `.output(name)`.

### Phase 3: Input and derived-input handles

Checklist:

- [x] Add `AuxiliaryInputHandle`.
- [x] Return handles from `registerBoundaryIntegralInput(...)`.
- [x] Add handle-based `.bind(...)`.
- [x] Add `derivedInput(name, expr)`.
- [x] Add dependency-order tests for derived inputs.

### Phase 4: Documentation

Checklist:

- [x] Update builder header examples.
- [x] Update AuxiliaryState README. *(Math-first authoring section, optional params, bounds/scaling, grouping, conservation, summary, unused-symbol policy, FE-backed quantities, monolithic coupling — all documented.)*
- [x] Add one client migration example.
- [x] Ensure all FE-library docs remain physics-agnostic.

### Phase 5: Client migration

Checklist:

- [x] Migrate the Navier-Stokes RCR path to the new authoring surface.
- [x] Keep behavior identical through tests.
- [x] Remove no longer needed local plumbing from the client code.

## Acceptance Criteria

The plan is complete when all of the following are true:

- [x] A user can define a local auxiliary model without repeated `modelInput/modelState/modelParam` wrappers.
- [x] A user can write model equations in a visibly equation-like syntax.
- [x] A user can register FE-coupled auxiliary inputs and receive typed handles back.
- [x] A user can define simple derived auxiliary inputs without low-level callback boilerplate.
- [x] A user can deploy a model and receive a handle for referencing outputs.
- [x] The FE-library API remains physics-agnostic.
- [x] The Navier-Stokes RCR client becomes materially shorter and closer to the math.
- [x] Existing runtime behavior remains covered by unit/integration tests.

## Recommended First Slice

If this work is staged, the highest-value first slice is:

1. typed builder symbols
2. equation DSL (`ddt`, `out`, `builder << ...`)
3. named params / named initial state
4. deployment handle `.output(name)`

That slice gives the biggest readability win before touching the richer input-handle and derived-input ergonomics.
