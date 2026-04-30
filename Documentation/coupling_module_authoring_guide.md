# Writing A Coupling Module

This guide describes the preferred path for authoring PDE coupling modules with
the definition-backed FE/Coupling facade. Physics modules should describe the
participants, fields, relation, and coupling equations. The FE layers remain
responsible for context lookup, graph validation, Forms installation,
dependency metadata, partitioned exchange planning, and diagnostics.

## Scope

Use coupling modules for PDE-to-PDE relations. A single-PDE ODE or DAE boundary
model should stay in AuxiliaryState. If an ODE, DAE, algebraic model, or global
scalar depends on state from multiple PDE participants, then the coupling
module should declare the cross-PDE dependency while AuxiliaryState remains the
owner of the model equations and storage.

## Preferred Shape

New physics coupling modules should derive from
`FE::coupling::DefinitionBackedCouplingContract` and implement:

```cpp
std::string name() const override;

std::string contractInstanceName() const override;

void define(FE::coupling::CouplingDefinitionBuilder& c) const override;

void validateDefinitionOptions(
    const FE::coupling::CouplingContext& ctx,
    FE::coupling::CouplingValidationResult& result) const override;
```

`define(...)` should contain the normal physics authoring path:

1. Declare participant roles.
2. Declare field roles and expected value shapes.
3. Declare shared interfaces or region relations.
4. Declare relation lowering capabilities.
5. Add monolithic Forms residuals when the relation contributes to one solve.
6. Add partitioned exchange channels when a partitioned solve is selected.

`validateDefinitionOptions(...)` is only for option checks that cannot be
expressed as FE/Coupling declarations, such as mutually exclusive physics
options. Generic checks for fields, participants, regions, topology, same-system
compatibility, relation capabilities, and partitioned endpoints belong in
FE/Coupling.

Direct `CouplingContract` overrides remain appropriate for expert contracts
that cannot yet be represented through public Forms, Systems, Analysis, or
Coupling records. Expert paths must still provide equivalent diagnostics and
metadata before they are considered complete.

## Naming

Use stable generated names for coupling-owned residuals, exchanges, variables,
and diagnostics:

```text
<contract_name>.<relation_name>.<local_name>
```

Use `FE::coupling::makeCouplingGeneratedName(...)` when the definition facade
does not generate the name directly. Explicit overrides are reserved for
advanced cases that must preserve an external interface.

## Monolithic Forms

When coupling math contributes to a monolithic residual, author it as
FE/Forms expressions. Forms metadata is the evidence used for residual rows,
dependencies, expected blocks, temporal symbols, and geometry terminals.
Physics modules should not duplicate that metadata by hand unless they are in
an expert verification path.

Interface-side helpers map physics-side notation to existing Forms machinery:

- `state(...)`, `test(...)`, and `dt(...)` lower through `CouplingFormBuilder`.
- Side restrictions lower through the existing minus and plus trace operators.
- `normal(...)` lowers through geometry terminals.
- Interface integrals lower through shared-region integration.

Geometry-sensitive terms should use the builder's geometry terminals so
`FormAnalysisBridge` and `CouplingGraph` can report provenance and sensitivity.

## Partitioned Exchanges

Partitioned coupling should be authored as logical exchange channels. The
physics module chooses the high-level strategy and transfer declarations; the
partitioned backend validates endpoint regions, temporal slots, transfer
compatibility, runtime handles, group hints, and plan coverage.

Iteration algorithms, relaxation, subcycling, convergence checks, and
driver-owned scheduling are partitioned strategy metadata. They should not be
encoded in Forms residual expressions.

## Relation Capabilities

Each relation should declare the lowerings it supports:

- `MonolithicForms` for Forms-authored residuals.
- `PartitionedExchange` for partitioned exchange plans.
- Expert lowerings only when explicitly opted in.

Each capability should also declare its fidelity:

- `Exact` means the lowering represents the stated relation exactly for the
  selected strategy.
- `Approximate` means the lowering is a controlled approximation.
- `Lagged` means the lowering uses lagged data or exchange state.
- `Unavailable` means the lowering is intentionally unsupported and must carry
  a reason.

Unsupported strategies should fail before Forms installation or plan execution.
Approximate and lagged lowerings should appear in diagnostics.

## Configuration Surface

Keep default physics options short and domain-focused:

- relation law;
- enforcement strategy;
- solve strategy;
- named transfers for partitioned mode.

Advanced options may include relaxation, convergence, time-window, frame
transform, projection, and geometry sensitivity controls. Expert options may
include custom fallback hooks, driver-owned payloads, custom transfer maps, and
metadata overrides. Expert options must document their validation and diagnostic
obligations.

## General Patterns

The same authoring path should support two-sided interfaces, N-participant
junctions, hub-and-spoke relations, optional participants, repeated participant
groups, self-coupling across regions, mixed-dimensional relations, multipliers,
global variables, and multi-PDE AuxiliaryState dependencies.

Use FE/Forms when the coupling is a weak residual, constraint residual,
penalty residual, Nitsche residual, multiplier residual, or algebraic residual
that belongs in a monolithic solve. Use FE/Coupling exchange declarations when
the coupling is data movement, lagged input/output, driver-owned transfer, or a
partitioned exchange. Use FE/Analysis variable metadata for auxiliary states,
global scalars, boundary functionals, material state, and other non-field
variables whose dependencies span PDE participants.

Transform, orientation, frame, unit, and geometry lifecycle policies should be
declared as metadata or expressed directly in Forms terms. Flux, traction,
normal, and tangential laws should make sign and frame conventions explicit
enough for diagnostics.

## FSI Sketch

```cpp
void FSICouplingModule::define(fec::CouplingDefinitionBuilder& c) const
{
    c.participant(options_.fluid_name);
    c.participant(options_.solid_name);
    c.fieldRequirement(vectorField(options_.fluid_name,
                                   options_.fluid_velocity_field));
    c.fieldRequirement(scalarField(options_.fluid_name,
                                   options_.fluid_pressure_field));
    c.fieldRequirement(vectorField(options_.solid_name,
                                   options_.solid_displacement_field));
    c.sharedInterface(interfaceRequirement(options_));
    c.regionRelation(fsiRelation(options_));

    c.monolithic([options = options_](const fec::CouplingContext& ctx,
                                      const fec::CouplingFormBuilder& forms) {
        return buildFSIForms(options, ctx, forms);
    });

    appendFSIExchanges(options_, c);
}
```

The monolithic forms should show velocity continuity and traction balance in
one reviewable location. Partitioned mode should expose only the physical
exchange channels and transfer choices.

## Thermal Interface Sketch

```cpp
void ThermalInterfaceCouplingModule::define(
    fec::CouplingDefinitionBuilder& c) const
{
    c.participant(options_.side_a_name);
    c.participant(options_.side_b_name);
    c.fieldRequirement(scalarField(options_.side_a_name,
                                   options_.side_a_temperature_field));
    c.fieldRequirement(scalarField(options_.side_b_name,
                                   options_.side_b_temperature_field));
    c.sharedInterface(interfaceRequirement(options_));
    c.regionRelation(thermalRelation(options_));

    c.monolithic([options = options_](const fec::CouplingContext& ctx,
                                      const fec::CouplingFormBuilder& forms) {
        return buildThermalForms(options, ctx, forms);
    });

    appendThermalExchanges(options_, c);
}
```

The thermal module should keep temperature-continuity math in Forms for
monolithic mode and expose temperature or heat-flux channels for partitioned
mode.
