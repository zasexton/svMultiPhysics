# Coupling Module Math-First Authoring Migration Plan

## Goal

Migrate physics-specific coupling modules toward a concise, equation-centered
authoring model while preserving the existing FE/Coupling, FE/Forms,
FE/Systems, and FE/Analysis infrastructure as the source of truth for
validation, lowering, metadata, and diagnostics.

The desired split is:

```text
FE/Coupling:
  Backend contract compilation, context lookup, graph validation, shared-region
  validation, monolithic form resolution, partitioned exchange planning,
  transfer validation, and diagnostics.

FE/Forms, FE/Systems, FE/Analysis:
  Weak-form vocabulary, residual installation, mixed-form lowering, installed
  dependency/block metadata, temporal metadata, and geometry terminal metadata.

Physics/Coupling:
  Physical roles, physical field names, high-level strategy choices, and the
  coupling equations.
```

The migration should make future physics coupling files read mostly as:

```text
1. Declare participant roles.
2. Declare field roles and expected value shapes.
3. Declare the shared interface.
4. Define the physical coupling relation once.
5. Provide monolithic Forms lowering when the relation contributes residuals.
6. Provide partitioned exchange lowering when a partitioned strategy is enabled.
```

Physics modules should not manually duplicate graph checks, topology checks,
field lookup checks, dependency declarations, expected-block declarations, or
partitioned endpoint plumbing except through explicit expert overrides.

## PDE Coupling Scope

This infrastructure is primarily for coupling PDE systems to other PDE systems.
Ordinary ODE or DAE models attached to a single PDE problem, such as common
0D hydraulic outlet circuit analogs used as fluid boundary conditions, should
remain in the AuxiliaryState infrastructure.

An ODE, DAE, algebraic model, or global scalar relation enters the coupling
framework only when it is part of a multi-PDE coupling relation, for example:

- [ ] its inputs depend on fields from multiple PDE participants;
- [ ] its outputs contribute residuals or exchanges for multiple PDE
      participants;
- [ ] it introduces a shared unknown or constraint that couples multiple PDE
      fields;
- [ ] it is needed to make a PDE-to-PDE partitioned exchange or monolithic
      residual well-defined.

This boundary keeps single-physics boundary models in AuxiliaryState and keeps
FE/Coupling focused on PDE-to-PDE contracts.

When an AuxiliaryState model is admitted into a coupling contract, it must use
the same selected coupling strategy as the PDE relation:

- [ ] in monolithic mode, its variables enter the same monolithic dependency
      graph and any PDE residual contributions are installed through the
      monolithic Forms path;
- [ ] in partitioned mode, its inputs and outputs are synchronized through the
      same partitioned exchange plan, temporal policy, relaxation policy, and
      driver schedule as the rest of the coupling contract;
- [ ] mixed strategies inside one coupling contract are invalid unless they are
      declared as an explicit nested expert strategy with complete diagnostics.

AuxiliaryState remains the owner of the ODE/DAE equations and storage. The
coupling layer only owns the cross-PDE dependency contract and the selected
monolithic or partitioned lowering.

## Existing Infrastructure To Reuse

The migration must extend these existing FE tools rather than replacing them:

- `FE/Coupling/CouplingDeclaration.h`
  - Keep `CouplingContractDeclaration`, `CouplingFormContribution`, and
    `CouplingExchangeDeclaration` as the backend declaration records.
- `FE/Coupling/CouplingGraph.h/.cpp`
  - Keep this as the validation, dependency, topology, and diagnostics
    authority.
- `FE/Coupling/CouplingFormBuilder.h/.cpp`
  - Extend this with interface-side ergonomics instead of adding a second Forms
    authoring API.
- `FE/Coupling/MonolithicCouplingBuilder.h/.cpp`
  - Keep this as the form resolver, installer, and metadata adapter.
- `FE/Coupling/PartitionedCouplingPlanGenerator.h/.cpp`
  - Keep this as the partitioned validation and plan-generation path.
- `FE/Coupling/SharedRegionRegistry.h/.cpp`
  - Reuse this for shared-interface participant lookup and topology validation.
- `FE/Systems/FormsInstaller.h`
  - Keep `installFormulationWithMetadata()` as the monolithic Forms install
    path.
- `FE/Analysis/FormAnalysisBridge.h`
  - Reuse bridge metadata for field terminals, non-field terminals, temporal
    terminals, geometry terminals, installed dependencies, and installed
    blocks.
- `FE/Forms/InterfaceConditions.h` and `FE/Forms/BoundaryConditions.h`
  - Reuse the existing trace, jump, average, Nitsche, and interface helper
    vocabulary when it matches the coupling math.

## Non-Goals

- [ ] Do not create a second Forms scanner.
- [ ] Do not create a second monolithic installer.
- [ ] Do not create a second dependency graph.
- [ ] Do not create a second partitioned plan generator.
- [ ] Do not create a new field registry.
- [ ] Do not create a new region registry.
- [ ] Do not add FE-level physical terms such as fluid, solid, pressure,
      temperature, displacement, or heat flux.
- [ ] Do not make physics modules manually reproduce metadata that can be
      inferred from Forms or existing FE/Coupling declarations.

## Target Authoring Shape

The physics-facing surface should be a thin facade over existing FE/Coupling
records. A representative FSI module should eventually look like this:

```cpp
void FSICouplingModule::define(fec::CouplingDefinitionBuilder& c) const
{
    auto fluid = c.participant("fluid", options_.fluid_name);
    auto solid = c.participant("solid", options_.solid_name);
    auto gamma = c.sharedInterface(options_.interface_name, fluid, solid);

    auto u_f = fluid.vectorField(options_.fluid_velocity_field);
    auto p_f = fluid.scalarField(options_.fluid_pressure_field);
    auto d_s = solid.vectorField(options_.solid_displacement_field);

    c.monolithic([&](fec::CouplingForms& f) {
        auto uf = gamma.side(fluid).state(u_f, "u_f");
        auto wf = gamma.side(fluid).test(u_f, "w_f");
        auto pf = gamma.side(fluid).state(p_f, "p_f");
        auto ws = gamma.side(solid).test(d_s, "w_s");
        auto n = gamma.side(fluid).normal();

        auto vs = options_.use_solid_displacement_derivative
            ? gamma.side(solid).dt(d_s, "dt_d_s")
            : gamma.side(solid).state(options_.solid_velocity_field, "v_s");

        f.residual("velocity_continuity", gamma.integral(inner(uf - vs, wf)));
        f.residual("pressure_traction_balance",
                   gamma.integral(-inner(pf * n, ws)));
    });

    c.partitioned([&](fec::PartitionedCoupling& p) {
        p.exchange("solid_displacement", solid.field(d_s), fluid.field(u_f))
            .transfer(options_.solid_to_fluid_transfer);

        p.exchange("fluid_load", fluid.field(u_f), solid.field(d_s))
            .transfer(options_.fluid_to_solid_transfer);
    });
}
```

This facade should compile to existing FE/Coupling backend records. It should
not own independent graph, form-analysis, install, or transfer-plan logic.

## Resolved Authoring Decisions

These decisions are the scope of this migration. They keep the plan focused on
the compact, math-first facade over the coupling infrastructure that already
exists.

### Physics Authoring Rule

Physics coupling files must remain math-first and concise. They should declare
physical roles, field roles, coupling relations, and high-level strategy
options. They should not own generic validation, graph construction, transfer
resolution, metadata reconciliation, or partitioned driver execution.

Checklist:

- [ ] Physics modules define the physical coupling relation once.
- [ ] Physics modules provide equation-level Forms lowering for monolithic
      residuals.
- [ ] Physics modules provide logical exchange-channel lowering for partitioned
      strategies.
- [ ] Physics modules expose only high-level strategy options by default.
- [ ] Advanced or expert configuration is isolated from the normal physics
      authoring path.

### Minimum Public Facade

The first public facade should be intentionally small:

- [ ] participant roles;
- [ ] field roles and expected value shapes;
- [ ] shared-interface and region-relation handles;
- [ ] relation declaration with a relation law name;
- [ ] monolithic Forms block;
- [ ] partitioned exchange block;
- [ ] high-level strategy options.

N-way fixtures, contact examples, symbolic derived endpoints, and multi-PDE
auxiliary examples should validate that the direction is robust, but they should
not force a large first API surface.

### Stable Generated Names

The facade must generate stable backend names so compact physics code still
produces readable diagnostics, reproducible tests, and predictable metadata.

Naming rule:

```text
<contract_name>.<relation_name>.<contribution_or_exchange_name>
```

Checklist:

- [x] Shared generated-name helper uses
      `<contract_name>.<relation_name>.<contribution_or_exchange_name>` when no
      explicit override is supplied.
- [ ] Residual contribution names are stable.
- [x] Exchange names are stable.
- [ ] Coupling-owned field and variable names are stable.
- [x] Explicit name overrides are available for advanced cases.
- [ ] Generated names appear in diagnostics and installed metadata.

### Relation Lowering Capabilities

Each relation should declare whether it supports monolithic Forms lowering,
partitioned exchange lowering, or an explicit expert path. Unsupported
strategies should fail with clear diagnostics.

Checklist:

- [x] Relation lowerings are declared in one place.
- [x] Unsupported monolithic lowering fails before Forms installation.
- [x] Unsupported partitioned lowering fails before plan generation.
- [x] Expert fallback requires explicit opt-in.
- [ ] Physics modules do not hand-code backend capability branching.

### AuxiliaryState Strategy Inheritance

AuxiliaryState models admitted into a coupling contract inherit the coupling
contract strategy. They do not choose a separate solve strategy.

Checklist:

- [ ] Single-PDE ODE/DAE boundary models remain in AuxiliaryState.
- [ ] Multi-PDE AuxiliaryState coupling follows the selected coupling strategy.
- [ ] Monolithic lowering declares whether auxiliary coupling is exact,
      approximate, or lagged.
- [ ] Partitioned lowering routes auxiliary inputs and outputs through the same
      coupling exchange strategy as the PDE endpoints.
- [ ] Mixed AuxiliaryState/PDE strategies inside one contract are rejected
      unless explicitly represented as an expert strategy.

AuxiliaryState remains responsible for model equations, state storage,
model-local time integration, and model-local residual evaluation.

### Derivative And Lagging Disclosure

Physics authors should not expose backend derivative machinery, but the facade
must let a relation disclose whether its monolithic contribution is exact or
lagged.

Checklist:

- [ ] Relation lowerings can be marked `Exact`, `Approximate`, `Lagged`, or
      `Unavailable`.
- [ ] Approximate or lagged lowerings appear in diagnostics.
- [ ] Exact monolithic lowerings require metadata sufficient for dependency and
      block validation.

### User Configuration Surface

Use layered configuration:

```text
Default layer:
  relation law, enforcement strategy, solve strategy, and named transfers.

Advanced layer:
  relaxation, convergence, time-window, frame transform, projection, and
  geometry sensitivity options.

Expert layer:
  explicit fallback hooks, driver-owned payloads, custom transfer maps, and
  metadata overrides.
```

Implementation requirements:

- [ ] Keep default physics options short and domain-focused.
- [ ] Hide FE/Coupling backend records from normal physics configuration.
- [ ] Require explicit opt-in for expert hooks.
- [ ] Document every expert option with required diagnostics and validation
      obligations.

### Migration Compatibility

The new authoring facade should be layered over the existing low-level
`CouplingContract` API.

Checklist:

- [x] Existing low-level contracts continue to compile.
- [ ] FSI and thermal migrate behind the new facade after parity tests exist.
- [ ] Duplicated physics-local validation is removed only after backend parity
      is demonstrated.

## Out Of Scope For This Authoring Plan

These topics are important backend/runtime concerns, but they should not drive
the compact authoring migration:

- [ ] MPI rank ownership, reductions, ghost values, and communicator policy.
- [ ] Transfer-map cache invalidation and moving-interface rebuild policy.
- [ ] Full partitioned driver execution-loop design.
- [ ] Restart/checkpoint lifecycle details beyond stable naming requirements.
- [ ] Multi-system monolithic solve contexts beyond existing backend support.
- [ ] Conservation or energy audit machinery beyond relation-level examples and
      tests.
- [ ] Detailed storage ownership internals for global or algebraic variables.

## Example Strategy Sketches

These examples are future-shape sketches used to test the authoring model. They
are not implementation commitments for the current migration. The important
rule is that residual mathematics lowers through FE/Forms when appropriate,
while data movement, partitioned algorithms, transfer setup, and driver
scheduling lower through FE/Coupling declarations and plan metadata.

The examples intentionally focus on PDE-to-PDE coupling. Single-PDE ODE/DAE
boundary models remain AuxiliaryState use cases unless the model consumes or
drives multiple PDE participants.

Every sketch below declares one physical relation and then shows both:

- [ ] monolithic lowering through Forms residuals when the relation contributes
      to a shared solve; and
- [ ] partitioned lowering through exchange declarations and strategy metadata
      when the relation is solved by a partitioned driver.

If a future relation cannot support one of those lowerings, the relation should
declare that limitation explicitly and FE/Coupling should produce an
unsupported-strategy diagnostic.

### Thermal Interface: Symmetric Nitsche

The monolithic strategy uses Forms because the interface condition contributes
weak residual terms to one shared solve. The partitioned strategy uses the same
temperature-continuity and flux-balance relation to declare temperature and
heat-flux exchanges.

```cpp
void ThermalInterfaceCouplingModule::define(fec::CouplingDefinitionBuilder& c) const
{
    auto side_a = c.participant("side_a", options_.side_a_name);
    auto side_b = c.participant("side_b", options_.side_b_name);
    auto gamma = c.sharedInterface(options_.interface_name, side_a, side_b)
                  .enforcement(fec::EnforcementStrategy::Nitsche);

    auto T_a = side_a.scalarField(options_.side_a_temperature_field);
    auto T_b = side_b.scalarField(options_.side_b_temperature_field);
    auto q_a = side_a.scalarField(options_.side_a_heat_flux_field);
    auto q_b = side_b.scalarField(options_.side_b_heat_flux_field);

    auto relation = c.relation("thermal_interface")
                     .fields(T_a, T_b, q_a, q_b)
                     .interface(gamma)
                     .law(fec::CouplingLaw::TemperatureContinuityAndFluxBalance);

    c.monolithic([&](fec::CouplingForms& f) {
        auto Ta = gamma.side(side_a).state(T_a, "T_a");
        auto wa = gamma.side(side_a).test(T_a, "w_a");
        auto Tb = gamma.side(side_b).state(T_b, "T_b");
        auto wb = gamma.side(side_b).test(T_b, "w_b");

        auto k_a = side_a.coefficient(options_.side_a_conductivity_name);
        auto k_b = side_b.coefficient(options_.side_b_conductivity_name);

        auto residual = gamma.nitscheScalarContinuity(Ta, wa, k_a,
                                                      Tb, wb, k_b,
                                                      options_.nitsche);
        f.residual("temperature_continuity", residual);
    });

    c.partitioned([&](fec::PartitionedCoupling& p) {
        p.strategy(options_.partitioned_strategy);

        p.exchange("temperature_a_to_b", side_a.field(T_a), side_b.field(T_b))
            .sharedInterface(gamma)
            .transfer(options_.temperature_transfer);

        p.exchange("heat_flux_b_to_a", side_b.field(q_b), side_a.field(q_a))
            .sharedInterface(gamma)
            .transfer(options_.heat_flux_transfer);
    });
}
```

Backend expectations:

- [ ] The physical relation is declared once.
- [ ] `gamma.nitscheScalarContinuity(...)` lowers to ordinary `forms::FormExpr`
      terms using existing FE/Forms interface and trace helpers.
- [ ] Partitioned lowering maps the same temperature-continuity and flux-balance
      relation to temperature and heat-flux exchange channels.
- [ ] Conductivity coefficients are recorded as non-field dependencies by the
      Forms metadata bridge.
- [ ] The shared interface, sides, normals, measures, and penalty terms are
      validated by FE/Coupling and FE/Forms metadata.

### FSI Interface: Multiplier Or Mortar Constraint

This strategy introduces a coupling-owned interface unknown for monolithic
lowering. The same velocity-continuity and traction-balance law can lower to
partitioned exchanges, using the same mortar/projection metadata for transfer
rather than installing multiplier rows.

```cpp
void FSIMortarCouplingModule::define(fec::CouplingDefinitionBuilder& c) const
{
    auto fluid = c.participant("fluid", options_.fluid_name);
    auto solid = c.participant("solid", options_.solid_name);
    auto gamma = c.sharedInterface(options_.interface_name, fluid, solid)
                  .enforcement(fec::EnforcementStrategy::Multiplier)
                  .mortar(options_.mortar);

    auto u_f = fluid.vectorField(options_.fluid_velocity_field);
    auto d_s = solid.vectorField(options_.solid_displacement_field);
    auto t_f = fluid.vectorField(options_.fluid_traction_field);
    auto t_s = solid.vectorField(options_.solid_traction_field);
    auto lambda = c.interfaceMultiplier("lambda", gamma)
                   .components(options_.interface_components)
                   .space(options_.multiplier_space);

    auto relation = c.relation("fsi_mortar_interface")
                     .fields(u_f, d_s, t_f, t_s)
                     .interface(gamma)
                     .law(fec::CouplingLaw::VelocityContinuityAndTractionBalance);

    c.monolithic([&](fec::CouplingForms& f) {
        auto uf = gamma.side(fluid).state(u_f, "u_f");
        auto wf = gamma.side(fluid).test(u_f, "w_f");
        auto vs = gamma.side(solid).dt(d_s, "dt_d_s");
        auto ws = gamma.side(solid).test(d_s, "w_s");
        auto l = gamma.multiplier(lambda).state("lambda");
        auto mu = gamma.multiplier(lambda).test("mu");

        f.residual("fluid_multiplier_load",
                   gamma.integral(inner(l, wf), fluid));
        f.residual("solid_multiplier_load",
                   gamma.integral(-inner(l, ws), solid));
        f.residual("velocity_constraint",
                   gamma.integral(inner(uf - vs, mu), fluid));
    });

    c.partitioned([&](fec::PartitionedCoupling& p) {
        p.strategy(options_.partitioned_strategy)
            .projection(options_.mortar.partitioned_projection);

        p.exchange("solid_motion_to_fluid", solid.field(d_s), fluid.field(u_f))
            .sharedInterface(gamma)
            .transfer(options_.solid_to_fluid_transfer);

        p.exchange("fluid_traction_to_solid", fluid.field(t_f), solid.field(t_s))
            .sharedInterface(gamma)
            .transfer(options_.fluid_to_solid_transfer);
    });
}
```

Backend expectations:

- [ ] The physical velocity-continuity and traction-balance relation is declared
      once.
- [ ] `interfaceMultiplier(...)` lowers to
      `CouplingAdditionalFieldDeclaration`.
- [ ] Multiplier field IDs are resolved before Forms are built.
- [ ] Multiplier rows and columns appear in installed block metadata.
- [ ] Partitioned lowering uses the same mortar/projection metadata for exchange
      transfer rather than installing multiplier rows.
- [ ] Mortar compatibility and trace-space checks are FE/Coupling validation,
      not physics-local validation.

### N-Way PDE Junction: Conservation With One Shared Algebraic Variable

This strategy couples many PDE participants at a junction. The conservation law
is a residual relation; the shared junction pressure is a global algebraic
variable owned by the coupling contract. The boundary flow terms can be
represented through Forms or through boundary-functional metadata, depending on
the final public API available at implementation time.

```cpp
void FlowJunctionCouplingModule::define(fec::CouplingDefinitionBuilder& c) const
{
    auto branches = c.participantGroup("branches", options_.branch_names);
    auto junction = c.regionRelation("junction", branches)
                     .kind(fec::RegionRelationKind::NWayInterface);

    auto p_j = c.globalScalar("junction_pressure");
    auto relation = c.relation("junction_conservation")
                     .participants(branches)
                     .regionRelation(junction)
                     .law(fec::CouplingLaw::NWayConservation);

    c.monolithic([&](fec::CouplingForms& f) {
        std::vector<forms::FormExpr> flow_terms;
        for (auto branch : branches) {
            auto u = junction.endpoint(branch).state(
                branch.vectorField(options_.velocity_field), "u");
            auto n = junction.endpoint(branch).normal();
            flow_terms.push_back(junction.endpoint(branch).integral(inner(u, n)));
        }

        f.algebraicResidual("junction_mass_balance",
                            p_j,
                            forms::sum(flow_terms));

        for (auto branch : branches) {
            auto v = junction.endpoint(branch).test(
                branch.vectorField(options_.velocity_field), "v");
            auto n = junction.endpoint(branch).normal();
            f.residual("junction_pressure_load",
                       junction.endpoint(branch).integral(-p_j * inner(v, n)));
        }
    });

    c.partitioned([&](fec::PartitionedCoupling& p) {
        p.strategy(options_.partitioned_strategy);

        for (auto branch : branches) {
            p.exchange("branch_flow_to_junction",
                       branch.boundaryFunctional(options_.flow_functional_name),
                       p_j.input("branch_flow", branch))
                .regionRelation(junction)
                .transfer(options_.flow_transfer);

            p.exchange("junction_pressure_to_branch",
                       p_j,
                       branch.boundaryInput(options_.pressure_input_name))
                .regionRelation(junction)
                .transfer(options_.pressure_transfer);
        }
    });
}
```

Backend expectations:

- [ ] The N-way conservation relation is declared once.
- [ ] Monolithic lowering installs the conservation residual and branch pressure
      loads.
- [ ] Partitioned lowering maps branch flow functionals and junction pressure
      updates to exchange channels.
- [ ] N-way participant and endpoint validation is handled by
      `CouplingRegionRelationRequirement`.
- [ ] `globalScalar(...)` lowers through FE/Analysis variable metadata.
- [ ] Boundary integrals and global scalar dependencies are visible in
      finalized coupling graph diagnostics.
- [ ] No two-side `minus()` / `plus()` assumption is required.

### Multi-PDE Auxiliary State Coupling

This strategy is only in the coupling framework because one auxiliary model
depends on state from multiple PDE participants. A single-PDE outlet circuit or
other one-field boundary model should stay in AuxiliaryState. Here, the shared
algebraic variables enter multiple PDE residuals or exchanges, so FE/Coupling
must validate the cross-PDE dependencies.

```cpp
void MultiPDEAuxiliaryCouplingModule::define(fec::CouplingDefinitionBuilder& c) const
{
    auto fluid = c.participant("fluid", options_.fluid_name);
    auto solid = c.participant("solid", options_.solid_name);
    auto relation = c.regionRelation("fluid_solid_auxiliary", fluid, solid)
                     .kind(fec::RegionRelationKind::AuxiliaryPDECoupling);

    auto u_f = fluid.vectorField(options_.fluid_velocity_field);
    auto d_s = solid.vectorField(options_.solid_displacement_field);

    auto model = c.auxiliaryModel("shared_interface_model",
                                  options_.auxiliary_model_name)
                  .input("fluid_flow", relation.endpoint(fluid).normalFlux(u_f))
                  .input("solid_displacement",
                         relation.endpoint(solid).meanValue(d_s));

    auto alpha = model.output("interface_response");

    c.monolithic([&](fec::CouplingForms& f) {
        auto wf = relation.endpoint(fluid).test(u_f, "w_f");
        auto ws = relation.endpoint(solid).test(d_s, "w_s");
        auto nf = relation.endpoint(fluid).normal();
        auto ns = relation.endpoint(solid).normal();

        f.residual("fluid_auxiliary_load",
                   relation.endpoint(fluid).integral(-alpha * inner(wf, nf)));
        f.residual("solid_auxiliary_load",
                   relation.endpoint(solid).integral(alpha * inner(ws, ns)));
    });

    c.partitioned([&](fec::PartitionedCoupling& p) {
        p.exchange("fluid_flow_to_auxiliary",
                   fluid.boundaryFunctional("interface_flow"),
                   model.input("fluid_flow"));
        p.exchange("solid_state_to_auxiliary",
                   solid.boundaryFunctional("interface_displacement"),
                   model.input("solid_displacement"));
        p.exchange("auxiliary_response",
                   model.output("interface_response"),
                   relation.sharedInput("interface_response"));
    });
}
```

Backend expectations:

- [ ] Field residual terms lower through Forms.
- [ ] Auxiliary variables lower through AuxiliaryState deployment records and
      Analysis variable keys.
- [ ] The auxiliary model follows the coupling contract strategy: monolithic
      variables and PDE residual terms in monolithic mode, partitioned exchanges
      and driver schedule in partitioned mode.
- [ ] AuxiliaryState remains the owner of the ODE/DAE equations; FE/Coupling
      only validates cross-PDE dependencies and routes variables into PDE
      residuals or exchanges.
- [ ] FE/Coupling validates that the auxiliary relation depends on multiple PDE
      participants before accepting it as a coupling contract.
- [ ] Partitioned exchanges lower to existing endpoint declarations.
- [ ] Transfer and temporal-slot validation stay in
      `PartitionedCouplingPlanGenerator`.

### Electro-Thermal Coupling: Same-Domain PDE Relation

This strategy does not require a shared interface. It is a volume relation
between fields that may live in the same participant or in compatible
participants sharing the same domain. Monolithic lowering installs the heat
source residual directly; partitioned lowering exchanges temperature and
derived heat-source data between the electrical and thermal PDE solves.

```cpp
void ElectroThermalCouplingModule::define(fec::CouplingDefinitionBuilder& c) const
{
    auto electric = c.participant("electric", options_.electric_name);
    auto thermal = c.participant("thermal", options_.thermal_name);
    auto relation = c.domainRelation("joule_heating", electric, thermal)
                     .law(fec::CouplingLaw::JouleHeating);

    auto phi = electric.scalarField(options_.electric_potential_field);
    auto T = thermal.scalarField(options_.temperature_field);
    auto heat = thermal.scalarField(options_.joule_heat_source_field);
    auto sigma = electric.coefficient(options_.conductivity_name);

    c.monolithic([&](fec::CouplingForms& f) {
        auto ph = relation.state(phi, "phi");
        auto theta = relation.test(T, "theta");

        auto joule_heat = sigma * inner(grad(ph), grad(ph));
        f.residual("joule_heat_source",
                   relation.volumeIntegral(-joule_heat * theta));
    });

    c.partitioned([&](fec::PartitionedCoupling& p) {
        p.strategy(options_.partitioned_strategy);

        p.exchange("temperature_to_electric",
                   thermal.field(T),
                   electric.materialInput(options_.temperature_input_name))
            .regionRelation(relation)
            .transfer(options_.temperature_transfer);

        p.exchange("joule_heat_to_thermal",
                   electric.derivedField("joule_heat", sigma * inner(grad(phi), grad(phi))),
                   thermal.field(heat))
            .regionRelation(relation)
            .transfer(options_.heat_source_transfer);
    });
}
```

Backend expectations:

- [ ] The electro-thermal relation is declared once.
- [ ] Domain relations use existing Forms volume measures.
- [ ] Cross-field dependencies are inferred from installed Forms metadata.
- [ ] Coefficients are recorded as non-field dependencies.
- [ ] Partitioned lowering maps temperature and heat-source data through
      exchange declarations.
- [ ] No shared-interface machinery is required.

### FSI Relation Reused Across Monolithic And Partitioned Strategies

The physical coupling relation should be strategy-independent where practical.
The same FSI relation declares velocity continuity and traction balance once.
Monolithic lowering turns the relation into Forms residuals; partitioned
lowering turns the relation into displacement and traction exchanges with
fixed-point strategy metadata.

```cpp
void FSICouplingModule::define(fec::CouplingDefinitionBuilder& c) const
{
    auto fluid = c.participant("fluid", options_.fluid_name);
    auto solid = c.participant("solid", options_.solid_name);
    auto gamma = c.sharedInterface(options_.interface_name, fluid, solid);

    auto u_f = fluid.vectorField(options_.fluid_velocity_field);
    auto p_f = fluid.scalarField(options_.fluid_pressure_field);
    auto d_s = solid.vectorField(options_.solid_displacement_field);

    auto relation = c.relation("fsi_interface")
                     .fields(u_f, p_f, d_s)
                     .interface(gamma)
                     .law(fec::CouplingLaw::VelocityContinuityAndTractionBalance);

    c.monolithic([&](fec::CouplingForms& f) {
        auto uf = gamma.side(fluid).state(u_f, "u_f");
        auto wf = gamma.side(fluid).test(u_f, "w_f");
        auto pf = gamma.side(fluid).state(p_f, "p_f");
        auto vs = gamma.side(solid).dt(d_s, "dt_d_s");
        auto ws = gamma.side(solid).test(d_s, "w_s");
        auto n = gamma.side(fluid).normal();

        f.residual("velocity_continuity",
                   gamma.integral(inner(uf - vs, wf), fluid));
        f.residual("traction_balance",
                   gamma.integral(-inner(pf * n, ws), fluid));
    });

    auto d_f = fluid.vectorField(options_.fluid_mesh_displacement_field);
    auto t_f = fluid.vectorField(options_.fluid_traction_field);
    auto t_s = solid.vectorField(options_.solid_traction_field);

    c.partitioned([&](fec::PartitionedCoupling& p) {
        p.strategy(fec::PartitionedStrategy::FixedPoint)
            .relaxation(fec::RelaxationStrategy::Dynamic)
            .convergenceNorm(fec::ConvergenceNorm::InterfaceIncrement)
            .maximumIterations(options_.maximum_iterations);

        p.exchange("solid_displacement", solid.field(d_s), fluid.field(d_f))
            .sharedInterface(gamma)
            .transfer(options_.solid_to_fluid_transfer)
            .producerTemporal(options_.solid_displacement_source)
            .consumerTemporal(options_.fluid_displacement_target);

        p.exchange("fluid_traction", fluid.field(t_f), solid.field(t_s))
            .sharedInterface(gamma)
            .transfer(options_.fluid_to_solid_transfer)
            .producerTemporal(options_.fluid_load_source)
            .consumerTemporal(options_.solid_load_target);
    });
}
```

Backend expectations:

- [ ] The physical relation is declared once.
- [ ] Monolithic strategy lowers the relation to Forms residuals.
- [ ] Exchange records lower to `CouplingExchangeDeclaration`.
- [ ] Strategy, relaxation, convergence, and iteration limits are plan or
      driver metadata, not Forms expressions.
- [ ] Transfer compatibility and endpoint regions are validated by
      `PartitionedCouplingPlanGenerator`.
- [ ] If a relation cannot lower to a selected strategy, FE/Coupling reports an
      unsupported strategy diagnostic instead of requiring physics-local
      branching.

### Contact Or Friction Interface: Expert-Gated Inequality Relation

This strategy can support both monolithic and partitioned lowering, but the
monolithic path may require an expert hook until public complementarity and
active-set metadata are available. The partitioned path uses driver-owned
exchange channels for motion and contact traction data.

```cpp
void ContactCouplingModule::define(fec::CouplingDefinitionBuilder& c) const
{
    auto master = c.participant("master", options_.master_name);
    auto slave = c.participant("slave", options_.slave_name);
    auto contact = c.sharedInterface(options_.contact_interface, master, slave)
                    .enforcement(fec::EnforcementStrategy::ActiveSet)
                    .geometry(options_.contact_geometry);

    auto d_m = master.vectorField(options_.master_displacement_field);
    auto d_s = slave.vectorField(options_.slave_displacement_field);

    c.monolithic([&](fec::CouplingForms& f) {
        auto gap = contact.normalGap(d_m, d_s);
        auto pressure = contact.contactPressure("p_c");
        auto test_gap = contact.normalGapTest(d_m, d_s);

        f.inequalityResidual("normal_contact",
                             pressure,
                             gap,
                             test_gap,
                             options_.active_set);
    });

    c.partitioned([&](fec::PartitionedCoupling& p) {
        p.strategy(fec::PartitionedStrategy::DriverOwned)
            .driver(options_.contact_driver_name);

        p.exchange("master_motion_to_contact", master.field(d_m),
                   contact.driverInput("master_motion"))
            .sharedInterface(contact)
            .transfer(options_.master_motion_transfer);

        p.exchange("slave_motion_to_contact", slave.field(d_s),
                   contact.driverInput("slave_motion"))
            .sharedInterface(contact)
            .transfer(options_.slave_motion_transfer);

        p.exchange("contact_traction_to_master",
                   contact.driverOutput("master_contact_traction"),
                   master.boundaryInput(options_.master_contact_traction_input))
            .sharedInterface(contact)
            .transfer(options_.master_traction_transfer);

        p.exchange("contact_traction_to_slave",
                   contact.driverOutput("slave_contact_traction"),
                   slave.boundaryInput(options_.slave_contact_traction_input))
            .sharedInterface(contact)
            .transfer(options_.slave_traction_transfer);
    });
}
```

Backend expectations:

- [ ] The contact relation is declared once.
- [ ] Monolithic lowering uses Forms only when public Forms and Systems metadata
      can represent the inequality residual.
- [ ] Partitioned lowering uses driver-owned exchange channels for motion input,
      active-set/contact solve state, and contact tractions.
- [ ] If `inequalityResidual(...)` is not supported by public Forms and
      Systems metadata, this module must use an expert path.
- [ ] The expert path must still provide equivalent dependency, block,
      geometry, and active-set diagnostics.
- [ ] Geometry search, gap evaluation, and active-set state remain backend or
      driver-owned mechanisms.

## Phase 0: Generality Requirements And Forms Applicability

### Rationale

The math-first coupling path must support more than two-participant field
interfaces. Future modules may couple PDE domains through constraints, flux
balances, multipliers, shared algebraic variables, multi-PDE auxiliary states,
nonmatching regions, moving interfaces, or partitioned driver algorithms.

Math-first authoring should therefore be broader than Forms-only authoring:

```text
Use FE/Forms when the coupling is a weak residual, interface residual,
constraint residual, penalty residual, multiplier residual, or algebraic
residual that belongs in a monolithic solve.

Use FE/Coupling exchange declarations when the coupling is data movement,
partitioned transfer, lagged input/output, driver-owned transfer, or an
external-buffer exchange.

Use existing FE/Analysis variable metadata when the coupling touches auxiliary
states, global scalars, boundary functionals, material state, or variables whose
dependencies span multiple PDE participants.

Use direct expert hooks only when a coupling cannot yet be represented through
the public Forms, Systems, Analysis, or Coupling surfaces.
```

The authoring facade must make those choices explicit without moving backend
responsibility out of the existing FE layers.

### Coupling Relation Types

The definition layer must represent these relation families:

- [ ] Strong or weak equality constraints.
- [ ] Flux, traction, current, or mass conservation balances.
- [ ] Robin, impedance, resistance, compliance, and admittance interface laws.
- [ ] Penalty and Nitsche-style interface enforcement.
- [ ] Lagrange multiplier and mortar constraints.
- [ ] Contact, inequality, complementarity, and active-set style interface laws.
- [ ] Frictional or tangential interface laws.
- [ ] Constitutive interface laws that evaluate traction, flux, source, or
      reaction terms from one or more side states.
- [ ] PDE-to-PDE couplings involving global scalar, auxiliary, or algebraic
      variables whose dependencies span multiple PDE participants.

When a relation contributes to the monolithic residual, it should be expressible
with FE/Forms whenever the public Forms vocabulary can represent the needed
operators. The builder should provide relation-level conveniences, but those
conveniences must lower to normal `forms::FormExpr` residuals and
`installFormulationWithMetadata()`.

### Participant Cardinality

The authoring path must support:

- [ ] Two-participant side-paired interfaces.
- [ ] N-participant junctions and conservation nodes.
- [ ] Hub-and-spoke coupling patterns.
- [ ] Optional participants and optional fields.
- [ ] Repeated participant groups, such as many PDE branches coupled through one
      shared junction relation.
- [ ] Self-coupling across multiple regions of one participant.

Two-side helpers such as `minus()` and `plus()` should remain available for
side-paired interfaces, but the generic model must not require every coupling
to have exactly two sides.

### Mixed-Dimensional Coupling

The authoring path must support region relations beyond matching
interface-face pairs:

- [ ] Volume-to-boundary coupling.
- [ ] Surface-to-surface coupling.
- [ ] Line-to-surface or curve-to-surface coupling.
- [ ] Point-to-volume and point-to-boundary coupling.
- [ ] Multi-PDE auxiliary-state coupling when an auxiliary model consumes or
      drives multiple PDE participants.
- [ ] Boundary-functional or integrated-quantity coupling.
- [ ] Coupling through shared global variables or auxiliary state variables
      whose dependencies span multiple PDE participants.

When these relations are weak residuals, the facade should still lower to
Forms measures and existing Forms/Systems metadata. When they are data movement
or auxiliary-state synchronization, they should lower to existing FE/Coupling
exchange, endpoint, AuxiliaryState, or variable-dependency records.

### Additional Unknowns And Algebraic Variables

The definition layer must expose clean authoring for coupling-owned unknowns:

- [ ] Interface-owned multiplier fields.
- [ ] Interface-owned penalty or stabilization fields.
- [ ] Contract-owned auxiliary state variables.
- [ ] Auxiliary input/output variables.
- [ ] Boundary functional variables.
- [ ] Global scalar variables.
- [ ] AuxiliaryState variables whose dependencies span multiple PDE
      participants.
- [ ] Static-condensation or local-elimination policies.

Coupling-owned fields should continue to lower through
`CouplingAdditionalFieldDeclaration`. Non-field unknowns and dependencies should
continue to lower through existing FE/Analysis variable keys and
`CouplingNonFieldDependencyRequirement` records. Single-PDE ODE/DAE boundary
models should continue to be authored and deployed through AuxiliaryState.

### Coupling Strategy Options

The public physics options should separate the mathematical relation from the
solution strategy:

- [ ] Enforcement strategy: strong, penalty, Nitsche, multiplier, mortar,
      explicit lagged, or expert.
- [ ] Solve strategy: monolithic, partitioned explicit, partitioned staggered,
      fixed-point, relaxed fixed-point, or driver-owned.
- [ ] Partitioned convergence strategy: residual norm, exchange increment norm,
      energy/work norm, maximum iterations, and failure policy.
- [ ] Relaxation strategy: constant relaxation, Aitken relaxation, or
      driver-provided relaxation.
- [ ] Time strategy: current, accepted, predicted, history, stage, external,
      subcycling, and time-window exchange.

Only weak residual and algebraic residual pieces should be authored as Forms.
Iteration algorithms, relaxation, subcycling, and driver scheduling should
remain partitioned-plan or driver metadata.

AuxiliaryState models admitted into a coupling contract must not choose an
independent solve strategy. They inherit the coupling contract strategy, while
AuxiliaryState remains responsible for model equations, state storage, and
model-local time integration details.

### Strategy-Independent Physical Relations

The preferred authoring model is:

```text
Define the physical coupling relation once.
Lower that relation to a monolithic Forms residual when monolithic mode is
selected.
Lower that relation to exchange declarations and strategy metadata when
partitioned mode is selected.
Reject unsupported strategy combinations through FE/Coupling diagnostics.
```

This is the right default because velocity continuity, traction balance,
temperature continuity, flux conservation, and similar physical laws should not
be rewritten in unrelated APIs for each solve strategy.

There are valid exceptions:

- [ ] Some enforcement mechanisms are inherently monolithic, such as multiplier
      rows that require a shared linear system.
- [ ] Some partitioned strategies exchange lagged approximations rather than
      assembling the exact residual relation.
- [ ] Some stabilization terms are strategy-specific.
- [ ] Some AuxiliaryState models may be usable only with one lowering until the
      coupling layer can represent the needed dependency, exchange, or schedule
      metadata.
- [ ] Some expert hooks may only support one lowering path until Forms,
      Systems, or plan metadata grows the needed public feature.

The facade should therefore model physical relation and strategy separately,
then require each relation to declare which lowerings it supports.

### Transform, Orientation, And Units Semantics

The definition layer must make transform requirements explicit enough for
FE/Coupling to validate them:

- [ ] Normal and tangential projection policies.
- [ ] Sign convention and outward-normal ownership.
- [ ] Reference versus current configuration.
- [ ] Frame transform source and target policies.
- [ ] Conservative versus interpolatory transfer policy.
- [ ] Component layout and tensor packing.
- [ ] Unit or dimension compatibility where metadata exists.
- [ ] Orientation-sensitive diagnostics for flux and traction balances.

Form-authored terms should use explicit Forms expressions for projections,
normals, tangential components, and frame-aware geometry terminals whenever
those quantities enter a residual.

### Geometry Lifecycle

The definition layer must support geometry policies for:

- [ ] Reference and current configurations.
- [ ] Moving interfaces.
- [ ] Sliding interfaces.
- [ ] Cut or embedded interfaces.
- [ ] Topology revision checks.
- [ ] Geometry revision checks.
- [ ] Mesh-motion ownership.
- [ ] Geometry sensitivity policy.
- [ ] Normal, measure, Jacobian, and quadrature sensitivity.

Geometry values in monolithic residuals should enter through
`CouplingFormBuilder` geometry terminals so `FormAnalysisBridge` and
`CouplingGraph` can validate provenance and sensitivity.

### Definition Lifecycle

The plan must keep the lifecycle explicit:

- [ ] Definition-time records: physical role names, field role names,
      high-level options, requested relations, and strategy choices.
- [ ] Context-resolution records: participants, fields, regions, shared-region
      mappings, endpoint resolution, and value-shape checks.
- [ ] Additional-field registration records: coupling-owned fields and their
      resolved `FieldId`s.
- [ ] Form-build records: Forms residuals, terminal declarations, and install
      options.
- [ ] Install records: installed Forms metadata, dependencies, expected blocks,
      temporal symbols, and geometry terminal provenance.
- [ ] Plan records: partitioned exchanges, transfer plans, runtime handles, and
      driver-facing execution metadata.

Each lifecycle stage should have one owning FE layer. Physics modules should
not manually duplicate records from later stages.

### Fallback And Expert Policy

Direct `CouplingContract` overrides remain necessary for some advanced cases.
The plan should define when that is acceptable:

- [ ] The relation cannot be represented by current public Forms vocabulary.
- [ ] The install path requires custom Systems extension points.
- [ ] The partitioned driver owns algorithmic state that is not representable as
      exchange declarations.
- [ ] The coupling requires a custom transfer map that is not yet expressible
      through `TransferPlan`.
- [ ] The metadata bridge cannot yet report the evidence needed for safe graph
      validation.

Expert paths must still provide diagnostics and metadata equivalent to the
definition-backed path before they are considered complete.

## Phase 1: Centralize Field And Interface Requirements

### Rationale

Physics modules currently perform many common checks by hand: nonempty names,
component counts, scalar/vector expectations, shared-region participant mapping,
opposite-side checks, and monolithic same-system checks. These are generic
coupling checks and should be owned by FE/Coupling.

### Concrete Changes

- [x] Add a field value-shape declaration adjacent to `CouplingFieldUse`.
      Suggested shape:

```cpp
struct CouplingFieldRequirement {
    CouplingFieldUse field;
    CouplingValueDescriptor value;
    std::optional<systems::FieldScope> required_scope;
    CouplingRequirement requirement{CouplingRequirement::Required};
};
```

- [x] Add field requirements to `CouplingContractDeclaration`.
- [x] Keep `fields` for compatibility during migration, but prefer
      `field_requirements` in new code.
- [x] Add shared-interface participant requirements to
      `CouplingSharedRegionUse` or an adjacent declaration:

```cpp
struct CouplingSharedInterfaceRequirement {
    std::string shared_region_name;
    std::vector<std::string> participant_names;
    CouplingRegionKind required_region_kind{CouplingRegionKind::InterfaceFace};
    bool require_all_participants{true};
    bool require_opposite_sides_for_two_participants{true};
    bool require_monolithic_topology{false};
};
```

- [x] Add a more general region-relation declaration for non-two-sided and
      mixed-dimensional coupling. Suggested shape:

```cpp
enum class CouplingRegionRelationKind {
    SidePairedInterface,
    NWayInterface,
    SameParticipantRegions,
    EmbeddedRelation,
    VolumeBoundaryRelation,
    AuxiliaryPDECoupling,
};

struct CouplingRegionRelationRequirement {
    std::string relation_name;
    CouplingRegionRelationKind relation_kind{
        CouplingRegionRelationKind::SidePairedInterface};
    std::vector<CouplingRegionEndpointDeclaration> endpoints;
    std::optional<CouplingRegionKind> required_region_kind;
    bool require_all_endpoints{true};
    bool require_distinct_participants{false};
    bool require_opposite_sides_for_side_pair{false};
    bool require_common_monolithic_system{false};
    bool require_registered_topology{false};
};
```

- [x] Add relation lowering capability records and declaration-time
      diagnostics for missing, duplicate, or unsupported lowering declarations.
- [ ] Prefer `CouplingRegionRelationRequirement` for new code when a relation is
      not exactly a simple two-side shared interface.
- [x] Teach `CouplingGraph` to validate field rank, component count, and field
      scope against `CouplingContext`.
- [x] Teach `CouplingGraph` to validate shared-interface participant mappings
      against `SharedRegionRegistry`.
- [x] Teach `CouplingGraph` to validate opposite-side mappings for
      two-participant interfaces.
- [x] Teach `CouplingGraph` to validate N-way and mixed-dimensional region
      relation endpoints.
- [x] Teach `CouplingGraph` to validate optional endpoint policies without
      requiring physics modules to hand-code participant branches.
- [x] Teach `CouplingGraph` to validate monolithic same-system compatibility
      from resolved relation endpoints and existing installed dependency
      metadata.
- [x] Teach `CouplingGraph` to validate registered interface topology for
      monolithic interface-face forms.

### Completion Checklist

- [x] FSI no longer needs local vector component checks.
- [x] FSI no longer needs local scalar pressure checks.
- [x] FSI no longer needs local interface participant mapping checks.
- [x] FSI no longer needs local opposite-side checks.
- [x] FSI no longer needs local monolithic same-system checks.
- [x] Thermal no longer needs local scalar/vector field shape checks.
- [ ] New N-participant and mixed-dimensional relation fixtures validate through
      `CouplingGraph`.
- [x] Relation capability records reject unsupported mode/lowering combinations
      before Forms installation or partitioned plan execution.
- [ ] Orientation policy diagnostics cover side-paired, N-way, flux-balance,
      and traction-balance relations.
- [ ] Existing `CouplingGraph` diagnostics remain the single validation surface.

## Phase 2: Extend CouplingFormBuilder With Interface Views

### Rationale

Physics modules should not manually translate shared-interface roles into
`minus()` or `plus()` restrictions, geometry terminal scopes, and shared-region
integrals. That mapping is generic FE/Coupling work.

### Concrete Changes

- [x] Add lightweight interface helper types in FE/Coupling:
      `CouplingSharedInterfaceView` and `CouplingInterfaceSideView`.
- [ ] Add `CouplingInterfaceIntegralView` if a dedicated integral object is
      needed beyond `CouplingSharedInterfaceView::integral(...)`.
- [x] Add lightweight region-relation helper types for N-way and
      mixed-dimensional relations that do not have `minus()` / `plus()` sides.
- [x] Implement the helpers inside or next to `CouplingFormBuilder`.
- [ ] Add:

```cpp
CouplingSharedInterfaceView sharedInterface(std::string_view name) const;
CouplingInterfaceSideView CouplingSharedInterfaceView::side(
    std::string_view participant) const;
forms::FormExpr CouplingInterfaceSideView::state(...);
forms::FormExpr CouplingInterfaceSideView::test(...);
forms::FormExpr CouplingInterfaceSideView::dt(...);
forms::FormExpr CouplingInterfaceSideView::normal(...);
forms::FormExpr CouplingSharedInterfaceView::integral(
    const forms::FormExpr& integrand,
    std::string_view integration_participant) const;
```

- [ ] Add generic relation helpers:

```cpp
CouplingRegionRelationView regionRelation(std::string_view name) const;
CouplingRegionEndpointView CouplingRegionRelationView::endpoint(
    std::string_view endpoint_name) const;
forms::FormExpr CouplingRegionRelationView::integral(
    const forms::FormExpr& integrand,
    std::string_view endpoint_name) const;
forms::FormExpr CouplingRegionRelationView::sum(
    std::span<const forms::FormExpr> endpoint_terms) const;
```

- [x] Implement `state`, `test`, and `dt` through existing
      `CouplingFormBuilder::state`, `test`, and `timeDerivative`.
- [x] Implement side restriction through existing `FormExpr::minus()` and
      `FormExpr::plus()`.
- [x] Implement `normal()` through existing `geometryTerminal(...)`.
- [x] Implement `integral(...)` through existing `integrateShared(...)`.
- [x] Implement relation integrals through existing Forms measures and
      `CouplingFormBuilder::integrate(...)` / `integrateShared(...)`, choosing
      the measure from resolved region metadata.
- [x] Expose projection helpers that lower to Forms expressions for normal and
      tangential components.
- [x] Preserve existing lower-level methods for expert use.

### Completion Checklist

- [x] FSI monolithic forms no longer manually call `restrictToInterfaceSide`.
- [x] FSI monolithic forms no longer manually construct
      `CouplingGeometryTerminalScope`.
- [ ] Thermal monolithic forms can use the same interface-view API.
- [x] N-way conservation residuals can be written without manually iterating
      over low-level context records.
- [x] Mixed-dimensional residuals can select the correct resolved integration
      domain through the builder.
- [x] Existing `CouplingFormBuilder` tests still pass.
- [x] New tests cover missing side mappings and missing shared-region mappings.

## Phase 3: Infer Monolithic Declarations From Forms Metadata

### Rationale

Monolithic physics modules currently duplicate information. They manually
declare dependencies and expected blocks, then build Forms residuals that imply
the same dependencies. The installed Forms metadata should become the primary
evidence.

### Concrete Changes

- [ ] Extend `MonolithicCouplingBuilder` to run declaration validation in two
      stages:
      1. Pre-install validation from declared participants, fields, and
         shared-region requirements.
      2. Finalized validation from installed `CouplingFormAnalysisMetadata`.
- [ ] Use `FormAnalysisBridge` metadata to infer:
      - residual rows from test fields
      - dependencies from state fields and trial fields
      - non-field dependencies from auxiliary, parameter, global scalar,
        boundary-functional, boundary-integral, and material-state terminals
      - expected blocks from installed block metadata
      - temporal requirements from time-derivative and previous-solution
        terminals
      - geometry requirements from geometry terminals
      - geometry-sensitivity field uses from install options and bridge metadata
- [x] Add a policy enum for dependency declaration mode:

```cpp
enum class CouplingDependencyDeclarationMode {
    InferFromInstalledForms,
    DeclareAndVerify,
    ExpertProvided,
};
```

- [x] Default new physics modules to `InferFromInstalledForms`.
- [x] Keep current explicit `dependencies` and `expected_blocks` as
      compatibility and expert-check paths.
- [ ] Add diagnostics when explicit declarations disagree with installed Forms
      metadata.

### Completion Checklist

- [x] FSI no longer manually appends monolithic dependencies.
- [x] FSI no longer manually appends expected blocks.
- [x] Temporal requirements from `dt(...)` can be inferred or verified.
- [x] Geometry terminal requirements from `normal()` can be inferred or
      verified.
- [ ] Non-field variable dependencies from Forms terminals can be inferred or
      verified.
- [ ] Coupling-owned additional fields participate in dependency inference.
- [ ] Existing bridge feature gates are reported through `CouplingGraph`.
- [ ] Missing bridge metadata produces actionable diagnostics rather than
      silent acceptance.

## Phase 4: Add Partitioned Exchange Convenience Helpers

### Rationale

Partitioned coupling should remain driven by `CouplingExchangeDeclaration` and
`PartitionedCouplingPlanGenerator`, but physics modules should author logical
channels instead of endpoint mechanics.

### Concrete Changes

- [x] Add a small `PartitionedCouplingBuilder` that creates
      `CouplingExchangeDeclaration` records.
- [x] Support:

```cpp
exchange(name, producer_field, consumer_field)
    .sharedInterface(interface)
    .value(value_descriptor)
    .transfer(transfer_declaration)
    .producerTemporal(slot)
    .consumerTemporal(slot);
```

- [x] Support non-field endpoints:

```cpp
exchange(name, producer_endpoint, consumer_endpoint)
    .regionRelation(relation)
    .value(value_descriptor)
    .transfer(transfer_declaration);
```

- [x] Infer `CouplingValueDescriptor` from field requirements when possible.
- [ ] Require explicit value descriptors only for mixed-block, packed tensor, or
      driver-owned payloads.
- [ ] Support exchanges involving field endpoints, region data, auxiliary
      state/input/output, parameters, boundary functionals, external buffers,
      global scalars, and driver-owned buffers.
- [x] Support N-way exchange groups for junctions and hub-and-spoke couplings.
- [x] Add partitioned strategy metadata for explicit lagging, staggered
      fixed-point, relaxation, convergence norms, subcycling, and time-window
      exchange without encoding those algorithms in Forms.
- [x] Move shared-region endpoint attachment into FE/Coupling preprocessing or
      `PartitionedCouplingPlanGenerator`.
- [ ] Keep transfer validation in `PartitionedCouplingPlanGenerator`.
- [x] Keep endpoint scope, temporal slot, frame transform, component layout, and
      runtime handle validation in `PartitionedCouplingPlanGenerator`.

### Completion Checklist

- [x] FSI partitioned exchange definitions are reduced to logical channels and
      transfer choices.
- [ ] Thermal partitioned exchange definitions are reduced to logical channels
      and transfer choices.
- [x] Physics modules do not manually attach producer/consumer region endpoints.
- [ ] Multi-PDE auxiliary and global-scalar exchanges are expressible through
      existing endpoint records.
- [ ] N-way partitioned exchange groups validate through the existing plan
      generator.
- [x] Existing partitioned plan tests continue to validate resolved exchanges.
- [x] New tests verify inferred value descriptors from scalar and vector fields.

## Phase 5: Add A Thin Definition-Backed Contract Adapter

### Rationale

The current `CouplingContract` API is a good backend interface, but it is too
low-level for normal physics module authoring. Add one thin adapter that lets
new modules implement `define(...)` while still using the current backend.

### Concrete Changes

- [x] Add a `DefinitionBackedCouplingContract` class under FE/Coupling.
- [x] Add:

```cpp
class DefinitionBackedCouplingContract : public CouplingContract {
public:
    [[nodiscard]] CouplingContractDeclaration declare() const override;
    void validate(const CouplingContext& ctx) const override;
    [[nodiscard]] bool supportsMonolithicLowering() const override;
    [[nodiscard]] bool supportsPartitionedLowering() const override;
    [[nodiscard]] std::vector<CouplingFormContribution>
    buildMonolithicForms(const CouplingContext& ctx,
                         const CouplingFormBuilder& forms) const override;
    [[nodiscard]] std::vector<CouplingExchangeDeclaration>
    buildPartitionedExchangeDeclarations(const CouplingContext& ctx) const override;

protected:
    virtual void define(CouplingDefinitionBuilder& builder) const = 0;
};
```

- [x] Implement the adapter by compiling a `CouplingDefinitionBuilder` result
      into existing `CouplingContractDeclaration`, `CouplingFormContribution`,
      and `CouplingExchangeDeclaration` records.
- [x] Keep direct `CouplingContract` overrides available for unusual contracts.
- [x] Keep `CouplingRegistry` factories returning `CouplingContract` pointers.

### Completion Checklist

- [x] New physics couplings can implement only `name()` and `define(...)`.
- [x] Existing low-level contracts still compile unchanged.
- [x] `CouplingRegistry` does not need a second registration path.
- [x] The adapter delegates validation to `CouplingGraph` and
      `PartitionedCouplingPlanGenerator`.

## Phase 6: Add Relation Lowering Capability Diagnostics

### Rationale

The authoring facade should reject unsupported strategy combinations before
lowering. This keeps physics modules clean: they declare relation law and
strategy, while FE/Coupling decides whether that combination is supported.

### Concrete Changes

- [x] Add the relation lowering capability declarations described in "Resolved
      Authoring Decisions".
- [ ] Add capability records for the initial FSI and thermal relations.
- [ ] Add capability records for fixture relations: N-way conservation,
      multiplier, multi-PDE auxiliary, electro-thermal, and contact/friction.
- [x] Add validation that checks selected `CouplingMode`, enforcement strategy,
      partitioned strategy, and expert fallback options against the relation
      lowering capabilities.
- [x] Add diagnostics that identify relation name, relation kind, selected
      strategy, missing capability, and available capabilities.

### Completion Checklist

- [x] Unsupported monolithic lowering fails before Forms installation.
- [x] Unsupported partitioned lowering fails before plan generation.
- [x] Expert fallback is rejected unless explicitly selected.
- [ ] Physics modules do not hand-code relation capability branching.

## Phase 7: Migrate FSI Coupling

### Rationale

`FSICouplingModule.cpp` already uses Forms for monolithic coupling, but it also
contains substantial generic validation and declaration plumbing. After the
backend work above, the module should become mostly physical role declarations,
equation definitions, and partitioned strategy choices.

### Concrete Changes

- [x] Derive `FSICouplingModule` from `DefinitionBackedCouplingContract`.
- [x] Replace `declare()`, `supportsMonolithicLowering()`,
      `buildMonolithicForms()`, and `buildPartitionedExchangeDeclarations()`
      with `define(...)`.
- [ ] Move remaining option-specific `validate()` checks into a
      definition-backed validation hook once that hook exists.
- [x] Declare participant roles:
      - fluid
      - solid
      - optional mesh
- [x] Declare field roles:
      - fluid velocity as interface vector
      - fluid pressure as scalar
      - solid displacement as interface vector
      - optional solid velocity as interface vector
      - optional mesh displacement as interface vector
- [x] Define the monolithic velocity-continuity residual with interface-view
      Forms.
- [x] Define the monolithic pressure-traction residual with interface-view
      Forms.
- [x] Preserve optional solid velocity versus time derivative of solid
      displacement.
- [x] Preserve optional mesh-displacement geometry sensitivity.
- [x] Define partitioned channels through `PartitionedCouplingBuilder`.
- [x] Remove local helpers that duplicate backend responsibilities:
      - [x] `validateVectorFieldComponents`
      - [x] `validateScalarPressureComponents`
      - [x] `validateInterfaceRegionMappings`
      - [x] `validateMonolithicFieldSystems`
      - [x] `validateMonolithicInterfaceTopology`
      - [x] manual partitioned endpoint attachment
      - [x] manual monolithic dependency appending
      - [x] manual expected-block appending

### Completion Checklist

- [x] FSI behavior is unchanged in existing tests.
- [x] FSI file length is substantially reduced.
- [x] FSI equations remain easy to review in one location.
- [ ] FSI validation errors still identify contract, participant, field, and
      region context.

## Phase 8: Add General Coupling Fixtures Before Broad Migration

### Rationale

FSI and thermal should not be the only proof points for the authoring facade.
Before declaring the API general, add small fixtures that exercise relation
families future modules are likely to need.

### Concrete Changes

- [x] Add a scalar N-way conservation fixture.
- [x] Add a multiplier-enforced interface equality fixture.
- [x] Add a penalty or Nitsche scalar interface fixture using existing
      FE/Forms interface helpers.
- [x] Add a multi-PDE auxiliary-state fixture through non-field or global scalar
      variables.
- [x] Add a mixed-dimensional boundary-functional fixture.
- [x] Add a moving-interface geometry terminal fixture.
- [x] Add a partitioned fixed-point exchange fixture with relaxation metadata.

### Completion Checklist

- [x] Each fixture is expressible through the definition-backed path or has a
      documented expert-path reason.
- [x] Each monolithic fixture lowers through Forms where the coupling is a
      residual.
- [ ] Each partitioned fixture lowers to existing exchange declarations and
      plan-generator validation.
- [ ] Each fixture produces graph diagnostics from backend metadata rather than
      custom physics-local checks.

## Phase 9: Migrate Thermal Interface Coupling

### Rationale

`ThermalInterfaceCouplingModule.cpp` currently declares participants, fields,
shared-region usage, and partitioned exchanges, but it does not provide
monolithic Forms lowering even though its default mode is monolithic. This
migration should fix that mismatch and make thermal a clean example of the new
authoring path.

### Concrete Changes

- [x] Derive `ThermalInterfaceCouplingModule` from
      `DefinitionBackedCouplingContract`.
- [x] Replace direct declaration code with `define(...)`.
- [ ] Move remaining option-specific `validate()` checks into a
      definition-backed validation hook once that hook exists.
- [x] Declare participant roles:
      - side A
      - side B
- [x] Declare field roles:
      - side A temperature as scalar by default
      - side B temperature as scalar by default
      - side A heat flux as scalar or vector depending on selected strategy
      - side B heat flux as scalar or vector depending on selected strategy
- [x] Add an explicit thermal interface formulation option:

```cpp
enum class ThermalInterfaceFormulation {
    TemperatureContinuityPenalty,
    SymmetricNitscheDiffusion,
    ExplicitFluxBalance,
};
```

- [x] Implement monolithic temperature-continuity Forms for the selected
      formulation.
- [ ] Reuse `FE/Forms/InterfaceConditions.h` and `BoundaryConditions.h` helpers
      for Nitsche-style scalar diffusion when applicable.
- [x] Define partitioned temperature and heat-flux channels through
      `PartitionedCouplingBuilder`.
- [ ] Remove local validation that backend field and partitioned checks now
      cover.

### Completion Checklist

- [x] Thermal default monolithic mode has valid monolithic lowering.
- [x] Thermal partitioned mode behavior is preserved.
- [x] Thermal tests cover scalar temperature continuity.
- [x] Thermal tests cover partitioned temperature and heat-flux exchanges.
- [ ] Thermal examples are short enough to serve as authoring documentation.

## Phase 10: Tests

### FE/Coupling Unit Tests

- [ ] Test field requirement validation for scalar fields.
- [ ] Test field requirement validation for vector fields.
- [ ] Test component-count mismatch diagnostics.
- [ ] Test required participant lookup diagnostics.
- [ ] Test required field lookup diagnostics.
- [ ] Test required shared-region lookup diagnostics.
- [ ] Test missing shared-interface participant diagnostics.
- [ ] Test same-side interface diagnostics.
- [ ] Test N-way interface diagnostics.
- [ ] Test mixed-dimensional region relation diagnostics.
- [ ] Test optional endpoint validation.
- [ ] Test monolithic mixed-system diagnostics.
- [ ] Test missing interface topology diagnostics.
- [ ] Test transform, orientation, component-layout, and frame-policy
      diagnostics.
- [x] Test relation lowering capability validation.
- [x] Test unsupported monolithic capability diagnostics.
- [x] Test unsupported partitioned capability diagnostics.
- [x] Test explicit expert fallback opt-in diagnostics.
- [ ] Test stable generated names for residuals, exchanges, and relation nodes.

### CouplingFormBuilder Tests

- [ ] Test interface-side `state(...)` maps to the expected side restriction.
- [ ] Test interface-side `test(...)` maps to the expected side restriction.
- [ ] Test interface-side `dt(...)` records temporal provenance.
- [ ] Test interface-side `normal()` records geometry terminal provenance.
- [ ] Test interface `integral(...)` uses the shared-region integration path.
- [ ] Test missing side mappings produce clear diagnostics.
- [ ] Test generic relation endpoints for N-way residual authoring.
- [ ] Test mixed-dimensional relation integrals select the expected domain.
- [x] Test normal and tangential projection helpers lower to Forms expressions.

### Metadata Inference Tests

- [ ] Test residual-row inference from test fields.
- [ ] Test dependency inference from state fields.
- [x] Test expected-block inference from installed block metadata.
- [x] Test temporal requirement inference from time derivatives.
- [x] Test geometry requirement inference from normal terminals.
- [ ] Test geometry-sensitivity field-use inference.
- [ ] Test non-field dependency inference.
- [ ] Test global scalar dependency inference.
- [ ] Test coupling-owned field dependency inference.
- [ ] Test approximate and lagged lowering disclosures appear in diagnostics.
- [ ] Test explicit dependency declarations can still be verified.
- [ ] Test mismatched explicit declarations produce diagnostics.

### Partitioned Tests

- [x] Test scalar field exchange descriptor inference.
- [x] Test vector field exchange descriptor inference.
- [ ] Test explicit descriptor override for non-field payloads.
- [ ] Test automatic shared-region endpoint attachment.
- [ ] Test auxiliary endpoint exchange descriptors.
- [ ] Test global-scalar exchange descriptors.
- [ ] Test N-way exchange group validation.
- [x] Test partitioned relaxation and convergence metadata validation.
- [x] Test subcycling and time-window metadata validation.
- [ ] Test AuxiliaryState endpoints admitted into a partitioned coupling inherit
      the coupling contract partitioned strategy and temporal policy.
- [ ] Test transfer-kind validation remains in `PartitionedCouplingPlanGenerator`.
- [ ] Test temporal slot validation remains in `PartitionedCouplingPlanGenerator`.
- [ ] Test interface runtime handle validation remains unchanged.

### Physics Tests

- [ ] Test migrated FSI declarations match current participant, field, and
      shared-region requirements.
- [ ] Test migrated FSI monolithic forms install the same contribution names and
      dependencies.
- [ ] Test migrated FSI partitioned exchanges match current logical channels.
- [ ] Test migrated thermal monolithic lowering.
- [ ] Test migrated thermal partitioned exchanges.
- [x] Test at least one N-way conservation coupling.
- [x] Test at least one multiplier coupling.
- [x] Test at least one multi-PDE auxiliary/global-scalar coupling.
- [ ] Test multi-PDE auxiliary coupling in both monolithic and partitioned
      strategies.
- [ ] Test that a mismatched AuxiliaryState strategy inside one coupling
      contract is rejected unless an explicit expert strategy is declared.
- [x] Test at least one mixed-dimensional coupling.
- [x] Test at least one moving-interface coupling.
- [ ] Keep existing `test_fe_coupling` and `test_physics` coverage green.

## Phase 11: Documentation

- [ ] Add a short "Writing A Coupling Module" guide.
- [ ] Document the preferred `define(builder)` path.
- [ ] Document when direct `CouplingContract` overrides are appropriate.
- [ ] Document how interface views map to `minus()` and `plus()`.
- [ ] Document how monolithic Forms metadata drives dependencies and expected
      blocks.
- [ ] Document how partitioned channels map to exchange declarations and plans.
- [ ] Document when coupling math should be authored as Forms and when it
      should be authored as exchanges, variables, or expert hooks.
- [ ] Document relation families: equality, conservation, penalty, Nitsche,
      multiplier, mortar, contact, friction, multi-PDE auxiliary, and
      global-scalar coupling.
- [ ] Document relation lowering capabilities and unsupported-strategy
      diagnostics.
- [ ] Document stable generated naming rules.
- [ ] Document exact, approximate, lagged, and unavailable lowerings.
- [ ] Document the layered configuration surface: default, advanced, expert.
- [ ] Document N-participant, mixed-dimensional, and optional-participant
      patterns.
- [ ] Document transform, orientation, frame, and geometry lifecycle policies.
- [ ] Include compact FSI and thermal examples.
- [ ] Update `Documentation/plan_coupling_module_infrastructure.md` to point to
      this follow-on authoring migration plan.

## Review Gates

The migration is complete when:

- [ ] Physics coupling files primarily contain physical roles, equations, and
      high-level strategy options.
- [ ] Generic validation lives in FE/Coupling.
- [ ] Monolithic dependency and expected-block evidence comes from installed
      Forms metadata by default.
- [ ] Partitioned exchange validation and plan generation remain in
      `PartitionedCouplingPlanGenerator`.
- [ ] Relation capability validation rejects unsupported lowerings before
      installation or execution.
- [ ] Stable generated names appear in residual, exchange, relation, diagnostic,
      and metadata records.
- [ ] Exact, approximate, lagged, and unavailable lowerings are visible through
      relation diagnostics.
- [ ] Default physics configuration stays short, with advanced and expert
      options isolated.
- [ ] FSI and thermal modules no longer duplicate field, shared-region,
      topology, same-system, or endpoint-region checks.
- [ ] At least one N-participant coupling, one mixed-dimensional coupling, one
      multiplier coupling, and one multi-PDE auxiliary or global-variable
      coupling are represented by the authoring facade.
- [ ] AuxiliaryState models admitted into a coupling contract inherit the
      selected monolithic or partitioned strategy for that contract.
- [ ] Mixed AuxiliaryState/PDE coupling strategies inside one contract are
      rejected unless represented by an explicit expert strategy with complete
      diagnostics.
- [ ] Coupling math that belongs in a monolithic residual is expressible through
      Forms unless a documented public Forms gap requires an expert hook.
- [ ] Partitioned driver algorithms are represented as strategy metadata and do
      not leak into Forms residual authoring.
- [ ] Direct low-level `CouplingContract` authoring remains available for expert
      contracts.
- [ ] No duplicate FE registry, graph, form scanner, installer, or partitioned
      planner has been introduced.

## Suggested Implementation Order

1. Add field-shape and shared-interface requirement records.
2. Add general region-relation requirement records.
3. Add stable generated naming rules.
4. Add relation lowering capability diagnostics.
5. Move generic validation into `CouplingGraph`.
6. Extend `CouplingFormBuilder` with interface and relation views.
7. Add metadata-driven monolithic declaration inference.
8. Add partitioned exchange convenience helpers.
9. Add `DefinitionBackedCouplingContract`.
10. Add broad coupling fixtures for N-way, mixed-dimensional, multiplier, and
   multi-PDE auxiliary or global-variable cases.
11. Migrate FSI.
12. Migrate thermal and add monolithic thermal Forms.
13. Remove obsolete physics-local validation helpers.
14. Add documentation and keep tests green at each step.
