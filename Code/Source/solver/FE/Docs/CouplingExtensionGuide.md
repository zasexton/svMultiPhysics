# FE Coupling Extension Guide

This guide lists the public extension points used to add a new coupling
contract. Coupling code should keep physics-specific names and options inside
the contract layer, while `FE/Coupling` receives only participant, field,
region, value, transfer, and form metadata.

## Contract Declaration

Add a contract type by deriving from `coupling::CouplingContract` and returning
a `coupling::CouplingContractDeclaration` from `declare()`.

Required declaration records:

- `contract_type` and `contract_name` identify the contract family and instance.
- `participants`, `fields`, `regions`, and `shared_regions` describe the FE
  resources the contract needs.
- `dependencies` and `expected_blocks` describe monolithic graph structure.
- `temporal_requirements` and `geometry_requirements` describe setup-time data
  needs before any form is installed.
- `exchanges` and `group_hints` describe partitioned metadata when the contract
  supports partitioned mode.

Use stable contract instance names in all `CouplingPortId` records. Field names,
region names, endpoint names, and port names are contract-owned labels and are
resolved only through `CouplingContext`.

## Validation

Implement `validate(const CouplingContext&)` for contract-local rules that the
generic graph cannot infer. Examples include option combinations, component
counts with physical meaning, and mode-specific requirements.

Generic validation is handled by:

- `CouplingContextBuilder::build()` for participant, field, region, shared
  region, external-buffer, and driver-owned transfer registries.
- `CouplingGraph::buildDeclarationGraph()` for declaration-only topology.
- `CouplingGraph::buildFinalizedGraph()` for installed dependency, block,
  temporal, and geometry-terminal metadata.
- `PartitionedCouplingPlanGenerator::generate()` for resolved partitioned
  endpoint and transfer metadata.

## Additional Fields

Declare coupling-created FE fields with
`coupling::CouplingAdditionalFieldDeclaration`.

Use `CouplingAdditionalFieldNamespace::Participant` when the field should be
owned by one participant namespace. Use
`CouplingAdditionalFieldNamespace::Contract` when the field belongs to one
contract instance. Interface fields must also declare their interface region or
shared-region scope and their registration target policy through the declaration
record.

The monolithic builder registers additional fields before final graph
validation, refreshes the `CouplingContext`, and then checks that all field uses
resolve to concrete `FESystem` fields.

## Monolithic Forms

Prefer Forms-authored monolithic contributions through
`coupling::CouplingFormContribution`.

Public authoring entry points:

- `CouplingFormBuilder::state()`, `test()`, `timeDerivative()`, and
  `previousSolution()` for field terminals.
- `CouplingFormBuilder::time()`, `timeStep()`, and `effectiveTimeStep()` for
  global temporal terminals.
- `CouplingFormBuilder::meshTemporal()` and geometry terminal helpers for
  mesh-motion and geometry requirements.
- `CouplingFormContribution::field_uses` and `extra_trial_field_uses` for
  residual rows and additional tangent columns.
- `CouplingFormInstallOptionsDeclaration` for declaration-time AD mode,
  compiler options, and geometry-sensitivity intent.
- `systems::FormInstallOptions` for resolved install options forwarded to
  `systems::installFormulation()`.

Installed form metadata must come from the public Forms/Systems analysis bridge.
Graph validation compares declaration-side requirements against installed
`CouplingFormAnalysisMetadata`, including dependency rows, dependency columns,
domains, blocks, temporal terminals, geometry terminals, and geometry
sensitivity provenance.

## Expert Monolithic Hooks

Use `installMonolithicTerms()` only when the coupling cannot be expressed with
Forms. Expert hooks must install through approved `FESystem` extension points
and return resolved `coupling::CouplingInstallMetadata`.

Required metadata:

- `contribution_name`, `origin`, `system_name`, and `operator_name`.
- `installed_dependencies` with resolved `analysis::VariableKey` rows and
  dependencies, `analysis::DomainKind`, matrix/vector evidence, and provider.
- `installed_blocks` with resolved row/dependency pairs, contributing domains,
  and matrix/vector flags.

Declaration-shaped metadata is not sufficient for expert hooks because graph
validation must compare the contract declaration with concrete installed FE
state.

## Partitioned Plans

Declare partitioned exchanges with `coupling::CouplingExchangeDeclaration`.

Each exchange should provide:

- Stable producer and consumer `CouplingPortId` values.
- Producer and consumer `CouplingEndpointRef` records.
- A `CouplingValueDescriptor` with rank, component count, component layout, and
  tensor packing when needed.
- Region or shared-region scope for interface or region-data transfers.
- An explicit `CouplingTransferDeclaration`; `Unspecified` is a declaration
  default and must not reach a valid plan.

Supported endpoint kinds include FE fields, region data, auxiliary state,
auxiliary input, auxiliary output, parameters, and external buffers. Resolved
plans retain durable endpoint identity, owning system provenance, temporal
backing, runtime transfer handles when available, and driver-owned transfer
descriptors.

## Tests

Add tests for every new contract path:

- Context tests for participant, field, region, shared-region, external-buffer,
  and driver-owned transfer setup.
- Graph tests for declarations, dependencies, blocks, temporal requirements, and
  geometry requirements.
- Monolithic builder tests for additional fields, Forms contributions, expert
  hooks, install metadata, and setup ordering.
- Partitioned plan tests for exchange resolution, endpoint identity, temporal
  backing, value compatibility, transfer options, interface-map provenance, and
  cycles.
- Physics-module tests that keep physical vocabulary inside the physics module
  and verify that FE/Coupling receives only physics-agnostic declarations.

Use `Code/Source/solver/FE/Tests/Unit/Coupling/CouplingTestHelpers.h` for
synthetic participant bindings, contract names, endpoints, temporal slots,
transfer descriptors, geometry provenance, form install options, expert
metadata, and value descriptors.
