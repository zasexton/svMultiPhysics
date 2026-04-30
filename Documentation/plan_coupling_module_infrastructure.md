# Coupling Module Infrastructure Plan

## Goal

Establish a coupling-module infrastructure for the new OOP solver that supports
both monolithic and partitioned multiphysics workflows, including
N-participant coupling contracts.

The design should keep physics-agnostic data-flow and formulation construction
support inside the FE library, while placing physics-specific coupling contracts
inside `Physics/Coupling`.

The intended split is:

```text
FE/Coupling:
  Physics-agnostic participant/endpoint vocabulary, field lookup, validation
  helpers, monolithic residual-install hooks, partitioned data-flow plans,
  transfer declarations, resolved transfer options, and coupling-plan
  construction.

Physics/Coupling:
  Physics-specific coupling contracts such as FSI, thermal-fluid,
  thermo-elasticity, electro-thermal coupling, and related options.
```

## Proposed Directory Layout

```text
Code/Source/solver/FE/Coupling/
  CouplingTypes.h
  CouplingTypes.cpp
  CouplingContext.h
  CouplingContext.cpp
  CouplingDeclaration.h
  CouplingDeclaration.cpp
  CouplingContract.h
  CouplingContract.cpp
  CouplingFormBuilder.h
  CouplingFormBuilder.cpp
  CouplingGraph.h
  CouplingGraph.cpp
  CouplingTemporalRequirements.h
  CouplingTemporalRequirements.cpp
  CouplingGeometryRequirements.h
  CouplingGeometryRequirements.cpp
  MonolithicCouplingBuilder.h
  MonolithicCouplingBuilder.cpp
  PartitionedCouplingPlan.h
  PartitionedCouplingPlan.cpp
  PartitionedCouplingPlanGenerator.h
  PartitionedCouplingPlanGenerator.cpp
  CouplingRegistry.h
  CouplingRegistry.cpp
  CouplingDiagnostics.h
  CouplingDiagnostics.cpp
  SharedRegionRegistry.h
  SharedRegionRegistry.cpp
  TransferPlan.h
  TransferPlan.cpp

Code/Source/solver/Physics/Coupling/
  FSICouplingModule.h
  FSICouplingModule.cpp
  ThermalInterfaceCouplingModule.h
  ThermalInterfaceCouplingModule.cpp
```

Small value-only helpers may remain header-only when that matches existing FE
library style, but every non-trivial validator, registry, builder, graph, plan
generator, and diagnostics surface should have an owned `.cpp` implementation and
an explicit build-system entry.

The FE layer should not know what FSI, thermal-fluid coupling, or
thermo-elasticity mean physically. It should only provide the common mechanics
needed to connect participants, fields, regions, residual terms, exchange
graphs, transfer declarations, and resolved transfer options.

## Core Concepts

### Terminology Boundary

FE/Coupling should use only generic terms:

```text
participant
field
region
shared region
port
endpoint
exchange
value descriptor
transfer declaration
resolved transfer options
contract declaration
coupling graph
form builder
dependency
block diagnostic
residual contribution
temporal requirement
time derivative
```

Physics/Coupling should own physical terms:

```text
fluid
solid
thermal
electric
displacement
velocity
traction
pressure
temperature
heat flux
ALE mesh-motion model choices
```

If a term describes the physical meaning of a field or exchanged value, it
belongs in `Physics/Coupling` or higher-level problem input, not in
`FE/Coupling`.
Existing FE/Forms and FE/Systems geometry vocabulary already includes
`meshDisplacement()`, `meshVelocity()`, mesh-motion geometry-sensitivity
options, and mesh-motion role/provider bindings. `FE/Coupling` may use those
terms only for the physics-agnostic geometry/temporal infrastructure; the
physical interpretation of an ALE or FSI mesh-motion model remains in
`Physics/Coupling`.

### Alignment With Current FE/Forms Infrastructure

The coupling infrastructure must use the current public FE authoring path
rather than introducing a parallel formulation language.

Required alignment rules:

```text
Forms authoring:
  use FE/Forms/Vocabulary.h field-bound StateField/TestField helpers
  use symbolic dt(u,k), t(), deltat(), and deltat_eff() terminals
  express interface terms with existing Forms primitives and measures such as
  .ds(marker), .dS(), and .dI(interface_marker)

Forms installation:
  install Forms-authored coupling residuals through
  FE/Systems/FormsInstaller.h installFormulation()
  rely on public Forms/Systems metadata for installed residual rows, active
  dependencies, sparsity, temporal symbols, and block diagnostics

Module boundaries:
  FE/Coupling may adapt participant/field/region names to Forms symbols
  FE/Coupling must not parse private Forms AST internals directly
  FE/Coupling must not duplicate DOF, assembly, time-step, or solver ownership
  FE/Coupling must keep expert/custom hooks routed through approved Systems
  extension points and metadata-producing install contexts
```

This keeps `FE/Coupling` as orchestration and validation infrastructure on top
of the existing FE library rather than a second Forms, Systems, or Assembly
layer.

The current `MixedKernelPlan` reports lowered block and domain structure, but
it is not by itself a complete coupling-diagnostics API: it does not expose all
source-level `StateField`, non-field Forms terminal, geometry-sensitivity,
geometry-terminal, or temporal-symbol provenance needed by the coupling graph.
The coupling implementation must therefore first
add one public metadata bridge, either by extending `FormsInstaller` /
`MixedKernelPlan` output or by adding a public Forms/Systems analysis helper
with equivalent semantics. The bridge should expose the
`CouplingFormAnalysisMetadata` shape described below: installed field order,
contribution name/origin and owning-system provenance, field-use provenance
including geometry-sensitivity field-use summaries, installed
geometry-sensitivity options including the mesh-motion field, structured
geometry-sensitivity provenance, non-field dependency provenance for parameter,
coefficient, boundary-functional, boundary-integral, auxiliary-state,
`AuxiliaryInput`, `AuxiliaryOutput`, and material-state terminals,
monolithic variable-dependency provenance using scoped `analysis::VariableKey`
names, including Analysis/expert `GlobalScalar` variable dependencies,
declaration terminal-provenance records captured from `CouplingFormBuilder`,
temporal-symbol provenance including
field/trial-scoped `PreviousSolutionRef(k)` and mesh temporal owner-scope plus
`systems::MeshMotionFieldRole` provenance, geometry-terminal provenance with
integration-location, resolved `analysis::DomainKind`, frame-transform
configuration, and geometry-revision metadata, installed dependency provenance
with domain/provider/matrix/vector evidence, and installed block/domain
provenance.
The lower-level Forms/Systems bridge may use native Forms/Systems types;
`FE/Coupling` should adapt those records into `CouplingFormAnalysisMetadata`
without requiring Forms or Systems to depend on `FE/Coupling`.
The bridge should reuse the existing analysis infrastructure where possible:
`FE/Analysis/ContributionDescriptor` is the primary normalized contribution
metadata, with `FormulationRecord` retained as a fallback/source artifact. The
coupling plan must not introduce another independent contribution-analysis IR
for blocks, temporal metadata, or cross-field coupling when the existing
analysis records can be extended or adapted.

The first implementation must treat the current Forms scanner and analysis
records as partial inputs, not as complete coupling metadata. In particular,
scanner fields whose names currently reflect only one consumer view, such as a
boundary-integral terminal being accumulated into boundary-functional-style
names, must be normalized behind the public bridge before `FE/Coupling` consumes
them. Coupling diagnostics should depend on explicit terminal kind, provider
identity, graph-variable identity when one exists, and location/provenance
metadata, not on scanner container names.

### Coupling Mode

Coupling modules should be able to lower the same physical coupling contract in
different ways:

```cpp
enum class CouplingMode {
    Monolithic,
    Partitioned
};
```

In monolithic mode, the coupler installs residual terms into a shared
`FESystem`.

In partitioned mode, the coupler produces a data-flow plan that a higher-level
partitioned driver executes.

### Coupling Ports and Channels

The FE layer should not define named physical quantities such as displacement,
velocity, traction, pressure, heat flux, or mesh velocity. Those names are
physics-specific and belong in `Physics/Coupling`.

Instead, the FE layer should only provide opaque data-flow identifiers and
shape/location metadata:

```cpp
struct CouplingPortId {
    // Configured contract instance namespace, not the reusable contract type
    // registered in CouplingRegistry.
    std::string contract_instance_name;
    std::string port_name;      // opaque to FE
};

enum class CouplingValueRank {
    Scalar,
    Vector,
    Rank2Tensor,
    SymmetricTensor,
    MixedBlock,
    GeneralTensor
};

struct CouplingValueDescriptor {
    CouplingValueRank rank = CouplingValueRank::Scalar;
    int components = 1;

    // Optional logical component labels/layout. Required when a frame transform
    // leaves trailing components untouched, or when a MixedBlock payload is
    // packed into a flat component vector.
    std::vector<std::string> component_layout;

    // Only used for GeneralTensor payloads. FE interface transfers ignore this
    // and accept only Scalar, Vector, Rank2Tensor, or MixedBlock.
    std::vector<int> tensor_extents;
    std::string tensor_packing;
};

enum class CouplingFrameSourceEmbeddingPolicy {
    None,
    Embed2DInXY,
    Embed2DInXZ,
    Embed2DInYZ,
    DriverProvided
};

enum class CouplingFrameTargetRestrictionPolicy {
    None,
    RestrictToXY,
    RestrictToXZ,
    RestrictToYZ,
    DriverProvided
};
```

Physics-specific coupling modules may define their own port names. The FE
layer treats those names as opaque labels. It validates that producers and
consumers agree on endpoint ownership, value shape, region scope, and transfer
requirements; it does not interpret the physical meaning of a port label.
For FE-backed endpoints, the payload descriptor should be derived from or
validated against the registered `FieldRecord` and its `FunctionSpace`
signature. For interface transfers, it must be mappable to the existing
`InterfaceFieldKind` categories rather than inventing independent shape rules.
`Scalar`, `Vector`, `Rank2Tensor`, and `MixedBlock` map directly to the existing
interface transfer field kinds. `SymmetricTensor` has no distinct
`InterfaceFieldKind`; an interface exchange must either expand it to an explicit
rank-2 tensor payload with documented packing or reject it during validation.
General rank-N tensors should not be supported by FE interface transfers in the
initial infrastructure because `FE/Systems/InterfaceOperators` only defines
scalar, vector, rank-2 tensor, and mixed-block payload categories. Supporting
general tensors later would require a new interface field kind, explicit extents
and packing metadata, frame-transform semantics, diagnostics, and unit tests.
Until that exists, `GeneralTensor` is only valid with
`CouplingTransferKind::DriverOwned`, because the driver-owned transfer operator
must define the rank-N extents, storage order, and validation contract.
`GeneralTensor` is a value-shape category, not an ownership category; ownership
comes from the transfer kind and resolved endpoint metadata. For
`GeneralTensor`, `tensor_extents` must be nonempty with all extents positive,
`components` must equal the product of those extents, and `tensor_packing` must
name the driver-defined storage order. For every other value rank,
`tensor_extents` and `tensor_packing` must be empty. When `component_layout` is
present, its size must match `components`; it is optional for ordinary
homogeneous scalar/vector/rank-2 payloads and required for mixed payloads or any
payload with pass-through components.

Interface frame transforms must also follow the existing
`systems::InterfaceTransferOptions` vocabulary. `SourceToTargetVector` is valid
only for vector payloads with at least the three components currently transformed
by `InterfaceOperators`; any extra components are pass-through metadata only
when `component_layout` documents their logical meaning and ordering. A true 2D
vector requires a non-`None` `CouplingFrameSourceEmbeddingPolicy` and a
non-`None` `CouplingFrameTargetRestrictionPolicy` that document how the source
vector is embedded into the 3D rotation space and how the transformed target
value is restricted back to the endpoint layout before frame transforms are
accepted. Source embedding and target restriction are intentionally separate
because an exchange may embed one 2D basis and restrict to a different endpoint
layout.
`SourceToTargetRank2Tensor` is valid only for rank-2 tensor payloads with at
least the nine components currently transformed by `InterfaceOperators`; extra
components are pass-through metadata only when `component_layout` documents
their logical meaning and ordering. `MixedBlock` payloads must provide
`component_layout` whenever they are represented as a flat component vector.
`Scalar` and `MixedBlock` payloads must use `None`, and `SymmetricTensor` must
first be expanded to a documented rank-2 tensor packing before any rank-2 frame
transform is allowed.

## FE Coupling Context

Add a physics-agnostic context object that maps participant-level names to
registered FE fields and regions.

Example:

```cpp
struct CouplingParticipantRef {
    std::string participant_name;
    std::string system_name;

    // Non-owning. FieldIds, region markers, and interface markers are only
    // meaningful relative to this FESystem.
    const systems::FESystem* system = nullptr;
};

struct CouplingFieldRef {
    std::string participant_name;
    std::string system_name;
    const systems::FESystem* system = nullptr;
    std::string field_name;
    FieldId field_id;
    std::shared_ptr<const spaces::FunctionSpace> space;
    int components = 1;
    systems::FieldScope scope = systems::FieldScope::VolumeCell;
    int interface_marker = -1;
};

enum class CouplingRegionKind {
    Domain,
    Boundary,
    InteriorFace,
    InterfaceFace,
    UserDefined
};

enum class CouplingInterfaceSide {
    None,
    Minus,
    Plus
};

enum class CouplingCoordinateConfiguration {
    Reference,
    Current
};

struct CouplingRegionRef {
    std::string participant_name;
    std::string system_name;
    const systems::FESystem* system = nullptr;
    std::string region_name;
    CouplingRegionKind kind = CouplingRegionKind::UserDefined;
    int marker = -1;
    CouplingInterfaceSide side = CouplingInterfaceSide::None;
    CouplingCoordinateConfiguration coordinate_configuration =
        CouplingCoordinateConfiguration::Reference;
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    std::optional<svmp::search::LogicalInterfaceRegionId> logical_region;
    std::optional<svmp::search::InterfaceRevisionSnapshot> revision_snapshot;
#endif
    std::uint64_t geometry_revision = 0;
    std::uint64_t topology_revision = 0;
};

struct SharedRegionRef {
    std::string name;
    std::optional<CouplingRegionKind> required_region_kind;
    std::vector<CouplingRegionRef> participant_regions;
};
```

`CouplingCoordinateConfiguration` is the coupling declaration type. When mesh
support is enabled, interface-map resolution must convert it explicitly to the
Mesh/Search `svmp::Configuration` used by `InterfaceSideSpec` and
`CouplingInterfaceMapProvenance`. The resolver should reject unknown or
unsupported mappings instead of relying on enum-value coincidence, and should
record both the declaration configuration and the resolved Mesh/Search
configuration in diagnostics when they differ by policy.

The context should provide lookup and validation helpers:

```cpp
struct CouplingExternalBufferDescriptor;
struct CouplingDriverOwnedTransferDescriptor;
struct CouplingInterfaceMapProvenance;
struct CouplingInterfaceMapRuntimeHandles;

class CouplingContext {
public:
    CouplingParticipantRef participant(std::string_view participant) const;

    CouplingFieldRef field(std::string_view participant,
                           std::string_view field) const;

    CouplingRegionRef region(std::string_view participant,
                             std::string_view region) const;

    CouplingRegionRef sharedRegion(std::string_view name,
                                   std::string_view participant) const;

    SharedRegionRef sharedRegionGroup(std::string_view name) const;

    bool hasParticipant(std::string_view participant) const;

    bool hasField(std::string_view participant,
                  std::string_view field) const;

    const CouplingExternalBufferDescriptor* externalBufferDescriptor(
        std::optional<std::string_view> participant,
        std::string_view external_buffer_key) const;

    const CouplingDriverOwnedTransferDescriptor* driverOwnedTransfer(
        std::string_view transfer_name) const;

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    CouplingInterfaceMapRuntimeHandles interfaceMapHandles(
        const CouplingInterfaceMapProvenance& provenance) const;

    const svmp::search::InterfaceSearchRegistry* interfaceSearchRegistry(
        std::string_view interface_search_registry_name) const;
#endif
};
```

Example contents:

```text
participant = "A"
  field "primary" -> a
  region "coupling_surface" -> marker 12

participant = "B"
  field "primary" -> b
  region "coupling_surface" -> marker 7
```

This context is the main bridge between independent physics modules and
coupling modules. It must preserve participant-to-`FESystem` ownership because
`FieldId`s, region markers, interface markers, and interface topology are local
to a specific FE system. Monolithic Forms contributions may only combine fields
whose participant references resolve to the same `FESystem`. Partitioned plans
may connect different `FESystem` instances, but every FE-backed endpoint,
region, interface map, and transfer option must carry enough system/topology
provenance to be resolved without treating raw `FieldId`s or markers as global
identifiers.
When mesh support is enabled, `CouplingContext` must also expose interface-map
lookup through the existing Mesh/Search and Systems interface infrastructure.
It should resolve durable `CouplingInterfaceMapProvenance` into execution-only
`CouplingInterfaceMapRuntimeHandles` without making raw pointers part of plan
identity, and it should preserve `InterfaceSideSpec`,
`LogicalInterfaceRegionId`, `InterfaceRevisionSnapshot`, map state,
sliding-map kind, source/target coordinate configuration, and search-registry
identity instead of reducing interface provenance to marker integers.
`CouplingRegionRef` must preserve the same kind of durable region provenance
for ordinary region lookup: participant/system ownership, marker, side,
coupling-neutral coordinate configuration, logical interface identity when mesh
support is enabled, geometry/topology revision keys, and any available interface
revision snapshot. Marker plus side is not enough to validate stale interface
topology or moving-geometry use. Forms-specific
`forms::GeometryConfiguration` values remain terminal/frame-transform metadata
and are introduced only when constructing Forms expressions.
Driver-owned external-buffer descriptors and driver-owned transfer operators
are also context-registered resources. The context or its builder must own the
explicit lookup surfaces for these driver-provided registries so partitioned
plan validation has one deterministic source of truth for external-buffer shape,
temporal-slot support, layout/data revision keys, and named driver-owned
transfer operators.

Shared regions are important for N-multiphysics problems because multiple
contracts may refer to the same geometric relationship. For example, one
contract may install a monolithic interface residual on a shared region while
another contract builds a partitioned exchange on the same shared region. The
FE layer should only ensure that all participants resolve the shared region
consistently; the physical meaning of that region remains contract-owned.
`SharedRegionRef::required_region_kind`, when present, is only a validation
constraint on the participant-region mappings. It is not a resolved FE
integration kind, not a marker, and not a replacement for the
participant-specific `CouplingRegionRef::kind` values. If it is absent, the
consuming Forms assembly, geometry-terminal, or partitioned-transfer operation
must derive the required concrete region kind from its own lowering rules.
Contracts may also declare participant-local region requirements. Those belong
in `CouplingRegionUse` and are validated through `CouplingContext::region(...)`
before Forms authoring, geometry-terminal validation, or partitioned endpoint
resolution. Shared-region requirements belong in `CouplingSharedRegionUse` and
validate the N-participant registry entry plus each participant mapping.

For Forms-authored monolithic residuals, region kinds must map cleanly onto the
implemented Forms measure vocabulary:

```text
CouplingRegionKind::Domain        -> .dx()
CouplingRegionKind::Boundary      -> .ds(marker)
CouplingRegionKind::InteriorFace  -> .dS()
CouplingRegionKind::InterfaceFace -> .dI(interface_marker)
```

`CouplingRegionKind::UserDefined` is a declaration/registry placeholder only.
Before Forms residual authoring, interface transfer resolution, or
geometry-terminal validation, it must resolve through `CouplingContext` to one of
the concrete FE integration kinds above with marker, side, and topology
provenance. If no concrete FE kind is available, it is valid only for an explicit
provider-extension path that declares how it is evaluated; otherwise validation
must fail.

The coupling context should store enough marker and side/orientation metadata
to let a coupling contract author explicit Forms expressions over the correct
measure. It should not hide the weak-form integrand behind an opaque coupling
measure helper. Interface orientation, side restrictions, trace spaces, and
mortar spaces remain FE/Mesh, FE/Spaces, and FE/Forms concerns exposed through
their existing APIs.

Forms-authored monolithic coupling is only valid for fields registered in the
same shared `FESystem` and for regions whose topology is available to that
system. Interface-face coupling through `.dI(interface_marker)` requires a
registered interface topology for that marker, such as an `InterfaceMesh`
registered through the existing `FESystem` interface-mesh APIs. If participants
live on independent meshes or require nonmatching trace/mortar transfer that is
not represented in one `FESystem`, the coupling should be represented as a
partitioned exchange plan or deferred until the required FE/Spaces and
FE/Systems mixed-dimensional infrastructure exists.

## Coupling Contract Interface

Physics-specific coupling modules should implement a common FE-facing contract.
Contracts must be N-participant by default. A two-participant coupling is only
one special case.

The declaration step lets the FE layer build and validate a coupling graph
before any coupling-owned fields, residuals, or exchanges are installed.

```cpp
enum class CouplingTransferKind {
    Unspecified,
    Identity,
    InterfacePointwiseInterpolation,
    InterfaceConservativeProjection,
    InterfaceMortar,
    DriverOwned
};

enum class CouplingInterfaceFramePolicy {
    None,
    SourceToTargetVector,
    SourceToTargetRank2Tensor
};

struct CouplingInterfaceTransferDeclaration {
    CouplingInterfaceFramePolicy frame_policy = CouplingInterfaceFramePolicy::None;
    CouplingFrameSourceEmbeddingPolicy source_embedding_policy =
        CouplingFrameSourceEmbeddingPolicy::None;
    CouplingFrameTargetRestrictionPolicy target_restriction_policy =
        CouplingFrameTargetRestrictionPolicy::None;
    std::array<std::array<Real, 3>, 3> source_to_target_rotation{{
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.0, 0.0, 1.0}}}};
    Real conservation_tolerance = 1.0e-10;
};

struct CouplingTransferDeclaration {
    CouplingTransferKind kind = CouplingTransferKind::Unspecified;

    // Only set for interface transfer kinds. This is stable FE/Coupling
    // declaration metadata, not a Systems resolved options object.
    std::optional<CouplingInterfaceTransferDeclaration> interface_declaration;

    // Only set for DriverOwned transfers.
    std::string driver_owned_name;
};

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
struct CouplingInterfaceMapProvenance {
    std::string interface_map_name;
    std::string interface_entry_name;
    std::string interface_search_registry_name;
    std::string source_system_name;
    std::string target_system_name;
    int source_interface_marker = -1;
    int target_interface_marker = -1;
    systems::SlidingInterfaceMapKind sliding_map_kind =
        systems::SlidingInterfaceMapKind::Sliding;
    svmp::Configuration source_configuration = svmp::Configuration::Reference;
    svmp::Configuration target_configuration = svmp::Configuration::Reference;
    svmp::search::LogicalInterfaceRegionId source_logical_region{};
    svmp::search::LogicalInterfaceRegionId target_logical_region{};
    svmp::search::InterfaceRevisionSnapshot source_revision{};
    svmp::search::InterfaceRevisionSnapshot target_revision{};
    svmp::search::InterfaceMapState interface_map_state =
        svmp::search::InterfaceMapState::Empty;
    std::uint64_t interface_map_revision_key = 0;
    std::uint64_t source_search_revision_key = 0;
    std::uint64_t target_search_revision_key = 0;
    systems::InterfaceOperatorState operator_state =
        systems::InterfaceOperatorState::Empty;
    std::uint64_t accepted_revision_key = 0;
    std::uint64_t trial_revision_key = 0;
    std::uint64_t time_level_epoch = 0;
    Real map_time = 0.0;
};

struct CouplingInterfaceMapRuntimeHandles {
    // Non-owning execution-only handles resolved from CouplingContext. These
    // must not be treated as declaration metadata, serialized plan state, or
    // stable identity.
    const systems::FESystem* source_system = nullptr;
    const systems::FESystem* target_system = nullptr;
    const svmp::search::InterfaceSearchRegistry* search_registry = nullptr;
    const systems::SlidingInterfaceMap* sliding_map = nullptr;
    const svmp::search::InterfaceMap* interface_map = nullptr;
};
#endif

enum class CouplingTemporalSlot {
    Current,
    Accepted,
    Predicted,
    History,
    Stage,
    External
};

enum class CouplingEndpointKind {
    Field,
    RegionData,
    AuxiliaryState,
    AuxiliaryInput,
    AuxiliaryOutput,
    Parameter,
    ExternalBuffer
};

enum class CouplingExternalBufferAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite
};

enum class CouplingExternalBufferDistribution {
    RankLocal,
    DistributedOwned,
    DistributedOwnedGhosted,
    DriverDefined
};

enum class CouplingExternalBufferLifetime {
    StepLocal,
    TimeStepPersistent,
    RestartPersistent,
    DriverDefined
};

struct CouplingTemporalSlotDescriptor {
    CouplingTemporalSlot slot = CouplingTemporalSlot::Current;

    // Logical 1-based step-back index for History slots. For FE field history,
    // history_index == k maps to SystemStateView::u_history[k - 1]. For
    // auxiliary history, it maps to AuxiliaryHistoryBuffer::snapshot(k - 1).
    std::optional<int> history_index;

    // Logical 0-based stage index for Stage slots.
    std::optional<int> stage_index;
};

struct CouplingEndpointRef {
    CouplingEndpointKind kind = CouplingEndpointKind::ExternalBuffer;
    std::optional<std::string> participant_name;

    // Physics-opaque label that resolves through the endpoint-kind registry,
    // provider, or driver-owned registry: field name, parameter key, raw
    // auxiliary block name, AuxiliaryInput name, AuxiliaryOutput name,
    // FE quantity name, BoundaryReductionService functional name,
    // provider-extension key, or external-buffer key.
    std::string endpoint_name;
    CouplingTemporalSlotDescriptor temporal;
};

struct CouplingExternalBufferDescriptor {
    std::string buffer_name;
    std::string scalar_type = "Real";
    CouplingValueDescriptor value;
    CouplingExternalBufferAccess access = CouplingExternalBufferAccess::ReadWrite;
    CouplingExternalBufferDistribution distribution =
        CouplingExternalBufferDistribution::DriverDefined;
    CouplingExternalBufferLifetime lifetime =
        CouplingExternalBufferLifetime::DriverDefined;

    // Flat extents/strides for the driver-visible memory layout. For
    // GeneralTensor payloads these must agree with value.tensor_extents and
    // value.tensor_packing.
    std::vector<std::int64_t> extents;
    std::vector<std::int64_t> strides;
    std::string packing;

    // Temporal slot descriptors the driver-owned registry can actually provide
    // for this buffer. History and Stage entries include the logical
    // history_index or stage_index they support, so indexed endpoint requests
    // can be validated without relying on side-channel conventions.
    std::vector<CouplingTemporalSlotDescriptor> supported_temporal_slots;

    std::uint64_t layout_revision_key = 0;
    std::uint64_t data_revision_key = 0;
};

struct CouplingDriverOwnedTransferDescriptor {
    std::string transfer_name;
    // GeneralTensor support is represented by including GeneralTensor in this
    // list. There is no separate boolean because conflicting rank/capability
    // metadata must be impossible to encode.
    std::vector<CouplingValueRank> supported_ranks;
    bool preserves_component_layout = true;
    std::vector<CouplingTemporalSlotDescriptor> supported_source_temporal_slots;
    std::vector<CouplingTemporalSlotDescriptor> supported_target_temporal_slots;
    std::uint64_t registry_revision_key = 0;
};

struct ResolvedCouplingTransfer {
    CouplingTransferKind kind = CouplingTransferKind::Unspecified;

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    std::optional<systems::InterfaceTransferOptions> interface_options;
    std::optional<CouplingInterfaceMapProvenance> interface_map;
    // Coupling-side pre/post policy for true 2D vector frame transforms. This
    // is not part of systems::InterfaceTransferOptions, so FE/Coupling must
    // preserve and apply it around Systems interface transfer execution, or
    // reject the transfer if no adapter is implemented.
    CouplingFrameSourceEmbeddingPolicy source_embedding_policy =
        CouplingFrameSourceEmbeddingPolicy::None;
    CouplingFrameTargetRestrictionPolicy target_restriction_policy =
        CouplingFrameTargetRestrictionPolicy::None;
#endif

    std::string driver_owned_name;
    std::optional<CouplingDriverOwnedTransferDescriptor> driver_owned_descriptor;
};

enum class CouplingAuxiliaryEndpointResolutionKind {
    None,
    BlockState,
    InputSlot,
    Output,
    RegistryExtension
};

enum class CouplingRegionDataProviderKind {
    None,
    FEQuantity,
    BoundaryReductionFunctional,
    ProviderExtension
};

enum class CouplingResolvedTemporalBackingKind {
    None,
    SystemStateCurrent,
    SystemStateAccepted,
    SystemStatePredicted,
    SystemStateHistory,
    SystemStateStage,
    AuxiliaryCurrent,
    AuxiliaryCommitted,
    AuxiliaryPredicted,
    AuxiliaryHistory,
    AuxiliaryStage,
    InterfaceMapCurrent,
    InterfaceMapAccepted,
    InterfaceMapPredicted,
    InterfaceMapHistory,
    InterfaceMapStage,
    ProviderDefined,
    ExternalBuffer
};

struct ResolvedCouplingTemporalSlot {
    CouplingTemporalSlotDescriptor request;
    CouplingTemporalSlotDescriptor provided;
    CouplingResolvedTemporalBackingKind backing =
        CouplingResolvedTemporalBackingKind::None;
    std::string provider_name;
    std::optional<int> storage_index;
    std::uint64_t state_revision_key = 0;
    Real time = 0.0;
};

struct ResolvedCouplingEndpoint {
    // Declaration/request provenance retained for diagnostics only. Executable
    // lookup must use the resolved fields below.
    CouplingEndpointRef declaration_provenance;
    CouplingEndpointKind resolved_kind = CouplingEndpointKind::ExternalBuffer;
    CouplingValueDescriptor value;

    // Validated participant scope. Unset only for context-level global
    // driver-owned endpoints.
    std::optional<std::string> resolved_participant_name;

    std::string system_name;
    std::string registry_provider;
    // Registry-resolved stable key/name in the provider namespace. Examples:
    // field name, parameter key, FE quantity name, AuxiliaryInput name,
    // AuxiliaryOutput name, BoundaryReduction functional name, provider key,
    // or scoped external-buffer key.
    std::string resolved_endpoint_key;
    ResolvedCouplingTemporalSlot temporal;

    // Runtime-only FE owner for FE-backed endpoints. Persistent plan identity
    // must use system_name plus the kind-specific stable ids below.
    const systems::FESystem* system = nullptr;

    FieldId field_id = INVALID_FIELD_ID;
    std::optional<std::size_t> fe_quantity_id;
    std::optional<std::uint32_t> parameter_slot;
    params::ValueType parameter_value_type = params::ValueType::Any;
    CouplingRegionDataProviderKind region_data_provider_kind =
        CouplingRegionDataProviderKind::None;
    std::string region_data_provider_name;
    std::string boundary_functional_name;
    FieldId boundary_reduction_primary_field = INVALID_FIELD_ID;
    CouplingAuxiliaryEndpointResolutionKind auxiliary_kind =
        CouplingAuxiliaryEndpointResolutionKind::None;
    std::optional<std::size_t> auxiliary_block_index;
    std::optional<std::uint32_t> auxiliary_input_slot;
    std::optional<std::uint32_t> auxiliary_output_id;
    std::optional<std::uint32_t> auxiliary_output_flat_slot;
    std::string auxiliary_key;
    std::optional<CouplingExternalBufferDescriptor> external_buffer;

    std::uint64_t layout_revision_key = 0;
    std::uint64_t registry_revision_key = 0;
};

struct CouplingRegionEndpointDeclaration {
    std::string participant_name;
    std::string region_name;
    std::optional<std::string> shared_region_name;
};

struct CouplingExchangeDeclaration {
    CouplingPortId producer_port;
    CouplingPortId consumer_port;
    CouplingValueDescriptor value;
    std::optional<CouplingEndpointRef> producer;
    std::optional<CouplingEndpointRef> consumer;
    std::optional<std::string> shared_region_name;
    std::optional<CouplingRegionEndpointDeclaration> producer_region;
    std::optional<CouplingRegionEndpointDeclaration> consumer_region;
    CouplingTransferDeclaration transfer;
};

struct CouplingExchange {
    CouplingPortId producer_port;
    CouplingPortId consumer_port;
    CouplingValueDescriptor value;
    ResolvedCouplingEndpoint producer;
    ResolvedCouplingEndpoint consumer;
    std::optional<std::string> shared_region_name;
    std::optional<CouplingRegionRef> producer_region;
    std::optional<CouplingRegionRef> consumer_region;
    ResolvedCouplingTransfer transfer;
};

struct CouplingGroupHint {
    std::string name;
    std::vector<std::string> participant_names;
};

enum class CouplingRequirement {
    Required,
    Optional
};

enum class CouplingDependencyMode {
    ImplicitMonolithic,
    ExternalLagged
};

enum class CouplingTemporalQuantity {
    Time,
    TimeStep,
    EffectiveTimeStep,
    FieldDerivative,
    FieldHistoryValue,
    MeshVelocity,
    MeshAcceleration,
    PreviousMeshVelocity,
    PredictedMeshVelocity
};

enum class CouplingGeometryTerminalQuantity {
    MeshDisplacement,
    Coordinate,
    ReferenceCoordinate,
    CurrentCoordinate,
    PreviousCoordinate,
    ReferencePhysicalCoordinate,
    Jacobian,
    JacobianInverse,
    JacobianDeterminant,
    CurrentJacobian,
    ReferenceJacobian,
    CurrentJacobianDeterminant,
    ReferenceJacobianDeterminant,
    Normal,
    CurrentNormal,
    ReferenceNormal,
    CurrentMeasure,
    ReferenceMeasure,
    SurfaceJacobian,
    CellDiameter,
    CellVolume,
    FacetArea,
    CellDomainId
};

struct CouplingRegionEndpointDeclaration;

struct CouplingGeometryTerminalLocationDeclaration {
    CouplingRegionKind region_kind = CouplingRegionKind::Domain;
    std::optional<std::string> shared_region_name;
    CouplingInterfaceSide side = CouplingInterfaceSide::None;
    forms::GeometryConfiguration coordinate_configuration =
        forms::GeometryConfiguration::Reference;
    std::optional<forms::GeometryConfiguration> transform_from_configuration;
    std::optional<forms::GeometryConfiguration> transform_to_configuration;
};

struct CouplingGeometryTerminalLocationProvenance {
    CouplingRegionKind region_kind = CouplingRegionKind::Domain;
    std::optional<std::string> shared_region_name;
    int marker = -1;
    CouplingInterfaceSide side = CouplingInterfaceSide::None;
    forms::GeometryConfiguration coordinate_configuration =
        forms::GeometryConfiguration::Reference;
    std::optional<forms::GeometryConfiguration> transform_from_configuration;
    std::optional<forms::GeometryConfiguration> transform_to_configuration;
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    std::optional<svmp::search::LogicalInterfaceRegionId> logical_region;
#endif
    std::uint64_t geometry_revision = 0;
    std::uint64_t quadrature_policy_key = 0;
};

struct CouplingGeometryTerminalScope {
    // Required when more than one participant could own the geometry terminal.
    // A builder may infer it only in a validated single-participant context.
    std::optional<std::string> participant_name;
    std::optional<CouplingRegionEndpointDeclaration> region;
    std::optional<CouplingGeometryTerminalLocationDeclaration> location;
};

struct CouplingGeometryTerminalOwnerProvenance {
    std::string participant_name;
    std::string system_name;
    std::optional<std::string> region_name;
    std::optional<std::string> shared_region_name;
};

struct CouplingParticipantUse {
    std::string participant_name;
    CouplingRequirement requirement = CouplingRequirement::Required;
};

struct CouplingFieldUse {
    std::string participant_name;
    std::string field_name;
    CouplingRequirement requirement = CouplingRequirement::Required;
};

struct CouplingRegionUse {
    std::string participant_name;
    std::string region_name;
    std::optional<CouplingRegionKind> required_region_kind;
    CouplingRequirement requirement = CouplingRequirement::Required;
};

enum class CouplingVariableKind {
    Field,
    AuxiliaryState,
    AuxiliaryInput,
    AuxiliaryOutput,
    // Forms BoundaryIntegral provenance maps to this kind until Analysis grows
    // a distinct BoundaryIntegral variable kind.
    BoundaryFunctional,
    GlobalScalar
};

struct CouplingVariableUse {
    CouplingVariableKind kind = CouplingVariableKind::Field;
    std::string participant_name;
    std::string name;
    int component = -1;
    CouplingRequirement requirement = CouplingRequirement::Required;
};

struct CouplingSharedRegionUse {
    std::string shared_region_name;
    std::optional<CouplingRegionKind> required_region_kind;
    CouplingRequirement requirement = CouplingRequirement::Required;
};

struct CouplingResidualDependency {
    CouplingVariableUse residual_row;
    CouplingVariableUse dependency;
    CouplingDependencyMode mode = CouplingDependencyMode::ImplicitMonolithic;
};

enum class CouplingNonFieldDependencyRequirementKind {
    Parameter,
    Coefficient,
    MaterialStateOld,
    MaterialStateWork,
    BoundaryFunctional,
    // Provenance subtype that maps to BoundaryFunctional graph identity for now.
    BoundaryIntegral,
    AuxiliaryState,
    AuxiliaryInput,
    AuxiliaryOutput
};

struct CouplingNonFieldDependencyRequirement {
    CouplingNonFieldDependencyRequirementKind kind =
        CouplingNonFieldDependencyRequirementKind::AuxiliaryInput;
    std::string participant_name;
    std::string name;
    std::optional<CouplingRegionEndpointDeclaration> region;
    std::optional<CouplingRegionKind> required_region_kind;
    std::optional<params::ValueType> expected_parameter_value_type;
    std::string expected_value_type;
    std::optional<std::uint64_t> material_state_byte_offset;
    CouplingRequirement requirement = CouplingRequirement::Required;
    // True only for dependency kinds that must be adapted to
    // analysis::VariableKey and participate in coupling-graph dependency edges.
    bool require_analysis_variable_key = false;
};

struct CouplingTemporalRequirement {
    CouplingTemporalQuantity quantity = CouplingTemporalQuantity::Time;
    std::optional<CouplingFieldUse> field;
    std::optional<CouplingGeometryTerminalScope> mesh_motion_scope;
    std::optional<systems::MeshMotionFieldRole> mesh_motion_role;
    int derivative_order = 0;
    int history_index = 0;
    CouplingRequirement requirement = CouplingRequirement::Required;
};

struct CouplingGeometryTerminalRequirement {
    CouplingGeometryTerminalQuantity quantity =
        CouplingGeometryTerminalQuantity::MeshDisplacement;
    CouplingGeometryTerminalScope scope;
    std::optional<CouplingFieldUse> mesh_motion_field;
    CouplingRequirement requirement = CouplingRequirement::Required;
};

struct CouplingBlockExpectation {
    CouplingVariableUse residual_row;
    CouplingVariableUse dependency;
    bool expected_nonzero = true;
    bool expect_matrix_block = true;
};

struct CouplingGeometrySensitivityDeclaration {
    forms::GeometrySensitivityMode mode =
        forms::GeometrySensitivityMode::GeometryConstant;
    std::optional<CouplingFieldUse> mesh_motion_field;
    forms::GeometryTangentPath tangent_path = forms::GeometryTangentPath::Auto;
    bool use_symbolic_tangent = false;
};

enum class CouplingFormTerminalProvenanceKind {
    PreviousSolution,
    MeshTemporal,
    GeometryTerminal
};

struct CouplingFormTerminalProvenanceDeclaration {
    CouplingFormTerminalProvenanceKind kind =
        CouplingFormTerminalProvenanceKind::GeometryTerminal;
    // Stable declaration order assigned after matching by public FormExpr node
    // identity from CouplingFormBuilder's side table.
    std::uint64_t terminal_sequence = 0;
    std::optional<CouplingFieldUse> field;
    std::optional<CouplingGeometryTerminalScope> scope;
    CouplingTemporalQuantity temporal_quantity = CouplingTemporalQuantity::Time;
    CouplingGeometryTerminalQuantity geometry_quantity =
        CouplingGeometryTerminalQuantity::MeshDisplacement;
    std::optional<systems::MeshMotionFieldRole> mesh_motion_role;
    int derivative_order = 0;
    int history_index = 0;
};

struct CouplingSymbolicOptionsDeclaration {
    // Mirrors forms::SymbolicOptions declaration-time tuning fields except for
    // ad_mode, use_symbolic_tangent, geometry_tangent_path, and
    // geometry_sensitivity. Those are resolved from the top-level coupling
    // install declaration so declarations cannot carry raw FieldIds or duplicate
    // geometry tangent policy.
    forms::JITOptions jit{};
    bool simplify_expressions = true;
    bool exploit_sparsity = true;
    bool cache_expressions = true;
    bool verbose = false;
};

struct CouplingFormInstallOptionsDeclaration {
    forms::ADMode ad_mode{forms::ADMode::Forward};
    CouplingSymbolicOptionsDeclaration compiler_options{};
    CouplingGeometrySensitivityDeclaration geometry_sensitivity{};
};

struct CouplingFormContribution {
    std::string contribution_name;
    std::string origin;
    std::string operator_name = "equations";
    std::vector<CouplingFieldUse> field_uses;
    std::vector<CouplingFieldUse> extra_trial_field_uses;
    std::vector<CouplingFormTerminalProvenanceDeclaration> terminal_provenance;
    CouplingFormInstallOptionsDeclaration install_options{};
    forms::FormExpr residual;
};

struct ResolvedCouplingFormContribution {
    std::string contribution_name;
    std::string origin;
    std::string system_name;
    OperatorTag operator_name{"equations"};
    std::vector<FieldId> fields;
    std::vector<FieldId> extra_trial_fields;
    std::vector<CouplingFormTerminalProvenanceDeclaration> terminal_provenance;
    systems::FormInstallOptions install_options{};
    forms::FormExpr residual;
};

enum class CouplingAdditionalFieldScope {
    VolumeCell,
    InterfaceFace
};

enum class CouplingAdditionalFieldNamespace {
    Participant,
    Contract
};

struct CouplingAdditionalFieldDeclaration {
    CouplingAdditionalFieldNamespace field_namespace =
        CouplingAdditionalFieldNamespace::Participant;
    // Participant name for participant-scoped fields, or contract instance name
    // for contract-owned fields such as monolithic multipliers.
    std::string namespace_name;
    // Participant whose FESystem receives the field. Empty means namespace_name
    // for participant-scoped fields. For contract-owned fields, empty is allowed
    // only for interface fields when the shared-region participants resolve to a
    // single monolithic FESystem.
    std::string system_participant_name;
    std::string field_name;
    std::shared_ptr<const spaces::FunctionSpace> space;
    int components = 0;  // 0 means infer from space->value_dimension()
    CouplingAdditionalFieldScope scope = CouplingAdditionalFieldScope::VolumeCell;
    std::optional<std::string> region_name;
    std::optional<std::string> shared_region_name;
    CouplingRequirement requirement = CouplingRequirement::Required;
};

struct ResolvedCouplingAdditionalFieldDeclaration {
    CouplingAdditionalFieldDeclaration declaration;
    std::string system_name;
    systems::FieldSpec field_spec;
};

struct CouplingInstalledDependency {
    analysis::VariableKey residual_row;
    analysis::VariableKey dependency;
    CouplingDependencyMode mode = CouplingDependencyMode::ImplicitMonolithic;
    analysis::DomainKind domain = analysis::DomainKind::Cell;
    bool contributes_matrix_block = false;
    bool contributes_vector = true;
    std::string provider;
};

struct CouplingInstalledBlockProvenance {
    analysis::VariableKey residual_row;
    analysis::VariableKey dependency;
    std::vector<analysis::DomainKind> domains;
    bool has_matrix = false;
    bool has_vector = false;
};

struct CouplingInstallMetadata {
    std::string contribution_name;
    std::string origin;
    std::string system_name;
    OperatorTag operator_name{"equations"};
    std::vector<CouplingInstalledDependency> installed_dependencies;
    std::vector<CouplingInstalledBlockProvenance> installed_blocks;
};

struct CouplingFormFieldProvenance {
    FieldId residual_row = INVALID_FIELD_ID;
    FieldId field = INVALID_FIELD_ID;
    bool appears_as_test_field = false;
    bool appears_as_state_field = false;
    bool appears_as_discrete_field = false;
    bool appears_as_geometry_sensitivity = false;
};

enum class CouplingGeometrySensitivityProvenanceKind {
    None,
    MeshMotionUnknowns,
    CutGeometry,
    DriverProvided
};

struct CouplingGeometrySensitivityProvenance {
    CouplingGeometrySensitivityProvenanceKind kind =
        CouplingGeometrySensitivityProvenanceKind::None;
    FieldId mesh_motion_field = INVALID_FIELD_ID;
    std::string provenance_id;
    std::string construction_policy;
    std::string target_kind;
    std::uint64_t source_stable_id = 0;
    std::uint64_t cut_topology_revision = 0;
    std::uint64_t quadrature_policy_key = 0;
    MeshIndex parent_entity = static_cast<MeshIndex>(-1);
    bool ad_compatible = false;
    bool location_sensitivity_available = false;
    bool jacobian_sensitivity_available = false;
    bool measure_sensitivity_available = false;
    bool normal_sensitivity_available = false;
    bool quadrature_weight_sensitivity_available = false;
    std::vector<FieldId> geometry_fields;
    std::vector<MeshIndex> parent_geometry_dofs;
    std::vector<std::string> visible_to_assembly_paths;
    std::size_t sensitivity_sample_count = 0;
};

struct CouplingFormTemporalProvenance {
    std::optional<FieldId> field;
    std::optional<FieldId> active_trial_field;
    std::optional<analysis::VariableKey> residual_row;
    std::optional<analysis::VariableKey> trial_dependency;
    std::optional<std::string> trial_block_id;
    std::optional<CouplingGeometryTerminalScope> mesh_motion_scope;
    std::optional<systems::MeshMotionFieldRole> mesh_motion_role;
    CouplingTemporalQuantity quantity = CouplingTemporalQuantity::Time;
    int derivative_order = 0;
    int history_index = 0;
};

struct CouplingFormGeometryTerminalProvenance {
    CouplingGeometryTerminalQuantity quantity =
        CouplingGeometryTerminalQuantity::MeshDisplacement;
    FieldId mesh_motion_field = INVALID_FIELD_ID;
    CouplingGeometryTerminalLocationProvenance location;
    analysis::DomainKind analysis_domain = analysis::DomainKind::Cell;
    std::optional<CouplingGeometryTerminalOwnerProvenance> owner;
    std::string provider;
    bool value_available = false;
    bool gradient_or_jacobian_available = false;
    bool normal_available = false;
    bool measure_available = false;
};

enum class CouplingFormNonFieldDependencyKind {
    Parameter,
    Coefficient,
    MaterialStateOld,
    MaterialStateWork,
    BoundaryFunctional,
    // Provenance subtype. Graph-variable adaptation maps to
    // analysis::VariableKind::BoundaryFunctional until Analysis has a distinct
    // BoundaryIntegral kind.
    BoundaryIntegral,
    AuxiliaryState,
    AuxiliaryInput,
    AuxiliaryOutput
};

struct CouplingFormNonFieldDependencyProvenance {
    CouplingFormNonFieldDependencyKind kind =
        CouplingFormNonFieldDependencyKind::AuxiliaryInput;
    std::string participant_name;
    std::string system_name;
    std::string name;
    analysis::DomainKind domain = analysis::DomainKind::Cell;
    std::optional<std::string> region_name;
    std::optional<std::string> shared_region_name;
    int marker = -1;
    CouplingInterfaceSide side = CouplingInterfaceSide::None;
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    std::optional<svmp::search::LogicalInterfaceRegionId> logical_region;
#endif
    std::optional<std::uint32_t> slot;
    std::optional<std::uint32_t> output_id;
    std::optional<std::uint64_t> byte_offset;
    std::string provider;
    std::string value_type;
    std::optional<params::ValueType> parameter_value_type;
};

struct CouplingFormVariableDependencyProvenance {
    analysis::VariableKey residual_row;
    analysis::VariableKey dependency;
    CouplingDependencyMode mode = CouplingDependencyMode::ImplicitMonolithic;
    analysis::DomainKind domain = analysis::DomainKind::Cell;
    bool contributes_matrix_block = false;
    bool contributes_vector = true;
    std::string provider;
};

struct CouplingFormAnalysisMetadata {
    std::string contribution_name;
    std::string origin;
    std::string system_name;
    OperatorTag operator_name{"equations"};
    std::vector<FieldId> installed_fields;
    std::vector<CouplingFormFieldProvenance> field_uses;
    std::vector<CouplingFormNonFieldDependencyProvenance> non_field_dependencies;
    std::vector<CouplingFormVariableDependencyProvenance> variable_dependencies;
    std::vector<CouplingFormTerminalProvenanceDeclaration>
        declaration_terminal_provenance;
    std::vector<CouplingFormTemporalProvenance> temporal_symbols;
    std::vector<CouplingFormGeometryTerminalProvenance> geometry_terminals;
    forms::GeometrySensitivityOptions geometry_sensitivity{};
    std::vector<CouplingGeometrySensitivityProvenance> geometry_sensitivity_provenance;
    std::vector<CouplingInstalledDependency> installed_dependencies;
    std::vector<CouplingInstalledBlockProvenance> installed_blocks;
};

struct CouplingContractDeclaration {
    // Reusable contract type/registry key. Must match CouplingContract::name().
    std::string contract_type;
    // Configured instance namespace. Multiple declarations may share a
    // contract_type, but contract_name must be unique within the coupling setup.
    std::string contract_name;
    std::vector<CouplingParticipantUse> participants;
    std::vector<CouplingFieldUse> fields;
    std::vector<CouplingRegionUse> regions;
    std::vector<CouplingSharedRegionUse> shared_regions;
    std::vector<CouplingAdditionalFieldDeclaration> additional_fields;
    std::vector<CouplingNonFieldDependencyRequirement> non_field_dependencies;
    std::vector<CouplingResidualDependency> dependencies;
    std::vector<CouplingTemporalRequirement> temporal_requirements;
    std::vector<CouplingGeometryTerminalRequirement> geometry_requirements;
    std::vector<CouplingBlockExpectation> expected_blocks;
    std::vector<CouplingExchangeDeclaration> partitioned_exchange_declarations;
    std::vector<CouplingGroupHint> group_hints;
};
```

`CouplingContract::name()` is the reusable contract type or registry key, for
example an FSI contract implementation. `CouplingContractDeclaration::contract_name`
is the configured instance namespace for one use of that contract, for example
one particular fluid-solid interface. Multiple declarations may share the same
`contract_type`, but `contract_name` must be unique within a coupled problem and
must be the namespace used by contract-owned ports, additional fields, and
diagnostics.

`CouplingRegionUse` declares participant-local region requirements. It is used
for boundary, domain, interior-face, or interface-face regions that a contract
will access directly through `CouplingContext::region(participant, region)`.
`CouplingSharedRegionUse` declares N-participant shared-region requirements.
The two declaration types must remain distinct so a contract can validate both
ordinary participant-local boundary data and shared interface topology before
any `FieldId`, marker, or interface-map handle is resolved.

`CouplingVariableUse` is the declaration-side counterpart of the existing
`analysis::VariableKey` vocabulary. Field variables resolve through participant
field names and can carry a component selector. `AuxiliaryState`,
`AuxiliaryInput`, `AuxiliaryOutput`, `BoundaryFunctional`, and `GlobalScalar`
variables resolve to named non-field variables already represented by the
Analysis subsystem. Declaration records carry only participant/system scope and
stable names; resolved slots, output ids, and provider-local indices belong to
installed provenance records. FE/Coupling should adapt installed monolithic
dependencies to `analysis::VariableKey` rather than inventing a second coupling
graph variable model. Because `analysis::VariableKey::named(...)` stores only a
kind and stable name, the adapter must use canonical qualified names that include
the owning system or participant scope for non-field variables. Raw slots,
output ids, and provider-local indices are provenance used for lookup and
diagnostics; they must not become unscoped equality keys. Ordinary FE Jacobian
block expectations are still field-to-field matrix-block expectations; non-field
dependencies use the same variable reference shape with `expect_matrix_block ==
false` unless the existing Systems metadata reports a linearized contribution
that truly produces a matrix block. Installed block provenance must use
`analysis::VariableKey` rows and columns so true non-field matrix contributions
can be represented without falling back to field-only `FieldId` pairs.
The same declaration/resolved separation applies to expert custom install
hooks: contract declarations and expected blocks use `CouplingVariableUse`, but
each `CouplingInstallMetadata` is a resolved install record for one expert
contribution. Expert hooks must return one metadata record per installed custom
contribution and must report stable contribution name, diagnostic origin, owning
system, operator tag, installed dependencies, and blocks with scoped
`analysis::VariableKey` rows and dependencies, explicit `analysis::DomainKind`
provenance, and matrix/vector contribution flags.
They must not return declaration-shaped `CouplingVariableUse` records as if they
were proof of installed structure.

`CouplingNonFieldDependencyRequirement` is for Forms-visible data dependencies
that are not FE fields. Parameter, coefficient, and material-state requirements
are provenance/data requirements unless Systems promotes them to a public
`analysis::VariableKey` model. Boundary functionals, auxiliary state,
`AuxiliaryInput`, and `AuxiliaryOutput` may require `analysis::VariableKey`
adaptation when they participate in coupling-graph dependency edges. Forms
`BoundaryIntegral` terminals remain distinct provenance records in
`CouplingFormNonFieldDependencyProvenance`, but they map to
`CouplingVariableKind::BoundaryFunctional` and
`analysis::VariableKind::BoundaryFunctional` until Analysis exposes a distinct
boundary-integral variable kind. `CouplingResidualDependency` is therefore only
for dependencies that are graph variables; non-field data requirements that
cannot or should not become graph edges belong in `non_field_dependencies`.
The initial Forms terminal mapping is:

```text
Forms terminal                    Coupling provenance kind      Graph variable identity
BoundaryFunctionalSymbol           BoundaryFunctional           BoundaryFunctional
BoundaryIntegralSymbol             BoundaryIntegral             BoundaryFunctional fallback
BoundaryIntegralRef                BoundaryIntegral             BoundaryFunctional fallback
AuxiliaryStateSymbol/Ref           AuxiliaryState               AuxiliaryState when requested
AuxiliaryInputSymbol/Ref           AuxiliaryInput               AuxiliaryInput when requested
AuxiliaryOutputSymbol/Ref          AuxiliaryOutput              AuxiliaryOutput when requested
ParameterSymbol/Ref                Parameter                    none unless Analysis adds one
Coefficient                        Coefficient                  none unless Analysis adds one
MaterialStateOldRef/WorkRef        MaterialStateOld/Work        none unless Analysis adds one
```

`BoundaryFunctionalSymbol` is the graph-variable form of a named boundary
functional. `BoundaryIntegralSymbol` and `BoundaryIntegralRef` are
boundary-integral data/provenance terminals; they must remain distinct in
installed provenance even though expected-block graph adaptation uses
BoundaryFunctional identity until Analysis grows a dedicated boundary-integral
variable kind.
Requirements remain declaration-shaped: names, expected value/type constraints,
and optional declaration-side region/scope constraints are allowed, but resolved
markers, logical regions, parameter slots, auxiliary output ids, and
provider-local indices are reported only by resolved provenance. Material-state
requirements may use `material_state_byte_offset` because current Forms
material-state terminals are offset-addressed rather than name-addressed.

`CouplingAdditionalFieldDeclaration` has two separate identities: the namespace
used for coupling lookup and diagnostics, and the participant whose `FESystem`
is mutated during registration. Participant-scoped fields use
`field_namespace == Participant` and `namespace_name` equal to the participant
name; when `system_participant_name` is empty, that participant is also the
registration target. Contract-owned fields use `field_namespace == Contract` and
`namespace_name` equal to `CouplingContractDeclaration::contract_name`; they are
not participants and must not satisfy participant requirements. A contract-owned
field still needs a concrete registration target: either
`system_participant_name` names the participant whose `FESystem` receives the
field, or an interface/shared-region declaration proves all relevant
participants resolve to the same monolithic `FESystem`. Contracts should omit
disabled additional fields from `additional_fields`. If an additional-field
declaration is present with `Required`, the builder must register it or bind a
pre-existing compatible field and fail on mismatch. If it is present with
`Optional`, missing prerequisites are recorded as skipped diagnostics and no
field, dependency edge, or expected block may depend on it; when all
prerequisites are present and the field is selected, the normal compatibility
and collision checks still apply.

For `CouplingTemporalRequirement`, `FieldDerivative` requires `field` and a
positive `derivative_order`. `FieldHistoryValue` requires `field` and a positive
logical 1-based `history_index` matching Forms `PreviousSolutionRef(k)` and
`SystemStateView::u_history[k - 1]`. `MeshVelocity`, `MeshAcceleration`,
`PreviousMeshVelocity`, and `PredictedMeshVelocity` require the relevant
mesh-motion role or provider to be bound by the owning `FESystem` or
problem-level temporal policy, and in N-participant contexts they must provide a
`mesh_motion_scope` that identifies the owning participant, region, or
shared-region side. `mesh_motion_role` should resolve to the existing
`systems::MeshMotionFieldRole` value that backs the requested quantity. `Time`,
`TimeStep`, and `EffectiveTimeStep` should leave `field`, `mesh_motion_scope`,
and `mesh_motion_role` empty and ignore `derivative_order` and `history_index`.
`PreviousSolutionRef(k)` is field/trial-history scoped Forms temporal
provenance, not a named non-field coupling variable; diagnostics must therefore
associate it with the owning field and history index through
`CouplingFormTemporalProvenance`. For mixed forms, that provenance must also
identify the active trial field or installed trial block that owns the
`PreviousSolutionRef(k)` terminal. A helper call naming participant/field
metadata is valid only when the named field is the active trial field for the
contribution being installed, or when the metadata bridge can unambiguously bind
the terminal to the owning trial block. Global temporal symbols such as `t()`,
`deltat()`, and `deltat_eff()` leave residual-row and trial-block provenance
empty because they are not trial-field scoped. Mesh temporal terminals must
carry `mesh_motion_scope` and `mesh_motion_role` provenance so the metadata
bridge can validate them against owner-scoped mesh-motion declarations instead
of treating `meshVelocity()` and related terminals as globally owned.

`CouplingGeometryTerminalRequirement` covers Forms geometry terminals that are
not time requirements but still require setup validation: `meshDisplacement()`,
`x()`, `X()`, current/previous/reference coordinates, generic and
current/reference Jacobians and determinants, Jacobian inverses, generic and
current/reference normals, current/reference measures, surface Jacobians, cell
diameters, cell volumes, facet areas, and cell-domain ids. These requirements
validate that the selected geometry transaction, mesh-motion binding, and
Assembly metadata can provide the terminal values or derivatives used by a
residual. Detailed installed
provenance is reported separately through `CouplingFormGeometryTerminalProvenance`
so mesh/geometry-terminal diagnostics are not conflated with
time-integration diagnostics. Geometry-terminal declaration and installed
provenance must remain split. Declaration location records may name
declaration-side `CouplingRegionKind`, shared-region scope, side, and typed
`forms::GeometryConfiguration` intent. Installed provenance records carry the
resolved `analysis::DomainKind`, owning participant/system provenance,
shared-region name and resolved marker, minus/plus side when applicable, typed
`LogicalInterfaceRegionId` when mesh support is enabled, typed
`forms::GeometryConfiguration` value configuration, optional
`pullback()`/`pushforward()` from/to configuration provenance, geometry revision,
and quadrature-policy key.
Normals, measures, Jacobian determinants, cell metrics, and surface Jacobians are
integration-context quantities; the plan should never treat them as globally
available scalar properties without this location and provenance.
For N-participant contracts, geometry-terminal requirements must also name the
owning participant or a declaration-side participant/region attachment through
`CouplingRegionEndpointDeclaration`; a shared-region name alone is not enough to
identify which side owns a normal, measure, or boundary geometry terminal. If
both `scope.participant_name` and `scope.region` are present, validation must
reject the requirement unless the participant names agree. For
geometry-terminal requirements, `scope.region.shared_region_name` and
`scope.location.shared_region_name` must match when both are present. If
`scope.location.shared_region_name` is absent and the region owner names a
shared region, the location inherits that shared-region scope. If the owner is
participant-only and the location names a shared region, that shared-region name
defines the terminal location scope but does not replace the participant owner.
Boundary/interface terminals that need shared-side ownership must fail validation
when no shared-region scope is available.

For `CouplingTransferDeclaration`, `Unspecified` is only a declaration-time
sentinel and must fail validation for any exchange that is meant to be generated
or executed. `Identity` must be explicit; it is valid only when producer and
consumer endpoints already share compatible storage/layout semantics or when the
resolved endpoint metadata proves that a direct shape-preserving copy is
well-defined. It must not be used as the default for omitted transfer
declarations.

For `CouplingTransferDeclaration`, interface transfers must lower to the
existing `FE/Systems/InterfaceOperators` vocabulary when mesh support is enabled:
`InterfacePointwiseInterpolation` maps to `InterfaceOperatorKind::PointwiseInterpolation`,
`InterfaceConservativeProjection` maps to
`InterfaceOperatorKind::ConservativeProjection`, and `InterfaceMortar` maps to
`InterfaceOperatorKind::Mortar`. Declaration records should stay independent of
Systems-only option types. During partitioned plan resolution, FE/Coupling
combines `CouplingValueDescriptor`, endpoint metadata, and
`CouplingInterfaceTransferDeclaration` into a `ResolvedCouplingTransfer` with
`systems::InterfaceTransferOptions` when mesh support is enabled. Resolution must
validate component count, scalar/vector/rank-2 tensor kind, frame-transform
policy, source-to-target rotation, 2D source embedding and target restriction
policy when applicable, conservation tolerance, source/target interface markers,
source/target owning system names, interface-search-registry identity, search
revision keys, interface map revision/state, the
interface-search-registry entry name, and the concrete interface-map name that
the driver can resolve through the current context to the
`svmp::search::InterfaceMap` required by
`applyInterfaceTransfer()`. Frame transforms are not generic tensor operations:
`None` is required for scalar and mixed-block transfers, vector transforms
require vector payloads with at least the supported transformed component count,
true 2D vector transforms require explicit embedding/restriction policies,
and rank-2 tensor transforms require rank-2 tensor payloads with at least the
supported transformed component count. If `interface_declaration` is omitted for
an interface transfer,
FE/Coupling may derive conservative defaults from the resolved endpoints only
when the derivation is unambiguous; otherwise validation must fail. When
`SVMP_FE_WITH_MESH` is disabled, interface transfer kinds are unavailable and
must fail validation rather than compiling in a parallel options type.
Because the current `systems::InterfaceTransferOptions` type has no 2D
embedding/restriction fields, `ResolvedCouplingTransfer` must preserve
`source_embedding_policy` and `target_restriction_policy` separately. Execution
must either apply those policies in a coupling-side pre/post adapter around
`applyInterfaceTransfer()`, or reject true 2D FE frame transforms until that
adapter exists; the policies must not be silently dropped when constructing
Systems options.
`DriverOwned` is only for transfers implemented above FE/Coupling by a
partitioned driver or external adapter, and it must be explicitly named and
registered through a `CouplingDriverOwnedTransferDescriptor` before plan
validation accepts it. Resolved `DriverOwned` transfers must carry the resolved
descriptor or equivalent durable capability and revision provenance so execution
and diagnostics can verify supported ranks, temporal slots, component-layout
preservation, and general-tensor support without consulting declaration-only
state. `GeneralTensor` support is true only when
`CouplingValueRank::GeneralTensor` is present in `supported_ranks`; descriptors
must not add an out-of-band general-tensor capability flag because that would
create contradictory states. Durable plan equality and serialization use the
stable descriptor fields and revision keys, not registry object identity,
non-owning runtime handles, or the transient lookup result that produced the
descriptor.

`CouplingExchangeDeclaration` is declaration-time topology. It may omit concrete
endpoint bindings while still declaring ports, value shape, shared-region scope,
and transfer requirements. It must not carry `systems::InterfaceTransferOptions`,
resolved `CouplingRegionRef` records, `FESystem*` pointers, or marker values
whose identity depends on a resolved system. Declaration-side region attachments
use participant/region/shared-region names through
`CouplingRegionEndpointDeclaration`; resolved `CouplingExchange` records may then
carry `CouplingRegionRef` after lookup. `CouplingExchange` is the resolved
executable plan entry produced by partitioned-plan generation after endpoint
lookup has delegated to the existing FE registries and transfer declarations
have been resolved to Systems transfer options where applicable. For interface
exchanges, the resolved plan must keep
`systems::InterfaceTransferOptions` separate from
`CouplingInterfaceMapProvenance`: the options describe value-shape and transfer
operator settings, while the provenance describes which source/target systems,
registered interface search registry, interface entry name, concrete interface
map identity, source/target interface markers, revision keys, operator state,
and accepted/trial epoch the driver must use when calling Systems interface
operators. Shared-region names may appear at the exchange level and inside
producer/consumer `CouplingRegionEndpointDeclaration` records. These names all
refer to the same declaration-side shared-region registry entry for the
exchange; validation must reject conflicting names. If a region attachment omits
`shared_region_name`, the exchange-level value is inherited. If both the
exchange and the endpoint attachment omit it, the endpoint is participant-local
and cannot be used where a shared interface or side ownership is required.
Geometry-terminal shared-region consistency is validated by the separate
geometry-terminal requirement rules because those terminals may be declared
outside a partitioned exchange. `CouplingInterfaceMapProvenance` is durable
resolved metadata and must use stable names, markers, and revision keys rather
than raw `FESystem*` identity. Runtime execution may build a
`CouplingInterfaceMapRuntimeHandles` view from the current `CouplingContext`, but
those non-owning pointers are not declaration metadata and are not serialized or
used as stable equality keys. The runtime handle must include the actual
`svmp::search::InterfaceMap` passed to `applyInterfaceTransfer()`, and may also
retain the owning `SlidingInterfaceMap` state wrapper when Systems is managing
accepted/trial map state. Interface provenance must be validated against the
current `SlidingInterfaceMap` state and embedded `svmp::search::InterfaceMap`
metadata before execution so stale maps, wrong-system maps, or trial maps used
as accepted data are rejected deterministically.
`interface_entry_name` is the name passed to
`InterfaceSearchRegistry::interface_entry()` or map-building APIs;
`interface_map_name` is the concrete trial/committed map identity. The first
implementation may set them to the same string when the registry entry and map
share a name, but the schema must preserve both concepts so declaration
metadata, search registry entries, and resolved map state do not get conflated.

For `CouplingAdditionalFieldDeclaration`, the declaration must contain enough
information to lower directly to the existing field-registration APIs. Volume
fields lower to `FESystem::addField(systems::FieldSpec{...})`. Interface fields
lower to `FESystem::addInterfaceField(...)` or an equivalent `systems::FieldSpec` with
`FieldScope::InterfaceFace`, but declaration records must name a participant
region or shared region rather than carrying a raw interface marker. The
monolithic builder resolves that region through `CouplingContext` and stores the
resolved marker only in
`ResolvedCouplingAdditionalFieldDeclaration::field_spec.interface_marker`. The
authoritative shape for monolithic FE fields is the `FunctionSpace` plus
`components`, checked against `space->value_dimension()` and the resulting
`systems::FieldSpec`. A component count of `0` means "infer from
`space->value_dimension()`", matching the existing `FESystem::addField()`
behavior. A positive component count must exactly match the function-space value
dimension. Negative component counts are invalid in FE/Coupling declarations
and must be rejected before calling `FESystem::addField()`, even though the
lower-level Systems API currently treats nonpositive values as infer. Partitioned
payload shape remains `CouplingValueDescriptor` metadata on ports and exchanges;
it should be derived from or validated against registered FE fields rather than
duplicated inside additional-field declarations.
Additional-field attachment validation must reject ambiguous combinations.
`VolumeCell` declarations must leave both `region_name` and `shared_region_name`
unset because no interface marker is needed. `InterfaceFace` declarations must
set exactly one of `region_name` or `shared_region_name`. A participant-local
`region_name` must resolve through `CouplingContext::region()` to an
`InterfaceFace` region for the resolved registration-target participant. A
`shared_region_name` must resolve through `sharedRegion(name, participant)` for
the explicit `system_participant_name`, or for `namespace_name` when the field
is participant-scoped and no separate target participant is provided. For a
contract-owned field with no explicit target participant, the builder must check
all shared-region participants under the shared-monolithic-system policy. Both
paths produce the final `systems::FieldSpec` scope and interface marker during
resolution.

```cpp
class MonolithicCouplingInstallContext;

class CouplingContract {
public:
    virtual ~CouplingContract() = default;

    // Reusable contract type/registry key, not a configured instance name.
    virtual std::string name() const = 0;

    virtual CouplingContractDeclaration declare() const = 0;

    virtual void validate(const CouplingContext& ctx) const = 0;

    virtual std::vector<CouplingFormContribution> buildMonolithicForms(
        const CouplingContext& ctx,
        const CouplingFormBuilder& forms) const { return {}; }

    virtual std::vector<CouplingInstallMetadata> installMonolithicTerms(
        MonolithicCouplingInstallContext& install,
        const CouplingContext& ctx) { return {}; }

    virtual std::vector<CouplingExchangeDeclaration>
    buildPartitionedExchangeDeclarations(const CouplingContext& ctx) const
    {
        return {};
    }
};
```

This lets a coupling module define one physical relationship and support both
monolithic and partitioned lowering.

`regions`, `shared_regions`, `additional_fields`,
`partitioned_exchange_declarations`, and `group_hints` are part of the
declaration because the coupling graph needs to reason about region
requirements, field-name collisions, and partitioned topology before a driver
starts executing a solve.
The monolithic coupling builder, not individual contracts, performs actual
`FESystem` field registration from `additional_fields` and records the resulting
`FieldId`s in the refreshed context. This keeps validation, collision handling,
and failed-setup behavior centralized and avoids contracts mutating Systems state
before the declaration graph has been checked.

For monolithic mode:

```text
CouplingContract -> install residual/coupling terms into FESystem
```

Contracts that define mathematical coupling relations should prefer
`buildMonolithicForms()` when the relation can be expressed with the FE Forms
vocabulary. `installMonolithicTerms()` remains an expert fallback for custom
kernels, non-Forms operators, or transitional implementations. These hooks
should have empty defaults so a contract only implements the lowering paths it
supports. Expert hooks must return a `std::vector<CouplingInstallMetadata>` with
one record per installed custom contribution; otherwise the coupling graph cannot
verify expected monolithic block structure for custom installations. Each
metadata record is resolved install provenance: contribution identity, rows,
dependencies, and block records use scoped `analysis::VariableKey` values plus
domain and matrix/vector contribution flags. Declaration-side
`CouplingVariableUse`
expectations are not sufficient evidence that an expert hook actually installed
the claimed coupling.

Expert hooks should not call raw `FESystem::addCellKernel`,
`addBoundaryKernel`, `addInteriorFaceKernel`, or `addInterfaceFaceKernel`
directly. Those methods are internal Systems registration paths used by
`FormsInstaller`. `MonolithicCouplingInstallContext` should expose only
approved Systems-level extension points, such as installing a prebuilt
`MixedFormIR`, installing an approved custom contribution type, or recording
metadata for a contribution installed by another sanctioned subsystem. If no
approved extension point exists for a custom coupling, the implementation
should add one to FE/Systems instead of bypassing Systems ownership from a
physics contract.

For partitioned mode:

```text
CouplingContract -> produce exchange declarations/templates
FE/Coupling -> resolve declarations into a partitioned driver data-flow plan
```

For N-multiphysics, contract declarations should be merged into a global
declaration graph before any mode-specific lowering runs. The graph is then
refreshed after additional field registration, Forms installation, expert
custom installation, and partitioned plan generation so diagnostics can
compare declared topology with the actual installed or generated topology.

## FE Forms Authoring For Coupling Contracts

Physics-specific coupling contracts should be able to define their
mathematical monolithic coupling relations with FE Forms when the relation is
expressible as a symbolic residual.

`FE/Coupling` should provide a small form-building helper that resolves
participant/field names through `CouplingContext` and creates field-bound
Forms symbols:

```cpp
class CouplingFormBuilder {
public:
    forms::FormExpr state(std::string_view participant,
                          std::string_view field,
                          std::string_view symbol) const;

    forms::FormExpr test(std::string_view participant,
                         std::string_view field,
                         std::string_view symbol) const;

    forms::FormExpr timeDerivative(std::string_view participant,
                                   std::string_view field,
                                   std::string_view symbol,
                                   int order = 1) const;

    forms::FormExpr previousSolution(std::string_view participant,
                                     std::string_view field,
                                     int steps_back = 1) const;

    forms::FormExpr time() const;

    forms::FormExpr timeStep() const;

    forms::FormExpr effectiveTimeStep() const;

    forms::FormExpr meshDisplacement(
        const CouplingGeometryTerminalScope& scope) const;

    forms::FormExpr meshVelocity(
        const CouplingGeometryTerminalScope& scope) const;

    forms::FormExpr meshAcceleration(
        const CouplingGeometryTerminalScope& scope) const;

    forms::FormExpr previousMeshVelocity(
        const CouplingGeometryTerminalScope& scope) const;

    forms::FormExpr predictedMeshVelocity(
        const CouplingGeometryTerminalScope& scope) const;

    forms::FormExpr geometryTerminal(
        CouplingGeometryTerminalQuantity quantity,
        const CouplingGeometryTerminalScope& scope) const;

    CouplingFieldRef field(std::string_view participant,
                           std::string_view field) const;

    CouplingRegionRef region(std::string_view participant,
                             std::string_view region) const;

    CouplingRegionRef sharedRegion(std::string_view name,
                                   std::string_view participant) const;

    SharedRegionRef sharedRegionGroup(std::string_view name) const;

    std::vector<CouplingFormTerminalProvenanceDeclaration>
    terminalProvenanceFor(const forms::FormExpr& residual) const;
};
```

The helper should internally use the existing field-bound Forms helpers such
as `StateField`, `TestField`, symbolic `dt(.,k)`, `t()`, `deltat()`, and
`deltat_eff()`, the public Forms `FormExpr::previousSolution(k)` terminal, plus
public Forms helpers such as `meshDisplacement()`, `meshVelocity()`, and the
geometry-coordinate/Jacobian/normal/measure terminals, so coupling contracts can
author residuals in the same style as physics modules while still going through
the coupling context.
The coupling builder variants require an explicit
`CouplingGeometryTerminalScope` whenever more than one participant, region, or
shared-region side could own the terminal. They may infer owner scope only in a
validated single-participant context. The scope is declaration/provenance
metadata for coupling diagnostics; the returned expression still delegates to
the existing public Forms vocabulary terminal.
Because the existing Forms nodes do not carry coupling owner-scope metadata,
the builder must also record a coupling-owned terminal-provenance side table for
every helper call that creates `previousSolution(...)`, mesh temporal, or
geometry-terminal expressions with coupling metadata. The monolithic builder
attaches `terminalProvenanceFor(residual)` to each `CouplingFormContribution`
before resolution. The side table must use only public Forms identity, keyed by
the node ownership identity or raw node address obtained from
`FormExpr::nodeShared()` for the terminal expression, stored through a const
view if desired. `terminalProvenanceFor()` may use a public Forms metadata hook,
or a public node-identity traversal if Forms exposes one, to find which recorded
terminals are present in the residual.
It must assign `terminal_sequence` in deterministic encounter order after this
matching step. If a Forms transform rebuilds nodes before provenance capture, it
must copy a public provenance tag or provenance must be captured before the
transform. Diagnostics must not try to reconstruct participant, shared-region,
mesh-role, or frame-transform ownership by walking private Forms AST internals.
`previousSolution(...)` carries participant/field information for declaration
and diagnostics, but it has no symbolic-name parameter and lowers to the
existing Forms `PreviousSolutionRef(k)` terminal because that terminal is
trial-field scoped inside the Forms pipeline. The builder or metadata bridge
must reject ambiguous use in mixed forms: the participant/field supplied to
`previousSolution(...)` must match the active trial field for the installed
contribution, or the installed metadata must name the owning trial block that
makes the terminal unambiguous.
`geometryTerminal(...)` carries declaration/provenance owner and location
metadata, but it still delegates to existing Forms geometry terminals and the
measure/side restrictions on the residual remain the authoritative integration
context.

`CouplingFormBuilder` is only an adapter from participant/field/region names
to the existing Forms vocabulary. It should include `FE/Forms/Vocabulary.h` and
use the same public authoring path as ordinary physics modules. It should not
introduce new hidden weak-form operators that obscure the residual being
installed.

Example shape for a physics-specific contract:

```cpp
std::vector<CouplingFormContribution>
SomeCoupling::buildMonolithicForms(const CouplingContext& ctx,
                                   const CouplingFormBuilder& f) const
{
    auto a = f.state("participant_a", "primary", "a");
    auto b = f.state("participant_b", "primary", "b");
    auto w = f.test ("participant_b", "primary", "w");

    auto gamma_a = f.sharedRegion("shared_surface", "participant_a");
    auto gamma_b = f.sharedRegion("shared_surface", "participant_b");
    auto marker = gamma_b.marker;
    auto trace = [](const forms::FormExpr& expr, CouplingInterfaceSide side) {
        return side == CouplingInterfaceSide::Minus ? expr.minus() : expr.plus();
    };

    return {{
        .contribution_name = "participant_a_to_b_interface",
        .origin = "SomeCoupling::buildMonolithicForms",
        .operator_name = "equations",
        .field_uses = {
            {.participant_name = "participant_b", .field_name = "primary"}
        },
        .extra_trial_field_uses = {
            {.participant_name = "participant_a", .field_name = "primary"}
        },
        .residual = ((trace(a, gamma_a.side) - trace(b, gamma_b.side)) *
                     trace(w, gamma_b.side)).dI(marker)
    }};
}
```

The example is intentionally generic. FSI, thermal-interface, and other
physics-specific contracts should use domain-specific participant, field, and
port names inside `Physics/Coupling`, while `FE/Coupling` only provides the
lookup and Forms construction mechanics. If the shared region resolves to a
boundary instead of an interface face, the contract should use `.ds(marker)`;
if it resolves to an interior face, it should use `.dS()`. The weak form should
remain explicit in Forms terms such as side restrictions, normals, jumps,
averages, traces, and penalties.

The shared-region registry must define which participant region corresponds to
the Forms minus side and which corresponds to the plus side for each interface
marker. Contracts should not hard-code `minus()` and `plus()` assumptions
unless their shared-region declaration validated that mapping. The same
metadata should identify whether a participant field is evaluated as a volume
trace, interface field, mortar field, or externally transferred quantity.
`sharedRegion(name, participant)` returns a participant-specific
`CouplingRegionRef` with the marker and side/orientation metadata needed for
Forms authoring. `sharedRegionGroup(name)` returns only the N-participant
collection and validation constraint; it has no group-level FE marker and must
not be used as an integration-measure owner. Forms examples should derive trace
side restrictions from `CouplingRegionRef::side` and must validate that
participant-specific interface markers refer to the same registered interface
topology before using one marker in `.dI(marker)`.

The monolithic builder should resolve returned `CouplingFormContribution`
objects to `ResolvedCouplingFormContribution` before installation.
`CouplingFormContribution` contains declaration-time install intent: a stable
`contribution_name`, diagnostic `origin`, an operator name, primary field uses,
dependency-only extra trial field uses, Forms compiler options, and name-based
geometry-sensitivity declarations. It must not require a contract to inject raw
`FieldId` values into geometry-sensitivity options, extra-trial-field options,
or embedded `forms::SymbolicOptions` AD-mode, geometry-tangent, or
geometry-sensitivity records directly.
To avoid two declaration-time geometry or AD-mode knobs,
`CouplingSymbolicOptionsDeclaration` mirrors only the non-geometry symbolic
tuning fields and does not expose `forms::SymbolicOptions::ad_mode`,
`use_symbolic_tangent`, `geometry_tangent_path`, or
`geometry_sensitivity`. The monolithic builder is the only layer that
constructs the final `systems::FormInstallOptions`: it copies the allowed
symbolic tuning fields into `compiler_options`, leaves
`compiler_options.ad_mode` at its default or mirrors the top-level value
strictly for diagnostics, writes resolved geometry-sensitivity options,
`compiler_options.geometry_tangent_path`, and
`compiler_options.use_symbolic_tangent` from the coupling-level
geometry-sensitivity declaration after name-to-`FieldId` lookup, resolves
`extra_trial_field_uses` into `systems::FormInstallOptions::extra_trial_fields`,
and sets the top-level `systems::FormInstallOptions::ad_mode` from
`CouplingFormInstallOptionsDeclaration::ad_mode`. The embedded
`forms::SymbolicOptions::ad_mode` should remain at its default or be kept
consistent for diagnostics, but it must not become a second declaration-time AD
mode channel because `installFormulation()` consumes the top-level install
option. The resolved form contains the stable contribution name, diagnostic
origin, owning system name, concrete ordered primary `FieldId` list required by
`installFormulation()`, concrete ordered extra trial `FieldId` list mirrored in
`install_options.extra_trial_fields`, and the fully resolved
`systems::FormInstallOptions` that should be passed to that call. This keeps the
existing Forms pipeline responsible for residual decomposition, active field
detection, sparsity, geometry-sensitivity lowering, and Jacobian block
generation while keeping declaration-time field requirements readable in terms
of participants and field names.

`CouplingFormContribution::field_uses` must name the FE unknowns that own
residual rows, test fields, or primary active fields passed as the
`installFormulation()` `fields` span. `extra_trial_field_uses` must name
dependency-only FE unknowns that the residual references as `StateField`,
`DiscreteField`, geometry-sensitivity mesh-motion dependencies, or temporal
field operands but that should not add residual rows for this contribution. The
builder should deduplicate repeated uses within each list, reject ambiguous
aliases and overlap with incompatible roles, and preserve a deterministic order
when producing `ResolvedCouplingFormContribution::fields` and
`ResolvedCouplingFormContribution::extra_trial_fields`.
`CouplingFormContribution::terminal_provenance` is declaration metadata captured
from the `CouplingFormBuilder` side table for the contribution residual. It is
the durable source of participant/field owner scope, mesh-motion role, geometry
location, and frame-transform provenance for coupling-aware terminals. The
resolved contribution preserves that declaration metadata so the installed
Forms/Systems metadata bridge can compare source intent with installed
provenance.
The public Forms/Systems analysis metadata bridge must then verify that the
combined primary and extra-trial field lists cover all test fields,
state-field dependencies, discrete-field uses, geometry-sensitivity
dependencies, and temporal field operands actually present in the installed
residual. It must also report
non-field Forms dependencies that the current vocabulary exposes as first-class
terminals or resolved refs:
parameters, coefficients, boundary functionals, boundary integrals, raw
auxiliary state, `AuxiliaryInput`, `AuxiliaryOutput`, and material-state old/work
references. It must separately report geometry terminals such as mesh
displacement, coordinates, Jacobians, normals, measures, cell metrics, and
cell-domain ids together with the declaration-side region kind, resolved
`analysis::DomainKind`, owner-scope, marker/shared-region, side,
`forms::GeometryConfiguration` value configuration, optional frame-transform
from/to configurations, geometry-revision, and quadrature-policy provenance
needed to validate where those terminals are available. Explicit
`PreviousSolutionRef(k)` history terminals are field/trial-scoped temporal
provenance and must be reported through `CouplingFormTemporalProvenance`, not as
generic named non-field dependencies.
Parameter, coefficient, and material-state records are provenance/data
dependencies, not `analysis::VariableKey` coupling variables unless Systems
promotes them through a public variable model. Boundary functionals, auxiliary
state, `AuxiliaryInput`, and `AuxiliaryOutput` must be adapted to the existing
`analysis::VariableKey` vocabulary when the contract marks the corresponding
`CouplingNonFieldDependencyRequirement` as requiring graph-variable identity.
Forms boundary-integral terminals remain a distinct provenance subtype but map
to `analysis::VariableKind::BoundaryFunctional` for graph purposes until
Analysis grows a separate boundary-integral kind. `GlobalScalar` is
intentionally not a `CouplingFormNonFieldDependencyKind` because the current
Forms vocabulary does not expose a global-scalar terminal; global-scalar
coupling variables enter through Analysis/Systems contribution metadata or
expert install metadata as scoped `analysis::VariableKey` records. These
dependencies are not `CouplingFieldUse` records and must be preserved as
participant/system-scoped name, resolved slot/output-id, byte-offset, provider,
value-type, and location provenance so diagnostics do not misclassify valid
Forms dependencies as missing FE fields or merge location-sensitive boundary and
material-state dependencies from different regions. The metadata bridge must
compare reported non-field provenance against declaration-side
`non_field_dependencies`, using resolved slots, output ids, byte offsets,
provider metadata, and region/domain metadata only as provenance evidence, and
compare only graph-variable dependencies against `dependencies` and
`expected_blocks`.
The resolved contribution and analysis metadata must carry the owning `FESystem`
name because raw `FieldId`s and auxiliary/parameter/material-state slots are
scoped to that system and are not globally unique. If a contribution requests
mesh-motion geometry sensitivity through
`install_options.geometry_sensitivity.mode ==
forms::GeometrySensitivityMode::MeshMotionUnknowns`, the declaration must name
`install_options.geometry_sensitivity.mesh_motion_field` as a `CouplingFieldUse`.
During resolution, the builder must map that field use to a concrete `FieldId`,
write it into the resolved contribution's
`systems::FormInstallOptions::compiler_options.geometry_sensitivity.mesh_motion_field`,
require the field to be bound through the existing
`FESystem::bindMeshMotionField()` path, and require the metadata bridge to report
the same field as structured geometry-sensitivity provenance.
Geometry tangent policy must be validated before building
`systems::FormInstallOptions`. `GeometryConstant` requires no
`mesh_motion_field`; `forms::GeometryTangentPath::SymbolicRequired` and
`SymbolicWithADCheck` are invalid in that mode because there is no geometry
sensitivity tangent to require or check. `use_symbolic_tangent` may still request
the ordinary symbolic tangent path for a geometry-constant nonlinear residual.
`MeshMotionUnknowns` requires `mesh_motion_field`; all
`forms::GeometryTangentPath` values are valid, and `SymbolicRequired` or
`SymbolicWithADCheck` must force the resolved `compiler_options` onto the
symbolic geometry tangent path regardless of the `use_symbolic_tangent` default.

The coupling graph should compare declared dependencies and expected blocks
against metadata reported by a public Forms/Systems analysis bridge. The
current `installFormulation()` result and `MixedKernelPlan` provide lowered
kernel and active-block structure, but the coupling graph also needs public
source-level field, non-field terminal, variable-dependency,
geometry-sensitivity, geometry-terminal, and temporal-symbol provenance.
The required public metadata shape is represented by
`CouplingFormAnalysisMetadata`: installed field
order, contribution name/origin and owning-system provenance, field-use
provenance including geometry-sensitivity field-use summaries, installed
geometry-sensitivity options
including the mesh-motion field, structured geometry-sensitivity provenance,
non-field dependency provenance for parameter, coefficient, boundary-functional,
boundary-integral, auxiliary-state, `AuxiliaryInput`, `AuxiliaryOutput`, and
material-state terminals, variable-dependency provenance using scoped
`analysis::VariableKey` names for monolithic coupling variables, including
Analysis/expert `GlobalScalar` variables,
declaration terminal-provenance records captured from `CouplingFormBuilder`,
temporal-symbol provenance for `dt(.,k)`, time symbols, and
`PreviousSolutionRef(k)` history terminals with active-trial ownership, plus
mesh temporal terminals with owner-scope and `systems::MeshMotionFieldRole`
provenance, geometry-terminal provenance with integration-location, resolved
`analysis::DomainKind`, frame-transform configuration, and geometry-revision
metadata, installed dependency provenance with `analysis::DomainKind`,
matrix/vector/provider evidence, and installed block/domain provenance.
`variable_dependencies` records source-level variable relationships, while
`installed_dependencies` records the resolved installed evidence used for graph
coverage diagnostics. Installed block provenance records contributing domains as
`analysis::DomainKind` values rather than maintaining a parallel set of
domain-specific booleans, and its `has_matrix`/`has_vector` fields are installed
evidence rather than Forms assembly request flags.
The field-level `appears_as_geometry_sensitivity` flag is only a quick summary
for field-use diagnostics; detailed diagnostics must use
`CouplingGeometrySensitivityProvenance`. That record must distinguish at least
ordinary mesh-motion unknown sensitivity from cut/embedded geometry sensitivity
metadata and preserve provenance ids, construction policy, revision keys,
target kind, parent entity, parent geometry DOFs, visible assembly paths,
available sensitivity channels, AD compatibility, sample count, and FE geometry
fields when those are available from Systems/Assembly metadata such as
`CutGeometrySensitivityMetadata`. It should not copy large per-quadrature sample
arrays into FE/Coupling diagnostics unless an execution path explicitly needs
them; the stable provenance id and revision keys are the durable link back to
the owning Assembly metadata.
Forms/Systems may expose the same information through native metadata records,
with `FE/Coupling` adapting those records into this diagnostic shape. The adapter
must preserve enough origin information to map an installed contribution back to
the declaring contract and contribution even when several contracts install into
the same `OperatorTag`, such as `"equations"`.
Because the current `installFormulation()` API only accepts `OperatorTag`,
ordered fields, residual, and `systems::FormInstallOptions`, the implementation
must add one explicit metadata path before claiming full coupling diagnostics:
either extend Forms/Systems install metadata to accept contribution
name/origin/owning-system provenance, return an install record that can be
associated with that provenance, or maintain a Coupling-owned side table keyed by
a stable install handle. Operator tag alone is not enough identity. If a contract
declares an implicit dependency but the installed Forms residual does not
reference that dependency as a `StateField`, report it as a geometry-sensitivity
dependency, or expose it as an `analysis::VariableKey` non-field dependency,
diagnostics should flag the missing dependency before setup.
`FE/Coupling` should not implement this by walking private Forms AST internals
directly.

## Time Advancement Independence

Coupling contracts should be independent of the selected time advancement
method. A contract may state that it needs a time derivative, the current time,
the time step, or the effective time step, but it should not choose BDF,
generalized-alpha, Newmark, Runge-Kutta, or any other concrete scheme.

The ownership split should be:

```text
Physics/Coupling contract:
  declares temporal requirements
  authors symbolic time terms through FE Forms
  does not select a time integrator

Problem or driver layer:
  selects the time advancement method
  manages history/state storage
  supplies the assembly-time temporal context

FE/Systems:
  validates derivative orders and history availability
  lowers symbolic dt(.,k), PreviousSolutionRef(k), time, time-step,
  effective-time-step, mesh-velocity, previous-mesh-velocity,
  predicted-mesh-velocity, and mesh-acceleration symbols during assembly
```

For monolithic coupling, contracts should write time-dependent mathematical
relations with symbolic Forms expressions:

```cpp
auto d = f.state("participant", "primary", "d");
auto d_dot = f.timeDerivative("participant", "primary", "d", 1);
```

The selected transient context then lowers `d_dot` according to the user's
chosen integrator. The same coupling contract should work with any integrator
that supports the derivative orders it declares.
Mesh temporal symbols are not global solver time requirements. They must declare
or infer the owning participant/system and the existing Systems mesh-motion role
that provides them, such as `systems::MeshMotionFieldRole::Velocity`,
`systems::MeshMotionFieldRole::Acceleration`,
`systems::MeshMotionFieldRole::PreviousVelocity`, or
`systems::MeshMotionFieldRole::PredictedVelocity`. In an N-participant contract,
`CouplingTemporalRequirement::mesh_motion_scope` must identify the participant,
region, or shared-region side whose mesh-motion role is being requested. A
shared-region name alone is not enough because the two sides can have distinct
mesh-motion providers and geometry revisions.

For partitioned coupling, contracts should still avoid owning the time
advancement scheme. They may declare which endpoints exchange current,
predicted, or history-backed data, but the partitioned driver owns the
time-step loop, predictor/corrector policy, subcycling policy, history
management, relaxation, and convergence checks.

If the solver needs different time advancement methods for different
participants or fields inside one monolithic solve, that should be represented
by a problem-level temporal policy or field/participant-aware time integration
context. Coupling contracts should only declare what temporal symbols they use;
they should not special-case the chosen scheme.

The default and simplest monolithic configuration should be one shared time
step and one shared time-integration family across all fields. The
infrastructure should not require that, however. A monolithic solve remains
monolithic if, after time discretization, all current synchronization-time
unknowns are solved together in one nonlinear algebraic system:

```text
F(x^{n+1}; history) = 0

x^{n+1} = [
  participant_A_fields^{n+1},
  participant_B_fields^{n+1},
  coupling_fields^{n+1},
  ...
]
```

Different residual blocks may use different temporal stencils as long as they
produce one coherent coupled residual at the synchronization time. For example:

```text
R_A: BDF2 first-order field update
R_B: generalized-alpha or Newmark second-order dynamics
R_C: quasi-static field with no time derivative
R_coupling: current-time constraints or balances
```

This is still monolithic when the current-time unknowns in those residuals are
solved simultaneously and the implicit cross-dependencies are differentiated
in the global Jacobian. It stops being fully monolithic for a coupling term if
that term uses lagged or predicted data instead of the current coupled
unknowns, unless that lagging is deliberately part of the chosen discrete
formulation.

Field- or participant-specific time policies are mathematically defensible
when:

```text
all implicit coupling terms are evaluated at compatible synchronization or
stage points;
derived variables, such as velocities from displacements, are consistent with
the selected temporal policy;
the residual differentiates through all current-time dependencies;
history, stage, predictor, or subcycling data are treated as known data unless
they are promoted to monolithic unknowns;
the coupling graph diagnostics can distinguish implicit dependencies from
external/lagged temporal data.
```

For example, a monolithic ALE FSI solve may combine:

```text
fluid: first-order BDF velocity-pressure residual
solid: second-order structural dynamics residual
mesh motion: quasi-static pseudo-elastic residual
coupling: current-time kinematic and dynamic interface relations
```

The coupling contract should simply declare that it needs current fields and,
if needed, symbolic derivatives such as `dt(d_s,1)`. The problem-level
temporal policy decides how those symbolic derivatives are lowered.

The coupling graph should aggregate temporal requirements across all physics
and coupling contracts:

```text
maximum derivative order used by any contract
fields whose time derivatives are used symbolically
fields whose history values are referenced through PreviousSolutionRef(k)
maximum logical history depth required by Forms or partitioned endpoints
contracts that require time or time-step symbols
contracts that require effective time-step symbols
contracts that require mesh velocity, mesh acceleration, previous mesh velocity,
or predicted mesh velocity symbols, grouped by mesh-motion owner scope and role
contracts that require mesh displacement, coordinate, Jacobian, Jacobian-inverse,
normal, measure, surface-Jacobian, cell-metric, or cell-domain-id geometry
terminals
participants that require driver-supplied history data
```

Validation should fail before assembly if the selected time advancement policy
cannot satisfy a declared temporal requirement. This includes missing
`SystemStateView::u_history` depth for explicit previous-solution terminals and
missing or ambiguous owner-scoped mesh-motion role bindings for mesh temporal
terminals. Geometry-terminal
requirements should fail separately when the selected geometry transaction,
mesh-motion binding, or Assembly metadata cannot provide the requested
mesh-displacement, coordinate, Jacobian, Jacobian-inverse, normal, measure,
surface-Jacobian, cell-metric, or cell-domain-id terminal.

## Coupling Graph

Add a physics-agnostic `CouplingGraph` that is built from all
`CouplingContractDeclaration`s and the `CouplingContext`.

The graph has two explicit stages:

```text
declaration graph:
  built before mode-specific lowering
  validates required participants, base fields, participant-local regions,
  shared regions, additional-field declarations, dependencies, temporal
  requirements, expected blocks, and, when partitioned mode is requested, declared partitioned
  exchanges and group hints

finalized graph:
  rebuilt or augmented after coupling-owned fields are registered,
  Forms residuals are installed, expert hooks return install metadata, and
  partitioned plans are generated
  validates declared topology against actual registered fields, installed
  residual dependencies, installed block metadata, resolved transfer options, and
  partitioned exchange cycles
```

The graph should contain:

```text
participant nodes
field nodes
participant-local region nodes
non-field variable nodes matching scoped `analysis::VariableKey` names
non-field data/provenance requirement nodes for parameters, coefficients,
material state, and boundary-integral provenance that may not be graph variables
geometry-terminal requirement nodes
shared-region nodes
contract type nodes
contract instance nodes
implicit-monolithic dependency edges
external/lagged dependency edges
non-field data/provenance requirement edges
temporal requirement edges
geometry-terminal requirement edges
partitioned exchange edges
expected block edges
additional-field declaration edges
installed dependency/block metadata
```

This setup-time `FE/Coupling::CouplingGraph` must not duplicate or replace the
existing `FE/Analysis/CouplingGraphAnalyzer`. The setup graph validates coupling
contract declarations and resolved plans before `FESystem::setup()`, while the
analysis pass consumes installed `ContributionDescriptor` records and fallback
`FormulationRecord`s after Forms/Systems have lowered contributions. The
coupling graph should consume or adapt that public analysis metadata for
installed block/dependency diagnostics instead of owning a second analysis
model. Because current Analysis summaries expose auxiliary-state,
auxiliary-input, auxiliary-output, boundary-functional, and global-scalar
dependency concepts, the implementation should extend the public Analysis path
where needed so `FormulationRecord`, `ContributionDescriptor`, and
`FE/Analysis/CouplingGraphAnalyzer` preserve and consume all of them as scoped
`analysis::VariableKey` edges. `AuxiliaryInput`, `AuxiliaryOutput`, and
`GlobalScalar` dependencies must use canonical owning-system qualified names and
`analysis::DomainKind::AuxiliaryCoupling` when the dependency is not tied to a
cell, boundary, or interface operator domain.

The graph should validate:

```text
all required participants exist
contract types are registered and contract instance names are unique
each participant is bound to an owning `FESystem` when it references FE-backed
fields, regions, typed `ParameterRegistry` values, raw auxiliary state,
AuxiliaryInput slots, AuxiliaryOutput ids/names, FE quantities, boundary
reductions, or provider-extension endpoints
all required fields exist
all required non-field data/provenance requirements resolve to scoped
Systems provider metadata, and all non-field graph-variable requirements resolve
to scoped Systems/Analysis variable keys
all required participant-local regions exist
all required shared regions exist
additional field declarations are unique by namespace kind, namespace name,
field name, registration target system, scope, and attachment
each implicit-monolithic dependency can resolve to a `StateField`,
Systems-recognized geometry-sensitivity mesh-motion dependency, or supported
non-field `analysis::VariableKey` dependency
external/lagged dependencies are not expected to produce Jacobian blocks
expected nonzero monolithic blocks are supported by installed dependencies, and
non-field expectations are validated as coupling-variable dependencies unless
Systems reports a real matrix block
temporal requirements are supported by the selected temporal policy
geometry-terminal requirements are supported by the selected geometry
transaction, mesh-motion binding, or Assembly metadata path
shared regions are consistently resolved across all contracts
partitioned exchange cycles are visible to the driver when partitioned mode is
requested
partitioned exchange declarations/templates and generated partitioned plans
agree when partitioned mode is requested
partitioned transfers reject `Unspecified`, validate explicit `Identity`
compatibility, and never infer identity from a missing transfer declaration
interface transfer declarations resolve both `systems::InterfaceTransferOptions`
and the distinct interface-entry name, concrete interface-map identity,
interface-search-registry, logical-region, coordinate-configuration, revision,
ownership, search-registry, and state provenance required to apply the transfer
```

This is the main N-multiphysics safety mechanism. It lets the problem builder
reason about all coupling contracts together rather than validating each
contract in isolation.

## Monolithic Builder Lifecycle

The monolithic builder should run before `FESystem::setup()`.

Recommended lifecycle:

```text
1. Create the shared FESystem.
2. Register base physics fields.
3. Build the initial CouplingContext and shared-region registry.
4. Collect CouplingContractDeclaration records from all coupling contracts.
5. Build and validate the declaration-stage CouplingGraph. At this stage,
   validate required base fields, participant-local regions, shared regions,
   graph-variable dependency declarations, non-field data/provenance
   requirements, additional-field namespace and registration-target collisions,
   temporal declarations, and declared partitioned topology when partitioned
   mode is requested, but do not require coupling-owned FieldIds to exist yet.
6. Register validated coupling-owned additional fields, such as coupling
   multipliers, from the declaration records into their resolved target
   `FESystem`s.
7. Refresh/finalize the CouplingContext and CouplingGraph.
8. Validate temporal requirements against the problem-level temporal policy
   if that policy is available before setup.
9. Let base physics modules install their residual forms.
10. Let coupling contracts build Forms-based monolithic contributions.
11. Resolve returned coupling forms to concrete ordered `FieldId` lists, install
    them with `installFormulation()`, and record the returned
    FormsInstaller/MixedKernelPlan block metadata plus source-level
    dependency/temporal metadata from the public Forms/Systems analysis bridge.
12. Let coupling contracts install any expert custom monolithic terms and
    return explicit resolved `CouplingInstallMetadata` records with one
    contribution identity and `analysis::VariableKey` dependency/block
    provenance record per installed custom contribution.
13. Refresh the finalized CouplingGraph with all Forms and expert install
    metadata.
14. Validate expected block coverage.
15. Call FESystem::setup().
```

This keeps the current FE form machinery intact. Coupling contracts author
additional Forms contributions using field-bound Forms symbols, while the
builder performs the concrete `installFormulation()` call. Off-diagonal Jacobian
blocks are produced when residual terms depend on foreign fields as true
`StateField`s or through Systems-recognized geometry-sensitivity dependencies.

For N-multiphysics monolithic runs, the builder should also emit diagnostics
for expected and missing block structure. The diagnostics do not need to know
what any field physically means; they should report only participant, field,
contract, and block information.

## Partitioned Coupling Plan

Partitioned mode should not install cross-physics monolithic residual terms.
Instead, it should return a data-flow plan.

Example structures:

```cpp
struct PartitionedCouplingPlan {
    std::vector<CouplingExchange> exchanges;
    std::vector<CouplingGroupHint> group_hints;
};

class PartitionedCouplingPlanGenerator {
public:
    PartitionedCouplingPlan generate(
        const CouplingContext& ctx,
        std::span<const CouplingContractDeclaration> declarations,
        std::span<const CouplingExchangeDeclaration> exchange_templates) const;
};
```

`CouplingExchangeDeclaration` and `CouplingGroupHint` are available in
`CouplingContractDeclaration`, while `PartitionedCouplingPlan` contains resolved
`CouplingExchange` records. Physics contracts may provide additional
declarative exchange templates from `buildPartitionedExchangeDeclarations()`,
but `FE/Coupling` owns all endpoint, transfer, shared-region, and interface-map
resolution through the finalized `CouplingContext`. Contracts must not return
already resolved executable endpoints, raw `FESystem*` handles, raw `FieldId`s,
or Systems transfer option objects. The generated plan may fill in endpoint
details that were only declared abstractly, but it must not contradict declared
producers, consumers, value descriptors, shared regions, transfer declarations,
or group hints.

Endpoint resolution should reuse existing FE registries instead of becoming a
parallel registry:

```text
Field endpoints:
  participant_name is required.
  endpoint_name is physics-opaque but resolves as a FESystem/FieldRegistry field
  name in the owning participant system

Parameter endpoints:
  participant_name is required for the initial FESystem/ParameterRegistry path.
  endpoint_name is physics-opaque but resolves as a ParameterRegistry key.
  Resolution records the existing params::ValueType from ParameterRegistry::Spec.
  Initial partitioned parameter transfer executors may support only scalar Real
  payloads, but typed parameter endpoints must be represented explicitly and
  rejected unless the selected transfer or driver-owned mapping supports the
  resolved params::ValueType.

AuxiliaryState endpoints:
  participant_name is required.
  endpoint_name is physics-opaque but resolves only as a raw AuxiliaryState
  block. This is an advanced endpoint path.

AuxiliaryInput endpoints:
  participant_name is required.
  endpoint_name is physics-opaque but resolves as an AuxiliaryInputRegistry
  input name/slot. This is the preferred endpoint kind for values flowing into
  auxiliary models.

AuxiliaryOutput endpoints:
  participant_name is required.
  endpoint_name is physics-opaque but resolves as a deployed auxiliary model
  output name or stable deploy-time output id when available. This is the
  preferred endpoint kind for values produced by auxiliary models.

RegionData endpoints:
  participant_name is required.
  endpoint_name is physics-opaque but resolves through FEQuantityRegistry,
  BoundaryReductionService, or another existing FE quantity/reduction provider
  when one already models the data

ExternalBuffer endpoints:
  participant_name is optional. When omitted, the key resolves in the
  context-level driver-owned external-buffer registry. When present, the key
  resolves in that participant's driver-owned external-buffer scope.
  endpoint_name is physics-opaque but resolves through an explicit driver-owned
  external-buffer registry
```

Partitioned plan generation must resolve each `CouplingEndpointRef` into a
`ResolvedCouplingEndpoint` before producing a `CouplingExchange`. The declaration
endpoint carries only the endpoint kind, optional participant scope, registry key,
and temporal slot. The resolved endpoint carries the concrete FE or driver
identity needed for validation and execution: owning system name, non-owning
runtime `FESystem*` when FE-backed, registry-resolved stable endpoint key,
resolved endpoint kind, resolved `FieldId`, `FEQuantityRegistry` id,
`ParameterRegistry` slot and `params::ValueType`, region-data provider kind and
provider key,
BoundaryReductionService functional name plus primary field when used directly,
auxiliary resolution kind, auxiliary block index, AuxiliaryInput slot, or
AuxiliaryOutput stable id plus optional flattened materialized-output slot,
external-buffer descriptor, payload descriptor, registry provider, resolved
participant/global endpoint scope, resolved temporal backing, and layout/registry
revision keys.
`ResolvedCouplingEndpoint::declaration_provenance` is diagnostic/request
provenance only; execution, durable equality, serialization, and cache keys must
use `resolved_kind`, `resolved_participant_name`, `system_name`,
`registry_provider`, `resolved_endpoint_key`, kind-specific resolved ids, value
descriptor, temporal backing, and revision metadata.
Persistent plan identity should use stable names and ids; pointers in
`ResolvedCouplingEndpoint` are execution-only runtime conveniences.
Region-data providers should prefer normalized `FEQuantityDefinition` records
when one already models the value shape and explicit-evaluation capability. If a
region-data endpoint resolves directly through `BoundaryReductionService`, the
resolved endpoint must carry the service's owning primary field and functional
name rather than pretending it has an `FEQuantityRegistry` id. Auxiliary
endpoints must align with the Forms vocabulary: raw `AuxiliaryState` block
access remains distinct from `AuxiliaryInputRegistry` slots and
`AuxiliaryOutput` ids/names, with extension-provided auxiliary keys called out
as extensions rather than folded into raw auxiliary state. Existing Forms
auxiliary refs store `std::uint32_t` indices, while
`AuxiliaryInputRegistry::slotOf(...)`, `FESystem::auxiliaryOutputIdOf(...)`, and
`FESystem::auxiliaryOutputSlotOf(...)` return `std::size_t`. Endpoint resolution
must reject `std::size_t(-1)` lookup failures and any out-of-range narrowing
before storing a slot/id in `std::uint32_t` metadata or passing it to Forms refs.
The plan must also not conflate stable deployed auxiliary-output ids with
flattened materialized output-buffer slots. The stable output id comes from
`auxiliaryOutputIdOf(...)` and is the durable identity currently used by
installed Forms metadata and `AuxiliaryOutputRef` lowering. The flattened slot
comes from `auxiliaryOutputSlotOf(...)` and is optional execution-layout metadata
only when a partitioned transfer reads or writes the materialized auxiliary
output buffer directly. Both values must be range-checked independently, and
durable endpoint identity must use the stable id plus descriptor provenance, not
the flattened slot.

The explicit external-buffer registry must publish a
`CouplingExternalBufferDescriptor` for each driver-owned endpoint key. That
descriptor defines ownership/lifetime, scalar type, access direction, extents,
strides, packing, distribution, supported temporal-slot descriptors, payload
shape, and layout/data revision keys. `DriverOwned` transfers also require a
`CouplingDriverOwnedTransferDescriptor` from the explicit driver-owned transfer
registry, and the resolved plan must preserve that descriptor or equivalent
durable capability/revision metadata. Durable identity uses descriptor content
and revision keys rather than the registry object or runtime lookup handle.
External-buffer keys are resolved in an explicit scope: absent
`participant_name` selects the context-level global driver-owned registry, while
present `participant_name` selects that participant's driver-owned registry.
Duplicate keys are rejected within one scope but may exist in different scopes;
durable endpoint identity includes the external-buffer scope, key, descriptor
content, and descriptor revision keys.
`GeneralTensor` payloads are valid only when the endpoint descriptors and
resolved driver-owned transfer descriptor prove that both endpoints have
compatible shape, packing, indexed temporal-slot
support, read/write access, and a transfer operator that supports the rank-N
payload. For
`GeneralTensor`, the descriptor's extents and packing must agree with
`CouplingValueDescriptor::tensor_extents` and `tensor_packing`.

Opaque coupling ports describe contract-level data flow. Endpoint references
describe how that data flow attaches to existing FE or driver storage.
`CouplingEndpointRef::temporal` is a temporal slot descriptor that describes
which time location the endpoint uses and must map to existing FE state
vocabulary rather than an implicit port-name convention. `Current` means the
current nonlinear iterate or current assembly state (`SystemStateView::u` /
`AuxiliaryFieldStage::CurrentIterate`). `Accepted` means a committed/accepted
state supplied by the owning system or driver (`AuxiliaryFieldStage::Committed`
when applicable). `History` uses a logical 1-based step-back index:
`history_index == 1` means the immediately previous committed state. `Stage` uses
a logical 0-based stage index and is valid only when the selected
time-integration context or auxiliary infrastructure exposes stage state.
`Predicted` is valid only for endpoint registries that explicitly expose a
predictor slot, such as a mesh-motion predicted role, or for driver-owned
external buffers. `External` is only valid for `ExternalBuffer` endpoints whose
temporal meaning is outside FE/Coupling. `temporal.history_index` must be present
only for `History` and must be positive. `temporal.stage_index` must be present
only for `Stage` and must be nonnegative. Both optional indices must be unset for
every other slot, and validation must reject endpoint kinds whose registries
cannot provide the requested temporal slot.
For FE field endpoints, indexed `History` should use `SystemStateView::u_history`
as the canonical backing store. Existing `u_prev` and `u_prev2` spans are
compatibility/convenience views for the first two committed history states; they
must not create separate endpoint slot names or change the logical 1-based
history numbering.
`ResolvedCouplingEndpoint::temporal` must preserve both the declaration request
and the resolved provider backing. For example, a `History(1)` field endpoint
stores a `SystemStateHistory` backing with `storage_index == 0`, an AuxiliaryInput
provider's `PreviousStep` spelling resolves to the same logical `History(1)`
request, `Predicted` endpoints resolve to explicit `SystemStatePredicted`,
`AuxiliaryPredicted`, `InterfaceMapPredicted`, `ExternalBuffer`, or
`ProviderDefined` backings instead of collapsing to current/accepted state,
accepted interface-map endpoints record accepted `InterfaceOperatorState`
provenance, and external buffers record the driver-owned descriptor that
explicitly supports the requested temporal slot.

Endpoint and transfer-state temporal validation must be explicit:

```text
Field:
  Current maps to SystemStateView::u.
  History maps logical history_index == k to SystemStateView::u_history[k - 1].
  Accepted, Predicted, Stage, and External are rejected unless the owning system
  exposes those slots through a named registry extension.

RegionData:
  Current is valid when FEQuantityRegistry, BoundaryReductionService, or another
  registered provider can evaluate the quantity for the current state/context.
  Accepted, History, Stage, and Predicted require explicit provider support for
  that slot. External is rejected.

AuxiliaryState:
  Current maps to the block's work/current storage.
  Accepted maps to the block's committed storage.
  History maps logical history_index == k to AuxiliaryHistoryBuffer::snapshot(k - 1).
  Stage is valid only when the block exposes stage storage for the selected
  auxiliary integration context. Predicted requires matching auxiliary
  storage/registry support. External is rejected.

AuxiliaryInput:
  Current maps to an AuxiliaryInputRegistry value produced from
  AuxiliaryFieldStage::CurrentIterate or a non-field provider current value.
  Accepted maps to AuxiliaryFieldStage::Committed when the input is sampled from
  FE state, or to explicit provider support for accepted values.
  Existing AuxiliaryFieldStage::PreviousStep is an auxiliary-input provider
  spelling of the prior committed FE state and must resolve as History(1) when
  supported, not as a separate FE/Coupling temporal slot.
  Stage maps to AuxiliaryFieldStage::StageState only when the input provider can
  evaluate the requested stage index. Predicted and History require explicit
  provider support. External is rejected.

AuxiliaryOutput:
  Current and Accepted require the deployed auxiliary model/output registry to
  expose those output states. History and Stage require stored output provenance
  for the requested logical index. Predicted requires explicit output-provider
  support. External is rejected.

Interface transfer maps:
  Current uses the map state selected by the current nonlinear/partitioned
  iteration context.
  Accepted requires InterfaceOperatorState metadata that identifies accepted map
  revisions and rejects trial-only maps.
  History and Stage require stored map revision provenance for that slot.
  Predicted requires an explicitly predicted interface-map role. External is
  rejected.

ExternalBuffer:
  Current, Accepted, Predicted, History, Stage, and External are valid only when
  the driver-owned buffer registry declares the requested temporal-slot
  descriptor and its layout.

Parameter:
  Current is valid when ParameterRegistry can evaluate the key for the current
  state/context and the parameter type matches the endpoint value descriptor.
  Endpoint resolution records `params::ValueType` from `ParameterRegistry::Spec`.
  Initial transfer execution may be limited to scalar Real values, but non-Real
  typed parameters must be rejected explicitly unless a registry extension or
  driver-owned transfer declares the shape/type mapping. History, Stage,
  Accepted, Predicted, or External slots require a registry extension that
  explicitly defines those meanings.

GeneralTensor value descriptors:
  GeneralTensor is not an endpoint kind. In the initial infrastructure it is
  valid only for DriverOwned transfers, so temporal semantics come from the
  driver-owned ExternalBuffer endpoint or another explicit driver-owned registry
  entry that declares the requested slot and layout.
```

Interface-to-interface exchanges should reuse `FE/Systems/InterfaceOperators`
when the requested transfer is pointwise interpolation, conservative projection,
or mortar projection. The partitioned plan should carry enough metadata for the
driver to use resolved `systems::InterfaceTransferOptions`, apply the existing
operator, and report `InterfaceTransferDiagnostics`. It must also carry
`CouplingInterfaceMapProvenance` so the driver can recover the source/target
systems, interface search registry, interface entry name, concrete interface
map identity, source/target markers, map revisions, search revisions, logical
source/target interface region ids, source/target coordinate configurations,
`InterfaceRevisionSnapshot` records, search map state, sliding map kind,
operator state, accepted/trial revision keys, and epoch used to build the
transfer. Execution-only `FESystem*`,
`InterfaceSearchRegistry*`, `SlidingInterfaceMap`, and
`svmp::search::InterfaceMap` handles are resolved from that provenance and the
current `CouplingContext`; they are not part of durable plan identity. The
driver passes the resolved `svmp::search::InterfaceMap` to
`applyInterfaceTransfer()`; the `SlidingInterfaceMap` handle is retained only
when the driver also needs Systems accepted/trial map-state metadata. The
declaration should keep stable FE/Coupling transfer metadata separate from the
resolved Systems options object. FE/Coupling should not invent a second registry
for these interface transfer families.

For a concrete physics-specific coupling, the exchange plan might contain:

```text
contract.port_a -> contract.port_b
contract.port_c -> contract.port_d
```

The actual partitioned driver should live above `FE/Systems`. The FE coupling
layer should define the plan and data-flow requirements, not own every solver
loop policy.

### Partitioned Driver Policy

Partitioned iteration strategy is a driver-level concern, not an FE coupling
concept. Algorithms such as staggered Gauss-Seidel, Jacobi, Aitken relaxation,
or quasi-Newton acceleration should be configured in the higher-level
partitioned driver or simulation input.

The FE coupling plan should answer:

```text
what endpoints exchange data
where the exchange is scoped
what value shape is exchanged
which transfer declaration is requested and which resolved transfer options apply
which participants are suggested as strongly coupled groups
```

The driver should answer:

```text
solve order
outer iteration algorithm
relaxation or acceleration method
convergence norms and tolerances
maximum coupling iterations
state commit/rollback policy
```

Group hints are not solver algorithms. They are topology metadata. A
partitioned driver may use them to avoid splitting a strongly coupled subset of
participants too aggressively, but the driver remains responsible for the
actual solve order and iteration scheme.

## Physics-Specific Couplings

Physics-specific coupling contracts should live under `Physics/Coupling`.

The first implementation should be FSI because it exercises the important
cases:

```cpp
struct FSILagrangeMultiplierOptions {
    bool enabled = false;
    // Contract-owned namespace. Empty means use FSICouplingOptions::contract_name.
    std::string contract_field_namespace;
    // Optional participant whose FESystem receives the multiplier. If unset,
    // monolithic resolution must prove the interface participants share one
    // FESystem before registering the field.
    std::optional<std::string> system_participant_name;
    std::string field_name = "lambda";
    std::shared_ptr<const spaces::FunctionSpace> space;
    int components = 0;  // 0 means infer from the multiplier space.
    std::optional<std::string> shared_region_name;
    bool require_mortar_compatible_trace = true;
};

struct FSICouplingOptions {
    CouplingMode mode = CouplingMode::Monolithic;
    std::string contract_name = "fsi";
    std::string fluid_name = "fluid";
    std::string solid_name = "solid";
    std::optional<std::string> mesh_name;
    std::string interface_name;
    FSILagrangeMultiplierOptions multiplier{};
    CouplingTransferDeclaration solid_to_fluid_transfer{};
    CouplingTransferDeclaration fluid_to_solid_transfer{};
};
```

The multiplier option should default to `enabled = false`. Enabling it must
provide an explicit multiplier-space policy and validate the selected multiplier
space, optional contract-owned field namespace override, registration target
`FESystem`, field name, component count, shared-region or participant-region
attachment, trace/mortar compatibility, and expected block metadata rather than
relying on a hidden default. The empty namespace is the normal case and resolves
to `FSICouplingOptions::contract_name`.

Partitioned iteration algorithm, relaxation, tolerances, and maximum coupling
iterations should be owned by the partitioned driver configuration rather than
`FE/Coupling`.

### FSI Monolithic Behavior

The FSI contract should:

```text
1. Look up fluid velocity and pressure fields.
2. Look up solid displacement and/or velocity fields.
3. Look up mesh displacement if ALE is enabled.
4. Optionally declare an interface multiplier field.
5. Install interface kinematic and dynamic coupling terms.
```

For example, with fields:

```text
u_f      fluid velocity
p_f      fluid pressure
d_s      solid displacement
v_s      solid velocity
d_m      mesh displacement
lambda   optional interface traction multiplier
```

the coupled unknown vector with the multiplier enabled is:

```text
x = [u_f, p_f, d_s, v_s, d_m, lambda]
```

The FSI coupling contract should own terms such as:

```text
d_m = d_s on Gamma_FS
u_f = v_s on Gamma_FS, or u_f = dt(d_s) if velocity is derived
traction balance on Gamma_FS
```

It should not own the fluid volume ALE residual.

If the contract uses `dt(d_s)` rather than an independent solid velocity
field, it should declare a first-derivative temporal requirement on the solid
displacement field and author the relation with `CouplingFormBuilder` symbolic
time-derivative helpers. The selected time integrator should lower that
symbolic derivative during transient assembly.

### FSI Partitioned Behavior

In partitioned mode, the same FSI contract should produce a plan similar to:

```text
for each time step:
  initialize interface data

  for each coupling iteration:
    solve fluid using latest wall velocity / mesh displacement
    transfer fluid traction to solid

    solve solid using latest fluid traction
    transfer solid displacement / velocity to fluid and mesh motion

    check interface residual
```

The plan should define contract-owned exchange ports, declaration-time transfer
metadata, resolved transfer options, and endpoint temporal slots. A partitioned
driver should execute the plan and own convergence criteria.

FSI-owned port names could include:

```text
Physics/Coupling/FSI:
  "solid_displacement"
  "solid_velocity"
  "fluid_traction"
  "mesh_displacement"
```

The FE layer still treats these as opaque port names.

## ALE Mesh Motion

ALE mesh motion should be treated as a normal physics participant, not as
hidden logic inside the FSI coupling module.

For monolithic ALE FSI, the recommended decomposition is:

```text
FluidALEModule:
  owns R_fluid(u_f, p_f, d_m)

SolidModule:
  owns R_solid(d_s, v_s)

MeshMotionModule:
  owns R_mesh(d_m)

FSICouplingModule:
  owns R_interface(u_f, d_s, v_s, d_m, lambda)
```

The key tangent block is:

```text
J_fluid,mesh = d R_fluid / d d_m
```

That block can only be produced if the fluid ALE module represents `d_m` as a
true implicit dependency, either as an explicit `StateField` or through the
existing Systems geometry-sensitivity mesh-motion path. An FSI interface coupler
cannot recover this volume derivative after the fact.

Therefore:

```text
The coupling module can enforce interface relationships involving mesh motion.
The ALE-aware fluid module must own the fluid-volume dependency on mesh motion.
```

## N-Multiphysics Coupling

N-multiphysics support should be first-class. The infrastructure should not
assume that a coupling contract connects exactly two participants.

The coupling-module approach should scale through a sparse contract graph:

```text
participant nodes:
  participant A
  participant B
  participant C
  participant D

contract instance nodes:
  instance 1 of contract type X connects A, B
  instance 2 of contract type X connects A, C
  instance 3 of contract type Y connects B, C, D
```

Avoid a single global N-way contract unless the physics model is inherently
global. Prefer explicit contracts with clear participant, field, shared-region,
dependency, and exchange declarations. At the same time, the API must allow a
contract to reference more than two participants when the relationship is truly
multiway.

Each coupling contract should declare:

```text
required and optional participants
required and optional fields
required and optional participant-local regions
required shared regions
additional fields it declares
monolithic residual rows it contributes
implicit monolithic dependencies, resolved as `StateField` or
geometry-sensitivity dependencies
external/lagged dependencies
partitioned ports/channels it exchanges
expected monolithic block dependencies
partitioned group hints
```

The global coupling graph should then validate all contracts together. This is
necessary because a contract can be valid in isolation while the assembled
N-multiphysics problem is not. Examples include duplicate field declarations,
multiple contracts using inconsistent shared-region names, or expected
monolithic block dependencies that no installed residual can produce.

### Pairwise And Multiway Contracts

Pairwise contracts are appropriate for simple relationships between two
participants. Multiway contracts are appropriate when one coupling relationship
requires several participants at once.

The FE API should support both forms through the same declaration interface:

```text
pairwise:
  participants = [A, B]

multiway:
  participants = [A, B, C, ...]
```

The contract implementation decides whether its monolithic terms or
partitioned exchanges are pairwise or multiway. The FE layer only validates the
declared participant, field, endpoint, shared-region, and dependency topology.

### Thermo-Fluid-Structure Example

A thermo-fluid-structure simulation may combine several physics-specific
contracts:

```text
participants:
  fluid
  solid
  mesh_motion
  fluid_thermal
  solid_thermal

contracts:
  FSI mechanical interface
  thermal interface
  solid-to-mesh motion
  fluid thermal advection/property coupling
  solid thermal strain/property coupling
```

Some of those contracts are cleanly pairwise. Others may be multiway. For
example, a fluid thermal residual may depend on flow state and mesh motion, and
a solid mechanical residual may depend on a thermal state. In monolithic mode,
those dependencies must be represented as implicit monolithic dependencies,
resolved as `StateField` dependencies or Systems-recognized
geometry-sensitivity dependencies by the module or contract that owns the
residual row.

This reinforces the core rule:

```text
Coupling contracts can add residual terms and cross-participant relationships.
The residual owner must still expose every true implicit dependency needed for
consistent monolithic tangents.
```

### Partitioned N-Multiphysics

For N-multiphysics partitioned runs, a flat exchange list is not enough. The
plan should include an exchange graph plus optional group hints.

Example driver-level grouping:

```text
group 1: participants A and C
group 2: participant B
group 3: participant D
```

The FE layer should not choose the solve algorithm. It should only expose
enough graph metadata for the driver to choose a reasonable partitioned
strategy, detect cycles, and report which exchanges must be satisfied during
each outer coupling iteration.

### Shared-Region Registry

N-multiphysics coupling commonly reuses the same geometric relationship across
several contracts. A shared-region registry should let multiple contracts refer
to the same named shared region.

The FE layer should validate:

```text
all contracts resolve the shared region name consistently
each participant has a region mapping when required
region kinds are compatible with the requested assembly or exchange
interface side ownership uses typed minus/plus metadata and is unambiguous
duplicate shared-region declarations agree
```

The physical interpretation of the shared region remains in
`Physics/Coupling`.
Any shared-region `required_region_kind` is a duplicate-declaration and
consumer-compatibility constraint only. It must agree with every participant
mapping used for a concrete FE operation, but it must not be treated as an FE
marker or as the resolved integration kind for all participants.

## Consistent Tangents

The coupling-module approach can produce the same consistent monolithic tangent
as a single residual-composer design if every residual contribution includes
all of its true implicit dependencies.

Mathematically:

```text
R_total = R_A + R_B + R_C + ... + R_contract_1 + R_contract_2 + ...

dR_total/dx =
  dR_A/dx
+ dR_B/dx
+ dR_C/dx
+ ...
+ dR_contract_1/dx
+ dR_contract_2/dx
+ ...
```

So it does not matter whether residual terms are installed by one composer or
several coupling/physics modules, as long as each contribution is differentiated
with respect to all monolithic state fields it truly depends on.

The infrastructure should make this enforceable by requiring modules to
declare:

```text
owned residual rows
implicit monolithic dependencies, resolved as `StateField` or
geometry-sensitivity dependencies
external/lagged dependencies
additional coupling fields
expected Jacobian block dependencies
```

For N-multiphysics monolithic runs, expected block diagnostics should be a
first-class output of the coupling graph. The diagnostics should answer:

```text
which residual rows are owned by which participant or contract
which dependencies are implicit
which dependencies are external/lagged
which block pairs are expected to be nonzero
which expected block pairs were not installed
which installed block pairs were not declared
```

This makes it possible to catch a common N-multiphysics failure mode: a
coupling appears in the data model, but one residual owner treated a dependency
as lagged data, so the monolithic Jacobian is missing the corresponding block.

## Validation Rules

Validation should happen before `FESystem::setup()`.

Global coupling-graph validation should check:

```text
participant declarations are unique and resolvable
contract types are registered and contract instance names are unique
FE-backed participants are bound to owning `FESystem` instances before raw
`FieldId`s, markers, interface topology, or Forms symbols are resolved
required fields are present
required non-field data/provenance requirements are present or explicitly
optional, and required non-field graph variables adapt to
`analysis::VariableKey`
required participant-local regions are present
required shared regions are present
additional field declarations do not collide across participant and
contract-owned namespaces
additional field declarations have valid namespace kind, namespace name, and
registration target system policy
additional field declarations can lower to valid systems::FieldSpec or interface-field
registrations
interface additional field declarations resolve markers through participant or
shared-region context rather than declaration-time raw markers
additional field component counts are inferred from or match the function space
Forms-authored monolithic coupling fields belong to the same FESystem and have
compatible region/interface topology
interface-face shared regions reference registered interface meshes/topology
interface shared-region side ownership is typed, explicit, and unambiguous
implicit-monolithic dependencies are available as monolithic fields or Systems-recognized
geometry-sensitivity mesh-motion fields, or as supported non-field variables
that adapt to `analysis::VariableKey`
auxiliary-input and auxiliary-output dependencies reported by
`FormulationRecord` or `ContributionDescriptor` are visible to
`FE/Analysis/CouplingGraphAnalyzer` rather than silently dropped from fallback
analysis
external/lagged dependencies are not expected to produce Jacobian blocks
temporal derivative requirements are supported by the selected time policy
explicit previous-solution/history requirements are supported by available
`SystemStateView::u_history` depth and Forms/JIT history limits
mesh temporal requirements are supported by bound mesh-motion role fields or
provider metadata
geometry-terminal requirements are supported by the selected geometry
transaction, mesh-motion binding, and Assembly metadata capabilities
time-step and effective-time-step requirements are supported by the selected
temporal context
required history/state data can be supplied by the driver
partitioned exchange endpoints are resolvable when partitioned mode is
requested
partitioned exchange endpoints resolve through existing FE registries/providers
or explicit driver-owned registries whenever they refer to fields, typed
ParameterRegistry values, raw auxiliary state, AuxiliaryInput slots,
AuxiliaryOutput ids/names, FE quantities, boundary reductions,
provider-extension keys, external buffers, or driver-owned transfers
partitioned endpoint names remain physics-opaque labels but resolve as concrete
registry/provider keys for their endpoint kind
partitioned endpoint refs lower to resolved endpoint records with kind-specific
ids, resolved temporal backing, layout/revision metadata, and external-buffer
descriptors when applicable
partitioned endpoint temporal slots are valid for their endpoint kind, use
logical 1-based history indices, require `history_index` only for History, and
require `stage_index` only for Stage
partitioned exchange declarations/templates match generated partitioned plans
when partitioned mode is requested
external-buffer endpoints validate scalar type, access direction, lifetime,
distribution, extents, strides, packing, supported indexed temporal-slot
descriptors, payload shape, and layout/data revision keys
transfer declarations are explicit; `Unspecified` never appears in a generated
or executable resolved exchange
explicit `Identity` transfers have compatible endpoint ownership, layout, value
shape, and temporal slot semantics
interface transfer declarations map to existing FE interface operators and
resolve to complete `systems::InterfaceTransferOptions` when partitioned mode is
requested
interface transfer resolutions include `CouplingInterfaceMapProvenance` with
source/target systems, source/target interface markers, interface search
registry identity, interface entry name, concrete interface map name,
logical interface region ids, coordinate configurations, revision snapshots,
revision keys, search map state, sliding-map kind, operator state, and
accepted/trial epoch. The driver resolves that durable metadata to
runtime system, search-registry, sliding-map state, and concrete
`svmp::search::InterfaceMap` handles from the current context
interface transfer resolution rejects stale maps, wrong-system maps, and
trial/accepted state mismatches before execution
SymmetricTensor interface transfers are explicitly packed as Rank2Tensor payloads
or rejected
`GeneralTensor` payloads are rejected for FE interface transfers and accepted
only with explicitly validated `CouplingTransferKind::DriverOwned` transfers
driver-owned transfer declarations are explicitly user-provided when partitioned
mode is requested
group hints reference known participants when partitioned mode is requested
expected block declarations are internally consistent
expert monolithic install hooks return resolved block/dependency metadata with
scoped `analysis::VariableKey` identity and `analysis::DomainKind` provenance
expert monolithic install hooks use approved Systems extension points
```

FSI monolithic ALE should require:

```text
fluid.velocity
fluid.pressure
solid.displacement
mesh.displacement
fsi shared-region/interface topology declaration
```

If the coupling uses a multiplier, it should declare and validate:

```text
contract-owned <contract_name>.lambda, where <contract_name> is the FSI contract
instance namespace and not a participant
multiplier function space
component count matching interface dimension
shared-region or participant-region attachment policy and registration target
FESystem policy
trace or mortar compatibility
```

FSI partitioned should require:

```text
FSI-defined output port for fluid traction or load
FSI-defined input port for solid traction or load
FSI-defined output port for solid displacement or velocity
FSI-defined input port for fluid or mesh displacement/velocity data
transfer declaration in each direction, resolved to Systems transfer options when
applicable
temporal slot descriptor for each partitioned endpoint
driver-visible convergence port/endpoints if the selected driver needs them
```

Validation should also check:

```text
component counts
field existence
region/interface existence
compatible monolithic FESystem ownership
partitioned systems and transfer ownership
duplicate coupling field declarations
unsupported mode/transfer combinations
```

## Testing Plan

Start with FE-agnostic tests:

```text
Code/Source/solver/FE/Tests/Unit/Coupling/test_CouplingContext.cpp
Code/Source/solver/FE/Tests/Unit/Coupling/test_CouplingContractValidation.cpp
Code/Source/solver/FE/Tests/Unit/Coupling/test_CouplingFormBuilder.cpp
Code/Source/solver/FE/Tests/Unit/Coupling/test_CouplingGraph.cpp
Code/Source/solver/FE/Tests/Unit/Coupling/test_CouplingDiagnostics.cpp
Code/Source/solver/FE/Tests/Unit/Coupling/test_PartitionedCouplingPlan.cpp
Code/Source/solver/FE/Tests/Unit/Coupling/test_SharedRegionRegistry.cpp
Code/Source/solver/FE/Tests/Unit/Coupling/test_CouplingTemporalRequirements.cpp
Code/Source/solver/FE/Tests/Unit/Coupling/test_CouplingGeometryRequirements.cpp
```

Then add physics-specific coupling tests:

```text
Code/Source/solver/Physics/Tests/Unit/Coupling/test_FSICouplingModule.cpp
```

Suggested test coverage:

```text
1. CouplingContext resolves fields by participant and field name.
2. Missing fields produce clear validation failures.
3. FSI monolithic mode declares the expected multiplier field.
4. FSI monolithic mode installs expected interface residual terms.
5. Monolithic assembly produces expected off-diagonal block presence.
6. FSI partitioned mode returns the expected exchange plan.
7. ALE FSI validation fails if mesh displacement is missing.
8. Partitioned validation fails if no transfer declaration is provided.
9. CouplingGraph accepts N-participant contracts.
10. CouplingGraph rejects missing required participants, participant-local
    regions, or shared regions.
11. Expected block diagnostics detect missing implicit dependencies.
12. Partitioned plans preserve exchange cycles and group hints.
13. Forms-authored coupling terms using symbolic `dt(.,k)` lower through the
    selected transient context.
14. Temporal validation fails when the selected policy cannot satisfy a
    declared derivative order.
15. Changing the time integrator does not require changes to the coupling
    contract.
16. Interface shared regions map to the correct `.dI(marker)` measure and
    preserve side/orientation and interface revision metadata.
17. Forms dependency diagnostics use a public Forms/Systems metadata bridge
    that exposes block, `StateField`, `DiscreteField`, parameter, coefficient,
    boundary-functional, boundary-integral, auxiliary-state, `AuxiliaryInput`,
    `AuxiliaryOutput`, material-state, structured geometry-sensitivity,
    geometry-terminal location, resolved `analysis::DomainKind`, owner-scope,
    frame-transform configuration provenance, revision,
    `analysis::VariableKey`, global-scalar variable-dependency, and
    temporal-symbol provenance including field/trial-scoped
    `PreviousSolutionRef(k)` and mesh temporal owner-scope plus
    `systems::MeshMotionFieldRole` provenance rather than private AST
    traversal.
18. Expert monolithic hooks fail validation if they do not return resolved
    dependency and block metadata using scoped `analysis::VariableKey` identity.
19. Partitioned exchange declarations/templates and generated partitioned plans
    are compared for consistency.
20. Forms-authored monolithic interface coupling fails validation unless all
    fields are in one compatible `FESystem` and the interface topology is
    registered.
21. Partitioned endpoints resolve through existing registries/providers or
    explicit driver-owned registries whenever they reference fields, typed
    ParameterRegistry values, raw auxiliary state, AuxiliaryInput slots,
    AuxiliaryOutput ids/names, FE quantities, boundary reductions,
    provider-extension keys, external buffers, or driver-owned transfers, and
    generated exchanges carry resolved endpoint records with temporal backing
    instead of declaration refs.
22. Additional-field declarations infer component counts from function spaces,
    reject negative component counts, and reject explicit component mismatches.
23. Resolved interface transfers preserve `systems::InterfaceTransferOptions`
    and reject unsupported symmetric-tensor payloads without explicit
    `Rank2Tensor` packing.
24. Forms contribution resolution rejects missing field uses based on public
    Forms/Systems field, structured geometry-sensitivity, non-field variable,
    builder terminal side-table, explicit field/trial-scoped previous-solution,
    geometry-terminal, and temporal provenance metadata.
25. Partitioned endpoint temporal slots validate current, accepted, predicted,
    history, stage, and external data requirements, including logical 1-based
    history indices mapped to FE/Auxiliary 0-based storage.
26. Driver-owned general tensor transfers require explicit extents and packing,
    while FE interface transfers reject general tensors.
27. Forms contribution diagnostics preserve contribution name, origin, and
    owning-system provenance so two contracts installed into the same operator
    tag remain distinguishable without treating raw `FieldId`s as global.
28. Interface transfer provenance rejects stale interface maps, wrong-system
    maps, and trial/accepted state mismatches, preserves logical interface
    region ids, coordinate configurations, revision snapshots, search map state,
    interface entry name, interface map name, and sliding-map kind, and runtime
    handles include the actual search registry and interface map object used for
    transfer execution.
29. CouplingGraph diagnostics include AuxiliaryInput, AuxiliaryOutput, and
    GlobalScalar dependencies through the public Analysis fallback path, not only
    through direct Coupling-owned metadata.
30. Temporal and geometry diagnostics validate explicit `PreviousSolutionRef(k)`
    history depth, mesh temporal terminals, mesh displacement, and Forms geometry
    terminals independently from derivative-order requirements.
31. True 2D vector interface transforms preserve source embedding and target
    restriction policies separately from `systems::InterfaceTransferOptions` and
    fail validation unless a coupling-side execution adapter applies them.
32. Non-field dependency diagnostics preserve location-sensitive provenance so
    parameter, coefficient, boundary-functional, boundary-integral, auxiliary,
    and material-state requirements with the same name but different regions,
    domains, providers, slots, ids, or byte offsets do not collide.
33. Geometry tangent validation rejects invalid `GeometryConstant` combinations,
    accepts ordinary symbolic tangents independently from geometry sensitivity,
    and requires bound mesh-motion fields for mesh-motion geometry sensitivity.
34. The public Forms/Systems scanner or metadata bridge covers the required
    Forms vocabulary terminals directly instead of relying on private AST access.
```

The first integration test should be intentionally small: two scalar fields on
a simple mesh with an interface coupling term. The goal is to prove the
coupling infrastructure creates cross-field Jacobian structure before adding
full FSI physics.

## Implementation Milestones

### Milestone 0: Forms/Systems Metadata Bridge Spike

Before implementing installed-form coupling diagnostics, complete a focused
metadata bridge spike.

Deliverables:

```text
audit of current Forms/Systems scanner, structure analyzer, install metadata,
and Analysis records
chosen public bridge location and ownership boundary
normalized terminal record shape with explicit terminal kind, provider identity,
optional graph-variable identity, owner scope, location, and contribution
provenance
scanner-normalization rules for historical or overloaded scanner containers
minimal synthetic bridge fixture for expected-block/dependency diagnostics
feature gates for metadata categories that remain unavailable
```

### Milestone 1: FE Coupling Vocabulary And Context

Add:

```text
FE/Coupling/CouplingTypes.h
FE/Coupling/CouplingContext.h
FE/Coupling/CouplingFormBuilder.h
FE/Coupling/CouplingTemporalRequirements.h
FE/Coupling/CouplingGeometryRequirements.h
FE/Coupling/SharedRegionRegistry.h
FE/Coupling/TransferPlan.h
FE/Coupling/PartitionedCouplingPlan.h
FE/Coupling/PartitionedCouplingPlanGenerator.h
```

Deliverables:

```text
field references
region references
opaque coupling ports/channels
opaque coupling endpoints
coupling mode enum
generic monolithic variable references aligned with scoped `analysis::VariableKey`
transfer declarations and resolved transfer options
shared-region references
Forms symbol construction helper
symbolic time derivative construction helper
temporal requirement descriptors
geometry-terminal requirement descriptors
basic validation errors
```

### Milestone 2: Coupling Contract Declarations

Add:

```text
FE/Coupling/CouplingDeclaration.h
FE/Coupling/CouplingContract.h
FE/Coupling/CouplingRegistry.h
```

Deliverables:

```text
contract interface
N-participant contract declarations
required/optional participant declarations
field, participant-local region, and shared-region requirements
additional-field declarations
partitioned exchange and group-hint declarations
implicit-monolithic versus external/lagged dependency declarations
temporal requirement declarations
effective-time-step requirement declarations
explicit history and mesh temporal requirement declarations
mesh displacement and geometry-terminal requirement declarations
expected block and non-field dependency declarations
Forms-based monolithic contribution API
expert install metadata API
contract registration
contract validation
mode-specific dispatch
```

### Milestone 3: Coupling Graph And Diagnostics

Add:

```text
FE/Coupling/CouplingGraph.h
FE/Coupling/CouplingDiagnostics.h
```

Deliverables:

```text
global graph built from all contract declarations
declaration-stage and finalized graph validation
participant/field/shared-region validation
partitioned exchange graph
expected block diagnostics
missing field and non-field dependency diagnostics
public Forms/Systems coupling-analysis metadata consumption
Analysis fallback consumption for AuxiliaryInput/AuxiliaryOutput/GlobalScalar
dependencies
temporal requirement diagnostics
participant-local region diagnostics
group hint validation
```

### Milestone 4: Monolithic Coupling Builder

Add:

```text
FE/Coupling/MonolithicCouplingBuilder.h
```

Deliverables:

```text
pre-setup lifecycle for coupling fields
coupling graph validation before setup
Forms-based monolithic contribution collection
symbolic temporal term collection
monolithic install dispatch
Forms/Systems coupling-analysis metadata recording
monolithic `analysis::VariableKey` dependency recording
expert install metadata recording
field/context finalization
diagnostics for installed coupling blocks
```

### Milestone 5: Partitioned Plan Generation

Extend:

```text
FE/Coupling/PartitionedCouplingPlan.h
FE/Coupling/PartitionedCouplingPlanGenerator.h
FE/Coupling/TransferPlan.h
```

Deliverables:

```text
exchange graph
producer/consumer port metadata
producer/consumer endpoint metadata
transfer declaration metadata
resolved transfer option metadata
distinct interface-entry and interface-map ownership, revision, and state
provenance through the interface-search-registry
group hint metadata
exchange cycle visibility for the driver
```

This milestone does not need to implement the full partitioned driver.

### Milestone 6: Initial Physics Coupling Contracts

Add:

```text
Physics/Coupling/FSICouplingModule.h
Physics/Coupling/FSICouplingModule.cpp
Physics/Coupling/ThermalInterfaceCouplingModule.h
Physics/Coupling/ThermalInterfaceCouplingModule.cpp
```

Deliverables:

```text
FSI options
field and interface validation
monolithic interface terms
Forms-authored coupling residual examples
partitioned exchange plan
ALE mesh-motion requirements
N-participant declaration examples
thermal interface contract example
```

### Milestone 7: Minimal Integration Tests

Add tests for:

```text
coupling context lookup
shared-region lookup
N-participant coupling graph validation
FSI validation
partitioned FSI exchange plan
partitioned group hints
monolithic off-diagonal block presence
expected block diagnostics
Forms dependency diagnostics
ALE missing-mesh validation failure
```

## Implementation Phase Checklists

These checklists define the completion criteria for the coupling-module
infrastructure. An item should only be checked when the implementation exists,
is wired into the relevant build/test target, and has focused unit-test
coverage. The phase is not complete until all implementation and verification
items in that phase are checked.

Metadata bridge readiness is a phase gate. Coupling graph dependency
diagnostics, expected-block validation, geometry-terminal validation against
installed Forms evidence, temporal-terminal diagnostics, and non-field
dependency diagnostics must not be marked complete until a public Forms/Systems
metadata bridge exposes the terminal and provenance coverage required by this
plan. Until then, those validators may exist only as declaration-side checks or
synthetic-fixture tests.

MVP reject policy:

```text
true 2D FE interface frame transforms:
  reject unless a coupling-side adapter applies source embedding and target
  restriction around Systems transfer execution

driver-owned GeneralTensor transfer execution:
  validate descriptors and resolved metadata only; reject execution unless both
  the named driver-owned transfer operator and an executable driver adapter are
  explicitly registered

non-Real parameter endpoint transfer:
  reject on the built-in FE transfer path; accept only through an explicit typed
  parameter extension or driver-owned transfer mapping

cut/embedded geometry-sensitivity provenance:
  report when public Systems/Assembly metadata is available; otherwise reject
  requirements that depend on it rather than collapsing the provenance to a
  boolean field-use flag
```

Implementation risk burn-down gates:

```text
metadata bridge gate:
  run a bridge spike first to audit current Forms/Systems evidence, choose the
  public bridge location, and define the normalized terminal record; then
  implement only the first bridge increment needed by the generic monolithic
  fixture before enabling installed-form dependency, expected-block,
  temporal-terminal, geometry-terminal, or geometry-sensitivity diagnostics
  against real installed forms; until then, those diagnostics are limited to
  declaration-side checks or synthetic bridge fixtures

scanner normalization gate:
  adapt current scanner/analyzer outputs into explicit terminal-kind records,
  including distinct boundary-functional and boundary-integral provenance, before
  coupling graph code consumes non-field dependency evidence

minimal monolithic gate:
  the first executable slice is complete only when a generic two-participant
  monolithic fixture installs through `installFormulation()`, records bridge
  metadata, and verifies one cross-field block/dependency without FSI-specific
  FE vocabulary

partitioned execution gate:
  Phase 6 produces inspectable, durable, metadata-only partitioned plans; it must
  not expose a partitioned nonlinear driver or execute driver-owned,
  true-2D-adapted, or non-Real parameter transfers unless the corresponding
  adapter/driver extension is explicitly implemented and unit tested

geometry-sensitivity gate:
  mesh-motion geometry sensitivity may use current Systems support once exposed
  by the bridge; cut/embedded geometry-sensitivity requirements remain rejected
  unless public Assembly metadata is available and adapted to structured
  provenance
```

### Phase 0: Repository Integration And Guardrails

Implementation checklist:

- [x] Create the `Code/Source/solver/FE/Coupling/` directory with the planned
  public headers and source files using the existing solver namespace and file
  organization conventions.
- [x] Create the `Code/Source/solver/Physics/Coupling/` directory with the
  planned physics-specific coupling module files.
- [x] Add all new FE coupling files to the solver build system without
  introducing dependencies from `FE/Coupling` to `Physics/Coupling`.
- [x] Add all new physics coupling files to the physics build target without
  forcing applications that do not use coupling to instantiate coupling
  contracts.
- [x] Add coupling unit-test targets under `Code/Source/solver/FE/Tests/Unit`
  and `Code/Source/solver/Physics/Tests/Unit` following the existing test
  layout.
- [x] Define a small common diagnostics/error type or reuse an existing one so
  coupling setup failures include contract, participant, field, region, and
  endpoint names.
- [x] Confirm `FE/Coupling` headers use only physics-agnostic API vocabulary
  and do not define FE-owned types, enum values, methods, diagnostics, or
  defaults named for FSI, fluid, solid, thermal, traction, pressure,
  temperature, displacement, or other physical field meanings.
- [x] Confirm all public APIs avoid global singleton state except for an
  explicitly-owned registry passed through setup code.
- [x] Confirm Forms-facing code uses the canonical public authoring surface:
  `FE/Forms/Vocabulary.h` and `FE/Systems/FormsInstaller.h`.
- [x] Confirm `FE/Coupling` does not include or depend on private Forms
  implementation details for dependency or temporal analysis.
- [x] Add or identify the public Forms/Systems coupling-analysis metadata bridge,
  including contribution name/origin and owning-system provenance, before
  implementing expected-block diagnostics.
- [x] Extend the public Forms/Systems metadata collection path, such as
  `FormExprScanner`, `FormStructureAnalyzer`, `FormsInstaller` install metadata,
  or an equivalent public helper, so coupling diagnostics can collect every
  Forms vocabulary terminal required by this plan without private AST traversal.
- [x] Add a bridge-readiness audit that records which current Forms/Systems
  metadata sources are complete, partial, or unavailable for coupling use,
  including source-level field/test dependencies, non-field terminals, temporal
  terminals, geometry terminals, geometry-sensitivity provenance, contribution
  identity, owning-system identity, and installed block/domain evidence.
- [x] Define the public bridge's normalized terminal record shape before coupling
  graph integration: terminal kind, Forms-native provider identity, optional
  `analysis::VariableKey` graph identity, value type, owner scope, integration
  location, and source contribution provenance must be explicit fields rather
  than inferred from scanner container names.
- [x] Treat the metadata bridge as a prerequisite for installed-form dependency
  diagnostics: expected-block, non-field dependency, temporal-terminal,
  geometry-terminal, and geometry-sensitivity validators must either consume the
  bridge or run only against synthetic metadata fixtures until the bridge exists.
- [x] Add feature gates or explicit unsupported diagnostics so partially
  implemented bridge coverage cannot silently mark installed-form validators
  complete.
- [x] Confirm the new setup-time `FE/Coupling::CouplingGraph` is coordinated
  with existing `FE/Analysis/CouplingGraphAnalyzer`,
  `ContributionDescriptor`, and `FormulationRecord` metadata rather than
  duplicating their installed-form analysis role.
- [x] Confirm expert monolithic coupling hooks use approved Systems extension
  points and do not call internal raw kernel registration methods directly.

Unit-test verification checklist:

- [x] Add a build-only unit test or compile target that includes every public
  `FE/Coupling` header independently.
- [x] Add a build-only unit test or compile target that includes every public
  `Physics/Coupling` header independently.
- [x] Add a dependency-boundary test or build assertion that `FE/Coupling`
  compiles without including any `Physics/Coupling` header.
- [x] Add a dependency-boundary test or build assertion that `FE/Coupling`
  uses only public Forms/Systems headers for Forms residual installation and
  analysis.
- [x] Add a dependency-boundary test or build assertion that expert hook test
  fixtures cannot call internal `FESystem::add*Kernel` registration paths.
- [x] Add a bridge-readiness test that fails when installed-form validators are
  enabled without a public bridge or explicit synthetic metadata fixture.
- [x] Add a scanner-normalization test fixture that proves bridge consumers see
  explicit terminal kinds rather than relying on scanner container names.
- [x] Add a diagnostics formatting test that verifies missing context values
  include enough names to identify the failing contract and lookup.

### Phase 1: FE Coupling Vocabulary, Context, And Shared Regions

Implementation checklist:

- [x] Implement `CouplingMode`, `CouplingRequirement`,
  `CouplingDependencyMode`, `CouplingTemporalQuantity`,
  `CouplingRegionKind`, `CouplingInterfaceSide`,
  `CouplingCoordinateConfiguration`, `CouplingValueRank`,
  `CouplingFrameSourceEmbeddingPolicy`,
  `CouplingFrameTargetRestrictionPolicy`, and `CouplingEndpointKind`.
- [x] Implement `CouplingPortId` with stable equality, ordering or hashing as
  needed by graph and plan containers, using the configured contract instance
  namespace plus opaque port name rather than the reusable contract type.
- [x] Implement `CouplingValueDescriptor` as partitioned port/exchange payload
  metadata with component-count validation, rank/component compatibility checks,
  component-layout validation for mixed or pass-through components, and
  `GeneralTensor` extent/packing validation for explicitly driver-owned
  transfers.
- [x] Implement `CouplingParticipantRef` with participant name, stable system
  name, non-owning `FESystem` pointer, and validity checks.
- [x] Implement `CouplingFieldRef` with participant name, owning system metadata,
  field name, `FieldId`, function space pointer, component count,
  `systems::FieldScope`, interface marker provenance for interface fields, and
  validity checks.
- [x] Implement `CouplingRegionRef` with participant name, owning system metadata,
  region name, region kind, marker, typed interface side metadata, coordinate
  configuration using the coupling-neutral
  `CouplingCoordinateConfiguration`, logical interface identity when mesh support
  is enabled, interface revision snapshot, and geometry/topology revision keys
  needed by the mesh/assembly layer.
- [x] Implement explicit conversion and validation from
  `CouplingCoordinateConfiguration` to Mesh/Search `svmp::Configuration` during
  interface-map resolution, preserving both declaration and resolved
  configuration provenance in diagnostics.
- [x] Implement `SharedRegionRef` as an N-participant collection of
  `CouplingRegionRef` records with an optional `required_region_kind` validation
  constraint, no group-level FE marker, and no group-level resolved FE integration
  kind; all resolved markers remain participant-region scoped.
- [x] Implement participant-scoped shared-region lookup helpers so
  `sharedRegion(name, participant)` returns a concrete `CouplingRegionRef` and
  `sharedRegionGroup(name)` returns only the marker-free N-participant group.
- [x] Implement `CouplingTemporalSlot`, `CouplingTemporalSlotDescriptor`, and
  `CouplingEndpointRef` for field, region-data, raw auxiliary-state,
  `AuxiliaryInput`, `AuxiliaryOutput`, parameter, and external-buffer endpoints,
  including temporal slot metadata, `history_index` only for History,
  `stage_index` only for Stage, logical 1-based history indices, unset defaults
  for non-history/non-stage slots, participant scope required for FE-backed and
  parameter endpoints, optional participant scope only for external-buffer
  endpoints, and endpoint-kind-specific temporal validation rules.
- [x] Implement `CouplingResolvedTemporalBackingKind` and
  `ResolvedCouplingTemporalSlot` so executable endpoints preserve both the
  declaration temporal request and the provider-resolved backing, 0-based storage
  index when applicable, explicit predicted backings for system, auxiliary, and
  interface-map storage, provider name, state revision key, and time.
- [x] Implement `CouplingExternalBufferAccess`,
  `CouplingExternalBufferDistribution`, `CouplingExternalBufferLifetime`, and
  `CouplingExternalBufferDescriptor` as the explicit driver-owned external-buffer
  registry schema for scalar type, access, lifetime, distribution, extents,
  strides, packing, supported indexed temporal-slot descriptors, payload shape,
  and revision keys.
- [x] Implement `CouplingDriverOwnedTransferDescriptor` as the explicit
  driver-owned transfer-operator registry schema for supported value ranks,
  general-tensor support, component-layout preservation, supported source/target
  temporal slots, and registry revision keys, with `GeneralTensor` support
  represented only by `supported_ranks` containing
  `CouplingValueRank::GeneralTensor`.
- [x] Implement `ResolvedCouplingEndpoint` so partitioned execution plans carry
  resolved FE or driver identities: owning system name, registry provider,
  resolved endpoint kind, resolved participant/global endpoint scope,
  registry-resolved stable endpoint key, `FieldId`, FE quantity id, parameter
  slot and `params::ValueType`, region-data provider kind/provider key,
  BoundaryReductionService functional identity, auxiliary resolution kind, raw
  auxiliary block index, AuxiliaryInput slot, AuxiliaryOutput stable id/name plus
  optional flattened materialized-output slot, external-buffer descriptor, value
  descriptor, resolved temporal backing, and layout/registry revision keys, with
  `declaration_provenance` retained only for diagnostics and never used as the
  executable endpoint identity.
- [x] Implement `CouplingTransferDeclaration`,
  `CouplingInterfaceTransferDeclaration`, `ResolvedCouplingTransfer`,
  durable `CouplingInterfaceMapProvenance`,
  execution-only `CouplingInterfaceMapRuntimeHandles` including source/target
  systems, `InterfaceSearchRegistry`, optional `SlidingInterfaceMap` state
  wrapper, and the concrete `svmp::search::InterfaceMap`,
  `CouplingExchangeDeclaration`, resolved `CouplingExchange`, `CouplingGroupHint`,
  and `PartitionedCouplingPlan` as physics-agnostic plan structures.
- [x] Make `CouplingTransferKind::Unspecified` the declaration default and reject
  it during exchange resolution; require `Identity` transfers to be explicitly
  requested.
- [x] Map interface transfer declarations to the existing
  `FE/Systems/InterfaceOperators` kinds and resolve them to
  `systems::InterfaceTransferOptions` instead of introducing a parallel FE
  transfer implementation.
- [x] Preserve `source_embedding_policy` and `target_restriction_policy` in
  `ResolvedCouplingTransfer` separately from `systems::InterfaceTransferOptions`
  because Systems options currently have no fields for true 2D vector embedding
  or target-layout restriction.
- [x] Validate that interface transfer declarations include or can
  unambiguously derive `systems::InterfaceTransferOptions` fields: field kind,
  component count, frame transform policy, source-to-target rotation, and
  conservation tolerance, plus source/target systems, source/target interface
  markers, interface-map provenance, `LogicalInterfaceRegionId` records,
  `InterfaceRevisionSnapshot` records, coordinate configurations, search
  revision keys, map revision key, search map state, sliding-map kind, operator
  state, and accepted/trial epoch metadata.
- [x] Validate interface frame transforms against value rank and component
  counts: scalar and mixed-block transfers require `None`, vector transforms
  require vector payloads with at least the transformed component count, rank-2
  transforms require rank-2 tensor payloads with at least the transformed
  component count, extra components require documented component-layout
  pass-through metadata, and 2D vectors require explicit non-`None`
  `CouplingFrameSourceEmbeddingPolicy` source embedding and
  `CouplingFrameTargetRestrictionPolicy` target restriction before they are
  accepted; if no coupling-side pre/post adapter exists
  to apply those 2D policies around Systems transfer execution, reject the true
  2D FE frame transform during resolution rather than producing a plan that
  silently drops the policies.
- [x] Guard direct use of `systems::InterfaceTransferOptions` behind
  `SVMP_FE_WITH_MESH` and reject interface transfer kinds when
  mesh/interface-operator support is not compiled in.
- [x] Implement `CouplingContext` lookup APIs for fields, regions,
  participants, participant-specific shared regions, shared-region groups,
  global and participant-scoped driver-owned external-buffer descriptors,
  driver-owned transfer descriptors, interface search registries, interface-map
  runtime handles, and existence predicates.
- [x] Implement `CouplingContextBuilder` or equivalent setup helper for
  registering participant-to-`FESystem` bindings, fields, regions, shared
  regions, aliases, explicit global and participant-scoped driver-owned
  external-buffer registries, driver-owned transfer-operator registries,
  interface search registries, and sliding-interface-map providers before
  finalizing an immutable context.
- [x] Validate that raw `FieldId`s, region markers, interface markers, and
  interface maps are always interpreted relative to the participant's owning
  `FESystem`.
- [x] Validate that `CouplingEndpointRef::endpoint_name` is physics-opaque but
  still resolves as a concrete registry/provider key for its endpoint kind,
  including external-buffer keys supplied by the scoped driver-owned registry.
- [x] Validate endpoint participant scope rules: FE-backed field, region-data,
  auxiliary, and parameter endpoints require `participant_name`, external buffers
  may omit it only for the context-level global driver-owned registry, and lookup
  diagnostics identify the scope that was searched.
- [x] Validate `ExternalBuffer` endpoint names against the explicit driver-owned
  registry scope and reject descriptors with missing scalar type, unsupported
  indexed temporal-slot descriptor, inconsistent payload shape,
  extents/strides/packing, incompatible access direction, or stale layout/data
  revision keys.
- [x] Validate `DriverOwned` transfer declarations against the explicit
  driver-owned transfer registry and reject unsupported value ranks, unsupported
  source/target temporal-slot descriptors, unsupported general tensors,
  component-layout loss, or stale transfer-registry revision keys.
- [x] Preserve resolved `CouplingDriverOwnedTransferDescriptor` capability and
  revision provenance in `ResolvedCouplingTransfer` so executable plans do not
  rely on declaration-only driver-owned transfer names, and so durable plan
  equality/serialization uses descriptor fields and revision keys rather than
  registry object identity or runtime lookup handles.
- [x] Make duplicate participant/field/region registrations deterministic:
  exact duplicates are either accepted idempotently or rejected consistently,
  and conflicting duplicates are rejected.
- [x] Implement `SharedRegionRegistry` so multiple contracts can refer to one
  shared-region name and receive the same participant-region mapping.
- [x] Validate shared-region name consistency across exchange-level
  `shared_region_name`, producer/consumer `CouplingRegionEndpointDeclaration`
  records; inherit the exchange-level name when an endpoint attachment omits it,
  and reject conflicting endpoint names or shared-interface uses with no
  shared-region scope.
- [x] Validate geometry-terminal shared-region name consistency separately from
  partitioned exchange declarations: reject conflicts between
  `scope.region.shared_region_name` and `scope.location.shared_region_name`,
  inherit the region owner shared-region name into the location when omitted,
  and reject shared-side terminal uses with no shared-region scope.
- [x] Validate that every shared region required by a contract has all required
  participant-region mappings.
- [x] Validate that region kinds are compatible with their intended use:
  domain, boundary, interior face, interface face, and user-defined regions are
  not silently interchanged.
- [x] Require `CouplingRegionKind::UserDefined` regions to resolve through
  `CouplingContext` to a concrete FE integration kind before Forms residual
  lowering, interface transfer resolution, or geometry-terminal validation, or to
  declare an explicit provider-extension evaluation path; reject unresolved
  user-defined regions.
- [x] Map region kinds to the existing Forms measures `.dx()`, `.ds(marker)`,
  `.dS()`, and `.dI(interface_marker)` without introducing a parallel
  coupling-specific measure vocabulary.
- [x] Preserve marker and side/orientation metadata needed by interface-face
  Forms expressions and trace/mortar workflows.
- [x] Validate that Forms-authored monolithic fields belong to one compatible
  `FESystem` before constructing field-bound Forms symbols.
- [x] Validate that interface-face shared regions reference registered
  interface topology before allowing `.dI(interface_marker)` residuals.
- [x] Validate field component counts and function-space availability before
  monolithic or partitioned lowering begins.
- [x] Validate field scope and interface marker provenance against the resolved
  `FieldRecord` before using a field in interface-specific Forms or partitioned
  transfer paths.
- [x] Validate FE-backed partitioned value descriptors against the resolved
  `FieldRecord` and `FunctionSpace` signature, and ensure interface exchange
  value descriptors can map to `InterfaceFieldKind` categories.
- [x] Validate `SymmetricTensor` interface exchanges either expand to explicit
  `Rank2Tensor` transfer payloads with documented packing or fail validation.
- [x] Reject `GeneralTensor` for FE interface transfer kinds and
  require explicit positive extents, component count equal to the extent product,
  packing, `CouplingTransferKind::DriverOwned`, and driver validation for
  driver-owned general tensor transfers.
- [x] Keep all physical field names opaque to the FE layer; the FE context may
  store user-supplied strings but must not interpret them.

Unit-test verification checklist:

- [x] `test_CouplingContext.cpp` covers successful field lookup by participant
  and field name.
- [x] `test_CouplingContext.cpp` covers successful participant lookup with owning
  `FESystem` metadata.
- [x] `test_CouplingContext.cpp` covers successful region lookup by
  participant and region name.
- [x] `test_CouplingContext.cpp` verifies interface-map resolution converts
  `CouplingCoordinateConfiguration` to `svmp::Configuration` explicitly and
  rejects unsupported mappings instead of relying on enum-value equality.
- [x] `test_CouplingContext.cpp` covers missing participant, field, region, and
  shared-region diagnostics.
- [x] `test_CouplingContext.cpp` covers duplicate and conflicting field
  registration behavior.
- [x] `test_CouplingContext.cpp` rejects FE-backed fields or regions registered
  without participant system ownership.
- [x] `test_CouplingContext.cpp` covers component-count validation failures.
- [x] `test_CouplingContext.cpp` verifies `CouplingFieldRef` preserves
  `systems::FieldScope` and interface marker provenance from the registered
  `FieldRecord`.
- [x] `test_CouplingContext.cpp` verifies component-layout metadata is required
  for mixed payloads and for vector/rank-2 pass-through components.
- [x] `test_CouplingContext.cpp` verifies FE-backed partitioned value
  descriptors are checked against registered field space signatures.
- [x] `test_CouplingContext.cpp` verifies `GeneralTensor` requires
  positive extents, a component count matching their product, and nonempty
  driver-defined packing metadata when used with a driver-owned transfer.
- [x] `test_CouplingContext.cpp` verifies external-buffer descriptors validate
  scalar type, access, lifetime, distribution, extents, strides, packing,
  supported indexed temporal-slot descriptors, and layout/data revision keys.
- [x] `test_CouplingContext.cpp` verifies external-buffer lookup supports
  context-level global scope and participant-scoped registries, rejects duplicate
  keys within a scope, allows the same key in different scopes, and reports the
  searched scope in lookup diagnostics.
- [x] `test_CouplingContext.cpp` verifies driver-owned transfer descriptors
  validate supported ranks, general-tensor support, source/target temporal slots,
  component-layout preservation, and registry revision keys.
- [x] `test_CouplingContext.cpp` verifies parameter endpoints resolve the
  existing `params::ValueType` from `ParameterRegistry::Spec`, accept scalar
  Real on the initial built-in transfer path, and reject unsupported non-Real
  value descriptors unless a typed parameter/driver extension explicitly defines
  the mapping.
- [x] `test_CouplingContext.cpp` verifies FE-backed and parameter endpoints reject
  missing `participant_name`, while external-buffer endpoints accept missing
  `participant_name` only for registered global driver-owned buffers.
- [x] `test_CouplingContext.cpp` verifies predicted temporal slots resolve to
  explicit system, auxiliary, interface-map, external-buffer, or provider-defined
  predicted backings and are never silently treated as Current or Accepted.
- [x] `test_CouplingContext.cpp` verifies typed minus/plus interface sides are
  accepted and ambiguous or invalid side mappings are rejected.
- [x] `test_CouplingContext.cpp` verifies interface search registries and
  interface-map runtime handles resolve from durable provenance that includes
  interface-search-registry names, distinct interface entry and interface map
  names, logical region ids, coordinate configurations, revision snapshots,
  search map state, sliding-map kind, and search-registry identity.
- [x] `test_PartitionedCouplingPlan.cpp` verifies `SymmetricTensor` interface
  exchanges require explicit rank-2 packing or fail validation.
- [x] `test_PartitionedCouplingPlan.cpp` rejects `GeneralTensor` for
  FE interface transfer kinds and accepts it only for explicitly driver-owned
  transfers with validated extents and packing metadata.
- [x] `test_SharedRegionRegistry.cpp` covers two-participant shared-region
  registration and lookup.
- [x] `test_SharedRegionRegistry.cpp` covers N-participant shared-region
  registration and lookup.
- [x] `test_SharedRegionRegistry.cpp` rejects missing required participant
  region mappings.
- [x] `test_SharedRegionRegistry.cpp` rejects inconsistent duplicate
  shared-region declarations.
- [x] `test_SharedRegionRegistry.cpp` verifies `required_region_kind` is enforced
  as a compatibility constraint, rejects conflicts with participant-region kinds,
  and is never treated as a group-level marker or resolved FE integration kind.
- [x] `test_SharedRegionRegistry.cpp` verifies participant-scoped
  `sharedRegion(name, participant)` lookup returns the resolved
  `CouplingRegionRef` marker/side metadata while `sharedRegionGroup(name)`
  remains marker-free.
- [x] `test_SharedRegionRegistry.cpp` verifies exchange-level shared-region names
  are inherited by producer/consumer region endpoint declarations, and conflicting
  endpoint shared-region names are rejected.
- [x] `test_CouplingGeometryRequirements.cpp` verifies geometry-terminal
  shared-region names reject conflicts between region owner and location metadata,
  inherit region owner shared-region scope when the location omits it, and reject
  shared-side terminals with no shared-region scope.
- [x] `test_CouplingContext.cpp` verifies `UserDefined` regions fail Forms,
  interface-transfer, and geometry-terminal lowering unless they resolve to a
  concrete FE integration kind or explicit provider-extension path.
- [x] `test_SharedRegionRegistry.cpp` verifies interface shared regions expose
  per-participant `.dI(marker)` marker and side/orientation metadata.
- [x] `test_SharedRegionRegistry.cpp` verifies minus/plus side ownership is
  recorded per participant and rejected when ambiguous.
- [x] `test_SharedRegionRegistry.cpp` rejects interface shared regions whose
  marker has no registered interface topology for monolithic Forms use.
- [x] `test_CouplingContext.cpp` verifies boundary, interior-face, and
  interface-face regions map to the expected Forms measure categories.
- [x] `test_PartitionedCouplingPlan.cpp` rejects `Unspecified` transfer
  declarations and verifies `Identity` is accepted only when endpoint ownership,
  layout, value shape, and temporal slots are compatible.
- [x] `test_PartitionedCouplingPlan.cpp` validates endpoint temporal slots for
  current, accepted, predicted, history, stage, and external endpoints, including
  missing, present-on-wrong-slot, invalid history/stage index combinations,
  logical 1-based history mapping to 0-based FE/Auxiliary storage, no separate
  endpoint slots for `u_prev`/`u_prev2`, and endpoint-kind rules for Field,
  RegionData, raw AuxiliaryState, AuxiliaryInput, AuxiliaryOutput,
  ExternalBuffer, and Parameter endpoints, plus interface-map transfer-state
  rules and driver-owned GeneralTensor payload descriptors.
- [x] `test_PartitionedCouplingPlan.cpp` verifies resolved exchanges contain
  `ResolvedCouplingEndpoint` producer/consumer records rather than executable
  declaration endpoint refs, and that each endpoint stores the expected
  kind-specific resolved id or external-buffer descriptor, resolved
  participant/global scope, resolved temporal backing, and diagnostic-only
  declaration provenance.
- [x] `test_PartitionedCouplingPlan.cpp` verifies resolved interface transfers
  include `systems::InterfaceTransferOptions` plus durable
  `CouplingInterfaceMapProvenance`, preserve `source_embedding_policy` and
  `target_restriction_policy` outside Systems options, and resolve non-owning
  runtime system, search-registry, and interface-map handles only from the
  current context.
- [x] `test_PartitionedCouplingPlan.cpp` verifies interface frame transforms are
  rejected for scalar and mixed-block payloads, vector transforms require
  at least the transformed vector component count, rank-2 transforms require at
  least the transformed rank-2 tensor component count, and extra components are
  preserved only when their component layout is documented as pass-through
  metadata; true 2D vector transforms require explicit source embedding and
  target restriction policies and either have a coupling-side execution adapter
  available or fail validation before any `systems::InterfaceTransferOptions`
  object can silently drop those policies.
- [x] `test_PartitionedCouplingPlan.cpp` covers endpoint, port, value
  descriptor, transfer declaration, resolved transfer options, exchange
  declaration, resolved exchange, and group-hint value semantics.
- [x] `test_PartitionedCouplingPlan.cpp` verifies resolved `DriverOwned`
  transfers preserve the driver-owned transfer descriptor's supported ranks,
  temporal-slot support, component-layout policy, general-tensor support, and
  registry revision key, and that durable equality ignores registry object
  identity and runtime lookup handles.

### Phase 2: Coupling Contract Declaration And Registry

Implementation checklist:

- [x] Implement `CouplingParticipantUse`, `CouplingFieldUse`,
  `CouplingVariableKind`, `CouplingVariableUse`, `CouplingRegionUse`,
  `CouplingSharedRegionUse`, `CouplingResidualDependency`,
  `CouplingTemporalRequirement`, `CouplingGeometryTerminalQuantity`,
  `CouplingGeometryTerminalLocationDeclaration`,
  `CouplingGeometryTerminalLocationProvenance`,
  `CouplingGeometryTerminalScope`, `CouplingGeometryTerminalRequirement`,
  `CouplingBlockExpectation`, `CouplingAdditionalFieldDeclaration`,
  `ResolvedCouplingAdditionalFieldDeclaration`, `CouplingInstalledDependency`,
  `CouplingInstalledBlockProvenance`, `CouplingInstallMetadata`,
  `CouplingRegionEndpointDeclaration`,
  resolved driver-owned transfer descriptor provenance,
  `CouplingFormTerminalProvenanceDeclaration`,
  `CouplingGeometrySensitivityDeclaration`,
  `CouplingSymbolicOptionsDeclaration`, `CouplingFormContribution`, and
  `ResolvedCouplingFormContribution`, including declaration-time
  `CouplingFormInstallOptionsDeclaration` on form contribution records and
  resolved `systems::FormInstallOptions`, geometry tangent policy, and extra
  trial field metadata on resolved form contribution records.
- [x] Implement `CouplingFormFieldProvenance`,
  `CouplingGeometrySensitivityProvenance`, `CouplingFormTemporalProvenance`,
  `CouplingGeometryTerminalOwnerProvenance`,
  `CouplingFormGeometryTerminalProvenance`,
  `CouplingFormNonFieldDependencyProvenance`,
  `CouplingFormVariableDependencyProvenance`, and
  `CouplingFormAnalysisMetadata` as the FE/Coupling adapter shape consumed by
  coupling diagnostics.
- [x] Ensure declaration records use `CouplingVariableUse` while installed
  Forms/expert metadata uses resolved `analysis::VariableKey` rows and
  dependencies with `analysis::DomainKind`, matrix/vector, and provider
  provenance.
- [x] Add stable `contribution_name` and `origin` metadata to form contribution,
  resolved form contribution, `CouplingInstallMetadata`, and analysis metadata
  records, and add owning `FESystem` name provenance wherever raw `FieldId`s are
  reported, so diagnostics can distinguish multiple contracts installed into the
  same operator tag without treating `FieldId`s as globally unique.
- [x] Implement `CouplingContractDeclaration` with contract type/registry key
  and unique configured contract instance name,
  participant requirements, field requirements, participant-local region
  requirements, shared-region requirements, additional-field declarations,
  non-field data/provenance requirements, generic monolithic variable
  dependencies, temporal requirements, geometry-terminal requirements, expected
  field or non-field dependency expectations, partitioned exchange declarations,
  and group hints needed for graph construction.
- [x] Treat partitioned exchange declarations from `CouplingContractDeclaration`
  and `buildPartitionedExchangeDeclarations()` as declaration metadata only;
  all executable endpoint, transfer, and interface-map resolution belongs to
  `FE/Coupling` plan generation.
- [x] Make additional-field declarations carry enough declaration-time data to
  lower to `systems::FieldSpec` or `addInterfaceField()`: participant or
  contract-owned namespace, namespace name, registration target participant or
  shared-monolithic-`FESystem` policy, function space, component count, field
  scope, and participant-region or shared-region attachment for interface
  fields. Raw interface markers appear only after resolution in
  `ResolvedCouplingAdditionalFieldDeclaration::field_spec.interface_marker`.
- [x] Validate additional-field attachment combinations: `VolumeCell` rejects
  `region_name` and `shared_region_name`, `InterfaceFace` requires exactly one
  of `region_name` or `shared_region_name`, and both interface resolution paths
  must produce an `InterfaceFace` `CouplingRegionRef` and final
  `systems::FieldSpec`.
- [x] Treat `components == 0` in additional-field declarations as "infer from
  `space->value_dimension()`"; reject negative component counts before calling
  Systems registration APIs, and reject any positive component count that does
  not exactly match the function-space value dimension.
- [x] Implement the abstract `CouplingContract` interface with `name()`,
  `declare()`, `validate()`, `buildMonolithicForms()`,
  `installMonolithicTerms()`, and
  `buildPartitionedExchangeDeclarations()`.
- [x] Implement `MonolithicCouplingInstallContext` as the only expert
  monolithic install surface exposed to coupling contracts.
- [x] Make `installMonolithicTerms()` return a
  `std::vector<CouplingInstallMetadata>` with explicit dependency/block metadata
  for every expert custom contribution, one metadata record per contribution.
- [x] Keep `buildMonolithicForms()`, `installMonolithicTerms()`, and
  `buildPartitionedExchangeDeclarations()` optional with empty defaults so
  contracts can support one or both lowering modes incrementally.
- [x] Implement declaration validation that rejects empty contract instance
  names, missing or inconsistent contract types, duplicate contract instance names,
  duplicate participant uses, duplicate field uses, duplicate participant-local
  region uses, duplicate shared-region uses, duplicate additional-field
  declarations, invalid derivative orders, inconsistent non-field dependency
  requirements, inconsistent expected-block records, and inconsistent
  partitioned exchange declarations.
- [x] Validate `CouplingNonFieldDependencyRequirement` records as declaration
  metadata: participant/system-scoped names, expected value/type constraints,
  optional declaration-side region scope, optional required region kind, and
  material-state byte offsets are allowed, while resolved markers, logical
  regions, slots, output ids, provider indices, and provider-local handles are
  forbidden until installed provenance is available.
- [x] Treat `FieldDerivative` temporal requirements as positive-order
  derivative requirements, `FieldHistoryValue` as positive logical history-depth
  requirements, mesh temporal quantities as mesh-motion role/provider
  requirements with explicit `mesh_motion_scope` and
  `systems::MeshMotionFieldRole` resolution in N-participant contracts, and
  validate `EffectiveTimeStep` separately from ordinary `TimeStep`.
- [x] Treat mesh displacement, coordinate, Jacobian, Jacobian-inverse, normal,
  measure, surface-Jacobian, cell-metric, and cell-domain-id requirements as
  geometry-terminal requirements, not as time-integration requirements.
- [x] Support required and optional participant, field, participant-local region,
  and shared-region declarations consistently across validation, diagnostics,
  and graph construction.
- [x] Define optional additional-field semantics separately from optional
  participant/field/region declarations: disabled fields are omitted, optional
  present declarations may be skipped only when no dependency or expected block
  relies on them, and selected optional fields use the same compatibility checks
  as required fields.
- [x] Implement additional coupling-field declaration flow so contracts declare
  fields such as multipliers before `FESystem::setup()`, while the monolithic
  coupling builder owns the actual `FESystem` registration.
- [x] Lower volume coupling fields centrally through
  `FESystem::addField(systems::FieldSpec{...})` and interface coupling fields
  through `FESystem::addInterfaceField(...)` or an equivalent
  `systems::FieldSpec` with `FieldScope::InterfaceFace`.
- [x] Ensure additional field declarations are namespaced by participant or
  contract instance as needed to avoid accidental collisions, and validate that
  contract-owned fields are not treated as participant requirements.
- [x] Implement `CouplingRegistry` for registering, listing, looking up, and
  instantiating coupling contracts.
- [x] Make contract-type registration deterministic when duplicate type names are
  attempted: reject duplicates or require explicit replacement.
- [x] Add mode-specific dispatch helpers so setup code can ask whether a
  contract supports monolithic lowering, partitioned lowering, or both.
- [x] Ensure a two-participant contract uses the same API as an N-participant
  contract rather than a special pairwise-only path.

Unit-test verification checklist:

- [x] `test_CouplingContractValidation.cpp` accepts a minimal valid
  two-participant declaration.
- [x] `test_CouplingContractValidation.cpp` accepts a valid N-participant
  declaration.
- [x] `test_CouplingContractValidation.cpp` rejects empty names and duplicate
  contract instance names, participant, field, participant-local region,
  shared-region, dependency, and expected-block records.
- [x] `test_CouplingContractValidation.cpp` rejects declarations whose
  `contract_type` disagrees with `CouplingContract::name()` or whose
  contract-owned ports/additional fields use a different instance namespace.
- [x] `test_CouplingContractValidation.cpp` rejects duplicate additional-field
  declarations and inconsistent partitioned exchange declarations.
- [x] `test_CouplingContractValidation.cpp` rejects additional-field
  declarations that lack namespace, registration-target policy, function space,
  scope, or participant-region / shared-region attachment data needed for FE
  field registration; `components == 0` remains valid and means infer from the
  function space.
- [x] `test_CouplingContractValidation.cpp` verifies `VolumeCell` additional
  fields reject participant-region and shared-region attachments, while
  `InterfaceFace` additional fields require exactly one participant-region or
  shared-region attachment.
- [x] `test_CouplingContractValidation.cpp` verifies interface additional-field
  declarations do not carry raw markers, resolve markers only through
  `CouplingContext`, preserve declaration versus resolved-field metadata, and
  store the resolved marker only in
  `systems::FieldSpec::interface_marker`.
- [x] `test_CouplingContractValidation.cpp` verifies `components == 0` infers
  the function-space value dimension, negative component counts are rejected,
  and positive mismatches are rejected.
- [x] `test_CouplingContractValidation.cpp` verifies optional declarations do
  not fail validation when absent and required declarations do fail.
- [x] `test_CouplingContractValidation.cpp` verifies optional additional-field
  declarations are skipped with diagnostics when prerequisites are absent, but
  fail if a dependency or expected block still references the skipped field.
- [x] `test_CouplingContractValidation.cpp` verifies participant-scoped and
  contract-owned additional fields have distinct namespaces and resolve to an
  explicit target `FESystem` before registration.
- [x] `test_CouplingContractValidation.cpp` verifies implicit-monolithic and
  external/lagged dependency declarations are preserved distinctly, and that
  implicit dependencies can later resolve through `StateField` use, structured
  Systems geometry-sensitivity provenance, or supported non-field
  `analysis::VariableKey` dependencies.
- [x] `test_CouplingContractValidation.cpp` verifies
  `CouplingVariableUse` preserves field, auxiliary-state, AuxiliaryInput,
  AuxiliaryOutput, boundary-functional, and global-scalar variable identities
  without treating all dependency declarations as field-to-field blocks.
- [x] `test_CouplingContractValidation.cpp` verifies non-field
  `CouplingVariableUse` records adapt to `analysis::VariableKey` using
  canonical owning-system or participant-qualified names, while slots and output
  ids appear only in resolved lookup/provenance metadata.
- [x] `test_CouplingContractValidation.cpp` verifies
  `CouplingNonFieldDependencyRequirement` supports parameter, coefficient,
  material-state, boundary-functional, boundary-integral, auxiliary-state,
  `AuxiliaryInput`, and `AuxiliaryOutput` requirements without forcing
  provenance-only dependencies into `CouplingResidualDependency`.
- [x] `test_CouplingContractValidation.cpp` verifies non-field requirements use
  declaration-side names, optional region scope, optional region-kind
  constraints, material-state byte offsets, and expected value/type constraints,
  while resolved parameter slots, auxiliary ids, provider-local indices, markers,
  logical regions, and system provenance appear only in
  `CouplingFormNonFieldDependencyProvenance`.
- [x] `test_CouplingContractValidation.cpp` verifies BoundaryIntegral
  declaration/provenance maps to BoundaryFunctional graph identity until
  Analysis exposes a distinct boundary-integral variable kind.
- [x] `test_CouplingContractValidation.cpp` verifies temporal requirement
  derivative-order, explicit history-depth, mesh temporal, and
  effective-time-step declarations.
- [x] `test_CouplingContractValidation.cpp` verifies geometry-terminal
  requirement declarations for mesh displacement, coordinates, Jacobians,
  normals, measures, and surface Jacobians.
- [x] `test_CouplingRegistry.cpp` covers registration, lookup, duplicate
  handling, and deterministic iteration order.
- [x] `test_CouplingRegistry.cpp` covers mode-specific contract dispatch
  without invoking unsupported lowering hooks.
- [x] `test_CouplingContractValidation.cpp` verifies additional-field
  declarations reject collisions and accept unique coupling fields.
- [x] `test_CouplingContractValidation.cpp` verifies expert install hooks are
  invoked only through `MonolithicCouplingInstallContext`.
- [x] `test_CouplingContractValidation.cpp` verifies Forms contributions reject
  duplicate or empty contribution names within one contract and preserve
  contribution names distinctly across contracts sharing an operator tag.

### Phase 3: Forms Authoring And Temporal Requirement Support

Implementation checklist:

- [x] Implement `CouplingFormBuilder` so contracts can resolve state, test, and
  time-derivative symbols through `CouplingContext`.
- [x] Make `CouplingFormBuilder::state()` return field-bound `StateField`
  expressions for the requested participant and field.
- [x] Make `CouplingFormBuilder::test()` return field-bound `TestField`
  expressions for the requested participant and field.
- [x] Make `CouplingFormBuilder::timeDerivative()` return symbolic
  derivative expressions such as `dt(field,k)` without choosing a concrete
  time integrator.
- [x] Make `CouplingFormBuilder::previousSolution()` record participant/field
  provenance, expose no symbolic-name parameter, and lower through the public
  `FormExpr::previousSolution(k)` path to the existing field/trial-scoped
  `PreviousSolutionRef(k)` Forms terminal.
- [x] Validate `CouplingFormBuilder::previousSolution()` use in mixed forms by
  requiring the named participant/field to match the active trial field for the
  installed contribution, or by recording explicit owning-trial-block provenance
  from the metadata bridge.
- [x] Make `CouplingFormBuilder::time()` and `timeStep()` return the same
  symbolic time and time-step terms used by the existing FE Forms pipeline.
- [x] Make `CouplingFormBuilder::effectiveTimeStep()` return the existing
  Forms `deltat_eff()` terminal.
- [x] Make `CouplingFormBuilder` expose mesh displacement, mesh temporal, and
  geometry-terminal helpers that delegate to the existing Forms vocabulary
  rather than introducing new FE/Coupling terminals.
- [x] Make `CouplingFormBuilder` record a coupling-owned terminal-provenance
  side table for `previousSolution(...)`, mesh temporal, and geometry-terminal
  helper calls, and attach the matching provenance records to each
  `CouplingFormContribution` before resolution.
- [x] Key terminal-provenance matching by public Forms identity, either the
  node ownership identity or raw node address obtained from
  `FormExpr::nodeShared()` for the terminal expression, or an explicit public
  Forms metadata hook, assign `terminal_sequence` after matching in deterministic
  encounter order, and reject identity-destroying transformed expressions unless
  a public provenance tag is copied before the transform.
- [x] Make geometry-terminal helper and metadata capture preserve optional
  `CouplingGeometryTerminalLocationDeclaration` data while treating the Forms
  measure and side restrictions as the authoritative integration context, and
  record `CouplingGeometryTerminalLocationProvenance`, the resolved
  `analysis::DomainKind`, and owner provenance used by installed provenance,
  including marker/logical-region, revision, value configuration, and optional
  pullback/pushforward from/to configurations.
- [x] Make `CouplingFormBuilder` region and shared-region helpers delegate to
  `CouplingContext` and preserve diagnostics for failed lookups.
- [x] Resolve `CouplingFormInstallOptionsDeclaration` into
  `ResolvedCouplingFormContribution::install_options` so the monolithic builder
  can forward concrete `systems::FormInstallOptions` to `installFormulation()`.
- [x] Make `CouplingSymbolicOptionsDeclaration` mirror only non-geometry
  `forms::SymbolicOptions` fields, excluding both
  `forms::SymbolicOptions::ad_mode`, `use_symbolic_tangent`,
  `geometry_tangent_path`, and geometry sensitivity, with AD mode resolved into
  top-level `systems::FormInstallOptions::ad_mode` and geometry sensitivity plus
  geometry tangent policy resolved into
  `systems::FormInstallOptions::compiler_options`.
- [x] Reject declaration-time form install options that attempt to smuggle raw
  AD-mode overrides, `use_symbolic_tangent`, `geometry_tangent_path`,
  geometry-sensitivity settings, or `FieldId`s through any raw
  `forms::SymbolicOptions` path; mesh-motion sensitivity and geometry tangent
  policy must be declared through coupling-level install options and resolved by
  the builder.
- [x] Keep coupling residuals explicit in Forms terms such as `.ds(marker)`,
  `.dS()`, `.dI(marker)`, side restrictions, normals, jumps, averages, traces,
  and penalties rather than hiding them behind broad coupling-specific weak-form
  helpers.
- [x] Consume the public Forms/Systems coupling-analysis metadata bridge to
  compare referenced state fields, non-field Forms dependencies against
  `CouplingNonFieldDependencyRequirement` records,
  `analysis::VariableKey` coupling-variable dependencies,
  geometry-sensitivity dependencies, symbolic temporal quantities, and
  geometry-terminal quantities against contract declarations.
- [x] Implement the public Forms metadata scanner/bridge coverage needed for
  coupling diagnostics, including parameter refs/symbols, coefficients,
  boundary functionals, boundary integrals, raw auxiliary state,
  `AuxiliaryInput`, `AuxiliaryOutput`, material-state old/work refs,
  geometry terminals, temporal terminals, boundary-integral versus
  boundary-functional provenance, and location-sensitive domain/region evidence,
  using only public Forms/Systems APIs.
- [x] Normalize current scanner/analyzer output before coupling consumption so a
  terminal's semantic kind is explicit even when an existing scanner field name
  is historical, overloaded, or organized for a different consumer.
- [x] Implement the exact Forms-terminal mapping for non-field dependencies:
  `BoundaryFunctionalSymbol` reports BoundaryFunctional graph identity,
  `BoundaryIntegralSymbol` and `BoundaryIntegralRef` report BoundaryIntegral
  provenance and map to BoundaryFunctional graph identity only as a fallback,
  auxiliary refs/symbols map to their corresponding Analysis variable kinds when
  graph identity is requested, and parameter/coefficient/material-state
  terminals remain provenance-only unless Analysis adds variable kinds for them.
- [x] Require that bridge to return native metadata that can be adapted to
  installed field order, contribution name/origin/owning-system provenance,
  field-use provenance including geometry-sensitivity field-use summaries,
  installed geometry-sensitivity options including the mesh-motion field,
  structured `CouplingGeometrySensitivityProvenance` records for mesh-motion and
  cut/embedded geometry sensitivity, parameter, coefficient,
  boundary-functional, boundary-integral, auxiliary-state, `AuxiliaryInput`,
  `AuxiliaryOutput`, and material-state dependency provenance,
  variable-dependency provenance using scoped `analysis::VariableKey` names,
  including Analysis/expert global-scalar variables,
  declaration terminal-provenance records captured from `CouplingFormBuilder`,
  temporal-symbol provenance including field/trial-scoped
  `PreviousSolutionRef(k)` and mesh temporal owner-scope plus
  `systems::MeshMotionFieldRole` provenance, geometry-terminal location,
  resolved `analysis::DomainKind`, owner-scope, frame-transform configuration,
  and revision provenance, and installed dependency/block/domain provenance matching
  `CouplingFormAnalysisMetadata`.
- [x] Add an explicit install metadata path, install record, or stable side-table
  handle so contribution name, origin, and owning system survive the current
  `installFormulation()` call shape instead of being inferred from `OperatorTag`.
- [x] Avoid direct traversal of private Forms AST internals from `FE/Coupling`.
- [x] Define a temporal-policy validation API that uses the existing
  Systems-owned `TimeIntegrator` responsibilities for derivative-order support
  and `SystemState` availability for time, time-step, effective-time-step, and
  history data.
- [x] Validate declared temporal requirements against the selected
  problem-level temporal policy before assembly.
- [x] Validate that symbolic temporal terms used by Forms residuals are
  declared by the contract.
- [x] Validate that geometry terminals used by Forms residuals are declared by
  the contract and supported by the selected geometry transaction,
  mesh-motion binding, or Assembly metadata provider.
- [x] Validate geometry-terminal location/provenance for boundary and interface
  terminals, including declaration-side region kind, resolved
  `analysis::DomainKind`, owner-scope, marker or shared-region name, minus/plus
  side, typed logical interface region, typed `forms::GeometryConfiguration`
  value configuration, optional frame-transform from/to configurations,
  geometry revision, and quadrature-policy key.
- [x] Validate geometry-terminal requirement owner scope by requiring either an
  owning participant or a declaration-side participant/region attachment, so
  normals, measures, and interface-side geometry terminals are not ambiguous in
  N-participant contracts; if both `scope.participant_name` and `scope.region`
  are present, reject conflicting participant names.
- [x] Validate geometry-terminal shared-region scope by rejecting conflicts between
  `scope.region.shared_region_name` and `scope.location.shared_region_name`,
  inheriting the region owner shared-region name into the location when omitted,
  and rejecting shared-side terminals with no shared-region scope.
- [x] Validate that declared implicit dependencies appear as actual
  `StateField` or Systems-recognized geometry-sensitivity dependencies in
  Forms-authored residuals, or as supported non-field `analysis::VariableKey`
  dependencies, unless the contract intentionally uses the expert install hook.
- [x] Validate mesh-motion geometry-sensitivity requests by resolving
  `install_options.geometry_sensitivity.mesh_motion_field` from `field_uses` or
  `extra_trial_field_uses`, writing the resulting `FieldId` into resolved
  `systems::FormInstallOptions`, requiring an existing
  `FESystem::bindMeshMotionField()` binding, and verifying geometry-sensitivity
  provenance is reported as structured metadata by the public Forms/Systems
  metadata bridge.
- [x] Validate geometry tangent policy combinations before building
  `systems::FormInstallOptions`: `GeometryConstant` rejects
  `mesh_motion_field`, `SymbolicRequired`, and `SymbolicWithADCheck`;
  `use_symbolic_tangent` remains valid in `GeometryConstant` only for ordinary
  nonlinear symbolic tangent generation; `MeshMotionUnknowns` requires a
  resolved mesh-motion field and forwards all `GeometryTangentPath` values,
  forcing the symbolic geometry tangent path for `SymbolicRequired` and
  `SymbolicWithADCheck`.
- [x] Adapt cut/embedded geometry sensitivity metadata, when present, into
  `CouplingGeometrySensitivityProvenance` with provenance id, construction
  policy, target kind, source stable id, parent entity, parent geometry DOFs,
  cut topology revision, quadrature policy key, visible assembly paths, available
  sensitivity channels, AD compatibility, sample count, and FE geometry fields
  when applicable.
- [x] Ensure changing the concrete time integrator requires no changes to
  coupling contract code when the declared derivative orders remain supported.

Unit-test verification checklist:

- [x] `test_CouplingFormBuilder.cpp` verifies state and test symbols bind to
  the expected participant field.
- [x] `test_CouplingFormBuilder.cpp` verifies failed state, test, and region
  lookup diagnostics include contract, participant, field, and region names.
- [x] `test_CouplingFormBuilder.cpp` verifies symbolic first- and
  second-derivative construction.
- [x] `test_CouplingFormBuilder.cpp` verifies `previousSolution(k)` lowers
  through `FormExpr::previousSolution(k)` to the existing
  `PreviousSolutionRef(k)` terminal while preserving participant/field
  provenance for diagnostics.
- [x] `test_CouplingFormBuilder.cpp` verifies `previousSolution(k)` does not accept
  or record a source-symbol name because Forms history terminals are trial scoped.
- [x] `test_CouplingFormBuilder.cpp` verifies `previousSolution(k)` is rejected
  or reported as ambiguous when the named field is not the active trial field in
  a mixed contribution and no owning-trial-block provenance is available.
- [x] `test_CouplingFormBuilder.cpp` verifies `time()` and `timeStep()` terms
  lower through the existing Forms temporal path.
- [x] `test_CouplingFormBuilder.cpp` verifies `effectiveTimeStep()` lowers
  through the existing Forms temporal path.
- [x] `test_CouplingFormBuilder.cpp` verifies mesh displacement, mesh temporal,
  coordinate, Jacobian, Jacobian-inverse, normal, measure, surface-Jacobian,
  cell-metric, and cell-domain-id helpers lower through the existing Forms
  vocabulary, and that mesh temporal helper metadata records owner-scope plus
  `systems::MeshMotionFieldRole` provenance.
- [x] `test_CouplingFormBuilder.cpp` verifies builder terminal provenance is
  attached to each `CouplingFormContribution`, survives resolution, and is used
  instead of private Forms AST traversal to recover coupling owner scope.
- [x] `test_CouplingFormBuilder.cpp` verifies terminal-provenance matching uses
  public node ownership identity or raw node address from
  `FormExpr::nodeShared()`, or an explicit public Forms metadata hook, assigns
  deterministic `terminal_sequence` values after matching, and rejects or
  preserves provenance across transformed expressions according to whether a
  public provenance tag was copied before the transform.
- [x] `test_CouplingFormBuilder.cpp` verifies geometry-terminal metadata records
  declaration-side location separately from resolved provenance, including
  resolved `analysis::DomainKind`, owner-scope, marker/shared-region, side,
  typed `forms::GeometryConfiguration`, optional frame-transform from/to
  configurations, typed logical interface region, geometry-revision, and
  quadrature-policy provenance without changing the Forms measure semantics.
- [ ] `test_CouplingFormBuilder.cpp` verifies geometry-terminal declarations for
  boundary/interface terminals include participant or participant-region owner
  scope, reject ambiguous shared-region-only requirements, and reject conflicting
  participant names when both `scope.participant_name` and `scope.region` are
  present.
- [ ] `test_CouplingFormBuilder.cpp` verifies an interface residual is authored
  explicitly with `.dI(marker)` and side restrictions.
- [x] `test_CouplingFormBuilder.cpp` verifies dependency metadata consumption
  for one residual row and multiple state dependencies using the public
  Forms/Systems coupling-analysis metadata bridge.
- [ ] `test_CouplingFormBuilder.cpp` verifies the public scanner/bridge reports
  Forms parameter refs/symbols, coefficients, boundary functionals, boundary
  integrals, raw auxiliary state, `AuxiliaryInput`, `AuxiliaryOutput`,
  material-state old/work refs, geometry terminals, and temporal terminals using
  public Forms/Systems APIs, including distinct boundary-integral versus
  boundary-functional provenance.
- [x] `test_CouplingFormBuilder.cpp` verifies scanner-normalization adapts any
  historical or overloaded scanner container into explicit terminal-kind records
  before coupling graph validation consumes it.
- [ ] `test_CouplingFormBuilder.cpp` verifies declaration-time form install
  options are resolved into `systems::FormInstallOptions`, including
  name-to-`FieldId` geometry-sensitivity resolution and geometry tangent path
  forwarding.
- [ ] `test_CouplingFormBuilder.cpp` rejects declaration-time install options
  that bypass `CouplingSymbolicOptionsDeclaration` and set raw
  `forms::SymbolicOptions::ad_mode` or
  `forms::SymbolicOptions::geometry_tangent_path`,
  `forms::SymbolicOptions::use_symbolic_tangent`, or
  `forms::SymbolicOptions::geometry_sensitivity` instead of the coupling-level
  AD-mode, geometry-tangent, and geometry-sensitivity declarations.
- [ ] `test_CouplingFormBuilder.cpp` verifies `compiler_options.ad_mode` remains
  default or diagnostic-only consistent with top-level
  `systems::FormInstallOptions::ad_mode` and cannot override
  `CouplingFormInstallOptionsDeclaration::ad_mode`.
- [ ] `test_CouplingFormBuilder.cpp` verifies the metadata bridge reports test
  fields, state-field dependencies, geometry-sensitivity dependencies,
  installed geometry-sensitivity options, discrete-field uses, parameter,
  coefficient, boundary-functional, boundary-integral, auxiliary-state,
  `AuxiliaryInput`, `AuxiliaryOutput`, and material-state dependencies,
  participant/system scope and resolved slot/output-id/byte-offset/value-type
  provenance for non-field dependencies,
  variable-dependency provenance including Analysis/expert global-scalar
  variables, declaration terminal-provenance records, temporal symbols including
  field/trial-scoped previous-solution history and mesh temporal owner-scope plus
  `systems::MeshMotionFieldRole` provenance, geometry terminals with location,
  resolved `analysis::DomainKind`, owner-scope, frame-transform configuration,
  and revision provenance, owning-system names, structured geometry-sensitivity
  provenance, and installed block/domain provenance without private AST
  traversal.
- [x] `test_CouplingFormBuilder.cpp` verifies boundary-integral provenance is
  reported distinctly from boundary-functional provenance while graph-variable
  adaptation maps it to BoundaryFunctional identity until Analysis provides a
  dedicated kind.
- [x] `test_CouplingFormBuilder.cpp` verifies the non-field terminal mapping
  table exactly: `BoundaryFunctionalSymbol` is graph-variable BoundaryFunctional
  identity, `BoundaryIntegralSymbol` and `BoundaryIntegralRef` are distinct
  BoundaryIntegral provenance with BoundaryFunctional fallback identity, and
  parameter/coefficient/material-state terminals do not become graph variables
  unless explicitly promoted by Analysis metadata.
- [ ] `test_CouplingFormBuilder.cpp` verifies mesh-motion geometry-sensitivity
  install options require a declared and bound mesh-motion field and produce
  structured geometry-sensitivity provenance.
- [ ] `test_CouplingFormBuilder.cpp` verifies geometry tangent policy
  validation rejects `SymbolicRequired` and `SymbolicWithADCheck` in
  `GeometryConstant` mode, accepts `use_symbolic_tangent` in `GeometryConstant`
  as an ordinary symbolic-tangent request, requires `mesh_motion_field` for
  `MeshMotionUnknowns`, and forwards/forces the resolved geometry tangent path
  correctly for mesh-motion geometry sensitivity.
- [ ] `test_CouplingFormBuilder.cpp` verifies cut/embedded geometry sensitivity
  metadata is adapted into `CouplingGeometrySensitivityProvenance` without
  collapsing it to a boolean field-use flag, including target kind, parent
  entity, parent geometry DOFs, visible assembly paths, revision keys,
  sensitivity-channel booleans, AD compatibility, and sample count.
- [x] `test_CouplingFormBuilder.cpp` verifies contribution name, origin, and
  owning-system provenance survive resolution, installation metadata capture, and
  metadata adaptation.
- [ ] `test_CouplingFormBuilder.cpp` verifies undeclared Forms dependencies
  are diagnosed.
- [x] `test_CouplingTemporalRequirements.cpp` verifies temporal validation
  accepts supported derivative orders and rejects unsupported orders.
- [x] `test_CouplingTemporalRequirements.cpp` verifies time and time-step
  requirements are validated independently from field derivative requirements.
- [x] `test_CouplingTemporalRequirements.cpp` verifies effective-time-step
  requirements are validated independently from ordinary time-step requirements.
- [x] `test_CouplingTemporalRequirements.cpp` verifies
  `PreviousSolutionRef(k)`/`FieldHistoryValue` requirements validate available
  `SystemStateView::u_history` depth and reject missing or unsupported history
  indices.
- [ ] `test_CouplingTemporalRequirements.cpp` verifies mesh velocity, mesh
  acceleration, previous mesh velocity, and predicted mesh velocity requirements
  validate owner-scoped `systems::MeshMotionFieldRole` bindings or provider
  metadata, and reject ambiguous shared-region-only mesh temporal requirements in
  N-participant contracts.
- [x] `test_CouplingGeometryRequirements.cpp` verifies geometry-terminal
  requirements for mesh displacement, coordinates, Jacobians, Jacobian inverses,
  normals, measures, surface Jacobians, cell metrics, and cell-domain ids validate
  against geometry transaction and Assembly metadata support separately from
  temporal policy checks.
- [x] `test_CouplingGeometryRequirements.cpp` verifies boundary/interface
  geometry-terminal requirements require location, resolved
  `analysis::DomainKind`, and owner-scope provenance, and reject terminals whose
  marker, side, Forms geometry configuration, logical interface region, or
  geometry revision does not match the installed Forms integration context.
- [x] `test_CouplingGeometryRequirements.cpp` rejects N-participant
  geometry-terminal requirements whose participant or participant-region owner
  scope is missing or inconsistent with the shared-region side metadata.
- [x] `test_CouplingTemporalRequirements.cpp` verifies the same contract
  declaration validates against two different compatible temporal policies.

### Phase 4: Coupling Graph, Validation, And Diagnostics

Implementation checklist:

- [ ] Implement `CouplingGraph` nodes for participants, fields,
  participant-local regions, non-field variables adapted to
  `analysis::VariableKey`, non-field data/provenance requirements that remain
  provider metadata, shared regions, contract types, contract instances,
  temporal requirements, geometry-terminal requirements, partitioned exchange
  declarations, resolved partitioned exchanges, and expected blocks or
  dependency expectations.
- [x] Implement declaration-stage graph construction from all
  `CouplingContractDeclaration` records and the initial `CouplingContext`.
- [x] Implement finalized graph construction or augmentation after additional
  fields, Forms installs, expert installs, and partitioned plan generation.
- [x] Merge declarations from multiple contracts without assuming pairwise
  coupling.
- [x] Validate that all required participants, fields, participant-local regions,
  and shared regions are resolvable in the context.
- [ ] Validate that each contract declaration's `contract_type` matches the
  registered `CouplingContract::name()` and that configured contract instance
  names are unique.
- [ ] Validate that all required monolithic non-field graph variables resolve
  through existing Systems/Analysis metadata: auxiliary state names,
  AuxiliaryInput names/slots, AuxiliaryOutput names/ids,
  BoundaryReductionService functional names, and global scalar keys where used,
  with canonical owning-system or participant-qualified
  `analysis::VariableKey` names.
- [ ] Validate that provenance-only non-field requirements such as parameters,
  coefficients, material-state old/work data, and boundary-integral provenance
  resolve to provider metadata without forcing them into graph-variable edges.
- [x] Validate that FE-backed participant, field, and region nodes retain
  owning-`FESystem` provenance so raw `FieldId`s, region markers, and interface
  markers from different systems are never treated as globally unique.
- [x] Validate that optional participants, fields, participant-local regions, and
  shared regions are tracked when present and ignored with diagnostics or
  metadata when absent.
- [ ] Validate that additional field declarations across all contracts are
  unique by namespace kind, namespace name, field name, registration target
  system, scope, and attachment, and do not collide with base physics fields.
- [ ] Validate that contract-owned additional fields use a contract instance
  namespace, are not treated as participants, and resolve to a concrete target
  `FESystem` before registration.
- [ ] Validate optional additional fields are either selected and fully
  compatible or skipped without leaving dependency edges, expected blocks, or
  Forms contributions that reference them.
- [ ] Validate that additional field declarations can be lowered to existing FE
  field registration APIs before the monolithic coupling builder mutates the
  `FESystem`.
- [ ] Validate that interface additional-field declarations resolve their
  interface marker from participant-region or shared-region context and never
  treat declaration-time raw markers as portable identity; the resolved marker
  is stored only in `systems::FieldSpec::interface_marker`.
- [ ] Validate additional-field attachment combinations during graph
  validation: `VolumeCell` fields reject region attachments and `InterfaceFace`
  fields require exactly one participant-region or shared-region attachment.
- [x] Validate monolithic Forms contracts only reference fields and regions from
  one compatible shared `FESystem`.
- [x] Validate interface-face monolithic contracts reference registered
  interface topology.
- [x] Represent implicit-monolithic dependency edges separately from
  external/lagged dependency edges.
- [x] Validate that external/lagged dependencies are not expected to produce
  monolithic Jacobian blocks.
- [x] Validate expected nonzero and expected zero block declarations against
  dependency declarations.
- [ ] Validate non-field dependency expectations against installed
  `analysis::VariableKey` edges and only require matrix-block evidence when the
  relevant Systems metadata reports a true linearized matrix contribution.
- [ ] Validate `CouplingNonFieldDependencyRequirement` records against installed
  participant/system-scoped non-field provenance, including expected parameter
  value type, coefficient value type, material-state byte offset, resolved
  `analysis::DomainKind`, region/shared-region name, marker, side, and logical
  region where applicable, and validate only requirements marked as graph
  variables against installed `analysis::VariableKey` dependency edges.
- [ ] Record installed Forms dependencies from the public Forms/Systems
  coupling-analysis metadata bridge so expected block diagnostics can compare
  declared versus installed structure, including geometry-sensitivity
  dependencies and non-field parameter, coefficient, boundary-functional,
  boundary-integral, auxiliary-state, `AuxiliaryInput`, `AuxiliaryOutput`,
  and material-state dependencies, plus Analysis/expert global-scalar variable
  dependencies, field/trial-scoped previous-solution temporal provenance, and
  geometry-terminal declaration location plus resolved location provenance,
  resolved `analysis::DomainKind`, and frame-transform configuration provenance,
  with installed dependency evidence carrying `analysis::DomainKind`, provider,
  and matrix/vector contribution flags.
- [ ] Extend or adapt `FE/Analysis/CouplingGraphAnalyzer` so fallback Analysis
  consumption includes `FormulationRecord::auxiliary_input_dependencies`,
  `FormulationRecord::auxiliary_output_dependencies`, and global-scalar
  dependencies from `ContributionDescriptor`, `FormStructureSummary`, or an
  equivalent public Analysis record, not just boundary-functional and
  auxiliary-state dependencies.
- [x] Preserve contribution name and diagnostic origin when adapting installed
  Forms/Systems metadata so graph diagnostics can attribute blocks and
  dependencies to the declaring contract and contribution.
- [x] Reuse/adapt existing `ContributionDescriptor`, `FormulationRecord`, and
  `FE/Analysis/CouplingGraphAnalyzer` output where possible for installed
  contribution analysis rather than creating an independent installed-form graph
  model.
- [ ] Record expert-hook dependencies only from returned
  `CouplingInstallMetadata` records, and require each record to include stable
  contribution name/origin/owning-system/operator provenance and resolved
  `analysis::VariableKey` rows/dependencies plus explicit `analysis::DomainKind`
  and matrix/vector provenance rather than declaration-side `CouplingVariableUse`
  records.
- [x] Detect missing expected blocks, undeclared installed blocks, and
  declared-but-unused implicit dependencies, including declared geometry-sensitive
  dependencies that the installed residual metadata does not report.
- [x] Aggregate temporal requirements by contract, participant, field, derivative
  order, mesh-motion owner scope, and mesh-motion role, including field-history
  depth, mesh temporal symbols, and effective-time-step requirements.
- [ ] Aggregate geometry-terminal requirements by contract and participant,
  including mesh displacement, coordinate, Jacobian, Jacobian-inverse, normal,
  measure, surface-Jacobian, cell-metric, and cell-domain-id terminals, with
  boundary/interface location, resolved `analysis::DomainKind`, owner-scope, side,
  typed `forms::GeometryConfiguration` value and frame-transform
  configurations, typed logical interface region, revision, and
  quadrature-policy provenance.
- [ ] Validate partitioned exchange declarations reference known participants,
  endpoints when declared, endpoint temporal slots, regions, value descriptors,
  and transfer declarations.
- [ ] Validate partitioned endpoint names as registry/provider-resolved keys for
  their endpoint kind while keeping their physical meaning opaque to
  `FE/Coupling`.
- [ ] Reject partitioned exchange declarations with `Unspecified` transfers and
  validate explicit `Identity` transfers against endpoint ownership, layout,
  value shape, and temporal-slot semantics.
- [ ] Validate field, typed `ParameterRegistry`, raw auxiliary-state,
  AuxiliaryInput, AuxiliaryOutput, and region-data partitioned endpoints against
  existing FE registries/providers, and validate external-buffer and driver-owned
  transfer endpoints against explicit driver-owned registries, rather than
  adding a parallel endpoint registry.
- [ ] Validate partitioned endpoint temporal slots against endpoint kind and
  registry support, including `history_index` only for History,
  `stage_index` only for Stage, logical 1-based history numbering, and the
  special restrictions for Field accepted/predicted/stage/external slots,
  RegionData provider-backed slots, interface-map accepted/trial revision state,
  Parameter typed-value and initial scalar-Real transfer defaults,
  AuxiliaryInput/AuxiliaryOutput provider support, and DriverOwned GeneralTensor
  payload semantics on
  external-buffer endpoints.
- [ ] Validate interface transfer provenance against source/target owning
  systems, interface search registry names, logical interface region ids,
  coordinate configurations, `InterfaceRevisionSnapshot` records, search
  revision keys, interface map revision/state, sliding-map kind, accepted/trial
  revision keys, search-registry identity, and time-level epoch before exposing a
  resolved plan.
- [x] Compare partitioned exchange declarations and group hints with
  generated resolved `PartitionedCouplingPlan` contents.
- [x] Detect partitioned exchange cycles and expose them to the driver without
  choosing a driver algorithm.
- [x] Validate group hints reference known participants and do not contain
  duplicates.
- [ ] Implement `CouplingDiagnostics` reports for graph summary, missing
  context values, dependency mismatches, temporal-policy failures, transfer
  failures, cycle visibility, and block-coverage mismatches.
- [x] Make diagnostics deterministic so unit tests and users see stable
  ordering.

Unit-test verification checklist:

- [x] `test_CouplingGraph.cpp` builds a graph for one valid two-participant
  contract.
- [ ] `test_CouplingGraph.cpp` builds a graph for multiple contracts sharing
  participants and shared regions.
- [ ] `test_CouplingGraph.cpp` distinguishes reusable contract type nodes from
  configured contract instance nodes and rejects duplicate instance names even
  when the contract type differs.
- [ ] `test_CouplingGraph.cpp` builds a graph for a valid multiway contract
  with three or more participants.
- [x] `test_CouplingGraph.cpp` rejects missing required participants, fields,
  participant-local regions, and shared regions.
- [x] `test_CouplingGraph.cpp` accepts absent optional participants, fields,
  participant-local regions, and shared regions.
- [ ] `test_CouplingGraph.cpp` rejects duplicate additional field declarations
  across contracts.
- [ ] `test_CouplingGraph.cpp` verifies participant-scoped and contract-owned
  additional field namespaces do not collide accidentally and that
  contract-owned fields resolve a concrete target `FESystem`.
- [ ] `test_CouplingGraph.cpp` verifies skipped optional additional fields leave
  no dependency edge, expected block, or Forms contribution reference.
- [ ] `test_CouplingGraph.cpp` rejects additional field declarations that
  cannot lower to existing FE field registration APIs.
- [ ] `test_CouplingGraph.cpp` verifies interface additional-field declarations
  carry only declaration-side region names before resolution and compare against
  resolved `systems::FieldSpec::interface_marker` provenance only after
  `CouplingContext` lookup.
- [ ] `test_CouplingGraph.cpp` verifies invalid additional-field attachment
  combinations fail before any `FESystem` field registration mutates state.
- [ ] `test_CouplingGraph.cpp` distinguishes declaration-stage validation from
  finalized graph validation for coupling-owned fields.
- [ ] `test_CouplingGraph.cpp` rejects monolithic Forms coupling across
  incompatible `FESystem` instances or unregistered interface topology.
- [ ] `test_CouplingGraph.cpp` verifies identical raw `FieldId`s or interface
  markers from two different `FESystem` instances are not treated as the same
  field or topology.
- [ ] `test_CouplingGraph.cpp` distinguishes implicit-monolithic dependency
  edges from external/lagged dependency edges.
- [x] `test_CouplingGraph.cpp` rejects expected monolithic blocks attached to
  external/lagged dependencies.
- [x] `test_CouplingGraph.cpp` detects missing expected blocks after Forms
  dependency analysis through the public Forms/Systems coupling-analysis
  metadata bridge.
- [ ] `test_CouplingGraph.cpp` validates declared non-field dependency
  expectations against installed `analysis::VariableKey` edges for
  BoundaryFunctional, AuxiliaryState, AuxiliaryInput, AuxiliaryOutput, and
  GlobalScalar variables.
- [ ] `test_CouplingGraph.cpp` validates provenance-only non-field requirements
  against provider metadata without requiring installed `analysis::VariableKey`
  edges for parameters, coefficients, material-state old/work data, and
  boundary-integral provenance unless explicitly marked as graph variables.
- [ ] `test_CouplingGraph.cpp` verifies location-sensitive non-field
  requirements use `analysis::DomainKind`, region/shared-region name, marker,
  side, logical region, provider, value-type, slot/output-id, and byte-offset
  provenance only as installed evidence, and that two dependencies with the same
  declaration name on different regions do not collide.
- [ ] `test_CouplingGraph.cpp` verifies boundary-integral provenance maps to
  BoundaryFunctional graph identity for expected-block diagnostics until
  Analysis exposes a distinct boundary-integral kind.
- [x] `test_CouplingGraph.cpp` validates installed matrix-block provenance uses
  `analysis::VariableKey` rows and columns so Systems-reported non-field matrix
  contributions are not lost.
- [ ] `test_CouplingGraph.cpp` verifies installed block provenance stores
  contributing domains as `analysis::DomainKind` values, including `Global`,
  `CoupledBoundary`, and `AuxiliaryCoupling` when Systems reports them.
- [ ] `test_CouplingGraph.cpp` verifies expert-hook
  `CouplingInstallMetadata` records are rejected when they lack resolved
  contribution identity, `analysis::VariableKey` identity,
  `analysis::DomainKind` provenance, or matrix/vector contribution flags.
- [ ] `test_CouplingGraph.cpp` verifies fallback
  `FE/Analysis/CouplingGraphAnalyzer` consumption includes
  `FormulationRecord::auxiliary_input_dependencies`,
  `FormulationRecord::auxiliary_output_dependencies`, and global-scalar
  dependencies from `ContributionDescriptor`, `FormStructureSummary`, or an
  equivalent public Analysis record.
- [x] `test_CouplingGraph.cpp` verifies installed-form diagnostics consume
  existing `ContributionDescriptor`/`FormulationRecord` analysis metadata or a
  direct public adapter with equivalent provenance.
- [x] `test_CouplingGraph.cpp` verifies contribution name, origin, and
  owning-system provenance distinguish two contracts that install into the same
  operator tag.
- [ ] `test_CouplingGraph.cpp` treats Systems-reported geometry-sensitivity
  dependencies as valid implicit dependencies for expected-block diagnostics.
- [x] `test_CouplingGraph.cpp` detects undeclared installed blocks.
- [x] `test_CouplingGraph.cpp` aggregates temporal requirements across
  multiple contracts.
- [ ] `test_CouplingGraph.cpp` aggregates explicit field-history and mesh
  temporal requirements across multiple contracts.
- [ ] `test_CouplingGraph.cpp` aggregates geometry-terminal requirements across
  multiple contracts and reports unsupported geometry terminals independently
  from temporal-policy failures, preserving boundary/interface location, resolved
  `analysis::DomainKind`, owner-scope, frame-transform configuration, and typed
  geometry-revision provenance.
- [x] `test_CouplingGraph.cpp` aggregates effective-time-step requirements.
- [x] `test_CouplingGraph.cpp` compares partitioned exchange declarations with a
  generated resolved partitioned plan.
- [ ] `test_CouplingGraph.cpp` rejects partitioned declarations with invalid
  endpoint temporal slots or invalid history/stage indices.
- [ ] `test_CouplingGraph.cpp` rejects `Unspecified` transfers, validates
  explicit `Identity` compatibility, and verifies interface transfers carry
  distinct interface-entry and interface-map plus interface-search-registry
  provenance, logical interface region ids, coordinate configurations, revision
  snapshots, search map state, and sliding-map kind.
- [ ] `test_CouplingGraph.cpp` rejects stale interface maps, wrong-system maps,
  and trial/accepted interface-map state mismatches.
- [x] `test_CouplingGraph.cpp` detects partitioned exchange cycles and reports
  them deterministically.
- [ ] `test_CouplingDiagnostics.cpp` verifies stable, actionable diagnostic
  text for each validation failure category.

### Phase 5: Monolithic Coupling Builder And FESystem Lifecycle

Implementation checklist:

- [x] Implement `MonolithicCouplingBuilder` as the owner of the pre-setup
  coupling lifecycle.
- [ ] Ensure the builder runs after base physics field registration and before
  `FESystem::setup()`.
- [ ] Build the initial `CouplingContext` from registered participants, fields,
  regions, and shared regions.
- [x] Require every FE-backed participant in the context to carry its owning
  `FESystem` before field, region, interface, or Forms resolution begins.
- [ ] Collect declarations from all selected coupling contracts before adding
  any coupling-owned fields.
- [x] Build and validate a declaration-stage `CouplingGraph` without requiring
  coupling-owned `FieldId`s to exist yet.
- [x] Register all validated `additional_fields` declarations with their
  resolved target `FESystem`; contracts must not mutate `FESystem` directly
  during this step, and contract-owned field namespaces must remain distinct
  from participant names.
- [ ] Refresh and finalize `CouplingContext` after additional fields are
  registered.
- [ ] Rebuild and validate the finalized `CouplingGraph` after additional
  fields are present.
- [ ] Validate temporal requirements against the selected temporal policy if
  the policy is available before setup.
- [ ] Allow base physics modules to install their residual forms before
  coupling Forms are installed.
- [x] Collect `CouplingFormContribution` objects from all contracts that
  support Forms-based monolithic lowering.
- [x] Resolve `CouplingFormContribution` objects to
  `ResolvedCouplingFormContribution` records with concrete ordered primary
  `FieldId` lists, concrete ordered extra trial `FieldId` lists, resolved
  `systems::FormInstallOptions`, captured terminal-provenance declarations,
  contribution names, diagnostic origins, and owning system names.
- [x] Resolve `extra_trial_field_uses` into
  `systems::FormInstallOptions::extra_trial_fields` so dependency-only
  `StateField`, `DiscreteField`, geometry-sensitivity, and temporal field
  dependencies are visible to `FormsInstaller` without becoming residual rows.
- [ ] Validate every `field_uses` and `extra_trial_field_uses` list covers all
  field-bound `TestField` symbols, implicit `StateField` dependencies,
  geometry-sensitivity dependencies, and temporal field operands in the installed
  residual; reject missing fields, ambiguous aliases, incompatible primary/extra
  roles, and nondeterministic duplicate ordering.
- [x] Install resolved coupling Forms through `installFormulation()` so the
  existing Forms pipeline owns active-field detection, sparsity, residual
  decomposition, and Jacobian block generation.
- [x] Forward `ResolvedCouplingFormContribution::install_options` unchanged to
  `installFormulation()`, including resolved
  top-level `ad_mode`, resolved `compiler_options.geometry_sensitivity`
  settings, resolved `compiler_options.geometry_tangent_path`, resolved
  `compiler_options.use_symbolic_tangent`, and resolved `extra_trial_fields`,
  while preventing `compiler_options.ad_mode` from acting as a second override
  channel.
- [x] Record block metadata returned by `installFormulation()`/`MixedKernelPlan`
  and dependency, non-field dependency, geometry-sensitivity, and temporal
  provenance returned by the public Forms/Systems coupling-analysis metadata
  bridge, preserving contribution name, origin, owning-system provenance, and
  builder terminal-provenance declarations for diagnostics.
- [ ] Dispatch `installMonolithicTerms()` after Forms-based installation for
  contracts that require expert custom terms.
- [ ] Require expert custom install hooks to return one `CouplingInstallMetadata`
  record per installed custom contribution, with contribution identity, resolved
  `analysis::VariableKey` residual rows/dependencies, domain provenance, and
  matrix/vector contribution flags, and record those installed dependencies for
  graph diagnostics.
- [ ] Ensure expert custom install hooks use `MonolithicCouplingInstallContext`
  and approved Systems extension points rather than internal raw kernel
  registration methods.
- [ ] Refresh the finalized graph after Forms and expert install metadata are
  available.
- [ ] Validate expected block coverage before `FESystem::setup()`.
- [ ] Ensure builder failures leave `FESystem` in a clear pre-setup failed
  state and do not partially run setup.
- [x] Keep monolithic coupling independent of partitioned driver policy.

Unit-test verification checklist:

- [ ] `test_MonolithicCouplingBuilder.cpp` verifies the lifecycle order from
  context creation through pre-setup validation.
- [x] `test_MonolithicCouplingBuilder.cpp` verifies additional fields are
  registered before `FESystem::setup()`.
- [ ] `test_MonolithicCouplingBuilder.cpp` verifies participant-scoped and
  contract-owned additional fields register into the resolved target `FESystem`
  and are exposed under the correct lookup namespace after context refresh.
- [x] `test_MonolithicCouplingBuilder.cpp` verifies interface additional fields
  resolve participant/shared-region declarations into markers only during
  builder resolution.
- [ ] `test_MonolithicCouplingBuilder.cpp` verifies contract-owned interface
  fields without an explicit target participant are accepted only when the
  shared-region participants resolve to one monolithic `FESystem`.
- [x] `test_MonolithicCouplingBuilder.cpp` verifies contracts cannot directly
  register additional fields on `FESystem`.
- [ ] `test_MonolithicCouplingBuilder.cpp` verifies duplicate coupling fields
  fail before setup.
- [x] `test_MonolithicCouplingBuilder.cpp` verifies Forms contributions are
  installed through the standard `installFormulation()` path.
- [ ] `test_MonolithicCouplingBuilder.cpp` verifies
  `systems::FormInstallOptions`, including top-level `ad_mode` and
  `compiler_options.geometry_sensitivity`, `compiler_options.geometry_tangent_path`,
  and `compiler_options.use_symbolic_tangent`, are resolved from
  declaration-time options and forwarded unchanged to `installFormulation()`.
- [ ] `test_MonolithicCouplingBuilder.cpp` verifies
  `compiler_options.ad_mode` cannot override top-level
  `systems::FormInstallOptions::ad_mode` during resolution or forwarding.
- [ ] `test_MonolithicCouplingBuilder.cpp` verifies Forms contributions are
  resolved to concrete ordered `FieldId` lists before calling
  `installFormulation()`.
- [ ] `test_MonolithicCouplingBuilder.cpp` verifies
  `extra_trial_field_uses` resolve to
  `systems::FormInstallOptions::extra_trial_fields` and are forwarded to
  `installFormulation()` for dependency-only trial fields.
- [x] `test_MonolithicCouplingBuilder.cpp` rejects a Forms contribution whose
  combined primary and extra-trial field-use lists omit a `TestField`,
  implicit `StateField`, geometry-sensitivity dependency, or temporal field
  operand reported by the metadata bridge.
- [ ] `test_MonolithicCouplingBuilder.cpp` rejects incompatible overlap between
  primary field uses and dependency-only extra trial field uses.
- [ ] `test_MonolithicCouplingBuilder.cpp` verifies duplicate `field_uses` and
  `extra_trial_field_uses` entries are deduplicated deterministically or rejected
  with stable diagnostics.
- [ ] `test_MonolithicCouplingBuilder.cpp` verifies geometry-sensitivity install
  options require a resolved mesh-motion field in `field_uses` or
  `extra_trial_field_uses` and a matching `FESystem::bindMeshMotionField()`
  binding.
- [ ] `test_MonolithicCouplingBuilder.cpp` verifies coupling builder mesh and
  geometry terminal helpers require explicit owner scope in N-participant
  contracts and preserve participant/region/location provenance in the metadata
  bridge output.
- [ ] `test_MonolithicCouplingBuilder.cpp` verifies captured builder terminal
  provenance is attached to resolved form contributions and included in
  installed diagnostics.
- [x] `test_MonolithicCouplingBuilder.cpp` verifies contribution name, origin, and
  owning-system provenance are retained when installed metadata is adapted for
  graph diagnostics.
- [ ] `test_MonolithicCouplingBuilder.cpp` verifies expert custom install hooks
  are dispatched after Forms contributions and must return one install metadata
  record per custom contribution.
- [ ] `test_MonolithicCouplingBuilder.cpp` verifies Forms dependency/block
  metadata, including non-field dependency provenance and structured
  geometry-sensitivity provenance, is recorded from the public Forms/Systems
  coupling-analysis metadata bridge.
- [ ] `test_MonolithicCouplingBuilder.cpp` verifies expert hooks cannot bypass
  `MonolithicCouplingInstallContext`.
- [x] `test_MonolithicCouplingBuilder.cpp` verifies missing expected blocks
  fail before setup.
- [ ] `test_MonolithicCouplingBuilder.cpp` verifies external/lagged
  dependencies do not produce expected block requirements.
- [ ] `test_MonolithicCouplingBuilder.cpp` verifies temporal-policy failures
  stop setup before assembly.
- [x] `test_MonolithicCouplingBuilder.cpp` verifies a minimal two-field
  coupling produces the expected off-diagonal Jacobian block metadata.
- [ ] `test_MonolithicCouplingBuilder.cpp` verifies an N-contract graph can
  install multiple coupling contributions into one shared `FESystem`.

### Phase 6: Partitioned Coupling Plan Generation

Implementation checklist:

- [x] Implement partitioned plan generation as metadata only; do not implement
  the partitioned nonlinear/outer-iteration driver in this phase.
- [x] Keep partitioned transfer execution out of Phase 6 unless a later phase
  explicitly adds the driver contract, rollback/state-acceptance semantics,
  adapter hooks, and tests; plan-generation APIs should return data structures
  and diagnostics, not perform data movement.
- [x] Collect declarative `CouplingExchangeDeclaration` records from
  `CouplingContractDeclaration` and any
  `buildPartitionedExchangeDeclarations()` hooks on contracts that support
  partitioned lowering.
- [x] Merge per-contract exchange declarations/templates into a global
  partitioned exchange graph before endpoint and transfer resolution.
- [x] Compare generated resolved exchanges and group hints against the contract
  declaration's partitioned exchange declarations, hook-produced declarations,
  and `group_hints`.
- [x] Implement an FE-owned `PartitionedCouplingPlanGenerator` that resolves all
  endpoint refs, shared regions, transfer declarations, interface-map
  provenance, and driver-owned descriptors from the finalized
  `CouplingContext`; contracts must never return already resolved executable
  `CouplingExchange` records.
- [x] Validate every producer and consumer endpoint against the finalized
  `CouplingContext`, including explicitly registered driver-owned
  global/participant-scoped external-buffer and transfer-operator registries.
- [x] Resolve every producer and consumer endpoint into a
  `ResolvedCouplingEndpoint` carrying the kind-specific FE or driver identity,
  explicit resolved endpoint kind, registry-resolved stable endpoint key,
  resolved participant/global endpoint scope, and resolved temporal backing
  required for execution, with declaration provenance retained only for
  diagnostics; generated `CouplingExchange` records must not retain
  declaration-only endpoint refs as their executable endpoint representation.
- [x] Preserve owning-system provenance for every FE-backed endpoint and region
  so endpoint lookup, raw `FieldId`s, markers, and interface topology are always
  resolved relative to the correct `FESystem`.
- [x] Validate endpoint participant scope: FE-backed field, region-data,
  auxiliary, and parameter endpoints require a known participant; external-buffer
  endpoints may be context-level global or participant-scoped and must resolve in
  exactly that declared scope.
- [x] Resolve field endpoints through `FESystem`/`FieldRegistry`, parameter
  endpoints through `ParameterRegistry` with `params::ValueType` validation, raw
  auxiliary-state endpoints through auxiliary block storage, AuxiliaryInput
  endpoints through `AuxiliaryInputRegistry`, AuxiliaryOutput endpoints through
  the deployed auxiliary output registry, and region-data endpoints through
  `FEQuantityRegistry`, `BoundaryReductionService`, or another existing FE
  quantity/reduction provider.
- [x] Validate `std::size_t(-1)` lookup failures and integer narrowing for
  AuxiliaryInput slots and AuxiliaryOutput stable ids/materialized-output slots
  before writing `std::uint32_t` metadata or Forms refs.
- [x] Preserve the distinction between stable AuxiliaryOutput deploy ids and
  flattened materialized-output buffer slots during endpoint resolution; use the
  stable id for durable identity and current Forms `AuxiliaryOutputRef` metadata,
  and use the flattened slot only for direct materialized-buffer access.
- [x] Preserve the resolved endpoint identity for each non-field provider:
  raw auxiliary block state versus AuxiliaryInput slot versus AuxiliaryOutput
  id/name versus auxiliary extension key, FEQuantity id versus
  BoundaryReductionService functional name/primary field versus region-data
  provider extension key.
- [x] Resolve external-buffer endpoints through the explicit driver-owned
  external-buffer registry scope and validate scalar type, access direction,
  lifetime, distribution, extents, strides, packing, supported indexed
  temporal-slot descriptors, payload descriptor, and layout/data revision keys.
- [x] Treat endpoint names as physics-opaque but registry-resolved keys; reject
  endpoint kinds whose names cannot be resolved by the appropriate registry.
- [x] Validate producer and consumer value descriptors have compatible rank and
  component counts, and require documented component layout for mixed payloads or
  pass-through components.
- [x] Validate interface transfer value ranks against `InterfaceFieldKind`;
  reject `SymmetricTensor` unless explicit rank-2 packing metadata is present,
  require explicit source embedding and target restriction policies for 2D vector
  frame transforms, and reject `GeneralTensor` for FE interface transfer kinds.
- [x] Validate driver-owned general tensor transfers have positive extents,
  component count equal to the extent product, explicit packing metadata, matching
  external-buffer descriptor layout, and an explicitly named
  `CouplingTransferKind::DriverOwned` operator in the driver-owned transfer
  registry.
- [x] Reject driver-owned transfer execution in Phase 6 even when descriptors
  validate, unless the named driver-owned transfer operator also provides an
  explicit executable adapter registered with the driver-facing extension point.
- [x] Validate producer and consumer region scopes are present and compatible
  with the requested transfer declaration.
- [x] Validate interface transfer declarations against the existing
  `FE/Systems/InterfaceOperators` kinds and resolved options.
- [x] Validate interface transfer declarations include or derive complete
  `systems::InterfaceTransferOptions`, including frame transform policy, rotation,
  component count, field kind, and conservation tolerance.
- [x] Validate interface transfers carry source/target interface markers and
  durable interface-entry and interface-map plus interface-search-registry
  provenance, source/target system names,
  `LogicalInterfaceRegionId` records, `InterfaceRevisionSnapshot` records,
  coordinate configurations, search revision keys, map revision key, search map
  state, sliding-map kind, operator state, accepted/trial revision keys, and
  epoch resolvable through the current context to the `svmp::search::InterfaceMap`
  used by `applyInterfaceTransfer()`.
- [x] Resolve interface transfer runtime handles to source/target `FESystem`
  pointers plus the current `InterfaceSearchRegistry`, optional
  `SlidingInterfaceMap` state wrapper, and concrete `svmp::search::InterfaceMap`
  passed to `applyInterfaceTransfer()`, while keeping those handles out of
  durable plan identity.
- [x] Reject interface transfer resolutions whose provenance is stale, whose map
  belongs to a different source/target system, or whose trial/accepted state does
  not match the requested temporal slot.
- [x] Validate interface frame transforms reject scalar and mixed-block payloads,
  require at least the transformed vector component count for vector transforms,
  require at least the transformed rank-2 tensor component count for rank-2
  tensor transforms, and preserve extra components only when their
  component-layout metadata documents pass-through semantics; true 2D vector
  transforms require explicit source embedding and target restriction policies.
- [x] Preserve true 2D vector source embedding and target restriction policies
  in `ResolvedCouplingTransfer` as coupling-side execution metadata separate
  from `systems::InterfaceTransferOptions`, and reject executable FE interface
  transfers that need those policies when no coupling-side adapter is available
  to apply them around `applyInterfaceTransfer()`.
- [x] Reject `Unspecified` transfer declarations during plan generation and
  validate explicit `Identity` transfers against endpoint ownership, layout,
  value shape, and temporal-slot semantics.
- [x] Validate `DriverOwned` transfer declarations against the explicit
  driver-owned transfer registry, including supported ranks, supported
  source/target temporal slots, component-layout preservation, general-tensor
  support, and transfer-registry revision keys.
- [ ] Validate endpoint temporal slots for every producer and consumer endpoint,
  including current, accepted, predicted, history, stage, and external data
  requirements, with logical 1-based history indices mapped to 0-based storage,
  `history_index` present only for History, `stage_index` present only for
  Stage, no separate endpoint slots for `u_prev`/`u_prev2`, and
  endpoint-kind-specific registry support for Field, RegionData, raw
  AuxiliaryState, AuxiliaryInput, AuxiliaryOutput, ExternalBuffer, and Parameter
  endpoints, explicit predicted backing kinds, plus interface-map transfer-state
  rules and DriverOwned GeneralTensor payload descriptors.
- [x] Preserve contract-owned port names as opaque labels in the FE layer.
- [x] Preserve directed exchanges and cycles so the higher-level driver can
  choose solve order and convergence policy.
- [x] Merge group hints from multiple contracts and reject hints with unknown
  participants or duplicate names when duplicates are ambiguous.
- [x] Expose a driver-facing summary of exchanges, endpoints, transfer
  declarations, resolved transfer options, group hints, and cycles without
  embedding Gauss-Seidel, Jacobi, Aitken, quasi-Newton, relaxation, convergence,
  or rollback policy.
- [x] Ensure partitioned and monolithic modes can be requested independently
  from the same contract declaration when the contract supports both modes.

Unit-test verification checklist:

- [x] `test_PartitionedCouplingPlan.cpp` validates a single two-endpoint
  exchange.
- [x] `test_PartitionedCouplingPlan.cpp` validates multiple exchanges from one
  contract.
- [x] `test_PartitionedCouplingPlan.cpp` validates merged exchanges from
  multiple contracts.
- [x] `test_PartitionedCouplingPlan.cpp` rejects generated exchanges that
  contradict declared producers, consumers, shared regions, value descriptors,
  transfer declarations, or resolved transfer options.
- [x] `test_PartitionedCouplingPlan.cpp` rejects unknown producer and consumer
  participants for endpoint kinds that require participant scope.
- [x] `test_PartitionedCouplingPlan.cpp` verifies endpoint resolution delegates
  to existing FE registries for fields, typed `ParameterRegistry` values, raw
  auxiliary state, AuxiliaryInput, AuxiliaryOutput, and FE quantities.
- [x] `test_PartitionedCouplingPlan.cpp` verifies contracts provide only
  declarative exchange records and that the FE-owned plan generator performs
  all endpoint, transfer, shared-region, interface-map, and driver-owned
  descriptor resolution.
- [x] `test_PartitionedCouplingPlan.cpp` verifies generated exchanges store
  `ResolvedCouplingEndpoint` producer/consumer records with resolved field ids,
  FE quantity ids, BoundaryReductionService functional identities, parameter
  slots/value types, raw auxiliary block indices, AuxiliaryInput slots,
  AuxiliaryOutput stable ids/names and optional flattened materialized-output
  slots, auxiliary extension keys, or external-buffer descriptors as appropriate,
  plus explicit resolved endpoint kinds, registry-resolved stable endpoint keys,
  resolved participant/global endpoint scope, temporal backing, and revision
  provenance.
- [x] `test_PartitionedCouplingPlan.cpp` verifies
  `ResolvedCouplingEndpoint::declaration_provenance` is diagnostic/request
  provenance only and is not used for execution lookup, durable plan equality,
  serialization identity, or cache keys.
- [x] `test_PartitionedCouplingPlan.cpp` verifies durable endpoint identity uses
  resolved endpoint kind, resolved participant/global scope, system name,
  registry provider, registry-resolved endpoint key, kind-specific resolved ids,
  value descriptor, temporal backing, and revision metadata rather than
  declaration-only names or runtime pointers.
- [x] `test_PartitionedCouplingPlan.cpp` verifies AuxiliaryInput endpoint
  resolution rejects lookup failures and out-of-range slot narrowing.
- [x] `test_PartitionedCouplingPlan.cpp` verifies AuxiliaryOutput endpoint
  resolution preserves stable deploy ids separately from flattened
  materialized-output buffer slots, uses stable ids for durable identity and
  current Forms ref metadata, and rejects lookup failures or out-of-range
  narrowing for both ids and slots.
- [x] `test_PartitionedCouplingPlan.cpp` verifies endpoint names remain
  physics-opaque while resolving as field names, parameter keys, auxiliary keys,
  AuxiliaryInput names, AuxiliaryOutput names, FE quantity names,
  BoundaryReductionService functional names, provider-extension keys, or
  scoped external-buffer keys.
- [x] `test_PartitionedCouplingPlan.cpp` verifies parameter endpoint resolution
  records `params::ValueType` from `ParameterRegistry::Spec`, accepts scalar
  Real only for the initial built-in transfer path, and rejects unsupported
  non-Real typed payloads unless an explicit typed parameter extension or
  driver-owned transfer mapping is registered.
- [x] `test_PartitionedCouplingPlan.cpp` verifies external-buffer endpoint
  descriptors reject incompatible scalar type, access, lifetime, distribution,
  extents, strides, packing, indexed temporal-slot descriptors, payload shape,
  and stale layout/data revisions.
- [x] `test_PartitionedCouplingPlan.cpp` verifies global and participant-scoped
  external-buffer endpoints resolve through the declared scope, reject missing
  scoped descriptors, and allow the same key in different scopes without
  ambiguity.
- [x] `test_PartitionedCouplingPlan.cpp` rejects incompatible value descriptor
  ranks and component counts.
- [x] `test_PartitionedCouplingPlan.cpp` rejects missing or unsupported transfer
  declarations.
- [x] `test_PartitionedCouplingPlan.cpp` rejects `Unspecified` transfer
  declarations and accepts explicit `Identity` only for compatible endpoint
  ownership, layout, value shape, and temporal-slot semantics.
- [x] `test_PartitionedCouplingPlan.cpp` verifies interface transfer declarations
  map to the existing `FE/Systems/InterfaceOperators` vocabulary.
- [x] `test_PartitionedCouplingPlan.cpp` verifies interface transfer options are
  preserved in the resolved plan and invalid option combinations fail
  validation.
- [ ] `test_PartitionedCouplingPlan.cpp` verifies
  `source_embedding_policy` and `target_restriction_policy` remain present in
  `ResolvedCouplingTransfer` after `systems::InterfaceTransferOptions`
  construction, and validates the reject path when a true 2D transfer would
  otherwise reach execution without a coupling-side adapter.
- [x] `test_PartitionedCouplingPlan.cpp` verifies resolved interface transfers
  include source/target interface markers plus distinct interface-entry,
  interface-map, and interface-search-registry provenance, source/target system names, logical
  interface region ids, coordinate configurations, `InterfaceRevisionSnapshot`
  records, revision keys, search map state, sliding-map kind, operator state,
  and accepted/trial epoch metadata, with non-owning runtime system,
  search-registry, optional sliding-map state, and concrete interface-map handles
  resolved separately from durable plan metadata.
- [x] `test_PartitionedCouplingPlan.cpp` verifies frame-transform policies are
  accepted only for compatible vector or rank-2 tensor payloads and preserve
  pass-through components only when component layout is documented.
- [x] `test_PartitionedCouplingPlan.cpp` rejects stale interface maps,
  wrong-system maps, and trial/accepted interface-map state mismatches.
- [ ] `test_PartitionedCouplingPlan.cpp` verifies interface transfer declarations
  fail validation cleanly when mesh/interface-operator support is disabled.
- [x] `test_PartitionedCouplingPlan.cpp` rejects incompatible region scopes for
  a requested transfer declaration.
- [x] `test_PartitionedCouplingPlan.cpp` validates driver-owned transfer
  descriptors for supported ranks, source/target temporal slots,
  component-layout preservation, general-tensor support, and stale registry
  revision keys.
- [x] `test_PartitionedCouplingPlan.cpp` rejects invalid endpoint temporal-slot
  combinations and validates current, accepted, predicted, history, stage, and
  external endpoints, including `history_index` only for History, `stage_index`
  only for Stage, logical 1-based history mapping, endpoint-kind-specific
  support/rejection rules for Field, RegionData, raw AuxiliaryState,
  AuxiliaryInput, AuxiliaryOutput, ExternalBuffer, and Parameter endpoints, and
  DriverOwned GeneralTensor payload temporal semantics through external-buffer
  and driver-owned transfer-registry metadata.
- [ ] `test_PartitionedCouplingPlan.cpp` verifies `Predicted` temporal slots
  resolve to explicit predicted backing kinds or a declared provider-defined
  backing and never reuse Current or Accepted backing metadata.
- [x] `test_PartitionedCouplingPlan.cpp` rejects general tensors for FE interface
  transfers and accepts them only for explicitly driver-owned transfers with
  validated extents and packing metadata.
- [x] `test_PartitionedCouplingPlan.cpp` preserves a directed exchange cycle in
  graph output.
- [x] `test_PartitionedCouplingPlan.cpp` validates group hints and rejects
  unknown participants.
- [x] `test_PartitionedCouplingPlan.cpp` verifies no driver iteration policy is
  required to build or validate the FE partitioned plan.
- [x] `test_PartitionedCouplingPlan.cpp` verifies Phase 6 APIs do not execute
  partitioned transfers and reject attempted driver-owned, true-2D-adapted, or
  non-Real parameter transfer execution unless an explicit tested adapter is
  registered.

### Phase 7: Physics-Specific Coupling Modules

Implementation checklist:

- [x] Implement `FSICouplingOptions` with mode, configured contract instance
  name, participant names,
  optional mesh participant, interface shared-region name,
  multiplier option, transfer declarations, and endpoint temporal-slot choices.
- [x] Implement `FSICouplingModule` as a `CouplingContract` under
  `Physics/Coupling`.
- [x] Keep FSI physical names, FSI port names, and FSI-specific option parsing
  inside `Physics/Coupling`.
- [x] Make FSI declaration required fields include fluid velocity, fluid
  pressure, solid displacement and/or solid velocity, interface shared region,
  and optional mesh displacement when ALE is enabled.
- [x] Make FSI declaration include an optional interface multiplier field when
  the multiplier formulation is requested.
- [x] Validate multiplier options through `FSILagrangeMultiplierOptions`:
  `enabled`, optional contract-owned field namespace override, empty-namespace
  fallback to `FSICouplingOptions::contract_name`, optional registration target
  participant, field name, function space, component count, shared-region policy,
  and trace/mortar compatibility must be explicit when the multiplier formulation
  is enabled.
- [ ] Validate FSI field component counts against spatial/interface dimension.
- [ ] Validate FSI interface region existence and participant-region mappings.
- [ ] Validate ALE FSI fails when the mesh participant or mesh displacement
  field is required but absent.
- [ ] Author at least one Forms-based FSI-like monolithic kinematic constraint
  through `CouplingFormBuilder` using explicit Forms interface measures and
  trace/side restrictions.
- [ ] Validate monolithic FSI fields are registered in one compatible
  `FESystem` and that the FSI interface marker has registered interface
  topology before authoring `.dI(marker)` residuals.
- [ ] Author or install monolithic dynamic/traction balance terms through Forms
  when expressible, or through the expert hook with explicit dependency
  metadata when not yet expressible.
- [x] Declare temporal requirements when FSI uses `dt(solid_displacement)`
  instead of an independent solid velocity field.
- [x] Build a partitioned FSI exchange plan with opaque ports for
  solid-displacement, solid-velocity, fluid-traction/load, and
  mesh-displacement exchanges as applicable.
- [ ] Validate FSI partitioned transfer declarations for each requested direction,
  using resolved FE interface operators for interface interpolation,
  conservative projection, and mortar transfers.
- [x] Reject FSI partitioned plans with `Unspecified` transfers; require each
  transfer direction to choose an explicit interface transfer, explicit
  `Identity` with compatible endpoints, or an explicitly named driver-owned
  transfer.
- [ ] Validate FSI partitioned endpoint temporal slots so current, accepted,
  predicted, history, stage, and external data are explicit rather than inferred
  from port names.
- [x] Implement a small thermal-interface or thermal-fluid/thermo-elastic
  contract example under `Physics/Coupling` to exercise a second coupling
  family.
- [x] Make the second coupling contract reuse `FE/Coupling` context, graph,
  Forms, temporal, shared-region, and partitioned-plan infrastructure without
  adding physics-specific concepts to the FE layer.
- [x] Provide at least one N-participant or multi-contract example declaration
  that combines FSI, mesh motion, and another physics participant.

Unit-test verification checklist:

- [x] `test_FSICouplingModule.cpp` verifies a valid monolithic FSI declaration.
- [x] `test_FSICouplingModule.cpp` verifies a valid partitioned FSI
  declaration.
- [x] `test_FSICouplingModule.cpp` verifies multiplier-field declaration when
  `FSILagrangeMultiplierOptions::enabled` is true and validates the configured
  contract-owned namespace fallback/override, registration target `FESystem`
  policy, field name, function space, component count, shared-region policy, and trace/mortar
  compatibility.
- [x] `test_FSICouplingModule.cpp` verifies no multiplier-field declaration
  when the multiplier option is disabled.
- [x] `test_FSICouplingModule.cpp` rejects missing fluid velocity, fluid
  pressure, solid displacement, solid velocity when required, and interface
  shared-region mappings.
- [ ] `test_FSICouplingModule.cpp` rejects ALE FSI without mesh displacement.
- [x] `test_FSICouplingModule.cpp` verifies temporal requirements are declared
  when `dt(solid_displacement)` is used.
- [ ] `test_FSICouplingModule.cpp` verifies Forms-authored FSI-like
  monolithic residuals use the expected interface measure and report the
  expected implicit dependencies, including structured geometry-sensitivity
  provenance when ALE mesh motion is handled through the Systems
  geometry-sensitivity path.
- [ ] `test_FSICouplingModule.cpp` rejects Forms-authored monolithic FSI when
  participants are not in one compatible `FESystem` or the interface topology
  is not registered.
- [ ] `test_FSICouplingModule.cpp` verifies expected FSI monolithic block
  declarations match installed dependencies.
- [ ] `test_FSICouplingModule.cpp` verifies FSI partitioned exchange ports,
  endpoints, endpoint temporal slots, transfer declarations, resolved transfer
  options, value descriptors, and group hints.
- [x] `test_FSICouplingModule.cpp` rejects `Unspecified` FSI partitioned
  transfers and validates explicit `Identity` only for compatible endpoints.
- [x] `test_FSICouplingModule.cpp` verifies unsupported transfer declarations or
  unsupported mode combinations fail validation.
- [x] `test_ThermalInterfaceCouplingModule.cpp` verifies the second
  physics-specific contract declares, validates, and lowers through the common
  FE coupling APIs.
- [x] `test_ThermalInterfaceCouplingModule.cpp` verifies the second contract
  can coexist with FSI in one global graph.

### Phase 8: End-To-End Unit Verification And Completion Criteria

Implementation checklist:

- [x] Add a minimal scalar two-participant monolithic coupling test fixture
  that does not depend on full FSI physics.
- [ ] Add a minimal scalar or vector N-participant graph fixture with at least
  three participants and two contracts.
- [x] Add a minimal partitioned-plan fixture with a directed exchange cycle and
  group hints.
- [ ] Add a temporal-policy and geometry-terminal fixture that supports first
  derivative, second derivative, time, time-step, previous-solution history, mesh
  temporal, mesh displacement, coordinate/Jacobian/Jacobian-inverse, normal,
  measure, surface-Jacobian, cell-metric, and cell-domain-id validation cases,
  including boundary/interface terminal location, resolved
  `analysis::DomainKind`, owner-scope, and geometry-revision provenance.
- [x] Add reusable test helpers for constructing synthetic participants,
  fields, regions, shared regions, contracts, transfer declarations, resolved
  transfer options, endpoint temporal slots, and driver-owned transfer
  registries.
- [ ] Make test helpers construct participant-to-`FESystem` bindings,
  reusable contract type keys, unique contract instance namespaces,
  participant-scoped and contract-owned additional-field declarations with
  explicit registration target policies,
  distinct interface-entry and interface-map plus interface-search-registry
  provenance
  including logical region ids, coordinate configurations, revision snapshots,
  search map state, and sliding-map kind, `CouplingTemporalSlotDescriptor`
  records with logical 1-based history indices, explicit
  `Unspecified`/`Identity` transfer cases, declaration-time form install options
  plus resolved
  `systems::FormInstallOptions` with top-level AD mode and geometry-sensitivity
  settings, diagnostic-only `compiler_options.ad_mode` consistency,
  contribution names/origins, owning-system provenance,
  registry-resolved endpoint keys, `ResolvedCouplingEndpoint` records including
  declaration endpoint participant scope, explicit resolved endpoint kind,
  resolved participant/global endpoint scope, diagnostic declaration provenance,
  auxiliary and region-data provider identities, resolved temporal backing,
  external-buffer descriptors, structured
  geometry-sensitivity provenance, geometry-terminal location, resolved
  `analysis::DomainKind`, owner-scope provenance, frame-transform configuration,
  resolved expert-install metadata, and value descriptors with component-layout metadata and
  driver-owned `GeneralTensor` descriptors.
- [x] Ensure test helpers do not encode FSI-specific physical vocabulary in FE
  coupling tests.
- [x] Add build-system entries so all FE and physics coupling unit tests run in
  the normal solver unit-test workflow.
- [ ] Document the final public extension points for adding a new coupling
  contract: declaration, validation, additional fields, monolithic Forms,
  expert monolithic hooks, partitioned plan, and tests.

Unit-test verification checklist:

- [x] The full FE coupling unit-test suite passes locally.
- [x] The full physics coupling unit-test suite passes locally.
- [x] The minimal scalar monolithic fixture verifies off-diagonal block
  presence for an implicit cross-field dependency.
- [x] The minimal scalar monolithic fixture verifies no off-diagonal block is
  expected for an external/lagged dependency.
- [ ] The minimal N-participant fixture verifies graph validation, shared-region
  reuse, expected block diagnostics, and deterministic diagnostics.
- [x] The minimal partitioned fixture verifies exchange graph construction,
  transfer validation, resolved endpoint records with temporal backing, endpoint
  temporal-slot validation, cycle visibility, and group hint validation.
- [ ] The minimal partitioned fixture verifies accepted-state temporal slots,
  distinct interface-entry and interface-map plus interface-search-registry
  provenance, logical region ids, revision snapshots, search map state,
  `Unspecified` rejection, and explicit `Identity` compatibility validation.
- [ ] The minimal partitioned fixture verifies `history_index` only for History,
  `stage_index` only for Stage, logical 1-based history mapping to 0-based
  FE/Auxiliary storage, explicit predicted backing kinds, no separate endpoint
  slots for `u_prev`/`u_prev2`, and rejects stale or wrong-system interface-map
  provenance.
- [ ] The temporal-policy fixture verifies declared time derivative, time,
  time-step/effective-time-step, previous-solution history, and mesh temporal
  requirements succeed or fail according to the selected policy.
- [ ] The geometry-terminal fixture verifies mesh displacement, coordinate,
  Jacobian, Jacobian-inverse, normal, measure, surface-Jacobian, cell-metric, and
  cell-domain-id requirements succeed or fail according to geometry transaction
  and Assembly metadata support.
- [ ] The geometry-terminal fixture verifies boundary/interface terminal
  validation uses declaration-side region kind, resolved `analysis::DomainKind`,
  marker/shared-region, owner-scope, side, typed `forms::GeometryConfiguration`
  value and frame-transform configurations, typed logical interface region,
  geometry-revision, and quadrature-policy provenance.
- [ ] The Forms dependency fixture verifies undeclared `StateField`
  dependencies, undeclared non-field data/provenance requirements, undeclared
  graph-variable dependencies, undeclared geometry-sensitivity dependencies, and
  declared-but-unused implicit dependencies are diagnosed.
- [ ] The Forms dependency fixture verifies diagnostics are based on public
  Forms/Systems coupling-analysis metadata.
- [ ] The Forms dependency fixture verifies geometry-sensitivity
  `FormInstallOptions` are resolved from declaration-time options, require a
  bound mesh-motion field, and produce structured geometry-sensitivity provenance.
- [ ] The Forms dependency fixture verifies cut/embedded geometry sensitivity
  provenance remains structured and includes target kind, parent entity, parent
  geometry DOFs, visible assembly paths, revision/capability metadata, AD
  compatibility, and sample count.
- [ ] The Forms dependency fixture verifies contribution name, origin, and
  owning-system provenance are preserved when multiple contracts install into the
  same operator tag.
- [ ] The Forms dependency fixture verifies boundary-integral provenance is
  distinct from boundary-functional provenance and maps to BoundaryFunctional
  graph identity until Analysis exposes a dedicated kind.
- [ ] The Forms dependency fixture verifies location-sensitive non-field
  provenance for boundary functionals, boundary integrals, and material-state
  refs, including `analysis::DomainKind`, region/shared-region, marker, side,
  logical region, provider, slot/output-id, value type, and byte-offset evidence,
  and verifies same-name dependencies on different regions remain distinct.
- [ ] The Forms dependency fixture verifies declaration-time AD mode comes only
  from `CouplingFormInstallOptionsDeclaration::ad_mode`, not from
  `compiler_options.ad_mode`.
- [ ] The Forms dependency fixture verifies declaration-time geometry tangent
  path and symbolic-tangent policy come only from
  `CouplingGeometrySensitivityDeclaration`, not from raw
  `forms::SymbolicOptions`.
- [ ] The Forms dependency fixture verifies geometry tangent mode combinations:
  `GeometryConstant` rejects mesh-motion fields, `SymbolicRequired`, and
  `SymbolicWithADCheck`; `GeometryConstant` accepts ordinary
  `use_symbolic_tangent`; `MeshMotionUnknowns` requires a bound mesh-motion field
  and forwards or forces the resolved geometry tangent path as declared.
- [ ] The monolithic topology fixture verifies Forms-authored coupling is
  accepted only for one compatible `FESystem` with registered interface topology
  when `.dI(marker)` is used.
- [ ] Expert-hook fixtures verify missing `CouplingInstallMetadata` records for
  installed custom contributions prevent expected-block validation from passing.
- [ ] Expert-hook fixtures verify declaration-shaped metadata is rejected and
  expert installs must report contribution identity, scoped
  `analysis::VariableKey` dependencies, installed block/`analysis::DomainKind`
  provenance, and matrix/vector contribution flags.
- [ ] Expert-hook fixtures verify custom installs use approved Systems
  extension points.
- [x] Partitioned fixtures verify declared exchanges and generated plans remain
  consistent.
- [x] Partitioned fixtures verify endpoints resolve through existing FE
  registries or explicitly declared scoped external buffers and produce
  `ResolvedCouplingEndpoint` records with explicit resolved endpoint kind,
  resolved participant/global endpoint scope, resolved temporal backing, and
  diagnostic-only declaration provenance.
- [x] Partitioned fixtures verify endpoint names are physics-opaque but
  registry-resolved for each endpoint kind.
- [x] Partitioned fixtures verify endpoint participant scope rules for
  FE-backed, parameter, global external-buffer, and participant-scoped
  external-buffer endpoints.
- [x] Partitioned fixtures verify declaration-time transfer metadata stays
  separate from resolved `systems::InterfaceTransferOptions`.
- [x] Partitioned fixtures verify true 2D vector source embedding and target
  restriction policies stay separate from resolved
  `systems::InterfaceTransferOptions`, are preserved in
  `ResolvedCouplingTransfer`, and are either applied by an available adapter or
  rejected before execution.
- [x] Partitioned fixtures verify interface runtime handles include the actual
  search registry and interface map object used for `applyInterfaceTransfer()`
  and are not part of durable plan equality.
- [x] Partitioned fixtures verify external-buffer descriptors validate scalar
  type, access, lifetime, distribution, extents, strides, packing, supported
  indexed temporal-slot descriptors, payload shape, global versus
  participant-scoped lookup, and layout/data revision keys.
- [x] Partitioned fixtures verify driver-owned general tensor payloads require
  explicit extents and packing, use `CouplingTransferKind::DriverOwned`, and are
  rejected for FE interface transfers.
- [ ] Negative tests cover every validation rule listed in this plan document.
- [ ] Positive tests cover every public data structure and every public method
  in `FE/Coupling`.
- [ ] Positive tests cover every public option, validation branch, and lowering
  branch in the initial `Physics/Coupling` contracts.
- [x] The completed implementation does not require a partitioned driver to
  validate or inspect partitioned coupling plans.
- [x] The completed implementation allows one physical contract to support both
  monolithic and partitioned lowering without duplicating its participant,
  field, shared-region, temporal, geometry-terminal, or dependency declarations.
- [ ] The completed implementation has no unchecked items in Phases 0 through
  8.
- [x] The first executable slice has its own completion marker: FE/Coupling
  skeleton, context, shared-region registry, contract declaration, metadata bridge
  spike, first public bridge increment, declaration-side graph validation, minimal
  generic monolithic Forms fixture, and bridge-backed dependency/block tests must
  all pass before implementation proceeds to partitioned execution or
  physics-specific FSI work.

## Recommended First Implementation Scope

The first pass should avoid building the full partitioned solver driver and
should not try to implement every endpoint, transfer, and geometry-provenance
path at once. The first executable slice should prove that FE/Coupling can
declare participants, resolve fields/regions, author a Forms residual through
the existing Forms vocabulary, install it through Systems, and validate the
installed dependency evidence through the public metadata bridge.

First executable slice:

```text
1. Add FE coupling vocabulary and context, including participant-to-`FESystem`
   ownership metadata for every FE-backed field, region, and endpoint.
2. Add shared-region registry.
3. Add coupling contract declaration interface with reusable contract type keys,
   configured contract instance namespaces, contract-owned port namespaces,
   participant-scoped and contract-owned additional fields, and separate
   non-field data/provenance requirements.
4. Run the public Forms/Systems metadata bridge spike before implementing
   installed-form dependency diagnostics. The spike audits current metadata
   sources, chooses the public bridge location, defines the normalized terminal
   record, and adds synthetic bridge fixtures plus feature gates for incomplete
   metadata categories.
5. Add the first public metadata bridge increment for the minimal generic
   monolithic fixture. It must cover contribution identity, owning system,
   installed field order, StateField/TestField/DiscreteField uses, active
   domains, block evidence, `PreviousSolutionRef(k)`, scanner-normalized
   boundary-functional versus boundary-integral evidence when used by the
   fixture, and only the non-field, geometry-terminal, temporal, and
   geometry-sensitivity categories exercised by that fixture. Broader
   parameter/coefficient/material-state/auxiliary/geometry-sensitivity coverage
   remains in the full phase checklists and must stay feature-gated until the
   bridge exposes it.
6. Add declaration-side coupling graph validation that does not require
   installed Forms evidence.
7. Add installed-form dependency and expected-block diagnostics only after they
   consume the public metadata bridge or synthetic bridge fixtures.
8. Implement a minimal generic two-participant monolithic coupling contract that
   uses no FSI-specific physical vocabulary in FE tests.
9. Implement Forms-authored monolithic interface terms for that minimal
   coupling using explicit Forms measures such as `.dI(marker)`.
10. Add monolithic topology validation for shared-FESystem/interface topology
   requirements.
11. Add temporal requirement declarations for first derivative and
    `PreviousSolutionRef(k)` history, plus validation against `SystemStateView`
    history depth.
12. Add geometry-terminal declaration/provenance validation only for terminals
    whose public metadata is available in the bridge.
13. Add tests that inspect fields, validation, metadata bridge output, graph
    structure, and installed block structure for the minimal coupling fixture.
```

First-slice risk burn-down order:

```text
1. Land the FE/Coupling skeleton and build-only dependency-boundary tests.
2. Land context, participants, fields, regions, shared regions, and declaration
   graph validation without installed Forms evidence.
3. Land the metadata bridge spike, first bridge increment, and scanner
   normalization tests before enabling installed-form validators.
4. Land the generic monolithic Forms fixture through `installFormulation()`.
5. Enable bridge-backed expected-block and dependency diagnostics only for the
   terminals covered by the bridge increment.
6. Keep partitioned execution, true 2D frame adapters, driver-owned transfer
   execution, non-Real parameter transfers, and cut/embedded geometry-sensitivity
   adaptation in explicit reject or follow-on status.
```

Follow-on scope after the first executable slice:

```text
1. Add FSI coupling contract skeleton under `Physics/Coupling`.
2. Add a second simple contract skeleton to exercise N-contract graph behavior.
3. Add expert install metadata validation and approved Systems extension paths
   for custom monolithic contributions.
4. Add partitioned plan generation as metadata and compare it against declared
   partitioned topology.
5. Add partitioned endpoint resolution through existing FE registries, including
   registry-resolved endpoint names, optional endpoint participant scope,
   endpoint temporal slot descriptors, explicit transfer validation, resolved
   endpoint records with stable endpoint keys, participant/global endpoint
   scope, temporal backing, and diagnostic declaration provenance.
6. Add external-buffer descriptors, driver-owned transfer descriptors, resolved
   transfer options, distinct interface-entry/interface-map/interface-search
   registry ownership and revision/state provenance, logical region ids,
   coordinate configurations, revision snapshots, and execution-only
   search-registry, optional sliding-map state, and concrete interface-map
   runtime handles.
7. Add mesh temporal, effective-time-step, cut/embedded geometry-sensitivity,
   and full geometry-terminal provenance once the public bridge exposes the
   required Systems/Assembly metadata.
8. Add physics-specific FSI and thermal-interface tests after the FE-owned
   generic coupling tests are passing.
```

This creates the architectural foundation without prematurely locking in the
partitioned driver implementation.
