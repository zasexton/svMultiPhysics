/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_FESYSTEM_H
#define SVMP_FE_SYSTEMS_FESYSTEM_H

#include "Core/Types.h"
#include "Core/FEException.h"

#include "Forms/BoundaryFunctional.h"
#include "Forms/FormExpr.h"
#include "Auxiliary/AuxiliaryBindings.h"
#include "Auxiliary/AuxiliaryInputRegistry.h"
#include "Auxiliary/AuxiliaryStateTypes.h"
#include "Systems/FEQuantityRegistry.h"
#include "Systems/BoundaryReductionService.h"
#include "Systems/FieldRegistry.h"
#include "Systems/GeometricNonlinearity.h"
#include "Systems/GeometryTransaction.h"
#include "Systems/GlobalKernel.h"
#include "Systems/GlobalKernelStateProvider.h"
#include "Systems/OperatorRegistry.h"
#include "Systems/ParameterRegistry.h"
#include "Systems/SearchAccess.h"
#include "Systems/SetupStoragePlan.h"
#include "Constraints/SystemConstraint.h"
#include "Systems/SystemState.h"
#include "Systems/SystemSetup.h"
#include "PostProcessing/DerivedResultOutput.h"
#include "PostProcessing/DerivedResultRegistry.h"
#include "Interfaces/FreeSurfaceGeometrySnapshot.h"

#include "Assembly/Assembler.h"

#include "Backends/Interfaces/LinearSolver.h"

#include "Constraints/AffineConstraints.h"
#include "Constraints/Constraint.h"
#include "Constraints/GaugeRegistry.h"
#include "Constraints/SmallCutAggregationConstraint.h"

#include "Analysis/ProblemAnalysisTypes.h"
#include "Analysis/ProblemAnalysisContext.h"
#include "Analysis/AnalysisSummaryTypes.h"
#include "Analysis/FormulationRecord.h"
#include "Analysis/BoundaryConditionDescriptor.h"
#include "Analysis/TopologyAnalysisContext.h"
#include "Analysis/ContributionDescriptor.h"
#include "Analysis/ConstraintAnalysisSummary.h"
#include "Analysis/GeneratedBoundaryAggregateTraceCertificate.h"
#include "Analysis/InterfaceTopologyContext.h"

#include "Dofs/DofHandler.h"
#include "Dofs/FieldDofMap.h"
#include "Dofs/BlockDofMap.h"

#include "Sparsity/SparsityPattern.h"
#include "Sparsity/SparsityBuilder.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <array>
#include <functional>
#include <string>
#include <string_view>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Systems/FEAdaptivityTransfer.h"
#include "Mesh/Core/MeshTypes.h"
#include "Mesh/Motion/MotionState.h"
namespace svmp {
struct AdaptivityOptions;
class InterfaceMesh;
struct RefinementDelta;
}
#endif

namespace svmp {
class MeshBase;

namespace FE {

namespace sparsity {
class DistributedSparsityPattern;
} // namespace sparsity

namespace assembly {
class GlobalSystemView;
struct MatrixFreeOptions;
class IMatrixFreeKernel;
class MatrixFreeOperator;
class FunctionalKernel;
class CutIntegrationContext;
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
class CompositeMeshAccess;
#endif
}

namespace backends {
struct DofPermutation;
class GenericMatrix;
} // namespace backends

namespace forms {
class FormExpr;
class MixedFormIR;
}

namespace systems {

using BoundaryId = int;
using InterfaceId = int;
class OperatorBackends;
class BoundaryReductionService;
class AuxiliaryStateManager;
class AuxiliaryOperatorRegistry;
class AuxiliaryInputRegistry;
class AuxiliaryDeployedInstance;
class AuxiliaryInputHandle;
class AuxiliaryInstanceHandle;
class AuxiliaryStateModel;
class AuxiliaryMultirateScheduler;
class AuxiliaryStateStepper;
class AuxiliaryDerivativeProvider;
class AuxiliaryEventManager;
class AuxiliaryBlockStorage;
struct MixedSystemLayout;
struct FormInstallOptions;
struct CoupledResidualKernels;

struct SetupOptions {
    dofs::DofDistributionOptions dof_options{};
    assembly::AssemblyOptions assembly_options{};
    sparsity::SparsityBuildOptions sparsity_options{};

    std::string assembler_name{"StandardAssembler"};

    sparsity::CouplingMode coupling_mode{sparsity::CouplingMode::Full};
    std::vector<sparsity::FieldCoupling> custom_couplings{};

    bool use_constraints_in_assembly{true};
    bool use_backend_row_ownership_for_assembly{false};
    // Keep the replicated serial sparsity graph even when a distributed graph exists.
    // Large MPI backend runs can disable this to avoid setup-time memory pressure.
    bool retain_serial_sparsity{true};

    // Iterative-solver leverage (explicit opt-in): auto-register eligible matrix-free operators.
    bool auto_register_matrix_free{false};

    /// When true, print a detailed GaugeRegistry diagnostic report to stderr
    /// after nullspace resolution during setup().  Useful for debugging
    /// nullspace detection and enforcement decisions.
    bool gauge_diagnostics{false};
};

struct AssemblyRequest {
    OperatorTag op;
    bool want_matrix{false};
    bool want_vector{false};
    bool zero_outputs{true};
    bool assemble_boundary_terms{true};
    bool assemble_interior_face_terms{true};
    bool assemble_interface_face_terms{true};
    bool assemble_global_terms{true};

    /// When true, constrained assembly distributes matrix and vector independently
    /// (suppressing the -K*g Dirichlet inhomogeneity correction that joint distribution
    /// adds).  Set to true for nonlinear Newton solves where the residual R(u) is already
    /// evaluated at the constrained state.
    bool suppress_constraint_inhomogeneity{false};

    /// When true, EachNonlinearIteration auxiliary inputs are refreshed.
    /// Set to true on each Newton iteration within a time step.
    bool is_nonlinear_iteration{false};

    /// Assemble an isolated FE diagnostic operator without advancing,
    /// preparing, or injecting generalized auxiliary coupling. The ordinary
    /// FE operator is still assembled, but production auxiliary recovery
    /// caches (rank-one/reduced updates and bordered/local-condensed state)
    /// are preserved exactly as left by the preceding production assembly.
    bool suppress_auxiliary_coupling_assembly{false};
};

enum class MeshMotionFieldRole : std::uint8_t {
    Displacement,
    Velocity,
    Acceleration,
    PreviousCoordinates,
    PreviousDisplacement,
    PreviousVelocity,
    PredictedVelocity
};

enum class MeshNormalBoundaryQuantity : std::uint8_t {
    DisplacementTrace,
    MeshVelocityTrace
};

enum class MeshNormalBoundaryTargetKind : std::uint8_t {
    PrescribedDisplacement,
    TimeScaledPrescribedVelocity,
    FluidNormalVelocity
};

struct MeshNormalBoundaryRelatedFluidConsumerBinding {
    analysis::EnforcementKind enforcement_kind{
        analysis::EnforcementKind::WeakPenalty};
    std::string descriptor_source{};

    [[nodiscard]] friend bool operator==(
        const MeshNormalBoundaryRelatedFluidConsumerBinding&,
        const MeshNormalBoundaryRelatedFluidConsumerBinding&) = default;
};

struct MeshNormalBoundaryConstraintConsumerBinding {
    std::string operator_tag{};
    std::string mesh_descriptor_source{};
    std::optional<MeshNormalBoundaryRelatedFluidConsumerBinding>
        related_fluid{};

    [[nodiscard]] friend bool operator==(
        const MeshNormalBoundaryConstraintConsumerBinding&,
        const MeshNormalBoundaryConstraintConsumerBinding&) = default;
};

struct MeshNormalBoundaryConstraintDeclaration {
    FieldId mesh_displacement_field{INVALID_FIELD_ID};
    int boundary_marker{-1};
    MeshNormalBoundaryQuantity quantity{
        MeshNormalBoundaryQuantity::DisplacementTrace};
    MeshNormalBoundaryTargetKind target_kind{
        MeshNormalBoundaryTargetKind::PrescribedDisplacement};
    forms::FormExpr target_expression{};
    analysis::EnforcementKind enforcement_kind{
        analysis::EnforcementKind::WeakPenalty};
    FieldId related_velocity_field{INVALID_FIELD_ID};
    std::string owner_component{};
    std::optional<MeshNormalBoundaryConstraintConsumerBinding>
        consumer_binding{};
};

struct MeshNormalBoundaryConstraintHistoryRecord {
    std::uint64_t accepted_step{0};
    Real accepted_time{0.0};
    Real dt{0.0};
    std::uint64_t state_revision{0};
    /** Rank-local mesh geometry stamp retained as local provenance. */
    std::uint64_t mesh_geometry_revision{0};
    /** Exact in-memory symbolic declaration; not an evaluated target value. */
    MeshNormalBoundaryConstraintDeclaration declaration{};
};

/** Canonical identity of one fitted-ALE normal-kinematics measurement. */
struct FittedALENormalMeasurementKey {
    FieldId mesh_displacement_field{INVALID_FIELD_ID};
    int boundary_marker{-1};

    [[nodiscard]] friend bool operator==(
        const FittedALENormalMeasurementKey&,
        const FittedALENormalMeasurementKey&) = default;
};

/**
 * @brief Generalized-alpha coordinates for a neutral operator-stage record.
 *
 * These coordinates describe where the converged operator sampled state and
 * rate.  They do not assert that either sample is an accepted-step endpoint.
 */
struct OperatorStageGeneralizedAlphaMetadata {
    Real alpha_f{1.0};
    Real alpha_m{1.0};

    [[nodiscard]] friend bool operator==(
        const OperatorStageGeneralizedAlphaMetadata&,
        const OperatorStageGeneralizedAlphaMetadata&) = default;
};

/** Rank-local mesh stamp captured with an operator-stage state snapshot. */
struct OperatorStageGeometryMetadata {
    std::uint64_t geometry_revision{0};
    std::uint64_t topology_revision{0};
    std::uint64_t ownership_revision{0};
    std::uint64_t numbering_revision{0};
    std::uint64_t field_layout_revision{0};
    std::uint64_t label_revision{0};
    std::uint64_t active_configuration_epoch{0};
    std::uint64_t coordinate_configuration_key{0};

    [[nodiscard]] friend bool operator==(
        const OperatorStageGeometryMetadata&,
        const OperatorStageGeometryMetadata&) = default;
};

/**
 * @brief Time-integrator-neutral provenance for a prospective accepted stage.
 *
 * `state_time` and `rate_time` are the exact converged operator sampling
 * times.  For generalized-alpha they are generally not the step endpoint.
 * `derivative_fields` states which entries of the supplied exact-rate vector
 * are meaningful time derivatives.
 */
struct OperatorStageMeasurementMetadata {
    /** V1 identity: BackwardEuler, DG0, or GeneralizedAlphaFirstOrder. */
    std::string scheme_name{};
    int temporal_order{0};
    std::uint64_t prospective_accepted_step{0};
    std::uint64_t prospective_attempt{0};
    Real step_start_time{0.0};
    Real step_end_time{0.0};
    Real state_time{0.0};
    Real rate_time{0.0};
    Real dt{0.0};
    std::optional<OperatorStageGeneralizedAlphaMetadata>
        generalized_alpha{};
    /** Expected rank-local stamp; deliberately excluded from MPI consensus. */
    std::optional<OperatorStageGeometryMetadata> expected_stage_geometry{};
    /**
     * Communicator-certified exact algebraic fingerprint/revision of the
     * copied operator-stage state.  The core compares the supplied token
     * across ranks but cannot independently prove the caller's certification.
     */
    std::uint64_t state_revision{0};
    /**
     * Communicator-certified exact algebraic fingerprint/revision of the
     * copied rate alias, subject to the same caller-certification boundary.
     */
    std::uint64_t rate_revision{0};
    std::vector<FieldId> derivative_fields{};

    [[nodiscard]] friend bool operator==(
        const OperatorStageMeasurementMetadata&,
        const OperatorStageMeasurementMetadata&) = default;
};

/**
 * @brief Registered fitted-ALE normal measurement and its reduction names.
 *
 * The copied constraint is the exact, consumer-bound symbolic declaration
 * measured by this record.  The mesh-displacement field remains the primary
 * BoundaryReductionService field because only the primary field owns history.
 */
struct FittedALENormalMeasurementDeclaration {
    FittedALENormalMeasurementKey key{};
    FieldId related_velocity_field{INVALID_FIELD_ID};
    MeshNormalBoundaryConstraintDeclaration normal_constraint{};
    std::string mesh_normal_integral_functional{};
    std::string fluid_normal_integral_functional{};
    std::string normal_gap_squared_integral_functional{};
};

/**
 * @brief Raw fitted-ALE normal moments sampled at one operator stage.
 *
 * A is boundary measure, Wn is the integral of mesh normal velocity, Un is
 * the integral of fluid normal velocity, and gap_sq is the integral of their
 * squared difference.  These are raw operator-stage moments: they are not
 * endpoint values and must not be reported as work or dissipation.
 */
struct FittedALENormalOperatorStageRawValue {
    FittedALENormalMeasurementKey key{};
    Real A{0.0};
    Real Wn{0.0};
    Real Un{0.0};
    Real gap_sq{0.0};
    /** Complete rank-local live stamp; deliberately excluded from consensus. */
    OperatorStageGeometryMetadata stage_mesh_revision{};
};

/**
 * @brief Pending or accepted fitted-ALE operator-stage measurement record.
 *
 * Generalized-alpha records retain their operator-stage coordinates and are
 * not silently relabelled as endpoint observations.  The raw moments carry no
 * work, power, penalty-energy, or dissipation interpretation.
 */
struct FittedALENormalOperatorStageHistoryRecord {
    FittedALENormalMeasurementDeclaration declaration{};
    OperatorStageMeasurementMetadata stage{};
    FittedALENormalOperatorStageRawValue raw{};
};

enum class MeshTangentialBoundaryPolicy : std::uint8_t {
    Free,
    SmoothingOnly,
    Prescribed
};

struct MeshTangentialBoundaryPolicyDeclaration {
    FieldId mesh_displacement_field{INVALID_FIELD_ID};
    int boundary_marker{-1};
    MeshTangentialBoundaryPolicy policy{
        MeshTangentialBoundaryPolicy::SmoothingOnly};
    std::string owner_component{};
    bool consumer_bound{false};
    std::string consumer_operator_tag{};
    std::string consumer_source{};
};

struct MeshTangentialBoundaryPolicyHistoryRecord {
    std::uint64_t accepted_step{0};
    Real accepted_time{0.0};
    Real dt{0.0};
    std::uint64_t state_revision{0};
    /** Rank-local mesh geometry stamp retained as local provenance. */
    std::uint64_t mesh_geometry_revision{0};
    FieldId mesh_displacement_field{INVALID_FIELD_ID};
    int boundary_marker{-1};
    MeshTangentialBoundaryPolicy policy{
        MeshTangentialBoundaryPolicy::SmoothingOnly};
    std::string owner_component{};
    bool consumer_bound{false};
    std::string consumer_operator_tag{};
    std::string consumer_source{};
};

/** Static one-phase capillary-balance construction selected by the owner. */
enum class FreeSurfaceCapillaryBalanceMethod : std::uint8_t {
    Unselected,
    /** Snapshot energy variation plus a fixed-volume stationary geometry. */
    DiscreteEnergyVolumeStationarity,
    /** Total surface-plus-Young-wall energy gradient represented as traction. */
    KinematicAreaGradientEnergyTraction
};

/** Evidence boundary for the selected capillary-balance construction. */
enum class FreeSurfaceCapillaryBalanceQualification : std::uint8_t {
    Unselected,
    /** Low-level identities only; static-cap qualification is still open. */
    PrerequisiteOnly,
    Qualified
};

struct FreeSurfaceDiscreteFunctionalDeclaration {
    int interface_marker{-1};
    FieldId level_set_field{INVALID_FIELD_ID};
    /** Prescribed curvature carrying the selected total energy gradient. */
    FieldId curvature_field{INVALID_FIELD_ID};
    FieldId velocity_field{INVALID_FIELD_ID};
    std::string geometry_domain_id{};
    interfaces::FreeSurfaceDiscreteFunctionalParameters parameters{};
    std::optional<interfaces::FreeSurfaceActiveVolumeEnergyParameters>
        active_volume_energy_parameters{};
    /**
     * The active-volume gravitational acceleration is the complete
     * velocity-independent body acceleration present in the production
     * static momentum residual.  Static equilibrium certification fails
     * closed unless the owning physics module makes this declaration.
     */
    bool static_conservative_body_force_complete{false};
    std::optional<
        interfaces::FreeSurfaceActiveVolumeDissipationParameters>
        active_volume_dissipation_parameters{};
    std::optional<
        interfaces::FreeSurfaceExternalPressurePowerParameters>
        external_pressure_power_parameters{};
    bool endpoint_functional_power_enabled{false};
    FreeSurfaceCapillaryBalanceMethod capillary_balance_method{
        FreeSurfaceCapillaryBalanceMethod::Unselected};
    FreeSurfaceCapillaryBalanceQualification
        capillary_balance_qualification{
            FreeSurfaceCapillaryBalanceQualification::Unselected};
    std::string owner_component{};
};

/**
 * Authentic first-order generalized-alpha parameters retained with an
 * accepted contact stage.  This is optional only for endpoint stages.  The
 * supported TimeLoop rho-infinity family has alpha_f == gamma in [0.5, 1]
 * and alpha_m == 2*alpha_f - 0.5 in [0.5, 1.5].
 */
struct FreeSurfaceFirstOrderGeneralizedAlphaProvenance {
    Real alpha_m{1.0};
    Real alpha_f{1.0};
    Real gamma{1.0};
    Real dt{0.0};

    [[nodiscard]] friend bool operator==(
        const FreeSurfaceFirstOrderGeneralizedAlphaProvenance&,
        const FreeSurfaceFirstOrderGeneralizedAlphaProvenance&) = default;
};

struct FreeSurfaceAcceptedContactStageState {
    Real stage_time{0.0};
    Real stage_alpha_f{1.0};
    std::optional<FreeSurfaceFirstOrderGeneralizedAlphaProvenance>
        first_order_generalized_alpha{};
    // Communicator-consistent fingerprints of FE-ordered algebraic content.
    std::uint64_t previous_state_revision{0};
    std::uint64_t endpoint_state_revision{0};
    // Composite fingerprint over the previous/endpoint content, accepted
    // snapshot, stage time/parameters, and stage solution.
    std::uint64_t stage_state_revision{0};
    interfaces::FreeSurfaceGeometryRevision geometry_revision{};
    interfaces::FreeSurfaceDynamicContactState state{};
};

/**
 * Accepted-stage contact-centroid kinematics derived from two consecutive
 * history records.
 *
 * This is a secant velocity of the measure-weighted contact-position
 * centroid projected onto a common oriented footprint direction.  It is not
 * a pointwise contact-line velocity.  The previous accepted record and stage
 * revisions make the comparison with the instantaneous fluid-trace speed
 * reproducible.
 */
struct FreeSurfaceAcceptedContactLineKinematics {
    int boundary_marker{-1};
    std::uint64_t previous_accepted_step{0};
    Real previous_accepted_time{0.0};
    Real previous_stage_time{0.0};
    std::uint64_t previous_stage_state_revision{0};
    std::uint64_t previous_snapshot_revision_key{0};
    Real stage_time_interval{0.0};
    std::array<Real, 3> previous_mean_contact_position{
        {0.0, 0.0, 0.0}};
    std::array<Real, 3> projection_direction{{0.0, 0.0, 0.0}};
    Real projected_contact_centroid_speed{0.0};
    Real mean_fluid_contact_speed{0.0};
    Real fluid_minus_geometric_contact_speed{0.0};
};

struct AcceptedFreeSurfaceDiscreteFunctionalState {
    int interface_marker{-1};
    interfaces::FreeSurfaceGeometryRevision geometry_revision{};
    // Communicator-consistent topology-only fingerprint for the exact
    // authoritative snapshot rules used by this state.
    std::uint64_t cut_topology_revision{0};
    interfaces::FreeSurfaceDiscreteFunctionalState state{};
    std::optional<
        interfaces::FreeSurfaceDiscreteFunctionalVariationState>
        endpoint_functional_power{};
    std::optional<interfaces::FreeSurfaceActiveVolumeEnergyState>
        active_volume_energy{};
    std::optional<
        interfaces::FreeSurfaceActiveVolumeDissipationState>
        active_volume_dissipation{};
    std::optional<
        interfaces::FreeSurfaceExternalPressurePowerState>
        external_pressure_power{};
    std::optional<
        interfaces::FreeSurfaceBackwardEulerKineticWorkState>
        backward_euler_kinetic_work{};
    std::optional<FreeSurfaceAcceptedContactStageState> contact_stage{};
    std::vector<FreeSurfaceAcceptedContactLineKinematics>
        contact_line_kinematics{};
};

struct FreeSurfaceDiscreteFunctionalHistoryRecord {
    std::uint64_t accepted_step{0};
    Real accepted_time{0.0};
    Real dt{0.0};
    // Communicator-consistent fingerprints before and after maintenance.
    std::uint64_t pre_maintenance_endpoint_state_revision{0};
    std::uint64_t state_revision{0};
    std::optional<std::uint64_t> extension_map_revision{};
    FreeSurfaceDiscreteFunctionalDeclaration declaration{};
    interfaces::FreeSurfaceGeometryRevision geometry_revision{};
    std::uint64_t cut_topology_revision{0};
    interfaces::FreeSurfaceDiscreteFunctionalState state{};
    std::optional<
        interfaces::FreeSurfaceDiscreteFunctionalVariationState>
        endpoint_functional_power{};
    std::optional<interfaces::FreeSurfaceActiveVolumeEnergyState>
        active_volume_energy{};
    std::optional<
        interfaces::FreeSurfaceActiveVolumeDissipationState>
        active_volume_dissipation{};
    std::optional<
        interfaces::FreeSurfaceExternalPressurePowerState>
        external_pressure_power{};
    std::optional<
        interfaces::FreeSurfaceBackwardEulerKineticWorkState>
        backward_euler_kinetic_work{};
    std::optional<FreeSurfaceAcceptedContactStageState> contact_stage{};
    std::vector<FreeSurfaceAcceptedContactLineKinematics>
        contact_line_kinematics{};
};

[[nodiscard]] std::vector<FreeSurfaceAcceptedContactLineKinematics>
deriveFreeSurfaceAcceptedContactLineKinematics(
    const FreeSurfaceDiscreteFunctionalHistoryRecord& previous,
    std::uint64_t accepted_step,
    const FreeSurfaceAcceptedContactStageState& current);

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
enum class MeshCoordinateUpdateMode : std::uint8_t {
    AbsoluteFromReference,
    IncrementalFromCurrent
};

enum class MeshCoordinateUpdateStage : std::uint8_t {
    TrialNonlinearIterate,
    AcceptedNonlinearState,
    AcceptedTimeStep,
    AcceptedRemeshRezoneState
};

struct MeshCoordinateUpdateOptions {
    MeshCoordinateUpdateMode mode{MeshCoordinateUpdateMode::AbsoluteFromReference};
    MeshCoordinateUpdateStage stage{MeshCoordinateUpdateStage::TrialNonlinearIterate};
    bool exchange_ghost_coordinates{true};
    bool notify_geometry_advanced{true};
};

struct MeshCoordinateUpdateResult {
    std::size_t vertices_updated{0};
    std::size_t components_updated{0};
    MeshCoordinateUpdateStage stage{MeshCoordinateUpdateStage::TrialNonlinearIterate};
    std::uint64_t geometry_revision{0};
};

struct FEAdaptedStateTransferRequest {
    std::span<const Real> solution{};
    std::span<const Real> previous_solution{};
    std::span<const Real> previous_solution2{};
    std::vector<Real>* transferred_solution{nullptr};
    std::vector<Real>* transferred_previous_solution{nullptr};
    std::vector<Real>* transferred_previous_solution2{nullptr};
    FEFieldTransferOptions field_transfer_options{};
    SetupOptions setup_options{};
    bool rebuild_setup{true};
    bool transfer_auxiliary_state{true};
    bool transfer_material_state{true};
    bool transfer_boundary_and_coupling_state{true};
};

struct FEAdaptedStateTransferResult {
    bool dof_handler_rebuilt{false};
    bool constraint_layout_rebuilt{false};
    bool sparsity_rebuilt{false};
    bool solution_transferred{false};
    bool previous_solution_transferred{false};
    bool previous_solution2_transferred{false};
    bool auxiliary_state_transfer_handled{false};
    bool material_state_transfer_handled{false};
    bool boundary_coupling_state_transfer_handled{false};
    std::size_t values_transferred{0};
    FELayoutRevisionState layout_before{};
    FELayoutRevisionState layout_after{};
    std::vector<std::string> diagnostics{};
};
#endif

struct GeometryTransactionCallback {
    std::string name;
    std::function<void(const GeometryTransactionDiagnostics&)> callback;
};

struct CutIntegrationContextUpdateCallback {
    std::string name;

    /**
     * The argument is the authoritative candidate and may be null.  Setters
     * invoke callbacks before publication, so cutIntegrationContext() still
     * exposes the previous context while the callback runs.  A throw rejects
     * publication; callbacks that completed earlier may run again on retry.
     *
     * Rollback invokes the same contract after restoring the saved state.
     * Callbacks must therefore use this argument rather than infer candidate
     * identity from the system getter.  The candidate pointee must remain
     * immutable throughout dispatch, including through any retained mutable
     * alias.  A callback that enters collectives must coordinate every
     * rank-local preflight failure before doing so.  Callback dispatch is
     * non-reentrant: callbacks may not register another context callback,
     * publish or clear a context, or mutate a context transaction.
     * A distributed abstract mesh without an available multi-rank system
     * communicator must defer context publication until its DOF layout is
     * finalized.
     */
    std::function<void(const assembly::CutIntegrationContext*)> callback;
};

struct MeshParticipantInfo {
    std::string name{};
    std::optional<int> domain_id{};
    GlobalIndex cell_offset{0};
    GlobalIndex num_cells{0};
    GlobalIndex vertex_offset{0};
    GlobalIndex num_vertices{0};
    GlobalIndex boundary_face_offset{0};
    GlobalIndex num_boundary_faces{0};
    GlobalIndex interior_face_offset{0};
    GlobalIndex num_interior_faces{0};
};

using ExteriorBoundaryMeasurePolicyId = std::uint64_t;
inline constexpr ExteriorBoundaryMeasurePolicyId
    INVALID_EXTERIOR_BOUNDARY_MEASURE_POLICY_ID{0u};
inline constexpr std::size_t
    NO_EXTERIOR_BOUNDARY_MEASURE_FORMULATION_RECORD{
        std::numeric_limits<std::size_t>::max()};

/**
 * Definition-time intent for one exterior-boundary form measure.
 *
 * LegacyBoundary and LegacyInterface retain enough information to reject a
 * raw ds/dI route if a later cut context gives the same marker generated-active
 * provenance. FullPhysical and GeneratedActiveSubset are explicit, disjoint
 * selector intents.
 */
enum class ExteriorBoundaryMeasureIntent : std::uint8_t {
    LegacyBoundary,
    LegacyInterface,
    FullPhysical,
    GeneratedActiveSubset
};

struct ExteriorBoundaryMeasurePolicy {
    ExteriorBoundaryMeasurePolicyId id{
        INVALID_EXTERIOR_BOUNDARY_MEASURE_POLICY_ID};
    OperatorTag op{};
    ExteriorBoundaryMeasureIntent intent{
        ExteriorBoundaryMeasureIntent::LegacyBoundary};
    /// Raw ds marker or the physical owner of an explicit selector.
    int physical_boundary_marker{-1};
    /// Raw dI marker or the generated marker of an explicit active subset.
    int generated_active_boundary_marker{-1};
    /// Formulation record owning this route, or the sentinel above for a
    /// lower-level precompiled IR installation.
    std::size_t source_formulation_record_index{
        NO_EXTERIOR_BOUNDARY_MEASURE_FORMULATION_RECORD};
};

using GeneratedBoundaryNitscheTracePolicyId = std::uint64_t;
inline constexpr GeneratedBoundaryNitscheTracePolicyId
    INVALID_GENERATED_BOUNDARY_NITSCHE_TRACE_POLICY_ID{0u};

/**
 * Immutable setup-time requirement for one generated-boundary velocity
 * Nitsche route.
 *
 * The policy is derived from an opaque form binding that seals the canonical
 * state/test roles and complete velocity-space signature and whose exact
 * route anchor was verified once as an unscaled top-level additive summand in
 * the residual installed for its source formulation. Certification consumes
 * the effective penalty sealed by that binding; it does not mutate or select
 * a penalty. Symmetric routes are accepted only when the grouped certified
 * trace-to-penalty ratio for their operator is no greater than a
 * downward-safe cap for the configured positive energy floor on every
 * accepted state. At this generic FE layer that floor is conditional on the
 * caller installing the matching bulk viscous energy; the bound route
 * authenticates only the boundary consistency and penalty contribution.
 * Unsymmetric routes retain the same revision-bound trace certificate as a
 * continuity diagnostic without applying the symmetric coercivity
 * inequality.
 */
struct GeneratedBoundaryNitscheTracePolicy {
    GeneratedBoundaryNitscheTracePolicyId id{
        INVALID_GENERATED_BOUNDARY_NITSCHE_TRACE_POLICY_ID};
    OperatorTag op{};
    FieldId velocity_field{INVALID_FIELD_ID};
    forms::SpaceSignature velocity_space_signature{};
    int physical_boundary_marker{-1};
    int volume_interface_marker{-1};
    int generated_active_boundary_marker{-1};
    Real dynamic_viscosity{0.0};
    Real penalty_gamma{0.0};
    bool scale_with_polynomial_order{true};
    int penalty_polynomial_order{0};
    Real effective_penalty_multiplier{0.0};
    bool symmetric{true};
    Real minimum_symmetric_energy_ratio{0.25};
    std::size_t maximum_reduced_dimension{128u};
    std::size_t source_formulation_record_index{
        std::numeric_limits<std::size_t>::max()};
    std::uint64_t form_binding_digest{0u};
};

/**
 * Current eager certificate and the exact policy coefficient it validates.
 *
 * For symmetric routes, every record on the same operator carries the same
 * grouped normalized trace sum. Each guaranteed finite-space energy ratio
 * equals that route's configured floor. The ratio is intentionally absent
 * for unsymmetric routes.
 */
struct GeneratedBoundaryNitscheTraceCertificateRecord {
    GeneratedBoundaryNitscheTracePolicy policy{};
    analysis::GeneratedBoundaryAggregateTraceCertificate certificate{};
    constraints::ConstraintRevisionSnapshot constraint_revision{};
    std::shared_ptr<
        const constraints::SmallCutAggregationProlongationReport>
        aggregation_report{};
    int polynomial_order{0};
    Real effective_penalty_multiplier{0.0};
    Real trace_to_penalty_ratio{0.0};
    Real grouped_symmetric_trace_to_penalty_ratio{0.0};
    /// Conditional lower bound for the matching bulk-plus-penalty energy
    /// model; generic FE registration does not authenticate the bulk term.
    std::optional<Real> symmetric_energy_ratio_lower_bound{};
};

class FESystem {
public:
    struct FormCellDomainRestriction {
        InterfaceId interface_marker{-1};
        geometry::CutIntegrationSide side{geometry::CutIntegrationSide::Negative};
        FieldId level_set_field{INVALID_FIELD_ID};
        Real isovalue{0.0};
        bool enable_level_set_shape_tangent{false};
        std::string diagnostic{};
    };

    class FormCellDomainRestrictionScope {
    public:
        FormCellDomainRestrictionScope() noexcept = default;
        FormCellDomainRestrictionScope(
            FESystem& system,
            std::vector<FormCellDomainRestriction> restrictions);
        ~FormCellDomainRestrictionScope();

        FormCellDomainRestrictionScope(const FormCellDomainRestrictionScope&) = delete;
        FormCellDomainRestrictionScope& operator=(const FormCellDomainRestrictionScope&) = delete;

        FormCellDomainRestrictionScope(FormCellDomainRestrictionScope&& other) noexcept;
        FormCellDomainRestrictionScope& operator=(FormCellDomainRestrictionScope&& other) noexcept;

        void restore() noexcept;
        [[nodiscard]] bool active() const noexcept { return system_ != nullptr; }

    private:
        FESystem* system_{nullptr};
        std::vector<FormCellDomainRestriction> previous_{};
    };

    explicit FESystem(std::shared_ptr<const assembly::IMeshAccess> mesh_access);
    FESystem(std::shared_ptr<const assembly::IMeshAccess> mesh_access,
             std::vector<MeshParticipantInfo> participants);
    ~FESystem();

    FESystem(FESystem&&) noexcept;
    FESystem& operator=(FESystem&&) noexcept;

    FESystem(const FESystem&) = delete;
    FESystem& operator=(const FESystem&) = delete;

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    explicit FESystem(std::shared_ptr<const svmp::Mesh> mesh,
                      svmp::Configuration coord_cfg = svmp::Configuration::Reference);
    FESystem(std::shared_ptr<const svmp::Mesh> mesh,
             MeshParticipantInfo participant,
             svmp::Configuration coord_cfg = svmp::Configuration::Reference);
    explicit FESystem(std::shared_ptr<assembly::CompositeMeshAccess> mesh_access);
    explicit FESystem(std::shared_ptr<const assembly::CompositeMeshAccess> mesh_access);
#endif

    // ---- Definition phase ----
    FieldId addField(FieldSpec spec);
    FieldId addMeshMotionDataField(std::string name,
                                   std::shared_ptr<const spaces::FunctionSpace> space,
                                   int components = 0);
    FieldId addDerivedMeshVelocityField(std::string name,
                                        std::shared_ptr<const spaces::FunctionSpace> space,
                                        FieldId mesh_displacement_field,
                                        int components = 0);
    [[nodiscard]] FieldId findFieldByName(std::string_view name) const noexcept;
    [[nodiscard]] bool hasField(std::string_view name) const noexcept;
    [[nodiscard]] bool fieldParticipatesInUnknownVector(FieldId field) const;
    [[nodiscard]] std::vector<FieldId> unknownFieldIdsInDofMapOrder() const;
    void setPrescribedFieldCoefficients(FieldId field, std::span<const Real> coefficients);
    void clearPrescribedFieldCoefficients(FieldId field);
    [[nodiscard]] std::span<const Real> prescribedFieldCoefficients(FieldId field) const;
    [[nodiscard]] std::uint64_t prescribedFieldRevision(FieldId field) const;
    void bindMeshMotionField(MeshMotionFieldRole role, FieldId field);
    void bindMeshMotionField(std::string_view role_name, FieldId field);
    void bindMeshMotionField(std::string_view role_name, std::string_view field_name);
    [[nodiscard]] std::optional<FieldId> meshMotionField(MeshMotionFieldRole role) const noexcept;
    [[nodiscard]] assembly::MeshMotionFieldAccess meshMotionFieldAccess() const noexcept;
    void declareMeshNormalBoundaryConstraint(
        MeshNormalBoundaryConstraintDeclaration declaration);
    void bindMeshNormalBoundaryConstraintConsumer(
        FieldId mesh_displacement_field,
        int boundary_marker,
        std::string operator_tag,
        std::string mesh_descriptor_source,
        std::optional<std::string> related_fluid_descriptor_source =
            std::nullopt);
    /**
     * @brief Register raw operator-stage moments for one bound fitted-ALE
     * fluid-normal constraint.
     *
     * Registration is deterministic and an exact duplicate is a no-op.  It
     * must occur after the constraint's reciprocal consumer binding is
     * complete and before a measurement stage or accepted history exists.
     */
    void registerFittedALENormalOperatorStageMeasurement(
        FieldId mesh_displacement_field,
        int boundary_marker);
    [[nodiscard]] std::span<
        const FittedALENormalMeasurementDeclaration>
    fittedALENormalOperatorStageMeasurementDeclarations() const noexcept {
        return fitted_ale_normal_measurement_declarations_;
    }
    /**
     * @brief Evaluate every registered fitted-ALE normal declaration at the
     * supplied converged operator stage.
     *
     * `state.u`/`state.u_vector` supplies the current stage and
     * `state.u_prev`/`state.u_prev_vector` supplies an exact first-rate alias.
     * Only fields named by `metadata.derivative_fields` are advertised as
     * derivatives.  Results remain pending until explicit accepted-step
     * publication.
     */
    void stageFittedALENormalOperatorStageMeasurements(
        const SystemStateView& state,
        OperatorStageMeasurementMetadata metadata);
    /** Discard a rejected candidate without throwing or publishing history. */
    void discardPendingFittedALENormalOperatorStageMeasurements() noexcept;
    /**
     * @brief Collectively publish the pending group after irreversible step
     * acceptance.
     *
     * The accepted step/time/dt must exactly match the prospective metadata
     * captured at staging.  This API is intended for the accepted-step hook,
     * not a commit-ready hook.
     */
    void commitPendingFittedALENormalOperatorStageMeasurements(
        std::uint64_t accepted_step,
        Real accepted_time,
        Real dt);
    [[nodiscard]] std::span<
        const FittedALENormalOperatorStageHistoryRecord>
    pendingFittedALENormalOperatorStageMeasurements() const noexcept {
        return pending_fitted_ale_normal_measurements_;
    }
    [[nodiscard]] std::span<
        const FittedALENormalOperatorStageHistoryRecord>
    fittedALENormalOperatorStageMeasurementHistory() const noexcept {
        return fitted_ale_normal_measurement_history_;
    }
    [[nodiscard]] std::span<
        const MeshNormalBoundaryConstraintDeclaration>
    meshNormalBoundaryConstraints() const noexcept {
        return mesh_normal_boundary_constraints_;
    }
    /**
     * @brief Publish normal history in a serial/rank-local workflow.
     *
     * A distributed system must use @ref
     * recordAcceptedMeshBoundaryProvenance from its first nonempty accepted
     * record so normal and tangential publication share one transaction.
     */
    void recordAcceptedMeshNormalBoundaryConstraints(
        std::uint64_t accepted_step,
        Real accepted_time,
        Real dt,
        std::uint64_t state_revision);
    [[nodiscard]] std::span<
        const MeshNormalBoundaryConstraintHistoryRecord>
    meshNormalBoundaryConstraintHistory() const noexcept {
        return mesh_normal_boundary_constraint_history_;
    }
    void declareMeshTangentialBoundaryPolicy(
        MeshTangentialBoundaryPolicyDeclaration declaration);
    void bindMeshTangentialBoundaryPolicyConsumer(
        FieldId mesh_displacement_field,
        int boundary_marker,
        MeshTangentialBoundaryPolicy policy,
        std::string operator_tag,
        std::string consumer_source);
    [[nodiscard]] std::span<const MeshTangentialBoundaryPolicyDeclaration>
    meshTangentialBoundaryPolicies() const noexcept {
        return mesh_tangential_boundary_policies_;
    }
    /**
     * @brief Publish tangential history in a serial/rank-local workflow.
     *
     * A distributed system must use @ref
     * recordAcceptedMeshBoundaryProvenance from its first nonempty accepted
     * record so normal and tangential publication share one transaction.
     */
    void recordAcceptedMeshTangentialBoundaryPolicies(
        std::uint64_t accepted_step,
        Real accepted_time,
        Real dt,
        std::uint64_t state_revision);
    [[nodiscard]] std::span<
        const MeshTangentialBoundaryPolicyHistoryRecord>
    meshTangentialBoundaryPolicyHistory() const noexcept {
        return mesh_tangential_boundary_policy_history_;
    }
    /**
     * @brief Atomically publish accepted normal and tangential mesh-boundary
     * provenance across the active FE communicator.
     *
     * Every communicator rank must call this method after every participating
     * system is fully set up with the same finalized active communicator.
     * Entering the collective with divergent setup or communicator state is a
     * caller error and cannot be coordinated. A local validation or allocation
     * failure after entry rolls both history vectors back on every rank. The
     * accepted step metadata and symbolic declarations must match exactly;
     * only each record's mesh geometry revision is deliberately rank-local.
     * Opaque callback targets are rejected for multi-rank publication until
     * they have a stable cross-rank identity contract. A globally empty
     * declaration set is a no-op and does not seal later declarations.
     */
    void recordAcceptedMeshBoundaryProvenance(
        std::uint64_t accepted_step,
        Real accepted_time,
        Real dt,
        std::uint64_t state_revision);
    void declareFreeSurfaceDiscreteFunctional(
        FreeSurfaceDiscreteFunctionalDeclaration declaration);
    [[nodiscard]] std::span<
        const FreeSurfaceDiscreteFunctionalDeclaration>
    freeSurfaceDiscreteFunctionalDeclarations() const noexcept {
        return free_surface_discrete_functional_declarations_;
    }
    void recordAcceptedFreeSurfaceDiscreteFunctionals(
        std::uint64_t accepted_step,
        Real accepted_time,
        Real dt,
        std::uint64_t pre_maintenance_endpoint_state_revision,
        std::uint64_t state_revision,
        std::span<const AcceptedFreeSurfaceDiscreteFunctionalState> states,
        std::optional<std::uint64_t> extension_map_revision =
            std::nullopt);
    [[nodiscard]] std::span<
        const FreeSurfaceDiscreteFunctionalHistoryRecord>
    freeSurfaceDiscreteFunctionalHistory() const noexcept {
        return free_surface_discrete_functional_history_;
    }
    void setGeometricNonlinearityPolicy(GeometricNonlinearityPolicy policy);
    [[nodiscard]] const GeometricNonlinearityPolicy& geometricNonlinearityPolicy() const noexcept;
    [[nodiscard]] bool geometricNonlinearityEnabled() const noexcept;
    GeometricNonlinearityTransactionEvent beginGeometricNonlinearityTrial(
        const SystemStateView& state);
    GeometricNonlinearityTransactionEvent acceptGeometricNonlinearityState(
        const SystemStateView& state,
        GeometricNonlinearityUpdatePoint update_point);
    GeometricNonlinearityTransactionEvent rollbackGeometricNonlinearityTrial(
        bool force = false);
    FieldId addInterfaceField(std::string name,
                              std::shared_ptr<const spaces::FunctionSpace> space,
                              InterfaceId interface_marker,
                              int components = 0);
    void addConstraint(std::unique_ptr<constraints::Constraint> c);
    void addSystemConstraint(std::unique_ptr<constraints::ISystemConstraint> c);
    [[nodiscard]] std::vector<
        constraints::SmallCutAggregationRefreshReport>
    completedSmallCutAggregationRefreshReports() const;
    [[nodiscard]] std::vector<std::shared_ptr<
        const constraints::SmallCutAggregationProlongationReport>>
    finalizedSmallCutAggregationProlongations() const;
    [[nodiscard]] std::span<const ExteriorBoundaryMeasurePolicy>
    exteriorBoundaryMeasurePolicies() const noexcept
    {
        return exterior_boundary_measure_policies_;
    }
    [[nodiscard]] std::span<
        const GeneratedBoundaryNitscheTracePolicy>
    generatedBoundaryNitscheTracePolicies() const noexcept
    {
        return generated_boundary_nitsche_trace_policies_;
    }
    [[nodiscard]] std::span<
        const GeneratedBoundaryNitscheTraceCertificateRecord>
    generatedBoundaryNitscheTraceCertificates() const;

    void addOperator(OperatorTag name);
    void setFormInstallCellDomainRestrictions(
        std::vector<FormCellDomainRestriction> restrictions);
    [[nodiscard]] const std::vector<FormCellDomainRestriction>&
    formInstallCellDomainRestrictions() const noexcept;
    [[nodiscard]] FormCellDomainRestrictionScope
    scopedFormInstallCellDomainRestrictions(
        std::vector<FormCellDomainRestriction> restrictions);

    /// @name Kernel registration (internal — do not use in physics modules)
    ///
    /// These methods are called by FormsInstaller internally. Physics modules
    /// should use the public FormsInstaller API instead:
    ///   - installFormulation()    for residual physics
    ///   - installMixedBilinear()  for mixed bilinear operators
    ///   - installMixedLinear()    for mixed linear operators
    /// @{

    void addCellKernel(OperatorTag op, FieldId field,
                       std::shared_ptr<assembly::AssemblyKernel> kernel);
    void addCellKernel(OperatorTag op, FieldId test_field, FieldId trial_field,
                       std::shared_ptr<assembly::AssemblyKernel> kernel);

    void addBoundaryKernel(OperatorTag op, BoundaryId boundary, FieldId field,
                           std::shared_ptr<assembly::AssemblyKernel> kernel);
    void addBoundaryKernel(OperatorTag op, BoundaryId boundary, FieldId test_field, FieldId trial_field,
                           std::shared_ptr<assembly::AssemblyKernel> kernel);

    void addInteriorFaceKernel(OperatorTag op, FieldId field,
                               std::shared_ptr<assembly::AssemblyKernel> kernel);
    void addInteriorFaceKernel(OperatorTag op, FieldId test_field, FieldId trial_field,
                               std::shared_ptr<assembly::AssemblyKernel> kernel);
    void addInteriorFaceKernel(OperatorTag op, int interior_facet_marker, FieldId field,
                               std::shared_ptr<assembly::AssemblyKernel> kernel);
    void addInteriorFaceKernel(OperatorTag op, int interior_facet_marker,
                               FieldId test_field, FieldId trial_field,
                               std::shared_ptr<assembly::AssemblyKernel> kernel);

    void addInterfaceFaceKernel(OperatorTag op, InterfaceId interface_marker, FieldId field,
                                std::shared_ptr<assembly::AssemblyKernel> kernel);
    void addInterfaceFaceKernel(OperatorTag op, InterfaceId interface_marker, FieldId test_field, FieldId trial_field,
                                std::shared_ptr<assembly::AssemblyKernel> kernel);

    void addCutVolumeKernel(OperatorTag op,
                            InterfaceId interface_marker,
                            geometry::CutIntegrationSide side,
                            FieldId field,
                            std::shared_ptr<assembly::AssemblyKernel> kernel);
    void addCutVolumeKernel(OperatorTag op,
                            InterfaceId interface_marker,
                            geometry::CutIntegrationSide side,
                            FieldId test_field,
                            FieldId trial_field,
                            std::shared_ptr<assembly::AssemblyKernel> kernel,
                            std::string source_component_tag = {});

    void addGlobalKernel(OperatorTag op,
                         std::shared_ptr<GlobalKernel> kernel);

    /// @}

    // ---- Optional operator backends (Milestone 5) ----
    void addMatrixFreeKernel(OperatorTag op,
                             std::shared_ptr<assembly::IMatrixFreeKernel> kernel);
    void addMatrixFreeKernel(OperatorTag op,
                             std::shared_ptr<assembly::IMatrixFreeKernel> kernel,
                             const assembly::MatrixFreeOptions& options);
    [[nodiscard]] std::shared_ptr<assembly::MatrixFreeOperator>
    matrixFreeOperator(const OperatorTag& op) const;
    [[nodiscard]] std::size_t matrixFreeOperatorRebuildCount(const OperatorTag& op) const;
    [[nodiscard]] OperatorRevisionSnapshot matrixFreeOperatorRevisionSnapshot(const OperatorTag& op) const;
    [[nodiscard]] OperatorInvalidationDecision matrixFreeOperatorLastInvalidation(const OperatorTag& op) const;

    post::DerivedResultHandle addDerivedResult(post::DerivedResultDefinition def);
    [[nodiscard]] std::span<const post::DerivedResultDefinition> derivedResults() const noexcept;
    void appendDerivedResultFields(
        svmp::MeshBase& mesh,
        const SystemStateView& state,
        const post::DerivedResultOutputOptions& options = {}) const;

    void addFunctionalKernel(std::string tag,
                             std::shared_ptr<assembly::FunctionalKernel> kernel);
    [[nodiscard]] Real evaluateFunctional(const std::string& tag,
                                          const SystemStateView& state) const;
    [[deprecated("use BoundaryReductionService for boundary reductions")]]
    [[nodiscard]] Real evaluateBoundaryFunctional(
        const std::string& tag,
        int boundary_marker,
        const SystemStateView& state) const;

    /**
     * @brief Access the boundary reduction service for a given primary field.
     *
     * Lazily creates the service on first access.  The service provides
     * physics-agnostic boundary-integral evaluation and is shared with
     * the AuxiliaryInputRegistry-backed FE input pipeline.
     */
    BoundaryReductionService& boundaryReductionService(FieldId primary_field);
    [[nodiscard]] BoundaryReductionService* boundaryReductionServiceIfPresent(FieldId primary_field) noexcept
    {
        auto it = boundary_reduction_services_.find(primary_field);
        return it != boundary_reduction_services_.end() ? it->second.get() : nullptr;
    }
    [[nodiscard]] const BoundaryReductionService* boundaryReductionServiceIfPresent(FieldId primary_field) const noexcept
    {
        auto it = boundary_reduction_services_.find(primary_field);
        return it != boundary_reduction_services_.end() ? it->second.get() : nullptr;
    }

    /**
     * @brief Access the generalized auxiliary state manager.
     *
     * Lazily creates the manager on first access.  The manager owns
     * auxiliary blocks with any scope (Global, Node, Cell, etc.) and
     * provides distributed ownership, sync, and checkpoint APIs.
     */
    AuxiliaryStateManager& auxiliaryStateManager();
    [[nodiscard]] AuxiliaryStateManager* auxiliaryStateManagerIfPresent() noexcept
    {
        return auxiliary_state_manager_.get();
    }
    [[nodiscard]] const AuxiliaryStateManager* auxiliaryStateManagerIfPresent() const noexcept
    {
        return auxiliary_state_manager_.get();
    }

    /**
     * @brief Access the auxiliary operator registry.
     *
     * Lazily creates the registry on first access.  Owns auxiliary
     * operators, coupling graph, and monolithic unknown layouts.
     */
    AuxiliaryOperatorRegistry& auxiliaryOperatorRegistry();
    [[nodiscard]] AuxiliaryOperatorRegistry* auxiliaryOperatorRegistryIfPresent() noexcept
    {
        return auxiliary_operator_registry_.get();
    }
    [[nodiscard]] const AuxiliaryOperatorRegistry* auxiliaryOperatorRegistryIfPresent() const noexcept
    {
        return auxiliary_operator_registry_.get();
    }

    /**
     * @brief Access the auxiliary input registry.
     *
     * Lazily creates the registry on first access.
     */
    AuxiliaryInputRegistry& auxiliaryInputRegistry();
    [[nodiscard]] AuxiliaryInputRegistry* auxiliaryInputRegistryIfPresent() noexcept
    {
        return auxiliary_input_registry_.get();
    }
    [[nodiscard]] const AuxiliaryInputRegistry* auxiliaryInputRegistryIfPresent() const noexcept
    {
        return auxiliary_input_registry_.get();
    }

    /**
     * @brief Access the FE-backed quantity definition registry.
     *
     * Lazily creates the registry on first access.
     */
    FEQuantityRegistry& feQuantityRegistry();
    [[nodiscard]] const FEQuantityRegistry* feQuantityRegistryIfPresent() const noexcept
    {
        return fe_quantity_registry_.get();
    }

    /**
     * @brief Register a sampled-state-field auxiliary input.
     *
     * Creates an entity-local input that samples the named FE field at
     * each node using direct DOF lookup (fast path for Lagrange elements).
     * Must be called after `setup()` (so DOF handlers exist) and before
     * `finalizeAuxiliaryLayout()`.
     *
     * @param input_name   Registry name for the input.
     * @param field_name   Name of the FE field to sample.
     * @param n_entities   Number of entities (nodes).
     */
    void registerSampledFieldInput(
        const std::string& input_name,
        const std::string& field_name,
        std::size_t n_entities);

    /**
     * @brief Register a boundary-face nodal sum auxiliary input.
     *
     * Creates a global input that sums all field DOF components at unique
     * boundary face vertices with the given marker.  The output size
     * equals the field component count.
     *
     * Requires `setup()` to have been called and a vertex-based (Lagrange)
     * FE space; throws if the field has no vertex DOFs.  For a quadrature-
     * weighted boundary integral, use the BoundaryFunctional assembly pipeline.
     */
    void registerBoundaryNodalSumInput(
        const std::string& input_name,
        const std::string& field_name,
        int boundary_marker);

    /**
     * @brief Register a true quadrature-weighted boundary integral as an auxiliary input.
     *
     * Creates a global scalar input backed by a real FE boundary functional
     * (not a nodal sum surrogate).  The input is evaluated via the
     * BoundaryReductionService and stored in the AuxiliaryInputRegistry
     * so that `AuxiliaryInput("name")` resolves to the integral value.
     *
     * This is the physics-agnostic API for registering boundary-integral
     * auxiliary inputs.  It supports:
     *
     * - `Sum` reduction (default): raw integral value.
     * - `Average` reduction: integral divided by boundary measure.
     * - Room for `Min`/`Max` in future extensions.
     *
     * ## Lifecycle
     *
     * May be called before or after `setup()`.  Must be called before
     * `installFormulation()` if the input name appears in an
     * `AuxiliaryInput(...)` symbol, and before `finalizeAuxiliaryLayout()`
     * if the input feeds an auxiliary model.  Both constraints are naturally
     * satisfied when called from a module's `registerOn()` method before
     * form installation.
     *
     * ## Multi-field support
     *
     * Integrands may reference multiple FE fields.  The first referenced
     * field provides the DOF layout and quadrature context; secondary
     * fields are automatically bound via `registerSecondaryField()` with
     * correct `field_type`, `component_offset`, and block DOF mapping.
     *
     * @param input_name   Registry name for the input (e.g., "Q").
     * @param functional   Boundary functional definition (integrand, marker, reduction).
     * @param schedule     When the input is re-evaluated (default: OncePerTimeStep).
     */
    void registerBoundaryIntegralInput(
        const std::string& input_name,
        forms::BoundaryFunctional functional,
        AuxiliaryInputUpdateSchedule schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep);

    /**
     * @brief Register a true quadrature-weighted boundary integral as an auxiliary input.
     *
     * Convenience overload that constructs a BoundaryFunctional from components.
     *
     * @param input_name       Registry name for the input.
     * @param integrand        Scalar-valued integrand expression.
     * @param boundary_marker  Boundary label to integrate over.
     * @param reduction        Reduction mode (default: Sum).
     * @param schedule         When the input is re-evaluated.
     */
    void registerBoundaryIntegralInput(
        const std::string& input_name,
        forms::FormExpr integrand,
        int boundary_marker,
        forms::BoundaryFunctional::Reduction reduction = forms::BoundaryFunctional::Reduction::Sum,
        AuxiliaryInputUpdateSchedule schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep);

    // ---- Setup phase ----
    [[nodiscard]] SetupStoragePlan computeSetupStoragePlan() const;
    [[nodiscard]] const SetupStoragePlan& setupStoragePlan() const noexcept { return setup_storage_plan_; }
    void setup(const SetupOptions& opts = {}, const SetupInputs& inputs = {});

    // ---- Constraints lifecycle ----
    void updateConstraints(double time, double dt = 0.0);
    void rebuildConstraintState();
    [[nodiscard]] constraints::ConstraintDependencyDeclaration constraintDependencyDeclaration() const;
    [[nodiscard]] constraints::ConstraintRevisionSnapshot constraintRevisionSnapshot() const noexcept;
    [[nodiscard]] bool constraintStateStaleForCurrentRevisions() const;
    constraints::ConstraintRefreshResult refreshConstraintStateForCurrentRevisions(
        double time = 0.0,
        double dt = 0.0,
        bool allow_structural_rebuild = true);

    // ---- Assembly phase ----
    assembly::AssemblyResult assemble(
        const AssemblyRequest& req,
        const SystemStateView& state,
        assembly::GlobalSystemView* matrix_out,
        assembly::GlobalSystemView* vector_out);

    assembly::AssemblyResult assembleResidual(
        const SystemStateView& state,
        assembly::GlobalSystemView& rhs_out);

    assembly::AssemblyResult assembleJacobian(
        const SystemStateView& state,
        assembly::GlobalSystemView& jac_out);

    assembly::AssemblyResult assembleMass(
        const SystemStateView& state,
        assembly::GlobalSystemView& mass_out);

    // ---- Time stepping lifecycle ----
    void beginTimeStep(bool reset_auxiliary_state = true,
                       bool invalidate_auxiliary_inputs = true);
    void commitTimeStep();

    // ---- Auxiliary model deployment ----

    /**
     * @brief Deploy an auxiliary model instance into the system.
     *
     * Collects the instance for setup-time finalization.  Must be called
     * before `setup()`.  During `finalizeAuxiliaryLayout()`, deployed
     * instances are registered as blocks, inputs, and steppers.
     */
    void deployAuxiliaryModel(AuxiliaryDeployedInstance instance);

    /**
     * @brief Deploy an auxiliary model and return a typed instance handle.
     *
     * Preferred over `deployAuxiliaryModel()` — returns a handle for
     * string-free output access.
     *
     * ```cpp
     * auto rcr = system.deploy(use(model).name("rcr_1")...);
     * auto p_out = rcr.output("P_out");
     * ```
     */
    AuxiliaryInstanceHandle deploy(AuxiliaryDeployedInstance instance);

    /// Select the active key for an auxiliary variant group.
    void selectAuxiliaryVariant(std::string group, std::string key);

    /// Clear a previously selected auxiliary variant group.
    void clearAuxiliaryVariantSelection(std::string_view group);

    // ---- Handle-returning auxiliary input registration ----

    /**
     * @brief Register a boundary integral as an auxiliary input and return a handle.
     *
     * ```cpp
     * auto Q = system.boundaryIntegral(inner(u, n), marker);
     * ```
     */
    [[deprecated("boundaryIntegral(name, ...) is deprecated; use boundaryIntegral(...) without an explicit name")]]
    AuxiliaryInputHandle boundaryIntegral(
        const std::string& input_name,
        forms::FormExpr integrand,
        int boundary_marker,
        forms::BoundaryFunctional::Reduction reduction = forms::BoundaryFunctional::Reduction::Sum,
        AuxiliaryInputUpdateSchedule schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep);

    /**
     * @brief Register a boundary integral with an auto-generated internal name.
     */
    AuxiliaryInputHandle boundaryIntegral(
        forms::FormExpr integrand,
        int boundary_marker,
        forms::BoundaryFunctional::Reduction reduction = forms::BoundaryFunctional::Reduction::Sum,
        AuxiliaryInputUpdateSchedule schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep);

    /**
     * @brief Register a boundary integral (full functional) and return a handle.
     */
    [[deprecated("boundaryIntegral(name, functional, ...) is deprecated; use boundaryIntegral(functional, ...) without an explicit name")]]
    AuxiliaryInputHandle boundaryIntegral(
        const std::string& input_name,
        forms::BoundaryFunctional functional,
        AuxiliaryInputUpdateSchedule schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep);

    /**
     * @brief Register a boundary integral functional with an auto-generated internal name.
     */
    AuxiliaryInputHandle boundaryIntegral(
        forms::BoundaryFunctional functional,
        AuxiliaryInputUpdateSchedule schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep);

    /**
     * @brief Register a derived auxiliary input computed from an expression.
     *
     * Auto-discovers dependencies on other auxiliary inputs referenced in `expr`.
     * The expression is evaluated using the current auxiliary input values.
     *
     * ```cpp
     * auto P_out = system.derivedInput("P_out", Pd + (Rp + Rd) * Q);
     * ```
     *
     * @param name  Registry name for the derived input.
     * @param expr  Expression to evaluate (may reference other auxiliary inputs).
     * @param schedule  When the input is re-evaluated.
     * @return Handle for binding and expression use.
     */
    AuxiliaryInputHandle derivedInput(
        const std::string& name,
        forms::FormExpr expr,
        AuxiliaryInputUpdateSchedule schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep);

    /**
     * @brief Register a sampled FE field as an auxiliary input and return a handle.
     *
     * ```cpp
     * auto u_sample = system.sampledField("u_sample", "u", n_nodes);
     * ```
     */
    AuxiliaryInputHandle sampledField(
        const std::string& input_name,
        const std::string& field_name,
        std::size_t n_entities);

    /**
     * @brief Register a boundary nodal sum as an auxiliary input and return a handle.
     *
     * ```cpp
     * auto Q_nodal = system.boundaryNodalSum("Q_nodal", "u", marker);
     * ```
     */
    AuxiliaryInputHandle boundaryNodalSum(
        const std::string& input_name,
        const std::string& field_name,
        int boundary_marker);

    /**
     * @brief Register a boundary average as an auxiliary input.
     *
     * Computes `∫_Γ expr ds / ∫_Γ 1 ds` on the boundary with the given marker.
     */
    AuxiliaryInputHandle boundaryAverage(
        const std::string& input_name,
        forms::FormExpr integrand,
        int boundary_marker,
        AuxiliaryInputUpdateSchedule schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep);

    /**
     * @brief Register a domain integral as an auxiliary input.
     *
     * Computes `∫_Ω expr dx` over all cells.
     */
    AuxiliaryInputHandle domainIntegral(
        const std::string& input_name,
        forms::FormExpr integrand,
        AuxiliaryInputUpdateSchedule schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep);

    /**
     * @brief Register a domain average as an auxiliary input.
     *
     * Computes `∫_Ω expr dx / ∫_Ω 1 dx` over all cells.
     */
    AuxiliaryInputHandle domainAverage(
        const std::string& input_name,
        forms::FormExpr integrand,
        AuxiliaryInputUpdateSchedule schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep);

    /**
     * @brief Register a topology-region-local integral as an entity-local input.
     *
     * Computes one `∫_{R_i} expr dx` value per topology region.  Region-scoped
     * auxiliary blocks consume the value for their materialized region entity
     * through the normal entity-local input registry path.
     */
    AuxiliaryInputHandle regionIntegral(
        const std::string& input_name,
        forms::FormExpr integrand,
        AuxiliaryInputUpdateSchedule schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep);

    /**
     * @brief Register a region-restricted integral as an auxiliary input.
     *
     * Computes `∫_R expr dx` over cells matching the given marker.
     */
    AuxiliaryInputHandle regionIntegral(
        const std::string& input_name,
        forms::FormExpr integrand,
        int region_marker,
        AuxiliaryInputUpdateSchedule schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep);

    /**
     * @brief Register a topology-region-local average as an entity-local input.
     *
     * Computes one `∫_{R_i} expr dx / ∫_{R_i} 1 dx` value per topology region.
     */
    AuxiliaryInputHandle regionAverage(
        const std::string& input_name,
        forms::FormExpr integrand,
        AuxiliaryInputUpdateSchedule schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep);

    /**
     * @brief Register a region-restricted average as an auxiliary input.
     *
     * Computes `∫_R expr dx / ∫_R 1 dx` over cells matching the given marker.
     */
    AuxiliaryInputHandle regionAverage(
        const std::string& input_name,
        forms::FormExpr integrand,
        int region_marker,
        AuxiliaryInputUpdateSchedule schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep);

    /**
     * @brief Register a generic FE expression as an auxiliary input.
     *
     * Evaluates the expression at a representative point (cell centroid)
     * for each entity.  For scalar global quantities, use domainIntegral()
     * or boundaryIntegral() instead.
     */
    AuxiliaryInputHandle feExpression(
        const std::string& input_name,
        forms::FormExpr expression,
        AuxiliaryInputUpdateSchedule schedule = AuxiliaryInputUpdateSchedule::OncePerTimeStep);

    // ---- Auxiliary state lifecycle ----

    /**
     * @brief Prepare auxiliary inputs and state for assembly.
     *
     * Evaluates all auxiliary input providers (respecting schedules),
     * and binds auxiliary values into the assembler context.
     * Called before each PDE assembly (including within Newton iterations).
     *
     * @param state  Current system state view.
     * @param is_nonlinear_iteration  If true, refreshes inputs with
     *        `EachNonlinearIteration` schedule.
     */
    void prepareAuxiliaryForAssembly(const SystemStateView& state,
                                      bool is_nonlinear_iteration = false);

    /**
     * @brief Advance all Partitioned auxiliary blocks by one time step.
     *
     * Dispatches to the per-block stepper.  Monolithic blocks are NOT
     * advanced here — their time discretization is part of the global
     * assembled solve.
     *
     * Respects per-block scheduling (SingleRate, Subcycled, Multirate).
     *
     * @warning This overload does NOT update the cached system state used by
     * FE-coupled auxiliary input callbacks (boundary integrals, sampled fields).
     * Those callbacks read from cached spans/pointers that were populated by the
     * most recent call to `prepareAuxiliaryForAssembly()` or
     * `advanceAuxiliaryState(const SystemStateView&)`.  If no such call has been
     * made in this time step, or if the underlying data (solution vectors) has
     * been freed or overwritten, the callbacks will read stale or invalid data.
     *
     * Use `advanceAuxiliaryState(const SystemStateView&)` instead when any
     * registered auxiliary input depends on FE field state.
     *
     * @param time  Current simulation time.
     * @param dt    PDE time step.
     */
    void advanceAuxiliaryState(Real time, Real dt);

    /**
     * @brief Advance auxiliary state with full system state context.
     *
     * Caches the system state (solution, previous solutions, time integration
     * context, user data) before evaluating auxiliary inputs and stepping.
     * This ensures that FE-coupled input callbacks (boundary integrals, sampled
     * fields) have access to valid, current data.
     *
     * **This is the preferred overload** when any registered auxiliary input
     * depends on FE field state (e.g., boundary-integral inputs registered via
     * `registerBoundaryIntegralInput()`).
     *
     * @param state  Full system state (time, dt, solution, history, etc.).
     */
    void advanceAuxiliaryState(const SystemStateView& state);

    /**
     * @brief Advance auxiliary state with full system state context and
     *        nonlinear-iteration-aware input refresh.
     *
     * Behaves like `advanceAuxiliaryState(const SystemStateView&)`, but when
     * `is_nonlinear_iteration` is true it also refreshes auxiliary inputs whose
     * update schedule is `EachNonlinearIteration` before stepping.
     *
     * @param state  Full system state (time, dt, solution, history, etc.).
     * @param is_nonlinear_iteration  If true, refreshes inputs with
     *        `EachNonlinearIteration` schedule before stepping.
     */
    void advanceAuxiliaryState(const SystemStateView& state,
                               bool is_nonlinear_iteration);

    /**
     * @brief Assemble monolithic auxiliary residual and Jacobian.
     *
     * Evaluates the residual F(xdot, x, ...) and Jacobian dF/dx for all
     * monolithic auxiliary blocks.  Results are stored in the provided
     * dense vectors/matrices.
     *
     * @param time  Current time.
     * @param dt    Time step (for xdot computation).
     * @param residual_out  Output residual vector (sized to total_aux_unknowns).
     * @param jacobian_out  Output Jacobian matrix (row-major, n×n).
     * @param is_nonlinear_iteration  When true, EachNonlinearIteration
     *        auxiliary inputs are refreshed before assembly.  Pass true
     *        on each Newton iteration; false (default) on the first call.
     */
    void assembleMonolithicAuxiliary(
        Real time, Real dt,
        std::span<Real> residual_out,
        std::span<Real> jacobian_out,
        bool is_nonlinear_iteration = false);

    /**
     * @brief Assemble mixed auxiliary contributions into dense outputs (test helper).
     *
     * Assembles the monolithic auxiliary blocks + chain-rule field-auxiliary
     * coupling into dense vector/matrix outputs sized for the mixed system
     * (n_field_dofs + n_aux_dofs).
     *
     * @param state           System state.
     * @param n_field_dofs    Number of FE field DOFs.
     * @param residual_out    Dense vector (n_field + n_aux).
     * @param matrix_out      Dense row-major matrix ((n_field+n_aux)^2).
     */
    void assembleMixedAuxiliaryDense(
        const SystemStateView& state,
        std::size_t n_field_dofs,
        std::vector<Real>& residual_out,
        std::vector<Real>& matrix_out);

    /**
     * @brief Get the composed mixed system layout.
     *
     * Only valid after `finalizeAuxiliaryLayout()`.
     * @param n_field_unknowns  Number of FE field DOFs.
     */
    [[nodiscard]] MixedSystemLayout composeMixedSystemLayout(
        std::size_t n_field_unknowns = 0) const;

    /**
     * @brief Copy solver options and enrich them with mixed field/auxiliary block metadata.
     *
     * The returned options keep the caller's solver/preconditioner settings and add:
     * - an absolute-offset mixed block layout
     * - unambiguous backend-facing role-to-name mappings
     *
     * @param base              Existing solver options to augment.
     * @param n_field_unknowns  FE-field unknown count for the active operator.
     */
    [[nodiscard]] backends::SolverOptions augmentSolverOptions(
        const backends::SolverOptions& base,
        std::size_t n_field_unknowns) const;

    /**
     * @brief Convenience overload that uses the configured FE DOF count when available.
     */
    [[nodiscard]] backends::SolverOptions augmentSolverOptions(
        const backends::SolverOptions& base) const;

    /**
     * @brief Rollback all auxiliary blocks to their committed state.
     *
     * Used after a failed nonlinear solve or rejected time step.
     */
    void rollbackAuxiliaryState();

    /**
     * @brief Convert monolithic auxiliary stage values to end-of-step values.
     *
     * Generalized-alpha stage solves store differential auxiliary variables at
     * the stage state x_{n+alpha_f}. Before commit, these must be mapped back
     * to x_{n+1}. Algebraic rows are left unchanged.
     *
     * @param alpha_f    Generalized-alpha stage weight.
     * @param final_time Physical time of the accepted end-of-step state.
     */
    void finalizeMonolithicAuxiliaryStageState(Real alpha_f, Real final_time);

    /**
     * @brief Finalize monolithic auxiliary stage values and update stored
     *        committed rates for first-order generalized-alpha.
     *
     * @param alpha_f    Generalized-alpha stage weight.
     * @param gamma      Generalized-alpha gamma parameter.
     * @param dt         Full time-step size.
     * @param final_time Physical time of the accepted end-of-step state.
     */
    void finalizeMonolithicAuxiliaryStageState(Real alpha_f, Real gamma, Real dt, Real final_time);

    /**
     * @brief Finalize auxiliary layouts during setup.
     *
     * Called from `setup()` after all auxiliary model instances have been
     * deployed.  Finalizes monolithic unknown layouts and builds any
     * requested symbolic derivative artifacts.
     */
    void finalizeAuxiliaryLayout();

    /**
     * @brief Pack all auxiliary state for checkpoint.
     */
    [[nodiscard]] std::vector<Real> checkpointAuxiliaryState() const;

    /**
     * @brief Restore auxiliary state from checkpoint data.
     */
    void restoreAuxiliaryState(std::span<const Real> data);

    /**
     * @brief Stable logical descriptor for one deployed auxiliary output.
     */
    struct AuxiliaryOutputDescriptor {
        std::uint32_t id{0u};
        std::string instance_name{};
        std::string output_name{};
        std::size_t output_index{0u};
    };

    /**
     * @brief Get the flattened evaluated auxiliary output values.
     *
     * Updated by `prepareAuxiliaryForAssembly()`.  Empty if no
     * deployed models have output expressions.
     */
    [[nodiscard]] std::span<const Real> auxiliaryOutputValues() const noexcept;

    /**
     * @brief Get the flattened current auxiliary work-state values.
     *
     * Updated on demand from the auxiliary state manager and returned in block
     * registration order. Empty if no auxiliary blocks are deployed.
     */
    [[nodiscard]] std::span<const Real> auxiliaryStateValues() const noexcept;

    /**
     * @brief Get the stable logical id of a named auxiliary output.
     *
     * Unlike `auxiliaryOutputSlotOf(...)`, this identity is assigned at deploy
     * time and does not depend on entity counts or finalized flattened layout.
     */
    [[nodiscard]] std::size_t auxiliaryOutputIdOf(std::string_view output_name) const;

    /// Instance-qualified stable logical output id lookup.
    [[nodiscard]] std::size_t auxiliaryOutputIdOf(
        std::string_view instance_name, std::string_view output_name) const;

    /// Lookup the deploy-time descriptor for a stable logical output id.
    [[nodiscard]] const AuxiliaryOutputDescriptor* auxiliaryOutputDescriptor(
        std::size_t output_id) const noexcept;

    /**
     * @brief Get the flattened slot index of a named auxiliary output.
     *
     * Outputs are flattened across all deployed models in deployment order.
     * Each model contributes `n_entities * n_outputs` slots in the flat
     * buffer.  Returns the entity-0 slot for the named output; per-entity
     * access is `slot + entity_index * n_outputs_for_that_model`.
     *
     * Safe to call after `finalizeAuxiliaryLayout()` (does not depend on
     * runtime-populated output buffers).
     *
     * @return Slot index, or `std::size_t(-1)` if not found.
     */
    [[nodiscard]] std::size_t auxiliaryOutputSlotOf(std::string_view output_name) const;

    /**
     * @brief Instance-qualified output slot lookup.
     *
     * Use this overload when multiple deployed models have outputs with
     * the same name (e.g., two RCR models each exposing "P_out").
     *
     * @param instance_name  The deployed instance name.
     * @param output_name    The output name within that instance.
     * @return Slot index, or `std::size_t(-1)` if not found.
     */
    [[nodiscard]] std::size_t auxiliaryOutputSlotOf(
        std::string_view instance_name, std::string_view output_name) const;

    /**
     * @brief Lowered algebraic output expression lookup by symbolic output name.
     *
     * When a deployed AuxiliaryState output can be expressed directly in terms
     * of runtime-available terminals (for example AuxiliaryStateRef,
     * AuxiliaryInputRef, and constants), setup may lower that output to a
     * direct expression for assembly-time substitution in a plain NaturalBC.
     *
     * For live monolithic blocks, formulation metadata may still preserve the
     * original AuxiliaryOutputRef so the bordered direct-coupling path can
     * extract dR/d(output). Fully lowered direct-only blocks instead lower the
     * metadata as well because there is no live auxiliary unknown to couple.
     */
    [[nodiscard]] std::optional<forms::FormExpr>
    loweredAuxiliaryOutputExpr(std::string_view output_name) const;

    /// Slot-based lowered algebraic output lookup.
    [[nodiscard]] std::optional<forms::FormExpr>
    loweredAuxiliaryOutputExpr(std::size_t slot) const;

    /**
     * @brief Evaluate a scalar auxiliary-backed value for lowered system constraints.
     *
     * The binding may reference either a named auxiliary state or a named
     * auxiliary output on the deployed instance.
     */
    [[nodiscard]] Real auxiliaryConstraintValue(std::string_view instance_name,
                                                const AuxiliaryConstraintBinding& binding,
                                                Real time,
                                                Real dt) const;

    /**
     * @brief Return true when formulation metadata should keep AuxiliaryOutputRef.
     *
     * Live monolithic blocks preserve the output reference in metadata so the
     * bordered direct-coupling path can still assemble dR/d(output). Direct-only
     * lowered blocks return false because their metadata should use the same
     * lowered expression as assembly.
     */
    [[nodiscard]] bool auxiliaryOutputMetadataUsesRef(std::string_view output_name) const;

    [[nodiscard]] std::vector<analysis::AuxiliaryOutputConsumerRecord>
    consumersOfAuxiliaryOutput(std::size_t output_id) const;

    [[nodiscard]] std::vector<analysis::AuxiliaryOutputConsumerRecord>
    consumersOfInstance(std::string_view instance_name) const;

    /**
     * @brief Get an analysis summary of auxiliary blocks and inputs.
     */
    struct AuxiliaryAnalysisSummary {
        std::size_t n_blocks{0};
        std::size_t n_partitioned{0};
        std::size_t n_monolithic{0};
        std::size_t n_inputs{0};
        std::size_t total_aux_unknowns{0};
        std::size_t n_constraint_like_blocks{0};
        std::size_t n_schur_eliminable_blocks{0};
        std::size_t n_special_precondition_blocks{0};
        std::vector<std::string> block_names{};
        std::vector<std::string> input_names{};
        std::vector<std::string> constraint_like_block_names{};
        std::vector<std::string> schur_eliminable_block_names{};
        std::vector<std::string> special_precondition_block_names{};
    };
    [[nodiscard]] AuxiliaryAnalysisSummary auxiliaryAnalysisSummary() const;

	    // ---- Accessors ----
	    [[nodiscard]] const assembly::IMeshAccess& meshAccess() const;
	    [[nodiscard]] std::span<const MeshParticipantInfo> meshParticipants() const noexcept;
	    [[nodiscard]] bool hasMeshParticipants() const noexcept { return !mesh_participants_.empty(); }
	    [[nodiscard]] bool hasSingleMeshParticipant() const noexcept { return mesh_participants_.size() == 1u; }
	    [[nodiscard]] bool hasCompositeMeshAccess() const noexcept { return mesh_participants_.size() > 1u; }
	    [[nodiscard]] const MeshParticipantInfo* meshParticipantByName(std::string_view name) const noexcept;
	    [[nodiscard]] const MeshParticipantInfo* meshParticipantByDomain(int domain_id) const noexcept;
	    [[nodiscard]] const MeshParticipantInfo* meshParticipantForCell(GlobalIndex cell_id) const noexcept;
	    [[nodiscard]] const MeshParticipantInfo* fieldMeshParticipant(FieldId field) const;
	    [[nodiscard]] bool fieldActiveOnCell(FieldId field, GlobalIndex cell_id) const;
	    [[nodiscard]] std::string assemblerName() const;
	    [[nodiscard]] std::string assemblerSelectionReport() const;
	    [[nodiscard]] const ISearchAccess* searchAccess() const noexcept { return search_access_.get(); }
	    void setSearchAccess(std::shared_ptr<const ISearchAccess> access) { search_access_ = std::move(access); }

	    /**
	     * @brief Locate a physical point in the mesh using the configured search access.
	     */
	    [[nodiscard]] ISearchAccess::PointLocation locatePoint(const std::array<Real, 3>& point,
	                                                           GlobalIndex hint_cell = INVALID_GLOBAL_INDEX) const;

	    /**
	     * @brief Evaluate a field at a physical point (search + reference-space interpolation).
	     *
	     * @return nullopt if no search access is configured or the point is not located in the mesh.
	     */
	    [[nodiscard]] std::optional<std::array<Real, 3>> evaluateFieldAtPoint(FieldId field,
	                                                                          const SystemStateView& state,
	                                                                          const std::array<Real, 3>& point,
	                                                                          GlobalIndex hint_cell = INVALID_GLOBAL_INDEX) const;

	    /**
	     * @brief Evaluate a field at all mesh vertices by direct DOF coefficient lookup.
	     *
	     * For Lagrange elements, basis functions equal 1 at their associated node and 0
	     * at all others, so the field value at a vertex equals the DOF coefficient.
	     * This avoids the expensive locatePoint + evaluate per vertex.
	     *
	     * @param field       Field to evaluate
	     * @param state       Current system state
	     * @param n_vertices  Number of mesh vertices
	     * @param out         Output buffer, size >= n_vertices * max(1, components)
	     * @return true if direct nodal evaluation was used, false if not supported
	     *         (caller should fall back to evaluateFieldAtPoint)
	     */
	    [[nodiscard]] bool evaluateFieldAtVertices(FieldId field,
	                                               const SystemStateView& state,
	                                               GlobalIndex n_vertices,
	                                               std::span<double> out) const;

	    struct MeshVertexFieldProjectionResult {
	        std::size_t values_written{0};
	        std::size_t unassigned_dofs{0};
	    };

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
	    /**
	     * @brief Project mesh vertex field values into nodal FE coefficients.
	     *
	     * The projection uses EntityDofMap vertex, edge, and cell-interior
	     * associations. It fails closed for non-nodal spaces or unsupported
	     * high-order entity layouts instead of relying on cell-local DOF order.
	     */
	    [[nodiscard]] MeshVertexFieldProjectionResult
	    projectMeshVertexValuesToFieldCoefficients(
	        FieldId field,
	        std::span<const Real> mesh_values,
	        std::size_t mesh_components,
	        std::span<Real> coefficients,
	        std::span<std::uint8_t> assigned = {},
	        std::string_view context =
	            "FESystem::projectMeshVertexValuesToFieldCoefficients") const;
#endif

	    // ---- Global-kernel persistent state (optional) ----
	    [[nodiscard]] assembly::MaterialStateView globalKernelCellState(const GlobalKernel& kernel,
	                                                                    GlobalIndex cell_id,
	                                                                    LocalIndex num_qpts) const;
    [[nodiscard]] assembly::MaterialStateView globalKernelBoundaryFaceState(const GlobalKernel& kernel,
                                                                            GlobalIndex face_id,
                                                                            LocalIndex num_qpts) const;
    [[nodiscard]] assembly::MaterialStateView globalKernelInteriorFaceState(const GlobalKernel& kernel,
                                                                            GlobalIndex face_id,
                                                                            LocalIndex num_qpts) const;

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
	    [[nodiscard]] const svmp::Mesh* mesh() const noexcept { return mesh_.get(); }
	    [[nodiscard]] bool hasSingleNativeMesh() const noexcept { return mesh_ != nullptr; }
	    [[nodiscard]] const svmp::Mesh& singleMesh(std::string_view api_name = "FESystem") const;
	    [[nodiscard]] svmp::Configuration coordinateConfiguration() const noexcept { return coord_cfg_; }

	    FieldId addMeshDisplacementUnknown(std::string name,
	                                       std::shared_ptr<const spaces::FunctionSpace> space,
	                                       int components = 0);
	    void beginMeshCoordinateTransaction();
	    MeshCoordinateUpdateResult updateCurrentCoordinatesFromMeshDisplacement(
	        const SystemStateView& state,
	        const MeshCoordinateUpdateOptions& options = {});
	    void commitMeshCoordinateTransaction();
	    void rollbackMeshCoordinateTransaction();
	    void rebaseMeshReferenceToCurrent(const svmp::ReferenceRebaseOptions& options = {});
	    void rebaseMeshReferenceCoordinates(
	        std::vector<svmp::real_t> Xref,
	        const svmp::ReferenceRebaseOptions& options = {});
	    [[nodiscard]] bool meshCoordinateTransactionActive() const noexcept;
	    [[nodiscard]] GeometryTransactionState meshCoordinateTransactionState() const noexcept;
	    [[nodiscard]] GeometryConfigurationUse geometryConfigurationUse() const noexcept;
	    [[nodiscard]] GeometryTransactionDiagnostics geometryTransactionDiagnostics() const;
	    void addGeometryTransactionCallback(GeometryTransactionCallback hook);
	    bool rebaseGeometricNonlinearityReference(
	        const svmp::ReferenceRebaseOptions& options = {});
	    std::size_t resetBoundMeshMotionField(MeshMotionFieldRole role, Real value = Real(0));

	    std::size_t bindStandardMeshMotionFieldsByName();
	    std::size_t syncPrescribedVertexFieldsFromMeshFields();
	    std::size_t syncBoundMeshMotionFieldsToPrescribedBuffers();
	    std::size_t syncBoundMeshMotionFieldsToState(std::span<Real> state) const;
	    std::size_t syncBoundMeshMotionFieldsToState(assembly::GlobalSystemView& vector_view) const;
	    void notifyMeshGeometryAdvanced();
	    void notifyMeshReferenceRebased();
	    void notifyMeshTopologyLayoutChanged();
	    FEAdaptedStateTransferResult onMeshAdapted(
	        const svmp::MeshBase& old_mesh,
	        const svmp::MeshBase& new_mesh,
	        const svmp::RefinementDelta& delta,
	        const svmp::AdaptivityOptions& options,
	        const FEAdaptedStateTransferRequest& request = {});

	    void setInterfaceMesh(InterfaceId marker, std::shared_ptr<const svmp::InterfaceMesh> mesh);
	    [[nodiscard]] bool hasInterfaceMesh(InterfaceId marker) const noexcept;
	    [[nodiscard]] const svmp::InterfaceMesh& interfaceMesh(InterfaceId marker) const;

	    void setInterfaceMeshFromFaceSet(InterfaceId marker,
	                                     const std::string& face_set_name,
	                                     bool compute_orientation = true);
	    void setInterfaceMeshFromBoundaryLabel(InterfaceId marker,
	                                           int boundary_label,
	                                           bool compute_orientation = true);
#endif

    [[nodiscard]] const dofs::DofHandler& dofHandler() const noexcept { return dof_handler_; }
    [[nodiscard]] const FieldRecord& fieldRecord(FieldId field) const;
    [[nodiscard]] const dofs::DofHandler& fieldDofHandler(FieldId field) const;
    [[nodiscard]] GlobalIndex fieldDofOffset(FieldId field) const;
    [[nodiscard]] const dofs::FieldDofMap& fieldMap() const noexcept { return field_map_; }
    [[nodiscard]] const dofs::BlockDofMap* blockMap() const noexcept { return block_map_.get(); }
    [[nodiscard]] const constraints::AffineConstraints& constraints() const noexcept { return affine_constraints_; }
    [[nodiscard]] FELayoutRevisionState feLayoutRevisionState() const noexcept { return fe_layout_revisions_; }
    [[nodiscard]] std::uint64_t spaceRevision() const noexcept { return fe_layout_revisions_.space; }
    [[nodiscard]] std::uint64_t dofLayoutRevision() const noexcept { return fe_layout_revisions_.dof_layout; }
    [[nodiscard]] std::uint64_t constraintLayoutRevision() const noexcept { return fe_layout_revisions_.constraint_layout; }
    [[nodiscard]] std::uint64_t blockLayoutRevision() const noexcept { return fe_layout_revisions_.block_layout; }
    [[nodiscard]] std::uint64_t systemLayoutRevision() const noexcept;
    [[nodiscard]] OperatorRevisionSnapshot operatorRevisionSnapshot() const noexcept;
    [[nodiscard]] OperatorInvalidationDecision operatorInvalidationDecision(
        const OperatorRevisionSnapshot& cached,
        bool allow_lagged_jacobian_on_geometry_change = false) const;
    [[nodiscard]] const sparsity::SparsityPattern& sparsity(const OperatorTag& op) const;
    [[nodiscard]] const sparsity::DistributedSparsityPattern* distributedSparsityIfAvailable(const OperatorTag& op) const noexcept;
    [[nodiscard]] std::shared_ptr<const backends::DofPermutation> dofPermutation() const noexcept { return dof_permutation_; }

    /**
     * @brief Revision counter for the stored sparsity patterns.
     *
     * Bumped whenever setup() rebuilds the patterns and whenever
     * rebuildConstraintState() re-augments them because the algebraic
     * constraint structure (slave/master topology) changed post-setup,
     * e.g. when an interface-tracking constraint reclassifies cut cells.
     * Solvers holding matrices allocated from these patterns must
     * reallocate when this revision changes; otherwise backends that
     * cannot grow their stored pattern (Eigen CSR) silently drop the
     * new constraint-fill entries during assembly.
     */
    [[nodiscard]] std::uint64_t sparsityPatternRevision() const noexcept { return sparsity_pattern_revision_; }

    [[nodiscard]] bool isSetup() const noexcept { return is_setup_; }
#if FE_HAS_MPI
    /**
     * @brief Communicator governing this system's collective operations.
     *
     * Once the DOF handler is finalized this is its communicator, including
     * during setup and after a failed or invalidated setup.  Before a DOF
     * layout exists, systems constructed from a native distributed mesh use
     * that mesh's communicator.  Systems without communicator metadata are
     * local to MPI_COMM_SELF.
     */
    [[nodiscard]] MPI_Comm activeMpiCommunicator() const noexcept;
#endif
    [[nodiscard]] int temporalOrder() const noexcept;
    [[nodiscard]] bool hasExplicitTimeDependency() const noexcept;
    [[nodiscard]] bool hasTimeDependentConstraints() const noexcept;
    [[nodiscard]] bool requiresTimeAdvancement() const noexcept;
    [[nodiscard]] bool isTransient() const noexcept { return temporalOrder() > 0; }
    [[nodiscard]] std::vector<FieldId> timeDerivativeFields(const OperatorTag& op) const;
    [[nodiscard]] std::vector<FieldId> timeDerivativeFields() const;

	    // ---- Parameter requirements (optional) ----
	    [[nodiscard]] const ParameterRegistry& parameterRegistry() const noexcept { return parameter_registry_; }

    // ---- Gauge / nullspace detection (optional) ----
    /**
     * @brief Access the GaugeRegistry, creating it on first call
     *
     * The GaugeRegistry is an optional component for automatic nullspace
     * detection and enforcement.  It is created lazily on first access.
     */
    [[nodiscard]] gauge::GaugeRegistry& gaugeRegistry();
    [[nodiscard]] const gauge::GaugeRegistry* gaugeRegistryIfPresent() const noexcept {
        return gauge_registry_.get();
    }
    [[nodiscard]] bool hasGaugeRegistry() const noexcept { return gauge_registry_ != nullptr; }

    // ---- Problem analysis subsystem ----

    void addFormulationRecord(analysis::FormulationRecord record);
    void addBoundaryConditionDescriptor(analysis::BoundaryConditionDescriptor desc);
    void addBoundaryConditionDescriptor(
        analysis::BoundaryConditionDescriptor desc,
        std::string_view operator_tag);
    void addContribution(analysis::ContributionDescriptor desc);
    void addVariableDescriptor(analysis::VariableDescriptor desc);
    void addInvariantDomainDescriptor(analysis::InvariantDomainDescriptor desc);
    void registerGeneratedEmbeddedInterfaceMarker(InterfaceId interface_marker);
    [[nodiscard]] bool isGeneratedEmbeddedInterfaceMarkerRegistered(
        InterfaceId interface_marker) const noexcept {
        return generated_embedded_interface_markers_.find(interface_marker) !=
               generated_embedded_interface_markers_.end();
    }
    void setAnalysisSolverOptions(backends::SolverOptions options);
    void clearAnalysisSolverOptions();
    void addAnalysisSummaries(analysis::AnalysisSummarySet summaries);
    [[nodiscard]] bool updateAnalysisSummariesFromAssembledOperator(
        const backends::GenericMatrix& matrix,
        const OperatorTag& op,
        const SystemStateView* state = nullptr);
    void clearAnalysisSummaries();
    [[nodiscard]] const analysis::AnalysisSummarySet* latestAnalysisSummaries() const noexcept {
        return analysis_summaries_ ? &*analysis_summaries_ : nullptr;
    }

    [[nodiscard]] const std::vector<analysis::FormulationRecord>& formulationRecords() const noexcept {
        return formulation_records_;
    }
    [[nodiscard]] const std::vector<analysis::BoundaryConditionDescriptor>& boundaryConditionDescriptors() const noexcept {
        return bc_descriptors_;
    }
    [[nodiscard]] std::span<const std::string>
    boundaryConditionDescriptorOperatorTags() const noexcept {
        return bc_descriptor_operator_tags_;
    }
    [[nodiscard]] const std::vector<analysis::VariableDescriptor>& variableDescriptors() const noexcept {
        return variable_descriptors_;
    }
    [[nodiscard]] const std::vector<analysis::ContributionDescriptor>& contributionDescriptors() const noexcept {
        return contributions_;
    }

    /// Build and store topology context from the mesh
    void buildTopologyContext();
    [[nodiscard]] const analysis::TopologyAnalysisContext* topologyContext() const noexcept {
        return topology_context_ ? &*topology_context_ : nullptr;
    }

    /// Build and store interface topology from registered InterfaceMesh objects
    void buildInterfaceTopologyContext();
    [[nodiscard]] const analysis::InterfaceTopologyContext* interfaceTopologyContext() const noexcept {
        return interface_topology_context_ ? &*interface_topology_context_ : nullptr;
    }

    /// Build and store constraint summary from current AffineConstraints
    void buildConstraintSummary();

    /**
     * @brief Re-augment stored sparsity patterns after a post-setup change in
     * constraint structure (slave/master topology). No-op when the structure
     * signature is unchanged. Bumps sparsityPatternRevision() on change.
     */
    void refreshSparsityForConstraintStructureChange();
    [[nodiscard]] static std::uint64_t computeConstraintStructureSignature(
        const constraints::AffineConstraints& constraints);
    [[nodiscard]] const analysis::ConstraintAnalysisSummary* constraintSummary() const noexcept {
        return constraint_summary_ ? &*constraint_summary_ : nullptr;
    }

    /// Invalidate cached analysis report (called automatically by mutation methods)
    void invalidateAnalysisCache() noexcept;

    /// Run all analysis passes and return a fresh report
    [[nodiscard]] analysis::ProblemAnalysisReport runProblemAnalysis() const;

    /// Cached version — re-runs only if inputs have changed
    [[nodiscard]] const analysis::ProblemAnalysisReport& analysisReport() const;

    // ---- Operator registry query (for tests and diagnostics) ----

    /**
     * @brief Query the registered operator definition for an operator tag
     *
     * Returns the OperatorDefinition containing all registered cell, boundary,
     * interior-face, interface-face, and global terms. This is the structural
     * view of what was installed — useful for parity tests that verify mixed
     * and manual installation paths produce identical block structure.
     */
    [[nodiscard]] const OperatorDefinition& operatorDefinition(const OperatorTag& op) const {
        return operator_registry_.get(op);
    }

    [[nodiscard]] bool hasOperator(const OperatorTag& op) const noexcept {
        return operator_registry_.has(op);
    }

    [[nodiscard]] std::size_t cutVolumeKernelCount(
        int interface_marker,
        geometry::CutIntegrationSide side) const;

    void addCutIntegrationContextUpdateCallback(
        CutIntegrationContextUpdateCallback hook);

    void beginCutIntegrationContextTransaction();
    void commitCutIntegrationContextTransaction();
    void rollbackCutIntegrationContextTransaction();
    [[nodiscard]] bool cutIntegrationContextTransactionActive() const noexcept {
        return cut_integration_context_transaction_backup_ != nullptr;
    }

    void setCutIntegrationContext(
        std::shared_ptr<const assembly::CutIntegrationContext> context);

    void clearCutIntegrationContext() {
        setCutIntegrationContext(nullptr);
    }

    [[nodiscard]] const assembly::CutIntegrationContext* cutIntegrationContext() const noexcept {
        return cut_integration_context_.get();
    }

    /**
     * @brief True when setup determined that an operator's matrix is state-independent.
     *
     * The decision is computed from installed kernel metadata during setup and is
     * conservative: false means "assemble normally"; true allows Newton/time
     * stepping infrastructure to reuse the matrix within an unchanged setup state.
     */
    [[nodiscard]] bool operatorMatrixStateIndependent(const OperatorTag& op) const;

    // ---- Rank-1 updates from coupled Jacobian assembly ----
    [[nodiscard]] std::span<const backends::RankOneUpdate> lastRankOneUpdates() const noexcept;
    void clearRankOneUpdates() noexcept;
    [[nodiscard]] std::span<const backends::ReducedFieldUpdate> lastReducedFieldUpdates() const noexcept;
    void clearReducedFieldUpdates() noexcept;
    [[nodiscard]] std::span<const Real> lastLocalCondensedRhsShift() const noexcept;
    [[nodiscard]] bool hasLocalCondensedRecovery() const noexcept
    {
        return !last_local_condensed_records_.empty();
    }
    void applyLocalCondensedRecovery(std::span<const Real> dense_du, Real alpha = Real(1.0));
    void clearLocalCondensedRecovery() noexcept;

    /// @cond INTERNAL
    // Internal — used by FormsInstaller for transactional kernel registration.
    // Public only because C++ templates cannot be called from non-friend TUs
    // when private. Do not call from physics modules.
    template <typename Fn>
    auto executeWithOperatorRollback_(Fn&& fn) -> decltype(fn()) {
        auto snap = operator_registry_.snapshot();
        auto field_registry_snapshot = field_registry_;
        auto generated_interface_marker_snapshot =
            generated_embedded_interface_markers_;
        const auto formulation_record_size =
            formulation_records_.size();
        auto contribution_snapshot =
            contributions_;
        const auto contribution_definition_count =
            contributions_def_count_;
        const auto auxiliary_output_consumer_size =
            auxiliary_output_consumers_.size();
        try {
            return fn();
        } catch (...) {
            operator_registry_.rollback(snap);
            field_registry_ =
                std::move(field_registry_snapshot);
            generated_embedded_interface_markers_ =
                std::move(
                    generated_interface_marker_snapshot);
            formulation_records_.resize(
                formulation_record_size);
            contributions_ =
                std::move(contribution_snapshot);
            contributions_def_count_ =
                contribution_definition_count;
            auxiliary_output_consumers_.resize(
                auxiliary_output_consumer_size);
            throw;
        }
    }
    /// @endcond

private:
    void emitAcceptedMeshNormalBoundaryConstraintHistory(
        std::size_t group_begin) const noexcept;
    void emitAcceptedMeshTangentialBoundaryPolicyHistory(
        std::size_t group_begin) const noexcept;
    [[nodiscard]] analysis::ProblemAnalysisContext buildProblemAnalysisContext() const;
    [[nodiscard]] analysis::ProblemAnalysisReport runProblemAnalysisPlanOnly() const;
    [[nodiscard]] std::unique_ptr<sparsity::SparsityPattern>
    buildActiveSparsityPatternFromBase(const sparsity::SparsityPattern& base) const;
    [[nodiscard]] std::unique_ptr<sparsity::DistributedSparsityPattern>
    buildActiveDistributedSparsityPatternFromBase(
        const sparsity::DistributedSparsityPattern& base,
        const sparsity::SparsityPattern* active_serial) const;

    struct PlannedCellTerm {
        FieldId test_field{INVALID_FIELD_ID};
        FieldId trial_field{INVALID_FIELD_ID};
        const spaces::FunctionSpace* test_space{nullptr};
        const spaces::FunctionSpace* trial_space{nullptr};
        assembly::AssemblyKernel* kernel{nullptr};
        const dofs::DofMap* row_dof_map{nullptr};
        const dofs::DofMap* col_dof_map{nullptr};
        GlobalIndex row_dof_offset{0};
        GlobalIndex col_dof_offset{0};
        std::string participant_scope{};
        assembly::SemanticKernelKind semantic_kind{assembly::SemanticKernelKind::SingleForm};
        bool matrix_capable{false};
        bool vector_capable{false};
    };

    struct LocalCondensedEntityRecord {
        std::string block_name{};
        std::size_t entity_index{0};
        std::size_t block_ordinal{0};
        std::uint64_t global_entity_key{0};
        bool has_aux_equation_terms{false};
        std::vector<std::vector<std::pair<GlobalIndex, Real>>> B_columns{};
        std::vector<std::vector<std::pair<GlobalIndex, Real>>> Ct_rows{};
        std::vector<Real> D_inv{};
        std::vector<Real> g{};
    };

    struct PlannedBoundaryTerm {
        int marker{0};
        FieldId test_field{INVALID_FIELD_ID};
        FieldId trial_field{INVALID_FIELD_ID};
        const spaces::FunctionSpace* test_space{nullptr};
        const spaces::FunctionSpace* trial_space{nullptr};
        assembly::AssemblyKernel* kernel{nullptr};
        const dofs::DofMap* row_dof_map{nullptr};
        const dofs::DofMap* col_dof_map{nullptr};
        GlobalIndex row_dof_offset{0};
        GlobalIndex col_dof_offset{0};
        std::string participant_scope{};
        bool matrix_capable{false};
        bool vector_capable{false};
    };

    struct PlannedInteriorFaceTerm {
        int marker{-1};
        FieldId test_field{INVALID_FIELD_ID};
        FieldId trial_field{INVALID_FIELD_ID};
        const spaces::FunctionSpace* test_space{nullptr};
        const spaces::FunctionSpace* trial_space{nullptr};
        assembly::AssemblyKernel* kernel{nullptr};
        const dofs::DofMap* row_dof_map{nullptr};
        const dofs::DofMap* col_dof_map{nullptr};
        GlobalIndex row_dof_offset{0};
        GlobalIndex col_dof_offset{0};
        bool matrix_capable{false};
        bool vector_capable{false};
    };

    struct PlannedInterfaceFaceTerm {
        int marker{0};
        FieldId test_field{INVALID_FIELD_ID};
        FieldId trial_field{INVALID_FIELD_ID};
        const spaces::FunctionSpace* test_space{nullptr};
        const spaces::FunctionSpace* trial_space{nullptr};
        assembly::AssemblyKernel* kernel{nullptr};
        const dofs::DofMap* row_dof_map{nullptr};
        const dofs::DofMap* col_dof_map{nullptr};
        GlobalIndex row_dof_offset{0};
        GlobalIndex col_dof_offset{0};
        bool matrix_capable{false};
        bool vector_capable{false};
    };

    struct PlannedCutVolumeTerm {
        int marker{0};
        geometry::CutIntegrationSide side{geometry::CutIntegrationSide::Negative};
        FieldId test_field{INVALID_FIELD_ID};
        FieldId trial_field{INVALID_FIELD_ID};
        std::string source_component_tag{};
        const spaces::FunctionSpace* test_space{nullptr};
        const spaces::FunctionSpace* trial_space{nullptr};
        assembly::AssemblyKernel* kernel{nullptr};
        const dofs::DofMap* row_dof_map{nullptr};
        const dofs::DofMap* col_dof_map{nullptr};
        GlobalIndex row_dof_offset{0};
        GlobalIndex col_dof_offset{0};
        bool matrix_capable{false};
        bool vector_capable{false};
    };

    struct OperatorAssemblyPlan {
        std::vector<PlannedCellTerm> cell_terms{};
        std::vector<PlannedBoundaryTerm> boundary_terms{};
        std::vector<PlannedInteriorFaceTerm> interior_terms{};
        std::vector<PlannedInterfaceFaceTerm> interface_terms{};
        std::vector<PlannedCutVolumeTerm> cut_volume_terms{};
        std::vector<GlobalKernel*> global_terms{};
        bool matrix_state_independent{false};
    };

    friend assembly::AssemblyResult assembleOperator(
        FESystem& system,
        const AssemblyRequest& request,
        const SystemStateView& state,
        assembly::GlobalSystemView* matrix_out,
        assembly::GlobalSystemView* vector_out);
    friend CoupledResidualKernels installFormulation(
        FESystem& system,
        const OperatorTag& op,
        std::span<const FieldId> fields,
        const forms::FormExpr& residual,
        const FormInstallOptions& options);
    friend std::vector<std::vector<
        std::shared_ptr<assembly::AssemblyKernel>>>
    installMixedFormIR(
        FESystem& system,
        const OperatorTag& op,
        std::span<const FieldId> test_fields,
        std::span<const FieldId> trial_fields,
        const forms::MixedFormIR& mixed_ir,
        const FormInstallOptions& options);
    friend class BoundaryReductionService;
    friend class OperatorBackends;

    void prepareFormBoundExteriorBoundaryMeasurePolicies(
        const std::vector<ExteriorBoundaryMeasurePolicy>& policies);
    void commitPreparedFormBoundExteriorBoundaryMeasurePolicies(
        std::vector<ExteriorBoundaryMeasurePolicy> policies) noexcept;
    void validateExteriorBoundaryMeasurePoliciesAgainstCutContext(
        std::span<const ExteriorBoundaryMeasurePolicy> policies,
        const assembly::CutIntegrationContext* context,
        bool require_generated_active_context) const;
    void requireCurrentExteriorBoundaryMeasurePolicies(
        const OperatorTag& op) const;
    void requireCurrentBoundaryReductionExteriorMeasures(
        bool use_dof_handler_communicator = false) const;
    void requireBoundaryReductionExteriorMeasure(
        const forms::BoundaryFunctional& functional) const;
    void requireConsistentCutIntegrationContextCandidate(
        const assembly::CutIntegrationContext* context,
        bool use_dof_handler_communicator = false) const;
    void runCutIntegrationContextUpdateCallbacks(
        const assembly::CutIntegrationContext* context,
        bool use_dof_handler_communicator = false);
    void prepareFormBoundGeneratedBoundaryNitscheTracePolicies(
        const std::vector<
            GeneratedBoundaryNitscheTracePolicy>& policies);
    void commitPreparedFormBoundGeneratedBoundaryNitscheTracePolicies(
        std::vector<
            GeneratedBoundaryNitscheTracePolicy> policies) noexcept;
    void invalidateSetup() noexcept;
    void publishFinalizedSmallCutAggregationProlongations();
    void invalidateGeneratedBoundaryNitscheTraceCertificates() noexcept;
    void refreshGeneratedBoundaryNitscheTraceCertificates(
        bool allow_missing_cut_context);
    void requireCurrentGeneratedBoundaryNitscheTraceCertificates(
        const OperatorTag& op) const;
    void requireSetup() const;
    void requireSingleFieldSetup() const;
    void buildAssemblyPlans();
    void bumpSpaceRevision() noexcept { ++fe_layout_revisions_.space; }
    void bumpDofLayoutRevision() noexcept { ++fe_layout_revisions_.dof_layout; }
    void bumpConstraintLayoutRevision() noexcept { ++fe_layout_revisions_.constraint_layout; }
    void bumpBlockLayoutRevision() noexcept { ++fe_layout_revisions_.block_layout; }
    [[nodiscard]] constraints::ConstraintRevisionSnapshot captureConstraintRevisionSnapshot(
        bool include_mesh_field_values = false) const noexcept;

    [[nodiscard]] const FieldRecord& singleField() const;

    std::shared_ptr<const assembly::IMeshAccess> mesh_access_;
    std::shared_ptr<const ISearchAccess> search_access_{};
    std::vector<MeshParticipantInfo> mesh_participants_{};

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
	    std::shared_ptr<const svmp::Mesh> mesh_{};
	    svmp::Configuration coord_cfg_{svmp::Configuration::Reference};
	    std::unordered_map<InterfaceId, std::shared_ptr<const svmp::InterfaceMesh>> interface_meshes_{};
	    std::optional<svmp::motion::MotionCoordinateBackup> mesh_coordinate_backup_{};
	    struct MeshMotionFieldBackup {
	        svmp::FieldHandle handle{};
	        std::string name{};
	        std::vector<svmp::real_t> values{};
	        std::size_t components{0};
	        std::size_t entity_count{0};
	    };
	    std::vector<MeshMotionFieldBackup> mesh_motion_field_backup_{};
	    GeometryTransactionState geometry_transaction_state_{GeometryTransactionState::Committed};
	    OperatorRevisionSnapshot geometry_transaction_start_revision_{};
	    OperatorRevisionSnapshot geometry_transaction_last_revision_{};
	    std::string geometry_transaction_last_event_{};
	    std::vector<GeometryTransactionCallback> geometry_transaction_callbacks_{};
#endif

    FieldRegistry field_registry_;
    OperatorRegistry operator_registry_;
    std::vector<FormCellDomainRestriction> form_install_cell_domain_restrictions_{};
    std::shared_ptr<const assembly::CutIntegrationContext> cut_integration_context_{};
    std::unordered_map<int, std::string>
        generated_active_boundary_owner_bindings_{};
    struct CutIntegrationContextTransactionBackup {
        std::shared_ptr<const assembly::CutIntegrationContext> context{};
        std::shared_ptr<const assembly::CutIntegrationContext>
            context_content_snapshot{};
        std::uint64_t context_content_revision{0u};
        bool rollback_started{false};
        std::unordered_map<int, std::string>
            generated_active_boundary_owner_bindings{};
        constraints::AffineConstraints affine_constraints{};
        FELayoutRevisionState fe_layout_revisions{};
        constraints::ConstraintRevisionSnapshot constraint_revision_snapshot{};
        std::uint64_t constraint_structure_signature{0};
        std::uint64_t sparsity_pattern_revision{0};
        std::optional<analysis::ConstraintAnalysisSummary> constraint_summary{};
        std::optional<analysis::ProblemAnalysisReport> analysis_report_cache{};
        std::uint64_t analysis_inputs_version{0};
        std::uint64_t analysis_report_version{0};
        std::vector<std::shared_ptr<
            const constraints::SmallCutAggregationProlongationReport>>
            finalized_small_cut_aggregation_prolongations{};
        std::vector<GeneratedBoundaryNitscheTraceCertificateRecord>
            generated_boundary_nitsche_trace_certificates{};
        std::vector<std::shared_ptr<
            const constraints::SmallCutAggregationConstraint::
                LifecycleCheckpoint>>
            small_cut_aggregation_lifecycle_checkpoints{};
    };
    std::unique_ptr<CutIntegrationContextTransactionBackup>
        cut_integration_context_transaction_backup_{};
    std::vector<CutIntegrationContextUpdateCallback>
        cut_integration_context_update_callbacks_{};
    mutable bool
        cut_integration_context_callback_dispatch_active_{false};
    std::set<InterfaceId> generated_embedded_interface_markers_{};
    std::vector<std::unique_ptr<constraints::Constraint>> constraint_defs_;
    std::vector<std::unique_ptr<constraints::ISystemConstraint>> system_constraint_defs_;

    dofs::DofHandler dof_handler_{};
    std::vector<dofs::DofHandler> field_dof_handlers_{};
    std::vector<GlobalIndex> field_dof_offsets_{};
    struct PrescribedFieldBuffer {
        std::vector<Real> coefficients{};
        std::uint64_t revision{0};
    };
    std::vector<PrescribedFieldBuffer> prescribed_field_buffers_{};
    SetupStoragePlan setup_storage_plan_{};
    dofs::FieldDofMap field_map_{};
    std::unique_ptr<dofs::BlockDofMap> block_map_{};
	    constraints::AffineConstraints affine_constraints_{};
    FELayoutRevisionState fe_layout_revisions_{};
    SetupOptions last_setup_options_{};
    SetupInputs last_setup_inputs_{};
    bool has_last_setup_{false};
    constraints::ConstraintRevisionSnapshot constraint_revision_snapshot_{};
    std::uint64_t constraint_time_epoch_{0};
    bool has_last_constraint_update_time_{false};
    double last_constraint_update_time_{0.0};
    double last_constraint_update_dt_{0.0};
    assembly::MeshMotionFieldAccess mesh_motion_fields_{};
    std::vector<MeshNormalBoundaryConstraintDeclaration>
        mesh_normal_boundary_constraints_{};
    std::vector<MeshNormalBoundaryConstraintHistoryRecord>
        mesh_normal_boundary_constraint_history_{};
    std::vector<FittedALENormalMeasurementDeclaration>
        fitted_ale_normal_measurement_declarations_{};
    std::vector<FittedALENormalOperatorStageHistoryRecord>
        pending_fitted_ale_normal_measurements_{};
    std::vector<FittedALENormalOperatorStageHistoryRecord>
        fitted_ale_normal_measurement_history_{};
    bool fitted_ale_normal_measurement_declarations_frozen_{false};
    bool fitted_ale_normal_measurement_transaction_active_{false};
    std::vector<MeshTangentialBoundaryPolicyDeclaration>
        mesh_tangential_boundary_policies_{};
    std::vector<MeshTangentialBoundaryPolicyHistoryRecord>
        mesh_tangential_boundary_policy_history_{};
    bool mesh_boundary_history_transaction_active_{false};
    bool mesh_boundary_history_collective_call_active_{false};
    bool mesh_boundary_history_defer_logging_{false};
    std::vector<FreeSurfaceDiscreteFunctionalDeclaration>
        free_surface_discrete_functional_declarations_{};
    std::vector<FreeSurfaceDiscreteFunctionalHistoryRecord>
        free_surface_discrete_functional_history_{};
    GeometricNonlinearityPolicy geometric_nonlinearity_policy_{};

		    std::unordered_map<OperatorTag, std::unique_ptr<sparsity::SparsityPattern>> sparsity_by_op_{};
		    std::unordered_map<OperatorTag, std::unique_ptr<sparsity::DistributedSparsityPattern>> distributed_sparsity_by_op_{};
		    std::unordered_map<OperatorTag, std::unique_ptr<sparsity::SparsityPattern>> base_sparsity_by_op_{};
		    std::unordered_map<OperatorTag, std::unique_ptr<sparsity::DistributedSparsityPattern>> base_distributed_sparsity_by_op_{};
		    std::shared_ptr<const backends::DofPermutation> dof_permutation_{};
		    int distributed_sparsity_dof_per_node_{0};
		    std::uint64_t constraint_structure_signature_{0};
		    std::uint64_t sparsity_pattern_revision_{0};

	    std::unique_ptr<assembly::Assembler> assembler_{};
        bool use_constraints_in_assembly_{true};
        bool use_backend_row_ownership_for_assembly_{false};
			    std::string assembler_selection_report_{};
		    std::unique_ptr<assembly::IMaterialStateProvider> material_state_provider_{};
	    std::unique_ptr<GlobalKernelStateProvider> global_kernel_state_provider_{};
    std::unique_ptr<OperatorBackends> operator_backends_{};
    std::unordered_map<FieldId, std::unique_ptr<BoundaryReductionService>> boundary_reduction_services_{};
    std::unique_ptr<AuxiliaryStateManager> auxiliary_state_manager_{};
    std::unique_ptr<AuxiliaryOperatorRegistry> auxiliary_operator_registry_{};
    std::unique_ptr<AuxiliaryInputRegistry> auxiliary_input_registry_{};
    std::unique_ptr<FEQuantityRegistry> fe_quantity_registry_{};
    std::unique_ptr<post::DerivedResultRegistry> derived_result_registry_{};

    /// Cached system state for FE-coupled auxiliary input callbacks.
    /// Set by cacheSystemState() which is called from prepareAuxiliaryForAssembly(),
    /// assembleMixedAuxiliaryIntoGlobal(), and advanceAuxiliaryState().
    /// Callbacks capture `this` and read from these members.
    mutable std::span<const Real> cached_solution_u_{};
    mutable const backends::GenericVector* cached_solution_vector_{nullptr};
    mutable std::span<const Real> cached_solution_u_prev_{};
    mutable const backends::GenericVector* cached_solution_prev_vector_{nullptr};
    mutable std::span<const Real> cached_solution_u_prev2_{};
    mutable const backends::GenericVector* cached_solution_prev2_vector_{nullptr};
    mutable const assembly::TimeIntegrationContext* cached_time_integration_{nullptr};
    mutable const void* cached_user_data_{nullptr};

    /// Cache a SystemStateView's fields for auxiliary input callbacks.
    void cacheSystemState(const SystemStateView& state) const;

    /**
     * @brief Freeze, certify, and execute the due auxiliary input callbacks.
     *
     * Distributed callers agree on registry presence and the complete ordered
     * callback schedule before any provider runs.  Provider failures are then
     * coordinated after every frozen input so no peer can enter a later
     * collective route after another rank has already failed.
     */
    void evaluateAuxiliaryInputsCollectively_(
        Real time,
        Real dt,
        bool is_nonlinear_iteration,
        std::string_view phase);

    mutable std::vector<Real> field_endpoint_scratch_src_{}; ///< Scratch for distributed field source endpoint.
    mutable std::vector<Real> field_endpoint_scratch_tgt_{}; ///< Scratch for distributed field target endpoint.

    // Deployed auxiliary model instances (collected before setup, consumed during finalize).
    struct DeployedAuxEntry {
        std::shared_ptr<AuxiliaryStateModel> model{};
        std::string instance_name{};
        AuxiliaryStateSpec spec{};
        AuxiliaryStepperSpec stepper_spec{};
        std::vector<Real> initial_values{};
        std::map<std::string, std::string> input_bindings{}; ///< Ordered for deterministic iteration
        std::unordered_map<std::string, AuxiliaryInputHandle> coupled_bindings{}; ///< For chain-rule coupling
        std::unordered_map<std::string, Real> param_values{};
        std::vector<AuxiliaryConstraintBinding> constraint_bindings{};
        std::optional<AuxiliaryBlockSolverMetadata> solver_metadata{};
        std::size_t explicit_entity_count{0}; ///< 0 = auto from scope/mesh
        std::vector<std::uint32_t> output_ids{};
        AuxiliaryDeployedInstance::RaggedEntitySizeProvider ragged_entity_size_provider{};
        std::vector<std::size_t> ragged_component_offsets{};
        FieldId quadrature_reference_field{INVALID_FIELD_ID};
        std::string quadrature_reference_operator{};
        std::string variant_group{};
        std::string variant_key{};
        AuxiliaryActivationMode activation_mode{AuxiliaryActivationMode::Auto};
        std::unique_ptr<AuxiliaryStateStepper> stepper{};
        std::unique_ptr<AuxiliaryDerivativeProvider> deriv_provider{};
        std::vector<std::unique_ptr<AuxiliaryEventManager>> event_managers{};
        std::vector<Real> output_buffer{}; ///< Evaluated output values
        bool lower_to_direct_only{false};  ///< Keep semantic block/output, but exclude from live monolithic solve.
        bool local_condensed{false};       ///< Eliminate locally into reduced field updates instead of dense bordered layout.
        bool selected{true};
        bool materialized{false};
        bool consistent_initialization_done{false};
        /// Entity map: indices of entities this block covers.
        /// Empty = all entities (WholeDomain / no restriction).
        std::vector<std::size_t> entity_map{};
        std::vector<std::size_t> qp_offsets{};
    };
    struct AuxiliaryScopeResolution {
        std::size_t entity_count{0};
        std::size_t owned_entity_count{0};
    };
    struct AuxiliaryRegionLookupCache {
        std::vector<std::size_t> region_ids{};
        std::vector<std::vector<std::size_t>> region_to_cells{};
        std::vector<std::vector<std::size_t>> region_to_nodes{};
        std::vector<std::vector<int>> region_to_boundary_markers{};
        std::vector<std::vector<std::size_t>> region_to_interface_faces{};
        std::vector<int> region_owner_ranks{};
    };
    std::vector<DeployedAuxEntry> deployed_aux_entries_{};
    std::unordered_map<std::string, forms::FormExpr> lowered_aux_output_exprs_by_name_{};
    std::unordered_map<std::size_t, forms::FormExpr> lowered_aux_output_exprs_by_id_{};
    std::vector<AuxiliaryOutputDescriptor> auxiliary_output_descriptors_{};
    std::unordered_map<std::string, std::uint32_t> auxiliary_output_id_by_qualified_name_{};
    std::unordered_map<std::string, std::string> auxiliary_variant_selection_{};
    std::vector<analysis::AuxiliaryOutputConsumerRecord> auxiliary_output_consumers_{};
    std::size_t lowered_auxiliary_constraint_offset_{
        std::numeric_limits<std::size_t>::max()};
    [[nodiscard]] bool canLowerAlgebraicAuxiliaryToDirectOnly_(const DeployedAuxEntry& entry) const;
    [[nodiscard]] std::optional<forms::FormExpr>
    synthesizeLoweredAuxiliaryOutputExpr_(const DeployedAuxEntry& entry,
                                          std::string_view output_name) const;
    void buildAuxiliaryOutputBindings_();
    void ensureAuxiliaryRegionLookupCache_();
    [[nodiscard]] std::size_t auxiliaryTopologyRegionInputEntityCount_() const;
    [[nodiscard]] std::vector<GlobalIndex>
    auxiliaryTopologyRegionCells_(std::size_t region_id) const;
    [[nodiscard]] AuxiliaryEntityRemapMetadata
    buildAuxiliaryEntityRemapMetadata_(const DeployedAuxEntry& entry,
                                       const AuxiliaryScopeResolution& resolution);
    [[nodiscard]] std::vector<std::size_t>
    buildAuxiliaryRaggedComponentOffsets_(const DeployedAuxEntry& entry,
                                          const AuxiliaryScopeResolution& resolution) const;
    [[nodiscard]] std::vector<Real>
    buildAuxiliaryRaggedInitialValues_(const DeployedAuxEntry& entry,
                                       std::span<const std::size_t> component_offsets) const;
    void validateEntityLocalAuxiliaryBindings_() const;
    [[nodiscard]] std::vector<int>
    buildAuxiliaryRegionRowOwnerRanks_(const DeployedAuxEntry& entry,
                                       std::size_t entity_count);
    [[nodiscard]] int nodeAuxiliaryOwnerRank_(std::size_t node_id) const;
    [[nodiscard]] std::vector<int>
    buildAuxiliaryNodeRowOwnerRanks_(const DeployedAuxEntry& entry,
                                     std::size_t entity_count) const;
    /// Deferred dependency pairs (dependent, dependency) from derivedInput().
    /// Wired by finalizeDeferredInputDeps() when all inputs are registered.
    std::vector<std::pair<std::string, std::string>> deferred_input_deps_{};
    /// Deferred derived-input expressions needing AuxiliaryInputSymbol→Ref resolution.
    std::vector<std::pair<std::string, std::shared_ptr<forms::FormExpr>>> deferred_derived_exprs_{};
    /// Resolve deferred derived-input expressions and wire dependency edges.
    /// Safe to call multiple times — clears the deferred lists on first run.
    void finalizeDeferredInputDeps();
    void buildLoweredAuxiliaryOutputExpressions_();
    [[nodiscard]] bool isAuxiliaryDeploymentSelected_(const DeployedAuxEntry& entry) const;
    [[nodiscard]] bool isAuxiliaryDeploymentVisibleForBareLookup_(
        const DeployedAuxEntry& entry) const;
    [[nodiscard]] bool hasAuxiliaryConsumers_(
        const DeployedAuxEntry& entry) const;
    [[nodiscard]] bool hasCellVolumeAuxiliaryConsumers_(
        const DeployedAuxEntry& entry) const;
    [[nodiscard]] std::vector<analysis::AuxiliaryOutputConsumerRecord>
    consumersOfEntry_(const DeployedAuxEntry& entry) const;
    [[nodiscard]] std::vector<std::size_t> collectCoveredCells_(
        const DeployedAuxEntry& entry) const;
    [[nodiscard]] AuxiliaryScopeResolution
    resolveAuxiliaryDeploymentScope_(DeployedAuxEntry& entry);
    void validateMonolithicAuxiliaryLifecycle_(const DeployedAuxEntry& entry) const;
    void validateRaggedMonolithicLocalCondensationEligibility_(
        const DeployedAuxEntry& entry) const;
    void validateAuxiliaryMixedLayoutContract_() const;
    void inferQuadraturePointLayout_(DeployedAuxEntry& entry);
    void assignAuxiliaryOutputIds_(DeployedAuxEntry& entry);
    AuxiliaryInputHandle registerBoundaryIntegralHandle_(
        const std::string& input_name,
        forms::FormExpr integrand,
        int boundary_marker,
        forms::BoundaryFunctional::Reduction reduction,
        AuxiliaryInputUpdateSchedule schedule);
    AuxiliaryInputHandle registerBoundaryIntegralHandle_(
        const std::string& input_name,
        forms::BoundaryFunctional functional,
        AuxiliaryInputUpdateSchedule schedule);
    [[nodiscard]] std::string generateUniqueAuxiliaryInputName_(std::string_view prefix);
    [[nodiscard]] std::string makeScopeAwareInstanceBaseName_(const AuxiliaryDeployedInstance& instance) const;
    [[nodiscard]] std::string resolveDeploymentInstanceName_(const AuxiliaryDeployedInstance& instance) const;
    [[nodiscard]] bool hasDeployedInstanceName_(std::string_view instance_name) const;
    std::size_t generated_boundary_input_counter_{0};

    /// Bind secondary fields and set dof_per_node on a BoundaryReductionService
    /// for multi-field integrand evaluation.
    void bindSecondaryFields(BoundaryReductionService& svc,
                              FieldId primary_fid,
                              const std::vector<FieldId>& referenced_fields);
    [[nodiscard]] const MeshNormalBoundaryConstraintDeclaration&
    validatedFittedALENormalConstraint_(
        FieldId mesh_displacement_field,
        int boundary_marker) const;
    std::unique_ptr<AuxiliaryMultirateScheduler> aux_scheduler_{};

public:
    /// Assemble boundary gradient dI/du for a functional with the given
    /// integrand (already transformed: DiscreteField → TrialFunction).
    /// Returns sparse (DOF, value) pairs.
    std::vector<BoundaryReductionService::SensitivityEntry>
    assembleBoundaryGradient(FieldId field,
                              const forms::FormExpr& integrand_trial,
                              int boundary_marker,
                              const SystemStateView& state,
                              bool apply_constraints = true,
                              int region_marker = -1,
                              std::span<const GlobalIndex> cell_filter = {},
                              std::optional<int> generated_active_boundary_marker =
                                  std::nullopt,
                              bool explicit_cell_filter = false);

private:
    void advanceOneEntry(DeployedAuxEntry& entry, Real time, Real dt, int substep_count);

    /// Run one-time DAE consistent initialization for materialized auxiliary blocks.
    void initializeAuxiliaryDAEBlocksIfNeeded_(Real time, Real dt);

    /// Build ordered parameter vector for a deployed entry.
    [[nodiscard]] std::vector<Real> buildParamVector(const DeployedAuxEntry& entry) const;

    /// Build ordered input vector for a deployed entry (non-entity-local).
    [[nodiscard]] std::vector<Real> buildInputVector(const DeployedAuxEntry& entry) const;
    void lowerAuxiliaryConstraintBindings_();
    [[nodiscard]] const DeployedAuxEntry& findDeployedAuxEntry_(
        std::string_view instance_name) const;
    [[nodiscard]] std::vector<std::span<const Real>> buildHistorySpans_(
        const AuxiliaryBlockStorage& blk,
        std::size_t entity_index,
        std::vector<std::vector<Real>>& storage) const;
    void validatePartitionedAuxiliaryEntityWidth_(
        const char* caller,
        const DeployedAuxEntry& entry,
        const AuxiliaryBlockStorage& blk,
        std::size_t entity_index,
        std::size_t actual_width,
        const char* slice_name) const;

    /// Rebuild input vector for a generic (non-built) model at a specific entity,
    /// using declared input names with name:size parsing.
    void rebuildGenericInputsForEntity(
        const DeployedAuxEntry& entry, std::size_t entity_index,
        std::vector<Real>& out) const;

    /// Ensure monolithic auxiliary committed-rate buffers exist and are seeded
    /// for first-order generalized-alpha stage assembly.
    void ensureMonolithicCommittedRates(const SystemStateView& state);

    /// Seed one monolithic block's committed-rate buffer from the committed
    /// state and previous-step FE inputs.
    void initializeMonolithicCommittedRate(const DeployedAuxEntry& entry,
                                           const SystemStateView& prev_state);

    /// Apply smooth event/reset hooks after an accepted monolithic step.
    void applyMonolithicAcceptedStepEvents_(Real step_start_time,
                                            Real dt,
                                            Real gamma);

    /// Update monolithic committed-rate buffers from accepted final states.
    void updateMonolithicFinalRates_(Real gamma, Real dt);

    /// Ensure a flat committed-rate buffer exists for the given block.
    void ensureMonolithicCommittedRateBuffer(const DeployedAuxEntry& entry,
                                             std::size_t storage_size);

    /// Gather a per-entity committed-rate vector from the flat block buffer.
    [[nodiscard]] std::vector<Real> gatherMonolithicCommittedRate(
        const DeployedAuxEntry& entry,
        std::size_t entity_index) const;

    /// Scatter a per-entity committed-rate vector into the flat block buffer.
    void scatterMonolithicCommittedRate(const DeployedAuxEntry& entry,
                                        std::size_t entity_index,
                                        std::span<const Real> values);

    /// Wire FE-coupled auxiliary input providers during finalization.
    void wireFECoupledInputProviders();

    /// Assemble monolithic auxiliary contributions into a global system view.
    /// @param n_field_dofs  Number of FE field DOFs (for mixed offset computation).
    /// @param is_nonlinear_iteration  When true, EachNonlinearIteration inputs refresh.
    void assembleMixedAuxiliaryIntoGlobal(
        const SystemStateView& state,
        assembly::GlobalSystemView* matrix_out,
        assembly::GlobalSystemView* vector_out,
        bool want_matrix, bool want_vector,
        std::size_t n_field_dofs,
        bool is_nonlinear_iteration = false);

    /// Parse "name:size" suffix from a declared input name.
    /// Returns (base_name, component_count).
    static std::pair<std::string, int> parseDeclaredInputName(const std::string& raw);

    /// Validate all declared input names at deployment time (catches malformed suffixes).
    static void validateDeclaredInputNames(const AuxiliaryStateModel& model);

	    ParameterRegistry parameter_registry_{};
    std::unique_ptr<gauge::GaugeRegistry> gauge_registry_{};
    std::vector<backends::RankOneUpdate> last_rank_one_updates_{};
    std::vector<backends::ReducedFieldUpdate> last_reduced_field_updates_{};
    std::vector<LocalCondensedEntityRecord> last_local_condensed_records_{};
    std::vector<Real> last_local_condensed_rhs_shift_{};
    std::unordered_map<OperatorTag, OperatorAssemblyPlan> assembly_plan_by_op_{};

    // Cached coupled Jacobian results.
    // For time-invariant sensitivities (e.g., RCR/resistance BCs), keyed by dt only.
    // For time-variant sensitivities, keyed by (time, dt) pair.
    // Time-invariance is auto-detected on the first computation by walking the
    // FormExpr trees for solution/time references.
    struct CoupledJacobianCache {
        double time{-1e30};
        double dt{-1e30};
        bool valid{false};
        bool is_time_invariant{false};  ///< True if sensitivity depends only on geometry + dt
        std::vector<backends::RankOneUpdate> rank_one_updates{};
        // For non-symmetric cases: cached outer-product matrix entries.
        struct SparseEntry {
            GlobalIndex row;
            std::vector<GlobalIndex> col_dofs;
            std::vector<Real> col_vals;
        };
        std::vector<SparseEntry> outer_product_entries{};

        void clear() noexcept {
            time = -1e30;
            dt = -1e30;
            valid = false;
            is_time_invariant = false;
            rank_one_updates.clear();
            outer_product_entries.clear();
        }
    };
    CoupledJacobianCache coupled_jac_cache_{};
    std::unordered_map<std::string, std::vector<Real>> monolithic_aux_committed_rates_{};
    std::unordered_set<std::string> monolithic_aux_committed_rates_valid_{};

    // ---- Analysis subsystem storage ----
    std::vector<analysis::FormulationRecord> formulation_records_;
    std::vector<analysis::ContributionDescriptor> contributions_;
    std::size_t contributions_def_count_{0}; ///< Watermark for definition-phase contributions
	    std::vector<analysis::BoundaryConditionDescriptor> bc_descriptors_;
        std::vector<std::string> bc_descriptor_operator_tags_{};
    std::vector<analysis::VariableDescriptor> variable_descriptors_;
    std::vector<analysis::InvariantDomainDescriptor> invariant_domain_descriptors_;

    std::optional<analysis::TopologyAnalysisContext> topology_context_;
    std::optional<analysis::InterfaceTopologyContext> interface_topology_context_;
    std::optional<AuxiliaryRegionLookupCache> auxiliary_region_lookup_cache_;
    std::optional<analysis::ConstraintAnalysisSummary> constraint_summary_;
    std::optional<backends::SolverOptions> analysis_solver_options_;
    std::optional<analysis::AnalysisSummarySet> registered_analysis_summaries_;
    std::optional<analysis::AnalysisSummarySet> analysis_summaries_;
    bool assembled_tangent_analysis_summary_attempted_{false};
    mutable std::optional<analysis::ProblemAnalysisReport> analysis_report_cache_;
    mutable std::uint64_t analysis_inputs_version_{0};
    mutable std::uint64_t analysis_report_version_{std::numeric_limits<std::uint64_t>::max()};

	    bool is_setup_{false};
	    Real last_auxiliary_advance_time_{0.0};
        bool partitioned_auxiliary_advance_valid_{false};
        Real partitioned_auxiliary_advance_time_{std::numeric_limits<Real>::quiet_NaN()};
        Real partitioned_auxiliary_advance_dt_{std::numeric_limits<Real>::quiet_NaN()};
    mutable std::vector<Real> aux_state_flat_{}; ///< Flattened work-state values for assembly
    mutable std::vector<Real> aux_output_flat_{}; ///< Flattened output values for assembly
    std::vector<assembly::AuxiliaryOutputBinding> auxiliary_output_bindings_{};

public:
    /// Bordered coupling data for monolithic auxiliary DOFs.
    /// Populated by assembleMixedAuxiliaryIntoGlobal when monolithic blocks exist.
    /// Consumed by the Newton solver to apply a bordered system correction
    /// after the PDE linear solve.
    struct BorderedCouplingData {
        bool active{false};             ///< True if monolithic aux DOFs exist
        bool globally_reduced{false};   ///< True once dense bordered blocks have been summed for replicated MPI use
        bool aux_self_terms_replicated{false}; ///< True when D/g/dF_dxdot are already identical on every rank
        int n_aux{0};                   ///< Number of auxiliary unknowns
        std::size_t n_field_dofs{0};    ///< Number of PDE DOFs
        std::vector<Real> D;            ///< Aux-aux Jacobian (n_aux × n_aux, row-major)
        std::vector<Real> g;            ///< Auxiliary residual (n_aux)
        std::vector<Real> B;            ///< dR_PDE/dx_aux columns (n_field_dofs × n_aux, col-major)
        std::vector<Real> Ct;           ///< dR_aux/du rows (n_aux × n_field_dofs, row-major)
        std::vector<Real> dF_dxdot;     ///< Raw dF/dxdot block (n_aux × n_aux, row-major)
        std::vector<AuxiliaryVariableKind> aux_variable_kinds{}; ///< Per-aux unknown classification in mixed-storage order
        std::vector<int> aux_row_owner_ranks{}; ///< Owner rank per bordered auxiliary row when owner-routed.
        std::vector<char> aux_row_owner_routed{}; ///< Nonzero when a row must be inserted only by its owner.
        std::vector<int> aux_row_local_contribution_flags{}; ///< 1 if this rank contributed the owner-routed row.
        std::vector<int> aux_row_global_contributor_counts{}; ///< MPI sum of local contribution flags.

        void clear() {
            active = false;
            globally_reduced = false;
            aux_self_terms_replicated = false;
            n_aux = 0;
            n_field_dofs = 0;
            D.clear(); g.clear(); B.clear(); Ct.clear();
            dF_dxdot.clear();
            aux_variable_kinds.clear();
            aux_row_owner_ranks.clear();
            aux_row_owner_routed.clear();
            aux_row_local_contribution_flags.clear();
            aux_row_global_contributor_counts.clear();
            aux_blocks.clear();
            dF_dinputs.clear();
            dO_dx.clear();
            dO_dI.clear();
            direct_coupling_records.clear();
        }
        /// Per-block info for auxiliary state update after solve.
        struct AuxBlock { std::string name; int dim; };
        std::vector<AuxBlock> aux_blocks;

        struct DirectCouplingRecord {
            std::size_t output_slot{static_cast<std::size_t>(-1)};
            std::size_t entity_index{0};
            std::vector<std::size_t> aux_local_indices{};
            std::vector<Real> dF_dinputs{};
            std::vector<Real> dO_dx{};
            std::vector<Real> dO_dI{};
            std::vector<std::vector<std::pair<GlobalIndex, Real>>> input_gradients{};
            std::vector<std::pair<GlobalIndex, Real>> output_gradient{};
        };

        /// dF/d(inputs) per aux DOF (needed for B computation from Ct).
        std::vector<Real> dF_dinputs;  ///< (n_aux × n_inputs_per_block)
        /// d(output)/d(state) per aux DOF.
        std::vector<Real> dO_dx;       ///< (n_outputs × n_aux)
        std::vector<Real> dO_dI;       ///< d(output)/d(input), e.g., Rp for RCR
        /// Per-output/entity direct-coupling metadata for debugging and
        /// verification. Unlike the flat compatibility vectors above, these do
        /// not collapse multiple outputs or boundary entities into one record.
        std::vector<DirectCouplingRecord> direct_coupling_records{};

        void resize(int na, std::size_t nf) {
            n_aux = na;
            n_field_dofs = nf;
            globally_reduced = false;
            aux_self_terms_replicated = false;
            D.assign(static_cast<std::size_t>(na * na), 0.0);
            g.assign(static_cast<std::size_t>(na), 0.0);
            B.assign(nf * static_cast<std::size_t>(na), 0.0);
            Ct.assign(static_cast<std::size_t>(na) * nf, 0.0);
            dF_dxdot.assign(static_cast<std::size_t>(na * na), 0.0);
            aux_variable_kinds.assign(static_cast<std::size_t>(na),
                                      AuxiliaryVariableKind::Differential);
            aux_row_owner_ranks.assign(static_cast<std::size_t>(na), -1);
            aux_row_owner_routed.assign(static_cast<std::size_t>(na), char{0});
            aux_row_local_contribution_flags.assign(static_cast<std::size_t>(na), 0);
            aux_row_global_contributor_counts.assign(static_cast<std::size_t>(na), 0);
            dF_dinputs.clear();
            dO_dx.clear();
            dO_dI.clear();
            direct_coupling_records.clear();
            active = true;
        }
    };

    /// Access bordered coupling data (populated during assembly).
    [[nodiscard]] BorderedCouplingData& borderedCoupling() noexcept { return bordered_coupling_; }
    [[nodiscard]] const BorderedCouplingData& borderedCoupling() const noexcept { return bordered_coupling_; }
    [[nodiscard]] std::span<const assembly::AuxiliaryOutputBinding>
    auxiliaryOutputBindings() const noexcept
    {
        return auxiliary_output_bindings_;
    }

private:
    BorderedCouplingData bordered_coupling_{};
    std::vector<std::shared_ptr<
        const constraints::SmallCutAggregationProlongationReport>>
        finalized_small_cut_aggregation_prolongations_{};
    std::vector<ExteriorBoundaryMeasurePolicy>
        exterior_boundary_measure_policies_{};
    ExteriorBoundaryMeasurePolicyId
        next_exterior_boundary_measure_policy_id_{1u};
    std::vector<GeneratedBoundaryNitscheTracePolicy>
        generated_boundary_nitsche_trace_policies_{};
    std::vector<GeneratedBoundaryNitscheTraceCertificateRecord>
        generated_boundary_nitsche_trace_certificates_{};
    GeneratedBoundaryNitscheTracePolicyId
        next_generated_boundary_nitsche_trace_policy_id_{1u};
    bool generated_boundary_nitsche_trace_policy_shape_validated_{false};
    std::uint64_t generated_boundary_nitsche_trace_policy_signature_{0u};
	};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_FESYSTEM_H
