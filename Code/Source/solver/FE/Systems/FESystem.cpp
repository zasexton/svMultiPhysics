/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/FESystem.h"

#include "PostProcessing/DerivedResultEvaluator.h"

#include "Systems/SystemAssembly.h"
#include "Systems/AuxiliaryQuadratureLayout.h"
#include "Systems/FEAdaptivityTransfer.h"
#include "Systems/OperatorBackends.h"
#include "Systems/BoundaryReductionService.h"
#include "Auxiliary/AuxiliaryStateManager.h"
#include "Auxiliary/AuxiliaryOperatorRegistry.h"
#include "Auxiliary/AuxiliaryRowOwnership.h"
#include "Auxiliary/AuxiliaryInputRegistry.h"
#include "Auxiliary/AuxiliaryBindings.h"
#include "Auxiliary/AuxiliaryModelBuilder.h"
#include "Auxiliary/AuxiliaryStateStepper.h"
#include "Auxiliary/AuxiliaryMultirateScheduler.h"
#include "Auxiliary/AuxiliaryInitializationSolver.h"
#include "Auxiliary/AuxiliaryEventManager.h"
#include "Constraints/AuxiliaryDrivenDirichletConstraint.h"
#include "Forms/PointEvaluator.h"
#include "Auxiliary/AuxiliaryDerivativeProvider.h"
#include "Systems/SystemsExceptions.h"
 #include "Core/Logger.h"

#include "Assembly/AssemblyKernel.h"
#include "Assembly/CutIntegrationContext.h"
#include "Assembly/GlobalSystemView.h"

#include "Backends/Interfaces/GenericVector.h"
#include "Backends/Interfaces/DofPermutation.h"
#include "Backends/Utils/BackendOptions.h"
#include "Basis/NodeOrderingConventions.h"
#include "Dofs/EntityDofMap.h"
#include "Elements/ElementFactory.h"
#include "Elements/ElementValidator.h"
#include "Elements/ReferenceElement.h"
#include "Geometry/MappingFactory.h"

#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/SymbolicDifferentiation.h"
#include "Forms/JIT/JITKernelWrapper.h"

#include "Core/MpiCollectiveTrace.h"
#include "Spaces/FunctionSpace.h"
#include "Sparsity/DistributedSparsityPattern.h"

#include "Analysis/ProblemAnalysisContext.h"
#include "Analysis/ProblemAnalyzer.h"
#include "Analysis/AnalysisSummaryProducer.h"
#include "Analysis/AnalysisSummaryMatching.h"
#include "Analysis/FormExprScanner.h"
#include "Analysis/SparseMatrixSummaryScanner.h"
#include "Math/FiniteDifference.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#if FE_HAS_MPI
#  include <mpi.h>
#endif

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Assembly/CompositeMeshAccess.h"
#include "Assembly/MeshAccess.h"
#include "Mesh/Adaptivity/Options.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Core/InterfaceMesh.h"
#include "Mesh/Fields/MeshFields.h"
#include "Mesh/Motion/MotionFields.h"
#include "Mesh/Motion/MotionState.h"
#include "Mesh/Topology/CellTopology.h"
#include "Systems/MeshSearchAccess.h"
#endif

namespace svmp {
namespace FE {
namespace systems {

namespace {

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
constexpr std::uint64_t kConstraintRevisionHashOffset = 1469598103934665603ull;
constexpr std::uint64_t kConstraintRevisionHashPrime = 1099511628211ull;

void mixConstraintRevisionHash(std::uint64_t& h, std::uint64_t value) noexcept
{
    h ^= value;
    h *= kConstraintRevisionHashPrime;
}

void mixConstraintRevisionBytes(std::uint64_t& h,
                                const void* data,
                                std::size_t n_bytes) noexcept
{
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n_bytes; ++i) {
        mixConstraintRevisionHash(h, static_cast<std::uint64_t>(bytes[i]));
    }
}

void mixConstraintRevisionString(std::uint64_t& h,
                                 const std::string& value) noexcept
{
    mixConstraintRevisionHash(h, static_cast<std::uint64_t>(value.size()));
    mixConstraintRevisionBytes(h, value.data(), value.size());
}

[[nodiscard]] std::uint64_t meshFieldValueFingerprint(
    const svmp::MeshBase& mesh) noexcept
{
    std::uint64_t h = kConstraintRevisionHashOffset;
    const std::array<svmp::EntityKind, 4> kinds{{
        svmp::EntityKind::Vertex,
        svmp::EntityKind::Edge,
        svmp::EntityKind::Face,
        svmp::EntityKind::Volume,
    }};
    try {
        for (const auto kind : kinds) {
            mixConstraintRevisionHash(h, static_cast<std::uint64_t>(kind));
            const auto names = svmp::MeshFields::list_fields(mesh, kind);
            mixConstraintRevisionHash(h, static_cast<std::uint64_t>(names.size()));
            for (const auto& name : names) {
                mixConstraintRevisionString(h, name);
                const auto handle =
                    svmp::MeshFields::get_field_handle(mesh, kind, name);
                mixConstraintRevisionHash(h, static_cast<std::uint64_t>(handle.id));
                mixConstraintRevisionHash(h, static_cast<std::uint64_t>(
                                                 svmp::MeshFields::field_type(mesh, handle)));
                const auto components =
                    svmp::MeshFields::field_components(mesh, handle);
                const auto entities =
                    svmp::MeshFields::field_entity_count(mesh, handle);
                const auto bytes_per_entity =
                    svmp::MeshFields::field_bytes_per_entity(mesh, handle);
                mixConstraintRevisionHash(h, static_cast<std::uint64_t>(components));
                mixConstraintRevisionHash(h, static_cast<std::uint64_t>(entities));
                mixConstraintRevisionHash(h, static_cast<std::uint64_t>(bytes_per_entity));
                const auto n_bytes = entities * bytes_per_entity;
                const auto* data = svmp::MeshFields::field_data(mesh, handle);
                if (data != nullptr && n_bytes > 0u) {
                    mixConstraintRevisionBytes(h, data, n_bytes);
                }
            }
        }
    } catch (...) {
        mixConstraintRevisionHash(h, std::numeric_limits<std::uint64_t>::max());
    }
    return h;
}
#endif

[[nodiscard]] state::StateFrameTransformRequest makeStateFrameTransformRequest(
    state::StateFrameTransformEvent event,
    const OperatorRevisionSnapshot& snapshot)
{
    state::StateFrameTransformRequest request;
    request.event = event;
    request.source_lifecycle = state::StateVariableLifecycle::TrialWork;
    request.target_lifecycle = state::StateVariableLifecycle::TrialWork;
    if (snapshot.mesh.valid) {
        request.geometry_revision = snapshot.mesh.geometry;
        request.topology_revision = snapshot.mesh.topology;
        request.ownership_revision = snapshot.mesh.ownership;
        request.numbering_revision = snapshot.mesh.numbering;
        request.field_layout_revision = snapshot.mesh.field_layout;
        request.reference_rebase_epoch = snapshot.mesh.reference_rebase;
    }
    return request;
}

[[nodiscard]] bool nativeFaceRankOnePromotionEnabled() noexcept
{
    const char* env = std::getenv("SVMP_DISABLE_MPI_NATIVE_RANK1_PROMOTION");
    if (env == nullptr) {
        return true;
    }
    while (*env == ' ' || *env == '\t' || *env == '\n' || *env == '\r') {
        ++env;
    }
    if (*env == '\0') {
        return true;
    }
    return *env == '0';
}

[[nodiscard]] Real auxiliaryDirectCouplingSign(bool direct_only) noexcept
{
    const char* force_positive = std::getenv("SVMP_POSITIVE_AUX_DIRECT_COUPLING");
    if (force_positive != nullptr) {
        while (*force_positive == ' ' || *force_positive == '\t' || *force_positive == '\n' ||
               *force_positive == '\r') {
            ++force_positive;
        }
        if (*force_positive != '\0' && *force_positive != '0') {
            return Real(1.0);
        }
    }

    const char* force_negative = std::getenv("SVMP_NEGATE_AUX_DIRECT_COUPLING");
    if (force_negative != nullptr) {
        while (*force_negative == ' ' || *force_negative == '\t' || *force_negative == '\n' ||
               *force_negative == '\r') {
            ++force_negative;
        }
        if (*force_negative != '\0' && *force_negative != '0') {
            return Real(-1.0);
        }
    }

    return Real(1.0);
}

template <typename T>
[[nodiscard]] T mpiAllreduceSumIfActive(T value) noexcept
{
#if FE_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        return value;
    }

    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size <= 1) {
        return value;
    }

    T global = value;
    if constexpr (std::is_same_v<T, int>) {
        const auto seq = debug::nextMpiCollectiveTraceSeq();
        debug::traceMpiCollective("before", seq, "FESystem::mpiAllreduceSumIfActive<int>", 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&value, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        debug::traceMpiCollective("after", seq, "FESystem::mpiAllreduceSumIfActive<int>", 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    } else {
        const auto seq = debug::nextMpiCollectiveTraceSeq();
        debug::traceMpiCollective("before", seq, "FESystem::mpiAllreduceSumIfActive<double>", 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&value, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        debug::traceMpiCollective("after", seq, "FESystem::mpiAllreduceSumIfActive<double>", 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    return global;
#else
    return value;
#endif
}

#if FE_HAS_MPI
[[nodiscard]] bool mpiWorldActiveForAnalysis() noexcept
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        return false;
    }
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size > 1;
}

[[nodiscard]] std::uint64_t mpiSumUint64(std::uint64_t value) noexcept
{
    if (!mpiWorldActiveForAnalysis()) {
        return value;
    }
    unsigned long long local = static_cast<unsigned long long>(value);
    unsigned long long global = local;
    const auto seq = debug::nextMpiCollectiveTraceSeq();
    debug::traceMpiCollective("before", seq, "FESystem::mpiSumUint64", 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local, &global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    debug::traceMpiCollective("after", seq, "FESystem::mpiSumUint64", 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return static_cast<std::uint64_t>(global);
}

[[nodiscard]] double mpiReduceDouble(double value, MPI_Op op) noexcept
{
    if (!mpiWorldActiveForAnalysis()) {
        return value;
    }
    double global = value;
    const auto seq = debug::nextMpiCollectiveTraceSeq();
    debug::traceMpiCollective("before", seq, "FESystem::mpiReduceDouble", 1, MPI_DOUBLE, op, MPI_COMM_WORLD);
    MPI_Allreduce(&value, &global, 1, MPI_DOUBLE, op, MPI_COMM_WORLD);
    debug::traceMpiCollective("after", seq, "FESystem::mpiReduceDouble", 1, MPI_DOUBLE, op, MPI_COMM_WORLD);
    return global;
}
#endif

struct AnalysisFieldRange {
    FieldId field_id{INVALID_FIELD_ID};
    GlobalIndex begin{0};
    GlobalIndex end{0};
    LocalIndex components{1};
    std::string name;

    [[nodiscard]] bool valid() const noexcept {
        return field_id != INVALID_FIELD_ID && end > begin;
    }

    [[nodiscard]] bool contains(GlobalIndex dof) const noexcept {
        return dof >= begin && dof < end;
    }

    [[nodiscard]] GlobalIndex size() const noexcept {
        return end > begin ? end - begin : 0;
    }
};

struct AnalysisSaddlePair {
    AnalysisFieldRange primary;
    AnalysisFieldRange multiplier;

    [[nodiscard]] bool valid() const noexcept {
        return primary.valid() && multiplier.valid() &&
               primary.field_id != multiplier.field_id;
    }
};

struct CouplingMatrixStats {
    std::uint64_t active_multiplier_rows{0};
    std::uint64_t active_primary_rows{0};
    std::uint64_t multiplier_rows_with_primary_entries{0};
    std::uint64_t primary_rows_with_multiplier_entries{0};
    std::uint64_t primary_rows_with_diagonal{0};
    std::uint64_t nonfinite_entries{0};
    double multiplier_min_row_norm{std::numeric_limits<double>::infinity()};
    double primary_min_row_norm{std::numeric_limits<double>::infinity()};
    double multiplier_coupling_frob_sq{0.0};
    double primary_coupling_frob_sq{0.0};
    double primary_min_abs_diag{std::numeric_limits<double>::infinity()};
    double primary_max_abs_diag{0.0};
};

struct SparseRankEstimate {
    std::uint64_t row_count{0};
    std::uint64_t estimated_rank{0};
    std::uint64_t nullity{0};
    std::uint64_t near_zero_row_count{0};
    std::uint64_t near_zero_pivot_count{0};
    Real matrix_norm_estimate{};
    Real tolerance{};
    bool complete{false};
};

[[nodiscard]] std::optional<AnalysisFieldRange>
analysisFieldRangeById(const dofs::FieldDofMap& field_map, FieldId fid)
{
    if (fid == INVALID_FIELD_ID ||
        static_cast<std::size_t>(fid) >= field_map.numFields()) {
        return std::nullopt;
    }
    const auto idx = static_cast<std::size_t>(fid);
    const auto range = field_map.getFieldDofRange(idx);
    const auto& field = field_map.getField(idx);
    return AnalysisFieldRange{fid,
                              range.first,
                              range.second,
                              field.n_components,
                              field.name};
}

[[nodiscard]] std::optional<AnalysisFieldRange>
analysisFieldRangeByName(const dofs::FieldDofMap& field_map, std::string_view name)
{
    if (name.empty()) {
        return std::nullopt;
    }
    const int idx = field_map.getFieldIndex(std::string(name));
    if (idx < 0) {
        return std::nullopt;
    }
    return analysisFieldRangeById(field_map, static_cast<FieldId>(idx));
}

[[nodiscard]] std::optional<FieldId> fieldIdFromVariable(
    const analysis::VariableKey& variable,
    const dofs::FieldDofMap& field_map) noexcept
{
    if (variable.kind != analysis::VariableKind::FieldComponent ||
        variable.field_id == INVALID_FIELD_ID ||
        static_cast<std::size_t>(variable.field_id) >= field_map.numFields()) {
        return std::nullopt;
    }
    return variable.field_id;
}

[[nodiscard]] std::vector<analysis::VariableKey> blockVariables(
    const analysis::OperatorBlockId& block)
{
    std::vector<analysis::VariableKey> variables;
    const auto append = [&](const analysis::VariableKey& variable) {
        if (std::find(variables.begin(), variables.end(), variable) ==
            variables.end()) {
            variables.push_back(variable);
        }
    };
    for (const auto& variable : block.test_variables) {
        append(variable);
    }
    for (const auto& variable : block.trial_variables) {
        append(variable);
    }
    return variables;
}

[[nodiscard]] analysis::ReferenceCellFamily referenceCellFamilyFor(ElementType element) noexcept
{
    switch (element) {
        case ElementType::Line2:
        case ElementType::Line3:
        case ElementType::Triangle3:
        case ElementType::Triangle6:
        case ElementType::Tetra4:
        case ElementType::Tetra10:
            return analysis::ReferenceCellFamily::Simplex;
        case ElementType::Quad4:
        case ElementType::Quad8:
        case ElementType::Quad9:
        case ElementType::Hex8:
        case ElementType::Hex20:
        case ElementType::Hex27:
            return analysis::ReferenceCellFamily::TensorProduct;
        case ElementType::Wedge6:
        case ElementType::Wedge15:
        case ElementType::Wedge18:
            return analysis::ReferenceCellFamily::Wedge;
        case ElementType::Pyramid5:
        case ElementType::Pyramid13:
        case ElementType::Pyramid14:
            return analysis::ReferenceCellFamily::Pyramid;
        default:
            return analysis::ReferenceCellFamily::Unknown;
    }
}

[[nodiscard]] analysis::SpaceContinuityClass
continuityClassFor(Continuity continuity, spaces::SpaceType space_type) noexcept
{
    if (space_type == spaces::SpaceType::Trace ||
        space_type == spaces::SpaceType::Mortar) {
        return analysis::SpaceContinuityClass::TraceOnly;
    }

    switch (continuity) {
        case Continuity::C0:
        case Continuity::C1:
            return analysis::SpaceContinuityClass::Continuous;
        case Continuity::L2:
            return analysis::SpaceContinuityClass::Discontinuous;
        case Continuity::H_div:
            return analysis::SpaceContinuityClass::NormalContinuous;
        case Continuity::H_curl:
            return analysis::SpaceContinuityClass::TangentialContinuous;
        case Continuity::Custom:
            return analysis::SpaceContinuityClass::Custom;
    }
    return analysis::SpaceContinuityClass::Unknown;
}

[[nodiscard]] analysis::MappingTransform
mappingTransformFor(Continuity continuity, spaces::SpaceType space_type) noexcept
{
    if (space_type == spaces::SpaceType::Trace ||
        space_type == spaces::SpaceType::Mortar) {
        return analysis::MappingTransform::TracePullback;
    }
    if (continuity == Continuity::H_div) {
        return analysis::MappingTransform::ContravariantPiola;
    }
    if (continuity == Continuity::H_curl) {
        return analysis::MappingTransform::CovariantPiola;
    }
    if (continuity == Continuity::C0 ||
        continuity == Continuity::C1 ||
        continuity == Continuity::L2) {
        return analysis::MappingTransform::Identity;
    }
    return analysis::MappingTransform::Unknown;
}

[[nodiscard]] analysis::SpaceFamily
spaceFamilyFor(Continuity continuity, spaces::SpaceType space_type) noexcept
{
    if (space_type == spaces::SpaceType::Trace ||
        space_type == spaces::SpaceType::Mortar) {
        return analysis::SpaceFamily::Trace;
    }
    if (space_type == spaces::SpaceType::L2) {
        return analysis::SpaceFamily::L2;
    }

    switch (continuity) {
        case Continuity::C0:
        case Continuity::C1:
            return analysis::SpaceFamily::H1;
        case Continuity::H_div:
            return analysis::SpaceFamily::HDiv;
        case Continuity::H_curl:
            return analysis::SpaceFamily::HCurl;
        case Continuity::L2:
            return analysis::SpaceFamily::L2;
        case Continuity::Custom:
            return analysis::SpaceFamily::Custom;
    }
    return analysis::SpaceFamily::Unknown;
}

[[nodiscard]] analysis::ElementFamily
elementFamilyFor(const spaces::FunctionSpace& space) noexcept
{
    const auto space_type = space.space_type();
    if (space_type == spaces::SpaceType::Trace) {
        return analysis::ElementFamily::Trace;
    }
    if (space_type == spaces::SpaceType::Mortar) {
        return analysis::ElementFamily::Mortar;
    }
    if (space_type == spaces::SpaceType::GenericBasis ||
        space_type == spaces::SpaceType::Composite ||
        space_type == spaces::SpaceType::Mixed ||
        space_type == spaces::SpaceType::Adaptive) {
        return analysis::ElementFamily::Custom;
    }
    if (space_type == spaces::SpaceType::L2) {
        return analysis::ElementFamily::DG;
    }

    switch (space.element().basis().basis_type()) {
        case BasisType::Lagrange: return analysis::ElementFamily::Lagrange;
        case BasisType::RaviartThomas: return analysis::ElementFamily::RaviartThomas;
        case BasisType::BDM: return analysis::ElementFamily::BDM;
        case BasisType::Nedelec: return analysis::ElementFamily::Nedelec;
        case BasisType::Bubble: return analysis::ElementFamily::BubbleEnrichedLagrange;
        case BasisType::Custom: return analysis::ElementFamily::Custom;
        default: return analysis::ElementFamily::Custom;
    }
}

void applyBoundaryScopeMetadata(analysis::FieldDescriptor& fd,
                                const std::vector<analysis::BoundaryConditionDescriptor>& bcs)
{
    for (const auto& bc : bcs) {
        if (bc.primary_variable.kind != analysis::VariableKind::FieldComponent ||
            bc.primary_variable.field_id != fd.field_id) {
            continue;
        }
        fd.boundary_condition_scope_metadata_present = true;
        if (bc.enforcement_kind == analysis::EnforcementKind::Strong &&
            (bc.trace_kind == analysis::TraceKind::Value ||
             bc.trace_kind == analysis::TraceKind::AlgebraicRelation)) {
            fd.strong_dirichlet_boundary_present = true;
        }
        if (bc.trace_kind == analysis::TraceKind::NormalComponent ||
            bc.trace_kind == analysis::TraceKind::NormalFlux ||
            bc.trace_kind == analysis::TraceKind::Flux) {
            fd.normal_trace_boundary_scope_present = true;
        }
        if (bc.anchors_constant_mode) {
            fd.gauge_fixing_metadata_present = true;
        }
    }
}

[[nodiscard]] analysis::OperatorBlockId globalAnalysisBlock(
    const OperatorTag& op,
    const dofs::FieldDofMap& field_map)
{
    analysis::OperatorBlockId block;
    block.operator_tag = op;
    block.domain = analysis::DomainKind::Cell;
    block.role = analysis::ContributionRole::DiagonalBlock;
    for (std::size_t i = 0; i < field_map.numFields(); ++i) {
        const auto variable = analysis::VariableKey::field(static_cast<FieldId>(i));
        block.test_variables.push_back(variable);
        block.trial_variables.push_back(variable);
    }
    return block;
}

[[nodiscard]] analysis::OperatorBlockId saddlePairBlock(
    const OperatorTag& op,
    const AnalysisSaddlePair& pair)
{
    analysis::OperatorBlockId block;
    block.operator_tag = op;
    block.domain = analysis::DomainKind::Cell;
    block.role = analysis::ContributionRole::ConstraintBlock;
    block.test_variables = {
        analysis::VariableKey::field(pair.primary.field_id),
        analysis::VariableKey::field(pair.multiplier.field_id),
    };
    block.trial_variables = block.test_variables;
    return block;
}

[[nodiscard]] analysis::OperatorBlockId contributionBlock(
    const analysis::ContributionDescriptor& contribution,
    const OperatorTag& fallback_op)
{
    analysis::OperatorBlockId block;
    block.test_variables = contribution.test_variables;
    block.trial_variables = contribution.trial_variables;
    block.domain = contribution.domain;
    block.role = contribution.role;
    block.contribution_id = contribution.contribution_id;
    block.operator_tag = contribution.operator_tag.empty()
        ? fallback_op
        : contribution.operator_tag;
    if (contribution.domain == analysis::DomainKind::Boundary) {
        block.marker = contribution.boundary_marker;
    } else if (contribution.domain == analysis::DomainKind::InterfaceFace) {
        block.marker = contribution.interface_marker;
    }
    return block;
}

[[nodiscard]] std::optional<AnalysisSaddlePair>
findAnalysisSaddlePairFromSolverOptions(const dofs::FieldDofMap& field_map,
                                        const backends::SolverOptions* options)
{
    if (options == nullptr) {
        return std::nullopt;
    }

    const auto primary_name =
        options->resolveBlockNameForRole(backends::BlockRole::PrimaryField);
    const auto constraint_name =
        options->resolveBlockNameForRole(backends::BlockRole::ConstraintField);
    auto primary = analysisFieldRangeByName(field_map, primary_name);
    auto multiplier = analysisFieldRangeByName(field_map, constraint_name);
    if (primary && multiplier) {
        AnalysisSaddlePair pair{*primary, *multiplier};
        if (pair.valid()) {
            return pair;
        }
    }

    if (options->block_layout && options->block_layout->hasSaddlePoint()) {
        const auto* primary_block = options->block_layout->primaryFieldBlock();
        const auto* constraint_block = options->block_layout->constraintFieldBlock();
        primary = primary_block ? analysisFieldRangeByName(field_map, primary_block->name)
                                : std::nullopt;
        multiplier = constraint_block ? analysisFieldRangeByName(field_map, constraint_block->name)
                                      : std::nullopt;
        if (primary && multiplier) {
            AnalysisSaddlePair pair{*primary, *multiplier};
            if (pair.valid()) {
                return pair;
            }
        }
    }

    return std::nullopt;
}

[[nodiscard]] std::optional<AnalysisSaddlePair>
findAnalysisSaddlePairFromContributions(
    const dofs::FieldDofMap& field_map,
    const std::vector<analysis::ContributionDescriptor>& contributions)
{
    const auto make_pair_from_fields = [&](FieldId a, FieldId b)
        -> std::optional<AnalysisSaddlePair> {
        auto ar = analysisFieldRangeById(field_map, a);
        auto br = analysisFieldRangeById(field_map, b);
        if (!ar || !br || ar->field_id == br->field_id) {
            return std::nullopt;
        }
        if (ar->components > 1 && br->components == 1) {
            return AnalysisSaddlePair{*ar, *br};
        }
        if (br->components > 1 && ar->components == 1) {
            return AnalysisSaddlePair{*br, *ar};
        }
        return std::nullopt;
    };

    for (const auto& contribution : contributions) {
        if (contribution.role != analysis::ContributionRole::ConstraintBlock &&
            contribution.role != analysis::ContributionRole::OffDiagonalBlock &&
            contribution.role != analysis::ContributionRole::StabilizationBlock) {
            continue;
        }
        for (const auto& pairing : contribution.pairings) {
            auto row = fieldIdFromVariable(pairing.row_var, field_map);
            auto col = fieldIdFromVariable(pairing.col_var, field_map);
            if (row && col) {
                if (auto pair = make_pair_from_fields(*row, *col)) {
                    return pair;
                }
            }
        }
        for (const auto& test : contribution.test_variables) {
            auto test_fid = fieldIdFromVariable(test, field_map);
            if (!test_fid) {
                continue;
            }
            for (const auto& trial : contribution.trial_variables) {
                auto trial_fid = fieldIdFromVariable(trial, field_map);
                if (trial_fid) {
                    if (auto pair = make_pair_from_fields(*test_fid, *trial_fid)) {
                        return pair;
                    }
                }
            }
        }
    }

    std::optional<AnalysisFieldRange> first_vector;
    std::optional<AnalysisFieldRange> first_scalar;
    for (std::size_t i = 0; i < field_map.numFields(); ++i) {
        auto range = analysisFieldRangeById(field_map, static_cast<FieldId>(i));
        if (!range) {
            continue;
        }
        if (!first_vector && range->components > 1) {
            first_vector = *range;
        } else if (!first_scalar && range->components == 1) {
            first_scalar = *range;
        }
    }
    if (first_vector && first_scalar) {
        return AnalysisSaddlePair{*first_vector, *first_scalar};
    }
    return std::nullopt;
}

[[nodiscard]] CouplingMatrixStats scanCouplingStats(
    const analysis::SparseRowScanSource& source,
    const AnalysisSaddlePair& pair,
    const std::unordered_set<GlobalIndex>& constrained_dofs,
    Real tolerance)
{
    CouplingMatrixStats stats;
    source.forEachLocalRow(
        [&](analysis::GlobalDofId row,
            const std::vector<analysis::SparseMatrixRowEntry>& entries,
            int) {
            const bool row_constrained =
                constrained_dofs.find(row) != constrained_dofs.end();
            const bool row_is_multiplier = pair.multiplier.contains(row);
            const bool row_is_primary = pair.primary.contains(row);

            double multiplier_row_sq = 0.0;
            double primary_row_sq = 0.0;
            bool has_primary_diagonal = false;

            for (const auto& entry : entries) {
                const auto value = static_cast<double>(entry.value);
                if (!std::isfinite(value)) {
                    ++stats.nonfinite_entries;
                    continue;
                }
                if (row_is_multiplier && !row_constrained &&
                    pair.primary.contains(entry.col)) {
                    multiplier_row_sq += value * value;
                }
                if (row_is_primary && !row_constrained &&
                    pair.multiplier.contains(entry.col)) {
                    primary_row_sq += value * value;
                }
                if (row_is_primary && !row_constrained && entry.col == row) {
                    const double abs_diag = std::abs(value);
                    if (abs_diag > static_cast<double>(tolerance)) {
                        has_primary_diagonal = true;
                        stats.primary_min_abs_diag =
                            std::min(stats.primary_min_abs_diag, abs_diag);
                        stats.primary_max_abs_diag =
                            std::max(stats.primary_max_abs_diag, abs_diag);
                    }
                }
            }

            if (row_is_multiplier && !row_constrained) {
                ++stats.active_multiplier_rows;
                if (multiplier_row_sq > static_cast<double>(tolerance) *
                                            static_cast<double>(tolerance)) {
                    ++stats.multiplier_rows_with_primary_entries;
                    const double norm = std::sqrt(multiplier_row_sq);
                    stats.multiplier_min_row_norm =
                        std::min(stats.multiplier_min_row_norm, norm);
                    stats.multiplier_coupling_frob_sq += multiplier_row_sq;
                }
            }

            if (row_is_primary && !row_constrained) {
                ++stats.active_primary_rows;
                if (primary_row_sq > static_cast<double>(tolerance) *
                                     static_cast<double>(tolerance)) {
                    ++stats.primary_rows_with_multiplier_entries;
                    const double norm = std::sqrt(primary_row_sq);
                    stats.primary_min_row_norm =
                        std::min(stats.primary_min_row_norm, norm);
                    stats.primary_coupling_frob_sq += primary_row_sq;
                }
                if (has_primary_diagonal) {
                    ++stats.primary_rows_with_diagonal;
                }
            }
        });

#if FE_HAS_MPI
    stats.active_multiplier_rows = mpiSumUint64(stats.active_multiplier_rows);
    stats.active_primary_rows = mpiSumUint64(stats.active_primary_rows);
    stats.multiplier_rows_with_primary_entries =
        mpiSumUint64(stats.multiplier_rows_with_primary_entries);
    stats.primary_rows_with_multiplier_entries =
        mpiSumUint64(stats.primary_rows_with_multiplier_entries);
    stats.primary_rows_with_diagonal =
        mpiSumUint64(stats.primary_rows_with_diagonal);
    stats.nonfinite_entries = mpiSumUint64(stats.nonfinite_entries);
    stats.multiplier_min_row_norm =
        mpiReduceDouble(stats.multiplier_min_row_norm, MPI_MIN);
    stats.primary_min_row_norm =
        mpiReduceDouble(stats.primary_min_row_norm, MPI_MIN);
    stats.multiplier_coupling_frob_sq =
        mpiReduceDouble(stats.multiplier_coupling_frob_sq, MPI_SUM);
    stats.primary_coupling_frob_sq =
        mpiReduceDouble(stats.primary_coupling_frob_sq, MPI_SUM);
    stats.primary_min_abs_diag =
        mpiReduceDouble(stats.primary_min_abs_diag, MPI_MIN);
    stats.primary_max_abs_diag =
        mpiReduceDouble(stats.primary_max_abs_diag, MPI_MAX);
#endif

    if (!std::isfinite(stats.multiplier_min_row_norm)) {
        stats.multiplier_min_row_norm = 0.0;
    }
    if (!std::isfinite(stats.primary_min_row_norm)) {
        stats.primary_min_row_norm = 0.0;
    }
    if (!std::isfinite(stats.primary_min_abs_diag)) {
        stats.primary_min_abs_diag = 0.0;
    }
    return stats;
}

[[nodiscard]] SparseRankEstimate estimateSparseRankFromRows(
    const analysis::SparseRowScanSource& source,
    const std::unordered_set<GlobalIndex>& constrained_dofs)
{
    SparseRankEstimate estimate;
    source.forEachLocalRow(
        [&](analysis::GlobalDofId row,
            const std::vector<analysis::SparseMatrixRowEntry>& entries,
            int) {
            if (constrained_dofs.find(row) != constrained_dofs.end()) {
                return;
            }
            Real row_norm{};
            for (const auto& entry : entries) {
                if (constrained_dofs.find(entry.col) != constrained_dofs.end()) {
                    continue;
                }
                if (std::isfinite(static_cast<double>(entry.value))) {
                    row_norm += std::abs(entry.value);
                }
            }
            estimate.matrix_norm_estimate =
                std::max(estimate.matrix_norm_estimate, row_norm);
            ++estimate.row_count;
        });

    const Real scale = std::max(estimate.matrix_norm_estimate, Real{1});
    const Real n_scale =
        std::sqrt(static_cast<Real>(std::max<std::uint64_t>(estimate.row_count, 1u)));
    estimate.tolerance = std::max(static_cast<Real>(1.0e-14) * scale,
                                  std::numeric_limits<Real>::epsilon() *
                                      scale * n_scale * static_cast<Real>(64));

    source.forEachLocalRow(
        [&](analysis::GlobalDofId row,
            const std::vector<analysis::SparseMatrixRowEntry>& entries,
            int) {
            if (constrained_dofs.find(row) != constrained_dofs.end()) {
                return;
            }
            Real row_norm{};
            Real diagonal_abs{};
            bool diagonal_seen = false;
            for (const auto& entry : entries) {
                if (constrained_dofs.find(entry.col) != constrained_dofs.end()) {
                    continue;
                }
                if (!std::isfinite(static_cast<double>(entry.value))) {
                    continue;
                }
                row_norm += std::abs(entry.value);
                if (entry.col == row) {
                    diagonal_abs += std::abs(entry.value);
                    diagonal_seen = true;
                }
            }
            if (row_norm <= estimate.tolerance) {
                ++estimate.near_zero_row_count;
            }
            if (!diagonal_seen || diagonal_abs <= estimate.tolerance) {
                ++estimate.near_zero_pivot_count;
            }
        });

    estimate.estimated_rank =
        estimate.row_count >= estimate.near_zero_row_count
            ? estimate.row_count - estimate.near_zero_row_count
            : 0u;
    estimate.nullity =
        estimate.row_count >= estimate.estimated_rank
            ? estimate.row_count - estimate.estimated_rank
            : 0u;
    estimate.complete = source.hasCompleteGlobalRows() || !source.isDistributed();

#if FE_HAS_MPI
    if (source.isDistributed()) {
        estimate.row_count = mpiSumUint64(estimate.row_count);
        estimate.estimated_rank = mpiSumUint64(estimate.estimated_rank);
        estimate.nullity = mpiSumUint64(estimate.nullity);
        estimate.near_zero_row_count = mpiSumUint64(estimate.near_zero_row_count);
        estimate.near_zero_pivot_count =
            mpiSumUint64(estimate.near_zero_pivot_count);
        estimate.matrix_norm_estimate =
            static_cast<Real>(mpiReduceDouble(estimate.matrix_norm_estimate,
                                              MPI_MAX));
        estimate.tolerance =
            static_cast<Real>(mpiReduceDouble(estimate.tolerance, MPI_MAX));
    }
#endif
    return estimate;
}

[[nodiscard]] bool hasClaimFromAnalyzer(const analysis::ProblemAnalysisReport& report,
                                        analysis::PropertyKind kind,
                                        std::string_view analyzer)
{
    return std::any_of(report.claims.begin(), report.claims.end(),
                       [&](const analysis::PropertyClaim& claim) {
                           return claim.kind == kind &&
                                  claim.claim_origin == analyzer;
                       });
}

[[nodiscard]] bool hasRigidKernelClaim(
    const analysis::ProblemAnalysisReport& report)
{
    return std::any_of(report.claims.begin(), report.claims.end(),
                       [](const analysis::PropertyClaim& claim) {
                           return claim.kind == analysis::PropertyKind::Nullspace &&
                                  claim.nullspace_family &&
                                  *claim.nullspace_family ==
                                      analysis::NullspaceFamily::KernelOfSymGrad;
                       });
}

[[nodiscard]] std::vector<analysis::VariableKey> nullspaceClaimVariables(
    const analysis::ProblemAnalysisReport& report,
    const std::vector<analysis::VariableKey>& fallback)
{
    std::vector<analysis::VariableKey> variables;
    for (const auto& claim : report.claims) {
        if (claim.kind != analysis::PropertyKind::Nullspace &&
            claim.kind != analysis::PropertyKind::UnderConstraint) {
            continue;
        }
        for (const auto& variable : claim.variables) {
            if (std::find(variables.begin(), variables.end(), variable) ==
                variables.end()) {
                variables.push_back(variable);
            }
        }
    }
    return variables.empty() ? fallback : variables;
}

[[nodiscard]] analysis::NullspaceDegeneracySummary
makeNullspaceDegeneracySummary(
    const OperatorTag& op,
    const analysis::OperatorBlockId& block,
    const SparseRankEstimate& rank,
    std::uint64_t constrained_dofs,
    const std::vector<analysis::VariableKey>& affected_variables,
    const analysis::ProblemAnalysisReport& baseline_report,
    bool saddle_pair_present)
{
    analysis::NullspaceDegeneracySummary summary;
    summary.degeneracy_id = op.empty() ? "assembled-nullspace" : op + ":nullspace";
    summary.block = block;
    summary.affected_variables = affected_variables;
    summary.estimated_rank = rank.estimated_rank;
    summary.nullity = rank.nullity;
    summary.near_zero_pivot_count = rank.near_zero_pivot_count;
    summary.near_zero_row_count = rank.near_zero_row_count;
    summary.constrained_dof_count = constrained_dofs;
    summary.free_dof_count = rank.row_count;
    summary.matrix_norm_estimate = rank.matrix_norm_estimate;
    summary.rank_tolerance = rank.tolerance;
    summary.rank_estimate_present = rank.complete || rank.row_count > 0u;
    summary.constraint_mask_present = true;
    summary.saddle_pair_present = saddle_pair_present;
    summary.kernel_claim_evidence_present =
        hasClaimFromAnalyzer(baseline_report,
                             analysis::PropertyKind::Nullspace,
                             "KernelAnalyzer");
    summary.constraint_rank_evidence_present =
        hasClaimFromAnalyzer(baseline_report,
                             analysis::PropertyKind::UnderConstraint,
                             "ConstraintRankAnalyzer") ||
        hasClaimFromAnalyzer(baseline_report,
                             analysis::PropertyKind::InitialDataCompatibility,
                             "ConstraintRankAnalyzer");

    if (rank.nullity == 0u && constrained_dofs > 0u &&
        summary.kernel_claim_evidence_present) {
        summary.degeneracy_class = analysis::DegeneracyClass::ProjectedKernel;
        summary.nullspace_handling = analysis::NullspaceHandlingClass::ProjectedOut;
        summary.reason =
            "Kernel evidence is present, but the reduced free-free operator has no detected null rows";
    } else if (rank.nullity > 0u && hasRigidKernelClaim(baseline_report)) {
        summary.degeneracy_class =
            analysis::DegeneracyClass::DegenerateDiagnostic;
        summary.nullspace_handling =
            constrained_dofs > 0u
                ? analysis::NullspaceHandlingClass::Retained
                : analysis::NullspaceHandlingClass::Uncontrolled;
        summary.reason =
            "Rigid-body/symmetric-gradient kernel remains in the assembled diagnostic operator";
    } else if (rank.nullity > 0u && constrained_dofs == 0u &&
               summary.kernel_claim_evidence_present) {
        summary.degeneracy_class =
            analysis::DegeneracyClass::GaugeLikeNullspace;
        summary.nullspace_handling = analysis::NullspaceHandlingClass::Uncontrolled;
        summary.reason =
            "Kernel evidence is present and no strong constraint mask anchors the reduced operator";
    } else if (rank.nullity > 0u) {
        summary.degeneracy_class = analysis::DegeneracyClass::UnanchoredKernel;
        summary.nullspace_handling = analysis::NullspaceHandlingClass::Uncontrolled;
        summary.reason =
            "Near-zero reduced rows or pivots indicate an unanchored numerical kernel";
    } else {
        summary.degeneracy_class = analysis::DegeneracyClass::Unknown;
        summary.nullspace_handling =
            constrained_dofs > 0u
                ? analysis::NullspaceHandlingClass::AnchoredByConstraints
                : analysis::NullspaceHandlingClass::Unknown;
        summary.reason = "No reduced nullity was detected by sparse row diagnostics";
    }
    return summary;
}

template <typename T>
void appendAnalysisVector(std::vector<T>& dst, const std::vector<T>& src)
{
    dst.insert(dst.end(), src.begin(), src.end());
}

void appendRegisteredAnalysisSummarySet(analysis::AnalysisSummarySet& dst,
                                        const analysis::AnalysisSummarySet& src)
{
    appendAnalysisVector(dst.norm_metadata, src.norm_metadata);
    appendAnalysisVector(dst.coefficient_properties, src.coefficient_properties);
    appendAnalysisVector(dst.discrete_matrices, src.discrete_matrices);
    appendAnalysisVector(dst.reduced_matrices, src.reduced_matrices);
    appendAnalysisVector(dst.schur_complements, src.schur_complements);
    appendAnalysisVector(dst.nullspace_degeneracies,
                         src.nullspace_degeneracies);
    appendAnalysisVector(dst.robustness_trends, src.robustness_trends);
    appendAnalysisVector(dst.applicability, src.applicability);
    appendAnalysisVector(dst.numerical_error_budgets,
                         src.numerical_error_budgets);
    appendAnalysisVector(dst.local_stencils, src.local_stencils);
    appendAnalysisVector(dst.mesh_geometry_quality, src.mesh_geometry_quality);
    appendAnalysisVector(dst.flux_balances, src.flux_balances);
    appendAnalysisVector(dst.temporal_stability, src.temporal_stability);
    appendAnalysisVector(dst.boundary_symbols, src.boundary_symbols);
    appendAnalysisVector(dst.inf_sup_estimates, src.inf_sup_estimates);
    appendAnalysisVector(dst.inf_sup_pair_certifications,
                         src.inf_sup_pair_certifications);
    appendAnalysisVector(dst.energy_entropy, src.energy_entropy);
    appendAnalysisVector(dst.invariant_domains, src.invariant_domains);
    appendAnalysisVector(dst.equilibrium_preservation, src.equilibrium_preservation);
    appendAnalysisVector(dst.moving_domain, src.moving_domain);
    appendAnalysisVector(dst.transfer_operators, src.transfer_operators);
    appendAnalysisVector(dst.adjoint_consistency, src.adjoint_consistency);
    appendAnalysisVector(dst.parameter_scales, src.parameter_scales);
    appendAnalysisVector(dst.stabilization_adequacy, src.stabilization_adequacy);
    appendAnalysisVector(dst.initial_compatibility, src.initial_compatibility);
    appendAnalysisVector(dst.dae_structure_evidence, src.dae_structure_evidence);
    appendAnalysisVector(dst.compatible_complexes, src.compatible_complexes);
    appendAnalysisVector(dst.nonlinear_tangents, src.nonlinear_tangents);
    appendAnalysisVector(dst.spectral_structures, src.spectral_structures);
    appendAnalysisVector(dst.error_estimators, src.error_estimators);
    appendAnalysisVector(dst.quadrature_adequacy, src.quadrature_adequacy);
    appendAnalysisVector(dst.coupled_system_stability,
                         src.coupled_system_stability);
    appendAnalysisVector(dst.minimum_residual_stability,
                         src.minimum_residual_stability);
}

[[nodiscard]] std::string contributionIdentity(
    const analysis::ContributionDescriptor& contribution,
    std::string_view fallback)
{
    if (!contribution.contribution_id.empty()) {
        return contribution.contribution_id;
    }
    if (!contribution.operator_tag.empty()) {
        return contribution.operator_tag;
    }
    return std::string(fallback);
}

[[nodiscard]] std::vector<analysis::VariableKey> contributionVariables(
    const analysis::ContributionDescriptor& contribution)
{
    std::vector<analysis::VariableKey> variables = contribution.test_variables;
    for (const auto& variable : contribution.trial_variables) {
        if (std::find(variables.begin(), variables.end(), variable) ==
            variables.end()) {
            variables.push_back(variable);
        }
    }
    for (const auto& variable : contribution.related_variables) {
        if (std::find(variables.begin(), variables.end(), variable) ==
            variables.end()) {
            variables.push_back(variable);
        }
    }
    return variables;
}

[[nodiscard]] bool hasConstraintBlockStructure(
    const std::vector<analysis::ContributionDescriptor>& contributions)
{
    return std::any_of(
        contributions.begin(),
        contributions.end(),
        [](const analysis::ContributionDescriptor& contribution) {
            return contribution.role == analysis::ContributionRole::ConstraintBlock ||
                   (contribution.role == analysis::ContributionRole::OffDiagonalBlock &&
                    !contribution.pairings.empty());
        });
}

[[nodiscard]] bool hasSecondOrderScalarDiffusionShape(
    const dofs::FieldDofMap& field_map,
    const std::vector<analysis::ContributionDescriptor>& contributions)
{
    if (field_map.numFields() != 1u) {
        return false;
    }
    const auto& field = field_map.getField(0);
    if (field.n_components != 1) {
        return false;
    }
    return std::any_of(
        contributions.begin(),
        contributions.end(),
        [](const analysis::ContributionDescriptor& contribution) {
            return contribution.role == analysis::ContributionRole::DiagonalBlock &&
                   contribution.domain == analysis::DomainKind::Cell &&
                   analysis::hasFlag(contribution.traits,
                                     analysis::OperatorTraitFlags::HasSecondOrder) &&
                   !analysis::hasFlag(contribution.traits,
                                      analysis::OperatorTraitFlags::HasFirstOrder);
        });
}

void appendApplicabilitySummary(
    analysis::AnalysisSummarySet& summaries,
    analysis::TheoremFamily family,
    analysis::ApplicabilityClass applicability,
    const analysis::OperatorBlockId& block,
    std::vector<analysis::VariableKey> variables,
    std::string reason,
    bool field_descriptor_evidence,
    bool contribution_trait_evidence,
    bool block_structure_evidence)
{
    analysis::ApplicabilitySummary summary;
    summary.theorem_family = family;
    summary.applicability = applicability;
    summary.block = block;
    summary.variables = std::move(variables);
    summary.reason = std::move(reason);
    summary.inferred_from_field_descriptors = field_descriptor_evidence;
    summary.inferred_from_contribution_traits = contribution_trait_evidence;
    summary.inferred_from_block_structure = block_structure_evidence;
    summaries.applicability.push_back(std::move(summary));
}

void appendAutomaticApplicabilitySummaries(
    analysis::AnalysisSummarySet& summaries,
    const analysis::OperatorBlockId& global_block,
    const dofs::FieldDofMap& field_map,
    const std::vector<analysis::ContributionDescriptor>& contributions,
    bool saddle_pair_present)
{
    const auto variables = blockVariables(global_block);
    const bool mixed_or_constraint =
        saddle_pair_present || hasConstraintBlockStructure(contributions);
    const bool scalar_diffusion =
        hasSecondOrderScalarDiffusionShape(field_map, contributions);

    const auto scalar_applicability =
        mixed_or_constraint
            ? analysis::ApplicabilityClass::NotApplicable
            : (scalar_diffusion ? analysis::ApplicabilityClass::Applicable
                                : analysis::ApplicabilityClass::Unknown);
    const std::string scalar_reason =
        mixed_or_constraint
            ? "mixed saddle-point or constraint-block structure is not a scalar nodal diffusion theorem scope"
            : (scalar_diffusion
                   ? "single scalar field with second-order diagonal cell contribution"
                   : "scalar nodal diffusion prerequisites are incomplete");

    appendApplicabilitySummary(summaries,
                               analysis::TheoremFamily::ScalarDMP,
                               scalar_applicability,
                               global_block,
                               variables,
                               scalar_reason,
                               true,
                               true,
                               mixed_or_constraint);
    appendApplicabilitySummary(summaries,
                               analysis::TheoremFamily::MMatrix,
                               scalar_applicability,
                               global_block,
                               variables,
                               scalar_reason,
                               true,
                               true,
                               mixed_or_constraint);
    appendApplicabilitySummary(summaries,
                               analysis::TheoremFamily::InvariantDomain,
                               scalar_applicability,
                               global_block,
                               variables,
                               mixed_or_constraint
                                   ? "mixed block structure requires a problem-specific invariant set, not a scalar invariant-domain gate"
                                   : scalar_reason,
                               true,
                               true,
                               mixed_or_constraint);

    const auto mixed_applicability =
        saddle_pair_present ? analysis::ApplicabilityClass::Applicable
                            : analysis::ApplicabilityClass::NotApplicable;
    const std::string mixed_reason =
        saddle_pair_present
            ? "saddle-point block structure was inferred from solver options or contributions"
            : "no saddle-point pair was inferred from generic block metadata";
    appendApplicabilitySummary(summaries,
                               analysis::TheoremFamily::InfSup,
                               mixed_applicability,
                               global_block,
                               variables,
                               mixed_reason,
                               false,
                               true,
                               true);
    appendApplicabilitySummary(summaries,
                               analysis::TheoremFamily::Fortin,
                               mixed_applicability,
                               global_block,
                               variables,
                               mixed_reason,
                               false,
                               true,
                               true);
    appendApplicabilitySummary(summaries,
                               analysis::TheoremFamily::Schur,
                               mixed_applicability,
                               global_block,
                               variables,
                               mixed_reason,
                               false,
                               true,
                               true);
}

[[nodiscard]] std::string runIdFor(const OperatorTag& op, GlobalIndex dofs)
{
    std::ostringstream os;
    os << (op.empty() ? "assembled-operator" : op)
       << ":dofs=" << dofs;
    return os.str();
}

void appendSingleRunTrend(analysis::AnalysisSummarySet& summaries,
                          std::string metric_name,
                          const analysis::OperatorBlockId& block,
                          std::vector<analysis::VariableKey> variables,
                          Real value,
                          const OperatorTag& op,
                          GlobalIndex dofs,
                          bool explicit_uniform_bound = false)
{
    if (!std::isfinite(static_cast<double>(value))) {
        return;
    }
    analysis::RobustnessTrendSummary trend;
    trend.metric_name = std::move(metric_name);
    trend.block = block;
    trend.variables = std::move(variables);
    trend.sample_count = 1u;
    trend.run_ids.push_back(runIdFor(op, dofs));
    trend.dof_counts.push_back(dofs);
    trend.case_name = op.empty() ? "assembled-operator" : op;
    trend.mesh_revision = "current";
    trend.global_dof_count = dofs;
    trend.operator_tag = op;
    trend.parameter_hash = "unregistered";
    trend.min_value = value;
    trend.max_value = value;
    trend.explicit_uniform_lower_bound_present = explicit_uniform_bound;
    trend.explicit_uniform_lower_bound = explicit_uniform_bound ? value : Real{};
    trend.trend_class = analysis::RobustnessTrendClass::InsufficientSamples;
    summaries.robustness_trends.push_back(std::move(trend));
}

void appendAutomaticRobustnessTrends(analysis::AnalysisSummarySet& summaries,
                                     const OperatorTag& op,
                                     GlobalIndex dofs)
{
    for (const auto& summary : summaries.inf_sup_estimates) {
        appendSingleRunTrend(summaries,
                             "inf_sup_estimate",
                             summary.block,
                             {summary.primal_variable,
                              summary.multiplier_variable},
                             summary.estimate_value,
                             op,
                             dofs,
                             summary.uniform_lower_bound_value_present);
    }
    for (const auto& summary : summaries.schur_complements) {
        if (summary.condition_estimate_present) {
            appendSingleRunTrend(summaries,
                                 "schur_condition",
                                 summary.block,
                                 summary.variables,
                                 summary.condition_estimate,
                                 op,
                                 dofs);
        }
    }
    for (const auto& summary : summaries.discrete_matrices) {
        if (summary.condition_estimate) {
            appendSingleRunTrend(summaries,
                                 "matrix_condition",
                                 summary.block,
                                 blockVariables(summary.block),
                                 *summary.condition_estimate,
                                 op,
                                 dofs);
        } else if (summary.max_abs_entry > Real{}) {
            appendSingleRunTrend(summaries,
                                 "matrix_norm",
                                 summary.block,
                                 blockVariables(summary.block),
                                 summary.max_abs_entry,
                                 op,
                                 dofs);
        }
    }
    for (const auto& summary : summaries.stabilization_adequacy) {
        if (summary.stabilization_parameter_bounds_present) {
            appendSingleRunTrend(summaries,
                                 "stabilization_parameter_min",
                                 summary.block,
                                 summary.variables,
                                 summary.minimum_stabilization_parameter,
                                 op,
                                 dofs);
        }
    }
}

[[nodiscard]] analysis::NumericalErrorBudgetSummary
makeNumericalErrorBudgetSummary(
    const OperatorTag& op,
    const analysis::OperatorBlockId& block,
    const std::vector<analysis::VariableKey>& variables,
    const analysis::DiscreteMatrixSummary* matrix,
    const backends::SolverOptions* options)
{
    analysis::NumericalErrorBudgetSummary budget;
    budget.budget_id = op.empty() ? "assembled-error-budget" : op + ":error-budget";
    budget.block = block;
    budget.variables = variables;
    const Real eps = std::numeric_limits<Real>::epsilon();
    budget.matrix_norm_estimate =
        matrix ? std::max(matrix->max_abs_entry, Real{1}) : Real{1};
    budget.matrix_norm_present = matrix != nullptr;
    if (matrix && matrix->condition_estimate &&
        std::isfinite(static_cast<double>(*matrix->condition_estimate))) {
        budget.condition_estimate = *matrix->condition_estimate;
        budget.condition_estimate_present = true;
    } else if (matrix && matrix->coercivity_lower_bound &&
               *matrix->coercivity_lower_bound > Real{}) {
        budget.condition_estimate =
            budget.matrix_norm_estimate / *matrix->coercivity_lower_bound;
        budget.condition_estimate_present = true;
    } else {
        budget.condition_estimate = Real{1};
        budget.condition_estimate_present = false;
    }
    if (options != nullptr) {
        budget.linear_tolerance = options->rel_tol;
        budget.linear_tolerance_present = true;
        budget.verification_tolerance = options->abs_tol;
        budget.verification_tolerance_present = options->abs_tol > Real{};
    }
    const Real cond = std::max(budget.condition_estimate, Real{1});
    budget.machine_epsilon_amplification = eps * cond;
    budget.expected_absolute_floor =
        static_cast<Real>(100) * eps * cond * budget.matrix_norm_estimate;
    budget.expected_relative_floor = static_cast<Real>(100) * eps * cond;
    budget.recommended_verification_tolerance =
        std::max({budget.expected_absolute_floor,
                  budget.linear_tolerance_present
                      ? budget.linear_tolerance * budget.matrix_norm_estimate
                      : Real{},
                  Real{1.0e-14}});
    budget.recommended_tolerance_present = true;
    if (budget.verification_tolerance_present &&
        budget.verification_tolerance <
            static_cast<Real>(0.1) * budget.recommended_verification_tolerance) {
        budget.adequacy_class =
            analysis::ToleranceAdequacyClass::TooStrictForConditioning;
        budget.reason =
            "declared verification tolerance is below the conditioning-derived numerical floor";
    } else if (budget.verification_tolerance_present) {
        budget.adequacy_class = analysis::ToleranceAdequacyClass::Reasonable;
        budget.reason =
            "declared verification tolerance is consistent with the conditioning-derived numerical floor";
    } else {
        budget.adequacy_class = analysis::ToleranceAdequacyClass::Inconclusive;
        budget.reason =
            "no fixed verification tolerance was supplied for comparison";
    }
    return budget;
}

[[nodiscard]] analysis::OperatorBlockId boundaryConditionBlock(
    const analysis::BoundaryConditionDescriptor& bc,
    const OperatorTag& fallback_op)
{
    analysis::OperatorBlockId block;
    block.test_variables.push_back(bc.primary_variable);
    block.trial_variables.push_back(bc.primary_variable);
    block.domain = bc.domain;
    block.role = analysis::ContributionRole::BoundaryConstraint;
    block.operator_tag = bc.source.empty() ? fallback_op : bc.source;
    block.contribution_id = bc.source;
    if (bc.domain == analysis::DomainKind::InterfaceFace) {
        block.marker = bc.interface_marker;
    } else if (bc.domain == analysis::DomainKind::Boundary ||
               bc.domain == analysis::DomainKind::CoupledBoundary) {
        block.marker = bc.boundary_marker;
    }
    return block;
}

[[nodiscard]] analysis::TraceCapabilityFlags traceCoverageForBoundary(
    analysis::TraceKind trace_kind) noexcept
{
    switch (trace_kind) {
        case analysis::TraceKind::Value:
            return analysis::TraceCapabilityFlags::Value;
        case analysis::TraceKind::NormalComponent:
            return analysis::TraceCapabilityFlags::NormalComponent;
        case analysis::TraceKind::TangentialComponent:
            return analysis::TraceCapabilityFlags::TangentialComponent;
        case analysis::TraceKind::Flux:
        case analysis::TraceKind::NormalFlux:
            return analysis::TraceCapabilityFlags::NormalFlux;
        case analysis::TraceKind::Mixed:
            return analysis::TraceCapabilityFlags::Value |
                   analysis::TraceCapabilityFlags::NormalFlux;
        case analysis::TraceKind::AlgebraicRelation:
            return analysis::TraceCapabilityFlags::None;
    }
    return analysis::TraceCapabilityFlags::None;
}

[[nodiscard]] bool isWeakBoundaryEnforcement(
    analysis::EnforcementKind kind) noexcept
{
    return kind == analysis::EnforcementKind::WeakPenalty ||
           kind == analysis::EnforcementKind::WeakNitsche ||
           kind == analysis::EnforcementKind::WeakInequality;
}

[[nodiscard]] analysis::ParameterScaleRole parameterScaleRoleFor(
    const analysis::ContributionDescriptor& contribution) noexcept
{
    if (contribution.domain == analysis::DomainKind::Boundary ||
        contribution.domain == analysis::DomainKind::InterfaceFace ||
        contribution.domain == analysis::DomainKind::CoupledBoundary ||
        contribution.role == analysis::ContributionRole::BoundaryConstraint) {
        return analysis::ParameterScaleRole::WeakBoundaryPenalty;
    }
    if (contribution.temporal && contribution.temporal->derivative_order > 0) {
        return analysis::ParameterScaleRole::CflLike;
    }
    if (contribution.transport_character &&
        *contribution.transport_character != analysis::TransportCharacter::None) {
        return analysis::ParameterScaleRole::PecletLike;
    }
    if (contribution.scaling && contribution.scaling->dt_power != 0) {
        return analysis::ParameterScaleRole::CflLike;
    }
    return analysis::ParameterScaleRole::Generic;
}

[[nodiscard]] int polynomialOrderForVariable(const FieldRegistry& registry,
                                             const analysis::VariableKey& variable)
{
    if (variable.kind != analysis::VariableKind::FieldComponent ||
        variable.field_id == INVALID_FIELD_ID) {
        return -1;
    }
    for (const auto& record : registry.records()) {
        if (record.id == variable.field_id && record.space) {
            return record.space->polynomial_order();
        }
    }
    return -1;
}

[[nodiscard]] int polynomialOrderForVariables(
    const FieldRegistry& registry,
    const std::vector<analysis::VariableKey>& variables)
{
    int order = -1;
    for (const auto& variable : variables) {
        order = std::max(order, polynomialOrderForVariable(registry, variable));
    }
    return order;
}

[[nodiscard]] analysis::ParameterScaleSummary makeParameterScaleSummary(
    std::string id,
    analysis::ParameterScaleRole role,
    analysis::OperatorBlockId block,
    std::vector<analysis::VariableKey> variables,
    const analysis::ScalingDescriptor& scaling,
    const FieldRegistry& registry)
{
    analysis::ParameterScaleSummary summary;
    summary.nondimensional_parameter_id = std::move(id);
    summary.role = role;
    summary.block = std::move(block);
    summary.variables = std::move(variables);
    summary.contribution_id = summary.block.contribution_id;
    summary.min_scale_value = static_cast<Real>(1.0);
    summary.max_scale_value = static_cast<Real>(1.0);
    summary.polynomial_order =
        polynomialOrderForVariables(registry, summary.variables);
    summary.scale_theorem_id = "ScalingDescriptor";
    summary.mesh_quality_factor = static_cast<Real>(1.0);
    summary.coefficient_contrast_factor =
        scaling.coefficient_scaled ? static_cast<Real>(1.0) : static_cast<Real>(0.0);
    summary.trace_inverse_metadata_present =
        role == analysis::ParameterScaleRole::WeakBoundaryPenalty &&
        scaling.h_power < 0;
    if (summary.trace_inverse_metadata_present) {
        summary.trace_inverse_constant = static_cast<Real>(1.0);
    }
    return summary;
}

[[nodiscard]] analysis::TensorRank tensorRankFor(
    analysis::FormCoefficientRank rank) noexcept
{
    switch (rank) {
        case analysis::FormCoefficientRank::Scalar:
            return analysis::TensorRank::Scalar;
        case analysis::FormCoefficientRank::Vector:
            return analysis::TensorRank::Vector;
        case analysis::FormCoefficientRank::Rank2Tensor:
            return analysis::TensorRank::Rank2Tensor;
        case analysis::FormCoefficientRank::Rank3Tensor:
            return analysis::TensorRank::Rank3Tensor;
        case analysis::FormCoefficientRank::Rank4Tensor:
            return analysis::TensorRank::Rank4Tensor;
        case analysis::FormCoefficientRank::Unknown:
            return analysis::TensorRank::Unknown;
    }
    return analysis::TensorRank::Unknown;
}

[[nodiscard]] analysis::TensorRank tensorRankFromMetadataString(
    const std::string& value) noexcept
{
    if (value == "scalar" || value == "Scalar") {
        return analysis::TensorRank::Scalar;
    }
    if (value == "vector" || value == "Vector") {
        return analysis::TensorRank::Vector;
    }
    if (value == "rank2" || value == "Rank2Tensor" || value == "tensor2") {
        return analysis::TensorRank::Rank2Tensor;
    }
    if (value == "rank3" || value == "Rank3Tensor" || value == "tensor3") {
        return analysis::TensorRank::Rank3Tensor;
    }
    if (value == "rank4" || value == "Rank4Tensor" || value == "tensor4") {
        return analysis::TensorRank::Rank4Tensor;
    }
    return analysis::TensorRank::Unknown;
}

[[nodiscard]] analysis::SymmetryClass symmetryFromMetadataString(
    const std::string& value) noexcept
{
    if (value == "symmetric" || value == "Symmetric") {
        return analysis::SymmetryClass::Symmetric;
    }
    if (value == "skew" || value == "Skew") {
        return analysis::SymmetryClass::Skew;
    }
    if (value == "nonsymmetric" || value == "Nonsymmetric") {
        return analysis::SymmetryClass::Nonsymmetric;
    }
    if (value == "not_applicable" || value == "NotApplicable") {
        return analysis::SymmetryClass::NotApplicable;
    }
    return analysis::SymmetryClass::Unknown;
}

[[nodiscard]] analysis::PositivityClass positivityFromMetadataString(
    const std::string& value) noexcept
{
    if (value == "positive" || value == "Positive") {
        return analysis::PositivityClass::Positive;
    }
    if (value == "nonnegative" || value == "Nonnegative") {
        return analysis::PositivityClass::Nonnegative;
    }
    if (value == "negative" || value == "Negative") {
        return analysis::PositivityClass::Negative;
    }
    if (value == "nonpositive" || value == "Nonpositive") {
        return analysis::PositivityClass::Nonpositive;
    }
    if (value == "indefinite" || value == "Indefinite") {
        return analysis::PositivityClass::Indefinite;
    }
    return analysis::PositivityClass::Unknown;
}

[[nodiscard]] analysis::PositivityClass positivityFromRange(Real min_value,
                                                           Real max_value,
                                                           Real tolerance) noexcept
{
    if (!std::isfinite(static_cast<double>(min_value)) ||
        !std::isfinite(static_cast<double>(max_value))) {
        return analysis::PositivityClass::Unknown;
    }
    if (min_value > tolerance) {
        return analysis::PositivityClass::Positive;
    }
    if (min_value >= -tolerance) {
        return analysis::PositivityClass::Nonnegative;
    }
    if (max_value < -tolerance) {
        return analysis::PositivityClass::Negative;
    }
    if (max_value <= tolerance) {
        return analysis::PositivityClass::Nonpositive;
    }
    return analysis::PositivityClass::Indefinite;
}

struct RuntimeCoefficientNode {
    const forms::FormExprNode* node{nullptr};
    std::string name;
    analysis::FormCoefficientRank rank{analysis::FormCoefficientRank::Unknown};
    bool time_dependent{false};
    analysis::DomainKind domain{analysis::DomainKind::Cell};
    int boundary_marker{-1};
    int interface_marker{-1};
};

void collectRuntimeCoefficientNodes(
    const forms::FormExprNode& node,
    analysis::DomainKind domain,
    int boundary_marker,
    int interface_marker,
    std::vector<RuntimeCoefficientNode>& out)
{
    using FT = forms::FormExprType;

    auto child_domain = domain;
    auto child_boundary_marker = boundary_marker;
    auto child_interface_marker = interface_marker;
    if (node.type() == FT::CellIntegral) {
        child_domain = analysis::DomainKind::Cell;
        child_boundary_marker = -1;
        child_interface_marker = -1;
    } else if (node.type() == FT::BoundaryIntegral) {
        child_domain = analysis::DomainKind::Boundary;
        child_boundary_marker = node.boundaryMarker().value_or(-1);
        child_interface_marker = -1;
    } else if (node.type() == FT::InteriorFaceIntegral) {
        child_domain = analysis::DomainKind::InteriorFace;
        child_boundary_marker = -1;
        child_interface_marker = -1;
    } else if (node.type() == FT::InterfaceIntegral) {
        child_domain = analysis::DomainKind::InterfaceFace;
        child_boundary_marker = -1;
        child_interface_marker = node.interfaceMarker().value_or(-1);
    }

    if (node.type() == FT::Coefficient) {
        RuntimeCoefficientNode entry;
        entry.node = &node;
        entry.name = node.toString();
        if (node.scalarCoefficient() || node.timeScalarCoefficient()) {
            entry.rank = analysis::FormCoefficientRank::Scalar;
        } else if (node.vectorCoefficient()) {
            entry.rank = analysis::FormCoefficientRank::Vector;
        } else if (node.matrixCoefficient()) {
            entry.rank = analysis::FormCoefficientRank::Rank2Tensor;
        } else if (node.tensor3Coefficient()) {
            entry.rank = analysis::FormCoefficientRank::Rank3Tensor;
        } else if (node.tensor4Coefficient()) {
            entry.rank = analysis::FormCoefficientRank::Rank4Tensor;
        }
        entry.time_dependent = node.timeScalarCoefficient() != nullptr;
        entry.domain = domain;
        entry.boundary_marker = boundary_marker;
        entry.interface_marker = interface_marker;

        const auto duplicate = std::find_if(
            out.begin(), out.end(), [&](const RuntimeCoefficientNode& existing) {
                return existing.node == entry.node ||
                       (existing.name == entry.name &&
                        existing.rank == entry.rank &&
                        existing.domain == entry.domain &&
                        existing.boundary_marker == entry.boundary_marker &&
                        existing.interface_marker == entry.interface_marker);
            });
        if (duplicate == out.end()) {
            out.push_back(entry);
        }
    }

    for (const auto& child : node.childrenShared()) {
        if (child) {
            collectRuntimeCoefficientNodes(*child,
                                           child_domain,
                                           child_boundary_marker,
                                           child_interface_marker,
                                           out);
        }
    }
}

[[nodiscard]] Real cellDiameterFromCoordinates(
    const std::vector<std::array<Real, 3>>& coords) noexcept
{
    Real h_sq = Real{};
    for (std::size_t i = 0; i < coords.size(); ++i) {
        for (std::size_t j = i + 1; j < coords.size(); ++j) {
            Real d_sq = Real{};
            for (int k = 0; k < 3; ++k) {
                const Real d = coords[i][k] - coords[j][k];
                d_sq += d * d;
            }
            h_sq = std::max(h_sq, d_sq);
        }
    }
    return std::sqrt(h_sq);
}

template <class Callback>
void forEachOwnedQuadraturePoint(const assembly::IMeshAccess& mesh,
                                 Callback&& callback)
{
    mesh.forEachOwnedCell([&](GlobalIndex cell_id) {
        try {
            std::vector<std::array<Real, 3>> coords;
            mesh.getCellCoordinates(cell_id, coords);
            if (coords.empty()) {
                return;
            }

            std::vector<math::Vector<Real, 3>> nodes;
            nodes.reserve(coords.size());
            for (const auto& coord : coords) {
                nodes.emplace_back(math::Vector<Real, 3>{
                    coord[0], coord[1], coord[2]});
            }

            elements::ElementRequest element_request;
            element_request.element_type = mesh.getCellType(cell_id);
            element_request.basis_type = BasisType::Lagrange;
            element_request.field_type = FieldType::Scalar;
            element_request.continuity = Continuity::C0;
            const int geometry_order = mesh.getCellGeometryOrder(cell_id);
            element_request.order = geometry_order;
            auto element = elements::ElementFactory::create(element_request);
            if (!element) {
                return;
            }

            geometry::MappingRequest mapping_request;
            mapping_request.element_type = element_request.element_type;
            mapping_request.geometry_order = geometry_order;
            mapping_request.use_affine = geometry_order <= 1;
            auto mapping = geometry::MappingFactory::create(mapping_request, nodes);
            if (!mapping) {
                return;
            }

            const auto quadrature = element->quadrature();
            if (!quadrature) {
                return;
            }
            const Real h = cellDiameterFromCoordinates(coords);
            for (const auto& qp : quadrature->points()) {
                const auto x = mapping->map_to_physical(qp);
                callback(std::array<Real, 3>{x[0], x[1], x[2]}, h);
            }
        } catch (const std::exception&) {
            return;
        }
    });
}

struct MeshScaleStats {
    bool available{false};
    Real h_min{std::numeric_limits<Real>::infinity()};
    Real h_max{Real{}};
    std::uint64_t sample_count{0u};
};

[[nodiscard]] MeshScaleStats meshScaleStats(const assembly::IMeshAccess* mesh)
{
    MeshScaleStats stats;
    if (mesh == nullptr) {
        return stats;
    }
    mesh->forEachOwnedCell([&](GlobalIndex cell_id) {
        try {
            std::vector<std::array<Real, 3>> coords;
            mesh->getCellCoordinates(cell_id, coords);
            const Real h = cellDiameterFromCoordinates(coords);
            if (std::isfinite(static_cast<double>(h)) && h > Real{}) {
                stats.h_min = std::min(stats.h_min, h);
                stats.h_max = std::max(stats.h_max, h);
                ++stats.sample_count;
            }
        } catch (const std::exception&) {
            return;
        }
    });
#if FE_HAS_MPI
    stats.sample_count = mpiSumUint64(stats.sample_count);
    stats.h_min = static_cast<Real>(mpiReduceDouble(
        static_cast<double>(stats.h_min), MPI_MIN));
    stats.h_max = static_cast<Real>(mpiReduceDouble(
        static_cast<double>(stats.h_max), MPI_MAX));
#endif
    stats.available = stats.sample_count > 0u &&
                      std::isfinite(static_cast<double>(stats.h_min)) &&
                      std::isfinite(static_cast<double>(stats.h_max)) &&
                      stats.h_min > Real{};
    if (!stats.available) {
        stats.h_min = Real{};
        stats.h_max = Real{};
    }
    return stats;
}

[[nodiscard]] std::array<double, 3> symmetricEigenvalues3x3(
    std::array<std::array<Real, 3>, 3> input) noexcept
{
    std::array<std::array<double, 3>, 3> a{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            a[i][j] = static_cast<double>(input[i][j]);
        }
    }

    for (int iter = 0; iter < 32; ++iter) {
        int p = 0;
        int q = 1;
        double max_offdiag = std::abs(a[p][q]);
        for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 3; ++j) {
                const double value = std::abs(a[i][j]);
                if (value > max_offdiag) {
                    max_offdiag = value;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_offdiag < 1.0e-14) {
            break;
        }

        const double app = a[p][p];
        const double aqq = a[q][q];
        const double apq = a[p][q];
        const double tau = (aqq - app) / (2.0 * apq);
        const double t = (tau >= 0.0 ? 1.0 : -1.0) /
                         (std::abs(tau) + std::sqrt(1.0 + tau * tau));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;

        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for (int k = 0; k < 3; ++k) {
            if (k == p || k == q) {
                continue;
            }
            const double akp = a[k][p];
            const double akq = a[k][q];
            a[k][p] = c * akp - s * akq;
            a[p][k] = a[k][p];
            a[k][q] = s * akp + c * akq;
            a[q][k] = a[k][q];
        }
    }

    std::array<double, 3> values{a[0][0], a[1][1], a[2][2]};
    std::sort(values.begin(), values.end());
    return values;
}

struct CoefficientSampleStats {
    std::uint64_t sample_count{0u};
    std::uint64_t nonfinite_count{0u};
    std::uint64_t symmetry_violation_count{0u};
    Real min_value{std::numeric_limits<Real>::infinity()};
    Real max_value{-std::numeric_limits<Real>::infinity()};
    Real min_abs_positive{std::numeric_limits<Real>::infinity()};
    Real max_abs{Real{}};
};

void updateCoefficientRange(CoefficientSampleStats& stats, Real value)
{
    if (!std::isfinite(static_cast<double>(value))) {
        ++stats.nonfinite_count;
        return;
    }
    ++stats.sample_count;
    stats.min_value = std::min(stats.min_value, value);
    stats.max_value = std::max(stats.max_value, value);
    const Real abs_value = std::abs(value);
    stats.max_abs = std::max(stats.max_abs, abs_value);
    if (abs_value > Real{}) {
        stats.min_abs_positive = std::min(stats.min_abs_positive, abs_value);
    }
}

[[nodiscard]] Real vectorNorm3(const std::array<Real, 3>& v) noexcept
{
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

template <std::size_t N>
[[nodiscard]] Real arrayNorm(const std::array<Real, N>& v) noexcept
{
    Real sum = Real{};
    for (Real entry : v) {
        sum += entry * entry;
    }
    return std::sqrt(sum);
}

[[nodiscard]] Real matrixFrobeniusNorm(
    const std::array<std::array<Real, 3>, 3>& matrix) noexcept
{
    Real sum = Real{};
    for (const auto& row : matrix) {
        for (Real entry : row) {
            sum += entry * entry;
        }
    }
    return std::sqrt(sum);
}

[[nodiscard]] bool matrixIsSymmetric(
    const std::array<std::array<Real, 3>, 3>& matrix,
    Real tolerance) noexcept
{
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            if (std::abs(matrix[i][j] - matrix[j][i]) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] CoefficientSampleStats sampleCoefficientNode(
    const RuntimeCoefficientNode& coefficient,
    const assembly::IMeshAccess* mesh,
    const SystemStateView* state)
{
    CoefficientSampleStats stats;
    if (coefficient.node == nullptr || mesh == nullptr) {
        return stats;
    }

    const Real time = state != nullptr ? state->time : Real{};
    forEachOwnedQuadraturePoint(*mesh, [&](const std::array<Real, 3>& x, Real) {
        const auto* node = coefficient.node;
        if (const auto* f = node->scalarCoefficient()) {
            updateCoefficientRange(stats, (*f)(x[0], x[1], x[2]));
        } else if (const auto* f = node->timeScalarCoefficient()) {
            updateCoefficientRange(stats, (*f)(x[0], x[1], x[2], time));
        } else if (const auto* f = node->vectorCoefficient()) {
            updateCoefficientRange(stats, vectorNorm3((*f)(x[0], x[1], x[2])));
        } else if (const auto* f = node->matrixCoefficient()) {
            const auto matrix = (*f)(x[0], x[1], x[2]);
            if (matrixIsSymmetric(matrix, static_cast<Real>(1.0e-10))) {
                const auto eig = symmetricEigenvalues3x3(matrix);
                updateCoefficientRange(stats, static_cast<Real>(eig[0]));
                stats.max_value =
                    std::max(stats.max_value, static_cast<Real>(eig[2]));
                stats.max_abs =
                    std::max(stats.max_abs,
                             static_cast<Real>(std::max(std::abs(eig[0]),
                                                        std::abs(eig[2]))));
                if (std::abs(eig[0]) > 0.0) {
                    stats.min_abs_positive =
                        std::min(stats.min_abs_positive,
                                 static_cast<Real>(std::abs(eig[0])));
                }
            } else {
                ++stats.symmetry_violation_count;
                updateCoefficientRange(stats, matrixFrobeniusNorm(matrix));
            }
        } else if (const auto* f = node->tensor3Coefficient()) {
            updateCoefficientRange(stats, arrayNorm((*f)(x[0], x[1], x[2])));
        } else if (const auto* f = node->tensor4Coefficient()) {
            updateCoefficientRange(stats, arrayNorm((*f)(x[0], x[1], x[2])));
        }
    });

#if FE_HAS_MPI
    stats.sample_count = mpiSumUint64(stats.sample_count);
    stats.nonfinite_count = mpiSumUint64(stats.nonfinite_count);
    stats.symmetry_violation_count =
        mpiSumUint64(stats.symmetry_violation_count);
    stats.min_value = static_cast<Real>(mpiReduceDouble(
        static_cast<double>(stats.min_value), MPI_MIN));
    stats.max_value = static_cast<Real>(mpiReduceDouble(
        static_cast<double>(stats.max_value), MPI_MAX));
    stats.min_abs_positive = static_cast<Real>(mpiReduceDouble(
        static_cast<double>(stats.min_abs_positive), MPI_MIN));
    stats.max_abs = static_cast<Real>(mpiReduceDouble(
        static_cast<double>(stats.max_abs), MPI_MAX));
#endif

    if (!std::isfinite(static_cast<double>(stats.min_value))) {
        stats.min_value = Real{};
    }
    if (!std::isfinite(static_cast<double>(stats.max_value))) {
        stats.max_value = Real{};
    }
    if (!std::isfinite(static_cast<double>(stats.min_abs_positive))) {
        stats.min_abs_positive = Real{};
    }
    return stats;
}

[[nodiscard]] analysis::CoefficientPropertySummary
makeCoefficientSummaryFromRuntimeNode(
    const RuntimeCoefficientNode& coefficient,
    const analysis::ContributionDescriptor& contribution,
    const OperatorTag& fallback_op,
    const assembly::IMeshAccess* mesh,
    const SystemStateView* state)
{
    constexpr Real tolerance = static_cast<Real>(1.0e-12);

    analysis::CoefficientPropertySummary summary;
    summary.coefficient = coefficient.name;
    summary.block = contributionBlock(contribution, fallback_op);
    summary.variables = contributionVariables(contribution);
    summary.contribution_id = contributionIdentity(contribution, fallback_op);
    summary.tensor_rank = tensorRankFor(coefficient.rank);
    summary.domain = coefficient.domain;
    summary.time_dependent = coefficient.time_dependent;
    summary.positivity_tolerance = tolerance;
    summary.tolerance_metadata_present = true;
    summary.coverage_scope = "owned-cell quadrature runtime sampling";
    summary.producer_certificate_id = "FE::Forms::DAGRuntimeCoefficientSampler";

    if (coefficient.rank == analysis::FormCoefficientRank::Scalar) {
        summary.symmetry = analysis::SymmetryClass::NotApplicable;
    } else if (coefficient.rank == analysis::FormCoefficientRank::Rank2Tensor) {
        summary.symmetry = analysis::SymmetryClass::Symmetric;
    } else {
        summary.symmetry = analysis::SymmetryClass::Unknown;
    }

    const auto stats = sampleCoefficientNode(coefficient, mesh, state);
    if (stats.sample_count > 0u) {
        summary.min_eigenvalue = stats.min_value;
        summary.max_eigenvalue = stats.max_value;
        summary.local_symmetric_part_min_eigenvalue = stats.min_value;
        summary.local_symmetric_part_max_eigenvalue = stats.max_value;
        summary.local_skew_or_nonsymmetric_norm =
            static_cast<Real>(stats.symmetry_violation_count);
        summary.local_spectrum_sample_count = stats.sample_count;
        summary.local_spectrum_nonfinite_count = stats.nonfinite_count;
        summary.local_spectrum_coverage_present = true;
        summary.coefficient_region_coverage_complete = true;
        summary.quadrature_point_coverage_complete = true;
        summary.state_sample_coverage_complete = true;
        if (coefficient.rank == analysis::FormCoefficientRank::Rank2Tensor &&
            stats.symmetry_violation_count > 0u) {
            summary.symmetry = analysis::SymmetryClass::Nonsymmetric;
            summary.positivity = analysis::PositivityClass::Unknown;
            summary.lower_bound_valid_for_all_samples = false;
        } else if (coefficient.rank == analysis::FormCoefficientRank::Scalar ||
                   coefficient.rank == analysis::FormCoefficientRank::Rank2Tensor) {
            summary.positivity =
                positivityFromRange(stats.min_value, stats.max_value, tolerance);
            summary.lower_bound_valid_for_all_samples =
                stats.nonfinite_count == 0u &&
                summary.positivity != analysis::PositivityClass::Unknown;
        } else {
            summary.positivity = analysis::PositivityClass::Unknown;
            summary.lower_bound_valid_for_all_samples = false;
        }
        if (stats.min_abs_positive > Real{}) {
            summary.anisotropy_ratio = stats.max_abs / stats.min_abs_positive;
            summary.contrast_ratio = summary.anisotropy_ratio;
        }
        summary.local_spectrum_coverage_complete =
            summary.coefficient_region_coverage_complete &&
            stats.nonfinite_count == 0u;
        summary.local_spectrum_quadrature_coverage_complete =
            summary.quadrature_point_coverage_complete;
        if (stats.nonfinite_count > 0u) {
            summary.worst_local_spectrum_sample.note =
                "nonfinite runtime coefficient sample";
        }
    } else {
        summary.positivity = analysis::PositivityClass::Unknown;
        summary.lower_bound_valid_for_all_samples = false;
    }

    return summary;
}

[[nodiscard]] analysis::CoefficientPropertySummary
makeCoefficientSummaryFromConstitutiveLaw(
    const analysis::ConstitutiveLawMetadata& law,
    const OperatorTag& fallback_op)
{
    constexpr Real tolerance = static_cast<Real>(1.0e-12);

    analysis::CoefficientPropertySummary summary;
    summary.coefficient = law.name.empty() ? "constitutive_law" : law.name;
    summary.block.operator_tag = law.source_operator_tag.empty()
        ? fallback_op
        : law.source_operator_tag;
    summary.block.domain = analysis::DomainKind::Cell;
    summary.block.role = analysis::ContributionRole::DiagonalBlock;
    if (law.primary_field != INVALID_FIELD_ID) {
        const auto variable = analysis::VariableKey::field(law.primary_field);
        summary.variables.push_back(variable);
        summary.block.test_variables.push_back(variable);
        summary.block.trial_variables.push_back(variable);
    }
    summary.contribution_id = summary.block.operator_tag + ":constitutive:" +
                              summary.coefficient;
    summary.tensor_rank = law.tensor_rank.empty()
        ? analysis::TensorRank::Scalar
        : tensorRankFromMetadataString(law.tensor_rank);
    summary.symmetry = law.symmetry_class.empty()
        ? (summary.tensor_rank == analysis::TensorRank::Scalar
               ? analysis::SymmetryClass::NotApplicable
               : analysis::SymmetryClass::Unknown)
        : symmetryFromMetadataString(law.symmetry_class);
    summary.positivity = law.positivity_class.empty()
        ? analysis::PositivityClass::Unknown
        : positivityFromMetadataString(law.positivity_class);
    summary.domain = analysis::DomainKind::Cell;
    summary.state_dependent = law.state_dependent;
    summary.time_dependent = law.time_dependent;
    summary.positivity_tolerance = law.positivity_tolerance_present
        ? law.positivity_tolerance
        : tolerance;
    summary.tolerance_metadata_present =
        law.positivity_tolerance_present || law.constant_value_available;
    summary.robustness_theorem_id = law.robustness_theorem_id;
    summary.robustness_norm_id = law.robustness_norm_id;
    summary.robustness_parameter_range_scope =
        law.robustness_parameter_range_scope;
    summary.robustness_mesh_family_scope = law.robustness_mesh_family_scope;
    summary.robustness_uniform_constant_present =
        law.robustness_uniform_constant_present;
    summary.robustness_uniform_constant = law.robustness_uniform_constant;
    summary.robustness_certificate_present =
        !law.robustness_theorem_id.empty();
    summary.robustness_norm_metadata_present =
        !law.robustness_norm_id.empty();
    summary.robustness_parameter_range_metadata_present =
        !law.robustness_parameter_range_scope.empty();
    summary.robustness_mesh_family_metadata_present =
        !law.robustness_mesh_family_scope.empty();
    summary.coverage_scope = law.constant_value_available
        ? "constant constitutive metadata"
        : "constitutive metadata";
    summary.producer_certificate_id = "FE::ConstitutiveLawMetadata";

    if (law.constant_value_available) {
        summary.min_eigenvalue = law.constant_value;
        summary.max_eigenvalue = law.constant_value;
        summary.local_symmetric_part_min_eigenvalue = law.constant_value;
        summary.local_symmetric_part_max_eigenvalue = law.constant_value;
        summary.local_spectrum_sample_count = 1u;
        summary.local_spectrum_coverage_present = true;
        summary.local_spectrum_coverage_complete = true;
        summary.local_spectrum_quadrature_coverage_complete = true;
        summary.positivity =
            positivityFromRange(law.constant_value,
                                law.constant_value,
                                summary.positivity_tolerance);
        summary.coefficient_region_coverage_complete = true;
        summary.quadrature_point_coverage_complete = true;
        summary.state_sample_coverage_complete = true;
        summary.lower_bound_valid_for_all_samples =
            summary.positivity != analysis::PositivityClass::Unknown;
        const Real abs_value = std::abs(law.constant_value);
        summary.anisotropy_ratio = abs_value > Real{} ? static_cast<Real>(1) : Real{};
        summary.contrast_ratio = summary.anisotropy_ratio;
        summary.tolerance_metadata_present = true;
    }

    return summary;
}

[[nodiscard]] analysis::CoefficientPropertySummary
makeCoefficientSummaryFromParameterUsage(
    const analysis::FormParameterUsage& usage,
    const analysis::ContributionDescriptor& contribution,
    const OperatorTag& fallback_op,
    const ParameterRegistry& parameter_registry,
    const SystemStateView* state)
{
    constexpr Real tolerance = static_cast<Real>(1.0e-12);

    std::string key = usage.name;
    if (key.empty() && usage.slot.has_value()) {
        if (auto slot_key = parameter_registry.keyForSlot(*usage.slot)) {
            key = std::string(*slot_key);
        }
    }

    analysis::CoefficientPropertySummary summary;
    summary.coefficient = key.empty()
        ? "parameter_slot:" + std::to_string(usage.slot.value_or(0u))
        : "parameter:" + key;
    summary.block = contributionBlock(contribution, fallback_op);
    summary.variables = contributionVariables(contribution);
    summary.contribution_id = contributionIdentity(contribution, fallback_op);
    summary.tensor_rank = analysis::TensorRank::Scalar;
    summary.symmetry = analysis::SymmetryClass::NotApplicable;
    summary.domain = usage.domain;
    summary.positivity_tolerance = tolerance;
    summary.tolerance_metadata_present = true;
    summary.coverage_scope = "runtime parameter value";
    summary.producer_certificate_id = "FE::ParameterRegistryRuntimeMetadata";

    std::optional<Real> value;
    if (state != nullptr && !key.empty()) {
        auto getter = parameter_registry.makeRealGetter(*state);
        value = getter(key);
    }
    if (value.has_value()) {
        summary.min_eigenvalue = *value;
        summary.max_eigenvalue = *value;
        summary.local_symmetric_part_min_eigenvalue = *value;
        summary.local_symmetric_part_max_eigenvalue = *value;
        summary.local_spectrum_sample_count = 1u;
        summary.local_spectrum_coverage_present = true;
        summary.local_spectrum_coverage_complete = true;
        summary.local_spectrum_quadrature_coverage_complete = true;
        summary.positivity = positivityFromRange(*value, *value, tolerance);
        summary.coefficient_region_coverage_complete = true;
        summary.quadrature_point_coverage_complete = true;
        summary.state_sample_coverage_complete = true;
        summary.lower_bound_valid_for_all_samples =
            summary.positivity != analysis::PositivityClass::Unknown;
        summary.anisotropy_ratio = std::abs(*value) > Real{}
            ? static_cast<Real>(1)
            : Real{};
        summary.contrast_ratio = summary.anisotropy_ratio;
    } else {
        summary.positivity = analysis::PositivityClass::Unknown;
        summary.lower_bound_valid_for_all_samples = false;
    }

    if (!key.empty()) {
        if (const auto* spec = parameter_registry.find(key)) {
            summary.robustness_theorem_id = spec->theorem_id;
            summary.robustness_parameter_range_scope =
                spec->admissible_range_scope;
            summary.robustness_certificate_present =
                !spec->theorem_id.empty();
            summary.robustness_parameter_range_metadata_present =
                spec->lower_bound.has_value() ||
                spec->upper_bound.has_value() ||
                !spec->admissible_range_scope.empty();
            if (spec->lower_bound.has_value() &&
                value.has_value() &&
                *value >= *spec->lower_bound - tolerance) {
                summary.lower_bound_valid_for_all_samples = true;
            }
        }
    }

    return summary;
}

void addCoefficientSummaryIfAbsent(
    std::vector<analysis::CoefficientPropertySummary>& summaries,
    analysis::CoefficientPropertySummary summary)
{
    const auto duplicate = std::find_if(
        summaries.begin(), summaries.end(),
        [&](const analysis::CoefficientPropertySummary& existing) {
            return existing.coefficient == summary.coefficient &&
                   existing.contribution_id == summary.contribution_id &&
                   existing.block.operator_tag == summary.block.operator_tag;
        });
    if (duplicate == summaries.end()) {
        summaries.push_back(std::move(summary));
    }
}

void applyGlobalSpectrumFallback(
    std::vector<analysis::CoefficientPropertySummary>& coefficient_summaries,
    const analysis::DiscreteMatrixSummary* matrix)
{
    if (matrix == nullptr) {
        return;
    }
    const Real lower = matrix->min_eigenvalue_estimate.value_or(
        matrix->coercivity_lower_bound.value_or(Real{}));
    const Real upper = std::max(matrix->max_abs_entry, std::abs(lower));
    for (auto& summary : coefficient_summaries) {
        if (summary.local_spectrum_coverage_present) {
            continue;
        }
        if (!summary.block.operator_tag.empty() &&
            !matrix->block.operator_tag.empty() &&
            summary.block.operator_tag != matrix->block.operator_tag) {
            continue;
        }
        summary.local_symmetric_part_min_eigenvalue = lower;
        summary.local_symmetric_part_max_eigenvalue = upper;
        summary.local_skew_or_nonsymmetric_norm =
            matrix->nonsymmetry_indicator.value_or(Real{});
        summary.local_spectrum_sample_count = matrix->scanned_row_count;
        summary.local_spectrum_nonfinite_count = matrix->nonfinite_entry_count;
        summary.local_spectrum_coverage_present = matrix->scanned_row_count > 0u;
        summary.local_spectrum_coverage_complete = matrix->sign_evidence_complete;
        summary.local_spectrum_quadrature_coverage_complete = false;
        summary.local_spectrum_fallback_global = true;
        summary.worst_local_spectrum_sample.note =
            "fallback global assembled symmetric-part estimate";
    }
}

[[nodiscard]] analysis::CoefficientPropertySummary
makeImplicitFormCoefficientSummary(
    const std::vector<analysis::ContributionDescriptor>& contributions,
    const std::vector<analysis::FormulationRecord>& formulations,
    const OperatorTag& op)
{
    analysis::CoefficientPropertySummary summary;
    summary.coefficient = op.empty()
        ? "implicit_form_coefficients"
        : "implicit_form_coefficients:" + op;
    summary.block.operator_tag = op;
    summary.block.domain = analysis::DomainKind::Cell;
    summary.block.role = analysis::ContributionRole::DiagonalBlock;
    summary.domain = analysis::DomainKind::Cell;
    summary.tensor_rank = analysis::TensorRank::Unknown;
    summary.symmetry = analysis::SymmetryClass::Unknown;
    summary.positivity = analysis::PositivityClass::Unknown;
    summary.positivity_tolerance = static_cast<Real>(1.0e-12);
    summary.coverage_scope =
        "implicit form coefficients; no explicit coefficient or constitutive metadata recovered from DAG";
    summary.producer_certificate_id = "FE::ImplicitFormCoefficientMetadata";

    auto append_variable = [](std::vector<analysis::VariableKey>& variables,
                              const analysis::VariableKey& variable) {
        if (std::find(variables.begin(), variables.end(), variable) ==
            variables.end()) {
            variables.push_back(variable);
        }
    };

    for (const auto& contribution : contributions) {
        if (!op.empty() &&
            !contribution.operator_tag.empty() &&
            contribution.operator_tag != op) {
            continue;
        }
        for (const auto& variable : contributionVariables(contribution)) {
            append_variable(summary.variables, variable);
        }
        if (contribution.domain != analysis::DomainKind::Cell) {
            continue;
        }
        for (const auto& variable : contribution.test_variables) {
            append_variable(summary.block.test_variables, variable);
        }
        for (const auto& variable : contribution.trial_variables) {
            append_variable(summary.block.trial_variables, variable);
        }
    }

    for (const auto& formulation : formulations) {
        if (!op.empty() &&
            !formulation.operator_tag.empty() &&
            formulation.operator_tag != op) {
            continue;
        }
        for (const auto& variable : formulation.active_variables) {
            append_variable(summary.variables, variable);
        }
    }

    if (summary.block.test_variables.empty()) {
        summary.block.test_variables = summary.variables;
    }
    if (summary.block.trial_variables.empty()) {
        summary.block.trial_variables = summary.variables;
    }

    return summary;
}

[[nodiscard]] std::vector<analysis::CoefficientPropertySummary>
buildRuntimeCoefficientSummaries(
    const std::vector<analysis::ContributionDescriptor>& contributions,
    const std::vector<analysis::FormulationRecord>& formulations,
    const OperatorTag& op,
    const ParameterRegistry& parameter_registry,
    const assembly::IMeshAccess* mesh,
    const SystemStateView* state)
{
    std::vector<analysis::CoefficientPropertySummary> out;

    for (const auto& contribution : contributions) {
        std::vector<RuntimeCoefficientNode> coefficients;
        if (contribution.source_expression) {
            collectRuntimeCoefficientNodes(*contribution.source_expression,
                                           analysis::DomainKind::Cell,
                                           -1,
                                           -1,
                                           coefficients);
        }
        for (const auto& coefficient : coefficients) {
            addCoefficientSummaryIfAbsent(
                out,
                makeCoefficientSummaryFromRuntimeNode(coefficient,
                                                      contribution,
                                                      op,
                                                      mesh,
                                                      state));
        }
        if (contribution.source_expression) {
            auto scan = analysis::scanFormExpr(*contribution.source_expression);
            for (auto& law : scan.constitutive_laws) {
                if (law.source_operator_tag.empty()) {
                    law.source_operator_tag =
                        contribution.operator_tag.empty()
                            ? op
                            : contribution.operator_tag;
                }
                auto summary = makeCoefficientSummaryFromConstitutiveLaw(law, op);
                summary.block = contributionBlock(contribution, op);
                summary.variables = contributionVariables(contribution);
                summary.domain = contribution.domain;
                summary.block.domain = contribution.domain;
                summary.contribution_id =
                    contributionIdentity(contribution, op) +
                    ":constitutive:" + summary.coefficient;
                addCoefficientSummaryIfAbsent(out, std::move(summary));
            }
        }
        for (const auto& usage : contribution.parameter_usages) {
            addCoefficientSummaryIfAbsent(
                out,
                makeCoefficientSummaryFromParameterUsage(usage,
                                                         contribution,
                                                         op,
                                                         parameter_registry,
                                                         state));
        }
    }

    for (const auto& formulation : formulations) {
        for (const auto& law : formulation.constitutive_laws) {
            addCoefficientSummaryIfAbsent(
                out, makeCoefficientSummaryFromConstitutiveLaw(law, op));
        }
    }

    if (out.empty() && (!contributions.empty() || !formulations.empty())) {
        addCoefficientSummaryIfAbsent(
            out,
            makeImplicitFormCoefficientSummary(contributions, formulations, op));
    }

    return out;
}

[[nodiscard]] bool invariantDomainValueViolates(
    Real value,
    const analysis::InvariantDomainDescriptor& descriptor) noexcept
{
    if (!std::isfinite(static_cast<double>(value))) {
        return true;
    }
    const Real tolerance = descriptor.bound_tolerance > Real{}
        ? descriptor.bound_tolerance
        : static_cast<Real>(0);
    if (descriptor.lower_bound.has_value() &&
        value < *descriptor.lower_bound - tolerance) {
        return true;
    }
    if (descriptor.upper_bound.has_value() &&
        value > *descriptor.upper_bound + tolerance) {
        return true;
    }
    if (descriptor.excluded_value.has_value() &&
        std::abs(static_cast<double>(value - *descriptor.excluded_value)) <=
            static_cast<double>(tolerance)) {
        return true;
    }
    return false;
}

void appendInvariantDomainSummaryIfAbsent(
    analysis::AnalysisSummarySet& summaries,
    const analysis::InvariantDomainDescriptor& descriptor,
    const dofs::FieldDofMap& field_map,
    const SystemStateView* state);

[[nodiscard]] std::uint64_t countInvariantDomainDofViolations(
    const analysis::InvariantDomainDescriptor& descriptor,
    const dofs::FieldDofMap& field_map,
    const SystemStateView* state)
{
    if (descriptor.sampled_field == INVALID_FIELD_ID || state == nullptr) {
        return 0u;
    }
    const auto field_index = static_cast<std::size_t>(descriptor.sampled_field);
    if (!field_map.isFinalized() || field_index >= field_map.numFields()) {
        return 0u;
    }

    std::unique_ptr<assembly::GlobalSystemView> vector_view;
    if (state->u_vector != nullptr) {
        auto* mutable_vector = const_cast<backends::GenericVector*>(state->u_vector);
        vector_view = mutable_vector->createAssemblyView();
    }

    const auto read_value = [&](GlobalIndex dof, Real& value) -> bool {
        if (dof >= 0 && static_cast<std::size_t>(dof) < state->u.size()) {
            value = state->u[static_cast<std::size_t>(dof)];
            return true;
        }
        if (vector_view) {
            value = vector_view->getVectorEntry(dof);
            return true;
        }
        return false;
    };

    std::uint64_t violations = 0u;
    const auto check_dof = [&](GlobalIndex dof) {
        Real value{};
        if (read_value(dof, value) &&
            invariantDomainValueViolates(value, descriptor)) {
            ++violations;
        }
    };

    if (descriptor.sampled_component >= 0) {
        try {
            const auto dofs = field_map.getComponentDofs(
                field_index,
                static_cast<LocalIndex>(descriptor.sampled_component));
            for (const auto dof : dofs) {
                check_dof(dof);
            }
        } catch (...) {
            return 0u;
        }
    } else {
        try {
            const auto [begin, end] = field_map.getFieldDofRange(field_index);
            for (GlobalIndex dof = begin; dof < end; ++dof) {
                check_dof(dof);
            }
        } catch (...) {
            return 0u;
        }
    }

    return violations;
}

[[nodiscard]] analysis::InvariantDomainSummary makeInvariantDomainSummary(
    const analysis::InvariantDomainDescriptor& descriptor,
    const dofs::FieldDofMap& field_map,
    const SystemStateView* state)
{
    analysis::InvariantDomainSummary summary;
    summary.invariant_set_id = descriptor.invariant_set_id;
    summary.variables = descriptor.variables;
    if (descriptor.lower_bound.has_value()) {
        summary.lower_bound = *descriptor.lower_bound;
        summary.lower_bound_active = true;
    }
    if (descriptor.upper_bound.has_value()) {
        summary.upper_bound = *descriptor.upper_bound;
        summary.upper_bound_active = true;
    }
    if (descriptor.excluded_value.has_value()) {
        summary.excluded_value = *descriptor.excluded_value;
        summary.excluded_value_active = true;
    }
    if (descriptor.cfl_estimate.has_value()) {
        summary.cfl_estimate = *descriptor.cfl_estimate;
        summary.cfl_estimate_present = true;
    }
    if (descriptor.accepted_cfl_bound.has_value()) {
        summary.accepted_cfl_bound = *descriptor.accepted_cfl_bound;
        summary.accepted_cfl_bound_present = true;
    }
    if (descriptor.wave_speed_bound.has_value()) {
        summary.wave_speed_bound = *descriptor.wave_speed_bound;
        summary.wave_speed_bound_present = true;
    }
    summary.time_step_scope = descriptor.time_step_scope;
    summary.mesh_size_scope = descriptor.mesh_size_scope;
    summary.limiter_evidence_present = descriptor.limiter_evidence_present;
    summary.cfl_condition_satisfied = descriptor.cfl_condition_satisfied;
    summary.ssp_time_discretization_evidence_present =
        descriptor.ssp_time_discretization_evidence_present;
    summary.source_admissibility_evidence_present =
        descriptor.source_admissibility_evidence_present;
    summary.low_order_invariant_domain_evidence_present =
        descriptor.low_order_invariant_domain_evidence_present;
    summary.convex_limiting_evidence_present =
        descriptor.convex_limiting_evidence_present;
    summary.spatial_monotonicity_evidence_present =
        descriptor.spatial_monotonicity_evidence_present;
    summary.mass_positivity_evidence_present =
        descriptor.mass_positivity_evidence_present;
    summary.post_step_violation_count =
        countInvariantDomainDofViolations(descriptor, field_map, state);
    summary.invariant_domain_theorem_id = descriptor.theorem_id;
    return summary;
}

void appendInvariantDomainSummaryIfAbsent(
    analysis::AnalysisSummarySet& summaries,
    const analysis::InvariantDomainDescriptor& descriptor,
    const dofs::FieldDofMap& field_map,
    const SystemStateView* state)
{
    auto duplicate = std::any_of(
        summaries.invariant_domains.begin(),
        summaries.invariant_domains.end(),
        [&](const analysis::InvariantDomainSummary& existing) {
            if (existing.invariant_set_id != descriptor.invariant_set_id ||
                existing.variables != descriptor.variables) {
                return false;
            }
            const bool lower_same =
                existing.lower_bound_active == descriptor.lower_bound.has_value() &&
                (!existing.lower_bound_active ||
                 existing.lower_bound == *descriptor.lower_bound);
            const bool upper_same =
                existing.upper_bound_active == descriptor.upper_bound.has_value() &&
                (!existing.upper_bound_active ||
                 existing.upper_bound == *descriptor.upper_bound);
            const bool exclusion_same =
                existing.excluded_value_active ==
                    descriptor.excluded_value.has_value() &&
                (!existing.excluded_value_active ||
                 existing.excluded_value == *descriptor.excluded_value);
            return lower_same && upper_same && exclusion_same;
        });
    if (duplicate) {
        return;
    }
    summaries.invariant_domains.push_back(
        makeInvariantDomainSummary(descriptor, field_map, state));
}

void applyRangeFactor(Real& min_value,
                      Real& max_value,
                      Real factor_min,
                      Real factor_max)
{
    if (!std::isfinite(static_cast<double>(factor_min)) ||
        !std::isfinite(static_cast<double>(factor_max))) {
        return;
    }
    if (factor_min > factor_max) {
        std::swap(factor_min, factor_max);
    }
    min_value *= factor_min;
    max_value *= factor_max;
}

void applyPowerRange(Real& min_value,
                     Real& max_value,
                     Real base_min,
                     Real base_max,
                     int power)
{
    if (power == 0 ||
        base_min <= Real{} ||
        base_max <= Real{} ||
        !std::isfinite(static_cast<double>(base_min)) ||
        !std::isfinite(static_cast<double>(base_max))) {
        return;
    }
    const Real a = std::pow(base_min, power);
    const Real b = std::pow(base_max, power);
    applyRangeFactor(min_value, max_value, std::min(a, b), std::max(a, b));
}

[[nodiscard]] std::optional<Real> realParameterValue(
    const ParameterRegistry& registry,
    const SystemStateView* state,
    std::string_view key)
{
    if (state == nullptr || key.empty()) {
        return std::nullopt;
    }
    auto getter = registry.makeRealGetter(*state);
    return getter(key);
}

[[nodiscard]] analysis::ParameterScaleSummary makeRuntimeParameterScaleSummary(
    std::string id,
    analysis::ParameterScaleRole role,
    analysis::OperatorBlockId block,
    std::vector<analysis::VariableKey> variables,
    const analysis::FormScaleUsage& usage,
    const FieldRegistry& field_registry,
    const ParameterRegistry& parameter_registry,
    const std::unordered_map<std::string, std::pair<Real, Real>>&
        coefficient_abs_ranges,
    const assembly::IMeshAccess* mesh,
    const SystemStateView* state)
{
    analysis::ParameterScaleSummary summary;
    summary.nondimensional_parameter_id = std::move(id);
    summary.role = role;
    summary.block = std::move(block);
    summary.variables = std::move(variables);
    summary.contribution_id = summary.block.contribution_id;
    summary.min_scale_value = static_cast<Real>(1);
    summary.max_scale_value = static_cast<Real>(1);
    summary.polynomial_order =
        polynomialOrderForVariables(field_registry, summary.variables);
    summary.scale_theorem_id = usage.exact_for_analysis
        ? "FormExprDAG::RuntimeScale"
        : "FormExprDAG::SampledRuntimeScale";
    summary.mesh_quality_factor = static_cast<Real>(1);
    summary.coefficient_contrast_factor = usage.coefficient_names.empty()
        ? static_cast<Real>(0)
        : static_cast<Real>(1);

    const auto h_stats = meshScaleStats(mesh);
    if (h_stats.available) {
        applyPowerRange(summary.min_scale_value,
                        summary.max_scale_value,
                        h_stats.h_min,
                        h_stats.h_max,
                        usage.h_power);
    }

    const Real dt = state != nullptr
        ? std::abs(state->effective_dt > Real{} ? state->effective_dt : state->dt)
        : Real{};
    if (dt > Real{}) {
        applyPowerRange(summary.min_scale_value,
                        summary.max_scale_value,
                        dt,
                        dt,
                        usage.dt_power);
    }

    for (const auto& name : usage.parameter_names) {
        if (const auto value = realParameterValue(parameter_registry, state, name)) {
            const Real abs_value = std::abs(*value);
            applyRangeFactor(summary.min_scale_value,
                             summary.max_scale_value,
                             abs_value,
                             abs_value);
        }
        if (const auto* spec = parameter_registry.find(name)) {
            if (spec->lower_bound.has_value() &&
                !summary.required_lower_bound_present) {
                summary.required_lower_bound = *spec->lower_bound;
                summary.required_lower_bound_present = true;
            }
            if (!spec->theorem_id.empty()) {
                summary.scale_theorem_id = spec->theorem_id;
            }
        }
    }
    for (auto slot : usage.parameter_slots) {
        if (auto key = parameter_registry.keyForSlot(slot)) {
            if (const auto value =
                    realParameterValue(parameter_registry, state, *key)) {
                const Real abs_value = std::abs(*value);
                applyRangeFactor(summary.min_scale_value,
                                 summary.max_scale_value,
                                 abs_value,
                                 abs_value);
            }
        }
    }

    Real coefficient_contrast = static_cast<Real>(1);
    for (const auto& name : usage.coefficient_names) {
        const auto it = coefficient_abs_ranges.find(name);
        if (it == coefficient_abs_ranges.end()) {
            continue;
        }
        applyRangeFactor(summary.min_scale_value,
                         summary.max_scale_value,
                         it->second.first,
                         it->second.second);
        if (it->second.first > Real{}) {
            coefficient_contrast =
                std::max(coefficient_contrast,
                         it->second.second / it->second.first);
        }
    }
    if (!usage.coefficient_names.empty()) {
        summary.coefficient_contrast_factor = coefficient_contrast;
    }

    summary.trace_inverse_metadata_present =
        role == analysis::ParameterScaleRole::WeakBoundaryPenalty &&
        usage.h_power < 0;
    if (summary.trace_inverse_metadata_present) {
        summary.trace_inverse_constant = static_cast<Real>(1);
    }
    return summary;
}

[[nodiscard]] std::optional<analysis::MeshGeometryQualitySummary>
buildMeshGeometryQualitySummary(const assembly::IMeshAccess& mesh)
{
    constexpr Real quality_threshold = static_cast<Real>(100.0);

    std::uint64_t local_valid = 0u;
    std::uint64_t local_inverted = 0u;
    std::uint64_t local_poor = 0u;
    Real local_min_jacobian = std::numeric_limits<Real>::infinity();
    Real local_max_jacobian = -std::numeric_limits<Real>::infinity();
    Real local_max_condition = Real{};
    std::vector<analysis::ElementId> local_worst;

    const auto remember_worst = [&](GlobalIndex cell_id) {
        if (local_worst.size() < analysis::kDefaultWorstSampleLimit) {
            local_worst.push_back(static_cast<analysis::ElementId>(cell_id));
        }
    };

    mesh.forEachOwnedCell([&](GlobalIndex cell_id) {
        try {
            std::vector<std::array<Real, 3>> coords;
            mesh.getCellCoordinates(cell_id, coords);
            if (coords.empty()) {
                ++local_poor;
                remember_worst(cell_id);
                return;
            }

            std::vector<math::Vector<Real, 3>> nodes;
            nodes.reserve(coords.size());
            for (const auto& coord : coords) {
                nodes.emplace_back(math::Vector<Real, 3>{
                    coord[0], coord[1], coord[2]});
            }

            const auto cell_type = mesh.getCellType(cell_id);
            const int geometry_order = mesh.getCellGeometryOrder(cell_id);

            elements::ElementRequest element_request;
            element_request.element_type = cell_type;
            element_request.basis_type = BasisType::Lagrange;
            element_request.field_type = FieldType::Scalar;
            element_request.continuity = Continuity::C0;
            element_request.order = geometry_order;
            auto element = elements::ElementFactory::create(element_request);
            if (!element) {
                ++local_poor;
                remember_worst(cell_id);
                return;
            }

            geometry::MappingRequest mapping_request;
            mapping_request.element_type = cell_type;
            mapping_request.geometry_order = geometry_order;
            mapping_request.use_affine = geometry_order <= 1;
            auto mapping = geometry::MappingFactory::create(mapping_request, nodes);
            if (!mapping) {
                ++local_poor;
                remember_worst(cell_id);
                return;
            }

            const auto quality =
                elements::ElementValidator::validate(*element, *mapping);
            Real cell_max_detJ = quality.min_detJ;
            if (auto quadrature = element->quadrature()) {
                for (const auto& point : quadrature->points()) {
                    const Real detJ = mapping->jacobian_determinant(point);
                    if (std::isfinite(static_cast<double>(detJ))) {
                        cell_max_detJ = std::max(cell_max_detJ, detJ);
                    }
                }
            }
            const bool jacobian_finite =
                std::isfinite(static_cast<double>(quality.min_detJ));
            const bool condition_finite =
                std::isfinite(static_cast<double>(quality.max_condition_number));
            if (!jacobian_finite) {
                ++local_inverted;
                remember_worst(cell_id);
                return;
            }

            ++local_valid;
            local_min_jacobian =
                std::min(local_min_jacobian, quality.min_detJ);
            local_max_jacobian =
                std::max(local_max_jacobian, cell_max_detJ);
            if (condition_finite) {
                local_max_condition =
                    std::max(local_max_condition,
                             quality.max_condition_number);
            }
            if (!quality.positive_jacobian || quality.min_detJ <= Real{}) {
                ++local_inverted;
                remember_worst(cell_id);
            } else if (condition_finite &&
                       quality.max_condition_number > quality_threshold) {
                ++local_poor;
                remember_worst(cell_id);
            }
        } catch (const std::exception&) {
            ++local_poor;
            remember_worst(cell_id);
        }
    });

    std::uint64_t global_valid = local_valid;
    std::uint64_t global_inverted = local_inverted;
    std::uint64_t global_poor = local_poor;
    Real global_min_jacobian = local_min_jacobian;
    Real global_max_jacobian = local_max_jacobian;
    Real global_max_condition = local_max_condition;

#if FE_HAS_MPI
    global_valid = mpiSumUint64(global_valid);
    global_inverted = mpiSumUint64(global_inverted);
    global_poor = mpiSumUint64(global_poor);
    global_min_jacobian =
        static_cast<Real>(mpiReduceDouble(static_cast<double>(global_min_jacobian),
                                          MPI_MIN));
    global_max_jacobian =
        static_cast<Real>(mpiReduceDouble(static_cast<double>(global_max_jacobian),
                                          MPI_MAX));
    global_max_condition =
        static_cast<Real>(mpiReduceDouble(static_cast<double>(global_max_condition),
                                          MPI_MAX));
#endif

    if (global_valid == 0u ||
        !std::isfinite(static_cast<double>(global_min_jacobian)) ||
        !std::isfinite(static_cast<double>(global_max_jacobian))) {
        return std::nullopt;
    }

    analysis::MeshGeometryQualitySummary summary;
    summary.mesh_revision =
        mesh.revisionTrackingAvailable() ? mesh.geometryRevision() : 0u;
    summary.domain = analysis::DomainKind::Cell;
    summary.min_jacobian = global_min_jacobian;
    summary.max_jacobian = global_max_jacobian;
    summary.max_aspect_ratio = global_max_condition;
    summary.aspect_ratio_warning_threshold = quality_threshold;
    summary.inverted_element_count = global_inverted;
    summary.poor_quality_element_count = global_poor;
    summary.mesh_family_scope_present = mesh.revisionTrackingAvailable();
    if (summary.mesh_family_scope_present) {
        summary.mesh_family_scope =
            "geometry_revision=" + std::to_string(summary.mesh_revision);
    }
    summary.worst_elements = std::move(local_worst);
    return summary;
}

template <typename Summary, typename Inserter>
[[nodiscard]] std::optional<Summary> firstSummaryCoveringRequest(
    const std::vector<Summary>& summaries,
    const analysis::AnalysisSummaryRequest& request,
    Inserter insert)
{
    for (const auto& summary : summaries) {
        analysis::AnalysisSummarySet one;
        insert(one, summary);
        if (analysis::analysisSummarySetCoversRequest(one, request)) {
            return summary;
        }
    }
    return std::nullopt;
}

[[nodiscard]] bool summariesHaveAssembledEvidence(
    const analysis::AnalysisSummarySet* summaries) noexcept
{
    return summaries != nullptr &&
           (!summaries->discrete_matrices.empty() ||
            !summaries->reduced_matrices.empty() ||
            !summaries->nullspace_degeneracies.empty() ||
            !summaries->inf_sup_estimates.empty() ||
            !summaries->schur_complements.empty() ||
            !summaries->coefficient_properties.empty() ||
            !summaries->parameter_scales.empty() ||
            !summaries->stabilization_adequacy.empty() ||
            !summaries->boundary_symbols.empty() ||
            !summaries->temporal_stability.empty() ||
            !summaries->dae_structure_evidence.empty() ||
            !summaries->numerical_error_budgets.empty() ||
            !summaries->local_stencils.empty() ||
            !summaries->quadrature_adequacy.empty());
}

class FESystemAnalysisAssemblyAccess final : public analysis::AssemblyAccess {
public:
    FESystemAnalysisAssemblyAccess(const analysis::AnalysisSummarySet* summaries,
                                   bool pending)
        : summaries_(summaries), pending_(pending) {}

    [[nodiscard]] bool evidencePending() const noexcept override {
        return pending_;
    }

    [[nodiscard]] std::optional<analysis::NormMetadataSummary>
    normMetadata(const analysis::AnalysisSummaryRequest& request) const override
    {
        return first(summaries_ ? summaries_->norm_metadata : empty_.norm_metadata,
                     request,
                     [](auto& set, const auto& summary) {
                         set.norm_metadata.push_back(summary);
                     });
    }

    [[nodiscard]] std::optional<analysis::DiscreteMatrixSummary>
    discreteMatrixSummary(const analysis::AnalysisSummaryRequest& request) const override
    {
        return first(summaries_ ? summaries_->discrete_matrices
                                : empty_.discrete_matrices,
                     request,
                     [](auto& set, const auto& summary) {
                         set.discrete_matrices.push_back(summary);
                     });
    }

    [[nodiscard]] std::optional<analysis::ReducedMatrixSummary>
    reducedMatrixSummary(const analysis::AnalysisSummaryRequest& request) const override
    {
        return first(summaries_ ? summaries_->reduced_matrices
                                : empty_.reduced_matrices,
                     request,
                     [](auto& set, const auto& summary) {
                         set.reduced_matrices.push_back(summary);
                     });
    }

    [[nodiscard]] std::optional<analysis::CoefficientPropertySummary>
    coefficientProperties(const analysis::AnalysisSummaryRequest& request) const override
    {
        return first(summaries_ ? summaries_->coefficient_properties
                                : empty_.coefficient_properties,
                     request,
                     [](auto& set, const auto& summary) {
                         set.coefficient_properties.push_back(summary);
                     });
    }

    [[nodiscard]] std::optional<analysis::BoundarySymbolSummary>
    boundarySymbol(const analysis::AnalysisSummaryRequest& request) const override
    {
        return first(summaries_ ? summaries_->boundary_symbols
                                : empty_.boundary_symbols,
                     request,
                     [](auto& set, const auto& summary) {
                         set.boundary_symbols.push_back(summary);
                     });
    }

    [[nodiscard]] std::optional<analysis::InfSupEstimateSummary>
    infSupEstimate(const analysis::AnalysisSummaryRequest& request) const override
    {
        return first(summaries_ ? summaries_->inf_sup_estimates
                                : empty_.inf_sup_estimates,
                     request,
                     [](auto& set, const auto& summary) {
                         set.inf_sup_estimates.push_back(summary);
                     });
    }

    [[nodiscard]] std::optional<analysis::DAEStructureEvidenceSummary>
    daeStructureEvidence(const analysis::AnalysisSummaryRequest& request) const override
    {
        return first(summaries_ ? summaries_->dae_structure_evidence
                                : empty_.dae_structure_evidence,
                     request,
                     [](auto& set, const auto& summary) {
                         set.dae_structure_evidence.push_back(summary);
                     });
    }

    [[nodiscard]] std::optional<analysis::QuadratureAdequacySummary>
    quadratureAdequacy(const analysis::AnalysisSummaryRequest& request) const override
    {
        return first(summaries_ ? summaries_->quadrature_adequacy
                                : empty_.quadrature_adequacy,
                     request,
                     [](auto& set, const auto& summary) {
                         set.quadrature_adequacy.push_back(summary);
                     });
    }

    [[nodiscard]] std::optional<analysis::NullspaceDegeneracySummary>
    nullspaceDegeneracy(const analysis::AnalysisSummaryRequest& request) const override
    {
        return first(summaries_ ? summaries_->nullspace_degeneracies
                                : empty_.nullspace_degeneracies,
                     request,
                     [](auto& set, const auto& summary) {
                         set.nullspace_degeneracies.push_back(summary);
                     });
    }

    [[nodiscard]] std::optional<analysis::ParameterScaleSummary>
    parameterScale(const analysis::AnalysisSummaryRequest& request) const override
    {
        return first(summaries_ ? summaries_->parameter_scales
                                : empty_.parameter_scales,
                     request,
                     [](auto& set, const auto& summary) {
                         set.parameter_scales.push_back(summary);
                     });
    }

    [[nodiscard]] std::optional<analysis::StabilizationAdequacySummary>
    stabilizationAdequacy(const analysis::AnalysisSummaryRequest& request) const override
    {
        return first(summaries_ ? summaries_->stabilization_adequacy
                                : empty_.stabilization_adequacy,
                     request,
                     [](auto& set, const auto& summary) {
                         set.stabilization_adequacy.push_back(summary);
                     });
    }

    [[nodiscard]] std::optional<analysis::InitialCompatibilitySummary>
    initialCompatibility(const analysis::AnalysisSummaryRequest& request) const override
    {
        return first(summaries_ ? summaries_->initial_compatibility
                                : empty_.initial_compatibility,
                     request,
                     [](auto& set, const auto& summary) {
                         set.initial_compatibility.push_back(summary);
                     });
    }

    [[nodiscard]] std::optional<analysis::LocalStencilSummary>
    localStencil(const analysis::AnalysisSummaryRequest& request) const override
    {
        return first(summaries_ ? summaries_->local_stencils
                                : empty_.local_stencils,
                     request,
                     [](auto& set, const auto& summary) {
                         set.local_stencils.push_back(summary);
                     });
    }

    [[nodiscard]] std::optional<analysis::NumericalErrorBudgetSummary>
    numericalErrorBudget(const analysis::AnalysisSummaryRequest& request) const override
    {
        return first(summaries_ ? summaries_->numerical_error_budgets
                                : empty_.numerical_error_budgets,
                     request,
                     [](auto& set, const auto& summary) {
                         set.numerical_error_budgets.push_back(summary);
                     });
    }

private:
    template <typename Summary, typename Inserter>
    [[nodiscard]] static std::optional<Summary>
    first(const std::vector<Summary>& summaries,
          const analysis::AnalysisSummaryRequest& request,
          Inserter insert)
    {
        return firstSummaryCoveringRequest(summaries, request, insert);
    }

    const analysis::AnalysisSummarySet* summaries_{nullptr};
    bool pending_{false};
    static const analysis::AnalysisSummarySet empty_;
};

const analysis::AnalysisSummarySet FESystemAnalysisAssemblyAccess::empty_{};

class FESystemAnalysisMeshAccess final : public analysis::MeshAccess {
public:
    FESystemAnalysisMeshAccess(const assembly::IMeshAccess* mesh,
                               const analysis::AnalysisSummarySet* summaries,
                               std::string revision,
                               bool pending)
        : mesh_(mesh),
          summaries_(summaries),
          revision_(std::move(revision)),
          pending_(pending)
    {}

    [[nodiscard]] bool evidencePending() const noexcept override {
        return pending_;
    }

    [[nodiscard]] std::string meshRevision() const override {
        return revision_;
    }

    [[nodiscard]] std::optional<analysis::MeshGeometryQualitySummary>
    meshGeometryQuality(const analysis::AnalysisSummaryRequest& request) const override
    {
        if (summaries_) {
            auto summary = firstSummaryCoveringRequest(
                summaries_->mesh_geometry_quality,
                request,
                [](auto& set, const auto& value) {
                    set.mesh_geometry_quality.push_back(value);
                });
            if (summary) {
                return summary;
            }
        }
        if (mesh_ == nullptr) {
            return std::nullopt;
        }
        auto summary = buildMeshGeometryQualitySummary(*mesh_);
        if (!summary) {
            return std::nullopt;
        }
        summary->domain = request.domain;
        return summary;
    }

    [[nodiscard]] std::optional<analysis::RobustnessTrendSummary>
    refinementExperiment(const analysis::AnalysisSummaryRequest& request,
                         const analysis::AssemblyAccess&) const override
    {
        if (summaries_ == nullptr) {
            return std::nullopt;
        }
        return firstSummaryCoveringRequest(
            summaries_->robustness_trends,
            request,
            [](auto& set, const auto& summary) {
                set.robustness_trends.push_back(summary);
            });
    }

private:
    const assembly::IMeshAccess* mesh_{nullptr};
    const analysis::AnalysisSummarySet* summaries_{nullptr};
    std::string revision_;
    bool pending_{false};
};

class FESystemAnalysisSolverAccess final : public analysis::SolverAccess {
public:
    FESystemAnalysisSolverAccess(const analysis::AnalysisSummarySet* summaries,
                                 bool pending)
        : summaries_(summaries), pending_(pending) {}

    [[nodiscard]] bool evidencePending() const noexcept override {
        return pending_;
    }

    [[nodiscard]] std::optional<analysis::TemporalStabilitySummary>
    temporalStability(const analysis::AnalysisSummaryRequest& request) const override
    {
        if (summaries_ == nullptr) {
            return std::nullopt;
        }
        return firstSummaryCoveringRequest(
            summaries_->temporal_stability,
            request,
            [](auto& set, const auto& summary) {
                set.temporal_stability.push_back(summary);
            });
    }

    [[nodiscard]] std::optional<analysis::SchurComplementSummary>
    schurComplement(const analysis::AnalysisSummaryRequest& request) const override
    {
        if (summaries_ == nullptr) {
            return std::nullopt;
        }
        return firstSummaryCoveringRequest(
            summaries_->schur_complements,
            request,
            [](auto& set, const auto& summary) {
                set.schur_complements.push_back(summary);
            });
    }

private:
    const analysis::AnalysisSummarySet* summaries_{nullptr};
    bool pending_{false};
};

[[nodiscard]] bool hasTemporalMetadata(
    const std::vector<analysis::ContributionDescriptor>& contributions,
    const std::vector<analysis::VariableDescriptor>& variables)
{
    for (const auto& contribution : contributions) {
        if (contribution.temporal &&
            contribution.temporal->kind != analysis::TemporalContributionKind::None) {
            return true;
        }
    }
    for (const auto& variable : variables) {
        if (variable.temporal_state_kind != analysis::TemporalStateKind::Unknown) {
            return true;
        }
        if (variable.max_time_derivative_order > 0) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] std::vector<analysis::VariableKey> variablesFromDescriptors(
    const std::vector<analysis::VariableDescriptor>& descriptors)
{
    std::vector<analysis::VariableKey> variables;
    for (const auto& descriptor : descriptors) {
        if (std::find(variables.begin(), variables.end(), descriptor.key) ==
            variables.end()) {
            variables.push_back(descriptor.key);
        }
    }
    return variables;
}

[[nodiscard]] bool contributionIsNonlocalCoupling(
    const analysis::ContributionDescriptor& contribution) noexcept
{
    if (contribution.domain == analysis::DomainKind::InterfaceFace ||
        contribution.domain == analysis::DomainKind::CoupledBoundary ||
        contribution.domain == analysis::DomainKind::Global ||
        contribution.domain == analysis::DomainKind::AuxiliaryCoupling) {
        return true;
    }
    if (contribution.role == analysis::ContributionRole::GlobalCoupling ||
        contribution.role == analysis::ContributionRole::FieldToAuxiliary ||
        contribution.role == analysis::ContributionRole::AuxiliaryToField ||
        contribution.role == analysis::ContributionRole::AuxiliaryToAuxiliary ||
        contribution.role == analysis::ContributionRole::AuxiliarySelf) {
        return true;
    }
    return contribution.balance &&
           contribution.balance->role == analysis::BalanceRole::ExchangeLike;
}

[[nodiscard]] bool hasExplicitRoleName(const backends::SolverOptions& options,
                                       backends::BlockRole role) noexcept
{
    return std::any_of(options.block_role_names.begin(),
                       options.block_role_names.end(),
                       [&](const auto& entry) {
                           return entry.first == role && !entry.second.empty();
                       });
}

[[nodiscard]] backends::BlockRole inferFieldBlockRole(
    std::string_view field_name,
    const backends::SolverOptions& options) noexcept
{
    for (const auto& [role, name] : options.block_role_names) {
        if (!name.empty() && name == field_name) {
            return role;
        }
    }

    if (!options.momentum_block_name.empty() && options.momentum_block_name == field_name) {
        return backends::BlockRole::PrimaryField;
    }
    if (!options.constraint_block_name.empty() && options.constraint_block_name == field_name) {
        return backends::BlockRole::ConstraintField;
    }

    if (options.block_layout.has_value()) {
        if (const auto* desc = options.block_layout->findBlock(field_name)) {
            return desc->role;
        }
    }

    return backends::BlockRole::Generic;
}

[[nodiscard]] std::optional<int> uniqueMixedBlockIndexForRole(
    const backends::MixedBlockLayout& layout,
    backends::BlockRole role) noexcept
{
    std::optional<int> match{};
    for (std::size_t i = 0; i < layout.blocks.size(); ++i) {
        if (layout.blocks[i].role != role) {
            continue;
        }
        if (match.has_value()) {
            return std::nullopt;
        }
        match = static_cast<int>(i);
    }
    return match;
}

void addUnambiguousRoleMappings(backends::SolverOptions& options,
                                const backends::MixedBlockLayout& layout)
{
    for (const auto role : {backends::BlockRole::PrimaryField,
                            backends::BlockRole::ConstraintField,
                            backends::BlockRole::AuxiliaryField}) {
        if (hasExplicitRoleName(options, role)) {
            continue;
        }
        const auto block_index = uniqueMixedBlockIndexForRole(layout, role);
        if (!block_index.has_value()) {
            continue;
        }
        const auto idx = static_cast<std::size_t>(*block_index);
        if (!layout.blocks[idx].name.empty()) {
            options.block_role_names.emplace_back(role, layout.blocks[idx].name);
        }
    }
}

[[nodiscard]] const char* fieldScopeName(FieldScope scope) noexcept
{
    switch (scope) {
        case FieldScope::VolumeCell:
            return "volume";
        case FieldScope::InterfaceFace:
            return "interface";
    }
    return "unknown";
}

void validateKernelFieldScopes(const FieldRegistry& registry,
                               FieldId test_field,
                               FieldId trial_field,
                               analysis::DomainKind domain,
                               std::string_view api_name)
{
    const auto& test_rec = registry.get(test_field);
    const auto& trial_rec = registry.get(trial_field);

    const auto is_interface = [](const FieldRecord& rec) noexcept {
        return rec.scope == FieldScope::InterfaceFace;
    };

    if (domain == analysis::DomainKind::InterfaceFace) {
        FE_THROW_IF(is_interface(test_rec) && test_rec.interface_marker < 0,
                    InvalidArgumentException,
                    std::string(api_name) + ": interface-scoped test field '" +
                        test_rec.name + "' is missing an interface marker");
        FE_THROW_IF(is_interface(trial_rec) && trial_rec.interface_marker < 0,
                    InvalidArgumentException,
                    std::string(api_name) + ": interface-scoped trial field '" +
                        trial_rec.name + "' is missing an interface marker");
        return;
    }

    FE_THROW_IF(is_interface(test_rec), InvalidArgumentException,
                std::string(api_name) + ": " + fieldScopeName(test_rec.scope) +
                    "-scoped test field '" + test_rec.name +
                    "' can only be used in interface-face kernels");
    FE_THROW_IF(is_interface(trial_rec), InvalidArgumentException,
                std::string(api_name) + ": " + fieldScopeName(trial_rec.scope) +
                    "-scoped trial field '" + trial_rec.name +
                    "' can only be used in interface-face kernels");
}

} // namespace

/// Walk an expression tree and collect all FieldIds referenced by
/// DiscreteField or StateField nodes.
static void gatherFieldIds(const forms::FormExprNode& node, std::vector<FieldId>& out)
{
    const auto fid = node.fieldId();
    if (fid.has_value()) {
        if (std::find(out.begin(), out.end(), *fid) == out.end()) {
            out.push_back(*fid);
        }
    }
    for (const auto& child : node.childrenShared()) {
        if (child) gatherFieldIds(*child, out);
    }
}

[[nodiscard]] const char* scopeAutoNameToken(AuxiliaryStateScope scope) noexcept
{
    switch (scope) {
        case AuxiliaryStateScope::Global:
            return "g";
        case AuxiliaryStateScope::Node:
            return "node";
        case AuxiliaryStateScope::Cell:
            return "cell";
        case AuxiliaryStateScope::QuadraturePoint:
            return "qp";
        case AuxiliaryStateScope::Region:
            return "region";
        case AuxiliaryStateScope::Boundary:
            return "b";
        case AuxiliaryStateScope::Facet:
            return "facet";
    }
    return "aux";
}

constexpr Real kDirectCouplingEntryTol = static_cast<Real>(1e-30);

[[nodiscard]] Real effectiveAuxiliaryDt(const SystemStateView& state) noexcept
{
    if (std::isfinite(state.effective_dt) && state.effective_dt > 0.0) {
        return static_cast<Real>(state.effective_dt);
    }
    return state.dt;
}

[[nodiscard]] bool monolithicAuxTraceEnabled() noexcept
{
    static const bool enabled = [] {
        const char* env = std::getenv("SVMP_MONO_AUX_TRACE");
        if (env == nullptr) {
            return false;
        }
        std::string v(env);
        std::transform(v.begin(), v.end(), v.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return !(v == "0" || v == "false" || v == "off" || v == "no");
    }();
    return enabled;
}

[[nodiscard]] bool monolithicDirectTraceEnabled() noexcept
{
    static const bool enabled = [] {
        const char* env = std::getenv("SVMP_MONO_DIRECT_TRACE");
        if (env == nullptr) {
            return false;
        }
        std::string v(env);
        std::transform(v.begin(), v.end(), v.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return !(v == "0" || v == "false" || v == "off" || v == "no");
    }();
    return enabled;
}

template <class SpanLike>
[[nodiscard]] std::string formatTraceVector(const SpanLike& values)
{
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << values[i];
    }
    oss << "]";
    return oss.str();
}

struct AuxiliaryTemporalEvaluation {
    std::vector<Real> xdot{};
    std::vector<std::vector<Real>> history_storage{};
    std::vector<std::span<const Real>> history_spans{};
    Real dxdot_dx_coeff{0.0};
};

[[nodiscard]] std::vector<Real> gatherAuxiliaryFlatEntity(
    const AuxiliaryBlockStorage& blk,
    std::span<const Real> flat,
    std::size_t entity_idx)
{
    std::vector<Real> out;
    if (flat.empty()) {
        return out;
    }

    const auto stride = static_cast<std::size_t>(blk.componentStride());
    if (blk.layoutMode() == AuxiliaryLayoutMode::Ragged) {
        const auto offsets = blk.entityOffsets();
        FE_THROW_IF(entity_idx + 1 >= offsets.size(), InvalidArgumentException,
                    "gatherAuxiliaryFlatEntity: ragged entity index out of range");
        const auto off = offsets[entity_idx];
        const auto len = offsets[entity_idx + 1] - off;
        FE_THROW_IF(off + len > flat.size(), InvalidArgumentException,
                    "gatherAuxiliaryFlatEntity: ragged buffer out of range");
        out.assign(flat.begin() + static_cast<std::ptrdiff_t>(off),
                   flat.begin() + static_cast<std::ptrdiff_t>(off + len));
        return out;
    }

    out.assign(stride, Real(0.0));
    if (blk.ordering() == AuxiliaryEntityOrdering::ByEntityThenComponent) {
        const auto off = entity_idx * stride;
        FE_THROW_IF(off + stride > flat.size(), InvalidArgumentException,
                    "gatherAuxiliaryFlatEntity: entity-major buffer out of range");
        std::copy(flat.begin() + static_cast<std::ptrdiff_t>(off),
                  flat.begin() + static_cast<std::ptrdiff_t>(off + stride),
                  out.begin());
        return out;
    }

    const auto entity_count = blk.entityCount();
    FE_THROW_IF(stride * entity_count > flat.size(), InvalidArgumentException,
                "gatherAuxiliaryFlatEntity: component-major buffer out of range");
    for (std::size_t c = 0; c < stride; ++c) {
        out[c] = flat[c * entity_count + entity_idx];
    }
    return out;
}

void scatterAuxiliaryFlatEntity(const AuxiliaryBlockStorage& blk,
                                std::span<Real> flat,
                                std::size_t entity_idx,
                                std::span<const Real> values)
{
    if (flat.empty() || values.empty()) {
        return;
    }

    const auto stride = static_cast<std::size_t>(blk.componentStride());
    if (blk.layoutMode() == AuxiliaryLayoutMode::Ragged) {
        const auto offsets = blk.entityOffsets();
        FE_THROW_IF(entity_idx + 1 >= offsets.size(), InvalidArgumentException,
                    "scatterAuxiliaryFlatEntity: ragged entity index out of range");
        const auto off = offsets[entity_idx];
        const auto len = offsets[entity_idx + 1] - off;
        FE_THROW_IF(values.size() != len, InvalidArgumentException,
                    "scatterAuxiliaryFlatEntity: ragged value size mismatch");
        FE_THROW_IF(off + len > flat.size(), InvalidArgumentException,
                    "scatterAuxiliaryFlatEntity: ragged buffer out of range");
        std::copy(values.begin(), values.end(), flat.begin() + static_cast<std::ptrdiff_t>(off));
        return;
    }

    FE_THROW_IF(values.size() != stride, InvalidArgumentException,
                "scatterAuxiliaryFlatEntity: fixed-stride value size mismatch");
    if (blk.ordering() == AuxiliaryEntityOrdering::ByEntityThenComponent) {
        const auto off = entity_idx * stride;
        FE_THROW_IF(off + stride > flat.size(), InvalidArgumentException,
                    "scatterAuxiliaryFlatEntity: entity-major buffer out of range");
        std::copy(values.begin(), values.end(), flat.begin() + static_cast<std::ptrdiff_t>(off));
        return;
    }

    const auto entity_count = blk.entityCount();
    FE_THROW_IF(stride * entity_count > flat.size(), InvalidArgumentException,
                "scatterAuxiliaryFlatEntity: component-major buffer out of range");
    for (std::size_t c = 0; c < stride; ++c) {
        flat[c * entity_count + entity_idx] = values[c];
    }
}

[[nodiscard]] bool solveDenseSystemInPlace(std::vector<Real>& A,
                                           std::vector<Real>& b,
                                           Real pivot_tol = static_cast<Real>(1e-30))
{
    const auto n = b.size();
    if (A.size() != n * n) {
        return false;
    }
    if (n == 0) {
        return true;
    }

    std::vector<std::size_t> piv(n);
    std::iota(piv.begin(), piv.end(), std::size_t{0});

    for (std::size_t col = 0; col < n; ++col) {
        std::size_t max_row = col;
        Real max_val = std::abs(A[piv[col] * n + col]);
        for (std::size_t row = col + 1; row < n; ++row) {
            const Real v = std::abs(A[piv[row] * n + col]);
            if (v > max_val) {
                max_val = v;
                max_row = row;
            }
        }
        if (!(max_val > pivot_tol)) {
            return false;
        }
        std::swap(piv[col], piv[max_row]);

        const Real pivot = A[piv[col] * n + col];
        for (std::size_t row = col + 1; row < n; ++row) {
            const Real factor = A[piv[row] * n + col] / pivot;
            A[piv[row] * n + col] = Real(0.0);
            for (std::size_t k = col + 1; k < n; ++k) {
                A[piv[row] * n + k] -= factor * A[piv[col] * n + k];
            }
            b[piv[row]] -= factor * b[piv[col]];
        }
    }

    for (int row = static_cast<int>(n) - 1; row >= 0; --row) {
        const auto r = static_cast<std::size_t>(row);
        Real sum = b[piv[r]];
        for (std::size_t k = r + 1; k < n; ++k) {
            sum -= A[piv[r] * n + k] * b[k];
        }
        const Real diag = A[piv[r] * n + r];
        if (!(std::abs(diag) > pivot_tol)) {
            return false;
        }
        b[r] = sum / diag;
    }
    return true;
}

[[nodiscard]] bool invertDenseMatrix(std::vector<Real> A,
                                     std::size_t n,
                                     std::vector<Real>& A_inv)
{
    if (A.size() != n * n) {
        return false;
    }

    A_inv.assign(n * n, Real(0.0));
    for (std::size_t col = 0; col < n; ++col) {
        std::vector<Real> rhs(n, Real(0.0));
        rhs[col] = Real(1.0);
        auto A_work = A;
        if (!solveDenseSystemInPlace(A_work, rhs)) {
            return false;
        }
        for (std::size_t row = 0; row < n; ++row) {
            A_inv[row * n + col] = rhs[row];
        }
    }
    return true;
}

[[nodiscard]] bool tryPromoteDirectReducedToNativeRankOne(
    std::span<const std::pair<GlobalIndex, Real>> output_gradient,
    std::span<const std::pair<GlobalIndex, Real>> input_gradient,
    Real doutput_dinput,
    const dofs::IndexSet& owned_dofs,
    backends::RankOneUpdate& promoted)
{
    auto log_failure = [&](const std::string& reason) {
        if (!monolithicDirectTraceEnabled()) {
            return;
        }
        std::ostringstream oss;
        oss << "FESystem: native rank-one promotion rejected reason='" << reason << "'";
        FE_LOG_INFO(oss.str());
    };

    if (!nativeFaceRankOnePromotionEnabled()) {
        log_failure("disabled");
        return false;
    }

    constexpr Real kTol = static_cast<Real>(1e-14);
    if (!(std::abs(doutput_dinput) > kTol) || output_gradient.empty() || input_gradient.empty()) {
        log_failure("missing_data");
        return false;
    }

    std::unordered_map<GlobalIndex, Real> q_map;
    q_map.reserve(input_gradient.size());
    Real q_norm_sq = Real(0.0);
    for (const auto& [dof, value] : input_gradient) {
        q_map[dof] += value;
        q_norm_sq += value * value;
    }

    Real cross = Real(0.0);
    Real dRdQ_norm_sq = Real(0.0);
    Real local_residual_sq = Real(0.0);
    std::unordered_map<GlobalIndex, Real> dR_map;
    dR_map.reserve(output_gradient.size());
    for (const auto& [dof, dRi_dOk] : output_gradient) {
        const Real dRdQ = dRi_dOk * doutput_dinput;
        dR_map[dof] = dRdQ;
        dRdQ_norm_sq += dRdQ * dRdQ;
        const auto it = q_map.find(dof);
        if (it != q_map.end()) {
            cross += dRdQ * it->second;
        }
    }
    const int global_q_has = mpiAllreduceSumIfActive(q_map.empty() ? 0 : 1);
    const int global_dR_has = mpiAllreduceSumIfActive(dR_map.empty() ? 0 : 1);
    const Real global_q_norm_sq = mpiAllreduceSumIfActive(q_norm_sq);
    const Real global_dRdQ_norm_sq = mpiAllreduceSumIfActive(dRdQ_norm_sq);
    const Real global_cross = mpiAllreduceSumIfActive(cross);
    if (global_q_has == 0 || global_dR_has == 0 ||
        !(global_q_norm_sq > kTol * kTol) ||
        !(global_dRdQ_norm_sq > kTol * kTol)) {
        log_failure("degenerate_norm");
        return false;
    }

    const Real sigma = global_cross / global_q_norm_sq;
    if (!(std::abs(sigma) > kTol)) {
        log_failure("zero_sigma");
        return false;
    }

    for (const auto& [dof, q_val] : q_map) {
        const auto it = dR_map.find(dof);
        const Real dRdQ = (it != dR_map.end()) ? it->second : Real(0.0);
        const Real diff = dRdQ - sigma * q_val;
        local_residual_sq += diff * diff;
    }
    for (const auto& [dof, dRdQ] : dR_map) {
        if (q_map.find(dof) == q_map.end()) {
            local_residual_sq += dRdQ * dRdQ;
        }
    }

    constexpr Real kRelTolSq = static_cast<Real>(1e-4);
    const Real residual_sq = mpiAllreduceSumIfActive(local_residual_sq);
    if (!(residual_sq / std::max(global_dRdQ_norm_sq, Real(1e-30)) <= kRelTolSq)) {
        if (monolithicDirectTraceEnabled()) {
            std::ostringstream oss;
            oss << "FESystem: native rank-one promotion rejected"
                << " reason='non_rank_one'"
                << " doutput_dinput=" << doutput_dinput
                << " sigma=" << sigma
                << " q_norm=" << std::sqrt(global_q_norm_sq)
                << " dRdQ_norm=" << std::sqrt(global_dRdQ_norm_sq)
                << " rel_residual_sq="
                << residual_sq / std::max(global_dRdQ_norm_sq, Real(1e-30));
            FE_LOG_INFO(oss.str());
        }
        return false;
    }

    promoted = {};
    promoted.sigma = sigma;
    promoted.prefer_native_face = true;
    promoted.v.reserve(input_gradient.size());
    for (const auto& [dof, value] : input_gradient) {
        if (owned_dofs.contains(dof)) {
            promoted.v.emplace_back(dof, value);
        }
    }
    if (monolithicDirectTraceEnabled()) {
        std::ostringstream oss;
        oss << "FESystem: native rank-one promotion accepted"
            << " sigma=" << sigma
            << " q_norm=" << std::sqrt(global_q_norm_sq)
            << " dRdQ_norm=" << std::sqrt(global_dRdQ_norm_sq)
            << " owned_nnz=" << promoted.v.size();
        FE_LOG_INFO(oss.str());
    }
    return true;
}

[[nodiscard]] bool isPureAlgebraicAuxiliary(
    const AuxiliaryStateModel& model,
    std::size_t dim) noexcept
{
    if (dim == 0) {
        return false;
    }

    const auto meta = model.structuralMetadata();
    if (meta.variable_kinds.size() < dim) {
        return false;
    }

    return std::all_of(
        meta.variable_kinds.begin(),
        meta.variable_kinds.begin() + static_cast<std::ptrdiff_t>(dim),
        [](AuxiliaryVariableKind kind) {
            return kind == AuxiliaryVariableKind::Algebraic;
        });
}

[[nodiscard]] bool needsConsistentAuxiliaryInitialization(
    const AuxiliaryStateModel& model,
    std::size_t dim,
    AuxiliarySolveMode solve_mode)
{
    if (model.hasConsistentInitialization()) {
        return true;
    }
    if (solve_mode != AuxiliarySolveMode::Partitioned) {
        return false;
    }
    if (dim == 0) {
        return false;
    }

    const auto meta = model.structuralMetadata();
    if (meta.variable_kinds.size() < dim) {
        return false;
    }
    return std::any_of(
        meta.variable_kinds.begin(),
        meta.variable_kinds.begin() + static_cast<std::ptrdiff_t>(dim),
        [](AuxiliaryVariableKind kind) {
            return kind == AuxiliaryVariableKind::Algebraic;
        });
}

[[nodiscard]] forms::FormExpr exprFromNodeShared(
    const std::shared_ptr<const forms::FormExprNode>& node)
{
    return forms::FormExpr(std::const_pointer_cast<forms::FormExprNode>(node));
}

[[nodiscard]] bool nodeIsAuxiliaryStateRefSlot(
    const forms::FormExprNode& node,
    const std::uint32_t slot) noexcept
{
    if (node.type() != forms::FormExprType::AuxiliaryStateRef) {
        return false;
    }
    const auto s = node.slotIndex();
    return s.has_value() && *s == slot;
}

[[nodiscard]] bool exprContainsType(
    const forms::FormExprNode& node,
    const forms::FormExprType target) noexcept
{
    if (node.type() == target) {
        return true;
    }
    for (const auto* child : node.children()) {
        if (child && exprContainsType(*child, target)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool exprContainsAuxiliaryStateRefSlot(
    const forms::FormExprNode& node,
    const std::uint32_t slot) noexcept
{
    if (nodeIsAuxiliaryStateRefSlot(node, slot)) {
        return true;
    }
    for (const auto* child : node.children()) {
        if (child && exprContainsAuxiliaryStateRefSlot(*child, slot)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] std::optional<forms::FormExpr> negatedChildExpr(
    const std::shared_ptr<const forms::FormExprNode>& node)
{
    if (!node || node->type() != forms::FormExprType::Negate) {
        return std::nullopt;
    }
    const auto kids = node->childrenShared();
    if (kids.size() != 1 || !kids[0]) {
        return std::nullopt;
    }
    return exprFromNodeShared(kids[0]);
}

[[nodiscard]] std::optional<forms::FormExpr> tryExtractExplicitStateAssignment(
    const forms::FormExpr& residual,
    const std::uint32_t state_slot)
{
    if (!residual.isValid() || !residual.node()) {
        return std::nullopt;
    }

    const auto& node = *residual.node();
    const auto kids = node.childrenShared();

    switch (node.type()) {
        case forms::FormExprType::Subtract: {
            if (kids.size() != 2 || !kids[0] || !kids[1]) {
                return std::nullopt;
            }
            if (nodeIsAuxiliaryStateRefSlot(*kids[0], state_slot) &&
                !exprContainsAuxiliaryStateRefSlot(*kids[1], state_slot)) {
                return exprFromNodeShared(kids[1]);
            }
            if (nodeIsAuxiliaryStateRefSlot(*kids[1], state_slot) &&
                !exprContainsAuxiliaryStateRefSlot(*kids[0], state_slot)) {
                return exprFromNodeShared(kids[0]);
            }
            break;
        }
        case forms::FormExprType::Add: {
            if (kids.size() != 2 || !kids[0] || !kids[1]) {
                return std::nullopt;
            }
            if (nodeIsAuxiliaryStateRefSlot(*kids[0], state_slot)) {
                auto neg_rhs = negatedChildExpr(kids[1]);
                if (neg_rhs &&
                    (!neg_rhs->node() || !exprContainsAuxiliaryStateRefSlot(*neg_rhs->node(), state_slot))) {
                    return neg_rhs;
                }
            }
            if (nodeIsAuxiliaryStateRefSlot(*kids[1], state_slot)) {
                auto neg_lhs = negatedChildExpr(kids[0]);
                if (neg_lhs &&
                    (!neg_lhs->node() || !exprContainsAuxiliaryStateRefSlot(*neg_lhs->node(), state_slot))) {
                    return neg_lhs;
                }
            }
            break;
        }
        default:
            break;
    }

    return std::nullopt;
}

[[nodiscard]] bool solvePureAlgebraicAuxiliaryState(
    const AuxiliaryStateModel& model,
    const AuxiliaryDerivativeProvider& deriv,
    std::span<Real> x,
    const AuxiliaryLocalContext& base_ctx,
    int max_iterations = 25,
    Real tol_abs = static_cast<Real>(1e-12),
    Real tol_rel = static_cast<Real>(1e-10))
{
    const auto n = static_cast<std::size_t>(model.dimension());
    if (x.size() != n || n == 0) {
        return x.size() == n;
    }

    std::vector<Real> xdot(n, Real(0.0));
    std::vector<Real> residual(n, Real(0.0));
    std::vector<Real> dFdx(n * n, Real(0.0));

    auto residual_norm = [&](std::span<const Real> r) {
        Real norm_sq = Real(0.0);
        for (const Real v : r) {
            norm_sq += v * v;
        }
        return std::sqrt(norm_sq);
    };

    Real initial_norm = Real(-1.0);

    for (int it = 0; it < max_iterations; ++it) {
        AuxiliaryLocalContext ctx = base_ctx;
        ctx.x = x;
        ctx.xdot = xdot;

        AuxiliaryResidualRequest res_req;
        res_req.residual = residual;
        model.evaluateResidual(ctx, res_req);

        const Real norm = residual_norm(residual);
        if (initial_norm < Real(0.0)) {
            initial_norm = norm;
        }
        const Real scale = tol_abs + tol_rel * (Real(1.0) + initial_norm);
        if (norm <= scale) {
            return true;
        }

        AuxiliaryJacobianRequest jac_req;
        jac_req.dF_dx = dFdx;
        jac_req.n = static_cast<int>(n);
        deriv.evaluateJacobian(model, ctx, jac_req);

        std::vector<Real> delta = residual;
        for (Real& v : delta) {
            v = -v;
        }
        auto A = dFdx;
        if (!solveDenseSystemInPlace(A, delta)) {
            return false;
        }

        for (std::size_t i = 0; i < n; ++i) {
            x[i] += delta[i];
        }
    }

    return false;
}

#if FE_HAS_MPI
MPI_Datatype mpiRealType()
{
    if (sizeof(Real) == sizeof(double)) {
        return MPI_DOUBLE;
    }
    if (sizeof(Real) == sizeof(float)) {
        return MPI_FLOAT;
    }
    return MPI_LONG_DOUBLE;
}

MPI_Datatype mpiGlobalIndexType()
{
    if (sizeof(GlobalIndex) == sizeof(std::int64_t)) {
        return MPI_INT64_T;
    }
    if (sizeof(GlobalIndex) == sizeof(long long)) {
        return MPI_LONG_LONG;
    }
    if (sizeof(GlobalIndex) == sizeof(long)) {
        return MPI_LONG;
    }
    return MPI_LONG_LONG;
}

[[nodiscard]] std::vector<std::pair<GlobalIndex, Real>> allreduceSumSparsePairs(
    std::vector<std::pair<GlobalIndex, Real>> local,
    MPI_Comm comm)
{
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        return local;
    }

    int comm_size = 1;
    MPI_Comm_size(comm, &comm_size);
    if (comm_size <= 1) {
        return local;
    }

    const int local_n = static_cast<int>(local.size());
    std::vector<int> counts(static_cast<std::size_t>(comm_size), 0);
    MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);

    std::vector<int> displs(static_cast<std::size_t>(comm_size), 0);
    int total_n = 0;
    for (int r = 0; r < comm_size; ++r) {
        displs[static_cast<std::size_t>(r)] = total_n;
        total_n += counts[static_cast<std::size_t>(r)];
    }

    std::vector<GlobalIndex> idx_local(static_cast<std::size_t>(local_n), GlobalIndex(0));
    std::vector<Real> val_local(static_cast<std::size_t>(local_n), Real(0.0));
    for (int i = 0; i < local_n; ++i) {
        idx_local[static_cast<std::size_t>(i)] = local[static_cast<std::size_t>(i)].first;
        val_local[static_cast<std::size_t>(i)] = local[static_cast<std::size_t>(i)].second;
    }

    std::vector<GlobalIndex> idx_all(static_cast<std::size_t>(total_n), GlobalIndex(0));
    std::vector<Real> val_all(static_cast<std::size_t>(total_n), Real(0.0));
    MPI_Allgatherv(idx_local.data(), local_n, mpiGlobalIndexType(),
                   idx_all.data(), counts.data(), displs.data(), mpiGlobalIndexType(), comm);
    MPI_Allgatherv(val_local.data(), local_n, mpiRealType(),
                   val_all.data(), counts.data(), displs.data(), mpiRealType(), comm);

    std::vector<std::pair<GlobalIndex, Real>> merged;
    merged.reserve(static_cast<std::size_t>(total_n));
    for (int i = 0; i < total_n; ++i) {
        merged.emplace_back(idx_all[static_cast<std::size_t>(i)],
                            val_all[static_cast<std::size_t>(i)]);
    }

    std::sort(merged.begin(), merged.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<std::pair<GlobalIndex, Real>> out;
    out.reserve(merged.size());
    for (const auto& kv : merged) {
        if (out.empty() || kv.first != out.back().first) {
            out.push_back(kv);
        } else {
            out.back().second += kv.second;
        }
    }
    return out;
}

struct LocalCondensedSlotKey {
    unsigned long long block_ordinal{0};
    unsigned long long entity_key{0};
    unsigned long long component{0};
};

[[nodiscard]] std::vector<LocalCondensedSlotKey> allgatherLocalCondensedSlotKeys(
    std::vector<LocalCondensedSlotKey> local,
    MPI_Comm comm)
{
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        return local;
    }

    auto sort_unique = [](std::vector<LocalCondensedSlotKey>& keys) {
        std::sort(keys.begin(), keys.end(), [](const auto& a, const auto& b) {
            return std::tie(a.block_ordinal, a.entity_key, a.component) <
                   std::tie(b.block_ordinal, b.entity_key, b.component);
        });
        keys.erase(std::unique(keys.begin(), keys.end(), [](const auto& a, const auto& b) {
                       return a.block_ordinal == b.block_ordinal &&
                              a.entity_key == b.entity_key &&
                              a.component == b.component;
                   }),
                   keys.end());
    };

    int comm_size = 1;
    MPI_Comm_size(comm, &comm_size);
    if (comm_size <= 1) {
        sort_unique(local);
        return local;
    }

    const int local_n = static_cast<int>(local.size());
    std::vector<int> counts(static_cast<std::size_t>(comm_size), 0);
    MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);

    std::vector<int> displs(static_cast<std::size_t>(comm_size), 0);
    int total_n = 0;
    for (int r = 0; r < comm_size; ++r) {
        displs[static_cast<std::size_t>(r)] = total_n;
        total_n += counts[static_cast<std::size_t>(r)];
    }

    std::vector<unsigned long long> send(
        static_cast<std::size_t>(local_n) * 3u, 0ull);
    for (int i = 0; i < local_n; ++i) {
        const auto base = static_cast<std::size_t>(i) * 3u;
        send[base + 0u] = local[static_cast<std::size_t>(i)].block_ordinal;
        send[base + 1u] = local[static_cast<std::size_t>(i)].entity_key;
        send[base + 2u] = local[static_cast<std::size_t>(i)].component;
    }

    std::vector<int> counts3(counts.size(), 0);
    std::vector<int> displs3(displs.size(), 0);
    for (std::size_t i = 0; i < counts.size(); ++i) {
        counts3[i] = counts[i] * 3;
        displs3[i] = displs[i] * 3;
    }

    std::vector<unsigned long long> recv(
        static_cast<std::size_t>(total_n) * 3u, 0ull);
    MPI_Allgatherv(send.data(), local_n * 3, MPI_UNSIGNED_LONG_LONG,
                   recv.data(), counts3.data(), displs3.data(),
                   MPI_UNSIGNED_LONG_LONG, comm);

    std::vector<LocalCondensedSlotKey> gathered;
    gathered.reserve(static_cast<std::size_t>(total_n));
    for (int i = 0; i < total_n; ++i) {
        const auto base = static_cast<std::size_t>(i) * 3u;
        gathered.push_back(LocalCondensedSlotKey{
            recv[base + 0u], recv[base + 1u], recv[base + 2u]});
    }
    sort_unique(gathered);
    return gathered;
}
#endif

class ScopedAssemblerOptions final {
public:
    ScopedAssemblerOptions(assembly::Assembler* assembler,
                           assembly::AssemblyOptions options)
        : assembler_(assembler)
        , saved_options_(assembler != nullptr ? assembler->getOptions()
                                              : assembly::AssemblyOptions{})
        , active_(assembler != nullptr)
    {
        if (active_) {
            assembler_->setOptions(options);
        }
    }

    ScopedAssemblerOptions(const ScopedAssemblerOptions&) = delete;
    ScopedAssemblerOptions& operator=(const ScopedAssemblerOptions&) = delete;

    ~ScopedAssemblerOptions()
    {
        if (active_) {
            assembler_->setOptions(saved_options_);
        }
    }

private:
    assembly::Assembler* assembler_{nullptr};
    assembly::AssemblyOptions saved_options_{};
    bool active_{false};
};

[[nodiscard]] assembly::AssemblyOptions localSensitivityAssemblyOptions(
    const assembly::Assembler& assembler)
{
    auto options = assembler.getOptions();
    options.allow_unowned_row_accumulation = true;
    options.row_owner_rank = {};
    return options;
}

[[nodiscard]] std::vector<Real> reconstructRateFromHistory(
    std::span<const Real> committed,
    std::span<const std::span<const Real>> history,
    double dt_prev,
    double dt_current,
    std::span<const double> dt_history)
{
    std::vector<Real> rate(committed.size(), Real(0.0));
    if (committed.empty() || history.empty()) {
        return rate;
    }

    const double fallback_dt =
        (dt_prev > 0.0 && std::isfinite(dt_prev))
            ? dt_prev
            : ((dt_current > 0.0 && std::isfinite(dt_current)) ? dt_current : 1.0);

    auto historyDt = [&](int idx) -> double {
        if (idx >= 0 && idx < static_cast<int>(dt_history.size())) {
            const double v = dt_history[static_cast<std::size_t>(idx)];
            if (v > 0.0 && std::isfinite(v)) {
                return v;
            }
        }
        return fallback_dt;
    };

    std::vector<double> nodes;
    nodes.reserve(history.size() + 1);
    nodes.push_back(0.0);
    double accum = 0.0;
    for (std::size_t j = 0; j < history.size(); ++j) {
        accum += historyDt(static_cast<int>(j));
        nodes.push_back(-accum);
    }

    const auto w = math::finiteDifferenceWeights(/*derivative_order=*/1, /*x0=*/0.0, nodes);
    if (w.size() != nodes.size()) {
        return rate;
    }

    for (std::size_t i = 0; i < committed.size(); ++i) {
        Real val = static_cast<Real>(w[0]) * committed[i];
        for (std::size_t j = 0; j < history.size(); ++j) {
            if (i < history[j].size()) {
                val += static_cast<Real>(w[j + 1]) * history[j][i];
            }
        }
        rate[i] = val;
    }
    return rate;
}

[[nodiscard]] AuxiliaryTemporalEvaluation buildMonolithicAuxiliaryTemporalEvaluation(
    const AuxiliaryStepperSpec& stepper_spec,
    const AuxiliaryBlockStorage& blk,
    std::size_t entity_idx,
    std::span<const Real> entity_x,
    std::span<const Real> entity_committed,
    std::span<const Real> entity_committed_rate,
    const SystemStateView& state)
{
    AuxiliaryTemporalEvaluation out;
    out.xdot.assign(entity_x.size(), Real(0.0));

    const auto history_depth = blk.history().depth();
    out.history_storage.reserve(history_depth);
    for (std::size_t k = 0; k < history_depth; ++k) {
        out.history_storage.push_back(blk.gatherEntityHistory(k, entity_idx));
    }
    out.history_spans.reserve(out.history_storage.size());
    for (const auto& hist : out.history_storage) {
        out.history_spans.emplace_back(hist.data(), hist.size());
    }

    const auto* ti = state.time_integration ? state.time_integration->stencil(1) : nullptr;
    if (ti && !ti->a.empty()) {
        if (state.time_integration != nullptr &&
            state.time_integration->integrator_name == "GeneralizedAlpha(1stOrder)" &&
            ti->a.size() == 3u) {
            if (entity_committed_rate.size() == entity_x.size()) {
                out.dxdot_dx_coeff = ti->coeff(0);
                for (std::size_t i = 0; i < entity_x.size(); ++i) {
                    out.xdot[i] =
                        ti->coeff(0) * entity_x[i] +
                        ti->coeff(1) * entity_committed[i] +
                        ti->coeff(2) * entity_committed_rate[i];
                }
                return out;
            }
            if (!out.history_spans.empty()) {
                out.dxdot_dx_coeff = ti->coeff(0);
                const auto xdot_n = reconstructRateFromHistory(
                    entity_committed,
                    out.history_spans,
                    state.dt_prev,
                    state.dt,
                    state.dt_history);
                for (std::size_t i = 0; i < entity_x.size(); ++i) {
                    out.xdot[i] =
                        ti->coeff(0) * entity_x[i] +
                        ti->coeff(1) * entity_committed[i] +
                        ti->coeff(2) * xdot_n[i];
                }
                return out;
            }
        } else {
            out.dxdot_dx_coeff = ti->coeff(0);

            for (std::size_t i = 0; i < entity_x.size(); ++i) {
                Real val = ti->coeff(0) * entity_x[i];
                if (ti->a.size() > 1u) {
                    val += ti->coeff(1) * entity_committed[i];
                }
                for (std::size_t j = 2; j < ti->a.size(); ++j) {
                    const auto hist_idx = j - 2;
                    FE_THROW_IF(hist_idx >= out.history_spans.size() &&
                                    std::abs(ti->coeff(static_cast<int>(j))) > Real(1e-30),
                                InvalidStateException,
                                "FESystem: insufficient auxiliary history for time stencil of block entity");
                    if (hist_idx < out.history_spans.size() && i < out.history_spans[hist_idx].size()) {
                        val += ti->coeff(static_cast<int>(j)) * out.history_spans[hist_idx][i];
                    }
                }
                out.xdot[i] = val;
            }
            return out;
        }
    }

    const std::string_view monolithic_method = stepper_spec.method_name;
    if (monolithic_method == "BackwardEuler") {
        const Real h = state.dt;
        if (h > Real(0.0)) {
            out.dxdot_dx_coeff = Real(1.0) / h;
            for (std::size_t i = 0; i < entity_x.size(); ++i) {
                out.xdot[i] = (entity_x[i] - entity_committed[i]) / h;
            }
        }
        return out;
    }

    const Real aux_dt = effectiveAuxiliaryDt(state);
    if (aux_dt > 0.0) {
        out.dxdot_dx_coeff = Real(1.0) / aux_dt;
        for (std::size_t i = 0; i < entity_x.size(); ++i) {
            out.xdot[i] = (entity_x[i] - entity_committed[i]) / aux_dt;
        }
    }
    return out;
}

[[nodiscard]] std::vector<std::pair<GlobalIndex, Real>> reconstructInputGradientFromCt(
    const std::vector<Real>& ct,
    std::size_t n_field_dofs,
    std::size_t aux_row_offset,
    int dim,
    const std::vector<Real>& dF_dinputs,
    int n_inputs,
    int input_col)
{
    if (n_field_dofs == 0 || dim <= 0 || n_inputs <= 0 || input_col < 0 ||
        dF_dinputs.size() < static_cast<std::size_t>(dim * n_inputs)) {
        return {};
    }

    Real denom = 0.0;
    std::vector<Real> numer(n_field_dofs, 0.0);

    for (int i = 0; i < dim; ++i) {
        const Real dF_dI = dF_dinputs[static_cast<std::size_t>(i * n_inputs + input_col)];
        if (std::abs(dF_dI) <= kDirectCouplingEntryTol) {
            continue;
        }
        denom += dF_dI * dF_dI;

        const auto row = aux_row_offset + static_cast<std::size_t>(i);
        const auto row_offset = row * n_field_dofs;
        if (row_offset + n_field_dofs > ct.size()) {
            return {};
        }
        for (std::size_t k = 0; k < n_field_dofs; ++k) {
            numer[k] += dF_dI * ct[row_offset + k];
        }
    }

    if (!(denom > kDirectCouplingEntryTol * kDirectCouplingEntryTol)) {
        return {};
    }

    std::vector<std::pair<GlobalIndex, Real>> q_u;
    q_u.reserve(n_field_dofs);
    for (std::size_t k = 0; k < n_field_dofs; ++k) {
        const Real val = numer[k] / denom;
        if (std::abs(val) > kDirectCouplingEntryTol) {
            q_u.emplace_back(static_cast<GlobalIndex>(k), val);
        }
    }
    return q_u;
}

namespace {

MeshParticipantInfo completeParticipantInfo(MeshParticipantInfo info,
                                            const assembly::IMeshAccess& access)
{
    if (info.name.empty()) {
        info.name = "mesh";
    }
    if (info.num_cells == 0) {
        info.num_cells = access.numCells();
    }
    if (info.num_vertices == 0) {
        info.num_vertices = access.numVertices();
    }
    if (info.num_boundary_faces == 0) {
        info.num_boundary_faces = access.numBoundaryFaces();
    }
    if (info.num_interior_faces == 0) {
        info.num_interior_faces = access.numInteriorFaces();
    }
    return info;
}

std::vector<MeshParticipantInfo> singleParticipantInfo(
    MeshParticipantInfo info,
    const assembly::IMeshAccess& access)
{
    return {completeParticipantInfo(std::move(info), access)};
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
std::vector<MeshParticipantInfo> participantInfoFromComposite(
    const assembly::CompositeMeshAccess& access)
{
    std::vector<MeshParticipantInfo> participants;
    participants.reserve(access.numParticipants());
    for (std::size_t i = 0; i < access.numParticipants(); ++i) {
        MeshParticipantInfo info;
        info.name = access.participantName(i);
        info.domain_id = access.participantDomainId(i);

        bool saw_cell = false;
        for (GlobalIndex c = 0; c < access.numCells(); ++c) {
            const auto location = access.cellLocation(c);
            if (location.participant_index != i) {
                continue;
            }
            if (!saw_cell) {
                info.cell_offset = c;
                saw_cell = true;
            }
            ++info.num_cells;
        }

        bool saw_vertex = false;
        for (GlobalIndex v = 0; v < access.numVertices(); ++v) {
            const auto location = access.vertexLocation(v);
            if (location.participant_index != i) {
                continue;
            }
            if (!saw_vertex) {
                info.vertex_offset = v;
                saw_vertex = true;
            }
            ++info.num_vertices;
        }

        bool saw_boundary = false;
        for (GlobalIndex f = 0; f < access.numBoundaryFaces(); ++f) {
            const auto location = access.boundaryFaceLocation(f);
            if (location.participant_index != i) {
                continue;
            }
            if (!saw_boundary) {
                info.boundary_face_offset = f;
                saw_boundary = true;
            }
            ++info.num_boundary_faces;
        }

        bool saw_interior = false;
        for (GlobalIndex f = 0; f < access.numInteriorFaces(); ++f) {
            const auto location = access.interiorFaceLocation(f);
            if (location.participant_index != i) {
                continue;
            }
            if (!saw_interior) {
                info.interior_face_offset = f;
                saw_interior = true;
            }
            ++info.num_interior_faces;
        }

        participants.push_back(std::move(info));
    }
    return participants;
}
#endif

} // namespace

FESystem::FESystem(std::shared_ptr<const assembly::IMeshAccess> mesh_access)
    : FESystem(std::move(mesh_access), {})
{
}

FESystem::FESystem(std::shared_ptr<const assembly::IMeshAccess> mesh_access,
                   std::vector<MeshParticipantInfo> participants)
    : mesh_access_(std::move(mesh_access)),
      mesh_participants_(std::move(participants))
{
    // mesh_access_ may be null for auxiliary-only use (no FE field assembly).
    // Full FE operations (setup, assembly) require non-null mesh.
    operator_backends_ = std::make_unique<OperatorBackends>();
}

FESystem::~FESystem() = default;
FESystem::FESystem(FESystem&&) noexcept = default;
FESystem& FESystem::operator=(FESystem&&) noexcept = default;

std::uint64_t FESystem::systemLayoutRevision() const noexcept
{
    std::uint64_t key = 1469598103934665603ULL;
    const auto mix = [&key](std::uint64_t value) noexcept {
        key ^= value;
        key *= 1099511628211ULL;
    };
    mix(fe_layout_revisions_.space);
    mix(fe_layout_revisions_.dof_layout);
    mix(fe_layout_revisions_.constraint_layout);
    mix(fe_layout_revisions_.block_layout);
    return key;
}

constraints::ConstraintRevisionSnapshot
FESystem::captureConstraintRevisionSnapshot(
    bool include_mesh_field_values) const noexcept
{
    constraints::ConstraintRevisionSnapshot snapshot;
    snapshot.valid = true;
    snapshot.fe_space = fe_layout_revisions_.space;
    snapshot.fe_dof_layout = fe_layout_revisions_.dof_layout;
    snapshot.fe_constraint_layout = fe_layout_revisions_.constraint_layout;
    snapshot.fe_block_layout = fe_layout_revisions_.block_layout;
    snapshot.time_epoch = constraint_time_epoch_;
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    if (mesh_) {
        const auto& local_mesh = mesh_->local_mesh();
        snapshot.geometry = local_mesh.geometry_revision();
        snapshot.reference_rebase = local_mesh.reference_rebase_epoch();
        snapshot.topology = local_mesh.topology_revision();
        snapshot.ownership = local_mesh.ownership_revision();
        snapshot.numbering = local_mesh.numbering_revision();
        snapshot.mesh_field_layout = local_mesh.field_layout_revision();
        if (include_mesh_field_values) {
            snapshot.mesh_field_values = meshFieldValueFingerprint(local_mesh);
        }
        snapshot.labels = local_mesh.label_revision();
        snapshot.active_configuration = local_mesh.active_configuration_epoch();
    } else if (mesh_access_ && mesh_access_->revisionTrackingAvailable()) {
        snapshot.geometry = mesh_access_->geometryRevision();
        snapshot.reference_rebase = 0;
        snapshot.topology = mesh_access_->topologyRevision();
        snapshot.ownership = mesh_access_->ownershipRevision();
        snapshot.numbering = mesh_access_->numberingRevision();
        snapshot.mesh_field_layout = mesh_access_->fieldLayoutRevision();
        snapshot.mesh_field_values = include_mesh_field_values
                                         ? snapshot.mesh_field_layout
                                         : 0u;
        snapshot.labels = mesh_access_->labelRevision();
        snapshot.active_configuration = mesh_access_->activeConfigurationEpoch();
    }
#endif
    return snapshot;
}

constraints::ConstraintRevisionSnapshot
FESystem::constraintRevisionSnapshot() const noexcept
{
    return constraint_revision_snapshot_;
}

constraints::ConstraintDependencyDeclaration
FESystem::constraintDependencyDeclaration() const
{
    constraints::ConstraintDependencyDeclaration out;
    for (const auto& c : constraint_defs_) {
        FE_CHECK_NOT_NULL(c.get(), "FESystem::constraintDependencyDeclaration: constraint");
        constraints::merge_into(out, c->dependencyDeclaration());
    }
    for (const auto& c : system_constraint_defs_) {
        FE_CHECK_NOT_NULL(c.get(), "FESystem::constraintDependencyDeclaration: system constraint");
        constraints::merge_into(out, c->dependencyDeclaration());
    }
    return out;
}

bool FESystem::constraintStateStaleForCurrentRevisions() const
{
    constraints::ConstraintDependencyDeclaration deps;
    for (const auto& c : constraint_defs_) {
        if (c) {
            constraints::merge_into(deps, c->dependencyDeclaration());
        }
    }
    for (const auto& c : system_constraint_defs_) {
        if (c) {
            constraints::merge_into(deps, c->dependencyDeclaration());
        }
    }
    const bool include_mesh_field_values =
        deps.structural.mesh_field_values || deps.value.mesh_field_values;
    const auto current = captureConstraintRevisionSnapshot(include_mesh_field_values);
    return constraints::structural_dependency_changed(deps, constraint_revision_snapshot_, current) ||
           constraints::value_dependency_changed(deps, constraint_revision_snapshot_, current);
}

constraints::ConstraintRefreshResult
FESystem::refreshConstraintStateForCurrentRevisions(double time,
                                                    double dt,
                                                    bool allow_structural_rebuild)
{
    constraints::ConstraintRefreshResult result;
    requireSetup();
    has_last_constraint_update_time_ = true;
    last_constraint_update_time_ = time;
    last_constraint_update_dt_ = dt;

    const auto deps = constraintDependencyDeclaration();
    const bool include_mesh_field_values =
        deps.structural.mesh_field_values || deps.value.mesh_field_values;
    const auto current = captureConstraintRevisionSnapshot(include_mesh_field_values);
    const bool structural_changed =
        constraints::structural_dependency_changed(deps, constraint_revision_snapshot_, current);
    const bool value_changed =
        constraints::value_dependency_changed(deps, constraint_revision_snapshot_, current);

    if (!structural_changed && !value_changed) {
        result.reason = "constraint dependencies unchanged";
        return result;
    }

    result.dependency_changed = true;
    if (structural_changed) {
        if (!allow_structural_rebuild || !has_last_setup_) {
            result.skipped_no_cached_setup = true;
            result.reason = "constraint structural dependencies changed but cached setup is unavailable";
            return result;
        }
        result.structural_rebuild = true;
        result.reason = "constraint structural dependencies changed; rebuilt FE setup";
        setup(last_setup_options_, last_setup_inputs_);
        return result;
    }

    bool any_update = false;
    for (const auto& c : constraint_defs_) {
        FE_CHECK_NOT_NULL(c.get(), "FESystem::refreshConstraintStateForCurrentRevisions: constraint");
        const auto decl = c->dependencyDeclaration();
        if (constraints::value_dependency_changed(decl, constraint_revision_snapshot_, current)) {
            any_update = c->updateValues(affine_constraints_, time) || any_update;
        }
    }
    for (auto& c : system_constraint_defs_) {
        FE_CHECK_NOT_NULL(c.get(), "FESystem::refreshConstraintStateForCurrentRevisions: system constraint");
        const auto decl = c->dependencyDeclaration();
        if (constraints::value_dependency_changed(decl, constraint_revision_snapshot_, current)) {
            any_update = c->updateValues(*this, affine_constraints_, time, dt) || any_update;
        }
    }

    ++constraint_time_epoch_;
    constraint_revision_snapshot_ =
        captureConstraintRevisionSnapshot(include_mesh_field_values);
    buildConstraintSummary();
    invalidateAnalysisCache();
    result.value_update = any_update;
    result.reason = any_update ? "constraint value dependencies changed; updated inhomogeneities"
                               : "constraint value dependencies changed; no inhomogeneity updates were needed";
    return result;
}

OperatorRevisionSnapshot FESystem::operatorRevisionSnapshot() const noexcept
{
    OperatorRevisionSnapshot snapshot;
    snapshot.valid = true;
    snapshot.fe_layout = fe_layout_revisions_;
    snapshot.system_layout_key = systemLayoutRevision();
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    snapshot.geometry_state = geometry_transaction_state_;
    snapshot.geometry_use = geometryConfigurationUse();
    if (mesh_) {
        const auto& local_mesh = mesh_->local_mesh();
        snapshot.mesh.valid = true;
        snapshot.mesh.geometry = local_mesh.geometry_revision();
        snapshot.mesh.reference_geometry = local_mesh.reference_geometry_revision();
        snapshot.mesh.current_geometry = local_mesh.current_geometry_revision();
        snapshot.mesh.reference_rebase = local_mesh.reference_rebase_epoch();
        snapshot.mesh.topology = local_mesh.topology_revision();
        snapshot.mesh.ownership = local_mesh.ownership_revision();
        snapshot.mesh.numbering = local_mesh.numbering_revision();
        snapshot.mesh.field_layout = local_mesh.field_layout_revision();
        snapshot.mesh.labels = local_mesh.label_revision();
        snapshot.mesh.active_configuration = local_mesh.active_configuration_epoch();
    } else if (mesh_access_ && mesh_access_->revisionTrackingAvailable()) {
        snapshot.mesh.valid = true;
        snapshot.mesh.geometry = mesh_access_->geometryRevision();
        snapshot.mesh.reference_geometry = mesh_access_->geometryRevision();
        snapshot.mesh.current_geometry = mesh_access_->geometryRevision();
        snapshot.mesh.reference_rebase = 0;
        snapshot.mesh.topology = mesh_access_->topologyRevision();
        snapshot.mesh.ownership = mesh_access_->ownershipRevision();
        snapshot.mesh.numbering = mesh_access_->numberingRevision();
        snapshot.mesh.field_layout = mesh_access_->fieldLayoutRevision();
        snapshot.mesh.labels = mesh_access_->labelRevision();
        snapshot.mesh.active_configuration = mesh_access_->activeConfigurationEpoch();
    }
#else
    snapshot.geometry_state = GeometryTransactionState::Committed;
    snapshot.geometry_use = GeometryConfigurationUse::Reference;
#endif
    return snapshot;
}

OperatorInvalidationDecision FESystem::operatorInvalidationDecision(
    const OperatorRevisionSnapshot& cached,
    bool allow_lagged_jacobian_on_geometry_change) const
{
    return decideOperatorInvalidation(cached,
                                      operatorRevisionSnapshot(),
                                      allow_lagged_jacobian_on_geometry_change);
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
FESystem::FESystem(std::shared_ptr<const svmp::Mesh> mesh, svmp::Configuration coord_cfg)
    : FESystem(std::move(mesh), MeshParticipantInfo{.name = "mesh"}, coord_cfg)
{
}

FESystem::FESystem(std::shared_ptr<const svmp::Mesh> mesh,
                   MeshParticipantInfo participant,
                   svmp::Configuration coord_cfg)
    : mesh_(std::move(mesh)), coord_cfg_(coord_cfg)
{
    FE_CHECK_NOT_NULL(mesh_.get(), "FESystem::mesh");
    mesh_access_ = std::make_shared<assembly::MeshAccess>(*mesh_, coord_cfg_);
    search_access_ = std::make_shared<MeshSearchAccess>(*mesh_, coord_cfg_);
    FE_CHECK_NOT_NULL(mesh_access_.get(), "FESystem::mesh_access");
    mesh_participants_ = singleParticipantInfo(std::move(participant), *mesh_access_);
    operator_backends_ = std::make_unique<OperatorBackends>();
}

FESystem::FESystem(std::shared_ptr<assembly::CompositeMeshAccess> mesh_access)
    : FESystem(std::static_pointer_cast<const assembly::CompositeMeshAccess>(mesh_access))
{
}

FESystem::FESystem(std::shared_ptr<const assembly::CompositeMeshAccess> mesh_access)
    : FESystem(std::static_pointer_cast<const assembly::IMeshAccess>(mesh_access),
               mesh_access ? participantInfoFromComposite(*mesh_access)
                           : std::vector<MeshParticipantInfo>{})
{
    FE_CHECK_NOT_NULL(mesh_access.get(), "FESystem::CompositeMeshAccess");
}

const svmp::Mesh& FESystem::singleMesh(std::string_view api_name) const
{
    if (mesh_) {
        return *mesh_;
    }
    FE_THROW(InvalidStateException,
             std::string(api_name) +
                 ": requires a FESystem constructed from a single svmp::Mesh; "
                 "composite mesh systems expose participant metadata and meshAccess() instead");
}

void FESystem::setInterfaceMesh(InterfaceId marker, std::shared_ptr<const svmp::InterfaceMesh> mesh)
{
    invalidateSetup();
    FE_THROW_IF(marker < 0, InvalidArgumentException,
                "FESystem::setInterfaceMesh: marker must be >= 0");
    FE_CHECK_NOT_NULL(mesh.get(), "FESystem::setInterfaceMesh: mesh");
    interface_meshes_[marker] = std::move(mesh);
}

bool FESystem::hasInterfaceMesh(InterfaceId marker) const noexcept
{
    return interface_meshes_.find(marker) != interface_meshes_.end();
}

const svmp::InterfaceMesh& FESystem::interfaceMesh(InterfaceId marker) const
{
    auto it = interface_meshes_.find(marker);
    FE_THROW_IF(it == interface_meshes_.end() || !it->second, InvalidArgumentException,
                "FESystem::interfaceMesh: unknown interface marker " + std::to_string(marker));
    return *it->second;
}

void FESystem::setInterfaceMeshFromFaceSet(InterfaceId marker,
                                           const std::string& face_set_name,
                                           bool compute_orientation)
{
    const auto& mesh = singleMesh("FESystem::setInterfaceMeshFromFaceSet");
    auto iface = std::make_shared<svmp::InterfaceMesh>(
        svmp::InterfaceMesh::build_from_face_set(mesh.base(), face_set_name, compute_orientation));
    setInterfaceMesh(marker, std::move(iface));
}

void FESystem::setInterfaceMeshFromBoundaryLabel(InterfaceId marker,
                                                 int boundary_label,
                                                 bool compute_orientation)
{
    const auto& mesh = singleMesh("FESystem::setInterfaceMeshFromBoundaryLabel");
    auto iface = std::make_shared<svmp::InterfaceMesh>(
        svmp::InterfaceMesh::build_from_boundary_label(mesh.base(),
                                                       static_cast<svmp::label_t>(boundary_label),
                                                       compute_orientation));
    setInterfaceMesh(marker, std::move(iface));
}
#endif

void FESystem::invalidateSetup() noexcept
{
    is_setup_ = false;
    assembler_.reset();
    assembler_selection_report_.clear();
    material_state_provider_.reset();
    global_kernel_state_provider_.reset();
    sparsity_by_op_.clear();
    distributed_sparsity_by_op_.clear();
    dof_permutation_.reset();
    parameter_registry_.clear();
    use_backend_row_ownership_for_assembly_ = false;
    if (operator_backends_) {
        operator_backends_->invalidateCache();
    }
    assembly_plan_by_op_.clear();
    coupled_jac_cache_.clear();
    monolithic_aux_committed_rates_.clear();
    monolithic_aux_committed_rates_valid_.clear();

    // Clear setup-time auxiliary state hooks (sync, transfer) but
    // preserve block definitions and data.
    if (auxiliary_state_manager_) {
        auxiliary_state_manager_->invalidateSetup();
    }
    // Clear operator registry layout (rebuilt during setup).
    if (auxiliary_operator_registry_) {
        auxiliary_operator_registry_->clear();
    }

    // Clear setup-time analysis data that is rebuilt during setup().
    // Formulation records, BC descriptors, and definition-time contributions
    // are not cleared.
    // Only setup-time contributions (from kernel analysisContributions()) are
    // removed by truncating back to the definition-time watermark.
    contributions_.resize(contributions_def_count_);
    topology_context_.reset();
    interface_topology_context_.reset();
    auxiliary_region_lookup_cache_.reset();
    constraint_summary_.reset();

    // Note: GaugeRegistry is NOT cleared here. Candidate deduplication in
    // addCandidate() prevents accumulation on repeated setup(). Anchoring
    // evidence may accumulate from kernel sources, but resolve() overwrites
    // previous results. A full gauge lifecycle fix (clearing setup-time
    // evidence while preserving definition-time evidence) would require
    // a watermark pattern in GaugeRegistry itself.
    invalidateAnalysisCache();
}

void FESystem::requireSetup() const
{
    FE_THROW_IF(!is_setup_, InvalidStateException, "FESystem: setup() has not been called");
}

gauge::GaugeRegistry& FESystem::gaugeRegistry()
{
    if (!gauge_registry_) {
        gauge_registry_ = std::make_unique<gauge::GaugeRegistry>();
    }
    return *gauge_registry_;
}

// ============================================================================
// Problem analysis subsystem
// ============================================================================

void FESystem::addFormulationRecord(analysis::FormulationRecord record) {
    auxiliary_output_consumers_.insert(
        auxiliary_output_consumers_.end(),
        record.auxiliary_output_consumers.begin(),
        record.auxiliary_output_consumers.end());
    formulation_records_.push_back(std::move(record));
    invalidateAnalysisCache();
}

void FESystem::addBoundaryConditionDescriptor(analysis::BoundaryConditionDescriptor desc) {
    bc_descriptors_.push_back(std::move(desc));
    invalidateAnalysisCache();
}

void FESystem::addContribution(analysis::ContributionDescriptor desc) {
    contributions_.push_back(std::move(desc));
    // Track the definition-time watermark so invalidateSetup() preserves
    // contributions added before setup(). During setup(), the watermark is
    // frozen at the pre-setup level and setup-time contributions are added
    // above it.
    if (!is_setup_) {
        contributions_def_count_ = contributions_.size();
    }
    invalidateAnalysisCache();
}

void FESystem::addVariableDescriptor(analysis::VariableDescriptor desc) {
    variable_descriptors_.push_back(std::move(desc));
    invalidateAnalysisCache();
}

void FESystem::addInvariantDomainDescriptor(analysis::InvariantDomainDescriptor desc) {
    invariant_domain_descriptors_.push_back(std::move(desc));
    invalidateAnalysisCache();
}

void FESystem::setAnalysisSolverOptions(backends::SolverOptions options) {
    analysis_solver_options_.emplace(std::move(options));
    invalidateAnalysisCache();
}

void FESystem::clearAnalysisSolverOptions() {
    analysis_solver_options_.reset();
    invalidateAnalysisCache();
}

void FESystem::addAnalysisSummaries(analysis::AnalysisSummarySet summaries) {
    if (summaries.empty()) {
        return;
    }
    if (!registered_analysis_summaries_) {
        registered_analysis_summaries_.emplace();
    }
    appendRegisteredAnalysisSummarySet(*registered_analysis_summaries_, summaries);
    if (!analysis_summaries_) {
        analysis_summaries_.emplace();
    }
    appendRegisteredAnalysisSummarySet(*analysis_summaries_, summaries);
    invalidateAnalysisCache();
}

void FESystem::clearAnalysisSummaries() {
    registered_analysis_summaries_.reset();
    analysis_summaries_.reset();
    assembled_tangent_analysis_summary_attempted_ = false;
    invalidateAnalysisCache();
}

bool FESystem::updateAnalysisSummariesFromAssembledOperator(
    const backends::GenericMatrix& matrix,
    const OperatorTag& op,
    const SystemStateView* state)
{
    if (assembled_tangent_analysis_summary_attempted_) {
        return false;
    }

    assembled_tangent_analysis_summary_attempted_ = true;
    analysis_summaries_.reset();
    if (registered_analysis_summaries_) {
        analysis_summaries_.emplace(*registered_analysis_summaries_);
    }

    auto source = analysis::makeSparseRowScanSource(matrix);
    if (!source) {
        invalidateAnalysisCache();
        return false;
    }

    const auto baseline_report = runProblemAnalysisPlanOnly();
    const auto& plan = baseline_report.request_plan;
    if (plan.empty()) {
        invalidateAnalysisCache();
        return false;
    }

    analysis::AnalysisSummarySet summaries;
    if (registered_analysis_summaries_) {
        appendRegisteredAnalysisSummarySet(summaries, *registered_analysis_summaries_);
    }
    const auto registered_summary_count = summaries.totalSummaryCount();

    analysis::SparseMatrixScanOptions scan_options;
    scan_options.sign_tolerance = static_cast<Real>(1.0e-14);
    scan_options.row_sum_tolerance = static_cast<Real>(1.0e-10);
    scan_options.symmetry_tolerance = static_cast<Real>(1.0e-10);
#if FE_HAS_MPI
    if (source->isDistributed()) {
        scan_options.mpi_comm = MPI_COMM_WORLD;
    }
#endif

    const auto global_block = globalAnalysisBlock(op, field_map_);
    if (plan.has(analysis::AnalysisSummaryKind::DiscreteMatrix)) {
        summaries.discrete_matrices.push_back(
            analysis::scanSparseMatrixSummary(*source, global_block, scan_options).summary);
    }

    const auto constrained = affine_constraints_.getConstrainedDofs();
    std::unordered_set<GlobalIndex> constrained_lookup(constrained.begin(), constrained.end());
    if (plan.has(analysis::AnalysisSummaryKind::ReducedMatrix) &&
        matrix.numRows() == matrix.numCols()) {
        auto reduction = analysis::ConstraintReductionMask::fromConstrainedDofs(
            matrix.numRows(),
            constrained,
            analysis::ConstraintReductionKind::StrongDirichletElimination,
            /*affine_terms_accounted_for=*/false,
            /*reduction_exact_for_analysis=*/true);
        summaries.reduced_matrices.push_back(
            analysis::scanReducedFreeFreeSummary(*source,
                                                 reduction,
                                                 global_block,
                                                 scan_options));
    }

    std::optional<AnalysisSaddlePair> saddle_pair =
        findAnalysisSaddlePairFromSolverOptions(field_map_,
                                                analysis_solver_options_
                                                    ? &*analysis_solver_options_
                                                    : nullptr);
    if (!saddle_pair) {
        saddle_pair = findAnalysisSaddlePairFromContributions(field_map_, contributions_);
    }

    if (matrix.numRows() == matrix.numCols() &&
        (plan.has(analysis::AnalysisSummaryKind::NullspaceDegeneracy) ||
         plan.has(analysis::AnalysisSummaryKind::ReducedMatrix) ||
         plan.has(analysis::AnalysisSummaryKind::InfSupEstimate))) {
        const auto rank = estimateSparseRankFromRows(*source, constrained_lookup);
        summaries.nullspace_degeneracies.push_back(
            makeNullspaceDegeneracySummary(
                op,
                global_block,
                rank,
                static_cast<std::uint64_t>(constrained.size()),
                nullspaceClaimVariables(baseline_report,
                                        blockVariables(global_block)),
                baseline_report,
                saddle_pair && saddle_pair->valid()));
    }

    if (plan.has(analysis::AnalysisSummaryKind::Applicability) ||
        plan.has(analysis::AnalysisSummaryKind::DiscreteMatrix) ||
        plan.has(analysis::AnalysisSummaryKind::InvariantDomain) ||
        plan.has(analysis::AnalysisSummaryKind::InfSupEstimate) ||
        plan.has(analysis::AnalysisSummaryKind::SchurComplement)) {
        appendAutomaticApplicabilitySummaries(
            summaries,
            global_block,
            field_map_,
            contributions_,
            saddle_pair && saddle_pair->valid());
    }

    std::optional<CouplingMatrixStats> coupling_stats;
    if (saddle_pair && saddle_pair->valid() &&
        (plan.has(analysis::AnalysisSummaryKind::InfSupEstimate) ||
         plan.has(analysis::AnalysisSummaryKind::SchurComplement))) {
        coupling_stats = scanCouplingStats(*source,
                                           *saddle_pair,
                                           constrained_lookup,
                                           static_cast<Real>(1.0e-12));
    }

    if (saddle_pair && coupling_stats &&
        plan.has(analysis::AnalysisSummaryKind::InfSupEstimate)) {
        const auto block = saddlePairBlock(op, *saddle_pair);
        const bool multiplier_rows_covered =
            coupling_stats->active_multiplier_rows > 0u &&
            coupling_stats->multiplier_rows_with_primary_entries ==
                coupling_stats->active_multiplier_rows;
        const double estimate = multiplier_rows_covered
            ? coupling_stats->multiplier_min_row_norm
            : 0.0;

        analysis::InfSupEstimateSummary summary;
        summary.block = block;
        summary.primal_variable =
            analysis::VariableKey::field(saddle_pair->primary.field_id);
        summary.multiplier_variable =
            analysis::VariableKey::field(saddle_pair->multiplier.field_id);
        summary.estimate_value = static_cast<Real>(estimate);
        summary.estimate_tolerance = static_cast<Real>(1.0e-12);
        summary.uniform_lower_bound = static_cast<Real>(estimate);
        summary.test_rows =
            static_cast<GlobalIndex>(coupling_stats->active_multiplier_rows);
        summary.test_cols = saddle_pair->primary.size();
        summary.estimate_scope = "assembled coupling row-norm coverage";
        summary.nullspace_handling = constrained_lookup.empty()
            ? analysis::NullspaceHandlingClass::Unknown
            : analysis::NullspaceHandlingClass::AnchoredByConstraints;
        summary.estimator_metadata_present = true;
        summary.norm_metadata_present = false;
        summary.mesh_refinement_evidence_present = false;
        summary.uniform_lower_bound_evidence_present = false;
        summary.uniform_lower_bound_value_present = false;
        summaries.inf_sup_estimates.push_back(std::move(summary));
    }

    if (saddle_pair && coupling_stats &&
        plan.has(analysis::AnalysisSummaryKind::SchurComplement)) {
        analysis::SchurComplementSummary summary;
        summary.schur_id = op + ":assembled-saddle-blocks";
        summary.block = saddlePairBlock(op, *saddle_pair);
        summary.variables = summary.block.test_variables;
        summary.schur_available =
            coupling_stats->active_multiplier_rows > 0u &&
            coupling_stats->multiplier_rows_with_primary_entries > 0u &&
            coupling_stats->active_primary_rows > 0u;
        summary.reduction_exact_for_analysis = false;
        summary.primal_block_invertible_evidence_present =
            coupling_stats->active_primary_rows > 0u &&
            coupling_stats->primary_rows_with_diagonal ==
                coupling_stats->active_primary_rows &&
            coupling_stats->primary_min_abs_diag > 0.0;
        summary.inf_sup_evidence_present =
            coupling_stats->active_multiplier_rows > 0u &&
            coupling_stats->multiplier_rows_with_primary_entries ==
                coupling_stats->active_multiplier_rows &&
            coupling_stats->multiplier_min_row_norm > 0.0;
        summary.nullspace_handling = constrained_lookup.empty()
            ? analysis::NullspaceHandlingClass::Unknown
            : analysis::NullspaceHandlingClass::AnchoredByConstraints;
        summary.nullspace_handling_evidence_present =
            summary.nullspace_handling != analysis::NullspaceHandlingClass::Unknown;
        summary.schur_definiteness_evidence_present = false;
        summary.schur_positivity = analysis::PositivityClass::Unknown;
        if (coupling_stats->primary_min_abs_diag > 0.0 &&
            coupling_stats->primary_max_abs_diag >=
                coupling_stats->primary_min_abs_diag) {
            summary.condition_estimate_present = true;
            summary.condition_estimate =
                static_cast<Real>(coupling_stats->primary_max_abs_diag /
                                  coupling_stats->primary_min_abs_diag);
        }
        if (analysis_solver_options_) {
            summary.inexact_solve_tolerance_present = true;
            summary.inexact_solve_tolerance = analysis_solver_options_->rel_tol;
            summary.block_solve_tolerance_present = true;
            summary.block_solve_tolerance =
                analysis_solver_options_->fsils_blockschur_cg_rel_tol
                    .value_or(analysis_solver_options_->rel_tol);
            summary.approximate_schur_type = "assembled diagonal primary-block proxy";
            summary.preconditioner_type =
                std::string(backends::preconditionerToString(
                    analysis_solver_options_->preconditioner));
            if (analysis_solver_options_->method ==
                backends::SolverMethod::BlockSchur) {
                summary.preconditioner_type =
                    "BlockSchur/" + summary.preconditioner_type;
            }
        }
        if (summary.preconditioned_residual_contraction_present &&
            summary.preconditioned_residual_contraction <=
                std::max(summary.block_solve_tolerance, Real{1.0e-12})) {
            summary.condition_risk_class =
                analysis::SchurQualityClass::PreconditionedRobust;
        } else if (summary.condition_estimate_present &&
                   !summary.preconditioned_probe_available) {
            summary.condition_risk_class =
                analysis::SchurQualityClass::RawConditionOnly;
        } else if (!summary.schur_available) {
            summary.condition_risk_class =
                analysis::SchurQualityClass::Unavailable;
        } else {
            summary.condition_risk_class =
                analysis::SchurQualityClass::InsufficientEvidence;
        }
        summaries.schur_complements.push_back(std::move(summary));
    }

    if (plan.has(analysis::AnalysisSummaryKind::StabilizationAdequacy)) {
        for (const auto& contribution : contributions_) {
            if (contribution.role != analysis::ContributionRole::StabilizationBlock) {
                continue;
            }
            analysis::StabilizationAdequacySummary summary;
            summary.stabilization_id = contribution.contribution_id.empty()
                ? contribution.operator_tag
                : contribution.contribution_id;
            if (summary.stabilization_id.empty()) {
                summary.stabilization_id = op + ":stabilization";
            }
            summary.method_family = "assembled-stabilization-contribution";
            summary.block = contributionBlock(contribution, op);
            summary.variables = !contribution.related_variables.empty()
                ? contribution.related_variables
                : contribution.test_variables;
            summary.residual_consistency_evidence_present =
                contribution.consistency_kind.has_value() &&
                *contribution.consistency_kind != analysis::ConsistencyKind::Unknown;
            summary.method_scope_metadata_present = true;
            if (contribution.scaling.has_value()) {
                summary.parameter_formula_metadata_present = true;
                summary.scaling_law_metadata_present = true;
                summary.stabilization_theorem_id =
                    "ContributionDescriptor::ScalingDescriptor";
            }
            if (contribution.transport_character.has_value()) {
                summary.regime_metadata_present = true;
            }
            if (contribution.temporal.has_value() &&
                contribution.temporal->derivative_order > 0) {
                summary.cfl_scope = "ContributionDescriptor::TemporalDescriptor";
            }
            summary.violation_count = 0u;
            summaries.stabilization_adequacy.push_back(std::move(summary));
        }
    }

    const bool matrix_summaries_generated =
        !summaries.discrete_matrices.empty() ||
        !summaries.reduced_matrices.empty();
    if (mesh_access_ &&
        (plan.has(analysis::AnalysisSummaryKind::MeshGeometryQuality) ||
         matrix_summaries_generated)) {
        if (auto summary = buildMeshGeometryQualitySummary(*mesh_access_)) {
            summaries.mesh_geometry_quality.push_back(std::move(*summary));
        }
    }

    std::vector<analysis::CoefficientPropertySummary>
        runtime_coefficient_summaries;
    if (plan.has(analysis::AnalysisSummaryKind::CoefficientProperties) ||
        plan.has(analysis::AnalysisSummaryKind::ParameterScale)) {
        runtime_coefficient_summaries = buildRuntimeCoefficientSummaries(
            contributions_,
            formulation_records_,
            op,
            parameter_registry_,
            mesh_access_.get(),
            state);
    }
    if (plan.has(analysis::AnalysisSummaryKind::CoefficientProperties)) {
        for (auto& summary : runtime_coefficient_summaries) {
            addCoefficientSummaryIfAbsent(summaries.coefficient_properties,
                                          summary);
        }
        const auto* matrix_fallback = summaries.discrete_matrices.empty()
            ? nullptr
            : &summaries.discrete_matrices.front();
        applyGlobalSpectrumFallback(summaries.coefficient_properties,
                                    matrix_fallback);
    }

    std::unordered_map<std::string, std::pair<Real, Real>>
        coefficient_abs_ranges;
    for (const auto& summary : summaries.coefficient_properties) {
        const Real min_abs = std::min(std::abs(summary.min_eigenvalue),
                                      std::abs(summary.max_eigenvalue));
        const Real max_abs = std::max(std::abs(summary.min_eigenvalue),
                                      std::abs(summary.max_eigenvalue));
        if (!summary.coefficient.empty() &&
            std::isfinite(static_cast<double>(min_abs)) &&
            std::isfinite(static_cast<double>(max_abs))) {
            auto& range = coefficient_abs_ranges[summary.coefficient];
            if (range.first == Real{} || min_abs < range.first) {
                range.first = min_abs;
            }
            range.second = std::max(range.second, max_abs);
        }
    }
    for (const auto& summary : runtime_coefficient_summaries) {
        const Real min_abs = std::min(std::abs(summary.min_eigenvalue),
                                      std::abs(summary.max_eigenvalue));
        const Real max_abs = std::max(std::abs(summary.min_eigenvalue),
                                      std::abs(summary.max_eigenvalue));
        if (!summary.coefficient.empty() &&
            std::isfinite(static_cast<double>(min_abs)) &&
            std::isfinite(static_cast<double>(max_abs))) {
            auto& range = coefficient_abs_ranges[summary.coefficient];
            if (range.first == Real{} || min_abs < range.first) {
                range.first = min_abs;
            }
            range.second = std::max(range.second, max_abs);
        }
    }

    if (plan.has(analysis::AnalysisSummaryKind::ParameterScale)) {
        for (const auto& contribution : contributions_) {
            if (contribution.scale_usages.empty() &&
                !contribution.scaling.has_value()) {
                continue;
            }
            auto variables = contributionVariables(contribution);
            auto block = contributionBlock(contribution, op);
            if (!contribution.scale_usages.empty()) {
                int scale_index = 0;
                for (const auto& scale : contribution.scale_usages) {
                    auto scale_block = block;
                    auto scale_variables = variables;
                    auto id = contributionIdentity(contribution, op) +
                              ":scale:" + std::to_string(scale_index++);
                    summaries.parameter_scales.push_back(
                        makeRuntimeParameterScaleSummary(
                            std::move(id),
                            parameterScaleRoleFor(contribution),
                            std::move(scale_block),
                            std::move(scale_variables),
                            scale,
                            field_registry_,
                            parameter_registry_,
                            coefficient_abs_ranges,
                            mesh_access_.get(),
                            state));
                }
            } else {
                auto id = contributionIdentity(contribution, op) + ":scale";
                summaries.parameter_scales.push_back(
                    makeParameterScaleSummary(std::move(id),
                                              parameterScaleRoleFor(contribution),
                                              std::move(block),
                                              std::move(variables),
                                              *contribution.scaling,
                                              field_registry_));
            }
        }

        for (const auto& bc : bc_descriptors_) {
            if (!bc.scaling.has_value()) {
                continue;
            }
            std::vector<analysis::VariableKey> variables{bc.primary_variable};
            for (const auto& variable : bc.related_variables) {
                if (std::find(variables.begin(), variables.end(), variable) ==
                    variables.end()) {
                    variables.push_back(variable);
                }
            }
            auto block = boundaryConditionBlock(bc, op);
            auto id = (bc.source.empty() ? op : bc.source) + ":scale";
            const auto role = isWeakBoundaryEnforcement(bc.enforcement_kind)
                ? analysis::ParameterScaleRole::WeakBoundaryPenalty
                : analysis::ParameterScaleRole::Generic;
            summaries.parameter_scales.push_back(
                makeParameterScaleSummary(std::move(id),
                                          role,
                                          std::move(block),
                                          std::move(variables),
                                          *bc.scaling,
                                          field_registry_));
        }
    }

    if (plan.has(analysis::AnalysisSummaryKind::InvariantDomain)) {
        for (const auto& descriptor : invariant_domain_descriptors_) {
            appendInvariantDomainSummaryIfAbsent(
                summaries, descriptor, field_map_, state);
        }
        for (const auto& record : formulation_records_) {
            for (const auto& descriptor : record.invariant_domain_descriptors) {
                appendInvariantDomainSummaryIfAbsent(
                    summaries, descriptor, field_map_, state);
            }
        }
    }

    if (plan.has(analysis::AnalysisSummaryKind::BoundarySymbol)) {
        for (const auto& bc : bc_descriptors_) {
            if (bc.domain != analysis::DomainKind::Boundary &&
                bc.domain != analysis::DomainKind::InterfaceFace &&
                bc.domain != analysis::DomainKind::CoupledBoundary) {
                continue;
            }
            analysis::BoundarySymbolSummary summary;
            summary.block = boundaryConditionBlock(bc, op);
            const bool flux_like =
                bc.trace_kind == analysis::TraceKind::Flux ||
                bc.trace_kind == analysis::TraceKind::NormalFlux ||
                bc.trace_kind == analysis::TraceKind::Mixed;
            summary.principal_operator_order = flux_like ? 1 : 0;
            summary.boundary_operator_order = flux_like ? 1 : 0;
            summary.trace_coverage = traceCoverageForBoundary(bc.trace_kind);
            summary.evidence_scope = bc.source.empty()
                ? ("marker=" + std::to_string(summary.block.marker))
                : bc.source;
            summary.boundary_condition_count = 1u;
            summary.required_boundary_condition_count = 1u;
            summary.missing_symbol_count =
                summary.trace_coverage == analysis::TraceCapabilityFlags::None
                    ? 1u
                    : 0u;
            summary.component_coverage_complete = true;
            summary.dof_coverage_complete = true;
            summaries.boundary_symbols.push_back(std::move(summary));
        }
    }

    if (plan.has(analysis::AnalysisSummaryKind::TemporalStability) &&
        hasTemporalMetadata(contributions_, variable_descriptors_)) {
        analysis::TemporalStabilitySummary summary;
        summary.time_scheme = "FE temporal metadata";
        summary.block = global_block;
        summary.variables = variablesFromDescriptors(variable_descriptors_);
        if (summary.variables.empty()) {
            summary.variables = global_block.test_variables;
        }
        summary.operator_scope_id = op;
        summary.operator_scope_metadata_present = !op.empty();
        summaries.temporal_stability.push_back(std::move(summary));
    }

    if (plan.has(analysis::AnalysisSummaryKind::DAEStructureEvidence) &&
        hasTemporalMetadata(contributions_, variable_descriptors_)) {
        analysis::DAEStructureEvidenceSummary summary;
        summary.system_id = op.empty() ? "assembled-system" : op;
        summary.variables = variablesFromDescriptors(variable_descriptors_);
        if (summary.variables.empty()) {
            summary.variables = global_block.test_variables;
        }
        summary.dae_index_scope = "FE temporal descriptors";
        summaries.dae_structure_evidence.push_back(std::move(summary));
    }

    if (plan.has(analysis::AnalysisSummaryKind::InitialCompatibility)) {
        const bool has_constraint_scope =
            constraint_summary_ && !constraint_summary_->constrained_sets.empty();
        const bool has_boundary_scope = !bc_descriptors_.empty();
        if (has_constraint_scope || has_boundary_scope) {
            analysis::InitialCompatibilitySummary summary;
            summary.compatibility_scope = "FE constraint and boundary metadata";
            summary.algebraic_constraint_metadata_present = has_constraint_scope;
            summary.boundary_constraint_metadata_present = has_boundary_scope;
            summary.checked_constraint_family_count = has_constraint_scope
                ? static_cast<std::uint64_t>(
                      constraint_summary_->constrained_sets.size())
                : 0u;
            summary.checked_boundary_condition_count =
                static_cast<std::uint64_t>(bc_descriptors_.size());
            summaries.initial_compatibility.push_back(std::move(summary));
        }
    }

    if (plan.has(analysis::AnalysisSummaryKind::CoupledSystemStability)) {
        std::vector<analysis::VariableKey> variables;
        bool has_nonlocal_coupling = false;
        for (const auto& contribution : contributions_) {
            if (!contributionIsNonlocalCoupling(contribution)) {
                continue;
            }
            has_nonlocal_coupling = true;
            for (const auto& variable : contributionVariables(contribution)) {
                if (std::find(variables.begin(), variables.end(), variable) ==
                    variables.end()) {
                    variables.push_back(variable);
                }
            }
        }
        if (has_nonlocal_coupling) {
            analysis::CoupledSystemStabilitySummary summary;
            summary.coupling_group = op.empty() ? "nonlocal-fe-coupling" : op;
            summary.variables = std::move(variables);
            summary.monolithic_coupling = true;
            summary.coupling_operator_scope_id = summary.coupling_group;
            summary.coupling_operator_scope_metadata_present = true;
            summaries.coupled_system_stability.push_back(std::move(summary));
        }
    }

    if (plan.has(analysis::AnalysisSummaryKind::RobustnessTrend) ||
        plan.has(analysis::AnalysisSummaryKind::InfSupEstimate) ||
        plan.has(analysis::AnalysisSummaryKind::SchurComplement) ||
        plan.has(analysis::AnalysisSummaryKind::StabilizationAdequacy)) {
        appendAutomaticRobustnessTrends(summaries, op, matrix.numRows());
    }

    if (plan.has(analysis::AnalysisSummaryKind::NumericalErrorBudget) ||
        (analysis_solver_options_ && matrix_summaries_generated)) {
        const auto* matrix_summary = summaries.discrete_matrices.empty()
            ? nullptr
            : &summaries.discrete_matrices.front();
        summaries.numerical_error_budgets.push_back(
            makeNumericalErrorBudgetSummary(
                op,
                global_block,
                blockVariables(global_block),
                matrix_summary,
                analysis_solver_options_ ? &*analysis_solver_options_ : nullptr));
    }

    if (!summaries.empty()) {
        analysis_summaries_.emplace(std::move(summaries));
    }
    invalidateAnalysisCache();
    if (analysis_summaries_) {
#if FE_HAS_MPI
        if (mpiWorldActiveForAnalysis()) {
            return analysis_summaries_->totalSummaryCount() > registered_summary_count;
        }
#endif
        analysis_report_cache_ = runProblemAnalysis();
        analysis_report_version_ = analysis_inputs_version_;
        return analysis_summaries_->totalSummaryCount() > registered_summary_count;
    }
    return false;
}

void FESystem::buildTopologyContext() {
    topology_context_ = analysis::TopologyAnalysisContext::build(meshAccess());
    auxiliary_region_lookup_cache_.reset();
    invalidateAnalysisCache();
}

void FESystem::buildInterfaceTopologyContext() {
    analysis::InterfaceTopologyContext ctx;

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    for (const auto& [marker, imesh] : interface_meshes_) {
        if (!imesh) continue;

        const auto n_faces = static_cast<GlobalIndex>(imesh->n_faces());
        for (GlobalIndex f = 0; f < n_faces; ++f) {
            auto local_f = static_cast<MeshIndex>(f);
            analysis::InterfaceFaceRecord rec;
            rec.interface_marker = marker;

            auto cells = imesh->volume_cells(local_f);
            rec.minus_cell = static_cast<GlobalIndex>(cells[0]);
            rec.plus_cell = static_cast<GlobalIndex>(cells[1]);
            rec.is_two_sided = !imesh->is_boundary_face(local_f);
            rec.has_orientation = imesh->has_orientation();

            if (rec.is_two_sided) {
                rec.minus_local_face = imesh->local_face_in_cell_minus(local_f);
                rec.plus_local_face = imesh->local_face_in_cell_plus(local_f);
            } else {
                rec.minus_local_face = imesh->local_face_in_cell(local_f);
            }

            // Annotate with bulk region IDs if topology context is available
            if (topology_context_) {
                if (rec.minus_cell != INVALID_GLOBAL_INDEX) {
                    rec.minus_region = topology_context_->regionForCell(rec.minus_cell);
                }
                if (rec.plus_cell != INVALID_GLOBAL_INDEX) {
                    rec.plus_region = topology_context_->regionForCell(rec.plus_cell);
                }
            }

            auto face_idx = ctx.faces.size();
            ctx.faces.push_back(std::move(rec));
            ctx.marker_to_faces[marker].push_back(face_idx);
        }
    }
#endif

    interface_topology_context_ = std::move(ctx);
    auxiliary_region_lookup_cache_.reset();
    invalidateAnalysisCache();
}

void FESystem::buildConstraintSummary() {
    std::vector<analysis::ConstraintAnalysisSummary::FieldDofRange> ranges;
    for (const auto& fr : field_registry_.records()) {
        analysis::ConstraintAnalysisSummary::FieldDofRange r;
        r.field_id = fr.id;
        // Field DOF offsets are only valid after setup
        if (is_setup_ && fr.id < field_dof_offsets_.size()) {
            r.dof_offset = field_dof_offsets_[fr.id];
            r.num_dofs = field_dof_handlers_[fr.id].getStatistics().total_dofs;
            r.num_components = fr.components;
        }
        ranges.push_back(r);
    }

    // Build a DOF→region provider when topology is available.
    // Uses the EntityDofMap to map DOF → entity → cell → region.
    // Handles vertex, edge, face, and cell entities.
    const auto* topo = topology_context_ ? &*topology_context_ : nullptr;
    analysis::ConstraintAnalysisSummary::DofRegionProvider dof_region;
    if (topo && topo->numRegions() > 1) {
        const auto* emap = dof_handler_.getEntityDofMap();
        if (emap && mesh_access_) {
            // Pre-build vertex→cell map for O(1) lookup instead of O(n_cells) per DOF
            const auto n_cells = meshAccess().numCells();
            auto vertex_to_cell = std::make_shared<std::unordered_map<GlobalIndex, GlobalIndex>>();
            {
                std::vector<GlobalIndex> nodes;
                for (GlobalIndex c = 0; c < n_cells; ++c) {
                    nodes.clear();
                    meshAccess().getCellNodes(c, nodes);
                    for (auto n : nodes) {
                        vertex_to_cell->emplace(n, c);  // first cell wins
                    }
                }
            }

            dof_region = [topo, emap, vertex_to_cell, n_cells, this](GlobalIndex dof) -> int {
                auto ent = emap->getDofEntity(dof);
                if (!ent) return -1;

                switch (ent->kind) {
                    case dofs::EntityKind::Vertex: {
                        auto it = vertex_to_cell->find(ent->id);
                        if (it != vertex_to_cell->end()) {
                            return topo->regionForCell(it->second);
                        }
                        return -1;
                    }
                    case dofs::EntityKind::Cell: {
                        // Cell DOF — entity ID is the cell index
                        return topo->regionForCell(ent->id);
                    }
                    default: {
                        // Edge/Face DOFs: find an incident cell by scanning.
                        // This is O(n_cells) per DOF but only runs once during
                        // constraint summary build. For large meshes, a
                        // pre-built edge/face→cell map would be more efficient.
                        const auto& dmap = dof_handler_.getDofMap();
                        for (GlobalIndex c = 0; c < n_cells; ++c) {
                            auto cell_dofs = dmap.getCellDofs(c);
                            for (auto cd : cell_dofs) {
                                if (cd == dof) {
                                    return topo->regionForCell(c);
                                }
                            }
                        }
                        return -1;
                    }
                }
            };
        }
    }

    // Build component DOF provider from FieldDofMap.
    // Uses getComponentDofs() which works for any layout (component-blocked,
    // interleaved, or vector-basis). Returns empty for VectorBasis fields
    // where component extraction is not defined.
    analysis::ConstraintAnalysisSummary::ComponentDofProvider comp_dofs;
    if (is_setup_ && field_map_.numFields() > 0) {
        comp_dofs = [this](FieldId fid, int component) -> std::vector<GlobalIndex> {
            auto field_idx = static_cast<std::size_t>(fid);
            if (field_idx >= field_map_.numFields()) return {};
            const auto& fd = field_map_.getField(field_idx);
            if (fd.component_dof_layout != dofs::FieldComponentDofLayout::ComponentWise) return {};
            if (component < 0 || static_cast<LocalIndex>(component) >= fd.n_components) return {};
            auto idx_set = field_map_.getComponentDofs(field_idx, static_cast<LocalIndex>(component));
            return idx_set.toVector();
        };
    }

    constraint_summary_ = analysis::ConstraintAnalysisSummary::build(
        affine_constraints_, ranges, topo, dof_region, comp_dofs);
    invalidateAnalysisCache();
}

void FESystem::invalidateAnalysisCache() noexcept {
    ++analysis_inputs_version_;
}

analysis::ProblemAnalysisContext FESystem::buildProblemAnalysisContext() const {
    analysis::ProblemAnalysisContext ctx;

    // Populate field descriptors from FieldRegistry.
    for (const auto& fr : field_registry_.records()) {
        analysis::FieldDescriptor fd;
        fd.field_id = fr.id;
        fd.name = fr.name;
        fd.value_dimension = fr.components;
        fd.field_type = (fr.components > 1) ? FieldType::Vector : FieldType::Scalar;
        fd.domain = (fr.scope == FieldScope::InterfaceFace)
                        ? analysis::DomainKind::InterfaceFace
                        : analysis::DomainKind::Cell;
        fd.interface_marker = fr.interface_marker;
        if (fr.space) {
            fd.polynomial_order = fr.space->polynomial_order();
            fd.topological_dimension = fr.space->topological_dimension();
            fd.continuity = fr.space->continuity();
            fd.component_polynomial_orders.assign(
                static_cast<std::size_t>(std::max(1, fd.value_dimension)),
                fd.polynomial_order);
            fd.space_family = spaceFamilyFor(fd.continuity, fr.space->space_type());
            fd.element_family = elementFamilyFor(*fr.space);
            fd.continuity_class =
                continuityClassFor(fd.continuity, fr.space->space_type());
            fd.mapping_transform =
                mappingTransformFor(fd.continuity, fr.space->space_type());
            fd.reference_cell_family =
                referenceCellFamilyFor(fr.space->element_type());
            if (fd.reference_cell_family != analysis::ReferenceCellFamily::Unknown) {
                fd.mesh_family_scope = analysis::toString(fd.reference_cell_family);
            }
            fd.conformity.exact_sequence_member =
                fd.continuity == Continuity::H_div ||
                fd.continuity == Continuity::H_curl;
            fd.conformity.facet_orientation_consistent =
                fd.continuity == Continuity::H_div ||
                fd.continuity == Continuity::H_curl;
            fd.conformity.entity_dof_association_known =
                fd.element_family != analysis::ElementFamily::Unknown &&
                fd.element_family != analysis::ElementFamily::Custom;
            fd.conformity.commuting_projection_metadata_present =
                fd.conformity.exact_sequence_member &&
                (fd.element_family == analysis::ElementFamily::RaviartThomas ||
                 fd.element_family == analysis::ElementFamily::BDM ||
                 fd.element_family == analysis::ElementFamily::Nedelec);
            if (fd.element_family ==
                analysis::ElementFamily::BubbleEnrichedLagrange) {
                fd.enrichment.bubble_degree = fd.polynomial_order;
                fd.enrichment.visible_to_analysis = true;
            }

            // Derive component_extractable from the function space continuity.
            // H(div) and H(curl) spaces use vector-valued basis functions where
            // DOFs are NOT per-component — component extraction is not defined.
            // This works both pre-setup and post-setup.
            if (fd.continuity == Continuity::H_div ||
                fd.continuity == Continuity::H_curl) {
                fd.component_extractable = false;
            }

            // Phase 21: space family and trace capabilities from continuity
            switch (fd.continuity) {
                case Continuity::C0:
                case Continuity::C1:
                    fd.trace_capabilities = analysis::TraceCapabilityFlags::Value
                                          | analysis::TraceCapabilityFlags::NormalFlux;
                    break;
                case Continuity::H_div:
                    fd.trace_capabilities = analysis::TraceCapabilityFlags::NormalComponent
                                          | analysis::TraceCapabilityFlags::NormalFlux;
                    fd.has_exact_sequence_structure = true;
                    fd.supports_local_balance_closure = true;
                    break;
                case Continuity::H_curl:
                    fd.trace_capabilities = analysis::TraceCapabilityFlags::TangentialComponent;
                    fd.has_exact_sequence_structure = true;
                    break;
                case Continuity::L2:
                    fd.trace_capabilities = analysis::TraceCapabilityFlags::Jump
                                          | analysis::TraceCapabilityFlags::Average;
                    break;
                default:
                    break;
            }
        }
        applyBoundaryScopeMetadata(fd, bc_descriptors_);
        // Post-setup refinement: use the actual FieldDofMap layout descriptor
        // which is authoritative (handles edge cases like custom spaces).
        if (is_setup_ && fr.id < field_map_.numFields()) {
            const auto& fmd = field_map_.getField(static_cast<std::size_t>(fr.id));
            fd.component_extractable =
                (fmd.component_dof_layout == dofs::FieldComponentDofLayout::ComponentWise);
        }
        ctx.addFieldDescriptor(std::move(fd));
    }

    // Populate variable descriptors.
    for (const auto& vd : variable_descriptors_) {
        ctx.addVariableDescriptor(vd);
    }

    // Populate formulation records.
    for (const auto& rec : formulation_records_) {
        ctx.addFormulationRecord(rec);
    }

    // Populate normalized contributions.
    for (const auto& c : contributions_) {
        ctx.addContribution(c);
    }

    // Populate BC descriptors.
    for (const auto& desc : bc_descriptors_) {
        ctx.addBCDescriptor(desc);
    }

    // Populate topology context if available.
    if (topology_context_) {
        ctx.setTopologyContext(*topology_context_);
    }

    // Populate interface topology if available. Generated level-set interfaces
    // are backed by cut-interface quadrature rather than InterfaceMesh faces, so
    // expose their markers to analysis as covered embedded interfaces.
    if (interface_topology_context_ || cut_integration_context_ ||
        !generated_embedded_interface_markers_.empty()) {
        analysis::InterfaceTopologyContext interface_ctx =
            interface_topology_context_ ? *interface_topology_context_
                                        : analysis::InterfaceTopologyContext{};
        for (const int marker : generated_embedded_interface_markers_) {
            interface_ctx.addGeneratedEmbeddedMarker(marker);
        }
        if (cut_integration_context_) {
            for (const int marker :
                 cut_integration_context_->generatedInterfaceMarkers()) {
                interface_ctx.addGeneratedEmbeddedMarker(marker);
            }
        }
        ctx.setInterfaceTopologyContext(std::move(interface_ctx));
    }

    // Populate constraint summary if available.
    if (constraint_summary_) {
        ctx.setConstraintSummary(*constraint_summary_);
    }

    if (analysis_solver_options_) {
        ctx.setSolverOptions(*analysis_solver_options_);
    }

    if (analysis_summaries_) {
        ctx.setAnalysisSummaries(*analysis_summaries_);
    }

    return ctx;
}

analysis::ProblemAnalysisReport FESystem::runProblemAnalysisPlanOnly() const {
    auto analyzer = analysis::ProblemAnalyzer::createDefault();
    return analyzer.analyze(buildProblemAnalysisContext());
}

analysis::ProblemAnalysisReport FESystem::runProblemAnalysis() const {
    auto ctx = buildProblemAnalysisContext();
    auto analyzer = analysis::ProblemAnalyzer::createDefault();
    auto producer_registry =
        analysis::AnalysisSummaryProducerRegistry::createDefault();
    const auto* summaries =
        analysis_summaries_ ? &*analysis_summaries_ : nullptr;
    const bool assembled_backend_pending =
        !summariesHaveAssembledEvidence(summaries);
    const auto revision = operatorRevisionSnapshot();
    std::ostringstream mesh_revision;
    mesh_revision << "mesh:g" << revision.mesh.geometry
                  << ":t" << revision.mesh.topology
                  << ":o" << revision.mesh.ownership
                  << ":n" << revision.mesh.numbering
                  << ":layout" << revision.mesh.field_layout;
    FESystemAnalysisAssemblyAccess assembly_access(summaries,
                                                   assembled_backend_pending);
    FESystemAnalysisMeshAccess mesh_access(mesh_access_.get(),
                                           summaries,
                                           mesh_revision.str(),
                                           mesh_access_ == nullptr);
    FESystemAnalysisSolverAccess solver_access(summaries,
                                               assembled_backend_pending);
    return analyzer.analyzeWithEvidenceSynthesis(ctx,
                                                 producer_registry,
                                                 assembly_access,
                                                 mesh_access,
                                                 solver_access);
}

const analysis::ProblemAnalysisReport& FESystem::analysisReport() const {
    if (analysis_report_version_ != analysis_inputs_version_) {
        analysis_report_cache_ = runProblemAnalysis();
        analysis_report_version_ = analysis_inputs_version_;
    }
    return *analysis_report_cache_;
}

const FieldRecord& FESystem::singleField() const
{
    FE_THROW_IF(field_registry_.size() != 1u, NotImplementedException,
                "FESystem::singleField: this operation currently requires exactly one field");
    return field_registry_.records().front();
}

void FESystem::requireSingleFieldSetup() const
{
    requireSetup();
    (void)singleField();
}

FieldId FESystem::addField(FieldSpec spec)
{
    invalidateSetup();
    if (spec.components <= 0) {
        spec.components = spec.space ? spec.space->value_dimension() : 1;
    }
    if (spec.space) {
        FE_THROW_IF(spec.components != spec.space->value_dimension(), InvalidArgumentException,
                    "FESystem::addField: FieldSpec.components must match FunctionSpace::value_dimension()");
    }
    auto field = field_registry_.add(std::move(spec));
    if (static_cast<std::size_t>(field) >= prescribed_field_buffers_.size()) {
        prescribed_field_buffers_.resize(static_cast<std::size_t>(field) + 1u);
    }
    bumpSpaceRevision();
    bumpBlockLayoutRevision();
    return field;
}

FieldId FESystem::addMeshMotionDataField(std::string name,
                                         std::shared_ptr<const spaces::FunctionSpace> space,
                                         int components)
{
    FieldSpec spec;
    spec.name = std::move(name);
    spec.space = std::move(space);
    spec.components = components;
    spec.source_kind = FieldSourceKind::PrescribedData;
    return addField(std::move(spec));
}

FieldId FESystem::addDerivedMeshVelocityField(std::string name,
                                              std::shared_ptr<const spaces::FunctionSpace> space,
                                              FieldId mesh_displacement_field,
                                              int components)
{
    FE_THROW_IF(!field_registry_.has(mesh_displacement_field), InvalidArgumentException,
                "FESystem::addDerivedMeshVelocityField: unknown mesh displacement field");
    const auto& src = field_registry_.get(mesh_displacement_field);
    FE_THROW_IF(src.source_kind != FieldSourceKind::Unknown, InvalidArgumentException,
                "FESystem::addDerivedMeshVelocityField: mesh displacement source must be an Unknown field");
    FE_CHECK_NOT_NULL(src.space.get(),
                      "FESystem::addDerivedMeshVelocityField: mesh displacement source space");
    if (!space) {
        space = src.space;
    }
    FE_THROW_IF(space.get() != src.space.get(), InvalidArgumentException,
                "FESystem::addDerivedMeshVelocityField: derived mesh velocity must use the "
                "same FE space as its mesh-displacement source");
    if (components == 0) {
        components = src.components;
    }
    FE_THROW_IF(components != src.components, InvalidArgumentException,
                "FESystem::addDerivedMeshVelocityField: component count must match the "
                "mesh-displacement source");

    FieldSpec spec;
    spec.name = std::move(name);
    spec.space = std::move(space);
    spec.components = components;
    spec.source_kind = FieldSourceKind::DerivedFromUnknown;
    spec.derived.source_field = mesh_displacement_field;
    spec.derived.role = DerivedFieldRole::TimeDerivative;
    spec.derived.derivative_order = 1;
    return addField(std::move(spec));
}

bool FESystem::fieldParticipatesInUnknownVector(FieldId field) const
{
    const auto& rec = field_registry_.get(field);
    return rec.source_kind == FieldSourceKind::Unknown;
}

void FESystem::setPrescribedFieldCoefficients(FieldId field, std::span<const Real> coefficients)
{
    requireSetup();
    const auto& rec = field_registry_.get(field);
    FE_THROW_IF(rec.source_kind != FieldSourceKind::PrescribedData, InvalidArgumentException,
                "FESystem::setPrescribedFieldCoefficients: field '" + rec.name +
                    "' is not a prescribed data field");
    const auto idx = static_cast<std::size_t>(field);
    FE_THROW_IF(idx >= field_dof_handlers_.size(), InvalidStateException,
                "FESystem::setPrescribedFieldCoefficients: field DOFs are not finalized");
    const auto expected = static_cast<std::size_t>(field_dof_handlers_[idx].getNumDofs());
    FE_THROW_IF(coefficients.size() != expected, InvalidArgumentException,
                "FESystem::setPrescribedFieldCoefficients: field '" + rec.name +
                    "' expects " + std::to_string(expected) + " coefficients, got " +
                    std::to_string(coefficients.size()));
    if (idx >= prescribed_field_buffers_.size()) {
        prescribed_field_buffers_.resize(idx + 1u);
    }
    auto& buffer = prescribed_field_buffers_[idx];
    buffer.coefficients.assign(coefficients.begin(), coefficients.end());
    ++buffer.revision;
}

void FESystem::clearPrescribedFieldCoefficients(FieldId field)
{
    const auto& rec = field_registry_.get(field);
    FE_THROW_IF(rec.source_kind != FieldSourceKind::PrescribedData, InvalidArgumentException,
                "FESystem::clearPrescribedFieldCoefficients: field '" + rec.name +
                    "' is not a prescribed data field");
    const auto idx = static_cast<std::size_t>(field);
    if (idx >= prescribed_field_buffers_.size()) {
        return;
    }
    prescribed_field_buffers_[idx].coefficients.clear();
    ++prescribed_field_buffers_[idx].revision;
}

std::span<const Real> FESystem::prescribedFieldCoefficients(FieldId field) const
{
    const auto& rec = field_registry_.get(field);
    FE_THROW_IF(rec.source_kind != FieldSourceKind::PrescribedData, InvalidArgumentException,
                "FESystem::prescribedFieldCoefficients: field '" + rec.name +
                    "' is not a prescribed data field");
    const auto idx = static_cast<std::size_t>(field);
    if (idx >= prescribed_field_buffers_.size()) {
        return {};
    }
    return std::span<const Real>(prescribed_field_buffers_[idx].coefficients);
}

std::uint64_t FESystem::prescribedFieldRevision(FieldId field) const
{
    const auto& rec = field_registry_.get(field);
    FE_THROW_IF(rec.source_kind != FieldSourceKind::PrescribedData, InvalidArgumentException,
                "FESystem::prescribedFieldRevision: field '" + rec.name +
                    "' is not a prescribed data field");
    const auto idx = static_cast<std::size_t>(field);
    if (idx >= prescribed_field_buffers_.size()) {
        return 0;
    }
    return prescribed_field_buffers_[idx].revision;
}

FieldId FESystem::findFieldByName(std::string_view name) const noexcept
{
    return field_registry_.findByName(name);
}

bool FESystem::hasField(std::string_view name) const noexcept
{
    return findFieldByName(name) != INVALID_FIELD_ID;
}

std::vector<FieldId> FESystem::unknownFieldIdsInDofMapOrder() const
{
    requireSetup();

    std::vector<FieldId> fields;
    fields.reserve(field_map_.numFields());
    for (std::size_t i = 0; i < field_map_.numFields(); ++i) {
        const auto& descriptor = field_map_.getField(i);
        const auto field = findFieldByName(descriptor.name);
        FE_THROW_IF(field == INVALID_FIELD_ID, InvalidStateException,
                    "FESystem::unknownFieldIdsInDofMapOrder: field map references unknown field '" +
                        descriptor.name + "'");
        FE_THROW_IF(!fieldParticipatesInUnknownVector(field), InvalidStateException,
                    "FESystem::unknownFieldIdsInDofMapOrder: field map references non-unknown field '" +
                        descriptor.name + "'");
        fields.push_back(field);
    }
    return fields;
}

namespace {

MeshMotionFieldRole parseMeshMotionFieldRole(std::string_view role_name)
{
    if (role_name == "mesh_displacement" || role_name == "displacement") {
        return MeshMotionFieldRole::Displacement;
    }
    if (role_name == "mesh_velocity" || role_name == "velocity") {
        return MeshMotionFieldRole::Velocity;
    }
    if (role_name == "mesh_acceleration" || role_name == "acceleration") {
        return MeshMotionFieldRole::Acceleration;
    }
    if (role_name == "previous_coordinates" ||
        role_name == "previous_coordinate" ||
        role_name == "previous_coords") {
        return MeshMotionFieldRole::PreviousCoordinates;
    }
    if (role_name == "previous_mesh_displacement" ||
        role_name == "mesh_displacement_previous" ||
        role_name == "previous_displacement") {
        return MeshMotionFieldRole::PreviousDisplacement;
    }
    if (role_name == "previous_mesh_velocity" ||
        role_name == "mesh_velocity_previous" ||
        role_name == "previous_velocity") {
        return MeshMotionFieldRole::PreviousVelocity;
    }
    if (role_name == "predicted_mesh_velocity" ||
        role_name == "mesh_velocity_predicted" ||
        role_name == "predicted_velocity") {
        return MeshMotionFieldRole::PredictedVelocity;
    }
    FE_THROW(InvalidArgumentException,
             "FESystem::bindMeshMotionField: unknown mesh-motion role '" +
                 std::string(role_name) + "'");
}

FieldId& meshMotionRoleSlot(assembly::MeshMotionFieldAccess& access,
                            MeshMotionFieldRole role) noexcept
{
    switch (role) {
        case MeshMotionFieldRole::Displacement:
            return access.mesh_displacement;
        case MeshMotionFieldRole::Velocity:
            return access.mesh_velocity;
        case MeshMotionFieldRole::Acceleration:
            return access.mesh_acceleration;
        case MeshMotionFieldRole::PreviousCoordinates:
            return access.previous_coordinates;
        case MeshMotionFieldRole::PreviousDisplacement:
            return access.previous_mesh_displacement;
        case MeshMotionFieldRole::PreviousVelocity:
            return access.previous_mesh_velocity;
        case MeshMotionFieldRole::PredictedVelocity:
            return access.predicted_mesh_velocity;
    }
    return access.mesh_displacement;
}

FieldId meshMotionRoleValue(const assembly::MeshMotionFieldAccess& access,
                            MeshMotionFieldRole role) noexcept
{
    switch (role) {
        case MeshMotionFieldRole::Displacement:
            return access.mesh_displacement;
        case MeshMotionFieldRole::Velocity:
            return access.mesh_velocity;
        case MeshMotionFieldRole::Acceleration:
            return access.mesh_acceleration;
        case MeshMotionFieldRole::PreviousCoordinates:
            return access.previous_coordinates;
        case MeshMotionFieldRole::PreviousDisplacement:
            return access.previous_mesh_displacement;
        case MeshMotionFieldRole::PreviousVelocity:
            return access.previous_mesh_velocity;
        case MeshMotionFieldRole::PredictedVelocity:
            return access.predicted_mesh_velocity;
    }
    return INVALID_FIELD_ID;
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
svmp::motion::MotionFieldRole toMeshMotionRole(MeshMotionFieldRole role) noexcept
{
    switch (role) {
        case MeshMotionFieldRole::Displacement:
            return svmp::motion::MotionFieldRole::Displacement;
        case MeshMotionFieldRole::Velocity:
            return svmp::motion::MotionFieldRole::Velocity;
        case MeshMotionFieldRole::Acceleration:
            return svmp::motion::MotionFieldRole::Acceleration;
        case MeshMotionFieldRole::PreviousCoordinates:
            return svmp::motion::MotionFieldRole::PreviousCoordinates;
        case MeshMotionFieldRole::PreviousDisplacement:
            return svmp::motion::MotionFieldRole::PreviousDisplacement;
        case MeshMotionFieldRole::PreviousVelocity:
            return svmp::motion::MotionFieldRole::PreviousVelocity;
        case MeshMotionFieldRole::PredictedVelocity:
            return svmp::motion::MotionFieldRole::Velocity;
    }
    return svmp::motion::MotionFieldRole::Displacement;
}

constexpr std::array<MeshMotionFieldRole, 6> kFEMeshMotionRoles = {{
    MeshMotionFieldRole::Displacement,
    MeshMotionFieldRole::Velocity,
    MeshMotionFieldRole::Acceleration,
    MeshMotionFieldRole::PreviousCoordinates,
    MeshMotionFieldRole::PreviousDisplacement,
    MeshMotionFieldRole::PreviousVelocity,
}};

MeshCoordinateUpdateStage toMeshCoordinateUpdateStage(
    GeometricNonlinearityUpdatePoint update_point) noexcept
{
    switch (update_point) {
        case GeometricNonlinearityUpdatePoint::TrialIterate:
            return MeshCoordinateUpdateStage::TrialNonlinearIterate;
        case GeometricNonlinearityUpdatePoint::AcceptedNonlinearState:
            return MeshCoordinateUpdateStage::AcceptedNonlinearState;
        case GeometricNonlinearityUpdatePoint::AcceptedTimeStep:
            return MeshCoordinateUpdateStage::AcceptedTimeStep;
        case GeometricNonlinearityUpdatePoint::AcceptedRemeshOrRezoneState:
            return MeshCoordinateUpdateStage::AcceptedRemeshRezoneState;
        case GeometricNonlinearityUpdatePoint::RolledBackTrial:
            return MeshCoordinateUpdateStage::TrialNonlinearIterate;
    }
    return MeshCoordinateUpdateStage::TrialNonlinearIterate;
}
#endif

} // namespace

void FESystem::bindMeshMotionField(MeshMotionFieldRole role, FieldId field)
{
    FE_THROW_IF(!field_registry_.has(field), InvalidArgumentException,
                "FESystem::bindMeshMotionField: unknown FieldId");
    const auto& rec = field_registry_.get(field);
    FE_CHECK_NOT_NULL(rec.space.get(), "FESystem::bindMeshMotionField: field space");
    FE_THROW_IF(rec.scope != FieldScope::VolumeCell, InvalidArgumentException,
                "FESystem::bindMeshMotionField: mesh-motion fields must be volume fields");
    FE_THROW_IF(rec.space->field_type() != FieldType::Vector, InvalidArgumentException,
                "FESystem::bindMeshMotionField: mesh-motion field '" + rec.name +
                    "' must be vector-valued");
    const int mesh_dim = mesh_access_ ? mesh_access_->dimension() : rec.components;
    FE_THROW_IF(rec.components != mesh_dim, InvalidArgumentException,
                "FESystem::bindMeshMotionField: mesh-motion field '" + rec.name +
                    "' has component dimension " + std::to_string(rec.components) +
                    " but mesh dimension is " + std::to_string(mesh_dim));

    meshMotionRoleSlot(mesh_motion_fields_, role) = field;
}

void FESystem::bindMeshMotionField(std::string_view role_name, FieldId field)
{
    bindMeshMotionField(parseMeshMotionFieldRole(role_name), field);
}

void FESystem::bindMeshMotionField(std::string_view role_name, std::string_view field_name)
{
    const FieldId field = field_registry_.findByName(field_name);
    FE_THROW_IF(field == INVALID_FIELD_ID, InvalidArgumentException,
                "FESystem::bindMeshMotionField: unknown field '" +
                    std::string(field_name) + "'");
    bindMeshMotionField(parseMeshMotionFieldRole(role_name), field);
}

std::optional<FieldId> FESystem::meshMotionField(MeshMotionFieldRole role) const noexcept
{
    const FieldId field = meshMotionRoleValue(mesh_motion_fields_, role);
    if (field == INVALID_FIELD_ID) {
        return std::nullopt;
    }
    return field;
}

assembly::MeshMotionFieldAccess FESystem::meshMotionFieldAccess() const noexcept
{
    auto access = mesh_motion_fields_;
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    const int mesh_dim = mesh_access_ ? mesh_access_->dimension() : 0;
    auto infer_standard_field = [&](MeshMotionFieldRole role) {
        FieldId& slot = meshMotionRoleSlot(access, role);
        if (slot != INVALID_FIELD_ID) {
            return;
        }
        const FieldId field = field_registry_.findByName(
            svmp::motion::standard_motion_field_name(toMeshMotionRole(role)));
        if (field == INVALID_FIELD_ID || !field_registry_.has(field)) {
            return;
        }
        const auto& rec = field_registry_.get(field);
        if (rec.scope != FieldScope::VolumeCell || rec.space == nullptr ||
            rec.space->field_type() != FieldType::Vector) {
            return;
        }
        const int expected_dim = mesh_dim > 0 ? mesh_dim : rec.components;
        if (rec.components != expected_dim) {
            return;
        }
        slot = field;
    };

    for (const auto role : kFEMeshMotionRoles) {
        infer_standard_field(role);
    }
#endif
    return access;
}

void FESystem::setGeometricNonlinearityPolicy(GeometricNonlinearityPolicy policy)
{
    geometric_nonlinearity_policy_ = policy;
    if (operator_backends_) {
        operator_backends_->invalidateCache();
    }
    if (assembler_) {
        assembler_->invalidateGeometryCaches();
    }
    invalidateAnalysisCache();
}

const GeometricNonlinearityPolicy& FESystem::geometricNonlinearityPolicy() const noexcept
{
    return geometric_nonlinearity_policy_;
}

bool FESystem::geometricNonlinearityEnabled() const noexcept
{
    return geometric_nonlinearity_policy_.enabled;
}

GeometricNonlinearityTransactionEvent FESystem::beginGeometricNonlinearityTrial(
    const SystemStateView& state)
{
    GeometricNonlinearityTransactionEvent event{};
    event.update_point = GeometricNonlinearityUpdatePoint::TrialIterate;
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    event.geometry_state = geometry_transaction_state_;
    if (!shouldUpdateFieldAtPoint(geometric_nonlinearity_policy_,
                                  GeometricNonlinearityStateField::CurrentCoordinates,
                                  event.update_point) ||
        !mesh_ ||
        !meshMotionField(MeshMotionFieldRole::Displacement).has_value()) {
        return event;
    }

    MeshCoordinateUpdateOptions options;
    options.stage = MeshCoordinateUpdateStage::TrialNonlinearIterate;
    updateCurrentCoordinatesFromMeshDisplacement(state, options);
    event.geometry_state = geometry_transaction_state_;
#else
    (void)state;
#endif
    return event;
}

GeometricNonlinearityTransactionEvent FESystem::acceptGeometricNonlinearityState(
    const SystemStateView& state,
    GeometricNonlinearityUpdatePoint update_point)
{
    GeometricNonlinearityTransactionEvent event{};
    event.update_point = update_point;
    event.nonlinear_step_accepted =
        update_point == GeometricNonlinearityUpdatePoint::AcceptedNonlinearState;
    event.time_step_accepted =
        update_point == GeometricNonlinearityUpdatePoint::AcceptedTimeStep;
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    event.geometry_state = geometry_transaction_state_;
    if (!shouldUpdateFieldAtPoint(geometric_nonlinearity_policy_,
                                  GeometricNonlinearityStateField::CurrentCoordinates,
                                  update_point) ||
        !mesh_ ||
        !meshMotionField(MeshMotionFieldRole::Displacement).has_value()) {
        return event;
    }

    MeshCoordinateUpdateOptions options;
    options.stage = toMeshCoordinateUpdateStage(update_point);
    updateCurrentCoordinatesFromMeshDisplacement(state, options);

    if (mesh_coordinate_backup_.has_value()) {
        mesh_coordinate_backup_.reset();
        mesh_motion_field_backup_.clear();
        geometry_transaction_last_revision_ = operatorRevisionSnapshot();
        for (const auto& hook : geometry_transaction_callbacks_) {
            if (hook.callback) {
                hook.callback(geometryTransactionDiagnostics());
            }
        }
    }
    event.geometry_state = geometry_transaction_state_;
#else
    (void)state;
#endif
    return event;
}

GeometricNonlinearityTransactionEvent FESystem::rollbackGeometricNonlinearityTrial()
{
    GeometricNonlinearityTransactionEvent event{};
    event.update_point = GeometricNonlinearityUpdatePoint::RolledBackTrial;
    event.line_search_rejected = true;
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    event.geometry_state = geometry_transaction_state_;
    if (requiresLineSearchRollback(geometric_nonlinearity_policy_, event) &&
        meshCoordinateTransactionActive()) {
        rollbackMeshCoordinateTransaction();
        event.geometry_state = geometry_transaction_state_;
    }
#endif
    return event;
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
FieldId FESystem::addMeshDisplacementUnknown(std::string name,
                                             std::shared_ptr<const spaces::FunctionSpace> space,
                                             int components)
{
    FE_CHECK_NOT_NULL(space.get(), "FESystem::addMeshDisplacementUnknown: space");
    const int mesh_dim = mesh_access_ ? mesh_access_->dimension() : space->value_dimension();
    if (components == 0) {
        components = mesh_dim;
    }
    FE_THROW_IF(space->field_type() != FieldType::Vector, InvalidArgumentException,
                "FESystem::addMeshDisplacementUnknown: mesh displacement must use a vector FE space");
    FE_THROW_IF(components != mesh_dim, InvalidArgumentException,
                "FESystem::addMeshDisplacementUnknown: component dimension " +
                    std::to_string(components) +
                    " does not match mesh dimension " + std::to_string(mesh_dim));
    const auto field = addField(FieldSpec{
        .name = std::move(name),
        .space = std::move(space),
        .components = components,
    });
    bindMeshMotionField(MeshMotionFieldRole::Displacement, field);
    return field;
}

std::size_t FESystem::bindStandardMeshMotionFieldsByName()
{
    std::size_t bound = 0;
    for (const auto role : kFEMeshMotionRoles) {
        const auto mesh_role = toMeshMotionRole(role);
        const FieldId field = field_registry_.findByName(
            svmp::motion::standard_motion_field_name(mesh_role));
        if (field == INVALID_FIELD_ID) {
            continue;
        }
        bindMeshMotionField(role, field);
        ++bound;
    }
    return bound;
}

void FESystem::beginMeshCoordinateTransaction()
{
    const auto& mesh = singleMesh("FESystem::beginMeshCoordinateTransaction");
    FE_THROW_IF(mesh_coordinate_backup_.has_value(), InvalidStateException,
                "FESystem::beginMeshCoordinateTransaction: transaction already active");
    svmp::motion::MotionCoordinateBackup backup;
    svmp::motion::save_coordinates(mesh, backup);
    mesh_coordinate_backup_ = std::move(backup);
    mesh_motion_field_backup_.clear();

    const auto& local_mesh = mesh.local_mesh();
    for (const auto role : svmp::motion::standard_motion_field_roles()) {
        const std::string field_name(svmp::motion::standard_motion_field_name(role));
        const auto handle =
            svmp::MeshFields::get_field_handle(local_mesh, svmp::EntityKind::Vertex, field_name);
        if (handle.id == 0) {
            continue;
        }
        if (svmp::MeshFields::field_type(local_mesh, handle) != svmp::FieldScalarType::Float64) {
            continue;
        }
        FESystem::MeshMotionFieldBackup field_backup;
        field_backup.handle = handle;
        field_backup.name = field_name;
        field_backup.components = svmp::MeshFields::field_components(local_mesh, handle);
        field_backup.entity_count = svmp::MeshFields::field_entity_count(local_mesh, handle);
        const auto count = field_backup.components * field_backup.entity_count;
        const auto* values = svmp::MeshFields::field_data_as<svmp::real_t>(local_mesh, handle);
        FE_CHECK_NOT_NULL(values, "FESystem::beginMeshCoordinateTransaction: mesh-motion field data");
        field_backup.values.assign(values, values + count);
        mesh_motion_field_backup_.push_back(std::move(field_backup));
    }

    geometry_transaction_start_revision_ = operatorRevisionSnapshot();
    geometry_transaction_last_revision_ = geometry_transaction_start_revision_;
    geometry_transaction_state_ = GeometryTransactionState::Trial;
    geometry_transaction_last_event_ = "begin";
    for (const auto& hook : geometry_transaction_callbacks_) {
        if (hook.callback) {
            hook.callback(geometryTransactionDiagnostics());
        }
    }
}

MeshCoordinateUpdateResult FESystem::updateCurrentCoordinatesFromMeshDisplacement(
    const SystemStateView& state,
    const MeshCoordinateUpdateOptions& options)
{
    requireSetup();
    const auto& mesh_ref = singleMesh("FESystem::updateCurrentCoordinatesFromMeshDisplacement");

    const auto displacement_field = meshMotionField(MeshMotionFieldRole::Displacement);
    FE_THROW_IF(!displacement_field.has_value(), InvalidStateException,
                "FESystem::updateCurrentCoordinatesFromMeshDisplacement: no mesh-displacement "
                "FE field is bound");

    if (options.stage == MeshCoordinateUpdateStage::TrialNonlinearIterate &&
        !mesh_coordinate_backup_.has_value()) {
        beginMeshCoordinateTransaction();
    }

    auto& mesh = const_cast<svmp::Mesh&>(mesh_ref);
    auto& local_mesh = mesh.local_mesh();
    const int dim = mesh.dim();
    FE_THROW_IF(dim <= 0, InvalidStateException,
                "FESystem::updateCurrentCoordinatesFromMeshDisplacement: mesh dimension must be positive");

    const auto n_vertices = static_cast<GlobalIndex>(mesh.n_vertices());
    std::vector<double> displacement(
        static_cast<std::size_t>(n_vertices) * static_cast<std::size_t>(dim),
        0.0);
    const bool evaluated = evaluateFieldAtVertices(*displacement_field,
                                                   state,
                                                   n_vertices,
                                                   displacement);
    FE_THROW_IF(!evaluated, InvalidStateException,
                "FESystem::updateCurrentCoordinatesFromMeshDisplacement: mesh-displacement "
                "field must have vertex DOFs covering every mesh vertex");

    const auto& X_ref = local_mesh.X_ref();
    FE_THROW_IF(X_ref.size() != displacement.size(), InvalidStateException,
                "FESystem::updateCurrentCoordinatesFromMeshDisplacement: reference-coordinate "
                "size does not match displacement field");

    const bool use_current =
        options.mode == MeshCoordinateUpdateMode::IncrementalFromCurrent &&
        local_mesh.has_current_coords();
    const auto& X_base = use_current ? local_mesh.X_cur() : X_ref;
    FE_THROW_IF(X_base.size() != X_ref.size(), InvalidStateException,
                "FESystem::updateCurrentCoordinatesFromMeshDisplacement: base-coordinate "
                "size mismatch");

    std::vector<svmp::real_t> X_new(X_ref.size(), svmp::real_t{0});
    for (std::size_t k = 0; k < X_new.size(); ++k) {
        X_new[k] = static_cast<svmp::real_t>(X_base[k] + displacement[k]);
    }

    mesh.set_current_coords(X_new);
    local_mesh.use_current_configuration();
    if (options.exchange_ghost_coordinates) {
        mesh.update_exchange_ghost_coordinates(svmp::Configuration::Current);
    }
    if (options.notify_geometry_advanced) {
        notifyMeshGeometryAdvanced();
    }

    switch (options.stage) {
        case MeshCoordinateUpdateStage::TrialNonlinearIterate:
            geometry_transaction_state_ = GeometryTransactionState::Trial;
            geometry_transaction_last_event_ = "trial-update";
            break;
        case MeshCoordinateUpdateStage::AcceptedNonlinearState:
            geometry_transaction_state_ = GeometryTransactionState::Accepted;
            geometry_transaction_last_event_ = "accepted-nonlinear-state";
            break;
        case MeshCoordinateUpdateStage::AcceptedTimeStep:
            geometry_transaction_state_ = GeometryTransactionState::Committed;
            geometry_transaction_last_event_ = "accepted-time-step";
            break;
        case MeshCoordinateUpdateStage::AcceptedRemeshRezoneState:
            geometry_transaction_state_ = GeometryTransactionState::Committed;
            geometry_transaction_last_event_ = "accepted-remesh-rezone-state";
            break;
    }
    geometry_transaction_last_revision_ = operatorRevisionSnapshot();
    for (const auto& hook : geometry_transaction_callbacks_) {
        if (hook.callback) {
            hook.callback(geometryTransactionDiagnostics());
        }
    }

    return MeshCoordinateUpdateResult{
        .vertices_updated = static_cast<std::size_t>(n_vertices),
        .components_updated = static_cast<std::size_t>(n_vertices) * static_cast<std::size_t>(dim),
        .stage = options.stage,
        .geometry_revision = local_mesh.geometry_revision(),
    };
}

void FESystem::commitMeshCoordinateTransaction()
{
    geometry_transaction_state_ = GeometryTransactionState::Accepted;
    geometry_transaction_last_event_ = "commit";
    geometry_transaction_last_revision_ = operatorRevisionSnapshot();
    mesh_coordinate_backup_.reset();
    mesh_motion_field_backup_.clear();
    for (const auto& hook : geometry_transaction_callbacks_) {
        if (hook.callback) {
            hook.callback(geometryTransactionDiagnostics());
        }
    }
}

void FESystem::rollbackMeshCoordinateTransaction()
{
    const auto& mesh_ref = singleMesh("FESystem::rollbackMeshCoordinateTransaction");
    if (!mesh_coordinate_backup_.has_value()) {
        return;
    }
    auto& mesh = const_cast<svmp::Mesh&>(mesh_ref);
    svmp::motion::restore_coordinates(mesh, *mesh_coordinate_backup_);
    auto& local_mesh = mesh.local_mesh();
    for (const auto& field_backup : mesh_motion_field_backup_) {
        auto handle = field_backup.handle;
        if (handle.id == 0 ||
            !svmp::MeshFields::has_field(local_mesh, handle.kind, handle.name)) {
            continue;
        }
        const auto current_components = svmp::MeshFields::field_components(local_mesh, handle);
        const auto current_entities = svmp::MeshFields::field_entity_count(local_mesh, handle);
        if (current_components != field_backup.components ||
            current_entities != field_backup.entity_count ||
            svmp::MeshFields::field_type(local_mesh, handle) != svmp::FieldScalarType::Float64) {
            continue;
        }
        auto* values = svmp::MeshFields::field_data_as<svmp::real_t>(local_mesh, handle);
        FE_CHECK_NOT_NULL(values, "FESystem::rollbackMeshCoordinateTransaction: mesh-motion field data");
        std::copy(field_backup.values.begin(), field_backup.values.end(), values);
    }
    if (mesh.local_mesh().has_current_coords()) {
        mesh.update_exchange_ghost_coordinates(svmp::Configuration::Current);
    }
    if (material_state_provider_) {
        material_state_provider_->rollbackTimeStep();
    }
    if (global_kernel_state_provider_) {
        global_kernel_state_provider_->rollbackTimeStep();
    }
    if (auxiliary_state_manager_) {
        auxiliary_state_manager_->rollbackAll();
    }
    if (auxiliary_input_registry_) {
        auxiliary_input_registry_->invalidateAll();
    }
    if (operator_backends_) {
        operator_backends_->invalidateCache();
    }
    if (assembler_) {
        assembler_->invalidateGeometryCaches();
    }
    invalidateAnalysisCache();
    geometry_transaction_state_ = GeometryTransactionState::RolledBack;
    geometry_transaction_last_event_ = "rollback";
    geometry_transaction_last_revision_ = operatorRevisionSnapshot();
    mesh_coordinate_backup_.reset();
    mesh_motion_field_backup_.clear();
    for (const auto& hook : geometry_transaction_callbacks_) {
        if (hook.callback) {
            hook.callback(geometryTransactionDiagnostics());
        }
    }
}

void FESystem::rebaseMeshReferenceToCurrent(const svmp::ReferenceRebaseOptions& options)
{
    const auto& mesh_ref = singleMesh("FESystem::rebaseMeshReferenceToCurrent");
    FE_THROW_IF(mesh_coordinate_backup_.has_value(), InvalidStateException,
                "FESystem::rebaseMeshReferenceToCurrent: cannot rebase during an active "
                "mesh coordinate transaction");

    auto& mesh = const_cast<svmp::Mesh&>(mesh_ref);
    mesh.rebase_reference_to_current(options);
    if (mesh.has_current_coords()) {
        mesh.update_exchange_ghost_coordinates(svmp::Configuration::Current);
    }

    notifyMeshReferenceRebased();
    geometry_transaction_state_ = GeometryTransactionState::Committed;
    geometry_transaction_last_event_ = "reference-rebase-current";
    geometry_transaction_last_revision_ = operatorRevisionSnapshot();
    for (const auto& hook : geometry_transaction_callbacks_) {
        if (hook.callback) {
            hook.callback(geometryTransactionDiagnostics());
        }
    }
}

void FESystem::rebaseMeshReferenceCoordinates(
    std::vector<svmp::real_t> Xref,
    const svmp::ReferenceRebaseOptions& options)
{
    const auto& mesh_ref = singleMesh("FESystem::rebaseMeshReferenceCoordinates");
    FE_THROW_IF(mesh_coordinate_backup_.has_value(), InvalidStateException,
                "FESystem::rebaseMeshReferenceCoordinates: cannot rebase during an active "
                "mesh coordinate transaction");

    auto& mesh = const_cast<svmp::Mesh&>(mesh_ref);
    mesh.rebase_reference_coordinates(std::move(Xref), options);
    if (mesh.has_current_coords()) {
        mesh.update_exchange_ghost_coordinates(svmp::Configuration::Current);
    }

    notifyMeshReferenceRebased();
    geometry_transaction_state_ = GeometryTransactionState::Committed;
    geometry_transaction_last_event_ = "reference-rebase-explicit";
    geometry_transaction_last_revision_ = operatorRevisionSnapshot();
    for (const auto& hook : geometry_transaction_callbacks_) {
        if (hook.callback) {
            hook.callback(geometryTransactionDiagnostics());
        }
    }
}

bool FESystem::rebaseGeometricNonlinearityReference(
    const svmp::ReferenceRebaseOptions& options)
{
    if (!geometric_nonlinearity_policy_.enabled ||
        geometric_nonlinearity_policy_.reference_policy !=
            geometry::FiniteDeformationReferencePolicy::UpdatedLagrangian ||
        !mesh_) {
        return false;
    }

    rebaseMeshReferenceToCurrent(options);
    if (geometric_nonlinearity_policy_.reset_displacement_after_reference_rebase) {
        (void)resetBoundMeshMotionField(MeshMotionFieldRole::Displacement, Real(0));
    }
    return true;
}

std::size_t FESystem::resetBoundMeshMotionField(MeshMotionFieldRole role, Real value)
{
    requireSetup();
    const auto& mesh_ref = singleMesh("FESystem::resetBoundMeshMotionField");
    if (!meshMotionField(role).has_value()) {
        return 0;
    }

    auto& mesh = const_cast<svmp::Mesh&>(mesh_ref);
    auto& local_mesh = mesh.local_mesh();
    const auto mesh_role = toMeshMotionRole(role);
    const auto mesh_name = svmp::motion::standard_motion_field_name(mesh_role);
    const auto mesh_field =
        svmp::MeshFields::get_field_handle(local_mesh, svmp::EntityKind::Vertex, mesh_name);
    if (mesh_field.id == 0) {
        return 0;
    }
    svmp::motion::validate_motion_field(local_mesh, mesh_field, mesh_role, mesh.dim());
    auto* values = svmp::MeshFields::field_data_as<svmp::real_t>(local_mesh, mesh_field);
    FE_CHECK_NOT_NULL(values, "FESystem::resetBoundMeshMotionField: mesh field data");
    const auto count = svmp::MeshFields::field_components(local_mesh, mesh_field) *
                       svmp::MeshFields::field_entity_count(local_mesh, mesh_field);
    std::fill(values, values + count, static_cast<svmp::real_t>(value));
    notifyMeshGeometryAdvanced();
    return count;
}

bool FESystem::meshCoordinateTransactionActive() const noexcept
{
    return mesh_coordinate_backup_.has_value();
}

GeometryTransactionState FESystem::meshCoordinateTransactionState() const noexcept
{
    return geometry_transaction_state_;
}

GeometryConfigurationUse FESystem::geometryConfigurationUse() const noexcept
{
    if (!mesh_) {
        return GeometryConfigurationUse::Reference;
    }
    if (geometry_transaction_state_ == GeometryTransactionState::Trial) {
        return GeometryConfigurationUse::TrialCurrent;
    }
    if (geometry_transaction_state_ == GeometryTransactionState::Accepted) {
        return GeometryConfigurationUse::AcceptedCurrent;
    }
    if (geometry_transaction_state_ == GeometryTransactionState::RolledBack) {
        return mesh_->local_mesh().active_configuration() == svmp::Configuration::Current
                   ? GeometryConfigurationUse::RolledBackCurrent
                   : GeometryConfigurationUse::Reference;
    }
    const auto active = mesh_->local_mesh().active_configuration();
    if (coord_cfg_ == svmp::Configuration::Current ||
        coord_cfg_ == svmp::Configuration::Deformed ||
        active == svmp::Configuration::Current ||
        active == svmp::Configuration::Deformed) {
        return GeometryConfigurationUse::CommittedCurrent;
    }
    return GeometryConfigurationUse::Reference;
}

GeometryTransactionDiagnostics FESystem::geometryTransactionDiagnostics() const
{
    GeometryTransactionDiagnostics diag;
    diag.state = geometry_transaction_state_;
    diag.geometry_use = geometryConfigurationUse();
    diag.started_from = geometry_transaction_start_revision_;
    diag.current = operatorRevisionSnapshot();
    diag.last_event = geometry_transaction_last_event_;
    return diag;
}

void FESystem::addGeometryTransactionCallback(GeometryTransactionCallback hook)
{
    FE_THROW_IF(hook.name.empty(), InvalidArgumentException,
                "FESystem::addGeometryTransactionCallback: hook name must not be empty");
    FE_THROW_IF(!hook.callback, InvalidArgumentException,
                "FESystem::addGeometryTransactionCallback: callback must be callable");
    geometry_transaction_callbacks_.push_back(std::move(hook));
}

namespace {

struct MeshMotionSyncEntry {
    MeshMotionFieldRole fe_role{MeshMotionFieldRole::Displacement};
    FieldId fe_field{INVALID_FIELD_ID};
    svmp::motion::MotionFieldRole mesh_role{svmp::motion::MotionFieldRole::Displacement};
};

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
template <typename Callback>
FESystem::MeshVertexFieldProjectionResult enumerateMeshVertexDofBindings(
    const svmp::MeshBase& mesh,
    const dofs::EntityDofMap& entity_map,
    std::size_t components,
    std::size_t coefficient_count,
    std::string_view context,
    Callback&& callback)
{
    FESystem::MeshVertexFieldProjectionResult result{};
    if (components == 0u || coefficient_count == 0u) {
        result.unassigned_dofs = coefficient_count;
        return result;
    }

    const auto n_vertices = static_cast<GlobalIndex>(mesh.n_vertices());
    FE_THROW_IF(entity_map.numVertices() < n_vertices, InvalidStateException,
                std::string(context) + ": FE field does not cover every mesh vertex");

    std::vector<std::uint8_t> dof_expected(coefficient_count, 0u);
    std::vector<std::uint8_t> dof_bound(coefficient_count, 0u);
    const bool has_component_stride =
        components > 1u && (coefficient_count % components == 0u);
    const std::size_t component_stride =
        has_component_stride ? (coefficient_count / components) : 0u;
    auto bind = [&](svmp::index_t geometry_vertex,
                    std::size_t component,
                    GlobalIndex dof) {
        FE_THROW_IF(component >= components, InvalidArgumentException,
                    std::string(context) + ": component index out of range");
        FE_THROW_IF(geometry_vertex < 0 ||
                        static_cast<std::size_t>(geometry_vertex) >=
                            static_cast<std::size_t>(n_vertices),
                    InvalidStateException,
                    std::string(context) + ": mesh geometry vertex index out of range");
        FE_THROW_IF(dof < 0 ||
                        static_cast<std::size_t>(dof) >= coefficient_count,
                    InvalidStateException,
                    std::string(context) + ": FE coefficient index out of range");
        const auto sdof = static_cast<std::size_t>(dof);
        if (dof_bound[sdof] != 0u) {
            return;
        }
        callback(static_cast<std::size_t>(geometry_vertex), component, dof);
        dof_bound[sdof] = 1u;
        ++result.values_written;
    };
    auto entity_dof_for_component =
        [&](std::span<const GlobalIndex> entity_dofs,
            std::size_t entity_node_count,
            std::size_t component,
            std::size_t node_index,
            std::string_view entity_name) -> GlobalIndex {
        FE_THROW_IF(node_index >= entity_node_count, InvalidArgumentException,
                    std::string(context) + ": " + std::string(entity_name) +
                        " node index out of range");
        if (entity_dofs.size() == entity_node_count * components) {
            return entity_dofs[component * entity_node_count + node_index];
        }
        if (entity_dofs.size() == entity_node_count && has_component_stride) {
            const auto base = entity_dofs[node_index];
            FE_THROW_IF(base < 0, InvalidStateException,
                        std::string(context) + ": negative " +
                            std::string(entity_name) + " scalar DOF");
            const auto shifted =
                static_cast<std::size_t>(base) + component * component_stride;
            FE_THROW_IF(shifted >= coefficient_count, InvalidStateException,
                        std::string(context) + ": component-shifted " +
                            std::string(entity_name) +
                            " DOF is outside the coefficient span");
            return static_cast<GlobalIndex>(shifted);
        }
        FE_THROW_IF(true, InvalidStateException,
                    std::string(context) + ": " + std::string(entity_name) +
                        " DOF count does not match field components");
        return INVALID_GLOBAL_INDEX;
    };
    auto mark_expected = [&](std::span<const GlobalIndex> entity_dofs,
                             std::size_t entity_node_count,
                             std::string_view entity_name) {
        FE_THROW_IF(entity_node_count == 0u && !entity_dofs.empty(),
                    InvalidStateException,
                    std::string(context) + ": " + std::string(entity_name) +
                        " DOFs require matching mesh geometry nodes");
        for (std::size_t c = 0; c < components; ++c) {
            for (std::size_t j = 0; j < entity_node_count; ++j) {
                const auto dof =
                    entity_dof_for_component(
                        entity_dofs, entity_node_count, c, j, entity_name);
                FE_THROW_IF(dof < 0 ||
                                static_cast<std::size_t>(dof) >= coefficient_count,
                            InvalidStateException,
                            std::string(context) + ": expected " +
                                std::string(entity_name) +
                                " DOF is outside the coefficient span");
                dof_expected[static_cast<std::size_t>(dof)] = 1u;
            }
        }
    };

    for (GlobalIndex vertex = 0; vertex < n_vertices; ++vertex) {
        const auto vertex_dofs = entity_map.getVertexDofs(vertex);
        if (vertex_dofs.empty()) {
            continue;
        }
        mark_expected(vertex_dofs, 1u, "vertex");
        for (std::size_t c = 0; c < components; ++c) {
            bind(static_cast<svmp::index_t>(vertex),
                 c,
                 entity_dof_for_component(
                     vertex_dofs, 1u, c, 0u, "vertex"));
        }
    }

    std::map<std::pair<svmp::index_t, svmp::index_t>, svmp::index_t>
        edge_by_vertices;
    auto make_edge_key = [](svmp::index_t a, svmp::index_t b) {
        if (b < a) {
            std::swap(a, b);
        }
        return std::pair<svmp::index_t, svmp::index_t>{a, b};
    };
    const bool use_2d_faces_as_edges =
        mesh.dim() == 2 &&
        mesh.codim1_storage_mode() == svmp::MeshCodim1StorageMode::Full &&
        mesh.n_faces() > 0u &&
        !mesh.face2vertex_offsets().empty() &&
        !mesh.face2vertex().empty();
    auto edge_entity_endpoints =
        [&](svmp::index_t edge_entity) -> std::optional<std::array<svmp::index_t, 2>> {
        if (use_2d_faces_as_edges) {
            if (edge_entity < 0 ||
                static_cast<std::size_t>(edge_entity) >= mesh.n_faces()) {
                return std::nullopt;
            }
            auto [face_nodes, n_face_nodes] = mesh.face_vertices_span(edge_entity);
            if (face_nodes == nullptr || n_face_nodes < 2u) {
                return std::nullopt;
            }
            return std::array<svmp::index_t, 2>{
                face_nodes[0],
                face_nodes[n_face_nodes - 1u]};
        }
        if (edge_entity < 0 ||
            static_cast<std::size_t>(edge_entity) >= mesh.n_edges()) {
            return std::nullopt;
        }
        const auto vertices = mesh.edge_vertices(edge_entity);
        return std::array<svmp::index_t, 2>{vertices[0], vertices[1]};
    };
    const auto n_edge_entities =
        use_2d_faces_as_edges ? mesh.n_faces() : mesh.n_edges();
    for (svmp::index_t edge = 0;
         edge < static_cast<svmp::index_t>(n_edge_entities);
         ++edge) {
        const auto endpoints = edge_entity_endpoints(edge);
        if (!endpoints.has_value()) {
            continue;
        }
        edge_by_vertices.emplace(
            make_edge_key((*endpoints)[0], (*endpoints)[1]),
            edge);
    }

    auto bind_edge_interior = [&](svmp::index_t cell,
                                  int local_edge,
                                  svmp::index_t endpoint_a,
                                  svmp::index_t endpoint_b) {
        if (edge_by_vertices.empty()) {
            return;
        }
        const auto edge_it =
            edge_by_vertices.find(make_edge_key(endpoint_a, endpoint_b));
        if (edge_it == edge_by_vertices.end()) {
            return;
        }

        const auto edge = edge_it->second;
        const auto edge_geometry = mesh.cell_edge_geometry_dofs(cell, local_edge);
        const auto edge_dofs =
            entity_map.getEdgeDofs(static_cast<GlobalIndex>(edge));
        if (edge_dofs.empty()) {
            return;
        }
        if (edge_geometry.size() <= 2u) {
            mark_expected(edge_dofs, 0u, "edge");
            return;
        }

        const auto canonical = edge_entity_endpoints(edge);
        if (!canonical.has_value()) {
            mark_expected(edge_dofs, 0u, "edge");
            return;
        }
        const bool forward =
            edge_geometry.front() == (*canonical)[0] &&
            edge_geometry.back() == (*canonical)[1];
        const bool reverse =
            edge_geometry.front() == (*canonical)[1] &&
            edge_geometry.back() == (*canonical)[0];
        FE_THROW_IF(!forward && !reverse, InvalidStateException,
                    std::string(context) +
                        ": high-order edge geometry endpoints do not match mesh edge topology");

        const auto interior_count = edge_geometry.size() - 2u;
        mark_expected(edge_dofs, interior_count, "edge");

        for (std::size_t c = 0; c < components; ++c) {
            for (std::size_t j = 0; j < interior_count; ++j) {
                const auto geometry_index =
                    forward ? (j + 1u) : (edge_geometry.size() - 2u - j);
                const auto dof =
                    entity_dof_for_component(
                        edge_dofs, interior_count, c, j, "edge");
                bind(edge_geometry[geometry_index], c, dof);
            }
        }
    };

    struct GeometryOrderInfo {
        int order{1};
        svmp::CellTopology::HighOrderKind kind{
            svmp::CellTopology::HighOrderKind::Lagrange};
    };
    auto infer_geometry_order =
        [](svmp::CellFamily family,
           int declared_order,
           int num_corners,
           std::size_t node_count) {
            GeometryOrderInfo info{};
            info.order = std::max(1, declared_order);
            const int corners = std::max(0, num_corners);
            if (node_count <= static_cast<std::size_t>(corners)) {
                return info;
            }
            if (family == svmp::CellFamily::Line) {
                if (node_count >= 2u) {
                    info.order = static_cast<int>(node_count) - 1;
                }
                return info;
            }
            const int p_lag =
                svmp::CellTopology::infer_lagrange_order(family, node_count);
            const int p_ser =
                svmp::CellTopology::infer_serendipity_order(family, node_count);
            if (p_lag > 0 &&
                (declared_order <= 1 || p_lag == declared_order ||
                 p_ser != declared_order)) {
                info.order = p_lag;
                info.kind = svmp::CellTopology::HighOrderKind::Lagrange;
            } else if (p_ser > 0) {
                info.order = p_ser;
                info.kind = svmp::CellTopology::HighOrderKind::Serendipity;
            } else if (p_lag > 0) {
                info.order = p_lag;
                info.kind = svmp::CellTopology::HighOrderKind::Lagrange;
            }
            return info;
        };
    auto topological_dimension = [](svmp::CellFamily family) {
        switch (family) {
        case svmp::CellFamily::Line:
            return 1;
        case svmp::CellFamily::Triangle:
        case svmp::CellFamily::Quad:
        case svmp::CellFamily::Polygon:
            return 2;
        case svmp::CellFamily::Tetra:
        case svmp::CellFamily::Hex:
        case svmp::CellFamily::Wedge:
        case svmp::CellFamily::Pyramid:
        case svmp::CellFamily::Polyhedron:
            return 3;
        default:
            return 0;
        }
    };
    auto face_interior_geometry = [&](svmp::index_t face) {
        auto [face_nodes, n_face_nodes] = mesh.face_vertices_span(face);
        std::vector<svmp::index_t> interior;
        if (face_nodes == nullptr || n_face_nodes == 0u) {
            return interior;
        }

        svmp::CellShape shape{};
        shape.num_corners = static_cast<int>(n_face_nodes);
        const auto& face_shapes = mesh.face_shapes();
        if (static_cast<std::size_t>(face) < face_shapes.size()) {
            shape = face_shapes[static_cast<std::size_t>(face)];
        }

        if (topological_dimension(shape.family) != 2) {
            return interior;
        }

        const auto info =
            infer_geometry_order(shape.family,
                                 shape.order,
                                 shape.num_corners,
                                 n_face_nodes);
        std::vector<bool> on_boundary(n_face_nodes, false);
        const auto boundary =
            svmp::CellTopology::get_oriented_boundary_faces_view(shape.family);
        if (boundary.face_count <= 0) {
            const int corners =
                std::min<int>(std::max(0, shape.num_corners),
                              static_cast<int>(n_face_nodes));
            for (int i = 0; i < corners; ++i) {
                on_boundary[static_cast<std::size_t>(i)] = true;
            }
        } else {
            for (int local_face = 0; local_face < boundary.face_count;
                 ++local_face) {
                for (const auto local_node :
                     svmp::CellTopology::high_order_face_local_nodes(
                         shape.family, info.order, local_face, info.kind)) {
                    const auto idx = static_cast<std::size_t>(local_node);
                    if (idx < n_face_nodes) {
                        on_boundary[idx] = true;
                    }
                }
            }
        }

        for (std::size_t i = 0; i < n_face_nodes; ++i) {
            if (!on_boundary[i]) {
                interior.push_back(face_nodes[i]);
            }
        }
        return interior;
    };

    for (svmp::index_t cell = 0;
         cell < static_cast<svmp::index_t>(mesh.n_cells());
         ++cell) {
        auto [cell_vertices, n_cell_vertices] = mesh.cell_vertices_span(cell);
        if (cell_vertices == nullptr || n_cell_vertices == 0u) {
            continue;
        }

        const auto& shape = mesh.cell_shape(cell);
        if (shape.family == svmp::CellFamily::Polygon) {
            const int corner_count =
                shape.num_corners > 0
                    ? std::min<int>(shape.num_corners,
                                    static_cast<int>(n_cell_vertices))
                    : static_cast<int>(n_cell_vertices);
            if (corner_count >= 2) {
                const auto edges =
                    svmp::CellTopology::get_polygon_edges_view(corner_count);
                for (int local_edge = 0; local_edge < edges.edge_count;
                     ++local_edge) {
                    const auto local_a = edges.pairs_flat[2 * local_edge];
                    const auto local_b = edges.pairs_flat[2 * local_edge + 1];
                    if (local_a < 0 || local_b < 0 ||
                        static_cast<std::size_t>(local_a) >= n_cell_vertices ||
                        static_cast<std::size_t>(local_b) >= n_cell_vertices) {
                        continue;
                    }
                    bind_edge_interior(
                        cell,
                        local_edge,
                        cell_vertices[static_cast<std::size_t>(local_a)],
                        cell_vertices[static_cast<std::size_t>(local_b)]);
                }
            }
        } else {
            const auto edges =
                svmp::CellTopology::get_edges_view(shape.family);
            for (int local_edge = 0; local_edge < edges.edge_count;
                 ++local_edge) {
                const auto local_a = edges.pairs_flat[2 * local_edge];
                const auto local_b = edges.pairs_flat[2 * local_edge + 1];
                if (local_a < 0 || local_b < 0 ||
                    static_cast<std::size_t>(local_a) >= n_cell_vertices ||
                    static_cast<std::size_t>(local_b) >= n_cell_vertices) {
                    continue;
                }
                bind_edge_interior(
                    cell,
                    local_edge,
                    cell_vertices[static_cast<std::size_t>(local_a)],
                    cell_vertices[static_cast<std::size_t>(local_b)]);
            }
        }

        const auto cell_geometry = mesh.cell_interior_geometry_dofs(cell);
        const auto cell_dofs =
            entity_map.getCellInteriorDofs(static_cast<GlobalIndex>(cell));
        if (!cell_dofs.empty()) {
            mark_expected(cell_dofs, cell_geometry.size(), "cell-interior");
            for (std::size_t c = 0; c < components; ++c) {
                for (std::size_t j = 0; j < cell_geometry.size(); ++j) {
                    bind(cell_geometry[j], c,
                         entity_dof_for_component(
                             cell_dofs,
                             cell_geometry.size(),
                             c,
                             j,
                             "cell-interior"));
                }
            }
        }
    }

    for (svmp::index_t face = 0;
         face < static_cast<svmp::index_t>(mesh.n_faces());
         ++face) {
        const auto face_dofs =
            entity_map.getFaceDofs(static_cast<GlobalIndex>(face));
        if (face_dofs.empty()) {
            continue;
        }

        const auto face_geometry = face_interior_geometry(face);
        mark_expected(face_dofs, face_geometry.size(), "face");
        for (std::size_t c = 0; c < components; ++c) {
            for (std::size_t j = 0; j < face_geometry.size(); ++j) {
                bind(face_geometry[j], c,
                     entity_dof_for_component(
                         face_dofs,
                         face_geometry.size(),
                         c,
                         j,
                         "face"));
            }
        }
    }

    for (std::size_t dof = 0; dof < coefficient_count; ++dof) {
        if (dof_expected[dof] != 0u && dof_bound[dof] == 0u) {
            ++result.unassigned_dofs;
        }
    }
    return result;
}

std::vector<Real> solveDenseInterpolationSystem(
    std::vector<Real> matrix,
    std::vector<Real> rhs,
    std::string_view context)
{
    const std::size_t n = rhs.size();
    FE_THROW_IF(n == 0u || matrix.size() != n * n, InvalidArgumentException,
                std::string(context) +
                    ": invalid local interpolation system size");

    Real max_entry = Real{0};
    for (const auto value : matrix) {
        max_entry = std::max(max_entry, std::abs(value));
    }
    const Real pivot_tol =
        std::numeric_limits<Real>::epsilon() *
        Real{1024} * std::max<Real>(Real{1}, max_entry);

    for (std::size_t column = 0; column < n; ++column) {
        std::size_t pivot = column;
        Real pivot_abs = std::abs(matrix[column * n + column]);
        for (std::size_t row = column + 1u; row < n; ++row) {
            const Real candidate = std::abs(matrix[row * n + column]);
            if (candidate > pivot_abs) {
                pivot_abs = candidate;
                pivot = row;
            }
        }
        FE_THROW_IF(pivot_abs <= pivot_tol, InvalidStateException,
                    std::string(context) +
                        ": non-nodal mesh-vertex projection interpolation matrix is singular");
        if (pivot != column) {
            for (std::size_t j = column; j < n; ++j) {
                std::swap(matrix[column * n + j], matrix[pivot * n + j]);
            }
            std::swap(rhs[column], rhs[pivot]);
        }

        const Real diag = matrix[column * n + column];
        for (std::size_t row = column + 1u; row < n; ++row) {
            const Real factor = matrix[row * n + column] / diag;
            if (factor == Real{0}) {
                continue;
            }
            matrix[row * n + column] = Real{0};
            for (std::size_t j = column + 1u; j < n; ++j) {
                matrix[row * n + j] -= factor * matrix[column * n + j];
            }
            rhs[row] -= factor * rhs[column];
        }
    }

    std::vector<Real> solution(n, Real{0});
    for (std::size_t rev = 0; rev < n; ++rev) {
        const std::size_t row = n - 1u - rev;
        Real value = rhs[row];
        for (std::size_t j = row + 1u; j < n; ++j) {
            value -= matrix[row * n + j] * solution[j];
        }
        solution[row] = value / matrix[row * n + row];
    }
    return solution;
}

FESystem::MeshVertexFieldProjectionResult projectMeshVertexValuesByLocalInterpolation(
    const svmp::MeshBase& mesh,
    const spaces::FunctionSpace& space,
    const dofs::DofHandler& field_dofs,
    std::span<const Real> mesh_values,
    std::size_t mesh_components,
    std::size_t components,
    std::span<Real> coefficients,
    std::span<std::uint8_t> assigned,
    std::string_view context)
{
    FE_THROW_IF(space.is_variable_order(), InvalidStateException,
                std::string(context) +
                    ": non-nodal mesh-vertex projection does not support variable-order spaces");
    FE_THROW_IF(space.element().basis().is_vector_valued(), InvalidStateException,
                std::string(context) +
                    ": non-nodal mesh-vertex projection currently supports scalar H1/product bases only");

    const std::size_t scalar_dofs = space.element().basis().size();
    FE_THROW_IF(scalar_dofs == 0u, InvalidStateException,
                std::string(context) +
                    ": non-nodal mesh-vertex projection requires basis functions");
    FE_THROW_IF(space.dofs_per_element() != scalar_dofs * components,
                InvalidStateException,
                std::string(context) +
                    ": non-nodal mesh-vertex projection requires component-major scalar basis layout");

    const auto reference_nodes =
        basis::ReferenceNodeLayout::get_lagrange_node_coords(
            space.element_type(),
            space.polynomial_order());
    FE_THROW_IF(reference_nodes.size() != scalar_dofs, InvalidStateException,
                std::string(context) +
                    ": non-nodal mesh-vertex projection requires matching Lagrange interpolation nodes");

    std::vector<Real> interpolation_matrix(scalar_dofs * scalar_dofs, Real{0});
    std::vector<Real> values;
    for (std::size_t row = 0; row < scalar_dofs; ++row) {
        space.element().basis().evaluate_values(reference_nodes[row], values);
        FE_THROW_IF(values.size() != scalar_dofs, InvalidStateException,
                    std::string(context) +
                        ": non-nodal basis value count changed during projection");
        for (std::size_t col = 0; col < scalar_dofs; ++col) {
            interpolation_matrix[row * scalar_dofs + col] = values[col];
        }
    }

    std::vector<std::uint8_t> coefficient_written(coefficients.size(), 0u);
    if (!assigned.empty()) {
        std::fill(assigned.begin(),
                  assigned.begin() +
                      static_cast<std::ptrdiff_t>(coefficients.size()),
                  std::uint8_t{0});
    }

    FESystem::MeshVertexFieldProjectionResult result{};
    const auto write_coefficient =
        [&](std::size_t dof, Real value) {
            FE_THROW_IF(dof >= coefficients.size(), InvalidStateException,
                        std::string(context) +
                            ": non-nodal projection DOF is outside the coefficient span");
            if (coefficient_written[dof] != 0u) {
                const Real scale =
                    std::max<Real>(Real{1},
                                   std::max(std::abs(coefficients[dof]),
                                            std::abs(value)));
                FE_THROW_IF(std::abs(coefficients[dof] - value) >
                                Real{1.0e-10} * scale,
                            InvalidStateException,
                            std::string(context) +
                                ": non-nodal mesh-vertex projection produced inconsistent shared DOF values");
                return;
            }
            coefficients[dof] = value;
            coefficient_written[dof] = 1u;
            if (!assigned.empty()) {
                assigned[dof] = 1u;
            }
            ++result.values_written;
        };

    for (svmp::index_t cell = 0;
         cell < static_cast<svmp::index_t>(mesh.n_cells());
         ++cell) {
        auto [cell_vertices, n_cell_vertices] = mesh.cell_vertices_span(cell);
        FE_THROW_IF(cell_vertices == nullptr ||
                        n_cell_vertices != scalar_dofs,
                    InvalidStateException,
                    std::string(context) +
                        ": non-nodal mesh-vertex projection requires one mesh point per scalar FE DOF on each cell");
        const auto cell_dofs =
            field_dofs.getCellDofs(static_cast<GlobalIndex>(cell));
        FE_THROW_IF(cell_dofs.size() != scalar_dofs * components,
                    InvalidStateException,
                    std::string(context) +
                        ": non-nodal mesh-vertex projection cell DOF layout mismatch");

        for (std::size_t component = 0; component < components; ++component) {
            std::vector<Real> rhs(scalar_dofs, Real{0});
            for (std::size_t row = 0; row < scalar_dofs; ++row) {
                const auto vertex = cell_vertices[row];
                FE_THROW_IF(vertex < 0 ||
                                static_cast<std::size_t>(vertex) >=
                                    mesh.n_vertices(),
                            InvalidStateException,
                            std::string(context) +
                                ": non-nodal projection mesh vertex index out of range");
                rhs[row] =
                    mesh_values[static_cast<std::size_t>(vertex) *
                                    mesh_components +
                                component];
            }

            const auto local_coefficients =
                solveDenseInterpolationSystem(
                    interpolation_matrix,
                    std::move(rhs),
                    context);
            const std::size_t component_offset = component * scalar_dofs;
            for (std::size_t j = 0; j < scalar_dofs; ++j) {
                const auto dof = cell_dofs[component_offset + j];
                FE_THROW_IF(dof < 0, InvalidStateException,
                            std::string(context) +
                                ": non-nodal projection encountered a negative DOF");
                write_coefficient(static_cast<std::size_t>(dof),
                                  local_coefficients[j]);
            }
        }
    }

    for (std::size_t dof = 0; dof < coefficients.size(); ++dof) {
        if (coefficient_written[dof] == 0u) {
            ++result.unassigned_dofs;
        }
    }
    return result;
}
#endif

} // namespace

std::size_t FESystem::syncBoundMeshMotionFieldsToPrescribedBuffers()
{
    requireSetup();
    const auto& mesh = singleMesh("FESystem::syncBoundMeshMotionFieldsToPrescribedBuffers");

    std::size_t values_written = 0;
    auto& local_mesh = mesh.local_mesh();

    for (const auto role : kFEMeshMotionRoles) {
        const FieldId field = meshMotionRoleValue(mesh_motion_fields_, role);
        if (field == INVALID_FIELD_ID) {
            continue;
        }

        const auto& rec = field_registry_.get(field);
        if (rec.source_kind != FieldSourceKind::PrescribedData) {
            continue;
        }

        const auto field_idx = static_cast<std::size_t>(field);
        FE_THROW_IF(field_idx >= field_dof_handlers_.size(),
                    InvalidStateException,
                    "FESystem::syncBoundMeshMotionFieldsToPrescribedBuffers: invalid field layout for '" +
                        rec.name + "'");

        const auto mesh_role = toMeshMotionRole(role);
        const auto mesh_name = svmp::motion::standard_motion_field_name(mesh_role);
        const auto mesh_field =
            svmp::MeshFields::get_field_handle(local_mesh, svmp::EntityKind::Vertex, mesh_name);
        FE_THROW_IF(mesh_field.id == 0, InvalidArgumentException,
                    "FESystem::syncBoundMeshMotionFieldsToPrescribedBuffers: mesh is missing field '" +
                        std::string(mesh_name) + "'");
        svmp::motion::validate_motion_field(local_mesh,
                                            mesh_field,
                                            mesh_role,
                                            mesh.dim());

        const auto* values = svmp::MeshFields::field_data_as<svmp::real_t>(local_mesh, mesh_field);
        FE_CHECK_NOT_NULL(values,
                          "FESystem::syncBoundMeshMotionFieldsToPrescribedBuffers: mesh field data");

        const auto* entity_map = field_dof_handlers_[field_idx].getEntityDofMap();
        FE_THROW_IF(entity_map == nullptr, InvalidStateException,
                    "FESystem::syncBoundMeshMotionFieldsToPrescribedBuffers: FE field '" + rec.name +
                        "' must have vertex DOFs to sync from mesh fields");

        const auto components = static_cast<std::size_t>(std::max(1, rec.components));
        const auto mesh_components = svmp::MeshFields::field_components(local_mesh, mesh_field);
        FE_THROW_IF(mesh_components < components, InvalidArgumentException,
                    "FESystem::syncBoundMeshMotionFieldsToPrescribedBuffers: mesh field '" +
                        std::string(mesh_name) + "' has fewer components than FE field '" +
                        rec.name + "'");

        const auto n_dofs = static_cast<std::size_t>(field_dof_handlers_[field_idx].getNumDofs());
        if (field_idx >= prescribed_field_buffers_.size()) {
            prescribed_field_buffers_.resize(field_idx + 1u);
        }
        auto& buffer = prescribed_field_buffers_[field_idx];
        buffer.coefficients.assign(n_dofs, Real(0));

        const auto n_vertices = static_cast<GlobalIndex>(mesh.n_vertices());
        FE_THROW_IF(entity_map->numVertices() < n_vertices, InvalidStateException,
                    "FESystem::syncBoundMeshMotionFieldsToPrescribedBuffers: FE field '" + rec.name +
                        "' does not cover every mesh vertex");

        for (GlobalIndex v = 0; v < n_vertices; ++v) {
            const auto vertex_dofs = entity_map->getVertexDofs(v);
            FE_THROW_IF(vertex_dofs.size() != components, InvalidStateException,
                        "FESystem::syncBoundMeshMotionFieldsToPrescribedBuffers: FE field '" + rec.name +
                            "' vertex DOF component count mismatch");
            const auto v_base = static_cast<std::size_t>(v) * mesh_components;
            for (std::size_t c = 0; c < components; ++c) {
                const auto dof = vertex_dofs[c];
                FE_THROW_IF(dof < 0 || static_cast<std::size_t>(dof) >= buffer.coefficients.size(),
                            InvalidStateException,
                            "FESystem::syncBoundMeshMotionFieldsToPrescribedBuffers: field DOF index out of range");
                buffer.coefficients[static_cast<std::size_t>(dof)] =
                    static_cast<Real>(values[v_base + c]);
                ++values_written;
            }
        }
        ++buffer.revision;
    }

    return values_written;
}

std::size_t FESystem::syncPrescribedVertexFieldsFromMeshFields()
{
    requireSetup();
    const auto& mesh = singleMesh("FESystem::syncPrescribedVertexFieldsFromMeshFields");
    const auto& local_mesh = mesh.local_mesh();
    const auto n_vertices = static_cast<GlobalIndex>(mesh.n_vertices());

    std::size_t values_written = 0u;
    for (const auto& rec : field_registry_.records()) {
        if (rec.source_kind != FieldSourceKind::PrescribedData) {
            continue;
        }

        const auto mesh_field = svmp::MeshFields::get_field_handle(
            local_mesh, svmp::EntityKind::Vertex, rec.name);
        if (mesh_field.id == 0) {
            continue;
        }
        FE_THROW_IF(svmp::MeshFields::field_type(local_mesh, mesh_field) !=
                        svmp::FieldScalarType::Float64,
                    InvalidArgumentException,
                    "FESystem::syncPrescribedVertexFieldsFromMeshFields: mesh field '" +
                        rec.name + "' must be Float64");

        const auto* values =
            svmp::MeshFields::field_data_as<svmp::real_t>(local_mesh, mesh_field);
        FE_CHECK_NOT_NULL(values,
                          "FESystem::syncPrescribedVertexFieldsFromMeshFields: mesh field data");

        const auto field_idx = static_cast<std::size_t>(rec.id);
        FE_THROW_IF(field_idx >= field_dof_handlers_.size(), InvalidStateException,
                    "FESystem::syncPrescribedVertexFieldsFromMeshFields: invalid field layout for '" +
                        rec.name + "'");
        const auto* entity_map = field_dof_handlers_[field_idx].getEntityDofMap();
        FE_THROW_IF(entity_map == nullptr, InvalidStateException,
                    "FESystem::syncPrescribedVertexFieldsFromMeshFields: FE field '" +
                        rec.name + "' must have vertex DOFs to sync from mesh fields");

        const auto components =
            static_cast<std::size_t>(std::max(1, rec.components));
        const auto mesh_components =
            svmp::MeshFields::field_components(local_mesh, mesh_field);
        FE_THROW_IF(mesh_components < components, InvalidArgumentException,
                    "FESystem::syncPrescribedVertexFieldsFromMeshFields: mesh field '" +
                        rec.name + "' has fewer components than FE field '" +
                        rec.name + "'");
        const auto mesh_entity_count =
            svmp::MeshFields::field_entity_count(local_mesh, mesh_field);
        FE_THROW_IF(mesh_entity_count < static_cast<std::size_t>(n_vertices),
                    InvalidArgumentException,
                    "FESystem::syncPrescribedVertexFieldsFromMeshFields: mesh field '" +
                        rec.name + "' has fewer entries than mesh vertices");
        FE_THROW_IF(entity_map->numVertices() < n_vertices, InvalidStateException,
                    "FESystem::syncPrescribedVertexFieldsFromMeshFields: FE field '" +
                        rec.name + "' does not cover every mesh vertex");

        if (field_idx >= prescribed_field_buffers_.size()) {
            prescribed_field_buffers_.resize(field_idx + 1u);
        }
        auto& buffer = prescribed_field_buffers_[field_idx];
        buffer.coefficients.assign(
            static_cast<std::size_t>(field_dof_handlers_[field_idx].getNumDofs()),
            Real{0});

        const auto projection =
            projectMeshVertexValuesToFieldCoefficients(
                rec.id,
                std::span<const Real>(
                    reinterpret_cast<const Real*>(values),
                    mesh_entity_count * mesh_components),
                mesh_components,
                std::span<Real>(buffer.coefficients.data(),
                                buffer.coefficients.size()),
                {},
                "FESystem::syncPrescribedVertexFieldsFromMeshFields");
        FE_THROW_IF(projection.unassigned_dofs != 0u, InvalidStateException,
                    "FESystem::syncPrescribedVertexFieldsFromMeshFields: FE field '" +
                        rec.name + "' has " +
                        std::to_string(projection.unassigned_dofs) +
                        " coefficient(s) without a safe mesh-vertex projection");
        values_written += projection.values_written;
        ++buffer.revision;
    }

    return values_written;
}

std::size_t FESystem::syncBoundMeshMotionFieldsToState(std::span<Real> state) const
{
    requireSetup();
    const auto& mesh = singleMesh("FESystem::syncBoundMeshMotionFieldsToState");

    std::size_t values_written = 0;
    auto& local_mesh = mesh.local_mesh();

    std::vector<MeshMotionSyncEntry> entries;
    entries.reserve(kFEMeshMotionRoles.size());
    for (const auto role : kFEMeshMotionRoles) {
        const FieldId field = meshMotionRoleValue(mesh_motion_fields_, role);
        if (field == INVALID_FIELD_ID) {
            continue;
        }
        entries.push_back(MeshMotionSyncEntry{role, field, toMeshMotionRole(role)});
    }

    for (const auto& entry : entries) {
        const auto& rec = field_registry_.get(entry.fe_field);
        if (rec.source_kind != FieldSourceKind::Unknown) {
            continue;
        }
        const auto field_idx = static_cast<std::size_t>(entry.fe_field);
        FE_THROW_IF(field_idx >= field_dof_handlers_.size() ||
                        field_idx >= field_dof_offsets_.size(),
                    InvalidStateException,
                    "FESystem::syncBoundMeshMotionFieldsToState: invalid field layout for '" +
                        rec.name + "'");

        const auto mesh_name = svmp::motion::standard_motion_field_name(entry.mesh_role);
        const auto mesh_field =
            svmp::MeshFields::get_field_handle(local_mesh, svmp::EntityKind::Vertex, mesh_name);
        FE_THROW_IF(mesh_field.id == 0, InvalidArgumentException,
                    "FESystem::syncBoundMeshMotionFieldsToState: mesh is missing field '" +
                        std::string(mesh_name) + "'");
        svmp::motion::validate_motion_field(local_mesh,
                                            mesh_field,
                                            entry.mesh_role,
                                            mesh.dim());

        const auto* values = svmp::MeshFields::field_data_as<svmp::real_t>(local_mesh, mesh_field);
        FE_CHECK_NOT_NULL(values,
                          "FESystem::syncBoundMeshMotionFieldsToState: mesh field data");

        const auto* entity_map = field_dof_handlers_[field_idx].getEntityDofMap();
        FE_THROW_IF(entity_map == nullptr, InvalidStateException,
                    "FESystem::syncBoundMeshMotionFieldsToState: FE field '" + rec.name +
                        "' must have vertex DOFs to sync from mesh fields");

        const auto components = static_cast<std::size_t>(std::max(1, rec.components));
        const auto mesh_components = svmp::MeshFields::field_components(local_mesh, mesh_field);
        FE_THROW_IF(mesh_components < components, InvalidArgumentException,
                    "FESystem::syncBoundMeshMotionFieldsToState: mesh field '" +
                        std::string(mesh_name) + "' has fewer components than FE field '" +
                        rec.name + "'");

        const GlobalIndex offset = field_dof_offsets_[field_idx];
        const auto n_vertices = static_cast<GlobalIndex>(mesh.n_vertices());
        FE_THROW_IF(entity_map->numVertices() < n_vertices, InvalidStateException,
                    "FESystem::syncBoundMeshMotionFieldsToState: FE field '" + rec.name +
                        "' does not cover every mesh vertex");

        for (GlobalIndex v = 0; v < n_vertices; ++v) {
            const auto vertex_dofs = entity_map->getVertexDofs(v);
            FE_THROW_IF(vertex_dofs.size() != components, InvalidStateException,
                        "FESystem::syncBoundMeshMotionFieldsToState: FE field '" + rec.name +
                            "' vertex DOF component count mismatch");
            const auto v_base = static_cast<std::size_t>(v) * mesh_components;
            for (std::size_t c = 0; c < components; ++c) {
                const GlobalIndex dof = offset + vertex_dofs[c];
                FE_THROW_IF(dof < 0 || static_cast<std::size_t>(dof) >= state.size(),
                            InvalidArgumentException,
                            "FESystem::syncBoundMeshMotionFieldsToState: state vector is too small");
                state[static_cast<std::size_t>(dof)] = static_cast<Real>(values[v_base + c]);
                ++values_written;
            }
        }
    }

    return values_written;
}

std::size_t FESystem::syncBoundMeshMotionFieldsToState(assembly::GlobalSystemView& vector_view) const
{
    requireSetup();
    const auto& mesh = singleMesh("FESystem::syncBoundMeshMotionFieldsToState");

    std::size_t values_written = 0;
    auto& local_mesh = mesh.local_mesh();
    std::vector<GlobalIndex> dofs;
    std::vector<Real> values_out;

    for (const auto role : kFEMeshMotionRoles) {
        const FieldId field = meshMotionRoleValue(mesh_motion_fields_, role);
        if (field == INVALID_FIELD_ID) {
            continue;
        }

        const auto mesh_role = toMeshMotionRole(role);
        const auto& rec = field_registry_.get(field);
        if (rec.source_kind != FieldSourceKind::Unknown) {
            continue;
        }
        const auto field_idx = static_cast<std::size_t>(field);
        FE_THROW_IF(field_idx >= field_dof_handlers_.size() ||
                        field_idx >= field_dof_offsets_.size(),
                    InvalidStateException,
                    "FESystem::syncBoundMeshMotionFieldsToState: invalid field layout for '" +
                        rec.name + "'");

        const auto mesh_name = svmp::motion::standard_motion_field_name(mesh_role);
        const auto mesh_field =
            svmp::MeshFields::get_field_handle(local_mesh, svmp::EntityKind::Vertex, mesh_name);
        FE_THROW_IF(mesh_field.id == 0, InvalidArgumentException,
                    "FESystem::syncBoundMeshMotionFieldsToState: mesh is missing field '" +
                        std::string(mesh_name) + "'");
        svmp::motion::validate_motion_field(local_mesh,
                                            mesh_field,
                                            mesh_role,
                                            mesh.dim());
        const auto* values = svmp::MeshFields::field_data_as<svmp::real_t>(local_mesh, mesh_field);
        FE_CHECK_NOT_NULL(values,
                          "FESystem::syncBoundMeshMotionFieldsToState: mesh field data");

        const auto* entity_map = field_dof_handlers_[field_idx].getEntityDofMap();
        FE_THROW_IF(entity_map == nullptr, InvalidStateException,
                    "FESystem::syncBoundMeshMotionFieldsToState: FE field '" + rec.name +
                        "' must have vertex DOFs to sync from mesh fields");

        const auto components = static_cast<std::size_t>(std::max(1, rec.components));
        const auto mesh_components = svmp::MeshFields::field_components(local_mesh, mesh_field);
        FE_THROW_IF(mesh_components < components, InvalidArgumentException,
                    "FESystem::syncBoundMeshMotionFieldsToState: mesh field '" +
                        std::string(mesh_name) + "' has fewer components than FE field '" +
                        rec.name + "'");

        const GlobalIndex offset = field_dof_offsets_[field_idx];
        const auto n_vertices = static_cast<GlobalIndex>(mesh.n_vertices());
        dofs.clear();
        values_out.clear();
        dofs.reserve(static_cast<std::size_t>(n_vertices) * components);
        values_out.reserve(static_cast<std::size_t>(n_vertices) * components);

        for (GlobalIndex v = 0; v < n_vertices; ++v) {
            const auto vertex_dofs = entity_map->getVertexDofs(v);
            FE_THROW_IF(vertex_dofs.size() != components, InvalidStateException,
                        "FESystem::syncBoundMeshMotionFieldsToState: FE field '" + rec.name +
                            "' vertex DOF component count mismatch");
            const auto v_base = static_cast<std::size_t>(v) * mesh_components;
            for (std::size_t c = 0; c < components; ++c) {
                dofs.push_back(offset + vertex_dofs[c]);
                values_out.push_back(static_cast<Real>(values[v_base + c]));
            }
        }

        vector_view.setVectorEntries(dofs, values_out);
        values_written += values_out.size();
    }

    return values_written;
}

FEAdaptedStateTransferResult FESystem::onMeshAdapted(
    const svmp::MeshBase& old_mesh,
    const svmp::MeshBase& new_mesh,
    const svmp::RefinementDelta& delta,
    const svmp::AdaptivityOptions& options,
    const FEAdaptedStateTransferRequest& request)
{
    (void)options;
    requireSetup();

    const bool transfer_solution =
        request.transferred_solution != nullptr && !request.solution.empty();
    const bool transfer_previous =
        request.transferred_previous_solution != nullptr && !request.previous_solution.empty();
    const bool transfer_previous2 =
        request.transferred_previous_solution2 != nullptr && !request.previous_solution2.empty();
    const bool any_transfer = transfer_solution || transfer_previous || transfer_previous2;

    FE_THROW_IF(any_transfer && !request.rebuild_setup, InvalidArgumentException,
                "FESystem::onMeshAdapted: FE state transfer requires rebuild_setup=true");

    FEAdaptedStateTransferResult result;
    result.layout_before = feLayoutRevisionState();

    struct PendingFieldTransfer {
        FieldId field{INVALID_FIELD_ID};
        int components{1};
        std::vector<Real> solution{};
        std::vector<Real> previous{};
        std::vector<Real> previous2{};
        bool has_solution{false};
        bool has_previous{false};
        bool has_previous2{false};
    };

    std::vector<PendingFieldTransfer> pending;
    pending.reserve(field_registry_.size());

    const auto old_total_dofs = static_cast<std::size_t>(dof_handler_.getNumDofs());

    auto extract_vertex_values = [&](const FieldRecord& rec,
                                     std::span<const Real> state,
                                     std::vector<Real>& vertex_values,
                                     const char* label) {
        FE_THROW_IF(state.size() < old_total_dofs, InvalidArgumentException,
                    std::string("FESystem::onMeshAdapted: ") + label +
                        " vector is smaller than the pre-adaptation DOF layout");

        const auto field_idx = static_cast<std::size_t>(rec.id);
        FE_THROW_IF(field_idx >= field_dof_handlers_.size() ||
                        field_idx >= field_dof_offsets_.size(),
                    InvalidStateException,
                    "FESystem::onMeshAdapted: invalid field layout for '" + rec.name + "'");

        const auto* entity_map = field_dof_handlers_[field_idx].getEntityDofMap();
        FE_THROW_IF(entity_map == nullptr, InvalidStateException,
                    "FESystem::onMeshAdapted: field '" + rec.name +
                        "' has no entity DOF map for nodal transfer");
        FE_THROW_IF(static_cast<std::size_t>(entity_map->numVertices()) < old_mesh.n_vertices(),
                    InvalidStateException,
                    "FESystem::onMeshAdapted: field '" + rec.name +
                        "' does not cover every old mesh vertex");

        const auto components = static_cast<std::size_t>(std::max(1, rec.components));
        vertex_values.assign(old_mesh.n_vertices() * components, Real(0));
        const GlobalIndex offset = field_dof_offsets_[field_idx];
        for (std::size_t v = 0; v < old_mesh.n_vertices(); ++v) {
            const auto vertex_dofs = entity_map->getVertexDofs(static_cast<GlobalIndex>(v));
            FE_THROW_IF(vertex_dofs.size() < components, InvalidStateException,
                        "FESystem::onMeshAdapted: field '" + rec.name +
                            "' vertex DOF component count is smaller than the field component count");
            for (std::size_t c = 0; c < components; ++c) {
                const GlobalIndex dof = offset + vertex_dofs[c];
                FE_THROW_IF(dof < 0 || static_cast<std::size_t>(dof) >= state.size(),
                            InvalidArgumentException,
                            "FESystem::onMeshAdapted: " + std::string(label) +
                                " vector does not contain a required DOF for field '" +
                                rec.name + "'");
                vertex_values[v * components + c] = state[static_cast<std::size_t>(dof)];
            }
        }
    };

    auto prepare_transfer = [&](const FieldRecord& rec,
                                std::span<const Real> source,
                                std::vector<Real>& target_values,
                                const char* label) {
        std::vector<Real> old_values;
        extract_vertex_values(rec, source, old_values, label);
        target_values.assign(new_mesh.n_vertices() * static_cast<std::size_t>(std::max(1, rec.components)),
                             Real(0));
        const auto diag = transferNodalFieldByVertexProvenance(old_mesh,
                                                               new_mesh,
                                                               delta,
                                                               std::max(1, rec.components),
                                                               old_values,
                                                               target_values,
                                                               request.field_transfer_options);
        result.values_transferred += diag.values_transferred;
        result.diagnostics.insert(result.diagnostics.end(),
                                  diag.diagnostics.begin(),
                                  diag.diagnostics.end());
        if (!diag.success) {
            result.diagnostics.push_back("Nodal transfer for field '" + rec.name +
                                         "' reported incomplete diagnostics");
        }
    };

    if (any_transfer) {
        for (const auto& rec : field_registry_.records()) {
            if (rec.scope != FieldScope::VolumeCell) {
                continue;
            }
            const auto field_idx = static_cast<std::size_t>(rec.id);
            if (field_idx >= field_dof_handlers_.size()) {
                continue;
            }
            const auto* entity_map = field_dof_handlers_[field_idx].getEntityDofMap();
            if (entity_map == nullptr ||
                static_cast<std::size_t>(entity_map->numVertices()) < old_mesh.n_vertices()) {
                result.diagnostics.push_back("Skipping field '" + rec.name +
                                             "' during adaptation transfer because it has no complete vertex DOF layout");
                continue;
            }

            PendingFieldTransfer item;
            item.field = rec.id;
            item.components = std::max(1, rec.components);
            if (transfer_solution) {
                prepare_transfer(rec, request.solution, item.solution, "solution");
                item.has_solution = true;
            }
            if (transfer_previous) {
                prepare_transfer(rec, request.previous_solution, item.previous, "previous solution");
                item.has_previous = true;
            }
            if (transfer_previous2) {
                prepare_transfer(rec, request.previous_solution2, item.previous2, "second previous solution");
                item.has_previous2 = true;
            }
            if (item.has_solution || item.has_previous || item.has_previous2) {
                pending.push_back(std::move(item));
            }
        }
    }

    const auto state_transfer_request = makeStateFrameTransformRequest(
        state::StateFrameTransformEvent::AdaptivityTransfer,
        operatorRevisionSnapshot());
    if (request.transfer_material_state && material_state_provider_) {
        (void)material_state_provider_->applyStateFrameTransform(state_transfer_request);
    }
    if (request.transfer_auxiliary_state && auxiliary_state_manager_) {
        (void)auxiliary_state_manager_->applyStateFrameTransform(state_transfer_request);
    }

    bool auxiliary_transfer_complete = true;
    if (request.transfer_auxiliary_state && auxiliary_state_manager_) {
        const auto block_names = auxiliary_state_manager_->state().blockNames();
        for (const auto& block_name : block_names) {
            const auto& spec = auxiliary_state_manager_->getSpec(block_name);
            const auto& block = auxiliary_state_manager_->getBlock(block_name);
            std::optional<std::size_t> new_entity_count;

            switch (spec.scope) {
                case AuxiliaryStateScope::Global:
                case AuxiliaryStateScope::Boundary:
                case AuxiliaryStateScope::Region:
                    new_entity_count = block.entityCount();
                    break;
                case AuxiliaryStateScope::Node:
                    new_entity_count = new_mesh.n_vertices();
                    break;
                case AuxiliaryStateScope::Cell:
                    new_entity_count = new_mesh.n_cells();
                    break;
                case AuxiliaryStateScope::Facet:
                    new_entity_count = new_mesh.n_faces();
                    break;
                case AuxiliaryStateScope::QuadraturePoint:
                    if (spec.transfer_policy == AuxiliaryTransferPolicy::None) {
                        new_entity_count = block.entityCount();
                    } else {
                        auxiliary_transfer_complete = false;
                        result.diagnostics.push_back(
                            "Auxiliary quadrature-point block '" + block_name +
                            "' requires a remesh-specific quadrature transfer outside the generic Phase 10 path");
                    }
                    break;
            }

            if (!new_entity_count.has_value()) {
                continue;
            }

            if (spec.transfer_policy == AuxiliaryTransferPolicy::None) {
                auxiliary_state_manager_->reinitializeBlock(block_name, *new_entity_count);
                result.diagnostics.push_back(
                    "Auxiliary block '" + block_name +
                    "' was reinitialized because its transfer policy is None");
            } else {
                auxiliary_state_manager_->transferBlock(block_name, *new_entity_count);
                result.diagnostics.push_back(
                    "Auxiliary block '" + block_name +
                    "' was transferred through its registered physics-agnostic transfer hook or default resize policy");
            }
        }
    }

    const bool had_boundary_services = !boundary_reduction_services_.empty();
    const bool had_interface_meshes = !interface_meshes_.empty();
    if (had_boundary_services) {
        boundary_reduction_services_.clear();
        result.diagnostics.push_back(
            "Boundary reduction services were invalidated because the mesh topology changed");
    }
    if (had_interface_meshes) {
        interface_meshes_.clear();
        result.diagnostics.push_back(
            "Interface meshes were invalidated because remeshing changes interface entity identity");
    }

    notifyMeshTopologyLayoutChanged();
    if (request.rebuild_setup) {
        setup(request.setup_options);
        result.dof_handler_rebuilt = true;
        result.constraint_layout_rebuilt = true;
        result.sparsity_rebuilt = true;
    }

    auto scatter_vertex_values = [&](const std::vector<Real>& vertex_values,
                                     const PendingFieldTransfer& item,
                                     std::vector<Real>& state_out,
                                     const char* label) {
        const auto field_idx = static_cast<std::size_t>(item.field);
        FE_THROW_IF(field_idx >= field_dof_handlers_.size() ||
                        field_idx >= field_dof_offsets_.size(),
                    InvalidStateException,
                    "FESystem::onMeshAdapted: invalid post-adaptation field layout");

        const auto* entity_map = field_dof_handlers_[field_idx].getEntityDofMap();
        FE_THROW_IF(entity_map == nullptr, InvalidStateException,
                    "FESystem::onMeshAdapted: post-adaptation field has no entity DOF map");
        FE_THROW_IF(static_cast<std::size_t>(entity_map->numVertices()) < new_mesh.n_vertices(),
                    InvalidStateException,
                    "FESystem::onMeshAdapted: post-adaptation field does not cover every new mesh vertex");

        const auto components = static_cast<std::size_t>(std::max(1, item.components));
        FE_THROW_IF(vertex_values.size() < new_mesh.n_vertices() * components,
                    InvalidStateException,
                    "FESystem::onMeshAdapted: transferred vertex data are incomplete");

        const GlobalIndex offset = field_dof_offsets_[field_idx];
        for (std::size_t v = 0; v < new_mesh.n_vertices(); ++v) {
            const auto vertex_dofs = entity_map->getVertexDofs(static_cast<GlobalIndex>(v));
            FE_THROW_IF(vertex_dofs.size() < components, InvalidStateException,
                        "FESystem::onMeshAdapted: post-adaptation vertex DOF component count mismatch");
            for (std::size_t c = 0; c < components; ++c) {
                const GlobalIndex dof = offset + vertex_dofs[c];
                FE_THROW_IF(dof < 0 || static_cast<std::size_t>(dof) >= state_out.size(),
                            InvalidStateException,
                            "FESystem::onMeshAdapted: " + std::string(label) +
                                " output vector is smaller than the rebuilt DOF layout");
                state_out[static_cast<std::size_t>(dof)] = vertex_values[v * components + c];
            }
        }
    };

    const auto new_total_dofs = static_cast<std::size_t>(dof_handler_.getNumDofs());
    if (transfer_solution) {
        request.transferred_solution->assign(new_total_dofs, Real(0));
    }
    if (transfer_previous) {
        request.transferred_previous_solution->assign(new_total_dofs, Real(0));
    }
    if (transfer_previous2) {
        request.transferred_previous_solution2->assign(new_total_dofs, Real(0));
    }

    for (const auto& item : pending) {
        if (transfer_solution && item.has_solution) {
            scatter_vertex_values(item.solution, item, *request.transferred_solution, "solution");
        }
        if (transfer_previous && item.has_previous) {
            scatter_vertex_values(item.previous, item, *request.transferred_previous_solution, "previous solution");
        }
        if (transfer_previous2 && item.has_previous2) {
            scatter_vertex_values(item.previous2, item, *request.transferred_previous_solution2, "second previous solution");
        }
    }

    result.solution_transferred = transfer_solution;
    result.previous_solution_transferred = transfer_previous;
    result.previous_solution2_transferred = transfer_previous2;
    result.auxiliary_state_transfer_handled = request.transfer_auxiliary_state && auxiliary_transfer_complete;
    result.material_state_transfer_handled = request.transfer_material_state;
    result.boundary_coupling_state_transfer_handled = request.transfer_boundary_and_coupling_state;
    if (request.transfer_auxiliary_state) {
        if (!auxiliary_state_manager_) {
            result.diagnostics.push_back(
                "No auxiliary state manager was present during adaptation");
        } else {
            result.diagnostics.push_back(
                "Auxiliary state layout was invalidated and rebuilt through setup(); value remap used registered block transfer hooks where available");
        }
    }
    if (request.transfer_material_state) {
        result.diagnostics.push_back(
            "Material/global kernel state providers were invalidated and rebuilt through setup(); constitutive history transformations remain provider-owned");
    }
    if (request.transfer_boundary_and_coupling_state) {
        result.diagnostics.push_back(
            "Boundary/coupling state was invalidated at the topology-change boundary; callers must rebuild mesh-dependent coupling operators after remesh");
    }
    result.layout_after = feLayoutRevisionState();
    return result;
}

void FESystem::notifyMeshGeometryAdvanced()
{
    const auto transform_request = makeStateFrameTransformRequest(
        state::StateFrameTransformEvent::OrdinaryGeometryMotion,
        operatorRevisionSnapshot());
    if (material_state_provider_) {
        (void)material_state_provider_->applyStateFrameTransform(transform_request);
    }
    if (auxiliary_state_manager_) {
        (void)auxiliary_state_manager_->applyStateFrameTransform(transform_request);
    }
    if (assembler_) {
        assembler_->invalidateGeometryCaches();
    }
    invalidateAnalysisCache();
}

void FESystem::notifyMeshReferenceRebased()
{
    const auto transform_request = makeStateFrameTransformRequest(
        state::StateFrameTransformEvent::ReferenceRebase,
        operatorRevisionSnapshot());
    if (material_state_provider_) {
        (void)material_state_provider_->applyStateFrameTransform(transform_request);
    }
    if (auxiliary_state_manager_) {
        (void)auxiliary_state_manager_->applyStateFrameTransform(transform_request);
    }
    if (assembler_) {
        assembler_->invalidateGeometryCaches();
    }
    invalidateAnalysisCache();
    if (material_state_provider_) {
        material_state_provider_->rollbackTimeStep();
    }
    if (global_kernel_state_provider_) {
        global_kernel_state_provider_->rollbackTimeStep();
    }
    if (auxiliary_state_manager_) {
        auxiliary_state_manager_->resetAllToCommitted();
    }
    if (auxiliary_input_registry_) {
        auxiliary_input_registry_->invalidateAll();
    }
    partitioned_auxiliary_advance_valid_ = false;
    partitioned_auxiliary_advance_time_ = std::numeric_limits<Real>::quiet_NaN();
    partitioned_auxiliary_advance_dt_ = std::numeric_limits<Real>::quiet_NaN();
}

void FESystem::notifyMeshTopologyLayoutChanged()
{
    invalidateSetup();
    bumpDofLayoutRevision();
    bumpConstraintLayoutRevision();
    bumpBlockLayoutRevision();
    invalidateAnalysisCache();
}
#endif

FieldId FESystem::addInterfaceField(std::string name,
                                    std::shared_ptr<const spaces::FunctionSpace> space,
                                    InterfaceId interface_marker,
                                    int components)
{
    FieldSpec spec;
    spec.name = std::move(name);
    spec.space = std::move(space);
    spec.components = components;
    spec.scope = FieldScope::InterfaceFace;
    spec.interface_marker = interface_marker;
    return addField(std::move(spec));
}

void FESystem::addConstraint(std::unique_ptr<constraints::Constraint> c)
{
    invalidateSetup();
    FE_CHECK_NOT_NULL(c.get(), "FESystem::addConstraint: constraint");
    constraint_defs_.push_back(std::move(c));
    bumpConstraintLayoutRevision();
}

void FESystem::addSystemConstraint(std::unique_ptr<constraints::ISystemConstraint> c)
{
    invalidateSetup();
    FE_CHECK_NOT_NULL(c.get(), "FESystem::addSystemConstraint: constraint");
    system_constraint_defs_.push_back(std::move(c));
    bumpConstraintLayoutRevision();
}

void FESystem::addOperator(OperatorTag name)
{
    invalidateSetup();
    operator_registry_.addOperator(std::move(name));
}

void FESystem::setFormInstallCellDomainRestrictions(
    std::vector<FormCellDomainRestriction> restrictions)
{
    form_install_cell_domain_restrictions_ = std::move(restrictions);
}

const std::vector<FESystem::FormCellDomainRestriction>&
FESystem::formInstallCellDomainRestrictions() const noexcept
{
    return form_install_cell_domain_restrictions_;
}

FESystem::FormCellDomainRestrictionScope::FormCellDomainRestrictionScope(
    FESystem& system,
    std::vector<FormCellDomainRestriction> restrictions)
    : system_(&system)
    , previous_(system.formInstallCellDomainRestrictions())
{
    system_->setFormInstallCellDomainRestrictions(std::move(restrictions));
}

FESystem::FormCellDomainRestrictionScope::~FormCellDomainRestrictionScope()
{
    restore();
}

FESystem::FormCellDomainRestrictionScope::FormCellDomainRestrictionScope(
    FormCellDomainRestrictionScope&& other) noexcept
    : system_(std::exchange(other.system_, nullptr))
    , previous_(std::move(other.previous_))
{
}

FESystem::FormCellDomainRestrictionScope&
FESystem::FormCellDomainRestrictionScope::operator=(
    FormCellDomainRestrictionScope&& other) noexcept
{
    if (this != &other) {
        restore();
        system_ = std::exchange(other.system_, nullptr);
        previous_ = std::move(other.previous_);
    }
    return *this;
}

void FESystem::FormCellDomainRestrictionScope::restore() noexcept
{
    if (system_ == nullptr) {
        return;
    }
    system_->setFormInstallCellDomainRestrictions(std::move(previous_));
    system_ = nullptr;
}

FESystem::FormCellDomainRestrictionScope
FESystem::scopedFormInstallCellDomainRestrictions(
    std::vector<FormCellDomainRestriction> restrictions)
{
    return FormCellDomainRestrictionScope(*this, std::move(restrictions));
}

void FESystem::addCellKernel(OperatorTag op, FieldId field,
                             std::shared_ptr<assembly::AssemblyKernel> kernel)
{
    addCellKernel(std::move(op), field, field, std::move(kernel));
}

void FESystem::addCellKernel(OperatorTag op, FieldId test_field, FieldId trial_field,
                             std::shared_ptr<assembly::AssemblyKernel> kernel)
{
    if (!form_install_cell_domain_restrictions_.empty()) {
        for (const auto& restriction : form_install_cell_domain_restrictions_) {
            addCutVolumeKernel(op,
                               restriction.interface_marker,
                               restriction.side,
                               test_field,
                               trial_field,
                               kernel);
            if (kernel && kernel->hasInterfaceFace()) {
                addInterfaceFaceKernel(op,
                                       restriction.interface_marker,
                                       test_field,
                                       trial_field,
                                       kernel);
            }
        }
        return;
    }

    invalidateSetup();
    validateKernelFieldScopes(field_registry_, test_field, trial_field,
                              analysis::DomainKind::Cell,
                              "FESystem::addCellKernel");
    if (!operator_registry_.has(op)) {
        operator_registry_.addOperator(op);
    }
    auto& def = operator_registry_.get(op);
    if (kernel) {
        field_registry_.markTimeDependent(trial_field, kernel->maxTemporalDerivativeOrder());
    }
    def.cells.push_back(CellTerm{test_field, trial_field, std::move(kernel)});
}

void FESystem::addBoundaryKernel(OperatorTag op, BoundaryId boundary, FieldId field,
                                 std::shared_ptr<assembly::AssemblyKernel> kernel)
{
    addBoundaryKernel(std::move(op), boundary, field, field, std::move(kernel));
}

void FESystem::addBoundaryKernel(OperatorTag op, BoundaryId boundary, FieldId test_field,
                                 FieldId trial_field, std::shared_ptr<assembly::AssemblyKernel> kernel)
{
    invalidateSetup();
    validateKernelFieldScopes(field_registry_, test_field, trial_field,
                              analysis::DomainKind::Boundary,
                              "FESystem::addBoundaryKernel");
    if (!operator_registry_.has(op)) {
        operator_registry_.addOperator(op);
    }
    auto& def = operator_registry_.get(op);
    if (kernel) {
        field_registry_.markTimeDependent(trial_field, kernel->maxTemporalDerivativeOrder());
    }
    def.boundary.push_back(BoundaryTerm{boundary, test_field, trial_field, std::move(kernel)});
}

void FESystem::addInteriorFaceKernel(OperatorTag op, FieldId field,
                                     std::shared_ptr<assembly::AssemblyKernel> kernel)
{
    addInteriorFaceKernel(std::move(op), /*interior_facet_marker=*/-1,
                          field, field, std::move(kernel));
}

void FESystem::addInteriorFaceKernel(OperatorTag op, FieldId test_field, FieldId trial_field,
                                     std::shared_ptr<assembly::AssemblyKernel> kernel)
{
    addInteriorFaceKernel(std::move(op), /*interior_facet_marker=*/-1,
                          test_field, trial_field, std::move(kernel));
}

void FESystem::addInteriorFaceKernel(OperatorTag op,
                                     int interior_facet_marker,
                                     FieldId field,
                                     std::shared_ptr<assembly::AssemblyKernel> kernel)
{
    addInteriorFaceKernel(std::move(op), interior_facet_marker,
                          field, field, std::move(kernel));
}

void FESystem::addInteriorFaceKernel(OperatorTag op,
                                     int interior_facet_marker,
                                     FieldId test_field,
                                     FieldId trial_field,
                                     std::shared_ptr<assembly::AssemblyKernel> kernel)
{
    invalidateSetup();
    validateKernelFieldScopes(field_registry_, test_field, trial_field,
                              analysis::DomainKind::InteriorFace,
                              "FESystem::addInteriorFaceKernel");
    if (!operator_registry_.has(op)) {
        operator_registry_.addOperator(op);
    }
    auto& def = operator_registry_.get(op);
    if (kernel) {
        field_registry_.markTimeDependent(trial_field, kernel->maxTemporalDerivativeOrder());
    }
    def.interior.push_back(
        InteriorFaceTerm{interior_facet_marker, test_field, trial_field, std::move(kernel)});
}

void FESystem::addInterfaceFaceKernel(OperatorTag op, InterfaceId interface_marker, FieldId field,
                                      std::shared_ptr<assembly::AssemblyKernel> kernel)
{
    addInterfaceFaceKernel(std::move(op), interface_marker, field, field, std::move(kernel));
}

void FESystem::addInterfaceFaceKernel(OperatorTag op, InterfaceId interface_marker, FieldId test_field, FieldId trial_field,
                                      std::shared_ptr<assembly::AssemblyKernel> kernel)
{
    invalidateSetup();
    validateKernelFieldScopes(field_registry_, test_field, trial_field,
                              analysis::DomainKind::InterfaceFace,
                              "FESystem::addInterfaceFaceKernel");
    if (!operator_registry_.has(op)) {
        operator_registry_.addOperator(op);
    }
    auto& def = operator_registry_.get(op);
    if (kernel) {
        field_registry_.markTimeDependent(trial_field, kernel->maxTemporalDerivativeOrder());
    }
    def.interface_faces.push_back(InterfaceFaceTerm{interface_marker, test_field, trial_field, std::move(kernel)});
}

void FESystem::addCutVolumeKernel(OperatorTag op,
                                  InterfaceId interface_marker,
                                  geometry::CutIntegrationSide side,
                                  FieldId field,
                                  std::shared_ptr<assembly::AssemblyKernel> kernel)
{
    addCutVolumeKernel(std::move(op), interface_marker, side, field, field, std::move(kernel));
}

void FESystem::addCutVolumeKernel(OperatorTag op,
                                  InterfaceId interface_marker,
                                  geometry::CutIntegrationSide side,
                                  FieldId test_field,
                                  FieldId trial_field,
                                  std::shared_ptr<assembly::AssemblyKernel> kernel)
{
    invalidateSetup();
    FE_THROW_IF(side == geometry::CutIntegrationSide::Interface,
                InvalidArgumentException,
                "FESystem::addCutVolumeKernel: side must be Negative or Positive");
    validateKernelFieldScopes(field_registry_, test_field, trial_field,
                              analysis::DomainKind::Cell,
                              "FESystem::addCutVolumeKernel");
    if (!operator_registry_.has(op)) {
        operator_registry_.addOperator(op);
    }
    if (interface_marker >= 0) {
        generated_embedded_interface_markers_.insert(interface_marker);
    }
    auto& def = operator_registry_.get(op);
    if (kernel) {
        field_registry_.markTimeDependent(trial_field, kernel->maxTemporalDerivativeOrder());
    }
    def.cut_volumes.push_back(CutVolumeTerm{
        interface_marker, side, test_field, trial_field, std::move(kernel)});
}

std::size_t FESystem::cutVolumeKernelCount(
    int interface_marker,
    geometry::CutIntegrationSide side) const
{
    std::size_t count = 0u;
    for (const auto& op : operator_registry_.list()) {
        const auto& def = operator_registry_.get(op);
        for (const auto& term : def.cut_volumes) {
            if ((term.marker == interface_marker || term.marker < 0) &&
                term.side == side) {
                ++count;
            }
        }
    }
    return count;
}

void FESystem::addGlobalKernel(OperatorTag op, std::shared_ptr<GlobalKernel> kernel)
{
    invalidateSetup();
    if (!operator_registry_.has(op)) {
        operator_registry_.addOperator(op);
    }
    FE_CHECK_NOT_NULL(kernel.get(), "FESystem::addGlobalKernel: kernel");
    operator_registry_.get(op).global.push_back(std::move(kernel));
}

void FESystem::addMatrixFreeKernel(OperatorTag op,
                                   std::shared_ptr<assembly::IMatrixFreeKernel> kernel)
{
    FE_CHECK_NOT_NULL(operator_backends_.get(), "FESystem::operator_backends");
    operator_backends_->registerMatrixFree(std::move(op), std::move(kernel));
}

void FESystem::addMatrixFreeKernel(OperatorTag op,
                                   std::shared_ptr<assembly::IMatrixFreeKernel> kernel,
                                   const assembly::MatrixFreeOptions& options)
{
    FE_CHECK_NOT_NULL(operator_backends_.get(), "FESystem::operator_backends");
    operator_backends_->registerMatrixFree(std::move(op), std::move(kernel), options);
}

std::shared_ptr<assembly::MatrixFreeOperator> FESystem::matrixFreeOperator(const OperatorTag& op) const
{
    requireSingleFieldSetup();
    FE_CHECK_NOT_NULL(operator_backends_.get(), "FESystem::operator_backends");
    return operator_backends_->matrixFreeOperator(*this, op);
}

std::size_t FESystem::matrixFreeOperatorRebuildCount(const OperatorTag& op) const
{
    FE_CHECK_NOT_NULL(operator_backends_.get(), "FESystem::operator_backends");
    return operator_backends_->matrixFreeRebuildCount(op);
}

OperatorRevisionSnapshot FESystem::matrixFreeOperatorRevisionSnapshot(const OperatorTag& op) const
{
    FE_CHECK_NOT_NULL(operator_backends_.get(), "FESystem::operator_backends");
    return operator_backends_->matrixFreeRevisionSnapshot(op);
}

OperatorInvalidationDecision FESystem::matrixFreeOperatorLastInvalidation(const OperatorTag& op) const
{
    FE_CHECK_NOT_NULL(operator_backends_.get(), "FESystem::operator_backends");
    return operator_backends_->matrixFreeLastInvalidation(op);
}

post::DerivedResultHandle FESystem::addDerivedResult(post::DerivedResultDefinition def)
{
    if (!derived_result_registry_) {
        derived_result_registry_ = std::make_unique<post::DerivedResultRegistry>();
    }
    return derived_result_registry_->registerDefinition(std::move(def));
}

std::span<const post::DerivedResultDefinition> FESystem::derivedResults() const noexcept
{
    if (!derived_result_registry_) {
        return {};
    }
    return derived_result_registry_->all();
}

void FESystem::appendDerivedResultFields(
    svmp::MeshBase& mesh,
    const SystemStateView& state,
    const post::DerivedResultOutputOptions& options) const
{
    if (!derived_result_registry_ || derived_result_registry_->all().empty()) {
        return;
    }
    requireSetup();
    post::DerivedResultEvaluator evaluator(*this, state);
    for (const auto& def : derived_result_registry_->all()) {
        evaluator.evaluateToMeshField(mesh, def, options);
    }
}

void FESystem::addFunctionalKernel(std::string tag,
                                   std::shared_ptr<assembly::FunctionalKernel> kernel)
{
    FE_CHECK_NOT_NULL(operator_backends_.get(), "FESystem::operator_backends");
    operator_backends_->registerFunctional(std::move(tag), std::move(kernel));
}

Real FESystem::evaluateFunctional(const std::string& tag, const SystemStateView& state) const
{
    requireSingleFieldSetup();
    FE_CHECK_NOT_NULL(operator_backends_.get(), "FESystem::operator_backends");
    return operator_backends_->evaluateFunctional(*this, tag, state);
}

Real FESystem::evaluateBoundaryFunctional(const std::string& tag,
                                          int boundary_marker,
                                          const SystemStateView& state) const
{
    requireSingleFieldSetup();
    FE_CHECK_NOT_NULL(operator_backends_.get(), "FESystem::operator_backends");
    return operator_backends_->evaluateBoundaryFunctional(*this, tag, boundary_marker, state);
}

AuxiliaryStateManager& FESystem::auxiliaryStateManager()
{
    if (!auxiliary_state_manager_) {
        auxiliary_state_manager_ = std::make_unique<AuxiliaryStateManager>();
    }
    return *auxiliary_state_manager_;
}

AuxiliaryOperatorRegistry& FESystem::auxiliaryOperatorRegistry()
{
    if (!auxiliary_operator_registry_) {
        auxiliary_operator_registry_ = std::make_unique<AuxiliaryOperatorRegistry>();
    }
    return *auxiliary_operator_registry_;
}

AuxiliaryInputRegistry& FESystem::auxiliaryInputRegistry()
{
    if (!auxiliary_input_registry_) {
        auxiliary_input_registry_ = std::make_unique<AuxiliaryInputRegistry>();
    }
    return *auxiliary_input_registry_;
}

FEQuantityRegistry& FESystem::feQuantityRegistry()
{
    if (!fe_quantity_registry_) {
        fe_quantity_registry_ = std::make_unique<FEQuantityRegistry>();
    }
    return *fe_quantity_registry_;
}

std::span<const backends::RankOneUpdate> FESystem::lastRankOneUpdates() const noexcept
{
    return last_rank_one_updates_;
}

void FESystem::clearRankOneUpdates() noexcept
{
    last_rank_one_updates_.clear();
}

std::span<const backends::ReducedFieldUpdate> FESystem::lastReducedFieldUpdates() const noexcept
{
    return last_reduced_field_updates_;
}

void FESystem::clearReducedFieldUpdates() noexcept
{
    last_reduced_field_updates_.clear();
}

std::span<const Real> FESystem::lastLocalCondensedRhsShift() const noexcept
{
    return last_local_condensed_rhs_shift_;
}

void FESystem::applyLocalCondensedRecovery(std::span<const Real> dense_du, Real alpha)
{
    if (last_local_condensed_records_.empty() ||
        dense_du.empty() ||
        std::abs(alpha) <= Real(0.0) ||
        !auxiliary_state_manager_) {
        return;
    }

    for (const auto& rec : last_local_condensed_records_) {
        if (!rec.has_aux_equation_terms) {
            continue;
        }
        if (!auxiliary_state_manager_->hasBlock(rec.block_name)) {
            continue;
        }
        auto& blk = auxiliary_state_manager_->getBlock(rec.block_name);
        auto entity_state = blk.gatherEntityWork(rec.entity_index);
        const auto dim = entity_state.size();
        if (rec.D_inv.size() != dim * dim || rec.g.size() != dim ||
            rec.Ct_rows.size() != dim) {
            continue;
        }

        std::vector<Real> rhs = rec.g;
        for (std::size_t row = 0; row < dim; ++row) {
            for (const auto& [dof, val] : rec.Ct_rows[row]) {
                const auto dof_idx = static_cast<std::size_t>(dof);
                if (dof_idx < dense_du.size()) {
                    rhs[row] -= val * dense_du[dof_idx];
                }
            }
        }

        std::vector<Real> delta(dim, Real(0.0));
        for (std::size_t i = 0; i < dim; ++i) {
            for (std::size_t j = 0; j < dim; ++j) {
                delta[i] += rec.D_inv[i * dim + j] * rhs[j];
            }
        }

        for (std::size_t i = 0; i < dim; ++i) {
            entity_state[i] -= alpha * delta[i];
        }
        blk.scatterEntityWork(rec.entity_index, entity_state);
    }

    auxiliary_state_manager_->syncGhosts();
}

void FESystem::clearLocalCondensedRecovery() noexcept
{
    last_local_condensed_records_.clear();
    last_local_condensed_rhs_shift_.clear();
}

const assembly::IMeshAccess& FESystem::meshAccess() const
{
    FE_CHECK_NOT_NULL(mesh_access_.get(), "FESystem::meshAccess");
    return *mesh_access_;
}

std::span<const MeshParticipantInfo> FESystem::meshParticipants() const noexcept
{
    return mesh_participants_;
}

const MeshParticipantInfo* FESystem::meshParticipantByName(std::string_view name) const noexcept
{
    for (const auto& participant : mesh_participants_) {
        if (participant.name == name) {
            return &participant;
        }
    }
    return nullptr;
}

const MeshParticipantInfo* FESystem::meshParticipantByDomain(int domain_id) const noexcept
{
    for (const auto& participant : mesh_participants_) {
        if (participant.domain_id.has_value() && *participant.domain_id == domain_id) {
            return &participant;
        }
    }
    return nullptr;
}

const MeshParticipantInfo* FESystem::meshParticipantForCell(GlobalIndex cell_id) const noexcept
{
    if (cell_id < 0) {
        return nullptr;
    }
    for (const auto& participant : mesh_participants_) {
        const auto begin = participant.cell_offset;
        const auto end = begin + participant.num_cells;
        if (cell_id >= begin && cell_id < end) {
            return &participant;
        }
    }
    return nullptr;
}

const MeshParticipantInfo* FESystem::fieldMeshParticipant(FieldId field) const
{
    const auto& rec = field_registry_.get(field);
    if (rec.participant_name.has_value()) {
        const auto* participant = meshParticipantByName(*rec.participant_name);
        FE_THROW_IF(participant == nullptr, InvalidArgumentException,
                    "FESystem::fieldMeshParticipant: field '" + rec.name +
                        "' references unknown mesh participant '" +
                        *rec.participant_name + "'");
        return participant;
    }
    if (rec.participant_domain_id.has_value()) {
        const auto* participant = meshParticipantByDomain(*rec.participant_domain_id);
        FE_THROW_IF(participant == nullptr, InvalidArgumentException,
                    "FESystem::fieldMeshParticipant: field '" + rec.name +
                        "' references unknown mesh participant domain " +
                        std::to_string(*rec.participant_domain_id));
        return participant;
    }
    return nullptr;
}

bool FESystem::fieldActiveOnCell(FieldId field, GlobalIndex cell_id) const
{
    const auto* participant = fieldMeshParticipant(field);
    if (participant == nullptr) {
        return true;
    }
    const auto begin = participant->cell_offset;
    const auto end = begin + participant->num_cells;
    return cell_id >= begin && cell_id < end;
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
FESystem::MeshVertexFieldProjectionResult
FESystem::projectMeshVertexValuesToFieldCoefficients(
    FieldId field,
    std::span<const Real> mesh_values,
    std::size_t mesh_components,
    std::span<Real> coefficients,
    std::span<std::uint8_t> assigned,
    std::string_view context) const
{
    requireSetup();
    const auto& mesh = singleMesh(context);
    const auto n_vertices = static_cast<std::size_t>(mesh.n_vertices());

    const auto field_idx = static_cast<std::size_t>(field);
    FE_THROW_IF(field < 0 || field_idx >= field_dof_handlers_.size(),
                InvalidArgumentException,
                std::string(context) + ": invalid FieldId");

    const auto& rec = field_registry_.get(field);
    const auto components =
        static_cast<std::size_t>(std::max(1, rec.components));
    FE_THROW_IF(mesh_components < components, InvalidArgumentException,
                std::string(context) +
                    ": mesh field has fewer components than the FE field");
    FE_THROW_IF(mesh_values.size() < n_vertices * mesh_components,
                InvalidArgumentException,
                std::string(context) +
                    ": mesh field value span is smaller than the mesh vertex count");

    const auto n_dofs =
        static_cast<std::size_t>(field_dof_handlers_[field_idx].getNumDofs());
    FE_THROW_IF(coefficients.size() < n_dofs, InvalidArgumentException,
                std::string(context) +
                    ": coefficient span is smaller than the FE field DOF count");
    FE_THROW_IF(!assigned.empty() && assigned.size() < n_dofs,
                InvalidArgumentException,
                std::string(context) +
                    ": assigned span is smaller than the FE field DOF count");

    FE_THROW_IF(!rec.space, InvalidStateException,
                std::string(context) + ": FE field has no function space");
    const auto basis_type = rec.space->element().basis().basis_type();

    if (!assigned.empty()) {
        std::fill(assigned.begin(), assigned.begin() +
                                      static_cast<std::ptrdiff_t>(n_dofs),
                  std::uint8_t{0});
    }

    const auto* entity_map =
        field_dof_handlers_[field_idx].getEntityDofMap();
    FE_THROW_IF(entity_map == nullptr, InvalidStateException,
                std::string(context) +
                    ": FE field must have EntityDofMap metadata");

    if (basis_type != BasisType::Lagrange &&
        basis_type != BasisType::Serendipity) {
        return projectMeshVertexValuesByLocalInterpolation(
            mesh.local_mesh(),
            *rec.space,
            field_dof_handlers_[field_idx],
            mesh_values,
            mesh_components,
            components,
            coefficients,
            assigned,
            context);
    }

    auto result =
        enumerateMeshVertexDofBindings(
            mesh.local_mesh(),
            *entity_map,
            components,
            n_dofs,
            context,
            [&](std::size_t vertex, std::size_t component, GlobalIndex local_dof) {
                const auto dof = static_cast<std::size_t>(local_dof);
                const auto value_index = vertex * mesh_components + component;
                coefficients[dof] =
                    static_cast<Real>(mesh_values[value_index]);
                if (!assigned.empty()) {
                    assigned[dof] = 1u;
                }
            });

    return result;
}
#endif

std::string FESystem::assemblerName() const
{
    if (!assembler_) {
        return {};
    }
    return assembler_->name();
}

std::string FESystem::assemblerSelectionReport() const
{
    return assembler_selection_report_;
}

ISearchAccess::PointLocation FESystem::locatePoint(const std::array<Real, 3>& point,
                                                   GlobalIndex hint_cell) const
{
    if (!search_access_) {
        return {};
    }
    return search_access_->locatePoint(point, hint_cell);
}

std::optional<std::array<Real, 3>> FESystem::evaluateFieldAtPoint(FieldId field,
                                                                  const SystemStateView& state,
                                                                  const std::array<Real, 3>& point,
                                                                  GlobalIndex hint_cell) const
{
    requireSetup();

    const auto loc = locatePoint(point, hint_cell);
    if (!loc.found || loc.cell_id == INVALID_GLOBAL_INDEX) {
        return std::nullopt;
    }

    const auto& rec = field_registry_.get(field);
    FE_CHECK_NOT_NULL(rec.space.get(), "FESystem::evaluateFieldAtPoint: field.space");

    // Reference coordinates (as provided by the search layer).
    spaces::FunctionSpace::Value xi;
    xi[0] = loc.xi[0];
    xi[1] = loc.xi[1];
    xi[2] = loc.xi[2];

    const auto field_idx = static_cast<std::size_t>(field);
    FE_THROW_IF(field < 0 || field_idx >= field_dof_handlers_.size(), InvalidArgumentException,
                "FESystem::evaluateFieldAtPoint: invalid FieldId");

    const auto cell_dofs_local = field_dof_handlers_[field_idx].getDofMap().getCellDofs(loc.cell_id);
    std::vector<Real> coeffs;
    coeffs.reserve(cell_dofs_local.size());

    const bool use_prescribed =
        rec.source_kind == FieldSourceKind::PrescribedData;
    const auto prescribed_coefficients =
        use_prescribed ? prescribedFieldCoefficients(field)
                       : std::span<const Real>{};
    FE_THROW_IF(use_prescribed && prescribed_coefficients.empty(),
                InvalidStateException,
                "FESystem::evaluateFieldAtPoint: prescribed field '" +
                    rec.name + "' has no coefficients");

    std::unique_ptr<assembly::GlobalSystemView> solution_view;
    if (!use_prescribed && state.u_vector != nullptr) {
        auto* vec = const_cast<backends::GenericVector*>(state.u_vector);
        solution_view = vec->createAssemblyView();
    }

    const GlobalIndex offset = field_dof_offsets_[field_idx];
    for (const auto d_local : cell_dofs_local) {
        const GlobalIndex d = use_prescribed ? d_local : d_local + offset;
        FE_THROW_IF(d < 0, InvalidArgumentException,
                    "FESystem::evaluateFieldAtPoint: negative DOF index");
        if (use_prescribed) {
            const auto idx = static_cast<std::size_t>(d);
            FE_THROW_IF(idx >= prescribed_coefficients.size(),
                        InvalidArgumentException,
                        "FESystem::evaluateFieldAtPoint: prescribed field coefficients are smaller than required");
            coeffs.push_back(prescribed_coefficients[idx]);
        } else if (solution_view) {
            coeffs.push_back(solution_view->getVectorEntry(d));
        } else {
            const auto idx = static_cast<std::size_t>(d);
            FE_THROW_IF(idx >= state.u.size(), InvalidArgumentException,
                        "FESystem::evaluateFieldAtPoint: state.u is smaller than required by DOF index");
            coeffs.push_back(state.u[idx]);
        }
    }

    const auto v = rec.space->evaluate(xi, coeffs);
    return std::array<Real, 3>{v[0], v[1], v[2]};
}

bool FESystem::evaluateFieldAtVertices(FieldId field,
                                        const SystemStateView& state,
                                        GlobalIndex n_vertices,
                                        std::span<double> out) const
{
    requireSetup();

    if (n_vertices <= 0) {
        return false;
    }

    const auto field_idx = static_cast<std::size_t>(field);
    FE_THROW_IF(field < 0 || field_idx >= field_dof_handlers_.size(), InvalidArgumentException,
                "FESystem::evaluateFieldAtVertices: invalid FieldId");

    const auto* entity_map = field_dof_handlers_[field_idx].getEntityDofMap();
    if (!entity_map) {
        return false;
    }

    if (entity_map->numVertices() < n_vertices) {
        return false; // Entity map doesn't cover all mesh vertices
    }

    const auto& rec = field_registry_.get(field);
    const auto ncomp = static_cast<std::size_t>(std::max(1, rec.components));
    FE_CHECK_NOT_NULL(rec.space.get(), "FESystem::evaluateFieldAtVertices: field.space");
    const auto basis_type = rec.space->element().basis().basis_type();
    if (basis_type != BasisType::Lagrange &&
        basis_type != BasisType::Serendipity) {
        return false;
    }

    FE_THROW_IF(out.size() < static_cast<std::size_t>(n_vertices) * ncomp, InvalidArgumentException,
                "FESystem::evaluateFieldAtVertices: output buffer too small");

    const bool use_prescribed =
        rec.source_kind == FieldSourceKind::PrescribedData;
    const auto prescribed_coefficients =
        use_prescribed ? prescribedFieldCoefficients(field)
                       : std::span<const Real>{};
    FE_THROW_IF(use_prescribed && prescribed_coefficients.empty(),
                InvalidStateException,
                "FESystem::evaluateFieldAtVertices: prescribed field '" +
                    rec.name + "' has no coefficients");

    const GlobalIndex offset = field_dof_offsets_[field_idx];

    // Create assembly view if backend vector is provided (MPI case)
    std::unique_ptr<assembly::GlobalSystemView> solution_view;
    if (!use_prescribed && state.u_vector != nullptr) {
        auto* vec = const_cast<backends::GenericVector*>(state.u_vector);
        solution_view = vec->createAssemblyView();
    }

    auto read_coefficient = [&](GlobalIndex local_dof) -> double {
        const GlobalIndex d = use_prescribed ? local_dof : local_dof + offset;
        FE_THROW_IF(d < 0, InvalidArgumentException,
                    "FESystem::evaluateFieldAtVertices: negative DOF index");
        if (use_prescribed) {
            const auto idx = static_cast<std::size_t>(d);
            FE_THROW_IF(idx >= prescribed_coefficients.size(),
                        InvalidArgumentException,
                        "FESystem::evaluateFieldAtVertices: prescribed field coefficients are smaller than required");
            return static_cast<double>(prescribed_coefficients[idx]);
        }
        if (solution_view) {
            return static_cast<double>(solution_view->getVectorEntry(d));
        }
        const auto idx = static_cast<std::size_t>(d);
        FE_THROW_IF(idx >= state.u.size(), InvalidArgumentException,
                    "FESystem::evaluateFieldAtVertices: state.u too small");
        return static_cast<double>(state.u[idx]);
    };

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    const auto& mesh = singleMesh("FESystem::evaluateFieldAtVertices");
    if (mesh.n_vertices() < static_cast<std::size_t>(n_vertices)) {
        return false;
    }
    std::vector<std::uint8_t> vertex_component_written(
        static_cast<std::size_t>(n_vertices) * ncomp, 0u);
    const auto projection =
        enumerateMeshVertexDofBindings(
            mesh.local_mesh(),
            *entity_map,
            ncomp,
            static_cast<std::size_t>(field_dof_handlers_[field_idx].getNumDofs()),
            "FESystem::evaluateFieldAtVertices",
            [&](std::size_t vertex, std::size_t component, GlobalIndex local_dof) {
                const auto out_index = vertex * ncomp + component;
                out[out_index] = read_coefficient(local_dof);
                vertex_component_written[out_index] = 1u;
            });
    (void)projection;
    return std::find(vertex_component_written.begin(),
                     vertex_component_written.end(),
                     std::uint8_t{0}) == vertex_component_written.end();
#else
    bool all_mesh_vertices_have_vertex_dofs = true;
    for (GlobalIndex v = 0; v < n_vertices; ++v) {
        if (entity_map->getVertexDofs(v).size() != ncomp) {
            all_mesh_vertices_have_vertex_dofs = false;
            break;
        }
    }
    if (!all_mesh_vertices_have_vertex_dofs) {
        return false;
    }

    for (GlobalIndex v = 0; v < n_vertices; ++v) {
        const auto vdofs = entity_map->getVertexDofs(v);
        const auto out_base = static_cast<std::size_t>(v) * ncomp;
        for (std::size_t c = 0; c < ncomp; ++c) {
            out[out_base + c] = read_coefficient(vdofs[c]);
        }
    }

    return true;
#endif
}

const FieldRecord& FESystem::fieldRecord(FieldId field) const
{
    return field_registry_.get(field);
}

assembly::MaterialStateView FESystem::globalKernelCellState(const GlobalKernel& kernel,
                                                            GlobalIndex cell_id,
                                                            LocalIndex num_qpts) const
{
    requireSetup();
    if (!global_kernel_state_provider_) return {};
    return global_kernel_state_provider_->getCellState(kernel, cell_id, num_qpts);
}

assembly::MaterialStateView FESystem::globalKernelBoundaryFaceState(const GlobalKernel& kernel,
                                                                    GlobalIndex face_id,
                                                                    LocalIndex num_qpts) const
{
    requireSetup();
    if (!global_kernel_state_provider_) return {};
    return global_kernel_state_provider_->getBoundaryFaceState(kernel, face_id, num_qpts);
}

assembly::MaterialStateView FESystem::globalKernelInteriorFaceState(const GlobalKernel& kernel,
                                                                    GlobalIndex face_id,
                                                                    LocalIndex num_qpts) const
{
    requireSetup();
    if (!global_kernel_state_provider_) return {};
    return global_kernel_state_provider_->getInteriorFaceState(kernel, face_id, num_qpts);
}

const sparsity::SparsityPattern& FESystem::sparsity(const OperatorTag& op) const
{
    requireSetup();
    auto it = sparsity_by_op_.find(op);
    FE_THROW_IF(it == sparsity_by_op_.end() || !it->second, InvalidArgumentException,
                "FESystem::sparsity: no sparsity pattern for operator '" + op + "'");
    return *it->second;
}

const sparsity::DistributedSparsityPattern*
FESystem::distributedSparsityIfAvailable(const OperatorTag& op) const noexcept
{
    if (!is_setup_) {
        return nullptr;
    }
    auto it = distributed_sparsity_by_op_.find(op);
    if (it == distributed_sparsity_by_op_.end()) {
        return nullptr;
    }
    return it->second.get();
}

int FESystem::temporalOrder() const noexcept
{
    int max_order = 0;
    for (const auto& tag : operator_registry_.list()) {
        const auto& def = operator_registry_.get(tag);
        for (const auto& term : def.cells) {
            if (term.kernel) max_order = std::max(max_order, term.kernel->maxTemporalDerivativeOrder());
        }
        for (const auto& term : def.boundary) {
            if (term.kernel) max_order = std::max(max_order, term.kernel->maxTemporalDerivativeOrder());
        }
        for (const auto& term : def.interior) {
            if (term.kernel) max_order = std::max(max_order, term.kernel->maxTemporalDerivativeOrder());
        }
        for (const auto& term : def.interface_faces) {
            if (term.kernel) max_order = std::max(max_order, term.kernel->maxTemporalDerivativeOrder());
        }
    }
    return max_order;
}

bool FESystem::hasExplicitTimeDependency() const noexcept
{
    for (const auto& tag : operator_registry_.list()) {
        const auto& def = operator_registry_.get(tag);
        for (const auto& term : def.cells) {
            if (term.kernel && term.kernel->hasExplicitTimeDependency()) return true;
        }
        for (const auto& term : def.boundary) {
            if (term.kernel && term.kernel->hasExplicitTimeDependency()) return true;
        }
        for (const auto& term : def.interior) {
            if (term.kernel && term.kernel->hasExplicitTimeDependency()) return true;
        }
        for (const auto& term : def.interface_faces) {
            if (term.kernel && term.kernel->hasExplicitTimeDependency()) return true;
        }
    }
    return false;
}

bool FESystem::hasTimeDependentConstraints() const noexcept
{
    for (const auto& c : constraint_defs_) {
        if (c && c->isTimeDependent()) {
            return true;
        }
    }
    for (const auto& c : system_constraint_defs_) {
        if (c && c->isTimeDependent()) {
            return true;
        }
    }
    return false;
}

bool FESystem::requiresTimeAdvancement() const noexcept
{
    return temporalOrder() > 0 || hasExplicitTimeDependency() || hasTimeDependentConstraints();
}

namespace {

void gatherTimeDerivativeFieldsFromNode(const forms::FormExprNode& node,
                                        FieldId kernel_trial_field,
                                        std::unordered_set<FieldId>& out)
{
    if (node.type() == forms::FormExprType::TimeDerivative) {
        const auto children = node.childrenShared();
        if (!children.empty() && children.front()) {
            const auto& child = *children.front();
            if (child.type() == forms::FormExprType::TrialFunction) {
                if (kernel_trial_field != INVALID_FIELD_ID) {
                    out.insert(kernel_trial_field);
                }
            } else if (child.type() == forms::FormExprType::StateField ||
                       child.type() == forms::FormExprType::DiscreteField) {
                if (const auto fid = child.fieldId()) {
                    out.insert(*fid);
                }
            }
        }
    }

    for (const auto& child : node.childrenShared()) {
        if (child) {
            gatherTimeDerivativeFieldsFromNode(*child, kernel_trial_field, out);
        }
    }
}

void gatherTimeDerivativeFieldsFromIR(const forms::FormIR& ir,
                                      FieldId kernel_trial_field,
                                      std::unordered_set<FieldId>& out)
{
    for (const auto& term : ir.terms()) {
        const auto* root = term.integrand.node();
        if (!root) {
            continue;
        }
        gatherTimeDerivativeFieldsFromNode(*root, kernel_trial_field, out);
    }
}

void gatherTimeDerivativeFieldsFromKernel(const assembly::AssemblyKernel* kernel,
                                          FieldId kernel_trial_field,
                                          std::unordered_set<FieldId>& out)
{
    if (!kernel) {
        return;
    }

    if (const auto* k = dynamic_cast<const forms::jit::JITKernelWrapper*>(kernel)) {
        gatherTimeDerivativeFieldsFromKernel(&k->fallbackKernel(), kernel_trial_field, out);
        return;
    }

    if (const auto* k = dynamic_cast<const forms::SymbolicNonlinearFormKernel*>(kernel)) {
        gatherTimeDerivativeFieldsFromIR(k->residualIR(), kernel_trial_field, out);
        gatherTimeDerivativeFieldsFromIR(k->tangentIR(), kernel_trial_field, out);
        return;
    }

    if (const auto* k = dynamic_cast<const forms::NonlinearFormKernel*>(kernel)) {
        gatherTimeDerivativeFieldsFromIR(k->residualIR(), kernel_trial_field, out);
        return;
    }

    if (const auto* k = dynamic_cast<const forms::FormKernel*>(kernel)) {
        gatherTimeDerivativeFieldsFromIR(k->ir(), kernel_trial_field, out);
        return;
    }

    if (const auto* k = dynamic_cast<const forms::LinearFormKernel*>(kernel)) {
        gatherTimeDerivativeFieldsFromIR(k->bilinearIR(), kernel_trial_field, out);
        if (k->linearIR().has_value()) {
            gatherTimeDerivativeFieldsFromIR(*k->linearIR(), kernel_trial_field, out);
        }
        return;
    }
}

std::vector<FieldId> sortedUnique(std::unordered_set<FieldId> ids)
{
    std::vector<FieldId> out;
    out.reserve(ids.size());
    for (const auto fid : ids) {
        out.push_back(fid);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

} // namespace

std::vector<FieldId> FESystem::timeDerivativeFields(const OperatorTag& op) const
{
    std::unordered_set<FieldId> fields;
    const auto& def = operator_registry_.get(op);

    for (const auto& term : def.cells) {
        gatherTimeDerivativeFieldsFromKernel(term.kernel.get(), term.trial_field, fields);
    }
    for (const auto& term : def.boundary) {
        gatherTimeDerivativeFieldsFromKernel(term.kernel.get(), term.trial_field, fields);
    }
    for (const auto& term : def.interior) {
        gatherTimeDerivativeFieldsFromKernel(term.kernel.get(), term.trial_field, fields);
    }
    for (const auto& term : def.interface_faces) {
        gatherTimeDerivativeFieldsFromKernel(term.kernel.get(), term.trial_field, fields);
    }
    for (const auto& term : def.cut_volumes) {
        gatherTimeDerivativeFieldsFromKernel(term.kernel.get(), term.trial_field, fields);
    }

    return sortedUnique(std::move(fields));
}

std::vector<FieldId> FESystem::timeDerivativeFields() const
{
    std::unordered_set<FieldId> fields;
    for (const auto& op : operator_registry_.list()) {
        const auto& def = operator_registry_.get(op);
        for (const auto& term : def.cells) {
            gatherTimeDerivativeFieldsFromKernel(term.kernel.get(), term.trial_field, fields);
        }
        for (const auto& term : def.boundary) {
            gatherTimeDerivativeFieldsFromKernel(term.kernel.get(), term.trial_field, fields);
        }
        for (const auto& term : def.interior) {
            gatherTimeDerivativeFieldsFromKernel(term.kernel.get(), term.trial_field, fields);
        }
        for (const auto& term : def.interface_faces) {
            gatherTimeDerivativeFieldsFromKernel(term.kernel.get(), term.trial_field, fields);
        }
        for (const auto& term : def.cut_volumes) {
            gatherTimeDerivativeFieldsFromKernel(term.kernel.get(), term.trial_field, fields);
        }
    }
    return sortedUnique(std::move(fields));
}

const dofs::DofHandler& FESystem::fieldDofHandler(FieldId field) const
{
    const auto idx = static_cast<std::size_t>(field);
    FE_THROW_IF(field < 0 || idx >= field_dof_handlers_.size(), InvalidArgumentException,
                "FESystem::fieldDofHandler: invalid field id");
    FE_THROW_IF(!field_dof_handlers_[idx].isFinalized(), InvalidStateException,
                "FESystem::fieldDofHandler: field DOFs not finalized");
    return field_dof_handlers_[idx];
}

GlobalIndex FESystem::fieldDofOffset(FieldId field) const
{
    const auto idx = static_cast<std::size_t>(field);
    FE_THROW_IF(field < 0 || idx >= field_dof_offsets_.size(), InvalidArgumentException,
                "FESystem::fieldDofOffset: invalid field id");
    return field_dof_offsets_[idx];
}

bool FESystem::operatorMatrixStateIndependent(const OperatorTag& op) const
{
    requireSetup();
    const auto it = assembly_plan_by_op_.find(op);
    FE_THROW_IF(it == assembly_plan_by_op_.end(), InvalidArgumentException,
                "FESystem::operatorMatrixStateIndependent: unknown operator '" + op + "'");
    return it->second.matrix_state_independent;
}

assembly::AssemblyResult FESystem::assemble(
    const AssemblyRequest& req,
    const SystemStateView& state,
    assembly::GlobalSystemView* matrix_out,
    assembly::GlobalSystemView* vector_out)
{
    return assembleOperator(*this, req, state, matrix_out, vector_out);
}

assembly::AssemblyResult FESystem::assembleResidual(
    const SystemStateView& state,
    assembly::GlobalSystemView& rhs_out)
{
    AssemblyRequest req;
    req.op = "residual";
    req.want_vector = true;
    return assemble(req, state, nullptr, &rhs_out);
}

assembly::AssemblyResult FESystem::assembleJacobian(
    const SystemStateView& state,
    assembly::GlobalSystemView& jac_out)
{
    AssemblyRequest req;
    req.op = "jacobian";
    req.want_matrix = true;
    return assemble(req, state, &jac_out, nullptr);
}

assembly::AssemblyResult FESystem::assembleMass(
    const SystemStateView& state,
    assembly::GlobalSystemView& mass_out)
{
    AssemblyRequest req;
    req.op = "mass";
    req.want_matrix = true;
    return assemble(req, state, &mass_out, nullptr);
}

void FESystem::beginTimeStep(bool reset_auxiliary_state,
                             bool invalidate_auxiliary_inputs)
{
    // requireSetup() is skipped for auxiliary-only use (no mesh/fields).
    // Material/global-kernel providers are null when not set up.
    if (material_state_provider_) {
        material_state_provider_->beginTimeStep();
    }
    if (global_kernel_state_provider_) {
        global_kernel_state_provider_->beginTimeStep();
    }
    // Reset generalized auxiliary state to committed values.
    if (reset_auxiliary_state && auxiliary_state_manager_) {
        auxiliary_state_manager_->resetAllToCommitted();
    }
    // Invalidate all auxiliary inputs for the new time step.
    if (invalidate_auxiliary_inputs && auxiliary_input_registry_) {
        auxiliary_input_registry_->invalidateAll();
    }
    if (reset_auxiliary_state || invalidate_auxiliary_inputs) {
        partitioned_auxiliary_advance_valid_ = false;
        partitioned_auxiliary_advance_time_ = std::numeric_limits<Real>::quiet_NaN();
        partitioned_auxiliary_advance_dt_ = std::numeric_limits<Real>::quiet_NaN();
    }
}

void FESystem::commitTimeStep()
{
    if (material_state_provider_) {
        material_state_provider_->commitTimeStep();
    }
    if (global_kernel_state_provider_) {
        global_kernel_state_provider_->commitTimeStep();
    }
    // Commit generalized auxiliary state with the last-known time.
    if (auxiliary_state_manager_) {
        auxiliary_state_manager_->commitAll(last_auxiliary_advance_time_);
    }
}

void FESystem::finalizeMonolithicAuxiliaryStageState(Real alpha_f, Real final_time)
{
    finalizeMonolithicAuxiliaryStageState(alpha_f, Real(-1.0), Real(-1.0), final_time);
}

void FESystem::finalizeMonolithicAuxiliaryStageState(
    Real alpha_f,
    Real gamma,
    Real dt,
    Real final_time)
{
    FE_THROW_IF(!(alpha_f > 0.0) || !std::isfinite(alpha_f), InvalidArgumentException,
                "FESystem::finalizeMonolithicAuxiliaryStageState: alpha_f must be finite and > 0");

    const Real previous_auxiliary_time = last_auxiliary_advance_time_;
    last_auxiliary_advance_time_ = final_time;

    if (!auxiliary_state_manager_) {
        return;
    }

    const bool transform_stage =
        std::abs(alpha_f - Real(1.0)) > Real(1e-14);
    if (transform_stage) {
        const Real inv_alpha_f = Real(1.0) / alpha_f;
        const Real c_prev = (alpha_f - Real(1.0)) * inv_alpha_f;

        for (auto& entry : deployed_aux_entries_) {
            if (!entry.materialized) {
                continue;
            }
            if (entry.spec.solve_mode != AuxiliarySolveMode::Monolithic) {
                continue;
            }
            if (!auxiliary_state_manager_->hasBlock(entry.instance_name)) {
                continue;
            }

            auto& blk = auxiliary_state_manager_->getBlock(entry.instance_name);
            const auto meta = entry.model->structuralMetadata();
            const auto& kinds = meta.variable_kinds;

            for (std::size_t e = 0; e < blk.entityCount(); ++e) {
                auto stage_state = blk.gatherEntityWork(e);
                const auto committed = blk.gatherEntityCommitted(e);
                bool changed = false;
                for (std::size_t i = 0; i < stage_state.size() && i < committed.size(); ++i) {
                    const bool differential =
                        i < kinds.size()
                            ? (kinds[i] == AuxiliaryVariableKind::Differential)
                            : true;
                    if (!differential) {
                        continue;
                    }
                    stage_state[i] = inv_alpha_f * stage_state[i] + c_prev * committed[i];
                    changed = true;
                }
                if (changed) {
                    blk.scatterEntityWork(e, stage_state);
                }
            }
        }
    }

    Real event_dt = dt;
    if (!(event_dt > Real(0.0)) ||
        !std::isfinite(static_cast<double>(event_dt))) {
        event_dt = final_time - previous_auxiliary_time;
    }
    if (event_dt > Real(0.0) &&
        std::isfinite(static_cast<double>(event_dt))) {
        applyMonolithicAcceptedStepEvents_(final_time - event_dt, event_dt, gamma);
    }

    updateMonolithicFinalRates_(gamma, dt);
    auxiliary_state_manager_->syncGhosts();
}

void FESystem::applyMonolithicAcceptedStepEvents_(Real step_start_time,
                                                  Real dt,
                                                  Real gamma)
{
    if (!auxiliary_state_manager_ ||
        !(dt > Real(0.0)) ||
        !std::isfinite(static_cast<double>(dt))) {
        return;
    }

    const bool rates_will_refresh =
        gamma > Real(0.0) && std::isfinite(static_cast<double>(gamma));

    for (auto& entry : deployed_aux_entries_) {
        if (!entry.materialized ||
            entry.spec.solve_mode != AuxiliarySolveMode::Monolithic ||
            entry.event_managers.empty() ||
            !entry.model ||
            !entry.model->hasEventFunctions() ||
            !auxiliary_state_manager_->hasBlock(entry.instance_name)) {
            continue;
        }

        auto& blk = auxiliary_state_manager_->getBlock(entry.instance_name);
        auto params = buildParamVector(entry);
        auto bound_inputs = buildInputVector(entry);

        bool has_entity_local_inputs = false;
        if (auxiliary_input_registry_) {
            for (const auto& [model_name, reg_name] : entry.input_bindings) {
                (void)model_name;
                if (auxiliary_input_registry_->hasInput(reg_name) &&
                    auxiliary_input_registry_->isEntityLocal(reg_name)) {
                    has_entity_local_inputs = true;
                    break;
                }
            }
        }

        const auto& emap = entry.entity_map;
        const auto n_authoritative_entities =
            std::min(blk.ownedEntityCount(), entry.event_managers.size());
        bool block_changed = false;

        for (std::size_t e = 0; e < n_authoritative_entities; ++e) {
            if (!entry.event_managers[e]) {
                continue;
            }

            const auto committed = blk.gatherEntityCommitted(e);
            auto final_state = blk.gatherEntityWork(e);
            const auto orig_e = emap.empty() ? e : emap[e];

            if (has_entity_local_inputs && auxiliary_input_registry_) {
                bound_inputs.clear();
                if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(entry.model.get())) {
                    for (const auto& inp : built->signature().inputs) {
                        auto bind_it = entry.input_bindings.find(inp.name);
                        if (bind_it != entry.input_bindings.end()) {
                            auto vals = auxiliary_input_registry_->valuesOf(bind_it->second, orig_e);
                            bound_inputs.insert(bound_inputs.end(), vals.begin(), vals.end());
                        } else {
                            bound_inputs.resize(
                                bound_inputs.size() + static_cast<std::size_t>(inp.size),
                                Real(0.0));
                        }
                    }
                } else {
                    rebuildGenericInputsForEntity(entry, orig_e, bound_inputs);
                }
            }

            std::vector<std::vector<Real>> history_storage;
            auto history_spans = buildHistorySpans_(blk, e, history_storage);
            auto& event_manager = *entry.event_managers[e];
            const auto events = event_manager.detectEvents(
                *entry.model,
                committed,
                final_state,
                step_start_time,
                dt,
                bound_inputs,
                params,
                history_spans,
                e);
            if (events.empty()) {
                continue;
            }

            for (const auto& event : events) {
                event_manager.applyTransition(
                    *entry.model,
                    event,
                    final_state,
                    event.event_time,
                    bound_inputs,
                    params,
                    history_spans,
                    e,
                    dt);
            }

            blk.scatterEntityWork(e, final_state);
            block_changed = true;
        }

        if (block_changed && blk.ownedEntityCount() < blk.entityCount()) {
            auxiliary_state_manager_->syncGhosts(entry.instance_name);
        }
        if (block_changed && !rates_will_refresh) {
            monolithic_aux_committed_rates_valid_.erase(entry.instance_name);
        }
    }
}

void FESystem::updateMonolithicFinalRates_(Real gamma, Real dt)
{
    if (!auxiliary_state_manager_ ||
        !(gamma > Real(0.0)) ||
        !(dt > Real(0.0)) ||
        !std::isfinite(static_cast<double>(gamma)) ||
        !std::isfinite(static_cast<double>(dt))) {
        return;
    }

    const Real inv_gamma_dt = Real(1.0) / (gamma * dt);
    const Real c_old = (Real(1.0) - gamma) / gamma;

    for (const auto& entry : deployed_aux_entries_) {
        if (!entry.materialized ||
            entry.spec.solve_mode != AuxiliarySolveMode::Monolithic ||
            !entry.model ||
            !auxiliary_state_manager_->hasBlock(entry.instance_name)) {
            continue;
        }

        const auto& blk = auxiliary_state_manager_->getBlock(entry.instance_name);
        const auto meta = entry.model->structuralMetadata();
        const auto& kinds = meta.variable_kinds;
        bool updated = false;

        for (std::size_t e = 0; e < blk.entityCount(); ++e) {
            const auto final_state = blk.gatherEntityWork(e);
            const auto committed = blk.gatherEntityCommitted(e);
            const auto committed_rate = gatherMonolithicCommittedRate(entry, e);
            if (committed_rate.size() != final_state.size()) {
                continue;
            }

            std::vector<Real> final_rate(final_state.size(), Real(0.0));
            for (std::size_t i = 0; i < final_state.size() && i < committed.size(); ++i) {
                const bool differential =
                    i < kinds.size()
                        ? (kinds[i] == AuxiliaryVariableKind::Differential)
                        : true;
                if (!differential) {
                    continue;
                }
                final_rate[i] =
                    inv_gamma_dt * (final_state[i] - committed[i]) -
                    c_old * committed_rate[i];
            }
            scatterMonolithicCommittedRate(entry, e, final_rate);
            updated = true;
        }

        if (updated) {
            monolithic_aux_committed_rates_valid_.insert(entry.instance_name);
        }
    }
}

void FESystem::ensureMonolithicCommittedRateBuffer(
    const DeployedAuxEntry& entry,
    std::size_t storage_size)
{
    auto& buf = monolithic_aux_committed_rates_[entry.instance_name];
    if (buf.size() != storage_size) {
        buf.assign(storage_size, Real(0.0));
        monolithic_aux_committed_rates_valid_.erase(entry.instance_name);
    }
}

std::vector<Real> FESystem::gatherMonolithicCommittedRate(
    const DeployedAuxEntry& entry,
    std::size_t entity_index) const
{
    auto it = monolithic_aux_committed_rates_.find(entry.instance_name);
    if (it == monolithic_aux_committed_rates_.end() ||
        monolithic_aux_committed_rates_valid_.count(entry.instance_name) == 0u ||
        !auxiliary_state_manager_ ||
        !auxiliary_state_manager_->hasBlock(entry.instance_name)) {
        return {};
    }
    const auto& blk = auxiliary_state_manager_->getBlock(entry.instance_name);
    return gatherAuxiliaryFlatEntity(blk, it->second, entity_index);
}

void FESystem::scatterMonolithicCommittedRate(
    const DeployedAuxEntry& entry,
    std::size_t entity_index,
    std::span<const Real> values)
{
    if (!auxiliary_state_manager_ || !auxiliary_state_manager_->hasBlock(entry.instance_name)) {
        return;
    }
    auto& blk = auxiliary_state_manager_->getBlock(entry.instance_name);
    ensureMonolithicCommittedRateBuffer(entry, blk.storageSize());
    auto it = monolithic_aux_committed_rates_.find(entry.instance_name);
    FE_THROW_IF(it == monolithic_aux_committed_rates_.end(), InvalidStateException,
                "FESystem::scatterMonolithicCommittedRate: missing rate buffer");
    std::span<Real> flat{it->second.data(), it->second.size()};
    scatterAuxiliaryFlatEntity(blk, flat, entity_index, values);
}

void FESystem::initializeMonolithicCommittedRate(
    const DeployedAuxEntry& entry,
    const SystemStateView& prev_state)
{
    if (!auxiliary_state_manager_ ||
        !auxiliary_state_manager_->hasBlock(entry.instance_name) ||
        !entry.deriv_provider) {
        return;
    }

    auto& blk = auxiliary_state_manager_->getBlock(entry.instance_name);
    ensureMonolithicCommittedRateBuffer(entry, blk.storageSize());

    const auto meta = entry.model->structuralMetadata();
    const auto& kinds = meta.variable_kinds;
    auto params = buildParamVector(entry);
    auto bound_inputs = buildInputVector(entry);
    const auto& emap = entry.entity_map;

    bool has_entity_local_inputs = false;
    if (auxiliary_input_registry_) {
        for (const auto& [model_name, reg_name] : entry.input_bindings) {
            if (auxiliary_input_registry_->hasInput(reg_name) &&
                auxiliary_input_registry_->isEntityLocal(reg_name)) {
                has_entity_local_inputs = true;
                break;
            }
        }
    }

    for (std::size_t e = 0; e < blk.entityCount(); ++e) {
        const auto entity_x = blk.gatherEntityCommitted(e);
        const auto orig_e = emap.empty() ? e : emap[e];

        if (has_entity_local_inputs && auxiliary_input_registry_) {
            bound_inputs.clear();
            if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(entry.model.get())) {
                for (const auto& inp : built->signature().inputs) {
                    auto bind_it = entry.input_bindings.find(inp.name);
                    if (bind_it != entry.input_bindings.end()) {
                        auto vals = auxiliary_input_registry_->valuesOf(bind_it->second, orig_e);
                        bound_inputs.insert(bound_inputs.end(), vals.begin(), vals.end());
                    } else {
                        bound_inputs.resize(
                            bound_inputs.size() + static_cast<std::size_t>(inp.size),
                            Real(0.0));
                    }
                }
            } else {
                rebuildGenericInputsForEntity(entry, orig_e, bound_inputs);
            }
        }

        std::vector<std::vector<Real>> history_storage;
        std::vector<std::span<const Real>> history_spans;
        history_storage.reserve(blk.history().depth());
        history_spans.reserve(blk.history().depth());
        for (std::size_t k = 0; k < blk.history().depth(); ++k) {
            history_storage.push_back(blk.gatherEntityHistory(k, e));
            history_spans.emplace_back(history_storage.back().data(), history_storage.back().size());
        }

        if (!history_spans.empty()) {
            auto reconstructed = reconstructRateFromHistory(
                entity_x,
                history_spans,
                prev_state.dt_prev,
                prev_state.dt,
                prev_state.dt_history);
            for (std::size_t i = 0; i < reconstructed.size(); ++i) {
                const bool differential =
                    i < kinds.size()
                        ? (kinds[i] == AuxiliaryVariableKind::Differential)
                        : true;
                if (!differential) {
                    reconstructed[i] = Real(0.0);
                }
            }
            scatterMonolithicCommittedRate(entry, e, reconstructed);
            continue;
        }

        std::vector<FieldValueEntry> field_vals;
        const auto& art = entry.deriv_provider->artifact();
        if (!art.referenced_fields.empty()) {
            field_vals.reserve(art.referenced_fields.size());
            for (const auto fid : art.referenced_fields) {
                const auto fidx = static_cast<std::size_t>(fid);
                if (fidx >= field_dof_offsets_.size() ||
                    fidx >= field_dof_handlers_.size()) {
                    continue;
                }
                const auto fld_off = field_dof_offsets_[fidx];
                const auto* femap = field_dof_handlers_[fidx].getEntityDofMap();
                if (!femap) {
                    continue;
                }
                auto vdofs = femap->getVertexDofs(static_cast<GlobalIndex>(orig_e));
                if (vdofs.empty()) {
                    continue;
                }
                FieldValueEntry fve;
                fve.field = fid;
                fve.n_components = static_cast<int>(vdofs.size());
                for (int c = 0; c < fve.n_components && c < MAX_FIELD_VALUE_COMPONENTS; ++c) {
                    const auto gidx = static_cast<std::size_t>(
                        vdofs[static_cast<std::size_t>(c)] + fld_off);
                    fve.components[c] = (gidx < prev_state.u.size()) ? prev_state.u[gidx] : Real(0.0);
                }
                field_vals.push_back(fve);
            }
        }

        std::vector<Real> zero_xdot(entity_x.size(), Real(0.0));
        std::vector<Real> residual(entity_x.size(), Real(0.0));
        std::vector<Real> dF_dxdot(entity_x.size() * entity_x.size(), Real(0.0));

        AuxiliaryLocalContext ctx;
        ctx.time = static_cast<Real>(prev_state.time);
        ctx.dt = static_cast<Real>(prev_state.dt);
        ctx.effective_dt = effectiveAuxiliaryDt(prev_state);
        ctx.x = entity_x;
        ctx.xdot = zero_xdot;
        ctx.history = history_spans;
        ctx.inputs = bound_inputs;
        ctx.params = params;
        ctx.entity_index = e;
        ctx.field_values = field_vals;
        ctx.user_data = prev_state.user_data;

        AuxiliaryResidualRequest res_req;
        res_req.residual = residual;
        entry.model->evaluateResidual(ctx, res_req);

        AuxiliaryJacobianRequest jac_req;
        jac_req.n = static_cast<int>(entity_x.size());
        jac_req.want_dF_dxdot = true;
        jac_req.dF_dxdot = dF_dxdot;
        entry.deriv_provider->evaluateJacobian(*entry.model, ctx, jac_req);

        std::vector<int> diff_idx;
        diff_idx.reserve(entity_x.size());
        for (std::size_t i = 0; i < entity_x.size(); ++i) {
            const bool differential =
                i < kinds.size()
                    ? (kinds[i] == AuxiliaryVariableKind::Differential)
                    : true;
            if (differential) {
                diff_idx.push_back(static_cast<int>(i));
            }
        }

        std::vector<Real> entity_rate(entity_x.size(), Real(0.0));
        if (!diff_idx.empty()) {
            const auto n_diff = diff_idx.size();
            std::vector<Real> M(n_diff * n_diff, Real(0.0));
            std::vector<Real> rhs(n_diff, Real(0.0));
            for (std::size_t ri = 0; ri < n_diff; ++ri) {
                rhs[ri] = -residual[static_cast<std::size_t>(diff_idx[ri])];
                for (std::size_t ci = 0; ci < n_diff; ++ci) {
                    M[ri * n_diff + ci] =
                        dF_dxdot[static_cast<std::size_t>(diff_idx[ri]) * entity_x.size() +
                                 static_cast<std::size_t>(diff_idx[ci])];
                }
            }

            if (solveDenseSystemInPlace(M, rhs)) {
                for (std::size_t i = 0; i < n_diff; ++i) {
                    entity_rate[static_cast<std::size_t>(diff_idx[i])] = rhs[i];
                }
            }
        }

        scatterMonolithicCommittedRate(entry, e, entity_rate);
    }
    monolithic_aux_committed_rates_valid_.insert(entry.instance_name);
}

void FESystem::ensureMonolithicCommittedRates(const SystemStateView& state)
{
    const auto* ti = state.time_integration ? state.time_integration->stencil(1) : nullptr;
    if (state.time_integration == nullptr ||
        state.time_integration->integrator_name != "GeneralizedAlpha(1stOrder)" ||
        ti == nullptr || ti->a.size() != 3u ||
        !auxiliary_state_manager_) {
        return;
    }

    bool needs_seed = false;
    for (const auto& entry : deployed_aux_entries_) {
        if (!entry.materialized) {
            continue;
        }
        if (entry.spec.solve_mode != AuxiliarySolveMode::Monolithic) {
            continue;
        }
        if (monolithic_aux_committed_rates_valid_.count(entry.instance_name) == 0u) {
            needs_seed = true;
            break;
        }
    }
    if (!needs_seed) {
        return;
    }

    auto inputRequiresPreviousFeState = [&](std::string_view registry_name) {
        if (!auxiliary_input_registry_ || !auxiliary_input_registry_->hasInput(registry_name)) {
            return false;
        }
        const auto& spec = auxiliary_input_registry_->specOf(registry_name);
        switch (spec.producer) {
            case AuxiliaryInputProducer::SampledStateField:
            case AuxiliaryInputProducer::CoupledField:
            case AuxiliaryInputProducer::CellAverage:
            case AuxiliaryInputProducer::CellSample:
            case AuxiliaryInputProducer::DomainAverage:
            case AuxiliaryInputProducer::DomainIntegral:
            case AuxiliaryInputProducer::RegionAverage:
            case AuxiliaryInputProducer::RegionIntegral:
            case AuxiliaryInputProducer::SampledBoundaryTrace:
            case AuxiliaryInputProducer::CoupledBoundaryTrace:
            case AuxiliaryInputProducer::SampledBoundaryReduction:
            case AuxiliaryInputProducer::CoupledBoundaryReduction:
                return true;
            default:
                return false;
        }
    };

    bool requires_prev_fe_state = false;
    for (const auto& entry : deployed_aux_entries_) {
        if (!entry.materialized) {
            continue;
        }
        if (entry.spec.solve_mode != AuxiliarySolveMode::Monolithic ||
            monolithic_aux_committed_rates_valid_.count(entry.instance_name) != 0u) {
            continue;
        }

        if (auxiliary_state_manager_ &&
            auxiliary_state_manager_->hasBlock(entry.instance_name) &&
            auxiliary_state_manager_->getBlock(entry.instance_name).history().depth() > 0u) {
            continue;
        }

        if (entry.deriv_provider &&
            !entry.deriv_provider->artifact().referenced_fields.empty()) {
            requires_prev_fe_state = true;
            break;
        }

        for (const auto& binding : entry.input_bindings) {
            if (inputRequiresPreviousFeState(binding.second)) {
                requires_prev_fe_state = true;
                break;
            }
        }
        if (requires_prev_fe_state) {
            break;
        }
    }

    FE_THROW_IF(requires_prev_fe_state && state.u_prev.empty(),
                InvalidStateException,
                "FESystem::ensureMonolithicCommittedRates: generalized-alpha requires previous FE state");

    SystemStateView prev_state = state;
    prev_state.time = state.time - effectiveAuxiliaryDt(state);
    prev_state.effective_dt = state.dt;
    prev_state.u = state.u_prev;
    prev_state.u_vector = state.u_prev_vector;
    prev_state.u_prev = state.u_prev2;
    prev_state.u_prev_vector = state.u_prev2_vector;
    prev_state.u_prev2 = {};
    prev_state.u_prev2_vector = nullptr;
    prev_state.u_history = {};
    prev_state.dt_history = {};
    prev_state.time_integration = nullptr;

    cacheSystemState(prev_state);
    if (auxiliary_input_registry_) {
        auxiliary_input_registry_->evaluate(
            static_cast<Real>(prev_state.time),
            static_cast<Real>(prev_state.dt),
            /*is_nonlinear_iteration=*/true);
    }

    for (const auto& entry : deployed_aux_entries_) {
        if (entry.spec.solve_mode != AuxiliarySolveMode::Monolithic ||
            monolithic_aux_committed_rates_valid_.count(entry.instance_name) != 0u) {
            continue;
        }
        initializeMonolithicCommittedRate(entry, prev_state);
    }

    if (auxiliary_input_registry_) {
        auxiliary_input_registry_->invalidateAll();
    }
}

// ---------------------------------------------------------------------------
//  System state cache for auxiliary input callbacks
// ---------------------------------------------------------------------------

void FESystem::cacheSystemState(const SystemStateView& state) const
{
    cached_solution_u_ = state.u;
    cached_solution_vector_ = state.u_vector;
    cached_solution_u_prev_ = state.u_prev;
    cached_solution_prev_vector_ = state.u_prev_vector;
    cached_solution_u_prev2_ = state.u_prev2;
    cached_solution_prev2_vector_ = state.u_prev2_vector;
    cached_time_integration_ = state.time_integration;
    cached_user_data_ = state.user_data;
}

// ---------------------------------------------------------------------------
//  Auxiliary lifecycle
// ---------------------------------------------------------------------------

void FESystem::prepareAuxiliaryForAssembly(const SystemStateView& state,
                                            bool is_nonlinear_iteration)
{
    // Resolve any deferred derived-input expressions and dependency edges
    // that were registered via derivedInput().  This runs at most once —
    // after finalization, both vectors are empty.
    finalizeDeferredInputDeps();

    // Cache the full system state for FE-coupled input callbacks.
    cacheSystemState(state);

    // Evaluate auxiliary input providers.
    if (auxiliary_input_registry_) {
        auxiliary_input_registry_->evaluate(state.time, state.dt, is_nonlinear_iteration);
    }

    initializeAuxiliaryDAEBlocksIfNeeded_(state.time, state.dt);
    ensureMonolithicCommittedRates(state);

    const Real aux_dt = effectiveAuxiliaryDt(state);

    auto hasEntityLocalInputs = [&](const DeployedAuxEntry& entry) {
        if (!auxiliary_input_registry_) {
            return false;
        }
        for (const auto& [model_name, reg_name] : entry.input_bindings) {
            (void)model_name;
            if (auxiliary_input_registry_->hasInput(reg_name) &&
                auxiliary_input_registry_->isEntityLocal(reg_name)) {
                return true;
            }
        }
        return false;
    };

    auto rebuildEntityInputs = [&](const DeployedAuxEntry& entry,
                                   std::size_t orig_e,
                                   std::vector<Real>& bound_inputs) {
        if (!auxiliary_input_registry_) {
            return;
        }
        bound_inputs.clear();
        if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(entry.model.get())) {
            for (const auto& inp : built->signature().inputs) {
                auto bind_it = entry.input_bindings.find(inp.name);
                if (bind_it != entry.input_bindings.end()) {
                    auto vals = auxiliary_input_registry_->valuesOf(bind_it->second, orig_e);
                    bound_inputs.insert(bound_inputs.end(), vals.begin(), vals.end());
                } else {
                    bound_inputs.resize(
                        bound_inputs.size() + static_cast<std::size_t>(inp.size),
                        Real(0.0));
                }
            }
            return;
        }
        rebuildGenericInputsForEntity(entry, orig_e, bound_inputs);
    };

    auto buildEntityFieldValues = [&](const DeployedAuxEntry& entry,
                                      std::size_t orig_e) {
        std::vector<FieldValueEntry> field_values;
        if (!entry.deriv_provider) {
            return field_values;
        }
        const auto& artifact = entry.deriv_provider->artifact();
        if (artifact.referenced_fields.empty()) {
            return field_values;
        }
        field_values.reserve(artifact.referenced_fields.size());
        for (const auto fid : artifact.referenced_fields) {
            const auto fidx = static_cast<std::size_t>(fid);
            if (fidx >= field_dof_offsets_.size() ||
                fidx >= field_dof_handlers_.size()) {
                continue;
            }
            const auto fld_off = field_dof_offsets_[fidx];
            const auto* femap = field_dof_handlers_[fidx].getEntityDofMap();
            if (!femap) {
                continue;
            }
            auto vdofs = femap->getVertexDofs(static_cast<GlobalIndex>(orig_e));
            if (vdofs.empty()) {
                continue;
            }
            FieldValueEntry fve;
            fve.field = fid;
            fve.n_components = static_cast<int>(vdofs.size());
            for (int c = 0; c < fve.n_components && c < MAX_FIELD_VALUE_COMPONENTS; ++c) {
                const auto gidx = static_cast<std::size_t>(
                    vdofs[static_cast<std::size_t>(c)] + fld_off);
                fve.components[c] = (gidx < state.u.size()) ? state.u[gidx] : Real(0.0);
            }
            field_values.push_back(fve);
        }
        return field_values;
    };

    // Purely algebraic monolithic blocks should be solved onto their current
    // algebraic manifold before output evaluation and mixed assembly, so the
    // nonlinear iteration sees the exact direct-feedthrough response instead
    // of stale work values.
    for (auto& entry : deployed_aux_entries_) {
        if (!entry.materialized) {
            continue;
        }
        if (entry.spec.solve_mode != AuxiliarySolveMode::Monolithic ||
            !entry.deriv_provider ||
            !auxiliary_state_manager_ ||
            !auxiliary_state_manager_->hasBlock(entry.instance_name) ||
            !isPureAlgebraicAuxiliary(*entry.model,
                                      static_cast<std::size_t>(entry.spec.size))) {
            continue;
        }

        auto& blk = auxiliary_state_manager_->getBlock(entry.instance_name);
        auto params = buildParamVector(entry);
        auto bound_inputs = buildInputVector(entry);
        const bool has_entity_local_inputs = hasEntityLocalInputs(entry);
        const auto& emap = entry.entity_map;

        for (std::size_t e = 0; e < blk.entityCount(); ++e) {
            auto entity_state_vec = blk.gatherEntityWork(e);
            const auto entity_state_before = entity_state_vec;
            const auto entity_committed = blk.gatherEntityCommitted(e);
            const auto entity_committed_rate = gatherMonolithicCommittedRate(entry, e);
            const auto orig_e = emap.empty() ? e : emap[e];

            if (has_entity_local_inputs) {
                rebuildEntityInputs(entry, orig_e, bound_inputs);
            }

            auto temporal = buildMonolithicAuxiliaryTemporalEvaluation(
                entry.stepper_spec,
                blk,
                e,
                entity_state_vec,
                entity_committed,
                entity_committed_rate,
                state);
            std::fill(temporal.xdot.begin(), temporal.xdot.end(), Real(0.0));

            auto field_values = buildEntityFieldValues(entry, orig_e);

            AuxiliaryLocalContext ctx;
            ctx.time = state.time;
            ctx.dt = state.dt;
            ctx.effective_dt = aux_dt;
            ctx.x = entity_state_vec;
            ctx.xdot = temporal.xdot;
            ctx.history = temporal.history_spans;
            ctx.inputs = bound_inputs;
            ctx.params = params;
            ctx.entity_index = e;
            ctx.field_values = field_values;
            ctx.user_data = state.user_data;

            const bool solved = solvePureAlgebraicAuxiliaryState(
                *entry.model,
                *entry.deriv_provider,
                entity_state_vec,
                ctx);
            if (monolithicAuxTraceEnabled()) {
                std::vector<Real> outputs(static_cast<std::size_t>(entry.model->outputCount()), Real(0.0));
                AuxiliaryLocalContext solved_ctx = ctx;
                solved_ctx.x = entity_state_vec;
                entry.model->evaluateOutputs(solved_ctx, outputs);

                auto format_values = [](std::span<const Real> values) {
                    std::ostringstream oss;
                    oss << "[";
                    for (std::size_t i = 0; i < values.size(); ++i) {
                        if (i != 0) {
                            oss << ", ";
                        }
                        oss << values[i];
                    }
                    oss << "]";
                    return oss.str();
                };

                std::ostringstream oss;
                oss << "prepareAuxiliaryForAssembly: algebraic block='" << entry.instance_name
                    << "' entity=" << e
                    << " solved=" << (solved ? 1 : 0)
                    << " inputs=" << format_values(bound_inputs)
                    << " x_before=" << format_values(entity_state_before)
                    << " x_after=" << format_values(entity_state_vec)
                    << " outputs=" << format_values(outputs);
                FE_LOG_INFO(oss.str());
            }

            if (solved) {
                blk.scatterEntityWork(e, entity_state_vec);
            }
        }
    }

    // Evaluate outputs for deployed models via the base-class output interface.
    for (auto& entry : deployed_aux_entries_) {
        if (!entry.materialized) {
            continue;
        }
        const auto n_outputs = static_cast<std::size_t>(entry.model->outputCount());
        if (n_outputs == 0) continue;
        if (!auxiliary_state_manager_ || !auxiliary_state_manager_->hasBlock(entry.instance_name))
            continue;

        auto& blk = auxiliary_state_manager_->getBlock(entry.instance_name);

        // Build param values: prefer base-class signature, then built-model, then map-key.
        std::vector<Real> params;
        auto declared_params = entry.model->declaredParameterNames();
        if (!declared_params.empty()) {
            params.resize(declared_params.size(), 0.0);
            for (std::size_t pi = 0; pi < declared_params.size(); ++pi) {
                auto it = entry.param_values.find(declared_params[pi]);
                if (it != entry.param_values.end()) params[pi] = it->second;
            }
        } else if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(entry.model.get())) {
            const auto& sig = built->signature();
            params.resize(sig.parameters.size(), 0.0);
            for (std::size_t pi = 0; pi < sig.parameters.size(); ++pi) {
                auto it = entry.param_values.find(sig.parameters[pi].name);
                if (it != entry.param_values.end()) params[pi] = it->second;
            }
        } else {
            for (const auto& [pname, pval] : entry.param_values) {
                params.push_back(pval);
            }
        }

        // Build bound inputs for output evaluation.
        // For built models: ordered by signature. For generic: ordered by binding key.
        std::vector<Real> bound_inputs;
        if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(entry.model.get())) {
            const auto& sig = built->signature();
            std::size_t total_input_size = 0;
            for (const auto& inp : sig.inputs) total_input_size += static_cast<std::size_t>(inp.size);
            bound_inputs.resize(total_input_size, 0.0);
            std::size_t inp_offset = 0;
            for (const auto& inp : sig.inputs) {
                auto bind_it = entry.input_bindings.find(inp.name);
                if (bind_it != entry.input_bindings.end() && auxiliary_input_registry_) {
                    auto reg_vals = auxiliary_input_registry_->valuesOf(bind_it->second);
                    for (std::size_t k = 0; k < std::min(reg_vals.size(), static_cast<std::size_t>(inp.size)); ++k)
                        bound_inputs[inp_offset + k] = reg_vals[k];
                }
                inp_offset += static_cast<std::size_t>(inp.size);
            }
        } else {
            // Non-built models: prefer declaredInputNames() with name:size
            // parsing, then map-key order.
            auto decl_in = entry.model->declaredInputNames();
            if (!decl_in.empty() && auxiliary_input_registry_) {
                for (const auto& raw : decl_in) {
                    auto [iname, input_size] = parseDeclaredInputName(raw);
                    auto bind_it = entry.input_bindings.find(iname);
                    if (bind_it != entry.input_bindings.end() &&
                        auxiliary_input_registry_->hasInput(bind_it->second)) {
                        auto vals = auxiliary_input_registry_->valuesOf(bind_it->second);
                        for (int k = 0; k < input_size; ++k) {
                            bound_inputs.push_back(
                                k < static_cast<int>(vals.size()) ? vals[static_cast<std::size_t>(k)] : 0.0);
                        }
                    } else {
                        bound_inputs.resize(bound_inputs.size() + static_cast<std::size_t>(input_size), 0.0);
                    }
                }
            } else if (!entry.input_bindings.empty() && auxiliary_input_registry_) {
                for (const auto& [model_name, reg_name] : entry.input_bindings) {
                    if (auxiliary_input_registry_->hasInput(reg_name)) {
                        auto vals = auxiliary_input_registry_->valuesOf(reg_name);
                        bound_inputs.insert(bound_inputs.end(), vals.begin(), vals.end());
                    }
                }
            }
        }

        const auto n_entities = blk.entityCount();
        entry.output_buffer.resize(n_entities * n_outputs);

        const auto& emap = entry.entity_map; // empty = identity mapping

        // Detect entity-local bindings for output eval.
        const bool has_entity_local_inputs = hasEntityLocalInputs(entry);

        for (std::size_t e = 0; e < n_entities; ++e) {
            // Layout-aware entity gather.
            auto entity_state_vec = blk.gatherEntityWork(e);

            // Rebuild bound inputs per entity when entity-local bindings exist.
            const auto orig_e = emap.empty() ? e : emap[e];
            if (has_entity_local_inputs) {
                rebuildEntityInputs(entry, orig_e, bound_inputs);
            }

            const auto entity_committed = blk.gatherEntityCommitted(e);
            const auto entity_committed_rate = gatherMonolithicCommittedRate(entry, e);
            auto temporal = buildMonolithicAuxiliaryTemporalEvaluation(
                entry.stepper_spec,
                blk,
                e,
                entity_state_vec,
                entity_committed,
                entity_committed_rate,
                state);

            // Populate field_values for models with direct FE field references.
            auto fv_prep = buildEntityFieldValues(entry, orig_e);

            AuxiliaryLocalContext ctx;
            ctx.time = state.time;
            ctx.dt = state.dt;
            ctx.effective_dt = aux_dt;
            ctx.x = entity_state_vec;
            ctx.xdot = temporal.xdot;
            ctx.history = temporal.history_spans;
            ctx.inputs = bound_inputs;
            ctx.params = params;
            ctx.entity_index = e;
            ctx.field_values = fv_prep;
            ctx.user_data = state.user_data;

            std::span<Real> out_span{
                entry.output_buffer.data() + e * n_outputs, n_outputs};
            entry.model->evaluateOutputs(ctx, out_span);
            if (monolithicAuxTraceEnabled()) {
                auto format_values = [](std::span<const Real> values) {
                    std::ostringstream oss;
                    oss << "[";
                    for (std::size_t i = 0; i < values.size(); ++i) {
                        if (i != 0) {
                            oss << ", ";
                        }
                        oss << values[i];
                    }
                    oss << "]";
                    return oss.str();
                };
                std::ostringstream oss;
                oss << "prepareAuxiliaryForAssembly: output buffer block='" << entry.instance_name
                    << "' entity=" << e
                    << " inputs=" << format_values(bound_inputs)
                    << " state=" << format_values(entity_state_vec)
                    << " outputs=" << format_values(std::span<const Real>(out_span.data(), out_span.size()));
                FE_LOG_INFO(oss.str());
            }
        }
    }
}

void FESystem::deployAuxiliaryModel(AuxiliaryDeployedInstance instance)
{
    if (!instance.hasExplicitName()) {
        instance.setResolvedInstanceName(resolveDeploymentInstanceName_(instance));
    }

    auto diag = instance.validate();
    FE_THROW_IF(!diag.empty(), InvalidArgumentException,
                "FESystem::deployAuxiliaryModel: " + diag);

    // Validate declared input name suffixes at deployment time.
    validateDeclaredInputNames(*instance.model());

    DeployedAuxEntry entry;
    entry.model = instance.model();
    entry.instance_name = instance.instanceName();

    // Build spec from deployment configuration.
    entry.spec.name = instance.instanceName();
    entry.spec.size = instance.model()->dimension();
    entry.spec.scope = instance.getScope();
    entry.spec.solve_mode = instance.getSolveMode();
    entry.spec.schedule_mode = instance.getSchedule();
    entry.spec.layout_mode = instance.getLayoutMode();
    entry.spec.ordering = instance.getEntityOrdering();
    entry.spec.deployment_region = instance.getRegion();
    entry.spec.failure_policy = instance.getFailurePolicy();
    const auto structural_metadata = instance.model()->structuralMetadata();
    entry.spec.variable_kinds = structural_metadata.variable_kinds;
    entry.spec.constraint_groups = structural_metadata.constraint_groups;
    entry.spec.dae_index_hint = structural_metadata.dae_index_hint;
    entry.spec.n_event_functions = structural_metadata.n_event_functions;
    entry.spec.has_nonsmooth =
        instance.model()->hasNonsmoothHooks() || structural_metadata.has_nonsmooth;
    entry.spec.nonsmooth_policy = instance.getNonsmoothPolicy();
    entry.spec.mass_diagonal = instance.model()->massDiagonal();
    FE_THROW_IF(!entry.spec.mass_diagonal.empty() &&
                    entry.spec.mass_diagonal.size() !=
                        static_cast<std::size_t>(entry.spec.size),
                InvalidArgumentException,
                "FESystem::deployAuxiliaryModel: auxiliary instance '" +
                    entry.instance_name + "' returned massDiagonal() of size " +
                    std::to_string(entry.spec.mass_diagonal.size()) +
                    " for block dimension " + std::to_string(entry.spec.size));
    entry.spec.has_mass_matrix =
        instance.model()->hasMassMatrix() || !entry.spec.mass_diagonal.empty();
    if (instance.model()->hasEventFunctions() || structural_metadata.has_events) {
        entry.spec.event_mode = AuxiliaryEventMode::EventHook;
    } else if (entry.spec.has_nonsmooth) {
        entry.spec.event_mode = AuxiliaryEventMode::ComplementarityHook;
    }
    // Copy derivative policy: prefer explicit instance policy, then built-model policy.
    if (instance.hasExplicitDerivativePolicy()) {
        entry.spec.derivative_policy = instance.getDerivativePolicy();
    } else if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(instance.model().get())) {
        entry.spec.derivative_policy = built->derivativePolicy();
    }
    entry.stepper_spec = instance.getStepperSpec();
    entry.initial_values = instance.initialValues();
    for (const auto& [k, v] : instance.inputBindings())
        entry.input_bindings[k] = v;
    for (const auto& [k, v] : instance.coupledBindings())
        entry.coupled_bindings[k] = v;
    entry.param_values = instance.paramValues();
    entry.constraint_bindings = instance.constraintBindings();
    entry.solver_metadata = instance.solverMetadata();
    if (entry.solver_metadata.has_value()) {
        entry.solver_metadata->block_name = entry.instance_name;
    }
    entry.explicit_entity_count = instance.getEntityCount();
    entry.qp_offsets.assign(instance.qpOffsets().begin(), instance.qpOffsets().end());
    entry.ragged_entity_size_provider = instance.raggedEntitySizeProvider();
    entry.ragged_component_offsets.assign(instance.raggedComponentOffsets().begin(),
                                          instance.raggedComponentOffsets().end());
    entry.quadrature_reference_field = instance.quadratureReferenceField();
    entry.quadrature_reference_operator = instance.quadratureReferenceOperator();
    entry.variant_group = instance.variantGroup();
    entry.variant_key = instance.variantKey();
    entry.activation_mode = instance.getActivationMode();
    assignAuxiliaryOutputIds_(entry);

    deployed_aux_entries_.push_back(std::move(entry));
}

AuxiliaryInstanceHandle FESystem::deploy(AuxiliaryDeployedInstance instance)
{
    if (!instance.hasExplicitName()) {
        instance.setResolvedInstanceName(resolveDeploymentInstanceName_(instance));
    }
    const std::string inst_name = instance.instanceName();
    deployAuxiliaryModel(std::move(instance));
    return AuxiliaryInstanceHandle(inst_name);
}

void FESystem::selectAuxiliaryVariant(std::string group, std::string key)
{
    FE_THROW_IF(group.empty(), InvalidArgumentException,
                "FESystem::selectAuxiliaryVariant: empty group");
    FE_THROW_IF(key.empty(), InvalidArgumentException,
                "FESystem::selectAuxiliaryVariant: empty key");
    FE_THROW_IF(is_setup_, InvalidStateException,
                "FESystem::selectAuxiliaryVariant: selection is frozen after setup()");
    auxiliary_variant_selection_[std::move(group)] = std::move(key);
}

void FESystem::clearAuxiliaryVariantSelection(std::string_view group)
{
    FE_THROW_IF(group.empty(), InvalidArgumentException,
                "FESystem::clearAuxiliaryVariantSelection: empty group");
    FE_THROW_IF(is_setup_, InvalidStateException,
                "FESystem::clearAuxiliaryVariantSelection: selection is frozen after setup()");
    auxiliary_variant_selection_.erase(std::string(group));
}

AuxiliaryInputHandle FESystem::boundaryIntegral(
    const std::string& input_name,
    forms::FormExpr integrand,
    int boundary_marker,
    forms::BoundaryFunctional::Reduction reduction,
    AuxiliaryInputUpdateSchedule schedule)
{
    return registerBoundaryIntegralHandle_(
        input_name, std::move(integrand), boundary_marker, reduction, schedule);
}

AuxiliaryInputHandle FESystem::boundaryIntegral(
    forms::FormExpr integrand,
    int boundary_marker,
    forms::BoundaryFunctional::Reduction reduction,
    AuxiliaryInputUpdateSchedule schedule)
{
    const auto input_name = generateUniqueAuxiliaryInputName_(
        "_boundary_integral_b" + std::to_string(boundary_marker));
    return registerBoundaryIntegralHandle_(
        input_name, std::move(integrand), boundary_marker, reduction, schedule);
}

AuxiliaryInputHandle FESystem::registerBoundaryIntegralHandle_(
    const std::string& input_name,
    forms::FormExpr integrand,
    int boundary_marker,
    forms::BoundaryFunctional::Reduction reduction,
    AuxiliaryInputUpdateSchedule schedule)
{
    // Gather referenced fields before moving integrand.
    std::vector<FieldId> refs;
    if (const auto* root = integrand.node()) {
        gatherFieldIds(*root, refs);
    }

    auto def = std::make_shared<FEQuantityDefinition>();
    def->name = input_name;
    def->kind = FEQuantityKind::BoundaryIntegral;
    def->shape = FEQuantityShape::scalar();
    def->referenced_fields = refs;
    def->expression = integrand;  // copy before move
    def->boundary_marker = boundary_marker;
    def->capabilities.explicit_evaluation = true;
    // Monolithic linearization for boundary integrals requires the
    // StandardAssembler to have a GlobalSystemView solution set.
    // This works in production (backends provide GenericVector) but
    // not in lightweight test configurations with raw span solutions.
    // Mark as supported — the runtime path is wired through
    // evaluateFunctionalGradient() → assembleBoundaryGradient().
    def->capabilities.monolithic_linearization = !refs.empty();

    registerBoundaryIntegralInput(input_name, std::move(integrand),
                                  boundary_marker, reduction, schedule);
    feQuantityRegistry().registerDefinition(*def);
    return AuxiliaryInputHandle(input_name, std::move(def));
}

AuxiliaryInputHandle FESystem::boundaryIntegral(
    const std::string& input_name,
    forms::BoundaryFunctional functional,
    AuxiliaryInputUpdateSchedule schedule)
{
    return registerBoundaryIntegralHandle_(
        input_name, std::move(functional), schedule);
}

AuxiliaryInputHandle FESystem::boundaryIntegral(
    forms::BoundaryFunctional functional,
    AuxiliaryInputUpdateSchedule schedule)
{
    const auto input_name = generateUniqueAuxiliaryInputName_(
        "_boundary_integral_b" + std::to_string(functional.boundary_marker));
    return registerBoundaryIntegralHandle_(
        input_name, std::move(functional), schedule);
}

AuxiliaryInputHandle FESystem::registerBoundaryIntegralHandle_(
    const std::string& input_name,
    forms::BoundaryFunctional functional,
    AuxiliaryInputUpdateSchedule schedule)
{
    std::vector<FieldId> refs;
    if (const auto* root = functional.integrand.node()) {
        gatherFieldIds(*root, refs);
    }

    auto def = std::make_shared<FEQuantityDefinition>();
    def->name = input_name;
    def->kind = FEQuantityKind::BoundaryIntegral;
    def->shape = FEQuantityShape::scalar();
    def->referenced_fields = refs;
    def->expression = functional.integrand;
    def->boundary_marker = functional.boundary_marker;
    def->capabilities.explicit_evaluation = true;
    def->capabilities.monolithic_linearization = !refs.empty();

    registerBoundaryIntegralInput(input_name, std::move(functional), schedule);
    feQuantityRegistry().registerDefinition(*def);
    return AuxiliaryInputHandle(input_name, std::move(def));
}

std::string FESystem::generateUniqueAuxiliaryInputName_(std::string_view prefix)
{
    std::string candidate;
    auto& input_reg = auxiliaryInputRegistry();
    auto& quantity_reg = feQuantityRegistry();
    do {
        candidate = std::string(prefix) + "_" +
                    std::to_string(generated_boundary_input_counter_++);
    } while (input_reg.hasInput(candidate) || quantity_reg.hasDefinition(candidate));
    return candidate;
}

bool FESystem::hasDeployedInstanceName_(std::string_view instance_name) const
{
    return std::any_of(
        deployed_aux_entries_.begin(), deployed_aux_entries_.end(),
        [&](const DeployedAuxEntry& entry) { return entry.instance_name == instance_name; });
}

std::string FESystem::makeScopeAwareInstanceBaseName_(
    const AuxiliaryDeployedInstance& instance) const
{
    const std::string model_name =
        (instance.model() && !instance.model()->modelName().empty())
            ? instance.model()->modelName()
            : std::string("aux");
    const auto scope = instance.getScope();
    if (scope == AuxiliaryStateScope::Boundary) {
        const auto& region = instance.getRegion();
        if (region.kind == AuxiliaryRegionKind::BoundarySet && !region.identity.empty()) {
            return model_name + "_b" + region.identity;
        }
        return model_name + "_b";
    }
    return model_name + "_" + scopeAutoNameToken(scope);
}

std::string FESystem::resolveDeploymentInstanceName_(
    const AuxiliaryDeployedInstance& instance) const
{
    if (instance.hasExplicitName()) {
        return instance.instanceName();
    }

    const std::string base = makeScopeAwareInstanceBaseName_(instance);
    if (instance.getScope() == AuxiliaryStateScope::Boundary) {
        if (!hasDeployedInstanceName_(base)) {
            return base;
        }
        for (std::size_t suffix = 1;; ++suffix) {
            const auto candidate = base + "_" + std::to_string(suffix);
            if (!hasDeployedInstanceName_(candidate)) {
                return candidate;
            }
        }
    }

    for (std::size_t counter = 0;; ++counter) {
        const auto candidate = base + std::to_string(counter);
        if (!hasDeployedInstanceName_(candidate)) {
            return candidate;
        }
    }
}

AuxiliaryInputHandle FESystem::derivedInput(
    const std::string& name,
    forms::FormExpr expr,
    AuxiliaryInputUpdateSchedule schedule)
{
    FE_THROW_IF(name.empty(), InvalidArgumentException,
                "FESystem::derivedInput: empty name");

    auto& reg = auxiliaryInputRegistry();

    AuxiliaryInputSpec spec;
    spec.name = name;
    spec.size = 1;
    spec.producer = AuxiliaryInputProducer::FormulationCallback;
    spec.update_schedule = schedule;

    // Auto-discover dependencies by scanning the expression for AuxiliaryInputSymbol
    // nodes referencing other registry inputs.  Must do this BEFORE moving expr.
    std::vector<std::string> deps;
    if (const auto* root = expr.node()) {
        std::function<void(const forms::FormExprNode&)> scan =
            [&](const forms::FormExprNode& n) {
                if (n.type() == forms::FormExprType::AuxiliaryInputSymbol) {
                    if (auto sym = n.symbolName()) {
                        deps.push_back(std::string(*sym));
                    }
                }
                for (const auto* child : n.children()) {
                    if (child) scan(*child);
                }
            };
        scan(*root);
    }

    // Reject self-references BEFORE any side effects (registration, deferred
    // expression storage).  A failed check must not leave a partially-registered
    // input in the registry or a dangling deferred expression.
    for (const auto& dep : deps) {
        FE_THROW_IF(dep == name, InvalidArgumentException,
                    "FESystem::derivedInput('" + name +
                        "'): expression references itself — "
                        "self-referential derived inputs are not allowed");
    }

    // Store the expression in a shared_ptr so it can be resolved to
    // slot-based refs during finalizeDeferredInputDeps() (after all inputs
    // are registered and slots are stable).
    auto resolved_expr = std::make_shared<forms::FormExpr>(std::move(expr));
    auto* reg_ptr = &reg;

    reg.registerInput(spec,
        [reg_ptr, resolved_expr](Real time, Real dt, std::span<Real> out) {
            forms::PointEvalContext pctx;
            pctx.time = time;
            pctx.dt = dt;
            pctx.auxiliary_inputs = reg_ptr->all();
            out[0] = forms::evaluateScalarAt(*resolved_expr, pctx);
        });

    // Store (name, shared_ptr) for deferred symbol resolution.
    deferred_derived_exprs_.emplace_back(name, resolved_expr);

    // Defer dependency wiring to finalizeDeferredInputDeps().
    // At registration time, referenced inputs may not yet exist.  Wiring
    // now would silently drop any forward references.  At finalization,
    // all inputs are registered, so any unresolved name is a real error.
    for (const auto& dep : deps) {
        deferred_input_deps_.emplace_back(name, dep);
    }

    auto def = std::make_shared<FEQuantityDefinition>();
    def->name = name;
    def->kind = FEQuantityKind::DerivedCallback;
    def->shape = FEQuantityShape::scalar();
    def->capabilities.explicit_evaluation = true;
    def->capabilities.monolithic_linearization = false;

    feQuantityRegistry().registerDefinition(*def);
    return AuxiliaryInputHandle(name, std::move(def));
}

AuxiliaryInputHandle FESystem::sampledField(
    const std::string& input_name,
    const std::string& field_name,
    std::size_t n_entities)
{
    registerSampledFieldInput(input_name, field_name, n_entities);

    // Determine field ID and components for the definition.
    const auto fid = field_registry_.findByName(field_name);
    const int components = field_registry_.get(fid).components;

    auto def = std::make_shared<FEQuantityDefinition>();
    def->name = input_name;
    def->kind = FEQuantityKind::SampledField;
    def->shape = (components == 1)
        ? FEQuantityShape::scalar()
        : FEQuantityShape::vector(components);
    def->referenced_fields = {fid};
    def->source_field_name = field_name;
    def->entity_count = n_entities;
    def->capabilities.explicit_evaluation = true;
    // Sampled field dI/du is identity at sampled DOFs.
    def->capabilities.monolithic_linearization = true;

    feQuantityRegistry().registerDefinition(*def);
    return AuxiliaryInputHandle(input_name, std::move(def));
}

AuxiliaryInputHandle FESystem::boundaryNodalSum(
    const std::string& input_name,
    const std::string& field_name,
    int boundary_marker)
{
    registerBoundaryNodalSumInput(input_name, field_name, boundary_marker);

    const auto fid = field_registry_.findByName(field_name);
    const int components = field_registry_.get(fid).components;

    auto def = std::make_shared<FEQuantityDefinition>();
    def->name = input_name;
    def->kind = FEQuantityKind::BoundaryNodalSum;
    def->shape = (components == 1)
        ? FEQuantityShape::scalar()
        : FEQuantityShape::vector(components);
    def->referenced_fields = {fid};
    def->source_field_name = field_name;
    def->boundary_marker = boundary_marker;
    def->capabilities.explicit_evaluation = true;
    def->capabilities.monolithic_linearization = false;  // nodal sum, not quadrature-weighted

    feQuantityRegistry().registerDefinition(*def);
    return AuxiliaryInputHandle(input_name, std::move(def));
}

AuxiliaryInputHandle FESystem::boundaryAverage(
    const std::string& input_name,
    forms::FormExpr integrand,
    int boundary_marker,
    AuxiliaryInputUpdateSchedule schedule)
{
    // Boundary average = boundary integral / boundary measure.
    // Use BoundaryFunctional with Average reduction mode.
    std::vector<FieldId> refs;
    if (const auto* root = integrand.node()) {
        gatherFieldIds(*root, refs);
    }

    auto def = std::make_shared<FEQuantityDefinition>();
    def->name = input_name;
    def->kind = FEQuantityKind::BoundaryAverage;
    def->shape = FEQuantityShape::scalar();
    def->referenced_fields = refs;
    def->expression = integrand;
    def->boundary_marker = boundary_marker;
    def->capabilities.explicit_evaluation = true;
    def->capabilities.monolithic_linearization = !refs.empty();

    registerBoundaryIntegralInput(input_name, std::move(integrand),
                                  boundary_marker,
                                  forms::BoundaryFunctional::Reduction::Average,
                                  schedule);
    feQuantityRegistry().registerDefinition(*def);
    return AuxiliaryInputHandle(input_name, std::move(def));
}

AuxiliaryInputHandle FESystem::domainIntegral(
    const std::string& input_name,
    forms::FormExpr integrand,
    AuxiliaryInputUpdateSchedule schedule)
{
    FE_THROW_IF(input_name.empty(), InvalidArgumentException,
                "FESystem::domainIntegral: empty input_name");

    std::vector<FieldId> refs;
    if (const auto* root = integrand.node()) {
        gatherFieldIds(*root, refs);
    }

    auto def = std::make_shared<FEQuantityDefinition>();
    def->name = input_name;
    def->kind = FEQuantityKind::DomainIntegral;
    def->shape = FEQuantityShape::scalar();
    def->referenced_fields = refs;
    def->expression = integrand;
    def->capabilities.explicit_evaluation = true;
    def->capabilities.monolithic_linearization = !refs.empty();

    // Domain integrals use the FunctionalAssembler over all cells.
    // Determine the primary field for mesh/space context.
    FieldId primary_fid = INVALID_FIELD_ID;
    if (!refs.empty()) {
        primary_fid = refs.front();
    } else {
        const auto& recs = field_registry_.records();
        if (!recs.empty()) primary_fid = static_cast<FieldId>(0);
    }
    FE_THROW_IF(primary_fid == INVALID_FIELD_ID, InvalidStateException,
                "FESystem::domainIntegral('" + input_name +
                    "'): at least one FE field must be registered");

    auto& reg = auxiliaryInputRegistry();

    AuxiliaryInputSpec spec;
    spec.name = input_name;
    spec.size = 1;
    spec.producer = AuxiliaryInputProducer::RegionIntegral;
    spec.update_schedule = schedule;
    spec.requires_mpi_reduction = true;

    // Evaluate via the BoundaryReductionService using a cell-domain
    // functional.  The service's functional assembler handles both
    // boundary and cell assembly.  We use boundary_marker = -1 to
    // signal a domain (all-cells) integral.
    auto captured_integrand = integrand;
    const auto captured_fid = primary_fid;
    const std::string func_name = input_name;

    // Register a domain functional with the per-field reduction service.
    auto& svc = boundaryReductionService(captured_fid);
    forms::BoundaryFunctional domain_func;
    domain_func.name = func_name;
    domain_func.integrand = std::move(integrand);
    domain_func.boundary_marker = -1;  // domain (all cells)
    domain_func.reduction = forms::BoundaryFunctional::Reduction::Sum;
    domain_func.is_domain_functional = true;
    svc.addBoundaryFunctional(domain_func);
    bindSecondaryFields(svc, captured_fid, refs);

    reg.registerInput(spec,
        [this, func_name, captured_fid]
        (Real time, Real dt, std::span<Real> out) {
            SystemStateView state;
            state.time = time;
            state.dt = dt;
            state.u = cached_solution_u_;
            state.u_vector = cached_solution_vector_;
            state.u_prev = cached_solution_u_prev_;
            state.u_prev_vector = cached_solution_prev_vector_;
            state.time_integration = cached_time_integration_;
            state.user_data = cached_user_data_;

            auto it = boundary_reduction_services_.find(captured_fid);
            if (it != boundary_reduction_services_.end() && it->second) {
                out[0] = it->second->evaluateFunctional(func_name, state);
            } else {
                out[0] = 0.0;
            }
        });

    feQuantityRegistry().registerDefinition(*def);
    return AuxiliaryInputHandle(input_name, std::move(def));
}

AuxiliaryInputHandle FESystem::domainAverage(
    const std::string& input_name,
    forms::FormExpr integrand,
    AuxiliaryInputUpdateSchedule schedule)
{
    FE_THROW_IF(input_name.empty(), InvalidArgumentException,
                "FESystem::domainAverage: empty input_name");

    std::vector<FieldId> refs;
    if (const auto* root = integrand.node()) {
        gatherFieldIds(*root, refs);
    }

    auto def = std::make_shared<FEQuantityDefinition>();
    def->name = input_name;
    def->kind = FEQuantityKind::DomainAverage;
    def->shape = FEQuantityShape::scalar();
    def->referenced_fields = refs;
    def->expression = integrand;
    def->capabilities.explicit_evaluation = true;
    def->capabilities.monolithic_linearization = !refs.empty();

    // Domain average = domain integral / domain measure.
    // Register two callbacks: the integral and the measure, then
    // combine in a derived callback.
    const std::string integral_name = input_name + "__integral";
    const std::string measure_name = input_name + "__measure";

    // Register the integral.
    domainIntegral(integral_name, integrand, schedule);

    // Register the measure (∫ 1 dx = total domain volume).
    domainIntegral(measure_name, forms::FormExpr::constant(1.0), schedule);

    // Register the average as a derived callback.
    auto& reg = auxiliaryInputRegistry();
    AuxiliaryInputSpec spec;
    spec.name = input_name;
    spec.size = 1;
    spec.producer = AuxiliaryInputProducer::RegionAverage;
    spec.update_schedule = schedule;

    auto* reg_ptr = &reg;
    const auto int_name = integral_name;
    const auto meas_name = measure_name;

    reg.registerInput(spec,
        [reg_ptr, int_name, meas_name](Real, Real, std::span<Real> out) {
            const Real integral = reg_ptr->get(int_name);
            const Real measure = reg_ptr->get(meas_name);
            out[0] = (measure > 0.0) ? integral / measure : 0.0;
        });
    reg.addDependency(input_name, integral_name);
    reg.addDependency(input_name, measure_name);

    feQuantityRegistry().registerDefinition(*def);
    return AuxiliaryInputHandle(input_name, std::move(def));
}

AuxiliaryInputHandle FESystem::regionIntegral(
    const std::string& input_name,
    forms::FormExpr integrand,
    AuxiliaryInputUpdateSchedule schedule)
{
    FE_THROW_IF(input_name.empty(), InvalidArgumentException,
                "FESystem::regionIntegral: empty input_name");

    std::vector<FieldId> refs;
    if (const auto* root = integrand.node()) {
        gatherFieldIds(*root, refs);
    }

    auto def = std::make_shared<FEQuantityDefinition>();
    def->name = input_name;
    def->kind = FEQuantityKind::RegionIntegral;
    def->shape = FEQuantityShape::scalar();
    def->referenced_fields = refs;
    def->expression = integrand;
    def->region_marker = -1;
    def->capabilities.explicit_evaluation = true;
    def->capabilities.monolithic_linearization = !refs.empty();

    FieldId primary_fid = INVALID_FIELD_ID;
    if (!refs.empty()) {
        primary_fid = refs.front();
    } else {
        const auto& recs = field_registry_.records();
        if (!recs.empty()) primary_fid = static_cast<FieldId>(0);
    }
    FE_THROW_IF(primary_fid == INVALID_FIELD_ID, InvalidStateException,
                "FESystem::regionIntegral('" + input_name +
                    "'): at least one FE field must be registered");

    auto& svc = boundaryReductionService(primary_fid);
    forms::BoundaryFunctional region_func;
    region_func.name = input_name;
    region_func.integrand = integrand;
    region_func.boundary_marker = -1;
    region_func.reduction = forms::BoundaryFunctional::Reduction::Sum;
    region_func.is_domain_functional = true;
    region_func.region_marker = -1;
    svc.addBoundaryFunctional(region_func);
    bindSecondaryFields(svc, primary_fid, refs);

    auto& reg = auxiliaryInputRegistry();
    AuxiliaryInputSpec spec;
    spec.name = input_name;
    spec.size = 1;
    spec.entity_count = auxiliaryTopologyRegionInputEntityCount_();
    spec.producer = AuxiliaryInputProducer::RegionIntegral;
    spec.update_schedule = schedule;
    spec.requires_mpi_reduction = true;

    const auto captured_fid = primary_fid;
    const std::string func_name = input_name;

    reg.registerEntityInput(spec,
        [this, func_name, captured_fid]
        (Real time, Real dt, std::size_t entity_index, std::span<Real> out) {
            SystemStateView state;
            state.time = time;
            state.dt = dt;
            state.u = cached_solution_u_;
            state.u_vector = cached_solution_vector_;
            state.u_prev = cached_solution_u_prev_;
            state.u_prev_vector = cached_solution_prev_vector_;
            state.u_prev2 = cached_solution_u_prev2_;
            state.u_prev2_vector = cached_solution_prev2_vector_;
            state.time_integration = cached_time_integration_;
            state.user_data = cached_user_data_;

            const auto cells = auxiliaryTopologyRegionCells_(entity_index);
            auto it = boundary_reduction_services_.find(captured_fid);
            if (it != boundary_reduction_services_.end() && it->second) {
                out[0] = it->second->evaluateFunctionalOverCells(func_name, cells, state);
            } else {
                out[0] = Real(0.0);
            }
        });

    feQuantityRegistry().registerDefinition(*def);
    return AuxiliaryInputHandle(input_name, std::move(def));
}

AuxiliaryInputHandle FESystem::regionIntegral(
    const std::string& input_name,
    forms::FormExpr integrand,
    int region_marker,
    AuxiliaryInputUpdateSchedule schedule)
{
    FE_THROW_IF(input_name.empty(), InvalidArgumentException,
                "FESystem::regionIntegral: empty input_name");

    std::vector<FieldId> refs;
    if (const auto* root = integrand.node()) {
        gatherFieldIds(*root, refs);
    }

    auto def = std::make_shared<FEQuantityDefinition>();
    def->name = input_name;
    def->kind = FEQuantityKind::RegionIntegral;
    def->shape = FEQuantityShape::scalar();
    def->referenced_fields = refs;
    def->expression = integrand;
    def->region_marker = region_marker;
    def->capabilities.explicit_evaluation = true;
    def->capabilities.monolithic_linearization = !refs.empty();

    // Region integrals use BoundaryReductionService with a domain functional
    // filtered by region marker (material/domain ID).
    FieldId primary_fid = INVALID_FIELD_ID;
    if (!refs.empty()) {
        primary_fid = refs.front();
    } else {
        const auto& recs = field_registry_.records();
        if (!recs.empty()) primary_fid = static_cast<FieldId>(0);
    }
    FE_THROW_IF(primary_fid == INVALID_FIELD_ID, InvalidStateException,
                "FESystem::regionIntegral('" + input_name +
                    "'): at least one FE field must be registered");

    auto& svc = boundaryReductionService(primary_fid);
    forms::BoundaryFunctional region_func;
    region_func.name = input_name;
    region_func.integrand = integrand;
    region_func.boundary_marker = -1;
    region_func.reduction = forms::BoundaryFunctional::Reduction::Sum;
    region_func.is_domain_functional = true;
    region_func.region_marker = region_marker;
    svc.addBoundaryFunctional(region_func);
    bindSecondaryFields(svc, primary_fid, refs);

    auto& reg = auxiliaryInputRegistry();
    AuxiliaryInputSpec spec;
    spec.name = input_name;
    spec.size = 1;
    spec.producer = AuxiliaryInputProducer::DomainIntegral;
    spec.update_schedule = schedule;
    spec.requires_mpi_reduction = true;

    const auto captured_fid = primary_fid;
    const std::string func_name = input_name;

    reg.registerInput(spec,
        [this, func_name, captured_fid]
        (Real time, Real dt, std::span<Real> out) {
            SystemStateView state;
            state.time = time;
            state.dt = dt;
            state.u = cached_solution_u_;
            state.u_vector = cached_solution_vector_;
            state.u_prev = cached_solution_u_prev_;
            state.u_prev_vector = cached_solution_prev_vector_;
            state.time_integration = cached_time_integration_;
            state.user_data = cached_user_data_;

            auto it = boundary_reduction_services_.find(captured_fid);
            if (it != boundary_reduction_services_.end() && it->second) {
                out[0] = it->second->evaluateFunctional(func_name, state);
            } else {
                out[0] = 0.0;
            }
        });

    feQuantityRegistry().registerDefinition(*def);
    return AuxiliaryInputHandle(input_name, std::move(def));
}

AuxiliaryInputHandle FESystem::regionAverage(
    const std::string& input_name,
    forms::FormExpr integrand,
    AuxiliaryInputUpdateSchedule schedule)
{
    FE_THROW_IF(input_name.empty(), InvalidArgumentException,
                "FESystem::regionAverage: empty input_name");

    std::vector<FieldId> refs;
    if (const auto* root = integrand.node()) {
        gatherFieldIds(*root, refs);
    }

    auto def = std::make_shared<FEQuantityDefinition>();
    def->name = input_name;
    def->kind = FEQuantityKind::RegionAverage;
    def->shape = FEQuantityShape::scalar();
    def->referenced_fields = refs;
    def->expression = integrand;
    def->region_marker = -1;
    def->capabilities.explicit_evaluation = true;
    def->capabilities.monolithic_linearization = !refs.empty();

    const std::string integral_name = input_name + "__integral";
    const std::string measure_name = input_name + "__measure";

    regionIntegral(integral_name, integrand, schedule);
    regionIntegral(measure_name, forms::FormExpr::constant(1.0), schedule);

    auto& reg = auxiliaryInputRegistry();
    AuxiliaryInputSpec spec;
    spec.name = input_name;
    spec.size = 1;
    spec.entity_count = auxiliaryTopologyRegionInputEntityCount_();
    spec.producer = AuxiliaryInputProducer::RegionAverage;
    spec.update_schedule = schedule;

    auto* reg_ptr = &reg;
    const auto int_name = integral_name;
    const auto meas_name = measure_name;

    reg.registerEntityInput(spec,
        [reg_ptr, int_name, meas_name]
        (Real, Real, std::size_t entity_index, std::span<Real> out) {
            const auto integral = reg_ptr->valuesOf(int_name, entity_index);
            const auto measure = reg_ptr->valuesOf(meas_name, entity_index);
            const Real m = measure.empty() ? Real(0.0) : measure[0];
            out[0] = (m > Real(0.0) && !integral.empty()) ? integral[0] / m : Real(0.0);
        });
    reg.addDependency(input_name, integral_name);
    reg.addDependency(input_name, measure_name);

    feQuantityRegistry().registerDefinition(*def);
    return AuxiliaryInputHandle(input_name, std::move(def));
}

AuxiliaryInputHandle FESystem::regionAverage(
    const std::string& input_name,
    forms::FormExpr integrand,
    int region_marker,
    AuxiliaryInputUpdateSchedule schedule)
{
    FE_THROW_IF(input_name.empty(), InvalidArgumentException,
                "FESystem::regionAverage: empty input_name");

    std::vector<FieldId> refs;
    if (const auto* root = integrand.node()) {
        gatherFieldIds(*root, refs);
    }

    auto def = std::make_shared<FEQuantityDefinition>();
    def->name = input_name;
    def->kind = FEQuantityKind::RegionAverage;
    def->shape = FEQuantityShape::scalar();
    def->referenced_fields = refs;
    def->expression = integrand;
    def->region_marker = region_marker;
    def->capabilities.explicit_evaluation = true;
    def->capabilities.monolithic_linearization = !refs.empty();

    // Region average = region integral / region measure.
    const std::string integral_name = input_name + "__integral";
    const std::string measure_name = input_name + "__measure";

    regionIntegral(integral_name, integrand, region_marker, schedule);
    regionIntegral(measure_name, forms::FormExpr::constant(1.0), region_marker, schedule);

    auto& reg = auxiliaryInputRegistry();
    AuxiliaryInputSpec spec;
    spec.name = input_name;
    spec.size = 1;
    spec.producer = AuxiliaryInputProducer::DomainAverage;
    spec.update_schedule = schedule;

    auto* reg_ptr = &reg;
    const auto int_name = integral_name;
    const auto meas_name = measure_name;

    reg.registerInput(spec,
        [reg_ptr, int_name, meas_name](Real, Real, std::span<Real> out) {
            const Real integral = reg_ptr->get(int_name);
            const Real measure = reg_ptr->get(meas_name);
            out[0] = (measure > 0.0) ? integral / measure : 0.0;
        });
    reg.addDependency(input_name, integral_name);
    reg.addDependency(input_name, measure_name);

    feQuantityRegistry().registerDefinition(*def);
    return AuxiliaryInputHandle(input_name, std::move(def));
}

AuxiliaryInputHandle FESystem::feExpression(
    const std::string& input_name,
    forms::FormExpr expression,
    AuxiliaryInputUpdateSchedule schedule)
{
    FE_THROW_IF(input_name.empty(), InvalidArgumentException,
                "FESystem::feExpression: empty input_name");

    std::vector<FieldId> refs;
    if (const auto* root = expression.node()) {
        gatherFieldIds(*root, refs);
    }

    auto def = std::make_shared<FEQuantityDefinition>();
    def->name = input_name;
    def->kind = FEQuantityKind::FEExpression;
    def->shape = FEQuantityShape::scalar();
    def->referenced_fields = refs;
    def->expression = expression;
    def->capabilities.explicit_evaluation = true;
    // FE expressions that reference fields support monolithic linearization
    // through the same domain-functional gradient assembly path.
    def->capabilities.monolithic_linearization = !refs.empty();

    // Use the domain-functional path (same as domainIntegral) so the
    // expression gets proper FE evaluation with quadrature and field
    // binding, AND supports symbolic gradient assembly for dI/du.
    FieldId primary_fid = INVALID_FIELD_ID;
    if (!refs.empty()) {
        primary_fid = refs.front();
    } else {
        const auto& recs = field_registry_.records();
        if (!recs.empty()) primary_fid = static_cast<FieldId>(0);
    }

    if (primary_fid != INVALID_FIELD_ID) {
        // Register as a domain functional through BoundaryReductionService.
        auto& svc = boundaryReductionService(primary_fid);
        forms::BoundaryFunctional domain_func;
        domain_func.name = input_name;
        domain_func.integrand = expression;
        domain_func.boundary_marker = -1;
        domain_func.reduction = forms::BoundaryFunctional::Reduction::Sum;
        domain_func.is_domain_functional = true;
        svc.addBoundaryFunctional(domain_func);
        bindSecondaryFields(svc, primary_fid, refs);

        auto& reg = auxiliaryInputRegistry();
        AuxiliaryInputSpec spec;
        spec.name = input_name;
        spec.size = 1;
        spec.producer = AuxiliaryInputProducer::DomainIntegral;
        spec.update_schedule = schedule;
        spec.requires_mpi_reduction = true;

        const auto captured_fid = primary_fid;
        const std::string func_name = input_name;

        reg.registerInput(spec,
            [this, func_name, captured_fid]
            (Real time, Real dt, std::span<Real> out) {
                SystemStateView state;
                state.time = time;
                state.dt = dt;
                state.u = cached_solution_u_;
                state.u_vector = cached_solution_vector_;
                state.u_prev = cached_solution_u_prev_;
                state.u_prev_vector = cached_solution_prev_vector_;
                state.time_integration = cached_time_integration_;
                state.user_data = cached_user_data_;

                auto it = boundary_reduction_services_.find(captured_fid);
                if (it != boundary_reduction_services_.end() && it->second) {
                    out[0] = it->second->evaluateFunctional(func_name, state);
                } else {
                    out[0] = 0.0;
                }
            });
    } else {
        // No field references: use PointEvaluator as a simple callback.
        auto& reg = auxiliaryInputRegistry();
        AuxiliaryInputSpec spec;
        spec.name = input_name;
        spec.size = 1;
        spec.producer = AuxiliaryInputProducer::FormulationCallback;
        spec.update_schedule = schedule;

        auto captured_expr = std::move(expression);

        reg.registerInput(spec,
            [this, captured_expr](Real time, Real dt, std::span<Real> out) {
                forms::PointEvalContext pctx;
                pctx.time = time;
                pctx.dt = dt;
                if (auxiliary_input_registry_) {
                    pctx.auxiliary_inputs = auxiliary_input_registry_->all();
                }
                out[0] = forms::evaluateScalarAt(captured_expr, pctx);
            });
    }

    feQuantityRegistry().registerDefinition(*def);
    return AuxiliaryInputHandle(input_name, std::move(def));
}

void FESystem::advanceAuxiliaryState(const SystemStateView& state)
{
    advanceAuxiliaryState(state, /*is_nonlinear_iteration=*/false);
}

void FESystem::advanceAuxiliaryState(const SystemStateView& state,
                                     bool is_nonlinear_iteration)
{
    // Cache the full system state so boundary-integral input callbacks
    // (and other FE-coupled callbacks) can access the current solution.
    cacheSystemState(state);

    // Pre-refresh inputs using the caller's nonlinear-iteration semantics.
    // The Real/Real overload will reuse the cached values and no-op for
    // clean OncePerTimeStep inputs.
    if (auxiliary_input_registry_) {
        auxiliary_input_registry_->evaluate(
            static_cast<Real>(state.time),
            static_cast<Real>(state.dt),
            is_nonlinear_iteration);
    }

    advanceAuxiliaryState(static_cast<Real>(state.time), static_cast<Real>(state.dt));
}

void FESystem::initializeAuxiliaryDAEBlocksIfNeeded_(Real time, Real dt)
{
    if (!auxiliary_state_manager_) {
        return;
    }

    for (auto& entry : deployed_aux_entries_) {
        if (!entry.materialized ||
            entry.consistent_initialization_done ||
            !entry.model ||
            !entry.deriv_provider ||
            !auxiliary_state_manager_->hasBlock(entry.instance_name) ||
            !needsConsistentAuxiliaryInitialization(
                *entry.model,
                static_cast<std::size_t>(entry.spec.size),
                entry.spec.solve_mode)) {
            continue;
        }

        auto& blk = auxiliary_state_manager_->getBlock(entry.instance_name);
        auto params = buildParamVector(entry);
        auto bound_inputs = buildInputVector(entry);

        bool has_entity_local_inputs = false;
        if (auxiliary_input_registry_) {
            for (const auto& [model_name, reg_name] : entry.input_bindings) {
                (void)model_name;
                if (auxiliary_input_registry_->hasInput(reg_name) &&
                    auxiliary_input_registry_->isEntityLocal(reg_name)) {
                    has_entity_local_inputs = true;
                    break;
                }
            }
        }

        const auto& emap = entry.entity_map;
        const auto n_authoritative_entities = blk.ownedEntityCount();

        for (std::size_t e = 0; e < n_authoritative_entities; ++e) {
            auto x = blk.gatherEntityWork(e);
            validatePartitionedAuxiliaryEntityWidth_(
                "FESystem::initializeAuxiliaryDAEBlocksIfNeeded",
                entry,
                blk,
                e,
                x.size(),
                "state");
            const auto orig_e = emap.empty() ? e : emap[e];

            if (has_entity_local_inputs && auxiliary_input_registry_) {
                bound_inputs.clear();
                if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(entry.model.get())) {
                    for (const auto& inp : built->signature().inputs) {
                        auto bind_it = entry.input_bindings.find(inp.name);
                        if (bind_it != entry.input_bindings.end()) {
                            auto vals = auxiliary_input_registry_->valuesOf(bind_it->second, orig_e);
                            bound_inputs.insert(bound_inputs.end(), vals.begin(), vals.end());
                        } else {
                            bound_inputs.resize(
                                bound_inputs.size() + static_cast<std::size_t>(inp.size),
                                Real(0.0));
                        }
                    }
                } else {
                    rebuildGenericInputsForEntity(entry, orig_e, bound_inputs);
                }
            }

            const auto result = AuxiliaryInitializationSolver::solve(
                *entry.model,
                *entry.deriv_provider,
                x,
                bound_inputs,
                params,
                time,
                InitializationOptions{},
                IndexReductionHook{},
                e);
            FE_THROW_IF(!result.converged, InvalidStateException,
                        "FESystem::initializeAuxiliaryDAEBlocksIfNeeded: auxiliary block '" +
                            entry.instance_name + "' failed consistent initialization for entity " +
                            std::to_string(e) + " at time " + std::to_string(time) +
                            " with residual norm " +
                            std::to_string(result.final_residual_norm));

            blk.scatterEntityWork(e, x);
        }

        if (blk.ownedEntityCount() < blk.entityCount()) {
            auxiliary_state_manager_->syncGhosts(entry.instance_name);
        }
        const auto initialized = blk.work();
        blk.initialize(initialized);
        entry.consistent_initialization_done = true;

        if (monolithicAuxTraceEnabled()) {
            FE_LOG_INFO("FESystem: consistent-initialized auxiliary block '" +
                        entry.instance_name + "' at time=" + std::to_string(time) +
                        " dt=" + std::to_string(dt));
        }
    }
}

void FESystem::validatePartitionedAuxiliaryEntityWidth_(
    const char* caller,
    const DeployedAuxEntry& entry,
    const AuxiliaryBlockStorage& blk,
    std::size_t entity_index,
    std::size_t actual_width,
    const char* slice_name) const
{
    const auto expected_width = static_cast<std::size_t>(entry.spec.size);
    FE_THROW_IF(actual_width != expected_width, InvalidArgumentException,
                std::string(caller) + ": partitioned auxiliary block '" +
                    entry.instance_name + "' " + slice_name + " slice for entity " +
                    std::to_string(entity_index) + " has " +
                    (blk.layoutMode() == AuxiliaryLayoutMode::Ragged
                         ? std::string("ragged ")
                         : std::string("")) +
                    "width " + std::to_string(actual_width) +
                    " but the AuxiliaryStateModel dimension is " +
                    std::to_string(expected_width) +
                    ". The current partitioned runtime requires every per-entity "
                    "slice to match the fixed model dimension; use uniform "
                    "ragged widths equal to the model dimension or provide a "
                    "variable-width model/runtime contract.");
}

void FESystem::advanceAuxiliaryState(Real time, Real dt)
{
    last_auxiliary_advance_time_ = time + dt;

    if (!auxiliary_state_manager_) return;

    // Ensure auxiliary inputs are evaluated before stepping reads them.
    if (auxiliary_input_registry_) {
        auxiliary_input_registry_->evaluate(time, dt);
    }

    initializeAuxiliaryDAEBlocksIfNeeded_(time, dt);

    // Check if any block uses Multirate scheduling (interleaved time ordering).
    bool has_multirate = false;
    for (const auto& entry : deployed_aux_entries_) {
        if (!entry.materialized) {
            continue;
        }
        if (entry.spec.solve_mode == AuxiliarySolveMode::Partitioned &&
            entry.spec.schedule_mode == AuxiliaryScheduleMode::Multirate) {
            has_multirate = true;
            break;
        }
    }

    if (has_multirate && aux_scheduler_) {
        // Multirate dispatch: use planSubsteps() for interleaved cross-block
        // time ordering.  Each substep advances one block by one dt_sub using
        // advanceFromWork(), which does NOT reset from committed state.
        auto plan = aux_scheduler_->planSubsteps(time, dt);

        // Track per-block x_prev buffers for advanceFromWork().
        // x_prev starts as committed state for the first substep of each block.
        std::unordered_map<std::string, std::vector<Real>> block_x_prev;

        for (const auto& ss : plan) {
            // Find the entry for this block.
            DeployedAuxEntry* ep = nullptr;
            for (auto& entry : deployed_aux_entries_) {
                if (!entry.materialized) {
                    continue;
                }
                if (entry.instance_name == ss.block_name &&
                    entry.spec.solve_mode == AuxiliarySolveMode::Partitioned &&
                    entry.stepper && entry.deriv_provider) {
                    ep = &entry;
                    break;
                }
            }
            if (!ep) continue;

            auto& blk = auxiliary_state_manager_->getBlock(ep->instance_name);
            auto params = buildParamVector(*ep);
            auto bound_inputs = buildInputVector(*ep);
            const auto n_entities = blk.entityCount();
            const auto& emap = ep->entity_map;

            // Detect entity-local inputs (same as standard path).
            bool has_entity_local = false;
            if (auxiliary_input_registry_) {
                for (const auto& [mn, rn] : ep->input_bindings) {
                    if (auxiliary_input_registry_->hasInput(rn) &&
                        auxiliary_input_registry_->isEntityLocal(rn)) {
                        has_entity_local = true;
                        break;
                    }
                }
            }

            for (std::size_t e = 0; e < n_entities; ++e) {
                auto ew = blk.gatherEntityWork(e);
                validatePartitionedAuxiliaryEntityWidth_(
                    "FESystem::advanceAuxiliaryState",
                    *ep,
                    blk,
                    e,
                    ew.size(),
                    "work");
                const auto orig_e = emap.empty() ? e : emap[e];

                // Initialize x_prev from committed on first substep.
                auto key = ep->instance_name + "_" + std::to_string(e);
                auto it = block_x_prev.find(key);
                if (it == block_x_prev.end()) {
                    auto ec = blk.gatherEntityCommitted(e);
                    validatePartitionedAuxiliaryEntityWidth_(
                        "FESystem::advanceAuxiliaryState",
                        *ep,
                        blk,
                        e,
                        ec.size(),
                        "committed");
                    block_x_prev[key] = std::vector<Real>(ec.begin(), ec.end());
                    std::copy(block_x_prev[key].begin(), block_x_prev[key].end(), ew.begin());
                }
                auto& x_prev = block_x_prev[key];

                // Rebuild inputs per entity when entity-local.
                if (has_entity_local && auxiliary_input_registry_) {
                    if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(ep->model.get())) {
                        bound_inputs.clear();
                        for (const auto& inp : built->signature().inputs) {
                            auto bi = ep->input_bindings.find(inp.name);
                            if (bi != ep->input_bindings.end()) {
                                auto v = auxiliary_input_registry_->valuesOf(bi->second, orig_e);
                                bound_inputs.insert(bound_inputs.end(), v.begin(), v.end());
                            } else {
                                bound_inputs.resize(bound_inputs.size() + static_cast<std::size_t>(inp.size), 0.0);
                            }
                        }
                    } else {
                        rebuildGenericInputsForEntity(*ep, orig_e, bound_inputs);
                    }
                }

                // Build history spans (same as standard path).
                std::vector<std::vector<Real>> hd;
                std::vector<std::span<const Real>> hs;
                for (std::size_t k = 0; k < blk.history().depth(); ++k) {
                    hd.push_back(blk.gatherEntityHistory(k, e));
                    validatePartitionedAuxiliaryEntityWidth_(
                        "FESystem::advanceAuxiliaryState",
                        *ep,
                        blk,
                        e,
                        hd.back().size(),
                        "history");
                    hs.push_back(hd.back());
                }

                const auto ew_start = ew;
                const auto x_prev_start = x_prev;
                AuxiliaryStepResult step_result{};
                bool converged = false;
                const int max_attempts =
                    std::max(0, ep->spec.failure_policy.max_local_retries) + 1;
                for (int attempt = 0; attempt < max_attempts; ++attempt) {
                    ew = ew_start;
                    x_prev = x_prev_start;
                    step_result = ep->stepper->advanceFromWork(
                        *ep->model, *ep->deriv_provider,
                        ew, x_prev,
                        hs, bound_inputs, params,
                        ss.t_start, ss.dt_sub, e);
                    if (step_result.converged) {
                        converged = true;
                        break;
                    }
                }
                if (!converged) {
                    if (ep->spec.failure_policy.reject_timestep_on_failure) {
                        FE_THROW(InvalidStateException,
                                 "FESystem::advanceAuxiliaryState: partitioned auxiliary block '" +
                                 ep->instance_name + "' failed to converge for entity " +
                                 std::to_string(e) + " after " +
                                 std::to_string(max_attempts) + " attempt(s); final residual norm=" +
                                 std::to_string(step_result.final_residual_norm));
                    }
                    ew = ew_start;
                    x_prev = x_prev_start;
                } else {
                    if (e < ep->event_managers.size() && ep->event_managers[e]) {
                        auto& event_manager = *ep->event_managers[e];
                        const auto events = event_manager.detectEvents(
                            *ep->model, x_prev_start, ew,
                            ss.t_start, ss.dt_sub, bound_inputs, params, hs, e);
                        for (const auto& event : events) {
                            event_manager.applyTransition(
                                *ep->model, event, ew, event.event_time,
                                bound_inputs, params, hs, e, ss.dt_sub);
                        }
                    }
                    std::copy(ew.begin(), ew.end(), x_prev.begin());
                }
                blk.scatterEntityWork(e, ew);
            }
        }
    } else {
        // Standard dispatch: each partitioned block advances once for the
        // full dt.  The stepper's substep_count handles Subcycled scheduling.
        for (auto& entry : deployed_aux_entries_) {
            if (!entry.materialized) {
                continue;
            }
            if (entry.spec.solve_mode != AuxiliarySolveMode::Partitioned) continue;
            if (!entry.stepper || !entry.deriv_provider) continue;
            advanceOneEntry(entry, time, dt, entry.stepper_spec.substep_count);
        }
    }

    // Partitioned blocks update their local work buffers directly.  Refresh
    // ghost copies before any downstream assembly reads node-scoped data.
    auxiliary_state_manager_->syncGhosts();

    partitioned_auxiliary_advance_valid_ = true;
    partitioned_auxiliary_advance_time_ = time;
    partitioned_auxiliary_advance_dt_ = dt;
}

void FESystem::advanceOneEntry(DeployedAuxEntry& entry, Real time, Real dt, int substep_count)
{
    auto& blk = auxiliary_state_manager_->getBlock(entry.instance_name);

    auto params = buildParamVector(entry);
    auto bound_inputs = buildInputVector(entry);

    const auto n_entities = blk.entityCount();
    const auto& emap = entry.entity_map;

    bool has_entity_local_inputs = false;
    if (auxiliary_input_registry_) {
        for (const auto& [mn, rn] : entry.input_bindings) {
            if (auxiliary_input_registry_->hasInput(rn) && auxiliary_input_registry_->isEntityLocal(rn)) {
                has_entity_local_inputs = true;
                break;
            }
        }
    }

    for (std::size_t e = 0; e < n_entities; ++e) {
        auto ew = blk.gatherEntityWork(e);
        auto ec = blk.gatherEntityCommitted(e);
        validatePartitionedAuxiliaryEntityWidth_(
            "FESystem::advanceAuxiliaryState",
            entry,
            blk,
            e,
            ew.size(),
            "work");
        validatePartitionedAuxiliaryEntityWidth_(
            "FESystem::advanceAuxiliaryState",
            entry,
            blk,
            e,
            ec.size(),
            "committed");
        const auto orig_e = emap.empty() ? e : emap[e];

        if (has_entity_local_inputs && auxiliary_input_registry_) {
            bound_inputs.clear();
            if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(entry.model.get())) {
                for (const auto& inp : built->signature().inputs) {
                    auto bi = entry.input_bindings.find(inp.name);
                    if (bi != entry.input_bindings.end()) {
                        auto v = auxiliary_input_registry_->valuesOf(bi->second, orig_e);
                        bound_inputs.insert(bound_inputs.end(), v.begin(), v.end());
                    } else {
                        bound_inputs.resize(bound_inputs.size() + static_cast<std::size_t>(inp.size), 0.0);
                    }
                }
            } else {
                rebuildGenericInputsForEntity(entry, orig_e, bound_inputs);
            }
        }

        std::vector<std::vector<Real>> hd;
        std::vector<std::span<const Real>> hs;
        for (std::size_t k = 0; k < blk.history().depth(); ++k) {
            hd.push_back(blk.gatherEntityHistory(k, e));
            validatePartitionedAuxiliaryEntityWidth_(
                "FESystem::advanceAuxiliaryState",
                entry,
                blk,
                e,
                hd.back().size(),
                "history");
            hs.push_back(hd.back());
        }

        AuxiliaryStepResult step_result{};
        bool converged = false;
        const int max_attempts =
            std::max(0, entry.spec.failure_policy.max_local_retries) + 1;
        for (int attempt = 0; attempt < max_attempts; ++attempt) {
            std::copy(ec.begin(), ec.end(), ew.begin());
            step_result = entry.stepper->advance(*entry.model, *entry.deriv_provider,
                                                  ew, ec, hs, bound_inputs, params,
                                                  time, dt, substep_count, e);
            if (step_result.converged) {
                converged = true;
                break;
            }
        }
        if (!converged) {
            if (entry.spec.failure_policy.reject_timestep_on_failure) {
                FE_THROW(InvalidStateException,
                         "FESystem::advanceAuxiliaryState: partitioned auxiliary block '" +
                         entry.instance_name + "' failed to converge for entity " +
                         std::to_string(e) + " after " +
                         std::to_string(max_attempts) + " attempt(s); final residual norm=" +
                         std::to_string(step_result.final_residual_norm));
            }
            std::copy(ec.begin(), ec.end(), ew.begin());
        }
        if (converged && e < entry.event_managers.size() && entry.event_managers[e]) {
            auto& event_manager = *entry.event_managers[e];
            const auto events = event_manager.detectEvents(
                *entry.model, ec, ew, time, dt, bound_inputs, params, hs, e);
            for (const auto& event : events) {
                event_manager.applyTransition(
                    *entry.model, event, ew, event.event_time, bound_inputs, params, hs, e, dt);
            }
        }
        blk.scatterEntityWork(e, ew);
    }
}

std::vector<Real> FESystem::buildParamVector(const DeployedAuxEntry& entry) const
{
    std::vector<Real> params;
    auto declared_params = entry.model->declaredParameterNames();
    if (!declared_params.empty()) {
        params.resize(declared_params.size(), 0.0);
        for (std::size_t i = 0; i < declared_params.size(); ++i) {
            auto it = entry.param_values.find(declared_params[i]);
            if (it != entry.param_values.end()) params[i] = it->second;
        }
    } else if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(entry.model.get())) {
        const auto& sig = built->signature();
        params.resize(sig.parameters.size(), 0.0);
        for (std::size_t i = 0; i < sig.parameters.size(); ++i) {
            auto it = entry.param_values.find(sig.parameters[i].name);
            if (it != entry.param_values.end()) params[i] = it->second;
        }
    } else {
        for (const auto& [pname, pval] : entry.param_values)
            params.push_back(pval);
    }
    return params;
}

std::vector<Real> FESystem::buildInputVector(const DeployedAuxEntry& entry) const
{
    std::vector<Real> bound_inputs;
    if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(entry.model.get())) {
        const auto& sig = built->signature();
        std::size_t total = 0;
        for (const auto& inp : sig.inputs) total += static_cast<std::size_t>(inp.size);
        bound_inputs.resize(total, 0.0);
        std::size_t off = 0;
        for (const auto& inp : sig.inputs) {
            auto bi = entry.input_bindings.find(inp.name);
            if (bi != entry.input_bindings.end() && auxiliary_input_registry_) {
                auto v = auxiliary_input_registry_->valuesOf(bi->second);
                for (std::size_t k = 0; k < std::min(v.size(), static_cast<std::size_t>(inp.size)); ++k)
                    bound_inputs[off + k] = v[k];
            }
            off += static_cast<std::size_t>(inp.size);
        }
    } else {
        auto decl = entry.model->declaredInputNames();
        if (!decl.empty() && auxiliary_input_registry_) {
            for (const auto& raw : decl) {
                auto [iname, input_size] = parseDeclaredInputName(raw);
                auto bi = entry.input_bindings.find(iname);
                if (bi != entry.input_bindings.end() && auxiliary_input_registry_->hasInput(bi->second)) {
                    auto v = auxiliary_input_registry_->valuesOf(bi->second);
                    for (int k = 0; k < input_size; ++k) {
                        bound_inputs.push_back(
                            k < static_cast<int>(v.size()) ? v[static_cast<std::size_t>(k)] : 0.0);
                    }
                } else {
                    bound_inputs.resize(bound_inputs.size() + static_cast<std::size_t>(input_size), 0.0);
                }
            }
        } else if (!entry.input_bindings.empty() && auxiliary_input_registry_) {
            for (const auto& [mn, rn] : entry.input_bindings) {
                if (auxiliary_input_registry_->hasInput(rn)) {
                    auto v = auxiliary_input_registry_->valuesOf(rn);
                    bound_inputs.insert(bound_inputs.end(), v.begin(), v.end());
                }
            }
        }
    }
    return bound_inputs;
}

void FESystem::lowerAuxiliaryConstraintBindings_()
{
    if (lowered_auxiliary_constraint_offset_ != std::numeric_limits<std::size_t>::max() &&
        lowered_auxiliary_constraint_offset_ <= system_constraint_defs_.size()) {
        system_constraint_defs_.resize(lowered_auxiliary_constraint_offset_);
    }
    lowered_auxiliary_constraint_offset_ = system_constraint_defs_.size();

    for (const auto& entry : deployed_aux_entries_) {
        if (!entry.materialized) {
            continue;
        }
        for (const auto& binding : entry.constraint_bindings) {
            if (binding.kind != AuxiliaryConstraintKind::StrongDirichlet) {
                FE_THROW(NotImplementedException,
                         "FESystem::lowerAuxiliaryConstraintBindings_: unsupported auxiliary "
                         "constraint kind on instance '" + entry.instance_name + "'");
            }
            system_constraint_defs_.push_back(
                std::make_unique<constraints::AuxiliaryDrivenDirichletConstraint>(
                    entry.instance_name,
                    binding));
        }
    }
}

const FESystem::DeployedAuxEntry& FESystem::findDeployedAuxEntry_(
    std::string_view instance_name) const
{
    auto it = std::find_if(
        deployed_aux_entries_.begin(),
        deployed_aux_entries_.end(),
        [&](const auto& entry) { return entry.instance_name == instance_name; });
    FE_THROW_IF(it == deployed_aux_entries_.end(), InvalidArgumentException,
                "FESystem: unknown auxiliary instance '" + std::string(instance_name) + "'");
    return *it;
}

std::vector<std::span<const Real>> FESystem::buildHistorySpans_(
    const AuxiliaryBlockStorage& blk,
    std::size_t entity_index,
    std::vector<std::vector<Real>>& storage) const
{
    storage.clear();
    storage.reserve(blk.history().depth());

    std::vector<std::span<const Real>> spans;
    spans.reserve(blk.history().depth());
    for (std::size_t k = 0; k < blk.history().depth(); ++k) {
        storage.push_back(blk.gatherEntityHistory(k, entity_index));
        spans.push_back(storage.back());
    }
    return spans;
}

std::pair<std::string, int> FESystem::parseDeclaredInputName(const std::string& raw)
{
    FE_THROW_IF(raw.empty(), InvalidArgumentException,
                "Declared input name is empty");

    auto colon = raw.find(':');
    if (colon == std::string::npos)
        return {raw, 1};

    auto base = raw.substr(0, colon);
    FE_THROW_IF(base.empty(), InvalidArgumentException,
                "Declared input name '" + raw +
                "': base name before ':' must not be empty");

    auto size_str = raw.substr(colon + 1);
    int sz = 0;
    std::size_t pos = 0;
    try {
        sz = std::stoi(size_str, &pos);
    } catch (const std::exception&) {
        FE_THROW(InvalidArgumentException,
                 "Declared input name '" + raw +
                 "': suffix after ':' must be a positive integer, got '" +
                 size_str + "'");
    }
    FE_THROW_IF(pos != size_str.size(), InvalidArgumentException,
                "Declared input name '" + raw +
                "': suffix after ':' must be a positive integer, got '" +
                size_str + "' (trailing characters)");
    FE_THROW_IF(sz < 1, InvalidArgumentException,
                "Declared input name '" + raw +
                "': size must be >= 1, got " + std::to_string(sz));
    return {base, sz};
}

void FESystem::validateDeclaredInputNames(const AuxiliaryStateModel& model)
{
    for (const auto& raw : model.declaredInputNames()) {
        parseDeclaredInputName(raw); // throws on malformed suffix
    }
}

void FESystem::rebuildGenericInputsForEntity(
    const DeployedAuxEntry& entry, std::size_t entity_index,
    std::vector<Real>& out) const
{
    out.clear();
    auto decl = entry.model->declaredInputNames();
    if (!decl.empty() && auxiliary_input_registry_) {
        for (const auto& raw : decl) {
            auto [iname, input_size] = parseDeclaredInputName(raw);
            auto bi = entry.input_bindings.find(iname);
            if (bi != entry.input_bindings.end() && auxiliary_input_registry_->hasInput(bi->second)) {
                auto v = auxiliary_input_registry_->valuesOf(bi->second, entity_index);
                for (int k = 0; k < input_size; ++k) {
                    out.push_back(
                        k < static_cast<int>(v.size()) ? v[static_cast<std::size_t>(k)] : 0.0);
                }
            } else {
                out.resize(out.size() + static_cast<std::size_t>(input_size), 0.0);
            }
        }
    } else {
        for (const auto& [mn, rn] : entry.input_bindings) {
            if (auxiliary_input_registry_ && auxiliary_input_registry_->hasInput(rn)) {
                auto v = auxiliary_input_registry_->valuesOf(rn, entity_index);
                out.insert(out.end(), v.begin(), v.end());
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  FE-coupled auxiliary input providers
// ---------------------------------------------------------------------------

void FESystem::wireFECoupledInputProviders()
{
    // No-op: FE-coupled input providers are registered by the caller
    // before finalization via registerSampledFieldInput() etc.
}

void FESystem::registerSampledFieldInput(
    const std::string& input_name,
    const std::string& field_name,
    std::size_t n_entities)
{
    auto& reg = auxiliaryInputRegistry();

    // Look up the field.  Requires setup() to have been called.
    const FieldId fid = field_registry_.findByName(field_name);
    FE_THROW_IF(fid == INVALID_FIELD_ID, InvalidArgumentException,
                "registerSampledFieldInput: unknown field '" + field_name + "'");
    const auto fidx_check = static_cast<std::size_t>(fid);
    FE_THROW_IF(fidx_check >= field_dof_handlers_.size(), InvalidStateException,
                "registerSampledFieldInput: must be called after setup() "
                "so field DOF handlers are available");
    {
        const auto* emap = field_dof_handlers_[fidx_check].getEntityDofMap();
        FE_THROW_IF(!emap || emap->numVertices() == 0, InvalidStateException,
                    "registerSampledFieldInput: field '" + field_name +
                    "' has no entity DOF map");
        const auto test_dofs = emap->getVertexDofs(0);
        FE_THROW_IF(test_dofs.empty(), InvalidArgumentException,
                    "registerSampledFieldInput: field '" + field_name +
                    "' has no vertex DOFs (requires vertex-based Lagrange space)");
    }
    const int components = field_registry_.get(fid).components;

    AuxiliaryInputSpec spec;
    spec.name = input_name;
    spec.size = components;
    spec.entity_count = n_entities;
    spec.producer = AuxiliaryInputProducer::SampledStateField;
    spec.field_stage = AuxiliaryFieldStage::CurrentIterate;
    spec.source_field_name = field_name;
    spec.update_schedule = AuxiliaryInputUpdateSchedule::EachNonlinearIteration;

    const auto field_idx = static_cast<std::size_t>(fid);
    const auto cap_comp = components;
    reg.registerEntityInput(spec,
        [this, field_idx, cap_comp]
        (Real /*t*/, Real /*dt*/, std::size_t entity_id, std::span<Real> out) {
            // Use per-field DOF handler and field-specific offset, matching
            // the logic in evaluateFieldAtVertices().
            std::fill(out.begin(), out.end(), 0.0);
            if (field_idx >= field_dof_handlers_.size()) return;

            const auto* emap = field_dof_handlers_[field_idx].getEntityDofMap();
            if (!emap) return;

            auto dofs = emap->getVertexDofs(static_cast<GlobalIndex>(entity_id));
            const GlobalIndex offset = (field_idx < field_dof_offsets_.size())
                ? field_dof_offsets_[field_idx] : 0;

            // Read from backend vector if available (MPI), else from cached span.
            for (int c = 0; c < cap_comp && c < static_cast<int>(out.size()); ++c) {
                if (c < static_cast<int>(dofs.size())) {
                    const GlobalIndex d = dofs[static_cast<std::size_t>(c)] + offset;
                    if (cached_solution_vector_) {
                        // MPI/distributed path: use backend vector for global access.
                        auto* vec = const_cast<backends::GenericVector*>(cached_solution_vector_);
                        auto view = vec->createAssemblyView();
                        out[static_cast<std::size_t>(c)] = view->getVectorEntry(d);
                    } else if (static_cast<std::size_t>(d) < cached_solution_u_.size()) {
                        out[static_cast<std::size_t>(c)] = cached_solution_u_[static_cast<std::size_t>(d)];
                    }
                }
            }
        });
}

void FESystem::registerBoundaryNodalSumInput(
    const std::string& input_name,
    const std::string& field_name,
    int boundary_marker)
{
    auto& reg = auxiliaryInputRegistry();

    const FieldId fid = field_registry_.findByName(field_name);
    FE_THROW_IF(fid == INVALID_FIELD_ID, InvalidArgumentException,
                "registerBoundaryNodalSumInput: unknown field '" + field_name + "'");

    // Validate vertex-DOF precondition: this helper requires setup() to have
    // been called (so DOF handlers are built) and the field to have vertex DOFs.
    const auto fidx = static_cast<std::size_t>(fid);
    FE_THROW_IF(fidx >= field_dof_handlers_.size(), InvalidStateException,
                "registerBoundaryNodalSumInput: must be called after setup() "
                "so field DOF handlers are available");
    const auto* emap = field_dof_handlers_[fidx].getEntityDofMap();
    FE_THROW_IF(!emap || emap->numVertices() == 0, InvalidStateException,
                "registerBoundaryNodalSumInput: field '" + field_name +
                "' has no entity DOF map (setup may not have completed)");
    {
        const auto test_dofs = emap->getVertexDofs(0);
        FE_THROW_IF(test_dofs.empty(), InvalidArgumentException,
                    "registerBoundaryNodalSumInput: field '" + field_name +
                    "' has no vertex DOFs (requires vertex-based Lagrange space)");
    }

    const int components = field_registry_.get(fid).components;

    AuxiliaryInputSpec spec;
    spec.name = input_name;
    spec.size = std::max(1, components);
    spec.producer = AuxiliaryInputProducer::SampledBoundaryReduction;
    spec.field_stage = AuxiliaryFieldStage::CurrentIterate;
    spec.boundary_marker = boundary_marker;
    spec.source_field_name = field_name;

    const auto field_idx = static_cast<std::size_t>(fid);
    const auto cap_marker = boundary_marker;
    reg.registerInput(spec,
        [this, field_idx, cap_marker]
        (Real /*t*/, Real /*dt*/, std::span<Real> out) {
            // Boundary-face nodal reduction: sum all field DOF components
            // at unique boundary face vertices.
            //
            // This is a nodal sum (not a quadrature-weighted boundary
            // integral).  For a true boundary integral, use
            // BoundaryFunctional + the assembly pipeline instead.
            // The output size equals the number of field components.
            const auto ncomp = static_cast<std::size_t>(
                field_registry_.get(static_cast<FieldId>(field_idx)).components);
            std::fill(out.begin(), out.end(), 0.0);
            if (!mesh_access_ || field_idx >= field_dof_handlers_.size()) return;

            const auto* emap = field_dof_handlers_[field_idx].getEntityDofMap();
            if (!emap) return;

            const GlobalIndex fld_offset = (field_idx < field_dof_offsets_.size())
                ? field_dof_offsets_[field_idx] : 0;

            std::unique_ptr<assembly::GlobalSystemView> solution_view;
            if (cached_solution_vector_) {
                auto* vec = const_cast<backends::GenericVector*>(cached_solution_vector_);
                solution_view = vec->createAssemblyView();
            }

            // Face-vertex maps for supported element types.
            static const std::vector<std::vector<int>> tet_faces =
                {{1,2,3}, {0,3,2}, {0,1,3}, {0,2,1}};
            static const std::vector<std::vector<int>> tri_faces =
                {{0,1}, {1,2}, {2,0}};
            static const std::vector<std::vector<int>> hex_faces =
                {{0,3,2,1}, {4,5,6,7}, {0,1,5,4},
                 {1,2,6,5}, {2,3,7,6}, {3,0,4,7}};
            static const std::vector<std::vector<int>> quad_faces =
                {{0,1}, {1,2}, {2,3}, {3,0}};

            auto getFaceMap = [](ElementType et) -> const std::vector<std::vector<int>>* {
                if (et == ElementType::Tetra4) return &tet_faces;
                if (et == ElementType::Triangle3) return &tri_faces;
                if (et == ElementType::Hex8) return &hex_faces;
                if (et == ElementType::Quad4) return &quad_faces;
                return nullptr;
            };

            std::unordered_set<GlobalIndex> visited;
            mesh_access_->forEachBoundaryFace(cap_marker,
                [&](GlobalIndex face_id, GlobalIndex cell_id) {
                    const auto local_face = mesh_access_->getLocalFaceIndex(face_id, cell_id);
                    std::vector<GlobalIndex> cell_nodes;
                    mesh_access_->getCellNodes(cell_id, cell_nodes);

                    const auto* fmap = getFaceMap(mesh_access_->getCellType(cell_id));
                    if (!fmap || local_face < 0 ||
                        static_cast<std::size_t>(local_face) >= fmap->size()) {
                        return; // Skip unsupported element types.
                    }

                    const auto& local_ids = (*fmap)[static_cast<std::size_t>(local_face)];
                    for (int li : local_ids) {
                        if (static_cast<std::size_t>(li) >= cell_nodes.size()) continue;
                        const auto node_id = cell_nodes[static_cast<std::size_t>(li)];
                        if (!visited.insert(node_id).second) continue;

                        auto dofs = emap->getVertexDofs(node_id);
                        for (std::size_t c = 0; c < std::min(ncomp, dofs.size()); ++c) {
                            const GlobalIndex d = dofs[c] + fld_offset;
                            Real val = 0.0;
                            if (solution_view) {
                                val = solution_view->getVectorEntry(d);
                            } else if (static_cast<std::size_t>(d) < cached_solution_u_.size()) {
                                val = cached_solution_u_[static_cast<std::size_t>(d)];
                            }
                            if (c < out.size()) out[c] += val;
                        }
                    }
                });
        });
}

// ---------------------------------------------------------------------------
//  Boundary reduction service
// ---------------------------------------------------------------------------

BoundaryReductionService& FESystem::boundaryReductionService(FieldId primary_field)
{
    auto& svc = boundary_reduction_services_[primary_field];
    if (!svc) {
        svc = std::make_unique<BoundaryReductionService>(*this, primary_field);
    }
    return *svc;
}

// ---------------------------------------------------------------------------
//  registerBoundaryIntegralInput
// ---------------------------------------------------------------------------

namespace {

[[nodiscard]] assembly::AuxiliaryOutputScope toAssemblyAuxiliaryOutputScope(
    AuxiliaryStateScope scope) noexcept
{
    switch (scope) {
        case AuxiliaryStateScope::Global:
            return assembly::AuxiliaryOutputScope::Global;
        case AuxiliaryStateScope::Boundary:
            return assembly::AuxiliaryOutputScope::Boundary;
        case AuxiliaryStateScope::Cell:
            return assembly::AuxiliaryOutputScope::Cell;
        case AuxiliaryStateScope::QuadraturePoint:
            return assembly::AuxiliaryOutputScope::QuadraturePoint;
        case AuxiliaryStateScope::Region:
            return assembly::AuxiliaryOutputScope::Region;
        case AuxiliaryStateScope::Facet:
            return assembly::AuxiliaryOutputScope::Facet;
        case AuxiliaryStateScope::Node:
            return assembly::AuxiliaryOutputScope::Node;
    }
    return assembly::AuxiliaryOutputScope::Global;
}


} // namespace

void FESystem::registerBoundaryIntegralInput(
    const std::string& input_name,
    forms::BoundaryFunctional functional,
    AuxiliaryInputUpdateSchedule schedule)
{
    FE_THROW_IF(input_name.empty(), InvalidArgumentException,
                "registerBoundaryIntegralInput: empty input_name");
    FE_THROW_IF(!functional.integrand.isValid(), InvalidArgumentException,
                "registerBoundaryIntegralInput: invalid integrand");
    FE_THROW_IF(functional.boundary_marker < 0, InvalidArgumentException,
                "registerBoundaryIntegralInput: boundary_marker must be >= 0");

    // The functional's name defaults to the input_name if not set.
    if (functional.name.empty()) {
        functional.name = input_name;
    }

    // Determine the primary field by scanning the integrand for field references.
    std::vector<FieldId> referenced_fields;
    if (const auto* root = functional.integrand.node()) {
        gatherFieldIds(*root, referenced_fields);
    }

    // Multi-field integrands are supported via secondary field bindings.
    // The primary field provides the DOF layout and mesh context; secondary
    // fields contribute solution data through the functional assembler's
    // field binding mechanism.

    FieldId primary_fid = INVALID_FIELD_ID;
    if (!referenced_fields.empty()) {
        primary_fid = referenced_fields.front();
    } else {
        // No field references in integrand (e.g., constant or geometry-only
        // integrand like ∫_Γ 1 ds).  The integrand doesn't depend on DOFs,
        // but quadrature requires a function space.  Use GEOMETRY_FIELD_ID
        // as a logical sentinel — resolved to the first registered field's
        // space for quadrature rule selection only.
        primary_fid = GEOMETRY_FIELD_ID;
    }

    // Resolve GEOMETRY_FIELD_ID: prefer the first registered field (for DOF
    // access in field-dependent code paths), but allow GEOMETRY_FIELD_ID to
    // pass through when no fields exist (geometry-only evaluation with a
    // default P1 space).
    if (primary_fid == GEOMETRY_FIELD_ID) {
        const auto& recs = field_registry_.records();
        if (!recs.empty()) {
            // Use first field for richer DOF access; the integrand doesn't
            // reference it, so only the quadrature rule matters.
            primary_fid = static_cast<FieldId>(0);
        }
        // else: keep GEOMETRY_FIELD_ID — BoundaryReductionService will
        // create a default P1 space from the mesh element type.
    }
    FE_THROW_IF(primary_fid == INVALID_FIELD_ID, InvalidStateException,
                "registerBoundaryIntegralInput('" + input_name +
                    "'): internal error — could not resolve primary field");

    // Register the functional with the per-field boundary reduction service.
    auto& svc = boundaryReductionService(primary_fid);
    svc.addBoundaryFunctional(functional);

    // Bind secondary fields and set dof_per_node for multi-field evaluation.
    bindSecondaryFields(svc, primary_fid, referenced_fields);

    // Register the input in the AuxiliaryInputRegistry with a callback
    // that evaluates the functional via the BoundaryReductionService.
    auto& reg = auxiliaryInputRegistry();

    AuxiliaryInputSpec spec;
    spec.name = input_name;
    spec.size = 1;  // boundary integrals are scalar
    spec.producer = AuxiliaryInputProducer::BoundaryReduction;
    spec.update_schedule = schedule;
    spec.boundary_marker = functional.boundary_marker;
    spec.requires_mpi_reduction = true;  // MPI reduction is handled inside the service

    const auto func_name = functional.name;
    const auto captured_fid = primary_fid;

    reg.registerInput(spec,
        [this, func_name, captured_fid]
        (Real time, Real dt, std::span<Real> out) {
            // Build a SystemStateView from the full cached system state.
            // cacheSystemState() is called by prepareAuxiliaryForAssembly(),
            // advanceAuxiliaryState(SystemStateView), and
            // assembleMixedAuxiliaryIntoGlobal() before evaluate() is invoked.
            SystemStateView state;
            state.time = time;
            state.dt = dt;
            state.u = cached_solution_u_;
            state.u_vector = cached_solution_vector_;
            state.u_prev = cached_solution_u_prev_;
            state.u_prev_vector = cached_solution_prev_vector_;
            state.u_prev2 = cached_solution_u_prev2_;
            state.u_prev2_vector = cached_solution_prev2_vector_;
            state.time_integration = cached_time_integration_;
            state.user_data = cached_user_data_;

            auto it = boundary_reduction_services_.find(captured_fid);
            if (it != boundary_reduction_services_.end() && it->second) {
                out[0] = it->second->evaluateFunctional(func_name, state);
            } else {
                out[0] = 0.0;
            }

            if (monolithicAuxTraceEnabled()) {
                Real local_u_sq = 0.0;
                for (const auto value : state.u) {
                    local_u_sq += value * value;
                }
                std::ostringstream oss;
                oss << "boundaryIntegralInput: name='" << func_name
                    << "' time=" << time
                    << " dt=" << dt
                    << " local_u_l2=" << std::sqrt(local_u_sq)
                    << " local_u_size=" << state.u.size()
                    << " value=" << out[0];
                FE_LOG_INFO(oss.str());
            }
        });
}

void FESystem::registerBoundaryIntegralInput(
    const std::string& input_name,
    forms::FormExpr integrand,
    int boundary_marker,
    forms::BoundaryFunctional::Reduction reduction,
    AuxiliaryInputUpdateSchedule schedule)
{
    forms::BoundaryFunctional functional;
    functional.name = input_name;
    functional.integrand = std::move(integrand);
    functional.boundary_marker = boundary_marker;
    functional.reduction = reduction;

    registerBoundaryIntegralInput(input_name, std::move(functional), schedule);
}

// ---------------------------------------------------------------------------
//  Mixed monolithic assembly into global system
// ---------------------------------------------------------------------------

void FESystem::assembleMixedAuxiliaryIntoGlobal(
    const SystemStateView& state,
    assembly::GlobalSystemView* matrix_out,
    assembly::GlobalSystemView* vector_out,
    bool want_matrix, bool want_vector,
    std::size_t n_field_dofs,
    bool is_nonlinear_iteration)
{
    if (!auxiliary_state_manager_ || !auxiliary_operator_registry_) return;
    if (!auxiliary_operator_registry_->isLayoutFinalized()) return;

    const auto mixed = auxiliary_operator_registry_->composeMixedLayout(n_field_dofs);
    const Real aux_dt = effectiveAuxiliaryDt(state);

    // Cache the full system state for FE-coupled input callbacks.
    cacheSystemState(state);

    // Evaluate inputs with nonlinear-iteration flag.
    if (auxiliary_input_registry_) {
        auxiliary_input_registry_->evaluate(state.time, state.dt, is_nonlinear_iteration);
    }

    std::vector<Real> dense_solution_storage;
    std::span<const Real> dense_solution = state.u;
    if (dense_solution.empty() && state.u_vector && n_field_dofs > 0) {
        auto view = const_cast<backends::GenericVector*>(state.u_vector)->createAssemblyView();
        if (view) {
            dense_solution_storage.resize(n_field_dofs, Real(0.0));
            for (std::size_t i = 0; i < n_field_dofs; ++i) {
                dense_solution_storage[i] =
                    view->getVectorEntry(static_cast<GlobalIndex>(i));
            }
            dense_solution = dense_solution_storage;
        }
    }

    if (want_matrix) {
        clearReducedFieldUpdates();
        clearLocalCondensedRecovery();
    } else if (want_vector) {
        if (!last_local_condensed_records_.empty()) {
            last_local_condensed_rhs_shift_.assign(n_field_dofs, Real(0.0));
            for (auto& rec : last_local_condensed_records_) {
                std::fill(rec.g.begin(), rec.g.end(), Real(0.0));
            }
        } else {
            last_local_condensed_rhs_shift_.clear();
        }
    }

    auto addSparseEntry = [](std::vector<std::pair<GlobalIndex, Real>>& entries,
                             GlobalIndex dof,
                             Real value) {
        if (std::abs(value) <= Real(1e-30)) {
            return;
        }
        for (auto& [existing_dof, existing_value] : entries) {
            if (existing_dof == dof) {
                existing_value += value;
                return;
            }
        }
        entries.emplace_back(dof, value);
    };

    auto isSingleOwnerBorderedScope = [](AuxiliaryStateScope scope) noexcept {
        return scope == AuxiliaryStateScope::Global ||
               scope == AuxiliaryStateScope::Boundary;
    };

    auto currentMpiRank = [&]() -> int {
#if FE_HAS_MPI
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (mpi_initialized) {
            int rank = 0;
            MPI_Comm_rank(dof_handler_.mpiComm(), &rank);
            return rank;
        }
#endif
        return 0;
    };

    auto singleOwnerRankForAuxBlock = [&](std::string_view block_name) -> int {
        for (const auto& bl : mixed.aux_layout.blocks) {
            if (bl.name == block_name &&
                bl.row_ownership == backends::MixedRowOwnershipPolicy::SingleOwner &&
                bl.single_owner_rank >= 0) {
                return bl.single_owner_rank;
            }
        }
        return 0;
    };

    auto useSingleOwnerBorderedAssembly =
        [&](const DeployedAuxEntry& aux_entry,
            bool aux_local_condensed,
            bool aux_direct_only) noexcept {
            return !aux_local_condensed && !aux_direct_only &&
                   isSingleOwnerBorderedScope(aux_entry.spec.scope);
        };

    auto useOwnerRoutedBorderedAssembly =
        [&](const DeployedAuxEntry& aux_entry,
            bool aux_local_condensed,
            bool aux_direct_only) noexcept {
        return !aux_local_condensed && !aux_direct_only &&
                   (aux_entry.spec.scope == AuxiliaryStateScope::Node ||
                    aux_entry.spec.scope == AuxiliaryStateScope::Region);
    };

    auto isSingleOwnerBorderedAssemblyRank =
        [&](const DeployedAuxEntry& aux_entry) {
            return currentMpiRank() ==
                   singleOwnerRankForAuxBlock(aux_entry.instance_name);
        };

    auto borderedAuxRowIndex = [&](GlobalIndex aux_dof) -> std::optional<std::size_t> {
        const auto nf = static_cast<GlobalIndex>(n_field_dofs);
        if (aux_dof < nf) {
            return std::nullopt;
        }
        const auto row = static_cast<std::size_t>(aux_dof - nf);
        if (row >= static_cast<std::size_t>(bordered_coupling_.n_aux)) {
            return std::nullopt;
        }
        return row;
    };

    auto ownsBorderedAuxRow = [&](GlobalIndex aux_dof) {
        const auto row = borderedAuxRowIndex(aux_dof);
        if (!row.has_value() ||
            *row >= bordered_coupling_.aux_row_owner_routed.size() ||
            bordered_coupling_.aux_row_owner_routed[*row] == char{0}) {
            return true;
        }
        return *row < bordered_coupling_.aux_row_owner_ranks.size() &&
               bordered_coupling_.aux_row_owner_ranks[*row] == currentMpiRank();
    };

    auto markBorderedAuxRowContribution = [&](GlobalIndex aux_dof) {
        const auto row = borderedAuxRowIndex(aux_dof);
        if (!row.has_value() ||
            *row >= bordered_coupling_.aux_row_owner_routed.size() ||
            bordered_coupling_.aux_row_owner_routed[*row] == char{0} ||
            *row >= bordered_coupling_.aux_row_local_contribution_flags.size()) {
            return;
        }
        bordered_coupling_.aux_row_local_contribution_flags[*row] = 1;
    };

    auto deploymentOrdinal = [&](std::string_view block_name) -> std::size_t {
        for (std::size_t i = 0; i < deployed_aux_entries_.size(); ++i) {
            if (deployed_aux_entries_[i].instance_name == block_name) {
                return i;
            }
        }
        return 0u;
    };

    auto qpCellOrdinalForEntity =
        [](const DeployedAuxEntry& aux_entry,
           std::size_t entity_index) -> std::size_t {
        if (aux_entry.qp_offsets.size() < 2u) {
            return 0u;
        }
        auto upper = std::upper_bound(aux_entry.qp_offsets.begin(),
                                      aux_entry.qp_offsets.end(),
                                      entity_index);
        if (upper == aux_entry.qp_offsets.begin()) {
            return 0u;
        }
        auto cell_ord = static_cast<std::size_t>(
            std::distance(aux_entry.qp_offsets.begin(), upper) - 1);
        if (cell_ord + 1u >= aux_entry.qp_offsets.size()) {
            cell_ord = aux_entry.qp_offsets.size() - 2u;
        }
        return cell_ord;
    };

    auto originalEntityForAuxEntity =
        [&](const DeployedAuxEntry& aux_entry,
            std::size_t entity_index) -> std::size_t {
        if (aux_entry.spec.scope == AuxiliaryStateScope::QuadraturePoint) {
            const auto cell_ord = qpCellOrdinalForEntity(aux_entry, entity_index);
            return aux_entry.entity_map.empty()
                ? cell_ord
                : aux_entry.entity_map.at(cell_ord);
        }
        return aux_entry.entity_map.empty()
            ? entity_index
            : aux_entry.entity_map.at(entity_index);
    };

    auto globalEntityKeyForAuxEntity =
        [&](const DeployedAuxEntry& aux_entry,
            std::size_t entity_index) -> std::uint64_t {
        const auto orig_entity = originalEntityForAuxEntity(aux_entry, entity_index);
        if (aux_entry.spec.scope == AuxiliaryStateScope::QuadraturePoint) {
            const auto cell_ord = qpCellOrdinalForEntity(aux_entry, entity_index);
            const auto qp_begin = aux_entry.qp_offsets.at(cell_ord);
            const auto q_local = entity_index - qp_begin;
            return (static_cast<std::uint64_t>(orig_entity) << 32u) |
                   static_cast<std::uint64_t>(q_local);
        }
        return static_cast<std::uint64_t>(orig_entity);
    };

    auto ownsLocalCondensedAuxEquation =
        [&](const DeployedAuxEntry& aux_entry,
            const AuxiliaryBlockStorage&,
            std::size_t,
            std::size_t orig_entity) {
        if (!aux_entry.local_condensed ||
            aux_entry.spec.scope != AuxiliaryStateScope::Node) {
            return true;
        }
        return nodeAuxiliaryOwnerRank_(orig_entity) == currentMpiRank();
    };

    auto findLocalCondensedRecord =
        [&](std::string_view block_name,
            std::size_t entity_index,
            std::uint64_t global_entity_key) -> LocalCondensedEntityRecord* {
            for (auto& rec : last_local_condensed_records_) {
                if (rec.block_name == block_name &&
                    rec.entity_index == entity_index &&
                    rec.global_entity_key == global_entity_key) {
                    return &rec;
                }
            }
            return nullptr;
        };

    auto ensureLocalCondensedRecord =
        [&](std::string_view block_name,
            std::size_t entity_index,
            std::size_t block_ordinal,
            std::uint64_t global_entity_key,
            int dim) -> LocalCondensedEntityRecord& {
            if (auto* existing =
                    findLocalCondensedRecord(block_name, entity_index, global_entity_key)) {
                return *existing;
            }
            last_local_condensed_records_.push_back(LocalCondensedEntityRecord{});
            auto& rec = last_local_condensed_records_.back();
            rec.block_name = std::string(block_name);
            rec.entity_index = entity_index;
            rec.block_ordinal = block_ordinal;
            rec.global_entity_key = global_entity_key;
            rec.B_columns.resize(static_cast<std::size_t>(dim));
            rec.Ct_rows.resize(static_cast<std::size_t>(dim));
            rec.g.assign(static_cast<std::size_t>(dim), Real(0.0));
            return rec;
        };

    auto buildEntityFieldValues =
        [&](const DeployedAuxEntry& entry,
            std::size_t orig_e,
            std::span<const Real> solution) {
            std::vector<FieldValueEntry> field_values;
            if (!entry.deriv_provider) {
                return field_values;
            }
            const auto& artifact = entry.deriv_provider->artifact();
            if (artifact.referenced_fields.empty()) {
                return field_values;
            }
            field_values.reserve(artifact.referenced_fields.size());
            for (const auto fid : artifact.referenced_fields) {
                const auto fidx = static_cast<std::size_t>(fid);
                if (fidx >= field_dof_offsets_.size() ||
                    fidx >= field_dof_handlers_.size()) {
                    continue;
                }
                const auto fld_off = field_dof_offsets_[fidx];
                const auto* femap = field_dof_handlers_[fidx].getEntityDofMap();
                if (!femap) {
                    continue;
                }
                auto vdofs = femap->getVertexDofs(static_cast<GlobalIndex>(orig_e));
                if (vdofs.empty()) {
                    continue;
                }
                FieldValueEntry fve;
                fve.field = fid;
                fve.n_components = static_cast<int>(vdofs.size());
                for (int c = 0; c < fve.n_components && c < MAX_FIELD_VALUE_COMPONENTS; ++c) {
                    const auto gidx = static_cast<std::size_t>(
                        vdofs[static_cast<std::size_t>(c)] + fld_off);
                    fve.components[c] =
                        (gidx < solution.size()) ? solution[gidx] : Real(0.0);
                }
                field_values.push_back(fve);
            }
            return field_values;
        };

    // For each monolithic auxiliary block, assemble its per-entity
    // contributions into the global matrix/vector at the auxiliary DOF offsets.
    // This matches the standalone assembleMonolithicAuxiliary() logic for
    // entity-local inputs, xdot computation, and input refresh.
    for (auto& entry : deployed_aux_entries_) {
        if (!entry.materialized) continue;
        if (entry.spec.solve_mode != AuxiliarySolveMode::Monolithic) continue;
        if (entry.lower_to_direct_only) continue;
        if (!auxiliary_state_manager_->hasBlock(entry.instance_name)) continue;

        auto& blk = auxiliary_state_manager_->getBlock(entry.instance_name);
        const int dim = entry.spec.size;
        const auto n_entities = blk.entityCount();
        const bool local_condensed = entry.local_condensed;

        // Find this block's offset in the mixed layout.
        std::size_t block_offset = 0;
        if (!local_condensed) {
            for (const auto& bl : mixed.aux_layout.blocks) {
                if (bl.name == entry.instance_name) {
                    block_offset = bl.offset + mixed.aux_layout.mixed_system_offset;
                    break;
                }
            }
        }

        auto params = buildParamVector(entry);
        auto bound_inputs = buildInputVector(entry);

        // Detect entity-local inputs (same as standalone monolithic path).
        bool has_entity_local_inputs = false;
        if (auxiliary_input_registry_) {
            for (const auto& [mn, rn] : entry.input_bindings) {
                if (auxiliary_input_registry_->hasInput(rn) &&
                    auxiliary_input_registry_->isEntityLocal(rn)) {
                    has_entity_local_inputs = true;
                    break;
                }
            }
        }

        const auto block_ordinal = deploymentOrdinal(entry.instance_name);

        for (std::size_t e = 0; e < n_entities; ++e) {
            auto entity_x = blk.gatherEntityWork(e);
            auto entity_committed = blk.gatherEntityCommitted(e);
            const auto orig_e = originalEntityForAuxEntity(entry, e);
            const auto global_entity_key = globalEntityKeyForAuxEntity(entry, e);
            if (local_condensed &&
                !ownsLocalCondensedAuxEquation(entry, blk, e, orig_e)) {
                continue;
            }
            const auto entity_committed_rate = gatherMonolithicCommittedRate(entry, e);
            auto temporal = buildMonolithicAuxiliaryTemporalEvaluation(
                entry.stepper_spec, blk, e, entity_x, entity_committed, entity_committed_rate, state);

            // Rebuild inputs per entity when entity-local bindings exist.
            if (has_entity_local_inputs && auxiliary_input_registry_) {
                bound_inputs.clear();
                if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(entry.model.get())) {
                    for (const auto& inp : built->signature().inputs) {
                        auto bi = entry.input_bindings.find(inp.name);
                        if (bi != entry.input_bindings.end()) {
                            auto v = auxiliary_input_registry_->valuesOf(bi->second, orig_e);
                            bound_inputs.insert(bound_inputs.end(), v.begin(), v.end());
                        } else {
                            bound_inputs.resize(bound_inputs.size() + static_cast<std::size_t>(inp.size), 0.0);
                        }
                    }
                } else {
                    rebuildGenericInputsForEntity(entry, orig_e, bound_inputs);
                }
            }

            // Populate field_values when the model directly references FE fields.
            // Needed for both residual evaluation and Jacobian evaluation via
            // PointEvaluator, which encounters DiscreteField/StateField terminals.
            std::vector<FieldValueEntry> field_vals;
            if (entry.deriv_provider) {
                const auto& art = entry.deriv_provider->artifact();
                if (!art.referenced_fields.empty()) {
                    field_vals.reserve(art.referenced_fields.size());
                    for (const auto fid : art.referenced_fields) {
                        const auto fidx = static_cast<std::size_t>(fid);
                        if (fidx >= field_dof_offsets_.size() ||
                            fidx >= field_dof_handlers_.size()) continue;
                        const auto fld_off = field_dof_offsets_[fidx];
                        const auto* femap = field_dof_handlers_[fidx].getEntityDofMap();
                        if (!femap) continue;
                        auto vdofs = femap->getVertexDofs(static_cast<GlobalIndex>(orig_e));
                        if (!vdofs.empty()) {
                            FieldValueEntry fve;
                            fve.field = fid;
                            fve.n_components = static_cast<int>(vdofs.size());
                            for (int c = 0; c < fve.n_components && c < MAX_FIELD_VALUE_COMPONENTS; ++c) {
                                const auto gidx = static_cast<std::size_t>(vdofs[static_cast<std::size_t>(c)] + fld_off);
                                fve.components[c] = (gidx < state.u.size()) ? state.u[gidx] : 0.0;
                            }
                            field_vals.push_back(fve);
                        }
                    }
                }
            }

            AuxiliaryLocalContext ctx;
            ctx.time = state.time; ctx.dt = state.dt; ctx.effective_dt = aux_dt;
            ctx.x = entity_x; ctx.xdot = temporal.xdot;
            ctx.history = temporal.history_spans;
            ctx.inputs = bound_inputs; ctx.params = params;
            ctx.entity_index = e;
            ctx.field_values = field_vals;
            ctx.user_data = state.user_data;

            // Build global DOF indices for this entity's auxiliary unknowns.
            std::vector<GlobalIndex> aux_dofs(static_cast<std::size_t>(dim));
            if (!local_condensed) {
                for (int i = 0; i < dim; ++i) {
                    aux_dofs[static_cast<std::size_t>(i)] = static_cast<GlobalIndex>(
                        block_offset + e * static_cast<std::size_t>(dim) +
                        static_cast<std::size_t>(i));
                }
            }

            std::vector<Real> entity_res;
            const bool need_entity_residual =
                want_vector || (local_condensed && want_matrix && !dense_solution.empty());

            // Residual.
            if (need_entity_residual) {
                entity_res.resize(static_cast<std::size_t>(dim));
                AuxiliaryResidualRequest res_req;
                res_req.residual = entity_res;
                entry.model->evaluateResidual(ctx, res_req);
                if (monolithicAuxTraceEnabled()) {
                    std::ostringstream oss;
                    oss << "FESystem: monolithic aux residual"
                        << " block='" << entry.instance_name << "'"
                        << " entity=" << e
                        << " time=" << ctx.time
                        << " dt=" << ctx.dt
                        << " effective_dt=" << ctx.effective_dt
                        << " x=" << formatTraceVector(ctx.x)
                        << " xdot=" << formatTraceVector(ctx.xdot)
                        << " inputs=" << formatTraceVector(ctx.inputs)
                        << " residual=" << formatTraceVector(entity_res);
                    FE_LOG_INFO(oss.str());
                }
                if (want_vector) {
                    if (local_condensed) {
                        auto& rec = ensureLocalCondensedRecord(
                            entry.instance_name, e, block_ordinal, global_entity_key, dim);
                        rec.has_aux_equation_terms = true;
                        rec.g = entity_res;
                    } else if (vector_out) {
                        vector_out->addVectorEntries(aux_dofs, entity_res);
                    }
                }
            }

            // Jacobian (aux-aux self-coupling block).
            if (want_matrix && entry.deriv_provider) {
                const auto n_inp = static_cast<int>(bound_inputs.size());
                std::vector<Real> entity_jac(static_cast<std::size_t>(dim * dim));
                std::vector<Real> entity_dFdi(static_cast<std::size_t>(dim * n_inp));
                std::vector<Real> entity_dFdxdot(static_cast<std::size_t>(dim * dim), 0.0);

                AuxiliaryJacobianRequest jac_req;
                jac_req.dF_dx = entity_jac;
                jac_req.n = dim;
                jac_req.want_dF_dxdot = true;
                jac_req.dF_dxdot = entity_dFdxdot;
                // Request dF/dinputs for chain-rule coupling.
                if (n_inp > 0 && !entry.coupled_bindings.empty()) {
                    jac_req.dF_dinputs = entity_dFdi;
                    jac_req.n_inputs = n_inp;
                }
                entry.deriv_provider->evaluateJacobian(*entry.model, ctx, jac_req);

                if (!local_condensed &&
                    !entity_dFdxdot.empty() && bordered_coupling_.active &&
                    bordered_coupling_.dF_dxdot.size() ==
                        static_cast<std::size_t>(bordered_coupling_.n_aux * bordered_coupling_.n_aux)) {
                    const auto na = static_cast<std::size_t>(bordered_coupling_.n_aux);
                    for (int r = 0; r < dim; ++r) {
                        const auto row_dof = aux_dofs[static_cast<std::size_t>(r)];
                        if (!ownsBorderedAuxRow(row_dof)) {
                            continue;
                        }
                        const auto aux_i = static_cast<std::size_t>(
                            row_dof - static_cast<GlobalIndex>(n_field_dofs));
                        if (aux_i >= na) continue;
                        for (int c = 0; c < dim; ++c) {
                            const auto aux_j = static_cast<std::size_t>(
                                aux_dofs[static_cast<std::size_t>(c)] - static_cast<GlobalIndex>(n_field_dofs));
                            if (aux_j >= na) continue;
                            bordered_coupling_.dF_dxdot[aux_i * na + aux_j] +=
                                entity_dFdxdot[static_cast<std::size_t>(r * dim + c)];
                            if (std::abs(entity_dFdxdot[static_cast<std::size_t>(r * dim + c)]) >
                                Real(1e-30)) {
                                markBorderedAuxRowContribution(row_dof);
                            }
                        }
                    }
                }

                // Add the current-stage time discretization contribution:
                // J = dF/dx + (d xdot / d x_current) * dF/dxdot.
                if (temporal.dxdot_dx_coeff != Real(0.0)) {
                    for (std::size_t i = 0; i < entity_jac.size(); ++i) {
                        entity_jac[i] += temporal.dxdot_dx_coeff * entity_dFdxdot[i];
                    }
                }
                if (!local_condensed && matrix_out) {
                    // Store dF/dinputs in bordered data for B computation.
                    if (!entity_dFdi.empty()) {
                        bordered_coupling_.dF_dinputs.assign(entity_dFdi.begin(), entity_dFdi.end());
                    }
                    matrix_out->addMatrixEntries(aux_dofs, aux_dofs, entity_jac);
                } else if (local_condensed) {
                    auto& rec = ensureLocalCondensedRecord(
                        entry.instance_name, e, block_ordinal, global_entity_key, dim);
                    rec.has_aux_equation_terms = true;
                    rec.D_inv.clear();
                    if (!invertDenseMatrix(entity_jac, static_cast<std::size_t>(dim), rec.D_inv)) {
                        FE_THROW(InvalidStateException,
                                 "FESystem::assembleMixedAuxiliaryIntoGlobal: failed to invert local condensed block '" +
                                     entry.instance_name + "' for entity " + std::to_string(e));
                    }
                    rec.Ct_rows.assign(static_cast<std::size_t>(dim), {});
                    rec.B_columns.assign(static_cast<std::size_t>(dim), {});
                }

                // Chain-rule coupling: dF/du = dF/dI * dI/du.
                // For each coupled binding, compute the field-auxiliary
                // Jacobian block and insert it into the global matrix.
                if (n_inp > 0 && !entity_dFdi.empty()) {
                    auto visitCoupledInputComponents = [&](auto&& fn) {
                        if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(entry.model.get())) {
                            int input_col = 0;
                            for (const auto& inp : built->signature().inputs) {
                                auto cb_it = entry.coupled_bindings.find(inp.name);
                                for (int ic = 0; ic < inp.size; ++ic) {
                                    if (cb_it != entry.coupled_bindings.end()) {
                                        fn(inp.name, cb_it->second, input_col + ic);
                                    }
                                }
                                input_col += inp.size;
                            }
                            return;
                        }

                        auto decl = entry.model->declaredInputNames();
                        if (!decl.empty()) {
                            int input_col = 0;
                            for (const auto& raw : decl) {
                                auto [iname, input_size] = parseDeclaredInputName(raw);
                                auto cb_it = entry.coupled_bindings.find(iname);
                                for (int ic = 0; ic < input_size; ++ic) {
                                    if (cb_it != entry.coupled_bindings.end()) {
                                        fn(iname, cb_it->second, input_col + ic);
                                    }
                                }
                                input_col += input_size;
                            }
                            return;
                        }

                        int input_col = 0;
                        for (const auto& [model_input, reg_input] : entry.input_bindings) {
                            int input_size = 1;
                            if (auxiliary_input_registry_ &&
                                auxiliary_input_registry_->hasInput(reg_input)) {
                                input_size = auxiliary_input_registry_->specOf(reg_input).size;
                            }
                            auto cb_it = entry.coupled_bindings.find(model_input);
                            for (int ic = 0; ic < input_size; ++ic) {
                                if (cb_it != entry.coupled_bindings.end()) {
                                    fn(model_input, cb_it->second, input_col + ic);
                                }
                            }
                            input_col += input_size;
                        }
                    };

                    visitCoupledInputComponents([&](const std::string& /*model_input*/,
                                                    const AuxiliaryInputHandle& handle,
                                                    int input_col) {
                        if (input_col < 0 || input_col >= n_inp) {
                            return;
                        }
                        if (handle.hasDefinition() &&
                            handle.supportsMonolithicLinearization()) {
                            // For sampled fields, dI/du is identity at sampled DOFs.
                            // For boundary integrals, dI/du comes from the
                            // BoundaryReductionService gradient assembly.
                            //
                            // For now, sampled-field chain rule is implemented:
                            // dF/du_j = dF/dI_k * delta(k, DOF_j)
                            // = dF/dI column for the k-th input, scattered to field DOFs.
                            if (handle.kind() == FEQuantityKind::SampledField) {
                                const auto& ref_fields = handle.referencedFields();
                                if (!ref_fields.empty()) {
                                    const auto fid = ref_fields[0];
                                    const auto fidx = static_cast<std::size_t>(fid);
                                    if (fidx < field_dof_offsets_.size() &&
                                        fidx < field_dof_handlers_.size()) {
                                        const auto fld_off = field_dof_offsets_[fidx];
                                        const auto* emap = field_dof_handlers_[fidx].getEntityDofMap();
                                        if (emap) {
                                            // dI/du for sampled field = identity at vertex DOFs.
                                            // Use the actual DOF map for vertex e.
                                            auto vertex_dofs = emap->getVertexDofs(
                                                static_cast<GlobalIndex>(orig_e));
                                            // Extract dF/dI column for this input.
                                            std::vector<Real> col(static_cast<std::size_t>(dim));
                                            for (int r = 0; r < dim; ++r) {
                                                col[static_cast<std::size_t>(r)] =
                                                    entity_dFdi[static_cast<std::size_t>(
                                                        r * n_inp + input_col)];
                                            }
                                            // Each vertex DOF gets a column of dF/dI.
                                            for (const auto local_dof : vertex_dofs) {
                                                const auto global_dof = static_cast<GlobalIndex>(
                                                    local_dof + fld_off);
                                                if (local_condensed) {
                                                    auto& rec = ensureLocalCondensedRecord(
                                                        entry.instance_name, e, block_ordinal,
                                                        global_entity_key, dim);
                                                    rec.has_aux_equation_terms = true;
                                                    for (int r = 0; r < dim; ++r) {
                                                        addSparseEntry(
                                                            rec.Ct_rows[static_cast<std::size_t>(r)],
                                                            global_dof,
                                                            col[static_cast<std::size_t>(r)]);
                                                    }
                                                } else if (matrix_out) {
                                                    std::vector<GlobalIndex> fd = {global_dof};
                                                    matrix_out->addMatrixEntries(aux_dofs, fd, col);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            else if (handle.kind() == FEQuantityKind::BoundaryIntegral ||
                                     handle.kind() == FEQuantityKind::BoundaryAverage ||
                                     handle.kind() == FEQuantityKind::DomainIntegral ||
                                     handle.kind() == FEQuantityKind::DomainAverage ||
                                     handle.kind() == FEQuantityKind::RegionIntegral ||
                                     handle.kind() == FEQuantityKind::RegionAverage ||
                                     handle.kind() == FEQuantityKind::FEExpression) {
                                // Integral dI/du via symbolic gradient assembly.
                                // For average kinds (DomainAverage, RegionAverage,
                                // BoundaryAverage), the public handle name is a
                                // derived callback over __integral and __measure.
                                // Use the __integral name for gradient lookup.
                                // For DomainAverage/RegionAverage, the service only
                                // knows about the __integral sub-functional.
                                // BoundaryAverage is registered directly as a
                                // BoundaryFunctional with Reduction::Average, so
                                // its gradient is already correct without __integral.
                                std::string func_name = handle.registryName();
                                const bool is_domain_region_avg =
                                    handle.kind() == FEQuantityKind::DomainAverage ||
                                    handle.kind() == FEQuantityKind::RegionAverage;
                                if (is_domain_region_avg) {
                                    func_name = handle.registryName() + "__integral";
                                }

                                const auto& ref_fields = handle.referencedFields();
                                if (!ref_fields.empty()) {
                                    const auto svc_fid = ref_fields[0];
                                    auto svc_it = boundary_reduction_services_.find(svc_fid);
                                    if (svc_it != boundary_reduction_services_.end() && svc_it->second) {
                                        const bool topology_region_quantity =
                                            (handle.kind() == FEQuantityKind::RegionIntegral ||
                                             handle.kind() == FEQuantityKind::RegionAverage) &&
                                            handle.definition() != nullptr &&
                                            handle.definition()->region_marker < 0 &&
                                            auxiliary_input_registry_ &&
                                            auxiliary_input_registry_->isEntityLocal(handle.registryName());
                                        for (const auto target_fid : ref_fields) {
                                            auto grad = topology_region_quantity
                                                ? svc_it->second->evaluateFunctionalGradientOverCells(
                                                    func_name,
                                                    target_fid,
                                                    auxiliaryTopologyRegionCells_(orig_e),
                                                    state,
                                                    use_constraints_in_assembly_)
                                                : svc_it->second->evaluateFunctionalGradient(
                                                    func_name, target_fid, state,
                                                    use_constraints_in_assembly_);

                                            // For averages, apply quotient rule:
                                            // d(I/M)/du = (dI/du)/M  (measure M is constant w.r.t. u
                                            // for geometry-independent integrands; for u-dependent
                                            // measure, the full quotient rule would be needed).
                                            if (is_domain_region_avg && auxiliary_input_registry_) {
                                                const std::string meas_name =
                                                    handle.registryName() + "__measure";
                                                if (auxiliary_input_registry_->hasInput(meas_name)) {
                                                    Real measure = Real(0.0);
                                                    if (topology_region_quantity &&
                                                        auxiliary_input_registry_->isEntityLocal(meas_name)) {
                                                        const auto meas =
                                                            auxiliary_input_registry_->valuesOf(
                                                                meas_name, orig_e);
                                                        measure = meas.empty() ? Real(0.0) : meas[0];
                                                    } else {
                                                        measure = auxiliary_input_registry_->get(meas_name);
                                                    }
                                                    if (measure > 0.0) {
                                                        for (auto& se : grad) se.value /= measure;
                                                    }
                                                }
                                            }

                                            std::vector<std::pair<GlobalIndex, Real>> gradient_entries;
                                            gradient_entries.reserve(grad.size());
                                            for (const auto& se : grad) {
                                                gradient_entries.emplace_back(se.dof, se.value);
                                            }

                                            const bool single_owner_bordered_insert =
                                                useSingleOwnerBorderedAssembly(
                                                    entry, local_condensed, /*aux_direct_only=*/false);
                                            const bool owner_routed_bordered_insert =
                                                useOwnerRoutedBorderedAssembly(
                                                    entry, local_condensed, /*aux_direct_only=*/false);
#if FE_HAS_MPI
                                            if (single_owner_bordered_insert ||
                                                owner_routed_bordered_insert) {
                                                gradient_entries = allreduceSumSparsePairs(
                                                    std::move(gradient_entries), dof_handler_.mpiComm());
                                            }
#endif
                                            if (single_owner_bordered_insert &&
                                                !isSingleOwnerBorderedAssemblyRank(entry)) {
                                                continue;
                                            }

                                            for (const auto& [dof, dI_du] : gradient_entries) {
                                                std::vector<Real> col(static_cast<std::size_t>(dim));
                                                for (int r = 0; r < dim; ++r) {
                                                    col[static_cast<std::size_t>(r)] =
                                                        entity_dFdi[static_cast<std::size_t>(
                                                            r * n_inp + input_col)];
                                                }
                                                for (auto& c : col) c *= dI_du;
                                                if (local_condensed) {
                                                    auto& rec = ensureLocalCondensedRecord(
                                                        entry.instance_name, e, block_ordinal,
                                                        global_entity_key, dim);
                                                    rec.has_aux_equation_terms = true;
                                                    for (int r = 0; r < dim; ++r) {
                                                        addSparseEntry(
                                                            rec.Ct_rows[static_cast<std::size_t>(r)],
                                                            dof,
                                                            col[static_cast<std::size_t>(r)]);
                                                    }
                                                } else if (matrix_out) {
                                                    std::vector<GlobalIndex> field_dof = {dof};
                                                    matrix_out->addMatrixEntries(aux_dofs, field_dof, col);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    });

                    if (local_condensed && want_matrix && !dense_solution.empty()) {
                        auto& rec = ensureLocalCondensedRecord(
                            entry.instance_name, e, block_ordinal, global_entity_key, dim);
                        rec.has_aux_equation_terms = true;
                        const bool needs_ct_rows = std::all_of(
                            rec.Ct_rows.begin(),
                            rec.Ct_rows.end(),
                            [](const auto& row) { return row.empty(); });
                        if (needs_ct_rows) {
                            constexpr Real kLocalCtFdEps = Real(1e-7);
                            std::vector<Real> base_solution(
                                dense_solution.begin(),
                                dense_solution.end());
                            for (std::size_t col = 0; col < n_field_dofs; ++col) {
                                std::vector<Real> pert_solution(base_solution);
                                pert_solution[col] += kLocalCtFdEps;

                                SystemStateView pert_state = state;
                                pert_state.u = pert_solution;
                                pert_state.u_vector = nullptr;
                                cacheSystemState(pert_state);
                                if (auxiliary_input_registry_) {
                                    auxiliary_input_registry_->invalidateAll();
                                    auxiliary_input_registry_->evaluate(
                                        pert_state.time,
                                        pert_state.dt,
                                        is_nonlinear_iteration);
                                }

                                auto pert_inputs = buildInputVector(entry);
                                if (has_entity_local_inputs) {
                                    rebuildGenericInputsForEntity(entry, orig_e, pert_inputs);
                                }

                                auto pert_fv =
                                    buildEntityFieldValues(entry, orig_e, pert_solution);
                                AuxiliaryLocalContext pert_ctx = ctx;
                                pert_ctx.inputs = pert_inputs;
                                pert_ctx.field_values = pert_fv;
                                pert_ctx.user_data = state.user_data;

                                std::vector<Real> pert_res(static_cast<std::size_t>(dim), Real(0.0));
                                AuxiliaryResidualRequest pert_req;
                                pert_req.residual = pert_res;
                                entry.model->evaluateResidual(pert_ctx, pert_req);
                                for (int r = 0; r < dim; ++r) {
                                    const Real coeff =
                                        (pert_res[static_cast<std::size_t>(r)] -
                                         entity_res[static_cast<std::size_t>(r)]) /
                                        kLocalCtFdEps;
                                    if (std::abs(coeff) <= kDirectCouplingEntryTol) {
                                        continue;
                                    }
                                    addSparseEntry(
                                        rec.Ct_rows[static_cast<std::size_t>(r)],
                                        static_cast<GlobalIndex>(col),
                                        coeff);
                                }
                            }

                            cacheSystemState(state);
                            if (auxiliary_input_registry_) {
                                auxiliary_input_registry_->invalidateAll();
                                auxiliary_input_registry_->evaluate(
                                    state.time,
                                    state.dt,
                                    is_nonlinear_iteration);
                            }
                        }
                    }
                }
            }
        }
    }

    // ----------------------------------------------------------------
    // ----------------------------------------------------------------
    // Direct field-derivative block: dF_aux/du from direct FE field
    // references in auxiliary residual expressions (not mediated through
    // AuxiliaryInputRef).  This handles models that directly reference
    // DiscreteField/StateField nodes in their expressions.
    //
    // For node-scoped models with Lagrange elements, the Kronecker
    // delta property gives φ_j(vertex_i) = δ_ij, so the contribution
    // at entity e is simply dF/d(field_value) scattered to vertex e's DOF.
    // The derivative expression may itself depend on the field value
    // (nonlinear case), so we populate field_values in the context.
    // ----------------------------------------------------------------
    if (want_matrix) {
        for (auto& entry : deployed_aux_entries_) {
            if (!entry.materialized) continue;
            if (entry.spec.solve_mode != AuxiliarySolveMode::Monolithic) continue;
            if (entry.lower_to_direct_only) continue;
            if (!entry.deriv_provider) continue;

            const auto& art = entry.deriv_provider->artifact();
            if (!art.valid || art.referenced_fields.empty()) continue;

            const int dim = entry.model->dimension();
            if (dim == 0) continue;

            if (!auxiliary_state_manager_ ||
                !auxiliary_state_manager_->hasBlock(entry.instance_name)) continue;

            auto& blk = auxiliary_state_manager_->getBlock(entry.instance_name);
            const auto n_ent = blk.entityCount();
            const bool local_condensed = entry.local_condensed;

            // Find this block's offset in the mixed layout.
            std::size_t block_offset = 0;
            if (!local_condensed) {
                for (const auto& bl : mixed.aux_layout.blocks) {
                    if (bl.name == entry.instance_name) {
                        block_offset = bl.offset + mixed.aux_layout.mixed_system_offset;
                        break;
                    }
                }
            }

            auto params = buildParamVector(entry);
            auto bound_inputs = buildInputVector(entry);

            // Entity-local input handling (same as chain-rule path).
            bool has_entity_local_inputs = false;
            if (auxiliary_input_registry_) {
                for (const auto& [mn, rn] : entry.input_bindings) {
                    if (auxiliary_input_registry_->hasInput(rn) &&
                        auxiliary_input_registry_->isEntityLocal(rn)) {
                        has_entity_local_inputs = true;
                        break;
                    }
                }
            }

            const auto block_ordinal = deploymentOrdinal(entry.instance_name);

            for (std::size_t e = 0; e < n_ent; ++e) {
                auto entity_x = blk.gatherEntityWork(e);
                auto entity_committed = blk.gatherEntityCommitted(e);
                const auto orig_e = originalEntityForAuxEntity(entry, e);
                const auto global_entity_key = globalEntityKeyForAuxEntity(entry, e);
                if (local_condensed &&
                    !ownsLocalCondensedAuxEquation(entry, blk, e, orig_e)) {
                    continue;
                }
                const auto entity_committed_rate = gatherMonolithicCommittedRate(entry, e);
                auto temporal = buildMonolithicAuxiliaryTemporalEvaluation(
                    entry.stepper_spec, blk, e, entity_x, entity_committed, entity_committed_rate, state);

                // Rebuild inputs per entity when entity-local bindings exist.
                if (has_entity_local_inputs && auxiliary_input_registry_) {
                    bound_inputs.clear();
                    if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(entry.model.get())) {
                        for (const auto& inp : built->signature().inputs) {
                            auto bi = entry.input_bindings.find(inp.name);
                            if (bi != entry.input_bindings.end()) {
                                auto v = auxiliary_input_registry_->valuesOf(bi->second, orig_e);
                                bound_inputs.insert(bound_inputs.end(), v.begin(), v.end());
                            } else {
                                bound_inputs.resize(bound_inputs.size() +
                                    static_cast<std::size_t>(inp.size), 0.0);
                            }
                        }
                    } else {
                        rebuildGenericInputsForEntity(entry, orig_e, bound_inputs);
                    }
                }

                // Build field_values from the global solution for this entity.
                // For vertex-based Lagrange elements, the field value at vertex
                // orig_e is simply the DOF coefficients (Kronecker delta property).
                std::vector<FieldValueEntry> field_vals;
                field_vals.reserve(art.referenced_fields.size());
                for (const auto fid : art.referenced_fields) {
                    const auto fidx = static_cast<std::size_t>(fid);
                    if (fidx >= field_dof_offsets_.size() ||
                        fidx >= field_dof_handlers_.size()) continue;
                    const auto fld_off = field_dof_offsets_[fidx];
                    const auto* femap = field_dof_handlers_[fidx].getEntityDofMap();
                    if (!femap) continue;
                    auto vdofs = femap->getVertexDofs(static_cast<GlobalIndex>(orig_e));
                    if (!vdofs.empty()) {
                        FieldValueEntry fve;
                        fve.field = fid;
                        fve.n_components = static_cast<int>(vdofs.size());
                        for (int c = 0; c < fve.n_components && c < MAX_FIELD_VALUE_COMPONENTS; ++c) {
                            const auto gidx = static_cast<std::size_t>(vdofs[static_cast<std::size_t>(c)] + fld_off);
                            fve.components[c] = (gidx < state.u.size()) ? state.u[gidx] : 0.0;
                        }
                        field_vals.push_back(fve);
                    }
                }

                AuxiliaryLocalContext ctx;
                ctx.time = state.time; ctx.dt = state.dt; ctx.effective_dt = aux_dt;
                ctx.x = entity_x; ctx.xdot = temporal.xdot;
                ctx.history = temporal.history_spans;
                ctx.inputs = bound_inputs; ctx.params = params;
                ctx.entity_index = e;
                ctx.field_values = field_vals;
                ctx.user_data = state.user_data;

                // Build global DOF indices for this entity's auxiliary unknowns.
                std::vector<GlobalIndex> aux_dofs(static_cast<std::size_t>(dim));
                if (!local_condensed) {
                    for (int i = 0; i < dim; ++i) {
                        aux_dofs[static_cast<std::size_t>(i)] = static_cast<GlobalIndex>(
                            block_offset + e * static_cast<std::size_t>(dim) +
                            static_cast<std::size_t>(i));
                    }
                }

                // For each referenced FE field, evaluate dF/d(field_comp) and
                // scatter to per-component vertex DOFs.
                //
                // evaluateFieldDerivative returns n_rows * n_comp values,
                // row-major: [row * nc + comp].  Each vertex DOF c at vertex
                // orig_e gets the column dF_i/d(field_comp_c).
                for (const auto fid : art.referenced_fields) {
                    auto dF_dfield = entry.deriv_provider->evaluateFieldDerivative(fid, ctx);
                    if (dF_dfield.empty()) continue;

                    const auto fidx = static_cast<std::size_t>(fid);
                    if (fidx >= field_dof_offsets_.size() ||
                        fidx >= field_dof_handlers_.size()) continue;

                    const auto fld_off = field_dof_offsets_[fidx];
                    const auto* femap = field_dof_handlers_[fidx].getEntityDofMap();
                    if (!femap) continue;

                    auto vertex_dofs = femap->getVertexDofs(
                        static_cast<GlobalIndex>(orig_e));
                    const auto nc = static_cast<int>(vertex_dofs.size());

                    for (int c = 0; c < nc; ++c) {
                        const auto global_dof = static_cast<GlobalIndex>(
                            vertex_dofs[static_cast<std::size_t>(c)] + fld_off);
                        std::vector<GlobalIndex> col = {global_dof};
                        std::vector<Real> col_vals(static_cast<std::size_t>(dim));
                        for (int i = 0; i < dim; ++i) {
                            const auto idx = static_cast<std::size_t>(i * nc + c);
                            col_vals[static_cast<std::size_t>(i)] =
                                (idx < dF_dfield.size()) ? dF_dfield[idx] : 0.0;
                        }
                        if (local_condensed) {
                            auto& rec = ensureLocalCondensedRecord(
                                entry.instance_name, e, block_ordinal, global_entity_key, dim);
                            rec.has_aux_equation_terms = true;
                            for (int i = 0; i < dim; ++i) {
                                addSparseEntry(
                                    rec.Ct_rows[static_cast<std::size_t>(i)],
                                    global_dof,
                                    col_vals[static_cast<std::size_t>(i)]);
                            }
                        } else if (matrix_out) {
                            matrix_out->addMatrixEntries(aux_dofs, col, col_vals);
                        }
                    }
                }
            }
        }
    }

    // Transpose Jacobian block: dR_PDE/dx_aux.
    //
    // When PDE forms reference AuxiliaryOutput nodes, the PDE residual
    // depends on auxiliary state through the output expressions.
    // Chain rule: dR_PDE/dx_j = Σ_k (dR_PDE/d(output_k)) * (d(output_k)/dx_j)
    //
    // dR_PDE/d(output_k): computed by FD perturbation of the output value
    //   in the assembler context and re-assembling the PDE residual.
    // d(output_k)/dx_j: computed by FD perturbation of the auxiliary state
    //   and re-evaluating the output expressions.
    // ----------------------------------------------------------------
    if (want_matrix && matrix_out) {
        for (auto& entry : deployed_aux_entries_) {
            if (!entry.materialized) continue;
            const bool is_monolithic = entry.spec.solve_mode == AuxiliarySolveMode::Monolithic;
            const bool is_partitioned = entry.spec.solve_mode == AuxiliarySolveMode::Partitioned;
            if (!is_monolithic && !is_partitioned) continue;
            const bool local_condensed = is_monolithic && entry.local_condensed;
            const bool direct_only =
                (is_monolithic && entry.lower_to_direct_only) || is_partitioned;
            const auto n_outputs = static_cast<int>(entry.model->outputCount());
            const int dim = entry.model->dimension();
            if (n_outputs == 0 || dim == 0) continue;
            if (!auxiliary_state_manager_ ||
                !auxiliary_state_manager_->hasBlock(entry.instance_name)) continue;

            auto& blk = auxiliary_state_manager_->getBlock(entry.instance_name);
            const auto n_entities = blk.entityCount();
            const bool apply_monolithic_functional_constraints =
                use_constraints_in_assembly_ && is_monolithic;

            std::size_t block_offset = 0;
            if (is_monolithic && !local_condensed && !direct_only) {
                for (const auto& bl : mixed.aux_layout.blocks) {
                    if (bl.name == entry.instance_name) {
                        block_offset = bl.offset + mixed.aux_layout.mixed_system_offset;
                        break;
                    }
                }
            }

            auto params = buildParamVector(entry);
            auto bound_inputs = buildInputVector(entry);

            bool has_entity_local_inputs = false;
            if (auxiliary_input_registry_) {
                for (const auto& [mn, rn] : entry.input_bindings) {
                    if (auxiliary_input_registry_->hasInput(rn) &&
                        auxiliary_input_registry_->isEntityLocal(rn)) {
                        has_entity_local_inputs = true;
                        break;
                    }
                }
            }

            struct EntityCouplingData {
                std::vector<GlobalIndex> aux_dofs{};
                std::vector<Real> dO_dx{};
                std::vector<Real> dO_dI{};
                std::vector<Real> dO_dI_effective{};
                std::vector<Real> dF_dx{};
                std::vector<Real> dF_dinputs{};
                std::vector<std::vector<std::pair<GlobalIndex, Real>>> input_gradients{};
                std::vector<char> input_gradient_sources{};
                int n_inputs{0};
            };

            std::vector<EntityCouplingData> entity_data(n_entities);
            const auto block_ordinal = deploymentOrdinal(entry.instance_name);

            for (std::size_t e = 0; e < n_entities; ++e) {
                auto entity_x = blk.gatherEntityWork(e);
                auto entity_committed = blk.gatherEntityCommitted(e);
                const auto orig_e = originalEntityForAuxEntity(entry, e);
                const auto global_entity_key = globalEntityKeyForAuxEntity(entry, e);
                const auto entity_committed_rate = gatherMonolithicCommittedRate(entry, e);
                auto temporal = buildMonolithicAuxiliaryTemporalEvaluation(
                    entry.stepper_spec, blk, e, entity_x, entity_committed, entity_committed_rate, state);

                if (has_entity_local_inputs && auxiliary_input_registry_) {
                    bound_inputs.clear();
                    if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(entry.model.get())) {
                        for (const auto& inp : built->signature().inputs) {
                            auto bi = entry.input_bindings.find(inp.name);
                            if (bi != entry.input_bindings.end()) {
                                auto v = auxiliary_input_registry_->valuesOf(bi->second, orig_e);
                                bound_inputs.insert(bound_inputs.end(), v.begin(), v.end());
                            } else {
                                bound_inputs.resize(bound_inputs.size() +
                                    static_cast<std::size_t>(inp.size), 0.0);
                            }
                        }
                    } else {
                        rebuildGenericInputsForEntity(entry, orig_e, bound_inputs);
                    }
                }

                std::vector<FieldValueEntry> field_vals;
                if (entry.deriv_provider) {
                    const auto& art = entry.deriv_provider->artifact();
                    if (!art.referenced_fields.empty()) {
                        field_vals.reserve(art.referenced_fields.size());
                        for (const auto fid : art.referenced_fields) {
                            const auto fidx = static_cast<std::size_t>(fid);
                            if (fidx >= field_dof_offsets_.size() ||
                                fidx >= field_dof_handlers_.size()) continue;
                            const auto fld_off = field_dof_offsets_[fidx];
                            const auto* femap = field_dof_handlers_[fidx].getEntityDofMap();
                            if (!femap) continue;
                            auto vdofs = femap->getVertexDofs(static_cast<GlobalIndex>(orig_e));
                            if (!vdofs.empty()) {
                                FieldValueEntry fve;
                                fve.field = fid;
                                fve.n_components = static_cast<int>(vdofs.size());
                                for (int c = 0; c < fve.n_components && c < MAX_FIELD_VALUE_COMPONENTS; ++c) {
                                    const auto gidx = static_cast<std::size_t>(
                                        vdofs[static_cast<std::size_t>(c)] + fld_off);
                                    fve.components[c] = (gidx < state.u.size()) ? state.u[gidx] : 0.0;
                                }
                                field_vals.push_back(fve);
                            }
                        }
                    }
                }

                AuxiliaryLocalContext ctx;
                ctx.time = state.time;
                ctx.dt = state.dt;
                ctx.effective_dt = aux_dt;
                ctx.x = entity_x;
                ctx.xdot = temporal.xdot;
                ctx.history = temporal.history_spans;
                ctx.inputs = bound_inputs;
                ctx.params = params;
                ctx.entity_index = e;
                ctx.field_values = field_vals;
                ctx.user_data = state.user_data;

                auto& ed = entity_data[e];
                if (is_monolithic && !local_condensed && !direct_only) {
                    ed.aux_dofs.resize(static_cast<std::size_t>(dim));
                    for (int j = 0; j < dim; ++j) {
                        ed.aux_dofs[static_cast<std::size_t>(j)] = static_cast<GlobalIndex>(
                            block_offset + e * static_cast<std::size_t>(dim) +
                            static_cast<std::size_t>(j));
                    }
                }

                std::vector<Real> base_outputs(static_cast<std::size_t>(n_outputs), 0.0);
                entry.model->evaluateOutputs(ctx, base_outputs);

                ed.dO_dx.assign(static_cast<std::size_t>(n_outputs * dim), 0.0);
                const Real eps = 1e-7;
                if (entry.deriv_provider) {
                    const auto& art = entry.deriv_provider->artifact();
                    if (art.valid && !art.dOutput_dx_exprs.empty() &&
                        art.n_outputs == n_outputs) {
                        forms::PointEvalContext pctx;
                        pctx.time = ctx.time;
                        pctx.dt = (ctx.effective_dt > 0.0) ? ctx.effective_dt : ctx.dt;
                        pctx.coupled_aux = ctx.x;
                        pctx.auxiliary_inputs = ctx.inputs;
                        pctx.jit_constants = ctx.params;
                        for (int k = 0; k < n_outputs; ++k) {
                            for (int j = 0; j < dim; ++j) {
                                const auto idx = static_cast<std::size_t>(k * dim + j);
                                if (idx < art.dOutput_dx_exprs.size()) {
                                    ed.dO_dx[idx] = forms::evaluateScalarAt(
                                        art.dOutput_dx_exprs[idx], pctx);
                                }
                            }
                        }
                    } else {
                        std::vector<Real> x_pert(entity_x.begin(), entity_x.end());
                        AuxiliaryLocalContext pert_ctx = ctx;
                        pert_ctx.x = x_pert;
                        std::vector<Real> pert_outputs(static_cast<std::size_t>(n_outputs), 0.0);
                        for (int j = 0; j < dim; ++j) {
                            const Real orig = x_pert[static_cast<std::size_t>(j)];
                            x_pert[static_cast<std::size_t>(j)] = orig + eps;
                            entry.model->evaluateOutputs(pert_ctx, pert_outputs);
                            x_pert[static_cast<std::size_t>(j)] = orig;
                            for (int k = 0; k < n_outputs; ++k) {
                                ed.dO_dx[static_cast<std::size_t>(k * dim + j)] =
                                    (pert_outputs[static_cast<std::size_t>(k)] -
                                     base_outputs[static_cast<std::size_t>(k)]) / eps;
                            }
                        }
                    }

                    if (!art.dOutput_dInputs_exprs.empty()) {
                        forms::PointEvalContext pctx2;
                        pctx2.time = ctx.time;
                        pctx2.dt = (ctx.effective_dt > 0.0) ? ctx.effective_dt : ctx.dt;
                        pctx2.coupled_aux = ctx.x;
                        pctx2.auxiliary_inputs = ctx.inputs;
                        pctx2.jit_constants = ctx.params;
                        ed.dO_dI.resize(art.dOutput_dInputs_exprs.size(), 0.0);
                        for (std::size_t idx = 0; idx < art.dOutput_dInputs_exprs.size(); ++idx) {
                            ed.dO_dI[idx] = forms::evaluateScalarAt(
                                art.dOutput_dInputs_exprs[idx], pctx2);
                        }
                    }
                }

                bordered_coupling_.dO_dx = ed.dO_dx;
                if (!ed.dO_dI.empty()) {
                    bordered_coupling_.dO_dI = ed.dO_dI;
                }

                auto visitCoupledInputComponents = [&](auto&& fn) {
                    if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(entry.model.get())) {
                        int input_col = 0;
                        for (const auto& inp : built->signature().inputs) {
                            auto cb_it = entry.coupled_bindings.find(inp.name);
                            for (int ic = 0; ic < inp.size; ++ic) {
                                if (cb_it != entry.coupled_bindings.end()) {
                                    fn(inp.name, cb_it->second, input_col + ic);
                                }
                            }
                            input_col += inp.size;
                        }
                        return;
                    }

                    auto decl = entry.model->declaredInputNames();
                    if (!decl.empty()) {
                        int input_col = 0;
                        for (const auto& raw : decl) {
                            auto [iname, input_size] = parseDeclaredInputName(raw);
                            auto cb_it = entry.coupled_bindings.find(iname);
                            for (int ic = 0; ic < input_size; ++ic) {
                                if (cb_it != entry.coupled_bindings.end()) {
                                    fn(iname, cb_it->second, input_col + ic);
                                }
                            }
                            input_col += input_size;
                        }
                        return;
                    }

                    int input_col = 0;
                    for (const auto& [model_input, reg_input] : entry.input_bindings) {
                        int input_size = 1;
                        if (auxiliary_input_registry_ &&
                            auxiliary_input_registry_->hasInput(reg_input)) {
                            input_size = auxiliary_input_registry_->specOf(reg_input).size;
                        }
                        auto cb_it = entry.coupled_bindings.find(model_input);
                        for (int ic = 0; ic < input_size; ++ic) {
                            if (cb_it != entry.coupled_bindings.end()) {
                                fn(model_input, cb_it->second, input_col + ic);
                            }
                        }
                        input_col += input_size;
                    }
                };

                auto exactInputGradient = [&](const AuxiliaryInputHandle& handle)
                    -> std::vector<std::pair<GlobalIndex, Real>> {
                    std::vector<std::pair<GlobalIndex, Real>> out;
                    if (!(handle.hasDefinition() && handle.supportsMonolithicLinearization())) {
                        return out;
                    }

                    const auto kind = handle.kind();
                    if (kind != FEQuantityKind::BoundaryIntegral &&
                        kind != FEQuantityKind::BoundaryAverage &&
                        kind != FEQuantityKind::DomainIntegral &&
                        kind != FEQuantityKind::DomainAverage &&
                        kind != FEQuantityKind::RegionIntegral &&
                        kind != FEQuantityKind::RegionAverage &&
                        kind != FEQuantityKind::FEExpression) {
                        return out;
                    }

                    const auto& ref_fields = handle.referencedFields();
                    if (ref_fields.empty()) {
                        return out;
                    }

                    std::string func_name = handle.registryName();
                    const bool is_domain_region_avg =
                        kind == FEQuantityKind::DomainAverage ||
                        kind == FEQuantityKind::RegionAverage;
                    if (is_domain_region_avg) {
                        func_name += "__integral";
                    }
                    const bool topology_region_quantity =
                        (kind == FEQuantityKind::RegionIntegral ||
                         kind == FEQuantityKind::RegionAverage) &&
                        handle.definition() != nullptr &&
                        handle.definition()->region_marker < 0 &&
                        auxiliary_input_registry_ &&
                        auxiliary_input_registry_->isEntityLocal(handle.registryName());

                    std::unordered_map<GlobalIndex, Real> accum;
                    const auto svc_fid = ref_fields.front();
                    auto svc_it = boundary_reduction_services_.find(svc_fid);
                    if (svc_it == boundary_reduction_services_.end() || !svc_it->second) {
                        return out;
                    }

                    for (const auto target_fid : ref_fields) {
                        auto grad = topology_region_quantity
                            ? svc_it->second->evaluateFunctionalGradientOverCells(
                                func_name,
                                target_fid,
                                auxiliaryTopologyRegionCells_(orig_e),
                                state,
                                apply_monolithic_functional_constraints)
                            : svc_it->second->evaluateFunctionalGradient(
                                func_name, target_fid, state,
                                apply_monolithic_functional_constraints);

                        if (is_domain_region_avg && auxiliary_input_registry_) {
                            const std::string meas_name = handle.registryName() + "__measure";
                            if (auxiliary_input_registry_->hasInput(meas_name)) {
                                Real measure = Real(0.0);
                                if (topology_region_quantity &&
                                    auxiliary_input_registry_->isEntityLocal(meas_name)) {
                                    const auto meas =
                                        auxiliary_input_registry_->valuesOf(meas_name, orig_e);
                                    measure = meas.empty() ? Real(0.0) : meas[0];
                                } else {
                                    measure = auxiliary_input_registry_->get(meas_name);
                                }
                                if (measure > Real(0.0)) {
                                    for (auto& se : grad) {
                                        se.value /= measure;
                                    }
                                }
                            }
                        }

                        for (const auto& se : grad) {
                            accum[se.dof] += se.value;
                        }
                    }

#if FE_HAS_MPI
                    {
                        std::vector<std::pair<GlobalIndex, Real>> local_pairs;
                        local_pairs.reserve(accum.size());
                        for (const auto& [dof, val] : accum) {
                            local_pairs.emplace_back(dof, val);
                        }
                        const auto global_pairs =
                            allreduceSumSparsePairs(std::move(local_pairs), dof_handler_.mpiComm());
                        out.assign(global_pairs.begin(), global_pairs.end());
                    }
#else
                    out.reserve(accum.size());
                    for (const auto& [dof, val] : accum) {
                        out.emplace_back(dof, val);
                    }
#endif

                    std::sort(out.begin(), out.end(),
                              [](const auto& a, const auto& b) { return a.first < b.first; });
                    return out;
                };

                ed.n_inputs = static_cast<int>(ctx.inputs.size());
                if (entry.deriv_provider && !entry.coupled_bindings.empty() &&
                    ed.n_inputs > 0) {
                    ed.input_gradients.resize(static_cast<std::size_t>(ed.n_inputs));
                    ed.input_gradient_sources.assign(static_cast<std::size_t>(ed.n_inputs), 0);
                    visitCoupledInputComponents([&](const std::string&,
                                                    const AuxiliaryInputHandle& handle,
                                                    int input_col) {
                        if (input_col < 0 || input_col >= ed.n_inputs) {
                            return;
                        }
                        auto grad = exactInputGradient(handle);
                        if (!grad.empty()) {
                            ed.input_gradients[static_cast<std::size_t>(input_col)] =
                                std::move(grad);
                            ed.input_gradient_sources[static_cast<std::size_t>(input_col)] = 1;
                        }
                    });
                }

                if (is_monolithic && entry.deriv_provider &&
                    !entry.coupled_bindings.empty() && ed.n_inputs > 0) {
                    std::vector<Real> direct_dF_dx(
                        static_cast<std::size_t>(dim * dim), 0.0);
                    std::vector<Real> direct_dF_dinputs(
                        static_cast<std::size_t>(dim * ed.n_inputs), 0.0);
                    AuxiliaryJacobianRequest jac_req;
                    jac_req.n = dim;
                    jac_req.dF_dx = direct_dF_dx;
                    jac_req.dF_dinputs = direct_dF_dinputs;
                    jac_req.n_inputs = ed.n_inputs;
                    entry.deriv_provider->evaluateJacobian(*entry.model, ctx, jac_req);

                    if (!local_condensed) {
                        bordered_coupling_.dF_dinputs = direct_dF_dinputs;
                    }
                    ed.dF_dx = direct_dF_dx;
                    ed.dF_dinputs = direct_dF_dinputs;
                    for (int input_col = 0; input_col < ed.n_inputs; ++input_col) {
                        auto& grad = ed.input_gradients[static_cast<std::size_t>(input_col)];
                        if (!grad.empty() || local_condensed ||
                            ed.aux_dofs.empty() ||
                            bordered_coupling_.Ct.size() <
                                static_cast<std::size_t>(bordered_coupling_.n_aux) * n_field_dofs) {
                            continue;
                        }
                        const auto aux_row_offset =
                            static_cast<std::size_t>(ed.aux_dofs.front() -
                                static_cast<GlobalIndex>(n_field_dofs));
                        grad = reconstructInputGradientFromCt(
                            bordered_coupling_.Ct,
                            n_field_dofs,
                            aux_row_offset,
                            dim,
                            direct_dF_dinputs,
                            ed.n_inputs,
                            input_col);
                        if (!grad.empty()) {
                            ed.input_gradient_sources[static_cast<std::size_t>(input_col)] = 2;
                        }
                    }

                    if (monolithicDirectTraceEnabled()) {
                        std::size_t total_input_grad_nnz = 0;
                        for (const auto& grad : ed.input_gradients) {
                            total_input_grad_nnz += grad.size();
                        }
                        std::ostringstream oss;
                        oss << "FESystem: monolithic direct-only precheck"
                            << " block='" << entry.instance_name << "'"
                            << " entity=" << e
                            << " lower_to_direct_only=" << (entry.lower_to_direct_only ? 1 : 0)
                            << " n_inputs=" << ed.n_inputs
                            << " dF_dx_size=" << ed.dF_dx.size()
                            << " dF_dinputs_size=" << ed.dF_dinputs.size()
                            << " dO_dx_size=" << ed.dO_dx.size()
                            << " dO_dI_size=" << ed.dO_dI.size()
                            << " coupled_bindings=" << entry.coupled_bindings.size()
                            << " input_grad_nnz=" << total_input_grad_nnz;
                        FE_LOG_INFO(oss.str());
                    }

                    if (entry.lower_to_direct_only &&
                        ed.dF_dx.size() == static_cast<std::size_t>(dim * dim) &&
                        ed.dF_dinputs.size() == static_cast<std::size_t>(dim * ed.n_inputs) &&
                        ed.dO_dx.size() == static_cast<std::size_t>(n_outputs * dim)) {
                        ed.dO_dI_effective.assign(
                            static_cast<std::size_t>(n_outputs * ed.n_inputs), Real(0.0));
                        if (!ed.dO_dI.empty()) {
                            const auto copy_count = std::min(ed.dO_dI_effective.size(), ed.dO_dI.size());
                            std::copy_n(ed.dO_dI.begin(),
                                        static_cast<std::ptrdiff_t>(copy_count),
                                        ed.dO_dI_effective.begin());
                        }

                        for (int input_col = 0; input_col < ed.n_inputs; ++input_col) {
                            std::vector<Real> rhs(static_cast<std::size_t>(dim), Real(0.0));
                            for (int row = 0; row < dim; ++row) {
                                rhs[static_cast<std::size_t>(row)] =
                                    ed.dF_dinputs[static_cast<std::size_t>(row * ed.n_inputs + input_col)];
                            }
                            auto A_work = ed.dF_dx;
                            if (!solveDenseSystemInPlace(A_work, rhs)) {
                                if (monolithicDirectTraceEnabled()) {
                                    FE_LOG_INFO("FESystem: monolithic direct-only effective dO_dI solve failed"
                                                " block='" + entry.instance_name + "'"
                                                " entity=" + std::to_string(e) +
                                                " input_col=" + std::to_string(input_col));
                                }
                                continue;
                            }
                            for (int output_idx = 0; output_idx < n_outputs; ++output_idx) {
                                Real effective = Real(0.0);
                                if (!ed.dO_dI.empty() &&
                                    static_cast<std::size_t>(output_idx * ed.n_inputs + input_col) <
                                        ed.dO_dI.size()) {
                                    effective =
                                        ed.dO_dI[static_cast<std::size_t>(output_idx * ed.n_inputs + input_col)];
                                }
                                for (int state_idx = 0; state_idx < dim; ++state_idx) {
                                    effective -=
                                        ed.dO_dx[static_cast<std::size_t>(output_idx * dim + state_idx)] *
                                        rhs[static_cast<std::size_t>(state_idx)];
                                }
                                ed.dO_dI_effective[static_cast<std::size_t>(
                                    output_idx * ed.n_inputs + input_col)] = effective;
                            }
                        }

                        if (monolithicDirectTraceEnabled()) {
                            std::ostringstream oss;
                            oss << "FESystem: monolithic direct-only effective dO_dI"
                                << " block='" << entry.instance_name << "'"
                                << " entity=" << e
                                << " values=[";
                            for (std::size_t idx = 0; idx < ed.dO_dI_effective.size(); ++idx) {
                                if (idx != 0) {
                                    oss << ", ";
                                }
                                oss << ed.dO_dI_effective[idx];
                            }
                            oss << "]";
                            FE_LOG_INFO(oss.str());
                        }
                    }

                    if (local_condensed &&
                        ownsLocalCondensedAuxEquation(entry, blk, e, orig_e) &&
                        !ed.dF_dinputs.empty() &&
                        !ed.input_gradients.empty()) {
                        auto& rec = ensureLocalCondensedRecord(
                            entry.instance_name, e, block_ordinal, global_entity_key, dim);
                        rec.has_aux_equation_terms = true;
                        const bool needs_ct_rows = std::all_of(
                            rec.Ct_rows.begin(),
                            rec.Ct_rows.end(),
                            [](const auto& row) { return row.empty(); });
                        if (needs_ct_rows) {
                            for (int input_col = 0; input_col < ed.n_inputs; ++input_col) {
                                if (input_col >= static_cast<int>(ed.input_gradients.size())) {
                                    continue;
                                }
                                const auto& grad =
                                    ed.input_gradients[static_cast<std::size_t>(input_col)];
                                if (grad.empty()) {
                                    continue;
                                }
                                for (const auto& [dof, value] : grad) {
                                    for (int r = 0; r < dim; ++r) {
                                        const auto coeff = ed.dF_dinputs[static_cast<std::size_t>(
                                                               r * ed.n_inputs + input_col)] *
                                            value;
                                        if (std::abs(coeff) <= kDirectCouplingEntryTol) {
                                            continue;
                                        }
                                        addSparseEntry(
                                            rec.Ct_rows[static_cast<std::size_t>(r)],
                                            dof,
                                            coeff);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            struct VecAccum final : public assembly::GlobalSystemView {
                std::unordered_map<GlobalIndex, Real> entries;
                GlobalIndex sz;
                explicit VecAccum(GlobalIndex s) : sz(s) {}
                void addMatrixEntries(std::span<const GlobalIndex>, std::span<const Real>, assembly::AddMode) override {}
                void addMatrixEntries(std::span<const GlobalIndex>, std::span<const GlobalIndex>, std::span<const Real>, assembly::AddMode) override {}
                void addMatrixEntry(GlobalIndex, GlobalIndex, Real, assembly::AddMode) override {}
                void setDiagonal(std::span<const GlobalIndex>, std::span<const Real>) override {}
                void setDiagonal(GlobalIndex, Real) override {}
                void zeroRows(std::span<const GlobalIndex>, bool) override {}
                void addVectorEntries(std::span<const GlobalIndex> d, std::span<const Real> v, assembly::AddMode) override {
                    for (std::size_t i = 0; i < d.size(); ++i) {
                        if (d[i] >= 0 && d[i] < sz) {
                            entries[d[i]] += v[i];
                        }
                    }
                }
                void addVectorEntry(GlobalIndex d, Real v, assembly::AddMode) override {
                    if (d >= 0 && d < sz) {
                        entries[d] += v;
                    }
                }
                void setVectorEntries(std::span<const GlobalIndex>, std::span<const Real>) override {}
                void zeroVectorEntries(std::span<const GlobalIndex> d) override {
                    for (auto x : d) {
                        entries.erase(x);
                    }
                }
                [[nodiscard]] Real getVectorEntry(GlobalIndex d) const override {
                    auto it = entries.find(d);
                    return it != entries.end() ? it->second : Real(0.0);
                }
                void beginAssemblyPhase() override {}
                void endAssemblyPhase() override {}
                void finalizeAssembly() override {}
                [[nodiscard]] assembly::AssemblyPhase getPhase() const noexcept override {
                    return assembly::AssemblyPhase::Building;
                }
                [[nodiscard]] bool hasMatrix() const noexcept override { return false; }
                [[nodiscard]] bool hasVector() const noexcept override { return true; }
                [[nodiscard]] GlobalIndex numRows() const noexcept override { return sz; }
                [[nodiscard]] GlobalIndex numCols() const noexcept override { return sz; }
                [[nodiscard]] std::string backendName() const override { return "VecAccum"; }
                void zero() override { entries.clear(); }
            };

            const auto output_names = entry.model->outputNames();
            for (int k = 0; k < n_outputs; ++k) {
                const auto& oname = output_names[static_cast<std::size_t>(k)];
                const auto base_slot = auxiliaryOutputSlotOf(entry.instance_name, oname);
                const auto output_id = auxiliaryOutputIdOf(entry.instance_name, oname);
                if (base_slot == static_cast<std::size_t>(-1) ||
                    output_id == static_cast<std::size_t>(-1)) continue;

                for (std::size_t e = 0; e < n_entities; ++e) {
                    const auto& ed = entity_data[e];
                    const auto global_entity_key = globalEntityKeyForAuxEntity(entry, e);
                    bool has_state_sensitivity = false;
                    if (!direct_only) {
                        for (int j = 0; j < dim; ++j) {
                            if (std::abs(ed.dO_dx[static_cast<std::size_t>(k * dim + j)]) > 1e-14) {
                                has_state_sensitivity = true;
                                break;
                            }
                        }
                    }

                    const auto& dO_dI_active =
                        entry.lower_to_direct_only && !ed.dO_dI_effective.empty()
                            ? ed.dO_dI_effective
                            : ed.dO_dI;

                    bool has_direct_sensitivity = false;
                    if (!dO_dI_active.empty() && ed.n_inputs > 0) {
                        for (int input_col = 0; input_col < ed.n_inputs; ++input_col) {
                            if (static_cast<std::size_t>(k * ed.n_inputs + input_col) < dO_dI_active.size() &&
                                std::abs(dO_dI_active[static_cast<std::size_t>(k * ed.n_inputs + input_col)]) >
                                    kDirectCouplingEntryTol &&
                                input_col < static_cast<int>(ed.input_gradients.size()) &&
                                !ed.input_gradients[static_cast<std::size_t>(input_col)].empty()) {
                                has_direct_sensitivity = true;
                                break;
                            }
                        }
                    }

                    if (!has_state_sensitivity && !has_direct_sensitivity) {
                        continue;
                    }

                    const auto slot = base_slot + e * static_cast<std::size_t>(n_outputs);
                    const auto output_id32 = static_cast<std::uint32_t>(output_id);
                    const auto& owned_dofs = dof_handler_.getPartition().locallyOwned();
                    const Real direct_coupling_sign = auxiliaryDirectCouplingSign(direct_only);

                    BorderedCouplingData::DirectCouplingRecord coupling_record;
                    coupling_record.output_slot = slot;
                    coupling_record.entity_index = e;
                    if (!direct_only) {
                        coupling_record.aux_local_indices.reserve(ed.aux_dofs.size());
                        for (const auto aux_dof : ed.aux_dofs) {
                            coupling_record.aux_local_indices.push_back(
                                static_cast<std::size_t>(aux_dof) - n_field_dofs);
                        }
                    }
                    coupling_record.dF_dinputs = ed.dF_dinputs;
                    if (!direct_only &&
                        static_cast<std::size_t>((k + 1) * dim) <= ed.dO_dx.size()) {
                        const auto dx_begin = ed.dO_dx.begin() + static_cast<std::ptrdiff_t>(k * dim);
                        coupling_record.dO_dx.assign(dx_begin, dx_begin + dim);
                    }
                    if (!dO_dI_active.empty() && ed.n_inputs > 0 &&
                        static_cast<std::size_t>((k + 1) * ed.n_inputs) <= dO_dI_active.size()) {
                        const auto di_begin =
                            dO_dI_active.begin() + static_cast<std::ptrdiff_t>(k * ed.n_inputs);
                        coupling_record.dO_dI.assign(di_begin, di_begin + ed.n_inputs);
                        for (auto& value : coupling_record.dO_dI) {
                            value *= direct_coupling_sign;
                        }
                    }
                    coupling_record.input_gradients = ed.input_gradients;
                    std::unordered_map<GlobalIndex, Real> direct_output_gradient_entries;

                    for (const auto& frec : formulation_records_) {
                        for (const auto& [block_key, block_node] : frec.block_residual_exprs) {
                            if (!block_node) continue;

                            bool references_slot = false;
                            std::function<void(const forms::FormExprNode&)> scan_refs =
                                [&](const forms::FormExprNode& n) {
                                    if (n.type() == forms::FormExprType::AuxiliaryOutputRef) {
                                        const auto s = n.slotIndex();
                                        if (s && *s == output_id32) references_slot = true;
                                    }
                                    for (const auto* c : n.children()) {
                                        if (c && !references_slot) scan_refs(*c);
                                    }
                                };
                            scan_refs(*block_node);
                            if (!references_slot) continue;

                            const auto block_residual = forms::FormExpr(
                                std::const_pointer_cast<forms::FormExprNode>(block_node));
                            const auto test_field = block_key.first;
                            const auto n_total = static_cast<GlobalIndex>(dof_handler_.getNumDofs());
                            if (n_total <= 0 || !assembler_) continue;

                            auto relevant = forms::extractTermsReferencing(
                                block_residual, forms::FormExprType::AuxiliaryOutputRef,
                                output_id32);
                            if (!relevant.isValid()) continue;

                            auto dR_dOk = forms::differentiateWrtAuxiliaryOutput(
                                relevant, output_id32);
                            if (!dR_dOk.isValid()) continue;

                            try {
                                forms::FormCompiler compiler;
                                auto ir = compiler.compileLinear(dR_dOk);
                                forms::FormKernel deriv_kernel(std::move(ir));

                                struct VecAccum final : public assembly::GlobalSystemView {
                                    std::unordered_map<GlobalIndex, Real> entries;
                                    GlobalIndex sz;
                                    explicit VecAccum(GlobalIndex s) : sz(s) {}
                                    void addMatrixEntries(std::span<const GlobalIndex>, std::span<const Real>, assembly::AddMode) override {}
                                    void addMatrixEntries(std::span<const GlobalIndex>, std::span<const GlobalIndex>, std::span<const Real>, assembly::AddMode) override {}
                                    void addMatrixEntry(GlobalIndex, GlobalIndex, Real, assembly::AddMode) override {}
                                    void setDiagonal(std::span<const GlobalIndex>, std::span<const Real>) override {}
                                    void setDiagonal(GlobalIndex, Real) override {}
                                    void zeroRows(std::span<const GlobalIndex>, bool) override {}
                                    void addVectorEntries(std::span<const GlobalIndex> d, std::span<const Real> v, assembly::AddMode) override {
                                        for (std::size_t i = 0; i < d.size(); ++i) {
                                            if (d[i] >= 0 && d[i] < sz) entries[d[i]] += v[i];
                                        }
                                    }
                                    void addVectorEntry(GlobalIndex d, Real v, assembly::AddMode) override {
                                        if (d >= 0 && d < sz) entries[d] += v;
                                    }
                                    void setVectorEntries(std::span<const GlobalIndex>, std::span<const Real>) override {}
                                    void zeroVectorEntries(std::span<const GlobalIndex> d) override { for (auto x : d) entries.erase(x); }
                                    [[nodiscard]] Real getVectorEntry(GlobalIndex d) const override {
                                        auto it = entries.find(d); return it != entries.end() ? it->second : 0.0;
                                    }
                                    void beginAssemblyPhase() override {}
                                    void endAssemblyPhase() override {}
                                    void finalizeAssembly() override {}
                                    [[nodiscard]] assembly::AssemblyPhase getPhase() const noexcept override { return assembly::AssemblyPhase::Building; }
                                    [[nodiscard]] bool hasMatrix() const noexcept override { return false; }
                                    [[nodiscard]] bool hasVector() const noexcept override { return true; }
                                    [[nodiscard]] GlobalIndex numRows() const noexcept override { return sz; }
                                    [[nodiscard]] GlobalIndex numCols() const noexcept override { return sz; }
                                    [[nodiscard]] std::string backendName() const override { return "VecAccum"; }
                                    void zero() override { entries.clear(); }
                                };

                                VecAccum dR_vec(n_total);
                                const auto* restore_constraints =
                                    use_constraints_in_assembly_ ? &affine_constraints_ : nullptr;
                                // Assemble dR/d(output) in the same constrained test space
                                // as the PDE operator.  The legacy coupled-boundary path
                                // does not disable constraints here, and the monolithic
                                // operator must match that free-DOF Jacobian.
                                assembler_->setConstraints(restore_constraints);

                                std::unique_ptr<assembly::GlobalSystemView> sol_view;
                                assembler_->setCurrentSolutionView(nullptr);
                                if (state.u_vector) {
                                    auto* vec = const_cast<backends::GenericVector*>(state.u_vector);
                                    sol_view = vec->createAssemblyView();
                                    assembler_->setCurrentSolutionView(sol_view.get());
                                }

                                const auto sensitivity_options =
                                    localSensitivityAssemblyOptions(*assembler_);
                                ScopedAssemblerOptions sensitivity_option_guard(
                                    assembler_.get(), sensitivity_options);

                                const auto& drec = fieldRecord(test_field);
                                if (drec.space) {
                                    const auto foff = fieldDofOffset(test_field);
                                    const auto& fdh = fieldDofHandler(test_field);
                                    assembler_->setRowDofMap(fdh.getDofMap(), foff);
                                    assembler_->setColDofMap(fdh.getDofMap(), foff);

                                    if (deriv_kernel.hasCell()) {
                                        assembler_->assembleVector(
                                            meshAccess(), *drec.space, deriv_kernel, dR_vec);
                                    }
                                    if (deriv_kernel.hasBoundaryFace()) {
                                        const auto scan = analysis::scanFormExpr(*block_node);
                                        const auto& mesh = meshAccess();
                                        if (scan.boundary_markers.empty()) {
                                            assembler_->assembleBoundaryFaces(
                                                mesh, /*boundary_marker=*/-1, *drec.space,
                                                deriv_kernel, nullptr, &dR_vec);
                                        } else {
                                            for (const int marker : scan.boundary_markers) {
                                                assembler_->assembleBoundaryFaces(
                                                    mesh, marker, *drec.space,
                                                    deriv_kernel, nullptr, &dR_vec);
                                            }
                                        }
                                    }
                                }
                                assembler_->setConstraints(restore_constraints);

                                Real residual_output_weight = Real(1.0);
                                if (state.time_integration) {
                                    residual_output_weight =
                                        state.time_integration->non_time_derivative_term_weight;
                                }
                                if (std::abs(residual_output_weight - Real(1.0)) > Real(1e-14)) {
                                    for (auto& [dof, val] : dR_vec.entries) {
                                        (void)dof;
                                        val *= residual_output_weight;
                                    }
                                }

                                std::unordered_map<GlobalIndex, Real> dR_rank1_entries = dR_vec.entries;
#if FE_HAS_MPI
                                if (!local_condensed) {
                                    std::vector<std::pair<GlobalIndex, Real>> local_pairs;
                                    local_pairs.reserve(dR_vec.entries.size());
                                    for (const auto& [dof, val] : dR_vec.entries) {
                                        local_pairs.emplace_back(dof, val);
                                    }
                                    const auto global_pairs =
                                        allreduceSumSparsePairs(std::move(local_pairs), dof_handler_.mpiComm());
                                    dR_rank1_entries.clear();
                                    dR_rank1_entries.reserve(global_pairs.size());
                                    for (const auto& [dof, val] : global_pairs) {
                                        dR_rank1_entries[dof] = val;
                                    }
                                }
#endif
                                // Local-condensed Cell/QP blocks reduce B columns later through
                                // globally ordered condensed slots. Avoid a per-entity collective
                                // here because not every rank owns every condensed entity.

                                for (const auto& [dof, val] : dR_rank1_entries) {
                                    if (std::abs(val) <= kDirectCouplingEntryTol) {
                                        continue;
                                    }
                                    direct_output_gradient_entries[dof] += val;
                                }

                                const bool disable_direct_coupling =
                                    std::getenv("SVMP_DISABLE_AUX_DIRECT_COUPLING") != nullptr;
                                if (!disable_direct_coupling &&
                                    !ed.input_gradients.empty() && !dO_dI_active.empty() &&
                                    ed.n_inputs > 0) {
                                    for (int input_col = 0; input_col < ed.n_inputs; ++input_col) {
                                        if (static_cast<std::size_t>(k * ed.n_inputs + input_col) >=
                                            dO_dI_active.size()) {
                                            continue;
                                        }
                                        const Real dOk_dIm = direct_coupling_sign * dO_dI_active[
                                            static_cast<std::size_t>(k * ed.n_inputs + input_col)];
                                        if (std::abs(dOk_dIm) <= kDirectCouplingEntryTol) {
                                            continue;
                                        }
                                        if (input_col >= static_cast<int>(ed.input_gradients.size())) {
                                            continue;
                                        }
                                        const auto& q_u =
                                            ed.input_gradients[static_cast<std::size_t>(input_col)];
                                        if (monolithicDirectTraceEnabled()) {
                                            const char* grad_source = "none";
                                            if (input_col < static_cast<int>(ed.input_gradient_sources.size())) {
                                                const auto src = ed.input_gradient_sources[
                                                    static_cast<std::size_t>(input_col)];
                                                grad_source = (src == 1) ? "exact"
                                                    : ((src == 2) ? "reconstructed" : "none");
                                            }
                                            Real q_norm_sq = Real(0.0);
                                            for (const auto& [dof, qj] : q_u) {
                                                (void)dof;
                                                q_norm_sq += qj * qj;
                                            }
                                            Real dR_norm_sq = Real(0.0);
                                            for (const auto& [dof, dRi_dOk] : dR_rank1_entries) {
                                                (void)dof;
                                                dR_norm_sq += dRi_dOk * dRi_dOk;
                                            }
                                            std::ostringstream oss;
                                            oss << "FESystem: monolithic direct coupling"
                                                << " block='" << entry.instance_name << "'"
                                                << " output='" << oname << "'"
                                                << " entity=" << e
                                                << " input_col=" << input_col
                                                << " dO_dI=" << dOk_dIm
                                                << " grad_source=" << grad_source
                                                << " grad_nnz=" << q_u.size()
                                                << " grad_norm=" << std::sqrt(q_norm_sq)
                                                << " dR_nnz=" << dR_rank1_entries.size()
                                                << " dR_norm=" << std::sqrt(dR_norm_sq);
                                            oss << " grad_entries=[";
                                            bool first = true;
                                            for (const auto& [dof, qj] : q_u) {
                                                if (!first) {
                                                    oss << ", ";
                                                }
                                                first = false;
                                                oss << "(" << dof << ":" << qj << ")";
                                            }
                                            oss << "]";
                                            oss << " dR_entries=[";
                                            first = true;
                                            for (const auto& [dof, dRi_dOk] : dR_rank1_entries) {
                                                if (!first) {
                                                    oss << ", ";
                                                }
                                                first = false;
                                                oss << "(" << dof << ":" << dRi_dOk << ")";
                                            }
                                            oss << "]";
                                            FE_LOG_INFO(oss.str());
                                        }
                                        // For live monolithic solves, prefer the exact
                                        // reduced-field form even when the outer product is
                                        // rank-one. Pure algebraic direct-only auxiliary
                                        // blocks are handled after aggregation so they can
                                        // recover the exact native face rank-one path when
                                        // appropriate.
                                        const bool global_q_u_nonempty =
                                            mpiAllreduceSumIfActive(q_u.empty() ? 0 : 1) > 0;
                                        if (!entry.lower_to_direct_only && global_q_u_nonempty) {
                                            if (monolithicDirectTraceEnabled()) {
                                                std::ostringstream oss;
                                                oss << "FESystem: monolithic direct coupling"
                                                    << " block='" << entry.instance_name << "'"
                                                    << " output='" << oname << "'"
                                                    << " entity=" << e
                                                    << " input_col=" << input_col
                                                    << " path='reduced_exact_update'"
                                                    << " local_grad_nnz=" << q_u.size()
                                                    << " global_grad_active="
                                                    << (global_q_u_nonempty ? 1 : 0);
                                                FE_LOG_INFO(oss.str());
                                            }
                                            backends::ReducedFieldUpdate reduced_update;
                                            reduced_update.sigma = dOk_dIm;

                                            const auto& owned_dofs = dof_handler_.getPartition().locallyOwned();
                                            reduced_update.left.reserve(dR_rank1_entries.size());
                                            reduced_update.right.reserve(q_u.size());

                                            for (const auto& [dof_i, dRi_dOk] : dR_rank1_entries) {
                                                if (!owned_dofs.contains(dof_i) ||
                                                    std::abs(dRi_dOk) <= kDirectCouplingEntryTol) {
                                                    continue;
                                                }
                                                reduced_update.left.emplace_back(dof_i, dRi_dOk);
                                            }
                                            for (const auto& [dof_j, qj] : q_u) {
                                                if (!owned_dofs.contains(dof_j) ||
                                                    std::abs(qj) <= kDirectCouplingEntryTol) {
                                                    continue;
                                                }
                                                reduced_update.right.emplace_back(dof_j, qj);
                                            }

                                            if (std::abs(reduced_update.sigma) > kDirectCouplingEntryTol) {
                                                // Preserve globally active reduced-update slots even
                                                // when this rank owns no entries on one side. The
                                                // distributed FSILS backend requires identical update
                                                // counts across ranks to keep overlap exchanges
                                                // ordered.
                                                last_reduced_field_updates_.push_back(
                                                    std::move(reduced_update));
                                            }
                                        }
                                    }
                                }

                                const bool single_owner_bordered_insert =
                                    useSingleOwnerBorderedAssembly(entry, local_condensed, direct_only);
                                const bool insert_bordered_output_gradient =
                                    !single_owner_bordered_insert ||
                                    isSingleOwnerBorderedAssemblyRank(entry);
                                const auto& bordered_output_gradient =
                                    single_owner_bordered_insert ? dR_rank1_entries : dR_vec.entries;
                                if (insert_bordered_output_gradient) {
                                    for (const auto& [dof_i, dRi_dOk] : bordered_output_gradient) {
                                        if (std::abs(dRi_dOk) < 1e-14) continue;
                                        for (int j = 0; j < dim; ++j) {
                                            const Real dOk_dxj =
                                                ed.dO_dx[static_cast<std::size_t>(k * dim + j)];
                                            if (std::abs(dOk_dxj) < 1e-14) continue;

                                            const Real val = dRi_dOk * dOk_dxj;
                                            if (!direct_only && !local_condensed) {
                                                std::vector<GlobalIndex> row = {dof_i};
                                                std::vector<GlobalIndex> col = {
                                                    ed.aux_dofs[static_cast<std::size_t>(j)]};
                                                std::vector<Real> mat = {val};
                                                matrix_out->addMatrixEntries(row, col, mat);
                                            }
                                        }
                                    }
                                }
                            } catch (const std::exception&) {
                                // Symbolic compilation may fail for complex expressions.
                            }
                        }
                    }

                    coupling_record.output_gradient.reserve(direct_output_gradient_entries.size());
                    for (const auto& [dof, val] : direct_output_gradient_entries) {
                        if (std::abs(val) <= kDirectCouplingEntryTol) {
                            continue;
                        }
                        coupling_record.output_gradient.emplace_back(dof, val);
                    }
                    std::sort(coupling_record.output_gradient.begin(),
                              coupling_record.output_gradient.end(),
                              [](const auto& a, const auto& b) {
                                  return a.first < b.first;
                    });
                    if (local_condensed) {
                        auto& rec = ensureLocalCondensedRecord(
                            entry.instance_name, e, block_ordinal, global_entity_key, dim);
                        for (const auto& [dof_i, dRi_dOk] : coupling_record.output_gradient) {
                            for (int j = 0; j < dim; ++j) {
                                const Real dOk_dxj =
                                    ed.dO_dx[static_cast<std::size_t>(k * dim + j)];
                                if (std::abs(dOk_dxj) <= kDirectCouplingEntryTol) {
                                    continue;
                                }
                                addSparseEntry(
                                    rec.B_columns[static_cast<std::size_t>(j)],
                                    dof_i,
                                    dRi_dOk * dOk_dxj);
                            }
                        }
                    }
                    if (direct_only && !coupling_record.output_gradient.empty()) {
                        bool promoted_direct_only = false;
                        int active_input_col = -1;
                        if (!coupling_record.dO_dI.empty()) {
                            for (std::size_t input_col = 0;
                                 input_col < coupling_record.dO_dI.size();
                                 ++input_col) {
                                if (std::abs(coupling_record.dO_dI[input_col]) <=
                                    kDirectCouplingEntryTol) {
                                    continue;
                                }
                                if (active_input_col >= 0) {
                                    active_input_col = -2;
                                    break;
                                }
                                active_input_col = static_cast<int>(input_col);
                            }
                        }

                        if (active_input_col >= 0 &&
                            static_cast<std::size_t>(active_input_col) <
                                coupling_record.input_gradients.size()) {
                            const auto& q_u =
                                coupling_record.input_gradients[static_cast<std::size_t>(active_input_col)];
                            if (!q_u.empty()) {
                                backends::RankOneUpdate promoted;
                                if (tryPromoteDirectReducedToNativeRankOne(
                                        std::span<const std::pair<GlobalIndex, Real>>(
                                            coupling_record.output_gradient.data(),
                                            coupling_record.output_gradient.size()),
                                        std::span<const std::pair<GlobalIndex, Real>>(
                                            q_u.data(), q_u.size()),
                                        coupling_record.dO_dI[static_cast<std::size_t>(active_input_col)],
                                        owned_dofs,
                                        promoted)) {
                                    last_rank_one_updates_.push_back(std::move(promoted));
                                    promoted_direct_only = true;
                                    if (monolithicDirectTraceEnabled()) {
                                        std::ostringstream oss;
                                        oss << "FESystem: monolithic direct coupling"
                                            << " block='" << entry.instance_name << "'"
                                            << " output='" << oname << "'"
                                            << " entity=" << e
                                            << " path='native_rank_one'";
                                        FE_LOG_INFO(oss.str());
                                    }
                                }
                            }
                        }

                        if (!promoted_direct_only) {
                            for (std::size_t input_col = 0;
                                 input_col < coupling_record.dO_dI.size();
                                 ++input_col) {
                                const Real dOk_dIm = coupling_record.dO_dI[input_col];
                                if (std::abs(dOk_dIm) <= kDirectCouplingEntryTol ||
                                    input_col >= coupling_record.input_gradients.size()) {
                                    continue;
                                }
                                const auto& q_u = coupling_record.input_gradients[input_col];
                                if (q_u.empty()) {
                                    continue;
                                }
                                backends::ReducedFieldUpdate reduced_update;
                                reduced_update.sigma = dOk_dIm;
                                reduced_update.left.reserve(coupling_record.output_gradient.size());
                                reduced_update.right.reserve(q_u.size());
                                for (const auto& [dof_i, dRi_dOk] : coupling_record.output_gradient) {
                                    if (!owned_dofs.contains(dof_i) ||
                                        std::abs(dRi_dOk) <= kDirectCouplingEntryTol) {
                                        continue;
                                    }
                                    reduced_update.left.emplace_back(dof_i, dRi_dOk);
                                }
                                for (const auto& [dof_j, qj] : q_u) {
                                    if (!owned_dofs.contains(dof_j) ||
                                        std::abs(qj) <= kDirectCouplingEntryTol) {
                                        continue;
                                    }
                                    reduced_update.right.emplace_back(dof_j, qj);
                                }
                                // Preserve globally active reduced-update slots even when the
                                // ownership partition leaves one side empty on this rank.
                                if (std::abs(reduced_update.sigma) > kDirectCouplingEntryTol) {
                                    last_reduced_field_updates_.push_back(std::move(reduced_update));
                                }
                            }
                        }
                    }
                    if (!local_condensed && !direct_only) {
                        bordered_coupling_.direct_coupling_records.push_back(
                            std::move(coupling_record));
                    }
                }
            }
        }
    }

    const int global_local_condensed_record_count =
        mpiAllreduceSumIfActive(static_cast<int>(last_local_condensed_records_.size()));
    if (global_local_condensed_record_count > 0) {
        const auto& owned_dofs = dof_handler_.getPartition().locallyOwned();

        struct CondensedSlot {
            std::size_t block_ordinal{0};
            std::uint64_t entity_key{0};
            std::size_t component{0};
        };

        auto sortUniqueSlots = [](std::vector<CondensedSlot>& slots) {
            std::sort(slots.begin(), slots.end(), [](const auto& a, const auto& b) {
                return std::tie(a.block_ordinal, a.entity_key, a.component) <
                       std::tie(b.block_ordinal, b.entity_key, b.component);
            });
            slots.erase(std::unique(slots.begin(), slots.end(), [](const auto& a, const auto& b) {
                            return a.block_ordinal == b.block_ordinal &&
                                   a.entity_key == b.entity_key &&
                                   a.component == b.component;
                        }),
                        slots.end());
        };

        auto findRecordForSlot = [&](const CondensedSlot& slot) -> const LocalCondensedEntityRecord* {
            for (const auto& rec : last_local_condensed_records_) {
                if (rec.block_ordinal == slot.block_ordinal &&
                    rec.global_entity_key == slot.entity_key) {
                    return &rec;
                }
            }
            return nullptr;
        };

        auto recordDimension = [](const LocalCondensedEntityRecord& rec) -> std::size_t {
            if (!rec.B_columns.empty()) {
                return rec.B_columns.size();
            }
            if (!rec.Ct_rows.empty()) {
                return rec.Ct_rows.size();
            }
            return rec.g.size();
        };

        auto hasEquationTerms =
            [&](const LocalCondensedEntityRecord& rec, std::size_t dim) {
            return rec.has_aux_equation_terms &&
                   dim > 0u &&
                   rec.D_inv.size() == dim * dim &&
                   rec.Ct_rows.size() == dim;
        };

        auto hasBColumns =
            [](const LocalCondensedEntityRecord& rec, std::size_t dim) {
            return dim > 0u && rec.B_columns.size() == dim;
        };

        auto gatherGlobalSlots = [&](std::vector<CondensedSlot> slots) {
#if FE_HAS_MPI
            {
                std::vector<LocalCondensedSlotKey> mpi_slots;
                mpi_slots.reserve(slots.size());
                for (const auto& slot : slots) {
                    mpi_slots.push_back(LocalCondensedSlotKey{
                        static_cast<unsigned long long>(slot.block_ordinal),
                        static_cast<unsigned long long>(slot.entity_key),
                        static_cast<unsigned long long>(slot.component)});
                }
                const auto gathered =
                    allgatherLocalCondensedSlotKeys(std::move(mpi_slots), dof_handler_.mpiComm());
                slots.clear();
                slots.reserve(gathered.size());
                for (const auto& slot : gathered) {
                    slots.push_back(CondensedSlot{
                        static_cast<std::size_t>(slot.block_ordinal),
                        static_cast<std::uint64_t>(slot.entity_key),
                        static_cast<std::size_t>(slot.component)});
                }
            }
#endif
            sortUniqueSlots(slots);
            return slots;
        };

        auto localCondensedSlots = [&]() {
            std::vector<CondensedSlot> slots;
            for (const auto& rec : last_local_condensed_records_) {
                const auto dim = recordDimension(rec);
                if (!hasBColumns(rec, dim) && !hasEquationTerms(rec, dim)) {
                    continue;
                }
                for (std::size_t j = 0; j < dim; ++j) {
                    slots.push_back(CondensedSlot{
                        rec.block_ordinal, rec.global_entity_key, j});
                }
            }
            return gatherGlobalSlots(std::move(slots));
        };

        if (want_matrix) {
            std::vector<CondensedSlot> update_slots = localCondensedSlots();

            for (const auto& slot : update_slots) {
                std::vector<std::pair<GlobalIndex, Real>> left_entries;
                std::vector<std::pair<GlobalIndex, Real>> right_entries;

                if (const auto* rec = findRecordForSlot(slot)) {
                    const auto dim = recordDimension(*rec);
                    if (slot.component < dim && hasBColumns(*rec, dim)) {
                        left_entries = rec->B_columns[slot.component];
                    }

                    if (slot.component < dim && hasEquationTerms(*rec, dim)) {
                        std::unordered_map<GlobalIndex, Real> right_dense;
                        for (std::size_t row = 0; row < dim; ++row) {
                            const Real coeff = rec->D_inv[slot.component * dim + row];
                            if (std::abs(coeff) <= Real(1e-30)) {
                                continue;
                            }
                            for (const auto& [dof, val] : rec->Ct_rows[row]) {
                                right_dense[dof] += coeff * val;
                            }
                        }
                        right_entries.reserve(right_dense.size());
                        for (const auto& [dof, val] : right_dense) {
                            right_entries.emplace_back(dof, val);
                        }
                    }
                }

#if FE_HAS_MPI
                left_entries =
                    allreduceSumSparsePairs(std::move(left_entries), dof_handler_.mpiComm());
                right_entries =
                    allreduceSumSparsePairs(std::move(right_entries), dof_handler_.mpiComm());
#endif

                backends::ReducedFieldUpdate reduced_update;
                reduced_update.sigma = Real(-1.0);
                for (const auto& [dof, val] : left_entries) {
                    if (owned_dofs.contains(dof) &&
                        std::abs(val) > kDirectCouplingEntryTol) {
                        reduced_update.left.emplace_back(dof, val);
                    }
                }
                for (const auto& [dof, val] : right_entries) {
                    if (owned_dofs.contains(dof) &&
                        std::abs(val) > kDirectCouplingEntryTol) {
                        reduced_update.right.emplace_back(dof, val);
                    }
                }

                // Every rank pushes the same globally ordered slot. Its factors
                // are owner-partitioned, so FSILS low-rank dot products are
                // globally reduced without mixing different cell/QP entities.
                last_reduced_field_updates_.push_back(std::move(reduced_update));
            }
        }

        if (want_vector) {
            std::vector<CondensedSlot> rhs_slots = localCondensedSlots();
            last_local_condensed_rhs_shift_.assign(n_field_dofs, Real(0.0));

            for (const auto& slot : rhs_slots) {
                std::vector<std::pair<GlobalIndex, Real>> left_entries;
                Real coeff = Real(0.0);

                if (const auto* rec = findRecordForSlot(slot)) {
                    const auto dim = recordDimension(*rec);
                    if (slot.component < dim && hasBColumns(*rec, dim)) {
                        left_entries = rec->B_columns[slot.component];
                    }
                    if (slot.component < dim && hasEquationTerms(*rec, dim) &&
                        rec->g.size() == dim) {
                        for (std::size_t row = 0; row < dim; ++row) {
                            coeff += rec->D_inv[slot.component * dim + row] *
                                     rec->g[row];
                        }
                    }
                }

#if FE_HAS_MPI
                left_entries =
                    allreduceSumSparsePairs(std::move(left_entries), dof_handler_.mpiComm());
                {
                    int mpi_initialized = 0;
                    MPI_Initialized(&mpi_initialized);
                    if (mpi_initialized) {
                        Real global_coeff = coeff;
                        MPI_Allreduce(&coeff, &global_coeff, 1, MPI_DOUBLE, MPI_SUM,
                                      dof_handler_.mpiComm());
                        coeff = global_coeff;
                    }
                }
#endif

                if (std::abs(coeff) <= Real(1e-30)) {
                    continue;
                }
                for (const auto& [dof, val] : left_entries) {
                    const auto dof_idx = static_cast<std::size_t>(dof);
                    if (dof_idx < last_local_condensed_rhs_shift_.size() &&
                        owned_dofs.contains(dof)) {
                        last_local_condensed_rhs_shift_[dof_idx] += val * coeff;
                    }
                }
            }
        }
    }

    // Purely algebraic monolithic blocks are lowered later in NewtonSolver,
    // after the full bordered data (B, C^T, D, g, direct-coupling metadata)
    // has been assembled. Keeping the bordered representation intact here
    // avoids overlapping FE-side and Newton-side lowering paths and lets the
    // solver apply the exact reduced RHS shift r - B D^{-1} g together with
    // the reduced Jacobian K - B D^{-1} C.

    // Assemble registered AuxiliaryOperator contributions.
    if (auxiliary_operator_registry_) {
        for (const auto& op_name : auxiliary_operator_registry_->operatorNames()) {
            const auto& op = auxiliary_operator_registry_->getOperator(op_name);
            if (!op.residual_fn && !op.jacobian_fn) continue;

            AuxiliaryOperatorContext op_ctx;
            op_ctx.time = state.time;
            op_ctx.dt = state.dt;

            // Helper to resolve an operator endpoint (source or target)
            // to data span, offset, and DOF count in the mixed system.
            // scratch_buf is per-endpoint to avoid overwriting when both
            // source and target are field references in the distributed case.
            auto resolveEndpoint = [&](const std::string& name,
                                       std::vector<Real>& scratch_buf,
                                       std::span<const Real>& data_out,
                                       std::size_t& entity_count_out,
                                       int& stride_out,
                                       std::size_t& offset_out,
                                       std::size_t& n_out) {
                // Check auxiliary block first.
                if (auxiliary_state_manager_->hasBlock(name)) {
                    auto& blk = auxiliary_state_manager_->getBlock(name);
                    data_out = blk.work();
                    entity_count_out = blk.entityCount();
                    stride_out = blk.componentStride();
                    for (const auto& bl : mixed.aux_layout.blocks) {
                        if (bl.name == name) {
                            offset_out = bl.offset + mixed.aux_layout.mixed_system_offset;
                            n_out = bl.n_unknowns;
                            return;
                        }
                    }
                }
                // Check if it's a field reference (possibly "field:name" syntax).
                std::string field_name = name;
                if (name.substr(0, 6) == "field:") {
                    field_name = name.substr(6);
                }
                const FieldId fid = field_registry_.findByName(field_name);
                if (fid != INVALID_FIELD_ID) {
                    const auto fidx = static_cast<std::size_t>(fid);
                    const auto& rec = field_registry_.get(fid);
                    stride_out = std::max(1, rec.components);

                    // Field DOF offset and count in the global system.
                    const std::size_t fld_off = (fidx < field_dof_offsets_.size())
                        ? static_cast<std::size_t>(field_dof_offsets_[fidx]) : 0;
                    offset_out = fld_off;
                    if (fidx < field_dof_handlers_.size()) {
                        n_out = static_cast<std::size_t>(
                            field_dof_handlers_[fidx].getNumDofs());
                    } else {
                        n_out = 0;
                    }
                    // DOF-tuple count: number of DOF groups of size `stride`.
                    // For vertex-based Lagrange: equals num vertices.
                    // For higher-order: equals total DOFs / components.
                    entity_count_out = (stride_out > 0) ? n_out / static_cast<std::size_t>(stride_out) : 0;

                    // Provide a field-local view into the solution vector.
                    if (!cached_solution_u_.empty() && fld_off + n_out <= cached_solution_u_.size()) {
                        data_out = cached_solution_u_.subspan(fld_off, n_out);
                    } else if (cached_solution_vector_ && n_out > 0) {
                        // Distributed case: materialize field DOFs from
                        // the backend vector into the per-endpoint scratch.
                        auto* vec = const_cast<backends::GenericVector*>(cached_solution_vector_);
                        auto view = vec->createAssemblyView();
                        scratch_buf.resize(n_out);
                        for (std::size_t i = 0; i < n_out; ++i) {
                            scratch_buf[i] = view->getVectorEntry(
                                static_cast<GlobalIndex>(fld_off + i));
                        }
                        data_out = scratch_buf;
                    } else {
                        data_out = {};
                    }
                }
            };

            std::size_t src_offset = 0, src_n = 0;
            {
                std::span<const Real> src_data;
                std::size_t src_ec = 0;
                int src_s = 0;
                resolveEndpoint(op.source_name, field_endpoint_scratch_src_,
                                src_data, src_ec, src_s,
                                src_offset, src_n);
                op_ctx.source_data = src_data;
                op_ctx.source_entity_count = src_ec;
                op_ctx.source_stride = src_s;
            }

            std::size_t tgt_offset = 0, tgt_n = 0;
            {
                std::span<const Real> tgt_data;
                std::size_t tgt_ec = 0;
                int tgt_s = 0;
                resolveEndpoint(op.target_name, field_endpoint_scratch_tgt_,
                                tgt_data, tgt_ec, tgt_s,
                                tgt_offset, tgt_n);
                op_ctx.target_data = tgt_data;
                op_ctx.target_entity_count = tgt_ec;
                op_ctx.target_stride = tgt_s;
            }

            // Residual contribution.
            if (want_vector && vector_out && op.residual_fn && tgt_n > 0) {
                std::vector<Real> op_res(tgt_n);
                op.residual_fn(op_ctx, op_res);
                std::vector<GlobalIndex> tgt_dofs(tgt_n);
                for (std::size_t i = 0; i < tgt_n; ++i)
                    tgt_dofs[i] = static_cast<GlobalIndex>(tgt_offset + i);
                vector_out->addVectorEntries(tgt_dofs, op_res);
            }

            // Jacobian contribution.
            if (want_matrix && matrix_out && op.jacobian_fn && tgt_n > 0 && src_n > 0) {
                std::vector<Real> op_jac(tgt_n * src_n);
                op.jacobian_fn(op_ctx, op_jac);
                std::vector<GlobalIndex> tgt_dofs(tgt_n), src_dofs(src_n);
                for (std::size_t i = 0; i < tgt_n; ++i)
                    tgt_dofs[i] = static_cast<GlobalIndex>(tgt_offset + i);
                for (std::size_t i = 0; i < src_n; ++i)
                    src_dofs[i] = static_cast<GlobalIndex>(src_offset + i);
                matrix_out->addMatrixEntries(tgt_dofs, src_dofs, op_jac);
            }
        }
    }
}

void FESystem::assembleMonolithicAuxiliary(
    Real time, Real dt,
    std::span<Real> residual_out,
    std::span<Real> jacobian_out,
    bool is_nonlinear_iteration)
{
    if (!auxiliary_state_manager_ || !auxiliary_operator_registry_) return;

    const auto& layout = auxiliary_operator_registry_->auxiliaryLayout();
    const auto n_total = layout.total_aux_unknowns;
    FE_THROW_IF(residual_out.size() < n_total, InvalidArgumentException,
                "assembleMonolithicAuxiliary: residual buffer too small");
    FE_THROW_IF(jacobian_out.size() < n_total * n_total, InvalidArgumentException,
                "assembleMonolithicAuxiliary: Jacobian buffer too small");

    std::fill(residual_out.begin(), residual_out.begin() + static_cast<std::ptrdiff_t>(n_total), 0.0);
    std::fill(jacobian_out.begin(), jacobian_out.begin() + static_cast<std::ptrdiff_t>(n_total * n_total), 0.0);

    // Ensure auxiliary inputs are evaluated for the current step.
    // Pass is_nonlinear_iteration so EachNonlinearIteration inputs refresh.
    if (auxiliary_input_registry_) {
        auxiliary_input_registry_->evaluate(time, dt, is_nonlinear_iteration);
    }
    initializeAuxiliaryDAEBlocksIfNeeded_(time, dt);

    SystemStateView mono_state;
    mono_state.time = time;
    mono_state.dt = dt;
    mono_state.effective_dt = dt;

    // Assemble contributions from each monolithic deployed block.
    for (auto& entry : deployed_aux_entries_) {
        if (!entry.materialized) continue;
        if (entry.spec.solve_mode != AuxiliarySolveMode::Monolithic) continue;
        if (entry.lower_to_direct_only) continue;
        if (!auxiliary_state_manager_->hasBlock(entry.instance_name)) continue;

        auto& blk = auxiliary_state_manager_->getBlock(entry.instance_name);
        const int dim = entry.spec.size;
        const auto n_entities = blk.entityCount();

        // Find this block's offset in the mixed layout.
        std::size_t block_offset = 0;
        for (const auto& bl : layout.blocks) {
            if (bl.name == entry.instance_name) {
                block_offset = bl.offset;
                break;
            }
        }

        auto params = buildParamVector(entry);
        auto bound_inputs = buildInputVector(entry);

        // Detect entity-local inputs (same logic as partitioned path).
        bool has_entity_local_inputs = false;
        if (auxiliary_input_registry_) {
            for (const auto& [mn, rn] : entry.input_bindings) {
                if (auxiliary_input_registry_->hasInput(rn) &&
                    auxiliary_input_registry_->isEntityLocal(rn)) {
                    has_entity_local_inputs = true;
                    break;
                }
            }
        }

        const auto& emap = entry.entity_map;

        // Per-entity assembly.
        for (std::size_t e = 0; e < n_entities; ++e) {
            auto entity_x = blk.gatherEntityWork(e);
            auto entity_committed = blk.gatherEntityCommitted(e);
            const auto row_base = block_offset + e * static_cast<std::size_t>(dim);
            const auto orig_e = emap.empty() ? e : emap[e];
            const auto entity_committed_rate = gatherMonolithicCommittedRate(entry, e);
            auto temporal = buildMonolithicAuxiliaryTemporalEvaluation(
                entry.stepper_spec, blk, e, entity_x, entity_committed, entity_committed_rate, mono_state);

            // Rebuild inputs per entity when entity-local bindings exist.
            if (has_entity_local_inputs && auxiliary_input_registry_) {
                bound_inputs.clear();
                if (auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(entry.model.get())) {
                    for (const auto& inp : built->signature().inputs) {
                        auto bi = entry.input_bindings.find(inp.name);
                        if (bi != entry.input_bindings.end()) {
                            auto v = auxiliary_input_registry_->valuesOf(bi->second, orig_e);
                            bound_inputs.insert(bound_inputs.end(), v.begin(), v.end());
                        } else {
                            bound_inputs.resize(bound_inputs.size() + static_cast<std::size_t>(inp.size), 0.0);
                        }
                    }
                } else {
                    rebuildGenericInputsForEntity(entry, orig_e, bound_inputs);
                }
            }

            AuxiliaryLocalContext ctx;
            ctx.time = time;
            ctx.dt = dt;
            ctx.effective_dt = dt;
            ctx.x = entity_x;
            ctx.xdot = temporal.xdot;
            ctx.history = temporal.history_spans;
            ctx.inputs = bound_inputs;
            ctx.params = params;
            ctx.entity_index = e;

            std::vector<Real> entity_res(static_cast<std::size_t>(dim));
            AuxiliaryResidualRequest res_req;
            res_req.residual = entity_res;
            entry.model->evaluateResidual(ctx, res_req);

            for (int i = 0; i < dim; ++i) {
                residual_out[row_base + static_cast<std::size_t>(i)] += entity_res[static_cast<std::size_t>(i)];
            }

            // Evaluate Jacobian (if derivative provider available).
            if (entry.deriv_provider) {
                std::vector<Real> entity_jac(static_cast<std::size_t>(dim * dim));
                AuxiliaryJacobianRequest jac_req;
                jac_req.dF_dx = entity_jac;
                jac_req.n = dim;
                jac_req.want_dF_dxdot = true;
                std::vector<Real> entity_dFdxdot(static_cast<std::size_t>(dim * dim), 0.0);
                jac_req.dF_dxdot = entity_dFdxdot;
                entry.deriv_provider->evaluateJacobian(*entry.model, ctx, jac_req);

                if (temporal.dxdot_dx_coeff != Real(0.0)) {
                    for (std::size_t i = 0; i < entity_jac.size(); ++i) {
                        entity_jac[i] += temporal.dxdot_dx_coeff * entity_dFdxdot[i];
                    }
                }

                for (int i = 0; i < dim; ++i) {
                    for (int j = 0; j < dim; ++j) {
                        const auto gi = row_base + static_cast<std::size_t>(i);
                        const auto gj = row_base + static_cast<std::size_t>(j);
                        jacobian_out[gi * n_total + gj] +=
                            entity_jac[static_cast<std::size_t>(i * dim + j)];
                    }
                }
            }
        }
    }
}

MixedSystemLayout FESystem::composeMixedSystemLayout(std::size_t n_field_unknowns) const
{
    if (auxiliary_operator_registry_ && auxiliary_operator_registry_->isLayoutFinalized()) {
        return auxiliary_operator_registry_->composeMixedLayout(n_field_unknowns);
    }
    MixedSystemLayout layout;
    layout.n_field_unknowns = n_field_unknowns;
    layout.total_unknowns = n_field_unknowns;
    return layout;
}

backends::SolverOptions FESystem::augmentSolverOptions(const backends::SolverOptions& base,
                                                       std::size_t n_field_unknowns) const
{
    backends::SolverOptions options = base;
    std::size_t effective_field_unknowns = n_field_unknowns;
    if (effective_field_unknowns == 0 && field_map_.isFinalized() && field_map_.totalDofs() > 0) {
        effective_field_unknowns = static_cast<std::size_t>(field_map_.totalDofs());
    }

    backends::MixedBlockLayout mixed_layout;
    mixed_layout.field_unknowns = static_cast<GlobalIndex>(effective_field_unknowns);

    if (field_map_.isFinalized()) {
        for (std::size_t field_idx = 0; field_idx < field_map_.numFields(); ++field_idx) {
            const auto& field = field_map_.getField(field_idx);
            const auto [begin, end] = field_map_.getFieldDofRange(field_idx);
            FE_THROW_IF(begin < 0 || end < begin, InvalidStateException,
                        "FESystem::augmentSolverOptions: invalid field DOF range for '"
                        + field.name + "'");
            if (effective_field_unknowns > 0) {
                FE_THROW_IF(static_cast<std::size_t>(end) > effective_field_unknowns,
                            InvalidArgumentException,
                            "FESystem::augmentSolverOptions: requested field unknown count "
                            + std::to_string(effective_field_unknowns)
                            + " is smaller than finalized field range for '" + field.name + "'");
            }

            backends::MixedBlockDescriptor block;
            block.name = field.name;
            block.offset = begin;
            block.size = end - begin;
            block.role = inferFieldBlockRole(field.name, base);
            block.kind = backends::MixedBlockKind::Field;
            mixed_layout.blocks.push_back(std::move(block));
        }
    }

    const auto mixed_system = composeMixedSystemLayout(effective_field_unknowns);
    mixed_layout.auxiliary_unknowns = static_cast<GlobalIndex>(mixed_system.n_aux_unknowns);
    mixed_layout.total_unknowns = static_cast<GlobalIndex>(mixed_system.total_unknowns);

    for (const auto& aux_block : mixed_system.aux_layout.blocks) {
        backends::MixedBlockDescriptor block;
        block.name = aux_block.name;
        block.offset = static_cast<GlobalIndex>(
            mixed_system.aux_layout.mixed_system_offset + aux_block.offset);
        block.size = static_cast<GlobalIndex>(aux_block.n_unknowns);
        block.role = aux_block.backend_role;
        block.kind = backends::MixedBlockKind::Auxiliary;
        block.block_diagonal_suitable = aux_block.block_diagonal_suitable;
        block.special_precondition =
            (aux_block.role == AuxiliaryBlockRole::SpecialPrecondition);
        block.schur_eliminable = aux_block.schur_eliminable;
        block.schur_complement_partner = aux_block.schur_complement_partner;
        block.assembly_mode = aux_block.assembly_mode;
        block.row_ownership = aux_block.row_ownership;
        block.single_owner_rank = aux_block.single_owner_rank;
        block.row_owner_ranks = aux_block.row_owner_ranks;
        mixed_layout.blocks.push_back(std::move(block));
    }

    mixed_layout.primary_block =
        uniqueMixedBlockIndexForRole(mixed_layout, backends::BlockRole::PrimaryField);
    mixed_layout.constraint_block =
        uniqueMixedBlockIndexForRole(mixed_layout, backends::BlockRole::ConstraintField);

    options.mixed_block_layout = mixed_layout;
    addUnambiguousRoleMappings(options, *options.mixed_block_layout);
    return options;
}

backends::SolverOptions FESystem::augmentSolverOptions(const backends::SolverOptions& base) const
{
    const auto n_field_unknowns =
        is_setup_ ? static_cast<std::size_t>(dof_handler_.getNumDofs()) : std::size_t{0};
    return augmentSolverOptions(base, n_field_unknowns);
}

FESystem::AuxiliaryScopeResolution
FESystem::resolveAuxiliaryDeploymentScope_(DeployedAuxEntry& entry)
{
    std::size_t entity_count = entry.explicit_entity_count;
    std::size_t owned_entity_count = entity_count;

    if (entity_count == 0) {
        switch (entry.spec.scope) {
            case AuxiliaryStateScope::Global:
                entity_count = 1;
                owned_entity_count = entity_count;
                break;
            case AuxiliaryStateScope::Node:
                if (mesh_access_) {
                    entity_count =
                        static_cast<std::size_t>(std::max<GlobalIndex>(0, mesh_access_->numVertices()));
                    owned_entity_count =
                        static_cast<std::size_t>(std::max<GlobalIndex>(0, mesh_access_->numOwnedVertices()));
                    FE_THROW_IF(owned_entity_count > entity_count, InvalidStateException,
                                "FESystem::finalizeAuxiliaryLayout: mesh reports "
                                "numOwnedVertices() > numVertices() for Node scope");
                } else {
                    FE_THROW(InvalidStateException,
                             "FESystem::finalizeAuxiliaryLayout: Node scope requires "
                             "mesh vertex count via IMeshAccess::numVertices() or "
                             "an explicit .entityCount()");
                }
                break;
            case AuxiliaryStateScope::Cell:
                if (mesh_access_) {
                    entry.entity_map.clear();
                    mesh_access_->forEachOwnedCell([&](GlobalIndex cell_id) {
                        if (cell_id >= 0) {
                            entry.entity_map.push_back(static_cast<std::size_t>(cell_id));
                        }
                    });
                    entity_count = entry.entity_map.size();
                    const auto reported_owned =
                        static_cast<std::size_t>(mesh_access_->numOwnedCells());
                    FE_THROW_IF(reported_owned != entity_count, InvalidStateException,
                                "FESystem::finalizeAuxiliaryLayout: Cell scope owned-cell "
                                "iteration count does not match numOwnedCells()");
                } else {
                    entity_count = 1;
                }
                owned_entity_count = entity_count;
                break;
            case AuxiliaryStateScope::Boundary:
                entity_count = 1;
                owned_entity_count = entity_count;
                break;
            case AuxiliaryStateScope::Facet:
                if (mesh_access_) {
                    entity_count = static_cast<std::size_t>(mesh_access_->numBoundaryFaces());
                } else {
                    entity_count = 1;
                }
                owned_entity_count = entity_count;
                break;
            case AuxiliaryStateScope::QuadraturePoint:
                if (!entry.qp_offsets.empty()) {
                    entity_count = entry.qp_offsets.back();
                    owned_entity_count = entity_count;
                } else {
                    entity_count = 0;
                    owned_entity_count = 0;
                }
                break;
            case AuxiliaryStateScope::Region:
                if (topology_context_) {
                    entity_count = static_cast<std::size_t>(std::max(0, topology_context_->numRegions()));
                } else if (mesh_access_) {
                    auto topo = analysis::TopologyAnalysisContext::build(*mesh_access_);
                    entity_count = static_cast<std::size_t>(std::max(0, topo.numRegions()));
                } else {
                    entity_count = 1;
                }
                owned_entity_count = entity_count;
                break;
        }
    } else if (entry.spec.scope == AuxiliaryStateScope::Node && mesh_access_) {
        const auto mesh_entity_count =
            static_cast<std::size_t>(std::max<GlobalIndex>(0, mesh_access_->numVertices()));
        const auto mesh_owned_entity_count =
            static_cast<std::size_t>(std::max<GlobalIndex>(0, mesh_access_->numOwnedVertices()));
        FE_THROW_IF(mesh_owned_entity_count > mesh_entity_count, InvalidStateException,
                    "FESystem::finalizeAuxiliaryLayout: mesh reports "
                    "numOwnedVertices() > numVertices() for Node scope");
        FE_THROW_IF(entity_count != mesh_entity_count, InvalidArgumentException,
                    "FESystem::finalizeAuxiliaryLayout: Node scope instance '" +
                    entry.instance_name + "' requested entityCount()=" +
                    std::to_string(entity_count) + " but the mesh exposes " +
                    std::to_string(mesh_entity_count) +
                    " vertices. Omit entityCount() and let the backend derive it.");
        owned_entity_count = mesh_owned_entity_count;
    }

    const auto& region = entry.spec.deployment_region;
    if (region.isRestricted() &&
        entry.spec.scope != AuxiliaryStateScope::Boundary &&
        entry.spec.scope != AuxiliaryStateScope::Global) {
        if (!region.explicit_entities.empty()) {
            entry.entity_map = region.explicit_entities;
        } else if (mesh_access_) {
            auto parse_region_identity = [&](std::string_view label) {
                try {
                    return std::stoi(region.identity);
                } catch (...) {
                    FE_THROW(InvalidArgumentException,
                             "FESystem::finalizeAuxiliaryLayout: " +
                                 std::string(label) +
                                 " identity must be an integer, got '" +
                                 region.identity + "'");
                }
                return 0;
            };

            auto sort_unique = [](std::vector<std::size_t>& values) {
                std::sort(values.begin(), values.end());
                values.erase(std::unique(values.begin(), values.end()), values.end());
            };

            auto union_region_ids_across_ranks =
                [&](std::vector<std::size_t> region_ids) {
                    sort_unique(region_ids);
#if FE_HAS_MPI
                    int mpi_initialized = 0;
                    MPI_Initialized(&mpi_initialized);
                    if (mpi_initialized) {
                        MPI_Comm comm = is_setup_ ? dof_handler_.mpiComm() : MPI_COMM_WORLD;
                        int comm_size = 1;
                        MPI_Comm_size(comm, &comm_size);
                        if (comm_size > 1) {
                            std::optional<analysis::TopologyAnalysisContext> local_topology;
                            const auto* topo = topology_context_ ? &*topology_context_ : nullptr;
                            if (topo == nullptr && mesh_access_) {
                                local_topology =
                                    analysis::TopologyAnalysisContext::build(*mesh_access_);
                                topo = &*local_topology;
                            }
                            const auto n_regions = topo
                                ? static_cast<std::size_t>(std::max(0, topo->numRegions()))
                                : std::size_t{0};
                            if (n_regions > 0u) {
                                FE_THROW_IF(
                                    n_regions >
                                        static_cast<std::size_t>(
                                            std::numeric_limits<int>::max()),
                                    InvalidStateException,
                                    "FESystem::finalizeAuxiliaryLayout: too many "
                                    "topology regions for MPI region-id union");
                                std::vector<int> local_flags(n_regions, 0);
                                for (const auto region_id : region_ids) {
                                    FE_THROW_IF(region_id >= n_regions,
                                                InvalidArgumentException,
                                                "FESystem::finalizeAuxiliaryLayout: "
                                                "Region-scope deployment references "
                                                "topology region " +
                                                    std::to_string(region_id) +
                                                    " but only " +
                                                    std::to_string(n_regions) +
                                                    " regions exist");
                                    local_flags[region_id] = 1;
                                }
                                std::vector<int> global_flags(n_regions, 0);
                                MPI_Allreduce(local_flags.data(),
                                              global_flags.data(),
                                              static_cast<int>(n_regions),
                                              MPI_INT,
                                              MPI_MAX,
                                              comm);
                                region_ids.clear();
                                for (std::size_t region_id = 0; region_id < n_regions;
                                     ++region_id) {
                                    if (global_flags[region_id] != 0) {
                                        region_ids.push_back(region_id);
                                    }
                                }
                            }
                        }
                    }
#endif
                    return region_ids;
                };

            auto cells_to_nodes = [&](std::span<const std::size_t> cells) {
                std::vector<std::size_t> nodes_out;
                std::vector<GlobalIndex> cell_nodes;
                for (const auto cell : cells) {
                    cell_nodes.clear();
                    mesh_access_->getCellNodes(static_cast<GlobalIndex>(cell), cell_nodes);
                    for (const auto node : cell_nodes) {
                        if (node >= 0) {
                            nodes_out.push_back(static_cast<std::size_t>(node));
                        }
                    }
                }
                sort_unique(nodes_out);
                return nodes_out;
            };

            auto face_cells_to_nodes =
                [&](std::span<const std::pair<std::size_t, std::size_t>> face_cells) {
                std::vector<std::size_t> nodes_out;
                std::vector<GlobalIndex> cell_nodes;
                for (const auto& [face, cell] : face_cells) {
                    cell_nodes.clear();
                    const auto cell_id = static_cast<GlobalIndex>(cell);
                    mesh_access_->getCellNodes(cell_id, cell_nodes);
                    const auto local_face = mesh_access_->getLocalFaceIndex(
                        static_cast<GlobalIndex>(face), cell_id);
                    const auto ref = elements::ReferenceElement::create(
                        mesh_access_->getCellType(cell_id));
                    const auto& face_nodes =
                        ref.face_nodes(static_cast<std::size_t>(local_face));
                    for (const auto local_node : face_nodes) {
                        const auto node_idx = static_cast<std::size_t>(local_node);
                        FE_THROW_IF(node_idx >= cell_nodes.size(), InvalidStateException,
                                    "FESystem::finalizeAuxiliaryLayout: face "
                                    "node index out of range while expanding "
                                    "Node deployment region for auxiliary "
                                    "instance '" +
                                        entry.instance_name + "'");
                        const auto node = cell_nodes[node_idx];
                        if (node >= 0) {
                            nodes_out.push_back(static_cast<std::size_t>(node));
                        }
                    }
                }
                sort_unique(nodes_out);
                return nodes_out;
            };

            auto cells_to_regions = [&](std::span<const std::size_t> cells) {
                std::vector<std::size_t> regions_out;
                std::optional<analysis::TopologyAnalysisContext> local_topology;
                const auto* topo = topology_context_ ? &*topology_context_ : nullptr;
                if (topo == nullptr) {
                    local_topology = analysis::TopologyAnalysisContext::build(*mesh_access_);
                    topo = &*local_topology;
                }
                for (const auto cell : cells) {
                    const int region_id = topo->regionForCell(static_cast<GlobalIndex>(cell));
                    if (region_id >= 0) {
                        regions_out.push_back(static_cast<std::size_t>(region_id));
                    }
                }
                sort_unique(regions_out);
                return union_region_ids_across_ranks(std::move(regions_out));
            };

            auto assign_from_cells = [&](std::vector<std::size_t> cells) {
                sort_unique(cells);
                switch (entry.spec.scope) {
                    case AuxiliaryStateScope::Cell:
                    case AuxiliaryStateScope::QuadraturePoint:
                        entry.entity_map = std::move(cells);
                        break;
                    case AuxiliaryStateScope::Node:
                        entry.entity_map = cells_to_nodes(cells);
                        break;
                    case AuxiliaryStateScope::Region:
                        entry.entity_map = cells_to_regions(cells);
                        break;
                    case AuxiliaryStateScope::Facet:
                        FE_THROW(InvalidArgumentException,
                                 "FESystem::finalizeAuxiliaryLayout: CellSet/MaterialIdSet "
                                 "deployment cannot be expanded to Facet scope for auxiliary "
                                 "instance '" + entry.instance_name + "'");
                        break;
                    case AuxiliaryStateScope::Global:
                    case AuxiliaryStateScope::Boundary:
                        break;
                }
            };

            auto assign_from_boundary_faces =
                [&](std::vector<std::size_t> faces,
                    std::vector<std::size_t> cells,
                    std::vector<std::pair<std::size_t, std::size_t>> face_cells) {
                    sort_unique(faces);
                    sort_unique(cells);
                    std::sort(face_cells.begin(), face_cells.end());
                    face_cells.erase(std::unique(face_cells.begin(), face_cells.end()),
                                     face_cells.end());
                    switch (entry.spec.scope) {
                        case AuxiliaryStateScope::Facet:
                            entry.entity_map = std::move(faces);
                            break;
                        case AuxiliaryStateScope::Cell:
                        case AuxiliaryStateScope::QuadraturePoint:
                            entry.entity_map = std::move(cells);
                            break;
                        case AuxiliaryStateScope::Node:
                            entry.entity_map = face_cells_to_nodes(face_cells);
                            break;
                        case AuxiliaryStateScope::Region:
                            entry.entity_map = cells_to_regions(cells);
                            break;
                        case AuxiliaryStateScope::Global:
                        case AuxiliaryStateScope::Boundary:
                            break;
                    }
                };

            auto topology_region_id = [&](std::string_view label) {
                const int parsed_region_id = parse_region_identity(label);
                FE_THROW_IF(parsed_region_id < 0,
                            InvalidArgumentException,
                            "FESystem::finalizeAuxiliaryLayout: " +
                                std::string(label) +
                                " identity must be a nonnegative topology "
                                "region id, got '" +
                                region.identity + "'");
                ensureAuxiliaryRegionLookupCache_();
                const auto region_id = static_cast<std::size_t>(parsed_region_id);
                const auto n_regions = auxiliary_region_lookup_cache_
                    ? auxiliary_region_lookup_cache_->region_ids.size()
                    : std::size_t{0};
                FE_THROW_IF(region_id >= n_regions,
                            InvalidArgumentException,
                            "FESystem::finalizeAuxiliaryLayout: " +
                                std::string(label) +
                                " references topology region " +
                                std::to_string(region_id) + " but only " +
                                std::to_string(n_regions) + " regions exist");
                return region_id;
            };

            auto assign_from_topology_region = [&](std::size_t region_id) {
                FE_THROW_IF(!auxiliary_region_lookup_cache_,
                            InvalidStateException,
                            "FESystem::finalizeAuxiliaryLayout: missing topology "
                            "region lookup cache while expanding auxiliary "
                            "deployment");
                const auto& cache = *auxiliary_region_lookup_cache_;
                switch (entry.spec.scope) {
                    case AuxiliaryStateScope::Cell:
                    case AuxiliaryStateScope::QuadraturePoint:
                        entry.entity_map = cache.region_to_cells[region_id];
                        break;
                    case AuxiliaryStateScope::Node:
                        entry.entity_map = cache.region_to_nodes[region_id];
                        break;
                    case AuxiliaryStateScope::Region:
                        entry.entity_map = {region_id};
                        break;
                    case AuxiliaryStateScope::Facet:
                        entry.entity_map = cache.region_to_interface_faces[region_id];
                        break;
                    case AuxiliaryStateScope::Global:
                    case AuxiliaryStateScope::Boundary:
                        break;
                }
            };

            switch (region.kind) {
                case AuxiliaryRegionKind::CellSet:
                case AuxiliaryRegionKind::MaterialIdSet: {
                    const int target_id = parse_region_identity("CellSet/MaterialIdSet");
                    std::vector<std::size_t> cells;
                    auto append_matching_cell = [&](GlobalIndex cell_id) {
                        if (mesh_access_->getCellDomainId(cell_id) == target_id) {
                            cells.push_back(static_cast<std::size_t>(cell_id));
                        }
                    };
                    if (entry.spec.scope == AuxiliaryStateScope::Region) {
                        mesh_access_->forEachCell(append_matching_cell);
                    } else {
                        mesh_access_->forEachOwnedCell(append_matching_cell);
                    }
                    assign_from_cells(std::move(cells));
                    break;
                }
                case AuxiliaryRegionKind::BoundarySet: {
                    const int marker = parse_region_identity("BoundarySet");
                    std::vector<std::size_t> faces;
                    std::vector<std::size_t> cells;
                    std::vector<std::pair<std::size_t, std::size_t>> face_cells;
                    mesh_access_->forEachBoundaryFace(marker,
                        [&](GlobalIndex face_id, GlobalIndex cell_id) {
                            if (face_id >= 0) {
                                faces.push_back(static_cast<std::size_t>(face_id));
                            }
                            if (cell_id >= 0) {
                                cells.push_back(static_cast<std::size_t>(cell_id));
                            }
                            if (face_id >= 0 && cell_id >= 0) {
                                face_cells.emplace_back(static_cast<std::size_t>(face_id),
                                                        static_cast<std::size_t>(cell_id));
                            }
                        });
                    assign_from_boundary_faces(std::move(faces),
                                               std::move(cells),
                                               std::move(face_cells));
                    break;
                }
                case AuxiliaryRegionKind::InterfaceSet: {
                    const int marker = parse_region_identity("InterfaceSet");
                    (void)marker;
                    std::vector<std::size_t> faces;
                    std::vector<std::size_t> cells;
                    std::vector<std::pair<std::size_t, std::size_t>> face_cells;
                    mesh_access_->forEachInteriorFace(
                        [&](GlobalIndex face_id, GlobalIndex c0, GlobalIndex c1) {
                            if (face_id >= 0) {
                                faces.push_back(static_cast<std::size_t>(face_id));
                            }
                            if (c0 >= 0) {
                                cells.push_back(static_cast<std::size_t>(c0));
                            }
                            if (c1 >= 0) {
                                cells.push_back(static_cast<std::size_t>(c1));
                            }
                            if (face_id >= 0 && c0 >= 0) {
                                face_cells.emplace_back(static_cast<std::size_t>(face_id),
                                                        static_cast<std::size_t>(c0));
                            }
                            if (face_id >= 0 && c1 >= 0) {
                                face_cells.emplace_back(static_cast<std::size_t>(face_id),
                                                        static_cast<std::size_t>(c1));
                            }
                        });
	                    assign_from_boundary_faces(std::move(faces),
	                                               std::move(cells),
	                                               std::move(face_cells));
	                    break;
	                }
	                case AuxiliaryRegionKind::TopologyRegion: {
	                    assign_from_topology_region(
	                        topology_region_id("TopologyRegion"));
	                    break;
	                }
	                case AuxiliaryRegionKind::WholeDomain:
	                    break;
                case AuxiliaryRegionKind::FormulationDefined:
                    FE_THROW(InvalidArgumentException,
                             "FESystem::finalizeAuxiliaryLayout: FormulationDefined "
                             "deployment region for auxiliary instance '" +
                             entry.instance_name + "' requires explicit_entities");
                    break;
            }
            FE_THROW_IF(entry.entity_map.empty() &&
                        region.kind != AuxiliaryRegionKind::WholeDomain,
                        InvalidStateException,
                        "FESystem::finalizeAuxiliaryLayout: marker-based region '"
                        + region.identity + "' expanded to 0 entities");
        } else {
            FE_THROW(InvalidStateException,
                     "FESystem::finalizeAuxiliaryLayout: deployment region "
                     "kind '" + region.identity + "' requires mesh access "
                     "for marker-based entity expansion, but no mesh was "
                     "provided to FESystem");
        }
        if (!entry.entity_map.empty()) {
            if (entry.spec.scope == AuxiliaryStateScope::QuadraturePoint) {
                owned_entity_count = entry.entity_map.size();
            } else if (entry.spec.scope == AuxiliaryStateScope::Node && mesh_access_) {
                const auto mesh_entity_count =
                    static_cast<std::size_t>(std::max<GlobalIndex>(0, mesh_access_->numVertices()));
                const auto mesh_owned_entity_count =
                    static_cast<std::size_t>(std::max<GlobalIndex>(0, mesh_access_->numOwnedVertices()));
                FE_THROW_IF(mesh_owned_entity_count > mesh_entity_count, InvalidStateException,
                            "FESystem::finalizeAuxiliaryLayout: mesh reports "
                            "numOwnedVertices() > numVertices() for Node scope");
                for (const auto node : entry.entity_map) {
                    FE_THROW_IF(node >= mesh_entity_count, InvalidArgumentException,
                                "FESystem::finalizeAuxiliaryLayout: Node scope instance '" +
                                entry.instance_name + "' references node " +
                                std::to_string(node) + " but the mesh exposes only " +
                                std::to_string(mesh_entity_count) + " vertices");
                }
                const auto first_ghost = std::stable_partition(
                    entry.entity_map.begin(),
                    entry.entity_map.end(),
                    [mesh_owned_entity_count](std::size_t node) {
                        return node < mesh_owned_entity_count;
                    });
                entity_count = entry.entity_map.size();
                owned_entity_count = static_cast<std::size_t>(
                    std::distance(entry.entity_map.begin(), first_ghost));
            } else {
                entity_count = entry.entity_map.size();
                owned_entity_count = entity_count;
            }
        }
    } else if (region.isRestricted() &&
               entry.spec.scope == AuxiliaryStateScope::Global &&
               !region.explicit_entities.empty()) {
        FE_THROW(InvalidArgumentException,
                 "FESystem::finalizeAuxiliaryLayout: Global scope auxiliary instance '" +
                 entry.instance_name +
                 "' treats deployment_region as metadata and does not support explicit_entities");
    }

    if (entry.spec.scope == AuxiliaryStateScope::QuadraturePoint) {
        inferQuadraturePointLayout_(entry);
        if (!entry.materialized) {
            return {};
        }
        entity_count = entry.qp_offsets.back();
        owned_entity_count = entity_count;
    } else {
        entry.materialized = true;
    }

    return AuxiliaryScopeResolution{entity_count, owned_entity_count};
}

void FESystem::ensureAuxiliaryRegionLookupCache_()
{
    if (auxiliary_region_lookup_cache_.has_value()) {
        return;
    }

    AuxiliaryRegionLookupCache cache;
    if (!topology_context_ && mesh_access_) {
        topology_context_ = analysis::TopologyAnalysisContext::build(*mesh_access_);
    }

    auto sort_unique = [](auto& values) {
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
    };

    if (topology_context_) {
        const auto n_regions =
            static_cast<std::size_t>(std::max(0, topology_context_->numRegions()));
        cache.region_ids.resize(n_regions);
        std::iota(cache.region_ids.begin(), cache.region_ids.end(), std::size_t{0});
        cache.region_to_cells.resize(n_regions);
        cache.region_to_nodes.resize(n_regions);
        cache.region_to_boundary_markers.resize(n_regions);
        cache.region_to_interface_faces.resize(n_regions);
        cache.region_owner_ranks.assign(n_regions, -1);

        std::vector<GlobalIndex> cell_nodes;
        for (const auto& component : topology_context_->components) {
            if (component.region_id < 0) {
                continue;
            }
            const auto region_id = static_cast<std::size_t>(component.region_id);
            if (region_id >= n_regions) {
                continue;
            }

            auto& cells = cache.region_to_cells[region_id];
            cells.clear();
            cells.reserve(component.cell_indices.size());
            for (const auto cell : component.cell_indices) {
                if (cell < 0) {
                    continue;
                }
                cells.push_back(static_cast<std::size_t>(cell));
                if (!mesh_access_) {
                    continue;
                }
                cell_nodes.clear();
                mesh_access_->getCellNodes(cell, cell_nodes);
                auto& nodes = cache.region_to_nodes[region_id];
                for (const auto node : cell_nodes) {
                    if (node >= 0) {
                        nodes.push_back(static_cast<std::size_t>(node));
                    }
                }
            }

            sort_unique(cells);
            sort_unique(cache.region_to_nodes[region_id]);
            cache.region_to_boundary_markers[region_id].assign(
                component.boundary_markers.begin(),
                component.boundary_markers.end());
            sort_unique(cache.region_to_boundary_markers[region_id]);
        }

        if (interface_topology_context_) {
            for (std::size_t face_idx = 0; face_idx < interface_topology_context_->faces.size();
                 ++face_idx) {
                const auto& face = interface_topology_context_->faces[face_idx];
                auto append_face = [&](int region_id) {
                    if (region_id < 0) {
                        return;
                    }
                    const auto rid = static_cast<std::size_t>(region_id);
                    if (rid < cache.region_to_interface_faces.size()) {
                        cache.region_to_interface_faces[rid].push_back(face_idx);
                    }
                };
                append_face(face.minus_region);
                append_face(face.plus_region);
            }
            for (auto& faces : cache.region_to_interface_faces) {
                sort_unique(faces);
            }
        }

        const auto sentinel = std::numeric_limits<long long>::max();
        std::vector<long long> local_min_cell(n_regions, sentinel);

        int rank = 0;
#if FE_HAS_MPI
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        MPI_Comm comm = MPI_COMM_WORLD;
        int comm_size = 1;
        if (mpi_initialized) {
            comm = is_setup_ ? dof_handler_.mpiComm() : MPI_COMM_WORLD;
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &comm_size);
        }
#endif

        for (std::size_t region_id = 0; region_id < cache.region_to_cells.size();
             ++region_id) {
            for (const auto cell : cache.region_to_cells[region_id]) {
                const auto cell_id = static_cast<GlobalIndex>(cell);
                if (mesh_access_ && !mesh_access_->isOwnedCell(cell_id)) {
                    continue;
                }
                local_min_cell[region_id] =
                    std::min(local_min_cell[region_id],
                             static_cast<long long>(cell_id));
            }
        }

        std::vector<long long> global_min_cell = local_min_cell;
#if FE_HAS_MPI
        if (mpi_initialized && comm_size > 1 && n_regions > 0u) {
            FE_THROW_IF(n_regions >
                            static_cast<std::size_t>(std::numeric_limits<int>::max()),
                        InvalidStateException,
                        "FESystem::finalizeAuxiliaryLayout: too many topology "
                        "regions for MPI region-owner reduction");
            MPI_Allreduce(local_min_cell.data(),
                          global_min_cell.data(),
                          static_cast<int>(n_regions),
                          MPI_LONG_LONG,
                          MPI_MIN,
                          comm);
        }
#endif

        std::vector<int> local_owner_candidate(n_regions,
                                               std::numeric_limits<int>::max());
        for (std::size_t region_id = 0; region_id < n_regions; ++region_id) {
            if (global_min_cell[region_id] != sentinel &&
                local_min_cell[region_id] == global_min_cell[region_id]) {
                local_owner_candidate[region_id] = rank;
            }
        }

        std::vector<int> global_owner = local_owner_candidate;
#if FE_HAS_MPI
        if (mpi_initialized && comm_size > 1 && n_regions > 0u) {
            MPI_Allreduce(local_owner_candidate.data(),
                          global_owner.data(),
                          static_cast<int>(n_regions),
                          MPI_INT,
                          MPI_MIN,
                          comm);
        }
#endif

        for (std::size_t region_id = 0; region_id < n_regions; ++region_id) {
            FE_THROW_IF(global_min_cell[region_id] == sentinel ||
                            global_owner[region_id] ==
                                std::numeric_limits<int>::max(),
                        InvalidStateException,
                        "FESystem::finalizeAuxiliaryLayout: topology region " +
                            std::to_string(region_id) +
                            " has no owned cell from which to derive a "
                            "deterministic region row owner");
            cache.region_owner_ranks[region_id] = global_owner[region_id];
        }
    }

    auxiliary_region_lookup_cache_ = std::move(cache);
}

std::size_t FESystem::auxiliaryTopologyRegionInputEntityCount_() const
{
    if (topology_context_) {
        return static_cast<std::size_t>(std::max(0, topology_context_->numRegions()));
    }
    FE_THROW_IF(!mesh_access_, InvalidStateException,
                "FESystem::regionIntegral: topology-region-local inputs require a mesh");
    auto topology = analysis::TopologyAnalysisContext::build(*mesh_access_);
    const auto n_regions = static_cast<std::size_t>(std::max(0, topology.numRegions()));
    if (n_regions > 0u) {
        return n_regions;
    }
    return static_cast<std::size_t>(std::max<GlobalIndex>(1, mesh_access_->numCells()));
}

std::vector<GlobalIndex>
FESystem::auxiliaryTopologyRegionCells_(std::size_t region_id) const
{
    std::vector<GlobalIndex> cells;
    if (!mesh_access_) {
        return cells;
    }

    const analysis::TopologyAnalysisContext* topology =
        topology_context_ ? &*topology_context_ : nullptr;
    std::optional<analysis::TopologyAnalysisContext> local_topology;
    if (!topology) {
        local_topology = analysis::TopologyAnalysisContext::build(*mesh_access_);
        topology = &*local_topology;
    }

    if (!topology || region_id >= static_cast<std::size_t>(std::max(0, topology->numRegions()))) {
        return cells;
    }

    mesh_access_->forEachCell([&](GlobalIndex cell_id) {
        const auto cell_region = topology->regionForCell(cell_id);
        if (cell_region >= 0 &&
            static_cast<std::size_t>(cell_region) == region_id) {
            cells.push_back(cell_id);
        }
    });

    std::sort(cells.begin(), cells.end());
    cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    return cells;
}

AuxiliaryEntityRemapMetadata
FESystem::buildAuxiliaryEntityRemapMetadata_(
    const DeployedAuxEntry& entry,
    const AuxiliaryScopeResolution& resolution)
{
    AuxiliaryEntityRemapMetadata metadata;
    metadata.scope = entry.spec.scope;
    metadata.deployment_region = entry.spec.deployment_region;
    metadata.owned_entity_count = resolution.owned_entity_count;

    if (entry.spec.scope == AuxiliaryStateScope::QuadraturePoint) {
        metadata.entity_ids.resize(resolution.entity_count);
        std::iota(metadata.entity_ids.begin(), metadata.entity_ids.end(), std::size_t{0});
    } else if (!entry.entity_map.empty()) {
        metadata.entity_ids = entry.entity_map;
    } else {
        metadata.entity_ids.resize(resolution.entity_count);
        std::iota(metadata.entity_ids.begin(), metadata.entity_ids.end(), std::size_t{0});
    }

    if (entry.spec.scope == AuxiliaryStateScope::QuadraturePoint) {
        metadata.qp_offsets = entry.qp_offsets;
        if (!entry.entity_map.empty()) {
            metadata.qp_cell_ids = entry.entity_map;
        } else if (entry.qp_offsets.size() > 1u) {
            metadata.qp_cell_ids.resize(entry.qp_offsets.size() - 1u);
            std::iota(metadata.qp_cell_ids.begin(), metadata.qp_cell_ids.end(), std::size_t{0});
        }
    }

    if (entry.spec.layout_mode == AuxiliaryLayoutMode::Ragged) {
        metadata.component_offsets = entry.ragged_component_offsets;
    }

    if (entry.spec.scope == AuxiliaryStateScope::Region) {
        ensureAuxiliaryRegionLookupCache_();
        metadata.region_membership.reserve(metadata.entity_ids.size());
        for (const auto region_id : metadata.entity_ids) {
            AuxiliaryRegionMembershipMetadata membership;
            membership.region_id = region_id;
            if (auxiliary_region_lookup_cache_ &&
                region_id < auxiliary_region_lookup_cache_->region_to_cells.size()) {
                membership.cell_ids =
                    auxiliary_region_lookup_cache_->region_to_cells[region_id];
                membership.node_ids =
                    auxiliary_region_lookup_cache_->region_to_nodes[region_id];
                membership.boundary_markers =
                    auxiliary_region_lookup_cache_->region_to_boundary_markers[region_id];
                membership.interface_face_ids =
                    auxiliary_region_lookup_cache_->region_to_interface_faces[region_id];
            }
            metadata.region_membership.push_back(std::move(membership));
        }
    }

    return metadata;
}

std::vector<std::size_t> FESystem::buildAuxiliaryRaggedComponentOffsets_(
    const DeployedAuxEntry& entry,
    const AuxiliaryScopeResolution& resolution) const
{
    FE_THROW_IF(entry.spec.layout_mode != AuxiliaryLayoutMode::Ragged,
                InvalidArgumentException,
                "FESystem::buildAuxiliaryRaggedComponentOffsets_: instance '" +
                    entry.instance_name + "' is not ragged");

    const auto entity_count = resolution.entity_count;

    if (!entry.ragged_component_offsets.empty()) {
        FE_THROW_IF(entry.ragged_component_offsets.size() != entity_count + 1u,
                    InvalidArgumentException,
                    "FESystem::finalizeAuxiliaryLayout: raggedComponentOffsets() "
                    "for instance '" +
                        entry.instance_name + "' has " +
                        std::to_string(entry.ragged_component_offsets.size()) +
                        " entries but resolved scope has " +
                        std::to_string(entity_count) + " entities");
        FE_THROW_IF(entry.ragged_component_offsets.front() != 0u,
                    InvalidArgumentException,
                    "FESystem::finalizeAuxiliaryLayout: raggedComponentOffsets() "
                    "for instance '" +
                        entry.instance_name + "' must start at 0");
        FE_THROW_IF(!std::is_sorted(entry.ragged_component_offsets.begin(),
                                    entry.ragged_component_offsets.end()),
                    InvalidArgumentException,
                    "FESystem::finalizeAuxiliaryLayout: raggedComponentOffsets() "
                    "for instance '" +
                        entry.instance_name + "' must be nondecreasing");
        return entry.ragged_component_offsets;
    }

    FE_THROW_IF(!entry.ragged_entity_size_provider, InvalidArgumentException,
                "FESystem::finalizeAuxiliaryLayout: ragged instance '" +
                    entry.instance_name +
                    "' has no raggedEntitySize() provider or component offsets");

    switch (entry.spec.scope) {
        case AuxiliaryStateScope::Node:
        case AuxiliaryStateScope::Cell:
        case AuxiliaryStateScope::QuadraturePoint:
        case AuxiliaryStateScope::Region:
            break;
        case AuxiliaryStateScope::Global:
        case AuxiliaryStateScope::Boundary:
        case AuxiliaryStateScope::Facet:
            FE_THROW(NotImplementedException,
                     "FESystem::finalizeAuxiliaryLayout: ragged deployment for '" +
                         entry.instance_name +
                         "' is currently supported only for Node, Cell, "
                         "QuadraturePoint, and Region scopes");
    }

    std::vector<std::size_t> offsets(entity_count + 1u, 0u);
    for (std::size_t entity = 0; entity < entity_count; ++entity) {
        AuxiliaryRaggedEntityContext ctx;
        ctx.scope = entry.spec.scope;
        ctx.materialized_entity_index = entity;
        ctx.original_entity_id = entity;

        if (entry.spec.scope == AuxiliaryStateScope::QuadraturePoint) {
            FE_THROW_IF(entry.qp_offsets.empty(), InvalidStateException,
                        "FESystem::finalizeAuxiliaryLayout: QuadraturePoint "
                        "ragged instance '" +
                            entry.instance_name +
                            "' has no resolved qpOffsets()");
            auto upper = std::upper_bound(entry.qp_offsets.begin(),
                                          entry.qp_offsets.end(),
                                          entity);
            FE_THROW_IF(upper == entry.qp_offsets.begin(), InvalidStateException,
                        "FESystem::finalizeAuxiliaryLayout: unable to map flat "
                        "QP entity to a covered cell for ragged instance '" +
                            entry.instance_name + "'");
            auto cell_ordinal = static_cast<std::size_t>(
                std::distance(entry.qp_offsets.begin(), upper) - 1);
            if (cell_ordinal + 1u >= entry.qp_offsets.size()) {
                cell_ordinal = entry.qp_offsets.size() - 2u;
            }
            ctx.cell_id = entry.entity_map.empty()
                ? cell_ordinal
                : entry.entity_map.at(cell_ordinal);
            ctx.local_qp_index = entity - entry.qp_offsets.at(cell_ordinal);
            ctx.original_entity_id = entity;
        } else if (!entry.entity_map.empty()) {
            ctx.original_entity_id = entry.entity_map.at(entity);
        }

        const auto width = entry.ragged_entity_size_provider(ctx);
        FE_THROW_IF(width >
                        std::numeric_limits<std::size_t>::max() - offsets[entity],
                    InvalidArgumentException,
                    "FESystem::finalizeAuxiliaryLayout: raggedEntitySize() "
                    "overflow for instance '" +
                        entry.instance_name + "'");
        offsets[entity + 1u] = offsets[entity] + width;
    }
    return offsets;
}

std::vector<Real> FESystem::buildAuxiliaryRaggedInitialValues_(
    const DeployedAuxEntry& entry,
    std::span<const std::size_t> component_offsets) const
{
    FE_THROW_IF(component_offsets.empty(), InvalidArgumentException,
                "FESystem::buildAuxiliaryRaggedInitialValues_: empty offsets "
                "for instance '" +
                    entry.instance_name + "'");

    if (entry.initial_values.empty()) {
        return {};
    }

    const auto total_storage = component_offsets.back();
    if (entry.initial_values.size() == total_storage) {
        return entry.initial_values;
    }
    FE_THROW_IF(total_storage == 0u, InvalidArgumentException,
                "FESystem::finalizeAuxiliaryLayout: ragged initial values for "
                "instance '" +
                    entry.instance_name +
                    "' are non-empty but the resolved ragged storage is empty");

    if (entry.initial_values.size() == 1u) {
        return std::vector<Real>(total_storage, entry.initial_values.front());
    }

    const auto model_dim = static_cast<std::size_t>(entry.spec.size);
    if (entry.initial_values.size() == model_dim) {
        std::vector<Real> full_init(total_storage, Real{0.0});
        for (std::size_t entity = 0; entity + 1u < component_offsets.size(); ++entity) {
            const auto begin = component_offsets[entity];
            const auto end = component_offsets[entity + 1u];
            const auto width = end - begin;
            FE_THROW_IF(width != model_dim, InvalidArgumentException,
                        "FESystem::finalizeAuxiliaryLayout: ragged initial "
                        "state for instance '" +
                            entry.instance_name +
                            "' has model-dimension values, but at least one "
                            "entity has a different ragged component count");
            std::copy(entry.initial_values.begin(),
                      entry.initial_values.end(),
                      full_init.begin() + static_cast<std::ptrdiff_t>(begin));
        }
        return full_init;
    }

    FE_THROW(InvalidArgumentException,
             "FESystem::finalizeAuxiliaryLayout: ragged initial_values size (" +
                 std::to_string(entry.initial_values.size()) +
                 ") for instance '" + entry.instance_name +
                 "' must be empty, exactly the resolved storage size (" +
                 std::to_string(total_storage) +
                 "), a single broadcast value, or the model dimension for "
                 "uniform-width ragged entities");
}

void FESystem::validateEntityLocalAuxiliaryBindings_() const
{
    if (!auxiliary_input_registry_) {
        return;
    }

    auto coveredOriginalEntities = [&](const DeployedAuxEntry& entry) {
        std::vector<std::size_t> entities;
        if (entry.spec.scope == AuxiliaryStateScope::QuadraturePoint) {
            if (!entry.entity_map.empty()) {
                return entry.entity_map;
            }
            if (entry.qp_offsets.size() > 1u) {
                entities.resize(entry.qp_offsets.size() - 1u);
                std::iota(entities.begin(), entities.end(), std::size_t{0});
            }
            return entities;
        }

        if (!entry.entity_map.empty()) {
            return entry.entity_map;
        }

        std::size_t count = entry.explicit_entity_count;
        if (auxiliary_state_manager_ &&
            auxiliary_state_manager_->hasBlock(entry.instance_name)) {
            count = auxiliary_state_manager_->getBlock(entry.instance_name).entityCount();
        }
        entities.resize(count);
        std::iota(entities.begin(), entities.end(), std::size_t{0});
        return entities;
    };
    auto scopeName = [](AuxiliaryStateScope scope) -> const char* {
        switch (scope) {
            case AuxiliaryStateScope::Global: return "Global";
            case AuxiliaryStateScope::Boundary: return "Boundary";
            case AuxiliaryStateScope::Node: return "Node";
            case AuxiliaryStateScope::Cell: return "Cell";
            case AuxiliaryStateScope::QuadraturePoint: return "QuadraturePoint";
            case AuxiliaryStateScope::Region: return "Region";
            case AuxiliaryStateScope::Facet: return "Facet";
        }
        return "Unknown";
    };
    auto producerName = [](AuxiliaryInputProducer producer) -> const char* {
        switch (producer) {
            case AuxiliaryInputProducer::BoundaryReduction: return "BoundaryReduction";
            case AuxiliaryInputProducer::FormulationCallback: return "FormulationCallback";
            case AuxiliaryInputProducer::ParameterDerived: return "ParameterDerived";
            case AuxiliaryInputProducer::DirectUserData: return "DirectUserData";
            case AuxiliaryInputProducer::AuxiliaryOutput: return "AuxiliaryOutput";
            case AuxiliaryInputProducer::SampledStateField: return "SampledStateField";
            case AuxiliaryInputProducer::CoupledField: return "CoupledField";
            case AuxiliaryInputProducer::CellAverage: return "CellAverage";
            case AuxiliaryInputProducer::CellSample: return "CellSample";
            case AuxiliaryInputProducer::DomainAverage: return "DomainAverage";
            case AuxiliaryInputProducer::DomainIntegral: return "DomainIntegral";
            case AuxiliaryInputProducer::RegionAverage: return "RegionAverage";
            case AuxiliaryInputProducer::RegionIntegral: return "RegionIntegral";
            case AuxiliaryInputProducer::SampledBoundaryTrace: return "SampledBoundaryTrace";
            case AuxiliaryInputProducer::CoupledBoundaryTrace: return "CoupledBoundaryTrace";
            case AuxiliaryInputProducer::SampledBoundaryReduction:
                return "SampledBoundaryReduction";
            case AuxiliaryInputProducer::CoupledBoundaryReduction:
                return "CoupledBoundaryReduction";
        }
        return "Unknown";
    };
    auto isFEBackedEntityLocalProducer = [](AuxiliaryInputProducer producer) noexcept {
        switch (producer) {
            case AuxiliaryInputProducer::SampledStateField:
            case AuxiliaryInputProducer::CoupledField:
            case AuxiliaryInputProducer::CellAverage:
            case AuxiliaryInputProducer::CellSample:
            case AuxiliaryInputProducer::RegionAverage:
            case AuxiliaryInputProducer::RegionIntegral:
            case AuxiliaryInputProducer::SampledBoundaryTrace:
            case AuxiliaryInputProducer::CoupledBoundaryTrace:
                return true;
            default:
                return false;
        }
    };
    auto isRaggedScopeCompatibleFEBackedProducer =
        [](AuxiliaryStateScope scope, AuxiliaryInputProducer producer) noexcept {
        switch (scope) {
            case AuxiliaryStateScope::Node:
                return producer == AuxiliaryInputProducer::SampledStateField ||
                       producer == AuxiliaryInputProducer::CoupledField;
            case AuxiliaryStateScope::Cell:
            case AuxiliaryStateScope::QuadraturePoint:
                return producer == AuxiliaryInputProducer::CellAverage ||
                       producer == AuxiliaryInputProducer::CellSample;
            case AuxiliaryStateScope::Region:
                return producer == AuxiliaryInputProducer::RegionAverage ||
                       producer == AuxiliaryInputProducer::RegionIntegral;
            case AuxiliaryStateScope::Global:
            case AuxiliaryStateScope::Boundary:
            case AuxiliaryStateScope::Facet:
                return false;
        }
        return false;
    };

    for (const auto& entry : deployed_aux_entries_) {
        if (!entry.materialized || entry.input_bindings.empty()) {
            continue;
        }

        const bool local_condensed_cell_like =
            entry.local_condensed &&
            (entry.spec.scope == AuxiliaryStateScope::Cell ||
             entry.spec.scope == AuxiliaryStateScope::QuadraturePoint);
        const auto original_entities = coveredOriginalEntities(entry);
        const auto max_entity_it =
            std::max_element(original_entities.begin(), original_entities.end());

        for (const auto& [model_input, registry_input] : entry.input_bindings) {
            (void)model_input;
            if (!auxiliary_input_registry_->hasInput(registry_input) ||
                !auxiliary_input_registry_->isEntityLocal(registry_input)) {
                continue;
            }

            const auto& input_spec =
                auxiliary_input_registry_->specOf(registry_input);
            if (max_entity_it != original_entities.end()) {
                FE_THROW_IF(
                    *max_entity_it >= input_spec.entity_count,
                    InvalidArgumentException,
                    "FESystem::finalizeAuxiliaryLayout: auxiliary instance '" +
                        entry.instance_name + "' binds entity-local input '" +
                        registry_input + "' with entity_count=" +
                        std::to_string(input_spec.entity_count) +
                        " but its " + std::string(scopeName(entry.spec.scope)) +
                        " deployment covers original entity id " +
                        std::to_string(*max_entity_it) +
                        ". Entity-local bindings are indexed by stable original "
                        "entity ids; register the provider over the full original "
                        "entity map or use a scope-matched FE quantity provider.");
            }

            FE_THROW_IF(
                local_condensed_cell_like &&
                    input_spec.producer == AuxiliaryInputProducer::AuxiliaryOutput,
                InvalidArgumentException,
                "FESystem::finalizeAuxiliaryLayout: local-condensed " +
                    std::string(scopeName(entry.spec.scope)) + " auxiliary instance '" +
                    entry.instance_name + "' cannot bind entity-local auxiliary "
                    "output input '" + registry_input +
                    "'. Cross-entity or auxiliary-to-auxiliary coupling does not "
                    "satisfy the independent local-condensation contract; use an "
                    "AuxiliaryOperator or a bordered/reduced formulation that "
                    "assembles the coupling explicitly.");

            FE_THROW_IF(
                entry.spec.layout_mode == AuxiliaryLayoutMode::Ragged &&
                    input_spec.producer == AuxiliaryInputProducer::AuxiliaryOutput,
                InvalidArgumentException,
                "FESystem::finalizeAuxiliaryLayout: ragged " +
                    std::string(scopeName(entry.spec.scope)) +
                    " auxiliary instance '" + entry.instance_name +
                    "' cannot bind entity-local auxiliary output input '" +
                    registry_input +
                    "'. Cross-entity or auxiliary-to-auxiliary coupling is not "
                    "part of the ragged AuxiliaryModel runtime contract; use an "
                    "AuxiliaryOperator or an explicitly assembled coupling.");

            FE_THROW_IF(
                entry.spec.layout_mode == AuxiliaryLayoutMode::Ragged &&
                    isFEBackedEntityLocalProducer(input_spec.producer) &&
                    !isRaggedScopeCompatibleFEBackedProducer(entry.spec.scope,
                                                             input_spec.producer),
                InvalidArgumentException,
                "FESystem::finalizeAuxiliaryLayout: ragged " +
                    std::string(scopeName(entry.spec.scope)) +
                    " auxiliary instance '" + entry.instance_name +
                    "' cannot bind FE-backed entity-local input '" +
                    registry_input + "' produced by " +
                    producerName(input_spec.producer) +
                    ". Ragged FE-backed inputs must use a scope-matched "
                    "quantity provider so stable entity ids and component "
                    "offsets remain well-defined.");
        }
    }
}

std::vector<int>
FESystem::buildAuxiliaryRegionRowOwnerRanks_(const DeployedAuxEntry& entry,
                                             std::size_t entity_count)
{
    FE_THROW_IF(entry.spec.scope != AuxiliaryStateScope::Region,
                InvalidArgumentException,
                "FESystem::buildAuxiliaryRegionRowOwnerRanks: entry is not "
                "Region-scoped");
    ensureAuxiliaryRegionLookupCache_();

    std::vector<std::size_t> entity_ids;
    if (!entry.entity_map.empty()) {
        entity_ids = entry.entity_map;
    } else {
        entity_ids.resize(entity_count);
        std::iota(entity_ids.begin(), entity_ids.end(), std::size_t{0});
    }

    std::vector<int> entity_owner_ranks;
    entity_owner_ranks.reserve(entity_ids.size());
    for (const auto region_id : entity_ids) {
        int owner = 0;
        if (auxiliary_region_lookup_cache_ &&
            !auxiliary_region_lookup_cache_->region_owner_ranks.empty()) {
            FE_THROW_IF(
                region_id >= auxiliary_region_lookup_cache_->region_owner_ranks.size(),
                InvalidArgumentException,
                "FESystem::finalizeAuxiliaryLayout: Region monolithic block '" +
                    entry.instance_name + "' references topology region " +
                    std::to_string(region_id) + " but owner metadata exists for only " +
                    std::to_string(
                        auxiliary_region_lookup_cache_->region_owner_ranks.size()) +
                    " regions");
            owner = auxiliary_region_lookup_cache_->region_owner_ranks[region_id];
        }
        FE_THROW_IF(owner < 0,
                    InvalidStateException,
                    "FESystem::finalizeAuxiliaryLayout: Region monolithic block '" +
                        entry.instance_name +
                        "' has no deterministic owner for topology region " +
                        std::to_string(region_id));
        entity_owner_ranks.push_back(owner);
    }

    return buildAuxiliaryRowOwnerRanks(
        AuxiliaryRowOwnershipSpec{
            .scope = AuxiliaryStateScope::Region,
            .policy = backends::MixedRowOwnershipPolicy::RegionOwner,
            .entity_count = entity_owner_ranks.size(),
            .stride = entry.spec.size,
            .single_owner_rank = -1,
            .entity_owner_ranks = std::span<const int>(entity_owner_ranks)});
}

int FESystem::nodeAuxiliaryOwnerRank_(std::size_t node_id) const
{
    auto ownerFromFieldDof = [&](GlobalIndex fe_dof) {
        int owner = -1;
        if (dof_permutation_ != nullptr &&
            !dof_permutation_->owner_rank.empty() &&
            fe_dof >= 0) {
            GlobalIndex backend_dof = fe_dof;
            if (!dof_permutation_->forward.empty() &&
                static_cast<std::size_t>(fe_dof) < dof_permutation_->forward.size()) {
                backend_dof = dof_permutation_->forward[static_cast<std::size_t>(fe_dof)];
            }
            if (backend_dof >= 0 &&
                static_cast<std::size_t>(backend_dof) <
                    dof_permutation_->owner_rank.size()) {
                owner =
                    dof_permutation_->owner_rank[static_cast<std::size_t>(backend_dof)];
            }
        }
        if (owner < 0) {
            owner = dof_handler_.getDofMap().getDofOwner(fe_dof);
        }
        return owner;
    };

    for (std::size_t field_idx = 0; field_idx < field_dof_handlers_.size(); ++field_idx) {
        if (field_idx >= field_dof_offsets_.size()) {
            break;
        }
        const auto* entity_map = field_dof_handlers_[field_idx].getEntityDofMap();
        if (entity_map == nullptr) {
            continue;
        }
        const auto vertex_dofs =
            entity_map->getVertexDofs(static_cast<GlobalIndex>(node_id));
        if (vertex_dofs.empty()) {
            continue;
        }
        return ownerFromFieldDof(
            static_cast<GlobalIndex>(vertex_dofs.front() + field_dof_offsets_[field_idx]));
    }

#if FE_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        int rank = 0;
        int comm_size = 1;
        MPI_Comm comm = is_setup_ ? dof_handler_.mpiComm() : MPI_COMM_WORLD;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &comm_size);
        if (comm_size <= 1) {
            return 0;
        }
        if (mesh_access_ != nullptr) {
            const auto owned_vertices =
                static_cast<std::size_t>(
                    std::max<GlobalIndex>(0, mesh_access_->numOwnedVertices()));
            if (node_id < owned_vertices) {
                return rank;
            }
        }
        FE_THROW(InvalidStateException,
                 "FESystem::nodeAuxiliaryOwnerRank: cannot derive backend row "
                 "owner for node " +
                     std::to_string(node_id) +
                     ". Node monolithic assembly in MPI requires a C0 nodal "
                     "field DOF owner or explicit backend permutation owner map.");
    }
#endif

    return 0;
}

std::vector<int>
FESystem::buildAuxiliaryNodeRowOwnerRanks_(const DeployedAuxEntry& entry,
                                           std::size_t entity_count) const
{
    FE_THROW_IF(entry.spec.scope != AuxiliaryStateScope::Node,
                InvalidArgumentException,
                "FESystem::buildAuxiliaryNodeRowOwnerRanks: entry is not "
                "Node-scoped");

    std::vector<std::size_t> entity_ids;
    if (!entry.entity_map.empty()) {
        entity_ids = entry.entity_map;
    } else {
        entity_ids.resize(entity_count);
        std::iota(entity_ids.begin(), entity_ids.end(), std::size_t{0});
    }

    std::vector<int> entity_owner_ranks;
    entity_owner_ranks.reserve(entity_ids.size());
    for (const auto node_id : entity_ids) {
        const int owner = nodeAuxiliaryOwnerRank_(node_id);
        FE_THROW_IF(owner < 0,
                    InvalidStateException,
                    "FESystem::finalizeAuxiliaryLayout: Node monolithic block '" +
                        entry.instance_name +
                        "' has no deterministic owner for node " +
                        std::to_string(node_id));
        entity_owner_ranks.push_back(owner);
    }

    return buildAuxiliaryRowOwnerRanks(
        AuxiliaryRowOwnershipSpec{
            .scope = AuxiliaryStateScope::Node,
            .policy = backends::MixedRowOwnershipPolicy::BackendDofOwner,
            .entity_count = entity_owner_ranks.size(),
            .stride = entry.spec.size,
            .single_owner_rank = -1,
            .entity_owner_ranks = std::span<const int>(entity_owner_ranks)});
}

void FESystem::validateMonolithicAuxiliaryLifecycle_(const DeployedAuxEntry& entry) const
{
    if (entry.spec.solve_mode != AuxiliarySolveMode::Monolithic) {
        return;
    }

    const auto structural = entry.model->structuralMetadata();
    const bool has_nonsmooth =
        entry.spec.has_nonsmooth || entry.model->hasNonsmoothHooks() ||
        structural.has_nonsmooth || entry.spec.event_mode == AuxiliaryEventMode::ActiveSetHook ||
        entry.spec.event_mode == AuxiliaryEventMode::ComplementarityHook;

    FE_THROW_IF(has_nonsmooth, NotImplementedException,
                "FESystem::finalizeAuxiliaryLayout: monolithic auxiliary instance '" +
                    entry.instance_name +
                    "' declares nonsmooth/complementarity hooks. Native monolithic "
                    "nonsmooth handling requires a semismooth or active-set policy "
                    "and is not part of the fixed-stride scope-completion path yet.");
}

void FESystem::validateRaggedMonolithicLocalCondensationEligibility_(
    const DeployedAuxEntry& entry) const
{
    if (entry.spec.solve_mode != AuxiliarySolveMode::Monolithic ||
        entry.spec.layout_mode != AuxiliaryLayoutMode::Ragged) {
        return;
    }

    auto scopeName = [](AuxiliaryStateScope scope) -> const char* {
        switch (scope) {
            case AuxiliaryStateScope::Global: return "Global";
            case AuxiliaryStateScope::Boundary: return "Boundary";
            case AuxiliaryStateScope::Node: return "Node";
            case AuxiliaryStateScope::Cell: return "Cell";
            case AuxiliaryStateScope::QuadraturePoint: return "QuadraturePoint";
            case AuxiliaryStateScope::Region: return "Region";
            case AuxiliaryStateScope::Facet: return "Facet";
        }
        return "Unknown";
    };

    const bool cell_like =
        entry.spec.scope == AuxiliaryStateScope::Cell ||
        entry.spec.scope == AuxiliaryStateScope::QuadraturePoint;
    const bool node_scope = entry.spec.scope == AuxiliaryStateScope::Node;
    FE_THROW_IF(!cell_like && !node_scope, NotImplementedException,
                "FESystem::finalizeAuxiliaryLayout: ragged monolithic "
                "auxiliary instance '" +
                    entry.instance_name + "' has " +
                    scopeName(entry.spec.scope) +
                    " scope. The current ragged monolithic contract is "
                    "limited to independent Cell/QuadraturePoint local "
                    "condensation or owner-backed Node local condensation. "
                    "Other scopes require explicit row-owner metadata, "
                    "compatible distributed sparsity, and zero dropped-entry "
                    "FSILS tests.");

    if (node_scope) {
        const bool has_backend_owner_map =
            use_backend_row_ownership_for_assembly_ &&
            dof_permutation_ != nullptr &&
            !dof_permutation_->empty() &&
            !dof_permutation_->owner_rank.empty();
        FE_THROW_IF(!has_backend_owner_map, InvalidStateException,
                    "FESystem::finalizeAuxiliaryLayout: ragged Node "
                    "monolithic auxiliary instance '" +
                        entry.instance_name +
                        "' requires backend row ownership metadata before it "
                        "can use owner-backed local condensation. Call setup() "
                        "with use_backend_row_ownership_for_assembly=true and "
                        "a backend permutation owner map, or keep the block "
                        "partitioned.");
    }

    FE_THROW_IF(!entry.local_condensed, InvalidStateException,
                "FESystem::finalizeAuxiliaryLayout: ragged " +
                    std::string(scopeName(entry.spec.scope)) +
                    " monolithic auxiliary instance '" + entry.instance_name +
                    "' is not marked for local condensation. Ragged Cell/QP "
                    "and Node monolithic blocks must use a proven "
                    "local-condensation contract and must not enter bordered "
                    "or native auxiliary-row assembly.");

    FE_THROW_IF(!auxiliary_state_manager_ ||
                    !auxiliary_state_manager_->hasBlock(entry.instance_name),
                InvalidStateException,
                "FESystem::finalizeAuxiliaryLayout: ragged monolithic "
                "auxiliary instance '" +
                    entry.instance_name +
                    "' was not materialized before local-condensation "
                    "eligibility validation");

    const auto& block = auxiliary_state_manager_->getBlock(entry.instance_name);
    FE_THROW_IF(block.layoutMode() != AuxiliaryLayoutMode::Ragged,
                InvalidStateException,
                "FESystem::finalizeAuxiliaryLayout: ragged monolithic "
                "auxiliary instance '" +
                    entry.instance_name +
                    "' was registered with a non-ragged storage layout");

    const auto offsets = block.entityOffsets();
    FE_THROW_IF(offsets.size() != block.entityCount() + 1u,
                InvalidStateException,
                "FESystem::finalizeAuxiliaryLayout: ragged monolithic "
                "auxiliary instance '" +
                    entry.instance_name +
                    "' has inconsistent component offsets for local "
                    "condensation");

    const auto model_dim = static_cast<std::size_t>(entry.spec.size);
    std::vector<std::size_t> entity_ids;
    if (node_scope) {
        if (!entry.entity_map.empty()) {
            entity_ids = entry.entity_map;
        } else {
            entity_ids.resize(block.entityCount());
            std::iota(entity_ids.begin(), entity_ids.end(), std::size_t{0});
        }
        FE_THROW_IF(entity_ids.size() != block.entityCount(),
                    InvalidStateException,
                    "FESystem::finalizeAuxiliaryLayout: ragged Node "
                    "monolithic auxiliary instance '" +
                        entry.instance_name +
                        "' has inconsistent node entity metadata for "
                        "owner-backed local condensation");
    }

    auto strictBackendOwnerForNode = [&](std::size_t node_id) {
        FE_THROW_IF(dof_permutation_ == nullptr ||
                        dof_permutation_->forward.empty() ||
                        dof_permutation_->owner_rank.empty(),
                    InvalidStateException,
                    "FESystem::finalizeAuxiliaryLayout: ragged Node "
                    "monolithic auxiliary instance '" +
                        entry.instance_name +
                        "' has no backend DOF owner map");

        for (std::size_t field_idx = 0;
             field_idx < field_dof_handlers_.size() &&
             field_idx < field_dof_offsets_.size();
             ++field_idx) {
            const auto* entity_map =
                field_dof_handlers_[field_idx].getEntityDofMap();
            if (entity_map == nullptr) {
                continue;
            }
            const auto vertex_dofs =
                entity_map->getVertexDofs(static_cast<GlobalIndex>(node_id));
            if (vertex_dofs.empty()) {
                continue;
            }
            const auto fe_dof = static_cast<GlobalIndex>(
                vertex_dofs.front() + field_dof_offsets_[field_idx]);
            FE_THROW_IF(fe_dof < 0 ||
                            static_cast<std::size_t>(fe_dof) >=
                                dof_permutation_->forward.size(),
                        InvalidStateException,
                        "FESystem::finalizeAuxiliaryLayout: ragged Node "
                        "monolithic auxiliary instance '" +
                            entry.instance_name +
                            "' cannot map node " + std::to_string(node_id) +
                            " to a backend-owned FE DOF");
            const auto backend_dof =
                dof_permutation_->forward[static_cast<std::size_t>(fe_dof)];
            FE_THROW_IF(backend_dof < 0 ||
                            static_cast<std::size_t>(backend_dof) >=
                                dof_permutation_->owner_rank.size(),
                        InvalidStateException,
                        "FESystem::finalizeAuxiliaryLayout: ragged Node "
                        "monolithic auxiliary instance '" +
                            entry.instance_name +
                            "' maps node " + std::to_string(node_id) +
                            " to a backend DOF without owner metadata");
            const int owner =
                dof_permutation_->owner_rank[static_cast<std::size_t>(backend_dof)];
            FE_THROW_IF(owner < 0, InvalidStateException,
                        "FESystem::finalizeAuxiliaryLayout: ragged Node "
                        "monolithic auxiliary instance '" +
                            entry.instance_name +
                            "' has invalid backend owner metadata for node " +
                            std::to_string(node_id));
            return owner;
        }

        FE_THROW(InvalidStateException,
                 "FESystem::finalizeAuxiliaryLayout: ragged Node monolithic "
                 "auxiliary instance '" +
                     entry.instance_name +
                     "' requires at least one C0 nodal FE field whose vertex "
                     "DOFs define backend row ownership for node " +
                     std::to_string(node_id));
        return -1;
    };

    std::vector<int> node_row_owner_ranks;
    if (node_scope) {
        node_row_owner_ranks.reserve(block.storageSize());
    }
    for (std::size_t entity = 0; entity < block.entityCount(); ++entity) {
        const auto width = offsets[entity + 1u] - offsets[entity];
        FE_THROW_IF(width != model_dim, InvalidArgumentException,
                    "FESystem::finalizeAuxiliaryLayout: ragged "
                        + std::string(scopeName(entry.spec.scope)) +
                        " monolithic auxiliary instance '" +
                        entry.instance_name +
                        "' does not satisfy the local-condensation contract: "
                        "entity " + std::to_string(entity) +
                        " has ragged width " + std::to_string(width) +
                        " but the fixed AuxiliaryStateModel dimension is " +
                        std::to_string(model_dim) +
                        ". Variable-width monolithic local condensation is "
                        "not implemented; use uniform ragged widths matching "
                        "the model dimension or provide an AuxiliaryOperator "
                        "with an explicit assembly contract.");
        if (node_scope) {
            const int owner =
                strictBackendOwnerForNode(entity_ids[entity]);
            for (std::size_t component = 0; component < width; ++component) {
                node_row_owner_ranks.push_back(owner);
            }
        }
    }
    if (node_scope) {
        FE_THROW_IF(node_row_owner_ranks.size() != block.storageSize(),
                    InvalidStateException,
                    "FESystem::finalizeAuxiliaryLayout: ragged Node "
                    "monolithic auxiliary instance '" +
                        entry.instance_name +
                        "' does not have one backend owner per ragged "
                        "component row");
    }
}

void FESystem::validateAuxiliaryMixedLayoutContract_() const
{
    if (!auxiliary_operator_registry_ ||
        !auxiliary_operator_registry_->isLayoutFinalized()) {
        return;
    }

    const auto& layout = auxiliary_operator_registry_->auxiliaryLayout();
    for (const auto& block : layout.blocks) {
        if (block.n_unknowns == 0u) {
            continue;
        }

        const bool global_like =
            block.scope == AuxiliaryStateScope::Global ||
            block.scope == AuxiliaryStateScope::Boundary;
        if (!global_like) {
            continue;
        }

        FE_THROW_IF(block.row_ownership != backends::MixedRowOwnershipPolicy::SingleOwner,
                    InvalidStateException,
                    "FESystem::finalizeAuxiliaryLayout: global monolithic auxiliary block '" +
                        block.name +
                        "' must use single-owner row metadata for MPI/FSILS");
        FE_THROW_IF(block.single_owner_rank < 0,
                    InvalidStateException,
                    "FESystem::finalizeAuxiliaryLayout: global monolithic auxiliary block '" +
                        block.name + "' has no valid single-owner rank");
        FE_THROW_IF(!block.row_owner_ranks.empty() &&
                        block.row_owner_ranks.size() != block.n_unknowns,
                    InvalidStateException,
                    "FESystem::finalizeAuxiliaryLayout: global monolithic auxiliary block '" +
                        block.name + "' has an incomplete row-owner map");
        FE_THROW_IF(!block.row_owner_ranks.empty() &&
                        std::any_of(block.row_owner_ranks.begin(),
                                    block.row_owner_ranks.end(),
                                    [&](int owner) {
                                        return owner != block.single_owner_rank;
                                    }),
                    InvalidStateException,
                    "FESystem::finalizeAuxiliaryLayout: global monolithic auxiliary block '" +
                        block.name + "' has row owners inconsistent with its single owner");
        FE_THROW_IF(block.assembly_mode == backends::MixedBlockAssemblyMode::NativeOwnedRows,
                    InvalidStateException,
                    "FESystem::finalizeAuxiliaryLayout: global monolithic auxiliary block '" +
                        block.name +
                        "' cannot be assembled as native FSILS rows. Use direct-only "
                        "lowering or bordered/reduced coupling unless a complete "
                        "nodal-interleaved FSILS layout is explicitly implemented.");
    }

    for (const auto& block : layout.blocks) {
        if (block.n_unknowns == 0u ||
            block.scope != AuxiliaryStateScope::Region) {
            continue;
        }

        FE_THROW_IF(block.row_ownership != backends::MixedRowOwnershipPolicy::RegionOwner,
                    InvalidStateException,
                    "FESystem::finalizeAuxiliaryLayout: Region monolithic auxiliary "
                    "block '" +
                        block.name +
                        "' must use explicit region-owner row metadata");
        FE_THROW_IF(block.row_owner_ranks.size() != block.n_unknowns,
                    InvalidStateException,
                    "FESystem::finalizeAuxiliaryLayout: Region monolithic auxiliary "
                    "block '" +
                        block.name + "' has an incomplete row-owner map");
        FE_THROW_IF(std::any_of(block.row_owner_ranks.begin(),
                                block.row_owner_ranks.end(),
                                [](int owner) { return owner < 0; }),
                    InvalidStateException,
                    "FESystem::finalizeAuxiliaryLayout: Region monolithic auxiliary "
                    "block '" +
                        block.name + "' has invalid row-owner ranks");
        FE_THROW_IF(block.assembly_mode == backends::MixedBlockAssemblyMode::NativeOwnedRows,
                    InvalidStateException,
                    "FESystem::finalizeAuxiliaryLayout: Region monolithic auxiliary "
                    "block '" +
                        block.name +
                        "' cannot be assembled as native FSILS rows until a "
                        "Region sparse/distributed row strategy represents the "
                        "layout in backend sparsity and solver metadata");
    }

    for (const auto& block : layout.blocks) {
        if (block.n_unknowns == 0u ||
            block.scope != AuxiliaryStateScope::Node) {
            continue;
        }

        FE_THROW_IF(block.row_ownership != backends::MixedRowOwnershipPolicy::BackendDofOwner,
                    InvalidStateException,
                    "FESystem::finalizeAuxiliaryLayout: Node monolithic auxiliary "
                    "block '" +
                        block.name +
                        "' must use backend-DOF-owner row metadata");
        FE_THROW_IF(block.row_owner_ranks.size() != block.n_unknowns,
                    InvalidStateException,
                    "FESystem::finalizeAuxiliaryLayout: Node monolithic auxiliary "
                    "block '" +
                        block.name + "' has an incomplete backend row-owner map");
        FE_THROW_IF(std::any_of(block.row_owner_ranks.begin(),
                                block.row_owner_ranks.end(),
                                [](int owner) { return owner < 0; }),
                    InvalidStateException,
                    "FESystem::finalizeAuxiliaryLayout: Node monolithic auxiliary "
                    "block '" +
                        block.name + "' has invalid backend row-owner ranks");
        FE_THROW_IF(block.assembly_mode == backends::MixedBlockAssemblyMode::NativeOwnedRows,
                    InvalidStateException,
                    "FESystem::finalizeAuxiliaryLayout: Node monolithic auxiliary "
                    "block '" +
                        block.name +
                        "' cannot be assembled as native FSILS rows until the "
                        "backend advertises a complete nodal component block "
                        "covering field and auxiliary components. Use the "
                        "owner-backed sparse condensed path or bordered/reduced "
                        "coupling.");
    }
}

void FESystem::rollbackAuxiliaryState()
{
    if (auxiliary_state_manager_) {
        auxiliary_state_manager_->rollbackAll();
    }
}

void FESystem::finalizeAuxiliaryLayout()
{
    monolithic_aux_committed_rates_.clear();
    monolithic_aux_committed_rates_valid_.clear();
    lowered_aux_output_exprs_by_name_.clear();
    lowered_aux_output_exprs_by_id_.clear();

    // Resolve deferred input-expression wiring before deciding whether pure
    // algebraic monolithic blocks can be lowered to direct-only coupling.
    // That lowering synthesizes outputs through bound inputs, so it must see
    // the finalized input-registry view rather than unresolved symbols.
    finalizeDeferredInputDeps();

    // Materialize deployed instances into blocks, steppers, and derivative providers.
    for (auto& entry : deployed_aux_entries_) {
        entry.selected = isAuxiliaryDeploymentSelected_(entry);
        entry.materialized = false;
        entry.entity_map.clear();

        if (entry.activation_mode == AuxiliaryActivationMode::Disabled ||
            !entry.selected) {
            FE_THROW_IF(hasAuxiliaryConsumers_(entry), InvalidStateException,
                        "FESystem::finalizeAuxiliaryLayout: auxiliary instance '" +
                            entry.instance_name +
                            "' is not active for this run but is still referenced by an installed consumer");
            continue;
        }

        auto& mgr = auxiliaryStateManager();

        const auto scope_resolution = resolveAuxiliaryDeploymentScope_(entry);
        if (!entry.materialized) {
            continue;
        }
        const auto entity_count = scope_resolution.entity_count;
        const auto owned_entity_count = scope_resolution.owned_entity_count;

        // Register the block.
        // Build initial values: if provided values match dim (not total),
        // replicate per entity to fill the full storage.
        // For ByComponentThenEntity ordering, transpose to component-major.
        std::vector<Real> full_init;
        if (!entry.initial_values.empty()) {
            const auto dim_sz = static_cast<std::size_t>(entry.spec.size);
            if (entry.initial_values.size() == dim_sz && entity_count > 1) {
                full_init.resize(entity_count * dim_sz);
                if (entry.spec.ordering == AuxiliaryEntityOrdering::ByComponentThenEntity) {
                    // Component-major: [comp0_e0, comp0_e1, ..., comp1_e0, comp1_e1, ...]
                    for (std::size_t c = 0; c < dim_sz; ++c) {
                        for (std::size_t e = 0; e < entity_count; ++e) {
                            full_init[c * entity_count + e] = entry.initial_values[c];
                        }
                    }
                } else {
                    // Entity-major (default): [e0_c0, e0_c1, ..., e1_c0, e1_c1, ...]
                    for (std::size_t e = 0; e < entity_count; ++e) {
                        std::copy(entry.initial_values.begin(),
                                  entry.initial_values.end(),
                                  full_init.begin() + static_cast<std::ptrdiff_t>(e * dim_sz));
                    }
                }
            } else {
                full_init = entry.initial_values;
            }
        }
        if (entry.spec.layout_mode == AuxiliaryLayoutMode::Ragged) {
            if (entry.spec.solve_mode == AuxiliarySolveMode::Monolithic) {
                const bool supported_ragged_monolithic =
                    entry.spec.scope == AuxiliaryStateScope::Node ||
                    entry.spec.scope == AuxiliaryStateScope::Cell ||
                    entry.spec.scope == AuxiliaryStateScope::QuadraturePoint;
                FE_THROW_IF(!supported_ragged_monolithic, NotImplementedException,
                            "FESystem::finalizeAuxiliaryLayout: ragged "
                            "monolithic deployment for '" +
                                entry.instance_name +
                                "' is enabled only for independent Cell/"
                                "QuadraturePoint local condensation or "
                                "owner-backed Node local condensation. Other "
                                "scopes still require explicit row-owner "
                                "metadata, compatible distributed sparsity, "
                                "and zero dropped-entry FSILS tests.");
            }

            auto component_offsets =
                buildAuxiliaryRaggedComponentOffsets_(entry, scope_resolution);
            auto ragged_init =
                buildAuxiliaryRaggedInitialValues_(entry, component_offsets);
            entry.ragged_component_offsets = component_offsets;

            const auto init_span = ragged_init.empty()
                ? std::span<const Real>{}
                : std::span<const Real>(ragged_init);
            if (entry.spec.scope == AuxiliaryStateScope::QuadraturePoint) {
                mgr.registerBlockRaggedWithQPOffsets(
                    entry.spec, entry.qp_offsets, component_offsets, init_span);
            } else {
                mgr.registerBlockRagged(
                    entry.spec, component_offsets, owned_entity_count, init_span);
            }
            mgr.setEntityRemapMetadata(
                entry.instance_name,
                buildAuxiliaryEntityRemapMetadata_(entry, scope_resolution));
        } else {
            const auto init_span = full_init.empty()
                ? std::span<const Real>{}
                : std::span<const Real>(full_init);
            if (entry.spec.scope == AuxiliaryStateScope::QuadraturePoint &&
                !entry.qp_offsets.empty()) {
                mgr.registerBlockWithQPOffsets(entry.spec, entry.qp_offsets, init_span);
            } else {
                mgr.registerBlock(entry.spec, entity_count, owned_entity_count, init_span);
            }
            mgr.setEntityRemapMetadata(
                entry.instance_name,
                buildAuxiliaryEntityRemapMetadata_(entry, scope_resolution));
        }

        // Create stepper and derivative provider for partitioned blocks.
        if (entry.spec.solve_mode == AuxiliarySolveMode::Partitioned) {
            entry.stepper = createStepper(entry.stepper_spec.method_name);
            entry.stepper->setup(entry.spec.size, entry.stepper_spec);

            entry.deriv_provider = std::make_unique<AuxiliaryDerivativeProvider>();
            entry.deriv_provider->setup(*entry.model, entry.spec.derivative_policy);
            if (entry.model->hasEventFunctions()) {
                const auto& blk = mgr.getBlock(entry.instance_name);
                entry.event_managers.clear();
                entry.event_managers.reserve(blk.entityCount());
                for (std::size_t e = 0; e < blk.entityCount(); ++e) {
                    auto manager = std::make_unique<AuxiliaryEventManager>();
                    manager->setNonsmoothPolicy(entry.spec.nonsmooth_policy);
                    entry.event_managers.push_back(std::move(manager));
                }
            }
        }

        if (entry.spec.solve_mode == AuxiliarySolveMode::Monolithic) {
            validateMonolithicAuxiliaryLifecycle_(entry);
            entry.deriv_provider = std::make_unique<AuxiliaryDerivativeProvider>();
            entry.deriv_provider->setup(*entry.model, entry.spec.derivative_policy);
            if (entry.spec.event_mode == AuxiliaryEventMode::EventHook ||
                entry.model->hasEventFunctions()) {
                const auto& blk = mgr.getBlock(entry.instance_name);
                entry.event_managers.clear();
                entry.event_managers.reserve(blk.entityCount());
                for (std::size_t e = 0; e < blk.entityCount(); ++e) {
                    auto manager = std::make_unique<AuxiliaryEventManager>();
                    manager->setNonsmoothPolicy(entry.spec.nonsmooth_policy);
                    entry.event_managers.push_back(std::move(manager));
                }
            }
            // Defer lower_to_direct_only until after all deferred FE-coupled
            // inputs and lowered output expressions are available. Purely
            // algebraic monolithic outlet models can then drop out of the
            // live bordered layout entirely instead of being reduced later
            // inside Newton.
            entry.lower_to_direct_only = false;
            entry.local_condensed = false;
        }

        // Validate direct FE field references in auxiliary residual expressions.
        if (entry.deriv_provider) {
            const auto& art = entry.deriv_provider->artifact();
            if (!art.referenced_fields.empty()) {
                // Reject non-Node scopes.  Direct DiscreteField/StateField nodes
                // are only meaningful for Node-scoped models, where the Kronecker
                // delta property of Lagrange elements gives exact field values.
                if (entry.spec.scope != AuxiliaryStateScope::Node) {
                    const char* scope_name = "unknown";
                    switch (entry.spec.scope) {
                        case AuxiliaryStateScope::Global: scope_name = "Global"; break;
                        case AuxiliaryStateScope::Boundary: scope_name = "Boundary"; break;
                        case AuxiliaryStateScope::Cell: scope_name = "Cell"; break;
                        case AuxiliaryStateScope::QuadraturePoint: scope_name = "QuadraturePoint"; break;
                        case AuxiliaryStateScope::Region: scope_name = "Region"; break;
                        case AuxiliaryStateScope::Facet: scope_name = "Facet"; break;
                        case AuxiliaryStateScope::Node: break; // unreachable
                    }
                    FE_THROW(InvalidArgumentException,
                        "FESystem::finalizeAuxiliaryLayout: " + std::string(scope_name)
                        + "-scoped auxiliary model '" + entry.instance_name
                        + "' directly references FE field(s) via DiscreteField/StateField "
                        "nodes.  Direct field references are only supported for Node-scoped "
                        "models (Lagrange Kronecker delta).  Use sampledField(), "
                        "boundaryIntegral(), domainAverage(), or feExpression() to mediate "
                        "field access, then bind via bind().");
                }

                // Validate that referenced fields have vertex DOFs with Lagrange
                // Kronecker delta semantics (H1/C0 spaces).  Scalar, vector,
                // and tensor fields are all supported; non-vertex spaces and
                // fields exceeding MAX_FIELD_VALUE_COMPONENTS are not.
                for (const auto fid : art.referenced_fields) {
                    if (!field_registry_.has(fid)) continue;
                    const auto& rec = field_registry_.get(fid);
                    if (rec.components > MAX_FIELD_VALUE_COMPONENTS) {
                        FE_THROW(InvalidArgumentException,
                            "FESystem::finalizeAuxiliaryLayout: auxiliary model '"
                            + entry.instance_name + "' references "
                            + std::to_string(rec.components) + "-component field '"
                            + rec.name + "' which exceeds MAX_FIELD_VALUE_COMPONENTS ("
                            + std::to_string(MAX_FIELD_VALUE_COMPONENTS) + ").");
                    }
                    // Require C0-continuous (nodal Lagrange) space for direct
                    // field references.  The Kronecker delta property (DOF
                    // coefficients equal pointwise vertex values) is only valid
                    // for C0 nodal Lagrange interpolation.  This includes both
                    // scalar H1 spaces and Product spaces built from H1 components
                    // (e.g., VectorSpace(H1, ...)).  L2, H(curl), H(div), C1,
                    // and other continuity types do not have this property.
                    if (rec.space) {
                        const auto ct = rec.space->continuity();
                        if (ct != Continuity::C0) {
                            FE_THROW(InvalidArgumentException,
                                "FESystem::finalizeAuxiliaryLayout: auxiliary model '"
                                + entry.instance_name + "' directly references field '"
                                + rec.name + "' which has non-C0 continuity.  Direct "
                                "DiscreteField/StateField references require C0 (nodal "
                                "Lagrange) spaces for the Kronecker delta property.  "
                                "Use sampledField() or feExpression() for L2 (DG), "
                                "H(div), H(curl), C1, or other space types.");
                        }
                    }
                    // Verify that the field's DOF handler has vertex DOFs.
                    // This is a defensive check: all C0 spaces in the current
                    // library are Lagrange and have vertex DOFs, but if a future
                    // C0 space (e.g., Bernstein, hierarchical) is added without
                    // nodal Kronecker semantics, this catches it at setup.
                    {
                        const auto fidx2 = static_cast<std::size_t>(fid);
                        if (fidx2 < field_dof_handlers_.size()) {
                            const auto* femap = field_dof_handlers_[fidx2].getEntityDofMap();
                            if (!femap) {
                                FE_THROW(InvalidArgumentException,
                                    "FESystem::finalizeAuxiliaryLayout: auxiliary model '"
                                    + entry.instance_name + "' directly references field '"
                                    + rec.name + "' which has no EntityDofMap.  Direct "
                                    "field references require vertex-based DOF mapping.");
                            }
                            auto vdofs = femap->getVertexDofs(static_cast<GlobalIndex>(0));
                            if (vdofs.empty()) {
                                FE_THROW(InvalidArgumentException,
                                    "FESystem::finalizeAuxiliaryLayout: auxiliary model '"
                                    + entry.instance_name + "' directly references field '"
                                    + rec.name + "' which has no vertex DOFs.  Direct "
                                    "field references require nodal Lagrange spaces with "
                                    "vertex-associated DOFs (Kronecker delta property).  "
                                    "Use sampledField() or feExpression() instead.");
                            }
                        }
                    }
                }
            }
        }
    }

    // Wire FE-coupled auxiliary input providers (SampledStateField, etc.)
    wireFECoupledInputProviders();

    // Build multirate scheduler from deployed block schedule modes.
    aux_scheduler_ = std::make_unique<AuxiliaryMultirateScheduler>();
    for (const auto& entry : deployed_aux_entries_) {
        if (!entry.materialized) continue;
        if (entry.spec.solve_mode != AuxiliarySolveMode::Partitioned) continue;

        MultirateBlockSchedule sched;
        sched.block_name = entry.instance_name;

        switch (entry.spec.schedule_mode) {
            case AuxiliaryScheduleMode::SingleRate:
                sched.rate_ratio = 1;
                break;
            case AuxiliaryScheduleMode::Subcycled:
                sched.rate_ratio = entry.stepper_spec.substep_count;
                break;
            case AuxiliaryScheduleMode::Multirate:
                sched.rate_ratio = entry.stepper_spec.substep_count;
                break;
        }

        aux_scheduler_->addBlockSchedule(std::move(sched));
    }

    finalizeDeferredInputDeps();
    buildLoweredAuxiliaryOutputExpressions_();
    buildAuxiliaryOutputBindings_();
    lowerAuxiliaryConstraintBindings_();

    for (auto& entry : deployed_aux_entries_) {
        if (!entry.materialized) {
            continue;
        }
        if (entry.spec.solve_mode != AuxiliarySolveMode::Monolithic) {
            continue;
        }
        const bool ragged_local_condensed_monolithic =
            entry.spec.layout_mode == AuxiliaryLayoutMode::Ragged &&
            (entry.spec.scope == AuxiliaryStateScope::Node ||
             entry.spec.scope == AuxiliaryStateScope::Cell ||
             entry.spec.scope == AuxiliaryStateScope::QuadraturePoint);
        entry.lower_to_direct_only = ragged_local_condensed_monolithic
            ? false
            : canLowerAlgebraicAuxiliaryToDirectOnly_(entry);
        const bool node_owner_backed_sparse =
            entry.spec.scope == AuxiliaryStateScope::Node &&
            use_backend_row_ownership_for_assembly_ &&
            dof_permutation_ != nullptr &&
            !dof_permutation_->owner_rank.empty();
        entry.local_condensed =
            !entry.lower_to_direct_only &&
            (node_owner_backed_sparse ||
             entry.spec.scope == AuxiliaryStateScope::Cell ||
             entry.spec.scope == AuxiliaryStateScope::QuadraturePoint ||
             entry.spec.scope == AuxiliaryStateScope::Facet);
        validateRaggedMonolithicLocalCondensationEligibility_(entry);
        if (monolithicAuxTraceEnabled()) {
            auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(entry.model.get());
            std::ostringstream oss;
            oss << "FESystem::finalizeAuxiliaryLayout lowerability"
                << " instance=" << entry.instance_name
                << " solve_mode=monolithic";
            if (built != nullptr) {
                const auto& names = built->stateNames();
                const auto& kinds = entry.model->structuralMetadata().variable_kinds;
                oss << " dim=" << names.size()
                    << " pure_algebraic="
                    << (isPureAlgebraicAuxiliary(*entry.model, names.size()) ? 1 : 0)
                    << " state_kinds=[";
                for (std::size_t i = 0; i < names.size(); ++i) {
                    if (i != 0) {
                        oss << ", ";
                    }
                    const auto kind = i < kinds.size()
                        ? kinds[i]
                        : AuxiliaryVariableKind::Differential;
                    oss << names[i] << ":"
                        << (kind == AuxiliaryVariableKind::Algebraic ? "alg" : "dyn");
                }
                oss << "]";
            } else {
                oss << " built=0";
            }
            if (entry.deriv_provider) {
                const auto& artifact = entry.deriv_provider->artifact();
                oss << " referenced_fields=" << artifact.referenced_fields.size();
            }
            const auto output_names = entry.model->outputNames();
            oss << " outputs=[";
            for (std::size_t i = 0; i < output_names.size(); ++i) {
                if (i != 0) {
                    oss << ", ";
                }
                const auto qualified_name = entry.instance_name + "/" + output_names[i];
                const auto lowered = loweredAuxiliaryOutputExpr(qualified_name);
                oss << output_names[i] << ":"
                    << (lowered.has_value() ? "lowerable" : "blocked");
            }
            oss << "]"
                << " input_bindings=" << entry.input_bindings.size()
                << " coupled_bindings=" << entry.coupled_bindings.size()
                << " lower_to_direct_only=" << (entry.lower_to_direct_only ? 1 : 0)
                << " local_condensed=" << (entry.local_condensed ? 1 : 0);
            FE_LOG_INFO(oss.str());
        }
        if (!entry.lower_to_direct_only && !entry.local_condensed) {
            std::size_t entity_count = entry.explicit_entity_count;
            if (auxiliary_state_manager_ &&
                auxiliary_state_manager_->hasBlock(entry.instance_name)) {
                entity_count = auxiliary_state_manager_->getBlock(entry.instance_name).entityCount();
            } else if (entity_count == 0 && !entry.entity_map.empty()) {
                entity_count = entry.entity_map.size();
            } else if (entity_count == 0) {
                entity_count = 1;
            }
            auto solver_metadata = entry.solver_metadata;
            const auto structural = entry.model->structuralMetadata();
            if (!solver_metadata.has_value() && !structural.constraint_groups.empty()) {
                AuxiliaryBlockSolverMetadata inferred;
                inferred.block_name = entry.instance_name;
                inferred.role = AuxiliaryBlockRole::Constraint;
                inferred.block_diagonal_suitable = false;
                solver_metadata = std::move(inferred);
            }
            if (solver_metadata.has_value()) {
                if (solver_metadata->block_name.empty()) {
                    solver_metadata->block_name = entry.instance_name;
                }
                auxiliaryOperatorRegistry().setBlockSolverMetadata(
                    entry.instance_name, *solver_metadata);
            }
            auxiliaryOperatorRegistry().registerMonolithicUnknowns(
                entry.instance_name, entity_count,
                entry.spec.size, entry.spec.scope,
                solver_metadata ? &*solver_metadata : nullptr,
                structural.constraint_groups);
            if (entry.spec.scope == AuxiliaryStateScope::Region) {
                auxiliaryOperatorRegistry().setBlockRowOwnerRanks(
                    entry.instance_name,
                    buildAuxiliaryRegionRowOwnerRanks_(entry, entity_count));
            } else if (entry.spec.scope == AuxiliaryStateScope::Node) {
                auxiliaryOperatorRegistry().setBlockRowOwnerRanks(
                    entry.instance_name,
                    buildAuxiliaryNodeRowOwnerRanks_(entry, entity_count));
            }
        }
    }

    validateEntityLocalAuxiliaryBindings_();

    if (!auxiliary_operator_registry_) {
        const bool has_monolithic_aux =
            std::any_of(deployed_aux_entries_.begin(),
                        deployed_aux_entries_.end(),
                        [](const auto& entry) {
                            return entry.spec.solve_mode == AuxiliarySolveMode::Monolithic;
                        });
        if (has_monolithic_aux) {
            (void)auxiliaryOperatorRegistry();
        }
    }

    if (auxiliary_operator_registry_ &&
        !auxiliary_operator_registry_->isLayoutFinalized()) {
        auxiliary_operator_registry_->finalizeLayout();
    }
    validateAuxiliaryMixedLayoutContract_();
}

void FESystem::assembleMixedAuxiliaryDense(
    const SystemStateView& state,
    std::size_t n_field_dofs,
    std::vector<Real>& residual_out,
    std::vector<Real>& matrix_out)
{
    // Compute total mixed size from the operator registry layout,
    // which accounts for entity counts (n_unknowns = entity_count * stride).
    std::size_t n_aux = 0;
    if (auxiliary_operator_registry_ && auxiliary_operator_registry_->isLayoutFinalized()) {
        n_aux = auxiliary_operator_registry_->auxiliaryLayout().total_aux_unknowns;
    } else {
        for (const auto& entry : deployed_aux_entries_) {
            if (!entry.materialized) {
                continue;
            }
            if (entry.spec.solve_mode == AuxiliarySolveMode::Monolithic) {
                if (entry.lower_to_direct_only) continue;
                n_aux += static_cast<std::size_t>(entry.model->dimension());
            }
        }
    }
    const auto n_total = n_field_dofs + n_aux;
    residual_out.assign(n_total, 0.0);
    matrix_out.assign(n_total * n_total, 0.0);

    // Dense GlobalSystemView that stores matrix and vector entries.
    struct DenseAccum final : public assembly::GlobalSystemView {
        std::vector<Real>& vec;
        std::vector<Real>& mat;
        GlobalIndex n;
        DenseAccum(std::vector<Real>& v, std::vector<Real>& m, GlobalIndex sz)
            : vec(v), mat(m), n(sz) {}

        void addMatrixEntries(std::span<const GlobalIndex> rows,
                              std::span<const Real> vals,
                              assembly::AddMode) override {
            // Square single-DOF-set: rows = cols.
            const auto nd = static_cast<int>(rows.size());
            for (int i = 0; i < nd; ++i)
                for (int j = 0; j < nd; ++j) {
                    auto r = rows[static_cast<std::size_t>(i)];
                    auto c = rows[static_cast<std::size_t>(j)];
                    if (r >= 0 && r < n && c >= 0 && c < n)
                        mat[static_cast<std::size_t>(r * n + c)] +=
                            vals[static_cast<std::size_t>(i * nd + j)];
                }
        }
        void addMatrixEntries(std::span<const GlobalIndex> rows,
                              std::span<const GlobalIndex> cols,
                              std::span<const Real> vals,
                              assembly::AddMode) override {
            const auto nr = static_cast<int>(rows.size());
            const auto nc = static_cast<int>(cols.size());
            for (int i = 0; i < nr; ++i)
                for (int j = 0; j < nc; ++j) {
                    auto r = rows[static_cast<std::size_t>(i)];
                    auto c = cols[static_cast<std::size_t>(j)];
                    if (r >= 0 && r < n && c >= 0 && c < n)
                        mat[static_cast<std::size_t>(r * n + c)] +=
                            vals[static_cast<std::size_t>(i * nc + j)];
                }
        }
        void addMatrixEntry(GlobalIndex r, GlobalIndex c, Real v,
                            assembly::AddMode) override {
            if (r >= 0 && r < n && c >= 0 && c < n)
                mat[static_cast<std::size_t>(r * n + c)] += v;
        }
        void setDiagonal(std::span<const GlobalIndex>, std::span<const Real>) override {}
        void setDiagonal(GlobalIndex, Real) override {}
        void zeroRows(std::span<const GlobalIndex>, bool) override {}
        void addVectorEntries(std::span<const GlobalIndex> dofs,
                              std::span<const Real> vals,
                              assembly::AddMode) override {
            for (std::size_t i = 0; i < dofs.size(); ++i) {
                auto d = dofs[i];
                if (d >= 0 && d < n) vec[static_cast<std::size_t>(d)] += vals[i];
            }
        }
        void addVectorEntry(GlobalIndex d, Real v, assembly::AddMode) override {
            if (d >= 0 && d < n) vec[static_cast<std::size_t>(d)] += v;
        }
        void setVectorEntries(std::span<const GlobalIndex>, std::span<const Real>) override {}
        void zeroVectorEntries(std::span<const GlobalIndex>) override {}
        [[nodiscard]] Real getVectorEntry(GlobalIndex d) const override {
            return (d >= 0 && d < n) ? vec[static_cast<std::size_t>(d)] : 0.0;
        }
        void beginAssemblyPhase() override {}
        void endAssemblyPhase() override {}
        void finalizeAssembly() override {}
        [[nodiscard]] assembly::AssemblyPhase getPhase() const noexcept override {
            return assembly::AssemblyPhase::Building;
        }
        [[nodiscard]] bool hasMatrix() const noexcept override { return true; }
        [[nodiscard]] bool hasVector() const noexcept override { return true; }
        [[nodiscard]] GlobalIndex numRows() const noexcept override { return n; }
        [[nodiscard]] GlobalIndex numCols() const noexcept override { return n; }
        [[nodiscard]] std::string backendName() const override { return "DenseAccum"; }
        void zero() override {
            std::fill(vec.begin(), vec.end(), 0.0);
            std::fill(mat.begin(), mat.end(), 0.0);
        }
    };

    DenseAccum accum(residual_out, matrix_out, static_cast<GlobalIndex>(n_total));
    assembleMixedAuxiliaryIntoGlobal(state, &accum, &accum,
                                      true, true, n_field_dofs, false);
}

void FESystem::finalizeDeferredInputDeps()
{
    // Resolve deferred derived-input expressions: AuxiliaryInputSymbol → AuxiliaryInputRef.
    // Safe to call multiple times — both vectors are cleared after first run.
    if (auxiliary_input_registry_ && !deferred_derived_exprs_.empty()) {
        for (auto& pair : deferred_derived_exprs_) {
            const auto& derived_name = pair.first;
            auto& expr_ptr = pair.second;
            auto* reg = auxiliary_input_registry_.get();
            auto resolve = [reg, &derived_name](const forms::FormExprNode& node)
                -> std::optional<forms::FormExpr> {
                if (node.type() == forms::FormExprType::AuxiliaryInputSymbol) {
                    if (auto sym = node.symbolName()) {
                        const std::string sname{*sym};
                        FE_THROW_IF(!reg->hasInput(sname),
                                    InvalidArgumentException,
                                    "FESystem: derived input '" + derived_name +
                                        "' references unknown input '" + sname + "'");
                        const auto slot = reg->slotOf(sname);
                        return forms::FormExpr::auxiliaryInputRef(
                            static_cast<std::uint32_t>(slot));
                    }
                }
                return std::nullopt;
            };
            *expr_ptr = expr_ptr->transformNodes(resolve);
        }
        deferred_derived_exprs_.clear();
    }

    // Wire deferred input dependencies.
    if (auxiliary_input_registry_ && !deferred_input_deps_.empty()) {
        for (const auto& pair : deferred_input_deps_) {
            const auto& dependent = pair.first;
            const auto& dependency = pair.second;
            FE_THROW_IF(!auxiliary_input_registry_->hasInput(dependency),
                        InvalidArgumentException,
                        "FESystem: derived input '" + dependent +
                            "' references unknown input '" + dependency +
                            "' — ensure all referenced inputs are "
                            "registered before setup()");
            auxiliary_input_registry_->addDependency(dependent, dependency);
        }
        deferred_input_deps_.clear();
    }
}

void FESystem::bindSecondaryFields(BoundaryReductionService& svc,
                                    FieldId primary_fid,
                                    const std::vector<FieldId>& referenced_fields)
{
    if (referenced_fields.size() <= 1) return;  // no secondary fields

    // Compute total dof_per_node from all registered fields.
    // For interleaved layouts, each node stores components from all fields.
    int total_dpn = 0;
    for (const auto& rec : field_registry_.records()) {
        total_dpn += rec.components;
    }
    if (total_dpn > 0) {
        svc.setDofPerNode(total_dpn);
    }

    // Compute per-field component_offset in the interleaved layout.
    // Fields are ordered by FieldId (registration order).
    std::unordered_map<FieldId, int> field_offsets;
    int offset = 0;
    for (std::size_t i = 0; i < field_registry_.records().size(); ++i) {
        const auto fid = static_cast<FieldId>(i);
        field_offsets[fid] = offset;
        offset += field_registry_.records()[i].components;
    }

    for (const auto fid : referenced_fields) {
        if (fid == primary_fid) continue;
        const auto& sec_rec = field_registry_.get(fid);
        if (!sec_rec.space) continue;

        assembly::FieldSolutionBinding binding;
        binding.field = fid;
        binding.space = sec_rec.space.get();
        binding.field_type = sec_rec.space->field_type();
        binding.value_dimension = sec_rec.components;
        binding.n_components = sec_rec.components;
        auto off_it = field_offsets.find(fid);
        binding.component_offset = (off_it != field_offsets.end()) ? off_it->second : 0;
        svc.registerSecondaryField(binding);
    }
}

std::vector<BoundaryReductionService::SensitivityEntry>
FESystem::assembleBoundaryGradient(FieldId field,
                                    const forms::FormExpr& integrand_trial,
                                    int boundary_marker,
                                    const SystemStateView& state,
                                    bool apply_constraints,
                                    int region_marker,
                                    std::span<const GlobalIndex> cell_filter)
{
    const auto& rec = fieldRecord(field);
    FE_CHECK_NOT_NULL(rec.space.get(),
                      "FESystem::assembleBoundaryGradient: field space is null");

    if (!assembler_) return {};

    const auto& fdh = fieldDofHandler(field);
    const auto field_off = fieldDofOffset(field);

    // Create the gradient kernel (forward-mode AD for exact ∂(integrand)/∂(trial_dof_j)).
    forms::BoundaryFunctionalGradientKernel grad_kernel(
        integrand_trial, boundary_marker);

    // Assemble using the StandardAssembler's boundary face pipeline with a
    // lightweight sparse vector accumulator (same pattern as SystemAssembly.cpp).
    const auto n_total = static_cast<GlobalIndex>(dof_handler_.getNumDofs());
    if (n_total <= 0) return {};

    // Lightweight GlobalSystemView that only accumulates vector entries.
    struct GradAccumulator final : public assembly::GlobalSystemView {
        std::unordered_map<GlobalIndex, Real> entries;
        GlobalIndex sz;
        explicit GradAccumulator(GlobalIndex s) : sz(s) {}

        // Matrix ops: no-op (we only need vector).
        void addMatrixEntries(std::span<const GlobalIndex>, std::span<const Real>,
                              assembly::AddMode) override {}
        void addMatrixEntries(std::span<const GlobalIndex>, std::span<const GlobalIndex>,
                              std::span<const Real>, assembly::AddMode) override {}
        void addMatrixEntry(GlobalIndex, GlobalIndex, Real, assembly::AddMode) override {}
        void setDiagonal(std::span<const GlobalIndex>, std::span<const Real>) override {}
        void setDiagonal(GlobalIndex, Real) override {}
        void zeroRows(std::span<const GlobalIndex>, bool) override {}

        // Vector ops.
        void addVectorEntries(std::span<const GlobalIndex> dofs,
                              std::span<const Real> vals,
                              assembly::AddMode) override {
            for (std::size_t i = 0; i < dofs.size(); ++i) {
                if (dofs[i] >= 0 && dofs[i] < sz) entries[dofs[i]] += vals[i];
            }
        }
        void addVectorEntry(GlobalIndex d, Real v, assembly::AddMode) override {
            if (d >= 0 && d < sz) entries[d] += v;
        }
        void setVectorEntries(std::span<const GlobalIndex> dofs,
                              std::span<const Real> vals) override {
            for (std::size_t i = 0; i < dofs.size(); ++i) {
                if (dofs[i] >= 0 && dofs[i] < sz) entries[dofs[i]] = vals[i];
            }
        }
        void zeroVectorEntries(std::span<const GlobalIndex> dofs) override {
            for (const auto d : dofs) entries.erase(d);
        }
        [[nodiscard]] Real getVectorEntry(GlobalIndex d) const override {
            auto it = entries.find(d);
            return (it != entries.end()) ? it->second : 0.0;
        }

        // Lifecycle ops.
        void beginAssemblyPhase() override {}
        void endAssemblyPhase() override {}
        void finalizeAssembly() override {}
        [[nodiscard]] assembly::AssemblyPhase getPhase() const noexcept override {
            return assembly::AssemblyPhase::Building;
        }
        [[nodiscard]] bool hasMatrix() const noexcept override { return false; }
        [[nodiscard]] bool hasVector() const noexcept override { return true; }
        [[nodiscard]] GlobalIndex numRows() const noexcept override { return sz; }
        [[nodiscard]] GlobalIndex numCols() const noexcept override { return sz; }
        [[nodiscard]] std::string backendName() const override { return "GradAccumulator"; }
        void zero() override { entries.clear(); }
    };

    GradAccumulator accum(n_total);

    // Configure the assembler for this field.
    // Assemble dI/du in the same constrained trial space used by the PDE
    // operator so monolithic direct-feedthrough uses free-DOF sensitivities
    // consistent with the assembled Jacobian.
    const auto* restore_constraints =
        (apply_constraints && use_constraints_in_assembly_) ? &affine_constraints_ : nullptr;
    assembler_->setConstraints(restore_constraints);
    assembler_->setRowDofMap(fdh.getDofMap(), field_off);
    assembler_->setColDofMap(fdh.getDofMap(), field_off);

    // Set the solution on the assembler so the gradient kernel can access
    // field values.  Use the GlobalSystemView from the cached solution vector
    // if available, otherwise create a temporary local-span view.
    // The StandardAssembler requires a GlobalSystemView for solution access.
    // Create one from whichever solution source is available.
    struct SpanSolutionView final : public assembly::GlobalSystemView {
        std::span<const Real> data;
        GlobalIndex sz;
        SpanSolutionView(std::span<const Real> d, GlobalIndex s) : data(d), sz(s) {}

        void addMatrixEntries(std::span<const GlobalIndex>, std::span<const Real>,
                              assembly::AddMode) override {}
        void addMatrixEntries(std::span<const GlobalIndex>, std::span<const GlobalIndex>,
                              std::span<const Real>, assembly::AddMode) override {}
        void addMatrixEntry(GlobalIndex, GlobalIndex, Real, assembly::AddMode) override {}
        void setDiagonal(std::span<const GlobalIndex>, std::span<const Real>) override {}
        void setDiagonal(GlobalIndex, Real) override {}
        void zeroRows(std::span<const GlobalIndex>, bool) override {}
        void addVectorEntries(std::span<const GlobalIndex>, std::span<const Real>,
                              assembly::AddMode) override {}
        void addVectorEntry(GlobalIndex, Real, assembly::AddMode) override {}
        void setVectorEntries(std::span<const GlobalIndex>, std::span<const Real>) override {}
        void zeroVectorEntries(std::span<const GlobalIndex>) override {}
        [[nodiscard]] Real getVectorEntry(GlobalIndex d) const override {
            if (d >= 0 && static_cast<std::size_t>(d) < data.size()) return data[static_cast<std::size_t>(d)];
            return 0.0;
        }
        void beginAssemblyPhase() override {}
        void endAssemblyPhase() override {}
        void finalizeAssembly() override {}
        [[nodiscard]] assembly::AssemblyPhase getPhase() const noexcept override {
            return assembly::AssemblyPhase::Building;
        }
        [[nodiscard]] bool hasMatrix() const noexcept override { return false; }
        [[nodiscard]] bool hasVector() const noexcept override { return true; }
        [[nodiscard]] GlobalIndex numRows() const noexcept override { return sz; }
        [[nodiscard]] GlobalIndex numCols() const noexcept override { return sz; }
        [[nodiscard]] std::string backendName() const override { return "SpanSolutionView"; }
        void zero() override {}
    };

    auto refreshGhostedCoefficients = [](const backends::GenericVector* vec_ptr) {
        if (vec_ptr == nullptr) {
            return;
        }
        auto* vec = const_cast<backends::GenericVector*>(vec_ptr);
        vec->updateGhosts();
    };

    std::function<int(const forms::FormExprNode&)> maxPreviousSolutionHistory =
        [&](const forms::FormExprNode& node) -> int {
        int max_history = 0;
        if (node.type() == forms::FormExprType::PreviousSolutionRef) {
            max_history = std::max(max_history, node.historyIndex().value_or(1));
        }
        for (const auto* child : node.children()) {
            if (child != nullptr) {
                max_history = std::max(max_history, maxPreviousSolutionHistory(*child));
            }
        }
        return max_history;
    };

    const int required_history =
        integrand_trial.node() ? maxPreviousSolutionHistory(*integrand_trial.node()) : 0;

    refreshGhostedCoefficients(state.u_vector);
    if (required_history >= 1) {
        refreshGhostedCoefficients(state.u_prev_vector);
    }
    if (required_history >= 2) {
        refreshGhostedCoefficients(state.u_prev2_vector);
    }

    std::unique_ptr<assembly::GlobalSystemView> temp_sol_view;
    std::unique_ptr<assembly::GlobalSystemView> temp_prev_view;
    std::unique_ptr<assembly::GlobalSystemView> temp_prev2_view;
    std::unique_ptr<SpanSolutionView> span_sol_view;
    std::unique_ptr<SpanSolutionView> span_prev_view;
    std::unique_ptr<SpanSolutionView> span_prev2_view;
    if (state.u_vector) {
        auto* vec = const_cast<backends::GenericVector*>(state.u_vector);
        temp_sol_view = vec->createAssemblyView();
        assembler_->setCurrentSolutionView(temp_sol_view.get());
    } else if (!state.u.empty()) {
        // Wrap the raw solution span as a GlobalSystemView so the
        // StandardAssembler can access field values during gradient assembly.
        span_sol_view = std::make_unique<SpanSolutionView>(state.u, n_total);
        assembler_->setCurrentSolutionView(span_sol_view.get());
    }

    if (required_history >= 1 && state.u_prev_vector) {
        auto* vec = const_cast<backends::GenericVector*>(state.u_prev_vector);
        temp_prev_view = vec->createAssemblyView();
        assembler_->setPreviousSolutionView(temp_prev_view.get());
    } else if (required_history >= 1 && !state.u_prev.empty()) {
        span_prev_view = std::make_unique<SpanSolutionView>(state.u_prev, n_total);
        assembler_->setPreviousSolution(state.u_prev);
        assembler_->setPreviousSolutionView(span_prev_view.get());
    }

    if (required_history >= 2 && state.u_prev2_vector) {
        auto* vec = const_cast<backends::GenericVector*>(state.u_prev2_vector);
        temp_prev2_view = vec->createAssemblyView();
        assembler_->setPreviousSolutionViewK(2, temp_prev2_view.get());
    } else if (required_history >= 2 && !state.u_prev2.empty()) {
        span_prev2_view = std::make_unique<SpanSolutionView>(state.u_prev2, n_total);
        assembler_->setPreviousSolutionK(2, state.u_prev2);
        assembler_->setPreviousSolutionViewK(2, span_prev2_view.get());
    }

    const auto sensitivity_options = localSensitivityAssemblyOptions(*assembler_);
    ScopedAssemblerOptions sensitivity_option_guard(
        assembler_.get(), sensitivity_options);

    if (boundary_marker >= 0) {
        // Boundary face gradient assembly.
        assembler_->assembleBoundaryFaces(
            meshAccess(), boundary_marker,
            *rec.space, grad_kernel,
            /*matrix_view=*/nullptr,
            /*vector_view=*/&accum);
    } else {
        // Domain (all-cells or region-filtered) gradient assembly.
        // BoundaryFunctionalGradientKernel has hasCell()=false, so we wrap
        // it in a cell-capable adapter that reuses its Dual-arithmetic
        // evaluation for cell QPs instead of boundary face QPs.
        struct CellGradKernelAdapter final : public assembly::AssemblyKernel {
            forms::BoundaryFunctionalGradientKernel& inner;
            const FESystem& system;
            int region_marker;
            std::span<const GlobalIndex> cell_filter;
            explicit CellGradKernelAdapter(forms::BoundaryFunctionalGradientKernel& k,
                                           const FESystem& s,
                                           int marker,
                                           std::span<const GlobalIndex> cells)
                : inner(k)
                , system(s)
                , region_marker(marker)
                , cell_filter(cells) {}
            [[nodiscard]] bool hasCell() const noexcept override { return true; }
            [[nodiscard]] bool hasBoundaryFace() const noexcept override { return false; }
            [[nodiscard]] bool hasInteriorFace() const noexcept override { return false; }
            [[nodiscard]] bool hasInterfaceFace() const noexcept override { return false; }
            [[nodiscard]] assembly::RequiredData getRequiredData() const noexcept override {
                return inner.getRequiredData();
            }
            [[nodiscard]] std::vector<assembly::FieldRequirement>
            fieldRequirements() const override {
                return inner.fieldRequirements();
            }
            void computeCell(const assembly::AssemblyContext& ctx,
                             assembly::KernelOutput& output) override {
                if (region_marker >= 0) {
                    const auto cell_id = ctx.cellId();
                    if (cell_id < 0 ||
                        system.meshAccess().getCellDomainId(cell_id) != region_marker) {
                        output.reserve(ctx.numTestDofs(), ctx.numTrialDofs(),
                                       /*need_matrix=*/false, /*need_vector=*/true);
                        std::fill(output.local_vector.begin(), output.local_vector.end(), 0.0);
                        return;
                    }
                }
                if (!cell_filter.empty()) {
                    const auto cell_id = ctx.cellId();
                    if (cell_id < 0 ||
                        std::find(cell_filter.begin(), cell_filter.end(), cell_id) ==
                            cell_filter.end()) {
                        output.reserve(ctx.numTestDofs(), ctx.numTrialDofs(),
                                       /*need_matrix=*/false, /*need_vector=*/true);
                        std::fill(output.local_vector.begin(), output.local_vector.end(), 0.0);
                        return;
                    }
                }
                // Reuse the boundary face computation logic (which uses
                // Dual arithmetic for per-DOF derivatives) but call it
                // for a cell context.  The gradient kernel's computeBoundaryFace
                // reads basis values and QP weights from the context, which
                // are also valid for cell QPs.
                inner.computeBoundaryFace(ctx, -1, output);
            }
        };

        CellGradKernelAdapter cell_adapter(grad_kernel, *this, region_marker, cell_filter);
        assembler_->assembleVector(
            meshAccess(), *rec.space, cell_adapter, accum);
    }

    // Convert to SensitivityEntry pairs.
    std::vector<BoundaryReductionService::SensitivityEntry> result;
    result.reserve(accum.entries.size());
    for (const auto& [dof, val] : accum.entries) {
        if (std::abs(val) > 1e-16) {
            result.push_back({dof, val});
        }
    }

    assembler_->setConstraints(restore_constraints);

    return result;
}

std::span<const Real> FESystem::auxiliaryOutputValues() const noexcept
{
    // Flatten output buffers from all deployed entries.
    aux_output_flat_.clear();
    for (const auto& entry : deployed_aux_entries_) {
        aux_output_flat_.insert(aux_output_flat_.end(),
                                 entry.output_buffer.begin(),
                                 entry.output_buffer.end());
    }
    return aux_output_flat_;
}

std::span<const Real> FESystem::auxiliaryStateValues() const noexcept
{
    aux_state_flat_.clear();
    if (!auxiliary_state_manager_) {
        return aux_state_flat_;
    }

    const auto& state = auxiliary_state_manager_->state();
    for (std::size_t i = 0; i < state.blockCount(); ++i) {
        const auto work = state.block(i).work();
        aux_state_flat_.insert(aux_state_flat_.end(), work.begin(), work.end());
    }
    return aux_state_flat_;
}

Real FESystem::auxiliaryConstraintValue(std::string_view instance_name,
                                        const AuxiliaryConstraintBinding& binding,
                                        Real time,
                                        Real dt) const
{
    const auto& entry = findDeployedAuxEntry_(instance_name);
    FE_THROW_IF(!auxiliary_state_manager_ ||
                    !auxiliary_state_manager_->hasBlock(entry.instance_name),
                InvalidStateException,
                "FESystem::auxiliaryConstraintValue: auxiliary block '" +
                    entry.instance_name + "' is not finalized");

    const auto& blk = auxiliary_state_manager_->getBlock(entry.instance_name);
    FE_THROW_IF(blk.entityCount() != 1u, NotImplementedException,
                "FESystem::auxiliaryConstraintValue: only single-entity auxiliary "
                "constraint sources are supported for instance '" + entry.instance_name + "'");

    auto state_vec =
        (binding.state_view == AuxiliaryOutputStateView::Committed)
            ? blk.gatherEntityCommitted(/*entity_index=*/0)
            : blk.gatherEntityWork(/*entity_index=*/0);

    if (binding.value_source == AuxiliaryConstraintValueSource::State) {
        auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(entry.model.get());
        FE_THROW_IF(!built, InvalidArgumentException,
                    "FESystem::auxiliaryConstraintValue: state-driven auxiliary "
                    "constraints require a BuiltAuxiliaryModel for instance '" +
                        entry.instance_name + "'");
        const auto& state_names = built->stateNames();
        auto it = std::find(state_names.begin(), state_names.end(), binding.value_name);
        FE_THROW_IF(it == state_names.end(), InvalidArgumentException,
                    "FESystem::auxiliaryConstraintValue: unknown state '" +
                        binding.value_name + "' on instance '" + entry.instance_name + "'");
        const auto idx = static_cast<std::size_t>(std::distance(state_names.begin(), it));
        FE_THROW_IF(idx >= state_vec.size(), InvalidStateException,
                    "FESystem::auxiliaryConstraintValue: state index out of range for '" +
                        binding.value_name + "'");
        return state_vec[idx];
    }

    FE_THROW_IF(entry.model->outputCount() <= 0, InvalidArgumentException,
                "FESystem::auxiliaryConstraintValue: instance '" + entry.instance_name +
                    "' has no outputs");
    FE_THROW_IF(entry.deriv_provider &&
                    !entry.deriv_provider->artifact().referenced_fields.empty(),
                NotImplementedException,
                "FESystem::auxiliaryConstraintValue: auxiliary-driven strong Dirichlet "
                "constraints do not yet support outputs that directly reference FE fields");

    auto params = buildParamVector(entry);
    auto inputs = buildInputVector(entry);

    std::vector<Real> xdot(state_vec.size(), Real{0.0});
    std::vector<std::vector<Real>> history_storage;
    auto history_spans = buildHistorySpans_(blk, /*entity_index=*/0, history_storage);

    AuxiliaryLocalContext ctx;
    ctx.time = time;
    ctx.dt = dt;
    ctx.effective_dt = dt;
    ctx.x = state_vec;
    ctx.xdot = xdot;
    ctx.history = history_spans;
    ctx.inputs = inputs;
    ctx.params = params;
    ctx.entity_index = 0;
    ctx.field_values = {};
    ctx.user_data = nullptr;

    std::vector<Real> outputs(static_cast<std::size_t>(entry.model->outputCount()), 0.0);
    entry.model->evaluateOutputs(ctx, outputs);

    const auto output_names = entry.model->outputNames();
    auto out_it = std::find(output_names.begin(), output_names.end(), binding.value_name);
    FE_THROW_IF(out_it == output_names.end(), InvalidArgumentException,
                "FESystem::auxiliaryConstraintValue: unknown output '" +
                    binding.value_name + "' on instance '" + entry.instance_name + "'");
    const auto out_idx = static_cast<std::size_t>(std::distance(output_names.begin(), out_it));
    FE_THROW_IF(out_idx >= outputs.size(), InvalidStateException,
                "FESystem::auxiliaryConstraintValue: output index out of range for '" +
                    binding.value_name + "'");
    return outputs[out_idx];
}

std::size_t FESystem::auxiliaryOutputSlotOf(std::string_view output_name) const
{
    const bool use_materialized_filter = std::any_of(
        deployed_aux_entries_.begin(),
        deployed_aux_entries_.end(),
        [](const auto& entry) { return entry.materialized; });

    int match_count = 0;
    std::string first_instance;
    for (const auto& entry : deployed_aux_entries_) {
        if (use_materialized_filter) {
            if (!entry.materialized) {
                continue;
            }
        } else if (!isAuxiliaryDeploymentVisibleForBareLookup_(entry)) {
            continue;
        }
        for (const auto& oname : entry.model->outputNames()) {
            if (oname == output_name) {
                ++match_count;
                if (match_count == 1) first_instance = entry.instance_name;
            }
        }
    }
    FE_THROW_IF(match_count > 1, InvalidArgumentException,
                "auxiliaryOutputSlotOf(\"" + std::string(output_name) +
                    "\"): ambiguous — " + std::to_string(match_count) +
                    " deployed models have this output name. "
                    "Use auxiliaryOutputSlotOf(instance_name, output_name) instead.");

    if (match_count == 0) return static_cast<std::size_t>(-1);
    return auxiliaryOutputSlotOf(first_instance, output_name);
}

std::size_t FESystem::auxiliaryOutputSlotOf(
    std::string_view instance_name, std::string_view output_name) const
{
    const bool use_materialized_filter = std::any_of(
        deployed_aux_entries_.begin(),
        deployed_aux_entries_.end(),
        [](const auto& entry) { return entry.materialized; });

    std::size_t slot = 0;
    for (const auto& entry : deployed_aux_entries_) {
        if (use_materialized_filter && !entry.materialized) {
            continue;
        }
        auto out_names = entry.model->outputNames();
        const auto n_outputs = out_names.size();
        if (n_outputs == 0) continue;

        std::size_t n_entities = 1;
        if (auxiliary_state_manager_ &&
            auxiliary_state_manager_->hasBlock(entry.instance_name)) {
            n_entities = auxiliary_state_manager_->getBlock(entry.instance_name).entityCount();
        } else if (entry.explicit_entity_count > 0) {
            n_entities = entry.explicit_entity_count;
        }

        if (entry.instance_name == instance_name) {
            for (std::size_t i = 0; i < n_outputs; ++i) {
                if (out_names[i] == output_name) {
                    return slot + i;
                }
            }
        }

        slot += n_entities * n_outputs;
    }
    return static_cast<std::size_t>(-1);
}

std::size_t FESystem::auxiliaryOutputIdOf(std::string_view output_name) const
{
    const auto slash = output_name.find('/');
    if (slash != std::string_view::npos) {
        return auxiliaryOutputIdOf(output_name.substr(0, slash),
                                   output_name.substr(slash + 1));
    }

    int match_count = 0;
    std::size_t match_id = static_cast<std::size_t>(-1);
    for (const auto& entry : deployed_aux_entries_) {
        if (!isAuxiliaryDeploymentVisibleForBareLookup_(entry)) {
            continue;
        }
        const auto output_names = entry.model->outputNames();
        for (std::size_t i = 0; i < output_names.size(); ++i) {
            if (output_names[i] != output_name) {
                continue;
            }
            ++match_count;
            if (i < entry.output_ids.size()) {
                match_id = static_cast<std::size_t>(entry.output_ids[i]);
            }
        }
    }

    FE_THROW_IF(match_count > 1, InvalidArgumentException,
                "auxiliaryOutputIdOf(\"" + std::string(output_name) +
                    "\"): ambiguous — " + std::to_string(match_count) +
                    " deployed models have this output name. "
                    "Use auxiliaryOutputIdOf(instance_name, output_name) instead.");
    return match_id;
}

std::size_t FESystem::auxiliaryOutputIdOf(
    std::string_view instance_name, std::string_view output_name) const
{
    const auto qualified = std::string(instance_name) + "/" + std::string(output_name);
    const auto it = auxiliary_output_id_by_qualified_name_.find(qualified);
    if (it == auxiliary_output_id_by_qualified_name_.end()) {
        return static_cast<std::size_t>(-1);
    }
    return static_cast<std::size_t>(it->second);
}

const FESystem::AuxiliaryOutputDescriptor* FESystem::auxiliaryOutputDescriptor(
    std::size_t output_id) const noexcept
{
    return output_id < auxiliary_output_descriptors_.size()
        ? &auxiliary_output_descriptors_[output_id]
        : nullptr;
}

bool
FESystem::canLowerAlgebraicAuxiliaryToDirectOnly_(const DeployedAuxEntry& entry) const
{
    if (entry.spec.solve_mode != AuxiliarySolveMode::Monolithic) {
        return false;
    }
    auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(entry.model.get());
    if (!built) {
        return false;
    }
    const auto dim = built->stateNames().size();
    if (dim == 0 || !isPureAlgebraicAuxiliary(*entry.model, dim)) {
        return false;
    }
    if (entry.deriv_provider) {
        const auto& artifact = entry.deriv_provider->artifact();
        if (!artifact.referenced_fields.empty()) {
            return false;
        }
    }
    const auto output_names = entry.model->outputNames();
    if (output_names.empty()) {
        return false;
    }
    for (const auto& output_name : output_names) {
        const auto qualified_name = entry.instance_name + "/" + output_name;
        if (lowered_aux_output_exprs_by_name_.find(qualified_name) !=
            lowered_aux_output_exprs_by_name_.end()) {
            continue;
        }
        if (synthesizeLoweredAuxiliaryOutputExpr_(entry, output_name).has_value()) {
            continue;
        }
        return false;
    }
    return true;
}

std::optional<forms::FormExpr>
FESystem::synthesizeLoweredAuxiliaryOutputExpr_(const DeployedAuxEntry& entry,
                                                std::string_view output_name) const
{
    auto trace_block = [&](std::string_view stage) {
        if (!monolithicAuxTraceEnabled()) {
            return;
        }
        FE_LOG_INFO("FESystem::synthesizeLoweredAuxiliaryOutputExpr blocked"
                    " instance=" + entry.instance_name +
                    " output=" + std::string(output_name) +
                    " stage=" + std::string(stage));
    };

    const auto* input_reg = auxiliaryInputRegistryIfPresent();
    if (!input_reg) {
        trace_block("no_input_registry");
        return std::nullopt;
    }

    auto* built = dynamic_cast<const BuiltAuxiliaryModel*>(entry.model.get());
    if (!built) {
        trace_block("not_built_model");
        return std::nullopt;
    }

    const auto& state_names = built->stateNames();
    const auto dim = state_names.size();
    const bool can_inline_state_assignments =
        (dim > 0) && isPureAlgebraicAuxiliary(*entry.model, dim);

    std::vector<forms::FormExpr> explicit_state_exprs;
    if (can_inline_state_assignments) {
        const auto residual_exprs = built->residualExpressions();
        if (residual_exprs.size() < dim) {
            trace_block("residual_size_mismatch");
            return std::nullopt;
        }

        explicit_state_exprs.resize(dim);
        for (std::size_t i = 0; i < dim; ++i) {
            auto explicit_rhs =
                tryExtractExplicitStateAssignment(residual_exprs[i], static_cast<std::uint32_t>(i));
            if (!explicit_rhs) {
                trace_block("explicit_state_assignment");
                return std::nullopt;
            }
            explicit_state_exprs[i] = std::move(*explicit_rhs);
        }

        for (std::size_t pass = 0; pass < dim; ++pass) {
            for (std::size_t i = 0; i < dim; ++i) {
                explicit_state_exprs[i] = explicit_state_exprs[i].transformNodes(
                    [&](const forms::FormExprNode& node) -> std::optional<forms::FormExpr> {
                        if (node.type() != forms::FormExprType::AuxiliaryStateRef) {
                            return std::nullopt;
                        }
                        const auto slot = node.slotIndex();
                        if (!slot || *slot >= explicit_state_exprs.size() || *slot == i ||
                            !explicit_state_exprs[*slot].isValid()) {
                            return std::nullopt;
                        }
                        return explicit_state_exprs[*slot];
                    });
            }
        }
    }

    const auto out_it =
        std::find_if(built->outputExpressions().begin(),
                     built->outputExpressions().end(),
                     [&](const auto& kv) { return kv.first == output_name; });
    if (out_it == built->outputExpressions().end()) {
        trace_block("missing_output_expr");
        return std::nullopt;
    }

    const auto& sig = built->signature();
    auto instantiate = [&](const forms::FormExpr& expr) -> std::optional<forms::FormExpr> {
        if (!expr.isValid()) {
            trace_block("output_expr_invalid");
            return std::nullopt;
        }

        auto replace_terminals =
            [&](const forms::FormExprNode& node) -> std::optional<forms::FormExpr> {
                switch (node.type()) {
                    case forms::FormExprType::AuxiliaryStateRef: {
                        if (!can_inline_state_assignments) {
                            return std::nullopt;
                        }
                        const auto slot = node.slotIndex();
                        if (!slot || *slot >= explicit_state_exprs.size() ||
                            !explicit_state_exprs[*slot].isValid()) {
                            trace_block("bad_state_slot");
                            return std::nullopt;
                        }
                        return explicit_state_exprs[*slot];
                    }
                    case forms::FormExprType::AuxiliaryInputRef: {
                        const auto slot = node.slotIndex();
                        if (!slot || *slot >= sig.inputs.size()) {
                            trace_block("bad_input_slot");
                            return std::nullopt;
                        }
                        const auto& port = sig.inputs[*slot];

                        std::string registry_name;
                        const auto bind_it = entry.input_bindings.find(port.name);
                        if (bind_it != entry.input_bindings.end()) {
                            registry_name = bind_it->second;
                        } else {
                            const auto coupled_it = entry.coupled_bindings.find(port.name);
                            if (coupled_it != entry.coupled_bindings.end()) {
                                registry_name = coupled_it->second.registryName();
                            }
                        }

                        if (!registry_name.empty() && input_reg->hasInput(registry_name)) {
                            if (entry.lower_to_direct_only) {
                                if (const auto coupled_it = entry.coupled_bindings.find(port.name);
                                    coupled_it != entry.coupled_bindings.end()) {
                                    if (const auto* def = coupled_it->second.definition();
                                        def != nullptr &&
                                        def->kind == FEQuantityKind::BoundaryIntegral &&
                                        def->expression.isValid() &&
                                        def->boundary_marker >= 0) {
                                        return forms::FormExpr::boundaryIntegral(
                                            def->expression, def->boundary_marker, registry_name);
                                    }
                                }
                            }
                            return forms::FormExpr::auxiliaryInputRef(
                                static_cast<std::uint32_t>(input_reg->slotOf(registry_name)));
                        }
                        if (port.optional && port.default_value.has_value()) {
                            return forms::FormExpr::constant(*port.default_value);
                        }
                        trace_block("unbound_input");
                        return std::nullopt;
                    }
                    case forms::FormExprType::ParameterRef: {
                        const auto slot = node.slotIndex();
                        if (!slot || *slot >= sig.parameters.size()) {
                            trace_block("bad_parameter_slot");
                            return std::nullopt;
                        }
                        const auto& port = sig.parameters[*slot];
                        const auto param_it = entry.param_values.find(port.name);
                        if (param_it != entry.param_values.end()) {
                            return forms::FormExpr::constant(param_it->second);
                        }
                        if (port.optional && port.default_value.has_value()) {
                            return forms::FormExpr::constant(*port.default_value);
                        }
                        trace_block("unbound_parameter");
                        return std::nullopt;
                    }
                    default:
                        return std::nullopt;
                }
            };

        auto lowered = expr;
        const std::size_t max_passes =
            std::max<std::size_t>(4, dim + sig.inputs.size() + sig.parameters.size() + 1);
        for (std::size_t pass = 0; pass < max_passes; ++pass) {
            lowered = lowered.transformNodes(replace_terminals);
        }

        if (!lowered.isValid() || !lowered.node()) {
            trace_block("lowered_invalid");
            return std::nullopt;
        }
        if (exprContainsType(*lowered.node(), forms::FormExprType::ParameterRef) ||
            exprContainsType(*lowered.node(), forms::FormExprType::ParameterSymbol) ||
            exprContainsType(*lowered.node(), forms::FormExprType::AuxiliaryOutputRef) ||
            exprContainsType(*lowered.node(), forms::FormExprType::AuxiliaryOutputSymbol) ||
            exprContainsType(*lowered.node(), forms::FormExprType::AuxiliaryInputSymbol)) {
            trace_block("unsupported_terminals_remaining");
            return std::nullopt;
        }
        if (entry.lower_to_direct_only &&
            exprContainsType(*lowered.node(), forms::FormExprType::AuxiliaryStateRef)) {
            trace_block("state_refs_remaining");
            return std::nullopt;
        }
        return lowered;
    };

    return instantiate(out_it->second);
}

std::optional<forms::FormExpr>
FESystem::loweredAuxiliaryOutputExpr(std::string_view output_name) const
{
    const auto it = lowered_aux_output_exprs_by_name_.find(std::string(output_name));
    if (it != lowered_aux_output_exprs_by_name_.end()) {
        return it->second;
    }

    const auto slash = output_name.find('/');
    if (slash != std::string_view::npos) {
        const std::string instance_name(output_name.substr(0, slash));
        const std::string local_name(output_name.substr(slash + 1));
        for (const auto& entry : deployed_aux_entries_) {
            if (entry.instance_name != instance_name) {
                continue;
            }
            return synthesizeLoweredAuxiliaryOutputExpr_(entry, local_name);
        }
    }

    std::optional<forms::FormExpr> synthesized;
    for (const auto& entry : deployed_aux_entries_) {
        if (!isAuxiliaryDeploymentVisibleForBareLookup_(entry)) {
            continue;
        }
        auto lowered = synthesizeLoweredAuxiliaryOutputExpr_(entry, output_name);
        if (!lowered) {
            continue;
        }
        FE_THROW_IF(synthesized.has_value(), InvalidArgumentException,
                    "loweredAuxiliaryOutputExpr(\"" + std::string(output_name) +
                        "\"): ambiguous lowered output; use qualified instance/output name");
        synthesized = std::move(lowered);
    }
    if (synthesized) {
        return synthesized;
    }

    const auto output_id = auxiliaryOutputIdOf(output_name);
    if (output_id == static_cast<std::size_t>(-1)) {
        return std::nullopt;
    }
    return loweredAuxiliaryOutputExpr(output_id);
}

std::optional<forms::FormExpr>
FESystem::loweredAuxiliaryOutputExpr(std::size_t output_id) const
{
    const auto it = lowered_aux_output_exprs_by_id_.find(output_id);
    if (it == lowered_aux_output_exprs_by_id_.end()) {
        return std::nullopt;
    }
    return it->second;
}

bool FESystem::auxiliaryOutputMetadataUsesRef(std::string_view output_name) const
{
    const auto slash = output_name.find('/');
    if (slash != std::string_view::npos) {
        const std::string instance_name(output_name.substr(0, slash));
        const std::string local_name(output_name.substr(slash + 1));
        for (const auto& entry : deployed_aux_entries_) {
            if (entry.instance_name != instance_name) {
                continue;
            }
            const auto output_names = entry.model->outputNames();
            const bool found = std::find(output_names.begin(), output_names.end(), local_name) !=
                output_names.end();
            return found && !entry.lower_to_direct_only;
        }
        return false;
    }

    const DeployedAuxEntry* match = nullptr;
    for (const auto& entry : deployed_aux_entries_) {
        if (!isAuxiliaryDeploymentVisibleForBareLookup_(entry)) {
            continue;
        }
        const auto output_names = entry.model->outputNames();
        const bool found = std::find(
            output_names.begin(), output_names.end(), std::string(output_name)) != output_names.end();
        if (!found) {
            continue;
        }
        FE_THROW_IF(
            match != nullptr,
            InvalidArgumentException,
            "auxiliaryOutputMetadataUsesRef(\"" + std::string(output_name) +
                "\"): ambiguous output; use qualified instance/output name");
        match = &entry;
    }

    return match != nullptr && !match->lower_to_direct_only;
}

std::vector<analysis::AuxiliaryOutputConsumerRecord>
FESystem::consumersOfAuxiliaryOutput(std::size_t output_id) const
{
    std::vector<analysis::AuxiliaryOutputConsumerRecord> consumers;
    for (const auto& consumer : auxiliary_output_consumers_) {
        if (consumer.output_id == output_id) {
            consumers.push_back(consumer);
        }
    }
    return consumers;
}

std::vector<analysis::AuxiliaryOutputConsumerRecord>
FESystem::consumersOfInstance(std::string_view instance_name) const
{
    std::vector<analysis::AuxiliaryOutputConsumerRecord> consumers;
    for (const auto& consumer : auxiliary_output_consumers_) {
        const auto* desc = auxiliaryOutputDescriptor(consumer.output_id);
        if (desc && desc->instance_name == instance_name) {
            consumers.push_back(consumer);
        }
    }
    return consumers;
}

void FESystem::buildLoweredAuxiliaryOutputExpressions_()
{
    lowered_aux_output_exprs_by_name_.clear();
    lowered_aux_output_exprs_by_id_.clear();

    for (const auto& entry : deployed_aux_entries_) {
        if (!entry.materialized && entry.activation_mode != AuxiliaryActivationMode::Always) {
            continue;
        }
        for (const auto& output_name : entry.model->outputNames()) {
            auto lowered = synthesizeLoweredAuxiliaryOutputExpr_(entry, output_name);
            if (!lowered) {
                continue;
            }

            const auto qualified_name = entry.instance_name + "/" + output_name;
            lowered_aux_output_exprs_by_name_[qualified_name] = *lowered;

            const auto output_id = auxiliaryOutputIdOf(entry.instance_name, output_name);
            if (output_id != static_cast<std::size_t>(-1)) {
                lowered_aux_output_exprs_by_id_[output_id] = *lowered;
            }
        }
    }
}

void FESystem::buildAuxiliaryOutputBindings_()
{
    auxiliary_output_bindings_.clear();

    for (const auto& entry : deployed_aux_entries_) {
        if (!entry.materialized) {
            continue;
        }
        const auto output_names = entry.model->outputNames();
        if (output_names.empty()) {
            continue;
        }

        for (std::size_t output_index = 0; output_index < output_names.size(); ++output_index) {
            const auto& output_name = output_names[output_index];
            const auto base_slot = auxiliaryOutputSlotOf(entry.instance_name, output_name);
            if (base_slot == static_cast<std::size_t>(-1)) {
                continue;
            }
            if (output_index >= entry.output_ids.size()) {
                continue;
            }

            assembly::AuxiliaryOutputBinding binding;
            binding.output_id = entry.output_ids[output_index];
            binding.storage_offset = static_cast<std::uint32_t>(base_slot);
            binding.scope = toAssemblyAuxiliaryOutputScope(entry.spec.scope);
            binding.outputs_per_entity =
                static_cast<std::uint32_t>(std::max<std::size_t>(1u, output_names.size()));
            binding.entity_map_data = entry.entity_map.empty()
                ? nullptr
                : entry.entity_map.data();
            binding.entity_map_size = entry.entity_map.size();
            binding.qp_offsets_data = entry.qp_offsets.empty()
                ? nullptr
                : entry.qp_offsets.data();
            binding.qp_offsets_size = entry.qp_offsets.size();
            auxiliary_output_bindings_.push_back(binding);
        }
    }
}

bool FESystem::isAuxiliaryDeploymentSelected_(const DeployedAuxEntry& entry) const
{
    if (entry.variant_group.empty()) {
        return true;
    }
    const auto it = auxiliary_variant_selection_.find(entry.variant_group);
    if (it == auxiliary_variant_selection_.end()) {
        return true;
    }
    return entry.variant_key == it->second;
}

bool FESystem::isAuxiliaryDeploymentVisibleForBareLookup_(
    const DeployedAuxEntry& entry) const
{
    return isAuxiliaryDeploymentSelected_(entry);
}

std::vector<analysis::AuxiliaryOutputConsumerRecord>
FESystem::consumersOfEntry_(const DeployedAuxEntry& entry) const
{
    std::vector<analysis::AuxiliaryOutputConsumerRecord> consumers;
    for (const auto output_id : entry.output_ids) {
        for (const auto& consumer : auxiliary_output_consumers_) {
            if (consumer.output_id == output_id) {
                consumers.push_back(consumer);
            }
        }
    }
    return consumers;
}

bool FESystem::hasAuxiliaryConsumers_(const DeployedAuxEntry& entry) const
{
    for (const auto output_id : entry.output_ids) {
        for (const auto& consumer : auxiliary_output_consumers_) {
            if (consumer.output_id == output_id) {
                return true;
            }
        }
    }
    if (!entry.constraint_bindings.empty()) {
        return true;
    }
    return false;
}

bool FESystem::hasCellVolumeAuxiliaryConsumers_(const DeployedAuxEntry& entry) const
{
    for (const auto output_id : entry.output_ids) {
        for (const auto& consumer : auxiliary_output_consumers_) {
            if (consumer.output_id == output_id &&
                consumer.domain_kind == analysis::DomainKind::Cell) {
                return true;
            }
        }
    }
    return false;
}

std::vector<std::size_t> FESystem::collectCoveredCells_(
    const DeployedAuxEntry& entry) const
{
    std::vector<std::size_t> cells;
    if (!entry.entity_map.empty()) {
        cells = entry.entity_map;
        return cells;
    }
    if (!mesh_access_) {
        return cells;
    }
    mesh_access_->forEachOwnedCell([&](GlobalIndex cell_id) {
        cells.push_back(static_cast<std::size_t>(cell_id));
    });
    return cells;
}

void FESystem::assignAuxiliaryOutputIds_(DeployedAuxEntry& entry)
{
    entry.output_ids.clear();
    const auto output_names = entry.model->outputNames();
    entry.output_ids.reserve(output_names.size());
    for (std::size_t output_index = 0; output_index < output_names.size(); ++output_index) {
        AuxiliaryOutputDescriptor descriptor;
        descriptor.id = static_cast<std::uint32_t>(auxiliary_output_descriptors_.size());
        descriptor.instance_name = entry.instance_name;
        descriptor.output_name = output_names[output_index];
        descriptor.output_index = output_index;

        const auto qualified_name =
            descriptor.instance_name + "/" + descriptor.output_name;
        FE_THROW_IF(auxiliary_output_id_by_qualified_name_.count(qualified_name) != 0u,
                    InvalidArgumentException,
                    "FESystem::deployAuxiliaryModel: duplicate auxiliary output '" +
                        qualified_name + "'");

        auxiliary_output_id_by_qualified_name_[qualified_name] = descriptor.id;
        entry.output_ids.push_back(descriptor.id);
        auxiliary_output_descriptors_.push_back(std::move(descriptor));
    }
}

void FESystem::inferQuadraturePointLayout_(DeployedAuxEntry& entry)
{
    FE_THROW_IF(entry.spec.scope != AuxiliaryStateScope::QuadraturePoint,
                InvalidArgumentException,
                "FESystem::inferQuadraturePointLayout_: instance '" +
                    entry.instance_name + "' is not QuadraturePoint scoped");

    const auto covered_cells = collectCoveredCells_(entry);
    entry.entity_map = covered_cells;

    const auto consumers = consumersOfEntry_(entry);
    std::vector<analysis::AuxiliaryOutputConsumerRecord> cell_consumers;
    std::vector<analysis::AuxiliaryOutputConsumerRecord> unsupported_consumers;
    for (const auto& consumer : consumers) {
        if (consumer.domain_kind == analysis::DomainKind::Cell) {
            cell_consumers.push_back(consumer);
        } else {
            unsupported_consumers.push_back(consumer);
        }
    }

    if (!unsupported_consumers.empty()) {
        std::ostringstream oss;
        oss << "FESystem::finalizeAuxiliaryLayout: QuadraturePoint instance '"
            << entry.instance_name
            << "' is consumed on unsupported non-cell domain(s): ";
        for (std::size_t i = 0; i < unsupported_consumers.size(); ++i) {
            if (i != 0) {
                oss << ", ";
            }
            oss << unsupported_consumers[i].operator_tag;
        }
        FE_THROW(InvalidArgumentException, oss.str());
    }

    const bool has_explicit_qp_layout_hint =
        entry.quadrature_reference_field != INVALID_FIELD_ID ||
        !entry.quadrature_reference_operator.empty();
    const bool has_any_consumers = hasAuxiliaryConsumers_(entry);

    auto append_unique_field = [](std::vector<FieldId>& fields, FieldId field) {
        if (field == INVALID_FIELD_ID) {
            return;
        }
        if (std::find(fields.begin(), fields.end(), field) == fields.end()) {
            fields.push_back(field);
        }
    };

    std::vector<FieldId> reference_fields;
    append_unique_field(reference_fields, entry.quadrature_reference_field);
    if (!entry.quadrature_reference_operator.empty()) {
        bool found_operator_layout = false;
        for (const auto& record : formulation_records_) {
            if (record.operator_tag != entry.quadrature_reference_operator ||
                std::find(record.active_domains.begin(),
                          record.active_domains.end(),
                          analysis::DomainKind::Cell) == record.active_domains.end()) {
                continue;
            }
            found_operator_layout = true;
            for (const auto field : record.active_fields) {
                append_unique_field(reference_fields, field);
            }
        }
        FE_THROW_IF(!found_operator_layout, InvalidArgumentException,
                    "FESystem::finalizeAuxiliaryLayout: QuadraturePoint instance '" +
                        entry.instance_name +
                        "' quadratureFromOperator('" +
                        entry.quadrature_reference_operator +
                        "') did not resolve to any cell-volume formulation");
    }
    for (const auto& consumer : cell_consumers) {
        append_unique_field(reference_fields, consumer.reference_field);
    }

    if (cell_consumers.empty() && entry.qp_offsets.empty()) {
        if (!has_explicit_qp_layout_hint) {
            FE_THROW_IF(has_any_consumers ||
                            entry.activation_mode == AuxiliaryActivationMode::Always,
                        InvalidStateException,
                        "FESystem::finalizeAuxiliaryLayout: QuadraturePoint instance '" +
                            entry.instance_name +
                            "' is active but has no cell-volume consumers, explicit qpOffsets(), "
                            "or quadratureLike()/quadratureFromOperator() hint");
            entry.materialized = false;
            return;
        }
        FE_THROW_IF(!has_any_consumers &&
                        entry.activation_mode != AuxiliaryActivationMode::Always,
                    InvalidStateException,
                    "FESystem::finalizeAuxiliaryLayout: QuadraturePoint instance '" +
                        entry.instance_name +
                        "' specifies quadratureLike()/quadratureFromOperator() but has no "
                        "active consumers; either reference the output in this run or mark the "
                        "deployment alwaysActive()");
        FE_THROW_IF(reference_fields.empty(), InvalidStateException,
                    "FESystem::finalizeAuxiliaryLayout: QuadraturePoint instance '" +
                        entry.instance_name +
                        "' has no usable quadrature reference field metadata");
    }

    const int global_covered_cells =
        mpiAllreduceSumIfActive(static_cast<int>(entry.entity_map.size()));
    if (entry.entity_map.empty()) {
        FE_THROW_IF(global_covered_cells <= 0,
                    InvalidStateException,
                    "FESystem::finalizeAuxiliaryLayout: QuadraturePoint instance '" +
                        entry.instance_name +
                        "' expanded to zero covered cells");
        entry.qp_offsets = {0u};
        entry.materialized = true;
        return;
    }

    if (entry.qp_offsets.empty()) {
        FE_THROW_IF(!mesh_access_, InvalidStateException,
                    "FESystem::finalizeAuxiliaryLayout: QuadraturePoint instance '" +
                        entry.instance_name +
                        "' requires mesh access to infer quadrature layout");
        FE_THROW_IF(reference_fields.empty(), InvalidStateException,
                    "FESystem::finalizeAuxiliaryLayout: QuadraturePoint instance '" +
                        entry.instance_name +
                        "' has no cell-volume consumers or quadrature hint capable of "
                        "supplying reference field metadata");

        const auto* first_space = fieldRecord(reference_fields.front()).space.get();
        FE_CHECK_NOT_NULL(first_space,
                          "FESystem::inferQuadraturePointLayout_: reference space");
        auto inferred_offsets = buildAuxiliaryCellQuadratureOffsets(
            *mesh_access_, *first_space, entry.entity_map);

        for (std::size_t i = 1; i < reference_fields.size(); ++i) {
            const auto* other_space = fieldRecord(reference_fields[i]).space.get();
            FE_CHECK_NOT_NULL(other_space,
                              "FESystem::inferQuadraturePointLayout_: comparison space");
            const auto other_offsets = buildAuxiliaryCellQuadratureOffsets(
                *mesh_access_, *other_space, entry.entity_map);
            FE_THROW_IF(other_offsets != inferred_offsets,
                        InvalidArgumentException,
                        "FESystem::finalizeAuxiliaryLayout: QuadraturePoint instance '" +
                            entry.instance_name +
                            "' has active consumers with incompatible quadrature layouts");
        }

        entry.qp_offsets = std::move(inferred_offsets);
    } else {
        FE_THROW_IF(entry.qp_offsets.size() != entry.entity_map.size() + 1u,
                    InvalidArgumentException,
                    "FESystem::finalizeAuxiliaryLayout: QuadraturePoint instance '" +
                        entry.instance_name +
                        "' has qpOffsets().size()=" +
                        std::to_string(entry.qp_offsets.size()) +
                        " but covers " + std::to_string(entry.entity_map.size()) +
                        " cells");
        if (!reference_fields.empty() && mesh_access_) {
            const auto* first_space = fieldRecord(reference_fields.front()).space.get();
            FE_CHECK_NOT_NULL(first_space,
                              "FESystem::inferQuadraturePointLayout_: explicit comparison space");
            const auto inferred_offsets = buildAuxiliaryCellQuadratureOffsets(
                *mesh_access_, *first_space, entry.entity_map);
            FE_THROW_IF(inferred_offsets != entry.qp_offsets,
                        InvalidArgumentException,
                        "FESystem::finalizeAuxiliaryLayout: QuadraturePoint instance '" +
                            entry.instance_name +
                            "' explicit qpOffsets() do not match inferred consumer/hint layout");
        }
    }

    entry.materialized = true;
}

std::vector<Real> FESystem::checkpointAuxiliaryState() const
{
    if (auxiliary_state_manager_) {
        return auxiliary_state_manager_->packAll();
    }
    return {};
}

void FESystem::restoreAuxiliaryState(std::span<const Real> data)
{
    if (auxiliary_state_manager_ && !data.empty()) {
        auxiliary_state_manager_->unpackAll(data);
    }
}

FESystem::AuxiliaryAnalysisSummary FESystem::auxiliaryAnalysisSummary() const
{
    AuxiliaryAnalysisSummary summary;

    if (auxiliary_state_manager_) {
        summary.n_blocks = auxiliary_state_manager_->blockCount();
        for (std::size_t i = 0; i < summary.n_blocks; ++i) {
            const auto& blk = auxiliary_state_manager_->state().block(i);
            summary.block_names.push_back(blk.name());
            const auto& spec = auxiliary_state_manager_->getSpec(blk.name());
            if (spec.solve_mode == AuxiliarySolveMode::Partitioned) {
                ++summary.n_partitioned;
            } else {
                ++summary.n_monolithic;
            }
        }
    }

    if (auxiliary_operator_registry_ && auxiliary_operator_registry_->isLayoutFinalized()) {
        summary.total_aux_unknowns = auxiliary_operator_registry_->auxiliaryLayout().total_aux_unknowns;
        summary.constraint_like_block_names = auxiliary_operator_registry_->constraintLikeBlocks();
        summary.schur_eliminable_block_names = auxiliary_operator_registry_->schurEliminableBlocks();
        summary.special_precondition_block_names = auxiliary_operator_registry_->specialPreconditionBlocks();
        summary.n_constraint_like_blocks = summary.constraint_like_block_names.size();
        summary.n_schur_eliminable_blocks = summary.schur_eliminable_block_names.size();
        summary.n_special_precondition_blocks = summary.special_precondition_block_names.size();
    }

    if (auxiliary_input_registry_) {
        summary.n_inputs = auxiliary_input_registry_->inputCount();
        summary.input_names = auxiliary_input_registry_->inputNames();
    }

    return summary;
}

void FESystem::updateConstraints(double time, double dt)
{
    requireSetup();
    (void)refreshConstraintStateForCurrentRevisions(time, dt, /*allow_structural_rebuild=*/true);

    has_last_constraint_update_time_ = true;
    last_constraint_update_time_ = time;
    last_constraint_update_dt_ = dt;
    bool any_updated = false;

    for (const auto& c : constraint_defs_) {
        FE_CHECK_NOT_NULL(c.get(), "FESystem::updateConstraints: constraint");
        if (c->isTimeDependent()) {
            any_updated = c->updateValues(affine_constraints_, time) || any_updated;
        }
    }

    for (auto& c : system_constraint_defs_) {
        FE_CHECK_NOT_NULL(c.get(), "FESystem::updateConstraints: system constraint");
        if (c->isTimeDependent()) {
            any_updated = c->updateValues(*this, affine_constraints_, time, dt) || any_updated;
        }
    }

    if (any_updated) {
        ++constraint_time_epoch_;
        const auto deps = constraintDependencyDeclaration();
        constraint_revision_snapshot_ = captureConstraintRevisionSnapshot(
            deps.structural.mesh_field_values || deps.value.mesh_field_values);
    }
}

} // namespace systems
} // namespace FE
} // namespace svmp
