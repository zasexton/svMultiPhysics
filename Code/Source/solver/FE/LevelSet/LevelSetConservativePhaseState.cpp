#include "LevelSet/LevelSetConservativePhaseState.h"

#include "Assembly/CutIntegrationContext.h"
#include "Basis/BasisFunction.h"
#include "Geometry/MappingFactory.h"
#include "Geometry/CutQuadratureMapping.h"
#include "Quadrature/QuadratureFactory.h"
#include "Spaces/FunctionSpace.h"
#include "Systems/FESystem.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <limits>
#include <memory>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if FE_HAS_MPI
#include <mpi.h>
#endif

namespace svmp::FE::level_set {
namespace {

struct ProjectionCollectiveContext {
    bool active{false};
    int rank{0};
    int size{1};
#if FE_HAS_MPI
    MPI_Comm communicator{MPI_COMM_NULL};
#endif
};

#if FE_HAS_MPI
[[nodiscard]] MPI_Datatype mpiRealType() noexcept
{
    if constexpr (std::is_same_v<Real, float>) {
        return MPI_FLOAT;
    } else if constexpr (std::is_same_v<Real, double>) {
        return MPI_DOUBLE;
    }
    return MPI_LONG_DOUBLE;
}

[[nodiscard]] MPI_Datatype mpiUnsigned64Type() noexcept
{
#ifdef MPI_UINT64_T
    return MPI_UINT64_T;
#else
    if constexpr (std::is_same_v<std::uint64_t, unsigned long>) {
        return MPI_UNSIGNED_LONG;
    }
    return MPI_UNSIGNED_LONG_LONG;
#endif
}
#endif

[[nodiscard]] ProjectionCollectiveContext projectionCollectiveContext(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& dofs)
{
    ProjectionCollectiveContext context;
    context.rank = mesh.parallelRank();
    context.size = mesh.parallelSize();
    if (context.rank < 0 || context.size < 1 ||
        context.rank >= context.size) {
        throw std::invalid_argument(
            "P1 phase projection received invalid mesh rank metadata");
    }
#if FE_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
        MPI_Finalized(&finalized);
    }
    if (context.size > 1) {
        if (initialized == 0 || finalized != 0 ||
            dofs.mpiComm() == MPI_COMM_NULL) {
            throw std::invalid_argument(
                "P1 phase projection requires an active field communicator");
        }
        int communicator_rank = 0;
        int communicator_size = 1;
        MPI_Comm_rank(dofs.mpiComm(), &communicator_rank);
        MPI_Comm_size(dofs.mpiComm(), &communicator_size);
        if (communicator_rank != context.rank ||
            communicator_size != context.size) {
            throw std::invalid_argument(
                "P1 phase projection mesh and field communicators disagree");
        }
        context.active = true;
        context.communicator = dofs.mpiComm();
    }
#else
    (void)dofs;
    if (context.size > 1) {
        throw std::invalid_argument(
            "P1 phase projection cannot use a multi-rank mesh without MPI support");
    }
#endif
    return context;
}

[[nodiscard]] bool synchronizeFailure(
    const ProjectionCollectiveContext& context,
    bool local_success,
    const std::string& local_diagnostic,
    std::string& collective_diagnostic)
{
#if FE_HAS_MPI
    if (context.active) {
        const int local_failed_rank =
            local_success ? context.size : context.rank;
        int first_failed_rank = context.size;
        MPI_Allreduce(&local_failed_rank, &first_failed_rank, 1,
                      MPI_INT, MPI_MIN, context.communicator);
        if (first_failed_rank < context.size) {
            constexpr std::size_t maximum_diagnostic_bytes = 4096u;
            int length = 0;
            if (context.rank == first_failed_rank) {
                length = static_cast<int>(std::min(
                    maximum_diagnostic_bytes, local_diagnostic.size()));
            }
            MPI_Bcast(&length, 1, MPI_INT, first_failed_rank,
                      context.communicator);
            std::vector<char> bytes(static_cast<std::size_t>(length));
            if (context.rank == first_failed_rank && length > 0) {
                std::copy_n(local_diagnostic.data(), length, bytes.data());
            }
            if (length > 0) {
                MPI_Bcast(bytes.data(), length, MPI_CHAR,
                          first_failed_rank, context.communicator);
            }
            collective_diagnostic =
                "P1 phase projection failed on rank " +
                std::to_string(first_failed_rank) + ": " +
                std::string(bytes.begin(), bytes.end());
            return false;
        }
    }
#else
    (void)context;
#endif
    if (!local_success) {
        collective_diagnostic = local_diagnostic;
        return false;
    }
    return true;
}

void allReduceRealBufferSum(const ProjectionCollectiveContext& context,
                            std::vector<Real>& values)
{
#if FE_HAS_MPI
    if (context.active && !values.empty()) {
        constexpr std::size_t maximum_chunk =
            static_cast<std::size_t>(std::numeric_limits<int>::max());
        std::vector<Real> reduced(values.size(), Real{0.0});
        for (std::size_t offset = 0u; offset < values.size();) {
            const std::size_t count =
                std::min(maximum_chunk, values.size() - offset);
            const auto displacement = static_cast<std::ptrdiff_t>(offset);
            MPI_Allreduce(values.data() + displacement,
                          reduced.data() + displacement,
                          static_cast<int>(count), mpiRealType(), MPI_SUM,
                          context.communicator);
            offset += count;
        }
        values.swap(reduced);
    }
#else
    (void)context;
    (void)values;
#endif
}

[[nodiscard]] Real allReduceRealSum(
    const ProjectionCollectiveContext& context,
    Real local)
{
#if FE_HAS_MPI
    if (context.active) {
        Real global{0.0};
        MPI_Allreduce(&local, &global, 1, mpiRealType(), MPI_SUM,
                      context.communicator);
        return global;
    }
#else
    (void)context;
#endif
    return local;
}

[[nodiscard]] Real allReduceRealMax(
    const ProjectionCollectiveContext& context,
    Real local)
{
#if FE_HAS_MPI
    if (context.active) {
        Real global{0.0};
        MPI_Allreduce(&local, &global, 1, mpiRealType(), MPI_MAX,
                      context.communicator);
        return global;
    }
#else
    (void)context;
#endif
    return local;
}

[[nodiscard]] Real allReduceRealMin(
    const ProjectionCollectiveContext& context,
    Real local)
{
#if FE_HAS_MPI
    if (context.active) {
        Real global{0.0};
        MPI_Allreduce(&local, &global, 1, mpiRealType(), MPI_MIN,
                      context.communicator);
        return global;
    }
#else
    (void)context;
#endif
    return local;
}

[[nodiscard]] std::uint64_t allReduceUnsigned64Sum(
    const ProjectionCollectiveContext& context,
    std::uint64_t local)
{
#if FE_HAS_MPI
    if (context.active) {
        std::uint64_t global = 0u;
        MPI_Allreduce(&local, &global, 1, mpiUnsigned64Type(), MPI_SUM,
                      context.communicator);
        return global;
    }
#else
    (void)context;
#endif
    return local;
}

[[nodiscard]] std::uint64_t allReduceUnsigned64Min(
    const ProjectionCollectiveContext& context,
    std::uint64_t local)
{
#if FE_HAS_MPI
    if (context.active) {
        std::uint64_t global = 0u;
        MPI_Allreduce(&local, &global, 1, mpiUnsigned64Type(), MPI_MIN,
                      context.communicator);
        return global;
    }
#else
    (void)context;
#endif
    return local;
}

[[nodiscard]] std::uint64_t allReduceUnsigned64Max(
    const ProjectionCollectiveContext& context,
    std::uint64_t local)
{
#if FE_HAS_MPI
    if (context.active) {
        std::uint64_t global = 0u;
        MPI_Allreduce(&local, &global, 1, mpiUnsigned64Type(), MPI_MAX,
                      context.communicator);
        return global;
    }
#else
    (void)context;
#endif
    return local;
}

[[nodiscard]] int resolvedQuadratureOrder(int requested,
                                          int geometry_order)
{
    if (requested > 0) {
        return requested;
    }
    return std::max(4, 3 * std::max(1, geometry_order));
}

[[nodiscard]] std::shared_ptr<geometry::GeometryMapping> makeCellMapping(
    const assembly::IMeshAccess& mesh,
    GlobalIndex cell)
{
    std::vector<std::array<Real, 3>> coordinates;
    mesh.getCellCoordinates(cell, coordinates);
    if (coordinates.empty()) {
        throw std::invalid_argument(
            "P1 phase projection found a cell without geometry nodes");
    }
    std::vector<math::Vector<Real, 3>> nodes;
    nodes.reserve(coordinates.size());
    for (const auto& coordinate : coordinates) {
        nodes.push_back(math::Vector<Real, 3>{
            coordinate[0], coordinate[1], coordinate[2]});
    }
    geometry::MappingRequest request;
    request.element_type = mesh.getCellType(cell);
    request.geometry_order = mesh.getCellGeometryOrder(cell);
    request.use_affine = request.geometry_order <= 1;
    return geometry::MappingFactory::create(request, nodes);
}

[[nodiscard]] Real scaledTolerance(
    Real tolerance,
    std::initializer_list<Real> values)
{
    Real scale = Real{1.0};
    for (const Real value : values) {
        scale = std::max(scale, std::abs(value));
    }
    return tolerance * scale;
}

} // namespace

LevelSetP1PhaseProjectionResult
projectLevelSetP1PhaseIndicatorFromCutContext(
    const systems::FESystem& system,
    FieldId liquid_indicator_field,
    const LevelSetP1PhaseTransportGraph& graph,
    const LevelSetP1PhaseProjectionOptions& options)
{
    LevelSetP1PhaseProjectionResult result;
    result.interface_marker = options.interface_marker;
    result.liquid_side = options.liquid_side;
    try {
        if (!graph.success || liquid_indicator_field == INVALID_FIELD_ID) {
            result.diagnostic =
                "P1 phase projection requires a valid phase graph and field";
            return result;
        }
        if (options.interface_marker < 0 ||
            options.liquid_side == geometry::CutIntegrationSide::Interface ||
            options.quadrature_order < 0 ||
            !std::isfinite(options.invariant_tolerance) ||
            options.invariant_tolerance < Real{0.0}) {
            result.diagnostic =
                "P1 phase projection received invalid marker, side, quadrature, or tolerance options";
            return result;
        }

        const auto& mesh = system.meshAccess();
        const auto& record = system.fieldRecord(liquid_indicator_field);
        const auto& dofs = system.fieldDofHandler(liquid_indicator_field);
        const auto collective = projectionCollectiveContext(mesh, dofs);
        bool local_preflight_success = true;
        std::string local_preflight_diagnostic;
        const auto reject_preflight = [&](std::string diagnostic) {
            if (local_preflight_success) {
                local_preflight_success = false;
                local_preflight_diagnostic = std::move(diagnostic);
            }
        };
        if (record.components != 1 || !record.space ||
            record.space->space_type() != spaces::SpaceType::H1 ||
            record.space->field_type() != FieldType::Scalar ||
            record.space->continuity() != Continuity::C0 ||
            record.space->value_dimension() != 1 ||
            record.space->is_variable_order() ||
            record.space->polynomial_order() != 1 ||
            !system.fieldParticipatesInUnknownVector(liquid_indicator_field)) {
            reject_preflight(
                "P1 phase projection requires a transported scalar P1 H1 field");
        }
        if (dofs.getNumDofs() <= 0 ||
            graph.nodes != static_cast<std::size_t>(dofs.getNumDofs()) ||
            graph.lumped_control_volume.size() != graph.nodes ||
            graph.dimension != mesh.dimension() ||
            graph.parallel_rank != collective.rank ||
            graph.parallel_size != collective.size ||
            graph.geometry_revision != mesh.geometryRevision() ||
            graph.topology_revision != mesh.topologyRevision() ||
            graph.ownership_revision != mesh.ownershipRevision() ||
            graph.numbering_revision != mesh.numberingRevision() ||
            graph.dof_layout_revision != dofs.dofLayoutRevision()) {
            reject_preflight(
                "P1 phase projection graph does not match the current mesh and field layout");
        }

        const auto* cut_context = system.cutIntegrationContext();
        if (cut_context == nullptr ||
            !cut_context->hasGeneratedVolumeMarker(options.interface_marker)) {
            reject_preflight(
                "P1 phase projection requires an authoritative generated cut-volume marker");
        }
        if (!synchronizeFailure(
                collective, local_preflight_success,
                local_preflight_diagnostic, result.diagnostic)) {
            return result;
        }
        result.cut_context_revision = cut_context->contentRevision();
        result.source_value_revision =
            cut_context->expectedGeneratedSourceValueRevision(
                options.interface_marker);
        const bool local_revision_valid =
            result.source_value_revision != 0u;
        const std::string revision_diagnostic =
            "P1 phase projection requires revision-tagged cut-volume rules";
        if (!synchronizeFailure(
                collective, local_revision_valid, revision_diagnostic,
                result.diagnostic)) {
            return result;
        }
        const auto minimum_source_revision = allReduceUnsigned64Min(
            collective, result.source_value_revision);
        const auto maximum_source_revision = allReduceUnsigned64Max(
            collective, result.source_value_revision);
        if (minimum_source_revision != maximum_source_revision) {
            result.diagnostic =
                "P1 phase projection requires one source value revision on every rank";
            return result;
        }
        result.source_value_revision = minimum_source_revision;

        std::vector<const geometry::CutQuadratureRule*> rules;
        bool local_rule_snapshot_success = true;
        std::string local_rule_snapshot_diagnostic;
        try {
            cut_context->assertAllFreeSurfaceGeometrySnapshotsCurrent(mesh);
            rules = cut_context->generatedVolumeRulesForMarkerAndSide(
                options.interface_marker, options.liquid_side);
        } catch (const std::exception& exception) {
            local_rule_snapshot_success = false;
            local_rule_snapshot_diagnostic = exception.what();
        }
        if (!synchronizeFailure(
                collective, local_rule_snapshot_success,
                local_rule_snapshot_diagnostic, result.diagnostic)) {
            return result;
        }
        result.nodes = graph.nodes;
        result.liquid_phase_mass.assign(result.nodes, Real{0.0});

        bool local_success = true;
        std::string local_diagnostic;
        Real local_measure{0.0};
        Real local_maximum_rule_closure{0.0};
        std::uint64_t local_owned_rules = 0u;
        std::uint64_t local_quadrature_points = 0u;
        std::set<GlobalIndex> owned_parent_cells;
        try {
            for (const auto* rule : rules) {
                if (rule == nullptr ||
                    rule->kind != geometry::CutQuadratureKind::Volume ||
                    rule->side != options.liquid_side ||
                    rule->provenance.marker != options.interface_marker ||
                    rule->provenance.source_value_revision !=
                        result.source_value_revision) {
                    throw std::invalid_argument(
                        "P1 phase projection received a malformed or stale cut-volume rule");
                }
                const auto cell = static_cast<GlobalIndex>(
                    rule->provenance.parent_entity);
                if (cell < 0 || cell >= mesh.numCells()) {
                    throw std::invalid_argument(
                        "P1 phase projection received a rule outside the local mesh");
                }
                if (!mesh.isOwnedCell(cell)) {
                    continue;
                }
                if (rule->provenance.owner_rank >= 0 &&
                    rule->provenance.owner_rank != collective.rank) {
                    throw std::invalid_argument(
                        "P1 phase projection found inconsistent rule ownership");
                }
                if (!owned_parent_cells.insert(cell).second) {
                    throw std::invalid_argument(
                        "P1 phase projection found duplicate liquid rules for one owned cell");
                }
                if (!rule->full_cell_equivalent &&
                    rule->exact_polynomial_order < 1) {
                    throw std::invalid_argument(
                        "P1 phase projection requires partial rules exact for linear moments");
                }

                const auto cell_dofs = dofs.getCellDofs(cell);
                const auto& element = record.space->getElement(
                    mesh.getCellType(cell), cell);
                const auto& basis = element.basis();
                if (basis.is_vector_valued() ||
                    basis.basis_type() != BasisType::Lagrange ||
                    basis.order() != 1 ||
                    basis.size() != cell_dofs.size()) {
                    throw std::invalid_argument(
                        "P1 phase projection found an incompatible cell basis");
                }
                for (const auto dof : cell_dofs) {
                    if (dof < 0 ||
                        static_cast<std::size_t>(dof) >= result.nodes) {
                        throw std::invalid_argument(
                            "P1 phase projection found a cell DOF outside the phase layout");
                    }
                }

                const auto mapping = makeCellMapping(mesh, cell);
                if (!mapping || mapping->dimension() != graph.dimension) {
                    throw std::invalid_argument(
                        "P1 phase projection found an incompatible geometry mapping");
                }
                std::vector<Real> local_moments(
                    cell_dofs.size(), Real{0.0});
                std::vector<Real> values;
                Real rule_measure{0.0};
                const auto accumulate_point =
                    [&](const spaces::FunctionSpace::Value& xi,
                        Real physical_weight) {
                        if (!std::isfinite(physical_weight) ||
                            !(physical_weight > Real{0.0})) {
                            throw std::invalid_argument(
                                "P1 phase projection found a nonpositive mapped quadrature weight");
                        }
                        basis.evaluate_values(xi, values);
                        if (values.size() != local_moments.size()) {
                            throw std::invalid_argument(
                                "P1 phase projection received inconsistent basis values");
                        }
                        Real value_sum{0.0};
                        for (std::size_t i = 0; i < values.size(); ++i) {
                            if (!std::isfinite(values[i])) {
                                throw std::invalid_argument(
                                    "P1 phase projection found a non-finite basis value");
                            }
                            value_sum += values[i];
                            local_moments[i] += physical_weight * values[i];
                        }
                        const Real partition_residual =
                            std::abs(value_sum - Real{1.0});
                        if (partition_residual > scaledTolerance(
                                options.invariant_tolerance,
                                {value_sum, Real{1.0}})) {
                            throw std::invalid_argument(
                                "P1 phase projection basis failed partition of unity");
                        }
                        rule_measure += physical_weight;
                        ++local_quadrature_points;
                    };

                if (rule->full_cell_equivalent) {
                    const int quadrature_order = resolvedQuadratureOrder(
                        options.quadrature_order,
                        mesh.getCellGeometryOrder(cell));
                    const auto full_rule =
                        quadrature::QuadratureFactory::create(
                            mesh.getCellType(cell), quadrature_order);
                    if (!full_rule || full_rule->num_points() == 0u) {
                        throw std::invalid_argument(
                            "P1 phase projection received an empty full-cell rule");
                    }
                    for (std::size_t q = 0;
                         q < full_rule->num_points(); ++q) {
                        const auto xi = full_rule->point(q);
                        const Real determinant =
                            mapping->jacobian_determinant(xi);
                        const Real weight = full_rule->weight(q);
                        if (!std::isfinite(determinant) ||
                            !(determinant > Real{0.0}) ||
                            !std::isfinite(weight) ||
                            !(weight > Real{0.0})) {
                            throw std::invalid_argument(
                                "P1 phase projection found invalid full-cell geometry");
                        }
                        accumulate_point(xi, determinant * weight);
                    }
                } else {
                    if (rule->points.empty()) {
                        throw std::invalid_argument(
                            "P1 phase projection received an empty partial rule");
                    }
                    for (const auto& point : rule->points) {
                        spaces::FunctionSpace::Value xi{};
                        const auto& coordinate =
                            rule->frame == geometry::CutGeometryFrame::Reference
                                ? point.point
                                : point.parent_coordinate;
                        for (int d = 0; d < graph.dimension; ++d) {
                            const auto index = static_cast<std::size_t>(d);
                            if (!std::isfinite(coordinate[index])) {
                                throw std::invalid_argument(
                                    "P1 phase projection found a non-finite parent coordinate");
                            }
                            xi[index] = coordinate[index];
                        }
                        Real physical_weight = point.weight;
                        if (rule->frame ==
                            geometry::CutGeometryFrame::Reference) {
                            const Real determinant =
                                mapping->jacobian_determinant(xi);
                            if (!std::isfinite(determinant) ||
                                !(determinant > Real{0.0})) {
                                throw std::invalid_argument(
                                    "P1 phase projection found invalid partial-rule geometry");
                            }
                            physical_weight *= determinant;
                        }
                        accumulate_point(xi, physical_weight);
                    }
                }

                Real moment_sum{0.0};
                for (std::size_t i = 0; i < local_moments.size(); ++i) {
                    const Real moment = local_moments[i];
                    if (!std::isfinite(moment)) {
                        throw std::invalid_argument(
                            "P1 phase projection produced a non-finite phase moment");
                    }
                    result.liquid_phase_mass[
                        static_cast<std::size_t>(cell_dofs[i])] += moment;
                    moment_sum += moment;
                }
                const Real closure = std::abs(moment_sum - rule_measure);
                local_maximum_rule_closure = std::max(
                    local_maximum_rule_closure, closure);
                if (closure > scaledTolerance(
                        options.invariant_tolerance,
                        {moment_sum, rule_measure})) {
                    throw std::invalid_argument(
                        "P1 phase projection failed a cut-rule moment identity");
                }
                local_measure += rule_measure;
                ++local_owned_rules;
            }
        } catch (const std::exception& exception) {
            local_success = false;
            local_diagnostic = exception.what();
        }
        if (!synchronizeFailure(
                collective, local_success, local_diagnostic,
                result.diagnostic)) {
            return result;
        }

        allReduceRealBufferSum(collective, result.liquid_phase_mass);
        result.retained_liquid_measure =
            allReduceRealSum(collective, local_measure);
        result.maximum_rule_moment_closure_residual =
            allReduceRealMax(collective, local_maximum_rule_closure);
        result.owned_rules = static_cast<std::size_t>(
            allReduceUnsigned64Sum(collective, local_owned_rules));
        result.quadrature_points = static_cast<std::size_t>(
            allReduceUnsigned64Sum(collective, local_quadrature_points));

        result.minimum_liquid_indicator =
            std::numeric_limits<Real>::infinity();
        result.maximum_liquid_indicator =
            -std::numeric_limits<Real>::infinity();
        result.liquid_indicator.assign(result.nodes, Real{0.0});
        result.phase_bounds_satisfied = true;
        result.complement_bounds_satisfied = true;
        for (std::size_t node = 0; node < result.nodes; ++node) {
            const Real mass = graph.lumped_control_volume[node];
            const Real phase_mass = result.liquid_phase_mass[node];
            if (!std::isfinite(mass) || !(mass > Real{0.0}) ||
                !std::isfinite(phase_mass)) {
                result.diagnostic =
                    "P1 phase projection encountered invalid nodal masses";
                return result;
            }
            const Real bound_tolerance = scaledTolerance(
                options.invariant_tolerance, {mass, phase_mass});
            result.maximum_lower_bound_violation = std::max(
                result.maximum_lower_bound_violation,
                std::max(Real{0.0}, -phase_mass));
            result.maximum_upper_bound_violation = std::max(
                result.maximum_upper_bound_violation,
                std::max(Real{0.0}, phase_mass - mass));
            result.phase_bounds_satisfied =
                result.phase_bounds_satisfied &&
                phase_mass >= -bound_tolerance &&
                phase_mass <= mass + bound_tolerance;
            result.complement_bounds_satisfied =
                result.complement_bounds_satisfied &&
                mass - phase_mass >= -bound_tolerance;
            const Real indicator = std::clamp(
                phase_mass / mass, Real{0.0}, Real{1.0});
            result.liquid_indicator[node] = indicator;
            result.minimum_liquid_indicator = std::min(
                result.minimum_liquid_indicator, indicator);
            result.maximum_liquid_indicator = std::max(
                result.maximum_liquid_indicator, indicator);
            result.projected_liquid_measure += mass * indicator;
        }
        result.measure_closure_residual =
            result.projected_liquid_measure -
            result.retained_liquid_measure;
        const Real measure_tolerance = scaledTolerance(
            options.invariant_tolerance,
            {graph.physical_measure,
             result.retained_liquid_measure,
             result.projected_liquid_measure});
        result.rule_moment_closure_satisfied =
            result.maximum_rule_moment_closure_residual <=
            measure_tolerance;
        result.global_measure_closure_satisfied =
            std::abs(result.measure_closure_residual) <=
            measure_tolerance;
        if (!result.phase_bounds_satisfied ||
            !result.complement_bounds_satisfied ||
            !result.rule_moment_closure_satisfied ||
            !result.global_measure_closure_satisfied ||
            result.retained_liquid_measure < -measure_tolerance ||
            result.retained_liquid_measure >
                graph.physical_measure + measure_tolerance) {
            result.diagnostic =
                "P1 phase projection failed a bound or retained-measure identity";
            return result;
        }

        result.success = true;
        result.diagnostic = "ok";
    } catch (const std::exception& exception) {
        result.success = false;
        result.diagnostic = exception.what();
    }
    return result;
}

LevelSetP1PhaseGeometrySensitivityResult
buildLevelSetP1PhaseGeometrySensitivity(
    const systems::FESystem& system,
    FieldId level_set_field,
    FieldId liquid_indicator_field,
    const LevelSetP1PhaseTransportGraph& graph,
    const LevelSetP1PhaseProjectionOptions& options,
    std::span<const Real> solution)
{
    LevelSetP1PhaseGeometrySensitivityResult result;
    result.interface_marker = options.interface_marker;
    try {
        if (!graph.success || level_set_field == INVALID_FIELD_ID ||
            liquid_indicator_field == INVALID_FIELD_ID ||
            level_set_field == liquid_indicator_field) {
            result.diagnostic =
                "P1 phase geometry sensitivity requires distinct valid fields and a current phase graph";
            return result;
        }
        if (options.interface_marker < 0 ||
            options.liquid_side == geometry::CutIntegrationSide::Interface ||
            !std::isfinite(options.invariant_tolerance) ||
            options.invariant_tolerance < Real{0.0}) {
            result.diagnostic =
                "P1 phase geometry sensitivity received invalid marker, side, or tolerance options";
            return result;
        }

        const auto& mesh = system.meshAccess();
        const auto& phase_record = system.fieldRecord(liquid_indicator_field);
        const auto& level_set_record = system.fieldRecord(level_set_field);
        const auto& phase_dofs =
            system.fieldDofHandler(liquid_indicator_field);
        const auto& level_set_dofs = system.fieldDofHandler(level_set_field);
        const auto collective = projectionCollectiveContext(mesh, phase_dofs);
        result.dimension = mesh.dimension();
        result.nodes = graph.nodes;

        const auto fixed_scalar_p1 = [](const auto& record) {
            return record.components == 1 && record.space &&
                   record.space->space_type() == spaces::SpaceType::H1 &&
                   record.space->field_type() == FieldType::Scalar &&
                   record.space->continuity() == Continuity::C0 &&
                   record.space->value_dimension() == 1 &&
                   !record.space->is_variable_order() &&
                   record.space->polynomial_order() == 1;
        };
        bool local_preflight_success = true;
        std::string local_preflight_diagnostic;
        const auto reject_preflight = [&](std::string diagnostic) {
            if (local_preflight_success) {
                local_preflight_success = false;
                local_preflight_diagnostic = std::move(diagnostic);
            }
        };
        if (!fixed_scalar_p1(phase_record) ||
            !fixed_scalar_p1(level_set_record)) {
            reject_preflight(
                "P1 phase geometry sensitivity requires fixed scalar P1 H1 fields");
        }
        if (phase_dofs.getNumDofs() <= 0 ||
            phase_dofs.getNumDofs() != level_set_dofs.getNumDofs() ||
            graph.nodes !=
                static_cast<std::size_t>(phase_dofs.getNumDofs()) ||
            graph.dimension != result.dimension ||
            graph.parallel_rank != collective.rank ||
            graph.parallel_size != collective.size ||
            graph.geometry_revision != mesh.geometryRevision() ||
            graph.topology_revision != mesh.topologyRevision() ||
            graph.ownership_revision != mesh.ownershipRevision() ||
            graph.numbering_revision != mesh.numberingRevision() ||
            graph.dof_layout_revision != phase_dofs.dofLayoutRevision()) {
            reject_preflight(
                "P1 phase geometry sensitivity graph or field layout is stale or incompatible");
        }
        const auto level_set_offset = static_cast<std::size_t>(
            system.fieldDofOffset(level_set_field));
        if (level_set_offset + graph.nodes > solution.size()) {
            reject_preflight(
                "P1 phase geometry sensitivity level-set slice exceeds the FE solution layout");
        }
        if (local_preflight_success) {
            try {
                mesh.forEachOwnedCell([&](GlobalIndex cell) {
                    const auto phase_cell_dofs =
                        phase_dofs.getCellDofs(cell);
                    const auto level_set_cell_dofs =
                        level_set_dofs.getCellDofs(cell);
                    if (phase_cell_dofs.size() !=
                            level_set_cell_dofs.size() ||
                        !std::equal(phase_cell_dofs.begin(),
                                    phase_cell_dofs.end(),
                                    level_set_cell_dofs.begin())) {
                        throw std::invalid_argument(
                            "P1 phase geometry sensitivity requires identical phase and level-set cell numbering");
                    }
                });
                result.field_layouts_identical = true;
            } catch (const std::exception& exception) {
                reject_preflight(exception.what());
            }
        }

        const auto* cut_context = system.cutIntegrationContext();
        if (cut_context == nullptr ||
            !cut_context->hasGeneratedInterfaceMarker(
                options.interface_marker)) {
            reject_preflight(
                "P1 phase geometry sensitivity requires an authoritative generated interface marker");
        }
        if (!synchronizeFailure(
                collective, local_preflight_success,
                local_preflight_diagnostic, result.diagnostic)) {
            return result;
        }

        result.cut_context_revision = cut_context->contentRevision();
        result.source_value_revision =
            cut_context->expectedGeneratedSourceValueRevision(
                options.interface_marker);
        if (!synchronizeFailure(
                collective, result.source_value_revision != 0u,
                "P1 phase geometry sensitivity requires revision-tagged interface rules",
                result.diagnostic)) {
            return result;
        }
        const auto minimum_source_revision = allReduceUnsigned64Min(
            collective, result.source_value_revision);
        const auto maximum_source_revision = allReduceUnsigned64Max(
            collective, result.source_value_revision);
        if (minimum_source_revision != maximum_source_revision) {
            result.diagnostic =
                "P1 phase geometry sensitivity requires one source value revision on every rank";
            return result;
        }
        result.source_value_revision = minimum_source_revision;

        std::vector<const geometry::CutQuadratureRule*> rules;
        bool local_rule_snapshot_success = true;
        std::string local_rule_snapshot_diagnostic;
        try {
            cut_context->assertAllFreeSurfaceGeometrySnapshotsCurrent(mesh);
            rules = cut_context->interfaceRulesForMarker(
                options.interface_marker);
        } catch (const std::exception& exception) {
            local_rule_snapshot_success = false;
            local_rule_snapshot_diagnostic = exception.what();
        }
        if (!synchronizeFailure(
                collective, local_rule_snapshot_success,
                local_rule_snapshot_diagnostic, result.diagnostic)) {
            return result;
        }

        std::map<std::pair<GlobalIndex, GlobalIndex>, std::size_t>
            edge_indices;
        for (std::size_t edge_index = 0u;
             edge_index < graph.edges.size(); ++edge_index) {
            const auto& edge = graph.edges[edge_index];
            if (edge.first_node < 0 || edge.second_node < 0 ||
                edge.first_node >= edge.second_node ||
                static_cast<std::size_t>(edge.second_node) >= graph.nodes ||
                !edge_indices
                     .emplace(std::pair{edge.first_node, edge.second_node},
                              edge_index)
                     .second) {
                result.diagnostic =
                    "P1 phase geometry sensitivity received a malformed phase graph edge";
                return result;
            }
        }

        result.diagonal.assign(graph.nodes, Real{0.0});
        std::vector<Real> edge_coefficients(graph.edges.size(), Real{0.0});
        Real local_interface_measure{0.0};
        Real local_minimum_gradient =
            std::numeric_limits<Real>::infinity();
        Real local_minimum_node_distance =
            std::numeric_limits<Real>::infinity();
        std::uint64_t local_owned_rules = 0u;
        std::uint64_t local_quadrature_points = 0u;
        bool local_assembly_success = true;
        std::string local_assembly_diagnostic;
        try {
            for (const auto* rule : rules) {
                if (rule == nullptr ||
                    rule->kind != geometry::CutQuadratureKind::Interface ||
                    rule->provenance.marker != options.interface_marker ||
                    rule->provenance.source_value_revision !=
                        result.source_value_revision ||
                    rule->frame != geometry::CutGeometryFrame::Reference ||
                    rule->points.empty()) {
                    throw std::invalid_argument(
                        "P1 phase geometry sensitivity received a malformed or stale interface rule");
                }
                const auto cell = static_cast<GlobalIndex>(
                    rule->provenance.parent_entity);
                if (cell < 0 || cell >= mesh.numCells()) {
                    throw std::invalid_argument(
                        "P1 phase geometry sensitivity received an interface rule outside the local mesh");
                }
                if (!mesh.isOwnedCell(cell)) {
                    continue;
                }
                if (rule->provenance.owner_rank >= 0 &&
                    rule->provenance.owner_rank != collective.rank) {
                    throw std::invalid_argument(
                        "P1 phase geometry sensitivity found inconsistent rule ownership");
                }

                const auto phase_cell_dofs = phase_dofs.getCellDofs(cell);
                const auto level_set_cell_dofs =
                    level_set_dofs.getCellDofs(cell);
                if (phase_cell_dofs.empty() ||
                    phase_cell_dofs.size() !=
                        level_set_cell_dofs.size() ||
                    !std::equal(phase_cell_dofs.begin(),
                                phase_cell_dofs.end(),
                                level_set_cell_dofs.begin())) {
                    throw std::invalid_argument(
                        "P1 phase geometry sensitivity found incompatible cell field layouts");
                }
                const auto& element = level_set_record.space->getElement(
                    mesh.getCellType(cell), cell);
                const auto& basis = element.basis();
                if (basis.is_vector_valued() ||
                    basis.basis_type() != BasisType::Lagrange ||
                    basis.order() != 1 ||
                    basis.size() != phase_cell_dofs.size()) {
                    throw std::invalid_argument(
                        "P1 phase geometry sensitivity found an incompatible cell basis");
                }
                for (const auto dof : phase_cell_dofs) {
                    if (dof < 0 ||
                        static_cast<std::size_t>(dof) >= graph.nodes) {
                        throw std::invalid_argument(
                            "P1 phase geometry sensitivity found a cell DOF outside the field layout");
                    }
                }

                std::vector<std::array<Real, 3>> coordinates;
                mesh.getCellCoordinates(cell, coordinates);
                for (std::size_t i = 0u; i < coordinates.size(); ++i) {
                    for (std::size_t j = i + 1u; j < coordinates.size(); ++j) {
                        Real distance_squared{0.0};
                        for (int d = 0; d < result.dimension; ++d) {
                            const auto coordinate =
                                static_cast<std::size_t>(d);
                            const Real difference =
                                coordinates[i][coordinate] -
                                coordinates[j][coordinate];
                            distance_squared += difference * difference;
                        }
                        if (distance_squared > Real{0.0} &&
                            std::isfinite(distance_squared)) {
                            local_minimum_node_distance = std::min(
                                local_minimum_node_distance,
                                std::sqrt(distance_squared));
                        }
                    }
                }

                const auto mapping = makeCellMapping(mesh, cell);
                const auto mapped_rule =
                    geometry::mapCutQuadratureRuleToPhysical(mesh, *rule);
                if (!mapping ||
                    mapping->dimension() != result.dimension ||
                    mapped_rule.points.size() != rule->points.size()) {
                    throw std::invalid_argument(
                        "P1 phase geometry sensitivity found incompatible mapped interface geometry");
                }

                std::vector<Real> values;
                std::vector<basis::Gradient> reference_gradients;
                for (const auto& point : mapped_rule.points) {
                    spaces::FunctionSpace::Value xi{};
                    for (int d = 0; d < result.dimension; ++d) {
                        xi[static_cast<std::size_t>(d)] =
                            point.reference_point[static_cast<std::size_t>(d)];
                    }
                    basis.evaluate_values(xi, values);
                    basis.evaluate_gradients(xi, reference_gradients);
                    if (values.size() != phase_cell_dofs.size() ||
                        reference_gradients.size() !=
                            phase_cell_dofs.size()) {
                        throw std::invalid_argument(
                            "P1 phase geometry sensitivity received inconsistent basis evaluations");
                    }
                    const auto inverse = mapping->jacobian_inverse(xi);
                    math::Vector<Real, 3> level_set_gradient{};
                    Real partition_sum{0.0};
                    for (std::size_t local = 0u;
                         local < phase_cell_dofs.size(); ++local) {
                        if (!std::isfinite(values[local])) {
                            throw std::invalid_argument(
                                "P1 phase geometry sensitivity found a non-finite basis value");
                        }
                        partition_sum += values[local];
                        const auto physical_gradient =
                            mapping->transform_gradient(
                                reference_gradients[local], inverse);
                        const Real coefficient =
                            solution[level_set_offset +
                                     static_cast<std::size_t>(
                                         level_set_cell_dofs[local])];
                        if (!std::isfinite(coefficient)) {
                            throw std::invalid_argument(
                                "P1 phase geometry sensitivity found a non-finite level-set coefficient");
                        }
                        for (int d = 0; d < result.dimension; ++d) {
                            level_set_gradient[
                                static_cast<std::size_t>(d)] +=
                                coefficient * physical_gradient[
                                                  static_cast<std::size_t>(d)];
                        }
                    }
                    if (std::abs(partition_sum - Real{1.0}) >
                        scaledTolerance(options.invariant_tolerance,
                                        {partition_sum, Real{1.0}})) {
                        throw std::invalid_argument(
                            "P1 phase geometry sensitivity basis failed partition of unity");
                    }
                    Real gradient_squared{0.0};
                    for (int d = 0; d < result.dimension; ++d) {
                        const Real value = level_set_gradient[
                            static_cast<std::size_t>(d)];
                        gradient_squared += value * value;
                    }
                    const Real gradient_norm = std::sqrt(gradient_squared);
                    if (!std::isfinite(gradient_norm) ||
                        !(gradient_norm > options.invariant_tolerance) ||
                        !std::isfinite(point.physical_weight) ||
                        !(point.physical_weight > Real{0.0})) {
                        throw std::invalid_argument(
                            "P1 phase geometry sensitivity found a degenerate interface gradient or weight");
                    }
                    local_minimum_gradient = std::min(
                        local_minimum_gradient, gradient_norm);
                    local_interface_measure += point.physical_weight;
                    const Real factor =
                        point.physical_weight / gradient_norm;
                    for (std::size_t i = 0u;
                         i < phase_cell_dofs.size(); ++i) {
                        const auto node_i = static_cast<std::size_t>(
                            phase_cell_dofs[i]);
                        result.diagonal[node_i] +=
                            factor * values[i] * values[i];
                        for (std::size_t j = i + 1u;
                             j < phase_cell_dofs.size(); ++j) {
                            const auto endpoints = std::minmax(
                                phase_cell_dofs[i], phase_cell_dofs[j]);
                            const auto edge_it = edge_indices.find(
                                {endpoints.first, endpoints.second});
                            if (edge_it == edge_indices.end()) {
                                throw std::invalid_argument(
                                    "P1 phase geometry sensitivity could not match a cell pair to the phase graph");
                            }
                            edge_coefficients[edge_it->second] +=
                                factor * values[i] * values[j];
                        }
                    }
                    ++local_quadrature_points;
                }
                ++local_owned_rules;
            }
        } catch (const std::exception& exception) {
            local_assembly_success = false;
            local_assembly_diagnostic = exception.what();
        }
        if (!synchronizeFailure(
                collective, local_assembly_success,
                local_assembly_diagnostic, result.diagnostic)) {
            return result;
        }

        allReduceRealBufferSum(collective, result.diagonal);
        allReduceRealBufferSum(collective, edge_coefficients);
        result.interface_measure = allReduceRealSum(
            collective, local_interface_measure);
        result.minimum_level_set_gradient = allReduceRealMin(
            collective, local_minimum_gradient);
        result.minimum_cell_node_distance = allReduceRealMin(
            collective, local_minimum_node_distance);
        result.owned_rules = static_cast<std::size_t>(
            allReduceUnsigned64Sum(collective, local_owned_rules));
        result.quadrature_points = static_cast<std::size_t>(
            allReduceUnsigned64Sum(collective, local_quadrature_points));
        result.edges.reserve(graph.edges.size());
        for (std::size_t edge_index = 0u;
             edge_index < graph.edges.size(); ++edge_index) {
            result.edges.push_back(LevelSetP1PhaseGeometrySensitivityEdge{
                .first_node = graph.edges[edge_index].first_node,
                .second_node = graph.edges[edge_index].second_node,
                .coefficient = edge_coefficients[edge_index],
            });
        }

        const Real maximum_diagonal = result.diagonal.empty()
                                          ? Real{0.0}
                                          : *std::max_element(
                                                result.diagonal.begin(),
                                                result.diagonal.end());
        const Real active_tolerance =
            options.invariant_tolerance *
            std::max(maximum_diagonal,
                     std::numeric_limits<Real>::min());
        result.positive_diagonal_satisfied = true;
        for (const Real diagonal : result.diagonal) {
            if (!std::isfinite(diagonal) || diagonal < Real{0.0}) {
                result.positive_diagonal_satisfied = false;
                break;
            }
            result.active_nodes += diagonal > active_tolerance ? 1u : 0u;
        }

        std::vector<Real> level_set(graph.nodes, Real{0.0});
        for (std::size_t node = 0u; node < graph.nodes; ++node) {
            level_set[node] = solution[level_set_offset + node];
        }
        std::vector<Real> null_residual(graph.nodes, Real{0.0});
        Real null_scale{0.0};
        for (std::size_t node = 0u; node < graph.nodes; ++node) {
            null_residual[node] = result.diagonal[node] * level_set[node];
            null_scale = std::max(
                null_scale,
                std::abs(result.diagonal[node] * level_set[node]));
        }
        for (const auto& edge : result.edges) {
            const auto first = static_cast<std::size_t>(edge.first_node);
            const auto second = static_cast<std::size_t>(edge.second_node);
            null_residual[first] += edge.coefficient * level_set[second];
            null_residual[second] += edge.coefficient * level_set[first];
            null_scale = std::max(
                null_scale,
                std::abs(edge.coefficient) *
                    std::max(std::abs(level_set[first]),
                             std::abs(level_set[second])));
        }
        for (const Real residual : null_residual) {
            result.maximum_level_set_null_residual = std::max(
                result.maximum_level_set_null_residual,
                std::abs(residual));
        }
        result.level_set_null_space_satisfied =
            result.maximum_level_set_null_residual <=
            options.invariant_tolerance *
                std::max(Real{1.0}, null_scale);

        if (!result.field_layouts_identical ||
            !result.positive_diagonal_satisfied ||
            !result.level_set_null_space_satisfied ||
            result.active_nodes == 0u || result.owned_rules == 0u ||
            result.quadrature_points == 0u ||
            !std::isfinite(result.interface_measure) ||
            !(result.interface_measure > Real{0.0}) ||
            !std::isfinite(result.minimum_level_set_gradient) ||
            !(result.minimum_level_set_gradient > Real{0.0}) ||
            !std::isfinite(result.minimum_cell_node_distance) ||
            !(result.minimum_cell_node_distance > Real{0.0})) {
            result.diagnostic =
                "P1 phase geometry sensitivity failed a layout, positivity, null-space, or interface-geometry identity";
            return result;
        }

        result.success = true;
        result.diagnostic = "ok";
    } catch (const std::exception& exception) {
        result.success = false;
        result.diagnostic = exception.what();
    }
    return result;
}

LevelSetP1PhaseGeometryCorrectionResult
solveLevelSetP1PhaseGeometryCorrection(
    const LevelSetP1PhaseGeometrySensitivityResult& sensitivity,
    geometry::CutIntegrationSide liquid_side,
    std::span<const Real> current_level_set,
    std::span<const Real> current_liquid_phase_mass,
    std::span<const Real> target_liquid_phase_mass,
    const LevelSetP1PhaseGeometryCorrectionOptions& options)
{
    LevelSetP1PhaseGeometryCorrectionResult result;
    try {
        const std::size_t nodes = sensitivity.nodes;
        if (!sensitivity.success || nodes == 0u ||
            sensitivity.diagonal.size() != nodes ||
            current_level_set.size() != nodes ||
            current_liquid_phase_mass.size() != nodes ||
            target_liquid_phase_mass.size() != nodes ||
            liquid_side == geometry::CutIntegrationSide::Interface) {
            result.diagnostic =
                "P1 phase geometry correction received incompatible sensitivity, side, or state spans";
            return result;
        }
        if (!std::isfinite(options.invariant_tolerance) ||
            options.invariant_tolerance < Real{0.0} ||
            !std::isfinite(options.relative_linear_tolerance) ||
            !(options.relative_linear_tolerance > Real{0.0}) ||
            options.relative_linear_tolerance > Real{1.0} ||
            options.maximum_linear_iterations < 0) {
            result.diagnostic =
                "P1 phase geometry correction received invalid solver options";
            return result;
        }

        Real matrix_scale{0.0};
        for (const Real diagonal : sensitivity.diagonal) {
            if (!std::isfinite(diagonal) || diagonal < Real{0.0}) {
                result.diagnostic =
                    "P1 phase geometry correction found an invalid diagonal coefficient";
                return result;
            }
            matrix_scale = std::max(matrix_scale, diagonal);
        }
        const Real active_tolerance =
            options.invariant_tolerance *
            std::max(matrix_scale, std::numeric_limits<Real>::min());
        std::vector<std::uint8_t> active(nodes, 0u);
        for (std::size_t node = 0u; node < nodes; ++node) {
            if (sensitivity.diagonal[node] > active_tolerance) {
                active[node] = 1u;
                ++result.active_nodes;
            }
        }
        if (result.active_nodes == 0u) {
            result.diagnostic =
                "P1 phase geometry correction found no interface-supported nodes";
            return result;
        }

        std::vector<std::size_t> parent(nodes);
        std::iota(parent.begin(), parent.end(), std::size_t{0u});
        const auto find_root = [&](std::size_t node) {
            while (parent[node] != node) {
                node = parent[node];
            }
            return node;
        };
        const auto unite = [&](std::size_t first, std::size_t second) {
            auto first_root = find_root(first);
            auto second_root = find_root(second);
            if (first_root != second_root) {
                if (first_root > second_root) {
                    std::swap(first_root, second_root);
                }
                parent[second_root] = first_root;
            }
        };
        for (const auto& edge : sensitivity.edges) {
            if (edge.first_node < 0 || edge.second_node < 0 ||
                edge.first_node >= edge.second_node ||
                static_cast<std::size_t>(edge.second_node) >= nodes ||
                !std::isfinite(edge.coefficient) ||
                edge.coefficient < Real{0.0}) {
                result.diagnostic =
                    "P1 phase geometry correction found a malformed sensitivity edge";
                return result;
            }
            const auto first = static_cast<std::size_t>(edge.first_node);
            const auto second = static_cast<std::size_t>(edge.second_node);
            if (active[first] != 0u && active[second] != 0u &&
                edge.coefficient > active_tolerance) {
                unite(first, second);
            }
        }

        std::map<std::size_t, std::size_t> component_indices;
        std::vector<std::size_t> component(nodes,
                                           std::numeric_limits<std::size_t>::max());
        for (std::size_t node = 0u; node < nodes; ++node) {
            if (active[node] == 0u) {
                continue;
            }
            const auto root = find_root(node);
            const auto [it, inserted] = component_indices.emplace(
                root, component_indices.size());
            (void)inserted;
            component[node] = it->second;
        }
        result.interface_components = component_indices.size();
        std::vector<Real> null_norm_squared(
            result.interface_components, Real{0.0});
        for (std::size_t node = 0u; node < nodes; ++node) {
            if (active[node] == 0u ||
                !std::isfinite(current_level_set[node])) {
                if (!std::isfinite(current_level_set[node])) {
                    result.diagnostic =
                        "P1 phase geometry correction found a non-finite level-set value";
                    return result;
                }
                continue;
            }
            const auto index = component[node];
            null_norm_squared[index] +=
                current_level_set[node] * current_level_set[node];
        }
        for (const Real norm_squared : null_norm_squared) {
            if (!std::isfinite(norm_squared) ||
                !(norm_squared > options.invariant_tolerance *
                                      options.invariant_tolerance)) {
                result.diagnostic =
                    "P1 phase geometry correction could not identify a nondegenerate level-set scaling mode";
                return result;
            }
        }

        const auto apply_matrix = [&](std::span<const Real> input,
                                      std::vector<Real>& output) {
            output.assign(nodes, Real{0.0});
            for (std::size_t node = 0u; node < nodes; ++node) {
                output[node] = sensitivity.diagonal[node] * input[node];
            }
            for (const auto& edge : sensitivity.edges) {
                const auto first =
                    static_cast<std::size_t>(edge.first_node);
                const auto second =
                    static_cast<std::size_t>(edge.second_node);
                output[first] += edge.coefficient * input[second];
                output[second] += edge.coefficient * input[first];
            }
        };
        const auto project_null_modes = [&](std::vector<Real>& values) {
            std::vector<Real> coefficients(result.interface_components,
                                           Real{0.0});
            for (std::size_t node = 0u; node < nodes; ++node) {
                if (active[node] != 0u) {
                    coefficients[component[node]] +=
                        values[node] * current_level_set[node];
                }
            }
            for (std::size_t index = 0u;
                 index < coefficients.size(); ++index) {
                coefficients[index] /= null_norm_squared[index];
            }
            for (std::size_t node = 0u; node < nodes; ++node) {
                if (active[node] != 0u) {
                    values[node] -= coefficients[component[node]] *
                                    current_level_set[node];
                } else {
                    values[node] = Real{0.0};
                }
            }
        };
        const auto dot_product = [&](std::span<const Real> first,
                                     std::span<const Real> second) {
            Real value{0.0};
            for (std::size_t node = 0u; node < nodes; ++node) {
                value += first[node] * second[node];
            }
            return value;
        };
        const auto norm = [&](std::span<const Real> values) {
            return std::sqrt(std::max(
                Real{0.0}, dot_product(values, values)));
        };

        const Real side_derivative =
            liquid_side == geometry::CutIntegrationSide::Negative
                ? Real{-1.0}
                : Real{1.0};
        std::vector<Real> right_hand_side(nodes, Real{0.0});
        Real inactive_residual{0.0};
        Real mass_scale = std::max(Real{1.0}, sensitivity.interface_measure);
        for (std::size_t node = 0u; node < nodes; ++node) {
            const Real current = current_liquid_phase_mass[node];
            const Real target = target_liquid_phase_mass[node];
            if (!std::isfinite(current) || !std::isfinite(target)) {
                result.diagnostic =
                    "P1 phase geometry correction found a non-finite phase moment";
                return result;
            }
            const Real difference = target - current;
            mass_scale = std::max(
                mass_scale, std::max(std::abs(current), std::abs(target)));
            right_hand_side[node] = side_derivative * difference;
            if (active[node] == 0u) {
                inactive_residual = std::max(
                    inactive_residual, std::abs(difference));
            }
        }
        result.right_hand_side_norm = norm(right_hand_side);
        std::vector<Real> projected_right_hand_side = right_hand_side;
        project_null_modes(projected_right_hand_side);
        std::vector<Real> removed_component(nodes, Real{0.0});
        for (std::size_t node = 0u; node < nodes; ++node) {
            removed_component[node] =
                right_hand_side[node] - projected_right_hand_side[node];
        }
        result.maximum_null_compatibility_residual = inactive_residual;
        for (const Real value : removed_component) {
            result.maximum_null_compatibility_residual = std::max(
                result.maximum_null_compatibility_residual,
                std::abs(value));
        }

        const Real absolute_tolerance =
            options.invariant_tolerance * mass_scale;
        const Real projected_norm = norm(projected_right_hand_side);
        if (result.right_hand_side_norm <= absolute_tolerance) {
            result.target_compatible = true;
            result.linear_solve_converged = true;
            result.level_set_increment.assign(nodes, Real{0.0});
            result.predicted_liquid_mass_change.assign(nodes, Real{0.0});
            result.success = true;
            result.diagnostic = "ok";
            return result;
        }
        if (inactive_residual > absolute_tolerance ||
            projected_norm <= absolute_tolerance) {
            result.diagnostic =
                "P1 phase geometry correction target has no interface-supported shape update";
            return result;
        }
        result.target_compatible = true;

        std::vector<Real> increment(nodes, Real{0.0});
        std::vector<Real> residual = projected_right_hand_side;
        std::vector<Real> projected_residual = residual;
        project_null_modes(projected_residual);
        std::vector<Real> direction = projected_residual;
        Real residual_norm_squared =
            dot_product(residual, projected_residual);
        const Real convergence_tolerance = std::max(
            absolute_tolerance,
            options.relative_linear_tolerance * projected_norm);
        const int maximum_iterations =
            options.maximum_linear_iterations > 0
                ? options.maximum_linear_iterations
                : static_cast<int>(std::min<std::size_t>(
                      5000u, std::max<std::size_t>(50u, 4u * nodes)));
        std::vector<Real> matrix_direction;
        for (int iteration = 0; iteration < maximum_iterations; ++iteration) {
            apply_matrix(direction, matrix_direction);
            project_null_modes(matrix_direction);
            const Real curvature =
                dot_product(direction, matrix_direction);
            const Real curvature_floor =
                std::numeric_limits<Real>::epsilon() *
                std::max(matrix_scale,
                         std::numeric_limits<Real>::min()) *
                std::max(dot_product(direction, direction),
                         std::numeric_limits<Real>::min());
            if (!std::isfinite(curvature) ||
                !(curvature > curvature_floor) ||
                !std::isfinite(residual_norm_squared) ||
                !(residual_norm_squared > Real{0.0})) {
                result.diagnostic =
                    "P1 phase geometry correction encountered a singular or indefinite projected sensitivity";
                return result;
            }
            const Real step = residual_norm_squared / curvature;
            for (std::size_t node = 0u; node < nodes; ++node) {
                increment[node] += step * direction[node];
                residual[node] -= step * matrix_direction[node];
            }
            project_null_modes(increment);
            project_null_modes(residual);
            result.iterations = iteration + 1;
            result.linear_residual_norm = norm(residual);
            if (result.linear_residual_norm <= convergence_tolerance) {
                result.linear_solve_converged = true;
                break;
            }
            projected_residual = residual;
            project_null_modes(projected_residual);
            const Real next_residual_norm_squared =
                dot_product(residual, projected_residual);
            if (!std::isfinite(next_residual_norm_squared) ||
                !(next_residual_norm_squared > Real{0.0})) {
                result.diagnostic =
                    "P1 phase geometry correction lost a positive projected residual norm";
                return result;
            }
            const Real beta =
                next_residual_norm_squared / residual_norm_squared;
            for (std::size_t node = 0u; node < nodes; ++node) {
                direction[node] = projected_residual[node] +
                                  beta * direction[node];
            }
            project_null_modes(direction);
            residual_norm_squared = next_residual_norm_squared;
        }
        if (!result.linear_solve_converged) {
            result.diagnostic =
                "P1 phase geometry correction did not converge within its bounded iteration count";
            return result;
        }

        std::vector<Real> matrix_increment;
        apply_matrix(increment, matrix_increment);
        result.level_set_increment = std::move(increment);
        result.predicted_liquid_mass_change.resize(nodes, Real{0.0});
        for (std::size_t node = 0u; node < nodes; ++node) {
            const Real predicted = side_derivative * matrix_increment[node];
            result.predicted_liquid_mass_change[node] = predicted;
            const Real target_change =
                target_liquid_phase_mass[node] -
                current_liquid_phase_mass[node];
            result.maximum_predicted_mass_residual = std::max(
                result.maximum_predicted_mass_residual,
                std::abs(target_change - predicted));
        }
        result.success = true;
        result.diagnostic = "ok";
    } catch (const std::exception& exception) {
        result.success = false;
        result.diagnostic = exception.what();
    }
    return result;
}

} // namespace svmp::FE::level_set
