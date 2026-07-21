#include "LevelSet/LevelSetConservativePhaseState.h"

#include "Assembly/CutIntegrationContext.h"
#include "Basis/BasisFunction.h"
#include "Geometry/MappingFactory.h"
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

} // namespace svmp::FE::level_set
