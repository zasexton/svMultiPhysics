#include "LevelSet/LevelSetConservativePhaseOperator.h"

#include "Basis/BasisFunction.h"
#include "Geometry/MappingFactory.h"
#include "Quadrature/QuadratureFactory.h"
#include "Spaces/FunctionSpace.h"
#include "Systems/FESystem.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <initializer_list>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace svmp::FE::level_set {
namespace {

using Vector3 = std::array<Real, 3>;

struct MutableGradientEdge {
    Vector3 first_test_second_gradient{};
    Vector3 second_test_first_gradient{};
};

[[nodiscard]] Real scaledTolerance(Real tolerance,
                                   std::initializer_list<Real> values)
{
    Real scale = Real{1.0};
    for (const Real value : values) {
        scale = std::max(scale, std::abs(value));
    }
    return tolerance * scale;
}

[[nodiscard]] Real dot(const Vector3& first,
                       const Vector3& second,
                       int dimension) noexcept
{
    Real value{0.0};
    for (int d = 0; d < dimension; ++d) {
        value += first[static_cast<std::size_t>(d)] *
                 second[static_cast<std::size_t>(d)];
    }
    return value;
}

[[nodiscard]] Real norm(const Vector3& value, int dimension) noexcept
{
    return std::sqrt(std::max(Real{0.0}, dot(value, value, dimension)));
}

void addScaled(Vector3& target,
               const math::Vector<Real, 3>& source,
               Real scale,
               int dimension) noexcept
{
    for (int d = 0; d < dimension; ++d) {
        target[static_cast<std::size_t>(d)] +=
            scale * source[static_cast<std::size_t>(d)];
    }
}

void add(Vector3& target, const Vector3& source, int dimension) noexcept
{
    for (int d = 0; d < dimension; ++d) {
        target[static_cast<std::size_t>(d)] +=
            source[static_cast<std::size_t>(d)];
    }
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
            "P1 conservative phase graph found a cell without geometry nodes");
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

} // namespace

LevelSetP1PhaseTransportGraph buildLevelSetP1PhaseTransportGraph(
    const systems::FESystem& system,
    FieldId liquid_indicator_field,
    const LevelSetP1PhaseGraphOptions& options)
{
    LevelSetP1PhaseTransportGraph result;
    try {
        if (liquid_indicator_field == INVALID_FIELD_ID) {
            result.diagnostic =
                "P1 conservative phase graph received an invalid field";
            return result;
        }
        if (!std::isfinite(options.invariant_tolerance) ||
            options.invariant_tolerance < Real{0.0} ||
            options.quadrature_order < 0) {
            result.diagnostic =
                "P1 conservative phase graph requires a nonnegative quadrature order and finite nonnegative tolerance";
            return result;
        }

        const auto& mesh = system.meshAccess();
        if (mesh.parallelSize() != 1 ||
            mesh.numOwnedCells() != mesh.numCells()) {
            result.diagnostic =
                "P1 conservative phase graph requires unique distributed edge ownership before multi-rank assembly";
            return result;
        }
        result.dimension = mesh.dimension();
        if (result.dimension < 2 || result.dimension > 3) {
            result.diagnostic =
                "P1 conservative phase graph supports two- and three-dimensional meshes";
            return result;
        }

        const auto& record = system.fieldRecord(liquid_indicator_field);
        if (record.components != 1 || !record.space ||
            record.space->space_type() != spaces::SpaceType::H1 ||
            record.space->field_type() != FieldType::Scalar ||
            record.space->continuity() != Continuity::C0 ||
            record.space->value_dimension() != 1 ||
            record.space->is_variable_order() ||
            record.space->polynomial_order() != 1) {
            result.diagnostic =
                "P1 conservative phase graph requires a fixed-order scalar P1 H1 field";
            return result;
        }
        if (!system.fieldParticipatesInUnknownVector(liquid_indicator_field)) {
            result.diagnostic =
                "P1 conservative phase graph requires a transported unknown field";
            return result;
        }

        const auto& dofs = system.fieldDofHandler(liquid_indicator_field);
        if (dofs.getNumDofs() <= 0 ||
            dofs.getNumLocalDofs() != dofs.getNumDofs()) {
            result.diagnostic =
                "P1 conservative phase graph requires a nonempty serial field layout";
            return result;
        }
        result.nodes = static_cast<std::size_t>(dofs.getNumDofs());
        result.lumped_control_volume.assign(result.nodes, Real{0.0});
        result.diagonal_gradient.assign(result.nodes, Vector3{});
        result.boundary_column_sum.assign(result.nodes, Vector3{});

        const GlobalIndex field_offset =
            system.fieldDofOffset(liquid_indicator_field);
        for (std::size_t i = 0; i < result.nodes; ++i) {
            if (system.constraints().isConstrained(
                    field_offset + static_cast<GlobalIndex>(i))) {
                result.diagnostic =
                    "P1 conservative phase graph does not accept constrained indicator nodes; boundary transport must enter through phase fluxes";
                return result;
            }
        }

        result.geometry_revision = mesh.geometryRevision();
        result.topology_revision = mesh.topologyRevision();
        result.ownership_revision = mesh.ownershipRevision();
        result.numbering_revision = mesh.numberingRevision();
        result.dof_layout_revision = dofs.dofLayoutRevision();
        result.minimum_jacobian_determinant =
            std::numeric_limits<Real>::infinity();

        std::map<std::pair<GlobalIndex, GlobalIndex>, MutableGradientEdge>
            assembled_edges;
        std::vector<Vector3> assembled_row_sum(result.nodes, Vector3{});
        Real maximum_gradient_coefficient{0.0};
        Real maximum_physical_basis_gradient{0.0};

        mesh.forEachOwnedCell([&](GlobalIndex cell) {
            if (record.space->polynomial_order(cell) != 1) {
                throw std::invalid_argument(
                    "P1 conservative phase graph found a non-P1 cell");
            }
            const auto cell_dofs = dofs.getCellDofs(cell);
            const auto& element = record.space->getElement(
                mesh.getCellType(cell), cell);
            const auto& basis = element.basis();
            if (basis.is_vector_valued() ||
                basis.basis_type() != BasisType::Lagrange ||
                basis.order() != 1 || basis.size() != cell_dofs.size() ||
                cell_dofs.size() <
                    static_cast<std::size_t>(result.dimension + 1)) {
                throw std::invalid_argument(
                    "P1 conservative phase graph requires one scalar linear Lagrange basis value per cell DOF");
            }
            for (const auto dof : cell_dofs) {
                if (dof < 0 ||
                    static_cast<std::size_t>(dof) >= result.nodes) {
                    throw std::invalid_argument(
                        "P1 conservative phase graph found a cell DOF outside the field layout");
                }
            }

            const auto mapping = makeCellMapping(mesh, cell);
            if (!mapping || mapping->dimension() != result.dimension) {
                throw std::invalid_argument(
                    "P1 conservative phase graph found an incompatible geometry mapping");
            }
            const int quadrature_order = resolvedQuadratureOrder(
                options.quadrature_order,
                mesh.getCellGeometryOrder(cell));
            result.maximum_quadrature_order = std::max(
                result.maximum_quadrature_order, quadrature_order);
            const auto quadrature = quadrature::QuadratureFactory::create(
                mesh.getCellType(cell), quadrature_order);
            if (!quadrature || quadrature->num_points() == 0u) {
                throw std::invalid_argument(
                    "P1 conservative phase graph received an empty quadrature rule");
            }

            const std::size_t local_size = cell_dofs.size();
            std::vector<Real> local_mass(local_size, Real{0.0});
            std::vector<Vector3> local_gradient(
                local_size * local_size, Vector3{});
            std::vector<Real> values;
            std::vector<basis::Gradient> reference_gradients;
            for (std::size_t q = 0; q < quadrature->num_points(); ++q) {
                const auto xi = quadrature->point(q);
                const Real determinant = mapping->jacobian_determinant(xi);
                const Real quadrature_weight = quadrature->weight(q);
                if (!std::isfinite(determinant) ||
                    !(determinant > Real{0.0}) ||
                    !std::isfinite(quadrature_weight) ||
                    !(quadrature_weight > Real{0.0})) {
                    throw std::invalid_argument(
                        "P1 conservative phase graph found a nonpositive or non-finite mapped quadrature weight");
                }
                result.minimum_jacobian_determinant = std::min(
                    result.minimum_jacobian_determinant, determinant);
                const Real weight = determinant * quadrature_weight;
                result.physical_measure += weight;

                basis.evaluate_values(xi, values);
                basis.evaluate_gradients(xi, reference_gradients);
                if (values.size() != local_size ||
                    reference_gradients.size() != local_size) {
                    throw std::invalid_argument(
                        "P1 conservative phase graph received inconsistent basis evaluations");
                }
                std::vector<math::Vector<Real, 3>> physical_gradients;
                physical_gradients.reserve(local_size);
                const auto inverse = mapping->jacobian_inverse(xi);
                Real value_sum{0.0};
                Vector3 gradient_sum{};
                for (std::size_t local = 0; local < local_size; ++local) {
                    if (!std::isfinite(values[local])) {
                        throw std::invalid_argument(
                            "P1 conservative phase graph found a non-finite basis value");
                    }
                    const auto physical_gradient =
                        mapping->transform_gradient(
                            reference_gradients[local], inverse);
                    for (int d = 0; d < result.dimension; ++d) {
                        if (!std::isfinite(
                                physical_gradient[static_cast<std::size_t>(d)])) {
                            throw std::invalid_argument(
                                "P1 conservative phase graph found a non-finite physical basis gradient");
                        }
                    }
                    physical_gradients.push_back(physical_gradient);
                    Vector3 physical_gradient_array{};
                    for (int d = 0; d < result.dimension; ++d) {
                        physical_gradient_array[static_cast<std::size_t>(d)] =
                            physical_gradient[static_cast<std::size_t>(d)];
                    }
                    maximum_physical_basis_gradient = std::max(
                        maximum_physical_basis_gradient,
                        norm(physical_gradient_array, result.dimension));
                    value_sum += values[local];
                    addScaled(gradient_sum, physical_gradient,
                              Real{1.0}, result.dimension);
                    local_mass[local] += weight * values[local];
                }
                result.maximum_partition_of_unity_residual = std::max(
                    result.maximum_partition_of_unity_residual,
                    std::abs(value_sum - Real{1.0}));
                result.maximum_gradient_partition_residual = std::max(
                    result.maximum_gradient_partition_residual,
                    norm(gradient_sum, result.dimension));

                for (std::size_t i = 0; i < local_size; ++i) {
                    for (std::size_t j = 0; j < local_size; ++j) {
                        addScaled(local_gradient[i * local_size + j],
                                  physical_gradients[j],
                                  weight * values[i], result.dimension);
                    }
                }
            }

            for (std::size_t i = 0; i < local_size; ++i) {
                const auto global_i = cell_dofs[i];
                const auto node_i = static_cast<std::size_t>(global_i);
                if (!std::isfinite(local_mass[i]) ||
                    !(local_mass[i] > Real{0.0})) {
                    throw std::invalid_argument(
                        "P1 conservative phase graph found a nonpositive local lumped control volume");
                }
                result.lumped_control_volume[node_i] += local_mass[i];
                for (std::size_t j = 0; j < local_size; ++j) {
                    const auto global_j = cell_dofs[j];
                    const auto& coefficient =
                        local_gradient[i * local_size + j];
                    add(assembled_row_sum[node_i], coefficient,
                        result.dimension);
                    add(result.boundary_column_sum[
                            static_cast<std::size_t>(global_j)],
                        coefficient, result.dimension);
                    maximum_gradient_coefficient = std::max(
                        maximum_gradient_coefficient,
                        norm(coefficient, result.dimension));
                    if (i == j) {
                        add(result.diagonal_gradient[node_i], coefficient,
                            result.dimension);
                        continue;
                    }
                    const auto endpoints = std::minmax(global_i, global_j);
                    auto& edge = assembled_edges[{endpoints.first,
                                                  endpoints.second}];
                    if (global_i == endpoints.first) {
                        add(edge.first_test_second_gradient, coefficient,
                            result.dimension);
                    } else {
                        add(edge.second_test_first_gradient, coefficient,
                            result.dimension);
                    }
                }
            }
            ++result.cells;
        });

        if (result.cells == 0u ||
            !std::isfinite(result.physical_measure) ||
            !(result.physical_measure > Real{0.0}) ||
            !std::isfinite(result.minimum_jacobian_determinant)) {
            result.diagnostic =
                "P1 conservative phase graph found no valid owned-cell measure";
            return result;
        }

        result.minimum_lumped_control_volume =
            std::numeric_limits<Real>::infinity();
        long double volume_sum = 0.0L;
        result.positive_control_volumes_satisfied = true;
        for (std::size_t node = 0; node < result.nodes; ++node) {
            const Real volume = result.lumped_control_volume[node];
            result.positive_control_volumes_satisfied =
                result.positive_control_volumes_satisfied &&
                std::isfinite(volume) && volume > Real{0.0};
            result.minimum_lumped_control_volume = std::min(
                result.minimum_lumped_control_volume, volume);
            volume_sum += static_cast<long double>(volume);
            result.maximum_gradient_row_sum_residual = std::max(
                result.maximum_gradient_row_sum_residual,
                norm(assembled_row_sum[node], result.dimension));
        }
        if (!result.positive_control_volumes_satisfied) {
            result.diagnostic =
                "P1 conservative phase graph left a node without positive owned-cell measure";
            return result;
        }
        result.total_lumped_control_volume = static_cast<Real>(volume_sum);
        result.measure_closure_residual =
            result.total_lumped_control_volume - result.physical_measure;

        const Real partition_tolerance =
            scaledTolerance(options.invariant_tolerance, {Real{1.0}});
        const Real gradient_partition_tolerance = scaledTolerance(
            options.invariant_tolerance,
            {maximum_physical_basis_gradient});
        const Real row_sum_tolerance = scaledTolerance(
            options.invariant_tolerance, {maximum_gradient_coefficient});
        const Real measure_tolerance = scaledTolerance(
            options.invariant_tolerance,
            {result.physical_measure, result.total_lumped_control_volume});
        result.partition_of_unity_satisfied =
            result.maximum_partition_of_unity_residual <=
            partition_tolerance;
        result.gradient_partition_satisfied =
            result.maximum_gradient_partition_residual <=
            gradient_partition_tolerance;
        result.gradient_row_sum_satisfied =
            result.maximum_gradient_row_sum_residual <= row_sum_tolerance;
        result.measure_closure_satisfied =
            std::abs(result.measure_closure_residual) <= measure_tolerance;
        if (!result.partition_of_unity_satisfied ||
            !result.gradient_partition_satisfied ||
            !result.gradient_row_sum_satisfied ||
            !result.measure_closure_satisfied) {
            result.diagnostic =
                "P1 conservative phase graph failed a partition, gradient, or measure identity";
            return result;
        }

        result.edges.reserve(assembled_edges.size());
        for (const auto& [endpoints, edge] : assembled_edges) {
            result.edges.push_back(LevelSetP1PhaseGradientEdge{
                .first_node = endpoints.first,
                .second_node = endpoints.second,
                .first_test_second_gradient =
                    edge.first_test_second_gradient,
                .second_test_first_gradient =
                    edge.second_test_first_gradient,
            });
        }
        result.success = true;
        result.diagnostic = "ok";
    } catch (const std::exception& exception) {
        result.success = false;
        result.diagnostic = exception.what();
    }
    return result;
}

LevelSetP1PhaseTransportStageResult
advanceLevelSetP1ConservativePhaseStage(
    const LevelSetP1PhaseTransportGraph& graph,
    std::span<const Real> previous_liquid_indicator,
    std::span<const Real> lower_liquid_indicator,
    std::span<const Real> upper_liquid_indicator,
    std::span<const std::array<Real, 3>> nodal_velocity,
    Real time_step,
    const LevelSetP1PhaseStageOptions& options)
{
    LevelSetP1PhaseTransportStageResult result;
    try {
        if (!graph.success) {
            result.diagnostic =
                "P1 conservative phase stage requires a valid assembled graph";
            return result;
        }
        const std::size_t node_count = graph.nodes;
        if (node_count == 0u ||
            graph.lumped_control_volume.size() != node_count ||
            graph.diagonal_gradient.size() != node_count ||
            graph.boundary_column_sum.size() != node_count ||
            previous_liquid_indicator.size() != node_count ||
            lower_liquid_indicator.size() != node_count ||
            upper_liquid_indicator.size() != node_count ||
            nodal_velocity.size() != node_count) {
            result.diagnostic =
                "P1 conservative phase stage received inconsistent nodal spans";
            return result;
        }
        if (!(time_step > Real{0.0}) || !std::isfinite(time_step) ||
            !std::isfinite(options.invariant_tolerance) ||
            options.invariant_tolerance < Real{0.0} ||
            !(options.maximum_courant > Real{0.0}) ||
            options.maximum_courant > Real{1.0} ||
            !std::isfinite(options.maximum_courant)) {
            result.diagnostic =
                "P1 conservative phase stage requires a positive finite time step, nonnegative finite tolerance, and Courant limit in (0,1]";
            return result;
        }

        result.nodal_courant.assign(node_count, Real{0.0});
        result.physical_boundary_mass_transfer.assign(
            node_count, Real{0.0});
        result.discrete_divergence_mass_source.assign(
            node_count, Real{0.0});
        std::vector<Real> row_divergence(node_count, Real{0.0});
        std::vector<Real> direct_strong_rate(node_count, Real{0.0});
        std::vector<Real> decomposed_strong_rate(node_count, Real{0.0});
        result.flux_edges.reserve(graph.edges.size());
        result.minimum_low_order_coefficient =
            std::numeric_limits<Real>::infinity();
        result.low_order_coefficients_nonnegative = true;

        for (std::size_t node = 0; node < node_count; ++node) {
            const auto& velocity = nodal_velocity[node];
            for (int d = 0; d < graph.dimension; ++d) {
                if (!std::isfinite(velocity[static_cast<std::size_t>(d)])) {
                    result.diagnostic =
                        "P1 conservative phase stage found a non-finite nodal velocity";
                    return result;
                }
            }
            const Real diagonal_velocity = dot(
                graph.diagonal_gradient[node], velocity, graph.dimension);
            row_divergence[node] = diagonal_velocity;
            direct_strong_rate[node] =
                -diagonal_velocity * previous_liquid_indicator[node];
        }

        for (const auto& edge : graph.edges) {
            if (edge.first_node < 0 || edge.second_node < 0 ||
                edge.first_node >= edge.second_node ||
                static_cast<std::size_t>(edge.second_node) >= node_count) {
                result.diagnostic =
                    "P1 conservative phase stage found a malformed graph edge";
                return result;
            }
            const auto first = static_cast<std::size_t>(edge.first_node);
            const auto second = static_cast<std::size_t>(edge.second_node);
            const Real first_to_second_speed = dot(
                edge.first_test_second_gradient,
                nodal_velocity[second], graph.dimension);
            const Real second_to_first_speed = dot(
                edge.second_test_first_gradient,
                nodal_velocity[first], graph.dimension);
            const Real diffusion = std::max(
                std::abs(first_to_second_speed),
                std::abs(second_to_first_speed));
            const Real first_coefficient =
                diffusion - first_to_second_speed;
            const Real second_coefficient =
                diffusion - second_to_first_speed;
            result.minimum_low_order_coefficient = std::min(
                result.minimum_low_order_coefficient,
                std::min(first_coefficient, second_coefficient));
            const Real coefficient_tolerance = scaledTolerance(
                options.invariant_tolerance,
                {diffusion, first_to_second_speed,
                 second_to_first_speed});
            result.low_order_coefficients_nonnegative =
                result.low_order_coefficients_nonnegative &&
                first_coefficient >= -coefficient_tolerance &&
                second_coefficient >= -coefficient_tolerance;

            result.nodal_courant[first] +=
                time_step * std::max(Real{0.0}, first_coefficient) /
                graph.lumped_control_volume[first];
            result.nodal_courant[second] +=
                time_step * std::max(Real{0.0}, second_coefficient) /
                graph.lumped_control_volume[second];

            const Real first_indicator = previous_liquid_indicator[first];
            const Real second_indicator = previous_liquid_indicator[second];
            const Real central_rate =
                second_to_first_speed * first_indicator -
                first_to_second_speed * second_indicator;
            const Real diffusive_rate =
                diffusion * (second_indicator - first_indicator);
            result.flux_edges.push_back(LevelSetPhaseFluxEdge{
                .first_node = edge.first_node,
                .second_node = edge.second_node,
                .low_order_mass_transfer =
                    time_step * (central_rate + diffusive_rate),
                .raw_antidiffusive_mass_transfer =
                    -time_step * diffusive_rate,
            });
            decomposed_strong_rate[first] += central_rate;
            decomposed_strong_rate[second] -= central_rate;

            row_divergence[first] += first_to_second_speed;
            row_divergence[second] += second_to_first_speed;
            direct_strong_rate[first] -=
                first_to_second_speed * second_indicator;
            direct_strong_rate[second] -=
                second_to_first_speed * first_indicator;
        }

        if (graph.edges.empty()) {
            result.minimum_low_order_coefficient = Real{0.0};
        }
        if (!result.low_order_coefficients_nonnegative) {
            result.diagnostic =
                "P1 conservative phase stage produced a negative low-order graph coefficient";
            return result;
        }

        result.maximum_courant = Real{0.0};
        Real maximum_integrated_strong_rate{0.0};
        for (std::size_t node = 0; node < node_count; ++node) {
            if (!std::isfinite(result.nodal_courant[node])) {
                result.diagnostic =
                    "P1 conservative phase stage produced a non-finite nodal Courant number";
                return result;
            }
            result.maximum_courant = std::max(
                result.maximum_courant, result.nodal_courant[node]);
            const Real indicator = previous_liquid_indicator[node];
            const Real boundary_rate =
                -indicator * dot(graph.boundary_column_sum[node],
                                 nodal_velocity[node], graph.dimension);
            const Real divergence_rate = indicator * row_divergence[node];
            result.physical_boundary_mass_transfer[node] =
                time_step * boundary_rate;
            result.discrete_divergence_mass_source[node] =
                time_step * divergence_rate;
            decomposed_strong_rate[node] +=
                boundary_rate + divergence_rate;
            direct_strong_rate[node] += divergence_rate;

            const Real residual = time_step *
                (decomposed_strong_rate[node] -
                 direct_strong_rate[node]);
            maximum_integrated_strong_rate = std::max(
                maximum_integrated_strong_rate,
                time_step * std::max(
                    std::abs(decomposed_strong_rate[node]),
                    std::abs(direct_strong_rate[node])));
            result.maximum_strong_form_decomposition_residual = std::max(
                result.maximum_strong_form_decomposition_residual,
                std::abs(residual));
        }

        const Real decomposition_tolerance = scaledTolerance(
            options.invariant_tolerance,
            {maximum_integrated_strong_rate});
        result.strong_form_decomposition_satisfied =
            result.maximum_strong_form_decomposition_residual <=
            decomposition_tolerance;
        if (!result.strong_form_decomposition_satisfied) {
            result.diagnostic =
                "P1 conservative phase stage failed the strong-CG edge/boundary decomposition identity";
            return result;
        }

        result.courant_satisfied =
            !options.enforce_courant_limit ||
            result.maximum_courant <=
                options.maximum_courant +
                    scaledTolerance(options.invariant_tolerance,
                                    {result.maximum_courant,
                                     options.maximum_courant});
        if (!result.courant_satisfied) {
            result.diagnostic =
                "P1 conservative phase stage rejected a time step outside its low-order Courant contract";
            return result;
        }

        result.correction =
            applyLevelSetConservativePhaseFluxCorrection(
                LevelSetPhaseFluxStageView{
                    .lumped_control_volume = graph.lumped_control_volume,
                    .previous_liquid_indicator =
                        previous_liquid_indicator,
                    .lower_liquid_indicator = lower_liquid_indicator,
                    .upper_liquid_indicator = upper_liquid_indicator,
                    .interior_edges = result.flux_edges,
                    .physical_boundary_mass_transfer =
                        result.physical_boundary_mass_transfer,
                    .discrete_divergence_mass_source =
                        result.discrete_divergence_mass_source,
                    .invariant_tolerance = options.invariant_tolerance,
                    .require_constant_preservation =
                        options.require_constant_preservation,
                });
        if (!result.correction.success) {
            result.diagnostic = result.correction.diagnostic;
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
