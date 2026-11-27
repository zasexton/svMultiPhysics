#include "LevelSet/LevelSetVolume.h"

#include "Basis/NodeOrderingConventions.h"
#include "Dofs/EntityDofMap.h"
#include "Geometry/CutQuadratureMapping.h"
#include "Geometry/GeometryMapping.h"
#include "Interfaces/LevelSetInterfaceBuilder.h"
#include "Quadrature/QuadratureFactory.h"
#include "Spaces/FunctionSpace.h"
#include "Systems/SystemCollectives.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp::FE::level_set {
namespace {

using interfaces::CutInterfaceDomainRequest;
using interfaces::LevelSetCellCutInput;
using interfaces::LevelSetInterfaceSource;

[[nodiscard]] systems::CollectiveCommunicator collectiveContext(
    const dofs::DofHandler& dof_handler)
{
#if FE_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
        MPI_Finalized(&finalized);
    }
    if (initialized != 0 && finalized == 0 &&
        dof_handler.mpiComm() != MPI_COMM_NULL) {
        return systems::CollectiveCommunicator(dof_handler.mpiComm());
    }
#else
    (void)dof_handler;
#endif
    return {};
}

#if FE_HAS_MPI
[[nodiscard]] MPI_Datatype mpiRealType() noexcept
{
    if constexpr (std::is_same_v<Real, float>) {
        return MPI_FLOAT;
    }
    return MPI_DOUBLE;
}
#endif

void globalizeVolumeResult(LevelSetVolumeResult& result,
                           const systems::CollectiveCommunicator& context,
                           const std::exception_ptr& local_failure = {})
{
    result.success = systems::agreeOnLocalPhase(
        local_failure, result.success, context, "measure local cut-cell volumes", "LevelSetVolume");
#if FE_HAS_MPI
    if (context.isParallel()) {
        const std::array<unsigned long long, 4> local_counts{{
            static_cast<unsigned long long>(result.cells),
            static_cast<unsigned long long>(result.cut_cells),
            static_cast<unsigned long long>(result.full_negative_cells),
            static_cast<unsigned long long>(result.full_positive_cells),
        }};
        std::array<unsigned long long, 4> global_counts{};
        MPI_Allreduce(local_counts.data(),
                      global_counts.data(),
                      static_cast<int>(local_counts.size()),
                      MPI_UNSIGNED_LONG_LONG,
                      MPI_SUM,
                      context.native());
        result.cells = static_cast<std::size_t>(global_counts[0]);
        result.cut_cells = static_cast<std::size_t>(global_counts[1]);
        result.full_negative_cells =
            static_cast<std::size_t>(global_counts[2]);
        result.full_positive_cells =
            static_cast<std::size_t>(global_counts[3]);

        const std::array<Real, 3> local_volumes{{
            result.total_volume,
            result.negative_volume,
            result.positive_volume,
        }};
        std::array<Real, 3> global_volumes{};
        MPI_Allreduce(local_volumes.data(),
                      global_volumes.data(),
                      static_cast<int>(local_volumes.size()),
                      mpiRealType(),
                      MPI_SUM,
                      context.native());
        result.total_volume = global_volumes[0];
        result.negative_volume = global_volumes[1];
        result.positive_volume = global_volumes[2];
    }
#else
    (void)context;
#endif
}

[[nodiscard]] std::array<Real, 3> sub(const std::array<Real, 3>& a,
                                      const std::array<Real, 3>& b) noexcept
{
    return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
}

[[nodiscard]] std::array<Real, 3> add(const std::array<Real, 3>& a,
                                      const std::array<Real, 3>& b) noexcept
{
    return {{a[0] + b[0], a[1] + b[1], a[2] + b[2]}};
}

[[nodiscard]] std::array<Real, 3> scale(const std::array<Real, 3>& a,
                                        Real s) noexcept
{
    return {{a[0] * s, a[1] * s, a[2] * s}};
}

[[nodiscard]] std::array<Real, 3> cross(const std::array<Real, 3>& a,
                                        const std::array<Real, 3>& b) noexcept
{
    return {{a[1] * b[2] - a[2] * b[1],
             a[2] * b[0] - a[0] * b[2],
             a[0] * b[1] - a[1] * b[0]}};
}

[[nodiscard]] Real dot(const std::array<Real, 3>& a,
                       const std::array<Real, 3>& b) noexcept
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

[[nodiscard]] Real norm(const std::array<Real, 3>& a) noexcept
{
    return std::sqrt(dot(a, a));
}

[[nodiscard]] Real triangleArea(const std::array<Real, 3>& a,
                                const std::array<Real, 3>& b,
                                const std::array<Real, 3>& c) noexcept
{
    return Real{0.5} * norm(cross(sub(b, a), sub(c, a)));
}

[[nodiscard]] Real tetraVolume(const std::array<Real, 3>& a,
                               const std::array<Real, 3>& b,
                               const std::array<Real, 3>& c,
                               const std::array<Real, 3>& d) noexcept
{
    return std::abs(dot(sub(b, a), cross(sub(c, a), sub(d, a)))) / Real{6.0};
}

[[nodiscard]] std::array<Real, 3> interpolatePoint(
    const std::array<Real, 3>& a,
    const std::array<Real, 3>& b,
    Real t) noexcept
{
    return add(scale(a, Real{1.0} - t), scale(b, t));
}

[[nodiscard]] std::size_t cornerCount(ElementType type)
{
    switch (type) {
    case ElementType::Triangle3:
    case ElementType::Triangle6:
        return 3u;
    case ElementType::Quad4:
    case ElementType::Quad8:
    case ElementType::Quad9:
    case ElementType::Tetra4:
    case ElementType::Tetra10:
        return 4u;
    case ElementType::Hex8:
    case ElementType::Hex20:
    case ElementType::Hex27:
        return 8u;
    case ElementType::Wedge6:
    case ElementType::Wedge15:
    case ElementType::Wedge18:
        return 6u;
    case ElementType::Pyramid5:
    case ElementType::Pyramid13:
    case ElementType::Pyramid14:
        return 5u;
    default:
        return 0u;
    }
}

[[nodiscard]] std::vector<std::array<std::size_t, 4>>
tetrahedralCornerDecomposition(ElementType type)
{
    switch (type) {
    case ElementType::Tetra4:
    case ElementType::Tetra10:
        return {{{0u, 1u, 2u, 3u}}};
    case ElementType::Hex8:
    case ElementType::Hex20:
    case ElementType::Hex27:
        return {{{0u, 1u, 2u, 6u}},
                {{0u, 2u, 3u, 6u}},
                {{0u, 3u, 7u, 6u}},
                {{0u, 7u, 4u, 6u}},
                {{0u, 4u, 5u, 6u}},
                {{0u, 5u, 1u, 6u}}};
    case ElementType::Wedge6:
    case ElementType::Wedge15:
    case ElementType::Wedge18:
        return {{{0u, 1u, 2u, 5u}},
                {{0u, 1u, 5u, 4u}},
                {{0u, 4u, 5u, 3u}}};
    case ElementType::Pyramid5:
    case ElementType::Pyramid13:
    case ElementType::Pyramid14:
        return {{{0u, 1u, 2u, 4u}},
                {{0u, 2u, 3u, 4u}}};
    default:
        return {};
    }
}

using Point = std::array<Real, 3>;

// Integrate the parent map on an affine simplex in parent coordinates. Cutting
// in physical vertex coordinates would replace its mapped faces by chords.
[[nodiscard]] Real mappedSimplexMeasure(
    const geometry::GeometryMapping& mapping,
    const quadrature::QuadratureRule& rule,
    std::span<const Point> vertices)
{
    const auto dimension = static_cast<std::size_t>(rule.dimension());
    const auto& origin = vertices[0];
    std::array<Point, 3> edges{};
    for (std::size_t d = 0; d < dimension; ++d) {
        edges[d] = sub(vertices[d + 1], origin);
    }
    const auto normal = cross(edges[0], edges[1]);
    const Real reference_jacobian = dimension == 2
        ? norm(normal)
        : std::abs(dot(normal, edges[2]));
    if (reference_jacobian == Real{0}) {
        return Real{0};
    }

    Real integral{0};
    for (std::size_t q = 0; q < rule.num_points(); ++q) {
        Point parent = origin;
        for (std::size_t d = 0; d < dimension; ++d) {
            parent = add(parent, scale(edges[d], rule.point(q)[d]));
        }
        const auto jacobian = mapping.jacobian_determinant(
            {parent[0], parent[1], parent[2]});
        if (!std::isfinite(jacobian)) {
            throw std::runtime_error("level-set volume found a non-finite mapped Jacobian");
        }
        integral += rule.weight(q) * std::abs(jacobian);
    }
    return reference_jacobian * integral;
}

[[nodiscard]] Real coefficientAtVertex(const dofs::EntityDofMap& entity_map,
                                       GlobalIndex vertex,
                                       std::span<const Real> coefficients)
{
    const auto dofs = entity_map.getVertexDofs(vertex);
    if (dofs.size() != 1u) {
        throw std::invalid_argument(
            "level-set volume calculation requires one scalar DOF per mesh vertex");
    }
    const auto dof = dofs.front();
    if (dof < 0 || static_cast<std::size_t>(dof) >= coefficients.size()) {
        throw std::invalid_argument(
            "level-set volume calculation found a vertex DOF outside the coefficient span");
    }
    return coefficients[static_cast<std::size_t>(dof)];
}

[[nodiscard]] Real negativeFractionFromValues(std::span<const Real> values,
                                              Real tolerance)
{
    std::size_t negative_count = 0u;
    std::size_t positive_count = 0u;
    for (const auto value : values) {
        if (value < -tolerance) {
            ++negative_count;
        } else if (value > tolerance) {
            ++positive_count;
        }
    }
    if (positive_count == 0u) {
        return Real{1.0};
    }
    if (negative_count == 0u) {
        return Real{0.0};
    }
    return Real{-1.0};
}

struct PhaseVolumes {
    Real negative{};
    Real positive{};
};

[[nodiscard]] PhaseVolumes phaseVolumesForLinearTetra(
    const CutInterfaceDomainRequest& request,
    GlobalIndex parent_cell,
    std::span<const Point, 4> coordinates,
    std::span<const Real, 4> signed_values,
    Real tolerance,
    const geometry::GeometryMapping* mapping,
    const quadrature::QuadratureRule* quadrature_rule)
{
    const auto measure_tet = [&](const Point& a, const Point& b, const Point& c, const Point& d) {
        if (!mapping) {
            return tetraVolume(a, b, c, d);
        }
        const std::array<Point, 4> vertices{a, b, c, d};
        return mappedSimplexMeasure(*mapping, *quadrature_rule, vertices);
    };
    const Real measure = measure_tet(coordinates[0], coordinates[1], coordinates[2], coordinates[3]);
    const Real full_side = negativeFractionFromValues(signed_values, tolerance);
    if (full_side >= 0) {
        return {full_side * measure, (1 - full_side) * measure};
    }
    if (!mapping) {
        LevelSetCellCutInput input{};
        input.parent_cell = parent_cell;
        input.element_type = ElementType::Tetra4;
        input.node_coordinates.assign(coordinates.begin(), coordinates.end());
        for (const Real value : signed_values) {
            input.level_set_values.push_back(value + request.isovalue);
        }
        const auto cut = interfaces::cutLinearLevelSetCell3D(request, input);
        for (const auto& fragment : cut.fragments) {
            if (fragment.active()) {
                return {fragment.negative_volume_fraction * measure,
                        fragment.positive_volume_fraction * measure};
            }
        }
    }
    // Mapped tetrahedra and cuts with no active surface still have a volume
    // on each side. Integrate both clipped regions so a thin phase survives.
    std::array<std::size_t, 4> negative{}, positive{};
    std::size_t num_negative = 0, num_positive = 0;
    for (std::size_t i = 0; i < signed_values.size(); ++i) {
        if (signed_values[i] <= tolerance) {
            negative[num_negative++] = i;
        } else {
            positive[num_positive++] = i;
        }
    }
    const auto intersection = [&](std::size_t a, std::size_t b) {
        if (std::abs(signed_values[b]) < std::abs(signed_values[a])) {
            std::swap(a, b);
        }
        if (std::abs(signed_values[a]) <= tolerance) {
            return coordinates[a];
        }
        const Real t = interfaces::levelSetEdgeFraction(signed_values[a], signed_values[b]);
        return interpolatePoint(coordinates[a], coordinates[b], t);
    };
    const auto prism = [&](const std::array<Point, 3>& a, const std::array<Point, 3>& b) {
        return measure_tet(a[0], a[1], a[2], b[2]) + measure_tet(a[0], a[1], b[1], b[2]) +
               measure_tet(a[0], b[0], b[1], b[2]);
    };
    if (num_negative == 1 || num_positive == 1) {
        const auto lone = num_negative == 1 ? negative[0] : positive[0];
        const auto& others = num_negative == 1 ? positive : negative;
        std::array<Point, 3> roots{}, opposite{};
        for (std::size_t i = 0; i < roots.size(); ++i) {
            roots[i] = intersection(lone, others[i]);
            opposite[i] = coordinates[others[i]];
        }
        const Real corner = measure_tet(coordinates[lone], roots[0], roots[1], roots[2]);
        const Real rest = prism(opposite, roots);
        return num_negative == 1 ? PhaseVolumes{corner, rest} : PhaseVolumes{rest, corner};
    }
    const auto a = negative[0], b = negative[1], c = positive[0], d = positive[1];
    const auto ac = intersection(a, c), ad = intersection(a, d);
    const auto bc = intersection(b, c), bd = intersection(b, d);
    return {prism({coordinates[a], ac, ad}, {coordinates[b], bc, bd}),
            prism({coordinates[c], ac, bc}, {coordinates[d], ad, bd})};
}

[[nodiscard]] LevelSetGeneratedInterfaceOptions generatedInterfaceOptionsForVolume(
    const systems::FESystem& system,
    FieldId level_set_field,
    const LevelSetVolumeOptions& options)
{
    LevelSetGeneratedInterfaceOptions generated;
    generated.level_set_field_name =
        options.level_set_field_name.empty()
            ? system.fieldRecord(level_set_field).name
            : options.level_set_field_name;
    generated.domain_id = options.generated_domain_id.empty()
                              ? std::string{"volume_correction"}
                              : options.generated_domain_id;
    generated.requested_interface_marker = options.requested_interface_marker;
    generated.isovalue = options.isovalue;
    generated.tolerance = options.tolerance;
    generated.coefficient_classification_policy = options.coefficient_classification_policy;
    if (options.quadrature_order.has_value()) {
        generated.quadrature_order = *options.quadrature_order;
    }
    if (options.interface_quadrature_order.has_value()) {
        generated.interface_quadrature_order =
            *options.interface_quadrature_order;
    }
    if (options.volume_quadrature_order.has_value()) {
        generated.volume_quadrature_order = *options.volume_quadrature_order;
    }
    generated.geometry_mode = options.geometry_mode;
    generated.implicit_cut_quadrature_backend =
        options.implicit_cut_quadrature_backend;
    generated.implicit_cut_fallback_policy =
        options.implicit_cut_fallback_policy;
    generated.geometry_tangent_policy = options.geometry_tangent_policy;
    generated.implicit_cut_root_tolerance =
        options.implicit_cut_root_tolerance;
    generated.implicit_cut_root_coordinate_tolerance =
        options.implicit_cut_root_coordinate_tolerance;
    generated.implicit_cut_root_max_iterations =
        options.implicit_cut_root_max_iterations;
    generated.implicit_cut_max_subdivision_depth =
        options.implicit_cut_max_subdivision_depth;
    generated.affected_cell_neighborhood_layers =
        options.affected_cell_neighborhood_layers;
    generated.allow_corner_linearized_geometry =
        options.allow_corner_linearized_geometry;
    generated.require_production_qualified_implicit_cut_backend =
        options.require_production_qualified_implicit_cut_backend;
    return generated;
}

void populateGeneratedVolumeDiagnostics(
    LevelSetVolumeResult& result,
    const LevelSetGeneratedInterfaceResult& generated)
{
    result.generated_value_revision = generated.value_revision;
    result.generated_cell_cache_hits = generated.cell_cache_hits;
    result.generated_cell_cache_misses = generated.cell_cache_misses;
    result.generated_cell_cache_unchanged_dof_hits =
        generated.cell_cache_unchanged_dof_hits;
    result.generated_cell_refresh_candidate_count =
        generated.cell_refresh_candidate_count;
    result.generated_directly_affected_cell_count =
        generated.directly_affected_cell_count;
    result.generated_affected_cell_neighborhood_count =
        generated.affected_cell_neighborhood_count;
    result.generated_domain_cache_hits = generated.domain_cache_hits;
    result.generated_linear_full_cell_fast_path_count =
        generated.linear_full_cell_fast_path_count;
    result.generated_backend_elapsed_seconds =
        generated.backend_elapsed_seconds;
}

[[nodiscard]] bool isFullGeneratedVolumeRule(
    const geometry::CutQuadratureRule& rule) noexcept
{
    if (rule.kind != geometry::CutQuadratureKind::Volume ||
        !rule.full_cell_equivalent || !std::isfinite(rule.measure) ||
        !std::isfinite(rule.parent_measure) ||
        !std::isfinite(rule.volume_fraction) ||
        !(rule.parent_measure > Real{0.0})) {
        return false;
    }
    const Real tolerance = Real{128.0} *
                           std::numeric_limits<Real>::epsilon();
    return std::abs(rule.volume_fraction - Real{1.0}) <= tolerance &&
           std::abs(rule.measure - rule.parent_measure) <=
               tolerance * std::max(Real{1.0}, std::abs(rule.parent_measure));
}

[[nodiscard]] Real physicalGeneratedVolumeRuleMeasure(
    const assembly::IMeshAccess& mesh,
    const geometry::CutQuadratureRule& rule,
    geometry::GeometryMapping& mapping)
{
    if (rule.kind != geometry::CutQuadratureKind::Volume) {
        throw std::invalid_argument(
            "generated level-set volume requires a volume quadrature rule");
    }
    if (rule.frame == geometry::CutGeometryFrame::Current) {
        if (!std::isfinite(rule.measure) || rule.measure < Real{0.0}) {
            throw std::runtime_error(
                "generated level-set volume received an invalid current-frame measure");
        }
        return rule.measure;
    }

    Real measure{0.0};
    if (isFullGeneratedVolumeRule(rule)) {
        const auto cell = static_cast<GlobalIndex>(
            rule.provenance.parent_entity);
        const int geometry_order =
            std::max(1, mesh.getCellGeometryOrder(cell));
        const auto full_rule = quadrature::QuadratureFactory::create(
            mesh.getCellType(cell), std::max(2, 2 * geometry_order));
        for (std::size_t q = 0; q < full_rule->num_points(); ++q) {
            const auto xi = full_rule->point(q);
            const Real determinant = mapping.jacobian_determinant(xi);
            const Real weight = full_rule->weight(q);
            if (!std::isfinite(determinant) || !std::isfinite(weight)) {
                throw std::runtime_error(
                    "generated level-set volume encountered a non-finite full-cell mapping");
            }
            measure += weight * std::abs(determinant);
        }
        return measure;
    }

    if (rule.points.empty()) {
        throw std::runtime_error(
            "generated level-set volume received an empty partial rule");
    }
    for (const auto& point : rule.points) {
        const math::Vector<Real, 3> xi{
            point.point[0], point.point[1], point.point[2]};
        const Real determinant = mapping.jacobian_determinant(xi);
        if (!std::isfinite(determinant) || !std::isfinite(point.weight) ||
            point.weight < Real{0.0}) {
            throw std::runtime_error(
                "generated level-set volume encountered an invalid pointwise mapping");
        }
        measure += point.weight * std::abs(determinant);
    }
    return measure;
}

[[nodiscard]] LevelSetVolumeResult computeGeneratedInterfaceLevelSetVolume(
    const systems::FESystem& system,
    FieldId level_set_field,
    const LevelSetVolumeOptions& options,
    std::span<const Real> solution,
    LevelSetGeneratedInterfaceLifecycle& lifecycle)
{
    LevelSetVolumeResult result;
    const auto& field_dofs = system.fieldDofHandler(level_set_field);
    const auto collective = collectiveContext(field_dofs);
    const auto n_field_dofs = static_cast<std::size_t>(field_dofs.getNumDofs());
    const auto offset = static_cast<std::size_t>(system.fieldDofOffset(level_set_field));
    if (offset + n_field_dofs > solution.size()) {
        throw std::invalid_argument(
            "level-set volume calculation received an incompatible system solution span");
    }

    const auto generated_options =
        generatedInterfaceOptionsForVolume(system, level_set_field, options);
    LevelSetGeneratedInterfaceResult generated;
    std::exception_ptr local_failure;
    try {
        generated = lifecycle.build(system, generated_options, solution);
    } catch (...) {
        local_failure = std::current_exception();
    }
    const bool generated_success = systems::agreeOnLocalPhase(
        local_failure, generated.success, collective, "construct generated cut-cell volumes", "LevelSetVolume");
    populateGeneratedVolumeDiagnostics(result, generated);
    if (!generated_success) {
        result.success = false;
        result.diagnostic = generated.success
                                ? "generated-interface volume construction failed on another MPI rank"
                                : generated.diagnostic;
        globalizeVolumeResult(result, collective);
        return result;
    }

    result.success = true;
    system.meshAccess().forEachOwnedCell(
        [&](GlobalIndex /*cell*/) { ++result.cells; });
    result.diagnostic = "generated_interface_quadrature";

    std::set<GlobalIndex> negative_cells;
    std::set<GlobalIndex> positive_cells;
    std::set<GlobalIndex> cut_cells;
    std::unordered_map<GlobalIndex,
                       std::shared_ptr<geometry::GeometryMapping>>
        mapping_by_cell;
    for (const auto& fragment : generated.domain.fragments()) {
        if (fragment.active() &&
            system.meshAccess().isOwnedCell(fragment.parent_cell)) {
            cut_cells.insert(fragment.parent_cell);
        }
    }
    for (const auto& rule : generated.domain.volumeQuadratureRules()) {
        if (rule.kind != geometry::CutQuadratureKind::Volume) {
            continue;
        }
        const auto parent_cell = static_cast<GlobalIndex>(
            rule.provenance.parent_entity);
        if (!system.meshAccess().isOwnedCell(parent_cell)) {
            continue;
        }
        auto [mapping, inserted] = mapping_by_cell.try_emplace(
            parent_cell, nullptr);
        if (inserted) {
            mapping->second = geometry::makeCutCellGeometryMapping(
                system.meshAccess(), parent_cell);
        }
        const Real physical_measure = physicalGeneratedVolumeRuleMeasure(
            system.meshAccess(), rule, *mapping->second);
        result.total_volume += physical_measure;
        if (rule.side == geometry::CutIntegrationSide::Negative) {
            negative_cells.insert(parent_cell);
            result.negative_volume += physical_measure;
        } else if (rule.side == geometry::CutIntegrationSide::Positive) {
            positive_cells.insert(parent_cell);
            result.positive_volume += physical_measure;
        }
    }
    for (const auto cell : negative_cells) {
        if (positive_cells.find(cell) != positive_cells.end()) {
            cut_cells.insert(cell);
        }
    }
    result.cut_cells = cut_cells.size();
    for (const auto cell : negative_cells) {
        if (cut_cells.find(cell) == cut_cells.end()) {
            ++result.full_negative_cells;
        }
    }
    for (const auto cell : positive_cells) {
        if (cut_cells.find(cell) == cut_cells.end()) {
            ++result.full_positive_cells;
        }
    }
    globalizeVolumeResult(result, collective);
    return result;
}

[[nodiscard]] LevelSetVolumeResult computeGeneratedInterfaceLevelSetVolume(
    const systems::FESystem& system,
    FieldId level_set_field,
    const LevelSetVolumeOptions& options,
    std::span<const Real> solution)
{
    LevelSetGeneratedInterfaceLifecycle lifecycle;
    return computeGeneratedInterfaceLevelSetVolume(
        system,
        level_set_field,
        options,
        solution,
        lifecycle);
}

} // namespace

LevelSetVolumeResult computeLevelSetCutCellVolume(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    const LevelSetVolumeOptions& options,
    std::span<const Real> coefficients)
{
    if (options.use_generated_interface_quadrature) {
        throw std::invalid_argument(
            "generated-interface level-set volume calculation requires the FESystem overload");
    }
    if (!(options.tolerance > 0.0) || !std::isfinite(options.tolerance) ||
        !std::isfinite(options.isovalue)) {
        throw std::invalid_argument(
            "level-set volume calculation requires a finite isovalue and positive finite tolerance");
    }
    if (!interfaces::validLevelSetCoefficientClassificationPolicy(options.coefficient_classification_policy)) {
        throw std::invalid_argument("level-set volume calculation requires a valid coefficient classification policy");
    }
    const auto expected = static_cast<std::size_t>(level_set_dofs.getNumDofs());
    if (coefficients.size() != expected) {
        throw std::invalid_argument(
            "level-set volume calculation received an incompatible coefficient span");
    }
    const auto* entity_map = level_set_dofs.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::invalid_argument("level-set volume calculation requires a scalar nodal field");
    }
    if (entity_map->numVertices() != mesh.numVertices()) {
        throw std::invalid_argument(
            "level-set volume calculation requires field and mesh vertex counts to match");
    }
    const auto collective = collectiveContext(level_set_dofs);

    CutInterfaceDomainRequest request{};
    request.source = LevelSetInterfaceSource::fromField(FieldId{0});
    request.interface_marker = 0;
    request.isovalue = options.isovalue;
    request.tolerance = options.tolerance;
    request.coefficient_classification_policy = options.coefficient_classification_policy;
    const Real coefficient_band = request.resolvedCoefficientClassificationBand();
    request.quadrature_order = 1;

    LevelSetVolumeResult result;
    result.success = true;

    std::vector<GlobalIndex> cell_nodes;
    std::vector<std::array<Real, 3>> cell_coordinates;
    const auto measure_cell = [&](GlobalIndex cell_id) {
        const auto type = mesh.getCellType(cell_id);
        const std::size_t count = cornerCount(type);
        if (count == 0u) {
            throw std::invalid_argument(
                "level-set volume calculation encountered an unsupported element type");
        }

        mesh.getCellNodes(cell_id, cell_nodes);
        mesh.getCellCoordinates(cell_id, cell_coordinates);
        if (cell_nodes.size() < count || cell_coordinates.size() < count) {
            throw std::invalid_argument(
                "level-set volume calculation found incomplete cell geometry");
        }

        std::vector<Real> signed_values;
        signed_values.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            signed_values.push_back(
                coefficientAtVertex(*entity_map, cell_nodes[i], coefficients) -
                options.isovalue);
        }

        // Affine simplices have a constant metric. Tensor, wedge, pyramid,
        // and curved parents need their own pointwise geometry map.
        const int dimension = mesh.dimension();
        const int geometry_order = std::max(1, mesh.getCellGeometryOrder(cell_id));
        const bool affine_triangle = count == 3 && geometry_order == 1;
        const bool affine_tetra = dimension == 3 && count == 4 && geometry_order == 1;
        Real measure;
        if (affine_triangle) {
            measure = triangleArea(cell_coordinates[0], cell_coordinates[1], cell_coordinates[2]);
        } else if (affine_tetra) {
            measure = tetraVolume(cell_coordinates[0], cell_coordinates[1],
                                  cell_coordinates[2], cell_coordinates[3]);
        } else {
            measure = geometry::physicalCellMeasureFromMapping(
                mesh, cell_id, dimension * geometry_order);
        }
        result.total_volume += measure;
        ++result.cells;

        const Real simple_fraction =
            negativeFractionFromValues(signed_values, coefficient_band);
        PhaseVolumes phase{};
        if (simple_fraction < Real{0.0}) {
            const auto mapping = affine_triangle || affine_tetra
                ? nullptr : geometry::makeCutCellGeometryMapping(mesh, cell_id);
            const auto simplex_type = dimension == 2 ? ElementType::Triangle3 : ElementType::Tetra4;
            // On a simplex subregion a tensor-product determinant can attain
            // total degree d*(d*p-1). On a simplex parent it is d*(p-1).
            const int mapped_order = count == static_cast<std::size_t>(dimension + 1)
                ? dimension * (geometry_order - 1)
                : dimension * (dimension * geometry_order - 1);
            const auto simplex_rule = mapping
                ? quadrature::QuadratureFactory::create_pinned(simplex_type, mapped_order)
                : nullptr;
            std::vector<Point> reference_corners;
            if (!affine_tetra) {
                reference_corners.reserve(count);
                for (std::size_t i = 0; i < count; ++i) {
                    const auto xi = basis::ReferenceNodeLayout::get_node_coords(type, i);
                    reference_corners.push_back({xi[0], xi[1], xi[2]});
                }
            }
            if (mesh.dimension() == 2) {
                LevelSetCellCutInput input{};
                input.parent_cell = cell_id;
                input.element_type = type;
                input.node_coordinates = reference_corners;
                input.level_set_values.reserve(count);
                for (const auto value : signed_values) {
                    input.level_set_values.push_back(value + options.isovalue);
                }
                auto cut_result =
                    interfaces::cutLinearLevelSetCell2D(request, input);
                if (!cut_result.supported) {
                    throw std::invalid_argument("level-set volume in cell " + std::to_string(cell_id) +
                                                ": " + cut_result.diagnostic);
                }
                for (const auto& region : cut_result.volume_regions) {
                    if (!region.active()) {
                        continue;
                    }
                    auto& side_measure = region.side == geometry::CutIntegrationSide::Negative
                        ? phase.negative : phase.positive;
                    if (!mapping || region.full_cell_equivalent) {
                        side_measure += measure * region.volume_fraction;
                        continue;
                    }
                    for (const auto& triangle : region.reference_subcells) {
                        side_measure += triangle.measure_scale *
                            mappedSimplexMeasure(*mapping, *simplex_rule,
                                std::span(triangle.vertices).first(triangle.vertex_count));
                    }
                }
                ++result.cut_cells;
            } else if (mesh.dimension() == 3) {
                const auto tets = tetrahedralCornerDecomposition(type);
                if (tets.empty()) {
                    throw std::invalid_argument(
                        "level-set volume calculation encountered an unsupported 3D element type");
                }
                const auto& corners = affine_tetra ? cell_coordinates : reference_corners;
                for (const auto& tet : tets) {
                    const std::array<std::array<Real, 3>, 4> tet_coordinates{{
                        corners[tet[0]],
                        corners[tet[1]],
                        corners[tet[2]],
                        corners[tet[3]],
                    }};
                    const std::array<Real, 4> tet_values{{
                        signed_values[tet[0]],
                        signed_values[tet[1]],
                        signed_values[tet[2]],
                        signed_values[tet[3]],
                    }};
                    const auto clipped = phaseVolumesForLinearTetra(
                        request,
                        cell_id,
                        tet_coordinates,
                        tet_values,
                        coefficient_band, mapping.get(), simplex_rule.get());
                    phase.negative += clipped.negative;
                    phase.positive += clipped.positive;
                }
                ++result.cut_cells;
            } else {
                throw std::invalid_argument(
                    "level-set volume calculation requires a 2D or 3D mesh");
            }
        } else if (simple_fraction >= Real{1.0}) {
            phase.negative = measure;
            ++result.full_negative_cells;
        } else {
            phase.positive = measure;
            ++result.full_positive_cells;
        }

        result.negative_volume += phase.negative;
        result.positive_volume += phase.positive;
    };

    std::exception_ptr local_failure;
    try {
        mesh.forEachOwnedCell(measure_cell);
    } catch (...) {
        local_failure = std::current_exception();
    }
    globalizeVolumeResult(result, collective, local_failure);
    return result;
}

LevelSetVolumeResult computeLevelSetCutCellVolume(
    const systems::FESystem& system,
    FieldId level_set_field,
    const LevelSetVolumeOptions& options,
    std::span<const Real> solution)
{
    if (options.use_generated_interface_quadrature) {
        return computeGeneratedInterfaceLevelSetVolume(
            system,
            level_set_field,
            options,
            solution);
    }

    const auto& field_dofs = system.fieldDofHandler(level_set_field);
    const auto n_field_dofs = static_cast<std::size_t>(field_dofs.getNumDofs());
    const auto offset = static_cast<std::size_t>(system.fieldDofOffset(level_set_field));
    if (offset + n_field_dofs > solution.size()) {
        throw std::invalid_argument(
            "level-set volume calculation received an incompatible system solution span");
    }

    return computeLevelSetCutCellVolume(
        system.meshAccess(),
        field_dofs,
        options,
        solution.subspan(offset, n_field_dofs));
}

} // namespace svmp::FE::level_set
