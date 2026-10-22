#include "LevelSet/LevelSetVolume.h"

#include "Dofs/EntityDofMap.h"
#include "Geometry/MappingFactory.h"
#include "Interfaces/LevelSetInterfaceBuilder.h"
#include "Quadrature/QuadratureFactory.h"
#include "Spaces/FunctionSpace.h"

#include <algorithm>
#include <array>
#include <cmath>
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

struct CollectiveContext {
#if FE_HAS_MPI
    MPI_Comm communicator{MPI_COMM_NULL};
#endif
    bool active{false};
};

[[nodiscard]] CollectiveContext collectiveContext(
    const dofs::DofHandler& dof_handler)
{
    CollectiveContext context;
#if FE_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
        MPI_Finalized(&finalized);
    }
    if (initialized != 0 && finalized == 0 &&
        dof_handler.mpiComm() != MPI_COMM_NULL) {
        int size = 1;
        MPI_Comm_size(dof_handler.mpiComm(), &size);
        context.communicator = dof_handler.mpiComm();
        context.active = size > 1;
    }
#else
    (void)dof_handler;
#endif
    return context;
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

[[nodiscard]] bool allReduceLogicalAnd(const CollectiveContext& context,
                                       bool local)
{
#if FE_HAS_MPI
    if (context.active) {
        const int local_value = local ? 1 : 0;
        int global_value = 0;
        MPI_Allreduce(&local_value,
                      &global_value,
                      1,
                      MPI_INT,
                      MPI_MIN,
                      context.communicator);
        return global_value != 0;
    }
#else
    (void)context;
#endif
    return local;
}

void globalizeVolumeResult(LevelSetVolumeResult& result,
                           const CollectiveContext& context)
{
    result.success = allReduceLogicalAnd(context, result.success);
#if FE_HAS_MPI
    if (context.active) {
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
                      context.communicator);
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
                      context.communicator);
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

[[nodiscard]] Real clampVolume(Real value, Real measure) noexcept
{
    if (value <= Real{0.0}) {
        return Real{0.0};
    }
    if (value >= measure) {
        return measure;
    }
    return value;
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

[[nodiscard]] Real parentMeasure(ElementType type,
                                 const std::vector<std::array<Real, 3>>& x)
{
    switch (type) {
    case ElementType::Triangle3:
    case ElementType::Triangle6:
        return triangleArea(x[0], x[1], x[2]);
    case ElementType::Quad4:
    case ElementType::Quad8:
    case ElementType::Quad9:
        return triangleArea(x[0], x[1], x[2]) + triangleArea(x[0], x[2], x[3]);
    case ElementType::Tetra4:
    case ElementType::Tetra10:
        return tetraVolume(x[0], x[1], x[2], x[3]);
    case ElementType::Hex8:
    case ElementType::Hex20:
    case ElementType::Hex27:
    case ElementType::Wedge6:
    case ElementType::Wedge15:
    case ElementType::Wedge18:
    case ElementType::Pyramid5:
    case ElementType::Pyramid13:
    case ElementType::Pyramid14: {
        Real volume = Real{0.0};
        for (const auto& tet : tetrahedralCornerDecomposition(type)) {
            volume += tetraVolume(x[tet[0]], x[tet[1]], x[tet[2]], x[tet[3]]);
        }
        return volume;
    }
    default:
        return Real{0.0};
    }
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

[[nodiscard]] Real negativeVolumeForLinearTetra(
    const CutInterfaceDomainRequest& request,
    GlobalIndex parent_cell,
    std::span<const std::array<Real, 3>, 4> coordinates,
    std::span<const Real, 4> signed_values,
    Real tolerance)
{
    const Real measure = tetraVolume(
        coordinates[0], coordinates[1], coordinates[2], coordinates[3]);
    const Real simple_fraction =
        negativeFractionFromValues(signed_values, tolerance);
    if (simple_fraction >= Real{0.0}) {
        return simple_fraction * measure;
    }

    auto clipped_volume = [&]() {
        std::array<std::size_t, 4> inside{};
        std::array<std::size_t, 4> outside{};
        std::size_t inside_count = 0u;
        std::size_t outside_count = 0u;
        for (std::size_t i = 0; i < signed_values.size(); ++i) {
            if (signed_values[i] <= tolerance) {
                inside[inside_count++] = i;
            } else {
                outside[outside_count++] = i;
            }
        }
        if (inside_count == 0u) {
            return Real{0.0};
        }
        if (outside_count == 0u) {
            return measure;
        }

        auto interface_point = [&](std::size_t inside_index,
                                   std::size_t outside_index) {
            const Real vi = signed_values[inside_index];
            const Real vj = signed_values[outside_index];
            if (std::abs(vi) <= tolerance) {
                return coordinates[inside_index];
            }
            if (std::abs(vj) <= tolerance) {
                return coordinates[outside_index];
            }
            const Real denom = vi - vj;
            Real t = denom != Real{0.0} ? vi / denom : Real{0.0};
            t = std::clamp(t, Real{0.0}, Real{1.0});
            return interpolatePoint(coordinates[inside_index],
                                    coordinates[outside_index],
                                    t);
        };

        if (inside_count == 1u) {
            const auto a = inside[0];
            const auto p0 = interface_point(a, outside[0]);
            const auto p1 = interface_point(a, outside[1]);
            const auto p2 = interface_point(a, outside[2]);
            return clampVolume(
                tetraVolume(coordinates[a], p0, p1, p2),
                measure);
        }
        if (inside_count == 3u) {
            const auto p = outside[0];
            const auto q0 = interface_point(inside[0], p);
            const auto q1 = interface_point(inside[1], p);
            const auto q2 = interface_point(inside[2], p);
            return clampVolume(
                measure - tetraVolume(coordinates[p], q0, q1, q2),
                measure);
        }

        const auto a = inside[0];
        const auto b = inside[1];
        const auto c = outside[0];
        const auto d = outside[1];
        const auto p_ac = interface_point(a, c);
        const auto p_ad = interface_point(a, d);
        const auto p_bc = interface_point(b, c);
        const auto p_bd = interface_point(b, d);
        const Real volume =
            tetraVolume(coordinates[a], coordinates[b], p_bd, p_bc) +
            tetraVolume(coordinates[a], p_ac, p_bc, p_bd) +
            tetraVolume(coordinates[a], p_ac, p_bd, p_ad);
        return clampVolume(volume, measure);
    };

    LevelSetCellCutInput input{};
    input.parent_cell = parent_cell;
    input.element_type = ElementType::Tetra4;
    input.node_coordinates.assign(coordinates.begin(), coordinates.end());
    input.level_set_values.reserve(signed_values.size());
    for (const auto value : signed_values) {
        input.level_set_values.push_back(value + request.isovalue);
    }

    auto cut_result = interfaces::cutLinearLevelSetCell3D(request, input);
    auto active = std::find_if(
        cut_result.fragments.begin(),
        cut_result.fragments.end(),
        [](const auto& fragment) { return fragment.active(); });
    if (active == cut_result.fragments.end()) {
        return clipped_volume();
    }
    return active->negative_volume_fraction * measure;
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

[[nodiscard]] std::shared_ptr<geometry::GeometryMapping>
makeGeneratedVolumeCellMapping(const assembly::IMeshAccess& mesh,
                               GlobalIndex cell)
{
    std::vector<std::array<Real, 3>> coordinates;
    mesh.getCellCoordinates(cell, coordinates);
    if (coordinates.empty()) {
        throw std::runtime_error(
            "generated level-set volume found a cell with no geometry coordinates");
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
    const auto generated =
        lifecycle.build(system, generated_options, solution);
    populateGeneratedVolumeDiagnostics(result, generated);
    const bool generated_success =
        allReduceLogicalAnd(collective, generated.success);
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
            mapping->second = makeGeneratedVolumeCellMapping(
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
    request.quadrature_order = 1;

    LevelSetVolumeResult result;
    result.success = true;

    std::vector<GlobalIndex> cell_nodes;
    std::vector<std::array<Real, 3>> cell_coordinates;
    mesh.forEachOwnedCell([&](GlobalIndex cell_id) {
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

        const auto measure = parentMeasure(type, cell_coordinates);
        result.total_volume += measure;
        ++result.cells;

        const Real simple_fraction =
            negativeFractionFromValues(signed_values, options.tolerance);
        Real negative_fraction = simple_fraction;
        if (simple_fraction < Real{0.0}) {
            if (mesh.dimension() == 2) {
                LevelSetCellCutInput input{};
                input.parent_cell = cell_id;
                input.element_type = type;
                input.node_coordinates.assign(
                    cell_coordinates.begin(),
                    cell_coordinates.begin() + static_cast<std::ptrdiff_t>(count));
                input.level_set_values.reserve(count);
                for (const auto value : signed_values) {
                    input.level_set_values.push_back(value + options.isovalue);
                }
                auto cut_result =
                    interfaces::cutLinearLevelSetCell2D(request, input);
                auto active = std::find_if(
                    cut_result.fragments.begin(),
                    cut_result.fragments.end(),
                    [](const auto& fragment) { return fragment.active(); });
                if (active == cut_result.fragments.end()) {
                    throw std::runtime_error(
                        "level-set volume calculation found a cut cell without an active interface fragment");
                }
                negative_fraction = active->negative_volume_fraction;
                ++result.cut_cells;
            } else if (mesh.dimension() == 3) {
                const auto tets = tetrahedralCornerDecomposition(type);
                if (tets.empty()) {
                    throw std::invalid_argument(
                        "level-set volume calculation encountered an unsupported 3D element type");
                }
                Real negative_volume = Real{0.0};
                for (const auto& tet : tets) {
                    const std::array<std::array<Real, 3>, 4> tet_coordinates{{
                        cell_coordinates[tet[0]],
                        cell_coordinates[tet[1]],
                        cell_coordinates[tet[2]],
                        cell_coordinates[tet[3]],
                    }};
                    const std::array<Real, 4> tet_values{{
                        signed_values[tet[0]],
                        signed_values[tet[1]],
                        signed_values[tet[2]],
                        signed_values[tet[3]],
                    }};
                    negative_volume += negativeVolumeForLinearTetra(
                        request,
                        cell_id,
                        std::span<const std::array<Real, 3>, 4>(
                            tet_coordinates.data(), tet_coordinates.size()),
                        std::span<const Real, 4>(
                            tet_values.data(), tet_values.size()),
                        options.tolerance);
                }
                negative_fraction =
                    measure > Real{0.0} ? negative_volume / measure : Real{0.0};
                ++result.cut_cells;
            } else {
                throw std::invalid_argument(
                    "level-set volume calculation requires a 2D or 3D mesh");
            }
        } else if (negative_fraction >= Real{1.0}) {
            ++result.full_negative_cells;
        } else {
            ++result.full_positive_cells;
        }

        const Real negative = negative_fraction * measure;
        result.negative_volume += negative;
        result.positive_volume += measure - negative;
    });

    globalizeVolumeResult(result, collective);
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
