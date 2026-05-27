#include "LevelSet/LevelSetVolume.h"

#include "Dofs/EntityDofMap.h"
#include "Interfaces/LevelSetInterfaceBuilder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp::FE::level_set {
namespace {

using interfaces::CutInterfaceDomainRequest;
using interfaces::LevelSetCellCutInput;
using interfaces::LevelSetInterfaceSource;

[[nodiscard]] std::array<Real, 3> sub(const std::array<Real, 3>& a,
                                      const std::array<Real, 3>& b) noexcept
{
    return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
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

[[nodiscard]] std::pair<Real, Real> vertexCoefficientRange(
    const assembly::IMeshAccess& mesh,
    const dofs::EntityDofMap& entity_map,
    std::span<const Real> coefficients)
{
    Real min_value = std::numeric_limits<Real>::infinity();
    Real max_value = -std::numeric_limits<Real>::infinity();
    for (GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        const Real value = coefficientAtVertex(entity_map, vertex, coefficients);
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
    }
    if (!std::isfinite(min_value) || !std::isfinite(max_value)) {
        throw std::invalid_argument("level-set volume correction requires finite coefficients");
    }
    return {min_value, max_value};
}

[[nodiscard]] std::vector<Real> shiftedCoefficients(std::span<const Real> coefficients,
                                                    Real shift)
{
    std::vector<Real> shifted(coefficients.begin(), coefficients.end());
    for (auto& value : shifted) {
        value += shift;
    }
    return shifted;
}

[[nodiscard]] std::pair<Real, Real> coefficientRange(std::span<const Real> coefficients)
{
    Real min_value = std::numeric_limits<Real>::infinity();
    Real max_value = -std::numeric_limits<Real>::infinity();
    for (const auto value : coefficients) {
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
    }
    if (!std::isfinite(min_value) || !std::isfinite(max_value)) {
        throw std::invalid_argument("level-set volume correction requires finite coefficients");
    }
    return {min_value, max_value};
}

[[nodiscard]] std::vector<Real> shiftedSystemSolution(
    std::span<const Real> solution,
    std::size_t offset,
    std::size_t field_dof_count,
    Real shift)
{
    std::vector<Real> shifted(solution.begin(), solution.end());
    for (std::size_t i = 0; i < field_dof_count; ++i) {
        shifted[offset + i] += shift;
    }
    return shifted;
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
        throw std::runtime_error(
            "level-set volume calculation found a tetrahedral subcell without an active interface fragment");
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

void accumulateGeneratedVolumeDiagnostics(
    LevelSetGlobalShiftCorrectionResult& correction,
    const LevelSetVolumeResult& volume)
{
    ++correction.generated_volume_measurement_count;
    correction.generated_cell_cache_hits += volume.generated_cell_cache_hits;
    correction.generated_cell_cache_misses += volume.generated_cell_cache_misses;
    correction.generated_cell_cache_unchanged_dof_hits +=
        volume.generated_cell_cache_unchanged_dof_hits;
    correction.generated_cell_refresh_candidate_count +=
        volume.generated_cell_refresh_candidate_count;
    correction.generated_directly_affected_cell_count +=
        volume.generated_directly_affected_cell_count;
    correction.generated_affected_cell_neighborhood_count +=
        volume.generated_affected_cell_neighborhood_count;
    correction.generated_domain_cache_hits += volume.generated_domain_cache_hits;
    correction.generated_linear_full_cell_fast_path_count +=
        volume.generated_linear_full_cell_fast_path_count;
    correction.generated_backend_elapsed_seconds +=
        volume.generated_backend_elapsed_seconds;
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
    if (!generated.success) {
        result.success = false;
        result.diagnostic = generated.diagnostic;
        return result;
    }

    result.success = true;
    result.cells = generated.cell_count;
    result.diagnostic = "generated_interface_quadrature";

    std::set<GlobalIndex> negative_cells;
    std::set<GlobalIndex> positive_cells;
    std::set<GlobalIndex> cut_cells;
    std::unordered_map<GlobalIndex, Real> physical_parent_scale_by_cell;
    std::vector<std::array<Real, 3>> coordinates;
    auto physical_parent_scale =
        [&](GlobalIndex cell, Real reference_parent_measure) {
        const auto [it, inserted] =
            physical_parent_scale_by_cell.try_emplace(cell, Real{1.0});
        if (inserted) {
            coordinates.clear();
            system.meshAccess().getCellCoordinates(cell, coordinates);
            const auto physical_parent_measure =
                parentMeasure(system.meshAccess().getCellType(cell),
                              coordinates);
            it->second =
                reference_parent_measure > Real{0.0} &&
                        physical_parent_measure > Real{0.0}
                    ? physical_parent_measure / reference_parent_measure
                    : Real{1.0};
        }
        return it->second;
    };
    for (const auto& fragment : generated.domain.fragments()) {
        if (fragment.active()) {
            cut_cells.insert(fragment.parent_cell);
        }
    }
    for (const auto& region : generated.domain.volumeRegions()) {
        if (!region.active()) {
            continue;
        }
        const auto parent_cell = static_cast<GlobalIndex>(region.parent_cell);
        const Real scale =
            physical_parent_scale(parent_cell, region.parent_measure);
        const Real physical_measure = region.measure * scale;
        result.total_volume += physical_measure;
        if (region.side == geometry::CutIntegrationSide::Negative) {
            negative_cells.insert(parent_cell);
            result.negative_volume += physical_measure;
        } else if (region.side == geometry::CutIntegrationSide::Positive) {
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
    if (!(options.tolerance > 0.0)) {
        throw std::invalid_argument("level-set volume calculation requires a positive tolerance");
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
    mesh.forEachCell([&](GlobalIndex cell_id) {
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

LevelSetGlobalShiftCorrectionResult applyGlobalLevelSetShiftCorrection(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    const LevelSetVolumeOptions& volume_options,
    const LevelSetGlobalShiftCorrectionOptions& correction_options,
    std::span<const Real> coefficients,
    std::vector<Real>& corrected_coefficients)
{
    if (!(correction_options.volume_tolerance > 0.0)) {
        throw std::invalid_argument("level-set global shift correction requires a positive volume tolerance");
    }
    if (correction_options.max_iterations <= 0) {
        throw std::invalid_argument("level-set global shift correction requires positive max_iterations");
    }

    auto initial = computeLevelSetCutCellVolume(
        mesh,
        level_set_dofs,
        volume_options,
        coefficients);
    const Real target = correction_options.target_negative_volume;
    if (target < -correction_options.volume_tolerance ||
        target > initial.total_volume + correction_options.volume_tolerance) {
        throw std::invalid_argument(
            "level-set global shift correction target volume is outside the total volume range");
    }

    LevelSetGlobalShiftCorrectionResult result;
    result.target_negative_volume = target;
    result.initial_negative_volume = initial.negative_volume;
    result.initial_volume = initial;
    result.corrected_volume = initial;
    result.corrected_negative_volume = initial.negative_volume;
    result.volume_error = initial.negative_volume - target;
    corrected_coefficients.assign(coefficients.begin(), coefficients.end());
    if (std::abs(result.volume_error) <= correction_options.volume_tolerance) {
        result.success = true;
        return result;
    }

    const auto* entity_map = level_set_dofs.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::invalid_argument("level-set global shift correction requires a scalar nodal field");
    }
    const auto [min_coeff, max_coeff] =
        vertexCoefficientRange(mesh, *entity_map, coefficients);
    const Real pad = std::max(volume_options.tolerance * Real{10.0},
                              Real{1.0e-12});
    Real lower = volume_options.isovalue - max_coeff - pad;
    Real upper = volume_options.isovalue - min_coeff + pad;
    if (!(lower < upper)) {
        lower -= Real{1.0};
        upper += Real{1.0};
    }

    Real best_shift = 0.0;
    Real best_error = std::abs(result.volume_error);
    LevelSetVolumeResult best_volume = initial;
    std::vector<Real> best_coefficients(coefficients.begin(), coefficients.end());

    for (int iter = 1; iter <= correction_options.max_iterations; ++iter) {
        const Real shift = Real{0.5} * (lower + upper);
        auto shifted = shiftedCoefficients(coefficients, shift);
        auto volume = computeLevelSetCutCellVolume(
            mesh,
            level_set_dofs,
            volume_options,
            shifted);
        const Real signed_error = volume.negative_volume - target;
        const Real abs_error = std::abs(signed_error);
        if (abs_error < best_error) {
            best_error = abs_error;
            best_shift = shift;
            best_volume = volume;
            best_coefficients = std::move(shifted);
        }

        result.iterations = iter;
        if (abs_error <= correction_options.volume_tolerance) {
            result.success = true;
            result.applied_shift = shift;
            result.corrected_negative_volume = volume.negative_volume;
            result.volume_error = signed_error;
            result.corrected_volume = volume;
            corrected_coefficients = std::move(best_coefficients);
            return result;
        }

        if (signed_error > 0.0) {
            lower = shift;
        } else {
            upper = shift;
        }
    }

    result.success = best_error <= correction_options.volume_tolerance;
    result.applied_shift = best_shift;
    result.corrected_negative_volume = best_volume.negative_volume;
    result.volume_error = best_volume.negative_volume - target;
    result.corrected_volume = best_volume;
    result.diagnostic = result.success
                            ? std::string{}
                            : "level-set global shift correction did not reach the requested volume tolerance";
    corrected_coefficients = std::move(best_coefficients);
    return result;
}

LevelSetGlobalShiftCorrectionResult applyGlobalLevelSetShiftCorrection(
    const systems::FESystem& system,
    FieldId level_set_field,
    const LevelSetVolumeOptions& volume_options,
    const LevelSetGlobalShiftCorrectionOptions& correction_options,
    std::span<const Real> solution,
    std::vector<Real>& corrected_solution)
{
    if (volume_options.use_generated_interface_quadrature) {
        if (!(correction_options.volume_tolerance > 0.0)) {
            throw std::invalid_argument(
                "level-set global shift correction requires a positive volume tolerance");
        }
        if (correction_options.max_iterations <= 0) {
            throw std::invalid_argument(
                "level-set global shift correction requires positive max_iterations");
        }

        const auto& field_dofs = system.fieldDofHandler(level_set_field);
        const auto n_field_dofs =
            static_cast<std::size_t>(field_dofs.getNumDofs());
        const auto offset =
            static_cast<std::size_t>(system.fieldDofOffset(level_set_field));
        if (offset + n_field_dofs > solution.size()) {
            throw std::invalid_argument(
                "level-set global shift correction received an incompatible system solution span");
        }

        LevelSetGeneratedInterfaceLifecycle volume_lifecycle;
        auto initial = computeGeneratedInterfaceLevelSetVolume(
            system,
            level_set_field,
            volume_options,
            solution,
            volume_lifecycle);
        if (!initial.success) {
            LevelSetGlobalShiftCorrectionResult result;
            result.success = false;
            result.initial_volume = initial;
            result.corrected_volume = initial;
            result.target_negative_volume =
                correction_options.target_negative_volume;
            result.initial_negative_volume = initial.negative_volume;
            result.corrected_negative_volume = initial.negative_volume;
            result.volume_error =
                initial.negative_volume -
                correction_options.target_negative_volume;
            result.diagnostic = initial.diagnostic;
            accumulateGeneratedVolumeDiagnostics(result, initial);
            corrected_solution.assign(solution.begin(), solution.end());
            return result;
        }
        const Real target = correction_options.target_negative_volume;
        if (target < -correction_options.volume_tolerance ||
            target > initial.total_volume + correction_options.volume_tolerance) {
            throw std::invalid_argument(
                "level-set global shift correction target volume is outside the total volume range");
        }

        LevelSetGlobalShiftCorrectionResult result;
        result.target_negative_volume = target;
        result.initial_negative_volume = initial.negative_volume;
        result.initial_volume = initial;
        result.corrected_volume = initial;
        result.corrected_negative_volume = initial.negative_volume;
        result.volume_error = initial.negative_volume - target;
        accumulateGeneratedVolumeDiagnostics(result, initial);
        corrected_solution.assign(solution.begin(), solution.end());
        if (std::abs(result.volume_error) <= correction_options.volume_tolerance) {
            result.success = true;
            return result;
        }

        const auto field_coefficients =
            solution.subspan(offset, n_field_dofs);
        const auto [min_coeff, max_coeff] =
            coefficientRange(field_coefficients);
        const Real pad = std::max(volume_options.tolerance * Real{10.0},
                                  Real{1.0e-12});
        Real lower = volume_options.isovalue - max_coeff - pad;
        Real upper = volume_options.isovalue - min_coeff + pad;
        if (!(lower < upper)) {
            lower -= Real{1.0};
            upper += Real{1.0};
        }

        Real best_shift = 0.0;
        Real best_error = std::abs(result.volume_error);
        LevelSetVolumeResult best_volume = initial;
        std::vector<Real> best_solution(solution.begin(), solution.end());

        for (int iter = 1; iter <= correction_options.max_iterations; ++iter) {
            const Real shift = Real{0.5} * (lower + upper);
            auto shifted = shiftedSystemSolution(
                solution,
                offset,
                n_field_dofs,
                shift);
            auto volume = computeGeneratedInterfaceLevelSetVolume(
                system,
                level_set_field,
                volume_options,
                shifted,
                volume_lifecycle);
            accumulateGeneratedVolumeDiagnostics(result, volume);
            if (!volume.success) {
                result.success = false;
                result.applied_shift = best_shift;
                result.corrected_negative_volume = best_volume.negative_volume;
                result.volume_error = best_volume.negative_volume - target;
                result.corrected_volume = best_volume;
                result.diagnostic = volume.diagnostic;
                corrected_solution = std::move(best_solution);
                return result;
            }
            const Real signed_error = volume.negative_volume - target;
            const Real abs_error = std::abs(signed_error);
            if (abs_error < best_error) {
                best_error = abs_error;
                best_shift = shift;
                best_volume = volume;
                best_solution = std::move(shifted);
            }

            result.iterations = iter;
            if (abs_error <= correction_options.volume_tolerance) {
                result.success = true;
                result.applied_shift = best_shift;
                result.corrected_negative_volume = best_volume.negative_volume;
                result.volume_error = best_volume.negative_volume - target;
                result.corrected_volume = best_volume;
                corrected_solution = std::move(best_solution);
                return result;
            }

            if (signed_error > 0.0) {
                lower = shift;
            } else {
                upper = shift;
            }
        }

        result.success = best_error <= correction_options.volume_tolerance;
        result.applied_shift = best_shift;
        result.corrected_negative_volume = best_volume.negative_volume;
        result.volume_error = best_volume.negative_volume - target;
        result.corrected_volume = best_volume;
        result.diagnostic =
            result.success
                ? std::string{}
                : "level-set global shift correction did not reach the requested volume tolerance";
        corrected_solution = std::move(best_solution);
        return result;
    }

    const auto& field_dofs = system.fieldDofHandler(level_set_field);
    const auto n_field_dofs = static_cast<std::size_t>(field_dofs.getNumDofs());
    const auto offset = static_cast<std::size_t>(system.fieldDofOffset(level_set_field));
    if (offset + n_field_dofs > solution.size()) {
        throw std::invalid_argument(
            "level-set global shift correction received an incompatible system solution span");
    }

    std::vector<Real> field_coefficients(
        solution.begin() + static_cast<std::ptrdiff_t>(offset),
        solution.begin() + static_cast<std::ptrdiff_t>(offset + n_field_dofs));
    std::vector<Real> corrected_field;
    auto result = applyGlobalLevelSetShiftCorrection(
        system.meshAccess(),
        field_dofs,
        volume_options,
        correction_options,
        field_coefficients,
        corrected_field);

    corrected_solution.assign(solution.begin(), solution.end());
    std::copy(corrected_field.begin(),
              corrected_field.end(),
              corrected_solution.begin() + static_cast<std::ptrdiff_t>(offset));
    return result;
}

} // namespace svmp::FE::level_set
