#include "Physics/Formulations/LevelSet/LevelSetVolume.h"

#include "FE/Dofs/EntityDofMap.h"
#include "Interfaces/LevelSetInterfaceBuilder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace level_set {
namespace {

using FE::interfaces::CutInterfaceDomainRequest;
using FE::interfaces::LevelSetCellCutInput;
using FE::interfaces::LevelSetInterfaceSource;

[[nodiscard]] std::array<FE::Real, 3> sub(const std::array<FE::Real, 3>& a,
                                          const std::array<FE::Real, 3>& b) noexcept
{
    return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
}

[[nodiscard]] std::array<FE::Real, 3> cross(const std::array<FE::Real, 3>& a,
                                            const std::array<FE::Real, 3>& b) noexcept
{
    return {{a[1] * b[2] - a[2] * b[1],
             a[2] * b[0] - a[0] * b[2],
             a[0] * b[1] - a[1] * b[0]}};
}

[[nodiscard]] FE::Real dot(const std::array<FE::Real, 3>& a,
                           const std::array<FE::Real, 3>& b) noexcept
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

[[nodiscard]] FE::Real norm(const std::array<FE::Real, 3>& a) noexcept
{
    return std::sqrt(dot(a, a));
}

[[nodiscard]] FE::Real triangleArea(const std::array<FE::Real, 3>& a,
                                    const std::array<FE::Real, 3>& b,
                                    const std::array<FE::Real, 3>& c) noexcept
{
    return FE::Real{0.5} * norm(cross(sub(b, a), sub(c, a)));
}

[[nodiscard]] FE::Real tetraVolume(const std::array<FE::Real, 3>& a,
                                   const std::array<FE::Real, 3>& b,
                                   const std::array<FE::Real, 3>& c,
                                   const std::array<FE::Real, 3>& d) noexcept
{
    return std::abs(dot(sub(b, a), cross(sub(c, a), sub(d, a)))) / FE::Real{6.0};
}

[[nodiscard]] std::size_t cornerCount(FE::ElementType type)
{
    switch (type) {
    case FE::ElementType::Triangle3:
    case FE::ElementType::Triangle6:
        return 3u;
    case FE::ElementType::Quad4:
    case FE::ElementType::Quad8:
    case FE::ElementType::Quad9:
    case FE::ElementType::Tetra4:
    case FE::ElementType::Tetra10:
        return 4u;
    default:
        return 0u;
    }
}

[[nodiscard]] FE::Real parentMeasure(FE::ElementType type,
                                     const std::vector<std::array<FE::Real, 3>>& x)
{
    switch (type) {
    case FE::ElementType::Triangle3:
    case FE::ElementType::Triangle6:
        return triangleArea(x[0], x[1], x[2]);
    case FE::ElementType::Quad4:
    case FE::ElementType::Quad8:
    case FE::ElementType::Quad9:
        return triangleArea(x[0], x[1], x[2]) + triangleArea(x[0], x[2], x[3]);
    case FE::ElementType::Tetra4:
    case FE::ElementType::Tetra10:
        return tetraVolume(x[0], x[1], x[2], x[3]);
    default:
        return FE::Real{0.0};
    }
}

[[nodiscard]] FE::Real coefficientAtVertex(const FE::dofs::EntityDofMap& entity_map,
                                           FE::GlobalIndex vertex,
                                           std::span<const FE::Real> coefficients)
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

[[nodiscard]] FE::Real negativeFractionFromValues(std::span<const FE::Real> values,
                                                  FE::Real tolerance)
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
        return FE::Real{1.0};
    }
    if (negative_count == 0u) {
        return FE::Real{0.0};
    }
    return FE::Real{-1.0};
}

[[nodiscard]] std::pair<FE::Real, FE::Real> vertexCoefficientRange(
    const FE::assembly::IMeshAccess& mesh,
    const FE::dofs::EntityDofMap& entity_map,
    std::span<const FE::Real> coefficients)
{
    FE::Real min_value = std::numeric_limits<FE::Real>::infinity();
    FE::Real max_value = -std::numeric_limits<FE::Real>::infinity();
    for (FE::GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        const FE::Real value = coefficientAtVertex(entity_map, vertex, coefficients);
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
    }
    if (!std::isfinite(min_value) || !std::isfinite(max_value)) {
        throw std::invalid_argument("level-set volume correction requires finite coefficients");
    }
    return {min_value, max_value};
}

[[nodiscard]] std::vector<FE::Real> shiftedCoefficients(std::span<const FE::Real> coefficients,
                                                        FE::Real shift)
{
    std::vector<FE::Real> shifted(coefficients.begin(), coefficients.end());
    for (auto& value : shifted) {
        value += shift;
    }
    return shifted;
}

} // namespace

LevelSetVolumeResult computeLevelSetCutCellVolume(
    const FE::assembly::IMeshAccess& mesh,
    const FE::dofs::DofHandler& level_set_dofs,
    const LevelSetVolumeOptions& options,
    std::span<const FE::Real> coefficients)
{
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
    request.source = LevelSetInterfaceSource::fromField(FE::FieldId{0});
    request.interface_marker = 0;
    request.isovalue = options.isovalue;
    request.tolerance = options.tolerance;
    request.quadrature_order = 1;

    LevelSetVolumeResult result;
    result.success = true;

    std::vector<FE::GlobalIndex> cell_nodes;
    std::vector<std::array<FE::Real, 3>> cell_coordinates;
    mesh.forEachCell([&](FE::GlobalIndex cell_id) {
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

        std::vector<FE::Real> signed_values;
        signed_values.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            signed_values.push_back(
                coefficientAtVertex(*entity_map, cell_nodes[i], coefficients) -
                options.isovalue);
        }

        const auto measure = parentMeasure(type, cell_coordinates);
        result.total_volume += measure;
        ++result.cells;

        const FE::Real simple_fraction =
            negativeFractionFromValues(signed_values, options.tolerance);
        FE::Real negative_fraction = simple_fraction;
        if (simple_fraction < FE::Real{0.0}) {
            LevelSetCellCutInput input{};
            input.parent_cell = cell_id;
            input.element_type = type;
            input.node_coordinates.assign(cell_coordinates.begin(),
                                          cell_coordinates.begin() +
                                              static_cast<std::ptrdiff_t>(count));
            input.level_set_values.reserve(count);
            for (const auto value : signed_values) {
                input.level_set_values.push_back(value + options.isovalue);
            }

            FE::interfaces::LevelSetCellCutResult cut_result;
            if (mesh.dimension() == 2) {
                cut_result = FE::interfaces::cutLinearLevelSetCell2D(request, input);
            } else if (mesh.dimension() == 3) {
                cut_result = FE::interfaces::cutLinearLevelSetCell3D(request, input);
            } else {
                throw std::invalid_argument(
                    "level-set volume calculation requires a 2D or 3D mesh");
            }
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
        } else if (negative_fraction >= FE::Real{1.0}) {
            ++result.full_negative_cells;
        } else {
            ++result.full_positive_cells;
        }

        const FE::Real negative = negative_fraction * measure;
        result.negative_volume += negative;
        result.positive_volume += measure - negative;
    });

    return result;
}

LevelSetVolumeResult computeLevelSetCutCellVolume(
    const FE::systems::FESystem& system,
    FE::FieldId level_set_field,
    const LevelSetVolumeOptions& options,
    std::span<const FE::Real> solution)
{
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
    const FE::assembly::IMeshAccess& mesh,
    const FE::dofs::DofHandler& level_set_dofs,
    const LevelSetVolumeOptions& volume_options,
    const LevelSetGlobalShiftCorrectionOptions& correction_options,
    std::span<const FE::Real> coefficients,
    std::vector<FE::Real>& corrected_coefficients)
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
    const FE::Real target = correction_options.target_negative_volume;
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
    const FE::Real pad = std::max(volume_options.tolerance * FE::Real{10.0},
                                  FE::Real{1.0e-12});
    FE::Real lower = volume_options.isovalue - max_coeff - pad;
    FE::Real upper = volume_options.isovalue - min_coeff + pad;
    if (!(lower < upper)) {
        lower -= FE::Real{1.0};
        upper += FE::Real{1.0};
    }

    FE::Real best_shift = 0.0;
    FE::Real best_error = std::abs(result.volume_error);
    LevelSetVolumeResult best_volume = initial;
    std::vector<FE::Real> best_coefficients(coefficients.begin(), coefficients.end());

    for (int iter = 1; iter <= correction_options.max_iterations; ++iter) {
        const FE::Real shift = FE::Real{0.5} * (lower + upper);
        auto shifted = shiftedCoefficients(coefficients, shift);
        auto volume = computeLevelSetCutCellVolume(
            mesh,
            level_set_dofs,
            volume_options,
            shifted);
        const FE::Real signed_error = volume.negative_volume - target;
        const FE::Real abs_error = std::abs(signed_error);
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
    const FE::systems::FESystem& system,
    FE::FieldId level_set_field,
    const LevelSetVolumeOptions& volume_options,
    const LevelSetGlobalShiftCorrectionOptions& correction_options,
    std::span<const FE::Real> solution,
    std::vector<FE::Real>& corrected_solution)
{
    const auto& field_dofs = system.fieldDofHandler(level_set_field);
    const auto n_field_dofs = static_cast<std::size_t>(field_dofs.getNumDofs());
    const auto offset = static_cast<std::size_t>(system.fieldDofOffset(level_set_field));
    if (offset + n_field_dofs > solution.size()) {
        throw std::invalid_argument(
            "level-set global shift correction received an incompatible system solution span");
    }

    std::vector<FE::Real> field_coefficients(
        solution.begin() + static_cast<std::ptrdiff_t>(offset),
        solution.begin() + static_cast<std::ptrdiff_t>(offset + n_field_dofs));
    std::vector<FE::Real> corrected_field;
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

} // namespace level_set
} // namespace formulations
} // namespace Physics
} // namespace svmp
