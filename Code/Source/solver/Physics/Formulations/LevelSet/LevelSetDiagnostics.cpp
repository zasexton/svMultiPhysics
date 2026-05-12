#include "Physics/Formulations/LevelSet/LevelSetDiagnostics.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace svmp {
namespace Physics {
namespace formulations {
namespace level_set {
namespace {

void appendScalar(std::vector<LevelSetScalarDiagnostic>& scalars,
                  std::string name,
                  FE::Real value)
{
    scalars.push_back(LevelSetScalarDiagnostic{
        .name = std::move(name),
        .value = value,
    });
}

void appendVolumeScalars(LevelSetOutputDiagnostics& result,
                         const LevelSetOutputDiagnosticsOptions& options)
{
    appendScalar(result.scalars, "level_set.total_volume", result.volume.total_volume);
    appendScalar(result.scalars, "level_set.negative_volume", result.volume.negative_volume);
    appendScalar(result.scalars, "level_set.positive_volume", result.volume.positive_volume);
    appendScalar(result.scalars,
                 "level_set.cut_cells",
                 static_cast<FE::Real>(result.volume.cut_cells));
    if (options.has_reference_negative_volume) {
        appendScalar(result.scalars, "level_set.negative_volume_loss", result.negative_volume_loss);
        appendScalar(result.scalars,
                     "level_set.relative_negative_volume_loss",
                     result.relative_negative_volume_loss);
    }
}

void appendSignedDistanceScalars(LevelSetOutputDiagnostics& result)
{
    appendScalar(result.scalars,
                 "level_set.signed_distance_max_error",
                 result.signed_distance_max_error);
    appendScalar(result.scalars,
                 "level_set.signed_distance_l2_error",
                 result.signed_distance_l2_error);
    appendScalar(result.scalars,
                 "level_set.signed_distance_samples",
                 static_cast<FE::Real>(result.signed_distance_samples));
}

void computeReferenceVolumeLoss(LevelSetOutputDiagnostics& result,
                                const LevelSetOutputDiagnosticsOptions& options)
{
    if (!options.has_reference_negative_volume) {
        return;
    }
    result.negative_volume_loss =
        options.reference_negative_volume - result.volume.negative_volume;
    const FE::Real denom = std::abs(options.reference_negative_volume);
    result.relative_negative_volume_loss =
        denom > FE::Real{0.0} ? result.negative_volume_loss / denom : FE::Real{0.0};
}

void computeSignedDistanceError(LevelSetOutputDiagnostics& result,
                                const LevelSetOutputDiagnosticsOptions& options,
                                const FE::assembly::IMeshAccess& mesh,
                                const FE::dofs::DofHandler& level_set_dofs,
                                std::span<const FE::Real> coefficients)
{
    if (!options.compute_signed_distance_error) {
        return;
    }
    if (!(options.signed_distance.interface_band_width > 0.0)) {
        throw std::invalid_argument(
            "level-set output diagnostics require a positive signed-distance interface band width");
    }

    std::vector<FE::Real> repaired;
    result.signed_distance = repairLevelSetSignedDistanceByProjection(
        mesh,
        level_set_dofs,
        options.signed_distance,
        coefficients,
        repaired);
    if (!result.signed_distance.success) {
        result.success = false;
        result.diagnostic = result.signed_distance.diagnostic;
        return;
    }
    if (repaired.size() != coefficients.size()) {
        throw std::runtime_error(
            "level-set output diagnostics received an incompatible repaired signed-distance field");
    }

    FE::Real squared_sum = 0.0;
    for (std::size_t i = 0; i < repaired.size(); ++i) {
        if (std::abs(repaired[i]) > options.signed_distance.interface_band_width) {
            continue;
        }
        const FE::Real error = std::abs(repaired[i] - coefficients[i]);
        result.signed_distance_max_error =
            std::max(result.signed_distance_max_error, error);
        squared_sum += error * error;
        ++result.signed_distance_samples;
    }

    if (result.signed_distance_samples > 0u) {
        result.signed_distance_l2_error =
            std::sqrt(squared_sum / static_cast<FE::Real>(result.signed_distance_samples));
    }
    appendSignedDistanceScalars(result);
}

} // namespace

LevelSetOutputDiagnostics computeLevelSetOutputDiagnostics(
    const FE::assembly::IMeshAccess& mesh,
    const FE::dofs::DofHandler& level_set_dofs,
    const LevelSetOutputDiagnosticsOptions& options,
    std::span<const FE::Real> coefficients)
{
    LevelSetOutputDiagnostics result;
    result.volume = computeLevelSetCutCellVolume(
        mesh,
        level_set_dofs,
        options.volume,
        coefficients);
    result.success = result.volume.success;
    result.diagnostic = result.volume.diagnostic;
    computeReferenceVolumeLoss(result, options);
    appendVolumeScalars(result, options);
    if (!result.success) {
        return result;
    }

    computeSignedDistanceError(result, options, mesh, level_set_dofs, coefficients);
    return result;
}

LevelSetOutputDiagnostics computeLevelSetOutputDiagnostics(
    const FE::systems::FESystem& system,
    FE::FieldId level_set_field,
    const LevelSetOutputDiagnosticsOptions& options,
    std::span<const FE::Real> solution)
{
    const auto& field_dofs = system.fieldDofHandler(level_set_field);
    const auto n_field_dofs = static_cast<std::size_t>(field_dofs.getNumDofs());
    const auto offset = static_cast<std::size_t>(system.fieldDofOffset(level_set_field));
    if (offset + n_field_dofs > solution.size()) {
        throw std::invalid_argument(
            "level-set output diagnostics received an incompatible system solution span");
    }

    return computeLevelSetOutputDiagnostics(
        system.meshAccess(),
        field_dofs,
        options,
        solution.subspan(offset, n_field_dofs));
}

} // namespace level_set
} // namespace formulations
} // namespace Physics
} // namespace svmp
