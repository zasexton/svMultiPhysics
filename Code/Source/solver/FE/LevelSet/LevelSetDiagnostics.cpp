#include "LevelSet/LevelSetDiagnostics.h"

#include <cmath>
#include <utility>

namespace svmp::FE::level_set {
namespace {

void appendScalar(std::vector<LevelSetScalarDiagnostic>& scalars,
                  std::string name,
                  Real value)
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
                 static_cast<Real>(result.volume.cut_cells));
    if (options.has_reference_negative_volume) {
        appendScalar(result.scalars, "level_set.negative_volume_loss", result.negative_volume_loss);
        appendScalar(result.scalars,
                     "level_set.relative_negative_volume_loss",
                     result.relative_negative_volume_loss);
    }
}

void computeReferenceVolumeLoss(LevelSetOutputDiagnostics& result,
                                const LevelSetOutputDiagnosticsOptions& options)
{
    if (!options.has_reference_negative_volume) {
        return;
    }
    result.negative_volume_loss =
        options.reference_negative_volume - result.volume.negative_volume;
    const Real denom = std::abs(options.reference_negative_volume);
    result.relative_negative_volume_loss =
        denom > Real{0.0} ? result.negative_volume_loss / denom : Real{0.0};
}

LevelSetOutputDiagnostics volumeDiagnostics(LevelSetVolumeResult volume,
                                            const LevelSetOutputDiagnosticsOptions& options)
{
    LevelSetOutputDiagnostics result;
    result.volume = std::move(volume);
    result.success = result.volume.success;
    result.diagnostic = result.volume.diagnostic;
    computeReferenceVolumeLoss(result, options);
    appendVolumeScalars(result, options);
    return result;
}

} // namespace

LevelSetOutputDiagnostics computeLevelSetOutputDiagnostics(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    const LevelSetOutputDiagnosticsOptions& options,
    std::span<const Real> coefficients)
{
    return volumeDiagnostics(
        computeLevelSetCutCellVolume(mesh, level_set_dofs, options.volume, coefficients), options);
}

LevelSetOutputDiagnostics computeLevelSetOutputDiagnostics(
    const systems::FESystem& system,
    FieldId level_set_field,
    const LevelSetOutputDiagnosticsOptions& options,
    std::span<const Real> solution)
{
    return volumeDiagnostics(
        computeLevelSetCutCellVolume(system, level_set_field, options.volume, solution), options);
}

} // namespace svmp::FE::level_set
