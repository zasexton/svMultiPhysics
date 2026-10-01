#pragma once

#include "FE/Core/Types.h"
#include "FE/Geometry/CutQuadrature.h"
#include "FE/LevelSet/LevelSetCurvatureProjection.h"
#include "FE/Systems/SystemState.h"

#include <array>
#include <cstddef>
#include <optional>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {
namespace assembly {
class IMeshAccess;
} // namespace assembly
namespace systems {
class FESystem;
} // namespace systems
} // namespace FE
} // namespace svmp

namespace application {
namespace core {

/** Map the exact reference sample used for a curvature value to physical space. */
[[nodiscard]] std::optional<std::array<svmp::FE::Real, 3>>
mapLevelSetCurvatureReferenceSampleToPhysical(
    const svmp::FE::assembly::IMeshAccess& mesh,
    svmp::FE::GlobalIndex cell,
    const std::array<svmp::FE::Real, 3>& reference_point);

[[nodiscard]] std::vector<svmp::FE::level_set::LevelSetCurvatureProjectionSample>
collectLevelSetCurvatureCutVolumeSupplementalSamples(
    const svmp::FE::systems::FESystem& system,
    const svmp::FE::systems::SystemStateView& state,
    svmp::FE::FieldId field,
    int interface_marker,
    svmp::FE::geometry::CutIntegrationSide side,
    std::uint64_t evaluated_state_source_revision);

/**
 * Collect one exactly paired interior sample for every high-order field cell
 * selected by the marker's authoritative interface rules.
 */
[[nodiscard]] std::vector<svmp::FE::level_set::LevelSetCurvatureProjectionSample>
collectLevelSetCurvatureHighOrderSupplementalSamples(
    const svmp::FE::systems::FESystem& system,
    const svmp::FE::systems::SystemStateView& state,
    svmp::FE::FieldId field,
    int interface_marker,
    std::uint64_t evaluated_state_source_revision);

/**
 * Insertion-ordered duplicate filter for curvature supplemental samples.
 *
 * A candidate duplicates an earlier sample when both have the same parent
 * cell and the same generated_interface_geometry flag, their squared
 * coordinate distance is at most kDuplicateDistanceSquaredTolerance and their
 * values differ by at most kDuplicateValueTolerance.  Only earlier samples of
 * the candidate's (cell, flag) bucket can match, so scanning that bucket in
 * insertion order finds the same first duplicate as a scan over all earlier
 * samples: the accepted samples and their order do not depend on the lookup.
 */
class LevelSetCurvatureSampleDuplicateFilter {
public:
  using Sample = svmp::FE::level_set::LevelSetCurvatureProjectionSample;

  static constexpr svmp::FE::Real kDuplicateDistanceSquaredTolerance{1.0e-24};
  static constexpr svmp::FE::Real kDuplicateValueTolerance{1.0e-12};

  /** The duplicate criterion for one ordered pair. */
  [[nodiscard]] static bool isDuplicate(const Sample& existing,
                                        const Sample& candidate) noexcept;

  /**
   * Append @p candidate to @p samples unless an earlier sample duplicates it.
   * Returns the first such earlier sample in insertion order, or nullptr when
   * @p candidate was appended.  @p samples must start empty and grow only
   * through this filter.
   */
  const Sample* appendUnlessDuplicate(std::vector<Sample>& samples,
                                      const Sample& candidate);

private:
  // Sample indices per parent cell, split by generated_interface_geometry.
  std::unordered_map<svmp::FE::MeshIndex,
                     std::array<std::vector<std::size_t>, 2>>
      buckets_;
};

} // namespace core
} // namespace application
