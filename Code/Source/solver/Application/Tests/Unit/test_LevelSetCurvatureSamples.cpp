#include <gtest/gtest.h>

#include "Application/Core/LevelSetCurvatureSamples.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <optional>
#include <random>
#include <vector>

namespace {

using Sample = svmp::FE::level_set::LevelSetCurvatureProjectionSample;
using Filter = application::core::LevelSetCurvatureSampleDuplicateFilter;

/// The linear scan that collectLevelSetCurvatureSupplementalSamples used
/// before the hashed filter, verbatim apart from returning the index of the
/// first duplicate instead of acting on it.
std::optional<std::size_t> appendWithLinearScan(std::vector<Sample>& samples,
                                                const Sample& candidate)
{
  constexpr svmp::FE::Real duplicate_tol2 = svmp::FE::Real{1.0e-24};
  constexpr svmp::FE::Real duplicate_value_tol = svmp::FE::Real{1.0e-12};
  for (std::size_t index = 0u; index < samples.size(); ++index) {
    const auto& existing = samples[index];
    if (existing.parent_cell != candidate.parent_cell ||
        existing.generated_interface_geometry !=
            candidate.generated_interface_geometry) {
      continue;
    }
    const auto dx = existing.coordinate[0] - candidate.coordinate[0];
    const auto dy = existing.coordinate[1] - candidate.coordinate[1];
    const auto dz = existing.coordinate[2] - candidate.coordinate[2];
    const auto dist2 = dx * dx + dy * dy + dz * dz;
    if (dist2 <= duplicate_tol2 &&
        std::abs(existing.value - candidate.value) <= duplicate_value_tol) {
      return index;
    }
  }
  samples.push_back(candidate);
  return std::nullopt;
}

bool bitwiseEqual(const Sample& a, const Sample& b)
{
  return a.parent_cell == b.parent_cell &&
         std::memcmp(a.coordinate.data(), b.coordinate.data(),
                     sizeof(a.coordinate)) == 0 &&
         std::memcmp(&a.value, &b.value, sizeof(a.value)) == 0 &&
         a.free_surface_snapshot_revision_key ==
             b.free_surface_snapshot_revision_key &&
         a.source_value_revision == b.source_value_revision &&
         a.cut_topology_revision == b.cut_topology_revision &&
         a.generated_interface_geometry == b.generated_interface_geometry;
}

/// Random candidates with many exact repeats and perturbations on both sides
/// of the distance and value tolerances, spread over few cells so that
/// buckets hold many samples.
std::vector<Sample> randomCandidates(std::uint32_t seed, std::size_t count)
{
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> cell_dist(0, 11);
  std::uniform_int_distribution<int> flag_dist(0, 1);
  std::uniform_real_distribution<double> coord_dist(-1.0, 1.0);
  std::uniform_int_distribution<int> choice(0, 9);
  std::uniform_int_distribution<int> revision_dist(1, 3);
  // Per-coordinate offsets: 3 * (5e-13)^2 = 7.5e-25 is inside the squared
  // distance tolerance, 3 * (7e-13)^2 = 1.47e-24 is outside it.
  const double coordinate_offsets[] = {0.0, 1.0e-14, 5.0e-13, 5.8e-13,
                                       7.0e-13, 1.0e-12, 1.0e-9};
  const double value_offsets[] = {0.0, 1.0e-13, 9.0e-13, 1.0e-12,
                                  1.1e-12, 1.0e-9};
  std::uniform_int_distribution<std::size_t> coordinate_offset_dist(
      0u, std::size(coordinate_offsets) - 1u);
  std::uniform_int_distribution<std::size_t> value_offset_dist(
      0u, std::size(value_offsets) - 1u);

  std::vector<Sample> bases;
  std::vector<Sample> candidates;
  candidates.reserve(count);
  for (std::size_t i = 0u; i < count; ++i) {
    Sample sample{};
    const int kind = choice(rng);
    if (bases.empty() || kind < 3) {
      sample.parent_cell = static_cast<svmp::FE::MeshIndex>(cell_dist(rng));
      sample.coordinate = {coord_dist(rng), coord_dist(rng),
                           kind == 0 ? 0.0 : coord_dist(rng)};
      sample.value = kind == 1 ? 0.0 : coord_dist(rng);
      sample.generated_interface_geometry = flag_dist(rng) != 0;
      bases.push_back(sample);
    } else {
      std::uniform_int_distribution<std::size_t> base_dist(0u,
                                                           bases.size() - 1u);
      sample = bases[base_dist(rng)];
      if (kind >= 5) {
        for (auto& x : sample.coordinate) {
          const double offset = coordinate_offsets[coordinate_offset_dist(rng)];
          x += flag_dist(rng) != 0 ? offset : -offset;
        }
        const double offset = value_offsets[value_offset_dist(rng)];
        sample.value += flag_dist(rng) != 0 ? offset : -offset;
      }
      if (kind == 9) {
        // Same point in another cell or of the other kind is never a
        // duplicate.
        if (flag_dist(rng) != 0) {
          sample.parent_cell = static_cast<svmp::FE::MeshIndex>(cell_dist(rng));
        } else {
          sample.generated_interface_geometry =
              !sample.generated_interface_geometry;
        }
      }
    }
    sample.free_surface_snapshot_revision_key =
        static_cast<std::uint64_t>(revision_dist(rng));
    sample.source_value_revision = static_cast<std::uint64_t>(revision_dist(rng));
    sample.cut_topology_revision = static_cast<std::uint64_t>(revision_dist(rng));
    candidates.push_back(sample);
  }
  return candidates;
}

} // namespace

TEST(LevelSetCurvatureSampleDuplicateFilter,
     MatchesLinearScanOnRandomizedSamplesWithDuplicates)
{
  std::size_t total_duplicates = 0u;
  std::size_t total_appended = 0u;
  for (std::uint32_t seed = 1u; seed <= 40u; ++seed) {
    const auto candidates = randomCandidates(seed, 3000u);
    std::vector<Sample> reference;
    std::vector<Sample> hashed;
    Filter filter;
    for (std::size_t i = 0u; i < candidates.size(); ++i) {
      const auto expected = appendWithLinearScan(reference, candidates[i]);
      const auto* existing = filter.appendUnlessDuplicate(hashed, candidates[i]);
      ASSERT_EQ(expected.has_value(), existing != nullptr)
          << "seed " << seed << " candidate " << i;
      if (expected.has_value()) {
        // The first duplicate decides the snapshot-consistency check of the
        // caller, so it must be the same earlier sample.
        ASSERT_EQ(*expected, static_cast<std::size_t>(existing - hashed.data()))
            << "seed " << seed << " candidate " << i;
        ++total_duplicates;
      } else {
        ++total_appended;
      }
      ASSERT_EQ(reference.size(), hashed.size());
    }
    for (std::size_t i = 0u; i < reference.size(); ++i) {
      ASSERT_TRUE(bitwiseEqual(reference[i], hashed[i]))
          << "seed " << seed << " sample " << i;
    }
  }
  // The generator must exercise both outcomes heavily.
  EXPECT_GT(total_duplicates, 20000u);
  EXPECT_GT(total_appended, 20000u);
}

TEST(LevelSetCurvatureSampleDuplicateFilter, AppliesTheToleranceCriterion)
{
  Sample base{};
  base.parent_cell = 7;
  base.coordinate = {0.25, -0.5, 0.125};
  base.value = 0.75;

  auto nearby = base;
  nearby.coordinate[0] += 5.0e-13;
  nearby.value += 5.0e-13;
  EXPECT_TRUE(Filter::isDuplicate(base, nearby));

  auto far = base;
  far.coordinate[0] += 2.0e-12;
  EXPECT_FALSE(Filter::isDuplicate(base, far));

  auto other_value = base;
  other_value.value += 2.0e-12;
  EXPECT_FALSE(Filter::isDuplicate(base, other_value));

  auto other_cell = base;
  other_cell.parent_cell = 8;
  EXPECT_FALSE(Filter::isDuplicate(base, other_cell));

  auto other_kind = base;
  other_kind.generated_interface_geometry = true;
  EXPECT_FALSE(Filter::isDuplicate(base, other_kind));

  // Revision fields do not take part in the criterion.
  auto other_revision = base;
  other_revision.free_surface_snapshot_revision_key = 99u;
  other_revision.source_value_revision = 98u;
  other_revision.cut_topology_revision = 97u;
  EXPECT_TRUE(Filter::isDuplicate(base, other_revision));

  std::vector<Sample> samples;
  Filter filter;
  EXPECT_EQ(filter.appendUnlessDuplicate(samples, base), nullptr);
  EXPECT_EQ(filter.appendUnlessDuplicate(samples, far), nullptr);
  EXPECT_EQ(filter.appendUnlessDuplicate(samples, other_kind), nullptr);
  EXPECT_EQ(filter.appendUnlessDuplicate(samples, nearby), &samples[0]);
  EXPECT_EQ(filter.appendUnlessDuplicate(samples, other_revision),
            &samples[0]);
  ASSERT_EQ(samples.size(), 3u);
  EXPECT_TRUE(bitwiseEqual(samples[0], base));
  EXPECT_TRUE(bitwiseEqual(samples[1], far));
  EXPECT_TRUE(bitwiseEqual(samples[2], other_kind));
}
