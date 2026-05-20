#include <gtest/gtest.h>

#include "Application/Core/NearestPointIndex.h"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using application::core::NearestPointIndex;
using application::core::NearestPointQueryResult;
using application::core::NearestPointRecord;

NearestPointQueryResult bruteForceNearest(
    int dimension,
    const std::vector<NearestPointRecord>& records,
    const std::array<double, 3>& query)
{
  NearestPointQueryResult result;
  for (const auto& record : records) {
    double d2 = 0.0;
    for (int d = 0; d < dimension; ++d) {
      const auto index = static_cast<std::size_t>(d);
      const double delta = query[index] - record.point[index];
      d2 += delta * delta;
    }
    if (!result.found ||
        d2 < result.distance_squared ||
        (d2 == result.distance_squared && record.payload < result.payload)) {
      result.found = true;
      result.payload = record.payload;
      result.distance_squared = d2;
    }
  }
  return result;
}

void expectMatchesBruteForce(
    int dimension,
    const std::vector<NearestPointRecord>& records,
    const std::vector<std::array<double, 3>>& queries)
{
  NearestPointIndex index(dimension, records);
  EXPECT_EQ(index.dimension(), dimension);
  EXPECT_EQ(index.size(), records.size());
  for (const auto& query : queries) {
    const auto actual = index.nearest(query);
    const auto expected = bruteForceNearest(dimension, records, query);
    EXPECT_EQ(actual.found, expected.found);
    EXPECT_EQ(actual.payload, expected.payload);
    EXPECT_DOUBLE_EQ(actual.distance_squared, expected.distance_squared);
  }
}

} // namespace

TEST(NearestPointIndex, EmptyIndexReportsNoNearestPoint)
{
  NearestPointIndex index(3, {});
  EXPECT_TRUE(index.empty());
  EXPECT_FALSE(index.nearest({0.0, 0.0, 0.0}).found);
}

TEST(NearestPointIndex, RejectsInvalidDimension)
{
  EXPECT_THROW(NearestPointIndex(0, {}), std::invalid_argument);
  EXPECT_THROW(NearestPointIndex(4, {}), std::invalid_argument);
}

TEST(NearestPointIndex, MatchesBruteForceNearestNeighborInTwoDimensions)
{
  const std::vector<NearestPointRecord> records = {
      {{0.0, 0.0, 0.0}, 10u},
      {{1.0, 0.0, 0.0}, 20u},
      {{0.0, 1.0, 0.0}, 30u},
      {{1.0, 1.0, 0.0}, 40u},
      {{0.45, 0.55, 0.0}, 50u},
  };
  const std::vector<std::array<double, 3>> queries = {
      {0.05, 0.05, 0.0},
      {0.95, 0.05, 0.0},
      {0.20, 0.90, 0.0},
      {0.46, 0.56, 0.0},
      {2.00, 2.00, 0.0},
  };
  expectMatchesBruteForce(2, records, queries);
}

TEST(NearestPointIndex, MatchesBruteForceNearestNeighborInThreeDimensions)
{
  const std::vector<NearestPointRecord> records = {
      {{0.0, 0.0, 0.0}, 1u},
      {{1.0, 0.0, 0.0}, 2u},
      {{0.0, 1.0, 0.0}, 3u},
      {{0.0, 0.0, 1.0}, 4u},
      {{1.0, 1.0, 1.0}, 5u},
      {{0.25, 0.25, 0.35}, 6u},
  };
  const std::vector<std::array<double, 3>> queries = {
      {0.20, 0.20, 0.20},
      {0.90, 0.10, 0.10},
      {0.10, 0.90, 0.10},
      {0.10, 0.10, 0.90},
      {0.80, 0.90, 0.85},
  };
  expectMatchesBruteForce(3, records, queries);
}

TEST(NearestPointIndex, TiesPreferLowestPayloadForDeterminism)
{
  const std::vector<NearestPointRecord> records = {
      {{0.0, 0.0, 0.0}, 5u},
      {{0.0, 0.0, 0.0}, 2u},
      {{1.0, 0.0, 0.0}, 1u},
  };
  const NearestPointIndex index(2, records);
  const auto nearest = index.nearest({0.0, 0.0, 0.0});
  ASSERT_TRUE(nearest.found);
  EXPECT_EQ(nearest.payload, 2u);
  EXPECT_DOUBLE_EQ(nearest.distance_squared, 0.0);
}
