#ifndef SVMP_APPLICATION_CORE_NEARESTPOINTINDEX_H
#define SVMP_APPLICATION_CORE_NEARESTPOINTINDEX_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace application {
namespace core {

struct NearestPointRecord {
  std::array<double, 3> point{0.0, 0.0, 0.0};
  std::size_t payload{0u};
};

struct NearestPointQueryResult {
  bool found{false};
  std::size_t payload{0u};
  double distance_squared{std::numeric_limits<double>::infinity()};
};

class NearestPointIndex {
public:
  NearestPointIndex() = default;

  NearestPointIndex(int dimension, std::vector<NearestPointRecord> records)
  {
    reset(dimension, std::move(records));
  }

  void reset(int dimension, std::vector<NearestPointRecord> records)
  {
    if (dimension < 1 || dimension > 3) {
      throw std::invalid_argument("NearestPointIndex: dimension must be in [1, 3]");
    }
    dimension_ = dimension;
    records_ = std::move(records);
    nodes_.clear();
    nodes_.reserve(records_.size());
    root_ = build(0u, records_.size());
  }

  [[nodiscard]] bool empty() const noexcept { return records_.empty(); }

  [[nodiscard]] std::size_t size() const noexcept { return records_.size(); }

  [[nodiscard]] int dimension() const noexcept { return dimension_; }

  [[nodiscard]] NearestPointQueryResult nearest(
      const std::array<double, 3>& query) const noexcept
  {
    NearestPointQueryResult result;
    search(root_, query, result);
    return result;
  }

private:
  struct Node {
    std::size_t record_index{0u};
    int axis{0};
    int left{-1};
    int right{-1};
  };

  int dimension_{3};
  std::vector<NearestPointRecord> records_{};
  std::vector<Node> nodes_{};
  int root_{-1};

  [[nodiscard]] int chooseAxis(std::size_t begin, std::size_t end) const noexcept
  {
    std::array<double, 3> min_coord{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()};
    std::array<double, 3> max_coord{
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()};

    for (std::size_t i = begin; i < end; ++i) {
      for (int d = 0; d < dimension_; ++d) {
        min_coord[static_cast<std::size_t>(d)] =
            std::min(min_coord[static_cast<std::size_t>(d)],
                     records_[i].point[static_cast<std::size_t>(d)]);
        max_coord[static_cast<std::size_t>(d)] =
            std::max(max_coord[static_cast<std::size_t>(d)],
                     records_[i].point[static_cast<std::size_t>(d)]);
      }
    }

    int axis = 0;
    double spread = max_coord[0] - min_coord[0];
    for (int d = 1; d < dimension_; ++d) {
      const double candidate_spread =
          max_coord[static_cast<std::size_t>(d)] -
          min_coord[static_cast<std::size_t>(d)];
      if (candidate_spread > spread) {
        spread = candidate_spread;
        axis = d;
      }
    }
    return axis;
  }

  int build(std::size_t begin, std::size_t end)
  {
    if (begin >= end) {
      return -1;
    }

    const int axis = chooseAxis(begin, end);
    const auto mid = begin + (end - begin) / 2u;
    auto axis_less = [axis](const NearestPointRecord& a,
                            const NearestPointRecord& b) {
      const auto axis_index = static_cast<std::size_t>(axis);
      if (a.point[axis_index] != b.point[axis_index]) {
        return a.point[axis_index] < b.point[axis_index];
      }
      return a.payload < b.payload;
    };
    std::nth_element(records_.begin() + static_cast<std::ptrdiff_t>(begin),
                     records_.begin() + static_cast<std::ptrdiff_t>(mid),
                     records_.begin() + static_cast<std::ptrdiff_t>(end),
                     axis_less);

    const int node_index = static_cast<int>(nodes_.size());
    nodes_.push_back(Node{mid, axis, -1, -1});
    nodes_[static_cast<std::size_t>(node_index)].left = build(begin, mid);
    nodes_[static_cast<std::size_t>(node_index)].right = build(mid + 1u, end);
    return node_index;
  }

  [[nodiscard]] double distanceSquared(
      const std::array<double, 3>& a,
      const std::array<double, 3>& b) const noexcept
  {
    double d2 = 0.0;
    for (int d = 0; d < dimension_; ++d) {
      const auto index = static_cast<std::size_t>(d);
      const double delta = a[index] - b[index];
      d2 += delta * delta;
    }
    return d2;
  }

  [[nodiscard]] static bool isBetter(double distance_squared,
                                     std::size_t payload,
                                     const NearestPointQueryResult& result) noexcept
  {
    return !result.found ||
           distance_squared < result.distance_squared ||
           (distance_squared == result.distance_squared &&
            payload < result.payload);
  }

  void search(int node_index,
              const std::array<double, 3>& query,
              NearestPointQueryResult& result) const noexcept
  {
    if (node_index < 0) {
      return;
    }

    const auto& node = nodes_[static_cast<std::size_t>(node_index)];
    const auto& record = records_[node.record_index];
    const double d2 = distanceSquared(query, record.point);
    if (isBetter(d2, record.payload, result)) {
      result.found = true;
      result.payload = record.payload;
      result.distance_squared = d2;
    }

    const auto axis = static_cast<std::size_t>(node.axis);
    const double delta = query[axis] - record.point[axis];
    const int near_child = delta <= 0.0 ? node.left : node.right;
    const int far_child = delta <= 0.0 ? node.right : node.left;
    search(near_child, query, result);
    if (delta * delta <= result.distance_squared) {
      search(far_child, query, result);
    }
  }
};

} // namespace core
} // namespace application

#endif // SVMP_APPLICATION_CORE_NEARESTPOINTINDEX_H
