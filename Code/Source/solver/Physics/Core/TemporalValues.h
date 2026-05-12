/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_CORE_TEMPORAL_VALUES_H
#define SVMP_PHYSICS_CORE_TEMPORAL_VALUES_H

#include "FE/Core/Types.h"

#include <memory>
#include <string>
#include <vector>

namespace svmp {
namespace Physics {

enum class TemporalEndBehavior {
  Clamp,
  Periodic
};

struct TemporalValues {
  int num_time_points{0};
  int num_components{1};
  TemporalEndBehavior end_behavior{TemporalEndBehavior::Clamp};
  std::vector<double> t{};
  std::vector<double> v{};

  [[nodiscard]] double firstTime() const;
  [[nodiscard]] double lastTime() const;
  [[nodiscard]] double period() const;
  [[nodiscard]] double sample(int time_idx, int component = 0) const;
  [[nodiscard]] double firstValue(int component = 0) const;
  [[nodiscard]] double lastValue(int component = 0) const;
  [[nodiscard]] double interpolate(double time, int component = 0) const;
};

[[nodiscard]] std::shared_ptr<TemporalValues>
readTemporalValuesFile(const std::string& file_path,
                       int num_components = 1,
                       TemporalEndBehavior end_behavior = TemporalEndBehavior::Clamp);

} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_CORE_TEMPORAL_VALUES_H
