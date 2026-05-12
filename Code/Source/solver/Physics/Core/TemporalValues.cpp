/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Core/TemporalValues.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace svmp {
namespace Physics {

namespace {

void validateComponent(const TemporalValues& values, int component, const char* where)
{
  if (component < 0 || component >= values.num_components) {
    throw std::out_of_range(std::string(where) + ": component out of range");
  }
}

void validateIndex(const TemporalValues& values, int time_idx, const char* where)
{
  if (time_idx < 0 || time_idx >= values.num_time_points) {
    throw std::out_of_range(std::string(where) + ": time index out of range");
  }
}

} // namespace

double TemporalValues::firstTime() const
{
  if (t.empty()) {
    throw std::runtime_error("TemporalValues::firstTime: empty table");
  }
  return t.front();
}

double TemporalValues::lastTime() const
{
  if (t.empty()) {
    throw std::runtime_error("TemporalValues::lastTime: empty table");
  }
  return t.back();
}

double TemporalValues::period() const
{
  if (num_time_points < 2) {
    return 0.0;
  }
  return lastTime() - firstTime();
}

double TemporalValues::sample(int time_idx, int component) const
{
  validateIndex(*this, time_idx, "TemporalValues::sample");
  validateComponent(*this, component, "TemporalValues::sample");
  return v[static_cast<std::size_t>(time_idx) * static_cast<std::size_t>(num_components) +
           static_cast<std::size_t>(component)];
}

double TemporalValues::firstValue(int component) const
{
  return sample(0, component);
}

double TemporalValues::lastValue(int component) const
{
  return sample(num_time_points - 1, component);
}

double TemporalValues::interpolate(double time, int component) const
{
  validateComponent(*this, component, "TemporalValues::interpolate");
  if (num_time_points <= 0 || t.empty()) {
    return 0.0;
  }
  if (num_time_points == 1) {
    return sample(0, component);
  }

  double tt = time;
  const double t0 = firstTime();
  const double t1 = lastTime();
  const double span = t1 - t0;

  if (end_behavior == TemporalEndBehavior::Periodic && span > 0.0 && std::isfinite(span)) {
    tt = std::fmod(time - t0, span);
    if (tt < 0.0) {
      tt += span;
    }
    tt += t0;
  } else {
    if (tt <= t0) {
      return firstValue(component);
    }
    if (tt >= t1) {
      return lastValue(component);
    }
  }

  if (tt <= t0) {
    return firstValue(component);
  }
  if (tt >= t1) {
    return lastValue(component);
  }

  const auto upper = std::upper_bound(t.begin(), t.end(), tt);
  const auto i1 = static_cast<int>(std::distance(t.begin(), upper));
  const int i0 = std::max(0, i1 - 1);

  const double ta = t[static_cast<std::size_t>(i0)];
  const double tb = t[static_cast<std::size_t>(i1)];
  const double dt = tb - ta;
  const double alpha = (dt > 0.0) ? ((tt - ta) / dt) : 0.0;

  const double va = sample(i0, component);
  const double vb = sample(i1, component);
  return (1.0 - alpha) * va + alpha * vb;
}

std::shared_ptr<TemporalValues>
readTemporalValuesFile(const std::string& file_path,
                       int num_components,
                       TemporalEndBehavior end_behavior)
{
  if (num_components <= 0) {
    throw std::runtime_error("[svMultiPhysics::Physics] Temporal BC file reader requires at least one component.");
  }

  std::ifstream in(file_path);
  if (!in.is_open()) {
    throw std::runtime_error("[svMultiPhysics::Physics] Failed to open temporal BC file '" + file_path + "'.");
  }

  int num_points = 0;
  int legacy_count = 0;
  in >> num_points >> legacy_count;
  if (in.fail() || num_points < 2) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Temporal BC file '" + file_path +
        "' must start with '<num_time_points> <legacy_count>' and contain at least 2 points.");
  }

  auto out = std::make_shared<TemporalValues>();
  out->num_time_points = num_points;
  out->num_components = num_components;
  out->end_behavior = end_behavior;
  out->t.resize(static_cast<std::size_t>(num_points));
  out->v.resize(static_cast<std::size_t>(num_points) * static_cast<std::size_t>(num_components));

  for (int i = 0; i < num_points; ++i) {
    double ti = 0.0;
    in >> ti;
    if (in.fail() || !std::isfinite(ti)) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Failed to read time for data point " + std::to_string(i) +
          " from temporal BC file '" + file_path + "'.");
    }
    if (i > 0 && !(ti > out->t[static_cast<std::size_t>(i - 1)])) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Temporal BC file '" + file_path +
          "' must have strictly increasing time values.");
    }
    out->t[static_cast<std::size_t>(i)] = ti;

    for (int c = 0; c < num_components; ++c) {
      double value = 0.0;
      in >> value;
      if (in.fail() || !std::isfinite(value)) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Failed to read component " + std::to_string(c) +
            " for data point " + std::to_string(i) +
            " from temporal BC file '" + file_path + "'.");
      }
      out->v[static_cast<std::size_t>(i) * static_cast<std::size_t>(num_components) +
             static_cast<std::size_t>(c)] = value;
    }
  }

  if (!(out->period() > 0.0)) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Temporal BC file '" + file_path + "' has non-positive time span.");
  }

  return out;
}

} // namespace Physics
} // namespace svmp
