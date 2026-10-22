/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#pragma once

/**
 * @file LevelSetVelocityExtensionWall.h
 * @ingroup fe_level_set
 * @brief The wall data of a velocity extension into the dry part of an
 *        unfitted level-set mesh: the strong wall conditions the extension
 *        keeps on dry wall vertices, and the largest admitted wet-to-dry
 *        speed amplification.
 *
 * Moved from Application/Core/LevelSetVelocityExtensionMap.h unchanged.
 */

#include "Mesh/Core/MeshTypes.h"

#include <array>

namespace svmp {
namespace FE {
namespace level_set {

inline constexpr double kVelocityExtensionMaxWetToDryAmplification = 16.0;

struct WallVelocityExtensionConstraint {
  svmp::label_t boundary_label{svmp::INVALID_LABEL};
  std::array<bool, 3> constrained_components{false, false, false};
  bool project_boundary_normal{false};
};

} // namespace level_set
} // namespace FE
} // namespace svmp
