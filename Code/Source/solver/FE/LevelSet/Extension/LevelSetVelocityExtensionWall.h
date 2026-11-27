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
 * @brief The wall data of a velocity extension into the inactive part of an
 *        unfitted level-set mesh: the strong wall conditions the extension
 *        keeps on inactive boundary vertices, and the largest admitted active-to-inactive
 *        speed amplification.
 *
 * Moved from Application/Core/LevelSetVelocityExtensionMap.h.
 */

#include "Mesh/Core/MeshTypes.h"

#include <array>

namespace svmp {
namespace FE {
namespace level_set {

inline constexpr double kVelocityExtensionMaxActiveToInactiveAmplification = 16.0;

// The velocity components a strong Dirichlet condition constrains on one
// wall.  On the inactive vertices of the wall the extension holds them at zero
// for a homogeneous condition where it would otherwise flow in through the
// wall (extendVelocityByPde), and, for a wall whose velocity is prescribed
// (any value: a moving wall, the exact velocity of a manufactured solution),
// keeps them as the source velocity, which carries the prescribed value
// there.
struct BoundaryVelocityExtensionConstraint {
  svmp::label_t boundary_label{svmp::INVALID_LABEL};
  std::array<bool, 3> constrained_components{false, false, false};
  bool project_boundary_normal{false};
  bool prescribed_velocity{false};
};

} // namespace level_set
} // namespace FE
} // namespace svmp
