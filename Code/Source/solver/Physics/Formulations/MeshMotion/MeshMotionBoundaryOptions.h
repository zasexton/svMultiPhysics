/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_FORMULATIONS_MESHMOTION_MESH_MOTION_BOUNDARY_OPTIONS_H
#define SVMP_PHYSICS_FORMULATIONS_MESHMOTION_MESH_MOTION_BOUNDARY_OPTIONS_H

/**
 * @file MeshMotionBoundaryOptions.h
 * @brief Shared boundary option types for mesh-motion formulations.
 */

#include "FE/Forms/BoundaryConditions.h"

#include <cstdint>

namespace svmp {
namespace Physics {
namespace formulations {
namespace mesh_motion {

enum class NormalConstraintQuantity : std::uint8_t {
    Displacement,
    Velocity
};

struct NormalConstraintBC {
    using ScalarValue = FE::forms::bc::ScalarValue;

    int boundary_marker{-1};
    NormalConstraintQuantity quantity{NormalConstraintQuantity::Displacement};
    ScalarValue target{0.0};
    ScalarValue penalty{1.0};
    ScalarValue velocity_time_scale{1.0};
};

} // namespace mesh_motion
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_MESHMOTION_MESH_MOTION_BOUNDARY_OPTIONS_H
