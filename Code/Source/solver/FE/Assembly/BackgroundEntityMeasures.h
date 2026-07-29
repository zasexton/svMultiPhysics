/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ASSEMBLY_BACKGROUNDENTITYMEASURES_H
#define SVMP_FE_ASSEMBLY_BACKGROUNDENTITYMEASURES_H

/**
 * @file BackgroundEntityMeasures.h
 * @brief Physical measures of a background cell and one parent face.
 */

#include "Core/Types.h"

namespace svmp {
namespace FE {
namespace assembly {

class IMeshAccess;

/**
 * Physical background-entity measures used by face and generated-boundary
 * assembly.
 */
struct BackgroundEntityMeasures {
    Real cell_diameter{0.0};
    Real physical_cell_measure{0.0};
    Real physical_parent_face_measure{0.0};
    Real h_normal{0.0};
};

/**
 * Compute physical measures for one locally visible cell and parent face.
 *
 * `parent_face_id` is the mesh-access face identifier accepted by
 * IMeshAccess::getLocalFaceIndex(). The integration order follows the
 * generated-boundary assembly policy: the maximum of the test-space,
 * trial-space, and cell-geometry polynomial orders is passed to
 * QuadratureFactory::recommended_order(..., false).
 */
[[nodiscard]] BackgroundEntityMeasures computeBackgroundEntityMeasures(
    const IMeshAccess& mesh,
    GlobalIndex cell_id,
    GlobalIndex parent_face_id,
    int test_polynomial_order,
    int trial_polynomial_order);

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_BACKGROUNDENTITYMEASURES_H
