/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_BOUNDARY_DOF_OWNER_COMPLETION_H
#define SVMP_FE_CONSTRAINTS_BOUNDARY_DOF_OWNER_COMPLETION_H

#include "Core/Types.h"

#include <cstddef>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {
class FESystem;
}
namespace constraints {

/**
 * @brief Give every boundary DOF owner the boundary DOFs that only other
 *        ranks can see.
 *
 * Strong boundary constraints collect the DOFs of the marker faces visited by
 * IMeshAccess::forEachBoundaryFace, which visits faces of owned cells only,
 * and constrain the DOFs they own.  A DOF on such a face can be owned by a
 * rank that holds the face's cell only as a ghost (or not at all); without
 * this step nobody constrains it and the result depends on the partition.
 *
 * Each rank sends the collected DOFs it does not own, with `stride` payload
 * values per DOF, to their owners; an owner adds every received DOF it does
 * not already have (the lowest source rank's payload wins).  The union over
 * ranks of the collected DOFs is the serial set, so the owned part of every
 * list equals the serial owned set on any partition and ghost depth.  Lists
 * whose owners already see every face are returned unchanged.
 *
 * @param dofs     sorted, unique system DOFs (updated in place)
 * @param payload  `stride` values per DOF in the order of `dofs`
 * @param context  names the constraint in the one-time diagnostic that rank 0
 *                 prints (WARNING) when any rank had to add DOFs
 * @return number of DOFs this rank added
 *
 * Collective over the system DOF communicator when it has more than one rank.
 */
std::size_t completeOwnedBoundaryDofs(const systems::FESystem& system,
                                      std::vector<GlobalIndex>& dofs,
                                      std::vector<Real>& payload,
                                      std::size_t stride,
                                      std::string_view context);

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_BOUNDARY_DOF_OWNER_COMPLETION_H
