/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_SNAPSHOTFIELDACCESS_H
#define SVMP_FE_SYSTEMS_SNAPSHOTFIELDACCESS_H

/**
 * @file SnapshotFieldAccess.h
 * @brief The snapshot field evaluators' view of a field of an FE system.
 */

#include "Core/Types.h"
#include "Interfaces/SnapshotFieldEvaluator.h"

namespace svmp {
namespace FE {
namespace systems {

class FESystem;

/**
 * @brief The SnapshotFieldAccess of field @p field of @p system.
 *
 * The field record and DOF offset are read now (FESystem::fieldRecord,
 * fieldDofOffset); the mesh and the cell DOFs at every use
 * (FESystem::meshAccess, fieldDofHandler).  Valid while @p system is.
 */
[[nodiscard]] interfaces::SnapshotFieldAccess snapshotFieldAccess(
    const FESystem& system,
    FieldId field);

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_SNAPSHOTFIELDACCESS_H
