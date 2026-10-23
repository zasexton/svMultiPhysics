/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/SnapshotFieldAccess.h"

#include "Assembly/Assembler.h"
#include "Dofs/DofHandler.h"
#include "Systems/FESystem.h"

namespace svmp {
namespace FE {
namespace systems {

interfaces::SnapshotFieldAccess snapshotFieldAccess(const FESystem& system, FieldId field)
{
    const auto& record = system.fieldRecord(field);
    interfaces::SnapshotFieldAccess access;
    access.space = record.space;
    access.components = record.components;
    access.dof_offset = system.fieldDofOffset(field);
    const FESystem* const fe_system = &system;
    access.mesh = [fe_system]() -> const assembly::IMeshAccess& {
        return fe_system->meshAccess();
    };
    access.cell_dofs = [fe_system, field](GlobalIndex cell) {
        return fe_system->fieldDofHandler(field).getCellDofs(cell);
    };
    return access;
}

} // namespace systems
} // namespace FE
} // namespace svmp
