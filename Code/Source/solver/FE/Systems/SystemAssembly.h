/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_SYSTEMASSEMBLY_H
#define SVMP_FE_SYSTEMS_SYSTEMASSEMBLY_H

#include "Core/Types.h"

namespace svmp {
namespace FE {

namespace assembly {
struct AssemblyResult;
class GlobalSystemView;
}

namespace systems {

struct AssemblyRequest;
struct SystemStateView;
class FESystem;

assembly::AssemblyResult assembleOperator(
    FESystem& system,
    const AssemblyRequest& request,
    const SystemStateView& state,
    assembly::GlobalSystemView* matrix_out,
    assembly::GlobalSystemView* vector_out);

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_SYSTEMASSEMBLY_H
