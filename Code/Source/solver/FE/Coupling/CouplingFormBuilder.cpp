/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/CouplingFormBuilder.h"

namespace svmp {
namespace FE {
namespace coupling {

CouplingFormBuilder::CouplingFormBuilder(const CouplingContext& context)
    : context_(&context)
{
}

const CouplingContext& CouplingFormBuilder::context() const noexcept
{
    return *context_;
}

} // namespace coupling
} // namespace FE
} // namespace svmp
