/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/MortarSpace.h"

namespace svmp {
namespace FE {
namespace spaces {

MortarSpace::MortarSpace(std::shared_ptr<FunctionSpace> interface_space,
                         int interface_marker)
    : interface_space_(std::move(interface_space))
    , interface_marker_(interface_marker)
{
    FE_CHECK_NOT_NULL(interface_space_.get(), "MortarSpace interface_space");
}

} // namespace spaces
} // namespace FE
} // namespace svmp
