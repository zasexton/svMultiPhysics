/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/CouplingContext.h"

namespace svmp {
namespace FE {
namespace coupling {

bool CouplingParticipantRef::valid() const noexcept
{
    return !participant_name.empty() && !system_name.empty() && system != nullptr;
}

bool CouplingFieldRef::valid() const noexcept
{
    return !participant_name.empty() && !system_name.empty() && system != nullptr &&
           !field_name.empty() && field_id != INVALID_FIELD_ID && components > 0;
}

bool CouplingRegionRef::valid() const noexcept
{
    return !participant_name.empty() && !system_name.empty() && system != nullptr &&
           !region_name.empty();
}

const std::vector<CouplingParticipantRef>& CouplingContext::participants() const noexcept
{
    return participants_;
}

const std::vector<CouplingFieldRef>& CouplingContext::fields() const noexcept
{
    return fields_;
}

const std::vector<CouplingRegionRef>& CouplingContext::regions() const noexcept
{
    return regions_;
}

const std::vector<SharedRegionRef>& CouplingContext::sharedRegions() const noexcept
{
    return shared_regions_;
}

} // namespace coupling
} // namespace FE
} // namespace svmp
