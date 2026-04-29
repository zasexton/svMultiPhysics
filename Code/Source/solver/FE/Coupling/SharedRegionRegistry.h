/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_COUPLING_SHAREDREGIONREGISTRY_H
#define SVMP_FE_COUPLING_SHAREDREGIONREGISTRY_H

/**
 * @file SharedRegionRegistry.h
 * @brief Shared-region declarations reused by multiple coupling contracts.
 */

#include "Coupling/CouplingContext.h"
#include "Coupling/CouplingDiagnostics.h"

#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace coupling {

class SharedRegionRegistry {
public:
    void add(SharedRegionRef region);

    [[nodiscard]] const SharedRegionRef* find(std::string_view name) const noexcept;
    [[nodiscard]] const CouplingRegionRef* findParticipantRegion(
        std::string_view name,
        std::string_view participant) const noexcept;

    [[nodiscard]] CouplingValidationResult validate() const;
    [[nodiscard]] const std::vector<SharedRegionRef>& records() const noexcept;

private:
    std::vector<SharedRegionRef> records_{};
};

} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_COUPLING_SHAREDREGIONREGISTRY_H
