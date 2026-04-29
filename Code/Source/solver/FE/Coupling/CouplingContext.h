/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_COUPLING_COUPLINGCONTEXT_H
#define SVMP_FE_COUPLING_COUPLINGCONTEXT_H

/**
 * @file CouplingContext.h
 * @brief Name-to-FE ownership records for coupling setup.
 */

#include "Core/Types.h"
#include "Coupling/CouplingTypes.h"
#include "Systems/FieldRegistry.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {

namespace spaces {
class FunctionSpace;
}

namespace systems {
class FESystem;
}

namespace coupling {

struct CouplingParticipantRef {
    std::string participant_name;
    std::string system_name;
    const systems::FESystem* system{nullptr};

    [[nodiscard]] bool valid() const noexcept;
};

struct CouplingFieldRef {
    std::string participant_name;
    std::string system_name;
    const systems::FESystem* system{nullptr};
    std::string field_name;
    FieldId field_id{INVALID_FIELD_ID};
    std::shared_ptr<const spaces::FunctionSpace> space;
    int components{1};
    systems::FieldScope scope{systems::FieldScope::VolumeCell};
    int interface_marker{-1};

    [[nodiscard]] bool valid() const noexcept;
};

struct CouplingRegionRef {
    std::string participant_name;
    std::string system_name;
    const systems::FESystem* system{nullptr};
    std::string region_name;
    CouplingRegionKind kind{CouplingRegionKind::UserDefined};
    int marker{-1};
    CouplingInterfaceSide side{CouplingInterfaceSide::None};
    CouplingCoordinateConfiguration coordinate_configuration{
        CouplingCoordinateConfiguration::Reference};
    std::uint64_t geometry_revision{0};
    std::uint64_t topology_revision{0};

    [[nodiscard]] bool valid() const noexcept;
};

struct SharedRegionRef {
    std::string name;
    std::optional<CouplingRegionKind> required_region_kind;
    std::vector<CouplingRegionRef> participant_regions;
};

class CouplingContext {
public:
    CouplingContext() = default;

    [[nodiscard]] const std::vector<CouplingParticipantRef>& participants() const noexcept;
    [[nodiscard]] const std::vector<CouplingFieldRef>& fields() const noexcept;
    [[nodiscard]] const std::vector<CouplingRegionRef>& regions() const noexcept;
    [[nodiscard]] const std::vector<SharedRegionRef>& sharedRegions() const noexcept;

private:
    std::vector<CouplingParticipantRef> participants_{};
    std::vector<CouplingFieldRef> fields_{};
    std::vector<CouplingRegionRef> regions_{};
    std::vector<SharedRegionRef> shared_regions_{};
};

} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_COUPLING_COUPLINGCONTEXT_H
