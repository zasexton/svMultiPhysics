#ifndef SVMP_PHYSICS_FORMULATIONS_LEVELSET_INTERFACE_LIFECYCLE_H
#define SVMP_PHYSICS_FORMULATIONS_LEVELSET_INTERFACE_LIFECYCLE_H

/**
 * @file LevelSetInterfaceLifecycle.h
 * @brief Physics-side orchestration for generated level-set interface domains.
 */

#include "Core/Types.h"
#include "FE/Systems/FESystem.h"
#include "Interfaces/LevelSetInterfaceDomain.h"

#include <cstdint>
#include <span>
#include <string>

namespace svmp {
namespace Physics {
namespace formulations {
namespace level_set {

struct LevelSetGeneratedInterfaceOptions {
    std::string level_set_field_name{"level_set"};
    std::string domain_id{"free_surface"};
    int requested_interface_marker{-1};
    FE::Real isovalue{0.0};
    FE::Real tolerance{1.0e-12};
    int quadrature_order{1};
    bool keep_degenerate_fragments{false};
};

struct LevelSetGeneratedInterfaceResult {
    bool success{false};
    int interface_marker{-1};
    std::uint64_t value_revision{0};
    FE::interfaces::LevelSetInterfaceDomain domain{};
    FE::interfaces::CutInterfaceDomainSummary summary{};
    std::string diagnostic{};
};

class LevelSetGeneratedInterfaceLifecycle {
public:
    explicit LevelSetGeneratedInterfaceLifecycle(int marker_base = 1000000,
                                                 int marker_range = 1000000);

    [[nodiscard]] LevelSetGeneratedInterfaceResult build(
        const FE::systems::FESystem& system,
        const LevelSetGeneratedInterfaceOptions& options,
        std::span<const FE::Real> solution);

    [[nodiscard]] std::uint64_t valueRevision() const noexcept { return value_revision_; }

private:
    FE::interfaces::GeneratedInterfaceMarkerRegistry marker_registry_;
    std::uint64_t value_revision_{0};
};

} // namespace level_set
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_LEVELSET_INTERFACE_LIFECYCLE_H
