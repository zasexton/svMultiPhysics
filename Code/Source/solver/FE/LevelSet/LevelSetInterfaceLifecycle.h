#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief Generated interface domain lifecycle for level-set fields.
 */

#include "Core/Types.h"
#include "Interfaces/LevelSetInterfaceDomain.h"
#include "Systems/FESystem.h"

#include <cstdint>
#include <span>
#include <string>

namespace svmp::FE::level_set {

struct LevelSetGeneratedInterfaceOptions {
    std::string level_set_field_name{"level_set"};
    std::string domain_id{"free_surface"};
    int requested_interface_marker{-1};
    Real isovalue{0.0};
    Real tolerance{1.0e-12};
    int quadrature_order{1};
    bool keep_degenerate_fragments{false};
};

struct LevelSetGeneratedInterfaceResult {
    bool success{false};
    int interface_marker{-1};
    std::uint64_t value_revision{0};
    interfaces::LevelSetInterfaceDomain domain{};
    interfaces::CutInterfaceDomainSummary summary{};
    std::string diagnostic{};
};

class LevelSetGeneratedInterfaceLifecycle {
public:
    explicit LevelSetGeneratedInterfaceLifecycle(int marker_base = 1000000,
                                                 int marker_range = 1000000);

    [[nodiscard]] LevelSetGeneratedInterfaceResult build(
        const systems::FESystem& system,
        const LevelSetGeneratedInterfaceOptions& options,
        std::span<const Real> solution);

    [[nodiscard]] std::uint64_t valueRevision() const noexcept { return value_revision_; }
    void restoreValueRevision(std::uint64_t value_revision) noexcept;

private:
    interfaces::GeneratedInterfaceMarkerRegistry marker_registry_;
    std::uint64_t value_revision_{0};
};

} // namespace svmp::FE::level_set
