#pragma once

#include "FormExpr.h"

#include <cstdint>
#include <optional>
#include <stdexcept>

namespace svmp {
namespace FE {
namespace forms {

enum class VolumeIntegrationScope : std::uint8_t {
    FullVolume,
    CutVolume
};

class VolumeIntegrationDomain {
public:
    [[nodiscard]] static VolumeIntegrationDomain fullVolume() noexcept
    {
        return VolumeIntegrationDomain(
            VolumeIntegrationScope::FullVolume,
            -1,
            CutVolumeSide::Negative);
    }

    [[nodiscard]] static VolumeIntegrationDomain cutVolume(
        int interface_marker,
        CutVolumeSide side)
    {
        if (interface_marker < 0) {
            throw std::invalid_argument(
                "VolumeIntegrationDomain::cutVolume requires a nonnegative interface marker");
        }
        if (side != CutVolumeSide::Negative &&
            side != CutVolumeSide::Positive) {
            throw std::invalid_argument(
                "VolumeIntegrationDomain::cutVolume requires an explicit cut-volume side");
        }
        return VolumeIntegrationDomain(
            VolumeIntegrationScope::CutVolume,
            interface_marker,
            side);
    }

    [[nodiscard]] VolumeIntegrationScope scope() const noexcept
    {
        return scope_;
    }

    [[nodiscard]] std::optional<int> interfaceMarker() const noexcept
    {
        if (scope_ == VolumeIntegrationScope::FullVolume) {
            return std::nullopt;
        }
        return interface_marker_;
    }

    [[nodiscard]] std::optional<CutVolumeSide> side() const noexcept
    {
        if (scope_ == VolumeIntegrationScope::FullVolume) {
            return std::nullopt;
        }
        return side_;
    }

    [[nodiscard]] friend bool operator==(
        const VolumeIntegrationDomain&,
        const VolumeIntegrationDomain&) noexcept = default;

private:
    VolumeIntegrationDomain(
        VolumeIntegrationScope scope,
        int interface_marker,
        CutVolumeSide side) noexcept
        : scope_(scope)
        , interface_marker_(interface_marker)
        , side_(side)
    {
    }

    VolumeIntegrationScope scope_;
    int interface_marker_;
    CutVolumeSide side_;
};

[[nodiscard]] inline FormExpr integrate(
    const FormExpr& integrand,
    const VolumeIntegrationDomain& domain)
{
    if (domain.scope() == VolumeIntegrationScope::FullVolume) {
        return integrand.dx();
    }
    return integrand.dCutVolume(
        *domain.interfaceMarker(),
        *domain.side());
}

} // namespace forms
} // namespace FE
} // namespace svmp
