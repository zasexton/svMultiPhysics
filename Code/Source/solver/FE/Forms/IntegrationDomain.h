#pragma once

/**
 * @file IntegrationDomain.h
 * @brief Typed integration domains for the measure methods of FormExpr
 *
 * Each domain names a set the form language already integrates over and
 * lowers to the existing measure, so the IR, the kernel grouping and the JIT
 * cache keys are those of the untyped call:
 *
 * | Domain                                       | Measure          | Lowers to                         |
 * |----------------------------------------------|------------------|-----------------------------------|
 * | VolumeDomain::full()                         | expr.dx(domain)  | expr.dx()                         |
 * | VolumeDomain::cut(m, side)                   | expr.dx(domain)  | expr.dCutVolume(m, side)          |
 * | InterfaceDomain(m)                           | expr.dI(domain)  | expr.dI(m)                        |
 * | ContactLineDomain(m, wall)                   | expr.dL(domain)  | expr.dI(m) (codimension-two rule) |
 * | BoundaryDomain::fullPhysical(b)              | expr.ds(domain)  | expr.dExteriorBoundary(domain)    |
 * | BoundaryDomain::generatedActiveSubset(b, g)  | expr.ds(domain)  | expr.dExteriorBoundary(domain)    |
 * | FacetDomain::cutAdjacent(m[, side])          | expr.dS(domain)  | expr.dS(m[, side])                |
 *
 * The domains carry markers only.  The physics-agnostic handles that compute
 * the generated markers of a level-set geometry live in
 * FE/LevelSet/LevelSetGeometry.h.
 */

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
    /// The whole mesh: lowers to dx().
    [[nodiscard]] static VolumeIntegrationDomain full() noexcept { return fullVolume(); }

    /// One side of the cut by the embedded interface @p interface_marker:
    /// lowers to dCutVolume(interface_marker, side).
    [[nodiscard]] static VolumeIntegrationDomain cut(int interface_marker, CutVolumeSide side)
    {
        return cutVolume(interface_marker, side);
    }

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

using VolumeDomain = VolumeIntegrationDomain;

/// An embedded (registered or generated) interface: lowers to dI(marker).
class InterfaceDomain {
public:
    explicit InterfaceDomain(int interface_marker)
        : marker_(interface_marker)
    {
        if (interface_marker < 0) {
            throw std::invalid_argument(
                "InterfaceDomain requires a nonnegative interface marker");
        }
    }

    [[nodiscard]] int marker() const noexcept { return marker_; }

    [[nodiscard]] friend bool operator==(const InterfaceDomain&, const InterfaceDomain&) noexcept = default;

private:
    int marker_;
};

/**
 * A codimension-two contact line: the generated intersection of an interface
 * with the physical boundary @p wall_boundary_marker.  The intersection is
 * imported into the cut-integration context as a generated rule with its own
 * marker and geometric dimension d-2, so it lowers to dI(intersection_marker),
 * exactly as dInterfaceBoundary() does.
 */
class ContactLineDomain {
public:
    explicit ContactLineDomain(int intersection_marker, int wall_boundary_marker = -1)
        : marker_(intersection_marker)
        , wall_boundary_marker_(wall_boundary_marker)
    {
        if (intersection_marker < 0) {
            throw std::invalid_argument(
                "ContactLineDomain requires a nonnegative intersection marker");
        }
    }

    [[nodiscard]] int marker() const noexcept { return marker_; }
    [[nodiscard]] int wallBoundaryMarker() const noexcept { return wall_boundary_marker_; }

    [[nodiscard]] friend bool operator==(const ContactLineDomain&, const ContactLineDomain&) noexcept = default;

private:
    int marker_;
    int wall_boundary_marker_;
};

/// A physical exterior boundary, whole or restricted to a generated active
/// subset: lowers to dExteriorBoundary(domain).
using BoundaryDomain = ExteriorBoundaryMeasure;

/// A marker-backed set of interior facets, optionally restricted to the
/// facets adjacent to one side of a cut: lowers to dS(marker[, side]), as
/// cutAdjacentFacetIntegral() does.
class FacetDomain {
public:
    [[nodiscard]] static FacetDomain cutAdjacent(int facet_set_marker)
    {
        return FacetDomain(facet_set_marker, std::nullopt);
    }

    [[nodiscard]] static FacetDomain cutAdjacent(int facet_set_marker, CutVolumeSide side)
    {
        return FacetDomain(facet_set_marker, side);
    }

    [[nodiscard]] int marker() const noexcept { return marker_; }
    [[nodiscard]] std::optional<CutVolumeSide> side() const noexcept { return side_; }

    [[nodiscard]] friend bool operator==(const FacetDomain&, const FacetDomain&) noexcept = default;

private:
    FacetDomain(int facet_set_marker, std::optional<CutVolumeSide> side)
        : marker_(facet_set_marker)
        , side_(side)
    {
        if (facet_set_marker < 0) {
            throw std::invalid_argument(
                "cutAdjacentFacetIntegral requires a nonnegative cut-adjacent facet-set marker");
        }
    }

    int marker_;
    std::optional<CutVolumeSide> side_;
};

[[nodiscard]] inline FormExpr integrate(
    const FormExpr& integrand,
    const VolumeIntegrationDomain& domain)
{
    return integrand.dx(domain);
}

} // namespace forms
} // namespace FE
} // namespace svmp
