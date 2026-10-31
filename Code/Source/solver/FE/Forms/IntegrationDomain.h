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
 * | VolumeDomain::cells(k)                       | expr.dx(domain)  | (eq(domainId(), k) * expr).dx()   |
 * | InterfaceDomain(m)                           | expr.dI(domain)  | expr.dI(m)                        |
 * | ContactLineDomain(m, wall)                   | expr.dL(domain)  | expr.dI(m) (codimension-two rule) |
 * | BoundaryDomain::fullPhysical(b)              | expr.ds(domain)  | expr.dExteriorBoundary(domain)    |
 * | BoundaryDomain::generatedActiveSubset(b, g)  | expr.ds(domain)  | expr.dExteriorBoundary(domain)    |
 * | FacetDomain::cutAdjacent(m[, side])          | expr.dS(domain)  | expr.dS(m[, side])                |
 *
 * The domains carry markers only.  The physics-agnostic handles that compute
 * the generated markers of a level-set geometry live in
 * FE/LevelSet/LevelSetGeometry.h.
 *
 * VolumeDomain::cells(k) is the fitted subdomain of the cells whose domain id
 * (the mesh region label, domainId()) is k.  Its measure is the cell
 * indicator of the subdomain times the integrand, so it lowers to the IR,
 * the JIT kernels and the cache keys of that indicator form; the cells of
 * other subdomains integrate zero, and a field scoped to the subdomain
 * (FieldSpec::participant_domain_id on a single mesh) has no degree of
 * freedom there.  InterfaceBetween names the sides of a fitted interface by
 * subdomain (side(), lowering to minus()/plus(); systems::interfaceBetween
 * reads the orientation from the interface mesh).
 */

#include "FormExpr.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

namespace svmp {
namespace FE {
namespace forms {

enum class VolumeIntegrationScope : std::uint8_t {
    FullVolume,
    CutVolume,
    CellDomain
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

    /// The cells whose domain id (domainId(), the mesh region label) is
    /// @p domain_id: lowers to (eq(domainId(), domain_id) * expr).dx().
    [[nodiscard]] static VolumeIntegrationDomain cells(int domain_id)
    {
        if (domain_id < 0) {
            throw std::invalid_argument(
                "VolumeIntegrationDomain::cells requires a nonnegative domain id");
        }
        return VolumeIntegrationDomain(VolumeIntegrationScope::CellDomain, domain_id, CutVolumeSide::Negative);
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
        if (scope_ != VolumeIntegrationScope::CutVolume) {
            return std::nullopt;
        }
        return interface_marker_;
    }

    [[nodiscard]] std::optional<CutVolumeSide> side() const noexcept
    {
        if (scope_ != VolumeIntegrationScope::CutVolume) {
            return std::nullopt;
        }
        return side_;
    }

    /// The domain id of a cells() domain.
    [[nodiscard]] std::optional<int> domainId() const noexcept
    {
        if (scope_ != VolumeIntegrationScope::CellDomain) {
            return std::nullopt;
        }
        return interface_marker_;
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
    int interface_marker_; ///< the interface marker, or the domain id of cells()
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

/**
 * A fitted interface between the subdomains cells(minus) and cells(plus):
 * the interface marker and the subdomain on each side of its faces.  side()
 * names a trace by subdomain; systems::interfaceBetween builds it from the
 * orientation of the interface mesh.
 */
class InterfaceBetween {
public:
    InterfaceBetween(int interface_marker, int minus_domain_id, int plus_domain_id)
        : marker_(interface_marker)
        , minus_(minus_domain_id)
        , plus_(plus_domain_id)
    {
        if (interface_marker < 0 || minus_domain_id < 0 || plus_domain_id < 0 ||
            minus_domain_id == plus_domain_id) {
            throw std::invalid_argument(
                "InterfaceBetween requires a nonnegative marker and two distinct nonnegative domain ids");
        }
    }

    [[nodiscard]] int marker() const noexcept { return marker_; }
    [[nodiscard]] int minusDomainId() const noexcept { return minus_; }
    [[nodiscard]] int plusDomainId() const noexcept { return plus_; }
    /// The interface as an integration domain: dI(marker()).
    [[nodiscard]] InterfaceDomain domain() const { return InterfaceDomain(marker_); }

    [[nodiscard]] friend bool operator==(const InterfaceBetween&, const InterfaceBetween&) noexcept = default;

private:
    int marker_;
    int minus_;
    int plus_;
};

/// The trace of @p expr on @p interface from the side of @p subdomain (a
/// cells() domain): expr.minus() or expr.plus().  Throws
/// std::invalid_argument for a subdomain on neither side.
[[nodiscard]] inline FormExpr side(const FormExpr& expr,
                                   const InterfaceBetween& interface,
                                   const VolumeIntegrationDomain& subdomain)
{
    const auto id = subdomain.domainId();
    if (id && *id == interface.minusDomainId()) {
        return expr.minus();
    }
    if (id && *id == interface.plusDomainId()) {
        return expr.plus();
    }
    throw std::invalid_argument("side(): the subdomain is on neither side of interface marker " +
                                std::to_string(interface.marker()));
}

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
                "FacetDomain::cutAdjacent requires a nonnegative cut-adjacent facet-set marker");
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
