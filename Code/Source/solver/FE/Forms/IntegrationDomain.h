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
 * freedom there.  InterfaceBetween names the sides of an interface by their
 * volumes, a fitted interface by subdomain (systems::interfaceBetween reads
 * the orientation from the interface mesh) and a generated one by cut side:
 * trace() lowers to minus()/plus(), normalFrom() to the oriented normal.
 */

#include "FormExpr.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

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

/// A weight attached to one side of an interface, named by the side's volume:
/// the arguments of InterfaceBetween::average().
struct SideWeight {
    SideWeight(VolumeIntegrationDomain side_volume, FormExpr side_weight)
        : side(side_volume)
        , weight(std::move(side_weight))
    {
    }
    SideWeight(VolumeIntegrationDomain side_volume, Real side_weight)
        : side(side_volume)
        , weight(FormExpr::constant(side_weight))
    {
    }

    VolumeIntegrationDomain side;
    FormExpr weight;
};

/**
 * An interface between two volumes, with the volume on each side: a fitted
 * interface between the subdomains cells(minus) and cells(plus) (an
 * interface mesh; systems::interfaceBetween reads its orientation), or the
 * generated interface {phi = c} of a level set between its sides
 * cut(marker, Negative) and cut(marker, Positive) (generated()).
 *
 * trace(), normalFrom() and average() name a side by its volume, so a term
 * means the same on both kinds of interface whichever side the assembler
 * calls minus:
 * @code
 *   const auto bn  = inner(beta, I.normalFrom(A));
 *   const auto cUp = conditional(ge(bn, FormExpr::constant(0.0)), I.trace(c, A), I.trace(c, B));
 *   auto face = (bn * cUp * (I.trace(v, A) - I.trace(v, B))).dI(I.domain());
 * @endcode
 * Exchanging A and B gives the same term (normalFrom(B) is -normalFrom(A)).
 */
class InterfaceBetween {
public:
    /// The fitted interface @p interface_marker with the subdomain
    /// cells(@p minus_domain_id) on the minus side of its faces and
    /// cells(@p plus_domain_id) on the plus side.
    InterfaceBetween(int interface_marker, int minus_domain_id, int plus_domain_id)
        : InterfaceBetween(interface_marker, checkedCells(interface_marker, minus_domain_id, plus_domain_id, true),
                           checkedCells(interface_marker, minus_domain_id, plus_domain_id, false),
                           /*generated=*/false)
    {
    }

    /// The two-sided generated interface @p interface_marker of a level set:
    /// the minus side is cut(marker, Negative) (phi < c), the plus side
    /// cut(marker, Positive).  Its terms need both sides retained; the boundary
    /// of one retained side has the outward normal of level_set::CutSide.
    [[nodiscard]] static InterfaceBetween generated(int interface_marker)
    {
        if (interface_marker < 0) {
            throw std::invalid_argument("InterfaceBetween::generated requires a nonnegative interface marker");
        }
        return InterfaceBetween(interface_marker,
                                VolumeIntegrationDomain::cut(interface_marker, CutVolumeSide::Negative),
                                VolumeIntegrationDomain::cut(interface_marker, CutVolumeSide::Positive),
                                /*generated=*/true);
    }

    [[nodiscard]] int marker() const noexcept { return marker_; }
    /// The volume on the minus side: cells(k), or cut(marker, Negative).
    [[nodiscard]] const VolumeIntegrationDomain& minusSide() const noexcept { return minus_; }
    /// The volume on the plus side: cells(k), or cut(marker, Positive).
    [[nodiscard]] const VolumeIntegrationDomain& plusSide() const noexcept { return plus_; }
    [[nodiscard]] bool isGenerated() const noexcept { return generated_; }
    /// The interface as an integration domain: dI(marker()).
    [[nodiscard]] InterfaceDomain domain() const { return InterfaceDomain(marker_); }

    /// The side of the interface that is not @p side.
    [[nodiscard]] const VolumeIntegrationDomain& otherSide(const VolumeIntegrationDomain& side) const
    {
        return isMinus(side, "otherSide") ? plus_ : minus_;
    }

    /// The trace of @p expr from @p side: expr.minus() or expr.plus().
    [[nodiscard]] FormExpr trace(const FormExpr& expr, const VolumeIntegrationDomain& side) const
    {
        return isMinus(side, "trace") ? expr.minus() : expr.plus();
    }

    /**
     * The unit normal pointing out of @p side toward the other side.
     *
     * normal().minus() is the outward normal of the minus side on both kinds
     * of interface: of the minus cell on a fitted one, the level-set normal
     * grad(phi)/|grad(phi)| on a generated one, in a one-sided and in a
     * two-sided evaluation alike (D70).  So the minus side gets
     * normal().minus() and the plus side -normal().minus(); both read the
     * one normal, and normalFrom(otherSide(A)) is exactly -normalFrom(A).
     * The minus restriction fixes the direction also in a plus row.
     */
    [[nodiscard]] FormExpr normalFrom(const VolumeIntegrationDomain& side) const
    {
        const bool minus = isMinus(side, "normalFrom");
        const auto out_of_minus = FormExpr::normal().minus();
        return minus ? out_of_minus : -out_of_minus;
    }

    /**
     * The weighted average a.weight trace(expr, a.side) + b.weight
     * trace(expr, b.side) of two sides named in either order (Hansbo or
     * harmonic weights): average(k, {A, wA}, {B, wB}) and
     * average(k, {B, wB}, {A, wA}) are equal.
     */
    [[nodiscard]] FormExpr average(const FormExpr& expr, const SideWeight& a, const SideWeight& b) const
    {
        if (isMinus(a.side, "average") == isMinus(b.side, "average")) {
            throw std::invalid_argument("InterfaceBetween::average: both weights name the same side of interface " +
                                        std::to_string(marker_));
        }
        return a.weight * trace(expr, a.side) + b.weight * trace(expr, b.side);
    }

    [[nodiscard]] friend bool operator==(const InterfaceBetween&, const InterfaceBetween&) noexcept = default;

private:
    InterfaceBetween(int interface_marker,
                     VolumeIntegrationDomain minus_side,
                     VolumeIntegrationDomain plus_side,
                     bool generated)
        : marker_(interface_marker)
        , minus_(minus_side)
        , plus_(plus_side)
        , generated_(generated)
    {
    }

    [[nodiscard]] static VolumeIntegrationDomain checkedCells(int interface_marker,
                                                              int minus_domain_id,
                                                              int plus_domain_id,
                                                              bool minus)
    {
        if (interface_marker < 0 || minus_domain_id < 0 || plus_domain_id < 0 ||
            minus_domain_id == plus_domain_id) {
            throw std::invalid_argument(
                "InterfaceBetween requires a nonnegative marker and two distinct nonnegative domain ids");
        }
        return VolumeIntegrationDomain::cells(minus ? minus_domain_id : plus_domain_id);
    }

    [[nodiscard]] bool isMinus(const VolumeIntegrationDomain& side, const char* what) const
    {
        if (side == minus_) {
            return true;
        }
        if (side == plus_) {
            return false;
        }
        throw std::invalid_argument(std::string("InterfaceBetween::") + what +
                                    ": the volume is on neither side of interface marker " +
                                    std::to_string(marker_));
    }

    int marker_;
    VolumeIntegrationDomain minus_;
    VolumeIntegrationDomain plus_;
    bool generated_;
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
