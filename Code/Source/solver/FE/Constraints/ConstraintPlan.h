/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_CONSTRAINTPLAN_H
#define SVMP_FE_CONSTRAINTS_CONSTRAINTPLAN_H

/**
 * @file ConstraintPlan.h
 * @brief Phase-ordered registration of a formulation's system constraints
 *
 * The order in which system constraints are registered is load-bearing: a
 * later constraint only fills degrees of freedom an earlier one left free
 * (per-DOF precedence), and small-cut aggregation must see every strong
 * condition before it builds its lines.  A ConstraintPlan lets a formulation
 * declare its constraints where it states them and registers them in the
 * fixed phase order
 *
 *   VertexValues < Essential < InactiveSupport < Aggregation < PointLaw
 *
 * and, within a phase, in declaration order:
 *
 * 1. VertexValues: prescribed values at mesh vertices (pressure pins).
 * 2. Essential: strong boundary conditions.  These are usually registered
 *    by a BoundaryConditionManager together with their weak terms, so the
 *    plan runs them through essentials(install) at the point of the call.
 * 3. InactiveSupport: degrees of freedom outside the active side of a cut
 *    are pinned, where no physical condition owns them, and so are those of
 *    a trace multiplier off its boundary or interface (restrictSupport).
 * 4. Aggregation: small-cut aggregation of the active side.
 * 5. PointLaw: point conditions that expand aggregated degrees of freedom
 *    (the strong contact-point velocity law).
 *
 * Declarations are held until the plan reaches their phase: commitThrough(p)
 * registers every held declaration of the phases up to p (one
 * FESystem::addSystemConstraint per declaration) and moves the plan to p;
 * essentials() first commits the phases before Essential.  Declaring into a
 * phase the plan has already passed throws std::logic_error, so the
 * registration order cannot be inverted by accident.  Gauge anchors are not
 * constraints of the plan; they go to the GaugeRegistry.  Declarations still
 * held when the plan is destroyed are dropped (a formulation that failed
 * before commit() leaves them unregistered).
 */

#include "Constraints/ContactPointVelocityConstraint.h"
#include "Constraints/InterfaceMeasureGaugeAnchor.h"
#include "Constraints/SmallCutAggregationConstraint.h"
#include "Constraints/SystemConstraint.h"
#include "Constraints/VertexDirichletConstraint.h"
#include "Core/Types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {

namespace systems {
class BoundaryConditionManager;
class FESystem;
}

namespace forms {
class FormExpr;
class ExteriorBoundaryMeasure;
class InterfaceDomain;
namespace bc {
struct StrongDirichlet;
}
}

namespace level_set {
class CutSide;
class LevelSetGeometry;
}

namespace constraints {

enum class ConstraintPhase : std::uint8_t {
    VertexValues = 0,
    Essential = 1,
    InactiveSupport = 2,
    Aggregation = 3,
    PointLaw = 4,
};

inline constexpr std::size_t kConstraintPhaseCount = 5u;

[[nodiscard]] const char* constraintPhaseName(ConstraintPhase phase) noexcept;

/// The unknowns of ConstraintPlan::activeOn on one side of a cut.
struct ActiveSideOptions {
    /// Aggregate the small cut cells of the side.
    bool aggregate{true};
    SmallCutAggregationGuardOptions guards{};
    /// The value of the pinned degrees of freedom off the side.
    Real inactive_value{Real{0.0}};
};

class ConstraintPlan {
public:
    explicit ConstraintPlan(systems::FESystem& system);
    ~ConstraintPlan();

    ConstraintPlan(const ConstraintPlan&) = delete;
    ConstraintPlan& operator=(const ConstraintPlan&) = delete;
    ConstraintPlan(ConstraintPlan&&) = delete;
    ConstraintPlan& operator=(ConstraintPlan&&) = delete;

    /// Holds @p constraint for registration in @p phase.
    ConstraintPlan& declare(ConstraintPhase phase, std::unique_ptr<ISystemConstraint> constraint);

    /// VertexValues: VertexDirichletConstraint(field, values, mode).
    ConstraintPlan& vertexValues(FieldId field,
                                 std::vector<VertexDirichletValue> values,
                                 VertexIdMode mode = VertexIdMode::GlobalVertexGid);

    /**
     * Essential: commits the earlier phases, then runs @p install, which
     * registers essential constraints with the system directly (for example
     * BoundaryConditionManager::applyAll or installStrongDirichlet), and
     * leaves the plan in the Essential phase.
     */
    ConstraintPlan& essentials(const std::function<void(systems::FESystem&)>& install);

    /// Essential: immediately apply the manager's full single-field BC route.
    ConstraintPlan& essentials(systems::BoundaryConditionManager& manager,
                               forms::FormExpr& residual,
                               const forms::FormExpr& state,
                               const forms::FormExpr& test,
                               FieldId field,
                               std::string_view operator_tag = {});

    /// Essential: immediately apply a manager containing only strong/algebraic BCs.
    ConstraintPlan& essentials(systems::BoundaryConditionManager& manager, FieldId field);

    /// Essential: install declarations immediately, retaining per-declaration progress.
    ConstraintPlan& essentials(std::span<const forms::bc::StrongDirichlet> declarations);

    /**
     * InactiveSupport: for each field, pins the vertices of the inactive
     * side to @p value: LevelSetActiveSideVertexDirichletConstraint(field,
     * level-set field name, side, isovalue, value, interface marker).
     */
    ConstraintPlan& inactiveSide(std::initializer_list<FieldId> fields,
                                 const level_set::CutSide& active_side,
                                 Real value = Real{0.0});

    /**
     * InactiveSupport: a trace multiplier written as a volume field (D54 T1).
     * For each field, SupportRestrictionConstraint(field, domain) pins the
     * degrees of freedom whose basis function vanishes on the domain: off the
     * closure of a physical boundary (`BoundaryDomain::fullPhysical(b)`) or of
     * a fitted interface (an InterfaceDomain with a registered InterfaceMesh).
     */
    ConstraintPlan& restrictSupport(std::initializer_list<FieldId> fields,
                                    const forms::ExteriorBoundaryMeasure& domain);
    ConstraintPlan& restrictSupport(std::initializer_list<FieldId> fields,
                                    const forms::InterfaceDomain& domain);

    /**
     * InactiveSupport: for each field, SupportRestrictionConstraint on the
     * generated interface of @p geometry: the field lives on the cells the
     * interface cuts, and the pinned set follows the level set (a new cut
     * epoch rebuilds it without a new setup).
     */
    ConstraintPlan& restrictSupport(std::initializer_list<FieldId> fields,
                                    const level_set::LevelSetGeometry& geometry);

    /**
     * Aggregation: for each field, SmallCutAggregationConstraint(field,
     * side, interface marker, no excluded markers or vertices, guards).
     */
    ConstraintPlan& aggregate(std::initializer_list<FieldId> fields,
                              const level_set::CutSide& active_side,
                              SmallCutAggregationGuardOptions guards = {});

    /**
     * The unknowns @p fields live on @p active_side: inactiveSide(fields,
     * side, inactive value) and, with options.aggregate, aggregate(fields,
     * side, guards), the same declarations in the same phases and order as
     * the two calls.  Aggregation also requires the cut to keep both cut
     * volumes (LevelSetGeometry::requireBothSidesRetained), which a side of
     * cutGeometry() records for its cut; the cut-volume request of the
     * Application states the retention itself.
     */
    ConstraintPlan& activeOn(std::initializer_list<FieldId> fields,
                             const level_set::CutSide& active_side,
                             ActiveSideOptions options = {});

    /// PointLaw: ContactPointVelocityConstraint(velocity, law).
    ConstraintPlan& contactPointVelocity(FieldId velocity, ContactPointVelocityLaw law);

    /**
     * PointLaw: the contact-point velocity law where the interface of
     * @p geometry meets the wall @p wall_marker (unit @p wall_normal out of
     * the fluid): the law reads the level set of the geometry at its
     * isovalue, and @p contact_speed maps cos(theta) of a contact cell to
     * the outward wall-tangential contact-point speed.
     */
    ConstraintPlan& contactPointVelocity(FieldId velocity,
                                         const level_set::LevelSetGeometry& geometry,
                                         int wall_marker,
                                         const std::array<Real, 3>& wall_normal,
                                         std::function<Real(Real)> contact_speed,
                                         std::string name);

    /**
     * Not a phase: records the gauge anchor of a natural condition on a
     * generated interface and adds its measure checks at once
     * (anchorGaugeOnInterfaceMeasure on @p operator_tag).
     */
    ConstraintPlan& anchorGauge(const std::string& operator_tag, const InterfaceMeasureGaugeAnchor& anchor);

    /// Registers the held declarations of every phase up to @p through, in
    /// phase and declaration order, and moves the plan to @p through.
    ConstraintPlan& commitThrough(ConstraintPhase through);

    /// commitThrough(ConstraintPhase::PointLaw).
    void commit();

    /// The phase the plan has reached (declarations before it are rejected).
    [[nodiscard]] ConstraintPhase phase() const noexcept { return phase_; }

    /// Number of declarations held for later registration.
    [[nodiscard]] std::size_t pendingCount() const noexcept;

    /// Number of declarations the plan has registered.
    [[nodiscard]] std::size_t registeredCount() const noexcept { return registered_; }

private:
    void requireNotPassed(ConstraintPhase phase, const char* operation) const;

    systems::FESystem* system_{nullptr};
    ConstraintPhase phase_{ConstraintPhase::VertexValues};
    std::array<std::vector<std::unique_ptr<ISystemConstraint>>, kConstraintPhaseCount> held_{};
    std::size_t registered_{0u};
};

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_CONSTRAINTPLAN_H
