/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Constraints/ConstraintPlan.h"

#include "Constraints/LevelSetActiveSideVertexDirichletConstraint.h"
#include "LevelSet/LevelSetGeometry.h"
#include "Systems/FESystem.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace svmp {
namespace FE {
namespace constraints {

namespace {

[[nodiscard]] std::size_t phaseIndex(ConstraintPhase phase) noexcept
{
    return static_cast<std::size_t>(phase);
}

[[nodiscard]] LevelSetConstraintSide levelSetConstraintSide(forms::CutVolumeSide side) noexcept
{
    return side == forms::CutVolumeSide::Positive ? LevelSetConstraintSide::Positive
                                                  : LevelSetConstraintSide::Negative;
}

[[nodiscard]] geometry::CutIntegrationSide cutIntegrationSide(forms::CutVolumeSide side) noexcept
{
    return side == forms::CutVolumeSide::Positive ? geometry::CutIntegrationSide::Positive
                                                  : geometry::CutIntegrationSide::Negative;
}

} // namespace

const char* constraintPhaseName(ConstraintPhase phase) noexcept
{
    switch (phase) {
    case ConstraintPhase::VertexValues:
        return "VertexValues";
    case ConstraintPhase::Essential:
        return "Essential";
    case ConstraintPhase::InactiveSupport:
        return "InactiveSupport";
    case ConstraintPhase::Aggregation:
        return "Aggregation";
    case ConstraintPhase::PointLaw:
        return "PointLaw";
    }
    return "Unknown";
}

ConstraintPlan::ConstraintPlan(systems::FESystem& system)
    : system_(&system)
{
}

ConstraintPlan::~ConstraintPlan() = default;

void ConstraintPlan::requireNotPassed(ConstraintPhase phase, const char* operation) const
{
    if (phaseIndex(phase) < phaseIndex(phase_)) {
        throw std::logic_error(
            std::string("ConstraintPlan::") + operation + ": phase " + constraintPhaseName(phase) +
            " comes before the phase the plan has reached (" + constraintPhaseName(phase_) +
            "); constraints are registered in the order VertexValues < Essential < "
            "InactiveSupport < Aggregation < PointLaw");
    }
}

ConstraintPlan& ConstraintPlan::declare(ConstraintPhase phase, std::unique_ptr<ISystemConstraint> constraint)
{
    if (!constraint) {
        throw std::invalid_argument("ConstraintPlan::declare: null constraint");
    }
    requireNotPassed(phase, "declare");
    held_[phaseIndex(phase)].push_back(std::move(constraint));
    return *this;
}

ConstraintPlan& ConstraintPlan::vertexValues(FieldId field,
                                             std::vector<VertexDirichletValue> values,
                                             VertexIdMode mode)
{
    return declare(ConstraintPhase::VertexValues,
                   std::make_unique<VertexDirichletConstraint>(field, std::move(values), mode));
}

ConstraintPlan& ConstraintPlan::essentials(const std::function<void(systems::FESystem&)>& install)
{
    if (!install) {
        throw std::invalid_argument("ConstraintPlan::essentials: empty installer");
    }
    requireNotPassed(ConstraintPhase::Essential, "essentials");
    commitThrough(ConstraintPhase::Essential);
    install(*system_);
    return *this;
}

ConstraintPlan& ConstraintPlan::inactiveSide(std::initializer_list<FieldId> fields,
                                             const level_set::CutSide& active_side,
                                             Real value)
{
    requireNotPassed(ConstraintPhase::InactiveSupport, "inactiveSide");
    const auto& geometry = active_side.geometry();
    const auto& level_set_name = system_->fieldRecord(geometry.levelSetField()).name;
    for (const auto field : fields) {
        declare(ConstraintPhase::InactiveSupport,
                std::make_unique<LevelSetActiveSideVertexDirichletConstraint>(
                    field,
                    level_set_name,
                    levelSetConstraintSide(active_side.side()),
                    geometry.isovalue(),
                    value,
                    geometry.interfaceMarker()));
    }
    return *this;
}

ConstraintPlan& ConstraintPlan::aggregate(std::initializer_list<FieldId> fields,
                                          const level_set::CutSide& active_side,
                                          SmallCutAggregationGuardOptions guards)
{
    requireNotPassed(ConstraintPhase::Aggregation, "aggregate");
    for (const auto field : fields) {
        declare(ConstraintPhase::Aggregation,
                std::make_unique<SmallCutAggregationConstraint>(
                    field,
                    cutIntegrationSide(active_side.side()),
                    active_side.geometry().interfaceMarker(),
                    std::vector<int>{},
                    std::vector<GlobalIndex>{},
                    guards));
    }
    return *this;
}

ConstraintPlan& ConstraintPlan::contactPointVelocity(FieldId velocity, ContactPointVelocityLaw law)
{
    return declare(ConstraintPhase::PointLaw,
                   std::make_unique<ContactPointVelocityConstraint>(velocity, std::move(law)));
}

ConstraintPlan& ConstraintPlan::contactPointVelocity(FieldId velocity,
                                                     const level_set::LevelSetGeometry& geometry,
                                                     int wall_marker,
                                                     const std::array<Real, 3>& wall_normal,
                                                     std::function<Real(Real)> contact_speed,
                                                     std::string name)
{
    ContactPointVelocityLaw law;
    law.wall_boundary_markers = {wall_marker};
    law.wall_normal = wall_normal;
    law.level_set_field_name = system_->fieldRecord(geometry.levelSetField()).name;
    law.isovalue = geometry.isovalue();
    law.contact_speed = std::move(contact_speed);
    law.name = std::move(name);
    return contactPointVelocity(velocity, std::move(law));
}

ConstraintPlan& ConstraintPlan::commitThrough(ConstraintPhase through)
{
    requireNotPassed(through, "commitThrough");
    for (std::size_t phase = 0; phase <= phaseIndex(through); ++phase) {
        auto& held = held_[phase];
        for (auto& constraint : held) {
            system_->addSystemConstraint(std::move(constraint));
            ++registered_;
        }
        held.clear();
    }
    phase_ = through;
    return *this;
}

void ConstraintPlan::commit()
{
    commitThrough(ConstraintPhase::PointLaw);
}

std::size_t ConstraintPlan::pendingCount() const noexcept
{
    std::size_t count = 0u;
    for (const auto& held : held_) {
        count += held.size();
    }
    return count;
}

} // namespace constraints
} // namespace FE
} // namespace svmp
