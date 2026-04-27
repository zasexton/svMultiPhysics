/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "TiedInterfaceConstraint.h"

#include <cmath>
#include <sstream>

namespace svmp {
namespace FE {
namespace constraints {

bool TiedInterfaceRelation::valid() const noexcept
{
    return slave_dof != INVALID_GLOBAL_INDEX && !masters.empty();
}

bool TiedInterfaceRelationMap::valid() const noexcept
{
    if (!built || !revision.valid) {
        return false;
    }
    for (const auto& relation : relations) {
        if (!relation.valid()) {
            return false;
        }
    }
    return true;
}

ConstraintDependencyDeclaration tiedInterfaceDependencyDeclaration() noexcept
{
    ConstraintDependencyDeclaration out;
    merge_into(out.structural, ConstraintDependencyMask::meshGeometry());
    merge_into(out.structural, ConstraintDependencyMask::meshBoundaryTopology());
    merge_into(out.structural, ConstraintDependencyMask::feDofLayout());
    return out;
}

TiedInterfaceConstraint::TiedInterfaceConstraint()
    : dependencies_(tiedInterfaceDependencyDeclaration())
{
}

TiedInterfaceConstraint::TiedInterfaceConstraint(TiedInterfaceRelationMap relation_map)
    : relation_map_(std::move(relation_map)),
      dependencies_(tiedInterfaceDependencyDeclaration())
{
}

TiedInterfaceConstraint::TiedInterfaceConstraint(
    TiedInterfaceRelationMap relation_map,
    ConstraintDependencyDeclaration dependencies)
    : relation_map_(std::move(relation_map)),
      dependencies_(std::move(dependencies))
{
}

void TiedInterfaceConstraint::apply(AffineConstraints& constraints) const
{
    const auto error = validate();
    if (!error.empty()) {
        CONSTRAINT_THROW(error);
    }

    for (const auto& relation : relation_map_.relations) {
        constraints.addLine(relation.slave_dof);
        for (const auto& [master, weight] : relation.masters) {
            if (std::abs(weight) > 1e-15) {
                constraints.addEntry(relation.slave_dof, master, weight);
            }
        }
        if (std::abs(relation.inhomogeneity) > 1e-15) {
            constraints.setInhomogeneity(relation.slave_dof, relation.inhomogeneity);
        }
    }
}

ConstraintInfo TiedInterfaceConstraint::getInfo() const
{
    ConstraintInfo info;
    info.name = relation_map_.name.empty() ? "TiedInterfaceConstraint" : relation_map_.name;
    info.type = ConstraintType::MultiPoint;
    info.num_constrained_dofs = relation_map_.relations.size();
    info.is_time_dependent = false;
    info.is_homogeneous = true;
    for (const auto& relation : relation_map_.relations) {
        if (std::abs(relation.inhomogeneity) > 1e-15) {
            info.is_homogeneous = false;
            break;
        }
    }
    return info;
}

ConstraintDependencyDeclaration TiedInterfaceConstraint::dependencyDeclaration() const
{
    ConstraintDependencyDeclaration out = Constraint::dependencyDeclaration();
    merge_into(out, dependencies_);
    return out;
}

void TiedInterfaceConstraint::setRelationMap(TiedInterfaceRelationMap relation_map)
{
    relation_map_ = std::move(relation_map);
}

bool TiedInterfaceConstraint::relationMapStaleFor(
    const ConstraintRevisionSnapshot& current) const noexcept
{
    if (!relation_map_.valid()) {
        return true;
    }
    return structural_dependency_changed(dependencyDeclaration(),
                                         relation_map_.revision,
                                         current);
}

std::string TiedInterfaceConstraint::validate() const
{
    if (!relation_map_.built) {
        return "TiedInterfaceConstraint relation map has not been built";
    }
    if (!relation_map_.revision.valid) {
        return "TiedInterfaceConstraint relation map has no valid revision snapshot";
    }

    for (std::size_t i = 0; i < relation_map_.relations.size(); ++i) {
        const auto& relation = relation_map_.relations[i];
        if (relation.slave_dof == INVALID_GLOBAL_INDEX) {
            std::ostringstream os;
            os << "TiedInterfaceConstraint relation " << i << " has an invalid slave DOF";
            return os.str();
        }
        if (relation.masters.empty()) {
            std::ostringstream os;
            os << "TiedInterfaceConstraint relation " << i << " has no master DOFs";
            return os.str();
        }
        for (const auto& [master, weight] : relation.masters) {
            if (master == INVALID_GLOBAL_INDEX) {
                std::ostringstream os;
                os << "TiedInterfaceConstraint relation " << i << " has an invalid master DOF";
                return os.str();
            }
            if (!std::isfinite(weight)) {
                std::ostringstream os;
                os << "TiedInterfaceConstraint relation " << i << " has a non-finite weight";
                return os.str();
            }
        }
        if (!std::isfinite(relation.inhomogeneity)) {
            std::ostringstream os;
            os << "TiedInterfaceConstraint relation " << i
               << " has a non-finite inhomogeneity";
            return os.str();
        }
    }
    return {};
}

} // namespace constraints
} // namespace FE
} // namespace svmp
