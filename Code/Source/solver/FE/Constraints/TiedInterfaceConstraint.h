/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_TIEDINTERFACECONSTRAINT_H
#define SVMP_FE_CONSTRAINTS_TIEDINTERFACECONSTRAINT_H

#include "Constraint.h"

#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {

/**
 * @brief One algebraic relation produced by a tied-interface relation map.
 *
 * The class is deliberately physics-neutral: it stores only a slave DOF, master
 * DOFs, interpolation/projection weights, and an optional inhomogeneous offset.
 */
struct TiedInterfaceRelation {
    GlobalIndex slave_dof{INVALID_GLOBAL_INDEX};
    std::vector<std::pair<GlobalIndex, double>> masters{};
    double inhomogeneity{0.0};
    std::string name{};

    [[nodiscard]] bool valid() const noexcept;
};

/**
 * @brief Revision-stamped relation map for tied-interface constraints.
 *
 * Producers can build this from fitted interface search, sliding-interface
 * pairing, mortar projection, or any other physics-agnostic relation builder.
 * Consumers can then tell whether the map is still valid for the current mesh
 * and FE layout revisions.
 */
struct TiedInterfaceRelationMap {
    std::string name{};
    std::vector<TiedInterfaceRelation> relations{};
    ConstraintRevisionSnapshot revision{};
    bool built{false};

    [[nodiscard]] bool empty() const noexcept { return relations.empty(); }
    [[nodiscard]] bool valid() const noexcept;
};

[[nodiscard]] ConstraintDependencyDeclaration tiedInterfaceDependencyDeclaration() noexcept;

/**
 * @brief Affine tied-interface constraint backed by a revision-stamped map.
 */
class TiedInterfaceConstraint final : public Constraint {
public:
    TiedInterfaceConstraint();
    explicit TiedInterfaceConstraint(TiedInterfaceRelationMap relation_map);
    TiedInterfaceConstraint(TiedInterfaceRelationMap relation_map,
                            ConstraintDependencyDeclaration dependencies);

    void apply(AffineConstraints& constraints) const override;

    [[nodiscard]] ConstraintType getType() const noexcept override {
        return ConstraintType::MultiPoint;
    }

    [[nodiscard]] ConstraintInfo getInfo() const override;
    [[nodiscard]] ConstraintDependencyDeclaration dependencyDeclaration() const override;

    [[nodiscard]] std::unique_ptr<Constraint> clone() const override {
        return std::make_unique<TiedInterfaceConstraint>(*this);
    }

    void setRelationMap(TiedInterfaceRelationMap relation_map);
    [[nodiscard]] const TiedInterfaceRelationMap& relationMap() const noexcept {
        return relation_map_;
    }

    [[nodiscard]] bool relationMapStaleFor(const ConstraintRevisionSnapshot& current) const noexcept;
    [[nodiscard]] std::string validate() const;

private:
    TiedInterfaceRelationMap relation_map_{};
    ConstraintDependencyDeclaration dependencies_{};
};

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_TIEDINTERFACECONSTRAINT_H
