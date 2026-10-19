/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_CONTACTPOINTVELOCITYCONSTRAINT_H
#define SVMP_FE_CONSTRAINTS_CONTACTPOINTVELOCITYCONSTRAINT_H

/**
 * @file ContactPointVelocityConstraint.h
 * @brief Essential condition on the wall-tangential velocity at the contact
 *        points of a P1 level set on a planar wall (2D).
 *
 * A contact point is the zero crossing x_c of the vertex level set on a wall
 * edge [w, d] (w on the side phi < isovalue, d on the other side).  With
 * s = (phi_w - c) / (phi_w - phi_d) the interpolated wall-tangential velocity
 * along the outward footprint direction t_f (from w towards d) is
 *     t_f . u(x_c) = (1 - s) t_f . u_w + s t_f . u_d,
 * and this constraint sets it to V(cos theta_cell), where theta_cell is the
 * through-liquid angle of the P1 level set in the wall cell
 * (cos theta = -n . n_wall, n = grad phi / |grad phi|) and V is supplied by
 * the caller (for the Ren-E law V = M gamma (cos theta_e - cos theta)).
 *
 * The constraint is applied after every other constraint of the field.  The
 * DOFs of the functional that are already constrained (typically the dry
 * vertex, an aggregation slave) are expanded through their closed lines, so
 * the new line is written for an unconstrained DOF of the expanded functional
 * (the one with the largest coefficient; ties by DOF number) and never forms
 * a cycle.  The wall-normal component is not involved.  Lines are computed on
 * the rank that owns the wall cell, gathered on the communicator and installed
 * on every rank on which the slave is locally relevant; a relevant slave with
 * a master that is not locally relevant fails closed
 * (InsufficientGhostHaloError).  The value is lagged with the geometry: it is
 * recomputed whenever the constraint is rebuilt from the current level set,
 * i.e. at every outer geometry pass.  Axis-aligned planar walls only.
 */

#include "Constraints/SystemConstraint.h"
#include "Core/Types.h"

#include <array>
#include <functional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {

struct ContactPointVelocityLaw {
    std::vector<int> wall_boundary_markers{};
    // Unit wall normal pointing out of the fluid into the solid; must be a
    // coordinate axis direction.
    std::array<Real, 3> wall_normal{{0.0, -1.0, 0.0}};
    std::string level_set_field_name{};
    Real isovalue{0.0};
    // Optional source of the vertex level-set values (mesh vertex index ->
    // value); when empty the vertex mesh field level_set_field_name is read.
    std::function<Real(GlobalIndex)> level_set_vertex_value{};
    // Outward wall-tangential contact-point speed as a function of
    // cos(theta_cell); positive values advance the contact point.
    std::function<Real(Real)> contact_speed{};
    std::string name{"contact_point_velocity"};
};

struct ContactPointVelocitySample {
    std::array<Real, 3> point{};
    Real fraction{0.0};
    Real cos_theta{0.0};
    Real target{0.0};
    GlobalIndex slave_dof{INVALID_GLOBAL_INDEX};
};

class ContactPointVelocityConstraint final : public ISystemConstraint {
public:
    ContactPointVelocityConstraint(FieldId velocity_field, ContactPointVelocityLaw law);

    void apply(const systems::FESystem& system, AffineConstraints& constraints) override;

    bool updateValues(const systems::FESystem& system,
                      AffineConstraints& constraints,
                      double time,
                      double dt) override;

    [[nodiscard]] bool isTimeDependent() const noexcept override { return false; }

    [[nodiscard]] ConstraintDependencyDeclaration dependencyDeclaration() const override;

    [[nodiscard]] systems::SetupStorageRequirements storageRequirements() const noexcept override;

    [[nodiscard]] std::string manifestDescription() const override;

    /// Contact points of the most recent apply() (communicator-global, sorted by slave).
    [[nodiscard]] const std::vector<ContactPointVelocitySample>& samples() const noexcept
    {
        return samples_;
    }

private:
    FieldId velocity_field_{INVALID_FIELD_ID};
    ContactPointVelocityLaw law_{};
    int tangential_axis_{-1};
    std::vector<ContactPointVelocitySample> samples_{};
};

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_CONTACTPOINTVELOCITYCONSTRAINT_H
