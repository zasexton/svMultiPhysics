/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Constraints/CoupledFieldGaugeConstraint.h"

#include "Spaces/FunctionSpace.h"
#include "Systems/FESystem.h"
#include "Systems/FieldRegistry.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#if FE_HAS_MPI
#  include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace constraints {

namespace {

[[nodiscard]] bool mpiIsActive() noexcept
{
#if FE_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
        MPI_Finalized(&finalized);
    }
    return initialized != 0 && finalized == 0;
#else
    return false;
#endif
}

[[nodiscard]] GlobalIndex communicatorMinimum(
    GlobalIndex local_value,
    const systems::FESystem& system)
{
#if FE_HAS_MPI
    if (mpiIsActive()) {
        GlobalIndex global_value = local_value;
        MPI_Allreduce(&local_value,
                      &global_value,
                      1,
                      MPI_INT64_T,
                      MPI_MIN,
                      system.dofHandler().mpiComm());
        return global_value;
    }
#else
    (void)system;
#endif
    return local_value;
}

void validateScalarUnknown(const systems::FESystem& system,
                           FieldId field,
                           const char* position)
{
    const auto& record = system.fieldRecord(field);
    if (record.source_kind != systems::FieldSourceKind::Unknown) {
        throw std::invalid_argument(
            std::string("CoupledFieldGaugeConstraint: ") + position +
            " field must be an unknown");
    }
    if (!record.space || record.components != 1 ||
        record.space->value_dimension() != 1) {
        throw std::invalid_argument(
            std::string("CoupledFieldGaugeConstraint: ") + position +
            " field must be scalar");
    }
}

} // namespace

CoupledFieldGaugeConstraint::CoupledFieldGaugeConstraint(
    FieldId first_field,
    FieldId second_field,
    Real pinned_value)
    : first_field_(first_field)
    , second_field_(second_field)
    , pinned_value_(pinned_value)
{
    if (first_field_ == INVALID_FIELD_ID ||
        second_field_ == INVALID_FIELD_ID) {
        throw std::invalid_argument(
            "CoupledFieldGaugeConstraint: both FieldIds must be valid");
    }
    if (first_field_ == second_field_) {
        throw std::invalid_argument(
            "CoupledFieldGaugeConstraint: fields must be distinct");
    }
    if (!std::isfinite(pinned_value_)) {
        throw std::invalid_argument(
            "CoupledFieldGaugeConstraint: pinned value must be finite");
    }
}

void CoupledFieldGaugeConstraint::apply(
    const systems::FESystem& system,
    AffineConstraints& constraints)
{
    validateScalarUnknown(system, first_field_, "first");
    validateScalarUnknown(system, second_field_, "second");

    const auto& owned = system.dofHandler().getPartition().locallyOwned();
    constexpr auto no_candidate = std::numeric_limits<GlobalIndex>::max();
    GlobalIndex local_candidate = no_candidate;

    for (const auto field : std::array<FieldId, 2>{
             first_field_, second_field_}) {
        const auto offset = system.fieldDofOffset(field);
        const auto count = system.fieldDofHandler(field).getNumDofs();
        for (GlobalIndex local = 0; local < count; ++local) {
            const auto dof = offset + local;
            if (owned.contains(dof) && !constraints.isConstrained(dof)) {
                local_candidate = std::min(local_candidate, dof);
                break;
            }
        }
    }

    const auto selected = communicatorMinimum(local_candidate, system);
    if (selected == no_candidate) {
        throw std::runtime_error(
            "CoupledFieldGaugeConstraint: no unconstrained DOF remains in either field");
    }

    constraints.addDirichlet(selected, pinned_value_);
    pinned_dof_ = selected;
}

bool CoupledFieldGaugeConstraint::updateValues(
    const systems::FESystem& system,
    AffineConstraints& constraints,
    double time,
    double dt)
{
    (void)system;
    (void)constraints;
    (void)time;
    (void)dt;
    return false;
}

ConstraintDependencyDeclaration
CoupledFieldGaugeConstraint::dependencyDeclaration() const
{
    ConstraintDependencyDeclaration out;
    out.structural.ownership = true;
    out.structural.numbering = true;
    out.structural.fe_dof_layout = true;
    out.structural.fe_constraint_layout = true;
    return out;
}

systems::SetupStorageRequirements
CoupledFieldGaugeConstraint::storageRequirements() const noexcept
{
    return {};
}

} // namespace constraints
} // namespace FE
} // namespace svmp
