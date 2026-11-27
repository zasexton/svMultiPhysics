/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Constraints/AuxiliaryDrivenDirichletConstraint.h"

#include "Constraints/BoundaryDofOwnerCompletion.h"

#include "Systems/FESystem.h"
#include "Systems/SystemsExceptions.h"

#include "Constraints/BoundaryFieldTrace.h"

#include <algorithm>
#include <string>
#include <utility>

namespace svmp {
namespace FE {
namespace constraints {

using systems::FESystem;
using systems::InvalidStateException;

namespace {

[[nodiscard]] std::vector<GlobalIndex> boundaryTraceDofs(const FESystem& system, FieldId field,
                                                         int boundary_marker, int component)
{
    std::vector<GlobalIndex> dofs;
    const auto& rec = system.fieldRecord(field);
    FE_CHECK_NOT_NULL(rec.space.get(), "AuxiliaryDrivenDirichletConstraint: field space");
    FE_THROW_IF(rec.space->continuity() == Continuity::H_curl ||
                    rec.space->continuity() == Continuity::H_div ||
                    rec.space->element().basis().is_vector_valued(),
                InvalidArgumentException,
                "AuxiliaryDrivenDirichletConstraint does not support H(curl)/H(div) "
                "vector-basis spaces; use the dedicated vector-basis constraint types instead");

    const auto& mesh = system.meshAccess();
    const auto& dh = system.fieldDofHandler(field);
    const auto& dof_map = dh.getDofMap();
    const GlobalIndex offset = system.fieldDofOffset(field);

    const int n_components = std::max(1, rec.space->value_dimension());
    FE_THROW_IF(component < -1, InvalidArgumentException,
                "AuxiliaryDrivenDirichletConstraint: component must be >= -1");
    FE_THROW_IF(component >= n_components, InvalidArgumentException,
                "AuxiliaryDrivenDirichletConstraint: component index out of range");

    detail::BoundaryFieldTraces traces;
    const auto& scalar = detail::scalarBoundarySpace(*rec.space);
    mesh.forEachBoundaryFace(boundary_marker, [&](GlobalIndex face, GlobalIndex cell) {
        const auto& restriction = traces.get(scalar.element_space(cell), mesh.getCellType(cell));
        const auto face_dofs = restriction.face_dofs(static_cast<int>(mesh.getLocalFaceIndex(face, cell)));
        const auto cell_dofs = dof_map.getCellDofs(cell);
        const auto n = cell_dofs.size() / static_cast<std::size_t>(n_components);
        FE_THROW_IF(n == 0 || n * n_components != cell_dofs.size(), InvalidStateException,
                    "AuxiliaryDrivenDirichletConstraint: invalid component layout");
        for (int c = component < 0 ? 0 : component; c < (component < 0 ? n_components : component + 1); ++c) {
            for (int local : face_dofs) {
                FE_THROW_IF(local < 0 || static_cast<std::size_t>(local) >= n, InvalidStateException,
                            "AuxiliaryDrivenDirichletConstraint: face coefficient outside the cell layout");
                dofs.push_back(offset + cell_dofs[c * n + local]);
            }
        }
    });
    std::sort(dofs.begin(), dofs.end());
    dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
    std::vector<Real> no_payload;
    completeOwnedBoundaryDofs(system, dofs, no_payload, 0,
        "constraint=AuxiliaryDrivenDirichlet field='" + rec.name + "' marker=" + std::to_string(boundary_marker)
        + " component=" + std::to_string(component));
    return dofs;
}

[[nodiscard]] int parseBoundaryMarker(const systems::AuxiliaryConstraintBinding& binding,
                                      std::string_view instance_name)
{
    FE_THROW_IF(binding.target_region.kind != systems::AuxiliaryRegionKind::BoundarySet,
                InvalidArgumentException,
                "AuxiliaryDrivenDirichletConstraint: instance '" + std::string(instance_name) +
                    "' strong-Dirichlet bindings must target BoundarySet regions");
    try {
        return std::stoi(binding.target_region.identity);
    } catch (const std::exception&) {
        FE_THROW(InvalidArgumentException,
                 "AuxiliaryDrivenDirichletConstraint: instance '" +
                     std::string(instance_name) +
                     "' has non-integer boundary marker '" +
                     binding.target_region.identity + "'");
    }
}

} // namespace

AuxiliaryDrivenDirichletConstraint::AuxiliaryDrivenDirichletConstraint(
    std::string instance_name,
    systems::AuxiliaryConstraintBinding binding)
    : instance_name_(std::move(instance_name))
    , binding_(std::move(binding))
{
    FE_THROW_IF(instance_name_.empty(), InvalidArgumentException,
                "AuxiliaryDrivenDirichletConstraint: empty instance name");
    FE_THROW_IF(binding_.target_field == INVALID_FIELD_ID, InvalidArgumentException,
                "AuxiliaryDrivenDirichletConstraint: invalid target field");
}

void AuxiliaryDrivenDirichletConstraint::apply(const FESystem& system,
                                               AffineConstraints& constraints)
{
    const int boundary_marker = parseBoundaryMarker(binding_, instance_name_);

    dofs_ = boundaryTraceDofs(system, binding_.target_field, boundary_marker, binding_.target_component);
    const auto& owned = system.dofHandler().getPartition().locallyOwned();
    const Real value = system.auxiliaryConstraintValue(
        instance_name_, binding_, /*time=*/Real{0.0}, /*dt=*/Real{0.0});

    for (const auto dof : dofs_) {
        if (!owned.contains(dof)) {
            continue;
        }
        constraints.addDirichlet(dof, value);
    }
}

bool AuxiliaryDrivenDirichletConstraint::updateValues(const FESystem& system,
                                                      AffineConstraints& constraints,
                                                      double time,
                                                      double dt)
{
    const Real value = system.auxiliaryConstraintValue(
        instance_name_, binding_, static_cast<Real>(time), static_cast<Real>(dt));
    const auto& owned = system.dofHandler().getPartition().locallyOwned();
    for (const auto dof : dofs_)
        if (owned.contains(dof)) constraints.updateInhomogeneity(dof, value);
    return true;
}

} // namespace constraints
} // namespace FE
} // namespace svmp
