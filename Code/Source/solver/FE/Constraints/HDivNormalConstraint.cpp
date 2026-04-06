/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/HDivNormalConstraint.h"

#include "Basis/VectorBasis.h"
#include "Elements/ReferenceElement.h"
#include "Spaces/FunctionSpace.h"
#include "Systems/FESystem.h"
#include "Systems/SystemsExceptions.h"

#include <algorithm>
#include <unordered_set>

namespace svmp {
namespace FE {
namespace systems {

HDivNormalConstraint::HDivNormalConstraint(FieldId field, int boundary_marker)
    : field_(field)
    , boundary_marker_(boundary_marker)
{
    if (field_ == INVALID_FIELD_ID) {
        throw std::invalid_argument("HDivNormalConstraint: invalid FieldId");
    }
}

void HDivNormalConstraint::apply(const FESystem& system, constraints::AffineConstraints& constraints)
{
    const auto& rec = system.fieldRecord(field_);
    FE_CHECK_NOT_NULL(rec.space.get(), "HDivNormalConstraint: field space");

    FE_THROW_IF(rec.space->continuity() != Continuity::H_div, InvalidArgumentException,
                "HDivNormalConstraint requires an H(div) space");
    FE_THROW_IF(rec.space->field_type() != FieldType::Vector, InvalidArgumentException,
                "HDivNormalConstraint requires a vector-valued field");

    const auto& basis = rec.space->element().basis();
    FE_THROW_IF(!basis.is_vector_valued(), InvalidArgumentException,
                "HDivNormalConstraint requires a vector-basis space");

    const auto* vbf = dynamic_cast<const basis::VectorBasisFunction*>(&basis);
    FE_THROW_IF(vbf == nullptr, InvalidArgumentException,
                "HDivNormalConstraint requires a VectorBasisFunction");

    const auto associations = vbf->dof_associations();
    FE_THROW_IF(associations.empty(), InvalidStateException,
                "HDivNormalConstraint: empty DOF association list");

    const auto& mesh = system.meshAccess();
    const auto& dh = system.fieldDofHandler(field_);
    const auto& dof_map = dh.getDofMap();
    const GlobalIndex offset = system.fieldDofOffset(field_);

    std::unordered_set<GlobalIndex> unique;
    unique.reserve(128u);

    mesh.forEachBoundaryFace(boundary_marker_, [&](GlobalIndex face_id, GlobalIndex cell_id) {
        const auto cell_dofs = dof_map.getCellDofs(cell_id);
        FE_THROW_IF(cell_dofs.empty(), InvalidStateException,
                    "HDivNormalConstraint: empty cell DOF list");
        FE_THROW_IF(cell_dofs.size() < associations.size(), InvalidStateException,
                    "HDivNormalConstraint: cell DOF list smaller than association list");

        const ElementType cell_type = mesh.getCellType(cell_id);
        const elements::ReferenceElement ref = elements::ReferenceElement::create(cell_type);

        const auto local_face = mesh.getLocalFaceIndex(face_id, cell_id);
        FE_THROW_IF(local_face < 0 || static_cast<std::size_t>(local_face) >= ref.num_faces(), InvalidArgumentException,
                    "HDivNormalConstraint: local face index out of range");

        for (std::size_t ldof = 0; ldof < associations.size(); ++ldof) {
            const auto& a = associations[ldof];

            bool constrain = false;
            if (a.entity_type == basis::DofEntity::Face) {
                constrain = (a.entity_id == static_cast<int>(local_face));
            } else if (a.entity_type == basis::DofEntity::Edge && ref.dimension() == 2) {
                // In 2D, codimension-1 facets are "faces" (edges). H(div) DOFs are edge-associated.
                constrain = (a.entity_id == static_cast<int>(local_face));
            }

            if (!constrain) continue;
            const auto dof_local = cell_dofs[ldof];
            unique.insert(dof_local + offset);
        }
    });

    dofs_.assign(unique.begin(), unique.end());
    std::sort(dofs_.begin(), dofs_.end());
    dofs_.erase(std::unique(dofs_.begin(), dofs_.end()), dofs_.end());

    const auto& owned = system.dofHandler().getPartition().locallyOwned();
    for (const auto dof : dofs_) {
        if (!owned.contains(dof)) {
            continue;
        }
        constraints.addDirichlet(dof, 0.0);
    }
}

} // namespace systems
} // namespace FE
} // namespace svmp
