/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/HCurlTangentialConstraint.h"

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

namespace {

[[nodiscard]] std::vector<int> localEdgesOnFace(const elements::ReferenceElement& ref, LocalIndex local_face_id)
{
    std::vector<int> edges;
    const auto& fn = ref.face_nodes(static_cast<std::size_t>(local_face_id));
    if (fn.size() < 2u) {
        return edges;
    }

    auto on_face = [&](LocalIndex v) -> bool {
        for (auto fv : fn) {
            if (fv == v) return true;
        }
        return false;
    };

    edges.reserve(ref.num_edges());
    for (std::size_t e = 0; e < ref.num_edges(); ++e) {
        const auto& en = ref.edge_nodes(e);
        if (en.size() != 2u) continue;
        if (on_face(en[0]) && on_face(en[1])) {
            edges.push_back(static_cast<int>(e));
        }
    }
    return edges;
}

} // namespace

HCurlTangentialConstraint::HCurlTangentialConstraint(FieldId field, int boundary_marker)
    : field_(field)
    , boundary_marker_(boundary_marker)
{
    if (field_ == INVALID_FIELD_ID) {
        throw std::invalid_argument("HCurlTangentialConstraint: invalid FieldId");
    }
}

void HCurlTangentialConstraint::apply(const FESystem& system, constraints::AffineConstraints& constraints)
{
    const auto& rec = system.fieldRecord(field_);
    FE_CHECK_NOT_NULL(rec.space.get(), "HCurlTangentialConstraint: field space");

    FE_THROW_IF(rec.space->continuity() != Continuity::H_curl, InvalidArgumentException,
                "HCurlTangentialConstraint requires an H(curl) space");
    FE_THROW_IF(rec.space->field_type() != FieldType::Vector, InvalidArgumentException,
                "HCurlTangentialConstraint requires a vector-valued field");

    const auto& basis = rec.space->element().basis();
    FE_THROW_IF(!basis.is_vector_valued(), InvalidArgumentException,
                "HCurlTangentialConstraint requires a vector-basis space");

    const auto* vbf = dynamic_cast<const basis::VectorBasisFunction*>(&basis);
    FE_THROW_IF(vbf == nullptr, InvalidArgumentException,
                "HCurlTangentialConstraint requires a VectorBasisFunction");

    const auto associations = vbf->dof_associations();
    FE_THROW_IF(associations.empty(), InvalidStateException,
                "HCurlTangentialConstraint: empty DOF association list");

    const auto& mesh = system.meshAccess();
    const auto& dh = system.fieldDofHandler(field_);
    const auto& dof_map = dh.getDofMap();
    const GlobalIndex offset = system.fieldDofOffset(field_);

    std::unordered_set<GlobalIndex> unique;
    unique.reserve(128u);

    mesh.forEachBoundaryFace(boundary_marker_, [&](GlobalIndex face_id, GlobalIndex cell_id) {
        const auto cell_dofs = dof_map.getCellDofs(cell_id);
        FE_THROW_IF(cell_dofs.empty(), InvalidStateException,
                    "HCurlTangentialConstraint: empty cell DOF list");
        FE_THROW_IF(cell_dofs.size() < associations.size(), InvalidStateException,
                    "HCurlTangentialConstraint: cell DOF list smaller than association list");

        const ElementType cell_type = mesh.getCellType(cell_id);
        const elements::ReferenceElement ref = elements::ReferenceElement::create(cell_type);

        const auto local_face = mesh.getLocalFaceIndex(face_id, cell_id);
        FE_THROW_IF(local_face < 0 || static_cast<std::size_t>(local_face) >= ref.num_faces(), InvalidArgumentException,
                    "HCurlTangentialConstraint: local face index out of range");

        const auto face_edges = localEdgesOnFace(ref, local_face);

        auto is_face_edge = [&](int edge_id) -> bool {
            for (int e : face_edges) {
                if (e == edge_id) return true;
            }
            return false;
        };

        for (std::size_t ldof = 0; ldof < associations.size(); ++ldof) {
            const auto& a = associations[ldof];

            bool constrain = false;
            if (a.entity_type == basis::DofEntity::Edge) {
                constrain = is_face_edge(a.entity_id);
            } else if (a.entity_type == basis::DofEntity::Face) {
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
