/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Constraints/SupportRestrictionConstraint.h"

#include "Assembly/CutIntegrationContext.h"
#include "Constraints/BoundaryDofOwnerCompletion.h"
#include "Dofs/DofMap.h"
#include "Forms/FormExpr.h"
#include "Forms/IntegrationDomain.h"
#include "Spaces/FaceRestriction.h"
#include "Spaces/FunctionSpace.h"
#include "Systems/FESystem.h"

#include "Mesh/Core/InterfaceMesh.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {

namespace {

[[nodiscard]] std::string elementTypeText(ElementType type)
{
    switch (type) {
    case ElementType::Line2: return "Line2";
    case ElementType::Triangle3: return "Triangle3";
    case ElementType::Quad4: return "Quad4";
    case ElementType::Tetra4: return "Tetra4";
    case ElementType::Hex8: return "Hex8";
    case ElementType::Wedge6: return "Wedge6";
    case ElementType::Pyramid5: return "Pyramid5";
    case ElementType::Line3: return "Line3";
    case ElementType::Triangle6: return "Triangle6";
    case ElementType::Quad9: return "Quad9";
    case ElementType::Quad8: return "Quad8";
    case ElementType::Tetra10: return "Tetra10";
    case ElementType::Hex27: return "Hex27";
    case ElementType::Hex20: return "Hex20";
    case ElementType::Wedge15: return "Wedge15";
    case ElementType::Wedge18: return "Wedge18";
    case ElementType::Pyramid13: return "Pyramid13";
    case ElementType::Pyramid14: return "Pyramid14";
    case ElementType::Point1: return "Point1";
    case ElementType::Unknown: break;
    }
    return "ElementType(" + std::to_string(static_cast<int>(type)) + ")";
}

/// The face DOF tables of one field, per cell type.
class FaceDofs {
public:
    FaceDofs(const spaces::FunctionSpace& space, std::string context)
        : order_(space.polynomial_order())
        , continuity_(space.continuity())
        , context_(std::move(context))
    {
    }

    /// The cell-local DOFs (of one component) on face @p local_face of a cell of @p type.
    [[nodiscard]] std::vector<int> onFace(ElementType type, int local_face)
    {
        auto it = cache_.find(static_cast<int>(type));
        if (it == cache_.end()) {
            try {
                it = cache_.emplace(static_cast<int>(type), spaces::FaceRestriction(type, order_, continuity_)).first;
            } catch (const std::exception& e) {
                throw std::invalid_argument(context_ + ": the face degrees of freedom of a " + elementTypeText(type) +
                                            " cell of order " + std::to_string(order_) +
                                            " are not available (" + e.what() + ")");
            }
        }
        try {
            return it->second.face_dofs(local_face);
        } catch (const std::exception& e) {
            throw std::invalid_argument(context_ + ": face " + std::to_string(local_face) + " of a " +
                                        elementTypeText(type) + " cell has no degree-of-freedom table (" +
                                        e.what() + ")");
        }
    }

private:
    int order_{1};
    Continuity continuity_{Continuity::C0};
    std::string context_;
    std::unordered_map<int, spaces::FaceRestriction> cache_;
};

} // namespace

const char* supportDomainKindName(SupportDomainKind kind) noexcept
{
    switch (kind) {
    case SupportDomainKind::PhysicalBoundary: return "physical_boundary";
    case SupportDomainKind::FittedInterface: return "fitted_interface";
    case SupportDomainKind::GeneratedInterface: return "generated_interface";
    }
    return "unknown";
}

SupportRestrictionConstraint::SupportRestrictionConstraint(FieldId field,
                                                           const forms::ExteriorBoundaryMeasure& domain)
    : field_(field)
    , kind_(SupportDomainKind::PhysicalBoundary)
    , marker_(domain.physicalBoundaryMarker())
{
    if (field_ == INVALID_FIELD_ID) {
        throw std::invalid_argument("SupportRestrictionConstraint: invalid FieldId");
    }
    if (!domain.isFullPhysical()) {
        throw std::invalid_argument(
            "SupportRestrictionConstraint: the generated active subset of boundary " +
            std::to_string(domain.physicalBoundaryMarker()) +
            " is not a support domain; restrict the field to the physical boundary or to an interface");
    }
}

SupportRestrictionConstraint::SupportRestrictionConstraint(FieldId field, const forms::InterfaceDomain& domain)
    : field_(field)
    , kind_(SupportDomainKind::FittedInterface)
    , marker_(domain.marker())
{
    if (field_ == INVALID_FIELD_ID) {
        throw std::invalid_argument("SupportRestrictionConstraint: invalid FieldId");
    }
}

SupportRestrictionConstraint::SupportRestrictionConstraint(FieldId field,
                                                           int interface_marker,
                                                           std::string level_set_field_name)
    : field_(field)
    , kind_(SupportDomainKind::GeneratedInterface)
    , marker_(interface_marker)
    , level_set_field_name_(std::move(level_set_field_name))
{
    if (field_ == INVALID_FIELD_ID) {
        throw std::invalid_argument("SupportRestrictionConstraint: invalid FieldId");
    }
    if (marker_ < 0) {
        throw std::invalid_argument("SupportRestrictionConstraint: a generated interface needs a nonnegative marker");
    }
    if (level_set_field_name_.empty()) {
        throw std::invalid_argument("SupportRestrictionConstraint: a generated interface needs its level-set field");
    }
}

namespace {

/// The field-local DOFs of every component of the cell-local DOFs @p local_dofs.
void componentDofs(std::span<const GlobalIndex> cell_dofs,
                   int n_components,
                   const std::vector<int>& local_dofs,
                   const std::string& context,
                   GlobalIndex cell,
                   const std::function<void(GlobalIndex)>& visit)
{
    const auto per_component = cell_dofs.size() / static_cast<std::size_t>(n_components);
    if (per_component == 0u || per_component * static_cast<std::size_t>(n_components) != cell_dofs.size()) {
        throw std::runtime_error(context + ": the cell DOF list of cell " + std::to_string(cell) +
                                 " is not divisible by the field's components");
    }
    for (const int ldof : local_dofs) {
        if (ldof < 0 || static_cast<std::size_t>(ldof) >= per_component) {
            throw std::runtime_error(context + ": face DOF " + std::to_string(ldof) +
                                     " is outside the cell DOF list of cell " + std::to_string(cell));
        }
        for (int c = 0; c < n_components; ++c) {
            visit(cell_dofs[static_cast<std::size_t>(c) * per_component + static_cast<std::size_t>(ldof)]);
        }
    }
}

void requireH1(const systems::FieldRecord& rec, const std::string& context)
{
    if (!rec.space) {
        throw std::invalid_argument(context + ": the field has no function space");
    }
    const bool h1 = rec.space->space_type() == spaces::SpaceType::H1 ||
                    rec.space->space_type() == spaces::SpaceType::Product;
    if (!h1 || rec.space->continuity() != Continuity::C0 || rec.space->element().basis().is_vector_valued()) {
        throw std::invalid_argument(context + ": the field must be an H1 (C0) scalar or Product H1 vector field");
    }
}

} // namespace

std::string SupportRestrictionConstraint::context(const systems::FESystem& system) const
{
    return std::string("SupportRestrictionConstraint field='") + system.fieldRecord(field_).name + "' " +
           supportDomainKindName(kind_) + " " + std::to_string(marker_);
}

void SupportRestrictionConstraint::forEachSupportEntity(
    const systems::FESystem& system,
    const std::function<void(GlobalIndex cell, int local_face)>& visit) const
{
    const auto& mesh = system.meshAccess();
    switch (kind_) {
    case SupportDomainKind::PhysicalBoundary: {
        mesh.forEachBoundaryFace(marker_, [&](GlobalIndex face_id, GlobalIndex cell_id) {
            visit(cell_id, static_cast<int>(mesh.getLocalFaceIndex(face_id, cell_id)));
        });
        break;
    }
    case SupportDomainKind::FittedInterface: {
        if (!system.hasInterfaceMesh(marker_)) {
            throw std::invalid_argument(
                context(system) + ": no InterfaceMesh is registered at this marker; a generated interface is "
                                  "restricted through its level-set geometry (ConstraintPlan::restrictSupport)");
        }
        const auto& interface_mesh = system.interfaceMesh(marker_);
        for (std::size_t f = 0; f < interface_mesh.n_faces(); ++f) {
            const auto face = static_cast<svmp::index_t>(f);
            const auto cells = interface_mesh.volume_cells(face);
            for (int side = 0; side < 2; ++side) {
                const auto cell = cells[static_cast<std::size_t>(side)];
                if (cell == svmp::INVALID_INDEX) {
                    continue;
                }
                visit(static_cast<GlobalIndex>(cell), side == 0 ? interface_mesh.local_face_in_cell_minus(face)
                                                                : interface_mesh.local_face_in_cell_plus(face));
            }
        }
        break;
    }
    case SupportDomainKind::GeneratedInterface: {
        const auto* cut_context = system.cutIntegrationContext();
        if (cut_context != nullptr) {
            cut_context->assertAllFreeSurfaceGeometrySnapshotsCurrent(mesh);
        }
        if (cut_context == nullptr || !cut_context->hasGeneratedInterfaceMarker(marker_)) {
            break;
        }
        const auto& rules = cut_context->interfaceRules();
        for (const auto index : cut_context->generatedInterfaceRuleIndexSpanForMarker(marker_)) {
            if (index >= rules.size()) {
                continue;
            }
            const auto& rule = rules[index];
            if (!(std::isfinite(rule.measure) && rule.measure > Real{0.0})) {
                continue;
            }
            const auto cell = static_cast<GlobalIndex>(rule.provenance.parent_entity);
            if (cell >= 0) {
                visit(cell, -1);
            }
        }
        break;
    }
    }
}

void SupportRestrictionConstraint::apply(const systems::FESystem& system, AffineConstraints& constraints)
{
    const auto& rec = system.fieldRecord(field_);
    const std::string ctx = context(system);
    requireH1(rec, ctx);

    const auto& dh = system.fieldDofHandler(field_);
    const auto& dof_map = dh.getDofMap();
    const GlobalIndex offset = system.fieldDofOffset(field_);
    const GlobalIndex n_field_dofs = dh.getNumDofs();
    const int n_components = std::max(1, rec.components);
    std::vector<unsigned char> on_support(static_cast<std::size_t>(n_field_dofs), static_cast<unsigned char>(0));
    const auto mark = [&](GlobalIndex local_dof) {
        if (local_dof < 0 || local_dof >= n_field_dofs) {
            throw std::runtime_error(ctx + ": a cell references DOF " + std::to_string(local_dof) +
                                     " outside the field");
        }
        on_support[static_cast<std::size_t>(local_dof)] = static_cast<unsigned char>(1);
    };

    FaceDofs face_dofs(*rec.space, ctx);
    const auto& mesh = system.meshAccess();
    forEachSupportEntity(system, [&](GlobalIndex cell, int local_face) {
        const auto cell_dofs = dof_map.getCellDofs(cell);
        if (local_face < 0) {
            for (const auto local_dof : cell_dofs) {
                mark(local_dof);
            }
            return;
        }
        componentDofs(cell_dofs, n_components, face_dofs.onFace(mesh.getCellType(cell), local_face), ctx, cell,
                      mark);
    });

    // Owned DOFs on faces or cut cells that only other ranks hold come from
    // those ranks, so the support does not depend on the partition.
    std::vector<GlobalIndex> support;
    for (GlobalIndex d = 0; d < n_field_dofs; ++d) {
        if (on_support[static_cast<std::size_t>(d)] != 0u) {
            support.push_back(offset + d);
        }
    }
    std::vector<Real> no_payload;
    (void)completeOwnedBoundaryDofs(system, support, no_payload, 0u, ctx);
    for (const auto dof : support) {
        const auto local = dof - offset;
        if (local >= 0 && local < n_field_dofs) {
            on_support[static_cast<std::size_t>(local)] = static_cast<unsigned char>(1);
        }
    }

    const auto& owned = system.dofHandler().getPartition().locallyOwned();
    owned_support_dofs_ = 0u;
    pinned_dofs_ = 0u;
    for (GlobalIndex d = 0; d < n_field_dofs; ++d) {
        const GlobalIndex dof = offset + d;
        if (!owned.contains(dof)) {
            continue;
        }
        if (on_support[static_cast<std::size_t>(d)] != 0u) {
            ++owned_support_dofs_;
            continue;
        }
        if (constraints.isConstrained(dof)) {
            continue;
        }
        constraints.addDirichlet(dof, Real{0.0});
        ++pinned_dofs_;
    }
}

SupportRestrictionConstraint::DeadRows SupportRestrictionConstraint::rowsWithoutFreeCoupling(
    const systems::FESystem& system,
    const AffineConstraints& constraints,
    const std::vector<FieldId>& coupled_fields) const
{
    DeadRows out;
    if (coupled_fields.empty()) {
        return out;
    }
    const auto& rec = system.fieldRecord(field_);
    const std::string ctx = context(system);
    const auto& mesh = system.meshAccess();
    const GlobalIndex offset = system.fieldDofOffset(field_);
    const int n_components = std::max(1, rec.components);
    const auto& dof_map = system.fieldDofHandler(field_).getDofMap();
    FaceDofs face_dofs(*rec.space, ctx);

    struct Coupled {
        FieldId field;
        GlobalIndex offset;
        int components;
        const dofs::DofMap* dof_map;
        std::unique_ptr<FaceDofs> faces;
    };
    std::vector<Coupled> coupled;
    for (const auto f : coupled_fields) {
        const auto& crec = system.fieldRecord(f);
        if (!crec.space || crec.scope == systems::FieldScope::Global) {
            continue;
        }
        coupled.push_back(Coupled{f, system.fieldDofOffset(f), std::max(1, crec.components),
                                  &system.fieldDofHandler(f).getDofMap(),
                                  std::make_unique<FaceDofs>(*crec.space, ctx + " coupled field '" + crec.name + "'")});
    }
    // A DOF is fixed when a Dirichlet line (a value, no dependence on other
    // DOFs) constrains it; a line that writes it through other DOFs leaves
    // its row and column alive.
    const auto fixed = [&](GlobalIndex dof) {
        const auto line = constraints.getConstraint(dof);
        return line.has_value() && line->isDirichlet();
    };

    // For each multiplier DOF on the support: whether some entity couples it
    // to a free DOF of a coupled field.
    std::unordered_map<GlobalIndex, bool> alive;
    forEachSupportEntity(system, [&](GlobalIndex cell, int local_face) {
        bool any_free = false;
        for (auto& c : coupled) {
            const auto cell_dofs = c.dof_map->getCellDofs(cell);
            const auto visit = [&](GlobalIndex local_dof) {
                any_free = any_free || !fixed(c.offset + local_dof);
            };
            if (local_face < 0) {
                for (const auto local_dof : cell_dofs) {
                    visit(local_dof);
                }
            } else {
                componentDofs(cell_dofs, c.components, c.faces->onFace(mesh.getCellType(cell), local_face), ctx,
                              cell, visit);
            }
        }
        const auto record = [&](GlobalIndex local_dof) {
            auto& state = alive[offset + local_dof];
            state = state || any_free;
        };
        const auto cell_dofs = dof_map.getCellDofs(cell);
        if (local_face < 0) {
            for (const auto local_dof : cell_dofs) {
                record(local_dof);
            }
        } else {
            componentDofs(cell_dofs, n_components, face_dofs.onFace(mesh.getCellType(cell), local_face), ctx, cell,
                          record);
        }
    });

    // The owner of a DOF decides from what every rank sees of it: the
    // support DOFs and those some entity keeps alive, completed to their
    // owners (collective).
    std::vector<GlobalIndex> seen;
    std::vector<GlobalIndex> kept;
    for (const auto& [dof, is_alive] : alive) {
        seen.push_back(dof);
        if (is_alive) {
            kept.push_back(dof);
        }
    }
    std::sort(seen.begin(), seen.end());
    std::sort(kept.begin(), kept.end());
    std::vector<Real> no_payload;
    (void)completeOwnedBoundaryDofs(system, seen, no_payload, 0u, ctx + " support");
    (void)completeOwnedBoundaryDofs(system, kept, no_payload, 0u, ctx + " coupled support");
    std::sort(kept.begin(), kept.end());
    kept.erase(std::unique(kept.begin(), kept.end()), kept.end());

    const auto& owned = system.dofHandler().getPartition().locallyOwned();
    std::vector<GlobalIndex> dead;
    for (const auto dof : seen) {
        if (owned.contains(dof) && !constraints.isConstrained(dof) &&
            !std::binary_search(kept.begin(), kept.end(), dof)) {
            dead.push_back(dof);
        }
    }
    std::sort(dead.begin(), dead.end());
    dead.erase(std::unique(dead.begin(), dead.end()), dead.end());
    out.count = dead.size();
    if (!dead.empty()) {
        out.first = dead.front();
    }
    return out;
}

bool SupportRestrictionConstraint::updateValues(const systems::FESystem& system,
                                                AffineConstraints& constraints,
                                                double time,
                                                double dt)
{
    (void)system;
    (void)constraints;
    (void)time;
    (void)dt;
    // The pinned values are 0; which DOFs are pinned changes only through a
    // structural rebuild.
    return false;
}

ConstraintDependencyDeclaration SupportRestrictionConstraint::dependencyDeclaration() const
{
    ConstraintDependencyDeclaration out = ISystemConstraint::dependencyDeclaration();
    if (kind_ == SupportDomainKind::GeneratedInterface) {
        // apply() reads the cut cells from the installed cut integration
        // context, not the level-set values: a new cut epoch installs a new
        // context, which rebuilds the constraint state (as for
        // SmallCutAggregationConstraint), so the pinned set follows the cut
        // without a new setup; the field's DOFs stay.
        out.structural.fe_constraint_layout = true;
        out.structural.mesh_field_layout = true;
        out.structural.active_configuration = true;
    }
    return out;
}

systems::SetupStorageRequirements SupportRestrictionConstraint::storageRequirements() const noexcept
{
    systems::SetupStorageRequirements req;
    req.entity_dof_map = true;
    req.boundary_face_topology = kind_ == SupportDomainKind::PhysicalBoundary;
    req.cell_topology = true;
    return req;
}

std::string SupportRestrictionConstraint::manifestDescription() const
{
    std::ostringstream oss;
    oss << "field=" << field_ << " domain=" << supportDomainKindName(kind_) << " marker=" << marker_;
    if (kind_ == SupportDomainKind::GeneratedInterface) {
        oss << " level_set='" << level_set_field_name_ << "'";
    }
    return oss.str();
}

} // namespace constraints
} // namespace FE
} // namespace svmp
