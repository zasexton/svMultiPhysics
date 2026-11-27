#ifndef SVMP_FE_FORMS_CONSTRAINT_BCS_H
#define SVMP_FE_FORMS_CONSTRAINT_BCS_H

/**
 * @file ConstraintBCs.h
 * @brief BoundaryCondition wrappers for algebraic (topological) constraints
 */

#include "Forms/BoundaryCondition.h"

#include "Constraints/ConstraintTools.h"
#include "Constraints/MultiPointConstraint.h"

#include "Geometry/MappingFactory.h"
#include "Spaces/FaceRestriction.h"
#include "Systems/FESystem.h"
#include "Systems/SystemCollectives.h"
#include "Systems/SystemsExceptions.h"

#include <algorithm>
#include <array>
#include <bit>
#include <map>
#include <set>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace bc {

namespace detail {

struct BoundaryPointDofsWithCoords {
    int num_components{1};
    std::vector<std::array<double, 3>> coords{};
    std::vector<GlobalIndex> dofs{}; // interleaved by component: [pt0_c0, pt0_c1, ..., pt1_c0, ...]
};

[[nodiscard]] inline BoundaryPointDofsWithCoords boundaryPointDofsWithCoordsByMarker(
    const systems::FESystem& system,
    FieldId field,
    int boundary_marker)
{
    BoundaryPointDofsWithCoords out;

    const auto& rec = system.fieldRecord(field);
    FE_CHECK_NOT_NULL(rec.space.get(), "boundaryPointDofsWithCoordsByMarker: field space");

    const int n_components = std::max(1, rec.space->value_dimension());
    out.num_components = n_components;

    const auto& mesh = system.meshAccess();
    const auto& dh = system.fieldDofHandler(field);
    const auto& dof_map = dh.getDofMap();
    const GlobalIndex offset = system.fieldDofOffset(field);

    struct PointEntry {
        std::array<double, 3> coord{};
        std::vector<GlobalIndex> dofs{};
    };
    std::map<GlobalIndex, PointEntry> by_base_dof;
    std::map<std::pair<ElementType, const elements::Element*>, spaces::FaceRestriction> restrictions;

    std::vector<std::array<Real, 3>> cell_coords;
    std::vector<math::Vector<Real, 3>> geom_nodes;

    mesh.forEachBoundaryFace(boundary_marker, [&](GlobalIndex face_id, GlobalIndex cell_id) {
        const ElementType cell_type = mesh.getCellType(cell_id);

        const auto& element = rec.space->getElement(cell_type, cell_id);
        FE_CHECK_ARG(!element.basis().is_vector_valued() &&
                         rec.space->continuity() != Continuity::H_div && rec.space->continuity() != Continuity::H_curl,
                     "PeriodicBC: coordinate matching requires a nodal field");
        const auto type = to_mesh_family(element.element_type()) == to_mesh_family(cell_type)
                              ? element.element_type() : cell_type;
        const auto key = std::make_pair(type, &element);
        auto restriction = restrictions.find(key);
        if (restriction == restrictions.end()) {
            restriction = type == rec.space->element_type() && &element == &rec.space->element()
                     ? restrictions.try_emplace(key, *rec.space).first
                     : restrictions.try_emplace(key, type, element.polynomial_order(), rec.space->continuity(),
                                                element.basis().basis_type()).first;
        }
        const auto& face_restriction = restriction->second;
        const auto face_local = static_cast<int>(mesh.getLocalFaceIndex(face_id, cell_id));
        const auto local_face_dofs = face_restriction.face_dofs(face_local);

        const auto cell_dofs = dof_map.getCellDofs(cell_id);
        FE_THROW_IF(cell_dofs.empty(), systems::InvalidStateException,
                    "boundaryPointDofsWithCoordsByMarker: empty cell DOF list");

        const auto dofs_per_component = cell_dofs.size() / static_cast<std::size_t>(n_components);
        FE_THROW_IF(dofs_per_component == 0 ||
                        dofs_per_component * static_cast<std::size_t>(n_components) != cell_dofs.size(),
                    systems::InvalidStateException,
                    "boundaryPointDofsWithCoordsByMarker: cell DOF list size is not divisible by field components");

        cell_coords.clear();
        mesh.getCellCoordinates(cell_id, cell_coords);
        FE_THROW_IF(cell_coords.empty(), systems::InvalidStateException,
                    "boundaryPointDofsWithCoordsByMarker: empty cell coordinate list");

        geom_nodes.resize(cell_coords.size());
        for (std::size_t i = 0; i < cell_coords.size(); ++i) {
            geom_nodes[i][0] = cell_coords[i][0];
            geom_nodes[i][1] = cell_coords[i][1];
            geom_nodes[i][2] = cell_coords[i][2];
        }

        geometry::MappingRequest req;
        req.element_type = cell_type;
        req.geometry_order = mesh.getCellGeometryOrder(cell_id);
        req.use_affine = req.geometry_order <= 1;
        auto mapping = geometry::MappingFactory::create(req, geom_nodes);
        FE_CHECK_NOT_NULL(mapping.get(), "boundaryPointDofsWithCoordsByMarker: mapping");

        const auto& dof_nodes = face_restriction.dof_nodes();

        for (int ldof : local_face_dofs) {
            FE_THROW_IF(ldof < 0, InvalidArgumentException,
                        "boundaryPointDofsWithCoordsByMarker: negative local DOF index");
            const auto ldof_u = static_cast<std::size_t>(ldof);
            FE_THROW_IF(ldof_u >= dofs_per_component, InvalidArgumentException,
                        "boundaryPointDofsWithCoordsByMarker: local DOF index out of range for cell DOF list");
            FE_THROW_IF(ldof_u >= dof_nodes.size(), InvalidArgumentException,
                        "boundaryPointDofsWithCoordsByMarker: local DOF index out of range for dof_nodes");

            const auto x = mapping->map_to_physical(dof_nodes[ldof_u]);

            const GlobalIndex base_dof = cell_dofs[ldof_u] + offset;
            auto it = by_base_dof.find(base_dof);
            if (it != by_base_dof.end()) {
                continue;
            }

            PointEntry entry;
            entry.coord = {static_cast<double>(x[0]),
                           static_cast<double>(x[1]),
                           static_cast<double>(x[2])};
            entry.dofs.resize(static_cast<std::size_t>(n_components));
            for (int comp = 0; comp < n_components; ++comp) {
                const auto local = static_cast<std::size_t>(comp) * dofs_per_component + ldof_u;
                FE_THROW_IF(local >= cell_dofs.size(), systems::InvalidStateException,
                            "boundaryPointDofsWithCoordsByMarker: component-local DOF index out of range");
                entry.dofs[static_cast<std::size_t>(comp)] = cell_dofs[local] + offset;
            }

            by_base_dof.emplace(base_dof, std::move(entry));
        }
    });

    for (const auto& [base, point] : by_base_dof) {
        out.coords.push_back(point.coord);
        out.dofs.insert(out.dofs.end(), point.dofs.begin(), point.dofs.end());
    }
    return out;
}

/// Pack the point and its component DOF IDs together so opposite markers
/// can be matched even when their incident cells live on different ranks.
inline std::array<BoundaryPointDofsWithCoords, 2> gatherPeriodicBoundaries(
    const systems::FESystem& system, FieldId field, const std::array<int, 2>& markers)
{
    systems::CollectiveCommunicator comm;
#if FE_HAS_MPI
    comm = systems::CollectiveCommunicator(system.activeMpiCommunicator());
#endif
    std::vector<std::uint64_t> local;
    std::size_t components = 0;
    std::exception_ptr failure;
    try {
        for (std::size_t side = 0; side < markers.size(); ++side) {
            const auto boundary = boundaryPointDofsWithCoordsByMarker(system, field, markers[side]);
            components = static_cast<std::size_t>(boundary.num_components);
            for (std::size_t i = 0; i < boundary.coords.size(); ++i) {
                local.push_back(side);
                for (const double x : boundary.coords[i]) local.push_back(std::bit_cast<std::uint64_t>(x));
                for (std::size_t c = 0; c < components; ++c) local.push_back(boundary.dofs[i * components + c]);
            }
        }
    } catch (...) {
        failure = std::current_exception();
    }
    systems::requireCollectivePhasePreparation(failure, comm, "periodic boundary coordinates");
    const auto words = systems::collectiveGatherWords(local, comm, "Periodic boundary coordinates");
    const auto stride = 4 + components;
    FE_CHECK_ARG(words.size() % stride == 0, "PeriodicBC: inconsistent component count across ranks");
    std::array<BoundaryPointDofsWithCoords, 2> boundaries;
    std::array<std::set<GlobalIndex>, 2> seen;
    for (std::size_t start = 0; start < words.size(); start += stride) {
        const auto side = words[start];
        FE_CHECK_ARG(side < boundaries.size(), "PeriodicBC: inconsistent boundary record across ranks");
        const auto first = static_cast<GlobalIndex>(words[start + 4]);
        if (!seen[side].insert(first).second) continue;
        auto& boundary = boundaries[side];
        boundary.coords.push_back({std::bit_cast<double>(words[start + 1]), std::bit_cast<double>(words[start + 2]),
                                   std::bit_cast<double>(words[start + 3])});
        for (std::size_t c = 0; c < components; ++c) boundary.dofs.push_back(words[start + 4 + c]);
    }
    return boundaries;
}

} // namespace detail

/**
 * @brief Periodic constraint between two boundary markers
 *
 * Establishes u_slave = u_master for matching DOF locations on the two
 * markers, using coordinate matching after applying the provided translation
 * vector to slave coordinates.
 *
 * Notes:
 * - This is a purely algebraic constraint; it does not contribute to the weak form.
 * - Applies to all components of the target field.
 */
class PeriodicBC final : public BoundaryCondition {
public:
    PeriodicBC(int master_marker,
               int slave_marker,
               std::vector<double> translation_vector)
        : master_marker_(master_marker)
        , slave_marker_(slave_marker)
    {
        if (master_marker_ < 0 || slave_marker_ < 0) {
            throw std::invalid_argument("PeriodicBC: master_marker and slave_marker must be >= 0");
        }
        if (translation_vector.size() != 3u) {
            throw std::invalid_argument("PeriodicBC: translation_vector must have length 3");
        }
        translation_ = {translation_vector[0], translation_vector[1], translation_vector[2]};
    }

    [[nodiscard]] int boundaryMarker() const override { return slave_marker_; }
    [[nodiscard]] std::string_view label() const override { return "PeriodicBC"; }
    [[nodiscard]] bool hasWeakTerms() const override { return false; }

    void setup(systems::FESystem& system, FieldId /*field_id*/) override
    {
        system_ = &system;
    }

    void contributeToResidual(FormExpr& /*residual*/,
                              const FormExpr& /*u*/,
                              const FormExpr& /*v*/) const override
    {
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId /*field_id*/) const override
    {
        return {};
    }

    void addAffineConstraints(constraints::AffineConstraints& constraints, FieldId field_id) const override
    {
        FE_CHECK_NOT_NULL(system_, "PeriodicBC::addAffineConstraints: missing system pointer (setup not called)");

        const auto boundaries = detail::gatherPeriodicBoundaries(*system_, field_id, {slave_marker_, master_marker_});
        const auto& slave = boundaries[0];
        const auto& master = boundaries[1];
        const auto translate = [this](std::array<double, 3> p) {
            for (std::size_t d = 0; d < p.size(); ++d) p[d] += translation_[d];
            return p;
        };
        auto pairs = constraints::matchNodalPeriodicDofs(slave.dofs, slave.coords, master.dofs, master.coords,
                                                        translate, {.matching_tolerance = matching_tolerance_});
        const auto& owned = system_->dofHandler().getPartition().locallyOwned();
        std::erase_if(pairs, [&](const auto& pair) { return !owned.contains(pair.slave_dof); });
        constraints::PeriodicBC(std::move(pairs)).apply(constraints);
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId field_id, const systems::FESystem* /*system*/) const override
    {
        analysis::BoundaryConditionDescriptor d;
        d.primary_variable = analysis::VariableKey::field(field_id);
        d.boundary_marker = slave_marker_;
        d.trace_kind = analysis::TraceKind::AlgebraicRelation;
        d.enforcement_kind = analysis::EnforcementKind::AffineRelation;
        d.anchors_constant_mode = false;
        d.anchors_rigid_body_translation = false;
        d.anchors_rigid_body_rotation = false;
        d.source = "PeriodicBC slave=" + std::to_string(slave_marker_)
                 + " master=" + std::to_string(master_marker_);
        return {d};
    }

private:
    int master_marker_{-1};
    int slave_marker_{-1};
    std::array<double, 3> translation_{0.0, 0.0, 0.0};
    double matching_tolerance_{1e-10};

    const systems::FESystem* system_{nullptr};
};

/**
 * @brief Multi-point constraint BC wrapper
 *
 * Adds general linear DOF relations to the system's affine constraints.
 */
class MultiPointConstraintBC final : public BoundaryCondition {
public:
    explicit MultiPointConstraintBC(std::vector<constraints::MPCEquation> equations,
                                    constraints::MPCOptions options = {})
        : mpc_(std::move(equations), options)
    {
    }

    explicit MultiPointConstraintBC(constraints::MultiPointConstraint mpc)
        : mpc_(std::move(mpc))
    {
    }

    [[nodiscard]] int boundaryMarker() const override { return -1; }
    [[nodiscard]] std::string_view label() const override { return "MultiPointConstraintBC"; }
    [[nodiscard]] bool hasWeakTerms() const override { return false; }

    void contributeToResidual(FormExpr& /*residual*/,
                              const FormExpr& /*u*/,
                              const FormExpr& /*v*/) const override
    {
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId /*field_id*/) const override
    {
        return {};
    }

    void addAffineConstraints(constraints::AffineConstraints& constraints, FieldId /*field_id*/) const override
    {
        mpc_.apply(constraints);
    }

private:
    constraints::MultiPointConstraint mpc_{};
};

} // namespace bc
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_CONSTRAINT_BCS_H
