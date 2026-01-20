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
#include "Systems/SystemsExceptions.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

[[nodiscard]] inline spaces::FaceRestriction& getFaceRestriction(
    std::unordered_map<int, spaces::FaceRestriction>& cache,
    ElementType element_type,
    int order,
    Continuity continuity)
{
    const int key = static_cast<int>(element_type);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    auto [ins, inserted] = cache.emplace(key, spaces::FaceRestriction(element_type, order, continuity));
    (void)inserted;
    return ins->second;
}

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
    std::unordered_map<GlobalIndex, PointEntry> by_base_dof;
    std::unordered_map<int, spaces::FaceRestriction> face_restriction_cache;

    std::vector<std::array<Real, 3>> cell_coords;
    std::vector<math::Vector<Real, 3>> geom_nodes;

    mesh.forEachBoundaryFace(boundary_marker, [&](GlobalIndex face_id, GlobalIndex cell_id) {
        const ElementType cell_type = mesh.getCellType(cell_id);

        auto& face_restriction = getFaceRestriction(face_restriction_cache,
                                                    cell_type,
                                                    rec.space->polynomial_order(),
                                                    rec.space->continuity());

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
        req.geometry_order = 1;
        req.use_affine = true;
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

    std::vector<GlobalIndex> keys;
    keys.reserve(by_base_dof.size());
    for (const auto& [base, _] : by_base_dof) {
        keys.push_back(base);
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

    out.coords.reserve(keys.size());
    out.dofs.reserve(keys.size() * static_cast<std::size_t>(n_components));
    for (const auto base : keys) {
        auto it = by_base_dof.find(base);
        FE_THROW_IF(it == by_base_dof.end(), systems::InvalidStateException,
                    "boundaryPointDofsWithCoordsByMarker: internal map missing key");
        out.coords.push_back(it->second.coord);
        for (int comp = 0; comp < n_components; ++comp) {
            out.dofs.push_back(it->second.dofs[static_cast<std::size_t>(comp)]);
        }
    }

    return out;
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

        const auto slave = detail::boundaryPointDofsWithCoordsByMarker(*system_, field_id, slave_marker_);
        const auto master = detail::boundaryPointDofsWithCoordsByMarker(*system_, field_id, master_marker_);

        const int nc = slave.num_components;
        FE_THROW_IF(nc != master.num_components, InvalidArgumentException,
                    "PeriodicBC::addAffineConstraints: component count mismatch");
        FE_THROW_IF(nc < 1, InvalidArgumentException,
                    "PeriodicBC::addAffineConstraints: invalid component count");

        const std::array<double, 3> t = translation_;
        auto transform = [t](std::array<double, 3> p) -> std::array<double, 3> {
            return {{p[0] + t[0], p[1] + t[1], p[2] + t[2]}};
        };

        const auto matches = constraints::findMatchingPoints(
            slave.coords, master.coords, transform, /*tolerance=*/matching_tolerance_);

        for (const auto& [slave_idx, master_idx] : matches) {
            const std::size_t s0 = slave_idx * static_cast<std::size_t>(nc);
            const std::size_t m0 = master_idx * static_cast<std::size_t>(nc);
            for (int comp = 0; comp < nc; ++comp) {
                const auto s = s0 + static_cast<std::size_t>(comp);
                const auto m = m0 + static_cast<std::size_t>(comp);
                FE_THROW_IF(s >= slave.dofs.size() || m >= master.dofs.size(), InvalidArgumentException,
                            "PeriodicBC::addAffineConstraints: DOF index out of range");

                const GlobalIndex slave_dof = slave.dofs[s];
                const GlobalIndex master_dof = master.dofs[m];

                constraints.addLine(slave_dof);
                constraints.addEntry(slave_dof, master_dof, 1.0);
            }
        }
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
