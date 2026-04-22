/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Constraints/VertexDirichletConstraint.h"

#include "Dofs/EntityDofMap.h"
#include "Spaces/FunctionSpace.h"
#include "Systems/FESystem.h"

#include <sstream>
#include <stdexcept>
#include <utility>

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Mesh.h"
#endif

#if FE_HAS_MPI
#  include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace constraints {

namespace {

[[nodiscard]] bool mpiInitialized() noexcept
{
#if FE_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    return initialized != 0;
#else
    return false;
#endif
}

[[nodiscard]] std::vector<int> allreduceMax(std::vector<int> values, const systems::FESystem& system)
{
#if FE_HAS_MPI
    if (!mpiInitialized() || values.empty()) {
        return values;
    }
    std::vector<int> reduced(values.size(), 0);
    MPI_Allreduce(values.data(),
                  reduced.data(),
                  static_cast<int>(values.size()),
                  MPI_INT,
                  MPI_MAX,
                  system.dofHandler().mpiComm());
    return reduced;
#else
    (void)system;
    return values;
#endif
}

[[nodiscard]] GlobalIndex resolveLocalVertex(const systems::FESystem& system,
                                             GlobalIndex vertex_id,
                                             VertexIdMode mode)
{
    if (vertex_id < 0) {
        throw std::invalid_argument("VertexDirichletConstraint: vertex_id must be non-negative");
    }

    if (mode == VertexIdMode::LocalVertexId) {
        return vertex_id;
    }

    if (mode != VertexIdMode::GlobalVertexGid) {
        throw std::invalid_argument("VertexDirichletConstraint: unsupported vertex id mode");
    }

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    const auto* mesh = system.mesh();
    if (mesh == nullptr) {
        throw std::invalid_argument(
            "VertexDirichletConstraint: GlobalVertexGid mode requires FESystem constructed from svmp::Mesh");
    }

    const auto local = mesh->base().global_to_local_vertex(static_cast<svmp::gid_t>(vertex_id));
    return static_cast<GlobalIndex>(local);
#else
    (void)system;
    throw std::invalid_argument(
        "VertexDirichletConstraint: GlobalVertexGid mode requires FE built with Mesh support");
#endif
}

} // namespace

VertexDirichletConstraint::VertexDirichletConstraint(FieldId field,
                                                     std::vector<VertexDirichletValue> values,
                                                     VertexIdMode mode)
    : field_(field)
    , values_(std::move(values))
    , mode_(mode)
{
    if (field_ == INVALID_FIELD_ID) {
        throw std::invalid_argument("VertexDirichletConstraint: invalid FieldId");
    }
    for (const auto& v : values_) {
        if (v.vertex_id < 0) {
            throw std::invalid_argument("VertexDirichletConstraint: vertex_id must be non-negative");
        }
    }
}

void VertexDirichletConstraint::apply(const systems::FESystem& system, AffineConstraints& constraints)
{
    const auto& rec = system.fieldRecord(field_);
    if (!rec.space) {
        throw std::invalid_argument("VertexDirichletConstraint: field has no function space");
    }
    if (rec.space->space_type() != spaces::SpaceType::H1 ||
        rec.space->continuity() != Continuity::C0 ||
        rec.space->value_dimension() != 1 ||
        rec.components != 1) {
        throw std::invalid_argument("VertexDirichletConstraint requires a scalar H1/C0 field");
    }

    const auto& dh = system.fieldDofHandler(field_);
    const auto* entity_map = dh.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::invalid_argument("VertexDirichletConstraint: field DofHandler has no EntityDofMap");
    }

    const GlobalIndex offset = system.fieldDofOffset(field_);
    const auto& owned = system.dofHandler().getPartition().locallyOwned();

    struct ResolvedConstraint {
        GlobalIndex dof{-1};
        Real value{0.0};
    };
    std::vector<ResolvedConstraint> resolved;
    resolved.reserve(values_.size());

    std::vector<int> found(values_.size(), 0);
    for (std::size_t i = 0; i < values_.size(); ++i) {
        const auto& value = values_[i];
        const GlobalIndex local_vertex = resolveLocalVertex(system, value.vertex_id, mode_);
        if (local_vertex < 0) {
            continue;
        }
        found[i] = 1;

        if (local_vertex >= entity_map->numVertices()) {
            std::ostringstream oss;
            oss << "VertexDirichletConstraint: local vertex " << local_vertex
                << " is outside the field EntityDofMap vertex range";
            throw std::invalid_argument(oss.str());
        }

        const auto vertex_dofs = entity_map->getVertexDofs(local_vertex);
        if (vertex_dofs.size() != 1u) {
            std::ostringstream oss;
            oss << "VertexDirichletConstraint: expected exactly one scalar H1 vertex DOF for vertex "
                << value.vertex_id << ", found " << vertex_dofs.size();
            throw std::invalid_argument(oss.str());
        }

        const GlobalIndex dof = vertex_dofs.front() + offset;
        if (owned.contains(dof)) {
            resolved.push_back(ResolvedConstraint{dof, value.value});
        }
    }

    const auto globally_found = allreduceMax(std::move(found), system);
    for (std::size_t i = 0; i < globally_found.size(); ++i) {
        if (globally_found[i] == 0) {
            std::ostringstream oss;
            oss << "VertexDirichletConstraint: vertex id " << values_[i].vertex_id
                << " was not found in the mesh";
            throw std::invalid_argument(oss.str());
        }
    }

    for (const auto& c : resolved) {
        constraints.addDirichlet(c.dof, c.value);
    }
}

bool VertexDirichletConstraint::updateValues(const systems::FESystem& system,
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

} // namespace constraints
} // namespace FE
} // namespace svmp
