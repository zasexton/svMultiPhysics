/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/ContactPenaltyKernel.h"

#include "Assembly/Assembler.h"
#include "Assembly/AssemblyConstraintDistributor.h"
#include "Assembly/GlobalSystemView.h"

#include "Dofs/DofHandler.h"
#include "Dofs/EntityDofMap.h"

#include "Elements/ReferenceElement.h"
#include "Sparsity/SparsityPattern.h"
#include "Systems/FESystem.h"
#include "Systems/SearchAccess.h"
#include "Systems/SystemsExceptions.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

namespace {

std::array<Real, 3> sub(const std::array<Real, 3>& a, const std::array<Real, 3>& b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

Real dotN(const std::array<Real, 3>& a, const std::array<Real, 3>& b, int n)
{
    Real s = 0.0;
    for (int i = 0; i < n; ++i) {
        s += a[static_cast<std::size_t>(i)] * b[static_cast<std::size_t>(i)];
    }
    return s;
}

Real normN(const std::array<Real, 3>& a, int n)
{
    return std::sqrt(dotN(a, a, n));
}

std::vector<GlobalIndex> faceVertexNodes(const assembly::IMeshAccess& mesh,
                                        GlobalIndex face_id,
                                        GlobalIndex cell_id)
{
    std::vector<GlobalIndex> cell_nodes;
    mesh.getCellNodes(cell_id, cell_nodes);

    const ElementType cell_type = mesh.getCellType(cell_id);
    const LocalIndex local_face = mesh.getLocalFaceIndex(face_id, cell_id);
    const auto ref = elements::ReferenceElement::create(cell_type);
    const auto& local = ref.face_nodes(static_cast<std::size_t>(local_face));

    std::vector<GlobalIndex> face_nodes;
    face_nodes.reserve(local.size());
    for (auto li : local) {
        const auto idx = static_cast<std::size_t>(li);
        FE_THROW_IF(idx >= cell_nodes.size(), FEException,
                    "PenaltyPointContactKernel: face node index out of range");
        face_nodes.push_back(cell_nodes[idx]);
    }
    return face_nodes;
}

std::unordered_set<GlobalIndex> collectBoundaryVertices(const assembly::IMeshAccess& mesh, int marker)
{
    std::unordered_set<GlobalIndex> verts;
    mesh.forEachBoundaryFace(marker, [&](GlobalIndex face_id, GlobalIndex cell_id) {
        const auto nodes = faceVertexNodes(mesh, face_id, cell_id);
        for (auto v : nodes) {
            verts.insert(v);
        }
    });
    return verts;
}

std::vector<GlobalIndex> vertexFieldDofs(const FESystem& system,
                                        FieldId field,
                                        GlobalIndex vertex_id,
                                        int n_components)
{
    const auto& dh = system.fieldDofHandler(field);
    const auto* entity = dh.getEntityDofMap();
    FE_THROW_IF(entity == nullptr, FEException,
                "PenaltyPointContactKernel: field DofHandler has no EntityDofMap");

    const auto dofs = entity->getVertexDofs(vertex_id);
    FE_THROW_IF(dofs.size() < static_cast<std::size_t>(n_components), FEException,
                "PenaltyPointContactKernel: vertex has insufficient DOFs for requested components");

    const GlobalIndex offset = system.fieldDofOffset(field);
    std::vector<GlobalIndex> out(static_cast<std::size_t>(n_components));
    for (int i = 0; i < n_components; ++i) {
        out[static_cast<std::size_t>(i)] = dofs[static_cast<std::size_t>(i)] + offset;
    }
    return out;
}

std::array<Real, 3> deformedPosition(const assembly::IMeshAccess& mesh,
                                     const SystemStateView& state,
                                     GlobalIndex vertex_id,
                                     std::span<const GlobalIndex> vertex_dofs,
                                     int n_components)
{
    auto x = mesh.getNodeCoordinates(vertex_id);
    if (!state.u.empty()) {
        for (int d = 0; d < n_components; ++d) {
            const auto dof = vertex_dofs[static_cast<std::size_t>(d)];
            FE_THROW_IF(dof < 0, FEException,
                        "PenaltyPointContactKernel: negative DOF index");
            const auto idx = static_cast<std::size_t>(dof);
            FE_THROW_IF(idx >= state.u.size(), FEException,
                        "PenaltyPointContactKernel: state.u too small for DOF access");
            x[static_cast<std::size_t>(d)] += state.u[idx];
        }
    }
    return x;
}

} // namespace

PenaltyPointContactKernel::PenaltyPointContactKernel(PenaltyContactConfig cfg)
    : cfg_(cfg)
{
    FE_THROW_IF(cfg_.field == INVALID_FIELD_ID, InvalidArgumentException,
                "PenaltyPointContactKernel: field == INVALID_FIELD_ID");
    FE_THROW_IF(cfg_.slave_marker == cfg_.master_marker, InvalidArgumentException,
                "PenaltyPointContactKernel: slave_marker must differ from master_marker");
    FE_THROW_IF(cfg_.search_radius <= Real(0), InvalidArgumentException,
                "PenaltyPointContactKernel: search_radius must be > 0");
    FE_THROW_IF(cfg_.activation_distance <= Real(0), InvalidArgumentException,
                "PenaltyPointContactKernel: activation_distance must be > 0");
    FE_THROW_IF(cfg_.penalty <= Real(0), InvalidArgumentException,
                "PenaltyPointContactKernel: penalty must be > 0");
}

void PenaltyPointContactKernel::addSparsityCouplings(const FESystem& system,
                                                     sparsity::SparsityPattern& pattern) const
{
    const auto& mesh = system.meshAccess();
    const int dim = mesh.dimension();
    if (dim <= 0 || dim > 3) {
        return;
    }

    const auto slave_vertices = collectBoundaryVertices(mesh, cfg_.slave_marker);
    const auto master_vertices = collectBoundaryVertices(mesh, cfg_.master_marker);
    if (slave_vertices.empty() || master_vertices.empty()) {
        return;
    }

    std::vector<GlobalIndex> slave_dofs;
    std::vector<GlobalIndex> master_dofs;

    slave_dofs.reserve(slave_vertices.size() * static_cast<std::size_t>(dim));
    master_dofs.reserve(master_vertices.size() * static_cast<std::size_t>(dim));

    for (auto v : slave_vertices) {
        const auto dofs = vertexFieldDofs(system, cfg_.field, v, dim);
        slave_dofs.insert(slave_dofs.end(), dofs.begin(), dofs.end());
    }

    for (auto v : master_vertices) {
        const auto dofs = vertexFieldDofs(system, cfg_.field, v, dim);
        master_dofs.insert(master_dofs.end(), dofs.begin(), dofs.end());
    }

    std::sort(slave_dofs.begin(), slave_dofs.end());
    slave_dofs.erase(std::unique(slave_dofs.begin(), slave_dofs.end()), slave_dofs.end());
    std::sort(master_dofs.begin(), master_dofs.end());
    master_dofs.erase(std::unique(master_dofs.begin(), master_dofs.end()), master_dofs.end());

    pattern.addElementCouplings(slave_dofs);
    pattern.addElementCouplings(master_dofs);
    pattern.addElementCouplings(slave_dofs, master_dofs);
    pattern.addElementCouplings(master_dofs, slave_dofs);
}

assembly::AssemblyResult PenaltyPointContactKernel::assemble(
    const FESystem& system,
    const AssemblyRequest& request,
    const SystemStateView& state,
    assembly::GlobalSystemView* matrix_out,
    assembly::GlobalSystemView* vector_out)
{
    assembly::AssemblyResult result;

    const auto* search = system.searchAccess();
    FE_THROW_IF(search == nullptr, FEException,
                "PenaltyPointContactKernel: no search access is configured (FESystem::searchAccess is null)");

    const auto& mesh = system.meshAccess();
    const int dim = mesh.dimension();
    FE_THROW_IF(dim <= 0 || dim > 3, FEException,
                "PenaltyPointContactKernel: unsupported mesh dimension");

    // Ensure the backend is in an assembly-ready phase for "global-only" operators.
    if (matrix_out) matrix_out->beginAssemblyPhase();
    if (vector_out && vector_out != matrix_out) vector_out->beginAssemblyPhase();

    search->build();

    const auto slave_vertices = collectBoundaryVertices(mesh, cfg_.slave_marker);
    const auto master_vertices = collectBoundaryVertices(mesh, cfg_.master_marker);

    if (slave_vertices.empty() || master_vertices.empty()) {
        return result;
    }

    assembly::AssemblyConstraintDistributor distributor(system.constraints());

    std::vector<GlobalIndex> dofs;
    std::vector<Real> local_rhs;
    std::vector<Real> local_mat;

    const Real h = cfg_.activation_distance;
    const Real k = cfg_.penalty;
    const Real eps = Real(1e-14);

    for (const auto v_slave : slave_vertices) {
        const auto slave_dofs = vertexFieldDofs(system, cfg_.field, v_slave, dim);
        const auto x_slave = deformedPosition(mesh, state, v_slave, slave_dofs, dim);

        auto candidates = search->verticesInRadius(x_slave, cfg_.search_radius);

        GlobalIndex best_master = INVALID_GLOBAL_INDEX;
        Real best_dist = std::numeric_limits<Real>::infinity();

        for (auto v_master : candidates) {
            if (v_master == v_slave) continue;
            if (master_vertices.find(v_master) == master_vertices.end()) continue;

            const auto master_dofs = vertexFieldDofs(system, cfg_.field, v_master, dim);
            const auto x_master = deformedPosition(mesh, state, v_master, master_dofs, dim);

            const auto r = sub(x_slave, x_master);
            const Real d = std::max(normN(r, dim), eps);
            if (d >= best_dist) continue;

            best_dist = d;
            best_master = v_master;
            if (cfg_.only_nearest) {
                // continue searching for potentially closer candidates
            }
        }

        if (best_master == INVALID_GLOBAL_INDEX) {
            continue;
        }

        const auto master_dofs = vertexFieldDofs(system, cfg_.field, best_master, dim);
        const auto x_master = deformedPosition(mesh, state, best_master, master_dofs, dim);

        const auto r = sub(x_slave, x_master);
        const Real d = std::max(normN(r, dim), eps);
        if (d >= h) {
            continue;
        }

        const Real gap = h - d;
        std::array<Real, 3> uhat{Real(0), Real(0), Real(0)};
        for (int i = 0; i < dim; ++i) {
            uhat[static_cast<std::size_t>(i)] = r[static_cast<std::size_t>(i)] / d;
        }

        // Force on slave, equal/opposite on master.
        std::array<Real, 3> f{Real(0), Real(0), Real(0)};
        for (int i = 0; i < dim; ++i) {
            f[static_cast<std::size_t>(i)] = k * gap * uhat[static_cast<std::size_t>(i)];
        }

        dofs.clear();
        dofs.insert(dofs.end(), slave_dofs.begin(), slave_dofs.end());
        dofs.insert(dofs.end(), master_dofs.begin(), master_dofs.end());

        local_rhs.assign(dofs.size(), Real(0));
        if (request.want_vector && vector_out != nullptr) {
            for (int i = 0; i < dim; ++i) {
                local_rhs[static_cast<std::size_t>(i)] = f[static_cast<std::size_t>(i)];
                local_rhs[static_cast<std::size_t>(dim + i)] = -f[static_cast<std::size_t>(i)];
            }
        }

        local_mat.clear();
        if (request.want_matrix && matrix_out != nullptr) {
            const auto n = static_cast<std::size_t>(2 * dim);
            local_mat.assign(n * n, Real(0));

            // Jacobian dF/dr for F = k*(h-d)*r/d.
            const Real a = k * (h / d - Real(1));
            const Real b = k * h / (d * d * d);

            // J = a I - b (r ⊗ r).
            std::array<std::array<Real, 3>, 3> J{};
            for (int i = 0; i < dim; ++i) {
                for (int j = 0; j < dim; ++j) {
                    const Real ij = (i == j) ? a : Real(0);
                    J[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                        ij - b * r[static_cast<std::size_t>(i)] * r[static_cast<std::size_t>(j)];
                }
            }

            auto at = [&](int row, int col) -> Real& {
                return local_mat[static_cast<std::size_t>(row) * n + static_cast<std::size_t>(col)];
            };

            // Blocks:
            // [ J  -J ]
            // [ -J  J ]
            for (int i = 0; i < dim; ++i) {
                for (int j = 0; j < dim; ++j) {
                    const Real v = J[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
                    at(i, j) += v;
                    at(i, dim + j) -= v;
                    at(dim + i, j) -= v;
                    at(dim + i, dim + j) += v;
                }
            }
        }

        if (request.want_matrix && matrix_out != nullptr && request.want_vector && vector_out != nullptr) {
            distributor.distributeLocalToGlobal(local_mat, local_rhs, dofs, *matrix_out, *vector_out);
            result.matrix_entries_inserted += static_cast<GlobalIndex>(dofs.size() * dofs.size());
            result.vector_entries_inserted += static_cast<GlobalIndex>(dofs.size());
        } else if (request.want_matrix && matrix_out != nullptr) {
            distributor.distributeMatrixToGlobal(local_mat, dofs, *matrix_out);
            result.matrix_entries_inserted += static_cast<GlobalIndex>(dofs.size() * dofs.size());
        } else if (request.want_vector && vector_out != nullptr) {
            distributor.distributeVectorToGlobal(local_rhs, dofs, *vector_out);
            result.vector_entries_inserted += static_cast<GlobalIndex>(dofs.size());
        }
    }

    return result;
}

} // namespace systems
} // namespace FE
} // namespace svmp
