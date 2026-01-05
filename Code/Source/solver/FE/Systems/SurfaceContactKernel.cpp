/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/SurfaceContactKernel.h"

#include "Assembly/AssemblyConstraintDistributor.h"
#include "Assembly/GlobalSystemView.h"

#include "Core/FEException.h"

#include "Elements/ElementTransform.h"
#include "Geometry/MappingFactory.h"
#include "Quadrature/QuadratureFactory.h"
#include "Quadrature/SurfaceQuadrature.h"
#include "Sparsity/SparsityPattern.h"
#include "Spaces/ProductSpace.h"
#include "Systems/FESystem.h"
#include "Systems/SearchAccess.h"
#include "Systems/SystemsExceptions.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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

math::Vector<Real, 3> toVec(const std::array<Real, 3>& x)
{
    return math::Vector<Real, 3>{x[0], x[1], x[2]};
}

std::array<Real, 3> toArray(const math::Vector<Real, 3>& x)
{
    return {x[0], x[1], x[2]};
}

struct CellCache {
    ElementType cell_type{ElementType::Unknown};
    std::shared_ptr<geometry::GeometryMapping> mapping{};
    std::vector<GlobalIndex> dofs_global{};
    std::vector<Real> u_local{};
};

const CellCache& getCellCache(std::unordered_map<GlobalIndex, CellCache>& cache,
                              const FESystem& system,
                              FieldId field,
                              GlobalIndex cell_id,
                              const SystemStateView& state)
{
    auto it = cache.find(cell_id);
    if (it != cache.end()) {
        return it->second;
    }

    const auto& mesh = system.meshAccess();
    const int dim = mesh.dimension();

    CellCache cc;
    cc.cell_type = mesh.getCellType(cell_id);

    std::vector<std::array<Real, 3>> coords;
    mesh.getCellCoordinates(cell_id, coords);
    std::vector<math::Vector<Real, 3>> nodes;
    nodes.reserve(coords.size());
    for (const auto& x : coords) {
        nodes.push_back(toVec(x));
    }

    geometry::MappingRequest req;
    req.element_type = cc.cell_type;
    req.geometry_order = 1;
    req.use_affine = true;
    cc.mapping = geometry::MappingFactory::create(req, nodes);

    const auto cell_dofs = system.fieldDofHandler(field).getCellDofs(cell_id);
    cc.dofs_global.resize(cell_dofs.size());
    cc.u_local.resize(cell_dofs.size(), Real(0));

    const GlobalIndex offset = system.fieldDofOffset(field);
    for (std::size_t i = 0; i < cell_dofs.size(); ++i) {
        const GlobalIndex dof = cell_dofs[i] + offset;
        cc.dofs_global[i] = dof;
        if (!state.u.empty()) {
            FE_THROW_IF(dof < 0, FEException, "PenaltySurfaceContactKernel: negative DOF index");
            const auto idx = static_cast<std::size_t>(dof);
            FE_THROW_IF(idx >= state.u.size(), FEException,
                        "PenaltySurfaceContactKernel: state.u too small for DOF access");
            cc.u_local[i] = state.u[idx];
        }
    }

    // Basic sanity: expected layout is [comp0 dofs][comp1 dofs]... for a ProductSpace.
    FE_THROW_IF(dim <= 0 || dim > 3, FEException, "PenaltySurfaceContactKernel: unsupported mesh dimension");
    FE_THROW_IF(cc.u_local.size() % static_cast<std::size_t>(dim) != 0u, FEException,
                "PenaltySurfaceContactKernel: field DOFs-per-cell not divisible by mesh dimension");

    auto [ins_it, ok] = cache.emplace(cell_id, std::move(cc));
    FE_THROW_IF(!ok, FEException, "PenaltySurfaceContactKernel: failed to cache cell data");
    return ins_it->second;
}

std::array<Real, 3> evaluateDisplacementAt(int dim,
                                           std::span<const Real> N,
                                           std::span<const Real> u_local)
{
    std::array<Real, 3> u{Real(0), Real(0), Real(0)};
    const std::size_t n_base = N.size();
    FE_THROW_IF(n_base == 0u, FEException, "PenaltySurfaceContactKernel: empty basis values");
    FE_THROW_IF(u_local.size() % n_base != 0u, FEException,
                "PenaltySurfaceContactKernel: unexpected u_local size for basis");

    for (int d = 0; d < dim; ++d) {
        const std::size_t base = static_cast<std::size_t>(d) * n_base;
        FE_THROW_IF(base + n_base > u_local.size(), FEException,
                    "PenaltySurfaceContactKernel: u_local layout mismatch");
        Real s = 0.0;
        for (std::size_t i = 0; i < n_base; ++i) {
            s += N[i] * u_local[base + i];
        }
        u[static_cast<std::size_t>(d)] = s;
    }
    return u;
}

struct ContactKinematics {
    bool active{false};
    std::array<Real, 3> force{Real(0), Real(0), Real(0)}; // force on slave
    std::array<std::array<Real, 3>, 3> dFdr{};
};

ContactKinematics computePenaltyForceAndJacobian(const std::array<Real, 3>& r,
                                                 int dim,
                                                 Real activation_distance,
                                                 Real penalty)
{
    ContactKinematics out;
    const Real eps = Real(1e-14);

    const Real d = std::max(normN(r, dim), eps);
    if (d >= activation_distance) {
        return out;
    }

    const Real gap = activation_distance - d;
    for (int i = 0; i < dim; ++i) {
        out.force[static_cast<std::size_t>(i)] = penalty * gap * r[static_cast<std::size_t>(i)] / d;
    }

    const Real a = penalty * (activation_distance / d - Real(1));
    const Real b = penalty * activation_distance / (d * d * d);

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            const Real ij = (i == j) ? a : Real(0);
            out.dFdr[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                ij - b * r[static_cast<std::size_t>(i)] * r[static_cast<std::size_t>(j)];
        }
    }

    out.active = true;
    return out;
}

struct PairContribution {
    GlobalIndex master_cell_id{INVALID_GLOBAL_INDEX};
    std::vector<GlobalIndex> dofs{};
    std::vector<Real> rhs{};
    std::vector<Real> mat{};
    std::size_t n_slave_base{0};
    std::size_t n_master_base{0};
};

void accumulateContribution(PairContribution& c,
                            int dim,
                            std::span<const Real> N_slave,
                            std::span<const Real> N_master,
                            const ContactKinematics& kin,
                            Real weight,
                            bool want_matrix,
                            bool want_vector)
{
    const std::size_t n_s = N_slave.size();
    const std::size_t n_m = N_master.size();
    FE_THROW_IF(n_s == 0u || n_m == 0u, FEException, "PenaltySurfaceContactKernel: empty basis");
    const std::size_t n_total = static_cast<std::size_t>(dim) * (n_s + n_m);

    if (want_vector) {
        FE_THROW_IF(c.rhs.size() != n_total, FEException, "PenaltySurfaceContactKernel: rhs size mismatch");
        for (int comp = 0; comp < dim; ++comp) {
            const Real f = kin.force[static_cast<std::size_t>(comp)] * weight;
            for (std::size_t i = 0; i < n_s; ++i) {
                c.rhs[static_cast<std::size_t>(comp) * n_s + i] += N_slave[i] * f;
            }
            const std::size_t master_off = static_cast<std::size_t>(dim) * n_s + static_cast<std::size_t>(comp) * n_m;
            for (std::size_t j = 0; j < n_m; ++j) {
                c.rhs[master_off + j] -= N_master[j] * f;
            }
        }
    }

    if (want_matrix) {
        FE_THROW_IF(c.mat.size() != n_total * n_total, FEException, "PenaltySurfaceContactKernel: mat size mismatch");

        auto at = [&](std::size_t row, std::size_t col) -> Real& {
            return c.mat[row * n_total + col];
        };

        struct Side {
            int sign{1};
            std::span<const Real> N{};
            std::size_t n_base{0};
            std::size_t offset{0};
        };

        const Side sides[2] = {
            Side{+1, N_slave, n_s, 0u},
            Side{-1, N_master, n_m, static_cast<std::size_t>(dim) * n_s},
        };

        for (const auto& row_side : sides) {
            for (int beta = 0; beta < dim; ++beta) {
                for (std::size_t i = 0; i < row_side.n_base; ++i) {
                    const std::size_t row = row_side.offset + static_cast<std::size_t>(beta) * row_side.n_base + i;
                    const Real row_fac = static_cast<Real>(row_side.sign) * row_side.N[i];

                    for (const auto& col_side : sides) {
                        for (int alpha = 0; alpha < dim; ++alpha) {
                            const Real J = kin.dFdr[static_cast<std::size_t>(beta)][static_cast<std::size_t>(alpha)] * weight;
                            if (J == Real(0)) continue;
                            for (std::size_t j = 0; j < col_side.n_base; ++j) {
                                const std::size_t col =
                                    col_side.offset + static_cast<std::size_t>(alpha) * col_side.n_base + j;
                                const Real col_fac = static_cast<Real>(col_side.sign) * col_side.N[j];
                                at(row, col) += row_fac * col_fac * J;
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace

PenaltySurfaceContactKernel::PenaltySurfaceContactKernel(PenaltySurfaceContactConfig cfg)
    : cfg_(cfg)
{
    FE_THROW_IF(cfg_.field == INVALID_FIELD_ID, InvalidArgumentException,
                "PenaltySurfaceContactKernel: field == INVALID_FIELD_ID");
    FE_THROW_IF(cfg_.slave_marker == cfg_.master_marker, InvalidArgumentException,
                "PenaltySurfaceContactKernel: slave_marker must differ from master_marker");
    FE_THROW_IF(cfg_.search_radius <= Real(0), InvalidArgumentException,
                "PenaltySurfaceContactKernel: search_radius must be > 0");
    FE_THROW_IF(cfg_.activation_distance <= Real(0), InvalidArgumentException,
                "PenaltySurfaceContactKernel: activation_distance must be > 0");
    FE_THROW_IF(cfg_.penalty <= Real(0), InvalidArgumentException,
                "PenaltySurfaceContactKernel: penalty must be > 0");
}

GlobalStateSpec PenaltySurfaceContactKernel::globalStateSpec() const noexcept
{
    if (!cfg_.track_contact_count) {
        return {};
    }

    GlobalStateSpec spec;
    spec.domain = GlobalStateSpec::Domain::BoundaryFace;
    spec.bytes_per_qpt = sizeof(int);
    spec.alignment = alignof(int);
    spec.max_qpts = (cfg_.max_state_qpts > 0) ? cfg_.max_state_qpts : LocalIndex(64);
    return spec;
}

void PenaltySurfaceContactKernel::addSparsityCouplings(const FESystem& system,
                                                       sparsity::SparsityPattern& pattern) const
{
    const auto& mesh = system.meshAccess();
    const int dim = mesh.dimension();
    if (dim <= 0 || dim > 3) return;

    std::unordered_set<GlobalIndex> slave_cells;
    std::unordered_set<GlobalIndex> master_cells;

    mesh.forEachBoundaryFace(cfg_.slave_marker, [&](GlobalIndex /*face_id*/, GlobalIndex cell_id) {
        slave_cells.insert(cell_id);
    });
    mesh.forEachBoundaryFace(cfg_.master_marker, [&](GlobalIndex /*face_id*/, GlobalIndex cell_id) {
        master_cells.insert(cell_id);
    });

    if (slave_cells.empty() || master_cells.empty()) return;

    std::vector<GlobalIndex> slave_dofs;
    std::vector<GlobalIndex> master_dofs;

    const GlobalIndex offset = system.fieldDofOffset(cfg_.field);
    const auto& slave_dh = system.fieldDofHandler(cfg_.field);

    for (const auto cell_id : slave_cells) {
        const auto dofs = slave_dh.getCellDofs(cell_id);
        for (auto d : dofs) {
            slave_dofs.push_back(d + offset);
        }
    }
    for (const auto cell_id : master_cells) {
        const auto dofs = slave_dh.getCellDofs(cell_id);
        for (auto d : dofs) {
            master_dofs.push_back(d + offset);
        }
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

assembly::AssemblyResult PenaltySurfaceContactKernel::assemble(const FESystem& system,
                                                               const AssemblyRequest& request,
                                                               const SystemStateView& state,
                                                               assembly::GlobalSystemView* matrix_out,
                                                               assembly::GlobalSystemView* vector_out)
{
    assembly::AssemblyResult result;

    const auto& mesh = system.meshAccess();
    const int dim = mesh.dimension();
    FE_THROW_IF(dim <= 0 || dim > 3, FEException,
                "PenaltySurfaceContactKernel: unsupported mesh dimension");

    const auto& field = system.fieldRecord(cfg_.field);
    FE_CHECK_NOT_NULL(field.space.get(), "PenaltySurfaceContactKernel: field.space");
    FE_THROW_IF(field.components != dim, InvalidArgumentException,
                "PenaltySurfaceContactKernel: field.components must match mesh dimension");

    const auto* prod = dynamic_cast<const spaces::ProductSpace*>(field.space.get());
    FE_THROW_IF(prod == nullptr, NotImplementedException,
                "PenaltySurfaceContactKernel: field space must be spaces::ProductSpace");
    const std::size_t scalar_dofs = prod->scalar_dofs_per_component();

    const int quad_order = (cfg_.quadrature_order > 0)
                               ? cfg_.quadrature_order
                               : quadrature::QuadratureFactory::recommended_order(field.space->polynomial_order(), false);

    const auto* search = system.searchAccess();
    FE_THROW_IF(search == nullptr, FEException,
                "PenaltySurfaceContactKernel: no search access is configured (FESystem::searchAccess is null)");

    // Ensure the backend is in an assembly-ready phase for "global-only" operators.
    if (matrix_out) matrix_out->beginAssemblyPhase();
    if (vector_out && vector_out != matrix_out) vector_out->beginAssemblyPhase();

    search->build();

    // Boundary-face -> cell lookup (needed to reconstruct master cell info from face_id).
    std::unordered_map<GlobalIndex, GlobalIndex> boundary_face_cell;
    boundary_face_cell.reserve(static_cast<std::size_t>(mesh.numBoundaryFaces()));
    mesh.forEachBoundaryFace(/*marker=*/-1, [&](GlobalIndex face_id, GlobalIndex cell_id) {
        boundary_face_cell.emplace(face_id, cell_id);
    });

    std::unordered_map<GlobalIndex, CellCache> cell_cache;

    assembly::AssemblyConstraintDistributor distributor(system.constraints());

    std::vector<Real> N_slave;
    std::vector<Real> N_master;

    std::unordered_map<GlobalIndex, PairContribution> pair_by_master_face;

    mesh.forEachBoundaryFace(cfg_.slave_marker, [&](GlobalIndex slave_face_id, GlobalIndex slave_cell_id) {
        pair_by_master_face.clear();

        const auto& slave_cc = getCellCache(cell_cache, system, cfg_.field, slave_cell_id, state);
        const ElementType cell_type = slave_cc.cell_type;
        const LocalIndex local_face = mesh.getLocalFaceIndex(slave_face_id, slave_cell_id);

        auto face_rule = quadrature::SurfaceQuadrature::face_rule(cell_type,
                                                                  static_cast<int>(local_face),
                                                                  quad_order);
        FE_THROW_IF(!face_rule || face_rule->num_points() == 0u, FEException,
                    "PenaltySurfaceContactKernel: failed to create face quadrature rule");

        assembly::MaterialStateView contact_count_state{};
        if (cfg_.track_contact_count) {
            contact_count_state =
                system.globalKernelBoundaryFaceState(*this, slave_face_id, static_cast<LocalIndex>(face_rule->num_points()));
            FE_THROW_IF(!static_cast<bool>(contact_count_state), InvalidStateException,
                        "PenaltySurfaceContactKernel: contact_count requested but no global boundary-face state is available");
            FE_THROW_IF(contact_count_state.bytes_per_qpt < sizeof(int), InvalidStateException,
                        "PenaltySurfaceContactKernel: contact_count state bytes_per_qpt too small");
        }

        for (std::size_t q = 0; q < face_rule->num_points(); ++q) {
            math::Vector<Real, 3> facet_coords = face_rule->point(q);
            if (face_rule->cell_family() == svmp::CellFamily::Line) {
                facet_coords[0] = (facet_coords[0] + Real(1)) * Real(0.5);
            } else if (face_rule->cell_family() == svmp::CellFamily::Quad) {
                facet_coords[0] = (facet_coords[0] + Real(1)) * Real(0.5);
                facet_coords[1] = (facet_coords[1] + Real(1)) * Real(0.5);
            }

            const auto xi_slave = elements::ElementTransform::facet_to_reference(cell_type,
                                                                                 static_cast<int>(local_face),
                                                                                 facet_coords);

            const auto frame = elements::ElementTransform::compute_facet_frame(*slave_cc.mapping,
                                                                               xi_slave,
                                                                               static_cast<int>(local_face),
                                                                               cell_type);
            const Real w = face_rule->weight(q) * frame.jacobian_det;

            field.space->element().basis().evaluate_values(xi_slave, N_slave);
            FE_THROW_IF(N_slave.size() != scalar_dofs, FEException,
                        "PenaltySurfaceContactKernel: unexpected slave basis size");

            const auto u_slave = evaluateDisplacementAt(dim, N_slave, slave_cc.u_local);
            const auto x_slave_ref = slave_cc.mapping->map_to_physical(xi_slave);
            auto x_slave = toArray(x_slave_ref);
            for (int d = 0; d < dim; ++d) {
                x_slave[static_cast<std::size_t>(d)] += u_slave[static_cast<std::size_t>(d)];
            }

            const auto cp = search->closestBoundaryPointOnMarker(cfg_.master_marker,
                                                                 x_slave,
                                                                 cfg_.search_radius);
            if (!cp.found || cp.face_id == INVALID_GLOBAL_INDEX) {
                continue;
            }

            const auto master_cell_it = boundary_face_cell.find(cp.face_id);
            if (master_cell_it == boundary_face_cell.end()) {
                continue;
            }

            const GlobalIndex master_cell_id = master_cell_it->second;
            if (master_cell_id == slave_cell_id) {
                continue;
            }

            const auto& master_cc = getCellCache(cell_cache, system, cfg_.field, master_cell_id, state);

            math::Vector<Real, 3> xi_master{};
            try {
                xi_master = master_cc.mapping->map_to_reference(toVec(cp.closest_point));
            } catch (const FEException&) {
                continue;
            }

            field.space->element().basis().evaluate_values(xi_master, N_master);
            FE_THROW_IF(N_master.size() != scalar_dofs, FEException,
                        "PenaltySurfaceContactKernel: unexpected master basis size");

            const auto u_master = evaluateDisplacementAt(dim, N_master, master_cc.u_local);
            const auto x_master_ref = master_cc.mapping->map_to_physical(xi_master);
            auto x_master = toArray(x_master_ref);
            for (int d = 0; d < dim; ++d) {
                x_master[static_cast<std::size_t>(d)] += u_master[static_cast<std::size_t>(d)];
            }

            const auto r = sub(x_slave, x_master);
            const auto kin = computePenaltyForceAndJacobian(r, dim, cfg_.activation_distance, cfg_.penalty);
            if (!kin.active) {
                continue;
            }

            if (cfg_.track_contact_count) {
                FE_THROW_IF(contact_count_state.stride_bytes < contact_count_state.bytes_per_qpt, FEException,
                            "PenaltySurfaceContactKernel: invalid contact_count state stride");
                auto* ptr =
                    contact_count_state.data_work + q * static_cast<std::size_t>(contact_count_state.stride_bytes);
                const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
                FE_THROW_IF((addr % alignof(int)) != 0u, FEException,
                            "PenaltySurfaceContactKernel: misaligned contact_count state access");
                auto* counter = reinterpret_cast<int*>(ptr);
                *counter += 1;
            }

            auto& pair = pair_by_master_face[cp.face_id];
            if (pair.dofs.empty()) {
                pair.master_cell_id = master_cell_id;
                pair.n_slave_base = scalar_dofs;
                pair.n_master_base = scalar_dofs;
                pair.dofs.reserve(slave_cc.dofs_global.size() + master_cc.dofs_global.size());
                pair.dofs.insert(pair.dofs.end(), slave_cc.dofs_global.begin(), slave_cc.dofs_global.end());
                pair.dofs.insert(pair.dofs.end(), master_cc.dofs_global.begin(), master_cc.dofs_global.end());

                const std::size_t n_total = static_cast<std::size_t>(dim) * (pair.n_slave_base + pair.n_master_base);
                if (request.want_vector && vector_out != nullptr) {
                    pair.rhs.assign(n_total, Real(0));
                }
                if (request.want_matrix && matrix_out != nullptr) {
                    pair.mat.assign(n_total * n_total, Real(0));
                }
            }

            accumulateContribution(pair,
                                   dim,
                                   N_slave,
                                   N_master,
                                   kin,
                                   w,
                                   request.want_matrix && matrix_out != nullptr,
                                   request.want_vector && vector_out != nullptr);
        }

        for (auto& kv : pair_by_master_face) {
            auto& pair = kv.second;
            if (pair.dofs.empty()) continue;

            if (request.want_matrix && matrix_out != nullptr && request.want_vector && vector_out != nullptr) {
                distributor.distributeLocalToGlobal(pair.mat, pair.rhs, pair.dofs, *matrix_out, *vector_out);
                result.matrix_entries_inserted += static_cast<GlobalIndex>(pair.dofs.size() * pair.dofs.size());
                result.vector_entries_inserted += static_cast<GlobalIndex>(pair.dofs.size());
            } else if (request.want_matrix && matrix_out != nullptr) {
                distributor.distributeMatrixToGlobal(pair.mat, pair.dofs, *matrix_out);
                result.matrix_entries_inserted += static_cast<GlobalIndex>(pair.dofs.size() * pair.dofs.size());
            } else if (request.want_vector && vector_out != nullptr) {
                distributor.distributeVectorToGlobal(pair.rhs, pair.dofs, *vector_out);
                result.vector_entries_inserted += static_cast<GlobalIndex>(pair.dofs.size());
            }
        }
    });

    return result;
}

} // namespace systems
} // namespace FE
} // namespace svmp
