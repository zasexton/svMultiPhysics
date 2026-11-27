/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/FEAdaptivityTransfer.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#include "Assembly/MeshAccess.h"
#include "Basis/BasisFunction.h"
#include "Core/FEException.h"
#include "Geometry/GeometryMapping.h"
#include "Quadrature/QuadratureFactory.h"
#include "Systems/SystemCollectives.h"
#include "Systems/SystemsExceptions.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace svmp {
namespace FE {
namespace systems {
namespace {

using WeightList = std::vector<std::pair<svmp::gid_t, double>>;

std::unordered_map<svmp::gid_t, WeightList>
build_vertex_provenance_lookup(const svmp::RefinementDelta& delta)
{
    std::unordered_map<svmp::gid_t, WeightList> out;
    out.reserve(delta.new_vertices.size());
    for (const auto& record : delta.new_vertices) {
        WeightList weights = !record.reference_coordinate_weights.empty()
            ? record.reference_coordinate_weights
            : record.parent_vertex_weights;
        out.emplace(record.new_vertex_gid, std::move(weights));
    }
    return out;
}

// Refinement can also repartition a mesh. Import the old samples needed
// locally, including parents outside the new partition's overlap.
std::unordered_map<svmp::gid_t, std::vector<Real>> remoteVertexValues(
    const svmp::MeshBase& old_mesh, const svmp::MeshBase& new_mesh,
    const std::unordered_map<svmp::gid_t, WeightList>& provenance,
    std::span<const Real> values, std::size_t stride, const CollectiveCommunicator& comm)
{
    std::unordered_map<svmp::gid_t, std::vector<Real>> result;
    if (!comm.isParallel()) return result;
    std::unordered_set<std::uint64_t> needed;
    for (const auto gid : new_mesh.vertex_gids()) {
        if (old_mesh.global_to_local_vertex(gid) != svmp::INVALID_INDEX) continue;
        needed.insert(gid);
        if (const auto p = provenance.find(gid); p != provenance.end()) {
            for (const auto& [parent, weight] : p->second) {
                if (old_mesh.global_to_local_vertex(parent) == svmp::INVALID_INDEX) needed.insert(parent);
            }
        }
    }
    const std::vector<std::uint64_t> local(needed.begin(), needed.end());
    auto requested = collectiveGatherWords(local, comm, "nodal transfer source requests");
    if (requested.empty()) return result;
    std::sort(requested.begin(), requested.end());
    requested.erase(std::unique(requested.begin(), requested.end()), requested.end());
    std::vector<std::uint64_t> reply;
    for (const auto gid : requested) {
        const auto v = old_mesh.global_to_local_vertex(gid);
        if (v == svmp::INVALID_INDEX) continue;
        reply.push_back(gid);
        for (std::size_t c = 0; c < stride; ++c)
            reply.push_back(std::bit_cast<std::uint64_t>(values[v * stride + c]));
    }
    const auto all = collectiveGatherWords(reply, comm, "nodal transfer source values");
    for (std::size_t i = 0; i < all.size(); i += stride + 1) {
        if (!needed.contains(all[i])) continue;
        auto [entry, inserted] = result.try_emplace(all[i]);
        if (inserted) {
            for (std::size_t c = 0; c < stride; ++c)
                entry->second.push_back(std::bit_cast<Real>(all[i + 1 + c]));
        }
    }
    return result;
}

// A MeshBase is a local mesh, possibly with ghost cells. Choose one copy
// of each stable cell ID before integrating an overlapping partition.
std::vector<Real> nodalIntegralWeights(const svmp::MeshBase& mesh,
                                       svmp::Configuration configuration,
                                       const CollectiveCommunicator& comm)
{
    std::unordered_map<svmp::gid_t, int> first_rank;
    if (comm.isParallel()) {
        if (collectiveAnyTrue(mesh.cell_gids().size() != mesh.n_cells(), comm))
            FE_THROW(InvalidStateException, "Conservative nodal transfer requires global cell IDs");
        std::vector<std::uint64_t> local;
        for (auto id : mesh.cell_gids()) {
            local.push_back(static_cast<std::uint64_t>(id));
            local.push_back(static_cast<std::uint64_t>(comm.rank()));
        }
        const auto all = collectiveGatherWords(local, comm, "nodal transfer cell ownership");
        for (std::size_t i = 0; i < all.size(); i += 2)
            first_rank.try_emplace(static_cast<svmp::gid_t>(all[i]), static_cast<int>(all[i + 1]));
    }
    std::vector<Real> weights(mesh.n_vertices(), 0);
    std::vector<Real> shape;
    std::exception_ptr failure;
    try {
        for (std::size_t c = 0; c < mesh.n_cells(); ++c) {
            if (comm.isParallel() && first_rank.at(mesh.cell_gids()[c]) != comm.rank()) continue;
            const auto [vertices, count] = mesh.cell_vertices_span(c);
            if (mesh.cell_shape(c).family == svmp::CellFamily::Point) {
                weights.at(vertices[0]) += 1;
                continue;
            }
            const auto mapping = assembly::meshCellGeometryMapping(mesh, c, configuration);
            // The determinant for a tensor geometry of order p has degree
            // at most d*p-1 in each coordinate; the nodal shape adds p.
            const int order = (mapping->dimension() + 1) * mesh.geometry_order(c) - 1;
            const auto quad = quadrature::QuadratureFactory::create_pinned(mapping->element_type(), order);
            for (std::size_t q = 0; q < quad->num_points(); ++q) {
                const auto x = quad->point(q);
                const math::Vector<Real, 3> xi{x[0], x[1], x[2]};
                mapping->geometryBasis().evaluate_values(xi, shape);
                const Real measure = std::abs(mapping->jacobian_determinant(xi));
                FE_THROW_IF(count != shape.size() || !std::isfinite(measure) || measure == 0,
                            InvalidStateException, "Invalid nodal transfer geometry at cell " + std::to_string(c));
                const Real weight = quad->weight(q) * measure;
                for (std::size_t a = 0; a < count; ++a) weights.at(vertices[a]) += weight * shape[a];
            }
        }
    } catch (...) {
        failure = std::current_exception();
    }
    requireCollectivePhasePreparation(failure, comm, "nodal transfer physical weights");
    return weights;
}

std::vector<double> componentIntegrals(std::span<const Real> weights, std::span<const Real> values,
                                       std::size_t stride, const CollectiveCommunicator& comm)
{
    std::vector<ExactAccumulator> sums(2 * stride + 1);
    for (std::size_t v = 0; v < weights.size(); ++v) {
        sums.back().add(weights[v]);
        for (std::size_t c = 0; c < stride; ++c) {
            const Real term = weights[v] * values[v * stride + c];
            sums[c].add(term);
            sums[stride + c].add(std::abs(term));
        }
    }
    return collectiveExactSums(sums, comm);
}

} // namespace

FEFieldTransferDiagnostics transferNodalFieldByVertexProvenance(
    const svmp::MeshBase& old_mesh,
    const svmp::MeshBase& new_mesh,
    const svmp::RefinementDelta& delta,
    int components,
    std::span<const Real> old_vertex_values,
    std::span<Real> new_vertex_values,
    const FEFieldTransferOptions& options)
{
    return transferNodalFieldByVertexProvenance(old_mesh, new_mesh, delta, components,
        old_vertex_values, new_vertex_values, options, CollectiveCommunicator{});
}

FEFieldTransferDiagnostics transferNodalFieldByVertexProvenance(
    const svmp::MeshBase& old_mesh, const svmp::MeshBase& new_mesh,
    const svmp::RefinementDelta& delta, int components,
    std::span<const Real> old_vertex_values, std::span<Real> new_vertex_values,
    const FEFieldTransferOptions& options, const CollectiveCommunicator& comm,
    std::optional<svmp::Configuration> configuration)
{
    const auto n_old = old_mesh.n_vertices();
    const auto n_new = new_mesh.n_vertices();
    const auto stride = static_cast<std::size_t>(components);
    const bool conservative = options.method == FEFieldTransferMethod::Conservative;
    if (comm.isParallel()) {
        const std::array<std::uint64_t, 5> declaration{static_cast<std::uint64_t>(components),
            static_cast<std::uint64_t>(options.method), options.require_all_vertices,
            std::bit_cast<std::uint64_t>(options.conservation_tolerance),
            configuration ? 1 + static_cast<std::uint64_t>(*configuration) : 0};
        const auto all = collectiveGatherWords(declaration, comm, "nodal transfer declaration");
        for (std::size_t i = 0; i < all.size(); i += declaration.size())
            FE_THROW_IF(!std::equal(declaration.begin(), declaration.end(), all.begin() + i),
                        InvalidArgumentException, "Inconsistent distributed nodal transfer options");
    }
    bool invalid = components <= 0 || (components > 0 && (old_vertex_values.size() / stride < n_old ||
                                                           new_vertex_values.size() / stride < n_new));
    if (conservative) {
        invalid |= !(std::isfinite(options.conservation_tolerance) && options.conservation_tolerance >= 0);
    }
    invalid = collectiveAnyTrue(invalid, comm);
    FE_THROW_IF(invalid, InvalidArgumentException, "Invalid nodal transfer dimensions or conservation tolerance");

    FEFieldTransferDiagnostics diagnostics;
    const auto provenance = build_vertex_provenance_lookup(delta);
    const auto remote = remoteVertexValues(old_mesh, new_mesh, provenance, old_vertex_values, stride, comm);
    const auto source = [&](svmp::gid_t gid) -> std::span<const Real> {
        const auto vertex = old_mesh.global_to_local_vertex(gid);
        if (vertex != svmp::INVALID_INDEX) return old_vertex_values.subspan(vertex * stride, stride);
        if (const auto it = remote.find(gid); it != remote.end()) return it->second;
        return {};
    };
    std::vector<Real> candidate(n_new * stride, 0);

    for (std::size_t v = 0; v < n_new; ++v) {
        const auto gid = new_mesh.vertex_gids().at(v);
        const auto original = source(gid);
        if (!original.empty()) {
            const auto new_base = v * stride;
            for (std::size_t c = 0; c < stride; ++c) {
                candidate[new_base + c] = original[c];
                ++diagnostics.values_transferred;
            }
            continue;
        }

        const auto pit = provenance.find(gid);
        if (pit == provenance.end() || pit->second.empty()) {
            diagnostics.success = false;
            diagnostics.diagnostics.push_back(
                "No vertex provenance available for new vertex GID " + std::to_string(gid));
            if (options.require_all_vertices) {
                continue;
            }
            diagnostics.values_transferred += stride;
            continue;
        }

        const auto new_base = v * stride;
        for (const auto& [parent_gid, weight] : pit->second) {
            const auto parent = source(parent_gid);
            if (parent.empty()) {
                diagnostics.success = false;
                diagnostics.diagnostics.push_back(
                    "Parent vertex GID " + std::to_string(parent_gid) +
                    " is not present in the old mesh");
                continue;
            }
            for (std::size_t c = 0; c < stride; ++c) {
                candidate[new_base + c] += static_cast<Real>(weight) * parent[c];
            }
        }
        diagnostics.values_transferred += stride;
    }

    diagnostics.success = !collectiveAnyTrue(!diagnostics.success, comm);
    FE_THROW_IF(options.require_all_vertices && !diagnostics.success, InvalidStateException,
                "transferNodalFieldByVertexProvenance: required vertex provenance is incomplete");
    if (conservative) {
        const auto before = nodalIntegralWeights(old_mesh, configuration.value_or(old_mesh.active_configuration()), comm);
        const auto after = nodalIntegralWeights(new_mesh, configuration.value_or(new_mesh.active_configuration()), comm);
        const auto old = componentIntegrals(before, old_vertex_values, stride, comm);
        const auto interpolated = componentIntegrals(after, candidate, stride, comm);
        const Real roundoff = 32 * std::numeric_limits<Real>::epsilon();
        for (std::size_t c = 0; c < stride; ++c) {
            const Real defect = old[c] - interpolated[c];
            const Real magnitude = old[stride + c] + interpolated[stride + c];
            FE_THROW_IF(!std::isfinite(defect) || !std::isfinite(magnitude), InvalidStateException,
                        "Conservative nodal transfer has a non-finite component integral");
            if (std::abs(defect) <= std::min(options.conservation_tolerance, roundoff * magnitude)) continue;
            if (std::abs(interpolated[c]) > roundoff * interpolated[stride + c] &&
                std::abs(old[c]) > roundoff * old[stride + c] &&
                std::signbit(old[c]) == std::signbit(interpolated[c])) {
                const Real factor = old[c] / interpolated[c];
                for (std::size_t v = 0; v < n_new; ++v) candidate[v * stride + c] *= factor;
            } else {
                // Near a zero mean, scaling can erase a nonconstant field.
                // A constant shift changes only its mean.
                FE_THROW_IF(!(interpolated.back() > 0), InvalidStateException,
                            "Conservative nodal transfer has no positive target measure");
                const Real shift = defect / interpolated.back();
                for (std::size_t v = 0; v < n_new; ++v) candidate[v * stride + c] += shift;
            }
        }
        const auto corrected = componentIntegrals(after, candidate, stride, comm);
        for (std::size_t c = 0; c < stride; ++c) {
            FE_THROW_IF(!std::isfinite(corrected[c]), InvalidStateException,
                        "Conservative nodal transfer produced a non-finite component integral");
            diagnostics.conservation_error = std::max(diagnostics.conservation_error, std::abs(corrected[c] - old[c]));
        }
        if (diagnostics.conservation_error > options.conservation_tolerance) {
            diagnostics.success = false;
            diagnostics.diagnostics.push_back("Conservative nodal transfer exceeded tolerance");
            FE_THROW_IF(options.require_all_vertices, InvalidStateException, diagnostics.diagnostics.back());
        }
    }

    std::fill(new_vertex_values.begin(), new_vertex_values.end(), Real(0));
    std::copy(candidate.begin(), candidate.end(), new_vertex_values.begin());
    return diagnostics;
}

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
