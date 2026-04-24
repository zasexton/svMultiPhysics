/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/FEAdaptivityTransfer.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#include "Core/FEException.h"
#include "Systems/SystemsExceptions.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

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
    FE_THROW_IF(components <= 0, InvalidArgumentException,
                "transferNodalFieldByVertexProvenance: components must be positive");

    const auto n_old = old_mesh.n_vertices();
    const auto n_new = new_mesh.n_vertices();
    const auto stride = static_cast<std::size_t>(components);

    FE_THROW_IF(old_vertex_values.size() < n_old * stride, InvalidArgumentException,
                "transferNodalFieldByVertexProvenance: old value array is too small");
    FE_THROW_IF(new_vertex_values.size() < n_new * stride, InvalidArgumentException,
                "transferNodalFieldByVertexProvenance: new value array is too small");

    FEFieldTransferDiagnostics diagnostics;
    const auto provenance = build_vertex_provenance_lookup(delta);

    std::fill(new_vertex_values.begin(), new_vertex_values.end(), Real(0));

    for (std::size_t v = 0; v < n_new; ++v) {
        const auto gid = new_mesh.vertex_gids().at(v);
        const auto old_v = old_mesh.global_to_local_vertex(gid);
        if (old_v != svmp::INVALID_INDEX) {
            const auto old_base = static_cast<std::size_t>(old_v) * stride;
            const auto new_base = v * stride;
            for (std::size_t c = 0; c < stride; ++c) {
                new_vertex_values[new_base + c] = old_vertex_values[old_base + c];
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
            const auto old_parent = old_mesh.global_to_local_vertex(parent_gid);
            if (old_parent == svmp::INVALID_INDEX) {
                diagnostics.success = false;
                diagnostics.diagnostics.push_back(
                    "Parent vertex GID " + std::to_string(parent_gid) +
                    " is not present in the old mesh");
                continue;
            }
            const auto old_base = static_cast<std::size_t>(old_parent) * stride;
            for (std::size_t c = 0; c < stride; ++c) {
                new_vertex_values[new_base + c] += static_cast<Real>(weight) * old_vertex_values[old_base + c];
            }
        }
        diagnostics.values_transferred += stride;
    }

    if (options.method == FEFieldTransferMethod::Conservative && n_new > 0u) {
        Real max_error = Real(0);
        for (std::size_t c = 0; c < stride; ++c) {
            long double old_sum = 0.0L;
            long double new_sum = 0.0L;
            for (std::size_t v = 0; v < n_old; ++v) {
                old_sum += static_cast<long double>(old_vertex_values[v * stride + c]);
            }
            for (std::size_t v = 0; v < n_new; ++v) {
                new_sum += static_cast<long double>(new_vertex_values[v * stride + c]);
            }

            if (std::abs(static_cast<double>(new_sum)) > options.conservation_tolerance) {
                const long double scale = old_sum / new_sum;
                for (std::size_t v = 0; v < n_new; ++v) {
                    new_vertex_values[v * stride + c] =
                        static_cast<Real>(static_cast<long double>(new_vertex_values[v * stride + c]) * scale);
                }
            } else {
                const long double fill = old_sum / static_cast<long double>(n_new);
                for (std::size_t v = 0; v < n_new; ++v) {
                    new_vertex_values[v * stride + c] = static_cast<Real>(fill);
                }
            }

            long double corrected_sum = 0.0L;
            for (std::size_t v = 0; v < n_new; ++v) {
                corrected_sum += static_cast<long double>(new_vertex_values[v * stride + c]);
            }
            max_error = std::max(max_error,
                                 static_cast<Real>(std::abs(static_cast<double>(corrected_sum - old_sum))));
        }
        diagnostics.conservation_error = max_error;
        if (max_error > options.conservation_tolerance) {
            diagnostics.success = false;
            diagnostics.diagnostics.push_back("Conservative nodal transfer exceeded tolerance");
        }
    }

    if (options.require_all_vertices && !diagnostics.success) {
        FE_THROW(InvalidStateException,
                 "transferNodalFieldByVertexProvenance: required vertex provenance is incomplete");
    }

    return diagnostics;
}

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
