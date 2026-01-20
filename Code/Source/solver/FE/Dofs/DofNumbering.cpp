/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "DofNumbering.h"
#include <algorithm>
#include <queue>
#include <stack>
#include <numeric>
#include <cmath>
#include <limits>
#include <cctype>

namespace svmp {
namespace FE {
namespace dofs {

// =============================================================================
// SequentialNumbering
// =============================================================================

std::vector<GlobalIndex> SequentialNumbering::computeNumbering(
    GlobalIndex n_dofs,
    std::span<const GlobalIndex> /*adjacency*/,
    std::span<const GlobalIndex> /*adj_indices*/) const {

    std::vector<GlobalIndex> perm(static_cast<std::size_t>(n_dofs));
    std::iota(perm.begin(), perm.end(), GlobalIndex{0});
    return perm;
}

// =============================================================================
// InterleavedNumbering
// =============================================================================

InterleavedNumbering::InterleavedNumbering(LocalIndex n_components)
    : n_components_(n_components) {
    if (n_components_ < 1) {
        n_components_ = 1;
    }
}

std::vector<GlobalIndex> InterleavedNumbering::computeNumbering(
    GlobalIndex n_dofs,
    std::span<const GlobalIndex> /*adjacency*/,
    std::span<const GlobalIndex> /*adj_indices*/) const {

    if (n_dofs == 0) return {};

    if (n_components_ <= 1) {
        std::vector<GlobalIndex> perm(static_cast<std::size_t>(n_dofs));
        std::iota(perm.begin(), perm.end(), GlobalIndex{0});
        return perm;
    }

    if (n_dofs % n_components_ != 0) {
        // Cannot infer a component-wise structure.
        std::vector<GlobalIndex> perm(static_cast<std::size_t>(n_dofs));
        std::iota(perm.begin(), perm.end(), GlobalIndex{0});
        return perm;
    }

    // Assume current ordering is block-by-component:
    //   [c0_0, c0_1, ..., c0_{N-1},  c1_0, ..., c{nc-1}_{N-1}]
    // Convert to interleaved-by-node:
    //   [c0_0, c1_0, ..., c{nc-1}_0, c0_1, c1_1, ...]
    const GlobalIndex n_per_component = n_dofs / n_components_;

    std::vector<GlobalIndex> perm(static_cast<std::size_t>(n_dofs), -1);
    for (LocalIndex c = 0; c < n_components_; ++c) {
        const GlobalIndex base_old = static_cast<GlobalIndex>(c) * n_per_component;
        for (GlobalIndex i = 0; i < n_per_component; ++i) {
            const GlobalIndex old_idx = base_old + i;
            const GlobalIndex new_idx = i * static_cast<GlobalIndex>(n_components_) + static_cast<GlobalIndex>(c);
            perm[static_cast<std::size_t>(old_idx)] = new_idx;
        }
    }

    return perm;
}

// =============================================================================
// BlockNumbering
// =============================================================================

BlockNumbering::BlockNumbering(std::vector<GlobalIndex> block_sizes)
    : block_sizes_(std::move(block_sizes)) {}

std::vector<GlobalIndex> BlockNumbering::computeNumbering(
    GlobalIndex n_dofs,
    std::span<const GlobalIndex> /*adjacency*/,
    std::span<const GlobalIndex> /*adj_indices*/) const {

    if (block_sizes_.empty()) {
        // No block info - return identity
        std::vector<GlobalIndex> perm(static_cast<std::size_t>(n_dofs));
        std::iota(perm.begin(), perm.end(), GlobalIndex{0});
        return perm;
    }

    // Compute block offsets
    std::vector<GlobalIndex> block_offsets;
    block_offsets.reserve(block_sizes_.size() + 1);
    block_offsets.push_back(0);
    for (auto size : block_sizes_) {
        block_offsets.push_back(block_offsets.back() + size);
    }

    GlobalIndex total_block_dofs = block_offsets.back();
    if (total_block_dofs != n_dofs) {
        // Block sizes don't match - return identity
        std::vector<GlobalIndex> perm(static_cast<std::size_t>(n_dofs));
        std::iota(perm.begin(), perm.end(), GlobalIndex{0});
        return perm;
    }

    // If blocks are uniform, we can interpret the current ordering as
    // interleaved-by-node across blocks and reorder to contiguous blocks.
    const auto n_blocks = static_cast<LocalIndex>(block_sizes_.size());
    const bool uniform_blocks = std::all_of(
        block_sizes_.begin(), block_sizes_.end(),
        [&](GlobalIndex s) { return s == block_sizes_.front(); });

    if (n_blocks <= 1 || !uniform_blocks) {
        std::vector<GlobalIndex> perm(static_cast<std::size_t>(n_dofs));
        std::iota(perm.begin(), perm.end(), GlobalIndex{0});
        return perm;
    }

    const GlobalIndex n_per_block = block_sizes_.front();

    // Current interleaved ordering:
    //   old = node * n_blocks + block
    // Target block ordering:
    //   new = block * n_per_block + node
    std::vector<GlobalIndex> perm(static_cast<std::size_t>(n_dofs), -1);
    for (GlobalIndex old = 0; old < n_dofs; ++old) {
        const GlobalIndex node = old / static_cast<GlobalIndex>(n_blocks);
        const GlobalIndex block = old % static_cast<GlobalIndex>(n_blocks);
        if (node >= n_per_block) {
            perm[static_cast<std::size_t>(old)] = old;
            continue;
        }
        perm[static_cast<std::size_t>(old)] = block * n_per_block + node;
    }

    return perm;
}

// =============================================================================
// HierarchicalNumbering
// =============================================================================

HierarchicalNumbering::HierarchicalNumbering(
    GlobalIndex n_vertex_dofs, GlobalIndex n_edge_dofs,
    GlobalIndex n_face_dofs, GlobalIndex n_cell_dofs)
    : n_vertex_dofs_(n_vertex_dofs)
    , n_edge_dofs_(n_edge_dofs)
    , n_face_dofs_(n_face_dofs)
    , n_cell_dofs_(n_cell_dofs) {}

std::vector<GlobalIndex> HierarchicalNumbering::computeNumbering(
    GlobalIndex n_dofs,
    std::span<const GlobalIndex> /*adjacency*/,
    std::span<const GlobalIndex> /*adj_indices*/) const {

    // DOFs are assumed to be initially arranged as:
    // [vertex DOFs] [edge DOFs] [face DOFs] [cell DOFs]
    // We return identity since this is already hierarchical order

    GlobalIndex expected = n_vertex_dofs_ + n_edge_dofs_ + n_face_dofs_ + n_cell_dofs_;
    if (expected != n_dofs) {
        // Mismatch - return identity
        std::vector<GlobalIndex> perm(static_cast<std::size_t>(n_dofs));
        std::iota(perm.begin(), perm.end(), GlobalIndex{0});
        return perm;
    }

    std::vector<GlobalIndex> perm(static_cast<std::size_t>(n_dofs));
    std::iota(perm.begin(), perm.end(), GlobalIndex{0});
    return perm;
}

// =============================================================================
// CuthillMcKeeNumbering
// =============================================================================

CuthillMcKeeNumbering::CuthillMcKeeNumbering(bool reverse,
                                              std::optional<GlobalIndex> start_vertex)
    : reverse_(reverse)
    , start_vertex_(start_vertex) {}

std::vector<GlobalIndex> CuthillMcKeeNumbering::computeNumbering(
    GlobalIndex n_dofs,
    std::span<const GlobalIndex> adjacency,
    std::span<const GlobalIndex> adj_indices) const {

    if (n_dofs == 0) return {};

    // Check adjacency is valid
    if (adjacency.size() < static_cast<std::size_t>(n_dofs + 1)) {
        // Invalid adjacency - return identity
        std::vector<GlobalIndex> perm(static_cast<std::size_t>(n_dofs));
        std::iota(perm.begin(), perm.end(), GlobalIndex{0});
        return perm;
    }

    std::vector<GlobalIndex> perm;
    perm.reserve(static_cast<std::size_t>(n_dofs));

    std::vector<bool> visited(static_cast<std::size_t>(n_dofs), false);

    // Find starting vertex
    GlobalIndex start = start_vertex_.value_or(findPeripheralVertex(n_dofs, adjacency, adj_indices));

    // BFS-based Cuthill-McKee
    std::queue<GlobalIndex> queue;

    // Process all connected components
    for (GlobalIndex component_start = 0; component_start < n_dofs; ++component_start) {
        GlobalIndex actual_start = (component_start == 0) ? start : component_start;

        if (visited[static_cast<std::size_t>(actual_start)]) continue;

        queue.push(actual_start);
        visited[static_cast<std::size_t>(actual_start)] = true;

        while (!queue.empty()) {
            GlobalIndex current = queue.front();
            queue.pop();
            perm.push_back(current);

            // Get neighbors and sort by degree
            auto adj_start = static_cast<std::size_t>(adjacency[static_cast<std::size_t>(current)]);
            auto adj_end = static_cast<std::size_t>(adjacency[static_cast<std::size_t>(current) + 1]);

            std::vector<std::pair<GlobalIndex, GlobalIndex>> neighbors; // (degree, index)
            for (std::size_t i = adj_start; i < adj_end && i < adj_indices.size(); ++i) {
                GlobalIndex neighbor = adj_indices[i];
                if (neighbor >= 0 && neighbor < n_dofs && !visited[static_cast<std::size_t>(neighbor)]) {
                    auto n_adj_start = static_cast<std::size_t>(adjacency[static_cast<std::size_t>(neighbor)]);
                    auto n_adj_end = static_cast<std::size_t>(adjacency[static_cast<std::size_t>(neighbor) + 1]);
                    GlobalIndex degree = static_cast<GlobalIndex>(n_adj_end - n_adj_start);
                    neighbors.emplace_back(degree, neighbor);
                }
            }

            // Sort neighbors by degree (ascending)
            std::sort(neighbors.begin(), neighbors.end());

            // Add to queue in degree order
            for (const auto& [degree, neighbor] : neighbors) {
                if (!visited[static_cast<std::size_t>(neighbor)]) {
                    visited[static_cast<std::size_t>(neighbor)] = true;
                    queue.push(neighbor);
                }
            }
        }
    }

    // Handle any disconnected vertices
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        if (!visited[static_cast<std::size_t>(i)]) {
            perm.push_back(i);
        }
    }

    // Reverse if requested (RCM)
    if (reverse_) {
        std::reverse(perm.begin(), perm.end());
    }

    // Convert to permutation: perm[old] = new
    std::vector<GlobalIndex> result(static_cast<std::size_t>(n_dofs));
    for (std::size_t new_idx = 0; new_idx < perm.size(); ++new_idx) {
        result[static_cast<std::size_t>(perm[new_idx])] = static_cast<GlobalIndex>(new_idx);
    }

    return result;
}

GlobalIndex CuthillMcKeeNumbering::findPeripheralVertex(
    GlobalIndex n_dofs,
    std::span<const GlobalIndex> adjacency,
    std::span<const GlobalIndex> adj_indices) const {

    static_cast<void>(adj_indices);
    if (n_dofs == 0) return 0;

    // Find minimum degree vertex as starting point
    GlobalIndex min_degree_vertex = 0;
    GlobalIndex min_degree = std::numeric_limits<GlobalIndex>::max();

    for (GlobalIndex v = 0; v < n_dofs; ++v) {
        auto start = static_cast<std::size_t>(adjacency[static_cast<std::size_t>(v)]);
        auto end = static_cast<std::size_t>(adjacency[static_cast<std::size_t>(v) + 1]);
        GlobalIndex degree = static_cast<GlobalIndex>(end - start);
        if (degree < min_degree) {
            min_degree = degree;
            min_degree_vertex = v;
        }
    }

    // Could do multiple BFS levels to find pseudo-peripheral node,
    // but minimum degree is a reasonable approximation
    return min_degree_vertex;
}

// =============================================================================
// NestedDissectionNumbering
// =============================================================================

NestedDissectionNumbering::NestedDissectionNumbering(GlobalIndex min_partition_size)
    : min_partition_size_(min_partition_size) {}

std::vector<GlobalIndex> NestedDissectionNumbering::computeNumbering(
    GlobalIndex n_dofs,
    std::span<const GlobalIndex> adjacency,
    std::span<const GlobalIndex> adj_indices) const {

    if (n_dofs == 0) return {};

    // For simplicity, use a basic recursive bisection
    // A full implementation would use graph partitioning (e.g., METIS)

    std::vector<GlobalIndex> partition(static_cast<std::size_t>(n_dofs));
    std::iota(partition.begin(), partition.end(), GlobalIndex{0});

    std::vector<GlobalIndex> numbering(static_cast<std::size_t>(n_dofs), -1);
    GlobalIndex next_number = 0;

    dissect(partition, adjacency, adj_indices, next_number, numbering);

    return numbering;
}

void NestedDissectionNumbering::dissect(
    std::span<GlobalIndex> partition,
    std::span<const GlobalIndex> adjacency,
    std::span<const GlobalIndex> adj_indices,
    GlobalIndex& next_number,
    std::vector<GlobalIndex>& numbering) const {

    auto n = static_cast<GlobalIndex>(partition.size());

    if (n <= min_partition_size_) {
        // Base case: number sequentially
        for (auto v : partition) {
            numbering[static_cast<std::size_t>(v)] = next_number++;
        }
        return;
    }

    // Simple bisection: split partition in half
    auto mid = partition.size() / 2;

    std::span<GlobalIndex> left(partition.data(), mid);
    std::span<GlobalIndex> right(partition.data() + mid, partition.size() - mid);

    // Recursively number left and right partitions
    dissect(left, adjacency, adj_indices, next_number, numbering);
    dissect(right, adjacency, adj_indices, next_number, numbering);

    // Note: A proper implementation would:
    // 1. Find a separator between left and right
    // 2. Number the separator last
    // This simplified version just does sequential bisection
}

// =============================================================================
// SpaceFillingCurveNumbering
// =============================================================================

SpaceFillingCurveNumbering::SpaceFillingCurveNumbering(CurveType type, int dim)
    : type_(type)
    , dim_(dim) {}

void SpaceFillingCurveNumbering::setCoordinates(std::span<const double> coords, int dim) {
    coords_.assign(coords.begin(), coords.end());
    dim_ = dim;
}

std::vector<GlobalIndex> SpaceFillingCurveNumbering::computeNumbering(
    GlobalIndex n_dofs,
    std::span<const GlobalIndex> /*adjacency*/,
    std::span<const GlobalIndex> /*adj_indices*/) const {

    if (n_dofs == 0) return {};
    FE_CHECK_ARG(dim_ > 0, "SpaceFillingCurveNumbering: dim must be positive");

    // If no coordinates, return identity
    if (coords_.empty()) {
        std::vector<GlobalIndex> perm(static_cast<std::size_t>(n_dofs));
        std::iota(perm.begin(), perm.end(), GlobalIndex{0});
        return perm;
    }

    // Compute curve index for each DOF
    const std::size_t dim_u = static_cast<std::size_t>(dim_);
    std::size_t n_vertices = coords_.size() / dim_u;
    std::vector<std::pair<uint64_t, GlobalIndex>> curve_indices;
    curve_indices.reserve(n_vertices);

    for (std::size_t v = 0; v < n_vertices && v < static_cast<std::size_t>(n_dofs); ++v) {
        const std::size_t base = v * dim_u;
        double x = coords_[base];
        double y = (dim_ >= 2) ? coords_[base + 1] : 0.0;
        double z = (dim_ >= 3) ? coords_[base + 2] : 0.0;

        uint64_t index = (type_ == CurveType::Hilbert)
            ? hilbertIndex(x, y, z)
            : mortonCode(x, y, z);

        curve_indices.emplace_back(index, static_cast<GlobalIndex>(v));
    }

    // Sort by curve index
    std::sort(curve_indices.begin(), curve_indices.end());

    // Create permutation
    std::vector<GlobalIndex> perm(static_cast<std::size_t>(n_dofs));
    for (std::size_t i = 0; i < curve_indices.size(); ++i) {
        perm[static_cast<std::size_t>(curve_indices[i].second)] = static_cast<GlobalIndex>(i);
    }

    // Handle DOFs beyond vertex count
    GlobalIndex next = static_cast<GlobalIndex>(curve_indices.size());
    for (GlobalIndex d = static_cast<GlobalIndex>(curve_indices.size()); d < n_dofs; ++d) {
        perm[static_cast<std::size_t>(d)] = next++;
    }

    return perm;
}

std::string SpaceFillingCurveNumbering::name() const {
    switch (type_) {
        case CurveType::Morton: return "Morton (Z-order)";
        case CurveType::Hilbert: return "Hilbert";
        default: return "Space-filling curve";
    }
}

uint64_t SpaceFillingCurveNumbering::mortonCode(double x, double y, double z) const {
    // Normalize coordinates to [0, 2^21) for 64-bit Morton code
    constexpr uint64_t MAX_COORD = (1ULL << 21) - 1;

    auto normalize = [](double v) -> uint64_t {
        // Assume coordinates are in [0, 1] or normalize accordingly
        v = std::max(0.0, std::min(1.0, v));
        return static_cast<uint64_t>(v * MAX_COORD);
    };

    uint64_t xi = normalize(x);
    uint64_t yi = normalize(y);
    uint64_t zi = normalize(z);

    // Interleave bits (Morton code)
    auto spread = [](uint64_t v) -> uint64_t {
        v = (v | (v << 32)) & 0x1f00000000ffff;
        v = (v | (v << 16)) & 0x1f0000ff0000ff;
        v = (v | (v << 8))  & 0x100f00f00f00f00f;
        v = (v | (v << 4))  & 0x10c30c30c30c30c3;
        v = (v | (v << 2))  & 0x1249249249249249;
        return v;
    };

    return spread(xi) | (spread(yi) << 1) | (spread(zi) << 2);
}

uint64_t SpaceFillingCurveNumbering::hilbertIndex(double x, double y, double z) const {
    // Simplified Hilbert curve - for a full implementation, use the
    // rotation-based algorithm from "Programming the Hilbert Curve"

    // For now, just use Morton code with bit manipulation
    // A proper Hilbert implementation is more complex
    return mortonCode(x, y, z);  // Placeholder
}

// =============================================================================
// Numbering Utilities
// =============================================================================

void applyNumbering(DofMap& dof_map, std::span<const GlobalIndex> permutation) {
    if (dof_map.isFinalized()) {
        throw FEException("applyNumbering: DofMap must be in Building state");
    }

    const auto n_dofs = dof_map.getNumDofs();
    if (n_dofs > 0 && permutation.size() < static_cast<std::size_t>(n_dofs)) {
        throw FEException("applyNumbering: permutation size is smaller than DOF count");
    }

    const GlobalIndex n_cells = dof_map.getNumCells();
    if (n_cells < 0) {
        throw FEException("applyNumbering: negative cell count");
    }

    const auto offsets = dof_map.getOffsets();
    const auto old_dof_indices = dof_map.getDofIndices();

    if (!offsets.empty() && offsets.size() < static_cast<std::size_t>(n_cells) + 1) {
        throw FEException("applyNumbering: DofMap offsets array is inconsistent with cell count");
    }

    // Apply permutation to flat DOF index buffer.
    std::vector<GlobalIndex> new_dof_indices(old_dof_indices.begin(), old_dof_indices.end());
    for (auto& dof : new_dof_indices) {
        if (dof >= 0 && static_cast<std::size_t>(dof) < permutation.size()) {
            dof = permutation[static_cast<std::size_t>(dof)];
        }
    }

    // Rebuild map in one pass to avoid corrupting CSR storage during mutation.
    std::vector<GlobalIndex> cell_ids(static_cast<std::size_t>(n_cells));
    std::iota(cell_ids.begin(), cell_ids.end(), GlobalIndex{0});

    DofMap rebuilt;
    const auto avg_dofs_per_cell =
        (n_cells > 0) ? static_cast<LocalIndex>(new_dof_indices.size() / static_cast<std::size_t>(n_cells))
                      : LocalIndex{0};
    rebuilt.reserve(n_cells, avg_dofs_per_cell);
    rebuilt.setNumDofs(dof_map.getNumDofs());
    rebuilt.setNumLocalDofs(dof_map.getNumLocalDofs());
    rebuilt.setCellDofsBatch(cell_ids, offsets, new_dof_indices);

    dof_map = std::move(rebuilt);
}

std::vector<GlobalIndex> invertPermutation(std::span<const GlobalIndex> permutation) {
    std::vector<GlobalIndex> inverse(permutation.size());
    for (std::size_t i = 0; i < permutation.size(); ++i) {
        auto j = static_cast<std::size_t>(permutation[i]);
        if (j < inverse.size()) {
            inverse[j] = static_cast<GlobalIndex>(i);
        }
    }
    return inverse;
}

std::vector<GlobalIndex> composePermutations(
    std::span<const GlobalIndex> first,
    std::span<const GlobalIndex> second) {

    std::vector<GlobalIndex> composed(first.size());
    for (std::size_t i = 0; i < first.size(); ++i) {
        auto j = static_cast<std::size_t>(first[i]);
        if (j < second.size()) {
            composed[i] = second[j];
        } else {
            composed[i] = static_cast<GlobalIndex>(i);
        }
    }
    return composed;
}

GlobalIndex computeBandwidth(
    std::span<const GlobalIndex> permutation,
    std::span<const GlobalIndex> adjacency,
    std::span<const GlobalIndex> adj_indices) {

    GlobalIndex bandwidth = 0;
    auto n = static_cast<GlobalIndex>(permutation.size());

    // Invert permutation for lookup
    auto inverse = invertPermutation(permutation);

    for (GlobalIndex i = 0; i < n; ++i) {
        auto new_i = permutation[static_cast<std::size_t>(i)];
        auto adj_start = static_cast<std::size_t>(adjacency[static_cast<std::size_t>(i)]);
        auto adj_end = static_cast<std::size_t>(adjacency[static_cast<std::size_t>(i) + 1]);

        for (std::size_t k = adj_start; k < adj_end && k < adj_indices.size(); ++k) {
            auto j = adj_indices[k];
            if (j >= 0 && static_cast<std::size_t>(j) < permutation.size()) {
                auto new_j = permutation[static_cast<std::size_t>(j)];
                GlobalIndex diff = std::abs(new_i - new_j);
                bandwidth = std::max(bandwidth, diff);
            }
        }
    }

    return bandwidth;
}

GlobalIndex computeProfile(
    std::span<const GlobalIndex> permutation,
    std::span<const GlobalIndex> adjacency,
    std::span<const GlobalIndex> adj_indices) {

    GlobalIndex profile = 0;
    auto n = static_cast<GlobalIndex>(permutation.size());

    for (GlobalIndex i = 0; i < n; ++i) {
        auto new_i = permutation[static_cast<std::size_t>(i)];
        auto adj_start = static_cast<std::size_t>(adjacency[static_cast<std::size_t>(i)]);
        auto adj_end = static_cast<std::size_t>(adjacency[static_cast<std::size_t>(i) + 1]);

        GlobalIndex min_j = new_i;
        for (std::size_t k = adj_start; k < adj_end && k < adj_indices.size(); ++k) {
            auto j = adj_indices[k];
            if (j >= 0 && static_cast<std::size_t>(j) < permutation.size()) {
                auto new_j = permutation[static_cast<std::size_t>(j)];
                min_j = std::min(min_j, new_j);
            }
        }

        profile += (new_i - min_j);
    }

    return profile;
}

NumberingStats computeNumberingStats(
    std::span<const GlobalIndex> permutation,
    std::span<const GlobalIndex> adjacency,
    std::span<const GlobalIndex> adj_indices,
    const std::string& strategy_name) {

    NumberingStats stats;
    stats.strategy_name = strategy_name;
    stats.bandwidth = computeBandwidth(permutation, adjacency, adj_indices);
    stats.profile = computeProfile(permutation, adjacency, adj_indices);

    // Compute max and average row nnz
    auto n = static_cast<GlobalIndex>(permutation.size());
    GlobalIndex total_nnz = 0;

    for (GlobalIndex i = 0; i < n && static_cast<std::size_t>(i + 1) < adjacency.size(); ++i) {
        auto row_nnz = adjacency[static_cast<std::size_t>(i) + 1] - adjacency[static_cast<std::size_t>(i)];
        stats.max_row_nnz = std::max(stats.max_row_nnz, row_nnz);
        total_nnz += row_nnz;
    }

    if (n > 0) {
        stats.avg_row_nnz = static_cast<double>(total_nnz) / n;
    }

    return stats;
}

// =============================================================================
// Factory Functions
// =============================================================================

std::unique_ptr<NumberingStrategy> createNumberingStrategy(const std::string& name) {
    // Convert to lowercase for comparison
    std::string lower_name;
    lower_name.reserve(name.size());
    for (char c : name) {
        lower_name.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    if (lower_name == "sequential" || lower_name == "identity") {
        return std::make_unique<SequentialNumbering>();
    }
    if (lower_name == "interleaved") {
        return std::make_unique<InterleavedNumbering>();
    }
    if (lower_name == "rcm" || lower_name == "cuthill-mckee" || lower_name == "cuthill_mckee") {
        return std::make_unique<CuthillMcKeeNumbering>(true);
    }
    if (lower_name == "cm" || lower_name == "forward-cuthill-mckee") {
        return std::make_unique<CuthillMcKeeNumbering>(false);
    }
    if (lower_name == "nested-dissection" || lower_name == "nested_dissection") {
        return std::make_unique<NestedDissectionNumbering>();
    }
    if (lower_name == "morton" || lower_name == "z-order" || lower_name == "z_order") {
        return std::make_unique<SpaceFillingCurveNumbering>(
            SpaceFillingCurveNumbering::CurveType::Morton);
    }
    if (lower_name == "hilbert") {
        return std::make_unique<SpaceFillingCurveNumbering>(
            SpaceFillingCurveNumbering::CurveType::Hilbert);
    }

    return nullptr;
}

} // namespace dofs
} // namespace FE
} // namespace svmp
