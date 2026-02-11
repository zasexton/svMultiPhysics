/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/FESystem.h"
#include "Systems/CoupledBoundaryManager.h"
#include "Systems/GlobalKernelStateProvider.h"
#include "Systems/MaterialStateProvider.h"
#include "Systems/OperatorBackends.h"

#include "Backends/Interfaces/DofPermutation.h"

#include "Assembly/Assembler.h"
#include "Assembly/AssemblyKernel.h"
#include "Assembly/AssemblerSelection.h"
#include "Assembly/MatrixFreeAssembler.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Assembly/MeshAccess.h"
#include "Mesh/Core/InterfaceMesh.h"
#include "Systems/MeshSearchAccess.h"
#endif

#include "Constraints/ParallelConstraints.h"

#include "Sparsity/ConstraintSparsityAugmenter.h"
#include "Sparsity/DistributedSparsityPattern.h"

#include "Systems/SystemsExceptions.h"

#include "Spaces/FunctionSpace.h"

#include "Dofs/EntityDofMap.h"

#include "Elements/ReferenceElement.h"

#include "Quadrature/QuadratureFactory.h"

#include "Core/FEConfig.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <optional>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#if FE_HAS_MPI
#  include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace systems {

namespace {

class AffineConstraintsQuery final : public sparsity::IConstraintQuery {
public:
    explicit AffineConstraintsQuery(constraints::AffineConstraints constraints)
        : constraints_(std::move(constraints))
    {}

    [[nodiscard]] bool isConstrained(GlobalIndex dof) const override
    {
        return constraints_.isConstrained(dof);
    }

    [[nodiscard]] std::vector<GlobalIndex> getMasterDofs(GlobalIndex constrained_dof) const override
    {
        auto view = constraints_.getConstraint(constrained_dof);
        if (!view.has_value()) {
            return {};
        }
        std::vector<GlobalIndex> masters;
        masters.reserve(view->entries.size());
        for (const auto& entry : view->entries) {
            masters.push_back(entry.master_dof);
        }
        return masters;
    }

    [[nodiscard]] std::vector<GlobalIndex> getAllConstrainedDofs() const override
    {
        return constraints_.getConstrainedDofs();
    }

    [[nodiscard]] std::size_t numConstraints() const override
    {
        return constraints_.numConstraints();
    }

private:
    constraints::AffineConstraints constraints_;
};

class PermutedAffineConstraintsQuery final : public sparsity::IConstraintQuery {
public:
    PermutedAffineConstraintsQuery(const constraints::AffineConstraints& constraints,
                                   std::span<const GlobalIndex> forward,
                                   std::span<const GlobalIndex> inverse)
        : constraints_(&constraints)
        , forward_(forward)
        , inverse_(inverse)
    {}

    [[nodiscard]] bool isConstrained(GlobalIndex dof) const override
    {
        FE_CHECK_NOT_NULL(constraints_, "PermutedAffineConstraintsQuery::constraints");
        if (dof < 0 || static_cast<std::size_t>(dof) >= inverse_.size()) {
            return false;
        }
        const auto fe = inverse_[static_cast<std::size_t>(dof)];
        if (fe < 0) {
            return false;
        }
        return constraints_->isConstrained(fe);
    }

    [[nodiscard]] std::vector<GlobalIndex> getMasterDofs(GlobalIndex constrained_dof) const override
    {
        FE_CHECK_NOT_NULL(constraints_, "PermutedAffineConstraintsQuery::constraints");
        if (constrained_dof < 0 || static_cast<std::size_t>(constrained_dof) >= inverse_.size()) {
            return {};
        }
        const auto fe_constrained = inverse_[static_cast<std::size_t>(constrained_dof)];
        if (fe_constrained < 0) {
            return {};
        }

        auto view = constraints_->getConstraint(fe_constrained);
        if (!view.has_value()) {
            return {};
        }

        std::vector<GlobalIndex> masters;
        masters.reserve(view->entries.size());
        for (const auto& entry : view->entries) {
            const auto fe_master = entry.master_dof;
            if (fe_master < 0 || static_cast<std::size_t>(fe_master) >= forward_.size()) {
                continue;
            }
            const auto fs_master = forward_[static_cast<std::size_t>(fe_master)];
            if (fs_master < 0) {
                continue;
            }
            masters.push_back(fs_master);
        }
        return masters;
    }

    [[nodiscard]] std::vector<GlobalIndex> getAllConstrainedDofs() const override
    {
        FE_CHECK_NOT_NULL(constraints_, "PermutedAffineConstraintsQuery::constraints");
        const auto fe_dofs = constraints_->getConstrainedDofs();
        std::vector<GlobalIndex> fs_dofs;
        fs_dofs.reserve(fe_dofs.size());
        for (const auto fe : fe_dofs) {
            if (fe < 0 || static_cast<std::size_t>(fe) >= forward_.size()) {
                continue;
            }
            const auto fs = forward_[static_cast<std::size_t>(fe)];
            if (fs < 0) {
                continue;
            }
            fs_dofs.push_back(fs);
        }
        std::sort(fs_dofs.begin(), fs_dofs.end());
        fs_dofs.erase(std::unique(fs_dofs.begin(), fs_dofs.end()), fs_dofs.end());
        return fs_dofs;
    }

    [[nodiscard]] std::size_t numConstraints() const override
    {
        FE_CHECK_NOT_NULL(constraints_, "PermutedAffineConstraintsQuery::constraints");
        return constraints_->numConstraints();
    }

private:
    const constraints::AffineConstraints* constraints_{nullptr};
    std::span<const GlobalIndex> forward_{};
    std::span<const GlobalIndex> inverse_{};
};

struct NodalInterleavedDofMap {
    int dof_per_node{0};
    GlobalIndex n_nodes{0};
    sparsity::IndexRange owned_range{};

    std::vector<GlobalIndex> fe_to_fs{};
    std::vector<GlobalIndex> fs_to_fe{};

    std::vector<int> ghost_nodes{};
    std::vector<unsigned char> node_is_relevant{};
    std::vector<unsigned char> node_is_ghost{};

    [[nodiscard]] bool isGhostNode(int node) const noexcept
    {
        if (node < 0 || static_cast<std::size_t>(node) >= node_is_ghost.size()) {
            return false;
        }
        return node_is_ghost[static_cast<std::size_t>(node)] != 0;
    }

    [[nodiscard]] bool isRelevantDof(GlobalIndex dof) const noexcept
    {
        if (dof_per_node <= 0 || dof < 0) {
            return false;
        }
        const auto node = static_cast<GlobalIndex>(dof / dof_per_node);
        if (node < 0 || static_cast<std::size_t>(node) >= node_is_relevant.size()) {
            return false;
        }
        return node_is_relevant[static_cast<std::size_t>(node)] != 0;
    }

    [[nodiscard]] GlobalIndex mapFeToFs(GlobalIndex fe_dof) const noexcept
    {
        if (fe_dof < 0 || static_cast<std::size_t>(fe_dof) >= fe_to_fs.size()) {
            return INVALID_GLOBAL_INDEX;
        }
        return fe_to_fs[static_cast<std::size_t>(fe_dof)];
    }
};

namespace {

constexpr std::uint32_t kSfcBits = 21u;
constexpr std::uint64_t kSfcMaxCoord = (1ULL << kSfcBits) - 1ULL;

[[nodiscard]] std::uint64_t morton3D(std::uint32_t xi, std::uint32_t yi, std::uint32_t zi)
{
    auto spread = [](std::uint64_t v) -> std::uint64_t {
        v = (v | (v << 32)) & 0x1f00000000ffffULL;
        v = (v | (v << 16)) & 0x1f0000ff0000ffULL;
        v = (v | (v << 8)) & 0x100f00f00f00f00fULL;
        v = (v | (v << 4)) & 0x10c30c30c30c30c3ULL;
        v = (v | (v << 2)) & 0x1249249249249249ULL;
        return v;
    };

    const std::uint64_t x = spread(xi);
    const std::uint64_t y = spread(yi);
    const std::uint64_t z = spread(zi);
    return x | (y << 1) | (z << 2);
}

[[nodiscard]] std::uint64_t hilbertIndexND(std::span<const std::uint32_t> coords, std::uint32_t bits)
{
    const int n = static_cast<int>(coords.size());
    if (n <= 0 || bits == 0u) {
        return 0;
    }
    FE_THROW_IF(n > 3, InvalidArgumentException, "hilbertIndexND: only 2D/3D supported");
    FE_THROW_IF(bits > 21u, InvalidArgumentException, "hilbertIndexND: bits must be <= 21 for uint64 index");

    std::array<std::uint32_t, 3> x{0u, 0u, 0u};
    for (int i = 0; i < n; ++i) {
        x[static_cast<std::size_t>(i)] = coords[static_cast<std::size_t>(i)];
    }

    // John Skilling, "Programming the Hilbert curve" (2004): coord -> Hilbert index (transpose form).
    std::uint32_t M = 1u << (bits - 1u);
    for (std::uint32_t Q = M; Q > 1u; Q >>= 1u) {
        const std::uint32_t P = Q - 1u;
        for (int i = 0; i < n; ++i) {
            if ((x[static_cast<std::size_t>(i)] & Q) != 0u) {
                x[0] ^= P;
            } else {
                const std::uint32_t t = (x[0] ^ x[static_cast<std::size_t>(i)]) & P;
                x[0] ^= t;
                x[static_cast<std::size_t>(i)] ^= t;
            }
        }
    }

    for (int i = 1; i < n; ++i) {
        x[static_cast<std::size_t>(i)] ^= x[static_cast<std::size_t>(i - 1)];
    }

    std::uint32_t t = 0u;
    for (std::uint32_t Q = M; Q > 1u; Q >>= 1u) {
        if ((x[static_cast<std::size_t>(n - 1)] & Q) != 0u) {
            t ^= (Q - 1u);
        }
    }
    for (int i = 0; i < n; ++i) {
        x[static_cast<std::size_t>(i)] ^= t;
    }

    // Interleave transpose bits into a single integer.
    std::uint64_t index = 0;
    for (int b = static_cast<int>(bits) - 1; b >= 0; --b) {
        for (int i = 0; i < n; ++i) {
            index <<= 1u;
            index |= static_cast<std::uint64_t>((x[static_cast<std::size_t>(i)] >> static_cast<std::uint32_t>(b)) & 1u);
        }
    }
    return index;
}

[[nodiscard]] std::uint64_t sfcCodeNormalized(double x, double y, double z, dofs::SpatialCurveType curve)
{
    auto normalize = [](double v) -> std::uint32_t {
        v = std::max(0.0, std::min(1.0, v));
        return static_cast<std::uint32_t>(v * static_cast<double>(kSfcMaxCoord));
    };

    const std::uint32_t xi = normalize(x);
    const std::uint32_t yi = normalize(y);
    const std::uint32_t zi = normalize(z);

    if (curve == dofs::SpatialCurveType::Hilbert) {
        const std::array<std::uint32_t, 3> c{xi, yi, zi};
        return hilbertIndexND(std::span<const std::uint32_t>(c.data(), 3), kSfcBits);
    }
    return morton3D(xi, yi, zi);
}

} // namespace

[[nodiscard]] std::optional<NodalInterleavedDofMap> tryBuildNodalInterleavedDofMap(const dofs::DofHandler& dof_handler,
                                                                                   const dofs::FieldDofMap& field_map,
                                                                                   const assembly::IMeshAccess& mesh,
                                                                                   const dofs::DofDistributionOptions& dof_options)
{
    const GlobalIndex total_dofs = dof_handler.getNumDofs();
    if (total_dofs <= 0) {
        return std::nullopt;
    }

    const std::size_t n_fields = field_map.numFields();
    if (n_fields == 0u) {
        return std::nullopt;
    }

    GlobalIndex n_nodes = -1;
    int dof_per_node = 0;
    for (std::size_t f = 0; f < n_fields; ++f) {
        const auto& field = field_map.getField(f);
        FE_THROW_IF(field.component_dof_layout != dofs::FieldComponentDofLayout::ComponentWise,
                    InvalidArgumentException,
                    "FESystem::setup: node-interleaved distributed sparsity requires component-wise fields");
        FE_THROW_IF(field.n_components <= 0, InvalidStateException,
                    "FESystem::setup: invalid field components for node-interleaved distributed sparsity");
        FE_THROW_IF(field.n_dofs % field.n_components != 0, InvalidStateException,
                    "FESystem::setup: field DOF count must be divisible by components for node-interleaved distributed sparsity");

        const GlobalIndex n_per_component = field.n_dofs / field.n_components;
        if (n_nodes < 0) {
            n_nodes = n_per_component;
        } else if (n_nodes != n_per_component) {
            return std::nullopt;
        }
        dof_per_node += field.n_components;
    }
    if (n_nodes <= 0 || dof_per_node <= 0) {
        return std::nullopt;
    }
    if (total_dofs != static_cast<GlobalIndex>(dof_per_node) * n_nodes) {
        return std::nullopt;
    }

#if !FE_HAS_MPI
    (void)dof_handler;
    (void)field_map;
    (void)mesh;
    (void)dof_options;
    return std::nullopt;
#else
    // In MPI, overlap backends (FSILS) require that owned rows form a contiguous range in the
    // backend (node-interleaved) indexing. When FE global numbering is process-count independent
    // (e.g., DenseGlobalIds), owned nodes are typically non-contiguous in node space. To keep the
    // backend happy without changing FE numbering, build a backend node permutation that groups
    // nodes by owner rank (owner-contiguous), and optionally orders nodes spatially within each
    // owner block (Morton/Hilbert).

    const int my_rank = dof_options.my_rank;
    const int world_size = dof_options.world_size;
    if (my_rank < 0 || world_size <= 1 || my_rank >= world_size) {
        return std::nullopt;
    }

    // Spatial ordering within the owner block is optional but default-on (note 18).
    const bool explicit_spatial =
        dof_options.numbering == dofs::DofNumberingStrategy::Morton ||
        dof_options.numbering == dofs::DofNumberingStrategy::Hilbert;
    const bool default_spatial =
        dof_options.enable_spatial_locality_ordering &&
        dof_options.numbering == dofs::DofNumberingStrategy::Sequential;
    const bool want_spatial = explicit_spatial || default_spatial;
    const dofs::SpatialCurveType curve =
        explicit_spatial
            ? (dof_options.numbering == dofs::DofNumberingStrategy::Hilbert ? dofs::SpatialCurveType::Hilbert
                                                                           : dofs::SpatialCurveType::Morton)
            : dof_options.spatial_curve;

    enum class NodeLayout { Block, Interleaved };
    // Decode per-field node/component indices using the actual monolithic field-map layout
    // (which may differ from dof_options.numbering when default spatial locality ordering is enabled).
    const NodeLayout node_layout =
        (field_map.layout() == dofs::FieldLayout::Block) ? NodeLayout::Block : NodeLayout::Interleaved;

    const auto& part = dof_handler.getPartition();
    const auto owned_size = part.localOwnedSize();
    if (owned_size <= 0) {
        return std::nullopt;
    }

    // Representative field/component used to identify nodes (stable across fields).
    const auto& rep_field = field_map.getField(0);
    if (rep_field.component_dof_layout != dofs::FieldComponentDofLayout::ComponentWise ||
        rep_field.n_components <= 0) {
        return std::nullopt;
    }

    const GlobalIndex rep_n_nodes = rep_field.n_dofs / std::max<GlobalIndex>(1, rep_field.n_components);
    if (rep_n_nodes != n_nodes) {
        return std::nullopt;
    }

    auto decode_node_comp = [&](const dofs::FieldDescriptor& field,
                                GlobalIndex local_dof) -> std::optional<std::pair<GlobalIndex, LocalIndex>> {
        if (local_dof < 0 || local_dof >= field.n_dofs || field.n_components <= 0) {
            return std::nullopt;
        }
        if (node_layout == NodeLayout::Interleaved) {
            const auto c = static_cast<LocalIndex>(local_dof % field.n_components);
            const auto node = local_dof / field.n_components;
            return std::make_pair(node, c);
        }
        const GlobalIndex n_per_component = field.n_dofs / field.n_components;
        if (n_per_component <= 0) {
            return std::nullopt;
        }
        const auto c = static_cast<LocalIndex>(local_dof / n_per_component);
	        const auto node = local_dof % n_per_component;
	        return std::make_pair(node, c);
	    };

	    // Derive locally relevant nodes across all fields/components, and record node ownership using
	    // locally present DOFs. We require nodal ownership to be consistent across the node's DOFs.
	    std::unordered_map<GlobalIndex, int> node_owner;
	    std::vector<GlobalIndex> relevant_nodes_fe;
	    {
	        const auto relevant_dofs = part.locallyRelevant().toVector();
	        relevant_nodes_fe.reserve(relevant_dofs.size() /
	                                      static_cast<std::size_t>(std::max<GlobalIndex>(1, dof_per_node)) +
	                                  8u);
	        node_owner.reserve(relevant_nodes_fe.capacity());

	        for (const auto fe : relevant_dofs) {
	            const auto fld = field_map.globalToField(fe);
	            if (!fld.has_value()) {
	                continue;
	            }
	            const auto& field = field_map.getField(static_cast<std::size_t>(fld->first));
	            const auto decoded = decode_node_comp(field, fld->second);
	            if (!decoded.has_value()) {
	                return std::nullopt;
	            }
	            const auto node = decoded->first;
	            if (node < 0 || node >= n_nodes) {
	                return std::nullopt;
	            }

	            const int owner = dof_handler.getDofMap().getDofOwner(fe);
	            if (owner < 0 || owner >= world_size) {
	                return std::nullopt;
	            }

	            auto [it, inserted] = node_owner.emplace(node, owner);
	            if (!inserted && it->second != owner) {
	                // Ownership differs across DOFs on the same node; cannot build a node-interleaved backend map.
	                return std::nullopt;
	            }

	            relevant_nodes_fe.push_back(node);
	        }
	    }

	    std::sort(relevant_nodes_fe.begin(), relevant_nodes_fe.end());
	    relevant_nodes_fe.erase(std::unique(relevant_nodes_fe.begin(), relevant_nodes_fe.end()), relevant_nodes_fe.end());
	    if (relevant_nodes_fe.empty()) {
	        return std::nullopt;
	    }

	    FE_THROW_IF(owned_size % static_cast<GlobalIndex>(dof_per_node) != 0,
	                InvalidArgumentException,
	                "FESystem::setup: nodal interleaved distributed sparsity requires owned DOFs to be a multiple of dof_per_node");
	    const GlobalIndex expected_owned_nodes = owned_size / static_cast<GlobalIndex>(dof_per_node);

	    std::vector<GlobalIndex> owned_nodes_fe;
	    std::vector<GlobalIndex> ghost_nodes_fe;
	    owned_nodes_fe.reserve(static_cast<std::size_t>(expected_owned_nodes) + 8u);
	    ghost_nodes_fe.reserve(relevant_nodes_fe.size());
	    for (const auto node : relevant_nodes_fe) {
	        const auto it = node_owner.find(node);
	        if (it == node_owner.end()) {
	            return std::nullopt;
	        }
	        const int owner = it->second;
	        if (owner == my_rank) {
	            owned_nodes_fe.push_back(node);
	        } else {
	            ghost_nodes_fe.push_back(node);
	        }
	    }

	    if (static_cast<GlobalIndex>(owned_nodes_fe.size()) != expected_owned_nodes) {
	        return std::nullopt;
	    }

    // Compute a spatial (Morton/Hilbert) ordering for owned nodes within this rank's owner block.
    struct OwnedNodeRec {
        std::uint64_t code{0};
        GlobalIndex node{-1};
        std::array<double, 3> xyz{0.0, 0.0, 0.0};
    };
    std::vector<OwnedNodeRec> owned_recs;
    owned_recs.reserve(owned_nodes_fe.size());

    const auto* emap = dof_handler.getEntityDofMap();
    const int dim = std::max(2, mesh.dimension());

    std::array<double, 3> min_xyz{std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::infinity()};
    std::array<double, 3> max_xyz{-std::numeric_limits<double>::infinity(),
                                  -std::numeric_limits<double>::infinity(),
                                  -std::numeric_limits<double>::infinity()};

    for (const auto node : owned_nodes_fe) {
        const GlobalIndex fe0 = field_map.componentToGlobal(0, 0, node);
        if (fe0 < 0 || fe0 >= total_dofs) {
            return std::nullopt;
        }

        std::array<double, 3> xyz{static_cast<double>(node), static_cast<double>(node), static_cast<double>(node)};
        bool have_xyz = false;
        if (want_spatial && emap) {
            if (const auto ent = emap->getDofEntity(fe0); ent && ent->kind == dofs::EntityKind::Vertex) {
                const auto p = mesh.getNodeCoordinates(ent->id);
                xyz = {static_cast<double>(p[0]), static_cast<double>(p[1]), static_cast<double>(p[2])};
                have_xyz = true;
            }
        }
        (void)have_xyz;

        for (int a = 0; a < dim && a < 3; ++a) {
            const auto ax = static_cast<std::size_t>(a);
            min_xyz[ax] = std::min(min_xyz[ax], xyz[ax]);
            max_xyz[ax] = std::max(max_xyz[ax], xyz[ax]);
        }

        owned_recs.push_back(OwnedNodeRec{0u, node, xyz});
    }

    auto normalize_axis = [&](double v, int axis) -> double {
        const auto ax = static_cast<std::size_t>(axis);
        const double lo = min_xyz[ax];
        const double hi = max_xyz[ax];
        if (!(hi > lo)) {
            return 0.0;
        }
        return (v - lo) / (hi - lo);
    };

    if (want_spatial) {
        for (auto& rec : owned_recs) {
            const double x = normalize_axis(rec.xyz[0], 0);
            const double y = (dim >= 2) ? normalize_axis(rec.xyz[1], 1) : 0.0;
            const double z = (dim >= 3) ? normalize_axis(rec.xyz[2], 2) : 0.0;
            rec.code = sfcCodeNormalized(x, y, z, curve);
        }
    }

    std::sort(owned_recs.begin(), owned_recs.end(), [&](const OwnedNodeRec& a, const OwnedNodeRec& b) {
        if (a.code != b.code) {
            return a.code < b.code;
        }
        return a.node < b.node;
    });

    const std::int64_t owned_count_local = static_cast<std::int64_t>(owned_recs.size());
    std::vector<std::int64_t> owned_counts(static_cast<std::size_t>(world_size), 0);
    MPI_Allgather(&owned_count_local, 1, MPI_INT64_T,
                  owned_counts.data(), 1, MPI_INT64_T, dof_options.mpi_comm);

    std::int64_t node_offset = 0;
    std::int64_t node_total = 0;
    for (int r = 0; r < world_size; ++r) {
        if (r < my_rank) {
            node_offset += owned_counts[static_cast<std::size_t>(r)];
        }
        node_total += owned_counts[static_cast<std::size_t>(r)];
    }
    if (node_total != static_cast<std::int64_t>(n_nodes)) {
        return std::nullopt;
    }

    std::unordered_map<GlobalIndex, GlobalIndex> fe_node_to_backend;
    fe_node_to_backend.reserve(owned_recs.size() + ghost_nodes_fe.size());
    for (std::size_t i = 0; i < owned_recs.size(); ++i) {
        const auto node = owned_recs[i].node;
        fe_node_to_backend.emplace(node, static_cast<GlobalIndex>(node_offset + static_cast<std::int64_t>(i)));
    }

	    // Resolve backend node ids for ghost nodes by querying their owner ranks.
	    std::vector<std::vector<GlobalIndex>> requests(static_cast<std::size_t>(world_size));
	    for (const auto node : ghost_nodes_fe) {
	        const auto it = node_owner.find(node);
	        if (it == node_owner.end()) {
	            return std::nullopt;
	        }
	        const int owner = it->second;
	        if (owner < 0 || owner >= world_size) {
	            return std::nullopt;
	        }
	        if (owner == my_rank) {
            continue;
        }
        requests[static_cast<std::size_t>(owner)].push_back(node);
    }
    for (auto& v : requests) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }

    std::vector<int> send_counts(static_cast<std::size_t>(world_size), 0);
    std::vector<int> recv_counts(static_cast<std::size_t>(world_size), 0);
    std::vector<int> send_displs(static_cast<std::size_t>(world_size), 0);
    std::vector<int> recv_displs(static_cast<std::size_t>(world_size), 0);

    std::vector<GlobalIndex> send_nodes;
    send_nodes.reserve(ghost_nodes_fe.size());
    for (int r = 0; r < world_size; ++r) {
        const auto& v = requests[static_cast<std::size_t>(r)];
        send_counts[static_cast<std::size_t>(r)] = static_cast<int>(v.size());
    }
    int total_send = 0;
    for (int r = 0; r < world_size; ++r) {
        send_displs[static_cast<std::size_t>(r)] = total_send;
        total_send += send_counts[static_cast<std::size_t>(r)];
    }
    send_nodes.resize(static_cast<std::size_t>(total_send));
    for (int r = 0; r < world_size; ++r) {
        const auto& v = requests[static_cast<std::size_t>(r)];
        std::copy(v.begin(), v.end(), send_nodes.begin() + send_displs[static_cast<std::size_t>(r)]);
    }

    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT, dof_options.mpi_comm);
    int total_recv = 0;
    for (int r = 0; r < world_size; ++r) {
        recv_displs[static_cast<std::size_t>(r)] = total_recv;
        total_recv += recv_counts[static_cast<std::size_t>(r)];
    }

    std::vector<GlobalIndex> recv_nodes(static_cast<std::size_t>(total_recv));
    MPI_Alltoallv(send_nodes.data(), send_counts.data(), send_displs.data(), MPI_INT64_T,
                  recv_nodes.data(), recv_counts.data(), recv_displs.data(), MPI_INT64_T,
                  dof_options.mpi_comm);

    std::vector<GlobalIndex> send_backends(static_cast<std::size_t>(total_recv), INVALID_GLOBAL_INDEX);
    for (std::size_t i = 0; i < recv_nodes.size(); ++i) {
        const auto node = recv_nodes[i];
        const auto it = fe_node_to_backend.find(node);
        if (it == fe_node_to_backend.end()) {
            return std::nullopt;
        }
        send_backends[i] = it->second;
    }

    std::vector<GlobalIndex> recv_backends(static_cast<std::size_t>(total_send), INVALID_GLOBAL_INDEX);
    MPI_Alltoallv(send_backends.data(), recv_counts.data(), recv_displs.data(), MPI_INT64_T,
                  recv_backends.data(), send_counts.data(), send_displs.data(), MPI_INT64_T,
                  dof_options.mpi_comm);

    std::vector<int> ghost_nodes_backend;
    ghost_nodes_backend.reserve(ghost_nodes_fe.size());
    for (std::size_t i = 0; i < send_nodes.size(); ++i) {
        const auto node = send_nodes[i];
        const auto backend = recv_backends[i];
        if (backend < 0 || backend >= n_nodes) {
            return std::nullopt;
        }
        fe_node_to_backend.emplace(node, backend);
        ghost_nodes_backend.push_back(static_cast<int>(backend));
    }

    std::sort(ghost_nodes_backend.begin(), ghost_nodes_backend.end());
    ghost_nodes_backend.erase(std::unique(ghost_nodes_backend.begin(), ghost_nodes_backend.end()),
                              ghost_nodes_backend.end());

    NodalInterleavedDofMap map;
    map.dof_per_node = dof_per_node;
    map.n_nodes = n_nodes;
    map.owned_range.first = static_cast<GlobalIndex>(node_offset) * dof_per_node;
    map.owned_range.last = static_cast<GlobalIndex>(node_offset + owned_count_local) * dof_per_node;
    map.ghost_nodes = std::move(ghost_nodes_backend);
    map.node_is_ghost.assign(static_cast<std::size_t>(n_nodes), 0);
    map.node_is_relevant.assign(static_cast<std::size_t>(n_nodes), 0);

    const GlobalIndex owned_node_start = static_cast<GlobalIndex>(node_offset);
    const GlobalIndex owned_node_end = static_cast<GlobalIndex>(node_offset + owned_count_local);
    for (GlobalIndex node = owned_node_start; node < owned_node_end; ++node) {
        if (node < 0 || node >= n_nodes) {
            return std::nullopt;
        }
        map.node_is_relevant[static_cast<std::size_t>(node)] = 1;
    }
    for (const int node : map.ghost_nodes) {
        if (node < 0 || static_cast<GlobalIndex>(node) >= n_nodes) {
            return std::nullopt;
        }
        map.node_is_ghost[static_cast<std::size_t>(node)] = 1;
        map.node_is_relevant[static_cast<std::size_t>(node)] = 1;
    }

	    map.fe_to_fs.assign(static_cast<std::size_t>(total_dofs), INVALID_GLOBAL_INDEX);
	    map.fs_to_fe.assign(static_cast<std::size_t>(total_dofs), INVALID_GLOBAL_INDEX);

	    // Fill mappings for locally relevant (owned + ghost) nodes only.
	    for (const auto node : relevant_nodes_fe) {
	        const auto it = fe_node_to_backend.find(node);
	        if (it == fe_node_to_backend.end()) {
	            return std::nullopt;
	        }
        const GlobalIndex backend_node = it->second;
        if (backend_node < 0 || backend_node >= n_nodes) {
            return std::nullopt;
        }

        int comp_offset = 0;
        for (std::size_t f = 0; f < n_fields; ++f) {
            const auto& field = field_map.getField(f);
            for (LocalIndex c = 0; c < field.n_components; ++c) {
                const GlobalIndex fe = field_map.componentToGlobal(f, c, node);
                const GlobalIndex fs =
                    backend_node * static_cast<GlobalIndex>(dof_per_node) + static_cast<GlobalIndex>(comp_offset);
                if (fe < 0 || fe >= total_dofs || fs < 0 || fs >= total_dofs) {
                    return std::nullopt;
                }
                auto& fwd = map.fe_to_fs[static_cast<std::size_t>(fe)];
                auto& inv = map.fs_to_fe[static_cast<std::size_t>(fs)];
                if (fwd != INVALID_GLOBAL_INDEX && fwd != fs) {
                    return std::nullopt;
                }
                if (inv != INVALID_GLOBAL_INDEX && inv != fe) {
                    return std::nullopt;
                }
                fwd = fs;
                inv = fe;
                ++comp_offset;
            }
        }
        if (comp_offset != dof_per_node) {
            return std::nullopt;
        }
    }

    return map;
#endif // FE_HAS_MPI
}

[[nodiscard]] bool numberingInterleavesComponents(const dofs::DofDistributionOptions& dof_options)
{
    const bool explicit_spatial =
        dof_options.numbering == dofs::DofNumberingStrategy::Morton ||
        dof_options.numbering == dofs::DofNumberingStrategy::Hilbert;
    bool apply_default_spatial =
        dof_options.enable_spatial_locality_ordering &&
        dof_options.numbering == dofs::DofNumberingStrategy::Sequential;

    // Mirror DofHandler::distributeDofs default-spatial disablement in MPI.
    if (dof_options.world_size > 1 &&
        (dof_options.global_numbering != dofs::GlobalNumberingMode::OwnerContiguous ||
         dof_options.reproducible_across_communicators)) {
        apply_default_spatial = false;
    }

    return dof_options.numbering == dofs::DofNumberingStrategy::Interleaved ||
           explicit_spatial ||
           apply_default_spatial;
}

[[nodiscard]] dofs::FieldLayout fieldLayoutForSystem(std::size_t n_fields,
                                                     const dofs::DofDistributionOptions& dof_options)
{
    const bool interleaved = numberingInterleavesComponents(dof_options);
    if (n_fields > 1u) {
        return interleaved ? dofs::FieldLayout::FieldBlock : dofs::FieldLayout::Block;
    }
    return interleaved ? dofs::FieldLayout::Interleaved : dofs::FieldLayout::Block;
}

LocalIndex maxCellQuadraturePoints(const assembly::IMeshAccess& mesh,
                                   const spaces::FunctionSpace& test_space)
{
    LocalIndex max_qpts = 0;
    mesh.forEachCell([&](GlobalIndex cell_id) {
        const ElementType cell_type = mesh.getCellType(cell_id);
        const auto& test_element = test_space.getElement(cell_type, cell_id);

        auto quad_rule = test_element.quadrature();
        if (!quad_rule) {
            const int quad_order = quadrature::QuadratureFactory::recommended_order(
                test_element.polynomial_order(), false);
            quad_rule = quadrature::QuadratureFactory::create(cell_type, quad_order);
        }

        const auto n = static_cast<LocalIndex>(quad_rule->num_points());
        max_qpts = std::max(max_qpts, n);
    });

    return max_qpts;
}

ElementType faceTypeForFace(const assembly::IMeshAccess& mesh,
                            GlobalIndex face_id,
                            GlobalIndex cell_id)
{
    const ElementType cell_type = mesh.getCellType(cell_id);
    const LocalIndex local_face_id = mesh.getLocalFaceIndex(face_id, cell_id);

    elements::ReferenceElement ref = elements::ReferenceElement::create(cell_type);
    const auto& face_nodes = ref.face_nodes(static_cast<std::size_t>(local_face_id));

    switch (face_nodes.size()) {
        case 2:
            return ElementType::Line2;
        case 3:
            return ElementType::Triangle3;
        case 4:
            return ElementType::Quad4;
        default:
            throw FEException("FESystem::setup: unsupported face topology",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
}

LocalIndex maxBoundaryFaceQuadraturePoints(const assembly::IMeshAccess& mesh,
                                           const spaces::FunctionSpace& test_space,
                                           const spaces::FunctionSpace& trial_space)
{
    LocalIndex max_qpts = 0;
    mesh.forEachBoundaryFace(/*marker=*/-1, [&](GlobalIndex face_id, GlobalIndex cell_id) {
        const ElementType cell_type = mesh.getCellType(cell_id);
        const auto& test_element = test_space.getElement(cell_type, cell_id);
        const auto& trial_element = trial_space.getElement(cell_type, cell_id);

        const int quad_order = quadrature::QuadratureFactory::recommended_order(
            std::max(test_element.polynomial_order(), trial_element.polynomial_order()), false);

        const ElementType face_type = faceTypeForFace(mesh, face_id, cell_id);
        auto quad_rule = quadrature::QuadratureFactory::create(face_type, quad_order);
        max_qpts = std::max(max_qpts, static_cast<LocalIndex>(quad_rule->num_points()));
    });

    return max_qpts;
}

LocalIndex maxInteriorFaceQuadraturePoints(const assembly::IMeshAccess& mesh,
                                           const spaces::FunctionSpace& test_space,
                                           const spaces::FunctionSpace& trial_space)
{
    LocalIndex max_qpts = 0;
    mesh.forEachInteriorFace([&](GlobalIndex face_id, GlobalIndex cell_minus, GlobalIndex cell_plus) {
        const ElementType cell_type_minus = mesh.getCellType(cell_minus);
        const ElementType cell_type_plus = mesh.getCellType(cell_plus);

        const auto& test_element_minus = test_space.getElement(cell_type_minus, cell_minus);
        const auto& trial_element_minus = trial_space.getElement(cell_type_minus, cell_minus);
        const auto& test_element_plus = test_space.getElement(cell_type_plus, cell_plus);
        const auto& trial_element_plus = trial_space.getElement(cell_type_plus, cell_plus);

        const int quad_order = quadrature::QuadratureFactory::recommended_order(
            std::max({test_element_minus.polynomial_order(),
                      trial_element_minus.polynomial_order(),
                      test_element_plus.polynomial_order(),
                      trial_element_plus.polynomial_order()}),
            false);

        const ElementType face_type_minus = faceTypeForFace(mesh, face_id, cell_minus);
        const ElementType face_type_plus = faceTypeForFace(mesh, face_id, cell_plus);
        FE_THROW_IF(face_type_minus != face_type_plus, InvalidStateException,
                    "FESystem::setup: interior face has mismatched face topology between adjacent cells");

        auto quad_rule = quadrature::QuadratureFactory::create(face_type_minus, quad_order);
        max_qpts = std::max(max_qpts, static_cast<LocalIndex>(quad_rule->num_points()));
    });

    return max_qpts;
}

} // namespace

void FESystem::setup(const SetupOptions& opts, const SetupInputs& inputs)
{
    invalidateSetup();

    FE_THROW_IF(field_registry_.size() == 0u, InvalidStateException,
                "FESystem::setup: no fields registered");

    // ---------------------------------------------------------------------
    // DOFs (multi-field)
    // ---------------------------------------------------------------------
    dof_handler_ = dofs::DofHandler{};
    field_dof_handlers_.clear();
    field_dof_offsets_.clear();

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    if (!mesh_ && inputs.mesh) {
        mesh_ = inputs.mesh;
        coord_cfg_ = inputs.coord_cfg;
        mesh_access_ = std::make_shared<assembly::MeshAccess>(*mesh_, coord_cfg_);
        if (!search_access_) {
            search_access_ = std::make_shared<MeshSearchAccess>(*mesh_, coord_cfg_);
        }
    }
#endif

    const auto n_fields = field_registry_.size();
    field_dof_handlers_.resize(n_fields);
    field_dof_offsets_.assign(n_fields, 0);

    auto distribute_field = [&](const FieldRecord& rec) {
        FE_CHECK_NOT_NULL(rec.space.get(), "FESystem::setup: field.space");

        dofs::DofHandler dh;

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
        if (mesh_) {
            dh.distributeDofs(*mesh_, *rec.space, opts.dof_options);
            dh.finalize();
            return dh;
        }
#endif

        if (inputs.topology_override.has_value()) {
            const auto& topology = *inputs.topology_override;
            FE_THROW_IF(topology.n_cells <= 0, InvalidArgumentException,
                        "FESystem::setup: topology_override has no cells");

            dh.distributeDofs(topology, *rec.space, opts.dof_options);
            dh.finalize();
            return dh;
        }

        FE_THROW(InvalidArgumentException,
                 "FESystem::setup: need Mesh (SVMP_FE_WITH_MESH) or topology_override for DOF distribution");
    };

    for (const auto& rec : field_registry_.records()) {
        const auto idx = static_cast<std::size_t>(rec.id);
        FE_THROW_IF(rec.id < 0 || idx >= field_dof_handlers_.size(), InvalidStateException,
                    "FESystem::setup: field registry contains invalid FieldId");
        field_dof_handlers_[idx] = distribute_field(rec);
    }

    // Assign monolithic offsets in field registration order.
    GlobalIndex total_dofs = 0;
    for (const auto& rec : field_registry_.records()) {
        const auto idx = static_cast<std::size_t>(rec.id);
        field_dof_offsets_[idx] = total_dofs;
        total_dofs += field_dof_handlers_[idx].getNumDofs();
    }

    // Build a monolithic DofMap + EntityDofMap so Systems can expose a single global DofHandler.
    const auto n_cells = meshAccess().numCells();
    LocalIndex approx_dofs_per_cell = 0;
    for (const auto& rec : field_registry_.records()) {
        const auto idx = static_cast<std::size_t>(rec.id);
        approx_dofs_per_cell = static_cast<LocalIndex>(
            approx_dofs_per_cell + field_dof_handlers_[idx].getDofMap().getMaxDofsPerCell());
    }

    dofs::DofMap monolithic_map(n_cells, total_dofs, approx_dofs_per_cell);
    monolithic_map.setNumDofs(total_dofs);
    monolithic_map.setNumLocalDofs(0);

    std::vector<GlobalIndex> cell_dofs;
    for (GlobalIndex cell = 0; cell < n_cells; ++cell) {
        cell_dofs.clear();
        for (const auto& rec : field_registry_.records()) {
            const auto idx = static_cast<std::size_t>(rec.id);
            const auto offset = field_dof_offsets_[idx];
            auto local = field_dof_handlers_[idx].getDofMap().getCellDofs(cell);
            cell_dofs.reserve(cell_dofs.size() + local.size());
            for (auto d : local) {
                cell_dofs.push_back(d + offset);
            }
        }
        monolithic_map.setCellDofs(cell, cell_dofs);
    }

    // Merge entity-DOF maps if available (Mesh-driven workflows).
    std::unique_ptr<dofs::EntityDofMap> merged_entity_map;
    {
        const dofs::EntityDofMap* first = nullptr;
        for (const auto& rec : field_registry_.records()) {
            const auto idx = static_cast<std::size_t>(rec.id);
            auto* map = field_dof_handlers_[idx].getEntityDofMap();
            if (map == nullptr) {
                first = nullptr;
                break;
            }
            if (!first) {
                first = map;
            }
        }

        if (first) {
            merged_entity_map = std::make_unique<dofs::EntityDofMap>();
            merged_entity_map->reserve(first->numVertices(), first->numEdges(),
                                       first->numFaces(), first->numCells());

            std::vector<GlobalIndex> entity_dofs;

            for (GlobalIndex v = 0; v < first->numVertices(); ++v) {
                entity_dofs.clear();
                for (const auto& rec : field_registry_.records()) {
                    const auto idx = static_cast<std::size_t>(rec.id);
                    const auto offset = field_dof_offsets_[idx];
                    const auto* emap = field_dof_handlers_[idx].getEntityDofMap();
                    FE_CHECK_NOT_NULL(emap, "FESystem::setup: EntityDofMap");
                    auto vdofs = emap->getVertexDofs(v);
                    entity_dofs.reserve(entity_dofs.size() + vdofs.size());
                    for (auto d : vdofs) {
                        entity_dofs.push_back(d + offset);
                    }
                }
                merged_entity_map->setVertexDofs(v, entity_dofs);
            }

            for (GlobalIndex e = 0; e < first->numEdges(); ++e) {
                entity_dofs.clear();
                for (const auto& rec : field_registry_.records()) {
                    const auto idx = static_cast<std::size_t>(rec.id);
                    const auto offset = field_dof_offsets_[idx];
                    const auto* emap = field_dof_handlers_[idx].getEntityDofMap();
                    FE_CHECK_NOT_NULL(emap, "FESystem::setup: EntityDofMap");
                    auto edofs = emap->getEdgeDofs(e);
                    entity_dofs.reserve(entity_dofs.size() + edofs.size());
                    for (auto d : edofs) {
                        entity_dofs.push_back(d + offset);
                    }
                }
                merged_entity_map->setEdgeDofs(e, entity_dofs);
            }

            for (GlobalIndex f = 0; f < first->numFaces(); ++f) {
                entity_dofs.clear();
                for (const auto& rec : field_registry_.records()) {
                    const auto idx = static_cast<std::size_t>(rec.id);
                    const auto offset = field_dof_offsets_[idx];
                    const auto* emap = field_dof_handlers_[idx].getEntityDofMap();
                    FE_CHECK_NOT_NULL(emap, "FESystem::setup: EntityDofMap");
                    auto fdofs = emap->getFaceDofs(f);
                    entity_dofs.reserve(entity_dofs.size() + fdofs.size());
                    for (auto d : fdofs) {
                        entity_dofs.push_back(d + offset);
                    }
                }
                merged_entity_map->setFaceDofs(f, entity_dofs);
            }

            for (GlobalIndex c = 0; c < first->numCells(); ++c) {
                entity_dofs.clear();
                for (const auto& rec : field_registry_.records()) {
                    const auto idx = static_cast<std::size_t>(rec.id);
                    const auto offset = field_dof_offsets_[idx];
                    const auto* emap = field_dof_handlers_[idx].getEntityDofMap();
                    FE_CHECK_NOT_NULL(emap, "FESystem::setup: EntityDofMap");
                    auto cdofs = emap->getCellInteriorDofs(c);
                    entity_dofs.reserve(entity_dofs.size() + cdofs.size());
                    for (auto d : cdofs) {
                        entity_dofs.push_back(d + offset);
                    }
                }
                merged_entity_map->setCellInteriorDofs(c, entity_dofs);
            }

            merged_entity_map->buildReverseMapping();
            merged_entity_map->finalize();
        }
    }

    dofs::DofPartition part;
    {
        std::vector<dofs::IndexInterval> owned_intervals;
        std::vector<GlobalIndex> owned_explicit;
        std::vector<GlobalIndex> ghost_explicit;

        for (const auto& rec : field_registry_.records()) {
            const auto idx = static_cast<std::size_t>(rec.id);
            const auto offset = field_dof_offsets_[idx];
            const auto& fpart = field_dof_handlers_[idx].getPartition();

            if (auto range = fpart.locallyOwned().contiguousRange()) {
                owned_intervals.push_back(dofs::IndexInterval{range->begin + offset, range->end + offset});
            } else {
                auto vec = fpart.locallyOwned().toVector();
                owned_explicit.reserve(owned_explicit.size() + vec.size());
                for (auto d : vec) {
                    owned_explicit.push_back(d + offset);
                }
            }

            auto ghosts = fpart.ghost().toVector();
            ghost_explicit.reserve(ghost_explicit.size() + ghosts.size());
            for (auto d : ghosts) {
                ghost_explicit.push_back(d + offset);
            }
        }

        dofs::IndexSet owned;
        if (!owned_intervals.empty()) {
            owned = dofs::IndexSet(std::move(owned_intervals));
        }
        if (!owned_explicit.empty()) {
            owned = owned.unionWith(dofs::IndexSet(std::move(owned_explicit)));
        }

        dofs::IndexSet ghost;
        if (!ghost_explicit.empty()) {
            ghost = dofs::IndexSet(std::move(ghost_explicit));
        }

        part = dofs::DofPartition(std::move(owned), std::move(ghost));
        part.setGlobalSize(total_dofs);
    }

    monolithic_map.setNumLocalDofs(part.localOwnedSize());

    // Systems assembly relies on DOF ownership queries for both OwnedRowsOnly and ReverseScatter
    // ghost policies. The monolithic DofMap is built by offsetting per-field DOF indices, so we
    // re-expose ownership by forwarding to each field's DofMap ownership function.
    monolithic_map.setMyRank(opts.dof_options.my_rank);
    {
        std::vector<GlobalIndex> offsets;
        std::vector<GlobalIndex> sizes;
        std::vector<const dofs::DofMap*> maps;
        offsets.reserve(field_registry_.size());
        sizes.reserve(field_registry_.size());
        maps.reserve(field_registry_.size());

        for (const auto& rec : field_registry_.records()) {
            const auto idx = static_cast<std::size_t>(rec.id);
            offsets.push_back(field_dof_offsets_[idx]);
            sizes.push_back(field_dof_handlers_[idx].getNumDofs());
            maps.push_back(&field_dof_handlers_[idx].getDofMap());
        }

        monolithic_map.setDofOwnership(
            [offsets = std::move(offsets),
             sizes = std::move(sizes),
             maps = std::move(maps)](GlobalIndex global_dof) -> int {
                if (global_dof < 0 || offsets.empty()) {
                    return 0;
                }

                const auto it = std::upper_bound(offsets.begin(), offsets.end(), global_dof);
                const std::size_t block =
                    (it == offsets.begin()) ? 0u : static_cast<std::size_t>(std::distance(offsets.begin(), it) - 1);
                if (block >= offsets.size() || block >= sizes.size() || block >= maps.size()) {
                    return 0;
                }

                const GlobalIndex local = global_dof - offsets[block];
                if (local < 0 || local >= sizes[block]) {
                    return 0;
                }

                const auto* map = maps[block];
                if (!map) {
                    return 0;
                }
                return map->getDofOwner(local);
            });
    }

    dof_handler_ = dofs::DofHandler{};
    dof_handler_.setDofMap(std::move(monolithic_map));
    dof_handler_.setPartition(std::move(part));
    dof_handler_.setRankInfo(opts.dof_options.my_rank, opts.dof_options.world_size);
    if (merged_entity_map) {
        dof_handler_.setEntityDofMap(std::move(merged_entity_map));
    }
    // Preserve per-cell orientation metadata (H(curl)/H(div)) by copying it from any field handler
    // that computed it during DOF distribution.
    for (const auto& rec : field_registry_.records()) {
        const auto idx = static_cast<std::size_t>(rec.id);
        if (idx < field_dof_handlers_.size() && field_dof_handlers_[idx].hasCellOrientations()) {
            dof_handler_.copyCellOrientationsFrom(field_dof_handlers_[idx]);
            break;
        }
    }
    dof_handler_.finalize();

    // ---------------------------------------------------------------------
    // Field/block metadata (monolithic across fields)
    // ---------------------------------------------------------------------
    field_map_ = dofs::FieldDofMap{};
    field_map_.setLayout(fieldLayoutForSystem(field_registry_.size(), opts.dof_options));

    for (const auto& rec : field_registry_.records()) {
        const auto idx = static_cast<std::size_t>(rec.id);
        const auto n_components = rec.space->value_dimension();
        const auto n_dofs_field = field_dof_handlers_[idx].getNumDofs();

        if (n_components <= 1) {
            field_map_.addScalarField(rec.name, n_dofs_field);
        } else {
            const bool is_vector_basis =
                (rec.space->continuity() == Continuity::H_curl || rec.space->continuity() == Continuity::H_div);
            if (is_vector_basis) {
                field_map_.addVectorBasisField(rec.name,
                                               static_cast<LocalIndex>(n_components),
                                               n_dofs_field);
            } else {
                FE_THROW_IF(n_dofs_field % n_components != 0, InvalidStateException,
                            "FESystem::setup: vector-valued field has non-divisible DOF count");
                field_map_.addVectorField(rec.name, static_cast<LocalIndex>(n_components),
                                          n_dofs_field / n_components);
            }
        }
    }
    field_map_.finalize();
    block_map_.reset();
    if (field_registry_.size() > 1u) {
        auto blocks = std::make_unique<dofs::BlockDofMap>();
        for (const auto& rec : field_registry_.records()) {
            const auto idx = static_cast<std::size_t>(rec.id);
            blocks->addBlock(rec.name, field_dof_handlers_[idx].getNumDofs());
        }
        blocks->finalize();
        block_map_ = std::move(blocks);
    }

    // ---------------------------------------------------------------------
    // Constraints
    // ---------------------------------------------------------------------
    affine_constraints_.clear();
    for (auto& c : system_constraint_defs_) {
        FE_CHECK_NOT_NULL(c.get(), "FESystem::setup: system constraint");
        c->apply(*this, affine_constraints_);
    }
    for (const auto& c : constraint_defs_) {
        FE_CHECK_NOT_NULL(c.get(), "FESystem::setup: constraint");
        c->apply(affine_constraints_);
    }

    // Synchronize in MPI before closing.
    //
    // NOTE: The partition-only ParallelConstraints ctor is a serial/no-op helper (world_size=1).
    // In MPI runs we must construct with an MPI communicator so constraints on shared/ghost DOFs
    // are imported from the owning rank before close()/assembly.
#if FE_HAS_MPI
    int mpi_initialized_constraints = 0;
    MPI_Initialized(&mpi_initialized_constraints);
    std::optional<constraints::ParallelConstraints> parallel;
    if (mpi_initialized_constraints) {
        parallel.emplace(MPI_COMM_WORLD, dof_handler_.getPartition());
    } else {
        parallel.emplace(dof_handler_.getPartition());
    }
#else
    std::optional<constraints::ParallelConstraints> parallel;
    parallel.emplace(dof_handler_.getPartition());
#endif
    if (parallel && parallel->isParallel()) {
        parallel->synchronize(affine_constraints_);
    }

    affine_constraints_.close();

    // ---------------------------------------------------------------------
    // Sparsity
    // ---------------------------------------------------------------------
    sparsity_by_op_.clear();
    distributed_sparsity_by_op_.clear();
    const auto op_tags = operator_registry_.list();
    const auto n_total_dofs = dof_handler_.getNumDofs();
    const auto n_cells_sparsity = meshAccess().numCells();

    const auto& partition = dof_handler_.getPartition();
    const bool mpi_parallel = (partition.globalSize() > 0) && (partition.globalSize() > partition.localOwnedSize());

    enum class DistSparsityMode { None, ContiguousRange, NodalInterleaved };

    DistSparsityMode dist_mode = DistSparsityMode::None;
    sparsity::IndexRange owned_range{};
    // Backend DOF permutation (FSILS overlap) is needed in MPI even when FE owned DOFs are already
    // owner-contiguous. Keep the distributed sparsity in Natural indexing when we can, but still
    // provide a node-interleaved backend mapping.
    std::optional<NodalInterleavedDofMap> backend_map{};
    std::optional<NodalInterleavedDofMap> nodal_map{};

#if FE_HAS_MPI
    int mpi_initialized_dof_map = 0;
    MPI_Initialized(&mpi_initialized_dof_map);
    if (mpi_parallel && mpi_initialized_dof_map) {
        backend_map = tryBuildNodalInterleavedDofMap(dof_handler_, field_map_, meshAccess(), opts.dof_options);
    }
#endif

    const auto owned_iv_opt = partition.locallyOwned().contiguousRange();
    if (mpi_parallel && owned_iv_opt.has_value()) {
        dist_mode = DistSparsityMode::ContiguousRange;
        owned_range = sparsity::IndexRange{owned_iv_opt->begin, owned_iv_opt->end};
    } else if (mpi_parallel && backend_map.has_value()) {
        dist_mode = DistSparsityMode::NodalInterleaved;
        owned_range = backend_map->owned_range;
        nodal_map = std::move(backend_map);
    }

    const auto& ghost_set = partition.ghost();
    const auto& relevant_set = partition.locallyRelevant();

    for (const auto& tag : op_tags) {
        auto pattern = std::make_unique<sparsity::SparsityPattern>(
            n_total_dofs, n_total_dofs);

        std::unique_ptr<sparsity::DistributedSparsityPattern> dist_pattern;
        std::unordered_map<GlobalIndex, std::vector<GlobalIndex>> ghost_row_cols;
        if (dist_mode != DistSparsityMode::None) {
            dist_pattern = std::make_unique<sparsity::DistributedSparsityPattern>(
                owned_range, owned_range, n_total_dofs, n_total_dofs);
            dist_pattern->setDofIndexing(dist_mode == DistSparsityMode::NodalInterleaved
                                             ? sparsity::DistributedSparsityPattern::DofIndexing::NodalInterleaved
                                             : sparsity::DistributedSparsityPattern::DofIndexing::Natural);
        }

		        const auto& def = operator_registry_.get(tag);

		        std::vector<std::pair<FieldId, FieldId>> cell_pairs;
		        std::vector<std::tuple<int, FieldId, FieldId>> boundary_pairs;
		        std::vector<std::pair<FieldId, FieldId>> interior_pairs;
		        std::vector<std::tuple<int, FieldId, FieldId>> interface_pairs;

		        cell_pairs.reserve(def.cells.size());
		        boundary_pairs.reserve(def.boundary.size());
		        interior_pairs.reserve(def.interior.size());
		        interface_pairs.reserve(def.interface_faces.size());

	        auto maybe_add_cell_pair =
	            [&](FieldId test, FieldId trial, const std::shared_ptr<assembly::AssemblyKernel>& kernel) {
	                if (!kernel || kernel->isVectorOnly()) {
	                    return;
	                }
	                cell_pairs.emplace_back(test, trial);
	            };

	        auto maybe_add_boundary_pair =
	            [&](int marker,
	                FieldId test,
	                FieldId trial,
	                const std::shared_ptr<assembly::AssemblyKernel>& kernel) {
	                if (!kernel || kernel->isVectorOnly()) {
	                    return;
	                }
	                boundary_pairs.emplace_back(marker, test, trial);
	            };

		        auto maybe_add_interior_pair =
		            [&](FieldId test, FieldId trial, const std::shared_ptr<assembly::AssemblyKernel>& kernel) {
		                if (!kernel || kernel->isVectorOnly()) {
		                    return;
		                }
		                interior_pairs.emplace_back(test, trial);
		            };

		        auto maybe_add_interface_pair =
		            [&](int marker,
		                FieldId test,
		                FieldId trial,
		                const std::shared_ptr<assembly::AssemblyKernel>& kernel) {
		                if (!kernel || kernel->isVectorOnly()) {
		                    return;
		                }
		                interface_pairs.emplace_back(marker, test, trial);
		            };

	        for (const auto& term : def.cells) {
	            maybe_add_cell_pair(term.test_field, term.trial_field, term.kernel);
	        }
	        for (const auto& term : def.boundary) {
	            maybe_add_boundary_pair(term.marker, term.test_field, term.trial_field, term.kernel);
	        }
		        for (const auto& term : def.interior) {
		            maybe_add_interior_pair(term.test_field, term.trial_field, term.kernel);
		        }
		        for (const auto& term : def.interface_faces) {
		            maybe_add_interface_pair(term.marker, term.test_field, term.trial_field, term.kernel);
		        }

	        std::sort(cell_pairs.begin(), cell_pairs.end());
	        cell_pairs.erase(std::unique(cell_pairs.begin(), cell_pairs.end()), cell_pairs.end());

	        std::sort(boundary_pairs.begin(), boundary_pairs.end());
	        boundary_pairs.erase(std::unique(boundary_pairs.begin(), boundary_pairs.end()), boundary_pairs.end());

		        std::sort(interior_pairs.begin(), interior_pairs.end());
		        interior_pairs.erase(std::unique(interior_pairs.begin(), interior_pairs.end()), interior_pairs.end());

		        std::sort(interface_pairs.begin(), interface_pairs.end());
		        interface_pairs.erase(std::unique(interface_pairs.begin(), interface_pairs.end()), interface_pairs.end());

		        std::vector<GlobalIndex> row_dofs;
		        std::vector<GlobalIndex> col_dofs;

			        auto add_element_couplings = [&](std::span<const GlobalIndex> rows,
			                                         std::span<const GlobalIndex> cols) {
			            pattern->addElementCouplings(rows, cols);
			            if (!dist_pattern) {
			                return;
			            }

			            if (dist_mode == DistSparsityMode::ContiguousRange) {
			                for (const auto global_row : rows) {
			                    if (owned_range.contains(global_row)) {
			                        for (const auto global_col : cols) {
			                            dist_pattern->addEntry(global_row, global_col);
			                        }
			                        continue;
			                    }

			                    if (!ghost_set.contains(global_row)) {
			                        continue;
			                    }

			                    auto& cols_for_row = ghost_row_cols[global_row];
			                    for (const auto global_col : cols) {
			                        if (relevant_set.contains(global_col)) {
			                            cols_for_row.push_back(global_col);
			                        }
			                    }
			                }
			                return;
			            }

			            FE_THROW_IF(dist_mode != DistSparsityMode::NodalInterleaved || !nodal_map.has_value(),
			                        InvalidStateException,
			                        "FESystem::setup: missing nodal interleaved mapping for distributed sparsity build");

			            const auto& nodal = *nodal_map;
			            const int dof = nodal.dof_per_node;

			            for (const auto global_row_fe : rows) {
			                const auto global_row_fs = nodal.mapFeToFs(global_row_fe);
			                if (global_row_fs == INVALID_GLOBAL_INDEX) {
			                    continue;
			                }

			                if (owned_range.contains(global_row_fs)) {
			                    for (const auto global_col_fe : cols) {
			                        const auto global_col_fs = nodal.mapFeToFs(global_col_fe);
			                        if (global_col_fs == INVALID_GLOBAL_INDEX) {
			                            continue;
			                        }
			                        dist_pattern->addEntry(global_row_fs, global_col_fs);
			                    }
			                    continue;
			                }

			                const int node = static_cast<int>(global_row_fs / dof);
			                if (!nodal.isGhostNode(node)) {
			                    continue;
			                }

			                auto& cols_for_row = ghost_row_cols[global_row_fs];
			                for (const auto global_col_fe : cols) {
			                    const auto global_col_fs = nodal.mapFeToFs(global_col_fe);
			                    if (global_col_fs == INVALID_GLOBAL_INDEX) {
			                        continue;
			                    }
			                    if (nodal.isRelevantDof(global_col_fs)) {
			                        cols_for_row.push_back(global_col_fs);
			                    }
			                }
			            }
			        };

		        auto add_cell_couplings = [&](GlobalIndex cell_id,
		                                      const dofs::DofMap& row_map,
		                                      const dofs::DofMap& col_map,
	                                      GlobalIndex row_offset,
	                                      GlobalIndex col_offset) {
	            auto row_local = row_map.getCellDofs(cell_id);
	            auto col_local = col_map.getCellDofs(cell_id);

	            row_dofs.resize(row_local.size());
	            for (std::size_t i = 0; i < row_local.size(); ++i) {
	                row_dofs[i] = row_local[i] + row_offset;
	            }

	            col_dofs.resize(col_local.size());
	            for (std::size_t j = 0; j < col_local.size(); ++j) {
	                col_dofs[j] = col_local[j] + col_offset;
	            }

		            add_element_couplings(row_dofs, col_dofs);
		        };

	        // Cell terms: all cells participate.
	        for (const auto& [test_field, trial_field] : cell_pairs) {
	            const auto test_idx = static_cast<std::size_t>(test_field);
	            const auto trial_idx = static_cast<std::size_t>(trial_field);

	            FE_THROW_IF(test_field < 0 || test_idx >= field_dof_handlers_.size(), InvalidStateException,
	                        "FESystem::setup: invalid test field in sparsity build");
	            FE_THROW_IF(trial_field < 0 || trial_idx >= field_dof_handlers_.size(), InvalidStateException,
	                        "FESystem::setup: invalid trial field in sparsity build");

            const auto& row_map = field_dof_handlers_[test_idx].getDofMap();
            const auto& col_map = field_dof_handlers_[trial_idx].getDofMap();
	            const auto row_offset = field_dof_offsets_[test_idx];
	            const auto col_offset = field_dof_offsets_[trial_idx];

	            for (GlobalIndex cell = 0; cell < n_cells_sparsity; ++cell) {
	                add_cell_couplings(cell, row_map, col_map, row_offset, col_offset);
	            }
	        }

	        // Boundary terms: only cells adjacent to the requested marker participate.
	        std::vector<GlobalIndex> marker_cells;
	        for (const auto& [marker, test_field, trial_field] : boundary_pairs) {
	            const auto test_idx = static_cast<std::size_t>(test_field);
	            const auto trial_idx = static_cast<std::size_t>(trial_field);

	            FE_THROW_IF(test_field < 0 || test_idx >= field_dof_handlers_.size(), InvalidStateException,
	                        "FESystem::setup: invalid test field in boundary sparsity build");
	            FE_THROW_IF(trial_field < 0 || trial_idx >= field_dof_handlers_.size(), InvalidStateException,
	                        "FESystem::setup: invalid trial field in boundary sparsity build");

	            const auto& row_map = field_dof_handlers_[test_idx].getDofMap();
	            const auto& col_map = field_dof_handlers_[trial_idx].getDofMap();
	            const auto row_offset = field_dof_offsets_[test_idx];
	            const auto col_offset = field_dof_offsets_[trial_idx];

	            marker_cells.clear();
	            meshAccess().forEachBoundaryFace(marker, [&](GlobalIndex /*face_id*/, GlobalIndex cell_id) {
	                marker_cells.push_back(cell_id);
	            });
	            std::sort(marker_cells.begin(), marker_cells.end());
	            marker_cells.erase(std::unique(marker_cells.begin(), marker_cells.end()), marker_cells.end());

	            for (const auto cell_id : marker_cells) {
	                add_cell_couplings(cell_id, row_map, col_map, row_offset, col_offset);
	            }
	        }

	        // Interior face terms (DG): include both self and cross-cell couplings.
	        std::vector<GlobalIndex> row_dofs_minus, row_dofs_plus;
	        std::vector<GlobalIndex> col_dofs_minus, col_dofs_plus;
	        for (const auto& [test_field, trial_field] : interior_pairs) {
	            const auto test_idx = static_cast<std::size_t>(test_field);
	            const auto trial_idx = static_cast<std::size_t>(trial_field);

	            FE_THROW_IF(test_field < 0 || test_idx >= field_dof_handlers_.size(), InvalidStateException,
	                        "FESystem::setup: invalid test field in interior-face sparsity build");
	            FE_THROW_IF(trial_field < 0 || trial_idx >= field_dof_handlers_.size(), InvalidStateException,
	                        "FESystem::setup: invalid trial field in interior-face sparsity build");

	            const auto& row_map = field_dof_handlers_[test_idx].getDofMap();
	            const auto& col_map = field_dof_handlers_[trial_idx].getDofMap();
	            const auto row_offset = field_dof_offsets_[test_idx];
	            const auto col_offset = field_dof_offsets_[trial_idx];

	            meshAccess().forEachInteriorFace(
	                [&](GlobalIndex /*face_id*/, GlobalIndex cell_minus, GlobalIndex cell_plus) {
	                    auto minus_row_local = row_map.getCellDofs(cell_minus);
	                    auto plus_row_local = row_map.getCellDofs(cell_plus);
	                    auto minus_col_local = col_map.getCellDofs(cell_minus);
	                    auto plus_col_local = col_map.getCellDofs(cell_plus);

	                    row_dofs_minus.resize(minus_row_local.size());
	                    for (std::size_t i = 0; i < minus_row_local.size(); ++i) {
	                        row_dofs_minus[i] = minus_row_local[i] + row_offset;
	                    }
	                    row_dofs_plus.resize(plus_row_local.size());
	                    for (std::size_t i = 0; i < plus_row_local.size(); ++i) {
	                        row_dofs_plus[i] = plus_row_local[i] + row_offset;
	                    }

	                    col_dofs_minus.resize(minus_col_local.size());
	                    for (std::size_t j = 0; j < minus_col_local.size(); ++j) {
	                        col_dofs_minus[j] = minus_col_local[j] + col_offset;
	                    }
	                    col_dofs_plus.resize(plus_col_local.size());
	                    for (std::size_t j = 0; j < plus_col_local.size(); ++j) {
	                        col_dofs_plus[j] = plus_col_local[j] + col_offset;
	                    }

		                    add_element_couplings(row_dofs_minus, col_dofs_minus);
		                    add_element_couplings(row_dofs_plus, col_dofs_plus);
		                    add_element_couplings(row_dofs_minus, col_dofs_plus);
		                    add_element_couplings(row_dofs_plus, col_dofs_minus);
		                });
		        }

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
		        // Interface face terms (InterfaceMesh subset): include both self and cross-cell couplings.
		        std::vector<GlobalIndex> iface_row_dofs_minus, iface_row_dofs_plus;
		        std::vector<GlobalIndex> iface_col_dofs_minus, iface_col_dofs_plus;
		        for (const auto& [marker, test_field, trial_field] : interface_pairs) {
		            const auto test_idx = static_cast<std::size_t>(test_field);
		            const auto trial_idx = static_cast<std::size_t>(trial_field);

		            FE_THROW_IF(test_field < 0 || test_idx >= field_dof_handlers_.size(), InvalidStateException,
		                        "FESystem::setup: invalid test field in interface-face sparsity build");
		            FE_THROW_IF(trial_field < 0 || trial_idx >= field_dof_handlers_.size(), InvalidStateException,
		                        "FESystem::setup: invalid trial field in interface-face sparsity build");

		            const auto& row_map = field_dof_handlers_[test_idx].getDofMap();
		            const auto& col_map = field_dof_handlers_[trial_idx].getDofMap();
		            const auto row_offset = field_dof_offsets_[test_idx];
		            const auto col_offset = field_dof_offsets_[trial_idx];

		            auto add_interface_mesh_couplings = [&](const svmp::InterfaceMesh& iface) {
		                for (std::size_t lf = 0; lf < iface.n_faces(); ++lf) {
		                    const auto local_face = static_cast<svmp::index_t>(lf);
		                    const GlobalIndex cell_minus = static_cast<GlobalIndex>(iface.volume_cell_minus(local_face));
		                    const GlobalIndex cell_plus = static_cast<GlobalIndex>(iface.volume_cell_plus(local_face));
		                    if (cell_minus == INVALID_GLOBAL_INDEX || cell_plus == INVALID_GLOBAL_INDEX) {
		                        continue; // One-sided interface faces: no cross-cell couplings.
		                    }

		                    auto minus_row_local = row_map.getCellDofs(cell_minus);
		                    auto plus_row_local = row_map.getCellDofs(cell_plus);
		                    auto minus_col_local = col_map.getCellDofs(cell_minus);
		                    auto plus_col_local = col_map.getCellDofs(cell_plus);

		                    iface_row_dofs_minus.resize(minus_row_local.size());
		                    for (std::size_t i = 0; i < minus_row_local.size(); ++i) {
		                        iface_row_dofs_minus[i] = minus_row_local[i] + row_offset;
		                    }
		                    iface_row_dofs_plus.resize(plus_row_local.size());
		                    for (std::size_t i = 0; i < plus_row_local.size(); ++i) {
		                        iface_row_dofs_plus[i] = plus_row_local[i] + row_offset;
		                    }

		                    iface_col_dofs_minus.resize(minus_col_local.size());
		                    for (std::size_t j = 0; j < minus_col_local.size(); ++j) {
		                        iface_col_dofs_minus[j] = minus_col_local[j] + col_offset;
		                    }
		                    iface_col_dofs_plus.resize(plus_col_local.size());
		                    for (std::size_t j = 0; j < plus_col_local.size(); ++j) {
		                        iface_col_dofs_plus[j] = plus_col_local[j] + col_offset;
		                    }

		                    add_element_couplings(iface_row_dofs_minus, iface_col_dofs_minus);
		                    add_element_couplings(iface_row_dofs_plus, iface_col_dofs_plus);
		                    add_element_couplings(iface_row_dofs_minus, iface_col_dofs_plus);
		                    add_element_couplings(iface_row_dofs_plus, iface_col_dofs_minus);
		                }
		            };

		            if (marker < 0) {
		                FE_THROW_IF(interface_meshes_.empty(), InvalidStateException,
		                            "FESystem::setup: interface-face kernels registered for all markers, but no InterfaceMesh was set");
		                for (const auto& kv : interface_meshes_) {
		                    if (!kv.second) continue;
		                    add_interface_mesh_couplings(*kv.second);
		                }
		            } else {
		                auto it = interface_meshes_.find(marker);
		                FE_THROW_IF(it == interface_meshes_.end() || !it->second, InvalidStateException,
		                            "FESystem::setup: missing InterfaceMesh for interface marker " + std::to_string(marker));
		                add_interface_mesh_couplings(*it->second);
		            }
		        }
#endif

	        // Global terms: allow kernels to conservatively augment sparsity.
	        for (const auto& kernel : def.global) {
	            if (kernel) {
	                kernel->addSparsityCouplings(*this, *pattern);
	            }
        }

	        // Coupled boundary-condition Jacobian: low-rank outer products introduce dense
	        // couplings between boundary-local DOFs on the BC marker and the DOFs that
	        // influence the referenced BoundaryFunctionals. Conservatively preallocate
	        // these couplings so strict backends (e.g., Trilinos) can accept insertions.
	        if (coupled_boundary_) {
	            const auto regs = coupled_boundary_->registeredBoundaryFunctionals();
	            if (!regs.empty() && !def.boundary.empty()) {
	                const auto primary_field = coupled_boundary_->primaryField();
	                const auto primary_idx = static_cast<std::size_t>(primary_field);
	                if (primary_field >= 0 && primary_idx < field_dof_handlers_.size()) {
	                    const auto& primary_map = field_dof_handlers_[primary_idx].getDofMap();
	                    const auto primary_offset = field_dof_offsets_[primary_idx];

	                    std::unordered_map<int, std::vector<GlobalIndex>> dofs_by_marker_primary;
	                    dofs_by_marker_primary.reserve(regs.size());

	                    auto collect_marker_dofs = [&](int marker,
	                                                  const dofs::DofMap& map,
	                                                  GlobalIndex offset,
	                                                  std::vector<GlobalIndex>& out) {
	                        out.clear();
	                        mesh_access_->forEachBoundaryFace(marker, [&](GlobalIndex /*face_id*/, GlobalIndex cell_id) {
	                            const auto cell_dofs = map.getCellDofs(cell_id);
	                            out.insert(out.end(), cell_dofs.begin(), cell_dofs.end());
	                        });
	                        for (auto& v : out) v += offset;
	                        std::sort(out.begin(), out.end());
	                        out.erase(std::unique(out.begin(), out.end()), out.end());
	                    };

	                    for (const auto& bf : regs) {
	                        const int marker = bf.def.boundary_marker;
	                        if (marker < 0) continue;
	                        if (dofs_by_marker_primary.count(marker) != 0u) continue;
	                        std::vector<GlobalIndex> tmp;
	                        collect_marker_dofs(marker, primary_map, primary_offset, tmp);
	                        dofs_by_marker_primary.emplace(marker, std::move(tmp));
	                    }

	                    std::unordered_map<std::uint64_t, std::vector<GlobalIndex>> row_dofs_cache;
	                    row_dofs_cache.reserve(def.boundary.size());

	                    auto key_of = [](int marker, FieldId test_field) -> std::uint64_t {
	                        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(marker)) << 32) |
	                               static_cast<std::uint32_t>(test_field);
	                    };

	                    for (const auto& term : def.boundary) {
	                        const int marker = term.marker;
	                        if (marker < 0) continue;

	                        const auto k = key_of(marker, term.test_field);
	                        if (row_dofs_cache.count(k) == 0u) {
	                            const auto test_idx = static_cast<std::size_t>(term.test_field);
	                            if (term.test_field < 0 || test_idx >= field_dof_handlers_.size()) {
	                                continue;
	                            }
	                            const auto& row_map = field_dof_handlers_[test_idx].getDofMap();
	                            const auto row_offset = field_dof_offsets_[test_idx];
	                            std::vector<GlobalIndex> tmp;
	                            collect_marker_dofs(marker, row_map, row_offset, tmp);
	                            row_dofs_cache.emplace(k, std::move(tmp));
	                        }

	                        const auto& row_dofs = row_dofs_cache.at(k);
	                        if (row_dofs.empty()) continue;

	                        for (const auto& bf : regs) {
	                            const int q_marker = bf.def.boundary_marker;
	                            auto it = dofs_by_marker_primary.find(q_marker);
	                            if (it == dofs_by_marker_primary.end()) continue;
	                            const auto& col_dofs = it->second;
	                            if (col_dofs.empty()) continue;
	                            pattern->addElementCouplings(row_dofs, col_dofs);
	                        }
	                    }
	                }
	            }
	        }

		        if (opts.sparsity_options.ensure_diagonal) {
		            pattern->ensureDiagonal();
		        }
	        if (opts.sparsity_options.ensure_non_empty_rows) {
	            pattern->ensureNonEmptyRows();
	        }
	        if (dist_pattern) {
	            if (opts.sparsity_options.ensure_diagonal) {
	                dist_pattern->ensureDiagonal();
	            }
	            if (opts.sparsity_options.ensure_non_empty_rows) {
	                dist_pattern->ensureNonEmptyRows();
	            }
	        }
	
		        if (opts.use_constraints_in_assembly && !affine_constraints_.empty()) {
		            auto query = std::make_shared<AffineConstraintsQuery>(affine_constraints_);
		            sparsity::ConstraintSparsityAugmenter augmenter(std::move(query));
		            augmenter.augment(*pattern, sparsity::AugmentationMode::EliminationFill);
		            if (dist_pattern) {
		                if (dist_mode == DistSparsityMode::NodalInterleaved) {
		                    FE_THROW_IF(!nodal_map.has_value(), InvalidStateException,
		                                "FESystem::setup: missing nodal mapping for constraint sparsity augmentation");
		                    auto dist_query = std::make_shared<PermutedAffineConstraintsQuery>(
		                        affine_constraints_,
		                        std::span<const GlobalIndex>(nodal_map->fe_to_fs),
		                        std::span<const GlobalIndex>(nodal_map->fs_to_fe));
		                    sparsity::ConstraintSparsityAugmenter dist_augmenter(std::move(dist_query));
		                    dist_augmenter.augment(*dist_pattern, sparsity::AugmentationMode::EliminationFill);
		                } else {
		                    augmenter.augment(*dist_pattern, sparsity::AugmentationMode::EliminationFill);
		                }
		            }
		        }
	
	        if (opts.sparsity_options.ensure_diagonal) {
	            pattern->ensureDiagonal();
	        }
	        if (opts.sparsity_options.ensure_non_empty_rows) {
	            pattern->ensureNonEmptyRows();
	        }
	        if (dist_pattern) {
	            if (opts.sparsity_options.ensure_diagonal) {
	                dist_pattern->ensureDiagonal();
	            }
	            if (opts.sparsity_options.ensure_non_empty_rows) {
	                dist_pattern->ensureNonEmptyRows();
	            }
	        }
	
		        pattern->finalize();
		        const auto* full_pattern = pattern.get();
		        sparsity_by_op_.emplace(tag, std::move(pattern));

		        if (dist_pattern) {
		            dist_pattern->finalize();

		            // Store optional ghost-row sparsity using the locally relevant (owned+ghost) overlap.
		            // This is required by overlap-style MPI backends (e.g., FSILS) so all column nodes
		            // referenced by owned rows are present locally.
		            if (dist_mode == DistSparsityMode::NodalInterleaved) {
		                FE_THROW_IF(!nodal_map.has_value(), InvalidStateException,
		                            "FESystem::setup: missing nodal mapping for ghost-row sparsity storage");
		                const auto& nodal = *nodal_map;
		                const int dof = nodal.dof_per_node;

		                std::vector<GlobalIndex> ghost_row_map;
		                ghost_row_map.reserve(nodal.ghost_nodes.size() * static_cast<std::size_t>(std::max(1, dof)));
		                for (const int node : nodal.ghost_nodes) {
		                    for (int c = 0; c < dof; ++c) {
		                        ghost_row_map.push_back(static_cast<GlobalIndex>(node) * dof + c);
		                    }
		                }

		                if (!ghost_row_map.empty()) {
		                    std::vector<GlobalIndex> ghost_row_ptr;
		                    std::vector<GlobalIndex> ghost_row_cols_flat;
		                    ghost_row_ptr.reserve(ghost_row_map.size() + 1);
		                    ghost_row_ptr.push_back(0);

			                    for (const auto row_fs : ghost_row_map) {
			                        std::vector<GlobalIndex> cols_vec;
			                        cols_vec.reserve(32);

			                        if (full_pattern != nullptr && row_fs >= 0 && row_fs < n_total_dofs) {
			                            const auto row_fe = nodal.fs_to_fe[static_cast<std::size_t>(row_fs)];
			                            if (row_fe >= 0 && row_fe < n_total_dofs) {
			                                const auto cols_fe = full_pattern->getRowSpan(row_fe);
			                                for (const auto col_fe : cols_fe) {
			                                    const auto col_fs = nodal.mapFeToFs(col_fe);
			                                    if (col_fs == INVALID_GLOBAL_INDEX) {
			                                        continue;
			                                    }
			                                    if (nodal.isRelevantDof(col_fs)) {
			                                        cols_vec.push_back(col_fs);
			                                    }
			                                }
			                            }
			                        }

			                        cols_vec.push_back(row_fs); // ensure diagonal

			                        cols_vec.erase(std::remove_if(cols_vec.begin(),
			                                                     cols_vec.end(),
			                                                     [&](GlobalIndex col) {
			                                                         return (col < 0) || (col >= n_total_dofs) || !nodal.isRelevantDof(col);
			                                                     }),
			                                       cols_vec.end());

		                        std::sort(cols_vec.begin(), cols_vec.end());
		                        cols_vec.erase(std::unique(cols_vec.begin(), cols_vec.end()), cols_vec.end());

			                        ghost_row_cols_flat.insert(ghost_row_cols_flat.end(), cols_vec.begin(), cols_vec.end());
			                        ghost_row_ptr.push_back(static_cast<GlobalIndex>(ghost_row_cols_flat.size()));
			                    }

		                    dist_pattern->setGhostRows(std::move(ghost_row_map),
		                                              std::move(ghost_row_ptr),
		                                              std::move(ghost_row_cols_flat));
		                } else {
		                    dist_pattern->clearGhostRows();
		                }
		            } else {
		                auto ghost_rows_all = ghost_set.toVector();
		                std::vector<GlobalIndex> ghost_row_map;
		                ghost_row_map.reserve(ghost_rows_all.size());
		                for (const auto row : ghost_rows_all) {
		                    if (row < 0 || row >= n_total_dofs) {
		                        continue;
		                    }
		                    if (owned_range.contains(row)) {
		                        continue;
		                    }
		                    ghost_row_map.push_back(row);
		                }

		                if (!ghost_row_map.empty()) {
		                    // ghost_set.toVector() is already sorted/unique; keep it that way after filtering.
		                    std::vector<GlobalIndex> ghost_row_ptr;
		                    std::vector<GlobalIndex> ghost_row_cols_flat;
		                    ghost_row_ptr.reserve(ghost_row_map.size() + 1);
		                    ghost_row_ptr.push_back(0);

			                    for (const auto row : ghost_row_map) {
			                        std::vector<GlobalIndex> cols_vec;
			                        cols_vec.reserve(32);

			                        if (full_pattern != nullptr && row >= 0 && row < n_total_dofs) {
			                            const auto cols = full_pattern->getRowSpan(row);
			                            for (const auto col : cols) {
			                                if (relevant_set.contains(col)) {
			                                    cols_vec.push_back(col);
			                                }
			                            }
			                        }

			                        cols_vec.push_back(row); // ensure diagonal

		                        // Only store columns that are locally present (owned+ghost) so overlap backends can map them.
		                        cols_vec.erase(std::remove_if(cols_vec.begin(),
		                                                     cols_vec.end(),
		                                                     [&](GlobalIndex col) {
		                                                         return (col < 0) || (col >= n_total_dofs) || !relevant_set.contains(col);
		                                                     }),
		                                       cols_vec.end());

		                        std::sort(cols_vec.begin(), cols_vec.end());
		                        cols_vec.erase(std::unique(cols_vec.begin(), cols_vec.end()), cols_vec.end());

		                        ghost_row_cols_flat.insert(ghost_row_cols_flat.end(), cols_vec.begin(), cols_vec.end());
		                        ghost_row_ptr.push_back(static_cast<GlobalIndex>(ghost_row_cols_flat.size()));
		                    }

		                    dist_pattern->setGhostRows(std::move(ghost_row_map),
		                                              std::move(ghost_row_ptr),
		                                              std::move(ghost_row_cols_flat));
		                } else {
		                    dist_pattern->clearGhostRows();
		                }
		            }

		            distributed_sparsity_by_op_.emplace(tag, std::move(dist_pattern));
		        }
			    }

    // Persist a node-interleaved backend permutation for overlap backends (FSILS). This is needed
    // in MPI even when we used Natural indexing for distributed sparsity (owner-contiguous FE IDs).
    if (nodal_map.has_value() || backend_map.has_value()) {
        auto& map = nodal_map.has_value() ? *nodal_map : *backend_map;
        auto perm = std::make_shared<backends::DofPermutation>();
        perm->forward = std::move(map.fe_to_fs);
        perm->inverse = std::move(map.fs_to_fe);
        dof_permutation_ = std::move(perm);
    } else {
        dof_permutation_.reset();
    }

    // ---------------------------------------------------------------------
    // Per-cell material state storage (optional; for RequiredData::MaterialState)
    // ---------------------------------------------------------------------
    material_state_provider_.reset();
    {
        std::vector<GlobalIndex> boundary_faces;
        meshAccess().forEachBoundaryFace(/*marker=*/-1, [&](GlobalIndex face_id, GlobalIndex /*cell_id*/) {
            boundary_faces.push_back(face_id);
        });

        std::vector<GlobalIndex> interior_faces;
        meshAccess().forEachInteriorFace([&](GlobalIndex face_id, GlobalIndex /*cell_minus*/, GlobalIndex /*cell_plus*/) {
            interior_faces.push_back(face_id);
        });

        auto provider = std::make_unique<MaterialStateProvider>(meshAccess().numCells(),
                                                                std::move(boundary_faces),
                                                                std::move(interior_faces));
        bool any = false;

        for (const auto& tag : operator_registry_.list()) {
            const auto& def = operator_registry_.get(tag);

            for (const auto& term : def.cells) {
                if (!term.kernel) continue;

                const auto required = term.kernel->getRequiredData();
                if (!assembly::hasFlag(required, assembly::RequiredData::MaterialState)) {
                    continue;
                }

                const auto spec = term.kernel->materialStateSpec();
                FE_THROW_IF(spec.bytes_per_qpt == 0u, InvalidArgumentException,
                            "FESystem::setup: kernel requests MaterialState but bytes_per_qpt == 0");

                const auto& test_field = field_registry_.get(term.test_field);
                FE_CHECK_NOT_NULL(test_field.space.get(), "FESystem::setup: test_field.space");
                const auto max_qpts = maxCellQuadraturePoints(meshAccess(), *test_field.space);
                FE_THROW_IF(max_qpts <= 0, InvalidStateException,
                            "FESystem::setup: failed to determine max quadrature points for MaterialState allocation");

                provider->addKernel(*term.kernel, spec, max_qpts);
                any = true;
            }

            for (const auto& term : def.boundary) {
                if (!term.kernel) continue;
                const auto required = term.kernel->getRequiredData();
                if (!assembly::hasFlag(required, assembly::RequiredData::MaterialState)) {
                    continue;
                }

                const auto spec = term.kernel->materialStateSpec();
                FE_THROW_IF(spec.bytes_per_qpt == 0u, InvalidArgumentException,
                            "FESystem::setup: kernel requests MaterialState but bytes_per_qpt == 0 (boundary)");

                const auto& test_field = field_registry_.get(term.test_field);
                const auto& trial_field = field_registry_.get(term.trial_field);
                FE_CHECK_NOT_NULL(test_field.space.get(), "FESystem::setup: boundary test_field.space");
                FE_CHECK_NOT_NULL(trial_field.space.get(), "FESystem::setup: boundary trial_field.space");

                const auto max_qpts = maxBoundaryFaceQuadraturePoints(meshAccess(), *test_field.space, *trial_field.space);
                if (max_qpts > 0) {
                    provider->addKernel(*term.kernel, spec,
                                        /*max_cell_qpts=*/0,
                                        /*max_boundary_face_qpts=*/max_qpts,
                                        /*max_interior_face_qpts=*/0);
                    any = true;
                }
            }
            for (const auto& term : def.interior) {
                if (!term.kernel) continue;
                const auto required = term.kernel->getRequiredData();
                if (!assembly::hasFlag(required, assembly::RequiredData::MaterialState)) {
                    continue;
                }

                const auto spec = term.kernel->materialStateSpec();
                FE_THROW_IF(spec.bytes_per_qpt == 0u, InvalidArgumentException,
                            "FESystem::setup: kernel requests MaterialState but bytes_per_qpt == 0 (interior)");

                const auto& test_field = field_registry_.get(term.test_field);
                const auto& trial_field = field_registry_.get(term.trial_field);
                FE_CHECK_NOT_NULL(test_field.space.get(), "FESystem::setup: interior test_field.space");
                FE_CHECK_NOT_NULL(trial_field.space.get(), "FESystem::setup: interior trial_field.space");

                const auto max_qpts = maxInteriorFaceQuadraturePoints(meshAccess(), *test_field.space, *trial_field.space);
                if (max_qpts > 0) {
                    provider->addKernel(*term.kernel, spec,
                                        /*max_cell_qpts=*/0,
                                        /*max_boundary_face_qpts=*/0,
                                        /*max_interior_face_qpts=*/max_qpts);
                    any = true;
                }
            }
        }

        if (any) {
            material_state_provider_ = std::move(provider);
        }
    }

    // ---------------------------------------------------------------------
    // Global-kernel persistent state storage (optional; for GlobalStateSpec)
    // ---------------------------------------------------------------------
    global_kernel_state_provider_.reset();
    {
        std::vector<GlobalIndex> boundary_faces;
        meshAccess().forEachBoundaryFace(/*marker=*/-1, [&](GlobalIndex face_id, GlobalIndex /*cell_id*/) {
            boundary_faces.push_back(face_id);
        });

        std::vector<GlobalIndex> interior_faces;
        meshAccess().forEachInteriorFace([&](GlobalIndex face_id, GlobalIndex /*cell_minus*/, GlobalIndex /*cell_plus*/) {
            interior_faces.push_back(face_id);
        });

        auto provider = std::make_unique<GlobalKernelStateProvider>(meshAccess().numCells(),
                                                                    std::move(boundary_faces),
                                                                    std::move(interior_faces));

        bool any = false;
        for (const auto& tag : operator_registry_.list()) {
            const auto& def = operator_registry_.get(tag);
            for (const auto& kernel : def.global) {
                if (!kernel) continue;
                const auto spec = kernel->globalStateSpec();
                if (spec.empty()) continue;

                provider->addKernel(*kernel, spec);
                any = true;
            }
        }

        if (any) {
            global_kernel_state_provider_ = std::move(provider);
        }
    }

    // ---------------------------------------------------------------------
    // Parameter requirements (optional)
    // ---------------------------------------------------------------------
    parameter_registry_.clear();
    {
        for (const auto& tag : operator_registry_.list()) {
            const auto& def = operator_registry_.get(tag);

            for (const auto& term : def.cells) {
                if (!term.kernel) continue;
                parameter_registry_.addAll(term.kernel->parameterSpecs(), term.kernel->name());
            }
            for (const auto& term : def.boundary) {
                if (!term.kernel) continue;
                parameter_registry_.addAll(term.kernel->parameterSpecs(), term.kernel->name());
            }
            for (const auto& term : def.interior) {
                if (!term.kernel) continue;
                parameter_registry_.addAll(term.kernel->parameterSpecs(), term.kernel->name());
            }
            for (const auto& kernel : def.global) {
                if (!kernel) continue;
                parameter_registry_.addAll(kernel->parameterSpecs(), kernel->name());
            }
        }

        if (coupled_boundary_) {
            parameter_registry_.addAll(coupled_boundary_->parameterSpecs(), "CoupledBoundaryManager");
        }
    }

    // ---------------------------------------------------------------------
    // Resolve FE/Forms constitutive calls for JIT-fast mode (setup-time)
    // ---------------------------------------------------------------------
    {
        for (const auto& tag : operator_registry_.list()) {
            const auto& def = operator_registry_.get(tag);

            for (const auto& term : def.cells) {
                if (term.kernel) {
                    term.kernel->resolveInlinableConstitutives();
                }
            }
            for (const auto& term : def.boundary) {
                if (term.kernel) {
                    term.kernel->resolveInlinableConstitutives();
                }
            }
            for (const auto& term : def.interior) {
                if (term.kernel) {
                    term.kernel->resolveInlinableConstitutives();
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Resolve FE/Forms ParameterSymbol -> ParameterRef(slot) (setup-time)
    // ---------------------------------------------------------------------
    {
        const auto slot_of_real_param = [&](std::string_view key) -> std::optional<std::uint32_t> {
            return parameter_registry_.slotOf(key);
        };

        for (const auto& tag : operator_registry_.list()) {
            const auto& def = operator_registry_.get(tag);

            for (const auto& term : def.cells) {
                if (term.kernel) {
                    term.kernel->resolveParameterSlots(slot_of_real_param);
                }
            }
            for (const auto& term : def.boundary) {
                if (term.kernel) {
                    term.kernel->resolveParameterSlots(slot_of_real_param);
                }
            }
            for (const auto& term : def.interior) {
                if (term.kernel) {
                    term.kernel->resolveParameterSlots(slot_of_real_param);
                }
            }
            for (const auto& term : def.interface_faces) {
                if (term.kernel) {
                    term.kernel->resolveParameterSlots(slot_of_real_param);
                }
            }
        }

        if (coupled_boundary_) {
            coupled_boundary_->resolveParameterSlots(slot_of_real_param);
        }
    }

    // ---------------------------------------------------------------------
    // Assembler configuration
    // ---------------------------------------------------------------------
    assembly::FormCharacteristics form_chars{};
    {
        for (const auto& tag : operator_registry_.list()) {
            const auto& def = operator_registry_.get(tag);

            form_chars.has_cell_terms = form_chars.has_cell_terms || !def.cells.empty();
            form_chars.has_boundary_terms = form_chars.has_boundary_terms || !def.boundary.empty();
            form_chars.has_interior_face_terms = form_chars.has_interior_face_terms || !def.interior.empty();
            form_chars.has_interface_face_terms = form_chars.has_interface_face_terms || !def.interface_faces.empty();
            form_chars.has_global_terms = form_chars.has_global_terms || !def.global.empty();

            const auto mergeKernelMeta = [&](const std::shared_ptr<assembly::AssemblyKernel>& k) {
                if (!k) return;
                form_chars.required_data |= k->getRequiredData();
                form_chars.max_time_derivative_order =
                    std::max(form_chars.max_time_derivative_order, k->maxTemporalDerivativeOrder());
                form_chars.has_field_requirements = form_chars.has_field_requirements || !k->fieldRequirements().empty();
                form_chars.has_parameter_specs = form_chars.has_parameter_specs || !k->parameterSpecs().empty();
            };

            for (const auto& term : def.cells) {
                mergeKernelMeta(term.kernel);
            }
            for (const auto& term : def.boundary) {
                mergeKernelMeta(term.kernel);
            }
            for (const auto& term : def.interior) {
                mergeKernelMeta(term.kernel);
            }
            for (const auto& term : def.interface_faces) {
                mergeKernelMeta(term.kernel);
            }
            for (const auto& k : def.global) {
                if (!k) continue;
                form_chars.has_parameter_specs = form_chars.has_parameter_specs || !k->parameterSpecs().empty();
            }
        }
    }

    assembly::SystemCharacteristics sys_chars{};
    sys_chars.num_fields = field_registry_.size();
    sys_chars.num_cells = meshAccess().numCells();
    sys_chars.dimension = meshAccess().dimension();
    sys_chars.num_dofs_total = dof_handler_.getDofMap().getNumDofs();
    sys_chars.max_dofs_per_cell = dof_handler_.getDofMap().getMaxDofsPerCell();

    for (const auto& rec : field_registry_.records()) {
        if (!rec.space) continue;
        sys_chars.max_polynomial_order = std::max(sys_chars.max_polynomial_order, rec.space->polynomial_order());
    }

    // Resolve thread count for reporting/heuristics (0 means "auto").
    sys_chars.num_threads = opts.assembly_options.num_threads;
    if (sys_chars.num_threads <= 0) {
        const auto hw = std::max(1u, std::thread::hardware_concurrency());
        sys_chars.num_threads = static_cast<int>(hw);
    }

    sys_chars.mpi_world_size = 1;
#if FE_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        MPI_Comm_size(MPI_COMM_WORLD, &sys_chars.mpi_world_size);
    }
#endif

    assembler_selection_report_.clear();
    assembler_ = assembly::createAssembler(opts.assembly_options, opts.assembler_name,
                                           form_chars, sys_chars, &assembler_selection_report_);
    FE_CHECK_NOT_NULL(assembler_.get(), "FESystem::setup: assembler");
    assembler_->setDofHandler(dof_handler_);

    if (opts.use_constraints_in_assembly) {
        assembler_->setConstraints(&affine_constraints_);
    } else {
        assembler_->setConstraints(nullptr);
    }

    assembler_->setMaterialStateProvider(material_state_provider_.get());
    assembler_->setOptions(opts.assembly_options);
    assembler_->initialize();

    // ---------------------------------------------------------------------
    // Optional: Auto-register matrix-free operators (explicit opt-in)
    // ---------------------------------------------------------------------
    if (opts.auto_register_matrix_free) {
        FE_CHECK_NOT_NULL(operator_backends_.get(), "FESystem::setup: operator_backends");
        FE_THROW_IF(field_registry_.size() != 1u, NotImplementedException,
                    "FESystem::setup: auto_register_matrix_free currently requires a single-field system");

        std::size_t registered = 0;
        for (const auto& tag : operator_registry_.list()) {
            if (operator_backends_->hasMatrixFree(tag)) {
                continue; // Respect explicit user registration.
            }

            const auto& def = operator_registry_.get(tag);
            if (def.cells.empty()) continue;
            if (!def.boundary.empty() || !def.interior.empty() || !def.global.empty()) {
                continue; // Cell-only operators only (initial conservative scope).
            }

            // Build a (possibly composite) cell kernel for this operator.
            std::shared_ptr<assembly::AssemblyKernel> kernel_to_wrap;
            if (def.cells.size() == 1u) {
                kernel_to_wrap = def.cells.front().kernel;
            } else {
                auto composite = std::make_shared<assembly::CompositeKernel>();
                for (const auto& term : def.cells) {
                    if (!term.kernel) continue;
                    composite->addKernel(term.kernel);
                }
                kernel_to_wrap = std::move(composite);
            }

            if (!kernel_to_wrap) continue;

            // Conservative eligibility: linear, steady, cell-only.
            const auto required = kernel_to_wrap->getRequiredData();
            if (assembly::hasFlag(required, assembly::RequiredData::SolutionCoefficients) ||
                assembly::hasFlag(required, assembly::RequiredData::SolutionValues) ||
                assembly::hasFlag(required, assembly::RequiredData::SolutionGradients) ||
                assembly::hasFlag(required, assembly::RequiredData::SolutionHessians) ||
                assembly::hasFlag(required, assembly::RequiredData::SolutionLaplacians) ||
                assembly::hasFlag(required, assembly::RequiredData::MaterialState) ||
                kernel_to_wrap->maxTemporalDerivativeOrder() > 0) {
                continue;
            }

            auto mf_unique = assembly::wrapAsMatrixFreeKernel(kernel_to_wrap);
            std::shared_ptr<assembly::IMatrixFreeKernel> mf_kernel = std::move(mf_unique);
            operator_backends_->registerMatrixFree(tag, std::move(mf_kernel));
            ++registered;
        }

        if (registered > 0u) {
            if (!assembler_selection_report_.empty()) {
                assembler_selection_report_.append("\n");
            }
            assembler_selection_report_.append("Auto-registered matrix-free operators: ");
            assembler_selection_report_.append(std::to_string(registered));
        }
    }

    is_setup_ = true;
}

} // namespace systems
} // namespace FE
} // namespace svmp
