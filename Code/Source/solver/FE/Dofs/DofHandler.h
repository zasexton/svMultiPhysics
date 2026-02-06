/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_DOFS_DOFHANDLER_H
#define SVMP_FE_DOFS_DOFHANDLER_H

/**
 * @file DofHandler.h
 * @brief High-level DOF management interface
 *
 * The DofHandler provides the primary interface for distributing DOFs
 * on a mesh given a function space. It coordinates:
 *  - DOF distribution based on element type and polynomial order
 *  - Canonical ordering on shared entities for consistency
 *  - Parallel ownership assignment
 *  - Index set construction for backends
 *
 * This class is MESH-LIBRARY-INDEPENDENT. It accepts topology data through
 * the MeshTopologyInfo struct, which can be populated from any mesh source.
 * Convenience overloads for MeshBase/Mesh are provided via
 * conditional compilation when the Mesh library is available.
 */

#include "DofMap.h"
#include "DofIndexSet.h"
#include "Core/Types.h"
#include "Core/FEConfig.h"
#include "Core/FEException.h"
#include "Spaces/OrientationManager.h"

#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <span>

#if FE_HAS_MPI
#  include <mpi.h>
#endif

// Forward declarations for Mesh types (avoid hard dependency)
namespace svmp {
    class MeshBase;
    class DistributedMesh;
    // Phase 5 (UNIFY_MESH): prefer the unified runtime mesh type name.
    // In the Mesh library, `Mesh` is currently an alias of `DistributedMesh`.
    using Mesh = DistributedMesh;
}

namespace svmp {
namespace FE {

// Forward declarations
namespace spaces {
    class FunctionSpace;
}

namespace dofs {

// Forward declarations
class EntityDofMap;
class GhostDofManager;

// Type alias for global IDs (mesh-library independent)
using gid_t = MeshGlobalId;

/**
 * @brief DOF numbering strategies
 */
enum class DofNumberingStrategy : std::uint8_t {
    Sequential,     ///< Number DOFs in mesh traversal order
    Interleaved,    ///< Interleave components for vector fields
    Block,          ///< Block all DOFs of same component together
    Hierarchical,   ///< Hierarchy: vertex DOFs, edge DOFs, face DOFs, cell DOFs
    Morton,         ///< Spatially reorder DOFs via Morton (Z-order) curve
    Hilbert         ///< Spatially reorder DOFs via Hilbert curve
};

enum class SpatialCurveType : std::uint8_t {
    Morton,
    Hilbert
};

/**
 * @brief Ownership assignment strategies for parallel DOFs
 */
enum class OwnershipStrategy : std::uint8_t {
    LowestRank,     ///< Lowest rank touching entity owns DOF
    HighestRank,    ///< Highest rank touching entity owns DOF
    CellOwner,      ///< Cell owner owns all cell's DOFs
    VertexGID       ///< Owner determined by vertex global ID hash
};

/**
 * @brief Global numbering modes for distributed DOF IDs
 *
 * Controls how global DOF IDs are assigned in parallel. This is independent of
 * the renumbering strategy (DofNumberingStrategy) applied after distribution.
 */
enum class GlobalNumberingMode : std::uint8_t {
    OwnerContiguous,  ///< Contiguous ID ranges per owner rank (default; scalable)
    GlobalIds,        ///< Process-count independent IDs derived from global entity IDs (may be sparse)
    DenseGlobalIds    ///< Process-count independent dense IDs via distributed compaction (contiguous 0..N-1; deterministic; auto-selects key-ordered vs hash-bucket ordering)
};

/**
 * @brief How DofHandler treats missing topology tables
 *
 * - DeriveMissing: DofHandler will derive missing edge/face connectivity from cell->vertex connectivity.
 * - RequireComplete: DofHandler will throw if required connectivity is missing (Mesh-authoritative mode).
 */
enum class TopologyCompletion : std::uint8_t {
    DeriveMissing,
    RequireComplete
};

/**
 * @brief Options for DOF distribution
 */
struct DofDistributionOptions {
    DofNumberingStrategy numbering{DofNumberingStrategy::Sequential};
    bool enable_spatial_locality_ordering{true};
    SpatialCurveType spatial_curve{SpatialCurveType::Morton};
    OwnershipStrategy ownership{OwnershipStrategy::LowestRank};
    GlobalNumberingMode global_numbering{GlobalNumberingMode::OwnerContiguous};
    TopologyCompletion topology_completion{TopologyCompletion::DeriveMissing};
    bool use_canonical_ordering{true};  ///< Use min-vertex-GID ordering on shared entities
    int my_rank{0};                      ///< MPI rank for parallel
    int world_size{1};                   ///< Total MPI ranks
#if FE_HAS_MPI
    MPI_Comm mpi_comm{MPI_COMM_WORLD};   ///< MPI communicator (ignored in serial)
#endif
    bool validate_parallel{false};       ///< Extra cross-rank validation (debug-heavy)
    bool reproducible_across_communicators{false}; ///< Stable numbering/ownership under MPI rank relabeling
    bool no_global_collectives{false};   ///< Avoid global MPI collectives (uses point-to-point reductions/scans)
};

/**
 * @brief Mesh topology information for DOF distribution (mesh-library-independent)
 *
 * This struct carries all the topology information needed for DOF distribution
 * without requiring any dependency on a specific mesh library. It can be populated
 * from MeshBase, MFEM meshes, deal.II meshes, or any other mesh source.
 *
 * For CG (continuous Galerkin) spaces, vertex sharing is determined by comparing
 * vertex global IDs (gids). Two cells share a DOF at a vertex if they share
 * the same vertex GID.
 *
 * Edge and face connectivity is optional and only needed for higher-order spaces.
 */
struct MeshTopologyInfo {
    // Basic counts
    GlobalIndex n_cells{0};           ///< Number of cells (elements)
    GlobalIndex n_vertices{0};        ///< Number of unique vertices
    GlobalIndex n_edges{0};           ///< Number of unique edges (optional for P1)
    GlobalIndex n_faces{0};           ///< Number of unique faces (optional for P1)
    int dim{0};                        ///< Spatial dimension (2D or 3D)

    // Cell-to-vertex connectivity (required)
    // For cell c, vertices are: cell2vertex_data[cell2vertex_offsets[c]] to
    //                           cell2vertex_data[cell2vertex_offsets[c+1]-1]
    std::vector<MeshOffset> cell2vertex_offsets;  ///< CSR offsets (size = n_cells + 1)
    std::vector<MeshIndex> cell2vertex_data;      ///< CSR data (vertex indices per cell)

    // Vertex global IDs for canonical ordering on shared entities
    std::vector<gid_t> vertex_gids;   ///< Global ID for each vertex (size = n_vertices)
    std::vector<Real> vertex_coords; ///< Optional vertex coordinates [n_vertices * dim]

    // Optional: Cell global IDs and owners (required for distributed ownership/ghosting)
    std::vector<gid_t> cell_gids;         ///< Global ID for each local cell (size = n_cells)
    std::vector<int> cell_owner_ranks;    ///< Owning rank for each local cell (size = n_cells)

    // Optional: Cell-to-edge connectivity (for P2+ spaces)
    std::vector<MeshOffset> cell2edge_offsets;
    std::vector<MeshIndex> cell2edge_data;

    // Optional: Cell-to-face connectivity (required for any face-interior DOFs, e.g. 3D Q2+ hex)
    std::vector<MeshOffset> cell2face_offsets;
    std::vector<MeshIndex> cell2face_data;

    // Optional: Edge-to-vertex connectivity (for canonical edge orientation)
    // Edge e connects: edge2vertex_data[2*e] and edge2vertex_data[2*e+1]
    std::vector<MeshIndex> edge2vertex_data;  ///< size = 2 * n_edges
    std::vector<gid_t> edge_gids;               ///< Optional global ID per local edge (size = n_edges)

    // Optional: Face-to-vertex connectivity (for canonical face orientation)
    std::vector<MeshOffset> face2vertex_offsets;
    std::vector<MeshIndex> face2vertex_data;
    std::vector<gid_t> face_gids;               ///< Optional global ID per local face (size = n_faces)

    // Optional: MPI neighbor ranks (for scalable neighbor-only communication).
    // If empty, DofHandler will attempt to infer neighbors from cell_owner_ranks
    // (requires ghost cells to be present). Providing this explicitly is
    // recommended for correctness when partitions share entities only via
    // vertices/edges (not faces).
    std::vector<int> neighbor_ranks;

    // Helper to get cell vertices
    std::span<const MeshIndex> getCellVertices(GlobalIndex cell_id) const {
        if (cell2vertex_offsets.empty()) return {};
        const auto cid = static_cast<std::size_t>(cell_id);
        auto start = static_cast<std::size_t>(cell2vertex_offsets[cid]);
        auto end = static_cast<std::size_t>(cell2vertex_offsets[cid + 1]);
        return {cell2vertex_data.data() + start, end - start};
    }

    // Helper to get cell edges
    std::span<const MeshIndex> getCellEdges(GlobalIndex cell_id) const {
        if (cell2edge_offsets.empty()) return {};
        const auto cid = static_cast<std::size_t>(cell_id);
        auto start = static_cast<std::size_t>(cell2edge_offsets[cid]);
        auto end = static_cast<std::size_t>(cell2edge_offsets[cid + 1]);
        return {cell2edge_data.data() + start, end - start};
    }

    // Helper to get cell faces
    std::span<const MeshIndex> getCellFaces(GlobalIndex cell_id) const {
        if (cell2face_offsets.empty()) return {};
        const auto cid = static_cast<std::size_t>(cell_id);
        auto start = static_cast<std::size_t>(cell2face_offsets[cid]);
        auto end = static_cast<std::size_t>(cell2face_offsets[cid + 1]);
        return {cell2face_data.data() + start, end - start};
    }

    // Helper to get edge vertices
    std::pair<MeshIndex, MeshIndex> getEdgeVertices(GlobalIndex edge_id) const {
        if (edge2vertex_data.empty()) return {-1, -1};
        auto idx = static_cast<std::size_t>(edge_id) * 2;
        return {edge2vertex_data[idx], edge2vertex_data[idx + 1]};
    }

    // Helper to get cell GID (falls back to cell_id for serial/test usage)
    gid_t getCellGid(GlobalIndex cell_id) const {
        if (cell_gids.empty()) return static_cast<gid_t>(cell_id);
        return cell_gids[static_cast<std::size_t>(cell_id)];
    }

    // Helper to get cell owner rank (falls back to default_rank for serial/test usage)
    int getCellOwnerRank(GlobalIndex cell_id, int default_rank = 0) const {
        if (cell_owner_ranks.empty()) return default_rank;
        return cell_owner_ranks[static_cast<std::size_t>(cell_id)];
    }
};

/**
 * @brief Non-owning view of MeshTopologyInfo / MeshBase topology
 *
 * This is used to avoid copying large connectivity arrays when the source
 * mesh can provide stable storage for the duration of DOF distribution.
 */
struct MeshTopologyView {
    // Basic counts
    GlobalIndex n_cells{0};
    GlobalIndex n_vertices{0};
    GlobalIndex n_edges{0};
    GlobalIndex n_faces{0};
    int dim{0};

    // Required cell->vertex connectivity and vertex global IDs.
    std::span<const MeshOffset> cell2vertex_offsets;
    std::span<const MeshIndex> cell2vertex_data;
    std::span<const gid_t> vertex_gids;
    std::span<const Real> vertex_coords;

    // Optional IDs / ownership.
    std::span<const gid_t> cell_gids;
    std::span<const int> cell_owner_ranks;

    // Optional edge topology.
    std::span<const MeshOffset> cell2edge_offsets;
    std::span<const MeshIndex> cell2edge_data;
    std::span<const MeshIndex> edge2vertex_data;  // flat [2*n_edges]
    std::span<const gid_t> edge_gids;

    // Optional face topology.
    std::span<const MeshOffset> cell2face_offsets;
    std::span<const MeshIndex> cell2face_data;
    std::span<const MeshOffset> face2vertex_offsets;
    std::span<const MeshIndex> face2vertex_data;
    std::span<const gid_t> face_gids;

    // Optional MPI neighbor ranks.
    std::span<const int> neighbor_ranks;

    static MeshTopologyView from(const MeshTopologyInfo& info) {
        MeshTopologyView view;
        view.n_cells = info.n_cells;
        view.n_vertices = info.n_vertices;
        view.n_edges = info.n_edges;
        view.n_faces = info.n_faces;
        view.dim = info.dim;

        view.cell2vertex_offsets = info.cell2vertex_offsets;
        view.cell2vertex_data = info.cell2vertex_data;
        view.vertex_gids = info.vertex_gids;
        view.vertex_coords = info.vertex_coords;

        view.cell_gids = info.cell_gids;
        view.cell_owner_ranks = info.cell_owner_ranks;

        view.cell2edge_offsets = info.cell2edge_offsets;
        view.cell2edge_data = info.cell2edge_data;
        view.edge2vertex_data = info.edge2vertex_data;
        view.edge_gids = info.edge_gids;

        view.cell2face_offsets = info.cell2face_offsets;
        view.cell2face_data = info.cell2face_data;
        view.face2vertex_offsets = info.face2vertex_offsets;
        view.face2vertex_data = info.face2vertex_data;
        view.face_gids = info.face_gids;

        view.neighbor_ranks = info.neighbor_ranks;
        return view;
    }

    [[nodiscard]] MeshTopologyInfo materialize() const {
        MeshTopologyInfo info;
        info.n_cells = n_cells;
        info.n_vertices = n_vertices;
        info.n_edges = n_edges;
        info.n_faces = n_faces;
        info.dim = dim;

        info.cell2vertex_offsets.assign(cell2vertex_offsets.begin(), cell2vertex_offsets.end());
        info.cell2vertex_data.assign(cell2vertex_data.begin(), cell2vertex_data.end());
        info.vertex_gids.assign(vertex_gids.begin(), vertex_gids.end());
        info.vertex_coords.assign(vertex_coords.begin(), vertex_coords.end());

        info.cell_gids.assign(cell_gids.begin(), cell_gids.end());
        info.cell_owner_ranks.assign(cell_owner_ranks.begin(), cell_owner_ranks.end());

        info.cell2edge_offsets.assign(cell2edge_offsets.begin(), cell2edge_offsets.end());
        info.cell2edge_data.assign(cell2edge_data.begin(), cell2edge_data.end());
        info.edge2vertex_data.assign(edge2vertex_data.begin(), edge2vertex_data.end());
        info.edge_gids.assign(edge_gids.begin(), edge_gids.end());

        info.cell2face_offsets.assign(cell2face_offsets.begin(), cell2face_offsets.end());
        info.cell2face_data.assign(cell2face_data.begin(), cell2face_data.end());
        info.face2vertex_offsets.assign(face2vertex_offsets.begin(), face2vertex_offsets.end());
        info.face2vertex_data.assign(face2vertex_data.begin(), face2vertex_data.end());
        info.face_gids.assign(face_gids.begin(), face_gids.end());

        info.neighbor_ranks.assign(neighbor_ranks.begin(), neighbor_ranks.end());
        return info;
    }

    [[nodiscard]] std::span<const MeshIndex> getCellVertices(GlobalIndex cell_id) const {
        if (cell_id < 0 || cell2vertex_offsets.empty()) return {};
        const auto cid = static_cast<std::size_t>(cell_id);
        if (cid + 1 >= cell2vertex_offsets.size()) return {};
        const auto start = static_cast<std::size_t>(cell2vertex_offsets[cid]);
        const auto end = static_cast<std::size_t>(cell2vertex_offsets[cid + 1]);
        return {cell2vertex_data.data() + start, end - start};
    }

    [[nodiscard]] std::span<const MeshIndex> getCellEdges(GlobalIndex cell_id) const {
        if (cell_id < 0 || cell2edge_offsets.empty()) return {};
        const auto cid = static_cast<std::size_t>(cell_id);
        if (cid + 1 >= cell2edge_offsets.size()) return {};
        const auto start = static_cast<std::size_t>(cell2edge_offsets[cid]);
        const auto end = static_cast<std::size_t>(cell2edge_offsets[cid + 1]);
        return {cell2edge_data.data() + start, end - start};
    }

    [[nodiscard]] std::span<const MeshIndex> getCellFaces(GlobalIndex cell_id) const {
        if (cell_id < 0 || cell2face_offsets.empty()) return {};
        const auto cid = static_cast<std::size_t>(cell_id);
        if (cid + 1 >= cell2face_offsets.size()) return {};
        const auto start = static_cast<std::size_t>(cell2face_offsets[cid]);
        const auto end = static_cast<std::size_t>(cell2face_offsets[cid + 1]);
        return {cell2face_data.data() + start, end - start};
    }

    [[nodiscard]] std::pair<MeshIndex, MeshIndex> getEdgeVertices(GlobalIndex edge_id) const {
        if (edge_id < 0 || edge2vertex_data.empty()) return {-1, -1};
        const auto idx = static_cast<std::size_t>(edge_id) * 2;
        if (idx + 1 >= edge2vertex_data.size()) return {-1, -1};
        return {edge2vertex_data[idx], edge2vertex_data[idx + 1]};
    }

    [[nodiscard]] std::span<const MeshIndex> getFaceVertices(GlobalIndex face_id) const {
        if (face_id < 0 || face2vertex_offsets.empty()) return {};
        const auto fid = static_cast<std::size_t>(face_id);
        if (fid + 1 >= face2vertex_offsets.size()) return {};
        const auto begin = static_cast<std::size_t>(face2vertex_offsets[fid]);
        const auto end = static_cast<std::size_t>(face2vertex_offsets[fid + 1]);
        if (begin > end || end > face2vertex_data.size()) return {};
        return {face2vertex_data.data() + begin, end - begin};
    }

    [[nodiscard]] gid_t getCellGid(GlobalIndex cell_id) const {
        if (cell_id < 0) return gid_t{-1};
        if (cell_gids.empty()) return static_cast<gid_t>(cell_id);
        return cell_gids[static_cast<std::size_t>(cell_id)];
    }

    [[nodiscard]] int getCellOwnerRank(GlobalIndex cell_id, int default_rank = 0) const {
        if (cell_id < 0) return default_rank;
        if (cell_owner_ranks.empty()) return default_rank;
        return cell_owner_ranks[static_cast<std::size_t>(cell_id)];
    }
};

/**
 * @brief DOF layout information (how many DOFs per entity type)
 *
 * This struct specifies how DOFs are distributed across mesh entities
 * for a given function space. It can be derived from a FunctionSpace
 * or specified directly for custom DOF layouts.
 *
 * For Lagrange P1: dofs_per_vertex=1, all others=0
 * For Lagrange P2 on triangles: dofs_per_vertex=1, dofs_per_edge=1
 * For Lagrange P3 on triangles: dofs_per_vertex=1, dofs_per_edge=2, dofs_per_cell=1
 */
struct DofLayoutInfo {
    LocalIndex dofs_per_vertex{0};     ///< DOFs at each vertex
    LocalIndex dofs_per_edge{0};       ///< Interior DOFs per edge (not counting vertices)
    LocalIndex dofs_per_face{0};       ///< Interior DOFs per face (not counting edges/vertices)
    LocalIndex dofs_per_cell{0};       ///< Interior DOFs per cell (bubble functions)
    int num_components{1};             ///< Number of field components (1=scalar, 3=vector)
    bool is_continuous{true};          ///< CG (true) vs DG (false)
    bool tensor_face_dof_layout{false};///< True when face DOFs follow tensor Lagrange interior ordering (quads/hex)

    // Total DOFs per element (convenience, computed from other fields and element type)
    LocalIndex total_dofs_per_element{0};

    // Factory for common Lagrange spaces
    static DofLayoutInfo Lagrange(int order, int dim, int num_verts_per_cell, int num_components = 1);
    static DofLayoutInfo DG(int order, int num_verts_per_cell, int num_components = 1);
};

/**
 * @brief High-level DOF management interface
 *
 * The DofHandler orchestrates DOF distribution on a mesh according to
 * a function space specification. It is the primary entry point for
 * setting up DOFs before assembly.
 *
 * Usage:
 * @code
 *   DofHandler dh;
 *   dh.distributeDofs(mesh, space, options);
 *   dh.finalize();
 *
 *   // Assembly
 *   for (auto cell_id : mesh) {
 *       auto dofs = dh.getDofMap().getCellDofs(cell_id);
 *       // ... assemble element
 *   }
 * @endcode
 *
 * Thread safety: After finalize(), read methods are thread-safe.
 */
class DofHandler {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    DofHandler();
    ~DofHandler();

    // Move semantics
    DofHandler(DofHandler&& other) noexcept;
    DofHandler& operator=(DofHandler&& other) noexcept;

    // No copy (expensive, use explicit clone() if needed)
    DofHandler(const DofHandler&) = delete;
    DofHandler& operator=(const DofHandler&) = delete;

    // =========================================================================
    // DOF Distribution (main entry point)
    // =========================================================================

    /**
     * @brief Distribute DOFs using mesh-independent topology info (RECOMMENDED)
     *
     * This is the primary, mesh-library-independent API for DOF distribution.
     * It supports both CG (continuous Galerkin) and DG (discontinuous Galerkin)
     * spaces through the layout.is_continuous flag.
     *
     * For CG spaces:
     * - Vertex DOFs are shared based on vertex_gids (same GID = same DOF)
     * - Edge/face DOFs use canonical ordering based on min vertex GID
     * - Cell interior DOFs are always unique per cell
     *
     * For DG spaces:
     * - All DOFs are unique per cell (no sharing)
     *
     * @param topology Mesh topology information
     * @param layout DOF layout per entity type
     * @param options Distribution options
     * @throws FEException if already finalized or distribution fails
     */
    void distributeDofs(const MeshTopologyInfo& topology,
                        const DofLayoutInfo& layout,
                        const DofDistributionOptions& options = {});

    /**
     * @brief Distribute DOFs using mesh-independent topology + a FunctionSpace
     *
     * This overload derives an appropriate @ref DofLayoutInfo from the space
     * (including H(curl)/H(div) entity DOF counts) and then calls the core
     * mesh-independent distribution routine.
     */
    void distributeDofs(const MeshTopologyInfo& topology,
                        const spaces::FunctionSpace& space,
                        const DofDistributionOptions& options = {});

    /**
     * @brief Distribute DOFs on a serial mesh (convenience wrapper)
     *
     * @param mesh The mesh to distribute DOFs on
     * @param space The function space defining DOF locations
     * @param options Distribution options
     * @throws FEException if already finalized or distribution fails
     *
     * @note Only available when compiled with Mesh library
     */
    void distributeDofs(const MeshBase& mesh,
                        const spaces::FunctionSpace& space,
                        const DofDistributionOptions& options = {});

    /**
     * @brief Distribute DOFs on the unified runtime mesh (convenience wrapper)
     *
     * In MPI builds this uses distributed ownership/ghost information from
     * `svmp::Mesh`. In serial builds it reduces to the local mesh behavior.
     *
     * @param mesh The runtime mesh (serial or MPI-distributed)
     * @param space The function space
     * @param options Distribution options (rank info from mesh if not set)
     * @throws FEException if already finalized or distribution fails
     *
     * @note Only available when compiled with Mesh library
     */
    void distributeDofs(const Mesh& mesh,
                        const spaces::FunctionSpace& space,
                        const DofDistributionOptions& options = {});

    /**
     * @brief Manually set DOF map (for custom numbering schemes)
     *
     * @param dof_map Pre-built DOF map
     * @throws FEException if already finalized
     */
    void setDofMap(DofMap dof_map);

    /**
     * @brief Manually set DOF partition (owned/ghost/relevant)
     *
     * This is primarily intended for composite/multi-field workflows that
     * construct a monolithic DOF map externally and need to provide matching
     * ownership metadata.
     *
     * @throws FEException if already finalized
     */
    void setPartition(DofPartition partition);

    /**
     * @brief Manually set the entity-to-DOF map
     *
     * Required for boundary DOF extraction helpers when DOFs are constructed
     * via @ref setDofMap rather than distributed through DofHandler.
     *
     * @throws FEException if already finalized
     */
    void setEntityDofMap(std::unique_ptr<EntityDofMap> entity_dof_map);

    /**
     * @brief Renumber DOFs using specified strategy
     *
     * @param strategy Renumbering strategy
     * @throws FEException if already finalized
     *
     * @note This invalidates any previously obtained DOF indices.
     */
    void renumberDofs(DofNumberingStrategy strategy);

    /**
     * @brief Finalize the DOF handler
     *
     * After finalization:
     * - All distribution methods throw
     * - Read methods are thread-safe
     * - DOF indices are stable
     *
     * @throws FEException if validation fails
     */
    void finalize();

    // =========================================================================
    // Query methods
    // =========================================================================

    /**
     * @brief Check if handler is finalized
     */
    [[nodiscard]] bool isFinalized() const noexcept { return finalized_; }

    /**
     * @brief Get the DOF map
     * @return Reference to internal DofMap
     */
    [[nodiscard]] const DofMap& getDofMap() const noexcept { return dof_map_; }

    /**
     * @brief Get mutable DOF map (only before finalize)
     * @throws FEException if already finalized
     */
    [[nodiscard]] DofMap& getDofMapMutable();

    /**
     * @brief Get the partition (owned/ghost/relevant index sets)
     */
    [[nodiscard]] const DofPartition& getPartition() const noexcept { return partition_; }

    /**
     * @brief Get entity-DOF map (if available)
     * @return Pointer to entity DOF map, or nullptr
     */
    [[nodiscard]] const EntityDofMap* getEntityDofMap() const noexcept {
        return entity_dof_map_.get();
    }

    /**
     * @brief Get ghost DOF manager (for parallel)
     * @return Pointer to ghost manager, or nullptr
     */
    [[nodiscard]] const GhostDofManager* getGhostManager() const noexcept {
        return ghost_manager_.get();
    }

    // =========================================================================
    // Orientation metadata (H(curl)/H(div) and other entity-oriented layouts)
    // =========================================================================

    /**
     * @brief Whether per-cell orientation metadata is available
     *
     * Populated for CG distributions when canonical ordering is enabled and
     * the input topology provides (or DofHandler derives) edge/face vertex lists.
     */
    [[nodiscard]] bool hasCellOrientations() const noexcept;

    /**
     * @brief Edge orientation signs for a cell (reference-edge order)
     *
     * Returns a span of size num_edges(cell_type). Empty if unavailable.
     */
    [[nodiscard]] std::span<const spaces::OrientationManager::Sign>
    cellEdgeOrientations(GlobalIndex cell_id) const;

    /**
     * @brief Face orientation descriptors for a cell (reference-face order)
     *
     * Returns a span of size num_faces(cell_type). Empty if unavailable.
     */
    [[nodiscard]] std::span<const spaces::OrientationManager::FaceOrientation>
    cellFaceOrientations(GlobalIndex cell_id) const;

    /**
     * @brief Copy per-cell orientation metadata from another DofHandler
     *
     * This is primarily intended for workflows that construct a monolithic
     * DofHandler by externally merging multiple field DofMaps. In that case,
     * the merged handler does not run distributeDofs(), so orientation tables
     * must be copied from a field handler that did.
     *
     * Must be called before finalize().
     */
    void copyCellOrientationsFrom(const DofHandler& other);

    // =========================================================================
    // Convenience accessors (delegate to DofMap)
    // =========================================================================

    /**
     * @brief Get total number of DOFs in the system
     */
    [[nodiscard]] GlobalIndex getNumDofs() const noexcept {
        return dof_map_.getNumDofs();
    }

    /**
     * @brief Get number of locally owned DOFs
     */
    [[nodiscard]] GlobalIndex getNumLocalDofs() const noexcept {
        return dof_map_.getNumLocalDofs();
    }

    /**
     * @brief Get local DOF range [begin, end) for contiguous ownership
     *
     * @return Pair (begin, end) if ownership is contiguous, or nullopt
     *
     * @note After renumbering, ownership may not be contiguous.
     */
    [[nodiscard]] std::optional<std::pair<GlobalIndex, GlobalIndex>>
    getLocalDofRange() const noexcept;

    /**
     * @brief Get DOFs for a cell
     */
    [[nodiscard]] std::span<const GlobalIndex> getCellDofs(GlobalIndex cell_id) const {
        return dof_map_.getCellDofs(cell_id);
    }

    /**
     * @brief Get ghost DOF indices
     */
    [[nodiscard]] std::span<const GlobalIndex> getGhostDofs() const;

    // =========================================================================
    // Statistics
    // =========================================================================

    /**
     * @brief Get DOF distribution statistics
     */
    struct Statistics {
        GlobalIndex total_dofs{0};
        GlobalIndex local_owned_dofs{0};
        GlobalIndex ghost_dofs{0};
        GlobalIndex min_dofs_per_cell{0};
        GlobalIndex max_dofs_per_cell{0};
        double avg_dofs_per_cell{0.0};
    };

    [[nodiscard]] Statistics getStatistics() const;

    /**
     * @brief Monotonic revision of the current DOF state
     *
     * This is incremented whenever the DOF state is modified (e.g., DOFs are
     * distributed or renumbered). It is primarily intended to support cache
     * validation and unit tests.
     */
    [[nodiscard]] std::uint64_t getDofStateRevision() const noexcept { return dof_state_revision_; }

    // =========================================================================
    // Parallel support
    // =========================================================================

	    /**
	     * @brief Build scatter contexts for MPI communication
	     *
	     * Creates optimized communication patterns for ghost exchange.
	     *
	     * @throws FEException if not finalized
	     */
	    void buildScatterContexts();

#if FE_HAS_MPI
	    /**
	     * @brief Synchronize ghost values using MPI
	     *
	     * Requires buildScatterContexts() to be called first.
	     *
	     * @param owned_values Packed-owned values in the ordering of partition_.locallyOwned()
	     * @param ghost_values Ghost buffer to fill (ordering of getGhostDofs())
	     *
	     * @note Implementation uses the most efficient mechanism available at compile time:
	     *       - MPI-3+: neighborhood collectives via `MPI_Ineighbor_alltoallv`
	     *       - MPI<3: nonblocking point-to-point (fallback)
	     */
	    void syncGhostValuesMPI(std::span<const double> owned_values,
	                            std::span<double> ghost_values);

	    /**
	     * @brief Synchronize ghost values using MPI persistent requests
	     *
	     * Requires buildScatterContexts() to be called first. The first call will
	     * lazily create persistent MPI requests and reuse them on subsequent calls.
	     *
	     * @note Implementation uses the most efficient mechanism available at compile time:
	     *       - MPI-4+: persistent neighborhood collectives via `MPI_Neighbor_alltoallv_init`
	     *       - MPI<4: persistent point-to-point (`MPI_Send_init`/`MPI_Recv_init`)
	     */
	    void syncGhostValuesMPIPersistent(std::span<const double> owned_values,
	                                      std::span<double> ghost_values);
#endif

	    /**
	     * @brief Set MPI rank information
	     */
	    void setRankInfo(int my_rank, int world_size);

private:
    // Internal helpers
    void checkNotFinalized() const;

    // Mesh-independent DOF distribution (primary implementation)
    void distributeDofsCore(const MeshTopologyView& topology,
                            const DofLayoutInfo& layout,
                            const DofDistributionOptions& options);
    void cacheSpatialDofCoordinates(const MeshTopologyView& topology,
                                    const DofLayoutInfo& layout);
    void clearSpatialDofCoordinates() noexcept;

    // Helper for CG distribution with shared DOFs
    void distributeCGDofs(const MeshTopologyView& topology,
                          const DofLayoutInfo& layout,
                          const DofDistributionOptions& options);

    // Helper for DG distribution (no sharing)
    void distributeDGDofs(const MeshTopologyView& topology,
                          const DofLayoutInfo& layout,
                          const DofDistributionOptions& options);

    // Helper for distributed (MPI) DOF distribution
    void distributeCGDofsParallel(const MeshTopologyView& topology,
                                  const DofLayoutInfo& layout,
                                  const DofDistributionOptions& options);
    void distributeDGDofsParallel(const MeshTopologyView& topology,
                                  const DofLayoutInfo& layout,
                                  const DofDistributionOptions& options);

    // Legacy mesh-based internal helper (for backward compatibility)
    void distributeDofsInternal(const MeshBase& mesh,
                                const spaces::FunctionSpace& space,
                                const DofDistributionOptions& options,
                                bool is_distributed);

    // Core data
    DofMap dof_map_;
    DofPartition partition_;

    // Optional components (created during distribution)
    std::unique_ptr<EntityDofMap> entity_dof_map_;
    std::unique_ptr<GhostDofManager> ghost_manager_;

    // Cached ghost indices for getGhostDofs()
    mutable std::vector<GlobalIndex> ghost_dofs_cache_;
    mutable bool ghost_cache_valid_{false};

		    // State
			    bool finalized_{false};
                std::uint64_t dof_state_revision_{0};
			    int my_rank_{0};
			    int world_size_{1};
			#if FE_HAS_MPI
			    MPI_Comm mpi_comm_{MPI_COMM_WORLD};
			#endif
		    std::vector<int> neighbor_ranks_;
		    GlobalNumberingMode global_numbering_{GlobalNumberingMode::OwnerContiguous};
		    bool no_global_collectives_{false};

		#if FE_HAS_MPI
		    struct GhostExchangeContextMPI;
		    std::unique_ptr<GhostExchangeContextMPI> ghost_exchange_mpi_;
	#endif

	    // Optional mesh association (used by Mesh convenience overloads only)
	    struct MeshCacheState;
	    std::unique_ptr<MeshCacheState> mesh_cache_;

	    // Associated mesh/space info (for validation)
	    GlobalIndex n_cells_{0};
	    int spatial_dim_{0};
	    LocalIndex num_components_{1};

        // Optional per-cell orientation metadata (CSR-style).
        std::vector<MeshOffset> cell_edge_orient_offsets_{};
        std::vector<spaces::OrientationManager::Sign> cell_edge_orient_data_{};
        std::vector<MeshOffset> cell_face_orient_offsets_{};
        std::vector<spaces::OrientationManager::FaceOrientation> cell_face_orient_data_{};
        std::vector<double> spatial_dof_coords_{};
        int spatial_dof_coord_dim_{0};
	};

} // namespace dofs
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_DOFS_DOFHANDLER_H
