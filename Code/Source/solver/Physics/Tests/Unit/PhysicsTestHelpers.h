#ifndef SVMP_PHYSICS_TESTS_UNIT_PHYSICS_TEST_HELPERS_H
#define SVMP_PHYSICS_TESTS_UNIT_PHYSICS_TEST_HELPERS_H

#include "FE/Assembly/Assembler.h"
#include "FE/Assembly/GlobalSystemView.h"
#include "FE/Dofs/DofHandler.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/SystemSetup.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Mesh.h"
#endif

namespace svmp {
namespace Physics {
namespace test {

class SingleTetraMeshAccess final : public FE::assembly::IMeshAccess {
public:
    SingleTetraMeshAccess()
    {
        nodes_ = {
            {0.0, 0.0, 0.0},  // 0
            {1.0, 0.0, 0.0},  // 1
            {0.0, 1.0, 0.0},  // 2
            {0.0, 0.0, 1.0}   // 3
        };
        cell_ = {0, 1, 2, 3};
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex /*cell_id*/) const override
    {
        return FE::ElementType::Tetra4;
    }

    void getCellNodes(FE::GlobalIndex /*cell_id*/, std::vector<FE::GlobalIndex>& nodes) const override
    {
        nodes.assign(cell_.begin(), cell_.end());
    }

    [[nodiscard]] std::array<FE::Real, 3> getNodeCoordinates(FE::GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(FE::GlobalIndex /*cell_id*/,
                            std::vector<std::array<FE::Real, 3>>& coords) const override
    {
        coords.resize(cell_.size());
        for (std::size_t i = 0; i < cell_.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(cell_[i]));
        }
    }

    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(FE::GlobalIndex /*face_id*/,
                                                   FE::GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(FE::GlobalIndex /*face_id*/) const override { return -1; }

    [[nodiscard]] std::pair<FE::GlobalIndex, FE::GlobalIndex>
    getInteriorFaceCells(FE::GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachOwnedCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachBoundaryFace(int /*marker*/,
                             std::function<void(FE::GlobalIndex, FE::GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(std::function<void(FE::GlobalIndex, FE::GlobalIndex, FE::GlobalIndex)> /*callback*/) const override
    {
    }

private:
    std::vector<std::array<FE::Real, 3>> nodes_{};
    std::array<FE::GlobalIndex, 4> cell_{};
};

[[nodiscard]] inline FE::dofs::MeshTopologyInfo makeSingleTetraTopology()
{
    FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 4;
    topo.n_edges = 0;
    topo.n_faces = 0;
    topo.dim = 3;

    topo.cell2vertex_offsets = {0, 4};
    topo.cell2vertex_data = {0, 1, 2, 3};
    topo.vertex_gids = {0, 1, 2, 3};
    topo.cell_gids = {0};
    topo.cell_owner_ranks = {0};

    return topo;
}

[[nodiscard]] inline FE::systems::SetupInputs makeSingleTetraSetupInputs()
{
    FE::systems::SetupInputs inputs;
    inputs.topology_override = makeSingleTetraTopology();
    return inputs;
}

inline void expectJacobianMatchesCentralFD(FE::systems::FESystem& system,
                                           const FE::systems::SystemStateView& base_state,
                                           FE::Real eps = 1e-6,
                                           FE::Real rtol = 5e-5,
                                           FE::Real atol = 1e-8)
{
    const auto n = system.dofHandler().getNumDofs();
    ASSERT_GT(n, 0);

    FE::assembly::DenseMatrixView J(n);
    {
        const auto result = system.assembleJacobian(base_state, J);
        ASSERT_TRUE(result.success) << result.error_message;
    }

    const std::vector<FE::Real> u0(base_state.u.begin(), base_state.u.end());
    ASSERT_EQ(static_cast<FE::GlobalIndex>(u0.size()), n);

    for (FE::GlobalIndex j = 0; j < n; ++j) {
        std::vector<FE::Real> u_plus = u0;
        std::vector<FE::Real> u_minus = u0;
        u_plus[static_cast<std::size_t>(j)] += eps;
        u_minus[static_cast<std::size_t>(j)] -= eps;

        FE::systems::SystemStateView state_plus = base_state;
        FE::systems::SystemStateView state_minus = base_state;
        state_plus.u = std::span<const FE::Real>(u_plus);
        state_minus.u = std::span<const FE::Real>(u_minus);

        FE::assembly::DenseVectorView r_plus(n);
        FE::assembly::DenseVectorView r_minus(n);
        {
            const auto rp = system.assembleResidual(state_plus, r_plus);
            ASSERT_TRUE(rp.success) << rp.error_message;
        }
        {
            const auto rm = system.assembleResidual(state_minus, r_minus);
            ASSERT_TRUE(rm.success) << rm.error_message;
        }

        for (FE::GlobalIndex i = 0; i < n; ++i) {
            const FE::Real fd = (r_plus[i] - r_minus[i]) / (2.0 * eps);
            const FE::Real Jij = J(i, j);
            const FE::Real tol = atol + rtol * std::max<FE::Real>(1.0, std::abs(fd));
            EXPECT_NEAR(Jij, fd, tol) << "Mismatch at (i=" << i << ", j=" << j << ")";
        }
    }
}

//------------------------------------------------------------------------------
// Optional Mesh + VTK helpers (for larger, still-simple meshes)
//------------------------------------------------------------------------------

[[nodiscard]] inline std::filesystem::path unitTestDataDir()
{
    return std::filesystem::path(__FILE__).parent_path() / "Data";
}

[[nodiscard]] inline std::filesystem::path squareMeshVtuPath()
{
    return unitTestDataDir() / "Square" / "mesh" / "mesh-complete.mesh.vtu";
}

[[nodiscard]] inline std::filesystem::path squareMeshSurfacesDir()
{
    return unitTestDataDir() / "Square" / "mesh" / "mesh-surfaces";
}

[[nodiscard]] inline std::filesystem::path squareMeshSurfaceVtpPath(std::string_view surface_name)
{
    return squareMeshSurfacesDir() / (std::string(surface_name) + ".vtp");
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
inline constexpr svmp::label_t kSquareBoundaryLeft = 1;
inline constexpr svmp::label_t kSquareBoundaryRight = 2;
inline constexpr svmp::label_t kSquareBoundaryBottom = 3;
inline constexpr svmp::label_t kSquareBoundaryTop = 4;

/**
 * @brief Mark boundary faces on a 2D volume mesh using a VTP surface mesh.
 *
 * The surface mesh is expected to contain line segments (2-node cells) that
 * coincide with boundary faces (edges) of the volume mesh.
 */
inline void mark2DBoundaryFacesFromVtp(svmp::MeshBase& volume_mesh,
                                       const std::filesystem::path& surface_vtp_path,
                                       svmp::label_t marker)
{
#  if !defined(MESH_HAS_VTK)
    (void)volume_mesh;
    (void)surface_vtp_path;
    (void)marker;
    throw std::runtime_error("mark2DBoundaryFacesFromVtp: Mesh was built without VTK support (MESH_HAS_VTK not defined)");
#  else
    if (volume_mesh.dim() != 2) {
        throw std::runtime_error("mark2DBoundaryFacesFromVtp: expected a 2D volume mesh");
    }
    if (marker < 0) {
        throw std::runtime_error("mark2DBoundaryFacesFromVtp: marker must be non-negative");
    }
    if (!std::filesystem::exists(surface_vtp_path)) {
        throw std::runtime_error("mark2DBoundaryFacesFromVtp: missing surface file: " + surface_vtp_path.string());
    }

    svmp::MeshIOOptions surf_opts;
    surf_opts.format = "vtp";
    surf_opts.path = surface_vtp_path.string();
    const svmp::MeshBase surface_mesh = svmp::MeshBase::load(surf_opts);

    if (surface_mesh.dim() != volume_mesh.dim()) {
        throw std::runtime_error("mark2DBoundaryFacesFromVtp: surface/volume dimension mismatch");
    }

    // Build a coordinate->vertex lookup for the volume mesh (exact for grid-aligned test meshes,
    // tolerant for general VTK float data).
    struct CoordKey {
        std::int64_t x;
        std::int64_t y;
    };
    struct CoordKeyHash {
        std::size_t operator()(const CoordKey& k) const noexcept
        {
            std::size_t h = 1469598103934665603ull;
            const auto mix = [&](std::int64_t v) {
                h ^= std::hash<std::int64_t>{}(v) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
            };
            mix(k.x);
            mix(k.y);
            return h;
        }
    };
    struct CoordKeyEq {
        bool operator()(const CoordKey& a, const CoordKey& b) const noexcept { return a.x == b.x && a.y == b.y; }
    };

    auto make_key = [](const std::array<svmp::real_t, 3>& xyz) -> CoordKey {
        constexpr double kKeyScale = 1e12;
        return CoordKey{static_cast<std::int64_t>(std::llround(static_cast<double>(xyz[0]) * kKeyScale)),
                        static_cast<std::int64_t>(std::llround(static_cast<double>(xyz[1]) * kKeyScale))};
    };

    std::unordered_map<CoordKey, svmp::index_t, CoordKeyHash, CoordKeyEq> volume_vertex_by_xy;
    volume_vertex_by_xy.reserve(volume_mesh.n_vertices());
    for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(volume_mesh.n_vertices()); ++v) {
        volume_vertex_by_xy.emplace(make_key(volume_mesh.get_vertex_coords(v)), v);
    }

    // Build a canonical-vertex-list -> face lookup for the volume mesh.
    struct IndexVecHash {
        std::size_t operator()(const std::vector<svmp::index_t>& v) const noexcept
        {
            std::size_t h = 1469598103934665603ull;
            for (svmp::index_t x : v) {
                const std::size_t k = std::hash<svmp::index_t>{}(x);
                h ^= k + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
            }
            return h;
        }
    };

    std::unordered_map<std::vector<svmp::index_t>, svmp::index_t, IndexVecHash> face_by_vertices;
    face_by_vertices.reserve(volume_mesh.n_faces());
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(volume_mesh.n_faces()); ++f) {
        auto verts = volume_mesh.face_vertices(f);
        std::sort(verts.begin(), verts.end());
        face_by_vertices.emplace(std::move(verts), f);
    }

    // Map surface vertices -> volume vertices by coordinates.
    std::vector<svmp::index_t> surf2vol(surface_mesh.n_vertices(), svmp::INVALID_INDEX);
    for (svmp::index_t sv = 0; sv < static_cast<svmp::index_t>(surface_mesh.n_vertices()); ++sv) {
        const auto sxyz = surface_mesh.get_vertex_coords(sv);
        const auto key = make_key(sxyz);
        const auto it = volume_vertex_by_xy.find(key);
        if (it != volume_vertex_by_xy.end()) {
            surf2vol[static_cast<std::size_t>(sv)] = it->second;
            continue;
        }

        // Fallback: nearest-neighbor match (tolerant to float32 IO differences).
        constexpr double kTol = 1e-6;
        constexpr double kTol2 = kTol * kTol;
        svmp::index_t best = svmp::INVALID_INDEX;
        double best2 = std::numeric_limits<double>::infinity();
        for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(volume_mesh.n_vertices()); ++v) {
            const auto vxyz = volume_mesh.get_vertex_coords(v);
            const double dx = static_cast<double>(vxyz[0]) - static_cast<double>(sxyz[0]);
            const double dy = static_cast<double>(vxyz[1]) - static_cast<double>(sxyz[1]);
            const double d2 = dx * dx + dy * dy;
            if (d2 < best2) {
                best2 = d2;
                best = v;
            }
        }
        if (!(best2 <= kTol2) || best == svmp::INVALID_INDEX) {
            throw std::runtime_error("mark2DBoundaryFacesFromVtp: surface vertex not found in volume mesh (by XY key or NN)");
        }
        surf2vol[static_cast<std::size_t>(sv)] = best;
    }

    // Mark boundary faces by matching surface cells to volume faces via vertex sets.
    for (svmp::index_t c = 0; c < static_cast<svmp::index_t>(surface_mesh.n_cells()); ++c) {
        auto [cell_ptr, n_cell_verts] = surface_mesh.cell_vertices_span(c);
        if (n_cell_verts < 2) {
            throw std::runtime_error("mark2DBoundaryFacesFromVtp: surface cell has <2 vertices");
        }

        std::vector<svmp::index_t> vol_face_verts;
        vol_face_verts.reserve(n_cell_verts);
        for (std::size_t i = 0; i < n_cell_verts; ++i) {
            const auto sv = cell_ptr[i];
            if (sv < 0 || static_cast<std::size_t>(sv) >= surf2vol.size()) {
                throw std::runtime_error("mark2DBoundaryFacesFromVtp: surface cell references invalid vertex index");
            }
            const auto vv = surf2vol[static_cast<std::size_t>(sv)];
            if (vv == svmp::INVALID_INDEX) {
                throw std::runtime_error("mark2DBoundaryFacesFromVtp: unmapped surface vertex");
            }
            vol_face_verts.push_back(vv);
        }
        std::sort(vol_face_verts.begin(), vol_face_verts.end());

        const auto fit = face_by_vertices.find(vol_face_verts);
        if (fit == face_by_vertices.end()) {
            throw std::runtime_error("mark2DBoundaryFacesFromVtp: surface cell does not match any face in volume mesh");
        }
        volume_mesh.set_boundary_label(fit->second, marker);
    }
#  endif
}

/**
 * @brief Load the square mesh (VTU) and label its boundary faces using per-side VTP files.
 *
 * Requires:
 * - FE built with Mesh integration (`FE_WITH_MESH=ON` -> `SVMP_FE_WITH_MESH=1`)
 * - Mesh built with VTK enabled (`MESH_ENABLE_VTK=ON` -> `MESH_HAS_VTK` defined)
 */
[[nodiscard]] inline std::shared_ptr<const svmp::Mesh> loadSquareMeshWithMarkedBoundaries()
{
#  if !defined(MESH_HAS_VTK)
    throw std::runtime_error("loadSquareMeshWithMarkedBoundaries: Mesh was built without VTK support (MESH_HAS_VTK not defined)");
#  else
    const auto vtu_path = squareMeshVtuPath();
    if (!std::filesystem::exists(vtu_path)) {
        throw std::runtime_error("loadSquareMeshWithMarkedBoundaries: missing test mesh file: " + vtu_path.string());
    }

    svmp::MeshIOOptions opts;
    opts.format = "vtu";
    opts.path = vtu_path.string();

    auto base = std::make_shared<svmp::MeshBase>(svmp::MeshBase::load(opts));

    // Register human-readable names for test convenience.
    base->register_label("left", kSquareBoundaryLeft);
    base->register_label("right", kSquareBoundaryRight);
    base->register_label("bottom", kSquareBoundaryBottom);
    base->register_label("top", kSquareBoundaryTop);

    mark2DBoundaryFacesFromVtp(*base, squareMeshSurfaceVtpPath("left"), kSquareBoundaryLeft);
    mark2DBoundaryFacesFromVtp(*base, squareMeshSurfaceVtpPath("right"), kSquareBoundaryRight);
    mark2DBoundaryFacesFromVtp(*base, squareMeshSurfaceVtpPath("bottom"), kSquareBoundaryBottom);
    mark2DBoundaryFacesFromVtp(*base, squareMeshSurfaceVtpPath("top"), kSquareBoundaryTop);

    // Sanity: every boundary face should have a marker now.
    for (const auto f : base->boundary_faces()) {
        const auto lbl = base->boundary_label(f);
        if (lbl == svmp::INVALID_LABEL) {
            throw std::runtime_error("loadSquareMeshWithMarkedBoundaries: found unlabeled boundary face");
        }
    }
    return svmp::create_mesh(std::move(base));
#  endif
}

#endif // defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

} // namespace test
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_TESTS_UNIT_PHYSICS_TEST_HELPERS_H
