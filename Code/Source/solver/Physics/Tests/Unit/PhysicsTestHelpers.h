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

    const std::vector<FE::Real> u0(base_state.u.begin(), base_state.u.end());
    ASSERT_EQ(static_cast<FE::GlobalIndex>(u0.size()), n);

    const auto& constraints = system.constraints();
    std::vector<FE::Real> base_u = u0;
    if (!constraints.empty()) {
        constraints.distribute(base_u);
    }

    FE::systems::SystemStateView constrained_base_state = base_state;
    constrained_base_state.u = std::span<const FE::Real>(base_u);

    FE::assembly::DenseMatrixView J(n);
    {
        const auto result = system.assembleJacobian(constrained_base_state, J);
        ASSERT_TRUE(result.success) << result.error_message;
    }

    for (FE::GlobalIndex j = 0; j < n; ++j) {
        if (constraints.isConstrained(j)) {
            continue;
        }

        std::vector<FE::Real> u_plus = base_u;
        std::vector<FE::Real> u_minus = base_u;
        u_plus[static_cast<std::size_t>(j)] += eps;
        u_minus[static_cast<std::size_t>(j)] -= eps;
        if (!constraints.empty()) {
            constraints.distribute(u_plus);
            constraints.distribute(u_minus);
        }

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
            if (constraints.isConstrained(i)) {
                continue;
            }

            const FE::Real fd = (r_plus[i] - r_minus[i]) / (2.0 * eps);
            const FE::Real Jij = J(i, j);
            const FE::Real tol = atol + rtol * std::max<FE::Real>(1.0, std::abs(fd));
            EXPECT_NEAR(Jij, fd, tol) << "Mismatch at (i=" << i << ", j=" << j << ")";
        }
    }
}

inline void expectOperatorJacobianMatchesCentralFD(FE::systems::FESystem& system,
                                                   const FE::systems::SystemStateView& base_state,
                                                   std::string_view op,
                                                   FE::Real eps = 1e-6,
                                                   FE::Real rtol = 5e-5,
                                                   FE::Real atol = 1e-8)
{
    ASSERT_FALSE(op.empty());

    const auto n = system.dofHandler().getNumDofs();
    ASSERT_GT(n, 0);

    const std::vector<FE::Real> u0(base_state.u.begin(), base_state.u.end());
    ASSERT_EQ(static_cast<FE::GlobalIndex>(u0.size()), n);

    const auto& constraints = system.constraints();
    std::vector<FE::Real> base_u = u0;
    if (!constraints.empty()) {
        constraints.distribute(base_u);
    }

    FE::systems::SystemStateView constrained_base_state = base_state;
    constrained_base_state.u = std::span<const FE::Real>(base_u);

    FE::assembly::DenseMatrixView J(n);
    {
        FE::systems::AssemblyRequest req;
        req.op = std::string(op);
        req.want_matrix = true;
        const auto result = system.assemble(req, constrained_base_state, &J, nullptr);
        ASSERT_TRUE(result.success) << result.error_message;
    }

    for (FE::GlobalIndex j = 0; j < n; ++j) {
        if (constraints.isConstrained(j)) {
            continue;
        }

        std::vector<FE::Real> u_plus = base_u;
        std::vector<FE::Real> u_minus = base_u;
        u_plus[static_cast<std::size_t>(j)] += eps;
        u_minus[static_cast<std::size_t>(j)] -= eps;
        if (!constraints.empty()) {
            constraints.distribute(u_plus);
            constraints.distribute(u_minus);
        }

        FE::systems::SystemStateView state_plus = base_state;
        FE::systems::SystemStateView state_minus = base_state;
        state_plus.u = std::span<const FE::Real>(u_plus);
        state_minus.u = std::span<const FE::Real>(u_minus);

        FE::assembly::DenseVectorView r_plus(n);
        FE::assembly::DenseVectorView r_minus(n);
        {
            FE::systems::AssemblyRequest req;
            req.op = std::string(op);
            req.want_vector = true;
            const auto rp = system.assemble(req, state_plus, nullptr, &r_plus);
            ASSERT_TRUE(rp.success) << rp.error_message;
        }
        {
            FE::systems::AssemblyRequest req;
            req.op = std::string(op);
            req.want_vector = true;
            const auto rm = system.assemble(req, state_minus, nullptr, &r_minus);
            ASSERT_TRUE(rm.success) << rm.error_message;
        }

        for (FE::GlobalIndex i = 0; i < n; ++i) {
            if (constraints.isConstrained(i)) {
                continue;
            }

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

[[nodiscard]] inline std::filesystem::path cubeMeshVtuPath()
{
    return unitTestDataDir() / "Cube" / "mesh" / "mesh-complete.mesh.vtu";
}

[[nodiscard]] inline std::filesystem::path elasticPipeDataDir()
{
    return unitTestDataDir() / "ElasticPipe";
}

[[nodiscard]] inline std::filesystem::path elasticPipeParticipantMeshVtuPath(
    std::string_view variant,
    std::string_view participant)
{
    return elasticPipeDataDir() / std::string(variant) /
           std::string(participant) / "mesh" / "mesh-complete.mesh.vtu";
}

[[nodiscard]] inline std::filesystem::path elasticPipeParticipantSurfaceVtpPath(
    std::string_view variant,
    std::string_view participant,
    std::string_view surface_name)
{
    return elasticPipeDataDir() / std::string(variant) /
           std::string(participant) / "mesh" / "mesh-surfaces" /
           (std::string(surface_name) + ".vtp");
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

inline constexpr svmp::label_t kCubeBoundaryLeft = 1;
inline constexpr svmp::label_t kCubeBoundaryRight = 2;
inline constexpr svmp::label_t kCubeBoundaryBack = 3;
inline constexpr svmp::label_t kCubeBoundaryFront = 4;
inline constexpr svmp::label_t kCubeBoundaryBottom = 5;
inline constexpr svmp::label_t kCubeBoundaryTop = 6;

struct CubeBoundaryBounds {
    double xmin{0.0};
    double xmax{0.0};
    double ymin{0.0};
    double ymax{0.0};
    double zmin{0.0};
    double zmax{0.0};
};

[[nodiscard]] inline CubeBoundaryBounds computeCubeBoundaryBounds(const svmp::MeshBase& mesh)
{
    CubeBoundaryBounds b;
    b.xmin = b.ymin = b.zmin = std::numeric_limits<double>::infinity();
    b.xmax = b.ymax = b.zmax = -std::numeric_limits<double>::infinity();
    for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(mesh.n_vertices()); ++v) {
        const auto xyz = mesh.get_vertex_coords(v);
        const double x = static_cast<double>(xyz[0]);
        const double y = static_cast<double>(xyz[1]);
        const double z = static_cast<double>(xyz[2]);
        b.xmin = std::min(b.xmin, x);
        b.xmax = std::max(b.xmax, x);
        b.ymin = std::min(b.ymin, y);
        b.ymax = std::max(b.ymax, y);
        b.zmin = std::min(b.zmin, z);
        b.zmax = std::max(b.zmax, z);
    }
    return b;
}

inline void registerCubeBoundaryLabels(svmp::MeshBase& mesh)
{
    mesh.register_label("left", kCubeBoundaryLeft);
    mesh.register_label("right", kCubeBoundaryRight);
    mesh.register_label("back", kCubeBoundaryBack);
    mesh.register_label("front", kCubeBoundaryFront);
    mesh.register_label("bottom", kCubeBoundaryBottom);
    mesh.register_label("top", kCubeBoundaryTop);
}

inline void markCubeBoundaryFacesByGeometry(svmp::MeshBase& mesh)
{
    if (mesh.dim() != 3) {
        throw std::runtime_error("markCubeBoundaryFacesByGeometry: expected a 3D volume mesh");
    }

    const auto b = computeCubeBoundaryBounds(mesh);
    const double scale = std::max({1.0,
                                   std::abs(b.xmax - b.xmin),
                                   std::abs(b.ymax - b.ymin),
                                   std::abs(b.zmax - b.zmin)});
    const double tol = 1e-10 * scale;

    for (const auto f : mesh.boundary_faces()) {
        const auto fv = mesh.face_vertices(f);
        if (fv.empty()) {
            continue;
        }

        double cx = 0.0;
        double cy = 0.0;
        double cz = 0.0;
        for (const auto v : fv) {
            const auto xyz = mesh.get_vertex_coords(v);
            cx += static_cast<double>(xyz[0]);
            cy += static_cast<double>(xyz[1]);
            cz += static_cast<double>(xyz[2]);
        }
        const double inv_n = 1.0 / static_cast<double>(fv.size());
        cx *= inv_n;
        cy *= inv_n;
        cz *= inv_n;

        svmp::label_t marker = svmp::INVALID_LABEL;
        if (std::abs(cx - b.xmin) <= tol) {
            marker = kCubeBoundaryLeft;
        } else if (std::abs(cx - b.xmax) <= tol) {
            marker = kCubeBoundaryRight;
        } else if (std::abs(cy - b.ymin) <= tol) {
            marker = kCubeBoundaryBack;
        } else if (std::abs(cy - b.ymax) <= tol) {
            marker = kCubeBoundaryFront;
        } else if (std::abs(cz - b.zmin) <= tol) {
            marker = kCubeBoundaryBottom;
        } else if (std::abs(cz - b.zmax) <= tol) {
            marker = kCubeBoundaryTop;
        }

        if (marker == svmp::INVALID_LABEL) {
            throw std::runtime_error("markCubeBoundaryFacesByGeometry: found boundary face not on a cube side");
        }
        mesh.set_boundary_label(f, marker);
    }
}

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
[[nodiscard]] inline std::shared_ptr<const svmp::Mesh>
loadSquareMeshWithMarkedBoundaries(svmp::MeshComm comm = svmp::MeshComm::world())
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

    // In MPI runs we need a truly distributed mesh (partitioned cells + ghost layer),
    // otherwise each rank would assemble and solve the full serial problem.
    if (comm.is_parallel()) {
        // Ensure partition interfaces are not treated as physical boundaries by FE boundary iteration.
        opts.kv["ghost_layers"] = "1";
        // Deterministic partitioning for unit tests.
        opts.kv["partition_method"] = "block";

        auto mesh = svmp::load_mesh(opts, comm);
        auto& base = mesh->base();

        base.register_label("left", kSquareBoundaryLeft);
        base.register_label("right", kSquareBoundaryRight);
        base.register_label("bottom", kSquareBoundaryBottom);
        base.register_label("top", kSquareBoundaryTop);

        const auto bbox = mesh->global_bounding_box();
        const double xmin = static_cast<double>(bbox.min[0]);
        const double xmax = static_cast<double>(bbox.max[0]);
        const double ymin = static_cast<double>(bbox.min[1]);
        const double ymax = static_cast<double>(bbox.max[1]);
        const double tol = 1e-10;

        const auto& f2c = base.face2cell();
        for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(f2c.size()); ++f) {
            const auto& fc = f2c[static_cast<std::size_t>(f)];
            const bool c0_valid = (fc[0] != svmp::INVALID_INDEX);
            const bool c1_valid = (fc[1] != svmp::INVALID_INDEX);
            if (c0_valid == c1_valid) {
                continue;
            }
            const auto adj_cell = static_cast<svmp::index_t>(c0_valid ? fc[0] : fc[1]);
            if (!mesh->is_owned_cell(adj_cell)) {
                continue;
            }

            const auto fv = base.face_vertices(f);
            if (fv.empty()) {
                continue;
            }
            double cx = 0.0;
            double cy = 0.0;
            for (const auto v : fv) {
                const auto xyz = base.get_vertex_coords(v);
                cx += static_cast<double>(xyz[0]);
                cy += static_cast<double>(xyz[1]);
            }
            cx /= static_cast<double>(fv.size());
            cy /= static_cast<double>(fv.size());

            svmp::label_t marker = svmp::INVALID_LABEL;
            if (std::abs(cx - xmin) <= tol) {
                marker = kSquareBoundaryLeft;
            } else if (std::abs(cx - xmax) <= tol) {
                marker = kSquareBoundaryRight;
            } else if (std::abs(cy - ymin) <= tol) {
                marker = kSquareBoundaryBottom;
            } else if (std::abs(cy - ymax) <= tol) {
                marker = kSquareBoundaryTop;
            }
            if (marker != svmp::INVALID_LABEL) {
                base.set_boundary_label(f, marker);
            }
        }

        // Sanity: every boundary face adjacent to an owned cell should have a marker now.
        for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(f2c.size()); ++f) {
            const auto& fc = f2c[static_cast<std::size_t>(f)];
            const bool c0_valid = (fc[0] != svmp::INVALID_INDEX);
            const bool c1_valid = (fc[1] != svmp::INVALID_INDEX);
            if (c0_valid == c1_valid) {
                continue;
            }
            const auto adj_cell = static_cast<svmp::index_t>(c0_valid ? fc[0] : fc[1]);
            if (!mesh->is_owned_cell(adj_cell)) {
                continue;
            }
            const auto lbl = base.boundary_label(f);
            if (lbl == svmp::INVALID_LABEL) {
                throw std::runtime_error("loadSquareMeshWithMarkedBoundaries: found unlabeled boundary face on owned partition");
            }
        }

        return mesh;
    }

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
    return svmp::create_mesh(std::move(base), comm);
#  endif
}

/**
 * @brief Load the tetrahedral cube mesh (VTU) and label its six planar boundary sides.
 *
 * Requires:
 * - FE built with Mesh integration (`FE_WITH_MESH=ON` -> `SVMP_FE_WITH_MESH=1`)
 * - Mesh built with VTK enabled (`MESH_ENABLE_VTK=ON` -> `MESH_HAS_VTK` defined)
 */
[[nodiscard]] inline std::shared_ptr<const svmp::Mesh>
loadCubeMeshWithMarkedBoundaries(svmp::MeshComm comm = svmp::MeshComm::world())
{
#  if !defined(MESH_HAS_VTK)
    throw std::runtime_error("loadCubeMeshWithMarkedBoundaries: Mesh was built without VTK support (MESH_HAS_VTK not defined)");
#  else
    const auto vtu_path = cubeMeshVtuPath();
    if (!std::filesystem::exists(vtu_path)) {
        throw std::runtime_error("loadCubeMeshWithMarkedBoundaries: missing test mesh file: " + vtu_path.string());
    }

    svmp::MeshIOOptions opts;
    opts.format = "vtu";
    opts.path = vtu_path.string();

    if (comm.is_parallel()) {
        opts.kv["ghost_layers"] = "1";
        opts.kv["partition_method"] = "block";

        auto mesh = svmp::load_mesh(opts, comm);
        auto& base = mesh->base();
        registerCubeBoundaryLabels(base);
        markCubeBoundaryFacesByGeometry(base);
        return mesh;
    }

    auto base = std::make_shared<svmp::MeshBase>(svmp::MeshBase::load(opts));
    registerCubeBoundaryLabels(*base);
    markCubeBoundaryFacesByGeometry(*base);
    return svmp::create_mesh(std::move(base), comm);
#  endif
}

#endif // defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

} // namespace test
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_TESTS_UNIT_PHYSICS_TEST_HELPERS_H
