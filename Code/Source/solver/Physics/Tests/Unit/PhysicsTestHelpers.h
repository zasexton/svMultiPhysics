#ifndef SVMP_PHYSICS_TESTS_UNIT_PHYSICS_TEST_HELPERS_H
#define SVMP_PHYSICS_TESTS_UNIT_PHYSICS_TEST_HELPERS_H

#include "FE/Assembly/Assembler.h"
#include "FE/Assembly/GlobalSystemView.h"
#include "FE/Dofs/DofHandler.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/SystemSetup.h"

#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
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

[[nodiscard]] inline std::filesystem::path squareTriMeshVtpPath()
{
    return unitTestDataDir() / "Square" / "square.vtp";
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
/**
 * @brief Load the square triangular mesh from a VTP file using Mesh + VTK IO.
 *
 * Requires:
 * - FE built with Mesh integration (`FE_WITH_MESH=ON` -> `SVMP_FE_WITH_MESH=1`)
 * - Mesh built with VTK enabled (`MESH_ENABLE_VTK=ON` -> `MESH_HAS_VTK` defined)
 */
[[nodiscard]] inline std::shared_ptr<const svmp::Mesh> loadSquareTriMeshFromVtp()
{
#  if !defined(MESH_HAS_VTK)
    throw std::runtime_error("loadSquareTriMeshFromVtp: Mesh was built without VTK support (MESH_HAS_VTK not defined)");
#  else
    const auto vtp_path = squareTriMeshVtpPath();
    if (!std::filesystem::exists(vtp_path)) {
        throw std::runtime_error("loadSquareTriMeshFromVtp: missing test mesh file: " + vtp_path.string());
    }

    svmp::MeshIOOptions opts;
    opts.format = "vtp";
    opts.path = vtp_path.string();

    auto base = std::make_shared<svmp::MeshBase>(svmp::MeshBase::load(opts));
    return svmp::create_mesh(std::move(base));
#  endif
}

#endif // defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

} // namespace test
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_TESTS_UNIT_PHYSICS_TEST_HELPERS_H
