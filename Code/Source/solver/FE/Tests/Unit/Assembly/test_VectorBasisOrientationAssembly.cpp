/**
 * @file test_VectorBasisOrientationAssembly.cpp
 * @brief Unit tests for StandardAssembler orientation handling on H(div)/H(curl) vector bases
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Dofs/DofHandler.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Spaces/HDivSpace.h"

#include <array>
#include <memory>
#include <sstream>
#include <vector>

namespace svmp {
namespace FE {
namespace test {

namespace {

class TwoTetraMeshAccess final : public assembly::IMeshAccess {
public:
    explicit TwoTetraMeshAccess(bool flipped)
    {
        nodes_ = {
            {0.0, 0.0, 0.0},  // 0
            {1.0, 0.0, 0.0},  // 1
            {0.0, 1.0, 0.0},  // 2
            {0.0, 0.0, 1.0},  // 3
            {1.0, 1.0, 1.0}   // 4
        };
        cells_[0] = {0, 1, 2, 3};
        // Use an even permutation of the shared-face vertices (1,2,3) so the
        // cell orientation (detJ) remains positive.
        cells_[1] = flipped ? std::array<GlobalIndex, 4>{2, 3, 1, 4}
                            : std::array<GlobalIndex, 4>{1, 2, 3, 4};
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Tetra4; }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        nodes.assign(cell.begin(), cell.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        coords.resize(cell.size());
        for (std::size_t i = 0; i < cell.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(cell[i]));
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/, GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override { return -1; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        callback(0);
        callback(1);
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        forEachCell(std::move(callback));
    }

    void forEachBoundaryFace(int /*marker*/,
                             std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

private:
    std::vector<std::array<Real, 3>> nodes_{};
    std::array<std::array<GlobalIndex, 4>, 2> cells_{};
};

dofs::MeshTopologyInfo buildTwoTetraTopology(bool flipped)
{
    dofs::MeshTopologyInfo topo;
    topo.dim = 3;
    topo.n_cells = 2;
    topo.n_vertices = 5;
    topo.n_faces = 7;

    topo.vertex_gids = {0, 1, 2, 3, 4};
    topo.cell_gids = {0, 1};
    topo.cell_owner_ranks = {0, 0};

    topo.cell2vertex_offsets = {0, 4, 8};
    topo.cell2vertex_data = {
        0, 1, 2, 3,
        (flipped ? 2 : 1), (flipped ? 3 : 2), (flipped ? 1 : 3), 4
    };

    // Canonical global faces (triangles) for the two-tetra configuration.
    const std::array<std::array<MeshIndex, 3>, 7> faces = {{
        {{0, 1, 2}},
        {{0, 1, 3}},
        {{0, 2, 3}},
        {{1, 2, 3}},
        {{1, 2, 4}},
        {{1, 3, 4}},
        {{2, 3, 4}}
    }};

    topo.face2vertex_offsets.resize(faces.size() + 1u);
    topo.face2vertex_offsets[0] = 0;
    topo.face2vertex_data.reserve(faces.size() * 3u);
    for (std::size_t f = 0; f < faces.size(); ++f) {
        topo.face2vertex_data.push_back(faces[f][0]);
        topo.face2vertex_data.push_back(faces[f][1]);
        topo.face2vertex_data.push_back(faces[f][2]);
        topo.face2vertex_offsets[f + 1u] = static_cast<MeshOffset>((f + 1u) * 3u);
    }

    // Cell-to-face connectivity in reference-face order for Tetra4:
    // faces = {{0,1,2},{0,1,3},{1,2,3},{0,2,3}}.
    topo.cell2face_offsets = {0, 4, 8};
    topo.cell2face_data = {
        // cell 0: {0,1,2,3}
        0, 1, 3, 2,
        // cell 1: baseline {1,2,3,4} or flipped {2,3,1,4}
        3,
        (flipped ? 6 : 4),
        (flipped ? 5 : 6),
        (flipped ? 4 : 5)
    };

    return topo;
}

} // namespace

TEST(VectorBasisOrientationAssembly, HDivMassMatrixInvariantToCellVertexPermutation)
{
    spaces::HDivSpace space(ElementType::Tetra4, /*order=*/0);

    forms::FormCompiler compiler;
    const auto u = forms::FormExpr::trialFunction(space, "u");
    const auto v = forms::FormExpr::testFunction(space, "v");
    auto ir = compiler.compileBilinear(inner(u, v).dx());
    forms::FormKernel kernel(std::move(ir));

    const auto assemble = [&](bool flipped) {
        dofs::DofHandler dh;
        dofs::DofDistributionOptions opts;
        opts.topology_completion = dofs::TopologyCompletion::RequireComplete;
        dh.distributeDofs(buildTwoTetraTopology(flipped), space, opts);
        dh.finalize();

        assembly::StandardAssembler assembler;
        assembler.setDofHandler(dh);

        const auto n = dh.getNumDofs();
        assembly::DenseMatrixView mat(n);
        mat.zero();

        TwoTetraMeshAccess mesh(flipped);
        (void)assembler.assembleMatrix(mesh, space, space, kernel, mat);
        return mat;
    };

    const auto mat_ref = assemble(/*flipped=*/false);
    const auto mat_flip = assemble(/*flipped=*/true);

    const auto make_trace = [&](bool flipped) -> std::string {
        dofs::DofHandler dh;
        dofs::DofDistributionOptions opts;
        opts.topology_completion = dofs::TopologyCompletion::RequireComplete;
        dh.distributeDofs(buildTwoTetraTopology(flipped), space, opts);
        dh.finalize();

        std::ostringstream oss;
        oss << "flipped=" << flipped;
        if (dh.hasCellOrientations()) {
            const auto faces = dh.cellFaceOrientations(/*cell_id=*/1);
            oss << " cell1_face_signs=[";
            for (std::size_t i = 0; i < faces.size(); ++i) {
                if (i) oss << ",";
                oss << faces[i].sign;
            }
            oss << "]";
        } else {
            oss << " (no orientations)";
        }
        const auto cell_dofs = dh.getDofMap().getCellDofs(/*cell_id=*/1);
        oss << " cell1_dofs=[";
        for (std::size_t i = 0; i < cell_dofs.size(); ++i) {
            if (i) oss << ",";
            oss << cell_dofs[i];
        }
        oss << "]";
        return oss.str();
    };

    SCOPED_TRACE(make_trace(false));
    SCOPED_TRACE(make_trace(true));

    ASSERT_EQ(mat_ref.numRows(), mat_flip.numRows());
    ASSERT_EQ(mat_ref.numCols(), mat_flip.numCols());

    for (GlobalIndex i = 0; i < mat_ref.numRows(); ++i) {
        for (GlobalIndex j = 0; j < mat_ref.numCols(); ++j) {
            EXPECT_NEAR(mat_ref.getMatrixEntry(i, j), mat_flip.getMatrixEntry(i, j), 1e-12)
                << "i=" << i << " j=" << j;
        }
    }
}

} // namespace test
} // namespace FE
} // namespace svmp
