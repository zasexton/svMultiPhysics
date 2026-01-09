/**
 * @file test_AssemblerSelection.cpp
 * @brief Systems-level tests for assembler selection wiring + DG validation
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"

#include "Assembly/AssemblyKernel.h"
#include "Spaces/H1Space.h"

#include <array>
#include <memory>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {
namespace test {

namespace {

class TwoCellDGMesh final : public assembly::IMeshAccess {
public:
    TwoCellDGMesh()
    {
        nodes_ = {
            {0.0, 0.0, 0.0},  // 0
            {1.0, 0.0, 0.0},  // 1
            {0.0, 1.0, 0.0},  // 2
            {0.0, 0.0, 1.0},  // 3
            {1.0, 1.0, 1.0},  // 4
        };
        cells_ = {
            {0, 1, 2, 3},
            {1, 2, 3, 4},
        };
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 1; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Tetra4; }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override
    {
        nodes.assign(cells_.at(static_cast<std::size_t>(cell_id)).begin(),
                     cells_.at(static_cast<std::size_t>(cell_id)).end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex cell_id, std::vector<std::array<Real, 3>>& coords) const override
    {
        coords.resize(4);
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        for (std::size_t i = 0; i < cell.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(cell[i]));
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/, GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override { return 0; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override
    {
        return {0, 1};
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

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback) const override
    {
        callback(/*face_id=*/0, /*cell_minus=*/0, /*cell_plus=*/1);
    }

private:
    std::vector<std::array<Real, 3>> nodes_{};
    std::vector<std::array<GlobalIndex, 4>> cells_{};
};

class MinimalDGKernel final : public assembly::AssemblyKernel {
public:
    [[nodiscard]] assembly::RequiredData getRequiredData() const override { return assembly::RequiredData::None; }
    [[nodiscard]] bool hasCell() const noexcept override { return false; }
    [[nodiscard]] bool hasInteriorFace() const noexcept override { return true; }

    void computeCell(const assembly::AssemblyContext& ctx, assembly::KernelOutput& out) override
    {
        // Not used by these tests, but required by the interface.
        (void)ctx;
        out.reserve(/*n_test=*/0, /*n_trial=*/0, /*need_matrix=*/true, /*need_vector=*/false);
    }
};

static dofs::MeshTopologyInfo buildTopology()
{
    dofs::MeshTopologyInfo topo;
    topo.n_cells = 2;
    topo.n_vertices = 5;
    topo.dim = 3;

    topo.cell2vertex_offsets = {0, 4, 8};
    topo.cell2vertex_data = {0, 1, 2, 3, 1, 2, 3, 4};

    topo.vertex_gids = {0, 1, 2, 3, 4};
    topo.cell_gids = {0, 1};
    topo.cell_owner_ranks = {0, 0};

    return topo;
}

} // namespace

TEST(SystemSetup, UsesAssemblerNameOptInSurface)
{
    auto mesh = std::make_shared<const TwoCellDGMesh>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    FESystem sys(mesh);
    auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");
    sys.addInteriorFaceKernel("op", u, u, std::make_shared<MinimalDGKernel>());

    SetupOptions opts;
    opts.assembler_name = "StandardAssembler";
    SetupInputs inputs;
    inputs.topology_override = buildTopology();
    sys.setup(opts, inputs);

    EXPECT_EQ(sys.assemblerName(), "StandardAssembler");
}

TEST(SystemSetup, DGIncompatibleAssemblerSelectionThrows)
{
    auto mesh = std::make_shared<const TwoCellDGMesh>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    FESystem sys(mesh);
    auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");
    sys.addInteriorFaceKernel("op", u, u, std::make_shared<MinimalDGKernel>());

    SetupOptions opts;
    opts.assembler_name = "WorkStreamAssembler"; // supportsDG() == false

    SetupInputs inputs;
    inputs.topology_override = buildTopology();

    EXPECT_THROW(sys.setup(opts, inputs), FEException);
}

TEST(SystemSetup, DecoratorsComposeViaAssemblyOptions)
{
    auto mesh = std::make_shared<const TwoCellDGMesh>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    FESystem sys(mesh);
    auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");
    sys.addInteriorFaceKernel("op", u, u, std::make_shared<MinimalDGKernel>());

    SetupOptions opts;
    opts.assembler_name = "StandardAssembler";
    opts.assembly_options.schedule_elements = true;
    opts.assembly_options.schedule_strategy = 0; // Natural
    opts.assembly_options.cache_element_data = true;
    opts.assembly_options.use_batching = true;
    opts.assembly_options.batch_size = 8;

    SetupInputs inputs;
    inputs.topology_override = buildTopology();
    sys.setup(opts, inputs);

    EXPECT_EQ(sys.assemblerName(), "Vectorized(Cached(Scheduled(StandardAssembler)))");
}

TEST(SystemSetup, SelectionReportIsPopulated)
{
    auto mesh = std::make_shared<const TwoCellDGMesh>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, /*order=*/1);

    FESystem sys(mesh);
    auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");
    sys.addInteriorFaceKernel("op", u, u, std::make_shared<MinimalDGKernel>());

    SetupOptions opts;
    opts.assembler_name = "StandardAssembler";

    SetupInputs inputs;
    inputs.topology_override = buildTopology();
    sys.setup(opts, inputs);

    const auto report = sys.assemblerSelectionReport();
    EXPECT_FALSE(report.empty());
    EXPECT_NE(report.find("Selected assembler:"), std::string::npos);
}

} // namespace test
} // namespace systems
} // namespace FE
} // namespace svmp
