#include "LevelSet/LevelSetVolume.h"

#include "Assembly/Assembler.h"
#include "Assembly/CutIntegrationContext.h"
#include "Assembly/GlobalSystemView.h"
#include "Basis/NodeOrderingConventions.h"
#include "Dofs/DofHandler.h"
#include "Dofs/EntityDofMap.h"
#include "Forms/WeakForm.h"
#include "Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "LevelSet/LevelSetInterfaceLifecycle.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"
#include "Spaces/H1Space.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Systems/SystemSetup.h"
#include "Systems/TimeIntegrator.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;

std::shared_ptr<svmp::Mesh> buildSingleCellMesh(
    int spatial_dim,
    std::span<const std::array<FE::Real, 3>> coordinates,
    svmp::CellFamily family);

std::shared_ptr<svmp::Mesh> buildSingleQuadMesh()
{
    auto base = std::make_shared<svmp::MeshBase>();

    const std::vector<svmp::real_t> x_ref = {
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0,
    };
    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4};
    const std::vector<svmp::index_t> cell2vertex = {0, 1, 2, 3};

    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(
        /*spatial_dim=*/2,
        x_ref,
        cell2vertex_offsets,
        cell2vertex,
        {shape});
    base->finalize();
    return svmp::create_mesh(std::move(base));
}

std::shared_ptr<svmp::Mesh> buildFourTriangleStripMesh()
{
    auto base = std::make_shared<svmp::MeshBase>();

    const std::vector<svmp::real_t> x_ref = {
        0.0, 0.0,
        1.0, 0.0,
        2.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        2.0, 1.0,
    };
    const std::vector<svmp::offset_t> cell2vertex_offsets = {
        0, 3, 6, 9, 12};
    const std::vector<svmp::index_t> cell2vertex = {
        0, 1, 4,
        0, 4, 3,
        1, 2, 5,
        1, 5, 4,
    };

    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Triangle;
    shape.num_corners = 3;
    shape.order = 1;
    base->build_from_arrays(
        /*spatial_dim=*/2,
        x_ref,
        cell2vertex_offsets,
        cell2vertex,
        {shape, shape, shape, shape});
    base->finalize();
    return svmp::create_mesh(std::move(base));
}

std::shared_ptr<svmp::Mesh> buildSingleCellMesh(
    int spatial_dim,
    std::span<const std::array<FE::Real, 3>> coordinates,
    svmp::CellFamily family)
{
    auto base = std::make_shared<svmp::MeshBase>();

    std::vector<svmp::real_t> x_ref;
    x_ref.reserve(coordinates.size() * static_cast<std::size_t>(spatial_dim));
    for (const auto& x : coordinates) {
        for (int d = 0; d < spatial_dim; ++d) {
            x_ref.push_back(static_cast<svmp::real_t>(x[static_cast<std::size_t>(d)]));
        }
    }

    std::vector<svmp::index_t> cell2vertex;
    cell2vertex.reserve(coordinates.size());
    for (std::size_t i = 0; i < coordinates.size(); ++i) {
        cell2vertex.push_back(static_cast<svmp::index_t>(i));
    }
    const std::vector<svmp::offset_t> cell2vertex_offsets = {
        0,
        static_cast<svmp::offset_t>(cell2vertex.size()),
    };

    svmp::CellShape shape{};
    shape.family = family;
    shape.num_corners = static_cast<int>(coordinates.size());
    shape.order = 1;
    base->build_from_arrays(
        spatial_dim,
        x_ref,
        cell2vertex_offsets,
        cell2vertex,
        {shape});
    base->finalize();
    return svmp::create_mesh(std::move(base));
}

class SingleTetraMeshAccess final : public FE::assembly::IMeshAccess {
public:
    SingleTetraMeshAccess()
    {
        nodes_ = {
            std::array<FE::Real, 3>{0.0, 0.0, 0.0},
            std::array<FE::Real, 3>{1.0, 0.0, 0.0},
            std::array<FE::Real, 3>{0.0, 1.0, 0.0},
            std::array<FE::Real, 3>{0.0, 0.0, 1.0},
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

    void getCellNodes(FE::GlobalIndex /*cell_id*/,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        nodes.assign(cell_.begin(), cell_.end());
    }

    [[nodiscard]] std::array<FE::Real, 3> getNodeCoordinates(
        FE::GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(
        FE::GlobalIndex /*cell_id*/,
        std::vector<std::array<FE::Real, 3>>& coords) const override
    {
        coords = nodes_;
    }

    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(
        FE::GlobalIndex /*face_id*/,
        FE::GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(FE::GlobalIndex /*face_id*/) const override
    {
        return -1;
    }

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

    void forEachBoundaryFace(
        int /*marker*/,
        std::function<void(FE::GlobalIndex, FE::GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(
        std::function<void(FE::GlobalIndex, FE::GlobalIndex, FE::GlobalIndex)>
            /*callback*/) const override
    {
    }

private:
    std::vector<std::array<FE::Real, 3>> nodes_{};
    std::array<FE::GlobalIndex, 4> cell_{};
};

[[nodiscard]] FE::systems::SetupInputs makeSingleTetraSetupInputs()
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

    FE::systems::SetupInputs inputs;
    inputs.topology_override = std::move(topo);
    return inputs;
}

struct ScalarFieldFixture {
    std::shared_ptr<SingleTetraMeshAccess> mesh{};
    FE::systems::FESystem system;
    FE::FieldId phi{FE::INVALID_FIELD_ID};

    ScalarFieldFixture()
        : mesh(std::make_shared<SingleTetraMeshAccess>()),
          system(mesh)
    {
        auto scalar_space =
            FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);
        phi = system.addField(FE::systems::FieldSpec{
            .name = "phi",
            .space = scalar_space,
            .components = 1,
        });
        system.setup({}, makeSingleTetraSetupInputs());
    }
};

[[nodiscard]] std::vector<FE::Real> planeCoefficients(const ScalarFieldFixture& fixture,
                                                      FE::Real offset)
{
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto* entity_map = field_dofs.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error("planeCoefficients: field has no entity DOF map");
    }

    std::vector<FE::Real> coefficients(
        static_cast<std::size_t>(field_dofs.getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto dofs = entity_map->getVertexDofs(vertex);
        if (dofs.size() != 1u) {
            throw std::runtime_error("planeCoefficients: expected one vertex DOF");
        }
        const auto x = fixture.mesh->getNodeCoordinates(vertex);
        coefficients[static_cast<std::size_t>(dofs.front())] =
            x[0] + x[1] + x[2] - offset;
    }
    return coefficients;
}

} // namespace

TEST(LevelSetVolume, CutCellVolumeUsesGeneratedInterfaceFractions)
{
    const ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto coefficients = planeCoefficients(fixture, FE::Real{0.5});

    level_set::LevelSetVolumeOptions volume_opts{};
    volume_opts.tolerance = 1.0e-12;
    const auto result = level_set::computeLevelSetCutCellVolume(
        *fixture.mesh,
        field_dofs,
        volume_opts,
        coefficients);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.cells, 1u);
    EXPECT_EQ(result.cut_cells, 1u);
    EXPECT_EQ(result.full_negative_cells, 0u);
    EXPECT_EQ(result.full_positive_cells, 0u);
    EXPECT_NEAR(result.total_volume, 1.0 / 6.0, 1.0e-12);
    EXPECT_NEAR(result.negative_volume, 1.0 / 48.0, 1.0e-12);
    EXPECT_NEAR(result.positive_volume, 7.0 / 48.0, 1.0e-12);
}

TEST(LevelSetVolume, CutCellVolumeSupportsLinearHexWedgeAndPyramidCuts)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    struct Case {
        const char* name;
        FE::ElementType element_type;
        svmp::CellFamily family;
        std::vector<std::array<FE::Real, 3>> coordinates;
        std::function<FE::Real(const std::array<FE::Real, 3>&)> level_set;
        FE::Real total_volume;
        FE::Real negative_volume;
    };

    const std::vector<Case> cases = {
        Case{
            .name = "hex",
            .element_type = FE::ElementType::Hex8,
            .family = svmp::CellFamily::Hex,
            .coordinates = {
                {{0.0, 0.0, 0.0}},
                {{1.0, 0.0, 0.0}},
                {{1.0, 1.0, 0.0}},
                {{0.0, 1.0, 0.0}},
                {{0.0, 0.0, 1.0}},
                {{1.0, 0.0, 1.0}},
                {{1.0, 1.0, 1.0}},
                {{0.0, 1.0, 1.0}},
            },
            .level_set = [](const auto& x) { return x[0] - FE::Real{0.5}; },
            .total_volume = FE::Real{1.0},
            .negative_volume = FE::Real{0.5},
        },
        Case{
            .name = "wedge",
            .element_type = FE::ElementType::Wedge6,
            .family = svmp::CellFamily::Wedge,
            .coordinates = {
                {{0.0, 0.0, -1.0}},
                {{1.0, 0.0, -1.0}},
                {{0.0, 1.0, -1.0}},
                {{0.0, 0.0, 1.0}},
                {{1.0, 0.0, 1.0}},
                {{0.0, 1.0, 1.0}},
            },
            .level_set = [](const auto& x) { return x[0] - FE::Real{0.5}; },
            .total_volume = FE::Real{1.0},
            .negative_volume = FE::Real{0.75},
        },
        Case{
            .name = "pyramid",
            .element_type = FE::ElementType::Pyramid5,
            .family = svmp::CellFamily::Pyramid,
            .coordinates = {
                {{-1.0, -1.0, 0.0}},
                {{1.0, -1.0, 0.0}},
                {{1.0, 1.0, 0.0}},
                {{-1.0, 1.0, 0.0}},
                {{0.0, 0.0, 1.0}},
            },
            .level_set = [](const auto& x) { return x[2] - FE::Real{0.5}; },
            .total_volume = FE::Real{4.0} / FE::Real{3.0},
            .negative_volume = FE::Real{7.0} / FE::Real{6.0},
        },
    };

    for (const auto& c : cases) {
        SCOPED_TRACE(c.name);
        auto mesh = buildSingleCellMesh(
            /*spatial_dim=*/3,
            std::span<const std::array<FE::Real, 3>>(
                c.coordinates.data(), c.coordinates.size()),
            c.family);
        auto phi_space =
            std::make_shared<FE::spaces::H1Space>(c.element_type, /*order=*/1);

        FE::systems::FESystem system(mesh);
        const auto phi = system.addField(FE::systems::FieldSpec{
            .name = "phi",
            .space = phi_space,
            .components = 1,
        });
        ASSERT_NO_THROW(system.setup());

        const auto& field_dofs = system.fieldDofHandler(phi);
        const auto* entity_map = field_dofs.getEntityDofMap();
        ASSERT_NE(entity_map, nullptr);
        ASSERT_EQ(static_cast<std::size_t>(entity_map->numVertices()),
                  c.coordinates.size());

        std::vector<FE::Real> coefficients(
            static_cast<std::size_t>(field_dofs.getNumDofs()), FE::Real{0.0});
        for (FE::GlobalIndex vertex = 0; vertex < entity_map->numVertices();
             ++vertex) {
            const auto dofs = entity_map->getVertexDofs(vertex);
            ASSERT_EQ(dofs.size(), 1u);
            ASSERT_GE(dofs.front(), 0);
            ASSERT_LT(static_cast<std::size_t>(dofs.front()), coefficients.size());
            coefficients[static_cast<std::size_t>(dofs.front())] =
                c.level_set(c.coordinates[static_cast<std::size_t>(vertex)]);
        }

        level_set::LevelSetVolumeOptions volume_opts{};
        volume_opts.tolerance = 1.0e-12;
        const auto result = level_set::computeLevelSetCutCellVolume(
            system.meshAccess(),
            field_dofs,
            volume_opts,
            coefficients);

        ASSERT_TRUE(result.success) << result.diagnostic;
        EXPECT_EQ(result.cells, 1u);
        EXPECT_EQ(result.cut_cells, 1u);
        EXPECT_EQ(result.full_negative_cells, 0u);
        EXPECT_EQ(result.full_positive_cells, 0u);
        EXPECT_NEAR(result.total_volume, c.total_volume, 1.0e-12);
        EXPECT_NEAR(result.negative_volume, c.negative_volume, 1.0e-12);
        EXPECT_NEAR(result.positive_volume,
                    c.total_volume - c.negative_volume,
                    1.0e-12);
    }
#endif
}

TEST(LevelSetVolume,
     GeneratedCutVolumeUsesPointwiseJacobianOnWarpedTensorProductCell)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    // This bilinear map has x=(xi+1)/2 and
    // y=(eta+1)(1+0.4*xi)/2.  Its Jacobian is
    // (1+0.4*xi)/4, so the xi<0 half has physical area 0.4 even
    // though its reference-space fraction is exactly one half.
    constexpr std::array<std::array<FE::Real, 3>, 4> coordinates = {{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {1.0, 1.4, 0.0},
        {0.0, 0.6, 0.0},
    }};
    auto mesh = buildSingleCellMesh(
        /*spatial_dim=*/2, coordinates, svmp::CellFamily::Quad);
    auto phi_space =
        std::make_shared<FE::spaces::H1Space>(FE::ElementType::Quad4,
                                              /*order=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = phi_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup());

    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto* entity_map = field_dofs.getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);
    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()),
        FE::Real{0.0});
    const auto offset = system.fieldDofOffset(phi);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto dofs = entity_map->getVertexDofs(vertex);
        ASSERT_EQ(dofs.size(), 1u);
        solution[static_cast<std::size_t>(offset + dofs.front())] =
            coordinates[static_cast<std::size_t>(vertex)][0] - FE::Real{0.5};
    }

    level_set::LevelSetVolumeOptions options{};
    options.use_generated_interface_quadrature = true;
    options.level_set_field_name = "phi";
    options.generated_domain_id = "warped_quad_pointwise_volume";
    options.allow_corner_linearized_geometry = true;
    options.quadrature_order = 4;

    const auto result = level_set::computeLevelSetCutCellVolume(
        system, phi, options, solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.cut_cells, 1u);
    EXPECT_NEAR(result.total_volume, 1.0, 1.0e-12);
    EXPECT_NEAR(result.negative_volume, 0.4, 1.0e-12);
    EXPECT_NEAR(result.positive_volume, 0.6, 1.0e-12);
    EXPECT_NEAR(result.negative_volume + result.positive_volume,
                result.total_volume,
                1.0e-12);
#endif
}

TEST(LevelSetVolume, FESystemOverloadUsesFieldSlice)
{
    const ScalarFieldFixture fixture;
    const auto coefficients = planeCoefficients(fixture, FE::Real{0.5});
    std::vector<FE::Real> solution(
        static_cast<std::size_t>(fixture.system.dofHandler().getNumDofs()), 2.0);
    const auto offset = static_cast<std::size_t>(fixture.system.fieldDofOffset(fixture.phi));
    std::copy(coefficients.begin(),
              coefficients.end(),
              solution.begin() + static_cast<std::ptrdiff_t>(offset));

    const auto result = level_set::computeLevelSetCutCellVolume(
        fixture.system,
        fixture.phi,
        level_set::LevelSetVolumeOptions{},
        solution);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.cut_cells, 1u);
    EXPECT_NEAR(result.negative_volume, 1.0 / 48.0, 1.0e-12);
}

TEST(LevelSetVolume, GeneratedInterfaceVolumeDiscoversInteriorIsland)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    auto mesh = buildSingleQuadMesh();
    auto phi_space =
        std::make_shared<FE::spaces::H1Space>(FE::ElementType::Quad4,
                                              /*order=*/2);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = phi_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup());

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    ASSERT_GE(cell_dofs.size(), 9u);
    const auto offset = system.fieldDofOffset(phi);
    constexpr FE::Real radius = FE::Real{0.2};
    for (std::size_t i = 0; i < 9u; ++i) {
        const auto xi =
            FE::basis::ReferenceNodeLayout::get_node_coords(FE::ElementType::Quad9, i);
        const auto x = FE::Real{0.5} * (xi[0] + FE::Real{1.0});
        const auto y = FE::Real{0.5} * (xi[1] + FE::Real{1.0});
        solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
            (x - FE::Real{0.5}) * (x - FE::Real{0.5}) +
            (y - FE::Real{0.5}) * (y - FE::Real{0.5}) -
            radius * radius;
    }

    const auto corner_linear = level_set::computeLevelSetCutCellVolume(
        system,
        phi,
        level_set::LevelSetVolumeOptions{},
        solution);
    ASSERT_TRUE(corner_linear.success) << corner_linear.diagnostic;
    EXPECT_EQ(corner_linear.cut_cells, 0u);
    EXPECT_NEAR(corner_linear.negative_volume, 0.0, 1.0e-12);

    level_set::LevelSetVolumeOptions generated_opts{};
    generated_opts.use_generated_interface_quadrature = true;
    generated_opts.level_set_field_name = "phi";
    generated_opts.generated_domain_id = "interior_island_volume";
    generated_opts.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    generated_opts.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle;
    generated_opts.implicit_cut_fallback_policy =
        level_set::ImplicitCutFallbackPolicy::Fail;
    generated_opts.interface_quadrature_order = 2;
    generated_opts.volume_quadrature_order = 2;
    generated_opts.implicit_cut_max_subdivision_depth = 8;
    generated_opts.require_production_qualified_implicit_cut_backend = true;

    const auto generated = level_set::computeLevelSetCutCellVolume(
        system,
        phi,
        generated_opts,
        solution);
    ASSERT_TRUE(generated.success) << generated.diagnostic;
    EXPECT_EQ(generated.cut_cells, 1u);
    EXPECT_GT(generated.negative_volume, 0.0);
    EXPECT_GT(generated.positive_volume, 0.0);
    EXPECT_NEAR(generated.negative_volume,
                3.141592653589793238462643383279502884 * radius * radius,
                3.0e-2);
#endif
}

TEST(LevelSetVolume, FullCellSnapshotRulesAssembleLikeDirectBasisEvaluation)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    // phi = x - 1.5 on the four-triangle strip [0,2]x[0,1] leaves the two
    // left triangles uncut: their snapshot rules are full-side rules with
    // identical reference points, which the assembler tabulates once and
    // shares. The assembled cut-volume mass + stiffness matrix must match
    // direct evaluation of the P1 basis at every retained rule point.
    auto mesh = buildFourTriangleStripMesh();
    auto space =
        std::make_shared<FE::spaces::H1Space>(FE::ElementType::Triangle3,
                                              /*order=*/1);
    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup());

    const auto& access = system.meshAccess();
    const auto vertex_x = [&](FE::GlobalIndex vertex) {
        return access.getNodeCoordinates(vertex);
    };
    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()),
        FE::Real{0.0});
    {
        const auto* entity_map = system.fieldDofHandler(phi).getEntityDofMap();
        ASSERT_NE(entity_map, nullptr);
        const auto offset = static_cast<std::size_t>(system.fieldDofOffset(phi));
        for (FE::GlobalIndex vertex = 0; vertex < entity_map->numVertices(); ++vertex) {
            const auto dofs = entity_map->getVertexDofs(vertex);
            ASSERT_EQ(dofs.size(), 1u);
            solution[offset + static_cast<std::size_t>(dofs.front())] =
                vertex_x(vertex)[0] - FE::Real{1.5};
        }
    }

    // Affine map of each triangle: x = x0 + J xi.
    struct CellMap {
        std::array<FE::GlobalIndex, 3> vertices{};
        std::array<FE::Real, 3> origin{};
        std::array<std::array<FE::Real, 2>, 2> J{};
    };
    std::vector<CellMap> cells(static_cast<std::size_t>(access.numCells()));
    for (FE::GlobalIndex cell = 0; cell < access.numCells(); ++cell) {
        std::vector<FE::GlobalIndex> nodes;
        access.getCellNodes(cell, nodes);
        ASSERT_EQ(nodes.size(), 3u);
        auto& map = cells[static_cast<std::size_t>(cell)];
        map.vertices = {{nodes[0], nodes[1], nodes[2]}};
        map.origin = vertex_x(nodes[0]);
        for (std::size_t r = 0; r < 2u; ++r) {
            for (std::size_t c = 0; c < 2u; ++c) {
                map.J[r][c] = vertex_x(nodes[c + 1u])[r] - map.origin[r];
            }
        }
    }

    level_set::LevelSetGeneratedInterfaceOptions interface_options{};
    interface_options.level_set_field_name = "phi";
    interface_options.domain_id = "full_cell_tabulation";
    interface_options.requested_interface_marker = 721;
    interface_options.interface_quadrature_order = 2;
    interface_options.volume_quadrature_order = 2;
    interface_options.allow_corner_linearized_geometry = true;
    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    auto generated = lifecycle.build(system, interface_options, solution);
    ASSERT_TRUE(generated.success) << generated.diagnostic;

    FE::interfaces::FreeSurfaceGeometrySnapshotPolicy snapshot_policy;
    snapshot_policy.require_complete_exterior_boundary_partition = false;
    FE::interfaces::FreeSurfaceGeometryScalarEvaluator scalar;
    scalar.value = [cells](FE::GlobalIndex cell,
                           const std::array<FE::Real, 3>& xi,
                           const FE::geometry::CutQuadratureProvenance&) {
        const auto& map = cells.at(static_cast<std::size_t>(cell));
        return map.origin[0] + map.J[0][0] * xi[0] + map.J[0][1] * xi[1] -
               FE::Real{1.5};
    };
    scalar.reference_gradient = [cells](FE::GlobalIndex cell,
                                        const std::array<FE::Real, 3>&,
                                        const FE::geometry::CutQuadratureProvenance&) {
        const auto& map = cells.at(static_cast<std::size_t>(cell));
        return std::array<FE::Real, 3>{{map.J[0][0], map.J[0][1], 0.0}};
    };
    const auto snapshot = FE::interfaces::buildFreeSurfaceGeometrySnapshot(
        std::move(generated.domain),
        {},
        {},
        access,
        snapshot_policy,
        std::move(scalar),
        "full_cell_tabulation");
    ASSERT_TRUE(snapshot);

    // At least two uncut cells share bitwise identical full-side points.
    std::vector<const FE::geometry::CutQuadratureRule*> full_rules;
    for (const auto* record : snapshot->retainedRules(
             FE::interfaces::FreeSurfaceGeometryRuleRole::NegativeVolume)) {
        ASSERT_NE(record, nullptr);
        if (record->reference_rule.full_cell_equivalent) {
            full_rules.push_back(&record->reference_rule);
        }
    }
    ASSERT_GE(full_rules.size(), 2u);
    ASSERT_EQ(full_rules[0]->points.size(), full_rules[1]->points.size());
    for (std::size_t q = 0; q < full_rules[0]->points.size(); ++q) {
        EXPECT_EQ(full_rules[0]->points[q].point, full_rules[1]->points[q].point);
        EXPECT_EQ(full_rules[0]->points[q].weight, full_rules[1]->points[q].weight);
    }

    FE::systems::FESystem assembly_system(mesh);
    const auto u_field = assembly_system.addField(FE::systems::FieldSpec{
        .name = "u",
        .space = space,
        .components = 1,
    });
    assembly_system.addOperator("mass_stiffness");
    const auto u = FE::forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = FE::forms::FormExpr::testFunction(*space, "v");
    const auto residual =
        (u * v + FE::forms::inner(FE::forms::grad(u), FE::forms::grad(v)))
            .dCutVolume(interface_options.requested_interface_marker,
                        FE::forms::CutVolumeSide::Negative);
    const auto installed = FE::systems::installFormulation(
        assembly_system, "mass_stiffness", {u_field}, residual);
    ASSERT_FALSE(installed.residual.empty());

    auto cut_context = std::make_shared<FE::assembly::CutIntegrationContext>();
    cut_context->addFreeSurfaceGeometrySnapshot(
        snapshot, FE::geometry::CutIntegrationSide::Negative);
    assembly_system.setCutIntegrationContext(cut_context);
    ASSERT_NO_THROW(assembly_system.setup());

    const auto n_dofs = assembly_system.dofHandler().getNumDofs();
    std::vector<FE::Real> state_values(static_cast<std::size_t>(n_dofs), FE::Real{0.0});
    FE::systems::SystemStateView state;
    state.u = state_values;
    FE::systems::AssemblyRequest request;
    request.op = "mass_stiffness";
    request.want_matrix = true;
    request.want_vector = false;
    FE::assembly::DenseSystemView assembled(n_dofs);
    assembled.zero();
    const auto assembly = assembly_system.assemble(request, state, &assembled, nullptr);
    ASSERT_TRUE(assembly.success) << assembly.error_message;

    // Direct evaluation: P1 values (1 - xi - eta, xi, eta) and their
    // physical gradients J^{-T} grad_xi at every retained rule point.
    const auto* u_map = assembly_system.fieldDofHandler(u_field).getEntityDofMap();
    ASSERT_NE(u_map, nullptr);
    const auto u_offset = assembly_system.fieldDofOffset(u_field);
    std::vector<FE::Real> expected(static_cast<std::size_t>(n_dofs * n_dofs), FE::Real{0.0});
    std::size_t evaluated_points = 0u;
    for (const auto* record : snapshot->retainedRules(
             FE::interfaces::FreeSurfaceGeometryRuleRole::NegativeVolume)) {
        const auto& rule = record->reference_rule;
        const auto& map = cells.at(static_cast<std::size_t>(rule.provenance.parent_entity));
        const FE::Real det = map.J[0][0] * map.J[1][1] - map.J[0][1] * map.J[1][0];
        const std::array<std::array<FE::Real, 2>, 2> inv{{
            {{map.J[1][1] / det, -map.J[0][1] / det}},
            {{-map.J[1][0] / det, map.J[0][0] / det}}}};
        const std::array<std::array<FE::Real, 2>, 3> grad_ref{{
            {{-1.0, -1.0}}, {{1.0, 0.0}}, {{0.0, 1.0}}}};
        std::array<std::array<FE::Real, 2>, 3> grad{};
        for (std::size_t a = 0; a < 3u; ++a) {
            for (std::size_t d = 0; d < 2u; ++d) {
                grad[a][d] = inv[0][d] * grad_ref[a][0] + inv[1][d] * grad_ref[a][1];
            }
        }
        std::array<FE::GlobalIndex, 3> dof{};
        for (std::size_t a = 0; a < 3u; ++a) {
            const auto dofs = u_map->getVertexDofs(map.vertices[a]);
            ASSERT_EQ(dofs.size(), 1u);
            dof[a] = u_offset + dofs.front();
        }
        for (const auto& qp : rule.points) {
            const std::array<FE::Real, 3> N{{
                FE::Real{1.0} - qp.point[0] - qp.point[1], qp.point[0], qp.point[1]}};
            const FE::Real w = qp.weight * std::abs(det);
            for (std::size_t a = 0; a < 3u; ++a) {
                for (std::size_t b = 0; b < 3u; ++b) {
                    expected[static_cast<std::size_t>(dof[a] * n_dofs + dof[b])] +=
                        w * (N[a] * N[b] + grad[a][0] * grad[b][0] + grad[a][1] * grad[b][1]);
                }
            }
            ++evaluated_points;
        }
    }
    ASSERT_GT(evaluated_points, 0u);
    FE::Real max_difference{0.0};
    for (FE::GlobalIndex row = 0; row < n_dofs; ++row) {
        for (FE::GlobalIndex col = 0; col < n_dofs; ++col) {
            const FE::Real difference = std::abs(
                assembled.getMatrixEntry(row, col) -
                expected[static_cast<std::size_t>(row * n_dofs + col)]);
            max_difference = std::max(max_difference, difference);
            EXPECT_LT(difference, FE::Real{1.0e-13}) << row << "," << col;
        }
    }
    RecordProperty("full_cell_snapshot_rules", static_cast<int>(full_rules.size()));
    RecordProperty("full_cell_tabulation_max_difference",
                   ::testing::PrintToString(max_difference));
#endif
}

TEST(LevelSetVolume, HandlesUncutCells)
{
    const ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    std::vector<FE::Real> coefficients(
        static_cast<std::size_t>(field_dofs.getNumDofs()), -1.0);

    const auto result = level_set::computeLevelSetCutCellVolume(
        *fixture.mesh,
        field_dofs,
        level_set::LevelSetVolumeOptions{},
        coefficients);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.cells, 1u);
    EXPECT_EQ(result.cut_cells, 0u);
    EXPECT_EQ(result.full_negative_cells, 1u);
    EXPECT_EQ(result.full_positive_cells, 0u);
    EXPECT_NEAR(result.total_volume, 1.0 / 6.0, 1.0e-12);
    EXPECT_NEAR(result.negative_volume, 1.0 / 6.0, 1.0e-12);
    EXPECT_NEAR(result.positive_volume, 0.0, 1.0e-12);
}

TEST(LevelSetVolume, CutCellVolumeHandlesTinyTetraFragmentWithoutActivePatch)
{
    const ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto* entity_map = field_dofs.getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);

    std::vector<FE::Real> coefficients(
        static_cast<std::size_t>(field_dofs.getNumDofs()), FE::Real{0.0});
    const std::array<FE::Real, 4> values{{
        FE::Real{-1.0e-9},
        FE::Real{1.0},
        FE::Real{1.0},
        FE::Real{1.0},
    }};
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto dofs = entity_map->getVertexDofs(vertex);
        ASSERT_EQ(dofs.size(), 1u);
        coefficients[static_cast<std::size_t>(dofs.front())] =
            values[static_cast<std::size_t>(vertex)];
    }

    level_set::LevelSetVolumeOptions volume_opts{};
    volume_opts.tolerance = 1.0e-12;
    const auto result = level_set::computeLevelSetCutCellVolume(
        *fixture.mesh,
        field_dofs,
        volume_opts,
        coefficients);

    ASSERT_TRUE(result.success) << result.diagnostic;
    const FE::Real ratio = FE::Real{1.0e-9} / (FE::Real{1.0} + FE::Real{1.0e-9});
    const FE::Real expected = (FE::Real{1.0} / FE::Real{6.0}) *
                              ratio * ratio * ratio;
    EXPECT_NEAR(result.negative_volume, expected, 1.0e-30);
    EXPECT_NEAR(result.positive_volume,
                FE::Real{1.0} / FE::Real{6.0} - expected,
                1.0e-14);
}

