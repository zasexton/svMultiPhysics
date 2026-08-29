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

std::shared_ptr<svmp::Mesh> buildSingleTriangleMesh()
{
    constexpr std::array<std::array<FE::Real, 3>, 3> coordinates = {{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
    }};
    return buildSingleCellMesh(
        /*spatial_dim=*/2, coordinates, svmp::CellFamily::Triangle);
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

TEST(LevelSetVolume, GeneratedInterfaceGlobalShiftRejectsHighOrderInteriorIsland)
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

    level_set::LevelSetVolumeOptions generated_opts{};
    generated_opts.use_generated_interface_quadrature = true;
    generated_opts.level_set_field_name = "phi";
    generated_opts.generated_domain_id = "interior_island_volume_correction";
    generated_opts.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    generated_opts.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::SayeHyperrectangle;
    generated_opts.implicit_cut_fallback_policy =
        level_set::ImplicitCutFallbackPolicy::Fail;
    generated_opts.interface_quadrature_order = 1;
    generated_opts.volume_quadrature_order = 2;
    generated_opts.implicit_cut_max_subdivision_depth = 8;
    generated_opts.require_production_qualified_implicit_cut_backend = true;

    auto unqualified_measure_opts = generated_opts;
    unqualified_measure_opts.require_production_qualified_implicit_cut_backend =
        false;
    const auto initial = level_set::computeLevelSetCutCellVolume(
        system,
        phi,
        unqualified_measure_opts,
        solution);
    ASSERT_TRUE(initial.success) << initial.diagnostic;
    ASSERT_GT(initial.negative_volume, 0.05);

    level_set::LevelSetGlobalShiftCorrectionOptions correction_opts{};
    correction_opts.target_negative_volume =
        FE::Real{0.5} * initial.negative_volume;
    correction_opts.volume_tolerance = 5.0e-4;
    correction_opts.max_iterations = 80;

    std::vector<FE::Real> corrected_solution;
    EXPECT_THROW(
        (void)level_set::applyGlobalLevelSetShiftCorrection(
            system,
            phi,
            generated_opts,
            correction_opts,
            solution,
            corrected_solution),
        std::invalid_argument);
    EXPECT_TRUE(corrected_solution.empty());
#endif
}

TEST(LevelSetVolume, GeneratedInterfaceGlobalShiftUsesGeneratedQuadrature)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    auto mesh = buildSingleTriangleMesh();
    auto pressure_space =
        std::make_shared<FE::spaces::H1Space>(FE::ElementType::Triangle3,
                                              /*order=*/1);
    auto phi_space =
        std::make_shared<FE::spaces::H1Space>(FE::ElementType::Triangle3,
                                              /*order=*/1);

    FE::systems::FESystem system(mesh);
    const auto pressure = system.addField(FE::systems::FieldSpec{
        .name = "pressure",
        .space = pressure_space,
        .components = 1,
    });
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = phi_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup());

    const auto n_total_dofs =
        static_cast<std::size_t>(system.dofHandler().getNumDofs());
    const auto pressure_offset =
        static_cast<std::size_t>(system.fieldDofOffset(pressure));
    const auto pressure_count = static_cast<std::size_t>(
        system.fieldDofHandler(pressure).getNumDofs());
    const auto phi_offset =
        static_cast<std::size_t>(system.fieldDofOffset(phi));
    const auto& phi_dofs = system.fieldDofHandler(phi);
    const auto phi_count = static_cast<std::size_t>(phi_dofs.getNumDofs());
    const auto* entity_map = phi_dofs.getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);

    std::vector<FE::Real> field_coefficients(phi_count, FE::Real{0.0});
    for (FE::GlobalIndex vertex = 0; vertex < entity_map->numVertices();
         ++vertex) {
        const auto dofs = entity_map->getVertexDofs(vertex);
        ASSERT_EQ(dofs.size(), 1u);
        const auto x = system.meshAccess().getNodeCoordinates(vertex);
        field_coefficients[static_cast<std::size_t>(dofs.front())] =
            x[0] + x[1] - FE::Real{0.5};
    }

    std::vector<FE::Real> solution(n_total_dofs, FE::Real{7.0});
    std::copy(field_coefficients.begin(),
              field_coefficients.end(),
              solution.begin() + static_cast<std::ptrdiff_t>(phi_offset));
    const auto original_solution = solution;

    level_set::LevelSetVolumeOptions generated_opts{};
    generated_opts.use_generated_interface_quadrature = true;
    generated_opts.level_set_field_name = "phi";
    generated_opts.generated_domain_id = "generated_volume_correction";
    generated_opts.interface_quadrature_order = 2;
    generated_opts.volume_quadrature_order = 2;

    const auto initial = level_set::computeLevelSetCutCellVolume(
        system,
        phi,
        generated_opts,
        solution);
    ASSERT_TRUE(initial.success) << initial.diagnostic;
    EXPECT_EQ(initial.diagnostic, "generated_interface_quadrature");
    EXPECT_NEAR(initial.negative_volume, 0.125, 1.0e-10);

    level_set::LevelSetGlobalShiftCorrectionOptions correction_opts{};
    correction_opts.target_negative_volume = 1.0 / 32.0;
    correction_opts.volume_tolerance = 1.0e-10;
    correction_opts.max_iterations = 80;

    std::vector<FE::Real> corrected_solution;
    const auto correction = level_set::applyGlobalLevelSetShiftCorrection(
        system,
        phi,
        generated_opts,
        correction_opts,
        solution,
        corrected_solution);

    ASSERT_TRUE(correction.success) << correction.diagnostic;
    EXPECT_EQ(correction.initial_volume.diagnostic,
              "generated_interface_quadrature");
    EXPECT_EQ(correction.corrected_volume.diagnostic,
              "generated_interface_quadrature");
    EXPECT_GT(correction.iterations, 0);
    EXPECT_NEAR(correction.applied_shift, 0.25, 1.0e-8);
    EXPECT_NEAR(correction.corrected_negative_volume,
                correction_opts.target_negative_volume,
                correction_opts.volume_tolerance);
    ASSERT_EQ(corrected_solution.size(), solution.size());
    for (std::size_t i = 0; i < pressure_count; ++i) {
        EXPECT_DOUBLE_EQ(corrected_solution[pressure_offset + i],
                         original_solution[pressure_offset + i]);
    }
    for (std::size_t i = 0; i < phi_count; ++i) {
        EXPECT_NEAR(corrected_solution[phi_offset + i],
                    field_coefficients[i] + correction.applied_shift,
                    1.0e-12);
    }
#endif
}

TEST(LevelSetVolume,
     WarpedGeneratedCorrectionTargetEqualsSnapshotConstantOneMeasure)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
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
    const auto offset = static_cast<std::size_t>(system.fieldDofOffset(phi));
    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()),
        FE::Real{0.0});
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto dofs = entity_map->getVertexDofs(vertex);
        ASSERT_EQ(dofs.size(), 1u);
        solution[offset + static_cast<std::size_t>(dofs.front())] =
            coordinates[static_cast<std::size_t>(vertex)][0] -
            FE::Real{0.5};
    }

    level_set::LevelSetVolumeOptions volume_options{};
    volume_options.use_generated_interface_quadrature = true;
    volume_options.level_set_field_name = "phi";
    volume_options.generated_domain_id = "warped_correction_measure";
    volume_options.allow_corner_linearized_geometry = true;
    volume_options.interface_quadrature_order = 4;
    volume_options.volume_quadrature_order = 4;

    const auto initial = level_set::computeLevelSetCutCellVolume(
        system, phi, volume_options, solution);
    ASSERT_TRUE(initial.success) << initial.diagnostic;
    EXPECT_NEAR(initial.negative_volume, FE::Real{0.4}, 1.0e-12);

    level_set::LevelSetGlobalShiftCorrectionOptions correction_options{};
    correction_options.target_negative_volume = FE::Real{0.2};
    correction_options.volume_tolerance = FE::Real{1.0e-11};
    correction_options.max_iterations = 80;
    std::vector<FE::Real> corrected_solution;
    const auto correction = level_set::applyGlobalLevelSetShiftCorrection(
        system,
        phi,
        volume_options,
        correction_options,
        solution,
        corrected_solution);
    ASSERT_TRUE(correction.success) << correction.diagnostic;
    ASSERT_TRUE(correction.target_reached);
    EXPECT_NEAR(correction.corrected_negative_volume,
                correction_options.target_negative_volume,
                correction_options.volume_tolerance);

    level_set::LevelSetGeneratedInterfaceOptions interface_options{};
    interface_options.level_set_field_name = "phi";
    interface_options.domain_id = "warped_correction_measure";
    interface_options.requested_interface_marker = 719;
    interface_options.interface_quadrature_order = 4;
    interface_options.volume_quadrature_order = 4;
    interface_options.allow_corner_linearized_geometry = true;
    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    auto generated = lifecycle.build(
        system, interface_options, corrected_solution);
    ASSERT_TRUE(generated.success) << generated.diagnostic;

    FE::interfaces::FreeSurfaceGeometrySnapshotPolicy snapshot_policy;
    snapshot_policy.require_complete_exterior_boundary_partition = false;
    FE::interfaces::FreeSurfaceGeometryScalarEvaluator scalar;
    scalar.value = [shift = correction.applied_shift](
                       FE::GlobalIndex,
                       const std::array<FE::Real, 3>& xi,
                       const FE::geometry::CutQuadratureProvenance&) {
        return FE::Real{0.5} * xi[0] + shift;
    };
    scalar.reference_gradient = [](
                                    FE::GlobalIndex,
                                    const std::array<FE::Real, 3>&,
                                    const FE::geometry::CutQuadratureProvenance&) {
        return std::array<FE::Real, 3>{{0.5, 0.0, 0.0}};
    };
    const auto snapshot = FE::interfaces::buildFreeSurfaceGeometrySnapshot(
        std::move(generated.domain),
        {},
        {},
        system.meshAccess(),
        snapshot_policy,
        std::move(scalar),
        "warped_correction_measure");
    ASSERT_TRUE(snapshot);

    FE::Real snapshot_constant_one_measure{0.0};
    for (const auto* record : snapshot->retainedRules(
             FE::interfaces::FreeSurfaceGeometryRuleRole::NegativeVolume)) {
        ASSERT_NE(record, nullptr);
        for (const auto& point : record->physical_rule.points) {
            snapshot_constant_one_measure += point.physical_weight;
        }
    }
    EXPECT_NEAR(snapshot_constant_one_measure,
                correction_options.target_negative_volume,
                correction_options.volume_tolerance);
    EXPECT_NEAR(snapshot->ledger().owned_retained_negative_physical_volume,
                snapshot_constant_one_measure,
                1.0e-12);
    EXPECT_NEAR(snapshot->ledger().maximum_constant_moment_error,
                0.0,
                1.0e-12);

    auto assembly_mesh = buildSingleCellMesh(
        /*spatial_dim=*/2, coordinates, svmp::CellFamily::Quad);
    FE::systems::FESystem assembly_system(assembly_mesh);
    const auto constant_field = assembly_system.addField(
        FE::systems::FieldSpec{
            .name = "constant_one",
            .space = phi_space,
            .components = 1,
        });
    assembly_system.addOperator("constant_one_measure");
    const auto state_field = FE::forms::FormExpr::stateField(
        constant_field, *phi_space, "constant_one");
    const auto test_function =
        FE::forms::FormExpr::testFunction(*phi_space, "test_constant_one");
    const auto residual = (state_field * test_function)
                              .dCutVolume(
                                  interface_options.requested_interface_marker,
                                  FE::forms::CutVolumeSide::Negative);
    const auto installed = FE::systems::installFormulation(
        assembly_system,
        "constant_one_measure",
        {constant_field},
        residual);
    ASSERT_FALSE(installed.residual.empty());

    auto cut_context =
        std::make_shared<FE::assembly::CutIntegrationContext>();
    cut_context->addFreeSurfaceGeometrySnapshot(
        snapshot, FE::geometry::CutIntegrationSide::Negative);
    assembly_system.setCutIntegrationContext(cut_context);
    ASSERT_NO_THROW(assembly_system.setup());

    std::vector<FE::Real> constant_state(
        static_cast<std::size_t>(assembly_system.dofHandler().getNumDofs()),
        FE::Real{1.0});
    FE::systems::SystemStateView state;
    state.u = constant_state;
    FE::systems::AssemblyRequest request;
    request.op = "constant_one_measure";
    request.want_matrix = false;
    request.want_vector = true;
    FE::assembly::DenseSystemView assembled(
        assembly_system.dofHandler().getNumDofs());
    assembled.zero();
    const auto assembly =
        assembly_system.assemble(request, state, nullptr, &assembled);
    ASSERT_TRUE(assembly.success) << assembly.error_message;

    FE::Real assembled_constant_one_measure{0.0};
    for (FE::GlobalIndex row = 0;
         row < assembly_system.dofHandler().getNumDofs();
         ++row) {
        assembled_constant_one_measure += assembled.getVectorEntry(row);
    }
    EXPECT_NEAR(assembled_constant_one_measure,
                snapshot_constant_one_measure,
                1.0e-12);
    const FE::Real constant_one_assembly_error =
        std::abs(assembled_constant_one_measure -
                 snapshot_constant_one_measure);
    const FE::Real constant_one_target_error =
        std::abs(snapshot_constant_one_measure -
                 correction_options.target_negative_volume);
    RecordProperty("volume_correction_target",
                   ::testing::PrintToString(
                       correction_options.target_negative_volume));
    RecordProperty("snapshot_constant_one_liquid_measure",
                   ::testing::PrintToString(snapshot_constant_one_measure));
    RecordProperty("assembled_constant_one_liquid_measure",
                   ::testing::PrintToString(assembled_constant_one_measure));
    RecordProperty("constant_one_assembly_error",
                   ::testing::PrintToString(constant_one_assembly_error));
    RecordProperty("constant_one_target_error",
                   ::testing::PrintToString(constant_one_target_error));
#endif
}

TEST(LevelSetVolume, GeneratedInterfaceGlobalShiftReusesLifecycleCacheAcrossBisection)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    auto mesh = buildFourTriangleStripMesh();
    auto phi_space =
        std::make_shared<FE::spaces::H1Space>(FE::ElementType::Triangle3,
                                              /*order=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = phi_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup());

    const auto n_total_dofs =
        static_cast<std::size_t>(system.dofHandler().getNumDofs());
    const auto phi_offset =
        static_cast<std::size_t>(system.fieldDofOffset(phi));
    const auto& phi_dofs = system.fieldDofHandler(phi);
    const auto phi_count = static_cast<std::size_t>(phi_dofs.getNumDofs());
    const auto* entity_map = phi_dofs.getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);

    std::vector<FE::Real> field_coefficients(phi_count, FE::Real{0.0});
    for (FE::GlobalIndex vertex = 0; vertex < entity_map->numVertices();
         ++vertex) {
        const auto dofs = entity_map->getVertexDofs(vertex);
        ASSERT_EQ(dofs.size(), 1u);
        const auto x = system.meshAccess().getNodeCoordinates(vertex);
        field_coefficients[static_cast<std::size_t>(dofs.front())] =
            x[0] - FE::Real{0.5};
    }

    std::vector<FE::Real> solution(n_total_dofs, FE::Real{0.0});
    std::copy(field_coefficients.begin(),
              field_coefficients.end(),
              solution.begin() + static_cast<std::ptrdiff_t>(phi_offset));

    level_set::LevelSetVolumeOptions generated_opts{};
    generated_opts.use_generated_interface_quadrature = true;
    generated_opts.level_set_field_name = "phi";
    generated_opts.generated_domain_id = "generated_volume_cache_correction";
    generated_opts.requested_interface_marker = 718;
    generated_opts.interface_quadrature_order = 1;
    generated_opts.volume_quadrature_order = 1;

    const auto initial = level_set::computeLevelSetCutCellVolume(
        system,
        phi,
        generated_opts,
        solution);
    ASSERT_TRUE(initial.success) << initial.diagnostic;
    EXPECT_EQ(initial.cells, 4u);
    EXPECT_EQ(initial.generated_cell_cache_hits, 0u);
    EXPECT_EQ(initial.generated_cell_cache_misses, 4u);
    EXPECT_NEAR(initial.negative_volume, 0.5, 1.0e-12);

    level_set::LevelSetGlobalShiftCorrectionOptions correction_opts{};
    correction_opts.target_negative_volume = 0.25;
    correction_opts.volume_tolerance = 1.0e-12;
    correction_opts.max_iterations = 80;

    std::vector<FE::Real> corrected_solution;
    const auto correction = level_set::applyGlobalLevelSetShiftCorrection(
        system,
        phi,
        generated_opts,
        correction_opts,
        solution,
        corrected_solution);

    ASSERT_TRUE(correction.success) << correction.diagnostic;
    EXPECT_GT(correction.iterations, 0);
    EXPECT_GT(correction.generated_volume_measurement_count, 1u);
    EXPECT_GT(correction.generated_cell_cache_hits, 0u);
    EXPECT_LT(correction.generated_cell_cache_misses,
              correction.generated_volume_measurement_count *
                  correction.initial_volume.cells);
    EXPECT_GT(correction.generated_linear_full_cell_fast_path_count, 0u);
    EXPECT_GT(correction.corrected_volume.generated_value_revision,
              correction.initial_volume.generated_value_revision);
    EXPECT_NEAR(correction.applied_shift, 0.25, 1.0e-8);
    EXPECT_NEAR(correction.corrected_negative_volume,
                correction_opts.target_negative_volume,
                correction_opts.volume_tolerance);
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

TEST(LevelSetVolume, GlobalShiftCorrectionMatchesTargetVolume)
{
    const ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto coefficients = planeCoefficients(fixture, FE::Real{0.5});

    level_set::LevelSetGlobalShiftCorrectionOptions correction_opts{};
    correction_opts.target_negative_volume = 1.0 / 384.0;
    correction_opts.volume_tolerance = 1.0e-12;
    correction_opts.max_iterations = 80;

    std::vector<FE::Real> corrected;
    const auto result = level_set::applyGlobalLevelSetShiftCorrection(
        *fixture.mesh,
        field_dofs,
        level_set::LevelSetVolumeOptions{},
        correction_opts,
        coefficients,
        corrected);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_GT(result.iterations, 0);
    EXPECT_NEAR(result.applied_shift, 0.25, 1.0e-8);
    EXPECT_NEAR(result.initial_negative_volume, 1.0 / 48.0, 1.0e-12);
    EXPECT_GT(std::abs(result.initial_negative_volume -
                       result.target_negative_volume),
              correction_opts.volume_tolerance);
    EXPECT_NEAR(result.corrected_negative_volume,
                correction_opts.target_negative_volume,
                correction_opts.volume_tolerance);
    EXPECT_NEAR(result.volume_error, 0.0, correction_opts.volume_tolerance);
    EXPECT_DOUBLE_EQ(result.max_contact_angle_change_radians, 0.0);
    EXPECT_TRUE(result.negative_component_topology_preserved);
    ASSERT_EQ(result.negative_component_volume_transfers.size(), 1u);
    EXPECT_EQ(result.negative_component_volume_transfers.front()
                  .component_global_vertex_id,
              0);
    EXPECT_NEAR(result.negative_component_volume_transfers.front()
                    .initial_negative_volume,
                result.initial_negative_volume,
                1.0e-12);
    EXPECT_NEAR(result.negative_component_volume_transfers.front()
                    .corrected_negative_volume,
                result.corrected_negative_volume,
                1.0e-12);
    EXPECT_NEAR(result.total_component_volume_transfer,
                result.corrected_negative_volume -
                    result.initial_negative_volume,
                1.0e-12);
    EXPECT_NEAR(result.total_absolute_component_volume_transfer,
                std::abs(result.total_component_volume_transfer),
                1.0e-12);
    EXPECT_NEAR(result.maximum_absolute_component_volume_transfer,
                result.total_absolute_component_volume_transfer,
                1.0e-12);
    ASSERT_EQ(corrected.size(), coefficients.size());
    for (std::size_t i = 0; i < coefficients.size(); ++i) {
        EXPECT_NEAR(corrected[i], coefficients[i] + result.applied_shift, 1.0e-12);
    }
}

TEST(LevelSetVolume,
     GlobalShiftReportsTopologyStableDisconnectedComponentTransfers)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    auto mesh = buildFourTriangleStripMesh();
    auto phi_space =
        std::make_shared<FE::spaces::H1Space>(FE::ElementType::Triangle3,
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
    std::vector<FE::Real> coefficients(
        static_cast<std::size_t>(field_dofs.getNumDofs()), FE::Real{0.0});
    const std::array<FE::Real, 6> vertex_values{{
        FE::Real{-1.0},
        FE::Real{1.0},
        FE::Real{-1.0},
        FE::Real{-1.0},
        FE::Real{1.0},
        FE::Real{-1.0},
    }};
    for (FE::GlobalIndex vertex = 0; vertex < 6; ++vertex) {
        const auto dofs = entity_map->getVertexDofs(vertex);
        ASSERT_EQ(dofs.size(), 1u);
        coefficients[static_cast<std::size_t>(dofs.front())] =
            vertex_values[static_cast<std::size_t>(vertex)];
    }
    auto target_coefficients = coefficients;
    for (auto& value : target_coefficients) {
        value += FE::Real{0.2};
    }
    const level_set::LevelSetVolumeOptions volume_options{};
    const auto target = level_set::computeLevelSetCutCellVolume(
        system.meshAccess(),
        field_dofs,
        volume_options,
        target_coefficients);
    ASSERT_TRUE(target.success) << target.diagnostic;

    level_set::LevelSetGlobalShiftCorrectionOptions correction_options{};
    correction_options.target_negative_volume = target.negative_volume;
    correction_options.volume_tolerance = FE::Real{1.0e-12};
    correction_options.max_iterations = 80;
    std::vector<FE::Real> corrected;
    const auto result = level_set::applyGlobalLevelSetShiftCorrection(
        system.meshAccess(),
        field_dofs,
        volume_options,
        correction_options,
        coefficients,
        corrected);

    ASSERT_TRUE(result.success) << result.diagnostic;
    ASSERT_TRUE(result.correction_applied);
    EXPECT_NEAR(result.applied_shift, 0.2, 1.0e-10);
    EXPECT_DOUBLE_EQ(result.max_contact_angle_change_radians, 0.0);
    EXPECT_TRUE(result.negative_component_topology_preserved);
    ASSERT_EQ(result.negative_component_volume_transfers.size(), 2u);
    EXPECT_EQ(result.negative_component_volume_transfers[0]
                  .component_global_vertex_id,
              0);
    EXPECT_EQ(result.negative_component_volume_transfers[1]
                  .component_global_vertex_id,
              2);
    FE::Real summed_initial = 0.0;
    FE::Real summed_corrected = 0.0;
    FE::Real summed_transfer = 0.0;
    for (const auto& component :
         result.negative_component_volume_transfers) {
        EXPECT_LT(component.volume_transfer, 0.0);
        summed_initial += component.initial_negative_volume;
        summed_corrected += component.corrected_negative_volume;
        summed_transfer += component.volume_transfer;
    }
    EXPECT_NEAR(summed_initial, result.initial_negative_volume, 1.0e-12);
    EXPECT_NEAR(summed_corrected,
                result.corrected_negative_volume,
                1.0e-12);
    EXPECT_NEAR(summed_transfer,
                result.total_component_volume_transfer,
                1.0e-12);
    EXPECT_NEAR(result.total_absolute_component_volume_transfer,
                -result.total_component_volume_transfer,
                1.0e-12);
    EXPECT_NEAR(result.maximum_absolute_component_volume_transfer,
                FE::Real{0.5} *
                    result.total_absolute_component_volume_transfer,
                1.0e-12);
#endif
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

TEST(LevelSetVolume, GlobalShiftCorrectionRejectsHighOrderSystemAndLowLevelOverloads)
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

    const auto n_total_dofs =
        static_cast<std::size_t>(system.dofHandler().getNumDofs());
    const auto phi_offset =
        static_cast<std::size_t>(system.fieldDofOffset(phi));
    const auto& phi_dofs = system.fieldDofHandler(phi);
    const auto phi_count = static_cast<std::size_t>(phi_dofs.getNumDofs());
    const auto* entity_map = phi_dofs.getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);
    ASSERT_GT(phi_count,
              static_cast<std::size_t>(entity_map->numVertices()));

    std::vector<FE::Real> field_coefficients(phi_count, FE::Real{0.0});
    for (std::size_t i = 0; i < phi_count; ++i) {
        field_coefficients[i] =
            FE::Real{10.0} + static_cast<FE::Real>(i);
    }
    for (FE::GlobalIndex vertex = 0; vertex < entity_map->numVertices();
         ++vertex) {
        const auto dofs = entity_map->getVertexDofs(vertex);
        ASSERT_EQ(dofs.size(), 1u);
        const auto x = system.meshAccess().getNodeCoordinates(vertex);
        field_coefficients[static_cast<std::size_t>(dofs.front())] =
            x[0] + x[1] - FE::Real{0.5};
    }

    std::vector<FE::Real> solution(n_total_dofs, FE::Real{3.0});
    std::copy(field_coefficients.begin(),
              field_coefficients.end(),
              solution.begin() + static_cast<std::ptrdiff_t>(phi_offset));
    level_set::LevelSetGlobalShiftCorrectionOptions correction_opts{};
    correction_opts.target_negative_volume = 1.0 / 32.0;
    correction_opts.volume_tolerance = 1.0e-12;
    correction_opts.max_iterations = 80;

    std::vector<FE::Real> corrected_solution;
    EXPECT_THROW(
        (void)level_set::applyGlobalLevelSetShiftCorrection(
            system,
            phi,
            level_set::LevelSetVolumeOptions{},
            correction_opts,
            solution,
            corrected_solution),
        std::invalid_argument);
    EXPECT_TRUE(corrected_solution.empty());

    std::vector<FE::Real> corrected_field{FE::Real{123.0}};
    EXPECT_THROW(
        (void)level_set::applyGlobalLevelSetShiftCorrection(
            system.meshAccess(),
            phi_dofs,
            level_set::LevelSetVolumeOptions{},
            correction_opts,
            field_coefficients,
            corrected_field),
        std::invalid_argument);
    ASSERT_EQ(corrected_field.size(), 1u);
    EXPECT_DOUBLE_EQ(corrected_field.front(), FE::Real{123.0});
#endif
}

TEST(LevelSetVolume, GlobalShiftCorrectionRejectsTensorProductQ1)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    auto mesh = buildSingleQuadMesh();
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

    const auto& phi_dofs = system.fieldDofHandler(phi);
    const auto* entity_map = phi_dofs.getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);
    std::vector<FE::Real> coefficients(
        static_cast<std::size_t>(phi_dofs.getNumDofs()), FE::Real{0.0});
    for (FE::GlobalIndex vertex = 0; vertex < entity_map->numVertices();
         ++vertex) {
        const auto dofs = entity_map->getVertexDofs(vertex);
        ASSERT_EQ(dofs.size(), 1u);
        const auto x = system.meshAccess().getNodeCoordinates(vertex);
        coefficients[static_cast<std::size_t>(dofs.front())] =
            x[0] - FE::Real{0.5};
    }

    level_set::LevelSetGlobalShiftCorrectionOptions correction_opts{};
    correction_opts.target_negative_volume = FE::Real{0.25};
    std::vector<FE::Real> corrected;
    EXPECT_THROW(
        (void)level_set::applyGlobalLevelSetShiftCorrection(
            system.meshAccess(),
            phi_dofs,
            level_set::LevelSetVolumeOptions{},
            correction_opts,
            coefficients,
            corrected),
        std::invalid_argument);
    EXPECT_TRUE(corrected.empty());
#endif
}

TEST(LevelSetVolume,
     GeneratedGlobalShiftRejectsNonAffineTensorProductQ1)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    auto mesh = buildSingleQuadMesh();
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

    const auto& phi_dofs = system.fieldDofHandler(phi);
    const auto* entity_map = phi_dofs.getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);
    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()),
        FE::Real{0.0});
    const std::array<FE::Real, 4> values{{-1.0, 1.0, 2.0, -1.0}};
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto dofs = entity_map->getVertexDofs(vertex);
        ASSERT_EQ(dofs.size(), 1u);
        solution[static_cast<std::size_t>(system.fieldDofOffset(phi) +
                                          dofs.front())] =
            values[static_cast<std::size_t>(vertex)];
    }

    level_set::LevelSetVolumeOptions volume_options{};
    volume_options.use_generated_interface_quadrature = true;
    volume_options.level_set_field_name = "phi";
    volume_options.generated_domain_id = "nonaffine_q1_correction_rejection";
    volume_options.allow_corner_linearized_geometry = true;
    const auto initial = level_set::computeLevelSetCutCellVolume(
        system, phi, volume_options, solution);
    ASSERT_TRUE(initial.success) << initial.diagnostic;

    level_set::LevelSetGlobalShiftCorrectionOptions correction_options{};
    correction_options.target_negative_volume =
        FE::Real{0.5} * initial.negative_volume;
    correction_options.volume_tolerance = FE::Real{1.0e-12};
    std::vector<FE::Real> corrected;
    EXPECT_THROW(
        (void)level_set::applyGlobalLevelSetShiftCorrection(
            system,
            phi,
            volume_options,
            correction_options,
            solution,
            corrected),
        std::invalid_argument);
    EXPECT_TRUE(corrected.empty() || corrected == solution);
#endif
}

TEST(LevelSetVolume, GlobalShiftCorrectionRejectsInterfaceWithoutStrictCrossing)
{
    const ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    std::vector<FE::Real> coefficients(
        static_cast<std::size_t>(field_dofs.getNumDofs()), FE::Real{1.0});

    level_set::LevelSetGlobalShiftCorrectionOptions correction_opts{};
    correction_opts.target_negative_volume = FE::Real{1.0e-3};
    std::vector<FE::Real> corrected;
    EXPECT_THROW(
        (void)level_set::applyGlobalLevelSetShiftCorrection(
            *fixture.mesh,
            field_dofs,
            level_set::LevelSetVolumeOptions{},
            correction_opts,
            coefficients,
            corrected),
        std::invalid_argument);
}

TEST(LevelSetVolume, GlobalShiftCorrectionRejectsVertexWithinInterfaceTolerance)
{
    const ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto* entity_map = field_dofs.getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);
    std::vector<FE::Real> coefficients(
        static_cast<std::size_t>(field_dofs.getNumDofs()), FE::Real{1.0});
    const auto origin_dofs = entity_map->getVertexDofs(0);
    ASSERT_EQ(origin_dofs.size(), 1u);
    coefficients[static_cast<std::size_t>(origin_dofs.front())] =
        FE::Real{0.5e-12};
    const auto negative_vertex_dofs = entity_map->getVertexDofs(1);
    ASSERT_EQ(negative_vertex_dofs.size(), 1u);
    coefficients[static_cast<std::size_t>(negative_vertex_dofs.front())] =
        FE::Real{-1.0};

    level_set::LevelSetGlobalShiftCorrectionOptions correction_opts{};
    correction_opts.target_negative_volume = FE::Real{1.0e-3};
    std::vector<FE::Real> corrected;
    EXPECT_THROW(
        (void)level_set::applyGlobalLevelSetShiftCorrection(
            *fixture.mesh,
            field_dofs,
            level_set::LevelSetVolumeOptions{},
            correction_opts,
            coefficients,
            corrected),
        std::invalid_argument);
}

TEST(LevelSetVolume, GlobalShiftCorrectionLeavesMatchedVolumeUnchanged)
{
    const ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    std::vector<FE::Real> coefficients(
        static_cast<std::size_t>(field_dofs.getNumDofs()), -1.0);

    level_set::LevelSetGlobalShiftCorrectionOptions correction_opts{};
    correction_opts.target_negative_volume = 1.0 / 6.0;

    std::vector<FE::Real> corrected;
    const auto result = level_set::applyGlobalLevelSetShiftCorrection(
        *fixture.mesh,
        field_dofs,
        level_set::LevelSetVolumeOptions{},
        correction_opts,
        coefficients,
        corrected);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.iterations, 0);
    EXPECT_DOUBLE_EQ(result.applied_shift, 0.0);
    EXPECT_NEAR(result.volume_error, 0.0, correction_opts.volume_tolerance);
    EXPECT_EQ(corrected, coefficients);
}

TEST(LevelSetVolume, GlobalShiftCorrectionRejectsDisabledDisplacementBound)
{
    const ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto coefficients = planeCoefficients(fixture, FE::Real{0.5});

    level_set::LevelSetGlobalShiftCorrectionOptions correction_opts{};
    correction_opts.target_negative_volume = 1.0 / 12.0;
    correction_opts.maximum_interface_displacement_fraction = 0.0;

    std::vector<FE::Real> corrected;
    EXPECT_THROW(
        (void)level_set::applyGlobalLevelSetShiftCorrection(
            *fixture.mesh,
            field_dofs,
            level_set::LevelSetVolumeOptions{},
            correction_opts,
            coefficients,
            corrected),
        std::invalid_argument);
}

TEST(LevelSetVolume, GlobalShiftCorrectionSkipsErrorsBelowFallbackTrigger)
{
    const ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto coefficients = planeCoefficients(fixture, FE::Real{0.5});
    const auto initial = level_set::computeLevelSetCutCellVolume(
        *fixture.mesh,
        field_dofs,
        level_set::LevelSetVolumeOptions{},
        coefficients);
    ASSERT_TRUE(initial.success) << initial.diagnostic;

    level_set::LevelSetGlobalShiftCorrectionOptions correction_opts{};
    correction_opts.target_negative_volume =
        initial.negative_volume - FE::Real{1.0e-5};
    correction_opts.volume_tolerance = 1.0e-12;
    correction_opts.minimum_relative_volume_error = 1.0e-3;

    std::vector<FE::Real> corrected;
    const auto result = level_set::applyGlobalLevelSetShiftCorrection(
        *fixture.mesh,
        field_dofs,
        level_set::LevelSetVolumeOptions{},
        correction_opts,
        coefficients,
        corrected);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_FALSE(result.correction_triggered);
    EXPECT_FALSE(result.correction_applied);
    EXPECT_FALSE(result.target_reached);
    EXPECT_GT(result.trigger_volume_error, std::abs(result.volume_error));
    EXPECT_EQ(corrected, coefficients);
}

TEST(LevelSetVolume, GlobalShiftCorrectionBoundsZeroSetAndContactLineMotion)
{
    const ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto coefficients = planeCoefficients(fixture, FE::Real{0.5});

    level_set::LevelSetGlobalShiftCorrectionOptions correction_opts{};
    correction_opts.target_negative_volume = 1.0 / 384.0;
    correction_opts.volume_tolerance = 1.0e-12;
    correction_opts.max_iterations = 80;
    correction_opts.maximum_interface_displacement_fraction = 0.05;

    std::vector<FE::Real> corrected;
    const auto result = level_set::applyGlobalLevelSetShiftCorrection(
        *fixture.mesh,
        field_dofs,
        level_set::LevelSetVolumeOptions{},
        correction_opts,
        coefficients,
        corrected);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_TRUE(result.correction_triggered);
    EXPECT_TRUE(result.correction_applied);
    EXPECT_TRUE(result.limited_by_displacement_bound);
    EXPECT_FALSE(result.target_reached);
    EXPECT_GT(result.minimum_edge_length, 0.0);
    EXPECT_GT(result.maximum_topology_stable_shift, 0.0);
    EXPECT_LE(std::abs(result.applied_shift),
              result.maximum_topology_stable_shift);
    EXPECT_LE(result.max_interface_displacement,
              result.maximum_allowed_interface_displacement + 1.0e-12);
    EXPECT_LE(result.max_contact_line_displacement,
              result.maximum_allowed_interface_displacement + 1.0e-12);
    EXPECT_DOUBLE_EQ(result.contact_line_displacement_bound,
                     result.max_contact_line_displacement);
    EXPECT_LT(std::abs(result.volume_error),
              std::abs(result.initial_negative_volume -
                       result.target_negative_volume));
}

TEST(LevelSetVolume, ReachableBoundedCorrectionIsNotReportedAsLimited)
{
    const ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto coefficients = planeCoefficients(fixture, FE::Real{0.5});
    auto target_coefficients = coefficients;
    for (auto& value : target_coefficients) {
        value += FE::Real{0.01};
    }
    const auto target = level_set::computeLevelSetCutCellVolume(
        *fixture.mesh,
        field_dofs,
        level_set::LevelSetVolumeOptions{},
        target_coefficients);
    ASSERT_TRUE(target.success) << target.diagnostic;

    level_set::LevelSetGlobalShiftCorrectionOptions correction_opts{};
    correction_opts.target_negative_volume = target.negative_volume;
    correction_opts.volume_tolerance = 1.0e-10;
    correction_opts.max_iterations = 80;
    correction_opts.maximum_interface_displacement_fraction = 0.05;

    std::vector<FE::Real> corrected;
    const auto result = level_set::applyGlobalLevelSetShiftCorrection(
        *fixture.mesh,
        field_dofs,
        level_set::LevelSetVolumeOptions{},
        correction_opts,
        coefficients,
        corrected);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_TRUE(result.target_reached);
    EXPECT_FALSE(result.limited_by_displacement_bound);
    EXPECT_NEAR(result.applied_shift, 0.01, 1.0e-8);
    EXPECT_GT(result.maximum_topology_stable_shift,
              std::abs(result.applied_shift));
    EXPECT_LE(result.max_interface_displacement,
              result.maximum_allowed_interface_displacement + 1.0e-12);
    EXPECT_LE(result.max_contact_line_displacement,
              result.maximum_allowed_interface_displacement + 1.0e-12);
}

TEST(LevelSetVolume, GlobalShiftTopologyBoundPreventsVertexSignChange)
{
    const ScalarFieldFixture fixture;
    const auto& field_dofs = fixture.system.fieldDofHandler(fixture.phi);
    const auto* entity_map = field_dofs.getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);
    std::vector<FE::Real> coefficients(
        static_cast<std::size_t>(field_dofs.getNumDofs()), FE::Real{1.0});
    const auto negative_vertex_dofs = entity_map->getVertexDofs(0);
    ASSERT_EQ(negative_vertex_dofs.size(), 1u);
    const auto negative_dof =
        static_cast<std::size_t>(negative_vertex_dofs.front());
    coefficients[negative_dof] = FE::Real{-0.1};

    level_set::LevelSetVolumeOptions volume_opts{};
    level_set::LevelSetGlobalShiftCorrectionOptions correction_opts{};
    correction_opts.target_negative_volume = FE::Real{0.0};
    correction_opts.volume_tolerance = FE::Real{1.0e-40};
    correction_opts.max_iterations = 80;
    correction_opts.maximum_interface_displacement_fraction = FE::Real{1.0};

    std::vector<FE::Real> corrected;
    const auto result = level_set::applyGlobalLevelSetShiftCorrection(
        *fixture.mesh,
        field_dofs,
        volume_opts,
        correction_opts,
        coefficients,
        corrected);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_TRUE(result.limited_by_displacement_bound);
    EXPECT_FALSE(result.target_reached);
    EXPECT_NEAR(result.maximum_topology_stable_shift,
                FE::Real{0.1} - FE::Real{2.0} * volume_opts.tolerance,
                FE::Real{1.0e-14});
    EXPECT_LE(std::abs(result.applied_shift),
              result.maximum_topology_stable_shift);
    ASSERT_EQ(corrected.size(), coefficients.size());
    EXPECT_LT(corrected[negative_dof], -volume_opts.tolerance);
    for (std::size_t i = 0; i < corrected.size(); ++i) {
        if (i != negative_dof) {
            EXPECT_GT(corrected[i], volume_opts.tolerance);
        }
    }
}

TEST(LevelSetVolume, GlobalShiftReportsAndBoundsWallContactPointMotion)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    auto mesh = buildSingleTriangleMesh();
    auto scalar_space =
        std::make_shared<FE::spaces::H1Space>(FE::ElementType::Triangle3,
                                              /*order=*/1);
    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup());

    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto* entity_map = field_dofs.getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);
    const auto offset = static_cast<std::size_t>(system.fieldDofOffset(phi));
    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < entity_map->numVertices(); ++vertex) {
        const auto dofs = entity_map->getVertexDofs(vertex);
        ASSERT_EQ(dofs.size(), 1u);
        const auto x = system.meshAccess().getNodeCoordinates(vertex);
        solution[offset + static_cast<std::size_t>(dofs.front())] =
            x[0] - FE::Real{0.25};
    }

    level_set::LevelSetGlobalShiftCorrectionOptions correction_opts{};
    correction_opts.target_negative_volume = 0.4;
    correction_opts.volume_tolerance = 1.0e-12;
    correction_opts.max_iterations = 80;
    correction_opts.maximum_interface_displacement_fraction = 0.05;

    std::vector<FE::Real> corrected;
    const auto result = level_set::applyGlobalLevelSetShiftCorrection(
        system,
        phi,
        level_set::LevelSetVolumeOptions{},
        correction_opts,
        solution,
        corrected);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_TRUE(result.limited_by_displacement_bound);
    EXPECT_FALSE(result.target_reached);
    EXPECT_NEAR(result.max_contact_line_displacement, 0.05, 1.0e-10);
    EXPECT_NEAR(result.contact_line_displacement_bound, 0.05, 1.0e-10);
    EXPECT_LE(result.max_contact_line_displacement,
              result.maximum_allowed_interface_displacement + 1.0e-12);
#endif
}

TEST(LevelSetVolume, GlobalShiftReportsAndBoundsThreeDimensionalWallContactLine)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    constexpr std::array<std::array<FE::Real, 3>, 4> coordinates = {{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    }};
    auto mesh = buildSingleCellMesh(/*spatial_dim=*/3,
                                    coordinates,
                                    svmp::CellFamily::Tetra);
    auto scalar_space =
        std::make_shared<FE::spaces::H1Space>(FE::ElementType::Tetra4,
                                              /*order=*/1);
    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup());

    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto* entity_map = field_dofs.getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);
    const auto offset = static_cast<std::size_t>(system.fieldDofOffset(phi));
    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < entity_map->numVertices(); ++vertex) {
        const auto dofs = entity_map->getVertexDofs(vertex);
        ASSERT_EQ(dofs.size(), 1u);
        const auto x = system.meshAccess().getNodeCoordinates(vertex);
        solution[offset + static_cast<std::size_t>(dofs.front())] =
            x[0] - FE::Real{0.25};
    }

    level_set::LevelSetGlobalShiftCorrectionOptions correction_opts{};
    correction_opts.target_negative_volume = FE::Real{1.0} / FE::Real{6.0};
    correction_opts.volume_tolerance = 1.0e-12;
    correction_opts.max_iterations = 80;
    correction_opts.maximum_interface_displacement_fraction = 0.05;

    std::vector<FE::Real> corrected;
    const auto result = level_set::applyGlobalLevelSetShiftCorrection(
        system,
        phi,
        level_set::LevelSetVolumeOptions{},
        correction_opts,
        solution,
        corrected);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_TRUE(result.limited_by_displacement_bound);
    EXPECT_FALSE(result.target_reached);
    EXPECT_GT(result.max_contact_line_displacement, 0.0);
    EXPECT_DOUBLE_EQ(result.contact_line_displacement_bound,
                     result.max_contact_line_displacement);
    EXPECT_LE(result.max_contact_line_displacement,
              result.maximum_allowed_interface_displacement + 1.0e-12);
#endif
}

TEST(LevelSetVolume, VolumeCorrectionUpdatesOutputTimeActiveVolume)
{
    const ScalarFieldFixture fixture;
    const auto coefficients = planeCoefficients(fixture, FE::Real{0.5});
    std::vector<FE::Real> solution(
        static_cast<std::size_t>(fixture.system.dofHandler().getNumDofs()), 0.0);
    const auto offset = static_cast<std::size_t>(fixture.system.fieldDofOffset(fixture.phi));
    std::copy(coefficients.begin(),
              coefficients.end(),
              solution.begin() + static_cast<std::ptrdiff_t>(offset));

    const auto initial_volume = level_set::computeLevelSetCutCellVolume(
        fixture.system,
        fixture.phi,
        level_set::LevelSetVolumeOptions{},
        solution);
    ASSERT_TRUE(initial_volume.success) << initial_volume.diagnostic;

    level_set::LevelSetGlobalShiftCorrectionOptions correction_opts{};
    correction_opts.target_negative_volume = 1.0 / 384.0;
    correction_opts.volume_tolerance = 1.0e-12;
    correction_opts.max_iterations = 80;

    std::vector<FE::Real> corrected_solution;
    const auto correction = level_set::applyGlobalLevelSetShiftCorrection(
        fixture.system,
        fixture.phi,
        level_set::LevelSetVolumeOptions{},
        correction_opts,
        solution,
        corrected_solution);
    ASSERT_TRUE(correction.success) << correction.diagnostic;
    EXPECT_NEAR(correction.corrected_negative_volume,
                correction_opts.target_negative_volume,
                correction_opts.volume_tolerance);

    const auto output_time_volume = level_set::computeLevelSetCutCellVolume(
        fixture.system,
        fixture.phi,
        level_set::LevelSetVolumeOptions{},
        corrected_solution);
    ASSERT_TRUE(output_time_volume.success) << output_time_volume.diagnostic;

    EXPECT_NEAR(initial_volume.negative_volume, 1.0 / 48.0, 1.0e-12);
    EXPECT_NEAR(output_time_volume.negative_volume,
                correction_opts.target_negative_volume,
                correction_opts.volume_tolerance);
    EXPECT_NE(output_time_volume.negative_volume, initial_volume.negative_volume);
}

TEST(LevelSetVolume, VolumeCorrectionRefreshesCutContextBeforeOutput)
{
    const ScalarFieldFixture fixture;
    const auto coefficients = planeCoefficients(fixture, FE::Real{0.5});
    std::vector<FE::Real> solution(
        static_cast<std::size_t>(fixture.system.dofHandler().getNumDofs()), 0.0);
    const auto offset = static_cast<std::size_t>(fixture.system.fieldDofOffset(fixture.phi));
    std::copy(coefficients.begin(),
              coefficients.end(),
              solution.begin() + static_cast<std::ptrdiff_t>(offset));

    level_set::LevelSetGlobalShiftCorrectionOptions correction_opts{};
    correction_opts.target_negative_volume = 1.0 / 384.0;
    correction_opts.volume_tolerance = 1.0e-12;
    correction_opts.max_iterations = 80;

    std::vector<FE::Real> corrected_solution;
    const auto correction = level_set::applyGlobalLevelSetShiftCorrection(
        fixture.system,
        fixture.phi,
        level_set::LevelSetVolumeOptions{},
        correction_opts,
        solution,
        corrected_solution);
    ASSERT_TRUE(correction.success) << correction.diagnostic;

    level_set::LevelSetGeneratedInterfaceOptions interface_opts{};
    interface_opts.level_set_field_name = "phi";
    interface_opts.domain_id = "output-fluid";
    interface_opts.requested_interface_marker = 916;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto stale_context =
        lifecycle.build(fixture.system, interface_opts, solution);
    const auto output_context =
        lifecycle.build(fixture.system, interface_opts, corrected_solution);

    ASSERT_TRUE(stale_context.success) << stale_context.diagnostic;
    ASSERT_TRUE(output_context.success) << output_context.diagnostic;
    EXPECT_EQ(stale_context.interface_marker, output_context.interface_marker);
    EXPECT_NE(stale_context.value_revision, output_context.value_revision);
    EXPECT_NE(stale_context.summary.negative_volume_measure,
              output_context.summary.negative_volume_measure);
    EXPECT_NEAR(stale_context.summary.negative_volume_measure,
                1.0 / 48.0,
                1.0e-12);
    EXPECT_NEAR(output_context.summary.negative_volume_measure,
                correction_opts.target_negative_volume,
                correction_opts.volume_tolerance);
}

TEST(LevelSetVolume, GlobalShiftCorrectionRejectsHighOrderBeforeMatchedVolumeNoOp)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    auto mesh = buildSingleQuadMesh();
    auto scalar_space =
        std::make_shared<FE::spaces::H1Space>(FE::ElementType::Quad4,
                                              /*order=*/2);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system.setup());

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    ASSERT_GE(cell_dofs.size(), 9u);
    const auto offset = system.fieldDofOffset(phi);
    for (std::size_t i = 0; i < 9u; ++i) {
        const auto xi =
            FE::basis::ReferenceNodeLayout::get_node_coords(FE::ElementType::Quad9, i);
        const auto x = FE::Real{0.5} * (xi[0] + FE::Real{1.0});
        const auto y = FE::Real{0.5} * (xi[1] + FE::Real{1.0});
        solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
            x + y - FE::Real{0.5};
    }

    const auto initial = level_set::computeLevelSetCutCellVolume(
        system,
        phi,
        level_set::LevelSetVolumeOptions{},
        solution);
    ASSERT_TRUE(initial.success) << initial.diagnostic;

    level_set::LevelSetGlobalShiftCorrectionOptions correction_opts{};
    correction_opts.target_negative_volume = initial.negative_volume;
    correction_opts.volume_tolerance = 1.0e-12;
    correction_opts.max_iterations = 80;

    std::vector<FE::Real> corrected_solution;
    EXPECT_THROW(
        (void)level_set::applyGlobalLevelSetShiftCorrection(
            system,
            phi,
            level_set::LevelSetVolumeOptions{},
            correction_opts,
            solution,
            corrected_solution),
        std::invalid_argument);
    EXPECT_TRUE(corrected_solution.empty());
#endif
}

TEST(LevelSetVolume, VolumeCorrectionSynchronizesHistoryAndCutContext)
{
    const ScalarFieldFixture fixture;
    const auto coefficients = planeCoefficients(fixture, FE::Real{0.5});
    std::vector<FE::Real> solution(
        static_cast<std::size_t>(fixture.system.dofHandler().getNumDofs()), 0.0);
    const auto offset = static_cast<std::size_t>(fixture.system.fieldDofOffset(fixture.phi));
    std::copy(coefficients.begin(),
              coefficients.end(),
              solution.begin() + static_cast<std::ptrdiff_t>(offset));

    level_set::LevelSetGlobalShiftCorrectionOptions correction_opts{};
    correction_opts.target_negative_volume = 1.0 / 384.0;
    correction_opts.volume_tolerance = 1.0e-12;
    correction_opts.max_iterations = 80;

    std::vector<FE::Real> corrected_solution;
    const auto correction = level_set::applyGlobalLevelSetShiftCorrection(
        fixture.system,
        fixture.phi,
        level_set::LevelSetVolumeOptions{},
        correction_opts,
        solution,
        corrected_solution);
    ASSERT_TRUE(correction.success) << correction.diagnostic;

    const auto accepted_solution = corrected_solution;
    const auto previous_solution = corrected_solution;
    ASSERT_EQ(accepted_solution.size(), previous_solution.size());
    for (std::size_t i = 0; i < accepted_solution.size(); ++i) {
        EXPECT_NEAR(accepted_solution[i], previous_solution[i], 1.0e-15);
    }

    level_set::LevelSetGeneratedInterfaceOptions interface_opts{};
    interface_opts.level_set_field_name = "phi";
    interface_opts.domain_id = "maintained-fluid";
    interface_opts.requested_interface_marker = 812;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto accepted_context =
        lifecycle.build(fixture.system, interface_opts, accepted_solution);
    const auto previous_context =
        lifecycle.build(fixture.system, interface_opts, previous_solution);

    ASSERT_TRUE(accepted_context.success) << accepted_context.diagnostic;
    ASSERT_TRUE(previous_context.success) << previous_context.diagnostic;
    EXPECT_EQ(previous_context.interface_marker, accepted_context.interface_marker);
    EXPECT_EQ(previous_context.domain.marker(), accepted_context.domain.marker());
    EXPECT_EQ(previous_context.value_revision, accepted_context.value_revision + 1u);
    EXPECT_NEAR(accepted_context.summary.negative_volume_measure,
                correction_opts.target_negative_volume,
                correction_opts.volume_tolerance);
    EXPECT_NEAR(previous_context.summary.negative_volume_measure,
                correction_opts.target_negative_volume,
                correction_opts.volume_tolerance);
}

TEST(LevelSetVolume, MaintainedPreviousStateLeavesBDF1ResidualNeutral)
{
    const ScalarFieldFixture fixture;
    const auto coefficients = planeCoefficients(fixture, FE::Real{0.5});
    std::vector<FE::Real> solution(
        static_cast<std::size_t>(fixture.system.dofHandler().getNumDofs()), 0.0);
    const auto offset = static_cast<std::size_t>(fixture.system.fieldDofOffset(fixture.phi));
    std::copy(coefficients.begin(),
              coefficients.end(),
              solution.begin() + static_cast<std::ptrdiff_t>(offset));

    level_set::LevelSetGlobalShiftCorrectionOptions correction_opts{};
    correction_opts.target_negative_volume = 1.0 / 384.0;
    correction_opts.volume_tolerance = 1.0e-12;
    correction_opts.max_iterations = 80;

    std::vector<FE::Real> corrected_solution;
    const auto correction = level_set::applyGlobalLevelSetShiftCorrection(
        fixture.system,
        fixture.phi,
        level_set::LevelSetVolumeOptions{},
        correction_opts,
        solution,
        corrected_solution);
    ASSERT_TRUE(correction.success) << correction.diagnostic;

    const auto accepted_solution = corrected_solution;
    const auto previous_solution = corrected_solution;
    auto older_history = solution;
    for (auto& value : older_history) {
        value -= FE::Real{10.0};
    }

    std::array<std::span<const FE::Real>, 2> history_spans = {
        std::span<const FE::Real>(previous_solution.data(), previous_solution.size()),
        std::span<const FE::Real>(older_history.data(), older_history.size()),
    };
    const std::array<double, 2> dt_history = {0.1, 0.1};
    FE::systems::SystemStateView state{};
    state.dt = 0.1;
    state.dt_prev = 0.1;
    state.u = std::span<const FE::Real>(
        accepted_solution.data(), accepted_solution.size());
    state.u_prev = history_spans[0];
    state.u_prev2 = history_spans[1];
    state.u_history = std::span<const std::span<const FE::Real>>(
        history_spans.data(), history_spans.size());
    state.dt_history = std::span<const double>(
        dt_history.data(), dt_history.size());

    const FE::systems::BDFIntegrator bdf1(1);
    const auto context = bdf1.buildContext(/*max_time_derivative_order=*/1, state);
    ASSERT_TRUE(context.dt1.has_value());
    ASSERT_EQ(context.dt1->a.size(), 2u);

    for (std::size_t i = 0; i < accepted_solution.size(); ++i) {
        const auto derivative =
            context.dt1->a[0] * accepted_solution[i] +
            context.dt1->a[1] * previous_solution[i];
        EXPECT_NEAR(derivative, 0.0, 1.0e-12);
    }

    auto alternate_older_history = older_history;
    for (auto& value : alternate_older_history) {
        value += FE::Real{25.0};
    }
    history_spans[1] = std::span<const FE::Real>(
        alternate_older_history.data(), alternate_older_history.size());
    state.u_prev2 = history_spans[1];
    state.u_history = std::span<const std::span<const FE::Real>>(
        history_spans.data(), history_spans.size());
    const auto alternate_context =
        bdf1.buildContext(/*max_time_derivative_order=*/1, state);
    ASSERT_TRUE(alternate_context.dt1.has_value());
    ASSERT_EQ(alternate_context.dt1->a.size(), context.dt1->a.size());
    EXPECT_EQ(alternate_context.dt1->a, context.dt1->a);
}
