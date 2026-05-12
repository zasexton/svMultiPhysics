/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"
#include "Physics/Materials/Fluid/CarreauYasudaViscosity.h"

#include "Analysis/ConstitutiveLawMetadata.h"
#include "FE/Backends/Interfaces/BackendFactory.h"
#include "FE/Backends/Interfaces/BackendKind.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Assembly/Assembler.h"
#include "FE/Spaces/SpaceFactory.h"
#include "FE/Systems/FESystem.h"

#include <array>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace svmp {
namespace Physics {
namespace test {

namespace {

class TwoQuadStripMeshAccess final : public FE::assembly::IMeshAccess {
public:
    struct BoundaryFace {
        FE::GlobalIndex cell_id{FE::INVALID_GLOBAL_INDEX};
        FE::LocalIndex local_face{FE::INVALID_LOCAL_INDEX};
        int marker{-1};
    };

    TwoQuadStripMeshAccess()
    {
        // Two stacked quads on reference domain [-1,1]x[-1,1]:
        //
        //  5 ----- 4   (top marker=3)
        //  |   1   |
        //  3 ----- 2
        //  |   0   |
        //  0 ----- 1   (bottom marker=2)
        //
        // left marker=1 on edges (0-3) and (3-5)
        // right marker=4 on edges (1-2) and (2-4)
        nodes_ = {
            {-1.0, -1.0, 0.0}, // 0
            {1.0, -1.0, 0.0},  // 1
            {1.0, 0.0, 0.0},   // 2
            {-1.0, 0.0, 0.0},  // 3
            {1.0, 1.0, 0.0},   // 4
            {-1.0, 1.0, 0.0}   // 5
        };

        cells_[0] = {0, 1, 2, 3};
        cells_[1] = {3, 2, 4, 5};

        boundary_faces_ = {
            // bottom cell boundaries
            BoundaryFace{.cell_id = 0, .local_face = 0, .marker = 2}, // bottom
            BoundaryFace{.cell_id = 0, .local_face = 1, .marker = 4}, // right (lower)
            BoundaryFace{.cell_id = 0, .local_face = 3, .marker = 1}, // left (lower)

            // top cell boundaries
            BoundaryFace{.cell_id = 1, .local_face = 1, .marker = 4}, // right (upper)
            BoundaryFace{.cell_id = 1, .local_face = 2, .marker = 3}, // top
            BoundaryFace{.cell_id = 1, .local_face = 3, .marker = 1}  // left (upper)
        };
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 2; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override
    {
        return static_cast<FE::GlobalIndex>(boundary_faces_.size());
    }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }

    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex /*cell_id*/) const override { return FE::ElementType::Quad4; }

    void getCellNodes(FE::GlobalIndex cell_id, std::vector<FE::GlobalIndex>& nodes) const override
    {
        FE_THROW_IF(cell_id < 0 || cell_id >= numCells(), FE::InvalidArgumentException,
                    "TwoQuadStripMeshAccess::getCellNodes: invalid cell id");
        const auto& c = cells_.at(static_cast<std::size_t>(cell_id));
        nodes.assign(c.begin(), c.end());
    }

    [[nodiscard]] std::array<FE::Real, 3> getNodeCoordinates(FE::GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(FE::GlobalIndex cell_id,
                            std::vector<std::array<FE::Real, 3>>& coords) const override
    {
        FE_THROW_IF(cell_id < 0 || cell_id >= numCells(), FE::InvalidArgumentException,
                    "TwoQuadStripMeshAccess::getCellCoordinates: invalid cell id");
        const auto& c = cells_.at(static_cast<std::size_t>(cell_id));
        coords.resize(c.size());
        for (std::size_t i = 0; i < c.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(c[i]));
        }
    }

    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(FE::GlobalIndex face_id,
                                                   FE::GlobalIndex /*cell_id*/) const override
    {
        const auto& f = boundary_faces_.at(static_cast<std::size_t>(face_id));
        return f.local_face;
    }

    [[nodiscard]] int getBoundaryFaceMarker(FE::GlobalIndex face_id) const override
    {
        const auto& f = boundary_faces_.at(static_cast<std::size_t>(face_id));
        return f.marker;
    }

    [[nodiscard]] std::pair<FE::GlobalIndex, FE::GlobalIndex>
    getInteriorFaceCells(FE::GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        for (FE::GlobalIndex c = 0; c < numCells(); ++c) {
            callback(c);
        }
    }

    void forEachOwnedCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        forEachCell(std::move(callback));
    }

    void forEachBoundaryFace(int marker,
                             std::function<void(FE::GlobalIndex, FE::GlobalIndex)> callback) const override
    {
        for (FE::GlobalIndex f = 0; f < numBoundaryFaces(); ++f) {
            const auto& bf = boundary_faces_.at(static_cast<std::size_t>(f));
            if (marker >= 0 && bf.marker != marker) {
                continue;
            }
            callback(f, bf.cell_id);
        }
    }

    void forEachInteriorFace(std::function<void(FE::GlobalIndex, FE::GlobalIndex, FE::GlobalIndex)> /*callback*/) const override
    {
    }

private:
    std::vector<std::array<FE::Real, 3>> nodes_{};
    std::array<std::array<FE::GlobalIndex, 4>, 2> cells_{};
    std::vector<BoundaryFace> boundary_faces_{};
};

[[nodiscard]] FE::dofs::MeshTopologyInfo makeTwoQuadStripTopology()
{
    FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 2;
    topo.n_vertices = 6;
    topo.n_edges = 0;
    topo.n_faces = 0;
    topo.dim = 2;

    topo.cell2vertex_offsets = {0, 4, 8};
    topo.cell2vertex_data = {0, 1, 2, 3,
                             3, 2, 4, 5};

    topo.vertex_gids = {0, 1, 2, 3, 4, 5};
    topo.cell_gids = {0, 1};
    topo.cell_owner_ranks = {0, 0};

    return topo;
}

std::size_t countConstrainedPressureDofs(const FE::systems::FESystem& system, const std::string& pressure_field_name)
{
    const auto& constraints = system.constraints();
    const auto p_dofs = system.fieldMap().getComponentDofs(pressure_field_name, /*component=*/0);
    std::size_t constrained = 0;
    for (const auto dof : p_dofs) {
        if (!constraints.isConstrained(dof)) {
            continue;
        }
        ++constrained;
        const auto c = constraints.getConstraint(dof);
        EXPECT_TRUE(c.has_value());
        if (c.has_value()) {
            EXPECT_TRUE(c->isDirichlet());
            EXPECT_NEAR(c->inhomogeneity, 0.0, 1e-14);
        }
    }
    return constrained;
}

} // namespace

TEST(NavierStokesPressureGauge, PressureNotPinnedWhenUnconstrainedBoundaryExists)
{
    auto mesh = std::make_shared<TwoQuadStripMeshAccess>();

    auto u_space = FE::spaces::VectorSpace(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/2);
    auto p_space = FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    formulations::navier_stokes::IncompressibleNavierStokesVMSOptions opts;
    opts.velocity_field_name = "u";
    opts.pressure_field_name = "p";
    opts.enable_convection = false;
    opts.enable_vms = false;
    opts.density = 1.0;
    opts.viscosity = 0.001;

    // Strong velocity Dirichlet on left/bottom/top, but NOT on the right boundary marker (4).
    opts.velocity_dirichlet = {
        {.boundary_marker = 1},
        {.boundary_marker = 2},
        {.boundary_marker = 3}
    };

    FE::systems::FESystem system(mesh);
    formulations::navier_stokes::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    FE::systems::SetupInputs inputs;
    inputs.topology_override = makeTwoQuadStripTopology();
    system.setup({}, inputs);

    EXPECT_EQ(countConstrainedPressureDofs(system, /*pressure_field_name=*/"p"), 0u);
}

TEST(NavierStokesInitialConditions, HydrostaticPressureInitializationFillsPressureVertices)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "Hydrostatic initialization test requires the Eigen backend.";
#else
    auto mesh = std::make_shared<TwoQuadStripMeshAccess>();

    auto u_space = FE::spaces::VectorSpace(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/2);
    auto p_space = FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    formulations::navier_stokes::IncompressibleNavierStokesVMSOptions opts;
    opts.velocity_field_name = "u";
    opts.pressure_field_name = "p";
    opts.enable_convection = false;
    opts.enable_vms = false;
    opts.density = 2.0;
    opts.viscosity = 0.001;
    opts.body_force = {0.0, -9.81, 0.0};
    opts.hydrostatic_pressure_initialization.enabled = true;
    opts.hydrostatic_pressure_initialization.reference_point = {0.0, 1.0, 0.0};
    opts.hydrostatic_pressure_initialization.reference_pressure = 100.0;

    FE::systems::FESystem system(mesh);
    formulations::navier_stokes::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    FE::systems::SetupInputs inputs;
    inputs.topology_override = makeTwoQuadStripTopology();
    system.setup({}, inputs);

    auto factory = FE::backends::BackendFactory::create(FE::backends::BackendKind::Eigen);
    auto state = factory->createVector(system.dofHandler().getNumDofs());
    state->zero();

    module.applyInitialConditions(system, *state);
    const auto values = state->localSpan();

    const auto p_id = system.findFieldByName("p");
    ASSERT_NE(p_id, FE::INVALID_FIELD_ID);
    const auto* entity_map = system.fieldDofHandler(p_id).getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);
    const auto pressure_offset = system.fieldDofOffset(p_id);

    for (FE::GlobalIndex vertex = 0; vertex < mesh->numVertices(); ++vertex) {
        const auto vertex_dofs = entity_map->getVertexDofs(vertex);
        ASSERT_EQ(vertex_dofs.size(), 1u);
        const auto x = mesh->getNodeCoordinates(vertex);
        const auto expected = 100.0 + 2.0 * (-9.81) * (x[1] - 1.0);
        const auto dof = pressure_offset + vertex_dofs.front();
        ASSERT_GE(dof, 0);
        ASSERT_LT(static_cast<std::size_t>(dof), values.size());
        EXPECT_NEAR(values[static_cast<std::size_t>(dof)], expected, 1.0e-12);
    }
#endif
}

TEST(NavierStokesPressureGauge, NodePressureConstraintPinsSelectedPressureVertex)
{
    auto mesh = std::make_shared<TwoQuadStripMeshAccess>();

    auto u_space = FE::spaces::VectorSpace(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/2);
    auto p_space = FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    formulations::navier_stokes::IncompressibleNavierStokesVMSOptions opts;
    opts.velocity_field_name = "u";
    opts.pressure_field_name = "p";
    opts.enable_convection = false;
    opts.enable_vms = false;
    opts.density = 1.0;
    opts.viscosity = 0.001;
    opts.node_pressure_constraints.id_type =
        formulations::navier_stokes::IncompressibleNavierStokesVMSOptions::
            NodePressureConstraintIdType::LocalVertexId;
    opts.node_pressure_constraints.values = {
        formulations::navier_stokes::IncompressibleNavierStokesVMSOptions::
            NodePressureConstraint{.node_id = 4, .pressure = 2.5}};

    FE::systems::FESystem system(mesh);
    formulations::navier_stokes::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    FE::systems::SetupInputs inputs;
    inputs.topology_override = makeTwoQuadStripTopology();
    system.setup({}, inputs);

    const auto p_id = system.findFieldByName("p");
    ASSERT_NE(p_id, FE::INVALID_FIELD_ID);
    const auto* entity_map = system.fieldDofHandler(p_id).getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);
    const auto vertex_dofs = entity_map->getVertexDofs(4);
    ASSERT_EQ(vertex_dofs.size(), 1u);

    const auto dof = system.fieldDofOffset(p_id) + vertex_dofs.front();
    EXPECT_TRUE(system.constraints().isConstrained(dof));
    EXPECT_NEAR(system.constraints().getInhomogeneity(dof), 2.5, 1.0e-12);
}

TEST(NavierStokesPressureGauge, PublishesDynamicViscosityConstitutiveMetadata)
{
    auto mesh = std::make_shared<TwoQuadStripMeshAccess>();

    auto u_space = FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1,
        mesh,
        /*order=*/1,
        /*components=*/2);
    auto p_space = FE::spaces::Space(
        FE::spaces::SpaceType::H1,
        mesh,
        /*order=*/1,
        /*components=*/1);

    formulations::navier_stokes::IncompressibleNavierStokesVMSOptions opts;
    opts.velocity_field_name = "u";
    opts.pressure_field_name = "p";
    opts.enable_convection = false;
    opts.enable_vms = false;
    opts.density = 1.0;
    opts.viscosity = 0.007;

    FE::systems::FESystem system(mesh);
    formulations::navier_stokes::IncompressibleNavierStokesVMSModule module(
        u_space,
        p_space,
        opts);
    module.registerOn(system);

    ASSERT_FALSE(system.formulationRecords().empty());
    const auto& record = system.formulationRecords().back();
    ASSERT_EQ(record.constitutive_laws.size(), 1u);
    const auto& law = record.constitutive_laws.front();
    EXPECT_EQ(law.name, "dynamic_viscosity");
    EXPECT_EQ(law.role, FE::analysis::ConstitutiveLawRole::DynamicViscosity);
    EXPECT_EQ(law.input_measure,
              FE::analysis::ConstitutiveLawInputMeasure::
                  SymmetricGradientSecondInvariant);
    EXPECT_EQ(law.primary_field, system.findFieldByName("u"));
    EXPECT_TRUE(law.constant_value_available);
    EXPECT_NEAR(law.constant_value, 0.007, 1e-14);
    EXPECT_EQ(law.model, nullptr);
    EXPECT_EQ(law.source_operator_tag, "equations");
}

TEST(NavierStokesPressureGauge, PublishesVariableDynamicViscosityFromResidualExpression)
{
    auto mesh = std::make_shared<TwoQuadStripMeshAccess>();
    auto u_space = FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1,
        mesh,
        /*order=*/1,
        /*components=*/2);
    auto p_space = FE::spaces::Space(
        FE::spaces::SpaceType::H1,
        mesh,
        /*order=*/1,
        /*components=*/1);

    auto viscosity_model =
        std::make_shared<materials::fluid::CarreauYasudaViscosity>(
            0.16,
            0.0035,
            8.2,
            0.2128,
            0.64);

    formulations::navier_stokes::IncompressibleNavierStokesVMSOptions opts;
    opts.velocity_field_name = "u";
    opts.pressure_field_name = "p";
    opts.enable_convection = false;
    opts.enable_vms = false;
    opts.density = 1.0;
    opts.viscosity_model = viscosity_model;

    FE::systems::FESystem system(mesh);
    formulations::navier_stokes::IncompressibleNavierStokesVMSModule module(
        u_space,
        p_space,
        opts);
    module.registerOn(system);

    ASSERT_FALSE(system.formulationRecords().empty());
    const auto& record = system.formulationRecords().back();
    ASSERT_EQ(record.constitutive_laws.size(), 1u);
    const auto& law = record.constitutive_laws.front();
    EXPECT_EQ(law.name, "dynamic_viscosity");
    EXPECT_EQ(law.role, FE::analysis::ConstitutiveLawRole::DynamicViscosity);
    EXPECT_EQ(law.input_measure,
              FE::analysis::ConstitutiveLawInputMeasure::
                  SymmetricGradientSecondInvariant);
    EXPECT_EQ(law.primary_field, system.findFieldByName("u"));
    EXPECT_FALSE(law.constant_value_available);
    EXPECT_EQ(law.model, viscosity_model);
    EXPECT_EQ(law.source_operator_tag, "equations");
}

TEST(NavierStokesPressureGauge, PressurePinnedWhenVelocityIsEssentialOnAllBoundaryMarkers)
{
    auto mesh = std::make_shared<TwoQuadStripMeshAccess>();

    auto u_space = FE::spaces::VectorSpace(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/2);
    auto p_space = FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    formulations::navier_stokes::IncompressibleNavierStokesVMSOptions opts;
    opts.velocity_field_name = "u";
    opts.pressure_field_name = "p";
    opts.enable_convection = false;
    opts.enable_vms = false;
    opts.density = 1.0;
    opts.viscosity = 0.001;

    // Strong velocity Dirichlet on ALL boundary markers => pressure has a constant nullspace
    // unless explicitly constrained.
    opts.velocity_dirichlet = {
        {.boundary_marker = 1},
        {.boundary_marker = 2},
        {.boundary_marker = 3},
        {.boundary_marker = 4}
    };

    FE::systems::FESystem system(mesh);
    formulations::navier_stokes::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    FE::systems::SetupInputs inputs;
    inputs.topology_override = makeTwoQuadStripTopology();
    system.setup({}, inputs);

    EXPECT_EQ(countConstrainedPressureDofs(system, /*pressure_field_name=*/"p"), 1u);
}

} // namespace test
} // namespace Physics
} // namespace svmp
