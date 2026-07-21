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
#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Assembly/GlobalSystemView.h"
#include "FE/LevelSet/LevelSetInterfaceLifecycle.h"
#include "FE/Spaces/SpaceFactory.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/TimeIntegrator.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Core/MeshBase.h"
#  include "Mesh/Fields/MeshFields.h"
#  include "Mesh/Mesh.h"
#  include "Mesh/Topology/CellShape.h"
#endif

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
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

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
[[nodiscard]] std::shared_ptr<Mesh> makeTwoQuadStripNativeMeshWithPhi()
{
    auto base = std::make_shared<MeshBase>();

    const std::vector<real_t> x_ref = {
        -1.0, -1.0,
         1.0, -1.0,
         1.0,  0.0,
        -1.0,  0.0,
         1.0,  1.0,
        -1.0,  1.0,
    };
    const std::vector<offset_t> cell2vertex_offsets = {0, 4, 8};
    const std::vector<index_t> cell2vertex = {
        0, 1, 2, 3,
        3, 2, 4, 5,
    };

    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(
        /*spatial_dim=*/2,
        x_ref,
        cell2vertex_offsets,
        cell2vertex,
        std::vector<CellShape>(2, shape));
    base->finalize();

    const auto phi_handle = MeshFields::attach_field(
        *base,
        EntityKind::Vertex,
        "phi",
        FieldScalarType::Float64,
        1);
    auto* phi = MeshFields::field_data_as<real_t>(*base, phi_handle);
    phi[0] = -1.0;
    phi[1] = -1.0;
    phi[2] = 0.25;
    phi[3] = 0.25;
    phi[4] = 1.0;
    phi[5] = 1.0;

    return create_mesh(std::move(base));
}
#endif

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

constexpr int pressure_anchor_interface_marker = 27015;
constexpr const char* pressure_anchor_domain_id =
    "pressure_anchor_measure_guard";

[[nodiscard]] formulations::navier_stokes::
    IncompressibleNavierStokesVMSOptions pressureAnchorOptions()
{
    namespace ns = formulations::navier_stokes;
    ns::IncompressibleNavierStokesVMSOptions options;
    options.velocity_field_name = "u";
    options.pressure_field_name = "p";
    options.enable_convection = false;
    options.enable_vms = false;
    options.density = 1.0;
    options.viscosity = 0.001;
    options.velocity_dirichlet = {
        {.boundary_marker = 1},
        {.boundary_marker = 2},
        {.boundary_marker = 3},
        {.boundary_marker = 4},
    };
    options.free_surface.push_back(
        ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
            .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
            .interface_marker = pressure_anchor_interface_marker,
            .level_set_field_name = "phi",
            .generated_interface_domain_id = pressure_anchor_domain_id,
            .level_set_isovalue = 0.0,
            .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
            .active_domain_method = ns::FreeSurfaceActiveDomainMethod::CutVolume,
            .external_pressure = 0.0,
            .surface_tension = 0.0,
            .use_level_set_curvature = false,
            .small_cut_aggregation = false,
        });
    return options;
}

[[nodiscard]] FE::FieldId registerPressureAnchorProblem(
    FE::systems::FESystem& system,
    const std::shared_ptr<FE::spaces::FunctionSpace>& velocity_space,
    const std::shared_ptr<FE::spaces::FunctionSpace>& pressure_space)
{
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = pressure_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::Unknown,
    });
    if (phi == FE::INVALID_FIELD_ID) {
        throw std::runtime_error(
            "failed to register pressure-anchor level-set field");
    }
    formulations::navier_stokes::IncompressibleNavierStokesVMSModule module(
        velocity_space, pressure_space, pressureAnchorOptions());
    module.registerOn(system);
    return phi;
}

void setPressureAnchorLevelSet(
    std::vector<FE::Real>& solution,
    const FE::systems::FESystem& system,
    FE::FieldId phi,
    const std::function<FE::Real(const std::array<FE::Real, 3>&)>& value)
{
    const auto* entity_map =
        system.fieldDofHandler(phi).getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error(
            "pressure-anchor level-set field has no entity DOF map");
    }
    const auto offset = system.fieldDofOffset(phi);
    for (FE::GlobalIndex vertex = 0;
         vertex < system.meshAccess().numVertices();
         ++vertex) {
        const auto dofs = entity_map->getVertexDofs(vertex);
        if (dofs.size() != 1u) {
            throw std::runtime_error(
                "pressure-anchor level-set field is not scalar P1");
        }
        solution.at(static_cast<std::size_t>(offset + dofs.front())) =
            value(system.meshAccess().getNodeCoordinates(vertex));
    }
}

[[nodiscard]] std::shared_ptr<FE::assembly::CutIntegrationContext>
makePressureAnchorCutContext(
    const FE::systems::FESystem& system,
    std::span<const FE::Real> solution,
    FE::level_set::LevelSetGeneratedInterfaceLifecycle& lifecycle)
{
    FE::level_set::LevelSetGeneratedInterfaceOptions options;
    options.level_set_field_name = "phi";
    options.domain_id = pressure_anchor_domain_id;
    options.requested_interface_marker = pressure_anchor_interface_marker;
    options.tolerance = 1.0e-12;
    options.quadrature_order = 2;
    options.interface_quadrature_order = 1;
    options.volume_quadrature_order = 2;
    const auto generated = lifecycle.build(system, options, solution);
    if (!generated.success) {
        throw std::runtime_error(generated.diagnostic);
    }

    auto context = std::make_shared<FE::assembly::CutIntegrationContext>();
    context->addGeneratedInterfaceDomain(
        generated.domain, FE::geometry::CutIntegrationSide::Negative);
    return context;
}

template <typename Action>
[[nodiscard]] std::string captureRuntimeError(Action&& action)
{
    try {
        std::forward<Action>(action)();
    } catch (const std::runtime_error& error) {
        return error.what();
    } catch (const std::exception& error) {
        ADD_FAILURE() << "expected std::runtime_error, got: " << error.what();
        return error.what();
    } catch (...) {
        ADD_FAILURE() << "expected std::runtime_error, got a non-standard exception";
        return {};
    }
    ADD_FAILURE() << "expected std::runtime_error, but no exception was thrown";
    return {};
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

TEST(NavierStokesPressureGauge,
     ActiveCutVolumeFreeSurfaceAnchorsAbsolutePressureWithoutInteriorPin)
{
    auto mesh = std::make_shared<TwoQuadStripMeshAccess>();
    auto u_space = FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/2);
    auto p_space = FE::spaces::Space(
        FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);

    FE::systems::FESystem system(mesh);
    ASSERT_NE(system.addField(FE::systems::FieldSpec{
                  .name = "phi",
                  .space = p_space,
                  .components = 1,
                  .source_kind = FE::systems::FieldSourceKind::PrescribedData,
              }),
              FE::INVALID_FIELD_ID);

    formulations::navier_stokes::IncompressibleNavierStokesVMSOptions opts;
    opts.velocity_field_name = "u";
    opts.pressure_field_name = "p";
    opts.enable_convection = false;
    opts.enable_vms = false;
    opts.density = 1.0;
    opts.viscosity = 0.001;
    opts.velocity_dirichlet = {
        {.boundary_marker = 1},
        {.boundary_marker = 2},
        {.boundary_marker = 3},
        {.boundary_marker = 4},
    };
    opts.free_surface.push_back(
        formulations::navier_stokes::IncompressibleNavierStokesVMSOptions::
            FreeSurfaceBoundary{
                .implementation = formulations::navier_stokes::
                    FreeSurfaceImplementation::UnfittedLevelSet,
                .interface_marker = 7,
                .level_set_field_name = "phi",
                .active_domain = formulations::navier_stokes::
                    FreeSurfaceActiveDomain::LevelSetNegative,
                .active_domain_method = formulations::navier_stokes::
                    FreeSurfaceActiveDomainMethod::CutVolume,
                .small_cut_aggregation = false,
            });

    formulations::navier_stokes::IncompressibleNavierStokesVMSModule module(
        u_space, p_space, opts);
    module.registerOn(system);

    FE::systems::SetupInputs inputs;
    inputs.topology_override = makeTwoQuadStripTopology();
    ASSERT_NO_THROW(system.setup({}, inputs));

    const auto p_id = system.findFieldByName("p");
    ASSERT_NE(p_id, FE::INVALID_FIELD_ID);
    EXPECT_EQ(countConstrainedPressureDofs(system, "p"), 0u);

    const auto* registry = system.gaugeRegistryIfPresent();
    ASSERT_NE(registry, nullptr);
    const auto evidence = std::find_if(
        registry->anchoring().begin(),
        registry->anchoring().end(),
        [&](const FE::gauge::AnchoringEvidence& entry) {
            return entry.field == p_id &&
                   entry.family ==
                       FE::gauge::NullspaceModeFamily::ScalarConstant &&
                   entry.verdict == FE::gauge::AnchoringVerdict::Anchored &&
                   entry.source.find("embedded free-surface natural traction") !=
                       std::string::npos;
        });
    EXPECT_NE(evidence, registry->anchoring().end());

    const auto resolved = std::find_if(
        registry->resolvedModes().begin(),
        registry->resolvedModes().end(),
        [&](const FE::gauge::ResolvedMode& mode) {
            return mode.candidate.field == p_id &&
                   mode.candidate.family ==
                       FE::gauge::NullspaceModeFamily::ScalarConstant;
        });
    ASSERT_NE(resolved, registry->resolvedModes().end());
    EXPECT_EQ(resolved->status, FE::gauge::GaugeStatus::Anchored);
    EXPECT_EQ(resolved->policy, FE::gauge::EnforcementPolicy::None);
}

TEST(NavierStokesPressureGauge,
     CutVolumePressureAnchorRejectsInitialAllWetAndAllDryContexts)
{
    for (const FE::Real level_set_value : {FE::Real{-1.0}, FE::Real{1.0}}) {
        SCOPED_TRACE(level_set_value < 0.0 ? "all wet" : "all dry");
        auto mesh = std::make_shared<TwoQuadStripMeshAccess>();
        auto velocity_space = FE::spaces::VectorSpace(
            FE::spaces::SpaceType::H1, mesh, /*order=*/1,
            /*components=*/2);
        auto pressure_space = FE::spaces::Space(
            FE::spaces::SpaceType::H1, mesh, /*order=*/1,
            /*components=*/1);
        FE::systems::FESystem system(mesh);
        const auto phi = registerPressureAnchorProblem(
            system, velocity_space, pressure_space);

        FE::systems::SetupInputs inputs;
        inputs.topology_override = makeTwoQuadStripTopology();
        ASSERT_NO_THROW(system.setup({}, inputs));

        std::vector<FE::Real> solution(
            static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
        setPressureAnchorLevelSet(
            solution, system, phi,
            [level_set_value](const auto&) { return level_set_value; });
        FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
        auto context =
            makePressureAnchorCutContext(system, solution, lifecycle);
        ASSERT_TRUE(context->interfaceRulesForMarker(
                                pressure_anchor_interface_marker)
                            .empty());

        const auto diagnostic = captureRuntimeError(
            [&] { system.setCutIntegrationContext(context); });
        EXPECT_NE(diagnostic.find("positive generated interface measure"),
                  std::string::npos);
        EXPECT_NE(diagnostic.find("validation_stage=cut_context_update"),
                  std::string::npos);
        EXPECT_NE(diagnostic.find("global_interface_rules=0"),
                  std::string::npos);
        EXPECT_NE(diagnostic.find(
                      "gauge_policy=reject_zero_interface_no_dynamic_gauge_insertion"),
                  std::string::npos);
    }
}

TEST(NavierStokesPressureGauge,
     CutVolumePressureAnchorRejectsEvolvedInterfaceDisappearance)
{
    auto mesh = std::make_shared<TwoQuadStripMeshAccess>();
    auto velocity_space = FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/2);
    auto pressure_space = FE::spaces::Space(
        FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);
    FE::systems::FESystem system(mesh);
    const auto phi = registerPressureAnchorProblem(
        system, velocity_space, pressure_space);

    FE::systems::SetupInputs inputs;
    inputs.topology_override = makeTwoQuadStripTopology();
    ASSERT_NO_THROW(system.setup({}, inputs));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    setPressureAnchorLevelSet(
        solution, system, phi,
        [](const auto& x) { return x[1] - FE::Real{0.25}; });
    auto initial_context =
        makePressureAnchorCutContext(system, solution, lifecycle);
    ASSERT_FALSE(initial_context
                     ->interfaceRulesForMarker(pressure_anchor_interface_marker)
                     .empty());
    ASSERT_NO_THROW(system.setCutIntegrationContext(initial_context));

    setPressureAnchorLevelSet(
        solution, system, phi,
        [](const auto&) { return FE::Real{-1.0}; });
    auto disappeared_context =
        makePressureAnchorCutContext(system, solution, lifecycle);
    ASSERT_TRUE(disappeared_context
                    ->interfaceRulesForMarker(pressure_anchor_interface_marker)
                    .empty());
    const auto diagnostic = captureRuntimeError(
        [&] { system.setCutIntegrationContext(disappeared_context); });
    EXPECT_NE(diagnostic.find("validation_stage=cut_context_update"),
              std::string::npos);
    EXPECT_NE(diagnostic.find("global_interface_rules=0"),
              std::string::npos);
    EXPECT_NE(diagnostic.find(
                  "gauge_policy=reject_zero_interface_no_dynamic_gauge_insertion"),
              std::string::npos);
}

TEST(NavierStokesPressureGauge,
     CutVolumePressureAnchorRejectsPreloadedZeroMeasureContextDuringSetup)
{
    auto mesh = std::make_shared<TwoQuadStripMeshAccess>();
    auto velocity_space = FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/2);
    auto pressure_space = FE::spaces::Space(
        FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);
    FE::systems::FESystem system(mesh);
    system.setCutIntegrationContext(
        std::make_shared<FE::assembly::CutIntegrationContext>());
    (void)registerPressureAnchorProblem(
        system, velocity_space, pressure_space);

    FE::systems::SetupInputs inputs;
    inputs.topology_override = makeTwoQuadStripTopology();
    const auto diagnostic =
        captureRuntimeError([&] { system.setup({}, inputs); });
    EXPECT_NE(diagnostic.find("validation_stage=setup"), std::string::npos);
    EXPECT_NE(diagnostic.find("cut_context=present"), std::string::npos);
    EXPECT_NE(diagnostic.find("global_interface_rules=0"),
              std::string::npos);
}

TEST(NavierStokesPressureGauge,
     CutVolumePressureAnchorRevalidatesMeasureAtAssembly)
{
    auto mesh = std::make_shared<TwoQuadStripMeshAccess>();
    auto velocity_space = FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/2);
    auto pressure_space = FE::spaces::Space(
        FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);
    FE::systems::FESystem system(mesh);
    const auto phi = registerPressureAnchorProblem(
        system, velocity_space, pressure_space);

    FE::systems::SetupInputs inputs;
    inputs.topology_override = makeTwoQuadStripTopology();
    ASSERT_NO_THROW(system.setup({}, inputs));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    setPressureAnchorLevelSet(
        solution, system, phi,
        [](const auto& x) { return x[1] - FE::Real{0.25}; });
    FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    auto context =
        makePressureAnchorCutContext(system, solution, lifecycle);
    ASSERT_NO_THROW(system.setCutIntegrationContext(context));

    // Simulate stale/corrupted generated-interface measure after a previously
    // valid refresh.  The solve-time guard must not rely only on the callback.
    const auto interface_rules =
        context->interfaceRulesForMarker(pressure_anchor_interface_marker);
    ASSERT_FALSE(interface_rules.empty());
    for (const auto* const_rule : interface_rules) {
        ASSERT_NE(const_rule, nullptr);
        auto* rule = const_cast<FE::geometry::CutQuadratureRule*>(const_rule);
        rule->measure = 0.0;
        for (auto& point : rule->points) {
            point.weight = 0.0;
        }
    }

    std::vector<FE::Real> previous = solution;
    FE::systems::SystemStateView state;
    state.dt = 0.1;
    state.u = solution;
    state.u_prev = previous;
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context =
        integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;

    FE::assembly::DenseMatrixView matrix(
        system.dofHandler().getNumDofs());
    FE::systems::AssemblyRequest request;
    request.op = "equations";
    request.want_matrix = true;
    const auto diagnostic = captureRuntimeError(
        [&] { (void)system.assemble(request, state, &matrix, nullptr); });
    EXPECT_NE(diagnostic.find("validation_stage=assembly"),
              std::string::npos);
    EXPECT_NE(diagnostic.find("global_stored_measure=0"),
              std::string::npos);
}

TEST(NavierStokesFieldRegistration, ReusesCompatiblePredeclaredVelocityField)
{
    auto mesh = std::make_shared<TwoQuadStripMeshAccess>();
    auto u_space = FE::spaces::VectorSpace(FE::spaces::SpaceType::H1,
                                           FE::ElementType::Quad4,
                                           1,
                                           2);
    auto p_space = FE::spaces::SpaceFactory::create_h1(FE::ElementType::Quad4, 1);

    FE::systems::FESystem system(mesh);
    const auto predeclared_velocity = system.addField(FE::systems::FieldSpec{
        .name = "Velocity",
        .space = u_space,
        .components = 2,
        .source_kind = FE::systems::FieldSourceKind::Unknown,
    });

    formulations::navier_stokes::IncompressibleNavierStokesVMSOptions opts;
    opts.velocity_field_name = "Velocity";
    opts.pressure_field_name = "Pressure";
    opts.density = 1.0;
    opts.viscosity = 0.01;

    formulations::navier_stokes::IncompressibleNavierStokesVMSModule module(
        u_space,
        p_space,
        opts);
    module.registerOn(system);

    EXPECT_EQ(system.findFieldByName("Velocity"), predeclared_velocity);
    EXPECT_NE(system.findFieldByName("Pressure"), FE::INVALID_FIELD_ID);
    EXPECT_TRUE(system.hasOperator("equations"));
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

TEST(NavierStokesInitialConditions, MeshVertexFieldsInitializeVelocityAndPressure)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN || !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Mesh-field initialization test requires Eigen and native mesh support.";
#else
    auto mesh = makeTwoQuadStripNativeMeshWithPhi();
    auto& local_mesh = mesh->local_mesh();

    const auto velocity_handle = MeshFields::attach_field(
        local_mesh,
        EntityKind::Vertex,
        "Velocity",
        FieldScalarType::Float64,
        3);
    auto* velocity = MeshFields::field_data_as<real_t>(local_mesh, velocity_handle);
    ASSERT_NE(velocity, nullptr);

    const auto pressure_handle = MeshFields::attach_field(
        local_mesh,
        EntityKind::Vertex,
        "Pressure",
        FieldScalarType::Float64,
        1);
    auto* pressure = MeshFields::field_data_as<real_t>(local_mesh, pressure_handle);
    ASSERT_NE(pressure, nullptr);

    for (index_t vertex = 0; vertex < local_mesh.n_vertices(); ++vertex) {
        const auto x = local_mesh.get_vertex_coords(vertex);
        const auto v_base = 3u * static_cast<std::size_t>(vertex);
        velocity[v_base] = 2.0 + x[0];
        velocity[v_base + 1u] = -3.0 + x[1];
        velocity[v_base + 2u] = 99.0;
        pressure[static_cast<std::size_t>(vertex)] = 100.0 + 10.0 * x[1];
    }

    auto u_space = FE::spaces::SpaceFactory::create_vector_h1(
        FE::ElementType::Quad4,
        /*order=*/1,
        /*components=*/2);
    auto p_space = FE::spaces::SpaceFactory::create_h1(
        FE::ElementType::Quad4,
        /*order=*/1);

    formulations::navier_stokes::IncompressibleNavierStokesVMSOptions opts;
    opts.velocity_field_name = "Velocity";
    opts.pressure_field_name = "Pressure";
    opts.enable_convection = false;
    opts.enable_vms = false;
    opts.density = 1.0;
    opts.viscosity = 0.001;
    opts.hydrostatic_pressure_initialization.enabled = false;

    FE::systems::FESystem system(mesh);
    formulations::navier_stokes::IncompressibleNavierStokesVMSModule module(
        u_space,
        p_space,
        opts);
    module.registerOn(system);
    ASSERT_NO_THROW(system.setup({}));

    auto factory = FE::backends::BackendFactory::create(FE::backends::BackendKind::Eigen);
    auto state = factory->createVector(system.dofHandler().getNumDofs());
    state->zero();

    module.applyInitialConditions(system, *state);
    const auto values = state->localSpan();

    const auto u_id = system.findFieldByName("Velocity");
    const auto p_id = system.findFieldByName("Pressure");
    ASSERT_NE(u_id, FE::INVALID_FIELD_ID);
    ASSERT_NE(p_id, FE::INVALID_FIELD_ID);
    const auto* u_entity_map = system.fieldDofHandler(u_id).getEntityDofMap();
    const auto* p_entity_map = system.fieldDofHandler(p_id).getEntityDofMap();
    ASSERT_NE(u_entity_map, nullptr);
    ASSERT_NE(p_entity_map, nullptr);
    const auto u_offset = system.fieldDofOffset(u_id);
    const auto p_offset = system.fieldDofOffset(p_id);

    for (FE::GlobalIndex vertex = 0;
         vertex < static_cast<FE::GlobalIndex>(local_mesh.n_vertices());
         ++vertex) {
        const auto u_dofs = u_entity_map->getVertexDofs(vertex);
        const auto p_dofs = p_entity_map->getVertexDofs(vertex);
        ASSERT_EQ(u_dofs.size(), 2u);
        ASSERT_EQ(p_dofs.size(), 1u);

        const auto v_base = static_cast<std::size_t>(vertex) * 3u;
        EXPECT_DOUBLE_EQ(values[static_cast<std::size_t>(u_offset + u_dofs[0])],
                         velocity[v_base]);
        EXPECT_DOUBLE_EQ(values[static_cast<std::size_t>(u_offset + u_dofs[1])],
                         velocity[v_base + 1u]);
        EXPECT_DOUBLE_EQ(values[static_cast<std::size_t>(p_offset + p_dofs[0])],
                         pressure[static_cast<std::size_t>(vertex)]);
    }
#endif
}

TEST(NavierStokesInitialConditions, ActiveDomainHydrostaticPressureInitializesActiveSupport)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Active-domain hydrostatic initialization test requires native mesh support.";
#else
    auto mesh = makeTwoQuadStripNativeMeshWithPhi();
    auto& local_mesh = mesh->local_mesh();
    const auto pressure_handle = MeshFields::attach_field(
        local_mesh,
        EntityKind::Vertex,
        "HydrostaticPressure",
        FieldScalarType::Float64,
        1);
    auto* prescribed_pressure =
        MeshFields::field_data_as<real_t>(local_mesh, pressure_handle);
    ASSERT_NE(prescribed_pressure, nullptr);
    for (index_t vertex = 0; vertex < local_mesh.n_vertices(); ++vertex) {
        prescribed_pressure[static_cast<std::size_t>(vertex)] =
            250.0 + 7.0 * static_cast<double>(vertex);
    }

    auto u_space = FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1,
        FE::ElementType::Quad4,
        /*order=*/1,
        /*components=*/2);
    auto p_space = FE::spaces::Space(
        FE::spaces::SpaceType::H1,
        FE::ElementType::Quad4,
        /*order=*/1,
        /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto p_id = system.addField(FE::systems::FieldSpec{
        .name = "p",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::Unknown,
    });
    ASSERT_NE(p_id, FE::INVALID_FIELD_ID);
    ASSERT_NO_THROW(system.setup({}));

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
    opts.hydrostatic_pressure_initialization.field_name =
        "HydrostaticPressure";
    opts.free_surface.push_back(
        formulations::navier_stokes::IncompressibleNavierStokesVMSOptions::
            FreeSurfaceBoundary{
                .implementation = formulations::navier_stokes::
                    FreeSurfaceImplementation::UnfittedLevelSet,
                .interface_marker = 7,
                .level_set_field_name = "phi",
                .level_set_isovalue = 0.0,
                .active_domain = formulations::navier_stokes::
                    FreeSurfaceActiveDomain::LevelSetNegative,
            });

    formulations::navier_stokes::IncompressibleNavierStokesVMSModule module(
        u_space,
        p_space,
        opts);

    auto factory = FE::backends::BackendFactory::create(FE::backends::BackendKind::FSILS);
    auto state = factory->createVector(system.dofHandler().getNumDofs());
    state->zero();

    module.applyInitialConditions(system, *state);
    const auto values = state->localSpan();

    const auto* entity_map = system.fieldDofHandler(p_id).getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);
    const auto pressure_offset = system.fieldDofOffset(p_id);
    const auto& mesh_access = system.meshAccess();

    for (FE::GlobalIndex vertex = 0; vertex < mesh_access.numVertices(); ++vertex) {
        const auto vertex_dofs = entity_map->getVertexDofs(vertex);
        ASSERT_EQ(vertex_dofs.size(), 1u);
        const bool active_pressure_support = vertex <= 3;
        const auto expected = active_pressure_support
            ? prescribed_pressure[static_cast<std::size_t>(vertex)]
            : 100.0;
        const auto dof = pressure_offset + vertex_dofs.front();
        ASSERT_GE(dof, 0);
        ASSERT_LT(static_cast<std::size_t>(dof), values.size());
        EXPECT_NEAR(values[static_cast<std::size_t>(dof)], expected, 1.0e-12)
            << "vertex " << vertex;
    }
#endif
}

TEST(NavierStokesInitialConditions, SmoothedIndicatorHydrostaticPressureInitializesActiveSupport)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Active-domain hydrostatic initialization test requires native mesh support.";
#else
    auto mesh = makeTwoQuadStripNativeMeshWithPhi();

    auto u_space = FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1,
        FE::ElementType::Quad4,
        /*order=*/1,
        /*components=*/2);
    auto p_space = FE::spaces::Space(
        FE::spaces::SpaceType::H1,
        FE::ElementType::Quad4,
        /*order=*/1,
        /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto p_id = system.addField(FE::systems::FieldSpec{
        .name = "p",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::Unknown,
    });
    ASSERT_NE(p_id, FE::INVALID_FIELD_ID);
    ASSERT_NO_THROW(system.setup({}));

    formulations::navier_stokes::IncompressibleNavierStokesVMSOptions opts;
    opts.velocity_field_name = "u";
    opts.pressure_field_name = "p";
    opts.enable_convection = false;
    opts.enable_vms = false;
    opts.input_configuration_schema_version = 1;
    opts.explicit_legacy_configuration = true;
    opts.density = 2.0;
    opts.viscosity = 0.001;
    opts.body_force = {0.0, -9.81, 0.0};
    opts.hydrostatic_pressure_initialization.enabled = true;
    opts.hydrostatic_pressure_initialization.reference_point = {0.0, 1.0, 0.0};
    opts.hydrostatic_pressure_initialization.reference_pressure = 100.0;
    opts.free_surface.push_back(
        formulations::navier_stokes::IncompressibleNavierStokesVMSOptions::
            FreeSurfaceBoundary{
                .implementation = formulations::navier_stokes::
                    FreeSurfaceImplementation::UnfittedLevelSet,
                .interface_marker = 7,
                .level_set_field_name = "phi",
                .level_set_isovalue = 0.0,
                .active_domain = formulations::navier_stokes::
                    FreeSurfaceActiveDomain::LevelSetNegative,
                .active_domain_method = formulations::navier_stokes::
                    FreeSurfaceActiveDomainMethod::SmoothedIndicator,
            });

    formulations::navier_stokes::IncompressibleNavierStokesVMSModule module(
        u_space,
        p_space,
        opts);

    auto factory = FE::backends::BackendFactory::create(FE::backends::BackendKind::FSILS);
    auto state = factory->createVector(system.dofHandler().getNumDofs());
    state->zero();

    module.applyInitialConditions(system, *state);
    const auto values = state->localSpan();

    const auto* entity_map = system.fieldDofHandler(p_id).getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);
    const auto pressure_offset = system.fieldDofOffset(p_id);
    const auto& mesh_access = system.meshAccess();

    for (FE::GlobalIndex vertex = 0; vertex < mesh_access.numVertices(); ++vertex) {
        const auto vertex_dofs = entity_map->getVertexDofs(vertex);
        ASSERT_EQ(vertex_dofs.size(), 1u);
        const auto x = mesh_access.getNodeCoordinates(vertex);
        const bool active_pressure_support = vertex <= 3;
        const auto expected = active_pressure_support
            ? 100.0 + 2.0 * (-9.81) * (x[1] - 1.0)
            : 100.0;
        const auto dof = pressure_offset + vertex_dofs.front();
        ASSERT_GE(dof, 0);
        ASSERT_LT(static_cast<std::size_t>(dof), values.size());
        EXPECT_NEAR(values[static_cast<std::size_t>(dof)], expected, 1.0e-12)
            << "vertex " << vertex;
    }
#endif
}

TEST(NavierStokesInitialConditions,
     CutVolumeHydrostaticPressureRequiresLevelSetAwareField)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Active-domain hydrostatic initialization test requires native mesh support.";
#else
    auto mesh = makeTwoQuadStripNativeMeshWithPhi();
    auto u_space = FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1, FE::ElementType::Quad4, 1, 2);
    auto p_space = FE::spaces::Space(
        FE::spaces::SpaceType::H1, FE::ElementType::Quad4, 1, 1);

    FE::systems::FESystem system(mesh);
    ASSERT_NE(system.addField(FE::systems::FieldSpec{
                  .name = "p",
                  .space = p_space,
                  .components = 1,
                  .source_kind = FE::systems::FieldSourceKind::Unknown,
              }),
              FE::INVALID_FIELD_ID);
    ASSERT_NO_THROW(system.setup({}));

    formulations::navier_stokes::IncompressibleNavierStokesVMSOptions opts;
    opts.velocity_field_name = "u";
    opts.pressure_field_name = "p";
    opts.enable_convection = false;
    opts.enable_vms = false;
    opts.density = 2.0;
    opts.viscosity = 0.001;
    opts.body_force = {0.0, -9.81, 0.0};
    opts.hydrostatic_pressure_initialization.enabled = true;
    opts.free_surface.push_back(
        formulations::navier_stokes::IncompressibleNavierStokesVMSOptions::
            FreeSurfaceBoundary{
                .implementation = formulations::navier_stokes::
                    FreeSurfaceImplementation::UnfittedLevelSet,
                .interface_marker = 7,
                .level_set_field_name = "phi",
                .active_domain = formulations::navier_stokes::
                    FreeSurfaceActiveDomain::LevelSetNegative,
                .active_domain_method = formulations::navier_stokes::
                    FreeSurfaceActiveDomainMethod::CutVolume,
            });

    formulations::navier_stokes::IncompressibleNavierStokesVMSModule module(
        u_space, p_space, opts);
    auto factory =
        FE::backends::BackendFactory::create(FE::backends::BackendKind::FSILS);
    auto state = factory->createVector(system.dofHandler().getNumDofs());
    state->zero();

    EXPECT_THROW(module.applyInitialConditions(system, *state),
                 std::invalid_argument);
#endif
}

TEST(NavierStokesInitialConditions,
     CutVolumeHydrostaticPressureRejectsNonfiniteFieldValue)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Active-domain hydrostatic initialization test requires native mesh support.";
#else
    auto mesh = makeTwoQuadStripNativeMeshWithPhi();
    auto& local_mesh = mesh->local_mesh();
    const auto pressure_handle = MeshFields::attach_field(
        local_mesh,
        EntityKind::Vertex,
        "HydrostaticPressure",
        FieldScalarType::Float64,
        1);
    auto* prescribed_pressure =
        MeshFields::field_data_as<real_t>(local_mesh, pressure_handle);
    ASSERT_NE(prescribed_pressure, nullptr);
    for (index_t vertex = 0; vertex < local_mesh.n_vertices(); ++vertex) {
        prescribed_pressure[static_cast<std::size_t>(vertex)] = 100.0;
    }
    prescribed_pressure[0] = std::numeric_limits<real_t>::quiet_NaN();

    auto u_space = FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1, FE::ElementType::Quad4, 1, 2);
    auto p_space = FE::spaces::Space(
        FE::spaces::SpaceType::H1, FE::ElementType::Quad4, 1, 1);
    FE::systems::FESystem system(mesh);
    ASSERT_NE(system.addField(FE::systems::FieldSpec{
                  .name = "p",
                  .space = p_space,
                  .components = 1,
                  .source_kind = FE::systems::FieldSourceKind::Unknown,
              }),
              FE::INVALID_FIELD_ID);
    ASSERT_NO_THROW(system.setup({}));

    formulations::navier_stokes::IncompressibleNavierStokesVMSOptions opts;
    opts.velocity_field_name = "u";
    opts.pressure_field_name = "p";
    opts.enable_convection = false;
    opts.enable_vms = false;
    opts.density = 2.0;
    opts.viscosity = 0.001;
    opts.hydrostatic_pressure_initialization.enabled = true;
    opts.hydrostatic_pressure_initialization.field_name =
        "HydrostaticPressure";
    opts.free_surface.push_back(
        formulations::navier_stokes::IncompressibleNavierStokesVMSOptions::
            FreeSurfaceBoundary{
                .implementation = formulations::navier_stokes::
                    FreeSurfaceImplementation::UnfittedLevelSet,
                .interface_marker = 7,
                .level_set_field_name = "phi",
                .active_domain = formulations::navier_stokes::
                    FreeSurfaceActiveDomain::LevelSetNegative,
            });

    formulations::navier_stokes::IncompressibleNavierStokesVMSModule module(
        u_space, p_space, opts);
    auto factory =
        FE::backends::BackendFactory::create(FE::backends::BackendKind::FSILS);
    auto state = factory->createVector(system.dofHandler().getNumDofs());
    state->zero();

    EXPECT_THROW(module.applyInitialConditions(system, *state),
                 std::runtime_error);
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

TEST(NavierStokesPressureGauge, ActiveDomainRejectsDryPressureConstraint)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Active-domain pressure constraint validation requires native mesh support.";
#else
    auto mesh = makeTwoQuadStripNativeMeshWithPhi();
    auto u_space = FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1,
        FE::ElementType::Quad4,
        /*order=*/1,
        /*components=*/2);
    auto p_space = FE::spaces::Space(
        FE::spaces::SpaceType::H1,
        FE::ElementType::Quad4,
        /*order=*/1,
        /*components=*/1);

    formulations::navier_stokes::IncompressibleNavierStokesVMSOptions opts;
    opts.velocity_field_name = "u";
    opts.pressure_field_name = "p";
    opts.enable_convection = false;
    opts.enable_vms = false;
    opts.density = 1.0;
    opts.viscosity = 0.001;
    opts.free_surface.push_back(
        formulations::navier_stokes::IncompressibleNavierStokesVMSOptions::
            FreeSurfaceBoundary{
                .implementation = formulations::navier_stokes::
                    FreeSurfaceImplementation::UnfittedLevelSet,
                .interface_marker = 7,
                .level_set_field_name = "phi",
                .level_set_isovalue = 0.0,
                .active_domain = formulations::navier_stokes::
                    FreeSurfaceActiveDomain::LevelSetNegative,
            });
    opts.node_pressure_constraints.id_type =
        formulations::navier_stokes::IncompressibleNavierStokesVMSOptions::
            NodePressureConstraintIdType::LocalVertexId;
    opts.node_pressure_constraints.values = {
        formulations::navier_stokes::IncompressibleNavierStokesVMSOptions::
            NodePressureConstraint{.node_id = 2, .pressure = 0.0}};

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    formulations::navier_stokes::IncompressibleNavierStokesVMSModule module(
        u_space,
        p_space,
        opts);
    try {
        module.registerOn(system);
        FAIL() << "expected dry-side pressure constraint rejection";
    } catch (const std::invalid_argument& ex) {
        EXPECT_NE(std::string(ex.what()).find("dry side"), std::string::npos)
            << ex.what();
    }
#endif
}

TEST(NavierStokesPressureGauge, ActiveDomainRejectsNearInterfacePressureConstraint)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Active-domain pressure constraint validation requires native mesh support.";
#else
    auto mesh = makeTwoQuadStripNativeMeshWithPhi();
    auto& local_mesh = mesh->local_mesh();
    const auto phi_handle =
        MeshFields::get_field_handle(local_mesh, EntityKind::Vertex, "phi");
    auto* phi = MeshFields::field_data_as<real_t>(local_mesh, phi_handle);
    phi[0] = -5.0e-9;

    auto u_space = FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1,
        FE::ElementType::Quad4,
        /*order=*/1,
        /*components=*/2);
    auto p_space = FE::spaces::Space(
        FE::spaces::SpaceType::H1,
        FE::ElementType::Quad4,
        /*order=*/1,
        /*components=*/1);

    formulations::navier_stokes::IncompressibleNavierStokesVMSOptions opts;
    opts.velocity_field_name = "u";
    opts.pressure_field_name = "p";
    opts.enable_convection = false;
    opts.enable_vms = false;
    opts.density = 1.0;
    opts.viscosity = 0.001;
    opts.free_surface.push_back(
        formulations::navier_stokes::IncompressibleNavierStokesVMSOptions::
            FreeSurfaceBoundary{
                .implementation = formulations::navier_stokes::
                    FreeSurfaceImplementation::UnfittedLevelSet,
                .interface_marker = 7,
                .level_set_field_name = "phi",
                .level_set_isovalue = 0.0,
                .active_domain = formulations::navier_stokes::
                    FreeSurfaceActiveDomain::LevelSetNegative,
            });
    opts.node_pressure_constraints.id_type =
        formulations::navier_stokes::IncompressibleNavierStokesVMSOptions::
            NodePressureConstraintIdType::LocalVertexId;
    opts.node_pressure_constraints.values = {
        formulations::navier_stokes::IncompressibleNavierStokesVMSOptions::
            NodePressureConstraint{.node_id = 0, .pressure = 0.0}};

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    formulations::navier_stokes::IncompressibleNavierStokesVMSModule module(
        u_space,
        p_space,
        opts);
    try {
        module.registerOn(system);
        FAIL() << "expected near-interface pressure constraint rejection";
    } catch (const std::invalid_argument& ex) {
        EXPECT_NE(std::string(ex.what()).find("too close"), std::string::npos)
            << ex.what();
    }
#endif
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
