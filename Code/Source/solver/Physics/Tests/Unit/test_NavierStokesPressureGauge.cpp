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
#include "FE/Systems/FormsInstaller.h"
#include "FE/Forms/Forms.h"
#include "FE/TimeStepping/TimeLoop.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Core/MeshBase.h"
#  include "Mesh/Fields/MeshFields.h"
#  include "Mesh/Mesh.h"
#  include "Mesh/Topology/CellShape.h"
#endif

#include <algorithm>
#include <array>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "FE/Constraints/SmallCutAggregationConstraint.h"

namespace {
// Every small-cut aggregation prolongation report requested in this test
// binary is also built at its closure and compared bit for bit with the
// report built on request.
[[maybe_unused]] const bool small_cut_aggregation_lazy_report_verification =
    [] {
        svmp::FE::constraints::
            setSmallCutAggregationLazyReportVerificationForTesting(true);
        return true;
    }();
} // namespace


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

[[nodiscard]] std::shared_ptr<Mesh> makeSingleTriangleNativeMesh()
{
    auto base = std::make_shared<MeshBase>();
    const std::vector<real_t> x_ref = {
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
    };
    const std::vector<offset_t> cell2vertex_offsets = {0, 3};
    const std::vector<index_t> cell2vertex = {0, 1, 2};
    CellShape shape{};
    shape.family = CellFamily::Triangle;
    shape.num_corners = 3;
    shape.order = 1;
    base->build_from_arrays(
        /*spatial_dim=*/2,
        x_ref,
        cell2vertex_offsets,
        cell2vertex,
        {shape});
    base->finalize();
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
            .external_pressure = 0.0,
            .surface_tension = 0.0,
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

#if SVMP_FE_WITH_MESH && FE_HAS_EIGEN
TEST(NavierStokesInitialConditions, FreshFreeSurfacePressureUsesUniformRestingExteriorLoad)
{
    namespace ns = formulations::navier_stokes;
    const auto velocity = FE::spaces::SpaceFactory::create_vector_h1(FE::ElementType::Quad4, 1, 2);
    const auto scalar = FE::spaces::SpaceFactory::create_h1(FE::ElementType::Quad4, 1);
    const auto factory = FE::backends::BackendFactory::create(FE::backends::BackendKind::Eigen);
    enum class Load { UniformNegative, UniformPositive, Gravity, Capillary, Variable, Conflicting };
    for (const auto load : {Load::UniformNegative, Load::UniformPositive, Load::Gravity,
                            Load::Capillary, Load::Variable, Load::Conflicting}) {
        SCOPED_TRACE(static_cast<int>(load));
        auto mesh = makeTwoQuadStripNativeMeshWithPhi();
        FE::systems::FESystem system(mesh);
        (void)system.addField({.name = "phi", .space = scalar, .source_kind = FE::systems::FieldSourceKind::PrescribedData});
        auto fluid = pressureAnchorOptions();
        fluid.jit_policy.enable = false;
        auto& surface = fluid.free_surface.front();
        surface.external_pressure = 37.0;
        if (load == Load::UniformPositive) surface.active_domain = ns::FreeSurfaceActiveDomain::LevelSetPositive;
        if (load == Load::Gravity) fluid.body_force[1] = -9.81;
        if (load == Load::Capillary) surface.surface_tension = 1.0;
        if (load == Load::Variable) surface.external_pressure = FE::forms::FormExpr::time();
        if (load == Load::Conflicting) fluid.node_pressure_constraints.values.push_back({.node_id = 0, .pressure = 13.0});
        ns::IncompressibleNavierStokesVMSModule module(velocity, scalar, fluid);
        module.registerOn(system);
        system.setup();
        auto state = factory->createVector(system.dofHandler().getNumDofs());
        state->zero();
        module.applyInitialConditions(system, *state);
        const bool uniform = load == Load::UniformNegative || load == Load::UniformPositive;
        for (const auto row : system.fieldMap().getComponentDofs(fluid.pressure_field_name, 0)) {
            EXPECT_DOUBLE_EQ(state->localSpan()[row], uniform ? 37.0 : 0.0);
        }
    }
}

TEST(NavierStokesInitialConditions, FreshFluidPressureUsesRestingOutletPressureOrExplicitMeshValues)
{
    namespace ns = formulations::navier_stokes;
    const auto velocity = FE::spaces::SpaceFactory::create_vector_h1(FE::ElementType::Triangle3, 1, 2);
    const auto scalar = FE::spaces::SpaceFactory::create_h1(FE::ElementType::Triangle3, 1);
    const auto factory = FE::backends::BackendFactory::create(FE::backends::BackendKind::Eigen);
    enum class Outlet { Resistance, RCR, RCRCR };
    for (auto outlet : {Outlet::Resistance, Outlet::RCR, Outlet::RCRCR}) {
        for (bool explicit_pressure : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "outlet=" << static_cast<int>(outlet)
                                              << ", explicit=" << explicit_pressure);
            auto mesh = makeSingleTriangleNativeMesh();
            auto& local = mesh->local_mesh();
            local.register_label("outlet", 8);
            local.set_boundary_label(local.boundary_faces().front(), 8);
            ns::IncompressibleNavierStokesVMSOptions fluid;
            fluid.enable_vms = false;
            fluid.jit_policy.enable = false;
            if (explicit_pressure) {
                const auto field = MeshFields::attach_field(local, EntityKind::Vertex, fluid.pressure_field_name,
                                                            FieldScalarType::Float64, 1);
                auto* values = MeshFields::field_data_as<real_t>(local, field);
                for (index_t v = 0; v < local.n_vertices(); ++v) {
                    values[v] = 2.0 * v;
                }
            }
            if (outlet == Outlet::RCRCR) {
                fluid.coupled_outflow_rcrcr.push_back({.boundary_marker = 8, .C1 = 1.0, .C2 = 1.0, .Pd = 23.0,
                    .P10 = 11.0, .P20 = 17.0, .functional_name = "outlet_flow"});
            } else {
                fluid.coupled_outflow_rcr.push_back({.boundary_marker = 8, .C = outlet == Outlet::RCR ? 1.0 : 0.0,
                    .Pd = 23.0, .X0 = 11.0, .functional_name = "outlet_flow"});
            }
            ns::IncompressibleNavierStokesVMSModule module(velocity, scalar, fluid);
            FE::systems::FESystem system(mesh);
            module.registerOn(system);
            system.setup();
            auto state = factory->createVector(system.dofHandler().getNumDofs());
            state->zero();
            module.applyInitialConditions(system, *state);
            const auto p = system.findFieldByName(fluid.pressure_field_name);
            const auto* dofs = system.fieldDofHandler(p).getEntityDofMap();
            const double resting_datum = outlet == Outlet::Resistance ? 23.0 : 11.0;
            for (index_t v = 0; v < local.n_vertices(); ++v) {
                const auto row = system.fieldDofOffset(p) + dofs->getVertexDofs(v).front();
                EXPECT_DOUBLE_EQ(state->localSpan()[row], explicit_pressure ? 2.0 * v : resting_datum);
            }
            if (!explicit_pressure) {
                // A caller may have already populated this endpoint pressure.
                const auto p0 = system.fieldDofOffset(p);
                state->zero();
                state->localSpan()[p0] = -43.0;
                module.applyInitialConditions(system, *state);
                EXPECT_DOUBLE_EQ(state->localSpan()[p0], -43.0);
                EXPECT_DOUBLE_EQ(state->localSpan()[p0 + 1], 0.0);

                state->zero();
                state->localSpan()[system.fieldDofOffset(system.findFieldByName(fluid.velocity_field_name))] = 1.0;
                module.applyInitialConditions(system, *state);
                EXPECT_DOUBLE_EQ(state->localSpan()[p0], 0.0);
            }
        }
    }
}

TEST(NavierStokesPressureGauge, GeneralizedAlphaPublishesNaturalTractionPressureAtTheEndpoint)
{
    namespace ns = formulations::navier_stokes;
    namespace fs = FE::systems;
    namespace f = FE::forms;
    namespace ts = FE::timestepping;
    for (double curvature : {0.0, 1.0}) {
        fs::FESystem system(makeSingleTriangleNativeMesh());
        const auto velocity = FE::spaces::SpaceFactory::create_vector_h1(FE::ElementType::Triangle3, 1, 2);
        const auto scalar = FE::spaces::SpaceFactory::create_l2(FE::ElementType::Triangle3, 0);
        ns::IncompressibleNavierStokesVMSOptions fluid;
        fluid.pressure_field_name = "liquid_head";
        fluid.enable_convection = false;
        fluid.enable_vms = false;
        fluid.jit_policy.enable = false;
        ns::IncompressibleNavierStokesVMSModule(velocity, scalar, fluid).registerOn(system);
        const auto u = system.findFieldByName(fluid.velocity_field_name);
        const auto p = system.findFieldByName(fluid.pressure_field_name);
        const auto multiplier = system.addField({.name = "p", .space = scalar, .solver_role = fs::SolverRole::Constraint});

        // The traction -P(t)n on every face balances a spatially constant pressure at zero velocity.
        const auto t = f::FormExpr::time();
        const auto load = t + curvature * t * t;
        const auto v = f::FormExpr::testFunction(u, *velocity, "v");
        const auto lambda = f::FormExpr::stateField(multiplier, *scalar, "lambda");
        const auto r = f::FormExpr::testFunction(multiplier, *scalar, "r");
        fs::FormInstallOptions form_options;
        form_options.compiler_options.jit.enable = false;
        (void)fs::installFormulation(system, fluid.operator_tag, {u, multiplier},
            (load * f::inner(f::FormExpr::normal(), v)).ds() + ((lambda - t) * r).dx(), form_options);
        system.gaugeRegistry().addAnchoring({.field = p, .family = FE::gauge::NullspaceModeFamily::ScalarConstant,
            .verdict = FE::gauge::AnchoringVerdict::Anchored, .source = "normal traction on every face"});
        system.setup();
        ASSERT_EQ(countConstrainedPressureDofs(system, fluid.pressure_field_name), 0u);
        const auto p_dof = system.fieldDofOffset(p);
        const auto lambda_dof = system.fieldDofOffset(multiplier);
        const auto exact = [curvature](double time) { return time + curvature * time * time; };

        const auto factory = FE::backends::BackendFactory::create(FE::backends::BackendKind::Eigen);
        FE::backends::SolverOptions direct;
        direct.method = FE::backends::SolverMethod::Direct;
        auto linear = factory->createLinearSolver(direct);
        fs::TransientSystem transient(system, std::make_shared<fs::BackwardDifferenceIntegrator>());
        for (double dt : {0.05, 0.025}) {
            SCOPED_TRACE(::testing::Message() << "curvature=" << curvature << ", dt=" << dt);
            ts::TimeLoopOptions options;
            options.t0 = 0.125;
            options.dt = dt;
            options.scheme = ts::SchemeKind::GeneralizedAlpha;
            options.generalized_alpha_rho_inf = 0.5;
            options.step_controller = std::make_shared<ts::SimpleStepController>(
                ts::SimpleStepControllerOptions{.max_dt = dt});
            options.newton.residual_op = options.newton.jacobian_op = fluid.operator_tag;
            options.newton.abs_tolerance = 1e-12;
            options.newton.rel_tolerance = 0.0;
            auto history = ts::TimeHistory::allocate(*factory, system.dofHandler().getNumDofs());
            history.uPrev().zero();
            history.uPrev().localSpan()[p_dof] = exact(options.t0);
            history.uPrev().localSpan()[lambda_dof] = options.t0;
            history.resetCurrentToPrevious();
            history.primeDtHistory(dt);
            history.ensureSecondOrderState(*factory);
            history.uDot().zero();

            ts::TimeLoopCallbacks callbacks;
            double last_stage_time = 0.0;
            int rejected = 0;
            callbacks.on_before_step_accept = [&](ts::TimeHistory&, const ts::NewtonReport&) {
                return rejected > 0;
            };
            callbacks.on_step_rejected = [&](const ts::TimeHistory& restored, ts::StepRejectReason reason,
                                             const ts::NewtonReport&) {
                ++rejected;
                EXPECT_EQ(reason, ts::StepRejectReason::ErrorTooLarge);
                EXPECT_DOUBLE_EQ(restored.uSpan()[p_dof], exact(options.t0));
            };
            callbacks.on_candidate_stage = [&](const ts::CandidateStageObservation& stage) {
                last_stage_time = stage.state_time;
                EXPECT_NEAR(stage.state_vector->localSpan()[p_dof], exact(last_stage_time), 1e-11);
            };
            callbacks.on_step_accepted = [&](const ts::TimeHistory& accepted) {
                const auto state = accepted.uSpan();
                // For quadratic P, linear stage interpolation has a bounded O(dt^2) endpoint error.
                EXPECT_NEAR(state[p_dof], exact(accepted.time()), curvature * dt * dt + 1e-11);
                EXPECT_NEAR(state[lambda_dof], last_stage_time, 1e-11);
                EXPECT_DOUBLE_EQ(accepted.uDotSpan()[p_dof], 0.0);
                for (FE::GlobalIndex i = 0; i < p_dof; ++i) {
                    EXPECT_NEAR(state[i], 0.0, 1e-11);
                }
            };
            // Continue from the accepted history with a shorter last step in each run.
            for (double end_time : {0.2, 0.235}) {
                options.t_end = end_time;
                const auto report = ts::TimeLoop(options).run(transient, *factory, *linear, history, callbacks);
                ASSERT_TRUE(report.success) << report.message;
                EXPECT_DOUBLE_EQ(report.final_time, end_time);
                options.t0 = history.time();
            }
            EXPECT_EQ(rejected, 1);
        }
    }
}
#endif

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

TEST(NavierStokesInitialConditions,
     TwoFluidMeshVertexFieldsInitializeBothPhasePairs)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Two-fluid mesh-field initialization requires native mesh support.";
#else
    auto mesh = makeSingleTriangleNativeMesh();
    auto& local_mesh = mesh->local_mesh();

    struct MeshFieldData {
        std::string name;
        std::size_t components;
        std::vector<FE::Real> values;
    };
    const std::array<MeshFieldData, 4> mesh_fields{{
        {"u_negative", 2u, {1.0, 2.0, 1.5, 2.0, 1.0, 2.5}},
        {"p_negative", 1u, {3.0, 3.5, 4.0}},
        {"u_positive", 2u, {-1.0, -2.0, 0.0, -2.0, -1.0, 1.0}},
        {"p_positive", 1u, {-3.0, -2.5, -2.0}},
    }};
    for (const auto& source : mesh_fields) {
        const auto handle = MeshFields::attach_field(
            local_mesh,
            EntityKind::Vertex,
            source.name,
            FieldScalarType::Float64,
            source.components);
        auto* values = MeshFields::field_data_as<real_t>(local_mesh, handle);
        ASSERT_NE(values, nullptr);
        std::copy(source.values.begin(), source.values.end(), values);
    }

    auto scalar_space = FE::spaces::SpaceFactory::create_h1(
        FE::ElementType::Triangle3, 1);
    auto velocity_space = FE::spaces::SpaceFactory::create_vector_h1(
        FE::ElementType::Triangle3, 1, 2);
    FE::systems::FESystem system(mesh);
    (void)system.addField(FE::systems::FieldSpec{
        .name = "level_set",
        .space = scalar_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::Unknown,
    });
    for (const auto& source : mesh_fields) {
        std::shared_ptr<const FE::spaces::FunctionSpace> field_space =
            scalar_space;
        if (source.components != 1u) {
            field_space = velocity_space;
        }
        (void)system.addField(FE::systems::FieldSpec{
            .name = source.name,
            .space = std::move(field_space),
            .components = static_cast<int>(source.components),
            .source_kind = FE::systems::FieldSourceKind::Unknown,
        });
    }
    ASSERT_NO_THROW(system.setup({}));

    formulations::navier_stokes::IncompressibleTwoFluidOptions options;
    options.interface_marker = 71;
    formulations::navier_stokes::IncompressibleNavierStokesVMSModule module(
        velocity_space,
        scalar_space,
        velocity_space,
        scalar_space,
        options);
    auto factory = FE::backends::BackendFactory::create(
        FE::backends::BackendKind::FSILS);
    auto state = factory->createVector(system.dofHandler().getNumDofs());
    state->zero();

    module.applyInitialConditions(system, *state);
    const auto values = state->localSpan();
    for (const auto& source : mesh_fields) {
        const auto field = system.findFieldByName(source.name);
        ASSERT_NE(field, FE::INVALID_FIELD_ID);
        const auto* entity_map =
            system.fieldDofHandler(field).getEntityDofMap();
        ASSERT_NE(entity_map, nullptr);
        const auto field_offset = system.fieldDofOffset(field);
        for (FE::GlobalIndex vertex = 0; vertex < 3; ++vertex) {
            const auto dofs = entity_map->getVertexDofs(vertex);
            ASSERT_EQ(dofs.size(), source.components);
            for (std::size_t component = 0u;
                 component < source.components;
                 ++component) {
                const auto expected_index =
                    static_cast<std::size_t>(vertex) * source.components +
                    component;
                EXPECT_DOUBLE_EQ(
                    values[static_cast<std::size_t>(
                        field_offset + dofs[component])],
                    source.values[expected_index]);
            }
        }
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
