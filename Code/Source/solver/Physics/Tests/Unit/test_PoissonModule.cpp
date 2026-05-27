/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Core/EquationModuleRegistry.h"
#include "Physics/Formulations/Poisson/PoissonModule.h"
#include "Physics/Tests/Unit/PhysicsTestHelpers.h"

#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Assembly/GlobalSystemView.h"
#include "FE/Backends/Interfaces/BackendFactory.h"
#include "FE/Backends/Interfaces/BackendKind.h"
#include "FE/Backends/Interfaces/LinearSolver.h"
#include "FE/Backends/Utils/BackendOptions.h"
#include "FE/Constraints/VertexDirichletConstraint.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Spaces/H1Space.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/TimeIntegrator.h"
#include "FE/Systems/TransientSystem.h"
#include "FE/Tests/Unit/Forms/FormsTestHelpers.h"
#include "FE/TimeStepping/NewtonSolver.h"
#include "FE/TimeStepping/TimeHistory.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if FE_HAS_MPI || defined(MESH_HAS_MPI)
#  include <mpi.h>
#endif

namespace svmp {
namespace Physics {
namespace test {

namespace {

std::shared_ptr<Mesh> buildPoissonNodeConstraintQuadMesh()
{
    auto base = std::make_shared<MeshBase>();

    const std::vector<real_t> x_ref = {
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0,
    };
    const std::vector<offset_t> cell2vertex_offsets = {0, 4};
    const std::vector<index_t> cell2vertex = {0, 1, 2, 3};

    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(/*spatial_dim=*/2, x_ref, cell2vertex_offsets, cell2vertex, {shape});
    base->set_vertex_gids({10, 20, 30, 40});
    base->finalize();

    return create_mesh(std::move(base));
}

std::filesystem::path uniqueNodePressureCsvPath(const void* token)
{
    return std::filesystem::temp_directory_path() /
           ("svmp_node_pressure_" + std::to_string(reinterpret_cast<std::uintptr_t>(token)) + ".csv");
}

void writeTextFile(const std::filesystem::path& path, const std::string& text)
{
    std::ofstream out(path);
    ASSERT_TRUE(out) << "failed to open " << path;
    out << text;
}

std::shared_ptr<Mesh> buildBoundaryMarkedQuadMesh(int marker)
{
    auto mesh = buildPoissonNodeConstraintQuadMesh();
    auto& base = mesh->base();
    base.register_label("boundary", marker);

    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(base.n_faces()); ++f) {
        const auto verts = base.face_vertices(f);
        if (verts.size() != 2u) {
            continue;
        }
        const bool has0 = (verts[0] == 0 || verts[1] == 0);
        const bool has3 = (verts[0] == 3 || verts[1] == 3);
        if (has0 && has3) {
            base.set_boundary_label(f, marker);
            base.add_to_set(EntityKind::Face, "boundary", f);
        }
    }

    return mesh;
}

std::shared_ptr<Mesh> buildStructuredQuadPressureMesh(int nx, int ny, int boundary_marker)
{
    auto base = std::make_shared<MeshBase>();

    std::vector<real_t> x_ref;
    x_ref.reserve(static_cast<std::size_t>((nx + 1) * (ny + 1) * 2));
    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            x_ref.push_back(static_cast<real_t>(static_cast<double>(i) / static_cast<double>(nx)));
            x_ref.push_back(static_cast<real_t>(static_cast<double>(j) / static_cast<double>(ny)));
        }
    }

    std::vector<offset_t> cell2vertex_offsets;
    std::vector<index_t> cell2vertex;
    cell2vertex_offsets.reserve(static_cast<std::size_t>(nx * ny + 1));
    cell2vertex.reserve(static_cast<std::size_t>(nx * ny * 4));
    cell2vertex_offsets.push_back(0);

    const auto vid = [nx](int i, int j) { return static_cast<index_t>(j * (nx + 1) + i); };
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            cell2vertex.push_back(vid(i, j));
            cell2vertex.push_back(vid(i + 1, j));
            cell2vertex.push_back(vid(i + 1, j + 1));
            cell2vertex.push_back(vid(i, j + 1));
            cell2vertex_offsets.push_back(static_cast<offset_t>(cell2vertex.size()));
        }
    }

    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    std::vector<CellShape> cell_shapes(static_cast<std::size_t>(nx * ny), shape);
    base->build_from_arrays(/*spatial_dim=*/2, x_ref, cell2vertex_offsets, cell2vertex, cell_shapes);

    std::vector<gid_t> vertex_gids(static_cast<std::size_t>((nx + 1) * (ny + 1)));
    for (std::size_t i = 0; i < vertex_gids.size(); ++i) {
        vertex_gids[i] = static_cast<gid_t>(1000 + i);
    }
    base->set_vertex_gids(std::move(vertex_gids));
    base->finalize();
    base->register_label("boundary", boundary_marker);

    constexpr double tol = 1e-12;
    for (index_t f = 0; f < static_cast<index_t>(base->n_faces()); ++f) {
        const auto verts = base->face_vertices(f);
        if (verts.empty()) {
            continue;
        }
        bool on_boundary = true;
        int side = -1;
        for (const auto v : verts) {
            const auto xyz = base->get_vertex_coords(v);
            int this_side = -1;
            if (std::abs(static_cast<double>(xyz[0])) <= tol) {
                this_side = 0;
            } else if (std::abs(static_cast<double>(xyz[0]) - 1.0) <= tol) {
                this_side = 1;
            } else if (std::abs(static_cast<double>(xyz[1])) <= tol) {
                this_side = 2;
            } else if (std::abs(static_cast<double>(xyz[1]) - 1.0) <= tol) {
                this_side = 3;
            } else {
                on_boundary = false;
                break;
            }
            if (side == -1) {
                side = this_side;
            } else if (side != this_side) {
                on_boundary = false;
                break;
            }
        }
        if (on_boundary) {
            base->set_boundary_label(f, boundary_marker);
            base->add_to_set(EntityKind::Face, "boundary", f);
        }
    }

    return create_mesh(std::move(base));
}

BoundaryConditionInput dirichletPressureBC(int marker, FE::Real value)
{
    BoundaryConditionInput bc{};
    bc.name = "boundary";
    bc.boundary_marker = marker;
    bc.params["Type"] = ParameterValue{true, "Dirichlet"};
    bc.params["Value"] = ParameterValue{true, std::to_string(static_cast<double>(value))};
    bc.params["Time_dependence"] = ParameterValue{true, "Steady"};
    return bc;
}

std::unique_ptr<FE::backends::BackendFactory> tryCreateBackend(FE::backends::BackendKind kind,
                                                               std::string& error)
{
    try {
        if (kind == FE::backends::BackendKind::FSILS) {
            FE::backends::BackendFactory::CreateOptions opts;
            opts.dof_per_node = 1;
            return FE::backends::BackendFactory::create(kind, opts);
        }
        return FE::backends::BackendFactory::create(kind);
    } catch (const std::exception& e) {
        error = e.what();
        return nullptr;
    }
}

FE::timestepping::TimeHistory solveSteadyHistory(FE::systems::FESystem& system,
                                                 FE::backends::BackendFactory& factory,
                                                 FE::backends::SolverMethod method,
                                                 FE::backends::PreconditionerType preconditioner,
                                                 FE::Real rel_tol = 1e-12,
                                                 FE::Real abs_tol = 1e-14)
{
    const auto n_dofs = system.dofHandler().getNumDofs();
    FE::backends::SolverOptions lopt;
    lopt.method = method;
    lopt.preconditioner = preconditioner;
    lopt.rel_tol = rel_tol;
    lopt.abs_tol = abs_tol;
    lopt.max_iter = 500;

    auto linear = factory.createLinearSolver(lopt);
    if (!linear) {
        throw std::runtime_error("failed to create linear solver");
    }

    auto integrator = std::make_shared<FE::systems::BackwardDifferenceIntegrator>();
    FE::systems::TransientSystem transient(system, integrator);

    FE::timestepping::NewtonOptions newton_options;
    newton_options.residual_op = "equations";
    newton_options.jacobian_op = "equations";
    FE::timestepping::NewtonSolver newton(newton_options);
    FE::timestepping::NewtonWorkspace ws;
    newton.allocateWorkspace(system, factory, ws);

    auto history = FE::timestepping::TimeHistory::allocate(factory, n_dofs, /*history_depth=*/2);
    history.setDt(1.0);
    history.setPrevDt(1.0);
    history.primeDtHistory(1.0);
    history.u().zero();
    system.constraints().distribute(history.u());
    history.u().updateGhosts();

    const auto report = newton.solveStep(transient, *linear, /*solve_time=*/0.0, history, ws);
    EXPECT_TRUE(report.converged) << "Newton did not converge (iters=" << report.iterations
                                  << ", |r|=" << report.residual_norm << ")";

    history.u().updateGhosts();
    return history;
}

std::vector<FE::Real> solveSteadySystem(FE::systems::FESystem& system,
                                        FE::backends::BackendFactory& factory,
                                        FE::backends::SolverMethod method,
                                        FE::backends::PreconditionerType preconditioner,
                                        FE::Real rel_tol = 1e-12,
                                        FE::Real abs_tol = 1e-14)
{
    auto history = solveSteadyHistory(system, factory, method, preconditioner, rel_tol, abs_tol);
    history.u().updateGhosts();
    const auto u_span = history.uSpan();
    return std::vector<FE::Real>(u_span.begin(), u_span.end());
}

std::shared_ptr<FE::assembly::CutIntegrationContext>
makePoissonSingleTetraGeneratedCutContext(FE::FieldId phi_field,
                                          int marker,
                                          std::uint64_t source_revision)
{
    auto cut_context = std::make_shared<FE::assembly::CutIntegrationContext>();

    FE::geometry::CutQuadratureRule rule;
    rule.kind = FE::geometry::CutQuadratureKind::Volume;
    rule.side = FE::geometry::CutIntegrationSide::Negative;
    rule.parent_measure = FE::Real{1.0} / FE::Real{6.0};
    rule.measure = FE::Real{1.0} / FE::Real{12.0};
    rule.volume_fraction = FE::Real{0.5};
    rule.frame = FE::geometry::CutGeometryFrame::Reference;
    rule.provenance.parent_entity = 0;
    rule.provenance.marker = marker;
    rule.provenance.source_value_revision = source_revision;
    rule.points.push_back(FE::geometry::CutQuadraturePoint{
        .point = {{FE::Real{0.25}, FE::Real{0.25}, FE::Real{0.25}}},
        .weight = rule.measure,
    });

    FE::assembly::CutCellAssemblyMetadata metadata;
    metadata.parent_entity = 0;
    metadata.side = FE::geometry::CutIntegrationSide::Negative;
    metadata.volume_fraction = rule.volume_fraction;
    metadata.source_value_revision = source_revision;
    cut_context->addGeneratedVolumeRule(marker, metadata, rule);

    FE::interfaces::CutInterfaceDomainRequest domain_request;
    domain_request.source =
        FE::interfaces::LevelSetInterfaceSource::fromField(
            phi_field,
            0u,
            source_revision);
    domain_request.interface_marker = marker;
    domain_request.quadrature_order = 0;
    domain_request.interface_quadrature_order = 0;
    domain_request.volume_quadrature_order = 0;
    FE::interfaces::LevelSetInterfaceDomain domain(domain_request);

    FE::interfaces::CutInterfaceFragment fragment;
    fragment.interface_marker = marker;
    fragment.parent_cell = 0;
    fragment.local_fragment_index = 0;
    fragment.stable_id = 1;
    fragment.kind = FE::interfaces::CutInterfaceFragmentKind::Polygon;
    fragment.measure = std::sqrt(FE::Real{3.0}) / FE::Real{8.0};
    fragment.normal = {{
        FE::Real{1.0} / std::sqrt(FE::Real{3.0}),
        FE::Real{1.0} / std::sqrt(FE::Real{3.0}),
        FE::Real{1.0} / std::sqrt(FE::Real{3.0}),
    }};
    FE::interfaces::CutInterfaceQuadraturePoint interface_qp;
    interface_qp.point = {{
        FE::Real{1.0} / FE::Real{6.0},
        FE::Real{1.0} / FE::Real{6.0},
        FE::Real{1.0} / FE::Real{6.0},
    }};
    interface_qp.parent_coordinate = interface_qp.point;
    interface_qp.normal = fragment.normal;
    interface_qp.weight = fragment.measure;
    fragment.quadrature_points.push_back(interface_qp);
    domain.addFragment(std::move(fragment));
    cut_context->addGeneratedInterfaceDomain(domain);

    return cut_context;
}

} // namespace

TEST(PoissonModule, AssembledJacobianMatchesFiniteDifference)
{
    auto mesh = std::make_shared<SingleTetraMeshAccess>();
    FE::systems::FESystem system(mesh);

    auto space = std::make_shared<FE::spaces::H1Space>(FE::ElementType::Tetra4, /*order=*/1);
    formulations::poisson::PoissonOptions opts;
    opts.field_name = "u";
    opts.diffusion = 1.0;
    opts.source = 0.25;

    formulations::poisson::PoissonModule module(space, opts);
    module.registerOn(system);

    std::vector<FE::constraints::VertexDirichletValue> anchor = {
        {.vertex_id = 0, .value = 0.10},
    };
    system.addSystemConstraint(std::make_unique<FE::constraints::VertexDirichletConstraint>(
        /*field=*/0, std::move(anchor), FE::constraints::VertexIdMode::LocalVertexId));

    system.setup({}, makeSingleTetraSetupInputs());
    ASSERT_TRUE(system.constraints().isConstrained(0));
    EXPECT_NEAR(system.constraints().getInhomogeneity(0), 0.10, 1e-12);

    const auto n = system.dofHandler().getNumDofs();
    ASSERT_EQ(n, 4);

    std::vector<FE::Real> u(static_cast<std::size_t>(n));
    u[0] = 0.10;
    u[1] = -0.05;
    u[2] = 0.20;
    u[3] = 0.03;

    FE::systems::SystemStateView state;
    state.u = std::span<const FE::Real>(u);

    FE::assembly::DenseMatrixView J(n);
    FE::systems::AssemblyRequest jac_req;
    jac_req.op = "equations";
    jac_req.want_matrix = true;
    const auto jr = system.assemble(jac_req, state, &J, nullptr);
    ASSERT_TRUE(jr.success) << jr.error_message;
    EXPECT_TRUE(J.isSymmetric(1e-12));

    expectOperatorJacobianMatchesCentralFD(system,
                                           state,
                                           "equations",
                                           /*eps=*/1e-6,
                                           /*rtol=*/1e-6,
                                           /*atol=*/1e-10);
}

TEST(PoissonModule, LevelSetCutDomainRestrictionInstallsShapeTangentForScalarPhysics)
{
    auto mesh = std::make_shared<SingleTetraMeshAccess>();
    FE::systems::FESystem system(mesh);

    auto space = std::make_shared<FE::spaces::H1Space>(
        FE::ElementType::Tetra4, /*order=*/1);
    const auto phi_field = system.addField(
        FE::systems::FieldSpec{.name = "phi", .space = space, .components = 1});

    constexpr int marker = 714;
    system.setFormInstallCellDomainRestrictions({
        FE::systems::FESystem::FormCellDomainRestriction{
            .interface_marker = marker,
            .side = FE::geometry::CutIntegrationSide::Negative,
            .level_set_field = phi_field,
            .enable_level_set_shape_tangent = true,
            .diagnostic = "poisson_level_set_cut_domain_regression"}});

    formulations::poisson::PoissonOptions opts;
    opts.field_name = "Temperature";
    opts.diffusion = 2.0;
    opts.source = 0.5;

    formulations::poisson::PoissonModule module(space, opts);
    module.registerOn(system);
    system.setFormInstallCellDomainRestrictions({});

    const auto u_field = system.findFieldByName("Temperature");
    ASSERT_NE(u_field, FE::INVALID_FIELD_ID);

    const auto& def = system.operatorDefinition("equations");
    EXPECT_TRUE(def.cells.empty());
    const auto scalar_cut = std::find_if(
        def.cut_volumes.begin(),
        def.cut_volumes.end(),
        [&](const auto& entry) {
            return entry.marker == marker &&
                   entry.side == FE::geometry::CutIntegrationSide::Negative &&
                   entry.test_field == u_field &&
                   entry.trial_field == u_field;
        });
    ASSERT_NE(scalar_cut, def.cut_volumes.end());

    const auto phi_cut = std::find_if(
        def.cut_volumes.begin(),
        def.cut_volumes.end(),
        [&](const auto& entry) {
            return entry.marker == marker &&
                   entry.side == FE::geometry::CutIntegrationSide::Negative &&
                   entry.test_field == u_field &&
                   entry.trial_field == phi_field;
        });
    ASSERT_NE(phi_cut, def.cut_volumes.end());

    const auto phi_interface = std::find_if(
        def.interface_faces.begin(),
        def.interface_faces.end(),
        [&](const auto& entry) {
            return entry.marker == marker &&
                   entry.test_field == u_field &&
                   entry.trial_field == phi_field;
        });
    ASSERT_NE(phi_interface, def.interface_faces.end());
    EXPECT_EQ(phi_interface->kernel, phi_cut->kernel);
}

TEST(PoissonModule, LevelSetCutDomainRestrictionAssemblesScalarCutVolume)
{
    auto mesh = std::make_shared<SingleTetraMeshAccess>();
    FE::systems::FESystem system(mesh);

    auto space = std::make_shared<FE::spaces::H1Space>(
        FE::ElementType::Tetra4, /*order=*/1);
    const auto phi_field = system.addField(
        FE::systems::FieldSpec{.name = "phi", .space = space, .components = 1});

    constexpr int marker = 715;
    constexpr std::uint64_t source_revision = 1;
    system.setFormInstallCellDomainRestrictions({
        FE::systems::FESystem::FormCellDomainRestriction{
            .interface_marker = marker,
            .side = FE::geometry::CutIntegrationSide::Negative,
            .level_set_field = phi_field,
            .enable_level_set_shape_tangent = true,
            .diagnostic = "poisson_level_set_cut_domain_assembly_regression"}});

    formulations::poisson::PoissonOptions opts;
    opts.field_name = "Temperature";
    opts.diffusion = 2.0;
    opts.source = 0.5;

    formulations::poisson::PoissonModule module(space, opts);
    module.registerOn(system);
    system.setFormInstallCellDomainRestrictions({});

    auto cut_context = std::make_shared<FE::assembly::CutIntegrationContext>();
    FE::geometry::CutQuadratureRule rule;
    rule.kind = FE::geometry::CutQuadratureKind::Volume;
    rule.side = FE::geometry::CutIntegrationSide::Negative;
    rule.parent_measure = FE::Real{1.0} / FE::Real{6.0};
    rule.measure = FE::Real{1.0} / FE::Real{12.0};
    rule.volume_fraction = FE::Real{0.5};
    rule.frame = FE::geometry::CutGeometryFrame::Reference;
    rule.provenance.parent_entity = 0;
    rule.provenance.marker = marker;
    rule.provenance.source_value_revision = source_revision;
    rule.points.push_back(FE::geometry::CutQuadraturePoint{
        .point = {{FE::Real{0.25}, FE::Real{0.25}, FE::Real{0.25}}},
        .weight = rule.measure,
    });

    FE::assembly::CutCellAssemblyMetadata metadata;
    metadata.parent_entity = 0;
    metadata.side = FE::geometry::CutIntegrationSide::Negative;
    metadata.volume_fraction = rule.volume_fraction;
    metadata.source_value_revision = source_revision;
    cut_context->addGeneratedVolumeRule(marker, metadata, rule);

    FE::interfaces::CutInterfaceDomainRequest domain_request;
    domain_request.source =
        FE::interfaces::LevelSetInterfaceSource::fromField(
            phi_field,
            0u,
            source_revision);
    domain_request.interface_marker = marker;
    domain_request.quadrature_order = 0;
    domain_request.interface_quadrature_order = 0;
    domain_request.volume_quadrature_order = 0;
    FE::interfaces::LevelSetInterfaceDomain domain(domain_request);
    FE::interfaces::CutInterfaceFragment fragment;
    fragment.interface_marker = marker;
    fragment.parent_cell = 0;
    fragment.local_fragment_index = 0;
    fragment.stable_id = 1;
    fragment.kind = FE::interfaces::CutInterfaceFragmentKind::Polygon;
    fragment.measure = std::sqrt(FE::Real{3.0}) / FE::Real{8.0};
    fragment.normal = {{
        FE::Real{1.0} / std::sqrt(FE::Real{3.0}),
        FE::Real{1.0} / std::sqrt(FE::Real{3.0}),
        FE::Real{1.0} / std::sqrt(FE::Real{3.0}),
    }};
    FE::interfaces::CutInterfaceQuadraturePoint interface_qp;
    interface_qp.point = {{
        FE::Real{1.0} / FE::Real{6.0},
        FE::Real{1.0} / FE::Real{6.0},
        FE::Real{1.0} / FE::Real{6.0},
    }};
    interface_qp.parent_coordinate = interface_qp.point;
    interface_qp.normal = fragment.normal;
    interface_qp.weight = fragment.measure;
    fragment.quadrature_points.push_back(interface_qp);
    domain.addFragment(std::move(fragment));
    cut_context->addGeneratedInterfaceDomain(domain);

    system.setCutIntegrationContext(cut_context);
    system.setup({}, makeSingleTetraSetupInputs());

    const auto temperature_field = system.findFieldByName("Temperature");
    ASSERT_NE(temperature_field, FE::INVALID_FIELD_ID);
    const auto& def = system.operatorDefinition("equations");
    EXPECT_FALSE(def.cut_volumes.empty());
    EXPECT_FALSE(def.interface_faces.empty());
    const auto n = system.dofHandler().getNumDofs();
    ASSERT_EQ(n, 8);

    std::vector<FE::Real> values(static_cast<std::size_t>(n), FE::Real{1.0});
    FE::systems::SystemStateView state;
    state.u = std::span<const FE::Real>(values);

    FE::assembly::DenseMatrixView matrix(n);
    FE::assembly::DenseVectorView residual(n);
    FE::systems::AssemblyRequest req;
    req.op = "equations";
    req.want_matrix = true;
    req.want_vector = true;

    const auto result = system.assemble(req, state, &matrix, &residual);
    ASSERT_TRUE(result.success) << result.error_message;
    EXPECT_GT(result.matrix_entries_inserted, 0);
    EXPECT_GT(result.vector_entries_inserted, 0);
    EXPECT_GT(result.interface_faces_assembled, 0);
}

TEST(PoissonModule,
     LevelSetCutDomainRestrictionPrescribedGeometryJacobianMatchesFiniteDifference)
{
    auto mesh = std::make_shared<SingleTetraMeshAccess>();
    FE::systems::FESystem system(mesh);

    auto space = std::make_shared<FE::spaces::H1Space>(
        FE::ElementType::Tetra4, /*order=*/1);
    const auto phi_field = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    constexpr int marker = 716;
    constexpr std::uint64_t source_revision = 1;
    system.setFormInstallCellDomainRestrictions({
        FE::systems::FESystem::FormCellDomainRestriction{
            .interface_marker = marker,
            .side = FE::geometry::CutIntegrationSide::Negative,
            .level_set_field = phi_field,
            .enable_level_set_shape_tangent = false,
            .diagnostic = "poisson_level_set_cut_domain_fd_regression"}});

    formulations::poisson::PoissonOptions opts;
    opts.field_name = "Temperature";
    opts.diffusion = 1.7;
    opts.source = 0.35;

    formulations::poisson::PoissonModule module(space, opts);
    module.registerOn(system);
    system.setFormInstallCellDomainRestrictions({});
    system.setCutIntegrationContext(
        makePoissonSingleTetraGeneratedCutContext(
            phi_field, marker, source_revision));
    system.setup({}, makeSingleTetraSetupInputs());
    const std::vector<FE::Real> prescribed_phi{
        FE::Real{-0.25}, FE::Real{0.15}, FE::Real{-0.10}, FE::Real{0.20}};
    system.setPrescribedFieldCoefficients(
        phi_field,
        std::span<const FE::Real>(prescribed_phi.data(),
                                  prescribed_phi.size()));

    const auto n = system.dofHandler().getNumDofs();
    ASSERT_EQ(n, 4);
    std::vector<FE::Real> values(static_cast<std::size_t>(n), FE::Real{0.0});
    values[0] = FE::Real{0.11};
    values[1] = FE::Real{-0.03};
    values[2] = FE::Real{0.07};
    values[3] = FE::Real{0.19};

    FE::systems::SystemStateView state;
    state.u = std::span<const FE::Real>(values);

    expectOperatorJacobianMatchesCentralFD(
        system,
        state,
        "equations",
        /*eps=*/1.0e-6,
        /*rtol=*/1.0e-6,
        /*atol=*/1.0e-10);
}

TEST(PoissonModule,
     LevelSetCutDomainRestrictionPrescribedGeometrySolveConservesGeneratedMeasure)
{
    std::string backend_error;
    auto factory = tryCreateBackend(FE::backends::BackendKind::Eigen,
                                    backend_error);
    if (!factory) {
        GTEST_SKIP() << "Requires Eigen backend: " << backend_error;
    }

    auto mesh = std::make_shared<SingleTetraMeshAccess>();
    FE::systems::FESystem system(mesh);

    auto space = std::make_shared<FE::spaces::H1Space>(
        FE::ElementType::Tetra4, /*order=*/1);
    const auto phi_field = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    constexpr int marker = 717;
    constexpr std::uint64_t source_revision = 1;
    system.setFormInstallCellDomainRestrictions({
        FE::systems::FESystem::FormCellDomainRestriction{
            .interface_marker = marker,
            .side = FE::geometry::CutIntegrationSide::Negative,
            .level_set_field = phi_field,
            .enable_level_set_shape_tangent = false,
            .diagnostic = "poisson_level_set_cut_domain_solve_regression"}});

    formulations::poisson::PoissonOptions opts;
    opts.field_name = "Temperature";
    opts.diffusion = 1.2;
    opts.source = 0.4;

    formulations::poisson::PoissonModule module(space, opts);
    module.registerOn(system);
    system.setFormInstallCellDomainRestrictions({});

    const auto temperature_field = system.findFieldByName("Temperature");
    ASSERT_NE(temperature_field, FE::INVALID_FIELD_ID);
    std::vector<FE::constraints::VertexDirichletValue> anchor = {
        {.vertex_id = 0, .value = 0.0},
    };
    system.addSystemConstraint(
        std::make_unique<FE::constraints::VertexDirichletConstraint>(
            temperature_field,
            std::move(anchor),
            FE::constraints::VertexIdMode::LocalVertexId));

    system.setCutIntegrationContext(
        makePoissonSingleTetraGeneratedCutContext(
            phi_field, marker, source_revision));
    system.setup({}, makeSingleTetraSetupInputs());
    const std::vector<FE::Real> prescribed_phi{
        FE::Real{-0.25}, FE::Real{0.15}, FE::Real{-0.10}, FE::Real{0.20}};
    system.setPrescribedFieldCoefficients(
        phi_field,
        std::span<const FE::Real>(prescribed_phi.data(),
                                  prescribed_phi.size()));

    ASSERT_TRUE(system.constraints().isConstrained(0));
    const auto* cut_context = system.cutIntegrationContext();
    ASSERT_NE(cut_context, nullptr);
    const auto rules =
        cut_context->generatedVolumeRulesForMarkerAndSide(
            marker, FE::geometry::CutIntegrationSide::Negative);
    ASSERT_EQ(rules.size(), 1u);
    ASSERT_NE(rules.front(), nullptr);
    EXPECT_NEAR(rules.front()->measure,
                rules.front()->parent_measure * rules.front()->volume_fraction,
                FE::Real{1.0e-14});
    EXPECT_NEAR(rules.front()->measure,
                FE::Real{1.0} / FE::Real{12.0},
                FE::Real{1.0e-14});
    EXPECT_NEAR(rules.front()->volume_fraction, FE::Real{0.5},
                FE::Real{1.0e-14});

    const auto solution = solveSteadySystem(system,
                                            *factory,
                                            FE::backends::SolverMethod::Direct,
                                            FE::backends::PreconditionerType::None);
    ASSERT_EQ(solution.size(), static_cast<std::size_t>(4));
    EXPECT_NEAR(solution.front(), FE::Real{0.0}, FE::Real{1.0e-12});
    for (const auto value : solution) {
        EXPECT_TRUE(std::isfinite(value));
    }
}

TEST(PoissonModule, CoupledNeumannRCRUsesModernAuxiliaryPath)
{
    constexpr int marker = 7;
    auto mesh = std::make_shared<FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    FE::systems::FESystem system(mesh);

    auto space = std::make_shared<FE::spaces::H1Space>(FE::ElementType::Tetra4, /*order=*/1);
    formulations::poisson::PoissonOptions opts;
    opts.field_name = "u";
    opts.diffusion = 1.0;
    opts.source = 0.0;
    opts.coupled_neumann_rcr.push_back(formulations::poisson::PoissonOptions::CoupledRCRNeumannBC{
        .boundary_marker = marker,
        .Rp = 10.0,
        .C = 0.001,
        .Rd = 100.0,
        .Pd = 50.0,
        .X0 = 50.0,
    });

    formulations::poisson::PoissonModule module(space, opts);
    module.registerOn(system);

    system.setup({}, makeSingleTetraSetupInputs());
    system.finalizeAuxiliaryLayout();

    const auto* aux_inputs = system.auxiliaryInputRegistryIfPresent();
    ASSERT_NE(aux_inputs, nullptr);
    EXPECT_EQ(aux_inputs->totalSize(), 1u);

    const auto summary = system.auxiliaryAnalysisSummary();
    EXPECT_EQ(summary.n_monolithic, 1u);
    EXPECT_EQ(summary.n_partitioned, 0u);

    const auto out_slot = system.auxiliaryOutputSlotOf("poisson_rcr_7", "flux");
    EXPECT_NE(out_slot, std::string::npos);
}

TEST(PoissonModule, BoundaryScalarValueAcceptsFormExpr)
{
    constexpr int marker = 8;
    auto mesh = std::make_shared<FE::forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    FE::systems::FESystem system(mesh);

    auto space = std::make_shared<FE::spaces::H1Space>(FE::ElementType::Tetra4, /*order=*/1);
    formulations::poisson::PoissonOptions opts;
    opts.field_name = "u";
    opts.diffusion = 1.0;
    opts.source = 0.0;
    opts.neumann.push_back(formulations::poisson::PoissonOptions::NeumannBC{
        .boundary_marker = marker,
        .flux = FE::forms::FormExpr::constant(2.0),
    });

    formulations::poisson::PoissonModule module(space, opts);
    module.registerOn(system);
    system.setup({}, makeSingleTetraSetupInputs());

    const auto n = system.dofHandler().getNumDofs();
    ASSERT_EQ(n, 4);
    std::vector<FE::Real> u(static_cast<std::size_t>(n), 0.0);
    FE::systems::SystemStateView state;
    state.u = std::span<const FE::Real>(u);

    FE::assembly::DenseVectorView residual(n);
    residual.zero();
    FE::systems::AssemblyRequest req;
    req.op = "equations";
    req.want_vector = true;
    const auto result = system.assemble(req, state, nullptr, &residual);
    ASSERT_TRUE(result.success) << result.error_message;

    FE::Real norm2 = 0.0;
    for (FE::GlobalIndex i = 0; i < n; ++i) {
        norm2 += residual[i] * residual[i];
    }
    EXPECT_GT(norm2, 0.0);
}

TEST(PoissonModule, OrdinaryPoissonDoesNotRegisterDarcyFluxPostprocessing)
{
    auto mesh = std::make_shared<SingleTetraMeshAccess>();
    FE::systems::FESystem system(mesh);

    auto space = std::make_shared<FE::spaces::H1Space>(FE::ElementType::Tetra4, /*order=*/1);
    formulations::poisson::PoissonOptions opts;
    opts.field_name = "u";
    opts.diffusion = 1.0;
    opts.source = 0.25;

    formulations::poisson::PoissonModule module(space, opts);
    module.registerOn(system);

    EXPECT_TRUE(system.derivedResults().empty());
}

TEST(PoissonModule, DarcyAliasReadsNodePressureCsvAndConstrainsGlobalVertexGid)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    auto mesh = buildPoissonNodeConstraintQuadMesh();
    const auto csv_path = uniqueNodePressureCsvPath(mesh.get());
    writeTextFile(csv_path, "node_id,pressure\n30,3.25\n");

    EquationModuleInput input{};
    input.equation_type = "darcy";
    input.mesh_name = "quad";
    input.mesh = mesh->local_mesh_ptr();
    input.node_pressure_constraints = NodePressureConstraintInput{
        .id_type = "Global_vertex_gid",
        .values_file_path = csv_path.string(),
    };

    FE::systems::FESystem system(mesh);
    auto module = EquationModuleRegistry::instance().create("darcy", input, system);
    ASSERT_TRUE(module);
    EXPECT_EQ(system.fieldRecord(0).name, "Pressure");

    ASSERT_EQ(system.derivedResults().size(), 2u);
    const auto& cell_flux = system.derivedResults()[0];
    EXPECT_EQ(cell_flux.name, "Darcy_flux");
    EXPECT_EQ(cell_flux.scope, FE::post::DerivedResultScope::Cell);
    EXPECT_EQ(cell_flux.policy, FE::post::DerivedResultPolicy::CellAverage);
    EXPECT_EQ(cell_flux.shape.components, 2);

    const auto& node_flux = system.derivedResults()[1];
    EXPECT_EQ(node_flux.name, "Darcy_flux_node");
    EXPECT_EQ(node_flux.scope, FE::post::DerivedResultScope::Vertex);
    EXPECT_EQ(node_flux.policy, FE::post::DerivedResultPolicy::PatchAverage);
    EXPECT_EQ(node_flux.shape.components, 2);

    ASSERT_NO_THROW(system.setup());

    const FE::FieldId field = 0;
    const auto* entity = system.fieldDofHandler(field).getEntityDofMap();
    ASSERT_NE(entity, nullptr);
    const auto vertex_dofs = entity->getVertexDofs(2);
    ASSERT_EQ(vertex_dofs.size(), 1u);
    const auto dof = vertex_dofs.front() + system.fieldDofOffset(field);

    EXPECT_TRUE(system.constraints().isConstrained(dof));
    EXPECT_NEAR(system.constraints().getInhomogeneity(dof), 3.25, 1e-12);

    std::filesystem::remove(csv_path);
#endif
}

TEST(PoissonModule, NodePressureCsvRejectsConflictingDuplicateNodeValues)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    auto mesh = buildPoissonNodeConstraintQuadMesh();
    const auto csv_path = uniqueNodePressureCsvPath(mesh.get());
    writeTextFile(csv_path, "node_id,pressure\n30,3.25\n30,4.0\n");

    EquationModuleInput input{};
    input.equation_type = "darcy";
    input.mesh_name = "quad";
    input.mesh = mesh->local_mesh_ptr();
    input.node_pressure_constraints = NodePressureConstraintInput{
        .id_type = "Global_vertex_gid",
        .values_file_path = csv_path.string(),
    };

    FE::systems::FESystem system(mesh);
    EXPECT_THROW((void)EquationModuleRegistry::instance().create("darcy", input, system), std::runtime_error);

    std::filesystem::remove(csv_path);
#endif
}

TEST(PoissonModule, NodePressureCsvAcceptsNoHeaderCommentsAndMatchingDuplicates)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    auto mesh = buildPoissonNodeConstraintQuadMesh();
    const auto csv_path = uniqueNodePressureCsvPath(mesh.get());
    writeTextFile(csv_path, "# known pressure nodes\n30, 3.25\n30,3.25\n");

    EquationModuleInput input{};
    input.equation_type = "darcy";
    input.mesh_name = "quad";
    input.mesh = mesh->local_mesh_ptr();
    input.node_pressure_constraints = NodePressureConstraintInput{
        .id_type = "Global_vertex_gid",
        .values_file_path = csv_path.string(),
    };

    FE::systems::FESystem system(mesh);
    auto module = EquationModuleRegistry::instance().create("darcy", input, system);
    ASSERT_TRUE(module);
    ASSERT_NO_THROW(system.setup());

    const auto* entity = system.fieldDofHandler(0).getEntityDofMap();
    ASSERT_NE(entity, nullptr);
    const auto vertex_dofs = entity->getVertexDofs(2);
    ASSERT_EQ(vertex_dofs.size(), 1u);
    const auto dof = vertex_dofs.front() + system.fieldDofOffset(0);

    EXPECT_TRUE(system.constraints().isConstrained(dof));
    EXPECT_NEAR(system.constraints().getInhomogeneity(dof), 3.25, 1e-12);

    std::filesystem::remove(csv_path);
#endif
}

TEST(PoissonModule, NodePressureCsvRejectsMissingFile)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    auto mesh = buildPoissonNodeConstraintQuadMesh();
    const auto csv_path = uniqueNodePressureCsvPath(mesh.get());
    std::filesystem::remove(csv_path);

    EquationModuleInput input{};
    input.equation_type = "darcy";
    input.mesh_name = "quad";
    input.mesh = mesh->local_mesh_ptr();
    input.node_pressure_constraints = NodePressureConstraintInput{
        .id_type = "Global_vertex_gid",
        .values_file_path = csv_path.string(),
    };

    FE::systems::FESystem system(mesh);
    EXPECT_THROW((void)EquationModuleRegistry::instance().create("darcy", input, system), std::runtime_error);
#endif
}

TEST(PoissonModule, NodePressureCsvRejectsMalformedRows)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    auto mesh = buildPoissonNodeConstraintQuadMesh();
    const auto csv_path = uniqueNodePressureCsvPath(mesh.get());
    writeTextFile(csv_path, "node_id,pressure\n30\n");

    EquationModuleInput input{};
    input.equation_type = "darcy";
    input.mesh_name = "quad";
    input.mesh = mesh->local_mesh_ptr();
    input.node_pressure_constraints = NodePressureConstraintInput{
        .id_type = "Global_vertex_gid",
        .values_file_path = csv_path.string(),
    };

    FE::systems::FESystem system(mesh);
    EXPECT_THROW((void)EquationModuleRegistry::instance().create("darcy", input, system), std::runtime_error);

    std::filesystem::remove(csv_path);
#endif
}

TEST(PoissonModule, NodePressureCsvRejectsUnsupportedIdType)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    auto mesh = buildPoissonNodeConstraintQuadMesh();
    const auto csv_path = uniqueNodePressureCsvPath(mesh.get());
    writeTextFile(csv_path, "node_id,pressure\n30,3.25\n");

    EquationModuleInput input{};
    input.equation_type = "darcy";
    input.mesh_name = "quad";
    input.mesh = mesh->local_mesh_ptr();
    input.node_pressure_constraints = NodePressureConstraintInput{
        .id_type = "Local_vertex_id",
        .values_file_path = csv_path.string(),
    };

    FE::systems::FESystem system(mesh);
    EXPECT_THROW((void)EquationModuleRegistry::instance().create("darcy", input, system), std::runtime_error);

    std::filesystem::remove(csv_path);
#endif
}

TEST(PoissonModule, DarcyFaceMarkerDirichletPressureStillConstrainsBoundaryDofs)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    constexpr int marker = 17;
    auto mesh = buildBoundaryMarkedQuadMesh(marker);

    EquationModuleInput input{};
    input.equation_type = "darcy";
    input.mesh_name = "quad";
    input.mesh = mesh->local_mesh_ptr();
    input.boundary_conditions.push_back(dirichletPressureBC(marker, 0.0));

    FE::systems::FESystem system(mesh);
    auto module = EquationModuleRegistry::instance().create("darcy", input, system);
    ASSERT_TRUE(module);
    ASSERT_NO_THROW(system.setup());

    const auto* entity = system.fieldDofHandler(0).getEntityDofMap();
    ASSERT_NE(entity, nullptr);
    for (const FE::GlobalIndex vertex : {FE::GlobalIndex{0}, FE::GlobalIndex{3}}) {
        const auto vertex_dofs = entity->getVertexDofs(vertex);
        ASSERT_EQ(vertex_dofs.size(), 1u);
        const auto dof = vertex_dofs.front() + system.fieldDofOffset(0);
        EXPECT_TRUE(system.constraints().isConstrained(dof));
        EXPECT_NEAR(system.constraints().getInhomogeneity(dof), 0.0, 1e-12);
    }
#endif
}

TEST(PoissonModule, DarcyNodePressureConflictsWithFaceMarkerDirichletPressure)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    constexpr int marker = 18;
    auto mesh = buildBoundaryMarkedQuadMesh(marker);
    const auto csv_path = uniqueNodePressureCsvPath(mesh.get());
    writeTextFile(csv_path, "node_id,pressure\n10,2.0\n");

    EquationModuleInput input{};
    input.equation_type = "darcy";
    input.mesh_name = "quad";
    input.mesh = mesh->local_mesh_ptr();
    input.boundary_conditions.push_back(dirichletPressureBC(marker, 0.0));
    input.node_pressure_constraints = NodePressureConstraintInput{
        .id_type = "Global_vertex_gid",
        .values_file_path = csv_path.string(),
    };

    FE::systems::FESystem system(mesh);
    auto module = EquationModuleRegistry::instance().create("darcy", input, system);
    ASSERT_TRUE(module);
    EXPECT_THROW(system.setup(), std::exception);

    std::filesystem::remove(csv_path);
#endif
}

TEST(PoissonModule, DarcySerialSolveHonorsInteriorNodePressureConstraint)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    std::string backend_error;
    auto factory = tryCreateBackend(FE::backends::BackendKind::FSILS, backend_error);
    if (!factory) {
        GTEST_SKIP() << "Requires FSILS backend: " << backend_error;
    }

    constexpr int marker = 19;
    auto mesh = buildStructuredQuadPressureMesh(/*nx=*/4, /*ny=*/4, marker);
    constexpr FE::GlobalIndex center_vertex = 12;
    constexpr gid_t center_gid = 1000 + center_vertex;
    constexpr FE::Real center_pressure = 2.75;

    const auto csv_path = uniqueNodePressureCsvPath(mesh.get());
    writeTextFile(csv_path, "node_id,pressure\n" + std::to_string(center_gid) + "," +
                                std::to_string(static_cast<double>(center_pressure)) + "\n");

    EquationModuleInput input{};
    input.equation_type = "darcy";
    input.mesh_name = "structured_quad";
    input.mesh = mesh->local_mesh_ptr();
    input.boundary_conditions.push_back(dirichletPressureBC(marker, 0.0));
    input.node_pressure_constraints = NodePressureConstraintInput{
        .id_type = "Global_vertex_gid",
        .values_file_path = csv_path.string(),
    };

    FE::systems::FESystem system(mesh);
    auto module = EquationModuleRegistry::instance().create("darcy", input, system);
    ASSERT_TRUE(module);
    ASSERT_NO_THROW(system.setup());

    const auto solution = solveSteadySystem(system,
                                            *factory,
                                            FE::backends::SolverMethod::CG,
                                            FE::backends::PreconditionerType::None);

    const auto* entity = system.fieldDofHandler(0).getEntityDofMap();
    ASSERT_NE(entity, nullptr);
    const auto vertex_dofs = entity->getVertexDofs(center_vertex);
    ASSERT_EQ(vertex_dofs.size(), 1u);
    const auto center_dof = vertex_dofs.front() + system.fieldDofOffset(0);
    ASSERT_LT(static_cast<std::size_t>(center_dof), solution.size());
    EXPECT_NEAR(solution[static_cast<std::size_t>(center_dof)], center_pressure, 1e-10);

    std::filesystem::remove(csv_path);
#endif
}

#if FE_HAS_MPI || defined(MESH_HAS_MPI)
namespace {

FE::GlobalIndex selectOwnedInteriorVertexGid(const Mesh& mesh)
{
    const auto bbox = mesh.global_bounding_box();
    const double xmin = static_cast<double>(bbox.min[0]);
    const double xmax = static_cast<double>(bbox.max[0]);
    const double ymin = static_cast<double>(bbox.min[1]);
    const double ymax = static_cast<double>(bbox.max[1]);
    constexpr double tol = 1e-10;

    const auto& base = mesh.base();
    const auto& gids = base.vertex_gids();
    FE::GlobalIndex local_candidate = -1;
    for (index_t v = 0; v < static_cast<index_t>(base.n_vertices()); ++v) {
        if (!mesh.is_owned_vertex(v)) {
            continue;
        }
        const auto xyz = base.get_vertex_coords(v);
        const double x = static_cast<double>(xyz[0]);
        const double y = static_cast<double>(xyz[1]);
        if (std::abs(x - xmin) <= tol || std::abs(x - xmax) <= tol ||
            std::abs(y - ymin) <= tol || std::abs(y - ymax) <= tol) {
            continue;
        }
        local_candidate = static_cast<FE::GlobalIndex>(gids[static_cast<std::size_t>(v)]);
        break;
    }

    const long long local_ll = static_cast<long long>(local_candidate);
    long long global_ll = -1;
    MPI_Allreduce(&local_ll, &global_ll, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    return static_cast<FE::GlobalIndex>(global_ll);
}

std::vector<BoundaryConditionInput> zeroSquareBoundaryConditions()
{
    return {
        dirichletPressureBC(static_cast<int>(kSquareBoundaryLeft), 0.0),
        dirichletPressureBC(static_cast<int>(kSquareBoundaryRight), 0.0),
        dirichletPressureBC(static_cast<int>(kSquareBoundaryBottom), 0.0),
        dirichletPressureBC(static_cast<int>(kSquareBoundaryTop), 0.0),
    };
}

} // namespace
#endif

TEST(PoissonModuleMPI, DarcyNodePressureCsvSolvesWithInteriorConstraint)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#elif !defined(MESH_HAS_VTK)
    GTEST_SKIP() << "Requires Mesh built with VTK support (MESH_ENABLE_VTK=ON).";
#elif !(FE_HAS_MPI || defined(MESH_HAS_MPI))
    GTEST_SKIP() << "Requires MPI-enabled FE or Mesh build.";
#else
    const auto world = MeshComm::world();
    if (world.size() != 2) {
        GTEST_SKIP() << "This test requires exactly 2 MPI ranks";
    }

    std::string backend_error;
    auto factory = tryCreateBackend(FE::backends::BackendKind::FSILS, backend_error);
    int backend_ok = factory ? 1 : 0;
    int all_backend_ok = 0;
    MPI_Allreduce(&backend_ok, &all_backend_ok, 1, MPI_INT, MPI_MIN, world.native());
    if (!all_backend_ok) {
        if (world.rank() == 0) {
            GTEST_SKIP() << "Requires FSILS backend: " << backend_error;
        }
        GTEST_SKIP();
    }

    auto mesh = loadSquareMeshWithMarkedBoundaries(world);
    ASSERT_TRUE(mesh);
    const auto known_gid = selectOwnedInteriorVertexGid(*mesh);
    if (known_gid < 0) {
        if (world.rank() == 0) {
            ADD_FAILURE() << "No owned interior vertex gid was available for MPI Darcy node-pressure test.";
        }
        return;
    }

    const FE::Real known_pressure = 1.5;
    const auto csv_path = std::filesystem::temp_directory_path() /
                          ("svmp_node_pressure_mpi_" + std::to_string(known_gid) + ".csv");
    if (world.rank() == 0) {
        writeTextFile(csv_path, "node_id,pressure\n" + std::to_string(known_gid) + "," +
                                    std::to_string(static_cast<double>(known_pressure)) + "\n");
    }
    MPI_Barrier(world.native());

    EquationModuleInput input{};
    input.equation_type = "darcy";
    input.mesh_name = "square";
    input.mesh = std::const_pointer_cast<Mesh>(mesh)->local_mesh_ptr();
    input.boundary_conditions = zeroSquareBoundaryConditions();
    input.node_pressure_constraints = NodePressureConstraintInput{
        .id_type = "Global_vertex_gid",
        .values_file_path = csv_path.string(),
    };

    FE::systems::FESystem system(mesh);
    auto module = EquationModuleRegistry::instance().create("darcy", input, system);
    ASSERT_TRUE(module);
    ASSERT_NO_THROW(system.setup());

    auto history = solveSteadyHistory(system,
                                      *factory,
                                      FE::backends::SolverMethod::CG,
                                      FE::backends::PreconditionerType::None,
                                      /*rel_tol=*/1e-8,
                                      /*abs_tol=*/1e-10);
    history.u().updateGhosts();
    auto u_view = history.u().createAssemblyView();

    const auto local_vertex = mesh->base().global_to_local_vertex(static_cast<gid_t>(known_gid));
    double local_error = 0.0;
    int local_checked = 0;
    if (local_vertex != INVALID_INDEX && mesh->is_owned_vertex(local_vertex)) {
        const auto* entity = system.fieldDofHandler(0).getEntityDofMap();
        ASSERT_NE(entity, nullptr);
        ASSERT_TRUE(u_view);
        const auto vertex_dofs = entity->getVertexDofs(static_cast<FE::GlobalIndex>(local_vertex));
        ASSERT_EQ(vertex_dofs.size(), 1u);
        const auto dof = vertex_dofs.front() + system.fieldDofOffset(0);
        local_error = std::abs(static_cast<double>(u_view->getVectorEntry(dof) - known_pressure));
        local_checked = 1;
    }

    double global_error = local_error;
    int global_checked = 0;
    MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_MAX, world.native());
    MPI_Allreduce(&local_checked, &global_checked, 1, MPI_INT, MPI_SUM, world.native());
    if (world.rank() == 0) {
        EXPECT_EQ(global_checked, 1);
        EXPECT_LT(global_error, 1e-8);
    }

    MPI_Barrier(world.native());
    if (world.rank() == 0) {
        std::filesystem::remove(csv_path);
    }
#endif
}

TEST(PoissonModuleMPI, DarcyNodePressureMissingGlobalGidFailsGlobally)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#elif !defined(MESH_HAS_VTK)
    GTEST_SKIP() << "Requires Mesh built with VTK support (MESH_ENABLE_VTK=ON).";
#elif !(FE_HAS_MPI || defined(MESH_HAS_MPI))
    GTEST_SKIP() << "Requires MPI-enabled FE or Mesh build.";
#else
    const auto world = MeshComm::world();
    if (world.size() != 2) {
        GTEST_SKIP() << "This test requires exactly 2 MPI ranks";
    }

    auto mesh = loadSquareMeshWithMarkedBoundaries(world);
    ASSERT_TRUE(mesh);

    constexpr FE::GlobalIndex missing_gid = 999999999;
    const auto csv_path = std::filesystem::temp_directory_path() /
                          ("svmp_node_pressure_mpi_missing_" + std::to_string(missing_gid) + ".csv");
    if (world.rank() == 0) {
        writeTextFile(csv_path, "node_id,pressure\n" + std::to_string(missing_gid) + ",1.0\n");
    }
    MPI_Barrier(world.native());

    EquationModuleInput input{};
    input.equation_type = "darcy";
    input.mesh_name = "square";
    input.mesh = std::const_pointer_cast<Mesh>(mesh)->local_mesh_ptr();
    input.boundary_conditions = zeroSquareBoundaryConditions();
    input.node_pressure_constraints = NodePressureConstraintInput{
        .id_type = "Global_vertex_gid",
        .values_file_path = csv_path.string(),
    };

    FE::systems::FESystem system(mesh);
    auto module = EquationModuleRegistry::instance().create("darcy", input, system);
    ASSERT_TRUE(module);

    int local_threw = 0;
    try {
        system.setup();
    } catch (const std::invalid_argument&) {
        local_threw = 1;
    }

    int all_threw = 0;
    MPI_Allreduce(&local_threw, &all_threw, 1, MPI_INT, MPI_MIN, world.native());
    if (world.rank() == 0) {
        EXPECT_EQ(all_threw, 1);
    }

    MPI_Barrier(world.native());
    if (world.rank() == 0) {
        std::filesystem::remove(csv_path);
    }
#endif
}

} // namespace test
} // namespace Physics
} // namespace svmp
