/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Core/EquationModuleRegistry.h"
#include "Physics/Formulations/MeshMotion/HarmonicMeshMotionModule.h"
#include "Physics/Formulations/MeshMotion/PseudoElasticMeshMotionModule.h"
#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"
#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"
#include "Physics/Tests/Unit/PhysicsTestHelpers.h"

#include "FE/Forms/FormExpr.h"
#include "FE/Forms/FormCompiler.h"
#include "FE/Forms/FormKernels.h"
#include "FE/Forms/StandardBCs.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Assembly/StandardAssembler.h"
#include "FE/Analysis/FormExprScanner.h"
#include "FE/Basis/LagrangeBasis.h"
#include "FE/Dofs/DofMap.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Geometry/FrameGeometry.h"
#include "FE/Geometry/IsoparametricMapping.h"
#include "FE/LevelSet/LevelSetInterfaceLifecycle.h"
#include "FE/Quadrature/QuadratureFactory.h"
#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/ProductSpace.h"
#include "FE/Spaces/SpaceFactory.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"
#include "FE/Systems/TimeIntegrator.h"
#include "FE/Tests/Unit/Forms/FormsTestHelpers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#if FE_HAS_MPI || defined(MESH_HAS_MPI)
#  include <mpi.h>
#endif

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "FE/Assembly/MeshAccess.h"
#  include "Mesh/Mesh.h"
#  include "Mesh/Topology/CellShape.h"
#endif

namespace svmp {
namespace Physics {
namespace test {
namespace {

using FE::forms::FormExpr;
using FE::forms::FormExprNode;
using FE::forms::FormExprType;
constexpr FE::FieldId kMeshVelocityField = 907;
namespace mm = formulations::mesh_motion;
namespace ls = FE::level_set;
namespace ns = formulations::navier_stokes;

bool containsExprType(const FormExprNode* node, FormExprType target)
{
    if (node == nullptr) {
        return false;
    }
    if (node->type() == target) {
        return true;
    }
    for (const auto* child : node->children()) {
        if (containsExprType(child, target)) {
            return true;
        }
    }
    return false;
}

bool containsExprType(const FormExpr& expr, FormExprType target)
{
    return expr.isValid() && containsExprType(expr.node(), target);
}

bool formulationRecordsContain(const FE::systems::FESystem& system, FormExprType target)
{
    for (const auto& record : system.formulationRecords()) {
        if (containsExprType(record.residual_expr.get(), target)) {
            return true;
        }
        for (const auto& [block, expr] : record.block_residual_exprs) {
            (void)block;
            if (containsExprType(expr.get(), target)) {
                return true;
            }
        }
    }
    return false;
}

bool formulationRecordsContainInterfaceMarker(const FE::systems::FESystem& system,
                                              int marker)
{
    for (const auto& record : system.formulationRecords()) {
        if (record.residual_expr) {
            const auto scan = FE::analysis::scanFormExpr(*record.residual_expr);
            if (std::find(scan.interface_markers.begin(),
                          scan.interface_markers.end(),
                          marker) != scan.interface_markers.end()) {
                return true;
            }
        }
        for (const auto& [block, expr] : record.block_residual_exprs) {
            (void)block;
            if (!expr) {
                continue;
            }
            const auto scan = FE::analysis::scanFormExpr(*expr);
            if (std::find(scan.interface_markers.begin(),
                          scan.interface_markers.end(),
                          marker) != scan.interface_markers.end()) {
                return true;
            }
        }
    }
    return false;
}

std::shared_ptr<SingleTetraMeshAccess> makeMesh()
{
    return std::make_shared<SingleTetraMeshAccess>();
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
std::shared_ptr<Mesh> makeRegistryQuadMesh()
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
    base->finalize();

    return create_mesh(std::move(base));
}

std::shared_ptr<Mesh> makeStructuredQuadMesh(int cells_per_axis,
                                             FE::Real min_coord,
                                             FE::Real max_coord)
{
    auto base = std::make_shared<MeshBase>();

    const int nodes_per_axis = cells_per_axis + 1;
    const FE::Real h = (max_coord - min_coord) /
                       static_cast<FE::Real>(cells_per_axis);

    std::vector<real_t> x_ref;
    x_ref.reserve(static_cast<std::size_t>(nodes_per_axis * nodes_per_axis * 2));
    for (int j = 0; j < nodes_per_axis; ++j) {
        for (int i = 0; i < nodes_per_axis; ++i) {
            x_ref.push_back(static_cast<real_t>(
                min_coord + h * static_cast<FE::Real>(i)));
            x_ref.push_back(static_cast<real_t>(
                min_coord + h * static_cast<FE::Real>(j)));
        }
    }

    std::vector<offset_t> cell2vertex_offsets;
    std::vector<index_t> cell2vertex;
    cell2vertex_offsets.reserve(
        static_cast<std::size_t>(cells_per_axis * cells_per_axis + 1));
    cell2vertex.reserve(
        static_cast<std::size_t>(cells_per_axis * cells_per_axis * 4));
    cell2vertex_offsets.push_back(0);
    for (int j = 0; j < cells_per_axis; ++j) {
        for (int i = 0; i < cells_per_axis; ++i) {
            const index_t v00 = static_cast<index_t>(j * nodes_per_axis + i);
            const index_t v10 = v00 + 1;
            const index_t v01 = static_cast<index_t>((j + 1) * nodes_per_axis + i);
            const index_t v11 = v01 + 1;
            cell2vertex.push_back(v00);
            cell2vertex.push_back(v10);
            cell2vertex.push_back(v11);
            cell2vertex.push_back(v01);
            cell2vertex_offsets.push_back(static_cast<offset_t>(cell2vertex.size()));
        }
    }

    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    const std::vector<CellShape> cell_shapes(
        static_cast<std::size_t>(cells_per_axis * cells_per_axis),
        shape);
    base->build_from_arrays(
        /*spatial_dim=*/2,
        x_ref,
        cell2vertex_offsets,
        cell2vertex,
        cell_shapes);
    base->finalize();

    return create_mesh(std::move(base));
}

std::shared_ptr<Mesh> makeOpenTankQuadMesh(int left_marker,
                                           int right_marker,
                                           int bottom_marker,
                                           int free_surface_marker,
                                           std::string_view free_surface_set,
                                           FE::Real bottom_y = -1.0,
                                           FE::Real middle_y = 0.0,
                                           FE::Real top_y = 1.0)
{
    auto base = std::make_shared<MeshBase>();

    const std::vector<real_t> x_ref = {
        -1.0, static_cast<real_t>(bottom_y),
         0.0, static_cast<real_t>(bottom_y),
         1.0, static_cast<real_t>(bottom_y),
        -1.0, static_cast<real_t>(middle_y),
         0.0, static_cast<real_t>(middle_y),
         1.0, static_cast<real_t>(middle_y),
        -1.0, static_cast<real_t>(top_y),
         0.0, static_cast<real_t>(top_y),
         1.0, static_cast<real_t>(top_y),
    };
    const std::vector<offset_t> cell2vertex_offsets = {0, 4, 8, 12, 16};
    const std::vector<index_t> cell2vertex = {
        0, 1, 4, 3,
        1, 2, 5, 4,
        3, 4, 7, 6,
        4, 5, 8, 7,
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
        std::vector<CellShape>(4, shape));
    base->finalize();

    base->register_label("wall_left", static_cast<label_t>(left_marker));
    base->register_label("wall_right", static_cast<label_t>(right_marker));
    base->register_label("wall_bottom", static_cast<label_t>(bottom_marker));
    base->register_label("free_surface", static_cast<label_t>(free_surface_marker));

    const auto coordinate = [&](index_t vertex, int component) {
        return base->X_ref().at(static_cast<std::size_t>(2 * vertex + component));
    };
    const auto all_vertices_match = [&](std::span<const index_t> vertices,
                                        int component,
                                        real_t value) {
        return std::all_of(vertices.begin(), vertices.end(), [&](index_t vertex) {
            return std::abs(coordinate(vertex, component) - value) < real_t(1.0e-14);
        });
    };

    for (index_t face = 0; face < static_cast<index_t>(base->n_faces()); ++face) {
        const auto vertices = base->face_vertices(face);
        if (vertices.size() != 2u) {
            continue;
        }
        label_t label = INVALID_LABEL;
        if (all_vertices_match(vertices, /*component=*/1, static_cast<real_t>(top_y))) {
            label = static_cast<label_t>(free_surface_marker);
        } else if (all_vertices_match(vertices, /*component=*/1, static_cast<real_t>(bottom_y))) {
            label = static_cast<label_t>(bottom_marker);
        } else if (all_vertices_match(vertices, /*component=*/0, real_t(-1.0))) {
            label = static_cast<label_t>(left_marker);
        } else if (all_vertices_match(vertices, /*component=*/0, real_t(1.0))) {
            label = static_cast<label_t>(right_marker);
        }

        if (label == INVALID_LABEL) {
            continue;
        }
        base->set_boundary_label(face, label);
        if (label == static_cast<label_t>(free_surface_marker)) {
            base->add_to_set(EntityKind::Face, std::string(free_surface_set), face);
        }
    }

    return create_mesh(std::move(base));
}
#endif

FE::systems::SetupInputs makeSingleTriangleSetupInputs()
{
    FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 3;
    topo.n_edges = 0;
    topo.n_faces = 0;
    topo.dim = 2;

    topo.cell2vertex_offsets = {0, 3};
    topo.cell2vertex_data = {0, 1, 2};
    topo.vertex_gids = {0, 1, 2};
    topo.cell_gids = {0};
    topo.cell_owner_ranks = {0};

    FE::systems::SetupInputs inputs;
    inputs.topology_override = std::move(topo);
    return inputs;
}

class SingleTetraBoundaryMeshAccess final : public FE::assembly::IMeshAccess {
public:
    explicit SingleTetraBoundaryMeshAccess(int marker)
        : marker_(marker)
    {
        reference_nodes_ = {
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
        };
        current_nodes_ = reference_nodes_;
        cell_ = {0, 1, 2, 3};
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numVertices() const override { return 4; }
    [[nodiscard]] FE::GlobalIndex numOwnedVertices() const override { return 4; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override { return geometry_revision_; }
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
        return current_nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(FE::GlobalIndex /*cell_id*/,
                            std::vector<std::array<FE::Real, 3>>& coords) const override
    {
        coords = current_nodes_;
    }

    [[nodiscard]] bool supportsCoordinateFrame(FE::assembly::CoordinateFrame frame) const override
    {
        return frame == FE::assembly::CoordinateFrame::Active ||
               frame == FE::assembly::CoordinateFrame::Reference ||
               frame == FE::assembly::CoordinateFrame::Current;
    }

    void getCellCoordinates(FE::GlobalIndex /*cell_id*/,
                            FE::assembly::CoordinateFrame frame,
                            std::vector<std::array<FE::Real, 3>>& coords) const override
    {
        switch (frame) {
            case FE::assembly::CoordinateFrame::Active:
            case FE::assembly::CoordinateFrame::Current:
                coords = current_nodes_;
                return;
            case FE::assembly::CoordinateFrame::Reference:
                coords = reference_nodes_;
                return;
        }
    }

    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(FE::GlobalIndex /*face_id*/,
                                                   FE::GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(FE::GlobalIndex /*face_id*/) const override
    {
        return marker_;
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

    void forEachBoundaryFace(int marker,
                             std::function<void(FE::GlobalIndex, FE::GlobalIndex)> callback) const override
    {
        if (marker < 0 || marker == marker_) {
            callback(0, 0);
        }
    }

    void forEachInteriorFace(std::function<void(FE::GlobalIndex, FE::GlobalIndex, FE::GlobalIndex)> /*callback*/) const override
    {
    }

    [[nodiscard]] const std::array<FE::Real, 3>& referenceNodeCoordinates(
        FE::GlobalIndex node_id) const
    {
        return reference_nodes_.at(static_cast<std::size_t>(node_id));
    }

    void setCurrentNodeCoordinates(FE::GlobalIndex node_id,
                                   std::array<FE::Real, 3> coords)
    {
        current_nodes_.at(static_cast<std::size_t>(node_id)) = coords;
        ++geometry_revision_;
    }

private:
    int marker_{-1};
    std::uint64_t geometry_revision_{1};
    std::vector<std::array<FE::Real, 3>> reference_nodes_{};
    std::vector<std::array<FE::Real, 3>> current_nodes_{};
    std::array<FE::GlobalIndex, 4> cell_{};
};

std::shared_ptr<FE::spaces::FunctionSpace> makeVelocitySpace(
    const std::shared_ptr<const FE::assembly::IMeshAccess>& mesh)
{
    return FE::spaces::VectorSpace(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/3);
}

std::shared_ptr<FE::spaces::FunctionSpace> makePressureSpace(
    const std::shared_ptr<const FE::assembly::IMeshAccess>& mesh)
{
    return FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/1);
}

ns::IncompressibleNavierStokesVMSOptions baseNavierStokesOptions()
{
    ns::IncompressibleNavierStokesVMSOptions opts;
    opts.velocity_field_name = "u";
    opts.pressure_field_name = "p";
    opts.density = 1.0;
    opts.viscosity = 0.01;
    opts.enable_convection = true;
    opts.enable_vms = false;
    return opts;
}

FormExpr manufacturedScalarField()
{
    using namespace FE::forms;
    const auto x0 = component(currentCoordinate(), 0);
    const auto x1 = component(currentCoordinate(), 1);
    return x0 * x0 + FormExpr::constant(0.5) * x1 + t();
}

FormExpr constantVector3(FE::Real x, FE::Real y, FE::Real z)
{
    return FormExpr::asVector({
        FormExpr::constant(x),
        FormExpr::constant(y),
        FormExpr::constant(z),
    });
}

FormExpr movingBoundaryKinematicResidual(const FormExpr& physical_velocity,
                                         const FormExpr& test_scalar)
{
    using namespace FE::forms;

    return test_scalar * dot(physical_velocity - meshVelocity(), currentNormal()) *
           currentMeasure();
}

FormExpr fsiDisplacementCompatibilityResidual(const FormExpr& structural_displacement,
                                              const FormExpr& test_scalar)
{
    using namespace FE::forms;

    return test_scalar * dot(structural_displacement - meshDisplacement(), currentNormal()) *
           currentMeasure();
}

FormExpr fsiSurfaceTractionPowerResidual(const FormExpr& current_traction,
                                         const FormExpr& interface_velocity_test)
{
    using namespace FE::forms;

    return inner(current_traction, interface_velocity_test) * currentMeasure();
}

FormExpr referenceSurfaceMeasureMismatchProbe()
{
    using namespace FE::forms;

    return currentMeasure() - referenceMeasure() +
           dot(currentNormal() - referenceNormal(),
               currentNormal() - referenceNormal());
}

FE::dofs::DofMap createSingleTetraDenseDofMap(FE::LocalIndex n_dofs)
{
    FE::dofs::DofMap dof_map(1, n_dofs, n_dofs);
    std::vector<FE::GlobalIndex> cell_dofs(static_cast<std::size_t>(n_dofs));
    for (FE::LocalIndex i = 0; i < n_dofs; ++i) {
        cell_dofs[static_cast<std::size_t>(i)] = i;
    }
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(n_dofs);
    dof_map.setNumLocalDofs(n_dofs);
    dof_map.finalize();
    return dof_map;
}

std::vector<FE::Real> constantScalarTetraCoefficients(FE::Real value)
{
    return std::vector<FE::Real>(4u, value);
}

std::vector<FE::Real> constantVectorTetraCoefficients(FE::Real x,
                                                      FE::Real y,
                                                      FE::Real z)
{
    std::vector<FE::Real> coeffs(12u, 0.0);
    for (std::size_t node = 0; node < 4u; ++node) {
        coeffs[node] = x;
        coeffs[4u + node] = y;
        coeffs[8u + node] = z;
    }
    return coeffs;
}

std::vector<FE::Real> affineXVectorTetraCoefficients()
{
    // ProductSpace coefficients are component-major.  The unit tetra nodal
    // coordinates are x={0,1,0,0}, y={0,0,1,0}, z={0,0,0,1}.
    std::vector<FE::Real> coeffs(12u, 0.0);
    coeffs[0] = 0.0;
    coeffs[1] = 1.0;
    coeffs[2] = 0.0;
    coeffs[3] = 0.0;
    return coeffs;
}

FE::Real residualNorm(FE::systems::FESystem& system,
                      const FE::systems::SystemStateView& state,
                      std::string_view op)
{
    const auto n = system.dofHandler().getNumDofs();
    FE::assembly::DenseVectorView residual(n);
    residual.zero();
    FE::systems::AssemblyRequest req;
    req.op = std::string(op);
    req.want_vector = true;
    const auto result = system.assemble(req, state, nullptr, &residual);
    EXPECT_TRUE(result.success) << result.error_message;

    FE::Real norm2 = 0.0;
    for (FE::GlobalIndex i = 0; i < n; ++i) {
        norm2 += residual[i] * residual[i];
    }
    return std::sqrt(norm2);
}

std::vector<FE::Real> residualVector(FE::systems::FESystem& system,
                                     const FE::systems::SystemStateView& state,
                                     std::string_view op)
{
    const auto n = system.dofHandler().getNumDofs();
    FE::assembly::DenseVectorView residual(n);
    residual.zero();
    FE::systems::AssemblyRequest req;
    req.op = std::string(op);
    req.want_vector = true;
    const auto result = system.assemble(req, state, nullptr, &residual);
    EXPECT_TRUE(result.success) << result.error_message;

    std::vector<FE::Real> out(static_cast<std::size_t>(n), 0.0);
    for (FE::GlobalIndex i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] = residual[i];
    }
    return out;
}

FE::Real vectorNorm(std::span<const FE::Real> values)
{
    FE::Real norm2 = 0.0;
    for (const auto value : values) {
        norm2 += value * value;
    }
    return std::sqrt(norm2);
}

std::vector<FE::Real> fittedFreeSurfaceResidualVector(FE::Real external_pressure,
                                                      FE::Real surface_tension,
                                                      FE::Real curvature)
{
    constexpr int marker = 32;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .external_pressure = external_pressure,
        .surface_tension = surface_tension,
        .curvature = curvature,
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);
    system.setup({}, makeSingleTetraSetupInputs());

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()),
        0.0);
    const std::vector<FE::Real> previous_solution = solution;
    FE::systems::SystemStateView state;
    state.dt = 1.0;
    state.u = std::span<const FE::Real>(solution);
    state.u_prev = std::span<const FE::Real>(previous_solution);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context = integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;
    return residualVector(system, state, "equations");
}

FE::assembly::DenseVectorView assembleMovingDomainScalarResidual(
    const FE::assembly::IMeshAccess& mesh,
    const FE::spaces::FunctionSpace& scalar_space,
    FE::dofs::DofMap& scalar_dof_map,
    const FE::spaces::FunctionSpace* mesh_velocity_space,
    const FE::dofs::DofMap* mesh_velocity_dof_map,
    const FormExpr& residual_integrand,
    const std::vector<FE::Real>& current_solution,
    std::span<const FE::Real> prescribed_mesh_velocity = {})
{
    using namespace FE::forms;

    FE::forms::FormCompiler compiler;
    const auto form = residual_integrand.dx();
    auto ir = compiler.compileResidual(form);
    FE::forms::NonlinearFormKernel kernel(std::move(ir), FE::forms::ADMode::Forward);

    FE::assembly::StandardAssembler assembler;
    assembler.setDofMap(scalar_dof_map);
    if (mesh_velocity_space != nullptr && mesh_velocity_dof_map != nullptr) {
        const std::array<FE::assembly::FieldSolutionAccess, 1> field_access = {{
            FE::assembly::FieldSolutionAccess{
                .field = kMeshVelocityField,
                .space = mesh_velocity_space,
                .dof_map = mesh_velocity_dof_map,
                .dof_offset = 0,
                .coefficient_source =
                    FE::assembly::FieldSolutionAccess::CoefficientSource::PrescribedData,
                .prescribed_coefficients = prescribed_mesh_velocity,
                .prescribed_revision = 1,
            },
        }};
        assembler.setFieldSolutionAccess(field_access);
        assembler.setMeshMotionFieldAccess(FE::assembly::MeshMotionFieldAccess{
            .mesh_velocity = kMeshVelocityField,
        });
    }
    assembler.setCurrentSolution(current_solution);

    FE::assembly::DenseVectorView residual(static_cast<FE::GlobalIndex>(scalar_dof_map.getNumDofs()));
    residual.zero();
    (void)assembler.assembleVector(mesh, scalar_space, kernel, residual);
    return residual;
}

FE::Real fieldComponentValue(const std::vector<FE::Real>& solution,
                             const FE::systems::FESystem& system,
                             FE::FieldId field,
                             FE::GlobalIndex vertex,
                             int component)
{
    const auto& handler = system.fieldDofHandler(field);
    const auto offset = system.fieldDofOffset(field);
    const auto* entity_map = handler.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error("fieldComponentValue: field has no entity DOF map");
    }
    const auto dofs = entity_map->getVertexDofs(vertex);
    if (component < 0 || static_cast<std::size_t>(component) >= dofs.size()) {
        throw std::runtime_error("fieldComponentValue: component is out of range");
    }
    const auto index = static_cast<std::size_t>(
        dofs[static_cast<std::size_t>(component)] + offset);
    if (index >= solution.size()) {
        throw std::runtime_error("fieldComponentValue: DOF index is out of range");
    }
    return solution[index];
}

void setFieldComponentValue(std::vector<FE::Real>& solution,
                            const FE::systems::FESystem& system,
                            FE::FieldId field,
                            FE::GlobalIndex vertex,
                            int component,
                            FE::Real value)
{
    const auto& handler = system.fieldDofHandler(field);
    const auto offset = system.fieldDofOffset(field);
    const auto* entity_map = handler.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error("setFieldComponentValue: field has no entity DOF map");
    }
    const auto dofs = entity_map->getVertexDofs(vertex);
    if (component < 0 || static_cast<std::size_t>(component) >= dofs.size()) {
        throw std::runtime_error("setFieldComponentValue: component is out of range");
    }
    const auto index = static_cast<std::size_t>(
        dofs[static_cast<std::size_t>(component)] + offset);
    if (index >= solution.size()) {
        throw std::runtime_error("setFieldComponentValue: DOF index is out of range");
    }
    solution[index] = value;
}

void updateBoundaryMeshCurrentCoordinates(SingleTetraBoundaryMeshAccess& mesh,
                                          const FE::systems::FESystem& system,
                                          FE::FieldId displacement,
                                          const std::vector<FE::Real>& solution)
{
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        auto coords = mesh.referenceNodeCoordinates(vertex);
        for (int component = 0; component < 3; ++component) {
            coords[static_cast<std::size_t>(component)] +=
                fieldComponentValue(solution, system, displacement, vertex, component);
        }
        mesh.setCurrentNodeCoordinates(vertex, coords);
    }
}

std::vector<FE::Real> assembleOperatorResidualWithCurrentMesh(
    FE::systems::FESystem& system,
    SingleTetraBoundaryMeshAccess& mesh,
    FE::FieldId displacement,
    const std::vector<FE::Real>& solution,
    std::string_view op)
{
    updateBoundaryMeshCurrentCoordinates(mesh, system, displacement, solution);

    FE::systems::SystemStateView state;
    state.u = std::span<const FE::Real>(solution);

    const auto n = system.dofHandler().getNumDofs();
    FE::assembly::DenseVectorView residual(n);
    residual.zero();

    FE::systems::AssemblyRequest req;
    req.op = std::string(op);
    req.want_vector = true;
    const auto result = system.assemble(req, state, nullptr, &residual);
    EXPECT_TRUE(result.success) << result.error_message;

    std::vector<FE::Real> out(static_cast<std::size_t>(n), 0.0);
    for (FE::GlobalIndex i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] = residual.getVectorEntry(i);
    }
    return out;
}

void expectOperatorJacobianMatchesMovingBoundaryFD(
    FE::systems::FESystem& system,
    SingleTetraBoundaryMeshAccess& mesh,
    FE::FieldId displacement,
    const std::vector<FE::Real>& base_solution,
    std::string_view op,
    FE::Real eps,
    FE::Real rtol,
    FE::Real atol)
{
    const auto n = system.dofHandler().getNumDofs();
    ASSERT_EQ(static_cast<FE::GlobalIndex>(base_solution.size()), n);

    updateBoundaryMeshCurrentCoordinates(mesh, system, displacement, base_solution);
    FE::systems::SystemStateView state;
    state.u = std::span<const FE::Real>(base_solution);

    FE::assembly::DenseMatrixView jacobian(n);
    jacobian.zero();
    {
        FE::systems::AssemblyRequest req;
        req.op = std::string(op);
        req.want_matrix = true;
        const auto result = system.assemble(req, state, &jacobian, nullptr);
        ASSERT_TRUE(result.success) << result.error_message;
    }

    for (FE::GlobalIndex col = 0; col < n; ++col) {
        std::vector<FE::Real> plus = base_solution;
        std::vector<FE::Real> minus = base_solution;
        plus[static_cast<std::size_t>(col)] += eps;
        minus[static_cast<std::size_t>(col)] -= eps;

        const auto r_plus =
            assembleOperatorResidualWithCurrentMesh(system, mesh, displacement, plus, op);
        const auto r_minus =
            assembleOperatorResidualWithCurrentMesh(system, mesh, displacement, minus, op);

        for (FE::GlobalIndex row = 0; row < n; ++row) {
            const FE::Real fd =
                (r_plus[static_cast<std::size_t>(row)] -
                 r_minus[static_cast<std::size_t>(row)]) /
                (FE::Real(2.0) * eps);
            const FE::Real actual = jacobian.getMatrixEntry(row, col);
            const FE::Real tol = atol + rtol * std::max<FE::Real>(1.0, std::abs(fd));
            SCOPED_TRACE(::testing::Message() << "row=" << row << ", col=" << col);
            EXPECT_NEAR(actual, fd, tol);
        }
    }
}

} // namespace

TEST(MovingDomainPhysics, NavierStokesALEDisabledDoesNotConsumeMovingDomainTerminals)
{
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_FALSE(system.hasField(opts.mesh_velocity_field_name));
    EXPECT_FALSE(system.meshMotionField(FE::systems::MeshMotionFieldRole::Velocity).has_value());
    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::MeshVelocity));
    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::CurrentMeasure));
    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::CurrentNormal));
}

TEST(MovingDomainPhysics, MovingMeshTangentPathDefaultsToSymbolicRequired)
{
    EXPECT_EQ(mm::HarmonicMeshMotionOptions{}.tangent_path,
              FE::forms::GeometryTangentPath::SymbolicRequired);
    EXPECT_EQ(mm::PseudoElasticMeshMotionOptions{}.tangent_path,
              FE::forms::GeometryTangentPath::SymbolicRequired);
    EXPECT_EQ(ns::IncompressibleNavierStokesVMSOptions{}.moving_mesh_tangent_path,
              FE::forms::GeometryTangentPath::SymbolicRequired);
}

TEST(MovingDomainPhysics, UnfittedFreeSurfaceActiveDomainDefaultsToInactiveCutVolume)
{
    const ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary free_surface{};
    EXPECT_EQ(free_surface.active_domain, ns::FreeSurfaceActiveDomain::None);
    EXPECT_EQ(free_surface.active_domain_method,
              ns::FreeSurfaceActiveDomainMethod::CutVolume);
}

TEST(MovingDomainPhysics, MeshMotionRegistryTranslatesHarmonicSmoothingEquation)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    auto mesh = makeRegistryQuadMesh();

    EquationModuleInput input{};
    input.equation_type = "mesh_motion";
    input.mesh_name = "quad";
    input.mesh = mesh->local_mesh_ptr();
    input.equation_params["Model"] = ParameterValue{true, "Harmonic"};
    input.equation_params["Field_name"] = ParameterValue{true, "mesh_displacement"};
    input.equation_params["Operator_tag"] = ParameterValue{true, "equations"};
    input.equation_params["Kappa"] = ParameterValue{true, "2.5"};
    input.equation_params["Moving_mesh_tangent_path"] =
        ParameterValue{true, "symbolic"};

    BoundaryConditionInput wall{};
    wall.name = "wall";
    wall.boundary_marker = 4;
    wall.params["Type"] = ParameterValue{true, "Dirichlet"};
    wall.params["Value"] = ParameterValue{true, "0.0"};
    input.boundary_conditions.push_back(std::move(wall));

    FE::systems::FESystem system(mesh);
    system.addOperator("equations");
    auto module = EquationModuleRegistry::instance().create("mesh_motion", input, system);

    ASSERT_TRUE(module);
    const auto displacement = system.findFieldByName("mesh_displacement");
    ASSERT_NE(displacement, FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.meshMotionField(FE::systems::MeshMotionFieldRole::Displacement),
              displacement);
    EXPECT_TRUE(system.hasOperator("equations"));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CellIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Gradient));
    ASSERT_NO_THROW(system.setup());
#endif
}

TEST(MovingDomainPhysics, MeshMotionRegistryTranslatesPseudoElasticSmoothingEquation)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    auto mesh = makeRegistryQuadMesh();

    EquationModuleInput input{};
    input.equation_type = "pseudo_elastic_mesh_motion";
    input.mesh_name = "quad";
    input.mesh = mesh->local_mesh_ptr();
    input.equation_params["Field_name"] = ParameterValue{true, "mesh_displacement"};
    input.equation_params["Operator_tag"] = ParameterValue{true, "equations"};
    input.equation_params["Lambda_mesh"] = ParameterValue{true, "3.0"};
    input.equation_params["Mu_mesh"] = ParameterValue{true, "1.5"};

    FE::systems::FESystem system(mesh);
    auto module = EquationModuleRegistry::instance().create(
        "pseudo_elastic_mesh_motion", input, system);

    ASSERT_TRUE(module);
    ASSERT_NE(system.findFieldByName("mesh_displacement"), FE::INVALID_FIELD_ID);
    EXPECT_TRUE(system.hasOperator("equations"));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::SymmetricPart));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Trace));
    ASSERT_NO_THROW(system.setup());
#endif
}

TEST(MovingDomainPhysics, NavierStokesALEEnabledRegistersMeshVelocityAndConsumesMeshVelocity)
{
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_ale = true;
    opts.mesh_velocity_field_name = "mesh_velocity";

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    const FE::FieldId mesh_velocity_id = system.findFieldByName("mesh_velocity");
    ASSERT_NE(mesh_velocity_id, FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.meshMotionField(FE::systems::MeshMotionFieldRole::Velocity), mesh_velocity_id);
    EXPECT_EQ(system.fieldRecord(mesh_velocity_id).source_kind,
              FE::systems::FieldSourceKind::PrescribedData);
    EXPECT_FALSE(system.fieldParticipatesInUnknownVector(mesh_velocity_id));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::MeshVelocity));

    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
    EXPECT_EQ(system.dofHandler().getNumDofs(), 16);
    EXPECT_EQ(system.fieldMap().numFields(), 2u);
    ASSERT_NE(system.blockMap(), nullptr);
    EXPECT_EQ(system.blockMap()->numBlocks(), 2u);
}

TEST(MovingDomainPhysics, NavierStokesFittedFreeSurfaceAddsBoundaryResidual)
{
    constexpr int marker = 31;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .external_pressure = 2.0,
        .surface_tension = 0.5,
        .curvature = 1.25,
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::BoundaryIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Normal));

    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
}

TEST(MovingDomainPhysics, ExternalPressureFreeSurfaceTractionHasExpectedSign)
{
    const auto external_pressure =
        fittedFreeSurfaceResidualVector(/*external_pressure=*/1.0,
                                        /*surface_tension=*/0.0,
                                        /*curvature=*/0.0);
    const auto surface_tension =
        fittedFreeSurfaceResidualVector(/*external_pressure=*/0.0,
                                        /*surface_tension=*/1.0,
                                        /*curvature=*/1.0);

    ASSERT_EQ(external_pressure.size(), surface_tension.size());
    EXPECT_GT(vectorNorm(external_pressure), 1.0e-14);
    for (std::size_t i = 0; i < external_pressure.size(); ++i) {
        EXPECT_NEAR(external_pressure[i], -surface_tension[i], 1.0e-12);
    }
}

TEST(MovingDomainPhysics, SurfaceTensionPressureJumpMatchesLaplaceLaw)
{
    constexpr FE::Real gamma = 0.072;

    constexpr FE::Real circle_radius = 0.45;
    const auto circle_jump =
        fittedFreeSurfaceResidualVector(/*external_pressure=*/gamma / circle_radius,
                                        /*surface_tension=*/gamma,
                                        /*curvature=*/1.0 / circle_radius);
    EXPECT_LT(vectorNorm(circle_jump), 1.0e-12);

    constexpr FE::Real sphere_radius = 0.60;
    const auto sphere_jump =
        fittedFreeSurfaceResidualVector(/*external_pressure=*/2.0 * gamma / sphere_radius,
                                        /*surface_tension=*/gamma,
                                        /*curvature=*/2.0 / sphere_radius);
    EXPECT_LT(vectorNorm(sphere_jump), 1.0e-12);
}

TEST(MovingDomainPhysics, StaticFlatWaterSurfaceWithGravityRemainsAtRest)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    constexpr int left_marker = 101;
    constexpr int right_marker = 102;
    constexpr int bottom_marker = 103;
    constexpr int free_surface_marker = 104;
    constexpr FE::GlobalIndex top_middle_vertex = 7;
    constexpr FE::Real density = 2.0;
    constexpr FE::Real gravity_y = -9.81;
    constexpr FE::Real surface_y = 1.0;
    constexpr FE::Real atmospheric_pressure = 0.0;

    auto mesh = makeOpenTankQuadMesh(left_marker,
                                     right_marker,
                                     bottom_marker,
                                     free_surface_marker,
                                     "free_surface");
    auto scalar_space = std::make_shared<FE::spaces::H1Space>(
        FE::ElementType::Quad4,
        /*order=*/1);
    auto u_space = std::make_shared<FE::spaces::ProductSpace>(
        scalar_space,
        /*components=*/2);
    auto p_space = scalar_space;

    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;
    opts.density = density;
    opts.viscosity = 1.0e-3;
    opts.body_force = {0.0, gravity_y, 0.0};
    opts.velocity_dirichlet = {
        ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
            .boundary_marker = left_marker,
        },
        ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
            .boundary_marker = right_marker,
        },
        ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
            .boundary_marker = bottom_marker,
        },
    };
    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = free_surface_marker,
        .external_pressure = atmospheric_pressure,
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);
    ASSERT_NO_THROW(system.setup());

    const auto u = system.findFieldByName(opts.velocity_field_name);
    const auto p = system.findFieldByName(opts.pressure_field_name);
    ASSERT_NE(u, FE::INVALID_FIELD_ID);
    ASSERT_NE(p, FE::INVALID_FIELD_ID);

    const auto* velocity_entity_map = system.fieldDofHandler(u).getEntityDofMap();
    ASSERT_NE(velocity_entity_map, nullptr);
    const auto top_middle_velocity_dofs =
        velocity_entity_map->getVertexDofs(top_middle_vertex);
    ASSERT_EQ(top_middle_velocity_dofs.size(), 2u);
    const auto velocity_offset = system.fieldDofOffset(u);
    for (const auto dof : top_middle_velocity_dofs) {
        EXPECT_FALSE(system.constraints().isConstrained(velocity_offset + dof));
    }

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()),
        0.0);
    for (FE::GlobalIndex vertex = 0;
         vertex < static_cast<FE::GlobalIndex>(mesh->n_vertices());
         ++vertex) {
        const auto x = system.meshAccess().getNodeCoordinates(vertex);
        const auto pressure =
            atmospheric_pressure +
            density * gravity_y * (x[1] - surface_y);
        setFieldComponentValue(solution, system, p, vertex, 0, pressure);
    }
    const auto previous_solution = solution;

    FE::systems::SystemStateView state;
    state.dt = 1.0;
    state.u = std::span<const FE::Real>(solution);
    state.u_prev = std::span<const FE::Real>(previous_solution);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context =
        integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;

    const auto residual = residualVector(system, state, "equations");
    EXPECT_LT(vectorNorm(residual), 1.0e-10);
#endif
}

TEST(MovingDomainPhysics, FittedAndUnfittedFlatStaticFreeSurfaceAgree)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    constexpr int left_marker = 111;
    constexpr int right_marker = 112;
    constexpr int bottom_marker = 113;
    constexpr int free_surface_marker = 114;
    constexpr int interface_marker = 115;
    constexpr FE::GlobalIndex top_middle_vertex = 7;
    constexpr FE::Real density = 2.0;
    constexpr FE::Real gravity_y = -9.81;
    constexpr FE::Real bottom_y = -1.0;
    constexpr FE::Real interface_y = -0.5;
    constexpr FE::Real fitted_middle_y = -0.75;
    constexpr FE::Real external_pressure = 1.25;
    constexpr FE::Real expected_surface_length = 2.0;

    auto scalar_space = std::make_shared<FE::spaces::H1Space>(
        FE::ElementType::Quad4,
        /*order=*/1);
    auto u_space = std::make_shared<FE::spaces::ProductSpace>(
        scalar_space,
        /*components=*/2);

    auto fitted_mesh = makeOpenTankQuadMesh(left_marker,
                                            right_marker,
                                            bottom_marker,
                                            free_surface_marker,
                                            "free_surface",
                                            bottom_y,
                                            fitted_middle_y,
                                            interface_y);
    auto fitted_opts = baseNavierStokesOptions();
    fitted_opts.enable_convection = false;
    fitted_opts.density = density;
    fitted_opts.viscosity = 1.0e-3;
    fitted_opts.body_force = {0.0, gravity_y, 0.0};
    fitted_opts.velocity_dirichlet = {
        ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
            .boundary_marker = left_marker,
        },
        ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
            .boundary_marker = right_marker,
        },
        ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
            .boundary_marker = bottom_marker,
        },
    };
    fitted_opts.free_surface.push_back(
        ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
            .implementation = ns::FreeSurfaceImplementation::FittedALE,
            .boundary_marker = free_surface_marker,
            .external_pressure = external_pressure,
        });

    FE::systems::FESystem fitted_system(fitted_mesh);
    ns::IncompressibleNavierStokesVMSModule fitted_module(
        u_space,
        scalar_space,
        fitted_opts);
    fitted_module.registerOn(fitted_system);
    ASSERT_NO_THROW(fitted_system.setup());

    const auto fitted_u =
        fitted_system.findFieldByName(fitted_opts.velocity_field_name);
    const auto fitted_p =
        fitted_system.findFieldByName(fitted_opts.pressure_field_name);
    ASSERT_NE(fitted_u, FE::INVALID_FIELD_ID);
    ASSERT_NE(fitted_p, FE::INVALID_FIELD_ID);
    const auto* fitted_velocity_entity_map =
        fitted_system.fieldDofHandler(fitted_u).getEntityDofMap();
    ASSERT_NE(fitted_velocity_entity_map, nullptr);
    const auto top_middle_velocity_dofs =
        fitted_velocity_entity_map->getVertexDofs(top_middle_vertex);
    ASSERT_EQ(top_middle_velocity_dofs.size(), 2u);
    const auto fitted_velocity_offset = fitted_system.fieldDofOffset(fitted_u);
    for (const auto dof : top_middle_velocity_dofs) {
        EXPECT_FALSE(
            fitted_system.constraints().isConstrained(fitted_velocity_offset + dof));
    }

    std::vector<FE::Real> fitted_solution(
        static_cast<std::size_t>(fitted_system.dofHandler().getNumDofs()),
        0.0);
    for (FE::GlobalIndex vertex = 0;
         vertex < static_cast<FE::GlobalIndex>(fitted_mesh->n_vertices());
         ++vertex) {
        const auto x = fitted_system.meshAccess().getNodeCoordinates(vertex);
        const auto pressure =
            external_pressure +
            density * gravity_y * (x[1] - interface_y);
        setFieldComponentValue(
            fitted_solution,
            fitted_system,
            fitted_p,
            vertex,
            0,
            pressure);
    }
    const auto fitted_previous_solution = fitted_solution;
    FE::systems::SystemStateView fitted_state;
    fitted_state.dt = 1.0;
    fitted_state.u = std::span<const FE::Real>(fitted_solution);
    fitted_state.u_prev = std::span<const FE::Real>(fitted_previous_solution);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto fitted_time_context =
        integrator.buildContext(/*max_time_derivative_order=*/1, fitted_state);
    fitted_state.time_integration = &fitted_time_context;
    const auto fitted_residual =
        residualVector(fitted_system, fitted_state, "equations");
    EXPECT_LT(vectorNorm(fitted_residual), 1.0e-10);

    auto background_mesh = makeOpenTankQuadMesh(left_marker,
                                                right_marker,
                                                bottom_marker,
                                                free_surface_marker,
                                                "outer_free_surface");
    FE::systems::FESystem unfitted_system(background_mesh);
    const auto phi = unfitted_system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(unfitted_system.setup());

    std::vector<FE::Real> unfitted_solution(
        static_cast<std::size_t>(unfitted_system.dofHandler().getNumDofs()),
        0.0);
    for (FE::GlobalIndex vertex = 0;
         vertex < static_cast<FE::GlobalIndex>(background_mesh->n_vertices());
         ++vertex) {
        const auto x = unfitted_system.meshAccess().getNodeCoordinates(vertex);
        setFieldComponentValue(
            unfitted_solution,
            unfitted_system,
            phi,
            vertex,
            0,
            x[1] - interface_y);
    }

    ls::LevelSetGeneratedInterfaceOptions interface_options{};
    interface_options.level_set_field_name = "phi";
    interface_options.domain_id = "flat_static_surface";
    interface_options.requested_interface_marker = interface_marker;
    interface_options.tolerance = 1.0e-12;

    ls::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto generated =
        lifecycle.build(unfitted_system, interface_options, unfitted_solution);
    ASSERT_TRUE(generated.success) << generated.diagnostic;
    EXPECT_EQ(generated.interface_marker, interface_marker);
    EXPECT_EQ(generated.summary.active_fragment_count, 2u);
    EXPECT_NEAR(generated.summary.measure, expected_surface_length, 1.0e-12);
    EXPECT_NEAR(external_pressure * generated.summary.measure,
                external_pressure * expected_surface_length,
                1.0e-12);

    for (const auto& fragment : generated.domain.fragments()) {
        if (!fragment.active()) {
            continue;
        }
        EXPECT_NEAR(fragment.measure, 1.0, 1.0e-12);
        EXPECT_NEAR(fragment.normal[0], 0.0, 1.0e-12);
        EXPECT_NEAR(fragment.normal[1], 1.0, 1.0e-12);
        EXPECT_NEAR(fragment.normal[2], 0.0, 1.0e-12);
    }

    auto unfitted_opts = baseNavierStokesOptions();
    unfitted_opts.enable_convection = false;
    unfitted_opts.density = density;
    unfitted_opts.viscosity = 1.0e-3;
    unfitted_opts.body_force = {0.0, gravity_y, 0.0};
    unfitted_opts.free_surface.push_back(
        ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
            .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
            .interface_marker = interface_marker,
            .level_set_field_name = "phi",
            .generated_interface_domain_id = "flat_static_surface",
            .external_pressure = external_pressure,
        });
    ns::IncompressibleNavierStokesVMSModule unfitted_module(
        u_space,
        scalar_space,
        unfitted_opts);
    unfitted_module.registerOn(unfitted_system);
    EXPECT_TRUE(formulationRecordsContain(unfitted_system,
                                          FormExprType::InterfaceIntegral));
    EXPECT_TRUE(formulationRecordsContain(unfitted_system, FormExprType::Gradient));
    EXPECT_TRUE(formulationRecordsContainInterfaceMarker(unfitted_system,
                                                        interface_marker));
#endif
}

TEST(MovingDomainPhysics, NavierStokesFittedFreeSurfaceALEUsesCurrentBoundaryGeometry)
{
    constexpr int marker = 34;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_ale = true;
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .external_pressure = 2.0,
        .surface_tension = 0.5,
        .curvature = 1.25,
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CurrentNormal));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CurrentMeasure));

    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
}

TEST(MovingDomainPhysics, CurrentFaceGeometryMeanCurvatureTracksCurvedHexFace)
{
    auto basis = std::make_shared<FE::basis::LagrangeBasis>(FE::ElementType::Hex27, 2);
    std::vector<FE::math::Vector<FE::Real, 3>> nodes;
    nodes.reserve(basis->nodes().size());

    constexpr FE::Real radius = 2.0;
    for (const auto& xi : basis->nodes()) {
        const FE::Real x = xi[0];
        const FE::Real y = xi[1];
        const FE::Real zeta = xi[2];
        const FE::Real top_offset =
            ((FE::Real(1) + zeta) * (x * x + y * y)) / (FE::Real(4) * radius);
        nodes.push_back({x, y, zeta - top_offset});
    }

    FE::geometry::IsoparametricMapping mapping(basis, nodes);
    const auto quad = FE::quadrature::QuadratureFactory::create(FE::ElementType::Quad4, 2);
    const auto face = FE::geometry::evaluateFaceFrame(mapping,
                                                      FE::ElementType::Hex27,
                                                      /*local_face_id=*/1,
                                                      FE::ElementType::Quad4,
                                                      *quad);

    ASSERT_EQ(face.mean_curvatures.size(), quad->num_points());
    for (std::size_t q = 0; q < quad->num_points(); ++q) {
        const auto point = quad->points()[q];
        const FE::Real r2 = point[0] * point[0] + point[1] * point[1];
        const FE::Real grad2 = r2 / (radius * radius);
        const FE::Real expected =
            FE::Real(2) * (FE::Real(2) + grad2) /
            (radius * std::pow(FE::Real(1) + grad2, FE::Real(1.5)));
        EXPECT_NEAR(face.mean_curvatures[q], expected, 1e-4);
    }
}

TEST(MovingDomainPhysics, NavierStokesFittedFreeSurfaceCanUseCurrentGeometryCurvature)
{
    constexpr int marker = 35;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_ale = true;
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .surface_tension = 0.5,
        .use_current_geometry_curvature = true,
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CurrentMeanCurvature));
}

TEST(MovingDomainPhysics, NavierStokesFittedFreeSurfacePenaltyKinematicsAddsBoundaryResidual)
{
    constexpr int marker = 33;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_ale = true;
    opts.enable_convection = false;
    opts.mesh_velocity_field_name = "mesh_velocity";

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .kinematic_enforcement = ns::FreeSurfaceKinematicEnforcement::Penalty,
        .kinematic_penalty = 12.0,
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::BoundaryIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::MeshVelocity));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CurrentNormal));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CurrentMeasure));

    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    const FE::FieldId mesh_velocity_id = system.findFieldByName("mesh_velocity");
    ASSERT_NE(mesh_velocity_id, FE::INVALID_FIELD_ID);
    system.setPrescribedFieldCoefficients(
        mesh_velocity_id,
        constantVectorTetraCoefficients(0.25, 0.5, 0.75));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    std::vector<FE::Real> previous_solution = solution;

    FE::systems::SystemStateView state;
    state.dt = 1.0;
    state.u = std::span<const FE::Real>(solution);
    state.u_prev = std::span<const FE::Real>(previous_solution);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context = integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;

    EXPECT_GT(residualNorm(system, state, "equations"), 0.0);
}

TEST(MovingDomainPhysics, NavierStokesFittedFreeSurfacePenaltyKinematicsRequiresALE)
{
    constexpr int marker = 34;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_ale = false;
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .kinematic_enforcement = ns::FreeSurfaceKinematicEnforcement::Penalty,
        .kinematic_penalty = 12.0,
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
}

TEST(MovingDomainPhysics, NavierStokesFittedFreeSurfacePenaltyKinematicsRequiresPenalty)
{
    constexpr int marker = 35;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_ale = true;
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .kinematic_enforcement = ns::FreeSurfaceKinematicEnforcement::Penalty,
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
}

TEST(MovingDomainPhysics, NavierStokesFittedFreeSurfaceNitscheKinematicsAddsBoundaryResidual)
{
    constexpr int marker = 36;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_ale = true;
    opts.enable_convection = false;
    opts.mesh_velocity_field_name = "mesh_velocity";
    opts.nitsche_gamma = 16.0;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .kinematic_enforcement = ns::FreeSurfaceKinematicEnforcement::Nitsche,
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::BoundaryIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::FacetArea));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::MeshVelocity));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CurrentNormal));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CurrentMeasure));

    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    const FE::FieldId mesh_velocity_id = system.findFieldByName("mesh_velocity");
    ASSERT_NE(mesh_velocity_id, FE::INVALID_FIELD_ID);
    system.setPrescribedFieldCoefficients(
        mesh_velocity_id,
        constantVectorTetraCoefficients(0.25, 0.5, 0.75));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    std::vector<FE::Real> previous_solution = solution;

    FE::systems::SystemStateView state;
    state.dt = 1.0;
    state.u = std::span<const FE::Real>(solution);
    state.u_prev = std::span<const FE::Real>(previous_solution);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context = integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;

    EXPECT_GT(residualNorm(system, state, "equations"), 0.0);
}

TEST(MovingDomainPhysics, NavierStokesFittedFreeSurfaceNitscheKinematicsRejectsNonPositiveGamma)
{
    constexpr int marker = 38;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_ale = true;
    opts.enable_convection = false;
    opts.nitsche_gamma = 0.0;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .kinematic_enforcement = ns::FreeSurfaceKinematicEnforcement::Nitsche,
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
}

TEST(MovingDomainPhysics, FittedFreeSurfaceKinematicPolicyOptionsAreExplicit)
{
    using FreeSurfaceBoundary = ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary;

    const FreeSurfaceBoundary bc{};

    EXPECT_EQ(bc.normal_kinematic_policy,
              ns::FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity);
    EXPECT_EQ(bc.tangential_mesh_policy,
              ns::FreeSurfaceTangentialMeshPolicy::SmoothingOnly);
    EXPECT_EQ(bc.kinematic_enforcement,
              ns::FreeSurfaceKinematicEnforcement::None);
    EXPECT_DOUBLE_EQ(std::get<FE::Real>(bc.prescribed_tangential_mesh_velocity[0]), 0.0);
    EXPECT_DOUBLE_EQ(std::get<FE::Real>(bc.prescribed_tangential_mesh_velocity[1]), 0.0);
    EXPECT_DOUBLE_EQ(std::get<FE::Real>(bc.prescribed_tangential_mesh_velocity[2]), 0.0);
}

TEST(MovingDomainPhysics, FreeSurfaceContactLineOptionsAreExplicit)
{
    using ContactLine = ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine;
    using FreeSurfaceBoundary = ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary;

    const ContactLine contact_line{};
    EXPECT_EQ(contact_line.model, ns::FreeSurfaceContactLineModel::None);
    EXPECT_EQ(contact_line.wall_boundary_marker, -1);
    EXPECT_EQ(contact_line.contact_line_marker, -1);
    EXPECT_DOUBLE_EQ(std::get<FE::Real>(contact_line.contact_angle_radians),
                     1.57079632679489661923);
    EXPECT_DOUBLE_EQ(std::get<FE::Real>(contact_line.wall_normal[0]), 0.0);
    EXPECT_DOUBLE_EQ(std::get<FE::Real>(contact_line.wall_normal[1]), 0.0);
    EXPECT_DOUBLE_EQ(std::get<FE::Real>(contact_line.wall_normal[2]), 0.0);
    EXPECT_DOUBLE_EQ(std::get<FE::Real>(contact_line.contact_angle_penalty), 1.0);
    EXPECT_DOUBLE_EQ(std::get<FE::Real>(contact_line.mobility), 0.0);
    EXPECT_EQ(contact_line.wall_slip_model, ns::FreeSurfaceWallSlipModel::None);
    EXPECT_DOUBLE_EQ(std::get<FE::Real>(contact_line.slip_length), 0.0);

    FreeSurfaceBoundary free_surface{};
    EXPECT_TRUE(free_surface.contact_lines.empty());

    free_surface.contact_lines.push_back(ContactLine{
        .model = ns::FreeSurfaceContactLineModel::PrescribedContactAngle,
        .wall_boundary_marker = 7,
        .contact_line_marker = 8,
        .contact_angle_radians = 0.78539816339744830962,
        .wall_normal = {0.0, 1.0, 0.0},
        .contact_angle_penalty = 12.0,
        .mobility = 0.25,
        .wall_slip_model = ns::FreeSurfaceWallSlipModel::Navier,
        .slip_length = 0.01,
    });

    ASSERT_EQ(free_surface.contact_lines.size(), 1u);
    EXPECT_EQ(free_surface.contact_lines.front().model,
              ns::FreeSurfaceContactLineModel::PrescribedContactAngle);
    EXPECT_EQ(free_surface.contact_lines.front().wall_boundary_marker, 7);
    EXPECT_EQ(free_surface.contact_lines.front().contact_line_marker, 8);
    EXPECT_EQ(free_surface.contact_lines.front().wall_slip_model,
              ns::FreeSurfaceWallSlipModel::Navier);
}

TEST(MovingDomainPhysics, FittedPinnedContactLineConstrainsMeshDisplacement)
{
    constexpr int marker = 40;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);

    FE::systems::FESystem system(mesh);
    const auto displacement = system.addField(FE::systems::FieldSpec{
        .name = "mesh_displacement",
        .space = u_space,
        .components = 3,
    });

    auto opts = baseNavierStokesOptions();
    opts.enable_ale = true;
    opts.enable_convection = false;
    opts.mesh_velocity_source = ns::ALEMeshVelocitySource::CoupledDisplacement;
    opts.mesh_displacement_field_name = "mesh_displacement";
    opts.mesh_velocity_field_name = "mesh_velocity";

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .contact_lines = {
            ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine{
                .model = ns::FreeSurfaceContactLineModel::Pinned,
                .contact_line_marker = marker,
            },
        },
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    const auto offset = system.fieldDofOffset(displacement);
    const auto n_displacement_dofs = system.fieldDofHandler(displacement).getNumDofs();
    std::size_t constrained_displacement_dofs = 0;
    for (FE::GlobalIndex local_dof = 0; local_dof < n_displacement_dofs; ++local_dof) {
        const auto global_dof = offset + local_dof;
        if (!system.constraints().isConstrained(global_dof)) {
            continue;
        }
        ++constrained_displacement_dofs;
        EXPECT_NEAR(system.constraints().getInhomogeneity(global_dof), 0.0, 1.0e-15);
    }
    EXPECT_GT(constrained_displacement_dofs, 0u);
}

TEST(MovingDomainPhysics, FittedPinnedContactLineRequiresALE)
{
    constexpr int marker = 41;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);

    auto opts = baseNavierStokesOptions();
    opts.enable_ale = false;
    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .contact_lines = {
            ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine{
                .model = ns::FreeSurfaceContactLineModel::Pinned,
                .contact_line_marker = marker,
            },
        },
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
}

TEST(MovingDomainPhysics, FittedPrescribedContactAngleAddsMeshMotionResidual)
{
    constexpr int marker = 42;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);

    FE::systems::FESystem system(mesh);
    system.addOperator("mesh_motion");
    (void)system.addField(FE::systems::FieldSpec{
        .name = "mesh_displacement",
        .space = u_space,
        .components = 3,
    });

    auto opts = baseNavierStokesOptions();
    opts.enable_ale = true;
    opts.enable_convection = false;
    opts.mesh_velocity_source = ns::ALEMeshVelocitySource::CoupledDisplacement;
    opts.mesh_displacement_field_name = "mesh_displacement";
    opts.mesh_velocity_field_name = "mesh_velocity";

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .contact_lines = {
            ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine{
                .model = ns::FreeSurfaceContactLineModel::PrescribedContactAngle,
                .contact_angle_radians = 0.0,
                .wall_normal = {1.0, 0.0, 0.0},
                .contact_angle_penalty = 5.0,
            },
        },
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_TRUE(system.hasOperator("mesh_motion"));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::BoundaryIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CurrentNormal));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CurrentMeasure));

    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(system.dofHandler().getNumDofs(), 0.0);
    FE::systems::SystemStateView state;
    state.u = solution;
    EXPECT_GT(residualNorm(system, state, "mesh_motion"), 0.0);
}

TEST(MovingDomainPhysics, FittedPrescribedContactAngleResidualVanishesForMatchedFlatSurface)
{
    constexpr int marker = 46;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);

    FE::systems::FESystem system(mesh);
    system.addOperator("mesh_motion");
    (void)system.addField(FE::systems::FieldSpec{
        .name = "mesh_displacement",
        .space = u_space,
        .components = 3,
    });

    auto opts = baseNavierStokesOptions();
    opts.enable_ale = true;
    opts.enable_convection = false;
    opts.mesh_velocity_source = ns::ALEMeshVelocitySource::CoupledDisplacement;
    opts.mesh_displacement_field_name = "mesh_displacement";
    opts.mesh_velocity_field_name = "mesh_velocity";

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .contact_lines = {
            ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine{
                .model = ns::FreeSurfaceContactLineModel::PrescribedContactAngle,
                .contact_angle_radians = 0.0,
                .wall_normal = {0.0, 0.0, -1.0},
                .contact_angle_penalty = 5.0,
            },
        },
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));

    std::vector<FE::Real> solution(system.dofHandler().getNumDofs(), 0.0);
    FE::systems::SystemStateView state;
    state.u = solution;
    EXPECT_NEAR(residualNorm(system, state, "mesh_motion"), 0.0, 1.0e-12);
}

TEST(MovingDomainPhysics, FittedPrescribedContactAngleRequiresALE)
{
    constexpr int marker = 43;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);

    auto opts = baseNavierStokesOptions();
    opts.enable_ale = false;
    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .contact_lines = {
            ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine{
                .model = ns::FreeSurfaceContactLineModel::PrescribedContactAngle,
                .contact_angle_radians = 0.0,
                .wall_normal = {1.0, 0.0, 0.0},
            },
        },
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
}

TEST(MovingDomainPhysics, NavierStokesFittedFreeSurfaceReservesBoundaryMarker)
{
    constexpr int marker = 32;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();

    opts.velocity_dirichlet.push_back(ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
        .boundary_marker = marker,
        .value = {0.0, 0.0, 0.0},
    });
    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .external_pressure = 1.0,
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
}

TEST(MovingDomainPhysics, NavierStokesUnfittedFreeSurfaceUsesLevelSetInterfaceGeometry)
{
    constexpr int interface_marker = 41;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .external_pressure = 1.0,
        .surface_tension = 0.25,
    });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::InterfaceIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Gradient));
}

TEST(MovingDomainPhysics, NavierStokesUnfittedFreeSurfaceUsesGeneratedInterfaceMarker)
{
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .level_set_field_name = "phi",
        .external_pressure = 1.0,
    });

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    FE::interfaces::GeneratedInterfaceMarkerKey key{};
    key.source = FE::interfaces::LevelSetInterfaceSource::fromField(phi);
    key.domain_id = "free_surface";
    const int expected_marker = FE::interfaces::stableGeneratedInterfaceMarker(key);

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::InterfaceIntegral));
    EXPECT_TRUE(formulationRecordsContainInterfaceMarker(system, expected_marker));
}

TEST(MovingDomainPhysics, NavierStokesUnfittedFreeSurfaceAddsCutCellStabilization)
{
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .level_set_field_name = "phi",
        .external_pressure = 1.0,
        .cut_cell_stabilization = {
            .enabled = true,
            .velocity_gradient_penalty = 2.0,
            .pressure_gradient_penalty = 0.25,
        },
    });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::InteriorFaceIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Jump));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Average));
    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::ParameterRef));
}

TEST(MovingDomainPhysics, NavierStokesUnfittedFreeSurfaceUsesCutMetadataScale)
{
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .level_set_field_name = "phi",
        .external_pressure = 1.0,
        .cut_cell_stabilization = {
            .enabled = true,
            .velocity_gradient_penalty = 2.0,
            .pressure_gradient_penalty = 0.25,
            .use_cut_metadata_scale = true,
        },
    });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::InteriorFaceIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::ParameterRef));
}

TEST(MovingDomainPhysics, NavierStokesUnfittedZeroTractionFreeSurfaceAvoidsInterfaceIntegral)
{
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .level_set_field_name = "phi",
        .external_pressure = 0.0,
        .surface_tension = 0.0,
        .cut_cell_stabilization = {
            .enabled = true,
            .velocity_gradient_penalty = 2.0,
            .pressure_gradient_penalty = 0.25,
        },
    });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::InterfaceIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::InteriorFaceIntegral));
}

TEST(MovingDomainPhysics, NavierStokesActiveDomainZeroTractionFreeSurfaceAvoidsInterfaceIntegral)
{
    constexpr int interface_marker = 46;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .external_pressure = 0.0,
        .surface_tension = 0.0,
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
    });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::InterfaceIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CutVolumeIntegral));
}

TEST(MovingDomainPhysics, NavierStokesInactiveActiveDomainKeepsFullCellVolumeKernels)
{
    constexpr int interface_marker = 47;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_vms = true;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
    });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    const auto& equations = system.operatorDefinition("equations");
    EXPECT_FALSE(equations.cells.empty());
    EXPECT_TRUE(equations.cut_volumes.empty());
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CellIntegral));
    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::CutVolumeIntegral));
}

TEST(MovingDomainPhysics, NavierStokesActiveDomainInstallsCutVolumeKernels)
{
    constexpr int interface_marker = 48;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_vms = true;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
    });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    const auto& equations = system.operatorDefinition("equations");
    EXPECT_TRUE(equations.cells.empty());
    ASSERT_FALSE(equations.cut_volumes.empty());
    for (const auto& term : equations.cut_volumes) {
        EXPECT_EQ(term.marker, interface_marker);
        EXPECT_EQ(term.side, FE::geometry::CutIntegrationSide::Negative);
    }
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CutVolumeIntegral));
    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::CellIntegral));
}

TEST(MovingDomainPhysics, NavierStokesActiveDomainPositiveUsesPositiveCutVolumeSide)
{
    constexpr int interface_marker = 49;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_vms = true;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetPositive,
    });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    const auto& equations = system.operatorDefinition("equations");
    EXPECT_TRUE(equations.cells.empty());
    ASSERT_FALSE(equations.cut_volumes.empty());
    for (const auto& term : equations.cut_volumes) {
        EXPECT_EQ(term.marker, interface_marker);
        EXPECT_EQ(term.side, FE::geometry::CutIntegrationSide::Positive);
    }
}

TEST(MovingDomainPhysics, NavierStokesSmoothedIndicatorActiveDomainUsesWeightedCellKernels)
{
    constexpr int interface_marker = 50;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_vms = true;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .active_domain_method = ns::FreeSurfaceActiveDomainMethod::SmoothedIndicator,
        .active_domain_smoothing_width = 0.125,
    });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    const auto& equations = system.operatorDefinition("equations");
    EXPECT_FALSE(equations.cells.empty());
    EXPECT_TRUE(equations.cut_volumes.empty());
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CellIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::SmoothHeaviside));
    EXPECT_FALSE(formulationRecordsContain(system, FormExprType::CutVolumeIntegral));
}

TEST(MovingDomainPhysics, NavierStokesRejectsCutCellStabilizationOnFittedSurface)
{
    constexpr int marker = 45;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .cut_cell_stabilization = {
            .enabled = true,
        },
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
}

TEST(MovingDomainPhysics, NavierStokesRejectsActiveDomainOnFittedSurface)
{
    constexpr int marker = 46;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
}

TEST(MovingDomainPhysics, NavierStokesUnfittedFreeSurfaceRejectsUnknownLevelSet)
{
    constexpr int interface_marker = 42;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "missing_phi",
    });

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
}

TEST(MovingDomainPhysics, NavierStokesUnfittedPrescribedContactAngleAddsLevelSetResidual)
{
    constexpr int interface_marker = 44;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_convection = false;

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .contact_lines = {
            ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine{
                .model = ns::FreeSurfaceContactLineModel::PrescribedContactAngle,
                .contact_angle_radians = 1.0471975511965977462,
                .wall_normal = {1.0, 0.0, 0.0},
                .contact_angle_penalty = 4.0,
            },
        },
    });

    FE::systems::FESystem system(mesh);
    system.addOperator("level_set");
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_TRUE(system.hasOperator("level_set"));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::InterfaceIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Gradient));
}

TEST(MovingDomainPhysics, NavierStokesUnfittedPrescribedContactAngleRequiresLevelSetUnknown)
{
    constexpr int interface_marker = 45;
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();

    opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = interface_marker,
        .level_set_field_name = "phi",
        .contact_lines = {
            ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine{
                .model = ns::FreeSurfaceContactLineModel::PrescribedContactAngle,
                .contact_angle_radians = 1.0471975511965977462,
                .wall_normal = {1.0, 0.0, 0.0},
            },
        },
    });

    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = p_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    EXPECT_THROW(module.registerOn(system), std::invalid_argument);
}

TEST(MovingDomainPhysics, NavierStokesCoupledALEDerivesMeshVelocityFromDisplacement)
{
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_ale = true;
    opts.mesh_velocity_source = ns::ALEMeshVelocitySource::CoupledDisplacement;
    opts.mesh_displacement_field_name = "mesh_displacement";
    opts.mesh_velocity_field_name = "mesh_velocity";

    FE::systems::FESystem system(mesh);
    const auto displacement =
        system.addField(FE::systems::FieldSpec{.name = "mesh_displacement",
                                               .space = u_space,
                                               .components = 3});

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    const FE::FieldId mesh_velocity_id = system.findFieldByName("mesh_velocity");
    ASSERT_NE(mesh_velocity_id, FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.fieldRecord(displacement).source_kind,
              FE::systems::FieldSourceKind::Unknown);
    EXPECT_EQ(system.fieldRecord(mesh_velocity_id).source_kind,
              FE::systems::FieldSourceKind::DerivedFromUnknown);
    EXPECT_EQ(system.fieldRecord(mesh_velocity_id).derived.source_field, displacement);
    EXPECT_EQ(system.fieldRecord(mesh_velocity_id).derived.role,
              FE::systems::DerivedFieldRole::TimeDerivative);
    EXPECT_FALSE(system.fieldParticipatesInUnknownVector(mesh_velocity_id));
    EXPECT_TRUE(system.geometricNonlinearityPolicy().enabled);
    EXPECT_TRUE(system.geometricNonlinearityPolicy().update_current_coordinates_on_trial);
    EXPECT_EQ(system.meshMotionField(FE::systems::MeshMotionFieldRole::Displacement),
              displacement);
    EXPECT_EQ(system.meshMotionField(FE::systems::MeshMotionFieldRole::Velocity),
              mesh_velocity_id);

    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
    EXPECT_EQ(system.dofHandler().getNumDofs(), 28);
    EXPECT_EQ(system.fieldMap().numFields(), 3u);
    ASSERT_NE(system.blockMap(), nullptr);
    EXPECT_EQ(system.blockMap()->numBlocks(), 3u);

    bool has_fluid_mesh_coupling = false;
    for (const auto& record : system.formulationRecords()) {
        for (const auto& [test_field, trial_field] : record.block_couplings) {
            if (trial_field == displacement &&
                (test_field == system.findFieldByName(opts.velocity_field_name) ||
                 test_field == system.findFieldByName(opts.pressure_field_name))) {
                has_fluid_mesh_coupling = true;
            }
        }
    }
    EXPECT_TRUE(has_fluid_mesh_coupling);
}

TEST(MovingDomainPhysics, NavierStokesCoupledALEAcceptsADReferenceTangentPathOverride)
{
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);
    auto opts = baseNavierStokesOptions();
    opts.enable_ale = true;
    opts.mesh_velocity_source = ns::ALEMeshVelocitySource::CoupledDisplacement;
    opts.mesh_displacement_field_name = "mesh_displacement";
    opts.mesh_velocity_field_name = "mesh_velocity";
    opts.moving_mesh_tangent_path = FE::forms::GeometryTangentPath::ADReference;

    FE::systems::FESystem system(mesh);
    const auto displacement =
        system.addField(FE::systems::FieldSpec{.name = "mesh_displacement",
                                               .space = u_space,
                                               .components = 3});

    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    EXPECT_EQ(system.meshMotionField(FE::systems::MeshMotionFieldRole::Displacement),
              displacement);
    EXPECT_NE(system.findFieldByName("mesh_velocity"), FE::INVALID_FIELD_ID);
    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
    EXPECT_EQ(system.dofHandler().getNumDofs(), 28);
}

TEST(MovingDomainPhysics, MixedFluidMeshBoundaryGeometryResidualMatchesFiniteDifference)
{
    using namespace FE::forms;

    constexpr int marker = 37;
    constexpr std::string_view op = "free_surface_boundary";
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto velocity_space = makeVelocitySpace(mesh);
    auto displacement_space = makeVelocitySpace(mesh);

    FE::systems::FESystem system(mesh);
    const auto velocity = system.addField(FE::systems::FieldSpec{
        .name = "fluid_velocity",
        .space = velocity_space,
        .components = 3,
    });
    const auto displacement = system.addField(FE::systems::FieldSpec{
        .name = "mesh_displacement",
        .space = displacement_space,
        .components = 3,
    });
    system.bindMeshMotionField(FE::systems::MeshMotionFieldRole::Displacement,
                               displacement);
    auto geometry_policy = system.geometricNonlinearityPolicy();
    geometry_policy.enabled = true;
    geometry_policy.update_current_coordinates_on_trial = true;
    system.setGeometricNonlinearityPolicy(geometry_policy);
    system.addOperator(std::string(op));

    const auto u = FormExpr::stateField(velocity, *velocity_space, "u");
    const auto v = FormExpr::testFunction(velocity, *velocity_space, "v");
    const auto normal = currentNormal();
    const auto residual =
        (dot(u, normal) * dot(v, normal) * currentMeasure()).ds(marker);

    FE::systems::FormInstallOptions install;
    install.compiler_options.geometry_sensitivity.mode =
        GeometrySensitivityMode::MeshMotionUnknowns;
    install.compiler_options.geometry_sensitivity.mesh_motion_field = displacement;
    install.compiler_options.geometry_tangent_path = GeometryTangentPath::SymbolicRequired;
    install.extra_trial_fields.push_back(displacement);
    const auto kernels =
        FE::systems::installFormulation(system, std::string(op), {velocity}, residual, install);

    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CurrentNormal));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CurrentMeasure));
    ASSERT_EQ(kernels.jacobian_blocks.size(), 1u);
    ASSERT_EQ(kernels.jacobian_blocks.front().size(), 2u);
    EXPECT_NE(kernels.jacobian_blocks.front()[0], nullptr);
    EXPECT_NE(kernels.jacobian_blocks.front()[1], nullptr);

    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
    ASSERT_EQ(system.dofHandler().getNumDofs(), 24);

    bool has_fluid_mesh_block = false;
    for (const auto& record : system.formulationRecords()) {
        for (const auto& [test_field, trial_field] : record.block_couplings) {
            if (test_field == velocity && trial_field == displacement) {
                has_fluid_mesh_block = true;
            }
        }
    }
    EXPECT_TRUE(has_fluid_mesh_block);

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = static_cast<FE::Real>(vertex);
        setFieldComponentValue(solution, system, velocity, vertex, 0,
                               FE::Real(0.35) + FE::Real(0.03) * x);
        setFieldComponentValue(solution, system, velocity, vertex, 1,
                               FE::Real(-0.20) + FE::Real(0.02) * x);
        setFieldComponentValue(solution, system, velocity, vertex, 2,
                               FE::Real(0.45) - FE::Real(0.015) * x);

        setFieldComponentValue(solution, system, displacement, vertex, 0,
                               FE::Real(0.04) + FE::Real(0.006) * x);
        setFieldComponentValue(solution, system, displacement, vertex, 1,
                               FE::Real(-0.025) + FE::Real(0.004) * x);
        setFieldComponentValue(solution, system, displacement, vertex, 2,
                               FE::Real(0.03) - FE::Real(0.005) * x);
    }

    expectOperatorJacobianMatchesMovingBoundaryFD(
        system, *mesh, displacement, solution, op,
        /*eps=*/1.0e-7, /*rtol=*/2.0e-5, /*atol=*/2.0e-7);
}

TEST(MovingDomainPhysics, NavierStokesWeakVelocityNitschePenaltyUsesTraceHeight)
{
    constexpr int marker = 21;
    auto u_space = FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1,
        FE::ElementType::Tetra4,
        /*order=*/2,
        /*components=*/3);
    auto p_space = FE::spaces::Space(
        FE::spaces::SpaceType::H1,
        FE::ElementType::Tetra4,
        /*order=*/1);

    auto opts = baseNavierStokesOptions();
    opts.velocity_dirichlet_weak.push_back(ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC{
        .boundary_marker = marker,
        .value = {0.0, 0.0, 0.0},
    });
    opts.nitsche_gamma = 8.0;
    opts.nitsche_scale_with_p = true;

    const auto u = FormExpr::trialFunction(*u_space, "u");
    const auto p = FormExpr::trialFunction(*p_space, "p");
    const auto v = FormExpr::testFunction(*u_space, "v");
    const auto q = FormExpr::testFunction(*p_space, "q");
    const auto mu = FormExpr::constant(0.04);
    auto momentum_form = FormExpr::constant(0.0).dx();
    auto continuity_form = FormExpr::constant(0.0).dx();

    ns::Factories::applyVelocityNitscheBCs(
        momentum_form,
        continuity_form,
        opts,
        /*dim=*/3,
        u,
        p,
        v,
        q,
        mu);

    EXPECT_TRUE(containsExprType(momentum_form, FormExprType::CellVolume));
    EXPECT_TRUE(containsExprType(momentum_form, FormExprType::FacetArea));
    EXPECT_FALSE(containsExprType(momentum_form, FormExprType::CellDiameter));
}

TEST(MovingDomainPhysics, NavierStokesVMS2DUsesPhysicalMetricShape)
{
    auto mesh = std::make_shared<FE::forms::test::SingleTriangleMeshAccess>();
    auto u_space = FE::spaces::VectorSpace(FE::spaces::SpaceType::H1, mesh, /*order=*/1, /*components=*/2);
    auto p_space = FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/1);
    auto opts = baseNavierStokesOptions();
    opts.enable_vms = true;
    opts.enable_convection = false;
    opts.velocity_field_name = "u";
    opts.pressure_field_name = "p";

    FE::systems::FESystem system(mesh);
    ns::IncompressibleNavierStokesVMSModule module(u_space, p_space, opts);
    module.registerOn(system);

    ASSERT_NO_THROW(system.setup({}, makeSingleTriangleSetupInputs()));
    ASSERT_EQ(system.dofHandler().getNumDofs(), 9);
}

TEST(MovingDomainPhysics, HarmonicMeshMotionRegistersDisplacementUnknownOnly)
{
    const auto mesh = makeMesh();
    auto d_space = makeVelocitySpace(mesh);

    mm::HarmonicMeshMotionOptions opts;
    opts.field_name = "mesh_displacement";
    opts.operator_tag = "mesh_motion";
    opts.kappa = 2.0;

    FE::systems::FESystem system(mesh);
    mm::HarmonicMeshMotionModule module(d_space, opts);
    module.registerOn(system);

    const auto displacement = system.findFieldByName("mesh_displacement");
    ASSERT_NE(displacement, FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.meshMotionField(FE::systems::MeshMotionFieldRole::Displacement),
              displacement);
    EXPECT_FALSE(system.meshMotionField(FE::systems::MeshMotionFieldRole::Velocity).has_value());
    EXPECT_FALSE(system.hasField("mesh_velocity"));
    EXPECT_EQ(system.fieldRecord(displacement).source_kind,
              FE::systems::FieldSourceKind::Unknown);
    EXPECT_TRUE(system.fieldParticipatesInUnknownVector(displacement));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Gradient));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CellIntegral));

    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
    EXPECT_EQ(system.dofHandler().getNumDofs(), 12);
    EXPECT_EQ(system.fieldMap().numFields(), 1u);
}

TEST(MovingDomainPhysics, HarmonicMeshMotionWithSpatialKappaMatchesFiniteDifference)
{
    const auto mesh = makeMesh();
    auto d_space = makeVelocitySpace(mesh);

    mm::HarmonicMeshMotionOptions opts;
    opts.operator_tag = "mesh_motion";
    opts.kappa = FE::forms::ScalarCoefficient{
        [](FE::Real x, FE::Real y, FE::Real z) {
            return 1.0 + x + 0.25 * y + 0.125 * z;
        }};

    FE::systems::FESystem system(mesh);
    mm::HarmonicMeshMotionModule module(d_space, opts);
    module.registerOn(system);
    system.setup({}, makeSingleTetraSetupInputs());

    const auto n = system.dofHandler().getNumDofs();
    ASSERT_EQ(n, 12);

    std::vector<FE::Real> u(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < u.size(); ++i) {
        u[i] = static_cast<FE::Real>(0.01 * (static_cast<int>(i) - 5));
    }

    FE::systems::SystemStateView state;
    state.u = std::span<const FE::Real>(u);
    expectOperatorJacobianMatchesCentralFD(
        system, state, "mesh_motion", /*eps=*/1e-6, /*rtol=*/1e-6, /*atol=*/1e-10);
}

TEST(MovingDomainPhysics, HarmonicMeshMotionNaturalBoundaryLoadAssembles)
{
    constexpr int marker = 7;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto d_space = makeVelocitySpace(mesh);

    mm::HarmonicMeshMotionOptions opts;
    opts.operator_tag = "mesh_motion";
    mm::HarmonicMeshMotionOptions::NaturalBC natural;
    natural.boundary_marker = marker;
    natural.value = {1.0, 0.0, 0.0};
    opts.natural.push_back(natural);

    FE::systems::FESystem system(mesh);
    mm::HarmonicMeshMotionModule module(d_space, opts);
    module.registerOn(system);
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::BoundaryIntegral));
    system.setup({}, makeSingleTetraSetupInputs());

    std::vector<FE::Real> u(static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    FE::systems::SystemStateView state;
    state.u = std::span<const FE::Real>(u);
    EXPECT_GT(residualNorm(system, state, "mesh_motion"), 0.0);
}

TEST(MovingDomainPhysics, HarmonicMeshMotionRobinBoundarySpringAssembles)
{
    constexpr int marker = 9;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto d_space = makeVelocitySpace(mesh);

    mm::HarmonicMeshMotionOptions opts;
    opts.operator_tag = "mesh_motion";
    mm::HarmonicMeshMotionOptions::RobinBC robin;
    robin.boundary_marker = marker;
    robin.alpha = 4.0;
    robin.target = {0.0, 1.0, 0.0};
    opts.robin.push_back(robin);

    FE::systems::FESystem system(mesh);
    mm::HarmonicMeshMotionModule module(d_space, opts);
    module.registerOn(system);
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::BoundaryIntegral));
    system.setup({}, makeSingleTetraSetupInputs());

    std::vector<FE::Real> u(static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    FE::systems::SystemStateView state;
    state.u = std::span<const FE::Real>(u);
    EXPECT_GT(residualNorm(system, state, "mesh_motion"), 0.0);
}

TEST(MovingDomainPhysics, HarmonicMeshMotionNormalConstraintAcceptsVelocityTarget)
{
    constexpr int marker = 10;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto d_space = makeVelocitySpace(mesh);

    mm::HarmonicMeshMotionOptions opts;
    opts.operator_tag = "mesh_motion";
    mm::NormalConstraintBC normal;
    normal.boundary_marker = marker;
    normal.quantity = mm::NormalConstraintQuantity::Velocity;
    normal.target = 2.0;
    normal.velocity_time_scale = 0.25;
    normal.penalty = 6.0;
    opts.normal_constraint.push_back(normal);

    FE::systems::FESystem system(mesh);
    mm::HarmonicMeshMotionModule module(d_space, opts);
    module.registerOn(system);
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::BoundaryIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Normal));
    system.setup({}, makeSingleTetraSetupInputs());

    std::vector<FE::Real> u(static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    FE::systems::SystemStateView state;
    state.u = std::span<const FE::Real>(u);
    EXPECT_GT(residualNorm(system, state, "mesh_motion"), 0.0);
}

TEST(MovingDomainPhysics, HarmonicMeshMotionTangentialPoliciesSelectBoundaryTerms)
{
    constexpr int marker = 11;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto d_space = makeVelocitySpace(mesh);

    const auto registers_boundary_integral =
        [&](mm::TangentialMeshPolicy policy) {
            mm::HarmonicMeshMotionOptions opts;
            opts.operator_tag = "mesh_motion";
            mm::TangentialPolicyBC tangent;
            tangent.boundary_marker = marker;
            tangent.policy = policy;
            tangent.quantity = mm::TangentialConstraintQuantity::Velocity;
            tangent.target = {1.0, 0.5, 0.0};
            tangent.velocity_time_scale = 0.25;
            tangent.penalty = 8.0;
            opts.tangential_policy.push_back(tangent);

            FE::systems::FESystem system(mesh);
            mm::HarmonicMeshMotionModule module(d_space, opts);
            module.registerOn(system);
            return formulationRecordsContain(system, FormExprType::BoundaryIntegral);
        };

    EXPECT_FALSE(registers_boundary_integral(mm::TangentialMeshPolicy::Free));
    EXPECT_FALSE(registers_boundary_integral(mm::TangentialMeshPolicy::SmoothingOnly));
    EXPECT_TRUE(registers_boundary_integral(mm::TangentialMeshPolicy::Prescribed));
}

TEST(MovingDomainPhysics, HarmonicMeshMotionWeakBoundaryTermsOnSameMarkerAreAdditive)
{
    constexpr int marker = 12;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto d_space = makeVelocitySpace(mesh);

    mm::HarmonicMeshMotionOptions::NaturalBC natural;
    natural.boundary_marker = marker;
    natural.value = {1.0, 0.0, 0.0};

    mm::HarmonicMeshMotionOptions::RobinBC robin;
    robin.boundary_marker = marker;
    robin.alpha = 2.0;
    robin.target = {0.0, 1.0, 0.0};

    const auto assemble = [&](const mm::HarmonicMeshMotionOptions& opts) {
        FE::systems::FESystem system(mesh);
        mm::HarmonicMeshMotionModule module(d_space, opts);
        module.registerOn(system);
        system.setup({}, makeSingleTetraSetupInputs());

        std::vector<FE::Real> u(static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
        FE::systems::SystemStateView state;
        state.u = std::span<const FE::Real>(u);
        return residualVector(system, state, "mesh_motion");
    };

    mm::HarmonicMeshMotionOptions combined_opts;
    combined_opts.operator_tag = "mesh_motion";
    combined_opts.natural.push_back(natural);
    combined_opts.robin.push_back(robin);

    mm::HarmonicMeshMotionOptions equivalent_opts;
    equivalent_opts.operator_tag = "mesh_motion";
    auto equivalent_robin = robin;
    // NaturalBC adds to RobinBC's boundary RHS, so alpha * target_x increases by 1.
    equivalent_robin.target = {0.5, 1.0, 0.0};
    equivalent_opts.robin.push_back(equivalent_robin);

    const auto combined_residual = assemble(combined_opts);
    const auto equivalent_residual = assemble(equivalent_opts);

    ASSERT_EQ(combined_residual.size(), equivalent_residual.size());
    for (std::size_t i = 0; i < combined_residual.size(); ++i) {
        EXPECT_NEAR(combined_residual[i],
                    equivalent_residual[i],
                    1.0e-12);
    }
}

TEST(MovingDomainPhysics, HarmonicMeshMotionRobinTargetMatchesEquivalentNaturalLoad)
{
    constexpr int marker = 13;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto d_space = makeVelocitySpace(mesh);

    mm::HarmonicMeshMotionOptions::RobinBC robin;
    robin.boundary_marker = marker;
    robin.alpha = 4.0;
    robin.target = {0.0, 1.5, -0.5};

    mm::HarmonicMeshMotionOptions robin_opts;
    robin_opts.operator_tag = "mesh_motion";
    robin_opts.robin.push_back(robin);

    mm::HarmonicMeshMotionOptions::RobinBC homogeneous_robin = robin;
    homogeneous_robin.target = {0.0, 0.0, 0.0};

    mm::HarmonicMeshMotionOptions::NaturalBC equivalent_load;
    equivalent_load.boundary_marker = marker;
    equivalent_load.value = {0.0, 6.0, -2.0};

    mm::HarmonicMeshMotionOptions split_opts;
    split_opts.operator_tag = "mesh_motion";
    split_opts.robin.push_back(homogeneous_robin);
    split_opts.natural.push_back(equivalent_load);

    const auto assemble = [&](const mm::HarmonicMeshMotionOptions& opts) {
        FE::systems::FESystem system(mesh);
        mm::HarmonicMeshMotionModule module(d_space, opts);
        module.registerOn(system);
        system.setup({}, makeSingleTetraSetupInputs());

        const auto n = system.dofHandler().getNumDofs();
        std::vector<FE::Real> u(static_cast<std::size_t>(n), 0.0);
        for (std::size_t i = 0; i < u.size(); ++i) {
            u[i] = static_cast<FE::Real>(0.01 * (static_cast<int>(i) + 1));
        }

        FE::systems::SystemStateView state;
        state.u = std::span<const FE::Real>(u);
        return residualVector(system, state, "mesh_motion");
    };

    const auto robin_residual = assemble(robin_opts);
    const auto split_residual = assemble(split_opts);

    ASSERT_EQ(robin_residual.size(), split_residual.size());
    for (std::size_t i = 0; i < robin_residual.size(); ++i) {
        EXPECT_NEAR(robin_residual[i], split_residual[i], 1.0e-12);
    }
}

TEST(MovingDomainPhysics, HarmonicMeshMotionDirichletConflictsWithWeakBoundaryTerms)
{
    constexpr int marker = 14;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto d_space = makeVelocitySpace(mesh);

    mm::HarmonicMeshMotionOptions::DirichletBC dirichlet;
    dirichlet.boundary_marker = marker;
    dirichlet.value = {0.0, 0.0, 0.0};

    mm::HarmonicMeshMotionOptions::NaturalBC natural;
    natural.boundary_marker = marker;
    natural.value = {1.0, 0.0, 0.0};

    mm::HarmonicMeshMotionOptions natural_conflict;
    natural_conflict.operator_tag = "mesh_motion";
    natural_conflict.dirichlet.push_back(dirichlet);
    natural_conflict.natural.push_back(natural);
    {
        FE::systems::FESystem system(mesh);
        mm::HarmonicMeshMotionModule module(d_space, natural_conflict);
        EXPECT_THROW(module.registerOn(system), std::invalid_argument);
    }

    mm::HarmonicMeshMotionOptions::RobinBC robin;
    robin.boundary_marker = marker;
    robin.alpha = 3.0;
    robin.target = {0.0, 0.0, 0.0};

    mm::HarmonicMeshMotionOptions robin_conflict;
    robin_conflict.operator_tag = "mesh_motion";
    robin_conflict.dirichlet.push_back(dirichlet);
    robin_conflict.robin.push_back(robin);
    {
        FE::systems::FESystem system(mesh);
        mm::HarmonicMeshMotionModule module(d_space, robin_conflict);
        EXPECT_THROW(module.registerOn(system), std::invalid_argument);
    }
}

TEST(MovingDomainPhysics, MeshMotionDirichletComponentCoefficientNamesUseComponentStyle)
{
    constexpr int marker = 15;
    const std::array<mm::HarmonicMeshMotionOptions::ScalarValue, 3> values = {
        mm::HarmonicMeshMotionOptions::ScalarValue{FE::forms::ScalarCoefficient(
            [](FE::Real, FE::Real, FE::Real) { return 1.0; })},
        mm::HarmonicMeshMotionOptions::ScalarValue{FE::forms::ScalarCoefficient(
            [](FE::Real, FE::Real, FE::Real) { return 2.0; })},
        mm::HarmonicMeshMotionOptions::ScalarValue{FE::forms::ScalarCoefficient(
            [](FE::Real, FE::Real, FE::Real) { return 3.0; })},
    };

    auto components = FE::forms::bc::toVectorExpr(
        values,
        /*dim=*/3,
        "mesh_displacement",
        marker,
        FE::forms::bc::ComponentValueNameStyle::Component);
    FE::forms::bc::EssentialBC bc(marker, std::move(components), "d_mesh");
    const auto strong = bc.getStrongConstraints(/*field_id=*/123);

    ASSERT_EQ(strong.size(), 3u);
    EXPECT_EQ(strong[0].value.toString(), "mesh_displacement_14_c0");
    EXPECT_EQ(strong[1].value.toString(), "mesh_displacement_14_c1");
    EXPECT_EQ(strong[2].value.toString(), "mesh_displacement_14_c2");
}

TEST(MovingDomainPhysics, HarmonicMeshMotionRejectsInvalidLiteralParameters)
{
    const auto mesh = makeMesh();
    auto d_space = makeVelocitySpace(mesh);

    const auto expect_invalid = [&](const mm::HarmonicMeshMotionOptions& opts,
                                    std::string_view expected_message) {
        FE::systems::FESystem system(mesh);
        mm::HarmonicMeshMotionModule module(d_space, opts);
        try {
            module.registerOn(system);
            FAIL() << "expected std::invalid_argument";
        } catch (const std::invalid_argument& ex) {
            EXPECT_NE(std::string(ex.what()).find(expected_message), std::string::npos)
                << ex.what();
        }
    };

    mm::HarmonicMeshMotionOptions zero_kappa;
    zero_kappa.kappa = 0.0;
    expect_invalid(zero_kappa, "kappa must be positive");

    mm::HarmonicMeshMotionOptions negative_kappa;
    negative_kappa.kappa = -1.0;
    expect_invalid(negative_kappa, "kappa must be positive");

    mm::HarmonicMeshMotionOptions zero_stiffness;
    zero_stiffness.stiffness = 0.0;
    expect_invalid(zero_stiffness, "stiffness must be positive");

    mm::HarmonicMeshMotionOptions negative_stiffness;
    negative_stiffness.stiffness = -1.0;
    expect_invalid(negative_stiffness, "stiffness must be positive");

    mm::HarmonicMeshMotionOptions conflicting_literals;
    conflicting_literals.kappa = 2.0;
    conflicting_literals.stiffness = 3.0;
    expect_invalid(conflicting_literals,
                   "both kappa and deprecated stiffness were set");
}

TEST(MovingDomainPhysics, PseudoElasticMeshMotionMatchesFiniteDifference)
{
    const auto mesh = makeMesh();
    auto d_space = makeVelocitySpace(mesh);

    mm::PseudoElasticMeshMotionOptions opts;
    opts.operator_tag = "mesh_motion";
    opts.lambda_mesh = FE::forms::ScalarCoefficient{
        [](FE::Real x, FE::Real, FE::Real) { return 1.5 + 0.25 * x; }};
    opts.mu_mesh = FE::forms::ScalarCoefficient{
        [](FE::Real, FE::Real y, FE::Real z) { return 0.75 + 0.125 * y + 0.0625 * z; }};

    FE::systems::FESystem system(mesh);
    mm::PseudoElasticMeshMotionModule module(d_space, opts);
    module.registerOn(system);
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::SymmetricPart));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Trace));
    system.setup({}, makeSingleTetraSetupInputs());

    const auto n = system.dofHandler().getNumDofs();
    ASSERT_EQ(n, 12);

    std::vector<FE::Real> u(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < u.size(); ++i) {
        u[i] = static_cast<FE::Real>(0.005 * (static_cast<int>(i) - 4));
    }

    FE::systems::SystemStateView state;
    state.u = std::span<const FE::Real>(u);
    expectOperatorJacobianMatchesCentralFD(
        system, state, "mesh_motion", /*eps=*/1e-6, /*rtol=*/1e-6, /*atol=*/1e-10);
}

TEST(MovingDomainPhysics, PseudoElasticMeshMotionWeakBoundaryTermsOnSameMarkerAreAdditive)
{
    constexpr int marker = 15;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto d_space = makeVelocitySpace(mesh);

    mm::PseudoElasticMeshMotionOptions::NaturalBC natural;
    natural.boundary_marker = marker;
    natural.value = {0.0, 2.0, 0.0};

    mm::PseudoElasticMeshMotionOptions::RobinBC robin;
    robin.boundary_marker = marker;
    robin.alpha = 4.0;
    robin.target = {1.0, 0.0, 0.0};

    const auto assemble = [&](const mm::PseudoElasticMeshMotionOptions& opts) {
        FE::systems::FESystem system(mesh);
        mm::PseudoElasticMeshMotionModule module(d_space, opts);
        module.registerOn(system);
        system.setup({}, makeSingleTetraSetupInputs());

        std::vector<FE::Real> u(static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
        FE::systems::SystemStateView state;
        state.u = std::span<const FE::Real>(u);
        return residualVector(system, state, "mesh_motion");
    };

    mm::PseudoElasticMeshMotionOptions combined_opts;
    combined_opts.operator_tag = "mesh_motion";
    combined_opts.natural.push_back(natural);
    combined_opts.robin.push_back(robin);

    mm::PseudoElasticMeshMotionOptions equivalent_opts;
    equivalent_opts.operator_tag = "mesh_motion";
    auto equivalent_robin = robin;
    equivalent_robin.target = {1.0, 0.5, 0.0};
    equivalent_opts.robin.push_back(equivalent_robin);

    const auto combined_residual = assemble(combined_opts);
    const auto equivalent_residual = assemble(equivalent_opts);

    ASSERT_EQ(combined_residual.size(), equivalent_residual.size());
    for (std::size_t i = 0; i < combined_residual.size(); ++i) {
        EXPECT_NEAR(combined_residual[i],
                    equivalent_residual[i],
                    1.0e-12);
    }
}

TEST(MovingDomainPhysics, MeshMotionModulesInstallEquivalentBoundaryConditionDescriptors)
{
    auto mesh = makeMesh();
    auto d_space = makeVelocitySpace(mesh);

    mm::HarmonicMeshMotionOptions harmonic_opts;
    harmonic_opts.operator_tag = "mesh_motion";
    harmonic_opts.natural.push_back(mm::HarmonicMeshMotionOptions::NaturalBC{
        .boundary_marker = 21,
        .value = {1.0, -2.0, 0.5},
    });
    harmonic_opts.robin.push_back(mm::HarmonicMeshMotionOptions::RobinBC{
        .boundary_marker = 22,
        .alpha = 3.0,
        .target = {0.25, -0.5, 1.0},
    });
    harmonic_opts.dirichlet.push_back(mm::HarmonicMeshMotionOptions::DirichletBC{
        .boundary_marker = 23,
        .value = {0.0, 0.1, -0.2},
    });

    mm::PseudoElasticMeshMotionOptions pseudo_opts;
    pseudo_opts.operator_tag = "mesh_motion";
    pseudo_opts.natural.push_back(mm::PseudoElasticMeshMotionOptions::NaturalBC{
        .boundary_marker = 21,
        .value = {1.0, -2.0, 0.5},
    });
    pseudo_opts.robin.push_back(mm::PseudoElasticMeshMotionOptions::RobinBC{
        .boundary_marker = 22,
        .alpha = 3.0,
        .target = {0.25, -0.5, 1.0},
    });
    pseudo_opts.dirichlet.push_back(mm::PseudoElasticMeshMotionOptions::DirichletBC{
        .boundary_marker = 23,
        .value = {0.0, 0.1, -0.2},
    });

    FE::systems::FESystem harmonic_system(mesh);
    mm::HarmonicMeshMotionModule harmonic_module(d_space, harmonic_opts);
    harmonic_module.registerOn(harmonic_system);

    FE::systems::FESystem pseudo_system(mesh);
    mm::PseudoElasticMeshMotionModule pseudo_module(d_space, pseudo_opts);
    pseudo_module.registerOn(pseudo_system);

    const auto& harmonic_desc = harmonic_system.boundaryConditionDescriptors();
    const auto& pseudo_desc = pseudo_system.boundaryConditionDescriptors();
    ASSERT_EQ(harmonic_desc.size(), pseudo_desc.size());
    ASSERT_EQ(harmonic_desc.size(), 5u);

    for (std::size_t i = 0; i < harmonic_desc.size(); ++i) {
        EXPECT_EQ(harmonic_desc[i].boundary_marker, pseudo_desc[i].boundary_marker);
        EXPECT_EQ(harmonic_desc[i].component, pseudo_desc[i].component);
        EXPECT_EQ(harmonic_desc[i].trace_kind, pseudo_desc[i].trace_kind);
        EXPECT_EQ(harmonic_desc[i].enforcement_kind, pseudo_desc[i].enforcement_kind);
        EXPECT_EQ(harmonic_desc[i].source, pseudo_desc[i].source);
    }
}

TEST(MovingDomainPhysics, PseudoElasticMeshMotionRejectsInvalidLiteralParameters)
{
    const auto mesh = makeMesh();
    auto d_space = makeVelocitySpace(mesh);

    const auto expect_invalid = [&](const mm::PseudoElasticMeshMotionOptions& opts,
                                    std::string_view expected_message) {
        FE::systems::FESystem system(mesh);
        mm::PseudoElasticMeshMotionModule module(d_space, opts);
        try {
            module.registerOn(system);
            FAIL() << "expected std::invalid_argument";
        } catch (const std::invalid_argument& ex) {
            EXPECT_NE(std::string(ex.what()).find(expected_message), std::string::npos)
                << ex.what();
        }
    };

    mm::PseudoElasticMeshMotionOptions zero_lambda;
    zero_lambda.lambda_mesh = 0.0;
    expect_invalid(zero_lambda, "lambda_mesh must be positive");

    mm::PseudoElasticMeshMotionOptions negative_lambda;
    negative_lambda.lambda_mesh = -1.0;
    expect_invalid(negative_lambda, "lambda_mesh must be positive");

    mm::PseudoElasticMeshMotionOptions zero_mu;
    zero_mu.mu_mesh = 0.0;
    expect_invalid(zero_mu, "mu_mesh must be positive");

    mm::PseudoElasticMeshMotionOptions negative_mu;
    negative_mu.mu_mesh = -1.0;
    expect_invalid(negative_mu, "mu_mesh must be positive");
}

TEST(MovingDomainPhysics, CoupledALEAndHarmonicMeshMotionShareDisplacementUnknown)
{
    const auto mesh = makeMesh();
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);

    FE::systems::FESystem system(mesh);

    mm::HarmonicMeshMotionOptions mesh_opts;
    mesh_opts.operator_tag = "mesh_motion";
    mm::HarmonicMeshMotionModule mesh_module(u_space, mesh_opts);
    mesh_module.registerOn(system);
    const auto displacement = system.findFieldByName("mesh_displacement");
    ASSERT_NE(displacement, FE::INVALID_FIELD_ID);

    auto ns_opts = baseNavierStokesOptions();
    ns_opts.enable_ale = true;
    ns_opts.mesh_velocity_source = ns::ALEMeshVelocitySource::CoupledDisplacement;
    ns_opts.mesh_displacement_field_name = "mesh_displacement";
    ns_opts.mesh_velocity_field_name = "mesh_velocity";

    ns::IncompressibleNavierStokesVMSModule ns_module(u_space, p_space, ns_opts);
    ns_module.registerOn(system);

    const auto mesh_velocity = system.findFieldByName("mesh_velocity");
    ASSERT_NE(mesh_velocity, FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.meshMotionField(FE::systems::MeshMotionFieldRole::Displacement),
              displacement);
    EXPECT_EQ(system.fieldRecord(mesh_velocity).source_kind,
              FE::systems::FieldSourceKind::DerivedFromUnknown);
    EXPECT_EQ(system.fieldRecord(mesh_velocity).derived.source_field, displacement);
    EXPECT_FALSE(system.fieldParticipatesInUnknownVector(mesh_velocity));

    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
    EXPECT_EQ(system.dofHandler().getNumDofs(), 28);
    EXPECT_EQ(system.fieldMap().numFields(), 3u);

    bool has_mesh_rows = false;
    bool has_fluid_mesh_columns = false;
    const auto u = system.findFieldByName(ns_opts.velocity_field_name);
    const auto p = system.findFieldByName(ns_opts.pressure_field_name);
    for (const auto& record : system.formulationRecords()) {
        for (const auto& [test_field, trial_field] : record.block_couplings) {
            if (test_field == displacement && trial_field == displacement) {
                has_mesh_rows = true;
            }
            if (trial_field == displacement && (test_field == u || test_field == p)) {
                has_fluid_mesh_columns = true;
            }
        }
    }
    EXPECT_TRUE(has_mesh_rows);
    EXPECT_TRUE(has_fluid_mesh_columns);
}

TEST(MovingDomainPhysics, CoupledFittedFreeSurfaceALEAndHarmonicMeshMotionSetup)
{
    constexpr int marker = 39;
    auto mesh = std::make_shared<SingleTetraBoundaryMeshAccess>(marker);
    auto u_space = makeVelocitySpace(mesh);
    auto p_space = makePressureSpace(mesh);

    FE::systems::FESystem system(mesh);

    mm::HarmonicMeshMotionOptions mesh_opts;
    mesh_opts.field_name = "mesh_displacement";
    mesh_opts.operator_tag = "mesh_motion";
    mesh_opts.kappa = 1.5;

    mm::NormalConstraintBC normal;
    normal.boundary_marker = marker;
    normal.quantity = mm::NormalConstraintQuantity::Velocity;
    normal.target = 0.15;
    normal.penalty = 6.0;
    normal.velocity_time_scale = 1.0;
    mesh_opts.normal_constraint.push_back(normal);

    mm::TangentialPolicyBC tangent;
    tangent.boundary_marker = marker;
    tangent.policy = mm::TangentialMeshPolicy::Prescribed;
    tangent.quantity = mm::TangentialConstraintQuantity::Velocity;
    tangent.target = {0.05, -0.02, 0.01};
    tangent.penalty = 5.0;
    tangent.velocity_time_scale = 1.0;
    mesh_opts.tangential_policy.push_back(tangent);

    mm::HarmonicMeshMotionModule mesh_module(u_space, mesh_opts);
    mesh_module.registerOn(system);
    const auto displacement = system.findFieldByName("mesh_displacement");
    ASSERT_NE(displacement, FE::INVALID_FIELD_ID);

    auto ns_opts = baseNavierStokesOptions();
    ns_opts.enable_ale = true;
    ns_opts.enable_convection = false;
    ns_opts.mesh_velocity_source = ns::ALEMeshVelocitySource::CoupledDisplacement;
    ns_opts.mesh_displacement_field_name = "mesh_displacement";
    ns_opts.mesh_velocity_field_name = "mesh_velocity";

    ns_opts.free_surface.push_back(ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::FittedALE,
        .boundary_marker = marker,
        .external_pressure = 1.25,
        .kinematic_enforcement = ns::FreeSurfaceKinematicEnforcement::Penalty,
        .kinematic_penalty = 9.0,
    });

    ns::IncompressibleNavierStokesVMSModule ns_module(u_space, p_space, ns_opts);
    ns_module.registerOn(system);

    const auto mesh_velocity = system.findFieldByName("mesh_velocity");
    ASSERT_NE(mesh_velocity, FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.meshMotionField(FE::systems::MeshMotionFieldRole::Displacement),
              displacement);
    EXPECT_EQ(system.meshMotionField(FE::systems::MeshMotionFieldRole::Velocity),
              mesh_velocity);
    EXPECT_EQ(system.fieldRecord(mesh_velocity).source_kind,
              FE::systems::FieldSourceKind::DerivedFromUnknown);
    EXPECT_EQ(system.fieldRecord(mesh_velocity).derived.source_field, displacement);
    EXPECT_FALSE(system.fieldParticipatesInUnknownVector(mesh_velocity));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::BoundaryIntegral));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::MeshVelocity));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CurrentNormal));
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CurrentMeasure));

    ASSERT_NO_THROW(system.setup({}, makeSingleTetraSetupInputs()));
    EXPECT_EQ(system.dofHandler().getNumDofs(), 28);
    EXPECT_EQ(system.fieldMap().numFields(), 3u);
    ASSERT_NE(system.blockMap(), nullptr);
    EXPECT_EQ(system.blockMap()->numBlocks(), 3u);

    const auto u = system.findFieldByName(ns_opts.velocity_field_name);
    const auto p = system.findFieldByName(ns_opts.pressure_field_name);
    bool has_mesh_rows = false;
    bool has_fluid_mesh_columns = false;
    bool has_mesh_fluid_columns = false;
    for (const auto& record : system.formulationRecords()) {
        for (const auto& [test_field, trial_field] : record.block_couplings) {
            if (test_field == displacement && trial_field == displacement) {
                has_mesh_rows = true;
            }
            if (trial_field == displacement && (test_field == u || test_field == p)) {
                has_fluid_mesh_columns = true;
            }
            if (test_field == displacement && trial_field == u) {
                has_mesh_fluid_columns = true;
            }
        }
    }
    EXPECT_TRUE(has_mesh_rows);
    EXPECT_TRUE(has_fluid_mesh_columns);
    EXPECT_TRUE(has_mesh_fluid_columns);

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    std::vector<FE::Real> previous_solution = solution;
    for (FE::GlobalIndex vertex = 0; vertex < 4; ++vertex) {
        const auto x = static_cast<FE::Real>(vertex);
        setFieldComponentValue(solution, system, u, vertex, 0,
                               FE::Real(0.30) + FE::Real(0.01) * x);
        setFieldComponentValue(solution, system, u, vertex, 1,
                               FE::Real(-0.10) + FE::Real(0.015) * x);
        setFieldComponentValue(solution, system, u, vertex, 2,
                               FE::Real(0.20) - FE::Real(0.005) * x);
        setFieldComponentValue(solution, system, p, vertex, 0,
                               FE::Real(0.05) + FE::Real(0.02) * x);

        setFieldComponentValue(solution, system, displacement, vertex, 0,
                               FE::Real(0.02) + FE::Real(0.004) * x);
        setFieldComponentValue(solution, system, displacement, vertex, 1,
                               FE::Real(-0.015) + FE::Real(0.003) * x);
        setFieldComponentValue(solution, system, displacement, vertex, 2,
                               FE::Real(0.01) - FE::Real(0.002) * x);
    }
    updateBoundaryMeshCurrentCoordinates(*mesh, system, displacement, solution);

    FE::systems::SystemStateView state;
    state.dt = 1.0;
    state.u = std::span<const FE::Real>(solution);
    state.u_prev = std::span<const FE::Real>(previous_solution);
    const FE::systems::BackwardDifferenceIntegrator integrator;
    const auto time_context = integrator.buildContext(/*max_time_derivative_order=*/1, state);
    state.time_integration = &time_context;

    EXPECT_GT(residualNorm(system, state, "mesh_motion"), 0.0);
    EXPECT_GT(residualNorm(system, state, "equations"), 0.0);
}

TEST(MovingDomainPhysics, ALEAdvectionDiffusionManufacturedResidualUsesPhysicalMinusMeshVelocity)
{
    using namespace FE::forms;

    const auto phi = manufacturedScalarField();
    const auto psi = FormExpr::constant(1.0);
    const auto rho = FormExpr::constant(2.0);
    const auto physical_advection = constantVector3(1.0, -0.25, 0.5);
    const auto w_mesh = meshVelocity();
    const auto relative_advection = physical_advection - w_mesh;

    const auto residual =
        rho * div(w_mesh) * phi * psi +
        rho * dot(relative_advection, grad(phi)) * psi +
        FormExpr::constant(0.01) * dot(grad(phi), grad(psi));

    EXPECT_TRUE(containsExprType(residual, FormExprType::MeshVelocity));
}

TEST(MovingDomainPhysics, ExplicitPhysicalMinusMeshVelocityAssemblesCorrectly)
{
    using namespace FE::forms;

    SingleTetraMeshAccess mesh;
    FE::spaces::H1Space scalar_space(FE::ElementType::Tetra4, 1);
    auto scalar_dof_map = createSingleTetraDenseDofMap(4);

    auto base_space = std::make_shared<FE::spaces::H1Space>(FE::ElementType::Tetra4, 1);
    FE::spaces::ProductSpace vector_space(base_space, 3);
    auto vector_dof_map = createSingleTetraDenseDofMap(12);

    const auto u = FormExpr::trialFunction(scalar_space, "temperature");
    const auto v = FormExpr::testFunction(scalar_space, "test");
    const auto rho = FormExpr::constant(2.0);

    const auto ale_relative = constantVector3(1.0, -0.25, 0.5) - meshVelocity();
    const auto static_equivalent = constantVector3(0.75, -0.125, 0.0);

    const auto ale_integrand = rho * dot(ale_relative, grad(u)) * v;
    const auto static_integrand = rho * dot(static_equivalent, grad(u)) * v;

    const std::vector<FE::Real> ale_solution = {0.0, 1.0, 1.0, 1.0};
    const auto mesh_velocity = constantVectorTetraCoefficients(0.25, -0.125, 0.5);

    const auto ale_residual = assembleMovingDomainScalarResidual(mesh,
                                                                 scalar_space,
                                                                 scalar_dof_map,
                                                                 &vector_space,
                                                                 &vector_dof_map,
                                                                 ale_integrand,
                                                                 ale_solution,
                                                                 mesh_velocity);

    const std::vector<FE::Real> static_solution = {0.0, 1.0, 1.0, 1.0};
    const auto static_residual = assembleMovingDomainScalarResidual(mesh,
                                                                    scalar_space,
                                                                    scalar_dof_map,
                                                                    nullptr,
                                                                    nullptr,
                                                                    static_integrand,
                                                                    static_solution);

    for (FE::GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(ale_residual.getVectorEntry(i),
                    static_residual.getVectorEntry(i),
                    1.0e-12);
    }
}

TEST(MovingDomainPhysics, MovingControlVolumeDivergenceTermAssemblesKnownValue)
{
    using namespace FE::forms;

    SingleTetraMeshAccess mesh;
    FE::spaces::H1Space scalar_space(FE::ElementType::Tetra4, 1);
    auto scalar_dof_map = createSingleTetraDenseDofMap(4);

    auto base_space = std::make_shared<FE::spaces::H1Space>(FE::ElementType::Tetra4, 1);
    FE::spaces::ProductSpace vector_space(base_space, 3);
    auto vector_dof_map = createSingleTetraDenseDofMap(12);

    const auto u = FormExpr::trialFunction(scalar_space, "temperature");
    const auto v = FormExpr::testFunction(scalar_space, "test");
    const auto integrand = FormExpr::constant(2.0) * div(meshVelocity()) * u * v;

    const std::vector<FE::Real> solution = constantScalarTetraCoefficients(3.0);
    const auto mesh_velocity = affineXVectorTetraCoefficients();

    const auto residual = assembleMovingDomainScalarResidual(mesh,
                                                             scalar_space,
                                                             scalar_dof_map,
                                                             &vector_space,
                                                             &vector_dof_map,
                                                             integrand,
                                                             solution,
                                                             mesh_velocity);

    // div(w)=1 for w=(x,0,0), u=3, rho=2, and int_T phi_i dx = volume/4 = 1/24.
    const FE::Real expected = 2.0 * 1.0 * 3.0 * (1.0 / 24.0);
    for (FE::GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(residual.getVectorEntry(i), expected, 1.0e-12);
    }
}

TEST(MovingDomainPhysics, ALEIncompressibleNavierStokesManufacturedResidualUsesMovingDomainExpressions)
{
    using namespace FE::forms;

    const auto x0 = component(currentCoordinate(), 0);
    const auto x1 = component(currentCoordinate(), 1);
    const auto x2 = component(currentCoordinate(), 2);
    const auto u = FormExpr::asVector({
        x0 + t(),
        x1 * x1,
        x2 - FormExpr::constant(0.25) * t(),
    });
    const auto p = x0 - x1 + FormExpr::constant(0.5) * x2;
    const auto v = constantVector3(0.5, -1.0, 0.25);
    const auto q = FormExpr::constant(2.0);
    const auto rho = FormExpr::constant(1.25);
    const auto mu = FormExpr::constant(0.02);
    const auto stress = FormExpr::constant(2.0) * mu * sym(grad(u));
    const auto w_mesh = meshVelocity();
    const auto relative_advection = u - w_mesh;

    const auto momentum =
        rho * inner(dt(u) + grad(u) * relative_advection, v) +
        rho * div(w_mesh) * inner(u, v) +
        FormExpr::constant(2.0) * mu * inner(sym(grad(u)), sym(grad(v))) -
        p * div(v);
    const auto continuity = q * div(u);
    const auto residual = momentum + continuity - inner(div(stress), v);

    EXPECT_TRUE(containsExprType(residual, FormExprType::MeshVelocity));
}

TEST(MovingDomainPhysics, MovingBoundaryFlowSmokeUsesGenericBoundaryTerminals)
{
    const auto test_scalar = FormExpr::constant(1.0);
    const auto boundary_velocity = constantVector3(0.0, 0.0, 1.0);

    const auto residual = movingBoundaryKinematicResidual(boundary_velocity, test_scalar);

    EXPECT_TRUE(containsExprType(residual, FormExprType::MeshVelocity));
    EXPECT_TRUE(containsExprType(residual, FormExprType::CurrentNormal));
    EXPECT_TRUE(containsExprType(residual, FormExprType::CurrentMeasure));
}

TEST(MovingDomainPhysics, FSIInterfaceKinematicsAndTractionsUseGenericGeometryTerminals)
{
    const auto test_scalar = FormExpr::constant(1.0);
    const auto structural_displacement = constantVector3(0.1, -0.2, 0.3);
    const auto traction = constantVector3(2.0, 3.0, 4.0);
    const auto velocity_test = constantVector3(0.25, 0.5, 0.75);

    const auto residual =
        fsiDisplacementCompatibilityResidual(structural_displacement, test_scalar) +
        fsiSurfaceTractionPowerResidual(traction, velocity_test) +
        referenceSurfaceMeasureMismatchProbe();

    EXPECT_TRUE(containsExprType(residual, FormExprType::MeshDisplacement));
    EXPECT_TRUE(containsExprType(residual, FormExprType::CurrentNormal));
    EXPECT_TRUE(containsExprType(residual, FormExprType::CurrentMeasure));
    EXPECT_TRUE(containsExprType(residual, FormExprType::ReferenceNormal));
    EXPECT_TRUE(containsExprType(residual, FormExprType::ReferenceMeasure));
}

} // namespace test
} // namespace Physics
} // namespace svmp
