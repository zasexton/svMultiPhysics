/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Physics/Formulations/MeshMotion/HarmonicMeshMotionModule.h"
#include "Physics/Formulations/MeshMotion/PseudoElasticMeshMotionModule.h"
#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"
#include "Physics/Tests/Unit/PhysicsTestHelpers.h"

#include "FE/Forms/FormExpr.h"
#include "FE/Forms/FormCompiler.h"
#include "FE/Forms/FormKernels.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Assembly/StandardAssembler.h"
#include "FE/Dofs/DofMap.h"
#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/ProductSpace.h"
#include "FE/Spaces/SpaceFactory.h"
#include "FE/Systems/FESystem.h"

#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace svmp {
namespace Physics {
namespace test {
namespace {

using FE::forms::FormExpr;
using FE::forms::FormExprNode;
using FE::forms::FormExprType;
constexpr FE::FieldId kMeshVelocityField = 907;
namespace mm = formulations::mesh_motion;
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

std::shared_ptr<SingleTetraMeshAccess> makeMesh()
{
    return std::make_shared<SingleTetraMeshAccess>();
}

class SingleTetraBoundaryMeshAccess final : public FE::assembly::IMeshAccess {
public:
    explicit SingleTetraBoundaryMeshAccess(int marker)
        : marker_(marker)
    {
        nodes_ = {
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
        };
        cell_ = {0, 1, 2, 3};
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 1; }
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

private:
    int marker_{-1};
    std::vector<std::array<FE::Real, 3>> nodes_{};
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
