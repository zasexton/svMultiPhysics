/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

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

#include <cstddef>
#include <functional>
#include <memory>
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

FormExpr meshVelocityVector(int dim)
{
    using namespace FE::forms;

    std::vector<FormExpr> components;
    components.reserve(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        components.push_back(component(meshVelocity(), d));
    }
    return FormExpr::asVector(std::move(components));
}

FormExpr relativeConvectiveVelocity(const FormExpr& material_velocity,
                                    int dim,
                                    bool ale_enabled)
{
    return ale_enabled ? (material_velocity - meshVelocityVector(dim)) : material_velocity;
}

FormExpr movingControlVolumeScalarTransient(const FormExpr& field,
                                            const FormExpr& test,
                                            const FormExpr& density,
                                            int dim,
                                            bool enabled)
{
    using namespace FE::forms;

    if (!enabled) {
        return FormExpr::constant(0.0);
    }
    return density * div(meshVelocityVector(dim)) * field * test;
}

FormExpr movingControlVolumeVectorTransient(const FormExpr& field,
                                            const FormExpr& test,
                                            const FormExpr& density,
                                            int dim,
                                            bool enabled)
{
    using namespace FE::forms;

    if (!enabled) {
        return FormExpr::constant(0.0);
    }
    return density * div(meshVelocityVector(dim)) * inner(field, test);
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

FE::assembly::DenseVectorView assembleMovingDomainScalarResidual(
    const FE::assembly::IMeshAccess& mesh,
    const FE::spaces::FunctionSpace& scalar_space,
    FE::dofs::DofMap& scalar_dof_map,
    const FE::spaces::FunctionSpace* mesh_velocity_space,
    const FE::dofs::DofMap* mesh_velocity_dof_map,
    const FormExpr& residual_integrand,
    const std::vector<FE::Real>& current_solution)
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
                .dof_offset = scalar_dof_map.getNumDofs(),
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
    EXPECT_TRUE(formulationRecordsContain(system, FormExprType::MeshVelocity));
}

TEST(MovingDomainPhysics, ALEAdvectionDiffusionManufacturedResidualUsesRelativeMeshVelocity)
{
    using namespace FE::forms;

    const auto phi = manufacturedScalarField();
    const auto psi = FormExpr::constant(1.0);
    const auto rho = FormExpr::constant(2.0);
    const auto physical_advection = constantVector3(1.0, -0.25, 0.5);
    const auto relative_advection =
        relativeConvectiveVelocity(physical_advection, /*dim=*/3, /*ale_enabled=*/true);

    const auto residual =
        movingControlVolumeScalarTransient(phi, psi, rho, /*dim=*/3, /*enabled=*/true) +
        rho * dot(relative_advection, grad(phi)) * psi +
        FormExpr::constant(0.01) * dot(grad(phi), grad(psi));

    EXPECT_TRUE(containsExprType(residual, FormExprType::MeshVelocity));
}

TEST(MovingDomainPhysics, RelativeConvectiveVelocityAssemblesAsPhysicalMinusMeshVelocity)
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

    const auto ale_relative =
        relativeConvectiveVelocity(constantVector3(1.0, -0.25, 0.5), 3, true);
    const auto static_equivalent =
        relativeConvectiveVelocity(constantVector3(0.75, -0.125, 0.0), 3, false);

    const auto ale_integrand = rho * dot(ale_relative, grad(u)) * v;
    const auto static_integrand = rho * dot(static_equivalent, grad(u)) * v;

    std::vector<FE::Real> ale_solution = {0.0, 1.0, 1.0, 1.0};
    const auto mesh_velocity = constantVectorTetraCoefficients(0.25, -0.125, 0.5);
    ale_solution.insert(ale_solution.end(), mesh_velocity.begin(), mesh_velocity.end());

    const auto ale_residual = assembleMovingDomainScalarResidual(mesh,
                                                                 scalar_space,
                                                                 scalar_dof_map,
                                                                 &vector_space,
                                                                 &vector_dof_map,
                                                                 ale_integrand,
                                                                 ale_solution);

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

TEST(MovingDomainPhysics, MovingControlVolumeScalarTransientAssemblesKnownDivergenceTerm)
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
    const auto integrand =
        movingControlVolumeScalarTransient(u, v, FormExpr::constant(2.0), 3, true);

    std::vector<FE::Real> solution = constantScalarTetraCoefficients(3.0);
    const auto mesh_velocity = affineXVectorTetraCoefficients();
    solution.insert(solution.end(), mesh_velocity.begin(), mesh_velocity.end());

    const auto residual = assembleMovingDomainScalarResidual(mesh,
                                                             scalar_space,
                                                             scalar_dof_map,
                                                             &vector_space,
                                                             &vector_dof_map,
                                                             integrand,
                                                             solution);

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
    const auto relative_advection = relativeConvectiveVelocity(u, /*dim=*/3, /*ale_enabled=*/true);

    const auto momentum =
        rho * inner(dt(u) + grad(u) * relative_advection, v) +
        movingControlVolumeVectorTransient(u, v, rho, /*dim=*/3, /*enabled=*/true) +
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
