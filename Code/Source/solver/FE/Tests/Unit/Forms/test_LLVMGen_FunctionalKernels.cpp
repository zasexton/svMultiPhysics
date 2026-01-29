/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/FunctionalAssembler.h"
#include "Forms/FormExpr.h"
#include "Forms/JIT/JITFunctionalKernelWrapper.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"

#include <array>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {
namespace {

class SingleTetraDomainMeshAccess final : public assembly::IMeshAccess {
public:
    explicit SingleTetraDomainMeshAccess(int domain_id)
        : domain_id_(domain_id)
    {
        nodes_ = {
            {0.0, 0.0, 0.0},  // 0
            {1.0, 0.0, 0.0},  // 1
            {0.0, 1.0, 0.0},  // 2
            {0.0, 0.0, 1.0}   // 3
        };
        cell_ = {0, 1, 2, 3};
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Tetra4; }

    [[nodiscard]] int getCellDomainId(GlobalIndex /*cell_id*/) const override { return domain_id_; }

    void getCellNodes(GlobalIndex /*cell_id*/, std::vector<GlobalIndex>& nodes) const override
    {
        nodes.assign(cell_.begin(), cell_.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex /*cell_id*/, std::vector<std::array<Real, 3>>& coords) const override
    {
        coords.resize(cell_.size());
        for (std::size_t i = 0; i < cell_.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(cell_[i]));
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/,
                                               GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override { return -1; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }

    void forEachBoundaryFace(int /*marker*/,
                             std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

private:
    int domain_id_{0};
    std::vector<std::array<Real, 3>> nodes_{};
    std::array<GlobalIndex, 4> cell_{};
};

} // namespace

TEST(LLVMGenFunctionalKernels, ConstantMatchesInterpreter)
{
    requireLLVMJITOrSkip();

    SingleTetraMeshAccess mesh;
    const auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto integrand = FormExpr::constant(Real(2.5));
    auto interp = makeFunctionalFormKernel(integrand, FunctionalFormKernel::Domain::Cell);

    auto throw_fallback = makeThrowingTotalKernelFor(integrand, /*has_cell=*/true, /*has_boundary=*/false);
    jit::JITFunctionalKernelWrapper jit_kernel(
        throw_fallback, integrand, jit::JITFunctionalKernelWrapper::Domain::Cell, makeUnitTestJITOptions());

    assembly::FunctionalAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setDofMap(dof_map);
    assembler.setSpace(space);

    const Real expected = assembler.assembleScalar(*interp);
    const Real actual = assembler.assembleScalar(jit_kernel);
    EXPECT_NEAR(actual, expected, 1e-12);
}

TEST(LLVMGenFunctionalKernels, ParameterRefLoadsFromJITConstants)
{
    requireLLVMJITOrSkip();

    SingleTetraMeshAccess mesh;
    const auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto integrand = FormExpr::parameterRef(1);
    auto interp = makeFunctionalFormKernel(integrand, FunctionalFormKernel::Domain::Cell);

    auto throw_fallback = makeThrowingTotalKernelFor(integrand, /*has_cell=*/true, /*has_boundary=*/false);
    jit::JITFunctionalKernelWrapper jit_kernel(
        throw_fallback, integrand, jit::JITFunctionalKernelWrapper::Domain::Cell, makeUnitTestJITOptions());

    const std::array<Real, 3> jit_constants = {Real(1.25), Real(3.75), Real(-2.0)};

    assembly::FunctionalAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setDofMap(dof_map);
    assembler.setSpace(space);
    assembler.setJITConstants(jit_constants);

    const Real expected = assembler.assembleScalar(*interp);
    const Real actual = assembler.assembleScalar(jit_kernel);
    EXPECT_NEAR(actual, expected, 1e-12);
}

TEST(LLVMGenFunctionalKernels, CoupledRefsLoadFromCoupledArrays)
{
    requireLLVMJITOrSkip();

    SingleTetraMeshAccess mesh;
    const auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto integrand = FormExpr::boundaryIntegralRef(0) + FormExpr::auxiliaryStateRef(1);
    auto interp = makeFunctionalFormKernel(integrand, FunctionalFormKernel::Domain::Cell);

    auto throw_fallback = makeThrowingTotalKernelFor(integrand, /*has_cell=*/true, /*has_boundary=*/false);
    jit::JITFunctionalKernelWrapper jit_kernel(
        throw_fallback, integrand, jit::JITFunctionalKernelWrapper::Domain::Cell, makeUnitTestJITOptions());

    const std::array<Real, 2> coupled_integrals = {Real(2.0), Real(9.0)};
    const std::array<Real, 2> coupled_aux = {Real(10.0), Real(-3.0)};

    assembly::FunctionalAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setDofMap(dof_map);
    assembler.setSpace(space);
    assembler.setCoupledValues(coupled_integrals, coupled_aux);

    const Real expected = assembler.assembleScalar(*interp);
    const Real actual = assembler.assembleScalar(jit_kernel);
    EXPECT_NEAR(actual, expected, 1e-12);
}

TEST(LLVMGenFunctionalKernels, TimeAndTimeStepLoadFromSideArgs)
{
    requireLLVMJITOrSkip();

    SingleTetraMeshAccess mesh;
    const auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto integrand = FormExpr::time() + FormExpr::timeStep();
    auto interp = makeFunctionalFormKernel(integrand, FunctionalFormKernel::Domain::Cell);

    auto throw_fallback = makeThrowingTotalKernelFor(integrand, /*has_cell=*/true, /*has_boundary=*/false);
    jit::JITFunctionalKernelWrapper jit_kernel(
        throw_fallback, integrand, jit::JITFunctionalKernelWrapper::Domain::Cell, makeUnitTestJITOptions());

    assembly::FunctionalAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setDofMap(dof_map);
    assembler.setSpace(space);
    assembler.setTime(Real(1.5));
    assembler.setTimeStep(Real(0.125));

    const Real expected = assembler.assembleScalar(*interp);
    const Real actual = assembler.assembleScalar(jit_kernel);
    EXPECT_NEAR(actual, expected, 1e-12);
}

TEST(LLVMGenFunctionalKernels, EntityMeasuresLoadFromSideArgs)
{
    requireLLVMJITOrSkip();

    SingleTetraMeshAccess mesh;
    const auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto integrand = FormExpr::cellDiameter() + FormExpr::cellVolume();
    auto interp = makeFunctionalFormKernel(integrand, FunctionalFormKernel::Domain::Cell);

    auto throw_fallback = makeThrowingTotalKernelFor(integrand, /*has_cell=*/true, /*has_boundary=*/false);
    jit::JITFunctionalKernelWrapper jit_kernel(
        throw_fallback, integrand, jit::JITFunctionalKernelWrapper::Domain::Cell, makeUnitTestJITOptions());

    assembly::FunctionalAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setDofMap(dof_map);
    assembler.setSpace(space);

    const Real expected = assembler.assembleScalar(*interp);
    const Real actual = assembler.assembleScalar(jit_kernel);
    EXPECT_NEAR(actual, expected, 1e-12);
}

TEST(LLVMGenFunctionalKernels, FacetAreaLoadsOnBoundaryFace)
{
    requireLLVMJITOrSkip();

    constexpr int boundary_marker = 2;
    SingleTetraOneBoundaryFaceMeshAccess mesh(boundary_marker);
    const auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto integrand = FormExpr::facetArea();
    auto interp = makeFunctionalFormKernel(integrand, FunctionalFormKernel::Domain::BoundaryFace);

    auto throw_fallback = makeThrowingTotalKernelFor(integrand, /*has_cell=*/false, /*has_boundary=*/true);
    jit::JITFunctionalKernelWrapper jit_kernel(
        throw_fallback, integrand, jit::JITFunctionalKernelWrapper::Domain::BoundaryFace, makeUnitTestJITOptions());

    assembly::FunctionalAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setDofMap(dof_map);
    assembler.setSpace(space);

    const Real expected = assembler.assembleBoundaryScalar(*interp, boundary_marker);
    const Real actual = assembler.assembleBoundaryScalar(jit_kernel, boundary_marker);
    EXPECT_NEAR(actual, expected, 1e-12);
}

TEST(LLVMGenFunctionalKernels, CellDomainIdLoadsFromMesh)
{
    requireLLVMJITOrSkip();

    SingleTetraDomainMeshAccess mesh(/*domain_id=*/7);
    const auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto integrand = FormExpr::cellDomainId();
    auto interp = makeFunctionalFormKernel(integrand, FunctionalFormKernel::Domain::Cell);

    auto throw_fallback = makeThrowingTotalKernelFor(integrand, /*has_cell=*/true, /*has_boundary=*/false);
    jit::JITFunctionalKernelWrapper jit_kernel(
        throw_fallback, integrand, jit::JITFunctionalKernelWrapper::Domain::Cell, makeUnitTestJITOptions());

    assembly::FunctionalAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setDofMap(dof_map);
    assembler.setSpace(space);

    const Real expected = assembler.assembleScalar(*interp);
    const Real actual = assembler.assembleScalar(jit_kernel);
    EXPECT_NEAR(actual, expected, 1e-12);
}

TEST(LLVMGenFunctionalKernels, CoordinatesLoadPhysicalAndReference)
{
    requireLLVMJITOrSkip();

    SingleTetraMeshAccess mesh;
    const auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto integrand = FormExpr::coordinate().component(0) + FormExpr::referenceCoordinate().component(1);
    auto interp = makeFunctionalFormKernel(integrand, FunctionalFormKernel::Domain::Cell);

    auto throw_fallback = makeThrowingTotalKernelFor(integrand, /*has_cell=*/true, /*has_boundary=*/false);
    jit::JITFunctionalKernelWrapper jit_kernel(
        throw_fallback, integrand, jit::JITFunctionalKernelWrapper::Domain::Cell, makeUnitTestJITOptions());

    assembly::FunctionalAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setDofMap(dof_map);
    assembler.setSpace(space);

    const Real expected = assembler.assembleScalar(*interp);
    const Real actual = assembler.assembleScalar(jit_kernel);
    EXPECT_NEAR(actual, expected, 1e-12);
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
