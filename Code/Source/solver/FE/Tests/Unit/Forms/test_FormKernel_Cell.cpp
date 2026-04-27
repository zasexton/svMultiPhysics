/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_FormKernel_Cell.cpp
 * @brief Unit tests for FE/Forms cell (dx) assembly
 */

#include <gtest/gtest.h>

#include "Assembly/AssemblyKernel.h"
#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Basis/NodeOrderingConventions.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Geometry/MappingFactory.h"
#include "Geometry/PushForward.h"
#include "Spaces/HCurlSpace.h"
#include "Spaces/HDivSpace.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

namespace {

[[nodiscard]] dofs::DofMap createSingleTetraDofMap(LocalIndex n_dofs)
{
    dofs::DofMap dof_map(1, static_cast<GlobalIndex>(n_dofs), n_dofs);
    std::vector<GlobalIndex> cell_dofs(static_cast<std::size_t>(n_dofs));
    for (LocalIndex i = 0; i < n_dofs; ++i) {
        cell_dofs[static_cast<std::size_t>(i)] = static_cast<GlobalIndex>(i);
    }
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(static_cast<GlobalIndex>(n_dofs));
    dof_map.setNumLocalDofs(static_cast<GlobalIndex>(n_dofs));
    dof_map.finalize();
    return dof_map;
}

[[nodiscard]] Real matrixInner(const basis::VectorJacobian& A,
                               const basis::VectorJacobian& B,
                               int value_dim,
                               int dim)
{
    Real sum = Real(0);
    for (int r = 0; r < value_dim; ++r) {
        for (int c = 0; c < dim; ++c) {
            sum += A(static_cast<std::size_t>(r), static_cast<std::size_t>(c)) *
                   B(static_cast<std::size_t>(r), static_cast<std::size_t>(c));
        }
    }
    return sum;
}

[[nodiscard]] basis::VectorJacobian sym3(const basis::VectorJacobian& A)
{
    basis::VectorJacobian out{};
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            out(r, c) = Real(0.5) * (A(r, c) + A(c, r));
        }
    }
    return out;
}

[[nodiscard]] assembly::DenseMatrixView assembleVectorGradientMatrix(
    const assembly::IMeshAccess& mesh,
    const spaces::FunctionSpace& space,
    const FormExpr& form)
{
    auto dof_map = createSingleTetraDofMap(static_cast<LocalIndex>(space.dofs_per_element()));

    FormCompiler compiler;
    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(static_cast<GlobalIndex>(space.dofs_per_element()));
    mat.zero();
    (void)assembler.assembleMatrix(mesh, space, space, kernel, mat);
    return mat;
}

[[nodiscard]] assembly::DenseMatrixView assembleVectorGradientMatrix(
    const spaces::FunctionSpace& space,
    const FormExpr& form)
{
    SingleTetraMeshAccess mesh;
    return assembleVectorGradientMatrix(mesh, space, form);
}

[[nodiscard]] std::shared_ptr<geometry::GeometryMapping> makeMappingForCell(
    const assembly::IMeshAccess& mesh,
    GlobalIndex cell_id)
{
    std::vector<std::array<Real, 3>> coords;
    mesh.getCellCoordinates(cell_id, coords);
    std::vector<math::Vector<Real, 3>> nodes(coords.size());
    for (std::size_t i = 0; i < coords.size(); ++i) {
        nodes[i] = math::Vector<Real, 3>{coords[i][0], coords[i][1], coords[i][2]};
    }
    geometry::MappingRequest request;
    request.element_type = mesh.getCellType(cell_id);
    request.geometry_order = mesh.getCellGeometryOrder(cell_id);
    request.use_affine = (request.geometry_order <= 1);
    return geometry::MappingFactory::create(request, nodes);
}

[[nodiscard]] std::shared_ptr<geometry::GeometryMapping> makeMappingForCellFrame(
    const assembly::IMeshAccess& mesh,
    GlobalIndex cell_id,
    assembly::CoordinateFrame frame)
{
    std::vector<std::array<Real, 3>> coords;
    mesh.getCellCoordinates(cell_id, frame, coords);
    std::vector<math::Vector<Real, 3>> nodes(coords.size());
    for (std::size_t i = 0; i < coords.size(); ++i) {
        nodes[i] = math::Vector<Real, 3>{coords[i][0], coords[i][1], coords[i][2]};
    }
    geometry::MappingRequest request;
    request.element_type = mesh.getCellType(cell_id);
    request.geometry_order = mesh.getCellGeometryOrder(cell_id);
    request.use_affine = (request.geometry_order <= 1);
    return geometry::MappingFactory::create(request, nodes);
}

[[nodiscard]] basis::VectorJacobian transformVectorBasisJacobianForSpace(
    const spaces::FunctionSpace& space,
    const math::Vector<Real, 3>& v_ref,
    const basis::VectorJacobian& jac_ref,
    const geometry::PushForward::PiolaVectorGradientGeometryData& data)
{
    if (space.continuity() == Continuity::H_div) {
        return geometry::PushForward::hdiv_vector_jacobian(v_ref, jac_ref, data);
    }
    if (space.continuity() == Continuity::H_curl) {
        return geometry::PushForward::hcurl_vector_jacobian(v_ref, jac_ref, data);
    }
    throw FEException("test helper only supports H(div)/H(curl) vector-basis spaces");
}

struct CurvedPiolaGradientCase {
    const char* name;
    ElementType geometry_type;
    ElementType space_type;
};

constexpr FieldId kCurvedFieldGradientField = 9101;

[[nodiscard]] std::vector<CurvedPiolaGradientCase> supportedCurvedVolumePiolaCases()
{
    return {
        {"Tetra10", ElementType::Tetra10, ElementType::Tetra4},
        {"Hex20", ElementType::Hex20, ElementType::Hex8},
        {"Hex27", ElementType::Hex27, ElementType::Hex8},
        {"Wedge15", ElementType::Wedge15, ElementType::Wedge6},
        {"Wedge18", ElementType::Wedge18, ElementType::Wedge6},
        {"Pyramid13", ElementType::Pyramid13, ElementType::Pyramid5},
        {"Pyramid14", ElementType::Pyramid14, ElementType::Pyramid5},
    };
}

[[nodiscard]] std::vector<CurvedPiolaGradientCase> lowerDimensionalCurvedPiolaCases()
{
    return {
        {"Triangle6", ElementType::Triangle6, ElementType::Triangle3},
        {"Quad8", ElementType::Quad8, ElementType::Quad4},
        {"Quad9", ElementType::Quad9, ElementType::Quad4},
    };
}

class MovingCurvedTetra10MeshAccess final : public assembly::IMeshAccess {
public:
    MovingCurvedTetra10MeshAccess()
    {
        const auto n_nodes = basis::NodeOrdering::num_nodes(ElementType::Tetra10);
        reference_nodes_.reserve(n_nodes);
        current_nodes_.reserve(n_nodes);
        cell_.reserve(n_nodes);
        for (std::size_t i = 0; i < n_nodes; ++i) {
            const auto xi = basis::NodeOrdering::get_node_coords(ElementType::Tetra10, i);
            const Real x = xi[0];
            const Real y = xi[1];
            const Real z = xi[2];
            const std::array<Real, 3> ref = {
                x + Real(0.030) * x * y + Real(0.018) * z * z,
                y + Real(0.024) * y * z + Real(0.014) * x * x,
                z + Real(0.020) * x * z + Real(0.016) * y * y,
            };
            reference_nodes_.push_back(ref);
            current_nodes_.push_back({
                ref[0] + Real(0.075) * ref[1] + Real(0.012) * ref[2] + Real(0.010) * ref[0] * ref[1],
                Real(1.060) * ref[1] - Real(0.020) * ref[0] + Real(0.010) * ref[2] * ref[2],
                Real(0.940) * ref[2] + Real(0.025) * ref[0] + Real(0.008) * ref[1] * ref[2],
            });
            cell_.push_back(static_cast<GlobalIndex>(i));
        }
    }

    void perturbCurrentGeometry()
    {
        for (auto& x : current_nodes_) {
            const Real x0 = x[0];
            const Real x1 = x[1];
            const Real x2 = x[2];
            x[0] += Real(0.010) + Real(0.009) * x1 + Real(0.003) * x2 * x2;
            x[1] += Real(0.006) * x0 - Real(0.004) * x2 + Real(0.002) * x0 * x1;
            x[2] += Real(0.008) * x0 + Real(0.003) * x1 * x2;
        }
        ++geometry_revision_;
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override { return geometry_revision_; }
    [[nodiscard]] std::uint64_t coordinateConfigurationKey() const override
    {
        return 0x23'00u + geometry_revision_;
    }
    [[nodiscard]] bool cellIdsAreDense() const override { return true; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex) const override { return ElementType::Tetra10; }
    [[nodiscard]] int getCellGeometryOrder(GlobalIndex) const override { return 2; }

    void getCellNodes(GlobalIndex, std::vector<GlobalIndex>& nodes) const override
    {
        nodes.assign(cell_.begin(), cell_.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return current_nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex, std::vector<std::array<Real, 3>>& coords) const override
    {
        coords.resize(cell_.size());
        for (std::size_t i = 0; i < cell_.size(); ++i) {
            coords[i] = current_nodes_.at(static_cast<std::size_t>(cell_[i]));
        }
    }

    [[nodiscard]] bool supportsCoordinateFrame(assembly::CoordinateFrame frame) const override
    {
        return frame == assembly::CoordinateFrame::Active ||
               frame == assembly::CoordinateFrame::Reference ||
               frame == assembly::CoordinateFrame::Current;
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            assembly::CoordinateFrame frame,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        if (frame == assembly::CoordinateFrame::Reference) {
            coords.resize(cell_.size());
            for (std::size_t i = 0; i < cell_.size(); ++i) {
                coords[i] = reference_nodes_.at(static_cast<std::size_t>(cell_[i]));
            }
            return;
        }
        getCellCoordinates(cell_id, coords);
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex, GlobalIndex) const override { return 0; }
    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex) const override { return -1; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override { callback(0); }

    void forEachBoundaryFace(int, std::function<void(GlobalIndex, GlobalIndex)>) const override {}
    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>) const override {}

private:
    std::vector<std::array<Real, 3>> reference_nodes_;
    std::vector<std::array<Real, 3>> current_nodes_;
    std::vector<GlobalIndex> cell_;
    std::uint64_t geometry_revision_{1};
};

[[nodiscard]] Real expectedCurvedVectorGradientEntry(
    const assembly::IMeshAccess& mesh,
    const spaces::FunctionSpace& space,
    GlobalIndex row,
    GlobalIndex col)
{
    const auto& basis = space.element().basis();
    const auto quad = space.element().quadrature();
    const auto mapping = makeMappingForCell(mesh, 0);
    EXPECT_FALSE(mapping->isAffine());

    std::vector<math::Vector<Real, 3>> values;
    std::vector<basis::VectorJacobian> jacobians;
    Real expected = Real(0);
    for (std::size_t q = 0; q < quad->num_points(); ++q) {
        const auto xi = quad->point(q);
        basis.evaluate_vector_values(xi, values);
        basis.evaluate_vector_jacobians(xi, jacobians);
        const auto data = geometry::PushForward::piola_vector_gradient_geometry_data(*mapping, xi);
        const auto grad_col = transformVectorBasisJacobianForSpace(
            space,
            values[static_cast<std::size_t>(col)],
            jacobians[static_cast<std::size_t>(col)],
            data);
        const auto grad_row = transformVectorBasisJacobianForSpace(
            space,
            values[static_cast<std::size_t>(row)],
            jacobians[static_cast<std::size_t>(row)],
            data);
        expected += quad->weight(q) * std::abs(data.determinant) *
                    matrixInner(grad_col, grad_row, space.value_dimension(), space.topological_dimension());
    }
    return expected;
}

[[nodiscard]] Real expectedCurvedVectorGradientEntryForFrame(
    const assembly::IMeshAccess& mesh,
    assembly::CoordinateFrame frame,
    const spaces::FunctionSpace& space,
    GlobalIndex row,
    GlobalIndex col)
{
    const auto& basis = space.element().basis();
    const auto quad = space.element().quadrature();
    const auto mapping = makeMappingForCellFrame(mesh, 0, frame);
    EXPECT_FALSE(mapping->isAffine());

    std::vector<math::Vector<Real, 3>> values;
    std::vector<basis::VectorJacobian> jacobians;
    Real expected = Real(0);
    for (std::size_t q = 0; q < quad->num_points(); ++q) {
        const auto xi = quad->point(q);
        basis.evaluate_vector_values(xi, values);
        basis.evaluate_vector_jacobians(xi, jacobians);
        const auto data = geometry::PushForward::piola_vector_gradient_geometry_data(*mapping, xi);
        const auto grad_col = transformVectorBasisJacobianForSpace(
            space,
            values[static_cast<std::size_t>(col)],
            jacobians[static_cast<std::size_t>(col)],
            data);
        const auto grad_row = transformVectorBasisJacobianForSpace(
            space,
            values[static_cast<std::size_t>(row)],
            jacobians[static_cast<std::size_t>(row)],
            data);
        expected += quad->weight(q) * std::abs(data.determinant) *
                    matrixInner(grad_col, grad_row, space.value_dimension(), space.topological_dimension());
    }
    return expected;
}

void expectCurvedVectorGradientMatrixMatchesFrame(
    const assembly::IMeshAccess& mesh,
    assembly::CoordinateFrame frame,
    const spaces::FunctionSpace& space,
    const assembly::DenseMatrixView& mat,
    Real tol)
{
    for (GlobalIndex i = 0; i < static_cast<GlobalIndex>(space.dofs_per_element()); ++i) {
        for (GlobalIndex j = 0; j < static_cast<GlobalIndex>(space.dofs_per_element()); ++j) {
            const Real expected = expectedCurvedVectorGradientEntryForFrame(mesh, frame, space, i, j);
            EXPECT_NEAR(mat.getMatrixEntry(i, j), expected, tol)
                << "i=" << i << ", j=" << j;
        }
    }
}

void expectDenseMatrixNear(const assembly::DenseMatrixView& actual,
                           const assembly::DenseMatrixView& expected,
                           Real tol)
{
    ASSERT_EQ(actual.numRows(), expected.numRows());
    ASSERT_EQ(actual.numCols(), expected.numCols());
    for (GlobalIndex i = 0; i < actual.numRows(); ++i) {
        for (GlobalIndex j = 0; j < actual.numCols(); ++j) {
            EXPECT_NEAR(actual.getMatrixEntry(i, j), expected.getMatrixEntry(i, j), tol)
                << "i=" << i << ", j=" << j;
        }
    }
}

[[nodiscard]] Real maxDenseMatrixAbsDifference(const assembly::DenseMatrixView& a,
                                               const assembly::DenseMatrixView& b)
{
    EXPECT_EQ(a.numRows(), b.numRows());
    EXPECT_EQ(a.numCols(), b.numCols());
    Real max_diff = Real(0);
    const auto rows = std::min(a.numRows(), b.numRows());
    const auto cols = std::min(a.numCols(), b.numCols());
    for (GlobalIndex i = 0; i < rows; ++i) {
        for (GlobalIndex j = 0; j < cols; ++j) {
            max_diff = std::max(max_diff, std::abs(a.getMatrixEntry(i, j) - b.getMatrixEntry(i, j)));
        }
    }
    return max_diff;
}

[[nodiscard]] assembly::DenseVectorView assembleCurvedFieldGradientResidual(
    const assembly::IMeshAccess& mesh,
    const spaces::FunctionSpace& space,
    const FormExpr& field_expr,
    const std::vector<Real>& field_coefficients)
{
    auto dof_map = createSingleTetraDofMap(static_cast<LocalIndex>(space.dofs_per_element()));

    const auto v = FormExpr::testFunction(space, "v");
    const auto form = inner(grad(field_expr), grad(v)).dx();

    FormCompiler compiler;
    auto ir = compiler.compileLinear(form);
    FormKernel kernel(std::move(ir));

    const std::array<assembly::FieldSolutionAccess, 1> field_access = {{
        assembly::FieldSolutionAccess{
            .field = kCurvedFieldGradientField,
            .space = &space,
            .dof_map = &dof_map,
            .dof_offset = 0,
        },
    }};

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setFieldSolutionAccess(field_access);
    assembler.setCurrentSolution(field_coefficients);

    assembly::DenseVectorView vec(static_cast<GlobalIndex>(space.dofs_per_element()));
    vec.zero();
    (void)assembler.assembleVector(mesh, space, kernel, vec);
    return vec;
}

void expectCurvedFieldGradientResidualMatchesBasisGradientMatrix(
    const assembly::IMeshAccess& mesh,
    const spaces::FunctionSpace& space,
    const FormExpr& field_expr,
    Real tol)
{
    std::vector<Real> coeffs(space.dofs_per_element());
    for (std::size_t i = 0; i < coeffs.size(); ++i) {
        coeffs[i] = Real(0.07) * static_cast<Real>(i + 1) -
                    Real(0.015) * static_cast<Real>((i % 3) + 1);
    }

    const auto vec = assembleCurvedFieldGradientResidual(mesh, space, field_expr, coeffs);
    for (GlobalIndex i = 0; i < static_cast<GlobalIndex>(space.dofs_per_element()); ++i) {
        Real expected = Real(0);
        for (GlobalIndex j = 0; j < static_cast<GlobalIndex>(space.dofs_per_element()); ++j) {
            expected += coeffs[static_cast<std::size_t>(j)] *
                        expectedCurvedVectorGradientEntry(mesh, space, i, j);
        }
        EXPECT_NEAR(vec.getVectorEntry(i), expected, tol) << "i=" << i;
    }
}

} // namespace

TEST(FormKernelCellTest, LinearDxIntegratesBasisFunctions)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = v.dx();

    auto ir = compiler.compileLinear(form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseVectorView vec(4);
    vec.zero();

    (void)assembler.assembleVector(mesh, space, kernel, vec);

    const Real V = 1.0 / 6.0;
    const Real expected = V / 4.0;

    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(vec.getVectorEntry(i), expected, 1e-12);
    }
}

TEST(FormKernelCellTest, LinearDxWithConstantScaling)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = (FormExpr::constant(2.0) * v).dx();

    auto ir = compiler.compileLinear(form);
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseVectorView vec(4);
    vec.zero();

    (void)assembler.assembleVector(mesh, space, kernel, vec);

    const Real V = 1.0 / 6.0;
    const Real expected = 2.0 * (V / 4.0);

    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(vec.getVectorEntry(i), expected, 1e-12);
    }
}

TEST(FormKernelCellTest, LinearFormKernelFastScalarDiffusionMatchesStiffnessMatrixAndVector)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = (FormExpr::constant(2.5) * inner(grad(u), grad(v))).dx();

    auto bilinear_ir = compiler.compileBilinear(form);
    LinearFormKernel kernel(std::move(bilinear_ir), std::nullopt, LinearKernelOutput::Both);
    EXPECT_TRUE(kernel.hasScalarDiffusionCellFastPathForTesting());
    EXPECT_TRUE(kernel.hasStateIndependentMatrix());

    std::vector<Real> U = {1.0, 2.0, 3.0, 4.0};
    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setCurrentSolution(U);

    assembly::DenseSystemView out(4);
    out.zero();
    (void)assembler.assembleBoth(mesh, space, space, kernel, out, out);

    assembly::StiffnessKernel reference_kernel(2.5);
    assembly::DenseMatrixView reference(4);
    reference.zero();
    (void)assembler.assembleMatrix(mesh, space, space, reference_kernel, reference);

    for (GlobalIndex i = 0; i < 4; ++i) {
        Real expected_vector = 0.0;
        for (GlobalIndex j = 0; j < 4; ++j) {
            EXPECT_NEAR(out.getMatrixEntry(i, j), reference.getMatrixEntry(i, j), 1e-12);
            expected_vector += reference.getMatrixEntry(i, j) * U[static_cast<std::size_t>(j)];
        }
        EXPECT_NEAR(out.getVectorEntry(i), expected_vector, 1e-12);
    }

    auto vector_ir = compiler.compileBilinear(form);
    LinearFormKernel vector_kernel(std::move(vector_ir), std::nullopt, LinearKernelOutput::VectorOnly);
    EXPECT_TRUE(vector_kernel.hasScalarDiffusionCellFastPathForTesting());

    assembly::DenseVectorView vec(4);
    vec.zero();
    (void)assembler.assembleVector(mesh, space, vector_kernel, vec);
    for (GlobalIndex i = 0; i < 4; ++i) {
        EXPECT_NEAR(vec.getVectorEntry(i), out.getVectorEntry(i), 1e-12);
    }
}

TEST(FormKernelCellTest, HDivVectorBasisGradInnerProductUsesAnalyticJacobians)
{
    spaces::HDivSpace space(ElementType::Tetra4, 0, BasisType::RaviartThomas);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto mat = assembleVectorGradientMatrix(space, inner(grad(u), grad(v)).dx());

    const auto& basis = space.element().basis();
    const auto quad = space.element().quadrature();
    ASSERT_NE(quad.get(), nullptr);

    std::vector<basis::VectorJacobian> jacobians;
    for (GlobalIndex i = 0; i < static_cast<GlobalIndex>(space.dofs_per_element()); ++i) {
        for (GlobalIndex j = 0; j < static_cast<GlobalIndex>(space.dofs_per_element()); ++j) {
            Real expected = Real(0);
            for (std::size_t q = 0; q < quad->num_points(); ++q) {
                basis.evaluate_vector_jacobians(quad->point(q), jacobians);
                ASSERT_EQ(jacobians.size(), space.dofs_per_element());
                expected += quad->weight(q) *
                    matrixInner(jacobians[static_cast<std::size_t>(j)],
                                jacobians[static_cast<std::size_t>(i)],
                                space.value_dimension(),
                                space.topological_dimension());
            }
            EXPECT_NEAR(mat.getMatrixEntry(i, j), expected, 1e-12)
                << "i=" << i << ", j=" << j;
        }
    }
}

TEST(FormKernelCellTest, CurvedHDivVectorBasisGradInnerProductUsesCurvedPiolaDerivatives)
{
    CurvedTetra10MeshAccess mesh;
    spaces::HDivSpace space(ElementType::Tetra4, 0, BasisType::RaviartThomas);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto mat = assembleVectorGradientMatrix(mesh, space, inner(grad(u), grad(v)).dx());

    for (GlobalIndex i = 0; i < static_cast<GlobalIndex>(space.dofs_per_element()); ++i) {
        for (GlobalIndex j = 0; j < static_cast<GlobalIndex>(space.dofs_per_element()); ++j) {
            const Real expected = expectedCurvedVectorGradientEntry(mesh, space, i, j);
            EXPECT_NEAR(mat.getMatrixEntry(i, j), expected, 1e-11)
                << "i=" << i << ", j=" << j;
        }
    }
}

TEST(FormKernelCellTest, CurvedVolumeHDivVectorBasisGradientsCoverAllEnabledGeometryFamilies)
{
    for (const auto& c : supportedCurvedVolumePiolaCases()) {
        SCOPED_TRACE(c.name);
        CurvedSingleCellMeshAccess mesh(c.geometry_type, c.name);
        spaces::HDivSpace space(c.space_type, 0, BasisType::RaviartThomas);

        const auto mapping = makeMappingForCell(mesh, 0);
        ASSERT_FALSE(mapping->isAffine());

        const auto u = FormExpr::trialFunction(space, "u");
        const auto v = FormExpr::testFunction(space, "v");
        const auto mat = assembleVectorGradientMatrix(mesh, space, inner(grad(u), grad(v)).dx());

        for (GlobalIndex i = 0; i < static_cast<GlobalIndex>(space.dofs_per_element()); ++i) {
            for (GlobalIndex j = 0; j < static_cast<GlobalIndex>(space.dofs_per_element()); ++j) {
                const Real expected = expectedCurvedVectorGradientEntry(mesh, space, i, j);
                EXPECT_NEAR(mat.getMatrixEntry(i, j), expected, 2e-10)
                    << "i=" << i << ", j=" << j;
            }
        }
    }
}

TEST(FormKernelCellTest, CurvedTetra10BDMHDivVectorBasisGradientsUseCurvedPiolaDerivatives)
{
    CurvedSingleCellMeshAccess mesh(ElementType::Tetra10, "Tetra10-BDM");
    spaces::HDivSpace space(ElementType::Tetra4, 1, BasisType::BDM);

    const auto mapping = makeMappingForCell(mesh, 0);
    ASSERT_FALSE(mapping->isAffine());

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto mat = assembleVectorGradientMatrix(mesh, space, inner(grad(u), grad(v)).dx());

    for (GlobalIndex i = 0; i < static_cast<GlobalIndex>(space.dofs_per_element()); ++i) {
        for (GlobalIndex j = 0; j < static_cast<GlobalIndex>(space.dofs_per_element()); ++j) {
            const Real expected = expectedCurvedVectorGradientEntry(mesh, space, i, j);
            EXPECT_NEAR(mat.getMatrixEntry(i, j), expected, 2e-10)
                << "i=" << i << ", j=" << j;
        }
    }
}

TEST(FormKernelCellTest, CurvedTetra10HigherOrderRTAndNedelecVectorBasisGradientsUseCurvedPiolaDerivatives)
{
    CurvedSingleCellMeshAccess mesh(ElementType::Tetra10, "Tetra10-higher-order-RT-Nedelec");

    {
        spaces::HDivSpace space(ElementType::Tetra4, 1, BasisType::RaviartThomas);
        const auto mapping = makeMappingForCell(mesh, 0);
        ASSERT_FALSE(mapping->isAffine());

        const auto u = FormExpr::trialFunction(space, "u");
        const auto v = FormExpr::testFunction(space, "v");
        const auto mat = assembleVectorGradientMatrix(mesh, space, inner(grad(u), grad(v)).dx());

        for (GlobalIndex i = 0; i < static_cast<GlobalIndex>(space.dofs_per_element()); ++i) {
            for (GlobalIndex j = 0; j < static_cast<GlobalIndex>(space.dofs_per_element()); ++j) {
                const Real expected = expectedCurvedVectorGradientEntry(mesh, space, i, j);
                EXPECT_NEAR(mat.getMatrixEntry(i, j), expected, 2e-10)
                    << "i=" << i << ", j=" << j;
            }
        }
    }

    {
        spaces::HCurlSpace space(ElementType::Tetra4, 1, BasisType::Nedelec);
        const auto mapping = makeMappingForCell(mesh, 0);
        ASSERT_FALSE(mapping->isAffine());

        const auto u = FormExpr::trialFunction(space, "u");
        const auto v = FormExpr::testFunction(space, "v");
        const auto mat = assembleVectorGradientMatrix(mesh, space, inner(grad(u), grad(v)).dx());

        for (GlobalIndex i = 0; i < static_cast<GlobalIndex>(space.dofs_per_element()); ++i) {
            for (GlobalIndex j = 0; j < static_cast<GlobalIndex>(space.dofs_per_element()); ++j) {
                const Real expected = expectedCurvedVectorGradientEntry(mesh, space, i, j);
                EXPECT_NEAR(mat.getMatrixEntry(i, j), expected, 2e-10)
                    << "i=" << i << ", j=" << j;
            }
        }
    }
}

TEST(FormKernelCellTest, CurvedVectorBasisGradientsUseActiveCurrentConfiguration)
{
    MovingCurvedTetra10MeshAccess mesh;

    auto check_space = [&](const spaces::FunctionSpace& space) {
        const auto u = FormExpr::trialFunction(space, "u");
        const auto v = FormExpr::testFunction(space, "v");
        const auto mat = assembleVectorGradientMatrix(mesh, space, inner(grad(u), grad(v)).dx());

        expectCurvedVectorGradientMatrixMatchesFrame(
            mesh, assembly::CoordinateFrame::Current, space, mat, 2e-10);

        Real max_reference_current_delta = Real(0);
        for (GlobalIndex i = 0; i < static_cast<GlobalIndex>(space.dofs_per_element()); ++i) {
            for (GlobalIndex j = 0; j < static_cast<GlobalIndex>(space.dofs_per_element()); ++j) {
                const Real current = expectedCurvedVectorGradientEntryForFrame(
                    mesh, assembly::CoordinateFrame::Current, space, i, j);
                const Real reference = expectedCurvedVectorGradientEntryForFrame(
                    mesh, assembly::CoordinateFrame::Reference, space, i, j);
                max_reference_current_delta =
                    std::max(max_reference_current_delta, std::abs(current - reference));
            }
        }
        EXPECT_GT(max_reference_current_delta, Real(1.0e-5));
    };

    spaces::HDivSpace hdiv_space(ElementType::Tetra4, 0, BasisType::RaviartThomas);
    check_space(hdiv_space);

    spaces::HCurlSpace hcurl_space(ElementType::Tetra4, 0, BasisType::Nedelec);
    check_space(hcurl_space);
}

TEST(FormKernelCellTest, ReusedAssemblerRefreshesCurvedVectorBasisGradientsAfterGeometryRevision)
{
    auto check_space = [&](const spaces::FunctionSpace& space) {
        MovingCurvedTetra10MeshAccess mesh;
        auto dof_map = createSingleTetraDofMap(static_cast<LocalIndex>(space.dofs_per_element()));

        FormCompiler compiler;
        const auto u = FormExpr::trialFunction(space, "u");
        const auto v = FormExpr::testFunction(space, "v");
        auto ir = compiler.compileBilinear(inner(grad(u), grad(v)).dx());
        FormKernel kernel(std::move(ir));

        assembly::StandardAssembler reused;
        reused.setDofMap(dof_map);

        assembly::DenseMatrixView before(static_cast<GlobalIndex>(space.dofs_per_element()));
        before.zero();
        (void)reused.assembleMatrix(mesh, space, space, kernel, before);

        const auto initial_revision = mesh.geometryRevision();
        mesh.perturbCurrentGeometry();
        ASSERT_GT(mesh.geometryRevision(), initial_revision);

        assembly::DenseMatrixView reused_after(static_cast<GlobalIndex>(space.dofs_per_element()));
        reused_after.zero();
        (void)reused.assembleMatrix(mesh, space, space, kernel, reused_after);

        FormCompiler fresh_compiler;
        const auto fresh_u = FormExpr::trialFunction(space, "u_fresh");
        const auto fresh_v = FormExpr::testFunction(space, "v_fresh");
        FormKernel fresh_kernel(
            fresh_compiler.compileBilinear(inner(grad(fresh_u), grad(fresh_v)).dx()));

        assembly::StandardAssembler fresh;
        fresh.setDofMap(dof_map);

        assembly::DenseMatrixView expected_after(static_cast<GlobalIndex>(space.dofs_per_element()));
        expected_after.zero();
        (void)fresh.assembleMatrix(mesh, space, space, fresh_kernel, expected_after);

        expectDenseMatrixNear(reused_after, expected_after, 2e-10);
        EXPECT_GT(maxDenseMatrixAbsDifference(before, expected_after), Real(1.0e-6));
    };

    spaces::HDivSpace hdiv_space(ElementType::Tetra4, 0, BasisType::RaviartThomas);
    check_space(hdiv_space);

    spaces::HCurlSpace hcurl_space(ElementType::Tetra4, 0, BasisType::Nedelec);
    check_space(hcurl_space);
}

TEST(FormKernelCellTest, CurvedVectorBasisFieldGradientPopulationUsesCurvedPiolaDerivatives)
{
    CurvedSingleCellMeshAccess mesh(ElementType::Tetra10, "Tetra10-field-gradient");

    {
        spaces::HDivSpace space(ElementType::Tetra4, 0, BasisType::RaviartThomas);
        const auto discrete = FormExpr::discreteField(kCurvedFieldGradientField, space, "hdiv_discrete");
        expectCurvedFieldGradientResidualMatchesBasisGradientMatrix(mesh, space, discrete, 2e-10);

        const auto state = FormExpr::stateField(kCurvedFieldGradientField, space, "hdiv_state");
        expectCurvedFieldGradientResidualMatchesBasisGradientMatrix(mesh, space, state, 2e-10);
    }

    {
        spaces::HCurlSpace space(ElementType::Tetra4, 0, BasisType::Nedelec);
        const auto discrete = FormExpr::discreteField(kCurvedFieldGradientField, space, "hcurl_discrete");
        expectCurvedFieldGradientResidualMatchesBasisGradientMatrix(mesh, space, discrete, 2e-10);

        const auto state = FormExpr::stateField(kCurvedFieldGradientField, space, "hcurl_state");
        expectCurvedFieldGradientResidualMatchesBasisGradientMatrix(mesh, space, state, 2e-10);
    }
}

TEST(FormKernelCellTest, CurvedHigherOrderVectorBasisFieldGradientPopulationUsesCurvedPiolaDerivatives)
{
    CurvedSingleCellMeshAccess mesh(ElementType::Tetra10, "Tetra10-higher-order-field-gradient");

    {
        spaces::HDivSpace space(ElementType::Tetra4, 1, BasisType::RaviartThomas);
        const auto discrete = FormExpr::discreteField(kCurvedFieldGradientField, space, "rt1_discrete");
        expectCurvedFieldGradientResidualMatchesBasisGradientMatrix(mesh, space, discrete, 3e-10);

        const auto state = FormExpr::stateField(kCurvedFieldGradientField, space, "rt1_state");
        expectCurvedFieldGradientResidualMatchesBasisGradientMatrix(mesh, space, state, 3e-10);
    }

    {
        spaces::HDivSpace space(ElementType::Tetra4, 1, BasisType::BDM);
        const auto discrete = FormExpr::discreteField(kCurvedFieldGradientField, space, "bdm1_discrete");
        expectCurvedFieldGradientResidualMatchesBasisGradientMatrix(mesh, space, discrete, 3e-10);

        const auto state = FormExpr::stateField(kCurvedFieldGradientField, space, "bdm1_state");
        expectCurvedFieldGradientResidualMatchesBasisGradientMatrix(mesh, space, state, 3e-10);
    }

    {
        spaces::HCurlSpace space(ElementType::Tetra4, 1, BasisType::Nedelec);
        const auto discrete = FormExpr::discreteField(kCurvedFieldGradientField, space, "nedelec1_discrete");
        expectCurvedFieldGradientResidualMatchesBasisGradientMatrix(mesh, space, discrete, 3e-10);

        const auto state = FormExpr::stateField(kCurvedFieldGradientField, space, "nedelec1_state");
        expectCurvedFieldGradientResidualMatchesBasisGradientMatrix(mesh, space, state, 3e-10);
    }
}

TEST(FormKernelCellTest, HCurlVectorBasisSymGradInnerProductUsesAnalyticJacobians)
{
    spaces::HCurlSpace space(ElementType::Tetra4, 0, BasisType::Nedelec);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto mat = assembleVectorGradientMatrix(space, inner(sym(grad(u)), sym(grad(v))).dx());

    const auto& basis = space.element().basis();
    const auto quad = space.element().quadrature();
    ASSERT_NE(quad.get(), nullptr);

    std::vector<basis::VectorJacobian> jacobians;
    basis.evaluate_vector_jacobians(quad->point(0), jacobians);
    ASSERT_EQ(jacobians.size(), space.dofs_per_element());

    for (GlobalIndex i = 0; i < static_cast<GlobalIndex>(space.dofs_per_element()); ++i) {
        for (GlobalIndex j = 0; j < static_cast<GlobalIndex>(space.dofs_per_element()); ++j) {
            Real expected = Real(0);
            for (std::size_t q = 0; q < quad->num_points(); ++q) {
                basis.evaluate_vector_jacobians(quad->point(q), jacobians);
                expected += quad->weight(q) *
                    matrixInner(sym3(jacobians[static_cast<std::size_t>(j)]),
                                sym3(jacobians[static_cast<std::size_t>(i)]),
                                space.value_dimension(),
                                space.topological_dimension());
            }
            EXPECT_NEAR(mat.getMatrixEntry(i, j), expected, 1e-12)
                << "i=" << i << ", j=" << j;
        }
    }
}

TEST(FormKernelCellTest, CurvedHCurlVectorBasisGradInnerProductUsesCurvedPiolaDerivatives)
{
    CurvedTetra10MeshAccess mesh;
    spaces::HCurlSpace space(ElementType::Tetra4, 0, BasisType::Nedelec);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto mat = assembleVectorGradientMatrix(mesh, space, inner(grad(u), grad(v)).dx());

    for (GlobalIndex i = 0; i < static_cast<GlobalIndex>(space.dofs_per_element()); ++i) {
        for (GlobalIndex j = 0; j < static_cast<GlobalIndex>(space.dofs_per_element()); ++j) {
            const Real expected = expectedCurvedVectorGradientEntry(mesh, space, i, j);
            EXPECT_NEAR(mat.getMatrixEntry(i, j), expected, 1e-11)
                << "i=" << i << ", j=" << j;
        }
    }
}

TEST(FormKernelCellTest, CurvedVolumeHCurlVectorBasisGradientsCoverAllEnabledGeometryFamilies)
{
    for (const auto& c : supportedCurvedVolumePiolaCases()) {
        SCOPED_TRACE(c.name);
        CurvedSingleCellMeshAccess mesh(c.geometry_type, c.name);
        spaces::HCurlSpace space(c.space_type, 0, BasisType::Nedelec);

        const auto mapping = makeMappingForCell(mesh, 0);
        ASSERT_FALSE(mapping->isAffine());

        const auto u = FormExpr::trialFunction(space, "u");
        const auto v = FormExpr::testFunction(space, "v");
        const auto mat = assembleVectorGradientMatrix(mesh, space, inner(grad(u), grad(v)).dx());

        for (GlobalIndex i = 0; i < static_cast<GlobalIndex>(space.dofs_per_element()); ++i) {
            for (GlobalIndex j = 0; j < static_cast<GlobalIndex>(space.dofs_per_element()); ++j) {
                const Real expected = expectedCurvedVectorGradientEntry(mesh, space, i, j);
                EXPECT_NEAR(mat.getMatrixEntry(i, j), expected, 2e-10)
                    << "i=" << i << ", j=" << j;
            }
        }
    }
}

TEST(FormKernelCellTest, LowerDimensionalCurvedPiolaVectorBasisGradientsFailClosed)
{
    for (const auto& c : lowerDimensionalCurvedPiolaCases()) {
        SCOPED_TRACE(c.name);
        CurvedSingleCellMeshAccess mesh(c.geometry_type, c.name);

        {
            spaces::HDivSpace space(c.space_type, 0, BasisType::RaviartThomas);
            const auto u = FormExpr::trialFunction(space, "u");
            const auto v = FormExpr::testFunction(space, "v");
            EXPECT_THROW((void)assembleVectorGradientMatrix(mesh, space, inner(grad(u), grad(v)).dx()),
                         FEException);
        }

        {
            spaces::HCurlSpace space(c.space_type, 0, BasisType::Nedelec);
            const auto u = FormExpr::trialFunction(space, "u");
            const auto v = FormExpr::testFunction(space, "v");
            EXPECT_THROW((void)assembleVectorGradientMatrix(mesh, space, inner(grad(u), grad(v)).dx()),
                         FEException);
        }
    }
}

TEST(FormKernelCellTest, LowerDimensionalCurvedBDMVectorBasisGradientsFailClosed)
{
    for (const auto& c : lowerDimensionalCurvedPiolaCases()) {
        SCOPED_TRACE(c.name);
        CurvedSingleCellMeshAccess mesh(c.geometry_type, c.name);
        spaces::HDivSpace space(c.space_type, 1, BasisType::BDM);
        const auto u = FormExpr::trialFunction(space, "u");
        const auto v = FormExpr::testFunction(space, "v");
        EXPECT_THROW((void)assembleVectorGradientMatrix(mesh, space, inner(grad(u), grad(v)).dx()),
                     FEException);
    }
}

TEST(FormKernelCellTest, CurvedLinePiolaVectorBasisGradientFormsAreNotAdvertised)
{
    EXPECT_THROW((void)spaces::HDivSpace(ElementType::Line2, 0, BasisType::RaviartThomas),
                 FEException);
    EXPECT_THROW((void)spaces::HCurlSpace(ElementType::Line2, 0, BasisType::Nedelec),
                 FEException);
}

TEST(FormKernelCellTest, DtBilinearRequiresTransientContextAndSignalsTemporalOrder)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FormCompiler compiler;
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto form = (dt(u, 2) * v).dx();

    auto ir = compiler.compileBilinear(form);
    FormKernel kernel(std::move(ir));
    EXPECT_EQ(kernel.maxTemporalDerivativeOrder(), 2);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(4);
    mat.zero();

    try {
        (void)assembler.assembleMatrix(mesh, space, space, kernel, mat);
        FAIL() << "Expected assembly to fail without a transient time-integration context";
    } catch (const svmp::FE::FEException& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("dt(...) operator requires a transient time-integration context"), std::string::npos);
    }
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
