/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_FunctionalAssembler.cpp
 * @brief Unit tests for FunctionalAssembler and scalar functional assembly
 */

#include <gtest/gtest.h>

#include "Assembly/FunctionalAssembler.h"
#include "Assembly/AssemblyContext.h"

#include "Forms/BoundaryFunctional.h"
#include "Forms/FormExpr.h"
#include "Spaces/H1Space.h"
#include "Spaces/HDivSpace.h"
#include "Spaces/ProductSpace.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Assembly/MeshAccess.h"
#include "Tests/Unit/Systems/TraceMultiplierTestMeshes.h"
#endif

#include <array>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

namespace svmp {
namespace FE {
namespace assembly {
namespace test {

namespace {

class PhysicalHessianError final : public FunctionalKernel {
public:
    FieldId field{INVALID_FIELD_ID};
    int component{-1};
    Real expected_xx{0.0};

    RequiredData getRequiredData() const noexcept override {
        return RequiredData::SolutionHessians | RequiredData::SolutionLaplacians;
    }
    std::vector<FieldRequirement> fieldRequirements() const override {
        if (field == INVALID_FIELD_ID) return {};
        return {{field, RequiredData::SolutionHessians | RequiredData::SolutionLaplacians}};
    }
    bool hasBoundaryFace() const noexcept override { return true; }
    Real evaluateCell(const AssemblyContext& ctx, LocalIndex q) override {
        const auto H = field == INVALID_FIELD_ID ? ctx.solutionHessian(q) :
            component < 0 ? ctx.fieldHessian(field, q) : ctx.fieldComponentHessian(field, q, component);
        const Real lap = field == INVALID_FIELD_ID ? ctx.solutionLaplacian(q) :
            component < 0 ? ctx.fieldLaplacian(field, q) : ctx.fieldComponentLaplacian(field, q, component);
        Real error = (lap - expected_xx) * (lap - expected_xx);
        for (int i = 0; i < ctx.dimension(); ++i) {
            for (int j = 0; j < ctx.dimension(); ++j) {
                const Real delta = H[i][j] - (i == 0 && j == 0 ? expected_xx : 0.0);
                error += delta * delta;
            }
        }
        return error;
    }
    Real evaluateBoundaryFace(const AssemblyContext& ctx, LocalIndex q, int) override {
        return evaluateCell(ctx, q);
    }
};

dofs::DofMap singleCellDofs(LocalIndex n)
{
    dofs::DofMap map(1, n, n);
    std::vector<GlobalIndex> ids(n);
    std::iota(ids.begin(), ids.end(), 0);
    map.setCellDofs(0, ids);
    map.finalize();
    return map;
}

class BoundaryMeasureKernel final : public FunctionalKernel {
public:
    [[nodiscard]] RequiredData getRequiredData() const noexcept override
    {
        return RequiredData::IntegrationWeights;
    }

    [[nodiscard]] bool hasCell() const noexcept override { return false; }
    [[nodiscard]] bool hasBoundaryFace() const noexcept override { return true; }

    [[nodiscard]] Real evaluateCell(const AssemblyContext& /*ctx*/, LocalIndex /*q*/) override { return 0.0; }

    [[nodiscard]] Real evaluateBoundaryFace(const AssemblyContext& /*ctx*/,
                                           LocalIndex /*q*/,
                                           int /*boundary_marker*/) override
    {
        return 1.0;
    }

    [[nodiscard]] std::string name() const override { return "BoundaryMeasureKernel"; }
};

// Sums the raw local coefficients of the primary field: the only solution data
// it asks for, which needs no basis tables.
class CoefficientSumKernel final : public FunctionalKernel {
public:
    [[nodiscard]] RequiredData getRequiredData() const noexcept override
    {
        return RequiredData::IntegrationWeights | RequiredData::SolutionCoefficients;
    }

    [[nodiscard]] bool hasCell() const noexcept override { return true; }
    [[nodiscard]] bool hasBoundaryFace() const noexcept override { return true; }

    [[nodiscard]] Real evaluateCell(const AssemblyContext& ctx, LocalIndex /*q*/) override
    {
        return sum(ctx);
    }

    [[nodiscard]] Real evaluateBoundaryFace(const AssemblyContext& ctx,
                                           LocalIndex /*q*/,
                                           int /*boundary_marker*/) override
    {
        return sum(ctx);
    }

    [[nodiscard]] std::string name() const override { return "CoefficientSumKernel"; }

private:
    [[nodiscard]] static Real sum(const AssemblyContext& ctx)
    {
        Real total = 0.0;
        for (const Real c : ctx.solutionCoefficients()) {
            total += c;
        }
        return total;
    }
};

} // namespace

// ============================================================================
// FunctionalKernel Tests
// ============================================================================

TEST(FunctionalKernelTest, L2NormKernelRequiredData) {
    L2NormKernel kernel;
    auto required = kernel.getRequiredData();

    EXPECT_TRUE(hasFlag(required, RequiredData::IntegrationWeights));
    EXPECT_TRUE(hasFlag(required, RequiredData::SolutionValues));
}

TEST(FunctionalKernelTest, L2NormKernelProperties) {
    L2NormKernel kernel;

    EXPECT_TRUE(kernel.requiresSquareRoot());
    EXPECT_EQ(kernel.name(), "L2Norm");
    EXPECT_TRUE(kernel.hasCell());
    EXPECT_FALSE(kernel.hasBoundaryFace());
}

TEST(FunctionalKernelTest, L2NormPostProcess) {
    L2NormKernel kernel;

    // Post-processing should take square root
    EXPECT_NEAR(kernel.postProcess(4.0), 2.0, 1e-12);
    EXPECT_NEAR(kernel.postProcess(9.0), 3.0, 1e-12);
}

TEST(FunctionalKernelTest, H1SeminormKernelProperties) {
    H1SeminormKernel kernel;

    EXPECT_TRUE(kernel.requiresSquareRoot());
    EXPECT_EQ(kernel.name(), "H1Seminorm");
    EXPECT_TRUE(hasFlag(kernel.getRequiredData(), RequiredData::SolutionGradients));
}

TEST(FunctionalKernelTest, EnergyKernelProperties) {
    EnergyKernel kernel;

    EXPECT_FALSE(kernel.requiresSquareRoot());
    EXPECT_EQ(kernel.name(), "Energy");
}

TEST(FunctionalKernelTest, VolumeKernelProperties) {
    VolumeKernel kernel;

    EXPECT_FALSE(kernel.requiresSquareRoot());
    EXPECT_TRUE(kernel.isLinear());
    EXPECT_EQ(kernel.name(), "Volume");
}

TEST(FunctionalKernelTest, BoundaryFluxKernelProperties) {
    BoundaryFluxKernel kernel;

    EXPECT_FALSE(kernel.hasCell());
    EXPECT_TRUE(kernel.hasBoundaryFace());
    EXPECT_EQ(kernel.name(), "BoundaryFlux");
}

// ============================================================================
// L2ErrorKernel Tests
// ============================================================================

TEST(L2ErrorKernelTest, Construction) {
    auto exact = [](Real /*x*/, Real /*y*/, Real /*z*/) { return 1.0; };
    L2ErrorKernel kernel(exact);

    EXPECT_TRUE(kernel.requiresSquareRoot());
    EXPECT_EQ(kernel.name(), "L2Error");
}

TEST(L2ErrorKernelTest, RequiredData) {
    auto exact = [](Real /*x*/, Real /*y*/, Real /*z*/) { return 1.0; };
    L2ErrorKernel kernel(exact);

    auto required = kernel.getRequiredData();
    EXPECT_TRUE(hasFlag(required, RequiredData::SolutionValues));
    EXPECT_TRUE(hasFlag(required, RequiredData::PhysicalPoints));
}

// ============================================================================
// H1ErrorKernel Tests
// ============================================================================

TEST(H1ErrorKernelTest, Construction) {
    auto exact_grad = [](Real /*x*/, Real /*y*/, Real /*z*/) {
        return std::array<Real, 3>{1.0, 0.0, 0.0};
    };
    H1ErrorKernel kernel(exact_grad);

    EXPECT_TRUE(kernel.requiresSquareRoot());
    EXPECT_EQ(kernel.name(), "H1Error");
}

// ============================================================================
// FunctionalAssemblyOptions Tests
// ============================================================================

TEST(FunctionalAssemblyOptionsTest, Defaults) {
    FunctionalAssemblyOptions options;

    EXPECT_EQ(options.num_threads, 1);
    EXPECT_TRUE(options.deterministic);
    EXPECT_FALSE(options.use_kahan_summation);
    EXPECT_FALSE(options.verbose);
}

// ============================================================================
// FunctionalResult Tests
// ============================================================================

TEST(FunctionalResultTest, DefaultConstruction) {
    FunctionalResult result;

    EXPECT_DOUBLE_EQ(result.value, 0.0);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.elements_processed, 0);
    EXPECT_EQ(result.faces_processed, 0);
}

TEST(FunctionalResultTest, Conversion) {
    FunctionalResult result;
    result.value = 3.14;
    result.success = true;

    // Conversion to Real
    Real val = result;
    EXPECT_DOUBLE_EQ(val, 3.14);

    // Conversion to bool
    bool ok = result;
    EXPECT_TRUE(ok);
}

// ============================================================================
// FunctionalAssembler Tests
// ============================================================================

TEST(FunctionalAssemblerTest, DefaultConstruction) {
    FunctionalAssembler assembler;

    // Not configured yet
    EXPECT_FALSE(assembler.isConfigured());
}

TEST(FunctionalAssemblerTest, ConstructionWithOptions) {
    FunctionalAssemblyOptions options;
    options.num_threads = 4;
    options.use_kahan_summation = true;

    FunctionalAssembler assembler(options);
    const auto& opts = assembler.getOptions();

    EXPECT_EQ(opts.num_threads, 4);
    EXPECT_TRUE(opts.use_kahan_summation);
}

TEST(FunctionalAssemblerTest, AssembleScalarOverCellsTreatsEmptySubsetAsZero)
{
    forms::test::SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);
    auto dof_map = forms::test::createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FunctionalAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setDofMap(dof_map);
    assembler.setSpace(space);

    VolumeKernel kernel;
    const std::array<GlobalIndex, 1> one_cell{0};
    EXPECT_NEAR(assembler.assembleScalarOverCells(kernel, one_cell), 1.0 / 6.0, 1e-12);
    EXPECT_NEAR(assembler.assembleScalarOverCells(kernel, std::span<const GlobalIndex>{}),
                0.0, 1e-12);
}

TEST(FunctionalAssemblerTest, SetOptions) {
    FunctionalAssembler assembler;

    FunctionalAssemblyOptions options;
    options.deterministic = false;
    assembler.setOptions(options);

    EXPECT_FALSE(assembler.getOptions().deterministic);
}

// ============================================================================
// Factory Tests
// ============================================================================

TEST(FunctionalAssemblerFactoryTest, CreateDefault) {
    auto assembler = createFunctionalAssembler();
    EXPECT_NE(assembler, nullptr);
}

TEST(FunctionalAssemblerFactoryTest, CreateWithOptions) {
    FunctionalAssemblyOptions options;
    options.num_threads = 8;

    auto assembler = createFunctionalAssembler(options);
    EXPECT_NE(assembler, nullptr);
    EXPECT_EQ(assembler->getOptions().num_threads, 8);
}

// ============================================================================
// KahanAccumulator Tests (via FunctionalAssembler behavior)
// ============================================================================

TEST(KahanSummationTest, Precision) {
    // Test that Kahan summation provides better precision
    // by accumulating many small values

    FunctionalAssemblyOptions options_standard;
    options_standard.use_kahan_summation = false;

    FunctionalAssemblyOptions options_kahan;
    options_kahan.use_kahan_summation = true;

    // Both options should produce valid assemblers
    auto assembler_std = createFunctionalAssembler(options_standard);
    auto assembler_kahan = createFunctionalAssembler(options_kahan);

    EXPECT_NE(assembler_std, nullptr);
    EXPECT_NE(assembler_kahan, nullptr);
}

// ============================================================================
// Boundary Functional Assembly (Regression/Integration)
// ============================================================================

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
TEST(FunctionalAssemblerTest, CurvedQuadPhysicalQuadraticsOnCellsAndFaces)
{
    // F(r,s) = (r, s + 2r²). Each of y + k x² is exactly representable in Q2.
    const auto mesh = test_trace_multiplier::buildBentQuadraticQuad();
    MeshAccess access(*mesh);
    auto scalar = std::make_shared<spaces::H1Space>(ElementType::Quad9, 2);
    spaces::ProductSpace vector(scalar, 2);
    constexpr int n = 9;
    auto map = singleCellDofs(4 * n);
    std::vector<std::array<Real, 3>> nodes;
    access.getCellCoordinates(0, nodes);
    ASSERT_EQ(nodes.size(), n);
    std::vector<Real> coefficients(4 * n);
    for (int k = 0; k < 4; ++k) {
        for (int i = 0; i < n; ++i) {
            coefficients[k * n + i] = nodes[i][1] + k * nodes[i][0] * nodes[i][0];
        }
    }
    FunctionalAssembler a;
    a.setMesh(access);
    a.setDofMap(map);
    a.setSpace(*scalar);
    a.setPrimaryField(0);
    a.setDofPerNode(0);
    a.registerFieldBinding({.field=0, .space=scalar.get(), .field_type=FieldType::Scalar,
                            .value_dimension=1, .component_offset=0, .n_components=1});
    a.registerFieldBinding({.field=1, .space=scalar.get(), .field_type=FieldType::Scalar,
                            .value_dimension=1, .component_offset=n, .n_components=1});
    a.registerFieldBinding({.field=2, .space=&vector, .field_type=FieldType::Vector,
                            .value_dimension=2, .component_offset=2*n, .n_components=2});
    a.setSolution(coefficients);
    for (int k = 0; k < 4; ++k) {
        SCOPED_TRACE(k);
        PhysicalHessianError error;
        error.field = k < 2 ? static_cast<FieldId>(k) : 2;
        error.component = k < 2 ? -1 : k - 2;
        error.expected_xx = 2.0 * k;
        EXPECT_NEAR(a.assembleScalar(error), 0.0, 1e-23);
        EXPECT_NEAR(a.assembleBoundaryScalar(error, test_trace_multiplier::kWholeBoundary), 0.0, 1e-23);
        const std::array<FunctionalKernel*, 1> kernels{&error};
        EXPECT_NEAR(a.assembleMultiple(kernels).at(0), 0.0, 1e-23);
    }
}

TEST(FunctionalAssemblerBoundaryTest, CurvedQuad9FluxAndPressureForceUseOutwardNormals)
{
    const auto mesh = test_trace_multiplier::buildBentQuadraticQuad();
    MeshAccess access(*mesh);
    spaces::H1Space space(ElementType::Quad9, 2);
    dofs::DofMap dof_map(1, 9, 9);
    const std::array<GlobalIndex, 9> ids{0, 1, 2, 3, 4, 5, 6, 7, 8};
    dof_map.setCellDofs(0, ids);
    dof_map.finalize();
    FunctionalAssembler assembler;
    assembler.setMesh(access);
    assembler.setDofMap(dof_map);
    assembler.setSpace(space);
    const auto n_y = forms::FormExpr::normal().component(1);
    const auto y = forms::FormExpr::coordinate().component(1);
    const auto integral = [&](const forms::FormExpr& integrand) {
        constexpr auto marker = test_trace_multiplier::kWholeBoundary;
        auto kernel = forms::compileBoundaryFunctionalKernel(integrand, marker);
        return assembler.assembleBoundaryScalar(*kernel, marker);
    };
    EXPECT_NEAR(integral(n_y), 0.0, 1e-12);
    EXPECT_NEAR(integral(-y * n_y), -4.0, 1e-12);
}
#endif

TEST(FunctionalAssemblerTest, PhysicalCoordinatesHaveZeroHessianOnCurvedTetra)
{
    forms::test::CurvedTetra10MeshAccess mesh;
    spaces::H1Space space(ElementType::Tetra10, 2);
    auto map = singleCellDofs(10);
    FunctionalAssembler a;
    a.setMesh(mesh);
    a.setDofMap(map);
    a.setSpace(space);
    for (int d = 0; d < 3; ++d) {
        std::array<Real, 10> coordinates{};
        for (int i = 0; i < 10; ++i) coordinates[i] = mesh.getNodeCoordinates(i)[d];
        a.setSolution(coordinates);
        PhysicalHessianError error;
        EXPECT_NEAR(a.assembleScalar(error), 0.0, 1e-25) << d;
    }
}

TEST(FunctionalAssemblerBoundaryTest, IntegrateConstantOverBoundaryFace)
{
    forms::test::SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);
    auto dof_map = forms::test::createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FunctionalAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setDofMap(dof_map);
    assembler.setSpace(space);

    BoundaryMeasureKernel kernel;
    const Real area = assembler.assembleBoundaryScalar(kernel, /*boundary_marker=*/2);
    EXPECT_NEAR(area, 0.5, 1e-12);
}

TEST(FunctionalAssemblerBoundaryTest,
     PopulatesFrameExplicitBoundaryGeometryForFormsKernel)
{
    constexpr int boundary_marker = 2;
    forms::test::SingleTetraOneBoundaryFaceMeshAccess mesh(
        boundary_marker);
    auto dof_map = forms::test::createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    const auto current_normal = forms::FormExpr::currentNormal();
    const auto reference_normal = forms::FormExpr::referenceNormal();
    const auto integrand =
        forms::inner(current_normal, reference_normal) +
        forms::FormExpr::currentMeasure() +
        forms::FormExpr::referenceMeasure();
    auto kernel = forms::compileBoundaryFunctionalKernel(
        integrand, boundary_marker);

    FunctionalAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setDofMap(dof_map);
    assembler.setSpace(space);

    const Real value = assembler.assembleBoundaryScalar(
        *kernel, boundary_marker);
    EXPECT_NEAR(value, 1.5, 1e-12);
}

TEST(FunctionalAssemblerBoundaryTest, UniformlyTinyValidBoundaryFacePreservesMeasure)
{
    constexpr Real scale = 1e-16;
    forms::test::SingleTetraOneBoundaryFaceMeshAccess mesh(
        /*boundary_marker=*/2, scale);
    auto dof_map = forms::test::createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FunctionalAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setDofMap(dof_map);
    assembler.setSpace(space);

    BoundaryMeasureKernel kernel;
    const Real area = assembler.assembleBoundaryScalar(kernel, /*boundary_marker=*/2);
    ASSERT_TRUE(std::isfinite(area));
    EXPECT_NEAR(area / (scale * scale), 0.5, 1e-12);
}

TEST(FunctionalAssemblerBoundaryTest, BoundaryFluxUsesElementLocalCoefficients)
{
    forms::test::SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);

    // Cell DOFs are not 0..3 to ensure the assembler gathers element-local coefficients correctly.
    dofs::DofMap dof_map(/*n_cells=*/1, /*n_dofs_total=*/10, /*dofs_per_cell=*/4);
    const std::array<GlobalIndex, 4> cell_dofs = {5, 6, 7, 8};
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(10);
    dof_map.setNumLocalDofs(10);
    dof_map.finalize();

    spaces::H1Space space(ElementType::Tetra4, 1);

    // Linear field u=z on the reference tetra:
    // nodes (0,1,2) have z=0, node 3 has z=1.
    std::vector<Real> u(10, 100.0);
    u[5] = 0.0;
    u[6] = 0.0;
    u[7] = 0.0;
    u[8] = 1.0;

    FunctionalAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setDofMap(dof_map);
    assembler.setSpace(space);
    assembler.setSolution(u);

    BoundaryFluxKernel kernel;
    const Real flux = assembler.assembleBoundaryScalar(kernel, /*boundary_marker=*/2);

    // On face {0,1,2} (z=0 plane), outward normal is -e_z and grad(z)=+e_z.
    // So ∫ grad(u)·n ds = ∫ (-1) ds = -area = -0.5.
    EXPECT_NEAR(flux, -0.5, 1e-12);
}

// A kernel asking only for the raw coefficients of an H(div) field gets them,
// on cells and on boundary faces, although a functional's context has no
// tables for a vector basis: RT0 on one tetrahedron, coefficients 1..4 summed
// over the cell (volume 1/6) and over the marked face (area 1/2).
TEST(FunctionalAssemblerTest, VectorBasisKernelReceivesRawCoefficients)
{
    forms::test::SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);
    auto dof_map = forms::test::createSingleTetraDofMap();
    spaces::HDivSpace space(ElementType::Tetra4, /*order=*/0, BasisType::RaviartThomas);
    ASSERT_EQ(space.dofs_per_element(), 4u);
    const std::vector<Real> u = {1.0, 2.0, 3.0, 4.0};

    CoefficientSumKernel kernel;
    {
        FunctionalAssembler assembler;
        assembler.setMesh(mesh);
        assembler.setDofMap(dof_map);
        assembler.setSpace(space);
        assembler.setSolution(u);
        EXPECT_NEAR(assembler.assembleScalar(kernel), 10.0 / 6.0, 1e-12);
    }
    {
        FunctionalAssembler assembler;
        assembler.setMesh(mesh);
        assembler.setDofMap(dof_map);
        assembler.setSpace(space);
        assembler.setSolution(u);
        EXPECT_NEAR(assembler.assembleBoundaryScalar(kernel, /*boundary_marker=*/2), 5.0, 1e-12);
    }
}

TEST(FunctionalAssemblerBoundaryTest, FormsBoundaryFunctionalKernelEvaluatesDiscreteField)
{
    constexpr FieldId field = 0;

    forms::test::SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);

    dofs::DofMap dof_map(/*n_cells=*/1, /*n_dofs_total=*/10, /*dofs_per_cell=*/4);
    const std::array<GlobalIndex, 4> cell_dofs = {5, 6, 7, 8};
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(10);
    dof_map.setNumLocalDofs(10);
    dof_map.finalize();

    spaces::H1Space space(ElementType::Tetra4, 1);

    std::vector<Real> u(10, 123.0);
    u[5] = 1.0;
    u[6] = 1.0;
    u[7] = 1.0;
    u[8] = 1.0;

    const auto integrand = forms::FormExpr::discreteField(field, space, "u");
    auto kernel = forms::compileBoundaryFunctionalKernel(integrand, /*boundary_marker=*/2);

    FunctionalAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setDofMap(dof_map);
    assembler.setSpace(space);
    assembler.setPrimaryField(field);
    assembler.setSolution(u);

    const Real Q = assembler.assembleBoundaryScalar(*kernel, /*boundary_marker=*/2);
    EXPECT_NEAR(Q, 0.5, 1e-12);
}

TEST(FunctionalAssemblerBoundaryTest, BlockModePrimaryFieldBindingExtractsPrimarySlice)
{
    constexpr FieldId pressure_field = 1;

    forms::test::SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);

    dofs::DofMap system_dof_map(/*n_cells=*/1, /*n_dofs_total=*/300, /*dofs_per_cell=*/8);
    const std::array<GlobalIndex, 8> cell_dofs = {100, 101, 102, 103, 200, 201, 202, 203};
    system_dof_map.setCellDofs(0, cell_dofs);
    system_dof_map.setNumDofs(300);
    system_dof_map.setNumLocalDofs(300);
    system_dof_map.finalize();

    spaces::H1Space scalar_space(ElementType::Tetra4, 1);

    std::vector<Real> u(300, 0.0);
    u[100] = 100.0;
    u[101] = 100.0;
    u[102] = 100.0;
    u[103] = 100.0;
    u[200] = 1.0;
    u[201] = 1.0;
    u[202] = 1.0;
    u[203] = 1.0;

    const auto integrand = forms::FormExpr::discreteField(pressure_field, scalar_space, "p");
    auto kernel = forms::compileBoundaryFunctionalKernel(integrand, /*boundary_marker=*/2);

    FunctionalAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setDofMap(system_dof_map);
    assembler.setSpace(scalar_space);
    assembler.setPrimaryField(pressure_field);
    assembler.setDofPerNode(0);
    assembler.registerFieldBinding(assembly::FieldSolutionBinding{
        .field = pressure_field,
        .space = &scalar_space,
        .field_type = FieldType::Scalar,
        .value_dimension = 1,
        .component_offset = 4,
        .n_components = 1,
    });
    assembler.setSolution(u);

    const Real Q = assembler.assembleBoundaryScalar(*kernel, /*boundary_marker=*/2);
    EXPECT_NEAR(Q, 0.5, 1e-12);
}

TEST(FunctionalAssemblerBoundaryTest, BlockModePrimaryFieldBindingUsesMonolithicRangeBeforeFieldLocalMap)
{
    constexpr FieldId pressure_field = 1;

    forms::test::SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);

    dofs::DofMap system_dof_map(/*n_cells=*/1, /*n_dofs_total=*/300, /*dofs_per_cell=*/8);
    const std::array<GlobalIndex, 8> cell_dofs = {100, 101, 102, 103, 200, 201, 202, 203};
    system_dof_map.setCellDofs(0, cell_dofs);
    system_dof_map.setNumDofs(300);
    system_dof_map.setNumLocalDofs(300);
    system_dof_map.finalize();

    dofs::DofMap field_local_map(/*n_cells=*/1, /*n_dofs_total=*/100, /*dofs_per_cell=*/4);
    const std::array<GlobalIndex, 4> bogus_field_dofs = {50, 51, 52, 53};
    field_local_map.setCellDofs(0, bogus_field_dofs);
    field_local_map.setNumDofs(100);
    field_local_map.setNumLocalDofs(100);
    field_local_map.finalize();

    spaces::H1Space scalar_space(ElementType::Tetra4, 1);

    std::vector<Real> u(300, 0.0);
    u[100] = 100.0;
    u[101] = 100.0;
    u[102] = 100.0;
    u[103] = 100.0;
    u[200] = 1.0;
    u[201] = 1.0;
    u[202] = 1.0;
    u[203] = 1.0;

    const auto integrand = forms::FormExpr::discreteField(pressure_field, scalar_space, "p");
    auto kernel = forms::compileBoundaryFunctionalKernel(integrand, /*boundary_marker=*/2);

    FunctionalAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setDofMap(system_dof_map);
    assembler.setSpace(scalar_space);
    assembler.setPrimaryField(pressure_field);
    assembler.setDofPerNode(0);
    assembler.registerFieldBinding(assembly::FieldSolutionBinding{
        .field = pressure_field,
        .space = &scalar_space,
        .dof_map = &field_local_map,
        .dof_offset = 200,
        .field_global_size = 100,
        .field_type = FieldType::Scalar,
        .value_dimension = 1,
        .n_components = 1,
    });
    assembler.setSolution(u);

    const Real Q = assembler.assembleBoundaryScalar(*kernel, /*boundary_marker=*/2);
    EXPECT_NEAR(Q, 0.5, 1e-12);
}

TEST(FunctionalAssemblerBoundaryTest, BlockModeProductSpaceBindingUsesComponentBlockedOrdering)
{
    constexpr FieldId velocity_field = 2;

    forms::test::SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);

    dofs::DofMap system_dof_map(/*n_cells=*/1, /*n_dofs_total=*/400, /*dofs_per_cell=*/12);
    const std::array<GlobalIndex, 12> cell_dofs = {
        100, 101, 102, 103, 200, 201, 202, 203, 204, 205, 206, 207};
    system_dof_map.setCellDofs(0, cell_dofs);
    system_dof_map.setNumDofs(400);
    system_dof_map.setNumLocalDofs(400);
    system_dof_map.finalize();

    auto scalar_space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace velocity_space(scalar_space, /*components=*/2);

    std::vector<Real> u(400, 0.0);
    u[100] = 100.0;
    u[101] = 100.0;
    u[102] = 100.0;
    u[103] = 100.0;
    u[200] = 1.0;
    u[201] = 1.0;
    u[202] = 1.0;
    u[203] = 1.0;
    u[204] = 2.0;
    u[205] = 2.0;
    u[206] = 2.0;
    u[207] = 2.0;

    const auto integrand = forms::FormExpr::discreteField(velocity_field, velocity_space, "u").component(0);
    auto kernel = forms::compileBoundaryFunctionalKernel(integrand, /*boundary_marker=*/2);

    FunctionalAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setDofMap(system_dof_map);
    assembler.setSpace(velocity_space);
    assembler.setPrimaryField(velocity_field);
    assembler.setDofPerNode(0);
    assembler.registerFieldBinding(assembly::FieldSolutionBinding{
        .field = velocity_field,
        .space = &velocity_space,
        .field_type = FieldType::Vector,
        .value_dimension = 2,
        .component_offset = 4,
        .n_components = 2,
    });
    assembler.setSolution(u);

    const Real Q = assembler.assembleBoundaryScalar(*kernel, /*boundary_marker=*/2);
    EXPECT_NEAR(Q, 0.5, 1e-12);
}

} // namespace test
} // namespace assembly
} // namespace FE
} // namespace svmp
