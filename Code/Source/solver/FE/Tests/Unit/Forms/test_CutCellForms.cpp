#include "Forms/CutCellForms.h"

#include <gtest/gtest.h>

#include "Assembly/AssemblyContext.h"
#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Core/AlignedAllocator.h"
#include "Geometry/CutQuadrature.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/JIT/JITFunctionalKernelWrapper.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

using namespace svmp::FE::forms;

namespace {

using svmp::FE::AlignedAllocator;
using svmp::FE::kFEPreferredAlignmentBytes;
using svmp::FE::Real;
using svmp::FE::assembly::AssemblyContext;
using svmp::FE::forms::CutCellParameterSlots;
using svmp::FE::geometry::CutIntegrationSide;
using svmp::FE::geometry::CutQuadratureKind;
using svmp::FE::geometry::CutQuadratureRule;
using svmp::FE::assembly::ContextType;
using svmp::FE::assembly::RequiredData;

using JITConstants = std::vector<Real, AlignedAllocator<Real, kFEPreferredAlignmentBytes>>;

[[nodiscard]] JITConstants makeCutConstants(const CutQuadratureRule& rule,
                                            Real side_indicator,
                                            Real stabilization_scale,
                                            Real quadrature_weight_sensitivity)
{
    const CutCellParameterSlots slots;
    JITConstants constants(13u, Real(0.0));
    constants[slots.volume_fraction] = rule.volume_fraction;
    constants[slots.side_indicator] = side_indicator;
    if (!rule.points.empty()) {
        constants[slots.embedded_normal[0]] = rule.points.front().normal[0];
        constants[slots.embedded_normal[1]] = rule.points.front().normal[1];
        constants[slots.embedded_normal[2]] = rule.points.front().normal[2];
    }
    constants[slots.stabilization_scale] = stabilization_scale;
    constants[slots.quadrature_weight_sensitivity] = quadrature_weight_sensitivity;
    return constants;
}

[[nodiscard]] AssemblyContext makeCutAssemblyContext(const CutQuadratureRule& rule,
                                                     std::span<const Real> constants)
{
    AssemblyContext ctx;
    svmp::FE::spaces::H1Space space(svmp::FE::ElementType::Tetra4, /*order=*/1);
    if (rule.kind == CutQuadratureKind::Interface) {
        ctx.configureFace(/*face_id=*/0, /*cell_id=*/0, /*local_face_id=*/0,
                          space, space, RequiredData::None, ContextType::BoundaryFace);
    } else {
        ctx.configure(/*cell_id=*/0, space, space, RequiredData::None);
    }
    ctx.reserve(/*max_dofs=*/space.dofs_per_element(),
                static_cast<svmp::FE::LocalIndex>(rule.points.size()),
                /*dim=*/3);

    std::vector<AssemblyContext::Point3D> points;
    std::vector<AssemblyContext::Vector3D> normals;
    std::vector<Real> weights;
    points.reserve(rule.points.size());
    normals.reserve(rule.points.size());
    weights.reserve(rule.points.size());
    for (const auto& qp : rule.points) {
        points.push_back(qp.point);
        normals.push_back(qp.normal);
        weights.push_back(qp.weight);
    }

    ctx.setQuadratureData(points, weights);
    ctx.setPhysicalPoints(points);
    ctx.setNormals(normals);
    ctx.setIntegrationWeights(weights);
    ctx.setEntityMeasures(/*cell_diameter=*/Real(1.0),
                          rule.kind == CutQuadratureKind::Volume ? rule.measure : Real(0.0),
                          rule.kind == CutQuadratureKind::Interface ? rule.measure : Real(0.0));
    ctx.setJITConstants(constants);
    return ctx;
}

[[nodiscard]] FormExpr cutMetadataIntegrand(bool include_quadrature_normal)
{
    const auto terminals = cutCellTerminals();
    const auto x = FormExpr::coordinate();
    auto integrand =
        terminals.volume_fraction * Real(2.0) +
        terminals.side_indicator * Real(0.125) +
        terminals.stabilization_scale * Real(0.5) +
        terminals.quadrature_weight_sensitivity * Real(0.25) +
        x.component(0) * Real(0.5) +
        x.component(1) * Real(0.25) +
        x.component(2) * Real(0.125);

    if (include_quadrature_normal) {
        const auto n = FormExpr::normal();
        const auto embedded_normal_alignment =
            terminals.embedded_normal.component(0) * n.component(0) +
            terminals.embedded_normal.component(1) * n.component(1) +
            terminals.embedded_normal.component(2) * n.component(2);
        integrand = integrand + embedded_normal_alignment * Real(0.75);
    }

    return integrand;
}

[[nodiscard]] Real manualCutMetadataIntegral(const CutQuadratureRule& rule,
                                             std::span<const Real> constants,
                                             bool include_quadrature_normal)
{
    const CutCellParameterSlots slots;
    Real total = 0.0;
    for (const auto& qp : rule.points) {
        const Real normal_alignment =
            constants[slots.embedded_normal[0]] * qp.normal[0] +
            constants[slots.embedded_normal[1]] * qp.normal[1] +
            constants[slots.embedded_normal[2]] * qp.normal[2];
        Real value =
            constants[slots.volume_fraction] * Real(2.0) +
            constants[slots.side_indicator] * Real(0.125) +
            constants[slots.stabilization_scale] * Real(0.5) +
            constants[slots.quadrature_weight_sensitivity] * Real(0.25) +
            qp.point[0] * Real(0.5) +
            qp.point[1] * Real(0.25) +
            qp.point[2] * Real(0.125);
        if (include_quadrature_normal) {
            value += normal_alignment * Real(0.75);
        }
        total += value * qp.weight;
    }
    return total;
}

[[nodiscard]] Real assembleInterpreterCellTotal(const FormExpr& integrand,
                                                const AssemblyContext& ctx)
{
    auto kernel = svmp::FE::forms::test::makeFunctionalFormKernel(
        integrand, FunctionalFormKernel::Domain::Cell);
    return kernel->evaluateCellTotal(ctx);
}

[[nodiscard]] Real assembleJITCellTotal(const FormExpr& integrand,
                                        const AssemblyContext& ctx)
{
    auto fallback = svmp::FE::forms::test::makeThrowingTotalKernelFor(
        integrand, /*has_cell=*/true, /*has_boundary=*/false);
    svmp::FE::forms::jit::JITFunctionalKernelWrapper kernel(
        fallback, integrand, svmp::FE::forms::jit::JITFunctionalKernelWrapper::Domain::Cell,
        svmp::FE::forms::test::makeUnitTestJITOptions());
    return kernel.evaluateCellTotal(ctx);
}

[[nodiscard]] Real assembleInterpreterInterfaceTotal(const FormExpr& integrand,
                                                     const AssemblyContext& ctx,
                                                     int marker)
{
    auto kernel = svmp::FE::forms::test::makeFunctionalFormKernel(
        integrand, FunctionalFormKernel::Domain::BoundaryFace);
    return kernel->evaluateBoundaryFaceTotal(ctx, marker);
}

[[nodiscard]] Real assembleJITInterfaceTotal(const FormExpr& integrand,
                                             const AssemblyContext& ctx,
                                             int marker)
{
    auto fallback = svmp::FE::forms::test::makeThrowingTotalKernelFor(
        integrand, /*has_cell=*/false, /*has_boundary=*/true);
    svmp::FE::forms::jit::JITFunctionalKernelWrapper kernel(
        fallback, integrand, svmp::FE::forms::jit::JITFunctionalKernelWrapper::Domain::BoundaryFace,
        svmp::FE::forms::test::makeUnitTestJITOptions());
    return kernel.evaluateBoundaryFaceTotal(ctx, marker);
}

void expectCutMetadataResidualTangentConsistent(const FormExpr& residual,
                                                const svmp::FE::spaces::FunctionSpace& space,
                                                std::span<const Real> constants,
                                                const std::vector<Real>& solution,
                                                Real finite_difference_eps,
                                                Real finite_difference_tol)
{
    const auto n_dofs = static_cast<svmp::FE::GlobalIndex>(space.dofs_per_element());
    ASSERT_EQ(static_cast<svmp::FE::GlobalIndex>(solution.size()), n_dofs);

    svmp::FE::forms::test::SingleTetraMeshAccess mesh;
    auto dof_map = svmp::FE::forms::test::createSingleTetraDofMap();

    FormCompiler compiler;
    auto ad_ir = compiler.compileResidual(residual);
    auto symbolic_ir = compiler.compileResidual(residual);
    auto symbolic_vector_ir = compiler.compileResidual(residual);

    NonlinearFormKernel ad_kernel(
        std::move(ad_ir), ADMode::Forward, NonlinearKernelOutput::Both);
    SymbolicNonlinearFormKernel symbolic_kernel(
        std::move(symbolic_ir), NonlinearKernelOutput::Both);
    SymbolicNonlinearFormKernel symbolic_vector_kernel(
        std::move(symbolic_vector_ir), NonlinearKernelOutput::VectorOnly);
    symbolic_kernel.resolveInlinableConstitutives();
    symbolic_vector_kernel.resolveInlinableConstitutives();

    svmp::FE::assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setJITConstants(constants);

    svmp::FE::assembly::DenseMatrixView ad_jacobian(n_dofs);
    svmp::FE::assembly::DenseMatrixView symbolic_jacobian(n_dofs);
    svmp::FE::assembly::DenseVectorView ad_residual(n_dofs);
    svmp::FE::assembly::DenseVectorView symbolic_residual(n_dofs);
    ad_jacobian.zero();
    symbolic_jacobian.zero();
    ad_residual.zero();
    symbolic_residual.zero();

    assembler.setCurrentSolution(solution);
    (void)assembler.assembleBoth(mesh, space, space, ad_kernel, ad_jacobian, ad_residual);
    assembler.setCurrentSolution(solution);
    (void)assembler.assembleBoth(mesh, space, space, symbolic_kernel,
                                 symbolic_jacobian, symbolic_residual);

    for (svmp::FE::GlobalIndex i = 0; i < n_dofs; ++i) {
        EXPECT_NEAR(ad_residual.getVectorEntry(i), symbolic_residual.getVectorEntry(i), 1.0e-12);
        for (svmp::FE::GlobalIndex j = 0; j < n_dofs; ++j) {
            EXPECT_NEAR(ad_jacobian.getMatrixEntry(i, j),
                        symbolic_jacobian.getMatrixEntry(i, j),
                        1.0e-10);
        }
    }

    for (svmp::FE::GlobalIndex j = 0; j < n_dofs; ++j) {
        auto solution_plus = solution;
        auto solution_minus = solution;
        solution_plus[static_cast<std::size_t>(j)] += finite_difference_eps;
        solution_minus[static_cast<std::size_t>(j)] -= finite_difference_eps;

        svmp::FE::assembly::DenseVectorView residual_plus(n_dofs);
        svmp::FE::assembly::DenseVectorView residual_minus(n_dofs);
        residual_plus.zero();
        residual_minus.zero();

        assembler.setCurrentSolution(solution_plus);
        (void)assembler.assembleVector(mesh, space, symbolic_vector_kernel, residual_plus);
        assembler.setCurrentSolution(solution_minus);
        (void)assembler.assembleVector(mesh, space, symbolic_vector_kernel, residual_minus);

        for (svmp::FE::GlobalIndex i = 0; i < n_dofs; ++i) {
            const Real finite_difference =
                (residual_plus.getVectorEntry(i) - residual_minus.getVectorEntry(i)) /
                (Real(2.0) * finite_difference_eps);
            EXPECT_NEAR(symbolic_jacobian.getMatrixEntry(i, j),
                        finite_difference,
                        finite_difference_tol);
        }
    }
}

[[nodiscard]] std::vector<Real> solveSmallDenseSystem(const svmp::FE::assembly::DenseMatrixView& matrix,
                                                      const svmp::FE::assembly::DenseVectorView& rhs,
                                                      svmp::FE::GlobalIndex n_dofs)
{
    const auto n = static_cast<std::size_t>(n_dofs);
    std::vector<Real> a(n * n, Real(0.0));
    std::vector<Real> b(n, Real(0.0));
    for (std::size_t i = 0; i < n; ++i) {
        b[i] = rhs.getVectorEntry(static_cast<svmp::FE::GlobalIndex>(i));
        for (std::size_t j = 0; j < n; ++j) {
            a[i * n + j] = matrix.getMatrixEntry(
                static_cast<svmp::FE::GlobalIndex>(i),
                static_cast<svmp::FE::GlobalIndex>(j));
        }
    }

    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivot = k;
        Real pivot_abs = std::abs(a[k * n + k]);
        for (std::size_t row = k + 1; row < n; ++row) {
            const Real value = std::abs(a[row * n + k]);
            if (value > pivot_abs) {
                pivot = row;
                pivot_abs = value;
            }
        }
        if (!(pivot_abs > Real(1.0e-14))) {
            throw std::runtime_error("solveSmallDenseSystem: singular Newton tangent");
        }
        if (pivot != k) {
            for (std::size_t col = k; col < n; ++col) {
                std::swap(a[k * n + col], a[pivot * n + col]);
            }
            std::swap(b[k], b[pivot]);
        }

        const Real diagonal = a[k * n + k];
        for (std::size_t col = k; col < n; ++col) {
            a[k * n + col] /= diagonal;
        }
        b[k] /= diagonal;

        for (std::size_t row = k + 1; row < n; ++row) {
            const Real factor = a[row * n + k];
            if (factor == Real(0.0)) {
                continue;
            }
            for (std::size_t col = k; col < n; ++col) {
                a[row * n + col] -= factor * a[k * n + col];
            }
            b[row] -= factor * b[k];
        }
    }

    std::vector<Real> solution(n, Real(0.0));
    for (std::size_t ii = 0; ii < n; ++ii) {
        const auto i = n - 1u - ii;
        Real value = b[i];
        for (std::size_t j = i + 1; j < n; ++j) {
            value -= a[i * n + j] * solution[j];
        }
        solution[i] = value;
    }
    return solution;
}

[[nodiscard]] Real vectorMaxAbs(const svmp::FE::assembly::DenseVectorView& vector,
                                svmp::FE::GlobalIndex n_dofs)
{
    Real max_abs = Real(0.0);
    for (svmp::FE::GlobalIndex i = 0; i < n_dofs; ++i) {
        max_abs = std::max(max_abs, std::abs(vector.getVectorEntry(i)));
    }
    return max_abs;
}

struct CutNewtonResult {
    std::vector<Real> solution;
    std::vector<Real> residual_norms;
};

[[nodiscard]] CutNewtonResult runCutMetadataNewtonIterations(
    const FormExpr& residual,
    const svmp::FE::spaces::FunctionSpace& space,
    std::span<const Real> constants,
    std::vector<Real> solution,
    int max_iterations)
{
    const auto n_dofs = static_cast<svmp::FE::GlobalIndex>(space.dofs_per_element());
    svmp::FE::forms::test::SingleTetraMeshAccess mesh;
    auto dof_map = svmp::FE::forms::test::createSingleTetraDofMap();

    FormCompiler compiler;
    auto ir = compiler.compileResidual(residual);
    SymbolicNonlinearFormKernel kernel(std::move(ir), NonlinearKernelOutput::Both);
    kernel.resolveInlinableConstitutives();

    svmp::FE::assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setJITConstants(constants);

    CutNewtonResult result;
    result.solution = std::move(solution);
    for (int iteration = 0; iteration <= max_iterations; ++iteration) {
        svmp::FE::assembly::DenseMatrixView jacobian(n_dofs);
        svmp::FE::assembly::DenseVectorView residual_vector(n_dofs);
        jacobian.zero();
        residual_vector.zero();

        assembler.setCurrentSolution(result.solution);
        (void)assembler.assembleBoth(mesh, space, space, kernel, jacobian, residual_vector);
        result.residual_norms.push_back(vectorMaxAbs(residual_vector, n_dofs));
        if (iteration == max_iterations) {
            break;
        }

        svmp::FE::assembly::DenseVectorView rhs(n_dofs);
        rhs.zero();
        for (svmp::FE::GlobalIndex i = 0; i < n_dofs; ++i) {
            rhs.addVectorEntry(i, -residual_vector.getVectorEntry(i));
        }

        const auto correction = solveSmallDenseSystem(jacobian, rhs, n_dofs);
        for (svmp::FE::GlobalIndex i = 0; i < n_dofs; ++i) {
            result.solution[static_cast<std::size_t>(i)] +=
                correction[static_cast<std::size_t>(i)];
        }
    }

    return result;
}

[[nodiscard]] Real maxAbsDifference(std::span<const Real> a, std::span<const Real> b)
{
    Real max_abs = Real(0.0);
    const auto n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        max_abs = std::max(max_abs, std::abs(a[i] - b[i]));
    }
    return max_abs;
}

} // namespace

TEST(CutCellForms, BuildsParameterBackedCutMetadataTerminals)
{
    CutCellParameterSlots slots;
    slots.volume_fraction = 10;
    slots.side_indicator = 11;
    slots.embedded_normal = {{12, 13, 14}};
    slots.stabilization_scale = 15;
    slots.measure_sensitivity = {{16, 17, 18}};
    slots.normal_sensitivity = {{19, 20, 21}};
    slots.quadrature_weight_sensitivity = 22;

    const auto terminals = cutCellTerminals(slots);
    EXPECT_NE(terminals.volume_fraction.toString().find("param[10]"), std::string::npos);
    EXPECT_NE(terminals.side_indicator.toString().find("param[11]"), std::string::npos);
    EXPECT_NE(terminals.embedded_normal.toString().find("param[12]"), std::string::npos);
    EXPECT_NE(terminals.embedded_normal.toString().find("param[14]"), std::string::npos);
    EXPECT_NE(terminals.stabilization_scale.toString().find("param[15]"), std::string::npos);
    EXPECT_NE(terminals.measure_sensitivity.toString().find("param[16]"), std::string::npos);
    EXPECT_NE(terminals.measure_sensitivity.toString().find("param[18]"), std::string::npos);
    EXPECT_NE(terminals.normal_sensitivity.toString().find("param[19]"), std::string::npos);
    EXPECT_NE(terminals.normal_sensitivity.toString().find("param[21]"), std::string::npos);
    EXPECT_NE(terminals.quadrature_weight_sensitivity.toString().find("param[22]"), std::string::npos);
}

TEST(CutCellForms, CutVolumeIntegralJITMatchesInterpreterOnSameCutQuadrature)
{
    requireLLVMJITOrSkip();

    const auto rule = svmp::FE::geometry::makeAxisAlignedBoxCutVolumeQuadrature(
        {{0.0, 0.0, 0.0}},
        {{1.0, 1.0, 1.0}},
        /*axis=*/0,
        /*cut_coordinate=*/Real(0.25),
        CutIntegrationSide::Negative,
        "jit-volume-plane");
    ASSERT_EQ(rule.kind, CutQuadratureKind::Volume);
    ASSERT_FALSE(rule.points.empty());

    const auto constants = makeCutConstants(
        rule, /*side_indicator=*/Real(-1.0), /*stabilization_scale=*/Real(0.375),
        /*quadrature_weight_sensitivity=*/Real(0.0625));
    auto ctx = makeCutAssemblyContext(rule, constants);
    const auto integrand = cutMetadataIntegrand(/*include_quadrature_normal=*/false);

    const Real expected = manualCutMetadataIntegral(rule, constants, /*include_quadrature_normal=*/false);
    const Real interp = assembleInterpreterCellTotal(integrand, ctx);
    const Real jit = assembleJITCellTotal(integrand, ctx);

    EXPECT_NEAR(interp, expected, 1.0e-13);
    EXPECT_NEAR(jit, expected, 1.0e-13);
    EXPECT_NEAR(jit, interp, 1.0e-13);
}

TEST(CutCellForms, CutMetadataResidualTangentMatchesADAndFiniteDifferenceVerification)
{
    const auto rule = svmp::FE::geometry::makeAxisAlignedBoxCutVolumeQuadrature(
        {{0.0, 0.0, 0.0}},
        {{1.0, 1.0, 1.0}},
        /*axis=*/0,
        /*cut_coordinate=*/Real(0.375),
        CutIntegrationSide::Negative,
        "residual-tangent-plane");
    ASSERT_EQ(rule.kind, CutQuadratureKind::Volume);
    ASSERT_FALSE(rule.points.empty());

    auto constants = makeCutConstants(
        rule, /*side_indicator=*/Real(-1.0), /*stabilization_scale=*/Real(0.25),
        /*quadrature_weight_sensitivity=*/Real(0.03125));
    const CutCellParameterSlots slots;
    constants[slots.measure_sensitivity[0]] = Real(0.05);
    constants[slots.measure_sensitivity[1]] = Real(-0.025);
    constants[slots.measure_sensitivity[2]] = Real(0.075);
    constants[slots.normal_sensitivity[0]] = Real(0.0125);
    constants[slots.normal_sensitivity[1]] = Real(-0.01875);
    constants[slots.normal_sensitivity[2]] = Real(0.00625);

    svmp::FE::spaces::H1Space space(svmp::FE::ElementType::Tetra4, /*order=*/1);
    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto terminals = cutCellTerminals();

    const auto cut_coefficient =
        FormExpr::constant(Real(1.25)) +
        terminals.volume_fraction * Real(0.70) +
        terminals.side_indicator * Real(0.05) +
        terminals.embedded_normal.component(0) * Real(0.20) +
        terminals.embedded_normal.component(1) * Real(0.10) +
        terminals.stabilization_scale * Real(0.125) +
        terminals.measure_sensitivity.component(0) * Real(0.30) +
        terminals.measure_sensitivity.component(2) * Real(0.15) +
        terminals.normal_sensitivity.component(1) * Real(0.20) +
        terminals.quadrature_weight_sensitivity * Real(0.40);

    const auto nonlinear_value = u * u + (u * u * u) * Real(0.125);
    const auto nonlinear_gradient =
        inner(grad(u), grad(v)) * (FormExpr::constant(Real(1.0)) + u * Real(0.50));
    const auto residual =
        (cut_coefficient * (nonlinear_value * v + nonlinear_gradient * Real(0.20))).dx();

    const std::vector<Real> solution = {Real(0.16), Real(-0.11), Real(0.08), Real(0.03)};
    expectCutMetadataResidualTangentConsistent(
        residual, space, constants, solution,
        /*finite_difference_eps=*/Real(1.0e-6),
        /*finite_difference_tol=*/Real(5.0e-6));
}

TEST(CutCellForms, CutSensitivityTerminalsDriveSymbolicNewtonTangentConvergence)
{
    const auto rule = svmp::FE::geometry::makeAxisAlignedBoxCutVolumeQuadrature(
        {{0.0, 0.0, 0.0}},
        {{1.0, 1.0, 1.0}},
        /*axis=*/1,
        /*cut_coordinate=*/Real(0.40),
        CutIntegrationSide::Positive,
        "newton-sensitivity-plane");
    ASSERT_EQ(rule.kind, CutQuadratureKind::Volume);
    ASSERT_FALSE(rule.points.empty());

    auto constants = makeCutConstants(
        rule, /*side_indicator=*/Real(1.0), /*stabilization_scale=*/Real(0.1875),
        /*quadrature_weight_sensitivity=*/Real(0.045));
    const CutCellParameterSlots slots;
    constants[slots.measure_sensitivity[0]] = Real(0.080);
    constants[slots.measure_sensitivity[1]] = Real(-0.035);
    constants[slots.measure_sensitivity[2]] = Real(0.055);
    constants[slots.normal_sensitivity[0]] = Real(0.020);
    constants[slots.normal_sensitivity[1]] = Real(0.015);
    constants[slots.normal_sensitivity[2]] = Real(-0.010);

    svmp::FE::spaces::H1Space space(svmp::FE::ElementType::Tetra4, /*order=*/1);
    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto x = FormExpr::coordinate();
    const auto terminals = cutCellTerminals();

    const auto target = FormExpr::constant(Real(0.12));
    const auto displacement = u - target;
    const auto value_weight =
        FormExpr::constant(Real(1.40)) +
        terminals.volume_fraction * Real(0.35) +
        terminals.side_indicator * Real(0.025) +
        terminals.measure_sensitivity.component(0) * (FormExpr::constant(Real(0.50)) + x.component(0)) +
        terminals.measure_sensitivity.component(1) * (FormExpr::constant(Real(0.25)) + x.component(1)) +
        terminals.normal_sensitivity.component(2) * (FormExpr::constant(Real(0.30)) + x.component(2)) +
        terminals.quadrature_weight_sensitivity * (FormExpr::constant(Real(0.20)) + x.component(0));
    const auto gradient_weight =
        FormExpr::constant(Real(0.30)) +
        terminals.stabilization_scale * Real(0.20) +
        terminals.embedded_normal.component(1) * Real(0.05) +
        terminals.measure_sensitivity.component(2) * (FormExpr::constant(Real(0.10)) + x.component(2)) +
        terminals.normal_sensitivity.component(0) * (FormExpr::constant(Real(0.15)) + x.component(0)) +
        terminals.quadrature_weight_sensitivity * Real(0.10);

    const auto residual =
        (value_weight * (displacement + displacement * displacement * Real(0.70)) * v +
         gradient_weight * (FormExpr::constant(Real(1.0)) + displacement * Real(0.40)) *
             inner(grad(u), grad(v))).dx();

    const std::vector<Real> initial = {Real(0.19), Real(0.04), Real(0.16), Real(0.10)};
    auto inactive_constants = constants;
    inactive_constants[slots.measure_sensitivity[0]] = Real(0.0);
    inactive_constants[slots.measure_sensitivity[1]] = Real(0.0);
    inactive_constants[slots.measure_sensitivity[2]] = Real(0.0);
    inactive_constants[slots.normal_sensitivity[0]] = Real(0.0);
    inactive_constants[slots.normal_sensitivity[1]] = Real(0.0);
    inactive_constants[slots.normal_sensitivity[2]] = Real(0.0);
    inactive_constants[slots.quadrature_weight_sensitivity] = Real(0.0);

    const auto active_first_step =
        runCutMetadataNewtonIterations(residual, space, constants, initial, /*max_iterations=*/1);
    const auto inactive_first_step =
        runCutMetadataNewtonIterations(residual, space, inactive_constants, initial, /*max_iterations=*/1);
    EXPECT_GT(maxAbsDifference(active_first_step.solution, inactive_first_step.solution), Real(1.0e-6));

    const auto active_newton =
        runCutMetadataNewtonIterations(residual, space, constants, initial, /*max_iterations=*/6);
    ASSERT_GE(active_newton.residual_norms.size(), std::size_t{4});
    EXPECT_LT(active_newton.residual_norms[1], active_newton.residual_norms[0]);
    EXPECT_LT(active_newton.residual_norms[2], active_newton.residual_norms[1]);
    EXPECT_LT(active_newton.residual_norms.back(), Real(1.0e-11));
    for (const auto value : active_newton.solution) {
        EXPECT_NEAR(value, Real(0.12), Real(1.0e-10));
    }
}

TEST(CutCellForms, CutInterfaceIntegralJITMatchesInterpreterOnSameCutQuadrature)
{
    requireLLVMJITOrSkip();

    auto rule = svmp::FE::geometry::makeAxisAlignedBoxCutInterfaceQuadrature(
        {{0.0, 0.0, 0.0}},
        {{1.0, 1.0, 1.0}},
        /*axis=*/0,
        /*cut_coordinate=*/Real(0.25),
        "jit-interface-plane");
    rule.volume_fraction = Real(1.0);
    ASSERT_EQ(rule.kind, CutQuadratureKind::Interface);
    ASSERT_FALSE(rule.points.empty());

    const auto constants = makeCutConstants(
        rule, /*side_indicator=*/Real(0.0), /*stabilization_scale=*/Real(0.125),
        /*quadrature_weight_sensitivity=*/Real(0.5));
    auto ctx = makeCutAssemblyContext(rule, constants);
    const auto integrand =
        cutMetadataIntegrand(/*include_quadrature_normal=*/true) + FormExpr::facetArea() * Real(0.03125);

    const Real expected =
        manualCutMetadataIntegral(rule, constants, /*include_quadrature_normal=*/true) +
        rule.measure * rule.measure * Real(0.03125);
    const Real interp = assembleInterpreterInterfaceTotal(integrand, ctx, /*marker=*/17);
    const Real jit = assembleJITInterfaceTotal(integrand, ctx, /*marker=*/17);

    EXPECT_NEAR(interp, expected, 1.0e-13);
    EXPECT_NEAR(jit, expected, 1.0e-13);
    EXPECT_NEAR(jit, interp, 1.0e-13);
}
