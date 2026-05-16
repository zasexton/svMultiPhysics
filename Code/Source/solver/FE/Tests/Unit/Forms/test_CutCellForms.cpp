#include "Forms/CutCellForms.h"

#include <gtest/gtest.h>

#include "Assembly/AssemblyContext.h"
#include "Assembly/CutDomainAssembler.h"
#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Core/AlignedAllocator.h"
#include "Geometry/CutQuadrature.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/JIT/JITCompiler.h"
#include "Forms/JIT/JITFunctionalKernelWrapper.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
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
    auto parameters = cutCellParametersForRule(
        rule, slots, stabilization_scale, quadrature_weight_sensitivity);
    JITConstants constants(parameters.begin(), parameters.end());
    constants[slots.side_indicator] = side_indicator;
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

[[nodiscard]] svmp::FE::dofs::DofMap createSingleCellDofMap(std::size_t n_dofs)
{
    const auto local_n = static_cast<svmp::FE::LocalIndex>(n_dofs);
    const auto global_n = static_cast<svmp::FE::GlobalIndex>(n_dofs);
    svmp::FE::dofs::DofMap dof_map(1, global_n, local_n);
    std::vector<svmp::FE::GlobalIndex> cell_dofs(n_dofs);
    for (std::size_t i = 0u; i < n_dofs; ++i) {
        cell_dofs[i] = static_cast<svmp::FE::GlobalIndex>(i);
    }
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(global_n);
    dof_map.setNumLocalDofs(global_n);
    dof_map.finalize();
    return dof_map;
}

[[nodiscard]] std::vector<Real> deterministicScalarCoefficients(std::size_t n_dofs)
{
    std::vector<Real> coeffs(n_dofs);
    for (std::size_t i = 0u; i < n_dofs; ++i) {
        const Real sign = (i % 2u == 0u) ? Real(1.0) : Real(-1.0);
        coeffs[i] = sign * (Real(0.07) + Real(0.011) * static_cast<Real>(i));
    }
    return coeffs;
}

[[nodiscard]] CutQuadratureRule makeReferenceTetraCutRule(
    CutIntegrationSide side,
    std::vector<svmp::FE::geometry::CutQuadraturePoint> points,
    Real parent_measure = Real(1.0) / Real(6.0))
{
    CutQuadratureRule rule;
    rule.kind = CutQuadratureKind::Volume;
    rule.side = side;
    rule.parent_measure = parent_measure;
    rule.measure = Real(0.0);
    for (const auto& qp : points) {
        rule.measure += qp.weight;
    }
    rule.volume_fraction = rule.measure / parent_measure;
    rule.exact_for_constants = true;
    rule.provenance.parent_entity = 0;
    rule.provenance.cut_topology_revision = 101u + static_cast<std::uint64_t>(side);
    rule.provenance.cut_topology_id =
        side == CutIntegrationSide::Negative ? "negative-reference-tetra-cut"
                                             : "positive-reference-tetra-cut";
    rule.points = std::move(points);
    return rule;
}

void markRuleAsHighOrderImplicit(CutQuadratureRule& rule,
                                 int marker,
                                 const std::string& topology_id,
                                 int quadrature_order)
{
    rule.exact_polynomial_order = quadrature_order;
    rule.curved_geometry = true;
    rule.policy.kind = svmp::FE::geometry::CutQuadratureConstructionKind::MomentFittedImplicit;
    rule.policy.polynomial_order = quadrature_order;
    rule.policy.moment_fitted = true;
    rule.policy.name = "fixed-geometry-high-order-implicit";
    rule.provenance.marker = marker;
    rule.provenance.parent_entity = 0;
    rule.provenance.cut_topology_id = topology_id;
    rule.provenance.cut_topology_revision = 401u;
    rule.provenance.construction =
        svmp::FE::geometry::CutQuadratureConstructionKind::MomentFittedImplicit;
    rule.provenance.frame = svmp::FE::geometry::CutGeometryFrame::Reference;
    rule.provenance.implicit_geometry_mode = "curved-level-set";
    rule.provenance.implicit_quadrature_backend = "moment-fitted-implicit";
    rule.provenance.implicit_fallback_policy = "none";
    rule.provenance.requested_quadrature_order = quadrature_order;
    rule.provenance.achieved_quadrature_order = quadrature_order;
    rule.frame = svmp::FE::geometry::CutGeometryFrame::Reference;
}

[[nodiscard]] CutQuadratureRule makeHighOrderReferenceTetraInterfaceRule(int marker)
{
    CutQuadratureRule rule;
    rule.kind = CutQuadratureKind::Interface;
    rule.side = CutIntegrationSide::Interface;
    rule.parent_measure = Real(1.0) / Real(6.0);
    rule.points = {
        {{{0.22, 0.18, 0.12}}, {{1.0, 0.0, 0.0}}, Real(0.019)},
        {{{0.24, 0.32, 0.10}}, {{1.0, 0.0, 0.0}}, Real(0.017)},
        {{{0.20, 0.15, 0.30}}, {{1.0, 0.0, 0.0}}, Real(0.016)},
        {{{0.28, 0.20, 0.19}}, {{1.0, 0.0, 0.0}}, Real(0.013)}
    };
    for (const auto& qp : rule.points) {
        rule.measure += qp.weight;
    }
    rule.volume_fraction = Real(0.0);
    rule.exact_for_constants = true;
    rule.provenance_id = "fixed-geometry-interface";
    markRuleAsHighOrderImplicit(rule,
                                marker,
                                "fixed-geometry-interface",
                                /*quadrature_order=*/4);
    return rule;
}

void populateP1ReferenceTetraCutContext(AssemblyContext& ctx,
                                        const CutQuadratureRule& rule,
                                        const svmp::FE::spaces::FunctionSpace& space,
                                        RequiredData required,
                                        std::span<const Real> constants,
                                        std::span<const Real> solution)
{
    ctx.configure(/*cell_id=*/0, space, space, required);
    ctx.reserve(/*max_dofs=*/static_cast<svmp::FE::LocalIndex>(space.dofs_per_element()),
                static_cast<svmp::FE::LocalIndex>(rule.points.size()),
                /*dim=*/3);

    std::vector<AssemblyContext::Point3D> qpts;
    std::vector<AssemblyContext::Vector3D> normals;
    std::vector<AssemblyContext::Matrix3x3> jacobians;
    std::vector<AssemblyContext::Matrix3x3> inverse_jacobians;
    std::vector<Real> jacobian_dets;
    std::vector<Real> weights;
    std::vector<Real> basis_values;
    std::vector<AssemblyContext::Vector3D> basis_gradients;

    const AssemblyContext::Matrix3x3 identity{{{{1.0, 0.0, 0.0}},
                                               {{0.0, 1.0, 0.0}},
                                               {{0.0, 0.0, 1.0}}}};
    const std::array<AssemblyContext::Vector3D, 4> p1_gradients{{
        {{-1.0, -1.0, -1.0}},
        {{ 1.0,  0.0,  0.0}},
        {{ 0.0,  1.0,  0.0}},
        {{ 0.0,  0.0,  1.0}}}};

    qpts.reserve(rule.points.size());
    normals.reserve(rule.points.size());
    jacobians.reserve(rule.points.size());
    inverse_jacobians.reserve(rule.points.size());
    jacobian_dets.reserve(rule.points.size());
    weights.reserve(rule.points.size());
    basis_values.reserve(rule.points.size() * 4u);
    basis_gradients.reserve(rule.points.size() * 4u);

    for (const auto& qp : rule.points) {
        qpts.push_back(qp.point);
        normals.push_back(qp.normal);
        jacobians.push_back(identity);
        inverse_jacobians.push_back(identity);
        jacobian_dets.push_back(Real(1.0));
        weights.push_back(qp.weight);

        const Real xi = qp.point[0];
        const Real eta = qp.point[1];
        const Real zeta = qp.point[2];
        const std::array<Real, 4> phi{{Real(1.0) - xi - eta - zeta, xi, eta, zeta}};
        for (std::size_t i = 0u; i < phi.size(); ++i) {
            basis_values.push_back(phi[i]);
            basis_gradients.push_back(p1_gradients[i]);
        }
    }

    ctx.setQuadratureData(qpts, weights);
    ctx.setPhysicalPoints(qpts);
    ctx.setJacobianData(jacobians, inverse_jacobians, jacobian_dets);
    ctx.setIntegrationWeights(weights);
    ctx.setNormals(normals);
    ctx.setEntityMeasures(/*cell_diameter=*/std::sqrt(Real(2.0)),
                          rule.measure,
                          /*facet_area=*/Real(0.0));
    ctx.setTestBasisDataQptMajor(/*n_dofs=*/4, basis_values, basis_gradients);
    ctx.setPhysicalGradients(
        basis_gradients, std::span<const AssemblyContext::Vector3D>{});
    ctx.setSolutionCoefficients(solution);
    ctx.setJITConstants(constants);
}

struct ManualResidualTangent {
    std::vector<Real> residual;
    std::vector<Real> tangent;
};

[[nodiscard]] ManualResidualTangent manualReferenceTetraCutResidualTangent(
    const svmp::FE::assembly::CutIntegrationContext& cut_context,
    std::span<const Real> solution)
{
    const CutCellParameterSlots slots;
    ManualResidualTangent manual;
    manual.residual.assign(4u, Real(0.0));
    manual.tangent.assign(16u, Real(0.0));

    const std::array<AssemblyContext::Vector3D, 4> p1_gradients{{
        {{-1.0, -1.0, -1.0}},
        {{ 1.0,  0.0,  0.0}},
        {{ 0.0,  1.0,  0.0}},
        {{ 0.0,  0.0,  1.0}}}};

    for (std::size_t rule_index = 0u; rule_index < cut_context.volumeRules().size(); ++rule_index) {
        const auto& rule = cut_context.volumeRules()[rule_index];
        auto constants = cutCellParametersForRule(
            rule, slots, Real(0.20) + Real(0.05) * static_cast<Real>(rule_index),
            Real(0.01));
        const Real coefficient =
            Real(1.10) +
            constants[slots.volume_fraction] * Real(0.70) +
            constants[slots.side_indicator] * Real(0.05) +
            constants[slots.embedded_normal[0]] * Real(0.20) +
            constants[slots.embedded_normal[1]] * Real(0.10) +
            constants[slots.stabilization_scale] * Real(0.125) +
            constants[slots.quadrature_weight_sensitivity] * Real(0.40);

        for (const auto& qp : rule.points) {
            const Real xi = qp.point[0];
            const Real eta = qp.point[1];
            const Real zeta = qp.point[2];
            const std::array<Real, 4> phi{{Real(1.0) - xi - eta - zeta, xi, eta, zeta}};

            Real u_value = Real(0.0);
            AssemblyContext::Vector3D u_gradient{{0.0, 0.0, 0.0}};
            for (std::size_t j = 0u; j < 4u; ++j) {
                u_value += solution[j] * phi[j];
                u_gradient[0] += solution[j] * p1_gradients[j][0];
                u_gradient[1] += solution[j] * p1_gradients[j][1];
                u_gradient[2] += solution[j] * p1_gradients[j][2];
            }

            for (std::size_t i = 0u; i < 4u; ++i) {
                const Real grad_dot_test =
                    u_gradient[0] * p1_gradients[i][0] +
                    u_gradient[1] * p1_gradients[i][1] +
                    u_gradient[2] * p1_gradients[i][2];
                const Real value_term = (u_value * u_value +
                                         u_value * u_value * u_value * Real(0.125)) * phi[i];
                const Real gradient_term =
                    Real(0.20) * (Real(1.0) + u_value * Real(0.50)) * grad_dot_test;
                manual.residual[i] += qp.weight * coefficient * (value_term + gradient_term);

                for (std::size_t j = 0u; j < 4u; ++j) {
                    const Real grad_trial_dot_test =
                        p1_gradients[j][0] * p1_gradients[i][0] +
                        p1_gradients[j][1] * p1_gradients[i][1] +
                        p1_gradients[j][2] * p1_gradients[i][2];
                    const Real tangent_value =
                        (Real(2.0) * u_value + Real(0.375) * u_value * u_value) *
                        phi[j] * phi[i];
                    const Real tangent_gradient =
                        Real(0.20) *
                        (Real(0.50) * phi[j] * grad_dot_test +
                         (Real(1.0) + u_value * Real(0.50)) * grad_trial_dot_test);
                    manual.tangent[i * 4u + j] +=
                        qp.weight * coefficient * (tangent_value + tangent_gradient);
                }
            }
        }
    }

    return manual;
}

void expectHighOrderCutVolumeJacobianMatchesCentralFD(
    const svmp::FE::assembly::CutIntegrationContext& cut_context,
    int marker,
    CutIntegrationSide side,
    const svmp::FE::spaces::FunctionSpace& space,
    const FormExpr& residual,
    const std::vector<Real>& solution,
    Real eps,
    Real tol)
{
    svmp::FE::forms::test::SingleTetraMeshAccess mesh;
    auto dof_map = createSingleCellDofMap(space.dofs_per_element());
    const auto n_dofs = dof_map.getNumDofs();

    FormCompiler compiler;
    auto matrix_ir = compiler.compileResidual(residual);
    auto vector_ir = compiler.compileResidual(residual);
    SymbolicNonlinearFormKernel matrix_kernel(
        std::move(matrix_ir), NonlinearKernelOutput::Both);
    SymbolicNonlinearFormKernel vector_kernel(
        std::move(vector_ir), NonlinearKernelOutput::VectorOnly);
    matrix_kernel.resolveInlinableConstitutives();
    vector_kernel.resolveInlinableConstitutives();

    svmp::FE::assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    svmp::FE::assembly::DenseMatrixView jacobian(n_dofs);
    svmp::FE::assembly::DenseVectorView residual_vector(n_dofs);
    jacobian.zero();
    residual_vector.zero();

    assembler.setCurrentSolution(solution);
    const auto assembled = assembler.assembleCutVolumes(
        mesh, cut_context, marker, side, space, space, matrix_kernel,
        &jacobian, &residual_vector, /*assemble_matrix=*/true, /*assemble_vector=*/true);
    ASSERT_EQ(assembled.elements_assembled, svmp::FE::GlobalIndex{1});

    for (svmp::FE::GlobalIndex j = 0; j < n_dofs; ++j) {
        auto solution_plus = solution;
        auto solution_minus = solution;
        solution_plus[static_cast<std::size_t>(j)] += eps;
        solution_minus[static_cast<std::size_t>(j)] -= eps;

        svmp::FE::assembly::DenseVectorView residual_plus(n_dofs);
        svmp::FE::assembly::DenseVectorView residual_minus(n_dofs);
        residual_plus.zero();
        residual_minus.zero();

        assembler.setCurrentSolution(solution_plus);
        (void)assembler.assembleCutVolumes(
            mesh, cut_context, marker, side, space, space, vector_kernel,
            nullptr, &residual_plus, /*assemble_matrix=*/false, /*assemble_vector=*/true);
        assembler.setCurrentSolution(solution_minus);
        (void)assembler.assembleCutVolumes(
            mesh, cut_context, marker, side, space, space, vector_kernel,
            nullptr, &residual_minus, /*assemble_matrix=*/false, /*assemble_vector=*/true);

        for (svmp::FE::GlobalIndex i = 0; i < n_dofs; ++i) {
            const Real finite_difference =
                (residual_plus.getVectorEntry(i) - residual_minus.getVectorEntry(i)) /
                (Real(2.0) * eps);
            EXPECT_NEAR(jacobian.getMatrixEntry(i, j), finite_difference, tol);
        }
    }
}

void expectHighOrderCutInterfaceJacobianMatchesCentralFD(
    const svmp::FE::assembly::CutIntegrationContext& cut_context,
    const svmp::FE::spaces::FunctionSpace& space,
    const FormExpr& residual,
    const std::vector<Real>& solution,
    Real eps,
    Real tol)
{
    svmp::FE::forms::test::SingleTetraMeshAccess mesh;
    auto dof_map = createSingleCellDofMap(space.dofs_per_element());
    const auto n_dofs = dof_map.getNumDofs();

    FormCompiler compiler;
    auto matrix_ir = compiler.compileResidual(residual);
    auto vector_ir = compiler.compileResidual(residual);
    SymbolicNonlinearFormKernel matrix_kernel(
        std::move(matrix_ir), NonlinearKernelOutput::Both);
    SymbolicNonlinearFormKernel vector_kernel(
        std::move(vector_ir), NonlinearKernelOutput::VectorOnly);
    matrix_kernel.resolveInlinableConstitutives();
    vector_kernel.resolveInlinableConstitutives();

    svmp::FE::assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    svmp::FE::assembly::DenseMatrixView jacobian(n_dofs);
    svmp::FE::assembly::DenseVectorView residual_vector(n_dofs);
    jacobian.zero();
    residual_vector.zero();

    assembler.setCurrentSolution(solution);
    const auto assembled = assembler.assembleCutInterfaces(
        mesh, cut_context, /*interface_marker=*/-1, space, space, matrix_kernel,
        &jacobian, &residual_vector, /*assemble_matrix=*/true, /*assemble_vector=*/true);
    ASSERT_EQ(assembled.interface_faces_assembled, svmp::FE::GlobalIndex{1});

    for (svmp::FE::GlobalIndex j = 0; j < n_dofs; ++j) {
        auto solution_plus = solution;
        auto solution_minus = solution;
        solution_plus[static_cast<std::size_t>(j)] += eps;
        solution_minus[static_cast<std::size_t>(j)] -= eps;

        svmp::FE::assembly::DenseVectorView residual_plus(n_dofs);
        svmp::FE::assembly::DenseVectorView residual_minus(n_dofs);
        residual_plus.zero();
        residual_minus.zero();

        assembler.setCurrentSolution(solution_plus);
        (void)assembler.assembleCutInterfaces(
            mesh, cut_context, /*interface_marker=*/-1, space, space, vector_kernel,
            nullptr, &residual_plus, /*assemble_matrix=*/false, /*assemble_vector=*/true);
        assembler.setCurrentSolution(solution_minus);
        (void)assembler.assembleCutInterfaces(
            mesh, cut_context, /*interface_marker=*/-1, space, space, vector_kernel,
            nullptr, &residual_minus, /*assemble_matrix=*/false, /*assemble_vector=*/true);

        for (svmp::FE::GlobalIndex i = 0; i < n_dofs; ++i) {
            const Real finite_difference =
                (residual_plus.getVectorEntry(i) - residual_minus.getVectorEntry(i)) /
                (Real(2.0) * eps);
            EXPECT_NEAR(jacobian.getMatrixEntry(i, j), finite_difference, tol);
        }
    }
}

} // namespace

TEST(CutCellForms, CutAdjacentFacetVocabularyReusesInteriorFaceOperators)
{
    constexpr int facet_set_marker = 37;
    svmp::FE::spaces::H1Space space(svmp::FE::ElementType::Tetra4, /*order=*/1);
    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");

    EXPECT_EQ(cutAdjacentFacetJump(u).toString(), jump(u).toString());
    EXPECT_EQ(cutAdjacentFacetAverage(grad(u)).toString(), avg(grad(u)).toString());
    EXPECT_EQ(cutAdjacentFacetGradientJump(u).toString(), jump(grad(u)).toString());
    EXPECT_EQ(cutAdjacentFacetHessianJump(u).toString(), jump(hessian(u)).toString());
    const auto n_minus = FormExpr::normal().minus();
    EXPECT_EQ(cutAdjacentFacetSecondNormalDerivativeJump(u).toString(),
              inner(jump(hessian(u)), outer(n_minus, n_minus)).toString());

    const auto residual = cutAdjacentFacetIntegral(
        cutAdjacentFacetJump(u) * cutAdjacentFacetJump(v) +
            cutAdjacentFacetNormalGradientJump(u) * cutAdjacentFacetJump(v),
        facet_set_marker);

    FormCompiler compiler;
    const auto ir = compiler.compileResidual(residual);
    EXPECT_TRUE(ir.hasInteriorFaceTerms());
    ASSERT_EQ(ir.terms().size(), 2u);
    for (const auto& term : ir.terms()) {
        EXPECT_EQ(term.domain, IntegralDomain::InteriorFace);
        EXPECT_EQ(term.interface_marker, facet_set_marker);
    }
    EXPECT_THROW(
        (void)cutAdjacentFacetIntegral(cutAdjacentFacetJump(u), -1),
        std::invalid_argument);
}

TEST(CutCellForms, CutAdjacentSecondNormalDerivativeJumpRequiresHessians)
{
    constexpr int facet_set_marker = 38;
    svmp::FE::spaces::H1Space space(svmp::FE::ElementType::Tetra4, /*order=*/2);
    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");

    const auto bilinear = cutAdjacentFacetIntegral(
        cutAdjacentFacetSecondNormalDerivativeJump(u) *
            cutAdjacentFacetSecondNormalDerivativeJump(v),
        facet_set_marker);

    FormCompiler compiler;
    const auto ir = compiler.compileBilinear(bilinear);
    EXPECT_TRUE(ir.hasInteriorFaceTerms());
    EXPECT_TRUE(svmp::FE::assembly::hasFlag(
        ir.requiredData(), svmp::FE::assembly::RequiredData::BasisHessians));
    ASSERT_EQ(ir.terms().size(), 1u);
    EXPECT_EQ(ir.terms().front().domain, IntegralDomain::InteriorFace);
    EXPECT_EQ(ir.terms().front().interface_marker, facet_set_marker);
}

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

TEST(CutCellForms, CutVolumeIntegralJITCompilerEmitsMarkerSideKernels)
{
    requireLLVMJITOrSkip();

    svmp::FE::spaces::H1Space space(svmp::FE::ElementType::Tetra4, /*order=*/1);
    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto residual =
        (Real(2.0) * u * v).dCutVolume(51, CutVolumeSide::Negative) +
        (Real(3.0) * u * v).dCutVolume(51, CutVolumeSide::Positive) +
        (Real(5.0) * u * v).dCutVolume(52, CutVolumeSide::Negative);

    FormCompiler compiler;
    const auto ir = compiler.compileResidual(residual);
    auto jit = svmp::FE::forms::jit::JITCompiler::getOrCreate(
        svmp::FE::forms::test::makeUnitTestJITOptions());
    ASSERT_NE(jit, nullptr);

    const auto compiled = jit->compile(ir);
    ASSERT_TRUE(compiled.ok) << compiled.message;

    bool saw_51_negative = false;
    bool saw_51_positive = false;
    bool saw_52_negative = false;
    for (const auto& kernel : compiled.kernels) {
        ASSERT_NE(kernel.address, std::uintptr_t{0});
        EXPECT_EQ(kernel.domain, IntegralDomain::CutVolume);
        if (kernel.interface_marker == 51 &&
            kernel.cut_volume_side == CutVolumeSide::Negative) {
            saw_51_negative = true;
        }
        if (kernel.interface_marker == 51 &&
            kernel.cut_volume_side == CutVolumeSide::Positive) {
            saw_51_positive = true;
        }
        if (kernel.interface_marker == 52 &&
            kernel.cut_volume_side == CutVolumeSide::Negative) {
            saw_52_negative = true;
        }
    }

    EXPECT_TRUE(saw_51_negative);
    EXPECT_TRUE(saw_51_positive);
    EXPECT_TRUE(saw_52_negative);
}

TEST(CutCellForms, SymbolicTangentPreservesCutVolumeMeasure)
{
    svmp::FE::spaces::H1Space space(svmp::FE::ElementType::Tetra4, /*order=*/1);
    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto residual =
        ((u * u + Real(0.25) * u) * v).dCutVolume(73, CutVolumeSide::Positive);

    FormCompiler compiler;
    auto ir = compiler.compileResidual(residual);
    SymbolicNonlinearFormKernel kernel(std::move(ir), NonlinearKernelOutput::Both);

    ASSERT_NO_THROW(kernel.resolveInlinableConstitutives());
    const auto& tangent = kernel.tangentIR();
    ASSERT_TRUE(tangent.isCompiled());
    EXPECT_TRUE(tangent.hasCutVolumeTerms());
    EXPECT_FALSE(tangent.hasCellTerms());
    ASSERT_FALSE(tangent.terms().empty());
    for (const auto& term : tangent.terms()) {
        EXPECT_EQ(term.domain, IntegralDomain::CutVolume);
        EXPECT_EQ(term.interface_marker, 73);
        EXPECT_EQ(term.cut_volume_side, CutVolumeSide::Positive);
    }
}

TEST(CutCellForms, HighOrderCutVolumeTangentMatchesFixedGeometryFiniteDifference)
{
    constexpr int marker = 81;
    auto rule = makeReferenceTetraCutRule(
        CutIntegrationSide::Negative,
        {
            {{{0.10, 0.20, 0.15}}, {{0.0, 0.0, 1.0}}, Real(0.010)},
            {{{0.28, 0.12, 0.18}}, {{0.0, 0.0, 1.0}}, Real(0.012)},
            {{{0.18, 0.30, 0.11}}, {{0.0, 0.0, 1.0}}, Real(0.014)},
            {{{0.12, 0.15, 0.32}}, {{0.0, 0.0, 1.0}}, Real(0.011)},
            {{{0.24, 0.18, 0.20}}, {{0.0, 0.0, 1.0}}, Real(0.009)}
        });
    markRuleAsHighOrderImplicit(rule,
                                marker,
                                "fixed-geometry-volume",
                                /*quadrature_order=*/4);

    svmp::FE::assembly::CutCellAssemblyMetadata metadata;
    metadata.parent_entity = 0;
    metadata.volume_fraction = rule.volume_fraction;
    metadata.side = CutIntegrationSide::Negative;
    metadata.embedded_normal = rule.points.front().normal;
    metadata.cut_topology_id = rule.provenance.cut_topology_id;
    metadata.cut_topology_revision = rule.provenance.cut_topology_revision;

    svmp::FE::assembly::CutIntegrationContext cut_context;
    cut_context.addVolumeRule(std::move(metadata), std::move(rule));

    svmp::FE::spaces::H1Space space(svmp::FE::ElementType::Tetra4, /*order=*/2);
    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto x = FormExpr::coordinate();
    const auto terminals = cutCellTerminals();
    const auto coefficient =
        FormExpr::constant(Real(1.15)) +
        terminals.volume_fraction * Real(0.35) +
        terminals.side_indicator * Real(0.075) +
        x.component(0) * Real(0.20) +
        x.component(1) * Real(0.10);
    const auto residual =
        (coefficient *
         ((u * u + u * u * u * Real(0.08)) * v +
          Real(0.12) * (FormExpr::constant(Real(1.0)) + u * Real(0.25)) *
              inner(grad(u), grad(v)))).dCutVolume(marker, CutVolumeSide::Negative);

    expectHighOrderCutVolumeJacobianMatchesCentralFD(
        cut_context,
        marker,
        CutIntegrationSide::Negative,
        space,
        residual,
        deterministicScalarCoefficients(space.dofs_per_element()),
        /*eps=*/Real(2.0e-6),
        /*tol=*/Real(2.0e-6));
}

TEST(CutCellForms, HighOrderCutInterfaceTangentMatchesFixedGeometryFiniteDifference)
{
    constexpr int marker = 82;
    svmp::FE::assembly::CutIntegrationContext cut_context;
    cut_context.addInterfaceRule(makeHighOrderReferenceTetraInterfaceRule(marker));

    svmp::FE::spaces::H1Space space(svmp::FE::ElementType::Tetra4, /*order=*/2);
    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto x = FormExpr::coordinate();
    const auto n = FormExpr::normal();
    const auto coefficient =
        FormExpr::constant(Real(0.80)) +
        x.component(0) * Real(0.15) +
        x.component(1) * Real(0.05) +
        n.component(0) * Real(0.10);
    const auto residual =
        (coefficient * (u * u + u * Real(0.20)) * v +
         Real(0.06) * (FormExpr::constant(Real(1.0)) + u * Real(0.15)) *
             inner(grad(u), n) * v).dI(marker);

    expectHighOrderCutInterfaceJacobianMatchesCentralFD(
        cut_context,
        space,
        residual,
        deterministicScalarCoefficients(space.dofs_per_element()),
        /*eps=*/Real(2.0e-6),
        /*tol=*/Real(2.0e-6));
}

TEST(CutCellForms, ZeroTangentProbeSupportsMixedCutVolumeBlocks)
{
    requireLLVMJITOrSkip();

    auto base = std::make_shared<svmp::FE::spaces::H1Space>(
        svmp::FE::ElementType::Tetra4, /*order=*/1);
    svmp::FE::spaces::ProductSpace velocity_space(base, /*components=*/3);
    svmp::FE::spaces::H1Space pressure_space(svmp::FE::ElementType::Tetra4, /*order=*/1);

    const auto u = TrialFunction(velocity_space, "u");
    const auto q = TestFunction(pressure_space, "q");
    const auto residual =
        (FormExpr::constant(0.0) * u.component(0) * q).dCutVolume(91, CutVolumeSide::Negative);

    FormCompiler compiler;
    auto ir = compiler.compileResidual(residual);
    SymbolicNonlinearFormKernel kernel(std::move(ir), NonlinearKernelOutput::Both);

    ASSERT_NO_THROW(kernel.resolveInlinableConstitutives());
    ASSERT_TRUE(kernel.tangentIR().isCompiled());

    auto jit = svmp::FE::forms::jit::JITCompiler::getOrCreate(
        svmp::FE::forms::test::makeUnitTestJITOptions());
    ASSERT_NE(jit, nullptr);
    const auto compiled = jit->compile(kernel.tangentIR());
    ASSERT_TRUE(compiled.ok) << compiled.message;
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

TEST(CutCellForms, CutDomainAssemblerUsesCutRulesForResidualAndTangentKernels)
{
    using svmp::FE::assembly::CutCellAssemblyMetadata;
    using svmp::FE::assembly::CutDomainAssemblyOptions;
    using svmp::FE::assembly::CutIntegrationAssemblyPath;
    using svmp::FE::assembly::CutIntegrationBinding;
    using svmp::FE::assembly::CutIntegrationContext;

    CutIntegrationContext cut_context;
    auto negative_rule = makeReferenceTetraCutRule(
        CutIntegrationSide::Negative,
        {
            {{{0.10, 0.20, 0.15}}, {{ 1.0, 0.0, 0.0}}, Real(0.018)},
            {{{0.24, 0.12, 0.18}}, {{ 1.0, 0.0, 0.0}}, Real(0.027)}
        });
    auto positive_rule = makeReferenceTetraCutRule(
        CutIntegrationSide::Positive,
        {
            {{{0.18, 0.16, 0.20}}, {{-1.0, 0.0, 0.0}}, Real(0.015)},
            {{{0.12, 0.28, 0.10}}, {{-1.0, 0.0, 0.0}}, Real(0.020)}
        });

    CutCellAssemblyMetadata negative_metadata;
    negative_metadata.parent_entity = 0;
    negative_metadata.volume_fraction = negative_rule.volume_fraction;
    negative_metadata.side = CutIntegrationSide::Negative;
    negative_metadata.embedded_normal = negative_rule.points.front().normal;
    negative_metadata.cut_topology_revision = 11u;
    cut_context.addVolumeRule(negative_metadata, std::move(negative_rule));

    CutCellAssemblyMetadata positive_metadata;
    positive_metadata.parent_entity = 0;
    positive_metadata.volume_fraction = positive_rule.volume_fraction;
    positive_metadata.side = CutIntegrationSide::Positive;
    positive_metadata.embedded_normal = positive_rule.points.front().normal;
    positive_metadata.cut_topology_revision = 11u;
    cut_context.addVolumeRule(positive_metadata, std::move(positive_rule));

    for (std::size_t i = 0u; i < cut_context.volumeRules().size(); ++i) {
        CutIntegrationBinding binding;
        binding.parent_entity = 0;
        binding.kind = CutQuadratureKind::Volume;
        binding.side = cut_context.volumeRules()[i].side;
        binding.cut_topology_revision = 11u;
        binding.visible_to_paths = {
            CutIntegrationAssemblyPath::Standard,
            CutIntegrationAssemblyPath::AD,
            CutIntegrationAssemblyPath::SymbolicTangent};
        cut_context.addBinding(std::move(binding));
    }

    svmp::FE::spaces::H1Space space(svmp::FE::ElementType::Tetra4, /*order=*/1);
    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto terminals = cutCellTerminals();
    const auto cut_coefficient =
        FormExpr::constant(Real(1.10)) +
        terminals.volume_fraction * Real(0.70) +
        terminals.side_indicator * Real(0.05) +
        terminals.embedded_normal.component(0) * Real(0.20) +
        terminals.embedded_normal.component(1) * Real(0.10) +
        terminals.stabilization_scale * Real(0.125) +
        terminals.quadrature_weight_sensitivity * Real(0.40);
    const auto residual =
        (cut_coefficient *
         ((u * u + (u * u * u) * Real(0.125)) * v +
          Real(0.20) * (FormExpr::constant(Real(1.0)) + u * Real(0.50)) *
              inner(grad(u), grad(v)))).dx();

    FormCompiler compiler;
    auto ad_ir = compiler.compileResidual(residual);
    auto symbolic_ir = compiler.compileResidual(residual);
    NonlinearFormKernel ad_kernel(std::move(ad_ir), ADMode::Forward, NonlinearKernelOutput::Both);
    SymbolicNonlinearFormKernel symbolic_kernel(
        std::move(symbolic_ir), NonlinearKernelOutput::Both);
    symbolic_kernel.resolveInlinableConstitutives();

    const std::vector<Real> solution = {Real(0.16), Real(-0.04), Real(0.09), Real(0.02)};
    std::vector<JITConstants> cut_rule_constants;
    cut_rule_constants.reserve(cut_context.volumeRules().size());
    for (std::size_t rule_index = 0u; rule_index < cut_context.volumeRules().size(); ++rule_index) {
        auto parameters = cutCellParametersForRule(
            cut_context.volumeRules()[rule_index],
            CutCellParameterSlots{},
            Real(0.20) + Real(0.05) * static_cast<Real>(rule_index),
            Real(0.01));
        cut_rule_constants.emplace_back(parameters.begin(), parameters.end());
    }
    const auto make_builder = [&](RequiredData required) {
        return [&, required](const svmp::FE::assembly::CutRuleAssemblyRequest& request,
                             AssemblyContext& ctx) {
            ASSERT_NE(request.rule, nullptr);
            populateP1ReferenceTetraCutContext(
                ctx, *request.rule, space, required,
                cut_rule_constants.at(request.rule_index),
                solution);
        };
    };

    CutDomainAssemblyOptions options;
    options.path = CutIntegrationAssemblyPath::AD;
    options.include_interface_rules = false;
    const auto ad_summary = svmp::FE::assembly::assembleCutDomains(
        cut_context, ad_kernel, make_builder(ad_kernel.getRequiredData()), options);

    options.path = CutIntegrationAssemblyPath::SymbolicTangent;
    const auto symbolic_summary = svmp::FE::assembly::assembleCutDomains(
        cut_context, symbolic_kernel, make_builder(symbolic_kernel.getRequiredData()), options);

    ASSERT_EQ(ad_summary.volume_rule_count, std::size_t{2});
    ASSERT_EQ(symbolic_summary.volume_rule_count, std::size_t{2});
    ASSERT_TRUE(ad_summary.hasMatrix());
    ASSERT_TRUE(ad_summary.hasVector());
    ASSERT_TRUE(symbolic_summary.hasMatrix());
    ASSERT_TRUE(symbolic_summary.hasVector());

    const auto manual = manualReferenceTetraCutResidualTangent(cut_context, solution);
    ASSERT_EQ(ad_summary.total_output.local_vector.size(), manual.residual.size());
    ASSERT_EQ(ad_summary.total_output.local_matrix.size(), manual.tangent.size());
    ASSERT_EQ(symbolic_summary.total_output.local_vector.size(), manual.residual.size());
    ASSERT_EQ(symbolic_summary.total_output.local_matrix.size(), manual.tangent.size());

    for (std::size_t i = 0u; i < manual.residual.size(); ++i) {
        EXPECT_NEAR(ad_summary.total_output.local_vector[i], manual.residual[i], Real(1.0e-13));
        EXPECT_NEAR(symbolic_summary.total_output.local_vector[i], manual.residual[i], Real(1.0e-13));
    }
    for (std::size_t i = 0u; i < manual.tangent.size(); ++i) {
        EXPECT_NEAR(ad_summary.total_output.local_matrix[i], manual.tangent[i], Real(1.0e-12));
        EXPECT_NEAR(symbolic_summary.total_output.local_matrix[i], manual.tangent[i], Real(1.0e-12));
    }

    options.path = CutIntegrationAssemblyPath::JIT;
    const auto skipped_summary = svmp::FE::assembly::assembleCutDomains(
        cut_context, ad_kernel, make_builder(ad_kernel.getRequiredData()), options);
    EXPECT_EQ(skipped_summary.volume_rule_count, std::size_t{0});
    EXPECT_EQ(skipped_summary.skipped_rule_count, cut_context.volumeRules().size());
    EXPECT_FALSE(skipped_summary.hasMatrix());
    EXPECT_FALSE(skipped_summary.hasVector());
}

TEST(CutCellForms, CutVolumeFormTermsFilterByMarkerAndSide)
{
    using svmp::FE::assembly::CutCellAssemblyMetadata;
    using svmp::FE::assembly::CutDomainAssemblyOptions;
    using svmp::FE::assembly::CutIntegrationContext;

    CutIntegrationContext cut_context;
    auto add_volume_rule =
        [&](int marker, CutIntegrationSide side, CutQuadratureRule rule) {
            rule.side = side;
            rule.provenance.marker = marker;
            rule.provenance.parent_entity = 0;

            CutCellAssemblyMetadata metadata;
            metadata.parent_entity = 0;
            metadata.volume_fraction = rule.volume_fraction;
            metadata.side = side;
            cut_context.addVolumeRule(std::move(metadata), std::move(rule));
        };

    auto negative_rule = makeReferenceTetraCutRule(
        CutIntegrationSide::Negative,
        {
            {{{0.10, 0.20, 0.15}}, {{1.0, 0.0, 0.0}}, Real(0.018)},
            {{{0.24, 0.12, 0.18}}, {{1.0, 0.0, 0.0}}, Real(0.027)}
        });
    auto positive_rule = makeReferenceTetraCutRule(
        CutIntegrationSide::Positive,
        {
            {{{0.18, 0.16, 0.20}}, {{-1.0, 0.0, 0.0}}, Real(0.015)},
            {{{0.12, 0.28, 0.10}}, {{-1.0, 0.0, 0.0}}, Real(0.020)}
        });
    auto other_marker_rule = makeReferenceTetraCutRule(
        CutIntegrationSide::Negative,
        {
            {{{0.20, 0.10, 0.15}}, {{1.0, 0.0, 0.0}}, Real(0.050)}
        });

    add_volume_rule(51, CutIntegrationSide::Negative, std::move(negative_rule));
    add_volume_rule(51, CutIntegrationSide::Positive, std::move(positive_rule));
    add_volume_rule(52, CutIntegrationSide::Negative, std::move(other_marker_rule));

    svmp::FE::spaces::H1Space space(svmp::FE::ElementType::Tetra4, /*order=*/1);
    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto residual =
        (Real(2.0) * u * v).dCutVolume(51, CutVolumeSide::Negative) +
        (Real(3.0) * u * v).dCutVolume(51, CutVolumeSide::Positive) +
        (Real(11.0) * u * v).dCutVolume(52, CutVolumeSide::Negative);

    FormCompiler compiler;
    auto ir = compiler.compileResidual(residual);
    NonlinearFormKernel kernel(std::move(ir), ADMode::Forward, NonlinearKernelOutput::Both);
    auto jit_ir = compiler.compileResidual(residual);
    auto jit_fallback = std::make_shared<NonlinearFormKernel>(
        std::move(jit_ir), ADMode::Forward, NonlinearKernelOutput::Both);
    JITOptions jit_options;
    jit_options.enable = true;
    jit_options.specialization.enable = false;
    svmp::FE::forms::jit::JITKernelWrapper jit_kernel(jit_fallback, jit_options);

    const std::vector<Real> solution = {Real(0.16), Real(-0.04), Real(0.09), Real(0.02)};
    const JITConstants constants;
    CutDomainAssemblyOptions options;
    options.include_interface_rules = false;
    options.volume_marker = 51;

    auto assemble_with = [&](svmp::FE::assembly::AssemblyKernel& active_kernel) {
        return svmp::FE::assembly::assembleCutDomains(
            cut_context,
            active_kernel,
            [&](const svmp::FE::assembly::CutRuleAssemblyRequest& request,
                AssemblyContext& ctx) {
                ASSERT_NE(request.rule, nullptr);
                populateP1ReferenceTetraCutContext(
                    ctx, *request.rule, space, active_kernel.getRequiredData(), constants, solution);
            },
            options);
    };

    const auto summary = assemble_with(kernel);
    const auto jit_summary = assemble_with(jit_kernel);

    std::vector<Real> expected_vector(4u, Real(0.0));
    std::vector<Real> expected_matrix(16u, Real(0.0));
    auto accumulate_expected = [&](const CutQuadratureRule& rule, Real coefficient) {
        for (const auto& qp : rule.points) {
            const Real xi = qp.point[0];
            const Real eta = qp.point[1];
            const Real zeta = qp.point[2];
            const std::array<Real, 4> phi{{Real(1.0) - xi - eta - zeta,
                                           xi,
                                           eta,
                                           zeta}};
            Real u_q = Real(0.0);
            for (std::size_t j = 0u; j < phi.size(); ++j) {
                u_q += phi[j] * solution[j];
            }
            for (std::size_t i = 0u; i < phi.size(); ++i) {
                expected_vector[i] += coefficient * qp.weight * phi[i] * u_q;
                for (std::size_t j = 0u; j < phi.size(); ++j) {
                    expected_matrix[i * phi.size() + j] +=
                        coefficient * qp.weight * phi[i] * phi[j];
                }
            }
        }
    };
    accumulate_expected(cut_context.volumeRules()[0], Real(2.0));
    accumulate_expected(cut_context.volumeRules()[1], Real(3.0));

    auto expect_summary = [&](const auto& actual) {
        ASSERT_EQ(actual.volume_rule_count, std::size_t{2});
        EXPECT_EQ(actual.skipped_rule_count, std::size_t{1});
        ASSERT_TRUE(actual.hasVector());
        ASSERT_TRUE(actual.hasMatrix());
        ASSERT_EQ(actual.total_output.local_vector.size(), expected_vector.size());
        ASSERT_EQ(actual.total_output.local_matrix.size(), expected_matrix.size());
        for (std::size_t i = 0u; i < expected_vector.size(); ++i) {
            EXPECT_NEAR(actual.total_output.local_vector[i], expected_vector[i], Real(1.0e-13));
        }
        for (std::size_t i = 0u; i < expected_matrix.size(); ++i) {
            EXPECT_NEAR(actual.total_output.local_matrix[i], expected_matrix[i], Real(1.0e-13));
        }
    };
    expect_summary(summary);
    expect_summary(jit_summary);
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
