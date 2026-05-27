#include "Forms/CutCellForms.h"

#include <gtest/gtest.h>

#include "Assembly/AssemblyContext.h"
#include "Assembly/CutDomainAssembler.h"
#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Basis/NodeOrderingConventions.h"
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

[[nodiscard]] bool containsFormExprType(const FormExprNode& node,
                                        FormExprType type)
{
    if (node.type() == type) {
        return true;
    }
    for (const auto& child : node.childrenShared()) {
        if (child && containsFormExprType(*child, type)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] FormExprNode::SpaceSignature spaceSignatureFor(
    const svmp::FE::spaces::FunctionSpace& space)
{
    FormExprNode::SpaceSignature sig;
    sig.space_type = space.space_type();
    sig.field_type = space.field_type();
    sig.continuity = space.continuity();
    sig.value_dimension = space.value_dimension();
    sig.topological_dimension = space.topological_dimension();
    sig.polynomial_order = space.polynomial_order();
    sig.element_type = space.element_type();
    return sig;
}

[[nodiscard]] bool spaceSignaturesEqual(const FormExprNode::SpaceSignature& a,
                                        const FormExprNode::SpaceSignature& b)
{
    return a.space_type == b.space_type &&
           a.field_type == b.field_type &&
           a.continuity == b.continuity &&
           a.value_dimension == b.value_dimension &&
           a.topological_dimension == b.topological_dimension &&
           a.polynomial_order == b.polynomial_order &&
           a.element_type == b.element_type;
}

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

[[nodiscard]] CutQuadratureRule makeManyPointReferenceTetraCutRule(
    CutIntegrationSide side,
    std::size_t points_per_axis = 4u)
{
    std::vector<svmp::FE::geometry::CutQuadraturePoint> points;
    points.reserve(points_per_axis * points_per_axis * points_per_axis);
    const Real weight = Real(0.0015);
    for (std::size_t i = 0u; i < points_per_axis; ++i) {
        for (std::size_t j = 0u; j < points_per_axis; ++j) {
            for (std::size_t k = 0u; k < points_per_axis; ++k) {
                const Real xi = Real(0.05) + Real(0.10) * static_cast<Real>(i);
                const Real eta = Real(0.04) + Real(0.08) * static_cast<Real>(j);
                const Real zeta = Real(0.03) + Real(0.06) * static_cast<Real>(k);
                points.push_back({{{xi, eta, zeta}}, {{0.0, 0.0, 1.0}}, weight});
            }
        }
    }
    return makeReferenceTetraCutRule(side, std::move(points));
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

[[nodiscard]] CutQuadratureRule makeManyPointReferenceTetraInterfaceRule(int marker,
                                                                         std::size_t points_per_axis = 4u)
{
    CutQuadratureRule rule;
    rule.kind = CutQuadratureKind::Interface;
    rule.side = CutIntegrationSide::Interface;
    rule.parent_measure = Real(1.0) / Real(6.0);
    rule.points.reserve(points_per_axis * points_per_axis * points_per_axis);
    const Real weight = Real(0.00125);
    for (std::size_t i = 0u; i < points_per_axis; ++i) {
        for (std::size_t j = 0u; j < points_per_axis; ++j) {
            for (std::size_t k = 0u; k < points_per_axis; ++k) {
                const Real xi = Real(0.05) + Real(0.10) * static_cast<Real>(i);
                const Real eta = Real(0.04) + Real(0.08) * static_cast<Real>(j);
                const Real zeta = Real(0.03) + Real(0.06) * static_cast<Real>(k);
                rule.points.push_back({{{xi, eta, zeta}}, {{1.0, 0.0, 0.0}}, weight});
            }
        }
    }
    for (const auto& qp : rule.points) {
        rule.measure += qp.weight;
    }
    rule.volume_fraction = Real(0.0);
    rule.exact_for_constants = true;
    rule.provenance_id = "many-point-fixed-geometry-interface";
    markRuleAsHighOrderImplicit(rule,
                                marker,
                                "many-point-fixed-geometry-interface",
                                /*quadrature_order=*/6);
    return rule;
}

[[nodiscard]] Real quadraticLevelSetAt(const std::array<Real, 3>& x)
{
    return Real(-0.18) +
           x[0] +
           Real(0.35) * x[1] -
           Real(0.25) * x[2] +
           Real(0.40) * x[0] * x[0] +
           Real(0.20) * x[1] * x[1] -
           Real(0.10) * x[2] * x[2] +
           Real(0.15) * x[0] * x[1];
}

[[nodiscard]] AssemblyContext::Point3D pointOnQuadraticLevelSet(Real y, Real z)
{
    const Real a = Real(0.40);
    const Real b = Real(1.0) + Real(0.15) * y;
    const Real c =
        Real(-0.18) +
        Real(0.35) * y -
        Real(0.25) * z +
        Real(0.20) * y * y -
        Real(0.10) * z * z;
    const Real disc = b * b - Real(4.0) * a * c;
    if (!(disc >= Real(0.0))) {
        throw std::runtime_error("pointOnQuadraticLevelSet: no real root");
    }
    const Real x = (-b + std::sqrt(disc)) / (Real(2.0) * a);
    if (!(x >= Real(0.0) && x + y + z <= Real(1.0))) {
        throw std::runtime_error(
            "pointOnQuadraticLevelSet: root outside reference tetrahedron");
    }
    return {{x, y, z}};
}

[[nodiscard]] std::vector<Real> quadraticTetraP2LevelSetCoefficients(
    const svmp::FE::spaces::FunctionSpace& space)
{
    const auto n_dofs = space.dofs_per_element();
    if (n_dofs != svmp::FE::basis::ReferenceNodeLayout::num_nodes(svmp::FE::ElementType::Tetra10)) {
        throw std::runtime_error(
            "quadraticTetraP2LevelSetCoefficients expects a Tetra P2 scalar space");
    }

    std::vector<Real> coeffs(n_dofs, Real(0.0));
    for (std::size_t i = 0u; i < coeffs.size(); ++i) {
        const auto xi =
            svmp::FE::basis::ReferenceNodeLayout::get_node_coords(svmp::FE::ElementType::Tetra10, i);
        coeffs[i] = quadraticLevelSetAt({xi[0], xi[1], xi[2]});
    }
    return coeffs;
}

[[nodiscard]] AssemblyContext::Vector3D normalized3(const AssemblyContext::Vector3D& v)
{
    const Real n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (!(n > Real(1.0e-14))) {
        throw std::runtime_error("normalized3: degenerate vector");
    }
    return {{v[0] / n, v[1] / n, v[2] / n}};
}

[[nodiscard]] Real meanCurvatureFromGradientHessian(
    const AssemblyContext::Vector3D& gradient,
    const AssemblyContext::Matrix3x3& hessian)
{
    const Real g2 = gradient[0] * gradient[0] +
                    gradient[1] * gradient[1] +
                    gradient[2] * gradient[2];
    if (!(g2 > Real(1.0e-28))) {
        throw std::runtime_error("meanCurvatureFromGradientHessian: degenerate gradient");
    }
    const Real g = std::sqrt(g2);
    const Real trace = hessian[0][0] + hessian[1][1] + hessian[2][2];
    Real g_h_g = Real(0.0);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            g_h_g += gradient[static_cast<std::size_t>(r)] *
                     hessian[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] *
                     gradient[static_cast<std::size_t>(c)];
        }
    }
    return trace / g - g_h_g / (g2 * g);
}

[[nodiscard]] CutQuadratureRule makeQuadraticLevelSetReferenceTetraInterfaceRule(
    int marker,
    const svmp::FE::spaces::FunctionSpace& space,
    std::span<const Real> level_set_coefficients,
    int perturb_dof = -1,
    Real perturbation = Real(0.0),
    bool move_points_with_level_set = false)
{
    const auto n_dofs = space.dofs_per_element();
    if (level_set_coefficients.size() != n_dofs) {
        throw std::runtime_error(
            "makeQuadraticLevelSetReferenceTetraInterfaceRule: coefficient size mismatch");
    }
    if (perturb_dof >= static_cast<int>(n_dofs)) {
        throw std::runtime_error(
            "makeQuadraticLevelSetReferenceTetraInterfaceRule: perturbation DOF out of range");
    }

    const std::array<AssemblyContext::Point3D, 4> qpts{{
        pointOnQuadraticLevelSet(Real(0.17), Real(0.11)),
        pointOnQuadraticLevelSet(Real(0.28), Real(0.09)),
        pointOnQuadraticLevelSet(Real(0.12), Real(0.16)),
        pointOnQuadraticLevelSet(Real(0.19), Real(0.25)),
    }};
    const std::array<Real, 4> base_weights{{
        Real(0.019),
        Real(0.017),
        Real(0.016),
        Real(0.013),
    }};

    CutQuadratureRule rule;
    rule.kind = CutQuadratureKind::Interface;
    rule.side = CutIntegrationSide::Interface;
    rule.parent_measure = Real(1.0) / Real(6.0);
    rule.points.reserve(qpts.size());

    const auto& basis = space.element().basis();
    std::vector<Real> values;
    std::vector<svmp::FE::basis::Gradient> gradients;
    std::vector<svmp::FE::basis::Hessian> hessians;
    for (std::size_t q = 0u; q < qpts.size(); ++q) {
        auto qp = qpts[q];
        svmp::FE::math::Vector<Real, 3> xi{qp[0], qp[1], qp[2]};
        basis.evaluate_all(xi, values, gradients, hessians);
        if (values.size() != n_dofs ||
            gradients.size() != n_dofs ||
            hessians.size() != n_dofs) {
            throw std::runtime_error(
                "makeQuadraticLevelSetReferenceTetraInterfaceRule: basis size mismatch");
        }

        AssemblyContext::Vector3D gradient{{0.0, 0.0, 0.0}};
        AssemblyContext::Matrix3x3 hessian{};
        for (std::size_t a = 0u; a < n_dofs; ++a) {
            const Real coeff = level_set_coefficients[a];
            for (int d = 0; d < 3; ++d) {
                gradient[static_cast<std::size_t>(d)] +=
                    coeff * gradients[a][static_cast<std::size_t>(d)];
            }
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    hessian[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] +=
                        coeff * hessians[a](static_cast<std::size_t>(r),
                                             static_cast<std::size_t>(c));
                }
            }
        }

        AssemblyContext::Vector3D perturbed_gradient = gradient;
        Real weight = base_weights[q];
        if (perturb_dof >= 0 && perturbation != Real(0.0)) {
            const auto p = static_cast<std::size_t>(perturb_dof);
            const auto normal = normalized3(gradient);
            for (int d = 0; d < 3; ++d) {
                perturbed_gradient[static_cast<std::size_t>(d)] +=
                    perturbation * gradients[p][static_cast<std::size_t>(d)];
            }
            const Real grad_norm =
                std::sqrt(gradient[0] * gradient[0] +
                          gradient[1] * gradient[1] +
                          gradient[2] * gradient[2]);
            const Real normal_speed = -perturbation * values[p] / grad_norm;
            const Real curvature = meanCurvatureFromGradientHessian(gradient, hessian);
            weight *= Real(1.0) + curvature * normal_speed;
            if (move_points_with_level_set) {
                for (int d = 0; d < 3; ++d) {
                    qp[static_cast<std::size_t>(d)] +=
                        normal_speed * normal[static_cast<std::size_t>(d)];
                }
                xi = svmp::FE::math::Vector<Real, 3>{qp[0], qp[1], qp[2]};
                basis.evaluate_all(xi, values, gradients, hessians);
                if (values.size() != n_dofs ||
                    gradients.size() != n_dofs ||
                    hessians.size() != n_dofs) {
                    throw std::runtime_error(
                        "makeQuadraticLevelSetReferenceTetraInterfaceRule: moved-point basis size mismatch");
                }
                perturbed_gradient = {{0.0, 0.0, 0.0}};
                for (std::size_t a = 0u; a < n_dofs; ++a) {
                    const Real coeff =
                        level_set_coefficients[a] +
                        (a == p ? perturbation : Real(0.0));
                    for (int d = 0; d < 3; ++d) {
                        perturbed_gradient[static_cast<std::size_t>(d)] +=
                            coeff * gradients[a][static_cast<std::size_t>(d)];
                    }
                }
            }
        }

        if (!(weight > Real(0.0))) {
            throw std::runtime_error(
                "makeQuadraticLevelSetReferenceTetraInterfaceRule: non-positive perturbed weight");
        }
        rule.points.push_back({qp, normalized3(perturbed_gradient), weight});
        rule.measure += weight;
    }
    rule.volume_fraction = Real(0.0);
    rule.exact_for_constants = true;
    rule.provenance_id = "quadrature-local-level-set-shape-interface";
    markRuleAsHighOrderImplicit(rule,
                                marker,
                                "quadrature-local-level-set-shape-interface",
                                /*quadrature_order=*/4);
    return rule;
}

[[nodiscard]] svmp::FE::assembly::CutIntegrationContext
makeQuadraticLevelSetReferenceTetraInterfaceContext(
    int marker,
    const svmp::FE::spaces::FunctionSpace& space,
    std::span<const Real> level_set_coefficients,
    int perturb_dof = -1,
    Real perturbation = Real(0.0),
    bool move_points_with_level_set = false)
{
    svmp::FE::assembly::CutIntegrationContext cut_context;
    cut_context.addInterfaceRule(
        makeQuadraticLevelSetReferenceTetraInterfaceRule(
            marker,
            space,
            level_set_coefficients,
            perturb_dof,
            perturbation,
            move_points_with_level_set));
    return cut_context;
}

[[nodiscard]] Real levelSetGradientNormAtReferencePoint(
    const svmp::FE::spaces::FunctionSpace& space,
    std::span<const Real> level_set_coefficients,
    const AssemblyContext::Point3D& point)
{
    const auto n_dofs = space.dofs_per_element();
    if (level_set_coefficients.size() != n_dofs) {
        throw std::runtime_error(
            "levelSetGradientNormAtReferencePoint: coefficient size mismatch");
    }

    const auto& basis = space.element().basis();
    std::vector<svmp::FE::basis::Gradient> gradients;
    svmp::FE::math::Vector<Real, 3> xi{point[0], point[1], point[2]};
    basis.evaluate_gradients(xi, gradients);
    if (gradients.size() != n_dofs) {
        throw std::runtime_error(
            "levelSetGradientNormAtReferencePoint: basis gradient size mismatch");
    }

    AssemblyContext::Vector3D gradient{{0.0, 0.0, 0.0}};
    for (std::size_t a = 0u; a < n_dofs; ++a) {
        for (int d = 0; d < 3; ++d) {
            gradient[static_cast<std::size_t>(d)] +=
                level_set_coefficients[a] *
                gradients[a][static_cast<std::size_t>(d)];
        }
    }
    return std::sqrt(gradient[0] * gradient[0] +
                     gradient[1] * gradient[1] +
                     gradient[2] * gradient[2]);
}

[[nodiscard]] Real basisValueAtReferencePoint(
    const svmp::FE::spaces::FunctionSpace& space,
    std::size_t dof,
    const AssemblyContext::Point3D& point)
{
    const auto n_dofs = space.dofs_per_element();
    if (dof >= n_dofs) {
        throw std::runtime_error("basisValueAtReferencePoint: DOF out of range");
    }

    const auto& basis = space.element().basis();
    std::vector<Real> values;
    svmp::FE::math::Vector<Real, 3> xi{point[0], point[1], point[2]};
    basis.evaluate_values(xi, values);
    if (values.size() != n_dofs) {
        throw std::runtime_error(
            "basisValueAtReferencePoint: basis value size mismatch");
    }
    return values[dof];
}

[[nodiscard]] svmp::FE::assembly::CutIntegrationContext
makeQuadraticLevelSetReferenceTetraVolumeShapeFDContext(
    int marker,
    CutIntegrationSide side,
    const svmp::FE::spaces::FunctionSpace& space,
    std::span<const Real> level_set_coefficients,
    int perturb_dof = -1,
    Real perturbation = Real(0.0))
{
    auto volume_rule = makeManyPointReferenceTetraCutRule(side);
    markRuleAsHighOrderImplicit(volume_rule,
                                marker,
                                "quadrature-local-level-set-shape-volume",
                                /*quadrature_order=*/4);

    const auto interface_rule =
        makeQuadraticLevelSetReferenceTetraInterfaceRule(
            marker, space, level_set_coefficients);

    if (perturb_dof >= 0 && perturbation != Real(0.0)) {
        const auto p = static_cast<std::size_t>(perturb_dof);
        const Real side_sign =
            side == CutIntegrationSide::Negative ? Real{-1.0} : Real{1.0};
        for (const auto& qp : interface_rule.points) {
            const Real basis_value =
                basisValueAtReferencePoint(space, p, qp.point);
            const Real grad_norm =
                levelSetGradientNormAtReferencePoint(
                    space, level_set_coefficients, qp.point);
            if (!(grad_norm > Real(1.0e-14))) {
                throw std::runtime_error(
                    "makeQuadraticLevelSetReferenceTetraVolumeShapeFDContext: degenerate level-set gradient");
            }
            const Real delta_weight =
                side_sign * perturbation * basis_value * qp.weight / grad_norm;
            volume_rule.points.push_back(
                svmp::FE::geometry::CutQuadraturePoint{
                    qp.point, qp.normal, delta_weight});
            volume_rule.measure += delta_weight;
        }
        volume_rule.volume_fraction =
            volume_rule.measure / volume_rule.parent_measure;
    }

    svmp::FE::assembly::CutCellAssemblyMetadata metadata;
    metadata.parent_entity = 0;
    metadata.volume_fraction = volume_rule.volume_fraction;
    metadata.side = side;
    metadata.embedded_normal = volume_rule.points.front().normal;
    metadata.cut_topology_id = volume_rule.provenance.cut_topology_id;
    metadata.cut_topology_revision = volume_rule.provenance.cut_topology_revision;

    svmp::FE::assembly::CutIntegrationContext cut_context;
    cut_context.addVolumeRule(std::move(metadata), std::move(volume_rule));
    cut_context.addInterfaceRule(interface_rule);
    return cut_context;
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

TEST(CutCellForms, HighOrderGeneratedCutRulesJITMatchInterpreter)
{
    requireLLVMJITOrSkip();

    auto volume_rule = makeManyPointReferenceTetraCutRule(CutIntegrationSide::Negative);
    markRuleAsHighOrderImplicit(volume_rule,
                                /*marker=*/87,
                                "jit-parity-high-order-volume",
                                /*quadrature_order=*/6);
    const auto volume_constants = makeCutConstants(
        volume_rule, /*side_indicator=*/Real(-1.0), /*stabilization_scale=*/Real(0.25),
        /*quadrature_weight_sensitivity=*/Real(0.0625));
    auto volume_ctx = makeCutAssemblyContext(volume_rule, volume_constants);
    const auto volume_integrand = cutMetadataIntegrand(/*include_quadrature_normal=*/false);

    const Real volume_interp = assembleInterpreterCellTotal(volume_integrand, volume_ctx);
    const Real volume_jit = assembleJITCellTotal(volume_integrand, volume_ctx);
    EXPECT_NEAR(volume_jit, volume_interp, Real(1.0e-13));

    auto interface_rule = makeManyPointReferenceTetraInterfaceRule(/*marker=*/88);
    const auto interface_constants = makeCutConstants(
        interface_rule, /*side_indicator=*/Real(0.0), /*stabilization_scale=*/Real(0.125),
        /*quadrature_weight_sensitivity=*/Real(0.03125));
    auto interface_ctx = makeCutAssemblyContext(interface_rule, interface_constants);
    const auto interface_integrand =
        cutMetadataIntegrand(/*include_quadrature_normal=*/true) +
        FormExpr::facetArea() * Real(0.015625);

    const Real interface_interp =
        assembleInterpreterInterfaceTotal(interface_integrand, interface_ctx, /*marker=*/88);
    const Real interface_jit =
        assembleJITInterfaceTotal(interface_integrand, interface_ctx, /*marker=*/88);
    EXPECT_NEAR(interface_jit, interface_interp, Real(1.0e-13));
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

TEST(CutCellForms, HighOrderCutVolumeManyPointRuleKeepsBasisEvaluation)
{
    constexpr int marker = 83;
    auto rule = makeManyPointReferenceTetraCutRule(CutIntegrationSide::Negative);
    markRuleAsHighOrderImplicit(rule,
                                marker,
                                "many-point-fixed-geometry-volume",
                                /*quadrature_order=*/6);
    const Real expected_measure = rule.measure;

    std::vector<Real> expected(16u, Real(0.0));
    for (const auto& qp : rule.points) {
        const Real xi = qp.point[0];
        const Real eta = qp.point[1];
        const Real zeta = qp.point[2];
        const std::array<Real, 4> phi{{
            Real(1.0) - xi - eta - zeta,
            xi,
            eta,
            zeta}};
        for (std::size_t i = 0u; i < phi.size(); ++i) {
            for (std::size_t j = 0u; j < phi.size(); ++j) {
                expected[i * phi.size() + j] += qp.weight * phi[i] * phi[j];
            }
        }
    }

    svmp::FE::assembly::CutCellAssemblyMetadata metadata;
    metadata.parent_entity = 0;
    metadata.volume_fraction = rule.volume_fraction;
    metadata.side = CutIntegrationSide::Negative;
    metadata.embedded_normal = rule.points.front().normal;
    metadata.cut_topology_id = rule.provenance.cut_topology_id;
    metadata.cut_topology_revision = rule.provenance.cut_topology_revision;

    svmp::FE::assembly::CutIntegrationContext cut_context;
    cut_context.addVolumeRule(std::move(metadata), std::move(rule));

    svmp::FE::forms::test::SingleTetraMeshAccess mesh;
    svmp::FE::spaces::H1Space space(svmp::FE::ElementType::Tetra4, /*order=*/1);
    auto dof_map = createSingleCellDofMap(space.dofs_per_element());
    svmp::FE::assembly::DenseMatrixView mass(dof_map.getNumDofs());
    mass.zero();

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    FormCompiler compiler;
    FormKernel kernel(
        compiler.compileBilinear((u * v).dCutVolume(marker, CutVolumeSide::Negative)));

    svmp::FE::assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    const auto assembled = assembler.assembleCutVolumes(
        mesh, cut_context, marker, CutIntegrationSide::Negative, space, space, kernel,
        &mass, nullptr, /*assemble_matrix=*/true, /*assemble_vector=*/false);

    ASSERT_EQ(assembled.elements_assembled, svmp::FE::GlobalIndex{1});
    ASSERT_EQ(cut_context.volumeRules().front().points.size(), 64u);
    Real assembled_measure = Real(0.0);
    for (svmp::FE::GlobalIndex i = 0; i < dof_map.getNumDofs(); ++i) {
        for (svmp::FE::GlobalIndex j = 0; j < dof_map.getNumDofs(); ++j) {
            const auto idx = static_cast<std::size_t>(i * dof_map.getNumDofs() + j);
            const Real entry = mass.getMatrixEntry(i, j);
            EXPECT_NEAR(entry, expected[idx], Real(1.0e-13));
            assembled_measure += entry;
        }
    }
    EXPECT_NEAR(assembled_measure, expected_measure, Real(1.0e-13));
}

TEST(CutCellForms, HighOrderCutVolumePolynomialMomentsMatchGeneratedRule)
{
    constexpr int marker = 85;
    auto rule = makeManyPointReferenceTetraCutRule(CutIntegrationSide::Negative);
    markRuleAsHighOrderImplicit(rule,
                                marker,
                                "polynomial-moment-fixed-geometry-volume",
                                /*quadrature_order=*/6);

    Real expected_linear = Real(0.0);
    Real expected_quadratic = Real(0.0);
    for (const auto& qp : rule.points) {
        const Real x = qp.point[0];
        const Real y = qp.point[1];
        const Real z = qp.point[2];
        expected_linear += qp.weight * (x + Real(2.0) * y - Real(0.5) * z);
        expected_quadratic += qp.weight * (x * x + y * z + Real(0.25));
    }

    svmp::FE::assembly::CutCellAssemblyMetadata metadata;
    metadata.parent_entity = 0;
    metadata.volume_fraction = rule.volume_fraction;
    metadata.side = CutIntegrationSide::Negative;
    metadata.embedded_normal = rule.points.front().normal;
    metadata.cut_topology_id = rule.provenance.cut_topology_id;
    metadata.cut_topology_revision = rule.provenance.cut_topology_revision;

    svmp::FE::assembly::CutIntegrationContext cut_context;
    cut_context.addVolumeRule(std::move(metadata), std::move(rule));

    svmp::FE::forms::test::SingleTetraMeshAccess mesh;
    svmp::FE::spaces::H1Space space(svmp::FE::ElementType::Tetra4, /*order=*/1);
    auto dof_map = createSingleCellDofMap(space.dofs_per_element());

    svmp::FE::assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto x = FormExpr::coordinate();

    const auto assemble_moment = [&](const FormExpr& coefficient) {
        FormCompiler compiler;
        FormKernel kernel(
            compiler.compileBilinear((coefficient * u * v).dCutVolume(marker, CutVolumeSide::Negative)));
        svmp::FE::assembly::DenseMatrixView mass(dof_map.getNumDofs());
        mass.zero();
        const auto assembled = assembler.assembleCutVolumes(
            mesh, cut_context, marker, CutIntegrationSide::Negative, space, space, kernel,
            &mass, nullptr, /*assemble_matrix=*/true, /*assemble_vector=*/false);
        EXPECT_EQ(assembled.elements_assembled, svmp::FE::GlobalIndex{1});
        Real sum = Real(0.0);
        for (svmp::FE::GlobalIndex i = 0; i < dof_map.getNumDofs(); ++i) {
            for (svmp::FE::GlobalIndex j = 0; j < dof_map.getNumDofs(); ++j) {
                sum += mass.getMatrixEntry(i, j);
            }
        }
        return sum;
    };

    const auto linear =
        x.component(0) + Real(2.0) * x.component(1) - Real(0.5) * x.component(2);
    const auto quadratic =
        x.component(0) * x.component(0) +
        x.component(1) * x.component(2) +
        FormExpr::constant(Real(0.25));

    EXPECT_NEAR(assemble_moment(linear), expected_linear, Real(1.0e-13));
    EXPECT_NEAR(assemble_moment(quadratic), expected_quadratic, Real(1.0e-13));
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

TEST(CutCellForms, LevelSetCutDomainSymbolicTangentAddsInterfaceMeasureTerm)
{
    constexpr int marker = 83;
    constexpr svmp::FE::FieldId phi_field = 17;
    svmp::FE::spaces::H1Space space(svmp::FE::ElementType::Tetra4, /*order=*/2);

    const auto phi = TrialFunction(space, "phi");
    const auto v = TestFunction(space, "v");
    const auto residual = (phi * v).dI(marker);

    FormCompiler compiler;
    auto options = compiler.options();
    options.geometry_sensitivity.mode =
        GeometrySensitivityMode::LevelSetCutDomainUnknowns;
    options.geometry_sensitivity.level_set_cut_domains.push_back(
        LevelSetCutDomainSensitivity{
            .level_set_field = phi_field,
            .interface_marker = marker,
            .side = CutVolumeSide::Negative});
    compiler.setOptions(std::move(options));

    auto ir = compiler.compileResidual(residual);
    SymbolicNonlinearFormKernel kernel(std::move(ir), NonlinearKernelOutput::Both);
    ASSERT_NO_THROW(kernel.resolveInlinableConstitutives());

    int marker_interface_terms = 0;
    bool found_curvature_measure_term = false;
    for (const auto& term : kernel.tangentIR().terms()) {
        if (term.domain != IntegralDomain::InterfaceFace ||
            term.interface_marker != marker) {
            continue;
        }
        ++marker_interface_terms;
        if (term.integrand.node() != nullptr &&
            containsFormExprType(*term.integrand.node(),
                                 FormExprType::Divergence)) {
            found_curvature_measure_term = true;
        }
    }

    EXPECT_GE(marker_interface_terms, 2);
    EXPECT_TRUE(found_curvature_measure_term);
}

TEST(CutCellForms, LevelSetCutDomainSymbolicTangentAddsNormalTerminalVariation)
{
    constexpr int marker = 84;
    constexpr svmp::FE::FieldId phi_field = 17;
    svmp::FE::spaces::H1Space space(svmp::FE::ElementType::Tetra4, /*order=*/2);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto n = FormExpr::normal();
    const auto residual = ((u + n.component(0)) * v).dI(marker);

    FormCompiler compiler;
    auto options = compiler.options();
    options.geometry_sensitivity.mode =
        GeometrySensitivityMode::LevelSetCutDomainUnknowns;
    options.geometry_sensitivity.level_set_cut_domains.push_back(
        LevelSetCutDomainSensitivity{
            .level_set_field = phi_field,
            .interface_marker = marker,
            .side = CutVolumeSide::Negative});
    compiler.setOptions(std::move(options));

    auto ir = compiler.compileResidual(residual);
    SymbolicNonlinearFormKernel kernel(std::move(ir), NonlinearKernelOutput::Both);
    ASSERT_NO_THROW(kernel.resolveInlinableConstitutives());

    bool found_normal_variation_term = false;
    for (const auto& term : kernel.tangentIR().terms()) {
        if (term.domain != IntegralDomain::InterfaceFace ||
            term.interface_marker != marker ||
            term.integrand.node() == nullptr) {
            continue;
        }
        const auto& integrand = *term.integrand.node();
        if (!containsFormExprType(integrand, FormExprType::Divergence) &&
            containsFormExprType(integrand, FormExprType::Gradient) &&
            containsFormExprType(integrand, FormExprType::TrialFunction)) {
            found_normal_variation_term = true;
        }
    }

    EXPECT_TRUE(found_normal_variation_term);
}

TEST(CutCellForms, LevelSetCutDomainShapeTangentUsesConfiguredLevelSetSpace)
{
    constexpr int marker = 187;
    constexpr svmp::FE::FieldId phi_field = 17;
    svmp::FE::spaces::H1Space residual_space(svmp::FE::ElementType::Tetra4,
                                             /*order=*/1);
    svmp::FE::spaces::H1Space level_set_space(svmp::FE::ElementType::Tetra4,
                                              /*order=*/2);

    const auto u = TrialFunction(residual_space, "u");
    const auto v = TestFunction(residual_space, "v");
    const auto residual =
        ((u + FormExpr::constant(Real{0.5})) * v)
            .dCutVolume(marker, CutVolumeSide::Negative);

    FormCompiler compiler;
    auto options = compiler.options();
    options.geometry_sensitivity.mode =
        GeometrySensitivityMode::LevelSetCutDomainUnknowns;
    options.geometry_sensitivity.level_set_cut_domains.push_back(
        LevelSetCutDomainSensitivity{
            .level_set_field = phi_field,
            .interface_marker = marker,
            .side = CutVolumeSide::Negative,
            .level_set_space = spaceSignatureFor(level_set_space)});
    compiler.setOptions(std::move(options));

    auto ir = compiler.compileResidual(residual);
    SymbolicNonlinearFormKernel kernel(std::move(ir), NonlinearKernelOutput::Both);
    ASSERT_NO_THROW(kernel.resolveInlinableConstitutives());

    ASSERT_TRUE(kernel.tangentIR().trialSpace().has_value());
    EXPECT_TRUE(spaceSignaturesEqual(*kernel.tangentIR().trialSpace(),
                                     spaceSignatureFor(level_set_space)));

    const auto field_requirements = kernel.fieldRequirements();
    const auto phi_requirement =
        std::find_if(field_requirements.begin(),
                     field_requirements.end(),
                     [](const svmp::FE::assembly::FieldRequirement& req) {
                         return req.field == phi_field;
                     });
    ASSERT_NE(phi_requirement, field_requirements.end());
    EXPECT_TRUE(svmp::FE::assembly::hasFlag(
        phi_requirement->required,
        svmp::FE::assembly::RequiredData::SolutionGradients));
}

TEST(CutCellForms, LevelSetCutDomainInterfaceNormalMeasureTangentMatchesQuadratureLocalFD)
{
    constexpr int marker = 85;
    constexpr svmp::FE::FieldId phi_field = 17;
    svmp::FE::spaces::H1Space space(svmp::FE::ElementType::Tetra4, /*order=*/2);
    const auto n_dofs = static_cast<svmp::FE::GlobalIndex>(space.dofs_per_element());
    const auto level_set_coefficients =
        quadraticTetraP2LevelSetCoefficients(space);

    const auto phi = TrialFunction(space, "phi");
    const auto v = TestFunction(space, "v");
    const auto n = FormExpr::normal();
    const auto residual =
        ((Real(0.37) * phi +
          Real(1.15) * n.component(0) -
          Real(0.22) * n.component(1) +
          Real(0.09) * n.component(2)) * v).dI(marker);

    FormCompiler compiler;
    auto options = compiler.options();
    options.geometry_sensitivity.mode =
        GeometrySensitivityMode::LevelSetCutDomainUnknowns;
    options.geometry_sensitivity.level_set_cut_domains.push_back(
        LevelSetCutDomainSensitivity{
            .level_set_field = phi_field,
            .interface_marker = marker,
            .side = CutVolumeSide::Negative});
    compiler.setOptions(std::move(options));

    auto matrix_ir = compiler.compileResidual(residual);
    SymbolicNonlinearFormKernel matrix_kernel(
        std::move(matrix_ir), NonlinearKernelOutput::Both);
    ASSERT_NO_THROW(matrix_kernel.resolveInlinableConstitutives());

    const auto field_requirements = matrix_kernel.fieldRequirements();
    const auto phi_requirement =
        std::find_if(field_requirements.begin(),
                     field_requirements.end(),
                     [](const svmp::FE::assembly::FieldRequirement& req) {
                         return req.field == phi_field;
                     });
    ASSERT_NE(phi_requirement, field_requirements.end());
    EXPECT_TRUE(svmp::FE::assembly::hasFlag(
        phi_requirement->required,
        svmp::FE::assembly::RequiredData::SolutionGradients));
    EXPECT_TRUE(svmp::FE::assembly::hasFlag(
        phi_requirement->required,
        svmp::FE::assembly::RequiredData::SolutionHessians));

    FormCompiler vector_compiler;
    auto vector_ir = vector_compiler.compileResidual(residual);
    SymbolicNonlinearFormKernel vector_kernel(
        std::move(vector_ir), NonlinearKernelOutput::VectorOnly);
    ASSERT_NO_THROW(vector_kernel.resolveInlinableConstitutives());

    svmp::FE::forms::test::SingleTetraMeshAccess mesh;
    auto dof_map = createSingleCellDofMap(space.dofs_per_element());
    std::array<svmp::FE::assembly::FieldSolutionAccess, 1> field_access{{
        svmp::FE::assembly::FieldSolutionAccess{
            .field = phi_field,
            .space = &space,
            .dof_map = &dof_map,
            .dof_offset = 0,
            .coefficient_source =
                svmp::FE::assembly::FieldSolutionAccess::CoefficientSource::PrescribedData,
            .prescribed_coefficients = std::span<const Real>(level_set_coefficients),
            .prescribed_revision = 1u,
        }
    }};

    svmp::FE::assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setFieldSolutionAccess(field_access);

    const auto base_context =
        makeQuadraticLevelSetReferenceTetraInterfaceContext(
            marker, space, level_set_coefficients);

    svmp::FE::assembly::DenseMatrixView jacobian(n_dofs);
    svmp::FE::assembly::DenseVectorView residual_vector(n_dofs);
    jacobian.zero();
    residual_vector.zero();

    assembler.setCurrentSolution(level_set_coefficients);
    const auto assembled = assembler.assembleCutInterfaces(
        mesh, base_context, marker, space, space, matrix_kernel,
        &jacobian, &residual_vector,
        /*assemble_matrix=*/true,
        /*assemble_vector=*/true);
    ASSERT_EQ(assembled.interface_faces_assembled, svmp::FE::GlobalIndex{1});

    const Real eps = Real(1.0e-6);
    const Real tolerance = Real(5.0e-5);
    for (svmp::FE::GlobalIndex j = 0; j < n_dofs; ++j) {
        auto solution_plus = level_set_coefficients;
        auto solution_minus = level_set_coefficients;
        solution_plus[static_cast<std::size_t>(j)] += eps;
        solution_minus[static_cast<std::size_t>(j)] -= eps;

        const auto plus_context =
            makeQuadraticLevelSetReferenceTetraInterfaceContext(
                marker, space, level_set_coefficients,
                static_cast<int>(j), eps,
                /*move_points_with_level_set=*/true);
        const auto minus_context =
            makeQuadraticLevelSetReferenceTetraInterfaceContext(
                marker, space, level_set_coefficients,
                static_cast<int>(j), -eps,
                /*move_points_with_level_set=*/true);

        svmp::FE::assembly::DenseVectorView residual_plus(n_dofs);
        svmp::FE::assembly::DenseVectorView residual_minus(n_dofs);
        residual_plus.zero();
        residual_minus.zero();

        assembler.setCurrentSolution(solution_plus);
        (void)assembler.assembleCutInterfaces(
            mesh, plus_context, marker, space, space, vector_kernel,
            nullptr, &residual_plus,
            /*assemble_matrix=*/false,
            /*assemble_vector=*/true);
        assembler.setCurrentSolution(solution_minus);
        (void)assembler.assembleCutInterfaces(
            mesh, minus_context, marker, space, space, vector_kernel,
            nullptr, &residual_minus,
            /*assemble_matrix=*/false,
            /*assemble_vector=*/true);

        for (svmp::FE::GlobalIndex i = 0; i < n_dofs; ++i) {
            const Real finite_difference =
                (residual_plus.getVectorEntry(i) -
                 residual_minus.getVectorEntry(i)) /
                (Real(2.0) * eps);
            EXPECT_NEAR(jacobian.getMatrixEntry(i, j),
                        finite_difference,
                        tolerance)
                << "row=" << i << " col=" << j;
        }
    }
}

TEST(CutCellForms, LevelSetCutDomainCutVolumeShapeTangentMatchesQuadratureLocalFD)
{
    constexpr int marker = 86;
    constexpr svmp::FE::FieldId phi_field = 17;
    svmp::FE::spaces::H1Space space(svmp::FE::ElementType::Tetra4, /*order=*/2);
    const auto n_dofs = static_cast<svmp::FE::GlobalIndex>(space.dofs_per_element());
    const auto level_set_coefficients =
        quadraticTetraP2LevelSetCoefficients(space);

    const auto phi = TrialFunction(space, "phi");
    const auto v = TestFunction(space, "v");
    const auto x = FormExpr::coordinate();
    const auto residual =
        ((Real(0.43) * phi +
          Real(0.21) * x.component(0) -
          Real(0.14) * x.component(1) +
          Real(0.08) * x.component(2) +
          FormExpr::constant(Real(0.62))) * v)
             .dCutVolume(marker, CutVolumeSide::Negative);

    FormCompiler compiler;
    auto options = compiler.options();
    options.geometry_sensitivity.mode =
        GeometrySensitivityMode::LevelSetCutDomainUnknowns;
    options.geometry_sensitivity.level_set_cut_domains.push_back(
        LevelSetCutDomainSensitivity{
            .level_set_field = phi_field,
            .interface_marker = marker,
            .side = CutVolumeSide::Negative});
    compiler.setOptions(std::move(options));

    auto matrix_ir = compiler.compileResidual(residual);
    SymbolicNonlinearFormKernel matrix_kernel(
        std::move(matrix_ir), NonlinearKernelOutput::Both);
    ASSERT_NO_THROW(matrix_kernel.resolveInlinableConstitutives());

    FormCompiler vector_compiler;
    auto vector_ir = vector_compiler.compileResidual(residual);
    SymbolicNonlinearFormKernel vector_kernel(
        std::move(vector_ir), NonlinearKernelOutput::VectorOnly);
    ASSERT_NO_THROW(vector_kernel.resolveInlinableConstitutives());

    svmp::FE::forms::test::SingleTetraMeshAccess mesh;
    auto dof_map = createSingleCellDofMap(space.dofs_per_element());
    std::array<svmp::FE::assembly::FieldSolutionAccess, 1> field_access{{
        svmp::FE::assembly::FieldSolutionAccess{
            .field = phi_field,
            .space = &space,
            .dof_map = &dof_map,
            .dof_offset = 0,
            .coefficient_source =
                svmp::FE::assembly::FieldSolutionAccess::CoefficientSource::PrescribedData,
            .prescribed_coefficients = std::span<const Real>(level_set_coefficients),
            .prescribed_revision = 1u,
        }
    }};

    svmp::FE::assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setFieldSolutionAccess(field_access);

    const auto base_context =
        makeQuadraticLevelSetReferenceTetraVolumeShapeFDContext(
            marker,
            CutIntegrationSide::Negative,
            space,
            level_set_coefficients);

    svmp::FE::assembly::DenseMatrixView jacobian(n_dofs);
    svmp::FE::assembly::DenseVectorView residual_vector(n_dofs);
    jacobian.zero();
    residual_vector.zero();

    assembler.setCurrentSolution(level_set_coefficients);
    const auto volume_assembled = assembler.assembleCutVolumes(
        mesh, base_context, marker, CutIntegrationSide::Negative, space, space,
        matrix_kernel, &jacobian, &residual_vector,
        /*assemble_matrix=*/true,
        /*assemble_vector=*/true);
    ASSERT_EQ(volume_assembled.elements_assembled, svmp::FE::GlobalIndex{1});
    const auto interface_assembled = assembler.assembleCutInterfaces(
        mesh, base_context, marker, space, space, matrix_kernel,
        &jacobian, nullptr,
        /*assemble_matrix=*/true,
        /*assemble_vector=*/false);
    ASSERT_EQ(interface_assembled.interface_faces_assembled,
              svmp::FE::GlobalIndex{1});

    const Real eps = Real(1.0e-6);
    const Real tolerance = Real(7.5e-6);
    for (svmp::FE::GlobalIndex j = 0; j < n_dofs; ++j) {
        auto solution_plus = level_set_coefficients;
        auto solution_minus = level_set_coefficients;
        solution_plus[static_cast<std::size_t>(j)] += eps;
        solution_minus[static_cast<std::size_t>(j)] -= eps;

        const auto plus_context =
            makeQuadraticLevelSetReferenceTetraVolumeShapeFDContext(
                marker,
                CutIntegrationSide::Negative,
                space,
                level_set_coefficients,
                static_cast<int>(j),
                eps);
        const auto minus_context =
            makeQuadraticLevelSetReferenceTetraVolumeShapeFDContext(
                marker,
                CutIntegrationSide::Negative,
                space,
                level_set_coefficients,
                static_cast<int>(j),
                -eps);

        svmp::FE::assembly::DenseVectorView residual_plus(n_dofs);
        svmp::FE::assembly::DenseVectorView residual_minus(n_dofs);
        residual_plus.zero();
        residual_minus.zero();

        assembler.setCurrentSolution(solution_plus);
        (void)assembler.assembleCutVolumes(
            mesh, plus_context, marker, CutIntegrationSide::Negative, space,
            space, vector_kernel, nullptr, &residual_plus,
            /*assemble_matrix=*/false,
            /*assemble_vector=*/true);
        assembler.setCurrentSolution(solution_minus);
        (void)assembler.assembleCutVolumes(
            mesh, minus_context, marker, CutIntegrationSide::Negative, space,
            space, vector_kernel, nullptr, &residual_minus,
            /*assemble_matrix=*/false,
            /*assemble_vector=*/true);

        for (svmp::FE::GlobalIndex i = 0; i < n_dofs; ++i) {
            const Real finite_difference =
                (residual_plus.getVectorEntry(i) -
                 residual_minus.getVectorEntry(i)) /
                (Real(2.0) * eps);
            EXPECT_NEAR(jacobian.getMatrixEntry(i, j),
                        finite_difference,
                        tolerance)
                << "row=" << i << " col=" << j;
        }
    }
}

TEST(CutCellForms, LevelSetCutDomainInterfaceShapeTangentMatchesQuadratureLocalFD)
{
    constexpr int marker = 87;
    constexpr svmp::FE::FieldId phi_field = 17;
    svmp::FE::spaces::H1Space space(svmp::FE::ElementType::Tetra4, /*order=*/2);
    const auto n_dofs = static_cast<svmp::FE::GlobalIndex>(space.dofs_per_element());
    const auto level_set_coefficients =
        quadraticTetraP2LevelSetCoefficients(space);

    const auto phi = TrialFunction(space, "phi");
    const auto v = TestFunction(space, "v");
    const auto x = FormExpr::coordinate();
    const auto n = FormExpr::normal();
    const auto residual =
        ((Real(0.43) * phi +
          Real(0.21) * x.component(0) -
          Real(0.14) * x.component(1) +
          Real(0.08) * x.component(2) +
          Real(0.11) * n.component(0) -
          Real(0.07) * n.component(1) +
          Real(0.06) * inner(grad(phi), n) +
          FormExpr::constant(Real(0.62))) * v)
             .dI(marker);

    FormCompiler compiler;
    auto options = compiler.options();
    options.geometry_sensitivity.mode =
        GeometrySensitivityMode::LevelSetCutDomainUnknowns;
    options.geometry_sensitivity.level_set_cut_domains.push_back(
        LevelSetCutDomainSensitivity{
            .level_set_field = phi_field,
            .interface_marker = marker,
            .side = CutVolumeSide::Negative});
    compiler.setOptions(std::move(options));

    auto matrix_ir = compiler.compileResidual(residual);
    SymbolicNonlinearFormKernel matrix_kernel(
        std::move(matrix_ir), NonlinearKernelOutput::Both);
    ASSERT_NO_THROW(matrix_kernel.resolveInlinableConstitutives());

    FormCompiler vector_compiler;
    auto vector_ir = vector_compiler.compileResidual(residual);
    SymbolicNonlinearFormKernel vector_kernel(
        std::move(vector_ir), NonlinearKernelOutput::VectorOnly);
    ASSERT_NO_THROW(vector_kernel.resolveInlinableConstitutives());

    svmp::FE::forms::test::SingleTetraMeshAccess mesh;
    auto dof_map = createSingleCellDofMap(space.dofs_per_element());
    std::array<svmp::FE::assembly::FieldSolutionAccess, 1> field_access{{
        svmp::FE::assembly::FieldSolutionAccess{
            .field = phi_field,
            .space = &space,
            .dof_map = &dof_map,
            .dof_offset = 0,
            .coefficient_source =
                svmp::FE::assembly::FieldSolutionAccess::CoefficientSource::PrescribedData,
            .prescribed_coefficients = std::span<const Real>(level_set_coefficients),
            .prescribed_revision = 1u,
        }
    }};

    svmp::FE::assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setFieldSolutionAccess(field_access);

    const auto base_context =
        makeQuadraticLevelSetReferenceTetraInterfaceContext(
            marker,
            space,
            level_set_coefficients);

    svmp::FE::assembly::DenseMatrixView jacobian(n_dofs);
    svmp::FE::assembly::DenseVectorView residual_vector(n_dofs);
    jacobian.zero();
    residual_vector.zero();

    assembler.setCurrentSolution(level_set_coefficients);
    const auto interface_assembled = assembler.assembleCutInterfaces(
        mesh, base_context, marker, space, space, matrix_kernel,
        &jacobian, &residual_vector,
        /*assemble_matrix=*/true,
        /*assemble_vector=*/true);
    ASSERT_EQ(interface_assembled.interface_faces_assembled,
              svmp::FE::GlobalIndex{1});

    const Real eps = Real(1.0e-6);
    const Real tolerance = Real(1.0e-5);
    for (svmp::FE::GlobalIndex j = 0; j < n_dofs; ++j) {
        auto solution_plus = level_set_coefficients;
        auto solution_minus = level_set_coefficients;
        solution_plus[static_cast<std::size_t>(j)] += eps;
        solution_minus[static_cast<std::size_t>(j)] -= eps;

        const auto plus_context =
            makeQuadraticLevelSetReferenceTetraInterfaceContext(
                marker,
                space,
                level_set_coefficients,
                static_cast<int>(j),
                eps,
                /*move_points_with_level_set=*/true);
        const auto minus_context =
            makeQuadraticLevelSetReferenceTetraInterfaceContext(
                marker,
                space,
                level_set_coefficients,
                static_cast<int>(j),
                -eps,
                /*move_points_with_level_set=*/true);

        svmp::FE::assembly::DenseVectorView residual_plus(n_dofs);
        svmp::FE::assembly::DenseVectorView residual_minus(n_dofs);
        residual_plus.zero();
        residual_minus.zero();

        assembler.setCurrentSolution(solution_plus);
        (void)assembler.assembleCutInterfaces(
            mesh, plus_context, marker, space, space, vector_kernel,
            nullptr, &residual_plus,
            /*assemble_matrix=*/false,
            /*assemble_vector=*/true);
        assembler.setCurrentSolution(solution_minus);
        (void)assembler.assembleCutInterfaces(
            mesh, minus_context, marker, space, space, vector_kernel,
            nullptr, &residual_minus,
            /*assemble_matrix=*/false,
            /*assemble_vector=*/true);

        for (svmp::FE::GlobalIndex i = 0; i < n_dofs; ++i) {
            const Real finite_difference =
                (residual_plus.getVectorEntry(i) -
                 residual_minus.getVectorEntry(i)) /
                (Real(2.0) * eps);
            EXPECT_NEAR(jacobian.getMatrixEntry(i, j),
                        finite_difference,
                        tolerance)
                << "row=" << i << " col=" << j;
        }
    }
}

TEST(CutCellForms, LevelSetCutDomainInterfaceMeasureTangentExpandsAllMarkers)
{
    constexpr int marker_a = 183;
    constexpr int marker_b = 184;
    constexpr svmp::FE::FieldId phi_field = 17;
    svmp::FE::spaces::H1Space space(svmp::FE::ElementType::Tetra4, /*order=*/2);

    const auto phi = TrialFunction(space, "phi");
    const auto v = TestFunction(space, "v");
    const auto residual = (phi * v).dI();

    FormCompiler compiler;
    auto options = compiler.options();
    options.geometry_sensitivity.mode =
        GeometrySensitivityMode::LevelSetCutDomainUnknowns;
    options.geometry_sensitivity.level_set_cut_domains.push_back(
        LevelSetCutDomainSensitivity{
            .level_set_field = phi_field,
            .interface_marker = marker_a,
            .side = CutVolumeSide::Negative});
    options.geometry_sensitivity.level_set_cut_domains.push_back(
        LevelSetCutDomainSensitivity{
            .level_set_field = phi_field,
            .interface_marker = marker_b,
            .side = CutVolumeSide::Positive});
    compiler.setOptions(std::move(options));

    auto ir = compiler.compileResidual(residual);
    SymbolicNonlinearFormKernel kernel(std::move(ir), NonlinearKernelOutput::Both);
    ASSERT_NO_THROW(kernel.resolveInlinableConstitutives());

    std::vector<int> curvature_term_markers;
    for (const auto& term : kernel.tangentIR().terms()) {
        if (term.domain != IntegralDomain::InterfaceFace ||
            term.integrand.node() == nullptr ||
            !containsFormExprType(*term.integrand.node(),
                                  FormExprType::Divergence)) {
            continue;
        }
        curvature_term_markers.push_back(term.interface_marker);
    }
    std::sort(curvature_term_markers.begin(), curvature_term_markers.end());

    const std::vector<int> expected_markers = {marker_a, marker_b};
    EXPECT_EQ(curvature_term_markers, expected_markers);
}

TEST(CutCellForms, LevelSetCutDomainInterfaceTangentKeepsDistinctFieldsOnSameMarker)
{
    constexpr int marker = 185;
    constexpr svmp::FE::FieldId phi_a = 17;
    constexpr svmp::FE::FieldId phi_b = 18;
    svmp::FE::spaces::H1Space space(svmp::FE::ElementType::Tetra4, /*order=*/2);

    const auto phi = TrialFunction(space, "phi");
    const auto v = TestFunction(space, "v");
    const auto residual = (phi * v).dI(marker);

    FormCompiler compiler;
    auto options = compiler.options();
    options.geometry_sensitivity.mode =
        GeometrySensitivityMode::LevelSetCutDomainUnknowns;
    options.geometry_sensitivity.level_set_cut_domains.push_back(
        LevelSetCutDomainSensitivity{
            .level_set_field = phi_a,
            .interface_marker = marker,
            .side = CutVolumeSide::Negative});
    options.geometry_sensitivity.level_set_cut_domains.push_back(
        LevelSetCutDomainSensitivity{
            .level_set_field = phi_a,
            .interface_marker = marker,
            .side = CutVolumeSide::Positive});
    options.geometry_sensitivity.level_set_cut_domains.push_back(
        LevelSetCutDomainSensitivity{
            .level_set_field = phi_b,
            .interface_marker = marker,
            .side = CutVolumeSide::Negative});
    compiler.setOptions(std::move(options));

    auto ir = compiler.compileResidual(residual);
    SymbolicNonlinearFormKernel kernel(std::move(ir), NonlinearKernelOutput::Both);
    ASSERT_NO_THROW(kernel.resolveInlinableConstitutives());

    int curvature_measure_terms = 0;
    for (const auto& term : kernel.tangentIR().terms()) {
        if (term.domain != IntegralDomain::InterfaceFace ||
            term.interface_marker != marker ||
            term.integrand.node() == nullptr ||
            !containsFormExprType(*term.integrand.node(),
                                  FormExprType::Divergence)) {
            continue;
        }
        ++curvature_measure_terms;
    }

    EXPECT_EQ(curvature_measure_terms, 2);
}

TEST(CutCellForms,
     LevelSetCutDomainTangentDifferentiatesEachConfiguredLevelSetField)
{
    constexpr int marker = 186;
    constexpr svmp::FE::FieldId phi_a = 17;
    constexpr svmp::FE::FieldId phi_b = 18;
    svmp::FE::spaces::H1Space space(svmp::FE::ElementType::Tetra4, /*order=*/2);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto phi_a_state = FormExpr::stateField(phi_a, space, "phi_a");
    const auto phi_b_state = FormExpr::stateField(phi_b, space, "phi_b");
    const auto integrand =
        (phi_a_state + Real(2.0) * phi_b_state + Real(0.0) * u) * v;
    const auto residual =
        integrand.dI(marker) +
        integrand.dCutVolume(marker, CutVolumeSide::Negative);

    FormCompiler compiler;
    auto options = compiler.options();
    options.geometry_sensitivity.mode =
        GeometrySensitivityMode::LevelSetCutDomainUnknowns;
    options.geometry_sensitivity.level_set_cut_domains.push_back(
        LevelSetCutDomainSensitivity{
            .level_set_field = phi_a,
            .interface_marker = marker,
            .side = CutVolumeSide::Negative});
    options.geometry_sensitivity.level_set_cut_domains.push_back(
        LevelSetCutDomainSensitivity{
            .level_set_field = phi_b,
            .interface_marker = marker,
            .side = CutVolumeSide::Negative});
    compiler.setOptions(std::move(options));

    auto ir = compiler.compileResidual(residual);
    SymbolicNonlinearFormKernel kernel(std::move(ir), NonlinearKernelOutput::Both);
    ASSERT_NO_THROW(kernel.resolveInlinableConstitutives());

    int interface_integrand_derivative_terms = 0;
    int cut_volume_integrand_derivative_terms = 0;
    for (const auto& term : kernel.tangentIR().terms()) {
        if (term.integrand.node() == nullptr) {
            continue;
        }
        const auto text = term.integrand.node()->toString();
        const bool has_trial_variation =
            text.find("du") != std::string::npos;
        const bool has_curvature_measure =
            containsFormExprType(*term.integrand.node(),
                                 FormExprType::Divergence);
        if (term.domain == IntegralDomain::InterfaceFace &&
            term.interface_marker == marker && !has_curvature_measure &&
            has_trial_variation) {
            ++interface_integrand_derivative_terms;
        }
        if (term.domain == IntegralDomain::CutVolume &&
            term.interface_marker == marker &&
            term.cut_volume_side == CutVolumeSide::Negative &&
            has_trial_variation) {
            ++cut_volume_integrand_derivative_terms;
        }
    }

    EXPECT_GE(interface_integrand_derivative_terms, 2);
    EXPECT_GE(cut_volume_integrand_derivative_terms, 2);
}

TEST(CutCellForms, HighOrderCutInterfaceManyPointRuleRemapsBasisEvaluation)
{
    constexpr int marker = 84;
    auto rule = makeManyPointReferenceTetraInterfaceRule(marker);

    std::vector<Real> expected(16u, Real(0.0));
    for (const auto& qp : rule.points) {
        const Real xi = qp.point[0];
        const Real eta = qp.point[1];
        const Real zeta = qp.point[2];
        const std::array<Real, 4> phi{{
            Real(1.0) - xi - eta - zeta,
            xi,
            eta,
            zeta}};
        for (std::size_t i = 0u; i < phi.size(); ++i) {
            for (std::size_t j = 0u; j < phi.size(); ++j) {
                expected[i * phi.size() + j] += qp.weight * phi[i] * phi[j];
            }
        }
    }

    svmp::FE::assembly::CutIntegrationContext cut_context;
    cut_context.addInterfaceRule(std::move(rule));

    svmp::FE::forms::test::SingleTetraMeshAccess mesh;
    svmp::FE::spaces::H1Space space(svmp::FE::ElementType::Tetra4, /*order=*/1);
    auto dof_map = createSingleCellDofMap(space.dofs_per_element());
    svmp::FE::assembly::DenseMatrixView mass(dof_map.getNumDofs());
    mass.zero();

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    FormCompiler compiler;
    FormKernel kernel(compiler.compileBilinear((u * v).dI(marker)));

    svmp::FE::assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    const auto assembled = assembler.assembleCutInterfaces(
        mesh, cut_context, marker, space, space, kernel,
        &mass, nullptr, /*assemble_matrix=*/true, /*assemble_vector=*/false);

    ASSERT_EQ(assembled.interface_faces_assembled, svmp::FE::GlobalIndex{1});
    ASSERT_EQ(cut_context.interfaceRules().front().points.size(), 64u);
    for (svmp::FE::GlobalIndex i = 0; i < dof_map.getNumDofs(); ++i) {
        for (svmp::FE::GlobalIndex j = 0; j < dof_map.getNumDofs(); ++j) {
            const auto idx = static_cast<std::size_t>(i * dof_map.getNumDofs() + j);
            EXPECT_NEAR(mass.getMatrixEntry(i, j), expected[idx], Real(1.0e-13));
        }
    }
}

TEST(CutCellForms, HighOrderCutInterfaceSurfaceTractionMatchesGeneratedRule)
{
    constexpr int marker = 86;
    auto rule = makeManyPointReferenceTetraInterfaceRule(marker);
    Real expected_traction = Real(0.0);
    for (const auto& qp : rule.points) {
        expected_traction += qp.weight * qp.normal[0];
    }

    svmp::FE::assembly::CutIntegrationContext cut_context;
    cut_context.addInterfaceRule(std::move(rule));

    svmp::FE::forms::test::SingleTetraMeshAccess mesh;
    svmp::FE::spaces::H1Space space(svmp::FE::ElementType::Tetra4, /*order=*/1);
    auto dof_map = createSingleCellDofMap(space.dofs_per_element());
    svmp::FE::assembly::DenseVectorView rhs(dof_map.getNumDofs());
    rhs.zero();

    const auto v = TestFunction(space, "v");
    const auto n = FormExpr::normal();
    FormCompiler compiler;
    FormKernel kernel(compiler.compileLinear((n.component(0) * v).dI(marker)));

    svmp::FE::assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    const auto assembled = assembler.assembleCutInterfaces(
        mesh, cut_context, marker, space, space, kernel,
        nullptr, &rhs, /*assemble_matrix=*/false, /*assemble_vector=*/true);

    ASSERT_EQ(assembled.interface_faces_assembled, svmp::FE::GlobalIndex{1});
    Real assembled_traction = Real(0.0);
    for (svmp::FE::GlobalIndex i = 0; i < dof_map.getNumDofs(); ++i) {
        assembled_traction += rhs.getVectorEntry(i);
    }
    EXPECT_NEAR(assembled_traction, expected_traction, Real(1.0e-13));
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

TEST(CutCellForms, CutInterfaceSurfaceTractionSignFollowsQuadratureNormal)
{
    auto rule = svmp::FE::geometry::makeAxisAlignedBoxCutInterfaceQuadrature(
        {{0.0, 0.0, 0.0}},
        {{1.0, 1.0, 1.0}},
        /*axis=*/0,
        /*cut_coordinate=*/Real(0.25),
        "traction-plane");
    ASSERT_EQ(rule.kind, CutQuadratureKind::Interface);
    ASSERT_FALSE(rule.points.empty());

    for (auto& point : rule.points) {
        point.normal = {{1.0, 0.0, 0.0}};
    }
    const auto traction = FormExpr::asVector({
        FormExpr::constant(2.0),
        FormExpr::constant(-3.0),
        FormExpr::constant(0.5),
    });
    const auto normal_traction = inner(traction, FormExpr::normal());
    std::vector<Real> constants;

    const auto positive_ctx = makeCutAssemblyContext(rule, constants);
    const Real positive =
        assembleInterpreterInterfaceTotal(normal_traction, positive_ctx, /*marker=*/19);

    for (auto& point : rule.points) {
        point.normal = {{-1.0, 0.0, 0.0}};
    }
    const auto negative_ctx = makeCutAssemblyContext(rule, constants);
    const Real negative =
        assembleInterpreterInterfaceTotal(normal_traction, negative_ctx, /*marker=*/19);

    EXPECT_NEAR(positive, 2.0 * rule.measure, 1.0e-13);
    EXPECT_NEAR(negative, -2.0 * rule.measure, 1.0e-13);
}
