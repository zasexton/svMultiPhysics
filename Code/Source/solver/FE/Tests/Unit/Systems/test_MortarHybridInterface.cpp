/**
 * @file test_MortarHybridInterface.cpp
 * @brief Mortar and hybridized interface assembly regressions
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"
#include "Systems/TransientSystem.h"

#include "Assembly/AssemblyKernel.h"
#include "Assembly/GlobalSystemView.h"

#include "Dofs/EntityDofMap.h"

#include "Forms/Vocabulary.h"

#include "Spaces/H1Space.h"
#include "Spaces/HDivSpace.h"
#include "Spaces/L2Space.h"
#include "Spaces/MortarSpace.h"
#include "Spaces/ProductSpace.h"

#include "Systems/FormsInstallerDetail.h"
#include "Mesh/Mesh.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Core/InterfaceMesh.h"
#include "Mesh/Topology/CellShape.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

using svmp::CellFamily;
using svmp::CellShape;
using svmp::EntityKind;
using svmp::Mesh;
using svmp::MeshBase;

using svmp::FE::ElementType;
using svmp::FE::GlobalIndex;
using svmp::FE::LocalIndex;
using svmp::FE::Real;

using svmp::FE::assembly::DenseMatrixView;
using svmp::FE::assembly::DenseVectorView;
using svmp::FE::assembly::FacetMassKernel;
using svmp::FE::assembly::FacetSourceKernel;
using svmp::FE::assembly::InterfaceEvaluationSide;
using svmp::FE::assembly::MassKernel;
using svmp::FE::assembly::NormalTraceCouplingKernel;

using svmp::FE::spaces::H1Space;
using svmp::FE::spaces::HDivSpace;
using svmp::FE::spaces::L2Space;
using svmp::FE::spaces::MortarSpace;
using svmp::FE::spaces::ProductSpace;

using svmp::FE::systems::AssemblyRequest;
using svmp::FE::systems::BackwardDifferenceIntegrator;
using svmp::FE::systems::FESystem;
using svmp::FE::systems::FieldSpec;
using svmp::FE::systems::SystemStateView;
using svmp::FE::systems::TransientSystem;

namespace {

std::shared_ptr<Mesh> build_two_quad_mesh_with_interface_set(const std::string& set_name)
{
    auto base = std::make_shared<MeshBase>();

    const std::vector<svmp::real_t> X_ref = {
        0.0, 0.0,
        1.0, 0.0,
        2.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        2.0, 1.0
    };
    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4, 8};
    const std::vector<svmp::index_t> cell2vertex = {0, 1, 4, 3, 1, 2, 5, 4};

    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(/*spatial_dim=*/2, X_ref, cell2vertex_offsets, cell2vertex,
                            {shape, shape});
    base->finalize();

    auto mesh = svmp::create_mesh(std::move(base));
    auto& mbase = mesh->local_mesh();
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(mbase.n_faces()); ++f) {
        const auto verts = mbase.face_vertices(f);
        if (verts.size() != 2u) {
            continue;
        }
        const bool has1 = (verts[0] == 1 || verts[1] == 1);
        const bool has4 = (verts[0] == 4 || verts[1] == 4);
        if (has1 && has4) {
            mbase.add_to_set(EntityKind::Face, set_name, f);
            break;
        }
    }

    return mesh;
}

std::shared_ptr<Mesh> build_four_quad_mesh_with_interface_set(const std::string& set_name)
{
    auto base = std::make_shared<MeshBase>();

    const std::vector<svmp::real_t> X_ref = {
        0.0, 0.0,
        1.0, 0.0,
        2.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        2.0, 1.0,
        0.0, 2.0,
        1.0, 2.0,
        2.0, 2.0
    };
    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4, 8, 12, 16};
    const std::vector<svmp::index_t> cell2vertex = {
        0, 1, 4, 3,
        1, 2, 5, 4,
        3, 4, 7, 6,
        4, 5, 8, 7
    };

    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(/*spatial_dim=*/2, X_ref, cell2vertex_offsets, cell2vertex,
                            {shape, shape, shape, shape});
    base->finalize();

    auto mesh = svmp::create_mesh(std::move(base));
    auto& mbase = mesh->local_mesh();
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(mbase.n_faces()); ++f) {
        const auto verts = mbase.face_vertices(f);
        if (verts.size() != 2u) {
            continue;
        }
        const bool lower_segment =
            (verts[0] == 1 || verts[1] == 1) && (verts[0] == 4 || verts[1] == 4);
        const bool upper_segment =
            (verts[0] == 4 || verts[1] == 4) && (verts[0] == 7 || verts[1] == 7);
        if (lower_segment || upper_segment) {
            mbase.add_to_set(EntityKind::Face, set_name, f);
        }
    }

    return mesh;
}

GlobalIndex find_interface_face_id(const Mesh& mesh)
{
    const auto& mbase = mesh.local_mesh();
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(mbase.n_faces()); ++f) {
        const auto verts = mbase.face_vertices(f);
        if (verts.size() != 2u) {
            continue;
        }
        const bool has1 = (verts[0] == 1 || verts[1] == 1);
        const bool has4 = (verts[0] == 4 || verts[1] == 4);
        if (has1 && has4) {
            return static_cast<GlobalIndex>(f);
        }
    }
    return -1;
}

double block_abs_sum(const DenseMatrixView& A,
                     GlobalIndex r0,
                     GlobalIndex r1,
                     GlobalIndex c0,
                     GlobalIndex c1)
{
    double sum = 0.0;
    for (GlobalIndex i = r0; i < r1; ++i) {
        for (GlobalIndex j = c0; j < c1; ++j) {
            sum += std::abs(static_cast<double>(A.getMatrixEntry(i, j)));
        }
    }
    return sum;
}

std::vector<double> dense_solve(std::vector<double> A, std::vector<double> b, int n)
{
    const double eps = 1e-14;
    for (int k = 0; k < n; ++k) {
        int pivot = k;
        double pivot_abs = std::abs(A[static_cast<std::size_t>(k) * n + k]);
        for (int r = k + 1; r < n; ++r) {
            const double cand = std::abs(A[static_cast<std::size_t>(r) * n + k]);
            if (cand > pivot_abs) {
                pivot = r;
                pivot_abs = cand;
            }
        }
        EXPECT_GT(pivot_abs, eps);
        if (pivot != k) {
            for (int c = 0; c < n; ++c) {
                std::swap(A[static_cast<std::size_t>(pivot) * n + c],
                          A[static_cast<std::size_t>(k) * n + c]);
            }
            std::swap(b[static_cast<std::size_t>(pivot)], b[static_cast<std::size_t>(k)]);
        }

        const double diag = A[static_cast<std::size_t>(k) * n + k];
        for (int c = k; c < n; ++c) {
            A[static_cast<std::size_t>(k) * n + c] /= diag;
        }
        b[static_cast<std::size_t>(k)] /= diag;

        for (int r = 0; r < n; ++r) {
            if (r == k) {
                continue;
            }
            const double factor = A[static_cast<std::size_t>(r) * n + k];
            if (factor == 0.0) {
                continue;
            }
            for (int c = k; c < n; ++c) {
                A[static_cast<std::size_t>(r) * n + c] -=
                    factor * A[static_cast<std::size_t>(k) * n + c];
            }
            b[static_cast<std::size_t>(r)] -= factor * b[static_cast<std::size_t>(k)];
        }
    }
    return b;
}

std::vector<double> mat_vec(const std::vector<double>& A,
                            const std::vector<double>& x,
                            int rows,
                            int cols)
{
    std::vector<double> out(static_cast<std::size_t>(rows), 0.0);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            out[static_cast<std::size_t>(i)] +=
                A[static_cast<std::size_t>(i) * cols + j] * x[static_cast<std::size_t>(j)];
        }
    }
    return out;
}

std::vector<double> submatrix(const DenseMatrixView& A,
                              GlobalIndex r0,
                              GlobalIndex r1,
                              GlobalIndex c0,
                              GlobalIndex c1)
{
    const int rows = static_cast<int>(r1 - r0);
    const int cols = static_cast<int>(c1 - c0);
    std::vector<double> out(static_cast<std::size_t>(rows * cols), 0.0);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            out[static_cast<std::size_t>(i) * cols + j] =
                A.getMatrixEntry(r0 + i, c0 + j);
        }
    }
    return out;
}

std::vector<double> subvector(const DenseVectorView& b, GlobalIndex i0, GlobalIndex i1)
{
    std::vector<double> out(static_cast<std::size_t>(i1 - i0), 0.0);
    for (GlobalIndex i = i0; i < i1; ++i) {
        out[static_cast<std::size_t>(i - i0)] = b.getVectorEntry(i);
    }
    return out;
}

class InterfaceDtResidualKernel final : public svmp::FE::assembly::LinearFormKernel {
public:
    explicit InterfaceDtResidualKernel(Real coefficient = 1.0,
                                       InterfaceEvaluationSide side = InterfaceEvaluationSide::Minus)
        : coefficient_(coefficient)
        , side_(side)
    {
    }

    [[nodiscard]] svmp::FE::assembly::RequiredData getRequiredData() const override
    {
        using svmp::FE::assembly::RequiredData;
        return RequiredData::BasisValues | RequiredData::IntegrationWeights |
               RequiredData::SolutionValues;
    }

    void computeCell(const svmp::FE::assembly::AssemblyContext&, svmp::FE::assembly::KernelOutput&) override
    {
        FE_THROW(svmp::FE::FEException,
                 "InterfaceDtResidualKernel::computeCell is not implemented (face-only kernel)");
    }

    void computeBoundaryFace(const svmp::FE::assembly::AssemblyContext& ctx,
                             int,
                             svmp::FE::assembly::KernelOutput& output) override
    {
        output.reserve(ctx.numTestDofs(), ctx.numTrialDofs(),
                       /*need_matrix=*/false, /*need_vector=*/true);

        for (LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
            const Real du = ctx.solutionValue(q) - ctx.previousSolutionValue(q);
            const Real w = coefficient_ * ctx.integrationWeight(q);
            for (LocalIndex i = 0; i < ctx.numTestDofs(); ++i) {
                output.vectorEntry(i) += w * ctx.basisValue(i, q) * du;
            }
        }
    }

    [[nodiscard]] bool hasCell() const noexcept override { return false; }
    [[nodiscard]] bool hasBoundaryFace() const noexcept override { return true; }
    [[nodiscard]] bool hasSingleSidedInterfaceFace() const noexcept override { return true; }
    [[nodiscard]] InterfaceEvaluationSide interfaceEvaluationSide() const noexcept override
    {
        return side_;
    }
    [[nodiscard]] int maxTemporalDerivativeOrder() const noexcept override { return 1; }
    [[nodiscard]] std::string name() const override { return "InterfaceDtResidualKernel"; }

private:
    Real coefficient_{1.0};
    InterfaceEvaluationSide side_{InterfaceEvaluationSide::Minus};
};

struct InterfaceMovingMeshSnapshot {
    std::vector<Real> matrix;
    std::vector<Real> vector;
    GlobalIndex n_dofs{0};
    std::string kernel_name;
};

InterfaceMovingMeshSnapshot assemble_interface_moving_mesh_with_path(
    svmp::FE::forms::GeometryTangentPath path,
    bool enable_jit)
{
    constexpr int marker = 19;
    auto mesh = build_two_quad_mesh_with_interface_set("middle");
    auto scalar_space = std::make_shared<H1Space>(ElementType::Quad4, 1);
    auto vector_space = std::make_shared<ProductSpace>(scalar_space, /*components=*/2);

    FESystem sys(mesh);
    sys.setInterfaceMeshFromFaceSet(marker, "middle");
    const auto displacement =
        sys.addMeshDisplacementUnknown("mesh_displacement", vector_space);
    sys.addOperator("moving_interface");

    const auto u =
        svmp::FE::forms::FormExpr::trialFunction(*vector_space, "mesh_displacement");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*vector_space, "v");
    const auto residual =
        (inner(svmp::FE::forms::currentNormal().minus(), v.minus()) +
         inner(svmp::FE::forms::currentNormal().plus(), v.plus()) +
         svmp::FE::forms::currentMeasure().minus() * inner(u.minus(), v.minus()) +
         svmp::FE::forms::currentMeasure().plus() * inner(u.plus(), v.plus()) +
         component(svmp::FE::forms::surfaceJacobian().minus(), 0, 0) *
             v.minus().component(0) +
         component(svmp::FE::forms::surfaceJacobian().plus(), 0, 0) *
             v.plus().component(0)).dI(marker);

    svmp::FE::systems::FormInstallOptions opts;
    opts.compiler_options.geometry_sensitivity.mode =
        svmp::FE::forms::GeometrySensitivityMode::MeshMotionUnknowns;
    opts.compiler_options.geometry_sensitivity.mesh_motion_field = displacement;
    opts.compiler_options.geometry_tangent_path = path;
    opts.compiler_options.use_symbolic_tangent =
        path != svmp::FE::forms::GeometryTangentPath::ADReference;
    opts.compiler_options.jit = svmp::FE::forms::test::makeUnitTestJITOptions();
    opts.compiler_options.jit.enable = enable_jit;

    const auto installed = svmp::FE::systems::installResidualForm(
        sys, "moving_interface", displacement, displacement, residual, opts);
    EXPECT_NE(installed, nullptr);

    sys.setup();
    const auto n_dofs = sys.dofHandler().getNumDofs();
    EXPECT_EQ(n_dofs, 12);

    std::vector<Real> solution(static_cast<std::size_t>(n_dofs), 0.0);
    for (std::size_t i = 0; i < solution.size(); ++i) {
        const Real sign = (i % 2u == 0u) ? Real(1.0) : Real(-1.0);
        solution[i] = sign * (Real(0.010) + Real(0.002) * static_cast<Real>(i));
    }

    SystemStateView state;
    state.u = solution;

    AssemblyRequest req;
    req.op = "moving_interface";
    req.want_matrix = true;
    req.want_vector = true;

    svmp::FE::assembly::DenseSystemView out(n_dofs);
    out.zero();
    const auto result = sys.assemble(req, state, &out, &out);
    EXPECT_TRUE(result.success);

    InterfaceMovingMeshSnapshot snapshot;
    snapshot.n_dofs = n_dofs;
    snapshot.kernel_name = installed ? installed->name() : std::string{};
    snapshot.matrix.resize(static_cast<std::size_t>(n_dofs * n_dofs), 0.0);
    snapshot.vector.resize(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        snapshot.vector[static_cast<std::size_t>(i)] = out.getVectorEntry(i);
        for (GlobalIndex j = 0; j < n_dofs; ++j) {
            snapshot.matrix[static_cast<std::size_t>(i * n_dofs + j)] =
                out.getMatrixEntry(i, j);
        }
    }
    return snapshot;
}

} // namespace

TEST(FESystemMortar, SetupDistributesScopedFacetDofsOnInterfaceFaces)
{
    constexpr int marker = 17;
    auto mesh = build_two_quad_mesh_with_interface_set("middle");

    auto flux_space = std::make_shared<HDivSpace>(ElementType::Quad4, /*order=*/0);
    auto mortar_space = std::make_shared<MortarSpace>(
        std::make_shared<L2Space>(ElementType::Line2, /*order=*/0), marker);

    FESystem sys(mesh);
    sys.setInterfaceMeshFromFaceSet(marker, "middle");
    const auto q = sys.addField(FieldSpec{.name = "q", .space = flux_space, .components = flux_space->value_dimension()});
    const auto lambda = sys.addField(FieldSpec{.name = "lambda", .space = mortar_space});
    sys.setup();

    ASSERT_EQ(sys.fieldDofHandler(lambda).getNumDofs(), 1);

    const auto lambda_offset = sys.fieldDofOffset(lambda);
    const auto cell0_dofs = sys.dofHandler().getCellDofs(0);
    const auto cell1_dofs = sys.dofHandler().getCellDofs(1);
    EXPECT_NE(std::find(cell0_dofs.begin(), cell0_dofs.end(), lambda_offset), cell0_dofs.end());
    EXPECT_NE(std::find(cell1_dofs.begin(), cell1_dofs.end(), lambda_offset), cell1_dofs.end());

    const auto* emap = sys.fieldDofHandler(lambda).getEntityDofMap();
    ASSERT_NE(emap, nullptr);
    const auto iface_face = find_interface_face_id(*mesh);
    ASSERT_GE(iface_face, 0);
    auto face_dofs = emap->getFaceDofs(iface_face);
    ASSERT_EQ(face_dofs.size(), 1u);
    EXPECT_EQ(face_dofs[0], 0);

    EXPECT_EQ(sys.fieldDofOffset(q), 0);
    EXPECT_EQ(sys.fieldDofOffset(lambda), sys.fieldDofHandler(q).getNumDofs());
}

TEST(FESystemMortar, MovingMeshInterfaceSymbolicJITMatchesADReference)
{
    requireLLVMJITOrSkip();

    const auto ad = assemble_interface_moving_mesh_with_path(
        svmp::FE::forms::GeometryTangentPath::ADReference,
        /*enable_jit=*/false);
    const auto symbolic_jit = assemble_interface_moving_mesh_with_path(
        svmp::FE::forms::GeometryTangentPath::SymbolicRequired,
        /*enable_jit=*/true);

    ASSERT_EQ(symbolic_jit.n_dofs, ad.n_dofs);
    ASSERT_EQ(symbolic_jit.vector.size(), ad.vector.size());
    ASSERT_EQ(symbolic_jit.matrix.size(), ad.matrix.size());
    EXPECT_NE(symbolic_jit.kernel_name.find("JITKernelWrapper"), std::string::npos);
    EXPECT_NE(symbolic_jit.kernel_name.find("SymbolicNonlinearFormKernel"), std::string::npos);

    for (std::size_t i = 0; i < symbolic_jit.vector.size(); ++i) {
        SCOPED_TRACE(::testing::Message() << "vector i=" << i);
        EXPECT_NEAR(symbolic_jit.vector[i], ad.vector[i], 1.0e-12);
    }
    for (std::size_t i = 0; i < symbolic_jit.matrix.size(); ++i) {
        SCOPED_TRACE(::testing::Message() << "matrix flat i=" << i);
        EXPECT_NEAR(symbolic_jit.matrix[i], ad.matrix[i], 3.0e-10);
    }
}

TEST(FESystemMortar, MovingMeshInterfaceSymbolicInterpreterMatchesADReference)
{
    const auto ad = assemble_interface_moving_mesh_with_path(
        svmp::FE::forms::GeometryTangentPath::ADReference,
        /*enable_jit=*/false);
    const auto symbolic = assemble_interface_moving_mesh_with_path(
        svmp::FE::forms::GeometryTangentPath::SymbolicRequired,
        /*enable_jit=*/false);

    ASSERT_EQ(symbolic.n_dofs, ad.n_dofs);
    ASSERT_EQ(symbolic.vector.size(), ad.vector.size());
    ASSERT_EQ(symbolic.matrix.size(), ad.matrix.size());

    for (std::size_t i = 0; i < symbolic.vector.size(); ++i) {
        SCOPED_TRACE(::testing::Message() << "vector i=" << i);
        EXPECT_NEAR(symbolic.vector[i], ad.vector[i], 1.0e-12);
    }
    for (std::size_t i = 0; i < symbolic.matrix.size(); ++i) {
        SCOPED_TRACE(::testing::Message() << "matrix flat i=" << i);
        EXPECT_NEAR(symbolic.matrix[i], ad.matrix[i], 3.0e-10);
    }
}

TEST(FESystemMortar, MovingMeshInterfaceSymbolicWithADCheckUsesJITPrimary)
{
    requireLLVMJITOrSkip();

    const auto snapshot = assemble_interface_moving_mesh_with_path(
        svmp::FE::forms::GeometryTangentPath::SymbolicWithADCheck,
        /*enable_jit=*/true);

    EXPECT_NE(snapshot.kernel_name.find("SymbolicADReferenceCheckKernel"), std::string::npos);
    EXPECT_NE(snapshot.kernel_name.find("JITKernelWrapper"), std::string::npos);
    EXPECT_NE(snapshot.kernel_name.find("SymbolicNonlinearFormKernel"), std::string::npos);
}

TEST(FESystemMortar, MatchingInterfaceAssemblyBuildsMortarCouplingBlocks)
{
    constexpr int marker = 17;
    auto mesh = build_two_quad_mesh_with_interface_set("middle");

    auto volume_space = std::make_shared<L2Space>(ElementType::Quad4, /*order=*/0);
    auto mortar_space = std::make_shared<MortarSpace>(
        std::make_shared<L2Space>(ElementType::Line2, /*order=*/0), marker);

    FESystem sys(mesh);
    sys.setInterfaceMeshFromFaceSet(marker, "middle");
    const auto q = sys.addField(FieldSpec{.name = "q", .space = volume_space});
    const auto lambda = sys.addField(FieldSpec{.name = "lambda", .space = mortar_space});
    sys.addOperator("hybrid");
    sys.addCellKernel("hybrid", q, std::make_shared<MassKernel>(1.0));
    sys.addInterfaceFaceKernel("hybrid", marker, lambda, q,
                               std::make_shared<NormalTraceCouplingKernel>(1.0, InterfaceEvaluationSide::Minus));
    sys.addInterfaceFaceKernel("hybrid", marker, lambda, q,
                               std::make_shared<NormalTraceCouplingKernel>(1.0, InterfaceEvaluationSide::Plus));
    sys.addInterfaceFaceKernel("hybrid", marker, q, lambda,
                               std::make_shared<NormalTraceCouplingKernel>(1.0, InterfaceEvaluationSide::Minus));
    sys.addInterfaceFaceKernel("hybrid", marker, q, lambda,
                               std::make_shared<NormalTraceCouplingKernel>(1.0, InterfaceEvaluationSide::Plus));
    sys.addInterfaceFaceKernel("hybrid", marker, lambda,
                               std::make_shared<FacetMassKernel>(0.25));
    sys.setup();

    DenseMatrixView A(sys.dofHandler().getNumDofs());
    SystemStateView state;
    AssemblyRequest req;
    req.op = "hybrid";
    req.want_matrix = true;
    const auto result = sys.assemble(req, state, &A, nullptr);
    ASSERT_TRUE(result.success);

    const auto q_off = sys.fieldDofOffset(q);
    const auto lambda_off = sys.fieldDofOffset(lambda);
    const auto n_q = sys.fieldDofHandler(q).getNumDofs();
    const auto n_lambda = sys.fieldDofHandler(lambda).getNumDofs();

    EXPECT_GT(block_abs_sum(A, q_off, q_off + n_q, q_off, q_off + n_q), 0.0);
    EXPECT_GT(block_abs_sum(A, q_off, q_off + n_q, lambda_off, lambda_off + n_lambda), 0.0);
    EXPECT_GT(block_abs_sum(A, lambda_off, lambda_off + n_lambda, q_off, q_off + n_q), 0.0);
    EXPECT_GT(block_abs_sum(A, lambda_off, lambda_off + n_lambda,
                            lambda_off, lambda_off + n_lambda), 0.0);
}

TEST(FESystemMortar, HybridizedToyCondensationMatchesFullSolve)
{
    constexpr int marker = 17;
    auto mesh = build_two_quad_mesh_with_interface_set("middle");

    auto volume_space = std::make_shared<L2Space>(ElementType::Quad4, /*order=*/0);
    auto mortar_space = std::make_shared<MortarSpace>(
        std::make_shared<L2Space>(ElementType::Line2, /*order=*/0), marker);

    FESystem sys(mesh);
    sys.setInterfaceMeshFromFaceSet(marker, "middle");
    const auto q = sys.addField(FieldSpec{.name = "q", .space = volume_space});
    const auto lambda = sys.addField(FieldSpec{.name = "lambda", .space = mortar_space});
    sys.addOperator("hybrid_solve");
    sys.addCellKernel("hybrid_solve", q, std::make_shared<MassKernel>(1.0));
    sys.addInterfaceFaceKernel("hybrid_solve", marker, lambda, q,
                               std::make_shared<NormalTraceCouplingKernel>(1.0, InterfaceEvaluationSide::Minus));
    sys.addInterfaceFaceKernel("hybrid_solve", marker, lambda, q,
                               std::make_shared<NormalTraceCouplingKernel>(1.0, InterfaceEvaluationSide::Plus));
    sys.addInterfaceFaceKernel("hybrid_solve", marker, q, lambda,
                               std::make_shared<NormalTraceCouplingKernel>(1.0, InterfaceEvaluationSide::Minus));
    sys.addInterfaceFaceKernel("hybrid_solve", marker, q, lambda,
                               std::make_shared<NormalTraceCouplingKernel>(1.0, InterfaceEvaluationSide::Plus));
    sys.addInterfaceFaceKernel("hybrid_solve", marker, lambda,
                               std::make_shared<FacetMassKernel>(0.5));
    sys.addInterfaceFaceKernel("hybrid_solve", marker, lambda,
                               std::make_shared<FacetSourceKernel>(1.0));
    sys.setup();

    DenseMatrixView A(sys.dofHandler().getNumDofs());
    DenseVectorView b(sys.dofHandler().getNumDofs());
    SystemStateView state;
    AssemblyRequest req;
    req.op = "hybrid_solve";
    req.want_matrix = true;
    req.want_vector = true;
    const auto result = sys.assemble(req, state, &A, &b);
    ASSERT_TRUE(result.success);

    const auto q_off = sys.fieldDofOffset(q);
    const auto lambda_off = sys.fieldDofOffset(lambda);
    const int n_q = static_cast<int>(sys.fieldDofHandler(q).getNumDofs());
    const int n_lambda = static_cast<int>(sys.fieldDofHandler(lambda).getNumDofs());
    ASSERT_EQ(n_lambda, 1);

    const auto Aqq = submatrix(A, q_off, q_off + n_q, q_off, q_off + n_q);
    const auto Aql = submatrix(A, q_off, q_off + n_q, lambda_off, lambda_off + n_lambda);
    const auto Alq = submatrix(A, lambda_off, lambda_off + n_lambda, q_off, q_off + n_q);
    const auto All = submatrix(A, lambda_off, lambda_off + n_lambda,
                               lambda_off, lambda_off + n_lambda);
    const auto bq = subvector(b, q_off, q_off + n_q);
    const auto bl = subvector(b, lambda_off, lambda_off + n_lambda);

    std::vector<double> Afull(static_cast<std::size_t>((n_q + n_lambda) * (n_q + n_lambda)), 0.0);
    std::vector<double> bfull(static_cast<std::size_t>(n_q + n_lambda), 0.0);
    for (int i = 0; i < n_q + n_lambda; ++i) {
        bfull[static_cast<std::size_t>(i)] = b.getVectorEntry(i);
        for (int j = 0; j < n_q + n_lambda; ++j) {
            Afull[static_cast<std::size_t>(i) * (n_q + n_lambda) + j] = A.getMatrixEntry(i, j);
        }
    }
    const auto x_full = dense_solve(Afull, bfull, n_q + n_lambda);

    const auto Aqq_inv_bq = dense_solve(Aqq, bq, n_q);
    auto Aqq_inv_Aql_col = dense_solve(Aqq,
                                       std::vector<double>(Aql.begin(), Aql.begin() + n_q),
                                       n_q);

    const auto schur_rhs = bl[0] - mat_vec(Alq, Aqq_inv_bq, n_lambda, n_q)[0];
    const auto schur_mat = All[0] - mat_vec(Alq, Aqq_inv_Aql_col, n_lambda, n_q)[0];
    ASSERT_GT(std::abs(schur_mat), 1e-12);
    const double lambda_condensed = schur_rhs / schur_mat;

    std::vector<double> q_recovered_rhs = bq;
    for (int i = 0; i < n_q; ++i) {
        q_recovered_rhs[static_cast<std::size_t>(i)] -=
            Aql[static_cast<std::size_t>(i)] * lambda_condensed;
    }
    const auto q_condensed = dense_solve(Aqq, q_recovered_rhs, n_q);

    for (int i = 0; i < n_q; ++i) {
        EXPECT_NEAR(q_condensed[static_cast<std::size_t>(i)],
                    x_full[static_cast<std::size_t>(i)], 1e-10);
    }
    EXPECT_NEAR(lambda_condensed, x_full[static_cast<std::size_t>(n_q)], 1e-10);
}

TEST(FESystemMixedDimensional, SetupDistributesContinuousInterfaceFieldOnConnectedInterface)
{
    constexpr int marker = 23;
    auto mesh = build_four_quad_mesh_with_interface_set("middle_vertical");

    auto lambda_space = std::make_shared<H1Space>(ElementType::Line2, /*order=*/1);

    FESystem sys(mesh);
    sys.setInterfaceMeshFromFaceSet(marker, "middle_vertical");
    const auto lambda = sys.addInterfaceField("lambda", lambda_space, marker);
    sys.setup();

    const auto& iface = sys.interfaceMesh(marker);
    ASSERT_EQ(iface.n_faces(), 2u);
    ASSERT_EQ(sys.fieldDofHandler(lambda).getNumDofs(), 3);

    const auto first_face_dofs = sys.fieldDofHandler(lambda).getCellDofs(0);
    const auto second_face_dofs = sys.fieldDofHandler(lambda).getCellDofs(1);
    ASSERT_EQ(first_face_dofs.size(), 2u);
    ASSERT_EQ(second_face_dofs.size(), 2u);

    std::vector<GlobalIndex> shared;
    for (GlobalIndex dof : first_face_dofs) {
        if (std::find(second_face_dofs.begin(), second_face_dofs.end(), dof) != second_face_dofs.end()) {
            shared.push_back(dof);
        }
    }
    ASSERT_EQ(shared.size(), 1u);

    const auto* emap = sys.fieldDofHandler(lambda).getEntityDofMap();
    ASSERT_NE(emap, nullptr);

    GlobalIndex middle_dof = -1;
    for (std::size_t lv = 0; lv < iface.vertex_gids().size(); ++lv) {
        if (iface.vertex_gids()[lv] != 4) {
            continue;
        }
        const auto vertex_dofs = emap->getVertexDofs(static_cast<GlobalIndex>(lv));
        ASSERT_EQ(vertex_dofs.size(), 1u);
        middle_dof = vertex_dofs[0];
        break;
    }

    ASSERT_GE(middle_dof, 0);
    EXPECT_EQ(shared.front(), middle_dof);
}

TEST(FESystemMixedDimensional, InterfaceFieldHistoryAssemblyUsesPreviousState)
{
    constexpr int marker = 23;
    auto mesh = build_four_quad_mesh_with_interface_set("middle_vertical");

    auto lambda_space = std::make_shared<H1Space>(ElementType::Line2, /*order=*/1);

    FESystem sys(mesh);
    sys.setInterfaceMeshFromFaceSet(marker, "middle_vertical");
    const auto lambda = sys.addInterfaceField("lambda", lambda_space, marker);
    sys.addOperator("dt_interface");
    sys.addInterfaceFaceKernel("dt_interface", marker, lambda,
                               std::make_shared<InterfaceDtResidualKernel>());
    EXPECT_TRUE(sys.isTransient());
    EXPECT_EQ(sys.temporalOrder(), 1);
    sys.setup();

    ASSERT_EQ(sys.fieldDofHandler(lambda).getNumDofs(), 3);
    const auto lambda_offset = sys.fieldDofOffset(lambda);

    std::vector<Real> u_n = {2.0, 2.0, 2.0};
    std::vector<Real> u_prev = {1.0, 1.0, 1.0};

    SystemStateView state;
    state.dt = 1.0;
    state.u = u_n;
    state.u_prev = u_prev;

    DenseVectorView residual(sys.dofHandler().getNumDofs());
    residual.zero();

    AssemblyRequest req;
    req.op = "dt_interface";
    req.want_vector = true;

    TransientSystem transient(sys, std::make_shared<BackwardDifferenceIntegrator>());
    const auto result = transient.assemble(req, state, nullptr, &residual);
    ASSERT_TRUE(result.success);

    std::vector<Real> values;
    values.reserve(static_cast<std::size_t>(sys.fieldDofHandler(lambda).getNumDofs()));
    for (GlobalIndex i = 0; i < sys.fieldDofHandler(lambda).getNumDofs(); ++i) {
        values.push_back(residual.getVectorEntry(lambda_offset + i));
    }
    std::sort(values.begin(), values.end());

    ASSERT_EQ(values.size(), 3u);
    EXPECT_NEAR(values[0], 0.5, 1e-12);
    EXPECT_NEAR(values[1], 0.5, 1e-12);
    EXPECT_NEAR(values[2], 1.0, 1e-12);
}

TEST(FESystemMixedDimensional, InterfaceFieldCouplingConservesSymmetricExchange)
{
    constexpr int marker = 17;
    auto mesh = build_two_quad_mesh_with_interface_set("middle");

    auto volume_space = std::make_shared<L2Space>(ElementType::Quad4, /*order=*/0);
    auto interface_space = std::make_shared<L2Space>(ElementType::Line2, /*order=*/0);

    FESystem sys(mesh);
    sys.setInterfaceMeshFromFaceSet(marker, "middle");
    const auto q = sys.addField(FieldSpec{.name = "q", .space = volume_space});
    const auto lambda = sys.addInterfaceField("lambda", interface_space, marker);
    sys.addOperator("mixeddim_balance");
    sys.addCellKernel("mixeddim_balance", q, std::make_shared<MassKernel>(1.0));
    sys.addInterfaceFaceKernel("mixeddim_balance", marker, q, lambda,
                               std::make_shared<NormalTraceCouplingKernel>(1.0, InterfaceEvaluationSide::Minus));
    sys.addInterfaceFaceKernel("mixeddim_balance", marker, q, lambda,
                               std::make_shared<NormalTraceCouplingKernel>(1.0, InterfaceEvaluationSide::Plus));
    sys.addInterfaceFaceKernel("mixeddim_balance", marker, lambda, q,
                               std::make_shared<NormalTraceCouplingKernel>(1.0, InterfaceEvaluationSide::Minus));
    sys.addInterfaceFaceKernel("mixeddim_balance", marker, lambda, q,
                               std::make_shared<NormalTraceCouplingKernel>(1.0, InterfaceEvaluationSide::Plus));
    sys.addInterfaceFaceKernel("mixeddim_balance", marker, lambda,
                               std::make_shared<FacetSourceKernel>(1.0));
    sys.setup();

    DenseMatrixView A(sys.dofHandler().getNumDofs());
    DenseVectorView b(sys.dofHandler().getNumDofs());
    SystemStateView state;
    AssemblyRequest req;
    req.op = "mixeddim_balance";
    req.want_matrix = true;
    req.want_vector = true;
    const auto result = sys.assemble(req, state, &A, &b);
    ASSERT_TRUE(result.success);

    const auto q_off = sys.fieldDofOffset(q);
    const auto lambda_off = sys.fieldDofOffset(lambda);
    const int n_q = static_cast<int>(sys.fieldDofHandler(q).getNumDofs());
    const int n_lambda = static_cast<int>(sys.fieldDofHandler(lambda).getNumDofs());
    ASSERT_EQ(n_q, 2);
    ASSERT_EQ(n_lambda, 1);

    std::vector<double> Afull(static_cast<std::size_t>((n_q + n_lambda) * (n_q + n_lambda)), 0.0);
    std::vector<double> bfull(static_cast<std::size_t>(n_q + n_lambda), 0.0);
    for (int i = 0; i < n_q + n_lambda; ++i) {
        bfull[static_cast<std::size_t>(i)] = b.getVectorEntry(i);
        for (int j = 0; j < n_q + n_lambda; ++j) {
            Afull[static_cast<std::size_t>(i) * (n_q + n_lambda) + j] = A.getMatrixEntry(i, j);
        }
    }

    const auto x = dense_solve(Afull, bfull, n_q + n_lambda);

    EXPECT_GT(block_abs_sum(A, q_off, q_off + n_q, lambda_off, lambda_off + n_lambda), 0.0);
    EXPECT_GT(block_abs_sum(A, lambda_off, lambda_off + n_lambda, q_off, q_off + n_q), 0.0);

    EXPECT_NEAR(x[0], 0.5, 1e-12);
    EXPECT_NEAR(x[1], 0.5, 1e-12);
    EXPECT_NEAR(x[2], -0.5, 1e-12);
    EXPECT_NEAR(x[0] + x[1], 1.0, 1e-12);
}
