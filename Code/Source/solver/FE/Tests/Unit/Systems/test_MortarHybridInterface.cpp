/**
 * @file test_MortarHybridInterface.cpp
 * @brief Mortar and hybridized interface assembly regressions
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"

#include "Assembly/AssemblyKernel.h"
#include "Assembly/GlobalSystemView.h"

#include "Dofs/EntityDofMap.h"

#include "Spaces/HDivSpace.h"
#include "Spaces/L2Space.h"
#include "Spaces/MortarSpace.h"

#include "Mesh/Mesh.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Topology/CellShape.h"

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

using svmp::FE::spaces::HDivSpace;
using svmp::FE::spaces::L2Space;
using svmp::FE::spaces::MortarSpace;

using svmp::FE::systems::AssemblyRequest;
using svmp::FE::systems::FESystem;
using svmp::FE::systems::FieldSpec;
using svmp::FE::systems::SystemStateView;

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
