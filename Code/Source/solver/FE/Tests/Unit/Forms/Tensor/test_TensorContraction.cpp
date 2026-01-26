/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/Einsum.h"
#include "Forms/JIT/KernelIR.h"
#include "Forms/Tensor/TensorContraction.h"
#include "Forms/Index.h"

#include <gtest/gtest.h>

namespace svmp::FE::forms::tensor {

TEST(TensorContraction, AnalyzeFreeAndBoundIndices)
{
    const auto A = FormExpr::asVector({FormExpr::constant(1.0), FormExpr::constant(2.0), FormExpr::constant(3.0)});
    const auto B = FormExpr::asVector({FormExpr::constant(4.0), FormExpr::constant(5.0), FormExpr::constant(6.0)});

    forms::Index i("i");
    const auto expr_bound = A(i) * B(i);
    const auto a1 = analyzeContractions(expr_bound);
    EXPECT_TRUE(a1.ok);
    EXPECT_TRUE(a1.free_indices.empty());
    EXPECT_EQ(a1.bound_indices.size(), 1u);

    const auto expr_free = A(i);
    const auto a2 = analyzeContractions(expr_free);
    EXPECT_TRUE(a2.ok);
    EXPECT_EQ(a2.free_indices.size(), 1u);
}

TEST(TensorContraction, KernelIRCanonicalizesIndexIds)
{
    const auto A = FormExpr::asVector({FormExpr::constant(1.0), FormExpr::constant(2.0), FormExpr::constant(3.0)});
    const auto B = FormExpr::asVector({FormExpr::constant(4.0), FormExpr::constant(5.0), FormExpr::constant(6.0)});

    forms::Index i1("i");
    const auto expr1 = A(i1) * B(i1);

    forms::Index i2("i");
    const auto expr2 = A(i2) * B(i2);

    const auto kir1 = forms::jit::lowerToKernelIR(expr1).ir.stableHash64();
    const auto kir2 = forms::jit::lowerToKernelIR(expr2).ir.stableHash64();
    EXPECT_EQ(kir1, kir2);
}

TEST(TensorContraction, IndexedAccessAvoidsScalarExpansionIRBloat)
{
    const auto makeMat = [](Real scale) {
        std::array<std::array<Real, 3>, 3> m{};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                m[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                    scale * (1.0 + static_cast<Real>(i) + 0.1 * static_cast<Real>(j));
            }
        }
        return m;
    };

    const auto A = FormExpr::coefficient("A", [=](Real, Real, Real) { return makeMat(1.0); });
    const auto B = FormExpr::coefficient("B", [=](Real, Real, Real) { return makeMat(2.0); });
    const auto C = FormExpr::coefficient("C", [=](Real, Real, Real) { return makeMat(3.0); });
    const auto D = FormExpr::coefficient("D", [=](Real, Real, Real) { return makeMat(4.0); });

    forms::Index i("i");
    forms::Index j("j");
    forms::Index k("k");
    forms::Index l("l");

    // Fully-contracted 4-index sum: sum_{i,j,k,l} A_ij B_jk C_kl D_li (81 terms in 3D if expanded).
    const auto expr_indexed = A(i, j) * B(j, k) * C(k, l) * D(l, i);
    const auto expr_expanded = forms::einsum(expr_indexed);

    forms::jit::KernelIRBuildOptions opts;
    opts.cse = false;
    opts.canonicalize_commutative = false;

    const auto ir_indexed = forms::jit::lowerToKernelIR(expr_indexed, opts).ir.opCount();
    const auto ir_expanded = forms::jit::lowerToKernelIR(expr_expanded, opts).ir.opCount();

    EXPECT_LT(ir_indexed, 64u);
    EXPECT_GT(ir_expanded, ir_indexed * 10u);
}

} // namespace svmp::FE::forms::tensor
