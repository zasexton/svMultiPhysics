/**
 * @file test_FormExprOperators.cpp
 * @brief Operators of FormExpr: mixed Real operands, empty operands, and the
 *        integral domains of the trace operators
 *
 * The mixed Real/FormExpr operators are sugar only: each must build exactly
 * the tree of the spelled-out constant, so that kernels, manifests and JIT
 * cache keys do not change when a formulation uses them.  Each comparison uses
 * the ordered tree hash of the system manifest and the printed tree.
 */

#include <gtest/gtest.h>

#include "Forms/FormExpr.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Systems/SystemManifest.h"

#include <memory>

namespace svmp::FE::forms::test {
namespace {

void expectSameTree(const FormExpr& actual, const FormExpr& expected)
{
    ASSERT_TRUE(actual.isValid());
    ASSERT_TRUE(expected.isValid());
    EXPECT_EQ(systems::formExprTreeHash(actual), systems::formExprTreeHash(expected));
    EXPECT_EQ(actual.toString(), expected.toString());
}

struct Symbols {
    std::shared_ptr<spaces::H1Space> scalar{
        std::make_shared<spaces::H1Space>(ElementType::Triangle3, 1)};
    std::shared_ptr<spaces::ProductSpace> vector{
        std::make_shared<spaces::ProductSpace>(scalar, 2)};
    FormExpr u{StateField(FieldId{0}, *scalar, "u")};
    FormExpr v{TestField(FieldId{0}, *scalar, "v")};
    FormExpr w{StateField(FieldId{1}, *vector, "w")};
};

FormExpr c(Real value) { return FormExpr::constant(value); }

} // namespace

TEST(FormExprOperators, RealArithmeticBuildsTheSpelledOutTrees)
{
    Symbols s;
    expectSameTree(1.0 + s.u, c(1.0) + s.u);
    expectSameTree(s.u + 1.0, s.u + c(1.0));
    expectSameTree(1.0 - s.u, c(1.0) - s.u);
    expectSameTree(s.u - 1.0, s.u - c(1.0));
    expectSameTree(1.0 / s.u, c(1.0) / s.u);
    expectSameTree(s.u / 2.0, s.u / c(2.0));
    expectSameTree(2.0 * s.u, c(2.0) * s.u);
    expectSameTree(s.u * 2.0, c(2.0) * s.u);
    // Integer literals convert to Real.
    expectSameTree(2 - s.u, c(2.0) - s.u);
    expectSameTree(s.u + 3, s.u + c(3.0));
}

TEST(FormExprOperators, RealArithmeticKeepsOperandOrderAndAssociation)
{
    Symbols s;
    // (u - 1) * v and 1 - u * v keep the association of the written expression.
    expectSameTree((s.u - 1.0) * s.v, (s.u - c(1.0)) * s.v);
    expectSameTree(1.0 - s.u * s.v, c(1.0) - s.u * s.v);
    // The constant stays on the side it was written on: a + x and x + a differ.
    EXPECT_NE(systems::formExprTreeHash(1.0 + s.u), systems::formExprTreeHash(s.u + 1.0));
    // A scalar times a vector expression.
    expectSameTree(0.5 * inner(s.w, s.w) - 1.0, c(0.5) * inner(s.w, s.w) - c(1.0));
}

TEST(FormExprOperators, RealArgumentsOfScalarFunctionsBuildTheSpelledOutTrees)
{
    Symbols s;
    expectSameTree(pow(s.u, 2.0), pow(s.u, c(2.0)));
    expectSameTree(pow(h(), 4), pow(h(), c(4.0)));
    expectSameTree(min(s.u, 0.0), min(s.u, c(0.0)));
    expectSameTree(min(0.0, s.u), min(c(0.0), s.u));
    expectSameTree(max(s.u, 0.0), max(s.u, c(0.0)));
    expectSameTree(max(0.0, s.u), max(c(0.0), s.u));
    expectSameTree(lt(s.u, 0.5), lt(s.u, c(0.5)));
    expectSameTree(lt(0.5, s.u), lt(c(0.5), s.u));
    expectSameTree(le(s.u, 0.5), le(s.u, c(0.5)));
    expectSameTree(le(0.5, s.u), le(c(0.5), s.u));
    expectSameTree(gt(s.u, 0.5), gt(s.u, c(0.5)));
    expectSameTree(gt(0.5, s.u), gt(c(0.5), s.u));
    expectSameTree(ge(s.u, 0.5), ge(s.u, c(0.5)));
    expectSameTree(ge(0.5, s.u), ge(c(0.5), s.u));
    expectSameTree(eq(domainId(), 3), eq(domainId(), c(3.0)));
    expectSameTree(eq(3, domainId()), eq(c(3.0), domainId()));
    expectSameTree(ne(s.u, 0.0), ne(s.u, c(0.0)));
    expectSameTree(ne(0.0, s.u), ne(c(0.0), s.u));
}

TEST(FormExprOperators, RealArithmeticWorksOnConstitutiveCallsAndDerivedSymbols)
{
    Symbols s;
    // A ConstitutiveCall converts to its call expression; a measure-free
    // stand-in is enough to exercise overload resolution.
    const ConstitutiveCall call{s.u};
    expectSameTree(call - 1.0, s.u - c(1.0));
    expectSameTree(1.0 + call, c(1.0) + s.u);
}

} // namespace svmp::FE::forms::test
