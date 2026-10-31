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

#include "Forms/FormCompiler.h"
#include "Forms/FormExpr.h"
#include "Forms/SymbolicDifferentiation.h"
#include "Forms/Vocabulary.h"
#include "Spaces/GlobalSpace.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Systems/SystemManifest.h"

#include <memory>
#include <stdexcept>
#include <string>

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

namespace {

/// The message of the std::invalid_argument thrown by @p f, or "" when none.
template <class F>
std::string invalidArgumentMessage(F&& f)
{
    try {
        (void)f();
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return {};
}

} // namespace

TEST(FormExprOperators, TheTimeDerivativeOfACompositeIsRefusedWhereItIsBuilt)
{
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    const auto u = FormExpr::stateField(0, *space, "u");
    EXPECT_NO_THROW((void)dt(u));
    EXPECT_NO_THROW((void)dt(FormExpr::trialFunction(*space, "w"), 2));
    EXPECT_NO_THROW((void)dt(FormExpr::discreteField(0, *space, "u")));
    try {
        (void)dt(2.0 * u);
        FAIL() << "dt() of a composite builds a term no stencil evaluates";
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("the operand of dt() is a field"), std::string::npos) << message;
        EXPECT_NE(message.find((2.0 * u).toString()), std::string::npos) << message;
    }
}

TEST(FormExprEmptyOperands, CombiningAnEmptyFormThrowsAndNamesTheOperation)
{
    Symbols s;
    const FormExpr empty;

    const auto plus = invalidArgumentMessage([&] { return empty + s.u.dx(); });
    EXPECT_NE(plus.find("the left operand of '+'"), std::string::npos) << plus;
    EXPECT_NE(plus.find("sumOf"), std::string::npos) << plus;
    const auto minus = invalidArgumentMessage([&] { return s.u - empty; });
    EXPECT_NE(minus.find("the right operand of '-'"), std::string::npos) << minus;
    const auto both = invalidArgumentMessage([&] { return empty * empty; });
    EXPECT_NE(both.find("each operand of '*'"), std::string::npos) << both;
    const auto inner_message = invalidArgumentMessage([&] { return inner(s.w, empty); });
    EXPECT_NE(inner_message.find("inner()"), std::string::npos) << inner_message;
    const auto conditional_message =
        invalidArgumentMessage([&] { return conditional(gt(s.u, 0.0), s.u, empty); });
    EXPECT_NE(conditional_message.find("operand 3 of conditional()"), std::string::npos)
        << conditional_message;

    EXPECT_THROW((void)(empty / s.u), std::invalid_argument);
    EXPECT_THROW((void)(empty * 2.0), std::invalid_argument);
    EXPECT_THROW((void)(2.0 * empty), std::invalid_argument);
    EXPECT_THROW((void)(empty / 2.0), std::invalid_argument);
    EXPECT_THROW((void)(1.0 + empty), std::invalid_argument);
    EXPECT_THROW((void)(empty - 1.0), std::invalid_argument);
    EXPECT_THROW((void)outer(empty, s.w), std::invalid_argument);
    EXPECT_THROW((void)cross(s.w, empty), std::invalid_argument);
    EXPECT_THROW((void)doubleContraction(grad(s.w), empty), std::invalid_argument);
    EXPECT_THROW((void)pow(empty, 2.0), std::invalid_argument);
    EXPECT_THROW((void)min(empty, s.u), std::invalid_argument);
    EXPECT_THROW((void)max(s.u, empty), std::invalid_argument);
    EXPECT_THROW((void)lt(empty, s.u), std::invalid_argument);
    EXPECT_THROW((void)ne(s.u, empty), std::invalid_argument);
    EXPECT_THROW((void)smoothAbs(s.u, empty), std::invalid_argument);
    EXPECT_THROW((void)smoothMax(s.u, empty, c(1e-3)), std::invalid_argument);
    EXPECT_THROW((void)powm(grad(s.w), empty), std::invalid_argument);
    EXPECT_THROW((void)FormExpr::matrixExpDirectionalDerivative(grad(s.w), empty), std::invalid_argument);
}

TEST(FormExprEmptyOperands, UnaryOperationsAndMeasuresKeepAnEmptyFormEmpty)
{
    Symbols s;
    const FormExpr empty;
    // An empty FormExpr means "no term": it may be transformed, tested and
    // folded with sumOf, but not combined with another operand.
    EXPECT_FALSE((-empty).isValid());
    EXPECT_FALSE(grad(empty).isValid());
    EXPECT_FALSE(jump(empty).isValid());
    EXPECT_FALSE(empty.dx().isValid());
    EXPECT_FALSE(empty.ds(1).isValid());
    EXPECT_FALSE(empty.dI(3).isValid());
    const auto R = (s.u * s.v).dx();
    expectSameTree(sumOf(empty, R), R);
    expectSameTree(sumOf(R, R), R + R);
}

TEST(FormCompilerTraceOperators, JumpAndAverageOutsideFacetIntegralsAreRejectedAndNamed)
{
    spaces::H1Space space(ElementType::Triangle3, 1);
    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto base = (u * v).dx();

    FormCompiler compiler;
    const auto cell = invalidArgumentMessage([&] { return compiler.compileResidual(base + (jump(u) * v).dx()); });
    EXPECT_NE(cell.find("jump()"), std::string::npos) << cell;
    EXPECT_NE(cell.find("a cell integral (dx)"), std::string::npos) << cell;
    EXPECT_NE(cell.find("[[u]]"), std::string::npos) << cell;

    const auto boundary =
        invalidArgumentMessage([&] { return compiler.compileResidual(base + (avg(u) * v).ds(2)); });
    EXPECT_NE(boundary.find("avg()"), std::string::npos) << boundary;
    EXPECT_NE(boundary.find("exterior-boundary integral (ds)"), std::string::npos) << boundary;

    const auto cut = invalidArgumentMessage([&] {
        return compiler.compileResidual(base + (jump(u) * v).dCutVolume(4, CutVolumeSide::Negative));
    });
    EXPECT_NE(cut.find("cut-volume integral"), std::string::npos) << cut;

    // Facet integrals keep them.
    EXPECT_NO_THROW((void)compiler.compileResidual(base + (jump(u) * jump(v)).dS()));
    EXPECT_NO_THROW((void)compiler.compileResidual(base + (avg(u) * jump(v)).dI(3)));
}

TEST(FormCompilerTraceOperators, MixedFormsCheckEveryTerm)
{
    spaces::H1Space velocity_space(ElementType::Triangle3, 2);
    spaces::H1Space pressure_space(ElementType::Triangle3, 1);
    const auto u = TrialFunction(velocity_space, "u");
    const auto p = TrialFunction(pressure_space, "p");
    const auto v = TestFunction(velocity_space, "v");
    const auto q = TestFunction(pressure_space, "q");
    FormCompiler compiler;
    const auto message = invalidArgumentMessage([&] {
        return compiler.compileMixed((u * v).dx() + (jump(p) * q).dx() + (p * q).dx());
    });
    EXPECT_NE(message.find("jump()"), std::string::npos) << message;
    EXPECT_NE(message.find("[[p]]"), std::string::npos) << message;
}

TEST(GlobalMeasure, IsAPointMassTermOfItsOwn)
{
    spaces::GlobalSpace r1;
    const auto lam = FormExpr::stateField(/*field=*/1, r1, "lambda");
    const auto mu = FormExpr::testFunction(/*field=*/1, r1, "mu");
    const auto term = (mu * FormExpr::constant(2.0)).dGlobal();
    ASSERT_TRUE(term.isValid());
    EXPECT_EQ(term.node()->type(), FormExprType::GlobalIntegral);
    EXPECT_NE(term.toString().find("dGlobal"), std::string::npos) << term.toString();
    EXPECT_FALSE(FormExpr{}.dGlobal().isValid());

    // Rebuilding keeps the measure.
    const auto rebuilt = term.transformNodes([](const FormExprNode&) { return std::nullopt; });
    expectSameTree(rebuilt, term);

    // Differentiation goes through it: d/dlambda of (lambda^2 mu) dGlobal.
    const auto quadratic = (lam * lam * mu).dGlobal();
    const auto tangent = differentiateResidual(quadratic, /*field=*/FieldId{1});
    ASSERT_TRUE(tangent.isValid());
    EXPECT_EQ(tangent.node()->type(), FormExprType::GlobalIntegral) << tangent.toString();
}

TEST(GlobalMeasure, IsNotAnElementKernelTerm)
{
    spaces::H1Space h1(ElementType::Triangle3, 1);
    spaces::GlobalSpace r1;
    const auto v = TestFunction(h1, "v");
    const auto u = TrialFunction(h1, "u");
    FormCompiler compiler;
    const auto message = invalidArgumentMessage([&] {
        return compiler.compileResidual((u * v).dx() + (FormExpr::constant(1.0) * v).dGlobal());
    });
    EXPECT_NE(message.find("dGlobal()"), std::string::npos) << message;
    EXPECT_NE(message.find("installFormulation"), std::string::npos) << message;
}

} // namespace svmp::FE::forms::test
