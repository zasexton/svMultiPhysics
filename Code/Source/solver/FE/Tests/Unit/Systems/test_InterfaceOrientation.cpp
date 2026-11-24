/**
 * @file test_InterfaceOrientation.cpp
 * @brief An interface's sides named by volume: InterfaceBetween::trace, normalFrom and average
 *
 * "From A toward B" is stated once and means the same on a fitted interface,
 * whichever subdomain the interface mesh puts on the minus side, and on a
 * generated interface, whose minus side is the negative one.  Each case
 * assembles a residual and sums it over the test functions of each side.
 *
 * Fitted: the unit square split at x = 1/2 into the subdomains 1 (left) and 2
 * (right), a discontinuous P1 field c = 2 on the left cells and 5 on the right
 * ones, the interface x = 1/2 of length 1.  The cells are numbered from the
 * left or from the right, which puts either subdomain on the minus side.
 *
 * Generated: the level set phi = x - 0.43 on the unit square, a P1 field on
 * each side, c_neg = 2 and c_pos = 5, the interface x = 0.43 of length 1.
 *
 * In both: normalFrom(left or negative side) is +e_x; the upwind flux
 * (beta . n_AB) c_up (v_A - v_B) moves the value of the upstream side, and
 * its residual is unchanged when A and B are exchanged; a 1:100 weighted
 * average names its weights by side, in either order.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Forms/IntegrationDomain.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Spaces/L2Space.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Systems/SystemManifest.h"

#include "TraceMultiplierTestMeshes.h"

#include "Mesh/Core/InterfaceMesh.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace svmp::FE::test_interface_orientation {
namespace {

using forms::FormExpr;
using forms::InterfaceBetween;
using forms::VolumeDomain;
using systems::FESystem;
using systems::FieldSpec;

constexpr int kGamma = 31;
constexpr int kLeftDomain = 1;
constexpr int kRightDomain = 2;
constexpr int kGenerated = 41;

/**
 * The n x n Quad4 grid of the unit square (n even), subdomain 1 left of
 * x = 1/2 and 2 right of it, the interior faces on x = 1/2 in the face set
 * "gamma".  With @p from_the_right the cells of a row are numbered from
 * x = 1 to x = 0, so a cell of subdomain 2 has the lower id on each face of
 * gamma.
 */
std::shared_ptr<svmp::Mesh> splitSquare(int n, bool from_the_right)
{
    auto base = std::make_shared<svmp::MeshBase>();
    const auto id = [&](int i, int j) { return static_cast<svmp::index_t>(j * (n + 1) + i); };
    std::vector<svmp::real_t> X;
    for (int j = 0; j <= n; ++j) {
        for (int i = 0; i <= n; ++i) {
            X.push_back(static_cast<svmp::real_t>(static_cast<Real>(i) / static_cast<Real>(n)));
            X.push_back(static_cast<svmp::real_t>(static_cast<Real>(j) / static_cast<Real>(n)));
        }
    }
    std::vector<svmp::offset_t> offsets{0};
    std::vector<svmp::index_t> conn;
    std::vector<svmp::CellShape> shapes;
    std::vector<int> labels;
    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    for (int cy = 0; cy < n; ++cy) {
        for (int k = 0; k < n; ++k) {
            const int cx = from_the_right ? n - 1 - k : k;
            conn.insert(conn.end(), {id(cx, cy), id(cx + 1, cy), id(cx + 1, cy + 1), id(cx, cy + 1)});
            offsets.push_back(static_cast<svmp::offset_t>(conn.size()));
            shapes.push_back(shape);
            labels.push_back(2 * cx < n ? kLeftDomain : kRightDomain);
        }
    }
    base->build_from_arrays(/*spatial_dim=*/2, X, offsets, conn, shapes);
    base->finalize();
    for (std::size_t c = 0; c < labels.size(); ++c) {
        base->set_region_label(static_cast<svmp::index_t>(c), labels[c]);
    }
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(base->n_faces()); ++f) {
        if (base->face_cells(f)[1] == svmp::INVALID_INDEX) {
            continue;
        }
        bool on_gamma = true;
        for (const auto v : base->face_vertices(f)) {
            on_gamma = on_gamma && std::abs(static_cast<Real>(base->get_vertex_coords(v)[0]) - 0.5) < 1e-12;
        }
        if (on_gamma) {
            base->add_to_set(svmp::EntityKind::Face, "gamma", f);
        }
    }
    return svmp::create_mesh(std::move(base));
}

/// The assembled residual of operator "op" at @p state.
std::vector<Real> assembleResidual(FESystem& sys, const std::vector<Real>& state)
{
    const auto n_dofs = sys.dofHandler().getNumDofs();
    systems::SystemStateView view;
    view.u = state;
    systems::AssemblyRequest request;
    request.op = "op";
    request.want_vector = true;
    assembly::DenseVectorView R(n_dofs);
    R.zero();
    const auto result = sys.assemble(request, view, nullptr, &R);
    EXPECT_TRUE(result.success);
    std::vector<Real> out(static_cast<std::size_t>(n_dofs));
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        out[static_cast<std::size_t>(i)] = R.getVectorEntry(i);
    }
    return out;
}

systems::FormInstallOptions interpreted()
{
    systems::FormInstallOptions options;
    options.compiler_options.jit.enable = false;
    return options;
}

/// The direction (1, 0).
FormExpr ex() { return forms::as_vector({FormExpr::constant(1.0), FormExpr::constant(0.0)}); }

FormExpr velocity(Real b) { return forms::as_vector({FormExpr::constant(b), FormExpr::constant(0.0)}); }

/// The upwind flux from A toward B of the value c_A of A and c_B of B,
/// tested with v_A - v_B.
FormExpr upwindFlux(const InterfaceBetween& I,
                    const VolumeDomain& A,
                    const VolumeDomain& B,
                    const FormExpr& beta,
                    const FormExpr& cA,
                    const FormExpr& cB,
                    const FormExpr& vA,
                    const FormExpr& vB)
{
    const auto bn = forms::inner(beta, I.normalFrom(A));
    const auto up = forms::conditional(forms::ge(bn, FormExpr::constant(0.0)), I.trace(cA, A), I.trace(cB, B));
    return (bn * up * (I.trace(vA, A) - I.trace(vB, B))).dI(I.domain());
}

// ---------------------------------------------------------------------------
// Fitted interface
// ---------------------------------------------------------------------------

struct FittedCase {
    std::unique_ptr<FESystem> system;
    InterfaceBetween interface{0, 0, 1};
    FieldId c{INVALID_FIELD_ID};
};

using FittedTerms = std::function<FormExpr(const InterfaceBetween&, const FormExpr& c, const FormExpr& v)>;

/// Installs (0 c v).dx() + terms(I, c, v) for the discontinuous P1 field c.
FittedCase fittedCase(bool from_the_right, const FittedTerms& terms)
{
    FittedCase out;
    out.system = std::make_unique<FESystem>(splitSquare(4, from_the_right));
    auto& sys = *out.system;
    sys.setInterfaceMeshFromFaceSet(kGamma, "gamma");
    auto space = std::make_shared<spaces::L2Space>(ElementType::Quad4, 1);
    out.c = sys.addField(FieldSpec{.name = "c", .space = space});
    sys.addOperator("op");
    out.interface = systems::interfaceBetween(sys, VolumeDomain::cells(kLeftDomain),
                                              VolumeDomain::cells(kRightDomain), kGamma);
    const auto c = forms::StateField(out.c, *space, "c");
    const auto v = forms::TestField(out.c, *space, "v");
    (void)systems::installFormulation(sys, "op", {out.c},
                                      (FormExpr::constant(0.0) * c * v).dx() + terms(out.interface, c, v),
                                      interpreted());
    sys.setup();
    return out;
}

/// c = 2 on the cells of subdomain 1, 5 on those of subdomain 2.
std::vector<Real> fittedState(const FittedCase& problem)
{
    auto& sys = *problem.system;
    std::vector<Real> state(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 0.0);
    const auto offset = sys.fieldDofOffset(problem.c);
    const auto& mesh = sys.meshAccess();
    for (GlobalIndex cell = 0; cell < mesh.numCells(); ++cell) {
        const Real value = mesh.getCellDomainId(cell) == kLeftDomain ? 2.0 : 5.0;
        for (const auto dof : sys.fieldDofHandler(problem.c).getCellDofs(cell)) {
            state[static_cast<std::size_t>(offset + dof)] = value;
        }
    }
    return state;
}

struct SideSums {
    Real left{0.0};
    Real right{0.0};
};

/// The residual summed over the test functions of each subdomain.
SideSums fittedSums(const FittedCase& problem, const std::vector<Real>& residual)
{
    auto& sys = *problem.system;
    const auto offset = sys.fieldDofOffset(problem.c);
    const auto& mesh = sys.meshAccess();
    SideSums sums;
    for (GlobalIndex cell = 0; cell < mesh.numCells(); ++cell) {
        Real& sum = mesh.getCellDomainId(cell) == kLeftDomain ? sums.left : sums.right;
        for (const auto dof : sys.fieldDofHandler(problem.c).getCellDofs(cell)) {
            sum += residual[static_cast<std::size_t>(offset + dof)];
        }
    }
    return sums;
}

SideSums fittedResidualSums(bool from_the_right, const FittedTerms& terms)
{
    const auto problem = fittedCase(from_the_right, terms);
    return fittedSums(problem, assembleResidual(*problem.system, fittedState(problem)));
}

std::vector<Real> fittedResidual(bool from_the_right, const FittedTerms& terms)
{
    const auto problem = fittedCase(from_the_right, terms);
    return assembleResidual(*problem.system, fittedState(problem));
}

const VolumeDomain kLeft = VolumeDomain::cells(kLeftDomain);
const VolumeDomain kRight = VolumeDomain::cells(kRightDomain);

} // namespace

TEST(InterfaceOrientationFitted, EitherSubdomainCanBeTheMinusSide)
{
    const auto from_left = fittedCase(false, [](const InterfaceBetween& I, const FormExpr& c, const FormExpr& v) {
        return (I.trace(c, I.minusSide()) * I.trace(v, I.minusSide())).dI(I.domain());
    });
    const auto from_right = fittedCase(true, [](const InterfaceBetween& I, const FormExpr& c, const FormExpr& v) {
        return (I.trace(c, I.minusSide()) * I.trace(v, I.minusSide())).dI(I.domain());
    });
    EXPECT_NE(from_left.interface.minusSide(), from_right.interface.minusSide());
    for (const auto* problem : {&from_left, &from_right}) {
        const auto& I = problem->interface;
        EXPECT_FALSE(I.isGenerated());
        EXPECT_EQ(I.otherSide(I.minusSide()), I.plusSide());
        EXPECT_EQ(I.otherSide(I.plusSide()), I.minusSide());
        EXPECT_EQ(I.domain(), forms::InterfaceDomain(kGamma));
    }
}

TEST(InterfaceOrientationFitted, NormalFromPointsOutOfTheNamedSubdomain)
{
    for (const bool from_the_right : {false, true}) {
        // int_gamma (e_x . n_left) v_left = +1 and int_gamma (e_x . n_right) v_right = -1.
        const auto sums = fittedResidualSums(from_the_right, [](const InterfaceBetween& I, const FormExpr&,
                                                                const FormExpr& v) {
            return (forms::inner(ex(), I.normalFrom(kLeft)) * I.trace(v, kLeft)).dI(I.domain()) +
                   (forms::inner(ex(), I.normalFrom(kRight)) * I.trace(v, kRight)).dI(I.domain());
        });
        EXPECT_NEAR(sums.left, 1.0, 1e-12) << (from_the_right ? "subdomain 2 minus" : "subdomain 1 minus");
        EXPECT_NEAR(sums.right, -1.0, 1e-12) << (from_the_right ? "subdomain 2 minus" : "subdomain 1 minus");
    }
}

TEST(InterfaceOrientationFitted, TheUpwindFluxTakesTheUpstreamValueAndIsSymmetricInTheSides)
{
    for (const bool from_the_right : {false, true}) {
        for (const Real b : {1.0, -1.0}) {
            const auto beta = velocity(b);
            const auto left_to_right = [&](const InterfaceBetween& I, const FormExpr& c, const FormExpr& v) {
                return upwindFlux(I, kLeft, kRight, beta, c, c, v, v);
            };
            const auto right_to_left = [&](const InterfaceBetween& I, const FormExpr& c, const FormExpr& v) {
                return upwindFlux(I, kRight, kLeft, beta, c, c, v, v);
            };
            // The outflow of subdomain 1: b c_up, with c_up = 2 (left) for b > 0 and 5 (right) for b < 0.
            const Real outflow = b > 0.0 ? b * 2.0 : b * 5.0;
            const auto sums = fittedResidualSums(from_the_right, left_to_right);
            EXPECT_NEAR(sums.left, outflow, 1e-12);
            EXPECT_NEAR(sums.right, -outflow, 1e-12);
            const auto R_ab = fittedResidual(from_the_right, left_to_right);
            const auto R_ba = fittedResidual(from_the_right, right_to_left);
            ASSERT_EQ(R_ab.size(), R_ba.size());
            for (std::size_t i = 0; i < R_ab.size(); ++i) {
                EXPECT_EQ(R_ab[i], R_ba[i]) << "entry " << i << ", b = " << b;
            }
        }
    }
}

TEST(InterfaceOrientationFitted, AWeightedAverageNamesItsWeightsBySide)
{
    for (const bool from_the_right : {false, true}) {
        const auto as_written = fittedResidualSums(from_the_right, [](const InterfaceBetween& I, const FormExpr& c,
                                                                      const FormExpr& v) {
            return (I.average(c, {kLeft, 1.0}, {kRight, 100.0}) * I.trace(v, kLeft)).dI(I.domain());
        });
        const auto reordered = fittedResidualSums(from_the_right, [](const InterfaceBetween& I, const FormExpr& c,
                                                                     const FormExpr& v) {
            return (I.average(c, {kRight, 100.0}, {kLeft, 1.0}) * I.trace(v, kLeft)).dI(I.domain());
        });
        const auto other_weights = fittedResidualSums(from_the_right, [](const InterfaceBetween& I, const FormExpr& c,
                                                                         const FormExpr& v) {
            return (I.average(c, {kLeft, 100.0}, {kRight, 1.0}) * I.trace(v, kLeft)).dI(I.domain());
        });
        EXPECT_NEAR(as_written.left, 1.0 * 2.0 + 100.0 * 5.0, 1e-10);
        EXPECT_EQ(as_written.left, reordered.left);
        EXPECT_NEAR(other_weights.left, 100.0 * 2.0 + 1.0 * 5.0, 1e-10);
    }
}

TEST(InterfaceOrientationFitted, TheTreesAreTheSideRestrictions)
{
    const InterfaceBetween I(kGamma, kRightDomain, kLeftDomain);
    const auto c = FormExpr::parameter("c");
    EXPECT_EQ(systems::formExprTreeHash(I.trace(c, kRight)), systems::formExprTreeHash(c.minus()));
    EXPECT_EQ(systems::formExprTreeHash(I.trace(c, kLeft)), systems::formExprTreeHash(c.plus()));
    EXPECT_EQ(systems::formExprTreeHash(I.normalFrom(kRight)),
              systems::formExprTreeHash(FormExpr::normal().minus()));
    EXPECT_EQ(systems::formExprTreeHash(I.normalFrom(kLeft)),
              systems::formExprTreeHash(-FormExpr::normal().minus()));
    const auto w = FormExpr::parameter("w");
    EXPECT_EQ(systems::formExprTreeHash(I.average(c, {kLeft, w}, {kRight, 0.5})),
              systems::formExprTreeHash(w * c.plus() + FormExpr::constant(0.5) * c.minus()));

    EXPECT_THROW((void)I.trace(c, VolumeDomain::cells(7)), std::invalid_argument);
    EXPECT_THROW((void)I.normalFrom(VolumeDomain::full()), std::invalid_argument);
    EXPECT_THROW((void)I.average(c, {kLeft, 1.0}, {kLeft, 1.0}), std::invalid_argument);
    EXPECT_THROW((void)InterfaceBetween(kGamma, kLeftDomain, kLeftDomain), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Generated interface
// ---------------------------------------------------------------------------

namespace {

struct GeneratedCase {
    std::unique_ptr<FESystem> system;
    FieldId c_neg{INVALID_FIELD_ID};
    FieldId c_pos{INVALID_FIELD_ID};
};

using GeneratedTerms = std::function<FormExpr(const InterfaceBetween&, const forms::SidePair& c,
                                              const forms::SidePair& v)>;

/// c_neg on phi < 0 and c_pos on phi > 0 (P1 on the background mesh), with
/// (0 c_neg v_neg + 0 c_pos v_pos).dx() + terms(I, {c_neg, c_pos}, {v_neg, v_pos}).
GeneratedCase generatedCase(const GeneratedTerms& terms)
{
    GeneratedCase out;
    auto mesh = test_trace_multiplier::buildQuadGrid(4, 4, 1.0, 1.0, 1, test_trace_multiplier::kWholeBoundary);
    out.system = std::make_unique<FESystem>(mesh);
    auto& sys = *out.system;
    sys.setCutIntegrationContext(test_trace_multiplier::generatedInterfaceContext(
        *mesh, kGenerated, [](Real x, Real) { return x - 0.43; }, 1u, /*quadrature_order=*/2));
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);
    out.c_neg = sys.addField(FieldSpec{.name = "c_neg", .space = space});
    out.c_pos = sys.addField(FieldSpec{.name = "c_pos", .space = space});
    sys.addOperator("op");
    const forms::SidePair c{forms::StateField(out.c_neg, *space, "c_neg"),
                            forms::StateField(out.c_pos, *space, "c_pos")};
    const forms::SidePair v{forms::TestField(out.c_neg, *space, "v_neg"),
                            forms::TestField(out.c_pos, *space, "v_pos")};
    const auto I = InterfaceBetween::generated(kGenerated);
    const auto residual = (FormExpr::constant(0.0) * c.neg * v.neg).dx() +
                          (FormExpr::constant(0.0) * c.pos * v.pos).dx() + terms(I, c, v);
    (void)systems::installFormulation(sys, "op", {out.c_neg, out.c_pos}, residual, interpreted());
    sys.setup();
    return out;
}

/// The residual at c_neg = 2, c_pos = 5, summed over the test functions of
/// each field.
SideSums generatedResidualSums(const GeneratedTerms& terms, std::vector<Real>* residual_out = nullptr)
{
    const auto problem = generatedCase(terms);
    auto& sys = *problem.system;
    std::vector<Real> state(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 0.0);
    const auto fill = [&](FieldId field, Real value) {
        const auto offset = sys.fieldDofOffset(field);
        for (GlobalIndex i = 0; i < sys.fieldDofHandler(field).getNumDofs(); ++i) {
            state[static_cast<std::size_t>(offset + i)] = value;
        }
    };
    fill(problem.c_neg, 2.0);
    fill(problem.c_pos, 5.0);
    const auto residual = assembleResidual(sys, state);
    const auto sum = [&](FieldId field) {
        const auto offset = sys.fieldDofOffset(field);
        Real total = 0.0;
        for (GlobalIndex i = 0; i < sys.fieldDofHandler(field).getNumDofs(); ++i) {
            total += residual[static_cast<std::size_t>(offset + i)];
        }
        return total;
    };
    SideSums sums;
    sums.left = sum(problem.c_neg);
    sums.right = sum(problem.c_pos);
    if (residual_out != nullptr) {
        *residual_out = residual;
    }
    return sums;
}

} // namespace

TEST(InterfaceOrientationGenerated, TheSidesAreTheCutVolumes)
{
    const auto I = InterfaceBetween::generated(kGenerated);
    EXPECT_TRUE(I.isGenerated());
    EXPECT_EQ(I.minusSide(), VolumeDomain::cut(kGenerated, forms::CutVolumeSide::Negative));
    EXPECT_EQ(I.plusSide(), VolumeDomain::cut(kGenerated, forms::CutVolumeSide::Positive));
    const auto c = FormExpr::parameter("c");
    EXPECT_EQ(systems::formExprTreeHash(I.trace(c, I.minusSide())), systems::formExprTreeHash(c.minus()));
    EXPECT_EQ(systems::formExprTreeHash(I.trace(c, I.plusSide())), systems::formExprTreeHash(c.plus()));
    EXPECT_EQ(systems::formExprTreeHash(I.normalFrom(I.minusSide())),
              systems::formExprTreeHash(FormExpr::normal().minus()));
    EXPECT_EQ(systems::formExprTreeHash(I.normalFrom(I.plusSide())),
              systems::formExprTreeHash(-FormExpr::normal().minus()));
    EXPECT_THROW((void)InterfaceBetween::generated(-1), std::invalid_argument);
    EXPECT_THROW((void)I.trace(c, VolumeDomain::cut(kGenerated + 1, forms::CutVolumeSide::Negative)),
                 std::invalid_argument);
}

TEST(InterfaceOrientationGenerated, NormalFromPointsOutOfTheNamedSide)
{
    // int_gamma (e_x . n_neg) v_neg = +1 and int_gamma (e_x . n_pos) v_pos = -1.
    const auto sums = generatedResidualSums([](const InterfaceBetween& I, const forms::SidePair&,
                                               const forms::SidePair& v) {
        const auto N = I.minusSide();
        const auto P = I.plusSide();
        return (forms::inner(ex(), I.normalFrom(N)) * I.trace(v.neg, N)).dI(I.domain()) +
               (forms::inner(ex(), I.normalFrom(P)) * I.trace(v.pos, P)).dI(I.domain());
    });
    EXPECT_NEAR(sums.left, 1.0, 1e-12);
    EXPECT_NEAR(sums.right, -1.0, 1e-12);

    // The same with a jump term in the same kernel, which makes the
    // evaluation two-sided: a term means the same either way (D70).
    const auto with_jump = generatedResidualSums([](const InterfaceBetween& I, const forms::SidePair& c,
                                                    const forms::SidePair& v) {
        const auto N = I.minusSide();
        return (forms::inner(ex(), I.normalFrom(N)) * I.trace(v.neg, N)).dI(I.domain()) +
               (FormExpr::constant(0.0) * forms::jump(c.neg) * I.trace(v.neg, N)).dI(I.domain());
    });
    EXPECT_NEAR(with_jump.left, 1.0, 1e-12);
}

TEST(InterfaceOrientationGenerated, TheUpwindFluxTakesTheUpstreamValueAndIsSymmetricInTheSides)
{
    for (const Real b : {1.0, -1.0}) {
        const auto beta = velocity(b);
        std::vector<Real> R_ab;
        std::vector<Real> R_ba;
        const auto sums = generatedResidualSums(
            [&](const InterfaceBetween& I, const forms::SidePair& c, const forms::SidePair& v) {
                return upwindFlux(I, I.minusSide(), I.plusSide(), beta, c.neg, c.pos, v.neg, v.pos);
            },
            &R_ab);
        (void)generatedResidualSums(
            [&](const InterfaceBetween& I, const forms::SidePair& c, const forms::SidePair& v) {
                return upwindFlux(I, I.plusSide(), I.minusSide(), beta, c.pos, c.neg, v.pos, v.neg);
            },
            &R_ba);
        // The outflow of the negative side: b c_up, c_up = c_neg = 2 for b > 0 and c_pos = 5 for b < 0.
        const Real outflow = b > 0.0 ? b * 2.0 : b * 5.0;
        EXPECT_NEAR(sums.left, outflow, 1e-12) << "b = " << b;
        EXPECT_NEAR(sums.right, -outflow, 1e-12) << "b = " << b;
        ASSERT_EQ(R_ab.size(), R_ba.size());
        for (std::size_t i = 0; i < R_ab.size(); ++i) {
            EXPECT_EQ(R_ab[i], R_ba[i]) << "entry " << i << ", b = " << b;
        }
    }
}

TEST(InterfaceOrientationGenerated, AWeightedAverageNamesItsWeightsBySide)
{
    // A single field per side has one trace there, so the average of the
    // pair is written with SidePairWeights, and the interface average of a
    // field c read from both sides with SideWeight.
    const auto pair = generatedResidualSums([](const InterfaceBetween& I, const forms::SidePair& c,
                                               const forms::SidePair& v) {
        return (forms::avg(c, {.neg = FormExpr::constant(1.0), .pos = FormExpr::constant(100.0)}) *
                I.trace(v.neg, I.minusSide()))
            .dI(I.domain());
    });
    EXPECT_NEAR(pair.left, 1.0 * 2.0 + 100.0 * 5.0, 1e-10);
    const auto by_side = generatedResidualSums([](const InterfaceBetween& I, const forms::SidePair& c,
                                                  const forms::SidePair& v) {
        return (I.average(c.neg + c.pos, {I.plusSide(), 100.0}, {I.minusSide(), 1.0}) *
                I.trace(v.neg, I.minusSide()))
            .dI(I.domain());
    });
    EXPECT_NEAR(by_side.left, 101.0 * 7.0, 1e-10);

    const forms::SidePair s{FormExpr::parameter("a"), FormExpr::parameter("b")};
    const auto wn = FormExpr::parameter("wn");
    const auto wp = FormExpr::parameter("wp");
    EXPECT_EQ(systems::formExprTreeHash(forms::avg(s, {.neg = wn, .pos = wp})),
              systems::formExprTreeHash(wn * s.neg + wp * s.pos));
    EXPECT_EQ(systems::formExprTreeHash(forms::avg(s)),
              systems::formExprTreeHash(FormExpr::constant(0.5) * s.neg + FormExpr::constant(0.5) * s.pos));
}

} // namespace svmp::FE::test_interface_orientation
