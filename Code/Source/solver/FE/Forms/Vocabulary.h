#ifndef SVMP_FE_FORMS_VOCABULARY_H
#define SVMP_FE_FORMS_VOCABULARY_H

/**
 * @file Vocabulary.h
 * @brief High-level vocabulary helpers built on top of forms::FormExpr
 *
 * This header provides small, composable convenience helpers (UFL-style) that
 * build on the core AST nodes in FormExpr. These helpers do not own assembly
 * or DOF logic; they only return FormExpr trees.
 */

#include "Forms/FormExpr.h"
#include "Spaces/MixedSpace.h"

#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {

// ---------------------------------------------------------------------------
// Constitutive call ergonomics
// ---------------------------------------------------------------------------

/**
 * @brief Small helper wrapper for constitutive(...) calls
 *
 * This makes multi-output constitutive calls feel like expression objects:
 * @code
 * auto call = constitutive(model, gamma);
 * auto mu   = call.out(0);
 * @endcode
 *
 * The wrapper is implicitly convertible to FormExpr, representing the call
 * expression (default output 0).
 */
struct ConstitutiveCall {
    FormExpr call{};

    [[nodiscard]] FormExpr out(std::size_t output_index) const
    {
        return FormExpr::constitutiveOutput(call, output_index);
    }

    [[nodiscard]] FormExpr output(std::size_t output_index) const { return out(output_index); }

    [[nodiscard]] const FormExpr& expr() const noexcept { return call; }

    [[nodiscard]] operator FormExpr() const { return call; }
};

// Basic algebra forwarding so `ConstitutiveCall` can be used like a `FormExpr`.
inline FormExpr operator-(const ConstitutiveCall& a) { return -a.call; }
inline FormExpr operator+(const ConstitutiveCall& a, const ConstitutiveCall& b) { return a.call + b.call; }
inline FormExpr operator+(const ConstitutiveCall& a, const FormExpr& b) { return a.call + b; }
inline FormExpr operator+(const FormExpr& a, const ConstitutiveCall& b) { return a + b.call; }
inline FormExpr operator-(const ConstitutiveCall& a, const ConstitutiveCall& b) { return a.call - b.call; }
inline FormExpr operator-(const ConstitutiveCall& a, const FormExpr& b) { return a.call - b; }
inline FormExpr operator-(const FormExpr& a, const ConstitutiveCall& b) { return a - b.call; }
inline FormExpr operator*(const ConstitutiveCall& a, const ConstitutiveCall& b) { return a.call * b.call; }
inline FormExpr operator*(const ConstitutiveCall& a, const FormExpr& b) { return a.call * b; }
inline FormExpr operator*(const FormExpr& a, const ConstitutiveCall& b) { return a * b.call; }
inline FormExpr operator/(const ConstitutiveCall& a, const ConstitutiveCall& b) { return a.call / b.call; }
inline FormExpr operator/(const ConstitutiveCall& a, const FormExpr& b) { return a.call / b; }
inline FormExpr operator/(const FormExpr& a, const ConstitutiveCall& b) { return a / b.call; }

inline ConstitutiveCall constitutive(std::shared_ptr<const ConstitutiveModel> model, const FormExpr& input)
{
    return ConstitutiveCall{FormExpr::constitutive(std::move(model), input)};
}

inline ConstitutiveCall constitutive(std::shared_ptr<const ConstitutiveModel> model, std::vector<FormExpr> inputs)
{
    return ConstitutiveCall{FormExpr::constitutive(std::move(model), std::move(inputs))};
}

inline ConstitutiveCall constitutive(std::shared_ptr<const ConstitutiveModel> model, std::initializer_list<FormExpr> inputs)
{
    return ConstitutiveCall{FormExpr::constitutive(std::move(model), std::vector<FormExpr>(inputs.begin(), inputs.end()))};
}

template <class... Rest>
inline ConstitutiveCall constitutive(std::shared_ptr<const ConstitutiveModel> model,
                                     const FormExpr& a0,
                                     const FormExpr& a1,
                                     const Rest&... rest)
{
    std::vector<FormExpr> inputs;
    inputs.reserve(2u + sizeof...(rest));
    inputs.push_back(a0);
    inputs.push_back(a1);
    (inputs.push_back(rest), ...);
    return ConstitutiveCall{FormExpr::constitutive(std::move(model), std::move(inputs))};
}

// ---------------------------------------------------------------------------
// Function-space bound arguments (UFL-style)
// ---------------------------------------------------------------------------

inline FormExpr TrialFunction(const spaces::FunctionSpace& V, std::string name = "u")
{
    return FormExpr::trialFunction(V, std::move(name));
}

inline FormExpr TestFunction(const spaces::FunctionSpace& V, std::string name = "v")
{
    return FormExpr::testFunction(V, std::move(name));
}

/**
 * @brief Construct trial functions for each component of a MixedSpace (UFL-style)
 *
 * This is a convenience utility for mixed/multi-field Systems workflows:
 * each returned FormExpr is a standard single-field TrialFunction bound to the
 * corresponding component space, suitable for per-block compilation.
 *
 * The core FormCompiler intentionally does not support multiple TrialFunction
 * symbols in a single FormExpr; use `BlockBilinearForm` / `BlockLinearForm`
 * and compile each block independently.
 */
inline std::vector<FormExpr> TrialFunctions(const spaces::MixedSpace& W,
                                            std::vector<std::string> names = {})
{
    const auto n = W.num_components();
    if (!names.empty() && names.size() != n) {
        throw std::invalid_argument("TrialFunctions(MixedSpace): names.size() must match num_components()");
    }

    std::vector<FormExpr> u;
    u.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& comp = W.component(i);
        FE_CHECK_NOT_NULL(comp.space.get(), "TrialFunctions(MixedSpace) component space");
        std::string nm;
        if (!names.empty()) {
            nm = names[i];
        } else if (!comp.name.empty()) {
            nm = comp.name;
        } else {
            nm = "u" + std::to_string(i);
        }
        u.emplace_back(FormExpr::trialFunction(*comp.space, std::move(nm)));
    }
    return u;
}

/**
 * @brief Construct test functions for each component of a MixedSpace (UFL-style)
 *
 * Defaults to `v0, v1, ...` unless explicit names are provided.
 */
inline std::vector<FormExpr> TestFunctions(const spaces::MixedSpace& W,
                                           std::vector<std::string> names = {})
{
    const auto n = W.num_components();
    if (!names.empty() && names.size() != n) {
        throw std::invalid_argument("TestFunctions(MixedSpace): names.size() must match num_components()");
    }

    std::vector<FormExpr> v;
    v.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& comp = W.component(i);
        FE_CHECK_NOT_NULL(comp.space.get(), "TestFunctions(MixedSpace) component space");
        std::string nm;
        if (!names.empty()) {
            nm = names[i];
        } else {
            nm = "v" + std::to_string(i);
        }
        v.emplace_back(FormExpr::testFunction(*comp.space, std::move(nm)));
    }
    return v;
}

// ---------------------------------------------------------------------------
// Geometry terminals (UFL-like shorthands)
// ---------------------------------------------------------------------------

inline FormExpr x() { return FormExpr::coordinate(); }
inline FormExpr X() { return FormExpr::referenceCoordinate(); }
inline FormExpr t() { return FormExpr::time(); }
inline FormExpr deltat() { return FormExpr::timeStep(); }
inline FormExpr deltat_eff() { return FormExpr::effectiveTimeStep(); }

inline FormExpr J() { return FormExpr::jacobian(); }
inline FormExpr Jinv() { return FormExpr::jacobianInverse(); }
inline FormExpr detJ() { return FormExpr::jacobianDeterminant(); }

inline FormExpr h() { return FormExpr::cellDiameter(); }
inline FormExpr vol() { return FormExpr::cellVolume(); }
inline FormExpr area() { return FormExpr::facetArea(); }
inline FormExpr domainId() { return FormExpr::cellDomainId(); }

/**
 * @brief Directional cell size normal to the current facet
 *
 * Defined as h_n = 2 |K| / |F| (cell volume divided by facet area).
 * This is the facet-normal "height" of the element and is a more robust
 * choice than isotropic `h()` for penalty scaling on anisotropic meshes.
 */
inline FormExpr hNormal() { return (2.0 * vol()) / area(); }

// ---------------------------------------------------------------------------
// Differential operators (UFL-like shorthands)
// ---------------------------------------------------------------------------

inline FormExpr hessian(const FormExpr& a) { return a.hessian(); }

inline FormExpr laplacian(const FormExpr& a) { return trace(a.hessian()); }

// ---------------------------------------------------------------------------
// Geometric differential operators (surface/level-set helpers)
// ---------------------------------------------------------------------------

inline FormExpr safeNorm(const FormExpr& v, Real eps = Real(1e-12))
{
    return sqrt(inner(v, v) + FormExpr::constant(eps * eps));
}

inline FormExpr safeNormalize(const FormExpr& v, Real eps = Real(1e-12))
{
    return v / safeNorm(v, eps);
}

inline FormExpr unitNormalFromLevelSet(const FormExpr& phi, Real eps = Real(1e-12))
{
    return safeNormalize(grad(phi), eps);
}

inline FormExpr meanCurvatureFromLevelSet(const FormExpr& phi, Real eps = Real(1e-12))
{
    return div(unitNormalFromLevelSet(phi, eps));
}

inline FormExpr surfaceGradient(const FormExpr& f, const FormExpr& n)
{
    const auto gf = grad(f);
    return gf - inner(gf, n) * n;
}

inline FormExpr surfaceDivergence(const FormExpr& u, const FormExpr& n)
{
    return div(u) - inner(grad(u) * n, n);
}

inline FormExpr surfaceLaplacian(const FormExpr& f, const FormExpr& n)
{
    return surfaceDivergence(surfaceGradient(f, n), n);
}

// ---------------------------------------------------------------------------
// Vector/tensor constructors (UFL-like)
// ---------------------------------------------------------------------------

inline FormExpr as_vector(std::initializer_list<FormExpr> components)
{
    return FormExpr::asVector(std::vector<FormExpr>(components.begin(), components.end()));
}

inline FormExpr as_tensor(std::initializer_list<std::initializer_list<FormExpr>> rows)
{
    std::vector<std::vector<FormExpr>> r;
    r.reserve(rows.size());
    for (const auto& row : rows) {
        r.emplace_back(row.begin(), row.end());
    }
    return FormExpr::asTensor(std::move(r));
}

// ---------------------------------------------------------------------------
// Tensor algebra helpers
// ---------------------------------------------------------------------------

/**
 * @brief Single-index contraction (UFL-like tensor product contraction)
 *
 * Current semantics (evaluated at runtime based on operand shapes):
 * - Scalar * Scalar -> Scalar
 * - Scalar * Vector / Vector * Scalar -> Vector scaling
 * - Scalar * Matrix / Matrix * Scalar -> Matrix scaling
 * - Matrix * Vector -> Vector (matrix-vector product)
 * - Vector * Matrix -> Vector (row-vector * matrix)
 * - Matrix * Matrix -> Matrix (matrix multiplication)
 */
inline FormExpr contraction(const FormExpr& a, const FormExpr& b) { return a * b; }

/**
 * @brief Double contraction (Frobenius inner product for rank-2 tensors)
 */
inline FormExpr doubleContraction(const FormExpr& a, const FormExpr& b) { return a.doubleContraction(b); }

/**
 * @brief Symmetric tensor (rank-2) wrapper (UFL-like)
 *
 * This is a semantic alias for sym(A). Evaluation tags the result as a
 * symmetric-matrix kind for downstream checks/optimizations.
 */
inline FormExpr SymmetricTensor(const FormExpr& A) { return sym(A); }

/**
 * @brief Skew tensor (rank-2) wrapper (UFL-like)
 *
 * This is a semantic alias for skew(A). Evaluation tags the result as a
 * skew-matrix kind for downstream checks/optimizations.
 */
inline FormExpr SkewTensor(const FormExpr& A) { return skew(A); }

// ---------------------------------------------------------------------------
// Common scalar helper functions
// ---------------------------------------------------------------------------

inline FormExpr heaviside(const FormExpr& a)
{
    return conditional(gt(a, FormExpr::constant(0.0)),
                       FormExpr::constant(1.0),
                       FormExpr::constant(0.0));
}

inline FormExpr indicator(const FormExpr& predicate)
{
    return conditional(predicate,
                       FormExpr::constant(1.0),
                       FormExpr::constant(0.0));
}

inline FormExpr clamp(const FormExpr& a, const FormExpr& lo, const FormExpr& hi)
{
    return min(max(a, lo), hi);
}

inline FormExpr regionIndicator(int domain_id)
{
    return eq(domainId(), FormExpr::constant(static_cast<Real>(domain_id)));
}

inline FormExpr weightedAverage(const FormExpr& a, const FormExpr& w_plus, const FormExpr& w_minus)
{
    return w_plus * a.plus() + w_minus * a.minus();
}

inline FormExpr harmonicAverage(const FormExpr& k)
{
    // 2 k⁺ k⁻ / (k⁺ + k⁻)
    const auto kp = k.plus();
    const auto km = k.minus();
    return (2.0 * kp * km) / (kp + km);
}

// ---------------------------------------------------------------------------
// DG / interior-facet helpers
// ---------------------------------------------------------------------------

inline FormExpr upwindValue(const FormExpr& u, const FormExpr& beta)
{
    // u_up = {u} + 0.5 * sign( {beta} · n(-) ) * [[u]]
    const auto n_minus = FormExpr::normal().minus();
    return avg(u) + 0.5 * sign(inner(avg(beta), n_minus)) * jump(u);
}

inline FormExpr downwindValue(const FormExpr& u, const FormExpr& beta)
{
    // u_down = {u} - 0.5 * sign( {beta} · n(-) ) * [[u]]
    const auto n_minus = FormExpr::normal().minus();
    return avg(u) - 0.5 * sign(inner(avg(beta), n_minus)) * jump(u);
}

inline FormExpr interiorPenaltyCoefficient(Real eta, Real p = 1.0)
{
    // Typical SIPG scaling: eta * p^2 * avg(1 / h_n),
    // where h_n = 2|K|/|F| is the facet-normal cell size.
    return FormExpr::constant(eta * p * p) * avg(FormExpr::constant(1.0) / hNormal());
}

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_VOCABULARY_H
