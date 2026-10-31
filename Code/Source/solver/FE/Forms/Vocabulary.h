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
#include "Forms/IntegrationDomain.h"
#include "Forms/SymbolicDifferentiation.h"
#include "Spaces/MixedSpace.h"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
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

// Generic generated-interface/boundary intersections are imported into the
// cut-integration context as generated rules with their own marker and an
// explicit d-2 geometric dimension. This helper shares dI dispatch, while the
// rule metadata selects the codimension-two Jacobian in the assembler.
[[nodiscard]] inline FormExpr dInterfaceBoundary(const FormExpr& integrand,
                                                 int intersection_marker)
{
    return integrand.dI(intersection_marker);
}

/// Typed form of dInterfaceBoundary(): integrand.dL(contact_line), which is
/// integrand.dI(contact_line.marker()).
[[nodiscard]] inline FormExpr dInterfaceBoundary(const FormExpr& integrand,
                                                 const ContactLineDomain& contact_line)
{
    return integrand.dL(contact_line);
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
// Residual authoring helpers (field-bound)
//
// These are the recommended helpers for residual physics authoring via
// installFormulation(). Each symbol carries a FieldId so that
// installFormulation() can unambiguously map test rows to fields, even when
// multiple fields share the same FE space.
//
// Usage:
//   auto u = StateField(u_id, V, "u");
//   auto v = TestField(u_id, V, "v");
//   auto residual = (inner(grad(u), grad(v))).dx();
//   installFormulation(system, "equations", {u_id}, residual);
// ---------------------------------------------------------------------------

/// Create a field-bound state variable for residual authoring.
/// Equivalent to FormExpr::stateField(field, space, name).
inline FormExpr StateField(FieldId field, const spaces::FunctionSpace& V, std::string name)
{
    return FormExpr::stateField(field, V, std::move(name));
}

/// Create a field-bound test function for residual authoring.
/// The FieldId binding allows installFormulation() to map test rows to fields
/// unambiguously, which is required when multiple fields share the same space.
inline FormExpr TestField(FieldId field, const spaces::FunctionSpace& V, std::string name)
{
    return FormExpr::testFunction(field, V, std::move(name));
}

/// Create state variables for each field in a multi-field residual system.
/// Returns one StateField per field, in the same order as the field_ids.
inline std::vector<FormExpr> StateFields(
    std::span<const FieldId> field_ids,
    std::span<const std::shared_ptr<const spaces::FunctionSpace>> spaces,
    std::vector<std::string> names)
{
    if (field_ids.size() != spaces.size() || field_ids.size() != names.size()) {
        throw std::invalid_argument("StateFields: field_ids, spaces, and names must have the same size");
    }
    std::vector<FormExpr> result;
    result.reserve(field_ids.size());
    for (std::size_t i = 0; i < field_ids.size(); ++i) {
        result.push_back(FormExpr::stateField(field_ids[i], *spaces[i], std::move(names[i])));
    }
    return result;
}

/// Create field-bound test functions for each field in a multi-field residual.
/// Returns one TestField per field, in the same order as the field_ids.
inline std::vector<FormExpr> TestFields(
    std::span<const FieldId> field_ids,
    std::span<const std::shared_ptr<const spaces::FunctionSpace>> spaces,
    std::vector<std::string> names)
{
    if (field_ids.size() != spaces.size() || field_ids.size() != names.size()) {
        throw std::invalid_argument("TestFields: field_ids, spaces, and names must have the same size");
    }
    std::vector<FormExpr> result;
    result.reserve(field_ids.size());
    for (std::size_t i = 0; i < field_ids.size(); ++i) {
        result.push_back(FormExpr::testFunction(field_ids[i], *spaces[i], std::move(names[i])));
    }
    return result;
}

// ---------------------------------------------------------------------------
// Operator authoring helpers (space-bound, no FieldId)
//
// These are for bilinear/linear operator forms (installMixedBilinear,
// installMixedLinear) where field binding is not needed. Keep using
// TrialFunction/TestFunction for operator authoring.
// ---------------------------------------------------------------------------

/// Create a single TrialFunction (operator forms)
inline FormExpr TrialFunction(const spaces::FunctionSpace& V, std::string name = "u")
{
    return FormExpr::trialFunction(V, std::move(name));
}

/// Create a single TestFunction (operator forms, no field binding)
inline FormExpr TestFunction(const spaces::FunctionSpace& V, std::string name = "v")
{
    return FormExpr::testFunction(V, std::move(name));
}

// Note: TestFunction(FieldId, ...) was removed to avoid two ways to create
// the same field-bound test symbol. Use TestField(field, space, name) instead.

/**
 * @brief Create trial functions for each component of a MixedSpace
 *
 * Returns one unbound TrialFunction per MixedSpace component. These are for
 * operator-level workflows (installMixedBilinear, installMixedLinear) where
 * field binding is not needed.
 *
 * For residual physics authored via installFormulation(), prefer the
 * field-bound StateField/TestField helpers above — they carry FieldId
 * bindings that are required for correct multi-field block decomposition,
 * especially when fields share the same FE space.
 *
 * Example (operator workflow):
 * @code
 *   auto trials = TrialFunctions(W, {"u", "p"});
 *   auto tests  = TestFunctions(W, {"v", "q"});
 *   auto a = (inner(grad(trials[0]), grad(tests[0]))
 *           - trials[1] * div(tests[0]) + div(trials[0]) * tests[1]).dx();
 *   installMixedBilinear(system, "op", test_fields, trial_fields, a);
 * @endcode
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
inline FormExpr currentCoordinate() { return FormExpr::currentCoordinate(); }
inline FormExpr previousCoordinate() { return FormExpr::previousCoordinate(); }
inline FormExpr referenceCoordinatePhysical() { return FormExpr::referenceCoordinatePhysical(); }
inline FormExpr meshDisplacement() { return FormExpr::meshDisplacement(); }
inline FormExpr meshVelocity() { return FormExpr::meshVelocity(); }
inline FormExpr meshAcceleration() { return FormExpr::meshAcceleration(); }
inline FormExpr previousMeshVelocity() { return FormExpr::previousMeshVelocity(); }
inline FormExpr predictedMeshVelocity() { return FormExpr::predictedMeshVelocity(); }
inline FormExpr t() { return FormExpr::time(); }
inline FormExpr deltat() { return FormExpr::timeStep(); }
inline FormExpr deltat_eff() { return FormExpr::effectiveTimeStep(); }

// NOTE: Geometry Jacobians are stored internally as 3x3 "frame" matrices so they are invertible
// even for dim<3 mappings. At the FormExpr level, `J()` / `Jinv()` are shaped as (dim×dim),
// where dim is the current AssemblyContext spatial dimension, so contractions like `trace(Jinv())`
// and `trace(transpose(Jinv())*Jinv())` automatically use the physical dimension.
inline FormExpr J() { return FormExpr::jacobian(); }
inline FormExpr Jinv() { return FormExpr::jacobianInverse(); }
inline FormExpr detJ() { return FormExpr::jacobianDeterminant(); }
inline FormExpr currentJacobian() { return FormExpr::currentJacobian(); }
inline FormExpr referenceJacobian() { return FormExpr::referenceJacobian(); }
inline FormExpr currentJacobianDeterminant() { return FormExpr::currentJacobianDeterminant(); }
inline FormExpr referenceJacobianDeterminant() { return FormExpr::referenceJacobianDeterminant(); }
inline FormExpr currentNormal() { return FormExpr::currentNormal(); }
inline FormExpr currentMeanCurvature() { return FormExpr::currentMeanCurvature(); }
inline FormExpr referenceNormal() { return FormExpr::referenceNormal(); }
inline FormExpr currentMeasure() { return FormExpr::currentMeasure(); }
inline FormExpr referenceMeasure() { return FormExpr::referenceMeasure(); }
inline FormExpr surfaceJacobian() { return FormExpr::surfaceJacobian(); }
inline FormExpr currentSurfaceVector() { return currentMeasure() * currentNormal(); }
inline FormExpr referenceSurfaceVector() { return referenceMeasure() * referenceNormal(); }

inline FormExpr nanson(const FormExpr& reference_surface_vector)
{
    const auto F = currentJacobian() * inv(referenceJacobian());
    return cofactor(F) * reference_surface_vector;
}

inline FormExpr nanson()
{
    return nanson(referenceSurfaceVector());
}

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
// Trace/projection helpers
// ---------------------------------------------------------------------------

inline FormExpr normalTrace(const FormExpr& u, const FormExpr& n)
{
    return inner(u, n);
}

inline FormExpr normalTrace(const FormExpr& u)
{
    return normalTrace(u, FormExpr::normal());
}

inline FormExpr normalProjection(const FormExpr& u, const FormExpr& n)
{
    return normalTrace(u, n) * n;
}

inline FormExpr normalProjection(const FormExpr& u)
{
    return normalProjection(u, FormExpr::normal());
}

inline FormExpr projectNormal(const FormExpr& u, const FormExpr& n)
{
    return normalProjection(u, n);
}

inline FormExpr projectNormal(const FormExpr& u)
{
    return normalProjection(u);
}

inline FormExpr tangentialProjection(const FormExpr& u, const FormExpr& n)
{
    return u - normalProjection(u, n);
}

inline FormExpr tangentialProjection(const FormExpr& u)
{
    return tangentialProjection(u, FormExpr::normal());
}

inline FormExpr projectTangent(const FormExpr& u, const FormExpr& n)
{
    return tangentialProjection(u, n);
}

inline FormExpr projectTangent(const FormExpr& u)
{
    return tangentialProjection(u);
}

inline FormExpr tangentialTrace(const FormExpr& u, const FormExpr& n)
{
    return tangentialProjection(u, n);
}

inline FormExpr tangentialTrace(const FormExpr& u)
{
    return tangentialProjection(u);
}

// ---------------------------------------------------------------------------
// Differential operators (UFL-like shorthands)
// ---------------------------------------------------------------------------

/// Symmetric gradient of a vector field; exactly the tree sym(grad(u)).
inline FormExpr symgrad(const FormExpr& u) { return sym(grad(u)); }

/// Vector advection (a . grad) u; exactly grad(u) * a with the component-first
/// gradient convention. For scalar transport use inner(grad(u), a).
inline FormExpr advect(const FormExpr& a, const FormExpr& u) { return grad(u) * a; }

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

/**
 * @brief Tangential projector I - n (x) n onto the plane normal to n
 *
 * The tree is FormExpr::identity(dim) - outer(n, n), the form used by the
 * variational surface-stress term gamma (I - n (x) n) : grad(v).  It is not
 * the same tree as surfaceDivergence(), which is algebraically equal on unit
 * n but compiles to a different kernel.
 */
inline FormExpr tangentialProjector(const FormExpr& n, int dim)
{
    return FormExpr::identity(dim) - outer(n, n);
}

/**
 * @brief Unit conormal of a contact line
 *
 * The direction of n in the plane of a wall with unit normal n_w:
 * safeNormalize(n - (n . n_w) n_w).  With n the outward normal of the liquid
 * and n_w pointing from the fluid into the wall, it points out of the wetted
 * footprint, along the wall.
 */
inline FormExpr conormal(const FormExpr& n, const FormExpr& wall_normal, Real eps = Real(1e-12))
{
    return safeNormalize(n - dot(n, wall_normal) * wall_normal, eps);
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

inline FormExpr meanCurvatureFromNormal(const FormExpr& n)
{
    return surfaceDivergence(n, n);
}

inline FormExpr curvatureVectorFromNormal(const FormExpr& n)
{
    return meanCurvatureFromNormal(n) * n;
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

inline FormExpr zeroVector(int dim)
{
    if (dim <= 0) {
        throw std::invalid_argument("zeroVector: dimension must be positive");
    }
    std::vector<FormExpr> components;
    components.reserve(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        components.push_back(FormExpr::constant(0.0));
    }
    return FormExpr::asVector(std::move(components));
}

/// The first @p dim entries of @p values as a constant vector.
inline FormExpr constantVector(std::span<const Real> values, int dim)
{
    if (dim <= 0 || static_cast<std::size_t>(dim) > values.size()) {
        throw std::invalid_argument("constantVector: dimension must be positive and at most the number of values");
    }
    std::vector<FormExpr> components;
    for (int d = 0; d < dim; ++d) {
        components.push_back(FormExpr::constant(values[static_cast<std::size_t>(d)]));
    }
    return FormExpr::asVector(std::move(components));
}

/// a + b, or b when a is empty: the left fold of a form built term by term.
inline FormExpr sumOf(const FormExpr& a, const FormExpr& b)
{
    return a.isValid() ? a + b : b;
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

// ---------------------------------------------------------------------------
// One quantity as two fields, one on each side of an interface
// ---------------------------------------------------------------------------

/**
 * @brief A quantity held by two fields, one on each side of an interface
 *
 * neg is the field of the negative side (phi < c on a level-set interface,
 * the minus side of jump()) and pos the field of the positive side, e.g.
 * SidePair{u_neg, u_pos} of two StateFields or SidePair{v_neg, v_pos} of
 * their TestFields.  jump(), avg() and map() build ordinary trees, so an
 * interface coupling is one integral,
 *   (-inner(avg(map(sigma, U, P) * n, w_neg, w_pos), jump(V))
 *    + gamma * inner(jump(U), jump(V))).dI(m),
 * whose integrand holds the test functions of both fields; installFormulation
 * splits it into one integral per test field.
 */
struct SidePair {
    FormExpr neg;
    FormExpr pos;
};

/// neg - pos, the jump across the interface with the convention of jump().
inline FormExpr jump(const SidePair& a) { return a.neg - a.pos; }

/// w_neg neg + w_pos pos, the weighted average (Hansbo or viscosity weights).
inline FormExpr avg(const SidePair& a, const FormExpr& w_neg, const FormExpr& w_pos)
{
    return w_neg * a.neg + w_pos * a.pos;
}

inline FormExpr avg(const SidePair& a, Real w_neg, Real w_pos)
{
    return avg(a, FormExpr::constant(w_neg), FormExpr::constant(w_pos));
}

/// (neg + pos) / 2, written 0.5 neg + 0.5 pos.
inline FormExpr avg(const SidePair& a) { return avg(a, 0.5, 0.5); }

/**
 * @brief One law on both sides: {f(a.neg), f(a.pos)}
 *
 * f may take the side as its last argument (CutVolumeSide::Negative for neg,
 * Positive for pos) to select per-side parameters.
 */
template <class F>
inline SidePair map(F&& f, const SidePair& a)
{
    if constexpr (std::is_invocable_v<F&, const FormExpr&, CutVolumeSide>) {
        return {f(a.neg, CutVolumeSide::Negative), f(a.pos, CutVolumeSide::Positive)};
    } else {
        return {f(a.neg), f(a.pos)};
    }
}

/// {f(a.neg, b.neg), f(a.pos, b.pos)}, e.g. the stress map(sigma, U, P).
template <class F>
inline SidePair map(F&& f, const SidePair& a, const SidePair& b)
{
    if constexpr (std::is_invocable_v<F&, const FormExpr&, const FormExpr&, CutVolumeSide>) {
        return {f(a.neg, b.neg, CutVolumeSide::Negative), f(a.pos, b.pos, CutVolumeSide::Positive)};
    } else {
        return {f(a.neg, b.neg), f(a.pos, b.pos)};
    }
}

inline SidePair operator*(const SidePair& a, const FormExpr& b) { return {a.neg * b, a.pos * b}; }
inline SidePair operator*(const FormExpr& a, const SidePair& b) { return {a * b.neg, a * b.pos}; }
inline SidePair operator+(const SidePair& a, const SidePair& b) { return {a.neg + b.neg, a.pos + b.pos}; }
inline SidePair operator-(const SidePair& a, const SidePair& b) { return {a.neg - b.neg, a.pos - b.pos}; }
inline SidePair operator-(const SidePair& a) { return {-a.neg, -a.pos}; }

// ---------------------------------------------------------------------------
// Variations and linearization
// ---------------------------------------------------------------------------

/**
 * @brief The Gateaux derivative of @p form in the field of @p u along @p direction
 *
 * d/de form(u + e w)|_{e=0} with w = @p direction.  With @p form a measured
 * 0-form, such as an energy Pi or a Lagrangian, and w the test function of u,
 * it is the residual dPi/du[v]; with w = u it is the rate D'(u)[u] (2 D for a
 * quadratic dissipation potential D).  @p u is the field's StateField: every
 * StateField of its field varies, while its frozen values (DiscreteField,
 * frozen()), test functions, other fields and the geometry are held fixed.
 * The result is simplified (zero terms dropped).  Throws when @p u is not a StateField, when @p form holds a
 * Constitutive call (write the law as a FormExpr), and when the derivative is
 * zero (the form does not depend on the field).
 */
[[nodiscard]] inline FormExpr derivative(const FormExpr& form, const FormExpr& u, const FormExpr& direction)
{
    if (!form.isValid()) {
        throw std::invalid_argument("derivative: empty form");
    }
    if (!u.isValid() || u.node()->type() != FormExprType::StateField || !u.node()->fieldId()) {
        throw std::invalid_argument("derivative: the variable is a field's StateField, but it is '" +
                                    u.toString() + "'");
    }
    if (!direction.isValid()) {
        throw std::invalid_argument("derivative: empty direction");
    }
    auto result = directionalDerivativeWrtField(form, *u.node()->fieldId(), direction,
                                                /*discrete_fields_vary=*/false);
    const auto value = result.isValid() ? result.node()->constantValue() : std::optional<Real>{Real(0.0)};
    if (value.has_value() && *value == Real(0.0)) {
        throw std::invalid_argument("derivative: '" + form.toString() + "' does not depend on '" +
                                    u.toString() + "'");
    }
    return result;
}

/// The sum of the derivatives in several fields, sum_i dform/du_i[w_i] for
/// the pairs (u_i, w_i): the stationarity residual of a Lagrangian, e.g.
/// derivative(L, {{u, v}, {lam, mu}}).  A field the form does not depend on
/// contributes nothing.
[[nodiscard]] inline FormExpr derivative(const FormExpr& form,
                                         std::span<const std::pair<FormExpr, FormExpr>> variations)
{
    FormExpr sum;
    for (const auto& [u, direction] : variations) {
        if (!u.isValid() || u.node()->type() != FormExprType::StateField || !u.node()->fieldId()) {
            throw std::invalid_argument("derivative: the variable is a field's StateField, but it is '" +
                                        u.toString() + "'");
        }
        auto part = directionalDerivativeWrtField(form, *u.node()->fieldId(), direction,
                                                  /*discrete_fields_vary=*/false);
        const auto value = part.isValid() ? part.node()->constantValue() : std::optional<Real>{Real(0.0)};
        if (value.has_value() && *value == Real(0.0)) {
            continue;
        }
        sum = sumOf(sum, part);
    }
    if (!sum.isValid()) {
        throw std::invalid_argument("derivative: '" + form.toString() + "' depends on none of the fields");
    }
    return sum;
}

[[nodiscard]] inline FormExpr derivative(const FormExpr& form,
                                         std::initializer_list<std::pair<FormExpr, FormExpr>> variations)
{
    return derivative(form, std::span<const std::pair<FormExpr, FormExpr>>(variations.begin(), variations.size()));
}

/**
 * @brief @p expr at the current iterate, held fixed in the Jacobian
 *
 * Every StateField of @p expr becomes the DiscreteField of the same field,
 * space and name, which the kernels evaluate as data (a constant of the
 * dual numbers and of the symbolic tangent), so frozen(u) builds exactly the
 * tree FormExpr::discreteField(u_id, V, "u") and the same kernels: Picard
 * convection advect(frozen(u), u), a stabilization parameter frozen(tau(u))
 * kept out of the Jacobian, or (u - frozen(u)) * v, a residual zero to round-off
 * whose Jacobian is the mass matrix.  The residual is unchanged.  Refused for a
 * TrialFunction (state the unknown as a StateField), a test function, an
 * integral, and a current-configuration geometry terminal (the moving-domain
 * tangent would still differentiate it).
 */
[[nodiscard]] inline FormExpr frozen(const FormExpr& expr)
{
    if (!expr.isValid()) {
        throw std::invalid_argument("frozen: empty expression");
    }
    return expr.transformNodes([&](const FormExprNode& n) -> std::optional<FormExpr> {
        const auto refuse = [&](const char* what) {
            throw std::invalid_argument(std::string("frozen: '") + expr.toString() + "' holds " + what +
                                        " ('" + n.toString() + "')");
        };
        switch (n.type()) {
        case FormExprType::StateField:
            if (!n.fieldId() || n.spaceSignature() == nullptr) {
                refuse("a StateField without its field");
            }
            return FormExpr::discreteField(*n.fieldId(), *n.spaceSignature(), n.toString());
        case FormExprType::TrialFunction:
            refuse("a TrialFunction; freeze the field's StateField");
            break;
        case FormExprType::TestFunction:
            refuse("a test function");
            break;
        case FormExprType::CellIntegral:
        case FormExprType::BoundaryIntegral:
        case FormExprType::InteriorFaceIntegral:
        case FormExprType::InterfaceIntegral:
        case FormExprType::CutVolumeIntegral:
        case FormExprType::GlobalIntegral:
            refuse("an integral; freeze its integrand's terms");
            break;
        case FormExprType::CurrentCoordinate:
        case FormExprType::CurrentJacobian:
        case FormExprType::CurrentJacobianDeterminant:
        case FormExprType::CurrentNormal:
        case FormExprType::CurrentMeasure:
        case FormExprType::SurfaceJacobian:
        case FormExprType::MeshVelocity:
            refuse("a current-configuration geometry terminal, which the moving-domain tangent differentiates");
            break;
        default:
            break;
        }
        return std::nullopt;
    });
}

// ---------------------------------------------------------------------------
// Auxiliary state vocabulary (generalized, physics-agnostic)
//
// These are the canonical helpers for referencing auxiliary state, inputs,
// and outputs in formulations.  They use the neutral terminology from
// the generalized AuxiliaryState subsystem.
// ---------------------------------------------------------------------------

/// Reference an auxiliary input by name (resolved to slot at setup time).
/// Replaces the legacy `boundaryIntegralValue(name)` for new formulations.
inline FormExpr AuxiliaryInput(std::string name)
{
    return FormExpr::auxiliaryInput(std::move(name));
}

/// Reference an auxiliary input by slot (JIT-friendly, post-resolution).
inline FormExpr AuxiliaryInputSlot(std::uint32_t slot)
{
    return FormExpr::auxiliaryInputRef(slot);
}

/// Reference an auxiliary model output by name (first-class coupling surface).
/// This is the preferred formulation-facing surface for consuming auxiliary
/// model results.  Raw auxiliary state access remains available as an
/// advanced path via `AuxiliaryState(name)`.
///
/// When multiple deployed models have the same output name, use the
/// instance-qualified overload `AuxiliaryOutput(instance, name)` instead.
/// The bare overload throws on ambiguity during auto-resolution.
inline FormExpr AuxiliaryOutput(std::string name)
{
    return FormExpr::auxiliaryOutput(std::move(name));
}

/// Instance-qualified auxiliary output reference.
/// Use when multiple deployed models expose outputs with the same name.
/// The expression is created with "instance/name" syntax and resolved to
/// a slot during form installation via `auxiliaryOutputSlotOf(instance, name)`.
inline FormExpr AuxiliaryOutput(const std::string& instance, const std::string& name)
{
    return FormExpr::auxiliaryOutput(instance + "/" + name);
}

/// Reference an auxiliary model output by slot (JIT-friendly, post-resolution).
inline FormExpr AuxiliaryOutputSlot(std::uint32_t slot)
{
    return FormExpr::auxiliaryOutputRef(slot);
}

/// Reference an auxiliary state variable by name (raw access, advanced path).
/// For new formulations, prefer `AuxiliaryOutput(name)` instead.
inline FormExpr AuxiliaryState(std::string name)
{
    return FormExpr::auxiliaryState(std::move(name));
}

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_VOCABULARY_H
