#ifndef SVMP_FE_FORMS_FORM_IR_H
#define SVMP_FE_FORMS_FORM_IR_H

/**
 * @file FormIR.h
 * @brief Compiled representation for FE/Forms expressions
 */

#include "Assembly/AssemblyKernel.h"
#include "Forms/FormExpr.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {

/**
 * @brief Integration domain for an integral term
 */
enum class IntegralDomain : std::uint8_t {
    Cell,
    Boundary,
    InteriorFace
};

/**
 * @brief A single integral term (domain + integrand)
 */
struct IntegralTerm {
    IntegralDomain domain{IntegralDomain::Cell};

    // For boundary integrals: marker >= 0 restricts to that marker; marker < 0 means "all".
    int boundary_marker{-1};

    // Continuous-time metadata: if > 0, this term contains dt(·,k) and represents
    // the k-th time derivative contribution (time discretization handled elsewhere).
    int time_derivative_order{0};

    // Integrand expression (no measure wrapper)
    FormExpr integrand{};

    // Cached string form for diagnostics
    std::string debug_string{};

    // Data requirements for this term
    assembly::RequiredData required_data{assembly::RequiredData::None};
};

/**
 * @brief Kind of form (how TrialFunction is interpreted during evaluation)
 */
enum class FormKind : std::uint8_t {
    Linear,     ///< L(v): TrialFunction should not appear
    Bilinear,   ///< a(u,v): TrialFunction is interpreted as basis/test-trial variation
    Residual    ///< F(u;v): TrialFunction is interpreted as the current solution field
};

/**
 * @brief Intermediate representation of a compiled form
 *
 * FormIR owns the integral-term decomposition plus per-term metadata
 * (required assembly context data, markers, etc.). Execution is performed by
 * FE/Forms kernels that evaluate each term against an AssemblyContext.
 */
class FormIR {
public:
    FormIR();
    ~FormIR();

    FormIR(FormIR&&) noexcept;
    FormIR& operator=(FormIR&&) noexcept;

    FormIR(const FormIR&) = delete;
    FormIR& operator=(const FormIR&) = delete;

    [[nodiscard]] bool isCompiled() const noexcept;
    [[nodiscard]] FormKind kind() const noexcept;
    [[nodiscard]] assembly::RequiredData requiredData() const noexcept;
    [[nodiscard]] const std::vector<assembly::FieldRequirement>& fieldRequirements() const noexcept;
    [[nodiscard]] const std::optional<FormExprNode::SpaceSignature>& testSpace() const noexcept;
    [[nodiscard]] const std::optional<FormExprNode::SpaceSignature>& trialSpace() const noexcept;
    [[nodiscard]] int maxTimeDerivativeOrder() const noexcept;
    [[nodiscard]] bool isTransient() const noexcept { return maxTimeDerivativeOrder() > 0; }

    [[nodiscard]] bool hasCellTerms() const noexcept;
    [[nodiscard]] bool hasBoundaryTerms() const noexcept;
    [[nodiscard]] bool hasInteriorFaceTerms() const noexcept;

    [[nodiscard]] const std::vector<IntegralTerm>& terms() const noexcept;

    [[nodiscard]] std::string dump() const;

    /**
     * @brief Apply a node-level transformation to every term integrand
     *
     * This is intended for setup-time rewrites (e.g., resolving ParameterSymbol
     * terminals to ParameterRef slots). It is not used on the assembly hot path.
     */
    void transformIntegrands(const FormExpr::NodeTransform& transform);

    using TermTransform = std::function<std::optional<FormExpr>(const FormExprNode&, const IntegralTerm&)>;

    /**
     * @brief Apply a node-level transformation to every term integrand, with term context
     *
     * This overload provides access to the current IntegralTerm (domain, marker, etc.)
     * while rewriting. It is intended for setup-time rewrites that need domain-aware
     * bookkeeping (e.g., inlinable constitutive state-update programs).
     */
    void transformIntegrands(const TermTransform& transform);

private:
    friend class FormCompiler;

    void setCompiled(bool compiled);
    void setKind(FormKind kind);
    void setRequiredData(assembly::RequiredData required);
    void setFieldRequirements(std::vector<assembly::FieldRequirement> reqs);
    void setTestSpace(std::optional<FormExprNode::SpaceSignature> sig);
    void setTrialSpace(std::optional<FormExprNode::SpaceSignature> sig);
    void setMaxTimeDerivativeOrder(int order);
    void setTerms(std::vector<IntegralTerm> terms);
    void setDump(std::string dump);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_FORM_IR_H
