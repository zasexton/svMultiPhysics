#ifndef SVMP_FE_FORMS_STANDARD_BCS_H
#define SVMP_FE_FORMS_STANDARD_BCS_H

/**
 * @file StandardBCs.h
 * @brief Standard physics-agnostic boundary condition implementations
 */

#include "Constraints/HDivNormalConstraint.h"
#include "Constraints/NormalComponentConstraint.h"
#include "Forms/BoundaryCondition.h"
#include "Forms/ResidualContributions.h"
#include "Systems/FESystem.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace bc {

namespace detail {

[[nodiscard]] inline analysis::TraceKind traceKindFor(ScalarTraceOperator op)
{
    return op == ScalarTraceOperator::NormalComponent
               ? analysis::TraceKind::NormalComponent
               : analysis::TraceKind::Value;
}

[[nodiscard]] inline analysis::InequalitySense inequalitySenseFor(TraceInequalitySense sense)
{
    switch (sense) {
    case TraceInequalitySense::LessEqual:
        return analysis::InequalitySense::LessEqual;
    case TraceInequalitySense::GreaterEqual:
        return analysis::InequalitySense::GreaterEqual;
    }

    return analysis::InequalitySense::None;
}

[[nodiscard]] inline std::string sourceFor(std::string_view name,
                                           analysis::DomainKind domain,
                                           int marker)
{
    const std::string marker_name =
        (domain == analysis::DomainKind::InterfaceFace) ? "interface " : "marker ";
    return std::string(name) + " on " + marker_name + std::to_string(marker);
}

} // namespace detail

/**
 * @brief Marker-only boundary condition used for conflict validation
 *
 * This BC reserves a boundary marker in BoundaryConditionManager::validate()
 * without contributing weak-form terms or strong constraints.
 */
class ReservedBC final : public BoundaryCondition {
public:
    explicit ReservedBC(int boundary_marker)
        : boundary_marker_(boundary_marker)
    {
        if (boundary_marker_ < 0) {
            throw std::invalid_argument("ReservedBC: boundary_marker must be >= 0");
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return boundary_marker_; }
    [[nodiscard]] std::string_view label() const override { return "ReservedBC"; }
    [[nodiscard]] bool hasWeakTerms() const override { return false; }

    void contributeToResidual(FormExpr& /*residual*/,
                              const FormExpr& /*u*/,
                              const FormExpr& /*v*/) const override
    {
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId /*field_id*/) const override
    {
        return {};
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId /*field_id*/, const systems::FESystem* /*system*/) const override
    {
        return {};  // Reserved BC — no mathematical constraint
    }

private:
    int boundary_marker_{-1};
};

/**
 * @brief Natural (Neumann/traction) BC: adds -∫ flux · v ds(marker)
 */
class NaturalBC : public BoundaryCondition {
public:
    NaturalBC(Boundary target, FormExpr flux)
        : target_(std::move(target))
        , flux_(std::move(flux))
    {
        if (target_.isInterface()) {
            throw std::invalid_argument("NaturalBC: an interface flux is an InterfaceTraceLoadBC");
        }
        if (!flux_.isValid()) {
            throw std::invalid_argument("NaturalBC: invalid flux expression");
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return target_.marker(); }
    [[nodiscard]] std::string_view label() const override { return "NaturalBC"; }

    void contributeToResidual(FormExpr& residual,
                              const FormExpr& /*u*/,
                              const FormExpr& v) const override
    {
        residual = residual - target_.integrate(inner(flux_, v));
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId /*field_id*/) const override
    {
        return {};
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId field_id, const systems::FESystem* /*system*/) const override
    {
        analysis::BoundaryConditionDescriptor d;
        d.primary_variable = analysis::VariableKey::field(field_id);
        d.boundary_marker = target_.marker();
        d.trace_kind = analysis::TraceKind::Flux;
        d.enforcement_kind = analysis::EnforcementKind::WeakConsistent;
        d.anchors_constant_mode = false;
        d.anchors_rigid_body_translation = false;
        d.anchors_rigid_body_rotation = false;
        d.source = "NaturalBC on marker " + std::to_string(target_.marker());
        return {d};
    }

private:
    Boundary target_;
    FormExpr flux_{};
};

/**
 * @brief Robin BC: adds ∫ alpha * (u · v) ds(marker) - ∫ rhs · v ds(marker)
 */
class RobinBC : public BoundaryCondition {
public:
    RobinBC(Boundary target, FormExpr alpha, FormExpr rhs)
        : target_(std::move(target))
        , alpha_(std::move(alpha))
        , rhs_(std::move(rhs))
    {
        if (target_.isInterface()) {
            throw std::invalid_argument("RobinBC: an interface Robin law is an InterfaceTraceRobinBC");
        }
        if (!alpha_.isValid()) {
            throw std::invalid_argument("RobinBC: invalid alpha expression");
        }
        if (!rhs_.isValid()) {
            throw std::invalid_argument("RobinBC: invalid rhs expression");
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return target_.marker(); }
    [[nodiscard]] std::string_view label() const override { return "RobinBC"; }

    void contributeToResidual(FormExpr& residual,
                              const FormExpr& u,
                              const FormExpr& v) const override
    {
        residual = residual + target_.integrate(alpha_ * inner(u, v)) - target_.integrate(inner(rhs_, v));
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId /*field_id*/) const override
    {
        return {};
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId field_id, const systems::FESystem* /*system*/) const override
    {
        analysis::BoundaryConditionDescriptor d;
        d.primary_variable = analysis::VariableKey::field(field_id);
        d.boundary_marker = target_.marker();
        d.trace_kind = analysis::TraceKind::Mixed;
        d.enforcement_kind = analysis::EnforcementKind::WeakPenalty;
        d.anchors_constant_mode = true;
        d.anchors_rigid_body_translation = true;
        d.anchors_rigid_body_rotation = false;
        d.source = "RobinBC on marker " + std::to_string(target_.marker());
        return {d};
    }

private:
    Boundary target_;
    FormExpr alpha_{};
    FormExpr rhs_{};
};

// ---- Dirichlet conditions: one condition, the enforcement an argument ----

/// Strong enforcement: the constrained degrees of freedom take the value.
struct Strong {};

/// Penalty enforcement: the residual term beta (u - g) . v.
struct Penalty {
    FormExpr beta{};
};

/// How a condition u = g is enforced: strongly, by Nitsche's method
/// (BoundaryConditions.h, with the operator's NaturalFlux) or by a penalty.
using Enforcement = std::variant<Strong, Nitsche, Penalty>;

/// A part of a flux as a linear function of its unknown: applied to the
/// unknown's state it is the part of the flux; applied to the unknown's test
/// function it is the matching part of the adjoint flux.
using LinearFlux = std::function<FormExpr(const FormExpr&)>;

/**
 * @brief One part of an operator's conormal flux on a boundary
 *
 * The part is linear in one unknown: the condition's own field, or another
 * unknown named by its field and symbols (the pressure of a stress).  Nitsche's
 * method pairs the part of the unknown's state with the condition's test
 * function (consistency) and the part of the unknown's test function with
 * u - g in that unknown's row (adjoint consistency).
 */
struct FluxPart {
    enum class Kind : std::uint8_t {
        Normal,    ///< q(u), the normal flux of a scalar field (k grad(u).n)
        Conormal,  ///< T(u) n, the conormal of a tensor of a vector field (2 mu eps(u) n)
        Isotropic, ///< -s n, with s an unknown (the pressure part -p n of a stress)
    };
    Kind kind{Kind::Normal};
    LinearFlux of{};                ///< q or T (Normal, Conormal)
    FieldId field{INVALID_FIELD_ID}; ///< Isotropic: the unknown s
    FormExpr state{};               ///< Isotropic: the state of s
    FormExpr test{};                ///< Isotropic: the test function of s
    /// Conormal: the bilinear form whose flux T n is.  A Nitsche route on a
    /// generated boundary that the formulation certifies is built from it.
    std::optional<GeneratedBoundaryNitscheBilinearForm> form{};
};

/// q(u): the normal flux of a scalar field.
[[nodiscard]] inline FluxPart normalFlux(LinearFlux q)
{
    return FluxPart{.kind = FluxPart::Kind::Normal, .of = std::move(q)};
}

/// T(u) n: the conormal flux of a vector field, with the bilinear form T
/// derives from when a certified generated-boundary route needs it.
[[nodiscard]] inline FluxPart conormalFlux(LinearFlux T,
                                           std::optional<GeneratedBoundaryNitscheBilinearForm> form = std::nullopt)
{
    return FluxPart{.kind = FluxPart::Kind::Conormal, .of = std::move(T), .form = std::move(form)};
}

/// -s n: the isotropic part of a flux in the unknown s (field, state, test).
[[nodiscard]] inline FluxPart isotropicFlux(FieldId field, FormExpr state, FormExpr test)
{
    return FluxPart{.kind = FluxPart::Kind::Isotropic,
                    .field = field,
                    .state = std::move(state),
                    .test = std::move(test)};
}

/**
 * @brief The conormal flux of an operator on its field, stated once
 *
 * The flux is the sum of its parts, in the order Nitsche's terms are emitted;
 * the penalty weight w scales the Nitsche penalty gamma p^2 w.
 */
struct NaturalFlux {
    std::vector<FluxPart> parts{};
    FormExpr penalty_weight{};
};

/// The diffusion flux k grad(u).n of a scalar field, penalty weight k / h_n.
[[nodiscard]] inline NaturalFlux diffusionFlux(FormExpr k)
{
    const auto n = FormExpr::normal();
    const auto h = (2.0 * FormExpr::cellVolume()) / FormExpr::facetArea();
    auto q = [k, n](const FormExpr& w) { return k * inner(grad(w), n); };
    return NaturalFlux{.parts = {normalFlux(std::move(q))}, .penalty_weight = k / h};
}

/**
 * @brief The residual rows a condition's weak terms enter
 *
 * One row per unknown.  A condition adds or subtracts integrals in the rows of
 * the unknowns its terms test.  A sink that certifies generated-boundary
 * Nitsche routes receives their bindings.
 */
class RowSink {
public:
    virtual ~RowSink() = default;
    virtual void add(FieldId row, const FormExpr& integral) = 0;
    virtual void subtract(FieldId row, const FormExpr& integral) = 0;
    [[nodiscard]] virtual bool certifiesGeneratedRoutes() const { return false; }
    virtual void certify(GeneratedBoundaryNitscheTraceFormBinding /*binding*/)
    {
        throw std::logic_error("forms::bc::RowSink::certify: this sink does not certify generated routes");
    }
};

/// The rows of a ResidualContributions sink: an added integral is appended,
/// a subtracted one appended negated.  With @p bindings, generated-boundary
/// routes are certified and their bindings collected there.
class ResidualRows final : public RowSink {
public:
    explicit ResidualRows(ResidualContributions& rows,
                          std::vector<GeneratedBoundaryNitscheTraceFormBinding>* bindings = nullptr)
        : rows_(rows)
        , bindings_(bindings)
    {
    }

    void add(FieldId row, const FormExpr& integral) override { rows_.append(row, integral); }
    void subtract(FieldId row, const FormExpr& integral) override { rows_.append(row, -integral); }
    [[nodiscard]] bool certifiesGeneratedRoutes() const override { return bindings_ != nullptr; }
    void certify(GeneratedBoundaryNitscheTraceFormBinding binding) override
    {
        bindings_->push_back(std::move(binding));
    }

private:
    ResidualContributions& rows_;
    std::vector<GeneratedBoundaryNitscheTraceFormBinding>* bindings_{nullptr};
};

/**
 * @brief The Dirichlet condition u = g on a boundary, enforced as declared
 *
 * One condition for every enforcement:
 * - Strong{}: the degrees of freedom on the boundary take g (one declaration
 *   per component of a component list).
 * - Nitsche{...}: weak, from the operator's NaturalFlux: the consistency
 *   terms of the parts in u's row, in part order; the adjoint terms in the
 *   row of each part's unknown, the parts in u first; then the penalty
 *   gamma p^2 w (u - g) . v.  A flux with parts in other
 *   unknowns makes the condition multi-row (contributeToRows).
 * - Penalty{beta}: weak, beta (u - g) . v.
 *
 * The target is a Boundary: a marker or an exterior measure (the generated
 * wet part of a cut boundary: weak enforcement only).  The value is the whole field (a
 * scalar, or a vector expression), every component of a vector field, or, for
 * strong enforcement, some components; strong component conditions of one
 * boundary merge (mergeComponents, BoundaryConditionManager::add).  The
 * manifest names the condition by
 * its enforcement: EssentialBC (strong, the whole field or every component),
 * ComponentEssentialBC (strong, some components), NitscheBC, PenaltyBC.
 */
class Dirichlet final : public BoundaryCondition {
public:
    /// u = g for the whole field.
    Dirichlet(Boundary target,
              FormExpr value,
              Enforcement how = Strong{},
              NaturalFlux flux = {},
              std::string symbol = "u")
        : Dirichlet(std::move(target), std::vector<FormExpr>{std::move(value)}, std::move(how), std::move(flux),
                    std::move(symbol))
    {
    }

    /// u_i = g_i for every component i of a vector field.
    Dirichlet(Boundary target,
              std::vector<FormExpr> components,
              Enforcement how = Strong{},
              NaturalFlux flux = {},
              std::string symbol = "u")
        : how_(std::move(how))
        , flux_(std::move(flux))
        , symbol_(std::move(symbol))
    {
        setTarget(target);
        if (components.empty()) {
            throw std::invalid_argument("bc::Dirichlet: empty value component list");
        }
        components_.reserve(components.size());
        for (std::size_t c = 0; c < components.size(); ++c) {
            components_.emplace_back(components.size() == 1u ? -1 : static_cast<int>(c), std::move(components[c]));
        }
        validate();
    }

    /// u_i = g_i for the listed components i (strong enforcement); the
    /// descriptors name the condition @p source_name.
    Dirichlet(Boundary target,
              std::vector<std::pair<int, FormExpr>> components,
              std::string symbol,
              std::string source_name)
        : symbol_(std::move(symbol))
        , source_name_(std::move(source_name))
        , some_components_(true)
    {
        setTarget(target);
        if (components.empty()) {
            throw std::invalid_argument("bc::Dirichlet: empty component list");
        }
        for (const auto& [component, value] : components) {
            if (component < 0) {
                throw std::invalid_argument("bc::Dirichlet: component must be >= 0");
            }
        }
        components_ = std::move(components);
        validate();
    }

    [[nodiscard]] int boundaryMarker() const override { return boundary_marker_; }

    [[nodiscard]] std::string_view label() const override
    {
        if (isStrong()) {
            return some_components_ ? "ComponentEssentialBC" : "EssentialBC";
        }
        if (std::holds_alternative<Penalty>(how_)) {
            return "PenaltyBC";
        }
        return "NitscheBC";
    }

    [[nodiscard]] bool hasWeakTerms() const override { return !isStrong(); }

    [[nodiscard]] const Enforcement& enforcement() const noexcept { return how_; }

    /// Whether this is a strong condition of listed components, which others
    /// on its target can join (mergeComponents).
    [[nodiscard]] bool mergesComponents() const noexcept { return isStrong() && some_components_; }

    /// Whether @p other targets the same boundary with the same symbol.
    [[nodiscard]] bool sameTarget(const Dirichlet& other) const noexcept
    {
        return boundary_marker_ == other.boundary_marker_ && measure_ == other.measure_ &&
               symbol_ == other.symbol_;
    }

    /// The value of @p component, when the condition prescribes it.
    [[nodiscard]] const FormExpr* componentValue(int component) const noexcept
    {
        for (const auto& [c, value] : components_) {
            if (c == component) {
                return &value;
            }
        }
        return nullptr;
    }

    /// The prescribed components and their values, in order.
    [[nodiscard]] const std::vector<std::pair<int, FormExpr>>& components() const noexcept { return components_; }

    /// Adds the components of @p other, a strong component condition of the
    /// same target, to this one; the components stay in index order.  A
    /// component both prescribe is refused.
    void mergeComponents(const Dirichlet& other)
    {
        if (!mergesComponents() || !other.mergesComponents() || !sameTarget(other)) {
            throw std::invalid_argument(
                "bc::Dirichlet::mergeComponents: two strong component conditions of one target merge");
        }
        for (const auto& [component, value] : other.components_) {
            addComponent(component, value);
        }
    }

    /// Adds @p component with @p value to a strong component condition, in
    /// index order; a component already prescribed is refused.
    void addComponent(int component, FormExpr value)
    {
        if (!mergesComponents()) {
            throw std::invalid_argument("bc::Dirichlet::addComponent: not a strong component condition");
        }
        if (componentValue(component) != nullptr) {
            throw std::invalid_argument("bc::Dirichlet: component " + std::to_string(component) +
                                        " of boundary marker " + std::to_string(boundary_marker_) +
                                        " is prescribed twice");
        }
        const auto at = std::find_if(components_.begin(), components_.end(),
                                     [component](const auto& entry) { return entry.first > component; });
        components_.emplace(at, component, std::move(value));
    }

    void contributeToResidual(FormExpr& residual, const FormExpr& u, const FormExpr& v) const override
    {
        if (isStrong()) {
            return;
        }
        class OneRow final : public RowSink {
        public:
            explicit OneRow(FormExpr& residual)
                : residual_(residual)
            {
            }
            void add(FieldId /*row*/, const FormExpr& integral) override { residual_ = residual_ + integral; }
            void subtract(FieldId /*row*/, const FormExpr& integral) override { residual_ = residual_ - integral; }

        private:
            FormExpr& residual_;
        } row(residual);
        for (const auto& part : flux_.parts) {
            if (part.kind == FluxPart::Kind::Isotropic) {
                throw std::invalid_argument(
                    "bc::Dirichlet: a flux with a part in another unknown is multi-row; use contributeToRows");
            }
        }
        emit(row, u, v, fieldOf(u));
    }

    /// The weak terms into the rows of @p field (u, v) and of the unknowns of
    /// the flux's parts.
    void contributeToRows(RowSink& rows, const FormExpr& u, const FormExpr& v, FieldId field) const
    {
        if (!isStrong()) {
            emit(rows, u, v, field);
        }
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId field_id) const override
    {
        if (!isStrong()) {
            return {};
        }
        std::vector<StrongDirichlet> out;
        out.reserve(components_.size());
        for (const auto& [component, value] : components_) {
            out.push_back(strongDirichlet(field_id, boundary_marker_, value, symbol_, component));
        }
        return out;
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId field_id, const systems::FESystem* /*system*/) const override
    {
        const std::string name = source_name_.empty() ? std::string(label()) : source_name_;
        const std::string on_marker = " on marker " + std::to_string(boundary_marker_);
        auto descriptor = [&](int component) {
            analysis::BoundaryConditionDescriptor d;
            d.primary_variable = component < 0 ? analysis::VariableKey::field(field_id)
                                               : analysis::VariableKey::field(field_id, component);
            d.component = component;
            d.boundary_marker = boundary_marker_;
            d.trace_kind = analysis::TraceKind::Value;
            d.anchors_constant_mode = true;
            d.anchors_rigid_body_translation = true;
            d.anchors_rigid_body_rotation = false;
            d.source = component < 0 ? name + on_marker : name + " comp " + std::to_string(component) + on_marker;
            return d;
        };
        std::vector<analysis::BoundaryConditionDescriptor> out;
        if (isStrong()) {
            out.reserve(components_.size());
            for (const auto& [component, value] : components_) {
                auto d = descriptor(component);
                d.enforcement_kind = analysis::EnforcementKind::Strong;
                out.push_back(std::move(d));
            }
            return out;
        }
        auto d = descriptor(-1);
        if (std::holds_alternative<Penalty>(how_)) {
            d.enforcement_kind = analysis::EnforcementKind::WeakPenalty;
            out.push_back(std::move(d));
            return out;
        }
        const auto& nitsche = std::get<Nitsche>(how_);
        const bool symmetric = nitsche.variant == NitscheVariant::Symmetric;
        d.enforcement_kind = analysis::EnforcementKind::WeakNitsche;
        analysis::NitscheMetadata metadata;
        metadata.variant = symmetric ? analysis::NitscheVariant::Symmetric : analysis::NitscheVariant::Nonsymmetric;
        metadata.primal_consistency_terms_present = true;
        metadata.adjoint_consistency_terms_present = symmetric;
        metadata.rhs_consistency_terms_present = true;
        metadata.penalty_positive = nitsche.gamma > Real(0.0);
        metadata.penalty_scaling_verified =
            nitsche.scale_with_p && (!measure_.has_value() || measure_->isFullPhysical());
        d.nitsche = metadata;
        out.push_back(std::move(d));
        return out;
    }

private:
    [[nodiscard]] bool isStrong() const noexcept { return std::holds_alternative<Strong>(how_); }

    void setTarget(const Boundary& target)
    {
        if (target.isInterface()) {
            throw std::invalid_argument(
                "bc::Dirichlet: an interface condition names its trace (InterfaceTraceNitscheBC)");
        }
        boundary_marker_ = target.marker();
        measure_ = target.exteriorMeasure();
    }

    void validate() const
    {
        for (const auto& [component, value] : components_) {
            if (!value.isValid()) {
                throw std::invalid_argument("bc::Dirichlet: invalid value expression");
            }
        }
        if (isStrong()) {
            if (measure_.has_value() && !measure_->isFullPhysical()) {
                throw std::invalid_argument(
                    "bc::Dirichlet: strong enforcement acts on the mesh faces of a marker; a generated active "
                    "subset needs Nitsche{} or Penalty{}");
            }
            return;
        }
        if (some_components_) {
            throw std::invalid_argument("bc::Dirichlet: weak enforcement of some components is not supported");
        }
        for (const auto& [component, value] : components_) {
            if (value.hasTest() || value.hasTrial()) {
                throw std::invalid_argument("bc::Dirichlet: value must not contain test/trial functions");
            }
        }
        if (const auto* penalty = std::get_if<Penalty>(&how_)) {
            if (!penalty->beta.isValid()) {
                throw std::invalid_argument("bc::Dirichlet: invalid penalty coefficient");
            }
            return;
        }
        const auto& nitsche = std::get<Nitsche>(how_);
        if (!(nitsche.gamma > Real(0.0)) || !std::isfinite(nitsche.gamma)) {
            throw std::invalid_argument("bc::Dirichlet: Nitsche gamma must be finite and > 0");
        }
        if (flux_.parts.empty() || !flux_.penalty_weight.isValid()) {
            throw std::invalid_argument("bc::Dirichlet: Nitsche enforcement needs the operator's flux");
        }
        if (flux_.penalty_weight.hasTest() || flux_.penalty_weight.hasTrial()) {
            throw std::invalid_argument("bc::Dirichlet: the penalty weight must not contain test/trial functions");
        }
        for (const auto& part : flux_.parts) {
            const bool isotropic = part.kind == FluxPart::Kind::Isotropic;
            if (isotropic ? (part.field == INVALID_FIELD_ID || !part.state.isValid() || !part.test.isValid())
                          : !part.of) {
                throw std::invalid_argument("bc::Dirichlet: incomplete flux part");
            }
        }
    }

    [[nodiscard]] static FieldId fieldOf(const FormExpr& u)
    {
        const auto* node = u.node();
        const auto field = node == nullptr ? std::optional<FieldId>{} : node->fieldId();
        return field.value_or(INVALID_FIELD_ID);
    }

    [[nodiscard]] static bool isVectorField(const FormExpr& u)
    {
        const auto* node = u.node();
        const auto* signature = node == nullptr ? nullptr : node->spaceSignature();
        return signature != nullptr && signature->value_dimension > 1;
    }

    [[nodiscard]] FormExpr integrate(const FormExpr& integrand) const
    {
        return measure_.has_value() ? integrand.dExteriorBoundary(*measure_) : integrand.ds(boundary_marker_);
    }

    void emit(RowSink& rows, const FormExpr& u, const FormExpr& v, FieldId field) const
    {
        const bool vector = isVectorField(u);
        if (!vector && components_.size() > 1u) {
            throw std::invalid_argument("bc::Dirichlet: a component list needs a vector field");
        }
        const auto value =
            components_.size() == 1u ? components_.front().second : FormExpr::asVector(values());
        const auto diff = u - value;
        if (const auto* penalty = std::get_if<Penalty>(&how_)) {
            rows.add(field, integrate(vector ? penalty->beta * inner(diff, v) : penalty->beta * diff * v));
            return;
        }

        const auto& nitsche = std::get<Nitsche>(how_);
        const bool symmetric = nitsche.variant == NitscheVariant::Symmetric;
        const auto n = FormExpr::normal();
        const bool certified_route =
            measure_.has_value() && !measure_->isFullPhysical() && rows.certifiesGeneratedRoutes();

        // Consistency, part by part: -(flux . v).  A certified generated
        // route replaces the conormal part's three terms.
        for (const auto& part : flux_.parts) {
            switch (part.kind) {
            case FluxPart::Kind::Normal:
                if (certified_route) {
                    throw std::invalid_argument(
                        "bc::Dirichlet: a certified generated-boundary route needs a conormal flux");
                }
                rows.subtract(field, integrate(part.of(u) * v));
                break;
            case FluxPart::Kind::Conormal:
                if (certified_route) {
                    if (!part.form.has_value()) {
                        throw std::invalid_argument(
                            "bc::Dirichlet: a certified generated-boundary route needs the flux's bilinear form");
                    }
                    auto route = buildGeneratedBoundaryNitscheTraceTerms(
                        u, v, value, *part.form, boundary_marker_, measure_->generatedActiveBoundaryMarker(),
                        nitsche);
                    rows.add(field, route.route_contribution);
                    rows.certify(std::move(route.binding));
                    break;
                }
                rows.subtract(field, integrate(inner(part.of(u) * n, v)));
                break;
            case FluxPart::Kind::Isotropic:
                rows.add(field, integrate(part.state * inner(n, v)));
                break;
            }
        }
        // Adjoint consistency in the row of each part's unknown,
        // -(adjoint flux . (u - g)) when symmetric, + otherwise: the parts in
        // u first, then the parts in other unknowns, each in part order.
        const auto adjoint = [&](FieldId row, bool flux_sign_positive, const FormExpr& integral) {
            if (flux_sign_positive == symmetric) {
                rows.subtract(row, integral);
            } else {
                rows.add(row, integral);
            }
        };
        for (const auto& part : flux_.parts) {
            if (part.kind == FluxPart::Kind::Normal) {
                adjoint(field, true, integrate(part.of(v) * diff));
            } else if (part.kind == FluxPart::Kind::Conormal && !certified_route) {
                adjoint(field, true, integrate(inner(part.of(v) * n, diff)));
            }
        }
        for (const auto& part : flux_.parts) {
            if (part.kind == FluxPart::Kind::Isotropic) {
                adjoint(part.field, false, integrate(part.test * inner(n, diff)));
            }
        }
        if (certified_route) {
            return;
        }
        const auto penalty = buildTraceNitschePenalty(flux_.penalty_weight, u, nitsche);
        rows.add(field, integrate(vector ? penalty * inner(diff, v) : penalty * diff * v));
    }

    [[nodiscard]] std::vector<FormExpr> values() const
    {
        std::vector<FormExpr> out;
        out.reserve(components_.size());
        for (const auto& [component, value] : components_) {
            out.push_back(value);
        }
        return out;
    }

    int boundary_marker_{-1};
    std::optional<ExteriorBoundaryMeasure> measure_{};
    std::vector<std::pair<int, FormExpr>> components_{};
    Enforcement how_{Strong{}};
    NaturalFlux flux_{};
    std::string symbol_{"u"};
    std::string source_name_{};
    bool some_components_{false};
};

/// The strong Dirichlet condition of the active components of @p values on
/// @p marker, named @p source_name in its descriptors.
template <class Values, class ActiveComponents>
[[nodiscard]] inline std::unique_ptr<Dirichlet>
componentDirichlet(const Values& values,
                   const ActiveComponents& active_components,
                   int dim,
                   int marker,
                   std::string_view value_prefix,
                   std::string_view symbol,
                   std::string_view source_name,
                   ComponentValueNameStyle name_style = ComponentValueNameStyle::Indexed)
{
    std::vector<std::pair<int, FormExpr>> components;
    components.reserve(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        if (!active_components[static_cast<std::size_t>(d)]) {
            continue;
        }
        components.emplace_back(
            d,
            toScalarExpr(values[static_cast<std::size_t>(d)],
                         detail::makeComponentValueName(value_prefix, marker, d, name_style)));
    }

    return std::make_unique<Dirichlet>(marker, std::move(components), std::string(symbol), std::string(source_name));
}

/// Strong component conditions merged by target: each condition in order,
/// a later component condition on the target of an earlier one added to it
/// (Dirichlet::mergeComponents); other conditions are kept as they are.
[[nodiscard]] inline std::vector<std::unique_ptr<Dirichlet>>
mergeComponents(std::vector<std::unique_ptr<Dirichlet>> conditions)
{
    std::vector<std::unique_ptr<Dirichlet>> merged;
    merged.reserve(conditions.size());
    for (auto& condition : conditions) {
        if (!condition) {
            throw std::invalid_argument("forms::bc::mergeComponents: null condition");
        }
        const auto earlier = std::find_if(merged.begin(), merged.end(), [&](const auto& existing) {
            return condition->mergesComponents() && existing->mergesComponents() && existing->sameTarget(*condition);
        });
        if (earlier != merged.end()) {
            (*earlier)->mergeComponents(*condition);
        } else {
            merged.push_back(std::move(condition));
        }
    }
    return merged;
}

/**
 * @brief The strong conditions on a rate r = dt(u) implied by those on u
 *
 * @p rates are the rate conditions stated explicitly; they are merged by
 * target and kept.  On each component that a condition of @p values
 * prescribes and no rate condition does, a constant value gives the rate
 * zero: the component joins the rate condition of its boundary, or a new one
 * named @p symbol and @p source_name.  A value that varies in time or space
 * needs a rate condition of its own, and a constant value refuses a nonzero
 * rate on the same component.  The value conditions are strong component
 * conditions (componentDirichlet).
 */
[[nodiscard]] inline std::vector<std::unique_ptr<Dirichlet>>
derivedRateConditions(const std::vector<std::unique_ptr<Dirichlet>>& values,
                      std::vector<std::unique_ptr<Dirichlet>> rates,
                      const std::string& symbol,
                      const std::string& source_name)
{
    const auto constant = [](const FormExpr& value) {
        return value.node() != nullptr ? value.node()->constantValue() : std::optional<Real>{};
    };
    auto merged = mergeComponents(std::move(rates));
    for (const auto& value : values) {
        if (!value || !value->mergesComponents()) {
            throw std::invalid_argument("forms::bc::derivedRateConditions: values are strong component conditions");
        }
        Dirichlet* rate = nullptr;
        for (const auto& existing : merged) {
            if (existing->mergesComponents() && existing->boundaryMarker() == value->boundaryMarker()) {
                rate = existing.get();
                break;
            }
        }
        for (const auto& [component, g] : value->components()) {
            const auto fixed = constant(g);
            if (const FormExpr* prescribed = rate != nullptr ? rate->componentValue(component) : nullptr) {
                const auto r = constant(*prescribed);
                if (fixed.has_value() && !(r.has_value() && *r == Real{0.0})) {
                    throw std::invalid_argument(
                        "forms::bc::derivedRateConditions: a fixed value needs rate zero on the same component "
                        "(boundary marker " + std::to_string(value->boundaryMarker()) + ", component " +
                        std::to_string(component) + ")");
                }
                continue;
            }
            if (!fixed.has_value()) {
                throw std::invalid_argument(
                    "forms::bc::derivedRateConditions: a value that varies in time or space needs a rate condition "
                    "of its own (boundary marker " + std::to_string(value->boundaryMarker()) + ", component " +
                    std::to_string(component) + ")");
            }
            if (rate == nullptr) {
                merged.push_back(std::make_unique<Dirichlet>(
                    value->boundaryMarker(),
                    std::vector<std::pair<int, FormExpr>>{{component, FormExpr::constant(0.0)}}, symbol,
                    source_name));
                rate = merged.back().get();
                continue;
            }
            rate->addComponent(component, FormExpr::constant(0.0));
        }
    }
    return merged;
}

// ---- Conditions from typed boundary data (BoundaryConditions.h BoundaryData) ----

/// NaturalBC(marker, g): the residual term -(g v).ds(marker).
[[nodiscard]] inline std::unique_ptr<BoundaryCondition>
naturalBC(const BoundaryData<ScalarValue>& data, std::string_view prefix)
{
    return std::make_unique<NaturalBC>(data.boundary_marker, value(data, prefix));
}

/// NaturalBC(marker, asVector(g)) of a vector flux; every component must be active.
[[nodiscard]] inline std::unique_ptr<BoundaryCondition>
naturalBC(const BoundaryData<ComponentValues>& data,
          int dim,
          std::string_view prefix,
          ComponentValueNameStyle name_style = ComponentValueNameStyle::Indexed)
{
    for (int d = 0; d < dim; ++d) {
        if (!data.value.active[static_cast<std::size_t>(d)]) {
            throw std::invalid_argument("forms::bc::naturalBC: a vector flux has every component");
        }
    }
    return std::make_unique<NaturalBC>(data.boundary_marker,
                                       FormExpr::asVector(values(data, dim, prefix, name_style)));
}

/// RobinBC(marker, alpha, rhs): the residual term ((alpha u - rhs) v).ds(marker).
[[nodiscard]] inline std::unique_ptr<BoundaryCondition>
robinBC(const BoundaryData<RobinValue>& data, std::string_view alpha_prefix, std::string_view rhs_prefix)
{
    const int marker = boundaryMarkerOrThrow(data, "forms::bc::robinBC");
    return std::make_unique<RobinBC>(marker,
                                     toScalarExpr(data.value.alpha, markerValueName(alpha_prefix, marker)),
                                     toScalarExpr(data.value.rhs, markerValueName(rhs_prefix, marker)));
}

/// Dirichlet(marker, g, how, flux, symbol): u = g on the marker.
[[nodiscard]] inline std::unique_ptr<BoundaryCondition>
dirichlet(const BoundaryData<ScalarValue>& data,
          std::string_view prefix,
          Enforcement how = Strong{},
          NaturalFlux flux = {},
          std::string symbol = "u")
{
    return std::make_unique<Dirichlet>(data.boundary_marker, value(data, prefix), std::move(how), std::move(flux),
                                       std::move(symbol));
}

/// The strong Dirichlet condition of the active components: every component
/// gives Dirichlet(marker, {g_0, ..., g_{dim-1}}, Strong{}, {}, symbol), some
/// give the components named @p source_name (componentDirichlet).
[[nodiscard]] inline std::unique_ptr<BoundaryCondition>
dirichlet(const BoundaryData<ComponentValues>& data,
          int dim,
          std::string_view prefix,
          std::string_view symbol,
          std::string_view source_name,
          ComponentValueNameStyle name_style = ComponentValueNameStyle::Indexed)
{
    const int marker = boundaryMarkerOrThrow(data, "forms::bc::dirichlet");
    bool every_component = true;
    for (int d = 0; d < dim; ++d) {
        every_component = every_component && data.value.active[static_cast<std::size_t>(d)];
    }
    if (every_component) {
        return std::make_unique<Dirichlet>(marker, values(data, dim, prefix, name_style), Strong{}, NaturalFlux{},
                                           std::string(symbol));
    }
    return componentDirichlet(data.value.values, data.value.active, dim, marker, prefix, symbol, source_name,
                              name_style);
}

/**
 * @brief Essential BC on the scalar normal trace of an H(div) field: u·n = g
 */
class NormalTraceEssentialBC final : public BoundaryCondition {
public:
    explicit NormalTraceEssentialBC(int boundary_marker, FormExpr value)
        : boundary_marker_(boundary_marker)
        , value_(std::move(value))
    {
        if (boundary_marker_ < 0) {
            throw std::invalid_argument("NormalTraceEssentialBC: boundary_marker must be >= 0");
        }
        if (!value_.isValid()) {
            throw std::invalid_argument("NormalTraceEssentialBC: invalid value expression");
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return boundary_marker_; }
    [[nodiscard]] std::string_view label() const override { return "NormalTraceEssentialBC"; }
    [[nodiscard]] bool hasWeakTerms() const override { return false; }

    void contributeToResidual(FormExpr& /*residual*/,
                              const FormExpr& /*u*/,
                              const FormExpr& /*v*/) const override
    {
    }

    void installSystemConstraints(systems::FESystem& system, FieldId field_id) const override
    {
        system.addSystemConstraint(
            std::make_unique<constraints::HDivNormalConstraint>(field_id, boundary_marker_, value_));
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId /*field_id*/) const override
    {
        return {};
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId field_id, const systems::FESystem* /*system*/) const override
    {
        analysis::BoundaryConditionDescriptor d;
        d.primary_variable = analysis::VariableKey::field(field_id);
        d.boundary_marker = boundary_marker_;
        d.trace_kind = analysis::TraceKind::NormalComponent;
        d.enforcement_kind = analysis::EnforcementKind::Strong;
        d.anchors_constant_mode = true;
        d.anchors_rigid_body_translation = true;
        d.anchors_rigid_body_rotation = false;
        d.source = "NormalTraceEssentialBC on marker " + std::to_string(boundary_marker_);
        return {d};
    }

private:
    int boundary_marker_{-1};
    FormExpr value_{};
};

/**
 * @brief Strong normal component of a vector Lagrange field: u·n = g
 *
 * Free slip for g = 0.  Installs a constraints::NormalComponentConstraint on
 * the marker; the tangential components keep their natural condition.
 * Walls that meet at an edge need both normals there: state them in one
 * NormalComponentConstraint with all their markers.
 */
class NormalEssentialBC final : public BoundaryCondition {
public:
    explicit NormalEssentialBC(int boundary_marker,
                               FormExpr value = FormExpr::constant(0.0),
                               constraints::NormalComponentOptions options = {})
        : boundary_marker_(boundary_marker)
        , value_(std::move(value))
        , options_(options)
    {
        if (boundary_marker_ < 0) {
            throw std::invalid_argument("NormalEssentialBC: boundary_marker must be >= 0");
        }
        if (!value_.isValid()) {
            throw std::invalid_argument("NormalEssentialBC: invalid value expression");
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return boundary_marker_; }
    [[nodiscard]] std::string_view label() const override { return "NormalEssentialBC"; }
    [[nodiscard]] bool hasWeakTerms() const override { return false; }

    void contributeToResidual(FormExpr& /*residual*/,
                              const FormExpr& /*u*/,
                              const FormExpr& /*v*/) const override
    {
    }

    void installSystemConstraints(systems::FESystem& system, FieldId field_id) const override
    {
        system.addSystemConstraint(std::make_unique<constraints::NormalComponentConstraint>(
            field_id, std::vector<int>{boundary_marker_}, value_, options_));
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId /*field_id*/) const override
    {
        return {};
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId field_id, const systems::FESystem* /*system*/) const override
    {
        analysis::BoundaryConditionDescriptor d;
        d.primary_variable = analysis::VariableKey::field(field_id);
        d.boundary_marker = boundary_marker_;
        d.trace_kind = analysis::TraceKind::NormalComponent;
        d.enforcement_kind = analysis::EnforcementKind::Strong;
        d.anchors_constant_mode = true;
        d.anchors_rigid_body_translation = true;
        d.anchors_rigid_body_rotation = false;
        d.source = "NormalEssentialBC on marker " + std::to_string(boundary_marker_);
        return {d};
    }

private:
    int boundary_marker_{-1};
    FormExpr value_{};
    constraints::NormalComponentOptions options_{};
};

/**
 * @brief Weak trace load BC: adds -∫ g * tau(v) ds(marker)
 *
 * The default trace operator is the H(div) normal component tau(w) = w·n.
 */
class TraceLoadBC final : public BoundaryCondition {
public:
    TraceLoadBC(ExteriorBoundaryMeasure measure,
                FormExpr value,
                ScalarTraceOperator trace_operator =
                    ScalarTraceOperator::NormalComponent)
        : boundary_marker_(
              measure.physicalBoundaryMarker())
        , value_(std::move(value))
        , trace_operator_(trace_operator)
        , exterior_boundary_measure_(measure)
    {
        if (!value_.isValid()) {
            throw std::invalid_argument(
                "TraceLoadBC: invalid value expression");
        }
    }

    TraceLoadBC(int boundary_marker,
                FormExpr value,
                ScalarTraceOperator trace_operator = ScalarTraceOperator::NormalComponent)
        : boundary_marker_(boundary_marker)
        , value_(std::move(value))
        , trace_operator_(trace_operator)
    {
        if (boundary_marker_ < 0) {
            throw std::invalid_argument(
                "TraceLoadBC: boundary_marker must be >= 0");
        }
        if (!value_.isValid()) {
            throw std::invalid_argument(
                "TraceLoadBC: invalid value expression");
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return boundary_marker_; }
    [[nodiscard]] std::string_view label() const override { return "TraceLoadBC"; }

    void contributeToResidual(FormExpr& residual,
                              const FormExpr& /*u*/,
                              const FormExpr& v) const override
    {
        const auto integrand =
            value_ * applyScalarTrace(v, trace_operator_);
        residual =
            residual -
            (exterior_boundary_measure_.has_value()
                 ? integrand.dExteriorBoundary(
                       *exterior_boundary_measure_)
                 : integrand.ds(boundary_marker_));
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId /*field_id*/) const override
    {
        return {};
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId field_id, const systems::FESystem* /*system*/) const override
    {
        analysis::BoundaryConditionDescriptor d;
        d.primary_variable = analysis::VariableKey::field(field_id);
        d.boundary_marker = boundary_marker_;
        d.trace_kind = detail::traceKindFor(trace_operator_);
        d.enforcement_kind = analysis::EnforcementKind::WeakConsistent;
        d.anchors_constant_mode = false;
        d.anchors_rigid_body_translation = false;
        d.anchors_rigid_body_rotation = false;
        d.source = "TraceLoadBC on marker " + std::to_string(boundary_marker_);
        return {d};
    }

private:
    int boundary_marker_{-1};
    FormExpr value_{};
    ScalarTraceOperator trace_operator_{ScalarTraceOperator::NormalComponent};
    std::optional<ExteriorBoundaryMeasure>
        exterior_boundary_measure_{};
};

/**
 * @brief Weak Robin relation on a scalar trace:
 *        ∫ alpha * tau(u) * tau(v) ds - ∫ rhs * tau(v) ds
 *
 * The default trace operator is the H(div) normal component tau(w) = w·n.
 */
class TraceRobinBC final : public BoundaryCondition {
public:
    TraceRobinBC(ExteriorBoundaryMeasure measure,
                 FormExpr alpha,
                 FormExpr rhs,
                 ScalarTraceOperator trace_operator =
                     ScalarTraceOperator::NormalComponent)
        : boundary_marker_(
              measure.physicalBoundaryMarker())
        , alpha_(std::move(alpha))
        , rhs_(std::move(rhs))
        , trace_operator_(trace_operator)
        , exterior_boundary_measure_(measure)
    {
        if (!alpha_.isValid()) {
            throw std::invalid_argument(
                "TraceRobinBC: invalid alpha expression");
        }
        if (!rhs_.isValid()) {
            throw std::invalid_argument(
                "TraceRobinBC: invalid rhs expression");
        }
    }

    TraceRobinBC(int boundary_marker,
                 FormExpr alpha,
                 FormExpr rhs,
                 ScalarTraceOperator trace_operator = ScalarTraceOperator::NormalComponent)
        : boundary_marker_(boundary_marker)
        , alpha_(std::move(alpha))
        , rhs_(std::move(rhs))
        , trace_operator_(trace_operator)
    {
        if (boundary_marker_ < 0) {
            throw std::invalid_argument(
                "TraceRobinBC: boundary_marker must be >= 0");
        }
        if (!alpha_.isValid()) {
            throw std::invalid_argument(
                "TraceRobinBC: invalid alpha expression");
        }
        if (!rhs_.isValid()) {
            throw std::invalid_argument(
                "TraceRobinBC: invalid rhs expression");
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return boundary_marker_; }
    [[nodiscard]] std::string_view label() const override { return "TraceRobinBC"; }

    void contributeToResidual(FormExpr& residual,
                              const FormExpr& u,
                              const FormExpr& v) const override
    {
        const auto tau_u = applyScalarTrace(u, trace_operator_);
        const auto tau_v = applyScalarTrace(v, trace_operator_);
        const auto lhs = alpha_ * tau_u * tau_v;
        const auto rhs = rhs_ * tau_v;
        if (exterior_boundary_measure_.has_value()) {
            residual =
                residual +
                lhs.dExteriorBoundary(
                    *exterior_boundary_measure_) -
                rhs.dExteriorBoundary(
                    *exterior_boundary_measure_);
        } else {
            residual =
                residual + lhs.ds(boundary_marker_) -
                rhs.ds(boundary_marker_);
        }
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId /*field_id*/) const override
    {
        return {};
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId field_id, const systems::FESystem* /*system*/) const override
    {
        analysis::BoundaryConditionDescriptor d;
        d.primary_variable = analysis::VariableKey::field(field_id);
        d.boundary_marker = boundary_marker_;
        d.trace_kind = detail::traceKindFor(trace_operator_);
        d.enforcement_kind = analysis::EnforcementKind::WeakPenalty;
        d.anchors_constant_mode = true;
        d.anchors_rigid_body_translation = true;
        d.anchors_rigid_body_rotation = false;
        d.source = "TraceRobinBC on marker " + std::to_string(boundary_marker_);
        return {d};
    }

private:
    int boundary_marker_{-1};
    FormExpr alpha_{};
    FormExpr rhs_{};
    ScalarTraceOperator trace_operator_{ScalarTraceOperator::NormalComponent};
    std::optional<ExteriorBoundaryMeasure>
        exterior_boundary_measure_{};
};

/**
 * @brief Weak one-sided scalar-trace law:
 *        ∫ sign * penalty * positive_part(signed_gap) * tau(v) ds
 *
 * This is the first FE-side nonlinear inequality contract for scalar traces.
 * In semismooth mode it uses `max(0, signed_gap)`; in smooth mode it uses a
 * regularized `smoothMax`.
 */
class TraceInequalityBC final : public BoundaryCondition {
public:
    TraceInequalityBC(ExteriorBoundaryMeasure measure,
                      FormExpr bound,
                      FormExpr penalty,
                      TraceInequalityOptions options = {})
        : boundary_marker_(
              measure.physicalBoundaryMarker())
        , bound_(std::move(bound))
        , penalty_(std::move(penalty))
        , options_(std::move(options))
        , exterior_boundary_measure_(measure)
    {
        validate();
    }

    TraceInequalityBC(int boundary_marker,
                      FormExpr bound,
                      FormExpr penalty,
                      TraceInequalityOptions options = {})
        : boundary_marker_(boundary_marker)
        , bound_(std::move(bound))
        , penalty_(std::move(penalty))
        , options_(std::move(options))
    {
        if (boundary_marker_ < 0) {
            throw std::invalid_argument(
                "TraceInequalityBC: boundary_marker must be >= 0");
        }
        validate();
    }

    [[nodiscard]] int boundaryMarker() const override { return boundary_marker_; }
    [[nodiscard]] std::string_view label() const override { return "TraceInequalityBC"; }
    [[nodiscard]] bool allowsMarkerSharing() const override { return false; }

    void contributeToResidual(FormExpr& residual,
                              const FormExpr& u,
                              const FormExpr& v) const override
    {
        residual =
            exterior_boundary_measure_.has_value()
                ? applyTraceInequality(
                      std::move(residual),
                      u,
                      v,
                      *exterior_boundary_measure_,
                      bound_,
                      penalty_,
                      options_)
                : applyTraceInequality(
                      std::move(residual),
                      u,
                      v,
                      boundary_marker_,
                      bound_,
                      penalty_,
                      options_);
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId /*field_id*/) const override
    {
        return {};
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId field_id, const systems::FESystem* /*system*/) const override
    {
        analysis::BoundaryConditionDescriptor d;
        d.primary_variable = analysis::VariableKey::field(field_id);
        d.boundary_marker = boundary_marker_;
        d.trace_kind = detail::traceKindFor(options_.trace_operator);
        d.enforcement_kind = analysis::EnforcementKind::WeakInequality;
        d.anchors_constant_mode = true;
        d.anchors_rigid_body_translation = true;
        d.anchors_rigid_body_rotation = false;
        d.inequality_sense = detail::inequalitySenseFor(options_.sense);
        d.state_dependent_activation = true;
        d.source = "TraceInequalityBC on marker " + std::to_string(boundary_marker_);
        return {d};
    }

private:
    void validate() const
    {
        if (!bound_.isValid()) {
            throw std::invalid_argument("TraceInequalityBC: invalid bound expression");
        }
        if (!penalty_.isValid()) {
            throw std::invalid_argument("TraceInequalityBC: invalid penalty expression");
        }
        if (bound_.hasTest() || bound_.hasTrial()) {
            throw std::invalid_argument("TraceInequalityBC: bound must not contain test/trial functions");
        }
        if (penalty_.hasTest() || penalty_.hasTrial()) {
            throw std::invalid_argument("TraceInequalityBC: penalty must not contain test/trial functions");
        }
        if (options_.linearization == TraceInequalityLinearization::Smooth) {
            if (!options_.smoothing_epsilon.isValid()) {
                throw std::invalid_argument(
                    "TraceInequalityBC: smooth linearization requires a valid smoothing epsilon");
            }
            if (options_.smoothing_epsilon.hasTest() || options_.smoothing_epsilon.hasTrial()) {
                throw std::invalid_argument(
                    "TraceInequalityBC: smoothing epsilon must not contain test/trial functions");
            }
        }
    }

    int boundary_marker_{-1};
    FormExpr bound_{};
    FormExpr penalty_{};
    TraceInequalityOptions options_{};
    std::optional<ExteriorBoundaryMeasure>
        exterior_boundary_measure_{};
};

/**
 * @brief Weak trace load condition on an interface marker.
 *
 * Adds:
 *   -∫ g * tau_red(v) dI(marker)
 * where tau_red is a scalar trace on one side, or a jump/average reduction.
 */
class InterfaceTraceLoadBC final : public BoundaryCondition {
public:
    InterfaceTraceLoadBC(int interface_marker,
                         FormExpr value,
                         ScalarTraceOperator trace_operator = ScalarTraceOperator::NormalComponent,
                         InterfaceTraceReduction reduction = InterfaceTraceReduction::Minus)
        : interface_marker_(interface_marker)
        , value_(std::move(value))
        , trace_operator_(trace_operator)
        , reduction_(reduction)
    {
        if (interface_marker_ < 0) {
            throw std::invalid_argument("InterfaceTraceLoadBC: interface_marker must be >= 0");
        }
        if (!value_.isValid()) {
            throw std::invalid_argument("InterfaceTraceLoadBC: invalid value expression");
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return -1; }
    [[nodiscard]] std::string_view label() const override { return "InterfaceTraceLoadBC"; }
    [[nodiscard]] int interfaceMarker() const override { return interface_marker_; }

    void contributeToResidual(FormExpr& residual,
                              const FormExpr& /*u*/,
                              const FormExpr& v) const override
    {
        residual = residual - (value_ * applyInterfaceScalarTrace(v, trace_operator_, reduction_))
                                  .dI(interface_marker_);
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId /*field_id*/) const override
    {
        return {};
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId field_id, const systems::FESystem* /*system*/) const override
    {
        analysis::BoundaryConditionDescriptor d;
        d.primary_variable = analysis::VariableKey::field(field_id);
        d.domain = analysis::DomainKind::InterfaceFace;
        d.interface_marker = interface_marker_;
        d.trace_kind = detail::traceKindFor(trace_operator_);
        d.enforcement_kind = analysis::EnforcementKind::WeakConsistent;
        d.anchors_constant_mode = false;
        d.anchors_rigid_body_translation = false;
        d.anchors_rigid_body_rotation = false;
        d.source = detail::sourceFor("InterfaceTraceLoadBC",
                                     analysis::DomainKind::InterfaceFace,
                                     interface_marker_);
        return {d};
    }

private:
    int interface_marker_{-1};
    FormExpr value_{};
    ScalarTraceOperator trace_operator_{ScalarTraceOperator::NormalComponent};
    InterfaceTraceReduction reduction_{InterfaceTraceReduction::Minus};
};

/**
 * @brief Weak Robin relation on an interface scalar trace.
 *
 * Adds:
 *   ∫ alpha * tau_red(u) * tau_red(v) dI(marker) - ∫ rhs * tau_red(v) dI(marker)
 */
class InterfaceTraceRobinBC final : public BoundaryCondition {
public:
    InterfaceTraceRobinBC(int interface_marker,
                          FormExpr alpha,
                          FormExpr rhs,
                          ScalarTraceOperator trace_operator = ScalarTraceOperator::NormalComponent,
                          InterfaceTraceReduction reduction = InterfaceTraceReduction::Minus)
        : interface_marker_(interface_marker)
        , alpha_(std::move(alpha))
        , rhs_(std::move(rhs))
        , trace_operator_(trace_operator)
        , reduction_(reduction)
    {
        if (interface_marker_ < 0) {
            throw std::invalid_argument("InterfaceTraceRobinBC: interface_marker must be >= 0");
        }
        if (!alpha_.isValid()) {
            throw std::invalid_argument("InterfaceTraceRobinBC: invalid alpha expression");
        }
        if (!rhs_.isValid()) {
            throw std::invalid_argument("InterfaceTraceRobinBC: invalid rhs expression");
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return -1; }
    [[nodiscard]] std::string_view label() const override { return "InterfaceTraceRobinBC"; }
    [[nodiscard]] int interfaceMarker() const override { return interface_marker_; }

    void contributeToResidual(FormExpr& residual,
                              const FormExpr& u,
                              const FormExpr& v) const override
    {
        const auto tau_u = applyInterfaceScalarTrace(u, trace_operator_, reduction_);
        const auto tau_v = applyInterfaceScalarTrace(v, trace_operator_, reduction_);
        residual = residual + (alpha_ * tau_u * tau_v).dI(interface_marker_)
                            - (rhs_ * tau_v).dI(interface_marker_);
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId /*field_id*/) const override
    {
        return {};
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId field_id, const systems::FESystem* /*system*/) const override
    {
        analysis::BoundaryConditionDescriptor d;
        d.primary_variable = analysis::VariableKey::field(field_id);
        d.domain = analysis::DomainKind::InterfaceFace;
        d.interface_marker = interface_marker_;
        d.trace_kind = detail::traceKindFor(trace_operator_);
        d.enforcement_kind = analysis::EnforcementKind::WeakPenalty;
        d.anchors_constant_mode = true;
        d.anchors_rigid_body_translation = true;
        d.anchors_rigid_body_rotation = false;
        d.source = detail::sourceFor("InterfaceTraceRobinBC",
                                     analysis::DomainKind::InterfaceFace,
                                     interface_marker_);
        return {d};
    }

private:
    int interface_marker_{-1};
    FormExpr alpha_{};
    FormExpr rhs_{};
    ScalarTraceOperator trace_operator_{ScalarTraceOperator::NormalComponent};
    InterfaceTraceReduction reduction_{InterfaceTraceReduction::Minus};
};

/**
 * @brief Weak continuity penalty on an interface scalar trace jump.
 *
 * Adds:
 *   ∫ alpha * [[tau(u)]] * [[tau(v)]] dI(marker) - ∫ rhs * [[tau(v)]] dI(marker)
 *
 * For `ScalarTraceOperator::NormalComponent`, the jump uses outward normals on
 * each side, i.e. `u^-·n^- + u^+·n^+`.
 */
class InterfaceTraceJumpPenaltyBC final : public BoundaryCondition {
public:
    InterfaceTraceJumpPenaltyBC(int interface_marker,
                                FormExpr alpha,
                                FormExpr rhs,
                                ScalarTraceOperator trace_operator = ScalarTraceOperator::NormalComponent)
        : interface_marker_(interface_marker)
        , alpha_(std::move(alpha))
        , rhs_(std::move(rhs))
        , trace_operator_(trace_operator)
    {
        if (interface_marker_ < 0) {
            throw std::invalid_argument("InterfaceTraceJumpPenaltyBC: interface_marker must be >= 0");
        }
        if (!alpha_.isValid()) {
            throw std::invalid_argument("InterfaceTraceJumpPenaltyBC: invalid alpha expression");
        }
        if (!rhs_.isValid()) {
            throw std::invalid_argument("InterfaceTraceJumpPenaltyBC: invalid rhs expression");
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return -1; }
    [[nodiscard]] std::string_view label() const override { return "InterfaceTraceJumpPenaltyBC"; }
    [[nodiscard]] int interfaceMarker() const override { return interface_marker_; }

    void contributeToResidual(FormExpr& residual,
                              const FormExpr& u,
                              const FormExpr& v) const override
    {
        const auto jump_u =
            applyInterfaceScalarTrace(u, trace_operator_, InterfaceTraceReduction::Jump);
        const auto jump_v =
            applyInterfaceScalarTrace(v, trace_operator_, InterfaceTraceReduction::Jump);
        residual = residual + (alpha_ * jump_u * jump_v).dI(interface_marker_)
                            - (rhs_ * jump_v).dI(interface_marker_);
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId /*field_id*/) const override
    {
        return {};
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId field_id, const systems::FESystem* /*system*/) const override
    {
        analysis::BoundaryConditionDescriptor d;
        d.primary_variable = analysis::VariableKey::field(field_id);
        d.domain = analysis::DomainKind::InterfaceFace;
        d.interface_marker = interface_marker_;
        d.trace_kind = detail::traceKindFor(trace_operator_);
        d.enforcement_kind = analysis::EnforcementKind::WeakPenalty;
        d.anchors_constant_mode = true;
        d.anchors_rigid_body_translation = true;
        d.anchors_rigid_body_rotation = false;
        d.source = detail::sourceFor("InterfaceTraceJumpPenaltyBC",
                                     analysis::DomainKind::InterfaceFace,
                                     interface_marker_);
        return {d};
    }

private:
    int interface_marker_{-1};
    FormExpr alpha_{};
    FormExpr rhs_{};
    ScalarTraceOperator trace_operator_{ScalarTraceOperator::NormalComponent};
};

[[nodiscard]] inline std::unique_ptr<BoundaryCondition>
makeNormalTraceEssentialBC(int boundary_marker, FormExpr value)
{
    return std::make_unique<NormalTraceEssentialBC>(boundary_marker, std::move(value));
}

[[nodiscard]] inline std::unique_ptr<BoundaryCondition>
makeNormalEssentialBC(int boundary_marker, FormExpr value = FormExpr::constant(0.0))
{
    return std::make_unique<NormalEssentialBC>(boundary_marker, std::move(value));
}

[[nodiscard]] inline std::unique_ptr<BoundaryCondition>
makeTraceLoadBC(int boundary_marker,
                FormExpr value,
                ScalarTraceOperator trace_operator = ScalarTraceOperator::NormalComponent)
{
    return std::make_unique<TraceLoadBC>(
        boundary_marker, std::move(value), trace_operator);
}

[[nodiscard]] inline std::unique_ptr<BoundaryCondition>
makeTraceLoadBC(ExteriorBoundaryMeasure measure,
                FormExpr value,
                ScalarTraceOperator trace_operator =
                    ScalarTraceOperator::NormalComponent)
{
    return std::make_unique<TraceLoadBC>(
        measure, std::move(value), trace_operator);
}

[[nodiscard]] inline std::unique_ptr<BoundaryCondition>
makeTraceRobinBC(int boundary_marker,
                 FormExpr alpha,
                 FormExpr rhs,
                 ScalarTraceOperator trace_operator = ScalarTraceOperator::NormalComponent)
{
    return std::make_unique<TraceRobinBC>(
        boundary_marker, std::move(alpha), std::move(rhs), trace_operator);
}

[[nodiscard]] inline std::unique_ptr<BoundaryCondition>
makeTraceRobinBC(ExteriorBoundaryMeasure measure,
                 FormExpr alpha,
                 FormExpr rhs,
                 ScalarTraceOperator trace_operator =
                     ScalarTraceOperator::NormalComponent)
{
    return std::make_unique<TraceRobinBC>(
        measure,
        std::move(alpha),
        std::move(rhs),
        trace_operator);
}

[[nodiscard]] inline std::unique_ptr<BoundaryCondition>
makeTraceInequalityBC(int boundary_marker,
                      FormExpr bound,
                      FormExpr penalty,
                      TraceInequalityOptions options = {})
{
    return std::make_unique<TraceInequalityBC>(
        boundary_marker, std::move(bound), std::move(penalty), std::move(options));
}

[[nodiscard]] inline std::unique_ptr<BoundaryCondition>
makeTraceInequalityBC(ExteriorBoundaryMeasure measure,
                      FormExpr bound,
                      FormExpr penalty,
                      TraceInequalityOptions options = {})
{
    return std::make_unique<TraceInequalityBC>(
        measure,
        std::move(bound),
        std::move(penalty),
        std::move(options));
}

[[nodiscard]] inline std::unique_ptr<BoundaryCondition>
makeInterfaceTraceLoadBC(int interface_marker,
                         FormExpr value,
                         ScalarTraceOperator trace_operator = ScalarTraceOperator::NormalComponent,
                         InterfaceTraceReduction reduction = InterfaceTraceReduction::Minus)
{
    return std::make_unique<InterfaceTraceLoadBC>(
        interface_marker, std::move(value), trace_operator, reduction);
}

[[nodiscard]] inline std::unique_ptr<BoundaryCondition>
makeInterfaceTraceRobinBC(int interface_marker,
                          FormExpr alpha,
                          FormExpr rhs,
                          ScalarTraceOperator trace_operator = ScalarTraceOperator::NormalComponent,
                          InterfaceTraceReduction reduction = InterfaceTraceReduction::Minus)
{
    return std::make_unique<InterfaceTraceRobinBC>(
        interface_marker, std::move(alpha), std::move(rhs), trace_operator, reduction);
}

[[nodiscard]] inline std::unique_ptr<BoundaryCondition>
makeInterfaceTraceJumpPenaltyBC(int interface_marker,
                                FormExpr alpha,
                                FormExpr rhs,
                                ScalarTraceOperator trace_operator = ScalarTraceOperator::NormalComponent)
{
    return std::make_unique<InterfaceTraceJumpPenaltyBC>(
        interface_marker, std::move(alpha), std::move(rhs), trace_operator);
}

} // namespace bc
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_STANDARD_BCS_H
