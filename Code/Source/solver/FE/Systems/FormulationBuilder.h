/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_FORMULATIONBUILDER_H
#define SVMP_FE_SYSTEMS_FORMULATIONBUILDER_H

/**
 * @file FormulationBuilder.h
 * @brief Install options of one formulation, composed in call order
 *
 * A module configures one set of install options (symbolic tangent,
 * constitutive laws, extra trial fields, ALE coupling, certified
 * generated-boundary Nitsche traces) and installs its residual and its
 * separate kernel groups on one operator.  Every call takes effect
 * immediately and in call order, so the options and the installation
 * sequence are exactly those a module would write out by hand with
 * installFormulation:
 *
 *   FormulationBuilder F(system, "equations", base);
 *   F.symbolicTangent();
 *   F.installGroup({d}, mesh_rows, {.ale = &ale});              // copy of the options so far
 *   F.recordConstitutiveLaw(analysis::dynamicViscosityMetadata(u, mu)).configure(ale);
 *   F.install({u, p}, M + C);                                    // the options themselves
 *   F.installGroup({p}, pspg, {.source_component_tag = "pspg", .extra_trial_fields = {u}});
 *
 * The formulation also declares its functionals here, as quantities stated
 * by measured 0-forms (FunctionalForms.h):
 *
 *   const auto E = F.quantity("kinetic_energy", FunctionalKind::Stored, (0.5 * rho * inner(u, u)).dx());
 *   F.record(E);                                  // a column of the scalar history
 *
 * and installs its residual-work channels (exact additive groups of the
 * residual, each its own operator) and its opt-in diagnostic operators:
 *
 *   F.workChannel("work_pressure", {{.test_field = u, .form = pressure_work},
 *                                   {.test_field = p, .form = continuity_work, .extra_trial_fields = {u}}});
 *   F.diagnostic("set", "equations_diagnostic_x", {u}, form);
 */

#include "Core/LogRecord.h"
#include "Core/Types.h"
#include "Forms/IntegrationDomain.h"
#include "Systems/DiagnosticRegistry.h"
#include "Systems/FormsInstaller.h"
#include "Systems/FunctionalForms.h"
#include "Systems/MultiplierLowering.h"
#include "Systems/OperatorRegistry.h"

#include <initializer_list>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

class FESystem;
struct ALEBinding;

/// Whether construction ensures an operator exists or strictly adds a new one.
enum class OperatorCreation { Ensure, Add };

/// How a separate kernel group differs from the formulation's options.
struct FormulationGroup {
    std::string source_component_tag{};          ///< set when non-empty
    std::vector<FieldId> extra_trial_fields{};    ///< appended when missing, after the ALE configuration
    const ALEBinding* ale{nullptr};               ///< configureInstallOptions, first
    bool drop_generated_boundary_nitsche_trace_requests{false};
};

/// The fields of an install that a residual implies (inferInstallFields).
struct InferredInstallFields {
    std::vector<FieldId> test_fields{};         ///< the fields of the residual's test functions
    std::vector<FieldId> extra_trial_fields{};  ///< the other unknowns the residual holds
};

/**
 * @brief The fields of an install of @p residual on @p system
 *
 * The test fields are the fields of the test functions in @p residual; the
 * extra trial fields are the other fields whose StateField the residual
 * holds and that are unknowns of the system (in the unknown vector), so
 * that their Jacobian columns are assembled.  Both lists are in
 * registration order, the DOF order.  Throws InvalidArgumentException for a
 * test function without a field (FormExpr::testFunction(space)) and for a
 * residual without a test function.
 */
[[nodiscard]] InferredInstallFields inferInstallFields(const FESystem& system, const forms::FormExpr& residual);

/// One part of a residual-work channel: the form of one test field.
struct WorkChannelPart {
    FieldId test_field{INVALID_FIELD_ID};
    forms::FormExpr form{};                       ///< the part is skipped when empty
    std::vector<FieldId> extra_trial_fields{};    ///< appended when missing
    bool keep_generated_boundary_nitsche_trace_requests{false};
};

class FormulationBuilder {
public:
    /// Adds @p operator_tag to the system when it is missing.
    FormulationBuilder(FESystem& system, OperatorTag operator_tag, FormInstallOptions base);

    /// Add uses FESystem::addOperator unconditionally, including its duplicate rejection.
    FormulationBuilder(FESystem& system,
                       OperatorTag operator_tag,
                       FormInstallOptions base,
                       OperatorCreation creation);

    FormulationBuilder& symbolicTangent(bool enabled = true);

    /// Asks for a lowering of the system's global multipliers
    /// (FESystem::requestMultiplierLowering, requested by operatorTag()).
    FormulationBuilder& multiplierLowering(MultiplierLowering lowering);
    FormulationBuilder& recordConstitutiveLaw(analysis::ConstitutiveLawMetadata law);
    FormulationBuilder& configure(const ALEBinding& ale);

    /// Keeps the geometry of later installs constant on a moving domain
    /// (FormInstallOptions::moving_domain_geometry = false).
    FormulationBuilder& geometryConstant();

    /// Moves @p bindings into trace-certificate requests on @p volume_interface_marker.
    FormulationBuilder& certifyGeneratedBoundaryNitscheTraces(
        std::vector<forms::bc::GeneratedBoundaryNitscheTraceFormBinding>& bindings,
        int volume_interface_marker,
        Real minimum_symmetric_energy_ratio);

    /// Registers the generated marker of a contact line with the system
    /// (nothing for a marker < 0) and returns the domain.  The generated
    /// active parts of boundaries need no call: install() and installGroup()
    /// register those their residual integrates over.
    forms::ContactLineDomain registered(forms::ContactLineDomain domain);

    /// Declares the quantity @p name stated by the measured 0-form @p form
    /// (declareFunctionalForm), owned by @p owner, or by operatorTag() when
    /// @p owner is empty, and returns its handle.  Installs nothing.
    Quantity quantity(std::string name,
                      FunctionalKind kind,
                      forms::FormExpr form,
                      std::string owner = {});

    /// Records @p quantity, or the scalar global field @p field, in the
    /// system's scalar history (post::record, PostProcessing/ScalarHistory.h).
    /// Installs nothing.
    FormulationBuilder& record(const Quantity& quantity);
    FormulationBuilder& record(FieldId field);

    /// Declares the wall functional @p declaration (declareWallFunctionalForm),
    /// owned by operatorTag() when its owner is empty.  Installs nothing.
    FormulationBuilder& wallFunctional(WallFunctionalFormDeclaration declaration);

    [[nodiscard]] FormInstallOptions& options() noexcept { return options_; }
    [[nodiscard]] const FormInstallOptions& options() const noexcept { return options_; }
    [[nodiscard]] const OperatorTag& operatorTag() const noexcept { return operator_tag_; }
    [[nodiscard]] FESystem& system() const noexcept { return *system_; }

    /// The options of a group: a copy of options() changed as @p group says.
    [[nodiscard]] FormInstallOptions groupOptions(const FormulationGroup& group) const;

    /// installFormulation with options(), after registering the generated
    /// active boundaries the residual integrates over.
    CoupledResidualKernels install(std::span<const FieldId> fields, const forms::FormExpr& residual);
    CoupledResidualKernels install(std::initializer_list<FieldId> fields, const forms::FormExpr& residual);

    /**
     * @brief installFormulation of @p residual with the fields it implies
     *
     * install(inferInstallFields(...).test_fields, residual) with a copy of
     * options() whose extra trial fields gain, when missing and in order,
     * the other unknowns the residual holds, after registering the
     * generated active boundaries the residual integrates over.  A residual
     * written in the unknowns' symbols needs no field list.
     */
    CoupledResidualKernels residual(const forms::FormExpr& residual);

    /// installFormulation with groupOptions(@p group), after registering the
    /// generated active boundaries the residual integrates over.
    CoupledResidualKernels installGroup(std::span<const FieldId> fields,
                                        const forms::FormExpr& residual,
                                        const FormulationGroup& group);
    CoupledResidualKernels installGroup(std::initializer_list<FieldId> fields,
                                        const forms::FormExpr& residual,
                                        const FormulationGroup& group);

    /**
     * @brief Installs one residual-work channel as its own operator.
     *
     * The parts with a form are installed in order on the operator
     * "<operatorTag()>_<suffix>", each with a copy of options() that is
     * outside the problem analysis, names that operator as its source
     * component, appends the part's extra trial fields when missing and
     * drops the generated-boundary Nitsche trace requests unless the part
     * keeps them.  A work channel is an exact additive group of the
     * residual, so its operator evaluates that group alone.
     *
     * @return the operator tag, or an empty tag when no part has a form
     */
    OperatorTag workChannel(std::string_view suffix, std::initializer_list<WorkChannelPart> parts);

    /**
     * @brief Records @p term under the work-channel label @p label and
     * returns it unchanged
     *
     * The term holds the test function of one field.  A residual built as
     * R = sumOf(R, F.tag(label, term)) is the residual without the tags, and
     * tagged() derives the channel from the same terms.
     */
    [[nodiscard]] forms::FormExpr tag(std::string_view label, forms::FormExpr term);

    /// The terms tagged @p label in the row of @p test_field, summed in tag
    /// order with sumOf (the left fold of a residual built term by term), or
    /// an empty form when there is none.
    [[nodiscard]] forms::FormExpr tagged(std::string_view label, FieldId test_field) const;

    /// The diagnostic sets and operators of the system (diagnosticRegistry).
    [[nodiscard]] DiagnosticRegistry& diagnostics();

    /// Prints the log records a formulation declares for what it installs
    /// (LogRecord), at the point where it states it.
    FormulationBuilder& log(const LogRecord& record);
    FormulationBuilder& log(std::span<const LogRecord> records);

    /// Installs @p form as the operator @p operator_tag of the declared
    /// diagnostic @p set with options() (DiagnosticRegistry::install), usually
    /// after the production operators.
    CoupledResidualKernels diagnostic(std::string_view set,
                                      const OperatorTag& operator_tag,
                                      std::initializer_list<FieldId> fields,
                                      const forms::FormExpr& form,
                                      const DiagnosticOperatorOptions& options = {});

private:
    /// Registers the generated active boundary of each generated exterior
    /// measure of @p residual with the system.
    void registerGeneratedBoundaries(const forms::FormExpr& residual);

    struct TaggedTerm {
        std::string label;
        FieldId test_field{INVALID_FIELD_ID};
        forms::FormExpr term;
    };

    FESystem* system_{nullptr};
    OperatorTag operator_tag_{};
    FormInstallOptions options_{};
    std::vector<TaggedTerm> tagged_terms_{};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_FORMULATIONBUILDER_H
