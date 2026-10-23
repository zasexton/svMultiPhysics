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
 * The formulation also declares its functionals here, as measured 0-forms
 * (FunctionalForms.h):
 *
 *   F.functional("kinetic_energy", FunctionalKind::Stored, (0.5 * rho * inner(u, u)).dx());
 *
 * and installs its residual-work channels (exact additive groups of the
 * residual, each its own operator) and its opt-in diagnostic operators:
 *
 *   F.workChannel("work_pressure", {{.test_field = u, .form = pressure_work},
 *                                   {.test_field = p, .form = continuity_work, .extra_trial_fields = {u}}});
 *   F.diagnostic("set", "equations_diagnostic_x", {u}, form);
 */

#include "Core/Types.h"
#include "Forms/IntegrationDomain.h"
#include "Systems/DiagnosticRegistry.h"
#include "Systems/FormsInstaller.h"
#include "Systems/FunctionalForms.h"
#include "Systems/OperatorRegistry.h"

#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

class FESystem;
struct ALEBinding;

/// How a separate kernel group differs from the formulation's options.
struct FormulationGroup {
    std::string source_component_tag{};          ///< set when non-empty
    std::vector<FieldId> extra_trial_fields{};    ///< appended when missing, after the ALE configuration
    const ALEBinding* ale{nullptr};               ///< configureInstallOptions, first
    bool drop_generated_boundary_nitsche_trace_requests{false};
};

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

    FormulationBuilder& symbolicTangent(bool enabled = true);
    FormulationBuilder& recordConstitutiveLaw(analysis::ConstitutiveLawMetadata law);
    FormulationBuilder& configure(const ALEBinding& ale);

    /// Moves @p bindings into trace-certificate requests on @p volume_interface_marker.
    FormulationBuilder& certifyGeneratedBoundaryNitscheTraces(
        std::vector<forms::bc::GeneratedBoundaryNitscheTraceFormBinding>& bindings,
        int volume_interface_marker,
        Real minimum_symmetric_energy_ratio);

    /// Registers the generated marker of a contact line or of the generated
    /// part of a boundary with the system (nothing for a marker < 0) and
    /// returns the domain.
    forms::ContactLineDomain registered(forms::ContactLineDomain domain);
    forms::BoundaryDomain registered(forms::BoundaryDomain domain);

    /// Declares the measured functional @p form under @p name
    /// (declareFunctionalForm), owned by @p owner, or by operatorTag() when
    /// @p owner is empty.  Installs nothing.
    FormulationBuilder& functional(std::string name,
                                   FunctionalKind kind,
                                   forms::FormExpr form,
                                   std::string owner = {});

    [[nodiscard]] FormInstallOptions& options() noexcept { return options_; }
    [[nodiscard]] const FormInstallOptions& options() const noexcept { return options_; }
    [[nodiscard]] const OperatorTag& operatorTag() const noexcept { return operator_tag_; }
    [[nodiscard]] FESystem& system() const noexcept { return *system_; }

    /// The options of a group: a copy of options() changed as @p group says.
    [[nodiscard]] FormInstallOptions groupOptions(const FormulationGroup& group) const;

    /// installFormulation with options().
    CoupledResidualKernels install(std::initializer_list<FieldId> fields, const forms::FormExpr& residual);

    /// installFormulation with groupOptions(@p group).
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

    /// The diagnostic sets and operators of the system (diagnosticRegistry).
    [[nodiscard]] DiagnosticRegistry& diagnostics();

    /// Installs @p form as the operator @p operator_tag of the declared
    /// diagnostic @p set with options() (DiagnosticRegistry::install), usually
    /// after the production operators.
    CoupledResidualKernels diagnostic(std::string_view set,
                                      const OperatorTag& operator_tag,
                                      std::initializer_list<FieldId> fields,
                                      const forms::FormExpr& form,
                                      const DiagnosticOperatorOptions& options = {});

private:
    FESystem* system_{nullptr};
    OperatorTag operator_tag_{};
    FormInstallOptions options_{};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_FORMULATIONBUILDER_H
