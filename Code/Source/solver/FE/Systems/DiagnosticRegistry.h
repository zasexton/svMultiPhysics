/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_DIAGNOSTICREGISTRY_H
#define SVMP_FE_SYSTEMS_DIAGNOSTICREGISTRY_H

/**
 * @file DiagnosticRegistry.h
 * @brief Opt-in diagnostic operators of a formulation, by set
 *
 * A formulation declares its diagnostic sets and installs the operators of
 * a set that is switched on.  A set is switched on by the first of its
 * environment switches that is set and not empty: false for 0, false, no
 * or off (in any case), true otherwise; it is off when none is set.  Each
 * diagnostic operator is installed under its own operator tag with a copy
 * of the formulation's install options without the generated-boundary
 * Nitsche trace requests (the trace certificate belongs to the production
 * residual), and is recorded under its set, so that a solver can look the
 * operators of a set up by name.  A system's registry is one of its
 * attachments (diagnosticRegistry); FormulationBuilder::diagnostic installs
 * through it.
 */

#include "Core/Types.h"
#include "Systems/FormsInstaller.h"
#include "Systems/OperatorRegistry.h"

#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
class FormExpr;
} // namespace forms

namespace systems {

class FESystem;

/// How a diagnostic operator's install options differ from the formulation's.
struct DiagnosticOperatorOptions {
    std::string source_component_tag{};          ///< set when non-empty
    bool clear_extra_trial_fields{false};         ///< before extra_trial_fields are appended
    std::vector<FieldId> extra_trial_fields{};   ///< appended when missing
};

class DiagnosticRegistry {
public:
    /// Declares (or redeclares, forgetting its operators) a set.
    void declareSet(std::string set, std::vector<std::string> environment_switches);

    /// Whether @p set is switched on now; throws std::invalid_argument for an undeclared set.
    [[nodiscard]] bool enabled(std::string_view set) const;

    /// The install options of a diagnostic operator.
    [[nodiscard]] static FormInstallOptions operatorOptions(const FormInstallOptions& formulation_options,
                                                            const DiagnosticOperatorOptions& options);

    /// installFormulation of @p form as the operator @p operator_tag of @p set,
    /// with operatorOptions(@p formulation_options, @p options).
    CoupledResidualKernels install(FESystem& system,
                                   std::string_view set,
                                   const OperatorTag& operator_tag,
                                   std::initializer_list<FieldId> fields,
                                   const forms::FormExpr& form,
                                   const FormInstallOptions& formulation_options,
                                   const DiagnosticOperatorOptions& options = {});

    /// The operators installed for @p set, in installation order.
    [[nodiscard]] std::vector<OperatorTag> operators(std::string_view set) const;

private:
    struct Set {
        std::string name{};
        std::vector<std::string> environment_switches{};
        std::vector<OperatorTag> operators{};
    };

    [[nodiscard]] const Set& find(std::string_view set) const;
    [[nodiscard]] Set& find(std::string_view set);

    std::vector<Set> sets_{};
};

/// The diagnostic registry of @p system: an attachment, created on first use.
[[nodiscard]] DiagnosticRegistry& diagnosticRegistry(FESystem& system);

/// The diagnostic registry of @p system, or null when none was created.
[[nodiscard]] const DiagnosticRegistry* findDiagnosticRegistry(const FESystem& system) noexcept;

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_DIAGNOSTICREGISTRY_H
