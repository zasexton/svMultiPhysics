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
 *
 * A consumer of diagnostic operators (a solver extension such as a
 * certificate or an observer) holds DiagnosticOperatorHandles instead of
 * operator names: the formulation records a typed declaration of the
 * operators of a set (declare), and the consumer finds it by its type
 * (findDeclaration), so the names stay with the formulation.
 */

#include "Core/Types.h"
#include "Systems/FormsInstaller.h"
#include "Systems/OperatorRegistry.h"

#include <any>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
class FormExpr;
} // namespace forms

namespace systems {

class FESystem;

/// A diagnostic operator of a set, held by a consumer instead of its name.
/// The formulation may declare a handle whose operator it installs only on
/// request; installedOn tells whether it is installed on a system.
struct DiagnosticOperatorHandle {
    std::string set{};
    OperatorTag tag{};

    /// Whether the operator is installed on @p system (false for an empty handle).
    [[nodiscard]] bool installedOn(const FESystem& system) const;

    friend bool operator==(const DiagnosticOperatorHandle&, const DiagnosticOperatorHandle&) = default;
};

/// How a diagnostic operator's install options differ from the formulation's.
struct DiagnosticOperatorOptions {
    std::string source_component_tag{};          ///< set when non-empty
    bool clear_extra_trial_fields{false};         ///< before extra_trial_fields are appended
    std::vector<FieldId> extra_trial_fields{};   ///< appended when missing
};

class DiagnosticRegistry {
public:
    /// Declares @p set when it is missing.  The formulations that share a
    /// system share its registry (for example the two phases of a two-fluid
    /// formulation), so a set may be declared again: with the same
    /// environment switches this keeps the set and its operators; with
    /// others it throws std::invalid_argument.
    void declareSet(std::string set, std::vector<std::string> environment_switches);

    /// Whether @p set is switched on now; throws std::invalid_argument for an undeclared set.
    [[nodiscard]] bool enabled(std::string_view set) const;

    /// The install options of a diagnostic operator: the formulation's, as
    /// a measurement outside the problem analysis
    /// (contributes_to_problem_analysis = false), without trace requests.
    [[nodiscard]] static FormInstallOptions operatorOptions(const FormInstallOptions& formulation_options,
                                                            const DiagnosticOperatorOptions& options);

    /// installFormulation of @p form as the operator @p operator_tag of @p set,
    /// with operatorOptions(@p formulation_options, @p options).  An operator
    /// installed again (by another formulation) is recorded once.
    CoupledResidualKernels install(FESystem& system,
                                   std::string_view set,
                                   const OperatorTag& operator_tag,
                                   std::initializer_list<FieldId> fields,
                                   const forms::FormExpr& form,
                                   const FormInstallOptions& formulation_options,
                                   const DiagnosticOperatorOptions& options = {});

    /// The operators installed for @p set, in order of first installation.
    [[nodiscard]] std::vector<OperatorTag> operators(std::string_view set) const;

    /// The handle of the operator @p operator_tag of the declared @p set,
    /// whether or not it is installed; throws std::invalid_argument for an
    /// undeclared set.
    [[nodiscard]] DiagnosticOperatorHandle handle(std::string_view set, OperatorTag operator_tag) const;

    /// Records @p declaration, a typed view of operators of the declared
    /// @p set for the consumers that read them.  A declaration equal to one
    /// already recorded in the set is kept once (the formulations that share
    /// a system may declare it each); an unequal one of the same type is
    /// recorded after it.
    template <class Declaration>
    void declare(std::string_view set, Declaration declaration)
    {
        auto& entry = find(set);
        for (const auto& recorded : entry.declarations) {
            const auto* same_type = std::any_cast<Declaration>(&recorded);
            if (same_type != nullptr && *same_type == declaration) {
                return;
            }
        }
        entry.declarations.emplace_back(std::move(declaration));
    }

    /// The first declaration of type Declaration, in the order of the sets
    /// and of recording, or null when none is recorded.
    template <class Declaration>
    [[nodiscard]] const Declaration* findDeclaration() const
    {
        for (const auto& entry : sets_) {
            for (const auto& recorded : entry.declarations) {
                if (const auto* declaration = std::any_cast<Declaration>(&recorded)) {
                    return declaration;
                }
            }
        }
        return nullptr;
    }

private:
    struct Set {
        std::string name{};
        std::vector<std::string> environment_switches{};
        std::vector<OperatorTag> operators{};
        std::vector<std::any> declarations{};
    };

    [[nodiscard]] const Set& find(std::string_view set) const;
    [[nodiscard]] Set& find(std::string_view set);

    std::vector<Set> sets_{};
};

/// The diagnostic registry of @p system: an attachment, created on first use.
[[nodiscard]] DiagnosticRegistry& diagnosticRegistry(FESystem& system);

/// The diagnostic registry of @p system, or null when none was created.
[[nodiscard]] const DiagnosticRegistry* findDiagnosticRegistry(const FESystem& system) noexcept;

/// The first declaration of type Declaration in the diagnostic registry of
/// @p system (DiagnosticRegistry::findDeclaration), or null.
template <class Declaration>
[[nodiscard]] const Declaration* findDiagnosticDeclaration(const FESystem& system) noexcept
{
    const auto* registry = findDiagnosticRegistry(system);
    return registry != nullptr ? registry->findDeclaration<Declaration>() : nullptr;
}

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_DIAGNOSTICREGISTRY_H
