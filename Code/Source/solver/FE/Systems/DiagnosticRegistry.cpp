/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/DiagnosticRegistry.h"

#include "Forms/FormExpr.h"
#include "Systems/FESystem.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace svmp {
namespace FE {
namespace systems {

bool DiagnosticOperatorHandle::installedOn(const FESystem& system) const
{
    return !tag.empty() && system.hasOperator(tag);
}

void DiagnosticRegistry::declareSet(std::string set, std::vector<std::string> environment_switches)
{
    declareSet(std::move(set), std::move(environment_switches), environment::EnvironmentNameRegistry::solver());
}

void DiagnosticRegistry::declareSet(std::string set, std::vector<std::string> environment_switches,
                                    const environment::EnvironmentNameRegistry& names)
{
    for (const auto& existing : sets_) {
        if (existing.name == set) {
            if (existing.environment_switches != environment_switches || existing.names != &names) {
                throw std::invalid_argument("DiagnosticRegistry: diagnostic set '" + set +
                                            "' is already declared with other environment switches");
            }
            return;
        }
    }
    sets_.push_back(Set{.name = std::move(set),
                        .environment_switches = std::move(environment_switches),
                        .names = &names,
                        .operators = {}});
}

bool DiagnosticRegistry::enabled(std::string_view set) const
{
    const auto& declaration = find(set);
    return declaration.names->firstEnabled(declaration.environment_switches);
}

bool DiagnosticRegistry::enabled(std::string_view set,
                                 const environment::EnvironmentNameRegistry& names) const
{
    return names.firstEnabled(find(set).environment_switches);
}

FormInstallOptions DiagnosticRegistry::operatorOptions(const FormInstallOptions& formulation_options,
                                                       const DiagnosticOperatorOptions& options)
{
    FormInstallOptions install = formulation_options;
    // A diagnostic operator is a measurement: it does not take part in the
    // analysis of the solved problem (gauges, well-posedness).
    install.contributes_to_problem_analysis = false;
    install.generated_boundary_nitsche_trace_requests.clear();
    if (!options.source_component_tag.empty()) {
        install.source_component_tag = options.source_component_tag;
    }
    if (options.clear_extra_trial_fields) {
        install.extra_trial_fields.clear();
    }
    for (const FieldId field : options.extra_trial_fields) {
        if (std::find(install.extra_trial_fields.begin(), install.extra_trial_fields.end(), field) ==
            install.extra_trial_fields.end()) {
            install.extra_trial_fields.push_back(field);
        }
    }
    return install;
}

CoupledResidualKernels DiagnosticRegistry::install(FESystem& system,
                                                   std::string_view set,
                                                   const OperatorTag& operator_tag,
                                                   std::initializer_list<FieldId> fields,
                                                   const forms::FormExpr& form,
                                                   const FormInstallOptions& formulation_options,
                                                   const DiagnosticOperatorOptions& options)
{
    auto& entry = find(set);
    auto kernels = installFormulation(system, operator_tag, fields, form,
                                      operatorOptions(formulation_options, options));
    if (std::find(entry.operators.begin(), entry.operators.end(), operator_tag) == entry.operators.end()) {
        entry.operators.push_back(operator_tag);
    }
    return kernels;
}

std::vector<OperatorTag> DiagnosticRegistry::operators(std::string_view set) const
{
    return find(set).operators;
}

DiagnosticOperatorHandle DiagnosticRegistry::handle(std::string_view set, OperatorTag operator_tag) const
{
    return DiagnosticOperatorHandle{.set = find(set).name, .tag = std::move(operator_tag)};
}

const DiagnosticRegistry::Set& DiagnosticRegistry::find(std::string_view set) const
{
    for (const auto& entry : sets_) {
        if (entry.name == set) {
            return entry;
        }
    }
    throw std::invalid_argument("DiagnosticRegistry: undeclared diagnostic set '" + std::string(set) + "'");
}

DiagnosticRegistry::Set& DiagnosticRegistry::find(std::string_view set)
{
    return const_cast<Set&>(static_cast<const DiagnosticRegistry&>(*this).find(set));
}

DiagnosticRegistry& diagnosticRegistry(FESystem& system)
{
    return system.attachment<DiagnosticRegistry>();
}

const DiagnosticRegistry* findDiagnosticRegistry(const FESystem& system) noexcept
{
    return system.findAttachment<DiagnosticRegistry>();
}

} // namespace systems
} // namespace FE
} // namespace svmp
