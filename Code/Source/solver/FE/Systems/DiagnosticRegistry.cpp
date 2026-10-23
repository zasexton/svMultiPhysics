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
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace svmp {
namespace FE {
namespace systems {

void DiagnosticRegistry::declareSet(std::string set, std::vector<std::string> environment_switches)
{
    for (auto& existing : sets_) {
        if (existing.name == set) {
            existing.environment_switches = std::move(environment_switches);
            existing.operators.clear();
            return;
        }
    }
    sets_.push_back(Set{.name = std::move(set),
                        .environment_switches = std::move(environment_switches),
                        .operators = {}});
}

bool DiagnosticRegistry::enabled(std::string_view set) const
{
    for (const auto& name : find(set).environment_switches) {
        const char* env = std::getenv(name.c_str());
        if (env == nullptr || env[0] == '\0') {
            continue;
        }
        std::string normalized(env);
        std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        return normalized != "0" && normalized != "false" && normalized != "no" && normalized != "off";
    }
    return false;
}

FormInstallOptions DiagnosticRegistry::operatorOptions(const FormInstallOptions& formulation_options,
                                                       const DiagnosticOperatorOptions& options)
{
    FormInstallOptions install = formulation_options;
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
    entry.operators.push_back(operator_tag);
    return kernels;
}

std::vector<OperatorTag> DiagnosticRegistry::operators(std::string_view set) const
{
    return find(set).operators;
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
