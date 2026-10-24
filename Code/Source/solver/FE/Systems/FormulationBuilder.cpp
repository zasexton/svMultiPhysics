/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/FormulationBuilder.h"

#include "Forms/FormExpr.h"
#include "Systems/ALEBinding.h"
#include "Systems/FESystem.h"

#include <algorithm>
#include <string>
#include <utility>

namespace svmp {
namespace FE {
namespace systems {

FormulationBuilder::FormulationBuilder(FESystem& system,
                                       OperatorTag operator_tag,
                                       FormInstallOptions base)
    : system_(&system)
    , operator_tag_(std::move(operator_tag))
    , options_(std::move(base))
{
    if (!system_->hasOperator(operator_tag_)) {
        system_->addOperator(operator_tag_);
    }
}

FormulationBuilder& FormulationBuilder::symbolicTangent(bool enabled)
{
    options_.compiler_options.use_symbolic_tangent = enabled;
    return *this;
}

FormulationBuilder& FormulationBuilder::recordConstitutiveLaw(analysis::ConstitutiveLawMetadata law)
{
    options_.recordConstitutiveLaw(std::move(law));
    return *this;
}

FormulationBuilder& FormulationBuilder::configure(const ALEBinding& ale)
{
    ale.configureInstallOptions(options_);
    return *this;
}

FormulationBuilder& FormulationBuilder::certifyGeneratedBoundaryNitscheTraces(
    std::vector<forms::bc::GeneratedBoundaryNitscheTraceFormBinding>& bindings,
    int volume_interface_marker,
    Real minimum_symmetric_energy_ratio)
{
    options_.generated_boundary_nitsche_trace_requests.reserve(bindings.size());
    for (auto& binding : bindings) {
        options_.generated_boundary_nitsche_trace_requests.push_back(
            GeneratedBoundaryNitscheTraceInstallRequest{
                .binding = std::move(binding),
                .volume_interface_marker = volume_interface_marker,
                .minimum_symmetric_energy_ratio = minimum_symmetric_energy_ratio,
            });
    }
    return *this;
}

forms::ContactLineDomain FormulationBuilder::registered(forms::ContactLineDomain domain)
{
    system_->registerGeneratedEmbeddedInterfaceMarker(domain.marker());
    return domain;
}

forms::BoundaryDomain FormulationBuilder::registered(forms::BoundaryDomain domain)
{
    system_->registerGeneratedEmbeddedInterfaceMarker(domain.generatedActiveBoundaryMarker());
    return domain;
}

FormulationBuilder& FormulationBuilder::functional(std::string name,
                                                   FunctionalKind kind,
                                                   forms::FormExpr form,
                                                   std::string owner)
{
    declareFunctionalForm(*system_,
                          FunctionalFormDeclaration{
                              .name = std::move(name),
                              .kind = kind,
                              .form = std::move(form),
                              .owner = owner.empty() ? operator_tag_ : std::move(owner),
                          });
    return *this;
}

FormInstallOptions FormulationBuilder::groupOptions(const FormulationGroup& group) const
{
    FormInstallOptions group_options = options_;
    if (group.drop_generated_boundary_nitsche_trace_requests) {
        group_options.generated_boundary_nitsche_trace_requests.clear();
    }
    if (!group.source_component_tag.empty()) {
        group_options.source_component_tag = group.source_component_tag;
    }
    if (group.ale != nullptr) {
        group.ale->configureInstallOptions(group_options);
    }
    for (const FieldId field : group.extra_trial_fields) {
        if (std::find(group_options.extra_trial_fields.begin(), group_options.extra_trial_fields.end(), field) ==
            group_options.extra_trial_fields.end()) {
            group_options.extra_trial_fields.push_back(field);
        }
    }
    return group_options;
}

CoupledResidualKernels FormulationBuilder::install(std::initializer_list<FieldId> fields,
                                                   const forms::FormExpr& residual)
{
    return installFormulation(*system_, operator_tag_, fields, residual, options_);
}

CoupledResidualKernels FormulationBuilder::installGroup(std::initializer_list<FieldId> fields,
                                                        const forms::FormExpr& residual,
                                                        const FormulationGroup& group)
{
    return installFormulation(*system_, operator_tag_, fields, residual, groupOptions(group));
}

OperatorTag FormulationBuilder::workChannel(std::string_view suffix, std::initializer_list<WorkChannelPart> parts)
{
    const OperatorTag channel_tag = operator_tag_ + "_" + std::string(suffix);
    bool installed = false;
    for (const auto& part : parts) {
        if (!part.form.isValid()) {
            continue;
        }
        auto part_options = options_;
        part_options.contributes_to_problem_analysis = false;
        if (!part.keep_generated_boundary_nitsche_trace_requests) {
            part_options.generated_boundary_nitsche_trace_requests.clear();
        }
        part_options.source_component_tag = channel_tag;
        for (const FieldId field : part.extra_trial_fields) {
            if (std::find(part_options.extra_trial_fields.begin(), part_options.extra_trial_fields.end(), field) ==
                part_options.extra_trial_fields.end()) {
                part_options.extra_trial_fields.push_back(field);
            }
        }
        (void)installFormulation(*system_, channel_tag, {part.test_field}, part.form, part_options);
        installed = true;
    }
    return installed ? channel_tag : OperatorTag{};
}

DiagnosticRegistry& FormulationBuilder::diagnostics()
{
    return diagnosticRegistry(*system_);
}

FormulationBuilder& FormulationBuilder::log(const LogRecord& record)
{
    logRecord(record);
    return *this;
}

FormulationBuilder& FormulationBuilder::log(std::span<const LogRecord> records)
{
    logRecords(records);
    return *this;
}

CoupledResidualKernels FormulationBuilder::diagnostic(std::string_view set,
                                                      const OperatorTag& operator_tag,
                                                      std::initializer_list<FieldId> fields,
                                                      const forms::FormExpr& form,
                                                      const DiagnosticOperatorOptions& options)
{
    return diagnostics().install(*system_, set, operator_tag, fields, form, options_, options);
}

} // namespace systems
} // namespace FE
} // namespace svmp
