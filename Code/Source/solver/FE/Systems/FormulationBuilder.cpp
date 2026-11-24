/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/FormulationBuilder.h"

#include "Analysis/FormExprScanner.h"
#include "Forms/FormExpr.h"
#include "Forms/Vocabulary.h"
#include "PostProcessing/ScalarHistory.h"
#include "Systems/ALEBinding.h"
#include "Systems/FESystem.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace svmp {
namespace FE {
namespace systems {

InferredInstallFields inferInstallFields(const FESystem& system, const forms::FormExpr& residual)
{
    FE_THROW_IF(!residual.isValid(), InvalidArgumentException, "FE::systems::inferInstallFields: invalid residual");
    std::vector<FieldId> tests;
    std::vector<FieldId> states;
    const auto visit = [&](const auto& self, const forms::FormExprNode& node) -> void {
        if (node.type() == forms::FormExprType::TestFunction) {
            const auto field = node.fieldId();
            FE_THROW_IF(!field.has_value(), InvalidArgumentException,
                        "FE::systems::inferInstallFields: the test function '" + node.toString() +
                            "' names no field; state it with a field (FieldBinding::unknown)");
            tests.push_back(*field);
        } else if (node.type() == forms::FormExprType::StateField) {
            const auto field = node.fieldId();
            if (field && *field != CURRENT_SOLUTION_FIELD_ID && system.fieldParticipatesInUnknownVector(*field)) {
                states.push_back(*field);
            }
        }
        for (const auto& child : node.childrenShared()) {
            if (child) {
                self(self, *child);
            }
        }
    };
    visit(visit, *residual.node());
    FE_THROW_IF(tests.empty(), InvalidArgumentException,
                "FE::systems::inferInstallFields: the residual has no test function");
    const auto ordered = [](std::vector<FieldId> fields) {
        std::sort(fields.begin(), fields.end());
        fields.erase(std::unique(fields.begin(), fields.end()), fields.end());
        return fields;
    };
    InferredInstallFields out;
    out.test_fields = ordered(std::move(tests));
    for (const auto field : ordered(std::move(states))) {
        if (!std::binary_search(out.test_fields.begin(), out.test_fields.end(), field)) {
            out.extra_trial_fields.push_back(field);
        }
    }
    return out;
}

FormulationBuilder::FormulationBuilder(FESystem& system,
                                       OperatorTag operator_tag,
                                       FormInstallOptions base)
    : FormulationBuilder(system, std::move(operator_tag), std::move(base), OperatorCreation::Ensure)
{
}

FormulationBuilder::FormulationBuilder(FESystem& system,
                                       OperatorTag operator_tag,
                                       FormInstallOptions base,
                                       OperatorCreation creation)
    : system_(&system)
    , operator_tag_(std::move(operator_tag))
    , options_(std::move(base))
{
    if (creation == OperatorCreation::Add || !system_->hasOperator(operator_tag_)) {
        system_->addOperator(operator_tag_);
    }
}

FormulationBuilder& FormulationBuilder::multiplierLowering(MultiplierLowering lowering)
{
    system_->requestMultiplierLowering(lowering, operator_tag_);
    return *this;
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

FormulationBuilder& FormulationBuilder::freezeGeometryInTangent()
{
    options_.moving_domain_geometry = false;
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


Quantity FormulationBuilder::quantity(std::string name,
                                      FunctionalKind kind,
                                      forms::FormExpr form,
                                      std::string owner)
{
    declareFunctionalForm(*system_,
                          FunctionalFormDeclaration{
                              .name = name,
                              .kind = kind,
                              .form = form,
                              .owner = owner.empty() ? operator_tag_ : std::move(owner),
                          });
    return systems::quantity(std::move(name), kind, std::move(form));
}

FormulationBuilder& FormulationBuilder::record(const Quantity& quantity)
{
    post::record(*system_, quantity);
    return *this;
}

FormulationBuilder& FormulationBuilder::record(FieldId field)
{
    post::record(*system_, field);
    return *this;
}

FormulationBuilder& FormulationBuilder::wallFunctional(WallFunctionalFormDeclaration declaration)
{
    if (declaration.owner.empty()) {
        declaration.owner = operator_tag_;
    }
    declareWallFunctionalForm(*system_, std::move(declaration));
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

void FormulationBuilder::registerGeneratedBoundaries(const forms::FormExpr& residual)
{
    if (!residual.isValid() || residual.node() == nullptr) {
        return;
    }
    for (const auto& measure : analysis::scanFormExpr(*residual.node()).exterior_boundary_measures) {
        if (measure.isGeneratedActiveSubset()) {
            system_->registerGeneratedEmbeddedInterfaceMarker(measure.generatedActiveBoundaryMarker());
        }
    }
}

void FormulationBuilder::verifyTaggedOccurrences(const forms::FormExpr& residual)
{
    if (tagged_terms_.empty() || !residual.isValid() || residual.node() == nullptr) {
        return;
    }
    using FT = forms::FormExprType;
    const auto first_record = [&](const forms::FormExprNode* node) -> const TaggedTerm* {
        for (const auto& tagged_term : tagged_terms_) {
            if (tagged_term.term.node() == node) {
                return &tagged_term;
            }
        }
        return nullptr;
    };
    const auto describe = [&](const TaggedTerm& tagged_term) {
        return "the term tagged '" + tagged_term.label + "' in the row of '" +
               system_->fieldRecord(tagged_term.test_field).name + "'";
    };
    // Whether a subtree holds a tagged term (memoized: residuals share subtrees).
    std::unordered_map<const forms::FormExprNode*, bool> holds;
    const auto holds_tagged = [&](const auto& self, const forms::FormExprNode& node) -> bool {
        if (const auto it = holds.find(&node); it != holds.end()) {
            return it->second;
        }
        bool found = first_record(&node) != nullptr;
        for (const auto* child : node.children()) {
            if (!found && child != nullptr) {
                found = self(self, *child);
            }
        }
        holds.emplace(&node, found);
        return found;
    };
    // The summands of sign +1 of each tagged term along the additive spine:
    // sums, differences, negations and measures (an integral is linear in
    // its integrand); a tagged term anywhere else is refused.
    std::unordered_map<const forms::FormExprNode*, std::size_t> installed;
    const auto walk = [&](const auto& self, const forms::FormExprNode& node, int sign) -> void {
        if (!holds_tagged(holds_tagged, node)) {
            return;
        }
        if (const auto* tagged_term = first_record(&node)) {
            if (sign < 0) {
                throw std::invalid_argument(
                    "FormulationBuilder: " + describe(*tagged_term) +
                    " is installed with sign -1 (R - F.tag(label, a) installs -a); tag the signed term, "
                    "R + F.tag(label, -a), so the channel records what the residual installs");
            }
            ++installed[&node];
            return;
        }
        const auto kids = node.children();
        switch (node.type()) {
            case FT::Add:
                self(self, *kids[0], sign);
                self(self, *kids[1], sign);
                return;
            case FT::Subtract:
                self(self, *kids[0], sign);
                self(self, *kids[1], -sign);
                return;
            case FT::Negate:
            case FT::CellIntegral:
            case FT::BoundaryIntegral:
            case FT::InteriorFaceIntegral:
            case FT::InterfaceIntegral:
            case FT::CutVolumeIntegral:
            case FT::GlobalIntegral:
                self(self, *kids[0], node.type() == FT::Negate ? -sign : sign);
                return;
            default:
                break;
        }
        const auto inner_tagged = [&](const auto& find, const forms::FormExprNode& n) -> const TaggedTerm* {
            if (const auto* tagged_term = first_record(&n)) {
                return tagged_term;
            }
            for (const auto* child : n.children()) {
                if (child != nullptr && holds_tagged(holds_tagged, *child)) {
                    return find(find, *child);
                }
            }
            return nullptr;
        };
        throw std::invalid_argument("FormulationBuilder: " + describe(*inner_tagged(inner_tagged, node)) +
                                    " is installed inside '" + node.toString() +
                                    "', not as a summand of the residual");
    };
    walk(walk, *residual.node(), +1);
    for (const auto& [node, count] : installed) {
        const auto total = (tagged_installed_[node] += count);
        const auto tagged = static_cast<std::size_t>(
            std::count_if(tagged_terms_.begin(), tagged_terms_.end(),
                          [node = node](const TaggedTerm& t) { return t.term.node() == node; }));
        if (total > tagged) {
            throw std::invalid_argument("FormulationBuilder: " + describe(*first_record(node)) + " is installed " +
                                        std::to_string(total) + " times but tagged " + std::to_string(tagged) +
                                        " times; tag each summand the channel records");
        }
    }
}

void FormulationBuilder::requireTaggedTermsInstalled(std::string_view context) const
{
    for (const auto& tagged_term : tagged_terms_) {
        const auto* node = tagged_term.term.node();
        const auto tagged = static_cast<std::size_t>(
            std::count_if(tagged_terms_.begin(), tagged_terms_.end(),
                          [node](const TaggedTerm& t) { return t.term.node() == node; }));
        const auto it = tagged_installed_.find(node);
        const std::size_t installed = it == tagged_installed_.end() ? 0u : it->second;
        if (installed < tagged) {
            throw std::invalid_argument("FormulationBuilder::" + std::string(context) + ": the term tagged '" +
                                        tagged_term.label + "' in the row of '" +
                                        system_->fieldRecord(tagged_term.test_field).name + "' is installed " +
                                        std::to_string(installed) + " times but tagged " + std::to_string(tagged) +
                                        " times; a work channel reports terms of the installed residual");
        }
    }
}

CoupledResidualKernels FormulationBuilder::install(std::span<const FieldId> fields,
                                                   const forms::FormExpr& residual)
{
    verifyTaggedOccurrences(residual);
    registerGeneratedBoundaries(residual);
    return installFormulation(*system_, operator_tag_, fields, residual, options_);
}

CoupledResidualKernels FormulationBuilder::residual(const forms::FormExpr& residual)
{
    const auto inferred = inferInstallFields(*system_, residual);
    verifyTaggedOccurrences(residual);
    registerGeneratedBoundaries(residual);
    auto options = options_;
    for (const auto field : inferred.extra_trial_fields) {
        if (std::find(options.extra_trial_fields.begin(), options.extra_trial_fields.end(), field) ==
            options.extra_trial_fields.end()) {
            options.extra_trial_fields.push_back(field);
        }
    }
    return installFormulation(*system_, operator_tag_, inferred.test_fields, residual, options);
}

CoupledResidualKernels FormulationBuilder::install(std::initializer_list<FieldId> fields,
                                                   const forms::FormExpr& residual)
{
    return install(std::span<const FieldId>{fields.begin(), fields.size()}, residual);
}

CoupledResidualKernels FormulationBuilder::installGroup(std::span<const FieldId> fields,
                                                        const forms::FormExpr& residual,
                                                        const FormulationGroup& group)
{
    verifyTaggedOccurrences(residual);
    registerGeneratedBoundaries(residual);
    return installFormulation(*system_, operator_tag_, fields, residual, groupOptions(group));
}

CoupledResidualKernels FormulationBuilder::installGroup(std::initializer_list<FieldId> fields,
                                                        const forms::FormExpr& residual,
                                                        const FormulationGroup& group)
{
    return installGroup(std::span<const FieldId>{fields.begin(), fields.size()}, residual, group);
}

OperatorTag FormulationBuilder::workChannel(std::string_view suffix, std::initializer_list<WorkChannelPart> parts)
{
    requireTaggedTermsInstalled("workChannel");
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

namespace {

void collectTestFields(const forms::FormExprNode& node, std::vector<FieldId>& fields)
{
    if (node.type() == forms::FormExprType::TestFunction) {
        const auto field = node.fieldId();
        if (field && std::find(fields.begin(), fields.end(), *field) == fields.end()) {
            fields.push_back(*field);
        }
    }
    for (const auto* child : node.children()) {
        if (child != nullptr) {
            collectTestFields(*child, fields);
        }
    }
}

} // namespace

forms::FormExpr FormulationBuilder::tag(std::string_view label, forms::FormExpr term)
{
    if (!term.isValid()) {
        throw std::invalid_argument("FormulationBuilder::tag: the term tagged '" + std::string(label) +
                                    "' is empty");
    }
    std::vector<FieldId> test_fields;
    collectTestFields(*term.node(), test_fields);
    if (test_fields.size() != 1u) {
        throw std::invalid_argument("FormulationBuilder::tag: the term tagged '" + std::string(label) + "' holds " +
                                    std::to_string(test_fields.size()) +
                                    " fields' test functions; a tagged term holds the test function of one field");
    }
    tagged_terms_.push_back({std::string(label), test_fields.front(), term});
    return term;
}

forms::FormExpr FormulationBuilder::tagged(std::string_view label, FieldId test_field) const
{
    forms::FormExpr sum;
    for (const auto& tagged_term : tagged_terms_) {
        if (tagged_term.label == label && tagged_term.test_field == test_field) {
            sum = forms::sumOf(sum, tagged_term.term);
        }
    }
    return sum;
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
