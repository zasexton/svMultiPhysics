/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/FormIR.h"

#include <sstream>

namespace svmp {
namespace FE {
namespace forms {

struct FormIR::Impl {
    bool compiled{false};
    FormKind kind{FormKind::Linear};
    assembly::RequiredData required{assembly::RequiredData::None};
    std::vector<assembly::FieldRequirement> field_requirements{};
    std::optional<FormExprNode::SpaceSignature> test_space{};
    std::optional<FormExprNode::SpaceSignature> trial_space{};
    int max_time_derivative_order{0};
    std::vector<IntegralTerm> terms{};
    std::string dump{};
};

FormIR::FormIR() : impl_(std::make_unique<Impl>()) {}
FormIR::~FormIR() = default;

FormIR::FormIR(FormIR&&) noexcept = default;
FormIR& FormIR::operator=(FormIR&&) noexcept = default;

bool FormIR::isCompiled() const noexcept { return impl_->compiled; }
FormKind FormIR::kind() const noexcept { return impl_->kind; }
assembly::RequiredData FormIR::requiredData() const noexcept { return impl_->required; }
const std::vector<assembly::FieldRequirement>& FormIR::fieldRequirements() const noexcept { return impl_->field_requirements; }
const std::optional<FormExprNode::SpaceSignature>& FormIR::testSpace() const noexcept { return impl_->test_space; }
const std::optional<FormExprNode::SpaceSignature>& FormIR::trialSpace() const noexcept { return impl_->trial_space; }
int FormIR::maxTimeDerivativeOrder() const noexcept { return impl_->max_time_derivative_order; }

bool FormIR::hasCellTerms() const noexcept
{
    for (const auto& t : impl_->terms) {
        if (t.domain == IntegralDomain::Cell) return true;
    }
    return false;
}

bool FormIR::hasBoundaryTerms() const noexcept
{
    for (const auto& t : impl_->terms) {
        if (t.domain == IntegralDomain::Boundary) return true;
    }
    return false;
}

bool FormIR::hasInteriorFaceTerms() const noexcept
{
    for (const auto& t : impl_->terms) {
        if (t.domain == IntegralDomain::InteriorFace) return true;
    }
    return false;
}

bool FormIR::hasInterfaceFaceTerms() const noexcept
{
    for (const auto& t : impl_->terms) {
        if (t.domain == IntegralDomain::InterfaceFace) return true;
    }
    return false;
}

const std::vector<IntegralTerm>& FormIR::terms() const noexcept { return impl_->terms; }
std::string FormIR::dump() const { return impl_->dump; }

FormIR FormIR::clone() const
{
    FormIR out;
    *out.impl_ = *impl_;
    return out;
}

void FormIR::transformIntegrands(const FormExpr::NodeTransform& transform)
{
    if (!transform) {
        return;
    }
    for (auto& term : impl_->terms) {
        if (!term.integrand.isValid()) {
            continue;
        }
        term.integrand = term.integrand.transformNodes(transform);
        term.debug_string = term.integrand.toString();
    }
    // Any cached dump is now stale (term strings may have changed).
    impl_->dump.clear();
}

void FormIR::transformIntegrands(const TermTransform& transform)
{
    if (!transform) {
        return;
    }
    for (auto& term : impl_->terms) {
        if (!term.integrand.isValid()) {
            continue;
        }
        const auto per_term = [&](const FormExprNode& n) -> std::optional<FormExpr> {
            return transform(n, term);
        };
        term.integrand = term.integrand.transformNodes(per_term);
        term.debug_string = term.integrand.toString();
    }
    impl_->dump.clear();
}

void FormIR::setCompiled(bool compiled) { impl_->compiled = compiled; }
void FormIR::setKind(FormKind kind) { impl_->kind = kind; }
void FormIR::setRequiredData(assembly::RequiredData required) { impl_->required = required; }
void FormIR::setFieldRequirements(std::vector<assembly::FieldRequirement> reqs) { impl_->field_requirements = std::move(reqs); }
void FormIR::setTestSpace(std::optional<FormExprNode::SpaceSignature> sig) { impl_->test_space = std::move(sig); }
void FormIR::setTrialSpace(std::optional<FormExprNode::SpaceSignature> sig) { impl_->trial_space = std::move(sig); }
void FormIR::setMaxTimeDerivativeOrder(int order) { impl_->max_time_derivative_order = order; }
void FormIR::setTerms(std::vector<IntegralTerm> terms) { impl_->terms = std::move(terms); }
void FormIR::setDump(std::string dump) { impl_->dump = std::move(dump); }

} // namespace forms
} // namespace FE
} // namespace svmp
