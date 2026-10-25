/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/FunctionalFormEvaluator.h"

#include "Systems/BorderedGlobalViews.h"

#include "Assembly/CutIntegrationContext.h"
#include "Assembly/FunctionalAssembler.h"
#include "Assembly/GlobalSystemView.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Core/AlignedAllocator.h"
#include "Core/Alignment.h"
#include "Core/FEException.h"
#include "Forms/BoundaryFunctional.h"
#include "Forms/JIT/ExternalCalls.h"
#include "Spaces/FunctionSpace.h"
#include "Systems/ExactAccumulator.h"
#include "Systems/FESystem.h"
#include "Systems/FunctionalForms.h"
#include "Systems/SystemCollectives.h"
#include "Systems/SystemState.h"
#include "Systems/SystemsExceptions.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace svmp {
namespace FE {
namespace systems {

namespace {

constexpr const char* kWhere = "FE::systems::evaluateFunctionalForms: ";

enum class Measure : std::uint8_t {
    Cell,
    Boundary,
    CutVolume,
    Interface
};

/// One integral of a form, with the sign it has in the form.
struct IntegralTerm {
    std::size_t form{0};
    bool negated{false};
    Measure measure{Measure::Cell};
    int marker{-1};
    geometry::CutIntegrationSide side{geometry::CutIntegrationSide::Negative};
    forms::FormExpr integrand{};
};

[[nodiscard]] bool isIntegral(forms::FormExprType type) noexcept
{
    switch (type) {
    case forms::FormExprType::CellIntegral:
    case forms::FormExprType::BoundaryIntegral:
    case forms::FormExprType::InteriorFaceIntegral:
    case forms::FormExprType::InterfaceIntegral:
    case forms::FormExprType::CutVolumeIntegral:
        return true;
    default:
        return false;
    }
}

[[nodiscard]] bool containsIntegral(const forms::FormExprNode& node)
{
    if (isIntegral(node.type())) {
        return true;
    }
    for (const auto& child : node.childrenShared()) {
        if (child && containsIntegral(*child)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] std::string formLabel(std::size_t form)
{
    return "form " + std::to_string(form) + ": ";
}

void appendIntegralTerms(const std::shared_ptr<forms::FormExprNode>& node,
                         std::size_t form,
                         bool negated,
                         std::vector<IntegralTerm>& terms)
{
    FE_THROW_IF(!node, InvalidArgumentException,
                std::string(kWhere) + formLabel(form) + "the form is not a sum of integrals");
    const auto children = node->childrenShared();
    switch (node->type()) {
    case forms::FormExprType::Add:
    case forms::FormExprType::Subtract:
        FE_THROW_IF(children.size() != 2u, InvalidArgumentException,
                    std::string(kWhere) + formLabel(form) + "malformed sum");
        appendIntegralTerms(children[0], form, negated, terms);
        appendIntegralTerms(children[1],
                            form,
                            node->type() == forms::FormExprType::Subtract ? !negated : negated,
                            terms);
        return;
    case forms::FormExprType::Negate:
        FE_THROW_IF(children.size() != 1u, InvalidArgumentException,
                    std::string(kWhere) + formLabel(form) + "malformed negation");
        appendIntegralTerms(children[0], form, !negated, terms);
        return;
    default:
        break;
    }
    FE_THROW_IF(!isIntegral(node->type()), InvalidArgumentException,
                std::string(kWhere) + formLabel(form) +
                    "the form is not a sum of integrals (state it with a measure: dx, ds, dI, dL, "
                    "dCutVolume)");
    FE_THROW_IF(children.size() != 1u || !children[0], InvalidArgumentException,
                std::string(kWhere) + formLabel(form) + "an integral has no integrand");
    FE_THROW_IF(containsIntegral(*children[0]), InvalidArgumentException,
                std::string(kWhere) + formLabel(form) + "an integrand contains an integral");

    IntegralTerm term;
    term.form = form;
    term.negated = negated;
    term.integrand = forms::FormExpr(children[0]);
    switch (node->type()) {
    case forms::FormExprType::CellIntegral:
        term.measure = Measure::Cell;
        break;
    case forms::FormExprType::BoundaryIntegral:
        term.measure = Measure::Boundary;
        term.marker = node->boundaryMarker().value_or(-1);
        break;
    case forms::FormExprType::InterfaceIntegral:
        term.measure = Measure::Interface;
        term.marker = node->interfaceMarker().value_or(-1);
        FE_THROW_IF(term.marker < 0, InvalidArgumentException,
                    std::string(kWhere) + formLabel(form) +
                        "dI, dL and a generated ds need the marker of their rules");
        break;
    case forms::FormExprType::CutVolumeIntegral: {
        term.measure = Measure::CutVolume;
        term.marker = node->interfaceMarker().value_or(-1);
        FE_THROW_IF(term.marker < 0, InvalidArgumentException,
                    std::string(kWhere) + formLabel(form) +
                        "dCutVolume needs the marker of its rules");
        const auto side = node->cutVolumeSide();
        FE_THROW_IF(!side.has_value(), InvalidArgumentException,
                    std::string(kWhere) + formLabel(form) + "dCutVolume has no side");
        term.side = *side == forms::CutVolumeSide::Negative
                        ? geometry::CutIntegrationSide::Negative
                        : geometry::CutIntegrationSide::Positive;
        break;
    }
    default:
        FE_THROW(InvalidArgumentException,
                 std::string(kWhere) + formLabel(form) +
                     "dS (interior facets) is not evaluated; use dx, ds, dI, dL or dCutVolume");
    }
    terms.push_back(std::move(term));
}

[[nodiscard]] std::vector<IntegralTerm> integralTerms(std::span<const forms::FormExpr> forms)
{
    std::vector<IntegralTerm> terms;
    for (std::size_t form = 0; form < forms.size(); ++form) {
        const auto& expr = forms[form];
        FE_THROW_IF(!expr.isValid(), InvalidArgumentException,
                    std::string(kWhere) + formLabel(form) + "the form is empty");
        FE_THROW_IF(expr.hasTest() || expr.hasTrial(), InvalidArgumentException,
                    std::string(kWhere) + formLabel(form) +
                        "the form contains a test or trial function (a functional is a 0-form)");
        appendIntegralTerms(expr.nodeShared(), form, /*negated=*/false, terms);
    }
    return terms;
}

[[nodiscard]] bool isCutMeasure(Measure measure) noexcept
{
    return measure == Measure::CutVolume || measure == Measure::Interface;
}

/// Whether this rank's cut context has rules or a publication for the marker.
[[nodiscard]] bool knowsMarker(const assembly::CutIntegrationContext& context,
                               const IntegralTerm& term)
{
    if (context.hasFreeSurfaceGeometrySnapshotForMarker(term.marker) ||
        context.hasGeneratedLevelSetInterfaceMarker(term.marker)) {
        return true;
    }
    if (term.measure == Measure::CutVolume) {
        if (context.hasGeneratedVolumeMarker(term.marker)) {
            return true;
        }
        return std::any_of(context.volumeRules().begin(),
                           context.volumeRules().end(),
                           [&](const geometry::CutQuadratureRule& rule) {
                               return rule.kind == geometry::CutQuadratureKind::Volume &&
                                      rule.provenance.marker == term.marker;
                           });
    }
    if (context.hasGeneratedInterfaceMarker(term.marker) ||
        context.hasGeneratedActiveBoundaryMarker(term.marker) ||
        context.findGeneratedInterfaceBoundaryPublicationProvenance(term.marker) != nullptr) {
        return true;
    }
    return std::any_of(context.interfaceRules().begin(),
                       context.interfaceRules().end(),
                       [&](const geometry::CutQuadratureRule& rule) {
                           return rule.provenance.marker == term.marker;
                       });
}

/**
 * The integrand of one term, interpreted, adding each weighted quadrature
 * value w_q f_q (with the term's sign) and |w_q f_q| to exact accumulators.
 * The totals it returns to the assembler are zero.
 */
class ExactTermKernel final : public assembly::FunctionalKernel {
public:
    ExactTermKernel(std::shared_ptr<assembly::FunctionalKernel> integrand,
                    bool negated,
                    ExactAccumulator& value,
                    ExactAccumulator& absolute_mass)
        : integrand_(std::move(integrand))
        , negated_(negated)
        , value_(value)
        , absolute_mass_(absolute_mass)
    {
    }

    [[nodiscard]] assembly::RequiredData getRequiredData() const noexcept override
    {
        return integrand_->getRequiredData();
    }

    [[nodiscard]] std::vector<assembly::FieldRequirement> fieldRequirements() const override
    {
        return integrand_->fieldRequirements();
    }

    [[nodiscard]] bool hasCell() const noexcept override { return integrand_->hasCell(); }
    [[nodiscard]] bool hasBoundaryFace() const noexcept override
    {
        return integrand_->hasBoundaryFace();
    }

    [[nodiscard]] Real evaluateCell(const assembly::AssemblyContext& ctx, LocalIndex q) override
    {
        return integrand_->evaluateCell(ctx, q);
    }

    [[nodiscard]] Real evaluateCellTotal(const assembly::AssemblyContext& ctx) override
    {
        for (LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
            add(integrand_->evaluateCell(ctx, q) * ctx.integrationWeight(q));
        }
        return Real{0.0};
    }

    [[nodiscard]] Real evaluateBoundaryFace(const assembly::AssemblyContext& ctx,
                                            LocalIndex q,
                                            int boundary_marker) override
    {
        return integrand_->evaluateBoundaryFace(ctx, q, boundary_marker);
    }

    [[nodiscard]] Real evaluateBoundaryFaceTotal(const assembly::AssemblyContext& ctx,
                                                 int boundary_marker) override
    {
        for (LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
            add(integrand_->evaluateBoundaryFace(ctx, q, boundary_marker) *
                ctx.integrationWeight(q));
        }
        return Real{0.0};
    }

    [[nodiscard]] std::string name() const override { return "FunctionalFormTerm"; }

private:
    void add(Real term) noexcept
    {
        value_.add(negated_ ? -term : term);
        absolute_mass_.add(std::abs(term));
    }

    std::shared_ptr<assembly::FunctionalKernel> integrand_;
    bool negated_{false};
    ExactAccumulator& value_;
    ExactAccumulator& absolute_mass_;
};

/// The interpreted kernel of a term's integrand (the JIT is off).
[[nodiscard]] std::shared_ptr<assembly::FunctionalKernel> interpretedIntegrand(
    const IntegralTerm& term)
{
    forms::SymbolicOptions options;
    options.jit.enable = false;
    if (term.measure == Measure::Cell || term.measure == Measure::CutVolume) {
        forms::BoundaryFunctional functional;
        functional.integrand = term.integrand;
        functional.is_domain_functional = true;
        return forms::compileBoundaryFunctionalKernel(functional, options);
    }
    return forms::compileBoundaryFunctionalKernel(term.integrand, term.marker, options);
}

void evaluateTermLocally(assembly::FunctionalAssembler& assembler,
                         const IntegralTerm& term,
                         const assembly::CutIntegrationContext* cut_context,
                         ExactAccumulator& value,
                         ExactAccumulator& absolute_mass)
{
    ExactTermKernel kernel(interpretedIntegrand(term), term.negated, value, absolute_mass);
    switch (term.measure) {
    case Measure::Cell:
        (void)assembler.assembleScalar(kernel);
        break;
    case Measure::Boundary:
        (void)assembler.assembleBoundaryScalar(kernel, term.marker);
        break;
    case Measure::CutVolume:
        (void)assembler.assembleCutVolumeScalar(kernel, *cut_context, term.marker, term.side);
        break;
    case Measure::Interface:
        (void)assembler.assembleGeneratedInterfaceScalar(kernel, *cut_context, term.marker);
        break;
    }
    const auto& result = assembler.getLastResult();
    FE_THROW_IF(!result.success, InvalidStateException,
                std::string(kWhere) + formLabel(term.form) + result.error_message);
}

} // namespace

std::vector<FunctionalFormValue> evaluateFunctionalForms(
    assembly::FunctionalAssembler& assembler,
    std::span<const forms::FormExpr> forms,
    const assembly::CutIntegrationContext* cut_context,
    const CollectiveCommunicator& comm)
{
    // The forms are the same on every rank, so these checks throw on every
    // rank alike, before any collective.
    const auto terms = integralTerms(forms);

    const bool has_cut_terms = std::any_of(terms.begin(), terms.end(), [](const IntegralTerm& term) {
        return isCutMeasure(term.measure);
    });
    if (has_cut_terms) {
        FE_THROW_IF(collectiveAnyTrue(cut_context == nullptr, comm), InvalidArgumentException,
                    std::string(kWhere) +
                        "a cut measure (dCutVolume, dI, dL or a generated ds) needs a cut integration "
                        "context on every rank");
        for (const auto& term : terms) {
            if (!isCutMeasure(term.measure)) {
                continue;
            }
            FE_THROW_IF(!collectiveAnyTrue(knowsMarker(*cut_context, term), comm),
                        InvalidArgumentException,
                        std::string(kWhere) + formLabel(term.form) + "marker " +
                            std::to_string(term.marker) +
                            (term.measure == Measure::CutVolume
                                 ? " has no cut-volume rules in the cut integration context"
                                 : " is not an interface, contact-line or active-boundary marker of the "
                                   "cut integration context"));
        }
    }

    // values[form] and then absolute masses[form], for one reduction.
    std::vector<ExactAccumulator> sums(2u * forms.size());
    std::exception_ptr local_failure;
    try {
        for (const auto& term : terms) {
            evaluateTermLocally(assembler,
                                term,
                                cut_context,
                                sums[term.form],
                                sums[forms.size() + term.form]);
        }
    } catch (...) {
        local_failure = std::current_exception();
    }
    requireCollectivePhase(local_failure,
                           comm,
                           "failed the local evaluation of collective phase",
                           "functional_form_evaluation");

    const auto reduced = collectiveExactSums(sums, comm);
    std::vector<FunctionalFormValue> values(forms.size());
    for (std::size_t form = 0; form < forms.size(); ++form) {
        values[form].value = static_cast<Real>(reduced[form]);
        values[form].absolute_mass = static_cast<Real>(reduced[forms.size() + form]);
    }
    return values;
}

namespace {

std::vector<FunctionalFormValue> evaluateOnSystem(
    const FESystem& system,
    std::optional<FieldId> quadrature_field,
    std::span<const forms::FormExpr> forms,
    const SystemStateView& state,
    const CollectiveCommunicator& comm,
    const assembly::CutIntegrationContext* cut_context)
{
    FE_THROW_IF(!system.isSetup(), InvalidStateException,
                std::string(kWhere) + "the system is not set up");

    // Owner-to-ghost copies of the state's backend vectors (as for the
    // system's boundary functionals).
    std::exception_ptr ghost_failure;
    try {
        for (const auto* vector : {state.u_vector, state.u_prev_vector, state.u_prev2_vector}) {
            if (vector != nullptr) {
                const_cast<backends::GenericVector*>(vector)->updateGhosts();
            }
        }
    } catch (...) {
        ghost_failure = std::current_exception();
    }
    requireCollectivePhasePreparation(ghost_failure, comm, "functional_form_ghost_refresh");

    auto fields = system.unknownFieldIdsInDofMapOrder();
    FE_THROW_IF(fields.empty(), InvalidStateException,
                std::string(kWhere) + "the system has no unknown fields");
    if (system.hasBorderedGlobalFields()) {
        // Bordered global fields are not in the DOF map; they follow it.
        for (FieldId field = 0; static_cast<std::size_t>(field) < system.registeredFieldCount(); ++field) {
            if (system.fieldRecord(field).scope == FieldScope::Global) {
                fields.push_back(field);
            }
        }
    }
    // A global field has no cell quadrature of its own; the first other
    // field gives it.
    FieldId primary_field = fields.front();
    for (const auto field : fields) {
        if (system.fieldRecord(field).scope != FieldScope::Global) {
            primary_field = field;
            break;
        }
    }
    if (quadrature_field.has_value()) {
        FE_THROW_IF(std::find(fields.begin(), fields.end(), *quadrature_field) == fields.end() ||
                        system.fieldRecord(*quadrature_field).scope != FieldScope::VolumeCell,
                    InvalidArgumentException,
                    std::string(kWhere) + "the quadrature field must be a volume unknown of the system");
        primary_field = *quadrature_field;
    }
    const auto& primary = system.fieldRecord(primary_field);
    FE_CHECK_NOT_NULL(primary.space.get(), "FE::systems::evaluateFunctionalForms: primary field space");

    assembly::FunctionalAssembler assembler;
    assembler.setMesh(system.meshAccess());
    assembler.setDofMap(system.dofHandler().getDofMap());
    assembler.setPrimaryFieldDofOffset(0);
    assembler.setSpace(*primary.space);
    assembler.setPrimaryField(primary_field);
    assembler.setDofPerNode(0); // block DOF layout
    for (const auto field : fields) {
        const auto& record = system.fieldRecord(field);
        if ((record.scope != FieldScope::VolumeCell && record.scope != FieldScope::Global) ||
            !record.space) {
            continue;
        }
        const auto& dof_handler = system.fieldDofHandler(field);
        assembly::FieldSolutionBinding binding;
        binding.field = field;
        binding.space = record.space.get();
        binding.dof_map = &dof_handler.getDofMap();
        binding.dof_offset = system.fieldDofOffset(field);
        binding.field_global_size = dof_handler.getNumDofs();
        binding.field_type = record.space->field_type();
        binding.value_dimension = record.components;
        binding.n_components = record.components;
        assembler.registerFieldBinding(binding);
    }

    assembler.setTimeIntegrationContext(state.time_integration);
    assembler.setTime(static_cast<Real>(state.time));
    assembler.setTimeStep(static_cast<Real>(state.dt));

    const auto& parameters = system.parameterRegistry();
    const bool have_parameter_contracts = !parameters.specs().empty();
    std::function<std::optional<Real>(std::string_view)> get_real_parameter{};
    std::function<std::optional<params::Value>(std::string_view)> get_parameter{};
    if (have_parameter_contracts) {
        get_real_parameter = parameters.makeRealGetter(state);
        get_parameter = parameters.makeParamGetter(state);
    }
    assembler.setRealParameterGetter(have_parameter_contracts
                                         ? &get_real_parameter
                                         : (state.getRealParam ? &state.getRealParam : nullptr));
    assembler.setParameterGetter(have_parameter_contracts
                                     ? &get_parameter
                                     : (state.getParam ? &state.getParam : nullptr));
    forms::jit::external::ExternalCallTableV1 external_calls;
    external_calls.context = state.user_data;
    assembler.setUserData(&external_calls);
    std::vector<Real, AlignedAllocator<Real, kFEPreferredAlignmentBytes>> constants;
    if (have_parameter_contracts && parameters.slotCount() > 0u) {
        const auto slots = parameters.evaluateRealSlots(state);
        constants.assign(slots.begin(), slots.end());
    }
    assembler.setJITConstants(constants);
    assembler.setCoupledValues({}, {});

    assembler.setSolution(state.u);
    // Bordered global DOFs are read from the system.
    std::unique_ptr<assembly::GlobalSystemView> solution_view =
        makeStateReadView(system, state.u_vector, state.u, /*ghosted=*/true);
    if (solution_view != nullptr) {
        assembler.setSolutionView(solution_view.get());
    }
    // History spans of backend storage are not indexed by global DOF; with
    // backend vectors the previous states are read through their views.
    const bool previous_through_views =
        state.u_prev_vector != nullptr || state.u_prev2_vector != nullptr;
    if (!state.u_history.empty() && !previous_through_views) {
        for (std::size_t k = 0; k < state.u_history.size(); ++k) {
            assembler.setPreviousSolutionK(static_cast<int>(k + 1u), state.u_history[k]);
        }
    } else {
        assembler.setPreviousSolution(state.u_prev);
        assembler.setPreviousSolution2(state.u_prev2);
    }
    std::unique_ptr<assembly::GlobalSystemView> previous_view;
    std::unique_ptr<assembly::GlobalSystemView> previous2_view;
    if (state.u_prev_vector != nullptr) {
        previous_view = const_cast<backends::GenericVector*>(state.u_prev_vector)->createGhostedReadView();
        assembler.setPreviousSolutionView(previous_view.get());
    }
    if (state.u_prev2_vector != nullptr) {
        previous2_view = const_cast<backends::GenericVector*>(state.u_prev2_vector)->createGhostedReadView();
        assembler.setPreviousSolution2View(previous2_view.get());
    }

    return evaluateFunctionalForms(assembler,
                                   forms,
                                   cut_context != nullptr ? cut_context : system.cutIntegrationContext(),
                                   comm);
}

} // namespace

std::vector<FunctionalFormValue> evaluateFunctionalForms(
    const FESystem& system,
    std::span<const forms::FormExpr> forms,
    const SystemStateView& state,
    const CollectiveCommunicator& comm,
    const assembly::CutIntegrationContext* cut_context)
{
    return evaluateOnSystem(system, std::nullopt, forms, state, comm, cut_context);
}

std::vector<FunctionalFormValue> evaluateFunctionalFormsWithQuadratureOf(
    const FESystem& system,
    FieldId quadrature_field,
    std::span<const forms::FormExpr> forms,
    const SystemStateView& state,
    const CollectiveCommunicator& comm,
    const assembly::CutIntegrationContext* cut_context)
{
    return evaluateOnSystem(system, quadrature_field, forms, state, comm, cut_context);
}

std::vector<FunctionalFormValue> evaluateDeclaredFunctionalForms(
    const FESystem& system,
    const SystemStateView& state,
    const CollectiveCommunicator& comm,
    const assembly::CutIntegrationContext* cut_context)
{
    const auto declarations = functionalForms(system);
    std::vector<forms::FormExpr> forms;
    forms.reserve(declarations.size());
    for (const auto& declaration : declarations) {
        forms.push_back(declaration.form);
    }
    return evaluateFunctionalForms(system, forms, state, comm, cut_context);
}

} // namespace systems
} // namespace FE
} // namespace svmp
