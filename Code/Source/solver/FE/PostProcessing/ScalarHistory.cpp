/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "PostProcessing/ScalarHistory.h"

#include "Auxiliary/AuxiliaryBindings.h"
#include "Auxiliary/AuxiliaryInputRegistry.h"
#include "Core/FEException.h"
#include "Systems/ExactAccumulator.h"
#include "Systems/FESystem.h"
#include "Systems/FunctionalFormEvaluator.h"
#include "Systems/FunctionalForms.h"
#include "Systems/SystemsExceptions.h"
#include "TimeStepping/TimeHistory.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <exception>
#include <utility>

namespace svmp {
namespace FE {
namespace post {

namespace {

constexpr const char* kWhere = "FE::post::ScalarHistory: ";

void addColumn(systems::FESystem& system, ScalarHistoryColumn column)
{
    const std::string where = std::string(kWhere) + "column '" + column.name + "': ";
    FE_THROW_IF(column.name.empty() || column.name.find_first_not_of(" \t") == std::string::npos,
                InvalidArgumentException, where + "the name must be nonempty");
    FE_THROW_IF(column.name.find_first_of(",\"\r\n") != std::string::npos, InvalidArgumentException,
                where + "the name must not hold a comma, a quote or a line break");
    FE_THROW_IF(column.name == "step" || column.name == "time", InvalidArgumentException,
                where + "the name is the header of the step or time column");
    auto& declarations = system.attachment<ScalarHistoryDeclarations>();
    FE_THROW_IF(std::any_of(declarations.columns.begin(), declarations.columns.end(),
                            [&](const ScalarHistoryColumn& existing) { return existing.name == column.name; }),
                InvalidArgumentException, where + "a column of this name is already recorded");
    declarations.columns.push_back(std::move(column));
}

void appendNumber(std::string& out, double value)
{
    std::array<char, 64> buffer{};
    const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    FE_THROW_IF(result.ec != std::errc{}, systems::InvalidStateException,
                std::string(kWhere) + "a value could not be written");
    out.append(buffer.data(), result.ptr);
}

[[nodiscard]] systems::AuxiliaryConstraintBinding committedValue(systems::AuxiliaryConstraintValueSource source,
                                                                 const std::string& value)
{
    systems::AuxiliaryConstraintBinding binding;
    binding.value_source = source;
    binding.value_name = value;
    binding.state_view = systems::AuxiliaryOutputStateView::Committed;
    return binding;
}

} // namespace

void record(systems::FESystem& system, const systems::Quantity& quantity)
{
    FE_THROW_IF(!quantity.isValid(), InvalidArgumentException, std::string(kWhere) + "invalid quantity");
    addColumn(system, ScalarHistoryColumn{
                          .name = quantity.name(),
                          .source = ScalarHistorySource::Quantity,
                          .form = quantity.form(),
                      });
}

void record(systems::FESystem& system, FieldId field)
{
    FE_THROW_IF(!system.hasField(field), InvalidArgumentException,
                std::string(kWhere) + "field " + std::to_string(field) + " is not a field of the system");
    const auto& record = system.fieldRecord(field);
    FE_THROW_IF(record.scope != systems::FieldScope::Global, InvalidArgumentException,
                std::string(kWhere) + "field '" + record.name + "' is not a global field");
    addColumn(system, ScalarHistoryColumn{
                          .name = record.name,
                          .source = ScalarHistorySource::GlobalField,
                          .field = field,
                      });
}

void record(systems::FESystem& system, const systems::AuxiliaryInputHandle& input, std::string name)
{
    FE_THROW_IF(input.registryName().empty(), InvalidArgumentException,
                std::string(kWhere) + "the auxiliary input has no registry name");
    addColumn(system, ScalarHistoryColumn{
                          .name = std::move(name),
                          .source = ScalarHistorySource::AuxiliaryInput,
                          .target = input.registryName(),
                      });
}

void recordAuxiliaryState(systems::FESystem& system,
                          const systems::AuxiliaryInstanceHandle& instance,
                          const std::string& state)
{
    addColumn(system, ScalarHistoryColumn{
                          .name = instance.instanceName() + "." + state,
                          .source = ScalarHistorySource::AuxiliaryState,
                          .target = instance.instanceName(),
                          .value = state,
                      });
}

void recordAuxiliaryOutput(systems::FESystem& system,
                           const systems::AuxiliaryInstanceHandle& instance,
                           const std::string& output)
{
    addColumn(system, ScalarHistoryColumn{
                          .name = instance.instanceName() + "." + output,
                          .source = ScalarHistorySource::AuxiliaryOutput,
                          .target = instance.instanceName(),
                          .value = output,
                      });
}

std::span<const ScalarHistoryColumn> scalarHistoryColumns(const systems::FESystem& system) noexcept
{
    const auto* declarations = system.findAttachment<ScalarHistoryDeclarations>();
    if (declarations == nullptr) {
        return {};
    }
    return declarations->columns;
}

std::string scalarHistoryHeader(const systems::FESystem& system)
{
    std::string out = "step,time";
    for (const auto& column : scalarHistoryColumns(system)) {
        out += ',';
        out += column.name;
    }
    out += '\n';
    return out;
}

std::vector<Real> scalarHistoryValues(const systems::FESystem& system,
                                      const systems::SystemStateView& state,
                                      const systems::CollectiveCommunicator& comm)
{
    const auto columns = scalarHistoryColumns(system);
    std::vector<Real> values(columns.size(), Real{0.0});

    // The quantities, in one collective evaluation; exact sums give the same
    // value on every rank.
    std::vector<forms::FormExpr> quantity_forms;
    for (const auto& column : columns) {
        if (column.source == ScalarHistorySource::Quantity) {
            quantity_forms.push_back(column.form);
        }
    }
    if (!quantity_forms.empty()) {
        const auto evaluated = systems::evaluateFunctionalForms(system, quantity_forms, state, comm);
        std::size_t next = 0;
        for (std::size_t i = 0; i < columns.size(); ++i) {
            if (columns[i].source == ScalarHistorySource::Quantity) {
                values[i] = evaluated.at(next++).value;
            }
        }
    }

    // The other columns are read on every rank; rank 0's value (a global
    // field's owner) is given to every rank through an exact sum.
    std::exception_ptr failure;
    std::vector<systems::ExactAccumulator> rank_zero(columns.size());
    try {
        for (std::size_t i = 0; i < columns.size(); ++i) {
            const auto& column = columns[i];
            Real value = Real{0.0};
            switch (column.source) {
            case ScalarHistorySource::Quantity:
                continue;
            case ScalarHistorySource::GlobalField: {
                const auto field_values = system.globalFieldValues(column.field, state);
                FE_THROW_IF(field_values.size() != 1u, systems::InvalidStateException,
                            std::string(kWhere) + "global field '" + column.name + "' is not a scalar");
                value = field_values.front();
                break;
            }
            case ScalarHistorySource::AuxiliaryInput: {
                const auto* registry = system.auxiliaryInputRegistryIfPresent();
                FE_THROW_IF(registry == nullptr || !registry->hasInput(column.target), systems::InvalidStateException,
                            std::string(kWhere) + "auxiliary input '" + column.target + "' is not registered");
                value = registry->get(column.target);
                break;
            }
            case ScalarHistorySource::AuxiliaryState:
                value = system.auxiliaryConstraintValue(
                    column.target,
                    committedValue(systems::AuxiliaryConstraintValueSource::State, column.value),
                    state.time, state.dt);
                break;
            case ScalarHistorySource::AuxiliaryOutput:
                value = system.auxiliaryConstraintValue(
                    column.target,
                    committedValue(systems::AuxiliaryConstraintValueSource::Output, column.value),
                    state.time, state.dt);
                break;
            }
            if (comm.rank() == 0) {
                rank_zero[i].add(value);
            }
        }
    } catch (...) {
        failure = std::current_exception();
    }
    systems::requireCollectivePhasePreparation(failure, comm, "scalar_history_values");
    const auto shared = systems::collectiveExactSums(rank_zero, comm);
    for (std::size_t i = 0; i < columns.size(); ++i) {
        if (columns[i].source != ScalarHistorySource::Quantity) {
            values[i] = shared[i];
        }
    }
    return values;
}

std::string scalarHistoryRow(std::uint64_t step, Real time, std::span<const Real> values)
{
    std::string out = std::to_string(step);
    out += ',';
    appendNumber(out, time);
    for (const auto value : values) {
        out += ',';
        appendNumber(out, value);
    }
    out += '\n';
    return out;
}

ScalarHistoryWriter::ScalarHistoryWriter(const systems::FESystem& system,
                                         std::string path,
                                         systems::CollectiveCommunicator comm)
    : system_(&system)
    , path_(std::move(path))
    , comm_(comm)
{
    std::exception_ptr failure;
    if (comm_.rank() == 0) {
        try {
            out_.open(path_, std::ios::out | std::ios::trunc);
            FE_THROW_IF(!out_, systems::InvalidStateException, std::string(kWhere) + "cannot open '" + path_ + "'");
            out_ << scalarHistoryHeader(*system_) << std::flush;
        } catch (...) {
            failure = std::current_exception();
        }
    }
    systems::requireCollectivePhasePreparation(failure, comm_, "scalar_history_open");
}

timestepping::StepHook ScalarHistoryWriter::hooks() const noexcept
{
    return timestepping::StepHook::Accepted;
}

void ScalarHistoryWriter::onStepAccepted(timestepping::TimeHistory& history)
{
    systems::SystemStateView state;
    state.time = history.time();
    state.dt = history.dt();
    state.effective_dt = history.dt();
    state.u = history.uSpan();
    state.u_vector = &history.u();
    write(static_cast<std::uint64_t>(history.stepIndex()), state);
}

void ScalarHistoryWriter::write(std::uint64_t step, const systems::SystemStateView& state)
{
    const auto values = scalarHistoryValues(*system_, state, comm_);
    if (comm_.rank() == 0) {
        out_ << scalarHistoryRow(step, state.time, values) << std::flush;
    }
}

} // namespace post
} // namespace FE
} // namespace svmp
