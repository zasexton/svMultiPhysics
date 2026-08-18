#include "Auxiliary/AuxiliaryInputRegistry.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace svmp {
namespace FE {
namespace systems {

namespace {

void validateCollectiveRoute(const AuxiliaryInputSpec& spec,
                             std::string_view operation)
{
    FE_THROW_IF(spec.requires_mpi_reduction &&
                    spec.collective_route_key.empty(),
                InvalidArgumentException,
                "AuxiliaryInputRegistry::" + std::string(operation) +
                    ": input '" + spec.name +
                    "' requires a non-empty collective_route_key");
}

[[nodiscard]] std::size_t checkedInputValueCount(
    const AuxiliaryInputSpec& spec,
    std::string_view operation)
{
    const auto component_count =
        static_cast<std::size_t>(spec.size);
    if (spec.entity_count == 0) {
        return component_count;
    }
    FE_THROW_IF(
        component_count != 0 &&
            spec.entity_count >
                std::numeric_limits<std::size_t>::max() /
                    component_count,
        InvalidArgumentException,
        "AuxiliaryInputRegistry::" + std::string(operation) +
            ": entity_count * size overflows for input '" +
            spec.name + "'");
    return spec.entity_count * component_count;
}

[[nodiscard]] std::size_t checkedStorageEnd(
    std::size_t slot,
    std::size_t value_count,
    std::string_view operation,
    std::string_view input_name)
{
    FE_THROW_IF(
        value_count >
            std::numeric_limits<std::size_t>::max() - slot,
        InvalidArgumentException,
        "AuxiliaryInputRegistry::" + std::string(operation) +
            ": flat storage size overflows for input '" +
            std::string(input_name) + "'");
    return slot + value_count;
}

} // namespace

// ---------------------------------------------------------------------------
//  Frozen evaluation plan
// ---------------------------------------------------------------------------

struct AuxiliaryInputRegistry::EvaluationPlan::Storage {
    const AuxiliaryInputRegistry* owner{nullptr};
    std::uint64_t registry_revision{0};
    Real time{0.0};
    Real dt{0.0};
    bool is_nonlinear_iteration{false};
    bool started{false};
    bool staged{false};
    bool consumed{false};
    std::uint64_t execution_revision{0};
    std::size_t next_input{0};
    std::size_t staged_input{0};
    std::vector<std::size_t> due_input_indices{};
    std::vector<EvaluationInvocation> invocations{};
    std::vector<
        Real,
        AlignedAllocator<Real, kFEPreferredAlignmentBytes>>
        scratch_values{};
};

AuxiliaryInputRegistry::EvaluationPlan::EvaluationPlan(
    std::unique_ptr<Storage> storage) noexcept
    : storage_(std::move(storage))
{
}

AuxiliaryInputRegistry::EvaluationPlan::EvaluationPlan(
    EvaluationPlan&&) noexcept = default;

AuxiliaryInputRegistry::EvaluationPlan&
AuxiliaryInputRegistry::EvaluationPlan::operator=(EvaluationPlan&&) noexcept = default;

AuxiliaryInputRegistry::EvaluationPlan::~EvaluationPlan() = default;

std::span<const std::size_t>
AuxiliaryInputRegistry::EvaluationPlan::dueInputIndices() const noexcept
{
    if (!storage_) {
        return {};
    }
    return storage_->due_input_indices;
}

std::span<const AuxiliaryInputRegistry::EvaluationInvocation>
AuxiliaryInputRegistry::EvaluationPlan::invocations() const noexcept
{
    if (!storage_) {
        return {};
    }
    return storage_->invocations;
}

Real AuxiliaryInputRegistry::EvaluationPlan::time() const noexcept
{
    return storage_ ? storage_->time : Real{0.0};
}

Real AuxiliaryInputRegistry::EvaluationPlan::dt() const noexcept
{
    return storage_ ? storage_->dt : Real{0.0};
}

bool AuxiliaryInputRegistry::EvaluationPlan::isNonlinearIteration() const noexcept
{
    return storage_ && storage_->is_nonlinear_iteration;
}

bool AuxiliaryInputRegistry::EvaluationPlan::hasStagedEvaluation() const noexcept
{
    return storage_ && storage_->staged;
}

bool AuxiliaryInputRegistry::EvaluationPlan::consumed() const noexcept
{
    return !storage_ || storage_->consumed;
}

// ---------------------------------------------------------------------------
//  Internal helpers
// ---------------------------------------------------------------------------

std::size_t AuxiliaryInputRegistry::entryIndex(std::string_view name) const
{
    auto it = name_to_index_.find(std::string(name));
    FE_THROW_IF(it == name_to_index_.end(), InvalidArgumentException,
                "AuxiliaryInputRegistry: unknown input '" + std::string(name) + "'");
    return it->second;
}

void AuxiliaryInputRegistry::requireNoCallbackLifecycleMutation(
    std::string_view operation) const
{
    FE_THROW_IF(callback_execution_in_progress_, InvalidStateException,
                "AuxiliaryInputRegistry::" + std::string(operation) +
                    ": lifecycle mutation is not permitted during "
                    "callback execution");
}

// ---------------------------------------------------------------------------
//  Registration
// ---------------------------------------------------------------------------

std::size_t AuxiliaryInputRegistry::registerInput(
    const AuxiliaryInputSpec& spec,
    AuxiliaryInputCallback callback)
{
    requireNoCallbackLifecycleMutation("registerInput");
    FE_THROW_IF(spec.name.empty(), InvalidArgumentException,
                "AuxiliaryInputRegistry::registerInput: empty name");
    FE_THROW_IF(spec.size <= 0, InvalidArgumentException,
                "AuxiliaryInputRegistry::registerInput: size must be > 0");
    FE_THROW_IF(name_to_index_.count(spec.name) != 0u, InvalidArgumentException,
                "AuxiliaryInputRegistry::registerInput: duplicate '" + spec.name + "'");
    validateCollectiveRoute(spec, "registerInput");

    if (!spec.component_names.empty()) {
        FE_THROW_IF(static_cast<int>(spec.component_names.size()) != spec.size,
                    InvalidArgumentException,
                    "AuxiliaryInputRegistry::registerInput: component_names size mismatch");
    }

    const auto slot = values_.size();
    const auto total =
        checkedInputValueCount(spec, "registerInput");
    const auto idx = entries_.size();
    InputEntry entry;
    entry.spec = spec;
    entry.callback = std::move(callback);
    entry.slot = slot;
    entry.dirty = true;

    values_.resize(
        checkedStorageEnd(
            slot, total, "registerInput", spec.name),
        Real{0.0});
    try {
        entries_.push_back(std::move(entry));
        try {
            const auto [position, inserted] =
                name_to_index_.emplace(spec.name, idx);
            static_cast<void>(position);
            FE_THROW_IF(
                !inserted,
                InvalidStateException,
                "AuxiliaryInputRegistry::registerInput: name index "
                "insertion failed for '" +
                    spec.name + "'");
        } catch (...) {
            entries_.pop_back();
            throw;
        }
    } catch (...) {
        values_.resize(slot);
        throw;
    }
    ++evaluation_revision_;
    return slot;
}

std::size_t AuxiliaryInputRegistry::registerEntityInput(
    const AuxiliaryInputSpec& spec,
    AuxiliaryEntityInputCallback callback)
{
    requireNoCallbackLifecycleMutation("registerEntityInput");
    FE_THROW_IF(spec.name.empty(), InvalidArgumentException,
                "AuxiliaryInputRegistry::registerEntityInput: empty name");
    FE_THROW_IF(spec.size <= 0, InvalidArgumentException,
                "AuxiliaryInputRegistry::registerEntityInput: size must be > 0");
    FE_THROW_IF(spec.entity_count == 0, InvalidArgumentException,
                "AuxiliaryInputRegistry::registerEntityInput: entity_count must be > 0");
    FE_THROW_IF(name_to_index_.count(spec.name) != 0u, InvalidArgumentException,
                "AuxiliaryInputRegistry::registerEntityInput: duplicate '" + spec.name + "'");
    validateCollectiveRoute(spec, "registerEntityInput");

    const auto slot = values_.size();
    const auto total =
        checkedInputValueCount(spec, "registerEntityInput");
    const auto idx = entries_.size();
    InputEntry entry;
    entry.spec = spec;
    entry.entity_callback = std::move(callback);
    entry.slot = slot;
    entry.dirty = true;

    values_.resize(
        checkedStorageEnd(
            slot, total, "registerEntityInput", spec.name),
        Real{0.0});
    try {
        entries_.push_back(std::move(entry));
        try {
            const auto [position, inserted] =
                name_to_index_.emplace(spec.name, idx);
            static_cast<void>(position);
            FE_THROW_IF(
                !inserted,
                InvalidStateException,
                "AuxiliaryInputRegistry::registerEntityInput: name index "
                "insertion failed for '" +
                    spec.name + "'");
        } catch (...) {
            entries_.pop_back();
            throw;
        }
    } catch (...) {
        values_.resize(slot);
        throw;
    }
    ++evaluation_revision_;
    return slot;
}

// ---------------------------------------------------------------------------
//  Access
// ---------------------------------------------------------------------------

bool AuxiliaryInputRegistry::hasInput(std::string_view name) const noexcept
{
    return name_to_index_.find(std::string(name)) != name_to_index_.end();
}

bool AuxiliaryInputRegistry::hasCollectiveInput() const noexcept
{
    return std::any_of(
        entries_.begin(), entries_.end(), [](const InputEntry& entry) {
            return entry.spec.requires_mpi_reduction;
        });
}

std::size_t AuxiliaryInputRegistry::slotOf(std::string_view name) const
{
    return entries_[entryIndex(name)].slot;
}

const AuxiliaryInputSpec& AuxiliaryInputRegistry::specOf(std::string_view name) const
{
    return entries_[entryIndex(name)].spec;
}

std::span<const Real> AuxiliaryInputRegistry::valuesOf(std::string_view name) const
{
    const auto& e = entries_[entryIndex(name)];
    const auto total = (e.spec.entity_count > 0)
        ? e.spec.entity_count * static_cast<std::size_t>(e.spec.size)
        : static_cast<std::size_t>(e.spec.size);
    return {values_.data() + e.slot, total};
}

std::span<const Real> AuxiliaryInputRegistry::valuesOf(
    std::string_view name, std::size_t entity_index) const
{
    const auto& e = entries_[entryIndex(name)];
    if (e.spec.entity_count == 0) {
        // Global input: entity_index is ignored.
        return {values_.data() + e.slot, static_cast<std::size_t>(e.spec.size)};
    }
    FE_THROW_IF(entity_index >= e.spec.entity_count, InvalidArgumentException,
                "AuxiliaryInputRegistry::valuesOf: entity_index " +
                    std::to_string(entity_index) + " >= entity_count " +
                    std::to_string(e.spec.entity_count));
    const auto offset = e.slot + entity_index * static_cast<std::size_t>(e.spec.size);
    return {values_.data() + offset, static_cast<std::size_t>(e.spec.size)};
}

bool AuxiliaryInputRegistry::isEntityLocal(std::string_view name) const
{
    return entries_[entryIndex(name)].spec.entity_count > 0;
}

std::span<Real> AuxiliaryInputRegistry::mutableValuesOf(std::string_view name)
{
    requireNoCallbackLifecycleMutation("mutableValuesOf");
    const auto& e = entries_[entryIndex(name)];
    const auto total = (e.spec.entity_count > 0)
        ? e.spec.entity_count * static_cast<std::size_t>(e.spec.size)
        : static_cast<std::size_t>(e.spec.size);
    ++evaluation_revision_;
    return {values_.data() + e.slot, total};
}

Real AuxiliaryInputRegistry::get(std::string_view name) const
{
    const auto& e = entries_[entryIndex(name)];
    FE_THROW_IF(e.spec.size != 1, InvalidArgumentException,
                "AuxiliaryInputRegistry::get: input '" + std::string(name) +
                    "' has size " + std::to_string(e.spec.size) + ", not scalar");
    return values_[e.slot];
}

void AuxiliaryInputRegistry::set(std::string_view name, Real value)
{
    requireNoCallbackLifecycleMutation("set");
    const auto& e = entries_[entryIndex(name)];
    FE_THROW_IF(e.spec.size != 1, InvalidArgumentException,
                "AuxiliaryInputRegistry::set: input '" + std::string(name) +
                    "' has size " + std::to_string(e.spec.size) + ", not scalar");
    values_[e.slot] = value;
    ++evaluation_revision_;
}

std::vector<std::string> AuxiliaryInputRegistry::inputNames() const
{
    std::vector<std::string> names;
    names.reserve(entries_.size());
    for (const auto& e : entries_) {
        names.push_back(e.spec.name);
    }
    return names;
}

// ---------------------------------------------------------------------------
//  Evaluation lifecycle
// ---------------------------------------------------------------------------

void AuxiliaryInputRegistry::evaluate(Real time, Real dt,
                                       bool is_nonlinear_iteration)
{
    auto plan = prepareEvaluation(time, dt, is_nonlinear_iteration);
    executeEvaluationPlan(plan);
}

AuxiliaryInputRegistry::EvaluationPlan
AuxiliaryInputRegistry::prepareEvaluation(
    Real time, Real dt, bool is_nonlinear_iteration) const
{
    auto storage = std::make_unique<EvaluationPlan::Storage>();
    storage->owner = this;
    storage->registry_revision = evaluation_revision_;
    storage->time = time;
    storage->dt = dt;
    storage->is_nonlinear_iteration = is_nonlinear_iteration;

    const auto order = evaluationOrder();
    storage->due_input_indices.reserve(order.size());
    storage->invocations.reserve(order.size());
    std::size_t maximum_value_count = 0;

    for (const auto idx : order) {
        const auto& entry = entries_[idx];
        bool should_eval = false;
        switch (entry.spec.update_schedule) {
            case AuxiliaryInputUpdateSchedule::OnceAtSetup:
                should_eval = !entry.evaluated_at_setup;
                break;
            case AuxiliaryInputUpdateSchedule::OncePerTimeStep:
                should_eval = entry.dirty;
                break;
            case AuxiliaryInputUpdateSchedule::EachNonlinearIteration:
                should_eval = entry.dirty || is_nonlinear_iteration;
                break;
            case AuxiliaryInputUpdateSchedule::Manual:
                should_eval = entry.dirty;
                break;
        }

        if (!should_eval) {
            continue;
        }

        std::size_t callback_invocation_count = 0;
        bool uses_entity_callback = false;
        if (entry.entity_callback && entry.spec.entity_count > 0) {
            callback_invocation_count = entry.spec.entity_count;
            uses_entity_callback = true;
        } else if (entry.callback) {
            callback_invocation_count = 1;
        }

        if (callback_invocation_count == 0) {
            continue;
        }

        const auto value_count =
            checkedInputValueCount(entry.spec, "prepareEvaluation");

        storage->due_input_indices.push_back(idx);
        storage->invocations.push_back(EvaluationInvocation{
            .input_index = idx,
            .input_name = entry.spec.name,
            .producer = entry.spec.producer,
            .update_schedule = entry.spec.update_schedule,
            .requires_mpi_reduction = entry.spec.requires_mpi_reduction,
            .collective_route_key = entry.spec.collective_route_key,
            .callback_invocation_count = callback_invocation_count,
            .value_count = value_count,
            .entity_callback = uses_entity_callback,
        });
        maximum_value_count =
            std::max(maximum_value_count, value_count);
    }
    storage->scratch_values.resize(maximum_value_count);

    return EvaluationPlan(std::move(storage));
}

void AuxiliaryInputRegistry::executeEvaluationPlan(EvaluationPlan& plan)
{
    FE_THROW_IF(!plan.storage_, InvalidArgumentException,
                "AuxiliaryInputRegistry::executeEvaluationPlan: empty plan");
    FE_THROW_IF(plan.storage_->owner != this, InvalidArgumentException,
                "AuxiliaryInputRegistry::executeEvaluationPlan: plan belongs to "
                "a different registry");
    FE_THROW_IF(plan.storage_->consumed, InvalidStateException,
                "AuxiliaryInputRegistry::executeEvaluationPlan: plan was already consumed");

    if (plan.storage_->due_input_indices.empty()) {
        (void)stageNextEvaluation(plan);
        return;
    }

    while (!plan.storage_->consumed) {
        (void)executeNextEvaluation(plan);
    }
}

bool AuxiliaryInputRegistry::stageNextEvaluation(EvaluationPlan& plan)
{
    FE_THROW_IF(callback_execution_in_progress_, InvalidStateException,
                "AuxiliaryInputRegistry::stageNextEvaluation: reentrant "
                "evaluation is not permitted");
    FE_THROW_IF(!plan.storage_, InvalidArgumentException,
                "AuxiliaryInputRegistry::stageNextEvaluation: empty plan");
    FE_THROW_IF(plan.storage_->owner != this, InvalidArgumentException,
                "AuxiliaryInputRegistry::stageNextEvaluation: plan belongs to "
                "a different registry");
    FE_THROW_IF(plan.storage_->consumed, InvalidStateException,
                "AuxiliaryInputRegistry::stageNextEvaluation: plan was already consumed");
    FE_THROW_IF(plan.storage_->staged, InvalidStateException,
                "AuxiliaryInputRegistry::stageNextEvaluation: previous result "
                "has not been committed");

    if (!plan.storage_->started) {
        FE_THROW_IF(plan.storage_->registry_revision != evaluation_revision_,
                    InvalidStateException,
                    "AuxiliaryInputRegistry::stageNextEvaluation: plan is stale");
        plan.storage_->started = true;
        if (plan.storage_->due_input_indices.empty()) {
            plan.storage_->consumed = true;
            return false;
        }
        plan.storage_->execution_revision = evaluation_revision_;
    } else {
        FE_THROW_IF(plan.storage_->execution_revision != evaluation_revision_,
                    InvalidStateException,
                    "AuxiliaryInputRegistry::stageNextEvaluation: registry changed "
                    "during plan execution");
    }

    const auto idx =
        plan.storage_->due_input_indices[plan.storage_->next_input];
    auto& entry = entries_[idx];
    const auto value_count =
        plan.storage_->invocations[plan.storage_->next_input].value_count;
    std::copy_n(
        values_.data() + entry.slot,
        value_count,
        plan.storage_->scratch_values.data());

    callback_execution_in_progress_ = true;
    try {
        if (entry.entity_callback && entry.spec.entity_count > 0) {
            const auto sz = static_cast<std::size_t>(entry.spec.size);
            for (std::size_t entity = 0; entity < entry.spec.entity_count; ++entity) {
                std::span<Real> out{
                    plan.storage_->scratch_values.data() + entity * sz, sz};
                entry.entity_callback(plan.storage_->time,
                                      plan.storage_->dt,
                                      entity,
                                      out);
            }
        } else if (entry.callback) {
            std::span<Real> out{
                plan.storage_->scratch_values.data(), value_count};
            entry.callback(plan.storage_->time, plan.storage_->dt, out);
        }
    } catch (...) {
        callback_execution_in_progress_ = false;
        plan.storage_->consumed = true;
        plan.storage_->staged = false;
        throw;
    }
    callback_execution_in_progress_ = false;

    plan.storage_->staged = true;
    plan.storage_->staged_input = idx;
    return true;
}

bool AuxiliaryInputRegistry::canCommitStagedEvaluation(
    const EvaluationPlan& plan) const noexcept
{
    if (!plan.storage_ ||
        plan.storage_->owner != this ||
        plan.storage_->consumed ||
        !plan.storage_->staged ||
        plan.storage_->execution_revision != evaluation_revision_ ||
        plan.storage_->next_input >=
            plan.storage_->due_input_indices.size() ||
        plan.storage_->next_input >=
            plan.storage_->invocations.size()) {
        return false;
    }

    const auto idx =
        plan.storage_->due_input_indices[plan.storage_->next_input];
    if (plan.storage_->staged_input != idx ||
        idx >= entries_.size()) {
        return false;
    }
    const auto value_count =
        plan.storage_->invocations[plan.storage_->next_input].value_count;
    return value_count <= plan.storage_->scratch_values.size() &&
           entries_[idx].slot <= values_.size() &&
           value_count <= values_.size() - entries_[idx].slot;
}

void AuxiliaryInputRegistry::commitStagedEvaluation(EvaluationPlan& plan)
{
    FE_THROW_IF(
        !canCommitStagedEvaluation(plan),
        InvalidStateException,
        "AuxiliaryInputRegistry::commitStagedEvaluation: staged publication "
        "preconditions are not satisfied");

    const auto idx = plan.storage_->staged_input;
    auto& entry = entries_[idx];
    const auto value_count =
        plan.storage_->invocations[plan.storage_->next_input].value_count;
    std::copy_n(
        plan.storage_->scratch_values.data(),
        value_count,
        values_.data() + entry.slot);
    entry.dirty = false;
    if (entry.spec.update_schedule == AuxiliaryInputUpdateSchedule::OnceAtSetup) {
        entry.evaluated_at_setup = true;
    }

    plan.storage_->staged = false;
    ++plan.storage_->next_input;
    if (plan.storage_->next_input == plan.storage_->due_input_indices.size()) {
        plan.storage_->consumed = true;
    }
    ++evaluation_revision_;
    plan.storage_->execution_revision = evaluation_revision_;
}

bool AuxiliaryInputRegistry::executeNextEvaluation(EvaluationPlan& plan)
{
    const bool staged = stageNextEvaluation(plan);
    if (staged) {
        commitStagedEvaluation(plan);
    }
    return staged;
}

void AuxiliaryInputRegistry::markDirty(std::string_view name)
{
    requireNoCallbackLifecycleMutation("markDirty");
    entries_[entryIndex(name)].dirty = true;
    ++evaluation_revision_;
}

void AuxiliaryInputRegistry::invalidateAll()
{
    requireNoCallbackLifecycleMutation("invalidateAll");
    for (auto& e : entries_) {
        if (e.spec.update_schedule != AuxiliaryInputUpdateSchedule::OnceAtSetup ||
            !e.evaluated_at_setup) {
            e.dirty = true;
        }
    }
    ++evaluation_revision_;
}

void AuxiliaryInputRegistry::clear()
{
    requireNoCallbackLifecycleMutation("clear");
    entries_.clear();
    name_to_index_.clear();
    values_.clear();
    ++evaluation_revision_;
}

// ---------------------------------------------------------------------------
//  Dependency ordering
// ---------------------------------------------------------------------------

void AuxiliaryInputRegistry::addDependency(std::string_view dependent,
                                            std::string_view dependency)
{
    requireNoCallbackLifecycleMutation("addDependency");
    const auto dep_idx = entryIndex(dependent);
    const auto src_idx = entryIndex(dependency);
    entries_[dep_idx].depends_on.push_back(src_idx);
    ++evaluation_revision_;
}

std::vector<std::size_t> AuxiliaryInputRegistry::evaluationOrder() const
{
    const auto n = entries_.size();

    std::vector<std::size_t> in_degree(n, 0);
    std::vector<std::vector<std::size_t>> dependents(n);

    for (std::size_t i = 0; i < n; ++i) {
        for (auto dep : entries_[i].depends_on) {
            dependents[dep].push_back(i);
            in_degree[i]++;
        }
    }

    const auto lower_name_priority = [this](std::size_t lhs, std::size_t rhs) {
        return entries_[lhs].spec.name > entries_[rhs].spec.name;
    };
    std::priority_queue<
        std::size_t,
        std::vector<std::size_t>,
        decltype(lower_name_priority)> ready(lower_name_priority);

    for (std::size_t i = 0; i < n; ++i) {
        if (in_degree[i] == 0) {
            ready.push(i);
        }
    }

    std::vector<std::size_t> order;
    order.reserve(n);

    while (!ready.empty()) {
        const auto cur = ready.top();
        ready.pop();
        order.push_back(cur);
        for (auto next : dependents[cur]) {
            if (--in_degree[next] == 0) {
                ready.push(next);
            }
        }
    }

    // If order.size() < n, there's a cycle — this is a hard error.
    // Same-time cyclic dependencies must use Monolithic coupling or
    // AuxiliaryOperator instead of input bindings.
    if (order.size() < n) {
        std::vector<std::string_view> cycle_inputs;
        cycle_inputs.reserve(n - order.size());
        for (std::size_t i = 0; i < n; ++i) {
            if (in_degree[i] > 0) {
                cycle_inputs.push_back(entries_[i].spec.name);
            }
        }
        std::sort(cycle_inputs.begin(), cycle_inputs.end());

        std::string cycle_names;
        for (const auto name : cycle_inputs) {
            if (!cycle_names.empty()) {
                cycle_names += ", ";
            }
            cycle_names += name;
        }
        FE_THROW(InvalidStateException,
                 "AuxiliaryInputRegistry: dependency cycle detected among inputs: [" +
                     cycle_names + "]. Same-time cyclic dependencies must use "
                     "Monolithic coupling or AuxiliaryOperator.");
    }

    return order;
}

// ---------------------------------------------------------------------------
//  Debug inspection
// ---------------------------------------------------------------------------

std::string AuxiliaryInputRegistry::debugDump() const
{
    std::ostringstream oss;
    oss << "AuxiliaryInputRegistry: " << entries_.size() << " inputs, "
        << values_.size() << " total values\n";

    for (std::size_t i = 0; i < entries_.size(); ++i) {
        const auto& e = entries_[i];
        oss << "  [" << i << "] \"" << e.spec.name << "\" "
            << "(size=" << e.spec.size
            << ", slot=" << e.slot
            << ", dirty=" << (e.dirty ? "yes" : "no")
            << ", producer=" << static_cast<int>(e.spec.producer)
            << ")";

        if (e.spec.size <= 4) {
            oss << " values=[";
            for (int c = 0; c < e.spec.size; ++c) {
                if (c > 0) oss << ", ";
                oss << values_[e.slot + static_cast<std::size_t>(c)];
            }
            oss << "]";
        }
        oss << "\n";
    }

    return oss.str();
}

} // namespace systems
} // namespace FE
} // namespace svmp
