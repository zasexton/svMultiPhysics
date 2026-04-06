#include "Auxiliary/AuxiliaryInputRegistry.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace svmp {
namespace FE {
namespace systems {

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

// ---------------------------------------------------------------------------
//  Registration
// ---------------------------------------------------------------------------

std::size_t AuxiliaryInputRegistry::registerInput(
    const AuxiliaryInputSpec& spec,
    AuxiliaryInputCallback callback)
{
    FE_THROW_IF(spec.name.empty(), InvalidArgumentException,
                "AuxiliaryInputRegistry::registerInput: empty name");
    FE_THROW_IF(spec.size <= 0, InvalidArgumentException,
                "AuxiliaryInputRegistry::registerInput: size must be > 0");
    FE_THROW_IF(name_to_index_.count(spec.name) != 0u, InvalidArgumentException,
                "AuxiliaryInputRegistry::registerInput: duplicate '" + spec.name + "'");

    if (!spec.component_names.empty()) {
        FE_THROW_IF(static_cast<int>(spec.component_names.size()) != spec.size,
                    InvalidArgumentException,
                    "AuxiliaryInputRegistry::registerInput: component_names size mismatch");
    }

    const auto slot = values_.size();
    values_.resize(slot + static_cast<std::size_t>(spec.size), Real{0.0});

    const auto idx = entries_.size();
    InputEntry entry;
    entry.spec = spec;
    entry.callback = std::move(callback);
    entry.slot = slot;
    entry.dirty = true;
    entries_.push_back(std::move(entry));

    name_to_index_.emplace(spec.name, idx);
    return slot;
}

std::size_t AuxiliaryInputRegistry::registerEntityInput(
    const AuxiliaryInputSpec& spec,
    AuxiliaryEntityInputCallback callback)
{
    FE_THROW_IF(spec.name.empty(), InvalidArgumentException,
                "AuxiliaryInputRegistry::registerEntityInput: empty name");
    FE_THROW_IF(spec.size <= 0, InvalidArgumentException,
                "AuxiliaryInputRegistry::registerEntityInput: size must be > 0");
    FE_THROW_IF(spec.entity_count == 0, InvalidArgumentException,
                "AuxiliaryInputRegistry::registerEntityInput: entity_count must be > 0");
    FE_THROW_IF(name_to_index_.count(spec.name) != 0u, InvalidArgumentException,
                "AuxiliaryInputRegistry::registerEntityInput: duplicate '" + spec.name + "'");

    const auto slot = values_.size();
    const auto total = spec.entity_count * static_cast<std::size_t>(spec.size);
    values_.resize(slot + total, Real{0.0});

    const auto idx = entries_.size();
    InputEntry entry;
    entry.spec = spec;
    entry.entity_callback = std::move(callback);
    entry.slot = slot;
    entry.dirty = true;
    entries_.push_back(std::move(entry));

    name_to_index_.emplace(spec.name, idx);
    return slot;
}

// ---------------------------------------------------------------------------
//  Access
// ---------------------------------------------------------------------------

bool AuxiliaryInputRegistry::hasInput(std::string_view name) const noexcept
{
    return name_to_index_.find(std::string(name)) != name_to_index_.end();
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
    const auto& e = entries_[entryIndex(name)];
    return {values_.data() + e.slot, static_cast<std::size_t>(e.spec.size)};
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
    const auto& e = entries_[entryIndex(name)];
    FE_THROW_IF(e.spec.size != 1, InvalidArgumentException,
                "AuxiliaryInputRegistry::set: input '" + std::string(name) +
                    "' has size " + std::to_string(e.spec.size) + ", not scalar");
    values_[e.slot] = value;
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
    const auto order = evaluationOrder();

    for (const auto idx : order) {
        auto& entry = entries_[idx];

        // Check if this input needs evaluation.
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

        if (!should_eval) continue;
        if (!entry.callback && !entry.entity_callback) continue;

        if (entry.entity_callback && entry.spec.entity_count > 0) {
            // Entity-local evaluation: call once per entity.
            const auto sz = static_cast<std::size_t>(entry.spec.size);
            for (std::size_t e = 0; e < entry.spec.entity_count; ++e) {
                std::span<Real> out{values_.data() + entry.slot + e * sz, sz};
                entry.entity_callback(time, dt, e, out);
            }
        } else if (entry.callback) {
            // Global evaluation.
            const auto total = (entry.spec.entity_count > 0)
                ? entry.spec.entity_count * static_cast<std::size_t>(entry.spec.size)
                : static_cast<std::size_t>(entry.spec.size);
            std::span<Real> out{values_.data() + entry.slot, total};
            entry.callback(time, dt, out);
        }

        entry.dirty = false;
        if (entry.spec.update_schedule == AuxiliaryInputUpdateSchedule::OnceAtSetup) {
            entry.evaluated_at_setup = true;
        }
    }
}

void AuxiliaryInputRegistry::markDirty(std::string_view name)
{
    entries_[entryIndex(name)].dirty = true;
}

void AuxiliaryInputRegistry::invalidateAll()
{
    for (auto& e : entries_) {
        if (e.spec.update_schedule != AuxiliaryInputUpdateSchedule::OnceAtSetup ||
            !e.evaluated_at_setup) {
            e.dirty = true;
        }
    }
}

void AuxiliaryInputRegistry::clear()
{
    entries_.clear();
    name_to_index_.clear();
    values_.clear();
}

// ---------------------------------------------------------------------------
//  Dependency ordering
// ---------------------------------------------------------------------------

void AuxiliaryInputRegistry::addDependency(std::string_view dependent,
                                            std::string_view dependency)
{
    const auto dep_idx = entryIndex(dependent);
    const auto src_idx = entryIndex(dependency);
    entries_[dep_idx].depends_on.push_back(src_idx);
}

std::vector<std::size_t> AuxiliaryInputRegistry::evaluationOrder() const
{
    // Topological sort via Kahn's algorithm.
    const auto n = entries_.size();

    std::vector<std::size_t> in_degree(n, 0);
    std::vector<std::vector<std::size_t>> dependents(n);

    for (std::size_t i = 0; i < n; ++i) {
        for (auto dep : entries_[i].depends_on) {
            dependents[dep].push_back(i);
            in_degree[i]++;
        }
    }

    std::vector<std::size_t> queue;
    queue.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (in_degree[i] == 0) {
            queue.push_back(i);
        }
    }

    std::vector<std::size_t> order;
    order.reserve(n);
    std::size_t head = 0;

    while (head < queue.size()) {
        const auto cur = queue[head++];
        order.push_back(cur);
        for (auto next : dependents[cur]) {
            if (--in_degree[next] == 0) {
                queue.push_back(next);
            }
        }
    }

    // If order.size() < n, there's a cycle — this is a hard error.
    // Same-time cyclic dependencies must use Monolithic coupling or
    // AuxiliaryOperator instead of input bindings.
    if (order.size() < n) {
        std::string cycle_names;
        for (std::size_t i = 0; i < n; ++i) {
            if (in_degree[i] > 0) {
                if (!cycle_names.empty()) cycle_names += ", ";
                cycle_names += entries_[i].spec.name;
            }
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
