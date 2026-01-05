/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "AffineConstraints.h"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>

namespace svmp {
namespace FE {
namespace constraints {

// ============================================================================
// ConstraintLine implementation
// ============================================================================

void ConstraintLine::mergeEntries() {
    if (entries.size() < 2) return;

    // Sort by master DOF
    std::sort(entries.begin(), entries.end());

    // Merge duplicates
    std::vector<ConstraintEntry> merged;
    merged.reserve(entries.size());

    for (const auto& entry : entries) {
        if (!merged.empty() && merged.back().master_dof == entry.master_dof) {
            merged.back().weight += entry.weight;
        } else {
            merged.push_back(entry);
        }
    }

    // Remove entries with near-zero weight
    merged.erase(
        std::remove_if(merged.begin(), merged.end(),
                       [](const ConstraintEntry& e) {
                           return std::abs(e.weight) < 1e-15;
                       }),
        merged.end());

    entries = std::move(merged);
}

// ============================================================================
// AffineConstraints construction
// ============================================================================

AffineConstraints::AffineConstraints() = default;

AffineConstraints::AffineConstraints(const AffineConstraintsOptions& options)
    : options_(options) {}

AffineConstraints::~AffineConstraints() = default;

AffineConstraints::AffineConstraints(AffineConstraints&& other) noexcept = default;

AffineConstraints& AffineConstraints::operator=(AffineConstraints&& other) noexcept = default;

AffineConstraints::AffineConstraints(const AffineConstraints& other) = default;

AffineConstraints& AffineConstraints::operator=(const AffineConstraints& other) = default;

// ============================================================================
// Building phase
// ============================================================================

void AffineConstraints::addLine(GlobalIndex slave_dof) {
    checkNotClosed();

    if (slave_dof < 0) {
        CONSTRAINT_THROW_DOF("Invalid negative DOF index", slave_dof);
    }

    auto it = building_lines_.find(slave_dof);
    if (it != building_lines_.end()) {
        if (!options_.allow_overwrite) {
            CONSTRAINT_THROW_DOF("DOF is already constrained", slave_dof);
        }
        // Clear existing entries if overwriting
        it->second.entries.clear();
        it->second.inhomogeneity = 0.0;
    } else {
        ConstraintLine line;
        line.slave_dof = slave_dof;
        building_lines_[slave_dof] = std::move(line);
    }
}

void AffineConstraints::addLines(std::span<const GlobalIndex> slave_dofs) {
    for (GlobalIndex dof : slave_dofs) {
        addLine(dof);
    }
}

void AffineConstraints::addEntry(GlobalIndex slave_dof, GlobalIndex master_dof, double weight) {
    checkNotClosed();

    if (std::abs(weight) < options_.zero_tolerance) {
        return;  // Skip near-zero weights
    }

    auto* line = findLine(slave_dof);
    if (!line) {
        CONSTRAINT_THROW_DOF("Constraint line not found for DOF", slave_dof);
    }

    // Check for self-reference
    if (master_dof == slave_dof) {
        CONSTRAINT_THROW_DOF("Constraint cannot reference itself as master", slave_dof);
    }

    line->entries.push_back({master_dof, weight});
}

void AffineConstraints::addEntries(GlobalIndex slave_dof,
                                    std::span<const GlobalIndex> master_dofs,
                                    std::span<const double> weights) {
    if (master_dofs.size() != weights.size()) {
        CONSTRAINT_THROW("Master DOFs and weights arrays must have same size");
    }

    for (std::size_t i = 0; i < master_dofs.size(); ++i) {
        addEntry(slave_dof, master_dofs[i], weights[i]);
    }
}

void AffineConstraints::setInhomogeneity(GlobalIndex slave_dof, double value) {
    checkNotClosed();

    auto* line = findLine(slave_dof);
    if (!line) {
        CONSTRAINT_THROW_DOF("Constraint line not found for DOF", slave_dof);
    }

    line->inhomogeneity = value;
}

void AffineConstraints::addInhomogeneity(GlobalIndex slave_dof, double value) {
    checkNotClosed();

    auto* line = findLine(slave_dof);
    if (!line) {
        CONSTRAINT_THROW_DOF("Constraint line not found for DOF", slave_dof);
    }

    line->inhomogeneity += value;
}

void AffineConstraints::addConstraintLine(const ConstraintLine& line) {
    addLine(line.slave_dof);
    for (const auto& entry : line.entries) {
        addEntry(line.slave_dof, entry.master_dof, entry.weight);
    }
    setInhomogeneity(line.slave_dof, line.inhomogeneity);
}

void AffineConstraints::addDirichlet(GlobalIndex dof, double value) {
    addLine(dof);
    setInhomogeneity(dof, value);
}

void AffineConstraints::addDirichlet(std::span<const GlobalIndex> dofs, double value) {
    for (GlobalIndex dof : dofs) {
        addDirichlet(dof, value);
    }
}

void AffineConstraints::addDirichlet(std::span<const GlobalIndex> dofs,
                                      std::span<const double> values) {
    if (dofs.size() != values.size()) {
        CONSTRAINT_THROW("DOFs and values arrays must have same size");
    }

    for (std::size_t i = 0; i < dofs.size(); ++i) {
        addDirichlet(dofs[i], values[i]);
    }
}

void AffineConstraints::addPeriodic(GlobalIndex slave_dof, GlobalIndex master_dof) {
    addLine(slave_dof);
    addEntry(slave_dof, master_dof, 1.0);
}

void AffineConstraints::merge(const AffineConstraints& other, bool overwrite) {
    checkNotClosed();
    if (other.is_closed_) {
        CONSTRAINT_THROW("Cannot merge from closed AffineConstraints");
    }

    for (const auto& [slave, line] : other.building_lines_) {
        auto it = building_lines_.find(slave);
        if (it != building_lines_.end()) {
            if (!overwrite) {
                CONSTRAINT_THROW_DOF("Conflict merging constraint for DOF", slave);
            }
        }
        if (overwrite || it == building_lines_.end()) {
            building_lines_[slave] = line;
        }
    }
}

void AffineConstraints::clear() {
    building_lines_.clear();
    slave_dofs_.clear();
    slave_to_index_.clear();
    entry_offsets_.clear();
    entries_.clear();
    inhomogeneities_.clear();
    is_closed_ = false;
}

// ============================================================================
// Closing
// ============================================================================

void AffineConstraints::close() {
    if (is_closed_) {
        return;  // Already closed, no-op
    }

    // Merge duplicates in all lines
    if (options_.merge_duplicates) {
        mergeAllDuplicates();
    }

    // Compute transitive closure
    computeTransitiveClosure();

    // Sort for deterministic results
    if (options_.deterministic_order) {
        sortConstraints();
    }

    // Build CSR storage
    buildCSRStorage();

    is_closed_ = true;
}

void AffineConstraints::computeTransitiveClosure() {
    std::unordered_set<GlobalIndex> visiting;
    std::unordered_set<GlobalIndex> closed;

    for (auto& [slave, line] : building_lines_) {
        if (closed.find(slave) == closed.end()) {
            closeLine(slave, visiting, closed);
        }
    }
}

void AffineConstraints::closeLine(GlobalIndex slave_dof,
                                   std::unordered_set<GlobalIndex>& visiting,
                                   std::unordered_set<GlobalIndex>& closed) {
    if (closed.find(slave_dof) != closed.end()) {
        return;  // Already closed
    }

    if (visiting.find(slave_dof) != visiting.end()) {
        // Cycle detected
        if (options_.detect_cycles) {
            // Build cycle path for error message
            std::vector<GlobalIndex> cycle;
            cycle.push_back(slave_dof);
            throw ConstraintCycleException(cycle, __FILE__, __LINE__);
        }
        return;
    }

    auto it = building_lines_.find(slave_dof);
    if (it == building_lines_.end()) {
        return;  // Not constrained
    }

    ConstraintLine& line = it->second;
    visiting.insert(slave_dof);

    // New entries after closure
    std::vector<ConstraintEntry> new_entries;
    double new_inhomogeneity = line.inhomogeneity;

    for (const auto& entry : line.entries) {
        GlobalIndex master = entry.master_dof;
        double weight = entry.weight;

        // Check if master is also constrained
        auto master_it = building_lines_.find(master);
        if (master_it != building_lines_.end()) {
            // Recursively close the master's constraint
            closeLine(master, visiting, closed);

            // Expand: slave = weight * (master's constraint)
            const ConstraintLine& master_line = master_it->second;

            // Add weighted contributions from master's masters
            for (const auto& master_entry : master_line.entries) {
                new_entries.push_back({
                    master_entry.master_dof,
                    weight * master_entry.weight
                });
            }

            // Add weighted master's inhomogeneity
            new_inhomogeneity += weight * master_line.inhomogeneity;
        } else {
            // Master is not constrained, keep entry as-is
            new_entries.push_back(entry);
        }
    }

    // Update the line with closed entries
    line.entries = std::move(new_entries);
    line.inhomogeneity = new_inhomogeneity;

    // Merge any duplicates created by expansion
    line.mergeEntries();

    visiting.erase(slave_dof);
    closed.insert(slave_dof);
}

void AffineConstraints::sortConstraints() {
    // Sort building_lines by slave DOF for deterministic iteration
    // (the map itself is unordered, so we sort when building CSR)
}

void AffineConstraints::mergeAllDuplicates() {
    for (auto& [slave, line] : building_lines_) {
        line.mergeEntries();
    }
}

void AffineConstraints::buildCSRStorage() {
    const std::size_t n_constraints = building_lines_.size();

    // Collect and sort slave DOFs
    slave_dofs_.clear();
    slave_dofs_.reserve(n_constraints);
    for (const auto& [slave, line] : building_lines_) {
        slave_dofs_.push_back(slave);
    }

    // Sort for deterministic order
    std::sort(slave_dofs_.begin(), slave_dofs_.end());

    // Build index map
    slave_to_index_.clear();
    for (std::size_t i = 0; i < slave_dofs_.size(); ++i) {
        slave_to_index_[slave_dofs_[i]] = i;
    }

    // Count total entries
    std::size_t total_entries = 0;
    for (const auto& [slave, line] : building_lines_) {
        total_entries += line.entries.size();
    }

    // Build CSR storage
    entry_offsets_.clear();
    entry_offsets_.reserve(n_constraints + 1);
    entries_.clear();
    entries_.reserve(total_entries);
    inhomogeneities_.clear();
    inhomogeneities_.reserve(n_constraints);

    entry_offsets_.push_back(0);
    for (GlobalIndex slave : slave_dofs_) {
        const ConstraintLine& line = building_lines_.at(slave);

        // Sort entries within line for determinism
        std::vector<ConstraintEntry> sorted_entries = line.entries;
        std::sort(sorted_entries.begin(), sorted_entries.end());

        for (const auto& entry : sorted_entries) {
            entries_.push_back(entry);
        }
        entry_offsets_.push_back(static_cast<GlobalIndex>(entries_.size()));
        inhomogeneities_.push_back(line.inhomogeneity);
    }

    // Clear building storage to free memory
    building_lines_.clear();
}

// ============================================================================
// Query
// ============================================================================

bool AffineConstraints::isConstrained(GlobalIndex dof) const noexcept {
    if (is_closed_) {
        return slave_to_index_.find(dof) != slave_to_index_.end();
    }
    return building_lines_.find(dof) != building_lines_.end();
}

bool AffineConstraints::hasConstrainedDofs(std::span<const GlobalIndex> dofs) const noexcept {
    for (GlobalIndex dof : dofs) {
        if (isConstrained(dof)) {
            return true;
        }
    }
    return false;
}

std::optional<AffineConstraints::ConstraintView>
AffineConstraints::getConstraint(GlobalIndex dof) const {
    if (!is_closed_) {
        // Return view from building storage
        auto it = building_lines_.find(dof);
        if (it == building_lines_.end()) {
            return std::nullopt;
        }
        const ConstraintLine& line = it->second;
        return ConstraintView{
            line.slave_dof,
            std::span<const ConstraintEntry>(line.entries),
            line.inhomogeneity
        };
    }

    auto it = slave_to_index_.find(dof);
    if (it == slave_to_index_.end()) {
        return std::nullopt;
    }

    std::size_t idx = it->second;
    auto begin_offset = static_cast<std::size_t>(entry_offsets_[idx]);
    auto end_offset = static_cast<std::size_t>(entry_offsets_[idx + 1]);

    return ConstraintView{
        slave_dofs_[idx],
        std::span<const ConstraintEntry>(entries_.data() + begin_offset,
                                          end_offset - begin_offset),
        inhomogeneities_[idx]
    };
}

double AffineConstraints::getInhomogeneity(GlobalIndex dof) const noexcept {
    if (!is_closed_) {
        auto it = building_lines_.find(dof);
        if (it != building_lines_.end()) {
            return it->second.inhomogeneity;
        }
        return 0.0;
    }

    auto it = slave_to_index_.find(dof);
    if (it != slave_to_index_.end()) {
        return inhomogeneities_[it->second];
    }
    return 0.0;
}

std::vector<GlobalIndex> AffineConstraints::getConstrainedDofs() const {
    if (is_closed_) {
        return slave_dofs_;
    }

    std::vector<GlobalIndex> result;
    result.reserve(building_lines_.size());
    for (const auto& [slave, line] : building_lines_) {
        result.push_back(slave);
    }
    std::sort(result.begin(), result.end());
    return result;
}

ConstraintStatistics AffineConstraints::getStatistics() const {
    ConstraintStatistics stats;

    if (is_closed_) {
        stats.n_constraints = static_cast<GlobalIndex>(slave_dofs_.size());
        stats.total_entries = static_cast<GlobalIndex>(entries_.size());

        for (std::size_t i = 0; i < slave_dofs_.size(); ++i) {
            auto begin_offset = static_cast<std::size_t>(entry_offsets_[i]);
            auto end_offset = static_cast<std::size_t>(entry_offsets_[i + 1]);
            std::size_t n_entries = end_offset - begin_offset;

            if (n_entries == 0) {
                ++stats.n_dirichlet;
            } else if (n_entries == 1 &&
                       std::abs(entries_[begin_offset].weight - 1.0) < 1e-15) {
                ++stats.n_simple_periodic;
            } else if (n_entries > 1) {
                ++stats.n_multipoint;
            }

            if (std::abs(inhomogeneities_[i]) > 1e-15) {
                ++stats.n_inhomogeneous;
            }
        }
    } else {
        stats.n_constraints = static_cast<GlobalIndex>(building_lines_.size());

        for (const auto& [slave, line] : building_lines_) {
            stats.total_entries += static_cast<GlobalIndex>(line.entries.size());

            if (line.isDirichlet()) {
                ++stats.n_dirichlet;
            } else if (line.isSimplePeriodic()) {
                ++stats.n_simple_periodic;
            } else if (line.entries.size() > 1) {
                ++stats.n_multipoint;
            }

            if (!line.isHomogeneous()) {
                ++stats.n_inhomogeneous;
            }
        }
    }

    if (stats.n_constraints > 0) {
        stats.avg_masters_per_constraint =
            static_cast<double>(stats.total_entries) / static_cast<double>(stats.n_constraints);
    }

    return stats;
}

// ============================================================================
// Constraint application
// ============================================================================

void AffineConstraints::distribute(double* vec, GlobalIndex vec_size) const {
    if (!is_closed_) {
        CONSTRAINT_THROW("Cannot distribute: constraints not closed");
    }

    // Process constraints in reverse topological order to handle chains
    // (but since we've computed closure, order doesn't matter)
    for (std::size_t i = 0; i < slave_dofs_.size(); ++i) {
        GlobalIndex slave = slave_dofs_[i];
        if (slave >= vec_size) continue;

        auto begin_offset = static_cast<std::size_t>(entry_offsets_[i]);
        auto end_offset = static_cast<std::size_t>(entry_offsets_[i + 1]);

        double value = inhomogeneities_[i];
        for (std::size_t j = begin_offset; j < end_offset; ++j) {
            GlobalIndex master = entries_[j].master_dof;
            if (master < vec_size) {
                value += entries_[j].weight * vec[master];
            }
        }
        vec[slave] = value;
    }
}

void AffineConstraints::distributeHomogeneous(double* vec, GlobalIndex vec_size) const
{
    if (!is_closed_) {
        CONSTRAINT_THROW("Cannot distributeHomogeneous: constraints not closed");
    }

    for (std::size_t i = 0; i < slave_dofs_.size(); ++i) {
        GlobalIndex slave = slave_dofs_[i];
        if (slave >= vec_size) continue;

        auto begin_offset = static_cast<std::size_t>(entry_offsets_[i]);
        auto end_offset = static_cast<std::size_t>(entry_offsets_[i + 1]);

        double value = 0.0;
        for (std::size_t j = begin_offset; j < end_offset; ++j) {
            GlobalIndex master = entries_[j].master_dof;
            if (master < vec_size) {
                value += entries_[j].weight * vec[master];
            }
        }
        vec[slave] = value;
    }
}

void AffineConstraints::setConstrainedValues(double* vec, GlobalIndex vec_size) const {
    if (!is_closed_) {
        CONSTRAINT_THROW("Cannot set constrained values: constraints not closed");
    }

    for (std::size_t i = 0; i < slave_dofs_.size(); ++i) {
        GlobalIndex slave = slave_dofs_[i];
        if (slave < vec_size) {
            vec[slave] = inhomogeneities_[i];
        }
    }
}

// ============================================================================
// Inhomogeneity updates
// ============================================================================

void AffineConstraints::updateInhomogeneity(GlobalIndex dof, double value) {
    if (!is_closed_) {
        auto* line = findLine(dof);
        if (!line) {
            CONSTRAINT_THROW_DOF("Constraint not found for DOF", dof);
        }
        line->inhomogeneity = value;
        return;
    }

    auto it = slave_to_index_.find(dof);
    if (it == slave_to_index_.end()) {
        CONSTRAINT_THROW_DOF("Constraint not found for DOF", dof);
    }
    inhomogeneities_[it->second] = value;
}

void AffineConstraints::updateInhomogeneities(std::span<const GlobalIndex> dofs,
                                               std::span<const double> values) {
    if (dofs.size() != values.size()) {
        CONSTRAINT_THROW("DOFs and values arrays must have same size");
    }

    for (std::size_t i = 0; i < dofs.size(); ++i) {
        updateInhomogeneity(dofs[i], values[i]);
    }
}

void AffineConstraints::clearInhomogeneities() {
    if (!is_closed_) {
        for (auto& [slave, line] : building_lines_) {
            line.inhomogeneity = 0.0;
        }
        return;
    }

    std::fill(inhomogeneities_.begin(), inhomogeneities_.end(), 0.0);
}

// ============================================================================
// Validation
// ============================================================================

ValidationResult AffineConstraints::validate() const {
    ValidationResult result;

    // Check for negative DOF indices
    for (const auto& [slave, line] : building_lines_) {
        if (slave < 0) {
            result.valid = false;
            result.error_message = "Negative slave DOF index: " + std::to_string(slave);
            result.problematic_dofs.push_back(slave);
            return result;
        }

        for (const auto& entry : line.entries) {
            if (entry.master_dof < 0) {
                result.valid = false;
                result.error_message = "Negative master DOF index in constraint for DOF " +
                                       std::to_string(slave);
                result.problematic_dofs.push_back(slave);
                return result;
            }
            if (entry.master_dof == slave) {
                result.valid = false;
                result.error_message = "Self-reference in constraint for DOF " +
                                       std::to_string(slave);
                result.problematic_dofs.push_back(slave);
                return result;
            }
        }
    }

    // Check for cycles using DFS
    if (options_.detect_cycles && !is_closed_) {
        std::unordered_set<GlobalIndex> visited;
        std::unordered_set<GlobalIndex> rec_stack;

        std::function<bool(GlobalIndex, std::vector<GlobalIndex>&)> has_cycle =
            [&](GlobalIndex dof, std::vector<GlobalIndex>& path) -> bool {
            visited.insert(dof);
            rec_stack.insert(dof);
            path.push_back(dof);

            auto it = building_lines_.find(dof);
            if (it != building_lines_.end()) {
                for (const auto& entry : it->second.entries) {
                    GlobalIndex master = entry.master_dof;
                    if (building_lines_.find(master) != building_lines_.end()) {
                        if (rec_stack.find(master) != rec_stack.end()) {
                            // Found cycle
                            path.push_back(master);
                            return true;
                        }
                        if (visited.find(master) == visited.end()) {
                            if (has_cycle(master, path)) {
                                return true;
                            }
                        }
                    }
                }
            }

            path.pop_back();
            rec_stack.erase(dof);
            return false;
        };

        for (const auto& [slave, line] : building_lines_) {
            if (visited.find(slave) == visited.end()) {
                std::vector<GlobalIndex> path;
                if (has_cycle(slave, path)) {
                    result.valid = false;
                    result.error_message = "Constraint cycle detected";
                    result.problematic_dofs = path;
                    return result;
                }
            }
        }
    }

    return result;
}

// ============================================================================
// Iteration
// ============================================================================

void AffineConstraints::forEach(std::function<void(const ConstraintView&)> callback) const {
    if (!is_closed_) {
        for (const auto& [slave, line] : building_lines_) {
            ConstraintView view{
                line.slave_dof,
                std::span<const ConstraintEntry>(line.entries),
                line.inhomogeneity
            };
            callback(view);
        }
        return;
    }

    for (std::size_t i = 0; i < slave_dofs_.size(); ++i) {
        auto begin_offset = static_cast<std::size_t>(entry_offsets_[i]);
        auto end_offset = static_cast<std::size_t>(entry_offsets_[i + 1]);

        ConstraintView view{
            slave_dofs_[i],
            std::span<const ConstraintEntry>(entries_.data() + begin_offset,
                                              end_offset - begin_offset),
            inhomogeneities_[i]
        };
        callback(view);
    }
}

// ============================================================================
// Options
// ============================================================================

void AffineConstraints::setOptions(const AffineConstraintsOptions& options) {
    checkNotClosed();
    options_ = options;
}

// ============================================================================
// Internal helpers
// ============================================================================

void AffineConstraints::checkNotClosed() const {
    if (is_closed_) {
        CONSTRAINT_THROW("Cannot modify closed AffineConstraints");
    }
}

void AffineConstraints::checkClosed() const {
    if (!is_closed_) {
        CONSTRAINT_THROW("AffineConstraints must be closed for this operation");
    }
}

ConstraintLine& AffineConstraints::getOrCreateLine(GlobalIndex slave_dof) {
    auto it = building_lines_.find(slave_dof);
    if (it == building_lines_.end()) {
        ConstraintLine line;
        line.slave_dof = slave_dof;
        building_lines_[slave_dof] = std::move(line);
        return building_lines_[slave_dof];
    }
    return it->second;
}

ConstraintLine* AffineConstraints::findLine(GlobalIndex slave_dof) {
    auto it = building_lines_.find(slave_dof);
    if (it != building_lines_.end()) {
        return &it->second;
    }
    return nullptr;
}

const ConstraintLine* AffineConstraints::findLine(GlobalIndex slave_dof) const {
    auto it = building_lines_.find(slave_dof);
    if (it != building_lines_.end()) {
        return &it->second;
    }
    return nullptr;
}

} // namespace constraints
} // namespace FE
} // namespace svmp
