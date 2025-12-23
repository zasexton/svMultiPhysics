/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "ConstraintsIO.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <stack>

namespace svmp {
namespace FE {
namespace constraints {

// ============================================================================
// DetailedValidationResult
// ============================================================================

std::string DetailedValidationResult::summary() const {
    std::ostringstream oss;

    if (valid) {
        oss << "Constraints are valid.";
    } else {
        oss << "Constraints are INVALID.\n";
        oss << "Errors (" << errors.size() << "):\n";
        for (const auto& e : errors) {
            oss << "  - " << e << "\n";
        }
    }

    if (!warnings.empty()) {
        oss << "Warnings (" << warnings.size() << "):\n";
        for (const auto& w : warnings) {
            oss << "  - " << w << "\n";
        }
    }

    return oss.str();
}

// ============================================================================
// Validation
// ============================================================================

DetailedValidationResult validateConstraintsDetailed(const AffineConstraints& constraints,
                                      const ValidationOptions& options) {
    DetailedValidationResult result;

    if (!constraints.isClosed()) {
        result.addWarning("Constraints are not closed - validation may be incomplete");
    }

    // Check for cycles
    if (options.check_cycles) {
        auto cycles = detectCycles(constraints);
        if (!cycles.empty()) {
            result.cycle_dofs = cycles;
            result.addError("Constraint graph contains cycles involving " +
                           std::to_string(cycles.size()) + " DOFs");
        }
    }

    // Check for missing masters
    if (options.check_missing_masters && options.max_dof_index.has_value()) {
        auto constrained = constraints.getConstrainedDofs();
        GlobalIndex max_dof = *options.max_dof_index;

        for (GlobalIndex slave : constrained) {
            auto constraint = constraints.getConstraint(slave);
            if (!constraint) continue;

            for (const auto& entry : constraint->entries) {
                if (entry.master_dof < 0 || entry.master_dof > max_dof) {
                    result.missing_masters.push_back(entry.master_dof);
                    result.addError("Master DOF " + std::to_string(entry.master_dof) +
                                   " is out of valid range");
                }
            }
        }
    }

    // Check weights
    if (options.check_weights) {
        auto constrained = constraints.getConstrainedDofs();

        for (GlobalIndex slave : constrained) {
            auto constraint = constraints.getConstraint(slave);
            if (!constraint) continue;

            for (const auto& entry : constraint->entries) {
                if (!std::isfinite(entry.weight)) {
                    result.addError("Non-finite weight for DOF " +
                                   std::to_string(slave));
                }
            }
        }
    }

    return result;
}

std::vector<GlobalIndex> detectCycles(const AffineConstraints& constraints) {
    std::vector<GlobalIndex> cycle_dofs;

    // Build adjacency list from constraints
    auto constrained = constraints.getConstrainedDofs();

    std::unordered_map<GlobalIndex, std::vector<GlobalIndex>> adj;
    for (GlobalIndex slave : constrained) {
        auto constraint = constraints.getConstraint(slave);
        if (!constraint) continue;

        for (const auto& entry : constraint->entries) {
            adj[slave].push_back(entry.master_dof);
        }
    }

    // DFS for cycle detection
    std::unordered_set<GlobalIndex> visited;
    std::unordered_set<GlobalIndex> rec_stack;
    std::unordered_set<GlobalIndex> in_cycle;

    std::function<bool(GlobalIndex)> dfs = [&](GlobalIndex node) -> bool {
        visited.insert(node);
        rec_stack.insert(node);

        auto it = adj.find(node);
        if (it != adj.end()) {
            for (GlobalIndex neighbor : it->second) {
                if (rec_stack.find(neighbor) != rec_stack.end()) {
                    // Cycle found
                    in_cycle.insert(node);
                    in_cycle.insert(neighbor);
                    return true;
                }
                if (visited.find(neighbor) == visited.end()) {
                    if (dfs(neighbor)) {
                        in_cycle.insert(node);
                        return true;
                    }
                }
            }
        }

        rec_stack.erase(node);
        return false;
    };

    for (GlobalIndex slave : constrained) {
        if (visited.find(slave) == visited.end()) {
            dfs(slave);
        }
    }

    cycle_dofs.assign(in_cycle.begin(), in_cycle.end());
    std::sort(cycle_dofs.begin(), cycle_dofs.end());
    return cycle_dofs;
}

std::vector<GlobalIndex> findConflicts(const AffineConstraints& constraints) {
    // Conflicts would be detected during constraint addition
    // This function checks for any that slipped through
    std::vector<GlobalIndex> conflicts;

    auto constrained = constraints.getConstrainedDofs();
    std::unordered_set<GlobalIndex> seen;

    for (GlobalIndex dof : constrained) {
        if (seen.find(dof) != seen.end()) {
            conflicts.push_back(dof);
        }
        seen.insert(dof);
    }

    return conflicts;
}

// ============================================================================
// Serialization - Binary
// ============================================================================

namespace {

void writeBinary(std::ostream& out, const AffineConstraints& constraints) {
    // Magic number and version
    const uint32_t magic = 0x434F4E53;  // "CONS"
    const uint32_t version = 1;
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // Write constraint data
    auto constrained = constraints.getConstrainedDofs();
    uint64_t num_constraints = constrained.size();
    out.write(reinterpret_cast<const char*>(&num_constraints), sizeof(num_constraints));

    for (GlobalIndex slave : constrained) {
        auto constraint = constraints.getConstraint(slave);
        if (!constraint) continue;

        out.write(reinterpret_cast<const char*>(&slave), sizeof(slave));

        uint64_t num_entries = constraint->entries.size();
        out.write(reinterpret_cast<const char*>(&num_entries), sizeof(num_entries));

        for (const auto& entry : constraint->entries) {
            out.write(reinterpret_cast<const char*>(&entry.master_dof),
                     sizeof(entry.master_dof));
            out.write(reinterpret_cast<const char*>(&entry.weight),
                     sizeof(entry.weight));
        }

        out.write(reinterpret_cast<const char*>(&constraint->inhomogeneity),
                 sizeof(constraint->inhomogeneity));
    }
}

AffineConstraints readBinary(std::istream& in) {
    uint32_t magic, version;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != 0x434F4E53 || version != 1) {
        CONSTRAINT_THROW("Invalid constraint file format");
    }

    AffineConstraints constraints;

    uint64_t num_constraints;
    in.read(reinterpret_cast<char*>(&num_constraints), sizeof(num_constraints));

    for (uint64_t i = 0; i < num_constraints; ++i) {
        GlobalIndex slave;
        in.read(reinterpret_cast<char*>(&slave), sizeof(slave));

        constraints.addLine(slave);

        uint64_t num_entries;
        in.read(reinterpret_cast<char*>(&num_entries), sizeof(num_entries));

        for (uint64_t j = 0; j < num_entries; ++j) {
            GlobalIndex master;
            double weight;
            in.read(reinterpret_cast<char*>(&master), sizeof(master));
            in.read(reinterpret_cast<char*>(&weight), sizeof(weight));
            constraints.addEntry(slave, master, weight);
        }

        double inhom;
        in.read(reinterpret_cast<char*>(&inhom), sizeof(inhom));
        if (std::abs(inhom) > 1e-15) {
            constraints.setInhomogeneity(slave, inhom);
        }
    }

    return constraints;
}

void writeText(std::ostream& out, const AffineConstraints& constraints, int precision) {
    out << "# AffineConstraints\n";
    out << "# Format: slave_dof : weight1*master1 + weight2*master2 + ... + inhom\n";
    out << std::setprecision(precision);

    auto constrained = constraints.getConstrainedDofs();
    out << "# num_constraints = " << constrained.size() << "\n";

    for (GlobalIndex slave : constrained) {
        auto constraint = constraints.getConstraint(slave);
        if (!constraint) continue;

        out << slave << " :";

        bool first = true;
        for (const auto& entry : constraint->entries) {
            if (!first) out << " +";
            out << " " << entry.weight << "*" << entry.master_dof;
            first = false;
        }

        if (std::abs(constraint->inhomogeneity) > 1e-15) {
            out << " + " << constraint->inhomogeneity;
        }
        out << "\n";
    }
}

AffineConstraints readText(std::istream& in) {
    AffineConstraints constraints;
    std::string line;

    while (std::getline(in, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;

        // Parse: slave : weight*master + weight*master + ... + inhom
        std::istringstream iss(line);
        GlobalIndex slave;
        char colon;

        if (!(iss >> slave >> colon) || colon != ':') continue;

        constraints.addLine(slave);

        double inhom = 0.0;
        std::string token;
        while (iss >> token) {
            if (token == "+") continue;

            // Check if it's weight*master or just a number
            auto star_pos = token.find('*');
            if (star_pos != std::string::npos) {
                double weight = std::stod(token.substr(0, star_pos));
                GlobalIndex master = std::stoll(token.substr(star_pos + 1));
                constraints.addEntry(slave, master, weight);
            } else {
                // Might be inhomogeneity
                try {
                    inhom = std::stod(token);
                } catch (...) {
                    // Ignore parse errors
                }
            }
        }

        if (std::abs(inhom) > 1e-15) {
            constraints.setInhomogeneity(slave, inhom);
        }
    }

    return constraints;
}

}  // anonymous namespace

void serializeConstraints(const AffineConstraints& constraints,
                           std::ostream& out,
                           const SerializationOptions& options) {
    switch (options.format) {
        case SerializationFormat::Binary:
            writeBinary(out, constraints);
            break;
        case SerializationFormat::Text:
            writeText(out, constraints, options.precision);
            break;
        case SerializationFormat::JSON:
            out << constraintsToJson(constraints, options.pretty_print);
            break;
    }
}

AffineConstraints deserializeConstraints(std::istream& in,
                                          const SerializationOptions& options) {
    switch (options.format) {
        case SerializationFormat::Binary:
            return readBinary(in);
        case SerializationFormat::Text:
            return readText(in);
        case SerializationFormat::JSON: {
            std::ostringstream oss;
            oss << in.rdbuf();
            return constraintsFromJson(oss.str());
        }
    }
    return AffineConstraints{};
}

void saveConstraints(const AffineConstraints& constraints,
                      const std::string& filename,
                      const SerializationOptions& options) {
    std::ios_base::openmode mode = std::ios::out;
    if (options.format == SerializationFormat::Binary) {
        mode |= std::ios::binary;
    }

    std::ofstream out(filename, mode);
    if (!out) {
        CONSTRAINT_THROW("Failed to open file for writing: " + filename);
    }

    serializeConstraints(constraints, out, options);
}

AffineConstraints loadConstraints(const std::string& filename,
                                   const SerializationOptions& options) {
    std::ios_base::openmode mode = std::ios::in;
    if (options.format == SerializationFormat::Binary) {
        mode |= std::ios::binary;
    }

    std::ifstream in(filename, mode);
    if (!in) {
        CONSTRAINT_THROW("Failed to open file for reading: " + filename);
    }

    return deserializeConstraints(in, options);
}

// ============================================================================
// JSON Export/Import
// ============================================================================

std::string constraintsToJson(const AffineConstraints& constraints, bool pretty) {
    std::ostringstream oss;
    std::string indent = pretty ? "  " : "";
    std::string newline = pretty ? "\n" : "";

    oss << "{" << newline;
    oss << indent << "\"type\": \"AffineConstraints\"," << newline;
    oss << indent << "\"closed\": " << (constraints.isClosed() ? "true" : "false") << "," << newline;

    auto constrained = constraints.getConstrainedDofs();
    oss << indent << "\"num_constraints\": " << constrained.size() << "," << newline;
    oss << indent << "\"constraints\": [" << newline;

    bool first_constraint = true;
    for (GlobalIndex slave : constrained) {
        auto constraint = constraints.getConstraint(slave);
        if (!constraint) continue;

        if (!first_constraint) oss << "," << newline;
        first_constraint = false;

        oss << indent << indent << "{" << newline;
        oss << indent << indent << indent << "\"slave\": " << slave << "," << newline;
        oss << indent << indent << indent << "\"entries\": [";

        bool first_entry = true;
        for (const auto& entry : constraint->entries) {
            if (!first_entry) oss << ", ";
            first_entry = false;
            oss << "{\"master\": " << entry.master_dof
                << ", \"weight\": " << std::setprecision(15) << entry.weight << "}";
        }

        oss << "]," << newline;
        oss << indent << indent << indent << "\"inhomogeneity\": "
            << std::setprecision(15) << constraint->inhomogeneity << newline;
        oss << indent << indent << "}";
    }

    oss << newline << indent << "]" << newline;
    oss << "}" << newline;

    return oss.str();
}

AffineConstraints constraintsFromJson(const std::string& json) {
    // Simple JSON parser for our specific format
    AffineConstraints constraints;

    // Find constraints array
    auto constraints_pos = json.find("\"constraints\"");
    if (constraints_pos == std::string::npos) {
        return constraints;
    }

    // Parse each constraint object
    std::size_t pos = json.find('[', constraints_pos);
    while ((pos = json.find("\"slave\"", pos)) != std::string::npos) {
        // Find slave value
        std::size_t colon = json.find(':', pos);
        std::size_t comma = json.find(',', colon);
        GlobalIndex slave = std::stoll(json.substr(colon + 1, comma - colon - 1));

        constraints.addLine(slave);

        // Find entries array
        std::size_t entries_pos = json.find("\"entries\"", pos);
        std::size_t entries_start = json.find('[', entries_pos);
        std::size_t entries_end = json.find(']', entries_start);

        std::string entries_str = json.substr(entries_start, entries_end - entries_start + 1);

        // Parse each entry
        std::size_t entry_pos = 0;
        while ((entry_pos = entries_str.find("\"master\"", entry_pos)) != std::string::npos) {
            std::size_t m_colon = entries_str.find(':', entry_pos);
            std::size_t m_comma = entries_str.find(',', m_colon);
            GlobalIndex master = std::stoll(entries_str.substr(m_colon + 1, m_comma - m_colon - 1));

            std::size_t w_pos = entries_str.find("\"weight\"", m_comma);
            std::size_t w_colon = entries_str.find(':', w_pos);
            std::size_t w_end = entries_str.find_first_of(",}", w_colon);
            double weight = std::stod(entries_str.substr(w_colon + 1, w_end - w_colon - 1));

            constraints.addEntry(slave, master, weight);
            entry_pos = w_end;
        }

        // Find inhomogeneity
        std::size_t inhom_pos = json.find("\"inhomogeneity\"", entries_end);
        if (inhom_pos != std::string::npos && inhom_pos < json.find('}', entries_end) + 50) {
            std::size_t i_colon = json.find(':', inhom_pos);
            std::size_t i_end = json.find_first_of(",}\n", i_colon);
            double inhom = std::stod(json.substr(i_colon + 1, i_end - i_colon - 1));
            if (std::abs(inhom) > 1e-15) {
                constraints.setInhomogeneity(slave, inhom);
            }
        }

        pos = entries_end;
    }

    return constraints;
}

// ============================================================================
// DOT Graph Export
// ============================================================================

void constraintsToDot(const AffineConstraints& constraints,
                       std::ostream& out,
                       const DotExportOptions& options) {
    out << "digraph " << options.graph_name << " {\n";
    out << "  rankdir=LR;\n";
    out << "  node [shape=circle];\n";

    auto constrained = constraints.getConstrainedDofs();
    std::unordered_set<GlobalIndex> slave_set(constrained.begin(), constrained.end());
    std::unordered_set<GlobalIndex> master_set;

    // Collect all masters
    for (GlobalIndex slave : constrained) {
        auto constraint = constraints.getConstraint(slave);
        if (!constraint) continue;

        for (const auto& entry : constraint->entries) {
            if (slave_set.find(entry.master_dof) == slave_set.end()) {
                master_set.insert(entry.master_dof);
            }
        }
    }

    // Style slave nodes
    out << "\n  // Slave DOFs\n";
    for (GlobalIndex slave : constrained) {
        out << "  " << slave << " [style=filled, fillcolor=\""
            << options.slave_color << "\"];\n";
    }

    // Style master-only nodes
    out << "\n  // Master DOFs (not constrained)\n";
    for (GlobalIndex master : master_set) {
        out << "  " << master << " [style=filled, fillcolor=\""
            << options.master_color << "\"];\n";
    }

    // Add edges
    out << "\n  // Constraint edges\n";
    for (GlobalIndex slave : constrained) {
        auto constraint = constraints.getConstraint(slave);
        if (!constraint) continue;

        for (const auto& entry : constraint->entries) {
            if (std::abs(entry.weight) < options.min_weight_to_show) continue;

            out << "  " << slave << " -> " << entry.master_dof;

            if (options.show_weights) {
                out << " [label=\"" << std::setprecision(4) << entry.weight << "\"]";
            }

            out << ";\n";
        }

        // Add inhomogeneity as a special node if present
        if (options.show_inhomogeneity &&
            std::abs(constraint->inhomogeneity) > 1e-15) {
            out << "  inhom_" << slave << " [label=\""
                << std::setprecision(4) << constraint->inhomogeneity
                << "\", shape=box, style=dashed];\n";
            out << "  " << slave << " -> inhom_" << slave << " [style=dashed];\n";
        }
    }

    out << "}\n";
}

void saveConstraintsDot(const AffineConstraints& constraints,
                         const std::string& filename,
                         const DotExportOptions& options) {
    std::ofstream out(filename);
    if (!out) {
        CONSTRAINT_THROW("Failed to open file for writing: " + filename);
    }
    constraintsToDot(constraints, out, options);
}

std::string constraintsToDotString(const AffineConstraints& constraints,
                                    const DotExportOptions& options) {
    std::ostringstream oss;
    constraintsToDot(constraints, oss, options);
    return oss.str();
}

// ============================================================================
// Statistics
// ============================================================================

DetailedConstraintStatistics computeDetailedStatistics(const AffineConstraints& constraints) {
    DetailedConstraintStatistics stats;

    auto constrained = constraints.getConstrainedDofs();
    stats.num_constraints = static_cast<GlobalIndex>(constrained.size());

    if (constrained.empty()) {
        return stats;
    }

    stats.min_slave_dof = constrained[0];
    stats.max_slave_dof = constrained[0];
    stats.min_master_dof = std::numeric_limits<GlobalIndex>::max();
    stats.max_master_dof = 0;
    stats.min_weight = std::numeric_limits<double>::max();
    stats.max_weight = 0.0;

    GlobalIndex total_masters = 0;

    for (GlobalIndex slave : constrained) {
        stats.min_slave_dof = std::min(stats.min_slave_dof, slave);
        stats.max_slave_dof = std::max(stats.max_slave_dof, slave);

        auto constraint = constraints.getConstraint(slave);
        if (!constraint) continue;

        GlobalIndex num_masters = static_cast<GlobalIndex>(constraint->entries.size());
        total_masters += num_masters;
        stats.max_masters_per_slave = std::max(stats.max_masters_per_slave, num_masters);
        stats.num_entries += num_masters;

        if (std::abs(constraint->inhomogeneity) > 1e-15) {
            stats.num_inhomogeneous++;
        }

        // Check for identity constraint (u_s = u_m)
        if (constraint->entries.size() == 1 &&
            std::abs(constraint->entries[0].weight - 1.0) < 1e-14 &&
            std::abs(constraint->inhomogeneity) < 1e-15) {
            stats.num_identity++;
        }

        for (const auto& entry : constraint->entries) {
            stats.min_master_dof = std::min(stats.min_master_dof, entry.master_dof);
            stats.max_master_dof = std::max(stats.max_master_dof, entry.master_dof);

            double abs_weight = std::abs(entry.weight);
            if (abs_weight > 1e-15) {
                stats.min_weight = std::min(stats.min_weight, abs_weight);
            }
            stats.max_weight = std::max(stats.max_weight, abs_weight);
        }
    }

    if (stats.num_constraints > 0) {
        stats.avg_masters_per_slave =
            static_cast<double>(total_masters) / static_cast<double>(stats.num_constraints);
    }

    return stats;
}

void printConstraintSummary(const AffineConstraints& constraints, std::ostream& out) {
    auto stats = computeDetailedStatistics(constraints);

    out << "=== Constraint Summary ===\n";
    out << "Closed: " << (constraints.isClosed() ? "yes" : "no") << "\n";
    out << "Number of constraints: " << stats.num_constraints << "\n";
    out << "Total entries: " << stats.num_entries << "\n";
    out << "Avg masters per slave: " << std::fixed << std::setprecision(2)
        << stats.avg_masters_per_slave << "\n";
    out << "Max masters per slave: " << stats.max_masters_per_slave << "\n";
    out << "Inhomogeneous constraints: " << stats.num_inhomogeneous << "\n";
    out << "Identity constraints: " << stats.num_identity << "\n";
    out << "Slave DOF range: [" << stats.min_slave_dof << ", "
        << stats.max_slave_dof << "]\n";
    out << "Master DOF range: [" << stats.min_master_dof << ", "
        << stats.max_master_dof << "]\n";
    out << "Weight range: [" << std::scientific << stats.min_weight << ", "
        << stats.max_weight << "]\n";
}

void printConstraintDetails(const AffineConstraints& constraints,
                             std::ostream& out,
                             int max_constraints) {
    auto constrained = constraints.getConstrainedDofs();
    int count = 0;

    out << "=== Constraint Details ===\n";

    for (GlobalIndex slave : constrained) {
        if (max_constraints >= 0 && count >= max_constraints) {
            out << "... and " << (constrained.size() - static_cast<std::size_t>(count))
                << " more constraints\n";
            break;
        }

        out << constraintToString(slave, constraints) << "\n";
        ++count;
    }
}

// ============================================================================
// Comparison
// ============================================================================

ComparisonResult compareConstraints(const AffineConstraints& a,
                                     const AffineConstraints& b,
                                     double tolerance) {
    ComparisonResult result;

    auto dofs_a = a.getConstrainedDofs();
    auto dofs_b = b.getConstrainedDofs();

    std::unordered_set<GlobalIndex> set_a(dofs_a.begin(), dofs_a.end());
    std::unordered_set<GlobalIndex> set_b(dofs_b.begin(), dofs_b.end());

    // Check for DOFs only in A
    for (GlobalIndex dof : dofs_a) {
        if (set_b.find(dof) == set_b.end()) {
            result.different_dofs.push_back(dof);
            result.differences.push_back("DOF " + std::to_string(dof) +
                                         " constrained in A but not in B");
            result.identical = false;
            result.equivalent = false;
        }
    }

    // Check for DOFs only in B
    for (GlobalIndex dof : dofs_b) {
        if (set_a.find(dof) == set_a.end()) {
            result.different_dofs.push_back(dof);
            result.differences.push_back("DOF " + std::to_string(dof) +
                                         " constrained in B but not in A");
            result.identical = false;
            result.equivalent = false;
        }
    }

    // Compare shared DOFs
    for (GlobalIndex dof : dofs_a) {
        if (set_b.find(dof) == set_b.end()) continue;

        auto ca = a.getConstraint(dof);
        auto cb = b.getConstraint(dof);

        if (!ca || !cb) continue;

        // Compare inhomogeneity
        if (std::abs(ca->inhomogeneity - cb->inhomogeneity) > tolerance) {
            result.different_dofs.push_back(dof);
            result.differences.push_back("DOF " + std::to_string(dof) +
                                         " has different inhomogeneity");
            result.identical = false;
            result.equivalent = false;
        }

        // Compare entries
        if (ca->entries.size() != cb->entries.size()) {
            result.different_dofs.push_back(dof);
            result.differences.push_back("DOF " + std::to_string(dof) +
                                         " has different number of masters");
            result.identical = false;
            result.equivalent = false; // Structure mismatch
        } else {
            // Entries are sorted by master_dof in AffineConstraints::close()
            for (std::size_t i = 0; i < ca->entries.size(); ++i) {
                const auto& ea = ca->entries[i];
                const auto& eb = cb->entries[i];

                if (ea.master_dof != eb.master_dof) {
                    result.different_dofs.push_back(dof);
                    result.differences.push_back("DOF " + std::to_string(dof) +
                                                 " has different master set");
                    result.identical = false;
                    result.equivalent = false;
                    break;
                }

                if (std::abs(ea.weight - eb.weight) > tolerance) {
                    result.different_dofs.push_back(dof);
                    result.differences.push_back("DOF " + std::to_string(dof) +
                                                 " has different weight for master " +
                                                 std::to_string(ea.master_dof));
                    result.identical = false;
                    // Equivalence might still hold if weights are close, but here we use tolerance
                    result.equivalent = false;
                    break;
                }
            }
        }
    }

    return result;
}

// ============================================================================
// Debugging
// ============================================================================

std::string constraintToString(GlobalIndex slave_dof,
                                const AffineConstraints& constraints) {
    auto constraint = constraints.getConstraint(slave_dof);
    if (!constraint) {
        return "DOF " + std::to_string(slave_dof) + " is not constrained";
    }

    std::ostringstream oss;
    oss << "u[" << slave_dof << "] = ";

    bool first = true;
    for (const auto& entry : constraint->entries) {
        if (!first) {
            oss << (entry.weight >= 0 ? " + " : " - ");
            oss << std::abs(entry.weight);
        } else {
            oss << entry.weight;
        }
        oss << " * u[" << entry.master_dof << "]";
        first = false;
    }

    if (std::abs(constraint->inhomogeneity) > 1e-15) {
        oss << (constraint->inhomogeneity >= 0 ? " + " : " - ");
        oss << std::abs(constraint->inhomogeneity);
    }

    return oss.str();
}

std::string traceConstraintChain(GlobalIndex dof,
                                  const AffineConstraints& constraints) {
    std::ostringstream oss;
    std::unordered_set<GlobalIndex> visited;
    std::vector<GlobalIndex> chain;

    std::function<void(GlobalIndex, int)> trace = [&](GlobalIndex d, int depth) {
        std::string indent(static_cast<std::size_t>(depth * 2), ' ');

        if (visited.find(d) != visited.end()) {
            oss << indent << "u[" << d << "] -> CYCLE DETECTED\n";
            return;
        }

        visited.insert(d);

        auto constraint = constraints.getConstraint(d);
        if (!constraint) {
            oss << indent << "u[" << d << "] (unconstrained)\n";
            return;
        }

        oss << indent << constraintToString(d, constraints) << "\n";

        for (const auto& entry : constraint->entries) {
            trace(entry.master_dof, depth + 1);
        }
    };

    oss << "Constraint chain for DOF " << dof << ":\n";
    trace(dof, 0);

    return oss.str();
}

} // namespace constraints
} // namespace FE
} // namespace svmp
