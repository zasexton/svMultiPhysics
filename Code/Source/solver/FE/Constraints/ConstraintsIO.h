/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_CONSTRAINTSIO_H
#define SVMP_FE_CONSTRAINTS_CONSTRAINTSIO_H

/**
 * @file ConstraintsIO.h
 * @brief Serialization, validation, and visualization for constraints
 *
 * ConstraintsIO provides utilities for:
 * - Serialization/deserialization of AffineConstraints
 * - Validation (cycle detection, missing masters, conflicts)
 * - Visualization (DOT graph output, JSON export)
 * - Debugging and diagnostics
 *
 * Supported formats:
 * - Binary: Efficient for checkpointing
 * - Text: Human-readable, debugging
 * - JSON: Interoperability, visualization tools
 * - DOT: Graphviz visualization of constraint graph
 *
 * @see AffineConstraints for constraint storage
 */

#include "AffineConstraints.h"
#include "Core/Types.h"
#include "Core/FEException.h"

#include <vector>
#include <string>
#include <memory>
#include <ostream>
#include <istream>
#include <functional>
#include <optional>

namespace svmp {
namespace FE {
namespace constraints {

// ============================================================================
// Validation
// ============================================================================

/**
 * @brief Detailed result of constraint validation
 *
 * This extends the basic ValidationResult from AffineConstraints.h with
 * more detailed tracking of specific issues found during validation.
 */
struct DetailedValidationResult {
    bool valid{true};                          ///< Overall validity
    std::vector<std::string> errors;           ///< Error messages
    std::vector<std::string> warnings;         ///< Warning messages

    // Specific issues found
    std::vector<GlobalIndex> cycle_dofs;       ///< DOFs involved in cycles
    std::vector<GlobalIndex> missing_masters;  ///< Masters that don't exist
    std::vector<GlobalIndex> conflict_dofs;    ///< DOFs with conflicting constraints

    /**
     * @brief Add an error
     */
    void addError(const std::string& msg) {
        errors.push_back(msg);
        valid = false;
    }

    /**
     * @brief Add a warning
     */
    void addWarning(const std::string& msg) {
        warnings.push_back(msg);
    }

    /**
     * @brief Get formatted summary
     */
    [[nodiscard]] std::string summary() const;
};

/**
 * @brief Options for constraint validation
 */
struct ValidationOptions {
    bool check_cycles{true};                   ///< Check for circular dependencies
    bool check_missing_masters{true};          ///< Check for undefined master DOFs
    bool check_conflicts{true};                ///< Check for conflicting constraints
    bool check_weights{true};                  ///< Check weight validity
    double weight_tolerance{1e-14};            ///< Tolerance for zero weights
    std::optional<GlobalIndex> max_dof_index;  ///< Max valid DOF index (if known)
};

/**
 * @brief Validate AffineConstraints
 *
 * @param constraints Constraints to validate
 * @param options Validation options
 * @return Validation result
 */
[[nodiscard]] DetailedValidationResult validateConstraintsDetailed(
    const AffineConstraints& constraints,
    const ValidationOptions& options = {});

/**
 * @brief Detect cycles in constraint graph
 *
 * Uses depth-first search to find circular dependencies.
 *
 * @param constraints Constraints to check
 * @return Vector of DOFs involved in cycles (empty if none)
 */
[[nodiscard]] std::vector<GlobalIndex> detectCycles(
    const AffineConstraints& constraints);

/**
 * @brief Find conflicting constraints
 *
 * A conflict occurs when a DOF is constrained multiple times
 * with inconsistent definitions.
 *
 * @param constraints Constraints to check
 * @return Vector of DOFs with conflicts
 */
[[nodiscard]] std::vector<GlobalIndex> findConflicts(
    const AffineConstraints& constraints);

// ============================================================================
// Serialization
// ============================================================================

/**
 * @brief Serialization format
 */
enum class SerializationFormat {
    Binary,   ///< Compact binary format
    Text,     ///< Human-readable text
    JSON      ///< JSON format
};

/**
 * @brief Options for serialization
 */
struct SerializationOptions {
    SerializationFormat format{SerializationFormat::Binary};
    bool include_metadata{true};               ///< Include constraint info
    int precision{15};                         ///< Floating-point precision (text/JSON)
    bool pretty_print{true};                   ///< Pretty-print JSON
};

/**
 * @brief Serialize AffineConstraints to stream
 *
 * @param constraints Constraints to serialize
 * @param out Output stream
 * @param options Serialization options
 */
void serializeConstraints(const AffineConstraints& constraints,
                           std::ostream& out,
                           const SerializationOptions& options = {});

/**
 * @brief Deserialize AffineConstraints from stream
 *
 * @param in Input stream
 * @param options Serialization options
 * @return Deserialized constraints
 */
[[nodiscard]] AffineConstraints deserializeConstraints(
    std::istream& in,
    const SerializationOptions& options = {});

/**
 * @brief Serialize to file
 */
void saveConstraints(const AffineConstraints& constraints,
                      const std::string& filename,
                      const SerializationOptions& options = {});

/**
 * @brief Deserialize from file
 */
[[nodiscard]] AffineConstraints loadConstraints(
    const std::string& filename,
    const SerializationOptions& options = {});

// ============================================================================
// JSON Export/Import
// ============================================================================

/**
 * @brief Export constraints to JSON string
 *
 * @param constraints Constraints to export
 * @param pretty Pretty-print output
 * @return JSON string
 */
[[nodiscard]] std::string constraintsToJson(
    const AffineConstraints& constraints,
    bool pretty = true);

/**
 * @brief Import constraints from JSON string
 *
 * @param json JSON string
 * @return Deserialized constraints
 */
[[nodiscard]] AffineConstraints constraintsFromJson(const std::string& json);

// ============================================================================
// DOT Graph Export
// ============================================================================

/**
 * @brief Options for DOT graph export
 */
struct DotExportOptions {
    std::string graph_name{"constraints"};     ///< Graph name
    bool show_weights{true};                   ///< Show edge weights
    bool show_inhomogeneity{true};             ///< Show inhomogeneities
    bool cluster_by_type{false};               ///< Group constraints by type
    std::string slave_color{"red"};            ///< Color for slave nodes
    std::string master_color{"blue"};          ///< Color for master nodes
    std::string edge_color{"black"};           ///< Color for edges
    double min_weight_to_show{1e-10};          ///< Minimum weight to display
};

/**
 * @brief Export constraint graph to DOT format
 *
 * Creates a directed graph where:
 * - Nodes are DOFs
 * - Edges go from slave to master with weight labels
 *
 * @param constraints Constraints to export
 * @param out Output stream
 * @param options DOT options
 */
void constraintsToDot(const AffineConstraints& constraints,
                       std::ostream& out,
                       const DotExportOptions& options = {});

/**
 * @brief Export constraint graph to DOT file
 */
void saveConstraintsDot(const AffineConstraints& constraints,
                         const std::string& filename,
                         const DotExportOptions& options = {});

/**
 * @brief Generate DOT string
 */
[[nodiscard]] std::string constraintsToDotString(
    const AffineConstraints& constraints,
    const DotExportOptions& options = {});

// ============================================================================
// Diagnostics and Statistics
// ============================================================================

/**
 * @brief Detailed constraint statistics for I/O and diagnostics
 *
 * This extends the basic ConstraintStatistics from AffineConstraints.h
 * with additional detailed information for analysis and debugging.
 */
struct DetailedConstraintStatistics {
    GlobalIndex num_constraints{0};            ///< Total constraints
    GlobalIndex num_entries{0};                ///< Total master entries
    GlobalIndex max_masters_per_slave{0};      ///< Max masters for any slave
    double avg_masters_per_slave{0.0};         ///< Average masters per slave
    GlobalIndex num_inhomogeneous{0};          ///< Constraints with b != 0
    GlobalIndex num_identity{0};               ///< u_s = u_m constraints
    double min_weight{0.0};                    ///< Minimum weight magnitude
    double max_weight{0.0};                    ///< Maximum weight magnitude
    GlobalIndex min_slave_dof{0};              ///< Minimum slave DOF index
    GlobalIndex max_slave_dof{0};              ///< Maximum slave DOF index
    GlobalIndex min_master_dof{0};             ///< Minimum master DOF index
    GlobalIndex max_master_dof{0};             ///< Maximum master DOF index
};

/**
 * @brief Compute detailed constraint statistics
 *
 * @param constraints Constraints to analyze
 * @return Detailed statistics
 */
[[nodiscard]] DetailedConstraintStatistics computeDetailedStatistics(
    const AffineConstraints& constraints);

/**
 * @brief Print constraint summary to stream
 */
void printConstraintSummary(const AffineConstraints& constraints,
                             std::ostream& out);

/**
 * @brief Print detailed constraint listing
 *
 * @param constraints Constraints to print
 * @param out Output stream
 * @param max_constraints Maximum constraints to print (-1 = all)
 */
void printConstraintDetails(const AffineConstraints& constraints,
                             std::ostream& out,
                             int max_constraints = -1);

// ============================================================================
// Comparison
// ============================================================================

/**
 * @brief Result of constraint comparison
 */
struct ComparisonResult {
    bool identical{true};                      ///< Exactly identical
    bool equivalent{true};                     ///< Mathematically equivalent
    std::vector<GlobalIndex> different_dofs;   ///< DOFs with differences
    std::vector<std::string> differences;      ///< Description of differences
};

/**
 * @brief Compare two constraint sets
 *
 * @param a First constraint set
 * @param b Second constraint set
 * @param tolerance Tolerance for numerical comparison
 * @return Comparison result
 */
[[nodiscard]] ComparisonResult compareConstraints(
    const AffineConstraints& a,
    const AffineConstraints& b,
    double tolerance = 1e-14);

// ============================================================================
// Debugging utilities
// ============================================================================

/**
 * @brief Get string representation of a single constraint
 */
[[nodiscard]] std::string constraintToString(
    GlobalIndex slave_dof,
    const AffineConstraints& constraints);

/**
 * @brief Trace constraint chain for a DOF
 *
 * Shows the chain of constraints that leads to the final expression.
 *
 * @param dof DOF to trace
 * @param constraints Constraints (should be unclosed for full chain)
 * @return String representation of chain
 */
[[nodiscard]] std::string traceConstraintChain(
    GlobalIndex dof,
    const AffineConstraints& constraints);

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_CONSTRAINTSIO_H
