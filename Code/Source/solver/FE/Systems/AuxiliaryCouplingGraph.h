#ifndef SVMP_FE_SYSTEMS_AUXILIARY_COUPLING_GRAPH_H
#define SVMP_FE_SYSTEMS_AUXILIARY_COUPLING_GRAPH_H

/**
 * @file AuxiliaryCouplingGraph.h
 * @brief Sparse nonlocal coupling graph for auxiliary state blocks.
 *
 * Tracks which auxiliary blocks and FE fields couple to each other
 * and through which operators.  Used by the monolithic assembly path
 * to determine sparsity structure and communication patterns.
 *
 * ## Edge types
 *
 * | Type               | Description                                      |
 * |--------------------|--------------------------------------------------|
 * | `AuxToAux`         | Auxiliary block → auxiliary block coupling.       |
 * | `FieldToAux`       | FE field → auxiliary block coupling.              |
 * | `AuxToField`       | Auxiliary block → FE field coupling.              |
 * | `AuxSelf`          | Auxiliary block self-coupling (diagonal block).   |
 *
 * ## Distributed communication
 *
 * For monolithic blocks with distributed scopes (Node, Cell, etc.),
 * the coupling graph provides the communication plan metadata needed
 * for ghost exchange during mixed assembly.
 */

#include "Core/Types.h"
#include "Core/FEException.h"

#include "Systems/AuxiliaryStateTypes.h"
#include "Systems/SystemsExceptions.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

// ---------------------------------------------------------------------------
//  Coupling edge types
// ---------------------------------------------------------------------------

enum class AuxiliaryCouplingType : std::uint8_t {
    /// Auxiliary block self-coupling (diagonal Jacobian block).
    AuxSelf,

    /// Auxiliary block → auxiliary block off-diagonal coupling.
    AuxToAux,

    /// FE field → auxiliary block coupling.
    FieldToAux,

    /// Auxiliary block → FE field coupling.
    AuxToField
};

// ---------------------------------------------------------------------------
//  Coupling edge
// ---------------------------------------------------------------------------

/**
 * @brief An edge in the auxiliary coupling graph.
 */
struct AuxiliaryCouplingEdge {
    AuxiliaryCouplingType type{AuxiliaryCouplingType::AuxSelf};

    /// Source identifier (block name or field name).
    std::string source{};

    /// Target identifier (block name or field name).
    std::string target{};

    /// Name of the operator producing this coupling.
    std::string operator_name{};

    /// Whether this edge requires distributed communication.
    bool requires_communication{false};
};

// ---------------------------------------------------------------------------
//  Coupling graph
// ---------------------------------------------------------------------------

/**
 * @brief Sparse graph of auxiliary coupling relationships.
 *
 * Vertices are auxiliary block names and FE field names.
 * Edges represent coupling through registered operators.
 */
class AuxiliaryCouplingGraph {
public:
    AuxiliaryCouplingGraph() = default;

    // -----------------------------------------------------------------
    //  Edge registration
    // -----------------------------------------------------------------

    /**
     * @brief Add a coupling edge.
     */
    void addEdge(const AuxiliaryCouplingEdge& edge);

    /**
     * @brief Add a self-coupling edge for an auxiliary block.
     */
    void addSelfCoupling(std::string_view block_name,
                          std::string_view operator_name);

    /**
     * @brief Add an auxiliary-to-auxiliary coupling edge.
     */
    void addAuxToAux(std::string_view source_block,
                      std::string_view target_block,
                      std::string_view operator_name);

    /**
     * @brief Add a field-to-auxiliary coupling edge.
     */
    void addFieldToAux(std::string_view field_name,
                        std::string_view aux_block,
                        std::string_view operator_name);

    /**
     * @brief Add an auxiliary-to-field coupling edge.
     */
    void addAuxToField(std::string_view aux_block,
                        std::string_view field_name,
                        std::string_view operator_name);

    // -----------------------------------------------------------------
    //  Queries
    // -----------------------------------------------------------------

    /// Total number of edges.
    [[nodiscard]] std::size_t edgeCount() const noexcept { return edges_.size(); }

    /// All edges.
    [[nodiscard]] const std::vector<AuxiliaryCouplingEdge>& edges() const noexcept
    {
        return edges_;
    }

    /// Edges where the given block is the target.
    [[nodiscard]] std::vector<AuxiliaryCouplingEdge> incomingEdges(
        std::string_view block_name) const;

    /// Edges where the given block is the source.
    [[nodiscard]] std::vector<AuxiliaryCouplingEdge> outgoingEdges(
        std::string_view block_name) const;

    /// All unique auxiliary block names involved as vertices.
    [[nodiscard]] std::vector<std::string> auxiliaryVertices() const;

    /// All unique field names involved as vertices.
    [[nodiscard]] std::vector<std::string> fieldVertices() const;

    /// Whether a block has any coupling to FE fields.
    [[nodiscard]] bool hasCouplingToFields(std::string_view block_name) const;

    /// Whether a block has any coupling to other auxiliary blocks.
    [[nodiscard]] bool hasCouplingToAux(std::string_view block_name) const;

    // -----------------------------------------------------------------
    //  Lifecycle
    // -----------------------------------------------------------------

    void clear();

private:
    std::vector<AuxiliaryCouplingEdge> edges_{};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_AUXILIARY_COUPLING_GRAPH_H
