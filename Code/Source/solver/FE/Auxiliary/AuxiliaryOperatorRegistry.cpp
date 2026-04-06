#include "Auxiliary/AuxiliaryOperatorRegistry.h"

namespace svmp {
namespace FE {
namespace systems {

// ---------------------------------------------------------------------------
//  Operator registration
// ---------------------------------------------------------------------------

void AuxiliaryOperatorRegistry::registerOperator(
    const AuxiliaryOperatorDescriptor& desc)
{
    FE_THROW_IF(desc.name.empty(), InvalidArgumentException,
                "AuxiliaryOperatorRegistry::registerOperator: empty name");
    FE_THROW_IF(op_name_to_index_.count(desc.name) != 0u, InvalidArgumentException,
                "AuxiliaryOperatorRegistry::registerOperator: duplicate '" +
                    desc.name + "'");

    const auto idx = operators_.size();
    operators_.push_back(desc);
    op_name_to_index_.emplace(desc.name, idx);

    // Update coupling graph.
    AuxiliaryCouplingEdge edge;
    edge.type = desc.coupling_type;
    edge.source = desc.source_name;
    edge.target = desc.target_name;
    edge.operator_name = desc.name;
    coupling_graph_.addEdge(edge);
}

bool AuxiliaryOperatorRegistry::hasOperator(std::string_view name) const noexcept
{
    return op_name_to_index_.find(std::string(name)) != op_name_to_index_.end();
}

const AuxiliaryOperatorDescriptor& AuxiliaryOperatorRegistry::getOperator(
    std::string_view name) const
{
    auto it = op_name_to_index_.find(std::string(name));
    FE_THROW_IF(it == op_name_to_index_.end(), InvalidArgumentException,
                "AuxiliaryOperatorRegistry: unknown operator '" +
                    std::string(name) + "'");
    return operators_[it->second];
}

std::vector<std::string> AuxiliaryOperatorRegistry::operatorNames() const
{
    std::vector<std::string> names;
    names.reserve(operators_.size());
    for (const auto& op : operators_) {
        names.push_back(op.name);
    }
    return names;
}

// ---------------------------------------------------------------------------
//  Monolithic unknown registration
// ---------------------------------------------------------------------------

void AuxiliaryOperatorRegistry::registerMonolithicUnknowns(
    std::string_view name,
    std::size_t entity_count,
    int stride,
    AuxiliaryStateScope scope)
{
    FE_THROW_IF(layout_finalized_, InvalidStateException,
                "AuxiliaryOperatorRegistry::registerMonolithicUnknowns: "
                "layout already finalized");

    AuxiliaryBlockUnknownLayout blk;
    blk.name = std::string(name);
    blk.entity_count = entity_count;
    blk.stride = stride;
    blk.n_unknowns = entity_count * static_cast<std::size_t>(stride);
    blk.scope = scope;
    // Offset will be computed during finalizeLayout().
    aux_layout_.blocks.push_back(std::move(blk));
}

// ---------------------------------------------------------------------------
//  Layout composition
// ---------------------------------------------------------------------------

void AuxiliaryOperatorRegistry::finalizeLayout()
{
    std::size_t offset = 0;
    for (auto& blk : aux_layout_.blocks) {
        blk.offset = offset;
        offset += blk.n_unknowns;
    }
    aux_layout_.total_aux_unknowns = offset;
    layout_finalized_ = true;
}

MixedSystemLayout AuxiliaryOperatorRegistry::composeMixedLayout(
    std::size_t n_field_unknowns) const
{
    MixedSystemLayout layout;
    layout.n_field_unknowns = n_field_unknowns;
    layout.n_aux_unknowns = aux_layout_.total_aux_unknowns;
    layout.total_unknowns = n_field_unknowns + aux_layout_.total_aux_unknowns;

    layout.aux_layout = aux_layout_;
    layout.aux_layout.mixed_system_offset = n_field_unknowns;

    return layout;
}

// ---------------------------------------------------------------------------
//  Solver-facing metadata
// ---------------------------------------------------------------------------

void AuxiliaryOperatorRegistry::setBlockSolverMetadata(
    std::string_view block_name, AuxiliaryBlockSolverMetadata meta)
{
    block_solver_meta_[std::string(block_name)] = std::move(meta);
}

const AuxiliaryBlockSolverMetadata* AuxiliaryOperatorRegistry::getBlockSolverMetadata(
    std::string_view block_name) const
{
    auto it = block_solver_meta_.find(std::string(block_name));
    return (it != block_solver_meta_.end()) ? &it->second : nullptr;
}

AuxiliaryCouplingDiagnostics AuxiliaryOperatorRegistry::computeCouplingDiagnostics() const
{
    AuxiliaryCouplingDiagnostics diag;
    const auto& edges = coupling_graph_.edges();
    diag.n_coupling_edges = edges.size();

    std::unordered_map<std::string, std::size_t> degree;

    for (const auto& e : edges) {
        switch (e.type) {
            case AuxiliaryCouplingType::FieldToAux:
                diag.n_field_to_aux++;
                break;
            case AuxiliaryCouplingType::AuxToField:
                diag.n_aux_to_field++;
                break;
            case AuxiliaryCouplingType::AuxToAux:
                diag.n_aux_to_aux++;
                break;
            default:
                break;
        }
        degree[e.source]++;
        degree[e.target]++;
    }

    for (const auto& [name, deg] : degree) {
        diag.max_coupling_degree = std::max(diag.max_coupling_degree, deg);
    }

    // Acyclicity check via DFS on aux-to-aux edges.
    if (diag.n_aux_to_aux == 0) {
        diag.is_acyclic = true;
    } else {
        // Build adjacency list for aux-to-aux edges.
        std::unordered_map<std::string, std::vector<std::string>> adj;
        for (const auto& e : edges) {
            if (e.type == AuxiliaryCouplingType::AuxToAux) {
                adj[e.source].push_back(e.target);
            }
        }
        // DFS cycle detection.
        enum class Color { White, Gray, Black };
        std::unordered_map<std::string, Color> color;
        for (const auto& [node, _] : adj) {
            color[node] = Color::White;
            for (const auto& tgt : _) color[tgt] = Color::White;
        }
        bool has_cycle = false;
        std::function<void(const std::string&)> dfs = [&](const std::string& u) {
            if (has_cycle) return;
            color[u] = Color::Gray;
            auto it = adj.find(u);
            if (it != adj.end()) {
                for (const auto& v : it->second) {
                    if (color[v] == Color::Gray) { has_cycle = true; return; }
                    if (color[v] == Color::White) dfs(v);
                }
            }
            color[u] = Color::Black;
        };
        for (const auto& [node, c] : color) {
            if (c == Color::White) dfs(node);
        }
        diag.is_acyclic = !has_cycle;
        if (has_cycle) {
            diag.messages.push_back("Cycle detected in aux-to-aux coupling graph");
        }
    }

    if (diag.n_coupling_edges > 0) {
        diag.messages.push_back(
            "Coupling: " + std::to_string(diag.n_field_to_aux) + " field→aux, " +
            std::to_string(diag.n_aux_to_field) + " aux→field, " +
            std::to_string(diag.n_aux_to_aux) + " aux→aux");
    }

    return diag;
}

// ---------------------------------------------------------------------------
//  Lifecycle
// ---------------------------------------------------------------------------

void AuxiliaryOperatorRegistry::clear()
{
    operators_.clear();
    op_name_to_index_.clear();
    coupling_graph_.clear();
    aux_layout_ = {};
    layout_finalized_ = false;
    block_solver_meta_.clear();
}

} // namespace systems
} // namespace FE
} // namespace svmp
