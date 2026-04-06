#include "Auxiliary/AuxiliaryCouplingGraph.h"

#include <algorithm>
#include <unordered_set>

namespace svmp {
namespace FE {
namespace systems {

void AuxiliaryCouplingGraph::addEdge(const AuxiliaryCouplingEdge& edge)
{
    edges_.push_back(edge);
}

void AuxiliaryCouplingGraph::addSelfCoupling(
    std::string_view block_name, std::string_view operator_name)
{
    AuxiliaryCouplingEdge e;
    e.type = AuxiliaryCouplingType::AuxSelf;
    e.source = std::string(block_name);
    e.target = std::string(block_name);
    e.operator_name = std::string(operator_name);
    edges_.push_back(std::move(e));
}

void AuxiliaryCouplingGraph::addAuxToAux(
    std::string_view source_block, std::string_view target_block,
    std::string_view operator_name)
{
    AuxiliaryCouplingEdge e;
    e.type = AuxiliaryCouplingType::AuxToAux;
    e.source = std::string(source_block);
    e.target = std::string(target_block);
    e.operator_name = std::string(operator_name);
    edges_.push_back(std::move(e));
}

void AuxiliaryCouplingGraph::addFieldToAux(
    std::string_view field_name, std::string_view aux_block,
    std::string_view operator_name)
{
    AuxiliaryCouplingEdge e;
    e.type = AuxiliaryCouplingType::FieldToAux;
    e.source = std::string(field_name);
    e.target = std::string(aux_block);
    e.operator_name = std::string(operator_name);
    edges_.push_back(std::move(e));
}

void AuxiliaryCouplingGraph::addAuxToField(
    std::string_view aux_block, std::string_view field_name,
    std::string_view operator_name)
{
    AuxiliaryCouplingEdge e;
    e.type = AuxiliaryCouplingType::AuxToField;
    e.source = std::string(aux_block);
    e.target = std::string(field_name);
    e.operator_name = std::string(operator_name);
    edges_.push_back(std::move(e));
}

std::vector<AuxiliaryCouplingEdge> AuxiliaryCouplingGraph::incomingEdges(
    std::string_view block_name) const
{
    std::vector<AuxiliaryCouplingEdge> result;
    for (const auto& e : edges_) {
        if (e.target == block_name) {
            result.push_back(e);
        }
    }
    return result;
}

std::vector<AuxiliaryCouplingEdge> AuxiliaryCouplingGraph::outgoingEdges(
    std::string_view block_name) const
{
    std::vector<AuxiliaryCouplingEdge> result;
    for (const auto& e : edges_) {
        if (e.source == block_name) {
            result.push_back(e);
        }
    }
    return result;
}

std::vector<std::string> AuxiliaryCouplingGraph::auxiliaryVertices() const
{
    std::unordered_set<std::string> names;
    for (const auto& e : edges_) {
        if (e.type == AuxiliaryCouplingType::AuxSelf ||
            e.type == AuxiliaryCouplingType::AuxToAux) {
            names.insert(e.source);
            names.insert(e.target);
        } else if (e.type == AuxiliaryCouplingType::FieldToAux) {
            names.insert(e.target);
        } else if (e.type == AuxiliaryCouplingType::AuxToField) {
            names.insert(e.source);
        }
    }
    return {names.begin(), names.end()};
}

std::vector<std::string> AuxiliaryCouplingGraph::fieldVertices() const
{
    std::unordered_set<std::string> names;
    for (const auto& e : edges_) {
        if (e.type == AuxiliaryCouplingType::FieldToAux) {
            names.insert(e.source);
        } else if (e.type == AuxiliaryCouplingType::AuxToField) {
            names.insert(e.target);
        }
    }
    return {names.begin(), names.end()};
}

bool AuxiliaryCouplingGraph::hasCouplingToFields(std::string_view block_name) const
{
    for (const auto& e : edges_) {
        if (e.type == AuxiliaryCouplingType::FieldToAux && e.target == block_name)
            return true;
        if (e.type == AuxiliaryCouplingType::AuxToField && e.source == block_name)
            return true;
    }
    return false;
}

bool AuxiliaryCouplingGraph::hasCouplingToAux(std::string_view block_name) const
{
    for (const auto& e : edges_) {
        if (e.type == AuxiliaryCouplingType::AuxToAux) {
            if (e.source == block_name || e.target == block_name)
                return true;
        }
    }
    return false;
}

void AuxiliaryCouplingGraph::clear()
{
    edges_.clear();
}

} // namespace systems
} // namespace FE
} // namespace svmp
