#include "LevelSet/LevelSetVelocityExtensionConstraint.h"

#include "Analysis/ContributionDescriptor.h"
#include "Assembly/Assembler.h"
#include "Assembly/AssemblyConstraintDistributor.h"
#include "Assembly/GlobalSystemView.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Dofs/DofHandler.h"
#include "Dofs/EntityDofMap.h"
#include "Sparsity/SparsityPattern.h"
#include "Systems/FESystem.h"
#include "Systems/SystemState.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <utility>

namespace svmp::FE::level_set {
namespace {

[[nodiscard]] std::vector<GlobalIndex> vertexDofs(
    const systems::FESystem& system,
    FieldId field,
    GlobalIndex vertex,
    int components)
{
    const auto& handler = system.fieldDofHandler(field);
    const auto* entity_map = handler.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error(
            "LevelSetVelocityExtensionConstraintKernel requires nodal vertex DOFs");
    }
    const auto local = entity_map->getVertexDofs(vertex);
    if (local.size() < static_cast<std::size_t>(components)) {
        throw std::runtime_error(
            "LevelSetVelocityExtensionConstraintKernel found too few vertex components");
    }
    std::vector<GlobalIndex> out(static_cast<std::size_t>(components));
    const auto offset = system.fieldDofOffset(field);
    for (int component = 0; component < components; ++component) {
        out[static_cast<std::size_t>(component)] =
            offset + local[static_cast<std::size_t>(component)];
    }
    return out;
}

[[nodiscard]] GlobalIndex dependencyDof(
    const systems::FESystem& system,
    const LevelSetVelocityExtensionConstraintConfig& config,
    const VelocityExtensionDependency& dependency)
{
    const auto field =
        dependency.field == VelocityExtensionDependencyField::SourceVelocity
            ? config.source_velocity_field
            : config.extension_field;
    return vertexDofs(system, field, dependency.vertex, config.components)
        .at(static_cast<std::size_t>(dependency.component));
}

} // namespace

LevelSetVelocityExtensionConstraintKernel::
    LevelSetVelocityExtensionConstraintKernel(
        LevelSetVelocityExtensionConstraintConfig config)
    : config_(std::move(config))
{
    if (config_.extension_field == INVALID_FIELD_ID ||
        config_.source_velocity_field == INVALID_FIELD_ID ||
        config_.extension_field == config_.source_velocity_field) {
        throw std::invalid_argument(
            "LevelSetVelocityExtensionConstraintKernel requires distinct valid source and extension fields");
    }
    if (config_.components <= 0 || config_.components > 3) {
        throw std::invalid_argument(
            "LevelSetVelocityExtensionConstraintKernel components must be in [1,3]");
    }
    if (config_.operator_tag.empty()) {
        throw std::invalid_argument(
            "LevelSetVelocityExtensionConstraintKernel operator tag must be non-empty");
    }
}

void LevelSetVelocityExtensionConstraintKernel::setFrozenRows(
    std::vector<VelocityExtensionConstraintRow> rows,
    std::uint64_t revision)
{
    for (auto& row : rows) {
        if (row.vertex == INVALID_GLOBAL_INDEX || row.component < 0 ||
            row.component >= config_.components) {
            throw std::invalid_argument(
                "LevelSetVelocityExtensionConstraintKernel received an invalid row");
        }
        std::map<std::tuple<VelocityExtensionDependencyField, GlobalIndex, int>,
                 Real>
            combined;
        for (const auto& dependency : row.dependencies) {
            if (dependency.vertex == INVALID_GLOBAL_INDEX ||
                dependency.component < 0 ||
                dependency.component >= config_.components ||
                !std::isfinite(dependency.coefficient)) {
                throw std::invalid_argument(
                    "LevelSetVelocityExtensionConstraintKernel received an invalid dependency");
            }
            combined[{dependency.field,
                      dependency.vertex,
                      dependency.component}] += dependency.coefficient;
        }
        row.dependencies.clear();
        row.dependencies.reserve(combined.size());
        for (const auto& [key, coefficient] : combined) {
            if (std::abs(coefficient) <= Real{1.0e-15}) {
                continue;
            }
            row.dependencies.push_back(VelocityExtensionDependency{
                .field = std::get<0>(key),
                .vertex = std::get<1>(key),
                .component = std::get<2>(key),
                .coefficient = coefficient,
            });
        }
    }
    std::sort(rows.begin(), rows.end(), [](const auto& lhs, const auto& rhs) {
        return std::tie(lhs.vertex, lhs.component) <
               std::tie(rhs.vertex, rhs.component);
    });
    const auto duplicate = std::adjacent_find(
        rows.begin(), rows.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.vertex == rhs.vertex && lhs.component == rhs.component;
        });
    if (duplicate != rows.end()) {
        throw std::invalid_argument(
            "LevelSetVelocityExtensionConstraintKernel received duplicate rows");
    }
    rows_ = std::move(rows);
    frozen_map_revision_ = revision;
    map_is_valid_ = true;
}

void LevelSetVelocityExtensionConstraintKernel::addSparsityCouplings(
    const systems::FESystem& system,
    sparsity::SparsityPattern& pattern) const
{
    // The active set can change between time steps.  Its direct graph equation
    // always couples an extension vertex to itself, all one-ring extension
    // neighbors (including cross-components after wall projection), and the
    // same-vertex physical velocity.  Register that conservative fixed graph
    // before setup so later coefficient refreshes never mutate sparsity.
    const auto& mesh = system.meshAccess();
    std::map<GlobalIndex, std::set<GlobalIndex>> neighbors;
    mesh.forEachCell([&](GlobalIndex cell) {
        std::vector<GlobalIndex> vertices;
        mesh.getCellNodes(cell, vertices);
        for (const auto vertex : vertices) {
            auto& adjacent = neighbors[vertex];
            adjacent.insert(vertices.begin(), vertices.end());
        }
    });

    for (const auto& [vertex, adjacent] : neighbors) {
        const auto rows =
            vertexDofs(system, config_.extension_field, vertex,
                       config_.components);
        std::vector<GlobalIndex> columns = rows;
        const auto source =
            vertexDofs(system, config_.source_velocity_field, vertex,
                       config_.components);
        columns.insert(columns.end(), source.begin(), source.end());
        for (const auto neighbor : adjacent) {
            const auto neighbor_dofs =
                vertexDofs(system, config_.extension_field, neighbor,
                           config_.components);
            columns.insert(columns.end(), neighbor_dofs.begin(),
                           neighbor_dofs.end());
        }
        std::sort(columns.begin(), columns.end());
        columns.erase(std::unique(columns.begin(), columns.end()),
                      columns.end());
        pattern.addElementCouplings(rows, columns);
    }
}

std::vector<analysis::ContributionDescriptor>
LevelSetVelocityExtensionConstraintKernel::analysisContributions() const
{
    auto contribution = analysis::ContributionDescriptor::globalCoupling(
        {analysis::VariableKey::field(config_.extension_field)},
        {analysis::VariableKey::field(config_.extension_field),
         analysis::VariableKey::field(config_.source_velocity_field)},
        config_.operator_tag,
        "LevelSetVelocityExtensionConstraintKernel");
    contribution.role = analysis::ContributionRole::ConstraintBlock;
    contribution.ensureStableContributionId();
    return {std::move(contribution)};
}

assembly::AssemblyResult
LevelSetVelocityExtensionConstraintKernel::assemble(
    const systems::FESystem& system,
    const systems::AssemblyRequest& request,
    const systems::SystemStateView& state,
    assembly::GlobalSystemView* matrix_out,
    assembly::GlobalSystemView* vector_out)
{
    if (!map_is_valid_) {
        throw std::runtime_error(
            "LevelSetVelocityExtensionConstraintKernel has no frozen extension map; refresh it before the nonlinear solve");
    }

    assembly::AssemblyResult result;
    if (matrix_out != nullptr) {
        matrix_out->beginAssemblyPhase();
    }
    if (vector_out != nullptr && vector_out != matrix_out) {
        vector_out->beginAssemblyPhase();
    }

    std::unique_ptr<assembly::GlobalSystemView> state_view;
    if (state.u_vector != nullptr) {
        auto* vector = const_cast<backends::GenericVector*>(state.u_vector);
        state_view = vector->createAssemblyView();
    }
    const auto read = [&](GlobalIndex dof) -> Real {
        if (state_view) {
            return state_view->getVectorEntry(dof);
        }
        if (dof < 0 || static_cast<std::size_t>(dof) >= state.u.size()) {
            throw std::runtime_error(
                "LevelSetVelocityExtensionConstraintKernel state vector is not globally indexable");
        }
        return state.u[static_cast<std::size_t>(dof)];
    };

    assembly::AssemblyConstraintDistributor distributor(system.constraints());
    for (const auto& row : rows_) {
        const auto row_dof =
            vertexDofs(system, config_.extension_field, row.vertex,
                       config_.components)
                .at(static_cast<std::size_t>(row.component));

        std::vector<GlobalIndex> dofs{row_dof};
        dofs.reserve(1u + row.dependencies.size());
        for (const auto& dependency : row.dependencies) {
            dofs.push_back(dependencyDof(system, config_, dependency));
        }

        std::vector<Real> local_vector(dofs.size(), Real{0.0});
        if (request.want_vector && vector_out != nullptr) {
            Real residual = read(row_dof);
            for (std::size_t dependency = 0;
                 dependency < row.dependencies.size(); ++dependency) {
                residual -= row.dependencies[dependency].coefficient *
                            read(dofs[dependency + 1u]);
            }
            local_vector.front() = residual;
        }

        std::vector<Real> local_matrix;
        if (request.want_matrix && matrix_out != nullptr) {
            local_matrix.assign(dofs.size() * dofs.size(), Real{0.0});
            local_matrix.front() = Real{1.0};
            for (std::size_t dependency = 0;
                 dependency < row.dependencies.size(); ++dependency) {
                local_matrix[dependency + 1u] =
                    -row.dependencies[dependency].coefficient;
            }
        }

        if (request.want_matrix && matrix_out != nullptr &&
            request.want_vector && vector_out != nullptr) {
            distributor.distributeLocalToGlobal(
                local_matrix, local_vector, dofs, *matrix_out, *vector_out);
        } else if (request.want_matrix && matrix_out != nullptr) {
            distributor.distributeMatrixToGlobal(local_matrix, dofs,
                                                  *matrix_out);
        } else if (request.want_vector && vector_out != nullptr) {
            distributor.distributeVectorToGlobal(local_vector, dofs,
                                                  *vector_out);
        }
        result.matrix_entries_inserted +=
            request.want_matrix
                ? static_cast<GlobalIndex>(1u + row.dependencies.size())
                : 0;
        result.vector_entries_inserted += request.want_vector ? 1 : 0;
    }
    return result;
}

std::shared_ptr<LevelSetVelocityExtensionConstraintKernel>
findLevelSetVelocityExtensionConstraintKernel(
    const systems::FESystem& system,
    const std::string& operator_tag,
    FieldId extension_field)
{
    if (!system.hasOperator(operator_tag)) {
        return {};
    }
    for (const auto& kernel : system.operatorDefinition(operator_tag).global) {
        auto extension = std::dynamic_pointer_cast<
            LevelSetVelocityExtensionConstraintKernel>(kernel);
        if (extension && extension->extensionField() == extension_field) {
            return extension;
        }
    }
    return {};
}

} // namespace svmp::FE::level_set
