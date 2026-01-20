/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/FESystem.h"
#include "Systems/CoupledBoundaryManager.h"
#include "Systems/GlobalKernelStateProvider.h"
#include "Systems/MaterialStateProvider.h"
#include "Systems/OperatorBackends.h"

#include "Assembly/Assembler.h"
#include "Assembly/AssemblyKernel.h"
#include "Assembly/AssemblerSelection.h"
#include "Assembly/MatrixFreeAssembler.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Assembly/MeshAccess.h"
#include "Mesh/Core/InterfaceMesh.h"
#include "Systems/MeshSearchAccess.h"
#endif

#include "Constraints/ParallelConstraints.h"

#include "Sparsity/ConstraintSparsityAugmenter.h"
#include "Sparsity/DistributedSparsityPattern.h"

#include "Systems/SystemsExceptions.h"

#include "Spaces/FunctionSpace.h"

#include "Dofs/EntityDofMap.h"

#include "Elements/ReferenceElement.h"

#include "Quadrature/QuadratureFactory.h"

#include "Core/FEConfig.h"

#include <algorithm>
#include <thread>
#include <tuple>

#if FE_HAS_MPI
#  include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace systems {

namespace {

class AffineConstraintsQuery final : public sparsity::IConstraintQuery {
public:
    explicit AffineConstraintsQuery(constraints::AffineConstraints constraints)
        : constraints_(std::move(constraints))
    {}

    [[nodiscard]] bool isConstrained(GlobalIndex dof) const override
    {
        return constraints_.isConstrained(dof);
    }

    [[nodiscard]] std::vector<GlobalIndex> getMasterDofs(GlobalIndex constrained_dof) const override
    {
        auto view = constraints_.getConstraint(constrained_dof);
        if (!view.has_value()) {
            return {};
        }
        std::vector<GlobalIndex> masters;
        masters.reserve(view->entries.size());
        for (const auto& entry : view->entries) {
            masters.push_back(entry.master_dof);
        }
        return masters;
    }

    [[nodiscard]] std::vector<GlobalIndex> getAllConstrainedDofs() const override
    {
        return constraints_.getConstrainedDofs();
    }

    [[nodiscard]] std::size_t numConstraints() const override
    {
        return constraints_.numConstraints();
    }

private:
    constraints::AffineConstraints constraints_;
};

dofs::FieldLayout toFieldLayout(dofs::DofNumberingStrategy numbering)
{
    switch (numbering) {
        case dofs::DofNumberingStrategy::Block:
            return dofs::FieldLayout::Block;
        case dofs::DofNumberingStrategy::Interleaved:
            return dofs::FieldLayout::Interleaved;
        default:
            return dofs::FieldLayout::Interleaved;
    }
}

LocalIndex maxCellQuadraturePoints(const assembly::IMeshAccess& mesh,
                                   const spaces::FunctionSpace& test_space)
{
    LocalIndex max_qpts = 0;
    mesh.forEachCell([&](GlobalIndex cell_id) {
        const ElementType cell_type = mesh.getCellType(cell_id);
        const auto& test_element = test_space.getElement(cell_type, cell_id);

        auto quad_rule = test_element.quadrature();
        if (!quad_rule) {
            const int quad_order = quadrature::QuadratureFactory::recommended_order(
                test_element.polynomial_order(), false);
            quad_rule = quadrature::QuadratureFactory::create(cell_type, quad_order);
        }

        const auto n = static_cast<LocalIndex>(quad_rule->num_points());
        max_qpts = std::max(max_qpts, n);
    });

    return max_qpts;
}

ElementType faceTypeForFace(const assembly::IMeshAccess& mesh,
                            GlobalIndex face_id,
                            GlobalIndex cell_id)
{
    const ElementType cell_type = mesh.getCellType(cell_id);
    const LocalIndex local_face_id = mesh.getLocalFaceIndex(face_id, cell_id);

    elements::ReferenceElement ref = elements::ReferenceElement::create(cell_type);
    const auto& face_nodes = ref.face_nodes(static_cast<std::size_t>(local_face_id));

    switch (face_nodes.size()) {
        case 2:
            return ElementType::Line2;
        case 3:
            return ElementType::Triangle3;
        case 4:
            return ElementType::Quad4;
        default:
            throw FEException("FESystem::setup: unsupported face topology",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
}

LocalIndex maxBoundaryFaceQuadraturePoints(const assembly::IMeshAccess& mesh,
                                           const spaces::FunctionSpace& test_space,
                                           const spaces::FunctionSpace& trial_space)
{
    LocalIndex max_qpts = 0;
    mesh.forEachBoundaryFace(/*marker=*/-1, [&](GlobalIndex face_id, GlobalIndex cell_id) {
        const ElementType cell_type = mesh.getCellType(cell_id);
        const auto& test_element = test_space.getElement(cell_type, cell_id);
        const auto& trial_element = trial_space.getElement(cell_type, cell_id);

        const int quad_order = quadrature::QuadratureFactory::recommended_order(
            std::max(test_element.polynomial_order(), trial_element.polynomial_order()), false);

        const ElementType face_type = faceTypeForFace(mesh, face_id, cell_id);
        auto quad_rule = quadrature::QuadratureFactory::create(face_type, quad_order);
        max_qpts = std::max(max_qpts, static_cast<LocalIndex>(quad_rule->num_points()));
    });

    return max_qpts;
}

LocalIndex maxInteriorFaceQuadraturePoints(const assembly::IMeshAccess& mesh,
                                           const spaces::FunctionSpace& test_space,
                                           const spaces::FunctionSpace& trial_space)
{
    LocalIndex max_qpts = 0;
    mesh.forEachInteriorFace([&](GlobalIndex face_id, GlobalIndex cell_minus, GlobalIndex cell_plus) {
        const ElementType cell_type_minus = mesh.getCellType(cell_minus);
        const ElementType cell_type_plus = mesh.getCellType(cell_plus);

        const auto& test_element_minus = test_space.getElement(cell_type_minus, cell_minus);
        const auto& trial_element_minus = trial_space.getElement(cell_type_minus, cell_minus);
        const auto& test_element_plus = test_space.getElement(cell_type_plus, cell_plus);
        const auto& trial_element_plus = trial_space.getElement(cell_type_plus, cell_plus);

        const int quad_order = quadrature::QuadratureFactory::recommended_order(
            std::max({test_element_minus.polynomial_order(),
                      trial_element_minus.polynomial_order(),
                      test_element_plus.polynomial_order(),
                      trial_element_plus.polynomial_order()}),
            false);

        const ElementType face_type_minus = faceTypeForFace(mesh, face_id, cell_minus);
        const ElementType face_type_plus = faceTypeForFace(mesh, face_id, cell_plus);
        FE_THROW_IF(face_type_minus != face_type_plus, InvalidStateException,
                    "FESystem::setup: interior face has mismatched face topology between adjacent cells");

        auto quad_rule = quadrature::QuadratureFactory::create(face_type_minus, quad_order);
        max_qpts = std::max(max_qpts, static_cast<LocalIndex>(quad_rule->num_points()));
    });

    return max_qpts;
}

} // namespace

void FESystem::setup(const SetupOptions& opts, const SetupInputs& inputs)
{
    invalidateSetup();

    FE_THROW_IF(field_registry_.size() == 0u, InvalidStateException,
                "FESystem::setup: no fields registered");

    // ---------------------------------------------------------------------
    // DOFs (multi-field)
    // ---------------------------------------------------------------------
    dof_handler_ = dofs::DofHandler{};
    field_dof_handlers_.clear();
    field_dof_offsets_.clear();

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    if (!mesh_ && inputs.mesh) {
        mesh_ = inputs.mesh;
        coord_cfg_ = inputs.coord_cfg;
        mesh_access_ = std::make_shared<assembly::MeshAccess>(*mesh_, coord_cfg_);
        if (!search_access_) {
            search_access_ = std::make_shared<MeshSearchAccess>(*mesh_, coord_cfg_);
        }
    }
#endif

    const auto n_fields = field_registry_.size();
    field_dof_handlers_.resize(n_fields);
    field_dof_offsets_.assign(n_fields, 0);

    auto distribute_field = [&](const FieldRecord& rec) {
        FE_CHECK_NOT_NULL(rec.space.get(), "FESystem::setup: field.space");

        dofs::DofHandler dh;

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
        if (mesh_) {
            dh.distributeDofs(*mesh_, *rec.space, opts.dof_options);
            dh.finalize();
            return dh;
        }
#endif

        if (inputs.topology_override.has_value()) {
            const auto& topology = *inputs.topology_override;
            FE_THROW_IF(topology.n_cells <= 0, InvalidArgumentException,
                        "FESystem::setup: topology_override has no cells");

            dofs::DofLayoutInfo layout;
            const auto continuity = rec.space->continuity();
            const bool is_continuous =
                (continuity == Continuity::C0 || continuity == Continuity::C1);

            const int order = rec.space->polynomial_order();
            const int dim = topology.dim;
            const auto cell0 = topology.getCellVertices(0);
            FE_THROW_IF(cell0.empty(), InvalidArgumentException,
                        "FESystem::setup: topology_override cell 0 has no vertices");

            if (is_continuous) {
                layout = dofs::DofLayoutInfo::Lagrange(order, dim,
                                                      static_cast<int>(cell0.size()),
                                                      rec.space->value_dimension());
            } else {
                layout = dofs::DofLayoutInfo::DG(order, static_cast<int>(cell0.size()),
                                                 rec.space->value_dimension());
            }

            layout.total_dofs_per_element = static_cast<LocalIndex>(rec.space->dofs_per_element());
            dh.distributeDofs(topology, layout, opts.dof_options);
            dh.finalize();
            return dh;
        }

        FE_THROW(InvalidArgumentException,
                 "FESystem::setup: need Mesh (SVMP_FE_WITH_MESH) or topology_override for DOF distribution");
    };

    for (const auto& rec : field_registry_.records()) {
        const auto idx = static_cast<std::size_t>(rec.id);
        FE_THROW_IF(rec.id < 0 || idx >= field_dof_handlers_.size(), InvalidStateException,
                    "FESystem::setup: field registry contains invalid FieldId");
        field_dof_handlers_[idx] = distribute_field(rec);
    }

    // Assign monolithic offsets in field registration order.
    GlobalIndex total_dofs = 0;
    for (const auto& rec : field_registry_.records()) {
        const auto idx = static_cast<std::size_t>(rec.id);
        field_dof_offsets_[idx] = total_dofs;
        total_dofs += field_dof_handlers_[idx].getNumDofs();
    }

    // Build a monolithic DofMap + EntityDofMap so Systems can expose a single global DofHandler.
    const auto n_cells = meshAccess().numCells();
    LocalIndex approx_dofs_per_cell = 0;
    for (const auto& rec : field_registry_.records()) {
        const auto idx = static_cast<std::size_t>(rec.id);
        approx_dofs_per_cell = static_cast<LocalIndex>(
            approx_dofs_per_cell + field_dof_handlers_[idx].getDofMap().getMaxDofsPerCell());
    }

    dofs::DofMap monolithic_map(n_cells, total_dofs, approx_dofs_per_cell);
    monolithic_map.setNumDofs(total_dofs);
    monolithic_map.setNumLocalDofs(0);

    std::vector<GlobalIndex> cell_dofs;
    for (GlobalIndex cell = 0; cell < n_cells; ++cell) {
        cell_dofs.clear();
        for (const auto& rec : field_registry_.records()) {
            const auto idx = static_cast<std::size_t>(rec.id);
            const auto offset = field_dof_offsets_[idx];
            auto local = field_dof_handlers_[idx].getDofMap().getCellDofs(cell);
            cell_dofs.reserve(cell_dofs.size() + local.size());
            for (auto d : local) {
                cell_dofs.push_back(d + offset);
            }
        }
        monolithic_map.setCellDofs(cell, cell_dofs);
    }

    // Merge entity-DOF maps if available (Mesh-driven workflows).
    std::unique_ptr<dofs::EntityDofMap> merged_entity_map;
    {
        const dofs::EntityDofMap* first = nullptr;
        for (const auto& rec : field_registry_.records()) {
            const auto idx = static_cast<std::size_t>(rec.id);
            auto* map = field_dof_handlers_[idx].getEntityDofMap();
            if (map == nullptr) {
                first = nullptr;
                break;
            }
            if (!first) {
                first = map;
            }
        }

        if (first) {
            merged_entity_map = std::make_unique<dofs::EntityDofMap>();
            merged_entity_map->reserve(first->numVertices(), first->numEdges(),
                                       first->numFaces(), first->numCells());

            std::vector<GlobalIndex> entity_dofs;

            for (GlobalIndex v = 0; v < first->numVertices(); ++v) {
                entity_dofs.clear();
                for (const auto& rec : field_registry_.records()) {
                    const auto idx = static_cast<std::size_t>(rec.id);
                    const auto offset = field_dof_offsets_[idx];
                    const auto* emap = field_dof_handlers_[idx].getEntityDofMap();
                    FE_CHECK_NOT_NULL(emap, "FESystem::setup: EntityDofMap");
                    auto vdofs = emap->getVertexDofs(v);
                    entity_dofs.reserve(entity_dofs.size() + vdofs.size());
                    for (auto d : vdofs) {
                        entity_dofs.push_back(d + offset);
                    }
                }
                merged_entity_map->setVertexDofs(v, entity_dofs);
            }

            for (GlobalIndex e = 0; e < first->numEdges(); ++e) {
                entity_dofs.clear();
                for (const auto& rec : field_registry_.records()) {
                    const auto idx = static_cast<std::size_t>(rec.id);
                    const auto offset = field_dof_offsets_[idx];
                    const auto* emap = field_dof_handlers_[idx].getEntityDofMap();
                    FE_CHECK_NOT_NULL(emap, "FESystem::setup: EntityDofMap");
                    auto edofs = emap->getEdgeDofs(e);
                    entity_dofs.reserve(entity_dofs.size() + edofs.size());
                    for (auto d : edofs) {
                        entity_dofs.push_back(d + offset);
                    }
                }
                merged_entity_map->setEdgeDofs(e, entity_dofs);
            }

            for (GlobalIndex f = 0; f < first->numFaces(); ++f) {
                entity_dofs.clear();
                for (const auto& rec : field_registry_.records()) {
                    const auto idx = static_cast<std::size_t>(rec.id);
                    const auto offset = field_dof_offsets_[idx];
                    const auto* emap = field_dof_handlers_[idx].getEntityDofMap();
                    FE_CHECK_NOT_NULL(emap, "FESystem::setup: EntityDofMap");
                    auto fdofs = emap->getFaceDofs(f);
                    entity_dofs.reserve(entity_dofs.size() + fdofs.size());
                    for (auto d : fdofs) {
                        entity_dofs.push_back(d + offset);
                    }
                }
                merged_entity_map->setFaceDofs(f, entity_dofs);
            }

            for (GlobalIndex c = 0; c < first->numCells(); ++c) {
                entity_dofs.clear();
                for (const auto& rec : field_registry_.records()) {
                    const auto idx = static_cast<std::size_t>(rec.id);
                    const auto offset = field_dof_offsets_[idx];
                    const auto* emap = field_dof_handlers_[idx].getEntityDofMap();
                    FE_CHECK_NOT_NULL(emap, "FESystem::setup: EntityDofMap");
                    auto cdofs = emap->getCellInteriorDofs(c);
                    entity_dofs.reserve(entity_dofs.size() + cdofs.size());
                    for (auto d : cdofs) {
                        entity_dofs.push_back(d + offset);
                    }
                }
                merged_entity_map->setCellInteriorDofs(c, entity_dofs);
            }

            merged_entity_map->buildReverseMapping();
            merged_entity_map->finalize();
        }
    }

    dofs::DofPartition part;
    {
        std::vector<dofs::IndexInterval> owned_intervals;
        std::vector<GlobalIndex> owned_explicit;
        std::vector<GlobalIndex> ghost_explicit;

        for (const auto& rec : field_registry_.records()) {
            const auto idx = static_cast<std::size_t>(rec.id);
            const auto offset = field_dof_offsets_[idx];
            const auto& fpart = field_dof_handlers_[idx].getPartition();

            if (auto range = fpart.locallyOwned().contiguousRange()) {
                owned_intervals.push_back(dofs::IndexInterval{range->begin + offset, range->end + offset});
            } else {
                auto vec = fpart.locallyOwned().toVector();
                owned_explicit.reserve(owned_explicit.size() + vec.size());
                for (auto d : vec) {
                    owned_explicit.push_back(d + offset);
                }
            }

            auto ghosts = fpart.ghost().toVector();
            ghost_explicit.reserve(ghost_explicit.size() + ghosts.size());
            for (auto d : ghosts) {
                ghost_explicit.push_back(d + offset);
            }
        }

        dofs::IndexSet owned;
        if (!owned_intervals.empty()) {
            owned = dofs::IndexSet(std::move(owned_intervals));
        }
        if (!owned_explicit.empty()) {
            owned = owned.unionWith(dofs::IndexSet(std::move(owned_explicit)));
        }

        dofs::IndexSet ghost;
        if (!ghost_explicit.empty()) {
            ghost = dofs::IndexSet(std::move(ghost_explicit));
        }

        part = dofs::DofPartition(std::move(owned), std::move(ghost));
        part.setGlobalSize(total_dofs);
    }

    monolithic_map.setNumLocalDofs(part.localOwnedSize());

    dof_handler_ = dofs::DofHandler{};
    dof_handler_.setDofMap(std::move(monolithic_map));
    dof_handler_.setPartition(std::move(part));
    if (merged_entity_map) {
        dof_handler_.setEntityDofMap(std::move(merged_entity_map));
    }
    dof_handler_.finalize();

    // ---------------------------------------------------------------------
    // Field/block metadata (monolithic across fields)
    // ---------------------------------------------------------------------
    field_map_ = dofs::FieldDofMap{};

    if (field_registry_.size() > 1u) {
        field_map_.setLayout(opts.dof_options.numbering == dofs::DofNumberingStrategy::Block
                                 ? dofs::FieldLayout::Block
                                 : dofs::FieldLayout::FieldBlock);
    } else {
        field_map_.setLayout(toFieldLayout(opts.dof_options.numbering));
    }

    for (const auto& rec : field_registry_.records()) {
        const auto idx = static_cast<std::size_t>(rec.id);
        const auto n_components = rec.space->value_dimension();
        const auto n_dofs_field = field_dof_handlers_[idx].getNumDofs();

        if (n_components <= 1) {
            field_map_.addScalarField(rec.name, n_dofs_field);
        } else {
            FE_THROW_IF(n_dofs_field % n_components != 0, InvalidStateException,
                        "FESystem::setup: vector-valued field has non-divisible DOF count");
            field_map_.addVectorField(rec.name, static_cast<LocalIndex>(n_components),
                                      n_dofs_field / n_components);
        }
    }
    field_map_.finalize();
    block_map_.reset();
    if (field_registry_.size() > 1u) {
        auto blocks = std::make_unique<dofs::BlockDofMap>();
        for (const auto& rec : field_registry_.records()) {
            const auto idx = static_cast<std::size_t>(rec.id);
            blocks->addBlock(rec.name, field_dof_handlers_[idx].getNumDofs());
        }
        blocks->finalize();
        block_map_ = std::move(blocks);
    }

    // ---------------------------------------------------------------------
    // Constraints
    // ---------------------------------------------------------------------
    affine_constraints_.clear();
    for (auto& c : system_constraint_defs_) {
        FE_CHECK_NOT_NULL(c.get(), "FESystem::setup: system constraint");
        c->apply(*this, affine_constraints_);
    }
    for (const auto& c : constraint_defs_) {
        FE_CHECK_NOT_NULL(c.get(), "FESystem::setup: constraint");
        c->apply(affine_constraints_);
    }

    // Synchronize in MPI before closing.
    constraints::ParallelConstraints parallel(dof_handler_.getPartition());
    if (parallel.isParallel()) {
        parallel.synchronize(affine_constraints_);
    }

    affine_constraints_.close();

    // ---------------------------------------------------------------------
    // Sparsity
    // ---------------------------------------------------------------------
    sparsity_by_op_.clear();
    distributed_sparsity_by_op_.clear();
    const auto op_tags = operator_registry_.list();
    const auto n_total_dofs = dof_handler_.getNumDofs();
    const auto n_cells_sparsity = meshAccess().numCells();

    const auto& partition = dof_handler_.getPartition();
    const auto owned_iv_opt = partition.locallyOwned().contiguousRange();
    const bool build_distributed =
        (partition.globalSize() > 0) &&
        (partition.globalSize() > partition.localOwnedSize()) &&
        owned_iv_opt.has_value();
    const sparsity::IndexRange owned_range = build_distributed
                                                ? sparsity::IndexRange{owned_iv_opt->begin, owned_iv_opt->end}
                                                : sparsity::IndexRange{};
    const auto& ghost_set = partition.ghost();
    const auto& relevant_set = partition.locallyRelevant();

		    for (const auto& tag : op_tags) {
		        auto pattern = std::make_unique<sparsity::SparsityPattern>(
		            n_total_dofs, n_total_dofs);

		        std::unique_ptr<sparsity::DistributedSparsityPattern> dist_pattern;
		        std::unordered_map<GlobalIndex, std::vector<GlobalIndex>> ghost_row_cols;
		        if (build_distributed) {
		            dist_pattern = std::make_unique<sparsity::DistributedSparsityPattern>(
		                owned_range, owned_range, n_total_dofs, n_total_dofs);
		        }

		        const auto& def = operator_registry_.get(tag);

		        std::vector<std::pair<FieldId, FieldId>> cell_pairs;
		        std::vector<std::tuple<int, FieldId, FieldId>> boundary_pairs;
		        std::vector<std::pair<FieldId, FieldId>> interior_pairs;
		        std::vector<std::tuple<int, FieldId, FieldId>> interface_pairs;

		        cell_pairs.reserve(def.cells.size());
		        boundary_pairs.reserve(def.boundary.size());
		        interior_pairs.reserve(def.interior.size());
		        interface_pairs.reserve(def.interface_faces.size());

	        auto maybe_add_cell_pair =
	            [&](FieldId test, FieldId trial, const std::shared_ptr<assembly::AssemblyKernel>& kernel) {
	                if (!kernel || kernel->isVectorOnly()) {
	                    return;
	                }
	                cell_pairs.emplace_back(test, trial);
	            };

	        auto maybe_add_boundary_pair =
	            [&](int marker,
	                FieldId test,
	                FieldId trial,
	                const std::shared_ptr<assembly::AssemblyKernel>& kernel) {
	                if (!kernel || kernel->isVectorOnly()) {
	                    return;
	                }
	                boundary_pairs.emplace_back(marker, test, trial);
	            };

		        auto maybe_add_interior_pair =
		            [&](FieldId test, FieldId trial, const std::shared_ptr<assembly::AssemblyKernel>& kernel) {
		                if (!kernel || kernel->isVectorOnly()) {
		                    return;
		                }
		                interior_pairs.emplace_back(test, trial);
		            };

		        auto maybe_add_interface_pair =
		            [&](int marker,
		                FieldId test,
		                FieldId trial,
		                const std::shared_ptr<assembly::AssemblyKernel>& kernel) {
		                if (!kernel || kernel->isVectorOnly()) {
		                    return;
		                }
		                interface_pairs.emplace_back(marker, test, trial);
		            };

	        for (const auto& term : def.cells) {
	            maybe_add_cell_pair(term.test_field, term.trial_field, term.kernel);
	        }
	        for (const auto& term : def.boundary) {
	            maybe_add_boundary_pair(term.marker, term.test_field, term.trial_field, term.kernel);
	        }
		        for (const auto& term : def.interior) {
		            maybe_add_interior_pair(term.test_field, term.trial_field, term.kernel);
		        }
		        for (const auto& term : def.interface_faces) {
		            maybe_add_interface_pair(term.marker, term.test_field, term.trial_field, term.kernel);
		        }

	        std::sort(cell_pairs.begin(), cell_pairs.end());
	        cell_pairs.erase(std::unique(cell_pairs.begin(), cell_pairs.end()), cell_pairs.end());

	        std::sort(boundary_pairs.begin(), boundary_pairs.end());
	        boundary_pairs.erase(std::unique(boundary_pairs.begin(), boundary_pairs.end()), boundary_pairs.end());

		        std::sort(interior_pairs.begin(), interior_pairs.end());
		        interior_pairs.erase(std::unique(interior_pairs.begin(), interior_pairs.end()), interior_pairs.end());

		        std::sort(interface_pairs.begin(), interface_pairs.end());
		        interface_pairs.erase(std::unique(interface_pairs.begin(), interface_pairs.end()), interface_pairs.end());

		        std::vector<GlobalIndex> row_dofs;
		        std::vector<GlobalIndex> col_dofs;

		        auto add_element_couplings = [&](std::span<const GlobalIndex> rows,
		                                         std::span<const GlobalIndex> cols) {
		            pattern->addElementCouplings(rows, cols);
		            if (!dist_pattern) {
		                return;
		            }

		            for (const auto global_row : rows) {
		                if (owned_range.contains(global_row)) {
		                    for (const auto global_col : cols) {
		                        dist_pattern->addEntry(global_row, global_col);
		                    }
		                    continue;
		                }

		                if (!ghost_set.contains(global_row)) {
		                    continue;
		                }

		                auto& cols_for_row = ghost_row_cols[global_row];
		                for (const auto global_col : cols) {
		                    if (relevant_set.contains(global_col)) {
		                        cols_for_row.push_back(global_col);
		                    }
		                }
		            }
		        };

		        auto add_cell_couplings = [&](GlobalIndex cell_id,
		                                      const dofs::DofMap& row_map,
		                                      const dofs::DofMap& col_map,
	                                      GlobalIndex row_offset,
	                                      GlobalIndex col_offset) {
	            auto row_local = row_map.getCellDofs(cell_id);
	            auto col_local = col_map.getCellDofs(cell_id);

	            row_dofs.resize(row_local.size());
	            for (std::size_t i = 0; i < row_local.size(); ++i) {
	                row_dofs[i] = row_local[i] + row_offset;
	            }

	            col_dofs.resize(col_local.size());
	            for (std::size_t j = 0; j < col_local.size(); ++j) {
	                col_dofs[j] = col_local[j] + col_offset;
	            }

		            add_element_couplings(row_dofs, col_dofs);
		        };

	        // Cell terms: all cells participate.
	        for (const auto& [test_field, trial_field] : cell_pairs) {
	            const auto test_idx = static_cast<std::size_t>(test_field);
	            const auto trial_idx = static_cast<std::size_t>(trial_field);

	            FE_THROW_IF(test_field < 0 || test_idx >= field_dof_handlers_.size(), InvalidStateException,
	                        "FESystem::setup: invalid test field in sparsity build");
	            FE_THROW_IF(trial_field < 0 || trial_idx >= field_dof_handlers_.size(), InvalidStateException,
	                        "FESystem::setup: invalid trial field in sparsity build");

            const auto& row_map = field_dof_handlers_[test_idx].getDofMap();
            const auto& col_map = field_dof_handlers_[trial_idx].getDofMap();
	            const auto row_offset = field_dof_offsets_[test_idx];
	            const auto col_offset = field_dof_offsets_[trial_idx];

	            for (GlobalIndex cell = 0; cell < n_cells_sparsity; ++cell) {
	                add_cell_couplings(cell, row_map, col_map, row_offset, col_offset);
	            }
	        }

	        // Boundary terms: only cells adjacent to the requested marker participate.
	        std::vector<GlobalIndex> marker_cells;
	        for (const auto& [marker, test_field, trial_field] : boundary_pairs) {
	            const auto test_idx = static_cast<std::size_t>(test_field);
	            const auto trial_idx = static_cast<std::size_t>(trial_field);

	            FE_THROW_IF(test_field < 0 || test_idx >= field_dof_handlers_.size(), InvalidStateException,
	                        "FESystem::setup: invalid test field in boundary sparsity build");
	            FE_THROW_IF(trial_field < 0 || trial_idx >= field_dof_handlers_.size(), InvalidStateException,
	                        "FESystem::setup: invalid trial field in boundary sparsity build");

	            const auto& row_map = field_dof_handlers_[test_idx].getDofMap();
	            const auto& col_map = field_dof_handlers_[trial_idx].getDofMap();
	            const auto row_offset = field_dof_offsets_[test_idx];
	            const auto col_offset = field_dof_offsets_[trial_idx];

	            marker_cells.clear();
	            meshAccess().forEachBoundaryFace(marker, [&](GlobalIndex /*face_id*/, GlobalIndex cell_id) {
	                marker_cells.push_back(cell_id);
	            });
	            std::sort(marker_cells.begin(), marker_cells.end());
	            marker_cells.erase(std::unique(marker_cells.begin(), marker_cells.end()), marker_cells.end());

	            for (const auto cell_id : marker_cells) {
	                add_cell_couplings(cell_id, row_map, col_map, row_offset, col_offset);
	            }
	        }

	        // Interior face terms (DG): include both self and cross-cell couplings.
	        std::vector<GlobalIndex> row_dofs_minus, row_dofs_plus;
	        std::vector<GlobalIndex> col_dofs_minus, col_dofs_plus;
	        for (const auto& [test_field, trial_field] : interior_pairs) {
	            const auto test_idx = static_cast<std::size_t>(test_field);
	            const auto trial_idx = static_cast<std::size_t>(trial_field);

	            FE_THROW_IF(test_field < 0 || test_idx >= field_dof_handlers_.size(), InvalidStateException,
	                        "FESystem::setup: invalid test field in interior-face sparsity build");
	            FE_THROW_IF(trial_field < 0 || trial_idx >= field_dof_handlers_.size(), InvalidStateException,
	                        "FESystem::setup: invalid trial field in interior-face sparsity build");

	            const auto& row_map = field_dof_handlers_[test_idx].getDofMap();
	            const auto& col_map = field_dof_handlers_[trial_idx].getDofMap();
	            const auto row_offset = field_dof_offsets_[test_idx];
	            const auto col_offset = field_dof_offsets_[trial_idx];

	            meshAccess().forEachInteriorFace(
	                [&](GlobalIndex /*face_id*/, GlobalIndex cell_minus, GlobalIndex cell_plus) {
	                    auto minus_row_local = row_map.getCellDofs(cell_minus);
	                    auto plus_row_local = row_map.getCellDofs(cell_plus);
	                    auto minus_col_local = col_map.getCellDofs(cell_minus);
	                    auto plus_col_local = col_map.getCellDofs(cell_plus);

	                    row_dofs_minus.resize(minus_row_local.size());
	                    for (std::size_t i = 0; i < minus_row_local.size(); ++i) {
	                        row_dofs_minus[i] = minus_row_local[i] + row_offset;
	                    }
	                    row_dofs_plus.resize(plus_row_local.size());
	                    for (std::size_t i = 0; i < plus_row_local.size(); ++i) {
	                        row_dofs_plus[i] = plus_row_local[i] + row_offset;
	                    }

	                    col_dofs_minus.resize(minus_col_local.size());
	                    for (std::size_t j = 0; j < minus_col_local.size(); ++j) {
	                        col_dofs_minus[j] = minus_col_local[j] + col_offset;
	                    }
	                    col_dofs_plus.resize(plus_col_local.size());
	                    for (std::size_t j = 0; j < plus_col_local.size(); ++j) {
	                        col_dofs_plus[j] = plus_col_local[j] + col_offset;
	                    }

		                    add_element_couplings(row_dofs_minus, col_dofs_minus);
		                    add_element_couplings(row_dofs_plus, col_dofs_plus);
		                    add_element_couplings(row_dofs_minus, col_dofs_plus);
		                    add_element_couplings(row_dofs_plus, col_dofs_minus);
		                });
		        }

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
		        // Interface face terms (InterfaceMesh subset): include both self and cross-cell couplings.
		        std::vector<GlobalIndex> iface_row_dofs_minus, iface_row_dofs_plus;
		        std::vector<GlobalIndex> iface_col_dofs_minus, iface_col_dofs_plus;
		        for (const auto& [marker, test_field, trial_field] : interface_pairs) {
		            const auto test_idx = static_cast<std::size_t>(test_field);
		            const auto trial_idx = static_cast<std::size_t>(trial_field);

		            FE_THROW_IF(test_field < 0 || test_idx >= field_dof_handlers_.size(), InvalidStateException,
		                        "FESystem::setup: invalid test field in interface-face sparsity build");
		            FE_THROW_IF(trial_field < 0 || trial_idx >= field_dof_handlers_.size(), InvalidStateException,
		                        "FESystem::setup: invalid trial field in interface-face sparsity build");

		            const auto& row_map = field_dof_handlers_[test_idx].getDofMap();
		            const auto& col_map = field_dof_handlers_[trial_idx].getDofMap();
		            const auto row_offset = field_dof_offsets_[test_idx];
		            const auto col_offset = field_dof_offsets_[trial_idx];

		            auto add_interface_mesh_couplings = [&](const svmp::InterfaceMesh& iface) {
		                for (std::size_t lf = 0; lf < iface.n_faces(); ++lf) {
		                    const auto local_face = static_cast<svmp::index_t>(lf);
		                    const GlobalIndex cell_minus = static_cast<GlobalIndex>(iface.volume_cell_minus(local_face));
		                    const GlobalIndex cell_plus = static_cast<GlobalIndex>(iface.volume_cell_plus(local_face));
		                    if (cell_minus == INVALID_GLOBAL_INDEX || cell_plus == INVALID_GLOBAL_INDEX) {
		                        continue; // One-sided interface faces: no cross-cell couplings.
		                    }

		                    auto minus_row_local = row_map.getCellDofs(cell_minus);
		                    auto plus_row_local = row_map.getCellDofs(cell_plus);
		                    auto minus_col_local = col_map.getCellDofs(cell_minus);
		                    auto plus_col_local = col_map.getCellDofs(cell_plus);

		                    iface_row_dofs_minus.resize(minus_row_local.size());
		                    for (std::size_t i = 0; i < minus_row_local.size(); ++i) {
		                        iface_row_dofs_minus[i] = minus_row_local[i] + row_offset;
		                    }
		                    iface_row_dofs_plus.resize(plus_row_local.size());
		                    for (std::size_t i = 0; i < plus_row_local.size(); ++i) {
		                        iface_row_dofs_plus[i] = plus_row_local[i] + row_offset;
		                    }

		                    iface_col_dofs_minus.resize(minus_col_local.size());
		                    for (std::size_t j = 0; j < minus_col_local.size(); ++j) {
		                        iface_col_dofs_minus[j] = minus_col_local[j] + col_offset;
		                    }
		                    iface_col_dofs_plus.resize(plus_col_local.size());
		                    for (std::size_t j = 0; j < plus_col_local.size(); ++j) {
		                        iface_col_dofs_plus[j] = plus_col_local[j] + col_offset;
		                    }

		                    add_element_couplings(iface_row_dofs_minus, iface_col_dofs_minus);
		                    add_element_couplings(iface_row_dofs_plus, iface_col_dofs_plus);
		                    add_element_couplings(iface_row_dofs_minus, iface_col_dofs_plus);
		                    add_element_couplings(iface_row_dofs_plus, iface_col_dofs_minus);
		                }
		            };

		            if (marker < 0) {
		                FE_THROW_IF(interface_meshes_.empty(), InvalidStateException,
		                            "FESystem::setup: interface-face kernels registered for all markers, but no InterfaceMesh was set");
		                for (const auto& kv : interface_meshes_) {
		                    if (!kv.second) continue;
		                    add_interface_mesh_couplings(*kv.second);
		                }
		            } else {
		                auto it = interface_meshes_.find(marker);
		                FE_THROW_IF(it == interface_meshes_.end() || !it->second, InvalidStateException,
		                            "FESystem::setup: missing InterfaceMesh for interface marker " + std::to_string(marker));
		                add_interface_mesh_couplings(*it->second);
		            }
		        }
#endif

	        // Global terms: allow kernels to conservatively augment sparsity.
	        for (const auto& kernel : def.global) {
	            if (kernel) {
	                kernel->addSparsityCouplings(*this, *pattern);
	            }
        }

	        // Coupled boundary-condition Jacobian: low-rank outer products introduce dense
	        // couplings between boundary-local DOFs on the BC marker and the DOFs that
	        // influence the referenced BoundaryFunctionals. Conservatively preallocate
	        // these couplings so strict backends (e.g., Trilinos) can accept insertions.
	        if (coupled_boundary_) {
	            const auto regs = coupled_boundary_->registeredBoundaryFunctionals();
	            if (!regs.empty() && !def.boundary.empty()) {
	                const auto primary_field = coupled_boundary_->primaryField();
	                const auto primary_idx = static_cast<std::size_t>(primary_field);
	                if (primary_field >= 0 && primary_idx < field_dof_handlers_.size()) {
	                    const auto& primary_map = field_dof_handlers_[primary_idx].getDofMap();
	                    const auto primary_offset = field_dof_offsets_[primary_idx];

	                    std::unordered_map<int, std::vector<GlobalIndex>> dofs_by_marker_primary;
	                    dofs_by_marker_primary.reserve(regs.size());

	                    auto collect_marker_dofs = [&](int marker,
	                                                  const dofs::DofMap& map,
	                                                  GlobalIndex offset,
	                                                  std::vector<GlobalIndex>& out) {
	                        out.clear();
	                        mesh_access_->forEachBoundaryFace(marker, [&](GlobalIndex /*face_id*/, GlobalIndex cell_id) {
	                            const auto cell_dofs = map.getCellDofs(cell_id);
	                            out.insert(out.end(), cell_dofs.begin(), cell_dofs.end());
	                        });
	                        for (auto& v : out) v += offset;
	                        std::sort(out.begin(), out.end());
	                        out.erase(std::unique(out.begin(), out.end()), out.end());
	                    };

	                    for (const auto& bf : regs) {
	                        const int marker = bf.def.boundary_marker;
	                        if (marker < 0) continue;
	                        if (dofs_by_marker_primary.count(marker) != 0u) continue;
	                        std::vector<GlobalIndex> tmp;
	                        collect_marker_dofs(marker, primary_map, primary_offset, tmp);
	                        dofs_by_marker_primary.emplace(marker, std::move(tmp));
	                    }

	                    std::unordered_map<std::uint64_t, std::vector<GlobalIndex>> row_dofs_cache;
	                    row_dofs_cache.reserve(def.boundary.size());

	                    auto key_of = [](int marker, FieldId test_field) -> std::uint64_t {
	                        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(marker)) << 32) |
	                               static_cast<std::uint32_t>(test_field);
	                    };

	                    for (const auto& term : def.boundary) {
	                        const int marker = term.marker;
	                        if (marker < 0) continue;

	                        const auto k = key_of(marker, term.test_field);
	                        if (row_dofs_cache.count(k) == 0u) {
	                            const auto test_idx = static_cast<std::size_t>(term.test_field);
	                            if (term.test_field < 0 || test_idx >= field_dof_handlers_.size()) {
	                                continue;
	                            }
	                            const auto& row_map = field_dof_handlers_[test_idx].getDofMap();
	                            const auto row_offset = field_dof_offsets_[test_idx];
	                            std::vector<GlobalIndex> tmp;
	                            collect_marker_dofs(marker, row_map, row_offset, tmp);
	                            row_dofs_cache.emplace(k, std::move(tmp));
	                        }

	                        const auto& row_dofs = row_dofs_cache.at(k);
	                        if (row_dofs.empty()) continue;

	                        for (const auto& bf : regs) {
	                            const int q_marker = bf.def.boundary_marker;
	                            auto it = dofs_by_marker_primary.find(q_marker);
	                            if (it == dofs_by_marker_primary.end()) continue;
	                            const auto& col_dofs = it->second;
	                            if (col_dofs.empty()) continue;
	                            pattern->addElementCouplings(row_dofs, col_dofs);
	                        }
	                    }
	                }
	            }
	        }

		        if (opts.sparsity_options.ensure_diagonal) {
		            pattern->ensureDiagonal();
		        }
	        if (opts.sparsity_options.ensure_non_empty_rows) {
	            pattern->ensureNonEmptyRows();
	        }
	        if (dist_pattern) {
	            if (opts.sparsity_options.ensure_diagonal) {
	                dist_pattern->ensureDiagonal();
	            }
	            if (opts.sparsity_options.ensure_non_empty_rows) {
	                dist_pattern->ensureNonEmptyRows();
	            }
	        }
	
	        if (opts.use_constraints_in_assembly && !affine_constraints_.empty()) {
	            auto query = std::make_shared<AffineConstraintsQuery>(affine_constraints_);
	            sparsity::ConstraintSparsityAugmenter augmenter(std::move(query));
	            augmenter.augment(*pattern, sparsity::AugmentationMode::EliminationFill);
	            if (dist_pattern) {
	                augmenter.augment(*dist_pattern, sparsity::AugmentationMode::EliminationFill);
	            }
	        }
	
	        if (opts.sparsity_options.ensure_diagonal) {
	            pattern->ensureDiagonal();
	        }
	        if (opts.sparsity_options.ensure_non_empty_rows) {
	            pattern->ensureNonEmptyRows();
	        }
	        if (dist_pattern) {
	            if (opts.sparsity_options.ensure_diagonal) {
	                dist_pattern->ensureDiagonal();
	            }
	            if (opts.sparsity_options.ensure_non_empty_rows) {
	                dist_pattern->ensureNonEmptyRows();
	            }
	        }
	
	        pattern->finalize();
	        sparsity_by_op_.emplace(tag, std::move(pattern));

	        if (dist_pattern) {
	            dist_pattern->finalize();

	            // Store optional ghost-row sparsity using the locally relevant (owned+ghost) overlap.
	            // This is required by overlap-style MPI backends (e.g., FSILS) so all column nodes
	            // referenced by owned rows are present locally.
	            auto ghost_rows_all = ghost_set.toVector();
	            std::vector<GlobalIndex> ghost_row_map;
	            ghost_row_map.reserve(ghost_rows_all.size());
	            for (const auto row : ghost_rows_all) {
	                if (row < 0 || row >= n_total_dofs) {
	                    continue;
	                }
	                if (owned_range.contains(row)) {
	                    continue;
	                }
	                ghost_row_map.push_back(row);
	            }

	            if (!ghost_row_map.empty()) {
	                // ghost_set.toVector() is already sorted/unique; keep it that way after filtering.
	                std::vector<GlobalIndex> ghost_row_ptr;
	                std::vector<GlobalIndex> ghost_row_cols_flat;
	                ghost_row_ptr.reserve(ghost_row_map.size() + 1);
	                ghost_row_ptr.push_back(0);

	                for (const auto row : ghost_row_map) {
	                    auto cols_it = ghost_row_cols.find(row);
	                    std::vector<GlobalIndex> cols_vec;
	                    if (cols_it != ghost_row_cols.end()) {
	                        cols_vec = cols_it->second;
	                    }
	                    cols_vec.push_back(row); // ensure diagonal

	                    // Only store columns that are locally present (owned+ghost) so overlap backends can map them.
	                    cols_vec.erase(std::remove_if(cols_vec.begin(),
	                                                 cols_vec.end(),
	                                                 [&](GlobalIndex col) {
	                                                     return (col < 0) || (col >= n_total_dofs) || !relevant_set.contains(col);
	                                                 }),
	                                   cols_vec.end());

	                    std::sort(cols_vec.begin(), cols_vec.end());
	                    cols_vec.erase(std::unique(cols_vec.begin(), cols_vec.end()), cols_vec.end());

	                    ghost_row_cols_flat.insert(ghost_row_cols_flat.end(), cols_vec.begin(), cols_vec.end());
	                    ghost_row_ptr.push_back(static_cast<GlobalIndex>(ghost_row_cols_flat.size()));
	                }

	                dist_pattern->setGhostRows(std::move(ghost_row_map),
	                                          std::move(ghost_row_ptr),
	                                          std::move(ghost_row_cols_flat));
	            } else {
	                dist_pattern->clearGhostRows();
	            }

	            distributed_sparsity_by_op_.emplace(tag, std::move(dist_pattern));
	        }
	    }

    // ---------------------------------------------------------------------
    // Per-cell material state storage (optional; for RequiredData::MaterialState)
    // ---------------------------------------------------------------------
    material_state_provider_.reset();
    {
        std::vector<GlobalIndex> boundary_faces;
        meshAccess().forEachBoundaryFace(/*marker=*/-1, [&](GlobalIndex face_id, GlobalIndex /*cell_id*/) {
            boundary_faces.push_back(face_id);
        });

        std::vector<GlobalIndex> interior_faces;
        meshAccess().forEachInteriorFace([&](GlobalIndex face_id, GlobalIndex /*cell_minus*/, GlobalIndex /*cell_plus*/) {
            interior_faces.push_back(face_id);
        });

        auto provider = std::make_unique<MaterialStateProvider>(meshAccess().numCells(),
                                                                std::move(boundary_faces),
                                                                std::move(interior_faces));
        bool any = false;

        for (const auto& tag : operator_registry_.list()) {
            const auto& def = operator_registry_.get(tag);

            for (const auto& term : def.cells) {
                if (!term.kernel) continue;

                const auto required = term.kernel->getRequiredData();
                if (!assembly::hasFlag(required, assembly::RequiredData::MaterialState)) {
                    continue;
                }

                const auto spec = term.kernel->materialStateSpec();
                FE_THROW_IF(spec.bytes_per_qpt == 0u, InvalidArgumentException,
                            "FESystem::setup: kernel requests MaterialState but bytes_per_qpt == 0");

                const auto& test_field = field_registry_.get(term.test_field);
                FE_CHECK_NOT_NULL(test_field.space.get(), "FESystem::setup: test_field.space");
                const auto max_qpts = maxCellQuadraturePoints(meshAccess(), *test_field.space);
                FE_THROW_IF(max_qpts <= 0, InvalidStateException,
                            "FESystem::setup: failed to determine max quadrature points for MaterialState allocation");

                provider->addKernel(*term.kernel, spec, max_qpts);
                any = true;
            }

            for (const auto& term : def.boundary) {
                if (!term.kernel) continue;
                const auto required = term.kernel->getRequiredData();
                if (!assembly::hasFlag(required, assembly::RequiredData::MaterialState)) {
                    continue;
                }

                const auto spec = term.kernel->materialStateSpec();
                FE_THROW_IF(spec.bytes_per_qpt == 0u, InvalidArgumentException,
                            "FESystem::setup: kernel requests MaterialState but bytes_per_qpt == 0 (boundary)");

                const auto& test_field = field_registry_.get(term.test_field);
                const auto& trial_field = field_registry_.get(term.trial_field);
                FE_CHECK_NOT_NULL(test_field.space.get(), "FESystem::setup: boundary test_field.space");
                FE_CHECK_NOT_NULL(trial_field.space.get(), "FESystem::setup: boundary trial_field.space");

                const auto max_qpts = maxBoundaryFaceQuadraturePoints(meshAccess(), *test_field.space, *trial_field.space);
                if (max_qpts > 0) {
                    provider->addKernel(*term.kernel, spec,
                                        /*max_cell_qpts=*/0,
                                        /*max_boundary_face_qpts=*/max_qpts,
                                        /*max_interior_face_qpts=*/0);
                    any = true;
                }
            }
            for (const auto& term : def.interior) {
                if (!term.kernel) continue;
                const auto required = term.kernel->getRequiredData();
                if (!assembly::hasFlag(required, assembly::RequiredData::MaterialState)) {
                    continue;
                }

                const auto spec = term.kernel->materialStateSpec();
                FE_THROW_IF(spec.bytes_per_qpt == 0u, InvalidArgumentException,
                            "FESystem::setup: kernel requests MaterialState but bytes_per_qpt == 0 (interior)");

                const auto& test_field = field_registry_.get(term.test_field);
                const auto& trial_field = field_registry_.get(term.trial_field);
                FE_CHECK_NOT_NULL(test_field.space.get(), "FESystem::setup: interior test_field.space");
                FE_CHECK_NOT_NULL(trial_field.space.get(), "FESystem::setup: interior trial_field.space");

                const auto max_qpts = maxInteriorFaceQuadraturePoints(meshAccess(), *test_field.space, *trial_field.space);
                if (max_qpts > 0) {
                    provider->addKernel(*term.kernel, spec,
                                        /*max_cell_qpts=*/0,
                                        /*max_boundary_face_qpts=*/0,
                                        /*max_interior_face_qpts=*/max_qpts);
                    any = true;
                }
            }
        }

        if (any) {
            material_state_provider_ = std::move(provider);
        }
    }

    // ---------------------------------------------------------------------
    // Global-kernel persistent state storage (optional; for GlobalStateSpec)
    // ---------------------------------------------------------------------
    global_kernel_state_provider_.reset();
    {
        std::vector<GlobalIndex> boundary_faces;
        meshAccess().forEachBoundaryFace(/*marker=*/-1, [&](GlobalIndex face_id, GlobalIndex /*cell_id*/) {
            boundary_faces.push_back(face_id);
        });

        std::vector<GlobalIndex> interior_faces;
        meshAccess().forEachInteriorFace([&](GlobalIndex face_id, GlobalIndex /*cell_minus*/, GlobalIndex /*cell_plus*/) {
            interior_faces.push_back(face_id);
        });

        auto provider = std::make_unique<GlobalKernelStateProvider>(meshAccess().numCells(),
                                                                    std::move(boundary_faces),
                                                                    std::move(interior_faces));

        bool any = false;
        for (const auto& tag : operator_registry_.list()) {
            const auto& def = operator_registry_.get(tag);
            for (const auto& kernel : def.global) {
                if (!kernel) continue;
                const auto spec = kernel->globalStateSpec();
                if (spec.empty()) continue;

                provider->addKernel(*kernel, spec);
                any = true;
            }
        }

        if (any) {
            global_kernel_state_provider_ = std::move(provider);
        }
    }

    // ---------------------------------------------------------------------
    // Parameter requirements (optional)
    // ---------------------------------------------------------------------
    parameter_registry_.clear();
    {
        for (const auto& tag : operator_registry_.list()) {
            const auto& def = operator_registry_.get(tag);

            for (const auto& term : def.cells) {
                if (!term.kernel) continue;
                parameter_registry_.addAll(term.kernel->parameterSpecs(), term.kernel->name());
            }
            for (const auto& term : def.boundary) {
                if (!term.kernel) continue;
                parameter_registry_.addAll(term.kernel->parameterSpecs(), term.kernel->name());
            }
            for (const auto& term : def.interior) {
                if (!term.kernel) continue;
                parameter_registry_.addAll(term.kernel->parameterSpecs(), term.kernel->name());
            }
            for (const auto& kernel : def.global) {
                if (!kernel) continue;
                parameter_registry_.addAll(kernel->parameterSpecs(), kernel->name());
            }
        }

        if (coupled_boundary_) {
            parameter_registry_.addAll(coupled_boundary_->parameterSpecs(), "CoupledBoundaryManager");
        }
    }

    // ---------------------------------------------------------------------
    // Resolve FE/Forms constitutive calls for JIT-fast mode (setup-time)
    // ---------------------------------------------------------------------
    {
        for (const auto& tag : operator_registry_.list()) {
            const auto& def = operator_registry_.get(tag);

            for (const auto& term : def.cells) {
                if (term.kernel) {
                    term.kernel->resolveInlinableConstitutives();
                }
            }
            for (const auto& term : def.boundary) {
                if (term.kernel) {
                    term.kernel->resolveInlinableConstitutives();
                }
            }
            for (const auto& term : def.interior) {
                if (term.kernel) {
                    term.kernel->resolveInlinableConstitutives();
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Resolve FE/Forms ParameterSymbol -> ParameterRef(slot) (setup-time)
    // ---------------------------------------------------------------------
    {
        const auto slot_of_real_param = [&](std::string_view key) -> std::optional<std::uint32_t> {
            return parameter_registry_.slotOf(key);
        };

        for (const auto& tag : operator_registry_.list()) {
            const auto& def = operator_registry_.get(tag);

            for (const auto& term : def.cells) {
                if (term.kernel) {
                    term.kernel->resolveParameterSlots(slot_of_real_param);
                }
            }
            for (const auto& term : def.boundary) {
                if (term.kernel) {
                    term.kernel->resolveParameterSlots(slot_of_real_param);
                }
            }
            for (const auto& term : def.interior) {
                if (term.kernel) {
                    term.kernel->resolveParameterSlots(slot_of_real_param);
                }
            }
            for (const auto& term : def.interface_faces) {
                if (term.kernel) {
                    term.kernel->resolveParameterSlots(slot_of_real_param);
                }
            }
        }

        if (coupled_boundary_) {
            coupled_boundary_->resolveParameterSlots(slot_of_real_param);
        }
    }

    // ---------------------------------------------------------------------
    // Assembler configuration
    // ---------------------------------------------------------------------
    assembly::FormCharacteristics form_chars{};
    {
        for (const auto& tag : operator_registry_.list()) {
            const auto& def = operator_registry_.get(tag);

            form_chars.has_cell_terms = form_chars.has_cell_terms || !def.cells.empty();
            form_chars.has_boundary_terms = form_chars.has_boundary_terms || !def.boundary.empty();
            form_chars.has_interior_face_terms = form_chars.has_interior_face_terms || !def.interior.empty();
            form_chars.has_interface_face_terms = form_chars.has_interface_face_terms || !def.interface_faces.empty();
            form_chars.has_global_terms = form_chars.has_global_terms || !def.global.empty();

            const auto mergeKernelMeta = [&](const std::shared_ptr<assembly::AssemblyKernel>& k) {
                if (!k) return;
                form_chars.required_data |= k->getRequiredData();
                form_chars.max_time_derivative_order =
                    std::max(form_chars.max_time_derivative_order, k->maxTemporalDerivativeOrder());
                form_chars.has_field_requirements = form_chars.has_field_requirements || !k->fieldRequirements().empty();
                form_chars.has_parameter_specs = form_chars.has_parameter_specs || !k->parameterSpecs().empty();
            };

            for (const auto& term : def.cells) {
                mergeKernelMeta(term.kernel);
            }
            for (const auto& term : def.boundary) {
                mergeKernelMeta(term.kernel);
            }
            for (const auto& term : def.interior) {
                mergeKernelMeta(term.kernel);
            }
            for (const auto& term : def.interface_faces) {
                mergeKernelMeta(term.kernel);
            }
            for (const auto& k : def.global) {
                if (!k) continue;
                form_chars.has_parameter_specs = form_chars.has_parameter_specs || !k->parameterSpecs().empty();
            }
        }
    }

    assembly::SystemCharacteristics sys_chars{};
    sys_chars.num_fields = field_registry_.size();
    sys_chars.num_cells = meshAccess().numCells();
    sys_chars.dimension = meshAccess().dimension();
    sys_chars.num_dofs_total = dof_handler_.getDofMap().getNumDofs();
    sys_chars.max_dofs_per_cell = dof_handler_.getDofMap().getMaxDofsPerCell();

    for (const auto& rec : field_registry_.records()) {
        if (!rec.space) continue;
        sys_chars.max_polynomial_order = std::max(sys_chars.max_polynomial_order, rec.space->polynomial_order());
    }

    // Resolve thread count for reporting/heuristics (0 means "auto").
    sys_chars.num_threads = opts.assembly_options.num_threads;
    if (sys_chars.num_threads <= 0) {
        const auto hw = std::max(1u, std::thread::hardware_concurrency());
        sys_chars.num_threads = static_cast<int>(hw);
    }

    sys_chars.mpi_world_size = 1;
#if FE_HAS_MPI
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        MPI_Comm_size(MPI_COMM_WORLD, &sys_chars.mpi_world_size);
    }
#endif

    assembler_selection_report_.clear();
    assembler_ = assembly::createAssembler(opts.assembly_options, opts.assembler_name,
                                           form_chars, sys_chars, &assembler_selection_report_);
    FE_CHECK_NOT_NULL(assembler_.get(), "FESystem::setup: assembler");
    assembler_->setDofHandler(dof_handler_);

    if (opts.use_constraints_in_assembly) {
        assembler_->setConstraints(&affine_constraints_);
    } else {
        assembler_->setConstraints(nullptr);
    }

    assembler_->setMaterialStateProvider(material_state_provider_.get());
    assembler_->setOptions(opts.assembly_options);
    assembler_->initialize();

    // ---------------------------------------------------------------------
    // Optional: Auto-register matrix-free operators (explicit opt-in)
    // ---------------------------------------------------------------------
    if (opts.auto_register_matrix_free) {
        FE_CHECK_NOT_NULL(operator_backends_.get(), "FESystem::setup: operator_backends");
        FE_THROW_IF(field_registry_.size() != 1u, NotImplementedException,
                    "FESystem::setup: auto_register_matrix_free currently requires a single-field system");

        std::size_t registered = 0;
        for (const auto& tag : operator_registry_.list()) {
            if (operator_backends_->hasMatrixFree(tag)) {
                continue; // Respect explicit user registration.
            }

            const auto& def = operator_registry_.get(tag);
            if (def.cells.empty()) continue;
            if (!def.boundary.empty() || !def.interior.empty() || !def.global.empty()) {
                continue; // Cell-only operators only (initial conservative scope).
            }

            // Build a (possibly composite) cell kernel for this operator.
            std::shared_ptr<assembly::AssemblyKernel> kernel_to_wrap;
            if (def.cells.size() == 1u) {
                kernel_to_wrap = def.cells.front().kernel;
            } else {
                auto composite = std::make_shared<assembly::CompositeKernel>();
                for (const auto& term : def.cells) {
                    if (!term.kernel) continue;
                    composite->addKernel(term.kernel);
                }
                kernel_to_wrap = std::move(composite);
            }

            if (!kernel_to_wrap) continue;

            // Conservative eligibility: linear, steady, cell-only.
            const auto required = kernel_to_wrap->getRequiredData();
            if (assembly::hasFlag(required, assembly::RequiredData::SolutionCoefficients) ||
                assembly::hasFlag(required, assembly::RequiredData::SolutionValues) ||
                assembly::hasFlag(required, assembly::RequiredData::SolutionGradients) ||
                assembly::hasFlag(required, assembly::RequiredData::SolutionHessians) ||
                assembly::hasFlag(required, assembly::RequiredData::SolutionLaplacians) ||
                assembly::hasFlag(required, assembly::RequiredData::MaterialState) ||
                kernel_to_wrap->maxTemporalDerivativeOrder() > 0) {
                continue;
            }

            auto mf_unique = assembly::wrapAsMatrixFreeKernel(kernel_to_wrap);
            std::shared_ptr<assembly::IMatrixFreeKernel> mf_kernel = std::move(mf_unique);
            operator_backends_->registerMatrixFree(tag, std::move(mf_kernel));
            ++registered;
        }

        if (registered > 0u) {
            if (!assembler_selection_report_.empty()) {
                assembler_selection_report_.append("\n");
            }
            assembler_selection_report_.append("Auto-registered matrix-free operators: ");
            assembler_selection_report_.append(std::to_string(registered));
        }
    }

    is_setup_ = true;
}

} // namespace systems
} // namespace FE
} // namespace svmp
