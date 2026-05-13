/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "ParallelAssembler.h"
#include "Dofs/DofMap.h"
#include "Dofs/DofHandler.h"
#include "Dofs/GhostDofManager.h"
#include "Constraints/AffineConstraints.h"
#include "Sparsity/SparsityPattern.h"
#include "Spaces/FunctionSpace.h"
#include "Elements/Element.h"

#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iterator>

namespace svmp {
namespace FE {
namespace assembly {

#if FE_HAS_MPI
namespace {
void set_mpi_rank_and_size(MPI_Comm comm, int& rank, int& size)
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
    } else {
        rank = 0;
        size = 1;
    }
}
} // namespace
#endif

namespace {

[[nodiscard]] GlobalIndex traced_ghost_row_dof() noexcept
{
    static const GlobalIndex traced = []() noexcept {
        const char* env = std::getenv("SVMP_GHOST_TRACE_ROW_DOF");
        if (env == nullptr || *env == '\0') {
            return INVALID_GLOBAL_INDEX;
        }
        char* end = nullptr;
        const auto value = std::strtoll(env, &end, 10);
        if (end == env) {
            return INVALID_GLOBAL_INDEX;
        }
        return static_cast<GlobalIndex>(value);
    }();
    return traced;
}

[[nodiscard]] bool should_trace_ghost_row(GlobalIndex row) noexcept
{
    return row == traced_ghost_row_dof();
}

class GhostRoutingView final : public GlobalSystemView {
public:
    GhostRoutingView(GlobalSystemView& base,
                     GhostContributionManager& ghost_manager,
                     GhostPolicy policy)
        : base_(&base)
        , ghost_manager_(&ghost_manager)
        , policy_(policy)
    {
    }

    // Matrix operations
    void addMatrixEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> local_matrix,
                          AddMode mode = AddMode::Add) override
    {
        addMatrixEntries(dofs, dofs, local_matrix, mode);
    }

    void addMatrixEntries(std::span<const GlobalIndex> row_dofs,
                          std::span<const GlobalIndex> col_dofs,
                          std::span<const Real> local_matrix,
                          AddMode mode = AddMode::Add) override
    {
        FE_CHECK_NOT_NULL(base_, "GhostRoutingView::base");

        if (policy_ != GhostPolicy::ReverseScatter) {
            base_->addMatrixEntries(row_dofs, col_dofs, local_matrix, mode);
            return;
        }

        owned_contributions_.clear();
        ghost_manager_->addMatrixContributions(row_dofs, col_dofs, local_matrix, owned_contributions_, mode);
        for (const auto& c : owned_contributions_) {
            if (mode == AddMode::Insert) {
                ghost_manager_->addMatrixEntryOperation(c.global_row, c.global_col, c.value, mode);
            } else {
                base_->addMatrixEntry(c.global_row, c.global_col, c.value, mode);
            }
        }
    }

    void addMatrixEntry(GlobalIndex row,
                        GlobalIndex col,
                        Real value,
                        AddMode mode = AddMode::Add) override
    {
        FE_CHECK_NOT_NULL(base_, "GhostRoutingView::base");

        if (ghost_manager_->isOwned(row)) {
            if (policy_ == GhostPolicy::ReverseScatter && mode == AddMode::Insert) {
                ghost_manager_->addMatrixEntryOperation(row, col, value, mode);
            } else {
                base_->addMatrixEntry(row, col, value, mode);
            }
            return;
        }

        if (policy_ == GhostPolicy::OwnedRowsOnly) {
            return;
        }

        ghost_manager_->addMatrixContribution(row, col, value, mode);
    }

    void setDiagonal(std::span<const GlobalIndex> dofs,
                     std::span<const Real> values) override
    {
        FE_CHECK_NOT_NULL(base_, "GhostRoutingView::base");
        FE_THROW_IF(dofs.size() != values.size(), FEException,
                    "GhostRoutingView::setDiagonal: size mismatch");
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            setDiagonal(dofs[i], values[i]);
        }
    }

    void setDiagonal(GlobalIndex dof, Real value) override
    {
        FE_CHECK_NOT_NULL(base_, "GhostRoutingView::base");
        if (policy_ != GhostPolicy::ReverseScatter) {
            if (!ghost_manager_->isOwned(dof)) {
                return;
            }
            base_->setDiagonal(dof, value);
            return;
        }
        ghost_manager_->addMatrixEntryOperation(dof, dof, value, AddMode::Insert);
    }

    void zeroRows(std::span<const GlobalIndex> rows,
                  bool set_diagonal = true) override
    {
        FE_CHECK_NOT_NULL(base_, "GhostRoutingView::base");
        if (policy_ == GhostPolicy::ReverseScatter) {
            for (const auto r : rows) {
                ghost_manager_->addMatrixRowOperation(r, set_diagonal);
            }
            return;
        }
        owned_rows_.clear();
        owned_rows_.reserve(rows.size());
        for (const auto r : rows) {
            if (ghost_manager_->isOwned(r)) {
                owned_rows_.push_back(r);
            }
        }
        if (!owned_rows_.empty()) {
            base_->zeroRows(std::span<const GlobalIndex>(owned_rows_), set_diagonal);
        }
    }

    // Vector operations
    void addVectorEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> local_vector,
                          AddMode mode = AddMode::Add) override
    {
        FE_CHECK_NOT_NULL(base_, "GhostRoutingView::base");
        FE_THROW_IF(dofs.size() != local_vector.size(), FEException,
                    "GhostRoutingView::addVectorEntries: size mismatch");

        for (std::size_t i = 0; i < dofs.size(); ++i) {
            addVectorEntry(dofs[i], local_vector[i], mode);
        }
    }

    void addVectorEntry(GlobalIndex dof,
                        Real value,
                        AddMode mode = AddMode::Add) override
    {
        FE_CHECK_NOT_NULL(base_, "GhostRoutingView::base");

        if (ghost_manager_->isOwned(dof)) {
            if (policy_ == GhostPolicy::ReverseScatter && mode == AddMode::Insert) {
                ghost_manager_->addVectorEntryOperation(dof, value, mode);
            } else {
                base_->addVectorEntry(dof, value, mode);
            }
            return;
        }

        if (policy_ == GhostPolicy::OwnedRowsOnly) {
            return;
        }

        ghost_manager_->addVectorContribution(dof, value, mode);
    }

    void setVectorEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> values) override
    {
        FE_CHECK_NOT_NULL(base_, "GhostRoutingView::base");
        FE_THROW_IF(dofs.size() != values.size(), FEException,
                    "GhostRoutingView::setVectorEntries: size mismatch");
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            if (policy_ == GhostPolicy::ReverseScatter) {
                ghost_manager_->addVectorEntryOperation(dofs[i], values[i], AddMode::Insert);
            } else {
                if (!ghost_manager_->isOwned(dofs[i])) {
                    continue;
                }
                base_->setVectorEntries(std::span<const GlobalIndex>(&dofs[i], 1u),
                                        std::span<const Real>(&values[i], 1u));
            }
        }
    }

    void zeroVectorEntries(std::span<const GlobalIndex> dofs) override
    {
        FE_CHECK_NOT_NULL(base_, "GhostRoutingView::base");
        if (policy_ == GhostPolicy::ReverseScatter) {
            for (const auto d : dofs) {
                ghost_manager_->addVectorZeroOperation(d);
            }
            return;
        }
        owned_rows_.clear();
        owned_rows_.reserve(dofs.size());
        for (const auto d : dofs) {
            if (ghost_manager_->isOwned(d)) {
                owned_rows_.push_back(d);
            }
        }
        if (!owned_rows_.empty()) {
            base_->zeroVectorEntries(std::span<const GlobalIndex>(owned_rows_));
        }
    }

    [[nodiscard]] Real getVectorEntry(GlobalIndex dof) const override
    {
        FE_CHECK_NOT_NULL(base_, "GhostRoutingView::base");
        return base_->getVectorEntry(dof);
    }

    // Assembly lifecycle
    void beginAssemblyPhase() override
    {
        FE_CHECK_NOT_NULL(base_, "GhostRoutingView::base");
        base_->beginAssemblyPhase();
    }

    void endAssemblyPhase() override
    {
        FE_CHECK_NOT_NULL(base_, "GhostRoutingView::base");
        base_->endAssemblyPhase();
    }

    void finalizeAssembly() override
    {
        FE_CHECK_NOT_NULL(base_, "GhostRoutingView::base");
        base_->finalizeAssembly();
    }

    [[nodiscard]] AssemblyPhase getPhase() const noexcept override
    {
        return base_ ? base_->getPhase() : AssemblyPhase::NotStarted;
    }

    // Properties
    [[nodiscard]] bool hasMatrix() const noexcept override { return base_ && base_->hasMatrix(); }
    [[nodiscard]] bool hasVector() const noexcept override { return base_ && base_->hasVector(); }
    [[nodiscard]] GlobalIndex numRows() const noexcept override { return base_ ? base_->numRows() : 0; }
    [[nodiscard]] GlobalIndex numCols() const noexcept override { return base_ ? base_->numCols() : 0; }
    [[nodiscard]] std::string backendName() const override
    {
        return base_ ? ("GhostRouting(" + base_->backendName() + ")") : "GhostRouting(null)";
    }

    // Zero and access
    void zero() override
    {
        FE_CHECK_NOT_NULL(base_, "GhostRoutingView::base");
        base_->zero();
    }

    [[nodiscard]] Real getMatrixEntry(GlobalIndex row, GlobalIndex col) const override
    {
        FE_CHECK_NOT_NULL(base_, "GhostRoutingView::base");
        return base_->getMatrixEntry(row, col);
    }

private:
    GlobalSystemView* base_{nullptr};
    GhostContributionManager* ghost_manager_{nullptr};
    GhostPolicy policy_{GhostPolicy::ReverseScatter};

    std::vector<GhostContribution> owned_contributions_;
    std::vector<GlobalIndex> owned_rows_;
};

class CellListMeshAccess final : public IMeshAccess {
public:
    CellListMeshAccess(const IMeshAccess& base, std::span<const GlobalIndex> cell_ids)
        : base_(&base), cell_ids_(cell_ids)
    {
    }

    [[nodiscard]] GlobalIndex numCells() const override { return static_cast<GlobalIndex>(cell_ids_.size()); }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return static_cast<GlobalIndex>(cell_ids_.size()); }
    [[nodiscard]] GlobalIndex numVertices() const override { return base_->numVertices(); }
    [[nodiscard]] GlobalIndex numOwnedVertices() const override { return base_->numOwnedVertices(); }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return base_->numBoundaryFaces(); }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return base_->numInteriorFaces(); }
    [[nodiscard]] int dimension() const override { return base_->dimension(); }
    [[nodiscard]] bool isOwnedCell(GlobalIndex cell_id) const override { return base_->isOwnedCell(cell_id); }
    [[nodiscard]] ElementType getCellType(GlobalIndex cell_id) const override { return base_->getCellType(cell_id); }
    [[nodiscard]] int getCellDomainId(GlobalIndex cell_id) const override { return base_->getCellDomainId(cell_id); }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override
    {
        base_->getCellNodes(cell_id, nodes);
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return base_->getNodeCoordinates(node_id);
    }

    void getCellCoordinates(GlobalIndex cell_id, std::vector<std::array<Real, 3>>& coords) const override
    {
        base_->getCellCoordinates(cell_id, coords);
    }

    [[nodiscard]] bool supportsCoordinateFrame(CoordinateFrame frame) const override
    {
        return base_->supportsCoordinateFrame(frame);
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            CoordinateFrame frame,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        base_->getCellCoordinates(cell_id, frame, coords);
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex face_id, GlobalIndex cell_id) const override
    {
        return base_->getLocalFaceIndex(face_id, cell_id);
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex face_id) const override
    {
        return base_->getBoundaryFaceMarker(face_id);
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex face_id) const override
    {
        return base_->getInteriorFaceCells(face_id);
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (const auto cell : cell_ids_) {
            callback(cell);
        }
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        forEachCell(std::move(callback));
    }

    void forEachBoundaryFace(int marker,
                             std::function<void(GlobalIndex, GlobalIndex)> callback) const override
    {
        base_->forEachBoundaryFace(marker, std::move(callback));
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback) const override
    {
        base_->forEachInteriorFace(std::move(callback));
    }

private:
    const IMeshAccess* base_{nullptr};
    std::span<const GlobalIndex> cell_ids_{};
};

class AllLocalCellsMeshAccess final : public IMeshAccess {
public:
    explicit AllLocalCellsMeshAccess(const IMeshAccess& base)
        : base_(&base)
    {
    }

    [[nodiscard]] GlobalIndex numCells() const override { return base_->numCells(); }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return base_->numCells(); }
    [[nodiscard]] GlobalIndex numVertices() const override { return base_->numVertices(); }
    [[nodiscard]] GlobalIndex numOwnedVertices() const override { return base_->numVertices(); }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return base_->numBoundaryFaces(); }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return base_->numInteriorFaces(); }
    [[nodiscard]] int dimension() const override { return base_->dimension(); }
    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex cell_id) const override { return base_->getCellType(cell_id); }
    [[nodiscard]] int getCellDomainId(GlobalIndex cell_id) const override { return base_->getCellDomainId(cell_id); }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override
    {
        base_->getCellNodes(cell_id, nodes);
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return base_->getNodeCoordinates(node_id);
    }

    void getCellCoordinates(GlobalIndex cell_id, std::vector<std::array<Real, 3>>& coords) const override
    {
        base_->getCellCoordinates(cell_id, coords);
    }

    [[nodiscard]] bool supportsCoordinateFrame(CoordinateFrame frame) const override
    {
        return base_->supportsCoordinateFrame(frame);
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            CoordinateFrame frame,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        base_->getCellCoordinates(cell_id, frame, coords);
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex face_id, GlobalIndex cell_id) const override
    {
        return base_->getLocalFaceIndex(face_id, cell_id);
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex face_id) const override
    {
        return base_->getBoundaryFaceMarker(face_id);
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex face_id) const override
    {
        return base_->getInteriorFaceCells(face_id);
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        base_->forEachCell(std::move(callback));
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        base_->forEachCell(std::move(callback));
    }

    void forEachBoundaryFace(int marker,
                             std::function<void(GlobalIndex, GlobalIndex)> callback) const override
    {
        base_->forEachBoundaryFace(marker, std::move(callback));
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback) const override
    {
        base_->forEachInteriorFace(std::move(callback));
    }

private:
    const IMeshAccess* base_{nullptr};
};

class OwnedCellsMeshAccess final : public IMeshAccess {
public:
    explicit OwnedCellsMeshAccess(const IMeshAccess& base)
        : base_(&base)
    {
    }

    [[nodiscard]] GlobalIndex numCells() const override { return base_->numCells(); }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return base_->numCells(); }
    [[nodiscard]] GlobalIndex numVertices() const override { return base_->numVertices(); }
    [[nodiscard]] GlobalIndex numOwnedVertices() const override { return base_->numVertices(); }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return base_->numBoundaryFaces(); }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return base_->numInteriorFaces(); }
    [[nodiscard]] int dimension() const override { return base_->dimension(); }
    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex cell_id) const override { return base_->getCellType(cell_id); }
    [[nodiscard]] int getCellDomainId(GlobalIndex cell_id) const override { return base_->getCellDomainId(cell_id); }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override
    {
        base_->getCellNodes(cell_id, nodes);
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return base_->getNodeCoordinates(node_id);
    }

    void getCellCoordinates(GlobalIndex cell_id, std::vector<std::array<Real, 3>>& coords) const override
    {
        base_->getCellCoordinates(cell_id, coords);
    }

    [[nodiscard]] bool supportsCoordinateFrame(CoordinateFrame frame) const override
    {
        return base_->supportsCoordinateFrame(frame);
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            CoordinateFrame frame,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        base_->getCellCoordinates(cell_id, frame, coords);
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex face_id, GlobalIndex cell_id) const override
    {
        return base_->getLocalFaceIndex(face_id, cell_id);
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex face_id) const override
    {
        return base_->getBoundaryFaceMarker(face_id);
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex face_id) const override
    {
        return base_->getInteriorFaceCells(face_id);
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        base_->forEachOwnedCell(std::move(callback));
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        base_->forEachOwnedCell(std::move(callback));
    }

    void forEachBoundaryFace(int marker,
                             std::function<void(GlobalIndex, GlobalIndex)> callback) const override
    {
        base_->forEachBoundaryFace(marker, [&](GlobalIndex face_id, GlobalIndex cell_id) {
            if (base_->isOwnedCell(cell_id)) {
                callback(face_id, cell_id);
            }
        });
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback) const override
    {
        base_->forEachInteriorFace(
            [&](GlobalIndex face_id, GlobalIndex cell_minus, GlobalIndex cell_plus) {
                if (base_->isOwnedCell(cell_minus)) {
                    callback(face_id, cell_minus, cell_plus);
                }
            });
    }

private:
    const IMeshAccess* base_{nullptr};
};

template <typename Callback>
AssemblyResult withPolicyMeshAccess(const IMeshAccess& mesh,
                                    GhostPolicy policy,
                                    Callback&& callback)
{
    if (policy == GhostPolicy::ReverseScatter) {
        OwnedCellsMeshAccess owned_mesh(mesh);
        return callback(owned_mesh);
    }

    AllLocalCellsMeshAccess all_local_mesh(mesh);
    return callback(all_local_mesh);
}

} // namespace

// ============================================================================
// Construction
// ============================================================================

ParallelAssembler::ParallelAssembler()
{
#if FE_HAS_MPI
    set_mpi_rank_and_size(comm_, my_rank_, world_size_);
    ghost_manager_.setComm(comm_);
#endif
}

#if FE_HAS_MPI
ParallelAssembler::ParallelAssembler(MPI_Comm comm)
    : comm_(comm)
{
    set_mpi_rank_and_size(comm_, my_rank_, world_size_);
    ghost_manager_.setComm(comm_);
}
#endif

ParallelAssembler::ParallelAssembler(const AssemblyOptions& options)
    : options_(options)
    , ghost_policy_(options.ghost_policy)
    , overlap_comm_(options.overlap_communication)
{
#if FE_HAS_MPI
    set_mpi_rank_and_size(comm_, my_rank_, world_size_);
    ghost_manager_.setComm(comm_);
#endif
}

ParallelAssembler::~ParallelAssembler() = default;

ParallelAssembler::ParallelAssembler(ParallelAssembler&& other) noexcept = default;

ParallelAssembler& ParallelAssembler::operator=(ParallelAssembler&& other) noexcept = default;

// ============================================================================
// Configuration
// ============================================================================

void ParallelAssembler::setDofMap(const dofs::DofMap& dof_map)
{
    dof_map_ = &dof_map;
    local_assembler_.setDofMap(dof_map);
    ghost_manager_.setDofMap(dof_map);
    ghost_manager_.setOwnershipOffset(0);
}

void ParallelAssembler::setRowDofMap(const dofs::DofMap& dof_map,
                                     GlobalIndex row_offset,
                                     DofEntityScope row_scope)
{
    local_assembler_.setRowDofMap(dof_map, row_offset, row_scope);

    // If no system-level DOF handler is configured, fall back to using the row map
    // for ghost ownership queries. This mode cannot safely switch row maps during an
    // active assembly phase (ghost buffers would become inconsistent).
    if (dof_handler_ == nullptr) {
        FE_THROW_IF(assembly_in_progress_, FEException,
                    "ParallelAssembler::setRowDofMap: cannot change row DOF map during an active assembly phase "
                    "unless a system-level DofHandler is configured");
        dof_map_ = &dof_map;
        ghost_manager_.setDofMap(dof_map);
        ghost_manager_.setOwnershipOffset(row_offset);
    }
}

void ParallelAssembler::setColDofMap(const dofs::DofMap& dof_map,
                                     GlobalIndex col_offset,
                                     DofEntityScope col_scope)
{
    local_assembler_.setColDofMap(dof_map, col_offset, col_scope);
}

void ParallelAssembler::setDofHandler(const dofs::DofHandler& dof_handler)
{
    dof_handler_ = &dof_handler;
    dof_map_ = &dof_handler.getDofMap();
    local_assembler_.setDofHandler(dof_handler);
    ghost_manager_.setDofMap(*dof_map_);
    ghost_manager_.setOwnershipOffset(0);

    // Also set ghost DOF manager if available
    if (dof_handler.getGhostManager()) {
        setGhostDofManager(*dof_handler.getGhostManager());
    }
}

void ParallelAssembler::setConstraints(const constraints::AffineConstraints* constraints)
{
    constraints_ = constraints;
    local_assembler_.setConstraints(constraints);
}

void ParallelAssembler::setSuppressConstraintInhomogeneity(bool suppress)
{
    local_assembler_.setSuppressConstraintInhomogeneity(suppress);
}

void ParallelAssembler::setSparsityPattern(const sparsity::SparsityPattern* sparsity)
{
    sparsity_ = sparsity;
    local_assembler_.setSparsityPattern(sparsity);
}

void ParallelAssembler::setOptions(const AssemblyOptions& options)
{
    options_ = options;
    ghost_policy_ = options.ghost_policy;
    overlap_comm_ = options.overlap_communication;
    local_assembler_.setOptions(options);
    ghost_manager_.setPolicy(ghost_policy_);
    ghost_manager_.setDeterministic(options_.deterministic);
    ghost_manager_.setExplicitRowOwner(options_.row_owner_rank);
}

void ParallelAssembler::setCurrentSolution(std::span<const Real> solution)
{
    local_assembler_.setCurrentSolution(solution);
}

void ParallelAssembler::setCurrentSolutionView(const GlobalSystemView* solution_view)
{
    local_assembler_.setCurrentSolutionView(solution_view);
}

void ParallelAssembler::setFieldSolutionAccess(std::span<const FieldSolutionAccess> fields)
{
    local_assembler_.setFieldSolutionAccess(fields);
}

void ParallelAssembler::setMeshMotionFieldAccess(const MeshMotionFieldAccess& fields)
{
    local_assembler_.setMeshMotionFieldAccess(fields);
}

void ParallelAssembler::setPreviousSolution(std::span<const Real> solution)
{
    local_assembler_.setPreviousSolution(solution);
}

void ParallelAssembler::setPreviousSolution2(std::span<const Real> solution)
{
    local_assembler_.setPreviousSolution2(solution);
}

void ParallelAssembler::setPreviousSolutionView(const GlobalSystemView* solution_view)
{
    local_assembler_.setPreviousSolutionView(solution_view);
}

void ParallelAssembler::setPreviousSolution2View(const GlobalSystemView* solution_view)
{
    local_assembler_.setPreviousSolution2View(solution_view);
}

void ParallelAssembler::setPreviousSolutionK(int k, std::span<const Real> solution)
{
    local_assembler_.setPreviousSolutionK(k, solution);
}

void ParallelAssembler::setPreviousSolutionViewK(int k, const GlobalSystemView* solution_view)
{
    local_assembler_.setPreviousSolutionViewK(k, solution_view);
}

void ParallelAssembler::setTimeIntegrationContext(const TimeIntegrationContext* ctx)
{
    local_assembler_.setTimeIntegrationContext(ctx);
}

void ParallelAssembler::setTime(Real time)
{
    local_assembler_.setTime(time);
}

void ParallelAssembler::setTimeStep(Real dt)
{
    local_assembler_.setTimeStep(dt);
}

void ParallelAssembler::setRealParameterGetter(
    const std::function<std::optional<Real>(std::string_view)>* get_real_param) noexcept
{
    local_assembler_.setRealParameterGetter(get_real_param);
}

void ParallelAssembler::setParameterGetter(
    const std::function<std::optional<params::Value>(std::string_view)>* get_param) noexcept
{
    local_assembler_.setParameterGetter(get_param);
}

void ParallelAssembler::setUserData(const void* user_data) noexcept
{
    local_assembler_.setUserData(user_data);
}

void ParallelAssembler::setJITConstants(std::span<const Real> constants) noexcept
{
    local_assembler_.setJITConstants(constants);
}

void ParallelAssembler::setCoupledValues(std::span<const Real> integrals,
                                         std::span<const Real> aux_state) noexcept
{
    local_assembler_.setCoupledValues(integrals, aux_state);
}

void ParallelAssembler::setAuxiliaryValues(std::span<const Real> inputs,
                                           std::span<const Real> state,
                                           std::span<const Real> outputs) noexcept
{
    local_assembler_.setAuxiliaryValues(inputs, state, outputs);
}

void ParallelAssembler::setAuxiliaryOutputBindings(
    std::span<const AuxiliaryOutputBinding> bindings) noexcept
{
    local_assembler_.setAuxiliaryOutputBindings(bindings);
}

void ParallelAssembler::setMaterialStateProvider(IMaterialStateProvider* provider) noexcept
{
    local_assembler_.setMaterialStateProvider(provider);
}

const AssemblyOptions& ParallelAssembler::getOptions() const noexcept
{
    return options_;
}

#if FE_HAS_MPI
void ParallelAssembler::setComm(MPI_Comm comm)
{
    comm_ = comm;
    set_mpi_rank_and_size(comm_, my_rank_, world_size_);
    ghost_manager_.setComm(comm);
}
#endif

void ParallelAssembler::setGhostDofManager(const dofs::GhostDofManager& ghost_manager)
{
    ghost_dof_manager_ = &ghost_manager;
    ghost_manager_.setGhostDofManager(ghost_manager);
}

void ParallelAssembler::setGhostPolicy(GhostPolicy policy)
{
    ghost_policy_ = policy;
    options_.ghost_policy = policy;
    ghost_manager_.setPolicy(policy);
    local_assembler_.setOptions(options_);
}

bool ParallelAssembler::isConfigured() const noexcept
{
    return dof_map_ != nullptr;
}

// ============================================================================
// Lifecycle
// ============================================================================

void ParallelAssembler::initialize()
{
    if (!isConfigured()) {
        throw std::runtime_error("ParallelAssembler::initialize: not configured");
    }

    // Initialize local assembler
    local_assembler_.initialize();

    // Initialize ghost manager
    ghost_manager_.setPolicy(ghost_policy_);
    ghost_manager_.setDeterministic(options_.deterministic);
    ghost_manager_.setExplicitRowOwner(options_.row_owner_rank);
    ghost_manager_.initialize();

    // Reserve working storage
    const auto max_dofs = dof_map_->getMaxDofsPerCell();
    row_dofs_.reserve(static_cast<std::size_t>(max_dofs));
    col_dofs_.reserve(static_cast<std::size_t>(max_dofs));
    owned_contributions_.reserve(static_cast<std::size_t>(max_dofs * max_dofs));

    // Estimate ghost contributions
    ghost_manager_.reserveBuffers(1000);  // Heuristic

    initialized_ = true;
    assembly_in_progress_ = false;
}

void ParallelAssembler::finalize(GlobalSystemView* matrix_view, GlobalSystemView* vector_view)
{
    // Exchange ghost contributions
    if (ghost_policy_ == GhostPolicy::ReverseScatter) {
        exchangeGhostContributions();
        if (std::getenv("SVMP_PARALLEL_ASSEMBLER_TRACE") != nullptr) {
            const auto& stats = ghost_manager_.getLastExchangeStats();
            std::fprintf(stderr,
                         "[R%d] ParallelAssembler exchange neighbors=%d "
                         "matrix_sent=%zu matrix_recv=%zu vector_sent=%zu vector_recv=%zu "
                         "bytes_sent=%zu bytes_recv=%zu time=%g\n",
                         my_rank_,
                         ghost_manager_.numNeighbors(),
                         stats.matrix_entries_sent,
                         stats.matrix_entries_received,
                         stats.vector_entries_sent,
                         stats.vector_entries_received,
                         stats.bytes_sent,
                         stats.bytes_received,
                         stats.exchange_time_seconds);
        }

        auto recv_matrix = ghost_manager_.takeReceivedMatrixContributions();
        if (!recv_matrix.empty()) {
            if (pending_received_matrix_.empty()) {
                pending_received_matrix_ = std::move(recv_matrix);
            } else {
                pending_received_matrix_.insert(pending_received_matrix_.end(),
                                                std::make_move_iterator(recv_matrix.begin()),
                                                std::make_move_iterator(recv_matrix.end()));
            }
        }
        auto local_matrix_ops = ghost_manager_.takeLocalMatrixOperations();
        if (!local_matrix_ops.empty()) {
            pending_received_matrix_.insert(pending_received_matrix_.end(),
                                            std::make_move_iterator(local_matrix_ops.begin()),
                                            std::make_move_iterator(local_matrix_ops.end()));
        }

        auto recv_vector = ghost_manager_.takeReceivedVectorContributions();
        if (!recv_vector.empty()) {
            if (pending_received_vector_.empty()) {
                pending_received_vector_ = std::move(recv_vector);
            } else {
                pending_received_vector_.insert(pending_received_vector_.end(),
                                                std::make_move_iterator(recv_vector.begin()),
                                                std::make_move_iterator(recv_vector.end()));
            }
        }
        auto local_vector_ops = ghost_manager_.takeLocalVectorOperations();
        if (!local_vector_ops.empty()) {
            pending_received_vector_.insert(pending_received_vector_.end(),
                                            std::make_move_iterator(local_vector_ops.begin()),
                                            std::make_move_iterator(local_vector_ops.end()));
        }

        if (matrix_view) {
            auto apply_entry = [&](const GhostContribution& entry) {
                if (should_trace_ghost_row(entry.global_row)) {
                    std::fprintf(stderr,
                                 "[R%d] ParallelAssembler apply matrix row=%lld col=%lld "
                                 "value=%.17g mode=%d op=%d\n",
                                 my_rank_,
                                 static_cast<long long>(entry.global_row),
                                 static_cast<long long>(entry.global_col),
                                 static_cast<double>(entry.value),
                                 static_cast<int>(entry.mode),
                                 static_cast<int>(entry.op));
                }
                matrix_view->addMatrixEntry(entry.global_row, entry.global_col, entry.value, entry.mode);
            };
            auto apply_zero_row = [&](const GhostContribution& entry) {
                if (should_trace_ghost_row(entry.global_row)) {
                    std::fprintf(stderr,
                                 "[R%d] ParallelAssembler apply zero-row row=%lld op=%d\n",
                                 my_rank_,
                                 static_cast<long long>(entry.global_row),
                                 static_cast<int>(entry.op));
                }
                const GlobalIndex row = entry.global_row;
                matrix_view->zeroRows(std::span<const GlobalIndex>(&row, 1u),
                                      entry.op == GhostMatrixOp::ZeroRowSetDiagonal);
            };

            for (const auto& entry : pending_received_matrix_) {
                if (entry.op == GhostMatrixOp::Entry && entry.mode != AddMode::Insert) {
                    apply_entry(entry);
                }
            }
            for (const auto& entry : pending_received_matrix_) {
                if (entry.op == GhostMatrixOp::ZeroRow ||
                    entry.op == GhostMatrixOp::ZeroRowSetDiagonal) {
                    apply_zero_row(entry);
                }
            }
            for (const auto& entry : pending_received_matrix_) {
                if (entry.op == GhostMatrixOp::Entry && entry.mode == AddMode::Insert) {
                    apply_entry(entry);
                }
            }
        }
        if (vector_view) {
            auto apply_vector_entry = [&](const GhostVectorContribution& entry) {
                vector_view->addVectorEntry(entry.global_row, entry.value, entry.mode);
            };
            auto apply_vector_zero = [&](const GhostVectorContribution& entry) {
                const GlobalIndex row = entry.global_row;
                vector_view->zeroVectorEntries(std::span<const GlobalIndex>(&row, 1u));
            };

            for (const auto& entry : pending_received_vector_) {
                if (entry.op == GhostVectorOp::Entry && entry.mode != AddMode::Insert) {
                    apply_vector_entry(entry);
                }
            }
            for (const auto& entry : pending_received_vector_) {
                if (entry.op == GhostVectorOp::Zero) {
                    apply_vector_zero(entry);
                }
            }
            for (const auto& entry : pending_received_vector_) {
                if (entry.op == GhostVectorOp::Entry && entry.mode == AddMode::Insert) {
                    apply_vector_entry(entry);
                }
            }
        }

        pending_received_matrix_.clear();
        pending_received_vector_.clear();
    }

    // End assembly phases
    if (matrix_view) {
        matrix_view->endAssemblyPhase();
        matrix_view->finalizeAssembly();
    }

    if (vector_view && vector_view != matrix_view) {
        vector_view->endAssemblyPhase();
        vector_view->finalizeAssembly();
    }

    // Reset ghost assembly state for the next operator.
    ghost_manager_.clearSendBuffers();
    ghost_manager_.clearReceivedContributions();
    assembly_in_progress_ = false;
}

void ParallelAssembler::reset()
{
    local_assembler_.reset();
    ghost_manager_.clearSendBuffers();
    ghost_manager_.clearReceivedContributions();
    row_dofs_.clear();
    col_dofs_.clear();
    owned_contributions_.clear();
    pending_received_matrix_.clear();
    pending_received_vector_.clear();
    initialized_ = false;
    assembly_in_progress_ = false;
}

// ============================================================================
// Matrix Assembly
// ============================================================================

AssemblyResult ParallelAssembler::assembleMatrix(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view)
{
    return assembleCellsParallel(mesh, test_space, trial_space, kernel,
                                 &matrix_view, nullptr, true, false);
}

// ============================================================================
// Vector Assembly
// ============================================================================

AssemblyResult ParallelAssembler::assembleVector(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& space,
    AssemblyKernel& kernel,
    GlobalSystemView& vector_view)
{
    return assembleCellsParallel(mesh, space, space, kernel,
                                 nullptr, &vector_view, false, true);
}

// ============================================================================
// Combined Assembly
// ============================================================================

AssemblyResult ParallelAssembler::assembleBoth(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view,
    GlobalSystemView& vector_view)
{
    return assembleCellsParallel(mesh, test_space, trial_space, kernel,
                                 &matrix_view, &vector_view, true, true);
}

// ============================================================================
// Fused Multi-Term Cell Assembly
// ============================================================================

AssemblyResult ParallelAssembler::assembleCellsFused(
    const IMeshAccess& mesh,
    std::span<const FusedCellTerm> terms)
{
    if (!initialized_) {
        initialize();
    }

    if (options_.allow_unowned_row_accumulation) {
        // Internal temporary views, such as FE-system sensitivity accumulators,
        // intentionally collect all owned-cell row contributions locally before
        // a caller-managed reduction.
        OwnedCellsMeshAccess owned_mesh(mesh);
        return local_assembler_.assembleCellsFused(owned_mesh, terms);
    }

    beginGhostAssemblyIfNeeded();

    // Wrap each term's matrix/vector views with GhostRoutingView
    std::vector<std::unique_ptr<GhostRoutingView>> routed_views;
    routed_views.reserve(terms.size() * 2);

    std::vector<FusedCellTerm> routed_terms(terms.begin(), terms.end());
    for (auto& t : routed_terms) {
        if (t.matrix_view && t.assemble_matrix) {
            routed_views.push_back(std::make_unique<GhostRoutingView>(
                *t.matrix_view, ghost_manager_, ghost_policy_));
            t.matrix_view = routed_views.back().get();
        }
        if (t.vector_view && t.assemble_vector) {
            // Check if vector_view was the same pointer as matrix_view (before routing)
            // In that case, reuse the same routed view
            if (t.vector_view == terms[static_cast<std::size_t>(&t - routed_terms.data())].matrix_view &&
                t.matrix_view != nullptr) {
                t.vector_view = t.matrix_view;
            } else {
                routed_views.push_back(std::make_unique<GhostRoutingView>(
                    *t.vector_view, ghost_manager_, ghost_policy_));
                t.vector_view = routed_views.back().get();
            }
        }
    }

    // OwnedRowsOnly assembles all locally visible cells and filters rows.
    // ReverseScatter assembles each owned cell once and routes off-owner rows.
    return withPolicyMeshAccess(mesh, ghost_policy_, [&](const IMeshAccess& policy_mesh) {
        return local_assembler_.assembleCellsFused(policy_mesh, routed_terms);
    });
}

// ============================================================================
// Face Assembly
// ============================================================================

AssemblyResult ParallelAssembler::assembleBoundaryFaces(
    const IMeshAccess& mesh,
    int boundary_marker,
    const spaces::FunctionSpace& space,
    AssemblyKernel& kernel,
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view)
{
    if (!initialized_) {
        initialize();
    }

    if (options_.allow_unowned_row_accumulation) {
        OwnedCellsMeshAccess owned_mesh(mesh);
        return local_assembler_.assembleBoundaryFaces(owned_mesh, boundary_marker,
                                                      space, kernel, matrix_view,
                                                      vector_view);
    }

    beginGhostAssemblyIfNeeded();

    return withPolicyMeshAccess(mesh, ghost_policy_, [&](const IMeshAccess& policy_mesh) -> AssemblyResult {
        if (matrix_view && vector_view) {
            if (matrix_view == vector_view) {
                GhostRoutingView routed(*matrix_view, ghost_manager_, ghost_policy_);
                return local_assembler_.assembleBoundaryFaces(policy_mesh, boundary_marker, space, kernel, &routed,
                                                             &routed);
            }
            GhostRoutingView routed_matrix(*matrix_view, ghost_manager_, ghost_policy_);
            GhostRoutingView routed_vector(*vector_view, ghost_manager_, ghost_policy_);
            return local_assembler_.assembleBoundaryFaces(policy_mesh, boundary_marker, space, kernel, &routed_matrix,
                                                         &routed_vector);
        }

        if (matrix_view) {
            GhostRoutingView routed_matrix(*matrix_view, ghost_manager_, ghost_policy_);
            return local_assembler_.assembleBoundaryFaces(policy_mesh, boundary_marker, space, kernel, &routed_matrix,
                                                         nullptr);
        }

        if (vector_view) {
            GhostRoutingView routed_vector(*vector_view, ghost_manager_, ghost_policy_);
            return local_assembler_.assembleBoundaryFaces(policy_mesh, boundary_marker, space, kernel, nullptr,
                                                         &routed_vector);
        }

        return {};
    });
}

AssemblyResult ParallelAssembler::assembleBoundaryFaces(
    const IMeshAccess& mesh,
    int boundary_marker,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view)
{
    if (!initialized_) {
        initialize();
    }

    if (options_.allow_unowned_row_accumulation) {
        OwnedCellsMeshAccess owned_mesh(mesh);
        return local_assembler_.assembleBoundaryFaces(owned_mesh, boundary_marker,
                                                      test_space, trial_space, kernel,
                                                      matrix_view, vector_view);
    }

    beginGhostAssemblyIfNeeded();

    return withPolicyMeshAccess(mesh, ghost_policy_, [&](const IMeshAccess& policy_mesh) -> AssemblyResult {
        if (matrix_view && vector_view) {
            if (matrix_view == vector_view) {
                GhostRoutingView routed(*matrix_view, ghost_manager_, ghost_policy_);
                return local_assembler_.assembleBoundaryFaces(policy_mesh, boundary_marker, test_space, trial_space,
                                                             kernel, &routed, &routed);
            }
            GhostRoutingView routed_matrix(*matrix_view, ghost_manager_, ghost_policy_);
            GhostRoutingView routed_vector(*vector_view, ghost_manager_, ghost_policy_);
            return local_assembler_.assembleBoundaryFaces(policy_mesh, boundary_marker, test_space, trial_space,
                                                         kernel, &routed_matrix, &routed_vector);
        }

        if (matrix_view) {
            GhostRoutingView routed_matrix(*matrix_view, ghost_manager_, ghost_policy_);
            return local_assembler_.assembleBoundaryFaces(policy_mesh, boundary_marker, test_space, trial_space, kernel,
                                                         &routed_matrix, nullptr);
        }

        if (vector_view) {
            GhostRoutingView routed_vector(*vector_view, ghost_manager_, ghost_policy_);
            return local_assembler_.assembleBoundaryFaces(policy_mesh, boundary_marker, test_space, trial_space, kernel,
                                                         nullptr, &routed_vector);
        }

        return {};
    });
}

AssemblyResult ParallelAssembler::assembleInteriorFaces(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view,
    GlobalSystemView* vector_view)
{
    if (!initialized_) {
        initialize();
    }

    if (options_.allow_unowned_row_accumulation) {
        OwnedCellsMeshAccess owned_mesh(mesh);
        return local_assembler_.assembleInteriorFaces(owned_mesh, test_space,
                                                      trial_space, kernel,
                                                      matrix_view, vector_view);
    }

    beginGhostAssemblyIfNeeded();

    return withPolicyMeshAccess(mesh, ghost_policy_, [&](const IMeshAccess& policy_mesh) -> AssemblyResult {
        if (vector_view != nullptr) {
            if (&matrix_view == vector_view) {
                GhostRoutingView routed(matrix_view, ghost_manager_, ghost_policy_);
                return local_assembler_.assembleInteriorFaces(policy_mesh, test_space, trial_space, kernel, routed,
                                                             &routed);
            }
            GhostRoutingView routed_matrix(matrix_view, ghost_manager_, ghost_policy_);
            GhostRoutingView routed_vector(*vector_view, ghost_manager_, ghost_policy_);
            return local_assembler_.assembleInteriorFaces(policy_mesh, test_space, trial_space, kernel, routed_matrix,
                                                         &routed_vector);
        }

        GhostRoutingView routed_matrix(matrix_view, ghost_manager_, ghost_policy_);
        return local_assembler_.assembleInteriorFaces(policy_mesh, test_space, trial_space, kernel, routed_matrix,
                                                     nullptr);
    });
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
AssemblyResult ParallelAssembler::assembleInterfaceFaces(
    const IMeshAccess& mesh,
    const svmp::InterfaceMesh& interface_mesh,
    int interface_marker,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view,
    GlobalSystemView* vector_view)
{
    if (!initialized_) {
        initialize();
    }

    if (options_.allow_unowned_row_accumulation) {
        OwnedCellsMeshAccess owned_mesh(mesh);
        return local_assembler_.assembleInterfaceFaces(owned_mesh, interface_mesh,
                                                       interface_marker, test_space,
                                                       trial_space, kernel, matrix_view,
                                                       vector_view);
    }

    beginGhostAssemblyIfNeeded();

    return withPolicyMeshAccess(mesh, ghost_policy_, [&](const IMeshAccess& policy_mesh) -> AssemblyResult {
        if (vector_view != nullptr) {
            if (&matrix_view == vector_view) {
                GhostRoutingView routed(matrix_view, ghost_manager_, ghost_policy_);
                return local_assembler_.assembleInterfaceFaces(policy_mesh, interface_mesh, interface_marker,
                                                              test_space, trial_space, kernel, routed, &routed);
            }
            GhostRoutingView routed_matrix(matrix_view, ghost_manager_, ghost_policy_);
            GhostRoutingView routed_vector(*vector_view, ghost_manager_, ghost_policy_);
            return local_assembler_.assembleInterfaceFaces(policy_mesh, interface_mesh, interface_marker, test_space,
                                                          trial_space, kernel, routed_matrix, &routed_vector);
        }

        GhostRoutingView routed_matrix(matrix_view, ghost_manager_, ghost_policy_);
        return local_assembler_.assembleInterfaceFaces(policy_mesh, interface_mesh, interface_marker, test_space,
                                                      trial_space, kernel, routed_matrix, nullptr);
    });
}
#endif

// ============================================================================
// Internal Implementation
// ============================================================================

void ParallelAssembler::beginGhostAssemblyIfNeeded()
{
    if (assembly_in_progress_) {
        return;
    }

    // Start a new assembly phase: clear any buffered ghost data from the previous operator.
    ghost_manager_.clearSendBuffers();
    ghost_manager_.clearReceivedContributions();
    pending_received_matrix_.clear();
    pending_received_vector_.clear();

    // Ensure ghost communication patterns are initialized before any buffering occurs.
    if (ghost_policy_ == GhostPolicy::ReverseScatter && !ghost_manager_.isInitialized()) {
        ghost_manager_.setPolicy(ghost_policy_);
        ghost_manager_.setDeterministic(options_.deterministic);
        ghost_manager_.setExplicitRowOwner(options_.row_owner_rank);
        ghost_manager_.initialize();
    }

    assembly_in_progress_ = true;
}

AssemblyResult ParallelAssembler::assembleCellsParallel(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view,
    bool assemble_matrix,
    bool assemble_vector)
{
    if (!initialized_) {
        initialize();
    }

    if (options_.allow_unowned_row_accumulation) {
        OwnedCellsMeshAccess owned_mesh(mesh);
        if (assemble_matrix && matrix_view && assemble_vector && vector_view) {
            if (matrix_view == vector_view) {
                return local_assembler_.assembleBoth(owned_mesh, test_space, trial_space,
                                                     kernel, *matrix_view, *matrix_view);
            }
            return local_assembler_.assembleBoth(owned_mesh, test_space, trial_space,
                                                 kernel, *matrix_view, *vector_view);
        }
        if (assemble_matrix && matrix_view) {
            return local_assembler_.assembleMatrix(owned_mesh, test_space, trial_space,
                                                   kernel, *matrix_view);
        }
        if (assemble_vector && vector_view) {
            return local_assembler_.assembleVector(owned_mesh, test_space, kernel,
                                                   *vector_view);
        }
        return {};
    }

    beginGhostAssemblyIfNeeded();

    auto assemble_with_routing = [&](const IMeshAccess& mesh_view) -> AssemblyResult {
        return withPolicyMeshAccess(mesh_view, ghost_policy_, [&](const IMeshAccess& policy_mesh) -> AssemblyResult {
            if (assemble_matrix && matrix_view && assemble_vector && vector_view) {
                if (matrix_view == vector_view) {
                    GhostRoutingView routed(*matrix_view, ghost_manager_, ghost_policy_);
                    return local_assembler_.assembleBoth(policy_mesh, test_space, trial_space, kernel, routed, routed);
                }
                GhostRoutingView routed_matrix(*matrix_view, ghost_manager_, ghost_policy_);
                GhostRoutingView routed_vector(*vector_view, ghost_manager_, ghost_policy_);
                return local_assembler_.assembleBoth(policy_mesh, test_space, trial_space, kernel, routed_matrix,
                                                     routed_vector);
            }

            if (assemble_matrix && matrix_view) {
                GhostRoutingView routed_matrix(*matrix_view, ghost_manager_, ghost_policy_);
                return local_assembler_.assembleMatrix(policy_mesh, test_space, trial_space, kernel, routed_matrix);
            }

            if (assemble_vector && vector_view) {
                GhostRoutingView routed_vector(*vector_view, ghost_manager_, ghost_policy_);
                return local_assembler_.assembleVector(policy_mesh, test_space, kernel, routed_vector);
            }

            return {};
        });
    };

#if FE_HAS_MPI
    const bool can_overlap =
        overlap_comm_ && (ghost_policy_ == GhostPolicy::ReverseScatter) && (world_size_ > 1);
#else
    const bool can_overlap = false;
#endif

    if (!can_overlap) {
        return assemble_with_routing(mesh);
    }

    FE_CHECK_NOT_NULL(dof_map_, "ParallelAssembler::assembleCellsParallel: dof_map");

    std::vector<GlobalIndex> boundary_cells;
    std::vector<GlobalIndex> interior_cells;
    const auto owned_count = mesh.numOwnedCells();
    if (owned_count > 0) {
        const auto n = static_cast<std::size_t>(owned_count);
        boundary_cells.reserve(n);
        interior_cells.reserve(n);
    }

    mesh.forEachOwnedCell([&](GlobalIndex cell_id) {
        const auto dofs = dof_map_->getCellDofs(cell_id);
        bool has_ghost_rows = false;
        for (const auto dof : dofs) {
            if (!dof_map_->isOwnedDof(dof)) {
                has_ghost_rows = true;
                break;
            }
        }
        (has_ghost_rows ? boundary_cells : interior_cells).push_back(cell_id);
    });

    const CellListMeshAccess boundary_mesh(mesh, std::span<const GlobalIndex>(boundary_cells));
    const CellListMeshAccess interior_mesh(mesh, std::span<const GlobalIndex>(interior_cells));

    AssemblyResult boundary_result = assemble_with_routing(boundary_mesh);
    ghost_manager_.startExchange();
    AssemblyResult interior_result = assemble_with_routing(interior_mesh);
    ghost_manager_.waitExchange();

    auto recv_matrix = ghost_manager_.takeReceivedMatrixContributions();
    if (!recv_matrix.empty()) {
        if (pending_received_matrix_.empty()) {
            pending_received_matrix_ = std::move(recv_matrix);
        } else {
            pending_received_matrix_.insert(pending_received_matrix_.end(),
                                            std::make_move_iterator(recv_matrix.begin()),
                                            std::make_move_iterator(recv_matrix.end()));
        }
    }

    auto recv_vector = ghost_manager_.takeReceivedVectorContributions();
    if (!recv_vector.empty()) {
        if (pending_received_vector_.empty()) {
            pending_received_vector_ = std::move(recv_vector);
        } else {
            pending_received_vector_.insert(pending_received_vector_.end(),
                                            std::make_move_iterator(recv_vector.begin()),
                                            std::make_move_iterator(recv_vector.end()));
        }
    }

    if (boundary_result.success && !interior_result.success) {
        boundary_result.success = false;
        boundary_result.error_message = interior_result.error_message;
    }
    boundary_result.elements_assembled += interior_result.elements_assembled;
    boundary_result.boundary_faces_assembled += interior_result.boundary_faces_assembled;
    boundary_result.interior_faces_assembled += interior_result.interior_faces_assembled;
    boundary_result.interface_faces_assembled += interior_result.interface_faces_assembled;
    boundary_result.elapsed_time_seconds += interior_result.elapsed_time_seconds;
    boundary_result.matrix_entries_inserted += interior_result.matrix_entries_inserted;
    boundary_result.vector_entries_inserted += interior_result.vector_entries_inserted;

    return boundary_result;
}

void ParallelAssembler::insertLocalWithGhostHandling(
    const KernelOutput& output,
    std::span<const GlobalIndex> row_dofs,
    std::span<const GlobalIndex> col_dofs,
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view)
{
    if (ghost_policy_ == GhostPolicy::OwnedRowsOnly) {
        // Only insert to owned rows
        const auto n_rows = static_cast<GlobalIndex>(row_dofs.size());
        const auto n_cols = static_cast<GlobalIndex>(col_dofs.size());

        for (GlobalIndex i = 0; i < n_rows; ++i) {
            GlobalIndex row = row_dofs[static_cast<std::size_t>(i)];

            if (!ghost_manager_.isOwned(row)) continue;

            // Matrix row
            if (matrix_view && output.has_matrix) {
                for (GlobalIndex j = 0; j < n_cols; ++j) {
                    GlobalIndex col = col_dofs[static_cast<std::size_t>(j)];
                    Real val = output.matrixEntry(i, j);
                    matrix_view->addMatrixEntry(row, col, val);
                }
            }

            // Vector entry
            if (vector_view && output.has_vector) {
                Real val = output.vectorEntry(i);
                vector_view->addVectorEntry(row, val);
            }
        }
    } else {
        // ReverseScatter: insert all, buffering ghosts
        const auto n_rows = static_cast<GlobalIndex>(row_dofs.size());
        const auto n_cols = static_cast<GlobalIndex>(col_dofs.size());

        for (GlobalIndex i = 0; i < n_rows; ++i) {
            GlobalIndex row = row_dofs[static_cast<std::size_t>(i)];
            bool is_owned = ghost_manager_.isOwned(row);

            // Matrix entries
            if (output.has_matrix) {
                for (GlobalIndex j = 0; j < n_cols; ++j) {
                    GlobalIndex col = col_dofs[static_cast<std::size_t>(j)];
                    Real val = output.matrixEntry(i, j);

                    if (is_owned) {
                        if (matrix_view) {
                            matrix_view->addMatrixEntry(row, col, val);
                        }
                    } else {
                        ghost_manager_.addMatrixContribution(row, col, val);
                    }
                }
            }

            // Vector entry
            if (output.has_vector) {
                Real val = output.vectorEntry(i);

                if (is_owned) {
                    if (vector_view) {
                        vector_view->addVectorEntry(row, val);
                    }
                } else {
                    ghost_manager_.addVectorContribution(row, val);
                }
            }
        }
    }
}

void ParallelAssembler::exchangeGhostContributions()
{
    if (overlap_comm_) {
        if (!ghost_manager_.isExchangeInProgress()) {
            ghost_manager_.startExchange();
        }
        ghost_manager_.waitExchange();
        return;
    }
    ghost_manager_.exchangeContributions();
}

void ParallelAssembler::applyReceivedContributions(
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view)
{
    // Apply received matrix contributions
    if (matrix_view) {
        auto received = ghost_manager_.getReceivedMatrixContributions();
        for (const auto& entry : received) {
            if (entry.op == GhostMatrixOp::Entry) {
                matrix_view->addMatrixEntry(entry.global_row, entry.global_col, entry.value, entry.mode);
            } else {
                const GlobalIndex row = entry.global_row;
                matrix_view->zeroRows(std::span<const GlobalIndex>(&row, 1u),
                                      entry.op == GhostMatrixOp::ZeroRowSetDiagonal);
            }
        }
    }

    // Apply received vector contributions
    if (vector_view) {
        auto received = ghost_manager_.getReceivedVectorContributions();
        for (const auto& entry : received) {
            if (entry.op == GhostVectorOp::Entry) {
                vector_view->addVectorEntry(entry.global_row, entry.value, entry.mode);
            } else {
                const GlobalIndex row = entry.global_row;
                vector_view->zeroVectorEntries(std::span<const GlobalIndex>(&row, 1u));
            }
        }
    }

    ghost_manager_.clearReceivedContributions();
}

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<Assembler> createParallelAssembler()
{
    return std::make_unique<ParallelAssembler>();
}

#if FE_HAS_MPI
std::unique_ptr<Assembler> createParallelAssembler(MPI_Comm comm)
{
    return std::make_unique<ParallelAssembler>(comm);
}
#endif

std::unique_ptr<Assembler> createParallelAssembler(const AssemblyOptions& options)
{
    return std::make_unique<ParallelAssembler>(options);
}

} // namespace assembly
} // namespace FE
} // namespace svmp
