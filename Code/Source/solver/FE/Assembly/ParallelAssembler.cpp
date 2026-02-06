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
        ghost_manager_->addMatrixContributions(row_dofs, col_dofs, local_matrix, owned_contributions_);
        for (const auto& c : owned_contributions_) {
            base_->addMatrixEntry(c.global_row, c.global_col, c.value, mode);
        }
    }

    void addMatrixEntry(GlobalIndex row,
                        GlobalIndex col,
                        Real value,
                        AddMode mode = AddMode::Add) override
    {
        FE_CHECK_NOT_NULL(base_, "GhostRoutingView::base");

        if (ghost_manager_->isOwned(row)) {
            base_->addMatrixEntry(row, col, value, mode);
            return;
        }

        if (policy_ == GhostPolicy::OwnedRowsOnly) {
            return;
        }

        // Non-additive operations (e.g. Dirichlet row setDiagonal via constraints) must be
        // applied only by the owning rank to avoid over-counting under ReverseScatter.
        if (mode == AddMode::Insert) {
            return;
        }

        ghost_manager_->addMatrixContribution(row, col, value);
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
        if (!ghost_manager_->isOwned(dof)) {
            return;
        }
        base_->setDiagonal(dof, value);
    }

    void zeroRows(std::span<const GlobalIndex> rows,
                  bool set_diagonal = true) override
    {
        FE_CHECK_NOT_NULL(base_, "GhostRoutingView::base");
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
            base_->addVectorEntry(dof, value, mode);
            return;
        }

        if (policy_ == GhostPolicy::OwnedRowsOnly) {
            return;
        }

        // Insert semantics (used by constraints to set Dirichlet values) must not be
        // reverse-scattered additively from non-owning ranks.
        if (mode == AddMode::Insert) {
            return;
        }

        ghost_manager_->addVectorContribution(dof, value);
    }

    void setVectorEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> values) override
    {
        FE_CHECK_NOT_NULL(base_, "GhostRoutingView::base");
        FE_THROW_IF(dofs.size() != values.size(), FEException,
                    "GhostRoutingView::setVectorEntries: size mismatch");
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            if (!ghost_manager_->isOwned(dofs[i])) {
                continue;
            }
            base_->setVectorEntries(std::span<const GlobalIndex>(&dofs[i], 1u),
                                    std::span<const Real>(&values[i], 1u));
        }
    }

    void zeroVectorEntries(std::span<const GlobalIndex> dofs) override
    {
        FE_CHECK_NOT_NULL(base_, "GhostRoutingView::base");
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

void ParallelAssembler::setRowDofMap(const dofs::DofMap& dof_map, GlobalIndex row_offset)
{
    local_assembler_.setRowDofMap(dof_map, row_offset);

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

void ParallelAssembler::setColDofMap(const dofs::DofMap& dof_map, GlobalIndex col_offset)
{
    local_assembler_.setColDofMap(dof_map, col_offset);
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

        if (options_.deterministic) {
            std::sort(pending_received_matrix_.begin(), pending_received_matrix_.end());
            std::sort(pending_received_vector_.begin(), pending_received_vector_.end());
        }

        if (matrix_view) {
            for (const auto& entry : pending_received_matrix_) {
                matrix_view->addMatrixEntry(entry.global_row, entry.global_col, entry.value);
            }
        }
        if (vector_view) {
            for (const auto& entry : pending_received_vector_) {
                vector_view->addVectorEntry(entry.global_row, entry.value);
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
    beginGhostAssemblyIfNeeded();

    if (matrix_view && vector_view) {
        if (matrix_view == vector_view) {
            GhostRoutingView routed(*matrix_view, ghost_manager_, ghost_policy_);
            return local_assembler_.assembleBoundaryFaces(mesh, boundary_marker, space, kernel, &routed, &routed);
        }
        GhostRoutingView routed_matrix(*matrix_view, ghost_manager_, ghost_policy_);
        GhostRoutingView routed_vector(*vector_view, ghost_manager_, ghost_policy_);
        return local_assembler_.assembleBoundaryFaces(mesh, boundary_marker, space, kernel,
                                                     &routed_matrix, &routed_vector);
    }

    if (matrix_view) {
        GhostRoutingView routed_matrix(*matrix_view, ghost_manager_, ghost_policy_);
        return local_assembler_.assembleBoundaryFaces(mesh, boundary_marker, space, kernel, &routed_matrix, nullptr);
    }

    if (vector_view) {
        GhostRoutingView routed_vector(*vector_view, ghost_manager_, ghost_policy_);
        return local_assembler_.assembleBoundaryFaces(mesh, boundary_marker, space, kernel, nullptr, &routed_vector);
    }

    return {};
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
    beginGhostAssemblyIfNeeded();

    if (matrix_view && vector_view) {
        if (matrix_view == vector_view) {
            GhostRoutingView routed(*matrix_view, ghost_manager_, ghost_policy_);
            return local_assembler_.assembleBoundaryFaces(mesh, boundary_marker, test_space, trial_space, kernel,
                                                         &routed, &routed);
        }
        GhostRoutingView routed_matrix(*matrix_view, ghost_manager_, ghost_policy_);
        GhostRoutingView routed_vector(*vector_view, ghost_manager_, ghost_policy_);
        return local_assembler_.assembleBoundaryFaces(mesh, boundary_marker, test_space, trial_space, kernel,
                                                     &routed_matrix, &routed_vector);
    }

    if (matrix_view) {
        GhostRoutingView routed_matrix(*matrix_view, ghost_manager_, ghost_policy_);
        return local_assembler_.assembleBoundaryFaces(mesh, boundary_marker, test_space, trial_space, kernel,
                                                     &routed_matrix, nullptr);
    }

    if (vector_view) {
        GhostRoutingView routed_vector(*vector_view, ghost_manager_, ghost_policy_);
        return local_assembler_.assembleBoundaryFaces(mesh, boundary_marker, test_space, trial_space, kernel,
                                                     nullptr, &routed_vector);
    }

    return {};
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
    beginGhostAssemblyIfNeeded();

    if (vector_view != nullptr) {
        if (&matrix_view == vector_view) {
            GhostRoutingView routed(matrix_view, ghost_manager_, ghost_policy_);
            return local_assembler_.assembleInteriorFaces(mesh, test_space, trial_space, kernel, routed, &routed);
        }
        GhostRoutingView routed_matrix(matrix_view, ghost_manager_, ghost_policy_);
        GhostRoutingView routed_vector(*vector_view, ghost_manager_, ghost_policy_);
        return local_assembler_.assembleInteriorFaces(mesh, test_space, trial_space, kernel,
                                                     routed_matrix, &routed_vector);
    }

    GhostRoutingView routed_matrix(matrix_view, ghost_manager_, ghost_policy_);
    return local_assembler_.assembleInteriorFaces(mesh, test_space, trial_space, kernel, routed_matrix, nullptr);
}

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

    beginGhostAssemblyIfNeeded();

    auto assemble_with_routing = [&](const IMeshAccess& mesh_view) -> AssemblyResult {
        if (assemble_matrix && matrix_view && assemble_vector && vector_view) {
            if (matrix_view == vector_view) {
                GhostRoutingView routed(*matrix_view, ghost_manager_, ghost_policy_);
                return local_assembler_.assembleBoth(mesh_view, test_space, trial_space, kernel, routed, routed);
            }
            GhostRoutingView routed_matrix(*matrix_view, ghost_manager_, ghost_policy_);
            GhostRoutingView routed_vector(*vector_view, ghost_manager_, ghost_policy_);
            return local_assembler_.assembleBoth(mesh_view, test_space, trial_space, kernel,
                                                 routed_matrix, routed_vector);
        }

        if (assemble_matrix && matrix_view) {
            GhostRoutingView routed_matrix(*matrix_view, ghost_manager_, ghost_policy_);
            return local_assembler_.assembleMatrix(mesh_view, test_space, trial_space, kernel, routed_matrix);
        }

        if (assemble_vector && vector_view) {
            GhostRoutingView routed_vector(*vector_view, ghost_manager_, ghost_policy_);
            return local_assembler_.assembleVector(mesh_view, test_space, kernel, routed_vector);
        }

        return {};
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
            matrix_view->addMatrixEntry(entry.global_row, entry.global_col, entry.value);
        }
    }

    // Apply received vector contributions
    if (vector_view) {
        auto received = ghost_manager_.getReceivedVectorContributions();
        for (const auto& entry : received) {
            vector_view->addVectorEntry(entry.global_row, entry.value);
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
