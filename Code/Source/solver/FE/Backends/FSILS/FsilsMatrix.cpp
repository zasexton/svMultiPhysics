/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/FSILS/FsilsMatrix.h"

#include "Backends/FSILS/FsilsVector.h"
#include "Core/FEException.h"
#include "Sparsity/DistributedSparsityPattern.h"
#include "Sparsity/SparsityPattern.h"

#include "Backends/FSILS/liner_solver/commu.h"
#include "Backends/FSILS/liner_solver/lhs.h"
#include "Backends/FSILS/liner_solver/spar_mul.h"

#include "Array.h"
#include "Vector.h"

#include <algorithm>
#include <limits>
#include <mpi.h>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace backends {

namespace {

[[nodiscard]] fsi_linear_solver::FSILS_commuType make_fsils_commu(MPI_Comm comm)
{
    fsi_linear_solver::FSILS_commuType commu{};
    commu.foC = true;
    commu.comm = comm;
    commu.nTasks = 1;
    commu.task = 0;
    commu.master = 0;
    commu.masF = true;
    commu.tF = 0;

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        fsi_linear_solver::fsils_commu_create(commu, comm);
    }
    return commu;
}

[[nodiscard]] int as_int(GlobalIndex v, const char* what)
{
    FE_THROW_IF(v < 0, InvalidArgumentException, std::string("FSILS: negative ") + what);
    FE_THROW_IF(v > static_cast<GlobalIndex>(std::numeric_limits<int>::max()),
                InvalidArgumentException,
                std::string("FSILS: ") + what + " exceeds int range");
    return static_cast<int>(v);
}

[[nodiscard]] std::size_t block_entry_index(int dof, int row_comp, int col_comp)
{
    return static_cast<std::size_t>(row_comp * dof + col_comp);
}

void sort_row_columns_and_values(FsilsShared& shared, std::vector<Real>& values)
{
    const int dof = shared.dof;
    FE_THROW_IF(dof <= 0, FEException, "FsilsMatrix: invalid dof");

    auto& lhs = shared.lhs;
    const int nNo = lhs.nNo;
    const int nnz = lhs.nnz;
    FE_THROW_IF(nNo < 0 || nnz < 0, FEException, "FsilsMatrix: invalid lhs sizes");

    const std::size_t block_size = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
    FE_THROW_IF(values.size() != static_cast<std::size_t>(nnz) * block_size,
                FEException, "FsilsMatrix: values size mismatch");

    // Sort within each internal row for efficient binary search insertion.
    int* cols = lhs.colPtr.data();
    Real* vals = values.data();

    for (int row = 0; row < nNo; ++row) {
        const int start = lhs.rowPtr(0, row);
        const int end = lhs.rowPtr(1, row);
        FE_THROW_IF(start < 0 || end < start || end >= nnz, FEException,
                    "FsilsMatrix: invalid FSILS rowPtr range");

        const int len = end - start + 1;
        if (len <= 1) {
            lhs.diagPtr(row) = start;
            continue;
        }

        std::vector<std::pair<int, int>> key_idx(static_cast<std::size_t>(len));
        for (int k = 0; k < len; ++k) {
            const int idx = start + k;
            key_idx[static_cast<std::size_t>(k)] = {cols[idx], idx};
        }
        std::sort(key_idx.begin(), key_idx.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        std::vector<int> cols_sorted(static_cast<std::size_t>(len));
        std::vector<Real> vals_sorted(static_cast<std::size_t>(len) * block_size, 0.0);
        for (int k = 0; k < len; ++k) {
            const int old_idx = key_idx[static_cast<std::size_t>(k)].second;
            cols_sorted[static_cast<std::size_t>(k)] = cols[old_idx];
            const std::size_t src = static_cast<std::size_t>(old_idx) * block_size;
            const std::size_t dst = static_cast<std::size_t>(k) * block_size;
            std::copy(vals + src, vals + src + block_size, vals_sorted.data() + dst);
        }

        for (int k = 0; k < len; ++k) {
            const int idx = start + k;
            cols[idx] = cols_sorted[static_cast<std::size_t>(k)];
            const std::size_t dst = static_cast<std::size_t>(idx) * block_size;
            const std::size_t src = static_cast<std::size_t>(k) * block_size;
            std::copy(vals_sorted.data() + src, vals_sorted.data() + src + block_size, vals + dst);
        }

        // Recompute diagPtr after sorting.
        const auto* begin = cols + start;
        const auto* finish = cols + end + 1;
        const auto it = std::lower_bound(begin, finish, row);
        FE_THROW_IF(it == finish || *it != row, FEException,
                    "FsilsMatrix: missing diagonal block in FSILS pattern");
        lhs.diagPtr(row) = static_cast<int>(it - cols);
    }
}

void build_old_of_internal(FsilsShared& shared)
{
    auto& lhs = shared.lhs;
    const int nNo = lhs.nNo;
    shared.old_of_internal.assign(static_cast<std::size_t>(nNo), -1);
    for (int old = 0; old < nNo; ++old) {
        const int internal = lhs.map(old);
        FE_THROW_IF(internal < 0 || internal >= nNo, FEException, "FsilsMatrix: invalid lhs.map permutation");
        shared.old_of_internal[static_cast<std::size_t>(internal)] = old;
    }
    for (int internal = 0; internal < nNo; ++internal) {
        FE_THROW_IF(shared.old_of_internal[static_cast<std::size_t>(internal)] < 0,
                    FEException, "FsilsMatrix: invalid inverse permutation");
    }
}

class FsilsMatrixView final : public assembly::GlobalSystemView {
public:
    explicit FsilsMatrixView(FsilsMatrix& matrix) : matrix_(&matrix) {}

    void addMatrixEntries(std::span<const GlobalIndex> dofs,
                          std::span<const Real> local_matrix,
                          assembly::AddMode mode) override
    {
        addMatrixEntries(dofs, dofs, local_matrix, mode);
    }

    void addMatrixEntries(std::span<const GlobalIndex> row_dofs,
                          std::span<const GlobalIndex> col_dofs,
                          std::span<const Real> local_matrix,
                          assembly::AddMode mode) override
    {
        FE_CHECK_NOT_NULL(matrix_, "FsilsMatrixView::matrix");
        const GlobalIndex n_rows = static_cast<GlobalIndex>(row_dofs.size());
        const GlobalIndex n_cols = static_cast<GlobalIndex>(col_dofs.size());

        if (local_matrix.size() != static_cast<std::size_t>(n_rows * n_cols)) {
            FE_THROW(InvalidArgumentException, "FsilsMatrixView::addMatrixEntries: local_matrix size mismatch");
        }

        for (GlobalIndex i = 0; i < n_rows; ++i) {
            const GlobalIndex row = row_dofs[static_cast<std::size_t>(i)];
            if (row < 0 || row >= matrix_->numRows()) continue;

            for (GlobalIndex j = 0; j < n_cols; ++j) {
                const GlobalIndex col = col_dofs[static_cast<std::size_t>(j)];
                if (col < 0 || col >= matrix_->numCols()) continue;

                const auto local_idx = static_cast<std::size_t>(i * n_cols + j);
                matrix_->addValue(row, col, local_matrix[local_idx], mode);
            }
        }
    }

    void addMatrixEntry(GlobalIndex row, GlobalIndex col, Real value, assembly::AddMode mode) override
    {
        FE_CHECK_NOT_NULL(matrix_, "FsilsMatrixView::matrix");
        matrix_->addValue(row, col, value, mode);
    }

    void setDiagonal(std::span<const GlobalIndex> dofs,
                     std::span<const Real> values) override
    {
        if (dofs.size() != values.size()) {
            FE_THROW(InvalidArgumentException, "FsilsMatrixView::setDiagonal: size mismatch");
        }
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            setDiagonal(dofs[i], values[i]);
        }
    }

    void setDiagonal(GlobalIndex dof, Real value) override
    {
        addMatrixEntry(dof, dof, value, assembly::AddMode::Insert);
    }

    void zeroRows(std::span<const GlobalIndex> rows, bool set_diagonal) override
    {
        FE_CHECK_NOT_NULL(matrix_, "FsilsMatrixView::matrix");

        const auto shared = matrix_->shared();
        if (!shared) return;

        auto& lhs = *static_cast<fsi_linear_solver::FSILS_lhsType*>(matrix_->fsilsLhsPtr());
        const int dof = matrix_->fsilsDof();
        const std::size_t block_size = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
        Real* values = matrix_->fsilsValuesPtr();

        for (const GlobalIndex row_dof : rows) {
            if (row_dof < 0 || row_dof >= matrix_->numRows()) continue;

            const int global_node = static_cast<int>(row_dof / dof);
            const int row_comp = static_cast<int>(row_dof % dof);

            const int old = shared->globalNodeToOld(global_node);
            if (old < 0) continue;

            const int internal = lhs.map(old);
            const int start = lhs.rowPtr(0, internal);
            const int end = lhs.rowPtr(1, internal);

            for (int j = start; j <= end; ++j) {
                const std::size_t base = static_cast<std::size_t>(j) * block_size;
                for (int c = 0; c < dof; ++c) {
                    values[base + block_entry_index(dof, row_comp, c)] = 0.0;
                }
            }

            if (set_diagonal && row_dof < matrix_->numCols()) {
                matrix_->addValue(row_dof, row_dof, 1.0, assembly::AddMode::Insert);
            }
        }
    }

    // Vector operations (no-op)
    void addVectorEntries(std::span<const GlobalIndex>, std::span<const Real>, assembly::AddMode) override {}
    void addVectorEntry(GlobalIndex, Real, assembly::AddMode) override {}
    void setVectorEntries(std::span<const GlobalIndex>, std::span<const Real>) override {}
    void zeroVectorEntries(std::span<const GlobalIndex>) override {}

    void beginAssemblyPhase() override { phase_ = assembly::AssemblyPhase::Building; }
    void endAssemblyPhase() override { phase_ = assembly::AssemblyPhase::Flushing; }
    void finalizeAssembly() override { phase_ = assembly::AssemblyPhase::Finalized; }
    [[nodiscard]] assembly::AssemblyPhase getPhase() const noexcept override { return phase_; }

    [[nodiscard]] bool hasMatrix() const noexcept override { return true; }
    [[nodiscard]] bool hasVector() const noexcept override { return false; }
    [[nodiscard]] GlobalIndex numRows() const noexcept override { return matrix_ ? matrix_->numRows() : 0; }
    [[nodiscard]] GlobalIndex numCols() const noexcept override { return matrix_ ? matrix_->numCols() : 0; }
    [[nodiscard]] bool isDistributed() const noexcept override { return true; }
    [[nodiscard]] std::string backendName() const override { return "FSILSMatrix"; }

    void zero() override
    {
        FE_CHECK_NOT_NULL(matrix_, "FsilsMatrixView::matrix");
        matrix_->zero();
    }

    [[nodiscard]] Real getMatrixEntry(GlobalIndex row, GlobalIndex col) const override
    {
        FE_CHECK_NOT_NULL(matrix_, "FsilsMatrixView::matrix");
        return matrix_->getEntry(row, col);
    }

private:
    FsilsMatrix* matrix_{nullptr};
    assembly::AssemblyPhase phase_{assembly::AssemblyPhase::NotStarted};
};

} // namespace

FsilsMatrix::FsilsMatrix(const sparsity::SparsityPattern& sparsity)
    : FsilsMatrix(sparsity, /*dof_per_node=*/1)
{
}

FsilsMatrix::FsilsMatrix(const sparsity::SparsityPattern& pattern, int dof_per_node)
{
    FE_THROW_IF(!pattern.isFinalized(), InvalidArgumentException, "FsilsMatrix: sparsity pattern must be finalized");
    FE_THROW_IF(dof_per_node <= 0, InvalidArgumentException, "FsilsMatrix: dof_per_node must be > 0");

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        int comm_size = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
        FE_THROW_IF(comm_size != 1, NotImplementedException,
                    "FsilsMatrix: sequential SparsityPattern is only supported in serial; "
                    "use DistributedSparsityPattern for MPI runs");
    }

    global_rows_ = pattern.numRows();
    global_cols_ = pattern.numCols();
    FE_THROW_IF(global_rows_ != global_cols_, NotImplementedException,
                "FsilsMatrix: rectangular systems not supported");

    const int dof = dof_per_node;
    FE_THROW_IF(global_rows_ % dof != 0, InvalidArgumentException,
                "FsilsMatrix: global size must be divisible by dof_per_node");

    const GlobalIndex gnNo_g = global_rows_ / dof;
    const int gnNo = as_int(gnNo_g, "global node count");
    const int nNo = gnNo;

    const auto row_ptr = pattern.getRowPtr();
    const auto col_idx = pattern.getColIndices();

    std::vector<int> node_row_ptr(static_cast<std::size_t>(nNo + 1), 0);
    std::vector<int> node_col_ptr;
    node_col_ptr.reserve(static_cast<std::size_t>(row_ptr.back()));

    for (int node = 0; node < nNo; ++node) {
        std::vector<int> cols;
        cols.reserve(32);
        for (int r = 0; r < dof; ++r) {
            const GlobalIndex row_dof = static_cast<GlobalIndex>(node) * dof + r;
            const auto start = row_ptr[static_cast<std::size_t>(row_dof)];
            const auto end = row_ptr[static_cast<std::size_t>(row_dof + 1)];
            for (GlobalIndex k = start; k < end; ++k) {
                const GlobalIndex col_dof = col_idx[static_cast<std::size_t>(k)];
                if (col_dof < 0 || col_dof >= global_cols_) continue;
                cols.push_back(static_cast<int>(col_dof / dof));
            }
        }
        std::sort(cols.begin(), cols.end());
        cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
        if (cols.empty() || cols.front() != node) {
            cols.insert(std::lower_bound(cols.begin(), cols.end(), node), node);
        }

        node_row_ptr[static_cast<std::size_t>(node + 1)] =
            node_row_ptr[static_cast<std::size_t>(node)] + static_cast<int>(cols.size());
        node_col_ptr.insert(node_col_ptr.end(), cols.begin(), cols.end());
    }

    const int nnz = as_int(static_cast<GlobalIndex>(node_col_ptr.size()), "FSILS nnz blocks");
    nnz_ = nnz;

    auto shared = std::make_shared<FsilsShared>();
    shared->global_dofs = global_rows_;
    shared->dof = dof;
    shared->gnNo = gnNo;
    shared->owned_node_start = 0;
    shared->owned_node_count = nNo;

    Vector<int> gNodes(nNo);
    for (int i = 0; i < nNo; ++i) {
        gNodes(i) = i;
    }

    Vector<int> rowPtr(nNo + 1);
    for (int i = 0; i < nNo + 1; ++i) {
        rowPtr(i) = node_row_ptr[static_cast<std::size_t>(i)];
    }

    Vector<int> colPtr(nnz);
    for (int i = 0; i < nnz; ++i) {
        colPtr(i) = node_col_ptr[static_cast<std::size_t>(i)];
    }

    auto commu = make_fsils_commu(MPI_COMM_WORLD);
    fsi_linear_solver::fsils_lhs_create(shared->lhs, commu, gnNo, nNo, nnz, gNodes, rowPtr, colPtr, /*nFaces=*/0);

    build_old_of_internal(*shared);

    const std::size_t block_size = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
    values_.assign(static_cast<std::size_t>(nnz) * block_size, 0.0);
    sort_row_columns_and_values(*shared, values_);

    shared_ = std::move(shared);
}

FsilsMatrix::FsilsMatrix(const sparsity::DistributedSparsityPattern& pattern, int dof_per_node)
{
    FE_THROW_IF(!pattern.isFinalized(), InvalidArgumentException, "FsilsMatrix: distributed sparsity must be finalized");
    FE_THROW_IF(dof_per_node <= 0, InvalidArgumentException, "FsilsMatrix: dof_per_node must be > 0");
    FE_THROW_IF(!pattern.isSquare(), NotImplementedException, "FsilsMatrix: rectangular systems not supported");

    global_rows_ = pattern.globalRows();
    global_cols_ = pattern.globalCols();

    const int dof = dof_per_node;
    FE_THROW_IF(global_rows_ % dof != 0, InvalidArgumentException,
                "FsilsMatrix: global size must be divisible by dof_per_node");

    const GlobalIndex gnNo_g = global_rows_ / dof;
    const int gnNo = as_int(gnNo_g, "global node count");

    const auto& owned_rows = pattern.ownedRows();
    const auto& owned_cols = pattern.ownedCols();
    FE_THROW_IF(owned_rows.first % dof != 0 || owned_rows.size() % dof != 0,
                InvalidArgumentException,
                "FsilsMatrix: owned row range must align with dof_per_node blocks");
    FE_THROW_IF(owned_cols.first != owned_rows.first || owned_cols.last != owned_rows.last,
                InvalidArgumentException,
                "FsilsMatrix: FSILS backend requires identical row/column ownership ranges");

    const int owned_node_start = as_int(owned_rows.first / dof, "owned node start");
    const int owned_node_count = as_int(owned_rows.size() / dof, "owned node count");

    // Derive ghost nodes from stored ghost rows (overlap model).
    std::vector<int> ghost_nodes;
    if (pattern.numGhostRows() > 0) {
        auto ghost_row_map = pattern.getGhostRowMap();
        ghost_nodes.reserve(static_cast<std::size_t>(ghost_row_map.size()));

        std::vector<int> count_per_node;
        count_per_node.assign(static_cast<std::size_t>(gnNo), 0);

        for (const GlobalIndex row_dof : ghost_row_map) {
            FE_THROW_IF(row_dof % dof < 0, FEException, "FsilsMatrix: invalid ghost row index");
            const int node = static_cast<int>(row_dof / dof);
            FE_THROW_IF(node < 0 || node >= gnNo, InvalidArgumentException,
                        "FsilsMatrix: ghost row out of range");
            count_per_node[static_cast<std::size_t>(node)] += 1;
            ghost_nodes.push_back(node);
        }

        std::sort(ghost_nodes.begin(), ghost_nodes.end());
        ghost_nodes.erase(std::unique(ghost_nodes.begin(), ghost_nodes.end()), ghost_nodes.end());

        for (const int node : ghost_nodes) {
            FE_THROW_IF(node >= owned_node_start && node < owned_node_start + owned_node_count,
                        InvalidArgumentException,
                        "FsilsMatrix: ghost rows must not overlap owned rows");
            FE_THROW_IF(count_per_node[static_cast<std::size_t>(node)] != dof, InvalidArgumentException,
                        "FsilsMatrix: ghost rows must include all dof components for each ghost node");
        }
    }

    const int nNo = owned_node_count + static_cast<int>(ghost_nodes.size());
    const int nnz_max = std::numeric_limits<int>::max();

    // Build node-level CSR (old local ordering: owned nodes then ghosts).
    std::vector<int> node_row_ptr(static_cast<std::size_t>(nNo + 1), 0);
    std::vector<int> node_col_ptr;
    node_col_ptr.reserve(static_cast<std::size_t>(pattern.getLocalNnz()));

    auto shared = std::make_shared<FsilsShared>();
    shared->global_dofs = global_rows_;
    shared->dof = dof;
    shared->gnNo = gnNo;
    shared->owned_node_start = owned_node_start;
    shared->owned_node_count = owned_node_count;
    shared->ghost_nodes = ghost_nodes;

    auto gather_row_nodes = [&](GlobalIndex global_row_dof, std::vector<int>& out_nodes) {
        out_nodes.clear();
        if (owned_rows.contains(global_row_dof)) {
            const GlobalIndex local_row = global_row_dof - owned_rows.first;
            const auto diag_cols = pattern.getRowDiagCols(local_row);
            const auto offdiag_cols = pattern.getRowOffdiagCols(local_row);
            out_nodes.reserve(static_cast<std::size_t>(diag_cols.size() + offdiag_cols.size()));

            for (const GlobalIndex local_col : diag_cols) {
                const GlobalIndex global_col = local_col + owned_cols.first;
                out_nodes.push_back(static_cast<int>(global_col / dof));
            }
            for (const GlobalIndex ghost_idx : offdiag_cols) {
                const GlobalIndex global_col = pattern.ghostColToGlobal(ghost_idx);
                out_nodes.push_back(static_cast<int>(global_col / dof));
            }
            return;
        }

        const GlobalIndex ghost_row = pattern.globalToGhostRow(global_row_dof);
        FE_THROW_IF(ghost_row < 0, InvalidArgumentException,
                    "FsilsMatrix: missing ghost row sparsity for row " + std::to_string(global_row_dof));
        const auto cols = pattern.getGhostRowCols(ghost_row);
        out_nodes.reserve(cols.size());
        for (const GlobalIndex global_col : cols) {
            out_nodes.push_back(static_cast<int>(global_col / dof));
        }
    };

    std::vector<int> dof_row_nodes;
    std::vector<int> node_cols;

    for (int old = 0; old < nNo; ++old) {
        const int global_node = (old < owned_node_count)
                                    ? (owned_node_start + old)
                                    : ghost_nodes[static_cast<std::size_t>(old - owned_node_count)];

        node_cols.clear();
        for (int r = 0; r < dof; ++r) {
            const GlobalIndex row_dof = static_cast<GlobalIndex>(global_node) * dof + r;
            gather_row_nodes(row_dof, dof_row_nodes);
            node_cols.insert(node_cols.end(), dof_row_nodes.begin(), dof_row_nodes.end());
        }

        std::sort(node_cols.begin(), node_cols.end());
        node_cols.erase(std::unique(node_cols.begin(), node_cols.end()), node_cols.end());

        if (node_cols.empty() || node_cols.front() != global_node) {
            node_cols.insert(std::lower_bound(node_cols.begin(), node_cols.end(), global_node), global_node);
        }

        for (const int col_global_node : node_cols) {
            const int col_old = shared->globalNodeToOld(col_global_node);
            FE_THROW_IF(col_old < 0, InvalidArgumentException,
                        "FsilsMatrix: column node " + std::to_string(col_global_node) +
                            " is not present locally (ghost row closure required)");
            node_col_ptr.push_back(col_old);
        }

        FE_THROW_IF(node_col_ptr.size() > static_cast<std::size_t>(nnz_max), InvalidArgumentException,
                    "FsilsMatrix: local nnz exceeds FSILS int index range");
        node_row_ptr[static_cast<std::size_t>(old + 1)] = static_cast<int>(node_col_ptr.size());
    }

    const int nnz = static_cast<int>(node_col_ptr.size());
    nnz_ = nnz;

    Vector<int> gNodes(nNo);
    for (int old = 0; old < nNo; ++old) {
        const int global_node = (old < owned_node_count)
                                    ? (owned_node_start + old)
                                    : ghost_nodes[static_cast<std::size_t>(old - owned_node_count)];
        gNodes(old) = global_node;
    }

    Vector<int> rowPtr(nNo + 1);
    for (int i = 0; i < nNo + 1; ++i) {
        rowPtr(i) = node_row_ptr[static_cast<std::size_t>(i)];
    }

    Vector<int> colPtr(nnz);
    for (int i = 0; i < nnz; ++i) {
        colPtr(i) = node_col_ptr[static_cast<std::size_t>(i)];
    }

    auto commu = make_fsils_commu(MPI_COMM_WORLD);
    fsi_linear_solver::fsils_lhs_create(shared->lhs, commu, gnNo, nNo, nnz, gNodes, rowPtr, colPtr, /*nFaces=*/0);

    build_old_of_internal(*shared);

    const std::size_t block_size = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
    values_.assign(static_cast<std::size_t>(nnz) * block_size, 0.0);
    sort_row_columns_and_values(*shared, values_);

    shared_ = std::move(shared);
}

FsilsMatrix::~FsilsMatrix() = default;

FsilsMatrix::FsilsMatrix(FsilsMatrix&&) noexcept = default;
FsilsMatrix& FsilsMatrix::operator=(FsilsMatrix&&) noexcept = default;

GlobalIndex FsilsMatrix::numRows() const noexcept
{
    return global_rows_;
}

GlobalIndex FsilsMatrix::numCols() const noexcept
{
    return global_cols_;
}

void FsilsMatrix::zero()
{
    std::fill(values_.begin(), values_.end(), 0.0);
}

void FsilsMatrix::finalizeAssembly()
{
    // FSILS is assembled in-place; nothing to do.
}

void FsilsMatrix::mult(const GenericVector& x_in, GenericVector& y_in) const
{
    const auto* x = dynamic_cast<const FsilsVector*>(&x_in);
    auto* y = dynamic_cast<FsilsVector*>(&y_in);
    FE_THROW_IF(!x || !y, InvalidArgumentException, "FsilsMatrix::mult: backend mismatch");

    FE_THROW_IF(x->shared() != shared_.get() || y->shared() != shared_.get(),
                InvalidArgumentException, "FsilsMatrix::mult: vector layout mismatch");

    auto& lhs = shared_->lhs;
    const int dof = shared_->dof;
    const int nNo = lhs.nNo;
    const int nnz = lhs.nnz;
    const std::size_t block_size = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);

    FE_THROW_IF(static_cast<int>(x->data().size()) != dof * nNo ||
                    static_cast<int>(y->data().size()) != dof * nNo,
                InvalidArgumentException, "FsilsMatrix::mult: local size mismatch");
    FE_THROW_IF(values_.size() != static_cast<std::size_t>(nnz) * block_size, FEException,
                "FsilsMatrix::mult: invalid FSILS value storage");

    // Map input from old local ordering -> FSILS internal ordering.
    std::vector<double> u_internal(static_cast<std::size_t>(dof) * static_cast<std::size_t>(nNo), 0.0);
    const auto& x_old = x->data();
    for (int old = 0; old < nNo; ++old) {
        const int internal = lhs.map(old);
        for (int c = 0; c < dof; ++c) {
            u_internal[static_cast<std::size_t>(c) + static_cast<std::size_t>(internal) * static_cast<std::size_t>(dof)] =
                x_old[static_cast<std::size_t>(c) + static_cast<std::size_t>(old) * static_cast<std::size_t>(dof)];
        }
    }

    std::vector<double> ku_internal(static_cast<std::size_t>(dof) * static_cast<std::size_t>(nNo), 0.0);

    Array<double> K(dof * dof, nnz, const_cast<double*>(values_.data()));
    Array<double> U(dof, nNo, u_internal.data());
    Array<double> KU(dof, nNo, ku_internal.data());

    spar_mul::fsils_spar_mul_vv(lhs, lhs.rowPtr, lhs.colPtr, dof, K, U, KU);

    // Map output back to old local ordering.
    auto& y_old = y->data();
    for (int old = 0; old < nNo; ++old) {
        const int internal = lhs.map(old);
        for (int c = 0; c < dof; ++c) {
            y_old[static_cast<std::size_t>(c) + static_cast<std::size_t>(old) * static_cast<std::size_t>(dof)] =
                ku_internal[static_cast<std::size_t>(c) + static_cast<std::size_t>(internal) * static_cast<std::size_t>(dof)];
        }
    }
}

void FsilsMatrix::multAdd(const GenericVector& x_in, GenericVector& y_in) const
{
    auto* y = dynamic_cast<FsilsVector*>(&y_in);
    FE_THROW_IF(!y, InvalidArgumentException, "FsilsMatrix::multAdd: backend mismatch");

    FsilsVector tmp(shared());
    tmp.zero();
    mult(x_in, tmp);

    auto yspan = y->localSpan();
    auto tspan = tmp.localSpan();
    FE_THROW_IF(yspan.size() != tspan.size(), FEException, "FsilsMatrix::multAdd: size mismatch");

    for (std::size_t i = 0; i < yspan.size(); ++i) {
        yspan[i] += tspan[i];
    }
}

std::unique_ptr<assembly::GlobalSystemView> FsilsMatrix::createAssemblyView()
{
    return std::make_unique<FsilsMatrixView>(*this);
}

Real FsilsMatrix::getEntry(GlobalIndex row, GlobalIndex col) const
{
    if (row < 0 || row >= global_rows_ || col < 0 || col >= global_cols_) {
        return 0.0;
    }
    FE_THROW_IF(!shared_, FEException, "FsilsMatrix::getEntry: missing FSILS layout");

    const int dof = shared_->dof;
    const int global_row_node = static_cast<int>(row / dof);
    const int global_col_node = static_cast<int>(col / dof);
    const int row_comp = static_cast<int>(row % dof);
    const int col_comp = static_cast<int>(col % dof);

    const int row_old = shared_->globalNodeToOld(global_row_node);
    const int col_old = shared_->globalNodeToOld(global_col_node);
    if (row_old < 0 || col_old < 0) {
        return 0.0;
    }

    const auto& lhs = shared_->lhs;
    const int row_internal = lhs.map(row_old);
    const int col_internal = lhs.map(col_old);

    const int start = lhs.rowPtr(0, row_internal);
    const int end = lhs.rowPtr(1, row_internal);
    const int* cols = lhs.colPtr.data();
    const auto* begin = cols + start;
    const auto* finish = cols + end + 1;
    const auto it = std::lower_bound(begin, finish, col_internal);
    if (it == finish || *it != col_internal) {
        return 0.0;
    }

    const int nnz_idx = static_cast<int>(it - cols);
    const std::size_t block_size = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
    const std::size_t base = static_cast<std::size_t>(nnz_idx) * block_size;
    const std::size_t off = block_entry_index(dof, row_comp, col_comp);
    return values_[base + off];
}

void FsilsMatrix::addValue(GlobalIndex row, GlobalIndex col, Real value, assembly::AddMode mode)
{
    if (row < 0 || row >= global_rows_ || col < 0 || col >= global_cols_) {
        return;
    }
    FE_THROW_IF(!shared_, FEException, "FsilsMatrix::addValue: missing FSILS layout");

    const int dof = shared_->dof;
    const int global_row_node = static_cast<int>(row / dof);
    const int global_col_node = static_cast<int>(col / dof);
    const int row_comp = static_cast<int>(row % dof);
    const int col_comp = static_cast<int>(col % dof);

    const int row_old = shared_->globalNodeToOld(global_row_node);
    const int col_old = shared_->globalNodeToOld(global_col_node);
    if (row_old < 0 || col_old < 0) {
        return;
    }

    const auto& lhs = shared_->lhs;
    const int row_internal = lhs.map(row_old);
    const int col_internal = lhs.map(col_old);

    const int start = lhs.rowPtr(0, row_internal);
    const int end = lhs.rowPtr(1, row_internal);
    int* cols = lhs.colPtr.data();
    auto* begin = cols + start;
    auto* finish = cols + end + 1;
    const auto it = std::lower_bound(begin, finish, col_internal);
    if (it == finish || *it != col_internal) {
        return;
    }

    const int nnz_idx = static_cast<int>(it - cols);
    const std::size_t block_size = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
    const std::size_t base = static_cast<std::size_t>(nnz_idx) * block_size;
    const std::size_t off = block_entry_index(dof, row_comp, col_comp);
    Real& dst = values_[base + off];

    switch (mode) {
        case assembly::AddMode::Add:
            dst += value;
            break;
        case assembly::AddMode::Insert:
            dst = value;
            break;
        case assembly::AddMode::Max:
            dst = std::max(dst, value);
            break;
        case assembly::AddMode::Min:
            dst = std::min(dst, value);
            break;
    }
}

int FsilsMatrix::fsilsDof() const noexcept
{
    return shared_ ? shared_->dof : 0;
}

void* FsilsMatrix::fsilsLhsPtr() noexcept
{
    return shared_ ? static_cast<void*>(&shared_->lhs) : nullptr;
}

const void* FsilsMatrix::fsilsLhsPtr() const noexcept
{
    return shared_ ? static_cast<const void*>(&shared_->lhs) : nullptr;
}

Real* FsilsMatrix::fsilsValuesPtr() noexcept
{
    return values_.data();
}

const Real* FsilsMatrix::fsilsValuesPtr() const noexcept
{
    return values_.data();
}

GlobalIndex FsilsMatrix::fsilsNnz() const noexcept
{
    return nnz_;
}

} // namespace backends
} // namespace FE
} // namespace svmp

