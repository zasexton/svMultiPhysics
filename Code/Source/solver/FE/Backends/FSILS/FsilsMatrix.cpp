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
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace backends {

std::atomic<std::uint64_t> FsilsMatrix::dropped_entry_count_{0};

std::uint64_t FsilsMatrix::droppedEntryCount() noexcept
{
    return dropped_entry_count_.load(std::memory_order_relaxed);
}

void FsilsMatrix::resetDroppedEntryCount() noexcept
{
    dropped_entry_count_.store(0, std::memory_order_relaxed);
}

namespace {

[[nodiscard]] fe_fsi_linear_solver::FSILS_commuType make_fsils_commu(MPI_Comm comm)
{
    fe_fsi_linear_solver::FSILS_commuType commu{};
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
        fe_fsi_linear_solver::fsils_commu_create(commu, comm);
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

[[nodiscard]] bool ghost_rows_look_nodal_interleaved(std::span<const GlobalIndex> ghost_rows, int dof)
{
    if (dof <= 0) {
        return false;
    }
    if (ghost_rows.empty()) {
        return false;
    }

    const std::size_t per_node = static_cast<std::size_t>(dof);
    if (ghost_rows.size() % per_node != 0u) {
        return false;
    }

    for (std::size_t i = 0; i < ghost_rows.size(); i += per_node) {
        const GlobalIndex base = ghost_rows[i];
        if (base < 0 || (base % dof) != 0) {
            return false;
        }
        for (int c = 1; c < dof; ++c) {
            const std::size_t idx = i + static_cast<std::size_t>(c);
            if (idx >= ghost_rows.size() || ghost_rows[idx] != base + c) {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] std::shared_ptr<const DofPermutation> normalize_dof_permutation(std::shared_ptr<const DofPermutation> perm,
                                                                              GlobalIndex global_size,
                                                                              std::string_view context,
                                                                              bool allow_partial)
{
    if (!perm || perm->empty()) {
        return perm;
    }

    FE_THROW_IF(perm->forward.size() != static_cast<std::size_t>(global_size) ||
                    perm->inverse.size() != static_cast<std::size_t>(global_size),
                InvalidArgumentException,
                std::string(context) + ": dof permutation size mismatch with global system size");

    if (!allow_partial) {
        std::vector<GlobalIndex> inverse_from_forward(static_cast<std::size_t>(global_size), INVALID_GLOBAL_INDEX);
        for (GlobalIndex fe = 0; fe < global_size; ++fe) {
            const auto fe_idx = static_cast<std::size_t>(fe);
            const GlobalIndex be = perm->forward[fe_idx];
            FE_THROW_IF(be < 0 || be >= global_size,
                        InvalidArgumentException,
                        std::string(context) + ": dof permutation mapped FE DOF to out-of-range backend DOF");

            const auto be_idx = static_cast<std::size_t>(be);
            FE_THROW_IF(inverse_from_forward[be_idx] != INVALID_GLOBAL_INDEX,
                        InvalidArgumentException,
                        std::string(context) + ": dof permutation is not one-to-one");
            inverse_from_forward[be_idx] = fe;
        }

        for (GlobalIndex be = 0; be < global_size; ++be) {
            if (inverse_from_forward[static_cast<std::size_t>(be)] == INVALID_GLOBAL_INDEX) {
                FE_THROW(InvalidArgumentException,
                         std::string(context) + ": dof permutation is not onto");
            }
        }

        if (inverse_from_forward == perm->inverse) {
            return perm;
        }

        auto normalized = std::make_shared<DofPermutation>();
        normalized->forward = perm->forward;
        normalized->inverse = std::move(inverse_from_forward);
        return normalized;
    }

    // Partial permutations are allowed for distributed overlap runs: entries not present on this
    // rank may be left INVALID. Rebuild inverse from forward for mapped entries and normalize it.
    const auto& fwd = perm->forward;
    std::vector<GlobalIndex> inverse_from_forward(static_cast<std::size_t>(global_size), INVALID_GLOBAL_INDEX);
    for (GlobalIndex fe = 0; fe < global_size; ++fe) {
        const GlobalIndex be = fwd[static_cast<std::size_t>(fe)];
        if (be == INVALID_GLOBAL_INDEX) {
            continue;
        }
        FE_THROW_IF(be < 0 || be >= global_size,
                    InvalidArgumentException,
                    std::string(context) + ": dof permutation mapped FE DOF to out-of-range backend DOF");
        auto& slot = inverse_from_forward[static_cast<std::size_t>(be)];
        FE_THROW_IF(slot != INVALID_GLOBAL_INDEX && slot != fe,
                    InvalidArgumentException,
                    std::string(context) + ": dof permutation is not one-to-one on mapped entries");
        slot = fe;
    }

    if (inverse_from_forward == perm->inverse) {
        return perm;
    }

    auto normalized = std::make_shared<DofPermutation>();
    normalized->forward = perm->forward;
    normalized->inverse = std::move(inverse_from_forward);
    return normalized;
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
    auto* cols = lhs.colPtr.data();
    Real* vals = values.data();

    std::vector<std::pair<fe_fsi_linear_solver::fsils_int, int>> key_idx;
    std::vector<fe_fsi_linear_solver::fsils_int> cols_sorted;
    std::vector<Real> vals_sorted;

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

        key_idx.resize(static_cast<std::size_t>(len));
        for (int k = 0; k < len; ++k) {
            const int idx = start + k;
            key_idx[static_cast<std::size_t>(k)] = {cols[idx], idx};
        }
        std::sort(key_idx.begin(), key_idx.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        cols_sorted.resize(static_cast<std::size_t>(len));
        vals_sorted.resize(static_cast<std::size_t>(len) * block_size);
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
        const auto n_rows = static_cast<GlobalIndex>(row_dofs.size());
        const auto n_cols = static_cast<GlobalIndex>(col_dofs.size());

        if (local_matrix.size() != static_cast<std::size_t>(n_rows * n_cols)) {
            FE_THROW(InvalidArgumentException, "FsilsMatrixView::addMatrixEntries: local_matrix size mismatch");
        }

        const auto shared = matrix_->shared();
        if (!shared) {
            // Fallback: no shared metadata, use per-scalar insertion.
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
            return;
        }

        const int dof = shared->dof;
        const auto& lhs = shared->lhs;
        const auto perm = shared->dof_permutation;
        const bool have_perm = perm && !perm->empty();
        const GlobalIndex num_rows_global = matrix_->numRows();
        const GlobalIndex num_cols_global = matrix_->numCols();
        const int nNo = lhs.nNo;

        // --- Resolve all DOFs upfront: compute (old_node, component, internal_node) ---
        struct DofInfo {
            int old_node;
            int component;
            int internal_node;
        };

        thread_local std::vector<DofInfo> row_info;
        thread_local std::vector<DofInfo> col_info;
        row_info.resize(static_cast<std::size_t>(n_rows));
        col_info.resize(static_cast<std::size_t>(n_cols));

        auto resolve_dof = [&](GlobalIndex fe_dof) -> DofInfo {
            if (fe_dof < 0 || fe_dof >= num_rows_global) {
                return {-1, -1, -1};
            }
            GlobalIndex fs_dof = fe_dof;
            if (have_perm) {
                if (static_cast<std::size_t>(fe_dof) >= perm->forward.size()) {
                    return {-1, -1, -1};
                }
                fs_dof = perm->forward[static_cast<std::size_t>(fe_dof)];
            }
            if (fs_dof < 0 || fs_dof >= num_rows_global) {
                return {-1, -1, -1};
            }
            const int global_node = static_cast<int>(fs_dof / dof);
            const int comp = static_cast<int>(fs_dof % dof);
            const int old = shared->globalNodeToOld(global_node);
            if (old < 0 || old >= nNo) {
                return {-1, -1, -1};
            }
            return {old, comp, lhs.map(old)};
        };

        for (GlobalIndex i = 0; i < n_rows; ++i) {
            row_info[static_cast<std::size_t>(i)] = resolve_dof(row_dofs[static_cast<std::size_t>(i)]);
        }
        for (GlobalIndex j = 0; j < n_cols; ++j) {
            col_info[static_cast<std::size_t>(j)] = resolve_dof(col_dofs[static_cast<std::size_t>(j)]);
        }

        // --- Iterate by node pairs, building dof*dof sub-blocks ---
        thread_local std::vector<Real> block_buf;
        const std::size_t block_size = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
        block_buf.resize(block_size);

        for (GlobalIndex i0 = 0; i0 < n_rows; ) {
            const auto& ri0 = row_info[static_cast<std::size_t>(i0)];
            if (ri0.old_node < 0) { ++i0; continue; }

            // Find run of row DOFs sharing the same node.
            GlobalIndex i1 = i0 + 1;
            while (i1 < n_rows && row_info[static_cast<std::size_t>(i1)].old_node == ri0.old_node) {
                ++i1;
            }

            for (GlobalIndex j0 = 0; j0 < n_cols; ) {
                const auto& ci0 = col_info[static_cast<std::size_t>(j0)];
                if (ci0.old_node < 0) { ++j0; continue; }

                // Find run of col DOFs sharing the same node.
                GlobalIndex j1 = j0 + 1;
                while (j1 < n_cols && col_info[static_cast<std::size_t>(j1)].old_node == ci0.old_node) {
                    ++j1;
                }

                // Check if this is a complete dof*dof block (common case).
                const GlobalIndex row_run = i1 - i0;
                const GlobalIndex col_run = j1 - j0;

                if (row_run == dof && col_run == dof) {
                    // Build the full dof*dof block.
                    // Block layout: block_entry_index(dof, r, c) = r * dof + c
                    std::fill(block_buf.begin(), block_buf.begin() + static_cast<std::ptrdiff_t>(block_size), Real(0));
                    for (GlobalIndex di = i0; di < i1; ++di) {
                        const int r = row_info[static_cast<std::size_t>(di)].component;
                        for (GlobalIndex dj = j0; dj < j1; ++dj) {
                            const int c = col_info[static_cast<std::size_t>(dj)].component;
                            const auto local_idx = static_cast<std::size_t>(di * n_cols + dj);
                            block_buf[block_entry_index(dof, r, c)] = local_matrix[local_idx];
                        }
                    }
                    matrix_->addBlock(ri0.internal_node, ci0.internal_node,
                                      block_buf.data(), dof, mode);
                } else {
                    // Partial block: fall back to per-scalar insertion for this node pair.
                    for (GlobalIndex di = i0; di < i1; ++di) {
                        const auto& ri = row_info[static_cast<std::size_t>(di)];
                        for (GlobalIndex dj = j0; dj < j1; ++dj) {
                            const auto& ci = col_info[static_cast<std::size_t>(dj)];
                            const auto local_idx = static_cast<std::size_t>(di * n_cols + dj);
                            matrix_->addValue(row_dofs[static_cast<std::size_t>(di)],
                                              col_dofs[static_cast<std::size_t>(dj)],
                                              local_matrix[local_idx], mode);
                        }
                    }
                }

                j0 = j1;
            }
            i0 = i1;
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

        auto& lhs = *static_cast<fe_fsi_linear_solver::FSILS_lhsType*>(matrix_->fsilsLhsPtr());
        const int dof = matrix_->fsilsDof();
        const std::size_t block_size = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
	        Real* values = matrix_->fsilsValuesPtr();

		        for (const GlobalIndex row_dof : rows) {
		            if (row_dof < 0 || row_dof >= matrix_->numRows()) continue;

		            GlobalIndex row_fs = row_dof;
		            if (const auto perm = shared->dof_permutation; perm && !perm->empty()) {
		                if (static_cast<std::size_t>(row_dof) >= perm->forward.size()) {
		                    continue;
		                }
		                row_fs = perm->forward[static_cast<std::size_t>(row_dof)];
		                if (row_fs == INVALID_GLOBAL_INDEX) {
		                    // Partial permutations are permitted for distributed overlap runs; unmapped DOFs are not
		                    // locally present on this rank and must be skipped.
		                    continue;
		                }
		            }
		            if (row_fs < 0 || row_fs >= matrix_->numRows()) {
		                continue;
		            }

		            const int global_node = static_cast<int>(row_fs / dof);
		            const int row_comp = static_cast<int>(row_fs % dof);

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
    : FsilsMatrix(sparsity, /*dof_per_node=*/1, /*dof_permutation=*/{})
{
}

FsilsMatrix::FsilsMatrix(const sparsity::SparsityPattern& pattern,
                         int dof_per_node,
                         std::shared_ptr<const DofPermutation> dof_permutation)
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

    dof_permutation = normalize_dof_permutation(std::move(dof_permutation), global_rows_, "FsilsMatrix", /*allow_partial=*/false);
    const bool have_perm = dof_permutation && !dof_permutation->empty();

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
            const GlobalIndex row_fs = static_cast<GlobalIndex>(node) * dof + r;
            const GlobalIndex row_dof = have_perm
                                            ? dof_permutation->inverse[static_cast<std::size_t>(row_fs)]
                                            : row_fs;
            const auto start = row_ptr[static_cast<std::size_t>(row_dof)];
            const auto end = row_ptr[static_cast<std::size_t>(row_dof + 1)];
            for (GlobalIndex k = start; k < end; ++k) {
                const GlobalIndex col_dof = col_idx[static_cast<std::size_t>(k)];
                if (col_dof < 0 || col_dof >= global_cols_) continue;
                const GlobalIndex col_fs = have_perm
                                               ? dof_permutation->forward[static_cast<std::size_t>(col_dof)]
                                               : col_dof;
                cols.push_back(static_cast<int>(col_fs / dof));
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
    shared->dof_permutation = std::move(dof_permutation);

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
    fe_fsi_linear_solver::fsils_lhs_create(shared->lhs, commu, gnNo, nNo, nnz, gNodes, rowPtr, colPtr, /*nFaces=*/0);

    build_old_of_internal(*shared);
    shared->buildGlobalToOldTable();

    const std::size_t block_size = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
    values_.assign(static_cast<std::size_t>(nnz) * block_size, 0.0);
    sort_row_columns_and_values(*shared, values_);

    shared_ = std::move(shared);
}

FsilsMatrix::FsilsMatrix(const sparsity::DistributedSparsityPattern& pattern,
                         int dof_per_node,
                         std::shared_ptr<const DofPermutation> dof_permutation)
{
    FE_THROW_IF(!pattern.isFinalized(), InvalidArgumentException, "FsilsMatrix: distributed sparsity must be finalized");
    FE_THROW_IF(dof_per_node <= 0, InvalidArgumentException, "FsilsMatrix: dof_per_node must be > 0");
    FE_THROW_IF(!pattern.isSquare(), NotImplementedException, "FsilsMatrix: rectangular systems not supported");

    global_rows_ = pattern.globalRows();
    global_cols_ = pattern.globalCols();

    const int dof = dof_per_node;
    FE_THROW_IF(global_rows_ % dof != 0, InvalidArgumentException,
                "FsilsMatrix: global size must be divisible by dof_per_node");

    dof_permutation = normalize_dof_permutation(std::move(dof_permutation), global_rows_, "FsilsMatrix", /*allow_partial=*/true);
    const bool have_perm = dof_permutation && !dof_permutation->empty();

    const GlobalIndex gnNo_g = global_rows_ / dof;
    const int gnNo = as_int(gnNo_g, "global node count");

    const auto& owned_rows = pattern.ownedRows();
    const auto& owned_cols = pattern.ownedCols();
    FE_THROW_IF(owned_rows.size() % dof != 0,
                InvalidArgumentException,
                "FsilsMatrix: owned row count must be divisible by dof_per_node");
    FE_THROW_IF(owned_cols.first != owned_rows.first || owned_cols.last != owned_rows.last,
                InvalidArgumentException,
                "FsilsMatrix: FSILS backend requires identical row/column ownership ranges");

    const bool pattern_indices_are_backend =
        (pattern.dofIndexing() == sparsity::DistributedSparsityPattern::DofIndexing::NodalInterleaved);

    if (pattern_indices_are_backend && pattern.numGhostRows() > 0) {
        FE_THROW_IF(!ghost_rows_look_nodal_interleaved(pattern.getGhostRowMap(), dof),
                    InvalidArgumentException,
                    "FsilsMatrix: nodal-interleaved distributed sparsity must store ghost rows in node-block ordering");
    }

    auto map_pattern_to_backend = [&](GlobalIndex dof_pat) -> GlobalIndex {
        if (!have_perm || pattern_indices_are_backend) {
            return dof_pat;
        }
        if (dof_pat < 0 || dof_pat >= global_rows_) {
            return INVALID_GLOBAL_INDEX;
        }
        return dof_permutation->forward[static_cast<std::size_t>(dof_pat)];
    };

    auto map_backend_to_pattern = [&](GlobalIndex dof_fs) -> GlobalIndex {
        if (!have_perm || pattern_indices_are_backend) {
            return dof_fs;
        }
        if (dof_fs < 0 || dof_fs >= global_rows_) {
            return INVALID_GLOBAL_INDEX;
        }
        return dof_permutation->inverse[static_cast<std::size_t>(dof_fs)];
    };

    int owned_node_start = 0;
    int owned_node_count = 0;
    std::vector<int> owned_nodes;

    if (pattern_indices_are_backend) {
        FE_THROW_IF(owned_rows.first % dof != 0, InvalidArgumentException,
                    "FsilsMatrix: owned row range must align with dof_per_node blocks in nodal-interleaved indexing");
        owned_node_start = as_int(owned_rows.first / dof, "owned node start");
        owned_node_count = as_int(owned_rows.size() / dof, "owned node count");
    } else {
        std::unordered_map<int, int> count_per_node;
        count_per_node.reserve(static_cast<std::size_t>(owned_rows.size() / dof) + 1u);

        for (GlobalIndex row_pat = owned_rows.first; row_pat < owned_rows.last; ++row_pat) {
            const GlobalIndex row_fs = map_pattern_to_backend(row_pat);
            FE_THROW_IF(row_fs < 0 || row_fs >= global_rows_, InvalidArgumentException,
                        "FsilsMatrix: DOF permutation produced out-of-range backend row");
            const int node = static_cast<int>(row_fs / dof);
            FE_THROW_IF(node < 0 || node >= gnNo, InvalidArgumentException,
                        "FsilsMatrix: DOF permutation mapped owned row to out-of-range node");
            ++count_per_node[node];
        }

        owned_nodes.reserve(count_per_node.size());
        for (const auto& kv : count_per_node) {
            const int node = kv.first;
            const int count = kv.second;
            FE_THROW_IF(count != dof, InvalidArgumentException,
                        "FsilsMatrix: owned DOFs do not form complete node blocks after applying DOF permutation");
            owned_nodes.push_back(node);
        }

        std::sort(owned_nodes.begin(), owned_nodes.end());
        owned_node_count = static_cast<int>(owned_nodes.size());
        FE_THROW_IF(owned_node_count <= 0, InvalidArgumentException, "FsilsMatrix: no owned nodes");

        owned_node_start = owned_nodes.front();
        const bool contiguous = (owned_nodes.back() - owned_nodes.front() + 1) == owned_node_count;
        if (contiguous) {
            owned_nodes.clear();
        }
    }

    auto is_owned_node = [&](int node) -> bool {
        if (!owned_nodes.empty()) {
            return std::binary_search(owned_nodes.begin(), owned_nodes.end(), node);
        }
        return node >= owned_node_start && node < owned_node_start + owned_node_count;
    };

    // Derive ghost nodes from stored ghost rows (overlap model).
    std::vector<int> ghost_nodes;
    if (pattern.numGhostRows() > 0) {
        auto ghost_row_map = pattern.getGhostRowMap();

        std::unordered_map<int, int> count_per_node;
        count_per_node.reserve(static_cast<std::size_t>(ghost_row_map.size() / dof) + 1u);

        for (const GlobalIndex row_pat : ghost_row_map) {
            const GlobalIndex row_fs = map_pattern_to_backend(row_pat);
            FE_THROW_IF(row_fs < 0 || row_fs >= global_rows_, InvalidArgumentException,
                        "FsilsMatrix: invalid ghost row index after applying DOF permutation");
            const int node = static_cast<int>(row_fs / dof);
            FE_THROW_IF(node < 0 || node >= gnNo, InvalidArgumentException,
                        "FsilsMatrix: ghost row out of range");
            ++count_per_node[node];
        }

        ghost_nodes.reserve(count_per_node.size());
        for (const auto& kv : count_per_node) {
            const int node = kv.first;
            const int count = kv.second;
            FE_THROW_IF(is_owned_node(node), InvalidArgumentException,
                        "FsilsMatrix: ghost rows must not overlap owned rows");
            FE_THROW_IF(count != dof, InvalidArgumentException,
                        "FsilsMatrix: ghost rows must include all dof components for each ghost node");
            ghost_nodes.push_back(node);
        }

        std::sort(ghost_nodes.begin(), ghost_nodes.end());
        ghost_nodes.erase(std::unique(ghost_nodes.begin(), ghost_nodes.end()), ghost_nodes.end());
    }

    // For distributed overlap runs we allow partial DOF permutations, but the locally present
    // (owned + ghost) backend DOFs must still be mapped consistently.
    if (have_perm) {
        const auto& fwd = dof_permutation->forward;
        const auto& inv = dof_permutation->inverse;

        auto validate_backend_dof = [&](GlobalIndex dof_fs) {
            FE_THROW_IF(dof_fs < 0 || dof_fs >= global_rows_, InvalidArgumentException,
                        "FsilsMatrix: local backend DOF out of range");
            const GlobalIndex fe = inv[static_cast<std::size_t>(dof_fs)];
            FE_THROW_IF(fe == INVALID_GLOBAL_INDEX, InvalidArgumentException,
                        "FsilsMatrix: DOF permutation missing mapping for locally present backend DOF");
            FE_THROW_IF(fe < 0 || fe >= global_rows_, InvalidArgumentException,
                        "FsilsMatrix: DOF permutation inverse mapped to out-of-range FE DOF");
            const GlobalIndex dof_fs_back = fwd[static_cast<std::size_t>(fe)];
            FE_THROW_IF(dof_fs_back != dof_fs, InvalidArgumentException,
                        "FsilsMatrix: DOF permutation forward/inverse mismatch for locally present backend DOF");
        };

        auto validate_node = [&](int global_node) {
            FE_THROW_IF(global_node < 0 || global_node >= gnNo, InvalidArgumentException,
                        "FsilsMatrix: local node out of range");
            for (int r = 0; r < dof; ++r) {
                validate_backend_dof(static_cast<GlobalIndex>(global_node) * dof + r);
            }
        };

        if (!owned_nodes.empty()) {
            for (const int node : owned_nodes) {
                validate_node(node);
            }
        } else {
            for (int node = owned_node_start; node < owned_node_start + owned_node_count; ++node) {
                validate_node(node);
            }
        }
        for (const int node : ghost_nodes) {
            validate_node(node);
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
    shared->owned_nodes = std::move(owned_nodes);
    shared->ghost_nodes = ghost_nodes;
    shared->dof_permutation = dof_permutation;

    auto gather_row_nodes = [&](GlobalIndex row_fs, std::vector<int>& out_nodes) {
        out_nodes.clear();

        const GlobalIndex row_pat = map_backend_to_pattern(row_fs);
        if (row_pat < 0 || row_pat >= global_rows_) {
            return;
        }

        auto push_col_node = [&](GlobalIndex col_pat) {
            if (col_pat < 0 || col_pat >= global_cols_) {
                return;
            }
            const GlobalIndex col_fs = map_pattern_to_backend(col_pat);
            if (col_fs < 0 || col_fs >= global_cols_) {
                return;
            }
            out_nodes.push_back(static_cast<int>(col_fs / dof));
        };

        if (owned_rows.contains(row_pat)) {
            const GlobalIndex local_row = row_pat - owned_rows.first;
            const auto diag_cols = pattern.getRowDiagCols(local_row);
            const auto offdiag_cols = pattern.getRowOffdiagCols(local_row);
            out_nodes.reserve(static_cast<std::size_t>(diag_cols.size() + offdiag_cols.size()));

            for (const GlobalIndex local_col : diag_cols) {
                const GlobalIndex col_pat = local_col + owned_cols.first;
                push_col_node(col_pat);
            }
            for (const GlobalIndex ghost_idx : offdiag_cols) {
                const GlobalIndex col_pat = pattern.ghostColToGlobal(ghost_idx);
                push_col_node(col_pat);
            }
            return;
        }

        const GlobalIndex ghost_row = pattern.globalToGhostRow(row_pat);
        FE_THROW_IF(ghost_row < 0, InvalidArgumentException,
                    "FsilsMatrix: missing ghost row sparsity for row " + std::to_string(row_pat));
        const auto cols = pattern.getGhostRowCols(ghost_row);
        out_nodes.reserve(cols.size());
        for (const GlobalIndex col_pat : cols) {
            push_col_node(col_pat);
        }
    };

    std::vector<int> dof_row_nodes;
    std::vector<int> node_cols;

    for (int old = 0; old < nNo; ++old) {
        const int global_node = shared->oldToGlobalNode(old);
        FE_THROW_IF(global_node < 0, InvalidArgumentException,
                    "FsilsMatrix: invalid old->global node mapping");

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
        const int global_node = shared->oldToGlobalNode(old);
        FE_THROW_IF(global_node < 0, InvalidArgumentException,
                    "FsilsMatrix: invalid old->global node mapping");
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
    fe_fsi_linear_solver::fsils_lhs_create(shared->lhs, commu, gnNo, nNo, nnz, gNodes, rowPtr, colPtr, /*nFaces=*/0);

    build_old_of_internal(*shared);
    shared->buildGlobalToOldTable();

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
    resetDroppedEntryCount();
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

		    if (const auto perm = shared_->dof_permutation; perm && !perm->empty()) {
		        const auto& fwd = perm->forward;
		        if (static_cast<std::size_t>(row) >= fwd.size() || static_cast<std::size_t>(col) >= fwd.size()) {
		            return 0.0;
		        }
		        row = fwd[static_cast<std::size_t>(row)];
		        col = fwd[static_cast<std::size_t>(col)];
		    }
		    if (row < 0 || row >= global_rows_ || col < 0 || col >= global_cols_) {
		        return 0.0;
		    }

		    const int dof = shared_->dof;
		    const int global_row_node = static_cast<int>(row / dof);
		    const int global_col_node = static_cast<int>(col / dof);
		    const int row_comp = static_cast<int>(row % dof);
	    const int col_comp = static_cast<int>(col % dof);

	    const int row_old = shared_->globalNodeToOld(global_row_node);
	    const int col_old = shared_->globalNodeToOld(global_col_node);
	    const int nNo = shared_->lhs.nNo;
	    if (row_old < 0 || col_old < 0 || row_old >= nNo || col_old >= nNo) {
	        return 0.0;
	    }

	    const auto& lhs = shared_->lhs;
	    const int row_internal = lhs.map(row_old);
	    const int col_internal = lhs.map(col_old);
	    if (row_internal < 0 || col_internal < 0 || row_internal >= nNo || col_internal >= nNo) {
	        return 0.0;
	    }

	    const int start = lhs.rowPtr(0, row_internal);
	    const int end = lhs.rowPtr(1, row_internal);
	    if (start < 0 || end < start || end >= lhs.nnz) {
	        return 0.0;
	    }
	    const auto* cols = lhs.colPtr.data();
	    const auto* begin = cols + start;
	    const auto* finish = cols + end + 1;
	    const auto it = std::lower_bound(begin, finish, static_cast<fe_fsi_linear_solver::fsils_int>(col_internal));
	    if (it == finish || *it != col_internal) {
        return 0.0;
    }

	    const int nnz_idx = static_cast<int>(it - cols);
	    const std::size_t block_size = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
	    const std::size_t base = static_cast<std::size_t>(nnz_idx) * block_size;
	    const std::size_t off = block_entry_index(dof, row_comp, col_comp);
	    if (base + off >= values_.size()) {
	        return 0.0;
	    }
	    return values_[base + off];
}

	void FsilsMatrix::addValue(GlobalIndex row, GlobalIndex col, Real value, assembly::AddMode mode)
	{
	    if (row < 0 || row >= global_rows_ || col < 0 || col >= global_cols_) {
	        return;
	    }
	    FE_THROW_IF(!shared_, FEException, "FsilsMatrix::addValue: missing FSILS layout");

		    if (const auto perm = shared_->dof_permutation; perm && !perm->empty()) {
		        const auto& fwd = perm->forward;
		        if (static_cast<std::size_t>(row) >= fwd.size() || static_cast<std::size_t>(col) >= fwd.size()) {
		            return;
		        }
		        row = fwd[static_cast<std::size_t>(row)];
		        col = fwd[static_cast<std::size_t>(col)];
		    }
		    if (row < 0 || row >= global_rows_ || col < 0 || col >= global_cols_) {
		        return;
		    }

		    const int dof = shared_->dof;
		    const int global_row_node = static_cast<int>(row / dof);
		    const int global_col_node = static_cast<int>(col / dof);
		    const int row_comp = static_cast<int>(row % dof);
	    const int col_comp = static_cast<int>(col % dof);

	    const int row_old = shared_->globalNodeToOld(global_row_node);
	    const int col_old = shared_->globalNodeToOld(global_col_node);
	    const int nNo = shared_->lhs.nNo;
	    if (row_old < 0 || col_old < 0 || row_old >= nNo || col_old >= nNo) {
	        return;
	    }

	    const auto& lhs = shared_->lhs;
	    const int row_internal = lhs.map(row_old);
	    const int col_internal = lhs.map(col_old);
	    if (row_internal < 0 || col_internal < 0 || row_internal >= nNo || col_internal >= nNo) {
	        return;
	    }

	    const int start = lhs.rowPtr(0, row_internal);
	    const int end = lhs.rowPtr(1, row_internal);
	    if (start < 0 || end < start || end >= lhs.nnz) {
	        return;
	    }
	    auto* cols = lhs.colPtr.data();
	    auto* begin = cols + start;
	    auto* finish = cols + end + 1;
	    const auto it = std::lower_bound(begin, finish, static_cast<fe_fsi_linear_solver::fsils_int>(col_internal));
	    if (it == finish || *it != col_internal) {
        dropped_entry_count_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

	    const int nnz_idx = static_cast<int>(it - cols);
	    const std::size_t block_size = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
	    const std::size_t base = static_cast<std::size_t>(nnz_idx) * block_size;
	    const std::size_t off = block_entry_index(dof, row_comp, col_comp);
	    if (base + off >= values_.size()) {
	        return;
	    }
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

void FsilsMatrix::addBlock(int row_internal, int col_internal, const Real* block_data,
                           int dof, assembly::AddMode mode)
{
    const auto& lhs = shared_->lhs;
    const int nNo = lhs.nNo;
    if (row_internal < 0 || row_internal >= nNo || col_internal < 0 || col_internal >= nNo) {
        return;
    }

    const int start = lhs.rowPtr(0, row_internal);
    const int end = lhs.rowPtr(1, row_internal);
    if (start < 0 || end < start || end >= lhs.nnz) {
        return;
    }
    const auto* cols = lhs.colPtr.data();
    const auto* begin = cols + start;
    const auto* finish = cols + end + 1;
    const auto it = std::lower_bound(begin, finish, static_cast<fe_fsi_linear_solver::fsils_int>(col_internal));
    if (it == finish || *it != col_internal) {
        dropped_entry_count_.fetch_add(static_cast<std::uint64_t>(dof) * static_cast<std::uint64_t>(dof),
                                       std::memory_order_relaxed);
        return;
    }

    const int nnz_idx = static_cast<int>(it - cols);
    const std::size_t block_size = static_cast<std::size_t>(dof) * static_cast<std::size_t>(dof);
    const std::size_t base = static_cast<std::size_t>(nnz_idx) * block_size;
    if (base + block_size > values_.size()) {
        return;
    }
    Real* dst = values_.data() + base;

    switch (mode) {
        case assembly::AddMode::Add:
            for (std::size_t k = 0; k < block_size; ++k) {
                dst[k] += block_data[k];
            }
            break;
        case assembly::AddMode::Insert:
            for (std::size_t k = 0; k < block_size; ++k) {
                dst[k] = block_data[k];
            }
            break;
        case assembly::AddMode::Max:
            for (std::size_t k = 0; k < block_size; ++k) {
                dst[k] = std::max(dst[k], block_data[k]);
            }
            break;
        case assembly::AddMode::Min:
            for (std::size_t k = 0; k < block_size; ++k) {
                dst[k] = std::min(dst[k], block_data[k]);
            }
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
