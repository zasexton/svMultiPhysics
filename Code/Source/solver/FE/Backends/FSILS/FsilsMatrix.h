/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_FSILS_MATRIX_H
#define SVMP_FE_BACKENDS_FSILS_MATRIX_H

#include "Backends/Interfaces/GenericMatrix.h"
#include "Backends/Interfaces/DofPermutation.h"
#include "Backends/FSILS/FsilsShared.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#if defined(FE_HAS_MPI) && FE_HAS_MPI
#include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace sparsity {
class SparsityPattern;
class DistributedSparsityPattern;
} // namespace sparsity

namespace backends {

class FsilsMatrix final : public GenericMatrix {
public:
    explicit FsilsMatrix(const sparsity::SparsityPattern& sparsity);
    FsilsMatrix(const sparsity::SparsityPattern& sparsity,
                int dof_per_node,
                std::shared_ptr<const DofPermutation> dof_permutation = {}
#if defined(FE_HAS_MPI) && FE_HAS_MPI
                ,
                MPI_Comm comm = MPI_COMM_WORLD
#endif
    );
    FsilsMatrix(const sparsity::DistributedSparsityPattern& sparsity,
                int dof_per_node,
                std::shared_ptr<const DofPermutation> dof_permutation = {}
#if defined(FE_HAS_MPI) && FE_HAS_MPI
                ,
                MPI_Comm comm = MPI_COMM_WORLD
#endif
    );
    ~FsilsMatrix() override;

    FsilsMatrix(FsilsMatrix&&) noexcept;
    FsilsMatrix& operator=(FsilsMatrix&&) noexcept;

    FsilsMatrix(const FsilsMatrix&) = delete;
    FsilsMatrix& operator=(const FsilsMatrix&) = delete;

    [[nodiscard]] BackendKind backendKind() const noexcept override { return BackendKind::FSILS; }
    [[nodiscard]] GlobalIndex numRows() const noexcept override;
    [[nodiscard]] GlobalIndex numCols() const noexcept override;

    void zero() override;
    void finalizeAssembly() override;
    bool reinitFromPattern(const sparsity::SparsityPattern& pattern) override;
    bool reinitFromPattern(
        const sparsity::DistributedSparsityPattern& pattern) override;
    void mult(const GenericVector& x, GenericVector& y) const override;
    void multAdd(const GenericVector& x, GenericVector& y) const override;

    [[nodiscard]] std::unique_ptr<assembly::GlobalSystemView> createAssemblyView() override;
    [[nodiscard]] Real getEntry(GlobalIndex row, GlobalIndex col) const override;

    void addValue(GlobalIndex row, GlobalIndex col, Real value, assembly::AddMode mode);

    /// Layout of the vectors that pair with this matrix (FsilsFactory creates
    /// them from it).  Owned nodes plus the ghost nodes of the sparsity
    /// pattern's ghost rows.
    [[nodiscard]] std::shared_ptr<const FsilsShared> shared() const noexcept { return vector_shared_; }
    /// Layout of the stored operator (lhs, CSR, value slots).  Equal to
    /// shared() unless an owned row has columns outside that node set (for
    /// example constraint-elimination fill beyond the ghost layers): then it
    /// appends those column nodes as extra ghost nodes, after the vector
    /// layout's nodes in the old local ordering, so an array in the vector
    /// layout is a prefix of one in the operator layout.  Everything that reads
    /// the CSR or value slots must use this layout.
    [[nodiscard]] std::shared_ptr<const FsilsShared> operatorShared() const noexcept { return shared_; }
    [[nodiscard]] bool hasSeparateOperatorLayout() const noexcept { return shared_ != vector_shared_; }
    /// Copy a vector-layout vector into an operator-layout vector (extra
    /// column nodes zero), and back (extra nodes dropped).
    void copyToOperatorLayout(const GenericVector& vector_layout, GenericVector& operator_layout) const;
    void copyFromOperatorLayout(const GenericVector& operator_layout, GenericVector& vector_layout) const;
    [[nodiscard]] bool usesOwnedRowOperator() const noexcept;
    [[nodiscard]] bool ownsFeDofRow(GlobalIndex fe_dof) const noexcept;

    // Resolve local row/column FE DOFs to FSILS value-storage slots and reuse
    // that mapping across repeated assemblies with identical connectivity.
    void resolveMatrixEntrySlotsCached(std::span<const GlobalIndex> row_dofs,
                                       std::span<const GlobalIndex> col_dofs,
                                       std::span<GlobalIndex> resolved) const;
    void addResolvedMatrixEntries(std::span<const GlobalIndex> row_dofs,
                                  std::span<const GlobalIndex> col_dofs,
                                  std::span<const GlobalIndex> resolved,
                                  std::span<const Real> local_matrix,
                                  assembly::AddMode mode);
    void addMatrixEntriesCached(std::span<const GlobalIndex> row_dofs,
                                std::span<const GlobalIndex> col_dofs,
                                std::span<const Real> local_matrix,
                                assembly::AddMode mode);

    // Internal access for solver integration
    [[nodiscard]] int fsilsDof() const noexcept;
    [[nodiscard]] void* fsilsLhsPtr() noexcept;
    [[nodiscard]] const void* fsilsLhsPtr() const noexcept;
    [[nodiscard]] Real* fsilsValuesPtr() noexcept;
    [[nodiscard]] const Real* fsilsValuesPtr() const noexcept;
    [[nodiscard]] GlobalIndex fsilsNnz() const noexcept;
    [[nodiscard]] std::uint64_t layoutRevision() const noexcept
    {
        return layout_revision_;
    }

    /// Number of entries silently dropped by addValue() since last reset.
    [[nodiscard]] static std::uint64_t droppedEntryCount() noexcept;
    /// Reset the dropped-entry counter to zero.
    static void resetDroppedEntryCount() noexcept;
    /// Number of attempted matrix writes to locally present but non-owned rows since last reset.
    [[nodiscard]] static std::uint64_t offOwnerWriteCount() noexcept;
    /// Reset the off-owner write counter to zero.
    static void resetOffOwnerWriteCount() noexcept;

    /// Insert a dof*dof block for a single (row_internal, col_internal) node pair.
    /// Does ONE CSR column lookup instead of dof*dof individual lookups.
    void addBlock(int row_internal, int col_internal, const Real* block_data,
                  int dof, assembly::AddMode mode);

private:
    bool adoptCompatibleReinitialization(FsilsMatrix&& replacement);

    static std::atomic<std::uint64_t> dropped_entry_count_;
    static std::atomic<std::uint64_t> off_owner_write_count_;
    GlobalIndex global_rows_{0};
    GlobalIndex global_cols_{0};
    GlobalIndex nnz_{0};
    // Process-unique (nextFsilsLayoutStamp) so that a new matrix whose layout
    // reuses a destroyed layout's address never repeats a cached revision.
    std::uint64_t layout_revision_{nextFsilsLayoutStamp()};

#if defined(FE_HAS_MPI) && FE_HAS_MPI
    MPI_Comm comm_{MPI_COMM_WORLD};
#endif
    std::shared_ptr<FsilsShared> shared_{};        ///< operator layout (see operatorShared())
    std::shared_ptr<FsilsShared> vector_shared_{}; ///< vector layout (see shared()); == shared_ normally
    std::vector<Real> values_{}; // (dof*dof) x nnz (column-major for FSILS Array wrapper)
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_FSILS_MATRIX_H
