/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_BLOCK_MATRIX_H
#define SVMP_FE_BACKENDS_BLOCK_MATRIX_H

#include "Backends/Interfaces/BlockVector.h"
#include "Backends/Interfaces/GenericMatrix.h"
#include "Core/FEException.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace backends {

/**
 * @brief A sparse matrix composed of sub-matrices (block operator).
 *
 * Blocks are stored as an m-by-n grid. Each block (i,j) maps from column block j
 * into row block i.
 */
class BlockMatrix final : public GenericMatrix {
public:
    BlockMatrix(std::vector<GlobalIndex> row_block_sizes,
                std::vector<GlobalIndex> col_block_sizes,
                std::vector<std::vector<std::unique_ptr<GenericMatrix>>> blocks);

    [[nodiscard]] BackendKind backendKind() const noexcept override { return backend_kind_; }
    [[nodiscard]] GlobalIndex numRows() const noexcept override { return n_rows_; }
    [[nodiscard]] GlobalIndex numCols() const noexcept override { return n_cols_; }

    void zero() override;
    void finalizeAssembly() override;

    void mult(const GenericVector& x, GenericVector& y) const override;
    void multAdd(const GenericVector& x, GenericVector& y) const override;

    [[nodiscard]] std::unique_ptr<assembly::GlobalSystemView> createAssemblyView() override;

    [[nodiscard]] Real getEntry(GlobalIndex row, GlobalIndex col) const override;

    [[nodiscard]] std::size_t numRowBlocks() const noexcept { return row_block_sizes_.size(); }
    [[nodiscard]] std::size_t numColBlocks() const noexcept { return col_block_sizes_.size(); }

    [[nodiscard]] GlobalIndex rowBlockOffset(std::size_t i) const;
    [[nodiscard]] GlobalIndex colBlockOffset(std::size_t j) const;
    [[nodiscard]] GlobalIndex rowBlockSize(std::size_t i) const;
    [[nodiscard]] GlobalIndex colBlockSize(std::size_t j) const;

    [[nodiscard]] GenericMatrix* block(std::size_t i, std::size_t j) noexcept;
    [[nodiscard]] const GenericMatrix* block(std::size_t i, std::size_t j) const noexcept;

    [[nodiscard]] std::pair<std::size_t, GlobalIndex> locateRow(GlobalIndex row) const;
    [[nodiscard]] std::pair<std::size_t, GlobalIndex> locateCol(GlobalIndex col) const;

private:
    BackendKind backend_kind_{BackendKind::Eigen};
    GlobalIndex n_rows_{0};
    GlobalIndex n_cols_{0};

    std::vector<GlobalIndex> row_block_sizes_{};
    std::vector<GlobalIndex> col_block_sizes_{};
    std::vector<GlobalIndex> row_offsets_{};
    std::vector<GlobalIndex> col_offsets_{};
    std::vector<std::vector<std::unique_ptr<GenericMatrix>>> blocks_{};
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_BLOCK_MATRIX_H

