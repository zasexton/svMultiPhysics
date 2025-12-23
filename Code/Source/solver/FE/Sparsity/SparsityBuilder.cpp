/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "SparsityBuilder.h"
#include "Dofs/BlockDofMap.h"
#include "Dofs/FieldDofMap.h"
#include <algorithm>
#include <unordered_set>
#include <cstdint>

namespace svmp {
namespace FE {
namespace sparsity {

namespace {

class BlockDofMapFieldQuery final : public IDofFieldQuery {
public:
    explicit BlockDofMapFieldQuery(const dofs::BlockDofMap& map) : map_(&map) {}

    [[nodiscard]] FieldId getFieldId(GlobalIndex dof) const override {
        if (!map_) {
            return INVALID_FIELD_ID;
        }
        auto block = map_->globalToBlock(dof);
        if (!block.has_value()) {
            return INVALID_FIELD_ID;
        }
        auto idx = block->first;
        if (idx > static_cast<std::size_t>(INVALID_FIELD_ID)) {
            return INVALID_FIELD_ID;
        }
        return static_cast<FieldId>(idx);
    }

    [[nodiscard]] std::size_t numFields() const override {
        return map_ ? map_->numBlocks() : 0u;
    }

private:
    const dofs::BlockDofMap* map_{nullptr};
};

class FieldDofMapFieldQuery final : public IDofFieldQuery {
public:
    explicit FieldDofMapFieldQuery(const dofs::FieldDofMap& map) : map_(&map) {}

    [[nodiscard]] FieldId getFieldId(GlobalIndex dof) const override {
        if (!map_) {
            return INVALID_FIELD_ID;
        }
        auto field = map_->globalToField(dof);
        if (!field.has_value()) {
            return INVALID_FIELD_ID;
        }
        if (field->first < 0) {
            return INVALID_FIELD_ID;
        }
        auto idx = static_cast<std::size_t>(field->first);
        if (idx > static_cast<std::size_t>(INVALID_FIELD_ID)) {
            return INVALID_FIELD_ID;
        }
        return static_cast<FieldId>(idx);
    }

    [[nodiscard]] std::size_t numFields() const override {
        return map_ ? map_->numFields() : 0u;
    }

private:
    const dofs::FieldDofMap* map_{nullptr};
};

inline std::uint32_t make_pair_key(FieldId row_field, FieldId col_field) {
    return (static_cast<std::uint32_t>(row_field) << 16) |
           static_cast<std::uint32_t>(col_field);
}

} // namespace

// ============================================================================
// SparsityBuilder - Construction
// ============================================================================

SparsityBuilder::SparsityBuilder(const dofs::DofMap& dof_map)
    : row_dof_map_(std::make_shared<DofMapAdapter>(dof_map)),
      col_dof_map_(nullptr)  // Will use row map for square pattern
{
}

SparsityBuilder::SparsityBuilder(std::shared_ptr<IDofMapQuery> dof_map_query)
    : row_dof_map_(std::move(dof_map_query)),
      col_dof_map_(nullptr)
{
}

// ============================================================================
// SparsityBuilder - Configuration
// ============================================================================

void SparsityBuilder::setRowDofMap(const dofs::DofMap& dof_map) {
    row_dof_map_ = std::make_shared<DofMapAdapter>(dof_map);
}

void SparsityBuilder::setRowDofMap(std::shared_ptr<IDofMapQuery> dof_map_query) {
    row_dof_map_ = std::move(dof_map_query);
}

void SparsityBuilder::setColDofMap(const dofs::DofMap& dof_map) {
    col_dof_map_ = std::make_shared<DofMapAdapter>(dof_map);
}

void SparsityBuilder::setColDofMap(std::shared_ptr<IDofMapQuery> dof_map_query) {
    col_dof_map_ = std::move(dof_map_query);
}

void SparsityBuilder::addCoupling(FieldId row_field, FieldId col_field, bool bidirectional) {
    coupling_mode_ = CouplingMode::Custom;
    field_couplings_.push_back({row_field, col_field, bidirectional});
}

void SparsityBuilder::addCoupling(const FieldCoupling& coupling) {
    coupling_mode_ = CouplingMode::Custom;
    field_couplings_.push_back(coupling);
}

void SparsityBuilder::setRowFieldMap(std::shared_ptr<IDofFieldQuery> field_query) {
    row_field_map_ = std::move(field_query);
}

void SparsityBuilder::setColFieldMap(std::shared_ptr<IDofFieldQuery> field_query) {
    col_field_map_ = std::move(field_query);
}

void SparsityBuilder::setRowFieldMap(const dofs::BlockDofMap& block_map) {
    row_field_map_ = std::make_shared<BlockDofMapFieldQuery>(block_map);
}

void SparsityBuilder::setColFieldMap(const dofs::BlockDofMap& block_map) {
    col_field_map_ = std::make_shared<BlockDofMapFieldQuery>(block_map);
}

void SparsityBuilder::setRowFieldMap(const dofs::FieldDofMap& field_map) {
    row_field_map_ = std::make_shared<FieldDofMapFieldQuery>(field_map);
}

void SparsityBuilder::setColFieldMap(const dofs::FieldDofMap& field_map) {
    col_field_map_ = std::make_shared<FieldDofMapFieldQuery>(field_map);
}

void SparsityBuilder::setCouplingsFromBlockDofMap(const dofs::BlockDofMap& block_map) {
    coupling_mode_ = CouplingMode::Custom;
    field_couplings_.clear();

    for (const auto& info : block_map.getAllCouplings()) {
        if (info.block_i > static_cast<std::size_t>(INVALID_FIELD_ID) ||
            info.block_j > static_cast<std::size_t>(INVALID_FIELD_ID)) {
            continue;
        }

        bool bidirectional = (info.coupling == dofs::BlockCoupling::TwoWay ||
                              info.coupling == dofs::BlockCoupling::Full);
        field_couplings_.push_back(
            {static_cast<FieldId>(info.block_i),
             static_cast<FieldId>(info.block_j),
             bidirectional});
    }
}

// ============================================================================
// SparsityBuilder - Building
// ============================================================================

void SparsityBuilder::validateConfiguration() const {
    FE_CHECK_ARG(row_dof_map_ != nullptr, "Row DOF map not set");

    if (coupling_mode_ != CouplingMode::Full) {
        FE_CHECK_ARG(row_field_map_ != nullptr,
                     "Field coupling requested but row field map not set");
    }

    if (coupling_mode_ == CouplingMode::Custom) {
        FE_CHECK_ARG(!field_couplings_.empty(),
                     "CouplingMode::Custom requires at least one field coupling");
    }
}

IDofMapQuery* SparsityBuilder::getEffectiveColDofMap() const {
    return col_dof_map_ ? col_dof_map_.get() : row_dof_map_.get();
}

IDofFieldQuery* SparsityBuilder::getEffectiveColFieldMap() const {
    return col_field_map_ ? col_field_map_.get() : row_field_map_.get();
}

SparsityPattern SparsityBuilder::build() {
    validateConfiguration();

    IDofMapQuery* col_map = getEffectiveColDofMap();
    IDofFieldQuery* col_field_map = getEffectiveColFieldMap();

    GlobalIndex n_rows = row_dof_map_->getNumDofs();
    GlobalIndex n_cols = col_map->getNumDofs();
    GlobalIndex n_cells = row_dof_map_->getNumCells();
    FE_CHECK_ARG(col_map->getNumCells() == n_cells,
                 "Row and column DOF maps must have the same number of cells");

    // Create pattern
    SparsityPattern pattern(n_rows, n_cols);

    // Build allowed field pairs once (Custom mode)
    std::unordered_set<std::uint32_t> allowed_pairs;
    if (coupling_mode_ == CouplingMode::Custom) {
        allowed_pairs.reserve(field_couplings_.size() * 2u);
        for (const auto& c : field_couplings_) {
            allowed_pairs.insert(make_pair_key(c.row_field, c.col_field));
            if (c.bidirectional) {
                allowed_pairs.insert(make_pair_key(c.col_field, c.row_field));
            }
        }
    }

    const bool enforce_symmetric =
        options_.symmetric_pattern && (n_rows == n_cols);

    // Iterate over all cells
    for (GlobalIndex cell = 0; cell < n_cells; ++cell) {
        auto row_dofs = row_dof_map_->getCellDofs(cell);
        auto col_dofs = col_map->getCellDofs(cell);

        if (coupling_mode_ == CouplingMode::Full || row_field_map_ == nullptr) {
            pattern.addElementCouplings(row_dofs, col_dofs);
            if (enforce_symmetric) {
                pattern.addElementCouplings(col_dofs, row_dofs);
            }
            continue;
        }

        // Precompute field IDs for DOFs (skip out-of-range/invalid indices)
        std::vector<FieldId> row_fields(row_dofs.size(), INVALID_FIELD_ID);
        std::vector<FieldId> col_fields(col_dofs.size(), INVALID_FIELD_ID);

        for (std::size_t i = 0; i < row_dofs.size(); ++i) {
            GlobalIndex dof = row_dofs[i];
            if (dof < 0 || dof >= n_rows) {
                continue;
            }
            FieldId f = row_field_map_->getFieldId(dof);
            FE_CHECK_ARG(f != INVALID_FIELD_ID,
                         "Row DOF " + std::to_string(dof) + " is not mapped to any field");
            row_fields[i] = f;
        }

        for (std::size_t j = 0; j < col_dofs.size(); ++j) {
            GlobalIndex dof = col_dofs[j];
            if (dof < 0 || dof >= n_cols) {
                continue;
            }
            FieldId f = col_field_map->getFieldId(dof);
            FE_CHECK_ARG(f != INVALID_FIELD_ID,
                         "Column DOF " + std::to_string(dof) + " is not mapped to any field");
            col_fields[j] = f;
        }

        auto is_allowed = [&](FieldId rf, FieldId cf) -> bool {
            if (rf == INVALID_FIELD_ID || cf == INVALID_FIELD_ID) {
                return false;
            }
            if (coupling_mode_ == CouplingMode::Diagonal) {
                return rf == cf;
            }
            if (coupling_mode_ == CouplingMode::Custom) {
                return allowed_pairs.count(make_pair_key(rf, cf)) > 0;
            }
            return true;
        };

        for (std::size_t i = 0; i < row_dofs.size(); ++i) {
            GlobalIndex r = row_dofs[i];
            if (r < 0 || r >= n_rows) {
                continue;
            }
            FieldId rf = row_fields[i];

            for (std::size_t j = 0; j < col_dofs.size(); ++j) {
                GlobalIndex c = col_dofs[j];
                if (c < 0 || c >= n_cols) {
                    continue;
                }
                FieldId cf = col_fields[j];

                if (!is_allowed(rf, cf)) {
                    continue;
                }

                pattern.addEntry(r, c);
                if (enforce_symmetric && r != c) {
                    // Symmetric closure independent of field coupling direction.
                    if (c >= 0 && c < n_rows && r >= 0 && r < n_cols) {
                        pattern.addEntry(c, r);
                    }
                }
            }
        }
    }

    // Ensure diagonal entries
    if (options_.ensure_diagonal && n_rows == n_cols) {
        pattern.ensureDiagonal();
    }

    // Ensure non-empty rows
    if (options_.ensure_non_empty_rows) {
        pattern.ensureNonEmptyRows();
    }

    // Finalize to CSR format
    pattern.finalize();

    return pattern;
}

SparsityPattern SparsityBuilder::build(std::span<const GlobalIndex> cell_ids) {
    validateConfiguration();

    IDofMapQuery* col_map = getEffectiveColDofMap();
    IDofFieldQuery* col_field_map = getEffectiveColFieldMap();

    GlobalIndex n_rows = row_dof_map_->getNumDofs();
    GlobalIndex n_cols = col_map->getNumDofs();
    FE_CHECK_ARG(col_map->getNumCells() == row_dof_map_->getNumCells(),
                 "Row and column DOF maps must have the same number of cells");

    SparsityPattern pattern(n_rows, n_cols);

    std::unordered_set<std::uint32_t> allowed_pairs;
    if (coupling_mode_ == CouplingMode::Custom) {
        allowed_pairs.reserve(field_couplings_.size() * 2u);
        for (const auto& c : field_couplings_) {
            allowed_pairs.insert(make_pair_key(c.row_field, c.col_field));
            if (c.bidirectional) {
                allowed_pairs.insert(make_pair_key(c.col_field, c.row_field));
            }
        }
    }

    const bool enforce_symmetric =
        options_.symmetric_pattern && (n_rows == n_cols);

    // Iterate over specified cells only
    for (GlobalIndex cell : cell_ids) {
        if (cell < 0 || cell >= row_dof_map_->getNumCells()) continue;

        auto row_dofs = row_dof_map_->getCellDofs(cell);
        auto col_dofs = col_map->getCellDofs(cell);

        if (coupling_mode_ == CouplingMode::Full || row_field_map_ == nullptr) {
            pattern.addElementCouplings(row_dofs, col_dofs);
            if (enforce_symmetric) {
                pattern.addElementCouplings(col_dofs, row_dofs);
            }
            continue;
        }

        std::vector<FieldId> row_fields(row_dofs.size(), INVALID_FIELD_ID);
        std::vector<FieldId> col_fields(col_dofs.size(), INVALID_FIELD_ID);

        for (std::size_t i = 0; i < row_dofs.size(); ++i) {
            GlobalIndex dof = row_dofs[i];
            if (dof < 0 || dof >= n_rows) continue;
            FieldId f = row_field_map_->getFieldId(dof);
            FE_CHECK_ARG(f != INVALID_FIELD_ID,
                         "Row DOF " + std::to_string(dof) + " is not mapped to any field");
            row_fields[i] = f;
        }

        for (std::size_t j = 0; j < col_dofs.size(); ++j) {
            GlobalIndex dof = col_dofs[j];
            if (dof < 0 || dof >= n_cols) continue;
            FieldId f = col_field_map->getFieldId(dof);
            FE_CHECK_ARG(f != INVALID_FIELD_ID,
                         "Column DOF " + std::to_string(dof) + " is not mapped to any field");
            col_fields[j] = f;
        }

        auto is_allowed = [&](FieldId rf, FieldId cf) -> bool {
            if (rf == INVALID_FIELD_ID || cf == INVALID_FIELD_ID) {
                return false;
            }
            if (coupling_mode_ == CouplingMode::Diagonal) {
                return rf == cf;
            }
            if (coupling_mode_ == CouplingMode::Custom) {
                return allowed_pairs.count(make_pair_key(rf, cf)) > 0;
            }
            return true;
        };

        for (std::size_t i = 0; i < row_dofs.size(); ++i) {
            GlobalIndex r = row_dofs[i];
            if (r < 0 || r >= n_rows) continue;
            FieldId rf = row_fields[i];

            for (std::size_t j = 0; j < col_dofs.size(); ++j) {
                GlobalIndex c = col_dofs[j];
                if (c < 0 || c >= n_cols) continue;
                FieldId cf = col_fields[j];

                if (!is_allowed(rf, cf)) continue;

                pattern.addEntry(r, c);
                if (enforce_symmetric && r != c) {
                    if (c >= 0 && c < n_rows && r >= 0 && r < n_cols) {
                        pattern.addEntry(c, r);
                    }
                }
            }
        }
    }

    if (options_.ensure_diagonal && n_rows == n_cols) {
        pattern.ensureDiagonal();
    }

    if (options_.ensure_non_empty_rows) {
        pattern.ensureNonEmptyRows();
    }

    pattern.finalize();
    return pattern;
}

SparsityPattern SparsityBuilder::build(
    GlobalIndex n_rows,
    GlobalIndex n_cols,
    GlobalIndex n_elements,
    std::function<std::span<const GlobalIndex>(GlobalIndex)> get_row_dofs,
    std::function<std::span<const GlobalIndex>(GlobalIndex)> get_col_dofs) {

    SparsityPattern pattern(n_rows, n_cols);

    std::unordered_set<std::uint32_t> allowed_pairs;
    if (coupling_mode_ == CouplingMode::Custom) {
        allowed_pairs.reserve(field_couplings_.size() * 2u);
        for (const auto& c : field_couplings_) {
            allowed_pairs.insert(make_pair_key(c.row_field, c.col_field));
            if (c.bidirectional) {
                allowed_pairs.insert(make_pair_key(c.col_field, c.row_field));
            }
        }
    }

    const bool enforce_symmetric =
        options_.symmetric_pattern && (n_rows == n_cols);

    for (GlobalIndex elem = 0; elem < n_elements; ++elem) {
        auto row_dofs = get_row_dofs(elem);
        auto col_dofs = get_col_dofs(elem);

        if (coupling_mode_ == CouplingMode::Full || row_field_map_ == nullptr) {
            pattern.addElementCouplings(row_dofs, col_dofs);
            if (enforce_symmetric) {
                pattern.addElementCouplings(col_dofs, row_dofs);
            }
            continue;
        }

        IDofFieldQuery* col_field_map = getEffectiveColFieldMap();

        std::vector<FieldId> row_fields(row_dofs.size(), INVALID_FIELD_ID);
        std::vector<FieldId> col_fields(col_dofs.size(), INVALID_FIELD_ID);

        for (std::size_t i = 0; i < row_dofs.size(); ++i) {
            GlobalIndex dof = row_dofs[i];
            if (dof < 0 || dof >= n_rows) continue;
            FieldId f = row_field_map_->getFieldId(dof);
            FE_CHECK_ARG(f != INVALID_FIELD_ID,
                         "Row DOF " + std::to_string(dof) + " is not mapped to any field");
            row_fields[i] = f;
        }

        for (std::size_t j = 0; j < col_dofs.size(); ++j) {
            GlobalIndex dof = col_dofs[j];
            if (dof < 0 || dof >= n_cols) continue;
            FieldId f = col_field_map->getFieldId(dof);
            FE_CHECK_ARG(f != INVALID_FIELD_ID,
                         "Column DOF " + std::to_string(dof) + " is not mapped to any field");
            col_fields[j] = f;
        }

        auto is_allowed = [&](FieldId rf, FieldId cf) -> bool {
            if (rf == INVALID_FIELD_ID || cf == INVALID_FIELD_ID) {
                return false;
            }
            if (coupling_mode_ == CouplingMode::Diagonal) {
                return rf == cf;
            }
            if (coupling_mode_ == CouplingMode::Custom) {
                return allowed_pairs.count(make_pair_key(rf, cf)) > 0;
            }
            return true;
        };

        for (std::size_t i = 0; i < row_dofs.size(); ++i) {
            GlobalIndex r = row_dofs[i];
            if (r < 0 || r >= n_rows) continue;
            FieldId rf = row_fields[i];

            for (std::size_t j = 0; j < col_dofs.size(); ++j) {
                GlobalIndex c = col_dofs[j];
                if (c < 0 || c >= n_cols) continue;
                FieldId cf = col_fields[j];

                if (!is_allowed(rf, cf)) continue;

                pattern.addEntry(r, c);
                if (enforce_symmetric && r != c) {
                    if (c >= 0 && c < n_rows && r >= 0 && r < n_cols) {
                        pattern.addEntry(c, r);
                    }
                }
            }
        }
    }

    if (options_.ensure_diagonal && n_rows == n_cols) {
        pattern.ensureDiagonal();
    }

    if (options_.ensure_non_empty_rows) {
        pattern.ensureNonEmptyRows();
    }

    pattern.finalize();
    return pattern;
}

// ============================================================================
// SparsityBuilder - Static convenience methods
// ============================================================================

SparsityPattern SparsityBuilder::buildFromDofMap(
    const dofs::DofMap& dof_map,
    const SparsityBuildOptions& options) {

    SparsityBuilder builder(dof_map);
    builder.setOptions(options);
    return builder.build();
}

SparsityPattern SparsityBuilder::buildFromDofMaps(
    const dofs::DofMap& row_dof_map,
    const dofs::DofMap& col_dof_map,
    const SparsityBuildOptions& options) {

    SparsityBuilder builder;
    builder.setRowDofMap(row_dof_map);
    builder.setColDofMap(col_dof_map);
    builder.setOptions(options);
    return builder.build();
}

SparsityPattern SparsityBuilder::buildFromArrays(
    GlobalIndex n_rows,
    GlobalIndex n_cols,
    GlobalIndex n_elements,
    std::span<const GlobalIndex> elem_row_offsets,
    std::span<const GlobalIndex> elem_row_dofs,
    std::span<const GlobalIndex> elem_col_offsets,
    std::span<const GlobalIndex> elem_col_dofs,
    const SparsityBuildOptions& options) {

    FE_CHECK_ARG(static_cast<GlobalIndex>(elem_row_offsets.size()) >= n_elements + 1,
                 "Row offsets array too small");
    FE_CHECK_ARG(static_cast<GlobalIndex>(elem_col_offsets.size()) >= n_elements + 1,
                 "Column offsets array too small");

    SparsityPattern pattern(n_rows, n_cols);

    for (GlobalIndex elem = 0; elem < n_elements; ++elem) {
        GlobalIndex row_start = elem_row_offsets[static_cast<std::size_t>(elem)];
        GlobalIndex row_end = elem_row_offsets[static_cast<std::size_t>(elem) + 1];
        GlobalIndex col_start = elem_col_offsets[static_cast<std::size_t>(elem)];
        GlobalIndex col_end = elem_col_offsets[static_cast<std::size_t>(elem) + 1];

        std::span<const GlobalIndex> row_dofs(
            elem_row_dofs.data() + row_start,
            static_cast<std::size_t>(row_end - row_start));

        std::span<const GlobalIndex> col_dofs(
            elem_col_dofs.data() + col_start,
            static_cast<std::size_t>(col_end - col_start));

        pattern.addElementCouplings(row_dofs, col_dofs);
    }

    if (options.symmetric_pattern && n_rows == n_cols) {
        for (GlobalIndex elem = 0; elem < n_elements; ++elem) {
            GlobalIndex row_start = elem_row_offsets[static_cast<std::size_t>(elem)];
            GlobalIndex row_end = elem_row_offsets[static_cast<std::size_t>(elem) + 1];
            GlobalIndex col_start = elem_col_offsets[static_cast<std::size_t>(elem)];
            GlobalIndex col_end = elem_col_offsets[static_cast<std::size_t>(elem) + 1];

            std::span<const GlobalIndex> row_dofs(
                elem_row_dofs.data() + row_start,
                static_cast<std::size_t>(row_end - row_start));

            std::span<const GlobalIndex> col_dofs(
                elem_col_dofs.data() + col_start,
                static_cast<std::size_t>(col_end - col_start));

            pattern.addElementCouplings(col_dofs, row_dofs);
        }
    }

    if (options.ensure_diagonal && n_rows == n_cols) {
        pattern.ensureDiagonal();
    }

    if (options.ensure_non_empty_rows) {
        pattern.ensureNonEmptyRows();
    }

    pattern.finalize();
    return pattern;
}

SparsityPattern SparsityBuilder::buildFromArrays(
    GlobalIndex n_dofs,
    GlobalIndex n_elements,
    std::span<const GlobalIndex> elem_offsets,
    std::span<const GlobalIndex> elem_dofs,
    const SparsityBuildOptions& options) {

    return buildFromArrays(n_dofs, n_dofs, n_elements,
                          elem_offsets, elem_dofs,
                          elem_offsets, elem_dofs,
                          options);
}

// ============================================================================
// DistributedSparsityBuilder
// ============================================================================

DistributedSparsityBuilder::DistributedSparsityBuilder(
    const dofs::DofMap& dof_map,
    GlobalIndex first_owned_dof,
    GlobalIndex n_owned_dofs,
    GlobalIndex global_n_dofs)
    : row_dof_map_(std::make_shared<DofMapAdapter>(dof_map)),
      col_dof_map_(nullptr),
      owned_rows_{first_owned_dof, first_owned_dof + n_owned_dofs},
      owned_cols_{first_owned_dof, first_owned_dof + n_owned_dofs},
      global_rows_(global_n_dofs),
      global_cols_(global_n_dofs)
{
}

void DistributedSparsityBuilder::setRowOwnership(
    GlobalIndex first_owned, GlobalIndex n_owned, GlobalIndex global_n_rows) {

    owned_rows_ = {first_owned, first_owned + n_owned};
    global_rows_ = global_n_rows;
}

void DistributedSparsityBuilder::setColOwnership(
    GlobalIndex first_owned, GlobalIndex n_owned, GlobalIndex global_n_cols) {

    owned_cols_ = {first_owned, first_owned + n_owned};
    global_cols_ = global_n_cols;
}

void DistributedSparsityBuilder::setRowDofMap(const dofs::DofMap& dof_map) {
    row_dof_map_ = std::make_shared<DofMapAdapter>(dof_map);
}

void DistributedSparsityBuilder::setRowDofMap(std::shared_ptr<IDofMapQuery> dof_map_query) {
    row_dof_map_ = std::move(dof_map_query);
}

void DistributedSparsityBuilder::setColDofMap(const dofs::DofMap& dof_map) {
    col_dof_map_ = std::make_shared<DofMapAdapter>(dof_map);
}

void DistributedSparsityBuilder::setColDofMap(std::shared_ptr<IDofMapQuery> dof_map_query) {
    col_dof_map_ = std::move(dof_map_query);
}

DistributedSparsityPattern DistributedSparsityBuilder::build() {
    FE_CHECK_ARG(row_dof_map_ != nullptr, "Row DOF map not set");
    FE_CHECK_ARG(global_rows_ > 0, "Global row count not set");

    // Use row map for cols if col map not set
    IDofMapQuery* col_map = col_dof_map_ ? col_dof_map_.get() : row_dof_map_.get();

    // Set col ownership if not explicitly set
    if (global_cols_ == 0) {
        global_cols_ = global_rows_;
        owned_cols_ = owned_rows_;
    }

    // Create distributed pattern
    DistributedSparsityPattern pattern(owned_rows_, owned_cols_, global_rows_, global_cols_);

    auto add_couplings = [&](std::span<const GlobalIndex> row_dofs,
                             std::span<const GlobalIndex> col_dofs) {
        if (options_.include_ghost_rows) {
            // Full pattern (diag + offdiag)
            pattern.addElementCouplings(row_dofs, col_dofs);
            return;
        }

        // Diagonal block only (owned columns)
        for (GlobalIndex row : row_dofs) {
            if (!owned_rows_.contains(row)) continue;
            for (GlobalIndex col : col_dofs) {
                if (!owned_cols_.contains(col)) continue;
                if (col >= 0 && col < global_cols_) {
                    pattern.addEntry(row, col);
                }
            }
        }
    };

    // Iterate over all local cells
    GlobalIndex n_cells = row_dof_map_->getNumCells();
    for (GlobalIndex cell = 0; cell < n_cells; ++cell) {
        auto row_dofs = row_dof_map_->getCellDofs(cell);
        auto col_dofs = col_map->getCellDofs(cell);

        add_couplings(row_dofs, col_dofs);
    }

    // Symmetrize if requested
    if (options_.symmetric_pattern && global_rows_ == global_cols_) {
        for (GlobalIndex cell = 0; cell < n_cells; ++cell) {
            auto row_dofs = row_dof_map_->getCellDofs(cell);
            auto col_dofs = col_map->getCellDofs(cell);
            add_couplings(col_dofs, row_dofs);
        }
    }

    // Ensure diagonal entries for owned rows
    if (options_.ensure_diagonal) {
        pattern.ensureDiagonal();
    }

    // Ensure non-empty rows
    if (options_.ensure_non_empty_rows) {
        pattern.ensureNonEmptyRows();
    }

    // Finalize - separates diag/offdiag
    pattern.finalize();

    return pattern;
}

DistributedSparsityPattern DistributedSparsityBuilder::build(std::span<const GlobalIndex> cell_ids) {
    FE_CHECK_ARG(row_dof_map_ != nullptr, "Row DOF map not set");
    FE_CHECK_ARG(global_rows_ > 0, "Global row count not set");

    IDofMapQuery* col_map = col_dof_map_ ? col_dof_map_.get() : row_dof_map_.get();

    if (global_cols_ == 0) {
        global_cols_ = global_rows_;
        owned_cols_ = owned_rows_;
    }

    DistributedSparsityPattern pattern(owned_rows_, owned_cols_, global_rows_, global_cols_);

    auto add_couplings = [&](std::span<const GlobalIndex> row_dofs,
                             std::span<const GlobalIndex> col_dofs) {
        if (options_.include_ghost_rows) {
            pattern.addElementCouplings(row_dofs, col_dofs);
            return;
        }

        for (GlobalIndex row : row_dofs) {
            if (!owned_rows_.contains(row)) continue;
            for (GlobalIndex col : col_dofs) {
                if (!owned_cols_.contains(col)) continue;
                if (col >= 0 && col < global_cols_) {
                    pattern.addEntry(row, col);
                }
            }
        }
    };

    for (GlobalIndex cell : cell_ids) {
        if (cell < 0 || cell >= row_dof_map_->getNumCells()) continue;

        auto row_dofs = row_dof_map_->getCellDofs(cell);
        auto col_dofs = col_map->getCellDofs(cell);

        add_couplings(row_dofs, col_dofs);
    }

    if (options_.symmetric_pattern && global_rows_ == global_cols_) {
        for (GlobalIndex cell : cell_ids) {
            if (cell < 0 || cell >= row_dof_map_->getNumCells()) continue;

            auto row_dofs = row_dof_map_->getCellDofs(cell);
            auto col_dofs = col_map->getCellDofs(cell);
            add_couplings(col_dofs, row_dofs);
        }
    }

    if (options_.ensure_diagonal) {
        pattern.ensureDiagonal();
    }

    if (options_.ensure_non_empty_rows) {
        pattern.ensureNonEmptyRows();
    }

    pattern.finalize();
    return pattern;
}

} // namespace sparsity
} // namespace FE
} // namespace svmp
