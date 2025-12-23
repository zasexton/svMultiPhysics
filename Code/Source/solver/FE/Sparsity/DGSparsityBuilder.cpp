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

#include "DGSparsityBuilder.h"
#include <algorithm>
#include <numeric>

namespace svmp {
namespace FE {
namespace sparsity {

// ============================================================================
// SimpleFaceConnectivity implementation
// ============================================================================

GlobalIndex SimpleFaceConnectivity::addInteriorFace(GlobalIndex cell_plus,
                                                     GlobalIndex cell_minus) {
    GlobalIndex face_id = static_cast<GlobalIndex>(interior_faces_.size());
    interior_faces_.emplace_back(cell_plus, cell_minus);
    return face_id;
}

GlobalIndex SimpleFaceConnectivity::addBoundaryFace(GlobalIndex cell, int tag) {
    GlobalIndex face_id = static_cast<GlobalIndex>(boundary_faces_.size());
    boundary_faces_.emplace_back(cell, tag);
    return face_id;
}

void SimpleFaceConnectivity::clear() {
    interior_faces_.clear();
    boundary_faces_.clear();
}

GlobalIndex SimpleFaceConnectivity::getNumInteriorFaces() const {
    return static_cast<GlobalIndex>(interior_faces_.size());
}

GlobalIndex SimpleFaceConnectivity::getNumBoundaryFaces() const {
    return static_cast<GlobalIndex>(boundary_faces_.size());
}

std::pair<GlobalIndex, GlobalIndex>
SimpleFaceConnectivity::getInteriorFaceCells(GlobalIndex face_id) const {
    FE_CHECK_ARG(face_id >= 0 &&
                 static_cast<std::size_t>(face_id) < interior_faces_.size(),
                 "Interior face index out of range");
    return interior_faces_[static_cast<std::size_t>(face_id)];
}

GlobalIndex SimpleFaceConnectivity::getBoundaryFaceCell(GlobalIndex face_id) const {
    FE_CHECK_ARG(face_id >= 0 &&
                 static_cast<std::size_t>(face_id) < boundary_faces_.size(),
                 "Boundary face index out of range");
    return boundary_faces_[static_cast<std::size_t>(face_id)].first;
}

int SimpleFaceConnectivity::getBoundaryTag(GlobalIndex face_id) const {
    FE_CHECK_ARG(face_id >= 0 &&
                 static_cast<std::size_t>(face_id) < boundary_faces_.size(),
                 "Boundary face index out of range");
    return boundary_faces_[static_cast<std::size_t>(face_id)].second;
}

// ============================================================================
// DGDofMapAdapter implementation
// ============================================================================

DGDofMapAdapter::DGDofMapAdapter(std::shared_ptr<IDofMapQuery> dof_map)
    : dof_map_(std::move(dof_map))
{
    FE_THROW_IF(!dof_map_, InvalidArgumentException, "DOF map cannot be null");
}

GlobalIndex DGDofMapAdapter::getNumDofs() const {
    return dof_map_->getNumDofs();
}

GlobalIndex DGDofMapAdapter::getNumLocalDofs() const {
    return dof_map_->getNumLocalDofs();
}

std::span<const GlobalIndex> DGDofMapAdapter::getCellDofs(GlobalIndex cell_id) const {
    return dof_map_->getCellDofs(cell_id);
}

GlobalIndex DGDofMapAdapter::getNumCells() const {
    return dof_map_->getNumCells();
}

bool DGDofMapAdapter::isOwnedDof(GlobalIndex dof) const {
    return dof_map_->isOwnedDof(dof);
}

std::pair<GlobalIndex, GlobalIndex> DGDofMapAdapter::getOwnedRange() const {
    return dof_map_->getOwnedRange();
}

std::span<const GlobalIndex> DGDofMapAdapter::getFaceDofs(
    GlobalIndex cell_id, LocalIndex /*local_face*/) const
{
    // For a generic adapter, return all cell DOFs as face DOFs
    // A proper DG implementation would return only DOFs on the specific face
    return dof_map_->getCellDofs(cell_id);
}

LocalIndex DGDofMapAdapter::getNumFacesPerCell(GlobalIndex /*cell_id*/) const {
    // Return a reasonable default; actual value depends on element type
    return 4;  // Default for quads/tets
}

bool DGDofMapAdapter::isDG() const {
    return true;  // Adapter assumes DG usage
}

// ============================================================================
// DGSparsityBuilder construction
// ============================================================================

DGSparsityBuilder::DGSparsityBuilder(
    std::shared_ptr<IDGDofMapQuery> dof_map,
    std::shared_ptr<IFaceConnectivity> face_connectivity)
    : dof_map_(std::move(dof_map)),
      face_connectivity_(std::move(face_connectivity))
{
}

// ============================================================================
// Configuration
// ============================================================================

void DGSparsityBuilder::setDofMap(std::shared_ptr<IDGDofMapQuery> dof_map) {
    dof_map_ = std::move(dof_map);
}

void DGSparsityBuilder::setDofMap(std::shared_ptr<IDofMapQuery> dof_map) {
    dof_map_ = std::make_shared<DGDofMapAdapter>(std::move(dof_map));
}

void DGSparsityBuilder::setFaceConnectivity(
    std::shared_ptr<IFaceConnectivity> connectivity)
{
    face_connectivity_ = std::move(connectivity);
}

// ============================================================================
// Building methods
// ============================================================================

SparsityPattern DGSparsityBuilder::build() {
    validateConfiguration();

    // Reset statistics
    last_stats_ = DGSparsityStats{};
    last_stats_.n_cells = dof_map_->getNumCells();
    last_stats_.n_interior_faces = face_connectivity_->getNumInteriorFaces();
    last_stats_.n_boundary_faces = face_connectivity_->getNumBoundaryFaces();

    const GlobalIndex n_dofs = dof_map_->getNumDofs();
    SparsityPattern pattern(n_dofs, n_dofs);

    // Build cell couplings
    if (options_.include_cell_couplings) {
        GlobalIndex nnz_before = pattern.getNnz();
        buildCellCouplings(pattern);
        last_stats_.cell_couplings = pattern.getNnz() - nnz_before;
    }

    // Build face couplings
    if (options_.include_face_couplings) {
        GlobalIndex nnz_before = pattern.getNnz();
        buildFaceCouplings(pattern);
        last_stats_.face_couplings = pattern.getNnz() - nnz_before;
    }

    // Build boundary face couplings
    if (options_.include_boundary_couplings) {
        GlobalIndex nnz_before = pattern.getNnz();
        buildBoundaryFaceCouplings(pattern);
        last_stats_.boundary_couplings = pattern.getNnz() - nnz_before;
    }

    // Ensure diagonal if requested
    if (options_.ensure_diagonal) {
        pattern.ensureDiagonal();
    }

    // Symmetrize if requested
    if (options_.symmetric_pattern) {
        // Build symmetric pattern by adding (j,i) for each (i,j)
        // This is done before finalization
        SparsityPattern temp(n_dofs, n_dofs);
        // We need to iterate over the building pattern
        // Copy to temp then add transpose
        pattern.finalize();
        auto sym = symmetrize(pattern);
        last_stats_.total_nnz = sym.getNnz();
        return sym;
    }

    pattern.finalize();
    last_stats_.total_nnz = pattern.getNnz();
    return pattern;
}

SparsityPattern DGSparsityBuilder::build(DGTermType term_type) {
    validateConfiguration();

    const GlobalIndex n_dofs = dof_map_->getNumDofs();
    SparsityPattern pattern(n_dofs, n_dofs);

    switch (term_type) {
        case DGTermType::VolumeIntegral:
            buildCellCouplings(pattern);
            break;

        case DGTermType::InteriorFace:
            buildFaceCouplings(pattern);
            break;

        case DGTermType::BoundaryFace:
            buildBoundaryFaceCouplings(pattern);
            break;

        case DGTermType::Penalty:
        case DGTermType::Stabilization:
            // These typically have same stencil as face terms
            buildFaceCouplings(pattern);
            break;

        case DGTermType::All:
            buildCellCouplings(pattern);
            buildFaceCouplings(pattern);
            buildBoundaryFaceCouplings(pattern);
            break;
    }

    if (options_.ensure_diagonal) {
        pattern.ensureDiagonal();
    }

    pattern.finalize();
    return pattern;
}

void DGSparsityBuilder::buildCellCouplings(SparsityPattern& pattern) {
    FE_THROW_IF(pattern.isFinalized(), InvalidArgumentException,
                "Cannot modify finalized pattern");

    const GlobalIndex n_cells = dof_map_->getNumCells();

    for (GlobalIndex cell = 0; cell < n_cells; ++cell) {
        addCellCoupling(pattern, cell);
    }
}

void DGSparsityBuilder::buildFaceCouplings(SparsityPattern& pattern) {
    FE_THROW_IF(pattern.isFinalized(), InvalidArgumentException,
                "Cannot modify finalized pattern");

    const GlobalIndex n_faces = face_connectivity_->getNumInteriorFaces();

    for (GlobalIndex face = 0; face < n_faces; ++face) {
        auto [cell_plus, cell_minus] = face_connectivity_->getInteriorFaceCells(face);
        addFaceCoupling(pattern, cell_plus, cell_minus);
    }
}

void DGSparsityBuilder::buildFaceCouplings(
    SparsityPattern& pattern,
    std::span<const GlobalIndex> face_ids)
{
    FE_THROW_IF(pattern.isFinalized(), InvalidArgumentException,
                "Cannot modify finalized pattern");

    for (GlobalIndex face : face_ids) {
        auto [cell_plus, cell_minus] = face_connectivity_->getInteriorFaceCells(face);
        addFaceCoupling(pattern, cell_plus, cell_minus);
    }
}

void DGSparsityBuilder::buildBoundaryFaceCouplings(SparsityPattern& pattern) {
    FE_THROW_IF(pattern.isFinalized(), InvalidArgumentException,
                "Cannot modify finalized pattern");

    const GlobalIndex n_bfaces = face_connectivity_->getNumBoundaryFaces();

    for (GlobalIndex face = 0; face < n_bfaces; ++face) {
        GlobalIndex cell = face_connectivity_->getBoundaryFaceCell(face);
        addBoundaryCoupling(pattern, cell);
    }
}

void DGSparsityBuilder::buildBoundaryFaceCouplings(
    SparsityPattern& pattern, int boundary_tag)
{
    FE_THROW_IF(pattern.isFinalized(), InvalidArgumentException,
                "Cannot modify finalized pattern");

    const GlobalIndex n_bfaces = face_connectivity_->getNumBoundaryFaces();

    for (GlobalIndex face = 0; face < n_bfaces; ++face) {
        if (face_connectivity_->getBoundaryTag(face) == boundary_tag) {
            GlobalIndex cell = face_connectivity_->getBoundaryFaceCell(face);
            addBoundaryCoupling(pattern, cell);
        }
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

void DGSparsityBuilder::addCellCoupling(SparsityPattern& pattern,
                                         GlobalIndex cell_id) {
    auto dofs = dof_map_->getCellDofs(cell_id);
    pattern.addElementCouplings(dofs);
}

void DGSparsityBuilder::addFaceCoupling(SparsityPattern& pattern,
                                         GlobalIndex cell_plus,
                                         GlobalIndex cell_minus) {
    auto dofs_plus = dof_map_->getCellDofs(cell_plus);
    auto dofs_minus = dof_map_->getCellDofs(cell_minus);

    // DG face coupling creates all combinations:
    // - (K+, K+): cell-local on plus side
    // - (K-, K-): cell-local on minus side
    // - (K+, K-): cross coupling plus to minus
    // - (K-, K+): cross coupling minus to plus

    // Cell-local couplings (if not already added by buildCellCouplings)
    // Note: addElementCouplings handles duplicates via set semantics
    pattern.addElementCouplings(dofs_plus);
    pattern.addElementCouplings(dofs_minus);

    // Cross couplings (the DG-specific part)
    pattern.addElementCouplings(dofs_plus, dofs_minus);
    pattern.addElementCouplings(dofs_minus, dofs_plus);
}

void DGSparsityBuilder::addBoundaryCoupling(SparsityPattern& pattern,
                                             GlobalIndex cell_id) {
    // Boundary face coupling is typically cell-local
    // (boundary terms only involve DOFs from the adjacent cell)
    auto dofs = dof_map_->getCellDofs(cell_id);
    pattern.addElementCouplings(dofs);
}

void DGSparsityBuilder::validateConfiguration() const {
    FE_THROW_IF(!dof_map_, InvalidArgumentException,
                "DOF map not configured");
    FE_THROW_IF(!face_connectivity_, InvalidArgumentException,
                "Face connectivity not configured");
}

GlobalIndex DGSparsityBuilder::estimateNnz(
    GlobalIndex n_cells,
    GlobalIndex dofs_per_cell,
    double avg_neighbors)
{
    // Cell-local: n_cells * dofs_per_cell^2
    GlobalIndex cell_nnz = n_cells * dofs_per_cell * dofs_per_cell;

    // Face couplings: each interior face creates cross-couplings
    // Number of interior faces ~ n_cells * avg_neighbors / 2 (each face shared)
    // Each face adds 2 * dofs_per_cell^2 cross terms (plus to minus and vice versa)
    auto n_faces = static_cast<GlobalIndex>(static_cast<double>(n_cells) * avg_neighbors / 2.0);
    GlobalIndex face_nnz = n_faces * 2 * dofs_per_cell * dofs_per_cell;

    // Total (with some overlap from cell-local terms)
    return cell_nnz + face_nnz;
}

} // namespace sparsity
} // namespace FE
} // namespace svmp
