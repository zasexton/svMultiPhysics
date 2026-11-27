/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 * All Rights Reserved. See Copyright-SimVascular.txt for additional details.
 */
#ifndef SVMP_FE_ASSEMBLY_RESOLVED_CELL_TERMS_H
#define SVMP_FE_ASSEMBLY_RESOLVED_CELL_TERMS_H

#include "Assembly/Assembler.h"
#include "Forms/MixedBlockKernelSet.h"
#include "Forms/MonolithicCellKernel.h"

namespace svmp::FE::assembly {

template<class Visit>
bool visitResolvedCellContainer(const AssemblyKernel& kernel, Visit&& visit)
{
    if (const auto* mixed = dynamic_cast<const forms::MixedBlockKernelSet*>(&kernel); mixed && mixed->isResolved()) {
        visit(*mixed);
    } else if (const auto* mono = dynamic_cast<const forms::MonolithicCellKernel*>(&kernel); mono && mono->isResolved()) {
        visit(*mono);
    } else {
        return false;
    }
    return true;
}

template<class Assemble>
void forEachCellTermBlock(const FusedCellTerm& term, Assemble&& assemble)
{
    const bool resolved = visitResolvedCellContainer(*term.kernel, [&](const auto& container) {
        for (std::size_t b = 0; b < container.numBlocks(); ++b) {
            const auto& block = container.blockSpec(b);
            FusedCellTerm leaf = term;
            leaf.assemble_matrix = term.assemble_matrix && term.matrix_view && block.want_matrix;
            leaf.assemble_vector = term.assemble_vector && term.vector_view && block.want_vector;
            if (!leaf.assemble_matrix && !leaf.assemble_vector) continue;
            FE_CHECK_NOT_NULL(block.fallback_kernel.get(), "Resolved cell block requires an exact kernel");
            if (!block.fallback_kernel->hasCell()) continue;
            FE_CHECK_NOT_NULL(block.test_space, "Resolved cell block test space");
            FE_CHECK_NOT_NULL(block.trial_space, "Resolved cell block trial space");
            FE_CHECK_NOT_NULL(block.row_dof_map, "Resolved cell block row DOF map");
            FE_CHECK_NOT_NULL(block.col_dof_map, "Resolved cell block column DOF map");
            leaf.kernel = block.fallback_kernel.get();
            leaf.test_space = block.test_space;
            leaf.trial_space = block.trial_space;
            leaf.row_dof_map = block.row_dof_map;
            leaf.col_dof_map = block.col_dof_map;
            leaf.row_dof_offset = block.row_dof_offset;
            leaf.col_dof_offset = block.col_dof_offset;
            assemble(leaf, term.kernel);
        }
    });
    if (!resolved) assemble(term, nullptr);
}

} // namespace svmp::FE::assembly
#endif
