/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_TESTS_SUPPORT_DISTRIBUTED_TIME_HISTORY_H
#define SVMP_FE_TESTS_SUPPORT_DISTRIBUTED_TIME_HISTORY_H

#include "Backends/Interfaces/BackendFactory.h"
#include "Sparsity/DistributedSparsityPattern.h"
#include "TimeStepping/TimeHistory.h"

#include <gtest/gtest.h>

#include <array>

namespace svmp::FE::test {

inline void checkDistributedHistoryRepack(backends::BackendFactory& factory, int rank, int ranks)
{
    const GlobalIndex n = 3 * ranks;
    // Before the matrix exists the backend balances 3 rows per rank.
    auto history = timestepping::TimeHistory::allocate(factory, n, 3, true);
    const auto slots = [&]() {
        return std::array<backends::GenericVector*, 6>{
            &history.u(), &history.uPrev(), &history.uPrev2(),
            &history.uPrevK(3), &history.uDot(), &history.uDDot()};
    };
    int slot = 0;
    for (auto* vector : slots()) {
        auto view = vector->createAssemblyView();
        view->beginAssemblyPhase();
        for (auto gid : vector->ownedGlobalRows()) {
            view->addVectorEntry(gid, Real(100 * slot + gid + 1), assembly::AddMode::Insert);
        }
        view->finalizeAssembly();
        ++slot;
    }
    // Transfer rows from rank one to rank zero, leaving an empty owner on four ranks.
    // The final column also supplies
    // a ghost from a non-neighbouring rank on four ranks.
    const GlobalIndex move = ranks >= 3 ? 3 : ranks - 1;
    const GlobalIndex count = 3 + (rank == 0 ? move : rank == 1 ? -move : 0);
    const GlobalIndex start = 3 * rank + (rank == 1 ? move : 0);
    sparsity::DistributedSparsityPattern layout(start, count, start, count, n, n);
    for (GlobalIndex i = start; i < start + count; ++i) {
        layout.addEntry(i, i);
        layout.addEntry(i, n - 1);
    }
    layout.finalize();
    auto matrix = factory.createMatrix(layout);
    ASSERT_NO_THROW(history.repack(factory));
    slot = 0;
    for (auto* vector : slots()) {
        const auto owned = vector->ownedGlobalRows();
        ASSERT_EQ(owned.size(), static_cast<std::size_t>(count));
        auto read = vector->createGhostedReadView();
        for (GlobalIndex i = 0; i < count; ++i) {
            EXPECT_EQ(owned[static_cast<std::size_t>(i)], start + i);
            EXPECT_DOUBLE_EQ(read->getVectorEntry(start + i), 100 * slot + start + i + 1);
        }
        if (count > 0) {
            EXPECT_DOUBLE_EQ(read->getVectorEntry(n - 1), 100 * slot + n);
        }
        ++slot;
    }
}

} // namespace svmp::FE::test

#endif
