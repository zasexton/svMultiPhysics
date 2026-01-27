/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Forms/FormExpr.h"
#include "Forms/Index.h"
#include "Forms/Tensor/LoopStructure.h"
#include "Forms/Tensor/TensorAllocation.h"

#include <cstddef>
#include <cstdint>
#include <new>
#include <vector>

namespace svmp::FE::forms::tensor {

namespace {

[[nodiscard]] FormExpr makeTensor(int n)
{
    std::vector<std::vector<FormExpr>> rows;
    rows.resize(static_cast<std::size_t>(n));
    for (int r = 0; r < n; ++r) {
        rows[static_cast<std::size_t>(r)].resize(static_cast<std::size_t>(n));
        for (int c = 0; c < n; ++c) {
            rows[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] =
                FormExpr::constant(1.0 + 0.01 * static_cast<Real>(r) + 0.001 * static_cast<Real>(c));
        }
    }
    return FormExpr::asTensor(std::move(rows));
}

} // namespace

TEST(TensorAllocation, ReusesArenaForLargeTemporariesAcrossTerms)
{
    // Use extent 10 so the intermediate matrix temporary has 100 entries (>81).
    const auto A = makeTensor(10);
    const auto B = makeTensor(10);
    const auto C = makeTensor(10);

    forms::Index i("i", forms::IndexSet{10});
    forms::Index j("j", forms::IndexSet{10});
    forms::Index k("k", forms::IndexSet{10});

    const auto term = A(i, j) * B(j, k) * C(k, i);
    const auto expr = term + term;

    const auto p = generateLoopNest(expr);
    ASSERT_TRUE(p.ok) << p.message;
    EXPECT_TRUE(p.isScalar());
    ASSERT_EQ(p.contributions.size(), 2u);
    EXPECT_GE(p.contributions[0].available_after_op, 0);
    EXPECT_GE(p.contributions[1].available_after_op, 0);
    EXPECT_LT(p.contributions[0].available_after_op, p.contributions[1].available_after_op);

    TensorAllocationOptions alloc_opts;
    alloc_opts.stack_max_entries = 81;
    alloc_opts.alignment_bytes = 64;
    const auto plan = planTensorAllocation(p, alloc_opts);
    ASSERT_TRUE(plan.ok) << plan.message;

    EXPECT_EQ(plan.alignment_bytes, 64u);
    EXPECT_EQ(plan.arena_slots, 1);
    EXPECT_EQ(plan.stack_slots, 1);
    EXPECT_EQ(plan.arena_bytes % 64u, 0u);
    EXPECT_EQ(plan.stack_bytes % 64u, 0u);
    ASSERT_EQ(plan.arena_slot_offsets.size(), 1u);
    ASSERT_EQ(plan.stack_slot_offsets.size(), 1u);
    EXPECT_EQ(plan.arena_slot_offsets[0] % 64u, 0u);
    EXPECT_EQ(plan.stack_slot_offsets[0] % 64u, 0u);

    std::vector<int> arena_tensors;
    std::vector<int> stack_tensors;
    for (const auto& a : plan.allocations) {
        if (a.kind == TensorAllocKind::TemporaryArena) {
            arena_tensors.push_back(a.tensor_id);
            EXPECT_EQ(a.slot, 0);
            EXPECT_EQ(a.arena_offset % 64u, 0u);
            EXPECT_EQ(a.arena_offset, plan.arena_slot_offsets[static_cast<std::size_t>(a.slot)]);
        }
        if (a.kind == TensorAllocKind::TemporaryStack) {
            stack_tensors.push_back(a.tensor_id);
            EXPECT_EQ(a.slot, 0);
        }
    }
    EXPECT_GE(arena_tensors.size(), 2u); // one arena temporary per term (reused).
    EXPECT_GE(stack_tensors.size(), 2u); // scalar temporaries per term (reused).

    TensorTempWorkspace ws;
    ws.reset(plan, 2);

    const auto a0 = ws.spanForTensor(arena_tensors[0], 0);
    const auto a1 = ws.spanForTensor(arena_tensors[1], 0);
    ASSERT_FALSE(a0.empty());
    ASSERT_FALSE(a1.empty());
    EXPECT_EQ(a0.data(), a1.data()); // same arena slot => same pointer
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a0.data()) % 64u, 0u);

    const auto b0 = ws.spanForTensor(stack_tensors[0], 0);
    const auto b1 = ws.spanForTensor(stack_tensors[1], 0);
    ASSERT_FALSE(b0.empty());
    ASSERT_FALSE(b1.empty());
    EXPECT_EQ(b0.data(), b1.data()); // same stack slot => same pointer
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(b0.data()) % 64u, 0u);

    const auto a0_q1 = ws.spanForTensor(arena_tensors[0], 1);
    ASSERT_FALSE(a0_q1.empty());
    EXPECT_NE(a0_q1.data(), a0.data()); // distinct batch slices
}

TEST(TensorAllocation, ExternalBuffersBackTheWorkspace)
{
    const auto A = makeTensor(10);
    const auto B = makeTensor(10);
    const auto C = makeTensor(10);

    forms::Index i("i", forms::IndexSet{10});
    forms::Index j("j", forms::IndexSet{10});
    forms::Index k("k", forms::IndexSet{10});

    const auto term = A(i, j) * B(j, k) * C(k, i);
    const auto expr = term + term;

    const auto p = generateLoopNest(expr);
    ASSERT_TRUE(p.ok) << p.message;

    TensorAllocationOptions alloc_opts;
    alloc_opts.stack_max_entries = 81;
    alloc_opts.alignment_bytes = 64;
    const auto plan = planTensorAllocation(p, alloc_opts);
    ASSERT_TRUE(plan.ok) << plan.message;

    std::vector<int> arena_tensors;
    std::vector<int> stack_tensors;
    for (const auto& a : plan.allocations) {
        if (a.kind == TensorAllocKind::TemporaryArena) arena_tensors.push_back(a.tensor_id);
        if (a.kind == TensorAllocKind::TemporaryStack) stack_tensors.push_back(a.tensor_id);
    }
    ASSERT_FALSE(arena_tensors.empty());
    ASSERT_FALSE(stack_tensors.empty());

    constexpr std::size_t batch = 2;
    const std::size_t stack_total = plan.stack_bytes * batch;
    const std::size_t arena_total = plan.arena_bytes * batch;
    ASSERT_GT(stack_total, 0u);
    ASSERT_GT(arena_total, 0u);

    auto* stack_buf = static_cast<std::byte*>(::operator new(stack_total, std::align_val_t(plan.alignment_bytes)));
    auto* arena_buf = static_cast<std::byte*>(::operator new(arena_total, std::align_val_t(plan.alignment_bytes)));

    {
        TensorTempWorkspace ws;
        TensorTempWorkspace::ExternalBuffers bufs;
        bufs.stack = stack_buf;
        bufs.stack_bytes = stack_total;
        bufs.arena = arena_buf;
        bufs.arena_bytes = arena_total;
        ws.resetWithExternalBuffers(plan, batch, bufs);

        const auto s = ws.spanForTensor(stack_tensors[0], 0);
        ASSERT_FALSE(s.empty());
        const auto* sp = reinterpret_cast<const std::byte*>(s.data());
        EXPECT_GE(sp, stack_buf);
        EXPECT_LT(sp, stack_buf + stack_total);

        const auto a = ws.spanForTensor(arena_tensors[0], 0);
        ASSERT_FALSE(a.empty());
        const auto* ap = reinterpret_cast<const std::byte*>(a.data());
        EXPECT_GE(ap, arena_buf);
        EXPECT_LT(ap, arena_buf + arena_total);
    }

    // Workspace must not free externally-managed buffers.
    ::operator delete(stack_buf, std::align_val_t(plan.alignment_bytes));
    ::operator delete(arena_buf, std::align_val_t(plan.alignment_bytes));
}

} // namespace svmp::FE::forms::tensor
