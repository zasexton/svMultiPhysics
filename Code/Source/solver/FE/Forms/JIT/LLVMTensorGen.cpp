/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/JIT/LLVMTensorGen.h"

#ifndef SVMP_FE_ENABLE_LLVM_JIT
#define SVMP_FE_ENABLE_LLVM_JIT 0
#endif

#if SVMP_FE_ENABLE_LLVM_JIT
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Metadata.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace svmp {
namespace FE {
namespace forms {
namespace jit {

namespace {

[[nodiscard]] llvm::MDNode* makeLoopMetadata(llvm::LLVMContext& ctx,
                                             bool vectorize,
                                             int vector_width,
                                             bool polly) // NOLINT(*-missing-std-forward)
{
    llvm::SmallVector<llvm::Metadata*, 8> md_args;

    // Temporary node used to create the self-referential loop id.
    llvm::TempMDNode tmp = llvm::MDNode::getTemporary(ctx, {});

    md_args.push_back(tmp.get());

    const auto addBool = [&](const char* key, bool v) {
        auto* md_key = llvm::MDString::get(ctx, key);
        auto* md_val = llvm::ConstantAsMetadata::get(
            llvm::ConstantInt::get(llvm::Type::getInt1Ty(ctx), v ? 1 : 0));
        md_args.push_back(llvm::MDNode::get(ctx, {md_key, md_val}));
    };

    const auto addI32 = [&](const char* key, int v) {
        auto* md_key = llvm::MDString::get(ctx, key);
        auto* md_val = llvm::ConstantAsMetadata::get(
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), static_cast<std::uint32_t>(std::max(0, v))));
        md_args.push_back(llvm::MDNode::get(ctx, {md_key, md_val}));
    };

    if (vectorize) {
        addBool("llvm.loop.vectorize.enable", true);
        if (vector_width > 1) {
            addI32("llvm.loop.vectorize.width", vector_width);
        }
    }

    // Polly is optional; attach a conservative marker that is ignored by LLVM
    // passes when Polly is not present.
    if (polly) {
        addBool("llvm.loop.polly.enable", true);
    }

    auto* loop_id = llvm::MDNode::get(ctx, md_args);
    loop_id->replaceOperandWith(0, loop_id);
    return loop_id;
}

struct LoopBlocks {
    llvm::BasicBlock* preheader{nullptr};
    llvm::BasicBlock* header{nullptr};
    llvm::BasicBlock* body{nullptr};
    llvm::BasicBlock* latch{nullptr};
    llvm::BasicBlock* exit{nullptr};
};

[[nodiscard]] LoopBlocks createLoopBlocks(llvm::LLVMContext& ctx,
                                         llvm::Function& fn,
                                         std::string prefix)
{
    LoopBlocks b;
    b.preheader = nullptr; // filled by caller (current insertion block)
    b.header = llvm::BasicBlock::Create(ctx, prefix + ".h", &fn);
    b.body = llvm::BasicBlock::Create(ctx, prefix + ".b", &fn);
    b.latch = llvm::BasicBlock::Create(ctx, prefix + ".l", &fn);
    b.exit = llvm::BasicBlock::Create(ctx, prefix + ".x", &fn);
    return b;
}

[[nodiscard]] std::vector<std::uint64_t> denseStrides(const std::vector<int>& extents)
{
    const std::size_t r = extents.size();
    std::vector<std::uint64_t> strides(r, 1ULL);
    if (r == 0u) {
        return strides;
    }
    for (std::size_t k = r; k-- > 0u;) {
        if (k + 1u < r) {
            const auto e = static_cast<std::uint64_t>(std::max(0, extents[k + 1u]));
            strides[k] = strides[k + 1u] * std::max<std::uint64_t>(1ULL, e);
        } else {
            strides[k] = 1ULL;
        }
    }
    return strides;
}

} // namespace

LLVMTensorGen::LLVMTensorGen(llvm::LLVMContext& ctx,
                             llvm::IRBuilder<>& builder,
                             llvm::Function& fn,
                             LLVMTensorGenOptions options)
    : ctx_(&ctx),
      builder_(&builder),
      fn_(&fn),
      options_(options)
{
}

llvm::Value* LLVMTensorGen::emitScalar(const forms::tensor::TensorIR& ir,
                                       const ScalarEvalFn& eval_scalar,
                                       const LoadInputTensorElementFn& load_input) const
{
    if (!ctx_ || !builder_ || !fn_) {
        throw std::runtime_error("LLVMTensorGen: internal error (missing LLVM state)");
    }

    if (!ir.program.ok) {
        throw std::runtime_error(ir.program.message.empty() ? "LLVMTensorGen: invalid TensorIR program" : ir.program.message);
    }

    if (ir.program.output.rank != 0) {
        throw std::runtime_error("LLVMTensorGen: emitScalar requires scalar output (rank==0)");
    }

    auto& ctx = *ctx_;
    auto& builder = *builder_;
    auto& fn = *fn_;

    auto* i32 = builder.getInt32Ty();
    auto* i64 = builder.getInt64Ty();
    auto* f64 = builder.getDoubleTy();

    auto* f64c = [&](double v) -> llvm::Constant* { return llvm::ConstantFP::get(f64, v); };
    auto* i32c = [&](std::uint32_t v) -> llvm::ConstantInt* { return llvm::ConstantInt::get(i32, v); };
    auto* i64c = [&](std::uint64_t v) -> llvm::ConstantInt* { return llvm::ConstantInt::get(i64, v); };

    // Allocate temporary buffers at function entry (not inside loops).
    llvm::Value* stack_base = nullptr;
    llvm::Value* arena_base = nullptr;
    {
        llvm::IRBuilder<> alloca_builder(&fn.getEntryBlock(), fn.getEntryBlock().begin());
        if (ir.allocation.stack_bytes > 0u) {
            auto* bytes = i64c(static_cast<std::uint64_t>(ir.allocation.stack_bytes));
            auto* alloca = alloca_builder.CreateAlloca(alloca_builder.getInt8Ty(), bytes, "tensor.stack");
            alloca->setAlignment(llvm::Align(ir.allocation.alignment_bytes));
            stack_base = alloca;
        }
        if (ir.allocation.arena_bytes > 0u) {
            auto* bytes = i64c(static_cast<std::uint64_t>(ir.allocation.arena_bytes));
            auto* alloca = alloca_builder.CreateAlloca(alloca_builder.getInt8Ty(), bytes, "tensor.arena");
            alloca->setAlignment(llvm::Align(ir.allocation.alignment_bytes));
            arena_base = alloca;
        }
    }

    // Map tensor id -> base pointer (double*) for temporaries.
    std::vector<llvm::Value*> tensor_ptrs;
    tensor_ptrs.resize(ir.program.tensors.size(), nullptr);
    for (std::size_t tid = 0; tid < ir.program.tensors.size(); ++tid) {
        const auto& a = ir.allocation.allocations[tid];
        if (a.kind != forms::tensor::TensorAllocKind::TemporaryStack &&
            a.kind != forms::tensor::TensorAllocKind::TemporaryArena) {
            continue;
        }

        if (a.slot < 0) {
            throw std::runtime_error("LLVMTensorGen: temporary tensor allocation missing slot id");
        }

        std::uint64_t off = 0u;
        if (a.kind == forms::tensor::TensorAllocKind::TemporaryStack) {
            if (!stack_base) {
                throw std::runtime_error("LLVMTensorGen: missing stack base buffer");
            }
            if (static_cast<std::size_t>(a.slot) >= ir.allocation.stack_slot_offsets.size()) {
                throw std::runtime_error("LLVMTensorGen: stack slot index out of range");
            }
            off = static_cast<std::uint64_t>(ir.allocation.stack_slot_offsets[static_cast<std::size_t>(a.slot)]);
        } else {
            if (!arena_base) {
                throw std::runtime_error("LLVMTensorGen: missing arena base buffer");
            }
            off = static_cast<std::uint64_t>(a.arena_offset);
        }

        auto* base = (a.kind == forms::tensor::TensorAllocKind::TemporaryStack) ? stack_base : arena_base;
        auto* ptr_i8 = builder.CreateGEP(builder.getInt8Ty(), base, i64c(off), "t" + std::to_string(tid) + ".i8");
        auto* ptr_f64 = builder.CreateBitCast(ptr_i8, llvm::PointerType::getUnqual(f64), "t" + std::to_string(tid));
        tensor_ptrs[tid] = ptr_f64;
    }

    // Index environment: map canonical index id -> llvm i32 value.
    std::size_t max_index_id = 0u;
    for (const auto& t : ir.program.tensors) {
        for (const int id : t.axes) {
            if (id >= 0) max_index_id = std::max(max_index_id, static_cast<std::size_t>(id));
        }
    }
    for (const auto& op : ir.program.ops) {
        for (const auto& li : op.loops) {
            if (li.id >= 0) max_index_id = std::max(max_index_id, static_cast<std::size_t>(li.id));
        }
    }
    std::vector<llvm::Value*> idx_env(max_index_id + 1u, nullptr);

    const auto indexValueForId = [&](int id) -> llvm::Value* {
        if (id < 0) {
            throw std::runtime_error("LLVMTensorGen: negative index id");
        }
        const std::size_t k = static_cast<std::size_t>(id);
        if (k >= idx_env.size() || idx_env[k] == nullptr) {
            throw std::runtime_error("LLVMTensorGen: missing index assignment for id=" + std::to_string(id));
        }
        return idx_env[k];
    };

    const auto tensorOffset = [&](const forms::tensor::TensorSpec& spec) -> llvm::Value* {
        if (spec.rank == 0 || spec.axes.empty()) {
            return i64c(0u);
        }
        const auto strides = denseStrides(spec.extents);
        llvm::Value* idx = i64c(0u);
        for (std::size_t k = 0; k < spec.axes.size(); ++k) {
            const int id = spec.axes[k];
            auto* iv = indexValueForId(id);
            auto* iv64 = builder.CreateZExt(iv, i64);
            idx = builder.CreateAdd(idx, builder.CreateMul(iv64, i64c(strides[k])));
        }
        return idx;
    };

    const auto loadTensorElement = [&](int tensor_id, const forms::tensor::TensorSpec& spec) -> llvm::Value* {
        if (tensor_id < 0 || static_cast<std::size_t>(tensor_id) >= ir.program.tensors.size()) {
            throw std::runtime_error("LLVMTensorGen: tensor id out of range");
        }

        const auto& a = ir.allocation.allocations[static_cast<std::size_t>(tensor_id)];

        if (spec.storage == forms::tensor::TensorStorageKind::KroneckerDelta) {
            if (spec.rank == 0) {
                return f64c(1.0);
            }
            if (spec.rank == 1) {
                return f64c(1.0);
            }
            if (spec.rank == 2 && spec.axes.size() == 2u) {
                auto* ii = indexValueForId(spec.axes[0]);
                auto* jj = indexValueForId(spec.axes[1]);
                auto* is_eq = builder.CreateICmpEQ(ii, jj);
                return builder.CreateSelect(is_eq, f64c(1.0), f64c(0.0));
            }
            throw std::runtime_error("LLVMTensorGen: KroneckerDelta unsupported rank for codegen");
        }

        if (a.kind == forms::tensor::TensorAllocKind::Input) {
            return load_input(tensor_id, spec, idx_env);
        }

        if (a.kind != forms::tensor::TensorAllocKind::TemporaryStack &&
            a.kind != forms::tensor::TensorAllocKind::TemporaryArena) {
            throw std::runtime_error("LLVMTensorGen: tensor is not loadable (kind mismatch)");
        }

        auto* base_ptr = tensor_ptrs[static_cast<std::size_t>(tensor_id)];
        if (!base_ptr) {
            throw std::runtime_error("LLVMTensorGen: missing temporary buffer pointer for tensor id");
        }

        auto* off = tensorOffset(spec);
        auto* ptr = builder.CreateGEP(f64, base_ptr, off);
        return builder.CreateLoad(f64, ptr);
    };

    const auto storeTensorElement = [&](int tensor_id, const forms::tensor::TensorSpec& spec, llvm::Value* value) -> void {
        if (tensor_id < 0 || static_cast<std::size_t>(tensor_id) >= ir.program.tensors.size()) {
            throw std::runtime_error("LLVMTensorGen: tensor id out of range");
        }

        const auto& a = ir.allocation.allocations[static_cast<std::size_t>(tensor_id)];
        if (a.kind != forms::tensor::TensorAllocKind::TemporaryStack &&
            a.kind != forms::tensor::TensorAllocKind::TemporaryArena) {
            throw std::runtime_error("LLVMTensorGen: store to non-temporary tensor is not supported");
        }

        auto* base_ptr = tensor_ptrs[static_cast<std::size_t>(tensor_id)];
        if (!base_ptr) {
            throw std::runtime_error("LLVMTensorGen: missing temporary buffer pointer for tensor id");
        }

        auto* off = tensorOffset(spec);
        auto* ptr = builder.CreateGEP(f64, base_ptr, off);
        builder.CreateStore(value, ptr);
    };

    // Emit a nested reduction over sum-axes loops for a given contraction op.
    const auto emitSumLoops = [&](const auto& self,
                                  const forms::tensor::ContractionOp& op,
                                  std::size_t sum_level) -> llvm::Value* {
        if (sum_level >= op.sum_axes.size()) {
            const auto& lhs_spec = ir.program.tensors[static_cast<std::size_t>(op.lhs)];

            auto* lhs = loadTensorElement(op.lhs, lhs_spec);
            if (op.kind == forms::tensor::ContractionOp::Kind::Reduction) {
                return lhs;
            }
            const auto& rhs_spec = ir.program.tensors[static_cast<std::size_t>(op.rhs)];
            auto* rhs = loadTensorElement(op.rhs, rhs_spec);
            return builder.CreateFMul(lhs, rhs);
        }

        const int target_id = op.sum_axes[sum_level];
        auto it = std::find_if(op.loops.begin(), op.loops.end(),
                               [&](const forms::tensor::LoopIndex& li) { return li.id == target_id; });
        if (it == op.loops.end()) {
            throw std::runtime_error("LLVMTensorGen: missing LoopIndex metadata for sum axis id=" + std::to_string(target_id));
        }
        const auto& li = *it;
        if (li.extent <= 0) {
            return f64c(0.0);
        }

        const std::string prefix = "t.sum." + std::to_string(sum_level);
        auto blocks = createLoopBlocks(ctx, fn, prefix);
        blocks.preheader = builder.GetInsertBlock();

        auto* start = i32c(0);
        if (li.lower_bound_id >= 0) {
            auto* lb = indexValueForId(li.lower_bound_id);
            start = builder.CreateAdd(lb, i32c(static_cast<std::uint32_t>(std::max(0, li.lower_bound_offset))));
        }

        builder.CreateBr(blocks.header);

        builder.SetInsertPoint(blocks.header);
        auto* idx_phi = builder.CreatePHI(i32, 2, prefix + ".i");
        idx_phi->addIncoming(start, blocks.preheader);

        auto* acc_phi = builder.CreatePHI(f64, 2, prefix + ".acc");
        acc_phi->addIncoming(f64c(0.0), blocks.preheader);

        auto* cond = builder.CreateICmpULT(idx_phi, i32c(static_cast<std::uint32_t>(li.extent)));
        builder.CreateCondBr(cond, blocks.body, blocks.exit);

        builder.SetInsertPoint(blocks.body);
        llvm::Value* prev = nullptr;
        {
            const std::size_t id = static_cast<std::size_t>(li.id);
            prev = (id < idx_env.size()) ? idx_env[id] : nullptr;
            if (id >= idx_env.size()) idx_env.resize(id + 1u, nullptr);
            idx_env[id] = idx_phi;
        }

        auto* inner = self(self, op, sum_level + 1u);

        {
            const std::size_t id = static_cast<std::size_t>(li.id);
            idx_env[id] = prev;
        }

        builder.CreateBr(blocks.latch);

        builder.SetInsertPoint(blocks.latch);
        auto* acc_next = builder.CreateFAdd(acc_phi, inner);
        auto* idx_next = builder.CreateAdd(idx_phi, i32c(1));

        auto* backedge = builder.CreateBr(blocks.header);
        if (options_.vectorize && li.vectorize) {
            auto* md = makeLoopMetadata(ctx, /*vectorize=*/true, li.vector_width, options_.enable_polly);
            backedge->setMetadata("llvm.loop", md);
        } else if (options_.enable_polly) {
            auto* md = makeLoopMetadata(ctx, /*vectorize=*/false, /*vector_width=*/1, /*polly=*/true);
            backedge->setMetadata("llvm.loop", md);
        }

        idx_phi->addIncoming(idx_next, blocks.latch);
        acc_phi->addIncoming(acc_next, blocks.latch);

        builder.SetInsertPoint(blocks.exit);
        return acc_phi;
    };

    // Emit nested output loops for an op, writing the computed tensor element.
    const auto emitOutLoops = [&](const auto& self,
                                  const forms::tensor::ContractionOp& op,
                                  std::size_t out_level) -> void {
        const auto& out_spec = ir.program.tensors[static_cast<std::size_t>(op.out)];

        if (out_level >= op.out_axes.size()) {
            auto* acc = emitSumLoops(emitSumLoops, op, 0u);
            storeTensorElement(op.out, out_spec, acc);
            return;
        }

        const int target_id = op.out_axes[out_level];
        auto it = std::find_if(op.loops.begin(), op.loops.end(),
                               [&](const forms::tensor::LoopIndex& li) { return li.id == target_id; });
        if (it == op.loops.end()) {
            throw std::runtime_error("LLVMTensorGen: missing LoopIndex metadata for out axis id=" + std::to_string(target_id));
        }
        const auto& li = *it;
        if (li.extent <= 0) {
            return;
        }

        const std::string prefix = "t.out." + std::to_string(out_level);
        auto blocks = createLoopBlocks(ctx, fn, prefix);
        blocks.preheader = builder.GetInsertBlock();

        auto* start = i32c(0);
        if (li.lower_bound_id >= 0) {
            auto* lb = indexValueForId(li.lower_bound_id);
            start = builder.CreateAdd(lb, i32c(static_cast<std::uint32_t>(std::max(0, li.lower_bound_offset))));
        }

        builder.CreateBr(blocks.header);
        builder.SetInsertPoint(blocks.header);
        auto* idx_phi = builder.CreatePHI(i32, 2, prefix + ".i");
        idx_phi->addIncoming(start, blocks.preheader);
        auto* cond = builder.CreateICmpULT(idx_phi, i32c(static_cast<std::uint32_t>(li.extent)));
        builder.CreateCondBr(cond, blocks.body, blocks.exit);

        builder.SetInsertPoint(blocks.body);
        llvm::Value* prev = nullptr;
        {
            const std::size_t id = static_cast<std::size_t>(li.id);
            prev = (id < idx_env.size()) ? idx_env[id] : nullptr;
            if (id >= idx_env.size()) idx_env.resize(id + 1u, nullptr);
            idx_env[id] = idx_phi;
        }

        self(self, op, out_level + 1u);

        {
            const std::size_t id = static_cast<std::size_t>(li.id);
            idx_env[id] = prev;
        }

        builder.CreateBr(blocks.latch);
        builder.SetInsertPoint(blocks.latch);
        auto* idx_next = builder.CreateAdd(idx_phi, i32c(1));

        auto* backedge = builder.CreateBr(blocks.header);
        if (options_.vectorize && li.vectorize) {
            auto* md = makeLoopMetadata(ctx, /*vectorize=*/true, li.vector_width, options_.enable_polly);
            backedge->setMetadata("llvm.loop", md);
        } else if (options_.enable_polly) {
            auto* md = makeLoopMetadata(ctx, /*vectorize=*/false, /*vector_width=*/1, /*polly=*/true);
            backedge->setMetadata("llvm.loop", md);
        }

        idx_phi->addIncoming(idx_next, blocks.latch);

        builder.SetInsertPoint(blocks.exit);
    };

    // Emit all ops in order (temporaries are written by ops).
    llvm::Value* result = f64c(0.0);
    const int num_ops = static_cast<int>(ir.program.ops.size());

    const auto addContribution = [&](const forms::tensor::LoopNestProgram::Contribution& c) {
        if (c.tensor_id < 0) {
            auto* v = eval_scalar(c.scalar);
            result = builder.CreateFAdd(result, v);
            return;
        }

        if (static_cast<std::size_t>(c.tensor_id) >= ir.program.tensors.size()) {
            throw std::runtime_error("LLVMTensorGen: contribution tensor id out of range");
        }

        const auto& t = ir.program.tensors[static_cast<std::size_t>(c.tensor_id)];
        if (t.rank != 0) {
            throw std::runtime_error("LLVMTensorGen: scalar output requires scalar contribution tensors");
        }

        auto* scale = eval_scalar(c.scalar);
        auto* tv = loadTensorElement(c.tensor_id, t);
        result = builder.CreateFAdd(result, builder.CreateFMul(scale, tv));
    };

    // Emit contributions that are available before any contraction ops.
    for (const auto& c : ir.program.contributions) {
        if (c.available_after_op < 0) {
            addContribution(c);
        }
    }

    // Emit ops and interleave contribution accumulation as soon as it is safe.
    for (int op_idx = 0; op_idx < num_ops; ++op_idx) {
        const auto& op = ir.program.ops[static_cast<std::size_t>(op_idx)];
        switch (op.kind) {
            case forms::tensor::ContractionOp::Kind::Contraction:
            case forms::tensor::ContractionOp::Kind::Reduction:
                emitOutLoops(emitOutLoops, op, 0u);
                break;
            case forms::tensor::ContractionOp::Kind::DeltaSubstitution:
                throw std::runtime_error("LLVMTensorGen: DeltaSubstitution op kind not implemented in codegen");
        }

        for (const auto& c : ir.program.contributions) {
            if (c.available_after_op == op_idx) {
                addContribution(c);
            }
        }
    }

    return result;
}

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

#endif
