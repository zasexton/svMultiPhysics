/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/Tensor/TensorAllocation.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <unordered_map>

namespace svmp {
namespace FE {
namespace forms {
namespace tensor {

namespace {

[[nodiscard]] bool isPowerOfTwo(std::size_t v) noexcept
{
    return v != 0u && (v & (v - 1u)) == 0u;
}

[[nodiscard]] std::size_t alignUp(std::size_t value, std::size_t alignment) noexcept
{
    if (alignment == 0u) return value;
    return (value + alignment - 1u) & ~(alignment - 1u);
}

struct TempLifetime {
    int tensor_id{-1};
    int start{-1};
    int end{-1};
    std::size_t bytes{0};
    std::size_t alignment{0};
    bool use_stack{true};
};

struct Slot {
    std::size_t capacity{0};
    std::size_t offset{0}; // arena only
    int end{-1};           // last op index using this slot
};

[[nodiscard]] int bestFitSlot(const std::vector<Slot>& slots,
                              const std::vector<int>& free_list,
                              std::size_t bytes)
{
    int best = -1;
    std::size_t best_cap = 0;
    for (const int s : free_list) {
        if (s < 0 || static_cast<std::size_t>(s) >= slots.size()) continue;
        const auto cap = slots[static_cast<std::size_t>(s)].capacity;
        if (cap < bytes) continue;
        if (best < 0 || cap < best_cap) {
            best = s;
            best_cap = cap;
        }
    }
    return best;
}

} // namespace

TensorAllocationPlan planTensorAllocation(const LoopNestProgram& program,
                                          const TensorAllocationOptions& options)
{
    TensorAllocationPlan plan;

    if (!program.ok) {
        plan.ok = false;
        plan.message = "planTensorAllocation: program is invalid";
        return plan;
    }

    if (!isPowerOfTwo(options.alignment_bytes) || options.alignment_bytes < 32u) {
        plan.ok = false;
        plan.message = "planTensorAllocation: alignment_bytes must be power-of-two and >= 32";
        return plan;
    }

    plan.alignment_bytes = options.alignment_bytes;
    plan.allocations.resize(program.tensors.size());

    const int num_ops = static_cast<int>(program.ops.size());
    const int end_of_program = num_ops;

    std::vector<int> def_index(program.tensors.size(), -1);
    std::vector<int> last_use(program.tensors.size(), -1);

    for (int op_idx = 0; op_idx < num_ops; ++op_idx) {
        const auto& op = program.ops[static_cast<std::size_t>(op_idx)];
        if (op.out >= 0 && static_cast<std::size_t>(op.out) < program.tensors.size()) {
            def_index[static_cast<std::size_t>(op.out)] = op_idx;
        }
        if (op.lhs >= 0 && static_cast<std::size_t>(op.lhs) < program.tensors.size()) {
            last_use[static_cast<std::size_t>(op.lhs)] = std::max(last_use[static_cast<std::size_t>(op.lhs)], op_idx);
        }
        if (op.rhs >= 0 && static_cast<std::size_t>(op.rhs) < program.tensors.size()) {
            last_use[static_cast<std::size_t>(op.rhs)] = std::max(last_use[static_cast<std::size_t>(op.rhs)], op_idx);
        }
    }

    // Treat contribution tensors as consumed as soon as they are available.
    for (const auto& c : program.contributions) {
        if (c.tensor_id < 0) continue;
        if (static_cast<std::size_t>(c.tensor_id) >= program.tensors.size()) continue;
        int use_at = c.available_after_op;
        if (use_at < 0) use_at = 0;
        use_at = std::min(use_at, end_of_program);
        last_use[static_cast<std::size_t>(c.tensor_id)] = std::max(last_use[static_cast<std::size_t>(c.tensor_id)], use_at);
    }

    // Ensure defs are at least live through their defining op.
    for (std::size_t t = 0; t < program.tensors.size(); ++t) {
        if (def_index[t] >= 0) {
            last_use[t] = std::max(last_use[t], def_index[t]);
        }
    }

    std::vector<TempLifetime> temps;
    temps.reserve(program.tensors.size());

    for (std::size_t tid = 0; tid < program.tensors.size(); ++tid) {
        const auto& spec = program.tensors[tid];

        TensorAllocation a;
        a.tensor_id = static_cast<int>(tid);
        a.alignment = options.alignment_bytes;

        if (spec.storage == TensorStorageKind::KroneckerDelta) {
            a.kind = TensorAllocKind::Special;
            plan.allocations[tid] = a;
            continue;
        }

        const bool is_input = spec.base.isValid() && spec.base.node() != nullptr;
        a.entries = (spec.rank == 0) ? 1u : spec.size;
        a.bytes = a.entries * sizeof(Real);

        if (is_input) {
            a.kind = TensorAllocKind::Input;
            plan.allocations[tid] = a;
            continue;
        }

        // Temporary.
        const bool use_stack = a.entries <= options.stack_max_entries;
        const std::size_t bytes_aligned = alignUp(a.bytes, options.alignment_bytes);

        TempLifetime tl;
        tl.tensor_id = static_cast<int>(tid);
        tl.start = def_index[tid] >= 0 ? def_index[tid] : 0;
        tl.end = last_use[tid] >= 0 ? last_use[tid] : end_of_program;
        tl.bytes = bytes_aligned;
        tl.alignment = options.alignment_bytes;
        tl.use_stack = use_stack;
        temps.push_back(tl);

        a.kind = use_stack ? TensorAllocKind::TemporaryStack : TensorAllocKind::TemporaryArena;
        a.bytes = bytes_aligned;
        plan.allocations[tid] = a;
    }

    if (temps.empty()) {
        plan.stack_slots = 0;
        plan.arena_slots = 0;
        plan.stack_bytes = 0u;
        plan.arena_bytes = 0u;
        return plan;
    }

    std::sort(temps.begin(), temps.end(),
              [&](const TempLifetime& a, const TempLifetime& b) {
                  if (a.start != b.start) return a.start < b.start;
                  if (a.end != b.end) return a.end < b.end;
                  return a.tensor_id < b.tensor_id;
              });

    std::vector<Slot> stack_slots;
    std::vector<Slot> arena_slots;
    std::vector<int> stack_active;
    std::vector<int> arena_active;
    std::vector<int> stack_free;
    std::vector<int> arena_free;

    std::size_t arena_cursor = 0u;

    const auto expire = [&](int start,
                            std::vector<Slot>& slots,
                            std::vector<int>& active,
                            std::vector<int>& free_list) -> void {
        for (auto it = active.begin(); it != active.end();) {
            const int s = *it;
            if (s < 0 || static_cast<std::size_t>(s) >= slots.size()) {
                it = active.erase(it);
                continue;
            }
            if (slots[static_cast<std::size_t>(s)].end < start) {
                free_list.push_back(s);
                it = active.erase(it);
            } else {
                ++it;
            }
        }
    };

    for (const auto& t : temps) {
        if (!options.enable_reuse) {
            // Disable reuse by forcing all existing active slots to appear live.
        }

        if (t.use_stack) {
            expire(t.start, stack_slots, stack_active, stack_free);
            int slot_id = -1;
            if (options.enable_reuse) {
                slot_id = bestFitSlot(stack_slots, stack_free, t.bytes);
            }
            if (slot_id >= 0) {
                stack_free.erase(std::remove(stack_free.begin(), stack_free.end(), slot_id), stack_free.end());
            } else {
                slot_id = static_cast<int>(stack_slots.size());
                Slot s;
                s.capacity = t.bytes;
                s.offset = 0u;
                s.end = t.end;
                stack_slots.push_back(s);
            }
            stack_slots[static_cast<std::size_t>(slot_id)].end = std::max(stack_slots[static_cast<std::size_t>(slot_id)].end, t.end);
            stack_active.push_back(slot_id);

            auto& alloc = plan.allocations[static_cast<std::size_t>(t.tensor_id)];
            alloc.slot = slot_id;
            alloc.arena_offset = 0u;
        } else {
            expire(t.start, arena_slots, arena_active, arena_free);
            int slot_id = -1;
            if (options.enable_reuse) {
                slot_id = bestFitSlot(arena_slots, arena_free, t.bytes);
            }
            if (slot_id >= 0) {
                arena_free.erase(std::remove(arena_free.begin(), arena_free.end(), slot_id), arena_free.end());
            } else {
                slot_id = static_cast<int>(arena_slots.size());
                Slot s;
                s.capacity = t.bytes;
                s.offset = alignUp(arena_cursor, options.alignment_bytes);
                s.end = t.end;
                arena_cursor = s.offset + s.capacity;
                arena_slots.push_back(s);
            }
            arena_slots[static_cast<std::size_t>(slot_id)].end = std::max(arena_slots[static_cast<std::size_t>(slot_id)].end, t.end);
            arena_active.push_back(slot_id);

            auto& alloc = plan.allocations[static_cast<std::size_t>(t.tensor_id)];
            alloc.slot = slot_id;
            alloc.arena_offset = arena_slots[static_cast<std::size_t>(slot_id)].offset;
        }
    }

    plan.stack_slots = static_cast<int>(stack_slots.size());
    plan.arena_slots = static_cast<int>(arena_slots.size());

    // Compute stack slot offsets in a packed, aligned layout.
    plan.stack_slot_bytes.resize(stack_slots.size(), 0u);
    plan.stack_slot_offsets.resize(stack_slots.size(), 0u);
    std::size_t stack_cursor = 0u;
    for (std::size_t i = 0; i < stack_slots.size(); ++i) {
        plan.stack_slot_bytes[i] = stack_slots[i].capacity;
        stack_cursor = alignUp(stack_cursor, options.alignment_bytes);
        plan.stack_slot_offsets[i] = stack_cursor;
        stack_cursor += stack_slots[i].capacity;
    }
    plan.stack_bytes = alignUp(stack_cursor, options.alignment_bytes);

    plan.arena_slot_bytes.resize(arena_slots.size(), 0u);
    plan.arena_slot_offsets.resize(arena_slots.size(), 0u);
    for (std::size_t i = 0; i < arena_slots.size(); ++i) {
        plan.arena_slot_bytes[i] = arena_slots[i].capacity;
        plan.arena_slot_offsets[i] = arena_slots[i].offset;
    }
    plan.arena_bytes = alignUp(arena_cursor, options.alignment_bytes);

    return plan;
}

// ============================================================================
// TensorTempWorkspace
// ============================================================================

TensorTempWorkspace::AlignedBuffer::AlignedBuffer(AlignedBuffer&& other) noexcept
{
    data_ = other.data_;
    bytes_ = other.bytes_;
    alignment_ = other.alignment_;
    owns_ = other.owns_;
    other.data_ = nullptr;
    other.bytes_ = 0u;
    other.alignment_ = 0u;
    other.owns_ = false;
}

TensorTempWorkspace::AlignedBuffer& TensorTempWorkspace::AlignedBuffer::operator=(AlignedBuffer&& other) noexcept
{
    if (this == &other) return *this;
    clear();
    data_ = other.data_;
    bytes_ = other.bytes_;
    alignment_ = other.alignment_;
    owns_ = other.owns_;
    other.data_ = nullptr;
    other.bytes_ = 0u;
    other.alignment_ = 0u;
    other.owns_ = false;
    return *this;
}

TensorTempWorkspace::AlignedBuffer::~AlignedBuffer()
{
    clear();
}

void TensorTempWorkspace::AlignedBuffer::reset(std::size_t bytes, std::size_t alignment)
{
    clear();
    if (bytes == 0u) {
        return;
    }
    data_ = static_cast<std::byte*>(::operator new(bytes, std::align_val_t(alignment)));
    bytes_ = bytes;
    alignment_ = alignment;
    owns_ = true;
}

void TensorTempWorkspace::AlignedBuffer::bindExternal(std::byte* data, std::size_t bytes, std::size_t alignment)
{
    clear();
    data_ = data;
    bytes_ = bytes;
    alignment_ = alignment;
    owns_ = false;
}

void TensorTempWorkspace::AlignedBuffer::clear() noexcept
{
    if (data_ != nullptr && owns_) {
        ::operator delete(data_, std::align_val_t(alignment_));
    }
    data_ = nullptr;
    bytes_ = 0u;
    alignment_ = 0u;
    owns_ = false;
}

void TensorTempWorkspace::reset(const TensorAllocationPlan& plan, std::size_t batch_size)
{
    if (!plan.ok) {
        throw std::invalid_argument("TensorTempWorkspace::reset: plan is invalid");
    }
    if (!isPowerOfTwo(plan.alignment_bytes) || plan.alignment_bytes < 32u) {
        throw std::invalid_argument("TensorTempWorkspace::reset: plan alignment is invalid");
    }
    if (batch_size == 0u) {
        throw std::invalid_argument("TensorTempWorkspace::reset: batch_size must be positive");
    }

    alignment_bytes_ = plan.alignment_bytes;
    batch_size_ = batch_size;
    stack_stride_bytes_ = plan.stack_bytes;
    arena_stride_bytes_ = plan.arena_bytes;

    allocations_ = plan.allocations;
    stack_slot_offsets_ = plan.stack_slot_offsets;
    arena_slot_offsets_ = plan.arena_slot_offsets;

    const std::size_t stack_total = stack_stride_bytes_ * batch_size_;
    const std::size_t arena_total = arena_stride_bytes_ * batch_size_;
    stack_.reset(stack_total, alignment_bytes_);
    arena_.reset(arena_total, alignment_bytes_);
}

void TensorTempWorkspace::resetWithExternalBuffers(const TensorAllocationPlan& plan,
                                                   std::size_t batch_size,
                                                   const ExternalBuffers& buffers)
{
    if (!plan.ok) {
        throw std::invalid_argument("TensorTempWorkspace::resetWithExternalBuffers: plan is invalid");
    }
    if (!isPowerOfTwo(plan.alignment_bytes) || plan.alignment_bytes < 32u) {
        throw std::invalid_argument("TensorTempWorkspace::resetWithExternalBuffers: plan alignment is invalid");
    }
    if (batch_size == 0u) {
        throw std::invalid_argument("TensorTempWorkspace::resetWithExternalBuffers: batch_size must be positive");
    }

    alignment_bytes_ = plan.alignment_bytes;
    batch_size_ = batch_size;
    stack_stride_bytes_ = plan.stack_bytes;
    arena_stride_bytes_ = plan.arena_bytes;

    allocations_ = plan.allocations;
    stack_slot_offsets_ = plan.stack_slot_offsets;
    arena_slot_offsets_ = plan.arena_slot_offsets;

    const std::size_t stack_total = stack_stride_bytes_ * batch_size_;
    const std::size_t arena_total = arena_stride_bytes_ * batch_size_;

    if (stack_total > 0u && buffers.stack != nullptr) {
        if ((reinterpret_cast<std::uintptr_t>(buffers.stack) % alignment_bytes_) != 0u) {
            throw std::invalid_argument("TensorTempWorkspace::resetWithExternalBuffers: external stack buffer is misaligned");
        }
        if (buffers.stack_bytes < stack_total) {
            throw std::invalid_argument("TensorTempWorkspace::resetWithExternalBuffers: external stack buffer too small");
        }
        stack_.bindExternal(buffers.stack, stack_total, alignment_bytes_);
    } else {
        stack_.reset(stack_total, alignment_bytes_);
    }

    if (arena_total > 0u && buffers.arena != nullptr) {
        if ((reinterpret_cast<std::uintptr_t>(buffers.arena) % alignment_bytes_) != 0u) {
            throw std::invalid_argument("TensorTempWorkspace::resetWithExternalBuffers: external arena buffer is misaligned");
        }
        if (buffers.arena_bytes < arena_total) {
            throw std::invalid_argument("TensorTempWorkspace::resetWithExternalBuffers: external arena buffer too small");
        }
        arena_.bindExternal(buffers.arena, arena_total, alignment_bytes_);
    } else {
        arena_.reset(arena_total, alignment_bytes_);
    }
}

[[nodiscard]] std::span<Real> TensorTempWorkspace::spanForTensor(int tensor_id, std::size_t batch_index)
{
    if (tensor_id < 0 || static_cast<std::size_t>(tensor_id) >= allocations_.size()) {
        return {};
    }
    if (batch_index >= batch_size_) {
        return {};
    }

    const auto& a = allocations_[static_cast<std::size_t>(tensor_id)];
    if (a.kind != TensorAllocKind::TemporaryStack && a.kind != TensorAllocKind::TemporaryArena) {
        return {};
    }
    const std::size_t entries = a.entries;
    if (entries == 0u) {
        return {};
    }

    if (a.kind == TensorAllocKind::TemporaryStack) {
        if (a.slot < 0 || static_cast<std::size_t>(a.slot) >= stack_slot_offsets_.size()) {
            return {};
        }
        const std::size_t base_off = stack_stride_bytes_ * batch_index + stack_slot_offsets_[static_cast<std::size_t>(a.slot)];
        if (base_off + entries * sizeof(Real) > stack_.size()) {
            return {};
        }
        auto* ptr = reinterpret_cast<Real*>(stack_.data() + base_off);
        return std::span<Real>(ptr, entries);
    }

    // TemporaryArena
    if (a.slot < 0 || static_cast<std::size_t>(a.slot) >= arena_slot_offsets_.size()) {
        return {};
    }
    const std::size_t base_off = arena_stride_bytes_ * batch_index + arena_slot_offsets_[static_cast<std::size_t>(a.slot)];
    if (base_off + entries * sizeof(Real) > arena_.size()) {
        return {};
    }
    auto* ptr = reinterpret_cast<Real*>(arena_.data() + base_off);
    return std::span<Real>(ptr, entries);
}

[[nodiscard]] std::span<const Real> TensorTempWorkspace::spanForTensor(int tensor_id, std::size_t batch_index) const
{
    auto* self = const_cast<TensorTempWorkspace*>(this);
    const auto s = self->spanForTensor(tensor_id, batch_index);
    return std::span<const Real>(s.data(), s.size());
}

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp
