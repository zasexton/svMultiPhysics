#ifndef SVMP_FE_FORMS_TENSOR_TENSOR_ALLOCATION_H
#define SVMP_FE_FORMS_TENSOR_TENSOR_ALLOCATION_H

/**
 * @file TensorAllocation.h
 * @brief Temporary allocation planning for loop-based tensor evaluation
 *
 * This module defines a simple allocation strategy for tensor temporaries used by
 * `LoopNestProgram` evaluation:
 * - small tensors are mapped to a reusable "stack" pool (fixed-size buffers),
 * - larger tensors are mapped to a per-evaluation "arena" buffer,
 * - temporaries are reused via a lightweight liveness analysis over program ops,
 * - all buffers are aligned for SIMD-friendly access.
 */

#include "Forms/Tensor/LoopStructure.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace tensor {

struct TensorAllocationOptions {
    // Small tensors (<= 3×3×3×3) are stack-allocated by default.
    std::size_t stack_max_entries{81};

    // Required alignment for allocated buffers (32 or 64 bytes recommended).
    std::size_t alignment_bytes{64};

    // If true, reuse temporary buffers when lifetimes do not overlap.
    bool enable_reuse{true};
};

enum class TensorAllocKind : std::uint8_t {
    Input,
    TemporaryStack,
    TemporaryArena,
    Special, // e.g., KroneckerDelta (procedural; no stored scalars)
};

struct TensorAllocation {
    int tensor_id{-1};
    TensorAllocKind kind{TensorAllocKind::Input};

    std::size_t entries{0};
    std::size_t bytes{0};
    std::size_t alignment{0};

    // Slot id in the corresponding pool (TemporaryStack/TemporaryArena).
    int slot{-1};

    // Byte offset within the arena (TemporaryArena only).
    std::size_t arena_offset{0};
};

struct TensorAllocationPlan {
    bool ok{true};
    std::string message{};

    std::size_t alignment_bytes{0};
    std::size_t stack_bytes{0};
    std::size_t arena_bytes{0};

    int stack_slots{0};
    int arena_slots{0};

    // Slot layout (byte offsets are relative to the start of the corresponding pool).
    std::vector<std::size_t> stack_slot_bytes{};
    std::vector<std::size_t> stack_slot_offsets{};
    std::vector<std::size_t> arena_slot_bytes{};
    std::vector<std::size_t> arena_slot_offsets{};

    std::vector<TensorAllocation> allocations{};
};

/**
 * @brief Concrete temporary storage for a planned allocation.
 *
 * This is an optional helper for non-JIT evaluation paths and for validating
 * that allocation plans are self-consistent. JIT codegen may choose to emit
 * stack allocas and arena slices directly instead.
 */
class TensorTempWorkspace {
public:
    struct ExternalBuffers {
        std::byte* stack{nullptr};
        std::size_t stack_bytes{0};
        std::byte* arena{nullptr};
        std::size_t arena_bytes{0};
    };

    void reset(const TensorAllocationPlan& plan, std::size_t batch_size = 1);
    void resetWithExternalBuffers(const TensorAllocationPlan& plan,
                                  std::size_t batch_size,
                                  const ExternalBuffers& buffers);

    [[nodiscard]] std::size_t batchSize() const noexcept { return batch_size_; }
    [[nodiscard]] std::size_t alignmentBytes() const noexcept { return alignment_bytes_; }

    [[nodiscard]] std::span<Real> spanForTensor(int tensor_id, std::size_t batch_index = 0);
    [[nodiscard]] std::span<const Real> spanForTensor(int tensor_id, std::size_t batch_index = 0) const;

private:
    struct AlignedBuffer {
        AlignedBuffer() = default;
        AlignedBuffer(const AlignedBuffer&) = delete;
        AlignedBuffer& operator=(const AlignedBuffer&) = delete;
        AlignedBuffer(AlignedBuffer&& other) noexcept;
        AlignedBuffer& operator=(AlignedBuffer&& other) noexcept;
        ~AlignedBuffer();

        void reset(std::size_t bytes, std::size_t alignment);
        void bindExternal(std::byte* data, std::size_t bytes, std::size_t alignment);
        void clear() noexcept;
        [[nodiscard]] std::byte* data() noexcept { return data_; }
        [[nodiscard]] const std::byte* data() const noexcept { return data_; }
        [[nodiscard]] std::size_t size() const noexcept { return bytes_; }
        [[nodiscard]] std::size_t alignment() const noexcept { return alignment_; }

        std::byte* data_{nullptr};
        std::size_t bytes_{0};
        std::size_t alignment_{0};
        bool owns_{false};
    };

    std::size_t alignment_bytes_{0};
    std::size_t batch_size_{1};
    std::size_t stack_stride_bytes_{0};
    std::size_t arena_stride_bytes_{0};

    std::vector<TensorAllocation> allocations_{};
    std::vector<std::size_t> stack_slot_offsets_{};
    std::vector<std::size_t> arena_slot_offsets_{};

    AlignedBuffer stack_{};
    AlignedBuffer arena_{};
};

/**
 * @brief Plan temporary storage for a loop-nest program.
 *
 * Notes:
 * - Inputs (TensorSpec::base non-null) are treated as externally-provided.
 * - Contributions are assumed to be accumulated as soon as they are available
 *   (`LoopNestProgram::Contribution::available_after_op`), enabling reuse of
 *   final temporaries across terms.
 */
[[nodiscard]] TensorAllocationPlan planTensorAllocation(const LoopNestProgram& program,
                                                        const TensorAllocationOptions& options = {});

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_TENSOR_TENSOR_ALLOCATION_H
