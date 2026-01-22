/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/GlobalKernelStateProvider.h"

#include "Core/FEException.h"

#include <algorithm>
#include <cstring>
#include <new>

namespace svmp {
namespace FE {
namespace systems {

namespace {

[[nodiscard]] bool isPowerOfTwo(std::size_t x) noexcept
{
    return x != 0u && (x & (x - 1u)) == 0u;
}

[[nodiscard]] std::size_t alignUp(std::size_t value, std::size_t alignment) noexcept
{
    if (alignment == 0u) return value;
    const std::size_t mask = alignment - 1u;
    return (value + mask) & ~mask;
}

} // namespace

GlobalKernelStateProvider::AlignedBuffer::~AlignedBuffer()
{
    reset();
}

GlobalKernelStateProvider::AlignedBuffer::AlignedBuffer(AlignedBuffer&& other) noexcept
    : data(other.data), size_bytes(other.size_bytes), alignment(other.alignment)
{
    other.data = nullptr;
    other.size_bytes = 0u;
    other.alignment = 1u;
}

GlobalKernelStateProvider::AlignedBuffer&
GlobalKernelStateProvider::AlignedBuffer::operator=(AlignedBuffer&& other) noexcept
{
    if (this == &other) return *this;
    reset();
    data = other.data;
    size_bytes = other.size_bytes;
    alignment = other.alignment;
    other.data = nullptr;
    other.size_bytes = 0u;
    other.alignment = 1u;
    return *this;
}

void GlobalKernelStateProvider::AlignedBuffer::allocate(std::size_t size, std::size_t align)
{
    reset();
    FE_THROW_IF(size == 0u, InvalidArgumentException, "GlobalKernelStateProvider: buffer size must be > 0");
    FE_THROW_IF(align == 0u, InvalidArgumentException, "GlobalKernelStateProvider: alignment must be > 0");
    FE_THROW_IF(!isPowerOfTwo(align), InvalidArgumentException, "GlobalKernelStateProvider: alignment must be power-of-two");

    const std::size_t padded = alignUp(size, align);
    void* ptr = ::operator new(padded, std::align_val_t(align));
    data = static_cast<std::byte*>(ptr);
    size_bytes = padded;
    alignment = align;
    std::memset(data, 0, size_bytes);
}

void GlobalKernelStateProvider::AlignedBuffer::reset() noexcept
{
    if (data) {
        ::operator delete(data, std::align_val_t(alignment));
    }
    data = nullptr;
    size_bytes = 0u;
    alignment = 1u;
}

GlobalKernelStateProvider::GlobalKernelStateProvider(GlobalIndex num_cells,
                                                     std::vector<GlobalIndex> boundary_face_ids,
                                                     std::vector<GlobalIndex> interior_face_ids)
    : num_cells_(num_cells),
      boundary_face_ids_(std::move(boundary_face_ids)),
      interior_face_ids_(std::move(interior_face_ids))
{
    FE_THROW_IF(num_cells_ < 0, InvalidArgumentException, "GlobalKernelStateProvider: num_cells must be non-negative");

    std::sort(boundary_face_ids_.begin(), boundary_face_ids_.end());
    boundary_face_ids_.erase(std::unique(boundary_face_ids_.begin(), boundary_face_ids_.end()), boundary_face_ids_.end());
    for (std::size_t i = 0; i < boundary_face_ids_.size(); ++i) {
        boundary_face_index_.emplace(boundary_face_ids_[i], i);
    }

    std::sort(interior_face_ids_.begin(), interior_face_ids_.end());
    interior_face_ids_.erase(std::unique(interior_face_ids_.begin(), interior_face_ids_.end()), interior_face_ids_.end());
    for (std::size_t i = 0; i < interior_face_ids_.size(); ++i) {
        interior_face_index_.emplace(interior_face_ids_[i], i);
    }
}

GlobalKernelStateProvider::~GlobalKernelStateProvider() = default;
GlobalKernelStateProvider::GlobalKernelStateProvider(GlobalKernelStateProvider&&) noexcept = default;
GlobalKernelStateProvider& GlobalKernelStateProvider::operator=(GlobalKernelStateProvider&&) noexcept = default;

void GlobalKernelStateProvider::addKernel(const GlobalKernel& kernel, GlobalStateSpec spec)
{
    FE_THROW_IF(spec.empty(), InvalidArgumentException, "GlobalKernelStateProvider::addKernel: empty spec");
    FE_THROW_IF(spec.max_qpts <= 0, InvalidArgumentException, "GlobalKernelStateProvider::addKernel: max_qpts must be > 0");
    FE_THROW_IF(spec.alignment == 0u, InvalidArgumentException, "GlobalKernelStateProvider::addKernel: alignment must be > 0");
    FE_THROW_IF(!isPowerOfTwo(spec.alignment), InvalidArgumentException,
                "GlobalKernelStateProvider::addKernel: alignment must be power-of-two");

    KernelState state;
    state.bytes_per_qpt = spec.bytes_per_qpt;
    state.alignment = spec.alignment;
    state.stride_bytes = alignUp(state.bytes_per_qpt, state.alignment);

    auto checkExisting = [&]() {
        auto it = states_.find(&kernel);
        if (it == states_.end()) return false;
        FE_THROW_IF(it->second.bytes_per_qpt != state.bytes_per_qpt, InvalidArgumentException,
                    "GlobalKernelStateProvider: kernel state spec mismatch (bytes_per_qpt)");
        FE_THROW_IF(it->second.alignment != state.alignment, InvalidArgumentException,
                    "GlobalKernelStateProvider: kernel state spec mismatch (alignment)");
        return true;
    };
    if (checkExisting()) {
        return;
    }

    if (spec.domain == GlobalStateSpec::Domain::Cell) {
        state.max_cell_qpts = spec.max_qpts;
        state.cell_stride_bytes = static_cast<std::size_t>(state.max_cell_qpts) * state.stride_bytes;
        const std::size_t total = static_cast<std::size_t>(num_cells_) * state.cell_stride_bytes;
        if (total > 0u) {
            state.cell_old.allocate(total, state.alignment);
            state.cell_work.allocate(total, state.alignment);
        }
    } else if (spec.domain == GlobalStateSpec::Domain::BoundaryFace) {
        state.max_boundary_face_qpts = spec.max_qpts;
        state.boundary_face_stride_bytes = static_cast<std::size_t>(state.max_boundary_face_qpts) * state.stride_bytes;
        const std::size_t total = boundary_face_ids_.size() * state.boundary_face_stride_bytes;
        if (total > 0u) {
            state.boundary_old.allocate(total, state.alignment);
            state.boundary_work.allocate(total, state.alignment);
        }
    } else if (spec.domain == GlobalStateSpec::Domain::InteriorFace) {
        state.max_interior_face_qpts = spec.max_qpts;
        state.interior_face_stride_bytes = static_cast<std::size_t>(state.max_interior_face_qpts) * state.stride_bytes;
        const std::size_t total = interior_face_ids_.size() * state.interior_face_stride_bytes;
        if (total > 0u) {
            state.interior_old.allocate(total, state.alignment);
            state.interior_work.allocate(total, state.alignment);
        }
    } else {
        FE_THROW(InvalidArgumentException, "GlobalKernelStateProvider::addKernel: unsupported domain");
    }

    states_.emplace(&kernel, std::move(state));
}

assembly::MaterialStateView GlobalKernelStateProvider::getCellState(const GlobalKernel& kernel,
                                                                    GlobalIndex cell_id,
                                                                    LocalIndex num_qpts) const
{
    auto it = states_.find(&kernel);
    if (it == states_.end()) {
        return {};
    }
    const auto& st = it->second;
    if (st.max_cell_qpts <= 0) return {};

    FE_THROW_IF(cell_id < 0 || cell_id >= num_cells_, InvalidArgumentException,
                "GlobalKernelStateProvider: cell_id out of bounds");
    FE_THROW_IF(num_qpts < 0 || num_qpts > st.max_cell_qpts, InvalidArgumentException,
                "GlobalKernelStateProvider: requested num_qpts exceeds allocated max_qpts");

    auto* old_base = st.cell_old.data + static_cast<std::size_t>(cell_id) * st.cell_stride_bytes;
    auto* work_base = st.cell_work.data + static_cast<std::size_t>(cell_id) * st.cell_stride_bytes;
    return assembly::MaterialStateView{old_base, work_base, st.bytes_per_qpt, st.stride_bytes, st.alignment};
}

assembly::MaterialStateView GlobalKernelStateProvider::getBoundaryFaceState(const GlobalKernel& kernel,
                                                                            GlobalIndex face_id,
                                                                            LocalIndex num_qpts) const
{
    auto it = states_.find(&kernel);
    if (it == states_.end()) {
        return {};
    }
    const auto& st = it->second;
    if (st.max_boundary_face_qpts <= 0) return {};

    const auto idx_it = boundary_face_index_.find(face_id);
    if (idx_it == boundary_face_index_.end()) {
        return {};
    }

    FE_THROW_IF(num_qpts < 0 || num_qpts > st.max_boundary_face_qpts, InvalidArgumentException,
                "GlobalKernelStateProvider: requested boundary-face num_qpts exceeds allocated max_qpts");

    if (st.boundary_old.data == nullptr || st.boundary_work.data == nullptr) {
        return {};
    }

    const auto idx = idx_it->second;
    auto* old_base = st.boundary_old.data + idx * st.boundary_face_stride_bytes;
    auto* work_base = st.boundary_work.data + idx * st.boundary_face_stride_bytes;
    return assembly::MaterialStateView{old_base, work_base, st.bytes_per_qpt, st.stride_bytes, st.alignment};
}

assembly::MaterialStateView GlobalKernelStateProvider::getInteriorFaceState(const GlobalKernel& kernel,
                                                                            GlobalIndex face_id,
                                                                            LocalIndex num_qpts) const
{
    auto it = states_.find(&kernel);
    if (it == states_.end()) {
        return {};
    }
    const auto& st = it->second;
    if (st.max_interior_face_qpts <= 0) return {};

    const auto idx_it = interior_face_index_.find(face_id);
    if (idx_it == interior_face_index_.end()) {
        return {};
    }

    FE_THROW_IF(num_qpts < 0 || num_qpts > st.max_interior_face_qpts, InvalidArgumentException,
                "GlobalKernelStateProvider: requested interior-face num_qpts exceeds allocated max_qpts");

    if (st.interior_old.data == nullptr || st.interior_work.data == nullptr) {
        return {};
    }

    const auto idx = idx_it->second;
    auto* old_base = st.interior_old.data + idx * st.interior_face_stride_bytes;
    auto* work_base = st.interior_work.data + idx * st.interior_face_stride_bytes;
    return assembly::MaterialStateView{old_base, work_base, st.bytes_per_qpt, st.stride_bytes, st.alignment};
}

void GlobalKernelStateProvider::beginTimeStep()
{
    for (auto& kv : states_) {
        auto& st = kv.second;

        if (st.cell_old.data != nullptr && st.cell_work.data != nullptr) {
            FE_THROW_IF(st.cell_old.size_bytes != st.cell_work.size_bytes, FEException,
                        "GlobalKernelStateProvider: beginTimeStep cell size mismatch");
            std::memcpy(st.cell_work.data, st.cell_old.data, st.cell_old.size_bytes);
        }

        if (st.boundary_old.data != nullptr && st.boundary_work.data != nullptr) {
            FE_THROW_IF(st.boundary_old.size_bytes != st.boundary_work.size_bytes, FEException,
                        "GlobalKernelStateProvider: beginTimeStep boundary size mismatch");
            std::memcpy(st.boundary_work.data, st.boundary_old.data, st.boundary_old.size_bytes);
        }

        if (st.interior_old.data != nullptr && st.interior_work.data != nullptr) {
            FE_THROW_IF(st.interior_old.size_bytes != st.interior_work.size_bytes, FEException,
                        "GlobalKernelStateProvider: beginTimeStep interior size mismatch");
            std::memcpy(st.interior_work.data, st.interior_old.data, st.interior_old.size_bytes);
        }
    }
}

void GlobalKernelStateProvider::commitTimeStep()
{
    for (auto& kv : states_) {
        auto& st = kv.second;
        std::swap(st.cell_old, st.cell_work);
        std::swap(st.boundary_old, st.boundary_work);
        std::swap(st.interior_old, st.interior_work);
    }
}

} // namespace systems
} // namespace FE
} // namespace svmp
