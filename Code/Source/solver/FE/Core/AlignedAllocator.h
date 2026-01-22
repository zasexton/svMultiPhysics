#ifndef SVMP_FE_CORE_ALIGNED_ALLOCATOR_H
#define SVMP_FE_CORE_ALIGNED_ALLOCATOR_H

/**
 * @file AlignedAllocator.h
 * @brief Simple C++ allocator for std::vector with explicit alignment.
 *
 * This is used to guarantee alignment of assembly/JIT-packed arrays (basis tables,
 * geometry, solution data) for efficient SIMD codegen.
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <type_traits>

namespace svmp {
namespace FE {

namespace detail {

[[nodiscard]] constexpr bool isPowerOfTwo(std::size_t x) noexcept
{
    return x != 0u && (x & (x - 1u)) == 0u;
}

} // namespace detail

template <class T, std::size_t Alignment>
struct AlignedAllocator {
    static_assert(Alignment >= alignof(T), "AlignedAllocator: Alignment must be >= alignof(T)");
    static_assert(detail::isPowerOfTwo(Alignment), "AlignedAllocator: Alignment must be power-of-two");

    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    constexpr AlignedAllocator() noexcept = default;

    template <class U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept
    {
    }

    [[nodiscard]] T* allocate(std::size_t n)
    {
        if (n == 0u) {
            return nullptr;
        }
        if (n > max_size()) {
            throw std::bad_array_new_length();
        }
        void* p = ::operator new(n * sizeof(T), std::align_val_t(Alignment));
        return static_cast<T*>(p);
    }

    void deallocate(T* p, std::size_t /*n*/) noexcept
    {
        ::operator delete(p, std::align_val_t(Alignment));
    }

    [[nodiscard]] static constexpr std::size_t max_size() noexcept
    {
        return std::numeric_limits<std::size_t>::max() / sizeof(T);
    }

    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
};

template <class T1, std::size_t A, class T2>
[[nodiscard]] constexpr bool operator==(const AlignedAllocator<T1, A>&, const AlignedAllocator<T2, A>&) noexcept
{
    return true;
}

template <class T1, std::size_t A, class T2>
[[nodiscard]] constexpr bool operator!=(const AlignedAllocator<T1, A>&, const AlignedAllocator<T2, A>&) noexcept
{
    return false;
}

} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CORE_ALIGNED_ALLOCATOR_H
