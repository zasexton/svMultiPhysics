/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_MATH_ALIGNEDALLOCATOR_H
#define SVMP_FE_MATH_ALIGNEDALLOCATOR_H

#include <cstddef>
#include <cstdlib>
#include <new>
#include <type_traits>

#ifdef _MSC_VER
#include <malloc.h>
#endif

namespace svmp {
namespace FE {
namespace math {
namespace simd {

/**
 * @brief Aligned allocator for SIMD-friendly containers.
 *
 * Provides aligned allocations suitable for aligned SIMD loads/stores. The
 * default alignment is 64 bytes to satisfy AVX512 aligned load requirements;
 * it also works for AVX/SSE targets.
 */
template<typename T, std::size_t Alignment = 64>
class AlignedAllocator {
public:
    using value_type = T;
    using is_always_equal = std::true_type;

    AlignedAllocator() noexcept = default;

    template<class U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n == 0) {
            return nullptr;
        }
        if (n > (static_cast<std::size_t>(-1) / sizeof(T))) {
            throw std::bad_array_new_length();
        }

        void* ptr = nullptr;
#ifdef _MSC_VER
        ptr = _aligned_malloc(n * sizeof(T), Alignment);
        if (!ptr) {
            throw std::bad_alloc();
        }
#else
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
#endif
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
#ifdef _MSC_VER
        _aligned_free(p);
#else
        std::free(p);
#endif
    }

    template<class U>
    struct rebind { using other = AlignedAllocator<U, Alignment>; };
};

template<class T1, class T2, std::size_t A1, std::size_t A2>
constexpr bool operator==(const AlignedAllocator<T1, A1>&,
                          const AlignedAllocator<T2, A2>&) noexcept {
    return A1 == A2;
}

template<class T1, class T2, std::size_t A1, std::size_t A2>
constexpr bool operator!=(const AlignedAllocator<T1, A1>&,
                          const AlignedAllocator<T2, A2>&) noexcept {
    return A1 != A2;
}

} // namespace simd
} // namespace math
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_MATH_ALIGNEDALLOCATOR_H
