#ifndef SVMP_FE_FORMS_JIT_JIT_CACHE_STATS_H
#define SVMP_FE_FORMS_JIT_JIT_CACHE_STATS_H

#include <cstdint>

namespace svmp {
namespace FE {
namespace forms {
namespace jit {

struct JITKernelCacheStats {
    std::uint64_t hits{0};
    std::uint64_t misses{0};
    std::uint64_t engine_symbol_hits{0};
    std::uint64_t stores{0};
    std::uint64_t evictions{0};
    std::uint64_t size{0};
};

struct JITObjectCacheStats {
    std::uint64_t notify_compiled{0};
    std::uint64_t get_calls{0};
    std::uint64_t mem_hits{0};
    std::uint64_t disk_hits{0};
    std::uint64_t misses{0};
    std::uint64_t bytes_written{0};
    std::uint64_t bytes_read{0};
    std::uint64_t in_memory_entries{0};
};

struct JITCacheStats {
    JITKernelCacheStats kernel{};
    JITObjectCacheStats object{};
};

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_JIT_JIT_CACHE_STATS_H

