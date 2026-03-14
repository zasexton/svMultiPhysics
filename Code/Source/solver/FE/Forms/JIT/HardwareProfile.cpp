/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/JIT/HardwareProfile.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>

namespace svmp {
namespace FE {
namespace forms {
namespace jit {

namespace {

/// Read an integer from a sysfs file (e.g. "32768\n" → 32768).
/// Returns 0 on any failure.
[[nodiscard]] std::uint32_t readSysfsU32(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) return 0;
    std::uint32_t v = 0;
    f >> v;
    return f.fail() ? 0u : v;
}

/// Parse a size string like "32K" or "8192K" or "8M" or plain "32768".
[[nodiscard]] std::uint32_t parseSizeString(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) return 0;
    std::string s;
    std::getline(f, s);
    if (s.empty()) return 0;

    // Trim whitespace
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
        s.pop_back();

    char suffix = 0;
    std::uint64_t val = 0;
    const char* end = s.c_str() + s.size();
    // Check for K/M suffix
    if (!s.empty() && (s.back() == 'K' || s.back() == 'M')) {
        suffix = s.back();
        s.pop_back();
    }

    try {
        val = std::stoull(s);
    } catch (...) {
        return 0;
    }

    if (suffix == 'K') val *= 1024;
    else if (suffix == 'M') val *= 1024 * 1024;

    return static_cast<std::uint32_t>(val);
}

/// Detect SIMD width from compiler-defined macros at runtime.
[[nodiscard]] std::uint32_t detectSimdWidth() noexcept
{
#if defined(__AVX512F__)
    return 64;
#elif defined(__AVX2__) || defined(__AVX__)
    return 32;
#else
    return 16; // SSE2 baseline
#endif
}

/// Read a single cache level from Linux sysfs.
[[nodiscard]] CacheLevel readCacheIndex(int cpu, int index)
{
    const auto base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu)
                    + "/cache/index" + std::to_string(index) + "/";
    CacheLevel cl;
    cl.size_bytes = parseSizeString(base + "size");
    cl.line_size = readSysfsU32(base + "coherency_line_size");
    cl.ways = readSysfsU32(base + "ways_of_associativity");
    if (cl.line_size == 0) cl.line_size = 64;
    return cl;
}

/// Read the cache type string ("Data", "Instruction", "Unified").
[[nodiscard]] std::string readCacheType(int cpu, int index)
{
    const auto path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu)
                    + "/cache/index" + std::to_string(index) + "/type";
    std::ifstream f(path);
    if (!f.is_open()) return {};
    std::string s;
    std::getline(f, s);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r'))
        s.pop_back();
    return s;
}

/// Read the cache level number (1, 2, 3).
[[nodiscard]] int readCacheLevel(int cpu, int index)
{
    const auto path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu)
                    + "/cache/index" + std::to_string(index) + "/level";
    return static_cast<int>(readSysfsU32(path));
}

/// Conservative default profile for when sysfs is unavailable.
[[nodiscard]] HardwareProfile defaultProfile()
{
    HardwareProfile hp;
    hp.l1d = {32 * 1024, 64, 8};
    hp.l1i = {32 * 1024, 64, 8};
    hp.l2  = {256 * 1024, 64, 4};
    hp.l3  = {8 * 1024 * 1024, 64, 16};
    hp.simd_width_bytes = detectSimdWidth();
    return hp;
}

} // anonymous namespace

HardwareProfile discoverHardwareProfile()
{
    // Allow env var override for testing/containers
    if (const char* env = std::getenv("SVMP_CACHE_PROFILE")) {
        // Format: "L1d:32768,L1i:32768,L2:262144,L3:8388608"
        HardwareProfile hp = defaultProfile();
        std::string s(env);
        auto parseField = [&](const char* key) -> std::uint32_t {
            auto pos = s.find(key);
            if (pos == std::string::npos) return 0;
            pos += std::strlen(key);
            if (pos >= s.size() || s[pos] != ':') return 0;
            return static_cast<std::uint32_t>(std::strtoul(s.c_str() + pos + 1, nullptr, 10));
        };
        if (auto v = parseField("L1d")) hp.l1d.size_bytes = v;
        if (auto v = parseField("L1i")) hp.l1i.size_bytes = v;
        if (auto v = parseField("L2"))  hp.l2.size_bytes = v;
        if (auto v = parseField("L3"))  hp.l3.size_bytes = v;
        return hp;
    }

    HardwareProfile hp = defaultProfile();

    // Enumerate /sys/devices/system/cpu/cpu0/cache/index{0,1,2,3,...}
    for (int index = 0; index < 8; ++index) {
        const auto level = readCacheLevel(0, index);
        if (level == 0) break; // no more indices

        const auto type = readCacheType(0, index);
        const auto cl = readCacheIndex(0, index);
        if (cl.size_bytes == 0) continue;

        if (level == 1) {
            if (type == "Data")        hp.l1d = cl;
            else if (type == "Instruction") hp.l1i = cl;
        } else if (level == 2) {
            if (type == "Unified" || type == "Data") hp.l2 = cl;
        } else if (level == 3) {
            hp.l3 = cl;
        }
    }

    hp.simd_width_bytes = detectSimdWidth();
    return hp;
}

const HardwareProfile& hardwareProfile()
{
    static const HardwareProfile instance = discoverHardwareProfile();
    return instance;
}

void BytesPerOpCalibration::recordSample(std::uint64_t object_bytes, std::uint64_t ir_ops) noexcept
{
    if (ir_ops == 0 || object_bytes == 0) return;
    total_bytes_.fetch_add(object_bytes, std::memory_order_relaxed);
    total_ops_.fetch_add(ir_ops, std::memory_order_relaxed);
    n_samples_.fetch_add(1u, std::memory_order_relaxed);
}

std::uint64_t BytesPerOpCalibration::calibratedBytesPerOp(std::uint64_t fallback) const noexcept
{
    const auto n = n_samples_.load(std::memory_order_relaxed);
    const auto ops = total_ops_.load(std::memory_order_relaxed);
    if (n >= kMinSamples && ops > 0) {
        return total_bytes_.load(std::memory_order_relaxed) / ops;
    }
    return fallback;
}

BytesPerOpCalibration& bytesPerOpCalibration()
{
    static BytesPerOpCalibration instance;
    return instance;
}

BytesPerOpCalibration& rawBytesPerOpCalibration()
{
    static BytesPerOpCalibration instance;
    return instance;
}

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp
