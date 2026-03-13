#ifndef SVMP_FE_FORMS_JIT_HARDWARE_PROFILE_H
#define SVMP_FE_FORMS_JIT_HARDWARE_PROFILE_H

/**
 * @file HardwareProfile.h
 * @brief Hardware cache hierarchy discovery for JIT code generation budgets
 *
 * Reads the local CPU cache layout (L1d, L1i, L2, L3) from Linux sysfs and
 * derives default budgets for:
 *   - QP shared-cache buffer sizing (from L1d stack budget)
 *   - Colocated module grouping (from L1i capacity)
 *   - Specialization, tiling, and unroll limits (from cache + SIMD width)
 *
 * Falls back to conservative defaults when sysfs is not available (non-Linux
 * platforms, containers with restricted /sys access, etc.).
 */

#include <atomic>
#include <cstdint>

namespace svmp {
namespace FE {
namespace forms {
namespace jit {

struct CacheLevel {
    std::uint32_t size_bytes{0};    // Total cache size in bytes
    std::uint32_t line_size{64};    // Cache line size in bytes
    std::uint32_t ways{0};          // Associativity
};

struct HardwareProfile {
    CacheLevel l1d{};               // L1 data cache (per-core)
    CacheLevel l1i{};               // L1 instruction cache (per-core)
    CacheLevel l2{};                // L2 cache (per-core or shared)
    CacheLevel l3{};                // L3 / LLC (package-wide)
    std::uint32_t simd_width_bytes{16};  // SIMD register width (SSE2=16, AVX2=32, AVX-512=64)

    // ----- Derived budgets -----

    /// Maximum QP cache entries (doubles) that fit in ~1/4 of L1d.
    /// Used to size the shared QP intermediate cache allocas.
    [[nodiscard]] std::uint32_t qpCacheBudgetDoubles() const noexcept
    {
        // Reserve ~1/4 of L1d for QP cache (rest for basis, solution, etc.)
        const auto quarter = l1d.size_bytes / 4u;
        return quarter / 8u; // sizeof(double) == 8
    }

    /// Per-pool byte budget for one cache pool (QP shared cache or
    /// cross-block CSE cache).  Each pool gets ~1/4 of L1d independently;
    /// the caller is responsible for allocating separate budget counters
    /// per pool.
    [[nodiscard]] std::uint32_t qpCacheBudgetBytes() const noexcept
    {
        return l1d.size_bytes / 4u;
    }

    /// Maximum .text bytes per colocated module group.
    /// Sized to fit within L1i to avoid instruction cache thrashing.
    [[nodiscard]] std::uint32_t colocationTextBudgetBytes() const noexcept
    {
        // Use ~3/4 of L1i — leave room for the assembly/caller code
        return (l1i.size_bytes * 3u) / 4u;
    }

    /// Per-helper .text budget for term-group splitting inside coupled kernels.
    /// Sized to fit comfortably within L1i (3/4 of L1i capacity).
    [[nodiscard]] std::uint32_t helperTextBudgetBytes() const noexcept
    {
        return (l1i.size_bytes * 3u) / 4u;
    }

    /// Maximum trip count for full loop unrolling.
    /// Derived from L1i budget and estimated bytes per unrolled iteration.
    [[nodiscard]] std::uint32_t maxUnrollTripCount() const noexcept
    {
        // Conservative: 16 on typical 32KB L1i, scale linearly
        const auto l1i_kb = l1i.size_bytes / 1024u;
        if (l1i_kb <= 16u) return 8u;
        if (l1i_kb <= 32u) return 16u;
        return 32u;
    }

    /// Default tile size for loop tiling (from L1d line count).
    [[nodiscard]] std::uint32_t defaultTileSize() const noexcept
    {
        // Tile to keep working set in ~1/2 of L1d
        const auto half_lines = l1d.size_bytes / (2u * l1d.line_size);
        if (half_lines <= 16u) return 16u;
        if (half_lines <= 32u) return 32u;
        return 64u;
    }

    /// Minimum aggregate cost × reuse to justify trial-only caching.
    /// Derived from L1d: on smaller caches the store/load overhead is
    /// proportionally higher, so the savings threshold is raised.
    /// Linear: 12 @16KB, 8 @32KB, clamped to [6, 16].
    [[nodiscard]] std::uint32_t trialOnlyMinSavings() const noexcept
    {
        const auto l1d_kb = l1d.size_bytes / 1024u;
        const auto raw = (l1d_kb >= 64u) ? 0u : (16u - l1d_kb / 4u);
        return (raw < 6u) ? 6u : (raw > 16u ? 16u : raw);
    }

    /// Minimum number of trial-only ops required to consider caching.
    /// Even if cost × reuse passes the threshold, fewer than this many
    /// distinct cacheable ops makes the overhead of the cache setup
    /// disproportionate.  Binary: 3 on tight caches (<24KB), else 2.
    [[nodiscard]] std::uint32_t trialOnlyMinOps() const noexcept
    {
        return (l1d.size_bytes / 1024u < 24u) ? 3u : 2u;
    }

    /// Minimum subtree cost for cross-block CSE caching.
    /// Ops below this cost are cheaper to recompute than store/load.
    /// Linear: 6 @16KB, 4 @32KB, clamped to [3, 8].
    [[nodiscard]] std::uint32_t crossBlockCostThreshold() const noexcept
    {
        const auto l1d_kb = l1d.size_bytes / 1024u;
        const auto raw = (l1d_kb >= 64u) ? 2u : (8u - l1d_kb / 8u);
        return (raw < 3u) ? 3u : (raw > 8u ? 8u : raw);
    }

    /// Default n_test DOF estimate when not specialized at codegen time.
    /// Used for trial-only caching reuse estimation.  Binary: 3 on tight
    /// caches (<24KB L1d), else 4.
    [[nodiscard]] std::uint32_t defaultTestDofEstimate() const noexcept
    {
        return (l1d.size_bytes / 1024u < 24u) ? 3u : 4u;
    }

    /// .text budget for DOF-loop unrolling suppression (~3× L1i).
    /// Fully-unrolled kernels exceeding this trigger loop-based DOF iteration.
    [[nodiscard]] std::uint32_t textBudgetBytes() const noexcept
    {
        return l1i.size_bytes * 3u;  // 96KB for 32KB L1i
    }

    /// Stable hash of the hardware profile fields that affect codegen decisions.
    /// Used to key JIT disk caches so that kernels compiled for one hardware
    /// profile are not reused on a different machine with different cache sizes.
    [[nodiscard]] std::uint64_t stableHash64() const noexcept
    {
        // FNV-1a
        std::uint64_t h = 14695981039346656037ULL;
        auto mix = [&](std::uint64_t v) { h ^= v; h *= 1099511628211ULL; };
        mix(l1d.size_bytes);
        mix(l1i.size_bytes);
        mix(l2.size_bytes);
        mix(l3.size_bytes);
        mix(simd_width_bytes);
        return h;
    }

    // ----- Codegen calibration constants -----
    // These are empirical values from LLVM codegen output, NOT derived from
    // the CPU cache hierarchy.  They depend on the compiler (LLVM 14-18),
    // the KernelIR op mix, and the unrolling strategy.  Centralized here
    // so that all consumers (JITSpecializationOptions, CoupledBlockKernel,
    // etc.) share the same values.

    /// Estimated bytes of x86-64 machine code per KernelIR op after LLVM
    /// codegen with full QP/DOF unrolling.  Calibrated from FMA-heavy FEM
    /// kernels (diffusion, elasticity, stabilized flow) on LLVM 14.
    /// This is the static fallback; use BytesPerOpCalibration for
    /// telemetry-driven per-process estimates from actual compilations.
    static constexpr std::uint64_t kBytesPerOp = 58;

    /// Conservative fallback KernelIR op count per FormIR term, used when
    /// actual lowering is unavailable (e.g. FormIR not compiled, lowering
    /// throws).  Based on stabilized FEM tangent terms (~125-186 ops
    /// post-optimize).
    static constexpr std::uint64_t kFallbackOpsPerTerm = 180;
};

/// Discover the hardware profile from the local system.
/// On Linux, reads /sys/devices/system/cpu/cpu0/cache/indexN/.
/// On other platforms or on failure, returns conservative defaults.
[[nodiscard]] HardwareProfile discoverHardwareProfile();

/// Return a cached singleton hardware profile (thread-safe, discovered once).
[[nodiscard]] const HardwareProfile& hardwareProfile();

/// Process-level telemetry-driven bytes-per-op calibration.
/// Thread-safe: all members are atomic (relaxed ordering — eventual
/// consistency is sufficient for a heuristic calibration value).
///
/// After each LLVM compilation, call recordSample(object_bytes, ir_op_count)
/// to update the running estimate.  calibratedBytesPerOp() returns the
/// measured value once enough samples exist, or the caller-provided fallback.
struct BytesPerOpCalibration {
    /// Record one compilation's object size and total KernelIR op count.
    void recordSample(std::uint64_t object_bytes, std::uint64_t ir_ops) noexcept;

    /// Return the best current estimate of bytes per KernelIR op.
    /// Uses measured telemetry if >= kMinSamples compilations have been
    /// observed; otherwise returns @p fallback.
    [[nodiscard]] std::uint64_t calibratedBytesPerOp(
        std::uint64_t fallback = HardwareProfile::kBytesPerOp) const noexcept;

    static constexpr unsigned kMinSamples = 2;  // need at least 2 to average

private:
    std::atomic<std::uint64_t> total_bytes_{0};
    std::atomic<std::uint64_t> total_ops_{0};
    std::atomic<unsigned> n_samples_{0};
};

/// Return a process-global BytesPerOpCalibration instance (thread-safe).
[[nodiscard]] BytesPerOpCalibration& bytesPerOpCalibration();

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_JIT_HARDWARE_PROFILE_H
