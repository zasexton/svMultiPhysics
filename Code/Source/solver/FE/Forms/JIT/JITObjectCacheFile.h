#ifndef SVMP_FE_FORMS_JIT_JIT_OBJECT_CACHE_FILE_H
#define SVMP_FE_FORMS_JIT_JIT_OBJECT_CACHE_FILE_H

/**
 * @file JITObjectCacheFile.h
 * @brief On-disk format and file helpers for the LLVM JIT object cache.
 *
 * The filesystem object cache may be shared by many processes on many hosts
 * (for example a cache directory on a parallel filesystem).  These helpers
 * keep that safe:
 *
 * - every cached object is wrapped in a small header that records the module
 *   id, the object size and a checksum of both, so a truncated, corrupted or
 *   misplaced object is rejected before it can be linked or executed;
 * - objects are published with write-to-unique-temporary-then-rename, where
 *   the temporary name contains the host name, the process id, a random
 *   component and a per-process counter.
 *
 * The helpers do not depend on LLVM so the format can be unit tested in any
 * build configuration.
 */

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace svmp {
namespace FE {
namespace forms {
namespace jit {
namespace objcache {

/// Environment variable that overrides the default object cache directory.
inline constexpr const char* kCacheDirectoryEnvVar = "SVMP_JIT_CACHE_DIR";

/// Subdirectory of the (LLVM-version-validated) cache root that holds objects
/// in the checked format below.  Raw objects written by older builds live in
/// the root itself and are neither read nor overwritten by this format.
inline constexpr const char* kFormatSubdirectory = "objects-v2";

/// File layout (all integers little-endian):
///   [0, 8)    magic "SVMPJOC2"
///   [8, 16)   module id length L
///   [16, 24)  object size N
///   [24, 32)  FNV-1a 64-bit checksum over (L, N, module id, object bytes)
///   [32, 32+L)        module id
///   [32+L, 32+L+N)    object bytes
inline constexpr std::size_t kHeaderSize = 32u;
inline constexpr std::size_t kModuleIdLengthOffset = 8u;
inline constexpr std::size_t kObjectSizeOffset = 16u;
inline constexpr std::size_t kChecksumOffset = 24u;

enum class FileStatus : std::uint8_t {
    Valid,
    TooShort,
    BadMagic,
    ModuleIdMismatch,
    SizeMismatch,
    ChecksumMismatch,
};

[[nodiscard]] const char* toString(FileStatus status) noexcept;

struct DecodedFile {
    FileStatus status{FileStatus::TooShort};
    /// View into the decoded file image; empty unless status == Valid.
    std::string_view object{};
};

/// Checksum stored in the header.
[[nodiscard]] std::uint64_t checksum(std::string_view module_id, std::string_view object_bytes) noexcept;

/// Wrap object bytes in the checked on-disk format.
[[nodiscard]] std::string encodeFile(std::string_view module_id, std::string_view object_bytes);

/// Validate a complete file image written by encodeFile() for @p module_id.
/// Any mismatch (magic, module id, sizes, checksum) yields a non-Valid status.
[[nodiscard]] DecodedFile decodeFile(std::string_view module_id, std::string_view file_bytes) noexcept;

/// Temporary path in the directory of @p final_path whose name is unique
/// across hosts, processes and threads: it contains the sanitized host name,
/// the process id, a random 64-bit component and a per-process counter.  The
/// file name length is bounded independently of the final name.
[[nodiscard]] std::filesystem::path uniqueTempPath(const std::filesystem::path& final_path);

/// Write @p bytes to a new unique temporary file (created exclusively) and
/// rename it onto @p final_path.  Returns false, leaving no temporary file
/// behind, when any step fails.
[[nodiscard]] bool writeFileAtomically(const std::filesystem::path& final_path,
                                       std::string_view bytes) noexcept;

/// Effective cache root: @p configured_directory when non-empty; otherwise,
/// when @p cache_kernels is true, $SVMP_JIT_CACHE_DIR when set and non-empty,
/// else $HOME/.cache/svMultiPhysics/jit_cache; otherwise empty (no disk cache).
[[nodiscard]] std::string resolveCacheDirectory(std::string_view configured_directory,
                                                bool cache_kernels);

} // namespace objcache
} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_JIT_JIT_OBJECT_CACHE_FILE_H
