/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/JIT/JITObjectCacheFile.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <random>
#include <system_error>

#if defined(_WIN32)
#include <fstream>
#include <process.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace svmp {
namespace FE {
namespace forms {
namespace jit {
namespace objcache {

namespace {

constexpr std::string_view kMagic{"SVMPJOC2", 8u};
constexpr std::uint64_t kFNVOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFNVPrime = 1099511628211ULL;

// Bound temporary-name components so the name stays below NAME_MAX (255)
// regardless of the length of the final object name.
constexpr std::size_t kMaxTempPrefixLength = 96u;
constexpr std::size_t kMaxHostTokenLength = 64u;

void fnvMix(std::uint64_t& h, std::string_view bytes) noexcept
{
    for (const char ch : bytes) {
        h ^= static_cast<std::uint64_t>(static_cast<unsigned char>(ch));
        h *= kFNVPrime;
    }
}

void appendLE64(std::string& out, std::uint64_t value)
{
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<char>(static_cast<unsigned char>(value & 0xFFu)));
        value >>= 8;
    }
}

[[nodiscard]] std::uint64_t readLE64(std::string_view bytes, std::size_t offset) noexcept
{
    std::uint64_t value = 0u;
    for (std::size_t i = 0; i < 8u; ++i) {
        value |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[offset + i])) << (8u * i);
    }
    return value;
}

[[nodiscard]] std::string sanitizeToken(std::string_view s, std::size_t max_length)
{
    std::string out;
    out.reserve(std::min(s.size(), max_length));
    for (const char ch : s) {
        if (out.size() >= max_length) {
            break;
        }
        const bool ok =
            (ch >= 'a' && ch <= 'z') ||
            (ch >= 'A' && ch <= 'Z') ||
            (ch >= '0' && ch <= '9') ||
            (ch == '_' || ch == '-' || ch == '.');
        out.push_back(ok ? ch : '_');
    }
    return out;
}

[[nodiscard]] const std::string& hostToken()
{
    static const std::string token = []() {
        std::string host;
#if defined(_WIN32)
        if (const char* name = std::getenv("COMPUTERNAME")) {
            host = name;
        }
#else
        char buffer[256] = {};
        if (::gethostname(buffer, sizeof(buffer) - 1u) == 0) {
            host = buffer;
        }
#endif
        std::string out = sanitizeToken(host, kMaxHostTokenLength);
        return out.empty() ? std::string("unknown-host") : out;
    }();
    return token;
}

[[nodiscard]] long long processId() noexcept
{
#if defined(_WIN32)
    return static_cast<long long>(::_getpid());
#else
    return static_cast<long long>(::getpid());
#endif
}

[[nodiscard]] std::uint64_t splitMix64(std::uint64_t x) noexcept
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

[[nodiscard]] std::uint64_t processRandomSeed() noexcept
{
    static const std::uint64_t seed = []() noexcept {
        std::uint64_t s = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        s ^= static_cast<std::uint64_t>(
                 std::chrono::system_clock::now().time_since_epoch().count())
             << 1;
        s ^= static_cast<std::uint64_t>(processId()) << 40;
        try {
            std::random_device device;
            s ^= (static_cast<std::uint64_t>(device()) << 32) ^ static_cast<std::uint64_t>(device());
        } catch (...) {
            // Clock and process id still make the seed process specific.
        }
        return splitMix64(s);
    }();
    return seed;
}

[[nodiscard]] std::string hex16(std::uint64_t value)
{
    static constexpr char kDigits[] = "0123456789abcdef";
    std::string out(16u, '0');
    for (std::size_t i = 16u; i-- > 0u;) {
        out[i] = kDigits[value & 0xFu];
        value >>= 4;
    }
    return out;
}

#if !defined(_WIN32)
[[nodiscard]] bool writeAll(int fd, const char* data, std::size_t size) noexcept
{
    while (size > 0u) {
        const ssize_t written = ::write(fd, data, size);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (written == 0) {
            return false;
        }
        data += written;
        size -= static_cast<std::size_t>(written);
    }
    return true;
}
#endif

} // namespace

const char* toString(FileStatus status) noexcept
{
    switch (status) {
        case FileStatus::Valid:
            return "valid";
        case FileStatus::TooShort:
            return "shorter than header";
        case FileStatus::BadMagic:
            return "unrecognized header";
        case FileStatus::ModuleIdMismatch:
            return "module id mismatch";
        case FileStatus::SizeMismatch:
            return "size mismatch: truncated or extended";
        case FileStatus::ChecksumMismatch:
            return "checksum mismatch";
    }
    return "unknown";
}

std::uint64_t checksum(std::string_view module_id, std::string_view object_bytes) noexcept
{
    std::uint64_t h = kFNVOffset;
    std::uint64_t lengths[2] = {static_cast<std::uint64_t>(module_id.size()),
                                static_cast<std::uint64_t>(object_bytes.size())};
    for (std::uint64_t length : lengths) {
        for (int i = 0; i < 8; ++i) {
            h ^= length & 0xFFu;
            h *= kFNVPrime;
            length >>= 8;
        }
    }
    fnvMix(h, module_id);
    fnvMix(h, object_bytes);
    return h;
}

std::string encodeFile(std::string_view module_id, std::string_view object_bytes)
{
    std::string out;
    out.reserve(kHeaderSize + module_id.size() + object_bytes.size());
    out.append(kMagic);
    appendLE64(out, static_cast<std::uint64_t>(module_id.size()));
    appendLE64(out, static_cast<std::uint64_t>(object_bytes.size()));
    appendLE64(out, checksum(module_id, object_bytes));
    out.append(module_id);
    out.append(object_bytes);
    return out;
}

DecodedFile decodeFile(std::string_view module_id, std::string_view file_bytes) noexcept
{
    DecodedFile result;
    if (file_bytes.size() < kHeaderSize) {
        result.status = FileStatus::TooShort;
        return result;
    }
    if (file_bytes.substr(0u, kMagic.size()) != kMagic) {
        result.status = FileStatus::BadMagic;
        return result;
    }

    const std::uint64_t id_length = readLE64(file_bytes, kModuleIdLengthOffset);
    const std::uint64_t object_size = readLE64(file_bytes, kObjectSizeOffset);
    const std::uint64_t stored_checksum = readLE64(file_bytes, kChecksumOffset);

    if (id_length != static_cast<std::uint64_t>(module_id.size())) {
        result.status = FileStatus::ModuleIdMismatch;
        return result;
    }

    const auto body_size = static_cast<std::uint64_t>(file_bytes.size() - kHeaderSize);
    if (id_length > body_size || object_size != body_size - id_length || object_size == 0u) {
        result.status = FileStatus::SizeMismatch;
        return result;
    }

    if (file_bytes.substr(kHeaderSize, module_id.size()) != module_id) {
        result.status = FileStatus::ModuleIdMismatch;
        return result;
    }

    const std::string_view object =
        file_bytes.substr(kHeaderSize + module_id.size(), static_cast<std::size_t>(object_size));
    if (checksum(module_id, object) != stored_checksum) {
        result.status = FileStatus::ChecksumMismatch;
        return result;
    }

    result.status = FileStatus::Valid;
    result.object = object;
    return result;
}

std::filesystem::path uniqueTempPath(const std::filesystem::path& final_path)
{
    static std::atomic<std::uint64_t> counter{0u};
    const std::uint64_t sequence = counter.fetch_add(1u, std::memory_order_relaxed);
    const std::uint64_t random_part = splitMix64(processRandomSeed() + sequence * 0x9E3779B97F4A7C15ULL);

    std::string name = sanitizeToken(final_path.filename().string(), kMaxTempPrefixLength);
    name += ".tmp.";
    name += hostToken();
    name += '.';
    name += std::to_string(processId());
    name += '.';
    name += hex16(random_part);
    name += '.';
    name += std::to_string(sequence);
    return final_path.parent_path() / name;
}

bool writeFileAtomically(const std::filesystem::path& final_path, std::string_view bytes) noexcept
{
    std::error_code ec;
    std::filesystem::path tmp_path;
    try {
        tmp_path = uniqueTempPath(final_path);

#if defined(_WIN32)
        {
            std::ofstream os(tmp_path, std::ios::binary | std::ios::trunc);
            if (!os.good()) {
                return false;
            }
            os.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
            os.close();
            if (!os) {
                std::filesystem::remove(tmp_path, ec);
                return false;
            }
        }
#else
        // O_EXCL: never write into a file some other process created.
        const int fd = ::open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
        if (fd < 0) {
            return false;
        }
        const bool wrote = writeAll(fd, bytes.data(), bytes.size());
        const bool closed = (::close(fd) == 0);
        if (!wrote || !closed) {
            std::filesystem::remove(tmp_path, ec);
            return false;
        }
#endif

        std::filesystem::rename(tmp_path, final_path, ec);
        if (ec) {
            std::filesystem::remove(tmp_path, ec);
            return false;
        }
        return true;
    } catch (...) {
        if (!tmp_path.empty()) {
            std::filesystem::remove(tmp_path, ec);
        }
        return false;
    }
}

std::string resolveCacheDirectory(std::string_view configured_directory, bool cache_kernels)
{
    if (!configured_directory.empty()) {
        return std::string(configured_directory);
    }
    if (!cache_kernels) {
        return {};
    }
    if (const char* env_dir = std::getenv(kCacheDirectoryEnvVar); env_dir != nullptr && *env_dir != '\0') {
        return std::string(env_dir);
    }
    if (const char* home = std::getenv("HOME"); home != nullptr && *home != '\0') {
        return std::string(home) + "/.cache/svMultiPhysics/jit_cache";
    }
    return {};
}

} // namespace objcache
} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp
