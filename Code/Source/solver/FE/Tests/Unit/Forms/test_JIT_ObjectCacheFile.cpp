/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

// Unit tests for the on-disk JIT object cache format and file helpers.  These
// do not need LLVM: they cover the integrity check (every truncation or
// single-byte corruption is rejected), temporary-name uniqueness across
// threads/processes/hosts, the atomic publish step, and cache-directory
// resolution including the SVMP_JIT_CACHE_DIR override.

#include <gtest/gtest.h>

#include "Forms/JIT/JITObjectCacheFile.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

namespace {

namespace oc = jit::objcache;

class ScopedObjectCacheEnvVar final {
public:
    ScopedObjectCacheEnvVar(const char* key, const char* value)
        : key_(key)
    {
        if (const char* current = std::getenv(key); current != nullptr) {
            prior_ = std::string(current);
        }
        if (value != nullptr) {
            ::setenv(key_.c_str(), value, 1);
        } else {
            ::unsetenv(key_.c_str());
        }
    }

    ~ScopedObjectCacheEnvVar()
    {
        if (prior_) {
            ::setenv(key_.c_str(), prior_->c_str(), 1);
        } else {
            ::unsetenv(key_.c_str());
        }
    }

    ScopedObjectCacheEnvVar(const ScopedObjectCacheEnvVar&) = delete;
    ScopedObjectCacheEnvVar& operator=(const ScopedObjectCacheEnvVar&) = delete;

private:
    std::string key_;
    std::optional<std::string> prior_;
};

class ScopedObjectCacheTempDir final {
public:
    explicit ScopedObjectCacheTempDir(std::string_view label)
    {
        static std::atomic<std::uint64_t> counter{0u};
        const auto stamp = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        path_ = std::filesystem::temp_directory_path() /
                ("svmp_fe_" + std::string(label) + "_" + std::to_string(::getpid()) + "_" +
                 std::to_string(stamp) + "_" + std::to_string(counter.fetch_add(1u)));
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
        std::filesystem::create_directories(path_, ec);
    }

    ~ScopedObjectCacheTempDir()
    {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    ScopedObjectCacheTempDir(const ScopedObjectCacheTempDir&) = delete;
    ScopedObjectCacheTempDir& operator=(const ScopedObjectCacheTempDir&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

constexpr std::string_view kModuleId = "svmp_fe_jit_kernel_0123456789abcdef";

[[nodiscard]] std::string sampleObject(std::size_t size)
{
    std::string bytes(size, '\0');
    for (std::size_t i = 0; i < size; ++i) {
        bytes[i] = static_cast<char>(static_cast<unsigned char>((i * 131u + 7u) & 0xFFu));
    }
    // Looks like a raw ELF object, as written by older cache formats.
    bytes.replace(0u, 4u, "\x7f" "ELF");
    return bytes;
}

[[nodiscard]] std::string readFile(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

[[nodiscard]] std::size_t countEntries(const std::filesystem::path& dir)
{
    std::size_t n = 0u;
    std::error_code ec;
    for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        ++n;
    }
    return n;
}

[[nodiscard]] std::string expectedHostToken()
{
    char buffer[256] = {};
    if (::gethostname(buffer, sizeof(buffer) - 1u) != 0) {
        return "unknown-host";
    }
    std::string out;
    for (const char ch : std::string(buffer)) {
        if (out.size() >= 64u) {
            break;
        }
        const bool ok = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                        (ch >= '0' && ch <= '9') || ch == '_' || ch == '-' || ch == '.';
        out.push_back(ok ? ch : '_');
    }
    return out.empty() ? std::string("unknown-host") : out;
}

} // namespace

TEST(JITObjectCacheFile, RoundTripReturnsExactObjectBytes)
{
    const std::string object = sampleObject(777u);
    const std::string image = oc::encodeFile(kModuleId, object);
    ASSERT_EQ(image.size(), oc::kHeaderSize + kModuleId.size() + object.size());

    const auto decoded = oc::decodeFile(kModuleId, image);
    ASSERT_EQ(decoded.status, oc::FileStatus::Valid) << oc::toString(decoded.status);
    EXPECT_EQ(decoded.object, object);
}

TEST(JITObjectCacheFile, EverySingleByteCorruptionIsRejected)
{
    const std::string object = sampleObject(300u);
    const std::string image = oc::encodeFile(kModuleId, object);

    std::size_t accepted = 0u;
    for (std::size_t pos = 0; pos < image.size(); ++pos) {
        for (const unsigned int mask : {0x01u, 0x80u, 0xFFu}) {
            std::string corrupted = image;
            const auto original = static_cast<unsigned int>(static_cast<unsigned char>(corrupted[pos]));
            corrupted[pos] = static_cast<char>(static_cast<unsigned char>(original ^ mask));
            if (oc::decodeFile(kModuleId, corrupted).status == oc::FileStatus::Valid) {
                ++accepted;
                ADD_FAILURE() << "corruption accepted at byte " << pos << " mask " << mask;
            }
        }
    }
    EXPECT_EQ(accepted, 0u);
}

TEST(JITObjectCacheFile, EveryTruncationAndExtensionIsRejected)
{
    const std::string object = sampleObject(200u);
    const std::string image = oc::encodeFile(kModuleId, object);

    for (std::size_t len = 0; len < image.size(); ++len) {
        const auto status = oc::decodeFile(kModuleId, std::string_view(image).substr(0u, len)).status;
        EXPECT_NE(status, oc::FileStatus::Valid) << "truncated image of length " << len << " accepted";
    }
    EXPECT_EQ(oc::decodeFile(kModuleId, image + std::string(1u, '\0')).status, oc::FileStatus::SizeMismatch);
    EXPECT_EQ(oc::decodeFile(kModuleId, image.substr(0u, image.size() - 1u)).status,
              oc::FileStatus::SizeMismatch);
    EXPECT_EQ(oc::decodeFile(kModuleId, image.substr(0u, oc::kHeaderSize - 1u)).status,
              oc::FileStatus::TooShort);
}

TEST(JITObjectCacheFile, MisplacedLegacyAndEmptyObjectsAreRejected)
{
    const std::string object = sampleObject(128u);
    const std::string image = oc::encodeFile(kModuleId, object);

    // Same file under another module's name (different length, same length).
    EXPECT_EQ(oc::decodeFile("svmp_fe_jit_kernel_other", image).status, oc::FileStatus::ModuleIdMismatch);
    std::string same_length_id(kModuleId);
    same_length_id.back() = (same_length_id.back() == '0') ? '1' : '0';
    EXPECT_EQ(oc::decodeFile(same_length_id, image).status, oc::FileStatus::ModuleIdMismatch);

    // Raw objects from the previous (unchecked) format are never accepted.
    EXPECT_EQ(oc::decodeFile(kModuleId, object).status, oc::FileStatus::BadMagic);

    // An empty object is never valid.
    EXPECT_EQ(oc::decodeFile(kModuleId, oc::encodeFile(kModuleId, {})).status, oc::FileStatus::SizeMismatch);
}

TEST(JITObjectCacheFile, TempNamesEmbedHostAndProcessAndAreUniqueAcrossThreads)
{
    const std::filesystem::path final_path =
        std::filesystem::path("cache") / "objects" / "svmp_fe_jit_kernel_0123456789abcdef.objcache";

    const std::string marker = ".tmp." + expectedHostToken() + "." + std::to_string(::getpid()) + ".";
    const auto first = oc::uniqueTempPath(final_path);
    EXPECT_EQ(first.parent_path(), final_path.parent_path());
    EXPECT_EQ(first.filename().string().rfind(final_path.filename().string(), 0u), 0u) << first;
    EXPECT_NE(first.filename().string().find(marker), std::string::npos) << first << " lacks " << marker;
    EXPECT_NE(first.filename(), final_path.filename());

    constexpr int kThreads = 8;
    constexpr int kNamesPerThread = 4000;
    std::vector<std::vector<std::string>> names(static_cast<std::size_t>(kThreads));
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(kThreads));
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&names, &final_path, t]() {
            auto& out = names[static_cast<std::size_t>(t)];
            out.reserve(static_cast<std::size_t>(kNamesPerThread));
            for (int i = 0; i < kNamesPerThread; ++i) {
                out.push_back(oc::uniqueTempPath(final_path).filename().string());
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }

    std::set<std::string> unique;
    for (const auto& per_thread : names) {
        unique.insert(per_thread.begin(), per_thread.end());
    }
    EXPECT_EQ(unique.size(), static_cast<std::size_t>(kThreads * kNamesPerThread));

    // The random component differs between consecutive names, not only the counter.
    const std::string a = oc::uniqueTempPath(final_path).filename().string();
    const std::string b = oc::uniqueTempPath(final_path).filename().string();
    const auto random_of = [](const std::string& name) {
        const auto last = name.rfind('.');
        const auto prev = name.rfind('.', last - 1u);
        return name.substr(prev + 1u, last - prev - 1u);
    };
    EXPECT_EQ(random_of(a).size(), 16u);
    EXPECT_NE(random_of(a), random_of(b));

    // Long final names still give temporary names within NAME_MAX.
    const std::filesystem::path long_final =
        std::filesystem::path("cache") / (std::string(246u, 'k') + ".objcache");
    EXPECT_LE(oc::uniqueTempPath(long_final).filename().string().size(), 255u);
}

TEST(JITObjectCacheFile, AtomicWritePublishesOnlyTheFinalFile)
{
    ScopedObjectCacheTempDir dir("jit_objcache_atomic_write");
    const auto final_path = dir.path() / "svmp_fe_jit_kernel_atomic.objcache";

    const std::string first = oc::encodeFile(kModuleId, sampleObject(64u));
    ASSERT_TRUE(oc::writeFileAtomically(final_path, first));
    EXPECT_EQ(readFile(final_path), first);
    EXPECT_EQ(countEntries(dir.path()), 1u);

    // Replacing an existing file is atomic and leaves no temporaries.
    const std::string second = oc::encodeFile(kModuleId, sampleObject(96u));
    ASSERT_TRUE(oc::writeFileAtomically(final_path, second));
    EXPECT_EQ(readFile(final_path), second);
    EXPECT_EQ(countEntries(dir.path()), 1u);

    // Failure (missing directory) reports false and creates nothing.
    const auto missing = dir.path() / "missing" / "x.objcache";
    EXPECT_FALSE(oc::writeFileAtomically(missing, first));
    EXPECT_FALSE(std::filesystem::exists(missing.parent_path()));
    EXPECT_EQ(countEntries(dir.path()), 1u);
}

TEST(JITObjectCacheFile, CacheDirectoryResolutionPrecedence)
{
    ScopedObjectCacheEnvVar home("HOME", "/home/svmp-test-user");
    {
        ScopedObjectCacheEnvVar env(oc::kCacheDirectoryEnvVar, "/scratch/svmp-test/jit");
        EXPECT_EQ(oc::resolveCacheDirectory("/explicit/cache", true), "/explicit/cache");
        EXPECT_EQ(oc::resolveCacheDirectory("/explicit/cache", false), "/explicit/cache");
        EXPECT_EQ(oc::resolveCacheDirectory("", true), "/scratch/svmp-test/jit");
        EXPECT_EQ(oc::resolveCacheDirectory("", false), "");
    }
    {
        ScopedObjectCacheEnvVar env(oc::kCacheDirectoryEnvVar, "");
        EXPECT_EQ(oc::resolveCacheDirectory("", true), "/home/svmp-test-user/.cache/svMultiPhysics/jit_cache");
    }
    {
        ScopedObjectCacheEnvVar env(oc::kCacheDirectoryEnvVar, nullptr);
        EXPECT_EQ(oc::resolveCacheDirectory("", true), "/home/svmp-test-user/.cache/svMultiPhysics/jit_cache");
        ScopedObjectCacheEnvVar no_home("HOME", nullptr);
        EXPECT_EQ(oc::resolveCacheDirectory("", true), "");
    }
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
