#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fs = std::filesystem;

std::string sourcePath(std::string_view relative_path)
{
    return std::string(SVMP_FE_SOURCE_DIR) + "/" + std::string(relative_path);
}

std::string readRequiredPath(const fs::path& path)
{
    std::ifstream input(path);
    if (!input) {
        ADD_FAILURE() << "could not open source file: " << path;
        return {};
    }
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

std::string readRequiredSource(std::string_view relative_path)
{
    return readRequiredPath(sourcePath(relative_path));
}

std::vector<fs::path> couplingSourcePaths()
{
    std::vector<fs::path> paths;
    for (const auto& entry : fs::recursive_directory_iterator(sourcePath("Coupling"))) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto extension = entry.path().extension().string();
        if (extension == ".cpp" || extension == ".h" || extension == ".hpp") {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    if (paths.empty()) {
        ADD_FAILURE() << "no coupling source files found";
    }
    return paths;
}

void expectNoDirectKernelRegistration(std::string_view relative_path)
{
    static constexpr std::array<std::string_view, 6> kBlockedCalls{{
        ".addCellKernel(",
        ".addBoundaryKernel(",
        ".addInteriorFaceKernel(",
        ".addInterfaceFaceKernel(",
        ".addGlobalKernel(",
        ".addMatrixFreeKernel(",
    }};

    const auto source = readRequiredSource(relative_path);
    ASSERT_FALSE(source.empty());
    for (const auto call : kBlockedCalls) {
        EXPECT_EQ(source.find(call), std::string::npos)
            << relative_path << " should not call " << call;
    }
}

} // namespace

TEST(CouplingSourceBoundaries, MonolithicBuilderUsesFormsInstallerEntryPoint)
{
    const auto source = readRequiredSource("Coupling/MonolithicCouplingBuilder.cpp");
    ASSERT_FALSE(source.empty());

    EXPECT_NE(source.find("systems::installFormulationWithMetadata("),
              std::string::npos);
    expectNoDirectKernelRegistration("Coupling/MonolithicCouplingBuilder.cpp");
}

TEST(CouplingSourceBoundaries, ExpertHookFixturesAvoidDirectKernelRegistration)
{
    expectNoDirectKernelRegistration(
        "Tests/Unit/Coupling/test_MonolithicCouplingBuilder.cpp");
}

TEST(CouplingSourceBoundaries, CouplingSourcesDoNotIncludePhysicsCouplingHeaders)
{
    const auto paths = couplingSourcePaths();
    ASSERT_FALSE(paths.empty());

    for (const auto& path : paths) {
        const auto source = readRequiredPath(path);
        ASSERT_FALSE(source.empty());
        EXPECT_EQ(source.find("\"Physics/Coupling/"), std::string::npos)
            << path << " should not include Physics coupling headers";
        EXPECT_EQ(source.find("<Physics/Coupling/"), std::string::npos)
            << path << " should not include Physics coupling headers";
    }
}

TEST(CouplingSourceBoundaries, CouplingSourcesUsePublicFormsSystemsHeaders)
{
    static constexpr std::array<std::string_view, 7> kBlockedHeaders{{
        "Systems/FormsInstallerDetail.h",
        "Forms/FormCompiler.h",
        "Forms/FormKernels.h",
        "Forms/MixedBlockKernelSet.h",
        "Forms/MonolithicCellKernel.h",
        "Forms/JIT/",
        "Forms/Internal/",
    }};

    const auto paths = couplingSourcePaths();
    ASSERT_FALSE(paths.empty());

    for (const auto& path : paths) {
        const auto source = readRequiredPath(path);
        ASSERT_FALSE(source.empty());
        for (const auto header : kBlockedHeaders) {
            EXPECT_EQ(source.find(header), std::string::npos)
                << path << " should use public Forms/Systems headers";
        }
    }
}
