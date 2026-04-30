#include <gtest/gtest.h>

#include <array>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>

namespace {

std::string sourcePath(std::string_view relative_path)
{
    return std::string(SVMP_FE_SOURCE_DIR) + "/" + std::string(relative_path);
}

std::string readRequiredSource(std::string_view relative_path)
{
    const auto path = sourcePath(relative_path);
    std::ifstream input(path);
    if (!input) {
        ADD_FAILURE() << "could not open source file: " << path;
        return {};
    }
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
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
