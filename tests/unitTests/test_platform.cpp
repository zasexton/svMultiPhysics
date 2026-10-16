// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "Core/Exception.h"
#include "Parameters.h"
#include "lapack_defs.h"
#include "svOneD_interface/OneDSolverInterface.h"
#include "utils.h"

#include <gtest/gtest.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

extern "C" double cput_();

TEST(PlatformPaths, PreservesSpacesInStringParameters)
{
  Parameter<std::string> path;
  path.set(" \tC:\\solver data\\coupled library.dll\r\n");
  EXPECT_EQ(path.value(), "C:\\solver data\\coupled library.dll");
  path.set(" \t\r\n");
  EXPECT_TRUE(path.value().empty());
  Parameter<int> ranks;
  ranks.set(" 4 ");
  EXPECT_EQ(ranks.value(), 4);
}

TEST(DynamicLibrary, RejectsMissingLibraries)
{
  OneDSolverInterface one;
  EXPECT_THROW(one.load_library("missing library with spaces.dll"), std::exception);
}

TEST(DynamicLibrary, RejectsMissingInterfaceSymbols)
{
  OneDSolverInterface one;
  try {
    one.load_library(SVMP_LOADER_FIXTURE);
    FAIL() << "Accepted a library without interface symbols";
  } catch (const std::exception& error) {
    EXPECT_NE(std::string(error.what()).find("initialize"), std::string::npos);
  }
}

TEST(DynamicLibrary, DestructionWithoutLoadingIsSafe)
{
  OneDSolverInterface one;
}

class PlatformFiles : public testing::Test {
protected:
  std::filesystem::path directory = std::filesystem::temp_directory_path() /
      ("svmp files " + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  void SetUp() override { std::filesystem::create_directories(directory); }
  void TearDown() override { std::filesystem::remove_all(directory); }
};

TEST_F(PlatformFiles, RunsProcessWithLiteralPathsAndWaitsForOutput)
{
  const auto output = directory / "input ' with & %PATH%.txt";
  ASSERT_NO_THROW(utils::run_process(SVMP_PROCESS_FIXTURE, output.string()));
  std::string contents;
  std::getline(std::ifstream(output), contents);
  EXPECT_EQ(contents, "complete");
}

TEST(PlatformProcess, RejectsMissingExecutable)
{
  EXPECT_THROW(utils::run_process("missing program.exe", "input file"), std::system_error);
}

TEST(PlatformProcess, RejectsNonzeroExit)
{
  EXPECT_THROW(utils::run_process(SVMP_PROCESS_FIXTURE, "fail"), std::runtime_error);
}

TEST(PlatformSupport, HandlesSymbolsAndMemoryReporting)
{
  EXPECT_EQ(svmp::PlatformSupport::demangle_symbol(nullptr), "");
  EXPECT_FALSE(svmp::PlatformSupport::demangle_symbol("solver_symbol").empty());
  EXPECT_GT(utils::mem_usage(), 0.0);
}

#ifndef __MACH__ // The existing macOS timer measures wall time.
TEST(PlatformSupport, CpuTimerExcludesSleepingTime)
{
  const double before = cput_();
  ASSERT_GE(before, 0.0);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  const double after = cput_();
  EXPECT_GE(after, before);
  EXPECT_LT(after - before, 0.05); // CPU seconds, not elapsed wall time.
}
#endif

TEST(Lapack, SolvesAndInvertsWithIntIndices)
{
  int n = 2, nrhs = 1, info = -1, pivots[2];
  double matrix[] = {4.0, 1.0, 2.0, 3.0};
  double rhs[] = {8.0, 7.0};
  dgesv_(&n, &nrhs, matrix, &n, pivots, rhs, &n, &info);
  ASSERT_EQ(info, 0);
  EXPECT_NEAR(rhs[0], 1.0, 1e-14);
  EXPECT_NEAR(rhs[1], 2.0, 1e-14);
  double inverse[] = {4.0, 1.0, 2.0, 3.0};
  dgetrf_(&n, &n, inverse, &n, pivots, &info);
  ASSERT_EQ(info, 0);
  int workspace_size = 32;
  double workspace[32];
  dgetri_(&n, inverse, &n, pivots, workspace, &workspace_size, &info);
  ASSERT_EQ(info, 0);
  const double expected[] = {0.3, -0.1, -0.2, 0.4};
  for (int i = 0; i < 4; ++i) EXPECT_NEAR(inverse[i], expected[i], 1e-14);
}
