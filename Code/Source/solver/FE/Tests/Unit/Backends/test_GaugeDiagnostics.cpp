/**
 * @file test_GaugeDiagnostics.cpp
 * @brief Unit tests for GaugeDiagnostics — numerical nullspace validation
 */

#include <gtest/gtest.h>

#include "Constraints/GaugeDiagnostics.h"
#include "Constraints/GaugeRegistry.h"
#include "Backends/Interfaces/BackendFactory.h"
#include "Backends/Interfaces/GenericMatrix.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Sparsity/SparsityPattern.h"

#include <cmath>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::gauge;
using namespace svmp::FE::backends;

namespace {

/// Create a 4x4 1D Laplacian with Neumann BCs on both ends:
///   L = [ 1 -1  0  0 ]
///       [-1  2 -1  0 ]
///       [ 0 -1  2 -1 ]
///       [ 0  0 -1  1 ]
///
/// Row sums are all 0, so z=[1,1,1,1] is in the nullspace.
std::unique_ptr<GenericMatrix> createLaplacianMatrix(const BackendFactory& factory)
{
    sparsity::SparsityPattern sp(4, 4);
    // Row 0
    sp.addEntry(0, 0); sp.addEntry(0, 1);
    // Row 1
    sp.addEntry(1, 0); sp.addEntry(1, 1); sp.addEntry(1, 2);
    // Row 2
    sp.addEntry(2, 1); sp.addEntry(2, 2); sp.addEntry(2, 3);
    // Row 3
    sp.addEntry(3, 2); sp.addEntry(3, 3);
    sp.finalize();

    auto mat = factory.createMatrix(sp);
    auto view = mat->createAssemblyView();
    view->beginAssemblyPhase();

    std::array<GlobalIndex, 1> row;
    std::array<GlobalIndex, 1> col;
    std::array<Real, 1> val;

    auto insertEntry = [&](GlobalIndex r, GlobalIndex c, Real v) {
        row[0] = r; col[0] = c; val[0] = v;
        view->addMatrixEntries(row, col, val);
    };

    insertEntry(0, 0, 1.0);  insertEntry(0, 1, -1.0);
    insertEntry(1, 0, -1.0); insertEntry(1, 1, 2.0); insertEntry(1, 2, -1.0);
    insertEntry(2, 1, -1.0); insertEntry(2, 2, 2.0); insertEntry(2, 3, -1.0);
    insertEntry(3, 2, -1.0); insertEntry(3, 3, 1.0);

    view->finalizeAssembly();
    return mat;
}

/// Create a 4x4 diagonal SPD matrix (no nullspace)
std::unique_ptr<GenericMatrix> createSPDMatrix(const BackendFactory& factory)
{
    sparsity::SparsityPattern sp(4, 4);
    for (int i = 0; i < 4; ++i) sp.addEntry(i, i);
    sp.finalize();

    auto mat = factory.createMatrix(sp);
    auto view = mat->createAssemblyView();
    view->beginAssemblyPhase();

    std::array<GlobalIndex, 1> row, col;
    std::array<Real, 1> val;
    for (int i = 0; i < 4; ++i) {
        row[0] = i; col[0] = i; val[0] = 1.0 + 0.1 * i;
        view->addMatrixEntries(row, col, val);
    }
    view->finalizeAssembly();
    return mat;
}

} // namespace

// ============================================================================
// Validation with true nullspace
// ============================================================================

TEST(GaugeDiagnostics, LaplacianNullspace_Passes)
{
    auto factory = BackendFactory::create(BackendKind::FSILS);
    ASSERT_NE(factory, nullptr);

    auto mat = createLaplacianMatrix(*factory);
    ASSERT_NE(mat, nullptr);

    // Normalized constant vector: z = [0.5, 0.5, 0.5, 0.5], ||z|| = 1
    std::vector<std::vector<double>> basis = {
        {0.5, 0.5, 0.5, 0.5}
    };

    auto results = validateNullspaceBasis(*mat, *factory, basis);

    ASSERT_EQ(results.size(), 1u);
    EXPECT_TRUE(results[0].passed);
    EXPECT_LT(results[0].az_norm, 1e-10);
    EXPECT_NEAR(results[0].z_norm, 1.0, 1e-14);
    EXPECT_GT(results[0].a_norm_estimate, 0.0);
}

// ============================================================================
// Validation with non-nullspace vector (should fail)
// ============================================================================

TEST(GaugeDiagnostics, SPDMatrix_NonNullspace_Fails)
{
    auto factory = BackendFactory::create(BackendKind::FSILS);
    ASSERT_NE(factory, nullptr);

    auto mat = createSPDMatrix(*factory);

    // [0.5, 0.5, 0.5, 0.5] is NOT in the nullspace of a diagonal SPD matrix
    std::vector<std::vector<double>> basis = {
        {0.5, 0.5, 0.5, 0.5}
    };

    auto results = validateNullspaceBasis(*mat, *factory, basis);

    ASSERT_EQ(results.size(), 1u);
    EXPECT_FALSE(results[0].passed);
    EXPECT_GT(results[0].relative_residual, 0.1);
}

// ============================================================================
// Empty basis
// ============================================================================

TEST(GaugeDiagnostics, EmptyBasis_ReturnsEmpty)
{
    auto factory = BackendFactory::create(BackendKind::FSILS);
    auto mat = createSPDMatrix(*factory);

    std::vector<std::vector<double>> basis;
    auto results = validateNullspaceBasis(*mat, *factory, basis);

    EXPECT_TRUE(results.empty());
}

// ============================================================================
// Report formatting
// ============================================================================

TEST(GaugeDiagnostics, FormatReport_ContainsPassFail)
{
    std::vector<ValidationResult> results;

    ValidationResult r1;
    r1.mode_index = 0;
    r1.passed = true;
    r1.description = "Mode 0: passed";
    results.push_back(r1);

    ValidationResult r2;
    r2.mode_index = 1;
    r2.passed = false;
    r2.description = "Mode 1: failed";
    results.push_back(r2);

    auto report = formatValidationReport(results);
    EXPECT_NE(report.find("[PASS]"), std::string::npos);
    EXPECT_NE(report.find("[FAIL]"), std::string::npos);
    EXPECT_NE(report.find("Passed: 1"), std::string::npos);
    EXPECT_NE(report.find("Failed: 1"), std::string::npos);
}

// ============================================================================
// Environment variable check
// ============================================================================

TEST(GaugeDiagnostics, EnvVarCheck_DefaultOff)
{
    // The function should not crash regardless of env state
    [[maybe_unused]] bool result = isNullspaceValidationEnabled();
}

// ============================================================================
// Custom tolerance
// ============================================================================

TEST(GaugeDiagnostics, CustomTolerance_StrictStillPasses)
{
    auto factory = BackendFactory::create(BackendKind::FSILS);
    auto mat = createLaplacianMatrix(*factory);

    std::vector<std::vector<double>> basis = {
        {0.5, 0.5, 0.5, 0.5}
    };

    ValidationOptions strict;
    strict.tolerance = 1e-14;
    auto results = validateNullspaceBasis(*mat, *factory, basis, strict);

    ASSERT_EQ(results.size(), 1u);
    EXPECT_TRUE(results[0].passed);
}

// ============================================================================
// Multiple basis vectors — mixed pass/fail
// ============================================================================

TEST(GaugeDiagnostics, MultipleBasisVectors)
{
    auto factory = BackendFactory::create(BackendKind::FSILS);
    auto mat = createLaplacianMatrix(*factory);

    // First: actual nullspace, second: NOT nullspace
    std::vector<std::vector<double>> basis = {
        {0.5, 0.5, 0.5, 0.5},      // constant mode (nullspace)
        {1.0, 0.0, 0.0, 0.0}        // e_1 (not nullspace)
    };

    auto results = validateNullspaceBasis(*mat, *factory, basis);

    ASSERT_EQ(results.size(), 2u);
    EXPECT_TRUE(results[0].passed);
    EXPECT_FALSE(results[1].passed);
}
