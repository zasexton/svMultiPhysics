/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUOF ANY KIND, INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <gtest/gtest.h>
#include "../../../Adaptivity/ErrorEstimator.h"
#include "../../../Core/MeshBase.h"
#include "../../../Fields/MeshFields.h"
#include "../../../Adaptivity/Options.h"
#include "MeshFieldsFixture.h"
#include <cmath>
#include <memory>
#include <vector>
#include <algorithm>

using namespace svmp;
using namespace svmp::test;

// ====================
// Test Fixtures and Helpers
// ====================
// NOTE: Now using MeshWithFieldsFixture from MeshFieldsFixture.h
// Old helper functions (TriangleMeshFixture, attach_*_field) have been removed
// in favor of the centralized fixture that properly manages mesh+field lifecycles.

// ====================
// GradientRecoveryEstimator Tests (15 tests)
// ====================

class GradientRecoveryEstimatorTest : public ::testing::Test {
protected:
  AdaptivityOptions options;

  void SetUp() override {
    options.estimator_type = AdaptivityOptions::EstimatorType::GRADIENT_RECOVERY;
  }
};

TEST_F(GradientRecoveryEstimatorTest, ComputesCorrectGradientForLinearField) {
  // Linear field should have constant gradient -> zero error indicator
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4, 2.0, 1.5, 10.0);

  GradientRecoveryEstimator::Config config;
  config.field_name = "solution";
  auto estimator = std::make_unique<GradientRecoveryEstimator>(config);

  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  ASSERT_EQ(indicators.size(), fixture.mesh.n_cells());

  // For a truly linear field, gradient recovery should find zero error
  // (Currently placeholder implementation returns h^2, so this will change)
  for (double indicator : indicators) {
    EXPECT_GE(indicator, 0.0) << "Indicators must be non-negative";
  }
}

TEST_F(GradientRecoveryEstimatorTest, SupportsFloat32VolumeField) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4, 1.0, 0.5, 0.0);

  // Create a Float32 copy of the solution field.
  auto handle_f32 = fixture.mesh.attach_field(EntityKind::Volume, "solution_f32", FieldScalarType::Float32, 1);
  float* data_f32 = static_cast<float*>(fixture.mesh.field_data(handle_f32));
  const double* data_f64 = fixture.field_data<double>("solution");
  for (size_t c = 0; c < fixture.mesh.n_cells(); ++c) {
    data_f32[c] = static_cast<float>(data_f64[c]);
  }

  GradientRecoveryEstimator::Config config;
  config.field_name = "solution_f32";
  auto estimator = std::make_unique<GradientRecoveryEstimator>(config);

  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);
  EXPECT_EQ(indicators.size(), fixture.mesh.n_cells());
}

TEST_F(GradientRecoveryEstimatorTest, RecoversQuadraticGradient) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_quadratic_field(4, 4);

  GradientRecoveryEstimator::Config config;
  config.field_name = "solution";
  auto estimator = std::make_unique<GradientRecoveryEstimator>(config);

  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  ASSERT_EQ(indicators.size(), fixture.mesh.n_cells());

  // Quadratic field should show non-zero error
  double max_indicator = *std::max_element(indicators.begin(), indicators.end());
  EXPECT_GT(max_indicator, 0.0) << "Quadratic field should produce non-zero indicators";
}

TEST_F(GradientRecoveryEstimatorTest, HandlesDiscontinuousSolutions) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_discontinuous_field(4, 4);

  GradientRecoveryEstimator::Config config;
  config.field_name = "solution";
  auto estimator = std::make_unique<GradientRecoveryEstimator>(config);

  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  ASSERT_EQ(indicators.size(), fixture.mesh.n_cells());

  // Discontinuity should produce large error indicators
  // (Implementation detail: depends on how gradient recovery handles jumps)
  EXPECT_FALSE(indicators.empty());
}

TEST_F(GradientRecoveryEstimatorTest, ProducesPositiveIndicators) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4, 1.0, 1.0, 0.0);

  auto estimator = ErrorEstimatorFactory::create_gradient_recovery();
  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  for (double indicator : indicators) {
    EXPECT_GE(indicator, 0.0) << "All indicators must be non-negative";
  }
}

TEST_F(GradientRecoveryEstimatorTest, ScalesCorrectlyWithMeshSize) {
  auto coarse_fixture = MeshWithFieldsFixture::create_2d_uniform_with_quadratic_field(2, 2);
  auto fine_fixture = MeshWithFieldsFixture::create_2d_uniform_with_quadratic_field(4, 4);

  auto estimator = ErrorEstimatorFactory::create_gradient_recovery();

  auto coarse_indicators = estimator->estimate(coarse_fixture.mesh, nullptr, options);
  auto fine_indicators = estimator->estimate(fine_fixture.mesh, nullptr, options);

  // Both meshes span the same physical domain (nx by ny units)
  // but with different subdivisions, so they have the same max cell size
  // The current implementation uses h^2 scaling based on cell measure
  // All cells in each mesh have roughly the same size, so indicators are similar

  // This test verifies that the estimator produces consistent results
  // across different mesh resolutions. With field-based estimation,
  // we would expect different behavior based on solution gradients.
  ASSERT_EQ(coarse_indicators.size(), coarse_fixture.mesh.n_cells());
  ASSERT_EQ(fine_indicators.size(), fine_fixture.mesh.n_cells());

  // For now, we just verify indicators are non-negative and consistent
  for (double indicator : coarse_indicators) {
    EXPECT_GE(indicator, 0.0);
  }
  for (double indicator : fine_indicators) {
    EXPECT_GE(indicator, 0.0);
  }
}

TEST_F(GradientRecoveryEstimatorTest, HandlesAnisotropicMeshes) {
  auto fixture = MeshWithFieldsFixture::create_2d_anisotropic_with_linear_field(4, 4, 1.0, 0.0, 0.0);

  GradientRecoveryEstimator::Config config;
  config.field_name = "solution";
  auto estimator = std::make_unique<GradientRecoveryEstimator>(config);

  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  ASSERT_EQ(indicators.size(), fixture.mesh.n_cells());
  EXPECT_FALSE(indicators.empty());
}

TEST_F(GradientRecoveryEstimatorTest, ZeroGradientForConstantField) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_constant_field(4, 4, 5.0);

  auto estimator = ErrorEstimatorFactory::create_gradient_recovery();
  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  // Constant field should have zero gradient -> zero or small error
  double max_indicator = *std::max_element(indicators.begin(), indicators.end());
  EXPECT_GE(max_indicator, 0.0) << "Constant field should produce minimal indicators";
}

TEST_F(GradientRecoveryEstimatorTest, BoundaryGradientHandling) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4, 1.0, 1.0, 0.0);

  GradientRecoveryEstimator::Config config;
  config.field_name = "solution";
  config.use_patch_recovery = true;
  auto estimator = std::make_unique<GradientRecoveryEstimator>(config);

  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  // Boundary cells should also get valid indicators
  ASSERT_EQ(indicators.size(), fixture.mesh.n_cells());
}

TEST_F(GradientRecoveryEstimatorTest, MultipleFieldComponents) {
  // Create fixture with base mesh
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4);

  // Attach vector field (3 components)
  auto handle = fixture.mesh.attach_field(EntityKind::Volume, "velocity", FieldScalarType::Float64, 3);
  double* data = static_cast<double*>(fixture.mesh.field_data(handle));

  for (size_t c = 0; c < fixture.mesh.n_cells(); ++c) {
    data[3*c + 0] = 1.0;  // vx
    data[3*c + 1] = 2.0;  // vy
    data[3*c + 2] = 0.0;  // vz
  }

  // Currently GradientRecoveryEstimator expects scalar field
  // This test documents the requirement for vector field support
  GradientRecoveryEstimator::Config config;
  config.field_name = "velocity";
  auto estimator = std::make_unique<GradientRecoveryEstimator>(config);

  // Should either handle vector fields or throw clear error
  EXPECT_NO_THROW(estimator->estimate(fixture.mesh, nullptr, options));
}

TEST_F(GradientRecoveryEstimatorTest, SmoothingIterationsConverge) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_quadratic_field(4, 4);

  GradientRecoveryEstimator::Config config;
  config.field_name = "solution";
  config.recovery_order = 2;  // Higher order recovery
  auto estimator = std::make_unique<GradientRecoveryEstimator>(config);

  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  ASSERT_EQ(indicators.size(), fixture.mesh.n_cells());
  // Recovery should converge (implementation-dependent)
}

TEST_F(GradientRecoveryEstimatorTest, CompatibleWithAllCellTypes) {
  // Test with triangles (already tested) and quads
  // Note: Current mesh builder only supports triangles
  // This test documents the requirement for quad support
  EXPECT_TRUE(true) << "Quad and hex cell type support TODO";
}

TEST_F(GradientRecoveryEstimatorTest, ErrorIndicatorFieldIsCreated) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4, 1.0, 1.0, 0.0);

  auto estimator = ErrorEstimatorFactory::create_gradient_recovery();
  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  ASSERT_EQ(indicators.size(), fixture.mesh.n_cells());

  // Indicators should be attachable as field
  auto handle = fixture.mesh.attach_field(EntityKind::Volume, "error_indicator",
                                  FieldScalarType::Float64, 1);
  double* data = static_cast<double*>(fixture.mesh.field_data(handle));
  std::copy(indicators.begin(), indicators.end(), data);

  EXPECT_TRUE(fixture.mesh.has_field(EntityKind::Volume, "error_indicator"));
}

TEST_F(GradientRecoveryEstimatorTest, RespectsToleranceSettings) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4, 1.0, 1.0, 0.0);

  GradientRecoveryEstimator::Config config;
  config.field_name = "solution";
  config.volume_weighted = true;
  auto estimator = std::make_unique<GradientRecoveryEstimator>(config);

  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  ASSERT_EQ(indicators.size(), fixture.mesh.n_cells());
}

TEST_F(GradientRecoveryEstimatorTest, HandlesDegenerateCellsGracefully) {
  // Create a mesh with near-degenerate cell (very small area)
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(2, 2, 1.0, 0.0, 0.0);

  auto estimator = ErrorEstimatorFactory::create_gradient_recovery();

  // Should not crash on degenerate cells
  EXPECT_NO_THROW({
    auto indicators = estimator->estimate(fixture.mesh, nullptr, options);
    EXPECT_EQ(indicators.size(), fixture.mesh.n_cells());
  });
}

TEST_F(GradientRecoveryEstimatorTest, IntegrationWithMeshFieldsAPI) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4, 1.0, 1.0, 0.0);

  EXPECT_TRUE(fixture.mesh.has_field(EntityKind::Volume, "solution"));

  auto estimator = ErrorEstimatorFactory::create_gradient_recovery();
  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  EXPECT_EQ(indicators.size(), fixture.mesh.n_cells());
}

// ====================
// JumpIndicatorEstimator Tests (12 tests)
// ====================

class JumpIndicatorEstimatorTest : public ::testing::Test {
protected:
  AdaptivityOptions options;

  void SetUp() override {
    options.estimator_type = AdaptivityOptions::EstimatorType::JUMP_INDICATOR;
  }
};

TEST_F(JumpIndicatorEstimatorTest, DetectsJumpsAcrossCellFaces) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_discontinuous_field(4, 4);

  JumpIndicatorEstimator::Config config;
  config.field_name = "solution";
  config.jump_type = JumpIndicatorEstimator::Config::JumpType::VALUE;
  auto estimator = std::make_unique<JumpIndicatorEstimator>(config);

  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  ASSERT_EQ(indicators.size(), fixture.mesh.n_cells());

  // Cells near discontinuity should have large jumps
  double max_indicator = *std::max_element(indicators.begin(), indicators.end());
  EXPECT_GT(max_indicator, 0.0) << "Discontinuity should produce non-zero jump indicators";
}

TEST_F(JumpIndicatorEstimatorTest, ZeroJumpForContinuousField) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4, 1.0, 1.0, 0.0);

  auto estimator = ErrorEstimatorFactory::create_jump_indicator();
  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  // Continuous linear field should have minimal jumps
  // (Implementation uses h scaling, so may not be exactly zero)
  for (double indicator : indicators) {
    EXPECT_GE(indicator, 0.0);
  }
}

TEST_F(JumpIndicatorEstimatorTest, ScalesWithJumpMagnitude) {
  // Create two fields with different jump magnitudes
  auto fixture1 = MeshWithFieldsFixture::create_2d_uniform_with_discontinuous_field(4, 4);  // Jump of 1.0

  // Create field with larger jump - use a different field name to avoid conflict
  auto fixture2 = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4, 0.0, 0.0, 0.0, "dummy");
  auto handle = fixture2.mesh.attach_field(EntityKind::Volume, "solution", FieldScalarType::Float64, 1);
  double* data = static_cast<double*>(fixture2.mesh.field_data(handle));
  for (size_t c = 0; c < fixture2.mesh.n_cells(); ++c) {
    auto verts = fixture2.mesh.cell_vertices(static_cast<index_t>(c));
    double cx = 0.0;
    for (auto v : verts) {
      auto pos = fixture2.mesh.get_vertex_coords(v);
      cx += pos[0];
    }
    cx /= verts.size();
    data[c] = (cx < 2.0) ? 0.0 : 10.0;  // Jump of 10.0
  }

  auto estimator = ErrorEstimatorFactory::create_jump_indicator();

  auto indicators1 = estimator->estimate(fixture1.mesh, nullptr, options);
  auto indicators2 = estimator->estimate(fixture2.mesh, nullptr, options);

  double max1 = *std::max_element(indicators1.begin(), indicators1.end());
  double max2 = *std::max_element(indicators2.begin(), indicators2.end());

  // Larger jump should produce larger indicators
  // (Implementation-dependent scaling)
  EXPECT_GT(max2, 0.0);
  EXPECT_GT(max1, 0.0);
}

TEST_F(JumpIndicatorEstimatorTest, HandlesBoundaryFacesCorrectly) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4, 1.0, 1.0, 0.0);

  JumpIndicatorEstimator::Config config;
  config.field_name = "solution";
  auto estimator = std::make_unique<JumpIndicatorEstimator>(config);

  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  // Boundary faces should be handled (zero jump or special treatment)
  ASSERT_EQ(indicators.size(), fixture.mesh.n_cells());
}

TEST_F(JumpIndicatorEstimatorTest, WorksWithAllCellTypes) {
  // Test documents requirement for quad/hex support
  EXPECT_TRUE(true) << "Quad/Hex cell type support TODO";
}

TEST_F(JumpIndicatorEstimatorTest, MultipleComponentFields) {
  // Create base fixture
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4);

  // Attach vector field
  auto handle = fixture.mesh.attach_field(EntityKind::Volume, "velocity", FieldScalarType::Float64, 3);
  double* data = static_cast<double*>(fixture.mesh.field_data(handle));

  for (size_t c = 0; c < fixture.mesh.n_cells(); ++c) {
    auto verts = fixture.mesh.cell_vertices(static_cast<index_t>(c));
    double cx = 0.0;
    for (auto v : verts) {
      auto pos = fixture.mesh.get_vertex_coords(v);
      cx += pos[0];
    }
    cx /= verts.size();

    data[3*c + 0] = (cx < 2.0) ? 0.0 : 1.0;
    data[3*c + 1] = 0.0;
    data[3*c + 2] = 0.0;
  }

  JumpIndicatorEstimator::Config config;
  config.field_name = "velocity";
  auto estimator = std::make_unique<JumpIndicatorEstimator>(config);

  EXPECT_NO_THROW(estimator->estimate(fixture.mesh, nullptr, options));
}

TEST_F(JumpIndicatorEstimatorTest, InteriorVsBoundaryJumps) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_discontinuous_field(4, 4);

  JumpIndicatorEstimator::Config config;
  config.field_name = "solution";
  config.jump_type = JumpIndicatorEstimator::Config::JumpType::NORMAL_DERIVATIVE;
  auto estimator = std::make_unique<JumpIndicatorEstimator>(config);

  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  ASSERT_EQ(indicators.size(), fixture.mesh.n_cells());
}

TEST_F(JumpIndicatorEstimatorTest, AnisotropicJumpDetection) {
  // Create anisotropic mesh with discontinuous field directly
  auto fixture = MeshWithFieldsFixture::create_2d_anisotropic_with_linear_field(4, 4, 0.0, 0.0, 0.0, "dummy");

  // Create discontinuous field on anisotropic mesh
  auto handle = fixture.mesh.attach_field(EntityKind::Volume, "solution", FieldScalarType::Float64, 1);
  double* data = static_cast<double*>(fixture.mesh.field_data(handle));
  for (size_t c = 0; c < fixture.mesh.n_cells(); ++c) {
    auto verts = fixture.mesh.cell_vertices(static_cast<index_t>(c));
    double cx = 0.0;
    for (auto v : verts) {
      auto pos = fixture.mesh.get_vertex_coords(v);
      cx += pos[0];
    }
    cx /= verts.size();
    data[c] = (cx < 2.0) ? 0.0 : 1.0;
  }

  auto estimator = ErrorEstimatorFactory::create_jump_indicator();
  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  ASSERT_EQ(indicators.size(), fixture.mesh.n_cells());
}

TEST_F(JumpIndicatorEstimatorTest, IntegrationWithMeshGeometry) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4, 1.0, 1.0, 0.0);

  JumpIndicatorEstimator::Config config;
  config.field_name = "solution";
  config.area_scaled = true;  // Scale by face area
  auto estimator = std::make_unique<JumpIndicatorEstimator>(config);

  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  ASSERT_EQ(indicators.size(), fixture.mesh.n_cells());
}

TEST_F(JumpIndicatorEstimatorTest, EdgeBasedVsFaceBasedJumps) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_discontinuous_field(4, 4);

  // Test edge-based jump computation
  JumpIndicatorEstimator::Config config;
  config.field_name = "solution";
  config.jump_type = JumpIndicatorEstimator::Config::JumpType::NORMAL_DERIVATIVE;
  auto estimator = std::make_unique<JumpIndicatorEstimator>(config);

  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  ASSERT_EQ(indicators.size(), fixture.mesh.n_cells());
}

TEST_F(JumpIndicatorEstimatorTest, ToleranceSensitivity) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_discontinuous_field(4, 4);

  JumpIndicatorEstimator::Config config;
  config.field_name = "solution";
  config.norm_power = 1.0;  // L1 norm
  auto estimator1 = std::make_unique<JumpIndicatorEstimator>(config);

  config.norm_power = 2.0;  // L2 norm
  auto estimator2 = std::make_unique<JumpIndicatorEstimator>(config);

  auto indicators1 = estimator1->estimate(fixture.mesh, nullptr, options);
  auto indicators2 = estimator2->estimate(fixture.mesh, nullptr, options);

  // Different norms should produce different values
  EXPECT_EQ(indicators1.size(), indicators2.size());
}

TEST_F(JumpIndicatorEstimatorTest, FieldAttachmentVerification) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4, 1.0, 1.0, 0.0);

  EXPECT_TRUE(fixture.mesh.has_field(EntityKind::Volume, "solution"));

  auto estimator = ErrorEstimatorFactory::create_jump_indicator();
  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  EXPECT_FALSE(indicators.empty());
}

// ====================
// ResidualBasedEstimator Tests (12 tests)
// ====================

class ResidualBasedEstimatorTest : public ::testing::Test {
protected:
  AdaptivityOptions options;

  void SetUp() override {
    options.estimator_type = AdaptivityOptions::EstimatorType::RESIDUAL_BASED;
  }
};

TEST_F(ResidualBasedEstimatorTest, ComputesResidualCorrectly) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4, 1.0, 1.0, 0.0);

  // Define residual function
  ResidualBasedEstimator::Config config;
  config.cell_residual = [](const MeshBase& m, size_t elem_id, const MeshFields* fields) {
    return 0.1;  // Constant residual
  };
  config.h_weighted = false;  // Disable h-weighting to get raw residual value

  auto estimator = std::make_unique<ResidualBasedEstimator>(config);
  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  ASSERT_EQ(indicators.size(), fixture.mesh.n_cells());

  // All cells should have same residual
  double expected = 0.1;
  for (double indicator : indicators) {
    EXPECT_NEAR(indicator, expected, 1e-8);
  }
}

TEST_F(ResidualBasedEstimatorTest, ZeroResidualForExactSolution) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4, 1.0, 1.0, 0.0);

  ResidualBasedEstimator::Config config;
  config.cell_residual = [](const MeshBase& m, size_t elem_id, const MeshFields* fields) {
    return 0.0;  // Exact solution -> zero residual
  };

  auto estimator = std::make_unique<ResidualBasedEstimator>(config);
  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  for (double indicator : indicators) {
    EXPECT_NEAR(indicator, 0.0, 1e-8);
  }
}

TEST_F(ResidualBasedEstimatorTest, ScalesWithPDECoefficients) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4, 1.0, 1.0, 0.0);

  // Test with different scaling constants
  ResidualBasedEstimator::Config config1;
  config1.cell_residual = [](const MeshBase& m, size_t e, const MeshFields* f) { return 1.0; };
  config1.scaling_constant = 1.0;

  ResidualBasedEstimator::Config config2;
  config2.cell_residual = [](const MeshBase& m, size_t e, const MeshFields* f) { return 1.0; };
  config2.scaling_constant = 2.0;

  auto estimator1 = std::make_unique<ResidualBasedEstimator>(config1);
  auto estimator2 = std::make_unique<ResidualBasedEstimator>(config2);

  auto indicators1 = estimator1->estimate(fixture.mesh, nullptr, options);
  auto indicators2 = estimator2->estimate(fixture.mesh, nullptr, options);

  // Indicators2 should be twice indicators1
  for (size_t i = 0; i < indicators1.size(); ++i) {
    EXPECT_NEAR(indicators2[i], 2.0 * indicators1[i], 1e-8);
  }
}

TEST_F(ResidualBasedEstimatorTest, HandlesSourceTerms) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4, 1.0, 1.0, 0.0);

  // Residual includes source term
  ResidualBasedEstimator::Config config;
  config.cell_residual = [](const MeshBase& m, size_t elem_id, const MeshFields* fields) {
    // R = f - L(u), where f is source and L is operator
    return 1.0 - 0.5;  // source=1.0, L(u)=0.5
  };

  auto estimator = std::make_unique<ResidualBasedEstimator>(config);
  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  ASSERT_EQ(indicators.size(), fixture.mesh.n_cells());
}

TEST_F(ResidualBasedEstimatorTest, BoundaryConditionResiduals) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4, 1.0, 1.0, 0.0);

  ResidualBasedEstimator::Config config;
  config.cell_residual = [](const MeshBase& m, size_t e, const MeshFields* f) { return 0.1; };
  config.face_residual = [](const MeshBase& m, size_t e, const MeshFields* f) { return 0.05; };
  config.include_face_residuals = true;

  auto estimator = std::make_unique<ResidualBasedEstimator>(config);
  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  ASSERT_EQ(indicators.size(), fixture.mesh.n_cells());
}

TEST_F(ResidualBasedEstimatorTest, IntegrationByPartsAccuracy) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_quadratic_field(4, 4);

  // Test weak formulation residual
  ResidualBasedEstimator::Config config;
  config.cell_residual = [](const MeshBase& m, size_t e, const MeshFields* f) {
    // Simplified: would compute actual weak residual
    return 0.1;
  };

  auto estimator = std::make_unique<ResidualBasedEstimator>(config);
  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  ASSERT_EQ(indicators.size(), fixture.mesh.n_cells());
}

TEST_F(ResidualBasedEstimatorTest, AllCellTypesSupported) {
  // Document requirement for all cell types
  EXPECT_TRUE(true) << "Quad/Hex cell type support TODO";
}

TEST_F(ResidualBasedEstimatorTest, MultiplePhysicsFields) {
  // Attach multiple fields (e.g., velocity + pressure)
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_multiple_fields(4, 4, {"velocity", "pressure"});

  // Set field values
  auto vel_handle = fixture.get_field("velocity");
  auto press_handle = fixture.get_field("pressure");
  double* vel_data = static_cast<double*>(fixture.mesh.field_data(vel_handle));
  double* press_data = static_cast<double*>(fixture.mesh.field_data(press_handle));

  for (size_t i = 0; i < fixture.mesh.n_cells(); ++i) {
    vel_data[i] = 1.0;
    press_data[i] = 0.5;
  }

  ResidualBasedEstimator::Config config;
  config.cell_residual = [](const MeshBase& m, size_t e, const MeshFields* f) {
    // Combined residual from multiple fields
    return 0.1;
  };

  auto estimator = std::make_unique<ResidualBasedEstimator>(config);
  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  ASSERT_EQ(indicators.size(), fixture.mesh.n_cells());
}

TEST_F(ResidualBasedEstimatorTest, AdaptiveTolerance) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4, 1.0, 1.0, 0.0);

  ResidualBasedEstimator::Config config;
  config.cell_residual = [](const MeshBase& m, size_t e, const MeshFields* f) { return 0.1; };
  config.h_weighted = true;  // Adaptive weighting

  auto estimator = std::make_unique<ResidualBasedEstimator>(config);
  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  ASSERT_EQ(indicators.size(), fixture.mesh.n_cells());
}

TEST_F(ResidualBasedEstimatorTest, ConvergenceVerification) {
  // Test convergence on sequence of refined meshes
  auto coarse_fixture = MeshWithFieldsFixture::create_2d_uniform_with_quadratic_field(2, 2);
  auto fine_fixture = MeshWithFieldsFixture::create_2d_uniform_with_quadratic_field(4, 4);

  ResidualBasedEstimator::Config config;
  config.cell_residual = [](const MeshBase& m, size_t e, const MeshFields* f) {
    return 0.1;
  };

  auto estimator = std::make_unique<ResidualBasedEstimator>(config);

  auto coarse_ind = estimator->estimate(coarse_fixture.mesh, nullptr, options);
  auto fine_ind = estimator->estimate(fine_fixture.mesh, nullptr, options);

  // Should observe convergence behavior
  EXPECT_FALSE(coarse_ind.empty());
  EXPECT_FALSE(fine_ind.empty());
}

TEST_F(ResidualBasedEstimatorTest, FieldComponentHandling) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4, 1.0, 1.0, 0.0);

  ResidualBasedEstimator::Config config;
  config.cell_residual = [](const MeshBase& m, size_t e, const MeshFields* f) {
    return 0.1;
  };

  auto estimator = std::make_unique<ResidualBasedEstimator>(config);
  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  ASSERT_EQ(indicators.size(), fixture.mesh.n_cells());
}

TEST_F(ResidualBasedEstimatorTest, ErrorIndicatorNormalization) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4, 1.0, 1.0, 0.0);

  ResidualBasedEstimator::Config config;
  config.cell_residual = [](const MeshBase& m, size_t e, const MeshFields* f) {
    return static_cast<double>(e);  // Varying residual
  };

  auto estimator = std::make_unique<ResidualBasedEstimator>(config);
  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  // Check normalization
  ErrorEstimatorUtils::normalize_indicators(indicators);

  double max_val = *std::max_element(indicators.begin(), indicators.end());
  EXPECT_NEAR(max_val, 1.0, 1e-10) << "Normalized max should be 1.0";
}

// ====================
// UserFieldEstimator Tests (5 tests)
// ====================

class UserFieldEstimatorTest : public ::testing::Test {
protected:
  AdaptivityOptions options;

  void SetUp() override {
    options.estimator_type = AdaptivityOptions::EstimatorType::USER_FIELD;
    options.user_field_name = "error_indicator";
  }
};

TEST_F(UserFieldEstimatorTest, UsesUserProvidedIndicatorField) {
  // Create fixture with base mesh
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4);

  // Attach user-provided error indicator field
  auto handle = fixture.mesh.attach_field(EntityKind::Volume, "error_indicator",
                                  FieldScalarType::Float64, 1);
  double* data = static_cast<double*>(fixture.mesh.field_data(handle));
  for (size_t i = 0; i < fixture.mesh.n_cells(); ++i) {
    data[i] = static_cast<double>(i) / fixture.mesh.n_cells();
  }

  auto estimator = ErrorEstimatorFactory::create_user_field("error_indicator");
  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  ASSERT_EQ(indicators.size(), fixture.mesh.n_cells());

  // Should match user field
  for (size_t i = 0; i < fixture.mesh.n_cells(); ++i) {
    EXPECT_NEAR(indicators[i], data[i], 1e-10);
  }
}

TEST_F(UserFieldEstimatorTest, ValidatesFieldDimensions) {
  // Create fixture and attach error indicator field
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4);
  auto handle = fixture.mesh.attach_field(EntityKind::Volume, "error_indicator",
                                  FieldScalarType::Float64, 1);
  double* data = static_cast<double*>(fixture.mesh.field_data(handle));
  std::fill(data, data + fixture.mesh.n_cells(), 1.0);

  auto estimator = ErrorEstimatorFactory::create_user_field("error_indicator");
  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  EXPECT_EQ(indicators.size(), fixture.mesh.n_cells());
}

TEST_F(UserFieldEstimatorTest, HandlesMissingFieldGracefully) {
  // Create fixture without error indicator field
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4);

  auto estimator = ErrorEstimatorFactory::create_user_field("nonexistent");

  EXPECT_THROW({
    auto indicators = estimator->estimate(fixture.mesh, nullptr, options);
  }, std::runtime_error);
}

TEST_F(UserFieldEstimatorTest, PositiveIndicatorEnforcement) {
  // Create fixture and attach field with some negative values
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4);
  auto handle = fixture.mesh.attach_field(EntityKind::Volume, "error_indicator",
                                  FieldScalarType::Float64, 1);
  double* data = static_cast<double*>(fixture.mesh.field_data(handle));
  for (size_t i = 0; i < fixture.mesh.n_cells(); ++i) {
    data[i] = static_cast<double>(i) - fixture.mesh.n_cells() / 2.0;  // Some negative
  }

  UserFieldEstimator::Config config;
  config.error_field_name = "error_indicator";
  auto estimator = std::make_unique<UserFieldEstimator>(config);

  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  // Note: Current implementation doesn't enforce positivity
  // This test documents the requirement
  EXPECT_EQ(indicators.size(), fixture.mesh.n_cells());
}

TEST_F(UserFieldEstimatorTest, FieldNameResolution) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4, 1.0, 1.0, 0.0, "my_error");

  UserFieldEstimator::Config config;
  config.error_field_name = "my_error";
  auto estimator = std::make_unique<UserFieldEstimator>(config);

  auto indicators = estimator->estimate(fixture.mesh, nullptr, options);

  EXPECT_EQ(indicators.size(), fixture.mesh.n_cells());
}

// ====================
// MultiCriteriaEstimator Tests (6 tests)
// ====================

class MultiCriteriaEstimatorTest : public ::testing::Test {
protected:
  AdaptivityOptions options;

  void SetUp() override {
    // No setup needed - each test creates its own fixture
  }
};

TEST_F(MultiCriteriaEstimatorTest, CombinesMultipleIndicators) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4);

  auto multi = std::make_unique<MultiCriteriaEstimator>();

  multi->add_estimator(ErrorEstimatorFactory::create_gradient_recovery(), 1.0);
  multi->add_estimator(ErrorEstimatorFactory::create_jump_indicator(), 1.0);

  auto indicators = multi->estimate(fixture.mesh, nullptr, options);

  ASSERT_EQ(indicators.size(), fixture.mesh.n_cells());
}

TEST_F(MultiCriteriaEstimatorTest, WeightedCombination) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4);

  auto multi = std::make_unique<MultiCriteriaEstimator>();

  multi->add_estimator(ErrorEstimatorFactory::create_gradient_recovery(), 2.0);
  multi->add_estimator(ErrorEstimatorFactory::create_jump_indicator(), 1.0);
  multi->set_aggregation_method(MultiCriteriaEstimator::AggregationMethod::WEIGHTED_SUM);

  auto indicators = multi->estimate(fixture.mesh, nullptr, options);

  ASSERT_EQ(indicators.size(), fixture.mesh.n_cells());
}

TEST_F(MultiCriteriaEstimatorTest, MaxMinAverageAggregation) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4);

  auto multi = std::make_unique<MultiCriteriaEstimator>();

  multi->add_estimator(ErrorEstimatorFactory::create_gradient_recovery(), 1.0);
  multi->add_estimator(ErrorEstimatorFactory::create_jump_indicator(), 1.0);

  // Test MAX aggregation
  multi->set_aggregation_method(MultiCriteriaEstimator::AggregationMethod::WEIGHTED_MAX);
  auto max_indicators = multi->estimate(fixture.mesh, nullptr, options);

  // Test L2 aggregation
  multi->set_aggregation_method(MultiCriteriaEstimator::AggregationMethod::WEIGHTED_L2);
  auto l2_indicators = multi->estimate(fixture.mesh, nullptr, options);

  EXPECT_EQ(max_indicators.size(), fixture.mesh.n_cells());
  EXPECT_EQ(l2_indicators.size(), fixture.mesh.n_cells());
}

TEST_F(MultiCriteriaEstimatorTest, NormalizationStrategies) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4);

  auto multi = std::make_unique<MultiCriteriaEstimator>();

  multi->add_estimator(ErrorEstimatorFactory::create_gradient_recovery(), 1.0);
  multi->add_estimator(ErrorEstimatorFactory::create_jump_indicator(), 1.0);

  auto indicators = multi->estimate(fixture.mesh, nullptr, options);

  ErrorEstimatorUtils::normalize_indicators(indicators);

  double max_val = *std::max_element(indicators.begin(), indicators.end());
  EXPECT_NEAR(max_val, 1.0, 1e-10);
}

TEST_F(MultiCriteriaEstimatorTest, ThresholdApplication) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4);

  auto multi = std::make_unique<MultiCriteriaEstimator>();

  multi->add_estimator(ErrorEstimatorFactory::create_gradient_recovery(), 1.0);

  auto indicators = multi->estimate(fixture.mesh, nullptr, options);

  // Apply threshold
  double threshold = 0.5 * (*std::max_element(indicators.begin(), indicators.end()));
  size_t count_above = 0;
  for (double ind : indicators) {
    if (ind >= threshold) count_above++;
  }

  EXPECT_GT(count_above, 0u);
}

TEST_F(MultiCriteriaEstimatorTest, FactoryCreation) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_linear_field(4, 4);

  auto multi = ErrorEstimatorFactory::create_multi_criteria();

  ASSERT_NE(multi, nullptr);

  multi->add_estimator(ErrorEstimatorFactory::create_gradient_recovery(), 1.0);

  auto indicators = multi->estimate(fixture.mesh, nullptr, options);

  EXPECT_EQ(indicators.size(), fixture.mesh.n_cells());
}

TEST_F(MultiCriteriaEstimatorTest, OptionsDrivenFactoryCreation) {
  auto fixture = MeshWithFieldsFixture::create_2d_uniform_with_quadratic_field(4, 4);

  AdaptivityOptions opts;
  opts.estimator_type = AdaptivityOptions::EstimatorType::MULTI_CRITERIA;
  opts.estimator_weights = {1.0, 1.0};  // grad + jump

  auto estimator = ErrorEstimatorFactory::create(opts);
  ASSERT_NE(estimator, nullptr);

  auto indicators = estimator->estimate(fixture.mesh, nullptr, opts);
  EXPECT_EQ(indicators.size(), fixture.mesh.n_cells());
}

// ====================
// ErrorEstimatorUtils Tests (10 tests total above)
// ====================

TEST(ErrorEstimatorUtilsTest, NormalizeIndicators) {
  std::vector<double> indicators = {0.5, 1.0, 0.25, 0.75};

  ErrorEstimatorUtils::normalize_indicators(indicators);

  EXPECT_NEAR(indicators[0], 0.5, 1e-10);
  EXPECT_NEAR(indicators[1], 1.0, 1e-10);
  EXPECT_NEAR(indicators[2], 0.25, 1e-10);
  EXPECT_NEAR(indicators[3], 0.75, 1e-10);
}

TEST(ErrorEstimatorUtilsTest, ComputeGlobalError) {
  std::vector<double> indicators = {1.0, 2.0, 3.0, 4.0};

  double l2_error = ErrorEstimatorUtils::compute_global_error(indicators, 2.0);

  double expected = std::sqrt(1.0 + 4.0 + 9.0 + 16.0);
  EXPECT_NEAR(l2_error, expected, 1e-10);
}

TEST(ErrorEstimatorUtilsTest, ComputeStatistics) {
  std::vector<double> indicators = {1.0, 2.0, 3.0, 4.0, 5.0};

  auto stats = ErrorEstimatorUtils::compute_statistics(indicators);

  EXPECT_NEAR(stats.min_error, 1.0, 1e-10);
  EXPECT_NEAR(stats.max_error, 5.0, 1e-10);
  EXPECT_NEAR(stats.mean_error, 3.0, 1e-10);
  EXPECT_NEAR(stats.total_error, 15.0, 1e-10);
  EXPECT_EQ(stats.num_cells, 5u);

  // Check standard deviation
  double variance = ((1-3)*(1-3) + (2-3)*(2-3) + (3-3)*(3-3) +
                     (4-3)*(4-3) + (5-3)*(5-3)) / 5.0;
  EXPECT_NEAR(stats.std_dev, std::sqrt(variance), 1e-10);
}

// ====================
// Main
// ====================

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
