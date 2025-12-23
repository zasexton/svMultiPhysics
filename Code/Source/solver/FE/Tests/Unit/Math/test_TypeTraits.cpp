/**
 * @file test_TypeTraits.cpp
 * @brief Unit tests for TypeTraits.h - type traits and SFINAE helpers
 */

#include <gtest/gtest.h>
#include "FE/Math/TypeTraits.h"
#include "FE/Math/Vector.h"
#include "FE/Math/Matrix.h"
#include "FE/Math/Tensor.h"
#include "FE/Math/VectorExpr.h"
#include "FE/Math/MatrixExpr.h"

using namespace svmp::FE::math;

// Test fixture for TypeTraits tests
class TypeTraitsTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

namespace {
struct HasSizeMember {
    std::size_t size() const { return 0; }
};

struct HasSubscriptOp {
    int operator[](std::size_t) const { return 0; }
};

struct HasCallOp {
    int operator()(std::size_t, std::size_t) const { return 0; }
};

struct HasNothing {};
}  // namespace

// =============================================================================
// Vector Type Traits Tests
// =============================================================================

TEST_F(TypeTraitsTest, IsVector) {
    // Test Vector types
    EXPECT_TRUE((is_vector_v<Vector<double, 3>>));
    EXPECT_TRUE((is_vector_v<Vector<float, 2>>));
    EXPECT_TRUE((is_vector_v<Vector<int, 4>>));

    // Test non-Vector types
    EXPECT_FALSE(is_vector_v<int>);
    EXPECT_FALSE(is_vector_v<double>);
    EXPECT_FALSE((is_vector_v<Matrix<double, 3, 3>>));
    EXPECT_FALSE(is_vector_v<std::vector<double>>);
}

TEST_F(TypeTraitsTest, IsVectorLike) {
    using Vec3 = Vector<double, 3>;

    // Test vector-like types (Vector or VectorExpr)
    EXPECT_TRUE(is_vector_like_v<Vec3>);
    EXPECT_TRUE((is_vector_like_v<Vector<float, 4>>));

    // Test non-vector-like types
    EXPECT_FALSE(is_vector_like_v<double>);
    EXPECT_FALSE((is_vector_like_v<Matrix<double, 2, 2>>));
}

// =============================================================================
// Matrix Type Traits Tests
// =============================================================================

TEST_F(TypeTraitsTest, IsMatrix) {
    // Test Matrix types
    EXPECT_TRUE((is_matrix_v<Matrix<double, 3, 3>>));
    EXPECT_TRUE((is_matrix_v<Matrix<float, 2, 4>>));
    EXPECT_TRUE((is_matrix_v<Matrix<int, 4, 1>>));

    // Test non-Matrix types
    EXPECT_FALSE(is_matrix_v<int>);
    EXPECT_FALSE((is_matrix_v<Vector<double, 3>>));
    EXPECT_FALSE((is_matrix_v<std::array<double, 9>>));
}

TEST_F(TypeTraitsTest, IsMatrixLike) {
    using Mat3 = Matrix<double, 3, 3>;

    // Test matrix-like types (Matrix or MatrixExpr)
    EXPECT_TRUE(is_matrix_like_v<Mat3>);
    EXPECT_TRUE((is_matrix_like_v<Matrix<float, 2, 4>>));

    // Test non-matrix-like types
    EXPECT_FALSE(is_matrix_like_v<double>);
    EXPECT_FALSE((is_matrix_like_v<Vector<double, 3>>));
}

TEST_F(TypeTraitsTest, IsSquareMatrix) {
    // Test square matrices
    EXPECT_TRUE((is_square_matrix_v<Matrix<double, 2, 2>>));
    EXPECT_TRUE((is_square_matrix_v<Matrix<float, 3, 3>>));
    EXPECT_TRUE((is_square_matrix_v<Matrix<int, 4, 4>>));

    // Test non-square matrices
    EXPECT_FALSE((is_square_matrix_v<Matrix<double, 2, 3>>));
    EXPECT_FALSE((is_square_matrix_v<Matrix<float, 4, 1>>));

    // Test non-matrix types
    EXPECT_FALSE((is_square_matrix_v<Vector<double, 3>>));
    EXPECT_FALSE(is_square_matrix_v<int>);
}

// =============================================================================
// Scalar Type Traits Tests
// =============================================================================

TEST_F(TypeTraitsTest, IsScalar) {
    // Test scalar types
    EXPECT_TRUE(is_scalar_v<double>);
    EXPECT_TRUE(is_scalar_v<float>);
    EXPECT_TRUE(is_scalar_v<int>);
    EXPECT_TRUE(is_scalar_v<long>);

    // Test non-scalar types
    EXPECT_FALSE((is_scalar_v<Vector<double, 3>>));
    EXPECT_FALSE((is_scalar_v<Matrix<float, 2, 2>>));
}

TEST_F(TypeTraitsTest, IsFloatingPoint) {
    // Test floating-point types
    EXPECT_TRUE(is_floating_point_v<double>);
    EXPECT_TRUE(is_floating_point_v<float>);
    EXPECT_TRUE(is_floating_point_v<long double>);

    // Test non-floating-point types
    EXPECT_FALSE(is_floating_point_v<int>);
    EXPECT_FALSE((is_floating_point_v<Vector<double, 3>>));
}

TEST_F(TypeTraitsTest, IsIntegral) {
    // Test integral types
    EXPECT_TRUE(is_integral_v<int>);
    EXPECT_TRUE(is_integral_v<long>);
    EXPECT_TRUE(is_integral_v<std::size_t>);
    EXPECT_TRUE(is_integral_v<bool>);

    // Test non-integral types
    EXPECT_FALSE(is_integral_v<double>);
    EXPECT_FALSE(is_integral_v<float>);
    EXPECT_FALSE((is_integral_v<Vector<int, 3>>));
}

// =============================================================================
// Dimension Extraction Tests
// =============================================================================

TEST_F(TypeTraitsTest, VectorDimension) {
    // Test vector dimension extraction
    EXPECT_EQ((vector_dimension_v<Vector<double, 3>>), 3);
    EXPECT_EQ((vector_dimension_v<Vector<float, 2>>), 2);
    EXPECT_EQ((vector_dimension_v<Vector<int, 4>>), 4);
    EXPECT_EQ((vector_dimension_v<Vector<double, 10>>), 10);
}

TEST_F(TypeTraitsTest, MatrixDimensions) {
    // Test matrix dimension extraction
    using Mat23 = Matrix<double, 2, 3>;
    using Mat34 = Matrix<float, 3, 4>;

    EXPECT_EQ(matrix_rows_v<Mat23>, 2);
    EXPECT_EQ(matrix_cols_v<Mat23>, 3);
    EXPECT_EQ(matrix_rows_v<Mat34>, 3);
    EXPECT_EQ(matrix_cols_v<Mat34>, 4);
}

// =============================================================================
// Common Type Deduction Tests
// =============================================================================

TEST_F(TypeTraitsTest, CommonType) {
    // Test common type deduction
    static_assert(std::is_same_v<common_type_t<int, double>, double>);
    static_assert(std::is_same_v<common_type_t<float, double>, double>);
    static_assert(std::is_same_v<common_type_t<int, int>, int>);
}

// =============================================================================
// Thread Safety Type Traits Tests
// =============================================================================

TEST_F(TypeTraitsTest, ThreadSafetyTraits) {
    // Test if types are trivially copyable (important for thread safety)
    EXPECT_TRUE((std::is_trivially_copyable_v<Vector<double, 3>>));
    EXPECT_TRUE((std::is_trivially_copyable_v<Matrix<float, 2, 2>>));

    // Test if types are standard layout (important for C interop)
    EXPECT_TRUE((std::is_standard_layout_v<Vector<double, 3>>));
    EXPECT_TRUE((std::is_standard_layout_v<Matrix<float, 2, 2>>));
}

// =============================================================================
// Tensor Type Traits Tests
// =============================================================================

TEST_F(TypeTraitsTest, IsTensor) {
    EXPECT_TRUE((is_tensor_v<Tensor2<double, 3>>));
    EXPECT_TRUE((is_tensor_v<Tensor4<float, 3>>));
    EXPECT_TRUE((is_tensor_v<TensorBase<double, 3, 3>>));

    EXPECT_FALSE((is_tensor_v<Vector<double, 3>>));
    EXPECT_FALSE((is_tensor_v<Matrix<double, 3, 3>>));
    EXPECT_FALSE(is_tensor_v<int>);
}

TEST_F(TypeTraitsTest, ScalarTypeExtraction) {
    static_assert(std::is_same_v<scalar_type_t<double>, double>);
    static_assert(std::is_same_v<scalar_type_t<Vector<double, 3>>, double>);
    static_assert(std::is_same_v<scalar_type_t<Matrix<float, 2, 4>>, float>);
    static_assert(std::is_same_v<scalar_type_t<Tensor2<double, 3>>, double>);
    static_assert(std::is_same_v<scalar_type_t<Tensor4<float, 3>>, float>);
    static_assert(std::is_same_v<scalar_type_t<TensorBase<double, 4, 2>>, double>);
}

// =============================================================================
// Expression Template Detection Tests
// =============================================================================

TEST_F(TypeTraitsTest, ExpressionTemplateTraits) {
    Vector<double, 3> a{1.0, 2.0, 3.0};
    Vector<double, 3> b{4.0, 5.0, 6.0};
    auto vexpr_add = a + b;
    auto vexpr_sub = a - b;
    auto vexpr_scale1 = a * 2.0;
    auto vexpr_scale2 = 2.0 * a;
    auto vexpr_neg = -a;
    auto vexpr_div = a / 2.0;

    EXPECT_TRUE((is_vector_like_v<decltype(vexpr_add)>));
    EXPECT_TRUE((is_vector_expr_v<decltype(vexpr_add)>));
    EXPECT_TRUE((is_expression_template_v<decltype(vexpr_add)>));

    EXPECT_TRUE((is_vector_like_v<decltype(vexpr_sub)>));
    EXPECT_TRUE((is_vector_expr_v<decltype(vexpr_sub)>));
    EXPECT_TRUE((is_expression_template_v<decltype(vexpr_sub)>));

    EXPECT_TRUE((is_vector_like_v<decltype(vexpr_scale1)>));
    EXPECT_TRUE((is_vector_expr_v<decltype(vexpr_scale1)>));
    EXPECT_TRUE((is_expression_template_v<decltype(vexpr_scale1)>));

    EXPECT_TRUE((is_vector_like_v<decltype(vexpr_scale2)>));
    EXPECT_TRUE((is_vector_expr_v<decltype(vexpr_scale2)>));
    EXPECT_TRUE((is_expression_template_v<decltype(vexpr_scale2)>));

    EXPECT_TRUE((is_vector_like_v<decltype(vexpr_neg)>));
    EXPECT_TRUE((is_vector_expr_v<decltype(vexpr_neg)>));
    EXPECT_TRUE((is_expression_template_v<decltype(vexpr_neg)>));

    EXPECT_TRUE((is_vector_like_v<decltype(vexpr_div)>));
    EXPECT_TRUE((is_vector_expr_v<decltype(vexpr_div)>));
    EXPECT_TRUE((is_expression_template_v<decltype(vexpr_div)>));
    EXPECT_FALSE((is_vector_expr_v<Vector<double, 3>>));

    Matrix<double, 2, 2> A{{1.0, 2.0},
                           {3.0, 4.0}};
    auto mexpr_add = A + A;
    auto mexpr_sub = A - A;
    auto mexpr_scale1 = A * 2.0;
    auto mexpr_scale2 = 2.0 * A;
    auto mexpr_neg = -A;
    auto mexpr_div = A / 2.0;

    EXPECT_TRUE((is_matrix_like_v<decltype(mexpr_add)>));
    EXPECT_TRUE((is_matrix_expr_v<decltype(mexpr_add)>));
    EXPECT_TRUE((is_expression_template_v<decltype(mexpr_add)>));

    EXPECT_TRUE((is_matrix_like_v<decltype(mexpr_sub)>));
    EXPECT_TRUE((is_matrix_expr_v<decltype(mexpr_sub)>));
    EXPECT_TRUE((is_expression_template_v<decltype(mexpr_sub)>));

    EXPECT_TRUE((is_matrix_like_v<decltype(mexpr_scale1)>));
    EXPECT_TRUE((is_matrix_expr_v<decltype(mexpr_scale1)>));
    EXPECT_TRUE((is_expression_template_v<decltype(mexpr_scale1)>));

    EXPECT_TRUE((is_matrix_like_v<decltype(mexpr_scale2)>));
    EXPECT_TRUE((is_matrix_expr_v<decltype(mexpr_scale2)>));
    EXPECT_TRUE((is_expression_template_v<decltype(mexpr_scale2)>));

    EXPECT_TRUE((is_matrix_like_v<decltype(mexpr_neg)>));
    EXPECT_TRUE((is_matrix_expr_v<decltype(mexpr_neg)>));
    EXPECT_TRUE((is_expression_template_v<decltype(mexpr_neg)>));

    EXPECT_TRUE((is_matrix_like_v<decltype(mexpr_div)>));
    EXPECT_TRUE((is_matrix_expr_v<decltype(mexpr_div)>));
    EXPECT_TRUE((is_expression_template_v<decltype(mexpr_div)>));
    EXPECT_FALSE((is_matrix_expr_v<Matrix<double, 2, 2>>));
}

// =============================================================================
// Compatibility Traits Tests
// =============================================================================

TEST_F(TypeTraitsTest, CompatibilityForMultiplication) {
    EXPECT_TRUE((are_compatible_for_mult_v<Matrix<double, 2, 3>, Vector<double, 3>>));
    EXPECT_FALSE((are_compatible_for_mult_v<Matrix<double, 2, 3>, Vector<double, 2>>));

    EXPECT_TRUE((are_compatible_matrices_for_mult_v<Matrix<double, 2, 3>, Matrix<double, 3, 4>>));
    EXPECT_FALSE((are_compatible_matrices_for_mult_v<Matrix<double, 2, 3>, Matrix<double, 2, 3>>));
}

// =============================================================================
// Detection Traits Tests
// =============================================================================

TEST_F(TypeTraitsTest, DetectionTraits) {
    EXPECT_TRUE(has_size_v<HasSizeMember>);
    EXPECT_FALSE(has_size_v<HasNothing>);

    EXPECT_TRUE(has_subscript_v<HasSubscriptOp>);
    EXPECT_FALSE(has_subscript_v<HasNothing>);

    EXPECT_TRUE(has_call_operator_v<HasCallOp>);
    EXPECT_FALSE(has_call_operator_v<HasNothing>);
}

// =============================================================================
// SIMD Traits Tests
// =============================================================================

TEST_F(TypeTraitsTest, SimdTraits) {
    EXPECT_TRUE(supports_simd_v<float>);
    EXPECT_TRUE(supports_simd_v<double>);
    EXPECT_FALSE(supports_simd_v<int>);

    EXPECT_EQ(simd_alignment_v<float>, 32u);
    EXPECT_EQ(simd_alignment_v<double>, 32u);
    EXPECT_EQ(simd_alignment_v<int>, alignof(int));
}

TEST_F(TypeTraitsTest, CommonFloatingPointType) {
    static_assert(std::is_same_v<common_floating_point_t<int, float>, double>);
    static_assert(std::is_same_v<common_floating_point_t<float, double>, double>);
    static_assert(std::is_same_v<common_floating_point_t<float, float>, float>);
}
