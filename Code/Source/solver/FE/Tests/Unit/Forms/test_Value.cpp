/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_Value.cpp
 * @brief Unit tests for FE/Forms Value<T> inline/dynamic container
 */

#include <gtest/gtest.h>

#include "Core/Types.h"
#include "Forms/Value.h"

#include <array>
#include <cstddef>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

TEST(ValueTest, ValueKindTransitions)
{
    Value<Real> val;

    val.kind = Value<Real>::Kind::Scalar;
    EXPECT_EQ(val.vectorSize(), 0u);
    EXPECT_EQ(val.matrixRows(), 0u);
    EXPECT_EQ(val.matrixCols(), 0u);
    EXPECT_EQ(val.tensor3Dim0(), 0u);
    EXPECT_EQ(val.tensor3Dim1(), 0u);
    EXPECT_EQ(val.tensor3Dim2(), 0u);

    val.kind = Value<Real>::Kind::Vector;
    EXPECT_EQ(val.vectorSize(), 3u);
    EXPECT_EQ(val.matrixRows(), 0u);
    EXPECT_EQ(val.tensor3Dim0(), 0u);

    val.kind = Value<Real>::Kind::Matrix;
    EXPECT_EQ(val.vectorSize(), 0u);
    EXPECT_EQ(val.matrixRows(), 3u);
    EXPECT_EQ(val.matrixCols(), 3u);

    val.kind = Value<Real>::Kind::SymmetricMatrix;
    EXPECT_EQ(val.matrixRows(), 3u);
    EXPECT_EQ(val.matrixCols(), 3u);

    val.kind = Value<Real>::Kind::SkewMatrix;
    EXPECT_EQ(val.matrixRows(), 3u);
    EXPECT_EQ(val.matrixCols(), 3u);

    val.kind = Value<Real>::Kind::Tensor3;
    EXPECT_EQ(val.tensor3Dim0(), 3u);
    EXPECT_EQ(val.tensor3Dim1(), 3u);
    EXPECT_EQ(val.tensor3Dim2(), 3u);
    EXPECT_EQ(val.tensor3Size(), 27u);

    val.kind = Value<Real>::Kind::Tensor4;
    EXPECT_EQ(val.vectorSize(), 0u);
    EXPECT_EQ(val.matrixRows(), 0u);
    EXPECT_EQ(val.tensor3Dim0(), 0u);
}

TEST(ValueTest, VectorResizeInlineStorage)
{
    Value<Real> val;
    val.kind = Value<Real>::Kind::Vector;

    for (std::size_t n : {1u, 2u, 3u}) {
        SCOPED_TRACE(::testing::Message() << "n=" << n);
        val.resizeVector(n);
        EXPECT_TRUE(val.v_dyn.empty());
        EXPECT_EQ(val.vectorSize(), n);

        auto span = val.vectorSpan();
        EXPECT_EQ(span.size(), n);
        EXPECT_EQ(span.data(), val.v.data());

        for (std::size_t i = 0; i < n; ++i) {
            const Real value = static_cast<Real>(10.0 + i);
            val.vectorAt(i) = value;
            EXPECT_DOUBLE_EQ(val.v[i], value);
        }
    }
}

TEST(ValueTest, VectorResizeDynamicStorage)
{
    Value<Real> val;
    val.kind = Value<Real>::Kind::Vector;

    for (std::size_t n : {4u, 10u, 100u}) {
        SCOPED_TRACE(::testing::Message() << "n=" << n);
        val.resizeVector(n);
        EXPECT_EQ(val.v_dyn.size(), n);
        EXPECT_EQ(val.vectorSize(), n);

        auto span = val.vectorSpan();
        EXPECT_EQ(span.size(), n);
        EXPECT_EQ(span.data(), val.v_dyn.data());

        val.vectorAt(n - 1) = 3.14;
        EXPECT_DOUBLE_EQ(val.v_dyn[n - 1], 3.14);
    }
}

TEST(ValueTest, MatrixResizeInlineStorage)
{
    Value<Real> val;
    val.kind = Value<Real>::Kind::Matrix;

    val.resizeMatrix(3, 3);
    EXPECT_TRUE(val.m_dyn.empty());
    EXPECT_EQ(val.matrixRows(), 3u);
    EXPECT_EQ(val.matrixCols(), 3u);

    val.matrixAt(1, 2) = 7.0;
    EXPECT_DOUBLE_EQ(val.m[1][2], 7.0);
    EXPECT_EQ(&val.matrixAt(1, 2), &val.m[1][2]);

    val.resizeMatrix(2, 3);
    EXPECT_TRUE(val.m_dyn.empty());
    EXPECT_EQ(val.matrixRows(), 2u);
    EXPECT_EQ(val.matrixCols(), 3u);

    val.matrixAt(0, 1) = -2.0;
    EXPECT_DOUBLE_EQ(val.m[0][1], -2.0);
}

TEST(ValueTest, MatrixResizeDynamicStorage)
{
    Value<Real> val;
    val.kind = Value<Real>::Kind::Matrix;

    val.resizeMatrix(4, 4);
    EXPECT_EQ(val.m_dyn.size(), 16u);
    EXPECT_EQ(val.matrixRows(), 4u);
    EXPECT_EQ(val.matrixCols(), 4u);

    val.matrixAt(3, 1) = 5.0;
    EXPECT_DOUBLE_EQ(val.m_dyn[3u * 4u + 1u], 5.0);

    val.resizeMatrix(3, 5);
    EXPECT_EQ(val.m_dyn.size(), 15u);
    EXPECT_EQ(val.matrixRows(), 3u);
    EXPECT_EQ(val.matrixCols(), 5u);

    val.matrixAt(2, 4) = 9.0;
    EXPECT_DOUBLE_EQ(val.m_dyn[2u * 5u + 4u], 9.0);
}

TEST(ValueTest, MatrixAtAccessInlineVsDynamic)
{
    {
        Value<Real> val;
        val.kind = Value<Real>::Kind::Matrix;
        val.resizeMatrix(3, 3);

        val.matrixAt(2, 1) = 1.25;
        EXPECT_DOUBLE_EQ(val.m[2][1], 1.25);
        EXPECT_EQ(&val.matrixAt(2, 1), &val.m[2][1]);
    }

    {
        Value<Real> val;
        val.kind = Value<Real>::Kind::Matrix;
        val.resizeMatrix(3, 5);

        val.matrixAt(2, 4) = -4.5;
        EXPECT_DOUBLE_EQ(val.m_dyn[2u * 5u + 4u], -4.5);
        EXPECT_EQ(&val.matrixAt(2, 4), &val.m_dyn[2u * 5u + 4u]);
    }
}

TEST(ValueTest, Tensor3ResizeInlineStorage)
{
    Value<Real> val;
    val.kind = Value<Real>::Kind::Tensor3;
    val.resizeTensor3(3, 3, 3);

    EXPECT_TRUE(val.t3_dyn.empty());
    EXPECT_EQ(val.tensor3Size(), 27u);
    EXPECT_EQ(val.tensor3Span().data(), val.t3.data());

    val.tensor3At(2, 1, 0) = 6.0;
    const std::size_t idx = (2u * 3u + 1u) * 3u + 0u;
    EXPECT_DOUBLE_EQ(val.t3[idx], 6.0);
}

TEST(ValueTest, Tensor3ResizeDynamicStorage)
{
    Value<Real> val;
    val.kind = Value<Real>::Kind::Tensor3;
    val.resizeTensor3(4, 4, 4);

    EXPECT_EQ(val.t3_dyn.size(), 64u);
    EXPECT_EQ(val.tensor3Size(), 64u);
    EXPECT_EQ(val.tensor3Span().data(), val.t3_dyn.data());
}

TEST(ValueTest, Tensor3AtIndexing)
{
    {
        Value<Real> val;
        val.kind = Value<Real>::Kind::Tensor3;
        val.resizeTensor3(3, 2, 3);

        auto span = val.tensor3Span();
        for (std::size_t idx = 0; idx < span.size(); ++idx) {
            span[idx] = static_cast<Real>(idx);
        }

        const std::size_t d1 = val.tensor3Dim1();
        const std::size_t d2 = val.tensor3Dim2();
        for (std::size_t i = 0; i < val.tensor3Dim0(); ++i) {
            for (std::size_t j = 0; j < val.tensor3Dim1(); ++j) {
                for (std::size_t k = 0; k < val.tensor3Dim2(); ++k) {
                    const std::size_t idx = (i * d1 + j) * d2 + k;
                    EXPECT_DOUBLE_EQ(val.tensor3At(i, j, k), static_cast<Real>(idx));
                }
            }
        }
    }

    {
        Value<Real> val;
        val.kind = Value<Real>::Kind::Tensor3;
        val.resizeTensor3(4, 2, 3);

        auto span = val.tensor3Span();
        for (std::size_t idx = 0; idx < span.size(); ++idx) {
            span[idx] = static_cast<Real>(100.0 + static_cast<Real>(idx));
        }

        const std::size_t d1 = val.tensor3Dim1();
        const std::size_t d2 = val.tensor3Dim2();
        for (std::size_t i = 0; i < val.tensor3Dim0(); ++i) {
            for (std::size_t j = 0; j < val.tensor3Dim1(); ++j) {
                for (std::size_t k = 0; k < val.tensor3Dim2(); ++k) {
                    const std::size_t idx = (i * d1 + j) * d2 + k;
                    EXPECT_DOUBLE_EQ(val.tensor3At(i, j, k), 100.0 + static_cast<Real>(idx));
                }
            }
        }
    }
}

TEST(ValueTest, Tensor4FixedStorage)
{
    Value<Real> val;
    val.kind = Value<Real>::Kind::Tensor4;
    for (std::size_t i = 0; i < val.t4.size(); ++i) {
        val.t4[i] = static_cast<Real>(i);
    }

    for (std::size_t i = 0; i < val.t4.size(); ++i) {
        EXPECT_DOUBLE_EQ(val.t4[i], static_cast<Real>(i));
    }
}

TEST(ValueTest, ZeroSizedVector)
{
    Value<Real> val;
    val.kind = Value<Real>::Kind::Vector;
    val.resizeVector(0);

    EXPECT_EQ(val.vectorSize(), 0u);
    EXPECT_TRUE(val.v_dyn.empty());
    EXPECT_TRUE(val.vectorSpan().empty());
}

TEST(ValueTest, DimensionQueryConsistency)
{
    {
        Value<Real> val;
        val.kind = Value<Real>::Kind::Vector;
        val.resizeVector(7);
        EXPECT_EQ(val.vectorSize(), 7u);
    }

    {
        Value<Real> val;
        val.kind = Value<Real>::Kind::Matrix;
        val.resizeMatrix(2, 5);
        EXPECT_EQ(val.matrixRows(), 2u);
        EXPECT_EQ(val.matrixCols(), 5u);
    }

    {
        Value<Real> val;
        val.kind = Value<Real>::Kind::Tensor3;
        val.resizeTensor3(2, 3, 4);
        EXPECT_EQ(val.tensor3Dim0(), 2u);
        EXPECT_EQ(val.tensor3Dim1(), 3u);
        EXPECT_EQ(val.tensor3Dim2(), 4u);
        EXPECT_EQ(val.tensor3Size(), 24u);
    }
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
