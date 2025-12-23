/**
 * @file test_DofConstraints.cpp
 * @brief Unit tests for DofConstraints
 */

#include <gtest/gtest.h>

#include "FE/Dofs/DofConstraints.h"
#include "FE/Core/FEException.h"

#include <algorithm>
#include <cmath>
#include <vector>

using svmp::FE::FEException;
using svmp::FE::GlobalIndex;
using svmp::FE::dofs::AbstractMatrix;
using svmp::FE::dofs::AbstractVector;
using svmp::FE::dofs::ConstraintType;
using svmp::FE::dofs::DofConstraints;

namespace {

class DenseTestMatrix : public AbstractMatrix {
public:
    explicit DenseTestMatrix(GlobalIndex n)
        : n_(n), a_(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0) {}

    void addValues(std::span<const GlobalIndex> rows,
                   std::span<const GlobalIndex> cols,
                   std::span<const double> values) override {
        const auto n_rows = rows.size();
        const auto n_cols = cols.size();
        for (std::size_t i = 0; i < n_rows; ++i) {
            for (std::size_t j = 0; j < n_cols; ++j) {
                const auto r = rows[i];
                const auto c = cols[j];
                if (r < 0 || c < 0 || r >= n_ || c >= n_) continue;
                a_[static_cast<std::size_t>(r) * static_cast<std::size_t>(n_) + static_cast<std::size_t>(c)] +=
                    values[i * n_cols + j];
            }
        }
    }

    void setRowToIdentity(GlobalIndex row) override {
        if (row < 0 || row >= n_) return;
        for (GlobalIndex c = 0; c < n_; ++c) {
            a_[static_cast<std::size_t>(row) * static_cast<std::size_t>(n_) + static_cast<std::size_t>(c)] = 0.0;
        }
        a_[static_cast<std::size_t>(row) * static_cast<std::size_t>(n_) + static_cast<std::size_t>(row)] = 1.0;
    }

    void zeroRow(GlobalIndex row) override {
        if (row < 0 || row >= n_) return;
        for (GlobalIndex c = 0; c < n_; ++c) {
            a_[static_cast<std::size_t>(row) * static_cast<std::size_t>(n_) + static_cast<std::size_t>(c)] = 0.0;
        }
    }

    [[nodiscard]] GlobalIndex numRows() const override { return n_; }

    [[nodiscard]] double operator()(GlobalIndex r, GlobalIndex c) const {
        return a_[static_cast<std::size_t>(r) * static_cast<std::size_t>(n_) + static_cast<std::size_t>(c)];
    }

private:
    GlobalIndex n_;
    std::vector<double> a_;
};

class DenseTestVector : public AbstractVector {
public:
    explicit DenseTestVector(GlobalIndex n) : x_(static_cast<std::size_t>(n), 0.0) {}

    void setValues(std::span<const GlobalIndex> indices,
                   std::span<const double> values) override {
        for (std::size_t i = 0; i < indices.size() && i < values.size(); ++i) {
            const auto idx = indices[i];
            if (idx < 0 || static_cast<std::size_t>(idx) >= x_.size()) continue;
            x_[static_cast<std::size_t>(idx)] = values[i];
        }
    }

    void addValues(std::span<const GlobalIndex> indices,
                   std::span<const double> values) override {
        for (std::size_t i = 0; i < indices.size() && i < values.size(); ++i) {
            const auto idx = indices[i];
            if (idx < 0 || static_cast<std::size_t>(idx) >= x_.size()) continue;
            x_[static_cast<std::size_t>(idx)] += values[i];
        }
    }

    [[nodiscard]] double getValue(GlobalIndex index) const override {
        if (index < 0 || static_cast<std::size_t>(index) >= x_.size()) return 0.0;
        return x_[static_cast<std::size_t>(index)];
    }

    [[nodiscard]] GlobalIndex size() const override {
        return static_cast<GlobalIndex>(x_.size());
    }

    [[nodiscard]] double& operator[](GlobalIndex i) { return x_[static_cast<std::size_t>(i)]; }
    [[nodiscard]] double operator[](GlobalIndex i) const { return x_[static_cast<std::size_t>(i)]; }

private:
    std::vector<double> x_;
};

} // namespace

TEST(DofConstraints, AddAndQueryConstraints) {
    DofConstraints constraints;
    constraints.addDirichletBC(0, 2.0);
    constraints.addPeriodicBC(/*master=*/1, /*slave=*/3);

    EXPECT_EQ(constraints.numConstraints(), 2u);
    EXPECT_TRUE(constraints.isConstrained(0));
    EXPECT_TRUE(constraints.isConstrained(3));
    EXPECT_FALSE(constraints.isConstrained(1));

    auto v = constraints.getDirichletValue(0);
    ASSERT_TRUE(v.has_value());
    EXPECT_DOUBLE_EQ(*v, 2.0);

    auto line = constraints.getConstraintLine(3);
    ASSERT_TRUE(line.has_value());
    EXPECT_EQ(line->type, ConstraintType::Periodic);
    ASSERT_EQ(line->entries.size(), 1u);
    EXPECT_EQ(line->entries[0].dof, 1);
    EXPECT_DOUBLE_EQ(line->entries[0].coefficient, 1.0);
}

TEST(DofConstraints, CloseComputesTransitiveClosure) {
    DofConstraints constraints;

    // 1 = 3*2 + 1
    {
        const std::vector<GlobalIndex> dofs = {2};
        const std::vector<double> coeff = {3.0};
        constraints.addLinearConstraint(1, dofs, coeff, /*inhom=*/1.0);
    }
    // 0 = 2*1 + 0
    {
        const std::vector<GlobalIndex> dofs = {1};
        const std::vector<double> coeff = {2.0};
        constraints.addLinearConstraint(0, dofs, coeff, /*inhom=*/0.0);
    }

    constraints.close();
    EXPECT_TRUE(constraints.isClosed());

    auto line0 = constraints.getConstraintLine(0);
    ASSERT_TRUE(line0.has_value());
    ASSERT_EQ(line0->entries.size(), 1u);
    EXPECT_EQ(line0->entries[0].dof, 2);
    EXPECT_NEAR(line0->entries[0].coefficient, 6.0, 1e-14);
    EXPECT_NEAR(line0->inhomogeneity, 2.0, 1e-14);
}

TEST(DofConstraints, CloseThrowsOnCircularDependency) {
    DofConstraints constraints;
    constraints.addPeriodicBC(/*master=*/0, /*slave=*/1);
    constraints.addPeriodicBC(/*master=*/1, /*slave=*/0); // cycle
    EXPECT_THROW(constraints.close(), FEException);
}

TEST(DofConstraints, ApplySolutionConstraints) {
    DofConstraints constraints;
    constraints.addDirichletBC(2, 5.0);

    // 0 = 2*1 + 1
    {
        const std::vector<GlobalIndex> dofs = {1};
        const std::vector<double> coeff = {2.0};
        constraints.addLinearConstraint(0, dofs, coeff, /*inhom=*/1.0);
    }

    constraints.close();

    DenseTestVector sol(3);
    sol[1] = 3.0;
    sol[0] = 0.0;
    sol[2] = 0.0;

    constraints.applySolutionConstraints(sol);

    EXPECT_NEAR(sol[0], 7.0, 1e-14);
    EXPECT_NEAR(sol[2], 5.0, 1e-14);
}

TEST(DofConstraints, BuildConstraintMatrixHasIdentityRowsForUnconstrained) {
    DofConstraints constraints;
    constraints.addDirichletBC(0, 2.0);
    constraints.close();

    std::vector<GlobalIndex> row_offsets;
    std::vector<GlobalIndex> col_indices;
    std::vector<double> values;

    constraints.buildConstraintMatrix(/*n_total_dofs=*/3, row_offsets, col_indices, values);

    ASSERT_EQ(row_offsets.size(), 4u);
    EXPECT_EQ(row_offsets.front(), 0);
    EXPECT_EQ(row_offsets.back(), static_cast<GlobalIndex>(col_indices.size()));

    // Row 1 and 2 should contain identity entries (since they are unconstrained).
    bool saw_row1_identity = false;
    bool saw_row2_identity = false;
    for (std::size_t r = 0; r < 3; ++r) {
        const auto begin = static_cast<std::size_t>(row_offsets[r]);
        const auto end = static_cast<std::size_t>(row_offsets[r + 1]);
        for (std::size_t k = begin; k < end; ++k) {
            if (r == 1 && col_indices[k] == 0 && std::abs(values[k] - 1.0) < 1e-14) {
                saw_row1_identity = true;
            }
            if (r == 2 && col_indices[k] == 1 && std::abs(values[k] - 1.0) < 1e-14) {
                saw_row2_identity = true;
            }
        }
    }
    EXPECT_TRUE(saw_row1_identity);
    EXPECT_TRUE(saw_row2_identity);
}

