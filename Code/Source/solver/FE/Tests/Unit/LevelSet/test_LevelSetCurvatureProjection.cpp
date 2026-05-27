#include "LevelSet/LevelSetCurvatureProjection.h"

#include "Assembly/Assembler.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;

class StructuredQuadMeshAccess final : public FE::assembly::IMeshAccess {
public:
    StructuredQuadMeshAccess(int nx, int ny, FE::Real h)
        : nx_(nx), ny_(ny), h_(h)
    {
        for (int j = 0; j <= ny_; ++j) {
            for (int i = 0; i <= nx_; ++i) {
                nodes_.push_back({{
                    (static_cast<FE::Real>(i) -
                     static_cast<FE::Real>(nx_) * FE::Real{0.5}) * h_,
                    (static_cast<FE::Real>(j) -
                     static_cast<FE::Real>(ny_) * FE::Real{0.5}) * h_,
                    FE::Real{0.0},
                }});
            }
        }
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override
    {
        return static_cast<FE::GlobalIndex>(nx_ * ny_);
    }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override
    {
        return numCells();
    }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numVertices() const override
    {
        return static_cast<FE::GlobalIndex>(nodes_.size());
    }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override
    {
        return geometry_revision_;
    }
    [[nodiscard]] std::uint64_t topologyRevision() const override
    {
        return topology_revision_;
    }
    [[nodiscard]] std::uint64_t ownershipRevision() const override
    {
        return 1u;
    }
    [[nodiscard]] std::uint64_t numberingRevision() const override
    {
        return 1u;
    }
    [[nodiscard]] std::uint64_t coordinateConfigurationKey() const override
    {
        return 1u;
    }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex) const override { return true; }
    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex) const override
    {
        return FE::ElementType::Quad4;
    }

    void translate(FE::Real dx, FE::Real dy)
    {
        for (auto& node : nodes_) {
            node[0] += dx;
            node[1] += dy;
        }
        ++geometry_revision_;
    }

    void getCellNodes(FE::GlobalIndex cell_id,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        const int cell = static_cast<int>(cell_id);
        if (cell < 0 || cell >= nx_ * ny_) {
            throw std::out_of_range("invalid cell");
        }
        const int i = cell % nx_;
        const int j = cell / nx_;
        nodes = {
            vertex(i, j),
            vertex(i + 1, j),
            vertex(i + 1, j + 1),
            vertex(i, j + 1),
        };
    }

    [[nodiscard]] std::array<FE::Real, 3> getNodeCoordinates(
        FE::GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(
        FE::GlobalIndex cell_id,
        std::vector<std::array<FE::Real, 3>>& coords) const override
    {
        std::vector<FE::GlobalIndex> nodes;
        getCellNodes(cell_id, nodes);
        coords.clear();
        for (const auto node : nodes) {
            coords.push_back(getNodeCoordinates(node));
        }
    }

    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(FE::GlobalIndex,
                                                   FE::GlobalIndex) const override
    {
        return 0;
    }
    [[nodiscard]] int getBoundaryFaceMarker(FE::GlobalIndex) const override
    {
        return -1;
    }
    [[nodiscard]] std::pair<FE::GlobalIndex, FE::GlobalIndex>
    getInteriorFaceCells(FE::GlobalIndex) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        for (FE::GlobalIndex c = 0; c < numCells(); ++c) {
            callback(c);
        }
    }
    void forEachOwnedCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        forEachCell(std::move(callback));
    }
    void forEachBoundaryFace(
        int,
        std::function<void(FE::GlobalIndex, FE::GlobalIndex)>) const override
    {
    }
    void forEachInteriorFace(
        std::function<void(FE::GlobalIndex, FE::GlobalIndex, FE::GlobalIndex)>)
        const override
    {
    }

private:
    [[nodiscard]] FE::GlobalIndex vertex(int i, int j) const
    {
        return static_cast<FE::GlobalIndex>(j * (nx_ + 1) + i);
    }

    int nx_{0};
    int ny_{0};
    FE::Real h_{1.0};
    std::uint64_t geometry_revision_{1u};
    std::uint64_t topology_revision_{1u};
    std::vector<std::array<FE::Real, 3>> nodes_{};
};

class StructuredHexMeshAccess final : public FE::assembly::IMeshAccess {
public:
    StructuredHexMeshAccess(int nx, int ny, int nz, FE::Real h)
        : nx_(nx), ny_(ny), nz_(nz), h_(h)
    {
        for (int k = 0; k <= nz_; ++k) {
            for (int j = 0; j <= ny_; ++j) {
                for (int i = 0; i <= nx_; ++i) {
                    nodes_.push_back({{
                        (static_cast<FE::Real>(i) -
                         static_cast<FE::Real>(nx_) * FE::Real{0.5}) * h_,
                        (static_cast<FE::Real>(j) -
                         static_cast<FE::Real>(ny_) * FE::Real{0.5}) * h_,
                        (static_cast<FE::Real>(k) -
                         static_cast<FE::Real>(nz_) * FE::Real{0.5}) * h_,
                    }});
                }
            }
        }
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override
    {
        return static_cast<FE::GlobalIndex>(nx_ * ny_ * nz_);
    }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override
    {
        return numCells();
    }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numVertices() const override
    {
        return static_cast<FE::GlobalIndex>(nodes_.size());
    }
    [[nodiscard]] int dimension() const override { return 3; }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex) const override { return true; }
    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex) const override
    {
        return FE::ElementType::Hex8;
    }

    void getCellNodes(FE::GlobalIndex cell_id,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        const int cell = static_cast<int>(cell_id);
        if (cell < 0 || cell >= nx_ * ny_ * nz_) {
            throw std::out_of_range("invalid cell");
        }
        const int i = cell % nx_;
        const int j = (cell / nx_) % ny_;
        const int k = cell / (nx_ * ny_);
        nodes = {
            vertex(i, j, k),
            vertex(i + 1, j, k),
            vertex(i + 1, j + 1, k),
            vertex(i, j + 1, k),
            vertex(i, j, k + 1),
            vertex(i + 1, j, k + 1),
            vertex(i + 1, j + 1, k + 1),
            vertex(i, j + 1, k + 1),
        };
    }

    [[nodiscard]] std::array<FE::Real, 3> getNodeCoordinates(
        FE::GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(
        FE::GlobalIndex cell_id,
        std::vector<std::array<FE::Real, 3>>& coords) const override
    {
        std::vector<FE::GlobalIndex> nodes;
        getCellNodes(cell_id, nodes);
        coords.clear();
        for (const auto node : nodes) {
            coords.push_back(getNodeCoordinates(node));
        }
    }

    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(FE::GlobalIndex,
                                                   FE::GlobalIndex) const override
    {
        return 0;
    }
    [[nodiscard]] int getBoundaryFaceMarker(FE::GlobalIndex) const override
    {
        return -1;
    }
    [[nodiscard]] std::pair<FE::GlobalIndex, FE::GlobalIndex>
    getInteriorFaceCells(FE::GlobalIndex) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        for (FE::GlobalIndex c = 0; c < numCells(); ++c) {
            callback(c);
        }
    }
    void forEachOwnedCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        forEachCell(std::move(callback));
    }
    void forEachBoundaryFace(
        int,
        std::function<void(FE::GlobalIndex, FE::GlobalIndex)>) const override
    {
    }
    void forEachInteriorFace(
        std::function<void(FE::GlobalIndex, FE::GlobalIndex, FE::GlobalIndex)>)
        const override
    {
    }

private:
    [[nodiscard]] FE::GlobalIndex vertex(int i, int j, int k) const
    {
        return static_cast<FE::GlobalIndex>(
            (k * (ny_ + 1) + j) * (nx_ + 1) + i);
    }

    int nx_{0};
    int ny_{0};
    int nz_{0};
    FE::Real h_{1.0};
    std::vector<std::array<FE::Real, 3>> nodes_{};
};

FE::Real curvatureGraphTotalVariation(
    const FE::assembly::IMeshAccess& mesh,
    const std::vector<FE::Real>& values)
{
    std::vector<std::pair<FE::GlobalIndex, FE::GlobalIndex>> edges;
    std::vector<FE::GlobalIndex> nodes;
    mesh.forEachCell([&](FE::GlobalIndex cell) {
        mesh.getCellNodes(cell, nodes);
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            for (std::size_t j = i + 1u; j < nodes.size(); ++j) {
                auto a = nodes[i];
                auto b = nodes[j];
                if (a == b || a < 0 || b < 0) {
                    continue;
                }
                if (b < a) {
                    std::swap(a, b);
                }
                edges.push_back({a, b});
            }
        }
    });
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    FE::Real total = FE::Real{0.0};
    for (const auto& [a, b] : edges) {
        total += std::abs(values[static_cast<std::size_t>(a)] -
                          values[static_cast<std::size_t>(b)]);
    }
    return total;
}

FE::Real maxAbsDifference(const std::vector<FE::Real>& lhs,
                          const std::vector<FE::Real>& rhs)
{
    FE::Real value = FE::Real{0.0};
    const auto n = std::min(lhs.size(), rhs.size());
    for (std::size_t i = 0; i < n; ++i) {
        value = std::max(value, std::abs(lhs[i] - rhs[i]));
    }
    return value;
}

FE::Real circleCurvatureMeanError(int nx, FE::Real h, FE::Real radius)
{
    StructuredQuadMeshAccess mesh(nx, nx, h);
    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()), 0.0);
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        const auto x = mesh.getNodeCoordinates(v);
        phi[static_cast<std::size_t>(v)] =
            std::sqrt(x[0] * x[0] + x[1] * x[1]) - radius;
    }

    level_set::LevelSetCurvatureProjectionOptions options;
    options.max_neighbor_rings = 2;
    std::vector<FE::Real> curvature;
    const auto result = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, options, curvature);
    if (!result.success) {
        throw std::runtime_error(result.diagnostic);
    }

    FE::Real error_sum = 0.0;
    std::size_t samples = 0u;
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        const auto x = mesh.getNodeCoordinates(v);
        const auto r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
        if (r < radius - FE::Real{0.05} || r > radius + FE::Real{0.05}) {
            continue;
        }
        error_sum += std::abs(
            curvature[static_cast<std::size_t>(v)] - FE::Real{1.0} / r);
        ++samples;
    }
    if (samples == 0u) {
        throw std::runtime_error("no circle curvature samples in narrow band");
    }
    return error_sum / static_cast<FE::Real>(samples);
}

} // namespace

TEST(LevelSetCurvatureProjection, RecoversCircleCurvatureFromSignedDistance)
{
    StructuredQuadMeshAccess mesh(/*nx=*/16, /*ny=*/16, /*h=*/0.05);
    constexpr FE::Real radius = 0.30;
    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()), 0.0);
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        const auto x = mesh.getNodeCoordinates(v);
        phi[static_cast<std::size_t>(v)] =
            std::sqrt(x[0] * x[0] + x[1] * x[1]) - radius;
    }

    level_set::LevelSetCurvatureProjectionOptions options;
    options.max_neighbor_rings = 2;
    std::vector<FE::Real> curvature;
    const auto result = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, options, curvature);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(curvature.size(), phi.size());
    EXPECT_GT(result.fitted_vertices, 200u);
    EXPECT_LT(result.zero_fallback_vertices, 12u);
    EXPECT_GE(result.max_normalized_fit_residual, FE::Real{0.0});
    EXPECT_GE(result.mean_normalized_fit_residual, FE::Real{0.0});

    FE::Real error_sum = 0.0;
    std::size_t samples = 0u;
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        const auto x = mesh.getNodeCoordinates(v);
        const auto r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
        if (r < 0.25 || r > 0.35) {
            continue;
        }
        error_sum += std::abs(
            curvature[static_cast<std::size_t>(v)] - FE::Real{1.0} / r);
        ++samples;
    }
    ASSERT_GT(samples, 0u);
    EXPECT_LT(error_sum / static_cast<FE::Real>(samples), 0.08);
}

TEST(LevelSetCurvatureProjection, CircleCurvatureErrorImprovesWithRefinement)
{
    constexpr FE::Real radius = 0.30;
    const auto coarse_error =
        circleCurvatureMeanError(/*nx=*/8, /*h=*/0.10, radius);
    const auto fine_error =
        circleCurvatureMeanError(/*nx=*/16, /*h=*/0.05, radius);

    EXPECT_LT(fine_error, coarse_error);
    EXPECT_LT(fine_error, 0.08);
}

TEST(LevelSetCurvatureProjection, NarrowBandRestrictsRecoveryToInterfaceVertices)
{
    StructuredQuadMeshAccess mesh(/*nx=*/16, /*ny=*/16, /*h=*/0.05);
    constexpr FE::Real radius = 0.30;
    constexpr FE::Real band_width = 0.055;
    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()), 0.0);
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        const auto x = mesh.getNodeCoordinates(v);
        phi[static_cast<std::size_t>(v)] =
            std::sqrt(x[0] * x[0] + x[1] * x[1]) - radius;
    }

    level_set::LevelSetCurvatureProjectionOptions options;
    options.max_neighbor_rings = 2;
    options.narrow_band_width = band_width;
    std::vector<FE::Real> curvature;
    const auto result = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, options, curvature);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(curvature.size(), phi.size());
    EXPECT_DOUBLE_EQ(result.narrow_band_width, band_width);
    EXPECT_GT(result.narrow_band_vertices, 0u);
    EXPECT_GT(result.skipped_far_vertices, 0u);
    EXPECT_EQ(result.narrow_band_vertices + result.skipped_far_vertices,
              result.vertices);
    EXPECT_LT(result.fitted_vertices, result.vertices);

    FE::Real error_sum = 0.0;
    std::size_t samples = 0u;
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        const auto index = static_cast<std::size_t>(v);
        const auto x = mesh.getNodeCoordinates(v);
        const auto r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
        if (std::abs(phi[index]) <= band_width) {
            error_sum += std::abs(curvature[index] - FE::Real{1.0} / r);
            ++samples;
        } else {
            EXPECT_DOUBLE_EQ(curvature[index], FE::Real{0.0});
        }
    }
    ASSERT_GT(samples, 0u);
    EXPECT_LT(error_sum / static_cast<FE::Real>(samples), 0.10);
}

TEST(LevelSetCurvatureProjection, RecoversSphereCurvatureFromSignedDistance)
{
    StructuredHexMeshAccess mesh(/*nx=*/10, /*ny=*/10, /*nz=*/10, /*h=*/0.06);
    constexpr FE::Real radius = 0.30;
    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()), 0.0);
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        const auto x = mesh.getNodeCoordinates(v);
        phi[static_cast<std::size_t>(v)] =
            std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) - radius;
    }

    level_set::LevelSetCurvatureProjectionOptions options;
    options.max_neighbor_rings = 2;
    std::vector<FE::Real> curvature;
    const auto result = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, options, curvature);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(curvature.size(), phi.size());
    EXPECT_GT(result.fitted_vertices, 900u);
    EXPECT_EQ(result.zero_fallback_vertices, 0u);
    EXPECT_GE(result.max_normalized_fit_residual, FE::Real{0.0});
    EXPECT_GE(result.mean_normalized_fit_residual, FE::Real{0.0});

    FE::Real error_sum = 0.0;
    std::size_t samples = 0u;
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        const auto x = mesh.getNodeCoordinates(v);
        const auto r = std::sqrt(
            x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
        if (r < 0.24 || r > 0.36) {
            continue;
        }
        error_sum += std::abs(
            curvature[static_cast<std::size_t>(v)] - FE::Real{2.0} / r);
        ++samples;
    }
    ASSERT_GT(samples, 0u);
    EXPECT_LT(error_sum / static_cast<FE::Real>(samples), 0.35);
}

TEST(LevelSetCurvatureProjection, ReportsFailureForUnderresolvedStencil)
{
    StructuredQuadMeshAccess mesh(/*nx=*/1, /*ny=*/1, /*h=*/1.0);
    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()), 0.0);
    std::vector<FE::Real> curvature;
    const auto result = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, {}, curvature);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.fitted_vertices, 0u);
    EXPECT_EQ(curvature.size(), phi.size());
    EXPECT_NE(result.diagnostic.find("could not fit"), std::string::npos);
}

TEST(LevelSetCurvatureProjection, SupplementalSamplesAllowUnderresolvedQuadraticStencil)
{
    StructuredQuadMeshAccess mesh(/*nx=*/1, /*ny=*/1, /*h=*/1.0);
    const auto phi_function = [](const std::array<FE::Real, 3>& x) {
        return x[0] + FE::Real{0.25} * x[0] * x[0] +
               FE::Real{0.50} * x[1] * x[1];
    };

    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()), 0.0);
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        phi[static_cast<std::size_t>(v)] = phi_function(mesh.getNodeCoordinates(v));
    }

    std::vector<level_set::LevelSetCurvatureProjectionSample> samples{
        level_set::LevelSetCurvatureProjectionSample{
            .parent_cell = 0,
            .coordinate = {{FE::Real{0.0}, FE::Real{0.0}, FE::Real{0.0}}},
            .value = phi_function({{FE::Real{0.0}, FE::Real{0.0}, FE::Real{0.0}}})},
        level_set::LevelSetCurvatureProjectionSample{
            .parent_cell = 0,
            .coordinate = {{FE::Real{0.20}, FE::Real{-0.15}, FE::Real{0.0}}},
            .value = phi_function({{FE::Real{0.20}, FE::Real{-0.15}, FE::Real{0.0}}})},
    };

    level_set::LevelSetCurvatureProjectionOptions options;
    options.max_neighbor_rings = 1;
    std::vector<FE::Real> curvature;
    const auto result = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(curvature.size(), phi.size());
    EXPECT_EQ(result.supplemental_samples, samples.size());
    EXPECT_DOUBLE_EQ(result.supplemental_sample_weight, FE::Real{1.0});
    EXPECT_EQ(result.vertices_with_supplemental_samples, phi.size());
    EXPECT_GE(result.supplemental_sample_rows, samples.size() * phi.size());
    EXPECT_EQ(result.fitted_vertices, phi.size());
    EXPECT_GT(result.max_abs_curvature, FE::Real{0.0});
    EXPECT_LT(result.max_normalized_fit_residual, FE::Real{1.0e-10});
    EXPECT_LT(result.mean_normalized_fit_residual, FE::Real{1.0e-10});
}

TEST(LevelSetCurvatureProjection, SupplementalSampleWeightControlsFitInfluence)
{
    StructuredQuadMeshAccess mesh(/*nx=*/3, /*ny=*/3, /*h=*/0.25);
    const auto phi_function = [](const std::array<FE::Real, 3>& x) {
        return x[0] + FE::Real{0.25} * x[0] * x[0] +
               FE::Real{0.50} * x[1] * x[1];
    };

    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()), 0.0);
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        phi[static_cast<std::size_t>(v)] = phi_function(mesh.getNodeCoordinates(v));
    }

    std::vector<level_set::LevelSetCurvatureProjectionSample> samples{
        level_set::LevelSetCurvatureProjectionSample{
            .parent_cell = 4,
            .coordinate = {{FE::Real{0.04}, FE::Real{-0.03}, FE::Real{0.0}}},
            .value = phi_function({{FE::Real{0.04},
                                    FE::Real{-0.03},
                                    FE::Real{0.0}}}) +
                     FE::Real{0.08}},
    };

    level_set::LevelSetCurvatureProjectionOptions low_weight_options;
    low_weight_options.max_neighbor_rings = 2;
    low_weight_options.supplemental_sample_weight = FE::Real{1.0e-4};
    std::vector<FE::Real> low_weight_curvature;
    const auto low_weight = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, low_weight_options, low_weight_curvature);
    ASSERT_TRUE(low_weight.success) << low_weight.diagnostic;
    EXPECT_DOUBLE_EQ(low_weight.supplemental_sample_weight,
                     low_weight_options.supplemental_sample_weight);

    auto high_weight_options = low_weight_options;
    high_weight_options.supplemental_sample_weight = FE::Real{1.0e4};
    std::vector<FE::Real> high_weight_curvature;
    const auto high_weight = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, high_weight_options, high_weight_curvature);
    ASSERT_TRUE(high_weight.success) << high_weight.diagnostic;
    EXPECT_DOUBLE_EQ(high_weight.supplemental_sample_weight,
                     high_weight_options.supplemental_sample_weight);

    EXPECT_GT(maxAbsDifference(low_weight_curvature, high_weight_curvature),
              FE::Real{1.0e-2});
}

TEST(LevelSetCurvatureProjection, RejectsNonpositiveSupplementalSampleWeight)
{
    StructuredQuadMeshAccess mesh(/*nx=*/2, /*ny=*/2, /*h=*/1.0);
    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()),
                              FE::Real{0.0});

    level_set::LevelSetCurvatureProjectionOptions options;
    options.supplemental_sample_weight = FE::Real{0.0};
    std::vector<FE::Real> curvature;
    EXPECT_THROW((void)level_set::projectLevelSetMeanCurvatureToVertices(
                     mesh, phi, options, curvature),
                 std::invalid_argument);
}

TEST(LevelSetCurvatureProjection, RejectsNegativeNarrowBandWidth)
{
    StructuredQuadMeshAccess mesh(/*nx=*/2, /*ny=*/2, /*h=*/1.0);
    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()),
                              FE::Real{0.0});

    level_set::LevelSetCurvatureProjectionOptions options;
    options.narrow_band_width = FE::Real{-1.0e-3};
    std::vector<FE::Real> curvature;
    EXPECT_THROW((void)level_set::projectLevelSetMeanCurvatureToVertices(
                     mesh, phi, options, curvature),
                 std::invalid_argument);
}

TEST(LevelSetCurvatureProjection, FailsClosedWhenNarrowBandHasNoVertices)
{
    StructuredQuadMeshAccess mesh(/*nx=*/2, /*ny=*/2, /*h=*/1.0);
    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()),
                              FE::Real{1.0});

    level_set::LevelSetCurvatureProjectionOptions options;
    options.narrow_band_width = FE::Real{1.0e-3};
    std::vector<FE::Real> curvature;
    const auto result = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, options, curvature);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.narrow_band_vertices, 0u);
    EXPECT_EQ(result.skipped_far_vertices, result.vertices);
    EXPECT_NE(result.diagnostic.find("narrow band"), std::string::npos);
}

TEST(LevelSetCurvatureProjection, FailsClosedWhenFitResidualLimitIsExceeded)
{
    StructuredQuadMeshAccess mesh(/*nx=*/2, /*ny=*/2, /*h=*/1.0);
    const auto phi_function = [](const std::array<FE::Real, 3>& x) {
        return x[0] + FE::Real{0.25} * x[0] * x[0] +
               FE::Real{0.50} * x[1] * x[1];
    };

    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()), 0.0);
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        phi[static_cast<std::size_t>(v)] = phi_function(mesh.getNodeCoordinates(v));
    }

    std::vector<level_set::LevelSetCurvatureProjectionSample> samples{
        level_set::LevelSetCurvatureProjectionSample{
            .parent_cell = 0,
            .coordinate = {{FE::Real{0.1}, FE::Real{0.2}, FE::Real{0.0}}},
            .value = phi_function({{FE::Real{0.1}, FE::Real{0.2}, FE::Real{0.0}}}) +
                     FE::Real{0.5}},
    };

    level_set::LevelSetCurvatureProjectionOptions options;
    options.max_neighbor_rings = 2;
    options.max_normalized_fit_residual = 1.0e-6;
    std::vector<FE::Real> curvature;
    const auto result = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature);

    EXPECT_FALSE(result.success);
    EXPECT_GT(result.fit_residual_failure_vertices, 0u);
    EXPECT_NE(result.diagnostic.find("residual"), std::string::npos);
}

TEST(LevelSetCurvatureProjection, FailsClosedWhenZeroFallbackLimitIsExceeded)
{
    StructuredQuadMeshAccess mesh(/*nx=*/16, /*ny=*/16, /*h=*/0.05);
    constexpr FE::Real radius = 0.25;
    constexpr FE::Real corner_plateau = 10.0;
    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()), 0.0);
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        const auto x = mesh.getNodeCoordinates(v);
        phi[static_cast<std::size_t>(v)] =
            std::sqrt(x[0] * x[0] + x[1] * x[1]) - radius;
        if (x[0] <= FE::Real{-0.25} && x[1] <= FE::Real{-0.25}) {
            phi[static_cast<std::size_t>(v)] = corner_plateau;
        }
    }

    const std::vector<level_set::LevelSetCurvatureProjectionSample> samples{
        level_set::LevelSetCurvatureProjectionSample{
            .parent_cell = 0,
            .coordinate = {{FE::Real{-0.375},
                            FE::Real{-0.375},
                            FE::Real{0.0}}},
            .value = corner_plateau},
    };

    level_set::LevelSetCurvatureProjectionOptions options;
    options.max_neighbor_rings = 2;
    options.narrow_band_width = FE::Real{0.055};
    std::vector<FE::Real> curvature;
    const auto allowed = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature);

    ASSERT_TRUE(allowed.success) << allowed.diagnostic;
    ASSERT_GT(allowed.fitted_vertices, 0u);
    ASSERT_GT(allowed.zero_fallback_vertices, 0u);

    options.max_zero_fallback_vertices = 0;
    const auto rejected = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature);

    EXPECT_FALSE(rejected.success);
    EXPECT_EQ(rejected.zero_fallback_vertices, allowed.zero_fallback_vertices);
    EXPECT_NE(rejected.diagnostic.find("zero fallback vertices"),
              std::string::npos);
    EXPECT_NE(rejected.diagnostic.find("configured limit"),
              std::string::npos);
}

TEST(LevelSetCurvatureProjection,
     UsesBoundedFallbackWhenResidualOutliersHaveFittedNeighbors)
{
    StructuredQuadMeshAccess mesh(/*nx=*/8, /*ny=*/8, /*h=*/0.10);
    const auto phi_function = [](const std::array<FE::Real, 3>& x) {
        return x[0] + FE::Real{0.25} * x[0] * x[0] +
               FE::Real{0.50} * x[1] * x[1];
    };

    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()), 0.0);
    FE::GlobalIndex center_vertex = 0;
    FE::Real center_distance2 = std::numeric_limits<FE::Real>::infinity();
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        const auto x = mesh.getNodeCoordinates(v);
        phi[static_cast<std::size_t>(v)] = phi_function(x);
        const auto distance2 = x[0] * x[0] + x[1] * x[1];
        if (distance2 < center_distance2) {
            center_distance2 = distance2;
            center_vertex = v;
        }
    }
    phi[static_cast<std::size_t>(center_vertex)] += FE::Real{0.20};

    level_set::LevelSetCurvatureProjectionOptions options;
    options.max_neighbor_rings = 2;
    options.max_normalized_fit_residual = 2.0e-2;
    std::vector<FE::Real> curvature;
    const auto result = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, options, curvature);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_GT(result.fitted_vertices, 0u);
    EXPECT_GT(result.fit_residual_failure_vertices, 0u);
    EXPECT_GT(result.fallback_vertices, 0u);
    EXPECT_LT(result.zero_fallback_vertices, result.vertices);
    EXPECT_NE(result.diagnostic.find("neighbor fallback"), std::string::npos);
    for (const auto value : curvature) {
        EXPECT_TRUE(std::isfinite(value));
    }

    options.max_neighbor_fallback_vertices = 0;
    const auto rejected = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, options, curvature);
    EXPECT_FALSE(rejected.success);
    EXPECT_EQ(rejected.fallback_vertices, result.fallback_vertices);
    EXPECT_NE(rejected.diagnostic.find("neighbor fallback vertices"),
              std::string::npos);
    EXPECT_NE(rejected.diagnostic.find("configured limit"),
              std::string::npos);
}

TEST(LevelSetCurvatureProjection, OptionalSmoothingReducesCurvatureGraphVariation)
{
    StructuredQuadMeshAccess mesh(/*nx=*/8, /*ny=*/8, /*h=*/0.08);
    const auto phi_function = [](const std::array<FE::Real, 3>& x) {
        return x[0] + FE::Real{0.20} * x[0] * x[0] +
               FE::Real{0.35} * x[1] * x[1];
    };

    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()), 0.0);
    FE::GlobalIndex center_vertex = 0;
    FE::Real center_distance2 = std::numeric_limits<FE::Real>::infinity();
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        const auto x = mesh.getNodeCoordinates(v);
        phi[static_cast<std::size_t>(v)] = phi_function(x);
        const FE::Real distance2 = x[0] * x[0] + x[1] * x[1];
        if (distance2 < center_distance2) {
            center_distance2 = distance2;
            center_vertex = v;
        }
    }
    phi[static_cast<std::size_t>(center_vertex)] += FE::Real{0.03};

    level_set::LevelSetCurvatureProjectionOptions unsmoothed_options;
    unsmoothed_options.max_neighbor_rings = 2;
    std::vector<FE::Real> unsmoothed_curvature;
    const auto unsmoothed = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, unsmoothed_options, unsmoothed_curvature);
    ASSERT_TRUE(unsmoothed.success) << unsmoothed.diagnostic;
    EXPECT_EQ(unsmoothed.smoothing_iterations_applied, 0u);

    auto smoothed_options = unsmoothed_options;
    smoothed_options.smoothing_iterations = 3;
    smoothed_options.smoothing_relaxation = 0.5;
    std::vector<FE::Real> smoothed_curvature;
    const auto smoothed = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, smoothed_options, smoothed_curvature);

    ASSERT_TRUE(smoothed.success) << smoothed.diagnostic;
    EXPECT_EQ(smoothed.smoothing_iterations_applied, 3u);
    EXPECT_GT(smoothed.smoothing_mean_abs_update, FE::Real{0.0});
    EXPECT_GT(smoothed.smoothing_max_abs_update, FE::Real{0.0});
    EXPECT_LT(curvatureGraphTotalVariation(mesh, smoothed_curvature),
              curvatureGraphTotalVariation(mesh, unsmoothed_curvature));
}

TEST(LevelSetCurvatureProjection,
     ParsesCurvatureSmoothingModesAndRejectsUnknownTokens)
{
    EXPECT_EQ(level_set::parseLevelSetCurvatureSmoothingMode("local_graph"),
              level_set::LevelSetCurvatureSmoothingMode::LocalGraph);
    EXPECT_EQ(level_set::parseLevelSetCurvatureSmoothingMode("mass-stiffness"),
              level_set::LevelSetCurvatureSmoothingMode::MassStiffnessOperator);
    EXPECT_EQ(level_set::parseLevelSetCurvatureSmoothingMode("helmholtz"),
              level_set::LevelSetCurvatureSmoothingMode::MassStiffnessOperator);
    EXPECT_STREQ(level_set::levelSetCurvatureSmoothingModeName(
                     level_set::LevelSetCurvatureSmoothingMode::
                         MassStiffnessOperator),
                 "mass_stiffness_operator");
    EXPECT_THROW((void)level_set::parseLevelSetCurvatureSmoothingMode(
                     "unsupported"),
                 std::invalid_argument);
}

TEST(LevelSetCurvatureProjection,
     MassStiffnessOperatorSmoothingReducesCurvatureGraphVariation)
{
    StructuredQuadMeshAccess mesh(/*nx=*/8, /*ny=*/8, /*h=*/0.08);
    const auto phi_function = [](const std::array<FE::Real, 3>& x) {
        return x[0] + FE::Real{0.20} * x[0] * x[0] +
               FE::Real{0.35} * x[1] * x[1];
    };

    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()), 0.0);
    FE::GlobalIndex center_vertex = 0;
    FE::Real center_distance2 = std::numeric_limits<FE::Real>::infinity();
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        const auto x = mesh.getNodeCoordinates(v);
        phi[static_cast<std::size_t>(v)] = phi_function(x);
        const FE::Real distance2 = x[0] * x[0] + x[1] * x[1];
        if (distance2 < center_distance2) {
            center_distance2 = distance2;
            center_vertex = v;
        }
    }
    phi[static_cast<std::size_t>(center_vertex)] += FE::Real{0.03};

    level_set::LevelSetCurvatureProjectionOptions unsmoothed_options;
    unsmoothed_options.max_neighbor_rings = 2;
    std::vector<FE::Real> unsmoothed_curvature;
    const auto unsmoothed = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, unsmoothed_options, unsmoothed_curvature);
    ASSERT_TRUE(unsmoothed.success) << unsmoothed.diagnostic;

    auto smoothed_options = unsmoothed_options;
    smoothed_options.smoothing_mode =
        level_set::LevelSetCurvatureSmoothingMode::MassStiffnessOperator;
    smoothed_options.smoothing_iterations = 2;
    smoothed_options.smoothing_relaxation = 0.5;
    std::vector<FE::Real> smoothed_curvature;
    const auto smoothed = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, smoothed_options, smoothed_curvature);

    ASSERT_TRUE(smoothed.success) << smoothed.diagnostic;
    EXPECT_EQ(smoothed.smoothing_mode,
              level_set::LevelSetCurvatureSmoothingMode::MassStiffnessOperator);
    EXPECT_EQ(smoothed.smoothing_iterations_applied, 2u);
    EXPECT_GT(smoothed.smoothing_operator_edges, 0u);
    EXPECT_GT(smoothed.smoothing_mean_abs_update, FE::Real{0.0});
    EXPECT_GT(smoothed.smoothing_max_abs_update, FE::Real{0.0});
    EXPECT_LT(curvatureGraphTotalVariation(mesh, smoothed_curvature),
              curvatureGraphTotalVariation(mesh, unsmoothed_curvature));
}

TEST(LevelSetCurvatureProjection, WorkspaceReusesMeshAndSampleAdjacency)
{
    StructuredQuadMeshAccess mesh(/*nx=*/4, /*ny=*/4, /*h=*/0.20);
    const auto phi_function = [](const std::array<FE::Real, 3>& x,
                                 FE::Real offset) {
        return x[0] + FE::Real{0.25} * x[0] * x[0] +
               FE::Real{0.50} * x[1] * x[1] + offset;
    };

    std::vector<FE::Real> phi(static_cast<std::size_t>(mesh.numVertices()), 0.0);
    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        phi[static_cast<std::size_t>(v)] =
            phi_function(mesh.getNodeCoordinates(v), FE::Real{0.0});
    }

    std::vector<level_set::LevelSetCurvatureProjectionSample> samples{
        level_set::LevelSetCurvatureProjectionSample{
            .parent_cell = 0,
            .coordinate = {{FE::Real{-0.30}, FE::Real{-0.25}, FE::Real{0.0}}},
            .value = phi_function(
                {{FE::Real{-0.30}, FE::Real{-0.25}, FE::Real{0.0}}},
                FE::Real{0.0})},
        level_set::LevelSetCurvatureProjectionSample{
            .parent_cell = 5,
            .coordinate = {{FE::Real{0.05}, FE::Real{0.10}, FE::Real{0.0}}},
            .value = phi_function(
                {{FE::Real{0.05}, FE::Real{0.10}, FE::Real{0.0}}},
                FE::Real{0.0})},
    };

    level_set::LevelSetCurvatureProjectionOptions options;
    options.max_neighbor_rings = 2;
    level_set::LevelSetCurvatureProjectionWorkspace workspace;

    std::vector<FE::Real> curvature;
    const auto first = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature, workspace);
    ASSERT_TRUE(first.success) << first.diagnostic;
    EXPECT_FALSE(first.reused_vertex_adjacency);
    EXPECT_FALSE(first.reused_sample_adjacency);
    EXPECT_EQ(first.vertex_adjacency_builds, 1u);
    EXPECT_EQ(first.sample_adjacency_builds, 1u);

    for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
        phi[static_cast<std::size_t>(v)] =
            phi_function(mesh.getNodeCoordinates(v), FE::Real{0.01});
    }
    for (auto& sample : samples) {
        sample.value += FE::Real{0.01};
    }

    const auto second = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature, workspace);
    ASSERT_TRUE(second.success) << second.diagnostic;
    EXPECT_TRUE(second.reused_vertex_adjacency);
    EXPECT_TRUE(second.reused_sample_adjacency);
    EXPECT_EQ(second.vertex_adjacency_builds, 1u);
    EXPECT_EQ(second.sample_adjacency_builds, 1u);

    samples.front().coordinate[0] += FE::Real{0.01};
    const auto third = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature, workspace);
    ASSERT_TRUE(third.success) << third.diagnostic;
    EXPECT_TRUE(third.reused_vertex_adjacency);
    EXPECT_TRUE(third.reused_sample_adjacency);
    EXPECT_EQ(third.vertex_adjacency_builds, 1u);
    EXPECT_EQ(third.sample_adjacency_builds, 1u);

    samples.front().parent_cell = static_cast<FE::MeshIndex>(-1);
    samples.front().value =
        phi_function(samples.front().coordinate, FE::Real{0.01});
    const auto fourth = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature, workspace);
    ASSERT_TRUE(fourth.success) << fourth.diagnostic;
    EXPECT_TRUE(fourth.reused_vertex_adjacency);
    EXPECT_FALSE(fourth.reused_sample_adjacency);
    EXPECT_EQ(fourth.vertex_adjacency_builds, 1u);
    EXPECT_EQ(fourth.sample_adjacency_builds, 2u);

    samples.front().coordinate[0] += FE::Real{0.01};
    samples.front().value =
        phi_function(samples.front().coordinate, FE::Real{0.01});
    const auto fifth = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature, workspace);
    ASSERT_TRUE(fifth.success) << fifth.diagnostic;
    EXPECT_TRUE(fifth.reused_vertex_adjacency);
    EXPECT_FALSE(fifth.reused_sample_adjacency);
    EXPECT_EQ(fifth.vertex_adjacency_builds, 1u);
    EXPECT_EQ(fifth.sample_adjacency_builds, 3u);
}

TEST(LevelSetCurvatureProjection,
     WorkspaceRebuildsSampleAdjacencyWhenMeshGeometryRevisionChanges)
{
    StructuredQuadMeshAccess mesh(/*nx=*/4, /*ny=*/4, /*h=*/0.20);
    const auto phi_function = [](const std::array<FE::Real, 3>& x) {
        return x[0] + FE::Real{0.25} * x[0] * x[0] +
               FE::Real{0.50} * x[1] * x[1];
    };
    auto sample_value = [&](const std::array<FE::Real, 3>& x) {
        return phi_function(x);
    };
    auto vertex_phi = [&]() {
        std::vector<FE::Real> phi(
            static_cast<std::size_t>(mesh.numVertices()), FE::Real{0.0});
        for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
            phi[static_cast<std::size_t>(v)] =
                phi_function(mesh.getNodeCoordinates(v));
        }
        return phi;
    };

    std::vector<level_set::LevelSetCurvatureProjectionSample> samples{
        level_set::LevelSetCurvatureProjectionSample{
            .parent_cell = 0,
            .coordinate = {{FE::Real{-0.30}, FE::Real{-0.25}, FE::Real{0.0}}},
            .value = sample_value(
                {{FE::Real{-0.30}, FE::Real{-0.25}, FE::Real{0.0}}})},
        level_set::LevelSetCurvatureProjectionSample{
            .parent_cell = 5,
            .coordinate = {{FE::Real{0.05}, FE::Real{0.10}, FE::Real{0.0}}},
            .value = sample_value(
                {{FE::Real{0.05}, FE::Real{0.10}, FE::Real{0.0}}})},
    };

    level_set::LevelSetCurvatureProjectionOptions options;
    options.max_neighbor_rings = 2;
    level_set::LevelSetCurvatureProjectionWorkspace workspace;

    std::vector<FE::Real> curvature;
    auto phi = vertex_phi();
    const auto first = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature, workspace);
    ASSERT_TRUE(first.success) << first.diagnostic;
    EXPECT_FALSE(first.reused_vertex_adjacency);
    EXPECT_FALSE(first.reused_sample_adjacency);
    EXPECT_EQ(first.vertex_adjacency_builds, 1u);
    EXPECT_EQ(first.sample_adjacency_builds, 1u);

    mesh.translate(FE::Real{0.02}, FE::Real{-0.01});
    phi = vertex_phi();
    const auto second = level_set::projectLevelSetMeanCurvatureToVertices(
        mesh, phi, samples, options, curvature, workspace);
    ASSERT_TRUE(second.success) << second.diagnostic;
    EXPECT_TRUE(second.reused_vertex_adjacency);
    EXPECT_FALSE(second.reused_sample_adjacency);
    EXPECT_EQ(second.vertex_adjacency_builds, 1u);
    EXPECT_EQ(second.sample_adjacency_builds, 2u);
}
