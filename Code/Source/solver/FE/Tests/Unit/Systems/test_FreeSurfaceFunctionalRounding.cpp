/**
 * @file test_FreeSurfaceFunctionalRounding.cpp
 * @brief Tests of the derived rounding bound for sums of the same terms.
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"
#include "Systems/FreeSurfaceFunctionalRounding.h"

#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"

#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using svmp::FE::Real;
using svmp::FE::systems::freeSurfaceNonnegativeSummationAbsoluteBound;
using svmp::FE::systems::freeSurfaceNonnegativeSummationsAgree;
using svmp::FE::systems::freeSurfaceSummationErrorFactor;
using svmp::FE::systems::freeSurfaceSummationsAgree;
using svmp::FE::systems::freeSurfaceSummationTolerance;
using svmp::FE::systems::kFreeSurfaceFunctionalUnitRoundoff;

namespace {

Real recursiveSum(const std::vector<Real>& terms)
{
    Real sum{0.0};
    for (const auto term : terms) {
        sum += term;
    }
    return sum;
}

// Per-rule sums added to a total, as the functional state sums rule measures
// that are themselves recursive sums of the rule weights.
Real ruleGroupedSum(const std::vector<Real>& terms, std::uint64_t seed)
{
    std::mt19937_64 generator(seed);
    std::uniform_int_distribution<std::size_t> rule_size(1u, 12u);
    Real total{0.0};
    std::size_t begin = 0u;
    while (begin < terms.size()) {
        const auto end = std::min(terms.size(), begin + rule_size(generator));
        Real rule{0.0};
        for (std::size_t i = begin; i < end; ++i) {
            rule += terms[i];
        }
        total += rule;
        begin = end;
    }
    return total;
}

// Contiguous rank-local sums followed by a reduction tree over the ranks.
Real rankReducedSum(const std::vector<Real>& terms, std::size_t ranks)
{
    std::vector<Real> partial(ranks, Real{0.0});
    const std::size_t chunk = (terms.size() + ranks - 1u) / ranks;
    for (std::size_t i = 0u; i < terms.size(); ++i) {
        partial[i / chunk] += terms[i];
    }
    while (partial.size() > 1u) {
        std::vector<Real> next;
        for (std::size_t i = 0u; i < partial.size(); i += 2u) {
            next.push_back(i + 1u < partial.size() ? partial[i] + partial[i + 1u]
                                                   : partial[i]);
        }
        partial.swap(next);
    }
    return partial.front();
}

std::vector<Real> permuted(std::vector<Real> terms, std::uint64_t seed)
{
    std::mt19937_64 generator(seed);
    std::shuffle(terms.begin(), terms.end(), generator);
    return terms;
}

// Quadrature-weight-like terms spanning nine decades.
std::vector<Real> wideRangeTerms(std::size_t count, std::uint64_t seed)
{
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<Real> exponent(Real{-12.0}, Real{-3.0});
    std::vector<Real> terms(count);
    for (auto& term : terms) {
        term = std::pow(Real{10.0}, exponent(generator));
    }
    return terms;
}

// Terms of similar size, as on a uniform mesh.
std::vector<Real> nearlyEqualTerms(std::size_t count, std::uint64_t seed)
{
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<Real> value(Real{0.09}, Real{0.11});
    std::vector<Real> terms(count);
    for (auto& term : terms) {
        term = value(generator);
    }
    return terms;
}

enum class TermFamily { WideRange, NearlyEqual, Constant };

// Constant terms make recursive summation drift the most: the partial sum
// rounds the same way for long runs of additions.
std::vector<Real> familyTerms(TermFamily family,
                              std::size_t count,
                              std::uint64_t seed)
{
    switch (family) {
        case TermFamily::WideRange:
            return wideRangeTerms(count, seed);
        case TermFamily::NearlyEqual:
            return nearlyEqualTerms(count, seed);
        case TermFamily::Constant:
            break;
    }
    return std::vector<Real>(count, Real{0.1});
}

std::shared_ptr<svmp::Mesh> buildRoundingSingleQuadMesh()
{
    auto base = std::make_shared<svmp::MeshBase>();
    const std::vector<svmp::real_t> coordinates = {
        0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0};
    const std::vector<svmp::offset_t> offsets = {0, 4};
    const std::vector<svmp::index_t> vertices = {0, 1, 2, 3};
    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(
        /*spatial_dim=*/2, coordinates, offsets, vertices, {shape});
    base->finalize();
    return svmp::create_mesh(std::move(base));
}

bool fixedFiveHundredTwelveUlpNear(Real a, Real b)
{
    const Real scale = std::max({Real{1.0}, std::abs(a), std::abs(b)});
    return std::abs(a - b) <=
           Real{512.0} * std::numeric_limits<Real>::epsilon() * scale;
}

} // namespace

TEST(FreeSurfaceFunctionalRounding, ErrorFactorIsHighamGammaOfTheAdditionCount)
{
    const Real u = kFreeSurfaceFunctionalUnitRoundoff;
    EXPECT_EQ(u, std::ldexp(Real{1.0}, -53));
    EXPECT_EQ(freeSurfaceSummationErrorFactor(0u), Real{0.0});
    EXPECT_EQ(freeSurfaceSummationErrorFactor(1u), Real{0.0});
    EXPECT_EQ(freeSurfaceSummationErrorFactor(2u), u / (Real{1.0} - u));
    const Real k = Real{999999.0};
    EXPECT_EQ(freeSurfaceSummationErrorFactor(1000000u),
              k * u / (Real{1.0} - k * u));
    EXPECT_LT(freeSurfaceSummationErrorFactor(1000u),
              freeSurfaceSummationErrorFactor(1001u));
    EXPECT_TRUE(std::isinf(freeSurfaceSummationErrorFactor(
        std::numeric_limits<std::uint64_t>::max())));
}

TEST(FreeSurfaceFunctionalRounding, AtMostOneTermRequiresExactAgreement)
{
    EXPECT_EQ(freeSurfaceSummationTolerance(1u, 1u, Real{3.0}), Real{0.0});
    EXPECT_TRUE(freeSurfaceNonnegativeSummationsAgree(
        Real{0.25}, 1u, Real{0.25}, 1u));
    EXPECT_FALSE(freeSurfaceNonnegativeSummationsAgree(
        Real{0.25}, 1u, std::nextafter(Real{0.25}, Real{1.0}), 1u));
    EXPECT_TRUE(
        freeSurfaceNonnegativeSummationsAgree(Real{0.0}, 0u, Real{0.0}, 0u));
    EXPECT_FALSE(freeSurfaceNonnegativeSummationsAgree(
        std::numeric_limits<Real>::quiet_NaN(), 10u, Real{1.0}, 10u));
    EXPECT_FALSE(freeSurfaceNonnegativeSummationsAgree(
        std::numeric_limits<Real>::infinity(),
        10u,
        std::numeric_limits<Real>::infinity(),
        10u));
}

TEST(FreeSurfaceFunctionalRounding, ToleranceIsTheSumOfBothSidesBounds)
{
    const std::uint64_t a_terms = 300000u;
    const std::uint64_t b_terms = 4000u;
    const Real absolute_sum = Real{2.5};
    EXPECT_EQ(freeSurfaceSummationTolerance(a_terms, b_terms, absolute_sum),
              (freeSurfaceSummationErrorFactor(a_terms) +
               freeSurfaceSummationErrorFactor(b_terms)) *
                  absolute_sum);
    const Real a = Real{2.5};
    const Real bound =
        freeSurfaceNonnegativeSummationAbsoluteBound(a, a_terms, a, b_terms);
    EXPECT_EQ(bound,
              a / (Real{1.0} - freeSurfaceSummationErrorFactor(a_terms)));
}

TEST(FreeSurfaceFunctionalRounding,
     AcceptsEqualSumsOfUpToAMillionTermsInAnyOrderAndGrouping)
{
    std::size_t old_bound_rejections = 0u;
    for (const std::size_t count : {100000u, 300000u, 1000000u}) {
        for (const auto family : {TermFamily::WideRange,
                                  TermFamily::NearlyEqual,
                                  TermFamily::Constant}) {
            const auto terms = familyTerms(family, count, 11u + count);
            const auto n = static_cast<std::uint64_t>(count);
            const Real forward = recursiveSum(terms);
            const std::vector<Real> sums{
                recursiveSum(permuted(terms, 23u)),
                recursiveSum(permuted(terms, 29u)),
                ruleGroupedSum(terms, 31u),
                ruleGroupedSum(permuted(terms, 37u), 41u),
                rankReducedSum(terms, 7u),
                rankReducedSum(permuted(terms, 43u), 64u),
            };
            for (const auto other : sums) {
                EXPECT_TRUE(freeSurfaceNonnegativeSummationsAgree(
                    forward, n, other, n))
                    << "count=" << count
                    << " family=" << static_cast<int>(family)
                    << " forward=" << forward << " other=" << other;
                if (!fixedFiveHundredTwelveUlpNear(forward, other)) {
                    ++old_bound_rejections;
                }
            }
        }
    }
    // The fixed 512-ulp bound rejects some of these genuinely equal sums;
    // that is the false failure the derived bound removes.
    EXPECT_GT(old_bound_rejections, 0u);
}

TEST(FreeSurfaceFunctionalRounding,
     RejectsATenToTheMinusTenRelativeInconsistency)
{
    for (const std::size_t count : {100000u, 300000u}) {
        for (const auto family : {TermFamily::WideRange,
                                  TermFamily::NearlyEqual,
                                  TermFamily::Constant}) {
            const auto terms = familyTerms(family, count, 53u + count);
            const auto n = static_cast<std::uint64_t>(count);
            const Real forward = recursiveSum(terms);
            const Real grouped = ruleGroupedSum(permuted(terms, 61u), 67u);
            ASSERT_TRUE(
                freeSurfaceNonnegativeSummationsAgree(forward, n, grouped, n));
            const Real relative = Real{1.0e-10};
            EXPECT_FALSE(freeSurfaceNonnegativeSummationsAgree(
                forward, n, grouped * (Real{1.0} + relative), n))
                << "count=" << count
                << " family=" << static_cast<int>(family);
            EXPECT_FALSE(freeSurfaceNonnegativeSummationsAgree(
                forward * (Real{1.0} - relative), n, grouped, n))
                << "count=" << count
                << " family=" << static_cast<int>(family);

            // One term missing from one side whose relative size is 1e-10.
            auto inconsistent = terms;
            inconsistent.push_back(relative * forward);
            EXPECT_FALSE(freeSurfaceNonnegativeSummationsAgree(
                forward, n, recursiveSum(permuted(inconsistent, 71u)), n + 1u))
                << "count=" << count
                << " family=" << static_cast<int>(family);
        }
    }
}

TEST(FreeSurfaceFunctionalRounding, SignedTermsUseTheirAbsoluteSum)
{
    const std::size_t count = 300000u;
    auto terms = wideRangeTerms(count, 73u);
    std::mt19937_64 generator(79u);
    std::bernoulli_distribution negative(0.5);
    for (auto& term : terms) {
        if (negative(generator)) {
            term = -term;
        }
    }
    std::vector<Real> magnitudes(terms.size());
    std::transform(terms.begin(), terms.end(), magnitudes.begin(),
                   [](Real term) { return std::abs(term); });
    const auto n = static_cast<std::uint64_t>(count);
    const Real computed_absolute = recursiveSum(magnitudes);
    const Real absolute_sum = freeSurfaceNonnegativeSummationAbsoluteBound(
        computed_absolute, n, computed_absolute, n);
    const Real forward = recursiveSum(terms);
    const Real grouped = ruleGroupedSum(permuted(terms, 83u), 89u);
    EXPECT_TRUE(freeSurfaceSummationsAgree(forward, n, grouped, n, absolute_sum));
    EXPECT_FALSE(freeSurfaceSummationsAgree(
        forward, n, grouped + Real{1.0e-10} * absolute_sum, n, absolute_sum));
}

TEST(FreeSurfaceFunctionalRounding,
     AcceptedRecordAcceptsRegroupedLiquidVolumeAndRejectsInconsistency)
{
    using svmp::FE::ElementType;
    using svmp::FE::geometry::CutIntegrationSide;
    namespace interfaces = svmp::FE::interfaces;
    namespace systems = svmp::FE::systems;

    auto mesh = buildRoundingSingleQuadMesh();
    auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
        ElementType::Quad4, /*order=*/1);
    auto velocity_space = std::make_shared<svmp::FE::spaces::ProductSpace>(
        scalar_space, /*components=*/2);
    systems::FESystem system(mesh);
    const auto phi = system.addField(systems::FieldSpec{
        .name = "phi_rounding_bound", .space = scalar_space, .components = 1});
    const auto velocity = system.addField(systems::FieldSpec{
        .name = "u_rounding_bound", .space = velocity_space, .components = 2});
    ASSERT_NO_THROW(system.declareFreeSurfaceDiscreteFunctional(
        systems::FreeSurfaceDiscreteFunctionalDeclaration{
            .interface_marker = 91,
            .level_set_field = phi,
            .velocity_field = velocity,
            .geometry_domain_id = "rounding_bound",
            .parameters =
                interfaces::FreeSurfaceDiscreteFunctionalParameters{
                    .liquid_side = CutIntegrationSide::Negative,
                    .surface_tension = 1.0,
                },
            .active_volume_energy_parameters =
                interfaces::FreeSurfaceActiveVolumeEnergyParameters{
                    .liquid_side = CutIntegrationSide::Negative,
                    .density = 1.0,
                },
            .owner_component = "FreeSurfaceFunctionalRounding.Record",
        }));
    ASSERT_NO_THROW(system.setup({}));

    // 300000 equal weights, as on a uniform 3D mesh: the state sums them per
    // four-point rule, the energy point by point.  The two sums differ by
    // about 6.5e-12 relative, far above 512 ulp and below the derived bound.
    constexpr std::size_t point_count = 300000u;
    constexpr std::size_t points_per_rule = 4u;
    const std::vector<Real> weights(point_count, Real{0.1});
    const Real pointwise_volume = recursiveSum(weights);
    Real rulewise_volume{0.0};
    for (std::size_t begin = 0u; begin < point_count;
         begin += points_per_rule) {
        Real rule{0.0};
        for (std::size_t i = begin; i < begin + points_per_rule; ++i) {
            rule += weights[i];
        }
        rulewise_volume += rule;
    }
    ASSERT_FALSE(fixedFiveHundredTwelveUlpNear(pointwise_volume, rulewise_volume));

    const interfaces::FreeSurfaceGeometryRevision revision{
        .source_id = "field:" + std::to_string(phi),
        .domain_id = "rounding_bound",
        .interface_marker = 91,
        .isovalue = 0.0,
        .source_layout_revision = 3u,
        .source_value_revision = 9u,
        .mesh_geometry_revision = 4u,
        .mesh_topology_revision = 5u,
        .ownership_revision = 6u,
        .numbering_revision = 7u,
        .quadrature_policy_key = 8u,
        .snapshot_revision_key = 101u,
    };
    const Real liquid_gas_area = 0.4;
    const auto make_states = [&](Real energy_volume) {
        interfaces::FreeSurfaceDiscreteFunctionalState state{
            .snapshot_revision_key = 101u,
            .liquid_side = CutIntegrationSide::Negative,
            .surface_tension = 1.0,
            .volume_multiplier = 0.0,
            .owned_liquid_volume = rulewise_volume,
            .owned_liquid_gas_area = liquid_gas_area,
            .liquid_gas_surface_energy = Real{1.0} * liquid_gas_area,
            .volume_constraint_potential = Real{0.0} * rulewise_volume,
        };
        state.total_potential = state.liquid_gas_surface_energy +
                                state.young_wall_energy +
                                state.volume_constraint_potential;
        interfaces::FreeSurfaceActiveVolumeEnergyState energy{
            .snapshot_revision_key = 101u,
            .liquid_side = CutIntegrationSide::Negative,
            .density = 1.0,
            .owned_quadrature_point_count = point_count,
            .owned_liquid_volume = energy_volume,
        };
        return std::vector<systems::AcceptedFreeSurfaceDiscreteFunctionalState>{
            systems::AcceptedFreeSurfaceDiscreteFunctionalState{
                .interface_marker = 91,
                .geometry_revision = revision,
                .cut_topology_revision = 102u,
                .state = state,
                .active_volume_energy = energy,
            }};
    };

    EXPECT_NO_THROW(system.recordAcceptedFreeSurfaceDiscreteFunctionals(
        1u, 0.1, 0.1, 13u, 13u, make_states(pointwise_volume)));
    ASSERT_EQ(system.freeSurfaceDiscreteFunctionalHistory().size(), 1u);
    EXPECT_EQ(system.freeSurfaceDiscreteFunctionalHistory()
                  .front()
                  .active_volume_energy->owned_liquid_volume,
              pointwise_volume);

    for (const Real relative : {Real{1.0e-10}, Real{-1.0e-10}}) {
        try {
            system.recordAcceptedFreeSurfaceDiscreteFunctionals(
                2u,
                0.2,
                0.1,
                14u,
                14u,
                make_states(pointwise_volume * (Real{1.0} + relative)));
            ADD_FAILURE() << "a 1e-10 relative volume inconsistency was accepted";
        } catch (const svmp::FE::InvalidArgumentException& error) {
            EXPECT_NE(std::string(error.what()).find(
                          "active-volume energy liquid measure is inconsistent"),
                      std::string::npos)
                << error.what();
        }
    }
    EXPECT_EQ(system.freeSurfaceDiscreteFunctionalHistory().size(), 1u);
}
