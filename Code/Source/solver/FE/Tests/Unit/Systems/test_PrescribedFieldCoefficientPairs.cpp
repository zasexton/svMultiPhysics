#include <gtest/gtest.h>

#include "Systems/FESystem.h"
#include "Core/FEException.h"
#include "Dofs/EntityDofMap.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Mesh/Mesh.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Fields/MeshFields.h"
#include "Mesh/Topology/CellShape.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using svmp::FE::ElementType;
using svmp::FE::FieldId;
using svmp::FE::InvalidArgumentException;
using svmp::FE::Real;
using svmp::FE::spaces::H1Space;
using svmp::FE::spaces::ProductSpace;
using svmp::FE::systems::FESystem;
using svmp::FE::systems::FieldSourceKind;
using svmp::FE::systems::FieldSpec;

// The fallback exercises the existing single-part API in the baseline build.
// It supplies runtime failures without making a missing declaration a test.
template<class System>
void publishPairs(System& system, FieldId field, std::span<const Real> high,
                  std::span<const Real> low)
{
    if constexpr (requires { system.setPrescribedFieldCoefficientPairs(field, high, low); }) {
        system.setPrescribedFieldCoefficientPairs(field, high, low);
    } else {
        system.setPrescribedFieldCoefficients(field, high);
    }
}

template<class System>
std::span<const Real> lowParts(const System& system, FieldId field)
{
    if constexpr (requires { system.prescribedFieldCoefficientLowParts(field); }) {
        return system.prescribedFieldCoefficientLowParts(field);
    } else {
        return {};
    }
}

std::shared_ptr<svmp::Mesh> makeMesh()
{
    auto base = std::make_shared<svmp::MeshBase>();
    const std::vector<svmp::real_t> points{0, 0, 1, 0, 1, 1, 0, 1};
    const std::vector<svmp::offset_t> offsets{0, 4};
    const std::vector<svmp::index_t> vertices{0, 1, 2, 3};
    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(2, points, offsets, vertices, {shape});
    base->finalize();
    return svmp::create_mesh(std::move(base));
}

void expectBits(std::span<const Real> actual, std::span<const Real> expected)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(std::bit_cast<std::uint64_t>(actual[i]),
                  std::bit_cast<std::uint64_t>(expected[i])) << i;
    }
}

class PrescribedCoefficientPairs : public ::testing::Test {
protected:
    std::shared_ptr<svmp::Mesh> mesh = makeMesh();
    std::shared_ptr<H1Space> space = std::make_shared<H1Space>(ElementType::Quad4, 1);
    FESystem system{mesh};
    FieldId field{};
    const std::array<Real, 4> high{1, -1, Real{0}, -Real{0}};
    const std::array<Real, 4> low{0x1p-60, -0x1p-60, Real{0}, -Real{0}};

    void SetUp() override
    {
        field = system.addField(FieldSpec{.name = "paired_source", .space = space,
            .components = 1, .source_kind = FieldSourceKind::PrescribedData});
        system.setup();
    }

    void expectStored() const
    {
        expectBits(system.prescribedFieldCoefficients(field), high);
        expectBits(lowParts(system, field), low);
    }

    void attachDiagnosticValues()
    {
        const auto handle = svmp::MeshFields::attach_field(mesh->local_mesh(),
            svmp::EntityKind::Vertex, "paired_source", svmp::FieldScalarType::Float64, 1);
        auto* values = svmp::MeshFields::field_data_as<svmp::real_t>(mesh->local_mesh(), handle);
        ASSERT_NE(values, nullptr);
        for (std::size_t i = 0; i < mesh->n_vertices(); ++i) values[i] = Real{42};
    }
};

TEST_F(PrescribedCoefficientPairs, RetainsExactPartsWithOneRevision)
{
    const auto before = system.prescribedFieldRevision(field);
    publishPairs(system, field, high, low);
    expectStored();
    EXPECT_EQ(system.prescribedFieldRevision(field), before + 1);
}

TEST_F(PrescribedCoefficientPairs, AcceptsAliasedReplacementInputs)
{
    publishPairs(system, field, high, low);
    const auto before = system.prescribedFieldRevision(field);
    publishPairs(system, field, system.prescribedFieldCoefficients(field), lowParts(system, field));
    expectStored();
    EXPECT_EQ(system.prescribedFieldRevision(field), before + 1);
}

TEST_F(PrescribedCoefficientPairs, RejectsWrongExtentsWithoutMutation)
{
    publishPairs(system, field, high, low);
    const auto before = system.prescribedFieldRevision(field);
    EXPECT_THROW(publishPairs(system, field, std::span<const Real>(high).first(3), low),
                 InvalidArgumentException);
    EXPECT_THROW(publishPairs(system, field, high, std::span<const Real>(low).first(3)),
                 InvalidArgumentException);
    expectStored();
    EXPECT_EQ(system.prescribedFieldRevision(field), before);
}

TEST_F(PrescribedCoefficientPairs, RejectsNonfinitePartsWithoutMutation)
{
    publishPairs(system, field, high, low);
    const auto before = system.prescribedFieldRevision(field);
    for (const auto invalid : {std::numeric_limits<Real>::infinity(),
                              std::numeric_limits<Real>::quiet_NaN()}) {
        auto bad_high = high;
        auto bad_low = low;
        bad_high[0] = invalid;
        bad_low[0] = invalid;
        EXPECT_THROW(publishPairs(system, field, bad_high, low), InvalidArgumentException);
        EXPECT_THROW(publishPairs(system, field, high, bad_low), InvalidArgumentException);
    }
    expectStored();
    EXPECT_EQ(system.prescribedFieldRevision(field), before);
}

TEST_F(PrescribedCoefficientPairs, RejectsLostAndNoncanonicalTailsWithoutMutation)
{
    publishPairs(system, field, high, low);
    const auto before = system.prescribedFieldRevision(field);
    auto bad_low = low;
    bad_low[0] = std::numeric_limits<Real>::denorm_min();
    EXPECT_THROW(publishPairs(system, field, high, bad_low), InvalidArgumentException);
    bad_low[0] = Real{1};
    EXPECT_THROW(publishPairs(system, field, high, bad_low), InvalidArgumentException);
    expectStored();
    EXPECT_EQ(system.prescribedFieldRevision(field), before);
}

TEST_F(PrescribedCoefficientPairs, MeshSyncPreservesPairedPublication)
{
    attachDiagnosticValues();
    publishPairs(system, field, high, low);
    const auto before = system.prescribedFieldRevision(field);
    EXPECT_EQ(system.syncPrescribedVertexFieldsFromMeshFields(), 0u);
    expectStored();
    EXPECT_EQ(system.prescribedFieldRevision(field), before);
}

TEST_F(PrescribedCoefficientPairs, MeshSyncPreservesZeroLowPairOwnership)
{
    attachDiagnosticValues();
    const std::array<Real, 4> zeros{Real{0}, -Real{0}, Real{0}, -Real{0}};
    publishPairs(system, field, high, zeros);
    const auto before = system.prescribedFieldRevision(field);
    EXPECT_EQ(system.syncPrescribedVertexFieldsFromMeshFields(), 0u);
    expectBits(system.prescribedFieldCoefficients(field), high);
    expectBits(lowParts(system, field), zeros);
    EXPECT_EQ(system.prescribedFieldRevision(field), before);
}

TEST_F(PrescribedCoefficientPairs, LegacyReplacementReleasesPairOwnership)
{
    attachDiagnosticValues();
    publishPairs(system, field, high, low);
    const auto before = system.prescribedFieldRevision(field);
    system.setPrescribedFieldCoefficients(field, high);
    EXPECT_TRUE(lowParts(system, field).empty());
    EXPECT_EQ(system.prescribedFieldRevision(field), before + 1);
    EXPECT_EQ(system.syncPrescribedVertexFieldsFromMeshFields(), 4u);
    const std::array<Real, 4> diagnostic{42, 42, 42, 42};
    expectBits(system.prescribedFieldCoefficients(field), diagnostic);
}

TEST_F(PrescribedCoefficientPairs, ClearReleasesBothPartsAndOwnership)
{
    attachDiagnosticValues();
    publishPairs(system, field, high, low);
    const auto before = system.prescribedFieldRevision(field);
    system.clearPrescribedFieldCoefficients(field);
    EXPECT_TRUE(system.prescribedFieldCoefficients(field).empty());
    EXPECT_TRUE(lowParts(system, field).empty());
    EXPECT_EQ(system.prescribedFieldRevision(field), before + 1);
    EXPECT_EQ(system.syncPrescribedVertexFieldsFromMeshFields(), 4u);
}

TEST_F(PrescribedCoefficientPairs, PointEvaluationRetainsTailThroughCancellation)
{
    const std::array<Real, 4> vertex_high{0x1p60, -0x1p60, 0x1p60, -0x1p60};
    std::array<Real, 4> mapped_high{};
    std::array<Real, 4> mapped_low{};
    const auto* map = system.fieldDofHandler(field).getEntityDofMap();
    ASSERT_NE(map, nullptr);
    for (std::size_t vertex = 0; vertex < 4; ++vertex) {
        const auto dofs = map->getVertexDofs(static_cast<svmp::FE::GlobalIndex>(vertex));
        ASSERT_EQ(dofs.size(), 1u);
        const auto dof = static_cast<std::size_t>(dofs[0]);
        ASSERT_LT(dof, mapped_high.size());
        mapped_high[dof] = vertex_high[vertex];
        mapped_low[dof] = Real{1};
    }
    publishPairs(system, field, mapped_high, mapped_low);
    const auto value = system.evaluateFieldAtPoint(field, {}, {Real{0.5}, Real{0.5}, Real{0}}, 0);
    ASSERT_TRUE(value.has_value());
    EXPECT_DOUBLE_EQ((*value)[0], Real{1});
}

TEST_F(PrescribedCoefficientPairs, ZeroLowPointEvaluationPreservesLegacyBits)
{
    const std::array<Real, 4> values{0x1p60, Real{1}, -0x1p60, Real{1}};
    const std::array<Real, 4> zeros{};
    system.setPrescribedFieldCoefficients(field, values);
    const auto legacy = system.evaluateFieldAtPoint(field, {}, {Real{0.5}, Real{0.5}, Real{0}}, 0);
    ASSERT_TRUE(legacy.has_value());
    publishPairs(system, field, values, zeros);
    const auto paired = system.evaluateFieldAtPoint(field, {}, {Real{0.5}, Real{0.5}, Real{0}}, 0);
    ASSERT_TRUE(paired.has_value());
    expectBits(*paired, *legacy);
}

TEST(PrescribedCoefficientPairTypes, RejectsVectorStorageWithoutPublication)
{
    auto mesh = makeMesh();
    auto scalar = std::make_shared<H1Space>(ElementType::Quad4, 1);
    auto vector = std::make_shared<ProductSpace>(scalar, 2);
    FESystem system(mesh);
    const auto field = system.addField(FieldSpec{.name = "vector_source", .space = vector,
        .components = 2, .source_kind = FieldSourceKind::PrescribedData});
    system.setup();
    const std::array<Real, 8> high{};
    const std::array<Real, 8> low{};
    const auto before = system.prescribedFieldRevision(field);
    EXPECT_THROW(publishPairs(system, field, high, low), InvalidArgumentException);
    EXPECT_TRUE(system.prescribedFieldCoefficients(field).empty());
    EXPECT_EQ(system.prescribedFieldRevision(field), before);
}

} // namespace
