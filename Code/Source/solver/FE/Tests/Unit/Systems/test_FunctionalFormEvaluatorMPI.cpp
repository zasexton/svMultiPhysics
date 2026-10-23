/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include <mpi.h>

#include "Assembly/CutIntegrationContext.h"
#include "Assembly/FunctionalAssembler.h"
#include "Forms/BoundaryFunctional.h"
#include "Forms/FormExpr.h"
#include "Forms/IntegrationDomain.h"
#include "Spaces/H1Space.h"
#include "Systems/FunctionalFormEvaluator.h"
#include "Systems/SystemCollectives.h"
#include "Tests/Unit/Systems/FunctionalFormEvaluatorFixtures.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace forms = svmp::FE::forms;
namespace systems = svmp::FE::systems;
namespace fixture = svmp::FE::systems::functional_form_test;
using svmp::FE::GlobalIndex;
using svmp::FE::Real;
using forms::FormExpr;

constexpr GlobalIndex kCellCount = 16;
constexpr Real kExtent = 4.0;

/// A rotated elliptic cap on the wall (phi < 0 inside): a curved interface,
/// two contact points and a wetted wall between them.
[[nodiscard]] Real sessileCap(Real x, Real y)
{
    const Real angle = 0.35;
    const Real c = std::cos(angle);
    const Real s = std::sin(angle);
    const Real dx = x - Real{2.1};
    const Real dy = y - Real{0.2};
    const Real a = c * dx + s * dy;
    const Real b = -s * dx + c * dy;
    return a * a / (Real{0.85} * Real{0.85}) + b * b / (Real{0.55} * Real{0.55}) - Real{1.0};
}

[[nodiscard]] Real field(Real x, Real y)
{
    return Real{0.3} + Real{0.7} * x - Real{0.2} * y + Real{0.15} * x * y;
}

[[nodiscard]] std::vector<std::uint64_t> allGatherUnsignedValues(std::span<const std::uint64_t> local)
{
    const int local_count = static_cast<int>(local.size());
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<int> counts(static_cast<std::size_t>(size), 0);
    MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> displacements(static_cast<std::size_t>(size), 0);
    int global_count = 0;
    for (int peer = 0; peer < size; ++peer) {
        displacements[static_cast<std::size_t>(peer)] = global_count;
        global_count += counts[static_cast<std::size_t>(peer)];
    }
    std::vector<std::uint64_t> gathered(static_cast<std::size_t>(global_count));
    MPI_Allgatherv(local.empty() ? nullptr : local.data(),
                   local_count,
                   MPI_UINT64_T,
                   gathered.empty() ? nullptr : gathered.data(),
                   counts.data(),
                   displacements.data(),
                   MPI_UINT64_T,
                   MPI_COMM_WORLD);
    return gathered;
}

[[nodiscard]] svmp::FE::interfaces::FreeSurfaceGeometryOwnershipCollective worldCollective(int rank, int size)
{
    svmp::FE::interfaces::FreeSurfaceGeometryOwnershipCollective collective;
    collective.rank = rank;
    collective.size = size;
    collective.all_gather_owned_rule_identity_values = allGatherUnsignedValues;
    collective.all_gather_revision_values = allGatherUnsignedValues;
    return collective;
}

/// The strip, its field and its snapshot, as seen by one rank of @p size.
struct CapView {
    fixture::QuadStripMeshAccess mesh;
    svmp::FE::dofs::DofMap dof_map;
    svmp::FE::spaces::H1Space space{svmp::FE::ElementType::Quad4, 1};
    std::vector<Real> u_values;
    std::shared_ptr<svmp::FE::assembly::CutIntegrationContext> cut_context;
    svmp::FE::assembly::FunctionalAssembler assembler{};

    CapView(int rank, int size, svmp::FE::interfaces::FreeSurfaceGeometryOwnershipCollective collective)
        : mesh(rank, size, kCellCount, kExtent)
        , dof_map(fixture::quadStripDofMap(mesh))
        , u_values(fixture::nodalValues(mesh, field))
        , cut_context(fixture::quadStripCutContext(mesh, sessileCap, std::move(collective)))
    {
        fixture::configureQuadStripAssembler(assembler, mesh, dof_map, space, u_values);
    }

    [[nodiscard]] FormExpr u() const { return FormExpr::discreteField(fixture::kField, space, "u"); }
};

enum class Measure { Cell, Boundary, CutVolume, Interface };

/// One signed integral of a form, for the long-double reference.
struct Term {
    Real sign;
    Measure measure;
    int marker;
    svmp::FE::geometry::CutIntegrationSide side;
    FormExpr integrand;
};

[[nodiscard]] FormExpr integral(const Term& term)
{
    switch (term.measure) {
    case Measure::Cell:
        return term.integrand.dx();
    case Measure::Boundary:
        return term.integrand.ds(term.marker);
    case Measure::CutVolume:
        return term.integrand.dCutVolume(term.marker,
                                         term.side == svmp::FE::geometry::CutIntegrationSide::Negative
                                             ? forms::CutVolumeSide::Negative
                                             : forms::CutVolumeSide::Positive);
    case Measure::Interface:
        break;
    }
    return term.integrand.dI(term.marker);
}

[[nodiscard]] FormExpr form(const std::vector<Term>& terms)
{
    FormExpr sum{};
    for (const auto& term : terms) {
        const auto measured = integral(term);
        if (!sum.isValid()) {
            sum = term.sign < Real{0.0} ? -measured : measured;
        } else {
            sum = term.sign < Real{0.0} ? sum - measured : sum + measured;
        }
    }
    return sum;
}

/// Records every weighted quadrature value w_q f_q of an integrand.
class RecordingKernel final : public svmp::FE::assembly::FunctionalKernel {
public:
    RecordingKernel(std::shared_ptr<svmp::FE::assembly::FunctionalKernel> integrand, std::vector<Real>& terms)
        : integrand_(std::move(integrand))
        , terms_(terms)
    {
    }

    [[nodiscard]] svmp::FE::assembly::RequiredData getRequiredData() const noexcept override
    {
        return integrand_->getRequiredData();
    }
    [[nodiscard]] std::vector<svmp::FE::assembly::FieldRequirement> fieldRequirements() const override
    {
        return integrand_->fieldRequirements();
    }
    [[nodiscard]] bool hasCell() const noexcept override { return integrand_->hasCell(); }
    [[nodiscard]] bool hasBoundaryFace() const noexcept override { return integrand_->hasBoundaryFace(); }

    [[nodiscard]] Real evaluateCell(const svmp::FE::assembly::AssemblyContext& ctx, svmp::FE::LocalIndex q) override
    {
        return integrand_->evaluateCell(ctx, q);
    }
    [[nodiscard]] Real evaluateCellTotal(const svmp::FE::assembly::AssemblyContext& ctx) override
    {
        for (svmp::FE::LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
            terms_.push_back(integrand_->evaluateCell(ctx, q) * ctx.integrationWeight(q));
        }
        return Real{0.0};
    }
    [[nodiscard]] Real evaluateBoundaryFace(const svmp::FE::assembly::AssemblyContext& ctx,
                                            svmp::FE::LocalIndex q,
                                            int marker) override
    {
        return integrand_->evaluateBoundaryFace(ctx, q, marker);
    }
    [[nodiscard]] Real evaluateBoundaryFaceTotal(const svmp::FE::assembly::AssemblyContext& ctx, int marker) override
    {
        for (svmp::FE::LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
            terms_.push_back(integrand_->evaluateBoundaryFace(ctx, q, marker) * ctx.integrationWeight(q));
        }
        return Real{0.0};
    }

private:
    std::shared_ptr<svmp::FE::assembly::FunctionalKernel> integrand_;
    std::vector<Real>& terms_;
};

/// The sum of the form's terms on the one-rank view, in long double, rounded
/// once to double.
[[nodiscard]] Real longDoubleReference(CapView& view, const std::vector<Term>& terms)
{
    forms::SymbolicOptions options;
    options.jit.enable = false;
    long double sum = 0.0L;
    for (const auto& term : terms) {
        std::shared_ptr<svmp::FE::assembly::FunctionalKernel> integrand;
        if (term.measure == Measure::Cell || term.measure == Measure::CutVolume) {
            forms::BoundaryFunctional functional;
            functional.integrand = term.integrand;
            functional.is_domain_functional = true;
            integrand = forms::compileBoundaryFunctionalKernel(functional, options);
        } else {
            integrand = forms::compileBoundaryFunctionalKernel(term.integrand, term.marker, options);
        }
        std::vector<Real> values;
        RecordingKernel kernel(integrand, values);
        switch (term.measure) {
        case Measure::Cell:
            (void)view.assembler.assembleScalar(kernel);
            break;
        case Measure::Boundary:
            (void)view.assembler.assembleBoundaryScalar(kernel, term.marker);
            break;
        case Measure::CutVolume:
            (void)view.assembler.assembleCutVolumeScalar(kernel, *view.cut_context, term.marker, term.side);
            break;
        case Measure::Interface:
            (void)view.assembler.assembleGeneratedInterfaceScalar(kernel, *view.cut_context, term.marker);
            break;
        }
        EXPECT_TRUE(view.assembler.getLastResult().success) << view.assembler.getLastResult().error_message;
        EXPECT_FALSE(values.empty());
        for (const auto value : values) {
            sum += static_cast<long double>(term.sign) * static_cast<long double>(value);
        }
    }
    return static_cast<Real>(sum);
}

[[nodiscard]] Real ulp(Real value)
{
    const Real magnitude = std::abs(value);
    return std::nextafter(magnitude, std::numeric_limits<Real>::infinity()) - magnitude;
}

} // namespace

TEST(FunctionalFormEvaluatorMPI, ValuesAreTheSameAtEveryRankCount)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // The partitioned view (cell c on rank c % size) and, on every rank, the
    // one-rank view of the same strip.
    CapView parallel(rank, size, worldCollective(rank, size));
    CapView serial(0, 1, {});

    const auto negative = svmp::FE::geometry::CutIntegrationSide::Negative;
    const auto positive = svmp::FE::geometry::CutIntegrationSide::Positive;
    const auto x = FormExpr::coordinate().component(0);
    const auto y = FormExpr::coordinate().component(1);
    const auto build = [&](const CapView& view) {
        const auto u = view.u();
        return std::vector<std::vector<Term>>{
            {{1.0, Measure::Cell, -1, negative, Real{0.5} * u * u}},
            {{1.0, Measure::CutVolume, fixture::kInterfaceMarker, negative, Real{9.81} * y}},
            {{1.0, Measure::CutVolume, fixture::kInterfaceMarker, positive, inner(grad(u), grad(u))}},
            {{1.0, Measure::Interface, fixture::kInterfaceMarker, negative, FormExpr::constant(0.072)}},
            {{1.0, Measure::Interface, fixture::kContactMarker, negative, u * x}},
            {{1.0, Measure::Interface, fixture::kWettedWallMarker, negative, u - FormExpr::constant(0.5)}},
            {{1.0, Measure::Cell, -1, negative, x - FormExpr::constant(2.0)},
             {-1.0, Measure::Interface, fixture::kInterfaceMarker, negative, u},
             {1.0, Measure::Boundary, fixture::kWallMarker, negative, u * y + x},
             {-1.0, Measure::CutVolume, fixture::kInterfaceMarker, negative, u}},
        };
    };
    const auto parallel_terms = build(parallel);
    const auto serial_terms = build(serial);
    std::vector<FormExpr> parallel_forms;
    std::vector<FormExpr> serial_forms;
    for (std::size_t i = 0; i < parallel_terms.size(); ++i) {
        parallel_forms.push_back(form(parallel_terms[i]));
        serial_forms.push_back(form(serial_terms[i]));
    }

    const auto distributed = systems::evaluateFunctionalForms(
        parallel.assembler, parallel_forms, parallel.cut_context.get(), systems::CollectiveCommunicator(MPI_COMM_WORLD));
    const auto one_rank = systems::evaluateFunctionalForms(
        serial.assembler, serial_forms, serial.cut_context.get(), systems::CollectiveCommunicator{});
    ASSERT_EQ(distributed.size(), serial_forms.size());
    ASSERT_EQ(one_rank.size(), serial_forms.size());

    for (std::size_t i = 0; i < serial_forms.size(); ++i) {
        SCOPED_TRACE("form " + std::to_string(i));
        // Bitwise: the same rounding of the same exact sum.
        EXPECT_EQ(std::bit_cast<std::uint64_t>(distributed[i].value), std::bit_cast<std::uint64_t>(one_rank[i].value))
            << distributed[i].value << " vs " << one_rank[i].value;
        EXPECT_EQ(std::bit_cast<std::uint64_t>(distributed[i].absolute_mass),
                  std::bit_cast<std::uint64_t>(one_rank[i].absolute_mass));
        EXPECT_GT(one_rank[i].absolute_mass, Real{0.0});

        const Real reference = longDoubleReference(serial, serial_terms[i]);
        EXPECT_LE(std::abs(one_rank[i].value - reference), ulp(reference))
            << one_rank[i].value << " vs long-double " << reference;
    }

    // Every rank holds the same values.
    for (std::size_t i = 0; i < distributed.size(); ++i) {
        double low = distributed[i].value;
        double high = distributed[i].value;
        MPI_Allreduce(MPI_IN_PLACE, &low, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &high, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        EXPECT_EQ(low, high);
    }
    if (rank == 0) {
        RecordProperty("mpi_rank_count", size);
    }
}

TEST(FunctionalFormEvaluatorMPI, AMarkerNoRankKnowsIsRejectedOnEveryRank)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    CapView parallel(rank, size, worldCollective(rank, size));

    const std::vector<FormExpr> forms{FormExpr::constant(1.0).dI(999)};
    int rejected = 0;
    try {
        (void)systems::evaluateFunctionalForms(
            parallel.assembler, forms, parallel.cut_context.get(), systems::CollectiveCommunicator(MPI_COMM_WORLD));
    } catch (const svmp::FE::InvalidArgumentException& error) {
        rejected = std::string(error.what()).find("marker 999") != std::string::npos ? 1 : 0;
    }
    int everywhere = 0;
    MPI_Allreduce(&rejected, &everywhere, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    EXPECT_EQ(everywhere, 1);
}
