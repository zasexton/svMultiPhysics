/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

// A standalone cut geometry on a partitioned mesh: the unfitted Poisson
// problem with a Robin condition on a circle or sphere (cutGeometry, activeOn
// and the typed measures only) gives the one-rank results on two and four
// ranks.  The measure of the cut side is a sum of exact accumulators over the
// owned cells, so it is equal bit for bit; the solution differs only by the
// summation order of the assembly.

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Constraints/AffineConstraints.h"
#include "Constraints/ConstraintPlan.h"
#include "Core/FEConfig.h"
#include "Forms/Vocabulary.h"
#include "LevelSet/LevelSetGeometryService.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Systems/FunctionalFormEvaluator.h"
#include "Systems/SystemCollectives.h"

#if FE_HAS_EIGEN
#include <Eigen/Dense>
#endif

#include <mpi.h>

#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace svmp::FE::level_set::test {
namespace {

using forms::CutVolumeSide;
using forms::FormExpr;

constexpr Real kPi = 3.14159265358979323846;
constexpr Real kRadius = 0.31;

struct BoxArrays {
    int dim{2};
    std::vector<svmp::real_t> x;
    std::vector<svmp::offset_t> offsets{0};
    std::vector<svmp::index_t> connectivity;
    std::vector<svmp::CellShape> shapes;
};

/// The unit square (Triangle3, two per square; Quad4) or cube (Tetra4, six
/// per cube around its main diagonal), n cells per side.
BoxArrays boxArrays(svmp::CellFamily family, int n)
{
    BoxArrays a;
    a.dim = family == svmp::CellFamily::Tetra ? 3 : 2;
    const int nz = a.dim == 3 ? n : 0;
    const auto id = [n](int i, int j, int k) {
        return static_cast<svmp::index_t>((k * (n + 1) + j) * (n + 1) + i);
    };
    for (int k = 0; k <= nz; ++k) {
        for (int j = 0; j <= n; ++j) {
            for (int i = 0; i <= n; ++i) {
                a.x.push_back(static_cast<svmp::real_t>(i) / n);
                a.x.push_back(static_cast<svmp::real_t>(j) / n);
                if (a.dim == 3) {
                    a.x.push_back(static_cast<svmp::real_t>(k) / n);
                }
            }
        }
    }
    svmp::CellShape shape{};
    shape.family = family;
    shape.num_corners = family == svmp::CellFamily::Triangle ? 3 : 4;
    shape.order = 1;
    const auto add = [&](std::initializer_list<svmp::index_t> cell) {
        a.connectivity.insert(a.connectivity.end(), cell);
        a.offsets.push_back(static_cast<svmp::offset_t>(a.connectivity.size()));
        a.shapes.push_back(shape);
    };
    for (int k = 0; k < std::max(nz, 1); ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const auto c = [&](int p, int q, int r) { return id(i + p, j + q, a.dim == 3 ? k + r : 0); };
                if (family == svmp::CellFamily::Triangle) {
                    add({c(0, 0, 0), c(1, 0, 0), c(1, 1, 0)});
                    add({c(0, 0, 0), c(1, 1, 0), c(0, 1, 0)});
                } else if (family == svmp::CellFamily::Quad) {
                    add({c(0, 0, 0), c(1, 0, 0), c(1, 1, 0), c(0, 1, 0)});
                } else {
                    constexpr std::array<std::array<int, 3>, 6> paths{
                        {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}, {1, 0, 2}, {0, 2, 1}, {2, 1, 0}}};
                    for (std::size_t p = 0; p < paths.size(); ++p) {
                        std::array<std::array<int, 3>, 4> v{};
                        for (std::size_t s = 1; s < 4u; ++s) {
                            v[s] = v[s - 1];
                            v[s][static_cast<std::size_t>(paths[p][s - 1])] = 1;
                        }
                        if (p >= 3u) {
                            std::swap(v[1], v[2]);
                        }
                        add({c(v[0][0], v[0][1], v[0][2]), c(v[1][0], v[1][1], v[1][2]),
                             c(v[2][0], v[2][1], v[2][2]), c(v[3][0], v[3][1], v[3][2])});
                    }
                }
            }
        }
    }
    return a;
}

std::shared_ptr<svmp::Mesh> serialMesh(const BoxArrays& a)
{
    auto base = std::make_shared<svmp::MeshBase>();
    base->build_from_arrays(a.dim, a.x, a.offsets, a.connectivity, a.shapes);
    base->finalize();
    return svmp::create_mesh(std::move(base), svmp::MeshComm::self());
}

/// The partition of @p a; small-cut aggregation reads root cells several
/// layers away (with two ghost layers its simplex results change with the
/// rank count), so the halo has four.
std::shared_ptr<svmp::Mesh> partitionedMesh(const BoxArrays& a)
{
    auto mesh = std::make_shared<svmp::Mesh>(svmp::MeshComm(MPI_COMM_WORLD));
    mesh->build_from_arrays_global_and_partition(a.dim, a.x, a.offsets, a.connectivity, a.shapes,
                                                 svmp::PartitionHint::Cells, /*ghost_layers=*/4,
                                                 {{"partition_method", "block"}});
    return mesh;
}

ElementType linearElement(svmp::CellFamily family)
{
    return family == svmp::CellFamily::Triangle ? ElementType::Triangle3
           : family == svmp::CellFamily::Quad   ? ElementType::Quad4
                                                : ElementType::Tetra4;
}

Real exactT(Real x, Real y, Real z) { return std::cos(kPi * x) * std::cos(kPi * y) * std::cos(kPi * z); }

FormExpr scalarCoefficient(std::string name, std::function<Real(Real, Real, Real)> f)
{
    return FormExpr::coefficient(std::move(name), forms::ScalarCoefficient(std::move(f)));
}

/// The measure of the inclusion, the L2 error of T and the integral of T on it.
struct CutPoissonResult {
    Real measure{0.0};
    Real error{0.0};
    Real integral{0.0};
};

/// -k Laplace(T) = f inside the circle or sphere {phi < 0}, a Robin condition on its
/// generated interface with the data of the exact solution; solved on @p mesh with the
/// system assembled into a dense global view and combined over the ranks.
CutPoissonResult solveCutPoisson(const std::shared_ptr<svmp::Mesh>& mesh, svmp::CellFamily family, MPI_Comm comm)
{
    const int dim = mesh->dim();
    auto space = std::make_shared<spaces::H1Space>(linearElement(family), 1);
    systems::FESystem system(mesh);
    const auto phi_id = system.addField(
        systems::FieldSpec{.name = "phi", .space = space, .source_kind = systems::FieldSourceKind::PrescribedData});
    const auto T_id = system.addField(systems::FieldSpec{.name = "T", .space = space});
    system.addOperator("op");

    auto G = cutGeometry(system, *mesh, {.level_set_field_name = "phi", .domain_id = "inclusion"});
    const auto L = G.side(CutVolumeSide::Negative);
    const auto T = forms::StateField(T_id, *space, "T");
    const auto s = forms::TestField(T_id, *space, "s");
    const auto k = FormExpr::constant(2.0);
    const auto alpha = FormExpr::constant(3.0);
    const auto T_inf = FormExpr::constant(0.25);
    const auto T_exact = scalarCoefficient("T_exact", exactT);
    std::vector<FormExpr> gradient;
    for (int d = 0; d < dim; ++d) {
        gradient.push_back(scalarCoefficient("dT_exact_" + std::to_string(d), [d](Real x, Real y, Real z) {
            const std::array<Real, 3> c{std::cos(kPi * x), std::cos(kPi * y), std::cos(kPi * z)};
            const std::array<Real, 3> sn{std::sin(kPi * x), std::sin(kPi * y), std::sin(kPi * z)};
            Real value = -kPi;
            for (std::size_t e = 0; e < 3u; ++e) {
                value *= e == static_cast<std::size_t>(d) ? sn[e] : c[e];
            }
            return value;
        }));
    }
    const auto f = scalarCoefficient(
        "f", [dim](Real x, Real y, Real z) { return 2.0 * static_cast<Real>(dim) * kPi * kPi * exactT(x, y, z); });
    const auto g = k * forms::inner(FormExpr::asVector(gradient), L.outwardNormal()) + alpha * (T_exact - T_inf);
    (void)systems::installFormulation(system, "op", {T_id},
                                      (k * forms::inner(forms::grad(T), forms::grad(s)) - f * s).dx(L.volume()) +
                                          ((alpha * (T - T_inf) - g) * s).dI(L.interface()));
    constraints::ConstraintPlan plan(system);
    plan.activeOn({T_id}, L);
    plan.commit();
    system.setup();

    const std::array<Real, 3> centre =
        dim == 2 ? std::array<Real, 3>{0.5013, 0.4987, 0.0} : std::array<Real, 3>{0.5031, 0.4969, 0.5017};
    std::vector<Real> phi_at_vertices(mesh->n_vertices());
    for (std::size_t v = 0; v < phi_at_vertices.size(); ++v) {
        Real r2 = 0.0;
        for (int d = 0; d < dim; ++d) {
            const auto dx = static_cast<Real>(mesh->X_ref()[v * static_cast<std::size_t>(dim) +
                                                            static_cast<std::size_t>(d)]) -
                            centre[static_cast<std::size_t>(d)];
            r2 += dx * dx;
        }
        phi_at_vertices[v] = std::sqrt(r2) - kRadius;
    }
    std::vector<Real> coefficients(static_cast<std::size_t>(system.fieldDofHandler(phi_id).getNumDofs()), 0.0);
    (void)system.projectMeshVertexValuesToFieldCoefficients(phi_id, phi_at_vertices, 1u, coefficients);
    system.setPrescribedFieldCoefficients(phi_id, coefficients);
    (void)G.refresh();

    // Each rank writes its owned rows; the sum over the ranks is the global system.
    const auto n = system.dofHandler().getNumDofs();
    std::vector<Real> state(static_cast<std::size_t>(n), 0.0);
    assembly::DenseSystemView out(n);
    out.zero();
    systems::SystemStateView view;
    view.u = state;
    systems::AssemblyRequest request;
    request.op = "op";
    request.want_matrix = true;
    request.want_vector = true;
    request.suppress_constraint_inhomogeneity = true;
    EXPECT_TRUE(system.assemble(request, view, &out, &out).success);
    std::vector<Real> matrix(out.matrixData().begin(), out.matrixData().end());
    std::vector<Real> vector(out.vectorData().begin(), out.vectorData().end());
    MPI_Allreduce(MPI_IN_PLACE, matrix.data(), static_cast<int>(matrix.size()), MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, vector.data(), static_cast<int>(vector.size()), MPI_DOUBLE, MPI_SUM, comm);
#if FE_HAS_EIGEN
    const auto size = static_cast<Eigen::Index>(n);
    const Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A(matrix.data(),
                                                                                                    size, size);
    const Eigen::Map<const Eigen::Matrix<Real, Eigen::Dynamic, 1>> b(vector.data(), size);
    const Eigen::Matrix<Real, Eigen::Dynamic, 1> step = A.partialPivLu().solve(-b);
    for (Eigen::Index i = 0; i < size; ++i) {
        state[static_cast<std::size_t>(i)] += step[i];
    }
#endif
    system.constraints().distribute(state);

    const auto e = T - scalarCoefficient("T_exact", exactT);
    const std::vector<FormExpr> functionals{FormExpr::constant(1.0).dx(L.volume()), (e * e).dx(L.volume()),
                                            T.dx(L.volume())};
    view.u = state;
    const auto values =
        systems::evaluateFunctionalForms(system, functionals, view, systems::CollectiveCommunicator(comm));
    return {values[0].value, std::sqrt(values[1].value), values[2].value};
}

void expectOneRankResults(MPI_Comm comm)
{
    struct Case {
        svmp::CellFamily family;
        const char* name;
        int n;
    };
    for (const auto& [family, name, n] : {Case{svmp::CellFamily::Triangle, "Triangle3", 12},
                                          Case{svmp::CellFamily::Quad, "Quad4", 12},
                                          Case{svmp::CellFamily::Tetra, "Tetra4", 6}}) {
        SCOPED_TRACE(name);
        const auto arrays = boxArrays(family, n);
        const auto one = solveCutPoisson(serialMesh(arrays), family, MPI_COMM_SELF);
        const auto many = solveCutPoisson(partitionedMesh(arrays), family, comm);
        EXPECT_EQ(std::memcmp(&one.measure, &many.measure, sizeof(Real)), 0)
            << one.measure << " " << many.measure;
        EXPECT_NEAR(many.error, one.error, 1e-12 * std::max(Real(1.0), one.error));
        EXPECT_NEAR(many.integral, one.integral, 1e-12);
        EXPECT_GT(one.error, 0.0);
        EXPECT_LT(one.error, 0.05);
    }
}

/// n x n Quad9 cells of the unit square.
BoxArrays quad9Arrays(int n)
{
    BoxArrays a;
    const int m = 2 * n;
    for (int j = 0; j <= m; ++j) {
        for (int i = 0; i <= m; ++i) {
            a.x.push_back(static_cast<svmp::real_t>(i) / m);
            a.x.push_back(static_cast<svmp::real_t>(j) / m);
        }
    }
    const auto id = [m](int i, int j) { return static_cast<svmp::index_t>(j * (m + 1) + i); };
    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 2;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const int p = 2 * i, q = 2 * j;
            a.connectivity.insert(a.connectivity.end(),
                                  {id(p, q), id(p + 2, q), id(p + 2, q + 2), id(p, q + 2), id(p + 1, q),
                                   id(p + 2, q + 1), id(p + 1, q + 2), id(p, q + 1), id(p + 1, q + 1)});
            a.offsets.push_back(static_cast<svmp::offset_t>(a.connectivity.size()));
            a.shapes.push_back(shape);
        }
    }
    return a;
}

/// The topology key of the HighOrderImplicit cut of a wave on @p mesh (P2 on
/// Quad9 cells, subdivision depth 6 as in the MMS deck).
std::uint64_t highOrderCutKey(const std::shared_ptr<svmp::Mesh>& mesh)
{
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad9, 2);
    systems::FESystem system(mesh);
    const auto phi_id = system.addField(
        systems::FieldSpec{.name = "phi", .space = space, .source_kind = systems::FieldSourceKind::PrescribedData});
    (void)system.addField(systems::FieldSpec{.name = "T", .space = space});
    auto G = cutGeometry(system, *mesh, {.level_set_field_name = "phi", .domain_id = "wave"},
                         {.quadrature_order = 2,
                          .geometry_mode = GeneratedInterfaceGeometryMode::HighOrderImplicit,
                          .implicit_cut_backend = ImplicitCutQuadratureBackend::SayeHyperrectangle,
                          .implicit_cut_max_subdivision_depth = 6});
    (void)G.side(CutVolumeSide::Negative);
    system.setup();
    std::vector<Real> values(mesh->n_vertices());
    for (std::size_t v = 0; v < values.size(); ++v) {
        const Real x = mesh->X_ref()[2 * v];
        const Real y = mesh->X_ref()[2 * v + 1];
        values[v] = y - 0.43 - 0.05 * std::sin(2.0 * kPi * x);
    }
    std::vector<Real> coefficients(static_cast<std::size_t>(system.fieldDofHandler(phi_id).getNumDofs()), 0.0);
    (void)system.projectMeshVertexValuesToFieldCoefficients(phi_id, values, 1u, coefficients);
    system.setPrescribedFieldCoefficients(phi_id, coefficients);
    return G.refresh().topology_key;
}

void expectOneRankCutStructureKey()
{
    const auto arrays = quad9Arrays(8);
    const auto one = highOrderCutKey(serialMesh(arrays));
    const auto many = highOrderCutKey(partitionedMesh(arrays));
    EXPECT_NE(one, 0u);
    EXPECT_EQ(many, one);
}

} // namespace

TEST(CutGeometryMPI, TwoRankCutStructureKeyIsTheOneRankKey)
{
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        GTEST_SKIP() << "Run with exactly 2 MPI ranks";
    }
    expectOneRankCutStructureKey();
}

TEST(CutGeometryMPI, FourRankCutStructureKeyIsTheOneRankKey)
{
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 4) {
        GTEST_SKIP() << "Run with exactly 4 MPI ranks";
    }
    expectOneRankCutStructureKey();
}

TEST(CutGeometryMPI, TwoRankCutPoissonGivesTheOneRankResults)
{
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        GTEST_SKIP() << "Run with exactly 2 MPI ranks";
    }
    expectOneRankResults(MPI_COMM_WORLD);
}

TEST(CutGeometryMPI, FourRankCutPoissonGivesTheOneRankResults)
{
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 4) {
        GTEST_SKIP() << "Run with exactly 4 MPI ranks";
    }
    expectOneRankResults(MPI_COMM_WORLD);
}

} // namespace svmp::FE::level_set::test
