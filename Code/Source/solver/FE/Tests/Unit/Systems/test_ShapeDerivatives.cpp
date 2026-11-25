/**
 * @file test_ShapeDerivatives.cpp
 * @brief Complete shape derivatives of moving-domain forms (opt-in tangent path)
 *
 * A moving domain assembles on x = X + d, d the mesh-displacement unknown.
 * Each probe residual is installed on such a domain and its Jacobian applied
 * to a direction dd of the mesh columns is compared with central differences
 * of the residual, the coordinates moved for every perturbed state by the
 * coordinate update Newton uses (updateCurrentCoordinatesFromMeshDisplacement).
 *
 * The CompleteShapeDerivative suite runs representative cells in CTest.  The
 * full sweep over every cell type and order (Tri3, Quad4, Tet4, Hex8, Wedge6,
 * Pyr5; straight cells with P1 and P2 fields, curved P2 cells) is the suite
 * CompleteShapeDerivativeSweep, which the CTest entry leaves out; run it on
 * demand (about 12 minutes):
 *
 *     test_fe_systems --gtest_filter='CompleteShapeDerivativeSweep.*'
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Basis/LagrangeBasis.h"
#include "Basis/NodeOrderingConventions.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Forms/MonolithicCellKernel.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Systems/ALEBinding.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Systems/TimeIntegrator.h"

#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {

namespace forms = svmp::FE::forms;
namespace systems = svmp::FE::systems;
using forms::FormExpr;
using svmp::FE::ElementType;
using svmp::FE::FieldId;
using svmp::FE::GlobalIndex;
using svmp::FE::Real;

constexpr int kTop = 4;
constexpr int kSide = 5;
constexpr int kGamma = 7;   // interior faces on the undistorted x = 1/2 (fitted interface)

// ---------------------------------------------------------------------------
// Meshes: the unit square or cube cut into cells of one type, with the
// Lagrange nodes of the requested geometry order placed through the
// first-order map of each cell, then every node moved by a smooth field (the
// cells of order 2 become curved).  Faces on the last coordinate = 1 of the
// undistorted box carry kTop, the other boundary faces kSide.
// ---------------------------------------------------------------------------

enum class Cell { Triangle, Quad, Tetra, Hex, Wedge, Pyramid };

const char* cellName(Cell cell)
{
    switch (cell) {
        case Cell::Triangle: return "Triangle";
        case Cell::Quad: return "Quad";
        case Cell::Tetra: return "Tetra";
        case Cell::Hex: return "Hex";
        case Cell::Wedge: return "Wedge";
        case Cell::Pyramid: return "Pyramid";
    }
    return "?";
}

int cellDimension(Cell cell)
{
    return (cell == Cell::Triangle || cell == Cell::Quad) ? 2 : 3;
}

ElementType canonicalType(Cell cell)
{
    switch (cell) {
        case Cell::Triangle: return ElementType::Triangle3;
        case Cell::Quad: return ElementType::Quad4;
        case Cell::Tetra: return ElementType::Tetra4;
        case Cell::Hex: return ElementType::Hex8;
        case Cell::Wedge: return ElementType::Wedge6;
        case Cell::Pyramid: return ElementType::Pyramid5;
    }
    return ElementType::Unknown;
}

svmp::CellShape cellShape(Cell cell, int order)
{
    svmp::CellShape shape{};
    switch (cell) {
        case Cell::Triangle: shape.family = svmp::CellFamily::Triangle; shape.num_corners = 3; break;
        case Cell::Quad: shape.family = svmp::CellFamily::Quad; shape.num_corners = 4; break;
        case Cell::Tetra: shape.family = svmp::CellFamily::Tetra; shape.num_corners = 4; break;
        case Cell::Hex: shape.family = svmp::CellFamily::Hex; shape.num_corners = 8; break;
        case Cell::Wedge: shape.family = svmp::CellFamily::Wedge; shape.num_corners = 6; break;
        case Cell::Pyramid: shape.family = svmp::CellFamily::Pyramid; shape.num_corners = 5; break;
    }
    shape.order = order;
    return shape;
}

using Point = std::array<double, 3>;

// The corners of each first-order cell, in the reference corner order: nx
// cells along x, n along the other axes.
std::vector<std::vector<Point>> boxCellCorners(Cell cell, int nx, int n)
{
    std::vector<std::vector<Point>> out;
    const double sx = 1.0 / nx;
    const double s = 1.0 / n;
    const auto p = [sx, s](int i, int j, int k) { return Point{i * sx, j * s, k * s}; };
    const auto signedVolume = [](const Point& o, const Point& a, const Point& b, const Point& c) {
        const Point u{a[0] - o[0], a[1] - o[1], a[2] - o[2]};
        const Point v{b[0] - o[0], b[1] - o[1], b[2] - o[2]};
        const Point w{c[0] - o[0], c[1] - o[1], c[2] - o[2]};
        return u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0]) +
               u[2] * (v[0] * w[1] - v[1] * w[0]);
    };
    if (cellDimension(cell) == 2) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < nx; ++i) {
                const auto a = p(i, j, 0), b = p(i + 1, j, 0), c = p(i + 1, j + 1, 0), d = p(i, j + 1, 0);
                if (cell == Cell::Quad) {
                    out.push_back({a, b, c, d});
                } else {
                    out.push_back({a, b, c});
                    out.push_back({a, c, d});
                }
            }
        }
        return out;
    }
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < nx; ++i) {
                const auto a = p(i, j, k), b = p(i + 1, j, k), c = p(i + 1, j + 1, k), d = p(i, j + 1, k);
                const auto e = p(i, j, k + 1), f = p(i + 1, j, k + 1), g = p(i + 1, j + 1, k + 1),
                           h = p(i, j + 1, k + 1);
                switch (cell) {
                    case Cell::Hex:
                        out.push_back({a, b, c, d, e, f, g, h});
                        break;
                    case Cell::Wedge:
                        out.push_back({a, b, c, e, f, g});
                        out.push_back({a, c, d, e, g, h});
                        break;
                    case Cell::Tetra: {
                        // The six axis paths from a to g.
                        const std::array<std::array<int, 3>, 6> orders{
                            {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}}};
                        for (const auto& order : orders) {
                            std::array<int, 3> idx{i, j, k};
                            std::vector<Point> tet{p(idx[0], idx[1], idx[2])};
                            for (const int axis : order) {
                                ++idx[static_cast<std::size_t>(axis)];
                                tet.push_back(p(idx[0], idx[1], idx[2]));
                            }
                            if (signedVolume(tet[0], tet[1], tet[2], tet[3]) < 0.0) {
                                std::swap(tet[1], tet[2]);
                            }
                            out.push_back(std::move(tet));
                        }
                        break;
                    }
                    case Cell::Pyramid: {
                        const Point apex{(i + 0.5) * sx, (j + 0.5) * s, (k + 0.5) * s};
                        const std::array<std::array<Point, 4>, 6> faces{{
                            {a, b, c, d}, {e, f, g, h}, {a, b, f, e}, {b, c, g, f}, {c, d, h, g}, {d, a, e, h}}};
                        for (auto face : faces) {
                            if (signedVolume(face[0], face[1], face[2], apex) < 0.0) {
                                std::swap(face[1], face[3]);
                            }
                            out.push_back({face[0], face[1], face[2], face[3], apex});
                        }
                        break;
                    }
                    default:
                        break;
                }
            }
        }
    }
    return out;
}

Point distortion(const Point& x, int dim, double amplitude)
{
    Point dx{amplitude * std::sin(2.3 * x[0] + 1.1 * x[1] + 0.7 * x[2] + 0.3),
             amplitude * std::sin(1.7 * x[0] - 2.1 * x[1] + 0.5 * x[2] + 1.1),
             amplitude * std::sin(0.9 * x[0] + 1.3 * x[1] - 1.9 * x[2] + 0.6)};
    for (int c = dim; c < 3; ++c) {
        dx[static_cast<std::size_t>(c)] = 0.0;
    }
    return dx;
}

struct BoxMesh {
    std::shared_ptr<svmp::Mesh> mesh;
    int dim{0};
    ElementType element{ElementType::Unknown};
};

BoxMesh makeBoxMesh(Cell cell, int nx, int n, int order, double amplitude)
{
    const int dim = cellDimension(cell);
    const auto type = canonicalType(cell);
    const auto reference = svmp::FE::basis::ReferenceNodeLayout::get_lagrange_node_coords(type, order);
    const svmp::FE::basis::LagrangeBasis linear(type, 1);

    std::map<std::array<long long, 3>, svmp::index_t> index_of;
    std::vector<Point> nodes;
    std::vector<std::vector<svmp::index_t>> cells;
    std::vector<Real> values;
    for (const auto& corners : boxCellCorners(cell, nx, n)) {
        std::vector<svmp::index_t> connectivity;
        for (const auto& xi : reference) {
            linear.evaluate_values(xi, values);
            Point x{0.0, 0.0, 0.0};
            for (std::size_t c = 0; c < corners.size(); ++c) {
                for (int k = 0; k < 3; ++k) {
                    x[static_cast<std::size_t>(k)] += values[c] * corners[c][static_cast<std::size_t>(k)];
                }
            }
            const std::array<long long, 3> key{std::llround(x[0] * 1e9), std::llround(x[1] * 1e9),
                                               std::llround(x[2] * 1e9)};
            const auto [it, inserted] = index_of.try_emplace(key, static_cast<svmp::index_t>(nodes.size()));
            if (inserted) {
                nodes.push_back(x);
            }
            connectivity.push_back(it->second);
        }
        cells.push_back(std::move(connectivity));
    }

    std::vector<svmp::real_t> x_ref;
    for (const auto& x : nodes) {
        const auto dx = distortion(x, dim, amplitude);
        for (int k = 0; k < dim; ++k) {
            x_ref.push_back(static_cast<svmp::real_t>(x[static_cast<std::size_t>(k)] + dx[static_cast<std::size_t>(k)]));
        }
    }
    std::vector<svmp::offset_t> offsets{0};
    std::vector<svmp::index_t> connectivity;
    for (const auto& c : cells) {
        connectivity.insert(connectivity.end(), c.begin(), c.end());
        offsets.push_back(static_cast<svmp::offset_t>(connectivity.size()));
    }
    auto base = std::make_shared<svmp::MeshBase>();
    base->build_from_arrays(dim, x_ref, offsets, connectivity,
                            std::vector<svmp::CellShape>(cells.size(), cellShape(cell, order)));
    base->finalize();
    base->register_label("top", static_cast<svmp::label_t>(kTop));
    base->register_label("side", static_cast<svmp::label_t>(kSide));
    for (const auto face : base->boundary_faces()) {
        const auto vertices = base->face_vertices(face);
        const bool top = std::all_of(vertices.begin(), vertices.end(), [&](svmp::index_t v) {
            return std::abs(nodes[static_cast<std::size_t>(v)][static_cast<std::size_t>(dim - 1)] - 1.0) < 1e-12;
        });
        base->set_boundary_label(face, static_cast<svmp::label_t>(top ? kTop : kSide));
    }
    for (svmp::index_t face = 0; face < static_cast<svmp::index_t>(base->n_faces()); ++face) {
        if (base->face_cells(face)[1] == svmp::INVALID_INDEX) {
            continue;
        }
        const auto vertices = base->face_vertices(face);
        const bool mid = std::all_of(vertices.begin(), vertices.end(), [&](svmp::index_t v) {
            return std::abs(nodes[static_cast<std::size_t>(v)][0] - 0.5) < 1e-12;
        });
        if (mid) {
            base->add_to_set(svmp::EntityKind::Face, "gamma", face);
        }
    }
    return {svmp::create_mesh(std::move(base)), dim, type};
}

// ---------------------------------------------------------------------------
// Probes
// ---------------------------------------------------------------------------

struct Symbols {
    FormExpr u, v;     // vector field and its test function
    FormExpr p, q;     // scalar field and its test function
    FormExpr d, psi;   // mesh displacement and its test function
    int dim{0};
};

enum class Rows { Vector, Scalar, Mesh };

struct Probe {
    std::string name;
    Rows rows;
    std::function<FormExpr(const Symbols&)> residual;
    bool second_derivatives{false};   ///< vanishes for P1 fields on simplices
};

struct ProbeSetup {
    Cell cell{Cell::Triangle};
    int n{2};
    int nx{0};   ///< cells along x (n when 0)
    int geometry_order{1};
    int u_order{1};
    int p_order{1};
    double amplitude{0.04};
    bool complete{false};
    bool jit{false};
    bool interface{false};      ///< register the fitted interface kGamma (n even)
    bool record_residual{false};
};

struct DirectionalDefect {
    Real mesh{0.0};       ///< worst ||J dd - FD|| / ||FD|| over mesh directions
    Real other{0.0};      ///< the same over directions of the other fields
    Real fd_norm{0.0};
    Real jac_norm{0.0};
    std::vector<Real> jdd{};        ///< J dd on the probe rows for the first mesh direction
    std::vector<Real> residual{};   ///< the probe rows at the base state (ProbeSetup::record_residual)
    bool jit_mesh_blocks{false};    ///< every mesh block (cell and boundary terms) ran compiled
    std::string jit_message{};      ///< why it did not
};

Real norm2(const std::vector<Real>& v)
{
    Real s = 0.0;
    for (const auto x : v) s += x * x;
    return std::sqrt(s);
}

DirectionalDefect directionalDefect(const ProbeSetup& setup, const Probe& probe)
{
    const auto box = makeBoxMesh(setup.cell, setup.nx > 0 ? setup.nx : setup.n, setup.n, setup.geometry_order,
                                 setup.amplitude / setup.n);
    const int dim = box.dim;
    auto scalar_u = std::make_shared<svmp::FE::spaces::H1Space>(box.element, setup.u_order);
    auto vector_u = std::make_shared<svmp::FE::spaces::ProductSpace>(scalar_u, dim);
    auto scalar_p = std::make_shared<svmp::FE::spaces::H1Space>(box.element, setup.p_order);
    auto scalar_d = std::make_shared<svmp::FE::spaces::H1Space>(box.element, setup.geometry_order);
    auto vector_d = std::make_shared<svmp::FE::spaces::ProductSpace>(scalar_d, dim);

    systems::FESystem system(box.mesh, svmp::Configuration::Current);
    if (setup.interface) {
        system.setInterfaceMeshFromFaceSet(kGamma, "gamma");
    }
    const auto u_id = system.addField(systems::FieldSpec{.name = "u", .space = vector_u, .components = dim});
    const auto p_id = system.addField(systems::FieldSpec{.name = "p", .space = scalar_p, .components = 1});
    const auto d_id = system.addMeshDisplacementUnknown("d", vector_d, dim);
    systems::MovingDomainOptions moving;
    moving.shape_derivative = setup.complete ? forms::ShapeDerivative::Complete : forms::ShapeDerivative::FrameExplicit;
    (void)systems::movingDomain(system, d_id, moving);

    const Symbols s{
        .u = FormExpr::stateField(u_id, *vector_u, "u"),
        .v = FormExpr::testFunction(u_id, *vector_u, "v"),
        .p = FormExpr::stateField(p_id, *scalar_p, "p"),
        .q = FormExpr::testFunction(p_id, *scalar_p, "q"),
        .d = FormExpr::stateField(d_id, *vector_d, "d"),
        .psi = FormExpr::testFunction(d_id, *vector_d, "psi"),
        .dim = dim,
    };
    systems::FormInstallOptions install;
    install.compiler_options.use_symbolic_tangent = true;
    install.compiler_options.jit.enable = setup.jit;
    // The physics modules' JIT policy: specialized kernels with baked bases.
    install.compiler_options.jit.specialization.enable = setup.jit;
    system.addOperator("probe");
    FieldId row = d_id;
    if (probe.rows == Rows::Mesh) {
        (void)systems::installFormulation(system, "probe", {d_id}, probe.residual(s), install);
    } else {
        row = probe.rows == Rows::Vector ? u_id : p_id;
        const auto other = probe.rows == Rows::Vector ? (dt(s.p) * s.q).dx() : inner(dt(s.u), s.v).dx();
        (void)systems::installFormulation(system, "probe", {u_id, p_id}, probe.residual(s) + other, install);
        systems::FormInstallOptions mesh_install;
        mesh_install.moving_domain_geometry = false;
        (void)systems::installFormulation(system, "probe", {d_id}, inner(grad(s.d), grad(s.psi)).dx(), mesh_install);
    }
    systems::SetupOptions setup_options;
    setup_options.use_constraints_in_assembly = false;
    system.setup(setup_options);

    const auto n = system.dofHandler().getNumDofs();
    const auto range = [&](FieldId field) {
        const auto begin = system.fieldDofOffset(field);
        return std::pair<GlobalIndex, GlobalIndex>{begin, begin + system.fieldDofHandler(field).getNumDofs()};
    };
    const auto mesh_columns = range(d_id);
    const auto rows = range(row);
    const auto is_mesh = [&](GlobalIndex i) { return i >= mesh_columns.first && i < mesh_columns.second; };

    // A displaced (|d| <= 0.25 h), moving mesh and non-trivial fields.
    const Real h = Real(1.0) / setup.n;
    std::vector<Real> values(static_cast<std::size_t>(n), 0.0);
    std::vector<Real> previous(static_cast<std::size_t>(n), 0.0);
    for (GlobalIndex i = 0; i < n; ++i) {
        const auto k = static_cast<std::size_t>(i);
        const auto x = static_cast<Real>(i);
        values[k] = is_mesh(i) ? Real(0.12) * h * std::sin(Real(2.1) * x + Real(0.5))
                               : Real(0.25) * std::sin(Real(1.3) * x + Real(0.2)) + Real(0.1);
        previous[k] = is_mesh(i) ? Real(0.5) * values[k] : Real(0.9) * values[k];
    }

    const auto assemble = [&](const std::vector<Real>& state_values,
                              svmp::FE::assembly::GlobalSystemView* J,
                              svmp::FE::assembly::GlobalSystemView* R) {
        systems::SystemStateView state;
        state.dt = 0.5;
        state.u = std::span<const Real>(state_values);
        state.u_prev = std::span<const Real>(previous);
        const systems::BackwardDifferenceIntegrator integrator;
        const auto context = integrator.buildContext(1, state);
        state.time_integration = &context;
        (void)system.updateCurrentCoordinatesFromMeshDisplacement(state);
        systems::AssemblyRequest request;
        request.op = "probe";
        request.want_matrix = J != nullptr;
        request.want_vector = R != nullptr;
        const auto result = system.assemble(request, state, J, R);
        EXPECT_TRUE(result.success) << result.error_message;
    };
    const auto residualRows = [&](const std::vector<Real>& state_values) {
        svmp::FE::assembly::DenseVectorView R(n);
        R.zero();
        assemble(state_values, nullptr, &R);
        std::vector<Real> out;
        for (GlobalIndex r = rows.first; r < rows.second; ++r) {
            out.push_back(R.getVectorEntry(r));
        }
        return out;
    };

    svmp::FE::assembly::DenseMatrixView J(n);
    J.zero();
    assemble(values, &J, nullptr);

    DirectionalDefect out;
    if (setup.record_residual) {
        out.residual = residualRows(values);
    }
    // Cell blocks are compiled either as one fused kernel or each on its own
    // (the fused kernel falls back to its blocks' kernels).
    {
        bool all = true;
        int mesh_blocks = 0;
        const auto check = [&](FieldId trial, const std::shared_ptr<svmp::FE::assembly::AssemblyKernel>& kernel,
                               forms::IntegralDomain domain, int marker) {
            if (trial != d_id) {
                return;
            }
            ++mesh_blocks;
            const auto* wrapper = dynamic_cast<const forms::jit::JITKernelWrapper*>(kernel.get());
            if (wrapper == nullptr || !wrapper->hasCompiledTangentDispatch(domain, marker)) {
                all = false;
                out.jit_message += "a mesh block is interpreted; ";
            }
        };
        const auto& definition = system.operatorDefinition("probe");
        for (const auto& term : definition.cells) {
            if (const auto* fused = dynamic_cast<const forms::MonolithicCellKernel*>(term.kernel.get())) {
                if (fused->hasCompiledDispatch()) {
                    mesh_blocks += 1;
                    continue;
                }
                out.jit_message += fused->compileMessage() + "; ";
                for (std::size_t b = 0; b < fused->numBlocks(); ++b) {
                    check(fused->blockSpec(b).trial_field, fused->blockSpec(b).fallback_kernel,
                          forms::IntegralDomain::Cell, -1);
                }
            } else {
                check(term.trial_field, term.kernel, forms::IntegralDomain::Cell, -1);
            }
        }
        for (const auto& term : definition.boundary) {
            check(term.trial_field, term.kernel, forms::IntegralDomain::Boundary, term.marker);
        }
        out.jit_mesh_blocks = all && mesh_blocks > 0;
    }
    const Real eps = 1e-6;
    for (int k = 0; k < 3; ++k) {
        const bool mesh_direction = k < 2;
        std::vector<Real> direction(static_cast<std::size_t>(n), 0.0);
        for (GlobalIndex i = 0; i < n; ++i) {
            if (is_mesh(i) == mesh_direction) {
                direction[static_cast<std::size_t>(i)] =
                    std::sin(Real(0.7 + k) * static_cast<Real>(i) + Real(0.3 * k));
            }
        }
        auto plus = values, minus = values;
        for (GlobalIndex i = 0; i < n; ++i) {
            plus[static_cast<std::size_t>(i)] += eps * direction[static_cast<std::size_t>(i)];
            minus[static_cast<std::size_t>(i)] -= eps * direction[static_cast<std::size_t>(i)];
        }
        const auto r_plus = residualRows(plus);
        const auto r_minus = residualRows(minus);
        std::vector<Real> fd(r_plus.size()), jd(r_plus.size()), diff(r_plus.size());
        for (GlobalIndex r = rows.first; r < rows.second; ++r) {
            const auto idx = static_cast<std::size_t>(r - rows.first);
            fd[idx] = (r_plus[idx] - r_minus[idx]) / (Real(2) * eps);
            Real acc = 0.0;
            for (GlobalIndex c = 0; c < n; ++c) {
                acc += J.getMatrixEntry(r, c) * direction[static_cast<std::size_t>(c)];
            }
            jd[idx] = acc;
            diff[idx] = acc - fd[idx];
        }
        const Real fd_norm = norm2(fd);
        const Real defect = fd_norm > 1e-12 ? norm2(diff) / fd_norm : norm2(diff);
        if (k == 0) {
            out.jdd = jd;
        }
        if (mesh_direction) {
            out.mesh = std::max(out.mesh, defect);
            out.fd_norm = std::max(out.fd_norm, fd_norm);
            out.jac_norm = std::max(out.jac_norm, norm2(jd));
        } else {
            out.other = std::max(out.other, defect);
        }
    }
    return out;
}

} // namespace

namespace {

FormExpr zeroTimes(const FormExpr& f)
{
    return FormExpr::constant(0.0) * f;
}

// The volume probes: every geometric dependence of a cell integral.
std::vector<Probe> volumeProbes()
{
    using namespace svmp::FE::forms;
    return {
        {"q dx", Rows::Scalar, [](const Symbols& s) { return (s.q + zeroTimes(s.p * s.q)).dx(); }},
        {"u.v dx", Rows::Vector, [](const Symbols& s) { return inner(s.u, s.v).dx(); }},
        {"x().v dx", Rows::Vector,
         [](const Symbols& s) { return (inner(x(), s.v) + zeroTimes(inner(s.u, s.v))).dx(); }},
        {"currentCoordinate().v dx", Rows::Vector,
         [](const Symbols& s) { return (inner(currentCoordinate(), s.v) + zeroTimes(inner(s.u, s.v))).dx(); }},
        {"q div(u) dx", Rows::Scalar, [](const Symbols& s) { return (s.q * div(s.u) + zeroTimes(s.p * s.q)).dx(); }},
        {"-p div(v) dx", Rows::Vector,
         [](const Symbols& s) { return (-s.p * div(s.v) + zeroTimes(inner(s.u, s.v))).dx(); }},
        {"grad(u):grad(v) dx", Rows::Vector, [](const Symbols& s) { return inner(grad(s.u), grad(s.v)).dx(); }},
        {"grad(p).grad(q) dx", Rows::Scalar, [](const Symbols& s) { return inner(grad(s.p), grad(s.q)).dx(); }},
        {"(grad(u) u).v dx", Rows::Vector, [](const Symbols& s) { return inner(grad(s.u) * s.u, s.v).dx(); }},
        {"(grad(u) (u - w)).v dx", Rows::Vector,
         [](const Symbols& s) { return inner(grad(s.u) * (s.u - meshVelocity()), s.v).dx(); }},
        {"div(w) u.v dx", Rows::Vector,
         [](const Symbols& s) { return (div(meshVelocity()) * inner(s.u, s.v)).dx(); }},
        {"w.v dx", Rows::Vector,
         [](const Symbols& s) { return (inner(meshVelocity(), s.v) + zeroTimes(inner(s.u, s.v))).dx(); }},
        {"tr(J()) q dx", Rows::Scalar,
         [](const Symbols& s) { return (trace(J()) * s.q + zeroTimes(s.p * s.q)).dx(); }},
        {"detJ() q dx", Rows::Scalar, [](const Symbols& s) { return (detJ() * s.q + zeroTimes(s.p * s.q)).dx(); }},
        {"tr(currentJacobian()) q dx", Rows::Scalar,
         [](const Symbols& s) { return (trace(currentJacobian()) * s.q + zeroTimes(s.p * s.q)).dx(); }},
        {"currentJacobianDeterminant() q dx", Rows::Scalar,
         [](const Symbols& s) { return (currentJacobianDeterminant() * s.q + zeroTimes(s.p * s.q)).dx(); }},
        {"(u.Jinv^T Jinv u) q dx (VMS metric)", Rows::Scalar,
         [](const Symbols& s) {
             return (inner(s.u, transpose(Jinv()) * Jinv() * s.u) * s.q + zeroTimes(s.p * s.q)).dx();
         }},
        {"h() q dx", Rows::Scalar, [](const Symbols& s) { return (h() * s.q + zeroTimes(s.p * s.q)).dx(); }},
        {"vol() q dx", Rows::Scalar, [](const Symbols& s) { return (vol() * s.q + zeroTimes(s.p * s.q)).dx(); }},
        {"div(2 sym(grad(u))).v dx (viscous strong residual)", Rows::Vector,
         [](const Symbols& s) {
             return inner(div(FormExpr::constant(2.0) * sym(grad(s.u))), s.v).dx();
         },
         true},
        {"tr(hessian(p)) q dx", Rows::Scalar,
         [](const Symbols& s) { return (trace(hessian(s.p)) * s.q).dx(); }, true},
        {"curl(u).curl(v) dx", Rows::Vector, [](const Symbols& s) { return inner(curl(s.u), curl(s.v)).dx(); }},
        {"f(x) q dx (coefficient callback)", Rows::Scalar,
         [](const Symbols& s) {
             const auto f = FormExpr::coefficient(
                 "f", [](Real x, Real y, Real z) { return std::sin(1.3 * x + 0.7 * y - 0.4 * z) + x * y; });
             return (f * s.q + zeroTimes(s.p * s.q)).dx();
         }},
        {"grad(d):grad(psi) dx (mesh rows)", Rows::Mesh,
         [](const Symbols& s) { return inner(grad(s.d), grad(s.psi)).dx(); }},
    };
}

struct MeshCase {
    Cell cell;
    int geometry_order;
    int field_order;
    int n;
};

std::string caseName(const MeshCase& c)
{
    return std::string(cellName(c.cell)) + " geometry P" + std::to_string(c.geometry_order) + ", fields P" +
           std::to_string(c.field_order);
}

std::vector<MeshCase> everyMeshCase()
{
    std::vector<MeshCase> out;
    for (const auto cell : {Cell::Triangle, Cell::Quad, Cell::Tetra, Cell::Hex, Cell::Wedge, Cell::Pyramid}) {
        const int n = cellDimension(cell) == 2 ? 2 : 1;
        out.push_back({cell, 1, 1, n});   // straight cells (affine simplices), P1 fields
        out.push_back({cell, 1, 2, n});   // P2 fields on straight cells
        out.push_back({cell, 2, 2, n});   // curved P2 cells, P2 fields
    }
    return out;
}

// The representative cases of the CTest suite: one per cell family on
// straight (P1-geometry) cells with P1 fields, and one curved P2 case with P2
// fields per dimension.  CompleteShapeDerivativeSweep runs every case of
// everyMeshCase() on demand.
std::vector<MeshCase> representativeMeshCases()
{
    return {{Cell::Triangle, 1, 1, 2}, {Cell::Quad, 1, 1, 2}, {Cell::Tetra, 1, 1, 1}, {Cell::Hex, 1, 1, 1},
            {Cell::Wedge, 1, 1, 1},    {Cell::Pyramid, 1, 1, 1}, {Cell::Triangle, 2, 2, 1}, {Cell::Tetra, 2, 2, 1}};
}

void runProbes(const std::vector<Probe>& probes, const char* label,
               const std::vector<MeshCase>& mesh_cases = everyMeshCase())
{
    for (const auto& mesh_case : mesh_cases) {
        Real worst = 0.0;
        std::string worst_probe;
        for (const auto& probe : probes) {
            SCOPED_TRACE(caseName(mesh_case) + ": " + probe.name);
            ProbeSetup setup;
            setup.cell = mesh_case.cell;
            setup.n = mesh_case.n;
            setup.geometry_order = mesh_case.geometry_order;
            setup.u_order = mesh_case.field_order;
            setup.p_order = mesh_case.field_order;
            setup.complete = true;
            const auto defect = directionalDefect(setup, probe);
            EXPECT_LT(defect.mesh, 1e-6) << "|J dd| " << defect.jac_norm << ", |FD| " << defect.fd_norm;
            EXPECT_LT(defect.other, 1e-6);
            const bool simplex = mesh_case.cell == Cell::Triangle || mesh_case.cell == Cell::Tetra;
            const bool affine = simplex && mesh_case.geometry_order == 1;
            if (!(probe.second_derivatives && affine && mesh_case.field_order == 1)) {
                EXPECT_GT(defect.fd_norm, 1e-8);
            }
            if (defect.mesh > worst) {
                worst = defect.mesh;
                worst_probe = probe.name;
            }
        }
        std::cout << "[ shape derivative ] " << label << ", " << caseName(mesh_case)
                  << ": worst relative error of J dd " << worst << " (" << worst_probe << ")\n";
    }
}

} // namespace

// Volume terms: the gradient, divergence, Hessian, det J, J^{-1}, cell-size and
// measure rules on every element type, straight and curved, P1 and P2.
TEST(CompleteShapeDerivative, VolumeTermsMatchDirectionalDifferences)
{
    runProbes(volumeProbes(), "volume terms", representativeMeshCases());
}

TEST(CompleteShapeDerivativeSweep, VolumeTermsOnEveryCellAndOrder)
{
    runProbes(volumeProbes(), "volume terms");
}

namespace {

// The facet probes: boundary integrals on kTop, every interior facet, and the
// fitted interface kGamma.
std::vector<Probe> facetProbes()
{
    using namespace svmp::FE::forms;
    const auto n = FormExpr::normal();
    return {
        {"q ds", Rows::Scalar, [](const Symbols& s) { return (s.q + zeroTimes(s.p * s.q)).ds(kTop); }},
        {"u.v ds", Rows::Vector, [](const Symbols& s) { return inner(s.u, s.v).ds(kTop); }},
        {"(u.normal()) q ds", Rows::Scalar,
         [n](const Symbols& s) { return (inner(s.u, n) * s.q + zeroTimes(s.p * s.q)).ds(kTop); }},
        {"(u.currentNormal()) q ds", Rows::Scalar,
         [](const Symbols& s) { return (inner(s.u, currentNormal()) * s.q + zeroTimes(s.p * s.q)).ds(kTop); }},
        {"((u - w).normal()) q ds (kinematic)", Rows::Scalar,
         [n](const Symbols& s) { return (inner(s.u - meshVelocity(), n) * s.q + zeroTimes(s.p * s.q)).ds(kTop); }},
        {"q / hNormal() ds (Nitsche scale)", Rows::Scalar,
         [](const Symbols& s) { return (s.q / hNormal() + zeroTimes(s.p * s.q)).ds(kTop); }},
        {"area() q ds", Rows::Scalar, [](const Symbols& s) { return (area() * s.q + zeroTimes(s.p * s.q)).ds(kTop); }},
        {"vol() q ds", Rows::Scalar, [](const Symbols& s) { return (vol() * s.q + zeroTimes(s.p * s.q)).ds(kTop); }},
        {"h() q ds", Rows::Scalar, [](const Symbols& s) { return (h() * s.q + zeroTimes(s.p * s.q)).ds(kTop); }},
        {"currentMeasure() q ds", Rows::Scalar,
         [](const Symbols& s) { return (currentMeasure() * s.q + zeroTimes(s.p * s.q)).ds(kTop); }},
        {"x().v ds", Rows::Vector,
         [](const Symbols& s) { return (inner(x(), s.v) + zeroTimes(inner(s.u, s.v))).ds(kTop); }},
        {"(grad(u) normal()).v ds", Rows::Vector,
         [n](const Symbols& s) { return inner(grad(s.u) * n, s.v).ds(kTop); }},
        {"(I - n n):grad(v) ds (surface tension)", Rows::Vector,
         [n](const Symbols& s) {
             return (inner(tangentialProjector(n, s.dim), grad(s.v)) + zeroTimes(inner(s.u, s.v))).ds(kTop);
         }},
        {"(d.n)(psi.n)/hNormal() ds (mesh rows)", Rows::Mesh,
         [n](const Symbols& s) { return (inner(s.d, n) * inner(s.psi, n) / hNormal()).ds(kTop); }},
        {"q ds (every boundary face)", Rows::Scalar,
         [](const Symbols& s) { return (s.q + zeroTimes(s.p * s.q)).ds(); }},
        {"avg(p) avg(q) dS", Rows::Scalar, [](const Symbols& s) { return (avg(s.p) * avg(s.q)).dS(); }},
        {"jump(grad p).jump(grad q) avg(h) dS (ghost penalty)", Rows::Scalar,
         [](const Symbols& s) { return (inner(jump(grad(s.p)), jump(grad(s.q))) * avg(hNormal())).dS(); }},
        {"(avg(grad u) n-).avg(v) dS", Rows::Vector,
         [n](const Symbols& s) { return inner(avg(grad(s.u)) * n.minus(), avg(s.v)).dS(); }},
        {"p- q+ dI (fitted interface)", Rows::Scalar,
         [](const Symbols& s) { return (s.p.minus() * s.q.plus()).dI(kGamma); }},
        {"(u-.n-) avg(q) / avg(hNormal()) dI", Rows::Scalar,
         [n](const Symbols& s) { return (inner(s.u.minus(), n.minus()) * avg(s.q) / avg(hNormal())).dI(kGamma); }},
    };
}

} // namespace

// Facet terms: the surface measure, current normals, facet scales (area,
// hNormal) and the cell scales read on facets, on boundary, interior-facet
// and fitted-interface integrals of every element type, straight and curved.
namespace {

void runFacetProbes(const std::vector<MeshCase>& mesh_cases)
{
    const auto probes = facetProbes();
    for (const auto& mesh_case : mesh_cases) {
        Real worst = 0.0;
        std::string worst_probe;
        // Interface meshes of curved (P2) cells are not assembled.
        const bool interface = mesh_case.geometry_order == 1;
        for (const auto& probe : probes) {
            if (!interface && probe.name.find("dI") != std::string::npos) {
                continue;
            }
            SCOPED_TRACE(caseName(mesh_case) + ": " + probe.name);
            ProbeSetup setup;
            setup.cell = mesh_case.cell;
            setup.n = mesh_case.n;
            setup.nx = 2;   // the fitted interface x = 1/2
            setup.geometry_order = mesh_case.geometry_order;
            setup.u_order = mesh_case.field_order;
            setup.p_order = mesh_case.field_order;
            setup.complete = true;
            setup.interface = interface;
            const auto defect = directionalDefect(setup, probe);
            EXPECT_LT(defect.mesh, 1e-6) << "|J dd| " << defect.jac_norm << ", |FD| " << defect.fd_norm;
            EXPECT_LT(defect.other, 1e-6);
            EXPECT_GT(defect.fd_norm, 1e-8);
            if (defect.mesh > worst) {
                worst = defect.mesh;
                worst_probe = probe.name;
            }
        }
        std::cout << "[ shape derivative ] facet terms, " << caseName(mesh_case)
                  << ": worst relative error of J dd " << worst << " (" << worst_probe << ")\n";
    }
}

} // namespace

TEST(CompleteShapeDerivative, FacetTermsMatchDirectionalDifferences)
{
    // Every facet kind (boundary, interior facet, fitted interface) in 2D and
    // 3D on straight cells, and boundary and interior facets of curved cells.
    runFacetProbes({{Cell::Triangle, 1, 1, 1}, {Cell::Hex, 1, 1, 1}, {Cell::Triangle, 2, 2, 1}});
}

TEST(CompleteShapeDerivativeSweep, FacetTermsOnEveryCellAndOrder)
{
    runFacetProbes(everyMeshCase());
}

namespace {

// The installs of a complete moving domain on a triangle mesh, for the refusals.
struct RefusalSystem {
    std::shared_ptr<svmp::Mesh> mesh = makeBoxMesh(Cell::Triangle, 2, 2, 1, 0.02).mesh;
    std::shared_ptr<svmp::FE::spaces::H1Space> scalar = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Triangle3, 1);
    std::shared_ptr<svmp::FE::spaces::ProductSpace> vector = std::make_shared<svmp::FE::spaces::ProductSpace>(scalar, 2);
    systems::FESystem system{mesh, svmp::Configuration::Current};
    FieldId p{svmp::FE::INVALID_FIELD_ID};
    FieldId d{svmp::FE::INVALID_FIELD_ID};

    explicit RefusalSystem(forms::GeometryTangentPath path = forms::GeometryTangentPath::SymbolicRequired)
    {
        p = system.addField(systems::FieldSpec{.name = "p", .space = scalar, .components = 1});
        d = system.addMeshDisplacementUnknown("d", vector, 2);
        systems::MovingDomainOptions moving;
        moving.shape_derivative = forms::ShapeDerivative::Complete;
        moving.geometry_tangent_path = path;
        (void)systems::movingDomain(system, d, moving);
        system.addOperator("probe");
    }

    void install(const FormExpr& residual)
    {
        systems::FormInstallOptions options;
        options.compiler_options.jit.enable = false;
        (void)systems::installFormulation(system, "probe", {p}, residual, options);
    }

    [[nodiscard]] FormExpr pState() const { return FormExpr::stateField(p, *scalar, "p"); }
    [[nodiscard]] FormExpr q() const { return FormExpr::testFunction(p, *scalar, "q"); }
};

std::string installError(const std::function<void()>& install)
{
    try {
        install();
    } catch (const std::exception& e) {
        return e.what();
    }
    return {};
}

} // namespace

// A complete shape derivative differentiates a smooth deformation of a fitted
// domain.  A cut classification that changes with the mesh (cut volumes,
// cut-restricted cells, generated boundary rules) is a separate capability
// and is refused by name, as are the AD reference path, which seeds only the
// frame-explicit terminals, and a mesh displacement richer than the geometry.
TEST(CompleteShapeDerivative, MovingCutTopologyAndTheADPathAreRefusedByName)
{
    {
        RefusalSystem s;
        const auto message = installError([&] {
            s.install((s.pState() * s.q()).dCutVolume(9, forms::CutVolumeSide::Negative));
        });
        EXPECT_NE(message.find("moving cut topology"), std::string::npos) << message;
    }
    {
        RefusalSystem s;
        systems::FESystem::FormCellDomainRestriction restriction;
        restriction.interface_marker = 9;
        restriction.level_set_field = s.p;
        auto scope = s.system.scopedFormInstallCellDomainRestrictions({restriction});
        const auto message = installError([&] { s.install((s.pState() * s.q()).dx()); });
        EXPECT_NE(message.find("moving cut topology"), std::string::npos) << message;
    }
    {
        RefusalSystem s;
        const auto message = installError([&] {
            s.install((s.pState() * s.q()).dS(9, forms::CutVolumeSide::Negative));
        });
        EXPECT_NE(message.find("moving cut topology"), std::string::npos) << message;
    }
    {
        RefusalSystem s(forms::GeometryTangentPath::ADReference);
        const auto message = installError([&] { s.install((s.pState() * s.q()).dx()); });
        EXPECT_NE(message.find("AD reference path"), std::string::npos) << message;
    }
    {
        // A mesh displacement richer than the geometry: the coordinate update
        // moves the geometry nodes only, so x = X + d does not hold in between.
        auto mesh = makeBoxMesh(Cell::Triangle, 2, 2, 1, 0.02).mesh;
        auto scalar = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Triangle3, 1);
        auto quadratic = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Triangle3, 2);
        auto vector = std::make_shared<svmp::FE::spaces::ProductSpace>(quadratic, 2);
        systems::FESystem system(mesh, svmp::Configuration::Current);
        const auto p = system.addField(systems::FieldSpec{.name = "p", .space = scalar, .components = 1});
        const auto d = system.addMeshDisplacementUnknown("d", vector, 2);
        systems::MovingDomainOptions moving;
        moving.shape_derivative = forms::ShapeDerivative::Complete;
        (void)systems::movingDomain(system, d, moving);
        system.addOperator("probe");
        const auto message = installError([&] {
            systems::FormInstallOptions options;
            options.compiler_options.jit.enable = false;
            (void)systems::installFormulation(system, "probe", {p},
                                              (FormExpr::stateField(p, *scalar, "p") *
                                               FormExpr::testFunction(p, *scalar, "q")).dx(),
                                              options);
        });
        EXPECT_NE(message.find("no richer than the geometry"), std::string::npos) << message;
    }
    {
        // A fitted install of the same system is accepted.
        RefusalSystem s;
        EXPECT_EQ(installError([&] { s.install((s.pState() * s.q()).dx() + (s.pState() * s.q()).ds(kTop)); }), "");
    }
}

namespace {

// The residual-based VMS stabilization of the Navier-Stokes module, written
// out: the element metric K = Jinv^T Jinv, tau_M and tau_C, PSPG, SUPG,
// grad-div and cross-stress terms with the ALE transport velocity u - w.
struct VMSTerms {
    FormExpr K, a, tau_m, tau_c, residual, subscale;
};

VMSTerms vmsTerms(const Symbols& s)
{
    using namespace svmp::FE::forms;
    const auto nu = FormExpr::constant(0.05);
    const auto eps = FormExpr::constant(1e-12);
    const auto dts = FormExpr::timeStep();
    VMSTerms t;
    t.K = transpose(Jinv()) * Jinv();
    t.a = s.u - meshVelocity();
    t.tau_m = FormExpr::constant(1.0) /
              sqrt(FormExpr::constant(4.0) / (dts * dts) + inner(t.a, t.K * t.a) +
                   FormExpr::constant(3.0) * doubleContraction(t.K, t.K) * (nu * nu) + eps);
    t.tau_c = FormExpr::constant(1.0) / (t.tau_m * trace(t.K) + eps);
    t.residual = dt(s.u) + grad(s.u) * t.a - div(FormExpr::constant(2.0) * nu * sym(grad(s.u))) + grad(s.p);
    t.subscale = -t.tau_m * t.residual;
    return t;
}

std::vector<Probe> stabilizationProbes()
{
    using namespace svmp::FE::forms;
    return {
        {"tau_M q dx", Rows::Scalar, [](const Symbols& s) { return (vmsTerms(s).tau_m * s.q).dx(); }},
        {"tau_C q dx", Rows::Scalar, [](const Symbols& s) { return (vmsTerms(s).tau_c * s.q).dx(); }},
        {"tau_M grad(p).grad(q) dx (PSPG, pressure part)", Rows::Scalar,
         [](const Symbols& s) { return inner(grad(s.q), vmsTerms(s).tau_m * grad(s.p)).dx(); }},
        {"tau_M R_M.grad(q) dx (PSPG)", Rows::Scalar,
         [](const Symbols& s) {
             const auto t = vmsTerms(s);
             return (inner(grad(s.q), t.tau_m * t.residual) + zeroTimes(s.p * s.q)).dx();
         },
         true},
        {"(grad(v) a).u' dx (SUPG)", Rows::Vector,
         [](const Symbols& s) {
             const auto t = vmsTerms(s);
             return (-inner(grad(s.v) * t.a, t.subscale)).dx();
         },
         true},
        {"tau_C div(u) div(v) dx (grad-div)", Rows::Vector,
         [](const Symbols& s) { return (vmsTerms(s).tau_c * div(s.u) * div(s.v)).dx(); }},
        {"cross stress", Rows::Vector,
         [](const Symbols& s) {
             const auto t = vmsTerms(s);
             const auto tau_b = FormExpr::constant(1.0) /
                                sqrt(inner(t.subscale, t.K * t.subscale) + FormExpr::constant(1e-12));
             return inner(grad(s.v) * t.subscale, tau_b * (grad(s.u) * t.subscale)).dx();
         },
         true},
    };
}

} // namespace

// Stabilization: the VMS metric K = Jinv^T Jinv and the parameters tau_M and
// tau_C vary with J; their shape derivatives, with the PSPG, SUPG, grad-div
// and cross-stress terms built from them, match directional differences.
TEST(CompleteShapeDerivative, StabilizationMetricAndParametersMatchDirectionalDifferences)
{
    // Every term on non-affine cells, where the metric varies in the cell and
    // P1 fields have second derivatives.
    runProbes(stabilizationProbes(), "stabilization", {{Cell::Quad, 1, 1, 2}});
}

// One definition of field Hessians: every field's Hessian, the row's active
// unknown, the other fields and earlier time levels, carries the mapping
// curvature, as the basis Hessians do.  On a non-affine or curved cell the
// blocks between two fields of a term with second derivatives are then exact:
// the residual reads the other field's Hessian as the block reads its trial
// basis Hessian.  With the default tangent (no complete shape derivative),
// the u and p columns of PSPG, SUPG and cross-stress rows match residual
// differences on a non-simplex P1 cell, a curved Quad9 and a curved Tri6.
TEST(FieldHessianCurvature, BlocksBetweenFieldsOfSecondDerivativeTermsMatchResidualDifferences)
{
    std::vector<Probe> probes;
    for (const auto& probe : stabilizationProbes()) {
        if (probe.second_derivatives) {
            probes.push_back(probe);
        }
    }
    ASSERT_FALSE(probes.empty());
    for (const auto& mesh_case : std::vector<MeshCase>{{Cell::Quad, 1, 1, 2}, {Cell::Quad, 2, 2, 1},
                                                       {Cell::Triangle, 2, 2, 1}}) {
        for (const auto& probe : probes) {
            SCOPED_TRACE(caseName(mesh_case) + ": " + probe.name);
            ProbeSetup setup;
            setup.cell = mesh_case.cell;
            setup.n = mesh_case.n;
            setup.geometry_order = mesh_case.geometry_order;
            setup.u_order = mesh_case.field_order;
            setup.p_order = mesh_case.field_order;
            const auto defect = directionalDefect(setup, probe);
            EXPECT_LT(defect.other, 1e-6);
            std::cout << "[ field Hessians ] " << caseName(mesh_case) << ": " << probe.name
                      << ": u, p columns against residual differences " << defect.other << "\n";
        }
    }
}

TEST(CompleteShapeDerivativeSweep, StabilizationOnEveryCellType)
{
    // Every cell type with P1 fields, and curved P2 cells in 2D (the volume
    // sweep covers P2 fields and curved cells in 3D for the same rules).
    std::vector<MeshCase> cases;
    for (const auto& mesh_case : everyMeshCase()) {
        if (mesh_case.field_order == 1 || cellDimension(mesh_case.cell) == 2) {
            cases.push_back(mesh_case);
        }
    }
    runProbes(stabilizationProbes(), "stabilization", cases);
}

namespace {

// With the JIT on (specialized kernels, as the physics modules use it), the
// mesh blocks of a complete shape derivative are compiled where the JIT
// lowers every term: fused with the other cell blocks of the install, or each
// on its own where the fused kernel does not take a term.  They are
// interpreted per kernel where the JIT does not lower a term (the variations
// of h(), vol() and area(), and nested spatial derivatives of composite
// arguments, as in div(sym(grad(u)))).  Either way J dd equals the
// interpreter's to round-off, and so do the residual rows, second
// derivatives of the solution included; J dd matches residual differences.
void checkJITMatchesTheInterpreter(const std::vector<MeshCase>& mesh_cases)
{
    using namespace svmp::FE::forms;
    const auto n = FormExpr::normal();
    const auto half = FormExpr::constant(0.5);
    struct JITProbe {
        Probe probe;
        bool compiled;            ///< the mesh blocks are compiled
    };
    const std::vector<JITProbe> probes = {
        {{"x().v + u.v/2 dx", Rows::Vector,
          [half](const Symbols& s) { return (inner(x(), s.v) + half * inner(s.u, s.v)).dx(); }},
         true},
        {{"w.v + u.v/2 dx", Rows::Vector,
          [half](const Symbols& s) { return (inner(meshVelocity(), s.v) + half * inner(s.u, s.v)).dx(); }},
         true},
        {{"(tr(Jinv()) + p/2) q dx", Rows::Scalar,
          [half](const Symbols& s) { return ((trace(Jinv()) + half * s.p) * s.q).dx(); }},
         true},
        {{"(detJ() + p/2) q dx", Rows::Scalar, [half](const Symbols& s) { return ((detJ() + half * s.p) * s.q).dx(); }},
         true},
        {{"(grad(u) (u - w)).v dx", Rows::Vector,
          [](const Symbols& s) { return inner(grad(s.u) * (s.u - meshVelocity()), s.v).dx(); }},
         true},
        {{"div(w) u.v dx", Rows::Vector, [](const Symbols& s) { return (div(meshVelocity()) * inner(s.u, s.v)).dx(); }},
         true},
        {{"(div(u) + p/2) q dx", Rows::Scalar, [half](const Symbols& s) { return ((div(s.u) + half * s.p) * s.q).dx(); }},
         true},
        {{"grad(u):grad(v) dx", Rows::Vector, [](const Symbols& s) { return inner(grad(s.u), grad(s.v)).dx(); }},
         true},
        {{"tau_M grad(p).grad(q) dx", Rows::Scalar,
          [](const Symbols& s) { return inner(grad(s.q), vmsTerms(s).tau_m * grad(s.p)).dx(); }},
         false},
        {{"(u.normal() + p/2) q ds", Rows::Scalar,
          [n, half](const Symbols& s) { return ((inner(s.u, n) + half * s.p) * s.q).ds(kTop); }},
         true},
        {{"(1 / hNormal() + p/2) q ds", Rows::Scalar,
          [half](const Symbols& s) { return ((FormExpr::constant(1.0) / hNormal() + half * s.p) * s.q).ds(kTop); }},
         false},
        {{"div(2 sym(grad(u))).v dx", Rows::Vector,
          [](const Symbols& s) { return inner(div(FormExpr::constant(2.0) * sym(grad(s.u))), s.v).dx(); }},
         false},
    };
    const auto expectClose = [](const std::vector<Real>& a, const std::vector<Real>& b, const char* what) {
        ASSERT_EQ(a.size(), b.size()) << what;
        Real scale = 1.0;
        for (const auto x : a) scale = std::max(scale, std::abs(x));
        for (std::size_t i = 0; i < a.size(); ++i) {
            EXPECT_NEAR(a[i], b[i], 1e-12 * scale) << what << " row " << i;
        }
    };
    for (const auto& mesh_case : mesh_cases) {
        for (const auto& [probe, compiled] : probes) {
            SCOPED_TRACE(caseName(mesh_case) + ": " + probe.name);
            ProbeSetup setup;
            setup.cell = mesh_case.cell;
            setup.n = mesh_case.n;
            setup.geometry_order = mesh_case.geometry_order;
            setup.u_order = mesh_case.field_order;
            setup.p_order = mesh_case.field_order;
            setup.complete = true;
            setup.record_residual = true;
            const auto interpreted = directionalDefect(setup, probe);
            setup.jit = true;
            const auto with_jit = directionalDefect(setup, probe);
            if (compiled) {
                EXPECT_TRUE(with_jit.jit_mesh_blocks) << with_jit.jit_message;
            }
            expectClose(interpreted.jdd, with_jit.jdd, "J dd");
            expectClose(interpreted.residual, with_jit.residual, "residual");
            EXPECT_LT(with_jit.mesh, 1e-6) << "|J dd| " << with_jit.jac_norm << ", |FD| " << with_jit.fd_norm;
        }
    }
}

} // namespace

TEST(CompleteShapeDerivative, JITMatchesTheInterpreterOnAMovingDomain)
{
    checkJITMatchesTheInterpreter({{Cell::Quad, 1, 1, 2}, {Cell::Tetra, 1, 1, 1}});
}

TEST(CompleteShapeDerivativeSweep, JITMatchesTheInterpreterOnSeveralCells)
{
    checkJITMatchesTheInterpreter(
        {{Cell::Triangle, 1, 1, 2}, {Cell::Quad, 2, 2, 2}, {Cell::Hex, 1, 1, 1}, {Cell::Tetra, 2, 2, 1}, {Cell::Wedge, 1, 1, 1}});
}
