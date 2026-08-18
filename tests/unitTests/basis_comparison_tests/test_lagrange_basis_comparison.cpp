// Comparison harness: legacy nn:: basis evaluations vs OOP svmp::FE::basis::LagrangeBasis.
//
// Design notes:
//   - The two implementations use *different* node orderings (legacy: vertex L1,
//     L2, L3, L0 then mid-edges; OOP: VTK = L0, L1, L2, L3 then VTK edge order).
//     We discover the legacy <-> OOP node permutation at runtime by evaluating
//     the legacy basis at each OOP node location: the legacy basis function
//     that equals 1 there corresponds to that OOP node.
//   - We sample at OOP node locations + a fixed set of interior test points
//     (so we are independent of any specific quadrature rule and so we can
//     cover element types not present in the mesh-free legacy quadrature map).
//   - Outputs CSVs to $SVMP_BASIS_COMPARE_OUT (default ./basis_comparison_output).
//   - Run via: run_all_unit_tests --gtest_filter='LagrangeBasisComparison.*'

#include <gtest/gtest.h>

#include "nn.h"
#include "Array.h"
#include "Array3.h"
#include "Vector.h"
#include "consts.h"

#include "FE/Basis/LagrangeBasis.h"
#include "FE/Core/Types.h"
#include "Math/Vector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>
#include <vector>

namespace {

struct ElementCase {
    consts::ElementType legacy_type;
    svmp::FE::ElementType oop_type;
    int eNoN;
    int insd;
    int order;
    bool simplex;     // reference is a simplex (xi >= 0, sum(xi) <= 1)
    const char* name;
};

const std::vector<ElementCase> kCases = {
    {consts::ElementType::TRI3,  svmp::FE::ElementType::Triangle3,  3, 2, 1, true,  "TRI3"},
    {consts::ElementType::TRI6,  svmp::FE::ElementType::Triangle6,  6, 2, 2, true,  "TRI6"},
    {consts::ElementType::TET4,  svmp::FE::ElementType::Tetra4,     4, 3, 1, true,  "TET4"},
    {consts::ElementType::TET10, svmp::FE::ElementType::Tetra10,   10, 3, 2, true,  "TET10"},
    {consts::ElementType::HEX8,  svmp::FE::ElementType::Hex8,       8, 3, 1, false, "HEX8"},
    {consts::ElementType::HEX27, svmp::FE::ElementType::Hex27,     27, 3, 2, false, "HEX27"},
};

std::filesystem::path output_dir() {
    if (const char* env = std::getenv("SVMP_BASIS_COMPARE_OUT")) {
        return std::filesystem::path{env};
    }
    return std::filesystem::path{"basis_comparison_output"};
}

// Pack a single xi point into a legacy Array<double>(insd, 1).
Array<double> pack_xi(int insd, const std::array<double, 3>& xi) {
    Array<double> out(insd, 1);
    for (int d = 0; d < insd; ++d) out(d, 0) = xi[d];
    return out;
}

// Discover the legacy->OOP node permutation: for each OOP node k, evaluate the
// legacy basis at that node's reference coord and find which legacy index has
// value 1. perm_legacy_to_oop[j] = k means legacy node j corresponds to OOP k.
std::vector<int> discover_permutation(
    const ElementCase& ec,
    const std::vector<std::array<double, 3>>& oop_node_coords)
{
    std::vector<int> perm(ec.eNoN, -1);
    Array<double> N(ec.eNoN, 1);
    Array3<double> Nx(ec.insd, ec.eNoN, 1);

    for (int k = 0; k < ec.eNoN; ++k) {
        Array<double> xi_k = pack_xi(ec.insd, oop_node_coords[k]);
        nn::get_gnn(ec.insd, ec.legacy_type, ec.eNoN, 0, xi_k, N, Nx);

        int matched = -1;
        for (int j = 0; j < ec.eNoN; ++j) {
            if (std::abs(N(j, 0) - 1.0) < 1e-9) {
                matched = j;
                break;
            }
        }
        if (matched < 0) {
            // Fallback: pick the largest-magnitude entry (handles edge cases)
            double best = -1.0;
            for (int j = 0; j < ec.eNoN; ++j) {
                double v = std::abs(N(j, 0));
                if (v > best) { best = v; matched = j; }
            }
        }
        perm[matched] = k;
    }
    return perm;
}

// Generate a fixed set of interior test points for an element type.
// Combines OOP nodes with random interior samples for a moderate but
// reproducible coverage of the reference domain.
std::vector<std::array<double, 3>> generate_test_points(
    const ElementCase& ec,
    const std::vector<std::array<double, 3>>& oop_nodes)
{
    std::vector<std::array<double, 3>> points = oop_nodes;

    std::mt19937 rng(12345 + ec.eNoN);
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    const int extra = 40;
    int added = 0;
    int attempts = 0;
    while (added < extra && attempts < 10000) {
        ++attempts;
        std::array<double, 3> p{0.0, 0.0, 0.0};
        if (ec.simplex) {
            // Sample uniformly inside the unit simplex via barycentric squeeze
            std::array<double, 4> u{0.0, 0.0, 0.0, 0.0};
            for (int d = 0; d <= ec.insd; ++d) u[d] = -std::log(u01(rng) + 1e-30);
            double s = 0.0;
            for (int d = 0; d <= ec.insd; ++d) s += u[d];
            for (int d = 0; d < ec.insd; ++d) p[d] = u[d] / s;
        } else {
            for (int d = 0; d < ec.insd; ++d) p[d] = -1.0 + 2.0 * u01(rng);
        }
        // Skip if too close to apex of pyramid etc.; not relevant for current cases.
        points.push_back(p);
        ++added;
    }
    return points;
}

class LagrangeBasisComparison : public ::testing::Test {
protected:
    void SetUp() override {
        std::filesystem::create_directories(output_dir());
    }
};

TEST_F(LagrangeBasisComparison, ComputeAndDumpAllElements) {
    const auto out = output_dir();

    auto open_csv = [&](const std::string& name, const std::string& header) {
        std::ofstream f(out / name);
        f << std::setprecision(17);
        f << header;
        return f;
    };

    std::ofstream values_csv = open_csv(
        "basis_values.csv",
        "elem_type,sample_idx,oop_dof_index,legacy_dof_index,xi_x,xi_y,xi_z,"
        "N_legacy,N_oop,abs_err\n");

    std::ofstream gradients_csv = open_csv(
        "basis_gradients.csv",
        "elem_type,sample_idx,oop_dof_index,legacy_dof_index,dim,"
        "dN_legacy,dN_oop,abs_err\n");

    std::ofstream pou_csv = open_csv(
        "partition_of_unity.csv",
        "elem_type,sample_idx,xi_x,xi_y,xi_z,sum_N_legacy,sum_N_oop,"
        "grad_sum_norm_legacy,grad_sum_norm_oop\n");

    std::ofstream summary_csv = open_csv(
        "summary.csv",
        "elem_type,eNoN,n_samples,max_abs_err_value,max_abs_err_grad,"
        "max_abs_pou_dev_legacy,max_abs_pou_dev_oop,permutation\n");

    std::ofstream nodes_oop_csv = open_csv(
        "node_locations_oop.csv",
        "elem_type,oop_dof_index,xi_x,xi_y,xi_z\n");

    std::ofstream perm_csv = open_csv(
        "node_permutation.csv",
        "elem_type,legacy_dof_index,oop_dof_index\n");

    for (const auto& ec : kCases) {
        // Build the OOP basis. For aliases like Tetra10, the OOP type may
        // normalize internally to canonical (Tetra4 + order 2); the node coord
        // list still reports the canonical-order nodes.
        svmp::FE::basis::LagrangeBasis oop_basis(ec.oop_type, ec.order);
        ASSERT_EQ(oop_basis.size(), static_cast<std::size_t>(ec.eNoN))
            << "OOP basis size mismatch for " << ec.name;

        std::vector<std::array<double, 3>> oop_nodes;
        oop_nodes.reserve(ec.eNoN);
        for (const auto& n : oop_basis.nodes()) {
            oop_nodes.push_back({n[0], n[1], n[2]});
        }

        for (int k = 0; k < ec.eNoN; ++k) {
            nodes_oop_csv << ec.name << "," << k << ","
                          << oop_nodes[k][0] << ","
                          << oop_nodes[k][1] << ","
                          << oop_nodes[k][2] << "\n";
        }

        // Discover legacy <-> OOP node permutation.
        std::vector<int> perm_legacy_to_oop = discover_permutation(ec, oop_nodes);

        // Compute inverse permutation (oop -> legacy) for symmetric lookup.
        std::vector<int> perm_oop_to_legacy(ec.eNoN, -1);
        std::string perm_str;
        for (int j = 0; j < ec.eNoN; ++j) {
            int k = perm_legacy_to_oop[j];
            if (k >= 0 && k < ec.eNoN) perm_oop_to_legacy[k] = j;
            perm_csv << ec.name << "," << j << "," << k << "\n";
            if (j > 0) perm_str += " ";
            perm_str += std::to_string(k);
        }

        // Generate test points (OOP nodes + interior samples).
        const auto test_points = generate_test_points(ec, oop_nodes);

        double max_err_value = 0.0;
        double max_err_grad = 0.0;
        double max_pou_dev_legacy = 0.0;
        double max_pou_dev_oop = 0.0;

        std::vector<svmp::FE::Real> N_oop;
        std::vector<svmp::FE::basis::Gradient> grad_oop;
        Array<double> N_leg(ec.eNoN, 1);
        Array3<double> Nx_leg(ec.insd, ec.eNoN, 1);

        for (std::size_t s = 0; s < test_points.size(); ++s) {
            const auto& pt = test_points[s];

            // Legacy at this point
            Array<double> xi_pt = pack_xi(ec.insd, pt);
            nn::get_gnn(ec.insd, ec.legacy_type, ec.eNoN, 0, xi_pt, N_leg, Nx_leg);

            // OOP at this point
            svmp::FE::math::Vector<svmp::FE::Real, 3> xi_oop{pt[0], pt[1], pt[2]};
            oop_basis.evaluate_values(xi_oop, N_oop);
            oop_basis.evaluate_gradients(xi_oop, grad_oop);

            ASSERT_EQ(N_oop.size(), static_cast<std::size_t>(ec.eNoN));

            double sum_N_leg = 0.0;
            double sum_N_oop = 0.0;
            std::array<double, 3> grad_sum_leg{0.0, 0.0, 0.0};
            std::array<double, 3> grad_sum_oop{0.0, 0.0, 0.0};

            for (int j = 0; j < ec.eNoN; ++j) {
                int k = perm_legacy_to_oop[j];  // OOP index this legacy node corresponds to
                const double Nl = N_leg(j, 0);
                const double No = (k >= 0 && k < ec.eNoN)
                                      ? static_cast<double>(N_oop[k])
                                      : std::numeric_limits<double>::quiet_NaN();
                const double err = std::isnan(No) ? 0.0 : std::abs(Nl - No);
                if (!std::isnan(No)) max_err_value = std::max(max_err_value, err);

                sum_N_leg += Nl;
                sum_N_oop += std::isnan(No) ? 0.0 : No;

                values_csv << ec.name << "," << s << "," << k << "," << j << ","
                           << pt[0] << "," << pt[1] << "," << pt[2] << ","
                           << Nl << "," << (std::isnan(No) ? 0.0 : No) << "," << err << "\n";

                for (int d = 0; d < ec.insd; ++d) {
                    const double dNl = Nx_leg(d, j, 0);
                    const double dNo = (k >= 0 && k < ec.eNoN)
                                          ? static_cast<double>(grad_oop[k][d])
                                          : std::numeric_limits<double>::quiet_NaN();
                    const double derr = std::isnan(dNo) ? 0.0 : std::abs(dNl - dNo);
                    if (!std::isnan(dNo)) max_err_grad = std::max(max_err_grad, derr);

                    grad_sum_leg[d] += dNl;
                    grad_sum_oop[d] += std::isnan(dNo) ? 0.0 : dNo;

                    gradients_csv << ec.name << "," << s << "," << k << "," << j << ","
                                  << d << "," << dNl << ","
                                  << (std::isnan(dNo) ? 0.0 : dNo) << ","
                                  << derr << "\n";
                }
            }

            double gnorm_leg = 0.0;
            double gnorm_oop = 0.0;
            for (int d = 0; d < ec.insd; ++d) {
                gnorm_leg += grad_sum_leg[d] * grad_sum_leg[d];
                gnorm_oop += grad_sum_oop[d] * grad_sum_oop[d];
            }
            gnorm_leg = std::sqrt(gnorm_leg);
            gnorm_oop = std::sqrt(gnorm_oop);

            max_pou_dev_legacy = std::max(max_pou_dev_legacy, std::abs(sum_N_leg - 1.0));
            max_pou_dev_oop    = std::max(max_pou_dev_oop,    std::abs(sum_N_oop - 1.0));

            pou_csv << ec.name << "," << s << ","
                    << pt[0] << "," << pt[1] << "," << pt[2] << ","
                    << sum_N_leg << "," << sum_N_oop << ","
                    << gnorm_leg << "," << gnorm_oop << "\n";
        }

        summary_csv << ec.name << "," << ec.eNoN << "," << test_points.size() << ","
                    << max_err_value << "," << max_err_grad << ","
                    << max_pou_dev_legacy << "," << max_pou_dev_oop << ","
                    << "\"" << perm_str << "\"\n";

        std::cout << "[" << ec.name << "] eNoN=" << ec.eNoN
                  << " n_samples=" << test_points.size()
                  << " perm=[" << perm_str << "]"
                  << " max|N-N|=" << max_err_value
                  << " max|dN-dN|=" << max_err_grad
                  << " max|sumN-1|_legacy=" << max_pou_dev_legacy
                  << " max|sumN-1|_oop=" << max_pou_dev_oop
                  << std::endl;
    }
}

// -----------------------------------------------------------------------------
// Dense reference-element grid for contour plots.
// 2D: uniform grid in (xi, eta), masked to inside the simplex/quad.
// 3D: 2D slice at zeta=z_slice (one slice per element).
// -----------------------------------------------------------------------------
TEST_F(LagrangeBasisComparison, DumpReferenceGrid) {
    const auto out = output_dir();
    std::ofstream grid_csv(out / "contour_grid.csv");
    grid_csv << std::setprecision(17);
    grid_csv << "elem_type,oop_dof_index,legacy_dof_index,xi_x,xi_y,xi_z,"
                "N_legacy,N_oop,"
                "dN_legacy_x,dN_legacy_y,dN_legacy_z,"
                "dN_oop_x,dN_oop_y,dN_oop_z\n";

    for (const auto& ec : kCases) {
        svmp::FE::basis::LagrangeBasis oop_basis(ec.oop_type, ec.order);
        std::vector<std::array<double, 3>> oop_nodes;
        for (const auto& n : oop_basis.nodes()) oop_nodes.push_back({n[0], n[1], n[2]});
        const auto perm_legacy_to_oop = discover_permutation(ec, oop_nodes);

        // Build the dense xi grid for this element.
        std::vector<std::array<double, 3>> grid;
        const int Ngrid = 60;
        const double z_slice = ec.simplex ? 0.0 : 0.0;
        if (ec.insd == 2 && ec.simplex) {
            for (int i = 0; i <= Ngrid; ++i) {
                for (int j = 0; j <= Ngrid; ++j) {
                    double x = double(i) / Ngrid;
                    double y = double(j) / Ngrid;
                    if (x + y > 1.0 + 1e-12) continue;
                    grid.push_back({x, y, 0.0});
                }
            }
        } else if (ec.insd == 2 && !ec.simplex) {
            for (int i = 0; i <= Ngrid; ++i) {
                for (int j = 0; j <= Ngrid; ++j) {
                    grid.push_back({-1.0 + 2.0 * i / Ngrid, -1.0 + 2.0 * j / Ngrid, 0.0});
                }
            }
        } else if (ec.insd == 3 && ec.simplex) {
            for (int i = 0; i <= Ngrid; ++i) {
                for (int j = 0; j <= Ngrid; ++j) {
                    double x = double(i) / Ngrid;
                    double y = double(j) / Ngrid;
                    if (x + y + z_slice > 1.0 + 1e-12) continue;
                    grid.push_back({x, y, z_slice});
                }
            }
        } else if (ec.insd == 3 && !ec.simplex) {
            for (int i = 0; i <= Ngrid; ++i) {
                for (int j = 0; j <= Ngrid; ++j) {
                    grid.push_back({-1.0 + 2.0 * i / Ngrid,
                                    -1.0 + 2.0 * j / Ngrid, z_slice});
                }
            }
        }

        std::vector<svmp::FE::Real> N_oop;
        std::vector<svmp::FE::basis::Gradient> grad_oop;
        Array<double> N_leg(ec.eNoN, 1);
        Array3<double> Nx_leg(ec.insd, ec.eNoN, 1);

        for (const auto& pt : grid) {
            Array<double> xi_pt = pack_xi(ec.insd, pt);
            nn::get_gnn(ec.insd, ec.legacy_type, ec.eNoN, 0, xi_pt, N_leg, Nx_leg);

            svmp::FE::math::Vector<svmp::FE::Real, 3> xi_oop{pt[0], pt[1], pt[2]};
            oop_basis.evaluate_values(xi_oop, N_oop);
            oop_basis.evaluate_gradients(xi_oop, grad_oop);

            for (int j = 0; j < ec.eNoN; ++j) {
                int k = perm_legacy_to_oop[j];
                double dNl_x = ec.insd >= 1 ? Nx_leg(0, j, 0) : 0.0;
                double dNl_y = ec.insd >= 2 ? Nx_leg(1, j, 0) : 0.0;
                double dNl_z = ec.insd >= 3 ? Nx_leg(2, j, 0) : 0.0;
                double dNo_x = (k >= 0) ? static_cast<double>(grad_oop[k][0]) : 0.0;
                double dNo_y = (k >= 0) ? static_cast<double>(grad_oop[k][1]) : 0.0;
                double dNo_z = (k >= 0) ? static_cast<double>(grad_oop[k][2]) : 0.0;
                grid_csv << ec.name << "," << k << "," << j << ","
                         << pt[0] << "," << pt[1] << "," << pt[2] << ","
                         << N_leg(j, 0) << ","
                         << (k >= 0 ? static_cast<double>(N_oop[k])
                                    : std::numeric_limits<double>::quiet_NaN())
                         << "," << dNl_x << "," << dNl_y << "," << dNl_z
                         << "," << dNo_x << "," << dNo_y << "," << dNo_z
                         << "\n";
            }
        }
        std::cout << "[" << ec.name << "] grid points: " << grid.size() << std::endl;
    }
}

// -----------------------------------------------------------------------------
// Polynomial reproduction:
//   For each monomial m(xi) = xi_x^a * xi_y^b * xi_z^c with total degree
//   <= max_total_degree (and per-axis degree <= max_axis_degree for tensor
//   elements), build nodal coefficients c_oop[k] = m(oop_nodes[k]) and
//   c_leg[j] = m(legacy_nodes[j]). At interior sample points evaluate
//     f_oop  = Sum_k c_oop[k] * N_oop[k](xi)
//     f_leg  = Sum_j c_leg[j] * N_leg[j](xi)
//   and report the L_inf reconstruction error vs the true monomial value.
//
//   Bases are exact for monomials within their reproduction set:
//     simplex order p   : reproduces total degree <= p exactly
//     tensor  order p   : reproduces every axis degree <= p exactly
//   Monomials beyond the reproduction set are expected to disagree from the
//   true value (the basis is incapable, not buggy) - that "wall" is the figure.
// -----------------------------------------------------------------------------
TEST_F(LagrangeBasisComparison, DumpPolynomialReproduction) {
    const auto out = output_dir();
    std::ofstream poly_csv(out / "polynomial_reproduction.csv");
    poly_csv << std::setprecision(17);
    poly_csv << "elem_type,monomial_label,a,b,c,total_degree,max_axis_degree,"
                "in_reproduction_set,sample_idx,xi_x,xi_y,xi_z,"
                "f_true,f_legacy_recon,f_oop_recon,err_legacy,err_oop\n";

    const int max_degree = 4;  // include one degree above HEX27 axis-2 monomials

    for (const auto& ec : kCases) {
        svmp::FE::basis::LagrangeBasis oop_basis(ec.oop_type, ec.order);
        std::vector<std::array<double, 3>> oop_nodes;
        for (const auto& n : oop_basis.nodes()) oop_nodes.push_back({n[0], n[1], n[2]});
        const auto perm_legacy_to_oop = discover_permutation(ec, oop_nodes);

        // Legacy node coords: legacy node j sits at the OOP node coord that legacy
        // maps to via the discovered permutation.
        std::vector<std::array<double, 3>> legacy_nodes(ec.eNoN, {0, 0, 0});
        for (int j = 0; j < ec.eNoN; ++j) {
            int k = perm_legacy_to_oop[j];
            if (k >= 0) legacy_nodes[j] = oop_nodes[k];
        }

        const auto test_points = generate_test_points(ec, oop_nodes);

        std::vector<svmp::FE::Real> N_oop;
        Array<double> N_leg(ec.eNoN, 1);
        Array3<double> Nx_leg(ec.insd, ec.eNoN, 1);

        // Enumerate (a, b, c) monomials up to max_degree.
        for (int a = 0; a <= max_degree; ++a) {
            for (int b = 0; b <= (ec.insd >= 2 ? max_degree : 0); ++b) {
                for (int c = 0; c <= (ec.insd >= 3 ? max_degree : 0); ++c) {
                    if (ec.insd < 2 && b > 0) continue;
                    if (ec.insd < 3 && c > 0) continue;
                    int total = a + b + c;
                    int max_axis = std::max({a, b, c});
                    if (total > max_degree) continue;

                    bool in_set = ec.simplex ? (total <= ec.order) : (max_axis <= ec.order);

                    auto monomial = [&](const std::array<double, 3>& xi) {
                        return std::pow(xi[0], a) * std::pow(xi[1], b) * std::pow(xi[2], c);
                    };

                    // Build coefficients (nodal values of the monomial)
                    std::vector<double> c_oop(ec.eNoN, 0.0);
                    std::vector<double> c_leg(ec.eNoN, 0.0);
                    for (int k = 0; k < ec.eNoN; ++k) c_oop[k] = monomial(oop_nodes[k]);
                    for (int j = 0; j < ec.eNoN; ++j) c_leg[j] = monomial(legacy_nodes[j]);

                    std::string label = "x^" + std::to_string(a) + "y^" +
                                        std::to_string(b) + "z^" + std::to_string(c);

                    // Evaluate reconstruction at each interior sample point.
                    // (Skip sample 0..eNoN-1 which are the OOP nodes themselves;
                    // reconstruction is trivially exact at the nodes.)
                    for (std::size_t s = static_cast<std::size_t>(ec.eNoN);
                         s < test_points.size(); ++s) {
                        const auto& pt = test_points[s];
                        double f_true = monomial(pt);

                        Array<double> xi_pt = pack_xi(ec.insd, pt);
                        nn::get_gnn(ec.insd, ec.legacy_type, ec.eNoN, 0, xi_pt, N_leg, Nx_leg);

                        svmp::FE::math::Vector<svmp::FE::Real, 3> xi_oop{pt[0], pt[1], pt[2]};
                        oop_basis.evaluate_values(xi_oop, N_oop);

                        double f_leg_recon = 0.0;
                        for (int j = 0; j < ec.eNoN; ++j) f_leg_recon += c_leg[j] * N_leg(j, 0);
                        double f_oop_recon = 0.0;
                        for (int k = 0; k < ec.eNoN; ++k) f_oop_recon += c_oop[k] * N_oop[k];

                        poly_csv << ec.name << "," << label << ","
                                 << a << "," << b << "," << c << ","
                                 << total << "," << max_axis << ","
                                 << (in_set ? 1 : 0) << ","
                                 << s << ","
                                 << pt[0] << "," << pt[1] << "," << pt[2] << ","
                                 << f_true << ","
                                 << f_leg_recon << ","
                                 << f_oop_recon << ","
                                 << std::abs(f_leg_recon - f_true) << ","
                                 << std::abs(f_oop_recon - f_true) << "\n";
                    }
                }
            }
        }
        std::cout << "[" << ec.name << "] polynomial reproduction dumped" << std::endl;
    }
}

// -----------------------------------------------------------------------------
// Lagrange interpolation accuracy on smooth analytic test functions.
//
// For each (element, function), we form nodal coefficients c_i = f(node_i) for
// each implementation, evaluate the interpolant at random interior points, and
// dump the truth + reconstruction + reconstruction error per implementation.
//
// The legacy/OOP reconstructions should agree to machine precision (the
// pointwise basis values agree to at most ~1 ULP and the same coefficients
// are used). The error vs the *true* analytic function is non-trivial -- it
// reveals interpolation skill on transcendental targets.
// -----------------------------------------------------------------------------
TEST_F(LagrangeBasisComparison, DumpInterpolationError) {
    const auto out = output_dir();
    std::ofstream f_csv(out / "interpolation_error.csv");
    f_csv << std::setprecision(17);
    f_csv << "elem_type,function_name,sample_idx,xi_x,xi_y,xi_z,"
             "f_true,f_legacy_recon,f_oop_recon,err_legacy,err_oop\n";

    struct TestFn {
        const char* name;
        std::function<double(double, double, double)> f;
    };
    const std::vector<TestFn> kFns = {
        {"sin_cos_exp",
         [](double x, double y, double z) {
             return std::sin(M_PI * x) * std::cos(M_PI * y) * std::exp(0.5 * z);
         }},
        {"gauss_bump",
         [](double x, double y, double z) {
             return std::exp(-5.0 * (x * x + y * y + z * z));
         }},
        {"tanh_step",
         [](double x, double y, double z) {
             return std::tanh(5.0 * (x - 0.3));
         }},
        {"oscillatory",
         [](double x, double y, double z) {
             return std::cos(3.0 * M_PI * (x * y + 0.25 * z));
         }},
    };

    for (const auto& ec : kCases) {
        svmp::FE::basis::LagrangeBasis oop_basis(ec.oop_type, ec.order);
        std::vector<std::array<double, 3>> oop_nodes;
        for (const auto& n : oop_basis.nodes()) oop_nodes.push_back({n[0], n[1], n[2]});
        const auto perm_legacy_to_oop = discover_permutation(ec, oop_nodes);

        std::vector<std::array<double, 3>> legacy_nodes(ec.eNoN, {0, 0, 0});
        for (int j = 0; j < ec.eNoN; ++j) {
            int k = perm_legacy_to_oop[j];
            if (k >= 0) legacy_nodes[j] = oop_nodes[k];
        }

        const auto test_points = generate_test_points(ec, oop_nodes);

        std::vector<svmp::FE::Real> N_oop;
        Array<double> N_leg(ec.eNoN, 1);
        Array3<double> Nx_leg(ec.insd, ec.eNoN, 1);

        for (const auto& fn : kFns) {
            // Build nodal coefficients
            std::vector<double> c_oop(ec.eNoN, 0.0);
            std::vector<double> c_leg(ec.eNoN, 0.0);
            for (int k = 0; k < ec.eNoN; ++k) {
                c_oop[k] = fn.f(oop_nodes[k][0], oop_nodes[k][1], oop_nodes[k][2]);
            }
            for (int j = 0; j < ec.eNoN; ++j) {
                c_leg[j] = fn.f(legacy_nodes[j][0], legacy_nodes[j][1], legacy_nodes[j][2]);
            }

            // Evaluate interpolant at non-node samples (skip first eNoN where
            // reconstruction is exact by construction).
            for (std::size_t s = static_cast<std::size_t>(ec.eNoN);
                 s < test_points.size(); ++s) {
                const auto& pt = test_points[s];
                double f_true = fn.f(pt[0], pt[1], pt[2]);

                Array<double> xi_pt = pack_xi(ec.insd, pt);
                nn::get_gnn(ec.insd, ec.legacy_type, ec.eNoN, 0, xi_pt, N_leg, Nx_leg);

                svmp::FE::math::Vector<svmp::FE::Real, 3> xi_oop{pt[0], pt[1], pt[2]};
                oop_basis.evaluate_values(xi_oop, N_oop);

                double f_leg_recon = 0.0;
                for (int j = 0; j < ec.eNoN; ++j) f_leg_recon += c_leg[j] * N_leg(j, 0);
                double f_oop_recon = 0.0;
                for (int k = 0; k < ec.eNoN; ++k)
                    f_oop_recon += c_oop[k] * static_cast<double>(N_oop[k]);

                f_csv << ec.name << "," << fn.name << "," << s << ","
                      << pt[0] << "," << pt[1] << "," << pt[2] << ","
                      << f_true << "," << f_leg_recon << "," << f_oop_recon << ","
                      << std::abs(f_leg_recon - f_true) << ","
                      << std::abs(f_oop_recon - f_true) << "\n";
            }
        }
        std::cout << "[" << ec.name << "] interpolation error dumped" << std::endl;
    }
}

// -----------------------------------------------------------------------------
// Element mass matrix M_ij = sum_q w_q * N_i(xi_q) * N_j(xi_q).
//
// For each element type we use a quadrature rule that exactly integrates
// degree 2p (the order of N_i N_j). For elements present in the legacy
// mesh-free quadrature map (TRI3, TET4, HEX8, HEX27) we reuse those rules.
// For TRI6 and TET10 we hardcode the mesh-bound quadrature literals from
// nn_elem_gip.h:520 (TET10) and nn_elem_gip.h:581 (TRI6) verbatim.
// -----------------------------------------------------------------------------
namespace mass_quad {

struct Rule {
    std::vector<double> w;
    std::vector<std::array<double, 3>> pts;  // (xi, eta, zeta) - unused dims = 0
};

// TRI6 7-point Gauss rule (matches nn_elem_gip.h:581).
inline Rule tri6_rule() {
    Rule r;
    r.w = {
        0.225000000000000 * 0.5,
        0.125939180544827 * 0.5,
        0.125939180544827 * 0.5,
        0.125939180544827 * 0.5,
        0.132394152788506 * 0.5,
        0.132394152788506 * 0.5,
        0.132394152788506 * 0.5,
    };
    r.pts.resize(7);
    {
        double s = 0.333333333333333;
        r.pts[0] = {s, s, 0.0};
    }
    {
        double s = 0.797426985353087;
        double t = 0.101286507323456;
        r.pts[1] = {s, t, 0.0};
        r.pts[2] = {t, s, 0.0};
        r.pts[3] = {t, t, 0.0};
    }
    {
        double s = 0.059715871789770;
        double t = 0.470142064105115;
        r.pts[4] = {s, t, 0.0};
        r.pts[5] = {t, s, 0.0};
        r.pts[6] = {t, t, 0.0};
    }
    return r;
}

// TET10 15-point rule (matches nn_elem_gip.h:520).
inline Rule tet10_rule() {
    Rule r;
    r.w.resize(15);
    r.pts.resize(15);

    r.w[0] = 0.0302836780970890;
    for (int i = 1; i <= 4; ++i)  r.w[i] = 0.0060267857142860;
    for (int i = 5; i <= 8; ++i)  r.w[i] = 0.0116452490860290;
    for (int i = 9; i <= 14; ++i) r.w[i] = 0.0109491415613860;

    {
        double s = 0.250;
        r.pts[0] = {s, s, s};
    }
    {
        double s = 0.3333333333333330;
        double t = 0.0;
        r.pts[1] = {t, s, s};
        r.pts[2] = {s, t, s};
        r.pts[3] = {s, s, t};
        r.pts[4] = {s, s, s};
    }
    {
        double s = 0.0909090909090910;
        double t = 0.7272727272727270;
        r.pts[5] = {t, s, s};
        r.pts[6] = {s, t, s};
        r.pts[7] = {s, s, t};
        r.pts[8] = {s, s, s};
    }
    {
        double s = 0.0665501535736640;
        double t = 0.4334498464263360;
        r.pts[9]  = {s, s, t};
        r.pts[10] = {s, t, s};
        r.pts[11] = {s, t, t};
        r.pts[12] = {t, t, s};
        r.pts[13] = {t, s, t};
        r.pts[14] = {t, s, s};
    }
    return r;
}

// Build a Rule from the legacy mesh-free quadrature for the given (eType, nG).
inline Rule from_legacy(consts::ElementType etype, int insd, int nG) {
    Rule r;
    r.w.assign(nG, 0.0);
    Vector<double> w_v(nG);
    Array<double> xi_a(insd, nG);
    nn::get_gip(insd, etype, nG, w_v, xi_a);
    for (int g = 0; g < nG; ++g) {
        r.w[g] = w_v(g);
        std::array<double, 3> p{0, 0, 0};
        for (int d = 0; d < insd; ++d) p[d] = xi_a(d, g);
        r.pts.push_back(p);
    }
    return r;
}

}  // namespace mass_quad

TEST_F(LagrangeBasisComparison, DumpMassMatrix) {
    const auto out = output_dir();
    std::ofstream m_csv(out / "mass_matrix.csv");
    m_csv << std::setprecision(17);
    m_csv << "elem_type,i,j,M_legacy,M_oop,abs_diff\n";

    for (const auto& ec : kCases) {
        // Choose quadrature
        mass_quad::Rule rule;
        if (std::string(ec.name) == "TRI6") {
            rule = mass_quad::tri6_rule();
        } else if (std::string(ec.name) == "TET10") {
            rule = mass_quad::tet10_rule();
        } else {
            // Pick nG to integrate degree 2p exactly for the simpler cases.
            int nG = 0;
            if (ec.legacy_type == consts::ElementType::TRI3)  nG = 3;
            if (ec.legacy_type == consts::ElementType::TET4)  nG = 4;
            if (ec.legacy_type == consts::ElementType::HEX8)  nG = 8;
            if (ec.legacy_type == consts::ElementType::HEX27) nG = 27;
            rule = mass_quad::from_legacy(ec.legacy_type, ec.insd, nG);
        }

        svmp::FE::basis::LagrangeBasis oop_basis(ec.oop_type, ec.order);
        std::vector<std::array<double, 3>> oop_nodes;
        for (const auto& n : oop_basis.nodes()) oop_nodes.push_back({n[0], n[1], n[2]});
        const auto perm_legacy_to_oop = discover_permutation(ec, oop_nodes);

        const int n = ec.eNoN;
        std::vector<double> M_leg(n * n, 0.0);
        std::vector<double> M_oop(n * n, 0.0);

        std::vector<svmp::FE::Real> N_oop;
        Array<double> N_leg(n, 1);
        Array3<double> Nx_leg(ec.insd, n, 1);

        for (std::size_t q = 0; q < rule.pts.size(); ++q) {
            double w = rule.w[q];
            const auto& xi = rule.pts[q];
            Array<double> xi_pt = pack_xi(ec.insd, xi);
            nn::get_gnn(ec.insd, ec.legacy_type, n, 0, xi_pt, N_leg, Nx_leg);
            svmp::FE::math::Vector<svmp::FE::Real, 3> xi_oop{xi[0], xi[1], xi[2]};
            oop_basis.evaluate_values(xi_oop, N_oop);

            // Map OOP -> legacy index space so M_leg and M_oop are indexed
            // identically (legacy DOF order). For each legacy index a, the
            // corresponding OOP index is perm_legacy_to_oop[a].
            for (int a = 0; a < n; ++a) {
                double Nl_a = N_leg(a, 0);
                int oa = perm_legacy_to_oop[a];
                double No_a = (oa >= 0) ? static_cast<double>(N_oop[oa]) : 0.0;
                for (int b = 0; b < n; ++b) {
                    double Nl_b = N_leg(b, 0);
                    int ob = perm_legacy_to_oop[b];
                    double No_b = (ob >= 0) ? static_cast<double>(N_oop[ob]) : 0.0;
                    M_leg[a * n + b] += w * Nl_a * Nl_b;
                    M_oop[a * n + b] += w * No_a * No_b;
                }
            }
        }

        double max_abs = 0.0, max_rel = 0.0, sum_M = 0.0;
        for (int a = 0; a < n; ++a) {
            for (int b = 0; b < n; ++b) {
                double Mleg = M_leg[a * n + b];
                double Moop = M_oop[a * n + b];
                double diff = std::abs(Mleg - Moop);
                m_csv << ec.name << "," << a << "," << b << ","
                      << Mleg << "," << Moop << "," << diff << "\n";
                max_abs = std::max(max_abs, diff);
                if (std::abs(Mleg) > 1e-30) {
                    max_rel = std::max(max_rel, diff / std::abs(Mleg));
                }
                sum_M += std::abs(Mleg);
            }
        }
        std::cout << "[" << ec.name << "] mass matrix"
                  << " max|Mleg-Moop|=" << max_abs
                  << " max_rel=" << max_rel
                  << " sum|M|=" << sum_M
                  << std::endl;
    }
}

// -----------------------------------------------------------------------------
// 1D cross-sections through the reference element.
// For each element we define a small set of paths (edges + diagonals) and
// sample 50 points along each. Useful to inspect agreement at fine numerical
// detail when 2D contour overlays are too coarse.
// -----------------------------------------------------------------------------
namespace cross_sections {

struct Path {
    std::string name;
    std::array<double, 3> a;
    std::array<double, 3> b;
};

inline std::vector<Path> paths_for(const ElementCase& ec) {
    if (ec.simplex && ec.insd == 2) {
        return {
            {"edge_0_1", {0, 0, 0}, {1, 0, 0}},
            {"edge_1_2", {1, 0, 0}, {0, 1, 0}},
            {"edge_2_0", {0, 1, 0}, {0, 0, 0}},
            {"centroid_to_corner_0", {0, 0, 0}, {1.0/3, 1.0/3, 0}},
        };
    } else if (ec.simplex && ec.insd == 3) {
        return {
            {"edge_0_1", {0, 0, 0}, {1, 0, 0}},
            {"edge_0_2", {0, 0, 0}, {0, 1, 0}},
            {"edge_0_3", {0, 0, 0}, {0, 0, 1}},
            {"edge_1_2", {1, 0, 0}, {0, 1, 0}},
            {"centroid_diag", {0.25, 0.25, 0.25}, {1, 0, 0}},
        };
    } else if (!ec.simplex && ec.insd == 2) {
        return {
            {"bottom", {-1, -1, 0}, {1, -1, 0}},
            {"right",  {1, -1, 0},  {1, 1, 0}},
            {"top",    {1, 1, 0},   {-1, 1, 0}},
            {"diag_main", {-1, -1, 0}, {1, 1, 0}},
        };
    } else {
        return {
            {"x_axis_through_center", {-1, 0, 0}, {1, 0, 0}},
            {"y_axis_through_center", {0, -1, 0}, {0, 1, 0}},
            {"z_axis_through_center", {0, 0, -1}, {0, 0, 1}},
            {"face_diag_z_minus_1",   {-1, -1, -1}, {1, 1, -1}},
            {"body_diag",             {-1, -1, -1}, {1, 1, 1}},
        };
    }
}

}  // namespace cross_sections

TEST_F(LagrangeBasisComparison, DumpCrossSections) {
    const auto out = output_dir();
    std::ofstream cs_csv(out / "cross_sections.csv");
    cs_csv << std::setprecision(17);
    cs_csv << "elem_type,path_name,t,xi_x,xi_y,xi_z,oop_dof_index,legacy_dof_index,N_legacy,N_oop\n";

    const int n_pts = 50;

    for (const auto& ec : kCases) {
        svmp::FE::basis::LagrangeBasis oop_basis(ec.oop_type, ec.order);
        std::vector<std::array<double, 3>> oop_nodes;
        for (const auto& n : oop_basis.nodes()) oop_nodes.push_back({n[0], n[1], n[2]});
        const auto perm_legacy_to_oop = discover_permutation(ec, oop_nodes);

        const auto paths = cross_sections::paths_for(ec);

        std::vector<svmp::FE::Real> N_oop;
        Array<double> N_leg(ec.eNoN, 1);
        Array3<double> Nx_leg(ec.insd, ec.eNoN, 1);

        for (const auto& path : paths) {
            for (int i = 0; i < n_pts; ++i) {
                double t = double(i) / (n_pts - 1);
                std::array<double, 3> pt{
                    (1 - t) * path.a[0] + t * path.b[0],
                    (1 - t) * path.a[1] + t * path.b[1],
                    (1 - t) * path.a[2] + t * path.b[2],
                };
                Array<double> xi_pt = pack_xi(ec.insd, pt);
                nn::get_gnn(ec.insd, ec.legacy_type, ec.eNoN, 0, xi_pt, N_leg, Nx_leg);
                svmp::FE::math::Vector<svmp::FE::Real, 3> xi_oop{pt[0], pt[1], pt[2]};
                oop_basis.evaluate_values(xi_oop, N_oop);

                for (int j = 0; j < ec.eNoN; ++j) {
                    int k = perm_legacy_to_oop[j];
                    double No = (k >= 0) ? static_cast<double>(N_oop[k]) : 0.0;
                    cs_csv << ec.name << "," << path.name << "," << t << ","
                           << pt[0] << "," << pt[1] << "," << pt[2] << ","
                           << k << "," << j << ","
                           << N_leg(j, 0) << "," << No << "\n";
                }
            }
        }
        std::cout << "[" << ec.name << "] cross-sections dumped" << std::endl;
    }
}

}  // namespace
