#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_BC_FACTORIES_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_BC_FACTORIES_H

/**
 * @file NavierStokesBCFactories.h
 * @brief Translation helpers from Navier–Stokes option structs to FE boundary-condition objects
 *
 * Keeps `IncompressibleNavierStokesVMSModule.cpp` declarative by moving option
 * parsing and coupled-BC construction into reusable helpers.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"
#include "Mesh/Core/BoundaryGeometry.h"

#include "FE/Forms/BoundaryFunctional.h"
#include "FE/Forms/CoupledBCs.h"
#include "FE/Forms/StandardBCs.h"
#include "FE/Systems/AuxiliaryStateBuilder.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <cmath>
#include <limits>
#include <optional>
#include <unordered_map>

#if FE_HAS_MPI
#  include <mpi.h>
#endif

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {
namespace Factories {

namespace detail {

[[nodiscard]] inline std::string markerName(std::string_view prefix, int boundary_marker)
{
    std::string out;
    out.reserve(prefix.size() + 1 + 20);
    out.append(prefix.data(), prefix.size());
    out.push_back('_');
    out.append(std::to_string(boundary_marker));
    return out;
}

[[nodiscard]] inline std::string componentName(std::string_view prefix, int boundary_marker, int component)
{
    std::string out;
    out.reserve(prefix.size() + 1 + 20 + 2 + 8);
    out.append(prefix.data(), prefix.size());
    out.push_back('_');
    out.append(std::to_string(boundary_marker));
    out.append("_c");
    out.append(std::to_string(component));
    return out;
}

struct ParabolicProfileData {
  svmp::Mesh::geometry::Vec3d center{};
  std::vector<svmp::Mesh::geometry::Vec3d> perimeter_unit_dirs{};
  std::vector<double> perimeter_r2{};
};

inline ParabolicProfileData build_parabolic_profile_data(const svmp::MeshBase& mesh, int boundary_marker)
{
  using namespace svmp::Mesh::geometry;
  const auto g = global_marker_geometry(mesh, boundary_marker);
  if (!(g.area > 0.0)) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Boundary marker " + std::to_string(boundary_marker) +
        " has zero area; cannot construct parabolic profile.");
  }
  ParabolicProfileData out{};
  out.center = (1.0 / g.area) * g.center_sum;

  const auto all_perim = gather_perimeter_vertex_coords(mesh, boundary_marker);
  std::unordered_map<svmp::gid_t, Vec3d> unique;
  unique.reserve(all_perim.size());
  for (const auto& [gid, p] : all_perim) {
    if (unique.find(gid) == unique.end()) {
      unique.emplace(gid, p);
    }
  }

  out.perimeter_unit_dirs.clear();
  out.perimeter_r2.clear();
  out.perimeter_unit_dirs.reserve(unique.size());
  out.perimeter_r2.reserve(unique.size());

  for (const auto& [_, p] : unique) {
    const Vec3d v = p - out.center;
    const double r2 = norm2(v);
    if (!(r2 > 0.0)) {
      continue;
    }
    out.perimeter_r2.push_back(r2);
    out.perimeter_unit_dirs.push_back((1.0 / std::sqrt(r2)) * v);
  }

  if (out.perimeter_unit_dirs.empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Parabolic inflow profile for boundary marker " + std::to_string(boundary_marker) +
        " requires a non-empty perimeter. Ensure this face shares an edge with another boundary face "
        "(e.g., a wall surface), or set <Profile>Flat</Profile>.");
  }

  return out;
}

inline double parabolic_weight(const ParabolicProfileData& data, const svmp::Mesh::geometry::Vec3d& x)
{
  using namespace svmp::Mesh::geometry;
  const Vec3d r = x - data.center;
  const double r2 = norm2(r);
  if (!(r2 > 0.0)) {
    return 1.0;
  }

  double best = -std::numeric_limits<double>::infinity();
  std::size_t best_i = 0;
  for (std::size_t i = 0; i < data.perimeter_unit_dirs.size(); ++i) {
    const double d = dot(r, data.perimeter_unit_dirs[i]);
    if (d > best) {
      best = d;
      best_i = i;
    }
  }

  const double R2 = data.perimeter_r2[best_i];
  if (!(R2 > 0.0)) {
    return 0.0;
  }

  const double w = 1.0 - (r2 / R2);
  return (w > 0.0) ? w : 0.0;
}

inline double integrate_parabolic_weight_over_marker(const svmp::MeshBase& mesh, int boundary_marker, const ParabolicProfileData& data)
{
  using namespace svmp::Mesh::geometry;
  const auto faces = mesh.faces_with_label(static_cast<svmp::label_t>(boundary_marker));

  auto tri_area = [](const Vec3d& a, const Vec3d& b, const Vec3d& c) {
    const Vec3d ab = b - a;
    const Vec3d ac = c - a;
    return 0.5 * norm(cross(ab, ac));
  };

  auto tri_integral = [&](const Vec3d& p0, const Vec3d& p1, const Vec3d& p2) {
    const double a = tri_area(p0, p1, p2);
    if (!(a > 0.0)) {
      return 0.0;
    }
    // Degree-2 symmetric rule (3 points), weights sum to 1.
    const Vec3d x1 = (1.0 / 6.0) * p0 + (1.0 / 6.0) * p1 + (2.0 / 3.0) * p2;
    const Vec3d x2 = (1.0 / 6.0) * p0 + (2.0 / 3.0) * p1 + (1.0 / 6.0) * p2;
    const Vec3d x3 = (2.0 / 3.0) * p0 + (1.0 / 6.0) * p1 + (1.0 / 6.0) * p2;

    const double f = (parabolic_weight(data, x1) + parabolic_weight(data, x2) + parabolic_weight(data, x3)) / 3.0;
    return a * f;
  };

  auto line_integral = [&](const Vec3d& p0, const Vec3d& p1) {
    const double len = norm(p1 - p0);
    if (!(len > 0.0)) {
      return 0.0;
    }
    // 2-point Gauss rule for a line segment (exact for degree 3)
    const double sqrt3_3 = 0.5773502691896257645; // sqrt(1/3)
    const double w0 = 0.5 * (1.0 - sqrt3_3);
    const double w1 = 0.5 * (1.0 + sqrt3_3);

    const Vec3d x1 = (1.0 - w0) * p0 + w0 * p1;
    const Vec3d x2 = (1.0 - w1) * p0 + w1 * p1;

    // Both Gauss points have a weight of 0.5 relative to the segment length
    const double f = 0.5 * (parabolic_weight(data, x1) + parabolic_weight(data, x2));
    return len * f;
  };

  auto face_integral = [&](const std::vector<svmp::index_t>& verts) {
    if (verts.size() < 2u) {
      return 0.0;
    }

    std::vector<Vec3d> pts;
    pts.reserve(verts.size());
    for (const auto v : verts) {
      if (v == svmp::INVALID_INDEX) {
        continue;
      }
      pts.push_back(to_vec3(mesh.get_vertex_coords(v)));
    }
    
    if (pts.size() < 2u) {
      return 0.0;
    }

    if (pts.size() == 2u) {
      return line_integral(pts[0], pts[1]);
    }

    double val = 0.0;
    const Vec3d p0 = pts[0];
    for (std::size_t i = 1; i + 1 < pts.size(); ++i) {
      val += tri_integral(p0, pts[i], pts[i + 1]);
    }
    return val;
  };

  double local_sum = 0.0;
  for (const auto f : faces) {
    local_sum += face_integral(mesh.face_vertices(f));
  }

  double sum = local_sum;
#if FE_HAS_MPI
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized) {
    MPI_Allreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }
#endif
  return sum;
}

enum class InletProfileType { Flat, Parabolic };

struct InletProfileContext {
  int dim{0};
  InletProfileType profile{InletProfileType::Flat};
  bool use_normal_direction{true};
  std::array<int, 3> active_components{1, 1, 1};

  svmp::Mesh::geometry::Vec3d normal{};
  double scale{0.0};

  std::optional<ParabolicProfileData> parabolic{};

  double weight(const svmp::Mesh::geometry::Vec3d& x) const
  {
    switch (profile) {
      case InletProfileType::Flat: return 1.0;
      case InletProfileType::Parabolic:
        if (!parabolic.has_value()) {
          return 0.0;
        }
        return parabolic_weight(*parabolic, x);
    }
    return 1.0;
  }

  double componentValue(int component, const svmp::Mesh::geometry::Vec3d& x) const
  {
    if (component < 0 || component >= 3) {
      return 0.0;
    }
    const double w = weight(x);
    if (use_normal_direction) {
      const double nd = (component == 0) ? normal.x : (component == 1) ? normal.y : normal.z;
      return scale * w * nd;
    }
    const int active = active_components[static_cast<std::size_t>(component)];
    if (active == 0) {
      return 0.0;
    }
    return scale * w;
  }
};

} // namespace detail

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition> reserveMarker(
    const IncompressibleNavierStokesVMSOptions::VelocityDirichletBC& bc)
{
    const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "navier_stokes::Factories::reserveMarker");
    return std::make_unique<FE::forms::bc::ReservedBC>(marker);
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition> toTractionBC(
    const IncompressibleNavierStokesVMSOptions::TractionNeumannBC& bc,
    int dim)
{
    const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "navier_stokes::Factories::toTractionBC");

    std::vector<FE::forms::FormExpr> t_comp;
    t_comp.reserve(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        t_comp.push_back(FE::forms::bc::toScalarExpr(
            bc.traction[static_cast<std::size_t>(d)],
            detail::componentName("ns_traction_neumann", marker, d)));
    }
    return std::make_unique<FE::forms::bc::NaturalBC>(marker, FE::forms::FormExpr::asVector(std::move(t_comp)));
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition> toTractionRobinBC(
    const IncompressibleNavierStokesVMSOptions::TractionRobinBC& bc,
    int dim)
{
    const int marker =
        FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "navier_stokes::Factories::toTractionRobinBC");

    const auto alpha = FE::forms::bc::toScalarExpr(bc.alpha, detail::markerName("ns_traction_robin_alpha", marker));

    std::vector<FE::forms::FormExpr> r_comp;
    r_comp.reserve(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        r_comp.push_back(FE::forms::bc::toScalarExpr(
            bc.rhs[static_cast<std::size_t>(d)],
            detail::componentName("ns_traction_robin_rhs", marker, d)));
    }

    return std::make_unique<FE::forms::bc::RobinBC>(
        marker, alpha, FE::forms::FormExpr::asVector(std::move(r_comp)));
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition> toVelocityEssentialBC(
    const IncompressibleNavierStokesVMSOptions::VelocityDirichletBC& bc,
    int dim,
    std::string_view symbol)
{
    const int marker =
        FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "navier_stokes::Factories::toVelocityEssentialBC");

    std::vector<FE::forms::FormExpr> uD_comp;
    uD_comp.reserve(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        uD_comp.push_back(FE::forms::bc::toScalarExpr(
            bc.value[static_cast<std::size_t>(d)],
            detail::componentName("ns_u_dirichlet", marker, d)));
    }

    return std::make_unique<FE::forms::bc::EssentialBC>(marker, std::move(uD_comp), std::string(symbol));
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition> toPressureEssentialBC(
    const IncompressibleNavierStokesVMSOptions::PressureDirichletBC& bc,
    std::string_view symbol)
{
    const int marker =
        FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "navier_stokes::Factories::toPressureEssentialBC");
    const auto value = FE::forms::bc::toScalarExpr(bc.value, detail::markerName("ns_p_dirichlet", marker));
    return std::make_unique<FE::forms::bc::EssentialBC>(marker, value, std::string(symbol));
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition> toOutflowBC(
    const IncompressibleNavierStokesVMSOptions::PressureOutflowBC& bc,
    const FE::forms::FormExpr& u,
    const FE::forms::FormExpr& rho)
{
    const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "navier_stokes::Factories::toOutflowBC");

    const auto n = FE::forms::FormExpr::normal();
    const auto un = FE::forms::inner(u, n);
    const auto max_backflow =
        FE::forms::FormExpr::constant(0.5) * (FE::forms::abs(un) - un); // max(0, -u·n)

    const auto p_out = FE::forms::bc::toScalarExpr(bc.pressure, detail::markerName("ns_p_out", marker));
    const auto beta = FE::forms::bc::toScalarExpr(bc.backflow_beta, detail::markerName("ns_backflow_beta", marker));

    const auto flux = -p_out * n - beta * rho * max_backflow * u;
    return std::make_unique<FE::forms::bc::NaturalBC>(marker, flux);
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition> toCoupledOutflowBC(
    const IncompressibleNavierStokesVMSOptions::CoupledRCROutflowBC& bc,
    FE::FieldId u_id,
    const FE::spaces::FunctionSpace& velocity_space,
    std::string_view velocity_field_name,
    const FE::forms::FormExpr& u,
    const FE::forms::FormExpr& rho)
{
    const int marker =
        FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "navier_stokes::Factories::toCoupledOutflowBC");

    const std::string q_name = bc.functional_name.empty() ? ("ns_Q_" + std::to_string(marker)) : bc.functional_name;
    const std::string x_name = bc.state_name.empty() ? ("ns_X_" + std::to_string(marker)) : bc.state_name;

    const FE::Real Rp = bc.Rp;
    const FE::Real C = bc.C;
    const FE::Real Rd = bc.Rd;
    const FE::Real Pd = bc.Pd;

    if (Rd == 0.0) {
        throw std::invalid_argument("CoupledRCROutflowBC: Rd must be nonzero");
    }

    const auto n = FE::forms::FormExpr::normal();
    const auto un = FE::forms::inner(u, n);
    const auto max_backflow =
        FE::forms::FormExpr::constant(0.5) * (FE::forms::abs(un) - un); // max(0, -u·n)
    const auto beta =
        FE::forms::bc::toScalarExpr(bc.backflow_beta, detail::markerName("ns_rcr_backflow_beta", marker));

    const auto u_disc =
        FE::forms::FormExpr::discreteField(u_id, velocity_space, std::string(velocity_field_name));
    const auto Q_integrand = FE::forms::inner(u_disc, n);
    const auto Qsym = FE::forms::FormExpr::boundaryIntegral(Q_integrand, marker, q_name);

    std::vector<FE::systems::AuxiliaryStateRegistration> regs;

    FE::forms::FormExpr p_out;
    if (C == 0.0) {
        // Purely resistive limit: X = Pd + Rd*Q, so p_out = X + Rp*Q = Pd + (Rd+Rp)*Q.
        const FE::Real Rsum = Rd + Rp;
        p_out = FE::forms::FormExpr::constant(Pd) + FE::forms::FormExpr::constant(Rsum) * Qsym;
    } else {
        FE::forms::BoundaryFunctional Q;
        Q.integrand = Q_integrand;
        Q.boundary_marker = marker;
        Q.name = q_name;
        Q.reduction = FE::forms::BoundaryFunctional::Reduction::Sum;

        // ODE: dX/dt = (Q - (X - Pd)/Rd) / C
        const auto Xsym = FE::forms::FormExpr::auxiliaryState(x_name);
        const auto f = (Qsym - (Xsym - FE::forms::FormExpr::constant(Pd)) / FE::forms::FormExpr::constant(Rd)) /
            FE::forms::FormExpr::constant(C);

        FE::systems::AuxiliaryStateRegistration rcr;
        rcr.name = x_name;
        rcr.initial_value = bc.X0;
        rcr.derivative = f;
        rcr.dependency_functionals.push_back(std::move(Q));

        regs.push_back(std::move(rcr));

        p_out = Xsym + FE::forms::FormExpr::constant(Rp) * Qsym;
    }

    const auto flux = -p_out * n - beta * rho * max_backflow * u;
    return std::make_unique<FE::forms::bc::CoupledNaturalBC>(marker, flux, std::move(regs));
}

} // namespace Factories
} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_BC_FACTORIES_H
