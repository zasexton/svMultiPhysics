#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Core/EquationModuleRegistry.h"
#include "Physics/Core/InputHelpers.h"
#include "Physics/Materials/Fluid/CarreauYasudaViscosity.h"
#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"

#include "Application/IO/TemporalSpatialReader.h"
#include "Physics/Core/StringUtils.h"

#include "FE/Spaces/SpaceFactory.h"
#include "FE/Spaces/SpaceInference.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Core/BoundaryGeometry.h"

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using svmp::Physics::utils::parse_bool_relaxed;
using svmp::Physics::utils::parse_double;
using svmp::Physics::utils::trim_copy;
using svmp::Physics::utils::lower_copy;
using svmp::Physics::find_param;
using svmp::Physics::has_nonempty_defined;
using svmp::Physics::get_defined_double;
using svmp::Physics::select_single_domain;
using namespace svmp::Mesh::geometry;
using namespace svmp::Physics::formulations::navier_stokes::Factories::detail;

void apply_fluid_properties(const svmp::Physics::DomainInput& domain,
                            svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  using svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions;

  if (const auto rho = get_defined_double(domain.params, "Density")) {
    options.density = static_cast<svmp::FE::Real>(*rho);
  } else if (const auto rho2 = get_defined_double(domain.params, "Fluid_density")) {
    options.density = static_cast<svmp::FE::Real>(*rho2);
  }

  if (const auto fx = get_defined_double(domain.params, "Force_x")) {
    options.body_force[0] = static_cast<svmp::FE::Real>(*fx);
  }
  if (const auto fy = get_defined_double(domain.params, "Force_y")) {
    options.body_force[1] = static_cast<svmp::FE::Real>(*fy);
  }
  if (const auto fz = get_defined_double(domain.params, "Force_z")) {
    options.body_force[2] = static_cast<svmp::FE::Real>(*fz);
  }

  const auto* model_param = find_param(domain.params, "Viscosity.model");
  if (!model_param || !model_param->defined || trim_copy(model_param->value).empty()) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] <Viscosity model=\"...\"> is required for the new solver Navier-Stokes module. "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  const auto model_raw = trim_copy(model_param->value);
  const auto model = lower_copy(model_raw);

  if (model == "constant") {
    const auto* mu_param = find_param(domain.params, "Viscosity.Value");
    const double mu = mu_param ? parse_double(mu_param->value, "Viscosity/Value") : 0.0;
    if (!(mu > 0.0)) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Constant viscosity must be > 0 for the new solver Navier-Stokes module. "
          "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }
    options.viscosity = static_cast<svmp::FE::Real>(mu);
    options.viscosity_model.reset();
    return;
  }

  if (model == "carreau-yasuda" || model == "carreau_yasuda") {
    const auto* p_mu_inf = find_param(domain.params, "Viscosity.Limiting_high_shear_rate_viscosity");
    const auto* p_mu0 = find_param(domain.params, "Viscosity.Limiting_low_shear_rate_viscosity");
    const auto* p_lambda = find_param(domain.params, "Viscosity.Shear_rate_tensor_multiplier");
    const auto* p_n = find_param(domain.params, "Viscosity.Power_law_index");
    const auto* p_a = find_param(domain.params, "Viscosity.Shear_rate_tensor_exponent");

    const auto mu_inf = static_cast<svmp::FE::Real>(
        p_mu_inf ? parse_double(p_mu_inf->value, "Viscosity/Limiting_high_shear_rate_viscosity") : 0.0);
    const auto mu0 = static_cast<svmp::FE::Real>(
        p_mu0 ? parse_double(p_mu0->value, "Viscosity/Limiting_low_shear_rate_viscosity") : 0.0);
    const auto lambda = static_cast<svmp::FE::Real>(
        p_lambda ? parse_double(p_lambda->value, "Viscosity/Shear_rate_tensor_multiplier") : 0.0);
    const auto n =
        static_cast<svmp::FE::Real>(p_n ? parse_double(p_n->value, "Viscosity/Power_law_index") : 0.0);
    const auto a = static_cast<svmp::FE::Real>(
        p_a ? parse_double(p_a->value, "Viscosity/Shear_rate_tensor_exponent") : 0.0);

    try {
      options.viscosity_model = std::make_shared<svmp::Physics::materials::fluid::CarreauYasudaViscosity>(
          mu0, mu_inf, lambda, n, a);
    } catch (const std::exception& e) {
      throw std::runtime_error(
          std::string("[svMultiPhysics::Physics] Invalid Carreau-Yasuda viscosity parameters for the new solver "
                      "Navier-Stokes module: ") +
          e.what() + ". Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }

    if (p_mu0 && p_mu0->defined) {
      options.viscosity = mu0;
    }
    return;
  }

  if (model == "cassons" || model == "casson") {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Cassons viscosity model is not supported by the new solver Navier-Stokes module yet. "
        "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  throw std::runtime_error(
      "[svMultiPhysics::Physics] Fluid viscosity model '" + model_raw +
      "' is not supported by the new solver Navier-Stokes module. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> "
      "to use the legacy solver.");
}

std::vector<int> parse_int_list(std::string_view raw)
{
  std::istringstream iss{std::string(raw)};
  std::vector<int> out;
  int v = 0;
  while (iss >> v) {
    out.push_back(v);
  }
  return out;
}

svmp::FE::Real direction_component(const std::vector<int>& effective_dir, int component)
{
  if (component < 0) {
    return static_cast<svmp::FE::Real>(0.0);
  }
  const auto idx = static_cast<std::size_t>(component);
  if (idx >= effective_dir.size()) {
    return static_cast<svmp::FE::Real>(1.0);
  }
  return static_cast<svmp::FE::Real>(effective_dir[idx]);
}

template <class ScalarValue>
void fill_vector(std::array<ScalarValue, 3>& dst, int dim, const std::vector<int>& effective_dir,
                 svmp::FE::Real magnitude)
{
  dst = {ScalarValue{0.0}, ScalarValue{0.0}, ScalarValue{0.0}};
  for (int d = 0; d < dim; ++d) {
    const auto scale =
        effective_dir.empty() ? static_cast<svmp::FE::Real>(1.0) : direction_component(effective_dir, d);
    dst[static_cast<std::size_t>(d)] = ScalarValue{static_cast<svmp::FE::Real>(magnitude * scale)};
  }
}

void apply_fluid_bcs(const svmp::Physics::EquationModuleInput& input,
                     const svmp::Physics::DomainInput& domain,
                     svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions& options)
{
  using svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions;

  if (!input.mesh) {
    throw std::runtime_error("[svMultiPhysics::Physics] Navier-Stokes BC translation received null mesh.");
  }

  const int dim = input.mesh->dim();
  if (dim < 1 || dim > 3) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Unsupported mesh dimension for Navier-Stokes BC translation: " +
        std::to_string(dim) +
        ". Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  svmp::FE::Real backflow_beta = 0.0;
  if (const auto* p = find_param(domain.params, "Backflow_stabilization_coefficient")) {
    if (!trim_copy(p->value).empty()) {
      backflow_beta = static_cast<svmp::FE::Real>(parse_double(p->value, "Backflow_stabilization_coefficient"));
    }
  }

  for (const auto& bc : input.boundary_conditions) {
    if (bc.boundary_marker == svmp::INVALID_LABEL) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Boundary condition '" + bc.name +
          "' has invalid boundary marker; ensure <Add_face name=\"...\"> exists and is referenced correctly.");
    }

    const auto* bc_type_raw = find_param(bc.params, "Type");
    const std::string bc_type = bc_type_raw ? trim_copy(bc_type_raw->value) : std::string{};
    const std::string bc_type_lc = lower_copy(bc_type);

    const auto* time_dep = find_param(bc.params, "Time_dependence");
    const std::string time_value_raw =
        (time_dep && time_dep->defined) ? trim_copy(time_dep->value) : std::string("Steady");
    const std::string time_value_lc = lower_copy(time_value_raw);

    const bool is_steady = time_value_lc.empty() || time_value_lc == "steady";
    const bool is_general = time_value_lc == "general";
    const bool is_resistance = time_value_lc == "resistance";
    const bool is_rcr = (time_value_lc == "rcr" || time_value_lc == "windkessel");

    const bool has_temp_spat = has_nonempty_defined(bc.params, "Temporal_and_spatial_values_file_path");
    const bool has_other_files = has_nonempty_defined(bc.params, "Temporal_values_file_path") ||
        has_nonempty_defined(bc.params, "Spatial_values_file_path") || has_nonempty_defined(bc.params, "Bct_file_path") ||
        has_nonempty_defined(bc.params, "Traction_values_file_path") ||
        has_nonempty_defined(bc.params, "Fourier_coefficients_file_path") ||
        has_nonempty_defined(bc.params, "Spatial_profile_file_path");

    if (is_steady && (has_temp_spat || has_other_files)) {
      throw std::runtime_error(
          "[svMultiPhysics::Physics] Spatial/temporal boundary-condition files are not supported for the new solver "
          "Navier-Stokes module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }

    const auto* value_param = find_param(bc.params, "Value");
    const svmp::FE::Real magnitude =
        static_cast<svmp::FE::Real>(value_param ? parse_double(value_param->value, "Add_BC/Value") : 0.0);

    std::vector<int> effective_dir{};
    if (const auto* dir_param = find_param(bc.params, "Effective_direction")) {
      const auto s = trim_copy(dir_param->value);
      if (!s.empty()) {
        effective_dir = parse_int_list(s);
      }
    }

    if (bc_type_lc == "dir" || bc_type_lc == "dirichlet") {
      if (!is_steady && !is_general) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Only Steady and General boundary conditions are supported for the new solver "
            "Navier-Stokes module Dirichlet BCs (got Time_dependence='" +
            time_value_raw + "'). Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
      }

      if (const auto* p = find_param(bc.params, "Apply_along_normal_direction");
          p && p->defined && parse_bool_relaxed(p->value)) {
        // Legacy inputs sometimes set this flag for clarity; the new solver applies
        // velocity Dirichlet along the boundary normal by default when Effective_direction is unset.
        // Accept the flag for Dirichlet to improve legacy compatibility.
      }

      const auto* weak_param = find_param(bc.params, "Weakly_applied");
      const bool weak = weak_param && weak_param->defined && parse_bool_relaxed(weak_param->value);

      if (is_general) {
        if (!has_temp_spat || has_other_files) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] General Navier-Stokes Dirichlet BC currently supports only "
              "<Temporal_and_spatial_values_file_path>. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
        }
        if (weak) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] General Dirichlet BCs from temporal/spatial files are only supported as strong "
              "Dirichlet (Weakly_applied=false). Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
        }

        const auto* file_param = find_param(bc.params, "Temporal_and_spatial_values_file_path");
        const std::string file_path =
            (file_param && file_param->defined) ? trim_copy(file_param->value) : std::string{};
        if (file_path.empty()) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] General Dirichlet BC is missing Temporal_and_spatial_values_file_path.");
        }

        auto data = svmp::Application::io::read_temporal_and_spatial_values_file(*input.mesh, bc.boundary_marker, file_path);

        IncompressibleNavierStokesVMSOptions::VelocityDirichletBC dir{};
        dir.boundary_marker = bc.boundary_marker;

        for (int d = 0; d < dim; ++d) {
          if (d < data->dof) {
            const int comp = d;
            dir.value[static_cast<std::size_t>(d)] = svmp::FE::forms::TimeScalarCoefficient(
                [data, comp](svmp::FE::Real x, svmp::FE::Real y, svmp::FE::Real z, svmp::FE::Real t) -> svmp::FE::Real {
                  const std::array<svmp::FE::Real, 3> p{x, y, z};
                  const auto node = data->findNodeIndex(p);
                  return data->interpolate(node, t, comp);
                });
          } else {
            dir.value[static_cast<std::size_t>(d)] = IncompressibleNavierStokesVMSOptions::ScalarValue{0.0};
          }
        }

        options.velocity_dirichlet.push_back(std::move(dir));
        continue;
      }

      IncompressibleNavierStokesVMSOptions::VelocityDirichletBC dir{};
      dir.boundary_marker = bc.boundary_marker;

      // Legacy-style Dirichlet velocity BCs:
      // - default direction: boundary outward normal
      // - profile: Flat or Parabolic
      // - Impose_flux: if true, normalize profile so ∫ profile ds = 1 and interpret Value as flow rate
      const auto* impose_flux_param = find_param(bc.params, "Impose_flux");
      const bool impose_flux = impose_flux_param ? parse_bool_relaxed(impose_flux_param->value) : false;

      const auto* profile_param = find_param(bc.params, "Profile");
      const std::string profile_raw =
          (profile_param && profile_param->defined) ? trim_copy(profile_param->value) : std::string("Flat");
      const std::string profile_lc = lower_copy(profile_raw);

      InletProfileType profile = InletProfileType::Flat;
      if (profile_lc == "flat") {
        profile = InletProfileType::Flat;
      } else if (profile_lc == "parabolic") {
        profile = InletProfileType::Parabolic;
      } else if (profile_lc == "user_defined" || profile_lc == "user-defined" || profile_lc == "userdefined") {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Dirichlet BC Profile='User_defined' is not supported for the new solver "
            "Navier-Stokes module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
      } else {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Unknown Dirichlet BC Profile='" + profile_raw +
            "' for the new solver Navier-Stokes module. Supported profiles: Flat, Parabolic.");
      }

      // Common case: no-slip wall (Value=0) and other zero Dirichlet constraints.
      // Avoid expensive global geometry computations (MPI allgathers) when the imposed value is zero.
      if (magnitude == static_cast<svmp::FE::Real>(0.0)) {
        if (weak) {
          options.velocity_dirichlet_weak.push_back(std::move(dir));
        } else {
          options.velocity_dirichlet.push_back(std::move(dir));
        }
        continue;
      }

      int active_count = 0;
      std::array<int, 3> active{0, 0, 0};
      for (int d = 0; d < dim; ++d) {
        const int flag = (static_cast<std::size_t>(d) < effective_dir.size() && effective_dir[static_cast<std::size_t>(d)] != 0) ? 1 : 0;
        active[static_cast<std::size_t>(d)] = flag;
        active_count += flag;
      }
      const bool use_normal_direction = (active_count == 0 || active_count == dim);

      auto ctx = std::make_shared<InletProfileContext>();
      ctx->dim = dim;
      ctx->profile = profile;
      ctx->use_normal_direction = use_normal_direction;
      ctx->active_components = active;

      // Compute normal/area/center and profile normalization as needed.
      const auto g = global_marker_geometry(*input.mesh, bc.boundary_marker);
      if (!(g.area > 0.0)) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] Boundary marker " + std::to_string(bc.boundary_marker) +
            " has zero area; cannot apply Dirichlet BC '" + bc.name + "'.");
      }

      if (use_normal_direction) {
        const Vec3d n = normalized(g.normal_sum);
        if (!(norm2(n) > 0.0)) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] Boundary marker " + std::to_string(bc.boundary_marker) +
              " has a degenerate normal; cannot apply Dirichlet BC '" + bc.name + "'.");
        }
        ctx->normal = n;
      }

      if (profile == InletProfileType::Parabolic) {
        ctx->parabolic = build_parabolic_profile_data(*input.mesh, bc.boundary_marker);
      }

      double normalization = 1.0;
      if (impose_flux) {
        if (profile == InletProfileType::Flat) {
          normalization = g.area;
        } else {
          normalization = integrate_parabolic_weight_over_marker(*input.mesh, bc.boundary_marker, *ctx->parabolic);
        }
        if (!(normalization > 0.0)) {
          const auto local_faces = input.mesh->faces_with_label(static_cast<svmp::label_t>(bc.boundary_marker));
          const std::size_t perim_points =
              (ctx->parabolic.has_value()) ? ctx->parabolic->perimeter_unit_dirs.size() : 0u;
          throw std::runtime_error(
              "[svMultiPhysics::Physics] Failed to compute a positive normalization for <Impose_flux>true</Impose_flux> "
              "on Dirichlet BC '" + bc.name + "' (marker=" + std::to_string(bc.boundary_marker) +
              ", normalization=" + std::to_string(normalization) + ", area=" + std::to_string(g.area) +
              ", perim_points=" + std::to_string(perim_points) + ", local_faces=" + std::to_string(local_faces.size()) +
              ").");
        }
      }

      ctx->scale = static_cast<double>(magnitude) / normalization;

      for (int d = 0; d < dim; ++d) {
        const int comp = d;
        dir.value[static_cast<std::size_t>(d)] = svmp::FE::forms::TimeScalarCoefficient(
            [ctx, comp](svmp::FE::Real x, svmp::FE::Real y, svmp::FE::Real z, svmp::FE::Real /*t*/) -> svmp::FE::Real {
              const Vec3d p{static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)};
              return static_cast<svmp::FE::Real>(ctx->componentValue(comp, p));
            });
      }

      if (weak) {
        options.velocity_dirichlet_weak.push_back(std::move(dir));
      } else {
        options.velocity_dirichlet.push_back(std::move(dir));
      }
      continue;
    }

    if (bc_type_lc == "neu" || bc_type_lc == "neumann") {
      if (const auto* p = find_param(bc.params, "Apply_along_normal_direction");
          p && p->defined && parse_bool_relaxed(p->value)) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] <Apply_along_normal_direction>true</Apply_along_normal_direction> is not "
            "supported for the new solver Navier-Stokes module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to "
            "use the legacy solver.");
      }

      if (is_steady) {
        IncompressibleNavierStokesVMSOptions::PressureOutflowBC out{};
        out.boundary_marker = bc.boundary_marker;
        out.pressure = IncompressibleNavierStokesVMSOptions::ScalarValue{magnitude};
        out.backflow_beta = IncompressibleNavierStokesVMSOptions::ScalarValue{backflow_beta};
        options.pressure_outflow.push_back(std::move(out));
        continue;
      }

      if (is_resistance) {
        if (has_temp_spat || has_other_files) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] Resistance outflow BCs do not support spatial/temporal files for the new "
              "solver Navier-Stokes module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
        }

        IncompressibleNavierStokesVMSOptions::CoupledRCROutflowBC out{};
        out.boundary_marker = bc.boundary_marker;
        out.Rp = 0.0;
        out.C = 0.0;
        out.Rd = static_cast<svmp::FE::Real>(magnitude);
        out.Pd = 0.0;
        out.X0 = 0.0;
        out.backflow_beta = IncompressibleNavierStokesVMSOptions::ScalarValue{backflow_beta};
        options.coupled_outflow_rcr.push_back(std::move(out));
        continue;
      }

      if (is_rcr) {
        if (has_temp_spat || has_other_files) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] RCR outflow BCs do not support spatial/temporal files for the new solver "
              "Navier-Stokes module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
        }

        const auto Rp = get_defined_double(bc.params, "RCR.Proximal_resistance");
        const auto C = get_defined_double(bc.params, "RCR.Capacitance");
        const auto Rd = get_defined_double(bc.params, "RCR.Distal_resistance");
        const auto Pd = get_defined_double(bc.params, "RCR.Distal_pressure");
        const auto X0 = get_defined_double(bc.params, "RCR.Initial_pressure");

        if (!Rp.has_value() || !C.has_value() || !Rd.has_value()) {
          throw std::runtime_error(
              "[svMultiPhysics::Physics] RCR outflow BC '" + bc.name + "' is missing required <RCR_values> entries "
              "(Proximal_resistance, Capacitance, Distal_resistance).");
        }

        IncompressibleNavierStokesVMSOptions::CoupledRCROutflowBC out{};
        out.boundary_marker = bc.boundary_marker;
        out.Rp = static_cast<svmp::FE::Real>(*Rp);
        out.C = static_cast<svmp::FE::Real>(*C);
        out.Rd = static_cast<svmp::FE::Real>(*Rd);
        out.Pd = static_cast<svmp::FE::Real>(Pd.value_or(0.0));
        out.X0 = static_cast<svmp::FE::Real>(X0.value_or(0.0));
        out.backflow_beta = IncompressibleNavierStokesVMSOptions::ScalarValue{backflow_beta};
        options.coupled_outflow_rcr.push_back(std::move(out));
        continue;
      }

      throw std::runtime_error(
          "[svMultiPhysics::Physics] Neumann BC Time_dependence='" + time_value_raw +
          "' is not supported for the new solver Navier-Stokes module. Supported: Steady, Resistance, RCR. "
          "Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
    }

    if (bc_type_lc == "trac" || bc_type_lc == "traction") {
      if (!is_steady) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] General/Unsteady traction BCs are not supported for the new solver Navier-Stokes "
            "module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
      }
      IncompressibleNavierStokesVMSOptions::TractionNeumannBC trac{};
      trac.boundary_marker = bc.boundary_marker;
      fill_vector(trac.traction, dim, effective_dir, magnitude);
      options.traction_neumann.push_back(std::move(trac));
      continue;
    }

    if (bc_type_lc == "rbn" || bc_type_lc == "robin") {
      if (!is_steady) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] General/Unsteady Robin BCs are not supported for the new solver Navier-Stokes "
            "module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
      }
      if (const auto* p = find_param(bc.params, "Apply_along_normal_direction");
          p && p->defined && parse_bool_relaxed(p->value)) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] <Apply_along_normal_direction>true</Apply_along_normal_direction> is not "
            "supported for the new solver Navier-Stokes module yet. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to "
            "use the legacy solver.");
      }
      const auto* stiff_param = find_param(bc.params, "Stiffness");
      const svmp::FE::Real stiff =
          static_cast<svmp::FE::Real>(stiff_param ? parse_double(stiff_param->value, "Add_BC/Stiffness") : 0.0);

      IncompressibleNavierStokesVMSOptions::TractionRobinBC robin{};
      robin.boundary_marker = bc.boundary_marker;
      robin.alpha = IncompressibleNavierStokesVMSOptions::ScalarValue{stiff};
      fill_vector(robin.rhs, dim, effective_dir, magnitude);
      options.traction_robin.push_back(std::move(robin));
      continue;
    }

    throw std::runtime_error(
        "[svMultiPhysics::Physics] Boundary condition type '" + bc_type +
        "' is not supported for the new solver Navier-Stokes module. Supported types: Dir, Dirichlet, Neu, Neumann, "
        "Trac, Traction, Robin, Rbn. Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }
}

std::unique_ptr<svmp::Physics::PhysicsModule>
create_navier_stokes_from_input(const svmp::Physics::EquationModuleInput& input,
                                svmp::FE::systems::FESystem& system)
{
  if (!input.mesh) {
    throw std::runtime_error("[svMultiPhysics::Physics] Navier-Stokes module factory received null mesh.");
  }

  const auto& domain = select_single_domain(input, "Navier-Stokes");

  const int dim = input.mesh->dim();
  if (dim < 1 || dim > 3) {
    throw std::runtime_error(
        "[svMultiPhysics::Physics] Unsupported mesh dimension for Navier-Stokes spaces: " + std::to_string(dim) +
        ". Set <Use_new_OOP_solver>false</Use_new_OOP_solver> to use the legacy solver.");
  }

  const auto element_type = svmp::FE::spaces::infer_element_type(*input.mesh);
  const int vel_order = svmp::FE::spaces::infer_polynomial_order(*input.mesh);

  bool taylor_hood = false;
  if (const auto* p = find_param(input.equation_params, "Use_taylor_hood_type_basis"); p && p->defined) {
    taylor_hood = parse_bool_relaxed(p->value);
  }
  const int p_order = taylor_hood ? std::max(1, vel_order - 1) : vel_order;

  auto velocity_space =
      svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1, element_type, vel_order, dim);
  auto pressure_space = svmp::FE::spaces::SpaceFactory::create_h1(element_type, p_order);

  svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSOptions options{};
  options.velocity_field_name = "Velocity";
  options.pressure_field_name = "Pressure";
  options.enable_convection = (input.equation_type != "stokes");

  apply_fluid_properties(domain, options);
  apply_fluid_bcs(input, domain, options);

  auto module = std::make_unique<svmp::Physics::formulations::navier_stokes::IncompressibleNavierStokesVMSModule>(
      std::move(velocity_space), std::move(pressure_space), std::move(options));
  module->registerOn(system);
  return module;
}

} // namespace

SVMP_REGISTER_EQUATION("fluid", &create_navier_stokes_from_input);
SVMP_REGISTER_EQUATION("stokes", &create_navier_stokes_from_input);

namespace svmp::Physics::formulations::navier_stokes {

void forceLink_NavierStokesRegister() {}

} // namespace svmp::Physics::formulations::navier_stokes
