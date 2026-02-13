/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Core/EquationModuleRegistry.h"

#include "FE/Systems/FESystem.h"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Mesh.h"
#endif

namespace svmp::Physics::formulations::navier_stokes {
void forceLink_NavierStokesRegister();
}

namespace svmp::Physics::test {

namespace {

svmp::Physics::ParameterValue defined(std::string v)
{
    return svmp::Physics::ParameterValue{true, std::move(v)};
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

[[nodiscard]] std::filesystem::path repoRoot()
{
    auto p = std::filesystem::path(__FILE__).parent_path();
    for (int i = 0; i < 6; ++i) {
        p = p.parent_path();
    }
    return p;
}

[[nodiscard]] std::filesystem::path beamMeshPath()
{
    return repoRoot() / "tests" / "cases" / "linear-elasticity" / "beam" / "mesh" / "mesh-complete.mesh.vtu";
}

struct InletOutletMarkers {
    svmp::label_t inlet{101};
    svmp::label_t outlet{102};
    int axis{0};
};

InletOutletMarkers labelBeamInletOutlet(svmp::Mesh& mesh_mut)
{
    const auto bbox = mesh_mut.global_bounding_box();
    const double len[3] = {
        static_cast<double>(bbox.max[0] - bbox.min[0]),
        static_cast<double>(bbox.max[1] - bbox.min[1]),
        static_cast<double>(bbox.max[2] - bbox.min[2]),
    };

    int axis = 0;
    if (len[1] > len[axis]) axis = 1;
    if (len[2] > len[axis]) axis = 2;

    const double minv = static_cast<double>(bbox.min[axis]);
    const double maxv = static_cast<double>(bbox.max[axis]);
    const double tol = 1e-10 * std::max(1.0, std::abs(maxv - minv));

    auto& base = mesh_mut.base();
    const auto& f2c = base.face2cell();
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(f2c.size()); ++f) {
        const auto& fc = f2c[static_cast<std::size_t>(f)];
        const bool c0_valid = (fc[0] != svmp::INVALID_INDEX);
        const bool c1_valid = (fc[1] != svmp::INVALID_INDEX);
        if (c0_valid == c1_valid) {
            continue;
        }

        const auto c = base.face_center(f);
        const double v = static_cast<double>(c[axis]);
        if (std::abs(v - minv) <= tol) {
            base.set_boundary_label(f, static_cast<svmp::label_t>(101));
        } else if (std::abs(v - maxv) <= tol) {
            base.set_boundary_label(f, static_cast<svmp::label_t>(102));
        }
    }

    InletOutletMarkers out{};
    out.inlet = 101;
    out.outlet = 102;
    out.axis = axis;
    return out;
}

[[nodiscard]] std::shared_ptr<svmp::Mesh> loadBeamMesh()
{
    const auto path = beamMeshPath();
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Missing beam test mesh file: " + path.string());
    }

    svmp::MeshIOOptions opts;
    opts.format = "vtu";
    opts.path = path.string();

    return svmp::load_mesh(opts, svmp::MeshComm::world());
}

#endif

void expectParabolicInflowVaries(const svmp::FE::systems::FESystem& system, int component)
{
    const auto& constraints = system.constraints();
    const auto comp_dofs = system.fieldMap().getComponentDofs("Velocity", static_cast<svmp::FE::LocalIndex>(component)).toVector();
    std::vector<double> values;
    values.reserve(comp_dofs.size());

    for (const auto dof : comp_dofs) {
        if (!constraints.isConstrained(dof)) {
            continue;
        }
        const auto c = constraints.getConstraint(dof);
        ASSERT_TRUE(c.has_value());
        if (!c.has_value()) {
            continue;
        }
        ASSERT_TRUE(c->isDirichlet());
        values.push_back(c->inhomogeneity);
    }

    ASSERT_FALSE(values.empty());
    const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
    const double minv = *min_it;
    const double maxv = *max_it;
    EXPECT_GT(maxv, 0.0);
    EXPECT_LT(minv, 0.75 * maxv);
}

} // namespace

TEST(NavierStokesLegacyBCs, ParabolicFluxInflow_ResistanceOutflow_SetupSucceeds)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
#  if !defined(MESH_HAS_VTK)
    GTEST_SKIP() << "Requires Mesh built with VTK support (MESH_ENABLE_VTK=ON).";
#  else
    // Ensure the registrar TU is linked so the registry contains the "fluid" factory.
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    auto mesh = loadBeamMesh();
    ASSERT_TRUE(mesh);
    EXPECT_EQ(mesh->dim(), 3);

    const auto markers = labelBeamInletOutlet(*mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "beam";
    input.mesh = mesh->local_mesh_ptr();

    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    // Inflow: Dirichlet, parabolic, impose flux.
    {
        svmp::Physics::BoundaryConditionInput bc{};
        bc.name = "inflow";
        bc.boundary_marker = static_cast<int>(markers.inlet);
        bc.params["Type"] = defined("Dir");
        bc.params["Time_dependence"] = defined("Steady");
        bc.params["Profile"] = defined("Parabolic");
        bc.params["Impose_flux"] = defined("true");
        bc.params["Value"] = defined("-36.5");
        input.boundary_conditions.push_back(std::move(bc));
    }

    // Outflow: Neumann, resistance.
    {
        svmp::Physics::BoundaryConditionInput bc{};
        bc.name = "outlet";
        bc.boundary_marker = static_cast<int>(markers.outlet);
        bc.params["Type"] = defined("Neu");
        bc.params["Time_dependence"] = defined("Resistance");
        bc.params["Value"] = defined("16000");
        input.boundary_conditions.push_back(std::move(bc));
    }

    svmp::FE::systems::FESystem system(mesh);
    auto module = svmp::Physics::EquationModuleRegistry::instance().create("fluid", input, system);
    ASSERT_TRUE(module);
    ASSERT_NO_THROW(system.setup());

    // Coupled resistance outflow uses the CoupledBoundaryManager (boundary integral Q).
    const auto* coupled = system.coupledBoundaryManager();
    ASSERT_NE(coupled, nullptr);
    EXPECT_EQ(coupled->registeredBoundaryFunctionals().size(), 1u);
    EXPECT_EQ(coupled->auxiliaryState().size(), 0u);

    // Parabolic inflow should produce a non-uniform velocity component along the beam axis.
    expectParabolicInflowVaries(system, markers.axis);
#  endif
#endif
}

TEST(NavierStokesLegacyBCs, ParabolicFluxInflow_RCROutflow_SetupSucceeds)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
#  if !defined(MESH_HAS_VTK)
    GTEST_SKIP() << "Requires Mesh built with VTK support (MESH_ENABLE_VTK=ON).";
#  else
    svmp::Physics::formulations::navier_stokes::forceLink_NavierStokesRegister();

    auto mesh = loadBeamMesh();
    ASSERT_TRUE(mesh);
    EXPECT_EQ(mesh->dim(), 3);

    const auto markers = labelBeamInletOutlet(*mesh);

    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "fluid";
    input.mesh_name = "beam";
    input.mesh = mesh->local_mesh_ptr();

    input.default_domain.params["Density"] = defined("1.0");
    input.default_domain.params["Viscosity.model"] = defined("Constant");
    input.default_domain.params["Viscosity.Value"] = defined("0.01");

    // Inflow: Dirichlet, parabolic, impose flux.
    {
        svmp::Physics::BoundaryConditionInput bc{};
        bc.name = "inflow";
        bc.boundary_marker = static_cast<int>(markers.inlet);
        bc.params["Type"] = defined("Dir");
        bc.params["Time_dependence"] = defined("Steady");
        bc.params["Profile"] = defined("Parabolic");
        bc.params["Impose_flux"] = defined("true");
        bc.params["Value"] = defined("-36.5");
        input.boundary_conditions.push_back(std::move(bc));
    }

    // Outflow: Neumann, RCR.
    {
        svmp::Physics::BoundaryConditionInput bc{};
        bc.name = "outlet";
        bc.boundary_marker = static_cast<int>(markers.outlet);
        bc.params["Type"] = defined("Neu");
        bc.params["Time_dependence"] = defined("RCR");
        bc.params["RCR.Capacitance"] = defined("1.5e-5");
        bc.params["RCR.Distal_resistance"] = defined("1212");
        bc.params["RCR.Proximal_resistance"] = defined("121");
        bc.params["RCR.Distal_pressure"] = defined("0");
        bc.params["RCR.Initial_pressure"] = defined("0");
        input.boundary_conditions.push_back(std::move(bc));
    }

    svmp::FE::systems::FESystem system(mesh);
    auto module = svmp::Physics::EquationModuleRegistry::instance().create("fluid", input, system);
    ASSERT_TRUE(module);
    ASSERT_NO_THROW(system.setup());

    const auto* coupled = system.coupledBoundaryManager();
    ASSERT_NE(coupled, nullptr);
    EXPECT_EQ(coupled->registeredBoundaryFunctionals().size(), 1u);
    EXPECT_EQ(coupled->auxiliaryState().size(), 1u);

    expectParabolicInflowVaries(system, markers.axis);
#  endif
#endif
}

} // namespace svmp::Physics::test
