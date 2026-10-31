#include "Forms/MovingFrameForms.h"

#include "Assembly/GlobalSystemView.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <string_view>
#include <vector>

using namespace svmp::FE::forms;

TEST(MovingFrameForms, BuildsFrameTerminalsFromNamedParameters)
{
    const auto terminals = movingFrameTerminals("rotor");
    EXPECT_TRUE(terminals.current_coordinate.isValid());
    EXPECT_TRUE(terminals.mesh_velocity.isValid());
    EXPECT_TRUE(terminals.relative_velocity.isValid());

    const auto origin = terminals.frame_origin.toString();
    EXPECT_NE(origin.find("param(frame.rotor.origin_x)"), std::string::npos) << origin;
    EXPECT_NE(origin.find("param(frame.rotor.origin_y)"), std::string::npos) << origin;
    EXPECT_NE(origin.find("param(frame.rotor.origin_z)"), std::string::npos) << origin;

    const auto rel = terminals.relative_velocity.toString();
    EXPECT_NE(rel.find("meshVelocity"), std::string::npos);
    EXPECT_NE(rel.find("param(frame.rotor.angular_velocity_x)"), std::string::npos) << rel;
    EXPECT_NE(rel.find("x_current"), std::string::npos);

    const auto names = movingFrameParameterNames("rotor");
    ASSERT_EQ(names.size(), 15u);
    EXPECT_EQ(names.front(), "frame.rotor.origin_x");
    EXPECT_EQ(names.back(), "frame.rotor.angular_acceleration_z");
}

namespace {

svmp::FE::dofs::MeshTopologyInfo singleTetraTopology()
{
    svmp::FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 4;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 4};
    topo.cell2vertex_data = {0, 1, 2, 3};
    topo.vertex_gids = {0, 1, 2, 3};
    topo.cell_gids = {0};
    topo.cell_owner_ranks = {0};
    return topo;
}

/// The residual vector of (a + omega . (1, 2, 3)) v dx with a named
/// parameter a, which takes slot 0, next to the frame "rotor".
std::vector<svmp::FE::Real> assembleFrameResidual(bool jit)
{
    using svmp::FE::Real;
    auto mesh = std::make_shared<test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Tetra4, 1);
    svmp::FE::systems::FESystem sys(mesh);
    const auto u_id = sys.addField({.name = "u", .space = space});
    sys.addOperator("op");
    const auto u = StateField(u_id, *space, "u");
    const auto v = TestField(u_id, *space, "v");
    const auto weights = as_vector({FormExpr::constant(1.0), FormExpr::constant(2.0), FormExpr::constant(3.0)});
    const auto residual = (u * v + (FormExpr::parameter("a") + inner(frameAngularVelocity("rotor"), weights)) * v).dx();
    svmp::FE::systems::FormInstallOptions options;
    options.compiler_options.jit.enable = jit;
    svmp::FE::systems::installFormulation(sys, "op", {u_id}, residual, options);
    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const auto n = sys.dofHandler().getNumDofs();
    std::vector<Real> U(static_cast<std::size_t>(n), 0.0);
    svmp::FE::systems::SystemStateView state;
    state.u = U;
    state.getRealParam = [](std::string_view key) -> std::optional<Real> {
        if (key == "a") return 10.0;
        if (key == "frame.rotor.angular_velocity_x") return 1.0;
        if (key == "frame.rotor.angular_velocity_y") return -2.0;
        if (key == "frame.rotor.angular_velocity_z") return 0.5;
        if (key.substr(0, 12) == "frame.rotor.") return 0.0;
        return std::nullopt;
    };
    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_vector = true;
    svmp::FE::assembly::DenseVectorView out(n);
    out.zero();
    (void)sys.assemble(req, state, nullptr, &out);
    std::vector<Real> values;
    for (svmp::FE::GlobalIndex i = 0; i < n; ++i) {
        values.push_back(out.getVectorEntry(i));
    }
    return values;
}

} // namespace

TEST(MovingFrameForms, NamedFrameParametersDoNotAliasOtherParameters)
{
    // a + omega . (1, 2, 3) = 10 + 1 - 4 + 1.5 = 8.5 at every point, and the
    // P1 basis integrates to 1/24 on the unit tetrahedron.  With the frame on
    // the old fixed slots 0-14, omega_x would have read a.
    for (const bool jit : {false, true}) {
        const auto values = assembleFrameResidual(jit);
        ASSERT_EQ(values.size(), 4u);
        for (const auto value : values) {
            EXPECT_NEAR(value, 8.5 / 24.0, 1e-14) << "jit=" << jit;
        }
    }
}
