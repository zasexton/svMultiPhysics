/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

// Why there is no monolithic FSI coupling (decision D62 item 2): the interface
// forms the FSI coupling module used to install, written here directly in the
// forms vocabulary on one system that holds the fluid and the solid fields.
//
// The intended interface conditions are u_f = v_s and sigma_f n_f = sigma_s n_f.
// A consistent coupling vanishes at an exact state of the coupled problem: its
// fluid rows carry -int sigma_f n_f . w_f (the traction of the solid, which
// closes the fluid's natural boundary term) and its solid rows
// int sigma_f n_f . w_s.  The velocity-continuity form gives the fluid rows only
// the unit-weight penalty int (u_f - v_s) . w_f: at a matched interface
// velocity they get no traction (a traction-free boundary for the fluid), and a
// velocity jump enters with weight 1, independent of the viscosity and of h
// (the Robin condition sigma_f n_f = -(u_f - v_s), not velocity continuity).
// The traction-balance form cannot be assembled: the interpreter has no spatial
// derivative of the interface-restricted fluid velocity it differentiates (and
// the JIT refuses the gradient operand); its pressure part alone assembles.

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Dofs/EntityDofMap.h"
#include "Forms/Vocabulary.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Core/MeshBase.h"
#  include "Mesh/Mesh.h"
#  include "Mesh/Topology/CellShape.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace svmp;
namespace forms = svmp::FE::forms;

namespace {

constexpr int kInterfaceMarker = 10;

// [0, 1] x [0, 1] (the fluid) and [1, 2] x [0, 1] (the solid), one Quad4
// each, sharing the face x = 1 (face set "interface").
std::shared_ptr<svmp::Mesh> fluidAndSolidSquares()
{
    auto base = std::make_shared<svmp::MeshBase>();
    const std::vector<svmp::real_t> X_ref = {
        0.0, 0.0,  1.0, 0.0,  2.0, 0.0,
        0.0, 1.0,  1.0, 1.0,  2.0, 1.0};
    const std::vector<svmp::offset_t> offsets = {0, 4, 8};
    const std::vector<svmp::index_t> cells = {0, 1, 4, 3, 1, 2, 5, 4};
    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(2, X_ref, offsets, cells, {shape, shape});
    base->finalize();
    auto mesh = svmp::create_mesh(std::move(base));
    auto& mbase = mesh->local_mesh();
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(mbase.n_faces()); ++f) {
        const auto verts = mbase.face_vertices(f);
        if (verts.size() == 2u && std::count(verts.begin(), verts.end(), 1) + std::count(verts.begin(), verts.end(), 4) == 2) {
            mbase.add_to_set(svmp::EntityKind::Face, "interface", f);
        }
    }
    return mesh;
}

/// The interface residual summed over the x and y rows of the fluid velocity
/// (w_f = e_x, e_y) and of the solid displacement (w_s = e_x, e_y).
struct InterfaceResidualSums {
    FE::Real fluid_x{0.0};
    FE::Real fluid_y{0.0};
    FE::Real solid_x{0.0};
    FE::Real solid_y{0.0};
};

enum class InterfaceForm {
    VelocityContinuity,   // int (u_f - v_s) . w_f
    TractionBalance,      // int (-p_f n_f + 2 mu sym(grad u_f) n_f) . w_s
};

// Installs @p form on one system holding every field (the fluid on the minus
// side of the interface, the solid on the plus side) and evaluates it at
// u_f = (0, a x), p_f = p0 and v_s = (0, a + jump).  The interface is x = 1
// with unit length; the fluid traction there is t_f = sigma_f n_f = (-p0, mu a).
// mu = 0 leaves the pressure traction alone.
InterfaceResidualSums interfaceResidualSums(InterfaceForm form, FE::Real mu, FE::Real a, FE::Real p0,
                                            FE::Real jump)
{
    auto mesh = fluidAndSolidSquares();
    FE::systems::FESystem system(mesh);
    system.setInterfaceMeshFromFaceSet(kInterfaceMarker, "interface");
    const auto quad_space = [](int components) {
        return FE::spaces::Space(FE::spaces::SpaceType::H1, FE::ElementType::Quad4, 1, components);
    };
    const auto vector_space = quad_space(2);
    const auto scalar_space = quad_space(1);
    const auto u_f_id = system.addField({.name = "fluid_velocity", .space = vector_space, .components = 2});
    const auto p_f_id = system.addField({.name = "fluid_pressure", .space = scalar_space, .components = 1});
    const auto d_s_id = system.addField({.name = "solid_displacement", .space = vector_space, .components = 2});
    const auto v_s_id = system.addField({.name = "solid_velocity", .space = vector_space, .components = 2});

    const auto u_f = forms::StateField(u_f_id, *vector_space, "u_f").minus();
    const auto w_f = forms::TestField(u_f_id, *vector_space, "w_f").minus();
    const auto p_f = forms::StateField(p_f_id, *scalar_space, "p_f").minus();
    const auto v_s = forms::StateField(v_s_id, *vector_space, "v_s").plus();
    const auto w_s = forms::TestField(d_s_id, *vector_space, "w_s").plus();
    const auto n_f = forms::FormExpr::normal().minus();

    FE::systems::FormInstallOptions options;
    if (form == InterfaceForm::VelocityContinuity) {
        options.extra_trial_fields = {v_s_id};
        (void)FE::systems::installFormulation(system, "equations", {u_f_id},
                                              forms::inner(u_f - v_s, w_f).dI(kInterfaceMarker), options);
    } else {
        auto traction = -p_f * n_f;
        if (mu != 0.0) {
            const auto strain_rate = forms::sym(forms::grad(u_f));
            traction = traction + forms::FormExpr::constant(2.0) * forms::FormExpr::constant(mu) * strain_rate * n_f;
        }
        options.extra_trial_fields = {u_f_id, p_f_id};
        (void)FE::systems::installFormulation(system, "equations", {d_s_id},
                                              forms::inner(traction, w_s).dI(kInterfaceMarker), options);
    }
    // Interface terms alone leave null spaces; evaluate the unconstrained residual.
    FE::systems::SetupOptions setup_options;
    setup_options.use_constraints_in_assembly = false;
    system.setup(setup_options);

    const auto n = system.dofHandler().getNumDofs();
    std::vector<FE::Real> state_values(static_cast<std::size_t>(n), 0.0);
    const auto& X = mesh->local_mesh().X_ref();
    const auto set = [&](FE::FieldId field, FE::GlobalIndex vertex, int component, FE::Real value) {
        const auto dofs = system.fieldDofHandler(field).getEntityDofMap()->getVertexDofs(vertex);
        state_values[static_cast<std::size_t>(system.fieldDofOffset(field) + dofs[static_cast<std::size_t>(component)])] = value;
    };
    for (FE::GlobalIndex vertex = 0; vertex < 6; ++vertex) {
        set(u_f_id, vertex, 1, a * X[static_cast<std::size_t>(2 * vertex)]);
        set(p_f_id, vertex, 0, p0);
        set(v_s_id, vertex, 1, a + jump);
    }
    FE::systems::SystemStateView state;
    state.u = state_values;
    FE::systems::AssemblyRequest request;
    request.op = "equations";
    request.want_vector = true;
    FE::assembly::DenseVectorView R(n);
    R.zero();
    const auto result = system.assemble(request, state, nullptr, &R);
    if (!result.success) {
        throw std::runtime_error(result.error_message);
    }

    const auto component_sum = [&](FE::FieldId field, int component) {
        FE::Real sum = 0.0;
        for (FE::GlobalIndex vertex = 0; vertex < 6; ++vertex) {
            const auto dofs = system.fieldDofHandler(field).getEntityDofMap()->getVertexDofs(vertex);
            sum += R.getVectorEntry(system.fieldDofOffset(field) + dofs[static_cast<std::size_t>(component)]);
        }
        return sum;
    };
    return {component_sum(u_f_id, 0), component_sum(u_f_id, 1), component_sum(d_s_id, 0), component_sum(d_s_id, 1)};
}

} // namespace

TEST(FSIInterfaceForms, AreAUnitWeightRobinPenaltyAndAnUnassembledTraction)
{
    constexpr FE::Real mu = 0.25, a = 2.0, p0 = 3.0;
    const auto matched = interfaceResidualSums(InterfaceForm::VelocityContinuity, mu, a, p0, /*jump=*/0.0);
    // A consistent coupling gives the fluid rows -t_f = (p0, -mu a); the form gives nothing.
    EXPECT_NEAR(matched.fluid_x, 0.0, 1e-12);
    EXPECT_NEAR(matched.fluid_y, 0.0, 1e-12);
    for (const FE::Real viscosity : {mu, 100.0 * mu}) {
        SCOPED_TRACE(::testing::Message() << "mu = " << viscosity);
        const auto jumped = interfaceResidualSums(InterfaceForm::VelocityContinuity, viscosity, a, p0, /*jump=*/0.5);
        EXPECT_NEAR(jumped.fluid_x, 0.0, 1e-12);
        EXPECT_NEAR(jumped.fluid_y, -0.5, 1e-12);   // int (u_f - v_s).e_y over the unit face
        EXPECT_NEAR(jumped.solid_x, 0.0, 1e-12);
        EXPECT_NEAR(jumped.solid_y, 0.0, 1e-12);
    }
    std::string refusal;
    try {
        (void)interfaceResidualSums(InterfaceForm::TractionBalance, mu, a, p0, 0.0);
    } catch (const std::exception& error) {
        refusal = error.what();
    }
    // The interpreter's derivative of the interface-restricted velocity is not implemented.
    EXPECT_NE(refusal.find("Not implemented"), std::string::npos) << refusal;
    // The pressure part alone assembles: the solid rows get int -p0 n_f . w_s.
    const auto pressure_only = interfaceResidualSums(InterfaceForm::TractionBalance, 0.0, a, p0, 0.0);
    EXPECT_NEAR(pressure_only.solid_x, -p0, 1e-12);
    EXPECT_NEAR(pressure_only.solid_y, 0.0, 1e-12);
}

#endif
