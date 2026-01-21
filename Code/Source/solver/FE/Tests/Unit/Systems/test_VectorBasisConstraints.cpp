/**
 * @file test_VectorBasisConstraints.cpp
 * @brief Unit tests for Systems-side essential constraints on H(curl)/H(div) vector bases
 */

#include <gtest/gtest.h>

#include "Basis/VectorBasis.h"
#include "Elements/ReferenceElement.h"
#include "Systems/FESystem.h"
#include "Systems/HCurlTangentialConstraint.h"
#include "Systems/HDivNormalConstraint.h"
#include "Spaces/HCurlSpace.h"
#include "Spaces/HDivSpace.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <algorithm>
#include <unordered_set>

namespace svmp {
namespace FE {
namespace systems {
namespace test {

namespace {

dofs::MeshTopologyInfo singleTetraTopology()
{
    dofs::MeshTopologyInfo topo;
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

[[nodiscard]] std::vector<int> localEdgesOnFace(const elements::ReferenceElement& ref, LocalIndex local_face_id)
{
    std::vector<int> edges;
    const auto& fn = ref.face_nodes(static_cast<std::size_t>(local_face_id));
    if (fn.size() < 2u) {
        return edges;
    }

    auto on_face = [&](LocalIndex v) -> bool {
        for (auto fv : fn) {
            if (fv == v) return true;
        }
        return false;
    };

    edges.reserve(ref.num_edges());
    for (std::size_t e = 0; e < ref.num_edges(); ++e) {
        const auto& en = ref.edge_nodes(e);
        if (en.size() != 2u) continue;
        if (on_face(en[0]) && on_face(en[1])) {
            edges.push_back(static_cast<int>(e));
        }
    }
    return edges;
}

template <class SpaceT>
std::unordered_set<GlobalIndex> expectedBoundaryDofsForMarker(
    const FESystem& sys,
    FieldId field,
    const SpaceT& space,
    int marker)
{
    const auto* vbf = dynamic_cast<const basis::VectorBasisFunction*>(&space.element().basis());
    EXPECT_NE(vbf, nullptr);

    const auto assoc = vbf->dof_associations();
    const auto ref = elements::ReferenceElement::create(ElementType::Tetra4);

    std::unordered_set<GlobalIndex> expected;
    const auto offset = sys.fieldDofOffset(field);
    const auto cell_dofs = sys.fieldDofHandler(field).getDofMap().getCellDofs(0);

    sys.meshAccess().forEachBoundaryFace(marker, [&](GlobalIndex face_id, GlobalIndex cell_id) {
        (void)face_id;
        ASSERT_EQ(cell_id, 0);
        const auto local_face = sys.meshAccess().getLocalFaceIndex(0, 0);
        const auto face_edges = localEdgesOnFace(ref, local_face);

        auto is_face_edge = [&](int edge_id) -> bool {
            for (int e : face_edges) {
                if (e == edge_id) return true;
            }
            return false;
        };

        for (std::size_t ldof = 0; ldof < assoc.size(); ++ldof) {
            const auto& a = assoc[ldof];
            bool constrain = false;
            if (space.continuity() == Continuity::H_curl) {
                if (a.entity_type == basis::DofEntity::Edge) {
                    constrain = is_face_edge(a.entity_id);
                } else if (a.entity_type == basis::DofEntity::Face) {
                    constrain = (a.entity_id == static_cast<int>(local_face));
                }
            } else if (space.continuity() == Continuity::H_div) {
                if (a.entity_type == basis::DofEntity::Face) {
                    constrain = (a.entity_id == static_cast<int>(local_face));
                }
            }
            if (!constrain) continue;
            expected.insert(cell_dofs[ldof] + offset);
        }
    });

    return expected;
}

} // namespace

TEST(VectorBasisConstraints, HCurlTangentialConstrainsBoundaryFacetDofs)
{
    const int marker = 7;
    auto mesh = std::make_shared<forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<spaces::HCurlSpace>(ElementType::Tetra4, /*order=*/0);

    FESystem sys(mesh);
    const auto E = sys.addField(FieldSpec{.name = "E", .space = space, .components = space->value_dimension()});
    sys.addSystemConstraint(std::make_unique<HCurlTangentialConstraint>(E, marker));

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const auto expected = expectedBoundaryDofsForMarker(sys, E, *space, marker);
    EXPECT_FALSE(expected.empty());

    for (GlobalIndex dof = 0; dof < sys.dofHandler().getNumDofs(); ++dof) {
        const bool should = (expected.find(dof) != expected.end());
        EXPECT_EQ(sys.constraints().isConstrained(dof), should);
    }
}

TEST(VectorBasisConstraints, HDivNormalConstrainsBoundaryFacetDofs)
{
    const int marker = 7;
    auto mesh = std::make_shared<forms::test::SingleTetraOneBoundaryFaceMeshAccess>(marker);
    auto space = std::make_shared<spaces::HDivSpace>(ElementType::Tetra4, /*order=*/0);

    FESystem sys(mesh);
    const auto B = sys.addField(FieldSpec{.name = "B", .space = space, .components = space->value_dimension()});
    sys.addSystemConstraint(std::make_unique<HDivNormalConstraint>(B, marker));

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const auto expected = expectedBoundaryDofsForMarker(sys, B, *space, marker);
    EXPECT_FALSE(expected.empty());

    for (GlobalIndex dof = 0; dof < sys.dofHandler().getNumDofs(); ++dof) {
        const bool should = (expected.find(dof) != expected.end());
        EXPECT_EQ(sys.constraints().isConstrained(dof), should);
    }
}

TEST(VectorBasisConstraints, MonolithicDofHandlerPreservesCellOrientationsForVectorBasisFields)
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::HCurlSpace>(ElementType::Tetra4, /*order=*/0);

    FESystem sys(mesh);
    (void)sys.addField(FieldSpec{.name = "E", .space = space, .components = space->value_dimension()});
    sys.addOperator("op");

    SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    EXPECT_TRUE(sys.dofHandler().hasCellOrientations());
}

} // namespace test
} // namespace systems
} // namespace FE
} // namespace svmp
