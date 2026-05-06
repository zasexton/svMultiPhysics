#include <gtest/gtest.h>

#include "Application/Core/MeshCollection.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Parameters.h"

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

std::shared_ptr<svmp::Mesh> makeMesh()
{
  return svmp::create_mesh(std::make_shared<svmp::MeshBase>());
}

std::shared_ptr<svmp::Mesh> makeMeshWithLabel(const std::string& name,
                                              svmp::label_t label)
{
  auto mesh = makeMesh();
  mesh->base().register_label(name, label);
  return mesh;
}

application::core::MeshParticipant makeParticipant(
    std::string name,
    std::optional<int> domain_id = std::nullopt,
    std::map<std::string, svmp::label_t> face_labels = {})
{
  application::core::MeshParticipant participant;
  participant.name = std::move(name);
  participant.domain_id = domain_id;
  participant.mesh = makeMesh();
  participant.face_labels = std::move(face_labels);
  return participant;
}

void expectRuntimeErrorContains(const std::function<void()>& fn,
                                const std::string& text)
{
  try {
    fn();
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find(text), std::string::npos)
        << error.what();
    return;
  }
  FAIL() << "Expected std::runtime_error containing: " << text;
}

} // namespace

TEST(MeshCollection, StoresParticipantRecordAndLookups)
{
  application::core::MeshCollection collection;
  collection.addParticipant(
      makeParticipant("lumen", 1, {{"lumen_wall", 10}}));
  collection.addParticipant(makeParticipant("wall", 2, {{"wall_inner", 20}}));

  ASSERT_EQ(collection.size(), 2u);
  ASSERT_NE(collection.participantByName("lumen"), nullptr);
  EXPECT_EQ(collection.participantByName("lumen")->domain_id, 1);
  EXPECT_EQ(collection.participantByDomain(2)->name, "wall");
  EXPECT_EQ(collection.participantOwningFace("wall_inner")->name, "wall");

  const auto face_owner = collection.faceOwner("lumen_wall");
  ASSERT_TRUE(face_owner.has_value());
  EXPECT_EQ(face_owner->participant_name, "lumen");
  EXPECT_EQ(face_owner->label, 10);

  const auto mesh_map = collection.asMeshMap();
  EXPECT_EQ(mesh_map.size(), 2u);
  EXPECT_TRUE(mesh_map.count("lumen"));
  EXPECT_TRUE(mesh_map.count("wall"));
}

TEST(MeshCollection, RejectsDuplicateMeshNames)
{
  application::core::MeshCollection collection;
  collection.addParticipant(makeParticipant("mesh", 1));

  expectRuntimeErrorContains(
      [&collection] { collection.addParticipant(makeParticipant("mesh", 2)); },
      "Duplicate <Add_mesh name=\"mesh\">");
}

TEST(MeshCollection, RejectsDuplicateDomainIds)
{
  application::core::MeshCollection collection;
  collection.addParticipant(makeParticipant("fluid", 7));

  expectRuntimeErrorContains(
      [&collection] { collection.addParticipant(makeParticipant("solid", 7)); },
      "Duplicate <Domain> value 7 for meshes 'fluid' and 'solid'");
}

TEST(MeshCollection, RejectsDuplicateFaceNamesAcrossParticipants)
{
  application::core::MeshCollection collection;
  collection.addParticipant(makeParticipant("fluid", 1, {{"interface", 11}}));

  expectRuntimeErrorContains(
      [&collection] {
        collection.addParticipant(
            makeParticipant("solid", 2, {{"interface", 22}}));
      },
      "Duplicate face name 'interface' across meshes 'fluid' and 'solid'");
}

TEST(MeshCollection, LooksUpParticipantsByDomainId)
{
  application::core::MeshCollection collection;
  collection.addParticipant(makeParticipant("fluid", 10));
  collection.addParticipant(makeParticipant("solid", 20));

  ASSERT_NE(collection.participantByDomain(10), nullptr);
  EXPECT_EQ(collection.participantByDomain(10)->name, "fluid");
  ASSERT_NE(collection.participantByDomain(20), nullptr);
  EXPECT_EQ(collection.participantByDomain(20)->name, "solid");
  EXPECT_EQ(collection.participantByDomain(30), nullptr);
}

TEST(MeshCollection, RequiresDomainIdsWhenRequested)
{
  application::core::MeshCollection collection;
  collection.addParticipant(makeParticipant("fluid"));

  expectRuntimeErrorContains(
      [&collection] {
        collection.validateDomainIdsRequired("Equation translation");
      },
      "Equation translation requires <Add_mesh name=\"fluid\"> to define <Domain>");
}

TEST(MeshParticipant, CapturesLoadedMeshMetadataAndFaceLabels)
{
  MeshParameters parameters;
  parameters.name.set("lumen");
  parameters.domain_id.set("3");
  auto* face = new FaceParameters();
  face->name.set("lumen_wall");
  parameters.face_parameters.push_back(face);

  auto mesh = makeMeshWithLabel("lumen_wall", 42);

  const auto participant =
      application::core::MeshParticipant::fromLoadedMesh(parameters, mesh);

  EXPECT_EQ(participant.name, "lumen");
  EXPECT_EQ(participant.domain_id, 3);
  EXPECT_EQ(participant.mesh, mesh);
  ASSERT_EQ(participant.face_labels.size(), 1u);
  EXPECT_EQ(participant.face_labels.at("lumen_wall"), 42);
  EXPECT_EQ(participant.source_parameters, &parameters);
}

TEST(MeshParticipant, RejectsFaceMissingFromLoadedMesh)
{
  MeshParameters parameters;
  parameters.name.set("lumen");
  auto* face = new FaceParameters();
  face->name.set("missing_wall");
  parameters.face_parameters.push_back(face);

  expectRuntimeErrorContains(
      [&parameters] {
        (void)application::core::MeshParticipant::fromLoadedMesh(parameters,
                                                                 makeMesh());
      },
      "<Add_mesh name=\"lumen\"> face 'missing_wall'");
}
