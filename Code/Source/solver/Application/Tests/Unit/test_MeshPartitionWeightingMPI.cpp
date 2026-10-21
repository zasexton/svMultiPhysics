// Startup partition weighting (<Partition_weighting>free_surface</Partition_weighting>)
// on 2+ ranks: without weights the partition is the plain one, weighted
// loads are deterministic and leave no field behind, and the weights balance
// the cut-volume work of a liquid region that the unweighted partition puts
// on one rank.

#include <gtest/gtest.h>

#include "Application/Translators/MeshTranslator.h"
#include "Mesh/Core/StartupPartitionWeights.h"
#include "Mesh/Mesh.h"
#include "Parameters.h"

#ifdef MESH_HAS_VTK
#include "Mesh/IO/VTKWriter.h"
#endif

#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

int worldRank()
{
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

int worldSize()
{
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size;
}

bool allRanks(bool local)
{
  int value = local ? 1 : 0;
  int result = 0;
  MPI_Allreduce(&value, &result, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  return result == 1;
}

bool anyRank(bool local)
{
  int value = local ? 1 : 0;
  int result = 0;
  MPI_Allreduce(&value, &result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  return result == 1;
}

// Box of kNx x kNy x kNz unit cubes, six tetrahedra each, with the level set
// phi = x + 0.15 y - 4.3: the liquid (phi < 0) fills the x < ~4 end, a
// quarter of the box, so a cell-count partition of the long box leaves all
// of it on one or two ranks.
constexpr int kNx = 16;
constexpr int kNy = 6;
constexpr int kNz = 6;

double fixturePhi(double x, double y)
{
  return x + 0.15 * y - 4.3;
}

#ifdef MESH_HAS_VTK
void writeBoxFixture(const std::filesystem::path& path)
{
  const auto vid = [](int i, int j, int k) {
    return static_cast<svmp::index_t>((i * (kNy + 1) + j) * (kNz + 1) + k);
  };
  std::vector<svmp::real_t> x;
  for (int i = 0; i <= kNx; ++i) {
    for (int j = 0; j <= kNy; ++j) {
      for (int k = 0; k <= kNz; ++k) {
        x.insert(x.end(), {static_cast<svmp::real_t>(i), static_cast<svmp::real_t>(j),
                           static_cast<svmp::real_t>(k)});
      }
    }
  }
  // Kuhn split of a cube (corner b = x bit 0, y bit 1, z bit 2); each
  // tetrahedron is put in positive orientation.
  constexpr std::array<std::array<int, 4>, 6> kTets = {{
      {0, 1, 3, 7}, {0, 1, 5, 7}, {0, 2, 3, 7}, {0, 2, 6, 7}, {0, 4, 5, 7}, {0, 4, 6, 7}}};
  std::vector<svmp::offset_t> offsets = {0};
  std::vector<svmp::index_t> conn;
  for (int i = 0; i < kNx; ++i) {
    for (int j = 0; j < kNy; ++j) {
      for (int k = 0; k < kNz; ++k) {
        std::array<svmp::index_t, 8> corner{};
        for (int b = 0; b < 8; ++b) {
          corner[static_cast<std::size_t>(b)] = vid(i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1));
        }
        for (const auto& t : kTets) {
          std::array<svmp::index_t, 4> v{};
          for (int a = 0; a < 4; ++a) {
            v[static_cast<std::size_t>(a)] = corner[static_cast<std::size_t>(t[static_cast<std::size_t>(a)])];
          }
          const auto p = [&](int a, int d) {
            return x[static_cast<std::size_t>(v[static_cast<std::size_t>(a)]) * 3u + static_cast<std::size_t>(d)];
          };
          double e[3][3];
          for (int a = 0; a < 3; ++a) {
            for (int d = 0; d < 3; ++d) {
              e[a][d] = p(a + 1, d) - p(0, d);
            }
          }
          const double det = e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) -
                             e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
                             e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]);
          if (det < 0.0) {
            std::swap(v[2], v[3]);
          }
          conn.insert(conn.end(), v.begin(), v.end());
          offsets.push_back(static_cast<svmp::offset_t>(conn.size()));
        }
      }
    }
  }
  svmp::CellShape tet{};
  tet.family = svmp::CellFamily::Tetra;
  tet.num_corners = 4;
  tet.order = 1;
  std::vector<svmp::CellShape> shapes(offsets.size() - 1u, tet);

  svmp::MeshBase volume;
  volume.build_from_arrays(3, x, offsets, conn, shapes);
  std::vector<svmp::gid_t> vertex_gids(volume.n_vertices());
  for (std::size_t v = 0; v < vertex_gids.size(); ++v) {
    vertex_gids[v] = static_cast<svmp::gid_t>(v);
  }
  std::vector<svmp::gid_t> cell_gids(volume.n_cells());
  for (std::size_t c = 0; c < cell_gids.size(); ++c) {
    cell_gids[c] = static_cast<svmp::gid_t>(c);
  }
  volume.set_vertex_gids(std::move(vertex_gids));
  volume.set_cell_gids(std::move(cell_gids));
  const auto phi_handle = volume.attach_field(svmp::EntityKind::Vertex, "phi", svmp::FieldScalarType::Float64, 1);
  auto* phi = volume.field_data_as<double>(phi_handle);
  for (std::size_t v = 0; v < volume.n_vertices(); ++v) {
    phi[v] = fixturePhi(x[3u * v], x[3u * v + 1u]);
  }
  volume.finalize();

  svmp::MeshIOOptions options{};
  options.path = path.string();
  options.format = "vtu";
  options.kv["codim1_topology"] = "none";
  options.kv["edge_topology"] = "false";
  svmp::VTKWriter::write(volume, options);
}
#endif

// Sorted global IDs of this rank's owned cells.
std::vector<svmp::gid_t> ownedCellGids(const svmp::Mesh& mesh)
{
  std::vector<svmp::gid_t> gids;
  const auto& all = mesh.cell_gids();
  for (svmp::index_t c = 0; c < static_cast<svmp::index_t>(all.size()); ++c) {
    if (mesh.is_owned_cell(c)) {
      gids.push_back(all[static_cast<std::size_t>(c)]);
    }
  }
  std::sort(gids.begin(), gids.end());
  return gids;
}

struct Balance {
  double work_max_over_mean{0.0};
  double cells_max_over_mean{0.0};
};

// Cut-volume work of the owned cells (full liquid 1, cut 2, dry 0, from
// the loaded mesh's phi) and the owned cell count, max over mean.
Balance ownedBalance(const svmp::Mesh& mesh)
{
  const auto& base = mesh.local_mesh();
  const double* phi = nullptr;
  if (base.has_field(svmp::EntityKind::Vertex, "phi")) {
    phi = base.field_data_as<double>(base.field_handle(svmp::EntityKind::Vertex, "phi"));
  }
  double local[2] = {0.0, 0.0};
  for (svmp::index_t c = 0; phi != nullptr && c < static_cast<svmp::index_t>(base.n_cells()); ++c) {
    if (!mesh.is_owned_cell(c)) {
      continue;
    }
    auto [vptr, nv] = base.cell_vertices_span(c);
    bool below = true;
    bool above = true;
    for (std::size_t k = 0; k < nv; ++k) {
      below = below && phi[vptr[k]] < 0.0;
      above = above && phi[vptr[k]] > 0.0;
    }
    local[0] += below ? 1.0 : (above ? 0.0 : 2.0);
    local[1] += 1.0;
  }
  double max_values[2] = {0.0, 0.0};
  double sum_values[2] = {0.0, 0.0};
  MPI_Allreduce(local, max_values, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(local, sum_values, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  const double ranks = static_cast<double>(worldSize());
  if (sum_values[0] <= 0.0 || sum_values[1] <= 0.0) {
    ADD_FAILURE() << "no liquid work or no cells (phi missing?)";
    return {};
  }
  return {max_values[0] / (sum_values[0] / ranks), max_values[1] / (sum_values[1] / ranks)};
}

class MeshPartitionWeightingMPI : public ::testing::Test {
protected:
  static void SetUpTestSuite()
  {
    long long stamp = 0;
    if (worldRank() == 0) {
      stamp = static_cast<long long>(std::chrono::steady_clock::now().time_since_epoch().count());
    }
    MPI_Bcast(&stamp, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    path_ = std::filesystem::temp_directory_path() /
            ("svmp_partition_weighting_mpi_" + std::to_string(stamp) + ".vtu");
    int written = 1;
#ifdef MESH_HAS_VTK
    if (worldRank() == 0) {
      try {
        writeBoxFixture(path_);
      } catch (const std::exception&) {
        written = 0;
      }
    }
#else
    written = 0;
#endif
    MPI_Bcast(&written, 1, MPI_INT, 0, MPI_COMM_WORLD);
    written_ = written == 1;
    MPI_Barrier(MPI_COMM_WORLD);
  }

  static void TearDownTestSuite()
  {
    MPI_Barrier(MPI_COMM_WORLD);
    if (worldRank() == 0) {
      std::error_code ignored;
      std::filesystem::remove(path_, ignored);
    }
  }

  void SetUp() override
  {
    if (!written_) {
      GTEST_SKIP() << "VTK support is required to write the partition-weighting fixture.";
    }
    if (worldSize() < 2) {
      GTEST_SKIP() << "Partition weighting needs at least two ranks.";
    }
#if !defined(SVMP_HAS_PARMETIS)
    GTEST_SKIP() << "Partition weights act on the startup ParMETIS repartition (not built).";
#endif
  }

  // MeshParameters is neither copyable nor movable: fill the caller's.
  static void configure(MeshParameters& params)
  {
    params.name.set("box");
    params.mesh_file_path.set(path_.string());
  }

  // The weights SimulationBuilder passes for a LevelSetNegative free surface
  // on 'phi' at isovalue 0 with the default class weights.
  static application::translators::MeshPartitionWeights defaultWeights()
  {
    MeshParameters params;
    configure(params);
    params.partition_weighting.set("free_surface");
    return application::translators::MeshTranslator::freeSurfacePartitionWeights(params, "phi", 0.0, true);
  }

  static std::shared_ptr<svmp::Mesh> load(const application::translators::MeshPartitionWeights& weights)
  {
    MeshParameters params;
    configure(params);
    return application::translators::MeshTranslator::loadMesh(params, 0, weights);
  }

  static inline std::filesystem::path path_{};
  static inline bool written_{false};
};

TEST_F(MeshPartitionWeightingMPI, WithoutWeightsThePartitionIsThePlainStartupPartition)
{
  // Key absent (default parameters), key 'none', and a direct load with the
  // translator's options and no partition keys give the same owned cells.
  const auto absent = load({});
  MeshParameters none_params;
  configure(none_params);
  none_params.partition_weighting.set("none");
  ASSERT_FALSE(application::translators::MeshTranslator::freeSurfacePartitionWeightingRequested(none_params));
  const auto none = application::translators::MeshTranslator::loadMesh(none_params);

  svmp::MeshIOOptions options{};
  options.path = path_.string();
  options.format = "vtu";
  options.kv["codim1_topology"] = "none";
  options.kv["edge_topology"] = "false";
  options.kv["ghost_layers"] = "0";
  const auto direct = svmp::Mesh::load_parallel(options, svmp::MeshComm::world());

  ASSERT_NE(absent, nullptr);
  ASSERT_NE(none, nullptr);
  const auto reference = ownedCellGids(direct);
  EXPECT_TRUE(allRanks(ownedCellGids(*absent) == reference));
  EXPECT_TRUE(allRanks(ownedCellGids(*none) == reference));
  EXPECT_FALSE(absent->local_mesh().has_field(svmp::EntityKind::Volume, svmp::kStartupPartitionWeightField));
}

TEST_F(MeshPartitionWeightingMPI, WeightedLoadsAreDeterministicAndRemoveTheWeightField)
{
  const auto weights = defaultWeights();
  const auto first = load(weights);
  const auto second = load(weights);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  EXPECT_TRUE(allRanks(ownedCellGids(*first) == ownedCellGids(*second)));
  EXPECT_TRUE(allRanks(!first->local_mesh().has_field(svmp::EntityKind::Volume,
                                                      svmp::kStartupPartitionWeightField)));
  // The level set itself is kept.
  EXPECT_TRUE(allRanks(first->local_mesh().has_field(svmp::EntityKind::Vertex, "phi")));
}

TEST_F(MeshPartitionWeightingMPI, WeightsBalanceTheCutVolumeWork)
{
  const auto plain = load({});
  const auto weighted = load(defaultWeights());
  ASSERT_NE(plain, nullptr);
  ASSERT_NE(weighted, nullptr);

  const auto before = ownedBalance(*plain);
  const auto after = ownedBalance(*weighted);
  if (worldRank() == 0) {
    std::cout << "[partition weighting] ranks=" << worldSize()
              << " work max/mean unweighted=" << before.work_max_over_mean
              << " weighted=" << after.work_max_over_mean
              << " cells max/mean unweighted=" << before.cells_max_over_mean
              << " weighted=" << after.cells_max_over_mean << std::endl;
  }
  // The fixture puts the liquid at one end of a long box: the cell-count
  // partition is imbalanced in liquid work (about 2.0 / 2.8 / 3.1 at 2 / 3 /
  // 4 ranks), the weighted one is not (about 1.03), and the cell count stays
  // balanced (within 3%).
  EXPECT_GT(before.work_max_over_mean, 1.6);
  EXPECT_LT(after.work_max_over_mean, 1.35);
  EXPECT_LT(after.work_max_over_mean, 0.75 * before.work_max_over_mean);
  EXPECT_LT(after.cells_max_over_mean, 1.2);
  EXPECT_TRUE(anyRank(ownedCellGids(*plain) != ownedCellGids(*weighted)));
}

TEST_F(MeshPartitionWeightingMPI, AMissingLevelSetFieldFailsOnEveryRank)
{
  auto weights = defaultWeights();
  weights.vertex_field = "no_such_field";
  bool threw = false;
  try {
    (void)load(weights);
  } catch (const std::exception& error) {
    threw = std::string(error.what()).find("no_such_field") != std::string::npos ||
            std::string(error.what()).find("another rank") != std::string::npos;
  }
  EXPECT_TRUE(allRanks(threw));
}

} // namespace
