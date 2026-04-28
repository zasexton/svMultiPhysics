/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "CutCell.h"

#include "../Core/DistributedMesh.h"
#include "../Core/MeshBase.h"
#include "../Geometry/CurvilinearEval.h"
#include "../Geometry/MeshGeometry.h"
#include "../Geometry/PolyGeometry.h"
#include "../Geometry/PolyhedronTessellation.h"
#include "../Geometry/Tessellation.h"
#include "../Topology/CellTopology.h"

#ifdef MESH_HAS_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace svmp {
namespace search {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;
constexpr const char* kTrueCurvedArrangementPolicy = "true-curved-isoparametric-arrangement";
constexpr const char* kTrueCurvedSubdivisionArrangementPolicy = "true-curved-subdivision-arrangement";

std::uint64_t append_hash(std::uint64_t h, std::uint64_t value) noexcept {
  h ^= value;
  h *= kFnvPrime;
  return h;
}

std::uint64_t append_hash_string(std::uint64_t h, const std::string& value) noexcept {
  for (const unsigned char c : value) {
    h ^= static_cast<std::uint64_t>(c);
    h *= kFnvPrime;
  }
  return h;
}

std::uint64_t quantized_real(real_t value, real_t scale = 1.0e12) noexcept {
  if (!std::isfinite(value)) {
    return 0;
  }
  const auto q = static_cast<std::int64_t>(std::llround(value * scale));
  return static_cast<std::uint64_t>(q);
}

std::uint64_t append_hash_real(std::uint64_t h, real_t value) noexcept {
  return append_hash(h, quantized_real(value));
}

std::uint64_t append_hash_distributed_record(std::uint64_t h,
                                             const CutDistributedEntityRecord& entity) noexcept {
  h = append_hash(h, entity.stable_id);
  h = append_hash(h, entity.cut_topology_id);
  h = append_hash(h, static_cast<std::uint64_t>(entity.kind));
  h = append_hash(h, static_cast<std::uint64_t>(entity.side));
  h = append_hash(h, static_cast<std::uint64_t>(entity.parent_gid));
  h = append_hash(h, static_cast<std::uint64_t>(entity.owner_rank));
  h = append_hash_string(h, entity.provenance_id);
  h = append_hash(h, static_cast<std::uint64_t>(entity.integration_family));
  h = append_hash_real(h, entity.parent_measure);
  h = append_hash_real(h, entity.measure);
  h = append_hash_real(h, entity.volume_fraction);
  h = append_hash(h, entity.closed_topology ? 1u : 0u);
  h = append_hash_real(h, entity.point[0]);
  h = append_hash_real(h, entity.point[1]);
  h = append_hash_real(h, entity.point[2]);
  h = append_hash_real(h, entity.normal[0]);
  h = append_hash_real(h, entity.normal[1]);
  h = append_hash_real(h, entity.normal[2]);
  h = append_hash_real(h, entity.centroid[0]);
  h = append_hash_real(h, entity.centroid[1]);
  h = append_hash_real(h, entity.centroid[2]);
  for (const auto id : entity.vertex_ids) {
    h = append_hash(h, id);
  }
  for (const auto& face : entity.face_ids) {
    h = append_hash(h, static_cast<std::uint64_t>(face.size()));
    for (const auto id : face) {
      h = append_hash(h, id);
    }
  }
  return h;
}

std::string distributed_entity_key(const CutDistributedEntityRecord& entity) {
  return std::to_string(entity.stable_id) + ":" +
         std::to_string(static_cast<long long>(entity.parent_gid)) + ":" +
         std::to_string(static_cast<int>(entity.kind)) + ":" +
         std::to_string(static_cast<int>(entity.side));
}

void write_u64_vector(std::ostream& out, const std::vector<std::uint64_t>& values) {
  out << values.size();
  for (const auto value : values) {
    out << ' ' << value;
  }
}

void read_u64_vector(std::istream& in, std::vector<std::uint64_t>& values) {
  std::size_t count = 0;
  in >> count;
  values.resize(count);
  for (auto& value : values) {
    in >> value;
  }
}

void write_face_vector(std::ostream& out,
                       const std::vector<std::vector<std::uint64_t>>& faces) {
  out << faces.size();
  for (const auto& face : faces) {
    out << ' ';
    write_u64_vector(out, face);
  }
}

void read_face_vector(std::istream& in,
                      std::vector<std::vector<std::uint64_t>>& faces) {
  std::size_t count = 0;
  in >> count;
  faces.resize(count);
  for (auto& face : faces) {
    read_u64_vector(in, face);
  }
}

std::string serialize_cut_exchange_packet(const CutDistributedExchangePacket& packet) {
  std::ostringstream out;
  out << std::setprecision(17);
  out << "SVMP_CUT_PACKET 1\n";
  out << packet.revision_key << ' ' << packet.entities.size() << ' '
      << packet.diagnostics.size() << '\n';
  for (const auto& entity : packet.entities) {
    out << entity.stable_id << ' '
        << entity.cut_topology_id << ' '
        << static_cast<int>(entity.kind) << ' '
        << static_cast<int>(entity.side) << ' '
        << entity.parent_entity << ' '
        << entity.parent_gid << ' '
        << entity.owner_rank << ' '
        << std::quoted(entity.provenance_id) << ' '
        << entity.point[0] << ' ' << entity.point[1] << ' ' << entity.point[2] << ' '
        << entity.normal[0] << ' ' << entity.normal[1] << ' ' << entity.normal[2] << ' '
        << entity.centroid[0] << ' ' << entity.centroid[1] << ' ' << entity.centroid[2] << ' '
        << static_cast<int>(entity.integration_family) << ' '
        << entity.parent_measure << ' '
        << entity.measure << ' '
        << entity.volume_fraction << ' '
        << (entity.closed_topology ? 1 : 0) << ' ';
    write_u64_vector(out, entity.vertex_ids);
    out << ' ';
    write_face_vector(out, entity.face_ids);
    out << '\n';
  }
  for (const auto& diagnostic : packet.diagnostics) {
    out << std::quoted(diagnostic) << '\n';
  }
  return out.str();
}

CutDistributedExchangePacket deserialize_cut_exchange_packet(const std::string& text) {
  CutDistributedExchangePacket packet;
  if (text.empty()) {
    return packet;
  }
  std::istringstream in(text);
  std::string magic;
  int version = 0;
  in >> magic >> version;
  if (magic != "SVMP_CUT_PACKET" || version != 1) {
    packet.diagnostics.push_back("invalid distributed cut exchange packet header");
    return packet;
  }
  std::size_t entity_count = 0;
  std::size_t diagnostic_count = 0;
  in >> packet.revision_key >> entity_count >> diagnostic_count;
  packet.entities.reserve(entity_count);
  for (std::size_t i = 0; i < entity_count; ++i) {
    CutDistributedEntityRecord entity;
    int kind = 0;
    int side = 0;
    int family = 0;
    int closed = 0;
    in >> entity.stable_id
       >> entity.cut_topology_id
       >> kind
       >> side
       >> entity.parent_entity
       >> entity.parent_gid
       >> entity.owner_rank
       >> std::quoted(entity.provenance_id)
       >> entity.point[0] >> entity.point[1] >> entity.point[2]
       >> entity.normal[0] >> entity.normal[1] >> entity.normal[2]
       >> entity.centroid[0] >> entity.centroid[1] >> entity.centroid[2]
       >> family
       >> entity.parent_measure
       >> entity.measure
       >> entity.volume_fraction
       >> closed;
    entity.kind = static_cast<CutTopologyEntityKind>(kind);
    entity.side = static_cast<CutTopologySide>(side);
    entity.integration_family = static_cast<CellFamily>(family);
    entity.closed_topology = closed != 0;
    read_u64_vector(in, entity.vertex_ids);
    read_face_vector(in, entity.face_ids);
    packet.entities.push_back(std::move(entity));
  }
  packet.diagnostics.reserve(diagnostic_count);
  for (std::size_t i = 0; i < diagnostic_count; ++i) {
    std::string message;
    in >> std::quoted(message);
    packet.diagnostics.push_back(std::move(message));
  }
  return packet;
}

std::string serialize_gid_requests(const std::vector<gid_t>& gids) {
  if (gids.empty()) {
    return {};
  }
  std::ostringstream out;
  out << "SVMP_CUT_GID_REQUEST 1 " << gids.size();
  for (const auto gid : gids) {
    out << ' ' << gid;
  }
  return out.str();
}

std::vector<gid_t> deserialize_gid_requests(const std::string& text) {
  std::vector<gid_t> gids;
  if (text.empty()) {
    return gids;
  }
  std::istringstream in(text);
  std::string magic;
  int version = 0;
  std::size_t count = 0;
  in >> magic >> version >> count;
  if (magic != "SVMP_CUT_GID_REQUEST" || version != 1) {
    return gids;
  }
  gids.resize(count);
  for (auto& gid : gids) {
    in >> gid;
  }
  return gids;
}

CutDistributedExchangePacket packet_for_owner(
    const CutDistributedExchangePacket& packet,
    rank_t owner_rank) {
  CutDistributedExchangePacket out;
  out.revision_key = packet.revision_key;
  out.diagnostics = packet.diagnostics;
  for (const auto& entity : packet.entities) {
    if (entity.owner_rank == owner_rank) {
      out.entities.push_back(entity);
    }
  }
  return deduplicate_cut_exchange_packet(std::move(out));
}

CutDistributedExchangePacket packet_for_parent_gids(
    const CutDistributedExchangePacket& packet,
    const std::vector<gid_t>& parent_gids,
    rank_t owner_rank) {
  CutDistributedExchangePacket out;
  out.revision_key = packet.revision_key;
  if (parent_gids.empty()) {
    return deduplicate_cut_exchange_packet(std::move(out));
  }
  std::set<gid_t> requested(parent_gids.begin(), parent_gids.end());
  out.diagnostics = packet.diagnostics;
  for (const auto& entity : packet.entities) {
    if (entity.owner_rank == owner_rank &&
        entity.parent_gid != INVALID_GID &&
        requested.find(entity.parent_gid) != requested.end()) {
      out.entities.push_back(entity);
    }
  }
  return deduplicate_cut_exchange_packet(std::move(out));
}

#ifdef MESH_HAS_MPI
std::vector<std::string> exchange_neighbor_strings(
    MPI_Comm comm,
    const std::vector<rank_t>& neighbors,
    const std::vector<std::string>& send_payloads,
    int count_tag,
    int payload_tag) {
  std::vector<std::string> received(neighbors.size());
  if (neighbors.empty()) {
    return received;
  }
  std::vector<int> send_counts(neighbors.size(), 0);
  std::vector<int> recv_counts(neighbors.size(), 0);
  std::vector<MPI_Request> requests;
  requests.reserve(neighbors.size() * 2u);
  for (std::size_t i = 0; i < neighbors.size(); ++i) {
    send_counts[i] = static_cast<int>(send_payloads[i].size());
    MPI_Request req = MPI_REQUEST_NULL;
    MPI_Irecv(&recv_counts[i], 1, MPI_INT, neighbors[i], count_tag, comm, &req);
    requests.push_back(req);
  }
  for (std::size_t i = 0; i < neighbors.size(); ++i) {
    MPI_Request req = MPI_REQUEST_NULL;
    MPI_Isend(&send_counts[i], 1, MPI_INT, neighbors[i], count_tag, comm, &req);
    requests.push_back(req);
  }
  MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

  std::vector<std::vector<char>> recv_buffers(neighbors.size());
  requests.clear();
  for (std::size_t i = 0; i < neighbors.size(); ++i) {
    if (recv_counts[i] <= 0) {
      continue;
    }
    recv_buffers[i].resize(static_cast<std::size_t>(recv_counts[i]));
    MPI_Request req = MPI_REQUEST_NULL;
    MPI_Irecv(recv_buffers[i].data(),
              recv_counts[i],
              MPI_CHAR,
              neighbors[i],
              payload_tag,
              comm,
              &req);
    requests.push_back(req);
  }
  for (std::size_t i = 0; i < neighbors.size(); ++i) {
    if (send_counts[i] <= 0) {
      continue;
    }
    MPI_Request req = MPI_REQUEST_NULL;
    MPI_Isend(send_payloads[i].data(),
              send_counts[i],
              MPI_CHAR,
              neighbors[i],
              payload_tag,
              comm,
              &req);
    requests.push_back(req);
  }
  if (!requests.empty()) {
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
  }
  for (std::size_t i = 0; i < neighbors.size(); ++i) {
    if (!recv_buffers[i].empty()) {
      received[i].assign(recv_buffers[i].begin(), recv_buffers[i].end());
    }
  }
  return received;
}
#endif

struct CutDistributedExchangeResult {
  std::vector<CutDistributedExchangePacket> packets{};
  bool neighbor_sparse_exchange{false};
  std::vector<rank_t> communication_neighbors{};
  std::vector<rank_t> received_neighbor_ranks{};
};

CutDistributedExchangeResult gather_cut_exchange_packets(
    const DistributedMesh& mesh,
    const CutDistributedExchangePacket& local_packet) {
  CutDistributedExchangeResult result;
#ifdef MESH_HAS_MPI
  MPI_Comm comm = mesh.mpi_comm();
  if (comm == MPI_COMM_NULL) {
    result.packets.push_back(local_packet);
    return result;
  }
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    result.packets.push_back(local_packet);
    return result;
  }
  int size = 1;
  MPI_Comm_size(comm, &size);
  if (size <= 1) {
    result.packets.push_back(local_packet);
    return result;
  }

  result.neighbor_sparse_exchange = true;
  const rank_t my_rank = mesh.rank();
  std::set<rank_t> local_neighbors;
  for (const auto rank : mesh.neighbor_ranks()) {
    if (rank >= 0 && rank < size && rank != my_rank) {
      local_neighbors.insert(rank);
    }
  }
  for (const auto& entity : local_packet.entities) {
    if (entity.owner_rank >= 0 &&
        entity.owner_rank < size &&
        entity.owner_rank != my_rank) {
      local_neighbors.insert(entity.owner_rank);
    }
  }

  result.communication_neighbors.assign(local_neighbors.begin(), local_neighbors.end());

  std::map<rank_t, std::vector<gid_t>> requested_parent_gids;
  for (const auto& entity : local_packet.entities) {
    if (entity.owner_rank == my_rank ||
        entity.owner_rank < 0 ||
        entity.owner_rank >= size ||
        entity.parent_gid == INVALID_GID) {
      continue;
    }
    const index_t local_cell = mesh.global_to_local_cell(entity.parent_gid);
    if (local_cell == INVALID_INDEX || mesh.is_owned_cell(local_cell)) {
      continue;
    }
    requested_parent_gids[entity.owner_rank].push_back(entity.parent_gid);
  }
  for (auto& [rank, gids] : requested_parent_gids) {
    (void)rank;
    std::sort(gids.begin(), gids.end());
    gids.erase(std::unique(gids.begin(), gids.end()), gids.end());
  }

  std::vector<std::string> request_payloads(result.communication_neighbors.size());
  for (std::size_t i = 0; i < result.communication_neighbors.size(); ++i) {
    const auto it = requested_parent_gids.find(result.communication_neighbors[i]);
    if (it != requested_parent_gids.end()) {
      request_payloads[i] = serialize_gid_requests(it->second);
    }
  }
  const auto received_requests =
      exchange_neighbor_strings(comm,
                                result.communication_neighbors,
                                request_payloads,
                                6200,
                                6201);

  std::vector<std::string> response_payloads(result.communication_neighbors.size());
  for (std::size_t i = 0; i < result.communication_neighbors.size(); ++i) {
    const auto gids = deserialize_gid_requests(received_requests[i]);
    if (gids.empty()) {
      continue;
    }
    const auto response = packet_for_parent_gids(local_packet, gids, my_rank);
    if (!response.entities.empty() || !response.diagnostics.empty()) {
      response_payloads[i] = serialize_cut_exchange_packet(response);
    }
  }
  const auto received_responses =
      exchange_neighbor_strings(comm,
                                result.communication_neighbors,
                                response_payloads,
                                6202,
                                6203);

  result.packets.push_back(packet_for_owner(local_packet, my_rank));
  for (std::size_t i = 0; i < received_responses.size(); ++i) {
    if (received_responses[i].empty()) {
      continue;
    }
    auto packet = deserialize_cut_exchange_packet(received_responses[i]);
    if (!packet.entities.empty()) {
      result.received_neighbor_ranks.push_back(result.communication_neighbors[i]);
    }
    result.packets.push_back(std::move(packet));
  }
  return result;
#else
  (void)mesh;
  result.packets.push_back(local_packet);
  return result;
#endif
}

Configuration normalized_configuration(Configuration cfg) noexcept {
  return cfg == Configuration::Deformed ? Configuration::Current : cfg;
}

std::array<real_t, 3> add(const std::array<real_t, 3>& a,
                          const std::array<real_t, 3>& b) noexcept {
  return {{a[0] + b[0], a[1] + b[1], a[2] + b[2]}};
}

std::array<real_t, 3> sub(const std::array<real_t, 3>& a,
                          const std::array<real_t, 3>& b) noexcept {
  return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
}

std::array<real_t, 3> scale(const std::array<real_t, 3>& a, real_t s) noexcept {
  return {{s * a[0], s * a[1], s * a[2]}};
}

real_t dot(const std::array<real_t, 3>& a,
           const std::array<real_t, 3>& b) noexcept {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

real_t norm(const std::array<real_t, 3>& a) noexcept {
  return std::sqrt(dot(a, a));
}

std::array<real_t, 3> cross(const std::array<real_t, 3>& a,
                            const std::array<real_t, 3>& b) noexcept {
  return {{a[1] * b[2] - a[2] * b[1],
           a[2] * b[0] - a[0] * b[2],
           a[0] * b[1] - a[1] * b[0]}};
}

std::array<real_t, 3> unit_or_default(std::array<real_t, 3> n) noexcept {
  const real_t len = norm(n);
  if (len <= real_t{1.0e-30}) {
    return {{1.0, 0.0, 0.0}};
  }
  return scale(n, real_t{1.0} / len);
}

bool finite_point(const std::array<real_t, 3>& p) noexcept {
  return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

std::array<real_t, 3> closest_point_on_segment(const std::array<real_t, 3>& p,
                                               const std::array<real_t, 3>& a,
                                               const std::array<real_t, 3>& b) noexcept {
  const auto ab = sub(b, a);
  const real_t len2 = dot(ab, ab);
  if (len2 <= real_t{1.0e-30}) {
    return a;
  }
  real_t t = dot(sub(p, a), ab) / len2;
  t = std::max(real_t{0.0}, std::min(real_t{1.0}, t));
  return add(a, scale(ab, t));
}

std::array<real_t, 3> closest_point_on_triangle(const std::array<real_t, 3>& p,
                                                const std::array<real_t, 3>& a,
                                                const std::array<real_t, 3>& b,
                                                const std::array<real_t, 3>& c) noexcept {
  const auto ab = sub(b, a);
  const auto ac = sub(c, a);
  const auto ap = sub(p, a);
  const real_t d1 = dot(ab, ap);
  const real_t d2 = dot(ac, ap);
  if (d1 <= real_t{0.0} && d2 <= real_t{0.0}) return a;

  const auto bp = sub(p, b);
  const real_t d3 = dot(ab, bp);
  const real_t d4 = dot(ac, bp);
  if (d3 >= real_t{0.0} && d4 <= d3) return b;

  const real_t vc = d1 * d4 - d3 * d2;
  if (vc <= real_t{0.0} && d1 >= real_t{0.0} && d3 <= real_t{0.0}) {
    const real_t v = d1 / (d1 - d3);
    return add(a, scale(ab, v));
  }

  const auto cp = sub(p, c);
  const real_t d5 = dot(ab, cp);
  const real_t d6 = dot(ac, cp);
  if (d6 >= real_t{0.0} && d5 <= d6) return c;

  const real_t vb = d5 * d2 - d1 * d6;
  if (vb <= real_t{0.0} && d2 >= real_t{0.0} && d6 <= real_t{0.0}) {
    const real_t w = d2 / (d2 - d6);
    return add(a, scale(ac, w));
  }

  const real_t va = d3 * d6 - d5 * d4;
  if (va <= real_t{0.0} && (d4 - d3) >= real_t{0.0} && (d5 - d6) >= real_t{0.0}) {
    const real_t w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return add(b, scale(sub(c, b), w));
  }

  const real_t denom = real_t{1.0} / (va + vb + vc);
  const real_t v = vb * denom;
  const real_t w = vc * denom;
  return add(a, add(scale(ab, v), scale(ac, w)));
}

std::array<real_t, 3> triangle_normal(const EmbeddedSurfaceTriangle& tri) noexcept {
  return unit_or_default(cross(sub(tri.vertices[1], tri.vertices[0]),
                               sub(tri.vertices[2], tri.vertices[0])));
}

gid_t entity_gid(const MeshBase& mesh, CutEntityKind kind, index_t entity) {
  if (entity < 0) {
    return INVALID_GID;
  }
  const auto i = static_cast<std::size_t>(entity);
  switch (kind) {
    case CutEntityKind::Cell:
      return i < mesh.cell_gids().size() ? mesh.cell_gids()[i] : static_cast<gid_t>(entity);
    case CutEntityKind::Face:
      return i < mesh.face_gids().size() ? mesh.face_gids()[i] : static_cast<gid_t>(entity);
    case CutEntityKind::Edge:
      return i < mesh.edge_gids().size() ? mesh.edge_gids()[i] : static_cast<gid_t>(entity);
  }
  return static_cast<gid_t>(entity);
}

std::uint64_t stable_entity_id(const EmbeddedRegionProvenance& provenance,
                               gid_t parent_gid,
                               std::uint64_t a,
                               std::uint64_t b,
                               real_t fraction,
                               std::uint64_t salt) noexcept;

CutEntityRecord make_record(CutEntityKind kind,
                            index_t entity,
                            gid_t global_id,
                            rank_t owner_rank,
                            const std::vector<index_t>& dofs,
                            const std::vector<real_t>& signed_distances,
                            const std::vector<CutIntersectionPoint>& intersections,
                            real_t tolerance,
                            EmbeddedRegionProvenance provenance) {
  CutEntityRecord record;
  record.kind = kind;
  record.entity = entity;
  record.global_id = global_id;
  record.owner_rank = owner_rank;
  record.classification = classify_signed_distances(signed_distances, tolerance);
  if (!signed_distances.empty()) {
    const auto minmax = std::minmax_element(signed_distances.begin(), signed_distances.end());
    record.min_signed_distance = *minmax.first;
    record.max_signed_distance = *minmax.second;
  }
  record.intersections = intersections;
  record.provenance = std::move(provenance);
  record.cut_topology_id = stable_entity_id(record.provenance,
                                            global_id,
                                            static_cast<std::uint64_t>(kind),
                                            0u,
                                            record.min_signed_distance,
                                            17u);
  for (const real_t d : signed_distances) {
    if (!std::isfinite(d)) {
      record.diagnostics.push_back("non-finite embedded signed-distance value");
      break;
    }
  }
  (void)dofs;
  return record;
}

std::vector<CutIntersectionPoint> edge_intersections(
    const MeshBase& mesh,
    const EmbeddedGeometryDescriptor& embedded,
    const std::vector<index_t>& dofs,
    const std::vector<std::array<index_t, 2>>& local_edges,
    const std::vector<real_t>& signed_distances,
    Configuration cfg,
    real_t tolerance) {
  std::vector<CutIntersectionPoint> out;
  for (const auto& e : local_edges) {
    const auto ia = static_cast<std::size_t>(e[0]);
    const auto ib = static_cast<std::size_t>(e[1]);
    if (ia >= dofs.size() || ib >= dofs.size()) {
      continue;
    }
    const real_t da = signed_distances[ia];
    const real_t db = signed_distances[ib];
    if ((da > tolerance && db > tolerance) || (da < -tolerance && db < -tolerance)) {
      continue;
    }
    if (std::abs(da - db) <= tolerance && std::abs(da) > tolerance) {
      continue;
    }
    const real_t denom = da - db;
    real_t t = std::abs(denom) <= tolerance ? real_t{0.0} : da / denom;
    t = std::max(real_t{0.0}, std::min(real_t{1.0}, t));
    const auto pa = mesh.geometry_dof_coords(dofs[ia], cfg);
    const auto pb = mesh.geometry_dof_coords(dofs[ib], cfg);
    const auto p = add(scale(pa, real_t{1.0} - t), scale(pb, t));

    CutIntersectionPoint hit;
    hit.point = p;
    hit.normal = embedded.outward_normal(p);
    hit.edge_fraction = t;
    hit.endpoint_a = dofs[ia];
    hit.endpoint_b = dofs[ib];
    out.push_back(hit);
  }
  return out;
}

std::vector<std::array<index_t, 2>> cyclic_edges(std::size_t n) {
  std::vector<std::array<index_t, 2>> edges;
  if (n < 2u) {
    return edges;
  }
  edges.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    edges.push_back({{static_cast<index_t>(i), static_cast<index_t>((i + 1u) % n)}});
  }
  return edges;
}

std::vector<std::array<index_t, 2>> cell_local_edges(
    const MeshBase& mesh,
    index_t cell,
    const std::vector<index_t>& dofs) {
  const auto family = mesh.cell_shape(cell).family;
  if (family == CellFamily::Line && dofs.size() >= 2u) {
    return {{{0, 1}}};
  }
  if (family == CellFamily::Polygon) {
    return cyclic_edges(dofs.size());
  }
  if (family == CellFamily::Polyhedron) {
    std::map<index_t, index_t> global_to_local;
    for (std::size_t i = 0; i < dofs.size(); ++i) {
      global_to_local[dofs[i]] = static_cast<index_t>(i);
    }
    std::vector<std::array<index_t, 2>> edges;
    for (const auto face : mesh.cell_faces(cell)) {
      const auto [fv, nf] = mesh.face_vertices_span(face);
      if (nf < 2u) {
        continue;
      }
      for (std::size_t i = 0; i < nf; ++i) {
        const auto a_it = global_to_local.find(fv[i]);
        const auto b_it = global_to_local.find(fv[(i + 1u) % nf]);
        if (a_it == global_to_local.end() || b_it == global_to_local.end()) {
          continue;
        }
        std::array<index_t, 2> edge{{a_it->second, b_it->second}};
        if (edge[1] < edge[0]) {
          std::swap(edge[0], edge[1]);
        }
        edges.push_back(edge);
      }
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    return edges;
  }

  const auto eview = CellTopology::get_edges_view(family);
  std::vector<std::array<index_t, 2>> local_edges;
  local_edges.reserve(static_cast<std::size_t>(std::max(eview.edge_count, 0)));
  for (int e = 0; e < eview.edge_count; ++e) {
    local_edges.push_back({{eview.pairs_flat[2 * e], eview.pairs_flat[2 * e + 1]}});
  }
  return local_edges;
}

std::vector<real_t> signed_distances_for_dofs(
    const MeshBase& mesh,
    const EmbeddedGeometryDescriptor& embedded,
    const std::vector<index_t>& dofs,
    Configuration cfg) {
  std::vector<real_t> distances;
  distances.reserve(dofs.size());
  for (const index_t dof : dofs) {
    distances.push_back(embedded.signed_distance(mesh.geometry_dof_coords(dof, cfg)));
  }
  return distances;
}

std::vector<index_t> span_to_vector(std::pair<const index_t*, std::size_t> span) {
  std::vector<index_t> out;
  out.reserve(span.second);
  for (std::size_t i = 0; i < span.second; ++i) {
    out.push_back(span.first[i]);
  }
  return out;
}

std::uint64_t max_constraint_epoch(const std::vector<EmbeddedKinematicConstraint>& constraints) noexcept {
  std::uint64_t epoch = 0;
  for (const auto& c : constraints) {
    epoch = std::max(epoch, c.constraint_epoch);
  }
  return epoch;
}

EmbeddedGeometryRevisionState max_revision_state(
    EmbeddedGeometryRevisionState a,
    const EmbeddedGeometryRevisionState& b) noexcept {
  a.geometry_epoch = std::max(a.geometry_epoch, b.geometry_epoch);
  a.field_layout_revision = std::max(a.field_layout_revision, b.field_layout_revision);
  a.field_value_revision = std::max(a.field_value_revision, b.field_value_revision);
  a.source_surface_revision = std::max(a.source_surface_revision, b.source_surface_revision);
  a.provenance_revision = std::max(a.provenance_revision, b.provenance_revision);
  a.kinematic_constraint_revision = std::max(a.kinematic_constraint_revision, b.kinematic_constraint_revision);
  return a;
}

real_t effective_tolerance(const CutClassificationOptions& options) noexcept {
  if (options.predicate_policy.robust.intersection_tolerance > real_t{0.0}) {
    return options.predicate_policy.robust.intersection_tolerance;
  }
  return options.tolerance;
}

std::uint64_t stable_entity_id(const EmbeddedRegionProvenance& provenance,
                               gid_t parent_gid,
                               std::uint64_t a,
                               std::uint64_t b,
                               real_t fraction,
                               std::uint64_t salt) noexcept {
  std::uint64_t h = kFnvOffset;
  h = append_hash_string(h, provenance.persistent_id);
  h = append_hash_string(h, provenance.name);
  h = append_hash(h, static_cast<std::uint64_t>(parent_gid));
  h = append_hash(h, std::min(a, b));
  h = append_hash(h, std::max(a, b));
  h = append_hash_real(h, fraction);
  h = append_hash(h, salt);
  return h;
}

std::uint64_t stable_geometry_dof_key(const MeshBase& mesh, index_t dof) noexcept {
  if (dof >= 0) {
    const auto i = static_cast<std::size_t>(dof);
    if (i < mesh.vertex_gids().size() && mesh.vertex_gids()[i] != INVALID_GID) {
      return static_cast<std::uint64_t>(mesh.vertex_gids()[i]);
    }
  }
  return append_hash(0x9e3779b97f4a7c15ull, static_cast<std::uint64_t>(dof));
}

std::array<real_t, 3> first_valid_normal(const CutEntityRecord& record,
                                         const EmbeddedGeometryDescriptor& embedded) noexcept {
  for (const auto& hit : record.intersections) {
    if (norm(hit.normal) > real_t{0.0}) {
      return unit_or_default(hit.normal);
    }
  }
  return embedded.outward_normal({{0.0, 0.0, 0.0}});
}

void update_distributed_owners(const DistributedMesh& mesh, CutClassificationMap& map) {
  for (auto& record : map.cells) {
    record.owner_rank = mesh.owner_rank_cell(record.entity);
  }
  for (auto& record : map.faces) {
    record.owner_rank = mesh.owner_rank_face(record.entity);
  }
  for (auto& record : map.edges) {
    record.owner_rank = mesh.owner_rank_edge(record.entity);
  }
}

EmbeddedGeometryRestartRecord make_restart_record(const EmbeddedGeometryDescriptor& descriptor) {
  EmbeddedGeometryRestartRecord record;
  record.persistent_id = descriptor.provenance.persistent_id;
  record.name = descriptor.provenance.name;
  record.kind = descriptor.kind;
  record.configuration = normalized_configuration(descriptor.configuration);
  record.origin = descriptor.origin;
  record.normal = descriptor.normal;
  record.radius = descriptor.radius;
  record.revisions = descriptor.effective_revisions();
  record.provenance = descriptor.provenance;
  record.active = descriptor.active;
  record.boolean_operation = descriptor.boolean_operation;
  record.level_set_samples = descriptor.level_set_samples;
  record.surface_triangles = descriptor.surface_triangles;
  record.requires_application_reregistration =
      descriptor.kind == EmbeddedGeometryKind::SignedDistanceCallback;
  record.children.reserve(descriptor.children.size());
  for (const auto& child : descriptor.children) {
    record.children.push_back(make_restart_record(child));
  }
  return record;
}

EmbeddedGeometryDescriptor descriptor_from_restart_record(
    const EmbeddedGeometryRestartRecord& record) {
  EmbeddedGeometryDescriptor descriptor;
  descriptor.kind = record.kind;
  descriptor.configuration = normalized_configuration(record.configuration);
  descriptor.origin = record.origin;
  descriptor.normal = record.normal;
  descriptor.radius = record.radius;
  descriptor.geometry_epoch = record.revisions.geometry_epoch;
  descriptor.revisions = record.revisions;
  descriptor.provenance = record.provenance;
  descriptor.provenance.persistent_id = !record.persistent_id.empty()
                                            ? record.persistent_id
                                            : record.provenance.persistent_id;
  descriptor.provenance.name = !record.name.empty() ? record.name : record.provenance.name;
  descriptor.active = record.active;
  descriptor.boolean_operation = record.boolean_operation;
  descriptor.level_set_samples = record.level_set_samples;
  descriptor.surface_triangles = record.surface_triangles;
  descriptor.children.reserve(record.children.size());
  for (const auto& child : record.children) {
    descriptor.children.push_back(descriptor_from_restart_record(child));
  }
  return descriptor;
}

void collect_composition_child_records(
    const EmbeddedGeometryDescriptor& parent,
    std::size_t depth,
    std::vector<EmbeddedCompositionChildRestartRecord>& out) {
  if (parent.kind != EmbeddedGeometryKind::BooleanComposite) {
    return;
  }
  const std::string parent_id = !parent.provenance.persistent_id.empty()
                                    ? parent.provenance.persistent_id
                                    : parent.provenance.name;
  for (std::size_t i = 0; i < parent.children.size(); ++i) {
    const auto& child = parent.children[i];
    EmbeddedCompositionChildRestartRecord record;
    record.depth = depth + 1u;
    record.child_ordinal = i;
    record.parent_persistent_id = parent_id;
    record.provenance = child.provenance;
    record.kind = child.kind;
    record.boolean_operation = child.boolean_operation;
    record.revisions = child.effective_revisions();
    out.push_back(record);
    collect_composition_child_records(child, depth + 1u, out);
  }
}

rank_t owner_for_cell(const CutClassificationMap& map, index_t cell) noexcept {
  for (const auto& record : map.cells) {
    if (record.entity == cell) {
      return record.owner_rank;
    }
  }
  return 0;
}

std::string provenance_id(const EmbeddedRegionProvenance& provenance) {
  return !provenance.persistent_id.empty() ? provenance.persistent_id : provenance.name;
}

real_t interface_measure_from_topology(
    const CutTopologyRecord& topology,
    const CutInterfacePolygon& polygon) noexcept {
  std::vector<std::array<real_t, 3>> points;
  points.reserve(polygon.ordered_vertices.size());
  for (const auto id : polygon.ordered_vertices) {
    const auto it = std::find_if(topology.vertices.begin(), topology.vertices.end(), [&](const auto& v) {
      return v.stable_id == id;
    });
    if (it != topology.vertices.end()) {
      points.push_back(it->point);
    }
  }
  if (points.empty()) {
    return real_t{0.0};
  }
  if (points.size() == 1u) {
    return real_t{1.0};
  }
  if (points.size() == 2u) {
    return norm(sub(points[1], points[0]));
  }
  const real_t area = MeshGeometry::polygon_area(points);
  if (area > real_t{1.0e-30}) {
    return area;
  }
  real_t length = real_t{0.0};
  for (std::size_t i = 0; i < points.size(); ++i) {
    for (std::size_t j = i + 1u; j < points.size(); ++j) {
      length = std::max(length, norm(sub(points[j], points[i])));
    }
  }
  return length;
}

const CutTopologyVertex* find_topology_vertex(
    const CutTopologyRecord& topology,
    std::uint64_t stable_id) noexcept {
  const auto it = std::find_if(topology.vertices.begin(), topology.vertices.end(), [&](const auto& v) {
    return v.stable_id == stable_id;
  });
  return it == topology.vertices.end() ? nullptr : &(*it);
}

std::vector<const CutTopologyVertex*> polygon_topology_vertices(
    const CutTopologyRecord& topology,
    const CutInterfacePolygon& polygon) {
  std::vector<const CutTopologyVertex*> vertices;
  vertices.reserve(polygon.ordered_vertices.size());
  for (const auto id : polygon.ordered_vertices) {
    vertices.push_back(find_topology_vertex(topology, id));
  }
  return vertices;
}

std::vector<std::array<real_t, 3>> polygon_points(
    const std::vector<const CutTopologyVertex*>& vertices) {
  std::vector<std::array<real_t, 3>> points;
  points.reserve(vertices.size());
  for (const auto* vertex : vertices) {
    if (vertex != nullptr) {
      points.push_back(vertex->point);
    }
  }
  return points;
}

std::vector<std::array<real_t, 2>> project_polygon_points(
    const std::vector<std::array<real_t, 3>>& points,
    std::array<real_t, 3> normal) {
  normal = unit_or_default(normal);
  const auto tangent0 = std::abs(normal[0]) < real_t{0.9}
                            ? unit_or_default(cross(normal, {{1.0, 0.0, 0.0}}))
                            : unit_or_default(cross(normal, {{0.0, 1.0, 0.0}}));
  const auto tangent1 = cross(normal, tangent0);
  std::vector<std::array<real_t, 2>> out;
  out.reserve(points.size());
  for (const auto& p : points) {
    out.push_back({{dot(p, tangent0), dot(p, tangent1)}});
  }
  return out;
}

real_t orient2d(const std::array<real_t, 2>& a,
                const std::array<real_t, 2>& b,
                const std::array<real_t, 2>& c) noexcept {
  return (b[0] - a[0]) * (c[1] - a[1]) -
         (b[1] - a[1]) * (c[0] - a[0]);
}

bool on_segment2d(const std::array<real_t, 2>& a,
                  const std::array<real_t, 2>& b,
                  const std::array<real_t, 2>& p,
                  real_t tol) noexcept {
  if (std::abs(orient2d(a, b, p)) > tol) {
    return false;
  }
  return p[0] >= std::min(a[0], b[0]) - tol &&
         p[0] <= std::max(a[0], b[0]) + tol &&
         p[1] >= std::min(a[1], b[1]) - tol &&
         p[1] <= std::max(a[1], b[1]) + tol;
}

bool segments_intersect2d(const std::array<real_t, 2>& a,
                          const std::array<real_t, 2>& b,
                          const std::array<real_t, 2>& c,
                          const std::array<real_t, 2>& d,
                          real_t tol) noexcept {
  const real_t o1 = orient2d(a, b, c);
  const real_t o2 = orient2d(a, b, d);
  const real_t o3 = orient2d(c, d, a);
  const real_t o4 = orient2d(c, d, b);
  if (((o1 > tol && o2 < -tol) || (o1 < -tol && o2 > tol)) &&
      ((o3 > tol && o4 < -tol) || (o3 < -tol && o4 > tol))) {
    return true;
  }
  return on_segment2d(a, b, c, tol) ||
         on_segment2d(a, b, d, tol) ||
         on_segment2d(c, d, a, tol) ||
         on_segment2d(c, d, b, tol);
}

bool projected_polygon_self_intersects(
    const std::vector<std::array<real_t, 3>>& points,
    const std::array<real_t, 3>& normal,
    real_t tol) {
  if (points.size() < 4u) {
    return false;
  }
  const auto projected = project_polygon_points(points, normal);
  const std::size_t n = projected.size();
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t inext = (i + 1u) % n;
    for (std::size_t j = i + 1u; j < n; ++j) {
      const std::size_t jnext = (j + 1u) % n;
      if (i == j || inext == j || jnext == i) {
        continue;
      }
      if (i == 0u && jnext == 0u) {
        continue;
      }
      if (segments_intersect2d(projected[i], projected[inext],
                               projected[j], projected[jnext], tol)) {
        return true;
      }
    }
  }
  return false;
}

bool has_duplicate_or_short_polygon_edge(
    const std::vector<std::array<real_t, 3>>& points,
    real_t tol) noexcept {
  if (points.empty()) {
    return true;
  }
  if (points.size() == 1u) {
    return false;
  }
  for (std::size_t i = 0; i < points.size(); ++i) {
    if (norm(sub(points[i], points[(i + 1u) % points.size()])) <= tol) {
      return true;
    }
    for (std::size_t j = i + 1u; j < points.size(); ++j) {
      if (norm(sub(points[i], points[j])) <= tol) {
        return true;
      }
    }
  }
  return false;
}

struct SideMeasureEstimate {
  real_t parent_measure{0.0};
  real_t measure{0.0};
  real_t fraction{0.0};
  std::array<real_t, 3> centroid{{0.0, 0.0, 0.0}};
  bool available{false};
  bool used_tessellation{false};
};

std::array<real_t, 3> weighted_centroid(
    const std::array<real_t, 3>& a,
    real_t aw,
    const std::array<real_t, 3>& b,
    real_t bw) noexcept {
  const real_t w = aw + bw;
  if (w <= real_t{0.0}) {
    return {{0.0, 0.0, 0.0}};
  }
  return scale(add(scale(a, aw), scale(b, bw)), real_t{1.0} / w);
}

real_t positive_tolerance(real_t tol) noexcept {
  return tol > std::numeric_limits<real_t>::epsilon()
             ? tol
             : std::numeric_limits<real_t>::epsilon();
}

std::vector<std::array<real_t, 3>> unique_points(
    std::vector<std::array<real_t, 3>> points,
    real_t tol) {
  const real_t eps = positive_tolerance(tol);
  auto same = [&](const auto& a, const auto& b) {
    return norm(sub(a, b)) <= eps;
  };
  std::vector<std::array<real_t, 3>> out;
  for (const auto& p : points) {
    if (std::find_if(out.begin(), out.end(), [&](const auto& q) { return same(p, q); }) == out.end()) {
      out.push_back(p);
    }
  }
  return out;
}

struct ParametricClipPoint {
  std::array<real_t, 3> point{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> xi{{0.0, 0.0, 0.0}};
};

struct IsoparametricMeasureResult {
  real_t measure{0.0};
  std::array<real_t, 3> centroid{{0.0, 0.0, 0.0}};
  bool available{false};
};

std::vector<std::vector<int>> convex_hull_faces_indices(
    const std::vector<std::array<real_t, 3>>& input_points,
    real_t tol);

real_t convex_polyhedron_volume(
    const std::vector<std::array<real_t, 3>>& input_points,
    real_t tol);

std::array<real_t, 3> jacobian_action(
    const Jacobian& jacobian,
    const std::array<real_t, 3>& direction) noexcept {
  std::array<real_t, 3> out{{0.0, 0.0, 0.0}};
  const int cols = std::max(0, std::min(3, jacobian.parametric_dim));
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < cols; ++j) {
      out[static_cast<std::size_t>(i)] +=
          jacobian.matrix[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] *
          direction[static_cast<std::size_t>(j)];
    }
  }
  return out;
}

real_t determinant3(const std::array<real_t, 3>& a,
                    const std::array<real_t, 3>& b,
                    const std::array<real_t, 3>& c) noexcept {
  return dot(a, cross(b, c));
}

std::array<real_t, 3> interpolate_parametric(
    const std::array<real_t, 3>& a,
    const std::array<real_t, 3>& b,
    real_t t) noexcept {
  return add(scale(a, real_t{1.0} - t), scale(b, t));
}

std::array<real_t, 3> interpolate_parametric(
    const std::array<real_t, 3>& a,
    const std::array<real_t, 3>& b,
    const std::array<real_t, 3>& c,
    real_t r,
    real_t s) noexcept {
  return add(a, add(scale(sub(b, a), r), scale(sub(c, a), s)));
}

std::array<real_t, 3> interpolate_parametric(
    const std::array<real_t, 3>& a,
    const std::array<real_t, 3>& b,
    const std::array<real_t, 3>& c,
    const std::array<real_t, 3>& d,
    real_t r,
    real_t s,
    real_t t) noexcept {
  return add(a, add(add(scale(sub(b, a), r), scale(sub(c, a), s)), scale(sub(d, a), t)));
}

std::vector<ParametricClipPoint> unique_parametric_clip_points(
    std::vector<ParametricClipPoint> points,
    real_t tol) {
  const real_t eps = positive_tolerance(tol);
  std::vector<ParametricClipPoint> out;
  for (const auto& p : points) {
    if (std::find_if(out.begin(), out.end(), [&](const auto& q) {
          return norm(sub(p.point, q.point)) <= eps &&
                 norm(sub(p.xi, q.xi)) <= eps * real_t{10.0};
        }) == out.end()) {
      out.push_back(p);
    }
  }
  return out;
}

std::vector<ParametricClipPoint> clip_polygon_parametric_points(
    const std::vector<std::array<real_t, 3>>& points,
    const std::vector<std::array<real_t, 3>>& parametric_points,
    const std::vector<real_t>& values,
    CutTopologySide side,
    real_t tol) {
  std::vector<ParametricClipPoint> out;
  if (points.empty() ||
      points.size() != parametric_points.size() ||
      points.size() != values.size()) {
    return out;
  }
  const auto inside = [&](real_t d) {
    return side == CutTopologySide::Negative ? d <= tol : d >= -tol;
  };
  for (std::size_t i = 0; i < points.size(); ++i) {
    const std::size_t j = (i + 1u) % points.size();
    const real_t da = values[i];
    const real_t db = values[j];
    const bool a_in = inside(da);
    const bool b_in = inside(db);
    if (a_in && b_in) {
      out.push_back({points[j], parametric_points[j]});
    } else if (a_in != b_in) {
      real_t t = real_t{0.0};
      if (std::abs(da - db) > tol) {
        t = da / (da - db);
      }
      t = std::max(real_t{0.0}, std::min(real_t{1.0}, t));
      out.push_back({interpolate_parametric(points[i], points[j], t),
                     interpolate_parametric(parametric_points[i], parametric_points[j], t)});
      if (!a_in && b_in) {
        out.push_back({points[j], parametric_points[j]});
      }
    }
  }
  return unique_parametric_clip_points(std::move(out), tol);
}

std::vector<ParametricClipPoint> clipped_tet_parametric_points(
    const std::array<std::array<real_t, 3>, 4>& tet,
    const std::array<std::array<real_t, 3>, 4>& tet_parametric,
    const std::array<real_t, 4>& values,
    CutTopologySide side,
    real_t tol) {
  const auto inside = [&](real_t d) {
    return side == CutTopologySide::Negative ? d <= tol : d >= -tol;
  };
  std::vector<ParametricClipPoint> points;
  for (int i = 0; i < 4; ++i) {
    if (inside(values[static_cast<std::size_t>(i)])) {
      points.push_back({tet[static_cast<std::size_t>(i)],
                        tet_parametric[static_cast<std::size_t>(i)]});
    }
  }
  constexpr std::array<std::array<int, 2>, 6> edges{{
      {{0, 1}}, {{0, 2}}, {{0, 3}}, {{1, 2}}, {{1, 3}}, {{2, 3}}}};
  for (const auto& edge : edges) {
    const int a = edge[0];
    const int b = edge[1];
    const real_t da = values[static_cast<std::size_t>(a)];
    const real_t db = values[static_cast<std::size_t>(b)];
    if (inside(da) == inside(db) || std::abs(da - db) <= tol) {
      continue;
    }
    real_t t = da / (da - db);
    t = std::max(real_t{0.0}, std::min(real_t{1.0}, t));
    points.push_back({
        interpolate_parametric(tet[static_cast<std::size_t>(a)],
                               tet[static_cast<std::size_t>(b)],
                               t),
        interpolate_parametric(tet_parametric[static_cast<std::size_t>(a)],
                               tet_parametric[static_cast<std::size_t>(b)],
                               t)});
  }
  return unique_parametric_clip_points(std::move(points), tol);
}

IsoparametricMeasureResult isoparametric_line_measure(
    const MeshBase& mesh,
    index_t cell,
    const std::array<real_t, 3>& a,
    const std::array<real_t, 3>& b,
    Configuration cfg) {
  IsoparametricMeasureResult out;
  constexpr real_t g = real_t{0.57735026918962576451};
  const std::array<real_t, 2> points{{-g, g}};
  const auto half_delta = scale(sub(b, a), real_t{0.5});
  for (const auto s : points) {
    const auto xi = interpolate_parametric(a, b, (s + real_t{1.0}) * real_t{0.5});
    const auto eval = CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, cfg);
    const real_t w = norm(jacobian_action(eval.jacobian, half_delta));
    if (!std::isfinite(w)) {
      return {};
    }
    out.measure += w;
    out.centroid = add(out.centroid, scale(eval.coordinates, w));
  }
  if (out.measure > real_t{0.0}) {
    out.centroid = scale(out.centroid, real_t{1.0} / out.measure);
  }
  out.available = true;
  return out;
}

IsoparametricMeasureResult isoparametric_triangle_measure(
    const MeshBase& mesh,
    index_t cell,
    const std::array<real_t, 3>& a,
    const std::array<real_t, 3>& b,
    const std::array<real_t, 3>& c,
    Configuration cfg) {
  IsoparametricMeasureResult out;
  constexpr std::array<std::array<real_t, 2>, 3> q{{
      {{real_t{1.0} / real_t{6.0}, real_t{1.0} / real_t{6.0}}},
      {{real_t{2.0} / real_t{3.0}, real_t{1.0} / real_t{6.0}}},
      {{real_t{1.0} / real_t{6.0}, real_t{2.0} / real_t{3.0}}}}};
  const auto db = sub(b, a);
  const auto dc = sub(c, a);
  for (const auto& rs : q) {
    const auto xi = interpolate_parametric(a, b, c, rs[0], rs[1]);
    const auto eval = CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, cfg);
    const auto dx_dr = jacobian_action(eval.jacobian, db);
    const auto dx_ds = jacobian_action(eval.jacobian, dc);
    const real_t w = norm(cross(dx_dr, dx_ds)) / real_t{6.0};
    if (!std::isfinite(w)) {
      return {};
    }
    out.measure += w;
    out.centroid = add(out.centroid, scale(eval.coordinates, w));
  }
  if (out.measure > real_t{0.0}) {
    out.centroid = scale(out.centroid, real_t{1.0} / out.measure);
  }
  out.available = true;
  return out;
}

IsoparametricMeasureResult isoparametric_tet_measure(
    const MeshBase& mesh,
    index_t cell,
    const std::array<real_t, 3>& a,
    const std::array<real_t, 3>& b,
    const std::array<real_t, 3>& c,
    const std::array<real_t, 3>& d,
    Configuration cfg) {
  IsoparametricMeasureResult out;
  constexpr real_t alpha = real_t{0.5854101966249685};
  constexpr real_t beta = real_t{0.1381966011250105};
  constexpr std::array<std::array<real_t, 3>, 4> q{{
      {{beta, beta, beta}},
      {{alpha, beta, beta}},
      {{beta, alpha, beta}},
      {{beta, beta, alpha}}}};
  const auto db = sub(b, a);
  const auto dc = sub(c, a);
  const auto dd = sub(d, a);
  for (const auto& rst : q) {
    const auto xi = interpolate_parametric(a, b, c, d, rst[0], rst[1], rst[2]);
    const auto eval = CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, cfg);
    const auto dx_dr = jacobian_action(eval.jacobian, db);
    const auto dx_ds = jacobian_action(eval.jacobian, dc);
    const auto dx_dt = jacobian_action(eval.jacobian, dd);
    const real_t w = std::abs(determinant3(dx_dr, dx_ds, dx_dt)) / real_t{24.0};
    if (!std::isfinite(w)) {
      return {};
    }
    out.measure += w;
    out.centroid = add(out.centroid, scale(eval.coordinates, w));
  }
  if (out.measure > real_t{0.0}) {
    out.centroid = scale(out.centroid, real_t{1.0} / out.measure);
  }
  out.available = true;
  return out;
}

IsoparametricMeasureResult accumulate_isoparametric_measure(
    IsoparametricMeasureResult total,
    const IsoparametricMeasureResult& part) noexcept {
  if (!part.available) {
    total.available = false;
    return total;
  }
  total.centroid = weighted_centroid(total.centroid, total.measure, part.centroid, part.measure);
  total.measure += part.measure;
  total.available = total.available || part.available;
  return total;
}

IsoparametricMeasureResult isoparametric_polygon_measure(
    const MeshBase& mesh,
    index_t cell,
    const std::vector<std::array<real_t, 3>>& xi_points,
    Configuration cfg) {
  IsoparametricMeasureResult out;
  if (xi_points.size() == 2u) {
    return isoparametric_line_measure(mesh, cell, xi_points[0], xi_points[1], cfg);
  }
  if (xi_points.size() < 3u) {
    return out;
  }
  out.available = true;
  for (std::size_t i = 1; i + 1 < xi_points.size(); ++i) {
    out = accumulate_isoparametric_measure(
        out,
        isoparametric_triangle_measure(mesh, cell, xi_points[0], xi_points[i], xi_points[i + 1u], cfg));
    if (!out.available) {
      return {};
    }
  }
  return out;
}

IsoparametricMeasureResult isoparametric_polyhedron_measure(
    const MeshBase& mesh,
    index_t cell,
    std::vector<std::array<real_t, 3>> xi_points,
    Configuration cfg,
    real_t tol) {
  xi_points = unique_points(std::move(xi_points), tol);
  IsoparametricMeasureResult out;
  if (xi_points.size() < 4u) {
    return out;
  }
  std::array<real_t, 3> center{{0.0, 0.0, 0.0}};
  for (const auto& xi : xi_points) {
    center = add(center, xi);
  }
  center = scale(center, real_t{1.0} / static_cast<real_t>(xi_points.size()));

  const auto faces = convex_hull_faces_indices(xi_points, tol);
  out.available = !faces.empty();
  for (const auto& face : faces) {
    if (face.size() < 3u) {
      continue;
    }
    const auto& xi0 = xi_points[static_cast<std::size_t>(face[0])];
    for (std::size_t i = 1; i + 1 < face.size(); ++i) {
      const auto& xi1 = xi_points[static_cast<std::size_t>(face[i])];
      const auto& xi2 = xi_points[static_cast<std::size_t>(face[i + 1u])];
      out = accumulate_isoparametric_measure(
          out,
          isoparametric_tet_measure(mesh, cell, center, xi0, xi1, xi2, cfg));
      if (!out.available) {
        return {};
      }
    }
  }
  return out;
}

real_t parametric_measure(
    CellFamily family,
    const std::vector<std::array<real_t, 3>>& xi_points,
    real_t tol) {
  if (family == CellFamily::Line && xi_points.size() >= 2u) {
    return norm(sub(xi_points[1], xi_points[0]));
  }
  if ((family == CellFamily::Triangle ||
       family == CellFamily::Quad ||
       family == CellFamily::Polygon) &&
      xi_points.size() >= 3u) {
    real_t area = real_t{0.0};
    for (std::size_t i = 1; i + 1 < xi_points.size(); ++i) {
      area += real_t{0.5} * norm(cross(sub(xi_points[i], xi_points[0]),
                                       sub(xi_points[i + 1u], xi_points[0])));
    }
    return area;
  }
  if (xi_points.size() >= 4u) {
    return convex_polyhedron_volume(xi_points, tol);
  }
  return real_t{0.0};
}

IsoparametricMeasureResult isoparametric_measure_for_subcell(
    const MeshBase& mesh,
    index_t cell,
    CellFamily family,
    const std::vector<std::array<real_t, 3>>& xi_points,
    Configuration cfg,
    real_t tol) {
  if (family == CellFamily::Line && xi_points.size() >= 2u) {
    return isoparametric_line_measure(mesh, cell, xi_points[0], xi_points[1], cfg);
  }
  if ((family == CellFamily::Triangle ||
       family == CellFamily::Quad ||
       family == CellFamily::Polygon) &&
      xi_points.size() >= 3u) {
    return isoparametric_polygon_measure(mesh, cell, xi_points, cfg);
  }
  if ((family == CellFamily::Tetra ||
       family == CellFamily::Hex ||
       family == CellFamily::Wedge ||
       family == CellFamily::Pyramid ||
       family == CellFamily::Polyhedron) &&
      xi_points.size() >= 4u) {
    return isoparametric_polyhedron_measure(mesh, cell, xi_points, cfg, tol);
  }
  return {};
}

struct ParametricTet4 {
  std::array<std::array<real_t, 3>, 4> points{};
  std::array<std::array<real_t, 3>, 4> parent_parametric_points{};
};

std::vector<ParametricTet4> linear_cell_parametric_tets(
    CellFamily family,
    const std::vector<std::array<real_t, 3>>& points,
    const std::vector<std::array<real_t, 3>>& parent_parametric_points,
    real_t tol) {
  std::vector<ParametricTet4> out;
  if (points.size() != parent_parametric_points.size()) {
    return out;
  }

  const auto tet_indices =
      PolyhedronTessellation::linear_cell_tet_corner_indices(family, points.size());
  out.reserve(tet_indices.size());
  for (auto ids : tet_indices) {
    ParametricTet4 tet;
    for (std::size_t i = 0; i < ids.size(); ++i) {
      tet.points[i] = points[ids[i]];
      tet.parent_parametric_points[i] = parent_parametric_points[ids[i]];
    }
    const real_t vol6 = determinant3(sub(tet.points[1], tet.points[0]),
                                    sub(tet.points[2], tet.points[0]),
                                    sub(tet.points[3], tet.points[0]));
    if (std::abs(vol6) <= positive_tolerance(tol)) {
      continue;
    }
    if (vol6 < real_t{0.0}) {
      std::swap(tet.points[2], tet.points[3]);
      std::swap(tet.parent_parametric_points[2], tet.parent_parametric_points[3]);
    }
    out.push_back(tet);
  }
  return out;
}

IsoparametricMeasureResult clipped_isoparametric_measure_from_tets(
    const MeshBase& mesh,
    index_t cell,
    const std::vector<ParametricTet4>& tets,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    CutTopologySide side,
    real_t tol) {
  IsoparametricMeasureResult total;
  bool initialized = false;
  for (const auto& tet : tets) {
    const std::array<real_t, 4> vals{{embedded.signed_distance(tet.points[0]),
                                      embedded.signed_distance(tet.points[1]),
                                      embedded.signed_distance(tet.points[2]),
                                      embedded.signed_distance(tet.points[3])}};
    const auto clipped =
        clipped_tet_parametric_points(tet.points, tet.parent_parametric_points, vals, side, tol);
    if (clipped.size() < 4u) {
      continue;
    }
    std::vector<std::array<real_t, 3>> clipped_xi;
    clipped_xi.reserve(clipped.size());
    for (const auto& point : clipped) {
      clipped_xi.push_back(point.xi);
    }
    const auto part = isoparametric_polyhedron_measure(mesh, cell, clipped_xi, cfg, tol);
    if (!part.available) {
      return {};
    }
    if (!initialized) {
      total = part;
      initialized = true;
    } else {
      total = accumulate_isoparametric_measure(total, part);
      if (!total.available) {
        return {};
      }
    }
  }
  if (!initialized) {
    total.available = true;
  }
  return total;
}

std::vector<std::array<real_t, 3>> clip_polygon_points(
    const std::vector<std::array<real_t, 3>>& points,
    const std::vector<real_t>& values,
    CutTopologySide side,
    real_t tol) {
  std::vector<std::array<real_t, 3>> out;
  if (points.empty() || points.size() != values.size()) {
    return out;
  }
  const auto inside = [&](real_t d) {
    return side == CutTopologySide::Negative ? d <= tol : d >= -tol;
  };
  for (std::size_t i = 0; i < points.size(); ++i) {
    const auto& a = points[i];
    const auto& b = points[(i + 1u) % points.size()];
    const real_t da = values[i];
    const real_t db = values[(i + 1u) % points.size()];
    const bool a_in = inside(da);
    const bool b_in = inside(db);
    if (a_in && b_in) {
      out.push_back(b);
    } else if (a_in != b_in) {
      real_t t = real_t{0.0};
      if (std::abs(da - db) > tol) {
        t = da / (da - db);
      }
      t = std::max(real_t{0.0}, std::min(real_t{1.0}, t));
      out.push_back(add(scale(a, real_t{1.0} - t), scale(b, t)));
      if (!a_in && b_in) {
        out.push_back(b);
      }
    }
  }
  return unique_points(std::move(out), tol);
}

std::vector<std::array<real_t, 3>> clipped_tet_points(
    const std::array<std::array<real_t, 3>, 4>& tet,
    const std::array<real_t, 4>& values,
    CutTopologySide side,
    real_t tol) {
  const auto inside = [&](real_t d) {
    return side == CutTopologySide::Negative ? d <= tol : d >= -tol;
  };
  std::vector<std::array<real_t, 3>> points;
  for (int i = 0; i < 4; ++i) {
    if (inside(values[static_cast<std::size_t>(i)])) {
      points.push_back(tet[static_cast<std::size_t>(i)]);
    }
  }
  constexpr std::array<std::array<int, 2>, 6> edges{{
      {{0, 1}}, {{0, 2}}, {{0, 3}}, {{1, 2}}, {{1, 3}}, {{2, 3}}}};
  for (const auto& edge : edges) {
    const int a = edge[0];
    const int b = edge[1];
    const real_t da = values[static_cast<std::size_t>(a)];
    const real_t db = values[static_cast<std::size_t>(b)];
    if (inside(da) == inside(db) || std::abs(da - db) <= tol) {
      continue;
    }
    real_t t = da / (da - db);
    t = std::max(real_t{0.0}, std::min(real_t{1.0}, t));
    points.push_back(add(scale(tet[static_cast<std::size_t>(a)], real_t{1.0} - t),
                         scale(tet[static_cast<std::size_t>(b)], t)));
  }
  return unique_points(std::move(points), tol);
}

std::vector<std::vector<int>> convex_hull_faces_indices(
    const std::vector<std::array<real_t, 3>>& input_points,
    real_t tol) {
  tol = positive_tolerance(tol);
  const auto points = unique_points(input_points, tol);
  if (points.size() < 4u) {
    return {};
  }

  std::array<real_t, 3> centroid{{0.0, 0.0, 0.0}};
  for (const auto& p : points) {
    centroid = add(centroid, p);
  }
  centroid = scale(centroid, real_t{1.0} / static_cast<real_t>(points.size()));

  std::map<std::string, std::vector<int>> hull_faces;
  const int n = static_cast<int>(points.size());
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      for (int k = j + 1; k < n; ++k) {
        const auto normal = cross(sub(points[static_cast<std::size_t>(j)], points[static_cast<std::size_t>(i)]),
                                  sub(points[static_cast<std::size_t>(k)], points[static_cast<std::size_t>(i)]));
        if (norm(normal) <= tol) {
          continue;
        }

        bool has_positive = false;
        bool has_negative = false;
        for (int q = 0; q < n; ++q) {
          if (q == i || q == j || q == k) {
            continue;
          }
          const real_t d = dot(normal, sub(points[static_cast<std::size_t>(q)],
                                           points[static_cast<std::size_t>(i)]));
          has_positive = has_positive || d > tol;
          has_negative = has_negative || d < -tol;
          if (has_positive && has_negative) {
            break;
          }
        }
        if (has_positive && has_negative) {
          continue;
        }

        std::vector<int> face;
        for (int q = 0; q < n; ++q) {
          const real_t d = dot(normal, sub(points[static_cast<std::size_t>(q)],
                                           points[static_cast<std::size_t>(i)]));
          if (std::abs(d) <= tol) {
            face.push_back(q);
          }
        }
        std::sort(face.begin(), face.end());
        std::ostringstream key;
        for (const int q : face) {
          key << q << ',';
        }
        hull_faces.emplace(key.str(), std::move(face));
      }
    }
  }

  std::vector<std::vector<int>> out;
  out.reserve(hull_faces.size());
  for (const auto& entry : hull_faces) {
    auto face = entry.second;
    if (face.size() < 3u) {
      continue;
    }

    std::array<real_t, 3> face_centroid{{0.0, 0.0, 0.0}};
    for (const int idx : face) {
      face_centroid = add(face_centroid, points[static_cast<std::size_t>(idx)]);
    }
    face_centroid = scale(face_centroid, real_t{1.0} / static_cast<real_t>(face.size()));

    std::array<real_t, 3> normal{{0.0, 0.0, 0.0}};
    for (std::size_t a = 0; a < face.size() && norm(normal) <= tol; ++a) {
      for (std::size_t b = a + 1; b < face.size() && norm(normal) <= tol; ++b) {
        for (std::size_t c = b + 1; c < face.size() && norm(normal) <= tol; ++c) {
          normal = cross(sub(points[static_cast<std::size_t>(face[b])],
                             points[static_cast<std::size_t>(face[a])]),
                         sub(points[static_cast<std::size_t>(face[c])],
                             points[static_cast<std::size_t>(face[a])]));
        }
      }
    }
    if (norm(normal) <= tol) {
      continue;
    }
    normal = unit_or_default(normal);
    if (dot(normal, sub(face_centroid, centroid)) < real_t{0.0}) {
      normal = scale(normal, real_t{-1.0});
    }

    auto tangent0 = std::abs(normal[0]) < real_t{0.9}
                        ? unit_or_default(cross(normal, {{1.0, 0.0, 0.0}}))
                        : unit_or_default(cross(normal, {{0.0, 1.0, 0.0}}));
    auto tangent1 = cross(normal, tangent0);
    std::sort(face.begin(), face.end(), [&](int a, int b) {
      const auto ra = sub(points[static_cast<std::size_t>(a)], face_centroid);
      const auto rb = sub(points[static_cast<std::size_t>(b)], face_centroid);
      return std::atan2(dot(ra, tangent1), dot(ra, tangent0)) <
             std::atan2(dot(rb, tangent1), dot(rb, tangent0));
    });
    out.push_back(std::move(face));
  }
  return out;
}

real_t convex_polyhedron_volume(
    const std::vector<std::array<real_t, 3>>& input_points,
    real_t tol) {
  tol = positive_tolerance(tol);
  const auto points = unique_points(input_points, tol);
  if (points.size() < 4u) {
    return real_t{0.0};
  }

  std::array<real_t, 3> centroid{{0.0, 0.0, 0.0}};
  for (const auto& p : points) {
    centroid = add(centroid, p);
  }
  centroid = scale(centroid, real_t{1.0} / static_cast<real_t>(points.size()));

  std::map<std::string, std::vector<int>> hull_faces;
  const int n = static_cast<int>(points.size());
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      for (int k = j + 1; k < n; ++k) {
        const auto normal = cross(sub(points[static_cast<std::size_t>(j)], points[static_cast<std::size_t>(i)]),
                                  sub(points[static_cast<std::size_t>(k)], points[static_cast<std::size_t>(i)]));
        if (norm(normal) <= tol) {
          continue;
        }
        bool has_positive = false;
        bool has_negative = false;
        for (int q = 0; q < n; ++q) {
          if (q == i || q == j || q == k) {
            continue;
          }
          const real_t d = dot(normal, sub(points[static_cast<std::size_t>(q)],
                                           points[static_cast<std::size_t>(i)]));
          has_positive = has_positive || d > tol;
          has_negative = has_negative || d < -tol;
          if (has_positive && has_negative) {
            break;
          }
        }
        if (has_positive && has_negative) {
          continue;
        }

        std::vector<int> face;
        for (int q = 0; q < n; ++q) {
          const real_t d = dot(normal, sub(points[static_cast<std::size_t>(q)],
                                           points[static_cast<std::size_t>(i)]));
          if (std::abs(d) <= tol) {
            face.push_back(q);
          }
        }
        std::sort(face.begin(), face.end());
        std::ostringstream key;
        for (const int q : face) {
          key << q << ',';
        }
        hull_faces.emplace(key.str(), std::move(face));
      }
    }
  }

  real_t volume6 = real_t{0.0};
  for (const auto& [_, face] : hull_faces) {
    if (face.size() < 3u) {
      continue;
    }
    std::array<real_t, 3> face_centroid{{0.0, 0.0, 0.0}};
    for (const int idx : face) {
      face_centroid = add(face_centroid, points[static_cast<std::size_t>(idx)]);
    }
    face_centroid = scale(face_centroid, real_t{1.0} / static_cast<real_t>(face.size()));

    std::array<real_t, 3> normal{{0.0, 0.0, 0.0}};
    for (std::size_t a = 0; a < face.size() && norm(normal) <= tol; ++a) {
      for (std::size_t b = a + 1; b < face.size() && norm(normal) <= tol; ++b) {
        for (std::size_t c = b + 1; c < face.size() && norm(normal) <= tol; ++c) {
          normal = cross(sub(points[static_cast<std::size_t>(face[b])],
                             points[static_cast<std::size_t>(face[a])]),
                         sub(points[static_cast<std::size_t>(face[c])],
                             points[static_cast<std::size_t>(face[a])]));
        }
      }
    }
    if (norm(normal) <= tol) {
      continue;
    }
    normal = unit_or_default(normal);
    if (dot(normal, sub(face_centroid, centroid)) < real_t{0.0}) {
      normal = scale(normal, real_t{-1.0});
    }

    auto tangent0 = std::abs(normal[0]) < real_t{0.9}
                        ? unit_or_default(cross(normal, {{1.0, 0.0, 0.0}}))
                        : unit_or_default(cross(normal, {{0.0, 1.0, 0.0}}));
    auto tangent1 = cross(normal, tangent0);
    std::vector<int> ordered = face;
    std::sort(ordered.begin(), ordered.end(), [&](int a, int b) {
      const auto ra = sub(points[static_cast<std::size_t>(a)], face_centroid);
      const auto rb = sub(points[static_cast<std::size_t>(b)], face_centroid);
      return std::atan2(dot(ra, tangent1), dot(ra, tangent0)) <
             std::atan2(dot(rb, tangent1), dot(rb, tangent0));
    });

    if (ordered.size() >= 3u) {
      const auto& p0 = points[static_cast<std::size_t>(ordered[0])];
      for (std::size_t i = 1; i + 1 < ordered.size(); ++i) {
        auto p1 = points[static_cast<std::size_t>(ordered[i])];
        auto p2 = points[static_cast<std::size_t>(ordered[i + 1u])];
        const auto tri_normal = cross(sub(p1, p0), sub(p2, p0));
        if (dot(tri_normal, normal) < real_t{0.0}) {
          std::swap(p1, p2);
        }
        volume6 += dot(p0, cross(p1, p2));
      }
    }
  }
  return std::abs(volume6) / real_t{6.0};
}

SideMeasureEstimate clipped_tet_measure(
    const std::array<std::array<real_t, 3>, 4>& tet,
    const std::array<real_t, 4>& values,
    CutTopologySide side,
    real_t tol) {
  SideMeasureEstimate out;
  out.parent_measure = std::abs(MeshGeometry::tet_volume(tet[0], tet[1], tet[2], tet[3]));
  const auto clipped = clipped_tet_points(tet, values, side, tol);
  out.measure = std::min(out.parent_measure, convex_polyhedron_volume(clipped, tol));
  out.fraction = out.parent_measure > real_t{0.0} ? out.measure / out.parent_measure : real_t{0.0};
  for (const auto& p : clipped) {
    out.centroid = add(out.centroid, p);
  }
  if (!clipped.empty()) {
    out.centroid = scale(out.centroid, real_t{1.0} / static_cast<real_t>(clipped.size()));
  }
  out.available = true;
  return out;
}

void add_tet_contribution(
    SideMeasureEstimate& total,
    const std::array<std::array<real_t, 3>, 4>& tet,
    const std::array<real_t, 4>& values,
    CutTopologySide side,
    real_t tol) {
  const auto part = clipped_tet_measure(tet, values, side, tol);
  total.parent_measure += part.parent_measure;
  total.centroid = weighted_centroid(total.centroid, total.measure, part.centroid, part.measure);
  total.measure += part.measure;
  total.available = total.available || part.available;
}

bool cell_uses_high_order_geometry(const MeshBase& mesh, index_t cell) {
  return mesh.geometry_order(cell) > 1;
}

TessellationConfig high_order_cut_tessellation_config(
    const MeshBase& mesh,
    index_t cell,
    Configuration cfg) {
  TessellationConfig config;
  config.configuration = cfg;
  const int order = std::max(1, mesh.geometry_order(cell));
  const int suggested = std::max(1, Tessellator::suggest_refinement_level(order));
  config.refinement_level = suggested;
  config.min_refinement_level = suggested;
  config.max_refinement_level = std::max(suggested, std::min(4, suggested + 2));
  config.adaptive = true;
  config.curvature_threshold = real_t{0.025};
  return config;
}

std::vector<std::array<real_t, 3>> tessellated_subcell_points(
    const TessellatedCell& tessellated,
    int subcell_index) {
  std::vector<std::array<real_t, 3>> points;
  if (subcell_index < 0 ||
      subcell_index + 1 >= static_cast<int>(tessellated.offsets.size())) {
    return points;
  }
  const int begin = tessellated.offsets[static_cast<std::size_t>(subcell_index)];
  const int end = tessellated.offsets[static_cast<std::size_t>(subcell_index + 1)];
  if (begin < 0 || end < begin ||
      end > static_cast<int>(tessellated.connectivity.size())) {
    return points;
  }
  points.reserve(static_cast<std::size_t>(end - begin));
  for (int i = begin; i < end; ++i) {
    const auto vertex_index = tessellated.connectivity[static_cast<std::size_t>(i)];
    if (vertex_index < 0 ||
        static_cast<std::size_t>(vertex_index) >= tessellated.vertices.size()) {
      points.clear();
      return points;
    }
    points.push_back(tessellated.vertices[static_cast<std::size_t>(vertex_index)]);
  }
  return points;
}

std::vector<std::array<real_t, 3>> tessellated_subcell_parametric_points(
    const TessellatedCell& tessellated,
    int subcell_index) {
  std::vector<std::array<real_t, 3>> points;
  if (subcell_index < 0 ||
      subcell_index + 1 >= static_cast<int>(tessellated.offsets.size())) {
    return points;
  }
  const int begin = tessellated.offsets[static_cast<std::size_t>(subcell_index)];
  const int end = tessellated.offsets[static_cast<std::size_t>(subcell_index + 1)];
  if (begin < 0 || end < begin ||
      end > static_cast<int>(tessellated.connectivity.size())) {
    return points;
  }
  points.reserve(static_cast<std::size_t>(end - begin));
  for (int i = begin; i < end; ++i) {
    const auto vertex_index = tessellated.connectivity[static_cast<std::size_t>(i)];
    if (vertex_index < 0 ||
        static_cast<std::size_t>(vertex_index) >= tessellated.parametric_vertices.size()) {
      points.clear();
      return points;
    }
    points.push_back(tessellated.parametric_vertices[static_cast<std::size_t>(vertex_index)]);
  }
  return points;
}

void add_polygon_measure_contribution(
    SideMeasureEstimate& total,
    const std::vector<std::array<real_t, 3>>& points,
    const std::vector<real_t>& values,
    CutTopologySide side,
    real_t tol) {
  if (points.size() < 3u || points.size() != values.size()) {
    return;
  }
  const real_t parent_measure = MeshGeometry::polygon_area(points);
  const auto clipped = clip_polygon_points(points, values, side, tol);
  const real_t measure = MeshGeometry::polygon_area(clipped);
  std::array<real_t, 3> centroid{{0.0, 0.0, 0.0}};
  for (const auto& p : clipped) {
    centroid = add(centroid, p);
  }
  if (!clipped.empty()) {
    centroid = scale(centroid, real_t{1.0} / static_cast<real_t>(clipped.size()));
  }
  total.parent_measure += parent_measure;
  total.centroid = weighted_centroid(total.centroid, total.measure, centroid, measure);
  total.measure += measure;
  total.available = total.available || parent_measure > positive_tolerance(tol);
}

void add_tessellated_measure_contribution(
    SideMeasureEstimate& total,
    const MeshBase& mesh,
    index_t cell,
    CellFamily family,
    const std::vector<std::array<real_t, 3>>& points,
    const std::vector<std::array<real_t, 3>>& parametric_points,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    CutTopologySide side,
    real_t tol) {
  if (points.empty()) {
    return;
  }
  std::vector<real_t> values;
  values.reserve(points.size());
  for (const auto& point : points) {
    values.push_back(embedded.signed_distance(point));
  }

  const bool have_parametric = parametric_points.size() == points.size();
  if (have_parametric) {
    const auto parent_iso =
        isoparametric_measure_for_subcell(mesh, cell, family, parametric_points, cfg, tol);
    IsoparametricMeasureResult clipped_iso;
    if (family == CellFamily::Line && points.size() >= 2u) {
      const auto clipped = clip_polygon_parametric_points(
          points, parametric_points, values, side, tol);
      if (clipped.size() >= 2u) {
        clipped_iso = isoparametric_line_measure(mesh, cell, clipped[0].xi, clipped[1].xi, cfg);
      } else {
        clipped_iso.available = true;
      }
    } else if ((family == CellFamily::Triangle ||
                family == CellFamily::Quad ||
                family == CellFamily::Polygon) &&
               points.size() >= 3u) {
      const auto clipped = clip_polygon_parametric_points(
          points, parametric_points, values, side, tol);
      std::vector<std::array<real_t, 3>> clipped_xi;
      clipped_xi.reserve(clipped.size());
      for (const auto& point : clipped) {
        clipped_xi.push_back(point.xi);
      }
      if (clipped_xi.size() >= 3u) {
        clipped_iso = isoparametric_polygon_measure(mesh, cell, clipped_xi, cfg);
      } else {
        clipped_iso.available = true;
      }
    } else if ((family == CellFamily::Tetra ||
                family == CellFamily::Hex ||
                family == CellFamily::Wedge ||
                family == CellFamily::Pyramid) &&
               points.size() >= 4u) {
      clipped_iso = clipped_isoparametric_measure_from_tets(
          mesh,
          cell,
          linear_cell_parametric_tets(family, points, parametric_points, tol),
          embedded,
          cfg,
          side,
          tol);
    }

    if (parent_iso.available && clipped_iso.available) {
      total.parent_measure += parent_iso.measure;
      total.centroid =
          weighted_centroid(total.centroid, total.measure, clipped_iso.centroid, clipped_iso.measure);
      total.measure += clipped_iso.measure;
      total.available = total.available || parent_iso.measure > positive_tolerance(tol);
      return;
    }
  }

  if (family == CellFamily::Line && points.size() >= 2u) {
    const real_t parent_measure = norm(sub(points[1], points[0]));
    const auto clipped = clip_polygon_points(points, values, side, tol);
    real_t measure = real_t{0.0};
    std::array<real_t, 3> centroid{{0.0, 0.0, 0.0}};
    if (clipped.size() >= 2u) {
      measure = norm(sub(clipped[1], clipped[0]));
      centroid = scale(add(clipped[0], clipped[1]), real_t{0.5});
    }
    total.parent_measure += parent_measure;
    total.centroid = weighted_centroid(total.centroid, total.measure, centroid, measure);
    total.measure += measure;
    total.available = total.available || parent_measure > positive_tolerance(tol);
    return;
  }

  if ((family == CellFamily::Triangle ||
       family == CellFamily::Quad ||
       family == CellFamily::Polygon) &&
      points.size() >= 3u) {
    add_polygon_measure_contribution(total, points, values, side, tol);
    return;
  }

  const auto tets = PolyhedronTessellation::linear_cell_tets(family, points);
  for (const auto& tet : tets) {
    std::array<real_t, 4> vals{{embedded.signed_distance(tet.vertices[0]),
                                embedded.signed_distance(tet.vertices[1]),
                                embedded.signed_distance(tet.vertices[2]),
                                embedded.signed_distance(tet.vertices[3])}};
    add_tet_contribution(total, tet.vertices, vals, side, tol);
  }
}

SideMeasureEstimate estimate_side_measure_from_tessellated_cell(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    CutTopologySide side,
    real_t tol) {
  SideMeasureEstimate out;
  try {
    const auto tessellated =
        Tessellator::tessellate_cell(mesh, cell, high_order_cut_tessellation_config(mesh, cell, cfg));
    for (int i = 0; i < tessellated.n_sub_elements(); ++i) {
      add_tessellated_measure_contribution(out,
                                           mesh,
                                           cell,
                                           tessellated.sub_element_shape.family,
                                           tessellated_subcell_points(tessellated, i),
                                           tessellated_subcell_parametric_points(tessellated, i),
                                           embedded,
                                           cfg,
                                           side,
                                           tol);
    }
    out.used_tessellation = out.available;
  } catch (const std::exception&) {
    out.available = false;
  }
  out.fraction = out.parent_measure > real_t{0.0}
                     ? std::max(real_t{0.0}, std::min(real_t{1.0}, out.measure / out.parent_measure))
                     : real_t{0.0};
  return out;
}

SideMeasureEstimate estimate_side_measure_for_cell(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    CutTopologySide side,
    real_t tol) {
  SideMeasureEstimate out;
  const auto shape = mesh.cell_shape(cell);
  const auto [vptr, n_vertices_span] = mesh.cell_vertices_span(cell);
  const std::size_t n_corners =
      shape.num_corners > 0
          ? std::min<std::size_t>(static_cast<std::size_t>(shape.num_corners), n_vertices_span)
          : n_vertices_span;
  if (n_corners == 0u) {
    return out;
  }

  if (cell_uses_high_order_geometry(mesh, cell)) {
    return estimate_side_measure_from_tessellated_cell(mesh, cell, embedded, cfg, side, tol);
  }

  std::vector<std::array<real_t, 3>> points;
  std::vector<real_t> values;
  points.reserve(n_corners);
  values.reserve(n_corners);
  for (std::size_t i = 0; i < n_corners; ++i) {
    const auto p = mesh.geometry_dof_coords(vptr[i], cfg);
    points.push_back(p);
    values.push_back(embedded.signed_distance(p));
  }

  if (shape.family == CellFamily::Line && points.size() >= 2u) {
    out.parent_measure = norm(sub(points[1], points[0]));
    const auto clipped = clip_polygon_points(points, values, side, tol);
    if (clipped.size() >= 2u) {
      out.measure = norm(sub(clipped[1], clipped[0]));
      out.centroid = scale(add(clipped[0], clipped[1]), real_t{0.5});
    }
    out.fraction = out.parent_measure > real_t{0.0} ? out.measure / out.parent_measure : real_t{0.0};
    out.available = true;
    return out;
  }

  if ((shape.family == CellFamily::Triangle ||
       shape.family == CellFamily::Quad ||
       shape.family == CellFamily::Polygon) &&
      points.size() >= 3u) {
    out.parent_measure = MeshGeometry::polygon_area(points);
    const auto clipped = clip_polygon_points(points, values, side, tol);
    out.measure = MeshGeometry::polygon_area(clipped);
    out.fraction = out.parent_measure > real_t{0.0} ? out.measure / out.parent_measure : real_t{0.0};
    for (const auto& p : clipped) {
      out.centroid = add(out.centroid, p);
    }
    if (!clipped.empty()) {
      out.centroid = scale(out.centroid, real_t{1.0} / static_cast<real_t>(clipped.size()));
    }
    out.available = true;
    return out;
  }

  if (shape.family == CellFamily::Tetra ||
      shape.family == CellFamily::Hex ||
      shape.family == CellFamily::Wedge ||
      shape.family == CellFamily::Pyramid ||
      shape.family == CellFamily::Polyhedron) {
    const auto tets = PolyhedronTessellation::linear_cell_tets(mesh, cell, cfg);
    for (const auto& tet : tets) {
      std::array<real_t, 4> vals{{embedded.signed_distance(tet.vertices[0]),
                                  embedded.signed_distance(tet.vertices[1]),
                                  embedded.signed_distance(tet.vertices[2]),
                                  embedded.signed_distance(tet.vertices[3])}};
      add_tet_contribution(out, tet.vertices, vals, side, tol);
    }
    out.used_tessellation = !tets.empty();
  }

  if (out.parent_measure <= real_t{0.0}) {
    out.parent_measure = MeshGeometry::cell_measure(mesh, cell, cfg);
  }
  out.fraction = out.parent_measure > real_t{0.0}
                     ? std::max(real_t{0.0}, std::min(real_t{1.0}, out.measure / out.parent_measure))
                     : real_t{0.0};
  return out;
}

std::uint64_t stable_parent_dof_region_id(
    const EmbeddedRegionProvenance& provenance,
    gid_t parent_gid,
    std::uint64_t dof_key,
    CutTopologySide side) noexcept {
  return stable_entity_id(provenance,
                          parent_gid,
                          dof_key,
                          static_cast<index_t>(side),
                          real_t{0.0},
                          505u);
}

std::uint64_t stable_region_point_id(
    const EmbeddedRegionProvenance& provenance,
    gid_t parent_gid,
    const std::array<real_t, 3>& point,
    CutTopologySide side,
    std::uint64_t salt) noexcept {
  std::uint64_t h = kFnvOffset;
  h = append_hash_string(h, provenance.persistent_id);
  h = append_hash_string(h, provenance.name);
  h = append_hash(h, static_cast<std::uint64_t>(parent_gid));
  h = append_hash(h, static_cast<std::uint64_t>(side));
  h = append_hash_real(h, point[0]);
  h = append_hash_real(h, point[1]);
  h = append_hash_real(h, point[2]);
  h = append_hash(h, salt);
  return h;
}

std::uint64_t stable_subcell_id(
    const EmbeddedRegionProvenance& provenance,
    gid_t parent_gid,
    CutTopologySide side,
    CellFamily family,
    const std::vector<std::uint64_t>& vertices,
    std::uint64_t salt) noexcept {
  std::uint64_t h = kFnvOffset;
  h = append_hash_string(h, provenance.persistent_id);
  h = append_hash_string(h, provenance.name);
  h = append_hash(h, static_cast<std::uint64_t>(parent_gid));
  h = append_hash(h, static_cast<std::uint64_t>(side));
  h = append_hash(h, static_cast<std::uint64_t>(family));
  h = append_hash(h, salt);
  for (const auto id : vertices) {
    h = append_hash(h, id);
  }
  return h;
}

index_t matching_parent_dof(
    const std::vector<index_t>& dofs,
    const std::vector<std::array<real_t, 3>>& points,
    const std::array<real_t, 3>& point,
    real_t tol) noexcept {
  for (std::size_t i = 0; i < dofs.size() && i < points.size(); ++i) {
    if (norm(sub(points[i], point)) <= positive_tolerance(tol)) {
      return dofs[i];
    }
  }
  return INVALID_INDEX;
}

std::uint64_t stable_curved_patch_id(
    const EmbeddedRegionProvenance& provenance,
    gid_t parent_gid,
    const std::vector<std::uint64_t>& vertices,
    std::uint64_t salt) noexcept {
  std::uint64_t h = kFnvOffset;
  h = append_hash_string(h, provenance.persistent_id);
  h = append_hash_string(h, provenance.name);
  h = append_hash(h, static_cast<std::uint64_t>(parent_gid));
  h = append_hash(h, salt);
  for (const auto id : vertices) {
    h = append_hash(h, id);
  }
  return h;
}

struct CurvedPatchPoint {
  std::array<real_t, 3> point{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> parent_parametric_coordinate{{0.0, 0.0, 0.0}};
  bool has_parent_parametric_coordinate{false};
};

struct CurvedPatchQuadratureSample {
  std::array<real_t, 3> point{{0.0, 0.0, 0.0}};
  std::array<real_t, 3> normal{{1.0, 0.0, 0.0}};
  real_t weight{0.0};
};

std::array<real_t, 3> centroid_of(
    const std::vector<std::array<real_t, 3>>& points) noexcept {
  std::array<real_t, 3> centroid{{0.0, 0.0, 0.0}};
  for (const auto& p : points) {
    centroid = add(centroid, p);
  }
  if (!points.empty()) {
    centroid = scale(centroid, real_t{1.0} / static_cast<real_t>(points.size()));
  }
  return centroid;
}

std::vector<CurvedPatchPoint> unique_curved_patch_points(
    std::vector<CurvedPatchPoint> points,
    real_t tol) {
  const real_t eps = positive_tolerance(tol);
  std::vector<CurvedPatchPoint> out;
  for (const auto& p : points) {
    auto existing = std::find_if(out.begin(), out.end(), [&](const auto& q) {
      return norm(sub(p.point, q.point)) <= eps;
    });
    if (existing == out.end()) {
      out.push_back(p);
    } else if (!existing->has_parent_parametric_coordinate &&
               p.has_parent_parametric_coordinate) {
      existing->parent_parametric_coordinate = p.parent_parametric_coordinate;
      existing->has_parent_parametric_coordinate = true;
    }
  }
  return out;
}

std::vector<std::array<real_t, 3>> ordered_interface_points(
    std::vector<std::array<real_t, 3>> points,
    const std::array<real_t, 3>& normal,
    real_t tol) {
  points = unique_points(std::move(points), tol);
  if (points.size() < 3u) {
    return points;
  }

  const auto centroid = centroid_of(points);
  const auto n = unit_or_default(normal);
  auto tangent0 = std::abs(n[0]) < real_t{0.9}
                      ? unit_or_default(cross(n, {{1.0, 0.0, 0.0}}))
                      : unit_or_default(cross(n, {{0.0, 1.0, 0.0}}));
  auto tangent1 = cross(n, tangent0);
  std::sort(points.begin(), points.end(), [&](const auto& a, const auto& b) {
    const auto ra = sub(a, centroid);
    const auto rb = sub(b, centroid);
    return std::atan2(dot(ra, tangent1), dot(ra, tangent0)) <
           std::atan2(dot(rb, tangent1), dot(rb, tangent0));
  });
  return points;
}

std::vector<CurvedPatchPoint> ordered_curved_patch_points(
    std::vector<CurvedPatchPoint> points,
    const std::array<real_t, 3>& normal,
    real_t tol) {
  points = unique_curved_patch_points(std::move(points), tol);
  if (points.size() < 3u) {
    return points;
  }

  std::vector<std::array<real_t, 3>> physical_points;
  physical_points.reserve(points.size());
  for (const auto& p : points) {
    physical_points.push_back(p.point);
  }
  const auto centroid = centroid_of(physical_points);
  const auto n = unit_or_default(normal);
  auto tangent0 = std::abs(n[0]) < real_t{0.9}
                      ? unit_or_default(cross(n, {{1.0, 0.0, 0.0}}))
                      : unit_or_default(cross(n, {{0.0, 1.0, 0.0}}));
  auto tangent1 = cross(n, tangent0);
  std::sort(points.begin(), points.end(), [&](const auto& a, const auto& b) {
    const auto ra = sub(a.point, centroid);
    const auto rb = sub(b.point, centroid);
    return std::atan2(dot(ra, tangent1), dot(ra, tangent0)) <
           std::atan2(dot(rb, tangent1), dot(rb, tangent0));
  });
  return points;
}

void add_curved_line_patch_quadrature(
    std::vector<CurvedPatchQuadratureSample>& samples,
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    const std::array<real_t, 3>& a,
    const std::array<real_t, 3>& b) {
  constexpr real_t g = real_t{0.57735026918962576451};
  const std::array<real_t, 2> points{{-g, g}};
  const auto half_delta = scale(sub(b, a), real_t{0.5});
  for (const auto s : points) {
    const auto xi = interpolate_parametric(a, b, (s + real_t{1.0}) * real_t{0.5});
    const auto eval = CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, cfg);
    const real_t weight = norm(jacobian_action(eval.jacobian, half_delta));
    if (weight <= real_t{0.0} || !std::isfinite(weight)) {
      continue;
    }
    samples.push_back({eval.coordinates, embedded.outward_normal(eval.coordinates), weight});
  }
}

void add_curved_triangle_patch_quadrature(
    std::vector<CurvedPatchQuadratureSample>& samples,
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    const std::array<real_t, 3>& a,
    const std::array<real_t, 3>& b,
    const std::array<real_t, 3>& c) {
  constexpr std::array<std::array<real_t, 2>, 3> q{{
      {{real_t{1.0} / real_t{6.0}, real_t{1.0} / real_t{6.0}}},
      {{real_t{2.0} / real_t{3.0}, real_t{1.0} / real_t{6.0}}},
      {{real_t{1.0} / real_t{6.0}, real_t{2.0} / real_t{3.0}}}}};
  const auto db = sub(b, a);
  const auto dc = sub(c, a);
  for (const auto& rs : q) {
    const auto xi = interpolate_parametric(a, b, c, rs[0], rs[1]);
    const auto eval = CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, cfg);
    const auto dx_dr = jacobian_action(eval.jacobian, db);
    const auto dx_ds = jacobian_action(eval.jacobian, dc);
    auto normal = cross(dx_dr, dx_ds);
    const real_t weight = norm(normal) / real_t{6.0};
    if (weight <= real_t{0.0} || !std::isfinite(weight)) {
      continue;
    }
    normal = unit_or_default(normal);
    const auto embedded_normal = embedded.outward_normal(eval.coordinates);
    if (dot(normal, embedded_normal) < real_t{0.0}) {
      normal = scale(normal, real_t{-1.0});
    }
    samples.push_back({eval.coordinates, normal, weight});
  }
}

void populate_curved_patch_isoparametric_quadrature(
    CutCurvedPatchRecord& patch,
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg) {
  patch.quadrature_points.clear();
  patch.quadrature_normals.clear();
  patch.quadrature_weights.clear();
  patch.quadrature_measure = real_t{0.0};
  patch.isoparametric_quadrature_available = false;
  if (!patch.parametric_coordinates_valid ||
      patch.parent_parametric_coordinates.size() != patch.physical_points.size() ||
      patch.parent_parametric_coordinates.empty()) {
    return;
  }

  std::vector<CurvedPatchQuadratureSample> samples;
  const auto topo_kind = mesh.cell_shape(cell).topo_kind();
  try {
    if (topo_kind == EntityKind::Edge && patch.parent_parametric_coordinates.size() >= 1u) {
      const auto& point = patch.physical_points.front();
      samples.push_back({point, embedded.outward_normal(point), real_t{1.0}});
    } else if (topo_kind == EntityKind::Face && patch.parent_parametric_coordinates.size() >= 2u) {
      for (std::size_t i = 0; i + 1 < patch.parent_parametric_coordinates.size(); ++i) {
        add_curved_line_patch_quadrature(samples,
                                         mesh,
                                         cell,
                                         embedded,
                                         cfg,
                                         patch.parent_parametric_coordinates[i],
                                         patch.parent_parametric_coordinates[i + 1u]);
      }
    } else if (topo_kind == EntityKind::Volume && patch.parent_parametric_coordinates.size() >= 3u) {
      for (std::size_t i = 1; i + 1 < patch.parent_parametric_coordinates.size(); ++i) {
        add_curved_triangle_patch_quadrature(samples,
                                            mesh,
                                            cell,
                                            embedded,
                                            cfg,
                                            patch.parent_parametric_coordinates[0],
                                            patch.parent_parametric_coordinates[i],
                                            patch.parent_parametric_coordinates[i + 1u]);
      }
    }
  } catch (const std::exception&) {
    samples.clear();
  }

  if (samples.empty()) {
    return;
  }
  patch.quadrature_points.reserve(samples.size());
  patch.quadrature_normals.reserve(samples.size());
  patch.quadrature_weights.reserve(samples.size());
  for (const auto& sample : samples) {
    patch.quadrature_points.push_back(sample.point);
    patch.quadrature_normals.push_back(unit_or_default(sample.normal));
    patch.quadrature_weights.push_back(sample.weight);
    patch.quadrature_measure += sample.weight;
  }
  patch.isoparametric_quadrature_available = patch.quadrature_measure > real_t{0.0};
  if (patch.isoparametric_quadrature_available) {
    if (patch.construction_policy == "tessellated-curved-linearized-arrangement") {
      patch.construction_policy = "curved-isoparametric-topology-subdivision";
    }
  }
}

void add_interface_edge_crossing(
    std::vector<std::array<real_t, 3>>& points,
    const std::array<real_t, 3>& a,
    const std::array<real_t, 3>& b,
    real_t da,
    real_t db,
    real_t tol) {
  const real_t eps = positive_tolerance(tol);
  if (std::abs(da) <= eps) {
    points.push_back(a);
  }
  if (std::abs(db) <= eps) {
    points.push_back(b);
  }
  const bool crosses = (da < -eps && db > eps) || (da > eps && db < -eps);
  if (!crosses || std::abs(da - db) <= eps) {
    return;
  }
  real_t t = da / (da - db);
  t = std::max(real_t{0.0}, std::min(real_t{1.0}, t));
  points.push_back(add(scale(a, real_t{1.0} - t), scale(b, t)));
}

void add_interface_edge_crossing(
    std::vector<CurvedPatchPoint>& points,
    const CurvedPatchPoint& a,
    const CurvedPatchPoint& b,
    real_t da,
    real_t db,
    real_t tol) {
  const real_t eps = positive_tolerance(tol);
  if (std::abs(da) <= eps) {
    points.push_back(a);
  }
  if (std::abs(db) <= eps) {
    points.push_back(b);
  }
  const bool crosses = (da < -eps && db > eps) || (da > eps && db < -eps);
  if (!crosses || std::abs(da - db) <= eps) {
    return;
  }
  real_t t = da / (da - db);
  t = std::max(real_t{0.0}, std::min(real_t{1.0}, t));

  CurvedPatchPoint hit;
  hit.point = add(scale(a.point, real_t{1.0} - t), scale(b.point, t));
  if (a.has_parent_parametric_coordinate && b.has_parent_parametric_coordinate) {
    hit.parent_parametric_coordinate =
        add(scale(a.parent_parametric_coordinate, real_t{1.0} - t),
            scale(b.parent_parametric_coordinate, t));
    hit.has_parent_parametric_coordinate = true;
  }
  points.push_back(hit);
}

std::vector<std::array<real_t, 3>> interface_points_from_polyline_or_polygon(
    const std::vector<std::array<real_t, 3>>& points,
    const std::vector<real_t>& values,
    bool closed,
    real_t tol) {
  std::vector<std::array<real_t, 3>> out;
  if (points.size() < 2u || points.size() != values.size()) {
    return out;
  }
  const std::size_t edge_count = closed ? points.size() : points.size() - 1u;
  for (std::size_t i = 0; i < edge_count; ++i) {
    const std::size_t j = (i + 1u) % points.size();
    add_interface_edge_crossing(out, points[i], points[j], values[i], values[j], tol);
  }
  return unique_points(std::move(out), tol);
}

std::vector<CurvedPatchPoint> interface_points_from_polyline_or_polygon(
    const std::vector<std::array<real_t, 3>>& points,
    const std::vector<std::array<real_t, 3>>& parent_parametric_points,
    const std::vector<real_t>& values,
    bool closed,
    real_t tol) {
  std::vector<CurvedPatchPoint> out;
  if (points.size() < 2u ||
      points.size() != values.size() ||
      points.size() != parent_parametric_points.size()) {
    return out;
  }
  const std::size_t edge_count = closed ? points.size() : points.size() - 1u;
  for (std::size_t i = 0; i < edge_count; ++i) {
    const std::size_t j = (i + 1u) % points.size();
    CurvedPatchPoint a;
    a.point = points[i];
    a.parent_parametric_coordinate = parent_parametric_points[i];
    a.has_parent_parametric_coordinate = true;
    CurvedPatchPoint b;
    b.point = points[j];
    b.parent_parametric_coordinate = parent_parametric_points[j];
    b.has_parent_parametric_coordinate = true;
    add_interface_edge_crossing(out, a, b, values[i], values[j], tol);
  }
  return unique_curved_patch_points(std::move(out), tol);
}

std::vector<std::array<real_t, 3>> interface_points_from_tet(
    const std::array<std::array<real_t, 3>, 4>& tet,
    const std::array<real_t, 4>& values,
    real_t tol) {
  std::vector<std::array<real_t, 3>> out;
  constexpr std::array<std::array<int, 2>, 6> edges{{
      {{0, 1}}, {{0, 2}}, {{0, 3}}, {{1, 2}}, {{1, 3}}, {{2, 3}}}};
  for (const auto& edge : edges) {
    const int a = edge[0];
    const int b = edge[1];
    add_interface_edge_crossing(out,
                                tet[static_cast<std::size_t>(a)],
                                tet[static_cast<std::size_t>(b)],
                                values[static_cast<std::size_t>(a)],
                                values[static_cast<std::size_t>(b)],
                                tol);
  }
  return unique_points(std::move(out), tol);
}

std::vector<CurvedPatchPoint> interface_points_from_tet(
    const std::array<std::array<real_t, 3>, 4>& tet,
    const std::array<std::array<real_t, 3>, 4>& parent_parametric_points,
    const std::array<real_t, 4>& values,
    real_t tol) {
  std::vector<CurvedPatchPoint> out;
  constexpr std::array<std::array<int, 2>, 6> edges{{
      {{0, 1}}, {{0, 2}}, {{0, 3}}, {{1, 2}}, {{1, 3}}, {{2, 3}}}};
  for (const auto& edge : edges) {
    const int ai = edge[0];
    const int bi = edge[1];
    CurvedPatchPoint a;
    a.point = tet[static_cast<std::size_t>(ai)];
    a.parent_parametric_coordinate = parent_parametric_points[static_cast<std::size_t>(ai)];
    a.has_parent_parametric_coordinate = true;
    CurvedPatchPoint b;
    b.point = tet[static_cast<std::size_t>(bi)];
    b.parent_parametric_coordinate = parent_parametric_points[static_cast<std::size_t>(bi)];
    b.has_parent_parametric_coordinate = true;
    add_interface_edge_crossing(out,
                                a,
                                b,
                                values[static_cast<std::size_t>(ai)],
                                values[static_cast<std::size_t>(bi)],
                                tol);
  }
  return unique_curved_patch_points(std::move(out), tol);
}

void populate_parent_parametric_coordinate(
    CutTopologyVertex& vertex,
    const MeshBase& mesh,
    index_t cell,
    Configuration cfg,
    real_t tol) {
  vertex.has_parent_parametric_coordinate = false;
  vertex.parent_parametric_coordinate_valid = false;
  vertex.parent_parametric_residual = real_t{0.0};
  try {
    const auto inverse =
        CurvilinearEvaluator::inverse_map(mesh, cell, vertex.point, cfg, positive_tolerance(tol), 80);
    vertex.parent_parametric_coordinate = inverse.first;
    vertex.has_parent_parametric_coordinate = true;
    const auto eval =
        CurvilinearEvaluator::evaluate_geometry(mesh, cell, vertex.parent_parametric_coordinate, cfg);
    vertex.parent_parametric_residual = norm(sub(eval.coordinates, vertex.point));
    vertex.parent_parametric_coordinate_valid =
        inverse.second &&
        CurvilinearEvaluator::is_inside_reference_element(
            mesh.cell_shape(cell), vertex.parent_parametric_coordinate, positive_tolerance(tol) * 100.0);
  } catch (const std::exception&) {
    vertex.parent_parametric_coordinate = {{0.0, 0.0, 0.0}};
    vertex.has_parent_parametric_coordinate = false;
    vertex.parent_parametric_coordinate_valid = false;
    vertex.parent_parametric_residual = real_t{0.0};
  }
}

std::uint64_t upsert_curved_topology_vertex(
    CutTopologyRecord& topology,
    CutTopologyVertex vertex) {
  auto existing = std::find_if(topology.vertices.begin(),
                               topology.vertices.end(),
                               [&](const auto& v) { return v.stable_id == vertex.stable_id; });
  if (existing != topology.vertices.end()) {
    if (vertex.has_parent_parametric_coordinate &&
        !existing->has_parent_parametric_coordinate) {
      existing->parent_parametric_coordinate = vertex.parent_parametric_coordinate;
      existing->parent_parametric_residual = vertex.parent_parametric_residual;
      existing->has_parent_parametric_coordinate = true;
      existing->parent_parametric_coordinate_valid = vertex.parent_parametric_coordinate_valid;
    }
    return existing->stable_id;
  }
  topology.vertices.push_back(std::move(vertex));
  return topology.vertices.back().stable_id;
}

bool record_curved_patch_from_candidates(
    CutTopologyRecord& topology,
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    const EmbeddedRegionProvenance& provenance,
    gid_t parent_gid,
    std::vector<CurvedPatchPoint> candidates,
    real_t tol,
    std::uint64_t salt,
    bool exact_topology_available = false,
    std::string construction_policy = "tessellated-curved-linearized-arrangement") {
  const auto parent_shape = mesh.cell_shape(cell);
  const auto parent_kind = parent_shape.topo_kind();
  const std::size_t min_vertices =
      parent_kind == EntityKind::Volume ? 3u
      : parent_kind == EntityKind::Face ? 2u
                                        : 1u;
  if (candidates.size() < min_vertices) {
    return false;
  }

  std::vector<std::array<real_t, 3>> candidate_points;
  candidate_points.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    candidate_points.push_back(candidate.point);
  }
  const auto centroid = centroid_of(candidate_points);
  candidates = ordered_curved_patch_points(std::move(candidates), embedded.outward_normal(centroid), tol);
  if (candidates.size() < min_vertices) {
    return false;
  }

  CutCurvedPatchRecord patch;
  patch.parent_cell = cell;
  patch.parent_cell_gid = parent_gid;
  patch.parent_family = parent_shape.family;
  patch.geometry_order = mesh.geometry_order(cell);
  patch.embedded_kind = embedded.kind;
  patch.configuration = cfg;
  patch.provenance = provenance;
  patch.exact_topology_available = exact_topology_available;
  patch.linearized_surrogate = !exact_topology_available;
  patch.parametric_coordinates_valid = true;
  patch.construction_policy = std::move(construction_policy);

  patch.ordered_vertices.reserve(candidates.size());
  patch.parent_parametric_coordinates.reserve(candidates.size());
  patch.physical_points.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    CutTopologyVertex vertex;
    vertex.stable_id = stable_region_point_id(
        provenance, parent_gid, candidate.point, CutTopologySide::Interface, 707u);
    vertex.point = candidate.point;
    vertex.normal = embedded.outward_normal(candidate.point);
    vertex.parent_cell = cell;
    vertex.parent_cell_gid = parent_gid;
    vertex.provenance = provenance;
    if (candidate.has_parent_parametric_coordinate) {
      vertex.parent_parametric_coordinate = candidate.parent_parametric_coordinate;
      vertex.has_parent_parametric_coordinate = true;
      vertex.parent_parametric_coordinate_valid =
          CurvilinearEvaluator::is_inside_reference_element(
              mesh.cell_shape(cell), vertex.parent_parametric_coordinate, positive_tolerance(tol) * 100.0);
      try {
        const auto eval =
            CurvilinearEvaluator::evaluate_geometry(mesh, cell, vertex.parent_parametric_coordinate, cfg);
        vertex.parent_parametric_residual = norm(sub(eval.coordinates, vertex.point));
      } catch (const std::exception&) {
        vertex.parent_parametric_coordinate_valid = false;
        vertex.parent_parametric_residual = real_t{0.0};
      }
    } else {
      populate_parent_parametric_coordinate(vertex, mesh, cell, cfg, tol);
    }
    patch.ordered_vertices.push_back(upsert_curved_topology_vertex(topology, vertex));
    patch.parent_parametric_coordinates.push_back(vertex.parent_parametric_coordinate);
    patch.physical_points.push_back(vertex.point);
    patch.parametric_coordinates_valid =
        patch.parametric_coordinates_valid && vertex.parent_parametric_coordinate_valid;
    patch.max_parent_parametric_residual =
        std::max(patch.max_parent_parametric_residual, vertex.parent_parametric_residual);
  }

  patch.stable_id = stable_curved_patch_id(provenance, parent_gid, patch.ordered_vertices, salt);
  populate_curved_patch_isoparametric_quadrature(patch, mesh, cell, embedded, cfg);

  CutInterfacePolygon polygon;
  polygon.stable_id = patch.stable_id;
  polygon.parent_cell = cell;
  polygon.parent_cell_gid = parent_gid;
  polygon.ordered_vertices = patch.ordered_vertices;
  polygon.normal = embedded.outward_normal(centroid_of(patch.physical_points));
  polygon.provenance = provenance;
  topology.interface_polygons.push_back(std::move(polygon));
  topology.curved_patches.push_back(std::move(patch));
  return true;
}

bool record_curved_patch_from_points(
    CutTopologyRecord& topology,
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    const EmbeddedRegionProvenance& provenance,
    gid_t parent_gid,
    std::vector<std::array<real_t, 3>> points,
    real_t tol,
    std::uint64_t salt,
    bool exact_topology_available = false,
    std::string construction_policy = "tessellated-curved-linearized-arrangement") {
  std::vector<CurvedPatchPoint> candidates;
  candidates.reserve(points.size());
  for (const auto& point : points) {
    CurvedPatchPoint candidate;
    candidate.point = point;
    candidates.push_back(candidate);
  }
  return record_curved_patch_from_candidates(topology,
                                             mesh,
                                             cell,
                                             embedded,
                                             cfg,
                                             provenance,
                                             parent_gid,
                                             std::move(candidates),
                                             tol,
                                             salt,
                                             exact_topology_available,
                                             std::move(construction_policy));
}

std::size_t add_curved_patches_from_linear_subcell(
    CutTopologyRecord& topology,
    const MeshBase& mesh,
    index_t cell,
    CellFamily family,
    const std::vector<std::array<real_t, 3>>& points,
    const std::vector<std::array<real_t, 3>>& parent_parametric_points,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    const EmbeddedRegionProvenance& provenance,
    gid_t parent_gid,
    real_t tol,
    std::uint64_t salt,
    bool exact_topology_available = false,
    std::string construction_policy = "tessellated-curved-linearized-arrangement") {
  if (points.empty()) {
    return 0u;
  }
  std::vector<real_t> values;
  values.reserve(points.size());
  for (const auto& point : points) {
    values.push_back(embedded.signed_distance(point));
  }

  std::size_t added = 0u;
  if (family == CellFamily::Line && points.size() >= 2u) {
    added += record_curved_patch_from_candidates(
                 topology,
                 mesh,
                 cell,
                 embedded,
                 cfg,
                 provenance,
                 parent_gid,
                 interface_points_from_polyline_or_polygon(
                     points, parent_parametric_points, values, false, tol),
                 tol,
                 salt,
                 exact_topology_available,
                 construction_policy)
                 ? 1u
                 : 0u;
    return added;
  }

  if ((family == CellFamily::Triangle ||
       family == CellFamily::Quad ||
       family == CellFamily::Polygon) &&
      points.size() >= 3u) {
    added += record_curved_patch_from_candidates(
                 topology,
                 mesh,
                 cell,
                 embedded,
                 cfg,
                 provenance,
                 parent_gid,
                 interface_points_from_polyline_or_polygon(
                     points, parent_parametric_points, values, true, tol),
                 tol,
                 salt,
                 exact_topology_available,
                 construction_policy)
                 ? 1u
                 : 0u;
    return added;
  }

  const auto add_tet_patch =
      [&](const std::array<std::array<real_t, 3>, 4>& tet,
          const std::array<std::array<real_t, 3>, 4>* tet_parametric_points,
          std::uint64_t tet_salt) {
        std::array<real_t, 4> vals{{embedded.signed_distance(tet[0]),
                                    embedded.signed_distance(tet[1]),
                                    embedded.signed_distance(tet[2]),
                                    embedded.signed_distance(tet[3])}};
        const bool recorded =
            tet_parametric_points
                ? record_curved_patch_from_candidates(
                      topology,
                      mesh,
                      cell,
                      embedded,
                      cfg,
                      provenance,
                      parent_gid,
                      interface_points_from_tet(tet, *tet_parametric_points, vals, tol),
                      tol,
                      tet_salt,
                      exact_topology_available,
                      construction_policy)
                : record_curved_patch_from_points(topology,
                                                  mesh,
                                                  cell,
                                                  embedded,
                                                  cfg,
                                                  provenance,
                                                  parent_gid,
                                                  interface_points_from_tet(tet, vals, tol),
                                                  tol,
                                                  tet_salt,
                                                  exact_topology_available,
                                                  construction_policy);
        if (recorded) {
          ++added;
        }
      };

  if ((family == CellFamily::Tetra ||
       family == CellFamily::Hex ||
       family == CellFamily::Wedge ||
       family == CellFamily::Pyramid) &&
      points.size() >= 4u &&
      parent_parametric_points.size() == points.size()) {
    const auto tets = linear_cell_parametric_tets(family, points, parent_parametric_points, tol);
    for (std::size_t i = 0; i < tets.size(); ++i) {
      add_tet_patch(tets[i].points,
                    &tets[i].parent_parametric_points,
                    salt + static_cast<std::uint64_t>(i));
    }
    return added;
  }

  const auto tets = PolyhedronTessellation::linear_cell_tets(family, points);
  for (std::size_t i = 0; i < tets.size(); ++i) {
    add_tet_patch(tets[i].vertices, nullptr, salt + static_cast<std::uint64_t>(i));
  }
  return added;
}

std::size_t add_curved_patches_from_tessellated_cell(
    CutTopologyRecord& topology,
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    const EmbeddedRegionProvenance& provenance,
    gid_t parent_gid,
    real_t tol,
    bool exact_topology_available = false,
    std::string construction_policy = "tessellated-curved-linearized-arrangement") {
  if (!cell_uses_high_order_geometry(mesh, cell)) {
    return 0u;
  }

  std::size_t added = 0u;
  try {
    const auto tessellated =
        Tessellator::tessellate_cell(mesh, cell, high_order_cut_tessellation_config(mesh, cell, cfg));
    for (int i = 0; i < tessellated.n_sub_elements(); ++i) {
      added += add_curved_patches_from_linear_subcell(
          topology,
          mesh,
          cell,
          tessellated.sub_element_shape.family,
          tessellated_subcell_points(tessellated, i),
          tessellated_subcell_parametric_points(tessellated, i),
          embedded,
          cfg,
          provenance,
          parent_gid,
          tol,
          1300u + static_cast<std::uint64_t>(i) * 97u,
          exact_topology_available,
          construction_policy);
    }
  } catch (const std::exception& e) {
    topology.diagnostics.push_back(
        std::string("curved cut patch reconstruction failed: ") + e.what());
  }
  return added;
}

struct SideClosedTopology {
  std::vector<CutIntegrationVertex> vertices{};
  std::vector<CutIntegrationSubcell> subcells{};
  std::vector<std::uint64_t> vertex_ids{};
  std::vector<std::vector<std::uint64_t>> faces{};
  bool closed{false};
};

std::uint64_t add_integration_vertex(
    SideClosedTopology& topology,
    const std::array<real_t, 3>& point,
    index_t parent_dof,
    bool on_interface,
    bool synthetic,
    const EmbeddedRegionProvenance& provenance,
    gid_t parent_gid,
    CutTopologySide side,
    const std::array<real_t, 3>* parent_parametric_coordinate = nullptr,
    bool parent_parametric_coordinate_valid = false,
    bool curved_isoparametric = false) {
  const auto id = stable_region_point_id(provenance, parent_gid, point, side, 606u);
  const auto existing = std::find_if(topology.vertices.begin(),
                                     topology.vertices.end(),
                                     [&](const auto& v) {
                                       return v.stable_id == id;
                                     });
  if (existing == topology.vertices.end()) {
    CutIntegrationVertex vertex;
    vertex.stable_id = id;
    vertex.point = point;
    if (parent_parametric_coordinate != nullptr) {
      vertex.parent_parametric_coordinate = *parent_parametric_coordinate;
      vertex.has_parent_parametric_coordinate = true;
      vertex.parent_parametric_coordinate_valid = parent_parametric_coordinate_valid;
    }
    vertex.parent_geometry_dof = parent_dof;
    vertex.on_embedded_interface = on_interface;
    vertex.synthetic = synthetic;
    vertex.curved_isoparametric = curved_isoparametric;
    vertex.provenance = provenance;
    topology.vertices.push_back(std::move(vertex));
    topology.vertex_ids.push_back(id);
  } else {
    if (parent_parametric_coordinate != nullptr && !existing->has_parent_parametric_coordinate) {
      existing->parent_parametric_coordinate = *parent_parametric_coordinate;
      existing->has_parent_parametric_coordinate = true;
      existing->parent_parametric_coordinate_valid = parent_parametric_coordinate_valid;
    }
    existing->curved_isoparametric = existing->curved_isoparametric || curved_isoparametric;
  }
  return id;
}

void add_boundary_cycle_faces(
    SideClosedTopology& topology,
    const std::vector<std::uint64_t>& ids) {
  if (ids.size() < 2u) {
    return;
  }
  for (std::size_t i = 0; i < ids.size(); ++i) {
    topology.faces.push_back({ids[i], ids[(i + 1u) % ids.size()]});
  }
}

void add_integration_subcell(
    SideClosedTopology& topology,
    CellFamily family,
    const std::vector<std::array<real_t, 3>>& subcell_points,
    const std::vector<std::vector<int>>& local_faces,
    real_t measure,
    const std::array<real_t, 3>& centroid,
    const std::vector<index_t>& parent_dofs,
    const std::vector<std::array<real_t, 3>>& parent_points,
    const EmbeddedGeometryDescriptor& embedded,
    const EmbeddedRegionProvenance& provenance,
    gid_t parent_gid,
    CutTopologySide side,
    real_t tol,
    std::uint64_t salt,
    const std::vector<std::array<real_t, 3>>& parent_parametric_points = {},
    bool curved_isoparametric = false,
    bool measure_from_isoparametric_quadrature = false,
    std::string construction_policy = {}) {
  if (measure <= positive_tolerance(tol) || subcell_points.empty()) {
    return;
  }

  const bool have_parametric = parent_parametric_points.size() == subcell_points.size();
  std::vector<std::uint64_t> ids;
  ids.reserve(subcell_points.size());
  for (std::size_t i = 0; i < subcell_points.size(); ++i) {
    const auto& point = subcell_points[i];
    const auto parent_dof = matching_parent_dof(parent_dofs, parent_points, point, tol);
    const bool on_interface = std::abs(embedded.signed_distance(point)) <= positive_tolerance(tol);
    const bool synthetic = parent_dof == INVALID_INDEX && !on_interface;
    const auto* parent_xi = have_parametric ? &parent_parametric_points[i] : nullptr;
    ids.push_back(add_integration_vertex(topology,
                                         point,
                                         parent_dof,
                                         on_interface,
                                         synthetic,
                                         provenance,
                                         parent_gid,
                                         side,
                                         parent_xi,
                                         have_parametric,
                                         curved_isoparametric));
  }

  CutIntegrationSubcell subcell;
  subcell.family = family;
  subcell.vertices = ids;
  subcell.parent_parametric_vertices = parent_parametric_points;
  subcell.measure = measure;
  subcell.parent_parametric_measure =
      have_parametric ? parametric_measure(family, parent_parametric_points, tol) : real_t{0.0};
  subcell.centroid = centroid;
  subcell.provenance = provenance;
  subcell.curved_isoparametric = curved_isoparametric;
  subcell.measure_from_isoparametric_quadrature = measure_from_isoparametric_quadrature;
  subcell.construction_policy = !construction_policy.empty()
                                    ? std::move(construction_policy)
                                    : (curved_isoparametric
                                           ? "curved-isoparametric-topology-subdivision"
                                           : "linear-topology-subdivision");
  subcell.stable_id = stable_subcell_id(provenance, parent_gid, side, family, ids, salt);

  for (const auto& local_face : local_faces) {
    std::vector<std::uint64_t> face;
    face.reserve(local_face.size());
    for (const int idx : local_face) {
      if (idx >= 0 && static_cast<std::size_t>(idx) < ids.size()) {
        face.push_back(ids[static_cast<std::size_t>(idx)]);
      }
    }
    if (face.size() >= 2u) {
      subcell.faces.push_back(face);
      topology.faces.push_back(std::move(face));
    }
  }
  subcell.closed_topology =
      (family == CellFamily::Line && subcell.vertices.size() >= 2u && !subcell.faces.empty()) ||
      ((family == CellFamily::Triangle || family == CellFamily::Quad || family == CellFamily::Polygon) &&
       subcell.vertices.size() >= 3u && subcell.faces.size() >= 3u) ||
      ((family == CellFamily::Tetra || family == CellFamily::Hex || family == CellFamily::Wedge ||
        family == CellFamily::Pyramid || family == CellFamily::Polyhedron) &&
       subcell.vertices.size() >= 4u && subcell.faces.size() >= 4u);
  topology.closed = topology.closed || subcell.closed_topology;
  topology.subcells.push_back(std::move(subcell));
}

std::array<real_t, 3> line_parametric_point(real_t xi) noexcept {
  return {{xi, real_t{0.0}, real_t{0.0}}};
}

std::array<real_t, 3> evaluate_line_geometry_point(
    const MeshBase& mesh,
    index_t cell,
    real_t xi,
    Configuration cfg) {
  return CurvilinearEvaluator::evaluate_geometry(
      mesh, cell, line_parametric_point(xi), cfg).coordinates;
}

real_t signed_distance_on_line(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    real_t xi,
    Configuration cfg) {
  return embedded.signed_distance(evaluate_line_geometry_point(mesh, cell, xi, cfg));
}

std::vector<real_t> true_curved_line_roots(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    real_t tol) {
  const real_t eps = positive_tolerance(tol) * real_t{10.0};
  const int order = std::max(1, mesh.geometry_order(cell));
  const int sample_count = std::max(64, order * 32);
  std::vector<real_t> roots;
  roots.reserve(static_cast<std::size_t>(order + 2));

  const auto add_root = [&](real_t root) {
    root = std::max(real_t{-1.0}, std::min(real_t{1.0}, root));
    for (const auto existing : roots) {
      if (std::abs(existing - root) <= std::max(real_t{1.0e-10}, eps * real_t{100.0})) {
        return;
      }
    }
    roots.push_back(root);
  };

  real_t xi_prev = real_t{-1.0};
  real_t f_prev = signed_distance_on_line(mesh, cell, embedded, xi_prev, cfg);
  if (std::abs(f_prev) <= eps) {
    add_root(xi_prev);
  }

  for (int i = 1; i <= sample_count; ++i) {
    const real_t xi_curr =
        real_t{-1.0} + real_t{2.0} * static_cast<real_t>(i) / static_cast<real_t>(sample_count);
    const real_t f_curr = signed_distance_on_line(mesh, cell, embedded, xi_curr, cfg);
    if (std::abs(f_curr) <= eps) {
      add_root(xi_curr);
    }
    if ((f_prev < -eps && f_curr > eps) ||
        (f_prev > eps && f_curr < -eps)) {
      real_t a = xi_prev;
      real_t b = xi_curr;
      real_t fa = f_prev;
      for (int iter = 0; iter < 80; ++iter) {
        const real_t m = real_t{0.5} * (a + b);
        const real_t fm = signed_distance_on_line(mesh, cell, embedded, m, cfg);
        if (std::abs(fm) <= eps || std::abs(b - a) <= real_t{1.0e-14}) {
          a = m;
          b = m;
          break;
        }
        if ((fa < real_t{0.0} && fm > real_t{0.0}) ||
            (fa > real_t{0.0} && fm < real_t{0.0})) {
          b = m;
        } else {
          a = m;
          fa = fm;
        }
      }
      add_root(real_t{0.5} * (a + b));
    }
    xi_prev = xi_curr;
    f_prev = f_curr;
  }

  std::sort(roots.begin(), roots.end());
  roots.erase(std::unique(roots.begin(),
                          roots.end(),
                          [&](real_t a, real_t b) {
                            return std::abs(a - b) <=
                                   std::max(real_t{1.0e-10}, eps * real_t{100.0});
                          }),
              roots.end());
  return roots;
}

bool supports_true_curved_line_arrangement(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded) noexcept {
  return mesh.cell_shape(cell).family == CellFamily::Line &&
         mesh.geometry_order(cell) > 1 &&
         embedded.kind == EmbeddedGeometryKind::Plane;
}

CutSideRegion make_true_curved_line_side_region(
    const MeshBase& mesh,
    const CutEntityRecord& cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    CutTopologySide side,
    const std::vector<real_t>& breakpoints,
    const std::vector<std::uint64_t>& interface_vertex_ids,
    real_t tol) {
  CutSideRegion region;
  region.side = side;
  region.parent_cell = cell.entity;
  region.parent_cell_gid = cell.global_id;
  region.provenance = cell.provenance;
  region.integration_family = CellFamily::Line;
  region.curved_isoparametric_topology = true;
  region.measure_from_linear_topology = true;
  region.stable_id = stable_entity_id(cell.provenance,
                                      cell.global_id,
                                      0u,
                                      static_cast<index_t>(side),
                                      real_t{0.0},
                                      404u);
  region.cut_vertices = interface_vertex_ids;

  const auto dofs = mesh.cell_geometry_dofs(cell.entity);
  std::vector<std::array<real_t, 3>> parent_points;
  parent_points.reserve(dofs.size());
  std::vector<real_t> distances;
  distances.reserve(dofs.size());
  for (const auto dof : dofs) {
    const auto point = mesh.geometry_dof_coords(dof, cfg);
    parent_points.push_back(point);
    distances.push_back(embedded.signed_distance(point));
  }
  for (std::size_t i = 0; i < dofs.size(); ++i) {
    const bool on_side = side == CutTopologySide::Negative
                             ? distances[i] <= positive_tolerance(tol)
                             : distances[i] >= -positive_tolerance(tol);
    if (on_side) {
      region.parent_geometry_dofs.push_back(dofs[i]);
    }
  }

  for (std::size_t i = 0; i + 1 < breakpoints.size(); ++i) {
    const auto interval_measure =
        isoparametric_line_measure(mesh,
                                   cell.entity,
                                   line_parametric_point(breakpoints[i]),
                                   line_parametric_point(breakpoints[i + 1u]),
                                   cfg);
    if (interval_measure.available) {
      region.parent_measure += interval_measure.measure;
    }
  }

  SideClosedTopology closed;
  std::array<real_t, 3> weighted_centroid{{0.0, 0.0, 0.0}};
  for (std::size_t i = 0; i + 1 < breakpoints.size(); ++i) {
    const real_t a = breakpoints[i];
    const real_t b = breakpoints[i + 1u];
    if (std::abs(b - a) <= real_t{1.0e-14}) {
      continue;
    }
    const real_t mid = real_t{0.5} * (a + b);
    const real_t signed_mid = signed_distance_on_line(mesh, cell.entity, embedded, mid, cfg);
    const bool interval_on_side = side == CutTopologySide::Negative
                                      ? signed_mid <= positive_tolerance(tol)
                                      : signed_mid >= -positive_tolerance(tol);
    if (!interval_on_side) {
      continue;
    }

    const auto xi_a = line_parametric_point(a);
    const auto xi_b = line_parametric_point(b);
    const auto point_a = evaluate_line_geometry_point(mesh, cell.entity, a, cfg);
    const auto point_b = evaluate_line_geometry_point(mesh, cell.entity, b, cfg);
    const auto measure = isoparametric_line_measure(mesh, cell.entity, xi_a, xi_b, cfg);
    if (!measure.available || measure.measure <= positive_tolerance(tol)) {
      continue;
    }

    add_integration_subcell(closed,
                            CellFamily::Line,
                            {point_a, point_b},
                            {{0, 1}},
                            measure.measure,
                            measure.centroid,
                            dofs,
                            parent_points,
                            embedded,
                            cell.provenance,
                            cell.global_id,
                            side,
                            tol,
                            1700u + static_cast<std::uint64_t>(i),
                            {xi_a, xi_b},
                            true,
                            true,
                            kTrueCurvedArrangementPolicy);
    region.measure_estimate += measure.measure;
    weighted_centroid = add(weighted_centroid, scale(measure.centroid, measure.measure));
  }

  if (region.measure_estimate > real_t{0.0}) {
    region.centroid_estimate =
        scale(weighted_centroid, real_t{1.0} / region.measure_estimate);
    region.volume_fraction_estimate = region.parent_measure > real_t{0.0}
                                          ? region.measure_estimate / region.parent_measure
                                          : real_t{0.0};
  }
  region.integration_vertices = closed.vertices;
  region.integration_subcells = closed.subcells;
  region.integration_region_vertices = closed.vertex_ids;
  region.integration_region_faces = closed.faces;
  region.closed_integration_topology =
      !region.integration_subcells.empty() &&
      std::all_of(region.integration_subcells.begin(),
                  region.integration_subcells.end(),
                  [](const auto& subcell) {
                    return subcell.closed_topology;
                  });
  return region;
}

bool add_true_curved_line_arrangement(
    CutTopologyRecord& topology,
    const MeshBase& mesh,
    const CutEntityRecord& cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    real_t tol,
    std::uint64_t& h) {
  const auto roots = true_curved_line_roots(mesh, cell.entity, embedded, cfg, tol);
  if (roots.empty()) {
    topology.diagnostics.push_back(
        "true curved line arrangement found no plane roots in a cut high-order line cell");
    return false;
  }

  std::vector<real_t> breakpoints;
  breakpoints.reserve(roots.size() + 2u);
  breakpoints.push_back(real_t{-1.0});
  breakpoints.insert(breakpoints.end(), roots.begin(), roots.end());
  breakpoints.push_back(real_t{1.0});
  std::sort(breakpoints.begin(), breakpoints.end());
  breakpoints.erase(std::unique(breakpoints.begin(),
                                breakpoints.end(),
                                [](real_t a, real_t b) {
                                  return std::abs(a - b) <= real_t{1.0e-12};
                                }),
                    breakpoints.end());

  std::vector<std::uint64_t> interface_vertex_ids;
  interface_vertex_ids.reserve(roots.size());
  for (std::size_t i = 0; i < roots.size(); ++i) {
    const real_t root = roots[i];
    CurvedPatchPoint point;
    point.parent_parametric_coordinate = line_parametric_point(root);
    point.has_parent_parametric_coordinate = true;
    point.point = evaluate_line_geometry_point(mesh, cell.entity, root, cfg);
    const auto first_patch = topology.curved_patches.size();
    const bool recorded =
        record_curved_patch_from_candidates(topology,
                                            mesh,
                                            cell.entity,
                                            embedded,
                                            cfg,
                                            cell.provenance,
                                            cell.global_id,
                                            {point},
                                            tol,
                                            1900u + static_cast<std::uint64_t>(i),
                                            true,
                                            kTrueCurvedArrangementPolicy);
    if (!recorded || topology.curved_patches.size() <= first_patch) {
      continue;
    }
    const auto& patch = topology.curved_patches.back();
    if (!patch.ordered_vertices.empty()) {
      interface_vertex_ids.push_back(patch.ordered_vertices.front());
    }
    h = append_hash(h, patch.stable_id);
    h = append_hash(h, static_cast<std::uint64_t>(patch.parent_family));
    h = append_hash(h, static_cast<std::uint64_t>(patch.geometry_order));
    h = append_hash(h, patch.parametric_coordinates_valid ? 1u : 0u);
    h = append_hash(h, patch.exact_topology_available ? 1u : 0u);
    h = append_hash(h, patch.linearized_surrogate ? 1u : 0u);
    h = append_hash(h, patch.isoparametric_quadrature_available ? 1u : 0u);
    h = append_hash_string(h, patch.construction_policy);
    h = append_hash_real(h, patch.quadrature_measure);
    for (const auto& xi : patch.parent_parametric_coordinates) {
      h = append_hash_real(h, xi[0]);
      h = append_hash_real(h, xi[1]);
      h = append_hash_real(h, xi[2]);
    }
    for (const auto weight : patch.quadrature_weights) {
      h = append_hash_real(h, weight);
    }
    for (const auto id : patch.ordered_vertices) {
      h = append_hash(h, id);
    }
  }

  if (interface_vertex_ids.empty()) {
    topology.diagnostics.push_back(
        "true curved line arrangement failed to record interface point topology");
    return false;
  }

  for (const auto side : {CutTopologySide::Negative, CutTopologySide::Positive}) {
    auto region = make_true_curved_line_side_region(mesh,
                                                    cell,
                                                    embedded,
                                                    cfg,
                                                    side,
                                                    breakpoints,
                                                    interface_vertex_ids,
                                                    tol);
    if (!region.closed_integration_topology || region.integration_subcells.empty()) {
      std::ostringstream detail;
      detail << "true curved line arrangement failed to construct a closed side interval for side "
             << static_cast<int>(side) << " with " << region.integration_subcells.size()
             << " subcells and measure " << region.measure_estimate << "; breakpoints";
      for (const auto point : breakpoints) {
        detail << ' ' << point;
      }
      detail << "; midpoint signed distances";
      for (std::size_t i = 0; i + 1 < breakpoints.size(); ++i) {
        const real_t mid = real_t{0.5} * (breakpoints[i] + breakpoints[i + 1u]);
        detail << ' ' << signed_distance_on_line(mesh, cell.entity, embedded, mid, cfg);
      }
      detail << "; interval measures";
      for (std::size_t i = 0; i + 1 < breakpoints.size(); ++i) {
        const auto measure = isoparametric_line_measure(mesh,
                                                        cell.entity,
                                                        line_parametric_point(breakpoints[i]),
                                                        line_parametric_point(breakpoints[i + 1u]),
                                                        cfg);
        detail << ' ' << measure.measure << ':' << (measure.available ? 1 : 0);
      }
      topology.diagnostics.push_back(
          detail.str());
      return false;
    }
    topology.side_regions.push_back(std::move(region));
    const auto& stored = topology.side_regions.back();
    h = append_hash(h, stored.stable_id);
    h = append_hash_real(h, stored.parent_measure);
    h = append_hash_real(h, stored.measure_estimate);
    h = append_hash_real(h, stored.volume_fraction_estimate);
    for (const auto id : stored.integration_region_vertices) {
      h = append_hash(h, id);
    }
    for (const auto& face : stored.integration_region_faces) {
      h = append_hash(h, static_cast<std::uint64_t>(face.size()));
      for (const auto id : face) {
        h = append_hash(h, id);
      }
    }
    for (const auto& vertex : stored.integration_vertices) {
      h = append_hash(h, vertex.stable_id);
      h = append_hash_real(h, vertex.point[0]);
      h = append_hash_real(h, vertex.point[1]);
      h = append_hash_real(h, vertex.point[2]);
    }
    for (const auto& subcell : stored.integration_subcells) {
      h = append_hash(h, subcell.stable_id);
      h = append_hash(h, static_cast<std::uint64_t>(subcell.family));
      h = append_hash_real(h, subcell.measure);
      h = append_hash_real(h, subcell.parent_parametric_measure);
      h = append_hash(h, subcell.curved_isoparametric ? 1u : 0u);
      h = append_hash(h, subcell.measure_from_isoparametric_quadrature ? 1u : 0u);
      h = append_hash_string(h, subcell.construction_policy);
      for (const auto id : subcell.vertices) {
        h = append_hash(h, id);
      }
      for (const auto& xi : subcell.parent_parametric_vertices) {
        h = append_hash_real(h, xi[0]);
        h = append_hash_real(h, xi[1]);
        h = append_hash_real(h, xi[2]);
      }
      for (const auto& face : subcell.faces) {
        h = append_hash(h, static_cast<std::uint64_t>(face.size()));
        for (const auto id : face) {
          h = append_hash(h, id);
        }
      }
    }
  }

  return true;
}

struct GaussRule1D {
  std::vector<real_t> points{};
  std::vector<real_t> weights{};
};

GaussRule1D gauss_legendre_rule(int n) {
  n = std::max(2, n);
  GaussRule1D rule;
  rule.points.assign(static_cast<std::size_t>(n), real_t{0.0});
  rule.weights.assign(static_cast<std::size_t>(n), real_t{0.0});

  const int m = (n + 1) / 2;
  constexpr real_t pi = real_t{3.141592653589793238462643383279502884};
  for (int i = 0; i < m; ++i) {
    real_t z = std::cos(pi * (static_cast<real_t>(i) + real_t{0.75}) /
                        (static_cast<real_t>(n) + real_t{0.5}));
    real_t p1 = real_t{0.0};
    real_t p2 = real_t{0.0};
    real_t pp = real_t{0.0};
    for (int iter = 0; iter < 64; ++iter) {
      p1 = real_t{1.0};
      p2 = real_t{0.0};
      for (int j = 1; j <= n; ++j) {
        const real_t p3 = p2;
        p2 = p1;
        p1 = ((real_t{2.0} * static_cast<real_t>(j) - real_t{1.0}) * z * p2 -
              (static_cast<real_t>(j) - real_t{1.0}) * p3) /
             static_cast<real_t>(j);
      }
      pp = static_cast<real_t>(n) * (z * p1 - p2) / (z * z - real_t{1.0});
      const real_t z_next = z - p1 / pp;
      if (std::abs(z_next - z) <= real_t{1.0e-15}) {
        z = z_next;
        break;
      }
      z = z_next;
    }
    const real_t weight =
        real_t{2.0} / ((real_t{1.0} - z * z) * pp * pp);
    rule.points[static_cast<std::size_t>(i)] = -z;
    rule.points[static_cast<std::size_t>(n - 1 - i)] = z;
    rule.weights[static_cast<std::size_t>(i)] = weight;
    rule.weights[static_cast<std::size_t>(n - 1 - i)] = weight;
  }
  return rule;
}

struct FaceReferenceSpan {
  real_t r_min{0.0};
  real_t r_max{0.0};
  bool valid{false};
};

struct FaceGraphRoot {
  bool has_root{false};
  real_t r{0.0};
  real_t f_min{0.0};
  real_t f_max{0.0};
};

struct FaceGraphIntegral {
  real_t measure{0.0};
  std::array<real_t, 3> centroid{{0.0, 0.0, 0.0}};
  bool available{false};
};

std::pair<real_t, real_t> face_graph_s_bounds(CellFamily family) noexcept {
  if (family == CellFamily::Quad) {
    return {real_t{-1.0}, real_t{1.0}};
  }
  return {real_t{0.0}, real_t{1.0}};
}

FaceReferenceSpan face_reference_span(CellFamily family, real_t s, real_t tol) noexcept {
  if (family == CellFamily::Quad) {
    return {real_t{-1.0}, real_t{1.0}, true};
  }
  if (family == CellFamily::Triangle) {
    const real_t rmax = real_t{1.0} - s;
    return {real_t{0.0}, rmax, rmax >= -positive_tolerance(tol)};
  }
  return {};
}

std::array<real_t, 3> face_parametric_point(real_t r, real_t s) noexcept {
  return {{r, s, real_t{0.0}}};
}

std::array<real_t, 3> evaluate_face_geometry_point(
    const MeshBase& mesh,
    index_t cell,
    real_t r,
    real_t s,
    Configuration cfg) {
  return CurvilinearEvaluator::evaluate_geometry(
      mesh, cell, face_parametric_point(r, s), cfg).coordinates;
}

real_t signed_distance_on_face(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    real_t r,
    real_t s,
    Configuration cfg) {
  return embedded.signed_distance(evaluate_face_geometry_point(mesh, cell, r, s, cfg));
}

FaceGraphRoot true_curved_face_root_at_s(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    CellFamily family,
    real_t s,
    Configuration cfg,
    real_t tol) {
  FaceGraphRoot out;
  const auto span = face_reference_span(family, s, tol);
  if (!span.valid || span.r_max - span.r_min <= positive_tolerance(tol)) {
    return out;
  }
  const real_t eps = positive_tolerance(tol) * real_t{10.0};
  out.f_min = signed_distance_on_face(mesh, cell, embedded, span.r_min, s, cfg);
  out.f_max = signed_distance_on_face(mesh, cell, embedded, span.r_max, s, cfg);
  if (std::abs(out.f_min) <= eps) {
    out.has_root = true;
    out.r = span.r_min;
    return out;
  }
  if (std::abs(out.f_max) <= eps) {
    out.has_root = true;
    out.r = span.r_max;
    return out;
  }
  if (!((out.f_min < -eps && out.f_max > eps) ||
        (out.f_min > eps && out.f_max < -eps))) {
    return out;
  }

  real_t a = span.r_min;
  real_t b = span.r_max;
  real_t fa = out.f_min;
  for (int iter = 0; iter < 96; ++iter) {
    const real_t m = real_t{0.5} * (a + b);
    const real_t fm = signed_distance_on_face(mesh, cell, embedded, m, s, cfg);
    if (std::abs(fm) <= eps || std::abs(b - a) <= real_t{1.0e-14}) {
      a = m;
      b = m;
      break;
    }
    if ((fa < real_t{0.0} && fm > real_t{0.0}) ||
        (fa > real_t{0.0} && fm < real_t{0.0})) {
      b = m;
    } else {
      a = m;
      fa = fm;
    }
  }
  out.has_root = true;
  out.r = real_t{0.5} * (a + b);
  return out;
}

std::vector<real_t> true_curved_face_roots_at_s(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    CellFamily family,
    real_t s,
    Configuration cfg,
    real_t tol) {
  std::vector<real_t> roots;
  const auto span = face_reference_span(family, s, tol);
  if (!span.valid || span.r_max - span.r_min <= positive_tolerance(tol)) {
    return roots;
  }
  const int order = std::max(1, mesh.geometry_order(cell));
  const int samples = std::max(48, order * 32);
  const real_t eps = positive_tolerance(tol) * real_t{10.0};
  const auto add_root = [&](real_t r) {
    r = std::max(span.r_min, std::min(span.r_max, r));
    for (const auto existing : roots) {
      if (std::abs(existing - r) <= std::max(real_t{1.0e-10}, eps * real_t{100.0})) {
        return;
      }
    }
    roots.push_back(r);
  };

  real_t r_prev = span.r_min;
  real_t f_prev = signed_distance_on_face(mesh, cell, embedded, r_prev, s, cfg);
  if (std::abs(f_prev) <= eps) {
    add_root(r_prev);
  }
  for (int i = 1; i <= samples; ++i) {
    const real_t r_curr =
        span.r_min + (span.r_max - span.r_min) *
                         static_cast<real_t>(i) / static_cast<real_t>(samples);
    const real_t f_curr = signed_distance_on_face(mesh, cell, embedded, r_curr, s, cfg);
    if (std::abs(f_curr) <= eps) {
      add_root(r_curr);
    }
    if ((f_prev < -eps && f_curr > eps) ||
        (f_prev > eps && f_curr < -eps)) {
      real_t a = r_prev;
      real_t b = r_curr;
      real_t fa = f_prev;
      for (int iter = 0; iter < 80; ++iter) {
        const real_t m = real_t{0.5} * (a + b);
        const real_t fm = signed_distance_on_face(mesh, cell, embedded, m, s, cfg);
        if (std::abs(fm) <= eps || std::abs(b - a) <= real_t{1.0e-14}) {
          a = m;
          b = m;
          break;
        }
        if ((fa < real_t{0.0} && fm > real_t{0.0}) ||
            (fa > real_t{0.0} && fm < real_t{0.0})) {
          b = m;
        } else {
          a = m;
          fa = fm;
        }
      }
      add_root(real_t{0.5} * (a + b));
    }
    r_prev = r_curr;
    f_prev = f_curr;
  }
  std::sort(roots.begin(), roots.end());
  return roots;
}

bool true_curved_face_is_graph_compatible(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    CellFamily family,
    Configuration cfg,
    real_t tol) {
  const auto bounds = face_graph_s_bounds(family);
  const int order = std::max(1, mesh.geometry_order(cell));
  const int samples = std::max(16, order * 8);
  for (int i = 0; i <= samples; ++i) {
    const real_t s =
        bounds.first + (bounds.second - bounds.first) *
                           static_cast<real_t>(i) / static_cast<real_t>(samples);
    if (true_curved_face_roots_at_s(mesh, cell, embedded, family, s, cfg, tol).size() > 1u) {
      return false;
    }
  }
  return true;
}

std::vector<real_t> face_boundary_roots_in_s(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    CellFamily family,
    Configuration cfg,
    real_t tol,
    bool upper_boundary) {
  const auto bounds = face_graph_s_bounds(family);
  const int order = std::max(1, mesh.geometry_order(cell));
  const int samples = std::max(64, order * 32);
  const real_t eps = positive_tolerance(tol) * real_t{10.0};
  std::vector<real_t> roots;
  const auto boundary_value = [&](real_t s) {
    const auto span = face_reference_span(family, s, tol);
    const real_t r = upper_boundary ? span.r_max : span.r_min;
    return signed_distance_on_face(mesh, cell, embedded, r, s, cfg);
  };
  const auto add_root = [&](real_t s) {
    s = std::max(bounds.first, std::min(bounds.second, s));
    for (const auto existing : roots) {
      if (std::abs(existing - s) <= std::max(real_t{1.0e-10}, eps * real_t{100.0})) {
        return;
      }
    }
    roots.push_back(s);
  };

  real_t s_prev = bounds.first;
  real_t f_prev = boundary_value(s_prev);
  if (std::abs(f_prev) <= eps) {
    add_root(s_prev);
  }
  for (int i = 1; i <= samples; ++i) {
    const real_t s_curr =
        bounds.first + (bounds.second - bounds.first) *
                           static_cast<real_t>(i) / static_cast<real_t>(samples);
    const real_t f_curr = boundary_value(s_curr);
    if (std::abs(f_curr) <= eps) {
      add_root(s_curr);
    }
    if ((f_prev < -eps && f_curr > eps) ||
        (f_prev > eps && f_curr < -eps)) {
      real_t a = s_prev;
      real_t b = s_curr;
      real_t fa = f_prev;
      for (int iter = 0; iter < 80; ++iter) {
        const real_t m = real_t{0.5} * (a + b);
        const real_t fm = boundary_value(m);
        if (std::abs(fm) <= eps || std::abs(b - a) <= real_t{1.0e-14}) {
          a = m;
          b = m;
          break;
        }
        if ((fa < real_t{0.0} && fm > real_t{0.0}) ||
            (fa > real_t{0.0} && fm < real_t{0.0})) {
          b = m;
        } else {
          a = m;
          fa = fm;
        }
      }
      add_root(real_t{0.5} * (a + b));
    }
    s_prev = s_curr;
    f_prev = f_curr;
  }
  return roots;
}

std::vector<real_t> true_curved_face_breakpoints(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    CellFamily family,
    Configuration cfg,
    real_t tol) {
  const auto bounds = face_graph_s_bounds(family);
  const int order = std::max(1, mesh.geometry_order(cell));
  const int segments = std::max(16, order * 8);
  std::vector<real_t> out;
  out.reserve(static_cast<std::size_t>(segments + 8));
  for (int i = 0; i <= segments; ++i) {
    out.push_back(bounds.first + (bounds.second - bounds.first) *
                                     static_cast<real_t>(i) / static_cast<real_t>(segments));
  }
  auto lower = face_boundary_roots_in_s(mesh, cell, embedded, family, cfg, tol, false);
  auto upper = face_boundary_roots_in_s(mesh, cell, embedded, family, cfg, tol, true);
  out.insert(out.end(), lower.begin(), lower.end());
  out.insert(out.end(), upper.begin(), upper.end());
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(),
                        out.end(),
                        [](real_t a, real_t b) {
                          return std::abs(a - b) <= real_t{1.0e-12};
                        }),
            out.end());
  return out;
}

FaceGraphIntegral integrate_face_graph_strip(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    CellFamily family,
    Configuration cfg,
    real_t tol,
    real_t s0,
    real_t s1,
    bool use_root,
    bool lower_side_interval) {
  FaceGraphIntegral out;
  if (s1 - s0 <= positive_tolerance(tol)) {
    return out;
  }
  const int order = std::max(1, mesh.geometry_order(cell));
  const auto s_rule = gauss_legendre_rule(std::min(12, std::max(4, order + 3)));
  const auto r_rule = gauss_legendre_rule(std::min(12, std::max(4, order + 3)));
  const real_t half_s = real_t{0.5} * (s1 - s0);
  const real_t center_s = real_t{0.5} * (s0 + s1);
  out.available = true;
  for (std::size_t is = 0; is < s_rule.points.size(); ++is) {
    const real_t s = center_s + half_s * s_rule.points[is];
    const auto span = face_reference_span(family, s, tol);
    if (!span.valid || span.r_max - span.r_min <= positive_tolerance(tol)) {
      continue;
    }
    real_t r_lo = span.r_min;
    real_t r_hi = span.r_max;
    if (use_root) {
      const auto root = true_curved_face_root_at_s(mesh, cell, embedded, family, s, cfg, tol);
      if (!root.has_root) {
        return {};
      }
      if (lower_side_interval) {
        r_hi = root.r;
      } else {
        r_lo = root.r;
      }
    }
    if (r_hi - r_lo <= positive_tolerance(tol)) {
      continue;
    }
    const real_t half_r = real_t{0.5} * (r_hi - r_lo);
    const real_t center_r = real_t{0.5} * (r_lo + r_hi);
    for (std::size_t ir = 0; ir < r_rule.points.size(); ++ir) {
      const real_t r = center_r + half_r * r_rule.points[ir];
      const auto eval = CurvilinearEvaluator::evaluate_geometry(mesh, cell, face_parametric_point(r, s), cfg);
      const auto dx_dr = jacobian_action(eval.jacobian, {{real_t{1.0}, real_t{0.0}, real_t{0.0}}});
      const auto dx_ds = jacobian_action(eval.jacobian, {{real_t{0.0}, real_t{1.0}, real_t{0.0}}});
      const real_t jac = norm(cross(dx_dr, dx_ds));
      const real_t weight = s_rule.weights[is] * r_rule.weights[ir] * half_s * half_r * jac;
      if (!std::isfinite(weight)) {
        return {};
      }
      out.measure += weight;
      out.centroid = add(out.centroid, scale(eval.coordinates, weight));
    }
  }
  if (out.measure > real_t{0.0}) {
    out.centroid = scale(out.centroid, real_t{1.0} / out.measure);
  }
  return out;
}

std::vector<std::array<real_t, 3>> unique_xi_vertices(
    const std::vector<std::array<real_t, 3>>& vertices,
    real_t tol) {
  std::vector<std::array<real_t, 3>> out;
  for (const auto& vertex : vertices) {
    if (std::find_if(out.begin(), out.end(), [&](const auto& existing) {
          return norm(sub(vertex, existing)) <= positive_tolerance(tol) * real_t{10.0};
        }) == out.end()) {
      out.push_back(vertex);
    }
  }
  return out;
}

void add_true_curved_face_graph_subcell(
    SideClosedTopology& closed,
    const MeshBase& mesh,
    const CutEntityRecord& cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    CutTopologySide side,
    const std::vector<index_t>& dofs,
    const std::vector<std::array<real_t, 3>>& parent_points,
    std::vector<std::array<real_t, 3>> xi_vertices,
    const FaceGraphIntegral& measure,
    real_t tol,
    std::uint64_t salt) {
  xi_vertices = unique_xi_vertices(xi_vertices, tol);
  if (!measure.available ||
      measure.measure <= positive_tolerance(tol) ||
      xi_vertices.size() < 3u) {
    return;
  }

  std::vector<std::array<real_t, 3>> physical_points;
  physical_points.reserve(xi_vertices.size());
  for (const auto& xi : xi_vertices) {
    physical_points.push_back(CurvilinearEvaluator::evaluate_geometry(mesh, cell.entity, xi, cfg).coordinates);
  }

  std::vector<std::vector<int>> faces;
  faces.reserve(xi_vertices.size());
  for (std::size_t i = 0; i < xi_vertices.size(); ++i) {
    faces.push_back({static_cast<int>(i), static_cast<int>((i + 1u) % xi_vertices.size())});
  }
  add_integration_subcell(closed,
                          xi_vertices.size() == 3u ? CellFamily::Triangle : CellFamily::Quad,
                          physical_points,
                          faces,
                          measure.measure,
                          measure.centroid,
                          dofs,
                          parent_points,
                          embedded,
                          cell.provenance,
                          cell.global_id,
                          side,
                          tol,
                          salt,
                          xi_vertices,
                          true,
                          true,
                          kTrueCurvedArrangementPolicy);
}

bool supports_true_curved_face_arrangement(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded) noexcept {
  const auto family = mesh.cell_shape(cell).family;
  return (family == CellFamily::Triangle || family == CellFamily::Quad) &&
         mesh.geometry_order(cell) > 1 &&
         embedded.kind == EmbeddedGeometryKind::Plane;
}

bool populate_true_curved_face_patch_quadrature(
    CutCurvedPatchRecord& patch,
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    CellFamily family,
    Configuration cfg,
    const std::vector<real_t>& breakpoints,
    real_t tol) {
  patch.quadrature_points.clear();
  patch.quadrature_normals.clear();
  patch.quadrature_weights.clear();
  patch.quadrature_measure = real_t{0.0};
  patch.isoparametric_quadrature_available = false;
  const int order = std::max(1, mesh.geometry_order(cell));
  const auto s_rule = gauss_legendre_rule(std::min(12, std::max(4, order + 3)));
  for (std::size_t i = 0; i + 1 < breakpoints.size(); ++i) {
    const real_t s0 = breakpoints[i];
    const real_t s1 = breakpoints[i + 1u];
    const real_t smid = real_t{0.5} * (s0 + s1);
    if (!true_curved_face_root_at_s(mesh, cell, embedded, family, smid, cfg, tol).has_root) {
      continue;
    }
    const real_t half_s = real_t{0.5} * (s1 - s0);
    const real_t center_s = real_t{0.5} * (s0 + s1);
    for (std::size_t iq = 0; iq < s_rule.points.size(); ++iq) {
      const real_t s = center_s + half_s * s_rule.points[iq];
      const auto root = true_curved_face_root_at_s(mesh, cell, embedded, family, s, cfg, tol);
      if (!root.has_root) {
        return false;
      }
      const auto xi = face_parametric_point(root.r, s);
      const auto eval = CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, cfg);
      const auto dx_dr = jacobian_action(eval.jacobian, {{real_t{1.0}, real_t{0.0}, real_t{0.0}}});
      const auto dx_ds = jacobian_action(eval.jacobian, {{real_t{0.0}, real_t{1.0}, real_t{0.0}}});
      const real_t phi_r = dot(embedded.normal, dx_dr);
      const real_t phi_s = dot(embedded.normal, dx_ds);
      if (std::abs(phi_r) <= positive_tolerance(tol)) {
        return false;
      }
      const real_t drds = -phi_s / phi_r;
      const auto tangent = add(scale(dx_dr, drds), dx_ds);
      const real_t weight = s_rule.weights[iq] * half_s * norm(tangent);
      if (!std::isfinite(weight) || weight <= real_t{0.0}) {
        return false;
      }
      patch.quadrature_points.push_back(eval.coordinates);
      patch.quadrature_normals.push_back(unit_or_default(embedded.outward_normal(eval.coordinates)));
      patch.quadrature_weights.push_back(weight);
      patch.quadrature_measure += weight;
    }
  }
  patch.isoparametric_quadrature_available =
      patch.quadrature_measure > positive_tolerance(tol);
  patch.construction_policy = kTrueCurvedArrangementPolicy;
  return patch.isoparametric_quadrature_available;
}

CutSideRegion make_true_curved_face_side_region(
    const MeshBase& mesh,
    const CutEntityRecord& cell,
    const EmbeddedGeometryDescriptor& embedded,
    CellFamily family,
    Configuration cfg,
    CutTopologySide side,
    const std::vector<real_t>& breakpoints,
    const std::vector<std::uint64_t>& interface_vertex_ids,
    real_t tol) {
  CutSideRegion region;
  region.side = side;
  region.parent_cell = cell.entity;
  region.parent_cell_gid = cell.global_id;
  region.provenance = cell.provenance;
  region.integration_family = family;
  region.curved_isoparametric_topology = true;
  region.measure_from_linear_topology = true;
  region.stable_id = stable_entity_id(cell.provenance,
                                      cell.global_id,
                                      0u,
                                      static_cast<index_t>(side),
                                      real_t{0.0},
                                      404u);
  region.cut_vertices = interface_vertex_ids;

  const auto dofs = mesh.cell_geometry_dofs(cell.entity);
  std::vector<std::array<real_t, 3>> parent_points;
  parent_points.reserve(dofs.size());
  std::vector<real_t> distances;
  distances.reserve(dofs.size());
  for (const auto dof : dofs) {
    const auto point = mesh.geometry_dof_coords(dof, cfg);
    parent_points.push_back(point);
    distances.push_back(embedded.signed_distance(point));
  }
  for (std::size_t i = 0; i < dofs.size(); ++i) {
    const bool on_side = side == CutTopologySide::Negative
                             ? distances[i] <= positive_tolerance(tol)
                             : distances[i] >= -positive_tolerance(tol);
    if (on_side) {
      region.parent_geometry_dofs.push_back(dofs[i]);
    }
  }

  SideClosedTopology closed;
  std::array<real_t, 3> weighted_centroid{{0.0, 0.0, 0.0}};
  for (std::size_t i = 0; i + 1 < breakpoints.size(); ++i) {
    const real_t s0 = breakpoints[i];
    const real_t s1 = breakpoints[i + 1u];
    if (s1 - s0 <= positive_tolerance(tol)) {
      continue;
    }
    const real_t smid = real_t{0.5} * (s0 + s1);
    const auto span_mid = face_reference_span(family, smid, tol);
    if (!span_mid.valid || span_mid.r_max - span_mid.r_min <= positive_tolerance(tol)) {
      continue;
    }

    const auto full_measure = integrate_face_graph_strip(
        mesh, cell.entity, embedded, family, cfg, tol, s0, s1, false, true);
    if (full_measure.available) {
      region.parent_measure += full_measure.measure;
    }

    const auto root_mid = true_curved_face_root_at_s(mesh, cell.entity, embedded, family, smid, cfg, tol);
    const auto span0 = face_reference_span(family, s0, tol);
    const auto span1 = face_reference_span(family, s1, tol);
    if (!root_mid.has_root) {
      const real_t rmid = real_t{0.5} * (span_mid.r_min + span_mid.r_max);
      const real_t signed_mid = signed_distance_on_face(mesh, cell.entity, embedded, rmid, smid, cfg);
      const bool interval_on_side = side == CutTopologySide::Negative
                                        ? signed_mid <= positive_tolerance(tol)
                                        : signed_mid >= -positive_tolerance(tol);
      if (!interval_on_side) {
        continue;
      }
      const std::vector<std::array<real_t, 3>> xi_vertices{
          face_parametric_point(span0.r_min, s0),
          face_parametric_point(span0.r_max, s0),
          face_parametric_point(span1.r_max, s1),
          face_parametric_point(span1.r_min, s1)};
      add_true_curved_face_graph_subcell(closed,
                                         mesh,
                                         cell,
                                         embedded,
                                         cfg,
                                         side,
                                         dofs,
                                         parent_points,
                                         xi_vertices,
                                         full_measure,
                                         tol,
                                         2100u + static_cast<std::uint64_t>(i) * 3u);
      region.measure_estimate += full_measure.measure;
      weighted_centroid = add(weighted_centroid, scale(full_measure.centroid, full_measure.measure));
      continue;
    }

    const auto root0 = true_curved_face_root_at_s(mesh, cell.entity, embedded, family, s0, cfg, tol);
    const auto root1 = true_curved_face_root_at_s(mesh, cell.entity, embedded, family, s1, cfg, tol);
    if (!root0.has_root || !root1.has_root) {
      continue;
    }
    const real_t left_probe = real_t{0.5} * (span_mid.r_min + root_mid.r);
    const real_t right_probe = real_t{0.5} * (root_mid.r + span_mid.r_max);
    const bool left_negative =
        signed_distance_on_face(mesh, cell.entity, embedded, left_probe, smid, cfg) <= positive_tolerance(tol);
    const bool right_negative =
        signed_distance_on_face(mesh, cell.entity, embedded, right_probe, smid, cfg) <= positive_tolerance(tol);
    const bool want_negative = side == CutTopologySide::Negative;

    if (left_negative == want_negative) {
      const auto measure = integrate_face_graph_strip(
          mesh, cell.entity, embedded, family, cfg, tol, s0, s1, true, true);
      const std::vector<std::array<real_t, 3>> xi_vertices{
          face_parametric_point(span0.r_min, s0),
          face_parametric_point(root0.r, s0),
          face_parametric_point(root1.r, s1),
          face_parametric_point(span1.r_min, s1)};
      add_true_curved_face_graph_subcell(closed,
                                         mesh,
                                         cell,
                                         embedded,
                                         cfg,
                                         side,
                                         dofs,
                                         parent_points,
                                         xi_vertices,
                                         measure,
                                         tol,
                                         2200u + static_cast<std::uint64_t>(i) * 5u);
      region.measure_estimate += measure.measure;
      weighted_centroid = add(weighted_centroid, scale(measure.centroid, measure.measure));
    }
    if (right_negative == want_negative) {
      const auto measure = integrate_face_graph_strip(
          mesh, cell.entity, embedded, family, cfg, tol, s0, s1, true, false);
      const std::vector<std::array<real_t, 3>> xi_vertices{
          face_parametric_point(root0.r, s0),
          face_parametric_point(span0.r_max, s0),
          face_parametric_point(span1.r_max, s1),
          face_parametric_point(root1.r, s1)};
      add_true_curved_face_graph_subcell(closed,
                                         mesh,
                                         cell,
                                         embedded,
                                         cfg,
                                         side,
                                         dofs,
                                         parent_points,
                                         xi_vertices,
                                         measure,
                                         tol,
                                         2300u + static_cast<std::uint64_t>(i) * 5u);
      region.measure_estimate += measure.measure;
      weighted_centroid = add(weighted_centroid, scale(measure.centroid, measure.measure));
    }
  }

  if (region.measure_estimate > real_t{0.0}) {
    region.centroid_estimate =
        scale(weighted_centroid, real_t{1.0} / region.measure_estimate);
    region.volume_fraction_estimate = region.parent_measure > real_t{0.0}
                                          ? region.measure_estimate / region.parent_measure
                                          : real_t{0.0};
  }
  region.integration_vertices = closed.vertices;
  region.integration_subcells = closed.subcells;
  region.integration_region_vertices = closed.vertex_ids;
  region.integration_region_faces = closed.faces;
  region.closed_integration_topology =
      !region.integration_subcells.empty() &&
      std::all_of(region.integration_subcells.begin(),
                  region.integration_subcells.end(),
                  [](const auto& subcell) {
                    return subcell.closed_topology;
                  });
  return region;
}

void hash_curved_patch_record(std::uint64_t& h, const CutCurvedPatchRecord& patch) {
  h = append_hash(h, patch.stable_id);
  h = append_hash(h, static_cast<std::uint64_t>(patch.parent_family));
  h = append_hash(h, static_cast<std::uint64_t>(patch.geometry_order));
  h = append_hash(h, patch.parametric_coordinates_valid ? 1u : 0u);
  h = append_hash(h, patch.exact_topology_available ? 1u : 0u);
  h = append_hash(h, patch.linearized_surrogate ? 1u : 0u);
  h = append_hash(h, patch.isoparametric_quadrature_available ? 1u : 0u);
  h = append_hash_string(h, patch.construction_policy);
  h = append_hash_real(h, patch.quadrature_measure);
  for (const auto& xi : patch.parent_parametric_coordinates) {
    h = append_hash_real(h, xi[0]);
    h = append_hash_real(h, xi[1]);
    h = append_hash_real(h, xi[2]);
  }
  for (const auto& point : patch.quadrature_points) {
    h = append_hash_real(h, point[0]);
    h = append_hash_real(h, point[1]);
    h = append_hash_real(h, point[2]);
  }
  for (const auto weight : patch.quadrature_weights) {
    h = append_hash_real(h, weight);
  }
  for (const auto id : patch.ordered_vertices) {
    h = append_hash(h, id);
  }
}

void hash_side_region_record(std::uint64_t& h, const CutSideRegion& region) {
  h = append_hash(h, region.stable_id);
  h = append_hash_real(h, region.parent_measure);
  h = append_hash_real(h, region.measure_estimate);
  h = append_hash_real(h, region.volume_fraction_estimate);
  for (const auto id : region.integration_region_vertices) {
    h = append_hash(h, id);
  }
  for (const auto& face : region.integration_region_faces) {
    h = append_hash(h, static_cast<std::uint64_t>(face.size()));
    for (const auto id : face) {
      h = append_hash(h, id);
    }
  }
  for (const auto& vertex : region.integration_vertices) {
    h = append_hash(h, vertex.stable_id);
    h = append_hash_real(h, vertex.point[0]);
    h = append_hash_real(h, vertex.point[1]);
    h = append_hash_real(h, vertex.point[2]);
  }
  for (const auto& subcell : region.integration_subcells) {
    h = append_hash(h, subcell.stable_id);
    h = append_hash(h, static_cast<std::uint64_t>(subcell.family));
    h = append_hash_real(h, subcell.measure);
    h = append_hash_real(h, subcell.parent_parametric_measure);
    h = append_hash(h, subcell.curved_isoparametric ? 1u : 0u);
    h = append_hash(h, subcell.measure_from_isoparametric_quadrature ? 1u : 0u);
    h = append_hash_string(h, subcell.construction_policy);
    for (const auto id : subcell.vertices) {
      h = append_hash(h, id);
    }
    for (const auto& xi : subcell.parent_parametric_vertices) {
      h = append_hash_real(h, xi[0]);
      h = append_hash_real(h, xi[1]);
      h = append_hash_real(h, xi[2]);
    }
    for (const auto& face : subcell.faces) {
      h = append_hash(h, static_cast<std::uint64_t>(face.size()));
      for (const auto id : face) {
        h = append_hash(h, id);
      }
    }
  }
}

bool add_true_curved_face_arrangement(
    CutTopologyRecord& topology,
    const MeshBase& mesh,
    const CutEntityRecord& cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    real_t tol,
    std::uint64_t& h) {
  const auto family = mesh.cell_shape(cell.entity).family;
  if (!true_curved_face_is_graph_compatible(
          mesh, cell.entity, embedded, family, cfg, tol)) {
    topology.diagnostics.push_back(
        "true curved face arrangement requires a graph-compatible plane cut with at most one root per reference-column sample");
    return false;
  }
  const auto breakpoints =
      true_curved_face_breakpoints(mesh, cell.entity, embedded, family, cfg, tol);
  CurvedPatchPoint first_endpoint;
  CurvedPatchPoint last_endpoint;
  bool saw_endpoint = false;
  for (const auto s : breakpoints) {
    const auto root = true_curved_face_root_at_s(mesh, cell.entity, embedded, family, s, cfg, tol);
    if (!root.has_root) {
      continue;
    }
    CurvedPatchPoint point;
    point.parent_parametric_coordinate = face_parametric_point(root.r, s);
    point.has_parent_parametric_coordinate = true;
    point.point = evaluate_face_geometry_point(mesh, cell.entity, root.r, s, cfg);
    if (!saw_endpoint) {
      first_endpoint = point;
      saw_endpoint = true;
    }
    last_endpoint = point;
  }
  if (!saw_endpoint) {
    topology.diagnostics.push_back(
        "true curved face arrangement found no graph-compatible plane roots in a cut high-order face cell");
    return false;
  }
  std::vector<CurvedPatchPoint> endpoints{first_endpoint, last_endpoint};
  if (norm(sub(endpoints.front().point, endpoints.back().point)) <= positive_tolerance(tol)) {
    topology.diagnostics.push_back(
        "true curved face arrangement produced a degenerate interface curve");
    return false;
  }

  const auto first_patch = topology.curved_patches.size();
  const bool recorded =
      record_curved_patch_from_candidates(topology,
                                          mesh,
                                          cell.entity,
                                          embedded,
                                          cfg,
                                          cell.provenance,
                                          cell.global_id,
                                          endpoints,
                                          tol,
                                          2400u,
                                          true,
                                          kTrueCurvedArrangementPolicy);
  if (!recorded || topology.curved_patches.size() <= first_patch) {
    topology.diagnostics.push_back(
        "true curved face arrangement failed to record interface curve topology");
    return false;
  }
  auto& patch = topology.curved_patches.back();
  if (!populate_true_curved_face_patch_quadrature(
          patch, mesh, cell.entity, embedded, family, cfg, breakpoints, tol)) {
    topology.diagnostics.push_back(
        "true curved face arrangement failed to build curve quadrature");
    return false;
  }
  const auto interface_vertex_ids = patch.ordered_vertices;
  hash_curved_patch_record(h, patch);

  for (const auto side : {CutTopologySide::Negative, CutTopologySide::Positive}) {
    auto region = make_true_curved_face_side_region(mesh,
                                                    cell,
                                                    embedded,
                                                    family,
                                                    cfg,
                                                    side,
                                                    breakpoints,
                                                    interface_vertex_ids,
                                                    tol);
    if (!region.closed_integration_topology || region.integration_subcells.empty()) {
      topology.diagnostics.push_back(
          "true curved face arrangement failed to construct closed side-region topology");
      return false;
    }
    topology.side_regions.push_back(std::move(region));
    hash_side_region_record(h, topology.side_regions.back());
  }

  return true;
}

struct TetReferenceSpan {
  real_t r_min{0.0};
  real_t r_max{0.0};
  bool valid{false};
};

struct TetGraphRoot {
  bool has_root{false};
  real_t r{0.0};
  real_t f_min{0.0};
  real_t f_max{0.0};
};

struct TetBasePoint {
  real_t s{0.0};
  real_t t{0.0};
  real_t value{0.0};
};

struct TetBaseTriangle {
  std::array<TetBasePoint, 3> vertices{};
};

struct TetColumnInterval {
  bool available{false};
  real_t r_lo{0.0};
  real_t r_hi{0.0};
};

TetReferenceSpan tet_reference_span(real_t s, real_t t, real_t tol) noexcept {
  const real_t eps = positive_tolerance(tol);
  const real_t rmax = real_t{1.0} - s - t;
  const bool valid = s >= -eps && t >= -eps && rmax >= -eps;
  return {real_t{0.0}, std::max(real_t{0.0}, rmax), valid};
}

std::array<real_t, 3> tet_parametric_point(real_t r, real_t s, real_t t) noexcept {
  return {{r, s, t}};
}

std::array<real_t, 3> evaluate_tet_geometry_point(
    const MeshBase& mesh,
    index_t cell,
    real_t r,
    real_t s,
    real_t t,
    Configuration cfg) {
  return CurvilinearEvaluator::evaluate_geometry(
      mesh, cell, tet_parametric_point(r, s, t), cfg).coordinates;
}

real_t signed_distance_on_tet(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    real_t r,
    real_t s,
    real_t t,
    Configuration cfg) {
  return embedded.signed_distance(evaluate_tet_geometry_point(mesh, cell, r, s, t, cfg));
}

TetGraphRoot true_curved_tet_root_at_st(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    real_t s,
    real_t t,
    Configuration cfg,
    real_t tol) {
  TetGraphRoot out;
  const auto span = tet_reference_span(s, t, tol);
  if (!span.valid || span.r_max - span.r_min <= positive_tolerance(tol)) {
    return out;
  }
  const real_t eps = positive_tolerance(tol) * real_t{10.0};
  out.f_min = signed_distance_on_tet(mesh, cell, embedded, span.r_min, s, t, cfg);
  out.f_max = signed_distance_on_tet(mesh, cell, embedded, span.r_max, s, t, cfg);
  if (std::abs(out.f_min) <= eps) {
    out.has_root = true;
    out.r = span.r_min;
    return out;
  }
  if (std::abs(out.f_max) <= eps) {
    out.has_root = true;
    out.r = span.r_max;
    return out;
  }
  if (!((out.f_min < -eps && out.f_max > eps) ||
        (out.f_min > eps && out.f_max < -eps))) {
    return out;
  }

  real_t a = span.r_min;
  real_t b = span.r_max;
  real_t fa = out.f_min;
  for (int iter = 0; iter < 96; ++iter) {
    const real_t m = real_t{0.5} * (a + b);
    const real_t fm = signed_distance_on_tet(mesh, cell, embedded, m, s, t, cfg);
    if (std::abs(fm) <= eps || std::abs(b - a) <= real_t{1.0e-14}) {
      a = m;
      b = m;
      break;
    }
    if ((fa < real_t{0.0} && fm > real_t{0.0}) ||
        (fa > real_t{0.0} && fm < real_t{0.0})) {
      b = m;
    } else {
      a = m;
      fa = fm;
    }
  }
  out.has_root = true;
  out.r = real_t{0.5} * (a + b);
  return out;
}

std::vector<TetBaseTriangle> make_uniform_tet_base_triangles(int segments) {
  segments = std::max(1, segments);
  const auto point = [segments](int i, int j) {
    TetBasePoint p;
    p.s = static_cast<real_t>(i) / static_cast<real_t>(segments);
    p.t = static_cast<real_t>(j) / static_cast<real_t>(segments);
    return p;
  };

  std::vector<TetBaseTriangle> out;
  out.reserve(static_cast<std::size_t>(segments * segments));
  for (int i = 0; i < segments; ++i) {
    for (int j = 0; j < segments - i; ++j) {
      out.push_back({{point(i, j), point(i + 1, j), point(i, j + 1)}});
      if (i + j + 2 <= segments) {
        out.push_back({{point(i + 1, j), point(i + 1, j + 1), point(i, j + 1)}});
      }
    }
  }
  return out;
}

real_t tet_base_triangle_jacobian(const TetBaseTriangle& tri) noexcept {
  const auto& a = tri.vertices[0];
  const auto& b = tri.vertices[1];
  const auto& c = tri.vertices[2];
  return std::abs((b.s - a.s) * (c.t - a.t) -
                  (b.t - a.t) * (c.s - a.s));
}

TetBasePoint interpolate_base_point(
    const TetBasePoint& a,
    const TetBasePoint& b,
    real_t u) noexcept {
  TetBasePoint out;
  out.s = (real_t{1.0} - u) * a.s + u * b.s;
  out.t = (real_t{1.0} - u) * a.t + u * b.t;
  out.value = (real_t{1.0} - u) * a.value + u * b.value;
  return out;
}

real_t tet_side_availability_value(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    const TetBasePoint& base,
    Configuration cfg,
    CutTopologySide side,
    real_t tol) {
  const auto span = tet_reference_span(base.s, base.t, tol);
  if (!span.valid || span.r_max - span.r_min <= positive_tolerance(tol)) {
    return -positive_tolerance(tol);
  }
  const real_t f_min =
      signed_distance_on_tet(mesh, cell, embedded, span.r_min, base.s, base.t, cfg);
  const real_t f_max =
      signed_distance_on_tet(mesh, cell, embedded, span.r_max, base.s, base.t, cfg);
  return side == CutTopologySide::Negative
             ? -std::min(f_min, f_max)
             : std::max(f_min, f_max);
}

std::vector<TetBasePoint> clip_base_polygon_by_availability(
    std::vector<TetBasePoint> polygon,
    real_t tol) {
  const real_t eps = positive_tolerance(tol);
  if (polygon.empty()) {
    return {};
  }

  std::vector<TetBasePoint> out;
  for (std::size_t i = 0; i < polygon.size(); ++i) {
    const auto& a = polygon[i];
    const auto& b = polygon[(i + 1u) % polygon.size()];
    const bool a_in = a.value >= -eps;
    const bool b_in = b.value >= -eps;
    if (a_in && b_in) {
      out.push_back(b);
    } else if (a_in != b_in) {
      real_t u = real_t{0.0};
      const real_t denom = a.value - b.value;
      if (std::abs(denom) > eps) {
        u = a.value / denom;
      }
      u = std::max(real_t{0.0}, std::min(real_t{1.0}, u));
      out.push_back(interpolate_base_point(a, b, u));
      if (!a_in && b_in) {
        out.push_back(b);
      }
    }
  }

  std::vector<TetBasePoint> unique;
  for (const auto& p : out) {
    if (std::find_if(unique.begin(), unique.end(), [&](const auto& q) {
          return std::abs(p.s - q.s) <= eps * real_t{10.0} &&
                 std::abs(p.t - q.t) <= eps * real_t{10.0};
        }) == unique.end()) {
      unique.push_back(p);
    }
  }
  return unique;
}

std::vector<TetBasePoint> clipped_base_polygon_for_side(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    const TetBaseTriangle& tri,
    Configuration cfg,
    CutTopologySide side,
    real_t tol) {
  std::vector<TetBasePoint> polygon;
  polygon.reserve(3u);
  for (auto vertex : tri.vertices) {
    vertex.value = tet_side_availability_value(
        mesh, cell, embedded, vertex, cfg, side, tol);
    polygon.push_back(vertex);
  }
  return clip_base_polygon_by_availability(std::move(polygon), tol);
}

TetColumnInterval tet_side_interval_at_base_point(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    const TetBasePoint& base,
    Configuration cfg,
    CutTopologySide side,
    real_t tol,
    bool include_degenerate_boundary) {
  TetColumnInterval interval;
  const auto span = tet_reference_span(base.s, base.t, tol);
  if (!span.valid) {
    return interval;
  }
  const real_t eps = positive_tolerance(tol) * real_t{10.0};
  const auto root =
      true_curved_tet_root_at_st(mesh, cell, embedded, base.s, base.t, cfg, tol);
  const bool want_negative = side == CutTopologySide::Negative;
  if (root.has_root) {
    const real_t left_len = root.r - span.r_min;
    const real_t right_len = span.r_max - root.r;
    bool left_matches = false;
    bool right_matches = false;
    if (left_len > eps) {
      const real_t probe = real_t{0.5} * (span.r_min + root.r);
      left_matches =
          (signed_distance_on_tet(mesh, cell, embedded, probe, base.s, base.t, cfg) <= eps) ==
          want_negative;
    }
    if (right_len > eps) {
      const real_t probe = real_t{0.5} * (root.r + span.r_max);
      right_matches =
          (signed_distance_on_tet(mesh, cell, embedded, probe, base.s, base.t, cfg) <= eps) ==
          want_negative;
    }
    if (left_matches || (include_degenerate_boundary && left_len <= eps && !right_matches)) {
      interval.available = true;
      interval.r_lo = span.r_min;
      interval.r_hi = root.r;
      return interval;
    }
    if (right_matches || (include_degenerate_boundary && right_len <= eps && !left_matches)) {
      interval.available = true;
      interval.r_lo = root.r;
      interval.r_hi = span.r_max;
      return interval;
    }
    if (include_degenerate_boundary) {
      interval.available = true;
      interval.r_lo = root.r;
      interval.r_hi = root.r;
    }
    return interval;
  }

  if (span.r_max - span.r_min <= eps) {
    return interval;
  }
  const real_t probe = real_t{0.5} * (span.r_min + span.r_max);
  const bool whole_interval_matches =
      (signed_distance_on_tet(mesh, cell, embedded, probe, base.s, base.t, cfg) <= eps) ==
      want_negative;
  if (whole_interval_matches) {
    interval.available = true;
    interval.r_lo = span.r_min;
    interval.r_hi = span.r_max;
  }
  return interval;
}

FaceGraphIntegral integrate_tet_graph_region(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    const TetBaseTriangle& base_tri,
    Configuration cfg,
    real_t tol,
    bool restrict_to_side,
    CutTopologySide side) {
  FaceGraphIntegral out;
  const real_t base_jac = tet_base_triangle_jacobian(base_tri);
  if (base_jac <= positive_tolerance(tol)) {
    return out;
  }
  const int order = std::max(1, mesh.geometry_order(cell));
  const auto u_rule = gauss_legendre_rule(std::min(12, std::max(4, order + 3)));
  const auto v_rule = gauss_legendre_rule(std::min(12, std::max(4, order + 3)));
  const auto r_rule = gauss_legendre_rule(std::min(12, std::max(4, order + 3)));
  out.available = true;

  const auto& a = base_tri.vertices[0];
  const auto& b = base_tri.vertices[1];
  const auto& c = base_tri.vertices[2];
  for (std::size_t iu = 0; iu < u_rule.points.size(); ++iu) {
    const real_t u = real_t{0.5} * (u_rule.points[iu] + real_t{1.0});
    const real_t wu = real_t{0.5} * u_rule.weights[iu];
    for (std::size_t iv = 0; iv < v_rule.points.size(); ++iv) {
      const real_t v = real_t{0.5} * (v_rule.points[iv] + real_t{1.0});
      const real_t wv = real_t{0.5} * v_rule.weights[iv];
      const real_t l0 = (real_t{1.0} - u) * (real_t{1.0} - v);
      const real_t l1 = u;
      const real_t l2 = (real_t{1.0} - u) * v;
      TetBasePoint base;
      base.s = l0 * a.s + l1 * b.s + l2 * c.s;
      base.t = l0 * a.t + l1 * b.t + l2 * c.t;
      const real_t base_weight = wu * wv * base_jac * (real_t{1.0} - u);
      const auto span = tet_reference_span(base.s, base.t, tol);
      if (!span.valid || span.r_max - span.r_min <= positive_tolerance(tol)) {
        continue;
      }

      real_t r_lo = span.r_min;
      real_t r_hi = span.r_max;
      if (restrict_to_side) {
        const auto interval = tet_side_interval_at_base_point(
            mesh, cell, embedded, base, cfg, side, tol, false);
        if (!interval.available) {
          continue;
        }
        r_lo = interval.r_lo;
        r_hi = interval.r_hi;
      }
      if (r_hi - r_lo <= positive_tolerance(tol)) {
        continue;
      }

      const real_t half_r = real_t{0.5} * (r_hi - r_lo);
      const real_t center_r = real_t{0.5} * (r_lo + r_hi);
      for (std::size_t ir = 0; ir < r_rule.points.size(); ++ir) {
        const real_t r = center_r + half_r * r_rule.points[ir];
        const auto eval = CurvilinearEvaluator::evaluate_geometry(
            mesh, cell, tet_parametric_point(r, base.s, base.t), cfg);
        const auto dx_dr = jacobian_action(eval.jacobian, {{real_t{1.0}, real_t{0.0}, real_t{0.0}}});
        const auto dx_ds = jacobian_action(eval.jacobian, {{real_t{0.0}, real_t{1.0}, real_t{0.0}}});
        const auto dx_dt = jacobian_action(eval.jacobian, {{real_t{0.0}, real_t{0.0}, real_t{1.0}}});
        const real_t jac = std::abs(determinant3(dx_dr, dx_ds, dx_dt));
        const real_t weight = base_weight * r_rule.weights[ir] * half_r * jac;
        if (!std::isfinite(weight)) {
          return {};
        }
        out.measure += weight;
        out.centroid = add(out.centroid, scale(eval.coordinates, weight));
      }
    }
  }
  if (out.measure > real_t{0.0}) {
    out.centroid = scale(out.centroid, real_t{1.0} / out.measure);
  }
  return out;
}

std::vector<CurvedPatchPoint> true_curved_tet_edge_roots(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    real_t tol) {
  constexpr std::array<std::array<real_t, 3>, 4> vertices{{
      {{0.0, 0.0, 0.0}},
      {{1.0, 0.0, 0.0}},
      {{0.0, 1.0, 0.0}},
      {{0.0, 0.0, 1.0}}}};
  constexpr std::array<std::array<int, 2>, 6> edges{{
      {{0, 1}}, {{0, 2}}, {{0, 3}}, {{1, 2}}, {{1, 3}}, {{2, 3}}}};
  const real_t eps = positive_tolerance(tol) * real_t{10.0};
  std::vector<CurvedPatchPoint> out;
  const auto add_root = [&](std::array<real_t, 3> xi) {
    for (const auto& existing : out) {
      if (norm(sub(existing.parent_parametric_coordinate, xi)) <= eps * real_t{100.0}) {
        return;
      }
    }
    CurvedPatchPoint point;
    point.parent_parametric_coordinate = xi;
    point.has_parent_parametric_coordinate = true;
    point.point =
        CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, cfg).coordinates;
    out.push_back(point);
  };

  for (const auto& edge : edges) {
    const auto a = vertices[static_cast<std::size_t>(edge[0])];
    const auto b = vertices[static_cast<std::size_t>(edge[1])];
    real_t fa =
        embedded.signed_distance(CurvilinearEvaluator::evaluate_geometry(mesh, cell, a, cfg).coordinates);
    const real_t fb0 =
        embedded.signed_distance(CurvilinearEvaluator::evaluate_geometry(mesh, cell, b, cfg).coordinates);
    if (std::abs(fa) <= eps) {
      add_root(a);
    }
    if (std::abs(fb0) <= eps) {
      add_root(b);
    }
    if (!((fa < -eps && fb0 > eps) || (fa > eps && fb0 < -eps))) {
      continue;
    }
    real_t lo = real_t{0.0};
    real_t hi = real_t{1.0};
    for (int iter = 0; iter < 96; ++iter) {
      const real_t mid = real_t{0.5} * (lo + hi);
      const auto xi = interpolate_parametric(a, b, mid);
      const real_t fm =
          embedded.signed_distance(CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, cfg).coordinates);
      if (std::abs(fm) <= eps || std::abs(hi - lo) <= real_t{1.0e-14}) {
        lo = mid;
        hi = mid;
        break;
      }
      if ((fa < real_t{0.0} && fm > real_t{0.0}) ||
          (fa > real_t{0.0} && fm < real_t{0.0})) {
        hi = mid;
      } else {
        lo = mid;
        fa = fm;
      }
    }
    add_root(interpolate_parametric(a, b, real_t{0.5} * (lo + hi)));
  }
  return out;
}

bool populate_true_curved_tet_patch_quadrature(
    CutCurvedPatchRecord& patch,
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    const std::vector<TetBaseTriangle>& base_triangles,
    real_t tol) {
  patch.quadrature_points.clear();
  patch.quadrature_normals.clear();
  patch.quadrature_weights.clear();
  patch.quadrature_measure = real_t{0.0};
  patch.isoparametric_quadrature_available = false;

  const int order = std::max(1, mesh.geometry_order(cell));
  const auto u_rule = gauss_legendre_rule(std::min(12, std::max(4, order + 3)));
  const auto v_rule = gauss_legendre_rule(std::min(12, std::max(4, order + 3)));
  for (const auto& tri : base_triangles) {
    const real_t base_jac = tet_base_triangle_jacobian(tri);
    if (base_jac <= positive_tolerance(tol)) {
      continue;
    }
    const auto& a = tri.vertices[0];
    const auto& b = tri.vertices[1];
    const auto& c = tri.vertices[2];
    for (std::size_t iu = 0; iu < u_rule.points.size(); ++iu) {
      const real_t u = real_t{0.5} * (u_rule.points[iu] + real_t{1.0});
      const real_t wu = real_t{0.5} * u_rule.weights[iu];
      for (std::size_t iv = 0; iv < v_rule.points.size(); ++iv) {
        const real_t v = real_t{0.5} * (v_rule.points[iv] + real_t{1.0});
        const real_t wv = real_t{0.5} * v_rule.weights[iv];
        const real_t l0 = (real_t{1.0} - u) * (real_t{1.0} - v);
        const real_t l1 = u;
        const real_t l2 = (real_t{1.0} - u) * v;
        const real_t s = l0 * a.s + l1 * b.s + l2 * c.s;
        const real_t t = l0 * a.t + l1 * b.t + l2 * c.t;
        const auto root = true_curved_tet_root_at_st(
            mesh, cell, embedded, s, t, cfg, tol);
        if (!root.has_root) {
          continue;
        }
        const auto xi = tet_parametric_point(root.r, s, t);
        const auto eval = CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, cfg);
        const auto dx_dr = jacobian_action(eval.jacobian, {{real_t{1.0}, real_t{0.0}, real_t{0.0}}});
        const auto dx_ds = jacobian_action(eval.jacobian, {{real_t{0.0}, real_t{1.0}, real_t{0.0}}});
        const auto dx_dt = jacobian_action(eval.jacobian, {{real_t{0.0}, real_t{0.0}, real_t{1.0}}});
        const real_t phi_r = dot(embedded.normal, dx_dr);
        const real_t phi_s = dot(embedded.normal, dx_ds);
        const real_t phi_t = dot(embedded.normal, dx_dt);
        if (std::abs(phi_r) <= positive_tolerance(tol)) {
          return false;
        }
        const real_t drds = -phi_s / phi_r;
        const real_t drdt = -phi_t / phi_r;
        const auto tangent_s = add(dx_ds, scale(dx_dr, drds));
        const auto tangent_t = add(dx_dt, scale(dx_dr, drdt));
        const real_t base_weight = wu * wv * base_jac * (real_t{1.0} - u);
        const real_t weight = base_weight * norm(cross(tangent_s, tangent_t));
        if (!std::isfinite(weight) || weight <= real_t{0.0}) {
          continue;
        }
        patch.quadrature_points.push_back(eval.coordinates);
        patch.quadrature_normals.push_back(unit_or_default(embedded.outward_normal(eval.coordinates)));
        patch.quadrature_weights.push_back(weight);
        patch.quadrature_measure += weight;
      }
    }
  }

  patch.isoparametric_quadrature_available =
      patch.quadrature_measure > positive_tolerance(tol);
  patch.construction_policy = kTrueCurvedArrangementPolicy;
  return patch.isoparametric_quadrature_available;
}

bool add_true_curved_tet_graph_subcell(
    SideClosedTopology& closed,
    const MeshBase& mesh,
    const CutEntityRecord& cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    CutTopologySide side,
    const std::vector<index_t>& dofs,
    const std::vector<std::array<real_t, 3>>& parent_points,
    const std::vector<TetBasePoint>& base_polygon,
    const FaceGraphIntegral& measure,
    real_t tol,
    std::uint64_t salt) {
  if (!measure.available || measure.measure <= positive_tolerance(tol) ||
      base_polygon.size() < 3u) {
    return false;
  }

  const auto collect_vertices = [&]() {
    std::vector<std::array<real_t, 3>> vertices;
    vertices.reserve(base_polygon.size() * 2u);
    for (const auto& base : base_polygon) {
      const auto interval = tet_side_interval_at_base_point(
          mesh, cell.entity, embedded, base, cfg, side, tol, true);
      if (!interval.available) {
        continue;
      }
      vertices.push_back(tet_parametric_point(interval.r_lo, base.s, base.t));
      if (std::abs(interval.r_hi - interval.r_lo) > positive_tolerance(tol) * real_t{10.0}) {
        vertices.push_back(tet_parametric_point(interval.r_hi, base.s, base.t));
      }
    }
    return unique_xi_vertices(std::move(vertices), tol);
  };

  auto xi_vertices = collect_vertices();
  auto faces = convex_hull_faces_indices(xi_vertices, tol);
  if (xi_vertices.size() < 4u || faces.size() < 4u) {
    TetBasePoint centroid;
    for (const auto& base : base_polygon) {
      centroid.s += base.s;
      centroid.t += base.t;
    }
    centroid.s /= static_cast<real_t>(base_polygon.size());
    centroid.t /= static_cast<real_t>(base_polygon.size());
    const auto center_interval = tet_side_interval_at_base_point(
        mesh, cell.entity, embedded, centroid, cfg, side, tol, false);
    if (!center_interval.available ||
        center_interval.r_hi - center_interval.r_lo <= positive_tolerance(tol)) {
      return false;
    }

    const auto nearby_base = [&](std::size_t i) {
      TetBasePoint p;
      const auto& target = base_polygon[i % base_polygon.size()];
      p.s = real_t{0.75} * centroid.s + real_t{0.25} * target.s;
      p.t = real_t{0.75} * centroid.t + real_t{0.25} * target.t;
      return p;
    };
    const auto p1 = nearby_base(0u);
    const auto p2 = nearby_base(1u);
    const auto i1 = tet_side_interval_at_base_point(
        mesh, cell.entity, embedded, p1, cfg, side, tol, false);
    const auto i2 = tet_side_interval_at_base_point(
        mesh, cell.entity, embedded, p2, cfg, side, tol, false);
    if (!i1.available || !i2.available) {
      return false;
    }
    xi_vertices = unique_xi_vertices(
        {tet_parametric_point(center_interval.r_lo, centroid.s, centroid.t),
         tet_parametric_point(center_interval.r_hi, centroid.s, centroid.t),
         tet_parametric_point(i1.r_lo, p1.s, p1.t),
         tet_parametric_point(i2.r_lo, p2.s, p2.t)},
        tol);
    faces = convex_hull_faces_indices(xi_vertices, tol);
    if (xi_vertices.size() < 4u || faces.size() < 4u) {
      return false;
    }
  }

  std::vector<std::array<real_t, 3>> physical_points;
  physical_points.reserve(xi_vertices.size());
  for (const auto& xi : xi_vertices) {
    physical_points.push_back(
        CurvilinearEvaluator::evaluate_geometry(mesh, cell.entity, xi, cfg).coordinates);
  }
  const auto before = closed.subcells.size();
  add_integration_subcell(closed,
                          xi_vertices.size() == 4u ? CellFamily::Tetra : CellFamily::Polyhedron,
                          physical_points,
                          faces,
                          measure.measure,
                          measure.centroid,
                          dofs,
                          parent_points,
                          embedded,
                          cell.provenance,
                          cell.global_id,
                          side,
                          tol,
                          salt,
                          xi_vertices,
                          true,
                          true,
                          kTrueCurvedArrangementPolicy);
  return closed.subcells.size() > before;
}

CutSideRegion make_true_curved_tet_side_region(
    const MeshBase& mesh,
    const CutEntityRecord& cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    CutTopologySide side,
    const std::vector<TetBaseTriangle>& base_triangles,
    const std::vector<std::uint64_t>& interface_vertex_ids,
    real_t tol) {
  CutSideRegion region;
  region.side = side;
  region.parent_cell = cell.entity;
  region.parent_cell_gid = cell.global_id;
  region.provenance = cell.provenance;
  region.integration_family = CellFamily::Tetra;
  region.curved_isoparametric_topology = true;
  region.measure_from_linear_topology = true;
  region.stable_id = stable_entity_id(cell.provenance,
                                      cell.global_id,
                                      0u,
                                      static_cast<index_t>(side),
                                      real_t{0.0},
                                      404u);
  region.cut_vertices = interface_vertex_ids;

  const auto dofs = mesh.cell_geometry_dofs(cell.entity);
  std::vector<std::array<real_t, 3>> parent_points;
  parent_points.reserve(dofs.size());
  std::vector<real_t> distances;
  distances.reserve(dofs.size());
  for (const auto dof : dofs) {
    const auto point = mesh.geometry_dof_coords(dof, cfg);
    parent_points.push_back(point);
    distances.push_back(embedded.signed_distance(point));
  }
  for (std::size_t i = 0; i < dofs.size(); ++i) {
    const bool on_side = side == CutTopologySide::Negative
                             ? distances[i] <= positive_tolerance(tol)
                             : distances[i] >= -positive_tolerance(tol);
    if (on_side) {
      region.parent_geometry_dofs.push_back(dofs[i]);
    }
  }

  SideClosedTopology closed;
  std::array<real_t, 3> weighted_centroid{{0.0, 0.0, 0.0}};
  std::uint64_t subcell_salt = 0u;
  for (const auto& base_tri : base_triangles) {
    const auto full_measure =
        integrate_tet_graph_region(mesh,
                                   cell.entity,
                                   embedded,
                                   base_tri,
                                   cfg,
                                   tol,
                                   false,
                                   side);
    if (full_measure.available) {
      region.parent_measure += full_measure.measure;
    }

    const auto measure =
        integrate_tet_graph_region(mesh,
                                   cell.entity,
                                   embedded,
                                   base_tri,
                                   cfg,
                                   tol,
                                   true,
                                   side);
    if (!measure.available || measure.measure <= positive_tolerance(tol)) {
      continue;
    }
    const auto base_polygon =
        clipped_base_polygon_for_side(mesh, cell.entity, embedded, base_tri, cfg, side, tol);
    if (add_true_curved_tet_graph_subcell(closed,
                                          mesh,
                                          cell,
                                          embedded,
                                          cfg,
                                          side,
                                          dofs,
                                          parent_points,
                                          base_polygon,
                                          measure,
                                          tol,
                                          2600u + subcell_salt++)) {
      region.measure_estimate += measure.measure;
      weighted_centroid = add(weighted_centroid, scale(measure.centroid, measure.measure));
    }
  }

  if (region.measure_estimate > real_t{0.0}) {
    region.centroid_estimate =
        scale(weighted_centroid, real_t{1.0} / region.measure_estimate);
    region.volume_fraction_estimate = region.parent_measure > real_t{0.0}
                                          ? region.measure_estimate / region.parent_measure
                                          : real_t{0.0};
  }
  region.integration_vertices = closed.vertices;
  region.integration_subcells = closed.subcells;
  region.integration_region_vertices = closed.vertex_ids;
  region.integration_region_faces = closed.faces;
  region.closed_integration_topology =
      !region.integration_subcells.empty() &&
      std::all_of(region.integration_subcells.begin(),
                  region.integration_subcells.end(),
                  [](const auto& subcell) {
                    return subcell.closed_topology;
                  });
  return region;
}

bool supports_true_curved_tet_arrangement(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded) noexcept {
  return mesh.cell_shape(cell).family == CellFamily::Tetra &&
         mesh.geometry_order(cell) > 1 &&
         embedded.kind == EmbeddedGeometryKind::Plane;
}

bool add_true_curved_tet_arrangement(
    CutTopologyRecord& topology,
    const MeshBase& mesh,
    const CutEntityRecord& cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    real_t tol,
    std::uint64_t& h) {
  const int order = std::max(1, mesh.geometry_order(cell.entity));
  const int base_segments = std::max(24, order * 12);
  const auto base_triangles = make_uniform_tet_base_triangles(base_segments);
  const auto candidates =
      true_curved_tet_edge_roots(mesh, cell.entity, embedded, cfg, tol);
  if (candidates.size() < 3u) {
    topology.diagnostics.push_back(
        "true curved tetra arrangement found fewer than three boundary roots in a cut high-order tetra cell");
    return false;
  }

  const auto first_patch = topology.curved_patches.size();
  const bool recorded =
      record_curved_patch_from_candidates(topology,
                                          mesh,
                                          cell.entity,
                                          embedded,
                                          cfg,
                                          cell.provenance,
                                          cell.global_id,
                                          candidates,
                                          tol,
                                          2800u,
                                          true,
                                          kTrueCurvedArrangementPolicy);
  if (!recorded || topology.curved_patches.size() <= first_patch) {
    topology.diagnostics.push_back(
        "true curved tetra arrangement failed to record interface surface topology");
    return false;
  }
  auto& patch = topology.curved_patches.back();
  if (!populate_true_curved_tet_patch_quadrature(
          patch, mesh, cell.entity, embedded, cfg, base_triangles, tol)) {
    topology.diagnostics.push_back(
        "true curved tetra arrangement failed to build interface surface quadrature");
    return false;
  }
  const auto interface_vertex_ids = patch.ordered_vertices;
  hash_curved_patch_record(h, patch);

  for (const auto side : {CutTopologySide::Negative, CutTopologySide::Positive}) {
    auto region = make_true_curved_tet_side_region(mesh,
                                                   cell,
                                                   embedded,
                                                   cfg,
                                                   side,
                                                   base_triangles,
                                                   interface_vertex_ids,
                                                   tol);
    if (!region.closed_integration_topology || region.integration_subcells.empty()) {
      topology.diagnostics.push_back(
          "true curved tetra arrangement failed to construct closed side-region topology");
      return false;
    }
    topology.side_regions.push_back(std::move(region));
    hash_side_region_record(h, topology.side_regions.back());
  }

  return true;
}

struct HexBaseQuad {
  std::array<TetBasePoint, 4> vertices{};
};

std::array<real_t, 3> hex_parametric_point(real_t r, real_t s, real_t t) noexcept {
  return {{r, s, t}};
}

std::array<real_t, 3> evaluate_hex_geometry_point(
    const MeshBase& mesh,
    index_t cell,
    real_t r,
    real_t s,
    real_t t,
    Configuration cfg) {
  return CurvilinearEvaluator::evaluate_geometry(
      mesh, cell, hex_parametric_point(r, s, t), cfg).coordinates;
}

real_t signed_distance_on_hex(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    real_t r,
    real_t s,
    real_t t,
    Configuration cfg) {
  return embedded.signed_distance(evaluate_hex_geometry_point(mesh, cell, r, s, t, cfg));
}

std::size_t count_true_curved_hex_column_roots(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    real_t s,
    real_t t,
    Configuration cfg,
    real_t tol,
    bool& valid) {
  valid = true;
  const int order = std::max(1, mesh.geometry_order(cell));
  const int segments = std::max(32, order * 16);
  const real_t eps = positive_tolerance(tol) * real_t{10.0};
  const real_t merge_tol = std::max(real_t{1.0e-12}, eps * real_t{10.0});
  std::vector<real_t> roots;
  roots.reserve(4u);

  const auto add_root = [&](real_t r) {
    for (const auto existing : roots) {
      if (std::abs(existing - r) <= merge_tol) {
        return;
      }
    }
    roots.push_back(std::max(real_t{-1.0}, std::min(real_t{1.0}, r)));
  };

  real_t r_prev = real_t{-1.0};
  real_t f_prev = signed_distance_on_hex(mesh, cell, embedded, r_prev, s, t, cfg);
  if (!std::isfinite(f_prev)) {
    valid = false;
    return 0u;
  }
  if (std::abs(f_prev) <= eps) {
    add_root(r_prev);
  }

  for (int i = 1; i <= segments; ++i) {
    const real_t r = real_t{-1.0} + real_t{2.0} * static_cast<real_t>(i) /
                                       static_cast<real_t>(segments);
    const real_t f = signed_distance_on_hex(mesh, cell, embedded, r, s, t, cfg);
    if (!std::isfinite(f)) {
      valid = false;
      return roots.size();
    }
    if (std::abs(f) <= eps) {
      add_root(r);
    }
    if ((f_prev < -eps && f > eps) || (f_prev > eps && f < -eps)) {
      real_t a = r_prev;
      real_t b = r;
      real_t fa = f_prev;
      for (int iter = 0; iter < 96; ++iter) {
        const real_t m = real_t{0.5} * (a + b);
        const real_t fm = signed_distance_on_hex(mesh, cell, embedded, m, s, t, cfg);
        if (!std::isfinite(fm)) {
          valid = false;
          return roots.size();
        }
        if (std::abs(fm) <= eps || std::abs(b - a) <= real_t{1.0e-14}) {
          a = m;
          b = m;
          break;
        }
        if ((fa < real_t{0.0} && fm > real_t{0.0}) ||
            (fa > real_t{0.0} && fm < real_t{0.0})) {
          b = m;
        } else {
          a = m;
          fa = fm;
        }
      }
      add_root(real_t{0.5} * (a + b));
    }
    r_prev = r;
    f_prev = f;
  }

  return roots.size();
}

bool true_curved_hex_columns_are_graph_compatible(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    real_t tol) {
  const int order = std::max(1, mesh.geometry_order(cell));
  const int samples = std::max(8, order * 8);
  for (int i = 0; i <= samples; ++i) {
    const real_t s = real_t{-1.0} + real_t{2.0} * static_cast<real_t>(i) /
                                       static_cast<real_t>(samples);
    for (int j = 0; j <= samples; ++j) {
      const real_t t = real_t{-1.0} + real_t{2.0} * static_cast<real_t>(j) /
                                         static_cast<real_t>(samples);
      bool valid = true;
      const auto roots =
          count_true_curved_hex_column_roots(mesh, cell, embedded, s, t, cfg, tol, valid);
      if (!valid || roots > 1u) {
        return false;
      }
    }
  }
  return true;
}

TetGraphRoot true_curved_hex_root_at_st(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    real_t s,
    real_t t,
    Configuration cfg,
    real_t tol) {
  TetGraphRoot out;
  const real_t eps = positive_tolerance(tol) * real_t{10.0};
  out.f_min = signed_distance_on_hex(mesh, cell, embedded, real_t{-1.0}, s, t, cfg);
  out.f_max = signed_distance_on_hex(mesh, cell, embedded, real_t{1.0}, s, t, cfg);
  if (std::abs(out.f_min) <= eps) {
    out.has_root = true;
    out.r = real_t{-1.0};
    return out;
  }
  if (std::abs(out.f_max) <= eps) {
    out.has_root = true;
    out.r = real_t{1.0};
    return out;
  }
  if (!((out.f_min < -eps && out.f_max > eps) ||
        (out.f_min > eps && out.f_max < -eps))) {
    return out;
  }

  real_t a = real_t{-1.0};
  real_t b = real_t{1.0};
  real_t fa = out.f_min;
  for (int iter = 0; iter < 96; ++iter) {
    const real_t m = real_t{0.5} * (a + b);
    const real_t fm = signed_distance_on_hex(mesh, cell, embedded, m, s, t, cfg);
    if (std::abs(fm) <= eps || std::abs(b - a) <= real_t{1.0e-14}) {
      a = m;
      b = m;
      break;
    }
    if ((fa < real_t{0.0} && fm > real_t{0.0}) ||
        (fa > real_t{0.0} && fm < real_t{0.0})) {
      b = m;
    } else {
      a = m;
      fa = fm;
    }
  }
  out.has_root = true;
  out.r = real_t{0.5} * (a + b);
  return out;
}

std::vector<HexBaseQuad> make_uniform_hex_base_quads(int segments) {
  segments = std::max(1, segments);
  const auto point = [segments](int i, int j) {
    TetBasePoint p;
    p.s = real_t{-1.0} + real_t{2.0} * static_cast<real_t>(i) /
                             static_cast<real_t>(segments);
    p.t = real_t{-1.0} + real_t{2.0} * static_cast<real_t>(j) /
                             static_cast<real_t>(segments);
    return p;
  };

  std::vector<HexBaseQuad> out;
  out.reserve(static_cast<std::size_t>(segments * segments));
  for (int i = 0; i < segments; ++i) {
    for (int j = 0; j < segments; ++j) {
      out.push_back({{point(i, j),
                      point(i + 1, j),
                      point(i + 1, j + 1),
                      point(i, j + 1)}});
    }
  }
  return out;
}

real_t hex_side_availability_value(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    const TetBasePoint& base,
    Configuration cfg,
    CutTopologySide side,
    real_t tol) {
  const real_t f_min =
      signed_distance_on_hex(mesh, cell, embedded, real_t{-1.0}, base.s, base.t, cfg);
  const real_t f_max =
      signed_distance_on_hex(mesh, cell, embedded, real_t{1.0}, base.s, base.t, cfg);
  (void)tol;
  return side == CutTopologySide::Negative
             ? -std::min(f_min, f_max)
             : std::max(f_min, f_max);
}

std::vector<TetBasePoint> clipped_hex_base_polygon_for_side(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    const HexBaseQuad& quad,
    Configuration cfg,
    CutTopologySide side,
    real_t tol) {
  std::vector<TetBasePoint> polygon;
  polygon.reserve(4u);
  for (auto vertex : quad.vertices) {
    vertex.value = hex_side_availability_value(
        mesh, cell, embedded, vertex, cfg, side, tol);
    polygon.push_back(vertex);
  }
  return clip_base_polygon_by_availability(std::move(polygon), tol);
}

TetColumnInterval hex_side_interval_at_base_point(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    const TetBasePoint& base,
    Configuration cfg,
    CutTopologySide side,
    real_t tol,
    bool include_degenerate_boundary) {
  TetColumnInterval interval;
  const real_t eps = positive_tolerance(tol) * real_t{10.0};
  const auto root =
      true_curved_hex_root_at_st(mesh, cell, embedded, base.s, base.t, cfg, tol);
  const bool want_negative = side == CutTopologySide::Negative;
  if (root.has_root) {
    const real_t left_len = root.r - real_t{-1.0};
    const real_t right_len = real_t{1.0} - root.r;
    bool left_matches = false;
    bool right_matches = false;
    if (left_len > eps) {
      const real_t probe = real_t{0.5} * (real_t{-1.0} + root.r);
      left_matches =
          (signed_distance_on_hex(mesh, cell, embedded, probe, base.s, base.t, cfg) <= eps) ==
          want_negative;
    }
    if (right_len > eps) {
      const real_t probe = real_t{0.5} * (root.r + real_t{1.0});
      right_matches =
          (signed_distance_on_hex(mesh, cell, embedded, probe, base.s, base.t, cfg) <= eps) ==
          want_negative;
    }
    if (left_matches || (include_degenerate_boundary && left_len <= eps && !right_matches)) {
      interval.available = true;
      interval.r_lo = real_t{-1.0};
      interval.r_hi = root.r;
      return interval;
    }
    if (right_matches || (include_degenerate_boundary && right_len <= eps && !left_matches)) {
      interval.available = true;
      interval.r_lo = root.r;
      interval.r_hi = real_t{1.0};
      return interval;
    }
    if (include_degenerate_boundary) {
      interval.available = true;
      interval.r_lo = root.r;
      interval.r_hi = root.r;
    }
    return interval;
  }

  const bool whole_interval_matches =
      (signed_distance_on_hex(mesh, cell, embedded, real_t{0.0}, base.s, base.t, cfg) <= eps) ==
      want_negative;
  if (whole_interval_matches) {
    interval.available = true;
    interval.r_lo = real_t{-1.0};
    interval.r_hi = real_t{1.0};
  }
  return interval;
}

FaceGraphIntegral integrate_hex_graph_region(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    const HexBaseQuad& base_quad,
    Configuration cfg,
    real_t tol,
    bool restrict_to_side,
    CutTopologySide side) {
  FaceGraphIntegral out;
  const auto& a = base_quad.vertices[0];
  const auto& b = base_quad.vertices[1];
  const auto& d = base_quad.vertices[3];
  const real_t ds = b.s - a.s;
  const real_t dt = d.t - a.t;
  if (std::abs(ds * dt) <= positive_tolerance(tol)) {
    return out;
  }
  const int order = std::max(1, mesh.geometry_order(cell));
  const auto s_rule = gauss_legendre_rule(std::min(12, std::max(4, order + 3)));
  const auto t_rule = gauss_legendre_rule(std::min(12, std::max(4, order + 3)));
  const auto r_rule = gauss_legendre_rule(std::min(12, std::max(4, order + 3)));
  const real_t half_s = real_t{0.5} * ds;
  const real_t half_t = real_t{0.5} * dt;
  const real_t center_s = real_t{0.5} * (a.s + b.s);
  const real_t center_t = real_t{0.5} * (a.t + d.t);
  out.available = true;

  for (std::size_t is = 0; is < s_rule.points.size(); ++is) {
    const real_t s = center_s + half_s * s_rule.points[is];
    for (std::size_t it = 0; it < t_rule.points.size(); ++it) {
      const real_t t = center_t + half_t * t_rule.points[it];
      TetBasePoint base;
      base.s = s;
      base.t = t;
      real_t r_lo = real_t{-1.0};
      real_t r_hi = real_t{1.0};
      if (restrict_to_side) {
        const auto interval = hex_side_interval_at_base_point(
            mesh, cell, embedded, base, cfg, side, tol, false);
        if (!interval.available) {
          continue;
        }
        r_lo = interval.r_lo;
        r_hi = interval.r_hi;
      }
      if (r_hi - r_lo <= positive_tolerance(tol)) {
        continue;
      }

      const real_t half_r = real_t{0.5} * (r_hi - r_lo);
      const real_t center_r = real_t{0.5} * (r_lo + r_hi);
      for (std::size_t ir = 0; ir < r_rule.points.size(); ++ir) {
        const real_t r = center_r + half_r * r_rule.points[ir];
        const auto eval = CurvilinearEvaluator::evaluate_geometry(
            mesh, cell, hex_parametric_point(r, s, t), cfg);
        const auto dx_dr = jacobian_action(eval.jacobian, {{real_t{1.0}, real_t{0.0}, real_t{0.0}}});
        const auto dx_ds = jacobian_action(eval.jacobian, {{real_t{0.0}, real_t{1.0}, real_t{0.0}}});
        const auto dx_dt = jacobian_action(eval.jacobian, {{real_t{0.0}, real_t{0.0}, real_t{1.0}}});
        const real_t jac = std::abs(determinant3(dx_dr, dx_ds, dx_dt));
        const real_t weight =
            s_rule.weights[is] * t_rule.weights[it] * r_rule.weights[ir] *
            std::abs(half_s * half_t) * half_r * jac;
        if (!std::isfinite(weight)) {
          return {};
        }
        out.measure += weight;
        out.centroid = add(out.centroid, scale(eval.coordinates, weight));
      }
    }
  }
  if (out.measure > real_t{0.0}) {
    out.centroid = scale(out.centroid, real_t{1.0} / out.measure);
  }
  return out;
}

std::vector<CurvedPatchPoint> true_curved_hex_edge_roots(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    real_t tol) {
  constexpr std::array<std::array<real_t, 3>, 8> vertices{{
      {{-1.0, -1.0, -1.0}},
      {{ 1.0, -1.0, -1.0}},
      {{ 1.0,  1.0, -1.0}},
      {{-1.0,  1.0, -1.0}},
      {{-1.0, -1.0,  1.0}},
      {{ 1.0, -1.0,  1.0}},
      {{ 1.0,  1.0,  1.0}},
      {{-1.0,  1.0,  1.0}}}};
  const auto eview = CellTopology::get_edges_view(CellFamily::Hex);
  const real_t eps = positive_tolerance(tol) * real_t{10.0};
  std::vector<CurvedPatchPoint> out;
  const auto add_root = [&](std::array<real_t, 3> xi) {
    for (const auto& existing : out) {
      if (norm(sub(existing.parent_parametric_coordinate, xi)) <= eps * real_t{100.0}) {
        return;
      }
    }
    CurvedPatchPoint point;
    point.parent_parametric_coordinate = xi;
    point.has_parent_parametric_coordinate = true;
    point.point =
        CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, cfg).coordinates;
    out.push_back(point);
  };

  for (int ei = 0; ei < eview.edge_count; ++ei) {
    const auto a = vertices[static_cast<std::size_t>(eview.pairs_flat[2 * ei + 0])];
    const auto b = vertices[static_cast<std::size_t>(eview.pairs_flat[2 * ei + 1])];
    real_t fa =
        embedded.signed_distance(CurvilinearEvaluator::evaluate_geometry(mesh, cell, a, cfg).coordinates);
    const real_t fb0 =
        embedded.signed_distance(CurvilinearEvaluator::evaluate_geometry(mesh, cell, b, cfg).coordinates);
    if (std::abs(fa) <= eps) {
      add_root(a);
    }
    if (std::abs(fb0) <= eps) {
      add_root(b);
    }
    if (!((fa < -eps && fb0 > eps) || (fa > eps && fb0 < -eps))) {
      continue;
    }
    real_t lo = real_t{0.0};
    real_t hi = real_t{1.0};
    for (int iter = 0; iter < 96; ++iter) {
      const real_t mid = real_t{0.5} * (lo + hi);
      const auto xi = interpolate_parametric(a, b, mid);
      const real_t fm =
          embedded.signed_distance(CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, cfg).coordinates);
      if (std::abs(fm) <= eps || std::abs(hi - lo) <= real_t{1.0e-14}) {
        lo = mid;
        hi = mid;
        break;
      }
      if ((fa < real_t{0.0} && fm > real_t{0.0}) ||
          (fa > real_t{0.0} && fm < real_t{0.0})) {
        hi = mid;
      } else {
        lo = mid;
        fa = fm;
      }
    }
    add_root(interpolate_parametric(a, b, real_t{0.5} * (lo + hi)));
  }
  return out;
}

bool populate_true_curved_hex_patch_quadrature(
    CutCurvedPatchRecord& patch,
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    const std::vector<HexBaseQuad>& base_quads,
    real_t tol) {
  patch.quadrature_points.clear();
  patch.quadrature_normals.clear();
  patch.quadrature_weights.clear();
  patch.quadrature_measure = real_t{0.0};
  patch.isoparametric_quadrature_available = false;

  const int order = std::max(1, mesh.geometry_order(cell));
  const auto s_rule = gauss_legendre_rule(std::min(12, std::max(4, order + 3)));
  const auto t_rule = gauss_legendre_rule(std::min(12, std::max(4, order + 3)));
  for (const auto& quad : base_quads) {
    const auto& a = quad.vertices[0];
    const auto& b = quad.vertices[1];
    const auto& d = quad.vertices[3];
    const real_t half_s = real_t{0.5} * (b.s - a.s);
    const real_t half_t = real_t{0.5} * (d.t - a.t);
    if (std::abs(half_s * half_t) <= positive_tolerance(tol)) {
      continue;
    }
    const real_t center_s = real_t{0.5} * (a.s + b.s);
    const real_t center_t = real_t{0.5} * (a.t + d.t);
    for (std::size_t is = 0; is < s_rule.points.size(); ++is) {
      const real_t s = center_s + half_s * s_rule.points[is];
      for (std::size_t it = 0; it < t_rule.points.size(); ++it) {
        const real_t t = center_t + half_t * t_rule.points[it];
        const auto root =
            true_curved_hex_root_at_st(mesh, cell, embedded, s, t, cfg, tol);
        if (!root.has_root) {
          continue;
        }
        const auto xi = hex_parametric_point(root.r, s, t);
        const auto eval = CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, cfg);
        const auto dx_dr = jacobian_action(eval.jacobian, {{real_t{1.0}, real_t{0.0}, real_t{0.0}}});
        const auto dx_ds = jacobian_action(eval.jacobian, {{real_t{0.0}, real_t{1.0}, real_t{0.0}}});
        const auto dx_dt = jacobian_action(eval.jacobian, {{real_t{0.0}, real_t{0.0}, real_t{1.0}}});
        const real_t phi_r = dot(embedded.normal, dx_dr);
        const real_t phi_s = dot(embedded.normal, dx_ds);
        const real_t phi_t = dot(embedded.normal, dx_dt);
        if (std::abs(phi_r) <= positive_tolerance(tol)) {
          return false;
        }
        const real_t drds = -phi_s / phi_r;
        const real_t drdt = -phi_t / phi_r;
        const auto tangent_s = add(dx_ds, scale(dx_dr, drds));
        const auto tangent_t = add(dx_dt, scale(dx_dr, drdt));
        const real_t weight =
            s_rule.weights[is] * t_rule.weights[it] * std::abs(half_s * half_t) *
            norm(cross(tangent_s, tangent_t));
        if (!std::isfinite(weight) || weight <= real_t{0.0}) {
          continue;
        }
        patch.quadrature_points.push_back(eval.coordinates);
        patch.quadrature_normals.push_back(unit_or_default(embedded.outward_normal(eval.coordinates)));
        patch.quadrature_weights.push_back(weight);
        patch.quadrature_measure += weight;
      }
    }
  }

  patch.isoparametric_quadrature_available =
      patch.quadrature_measure > positive_tolerance(tol);
  patch.construction_policy = kTrueCurvedArrangementPolicy;
  return patch.isoparametric_quadrature_available;
}

bool add_true_curved_hex_graph_subcell(
    SideClosedTopology& closed,
    const MeshBase& mesh,
    const CutEntityRecord& cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    CutTopologySide side,
    const std::vector<index_t>& dofs,
    const std::vector<std::array<real_t, 3>>& parent_points,
    const std::vector<TetBasePoint>& base_polygon,
    const FaceGraphIntegral& measure,
    real_t tol,
    std::uint64_t salt) {
  if (!measure.available || measure.measure <= positive_tolerance(tol) ||
      base_polygon.size() < 3u) {
    return false;
  }

  std::vector<std::array<real_t, 3>> xi_vertices;
  xi_vertices.reserve(base_polygon.size() * 2u);
  for (const auto& base : base_polygon) {
    const auto interval = hex_side_interval_at_base_point(
        mesh, cell.entity, embedded, base, cfg, side, tol, true);
    if (!interval.available) {
      continue;
    }
    xi_vertices.push_back(hex_parametric_point(interval.r_lo, base.s, base.t));
    if (std::abs(interval.r_hi - interval.r_lo) > positive_tolerance(tol) * real_t{10.0}) {
      xi_vertices.push_back(hex_parametric_point(interval.r_hi, base.s, base.t));
    }
  }
  xi_vertices = unique_xi_vertices(xi_vertices, tol);
  auto faces = convex_hull_faces_indices(xi_vertices, tol);
  if (xi_vertices.size() < 4u || faces.size() < 4u) {
    return false;
  }

  std::vector<std::array<real_t, 3>> physical_points;
  physical_points.reserve(xi_vertices.size());
  for (const auto& xi : xi_vertices) {
    physical_points.push_back(
        CurvilinearEvaluator::evaluate_geometry(mesh, cell.entity, xi, cfg).coordinates);
  }
  const auto before = closed.subcells.size();
  add_integration_subcell(closed,
                          xi_vertices.size() == 8u ? CellFamily::Hex : CellFamily::Polyhedron,
                          physical_points,
                          faces,
                          measure.measure,
                          measure.centroid,
                          dofs,
                          parent_points,
                          embedded,
                          cell.provenance,
                          cell.global_id,
                          side,
                          tol,
                          salt,
                          xi_vertices,
                          true,
                          true,
                          kTrueCurvedArrangementPolicy);
  return closed.subcells.size() > before;
}

CutSideRegion make_true_curved_hex_side_region(
    const MeshBase& mesh,
    const CutEntityRecord& cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    CutTopologySide side,
    const std::vector<HexBaseQuad>& base_quads,
    const std::vector<std::uint64_t>& interface_vertex_ids,
    real_t tol) {
  CutSideRegion region;
  region.side = side;
  region.parent_cell = cell.entity;
  region.parent_cell_gid = cell.global_id;
  region.provenance = cell.provenance;
  region.integration_family = CellFamily::Hex;
  region.curved_isoparametric_topology = true;
  region.measure_from_linear_topology = true;
  region.stable_id = stable_entity_id(cell.provenance,
                                      cell.global_id,
                                      0u,
                                      static_cast<index_t>(side),
                                      real_t{0.0},
                                      404u);
  region.cut_vertices = interface_vertex_ids;

  const auto dofs = mesh.cell_geometry_dofs(cell.entity);
  std::vector<std::array<real_t, 3>> parent_points;
  parent_points.reserve(dofs.size());
  std::vector<real_t> distances;
  distances.reserve(dofs.size());
  for (const auto dof : dofs) {
    const auto point = mesh.geometry_dof_coords(dof, cfg);
    parent_points.push_back(point);
    distances.push_back(embedded.signed_distance(point));
  }
  for (std::size_t i = 0; i < dofs.size(); ++i) {
    const bool on_side = side == CutTopologySide::Negative
                             ? distances[i] <= positive_tolerance(tol)
                             : distances[i] >= -positive_tolerance(tol);
    if (on_side) {
      region.parent_geometry_dofs.push_back(dofs[i]);
    }
  }

  SideClosedTopology closed;
  std::array<real_t, 3> weighted_centroid{{0.0, 0.0, 0.0}};
  std::uint64_t subcell_salt = 0u;
  for (const auto& base_quad : base_quads) {
    const auto full_measure =
        integrate_hex_graph_region(mesh,
                                   cell.entity,
                                   embedded,
                                   base_quad,
                                   cfg,
                                   tol,
                                   false,
                                   side);
    if (full_measure.available) {
      region.parent_measure += full_measure.measure;
    }

    const auto measure =
        integrate_hex_graph_region(mesh,
                                   cell.entity,
                                   embedded,
                                   base_quad,
                                   cfg,
                                   tol,
                                   true,
                                   side);
    if (!measure.available || measure.measure <= positive_tolerance(tol)) {
      continue;
    }
    const auto base_polygon =
        clipped_hex_base_polygon_for_side(mesh, cell.entity, embedded, base_quad, cfg, side, tol);
    if (add_true_curved_hex_graph_subcell(closed,
                                          mesh,
                                          cell,
                                          embedded,
                                          cfg,
                                          side,
                                          dofs,
                                          parent_points,
                                          base_polygon,
                                          measure,
                                          tol,
                                          3000u + subcell_salt++)) {
      region.measure_estimate += measure.measure;
      weighted_centroid = add(weighted_centroid, scale(measure.centroid, measure.measure));
    }
  }

  if (region.measure_estimate > real_t{0.0}) {
    region.centroid_estimate =
        scale(weighted_centroid, real_t{1.0} / region.measure_estimate);
    region.volume_fraction_estimate = region.parent_measure > real_t{0.0}
                                          ? region.measure_estimate / region.parent_measure
                                          : real_t{0.0};
  }
  region.integration_vertices = closed.vertices;
  region.integration_subcells = closed.subcells;
  region.integration_region_vertices = closed.vertex_ids;
  region.integration_region_faces = closed.faces;
  region.closed_integration_topology =
      !region.integration_subcells.empty() &&
      std::all_of(region.integration_subcells.begin(),
                  region.integration_subcells.end(),
                  [](const auto& subcell) {
                    return subcell.closed_topology;
                  });
  return region;
}

bool supports_true_curved_hex_arrangement(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded) noexcept {
  return mesh.cell_shape(cell).family == CellFamily::Hex &&
         mesh.geometry_order(cell) > 1 &&
         embedded.kind == EmbeddedGeometryKind::Plane;
}

bool add_true_curved_hex_arrangement(
    CutTopologyRecord& topology,
    const MeshBase& mesh,
    const CutEntityRecord& cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    real_t tol,
    std::uint64_t& h) {
  const int order = std::max(1, mesh.geometry_order(cell.entity));
  const int base_segments = std::max(16, order * 8);
  const auto base_quads = make_uniform_hex_base_quads(base_segments);
  if (!true_curved_hex_columns_are_graph_compatible(
          mesh, cell.entity, embedded, cfg, tol)) {
    topology.diagnostics.push_back(
        "true curved hex arrangement requires a graph-compatible plane cut with at most one root per reference-column sample");
    return false;
  }
  const auto candidates =
      true_curved_hex_edge_roots(mesh, cell.entity, embedded, cfg, tol);
  if (candidates.size() < 3u) {
    topology.diagnostics.push_back(
        "true curved hex arrangement found fewer than three boundary roots in a cut high-order hex cell");
    return false;
  }

  const auto first_patch = topology.curved_patches.size();
  const bool recorded =
      record_curved_patch_from_candidates(topology,
                                          mesh,
                                          cell.entity,
                                          embedded,
                                          cfg,
                                          cell.provenance,
                                          cell.global_id,
                                          candidates,
                                          tol,
                                          3200u,
                                          true,
                                          kTrueCurvedArrangementPolicy);
  if (!recorded || topology.curved_patches.size() <= first_patch) {
    topology.diagnostics.push_back(
        "true curved hex arrangement failed to record interface surface topology");
    return false;
  }
  auto& patch = topology.curved_patches.back();
  if (!populate_true_curved_hex_patch_quadrature(
          patch, mesh, cell.entity, embedded, cfg, base_quads, tol)) {
    topology.diagnostics.push_back(
        "true curved hex arrangement failed to build interface surface quadrature");
    return false;
  }
  const auto interface_vertex_ids = patch.ordered_vertices;
  hash_curved_patch_record(h, patch);

  for (const auto side : {CutTopologySide::Negative, CutTopologySide::Positive}) {
    auto region = make_true_curved_hex_side_region(mesh,
                                                   cell,
                                                   embedded,
                                                   cfg,
                                                   side,
                                                   base_quads,
                                                   interface_vertex_ids,
                                                   tol);
    if (!region.closed_integration_topology || region.integration_subcells.empty()) {
      topology.diagnostics.push_back(
          "true curved hex arrangement failed to construct closed side-region topology");
      return false;
    }
    topology.side_regions.push_back(std::move(region));
    hash_side_region_record(h, topology.side_regions.back());
  }

  return true;
}

std::array<real_t, 3> wedge_parametric_point(real_t r, real_t s, real_t t) noexcept {
  return {{s, t, r}};
}

std::array<real_t, 3> evaluate_wedge_geometry_point(
    const MeshBase& mesh,
    index_t cell,
    real_t r,
    real_t s,
    real_t t,
    Configuration cfg) {
  return CurvilinearEvaluator::evaluate_geometry(
      mesh, cell, wedge_parametric_point(r, s, t), cfg).coordinates;
}

real_t signed_distance_on_wedge(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    real_t r,
    real_t s,
    real_t t,
    Configuration cfg) {
  return embedded.signed_distance(evaluate_wedge_geometry_point(mesh, cell, r, s, t, cfg));
}

bool wedge_base_point_valid(real_t s, real_t t, real_t tol) noexcept {
  const real_t eps = positive_tolerance(tol);
  return s >= -eps && t >= -eps && s + t <= real_t{1.0} + eps;
}

std::size_t count_true_curved_wedge_column_roots(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    real_t s,
    real_t t,
    Configuration cfg,
    real_t tol,
    bool& valid) {
  valid = wedge_base_point_valid(s, t, tol);
  if (!valid) {
    return 0u;
  }
  const int order = std::max(1, mesh.geometry_order(cell));
  const int segments = std::max(32, order * 16);
  const real_t eps = positive_tolerance(tol) * real_t{10.0};
  const real_t merge_tol = std::max(real_t{1.0e-12}, eps * real_t{10.0});
  std::vector<real_t> roots;
  roots.reserve(4u);

  const auto add_root = [&](real_t r) {
    for (const auto existing : roots) {
      if (std::abs(existing - r) <= merge_tol) {
        return;
      }
    }
    roots.push_back(std::max(real_t{-1.0}, std::min(real_t{1.0}, r)));
  };

  real_t r_prev = real_t{-1.0};
  real_t f_prev = signed_distance_on_wedge(mesh, cell, embedded, r_prev, s, t, cfg);
  if (!std::isfinite(f_prev)) {
    valid = false;
    return 0u;
  }
  if (std::abs(f_prev) <= eps) {
    add_root(r_prev);
  }

  for (int i = 1; i <= segments; ++i) {
    const real_t r = real_t{-1.0} + real_t{2.0} * static_cast<real_t>(i) /
                                       static_cast<real_t>(segments);
    const real_t f = signed_distance_on_wedge(mesh, cell, embedded, r, s, t, cfg);
    if (!std::isfinite(f)) {
      valid = false;
      return roots.size();
    }
    if (std::abs(f) <= eps) {
      add_root(r);
    }
    if ((f_prev < -eps && f > eps) || (f_prev > eps && f < -eps)) {
      real_t a = r_prev;
      real_t b = r;
      real_t fa = f_prev;
      for (int iter = 0; iter < 96; ++iter) {
        const real_t m = real_t{0.5} * (a + b);
        const real_t fm = signed_distance_on_wedge(mesh, cell, embedded, m, s, t, cfg);
        if (!std::isfinite(fm)) {
          valid = false;
          return roots.size();
        }
        if (std::abs(fm) <= eps || std::abs(b - a) <= real_t{1.0e-14}) {
          a = m;
          b = m;
          break;
        }
        if ((fa < real_t{0.0} && fm > real_t{0.0}) ||
            (fa > real_t{0.0} && fm < real_t{0.0})) {
          b = m;
        } else {
          a = m;
          fa = fm;
        }
      }
      add_root(real_t{0.5} * (a + b));
    }
    r_prev = r;
    f_prev = f;
  }

  return roots.size();
}

bool true_curved_wedge_columns_are_graph_compatible(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    real_t tol) {
  const int order = std::max(1, mesh.geometry_order(cell));
  const int samples = std::max(8, order * 8);
  for (int i = 0; i <= samples; ++i) {
    const real_t s = static_cast<real_t>(i) / static_cast<real_t>(samples);
    for (int j = 0; j <= samples - i; ++j) {
      const real_t t = static_cast<real_t>(j) / static_cast<real_t>(samples);
      bool valid = true;
      const auto roots =
          count_true_curved_wedge_column_roots(mesh, cell, embedded, s, t, cfg, tol, valid);
      if (!valid || roots > 1u) {
        return false;
      }
    }
  }
  return true;
}

TetGraphRoot true_curved_wedge_root_at_base(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    real_t s,
    real_t t,
    Configuration cfg,
    real_t tol) {
  TetGraphRoot out;
  if (!wedge_base_point_valid(s, t, tol)) {
    return out;
  }
  const real_t eps = positive_tolerance(tol) * real_t{10.0};
  out.f_min = signed_distance_on_wedge(mesh, cell, embedded, real_t{-1.0}, s, t, cfg);
  out.f_max = signed_distance_on_wedge(mesh, cell, embedded, real_t{1.0}, s, t, cfg);
  if (std::abs(out.f_min) <= eps) {
    out.has_root = true;
    out.r = real_t{-1.0};
    return out;
  }
  if (std::abs(out.f_max) <= eps) {
    out.has_root = true;
    out.r = real_t{1.0};
    return out;
  }
  if (!((out.f_min < -eps && out.f_max > eps) ||
        (out.f_min > eps && out.f_max < -eps))) {
    return out;
  }

  real_t a = real_t{-1.0};
  real_t b = real_t{1.0};
  real_t fa = out.f_min;
  for (int iter = 0; iter < 96; ++iter) {
    const real_t m = real_t{0.5} * (a + b);
    const real_t fm = signed_distance_on_wedge(mesh, cell, embedded, m, s, t, cfg);
    if (std::abs(fm) <= eps || std::abs(b - a) <= real_t{1.0e-14}) {
      a = m;
      b = m;
      break;
    }
    if ((fa < real_t{0.0} && fm > real_t{0.0}) ||
        (fa > real_t{0.0} && fm < real_t{0.0})) {
      b = m;
    } else {
      a = m;
      fa = fm;
    }
  }
  out.has_root = true;
  out.r = real_t{0.5} * (a + b);
  return out;
}

real_t wedge_side_availability_value(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    const TetBasePoint& base,
    Configuration cfg,
    CutTopologySide side,
    real_t tol) {
  if (!wedge_base_point_valid(base.s, base.t, tol)) {
    return -positive_tolerance(tol);
  }
  const real_t f_min =
      signed_distance_on_wedge(mesh, cell, embedded, real_t{-1.0}, base.s, base.t, cfg);
  const real_t f_max =
      signed_distance_on_wedge(mesh, cell, embedded, real_t{1.0}, base.s, base.t, cfg);
  return side == CutTopologySide::Negative
             ? -std::min(f_min, f_max)
             : std::max(f_min, f_max);
}

std::vector<TetBasePoint> clipped_wedge_base_polygon_for_side(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    const TetBaseTriangle& tri,
    Configuration cfg,
    CutTopologySide side,
    real_t tol) {
  std::vector<TetBasePoint> polygon;
  polygon.reserve(3u);
  for (auto vertex : tri.vertices) {
    vertex.value = wedge_side_availability_value(
        mesh, cell, embedded, vertex, cfg, side, tol);
    polygon.push_back(vertex);
  }
  return clip_base_polygon_by_availability(std::move(polygon), tol);
}

TetColumnInterval wedge_side_interval_at_base_point(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    const TetBasePoint& base,
    Configuration cfg,
    CutTopologySide side,
    real_t tol,
    bool include_degenerate_boundary) {
  TetColumnInterval interval;
  if (!wedge_base_point_valid(base.s, base.t, tol)) {
    return interval;
  }
  const real_t eps = positive_tolerance(tol) * real_t{10.0};
  const auto root =
      true_curved_wedge_root_at_base(mesh, cell, embedded, base.s, base.t, cfg, tol);
  const bool want_negative = side == CutTopologySide::Negative;
  if (root.has_root) {
    const real_t left_len = root.r - real_t{-1.0};
    const real_t right_len = real_t{1.0} - root.r;
    bool left_matches = false;
    bool right_matches = false;
    if (left_len > eps) {
      const real_t probe = real_t{0.5} * (real_t{-1.0} + root.r);
      left_matches =
          (signed_distance_on_wedge(mesh, cell, embedded, probe, base.s, base.t, cfg) <= eps) ==
          want_negative;
    }
    if (right_len > eps) {
      const real_t probe = real_t{0.5} * (root.r + real_t{1.0});
      right_matches =
          (signed_distance_on_wedge(mesh, cell, embedded, probe, base.s, base.t, cfg) <= eps) ==
          want_negative;
    }
    if (left_matches || (include_degenerate_boundary && left_len <= eps && !right_matches)) {
      interval.available = true;
      interval.r_lo = real_t{-1.0};
      interval.r_hi = root.r;
      return interval;
    }
    if (right_matches || (include_degenerate_boundary && right_len <= eps && !left_matches)) {
      interval.available = true;
      interval.r_lo = root.r;
      interval.r_hi = real_t{1.0};
      return interval;
    }
    if (include_degenerate_boundary) {
      interval.available = true;
      interval.r_lo = root.r;
      interval.r_hi = root.r;
    }
    return interval;
  }

  const bool whole_interval_matches =
      (signed_distance_on_wedge(mesh, cell, embedded, real_t{0.0}, base.s, base.t, cfg) <= eps) ==
      want_negative;
  if (whole_interval_matches) {
    interval.available = true;
    interval.r_lo = real_t{-1.0};
    interval.r_hi = real_t{1.0};
  }
  return interval;
}

FaceGraphIntegral integrate_wedge_graph_region(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    const TetBaseTriangle& base_tri,
    Configuration cfg,
    real_t tol,
    bool restrict_to_side,
    CutTopologySide side) {
  FaceGraphIntegral out;
  const real_t base_jac = tet_base_triangle_jacobian(base_tri);
  if (base_jac <= positive_tolerance(tol)) {
    return out;
  }
  const int order = std::max(1, mesh.geometry_order(cell));
  const auto u_rule = gauss_legendre_rule(std::min(12, std::max(4, order + 3)));
  const auto v_rule = gauss_legendre_rule(std::min(12, std::max(4, order + 3)));
  const auto r_rule = gauss_legendre_rule(std::min(12, std::max(4, order + 3)));
  out.available = true;

  const auto& a = base_tri.vertices[0];
  const auto& b = base_tri.vertices[1];
  const auto& c = base_tri.vertices[2];
  for (std::size_t iu = 0; iu < u_rule.points.size(); ++iu) {
    const real_t u = real_t{0.5} * (u_rule.points[iu] + real_t{1.0});
    const real_t wu = real_t{0.5} * u_rule.weights[iu];
    for (std::size_t iv = 0; iv < v_rule.points.size(); ++iv) {
      const real_t v = real_t{0.5} * (v_rule.points[iv] + real_t{1.0});
      const real_t wv = real_t{0.5} * v_rule.weights[iv];
      const real_t l0 = (real_t{1.0} - u) * (real_t{1.0} - v);
      const real_t l1 = u;
      const real_t l2 = (real_t{1.0} - u) * v;
      TetBasePoint base;
      base.s = l0 * a.s + l1 * b.s + l2 * c.s;
      base.t = l0 * a.t + l1 * b.t + l2 * c.t;
      const real_t base_weight = wu * wv * base_jac * (real_t{1.0} - u);

      real_t r_lo = real_t{-1.0};
      real_t r_hi = real_t{1.0};
      if (restrict_to_side) {
        const auto interval = wedge_side_interval_at_base_point(
            mesh, cell, embedded, base, cfg, side, tol, false);
        if (!interval.available) {
          continue;
        }
        r_lo = interval.r_lo;
        r_hi = interval.r_hi;
      }
      if (r_hi - r_lo <= positive_tolerance(tol)) {
        continue;
      }

      const real_t half_r = real_t{0.5} * (r_hi - r_lo);
      const real_t center_r = real_t{0.5} * (r_lo + r_hi);
      for (std::size_t ir = 0; ir < r_rule.points.size(); ++ir) {
        const real_t r = center_r + half_r * r_rule.points[ir];
        const auto eval = CurvilinearEvaluator::evaluate_geometry(
            mesh, cell, wedge_parametric_point(r, base.s, base.t), cfg);
        const auto dx_dr = jacobian_action(eval.jacobian, {{real_t{0.0}, real_t{0.0}, real_t{1.0}}});
        const auto dx_ds = jacobian_action(eval.jacobian, {{real_t{1.0}, real_t{0.0}, real_t{0.0}}});
        const auto dx_dt = jacobian_action(eval.jacobian, {{real_t{0.0}, real_t{1.0}, real_t{0.0}}});
        const real_t jac = std::abs(determinant3(dx_ds, dx_dt, dx_dr));
        const real_t weight = base_weight * r_rule.weights[ir] * half_r * jac;
        if (!std::isfinite(weight)) {
          return {};
        }
        out.measure += weight;
        out.centroid = add(out.centroid, scale(eval.coordinates, weight));
      }
    }
  }
  if (out.measure > real_t{0.0}) {
    out.centroid = scale(out.centroid, real_t{1.0} / out.measure);
  }
  return out;
}

std::vector<CurvedPatchPoint> true_curved_wedge_edge_roots(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    real_t tol) {
  constexpr std::array<std::array<real_t, 3>, 6> vertices{{
      {{0.0, 0.0, -1.0}},
      {{1.0, 0.0, -1.0}},
      {{0.0, 1.0, -1.0}},
      {{0.0, 0.0,  1.0}},
      {{1.0, 0.0,  1.0}},
      {{0.0, 1.0,  1.0}}}};
  const auto eview = CellTopology::get_edges_view(CellFamily::Wedge);
  const real_t eps = positive_tolerance(tol) * real_t{10.0};
  std::vector<CurvedPatchPoint> out;
  const auto add_root = [&](std::array<real_t, 3> xi) {
    for (const auto& existing : out) {
      if (norm(sub(existing.parent_parametric_coordinate, xi)) <= eps * real_t{100.0}) {
        return;
      }
    }
    CurvedPatchPoint point;
    point.parent_parametric_coordinate = xi;
    point.has_parent_parametric_coordinate = true;
    point.point =
        CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, cfg).coordinates;
    out.push_back(point);
  };

  for (int ei = 0; ei < eview.edge_count; ++ei) {
    const auto a = vertices[static_cast<std::size_t>(eview.pairs_flat[2 * ei + 0])];
    const auto b = vertices[static_cast<std::size_t>(eview.pairs_flat[2 * ei + 1])];
    real_t fa =
        embedded.signed_distance(CurvilinearEvaluator::evaluate_geometry(mesh, cell, a, cfg).coordinates);
    const real_t fb0 =
        embedded.signed_distance(CurvilinearEvaluator::evaluate_geometry(mesh, cell, b, cfg).coordinates);
    if (std::abs(fa) <= eps) {
      add_root(a);
    }
    if (std::abs(fb0) <= eps) {
      add_root(b);
    }
    if (!((fa < -eps && fb0 > eps) || (fa > eps && fb0 < -eps))) {
      continue;
    }
    real_t lo = real_t{0.0};
    real_t hi = real_t{1.0};
    for (int iter = 0; iter < 96; ++iter) {
      const real_t mid = real_t{0.5} * (lo + hi);
      const auto xi = interpolate_parametric(a, b, mid);
      const real_t fm =
          embedded.signed_distance(CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, cfg).coordinates);
      if (std::abs(fm) <= eps || std::abs(hi - lo) <= real_t{1.0e-14}) {
        lo = mid;
        hi = mid;
        break;
      }
      if ((fa < real_t{0.0} && fm > real_t{0.0}) ||
          (fa > real_t{0.0} && fm < real_t{0.0})) {
        hi = mid;
      } else {
        lo = mid;
        fa = fm;
      }
    }
    add_root(interpolate_parametric(a, b, real_t{0.5} * (lo + hi)));
  }
  return out;
}

bool populate_true_curved_wedge_patch_quadrature(
    CutCurvedPatchRecord& patch,
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    const std::vector<TetBaseTriangle>& base_triangles,
    real_t tol) {
  patch.quadrature_points.clear();
  patch.quadrature_normals.clear();
  patch.quadrature_weights.clear();
  patch.quadrature_measure = real_t{0.0};
  patch.isoparametric_quadrature_available = false;

  const int order = std::max(1, mesh.geometry_order(cell));
  const auto u_rule = gauss_legendre_rule(std::min(12, std::max(4, order + 3)));
  const auto v_rule = gauss_legendre_rule(std::min(12, std::max(4, order + 3)));
  for (const auto& tri : base_triangles) {
    const real_t base_jac = tet_base_triangle_jacobian(tri);
    if (base_jac <= positive_tolerance(tol)) {
      continue;
    }
    const auto& a = tri.vertices[0];
    const auto& b = tri.vertices[1];
    const auto& c = tri.vertices[2];
    for (std::size_t iu = 0; iu < u_rule.points.size(); ++iu) {
      const real_t u = real_t{0.5} * (u_rule.points[iu] + real_t{1.0});
      const real_t wu = real_t{0.5} * u_rule.weights[iu];
      for (std::size_t iv = 0; iv < v_rule.points.size(); ++iv) {
        const real_t v = real_t{0.5} * (v_rule.points[iv] + real_t{1.0});
        const real_t wv = real_t{0.5} * v_rule.weights[iv];
        const real_t l0 = (real_t{1.0} - u) * (real_t{1.0} - v);
        const real_t l1 = u;
        const real_t l2 = (real_t{1.0} - u) * v;
        const real_t s = l0 * a.s + l1 * b.s + l2 * c.s;
        const real_t t = l0 * a.t + l1 * b.t + l2 * c.t;
        const auto root = true_curved_wedge_root_at_base(
            mesh, cell, embedded, s, t, cfg, tol);
        if (!root.has_root) {
          continue;
        }
        const auto xi = wedge_parametric_point(root.r, s, t);
        const auto eval = CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, cfg);
        const auto dx_dr = jacobian_action(eval.jacobian, {{real_t{0.0}, real_t{0.0}, real_t{1.0}}});
        const auto dx_ds = jacobian_action(eval.jacobian, {{real_t{1.0}, real_t{0.0}, real_t{0.0}}});
        const auto dx_dt = jacobian_action(eval.jacobian, {{real_t{0.0}, real_t{1.0}, real_t{0.0}}});
        const real_t phi_r = dot(embedded.normal, dx_dr);
        const real_t phi_s = dot(embedded.normal, dx_ds);
        const real_t phi_t = dot(embedded.normal, dx_dt);
        if (std::abs(phi_r) <= positive_tolerance(tol)) {
          return false;
        }
        const real_t drds = -phi_s / phi_r;
        const real_t drdt = -phi_t / phi_r;
        const auto tangent_s = add(dx_ds, scale(dx_dr, drds));
        const auto tangent_t = add(dx_dt, scale(dx_dr, drdt));
        const real_t base_weight = wu * wv * base_jac * (real_t{1.0} - u);
        const real_t weight = base_weight * norm(cross(tangent_s, tangent_t));
        if (!std::isfinite(weight) || weight <= real_t{0.0}) {
          continue;
        }
        patch.quadrature_points.push_back(eval.coordinates);
        patch.quadrature_normals.push_back(unit_or_default(embedded.outward_normal(eval.coordinates)));
        patch.quadrature_weights.push_back(weight);
        patch.quadrature_measure += weight;
      }
    }
  }

  patch.isoparametric_quadrature_available =
      patch.quadrature_measure > positive_tolerance(tol);
  patch.construction_policy = kTrueCurvedArrangementPolicy;
  return patch.isoparametric_quadrature_available;
}

bool add_true_curved_wedge_graph_subcell(
    SideClosedTopology& closed,
    const MeshBase& mesh,
    const CutEntityRecord& cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    CutTopologySide side,
    const std::vector<index_t>& dofs,
    const std::vector<std::array<real_t, 3>>& parent_points,
    const std::vector<TetBasePoint>& base_polygon,
    const FaceGraphIntegral& measure,
    real_t tol,
    std::uint64_t salt) {
  if (!measure.available || measure.measure <= positive_tolerance(tol) ||
      base_polygon.size() < 3u) {
    return false;
  }

  std::vector<std::array<real_t, 3>> xi_vertices;
  xi_vertices.reserve(base_polygon.size() * 2u);
  for (const auto& base : base_polygon) {
    const auto interval = wedge_side_interval_at_base_point(
        mesh, cell.entity, embedded, base, cfg, side, tol, true);
    if (!interval.available) {
      continue;
    }
    xi_vertices.push_back(wedge_parametric_point(interval.r_lo, base.s, base.t));
    if (std::abs(interval.r_hi - interval.r_lo) > positive_tolerance(tol) * real_t{10.0}) {
      xi_vertices.push_back(wedge_parametric_point(interval.r_hi, base.s, base.t));
    }
  }
  xi_vertices = unique_xi_vertices(xi_vertices, tol);
  auto faces = convex_hull_faces_indices(xi_vertices, tol);
  if (xi_vertices.size() < 4u || faces.size() < 4u) {
    return false;
  }

  std::vector<std::array<real_t, 3>> physical_points;
  physical_points.reserve(xi_vertices.size());
  for (const auto& xi : xi_vertices) {
    physical_points.push_back(
        CurvilinearEvaluator::evaluate_geometry(mesh, cell.entity, xi, cfg).coordinates);
  }
  const auto before = closed.subcells.size();
  add_integration_subcell(closed,
                          xi_vertices.size() == 6u ? CellFamily::Wedge : CellFamily::Polyhedron,
                          physical_points,
                          faces,
                          measure.measure,
                          measure.centroid,
                          dofs,
                          parent_points,
                          embedded,
                          cell.provenance,
                          cell.global_id,
                          side,
                          tol,
                          salt,
                          xi_vertices,
                          true,
                          true,
                          kTrueCurvedArrangementPolicy);
  return closed.subcells.size() > before;
}

CutSideRegion make_true_curved_wedge_side_region(
    const MeshBase& mesh,
    const CutEntityRecord& cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    CutTopologySide side,
    const std::vector<TetBaseTriangle>& base_triangles,
    const std::vector<std::uint64_t>& interface_vertex_ids,
    real_t tol) {
  CutSideRegion region;
  region.side = side;
  region.parent_cell = cell.entity;
  region.parent_cell_gid = cell.global_id;
  region.provenance = cell.provenance;
  region.integration_family = CellFamily::Wedge;
  region.curved_isoparametric_topology = true;
  region.measure_from_linear_topology = true;
  region.stable_id = stable_entity_id(cell.provenance,
                                      cell.global_id,
                                      0u,
                                      static_cast<index_t>(side),
                                      real_t{0.0},
                                      404u);
  region.cut_vertices = interface_vertex_ids;

  const auto dofs = mesh.cell_geometry_dofs(cell.entity);
  std::vector<std::array<real_t, 3>> parent_points;
  parent_points.reserve(dofs.size());
  std::vector<real_t> distances;
  distances.reserve(dofs.size());
  for (const auto dof : dofs) {
    const auto point = mesh.geometry_dof_coords(dof, cfg);
    parent_points.push_back(point);
    distances.push_back(embedded.signed_distance(point));
  }
  for (std::size_t i = 0; i < dofs.size(); ++i) {
    const bool on_side = side == CutTopologySide::Negative
                             ? distances[i] <= positive_tolerance(tol)
                             : distances[i] >= -positive_tolerance(tol);
    if (on_side) {
      region.parent_geometry_dofs.push_back(dofs[i]);
    }
  }

  SideClosedTopology closed;
  std::array<real_t, 3> weighted_centroid{{0.0, 0.0, 0.0}};
  std::uint64_t subcell_salt = 0u;
  for (const auto& base_tri : base_triangles) {
    const auto full_measure =
        integrate_wedge_graph_region(mesh,
                                     cell.entity,
                                     embedded,
                                     base_tri,
                                     cfg,
                                     tol,
                                     false,
                                     side);
    if (full_measure.available) {
      region.parent_measure += full_measure.measure;
    }

    const auto measure =
        integrate_wedge_graph_region(mesh,
                                     cell.entity,
                                     embedded,
                                     base_tri,
                                     cfg,
                                     tol,
                                     true,
                                     side);
    if (!measure.available || measure.measure <= positive_tolerance(tol)) {
      continue;
    }
    const auto base_polygon =
        clipped_wedge_base_polygon_for_side(mesh, cell.entity, embedded, base_tri, cfg, side, tol);
    if (add_true_curved_wedge_graph_subcell(closed,
                                            mesh,
                                            cell,
                                            embedded,
                                            cfg,
                                            side,
                                            dofs,
                                            parent_points,
                                            base_polygon,
                                            measure,
                                            tol,
                                            3400u + subcell_salt++)) {
      region.measure_estimate += measure.measure;
      weighted_centroid = add(weighted_centroid, scale(measure.centroid, measure.measure));
    }
  }

  if (region.measure_estimate > real_t{0.0}) {
    region.centroid_estimate =
        scale(weighted_centroid, real_t{1.0} / region.measure_estimate);
    region.volume_fraction_estimate = region.parent_measure > real_t{0.0}
                                          ? region.measure_estimate / region.parent_measure
                                          : real_t{0.0};
  }
  region.integration_vertices = closed.vertices;
  region.integration_subcells = closed.subcells;
  region.integration_region_vertices = closed.vertex_ids;
  region.integration_region_faces = closed.faces;
  region.closed_integration_topology =
      !region.integration_subcells.empty() &&
      std::all_of(region.integration_subcells.begin(),
                  region.integration_subcells.end(),
                  [](const auto& subcell) {
                    return subcell.closed_topology;
                  });
  return region;
}

bool supports_true_curved_wedge_arrangement(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded) noexcept {
  return mesh.cell_shape(cell).family == CellFamily::Wedge &&
         mesh.geometry_order(cell) > 1 &&
         embedded.kind == EmbeddedGeometryKind::Plane;
}

bool add_true_curved_wedge_arrangement(
    CutTopologyRecord& topology,
    const MeshBase& mesh,
    const CutEntityRecord& cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    real_t tol,
    std::uint64_t& h) {
  const int order = std::max(1, mesh.geometry_order(cell.entity));
  const int base_segments = std::max(24, order * 8);
  const auto base_triangles = make_uniform_tet_base_triangles(base_segments);
  if (!true_curved_wedge_columns_are_graph_compatible(
          mesh, cell.entity, embedded, cfg, tol)) {
    topology.diagnostics.push_back(
        "true curved wedge arrangement requires a graph-compatible plane cut with at most one root per reference-column sample");
    return false;
  }
  const auto candidates =
      true_curved_wedge_edge_roots(mesh, cell.entity, embedded, cfg, tol);
  if (candidates.size() < 3u) {
    topology.diagnostics.push_back(
        "true curved wedge arrangement found fewer than three boundary roots in a cut high-order wedge cell");
    return false;
  }

  const auto first_patch = topology.curved_patches.size();
  const bool recorded =
      record_curved_patch_from_candidates(topology,
                                          mesh,
                                          cell.entity,
                                          embedded,
                                          cfg,
                                          cell.provenance,
                                          cell.global_id,
                                          candidates,
                                          tol,
                                          3600u,
                                          true,
                                          kTrueCurvedArrangementPolicy);
  if (!recorded || topology.curved_patches.size() <= first_patch) {
    topology.diagnostics.push_back(
        "true curved wedge arrangement failed to record interface surface topology");
    return false;
  }
  auto& patch = topology.curved_patches.back();
  if (!populate_true_curved_wedge_patch_quadrature(
          patch, mesh, cell.entity, embedded, cfg, base_triangles, tol)) {
    topology.diagnostics.push_back(
        "true curved wedge arrangement failed to build interface surface quadrature");
    return false;
  }
  const auto interface_vertex_ids = patch.ordered_vertices;
  hash_curved_patch_record(h, patch);

  for (const auto side : {CutTopologySide::Negative, CutTopologySide::Positive}) {
    auto region = make_true_curved_wedge_side_region(mesh,
                                                     cell,
                                                     embedded,
                                                     cfg,
                                                     side,
                                                     base_triangles,
                                                     interface_vertex_ids,
                                                     tol);
    if (!region.closed_integration_topology || region.integration_subcells.empty()) {
      topology.diagnostics.push_back(
          "true curved wedge arrangement failed to construct closed side-region topology");
      return false;
    }
    topology.side_regions.push_back(std::move(region));
    hash_side_region_record(h, topology.side_regions.back());
  }

  return true;
}

TetReferenceSpan pyramid_reference_span(real_t s, real_t t, real_t tol) noexcept {
  const real_t eps = positive_tolerance(tol);
  const real_t rmax = real_t{1.0} - std::max(std::abs(s), std::abs(t));
  const bool valid = s >= real_t{-1.0} - eps &&
                     s <= real_t{1.0} + eps &&
                     t >= real_t{-1.0} - eps &&
                     t <= real_t{1.0} + eps &&
                     rmax >= -eps;
  return {real_t{0.0}, std::max(real_t{0.0}, rmax), valid};
}

std::array<real_t, 3> pyramid_parametric_point(real_t r, real_t s, real_t t) noexcept {
  return {{s, t, r}};
}

std::array<real_t, 3> evaluate_pyramid_geometry_point(
    const MeshBase& mesh,
    index_t cell,
    real_t r,
    real_t s,
    real_t t,
    Configuration cfg) {
  return CurvilinearEvaluator::evaluate_geometry(
      mesh, cell, pyramid_parametric_point(r, s, t), cfg).coordinates;
}

real_t signed_distance_on_pyramid(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    real_t r,
    real_t s,
    real_t t,
    Configuration cfg) {
  return embedded.signed_distance(evaluate_pyramid_geometry_point(mesh, cell, r, s, t, cfg));
}

std::size_t count_true_curved_pyramid_column_roots(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    real_t s,
    real_t t,
    Configuration cfg,
    real_t tol,
    bool& valid) {
  const auto span = pyramid_reference_span(s, t, tol);
  valid = span.valid;
  if (!valid || span.r_max - span.r_min <= positive_tolerance(tol)) {
    return 0u;
  }
  const int order = std::max(1, mesh.geometry_order(cell));
  const int segments = std::max(32, order * 16);
  const real_t eps = positive_tolerance(tol) * real_t{10.0};
  const real_t merge_tol = std::max(real_t{1.0e-12}, eps * real_t{10.0});
  std::vector<real_t> roots;
  roots.reserve(4u);

  const auto add_root = [&](real_t r) {
    for (const auto existing : roots) {
      if (std::abs(existing - r) <= merge_tol) {
        return;
      }
    }
    roots.push_back(std::max(span.r_min, std::min(span.r_max, r)));
  };

  real_t r_prev = span.r_min;
  real_t f_prev = signed_distance_on_pyramid(mesh, cell, embedded, r_prev, s, t, cfg);
  if (!std::isfinite(f_prev)) {
    valid = false;
    return 0u;
  }
  if (std::abs(f_prev) <= eps) {
    add_root(r_prev);
  }

  for (int i = 1; i <= segments; ++i) {
    const real_t alpha = static_cast<real_t>(i) / static_cast<real_t>(segments);
    const real_t r = span.r_min + alpha * (span.r_max - span.r_min);
    const real_t f = signed_distance_on_pyramid(mesh, cell, embedded, r, s, t, cfg);
    if (!std::isfinite(f)) {
      valid = false;
      return roots.size();
    }
    if (std::abs(f) <= eps) {
      add_root(r);
    }
    if ((f_prev < -eps && f > eps) || (f_prev > eps && f < -eps)) {
      real_t a = r_prev;
      real_t b = r;
      real_t fa = f_prev;
      for (int iter = 0; iter < 96; ++iter) {
        const real_t m = real_t{0.5} * (a + b);
        const real_t fm = signed_distance_on_pyramid(mesh, cell, embedded, m, s, t, cfg);
        if (!std::isfinite(fm)) {
          valid = false;
          return roots.size();
        }
        if (std::abs(fm) <= eps || std::abs(b - a) <= real_t{1.0e-14}) {
          a = m;
          b = m;
          break;
        }
        if ((fa < real_t{0.0} && fm > real_t{0.0}) ||
            (fa > real_t{0.0} && fm < real_t{0.0})) {
          b = m;
        } else {
          a = m;
          fa = fm;
        }
      }
      add_root(real_t{0.5} * (a + b));
    }
    r_prev = r;
    f_prev = f;
  }

  return roots.size();
}

bool true_curved_pyramid_columns_are_graph_compatible(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    real_t tol) {
  const int order = std::max(1, mesh.geometry_order(cell));
  const int samples = std::max(8, order * 8);
  for (int i = 0; i <= samples; ++i) {
    const real_t s = real_t{-1.0} + real_t{2.0} * static_cast<real_t>(i) /
                                       static_cast<real_t>(samples);
    for (int j = 0; j <= samples; ++j) {
      const real_t t = real_t{-1.0} + real_t{2.0} * static_cast<real_t>(j) /
                                         static_cast<real_t>(samples);
      bool valid = true;
      const auto roots =
          count_true_curved_pyramid_column_roots(mesh, cell, embedded, s, t, cfg, tol, valid);
      if (!valid || roots > 1u) {
        return false;
      }
    }
  }
  return true;
}

TetGraphRoot true_curved_pyramid_root_at_base(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    real_t s,
    real_t t,
    Configuration cfg,
    real_t tol) {
  TetGraphRoot out;
  const auto span = pyramid_reference_span(s, t, tol);
  if (!span.valid || span.r_max - span.r_min <= positive_tolerance(tol)) {
    return out;
  }
  const real_t eps = positive_tolerance(tol) * real_t{10.0};
  out.f_min = signed_distance_on_pyramid(mesh, cell, embedded, span.r_min, s, t, cfg);
  out.f_max = signed_distance_on_pyramid(mesh, cell, embedded, span.r_max, s, t, cfg);
  if (std::abs(out.f_min) <= eps) {
    out.has_root = true;
    out.r = span.r_min;
    return out;
  }
  if (std::abs(out.f_max) <= eps) {
    out.has_root = true;
    out.r = span.r_max;
    return out;
  }
  if (!((out.f_min < -eps && out.f_max > eps) ||
        (out.f_min > eps && out.f_max < -eps))) {
    return out;
  }

  real_t a = span.r_min;
  real_t b = span.r_max;
  real_t fa = out.f_min;
  for (int iter = 0; iter < 96; ++iter) {
    const real_t m = real_t{0.5} * (a + b);
    const real_t fm = signed_distance_on_pyramid(mesh, cell, embedded, m, s, t, cfg);
    if (std::abs(fm) <= eps || std::abs(b - a) <= real_t{1.0e-14}) {
      a = m;
      b = m;
      break;
    }
    if ((fa < real_t{0.0} && fm > real_t{0.0}) ||
        (fa > real_t{0.0} && fm < real_t{0.0})) {
      b = m;
    } else {
      a = m;
      fa = fm;
    }
  }
  out.has_root = true;
  out.r = real_t{0.5} * (a + b);
  return out;
}

real_t pyramid_side_availability_value(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    const TetBasePoint& base,
    Configuration cfg,
    CutTopologySide side,
    real_t tol) {
  const auto span = pyramid_reference_span(base.s, base.t, tol);
  if (!span.valid) {
    return -positive_tolerance(tol);
  }
  const real_t f_min =
      signed_distance_on_pyramid(mesh, cell, embedded, span.r_min, base.s, base.t, cfg);
  const real_t f_max =
      signed_distance_on_pyramid(mesh, cell, embedded, span.r_max, base.s, base.t, cfg);
  return side == CutTopologySide::Negative
             ? -std::min(f_min, f_max)
             : std::max(f_min, f_max);
}

std::vector<TetBasePoint> clipped_pyramid_base_polygon_for_side(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    const HexBaseQuad& quad,
    Configuration cfg,
    CutTopologySide side,
    real_t tol) {
  std::vector<TetBasePoint> polygon;
  polygon.reserve(4u);
  for (auto vertex : quad.vertices) {
    vertex.value = pyramid_side_availability_value(
        mesh, cell, embedded, vertex, cfg, side, tol);
    polygon.push_back(vertex);
  }
  return clip_base_polygon_by_availability(std::move(polygon), tol);
}

TetColumnInterval pyramid_side_interval_at_base_point(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    const TetBasePoint& base,
    Configuration cfg,
    CutTopologySide side,
    real_t tol,
    bool include_degenerate_boundary) {
  TetColumnInterval interval;
  const auto span = pyramid_reference_span(base.s, base.t, tol);
  if (!span.valid) {
    return interval;
  }
  const real_t eps = positive_tolerance(tol) * real_t{10.0};
  const auto root =
      true_curved_pyramid_root_at_base(mesh, cell, embedded, base.s, base.t, cfg, tol);
  const bool want_negative = side == CutTopologySide::Negative;
  if (root.has_root) {
    const real_t left_len = root.r - span.r_min;
    const real_t right_len = span.r_max - root.r;
    bool left_matches = false;
    bool right_matches = false;
    if (left_len > eps) {
      const real_t probe = real_t{0.5} * (span.r_min + root.r);
      left_matches =
          (signed_distance_on_pyramid(mesh, cell, embedded, probe, base.s, base.t, cfg) <= eps) ==
          want_negative;
    }
    if (right_len > eps) {
      const real_t probe = real_t{0.5} * (root.r + span.r_max);
      right_matches =
          (signed_distance_on_pyramid(mesh, cell, embedded, probe, base.s, base.t, cfg) <= eps) ==
          want_negative;
    }
    if (left_matches || (include_degenerate_boundary && left_len <= eps && !right_matches)) {
      interval.available = true;
      interval.r_lo = span.r_min;
      interval.r_hi = root.r;
      return interval;
    }
    if (right_matches || (include_degenerate_boundary && right_len <= eps && !left_matches)) {
      interval.available = true;
      interval.r_lo = root.r;
      interval.r_hi = span.r_max;
      return interval;
    }
    if (include_degenerate_boundary) {
      interval.available = true;
      interval.r_lo = root.r;
      interval.r_hi = root.r;
    }
    return interval;
  }

  const real_t probe = real_t{0.5} * (span.r_min + span.r_max);
  const bool whole_interval_matches =
      (signed_distance_on_pyramid(mesh, cell, embedded, probe, base.s, base.t, cfg) <= eps) ==
      want_negative;
  if (whole_interval_matches) {
    interval.available = true;
    interval.r_lo = span.r_min;
    interval.r_hi = span.r_max;
  }
  return interval;
}

FaceGraphIntegral integrate_pyramid_graph_region(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    const HexBaseQuad& base_quad,
    Configuration cfg,
    real_t tol,
    bool restrict_to_side,
    CutTopologySide side) {
  FaceGraphIntegral out;
  const auto& a = base_quad.vertices[0];
  const auto& b = base_quad.vertices[1];
  const auto& d = base_quad.vertices[3];
  const real_t ds = b.s - a.s;
  const real_t dt = d.t - a.t;
  if (std::abs(ds * dt) <= positive_tolerance(tol)) {
    return out;
  }
  const int order = std::max(1, mesh.geometry_order(cell));
  const auto s_rule = gauss_legendre_rule(std::min(12, std::max(4, order + 3)));
  const auto t_rule = gauss_legendre_rule(std::min(12, std::max(4, order + 3)));
  const auto r_rule = gauss_legendre_rule(std::min(12, std::max(4, order + 3)));
  const real_t half_s = real_t{0.5} * ds;
  const real_t half_t = real_t{0.5} * dt;
  const real_t center_s = real_t{0.5} * (a.s + b.s);
  const real_t center_t = real_t{0.5} * (a.t + d.t);
  out.available = true;

  for (std::size_t is = 0; is < s_rule.points.size(); ++is) {
    const real_t s = center_s + half_s * s_rule.points[is];
    for (std::size_t it = 0; it < t_rule.points.size(); ++it) {
      const real_t t = center_t + half_t * t_rule.points[it];
      const auto span = pyramid_reference_span(s, t, tol);
      if (!span.valid || span.r_max - span.r_min <= positive_tolerance(tol)) {
        continue;
      }
      TetBasePoint base;
      base.s = s;
      base.t = t;
      real_t r_lo = span.r_min;
      real_t r_hi = span.r_max;
      if (restrict_to_side) {
        const auto interval = pyramid_side_interval_at_base_point(
            mesh, cell, embedded, base, cfg, side, tol, false);
        if (!interval.available) {
          continue;
        }
        r_lo = interval.r_lo;
        r_hi = interval.r_hi;
      }
      if (r_hi - r_lo <= positive_tolerance(tol)) {
        continue;
      }

      const real_t half_r = real_t{0.5} * (r_hi - r_lo);
      const real_t center_r = real_t{0.5} * (r_lo + r_hi);
      for (std::size_t ir = 0; ir < r_rule.points.size(); ++ir) {
        const real_t r = center_r + half_r * r_rule.points[ir];
        const auto eval = CurvilinearEvaluator::evaluate_geometry(
            mesh, cell, pyramid_parametric_point(r, s, t), cfg);
        const auto dx_dr = jacobian_action(eval.jacobian, {{real_t{0.0}, real_t{0.0}, real_t{1.0}}});
        const auto dx_ds = jacobian_action(eval.jacobian, {{real_t{1.0}, real_t{0.0}, real_t{0.0}}});
        const auto dx_dt = jacobian_action(eval.jacobian, {{real_t{0.0}, real_t{1.0}, real_t{0.0}}});
        const real_t jac = std::abs(determinant3(dx_ds, dx_dt, dx_dr));
        const real_t weight =
            s_rule.weights[is] * t_rule.weights[it] * r_rule.weights[ir] *
            std::abs(half_s * half_t) * half_r * jac;
        if (!std::isfinite(weight)) {
          return {};
        }
        out.measure += weight;
        out.centroid = add(out.centroid, scale(eval.coordinates, weight));
      }
    }
  }
  if (out.measure > real_t{0.0}) {
    out.centroid = scale(out.centroid, real_t{1.0} / out.measure);
  }
  return out;
}

std::vector<CurvedPatchPoint> true_curved_pyramid_edge_roots(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    real_t tol) {
  constexpr std::array<std::array<real_t, 3>, 5> vertices{{
      {{-1.0, -1.0, 0.0}},
      {{ 1.0, -1.0, 0.0}},
      {{ 1.0,  1.0, 0.0}},
      {{-1.0,  1.0, 0.0}},
      {{ 0.0,  0.0, 1.0}}}};
  const auto eview = CellTopology::get_edges_view(CellFamily::Pyramid);
  const real_t eps = positive_tolerance(tol) * real_t{10.0};
  std::vector<CurvedPatchPoint> out;
  const auto add_root = [&](std::array<real_t, 3> xi) {
    for (const auto& existing : out) {
      if (norm(sub(existing.parent_parametric_coordinate, xi)) <= eps * real_t{100.0}) {
        return;
      }
    }
    CurvedPatchPoint point;
    point.parent_parametric_coordinate = xi;
    point.has_parent_parametric_coordinate = true;
    point.point =
        CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, cfg).coordinates;
    out.push_back(point);
  };

  for (int ei = 0; ei < eview.edge_count; ++ei) {
    const auto a = vertices[static_cast<std::size_t>(eview.pairs_flat[2 * ei + 0])];
    const auto b = vertices[static_cast<std::size_t>(eview.pairs_flat[2 * ei + 1])];
    real_t fa =
        embedded.signed_distance(CurvilinearEvaluator::evaluate_geometry(mesh, cell, a, cfg).coordinates);
    const real_t fb0 =
        embedded.signed_distance(CurvilinearEvaluator::evaluate_geometry(mesh, cell, b, cfg).coordinates);
    if (std::abs(fa) <= eps) {
      add_root(a);
    }
    if (std::abs(fb0) <= eps) {
      add_root(b);
    }
    if (!((fa < -eps && fb0 > eps) || (fa > eps && fb0 < -eps))) {
      continue;
    }
    real_t lo = real_t{0.0};
    real_t hi = real_t{1.0};
    for (int iter = 0; iter < 96; ++iter) {
      const real_t mid = real_t{0.5} * (lo + hi);
      const auto xi = interpolate_parametric(a, b, mid);
      const real_t fm =
          embedded.signed_distance(CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, cfg).coordinates);
      if (std::abs(fm) <= eps || std::abs(hi - lo) <= real_t{1.0e-14}) {
        lo = mid;
        hi = mid;
        break;
      }
      if ((fa < real_t{0.0} && fm > real_t{0.0}) ||
          (fa > real_t{0.0} && fm < real_t{0.0})) {
        hi = mid;
      } else {
        lo = mid;
        fa = fm;
      }
    }
    add_root(interpolate_parametric(a, b, real_t{0.5} * (lo + hi)));
  }
  return out;
}

bool populate_true_curved_pyramid_patch_quadrature(
    CutCurvedPatchRecord& patch,
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    const std::vector<HexBaseQuad>& base_quads,
    real_t tol) {
  patch.quadrature_points.clear();
  patch.quadrature_normals.clear();
  patch.quadrature_weights.clear();
  patch.quadrature_measure = real_t{0.0};
  patch.isoparametric_quadrature_available = false;

  const int order = std::max(1, mesh.geometry_order(cell));
  const auto s_rule = gauss_legendre_rule(std::min(12, std::max(4, order + 3)));
  const auto t_rule = gauss_legendre_rule(std::min(12, std::max(4, order + 3)));
  for (const auto& quad : base_quads) {
    const auto& a = quad.vertices[0];
    const auto& b = quad.vertices[1];
    const auto& d = quad.vertices[3];
    const real_t half_s = real_t{0.5} * (b.s - a.s);
    const real_t half_t = real_t{0.5} * (d.t - a.t);
    if (std::abs(half_s * half_t) <= positive_tolerance(tol)) {
      continue;
    }
    const real_t center_s = real_t{0.5} * (a.s + b.s);
    const real_t center_t = real_t{0.5} * (a.t + d.t);
    for (std::size_t is = 0; is < s_rule.points.size(); ++is) {
      const real_t s = center_s + half_s * s_rule.points[is];
      for (std::size_t it = 0; it < t_rule.points.size(); ++it) {
        const real_t t = center_t + half_t * t_rule.points[it];
        const auto root = true_curved_pyramid_root_at_base(
            mesh, cell, embedded, s, t, cfg, tol);
        if (!root.has_root) {
          continue;
        }
        const auto xi = pyramid_parametric_point(root.r, s, t);
        const auto eval = CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, cfg);
        const auto dx_dr = jacobian_action(eval.jacobian, {{real_t{0.0}, real_t{0.0}, real_t{1.0}}});
        const auto dx_ds = jacobian_action(eval.jacobian, {{real_t{1.0}, real_t{0.0}, real_t{0.0}}});
        const auto dx_dt = jacobian_action(eval.jacobian, {{real_t{0.0}, real_t{1.0}, real_t{0.0}}});
        const real_t phi_r = dot(embedded.normal, dx_dr);
        const real_t phi_s = dot(embedded.normal, dx_ds);
        const real_t phi_t = dot(embedded.normal, dx_dt);
        if (std::abs(phi_r) <= positive_tolerance(tol)) {
          return false;
        }
        const real_t drds = -phi_s / phi_r;
        const real_t drdt = -phi_t / phi_r;
        const auto tangent_s = add(dx_ds, scale(dx_dr, drds));
        const auto tangent_t = add(dx_dt, scale(dx_dr, drdt));
        const real_t weight =
            s_rule.weights[is] * t_rule.weights[it] * std::abs(half_s * half_t) *
            norm(cross(tangent_s, tangent_t));
        if (!std::isfinite(weight) || weight <= real_t{0.0}) {
          continue;
        }
        patch.quadrature_points.push_back(eval.coordinates);
        patch.quadrature_normals.push_back(unit_or_default(embedded.outward_normal(eval.coordinates)));
        patch.quadrature_weights.push_back(weight);
        patch.quadrature_measure += weight;
      }
    }
  }

  patch.isoparametric_quadrature_available =
      patch.quadrature_measure > positive_tolerance(tol);
  patch.construction_policy = kTrueCurvedArrangementPolicy;
  return patch.isoparametric_quadrature_available;
}

bool add_true_curved_pyramid_graph_subcell(
    SideClosedTopology& closed,
    const MeshBase& mesh,
    const CutEntityRecord& cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    CutTopologySide side,
    const std::vector<index_t>& dofs,
    const std::vector<std::array<real_t, 3>>& parent_points,
    const std::vector<TetBasePoint>& base_polygon,
    const FaceGraphIntegral& measure,
    real_t tol,
    std::uint64_t salt) {
  if (!measure.available || measure.measure <= positive_tolerance(tol) ||
      base_polygon.size() < 3u) {
    return false;
  }

  std::vector<std::array<real_t, 3>> xi_vertices;
  xi_vertices.reserve(base_polygon.size() * 2u);
  for (const auto& base : base_polygon) {
    const auto interval = pyramid_side_interval_at_base_point(
        mesh, cell.entity, embedded, base, cfg, side, tol, true);
    if (!interval.available) {
      continue;
    }
    xi_vertices.push_back(pyramid_parametric_point(interval.r_lo, base.s, base.t));
    if (std::abs(interval.r_hi - interval.r_lo) > positive_tolerance(tol) * real_t{10.0}) {
      xi_vertices.push_back(pyramid_parametric_point(interval.r_hi, base.s, base.t));
    }
  }
  xi_vertices = unique_xi_vertices(xi_vertices, tol);
  auto faces = convex_hull_faces_indices(xi_vertices, tol);
  if (xi_vertices.size() < 4u || faces.size() < 4u) {
    TetBasePoint centroid;
    for (const auto& base : base_polygon) {
      centroid.s += base.s;
      centroid.t += base.t;
    }
    centroid.s /= static_cast<real_t>(base_polygon.size());
    centroid.t /= static_cast<real_t>(base_polygon.size());
    const auto center_interval = pyramid_side_interval_at_base_point(
        mesh, cell.entity, embedded, centroid, cfg, side, tol, false);
    if (!center_interval.available ||
        center_interval.r_hi - center_interval.r_lo <= positive_tolerance(tol)) {
      return false;
    }

    std::vector<std::array<real_t, 3>> fallback_vertices;
    fallback_vertices.reserve(2u + base_polygon.size() * 2u);
    fallback_vertices.push_back(
        pyramid_parametric_point(center_interval.r_lo, centroid.s, centroid.t));
    fallback_vertices.push_back(
        pyramid_parametric_point(center_interval.r_hi, centroid.s, centroid.t));
    for (std::size_t i = 0; i < base_polygon.size(); ++i) {
      TetBasePoint nearby;
      const auto& target = base_polygon[i];
      nearby.s = real_t{0.75} * centroid.s + real_t{0.25} * target.s;
      nearby.t = real_t{0.75} * centroid.t + real_t{0.25} * target.t;
      const auto interval = pyramid_side_interval_at_base_point(
          mesh, cell.entity, embedded, nearby, cfg, side, tol, false);
      if (!interval.available) {
        continue;
      }
      fallback_vertices.push_back(pyramid_parametric_point(interval.r_lo, nearby.s, nearby.t));
      if (std::abs(interval.r_hi - interval.r_lo) > positive_tolerance(tol) * real_t{10.0}) {
        fallback_vertices.push_back(pyramid_parametric_point(interval.r_hi, nearby.s, nearby.t));
      }
    }
    xi_vertices = unique_xi_vertices(std::move(fallback_vertices), tol);
    faces = convex_hull_faces_indices(xi_vertices, tol);
    if (xi_vertices.size() < 4u || faces.size() < 4u) {
      return false;
    }
  }

  std::vector<std::array<real_t, 3>> physical_points;
  physical_points.reserve(xi_vertices.size());
  for (const auto& xi : xi_vertices) {
    physical_points.push_back(
        CurvilinearEvaluator::evaluate_geometry(mesh, cell.entity, xi, cfg).coordinates);
  }
  const auto before = closed.subcells.size();
  add_integration_subcell(closed,
                          xi_vertices.size() == 5u ? CellFamily::Pyramid : CellFamily::Polyhedron,
                          physical_points,
                          faces,
                          measure.measure,
                          measure.centroid,
                          dofs,
                          parent_points,
                          embedded,
                          cell.provenance,
                          cell.global_id,
                          side,
                          tol,
                          salt,
                          xi_vertices,
                          true,
                          true,
                          kTrueCurvedArrangementPolicy);
  return closed.subcells.size() > before;
}

CutSideRegion make_true_curved_pyramid_side_region(
    const MeshBase& mesh,
    const CutEntityRecord& cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    CutTopologySide side,
    const std::vector<HexBaseQuad>& base_quads,
    const std::vector<std::uint64_t>& interface_vertex_ids,
    real_t tol) {
  CutSideRegion region;
  region.side = side;
  region.parent_cell = cell.entity;
  region.parent_cell_gid = cell.global_id;
  region.provenance = cell.provenance;
  region.integration_family = CellFamily::Pyramid;
  region.curved_isoparametric_topology = true;
  region.measure_from_linear_topology = true;
  region.stable_id = stable_entity_id(cell.provenance,
                                      cell.global_id,
                                      0u,
                                      static_cast<index_t>(side),
                                      real_t{0.0},
                                      404u);
  region.cut_vertices = interface_vertex_ids;

  const auto dofs = mesh.cell_geometry_dofs(cell.entity);
  std::vector<std::array<real_t, 3>> parent_points;
  parent_points.reserve(dofs.size());
  std::vector<real_t> distances;
  distances.reserve(dofs.size());
  for (const auto dof : dofs) {
    const auto point = mesh.geometry_dof_coords(dof, cfg);
    parent_points.push_back(point);
    distances.push_back(embedded.signed_distance(point));
  }
  for (std::size_t i = 0; i < dofs.size(); ++i) {
    const bool on_side = side == CutTopologySide::Negative
                             ? distances[i] <= positive_tolerance(tol)
                             : distances[i] >= -positive_tolerance(tol);
    if (on_side) {
      region.parent_geometry_dofs.push_back(dofs[i]);
    }
  }

  SideClosedTopology closed;
  std::array<real_t, 3> weighted_centroid{{0.0, 0.0, 0.0}};
  std::uint64_t subcell_salt = 0u;
  for (const auto& base_quad : base_quads) {
    const auto full_measure =
        integrate_pyramid_graph_region(mesh,
                                       cell.entity,
                                       embedded,
                                       base_quad,
                                       cfg,
                                       tol,
                                       false,
                                       side);
    if (full_measure.available) {
      region.parent_measure += full_measure.measure;
    }

    const auto measure =
        integrate_pyramid_graph_region(mesh,
                                       cell.entity,
                                       embedded,
                                       base_quad,
                                       cfg,
                                       tol,
                                       true,
                                       side);
    if (!measure.available || measure.measure <= positive_tolerance(tol)) {
      continue;
    }
    const auto base_polygon =
        clipped_pyramid_base_polygon_for_side(
            mesh, cell.entity, embedded, base_quad, cfg, side, tol);
    if (add_true_curved_pyramid_graph_subcell(closed,
                                              mesh,
                                              cell,
                                              embedded,
                                              cfg,
                                              side,
                                              dofs,
                                              parent_points,
                                              base_polygon,
                                              measure,
                                              tol,
                                              3800u + subcell_salt++)) {
      region.measure_estimate += measure.measure;
      weighted_centroid = add(weighted_centroid, scale(measure.centroid, measure.measure));
    }
  }

  if (region.measure_estimate > real_t{0.0}) {
    region.centroid_estimate =
        scale(weighted_centroid, real_t{1.0} / region.measure_estimate);
    region.volume_fraction_estimate = region.parent_measure > real_t{0.0}
                                          ? region.measure_estimate / region.parent_measure
                                          : real_t{0.0};
  }
  region.integration_vertices = closed.vertices;
  region.integration_subcells = closed.subcells;
  region.integration_region_vertices = closed.vertex_ids;
  region.integration_region_faces = closed.faces;
  region.closed_integration_topology =
      !region.integration_subcells.empty() &&
      std::all_of(region.integration_subcells.begin(),
                  region.integration_subcells.end(),
                  [](const auto& subcell) {
                    return subcell.closed_topology;
                  });
  return region;
}

bool supports_true_curved_pyramid_arrangement(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded) noexcept {
  return mesh.cell_shape(cell).family == CellFamily::Pyramid &&
         mesh.geometry_order(cell) > 1 &&
         embedded.kind == EmbeddedGeometryKind::Plane;
}

bool add_true_curved_pyramid_arrangement(
    CutTopologyRecord& topology,
    const MeshBase& mesh,
    const CutEntityRecord& cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    real_t tol,
    std::uint64_t& h) {
  const int order = std::max(1, mesh.geometry_order(cell.entity));
  const int base_segments = std::max(24, order * 10);
  const auto base_quads = make_uniform_hex_base_quads(base_segments);
  if (!true_curved_pyramid_columns_are_graph_compatible(
          mesh, cell.entity, embedded, cfg, tol)) {
    topology.diagnostics.push_back(
        "true curved pyramid arrangement requires a graph-compatible plane cut with at most one root per shrinking reference-column sample");
    return false;
  }
  const auto candidates =
      true_curved_pyramid_edge_roots(mesh, cell.entity, embedded, cfg, tol);
  if (candidates.size() < 3u) {
    topology.diagnostics.push_back(
        "true curved pyramid arrangement found fewer than three boundary roots in a cut high-order pyramid cell");
    return false;
  }

  const auto first_patch = topology.curved_patches.size();
  const bool recorded =
      record_curved_patch_from_candidates(topology,
                                          mesh,
                                          cell.entity,
                                          embedded,
                                          cfg,
                                          cell.provenance,
                                          cell.global_id,
                                          candidates,
                                          tol,
                                          4000u,
                                          true,
                                          kTrueCurvedArrangementPolicy);
  if (!recorded || topology.curved_patches.size() <= first_patch) {
    topology.diagnostics.push_back(
        "true curved pyramid arrangement failed to record interface surface topology");
    return false;
  }
  auto& patch = topology.curved_patches.back();
  if (!populate_true_curved_pyramid_patch_quadrature(
          patch, mesh, cell.entity, embedded, cfg, base_quads, tol)) {
    topology.diagnostics.push_back(
        "true curved pyramid arrangement failed to build interface surface quadrature");
    return false;
  }
  const auto interface_vertex_ids = patch.ordered_vertices;
  hash_curved_patch_record(h, patch);

  for (const auto side : {CutTopologySide::Negative, CutTopologySide::Positive}) {
    auto region = make_true_curved_pyramid_side_region(mesh,
                                                       cell,
                                                       embedded,
                                                       cfg,
                                                       side,
                                                       base_quads,
                                                       interface_vertex_ids,
                                                       tol);
    if (!region.closed_integration_topology || region.integration_subcells.empty()) {
      topology.diagnostics.push_back(
          "true curved pyramid arrangement failed to construct closed side-region topology");
      return false;
    }
    topology.side_regions.push_back(std::move(region));
    hash_side_region_record(h, topology.side_regions.back());
  }

  return true;
}

void add_clipped_tet_topology(
    SideClosedTopology& topology,
    const std::array<std::array<real_t, 3>, 4>& tet,
    const std::array<real_t, 4>& values,
    const std::vector<index_t>& parent_dofs,
    const std::vector<std::array<real_t, 3>>& parent_points,
    const EmbeddedGeometryDescriptor& embedded,
    const EmbeddedRegionProvenance& provenance,
    gid_t parent_gid,
    CutTopologySide side,
    real_t tol,
    std::uint64_t salt) {
  const auto clipped = clipped_tet_points(tet, values, side, tol);
  const auto points = unique_points(clipped, tol);
  if (points.size() < 4u) {
    return;
  }
  const real_t measure = convex_polyhedron_volume(points, tol);
  if (measure <= positive_tolerance(tol)) {
    return;
  }
  std::array<real_t, 3> centroid{{0.0, 0.0, 0.0}};
  for (const auto& p : points) {
    centroid = add(centroid, p);
  }
  centroid = scale(centroid, real_t{1.0} / static_cast<real_t>(points.size()));
  const auto faces = convex_hull_faces_indices(points, tol);
  add_integration_subcell(topology,
                          points.size() == 4u ? CellFamily::Tetra : CellFamily::Polyhedron,
                          points,
                          faces,
                          measure,
                          centroid,
                          parent_dofs,
                          parent_points,
                          embedded,
                          provenance,
                          parent_gid,
                          side,
                          tol,
                          salt);
}

void add_tessellated_subcell_topology(
    SideClosedTopology& topology,
    const MeshBase& mesh,
    index_t cell,
    CellFamily family,
    const std::vector<std::array<real_t, 3>>& points,
    const std::vector<std::array<real_t, 3>>& parent_parametric_points,
    const std::vector<index_t>& parent_dofs,
    const std::vector<std::array<real_t, 3>>& parent_points,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    const EmbeddedRegionProvenance& provenance,
    gid_t parent_gid,
    CutTopologySide side,
    real_t tol,
    std::uint64_t salt,
    std::string construction_policy = {}) {
  if (points.empty()) {
    return;
  }
  std::vector<real_t> values;
  values.reserve(points.size());
  for (const auto& point : points) {
    values.push_back(embedded.signed_distance(point));
  }

  const bool have_parametric = parent_parametric_points.size() == points.size();
  if (have_parametric && family == CellFamily::Line && points.size() >= 2u) {
    const auto clipped = clip_polygon_parametric_points(
        points, parent_parametric_points, values, side, tol);
    if (clipped.size() >= 2u) {
      const auto measure =
          isoparametric_line_measure(mesh, cell, clipped[0].xi, clipped[1].xi, cfg);
      if (measure.available && measure.measure > positive_tolerance(tol)) {
        add_integration_subcell(topology,
                                CellFamily::Line,
                                {clipped[0].point, clipped[1].point},
                                {{0, 1}},
                                measure.measure,
                                measure.centroid,
                                parent_dofs,
                                parent_points,
                                embedded,
                                provenance,
                                parent_gid,
                                side,
                                tol,
                                salt,
                                {clipped[0].xi, clipped[1].xi},
                                true,
                                true,
                                construction_policy);
      }
    }
    return;
  }

  if (have_parametric &&
      (family == CellFamily::Triangle ||
       family == CellFamily::Quad ||
       family == CellFamily::Polygon) &&
      points.size() >= 3u) {
    const auto clipped = clip_polygon_parametric_points(
        points, parent_parametric_points, values, side, tol);
    if (clipped.size() < 3u) {
      return;
    }
    for (std::size_t i = 1; i + 1 < clipped.size(); ++i) {
      const std::vector<std::array<real_t, 3>> tri{
          clipped[0].point, clipped[i].point, clipped[i + 1u].point};
      const std::vector<std::array<real_t, 3>> tri_xi{
          clipped[0].xi, clipped[i].xi, clipped[i + 1u].xi};
      const auto measure =
          isoparametric_triangle_measure(mesh, cell, tri_xi[0], tri_xi[1], tri_xi[2], cfg);
      if (!measure.available || measure.measure <= positive_tolerance(tol)) {
        continue;
      }
      add_integration_subcell(topology,
                              CellFamily::Triangle,
                              tri,
                              {{0, 1}, {1, 2}, {2, 0}},
                              measure.measure,
                              measure.centroid,
                              parent_dofs,
                              parent_points,
                              embedded,
                              provenance,
                              parent_gid,
                              side,
                              tol,
                              salt + static_cast<std::uint64_t>(i),
                              tri_xi,
                              true,
                              true,
                              construction_policy);
    }
    return;
  }

  if (have_parametric &&
      (family == CellFamily::Tetra ||
       family == CellFamily::Hex ||
       family == CellFamily::Wedge ||
       family == CellFamily::Pyramid) &&
      points.size() >= 4u) {
    const auto tets = linear_cell_parametric_tets(family, points, parent_parametric_points, tol);
    for (std::size_t i = 0; i < tets.size(); ++i) {
      const auto& tet = tets[i];
      const std::array<real_t, 4> vals{{embedded.signed_distance(tet.points[0]),
                                        embedded.signed_distance(tet.points[1]),
                                        embedded.signed_distance(tet.points[2]),
                                        embedded.signed_distance(tet.points[3])}};
      const auto clipped =
          clipped_tet_parametric_points(tet.points, tet.parent_parametric_points, vals, side, tol);
      if (clipped.size() < 4u) {
        continue;
      }
      std::vector<std::array<real_t, 3>> clipped_points;
      std::vector<std::array<real_t, 3>> clipped_xi;
      clipped_points.reserve(clipped.size());
      clipped_xi.reserve(clipped.size());
      for (const auto& point : clipped) {
        clipped_points.push_back(point.point);
        clipped_xi.push_back(point.xi);
      }
      const auto measure = isoparametric_polyhedron_measure(mesh, cell, clipped_xi, cfg, tol);
      if (!measure.available || measure.measure <= positive_tolerance(tol)) {
        continue;
      }
      const auto parent_measure =
          isoparametric_measure_for_subcell(mesh,
                                            cell,
                                            CellFamily::Tetra,
                                            {tet.parent_parametric_points[0],
                                             tet.parent_parametric_points[1],
                                             tet.parent_parametric_points[2],
                                             tet.parent_parametric_points[3]},
                                            cfg,
                                            tol);
      const real_t accepted_measure = parent_measure.available
                                          ? std::min(parent_measure.measure, measure.measure)
                                          : measure.measure;
      const auto faces = convex_hull_faces_indices(clipped_points, tol);
      add_integration_subcell(topology,
                              clipped_points.size() == 4u ? CellFamily::Tetra : CellFamily::Polyhedron,
                              clipped_points,
                              faces,
                              accepted_measure,
                              measure.centroid,
                              parent_dofs,
                              parent_points,
                              embedded,
                              provenance,
                              parent_gid,
                              side,
                              tol,
                              salt + static_cast<std::uint64_t>(i),
                              clipped_xi,
                              true,
                              true,
                              construction_policy);
    }
    return;
  }

  if (family == CellFamily::Line && points.size() >= 2u) {
    const auto clipped = clip_polygon_points(points, values, side, tol);
    if (clipped.size() >= 2u) {
      const real_t measure = norm(sub(clipped[1], clipped[0]));
      const auto c = scale(add(clipped[0], clipped[1]), real_t{0.5});
      add_integration_subcell(topology,
                              CellFamily::Line,
                              {clipped[0], clipped[1]},
                              {{0, 1}},
                              measure,
                              c,
                              parent_dofs,
                              parent_points,
                              embedded,
                              provenance,
                              parent_gid,
                              side,
                              tol,
                              salt);
    }
    return;
  }

  if ((family == CellFamily::Triangle ||
       family == CellFamily::Quad ||
       family == CellFamily::Polygon) &&
      points.size() >= 3u) {
    const auto clipped = clip_polygon_points(points, values, side, tol);
    if (clipped.size() < 3u) {
      return;
    }
    std::vector<std::array<index_t, 3>> triangles;
    if (!PolyGeometry::triangulate_planar_polygon(clipped, triangles)) {
      return;
    }
    for (std::size_t i = 0; i < triangles.size(); ++i) {
      const auto& indices = triangles[i];
      const std::vector<std::array<real_t, 3>> tri{
          clipped[static_cast<std::size_t>(indices[0])],
          clipped[static_cast<std::size_t>(indices[1])],
          clipped[static_cast<std::size_t>(indices[2])]};
      const real_t measure = MeshGeometry::triangle_area(tri[0], tri[1], tri[2]);
      const auto c = scale(add(add(tri[0], tri[1]), tri[2]), real_t{1.0} / real_t{3.0});
      add_integration_subcell(topology,
                              CellFamily::Triangle,
                              tri,
                              {{0, 1}, {1, 2}, {2, 0}},
                              measure,
                              c,
                              parent_dofs,
                              parent_points,
                              embedded,
                              provenance,
                              parent_gid,
                              side,
                              tol,
                              salt + static_cast<std::uint64_t>(i));
    }
    return;
  }

  const auto tets = PolyhedronTessellation::linear_cell_tets(family, points);
  for (std::size_t i = 0; i < tets.size(); ++i) {
    std::array<real_t, 4> vals{{embedded.signed_distance(tets[i].vertices[0]),
                                embedded.signed_distance(tets[i].vertices[1]),
                                embedded.signed_distance(tets[i].vertices[2]),
                                embedded.signed_distance(tets[i].vertices[3])}};
    add_clipped_tet_topology(topology,
                             tets[i].vertices,
                             vals,
                             parent_dofs,
                             parent_points,
                             embedded,
                             provenance,
                             parent_gid,
                             side,
                             tol,
                             salt + static_cast<std::uint64_t>(i));
  }
}

SideClosedTopology build_side_closed_topology_from_tessellated_cell(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    CutTopologySide side,
    real_t tol,
    const EmbeddedRegionProvenance& provenance,
    gid_t parent_gid,
    std::string construction_policy = {}) {
  SideClosedTopology topology;
  std::vector<index_t> parent_dofs = mesh.cell_geometry_dofs(cell);
  std::vector<std::array<real_t, 3>> parent_points;
  parent_points.reserve(parent_dofs.size());
  for (const auto dof : parent_dofs) {
    parent_points.push_back(mesh.geometry_dof_coords(dof, cfg));
  }

  try {
    const auto tessellated =
        Tessellator::tessellate_cell(mesh, cell, high_order_cut_tessellation_config(mesh, cell, cfg));
    for (int i = 0; i < tessellated.n_sub_elements(); ++i) {
      add_tessellated_subcell_topology(topology,
                                       mesh,
                                       cell,
                                       tessellated.sub_element_shape.family,
                                       tessellated_subcell_points(tessellated, i),
                                       tessellated_subcell_parametric_points(tessellated, i),
                                       parent_dofs,
                                       parent_points,
                                       embedded,
                                       cfg,
                                       provenance,
                                       parent_gid,
                                       side,
                                       tol,
                                       1200u + static_cast<std::uint64_t>(i) * 64u,
                                       construction_policy);
    }
  } catch (const std::exception&) {
    topology.closed = false;
  }

  return topology;
}

bool supports_true_curved_subdivision_arrangement(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded) noexcept {
  if (embedded.kind != EmbeddedGeometryKind::Plane ||
      mesh.geometry_order(cell) <= 1) {
    return false;
  }
  const auto family = mesh.cell_shape(cell).family;
  return family == CellFamily::Triangle ||
         family == CellFamily::Quad ||
         family == CellFamily::Hex ||
         family == CellFamily::Wedge ||
         family == CellFamily::Pyramid;
}

std::vector<std::uint64_t> interface_vertex_ids_from_patch_range(
    const CutTopologyRecord& topology,
    std::size_t first_patch) {
  std::vector<std::uint64_t> ids;
  for (std::size_t i = first_patch; i < topology.curved_patches.size(); ++i) {
    for (const auto id : topology.curved_patches[i].ordered_vertices) {
      if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
        ids.push_back(id);
      }
    }
  }
  return ids;
}

CutSideRegion make_true_curved_subdivision_side_region(
    const MeshBase& mesh,
    const CutEntityRecord& cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    CutTopologySide side,
    const std::vector<std::uint64_t>& interface_vertex_ids,
    real_t tol) {
  CutSideRegion region;
  region.side = side;
  region.parent_cell = cell.entity;
  region.parent_cell_gid = cell.global_id;
  region.provenance = cell.provenance;
  region.integration_family = mesh.cell_shape(cell.entity).family;
  region.curved_isoparametric_topology = true;
  region.stable_id = stable_entity_id(cell.provenance,
                                      cell.global_id,
                                      0u,
                                      static_cast<index_t>(side),
                                      real_t{0.0},
                                      404u);
  region.cut_vertices = interface_vertex_ids;

  const auto dofs = mesh.cell_geometry_dofs(cell.entity);
  std::vector<real_t> distances;
  distances.reserve(dofs.size());
  for (const auto dof : dofs) {
    distances.push_back(embedded.signed_distance(mesh.geometry_dof_coords(dof, cfg)));
  }
  for (std::size_t i = 0; i < dofs.size(); ++i) {
    const bool on_side = side == CutTopologySide::Negative
                             ? distances[i] <= positive_tolerance(tol)
                             : distances[i] >= -positive_tolerance(tol);
    if (on_side) {
      region.parent_geometry_dofs.push_back(dofs[i]);
    }
  }

  const auto measure =
      estimate_side_measure_from_tessellated_cell(mesh, cell.entity, embedded, cfg, side, tol);
  region.parent_measure = measure.parent_measure;
  region.measure_from_linear_topology = measure.available;

  const auto closed = build_side_closed_topology_from_tessellated_cell(
      mesh,
      cell.entity,
      embedded,
      cfg,
      side,
      tol,
      cell.provenance,
      cell.global_id,
      kTrueCurvedSubdivisionArrangementPolicy);
  region.integration_vertices = closed.vertices;
  region.integration_subcells = closed.subcells;
  region.integration_region_vertices = closed.vertex_ids;
  region.integration_region_faces = closed.faces;

  std::array<real_t, 3> weighted_centroid{{0.0, 0.0, 0.0}};
  for (const auto& subcell : region.integration_subcells) {
    if (subcell.measure <= real_t{0.0}) {
      continue;
    }
    region.measure_estimate += subcell.measure;
    weighted_centroid = add(weighted_centroid, scale(subcell.centroid, subcell.measure));
  }
  if (region.measure_estimate > real_t{0.0}) {
    region.centroid_estimate =
        scale(weighted_centroid, real_t{1.0} / region.measure_estimate);
    if (measure.available &&
        measure.measure > positive_tolerance(tol) &&
        std::isfinite(measure.measure)) {
      const real_t scale_factor = measure.measure / region.measure_estimate;
      if (std::isfinite(scale_factor) && scale_factor > real_t{0.0}) {
        for (auto& subcell : region.integration_subcells) {
          subcell.measure *= scale_factor;
        }
        region.measure_estimate = measure.measure;
      }
    }
  } else {
    region.measure_estimate = measure.measure;
    region.centroid_estimate = measure.centroid;
  }
  region.volume_fraction_estimate = region.parent_measure > real_t{0.0}
                                        ? std::max(real_t{0.0},
                                                   std::min(real_t{1.0},
                                                            region.measure_estimate /
                                                                region.parent_measure))
                                        : real_t{0.0};
  region.closed_integration_topology =
      !region.integration_subcells.empty() &&
      std::all_of(region.integration_subcells.begin(),
                  region.integration_subcells.end(),
                  [](const auto& subcell) {
                    return subcell.closed_topology &&
                           subcell.curved_isoparametric &&
                           subcell.measure_from_isoparametric_quadrature &&
                           subcell.construction_policy ==
                               kTrueCurvedSubdivisionArrangementPolicy;
                  });
  return region;
}

bool add_true_curved_subdivision_arrangement(
    CutTopologyRecord& topology,
    const MeshBase& mesh,
    const CutEntityRecord& cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    real_t tol,
    std::uint64_t& h) {
  const auto first_patch = topology.curved_patches.size();
  const auto added_patches =
      add_curved_patches_from_tessellated_cell(topology,
                                               mesh,
                                               cell.entity,
                                               embedded,
                                               cfg,
                                               cell.provenance,
                                               cell.global_id,
                                               tol,
                                               true,
                                               kTrueCurvedSubdivisionArrangementPolicy);
  if (added_patches == 0u || topology.curved_patches.size() <= first_patch) {
    topology.diagnostics.push_back(
        "true curved subdivision arrangement failed to record bounded high-order interface topology");
    return false;
  }

  for (std::size_t i = first_patch; i < topology.curved_patches.size(); ++i) {
    const auto& patch = topology.curved_patches[i];
    if (!patch.isoparametric_quadrature_available ||
        patch.linearized_surrogate ||
        !patch.exact_topology_available ||
        patch.construction_policy != kTrueCurvedSubdivisionArrangementPolicy) {
      topology.diagnostics.push_back(
          "true curved subdivision arrangement produced an incomplete curved patch record");
      return false;
    }
    hash_curved_patch_record(h, patch);
  }

  const auto interface_vertex_ids =
      interface_vertex_ids_from_patch_range(topology, first_patch);
  if (interface_vertex_ids.empty()) {
    topology.diagnostics.push_back(
        "true curved subdivision arrangement produced no interface vertices");
    return false;
  }

  std::vector<CutSideRegion> pending_regions;
  pending_regions.reserve(2u);
  for (const auto side : {CutTopologySide::Negative, CutTopologySide::Positive}) {
    auto region = make_true_curved_subdivision_side_region(mesh,
                                                           cell,
                                                           embedded,
                                                           cfg,
                                                           side,
                                                           interface_vertex_ids,
                                                           tol);
    if (!region.closed_integration_topology || region.integration_subcells.empty()) {
      topology.diagnostics.push_back(
          "true curved subdivision arrangement failed to construct closed side-region topology");
      return false;
    }
    pending_regions.push_back(std::move(region));
  }

  real_t parent_measure = real_t{0.0};
  real_t side_measure_sum = real_t{0.0};
  for (const auto& region : pending_regions) {
    parent_measure = std::max(parent_measure, region.parent_measure);
    side_measure_sum += region.measure_estimate;
  }
  if (parent_measure > positive_tolerance(tol) &&
      side_measure_sum > positive_tolerance(tol)) {
    const real_t scale_factor = parent_measure / side_measure_sum;
    if (std::isfinite(scale_factor) && scale_factor > real_t{0.0}) {
      for (auto& region : pending_regions) {
        region.measure_estimate *= scale_factor;
        region.volume_fraction_estimate =
            std::max(real_t{0.0},
                     std::min(real_t{1.0}, region.measure_estimate / parent_measure));
        for (auto& subcell : region.integration_subcells) {
          subcell.measure *= scale_factor;
        }
      }
    }
  }

  for (auto& region : pending_regions) {
    topology.side_regions.push_back(std::move(region));
    hash_side_region_record(h, topology.side_regions.back());
  }

  topology.diagnostics.push_back(
      "true curved high-order subdivision arrangement used bounded isoparametric subcell topology");
  return true;
}

SideClosedTopology build_side_closed_topology_for_cell(
    const MeshBase& mesh,
    index_t cell,
    const EmbeddedGeometryDescriptor& embedded,
    Configuration cfg,
    CutTopologySide side,
    real_t tol,
    const EmbeddedRegionProvenance& provenance,
    gid_t parent_gid) {
  SideClosedTopology topology;
  const auto shape = mesh.cell_shape(cell);
  const auto [vptr, n_vertices_span] = mesh.cell_vertices_span(cell);
  const std::size_t n_corners =
      shape.num_corners > 0
          ? std::min<std::size_t>(static_cast<std::size_t>(shape.num_corners), n_vertices_span)
          : n_vertices_span;
  if (n_corners == 0u) {
    return topology;
  }

  if (cell_uses_high_order_geometry(mesh, cell)) {
    return build_side_closed_topology_from_tessellated_cell(mesh,
                                                           cell,
                                                           embedded,
                                                           cfg,
                                                           side,
                                                           tol,
                                                           provenance,
                                                           parent_gid);
  }

  std::vector<index_t> dofs;
  std::vector<std::array<real_t, 3>> points;
  std::vector<real_t> values;
  dofs.reserve(n_corners);
  points.reserve(n_corners);
  values.reserve(n_corners);
  for (std::size_t i = 0; i < n_corners; ++i) {
    dofs.push_back(vptr[i]);
    points.push_back(mesh.geometry_dof_coords(vptr[i], cfg));
    values.push_back(embedded.signed_distance(points.back()));
  }

  if (shape.family == CellFamily::Line && points.size() >= 2u) {
    const auto clipped = clip_polygon_points(points, values, side, tol);
    if (clipped.size() >= 2u) {
      const real_t measure = norm(sub(clipped[1], clipped[0]));
      const auto c = scale(add(clipped[0], clipped[1]), real_t{0.5});
      add_integration_subcell(topology,
                              CellFamily::Line,
                              {clipped[0], clipped[1]},
                              {{0, 1}},
                              measure,
                              c,
                              dofs,
                              points,
                              embedded,
                              provenance,
                              parent_gid,
                              side,
                              tol,
                              700u);
    }
    return topology;
  }

  if ((shape.family == CellFamily::Triangle ||
       shape.family == CellFamily::Quad ||
       shape.family == CellFamily::Polygon) &&
      points.size() >= 3u) {
    const auto clipped = clip_polygon_points(points, values, side, tol);
    if (clipped.size() >= 3u) {
      std::vector<std::uint64_t> boundary_ids;
      boundary_ids.reserve(clipped.size());
      for (const auto& point : clipped) {
        const auto parent_dof = matching_parent_dof(dofs, points, point, tol);
        const bool on_interface = std::abs(embedded.signed_distance(point)) <= positive_tolerance(tol);
        boundary_ids.push_back(add_integration_vertex(topology,
                                                      point,
                                                      parent_dof,
                                                      on_interface,
                                                      parent_dof == INVALID_INDEX && !on_interface,
                                                      provenance,
                                                      parent_gid,
                                                      side));
      }
      add_boundary_cycle_faces(topology, boundary_ids);
      std::vector<std::array<index_t, 3>> triangles;
      if (!PolyGeometry::triangulate_planar_polygon(clipped, triangles)) {
        return topology;
      }
      for (std::size_t i = 0; i < triangles.size(); ++i) {
        const auto& indices = triangles[i];
        const std::vector<std::array<real_t, 3>> tri{
            clipped[static_cast<std::size_t>(indices[0])],
            clipped[static_cast<std::size_t>(indices[1])],
            clipped[static_cast<std::size_t>(indices[2])]};
        const real_t measure = MeshGeometry::triangle_area(tri[0], tri[1], tri[2]);
        const auto c = scale(add(add(tri[0], tri[1]), tri[2]), real_t{1.0} / real_t{3.0});
        add_integration_subcell(topology,
                                CellFamily::Triangle,
                                tri,
                                {{0, 1}, {1, 2}, {2, 0}},
                                measure,
                                c,
                                dofs,
                                points,
                                embedded,
                                provenance,
                                parent_gid,
                                side,
                                tol,
                                800u + static_cast<std::uint64_t>(i));
      }
    }
    return topology;
  }

  if (shape.family == CellFamily::Tetra ||
      shape.family == CellFamily::Hex ||
      shape.family == CellFamily::Wedge ||
      shape.family == CellFamily::Pyramid ||
      shape.family == CellFamily::Polyhedron) {
    const auto tets = PolyhedronTessellation::linear_cell_tets(mesh, cell, cfg);
    for (std::size_t i = 0; i < tets.size(); ++i) {
      std::array<real_t, 4> vals{{embedded.signed_distance(tets[i].vertices[0]),
                                  embedded.signed_distance(tets[i].vertices[1]),
                                  embedded.signed_distance(tets[i].vertices[2]),
                                  embedded.signed_distance(tets[i].vertices[3])}};
      add_clipped_tet_topology(topology,
                               tets[i].vertices,
                               vals,
                               dofs,
                               points,
                               embedded,
                               provenance,
                               parent_gid,
                               side,
                               tol,
                               900u + static_cast<std::uint64_t>(i));
    }
  }

  return topology;
}

} // namespace

std::uint64_t EmbeddedGeometryRevisionState::revision_key() const noexcept {
  std::uint64_t h = kFnvOffset;
  h = append_hash(h, geometry_epoch);
  h = append_hash(h, field_layout_revision);
  h = append_hash(h, field_value_revision);
  h = append_hash(h, source_surface_revision);
  h = append_hash(h, provenance_revision);
  h = append_hash(h, kinematic_constraint_revision);
  return h;
}

EmbeddedRevisionSnapshot EmbeddedRevisionSnapshot::capture(
    const MeshBase& mesh,
    Configuration configuration,
    std::uint64_t embedded_geometry_epoch,
    std::uint64_t embedded_constraint_epoch,
    std::uint64_t fe_layout_revision) {
  EmbeddedRevisionSnapshot snapshot;
  snapshot.configuration = normalized_configuration(configuration);
  snapshot.geometry_revision = mesh.geometry_revision();
  snapshot.topology_revision = mesh.topology_revision();
  snapshot.ownership_revision = mesh.ownership_revision();
  snapshot.numbering_revision = mesh.numbering_revision();
  snapshot.label_revision = mesh.label_revision();
  snapshot.active_configuration_epoch = mesh.active_configuration_epoch();
  snapshot.embedded_geometry_epoch = embedded_geometry_epoch;
  snapshot.embedded_constraint_epoch = embedded_constraint_epoch;
  snapshot.fe_layout_revision = fe_layout_revision;
  return snapshot;
}

EmbeddedRevisionSnapshot EmbeddedRevisionSnapshot::capture(
    const MeshBase& mesh,
    Configuration configuration,
    const EmbeddedGeometryRevisionState& embedded_revisions,
    std::uint64_t fe_layout_revision) {
  EmbeddedRevisionSnapshot snapshot = capture(mesh,
                                              configuration,
                                              embedded_revisions.geometry_epoch,
                                              embedded_revisions.kinematic_constraint_revision,
                                              fe_layout_revision);
  snapshot.embedded_field_layout_revision = embedded_revisions.field_layout_revision;
  snapshot.embedded_field_value_revision = embedded_revisions.field_value_revision;
  snapshot.embedded_source_surface_revision = embedded_revisions.source_surface_revision;
  snapshot.embedded_provenance_revision = embedded_revisions.provenance_revision;
  return snapshot;
}

bool EmbeddedRevisionSnapshot::matches(
    const MeshBase& mesh,
    Configuration configuration,
    std::uint64_t embedded_geometry_epoch,
    std::uint64_t embedded_constraint_epoch,
    std::uint64_t fe_layout_revision) const noexcept {
  return this->configuration == normalized_configuration(configuration) &&
         geometry_revision == mesh.geometry_revision() &&
         topology_revision == mesh.topology_revision() &&
         ownership_revision == mesh.ownership_revision() &&
         numbering_revision == mesh.numbering_revision() &&
         label_revision == mesh.label_revision() &&
         active_configuration_epoch == mesh.active_configuration_epoch() &&
         this->embedded_geometry_epoch == embedded_geometry_epoch &&
         this->embedded_constraint_epoch == embedded_constraint_epoch &&
         this->fe_layout_revision == fe_layout_revision;
}

bool EmbeddedRevisionSnapshot::matches(
    const MeshBase& mesh,
    Configuration configuration,
    const EmbeddedGeometryRevisionState& embedded_revisions,
    std::uint64_t fe_layout_revision) const noexcept {
  return this->configuration == normalized_configuration(configuration) &&
         geometry_revision == mesh.geometry_revision() &&
         topology_revision == mesh.topology_revision() &&
         ownership_revision == mesh.ownership_revision() &&
         numbering_revision == mesh.numbering_revision() &&
         label_revision == mesh.label_revision() &&
         active_configuration_epoch == mesh.active_configuration_epoch() &&
         embedded_geometry_epoch == embedded_revisions.geometry_epoch &&
         embedded_field_layout_revision == embedded_revisions.field_layout_revision &&
         embedded_field_value_revision == embedded_revisions.field_value_revision &&
         embedded_source_surface_revision == embedded_revisions.source_surface_revision &&
         embedded_provenance_revision == embedded_revisions.provenance_revision &&
         embedded_constraint_epoch == embedded_revisions.kinematic_constraint_revision &&
         this->fe_layout_revision == fe_layout_revision;
}

std::uint64_t EmbeddedRevisionSnapshot::revision_key() const noexcept {
  std::uint64_t h = kFnvOffset;
  h = append_hash(h, static_cast<std::uint64_t>(configuration));
  h = append_hash(h, geometry_revision);
  h = append_hash(h, topology_revision);
  h = append_hash(h, ownership_revision);
  h = append_hash(h, numbering_revision);
  h = append_hash(h, label_revision);
  h = append_hash(h, active_configuration_epoch);
  h = append_hash(h, embedded_geometry_epoch);
  h = append_hash(h, embedded_field_layout_revision);
  h = append_hash(h, embedded_field_value_revision);
  h = append_hash(h, embedded_source_surface_revision);
  h = append_hash(h, embedded_provenance_revision);
  h = append_hash(h, embedded_constraint_epoch);
  h = append_hash(h, fe_layout_revision);
  return h;
}

CutDistributedRevisionSnapshot CutDistributedRevisionSnapshot::capture(
    const DistributedMesh& mesh,
    const CutClassificationMap& map,
    const CutTopologyRecord& topology,
    const CutDistributedExchangePacket& local_packet,
    const CutDistributedExchangePacket& exchanged_packet) {
  CutDistributedRevisionSnapshot snapshot;
  snapshot.configuration = normalized_configuration(map.options.configuration);
  snapshot.geometry_revision = mesh.local_mesh().geometry_revision();
  snapshot.topology_revision = mesh.local_mesh().topology_revision();
  snapshot.ownership_revision = mesh.local_mesh().ownership_revision();
  snapshot.numbering_revision = mesh.local_mesh().numbering_revision();
  snapshot.label_revision = mesh.local_mesh().label_revision();
  snapshot.active_configuration_epoch = mesh.local_mesh().active_configuration_epoch();
  snapshot.classification_revision = map.revision_key();
  snapshot.cut_topology_revision = topology.topology_revision;
  snapshot.local_packet_revision = local_packet.revision_key;
  snapshot.exchanged_packet_revision = exchanged_packet.revision_key;
  snapshot.rank = mesh.rank();
  snapshot.world_size = static_cast<rank_t>(mesh.world_size());
  snapshot.distributed_revision = snapshot.revision_key();
  return snapshot;
}

bool CutDistributedRevisionSnapshot::matches(
    const DistributedMesh& mesh,
    const CutClassificationMap& map,
    const CutTopologyRecord& topology) const noexcept {
  return configuration == normalized_configuration(map.options.configuration) &&
         geometry_revision == mesh.local_mesh().geometry_revision() &&
         topology_revision == mesh.local_mesh().topology_revision() &&
         ownership_revision == mesh.local_mesh().ownership_revision() &&
         numbering_revision == mesh.local_mesh().numbering_revision() &&
         label_revision == mesh.local_mesh().label_revision() &&
         active_configuration_epoch == mesh.local_mesh().active_configuration_epoch() &&
         classification_revision == map.revision_key() &&
         cut_topology_revision == topology.topology_revision &&
         rank == mesh.rank() &&
         world_size == static_cast<rank_t>(mesh.world_size());
}

std::uint64_t CutDistributedRevisionSnapshot::revision_key() const noexcept {
  std::uint64_t h = kFnvOffset;
  h = append_hash(h, static_cast<std::uint64_t>(configuration));
  h = append_hash(h, geometry_revision);
  h = append_hash(h, topology_revision);
  h = append_hash(h, ownership_revision);
  h = append_hash(h, numbering_revision);
  h = append_hash(h, label_revision);
  h = append_hash(h, active_configuration_epoch);
  h = append_hash(h, classification_revision);
  h = append_hash(h, cut_topology_revision);
  h = append_hash(h, local_packet_revision);
  h = append_hash(h, exchanged_packet_revision);
  h = append_hash(h, static_cast<std::uint64_t>(rank));
  h = append_hash(h, static_cast<std::uint64_t>(world_size));
  return h;
}

bool CutDistributedState::valid_for(
    const DistributedMesh& mesh,
    const CutClassificationMap& map,
    const CutTopologyRecord& topology) const noexcept {
  return revision.matches(mesh, map, topology) &&
         revision.distributed_revision == revision.revision_key();
}

real_t EmbeddedGeometryDescriptor::signed_distance(
    const std::array<real_t, 3>& point) const noexcept {
  if (!finite_point(point)) {
    return std::numeric_limits<real_t>::quiet_NaN();
  }
  switch (kind) {
    case EmbeddedGeometryKind::Sphere:
      return norm(sub(point, origin)) - radius;
    case EmbeddedGeometryKind::SignedDistanceCallback:
      if (signed_distance_callback) {
        return signed_distance_callback(point);
      }
      return std::numeric_limits<real_t>::quiet_NaN();
    case EmbeddedGeometryKind::LevelSetField: {
      if (level_set_samples.empty()) {
        return std::numeric_limits<real_t>::quiet_NaN();
      }
      const EmbeddedLevelSetSample* best = &level_set_samples.front();
      real_t best_d2 = dot(sub(point, best->point), sub(point, best->point));
      for (const auto& sample : level_set_samples) {
        const real_t d2 = dot(sub(point, sample.point), sub(point, sample.point));
        if (d2 < best_d2) {
          best = &sample;
          best_d2 = d2;
        }
      }
      return best->value;
    }
    case EmbeddedGeometryKind::TriangulatedSurface: {
      if (surface_triangles.empty()) {
        return std::numeric_limits<real_t>::quiet_NaN();
      }
      real_t best_d2 = std::numeric_limits<real_t>::max();
      real_t best_signed = std::numeric_limits<real_t>::quiet_NaN();
      for (const auto& tri : surface_triangles) {
        const auto cp = closest_point_on_triangle(point, tri.vertices[0], tri.vertices[1], tri.vertices[2]);
        const auto n = triangle_normal(tri);
        const real_t d2 = dot(sub(point, cp), sub(point, cp));
        if (d2 < best_d2) {
          best_d2 = d2;
          best_signed = dot(sub(point, cp), n);
        }
      }
      return best_signed;
    }
    case EmbeddedGeometryKind::BooleanComposite: {
      if (children.empty()) {
        return std::numeric_limits<real_t>::quiet_NaN();
      }
      real_t value = children.front().signed_distance(point);
      if (boolean_operation == EmbeddedGeometryBooleanOperation::Difference && children.size() >= 2u) {
        value = std::max(value, -children[1].signed_distance(point));
        for (std::size_t i = 2; i < children.size(); ++i) {
          value = std::max(value, -children[i].signed_distance(point));
        }
        return value;
      }
      for (std::size_t i = 1; i < children.size(); ++i) {
        const real_t d = children[i].signed_distance(point);
        if (boolean_operation == EmbeddedGeometryBooleanOperation::Union) {
          value = std::min(value, d);
        } else {
          value = std::max(value, d);
        }
      }
      return value;
    }
    case EmbeddedGeometryKind::Plane:
      break;
  }
  return dot(sub(point, origin), unit_or_default(normal));
}

std::array<real_t, 3> EmbeddedGeometryDescriptor::outward_normal(
    const std::array<real_t, 3>& point) const noexcept {
  if (normal_callback) {
    return unit_or_default(normal_callback(point));
  }
  switch (kind) {
    case EmbeddedGeometryKind::Sphere:
      return unit_or_default(sub(point, origin));
    case EmbeddedGeometryKind::LevelSetField: {
      if (level_set_samples.empty()) {
        return unit_or_default(normal);
      }
      const EmbeddedLevelSetSample* best = &level_set_samples.front();
      real_t best_d2 = dot(sub(point, best->point), sub(point, best->point));
      for (const auto& sample : level_set_samples) {
        const real_t d2 = dot(sub(point, sample.point), sub(point, sample.point));
        if (d2 < best_d2) {
          best = &sample;
          best_d2 = d2;
        }
      }
      return unit_or_default(best->gradient);
    }
    case EmbeddedGeometryKind::TriangulatedSurface: {
      if (surface_triangles.empty()) {
        return unit_or_default(normal);
      }
      const EmbeddedSurfaceTriangle* best = &surface_triangles.front();
      auto cp = closest_point_on_triangle(point, best->vertices[0], best->vertices[1], best->vertices[2]);
      real_t best_d2 = dot(sub(point, cp), sub(point, cp));
      for (const auto& tri : surface_triangles) {
        cp = closest_point_on_triangle(point, tri.vertices[0], tri.vertices[1], tri.vertices[2]);
        const real_t d2 = dot(sub(point, cp), sub(point, cp));
        if (d2 < best_d2) {
          best = &tri;
          best_d2 = d2;
        }
      }
      return triangle_normal(*best);
    }
    case EmbeddedGeometryKind::BooleanComposite:
      if (!children.empty()) {
        return children.front().outward_normal(point);
      }
      break;
    case EmbeddedGeometryKind::SignedDistanceCallback:
    case EmbeddedGeometryKind::Plane:
      break;
  }
  return unit_or_default(normal);
}

std::array<real_t, 3> EmbeddedGeometryDescriptor::closest_point(
    const std::array<real_t, 3>& point) const noexcept {
  if (closest_point_callback) {
    return closest_point_callback(point);
  }
  switch (kind) {
    case EmbeddedGeometryKind::Sphere: {
      const auto n = unit_or_default(sub(point, origin));
      return add(origin, scale(n, radius));
    }
    case EmbeddedGeometryKind::TriangulatedSurface: {
      if (surface_triangles.empty()) {
        return point;
      }
      auto best = closest_point_on_triangle(point,
                                            surface_triangles.front().vertices[0],
                                            surface_triangles.front().vertices[1],
                                            surface_triangles.front().vertices[2]);
      real_t best_d2 = dot(sub(point, best), sub(point, best));
      for (const auto& tri : surface_triangles) {
        const auto cp = closest_point_on_triangle(point, tri.vertices[0], tri.vertices[1], tri.vertices[2]);
        const real_t d2 = dot(sub(point, cp), sub(point, cp));
        if (d2 < best_d2) {
          best = cp;
          best_d2 = d2;
        }
      }
      return best;
    }
    case EmbeddedGeometryKind::Plane: {
      const auto n = unit_or_default(normal);
      return sub(point, scale(n, dot(sub(point, origin), n)));
    }
    case EmbeddedGeometryKind::BooleanComposite:
      if (!children.empty()) {
        return children.front().closest_point(point);
      }
      break;
    case EmbeddedGeometryKind::SignedDistanceCallback:
    case EmbeddedGeometryKind::LevelSetField:
      break;
  }
  return point;
}

EmbeddedGeometryRevisionState EmbeddedGeometryDescriptor::effective_revisions() const noexcept {
  EmbeddedGeometryRevisionState out = revisions;
  out.geometry_epoch = std::max(out.geometry_epoch, geometry_epoch);
  out.provenance_revision = std::max(out.provenance_revision, provenance.provenance_epoch);
  for (const auto& child : children) {
    out = max_revision_state(out, child.effective_revisions());
  }
  return out;
}

EmbeddedGeometryQueryDiagnostic EmbeddedGeometryDescriptor::diagnose_query_support() const {
  EmbeddedGeometryQueryDiagnostic diagnostic;
  auto add = [&](std::string message) {
    diagnostic.ok = false;
    diagnostic.messages.push_back(std::move(message));
  };
  if (provenance.persistent_id.empty() && provenance.name.empty()) {
    add("embedded geometry has no persistent provenance identifier");
  }
  switch (kind) {
    case EmbeddedGeometryKind::SignedDistanceCallback:
      if (!signed_distance_callback) add("signed-distance callback is missing");
      if (!normal_callback) add("normal callback is missing");
      break;
    case EmbeddedGeometryKind::LevelSetField:
      if (level_set_samples.empty()) add("level-set embedded geometry has no samples");
      if (effective_revisions().field_value_revision == 0) {
        diagnostic.messages.push_back("level-set field value revision is zero");
      }
      break;
    case EmbeddedGeometryKind::TriangulatedSurface:
      if (surface_triangles.empty()) add("triangulated embedded surface has no triangles");
      if (effective_revisions().source_surface_revision == 0) {
        diagnostic.messages.push_back("source-surface revision is zero");
      }
      break;
    case EmbeddedGeometryKind::BooleanComposite:
      if (children.empty()) add("Boolean embedded geometry has no child geometries");
      for (const auto& child : children) {
        const auto child_diag = child.diagnose_query_support();
        if (!child_diag.ok) {
          add("Boolean child geometry lacks required query support");
          break;
        }
      }
      break;
    case EmbeddedGeometryKind::Plane:
    case EmbeddedGeometryKind::Sphere:
      break;
  }
  const real_t d = signed_distance(origin);
  if (!std::isfinite(d)) {
    add("signed-distance query returned a non-finite value at the descriptor origin");
  }
  if (norm(outward_normal(origin)) <= real_t{0.0}) {
    add("normal query returned a degenerate vector at the descriptor origin");
  }
  return diagnostic;
}

std::uint64_t CutPredicatePolicy::revision_key() const noexcept {
  std::uint64_t h = kFnvOffset;
  h = append_hash_string(h, name);
  h = append_hash_real(h, robust.intersection_tolerance);
  h = append_hash_real(h, robust.near_contact_tolerance);
  h = append_hash_real(h, robust.coplanar_tolerance);
  h = append_hash_real(h, robust.degenerate_tolerance);
  h = append_hash_real(h, robust.curved_sampling_tolerance);
  h = append_hash_real(h, robust.nonfinite_tolerance);
  h = append_hash_real(h, robust.aabb_padding);
  return h;
}

bool CutClassificationMap::valid_for(const MeshBase& mesh) const noexcept {
  auto revisions = embedded_geometry.effective_revisions();
  revisions.kinematic_constraint_revision =
      std::max(revisions.kinematic_constraint_revision, max_constraint_epoch(kinematic_constraints));
  return revision.matches(mesh,
                          options.configuration,
                          revisions,
                          options.fe_layout_revision);
}

std::uint64_t CutClassificationMap::revision_key() const noexcept {
  std::uint64_t h = revision.revision_key();
  h = append_hash(h, static_cast<std::uint64_t>(embedded_geometry.kind));
  h = append_hash(h, static_cast<std::uint64_t>(cells.size()));
  h = append_hash(h, static_cast<std::uint64_t>(faces.size()));
  h = append_hash(h, static_cast<std::uint64_t>(edges.size()));
  h = append_hash(h, static_cast<std::uint64_t>(kinematic_constraints.size()));
  h = append_hash(h, options.predicate_policy.revision_key());
  return h;
}

void CutClassificationMap::accept_trial() noexcept {
  if (state == CutClassificationState::Trial) {
    state = CutClassificationState::Committed;
  }
}

void CutClassificationMap::rollback_trial() {
  if (state == CutClassificationState::Trial) {
    cells.clear();
    faces.clear();
    edges.clear();
    state = CutClassificationState::RolledBack;
  }
}

CutClassificationTransaction::CutClassificationTransaction(CutClassificationMap& map)
    : map_(&map)
    , backup_(map)
    , state_(CutClassificationState::Trial) {}

void CutClassificationTransaction::stage(CutClassificationMap next) {
  if (!map_) {
    throw std::runtime_error("CutClassificationTransaction::stage: transaction has no map");
  }
  next.state = CutClassificationState::Trial;
  *map_ = std::move(next);
  state_ = CutClassificationState::Trial;
}

void CutClassificationTransaction::accept() {
  if (map_) {
    map_->accept_trial();
  }
  state_ = CutClassificationState::Committed;
}

void CutClassificationTransaction::rollback() {
  if (map_) {
    *map_ = backup_;
    map_->state = CutClassificationState::RolledBack;
  }
  state_ = CutClassificationState::RolledBack;
}

CutClassification classify_signed_distances(
    const std::vector<real_t>& signed_distances,
    real_t tolerance) noexcept {
  if (signed_distances.empty()) {
    return CutClassification::Degenerate;
  }
  bool has_negative = false;
  bool has_positive = false;
  bool has_zero = false;
  for (const real_t d : signed_distances) {
    if (!std::isfinite(d)) {
      return CutClassification::Degenerate;
    }
    has_negative = has_negative || d < -tolerance;
    has_positive = has_positive || d > tolerance;
    has_zero = has_zero || std::abs(d) <= tolerance;
  }
  if (has_negative && has_positive) {
    return CutClassification::Cut;
  }
  if (has_zero && (has_negative || has_positive)) {
    return CutClassification::Cut;
  }
  if (has_zero) {
    return CutClassification::Degenerate;
  }
  return has_negative ? CutClassification::Negative : CutClassification::Positive;
}

CutClassificationMap classify_embedded_geometry(
    const MeshBase& mesh,
    const EmbeddedGeometryDescriptor& embedded_geometry,
    const CutClassificationOptions& options) {
  CutClassificationMap map;
  map.name = embedded_geometry.provenance.name.empty()
                 ? embedded_geometry.provenance.persistent_id
                 : embedded_geometry.provenance.name;
  map.embedded_geometry = embedded_geometry;
  map.options = options;
  map.kinematic_constraints = options.kinematic_constraints;
  map.state = CutClassificationState::Trial;
  const auto query_diag = embedded_geometry.diagnose_query_support();
  map.diagnostics.push_back(query_diag);
  auto embedded_revisions = embedded_geometry.effective_revisions();
  embedded_revisions.kinematic_constraint_revision =
      std::max(embedded_revisions.kinematic_constraint_revision, max_constraint_epoch(map.kinematic_constraints));
  map.revision = EmbeddedRevisionSnapshot::capture(mesh,
                                                   options.configuration,
                                                   embedded_revisions,
                                                   options.fe_layout_revision);

  const Configuration cfg = normalized_configuration(options.configuration);
  const real_t tol = effective_tolerance(options);

  if (options.classify_cells) {
    map.cells.reserve(mesh.n_cells());
    for (index_t c = 0; c < static_cast<index_t>(mesh.n_cells()); ++c) {
      const auto dofs = mesh.cell_geometry_dofs(c);
      const auto distances = signed_distances_for_dofs(mesh, embedded_geometry, dofs, cfg);
      const auto local_edges = cell_local_edges(mesh, c, dofs);
      const auto intersections = edge_intersections(
          mesh, embedded_geometry, dofs, local_edges, distances, cfg, tol);
      map.cells.push_back(make_record(CutEntityKind::Cell,
                                      c,
                                      entity_gid(mesh, CutEntityKind::Cell, c),
                                      0,
                                      dofs,
                                      distances,
                                      intersections,
                                      tol,
                                      embedded_geometry.provenance));
    }
  }

  if (options.classify_faces) {
    map.faces.reserve(mesh.n_faces());
    for (index_t f = 0; f < static_cast<index_t>(mesh.n_faces()); ++f) {
      auto dofs = span_to_vector(mesh.face_vertices_span(f));
      const auto& face_shape = mesh.face_shapes().at(static_cast<std::size_t>(f));
      if (face_shape.num_corners > 0 &&
          static_cast<std::size_t>(face_shape.num_corners) < dofs.size()) {
        dofs.resize(static_cast<std::size_t>(face_shape.num_corners));
      }
      const auto distances = signed_distances_for_dofs(mesh, embedded_geometry, dofs, cfg);
      const auto intersections = edge_intersections(
          mesh, embedded_geometry, dofs, cyclic_edges(dofs.size()), distances, cfg, tol);
      map.faces.push_back(make_record(CutEntityKind::Face,
                                      f,
                                      entity_gid(mesh, CutEntityKind::Face, f),
                                      0,
                                      dofs,
                                      distances,
                                      intersections,
                                      tol,
                                      embedded_geometry.provenance));
    }
  }

  if (options.classify_edges) {
    map.edges.reserve(mesh.n_edges());
    for (index_t e = 0; e < static_cast<index_t>(mesh.n_edges()); ++e) {
      const auto ev = mesh.edge_vertices(e);
      const std::vector<index_t> dofs{ev[0], ev[1]};
      const auto distances = signed_distances_for_dofs(mesh, embedded_geometry, dofs, cfg);
      const auto intersections = edge_intersections(
          mesh,
          embedded_geometry,
          dofs,
          std::vector<std::array<index_t, 2>>{{{0, 1}}},
          distances,
          cfg,
          tol);
      map.edges.push_back(make_record(CutEntityKind::Edge,
                                      e,
                                      entity_gid(mesh, CutEntityKind::Edge, e),
                                      0,
                                      dofs,
                                      distances,
                                      intersections,
                                      tol,
                                      embedded_geometry.provenance));
    }
  }

  return map;
}

CutClassificationMap classify_embedded_geometry(
    const DistributedMesh& mesh,
    const EmbeddedGeometryDescriptor& embedded_geometry,
    const CutClassificationOptions& options) {
  auto map = classify_embedded_geometry(mesh.local_mesh(), embedded_geometry, options);
  update_distributed_owners(mesh, map);
  return map;
}

void EmbeddedGeometryRegistry::register_geometry(EmbeddedGeometryDescriptor descriptor) {
  const std::string id = !descriptor.provenance.persistent_id.empty()
                             ? descriptor.provenance.persistent_id
                             : descriptor.provenance.name;
  if (id.empty()) {
    throw std::invalid_argument("EmbeddedGeometryRegistry::register_geometry requires persistent provenance");
  }
  descriptor.provenance.persistent_id = id;
  if (descriptor.revisions.geometry_epoch == 0) {
    descriptor.revisions.geometry_epoch = descriptor.geometry_epoch;
  }
  if (descriptor.revisions.provenance_revision == 0) {
    descriptor.revisions.provenance_revision = descriptor.provenance.provenance_epoch;
  }
  geometries_[id] = std::move(descriptor);
  ++registry_epoch_;
}

bool EmbeddedGeometryRegistry::contains(const std::string& persistent_id) const noexcept {
  return geometries_.find(persistent_id) != geometries_.end();
}

const EmbeddedGeometryDescriptor* EmbeddedGeometryRegistry::find(
    const std::string& persistent_id) const noexcept {
  const auto it = geometries_.find(persistent_id);
  return it == geometries_.end() ? nullptr : &it->second;
}

const EmbeddedGeometryDescriptor& EmbeddedGeometryRegistry::require(
    const std::string& persistent_id) const {
  const auto* descriptor = find(persistent_id);
  if (!descriptor) {
    throw std::out_of_range("EmbeddedGeometryRegistry::require: unknown embedded geometry id");
  }
  return *descriptor;
}

void EmbeddedGeometryRegistry::erase(const std::string& persistent_id) {
  if (geometries_.erase(persistent_id) > 0u) {
    ++registry_epoch_;
  }
}

void EmbeddedGeometryRegistry::clear() {
  if (!geometries_.empty()) {
    geometries_.clear();
    ++registry_epoch_;
  }
}

std::vector<std::string> EmbeddedGeometryRegistry::active_geometry_ids() const {
  std::vector<std::string> ids;
  for (const auto& entry : geometries_) {
    if (entry.second.active) {
      ids.push_back(entry.first);
    }
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

std::vector<CutClassificationMap> EmbeddedGeometryRegistry::classify_active(
    const MeshBase& mesh,
    const CutClassificationOptions& options) const {
  std::vector<CutClassificationMap> maps;
  const auto ids = active_geometry_ids();
  maps.reserve(ids.size());
  for (const auto& id : ids) {
    maps.push_back(classify_embedded_geometry(mesh, require(id), options));
  }
  return maps;
}

EmbeddedGeometryRegistrySnapshot EmbeddedGeometryRegistry::snapshot() const noexcept {
  EmbeddedGeometryRegistrySnapshot snapshot;
  snapshot.registry_epoch = registry_epoch_;
  for (const auto& entry : geometries_) {
    if (entry.second.active) {
      ++snapshot.active_geometry_count;
    }
    const auto revisions = entry.second.effective_revisions();
    snapshot.geometry_revision = std::max(snapshot.geometry_revision, revisions.geometry_epoch);
    snapshot.field_layout_revision = std::max(snapshot.field_layout_revision, revisions.field_layout_revision);
    snapshot.field_value_revision = std::max(snapshot.field_value_revision, revisions.field_value_revision);
    snapshot.source_surface_revision = std::max(snapshot.source_surface_revision, revisions.source_surface_revision);
    snapshot.provenance_revision = std::max(snapshot.provenance_revision, revisions.provenance_revision);
    snapshot.kinematic_constraint_revision =
        std::max(snapshot.kinematic_constraint_revision, revisions.kinematic_constraint_revision);
  }
  return snapshot;
}

std::vector<EmbeddedGeometryDescriptor> EmbeddedGeometryRegistry::descriptors() const {
  std::vector<std::pair<std::string, EmbeddedGeometryDescriptor>> entries(
      geometries_.begin(), geometries_.end());
  std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
    return a.first < b.first;
  });
  std::vector<EmbeddedGeometryDescriptor> out;
  out.reserve(entries.size());
  for (auto& entry : entries) {
    out.push_back(std::move(entry.second));
  }
  return out;
}

EmbeddedGeometryQueryDiagnostic diagnose_embedded_geometry_query_support(
    const EmbeddedGeometryDescriptor& embedded_geometry) {
  return embedded_geometry.diagnose_query_support();
}

EmbeddedGeometryDescriptor make_triangulated_surface_descriptor(
    std::string persistent_id,
    std::vector<EmbeddedSurfaceTriangle> triangles,
    Configuration configuration,
    std::uint64_t source_surface_revision) {
  EmbeddedGeometryDescriptor descriptor;
  descriptor.kind = EmbeddedGeometryKind::TriangulatedSurface;
  descriptor.configuration = normalized_configuration(configuration);
  descriptor.provenance.persistent_id = std::move(persistent_id);
  descriptor.provenance.name = descriptor.provenance.persistent_id;
  descriptor.provenance.provenance_epoch = source_surface_revision;
  descriptor.geometry_epoch = source_surface_revision;
  descriptor.revisions.geometry_epoch = source_surface_revision;
  descriptor.revisions.source_surface_revision = source_surface_revision;
  descriptor.revisions.provenance_revision = source_surface_revision;
  descriptor.surface_triangles = std::move(triangles);
  for (auto& tri : descriptor.surface_triangles) {
    if (tri.provenance.empty()) {
      tri.provenance = descriptor.provenance;
    }
  }
  return descriptor;
}

EmbeddedGeometryDescriptor read_ascii_stl_embedded_surface(
    const std::string& path,
    std::string persistent_id,
    Configuration configuration,
    std::uint64_t source_surface_revision) {
  std::ifstream in(path);
  if (!in.good()) {
    throw std::runtime_error("read_ascii_stl_embedded_surface: cannot open '" + path + "'");
  }

  std::vector<EmbeddedSurfaceTriangle> triangles;
  std::vector<std::array<real_t, 3>> pending_vertices;
  std::string token;
  while (in >> token) {
    if (token == "vertex") {
      std::array<real_t, 3> p{{0.0, 0.0, 0.0}};
      in >> p[0] >> p[1] >> p[2];
      if (!in || !finite_point(p)) {
        throw std::runtime_error("read_ascii_stl_embedded_surface: malformed or non-finite vertex");
      }
      pending_vertices.push_back(p);
      if (pending_vertices.size() == 3u) {
        EmbeddedSurfaceTriangle tri;
        tri.vertices = {{pending_vertices[0], pending_vertices[1], pending_vertices[2]}};
        pending_vertices.clear();
        triangles.push_back(tri);
      }
    }
  }
  if (!pending_vertices.empty()) {
    throw std::runtime_error("read_ascii_stl_embedded_surface: incomplete triangle vertex list");
  }
  return make_triangulated_surface_descriptor(
      std::move(persistent_id), std::move(triangles), configuration, source_surface_revision);
}

std::vector<EmbeddedGeometryRestartRecord> make_embedded_geometry_restart_records(
    const EmbeddedGeometryRegistry& registry) {
  const auto descriptors = registry.descriptors();
  std::vector<EmbeddedGeometryRestartRecord> records;
  records.reserve(descriptors.size());
  for (const auto& descriptor : descriptors) {
    records.push_back(make_restart_record(descriptor));
  }
  return records;
}

EmbeddedGeometryRegistry restore_embedded_geometry_registry(
    const std::vector<EmbeddedGeometryRestartRecord>& records) {
  EmbeddedGeometryRegistry registry;
  for (const auto& record : records) {
    auto descriptor = descriptor_from_restart_record(record);
    if (record.requires_application_reregistration &&
        descriptor.kind == EmbeddedGeometryKind::SignedDistanceCallback) {
      descriptor.active = false;
    }
    registry.register_geometry(std::move(descriptor));
  }
  return registry;
}

CutOperationDiagnostic diagnose_cut_operation(
    const CutClassificationMap& map,
    std::string operation) {
  CutOperationDiagnostic diagnostic;
  diagnostic.operation = std::move(operation);
  diagnostic.predicate_policy_key = map.options.predicate_policy.revision_key();
  diagnostic.mesh_and_embedded_revision = map.revision;
  diagnostic.embedded_revision = map.embedded_geometry.effective_revisions();
  diagnostic.fe_layout_revision = map.options.fe_layout_revision;
  for (const auto& map_diag : map.diagnostics) {
    if (!map_diag.ok) {
      diagnostic.ok = false;
    }
    diagnostic.messages.insert(diagnostic.messages.end(),
                               map_diag.messages.begin(),
                               map_diag.messages.end());
  }
  const auto collect_record_diagnostics = [&](const std::vector<CutEntityRecord>& records) {
    for (const auto& record : records) {
      if (!record.diagnostics.empty()) {
        diagnostic.ok = false;
        diagnostic.messages.insert(diagnostic.messages.end(),
                                   record.diagnostics.begin(),
                                   record.diagnostics.end());
      }
    }
  };
  collect_record_diagnostics(map.cells);
  collect_record_diagnostics(map.faces);
  collect_record_diagnostics(map.edges);
  if (diagnostic.messages.empty()) {
    diagnostic.messages.push_back("cut operation completed with recorded revision and predicate metadata");
  }
  return diagnostic;
}

EmbeddedGeometryQueryDiagnostic diagnose_boolean_region_composition(
    const EmbeddedGeometryDescriptor& embedded_geometry,
    const std::vector<std::array<real_t, 3>>& sample_points,
    const CutPredicatePolicy& predicate_policy) {
  EmbeddedGeometryQueryDiagnostic diagnostic;
  if (embedded_geometry.kind != EmbeddedGeometryKind::BooleanComposite) {
    diagnostic.messages.push_back("embedded geometry is not a Boolean composition");
    return diagnostic;
  }
  if (embedded_geometry.children.empty()) {
    diagnostic.ok = false;
    diagnostic.messages.push_back("Boolean composition has no child regions");
    return diagnostic;
  }

  const real_t tol = predicate_policy.classification_tolerance();
  bool observed_overlap = false;
  bool observed_uncovered_intersection_sample = false;
  bool observed_difference_overlap = false;
  for (const auto& p : sample_points) {
    int inside_count = 0;
    bool inside_base = false;
    bool inside_subtracted = false;
    for (std::size_t i = 0; i < embedded_geometry.children.size(); ++i) {
      const bool inside = embedded_geometry.children[i].signed_distance(p) <= tol;
      inside_count += inside ? 1 : 0;
      if (i == 0) inside_base = inside;
      if (i > 0) inside_subtracted = inside_subtracted || inside;
    }
    observed_overlap = observed_overlap || inside_count > 1;
    observed_uncovered_intersection_sample =
        observed_uncovered_intersection_sample ||
        (embedded_geometry.boolean_operation == EmbeddedGeometryBooleanOperation::Intersection &&
         inside_count == 0);
    observed_difference_overlap =
        observed_difference_overlap ||
        (embedded_geometry.boolean_operation == EmbeddedGeometryBooleanOperation::Difference &&
         inside_base && inside_subtracted);
  }

  if (embedded_geometry.boolean_operation == EmbeddedGeometryBooleanOperation::Union && observed_overlap) {
    diagnostic.messages.push_back("Boolean union has overlapping child regions in sampled provenance checks");
  }
  if (observed_uncovered_intersection_sample) {
    diagnostic.messages.push_back("Boolean intersection samples include points outside every child region");
  }
  if (observed_difference_overlap) {
    diagnostic.messages.push_back("Boolean difference samples include base/subtracted overlap");
  }
  for (const auto& child : embedded_geometry.children) {
    if (child.kind != EmbeddedGeometryKind::BooleanComposite) {
      continue;
    }
    auto child_diagnostic =
        diagnose_boolean_region_composition(child, sample_points, predicate_policy);
    if (!child_diagnostic.ok) {
      diagnostic.ok = false;
    }
    const std::string child_id = !child.provenance.persistent_id.empty()
                                     ? child.provenance.persistent_id
                                     : child.provenance.name;
    for (auto& message : child_diagnostic.messages) {
      diagnostic.messages.push_back(
          "nested Boolean child '" + child_id + "': " + std::move(message));
    }
  }
  return diagnostic;
}

CutTopologyRecord reconstruct_cut_topology(
    const MeshBase& mesh,
    const CutClassificationMap& map,
    const CutTopologyOptions& options) {
  CutTopologyRecord topology;
  topology.predicate_policy_key = options.predicate_policy.revision_key();
  const bool true_curved_arrangement_requested =
      options.curved_arrangement_mode == CutCurvedArrangementMode::TrueArrangement;
  topology.linearized_cut_mode =
      !true_curved_arrangement_requested && options.allow_linearized_high_order_geometry;
  if (mesh.has_high_order_geometry() && true_curved_arrangement_requested) {
    topology.diagnostics.push_back("true curved high-order cut arrangement mode requested");
  } else if (mesh.has_high_order_geometry() && options.allow_linearized_high_order_geometry) {
    topology.diagnostics.push_back("high-order geometry classified in controlled linearized-cut mode");
  } else if (mesh.has_high_order_geometry()) {
    topology.supported = false;
    topology.diagnostics.push_back("curved high-order cut topology reconstruction is not enabled");
  }

  std::uint64_t h = kFnvOffset;
  h = append_hash(h, map.revision_key());
  h = append_hash(h, topology.predicate_policy_key);
  const Configuration cfg = normalized_configuration(options.configuration);
  const real_t tol = options.predicate_policy.classification_tolerance();

  for (const auto& cell : map.cells) {
    if (cell.classification != CutClassification::Cut) {
      continue;
    }
    if (cell_uses_high_order_geometry(mesh, cell.entity) && true_curved_arrangement_requested) {
      const auto try_true_curved_subdivision = [&]() {
        return supports_true_curved_subdivision_arrangement(mesh,
                                                            cell.entity,
                                                            map.embedded_geometry) &&
               add_true_curved_subdivision_arrangement(topology,
                                                       mesh,
                                                       cell,
                                                       map.embedded_geometry,
                                                       cfg,
                                                       tol,
                                                       h);
      };
      if (supports_true_curved_line_arrangement(mesh, cell.entity, map.embedded_geometry)) {
        if (add_true_curved_line_arrangement(topology,
                                             mesh,
                                             cell,
                                             map.embedded_geometry,
                                             cfg,
                                             tol,
                                             h)) {
          continue;
        }
        topology.supported = false;
        continue;
      }
      if (supports_true_curved_face_arrangement(mesh, cell.entity, map.embedded_geometry)) {
        const auto vertex_count = topology.vertices.size();
        const auto interface_polygon_count = topology.interface_polygons.size();
        const auto curved_patch_count = topology.curved_patches.size();
        const auto side_region_count = topology.side_regions.size();
        if (add_true_curved_face_arrangement(topology,
                                             mesh,
                                             cell,
                                             map.embedded_geometry,
                                             cfg,
                                             tol,
                                             h)) {
          continue;
        }
        topology.vertices.resize(vertex_count);
        topology.interface_polygons.resize(interface_polygon_count);
        topology.curved_patches.resize(curved_patch_count);
        topology.side_regions.resize(side_region_count);
        if (try_true_curved_subdivision()) {
          continue;
        }
        topology.supported = false;
        continue;
      }
      if (supports_true_curved_tet_arrangement(mesh, cell.entity, map.embedded_geometry)) {
        if (add_true_curved_tet_arrangement(topology,
                                            mesh,
                                            cell,
                                            map.embedded_geometry,
                                            cfg,
                                            tol,
                                            h)) {
          continue;
        }
        if (try_true_curved_subdivision()) {
          continue;
        }
        topology.supported = false;
        continue;
      }
      if (supports_true_curved_hex_arrangement(mesh, cell.entity, map.embedded_geometry)) {
        if (add_true_curved_hex_arrangement(topology,
                                            mesh,
                                            cell,
                                            map.embedded_geometry,
                                            cfg,
                                            tol,
                                            h)) {
          continue;
        }
        if (try_true_curved_subdivision()) {
          continue;
        }
        topology.supported = false;
        continue;
      }
      if (supports_true_curved_wedge_arrangement(mesh, cell.entity, map.embedded_geometry)) {
        if (add_true_curved_wedge_arrangement(topology,
                                              mesh,
                                              cell,
                                              map.embedded_geometry,
                                              cfg,
                                              tol,
                                              h)) {
          continue;
        }
        if (try_true_curved_subdivision()) {
          continue;
        }
        topology.supported = false;
        continue;
      }
      if (supports_true_curved_pyramid_arrangement(mesh, cell.entity, map.embedded_geometry)) {
        if (add_true_curved_pyramid_arrangement(topology,
                                                mesh,
                                                cell,
                                                map.embedded_geometry,
                                                cfg,
                                                tol,
                                                h)) {
          continue;
        }
        if (try_true_curved_subdivision()) {
          continue;
        }
        topology.supported = false;
        continue;
      }
      if (try_true_curved_subdivision()) {
        continue;
      }
      topology.supported = false;
      topology.diagnostics.push_back(
          "true curved arrangement currently supports high-order line, triangle, quad, tetra, hex, wedge, and pyramid cells cut by planes; "
          "non-graph plane face and volume cuts require bounded subdivision support, and non-plane high-order cuts should use LinearizedSurrogate mode until qualified");
      continue;
    }

    std::vector<CutTopologyVertex> local_vertices;
    local_vertices.reserve(cell.intersections.size());
    for (const auto& hit : cell.intersections) {
      CutTopologyVertex v;
      v.parent_cell = cell.entity;
      v.parent_cell_gid = cell.global_id;
      v.endpoint_a = hit.endpoint_a;
      v.endpoint_b = hit.endpoint_b;
      v.edge_fraction = hit.edge_fraction;
      v.point = hit.point;
      v.normal = hit.normal;
      v.provenance = cell.provenance;
      populate_parent_parametric_coordinate(v, mesh, cell.entity, cfg, tol);
      v.stable_id = stable_entity_id(cell.provenance,
                                     cell.global_id,
                                     stable_geometry_dof_key(mesh, hit.endpoint_a),
                                     stable_geometry_dof_key(mesh, hit.endpoint_b),
                                     hit.edge_fraction,
                                     101u);
      local_vertices.push_back(v);
    }

    std::sort(local_vertices.begin(), local_vertices.end(), [](const auto& a, const auto& b) {
      if (a.stable_id != b.stable_id) return a.stable_id < b.stable_id;
      if (a.point[0] != b.point[0]) return a.point[0] < b.point[0];
      if (a.point[1] != b.point[1]) return a.point[1] < b.point[1];
      return a.point[2] < b.point[2];
    });
    local_vertices.erase(std::unique(local_vertices.begin(),
                                     local_vertices.end(),
                                     [](const auto& a, const auto& b) {
                                       return a.stable_id == b.stable_id;
                                     }),
                         local_vertices.end());

    const auto normal = first_valid_normal(cell, map.embedded_geometry);
    if (local_vertices.size() >= 3u) {
      std::array<real_t, 3> centroid{{0.0, 0.0, 0.0}};
      for (const auto& v : local_vertices) {
        centroid = add(centroid, v.point);
      }
      centroid = scale(centroid, real_t{1.0} / static_cast<real_t>(local_vertices.size()));
      auto tangent0 = std::abs(normal[0]) < real_t{0.9}
                          ? unit_or_default(cross(normal, {{1.0, 0.0, 0.0}}))
                          : unit_or_default(cross(normal, {{0.0, 1.0, 0.0}}));
      auto tangent1 = cross(normal, tangent0);
      std::sort(local_vertices.begin(), local_vertices.end(), [&](const auto& a, const auto& b) {
        const auto ra = sub(a.point, centroid);
        const auto rb = sub(b.point, centroid);
        return std::atan2(dot(ra, tangent1), dot(ra, tangent0)) <
               std::atan2(dot(rb, tangent1), dot(rb, tangent0));
      });
    }

    for (const auto& v : local_vertices) {
      topology.vertices.push_back(v);
      h = append_hash(h, v.stable_id);
    }

    if (local_vertices.size() >= 2u) {
      for (std::size_t i = 0; i < local_vertices.size(); ++i) {
        const auto& a = local_vertices[i];
        const auto& b = local_vertices[(i + 1u) % local_vertices.size()];
        if (local_vertices.size() == 2u && i == 1u) {
          break;
        }
        CutTopologyEdge edge;
        edge.parent_cell = cell.entity;
        edge.parent_cell_gid = cell.global_id;
        edge.vertex_a = a.stable_id;
        edge.vertex_b = b.stable_id;
        edge.provenance = cell.provenance;
        edge.stable_id = stable_entity_id(cell.provenance,
                                          cell.global_id,
                                          static_cast<index_t>(i),
                                          static_cast<index_t>((i + 1u) % local_vertices.size()),
                                          real_t{0.0},
                                          202u);
        topology.edges.push_back(edge);
        h = append_hash(h, edge.stable_id);
      }
    } else {
      topology.diagnostics.push_back("cut cell has fewer than two unique cut vertices");
    }

    const auto parent_topo_kind = mesh.cell_shape(cell.entity).topo_kind();
    const std::size_t min_interface_vertices =
        parent_topo_kind == EntityKind::Volume ? 3u
        : parent_topo_kind == EntityKind::Face ? 2u
                                                : 1u;
    if (local_vertices.size() >= min_interface_vertices) {
      CutInterfacePolygon polygon;
      polygon.parent_cell = cell.entity;
      polygon.parent_cell_gid = cell.global_id;
      polygon.normal = normal;
      polygon.provenance = cell.provenance;
      polygon.stable_id = stable_entity_id(cell.provenance,
                                           cell.global_id,
                                           static_cast<index_t>(local_vertices.size()),
                                           0u,
                                           real_t{0.0},
                                           303u);
      for (const auto& v : local_vertices) {
        polygon.ordered_vertices.push_back(v.stable_id);
      }
      topology.interface_polygons.push_back(std::move(polygon));
      h = append_hash(h, topology.interface_polygons.back().stable_id);
    }

    if (cell_uses_high_order_geometry(mesh, cell.entity)) {
      const auto first_patch = topology.curved_patches.size();
      const auto added_patches = add_curved_patches_from_tessellated_cell(
          topology, mesh, cell.entity, map.embedded_geometry, cfg, cell.provenance, cell.global_id, tol);
      if (added_patches == 0u) {
        topology.diagnostics.push_back(
            "high-order cut cell has no tessellated curved interface patch descriptors");
      }
      for (std::size_t i = first_patch; i < topology.curved_patches.size(); ++i) {
        const auto& patch = topology.curved_patches[i];
        h = append_hash(h, patch.stable_id);
        h = append_hash(h, static_cast<std::uint64_t>(patch.parent_family));
        h = append_hash(h, static_cast<std::uint64_t>(patch.geometry_order));
        h = append_hash(h, patch.parametric_coordinates_valid ? 1u : 0u);
        h = append_hash(h, patch.exact_topology_available ? 1u : 0u);
        h = append_hash(h, patch.isoparametric_quadrature_available ? 1u : 0u);
        h = append_hash_real(h, patch.quadrature_measure);
        for (const auto& xi : patch.parent_parametric_coordinates) {
          h = append_hash_real(h, xi[0]);
          h = append_hash_real(h, xi[1]);
          h = append_hash_real(h, xi[2]);
        }
        for (const auto weight : patch.quadrature_weights) {
          h = append_hash_real(h, weight);
        }
        for (const auto id : patch.ordered_vertices) {
          h = append_hash(h, id);
        }
      }
    }

    const auto dofs = mesh.cell_geometry_dofs(cell.entity);
    std::vector<real_t> distances;
    distances.reserve(dofs.size());
    for (const auto dof : dofs) {
      distances.push_back(map.embedded_geometry.signed_distance(
          mesh.geometry_dof_coords(dof, cfg)));
    }

    for (const auto side : {CutTopologySide::Negative, CutTopologySide::Positive}) {
      CutSideRegion region;
      region.side = side;
      region.parent_cell = cell.entity;
      region.parent_cell_gid = cell.global_id;
      region.provenance = cell.provenance;
      region.stable_id = stable_entity_id(cell.provenance,
                                          cell.global_id,
                                          0u,
                                          static_cast<index_t>(side),
                                          real_t{0.0},
                                          404u);
      for (std::size_t i = 0; i < dofs.size(); ++i) {
        const bool on_side = side == CutTopologySide::Negative
                                 ? distances[i] <= tol
                                 : distances[i] >= -tol;
        if (on_side) {
          region.parent_geometry_dofs.push_back(dofs[i]);
        }
      }
      for (const auto& v : local_vertices) {
        region.cut_vertices.push_back(v.stable_id);
      }
      const auto measure = estimate_side_measure_for_cell(mesh,
                                                          cell.entity,
                                                          map.embedded_geometry,
                                                          cfg,
                                                          side,
                                                          tol);
      region.integration_family = mesh.cell_shape(cell.entity).family;
      region.parent_measure = measure.parent_measure;
      region.measure_estimate = measure.measure;
      region.volume_fraction_estimate = measure.fraction;
      region.centroid_estimate = measure.centroid;
      region.measure_from_linear_topology = measure.available;
      const auto closed_topology = build_side_closed_topology_for_cell(
          mesh,
          cell.entity,
          map.embedded_geometry,
          cfg,
          side,
          tol,
          cell.provenance,
          cell.global_id);
      region.integration_vertices = closed_topology.vertices;
      region.integration_subcells = closed_topology.subcells;
      region.integration_region_vertices = closed_topology.vertex_ids;
      region.integration_region_faces = closed_topology.faces;
      region.closed_integration_topology =
          !region.integration_subcells.empty() &&
          std::all_of(region.integration_subcells.begin(),
                      region.integration_subcells.end(),
                      [](const auto& subcell) {
                        return subcell.closed_topology;
                      });
      region.curved_isoparametric_topology =
          std::any_of(region.integration_subcells.begin(),
                      region.integration_subcells.end(),
                      [](const auto& subcell) {
                        return subcell.curved_isoparametric &&
                               subcell.measure_from_isoparametric_quadrature;
                      });
      if (region.integration_region_vertices.empty()) {
        for (const auto dof : region.parent_geometry_dofs) {
          region.integration_region_vertices.push_back(
              stable_parent_dof_region_id(region.provenance,
                                          cell.global_id,
                                          stable_geometry_dof_key(mesh, dof),
                                          side));
        }
        region.integration_region_vertices.insert(region.integration_region_vertices.end(),
                                                  region.cut_vertices.begin(),
                                                  region.cut_vertices.end());
      }
      if (region.integration_region_faces.empty()) {
        if (region.integration_family == CellFamily::Line) {
          if (region.integration_region_vertices.size() >= 2u) {
            region.integration_region_faces.push_back(region.integration_region_vertices);
          }
        } else if (region.integration_family == CellFamily::Triangle ||
                   region.integration_family == CellFamily::Quad ||
                   region.integration_family == CellFamily::Polygon) {
          if (region.integration_region_vertices.size() >= 3u) {
            region.integration_region_faces.push_back(region.integration_region_vertices);
          }
        } else if (region.cut_vertices.size() >= 3u) {
          region.integration_region_faces.push_back(region.cut_vertices);
        }
      }
      if (!measure.available) {
        topology.supported = false;
        topology.diagnostics.push_back(
            "linear side-measure reconstruction is unavailable for a cut side region");
      }
      topology.side_regions.push_back(std::move(region));
      h = append_hash(h, topology.side_regions.back().stable_id);
      h = append_hash_real(h, topology.side_regions.back().parent_measure);
      h = append_hash_real(h, topology.side_regions.back().measure_estimate);
      h = append_hash_real(h, topology.side_regions.back().volume_fraction_estimate);
      for (const auto id : topology.side_regions.back().integration_region_vertices) {
        h = append_hash(h, id);
      }
      for (const auto& face : topology.side_regions.back().integration_region_faces) {
        h = append_hash(h, static_cast<std::uint64_t>(face.size()));
        for (const auto id : face) {
          h = append_hash(h, id);
        }
      }
      for (const auto& vertex : topology.side_regions.back().integration_vertices) {
        h = append_hash(h, vertex.stable_id);
        h = append_hash_real(h, vertex.point[0]);
        h = append_hash_real(h, vertex.point[1]);
        h = append_hash_real(h, vertex.point[2]);
      }
      for (const auto& subcell : topology.side_regions.back().integration_subcells) {
        h = append_hash(h, subcell.stable_id);
        h = append_hash(h, static_cast<std::uint64_t>(subcell.family));
        h = append_hash_real(h, subcell.measure);
        h = append_hash_real(h, subcell.parent_parametric_measure);
        h = append_hash(h, subcell.curved_isoparametric ? 1u : 0u);
        h = append_hash(h, subcell.measure_from_isoparametric_quadrature ? 1u : 0u);
        for (const auto id : subcell.vertices) {
          h = append_hash(h, id);
        }
        for (const auto& xi : subcell.parent_parametric_vertices) {
          h = append_hash_real(h, xi[0]);
          h = append_hash_real(h, xi[1]);
          h = append_hash_real(h, xi[2]);
        }
        for (const auto& face : subcell.faces) {
          h = append_hash(h, static_cast<std::uint64_t>(face.size()));
          for (const auto id : face) {
            h = append_hash(h, id);
          }
        }
      }
    }
  }

  topology.topology_revision = h;
  return topology;
}

CutCurvedValidityDiagnostic diagnose_cut_topology_validity(
    const CutTopologyRecord& topology,
    bool high_order_parent_geometry,
    const CutCurvedValidityPolicy& policy) {
  CutCurvedValidityDiagnostic diagnostic;
  const real_t min_measure = std::max(policy.min_measure, real_t{0.0});
  const real_t min_fraction = std::max(policy.min_fraction, real_t{0.0});
  const real_t folding_tol = positive_tolerance(policy.folding_tolerance);
  const real_t closure_rel_tol = std::max(policy.closure_relative_tolerance, real_t{0.0});
  const real_t curved_closure_rel_tol = std::max(closure_rel_tol, real_t{1.0e-6});

  const auto fail = [&](const std::string& message) {
    diagnostic.ok = false;
    diagnostic.messages.push_back(message);
  };

  if (!topology.supported) {
    diagnostic.ok = false;
    diagnostic.messages.insert(diagnostic.messages.end(),
                               topology.diagnostics.begin(),
                               topology.diagnostics.end());
  }
  if (high_order_parent_geometry && topology.linearized_cut_mode) {
    diagnostic.requires_curved_geometry_support = true;
    diagnostic.messages.push_back(
        "high-order parent geometry is using recorded linearized-cut mode");
  }
  for (const auto& polygon : topology.interface_polygons) {
    const auto vertices = polygon_topology_vertices(topology, polygon);
    if (std::any_of(vertices.begin(), vertices.end(), [](const auto* vertex) {
          return vertex == nullptr;
        })) {
      diagnostic.has_degenerate_intersection = true;
      diagnostic.has_degenerate_polygon = true;
      fail("cut interface polygon references a missing cut vertex");
      break;
    }
    const auto points = polygon_points(vertices);
    const bool finite_polygon =
        finite_point(polygon.normal) &&
        std::all_of(vertices.begin(), vertices.end(), [](const auto* vertex) {
          return finite_point(vertex->point) && finite_point(vertex->normal);
        });
    if (!finite_polygon) {
      diagnostic.has_nonfinite_geometry = true;
      fail("cut interface polygon contains non-finite point or normal data");
      break;
    }
    if (projected_polygon_self_intersects(points, polygon.normal, folding_tol)) {
      diagnostic.has_folded_interface = true;
      fail("cut interface polygon is folded or self-intersecting in its projected tangent plane");
      break;
    }
    const real_t polygon_area = points.size() >= 3u ? MeshGeometry::polygon_area(points) : real_t{0.0};
    if (polygon_area > min_measure &&
        has_duplicate_or_short_polygon_edge(points, folding_tol)) {
      diagnostic.has_degenerate_intersection = true;
      diagnostic.has_degenerate_polygon = true;
      fail("cut interface polygon has duplicate vertices or a zero-length edge");
      break;
    }
    const auto polygon_normal = unit_or_default(polygon.normal);
    for (const auto* vertex : vertices) {
      if (norm(vertex->normal) > folding_tol &&
          dot(unit_or_default(vertex->normal), polygon_normal) < real_t{-0.25}) {
        diagnostic.has_folded_interface = true;
        fail("cut interface polygon has an inverted vertex normal");
        break;
      }
    }
    if (!diagnostic.ok) {
      break;
    }
    if (interface_measure_from_topology(topology, polygon) <= min_measure) {
      diagnostic.has_degenerate_polygon = true;
      fail("cut interface polygon is degenerate");
      break;
    }
  }

  for (const auto& patch : topology.curved_patches) {
    const bool finite_patch =
        std::isfinite(patch.max_parent_parametric_residual) &&
        std::all_of(patch.physical_points.begin(),
                    patch.physical_points.end(),
                    [](const auto& p) { return finite_point(p); }) &&
        std::all_of(patch.parent_parametric_coordinates.begin(),
                    patch.parent_parametric_coordinates.end(),
                    [](const auto& xi) { return finite_point(xi); });
    if (!finite_patch) {
      diagnostic.has_nonfinite_geometry = true;
      fail("curved cut patch contains non-finite physical or parametric coordinates");
      break;
    }
    if (patch.ordered_vertices.empty() ||
        patch.parent_parametric_coordinates.size() != patch.ordered_vertices.size() ||
        patch.physical_points.size() != patch.ordered_vertices.size()) {
      diagnostic.has_degenerate_intersection = true;
      fail("curved cut patch has inconsistent vertex, physical, or parametric coordinate counts");
      break;
    }
    if (!patch.parametric_coordinates_valid) {
      diagnostic.has_degenerate_intersection = true;
      fail("curved cut patch has invalid parent parametric coordinates");
      break;
    }
    if (patch.linearized_surrogate && !patch.exact_topology_available) {
      diagnostic.requires_curved_geometry_support = true;
      diagnostic.messages.push_back(
          "curved cut patch is recorded as a controlled linearized surrogate");
    }
    if (patch.isoparametric_quadrature_available) {
      if (patch.quadrature_points.size() != patch.quadrature_normals.size() ||
          patch.quadrature_points.size() != patch.quadrature_weights.size() ||
          !std::isfinite(patch.quadrature_measure) ||
          patch.quadrature_measure <= real_t{0.0}) {
        diagnostic.has_invalid_curved_measure = true;
        fail("curved cut patch has inconsistent isoparametric quadrature data");
        break;
      }
      real_t weight_sum = real_t{0.0};
      for (std::size_t i = 0; i < patch.quadrature_points.size(); ++i) {
        if (!finite_point(patch.quadrature_points[i]) ||
            !finite_point(patch.quadrature_normals[i]) ||
            !std::isfinite(patch.quadrature_weights[i]) ||
            patch.quadrature_weights[i] <= real_t{0.0}) {
          diagnostic.has_invalid_curved_measure = true;
          fail("curved cut patch contains invalid isoparametric quadrature samples");
          break;
        }
        weight_sum += patch.quadrature_weights[i];
      }
      if (!diagnostic.ok) {
        break;
      }
      const real_t qtol = std::max(min_measure, patch.quadrature_measure * closure_rel_tol);
      if (std::abs(weight_sum - patch.quadrature_measure) > qtol) {
        diagnostic.has_invalid_curved_measure = true;
        fail("curved cut patch quadrature weights do not sum to the patch measure");
        break;
      }
    }
  }
  if (!diagnostic.ok) {
    return diagnostic;
  }

  struct ParentMeasureBalance {
    real_t parent_measure{0.0};
    real_t side_measure_sum{0.0};
    bool has_negative{false};
    bool has_positive{false};
    bool has_curved_isoparametric{false};
  };
  std::map<std::pair<gid_t, index_t>, ParentMeasureBalance> parent_balances;

  for (const auto& region : topology.side_regions) {
    if (region.side != CutTopologySide::Negative &&
        region.side != CutTopologySide::Positive) {
      diagnostic.has_inconsistent_side_region = true;
      fail("cut side region has an invalid side");
      break;
    }
    const bool finite_region =
        std::isfinite(region.parent_measure) &&
        std::isfinite(region.measure_estimate) &&
        std::isfinite(region.volume_fraction_estimate) &&
        finite_point(region.centroid_estimate);
    if (!finite_region) {
      diagnostic.has_invalid_curved_measure = true;
      fail("cut side region contains non-finite measure, fraction, or centroid data");
      break;
    }
    if (region.parent_measure < real_t{0.0} ||
        region.measure_estimate < real_t{0.0} ||
        region.volume_fraction_estimate < real_t{0.0} ||
        region.volume_fraction_estimate > real_t{1.0}) {
      diagnostic.has_inconsistent_side_region = true;
      fail("cut side region has an invalid measure or volume-fraction estimate");
      break;
    }
    if (region.measure_estimate > min_measure &&
        region.parent_measure > min_measure &&
        region.volume_fraction_estimate < min_fraction) {
      diagnostic.has_curved_sliver = true;
      diagnostic.messages.push_back("cut side region is a sliver below the configured volume-fraction threshold");
      if (policy.reject_slivers) {
        diagnostic.ok = false;
        break;
      }
    }
    auto& balance = parent_balances[{region.parent_cell_gid, region.parent_cell}];
    balance.parent_measure = std::max(balance.parent_measure, region.parent_measure);
    balance.side_measure_sum += region.measure_estimate;
    balance.has_negative = balance.has_negative || region.side == CutTopologySide::Negative;
    balance.has_positive = balance.has_positive || region.side == CutTopologySide::Positive;
    balance.has_curved_isoparametric =
        balance.has_curved_isoparametric || region.curved_isoparametric_topology;

    for (const auto& vertex : region.integration_vertices) {
      if (!finite_point(vertex.point)) {
        diagnostic.has_nonfinite_geometry = true;
        fail("cut side region contains a non-finite integration vertex");
        break;
      }
      if (vertex.curved_isoparametric &&
          (!vertex.has_parent_parametric_coordinate ||
           !vertex.parent_parametric_coordinate_valid ||
           !finite_point(vertex.parent_parametric_coordinate))) {
        diagnostic.has_degenerate_intersection = true;
        fail("curved isoparametric integration vertex lacks a valid parent parametric coordinate");
        break;
      }
    }
    if (!diagnostic.ok) {
      break;
    }

    if (region.measure_estimate > min_measure) {
      if (!region.closed_integration_topology || region.integration_subcells.empty()) {
        diagnostic.has_open_subcell_topology = true;
        fail("positive-measure cut side region lacks closed integration subcell topology");
        break;
      }
      real_t subcell_measure = real_t{0.0};
      for (const auto& subcell : region.integration_subcells) {
        subcell_measure += subcell.measure;
        if (!subcell.closed_topology || subcell.vertices.empty() || subcell.faces.empty()) {
          diagnostic.has_open_subcell_topology = true;
          fail("cut integration subcell has open topology");
          break;
        }
        if (!std::isfinite(subcell.measure) ||
            subcell.measure < real_t{0.0} ||
            !finite_point(subcell.centroid)) {
          diagnostic.has_invalid_curved_measure = true;
          fail("cut integration subcell has invalid measure or centroid");
          break;
        }
        if (subcell.curved_isoparametric) {
          if (!subcell.measure_from_isoparametric_quadrature ||
              subcell.parent_parametric_vertices.size() != subcell.vertices.size() ||
              !std::isfinite(subcell.parent_parametric_measure) ||
              subcell.parent_parametric_measure <= real_t{0.0}) {
            diagnostic.has_invalid_curved_measure = true;
            fail("curved isoparametric subcell lacks valid parametric topology or quadrature measure");
            break;
          }
        }
      }
      if (!diagnostic.ok) {
        break;
      }
      real_t tol = std::max(min_measure, region.parent_measure * closure_rel_tol);
      if (region.curved_isoparametric_topology) {
        tol = std::max(tol, region.parent_measure * curved_closure_rel_tol);
      }
      if (std::abs(subcell_measure - region.measure_estimate) > tol) {
        diagnostic.has_inconsistent_side_region = true;
        fail("cut side-region subcell measures do not match the side measure");
        break;
      }
    }
  }
  if (diagnostic.ok) {
    for (const auto& entry : parent_balances) {
      const auto& balance = entry.second;
      if (balance.parent_measure <= min_measure) {
        continue;
      }
      if (!balance.has_negative || !balance.has_positive) {
        diagnostic.has_inconsistent_side_region = true;
        fail("cut topology is missing one side-region for a cut parent entity");
        break;
      }
      real_t tol = std::max(min_measure, balance.parent_measure * closure_rel_tol);
      if (balance.has_curved_isoparametric) {
        tol = std::max(tol, balance.parent_measure * curved_closure_rel_tol);
      }
      if (std::abs(balance.side_measure_sum - balance.parent_measure) > tol) {
        diagnostic.has_inconsistent_side_region = true;
        fail("cut side-region measures do not conserve the parent measure");
        break;
      }
    }
  }
  return diagnostic;
}

CutTopologyRecord project_cut_topology_to_embedded_geometry(
    const CutTopologyRecord& topology,
    const EmbeddedGeometryDescriptor& embedded_geometry,
    std::uint64_t new_topology_revision_salt) {
  CutTopologyRecord projected = topology;
  std::uint64_t h = kFnvOffset;
  h = append_hash(h, topology.topology_revision);
  h = append_hash(h, embedded_geometry.effective_revisions().revision_key());
  h = append_hash(h, new_topology_revision_salt);
  for (auto& vertex : projected.vertices) {
    vertex.point = embedded_geometry.closest_point(vertex.point);
    vertex.normal = embedded_geometry.outward_normal(vertex.point);
    h = append_hash(h, vertex.stable_id);
    h = append_hash_real(h, vertex.point[0]);
    h = append_hash_real(h, vertex.point[1]);
    h = append_hash_real(h, vertex.point[2]);
  }
  for (auto& polygon : projected.interface_polygons) {
    std::array<real_t, 3> normal{{0.0, 0.0, 0.0}};
    std::size_t count = 0;
    for (const auto id : polygon.ordered_vertices) {
      const auto it = std::find_if(projected.vertices.begin(), projected.vertices.end(), [&](const auto& v) {
        return v.stable_id == id;
      });
      if (it != projected.vertices.end()) {
        normal = add(normal, it->normal);
        ++count;
      }
    }
    if (count > 0u) {
      polygon.normal = unit_or_default(normal);
    }
    h = append_hash(h, polygon.stable_id);
  }
  projected.topology_revision = h;
  projected.diagnostics.push_back("cut topology projected to accepted embedded geometry descriptor");
  return projected;
}

CutDistributedExchangePacket make_distributed_cut_exchange_packet(
    const CutClassificationMap& map,
    const CutTopologyRecord& topology) {
  CutDistributedExchangePacket packet;
  packet.revision_key = append_hash(map.revision_key(), topology.topology_revision);
  auto add_record = [&](std::uint64_t stable_id,
                        std::uint64_t cut_topology_id,
                        CutTopologyEntityKind kind,
                        CutTopologySide side,
                        index_t parent_entity,
                        gid_t parent_gid,
                        const EmbeddedRegionProvenance& provenance) -> CutDistributedEntityRecord& {
    CutDistributedEntityRecord record;
    record.stable_id = stable_id;
    record.cut_topology_id = cut_topology_id;
    record.kind = kind;
    record.side = side;
    record.parent_entity = parent_entity;
    record.parent_gid = parent_gid;
    record.owner_rank = owner_for_cell(map, parent_entity);
    record.provenance_id = provenance_id(provenance);
    packet.entities.push_back(std::move(record));
    return packet.entities.back();
  };

  for (const auto& vertex : topology.vertices) {
    auto& record = add_record(vertex.stable_id,
                              topology.topology_revision,
                              CutTopologyEntityKind::CutVertex,
                              CutTopologySide::Interface,
                              vertex.parent_cell,
                              vertex.parent_cell_gid,
                              vertex.provenance);
    record.point = vertex.point;
    record.normal = vertex.normal;
  }
  for (const auto& edge : topology.edges) {
    auto& record = add_record(edge.stable_id,
                              topology.topology_revision,
                              CutTopologyEntityKind::CutEdge,
                              CutTopologySide::Interface,
                              edge.parent_cell,
                              edge.parent_cell_gid,
                              edge.provenance);
    record.vertex_ids = {edge.vertex_a, edge.vertex_b};
  }
  for (const auto& polygon : topology.interface_polygons) {
    auto& record = add_record(polygon.stable_id,
                              topology.topology_revision,
                              CutTopologyEntityKind::InterfacePolygon,
                              CutTopologySide::Interface,
                              polygon.parent_cell,
                              polygon.parent_cell_gid,
                              polygon.provenance);
    record.normal = polygon.normal;
    record.vertex_ids = polygon.ordered_vertices;
    record.measure = interface_measure_from_topology(topology, polygon);
    record.parent_measure = record.measure;
    if (!polygon.ordered_vertices.empty()) {
      record.face_ids.push_back(polygon.ordered_vertices);
    }
  }
  for (const auto& region : topology.side_regions) {
    auto& record = add_record(region.stable_id,
                              topology.topology_revision,
                              CutTopologyEntityKind::SideRegion,
                              region.side,
                              region.parent_cell,
                              region.parent_cell_gid,
                              region.provenance);
    record.centroid = region.centroid_estimate;
    record.integration_family = region.integration_family;
    record.parent_measure = region.parent_measure;
    record.measure = region.measure_estimate;
    record.volume_fraction = region.volume_fraction_estimate;
    record.closed_topology = region.closed_integration_topology;
    record.vertex_ids = region.integration_region_vertices;
    record.face_ids = region.integration_region_faces;
  }
  return deduplicate_cut_exchange_packet(std::move(packet));
}

CutDistributedExchangePacket deduplicate_cut_exchange_packet(
    CutDistributedExchangePacket packet) {
  std::sort(packet.entities.begin(), packet.entities.end(), [](const auto& a, const auto& b) {
    if (a.stable_id != b.stable_id) return a.stable_id < b.stable_id;
    if (a.parent_gid != b.parent_gid) return a.parent_gid < b.parent_gid;
    if (a.kind != b.kind) return static_cast<int>(a.kind) < static_cast<int>(b.kind);
    return static_cast<int>(a.side) < static_cast<int>(b.side);
  });
  packet.entities.erase(std::unique(packet.entities.begin(),
                                    packet.entities.end(),
                                    [](const auto& a, const auto& b) {
                                      return a.stable_id == b.stable_id &&
                                             a.parent_gid == b.parent_gid &&
                                             a.kind == b.kind &&
                                             a.side == b.side;
                                    }),
                        packet.entities.end());
  std::uint64_t h = append_hash(packet.revision_key, static_cast<std::uint64_t>(packet.entities.size()));
  for (const auto& entity : packet.entities) {
    h = append_hash_distributed_record(h, entity);
  }
  packet.revision_key = h;
  return packet;
}

CutDistributedState build_distributed_cut_state(
    const DistributedMesh& mesh,
    const CutClassificationMap& map,
    const CutTopologyRecord& topology) {
  CutDistributedState state;
  state.local_packet = make_distributed_cut_exchange_packet(map, topology);
  const auto exchange = gather_cut_exchange_packets(mesh, state.local_packet);
  state.neighbor_sparse_exchange = exchange.neighbor_sparse_exchange;
  state.communication_neighbors = exchange.communication_neighbors;
  state.received_neighbor_ranks = exchange.received_neighbor_ranks;

  CutDistributedExchangePacket exchanged;
  exchanged.revision_key = kFnvOffset;
  std::map<std::string, rank_t> owner_by_entity;
  for (const auto& packet : exchange.packets) {
    exchanged.revision_key = append_hash(exchanged.revision_key, packet.revision_key);
    exchanged.diagnostics.insert(exchanged.diagnostics.end(),
                                 packet.diagnostics.begin(),
                                 packet.diagnostics.end());
    for (const auto& entity : packet.entities) {
      if (entity.owner_rank < 0 || entity.owner_rank >= mesh.world_size()) {
        state.diagnostics.push_back("distributed cut entity has an owner rank outside the communicator");
      }
      const auto key = distributed_entity_key(entity);
      const auto [it, inserted] = owner_by_entity.emplace(key, entity.owner_rank);
      if (!inserted && it->second != entity.owner_rank) {
        state.diagnostics.push_back("distributed cut entity has conflicting owner ranks");
      }
      exchanged.entities.push_back(entity);
    }
  }
  state.exchanged_packet = deduplicate_cut_exchange_packet(std::move(exchanged));

  std::map<std::string, bool> exchanged_keys;
  for (const auto& entity : state.exchanged_packet.entities) {
    exchanged_keys[distributed_entity_key(entity)] = true;
    if (entity.owner_rank == mesh.rank()) {
      state.owned_records.push_back(entity);
    } else {
      state.imported_records.push_back(entity);
      if (entity.parent_gid != INVALID_GID) {
        const index_t local_cell = mesh.global_to_local_cell(entity.parent_gid);
        if (local_cell != INVALID_INDEX &&
            !mesh.is_owned_cell(local_cell)) {
          state.ghost_records.push_back(entity);
        }
      }
    }
  }

  for (const auto& entity : state.local_packet.entities) {
    if (entity.owner_rank == mesh.rank() || entity.parent_gid == INVALID_GID) {
      continue;
    }
    const index_t local_cell = mesh.global_to_local_cell(entity.parent_gid);
    if (local_cell == INVALID_INDEX || mesh.is_owned_cell(local_cell)) {
      continue;
    }
    if (exchanged_keys.find(distributed_entity_key(entity)) == exchanged_keys.end()) {
      state.diagnostics.push_back("local ghost cut entity is missing from exchanged owner payload");
    }
  }

  state.revision = CutDistributedRevisionSnapshot::capture(mesh,
                                                           map,
                                                           topology,
                                                           state.local_packet,
                                                           state.exchanged_packet);
  if (!state.local_packet.entities.empty() &&
      state.exchanged_packet.entities.empty()) {
    state.diagnostics.push_back("distributed cut state contains no exchanged cut entities");
  }
  return state;
}

CutDistributedStateDiagnostic diagnose_distributed_cut_state(
    const CutDistributedState& state) {
  CutDistributedStateDiagnostic diagnostic;
  if (!state.diagnostics.empty()) {
    diagnostic.ok = false;
    diagnostic.messages.insert(diagnostic.messages.end(),
                               state.diagnostics.begin(),
                               state.diagnostics.end());
  }
  if (state.revision.distributed_revision != state.revision.revision_key()) {
    diagnostic.ok = false;
    diagnostic.stale_revision = true;
    diagnostic.messages.push_back("distributed cut-state revision key is stale");
  }
  if (!state.local_packet.entities.empty() &&
      state.exchanged_packet.entities.empty()) {
    diagnostic.ok = false;
    diagnostic.missing_owner = true;
    diagnostic.messages.push_back("distributed cut-state exchange has no owner records");
  }

  std::map<std::string, rank_t> owner_by_entity;
  for (const auto& entity : state.exchanged_packet.entities) {
    if (entity.owner_rank < 0 ||
        entity.owner_rank >= state.revision.world_size) {
      diagnostic.ok = false;
      diagnostic.missing_owner = true;
      diagnostic.messages.push_back("cut entity owner rank is outside the recorded communicator");
    }
    const auto key = distributed_entity_key(entity);
    const auto [it, inserted] = owner_by_entity.emplace(key, entity.owner_rank);
    if (!inserted && it->second != entity.owner_rank) {
      diagnostic.ok = false;
      diagnostic.duplicate_owner = true;
      diagnostic.messages.push_back("cut entity has duplicate conflicting owners");
    }
    if (entity.kind == CutTopologyEntityKind::SideRegion &&
        entity.measure > real_t{0.0} &&
        (!entity.closed_topology ||
         entity.vertex_ids.empty() ||
         entity.face_ids.empty())) {
      diagnostic.ok = false;
      diagnostic.missing_ghost_payload = true;
      diagnostic.messages.push_back("positive-measure exchanged cut side region lacks closed-topology payload");
    }
  }
  return diagnostic;
}

std::vector<CutSupportMatrixEntry> cut_support_matrix() {
  std::vector<CutSupportMatrixEntry> entries;
  const auto add = [&](CellFamily family,
                       int order,
                       EmbeddedGeometryKind kind,
                       bool distributed,
                       std::string cut_mode,
                       std::string policy,
                       std::string conditioning_policy,
                       std::string fe_execution_path,
                       CutSupportStatus status,
                       std::string qualification) {
    CutSupportMatrixEntry entry;
    entry.parent_family = family;
    entry.geometry_order = order;
    entry.embedded_kind = kind;
    entry.distributed = distributed;
    entry.cut_mode = std::move(cut_mode);
    entry.quadrature_policy = std::move(policy);
    entry.conditioning_policy = std::move(conditioning_policy);
    entry.fe_execution_path = std::move(fe_execution_path);
    entry.status = status;
    entry.qualification = std::move(qualification);
    entries.push_back(std::move(entry));
  };

  const std::array<const char*, 6> fe_paths{
      "standard-assembly",
      "matrix-free",
      "forms-interpreter",
      "ad",
      "symbolic-tangent",
      "jit"};

  for (const auto family : {CellFamily::Line, CellFamily::Triangle, CellFamily::Quad,
                            CellFamily::Tetra, CellFamily::Hex,
                            CellFamily::Wedge, CellFamily::Pyramid,
                            CellFamily::Polygon, CellFamily::Polyhedron}) {
    for (const auto kind : {EmbeddedGeometryKind::Plane, EmbeddedGeometryKind::Sphere,
                            EmbeddedGeometryKind::TriangulatedSurface,
                            EmbeddedGeometryKind::LevelSetField,
                            EmbeddedGeometryKind::BooleanComposite}) {
      for (const auto* fe_path : fe_paths) {
        add(family,
            1,
            kind,
            false,
            "linearized-cut",
            "topology-subdivision",
            "geometric-conditioning-hooks",
            fe_path,
            CutSupportStatus::ImplementedUnqualified,
            "classification, deterministic topology IDs, linear side-region measures, restart summaries, FE quadrature metadata, conditioning hooks, and execution-path metadata are implemented; production validation of full subcell topology remains required");
        add(family,
            1,
            kind,
            true,
            "linearized-cut",
            "topology-subdivision",
            "geometric-conditioning-hooks",
            fe_path,
            CutSupportStatus::ImplementedUnqualified,
            "neighbor-sparse distributed owner/ghost exchange plus production migration, deterministic rebalance, and block/METIS/ParMETIS partition-method qualification are implemented");
      }
    }
  }
  for (const auto family : {CellFamily::Line, CellFamily::Triangle, CellFamily::Quad,
                            CellFamily::Tetra, CellFamily::Hex,
                            CellFamily::Wedge, CellFamily::Pyramid}) {
    for (const auto kind : {EmbeddedGeometryKind::Plane, EmbeddedGeometryKind::Sphere,
                            EmbeddedGeometryKind::TriangulatedSurface,
                            EmbeddedGeometryKind::LevelSetField,
                            EmbeddedGeometryKind::BooleanComposite}) {
      for (const auto* fe_path : fe_paths) {
        add(family,
            2,
            kind,
            false,
            "linearized-cut",
            "topology-subdivision",
            "geometric-conditioning-hooks",
            fe_path,
            CutSupportStatus::ImplementedUnqualified,
            "controlled high-order linearized cut topology uses CurvilinearEval/Tessellator with geometry-DOF provenance and execution-path metadata; exact curved arrangement topology remains future work");
      }
    }
  }
  for (const auto family : {CellFamily::Line, CellFamily::Triangle, CellFamily::Quad,
                            CellFamily::Tetra, CellFamily::Hex}) {
    for (const auto* fe_path : fe_paths) {
      add(family,
          2,
          EmbeddedGeometryKind::Plane,
          false,
          "curved-isoparametric-cut",
          "curved-topology-subdivision",
          "geometric-conditioning-hooks",
          fe_path,
          CutSupportStatus::ImplementedUnqualified,
          "quadratic starter path records parent-parametric cut topology, isoparametric topology-derived quadrature, and execution-path metadata for line, triangle, quad, tetra, and hex parents; full arbitrary-order arrangement qualification remains future work");
    }
  }
  for (const auto* fe_path : fe_paths) {
    add(CellFamily::Line,
        -1,
        EmbeddedGeometryKind::Plane,
        false,
        "curved-isoparametric-cut",
        "true-curved-arrangement",
        "geometric-conditioning-hooks",
        fe_path,
        CutSupportStatus::ImplementedUnqualified,
        "arbitrary-order high-order line/plane cuts use root-bracketed reference intervals, exact/non-surrogate curved patch metadata, isoparametric side measures, and execution-path metadata");
    for (const auto family : {CellFamily::Triangle, CellFamily::Quad}) {
      add(family,
          -1,
          EmbeddedGeometryKind::Plane,
          false,
          "curved-isoparametric-cut",
          "true-curved-arrangement",
          "geometric-conditioning-hooks",
          fe_path,
          CutSupportStatus::ImplementedUnqualified,
          "graph-compatible arbitrary-order high-order face/plane cuts use bracketed reference-space contour roots, exact/non-surrogate curved patch metadata, isoparametric side measures, and execution-path metadata");
    }
    add(CellFamily::Tetra,
        -1,
        EmbeddedGeometryKind::Plane,
        false,
        "curved-isoparametric-cut",
        "true-curved-arrangement",
        "geometric-conditioning-hooks",
        fe_path,
        CutSupportStatus::ImplementedUnqualified,
        "graph-compatible arbitrary-order high-order tetra/plane cuts use reference-space root surfaces, exact/non-surrogate curved patch metadata, analytic mapped-Jacobian side and interface quadrature, and execution-path metadata");
    add(CellFamily::Hex,
        -1,
        EmbeddedGeometryKind::Plane,
        false,
        "curved-isoparametric-cut",
        "true-curved-arrangement",
        "geometric-conditioning-hooks",
        fe_path,
        CutSupportStatus::ImplementedUnqualified,
        "graph-compatible arbitrary-order high-order hex/plane cuts use tensor-product reference-space root surfaces, exact/non-surrogate curved patch metadata, analytic mapped-Jacobian side and interface quadrature, and execution-path metadata");
    add(CellFamily::Wedge,
        -1,
        EmbeddedGeometryKind::Plane,
        false,
        "curved-isoparametric-cut",
        "true-curved-arrangement",
        "geometric-conditioning-hooks",
        fe_path,
        CutSupportStatus::ImplementedUnqualified,
        "graph-compatible arbitrary-order high-order wedge/plane cuts use triangular-base reference-space root surfaces, exact/non-surrogate curved patch metadata, analytic mapped-Jacobian side and interface quadrature, and execution-path metadata");
    add(CellFamily::Pyramid,
        -1,
        EmbeddedGeometryKind::Plane,
        false,
        "curved-isoparametric-cut",
        "true-curved-arrangement",
        "geometric-conditioning-hooks",
        fe_path,
        CutSupportStatus::ImplementedUnqualified,
        "graph-compatible arbitrary-order high-order pyramid/plane cuts use shrinking-column reference-space root surfaces, exact/non-surrogate curved patch metadata, analytic mapped-Jacobian side and interface quadrature, and execution-path metadata");
    for (const auto family : {CellFamily::Triangle,
                              CellFamily::Quad,
                              CellFamily::Hex,
                              CellFamily::Wedge,
                              CellFamily::Pyramid}) {
      add(family,
          -1,
          EmbeddedGeometryKind::Plane,
          false,
          "curved-isoparametric-cut",
          "true-curved-subdivision-arrangement",
          "geometric-conditioning-hooks",
          fe_path,
          CutSupportStatus::ImplementedUnqualified,
          "non-graph high-order triangle/quad/hex/wedge/pyramid plane cuts use bounded isoparametric subcell subdivision with exact/non-surrogate patch metadata, closed curved side subcells, deterministic topology IDs, and execution-path metadata; full non-plane/general arrangement validation remains future work");
    }
  }
  for (const auto family : {CellFamily::Line, CellFamily::Triangle, CellFamily::Quad,
                            CellFamily::Tetra, CellFamily::Hex}) {
    for (const auto* fe_path : fe_paths) {
      add(family,
          2,
          EmbeddedGeometryKind::Sphere,
          false,
          "curved-isoparametric-cut",
          "moment-fitted",
          "geometric-conditioning-hooks",
          fe_path,
          CutSupportStatus::Unsupported,
          "curved cut topology and high-order moment-fitted validation remain future work");
    }
  }
  return entries;
}

CutSupportMatrixEntry evaluate_cut_support(
    CellFamily parent_family,
    int geometry_order,
    EmbeddedGeometryKind embedded_kind,
    bool distributed,
    const std::string& cut_mode,
    const std::string& quadrature_policy) {
  const auto entries = cut_support_matrix();
  for (const auto& entry : entries) {
    const bool order_matches =
        entry.geometry_order == geometry_order ||
        (entry.geometry_order < 0 && geometry_order > 1);
    if (entry.parent_family == parent_family &&
        order_matches &&
        entry.embedded_kind == embedded_kind &&
        entry.distributed == distributed &&
        entry.cut_mode == cut_mode &&
        entry.quadrature_policy == quadrature_policy) {
      return entry;
    }
  }
  CutSupportMatrixEntry unsupported;
  unsupported.parent_family = parent_family;
  unsupported.geometry_order = geometry_order;
  unsupported.embedded_kind = embedded_kind;
  unsupported.distributed = distributed;
  unsupported.cut_mode = cut_mode;
  unsupported.quadrature_policy = quadrature_policy;
  unsupported.status = CutSupportStatus::Unsupported;
  unsupported.qualification = "combination is not advertised by the Phase 26 support matrix";
  return unsupported;
}

CutSupportMatrixEntry evaluate_cut_support(
    CellFamily parent_family,
    int geometry_order,
    EmbeddedGeometryKind embedded_kind,
    bool distributed,
    const std::string& cut_mode,
    const std::string& quadrature_policy,
    const std::string& conditioning_policy,
    const std::string& fe_execution_path) {
  const auto entries = cut_support_matrix();
  for (const auto& entry : entries) {
    const bool order_matches =
        entry.geometry_order == geometry_order ||
        (entry.geometry_order < 0 && geometry_order > 1);
    if (entry.parent_family == parent_family &&
        order_matches &&
        entry.embedded_kind == embedded_kind &&
        entry.distributed == distributed &&
        entry.cut_mode == cut_mode &&
        entry.quadrature_policy == quadrature_policy &&
        entry.conditioning_policy == conditioning_policy &&
        entry.fe_execution_path == fe_execution_path) {
      return entry;
    }
  }
  CutSupportMatrixEntry unsupported;
  unsupported.parent_family = parent_family;
  unsupported.geometry_order = geometry_order;
  unsupported.embedded_kind = embedded_kind;
  unsupported.distributed = distributed;
  unsupported.cut_mode = cut_mode;
  unsupported.quadrature_policy = quadrature_policy;
  unsupported.conditioning_policy = conditioning_policy;
  unsupported.fe_execution_path = fe_execution_path;
  unsupported.status = CutSupportStatus::Unsupported;
  unsupported.qualification =
      "combination is not advertised by the Phase 26 support matrix";
  return unsupported;
}

CutClassificationRestartRecord make_cut_classification_restart_record(
    const CutClassificationMap& map) {
  CutClassificationRestartRecord record;
  record.name = map.name;
  record.provenance = map.embedded_geometry.provenance;
  record.embedded_kind = map.embedded_geometry.kind;
  record.is_composed_region = map.embedded_geometry.kind == EmbeddedGeometryKind::BooleanComposite;
  record.composition_operation = map.embedded_geometry.boolean_operation;
  collect_composition_child_records(map.embedded_geometry, 0u, record.composition_children);
  record.revision_key = map.revision_key();
  const auto revisions = map.embedded_geometry.effective_revisions();
  record.embedded_geometry_epoch = revisions.geometry_epoch;
  record.embedded_field_layout_revision = revisions.field_layout_revision;
  record.embedded_field_value_revision = revisions.field_value_revision;
  record.embedded_source_surface_revision = revisions.source_surface_revision;
  record.embedded_provenance_revision = revisions.provenance_revision;
  record.embedded_constraint_epoch = max_constraint_epoch(map.kinematic_constraints);
  record.fe_layout_revision = map.options.fe_layout_revision;
  record.predicate_policy_key = map.options.predicate_policy.revision_key();
  std::uint64_t topology_hash = kFnvOffset;
  for (const auto& cell : map.cells) {
    topology_hash = append_hash(topology_hash, cell.cut_topology_id);
  }
  for (const auto& face : map.faces) {
    topology_hash = append_hash(topology_hash, face.cut_topology_id);
  }
  for (const auto& edge : map.edges) {
    topology_hash = append_hash(topology_hash, edge.cut_topology_id);
  }
  record.cut_topology_revision = topology_hash;
  record.cut_cell_count = static_cast<std::size_t>(std::count_if(
      map.cells.begin(), map.cells.end(), [](const auto& r) { return r.classification == CutClassification::Cut; }));
  record.cut_face_count = static_cast<std::size_t>(std::count_if(
      map.faces.begin(), map.faces.end(), [](const auto& r) { return r.classification == CutClassification::Cut; }));
  record.cut_edge_count = static_cast<std::size_t>(std::count_if(
      map.edges.begin(), map.edges.end(), [](const auto& r) { return r.classification == CutClassification::Cut; }));
  return record;
}

CutClassificationRestartRecord make_cut_classification_restart_record(
    const CutClassificationMap& map,
    const CutTopologyRecord& topology) {
  auto record = make_cut_classification_restart_record(map);
  record.cut_topology_revision = topology.topology_revision != 0u
                                     ? topology.topology_revision
                                     : record.cut_topology_revision;
  record.side_regions.reserve(topology.side_regions.size());
  for (const auto& region : topology.side_regions) {
    CutSideRegionRestartRecord side_record;
    side_record.stable_id = region.stable_id;
    side_record.side = region.side;
    side_record.parent_cell = region.parent_cell;
    side_record.parent_cell_gid = region.parent_cell_gid;
    side_record.measure_estimate = region.measure_estimate;
    side_record.volume_fraction_estimate = region.volume_fraction_estimate;
    side_record.provenance = region.provenance;
    record.side_regions.push_back(side_record);
  }
  return record;
}

} // namespace search
} // namespace svmp
