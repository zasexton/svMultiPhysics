/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// The functions here reproduce the subroutines defined in svFSILS/INCOMMU.f.

// To syncronize the data on the boundaries between the processors.
//
// This a both way communication with three main part:
// 1 - rTmp {in master} = R          {from slave}
// 2 - R    {in master} = R + rTmp   {both from master}
// 3 - rTmp {in master} = R          {from master}
// 4 - R    {in slave}  = rTmp       {from master}

#include "fsils.hpp"
#include "CmMod.h"
#include "Array3.h"

#include "fsils_std.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace fe_fsi_linear_solver {

namespace {

constexpr int kFsilsScalarCommTag = 1101;
constexpr int kFsilsVectorCommTagBase = 1200;

[[nodiscard]] inline int fsils_vector_comm_tag(int dof) noexcept
{
  return kFsilsVectorCommTagBase + ((dof > 0) ? dof : 0);
}

[[nodiscard]] bool has_owned_halo_plan(const FSILS_lhsType& lhs) noexcept
{
  return lhs.owned_row_operator &&
         lhs.commu.nTasks > 1 &&
         lhs.owned_halo_send_nodes.size() == lhs.owned_halo_neighbor_ranks.size() &&
         lhs.owned_halo_recv_nodes.size() == lhs.owned_halo_neighbor_ranks.size() &&
         (lhs.nNo <= lhs.mynNo || !lhs.owned_halo_neighbor_ranks.empty());
}

void require_owned_halo_plan(const FSILS_lhsType& lhs, const char* operation)
{
  if (!lhs.owned_row_operator) {
    throw std::runtime_error(std::string(operation) + " requires an explicit owned-row layout");
  }
  if (!has_owned_halo_plan(lhs)) {
    throw std::runtime_error(std::string(operation) + " requires an explicit owned-row halo plan");
  }
}

void fsils_syncs_owned_halo(const FSILS_lhsType& lhs, Vector<double>& R)
{
  const int n_neighbors = static_cast<int>(lhs.owned_halo_neighbor_ranks.size());
  if (n_neighbors == 0) {
    return;
  }

  for (fsils_int k = lhs.mynNo; k < lhs.nNo; ++k) {
    R(k) = 0.0;
  }

  std::size_t send_total = 0;
  std::size_t recv_total = 0;
  for (int i = 0; i < n_neighbors; ++i) {
    send_total += lhs.owned_halo_send_nodes[static_cast<std::size_t>(i)].size();
    recv_total += lhs.owned_halo_recv_nodes[static_cast<std::size_t>(i)].size();
  }

  auto& send_buffer = lhs.owned_halo_send_buffer;
  auto& recv_buffer = lhs.owned_halo_recv_buffer;
  send_buffer.resize(send_total);
  recv_buffer.resize(recv_total);

  std::vector<MPI_Request> recv_req;
  std::vector<MPI_Request> send_req;
  recv_req.reserve(static_cast<std::size_t>(n_neighbors));
  send_req.reserve(static_cast<std::size_t>(n_neighbors));

  const int mpi_tag = kFsilsScalarCommTag;
  std::size_t send_offset = 0;
  std::size_t recv_offset = 0;
  for (int i = 0; i < n_neighbors; ++i) {
    const auto& send_nodes = lhs.owned_halo_send_nodes[static_cast<std::size_t>(i)];
    const auto& recv_nodes = lhs.owned_halo_recv_nodes[static_cast<std::size_t>(i)];
    const int peer = lhs.owned_halo_neighbor_ranks[static_cast<std::size_t>(i)];

    for (std::size_t j = 0; j < send_nodes.size(); ++j) {
      send_buffer[send_offset + j] = R(send_nodes[j]);
    }

    if (!recv_nodes.empty()) {
      recv_req.push_back(MPI_REQUEST_NULL);
      MPI_Irecv(recv_buffer.data() + recv_offset,
                static_cast<int>(recv_nodes.size()),
                mpreal,
                peer,
                mpi_tag,
                lhs.commu.comm,
                &recv_req.back());
    }
    if (!send_nodes.empty()) {
      send_req.push_back(MPI_REQUEST_NULL);
      MPI_Isend(send_buffer.data() + send_offset,
                static_cast<int>(send_nodes.size()),
                mpreal,
                peer,
                mpi_tag,
                lhs.commu.comm,
                &send_req.back());
    }

    send_offset += send_nodes.size();
    recv_offset += recv_nodes.size();
  }

  if (!recv_req.empty()) {
    MPI_Waitall(static_cast<int>(recv_req.size()), recv_req.data(), MPI_STATUSES_IGNORE);
  }

  recv_offset = 0;
  for (int i = 0; i < n_neighbors; ++i) {
    const auto& recv_nodes = lhs.owned_halo_recv_nodes[static_cast<std::size_t>(i)];
    for (std::size_t j = 0; j < recv_nodes.size(); ++j) {
      R(recv_nodes[j]) = recv_buffer[recv_offset + j];
    }
    recv_offset += recv_nodes.size();
  }

  if (!send_req.empty()) {
    MPI_Waitall(static_cast<int>(send_req.size()), send_req.data(), MPI_STATUSES_IGNORE);
  }
}

void fsils_syncv_owned_halo(const FSILS_lhsType& lhs, int dof, Array<double>& R)
{
  const int n_neighbors = static_cast<int>(lhs.owned_halo_neighbor_ranks.size());
  if (n_neighbors == 0) {
    return;
  }

  for (fsils_int k = lhs.mynNo; k < lhs.nNo; ++k) {
    for (int l = 0; l < dof; ++l) {
      R(l, k) = 0.0;
    }
  }

  std::size_t send_nodes_total = 0;
  std::size_t recv_nodes_total = 0;
  for (int i = 0; i < n_neighbors; ++i) {
    send_nodes_total += lhs.owned_halo_send_nodes[static_cast<std::size_t>(i)].size();
    recv_nodes_total += lhs.owned_halo_recv_nodes[static_cast<std::size_t>(i)].size();
  }

  auto& send_buffer = lhs.owned_halo_send_buffer;
  auto& recv_buffer = lhs.owned_halo_recv_buffer;
  send_buffer.resize(static_cast<std::size_t>(std::max(dof, 0)) * send_nodes_total);
  recv_buffer.resize(static_cast<std::size_t>(std::max(dof, 0)) * recv_nodes_total);

  std::vector<MPI_Request> recv_req;
  std::vector<MPI_Request> send_req;
  recv_req.reserve(static_cast<std::size_t>(n_neighbors));
  send_req.reserve(static_cast<std::size_t>(n_neighbors));

  const int mpi_tag = fsils_vector_comm_tag(dof);
  std::size_t send_offset = 0;
  std::size_t recv_offset = 0;
  for (int i = 0; i < n_neighbors; ++i) {
    const auto& send_nodes = lhs.owned_halo_send_nodes[static_cast<std::size_t>(i)];
    const auto& recv_nodes = lhs.owned_halo_recv_nodes[static_cast<std::size_t>(i)];
    const int peer = lhs.owned_halo_neighbor_ranks[static_cast<std::size_t>(i)];

    for (std::size_t j = 0; j < send_nodes.size(); ++j) {
      const auto node = send_nodes[j];
      for (int l = 0; l < dof; ++l) {
        send_buffer[send_offset + j * static_cast<std::size_t>(dof) + static_cast<std::size_t>(l)] =
            R(l, node);
      }
    }

    if (!recv_nodes.empty()) {
      recv_req.push_back(MPI_REQUEST_NULL);
      MPI_Irecv(recv_buffer.data() + recv_offset,
                static_cast<int>(recv_nodes.size()) * dof,
                mpreal,
                peer,
                mpi_tag,
                lhs.commu.comm,
                &recv_req.back());
    }
    if (!send_nodes.empty()) {
      send_req.push_back(MPI_REQUEST_NULL);
      MPI_Isend(send_buffer.data() + send_offset,
                static_cast<int>(send_nodes.size()) * dof,
                mpreal,
                peer,
                mpi_tag,
                lhs.commu.comm,
                &send_req.back());
    }

    send_offset += send_nodes.size() * static_cast<std::size_t>(dof);
    recv_offset += recv_nodes.size() * static_cast<std::size_t>(dof);
  }

  if (!recv_req.empty()) {
    MPI_Waitall(static_cast<int>(recv_req.size()), recv_req.data(), MPI_STATUSES_IGNORE);
  }

  recv_offset = 0;
  for (int i = 0; i < n_neighbors; ++i) {
    const auto& recv_nodes = lhs.owned_halo_recv_nodes[static_cast<std::size_t>(i)];
    for (std::size_t j = 0; j < recv_nodes.size(); ++j) {
      const auto node = recv_nodes[j];
      for (int l = 0; l < dof; ++l) {
        R(l, node) =
            recv_buffer[recv_offset + j * static_cast<std::size_t>(dof) + static_cast<std::size_t>(l)];
      }
    }
    recv_offset += recv_nodes.size() * static_cast<std::size_t>(dof);
  }

  if (!send_req.empty()) {
    MPI_Waitall(static_cast<int>(send_req.size()), send_req.data(), MPI_STATUSES_IGNORE);
  }
}

void fsils_reverse_scatterv_owned_halo(const FSILS_lhsType& lhs, int dof, Array<double>& R)
{
  const int n_neighbors = static_cast<int>(lhs.owned_halo_neighbor_ranks.size());
  if (n_neighbors == 0) {
    return;
  }

  std::size_t send_nodes_total = 0;
  std::size_t recv_nodes_total = 0;
  for (int i = 0; i < n_neighbors; ++i) {
    // Reverse of owner->ghost sync: local ghosts are sent back to the owning
    // peer, and owned rows receive raw contributions from that peer's ghosts.
    send_nodes_total += lhs.owned_halo_recv_nodes[static_cast<std::size_t>(i)].size();
    recv_nodes_total += lhs.owned_halo_send_nodes[static_cast<std::size_t>(i)].size();
  }

  auto& send_buffer = lhs.owned_halo_send_buffer;
  auto& recv_buffer = lhs.owned_halo_recv_buffer;
  send_buffer.resize(static_cast<std::size_t>(std::max(dof, 0)) * send_nodes_total);
  recv_buffer.resize(static_cast<std::size_t>(std::max(dof, 0)) * recv_nodes_total);

  std::vector<MPI_Request> recv_req;
  std::vector<MPI_Request> send_req;
  recv_req.reserve(static_cast<std::size_t>(n_neighbors));
  send_req.reserve(static_cast<std::size_t>(n_neighbors));

  const int mpi_tag = fsils_vector_comm_tag(dof);
  std::size_t send_offset = 0;
  std::size_t recv_offset = 0;
  for (int i = 0; i < n_neighbors; ++i) {
    const auto& ghost_nodes = lhs.owned_halo_recv_nodes[static_cast<std::size_t>(i)];
    const auto& owned_nodes = lhs.owned_halo_send_nodes[static_cast<std::size_t>(i)];
    const int peer = lhs.owned_halo_neighbor_ranks[static_cast<std::size_t>(i)];

    for (std::size_t j = 0; j < ghost_nodes.size(); ++j) {
      const auto node = ghost_nodes[j];
      for (int l = 0; l < dof; ++l) {
        send_buffer[send_offset + j * static_cast<std::size_t>(dof) + static_cast<std::size_t>(l)] =
            R(l, node);
      }
    }

    if (!owned_nodes.empty()) {
      recv_req.push_back(MPI_REQUEST_NULL);
      MPI_Irecv(recv_buffer.data() + recv_offset,
                static_cast<int>(owned_nodes.size()) * dof,
                mpreal,
                peer,
                mpi_tag,
                lhs.commu.comm,
                &recv_req.back());
    }
    if (!ghost_nodes.empty()) {
      send_req.push_back(MPI_REQUEST_NULL);
      MPI_Isend(send_buffer.data() + send_offset,
                static_cast<int>(ghost_nodes.size()) * dof,
                mpreal,
                peer,
                mpi_tag,
                lhs.commu.comm,
                &send_req.back());
    }

    send_offset += ghost_nodes.size() * static_cast<std::size_t>(dof);
    recv_offset += owned_nodes.size() * static_cast<std::size_t>(dof);
  }

  if (!recv_req.empty()) {
    MPI_Waitall(static_cast<int>(recv_req.size()), recv_req.data(), MPI_STATUSES_IGNORE);
  }

  recv_offset = 0;
  for (int i = 0; i < n_neighbors; ++i) {
    const auto& owned_nodes = lhs.owned_halo_send_nodes[static_cast<std::size_t>(i)];
    for (std::size_t j = 0; j < owned_nodes.size(); ++j) {
      const auto node = owned_nodes[j];
      for (int l = 0; l < dof; ++l) {
        R(l, node) +=
            recv_buffer[recv_offset + j * static_cast<std::size_t>(dof) + static_cast<std::size_t>(l)];
      }
    }
    recv_offset += owned_nodes.size() * static_cast<std::size_t>(dof);
  }

  if (!send_req.empty()) {
    MPI_Waitall(static_cast<int>(send_req.size()), send_req.data(), MPI_STATUSES_IGNORE);
  }

  for (int i = 0; i < n_neighbors; ++i) {
    const auto& ghost_nodes = lhs.owned_halo_recv_nodes[static_cast<std::size_t>(i)];
    for (const auto node : ghost_nodes) {
      for (int l = 0; l < dof; ++l) {
        R(l, node) = 0.0;
      }
    }
  }
}

} // namespace

static void fsils_syncs_impl(const FSILS_lhsType& lhs, Vector<double>& R)
{
  if (lhs.commu.nTasks == 1) {
    return;
  }

  require_owned_halo_plan(lhs, "FSILS scalar owner-to-ghost sync");
  fsils_syncs_owned_halo(lhs, R);
}

void fsils_syncs_owned_to_ghost(const FSILS_lhsType& lhs, Vector<double>& R)
{
  fsils_syncs_impl(lhs, R);
}

void fsils_reduce_shared_face_values_owned_row(const FSILS_lhsType& lhs,
    int dof,
    const Vector<int>& face_nodes,
    Array<double>& face_values)
{
  if (lhs.commu.nTasks <= 1 || dof <= 0 || face_nodes.size() <= 0 || face_values.size() <= 0) {
    return;
  }

  const int face_node_count = std::min(face_nodes.size(), face_values.ncols());
  const int face_dof = std::min(dof, face_values.nrows());
  if (face_node_count <= 0 || face_dof <= 0) {
    return;
  }

  Array<double> contributions(dof, lhs.nNo);
  contributions = 0.0;
  for (int a = 0; a < face_node_count; ++a) {
    const int node = face_nodes(a);
    if (node < 0 || node >= lhs.nNo) {
      continue;
    }
    for (int i = 0; i < face_dof; ++i) {
      contributions(i, node) = face_values(i, a);
    }
  }

  fsils_reverse_scatterv_contribution_buffer(lhs, dof, contributions);
  fsils_syncv_owned_to_ghost(lhs, dof, contributions);

  for (int a = 0; a < face_node_count; ++a) {
    const int node = face_nodes(a);
    if (node < 0 || node >= lhs.nNo) {
      continue;
    }
    for (int i = 0; i < face_dof; ++i) {
      face_values(i, a) = contributions(i, node);
    }
  }
}

void fsils_apply_shared_dirichlet_face_mask(const FSILS_lhsType& lhs,
    int dof,
    const Vector<int>& face_nodes,
    Array<double>& face_values)
{
  if (lhs.commu.nTasks <= 1 || dof <= 0 || face_nodes.size() <= 0 || face_values.size() <= 0) {
    return;
  }

  const int face_node_count = std::min(face_nodes.size(), face_values.ncols());
  const int face_dof = std::min(dof, face_values.nrows());
  if (face_node_count <= 0 || face_dof <= 0) {
    return;
  }

  Array<double> counts(dof, face_node_count);
  counts = 0.0;
  for (int a = 0; a < face_node_count; ++a) {
    for (int i = 0; i < face_dof; ++i) {
      counts(i, a) = 1.0;
    }
  }

  fsils_reduce_shared_face_values_owned_row(lhs, dof, face_nodes, counts);

  constexpr double kMaskTol = 1e-12;
  for (int a = 0; a < face_node_count; ++a) {
    const int node = face_nodes(a);
    if (node < 0 || node >= lhs.nNo) {
      continue;
    }
    for (int i = 0; i < face_dof; ++i) {
      face_values(i, a) = (face_values(i, a) >= counts(i, a) - kMaskTol) ? 1.0 : 0.0;
    }
  }
}

static void fsils_reverse_scatterv_impl(const FSILS_lhsType& lhs, int dof, Array<double>& R)
{
  if (lhs.commu.nTasks == 1) {
    return;
  }

  require_owned_halo_plan(lhs, "FSILS reverse scatter");
  fsils_reverse_scatterv_owned_halo(lhs, dof, R);
}

void fsils_reverse_scatterv_contribution_buffer(const FSILS_lhsType& lhs, int dof, Array<double>& R)
{
  fsils_reverse_scatterv_impl(lhs, dof, R);
}

static void fsils_syncv_impl(const FSILS_lhsType& lhs, int dof, Array<double>& R)
{
  if (lhs.commu.nTasks == 1) {
    return;
  }

  require_owned_halo_plan(lhs, "FSILS vector owner-to-ghost sync");
  fsils_syncv_owned_halo(lhs, dof, R);
}

void fsils_syncv_owned_to_ghost(const FSILS_lhsType& lhs, int dof, Array<double>& R)
{
  fsils_syncv_impl(lhs, dof, R);
}

};
