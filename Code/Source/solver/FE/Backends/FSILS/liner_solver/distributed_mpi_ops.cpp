#include "distributed_mpi_ops.h"

#include <algorithm>

namespace fe_fsi_linear_solver {

CollectiveOps::CollectiveOps(FSILS_commuType& commu) noexcept
    : commu_(&commu)
{
}

CollectiveOps::CollectiveOps(const FSILS_commuType& commu) noexcept
    : commu_(&const_cast<FSILS_commuType&>(commu))
{
}

bool CollectiveOps::distributed() const noexcept
{
  return commu_ != nullptr && commu_->nTasks > 1;
}

void CollectiveOps::allreduce_sum(double local_value, double& global_value) const
{
  global_value = local_value;
  if (!distributed()) {
    return;
  }
  fsils_allreduce_sum(&local_value, &global_value, 1, cm_mod::mpreal, mutable_commu());
}

void CollectiveOps::allreduce_sum(int local_value, int& global_value) const
{
  global_value = local_value;
  if (!distributed()) {
    return;
  }
  fsils_allreduce_sum(&local_value, &global_value, 1, cm_mod::mpint, mutable_commu());
}

void CollectiveOps::allreduce_sum(const double* sendbuf, double* recvbuf, int count) const
{
  if (count <= 0 || recvbuf == nullptr || sendbuf == nullptr) {
    return;
  }
  if (!distributed()) {
    if (recvbuf != sendbuf) {
      std::copy(sendbuf, sendbuf + count, recvbuf);
    }
    return;
  }
  fsils_allreduce_sum(sendbuf, recvbuf, count, cm_mod::mpreal, mutable_commu());
}

void CollectiveOps::allreduce_sum(const int* sendbuf, int* recvbuf, int count) const
{
  if (count <= 0 || recvbuf == nullptr || sendbuf == nullptr) {
    return;
  }
  if (!distributed()) {
    if (recvbuf != sendbuf) {
      std::copy(sendbuf, sendbuf + count, recvbuf);
    }
    return;
  }
  fsils_allreduce_sum(sendbuf, recvbuf, count, cm_mod::mpint, mutable_commu());
}

void CollectiveOps::allreduce_sum_in_place(double* buffer, int count) const
{
  if (!distributed() || buffer == nullptr || count <= 0) {
    return;
  }
  fsils_allreduce_sum_in_place(buffer, count, cm_mod::mpreal, mutable_commu());
}

void CollectiveOps::allreduce_sum(std::vector<double>& values) const
{
  allreduce_sum_in_place(values.data(), static_cast<int>(values.size()));
}

FSILS_commuType& CollectiveOps::mutable_commu() const noexcept
{
  return *commu_;
}

HaloExchange::HaloExchange(const FSILS_lhsType& lhs) noexcept
    : lhs_(&lhs)
{
}

bool HaloExchange::has_owned_halo() const noexcept
{
  if (lhs_ == nullptr || lhs_->commu.nTasks <= 1 || !lhs_->owned_row_operator) {
    return false;
  }
  if (lhs_->owned_halo_send_nodes.size() != lhs_->owned_halo_neighbor_ranks.size() ||
      lhs_->owned_halo_recv_nodes.size() != lhs_->owned_halo_neighbor_ranks.size()) {
    return false;
  }
  return !lhs_->owned_halo_neighbor_ranks.empty();
}

void HaloExchange::sync_owned_to_ghost_scalar(Vector<double>& values, bool skip_sync) const
{
  if (skip_sync || lhs_ == nullptr || lhs_->commu.nTasks <= 1) {
    return;
  }
  fsils_syncs_owned_to_ghost(*lhs_, values);
}

void HaloExchange::sync_owned_to_ghost_vector(int dof, Array<double>& values, bool skip_sync) const
{
  if (skip_sync || lhs_ == nullptr || lhs_->commu.nTasks <= 1) {
    return;
  }
  fsils_syncv_owned_to_ghost(*lhs_, dof, values);
}

void HaloExchange::reverse_scatter_vector_contributions(int dof, Array<double>& values) const
{
  if (lhs_ == nullptr || lhs_->commu.nTasks <= 1) {
    return;
  }
  fsils_reverse_scatterv_contribution_buffer(*lhs_, dof, values);
}

}  // namespace fe_fsi_linear_solver
