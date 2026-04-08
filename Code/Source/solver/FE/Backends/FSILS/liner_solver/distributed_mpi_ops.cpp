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

bool HaloExchange::has_overlap() const noexcept
{
  return lhs_ != nullptr &&
         lhs_->commu.nTasks > 1 &&
         lhs_->nReq > 0 &&
         !lhs_->cS.empty();
}

void HaloExchange::sync_scalar(Vector<double>& values, bool skip_overlap_sum) const
{
  if (skip_overlap_sum || !has_overlap()) {
    return;
  }
  fsils_syncs(*lhs_, values);
}

void HaloExchange::sync_vector(int dof, Array<double>& values, bool skip_overlap_sum) const
{
  if (skip_overlap_sum || !has_overlap()) {
    return;
  }
  fsils_syncv(*lhs_, dof, values);
}

void HaloExchange::begin_scalar(Vector<double>& values) const
{
  if (!has_overlap()) {
    return;
  }
  fsils_commus_begin(*lhs_, values);
}

void HaloExchange::end_scalar(Vector<double>& values) const
{
  if (!has_overlap()) {
    return;
  }
  fsils_commus_end(*lhs_, values);
}

void HaloExchange::begin_vector(int dof, Array<double>& values) const
{
  if (!has_overlap()) {
    return;
  }
  fsils_commuv_begin(*lhs_, dof, values);
}

void HaloExchange::end_vector(int dof, Array<double>& values) const
{
  if (!has_overlap()) {
    return;
  }
  fsils_commuv_end(*lhs_, dof, values);
}

}  // namespace fe_fsi_linear_solver
