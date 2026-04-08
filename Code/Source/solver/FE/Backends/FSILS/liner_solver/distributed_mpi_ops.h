#ifndef SV_FE_LS_DISTRIBUTED_MPI_OPS_H
#define SV_FE_LS_DISTRIBUTED_MPI_OPS_H

#include "fsils_api.hpp"

#include <vector>

namespace fe_fsi_linear_solver {

class CollectiveOps {
 public:
  explicit CollectiveOps(FSILS_commuType& commu) noexcept;
  explicit CollectiveOps(const FSILS_commuType& commu) noexcept;

  [[nodiscard]] bool distributed() const noexcept;

  void allreduce_sum(double local_value, double& global_value) const;
  void allreduce_sum(int local_value, int& global_value) const;
  void allreduce_sum(const double* sendbuf, double* recvbuf, int count) const;
  void allreduce_sum(const int* sendbuf, int* recvbuf, int count) const;
  void allreduce_sum_in_place(double* buffer, int count) const;
  void allreduce_sum(std::vector<double>& values) const;

 private:
  [[nodiscard]] FSILS_commuType& mutable_commu() const noexcept;

  FSILS_commuType* commu_{nullptr};
};

class HaloExchange {
 public:
  explicit HaloExchange(const FSILS_lhsType& lhs) noexcept;

  [[nodiscard]] bool has_overlap() const noexcept;

  void sync_scalar(Vector<double>& values, bool skip_overlap_sum = false) const;
  void sync_vector(int dof, Array<double>& values, bool skip_overlap_sum = false) const;

  void begin_scalar(Vector<double>& values) const;
  void end_scalar(Vector<double>& values) const;
  void begin_vector(int dof, Array<double>& values) const;
  void end_vector(int dof, Array<double>& values) const;

 private:
  const FSILS_lhsType* lhs_{nullptr};
};

}  // namespace fe_fsi_linear_solver

#endif  // SV_FE_LS_DISTRIBUTED_MPI_OPS_H
