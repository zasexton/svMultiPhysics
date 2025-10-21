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

#ifndef SVMP_PARALLEL_ADAPTIVITY_H
#define SVMP_PARALLEL_ADAPTIVITY_H

#include "Options.h"
#include "AdaptivityManager.h"
#include <array>
#include <map>
#include <memory>
#include <set>
#include <vector>
#include <mpi.h>

namespace svmp {

// Forward declarations
class MeshBase;
class DistributedMesh;
class MeshFields;

/**
 * @brief Ghost layer information for parallel adaptivity
 */
struct GhostLayer {
  /** Ghost elements owned by other processors */
  std::vector<size_t> ghost_elements;

  /** Shared vertices with other processors */
  std::map<size_t, std::set<int>> shared_vertices;

  /** Shared edges with other processors */
  std::map<std::pair<size_t, size_t>, std::set<int>> shared_edges;

  /** Shared faces with other processors */
  std::map<std::vector<size_t>, std::set<int>> shared_faces;

  /** Owner processor for ghost elements */
  std::map<size_t, int> ghost_owners;

  /** Layer depth */
  size_t depth = 1;
};

/**
 * @brief Parallel communication pattern
 */
struct CommPattern {
  /** Processors to send data to */
  std::vector<int> send_procs;

  /** Processors to receive data from */
  std::vector<int> recv_procs;

  /** Send buffer sizes */
  std::vector<size_t> send_sizes;

  /** Receive buffer sizes */
  std::vector<size_t> recv_sizes;

  /** Send buffers */
  std::vector<std::vector<double>> send_buffers;

  /** Receive buffers */
  std::vector<std::vector<double>> recv_buffers;

  /** MPI requests for non-blocking communication */
  std::vector<MPI_Request> requests;
};

/**
 * @brief Load balancing information
 */
struct LoadBalance {
  /** Current element distribution */
  std::vector<size_t> element_counts;

  /** Target element distribution */
  std::vector<size_t> target_counts;

  /** Elements to migrate */
  std::map<size_t, int> migration_map;

  /** Load imbalance factor */
  double imbalance_factor;

  /** Is rebalancing needed */
  bool needs_rebalancing;
};

/**
 * @brief Parallel adaptivity statistics
 */
struct ParallelStats {
  /** Communication time */
  double comm_time = 0.0;

  /** Synchronization time */
  double sync_time = 0.0;

  /** Load balancing time */
  double balance_time = 0.0;

  /** Ghost update time */
  double ghost_time = 0.0;

  /** Number of migrated elements */
  size_t migrated_elements = 0;

  /** Number of communication rounds */
  size_t comm_rounds = 0;

  /** Maximum local refinement level */
  size_t max_local_level = 0;

  /** Load imbalance before */
  double imbalance_before = 0.0;

  /** Load imbalance after */
  double imbalance_after = 0.0;
};

/**
 * @brief Parallel adaptivity manager
 *
 * Manages distributed adaptive mesh refinement across MPI processes.
 */
class ParallelAdaptivityManager {
public:
  /**
   * @brief Configuration for parallel adaptivity
   */
  struct Config {
    /** Enable dynamic load balancing */
    bool enable_load_balancing = true;

    /** Load imbalance threshold for rebalancing */
    double imbalance_threshold = 1.2;

    /** Ghost layer depth */
    size_t ghost_depth = 1;

    /** Communication strategy */
    enum class CommStrategy {
      BLOCKING,      // Blocking communication
      NON_BLOCKING,  // Non-blocking communication
      COLLECTIVE,    // Collective operations
      NEIGHBORHOOD   // Neighborhood collectives
    };

    CommStrategy comm_strategy = CommStrategy::NON_BLOCKING;

    /** Partitioning method for load balancing */
    enum class PartitionMethod {
      GRAPH,         // Graph partitioning
      GEOMETRIC,     // Geometric partitioning
      SPACE_FILLING, // Space-filling curves
      RECURSIVE      // Recursive bisection
    };

    PartitionMethod partition_method = PartitionMethod::GRAPH;

    /** Synchronize marks across processors */
    bool synchronize_marks = true;

    /** Enforce global conformity */
    bool global_conformity = true;

    /** Maximum iterations for consensus */
    size_t max_consensus_iterations = 10;
  };

  ParallelAdaptivityManager(MPI_Comm comm, const Config& config = {});

  /**
   * @brief Perform parallel adaptive mesh refinement
   *
   * @param mesh Distributed mesh
   * @param fields Mesh fields
   * @param options Adaptivity options
   * @return Adaptivity result with parallel statistics
   */
  AdaptivityResult adapt_parallel(
      DistributedMesh& mesh,
      MeshFields* fields,
      const AdaptivityOptions& options);

  /**
   * @brief Exchange ghost layer information
   *
   * @param mesh Distributed mesh
   * @param marks Refinement marks
   */
  void exchange_ghost_marks(
      DistributedMesh& mesh,
      std::vector<MarkType>& marks);

  /**
   * @brief Synchronize refinement marks across processors
   *
   * @param mesh Distributed mesh
   * @param marks Local marks (modified)
   * @return Number of iterations to reach consensus
   */
  size_t synchronize_marks(
      DistributedMesh& mesh,
      std::vector<MarkType>& marks);

  /**
   * @brief Load balance after adaptation
   *
   * @param mesh Distributed mesh
   * @param fields Mesh fields
   * @return Number of migrated elements
   */
  size_t rebalance_load(
      DistributedMesh& mesh,
      MeshFields* fields);

  /**
   * @brief Get parallel statistics
   */
  ParallelStats get_statistics() const { return stats_; }

  /**
   * @brief Get MPI rank
   */
  int get_rank() const { return rank_; }

  /**
   * @brief Get MPI size
   */
  int get_size() const { return size_; }

private:
  MPI_Comm comm_;
  int rank_;
  int size_;
  Config config_;
  ParallelStats stats_;
  std::unique_ptr<AdaptivityManager> local_manager_;
  GhostLayer ghost_layer_;
  CommPattern comm_pattern_;

  /** Initialize communication pattern */
  void initialize_comm_pattern(const DistributedMesh& mesh);

  /** Update ghost layers */
  void update_ghost_layers(DistributedMesh& mesh);

  /** Exchange field data */
  void exchange_field_data(
      const DistributedMesh& mesh,
      std::vector<double>& field_data);

  /** Compute load balance */
  LoadBalance compute_load_balance(const DistributedMesh& mesh);

  /** Migrate elements for load balancing */
  void migrate_elements(
      DistributedMesh& mesh,
      const LoadBalance& balance);

  /** Check global conformity */
  bool check_global_conformity(
      const DistributedMesh& mesh,
      const std::vector<MarkType>& marks);

  /** Enforce conformity at processor boundaries */
  void enforce_boundary_conformity(
      DistributedMesh& mesh,
      std::vector<MarkType>& marks);

  /** Collective error estimation */
  void collective_error_estimation(
      const DistributedMesh& mesh,
      std::vector<double>& error_field);

  /** Global marking strategy */
  void apply_global_marking(
      const DistributedMesh& mesh,
      const std::vector<double>& error_field,
      std::vector<MarkType>& marks,
      const AdaptivityOptions& options);
};

/**
 * @brief Parallel communication utilities
 */
class ParallelCommUtils {
public:
  /**
   * @brief All-to-all communication pattern
   */
  template<typename T>
  static void all_to_all(
      MPI_Comm comm,
      const std::vector<std::vector<T>>& send_data,
      std::vector<std::vector<T>>& recv_data);

  /**
   * @brief Neighbor communication pattern
   */
  template<typename T>
  static void neighbor_exchange(
      MPI_Comm comm,
      const std::vector<int>& neighbors,
      const std::vector<T>& send_data,
      std::vector<T>& recv_data);

  /**
   * @brief Global reduction operations
   */
  static double global_min(MPI_Comm comm, double value);
  static double global_max(MPI_Comm comm, double value);
  static double global_sum(MPI_Comm comm, double value);

  /**
   * @brief Gather distributed data
   */
  template<typename T>
  static std::vector<T> gather_all(
      MPI_Comm comm,
      const std::vector<T>& local_data);

  /**
   * @brief Broadcast from root
   */
  template<typename T>
  static void broadcast(
      MPI_Comm comm,
      std::vector<T>& data,
      int root);
};

/**
 * @brief Distributed mesh partitioner
 */
class MeshPartitioner {
public:
  /**
   * @brief Partition configuration
   */
  struct Config {
    /** Partitioning objective */
    enum class Objective {
      BALANCE_ELEMENTS,  // Balance element count
      BALANCE_VERTICES,  // Balance vertex count
      MINIMIZE_EDGE_CUT, // Minimize communication
      MINIMIZE_MESSAGES  // Minimize message count
    };

    Objective objective = Objective::BALANCE_ELEMENTS;

    /** Imbalance tolerance */
    double imbalance_tolerance = 1.05;

    /** Use vertex weights */
    bool use_vertex_weights = false;

    /** Use edge weights */
    bool use_edge_weights = false;
  };

  /**
   * @brief Compute initial partitioning
   */
  static std::vector<int> partition_mesh(
      const MeshBase& mesh,
      int num_partitions,
      const Config& config = {});

  /**
   * @brief Repartition for load balancing
   */
  static std::vector<int> repartition_mesh(
      const DistributedMesh& mesh,
      const std::vector<double>& weights,
      const Config& config = {});

  /**
   * @brief Compute partition quality metrics
   */
  struct QualityMetrics {
    double edge_cut;
    double imbalance;
    size_t max_partition_size;
    size_t min_partition_size;
    std::vector<size_t> partition_sizes;
  };

  static QualityMetrics evaluate_partition(
      const MeshBase& mesh,
      const std::vector<int>& partition);

private:
  /** Graph partitioning using METIS/ParMETIS */
  static std::vector<int> graph_partition(
      const MeshBase& mesh,
      int num_partitions,
      const Config& config);

  /** Geometric partitioning using coordinates */
  static std::vector<int> geometric_partition(
      const MeshBase& mesh,
      int num_partitions,
      const Config& config);

  /** Space-filling curve partitioning */
  static std::vector<int> sfc_partition(
      const MeshBase& mesh,
      int num_partitions,
      const Config& config);

  /** Recursive coordinate bisection */
  static std::vector<int> rcb_partition(
      const MeshBase& mesh,
      int num_partitions,
      const Config& config);
};

/**
 * @brief Ghost layer manager
 */
class GhostLayerManager {
public:
  /**
   * @brief Build ghost layers
   */
  static GhostLayer build_ghost_layer(
      const DistributedMesh& mesh,
      size_t depth = 1);

  /**
   * @brief Update ghost values
   */
  static void update_ghost_values(
      const DistributedMesh& mesh,
      const GhostLayer& ghost,
      std::vector<double>& field_values);

  /**
   * @brief Synchronize ghost marks
   */
  static void synchronize_ghost_marks(
      const DistributedMesh& mesh,
      const GhostLayer& ghost,
      std::vector<MarkType>& marks);

  /**
   * @brief Check consistency across processors
   */
  static bool check_consistency(
      const DistributedMesh& mesh,
      const GhostLayer& ghost);

  /**
   * @brief Pack ghost data for communication
   */
  static std::vector<double> pack_ghost_data(
      const DistributedMesh& mesh,
      const GhostLayer& ghost,
      const std::vector<double>& field);

  /**
   * @brief Unpack received ghost data
   */
  static void unpack_ghost_data(
      const DistributedMesh& mesh,
      const GhostLayer& ghost,
      const std::vector<double>& packed_data,
      std::vector<double>& field);
};

/**
 * @brief Parallel load balancer
 */
class ParallelLoadBalancer {
public:
  /**
   * @brief Configuration for load balancing
   */
  struct Config {
    /** Balancing algorithm */
    enum class Algorithm {
      DIFFUSIVE,      // Diffusive load balancing
      DIRECT,         // Direct reassignment
      HIERARCHICAL,   // Hierarchical balancing
      WORK_STEALING   // Work stealing approach
    };

    Algorithm algorithm = Algorithm::DIFFUSIVE;

    /** Migration cost factor */
    double migration_cost = 1.0;

    /** Minimum migration size */
    size_t min_migration = 10;

    /** Balance computation elements */
    bool balance_computation = true;

    /** Balance communication */
    bool balance_communication = true;
  };

  /**
   * @brief Compute load balance
   */
  static LoadBalance compute_balance(
      const DistributedMesh& mesh,
      const Config& config = {});

  /**
   * @brief Execute load balancing
   */
  static void execute_balancing(
      DistributedMesh& mesh,
      MeshFields* fields,
      const LoadBalance& balance);

  /**
   * @brief Estimate balancing cost
   */
  static double estimate_cost(
      const DistributedMesh& mesh,
      const LoadBalance& balance);

private:
  /** Diffusive load balancing */
  static LoadBalance diffusive_balance(
      const DistributedMesh& mesh,
      const Config& config);

  /** Direct load balancing */
  static LoadBalance direct_balance(
      const DistributedMesh& mesh,
      const Config& config);

  /** Pack elements for migration */
  static std::vector<uint8_t> pack_elements(
      const DistributedMesh& mesh,
      const std::vector<size_t>& elements,
      MeshFields* fields);

  /** Unpack migrated elements */
  static void unpack_elements(
      DistributedMesh& mesh,
      const std::vector<uint8_t>& packed_data,
      MeshFields* fields);
};

/**
 * @brief Parallel adaptivity utilities
 */
class ParallelAdaptivityUtils {
public:
  /**
   * @brief Check if adaptation is balanced
   */
  static bool is_adaptation_balanced(
      MPI_Comm comm,
      const std::vector<MarkType>& marks);

  /**
   * @brief Compute global error indicators
   */
  static double compute_global_error(
      MPI_Comm comm,
      const std::vector<double>& local_error);

  /**
   * @brief Determine global marking threshold
   */
  static double compute_global_threshold(
      MPI_Comm comm,
      const std::vector<double>& local_error,
      double fraction);

  /**
   * @brief Gather adaptation statistics
   */
  static ParallelStats gather_statistics(
      MPI_Comm comm,
      const AdaptivityResult& local_result);

  /**
   * @brief Check parallel mesh consistency
   */
  static bool check_parallel_consistency(
      MPI_Comm comm,
      const DistributedMesh& mesh);

  /**
   * @brief Write parallel mesh for visualization
   */
  static void write_parallel_mesh(
      MPI_Comm comm,
      const DistributedMesh& mesh,
      const std::string& filename);
};

} // namespace svmp

#endif // SVMP_PARALLEL_ADAPTIVITY_H