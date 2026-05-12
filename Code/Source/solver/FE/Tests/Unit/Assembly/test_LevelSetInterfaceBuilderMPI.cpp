#include "Interfaces/LevelSetInterfaceBuilder.h"

#include <gtest/gtest.h>

#include <mpi.h>

using namespace svmp::FE;
using namespace svmp::FE::interfaces;

namespace {

CutInterfaceDomainRequest make_mpi_request()
{
    CutInterfaceDomainRequest request;
    request.source = LevelSetInterfaceSource::fromField(/*field_id=*/5,
                                                        /*layout_revision=*/1,
                                                        /*value_revision=*/1);
    request.interface_marker = 70;
    request.isovalue = 0.0;
    request.tolerance = 1.0e-12;
    return request;
}

} // namespace

TEST(LevelSetInterfaceBuilderMPI, OwnedFragmentsReduceToConsistentGlobalMeasure)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    ASSERT_GE(size, 2);

    LevelSetInterfaceDomain domain(make_mpi_request());
    if (rank == 0) {
        appendLinearLevelSetCellCut2D(
            domain,
            LevelSetCellCutInput{.parent_cell = 0,
                                 .element_type = ElementType::Triangle3,
                                 .node_coordinates = {{{0.0, 0.0, 0.0}},
                                                      {{1.0, 0.0, 0.0}},
                                                      {{0.0, 1.0, 0.0}}},
                                 .level_set_values = {-0.25, 0.75, -0.25}});
    } else if (rank == 1) {
        appendLinearLevelSetCellCut2D(
            domain,
            LevelSetCellCutInput{.parent_cell = 1,
                                 .element_type = ElementType::Quad4,
                                 .node_coordinates = {{{0.0, 0.0, 0.0}},
                                                      {{1.0, 0.0, 0.0}},
                                                      {{1.0, 1.0, 0.0}},
                                                      {{0.0, 1.0, 0.0}}},
                                 .level_set_values = {-0.5, 0.5, 0.5, -0.5}});
    }

    const auto local_summary = domain.summary();
    const auto local_fragments =
        static_cast<unsigned long long>(local_summary.active_fragment_count);
    unsigned long long global_fragments = 0;
    MPI_Allreduce(&local_fragments,
                  &global_fragments,
                  1,
                  MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    const double local_measure = static_cast<double>(local_summary.measure);
    double global_measure = 0.0;
    MPI_Allreduce(&local_measure,
                  &global_measure,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    const auto local_owned_cells =
        local_summary.active_fragment_count == 0u ? 0ull : 1ull;
    unsigned long long global_owned_cells = 0;
    MPI_Allreduce(&local_owned_cells,
                  &global_owned_cells,
                  1,
                  MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    EXPECT_EQ(global_fragments, 2ull);
    EXPECT_EQ(global_owned_cells, 2ull);
    EXPECT_NEAR(global_measure, 1.75, 1.0e-14);
}
