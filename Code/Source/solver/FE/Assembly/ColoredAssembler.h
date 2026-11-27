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

#ifndef SVMP_FE_ASSEMBLY_COLORED_ASSEMBLER_H
#define SVMP_FE_ASSEMBLY_COLORED_ASSEMBLER_H

/**
 * @file ColoredAssembler.h
 * @brief Graph-colored parallel assembly strategy (decorator)
 *
 * The standard base prepares physical cell contexts. Local kernels of the
 * same color may execute concurrently, then the base inserts their outputs
 * in traversal order, including projection onto shared constraint masters.
 * Other decorated backends retain their own assembly strategy.
 */

#include "Assembly/Coloring.h"
#include "Assembly/DecoratorAssembler.h"
#include "Assembly/AssemblyKernel.h"

#include <vector>
#include <span>
#include <memory>
#include <string>

namespace svmp {
namespace FE {

namespace assembly {

// ============================================================================
// Colored Assembler
// ============================================================================

/**
 * @brief Graph-colored parallel assembler decorator
 *
 * ColoredAssembler provides race-free parallel assembly by ensuring
 * elements processed simultaneously do not share DOFs.
 */
class ColoredAssembler final : public DecoratorAssembler {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor (wraps StandardAssembler)
     */
    ColoredAssembler();

    /**
     * @brief Construct with coloring options (wraps StandardAssembler)
     */
    explicit ColoredAssembler(const ColoringOptions& options);

    /**
     * @brief Construct as a decorator around an existing assembler
     */
    explicit ColoredAssembler(std::unique_ptr<Assembler> base,
                              const ColoringOptions& options = {});

    /**
     * @brief Destructor
     */
    ~ColoredAssembler() override;

    /**
     * @brief Move constructor
     */
    ColoredAssembler(ColoredAssembler&& other) noexcept;

    /**
     * @brief Move assignment
     */
    ColoredAssembler& operator=(ColoredAssembler&& other) noexcept;

    // Non-copyable
    ColoredAssembler(const ColoredAssembler&) = delete;
    ColoredAssembler& operator=(const ColoredAssembler&) = delete;

    // Bring base-class overloads into scope (avoid overload hiding).
    using Assembler::assembleMatrix;
    using Assembler::assembleBoth;
    using Assembler::assembleBoundaryFaces;

    // =========================================================================
    // Query
    // =========================================================================

    [[nodiscard]] std::string name() const override { return "Colored(" + base().name() + ")"; }


    // =========================================================================
    // Configuration (Assembler interface)
    // =========================================================================

    void setDofMap(const dofs::DofMap& dof_map) override;
    void setDofHandler(const dofs::DofHandler& dof_handler) override;
    void setRowDofMap(const dofs::DofMap& dof_map,
                      GlobalIndex row_offset = 0,
                      DofEntityScope row_scope = DofEntityScope::Cell) override;
    void setColDofMap(const dofs::DofMap& dof_map,
                      GlobalIndex col_offset = 0,
                      DofEntityScope col_scope = DofEntityScope::Cell) override;
    void setOptions(const AssemblyOptions& options) override;

    void reset() override;

    // =========================================================================
    // Assembly Operations (Assembler interface)
    // =========================================================================

    [[nodiscard]] AssemblyResult assembleMatrix(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view) override;

    [[nodiscard]] AssemblyResult assembleVector(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& space,
        AssemblyKernel& kernel,
        GlobalSystemView& vector_view) override;

    [[nodiscard]] AssemblyResult assembleBoth(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view,
        GlobalSystemView& vector_view) override;

    [[nodiscard]] AssemblyResult assembleBoundaryFaces(
        const IMeshAccess& mesh,
        int boundary_marker,
        const spaces::FunctionSpace& space,
        AssemblyKernel& kernel,
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view) override;

    [[nodiscard]] AssemblyResult assembleInteriorFaces(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view,
        GlobalSystemView* vector_view,
        int interior_facet_marker = -1,
        geometry::CutIntegrationSide interior_facet_side =
            geometry::CutIntegrationSide::Interface) override;

    [[nodiscard]] AssemblyResult assembleCutVolumes(
        const IMeshAccess& mesh,
        const CutIntegrationContext& cut_context,
        int interface_marker,
        geometry::CutIntegrationSide side,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view,
        bool assemble_matrix,
        bool assemble_vector) override;

    [[nodiscard]] AssemblyResult assembleCutVolumesFused(
        const IMeshAccess& mesh,
        const CutIntegrationContext& cut_context,
        int interface_marker,
        geometry::CutIntegrationSide side,
        std::span<const FusedCellTerm> terms) override;

    [[nodiscard]] AssemblyResult assembleCellsFused(
        const IMeshAccess& mesh,
        std::span<const FusedCellTerm> terms) override;

    // =========================================================================
    // Coloring API
    // =========================================================================

    void computeColoring(const IMeshAccess& mesh, const dofs::DofMap& dof_map);

    void setColoring(std::span<const int> colors, int num_colors);

    void invalidateColoring();

    [[nodiscard]] bool hasColoring() const noexcept { return has_coloring_; }
    [[nodiscard]] int numColors() const noexcept { return num_colors_; }
    [[nodiscard]] std::span<const int> getColors() const noexcept { return element_colors_; }
    [[nodiscard]] const ColoringStats& getColoringStats() const noexcept { return coloring_stats_; }

private:
    // =========================================================================
    // Internal Implementation
    // =========================================================================

    /**
     * @brief Build element connectivity graph
     */
    void buildElementGraph(const IMeshAccess& mesh, const dofs::DofMap& dof_map);

    /**
     * @brief Greedy coloring algorithm
     */
    int greedyColoring();

    /**
     * @brief DSatur coloring algorithm
     */
    int dsaturColoring();

    /**
     * @brief Largest-first coloring
     */
    int largestFirstColoring();

    /**
     * @brief Balance colors by reassigning elements
     */
    void balanceColors();

    [[nodiscard]] AssemblyResult assembleCells(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view);

    // =========================================================================
    // Data Members
    // =========================================================================

    ColoringOptions options_;
    const dofs::DofMap* dof_map_{nullptr};

    // Element graph
    ElementGraph element_graph_;

    // Coloring
    std::vector<int> element_colors_;
    int num_colors_{0};
    bool has_coloring_{false};
    ColoringStats coloring_stats_;
};

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_COLORED_ASSEMBLER_H
