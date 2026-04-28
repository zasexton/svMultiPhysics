/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_MESH_GEOMETRY_ANALYZER_H
#define SVMP_FE_ANALYSIS_MESH_GEOMETRY_ANALYZER_H

/**
 * @file MeshGeometryAnalyzer.h
 * @brief Consumes mesh mapping/quality summaries for geometry-validity claims.
 */

#include "Analysis/ProblemAnalyzer.h"

namespace svmp {
namespace FE {
namespace analysis {

class MeshGeometryAnalyzer : public AnalyzerPass {
public:
    MeshGeometryAnalyzer() = default;

    [[nodiscard]] std::string name() const override;

    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_MESH_GEOMETRY_ANALYZER_H
