/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_ADVANCED_STABILITY_ANALYZERS_H
#define SVMP_FE_ANALYSIS_ADVANCED_STABILITY_ANALYZERS_H

/**
 * @file AdvancedStabilityAnalyzers.h
 * @brief Phase 6 analyzer passes for summary-backed discrete stability checks.
 *
 * These passes consume generic symbolic claims and compact numeric summaries.
 * They do not inspect named physics modules or equation labels.
 */

#include "Analysis/ProblemAnalyzer.h"

namespace svmp {
namespace FE {
namespace analysis {

class TemporalStabilityAnalyzer : public AnalyzerPass {
public:
    [[nodiscard]] std::string name() const override;
    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

class EnergyEntropyLawAnalyzer : public AnalyzerPass {
public:
    [[nodiscard]] std::string name() const override;
    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

class CoefficientConstitutiveAnalyzer : public AnalyzerPass {
public:
    [[nodiscard]] std::string name() const override;
    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

class NonlinearTangentAnalyzer : public AnalyzerPass {
public:
    [[nodiscard]] std::string name() const override;
    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

class LockingRiskAnalyzer : public AnalyzerPass {
public:
    [[nodiscard]] std::string name() const override;
    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

class SpectralSpuriousModeAnalyzer : public AnalyzerPass {
public:
    [[nodiscard]] std::string name() const override;
    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

class ErrorEstimatorAnalyzer : public AnalyzerPass {
public:
    [[nodiscard]] std::string name() const override;
    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

class QuadratureAdequacyAnalyzer : public AnalyzerPass {
public:
    [[nodiscard]] std::string name() const override;
    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

class MinimumResidualStabilityAnalyzer : public AnalyzerPass {
public:
    [[nodiscard]] std::string name() const override;
    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

class PreservationStructureAnalyzer : public AnalyzerPass {
public:
    [[nodiscard]] std::string name() const override;
    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

class CoupledSystemStabilityAnalyzer : public AnalyzerPass {
public:
    [[nodiscard]] std::string name() const override;
    void run(const ProblemAnalysisContext& context,
             ProblemAnalysisReport& report) const override;
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_ADVANCED_STABILITY_ANALYZERS_H
