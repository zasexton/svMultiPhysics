/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Analysis/ProblemAnalyzer.h"
#include "Analysis/KernelAnalyzer.h"
#include "Analysis/CouplingGraphAnalyzer.h"
#include "Analysis/ConstraintRankAnalyzer.h"
#include "Analysis/MixedOperatorAnalyzer.h"
#include "Analysis/OperatorClassAnalyzer.h"
#include "Analysis/StabilizationAnalyzer.h"
#include "Analysis/CompatibilityAnalyzer.h"
#include "Analysis/TopologyScopeAnalyzer.h"
#include "Analysis/InterfaceValidationAnalyzer.h"
#include "Analysis/InfSupAnalyzer.h"
#include "Analysis/TransportCharacterAnalyzer.h"
#include "Analysis/ConservationAnalyzer.h"
#include "Analysis/DAEStructureAnalyzer.h"
#include "Analysis/SpaceCompatibilityAnalyzer.h"

namespace svmp {
namespace FE {
namespace analysis {

void ProblemAnalyzer::addPass(std::unique_ptr<AnalyzerPass> pass) {
    if (pass) {
        passes_.push_back(std::move(pass));
    }
}

std::vector<std::string> ProblemAnalyzer::passNames() const {
    std::vector<std::string> names;
    names.reserve(passes_.size());
    for (const auto& p : passes_) {
        names.push_back(p->name());
    }
    return names;
}

ProblemAnalysisReport ProblemAnalyzer::analyze(const ProblemAnalysisContext& context) const {
    ProblemAnalysisReport report;
    for (const auto& pass : passes_) {
        pass->run(context, report);
    }
    return report;
}

ProblemAnalyzer ProblemAnalyzer::createDefault() {
    ProblemAnalyzer analyzer;

    // Passes in dependency order:
    //   1. CouplingGraphAnalyzer (no dependencies)
    //   2. KernelAnalyzer (no dependencies)
    //   3. MixedOperatorAnalyzer (no dependencies)
    //   4. OperatorClassAnalyzer (no dependencies)
    //   5. StabilizationAnalyzer (no dependencies)
    //   6. ConstraintRankAnalyzer (reads KernelAnalyzer claims)
    //   7. CompatibilityAnalyzer (reads KernelAnalyzer + ConstraintRankAnalyzer claims)
    //   8. TopologyScopeAnalyzer (reads all prior claims)
    //   9. InterfaceValidationAnalyzer
    //  10. InfSupAnalyzer (reads MixedOperatorAnalyzer claims)
    //  11. TransportCharacterAnalyzer (no dependencies beyond contributions)
    //  12. ConservationAnalyzer (no dependencies beyond contributions)
    //  13. DAEStructureAnalyzer (no dependencies beyond contributions)
    //  14. SpaceCompatibilityAnalyzer (reads MixedSaddlePoint claims + field descriptors)

    analyzer.addPass(std::make_unique<CouplingGraphAnalyzer>());
    analyzer.addPass(std::make_unique<KernelAnalyzer>());
    analyzer.addPass(std::make_unique<MixedOperatorAnalyzer>());
    analyzer.addPass(std::make_unique<OperatorClassAnalyzer>());
    analyzer.addPass(std::make_unique<StabilizationAnalyzer>());
    analyzer.addPass(std::make_unique<ConstraintRankAnalyzer>());
    analyzer.addPass(std::make_unique<CompatibilityAnalyzer>());
    analyzer.addPass(std::make_unique<TopologyScopeAnalyzer>());
    analyzer.addPass(std::make_unique<InterfaceValidationAnalyzer>());
    analyzer.addPass(std::make_unique<InfSupAnalyzer>());
    analyzer.addPass(std::make_unique<TransportCharacterAnalyzer>());
    analyzer.addPass(std::make_unique<ConservationAnalyzer>());
    analyzer.addPass(std::make_unique<DAEStructureAnalyzer>());
    analyzer.addPass(std::make_unique<SpaceCompatibilityAnalyzer>());

    return analyzer;
}

} // namespace analysis
} // namespace FE
} // namespace svmp
