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

#ifndef SVMP_FE_FORMS_MIXED_FORM_IR_H
#define SVMP_FE_FORMS_MIXED_FORM_IR_H

/**
 * @file MixedFormIR.h
 * @brief Block-sparse mixed form IR for native multi-field support
 *
 * MixedFormIR wraps a block-sparse compilation of a multi-field weak form.
 * Each active (test_field, trial_field) block is a separately compiled FormIR,
 * preserving per-block JIT specialization and zero-block elimination while
 * presenting a single unified IR to the assembly and installation layers.
 *
 * This is the output of the mixed-form compiler:
 *   MixedFormExpr -> FormCompiler::compile() -> MixedFormIR
 *                                               |-- block(0,0): FormIR  (e.g., VV)
 *                                               |-- block(0,1): FormIR  (e.g., VP)
 *                                               |-- block(1,0): FormIR  (e.g., PV)
 *                                               +-- block(1,1): nullopt (zero block)
 *
 * ## Block classification rules
 *
 * Each integral term in a mixed expression is classified into exactly one
 * (test_idx, trial_idx) block by checking which TestFunction and TrialFunction
 * names appear in the integrand. The integral domain is preserved per-term:
 *
 *   - **Cell** terms:           integrands under `.dx()` measure
 *   - **Boundary** terms:       integrands under `.ds(marker)` measure
 *   - **Interior-face** terms:  integrands under `.dS()` measure
 *   - **Interface-face** terms: integrands under `.dI(marker)` measure
 *   - **Global** terms:         not yet classified by compileMixed()
 *
 * A block with no matching terms is a **zero block** (std::nullopt). Zero-block
 * elimination is a required property: zero blocks are never compiled and never
 * installed into Systems.
 *
 * A single block may contain terms from multiple domains (e.g., cell + boundary
 * terms for the same test/trial pair). The per-block FormIR preserves domain
 * classification per IntegralTerm.
 *
 * ## Valid mixed expressions
 *
 * A valid mixed expression for compileMixed() / compile() must satisfy:
 *   - At least one TestFunction is present.
 *   - Zero or more TrialFunction spaces are present (zero = linear form).
 *   - Every integral term must contain exactly one TestFunction and at most one
 *     TrialFunction.
 *   - BoundaryFunctionalSymbol / AuxiliaryStateSymbol nodes are rejected
 *     (coupled placeholders must be resolved before compilation).
 *   - IndexedAccess nodes are rejected unless JIT is enabled (einsum lowering
 *     required).
 */

#include "Core/Types.h"
#include "Forms/FormIR.h"

#include <algorithm>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {

/**
 * @brief Metadata for one field in a mixed formulation
 */
struct MixedFieldDescriptor {
    FieldId field_id{INVALID_FIELD_ID};
    std::string name;
    std::optional<FormExprNode::SpaceSignature> space_signature{};
    int value_dimension{1};
};

/**
 * @brief Provenance record linking a compiled block back to source integral terms
 *
 * Each active block may record which source-level integral terms contributed to it,
 * enabling diagnostics that point from a lowered block back to the original mixed
 * expression.
 */
struct BlockProvenance {
    /// Indices into the source expression's integral term list
    std::vector<std::size_t> contributing_term_indices{};

    /// Human-readable summary of the block's integrand (from source expression)
    std::string source_summary{};
};

/**
 * @brief Summary of integration domains present in a mixed form
 */
struct MixedFormDomainSummary {
    bool has_cell_terms{false};
    bool has_boundary_terms{false};
    bool has_interior_face_terms{false};
    bool has_interface_face_terms{false};

    /// Boundary markers referenced by boundary terms (-1 = all)
    std::vector<int> boundary_markers{};

    /// Interface markers referenced by interface terms (-1 = all)
    std::vector<int> interface_markers{};
};

/**
 * @brief Block-sparse mixed form IR
 *
 * Represents the compilation result of a mixed weak form expression
 * containing multiple test/trial function spaces. Preserves per-block
 * compilation and zero-block sparsity.
 */
class MixedFormIR {
public:
    MixedFormIR() = default;

    MixedFormIR(std::size_t num_test_fields, std::size_t num_trial_fields)
        : num_test_fields_(num_test_fields)
        , num_trial_fields_(num_trial_fields)
        , blocks_(num_test_fields * num_trial_fields)
    {
    }

    // -- Field descriptors --

    void setTestFields(std::vector<MixedFieldDescriptor> fields) {
        test_fields_ = std::move(fields);
    }

    void setTrialFields(std::vector<MixedFieldDescriptor> fields) {
        trial_fields_ = std::move(fields);
    }

    [[nodiscard]] std::span<const MixedFieldDescriptor> testFields() const noexcept {
        return test_fields_;
    }

    [[nodiscard]] std::span<const MixedFieldDescriptor> trialFields() const noexcept {
        return trial_fields_;
    }

    // -- Block access --

    void setBlock(std::size_t test_idx, std::size_t trial_idx, FormIR ir) {
        blocks_.at(test_idx * num_trial_fields_ + trial_idx) = std::move(ir);
    }

    [[nodiscard]] bool hasBlock(std::size_t test_idx, std::size_t trial_idx) const noexcept {
        const auto idx = test_idx * num_trial_fields_ + trial_idx;
        return idx < blocks_.size() && blocks_[idx].has_value();
    }

    [[nodiscard]] const FormIR& block(std::size_t test_idx, std::size_t trial_idx) const {
        return blocks_.at(test_idx * num_trial_fields_ + trial_idx).value();
    }

    [[nodiscard]] FormIR& block(std::size_t test_idx, std::size_t trial_idx) {
        return blocks_.at(test_idx * num_trial_fields_ + trial_idx).value();
    }

    [[nodiscard]] const std::optional<FormIR>& blockOpt(std::size_t test_idx,
                                                         std::size_t trial_idx) const {
        return blocks_.at(test_idx * num_trial_fields_ + trial_idx);
    }

    // -- Dimensions --

    [[nodiscard]] std::size_t numTestFields() const noexcept { return num_test_fields_; }
    [[nodiscard]] std::size_t numTrialFields() const noexcept { return num_trial_fields_; }

    [[nodiscard]] std::size_t numActiveBlocks() const noexcept {
        std::size_t count = 0;
        for (const auto& b : blocks_) {
            if (b.has_value()) ++count;
        }
        return count;
    }

    // -- Kind --

    void setKind(FormKind kind) noexcept { kind_ = kind; }
    [[nodiscard]] FormKind kind() const noexcept { return kind_; }

    // -- Source provenance --

    /**
     * @brief Store the original mixed source expression for diagnostics
     *
     * This is the user-authored FormExpr before decomposition into blocks.
     * Retained for analysis, error messages, and source-aware diagnostics.
     */
    void setSourceExpression(FormExpr expr) { source_expression_ = std::move(expr); }

    [[nodiscard]] const std::optional<FormExpr>& sourceExpression() const noexcept {
        return source_expression_;
    }

    /**
     * @brief Set provenance for a specific block
     */
    void setBlockProvenance(std::size_t test_idx, std::size_t trial_idx, BlockProvenance prov) {
        if (block_provenances_.size() != blocks_.size()) {
            block_provenances_.resize(blocks_.size());
        }
        block_provenances_.at(test_idx * num_trial_fields_ + trial_idx) = std::move(prov);
    }

    /**
     * @brief Get provenance for a specific block (if recorded)
     */
    [[nodiscard]] const std::optional<BlockProvenance>& blockProvenance(
        std::size_t test_idx, std::size_t trial_idx) const {
        static const std::optional<BlockProvenance> empty{};
        if (block_provenances_.empty()) return empty;
        return block_provenances_.at(test_idx * num_trial_fields_ + trial_idx);
    }

    // -- Whole-form metadata --

    void setDomainSummary(MixedFormDomainSummary summary) {
        domain_summary_ = std::move(summary);
    }

    [[nodiscard]] const MixedFormDomainSummary& domainSummary() const noexcept {
        return domain_summary_;
    }

    // -- Sparsity query --

    /**
     * @brief Return active block indices as (test_idx, trial_idx) pairs
     */
    [[nodiscard]] std::vector<std::pair<std::size_t, std::size_t>> activeBlocks() const {
        std::vector<std::pair<std::size_t, std::size_t>> result;
        for (std::size_t i = 0; i < num_test_fields_; ++i) {
            for (std::size_t j = 0; j < num_trial_fields_; ++j) {
                if (hasBlock(i, j)) {
                    result.emplace_back(i, j);
                }
            }
        }
        return result;
    }

    /**
     * @brief Union of all field requirements across active blocks
     */
    [[nodiscard]] std::vector<assembly::FieldRequirement> allFieldRequirements() const {
        std::unordered_map<FieldId, assembly::RequiredData> merged;
        for (const auto& b : blocks_) {
            if (!b.has_value()) continue;
            for (const auto& fr : b->fieldRequirements()) {
                merged[fr.field] |= fr.required;
            }
        }
        std::vector<assembly::FieldRequirement> out;
        out.reserve(merged.size());
        for (const auto& kv : merged) {
            out.push_back({kv.first, kv.second});
        }
        std::sort(out.begin(), out.end(),
                  [](const auto& a, const auto& b) { return a.field < b.field; });
        return out;
    }

private:
    std::size_t num_test_fields_{0};
    std::size_t num_trial_fields_{0};
    std::vector<std::optional<FormIR>> blocks_{};  // Row-major: test * num_trial + trial
    std::vector<MixedFieldDescriptor> test_fields_{};
    std::vector<MixedFieldDescriptor> trial_fields_{};
    FormKind kind_{FormKind::Bilinear};

    // Source provenance
    std::optional<FormExpr> source_expression_{};
    std::vector<std::optional<BlockProvenance>> block_provenances_{};

    // Whole-form metadata
    MixedFormDomainSummary domain_summary_{};
};

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_MIXED_FORM_IR_H
