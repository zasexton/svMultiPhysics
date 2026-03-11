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
 * This is the Stage 2 output of the mixed-form compiler:
 *   MixedFormExpr -> FormCompiler::compileMixed() -> MixedFormIR
 *                                                    |-- block(0,0): FormIR  (e.g., VV)
 *                                                    |-- block(0,1): FormIR  (e.g., VP)
 *                                                    |-- block(1,0): FormIR  (e.g., PV)
 *                                                    +-- block(1,1): nullopt (zero block)
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
};

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_MIXED_FORM_IR_H
