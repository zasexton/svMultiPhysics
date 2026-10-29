/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_FORMS_RESIDUALCONTRIBUTIONS_H
#define SVMP_FE_FORMS_RESIDUALCONTRIBUTIONS_H

#include "Forms/Vocabulary.h"

#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace svmp::FE::forms {

/// Borrowed destinations for one residual row; null observations are skipped.
struct ResidualContributionRow {
    FieldId test_field;
    FormExpr& residual;
    std::vector<FormExpr*> observations{};
};

/**
 * Appends each contribution immediately, production first, using sumOf's left
 * fold. Row bindings and observation lists are copied; all expression targets
 * are borrowed and must outlive this scoped sink. Every non-null target must be
 * distinct across all rows, so one append cannot update the same target twice.
 */
class ResidualContributions {
public:
    explicit ResidualContributions(std::initializer_list<ResidualContributionRow> rows)
        : rows_(rows)
    {
        std::vector<FormExpr*> targets;
        const auto add_target = [&targets](FormExpr* target) {
            if (!target) {
                return;
            }
            if (std::find(targets.begin(), targets.end(), target) != targets.end()) {
                throw std::invalid_argument("ResidualContributions: repeated expression target");
            }
            targets.push_back(target);
        };
        for (auto row = rows_.begin(); row != rows_.end(); ++row) {
            if (row->test_field == INVALID_FIELD_ID) {
                throw std::invalid_argument("ResidualContributions: invalid test field");
            }
            if (std::any_of(rows_.begin(), row, [&row](const auto& earlier) {
                    return earlier.test_field == row->test_field;
                })) {
                throw std::invalid_argument("ResidualContributions: duplicate test field");
            }
            add_target(&row->residual);
            for (auto* observation : row->observations) {
                add_target(observation);
            }
        }
    }

    void append(FieldId test_field, const FormExpr& term)
    {
        const auto row = std::find_if(rows_.begin(), rows_.end(), [test_field](const auto& binding) {
            return binding.test_field == test_field;
        });
        if (row == rows_.end()) {
            throw std::invalid_argument("ResidualContributions::append: unknown test field");
        }
        // The argument may refer to any borrowed target, including production.
        // Retain its immutable node before the first destination assignment.
        const auto contribution = term;
        row->residual = sumOf(row->residual, contribution);
        for (auto* observation : row->observations) {
            if (observation) {
                *observation = sumOf(*observation, contribution);
            }
        }
    }

private:
    std::vector<ResidualContributionRow> rows_;
};

} // namespace svmp::FE::forms

#endif // SVMP_FE_FORMS_RESIDUALCONTRIBUTIONS_H
