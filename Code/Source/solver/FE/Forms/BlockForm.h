#ifndef SVMP_FE_FORMS_BLOCK_FORM_H
#define SVMP_FE_FORMS_BLOCK_FORM_H

/**
 * @file BlockForm.h
 * @brief Lightweight block/mixed containers for FE/Forms
 *
 * These types are intentionally simple containers around `forms::FormExpr`.
 * Each block is compiled independently by `forms::FormCompiler`, preserving the
 * current single-trial/single-test constraints of the core compiler while
 * enabling mixed/multi-field Systems via block assembly.
 */

#include "Forms/FormExpr.h"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {

class BlockBilinearForm {
public:
    BlockBilinearForm(std::size_t num_test_fields, std::size_t num_trial_fields)
        : num_test_fields_(num_test_fields),
          num_trial_fields_(num_trial_fields),
          blocks_(num_test_fields * num_trial_fields)
    {
    }

    [[nodiscard]] std::size_t numTestFields() const noexcept { return num_test_fields_; }
    [[nodiscard]] std::size_t numTrialFields() const noexcept { return num_trial_fields_; }

    void setBlock(std::size_t test_field, std::size_t trial_field, FormExpr expr)
    {
        blocks_[index(test_field, trial_field)] = std::move(expr);
    }

    [[nodiscard]] const FormExpr& block(std::size_t test_field, std::size_t trial_field) const
    {
        return blocks_[index(test_field, trial_field)];
    }

    [[nodiscard]] FormExpr& block(std::size_t test_field, std::size_t trial_field)
    {
        return blocks_[index(test_field, trial_field)];
    }

    [[nodiscard]] bool hasBlock(std::size_t test_field, std::size_t trial_field) const
    {
        return block(test_field, trial_field).isValid();
    }

private:
    [[nodiscard]] std::size_t index(std::size_t test_field, std::size_t trial_field) const
    {
        if (test_field >= num_test_fields_ || trial_field >= num_trial_fields_) {
            throw std::out_of_range("BlockBilinearForm: block index out of range");
        }
        return test_field * num_trial_fields_ + trial_field;
    }

    std::size_t num_test_fields_{0};
    std::size_t num_trial_fields_{0};
    std::vector<FormExpr> blocks_{};
};

class BlockLinearForm {
public:
    explicit BlockLinearForm(std::size_t num_test_fields)
        : num_test_fields_(num_test_fields), blocks_(num_test_fields)
    {
    }

    [[nodiscard]] std::size_t numTestFields() const noexcept { return num_test_fields_; }

    void setBlock(std::size_t test_field, FormExpr expr)
    {
        blocks_[index(test_field)] = std::move(expr);
    }

    [[nodiscard]] const FormExpr& block(std::size_t test_field) const
    {
        return blocks_[index(test_field)];
    }

    [[nodiscard]] FormExpr& block(std::size_t test_field)
    {
        return blocks_[index(test_field)];
    }

    [[nodiscard]] bool hasBlock(std::size_t test_field) const
    {
        return block(test_field).isValid();
    }

private:
    [[nodiscard]] std::size_t index(std::size_t test_field) const
    {
        if (test_field >= num_test_fields_) {
            throw std::out_of_range("BlockLinearForm: block index out of range");
        }
        return test_field;
    }

    std::size_t num_test_fields_{0};
    std::vector<FormExpr> blocks_{};
};

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_BLOCK_FORM_H

