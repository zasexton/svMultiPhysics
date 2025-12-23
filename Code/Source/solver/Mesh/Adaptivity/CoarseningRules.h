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

#ifndef SVMP_COARSENING_RULES_H
#define SVMP_COARSENING_RULES_H

#include "../Core/MeshTypes.h"
#include "Options.h"
#include <memory>
#include <string>
#include <vector>

namespace svmp {

class MeshBase;

/**
 * @brief Coarsening pattern types (placeholder)
 */
enum class CoarseningPattern {
  NONE
};

/**
 * @brief Information about a coarsening operation (placeholder)
 */
struct CoarseningOperation {
  CoarseningPattern pattern = CoarseningPattern::NONE;
  std::vector<index_t> source_cells;
  bool valid = false;
  double predicted_quality = 0.0;
  double priority = 0.0;
};

/**
 * @brief Coarsening history for undoing operations (placeholder)
 */
struct CoarseningHistory {
  CoarseningPattern pattern = CoarseningPattern::NONE;
  size_t operation_id = 0;
};

/**
 * @brief Abstract base class for coarsening rules (placeholder)
 */
class CoarseningRule {
public:
  virtual ~CoarseningRule() = default;

  virtual bool can_coarsen(const MeshBase& mesh,
                           const std::vector<index_t>& cells) const = 0;

  virtual CoarseningOperation determine_coarsening(const MeshBase& mesh,
                                                   const std::vector<index_t>& cells) const = 0;

  virtual CoarseningHistory apply_coarsening(MeshBase& mesh,
                                             const CoarseningOperation& op) const = 0;

  virtual void undo_coarsening(MeshBase& mesh,
                               const CoarseningHistory& history) const = 0;

  virtual std::string name() const = 0;
};

/**
 * @brief No-op coarsening rule used until mesh coarsening is implemented.
 */
class NoopCoarseningRule final : public CoarseningRule {
public:
  bool can_coarsen(const MeshBase& mesh,
                   const std::vector<index_t>& cells) const override;

  CoarseningOperation determine_coarsening(const MeshBase& mesh,
                                           const std::vector<index_t>& cells) const override;

  CoarseningHistory apply_coarsening(MeshBase& mesh,
                                     const CoarseningOperation& op) const override;

  void undo_coarsening(MeshBase& mesh,
                       const CoarseningHistory& history) const override;

  std::string name() const override { return "NoopCoarsening"; }
};

} // namespace svmp

#endif // SVMP_COARSENING_RULES_H

