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

#include "CellShape.h"
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace svmp {

// Static storage for the registry
std::unordered_map<std::string, std::unordered_map<int, CellShape>>&
CellShapeRegistry::map_() {
  static std::unordered_map<std::string, std::unordered_map<int, CellShape>> registry;
  return registry;
}

void CellShapeRegistry::register_shape(const std::string& format, int type_id, const CellShape& shape) {
  map_()[format][type_id] = shape;
}

bool CellShapeRegistry::has(const std::string& format, int type_id) {
  auto& registry = map_();
  auto format_it = registry.find(format);
  if (format_it == registry.end()) {
    return false;
  }
  return format_it->second.find(type_id) != format_it->second.end();
}

CellShape CellShapeRegistry::get(const std::string& format, int type_id) {
  auto& registry = map_();
  auto format_it = registry.find(format);
  if (format_it == registry.end()) {
    throw std::runtime_error("CellShapeRegistry: Format '" + format + "' not registered");
  }

  auto type_it = format_it->second.find(type_id);
  if (type_it == format_it->second.end()) {
    throw std::runtime_error("CellShapeRegistry: Type ID " + std::to_string(type_id) +
                           " not registered for format '" + format + "'");
  }

  return type_it->second;
}

void CellShapeRegistry::clear_format(const std::string& format) {
  map_().erase(format);
}

void CellShapeRegistry::clear_all() {
  map_().clear();
}

std::vector<std::string> CellShapeRegistry::formats() {
  std::vector<std::string> result;
  for (const auto& [format, _] : map_()) {
    result.push_back(format);
  }
  std::sort(result.begin(), result.end());
  return result;
}

} // namespace svmp