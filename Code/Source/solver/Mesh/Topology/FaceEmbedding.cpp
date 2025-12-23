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

#include "FaceEmbedding.h"

#include "CellTopology.h"

#include <stdexcept>

namespace svmp {

namespace {

FaceParamPoint cell_corner_param(CellFamily family, int corner_id) {
  switch (family) {
    case CellFamily::Tetra: {
      switch (corner_id) {
        case 0: return {0, 0, 0};
        case 1: return {1, 0, 0};
        case 2: return {0, 1, 0};
        case 3: return {0, 0, 1};
        default: return {0, 0, 0};
      }
    }
    case CellFamily::Hex: {
      switch (corner_id) {
        case 0: return {-1, -1, -1};
        case 1: return {1, -1, -1};
        case 2: return {1, 1, -1};
        case 3: return {-1, 1, -1};
        case 4: return {-1, -1, 1};
        case 5: return {1, -1, 1};
        case 6: return {1, 1, 1};
        case 7: return {-1, 1, 1};
        default: return {0, 0, 0};
      }
    }
    case CellFamily::Wedge: {
      switch (corner_id) {
        case 0: return {0, 0, -1};
        case 1: return {1, 0, -1};
        case 2: return {0, 1, -1};
        case 3: return {0, 0, 1};
        case 4: return {1, 0, 1};
        case 5: return {0, 1, 1};
        default: return {0, 0, 0};
      }
    }
    case CellFamily::Pyramid: {
      switch (corner_id) {
        case 0: return {-1, -1, 0};
        case 1: return {1, -1, 0};
        case 2: return {1, 1, 0};
        case 3: return {-1, 1, 0};
        case 4: return {0, 0, 1};
        default: return {0, 0, 0};
      }
    }
    default:
      return {0, 0, 0};
  }
}

FaceParamPoint add3(const FaceParamPoint& a, const FaceParamPoint& b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

FaceParamPoint scale3(const FaceParamPoint& a, real_t s) {
  return {a[0] * s, a[1] * s, a[2] * s};
}

} // namespace

FaceParamPoint FaceEmbedding::embed_face_point(CellFamily cell_family,
                                               int local_face_id,
                                               const FaceParamPoint& xi_face) {
  const auto view = CellTopology::get_oriented_boundary_faces_view(cell_family);
  if (local_face_id < 0 || local_face_id >= view.face_count) {
    throw std::out_of_range("FaceEmbedding::embed_face_point: invalid local face id");
  }

  const int b = view.offsets[local_face_id];
  const int e = view.offsets[local_face_id + 1];
  const int fv = e - b;
  if (fv != 3 && fv != 4) {
    throw std::runtime_error("FaceEmbedding::embed_face_point: unsupported face arity");
  }

  if (fv == 3) {
    const real_t r = xi_face[0];
    const real_t s = xi_face[1];
    const real_t w0 = 1.0 - r - s;
    const real_t w1 = r;
    const real_t w2 = s;

    const int v0 = view.indices[b + 0];
    const int v1 = view.indices[b + 1];
    const int v2 = view.indices[b + 2];

    const auto A = cell_corner_param(cell_family, v0);
    const auto B = cell_corner_param(cell_family, v1);
    const auto C = cell_corner_param(cell_family, v2);

    FaceParamPoint out = {0, 0, 0};
    out = add3(out, scale3(A, w0));
    out = add3(out, scale3(B, w1));
    out = add3(out, scale3(C, w2));
    return out;
  }

  // fv == 4
  const real_t u = xi_face[0];
  const real_t v = xi_face[1];
  const real_t w0 = 0.25 * (1 - u) * (1 - v);
  const real_t w1 = 0.25 * (1 + u) * (1 - v);
  const real_t w2 = 0.25 * (1 + u) * (1 + v);
  const real_t w3 = 0.25 * (1 - u) * (1 + v);

  const int v0 = view.indices[b + 0];
  const int v1 = view.indices[b + 1];
  const int v2 = view.indices[b + 2];
  const int v3 = view.indices[b + 3];

  const auto A = cell_corner_param(cell_family, v0);
  const auto B = cell_corner_param(cell_family, v1);
  const auto C = cell_corner_param(cell_family, v2);
  const auto D = cell_corner_param(cell_family, v3);

  FaceParamPoint out = {0, 0, 0};
  out = add3(out, scale3(A, w0));
  out = add3(out, scale3(B, w1));
  out = add3(out, scale3(C, w2));
  out = add3(out, scale3(D, w3));
  return out;
}

} // namespace svmp
