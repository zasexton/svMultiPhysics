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

#ifndef SVMP_MOTION_FIELDS_H
#define SVMP_MOTION_FIELDS_H

/**
 * @file MotionFields.h
 * @brief Mesh-attached fields for mesh motion (displacement, velocity).
 *
 * This header declares lightweight helpers for attaching and updating
 * mesh-motion related fields on top of MeshBase. It operates purely on
 * geometry and Mesh's field system, without introducing FE/DOF concepts.
 */

#include "../Core/MeshTypes.h"

namespace svmp {

class MeshBase;

namespace motion {

/**
 * @brief Handles to mesh-motion related fields attached to a mesh.
 *
 * Both fields live on vertices:
 *  - displacement: u       [units: length] (used as a step increment by MeshMotion)
 *  - velocity:     w(X,t)  [units: length/time]
 */
struct MotionFieldHandles {
  FieldHandle displacement;
  FieldHandle velocity;
};

/**
 * @brief Attach (or lookup) standard mesh-motion fields on a mesh.
 *
 * This function ensures that displacement and velocity fields exist on
 * the given mesh as vertex-attached Float64 arrays with @p nsd components.
 * If the fields already exist, their handles are returned without
 * modifying existing data.
 *
 * @param mesh Mesh on which to attach motion fields.
 * @param nsd  Spatial dimension (typically 2 or 3).
 * @return Handles to the displacement and velocity fields.
 *
 * @throws std::runtime_error if @p nsd is incompatible with the mesh.
 */
MotionFieldHandles attach_motion_fields(MeshBase& mesh, int nsd);

/**
 * @brief Update mesh coordinates from a displacement field.
 *
 * The displacement field is interpreted in physical space as one vector
 * per vertex. When @p accumulate is false,
 * the new coordinates are computed as
 *
 *   X_cur = X_ref + u
 *
 * using the reference configuration as the base. When @p accumulate
 * is true and a current coordinate field already exists, the update
 * becomes
 *
 *   X_cur_new = X_cur_old + u
 *
 * which is useful for incremental motion schemes; if no current
 * coordinates exist yet, the reference configuration is used as the
 * base instead.
 *
 * This function only updates geometry; it does not modify the
 * displacement or velocity fields themselves.
 *
 * @param mesh       Mesh whose coordinates will be updated.
 * @param hnd        Handles to motion fields (displacement is required).
 * @param accumulate If true, displacement is applied on top of the
 *                   current configuration when available.
 */
void update_coordinates_from_displacement(MeshBase& mesh,
                                          const MotionFieldHandles& hnd,
                                          bool accumulate);

} // namespace motion
} // namespace svmp

#endif // SVMP_MOTION_FIELDS_H
