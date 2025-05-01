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

#ifndef SV_TOP_TEST_READ_FILES_H
#define SV_TOP_TEST_READ_FILES_H

#include <chrono>
#include <random>
#include <stdlib.h>

#include "gtest/gtest.h"
#include "read_files.h"
#include "ComMod.h"
#include "utils.h"
#include "Array.h"
#include "Vector.h"

// --------------------------------------------------------------
// ---------------------- Helper functions ----------------------
// --------------------------------------------------------------

/**
 * @brief
 *
 * @param[out]
 * @return
 */
class Test_FaceMatch : public ::testing::Test {
protected:
    ComMod com_mod;
    int nsd;

    void SetUp() override {
        // Called before each test
        nsd = 2; // default to 2D
        com_mod.nsd = nsd;
    }

    void TearDown() override {
        // Called after each test
    }

    // Helper to create a faceType with given coordinates
    faceType create_face(const std::string& name, int nsd_, const std::vector<std::vector<double>>& coords) {
        faceType face;
        face.name = name;
        face.nNo = static_cast<int>(coords.size());
        face.x.resize(nsd_, face.nNo);
        for (int a = 0; a < face.nNo; ++a)
            for (int i = 0; i < nsd_; ++i)
                face.x(i, a) = coords[a][i];
        return face;
    }
};
#endif //SV_TOP_TEST_READ_FILES_H
