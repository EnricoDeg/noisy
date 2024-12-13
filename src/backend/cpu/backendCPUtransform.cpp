/*
 * @file backendCPUtransform.cpp
 *
 * @copyright Copyright (C) 2024 Enrico Degregori <enrico.degregori@gmail.com>
 *
 * @author Enrico Degregori <enrico.degregori@gmail.com>
 * 
 * MIT License
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions: 
 * 
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include "src/backend/cpu/backendCPU.hpp"

#include <cmath>
#include <cassert>
#include <cstring>

template <typename Tdata>
void cpu_impl<Tdata>::transform::downsample(Tdata * __restrict__ inMat,
                                            Tdata * __restrict__ outMat,
                                            unsigned int dim,
                                            unsigned int stride,
                                            unsigned int mRows,
                                            unsigned int mCols) {

    assert(dim == 0 || dim == 1);

    if (dim == 0) {

        size_t count = 0;
        for (size_t i = 0; i < mRows; i+=stride, ++count);

        if (stride == 1) {
            std::memcpy(outMat, inMat, mRows*mCols*sizeof(Tdata));
            return;
        }

        for (unsigned int i = 0; i < count; ++i) {
            const Tdata * __restrict in = inMat + (i*stride)*mCols;
            Tdata * __restrict out = outMat + i*mCols;
            for (unsigned int j = 0; j < mCols; ++j)
                out[j] = in[j];
        }

    } else if (dim == 1) {

        size_t count = 0;
        for (size_t i = 0; i < mCols; i+=stride, ++count);

        if (stride == 1) {
            std::memcpy(outMat, inMat, mRows*mCols*sizeof(Tdata));
            return;
        }

        for (unsigned int i = 0; i < mRows; ++i) {
            const Tdata * __restrict in = inMat + i*mCols;
            Tdata * __restrict out = outMat + i*count;
            for (size_t j = 0; j < count; ++j)
                out[j] = in[j*stride];
        }
    }
}

template <typename Tdata>
void cpu_impl<Tdata>::transform::upsample(Tdata * __restrict__ inMat,
                                          Tdata * __restrict__ outMat,
                                          unsigned int  dim   ,
                                          unsigned int  nzeros,
                                          unsigned int  mRows ,
                                          unsigned int  mCols ) {

    assert(dim == 0 || dim == 1);

    if (dim == 0) {

        if (nzeros == 0) {
            std::memcpy(outMat, inMat, mRows*mCols*sizeof(Tdata));
            return;
        }

        for (unsigned int i = 0; i < mRows; ++i) {
            const Tdata * __restrict in = inMat + i*mCols;
            Tdata * __restrict out = outMat + i*(nzeros+1)*mCols;
            for (size_t j = 0; j < mCols; ++j)
                out[j] = in[j];
        }
        return;
    } else if (dim == 1) {

        unsigned int uCols = (mCols-1)*(nzeros)+mCols;

        if (nzeros == 0) {
            std::memcpy(outMat, inMat, mRows*mCols*sizeof(Tdata));
            return;
        }

        for (unsigned int i = 0; i < mRows; ++i) {
            const Tdata * __restrict in = inMat + i*mCols;
            Tdata * __restrict out = outMat + i*uCols;
            for (size_t j = 0, k=0; j < mCols; ++j, k+=(nzeros+1))
                out[k] = in[j];
        }
        return;
    }
}
