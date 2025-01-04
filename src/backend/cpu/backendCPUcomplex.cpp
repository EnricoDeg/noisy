/*
 * @file backendCPUcomplex.cpp
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
#include "src/utils/utils.hpp"

#include <cmath>
#include <cassert>
#include <cstring>

template <typename Tdata>
void cpu_complex_impl<Tdata>::op::corrComplex(std::complex<Tdata> * __restrict__ dataIn1,
                                              std::complex<Tdata> * __restrict__ dataIn2,
                                              std::complex<Tdata> * __restrict__ dataOut,
                                              unsigned int size) {

    for (unsigned int i = 0; i < size; ++i)
        dataOut[i] = dataIn1[i] * std::conj(dataIn2[i]);
}

template <typename Tdata>
void cpu_complex_impl<Tdata>::op::convComplex(std::complex<Tdata> * __restrict__ dataIn1,
                                              std::complex<Tdata> * __restrict__ dataIn2,
                                              std::complex<Tdata> * __restrict__ dataOut,
                                              unsigned int size) {

    for (unsigned int i = 0; i < size; ++i)
        dataOut[i] = dataIn1[i] * dataIn2[i];
}

template <typename Tdata>
void cpu_complex_impl<Tdata>::op::padMatrix(Tdata * __restrict__ dataIn ,
                                            Tdata * __restrict__ dataOut,
                                            unsigned int         inRows ,
                                            unsigned int         inCols ,
                                            unsigned int         outRows,
                                            unsigned int         outCols) {

    for (unsigned int i = 0; i < outRows; ++i) {
        Tdata * __restrict__ out = dataOut + i*outCols;
        for (unsigned int j = 0; j < outCols; ++j) {
            out[j] = 0;
        }
    }

    for (size_t i = 0; i < inRows; ++i) {

        const Tdata * __restrict__ in  = dataIn  + i * inCols ;
        Tdata * __restrict__ out = dataOut + i*outCols;
        std::memcpy(out, in, inCols * sizeof(Tdata));
    }
}

// dataIn and filter are assumed to be padded
template <typename Tdata>
void cpu_complex_impl<Tdata>::op::convData(Tdata * __restrict__ dataIn ,
                                           Tdata * __restrict__ filter ,
                                           Tdata * __restrict__ dataOut,
                                           unsigned int         mRows  ,
                                           unsigned int         mCols  ,
                                           unsigned int         fRows  ,
                                           unsigned int         fCols  ) {

    unsigned int rows = mRows + fRows - 1;
    unsigned int cols = mCols + fCols - 1;

    for (unsigned int i = 0; i < rows; ++i) {
        Tdata * __restrict__ out = dataOut + i*cols;
        for (unsigned int j = 0; j < cols; ++j) {
            out[j] = 0;
        }
    }

    for (unsigned int nr = 0; nr < rows; ++nr) {
        unsigned int low_mr = std::max((int)0, (int)nr - (int)fRows + 1);
        unsigned int high_mr = std::min((int)mRows - 1, (int)nr);
        for (unsigned int nc = 0; nc < cols; ++nc) {
            unsigned int low_mc = std::max((int)0, (int)nc - (int)fCols + 1);
            unsigned int high_mc = std::min((int)mCols - 1 , (int)nc);
            Tdata tmp = 0;
            for (unsigned int mr = low_mr; mr <= high_mr; ++mr)
                for (unsigned int mc = low_mc; mc <= high_mc; ++mc)
                    tmp += dataIn[mr * cols + mc] * filter[(nr-mr) * cols + (nc-mc)];
            dataOut[nr * cols + nc] = tmp;
        }
    }
}

template <typename Tdata>
void cpu_complex_impl<Tdata>::op::real2complex(Tdata               * __restrict__ dataIn ,
                                               std::complex<Tdata> * __restrict__ dataOut,
                                               unsigned int                       mRows  ,
                                               unsigned int                       mCols  ) {

    for (unsigned int i = 0; i < mRows * mCols; ++i)
        dataOut[i] = {dataIn[i], 0};
}