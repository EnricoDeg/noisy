/*
 * @file backendCPUop.cpp
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

template <typename Tdata>
void cpu_impl<Tdata>::op::normalize(Tdata * __restrict__ data, unsigned int size) {

    Tdata tmp = 0;
    for (unsigned int i = 0; i < size; ++i)
        tmp += std::abs(data[i]);

    for (unsigned int i = 0; i < size; ++i)
        data[i] /= tmp;
}

template <typename Tdata>
void cpu_impl<Tdata>::op::fliplr(Tdata * __restrict__ data, unsigned int dim,
                                 unsigned int mRows, unsigned int mCols) {

    assert(dim == 0 || dim == 1);

    if (dim == 0) {

        unsigned int hRows = mRows % 2 == 0 ? mRows / 2 : (mRows + 1) / 2;
        for (unsigned int i = 0; i < hRows; ++i) {

            Tdata * __restrict in1 = data + i*mCols;
            Tdata * __restrict in2 = data + (mRows - 1 - i)*mCols;
            for (unsigned int j = 0; j < mCols; ++j)
                _swap(in1 + j, in2 + j);
        }

    } else if (dim == 1) {

        unsigned int hCols = mCols % 2 == 0 ? mCols / 2 : (mCols + 1) / 2;
        for (unsigned int i = 0; i < mRows; ++i) {

            Tdata * __restrict in1 = data + i*mCols;
            Tdata * __restrict in2 = data + i*mCols + mCols - 1;
            for (unsigned int j = 0; j < hCols; ++j)
                _swap(in1 + j, in2 - j);
        }
    }
}

template <typename Tdata>
void cpu_impl<Tdata>::op::sumInPlace(Tdata * __restrict__ data1,
                                     const Tdata * __restrict__ data2,
                                     unsigned int size) {

    for (unsigned int i = 0; i < size; ++i)
        data1[i] += data2[i];
}

template <typename Tdata>
void cpu_impl<Tdata>::op::prodInPlace(Tdata * __restrict__ data1,
                                     const Tdata * __restrict__ data2,
                                     unsigned int size) {

    for (unsigned int i = 0; i < size; ++i)
        data1[i] *= data2[i];
}

template <typename Tdata>
void cpu_impl<Tdata>::op::divScalarInPlace(Tdata * __restrict__ data ,
                                           unsigned int         size ,
                                           Tdata                value) {

    for (unsigned int i = 0; i < size; ++i)
        data[i] /= value;
}

template <typename Tdata>
void cpu_impl<Tdata>::op::prodScalarInPlace(Tdata * __restrict__ data ,
                                            unsigned int         size ,
                                            Tdata                value) {

    for (unsigned int i = 0; i < size; ++i)
        data[i] *= value;
}

template <typename Tdata>
void cpu_impl<Tdata>::op::mirror(Tdata * __restrict__ inData ,
                                 Tdata * __restrict__ outData,
                                 unsigned int         size   ) {

    for (size_t i = 0; i < size; ++i)
        outData[i] = inData[i] * Tdata(std::pow(-1.0, i));
}

template <typename Tdata>
void cpu_impl<Tdata>::op::applyThreshold(Tdata        * __restrict__ data     ,
                                         Tdata                       threshold,
                                         unsigned int                size     ) {

    for (unsigned int i = 0; i < size; ++i)
        if (std::abs(data[i]) < std::abs(threshold))
            data[i] = 0 ;
}
