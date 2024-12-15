/*
 * @file backendCPU.hpp
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

#ifndef BACKENDCPU_HPP_
#define BACKENDCPU_HPP_

#include <complex>

#include "src/backend/cpu/backendCPUfourier.hpp"

template <typename Tdata>
class cpu_impl {
public:

    class memory;
    class op;
    class transform;
    using complex = std::complex<Tdata>;
};

template<typename Tdata>
class cpu_fft_impl {
public:

    using fourier = typename cpu::details::fourier_helper<Tdata>::type;
};

template <typename Tdata>
class cpu_impl<Tdata>::memory {

public:
    static Tdata * allocate(unsigned int elements);
    static void free(Tdata *data);
    static void copy(Tdata* dst, Tdata *src, unsigned int size);
    static void fill(Tdata * __restrict__ data, unsigned int size, Tdata value);
};

template <typename Tdata>
class cpu_impl<Tdata>::op {

public:
    static void normalize(Tdata * __restrict__ data, unsigned int size);

    static void fliplr(Tdata * __restrict__ data, unsigned int dim,
                       unsigned int mRows, unsigned int mCols);
};

template <typename Tdata>
class cpu_impl<Tdata>::transform {

public:
    static void downsample(Tdata * __restrict__ in,
                           Tdata * __restrict__ out,
                           unsigned int dim,
                           unsigned int stride,
                           unsigned int mRows,
                           unsigned int mCols);

    static void upsample(Tdata * __restrict__ in,
                         Tdata * __restrict__ out,
                         unsigned int  dim   ,
                         unsigned int  nzeros,
                         unsigned int  mRows ,
                         unsigned int  mCols );
};

template class cpu_impl<float>;
template class cpu_impl<std::complex<float>>;
template class cpu_fft_impl<float>;
#endif
