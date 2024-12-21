/*
 * @file backendCUDA.hpp
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

#ifndef BACKENDCUDA_HPP_
#define BACKENDCUDA_HPP_

#include "src/backend/cuda/backendCUDAutils.hpp"
#include "src/backend/cuda/backendCUDAparams.hpp"
#include "src/backend/cuda/backendCUDAfourier.hpp"
#include "thrust/complex.h"

template <typename Tdata>
class cuda_impl {
 public:

    class memory;
    class op;
    class transform;
    using complex = thrust::complex<Tdata>;
};

template <typename Tdata>
class cuda_fft_impl {
 public:

    using fourier = typename cuda::details::fourier_helper<Tdata>::type;
};

template<typename Tdata>
class cuda_impl<Tdata>::memory {

public:
    static Tdata * allocate(unsigned int elements) ;
    static void free(Tdata *data) ;
    static void copy(Tdata* dst, Tdata *src, unsigned int size);
    static void fill(Tdata * __restrict__ data, unsigned int size, Tdata value);
};

template<typename Tdata>
class cuda_impl<Tdata>::op {

public:
    static void normalize(Tdata * __restrict__ data, unsigned int size);
    static void fliplr(Tdata * __restrict__ data, unsigned int dim,
                       unsigned int mRows, unsigned int mCols);
    static void sumInPlace(Tdata * __restrict__ data1,
                           const Tdata * __restrict__ data2,
                           unsigned int size);
    static void prodInPlace(Tdata * __restrict__ data1,
                           const Tdata * __restrict__ data2,
                           unsigned int size);
    static void divScalarInPlace(Tdata * __restrict__ data ,
                                 unsigned int         size ,
                                 Tdata                value);
};

template<typename Tdata>
class cuda_impl<Tdata>::transform {

public:
    static void downsample(Tdata * __restrict__ inMat,
                           Tdata * __restrict__ outMat,
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
    static void pad(Tdata * __restrict__ in   ,
                    Tdata * __restrict__ out  ,
                    unsigned int         nRows,
                    unsigned int         nCols,
                    unsigned int         mRows,
                    unsigned int         mCols);
    static void dshear(Tdata * __restrict__ inData ,
                       Tdata * __restrict__ outData,
                       long int             k      ,
                       unsigned int         dim    ,
                       unsigned int         mRows  ,
                       unsigned int         mCols  );
    static void transpose(Tdata * __restrict__ inData ,
                          Tdata * __restrict__ outData,
                          unsigned int         mRows  ,
                          unsigned int         mCols  );
    static void normL2(Tdata * __restrict__ inData ,
                       Tdata * __restrict__ outData,
                       unsigned int         size   );
};

template <typename Tdata>
class cuda_complex_impl {
public:

    class op;
    using complex = thrust::complex<Tdata>;
};

template <typename Tdata>
class cuda_complex_impl<Tdata>::op {

public:
    static void corrComplex(thrust::complex<Tdata> * __restrict__ dataIn1,
                            thrust::complex<Tdata> * __restrict__ dataIn2,
                            thrust::complex<Tdata> * __restrict__ dataOut,
                            unsigned int size);
    static void convComplex(thrust::complex<Tdata> * __restrict__ dataIn1,
                            thrust::complex<Tdata> * __restrict__ dataIn2,
                            thrust::complex<Tdata> * __restrict__ dataOut,
                            unsigned int size);
};

template class cuda_impl<float>;
template class cuda_impl<thrust::complex<float>>;
template class cuda_fft_impl<float>;

template class cuda_complex_impl<float>;

#endif
