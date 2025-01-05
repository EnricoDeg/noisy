/*
 * @file transformMatrix.cpp
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

#include "src/transform/transformMatrix.hpp"
#include "src/backend/cpu/backendCPU.hpp"
#ifdef CUDA
#include "src/backend/cuda/backendCUDA.hpp"
#endif

#include <iostream>
#include <cassert>

template <typename Tdata, template <class> class  backend>
void dshear(const DSmatrix<Tdata, backend>& inMat ,
                  DSmatrix<Tdata, backend>& outMat,
                  long int                  k     ,
                  unsigned int              dim   ) {

    t_dims dims = inMat.dims();
    t_dims dimsOut = outMat.dims();

    assert(dims.rows == dimsOut.rows);
    assert(dims.cols == dimsOut.cols);

    Tdata * __restrict__ inData = inMat.data();
    Tdata * __restrict__ outData = outMat.data();
    backend<Tdata>::transform::dshear(inData, outData,
                                      k, dim,
                                      dims.rows, dims.cols);
}

template <typename Tdata, template <class> class  backend>
void transpose(const DSmatrix<Tdata, backend>& inMat ,
                     DSmatrix<Tdata, backend>& outMat) {

    t_dims dims = inMat.dims();
    t_dims dimsOut = outMat.dims();
    assert(dims.rows == dimsOut.cols);
    assert(dims.cols == dimsOut.rows);
    Tdata * __restrict__ inData = inMat.data();
    Tdata * __restrict__ outData = outMat.data();

    backend<Tdata>::transform::transpose(inData, outData, dims.rows, dims.cols);
}

template <typename Tdata, template <class> class  backend>
void normL2(const DSmatrix<Tdata, backend>&  inMat,
                  Tdata                     *out  ) {

    backend<Tdata>::transform::normL2(inMat.data(), out, inMat.size());
}

// INSTANTIATE

// CPU
template void dshear(const DSmatrix<float, cpu_impl>& inMat ,
                           DSmatrix<float, cpu_impl>& outMat,
                           long int                   k     ,
                           unsigned int               dim   );
template void dshear(const DSmatrix<std::complex<float>, cpu_impl>& inMat ,
                           DSmatrix<std::complex<float>, cpu_impl>& outMat,
                           long int                                 k     ,
                           unsigned int                             dim   );
template void dshear(const DSmatrix<double, cpu_impl>& inMat ,
                           DSmatrix<double, cpu_impl>& outMat,
                           long int                    k     ,
                           unsigned int                dim   );
template void dshear(const DSmatrix<std::complex<double>, cpu_impl>& inMat ,
                           DSmatrix<std::complex<double>, cpu_impl>& outMat,
                           long int                                  k     ,
                           unsigned int                              dim   );

// CUDA
#ifdef CUDA
template void dshear(const DSmatrix<float, cuda_impl>& inMat ,
                           DSmatrix<float, cuda_impl>& outMat,
                           long int                    k     ,
                           unsigned int                dim   );
template void dshear(const DSmatrix<thrust::complex<float>, cuda_impl>& inMat ,
                           DSmatrix<thrust::complex<float>, cuda_impl>& outMat,
                           long int                                     k     ,
                           unsigned int                                 dim   );
template void dshear(const DSmatrix<double, cuda_impl>& inMat ,
                           DSmatrix<double, cuda_impl>& outMat,
                           long int                     k     ,
                           unsigned int                 dim   );
template void dshear(const DSmatrix<thrust::complex<double>, cuda_impl>& inMat ,
                           DSmatrix<thrust::complex<double>, cuda_impl>& outMat,
                           long int                                      k     ,
                           unsigned int                                  dim   );
#endif

// CPU
template void transpose(const DSmatrix<float, cpu_impl>& inMat ,
                              DSmatrix<float, cpu_impl>& outMat);
template void transpose(const DSmatrix<std::complex<float>, cpu_impl>& inMat ,
                              DSmatrix<std::complex<float>, cpu_impl>& outMat);
template void transpose(const DSmatrix<double, cpu_impl>& inMat ,
                              DSmatrix<double, cpu_impl>& outMat);
template void transpose(const DSmatrix<std::complex<double>, cpu_impl>& inMat ,
                              DSmatrix<std::complex<double>, cpu_impl>& outMat);

// CUDA
#ifdef CUDA
template void transpose(const DSmatrix<float, cuda_impl>& inMat ,
                              DSmatrix<float, cuda_impl>& outMat);
template void transpose(const DSmatrix<thrust::complex<float>, cuda_impl>& inMat ,
                              DSmatrix<thrust::complex<float>, cuda_impl>& outMat);
template void transpose(const DSmatrix<double, cuda_impl>& inMat ,
                              DSmatrix<double, cuda_impl>& outMat);
template void transpose(const DSmatrix<thrust::complex<double>, cuda_impl>& inMat ,
                              DSmatrix<thrust::complex<double>, cuda_impl>& outMat);
#endif

// CPU
template void normL2(const DSmatrix<float, cpu_impl>&  inMat,
                           float                      *out  );
template void normL2(const DSmatrix<std::complex<float>, cpu_impl>&  inMat,
                           std::complex<float>                      *out  );
template void normL2(const DSmatrix<double, cpu_impl>&  inMat,
                           double                      *out  );
template void normL2(const DSmatrix<std::complex<double>, cpu_impl>&  inMat,
                           std::complex<double>                      *out  );

// CUDA
#ifdef CUDA
template void normL2(const DSmatrix<float, cuda_impl>&  inMat,
                           float                       *out  );
template void normL2(const DSmatrix<thrust::complex<float>, cuda_impl>&  inMat,
                           thrust::complex<float>                       *out  );
template void normL2(const DSmatrix<double, cuda_impl>&  inMat,
                           double                       *out  );
template void normL2(const DSmatrix<thrust::complex<double>, cuda_impl>&  inMat,
                           thrust::complex<double>                       *out  );
#endif
