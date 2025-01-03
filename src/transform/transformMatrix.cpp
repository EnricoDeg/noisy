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
void downsample(const DSmatrix<Tdata, backend>& inMat  ,
                      unsigned int              dim    ,
                      unsigned int              stride ,
                      DSmatrix<Tdata, backend>& outMat ) {

    t_dims dims = inMat.dims();
    if (dim == 0) {
        unsigned int count = 0;
        for (unsigned int i = 0; i < dims.rows; i+=stride, ++count);
        t_dims dimsOut = outMat.dims();
        assert(dimsOut.rows == count);
    } else {
        unsigned int count = 0;
        for (unsigned int i = 0; i < dims.cols; i+=stride, ++count);
        t_dims dimsOut = outMat.dims();
        assert(dimsOut.cols == count);
    }

    Tdata * __restrict__ inData = inMat.data();
    Tdata * __restrict__ outData = outMat.data();
    backend<Tdata>::transform::downsample(inData, outData, dim, stride, dims.rows, dims.cols);
}

template <typename Tdata, template <class> class  backend>
t_dims upsample(const DSmatrix<Tdata, backend>& inMat  ,
                    unsigned int              dim    ,
                    unsigned int              nzeros ,
                    DSmatrix<Tdata, backend>& outMat ) {

    t_dims dims = inMat.dims();

    if (outMat.is_empty()) {\
        std::cout << "empty output matrix, return only dimensions" << std::endl;
        t_dims dimsOutCompute;
        if (dim == 0)
            dimsOutCompute = {.rows = (dims.rows-1)*(nzeros)+dims.rows, .cols = dims.cols};
        else
            dimsOutCompute = {.rows = dims.rows, .cols = (dims.cols-1)*(nzeros)+dims.cols};
        return dimsOutCompute;
    }

    t_dims dimsOut = outMat.dims();
    if (dim == 0) {
        assert(dimsOut.rows == (dims.rows-1)*(nzeros)+dims.rows);
    } else {
        assert(dimsOut.cols == (dims.cols-1)*(nzeros)+dims.cols);
    }

    Tdata * __restrict__ inData = inMat.data();
    Tdata * __restrict__ outData = outMat.data();
    backend<Tdata>::transform::upsample(inData, outData, dim, nzeros, dims.rows, dims.cols);

    return dimsOut;
}

template <typename Tdata, template <class> class  backend>
t_dims upsample(const DSmatrix<Tdata, backend>& inMat  ,
                      unsigned int              dim    ,
                      unsigned int              nzeros ) {

    DSmatrix<Tdata, backend> emptyMat{};
    return upsample(inMat, dim, nzeros, emptyMat);
}

template <typename Tdata, template <class> class  backend>
void pad(const DSmatrix<Tdata, backend>& inMat ,
               DSmatrix<Tdata, backend>& outMat) {

    t_dims dims = inMat.dims();
    t_dims dimsOut = outMat.dims();

    Tdata * __restrict__ inData = inMat.data();
    Tdata * __restrict__ outData = outMat.data();
    backend<Tdata>::transform::pad(inData, outData,
                                   dimsOut.rows, dimsOut.cols,
                                   dims.rows, dims.cols);
}

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

template <typename Tdata, template <class> class  backend,
                          template <class> class  backendC>
void convolve(const DSmatrix<Tdata, backend>&  inMat ,
              const DSmatrix<Tdata, backend>&  filter,
                    DSmatrix<Tdata, backend>&  outMat) {

    t_dims inDims  = inMat.dims();
    t_dims fDims   = filter.dims();
    t_dims outDims = outMat.dims();
    assert(outDims.rows == inDims.rows + fDims.rows - 1);
    assert(outDims.cols == inDims.cols + fDims.cols - 1);

    DSmatrix<Tdata, backend> inMatPadded(outDims.rows, outDims.cols);
    backendC<Tdata>::op::padMatrix(inMat.data(),
                                   inMatPadded.data(),
                                   inDims.rows,
                                   inDims.cols,
                                   outDims.rows,
                                   outDims.cols);

    DSmatrix<Tdata, backend> filterPadded(outDims.rows, outDims.cols);
    backendC<Tdata>::op::padMatrix(filter.data(),
                                   filterPadded.data(),
                                   fDims.rows,
                                   fDims.cols,
                                   outDims.rows,
                                   outDims.cols);

    backendC<Tdata>::op::convData(inMatPadded.data(),
                                  filterPadded.data(),
                                  outMat.data(),
                                  inDims.rows,
                                  inDims.cols,
                                  fDims.rows,
                                  fDims.cols);
}

// INSTANTIATE

template void convolve<float, cpu_impl, cpu_complex_impl>
                     (const DSmatrix<float, cpu_impl>&  inMat ,
                      const DSmatrix<float, cpu_impl>&  filter,
                            DSmatrix<float, cpu_impl>&  outMat);

// CPU
template void downsample(const DSmatrix<float, cpu_impl>& inMat  ,
                               unsigned int               dim    ,
                               unsigned int               stride ,
                               DSmatrix<float, cpu_impl>& outMat );
template void downsample(const DSmatrix<std::complex<float>, cpu_impl>& inMat  ,
                               unsigned int                             dim    ,
                               unsigned int                             stride ,
                               DSmatrix<std::complex<float>, cpu_impl>& outMat );
template void downsample(const DSmatrix<double, cpu_impl>& inMat  ,
                               unsigned int                dim    ,
                               unsigned int                stride ,
                               DSmatrix<double, cpu_impl>& outMat );
template void downsample(const DSmatrix<std::complex<double>, cpu_impl>& inMat  ,
                               unsigned int                              dim    ,
                               unsigned int                              stride ,
                               DSmatrix<std::complex<double>, cpu_impl>& outMat );
// CUDA
#ifdef CUDA
template void downsample(const DSmatrix<float, cuda_impl>& inMat  ,
                               unsigned int                dim    ,
                               unsigned int                stride ,
                               DSmatrix<float, cuda_impl>& outMat );
template void downsample(const DSmatrix<thrust::complex<float>, cuda_impl>& inMat  ,
                               unsigned int                                 dim    ,
                               unsigned int                                 stride ,
                               DSmatrix<thrust::complex<float>, cuda_impl>& outMat );
template void downsample(const DSmatrix<double, cuda_impl>& inMat  ,
                               unsigned int                 dim    ,
                               unsigned int                 stride ,
                               DSmatrix<double, cuda_impl>& outMat );
template void downsample(const DSmatrix<thrust::complex<double>, cuda_impl>& inMat  ,
                               unsigned int                                  dim    ,
                               unsigned int                                  stride ,
                               DSmatrix<thrust::complex<double>, cuda_impl>& outMat );
#endif

// CPU
template t_dims upsample(const DSmatrix<float, cpu_impl>& inMat  ,
                               unsigned int               dim    ,
                               unsigned int               nzeros ,
                               DSmatrix<float, cpu_impl>& outMat );
template t_dims upsample(const DSmatrix<std::complex<float>, cpu_impl>& inMat  ,
                               unsigned int                             dim    ,
                               unsigned int                             nzeros ,
                               DSmatrix<std::complex<float>, cpu_impl>& outMat );
template t_dims upsample(const DSmatrix<double, cpu_impl>& inMat  ,
                               unsigned int                dim    ,
                               unsigned int                nzeros ,
                               DSmatrix<double, cpu_impl>& outMat );
template t_dims upsample(const DSmatrix<std::complex<double>, cpu_impl>& inMat  ,
                               unsigned int                              dim    ,
                               unsigned int                              nzeros ,
                               DSmatrix<std::complex<double>, cpu_impl>& outMat );

template t_dims upsample(const DSmatrix<float, cpu_impl>& inMat  ,
                               unsigned int               dim    ,
                               unsigned int               nzeros );

// CUDA
#ifdef CUDA
template t_dims upsample(const DSmatrix<float, cuda_impl>& inMat  ,
                             unsigned int                dim    ,
                             unsigned int                nzeros ,
                             DSmatrix<float, cuda_impl>& outMat );
template t_dims upsample(const DSmatrix<thrust::complex<float>, cuda_impl>& inMat  ,
                             unsigned int                                 dim    ,
                             unsigned int                                 nzeros ,
                             DSmatrix<thrust::complex<float>, cuda_impl>& outMat );
template t_dims upsample(const DSmatrix<double, cuda_impl>& inMat  ,
                             unsigned int                 dim    ,
                             unsigned int                 nzeros ,
                             DSmatrix<double, cuda_impl>& outMat );
template t_dims upsample(const DSmatrix<thrust::complex<double>, cuda_impl>& inMat  ,
                             unsigned int                                  dim    ,
                             unsigned int                                  nzeros ,
                             DSmatrix<thrust::complex<double>, cuda_impl>& outMat );
#endif

// CPU
template void pad(const DSmatrix<float, cpu_impl>& inMat ,
                        DSmatrix<float, cpu_impl>& outMat);
template void pad(const DSmatrix<std::complex<float>, cpu_impl>& inMat ,
                        DSmatrix<std::complex<float>, cpu_impl>& outMat);
template void pad(const DSmatrix<double, cpu_impl>& inMat ,
                        DSmatrix<double, cpu_impl>& outMat);
template void pad(const DSmatrix<std::complex<double>, cpu_impl>& inMat ,
                        DSmatrix<std::complex<double>, cpu_impl>& outMat);

// CUDA
#ifdef CUDA
template void pad(const DSmatrix<float, cuda_impl>& inMat ,
                        DSmatrix<float, cuda_impl>& outMat);
template void pad(const DSmatrix<thrust::complex<float>, cuda_impl>& inMat ,
                        DSmatrix<thrust::complex<float>, cuda_impl>& outMat);
template void pad(const DSmatrix<double, cuda_impl>& inMat ,
                        DSmatrix<double, cuda_impl>& outMat);
template void pad(const DSmatrix<thrust::complex<double>, cuda_impl>& inMat ,
                        DSmatrix<thrust::complex<double>, cuda_impl>& outMat);
#endif

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
