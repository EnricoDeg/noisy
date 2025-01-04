/*
 * @file transformMatrix.hpp
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

#ifndef TRANSFORMMATRIX_HPP_
#define TRANSFORMMATRIX_HPP_

#include <iostream>
#include <cassert>

#include "src/dataStructure/dataStruct.hpp"
#include "src/backend/cpu/backendCPU.hpp"
#ifdef CUDA
#include "src/backend/cuda/backendCUDA.hpp"
#endif

template <typename Tdata, template <class> class  backend>
void downsample(const DSmatrix<Tdata, backend>& inMat ,
                      unsigned int              dim   ,
                      unsigned int              stride,
                      DSmatrix<Tdata, backend>& outMat);

template <typename Tdata, template <class> class  backend>
t_dims upsample(const DSmatrix<Tdata, backend>& inMat  ,
                    unsigned int              dim    ,
                    unsigned int              nzeros ,
                    DSmatrix<Tdata, backend>* outMat = nullptr) {

    t_dims dims = inMat.dims();

    if (outMat == nullptr) {
        t_dims dimsOutCompute;
        if (dim == 0)
            dimsOutCompute = {.rows = (dims.rows-1)*(nzeros)+dims.rows, .cols = dims.cols};
        else
            dimsOutCompute = {.rows = dims.rows, .cols = (dims.cols-1)*(nzeros)+dims.cols};
        return dimsOutCompute;
    }

    t_dims dimsOut = outMat->dims();
    if (dim == 0) {
        assert(dimsOut.rows == (dims.rows-1)*(nzeros)+dims.rows);
    } else {
        assert(dimsOut.cols == (dims.cols-1)*(nzeros)+dims.cols);
    }

    Tdata * __restrict__ inData = inMat.data();
    Tdata * __restrict__ outData = outMat->data();
    backend<Tdata>::transform::upsample(inData, outData, dim, nzeros, dims.rows, dims.cols);

    return dimsOut;
}

template <typename Tdata, template <class> class  backend>
void pad(const DSmatrix<Tdata, backend>& inMat ,
               DSmatrix<Tdata, backend>& outMat);

template <typename Tdata, template <class> class  backend>
void dshear(const DSmatrix<Tdata, backend>& inMat ,
                  DSmatrix<Tdata, backend>& outMat,
                  long int                  k     ,
                  unsigned int              dim   );

template <typename Tdata, template <class> class  backend>
void transpose(const DSmatrix<Tdata, backend>& inMat ,
                     DSmatrix<Tdata, backend>& outMat);

template <typename Tdata, template <class> class  backend>
void normL2(const DSmatrix<Tdata, backend>&  inMat,
                  Tdata                     *out  );

template <typename Tdata, template <class> class  backend,
                          template <class> class  backendC>
struct convolve_impl {

    static t_dims doit(const DSmatrix<Tdata, backend>&  inMat ,
                       const DSmatrix<Tdata, backend>&  filter,
                             DSmatrix<Tdata, backend>*  outMat) {

        t_dims inDims  = inMat.dims();
        t_dims fDims   = filter.dims();
        if (outMat == nullptr) {

            return t_dims{.rows = inDims.rows + fDims.rows - 1,
                        .cols = inDims.cols + fDims.cols - 1};
        }

        t_dims outDims = outMat->dims();
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
                                    outMat->data(),
                                    inDims.rows,
                                    inDims.cols,
                                    fDims.rows,
                                    fDims.cols);
        return outDims;
    }
};

template<typename T, template <class> class  backend> struct convolve_helper;

template<>
struct convolve_helper<float, cpu_impl> {
    using type = convolve_impl<float, cpu_impl, cpu_complex_impl>;
};

template<typename Tdata, template <class> class  backend>
using convolveCaller = typename convolve_helper<Tdata, backend>::type;

template<typename Tdata, template <class> class  backend>
inline
t_dims convolve(const DSmatrix<Tdata, backend>&  inMat ,
                const DSmatrix<Tdata, backend>&  filter,
                      DSmatrix<Tdata, backend>*  outMat = nullptr) {

    return convolveCaller<Tdata, backend>::doit(inMat, filter, outMat);
}

#endif
