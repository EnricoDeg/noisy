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
void upsample(const DSmatrix<Tdata, backend>& inMat  ,
                    unsigned int              dim    ,
                    unsigned int              nzeros ,
                    DSmatrix<Tdata, backend>& outMat ) {

    t_dims dims = inMat.dims();
    t_dims dimsOut = outMat.dims();
    if (dim == 0) {
        assert(dimsOut.rows == (dims.rows-1)*(nzeros)+dims.rows);
    } else {
        assert(dimsOut.cols == (dims.cols-1)*(nzeros)+dims.cols);
    }

    Tdata * __restrict__ inData = inMat.data();
    Tdata * __restrict__ outData = outMat.data();
    backend<Tdata>::transform::upsample(inData, outData, dim, nzeros, dims.rows, dims.cols);
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
