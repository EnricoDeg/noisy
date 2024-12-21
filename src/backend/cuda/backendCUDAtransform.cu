/*
 * @file backendCUDAtransform.cu
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

#include "src/backend/cuda/backendCUDA.hpp"

#include "cuAlgo.hpp"

template <typename Tdata>
void cuda_impl<Tdata>::transform::downsample(Tdata * __restrict__ inMat,
                                            Tdata * __restrict__ outMat,
                                            unsigned int dim,
                                            unsigned int stride,
                                            unsigned int mRows,
                                            unsigned int mCols) {
    cuAlgo::downsample1dMatrix(inMat ,
                               outMat,
                               dim   ,
                               stride,
                               mRows ,
                               mCols );
}

template <typename Tdata>
void cuda_impl<Tdata>::transform::upsample(Tdata * __restrict__ in,
                                           Tdata * __restrict__ out,
                                           unsigned int  dim   ,
                                           unsigned int  nzeros,
                                           unsigned int  mRows ,
                                           unsigned int  mCols ) {

    cuAlgo::upsample1dMatrix(in    ,
                             out   ,
                             dim   ,
                             nzeros,
                             mRows ,
                             mCols );
}

template <typename Tdata>
void cuda_impl<Tdata>::transform::pad(Tdata * __restrict__ in   ,
                                      Tdata * __restrict__ out  ,
                                      unsigned int         nRows,
                                      unsigned int         nCols,
                                      unsigned int         mRows,
                                      unsigned int         mCols) {

    cuAlgo::padarray2dMatrix(in, out, nRows, nCols, mRows, mCols);
}

template <typename Tdata>
void cuda_impl<Tdata>::transform::dshear(Tdata * __restrict__ inData ,
                                         Tdata * __restrict__ outData,
                                         long int             k      ,
                                         unsigned int         dim    ,
                                         unsigned int         mRows  ,
                                         unsigned int         mCols  ) {

    cuAlgo::dshear1dMatrix(inData, outData, k, dim, mRows, mCols);
}
