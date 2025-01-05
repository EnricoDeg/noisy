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

        {
            const Tdata * __restrict in = inMat;
            Tdata * __restrict out = outMat;
            for (unsigned int j = 0; j < mCols; ++j)
                out[j] = in[j];
        }

        for (unsigned int i = 1; i < mRows; ++i) {
            for (unsigned int m = 1; m < nzeros+1; ++m) {
                Tdata * __restrict outZ = outMat + (i*(nzeros+1)-m)*mCols;
                for (unsigned int j = 0; j < mCols; ++j)
                    outZ[j] =0;
            }
            const Tdata * __restrict in = inMat + i*mCols;
            Tdata * __restrict out = outMat + i*(nzeros+1)*mCols;
            for (unsigned int j = 0; j < mCols; ++j)
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
            out[0] = in[0];
            for (unsigned int j = 1, k=nzeros+1; j < mCols; ++j, k+=(nzeros+1)) {
                for (unsigned int m = 1; m < nzeros+1; ++m)
                    out[k-m] = 0;
                out[k] = in[j];
            }
        }
        return;
    }
}

template <typename Tdata>
void cpu_impl<Tdata>::transform::pad(Tdata * __restrict__ in   ,
                                     Tdata * __restrict__ out  ,
                                     unsigned int         nRows,
                                     unsigned int         nCols,
                                     unsigned int         mRows,
                                     unsigned int         mCols) {

    assert(nRows >= mRows);
    assert(nCols >= mCols);

    // set everything to zero
    for (unsigned int i = 0; i < nRows; ++i) {
        Tdata * __restrict outRow  = out  + i * nCols ;
        for (unsigned int j = 0; j < nCols; ++j) {
            outRow[j] = static_cast<Tdata>(0);
        }
    }

    // copy input to output (everywhere else zeros)
    unsigned int offsetRows = ( nRows - mRows ) / 2 + ( nRows - mRows ) % 2;
    unsigned int offsetCols = ( nCols - mCols ) / 2 + ( nCols - mCols ) % 2;
    for (unsigned int i = offsetRows; i < offsetRows + mRows; ++i) {

        const Tdata * __restrict inRow  = in  + (i - offsetRows) * mCols ;
        Tdata * __restrict outRow = out + i*nCols;
        std::memcpy(outRow + offsetCols, inRow, mCols * sizeof(Tdata));
    }
}

template <typename Tdata>
void cpu_impl<Tdata>::transform::dshear(Tdata * __restrict__ inData ,
                                        Tdata * __restrict__ outData,
                                        long int             k      ,
                                        unsigned int         dim    ,
                                        unsigned int         mRows  ,
                                        unsigned int         mCols  ) {

    assert(dim == 0 || dim == 1);

    if ( dim == 0 ) {

         for (unsigned int j = 0; j < mCols; ++j) {

            long int shift = -k*(mCols / 2 - j);
            if (shift < 0) {

                for (unsigned int i = 0; i < mRows+shift; ++i )
                    outData[i * mCols + j] = * (inData + (i-shift) * mCols + j);
                for (unsigned int i = mRows+shift; i < mRows; ++i)
                    outData[i * mCols + j] = * (inData + (i - (mRows+shift)) * mCols + j);
            } else {

                for (unsigned int i = 0; i < shift; ++i)
                    outData[i * mCols + j] = * (inData + (mRows-shift+i) * mCols + j);
                for (unsigned int i = shift; i < mRows; ++i)
                    outData[i * mCols + j] = * (inData + (i-shift) * mCols + j);
            }

         }

    } else if ( dim == 1 ) {

        for (unsigned int  i = 0; i < mRows; ++i) {

            long int shift = -k*(mRows / 2 - i);
            const Tdata * __restrict in  = inData  + i * mCols ;
            Tdata * __restrict out = outData + i * mCols ;
            if (shift < 0) {
                for (unsigned int j = 0; j < mCols+shift; ++j )
                    out[j] = in[j-shift];
                for (unsigned int j = mCols+shift; j < mCols; ++j)
                    out[j] = in[j - (mCols + shift)];
            } else {
                for (unsigned int j = 0; j < shift; ++j)
                    out[j] = in[mCols-shift+j];
                for (unsigned int j = shift; j < mCols; ++j)
                    out[j] = in[j-shift];
            }
        }
    }
}

// TODO: use template argument for stride
template <typename Tdata>
void cpu_impl<Tdata>::transform::transpose(Tdata * __restrict__ inData ,
                                           Tdata * __restrict__ outData,
                                           unsigned int         mRows  ,
                                           unsigned int         mCols  ) {

    unsigned int stride = std::min(mRows, mCols);
    stride = std::min(stride, 32U);
    for (unsigned int i = 0; i < mCols; i+=stride) {
        for (unsigned int j = 0; j < mRows; j+=stride ) {
            for (unsigned int ii = i; ii < std::min(mCols, i+stride) ; ++ii ) {
                const Tdata * __restrict in  = inData  + ii ;
                Tdata * __restrict out = outData + ii * mRows ;
                for (unsigned int jj = j; jj < std::min(mRows, j+stride); ++jj) {
                    out[jj] = in[jj*mCols];
                }
            }
        }
    }
}

template <typename Tdata>
void cpu_impl<Tdata>::transform::normL2(Tdata * __restrict__ inData ,
                                        Tdata * __restrict__ outData,
                                        unsigned int         size   ) {

    *outData = 0;
    for (unsigned int i = 0; i < size; ++i)
        *outData += std::abs(inData[i]) * std::abs(inData[i]);
}

template <typename Tdata>
void cpu_impl<Tdata>::transform::matMul(Tdata * __restrict__ inDataL,
                                        Tdata * __restrict__ inDataR,
                                        Tdata * __restrict__ outData,
                                        unsigned int inRowsL,
                                        unsigned int inColsL,
                                        unsigned int inRowsR,
                                        unsigned int inColsR) {

    for (unsigned int i = 0; i < inRowsL; ++i) {
        for (unsigned int j = 0; j < inColsR; ++j) {
            for (unsigned int k = 0; k < inColsL; ++k ) {
                outData[i * inColsR + j] += inDataL[i * inColsL + k] *
                                            inDataR[k * inColsR + j];
            }
        }
    }
}
