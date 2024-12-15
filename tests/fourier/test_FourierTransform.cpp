/*
 * @file test_FourierTransform.cpp
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

#include <iostream>
#include <cstring>

#include "src/backend/cpu/backendCPU.hpp"

#include <gtest/gtest.h>
#ifdef CUDA
#include "src/backend/cuda/backendCUDA.hpp"
#include "tests/utils/test_utils.hpp"
#endif

#include "src/fourier/FourierTransform.hpp"

template<typename T>
void fftshiftMatrixCPU(T * idata, T * odata,
                       unsigned int mRows, unsigned int mCols) {

    if (mRows % 2 == 0 && mCols % 2 == 0) {
        for (unsigned int i = 0; i < mRows / 2; ++i) {

            T * __restrict in = idata +  (mRows / 2 + i) * mCols ;
            T * __restrict out = odata + i*mCols;
            for (unsigned int j = 0; j < mCols  / 2; ++j) {
                out[j] = in[mCols / 2 + j];
            }
            for (unsigned int j = mCols / 2; j < mCols; ++j) {
                out[j] = in[j - mCols / 2];
            }
        }

        for (unsigned int i = mRows / 2; i < mRows; ++i) {

            T * __restrict in = idata +  (i - mRows / 2) * mCols ;
            T * __restrict out = odata + i*mCols;
            for (unsigned int j = 0; j < mCols  / 2; ++j) {
                out[j] = in[mCols / 2 + j];
            }
            for (unsigned int j = mCols / 2; j < mCols; ++j) {
                out[j] = in[j - mCols / 2];
            }
        }
    } else if (mRows % 2 == 0 && mCols % 2 == 1) {

        for (unsigned int i = 0; i < mRows / 2; ++i) {

            T * __restrict in = idata +  (mRows / 2 + i) * mCols ;
            T * __restrict out = odata + i*mCols;
            for (unsigned int j = 0; j < (mCols - 1) / 2; ++j) {
                out[j] = in[(mCols + 1) / 2 + j];
            }
            for (unsigned int j = (mCols - 1) / 2; j < mCols; ++j) {
                out[j] = in[j - (mCols - 1) / 2];
            }
        }

        for (unsigned int i = mRows / 2; i < mRows; ++i) {

            T * __restrict in = idata +  (i - mRows / 2) * mCols ;
            T * __restrict out = odata + i*mCols;
            for (unsigned int j = 0; j < (mCols - 1) / 2; ++j) {
                out[j] = in[(mCols + 1) / 2 + j];
            }
            for (unsigned int j = (mCols - 1) / 2; j < mCols; ++j) {
                out[j] = in[j - (mCols - 1) / 2];
            }
        }
    }
}

template<typename Tdata>
void ifftshiftMatrixCPU(Tdata * idata, Tdata * odata,
                       unsigned int mRows, unsigned int mCols) {

    if (mRows % 2 == 0 && mCols % 2 == 0) {

        size_t hRows = mRows / 2;
        size_t hCols = mCols / 2;
        for (size_t i = 0; i < hRows; ++i) {

            Tdata * __restrict in = idata +  (hRows + i) * mCols + hCols;
            Tdata * __restrict out = odata + i*mCols;
            std::memcpy(out, in, hCols * sizeof(Tdata));
            in -= mCols;
            std::memcpy(out + hCols, in + hCols, hCols * sizeof(Tdata));
        }

        for (size_t i = hRows; i < mRows; ++i) {

            Tdata * __restrict in = idata +  (i - hRows) * mCols + hCols ;
            Tdata * __restrict out = odata + i*mCols;
            std::memcpy(out, in, hCols * sizeof(Tdata));
            in -= mCols;
            std::memcpy(out + hCols, in + hCols, hCols * sizeof(Tdata));
        }
    } else if (mRows % 2 == 0 && mCols % 2 == 1) {

        for (unsigned int i = 0; i < mRows / 2; ++i) {

            Tdata * __restrict in = idata +  (mRows / 2 + i) * mCols ;
            Tdata * __restrict out = odata + i*mCols;
            for (unsigned int j = 0; j < (mCols + 1) / 2; ++j) {
                out[j] = in[(mCols - 1) / 2 + j];
            }
            for (unsigned int j = (mCols + 1) / 2; j < mCols; ++j) {
                out[j] = in[j - (mCols + 1) / 2];
            }
        }

        for (unsigned int i = mRows / 2; i < mRows; ++i) {

            Tdata * __restrict in = idata +  (i - mRows / 2) * mCols ;
            Tdata * __restrict out = odata + i*mCols;
            for (unsigned int j = 0; j < (mCols + 1) / 2; ++j) {
                out[j] = in[(mCols - 1) / 2 + j];
            }
            for (unsigned int j = (mCols + 1) / 2; j < mCols; ++j) {
                out[j] = in[j - (mCols + 1) / 2];
            }
        }
    } else if (mRows % 2 == 1 && mCols % 2 == 0) {

        for (unsigned int i = 0; i < (mRows + 1) / 2; ++i) {

            Tdata * __restrict in = idata +  ((mRows - 1) / 2 + i) * mCols ;
            Tdata * __restrict out = odata + i*mCols;
            for (unsigned int j = 0; j < mCols  / 2; ++j) {
                out[j] = in[mCols / 2 + j];
            }
            for (unsigned int j = mCols / 2; j < mCols; ++j) {
                out[j] = in[j - mCols / 2];
            }
        }

        for (unsigned int i = (mRows + 1) / 2; i < mRows; ++i) {

            Tdata * __restrict in = idata +  (i - (mRows + 1) / 2) * mCols ;
            Tdata * __restrict out = odata + i*mCols;
            for (unsigned int j = 0; j < mCols  / 2; ++j) {
                out[j] = in[mCols / 2 + j];
            }
            for (unsigned int j = mCols / 2; j < mCols; ++j) {
                out[j] = in[j - mCols / 2];
            }
        }
    } else {

        for (unsigned int i = 0; i < (mRows + 1) / 2; ++i) {

            Tdata * __restrict in = idata +  ((mRows - 1) / 2 + i) * mCols ;
            Tdata * __restrict out = odata + i*mCols;
            for (unsigned int j = 0; j < (mCols + 1)  / 2; ++j) {
                out[j] = in[(mCols - 1) / 2 + j];
            }
            for (unsigned int j = (mCols + 1) / 2; j < mCols; ++j) {
                out[j] = in[j - (mCols + 1) / 2];
            }
        }

        for (unsigned int i = (mRows + 1) / 2; i < mRows; ++i) {

            Tdata * __restrict in = idata +  (i - (mRows + 1) / 2) * mCols ;
            Tdata * __restrict out = odata + i*mCols;
            for (unsigned int j = 0; j < (mCols + 1) / 2; ++j) {
                out[j] = in[(mCols - 1) / 2 + j];
            }
            for (unsigned int j = (mCols + 1) / 2; j < mCols; ++j) {
                out[j] = in[j - (mCols + 1) / 2];
            }
        }
    }
}

TEST(fourier, fftshift_CPU) {

    unsigned int rows = 1024;
    unsigned int cols =  512;
    FourierTransform<float, cpu_impl<float>> fftOp(rows, cols);
    DSmatrix<std::complex<float>, cpu_impl> cMatrix(rows, cols);
    generate_random_values(cMatrix.data(), rows*cols, -10.0f, 10.0f);
    DSmatrix<std::complex<float>, cpu_impl> rMatrix(rows, cols);
    fftshiftMatrixCPU(cMatrix.data(), rMatrix.data(), rows, cols);
    fftOp.fftshift(cMatrix);
    test_equality(cMatrix.data(), rMatrix.data(), rows*cols);
}

TEST(fourier, ifftshift_CPU) {

    unsigned int rows = 1024;
    unsigned int cols =  512;
    FourierTransform<float, cpu_impl<float>> fftOp(rows, cols);
    DSmatrix<std::complex<float>, cpu_impl> cMatrix(rows, cols);
    generate_random_values(cMatrix.data(), rows*cols, -10.0f, 10.0f);
    DSmatrix<std::complex<float>, cpu_impl> rMatrix(rows, cols);
    ifftshiftMatrixCPU(cMatrix.data(), rMatrix.data(), rows, cols);
    fftOp.ifftshift(cMatrix);
    test_equality(cMatrix.data(), rMatrix.data(), rows*cols);
}

#ifdef CUDA
TEST(fourier, constructor_destructor_CUDA) {

    unsigned int rows = 1024;
    unsigned int cols =  512;
    FourierTransform<float, cuda_impl<float>> fftOp(rows, cols);
    DSmatrix<thrust::complex<float>, cuda_impl> myMatrix(rows, cols);
    fftOp.fft(myMatrix);
}
#endif
