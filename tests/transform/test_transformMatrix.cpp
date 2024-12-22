/*
 * @file test_transformMatrix.cpp
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

#include <algorithm>
#include <iostream>

#include "src/dataStructure/dataStruct.hpp"
#include "src/transform/transformMatrix.hpp"
#include "src/backend/cpu/backendCPU.hpp"
#include "src/backend/cuda/backendCUDA.hpp"

#include <gtest/gtest.h>
#include "tests/utils/test_utils.hpp"

template<typename T>
void transposeCPU(T *inData, T *outData,
                  unsigned int mRows, unsigned int mCols) {

    for (unsigned int i = 0; i < mRows; ++i)
        for (unsigned int j = 0; j < mCols; ++j)
            outData[j*mRows+i] = inData[i*mCols+j];
}

TEST(transform, downsample_dim0_CPU) {

    unsigned int rows = 32;
    unsigned int cols = 64;
    unsigned int stride = 2;
    DSmatrix<float, cpu_impl> myMatrix(rows, cols);
    generate_random_values(myMatrix.data(), rows*cols, -10.0f, 10.0f);
    DSmatrix<float, cpu_impl> myMatrixDown(rows/2, cols);
    downsample(myMatrix, 0, stride, myMatrixDown);
    for (unsigned int i = 0; i < rows/2; ++i)
        for (unsigned int j = 0; j < cols; ++j)
            ASSERT_EQ(myMatrixDown(i,j), myMatrix(stride*i,j));
}

TEST(transform, upsample_dim0_CPU) {

    unsigned int rows = 32;
    unsigned int cols = 64;
    unsigned int nzeros = 2;
    unsigned int rowsUp = (rows-1)*(nzeros)+rows;
    DSmatrix<float, cpu_impl> myMatrix(rows, cols);
    generate_random_values(myMatrix.data(), rows*cols, -10.0f, 10.0f);
    DSmatrix<float, cpu_impl> myMatrixUp(rowsUp, cols);
    upsample(myMatrix, 0, nzeros, myMatrixUp);
    for (unsigned int j = 0; j < cols; ++j)
        ASSERT_EQ(myMatrixUp(0,j), myMatrix(0,j));
    for (unsigned int i = 1; i < rows; ++i)
        for (unsigned int j = 0; j < cols; ++j) {
            for (unsigned int k = 1; k < nzeros+1; ++k)
                ASSERT_EQ(myMatrixUp((nzeros+1)*i-k,j), 0);
            ASSERT_EQ(myMatrixUp((nzeros+1)*i,j), myMatrix(i,j));
        }
}

TEST(transform, transpose_CPU) {

    unsigned int rows = 32;
    unsigned int cols = 64;
    DSmatrix<float, cpu_impl> myMatrix(rows, cols);
    generate_random_values(myMatrix.data(), rows*cols, -10.0f, 10.0f);
    DSmatrix<float, cpu_impl> myMatrixTranspose(cols, rows);
    DSmatrix<float, cpu_impl> myMatrixSolution(cols, rows);
    transposeCPU(myMatrix.data(), myMatrixSolution.data(),rows, cols);
    transpose<float, cpu_impl>(myMatrix, myMatrixTranspose);
    test_equality(myMatrixTranspose.data(), myMatrixSolution.data(), rows*cols);
}

TEST(transform, normL2_CPU) {

    unsigned int rows = 32;
    unsigned int cols = 64;
    DSmatrix<float, cpu_impl> myMatrix(rows, cols, 1.0);
    float result;
    normL2<float, cpu_impl>(myMatrix, &result);
    ASSERT_EQ(result, rows*cols);
}

TEST(transform, normL2_complex_CPU) {

    unsigned int rows = 32;
    unsigned int cols = 64;
    DSmatrix<std::complex<float>, cpu_impl> myMatrix(rows, cols);
    for (unsigned int i = 0; i < rows; ++i)
        for (unsigned int j = 0; j < cols; ++j)
            myMatrix(i,j) = std::complex<float>(1.0, 1.0);
    std::complex<float> result;
    normL2<std::complex<float>, cpu_impl>(myMatrix, &result);
    ASSERT_NEAR(result.real(), 2*rows*cols, 1e-3);
    ASSERT_EQ(result.imag(), 0);
}

#ifdef CUDA
TEST(transform, normL2_complex_CUDA) {

    unsigned int rows = 1;
    unsigned int cols = 1024;
    DSmatrix<std::complex<float>, cpu_impl> myMatrixCPU(rows, cols);
    for (unsigned int i = 0; i < rows; ++i)
        for (unsigned int j = 0; j < cols; ++j)
            myMatrixCPU(i,j) = std::complex<float>(1.0, 1.0);
    DSmatrix<thrust::complex<float>, cuda_impl> myMatrixCUDA(rows, cols);
    test_copy_h2d(reinterpret_cast<std::complex<float>*>(myMatrixCUDA.data()), myMatrixCPU.data(), myMatrixCPU.size());
    std::complex<float> resultCPU;
    DSmatrix<thrust::complex<float>, cuda_impl> resultCUDA(1, 1);
    normL2<thrust::complex<float>, cuda_impl>(myMatrixCUDA, resultCUDA.data());
    test_copy_d2h(&resultCPU, reinterpret_cast<std::complex<float>*>(resultCUDA.data()), 1);
    ASSERT_NEAR(resultCPU.real(), 2*rows*cols, 1e-3);
    ASSERT_EQ(resultCPU.imag(), 0);
}
#endif