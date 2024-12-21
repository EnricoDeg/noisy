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
