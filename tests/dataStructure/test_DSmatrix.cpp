/*
 * @file test_DSmatrix.cpp
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

#include "src/dataStructure/dataStruct.hpp"
#include "src/dataStructure/cpu/backendCPU.hpp"
#include "src/dataStructure/cuda/backendCUDA.hpp"

#include <gtest/gtest.h>
#ifdef CUDA
#include "tests/utils/test_utils.hpp"
#endif

TEST(DSmatrix, constructor_destructor_CPU) {

    unsigned int rows = 1024;
    unsigned int cols =  512;
    DSmatrix<float, cpu_impl> myMatrix(rows, cols);
    float * data = myMatrix.data();
    ASSERT_TRUE(data != nullptr);
}

#ifdef CUDA
TEST(DSmatrix, constructor_destructor_CUDA) {

    unsigned int rows = 1024;
    unsigned int cols =  512;
    DSmatrix<float, cuda_impl> myMatrix(rows, cols, 1.0);
    float * data = myMatrix.data();
    ASSERT_TRUE(data != nullptr);
}

TEST(DSmatrix, normalize_CUDA) {

    unsigned int rows = 1024;
    unsigned int cols =  512;
    DSmatrix<float, cuda_impl> myMatrix(rows, cols, 1.0);
    myMatrix.normalize();
    // test results
    test_check_device_results(myMatrix.data(), rows * cols, 1.0f / (rows * cols), 1e-7f);
}
#endif