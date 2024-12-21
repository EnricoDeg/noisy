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

#include <algorithm>
#include <iostream>

#include "src/dataStructure/dataStruct.hpp"
#include "src/backend/cpu/backendCPU.hpp"
#include "src/backend/cuda/backendCUDA.hpp"

#include <gtest/gtest.h>
#ifdef CUDA
#include "tests/utils/test_utils.hpp"
#endif

template <class T>
class DSmatrixTemplate : public testing::Test {};
typedef ::testing::Types<float, double> MyTypesCPU ;
TYPED_TEST_CASE(DSmatrixTemplate, MyTypesCPU);

TYPED_TEST(DSmatrixTemplate, constructor_default_CPU) {

    unsigned int rows = 1024;
    unsigned int cols =  512;
    DSmatrix<TypeParam, cpu_impl> myMatrix(rows, cols);
    TypeParam * data = myMatrix.data();
    ASSERT_TRUE(data != nullptr);
}

TYPED_TEST(DSmatrixTemplate, constructor_from_matrix_CPU) {

    set_seed();
    unsigned int rows = 1024;
    unsigned int cols =  512;
    DSmatrix<TypeParam, cpu_impl> myMatrix(rows, cols);
    generate_random_values(myMatrix.data(), rows*cols, TypeParam(-10.0), TypeParam(10.0));
    DSmatrix<TypeParam, cpu_impl> copyMatrix(myMatrix);
    test_equality(myMatrix.data(), copyMatrix.data(), rows*cols);
}

TYPED_TEST(DSmatrixTemplate, assign_CPU) {

    set_seed();
    unsigned int rows = 1024;
    unsigned int cols =  512;
    DSmatrix<TypeParam, cpu_impl> myMatrix(rows, cols);
    generate_random_values(myMatrix.data(), rows*cols, TypeParam(-10.0), TypeParam(10.0));
    DSmatrix<TypeParam, cpu_impl> copyMatrix = myMatrix;
    test_equality(myMatrix.data(), copyMatrix.data(), rows*cols);
}

TYPED_TEST(DSmatrixTemplate, plus_equal_CPU) {

    set_seed();
    unsigned int rows = 1024;
    unsigned int cols =  512;
    DSmatrix<TypeParam, cpu_impl> Matrix1(rows, cols);
    generate_random_values(Matrix1.data(), rows*cols, TypeParam(-10.0), TypeParam(10.0));
    DSmatrix<TypeParam, cpu_impl> Matrix2(rows, cols);
    generate_random_values(Matrix2.data(), rows*cols, TypeParam(-10.0), TypeParam(10.0));
    DSmatrix<TypeParam, cpu_impl> Matrix3 = Matrix1;
    Matrix1 += Matrix2;
    for (unsigned int i = 0; i < rows; ++i)
        for (unsigned int j = 0; j < cols; ++j)
            ASSERT_EQ(Matrix1(i,j), Matrix2(i,j)+Matrix3(i,j));
}

TYPED_TEST(DSmatrixTemplate, prod_equal_CPU) {

    set_seed();
    unsigned int rows = 1024;
    unsigned int cols =  512;
    DSmatrix<TypeParam, cpu_impl> Matrix1(rows, cols);
    generate_random_values(Matrix1.data(), rows*cols, TypeParam(-10.0), TypeParam(10.0));
    DSmatrix<TypeParam, cpu_impl> Matrix2(rows, cols);
    generate_random_values(Matrix2.data(), rows*cols, TypeParam(-10.0), TypeParam(10.0));
    DSmatrix<TypeParam, cpu_impl> Matrix3 = Matrix1;
    Matrix1 *= Matrix2;
    for (unsigned int i = 0; i < rows; ++i)
        for (unsigned int j = 0; j < cols; ++j)
            ASSERT_EQ(Matrix1(i,j), Matrix2(i,j)*Matrix3(i,j));
}

TYPED_TEST(DSmatrixTemplate, fliplr_dim0_CPU) {

    set_seed();
    unsigned int rows = 1024;
    unsigned int cols =  512;
    DSmatrix<TypeParam, cpu_impl> Matrix1(rows, cols);
    generate_random_values(Matrix1.data(), rows*cols, TypeParam(-10.0), TypeParam(10.0));
    DSmatrix<TypeParam, cpu_impl> Matrix2 = Matrix1;
    Matrix2.fliplr(0);
    for (unsigned int i = 0; i < rows; ++i)
        for (unsigned int j = 0; j < cols; ++j)
            ASSERT_EQ(Matrix1(i,j), Matrix2(rows-1-i,j));
}

TYPED_TEST(DSmatrixTemplate, fliplr_complex_dim0_CPU) {

    set_seed();
    unsigned int rows = 1024;
    unsigned int cols =  512;
    DSmatrix<std::complex<TypeParam>, cpu_impl> Matrix1(rows, cols);
    generate_random_values(Matrix1.data(), rows*cols, TypeParam(-10.0f), TypeParam(-10.0f));
    DSmatrix<std::complex<TypeParam>, cpu_impl> Matrix2 = Matrix1;
    Matrix2.fliplr(0);
    for (unsigned int i = 0; i < rows; ++i)
        for (unsigned int j = 0; j < cols; ++j)
            ASSERT_EQ(Matrix1(i,j), Matrix2(rows-1-i,j));
}

TYPED_TEST(DSmatrixTemplate, fliplr_dim1_CPU) {

    set_seed();
    unsigned int rows = 1024;
    unsigned int cols =  512;
    DSmatrix<TypeParam, cpu_impl> Matrix1(rows, cols);
    generate_random_values(Matrix1.data(), rows*cols, TypeParam(-10.0), TypeParam(10.0));
    DSmatrix<TypeParam, cpu_impl> Matrix2 = Matrix1;
    Matrix2.fliplr(1);
    for (unsigned int i = 0; i < rows; ++i)
        for (unsigned int j = 0; j < cols; ++j)
            ASSERT_EQ(Matrix1(i,j), Matrix2(i,cols-1-j));
}

TYPED_TEST(DSmatrixTemplate, normalize_CPU) {

    unsigned int rows = 1024;
    unsigned int cols =  512;
    DSmatrix<TypeParam, cpu_impl> Matrix1(rows, cols, TypeParam(1.0));
    Matrix1.normalize();
    for (unsigned int i = 0; i < rows; ++i)
        for (unsigned int j = 0; j < cols; ++j)
            ASSERT_EQ(Matrix1(i,j), TypeParam(1.0) / (rows * cols));
}

#ifdef CUDA
TYPED_TEST(DSmatrixTemplate, constructor_default_CUDA) {

    unsigned int rows = 1024;
    unsigned int cols =  512;
    DSmatrix<TypeParam, cuda_impl> myMatrix(rows, cols, 1.0);
    TypeParam * data = myMatrix.data();
    ASSERT_TRUE(data != nullptr);
}

TYPED_TEST(DSmatrixTemplate, constructor_from_matrix_CUDA) {

    set_seed();
    unsigned int rows = 1024;
    unsigned int cols =  512;
    DSmatrix<TypeParam, cpu_impl> myMatrixCPU(rows, cols);
    generate_random_values(myMatrixCPU.data(), rows*cols, TypeParam(-10.0), TypeParam(10.0));
    DSmatrix<TypeParam, cuda_impl> myMatrixCUDA(rows, cols);
    test_copy_h2d(myMatrixCUDA.data(), myMatrixCPU.data(), myMatrixCUDA.size());
    DSmatrix<TypeParam, cuda_impl> copyMatrixCUDA(myMatrixCUDA);
    DSmatrix<TypeParam, cpu_impl>  copyMatrixCPU(rows, cols);
    test_copy_d2h(copyMatrixCPU.data(), copyMatrixCUDA.data(), copyMatrixCUDA.size());
    test_equality(myMatrixCPU.data(), copyMatrixCPU.data(), rows*cols);
}

TYPED_TEST(DSmatrixTemplate, normalize_CUDA) {

    unsigned int rows = 1024;
    unsigned int cols =  512;
    DSmatrix<TypeParam, cuda_impl> myMatrix(rows, cols, 1.0);
    myMatrix.normalize();
    // test results
    test_check_device_results(myMatrix.data(), rows * cols, TypeParam(1.0) / (rows * cols), TypeParam(1e-7));
}
#endif
