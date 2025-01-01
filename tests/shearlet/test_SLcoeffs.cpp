/*
 * @file test_SLcoeffs.cpp
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

#include "src/shearlet/SLsystem.hpp"

#include <gtest/gtest.h>
#include "tests/utils/test_utils.hpp"

template <class T>
class SLcoeffsTemplate : public testing::Test {};
typedef ::testing::Types<float, double> MyTypesCPU ;
TYPED_TEST_CASE(SLcoeffsTemplate, MyTypesCPU);

TYPED_TEST(SLcoeffsTemplate, addElement_CPU) {

    SLcoeffs<TypeParam, cpu_impl> coeffs{};

    unsigned int rows = 1024;
    unsigned int cols =  512;

    // first element
    DSmatrix<TypeParam, cpu_impl> myMatrix1(rows, cols);
    generate_random_values(myMatrix1.data(), rows*cols, TypeParam(-10.0), TypeParam(10.0));
    coeffs.addElement(myMatrix1);

    // second element
    DSmatrix<TypeParam, cpu_impl> myMatrix2(rows, cols);
    generate_random_values(myMatrix2.data(), rows*cols, TypeParam(-10.0), TypeParam(10.0));
    coeffs.addElement(myMatrix2);

    // check first element
    DSmatrix<TypeParam, cpu_impl> *mat = coeffs.getElement(0);
    test_equality(myMatrix1.data(), mat->data(), rows*cols);

    // check second element
    mat =  coeffs.getElement(1);
    test_equality(myMatrix2.data(), mat->data(), rows*cols);
}

TYPED_TEST(SLcoeffsTemplate, muteShearlet_CPU) {

    SLcoeffs<TypeParam, cpu_impl> coeffs{};

    unsigned int rows = 1024;
    unsigned int cols =  512;

    // first element
    DSmatrix<TypeParam, cpu_impl> myMatrix1(rows, cols);
    generate_random_values(myMatrix1.data(), rows*cols, TypeParam(-10.0), TypeParam(10.0));
    coeffs.addElement(myMatrix1);

    // second element
    DSmatrix<TypeParam, cpu_impl> myMatrix2(rows, cols);
    generate_random_values(myMatrix2.data(), rows*cols, TypeParam(-10.0), TypeParam(10.0));
    coeffs.addElement(myMatrix2);

    // mute
    coeffs.muteShearlet(1);

    // check second element (muted)
    DSmatrix<TypeParam, cpu_impl> *mat = coeffs.getElement(1);
    TypeParam * __restrict__ data = mat->data();
    for (unsigned int i = 0; i < rows*cols; ++i)
        ASSERT_EQ(data[i], 0);
}

TYPED_TEST(SLcoeffsTemplate, applyThreshold_CPU) {

    SLcoeffs<TypeParam, cpu_impl> coeffs{};

    unsigned int rows = 1024;
    unsigned int cols =  512;

    // first element
    DSmatrix<TypeParam, cpu_impl> myMatrix1(rows, cols);
    generate_random_values(myMatrix1.data(), rows*cols, TypeParam(-10.0), TypeParam(10.0));
    coeffs.addElement(myMatrix1);

    // second element
    DSmatrix<TypeParam, cpu_impl> myMatrix2(rows, cols);
    generate_random_values(myMatrix2.data(), rows*cols, TypeParam(-10.0), TypeParam(10.0));
    coeffs.addElement(myMatrix2);

    // threshold
    std::vector<TypeParam> threshold{};
    threshold.push_back(1.0);
    threshold.push_back(0.5);

    coeffs.applyThreshold(threshold);

    // check first element
    myMatrix1.applyThreshold(threshold[0]);
    TypeParam * __restrict__ refData = myMatrix1.data();
    DSmatrix<TypeParam, cpu_impl> *mat = coeffs.getElement(0);
    TypeParam * __restrict__ data = mat->data();
    for (unsigned int i = 0; i < rows*cols; ++i)
        if (std::abs(data[i]) < threshold[0])
            ASSERT_EQ(data[i], 0);
    for (unsigned int i = 0; i < rows*cols; ++i)
        ASSERT_EQ(data[i], refData[i]);

    // check second element
    myMatrix2.applyThreshold(threshold[1]);
    refData = myMatrix2.data();
    mat = coeffs.getElement(1);
    data = mat->data();
    for (unsigned int i = 0; i < rows*cols; ++i)
        if (std::abs(data[i]) < threshold[1])
            ASSERT_EQ(data[i], 0);
    for (unsigned int i = 0; i < rows*cols; ++i)
        ASSERT_EQ(data[i], refData[i]);
}

#ifdef CUDA

TYPED_TEST(SLcoeffsTemplate, addElement_CUDA) {

    set_seed();

    SLcoeffs<TypeParam, cuda_impl> coeffs{};

    unsigned int rows = 1024;
    unsigned int cols =  512;

    // first element
    DSmatrix<TypeParam, cpu_impl> myMatrix1CPU(rows, cols);
    generate_random_values(myMatrix1CPU.data(), rows*cols, TypeParam(-10.0), TypeParam(10.0));
    DSmatrix<TypeParam, cuda_impl> myMatrixCUDA(rows, cols);
    test_copy_h2d(myMatrixCUDA.data(), myMatrix1CPU.data(), myMatrixCUDA.size());
    coeffs.addElement(myMatrixCUDA);

    // second element
    DSmatrix<TypeParam, cpu_impl> myMatrix2CPU(rows, cols);
    generate_random_values(myMatrix2CPU.data(), rows*cols, TypeParam(-10.0), TypeParam(10.0));
    test_copy_h2d(myMatrixCUDA.data(), myMatrix2CPU.data(), myMatrixCUDA.size());
    coeffs.addElement(myMatrixCUDA);

    // check first element
    DSmatrix<TypeParam, cuda_impl> *mat = coeffs.getElement(0);
    DSmatrix<TypeParam, cpu_impl> myMatrixCheck(rows, cols);
    test_copy_d2h(myMatrixCheck.data(), mat->data(), mat->size());
    test_equality(myMatrixCheck.data(), myMatrix1CPU.data(), rows*cols);

    // check second element
    mat = coeffs.getElement(1);
    test_copy_d2h(myMatrixCheck.data(), mat->data(), mat->size());
    test_equality(myMatrixCheck.data(), myMatrix2CPU.data(), rows*cols);
}

TYPED_TEST(SLcoeffsTemplate, muteShearlet_CUDA) {

    set_seed();

    SLcoeffs<TypeParam, cuda_impl> coeffs{};

    unsigned int rows = 1024;
    unsigned int cols =  512;

    // first element
    DSmatrix<TypeParam, cpu_impl> myMatrix1CPU(rows, cols);
    generate_random_values(myMatrix1CPU.data(), rows*cols, TypeParam(-10.0), TypeParam(10.0));
    DSmatrix<TypeParam, cuda_impl> myMatrixCUDA(rows, cols);
    test_copy_h2d(myMatrixCUDA.data(), myMatrix1CPU.data(), myMatrixCUDA.size());
    coeffs.addElement(myMatrixCUDA);

    // second element
    DSmatrix<TypeParam, cpu_impl> myMatrix2CPU(rows, cols);
    generate_random_values(myMatrix2CPU.data(), rows*cols, TypeParam(-10.0), TypeParam(10.0));
    test_copy_h2d(myMatrixCUDA.data(), myMatrix2CPU.data(), myMatrixCUDA.size());
    coeffs.addElement(myMatrixCUDA);

    // mute
    coeffs.muteShearlet(1);

    // check second element (muted)
    DSmatrix<TypeParam, cuda_impl> *mat = coeffs.getElement(1);
    DSmatrix<TypeParam, cpu_impl> myMatrixCheck(rows, cols);
    test_copy_d2h(myMatrixCheck.data(), mat->data(), mat->size());

    TypeParam * __restrict__ data = myMatrixCheck.data();
    for (unsigned int i = 0; i < rows*cols; ++i)
        ASSERT_EQ(data[i], 0);
}

TYPED_TEST(SLcoeffsTemplate, applyThreshold_CUDA) {

    set_seed();

    SLcoeffs<TypeParam, cuda_impl> coeffs{};

    unsigned int rows = 1024;
    unsigned int cols =  512;

    // first element
    DSmatrix<TypeParam, cpu_impl> myMatrix1CPU(rows, cols);
    generate_random_values(myMatrix1CPU.data(), rows*cols, TypeParam(-10.0), TypeParam(10.0));
    DSmatrix<TypeParam, cuda_impl> myMatrixCUDA(rows, cols);
    test_copy_h2d(myMatrixCUDA.data(), myMatrix1CPU.data(), myMatrixCUDA.size());
    coeffs.addElement(myMatrixCUDA);

    // second element
    DSmatrix<TypeParam, cpu_impl> myMatrix2CPU(rows, cols);
    generate_random_values(myMatrix2CPU.data(), rows*cols, TypeParam(-10.0), TypeParam(10.0));
    test_copy_h2d(myMatrixCUDA.data(), myMatrix2CPU.data(), myMatrixCUDA.size());
    coeffs.addElement(myMatrixCUDA);

    // threshold
    std::vector<TypeParam> threshold{};
    threshold.push_back(1.0);
    threshold.push_back(0.5);

    coeffs.applyThreshold(threshold);

    // check first element
    myMatrix1CPU.applyThreshold(threshold[0]);
    TypeParam * __restrict__ refData = myMatrix1CPU.data();
    DSmatrix<TypeParam, cuda_impl> *mat = coeffs.getElement(0);
    DSmatrix<TypeParam, cpu_impl> myMatrixCheck(rows, cols);
    test_copy_d2h(myMatrixCheck.data(), mat->data(), mat->size());
    TypeParam * __restrict__ data = myMatrixCheck.data();
    for (unsigned int i = 0; i < rows*cols; ++i)
        if (std::abs(data[i]) < threshold[0])
            ASSERT_EQ(data[i], 0);
    for (unsigned int i = 0; i < rows*cols; ++i)
        ASSERT_EQ(data[i], refData[i]);

    // check second element
    myMatrix2CPU.applyThreshold(threshold[1]);
    refData = myMatrix2CPU.data();
    mat = coeffs.getElement(1);
    test_copy_d2h(myMatrixCheck.data(), mat->data(), mat->size());
    data = myMatrixCheck.data();
    for (unsigned int i = 0; i < rows*cols; ++i)
        if (std::abs(data[i]) < threshold[1])
            ASSERT_EQ(data[i], 0);
    for (unsigned int i = 0; i < rows*cols; ++i)
        ASSERT_EQ(data[i], refData[i]);
}

#endif
