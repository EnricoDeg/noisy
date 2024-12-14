/*
 * @file test_fourier.cpp
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


#include "src/backend/cpu/backendCPU.hpp"

#include <gtest/gtest.h>
#ifdef CUDA
#include "src/backend/cuda/backendCUDA.hpp"
#include "tests/utils/test_utils.hpp"
#endif

#include "src/fourier/FourierTransform.hpp"

TEST(fourier, constructor_destructor_CPU) {

    unsigned int rows = 1024;
    unsigned int cols =  512;
    cpu_fft_impl<float>::fourier fftOp(rows, cols);
}

TEST(fourier, constructor_destructor_CPU_high) {

    unsigned int rows = 1024;
    unsigned int cols =  512;
    FourierTransform<float, cpu_fft_impl> fftOp(rows, cols);
}

#ifdef CUDA
TEST(fourier, constructor_destructor_CUDA) {

    unsigned int rows = 1024;
    unsigned int cols =  512;
    cuda_fft_impl<float>::fourier fftOp(rows, cols);
}
#endif
