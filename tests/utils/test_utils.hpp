/*
 * @file test_utils.hpp
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

#ifndef TEST_UTILS_HPP_
#define TEST_UTILS_HPP_

void set_seed() {

    srand((unsigned) time(NULL));
}

template<typename T>
void generate_random_values(T *data, unsigned int size, T min, T max) {

    for (unsigned int i = 0; i < size; ++i) {
        data[i] = min + static_cast<T>(rand()) /( static_cast<T>(RAND_MAX/(max - min)));
    }
}

template<typename T>
void generate_random_values(std::complex<T> *data, unsigned int size, T min, T max) {

    for (unsigned int i = 0; i < size; ++i) {
        data[i] = std::complex<T>(min + static_cast<T>(rand()) /( static_cast<T>(RAND_MAX/(max - min))),
        min + static_cast<T>(rand()) /( static_cast<T>(RAND_MAX/(max - min))));
    }
}


template<typename T>
void test_equality(T *result, T *reference, unsigned int size) {

    for (unsigned int i = 0; i < size; ++i) {
        ASSERT_EQ(result[i], reference[i]);
    }
}

#ifdef CUDA
template<typename Tdata>
void test_check_device_results(Tdata * device_data, unsigned int size, Tdata host_ref, Tdata tolerance) {

    Tdata * host_data = (Tdata *)malloc(size * sizeof(Tdata));
    cudaMemcpy ( host_data, device_data, size *sizeof(Tdata), cudaMemcpyDeviceToHost );
    for (unsigned int i = 0; i < size; ++i)
        ASSERT_TRUE(std::abs(host_data[i] - host_ref) / host_ref < tolerance);
}

template<typename Tdata>
void test_copy_h2d(Tdata * device_data, Tdata *host_data, unsigned int size) {

    cudaMemcpy ( device_data, host_data, size *sizeof(Tdata), cudaMemcpyHostToDevice );
}

template<typename Tdata>
void test_copy_d2h(Tdata * host_data, Tdata *device_data, unsigned int size) {

    cudaMemcpy ( host_data, device_data, size *sizeof(Tdata), cudaMemcpyDeviceToHost );
}
#endif

#endif