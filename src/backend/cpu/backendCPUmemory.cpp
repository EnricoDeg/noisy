/*
 * @file backendCPUmemory.cpp
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

#include <cstring>

#include <fftw3.h>

template <typename Tdata>
Tdata * cpu_impl<Tdata>::memory::allocate(unsigned int elements) {

    return (Tdata*) fftw_malloc(sizeof(Tdata) * elements);
}

template <typename Tdata>
void cpu_impl<Tdata>::memory::free(Tdata *data) {

    fftw_free(data);
}

template <typename Tdata>
void cpu_impl<Tdata>::memory::copy(Tdata* dst, Tdata *src, unsigned int size) {

    std::memcpy(dst, src, size * sizeof(Tdata));
}

template <typename Tdata>
void cpu_impl<Tdata>::memory::copy_d2h(Tdata* dst, Tdata *src, unsigned int size) {

    std::memcpy(dst, src, size * sizeof(Tdata));
}

template <typename Tdata>
void cpu_impl<Tdata>::memory::copy_h2d(Tdata* dst, Tdata *src, unsigned int size) {

    std::memcpy(dst, src, size * sizeof(Tdata));
}


template <typename Tdata>
void cpu_impl<Tdata>::memory::fill(Tdata * __restrict__ data, unsigned int size, Tdata value) {

    for (unsigned int i = 0; i < size; ++i)
        data[i] = value;
}