/*
 * @file backendCUDAfourier.hpp
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

#ifndef BACKENDCUDAFOURIER_HPP_
#define BACKENDCUDAFOURIER_HPP_

#include <cufft.h>
#include <thrust/complex.h>

namespace cuda {

    namespace details {

        template<typename T, typename ComplexT, cufftType type>
        class fourier_impl {
        private:
            unsigned int m_rows;
            unsigned int m_cols;
            cufftHandle m_plan ;
        public:
            fourier_impl(unsigned int rows, unsigned int cols);
            ~fourier_impl();
            void fft(thrust::complex<T> * data);
        };

        template<typename T> struct fourier_helper;
        template<> struct fourier_helper<float>  {
            using type = fourier_impl<float, cufftComplex, CUFFT_C2C>;
        };
        template<typename Tdata>
        using fourier = typename cuda::details::fourier_helper<Tdata>::type;

    }

}

#endif
