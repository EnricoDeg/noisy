/*
 * @file backendCPUfourier.hpp
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

#ifndef BACKENDCPUFOURIER_HPP_
#define BACKENDCPUFOURIER_HPP_

#include <complex>
#include "fftw3.h"

namespace cpu {

    namespace details {

        template<
        typename T,
        typename ComplexT,
        typename planT,
        planT plan_dft_2d(int, int, ComplexT*, ComplexT*, int, unsigned int),
        void destroy_plan(planT), 
        void execute_dft(planT, ComplexT *, ComplexT *)
        >
        class fourier_impl {
        private:
            unsigned int m_rows;
            unsigned int m_cols;
            planT m_plan_fft          ;
            planT m_plan_ifft         ;
            planT m_plan_inplace_fft  ;
            planT m_plan_inplace_ifft ;
        public:
            fourier_impl(unsigned int rows, unsigned int cols);
            ~fourier_impl();
            void fft(std::complex<T> *data);
            void fftshift(std::complex<T> *data);
        };

        template<typename T> struct fourier_helper;
        template<> struct fourier_helper<float>  {
            using type = fourier_impl<float, fftwf_complex,
                                      fftwf_plan, fftwf_plan_dft_2d,
                                      fftwf_destroy_plan, fftwf_execute_dft>;
        };

        template<> struct fourier_helper<double> {
            using type = fourier_impl<double, fftw_complex,
                                      fftw_plan, fftw_plan_dft_2d,
                                      fftw_destroy_plan, fftw_execute_dft>;
        };
        template<typename Tdata>
        using fourier = typename cpu::details::fourier_helper<Tdata>::type;

    }

}

#endif
