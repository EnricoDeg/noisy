/*
 * @file backendCPUfourier.cpp
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

#include "src/backend/cpu/backendCPUfourier.hpp"

#include <iostream>

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
        fourier_impl<T, ComplexT, planT, plan_dft_2d, destroy_plan, execute_dft>::fourier_impl(unsigned int rows, unsigned int cols)
        : m_rows(rows),
         m_cols(cols)
        {

            ComplexT *fmatInOut  = (ComplexT*) fftw_malloc(sizeof(ComplexT) * rows * cols);
            m_plan_inplace_fft  = plan_dft_2d(rows, cols,
                                                    fmatInOut, fmatInOut,
                                                    FFTW_FORWARD,
                                                    FFTW_ESTIMATE) ;
            m_plan_inplace_ifft = plan_dft_2d(rows, cols,
                                                    fmatInOut, fmatInOut,
                                                    FFTW_BACKWARD,
                                                    FFTW_ESTIMATE) ;
            fftw_free(fmatInOut);

            ComplexT *fmatIn  = (ComplexT*) fftw_malloc(sizeof(ComplexT) * rows * cols);
            ComplexT *fmatOut = (ComplexT*) fftw_malloc(sizeof(ComplexT) * rows * cols);
            m_plan_fft  = plan_dft_2d(rows, cols,
                                            fmatIn, fmatOut,
                                            FFTW_FORWARD,
                                            FFTW_ESTIMATE) ;
            m_plan_ifft = plan_dft_2d(rows, cols,
                                            fmatIn, fmatOut,
                                            FFTW_BACKWARD,
                                            FFTW_ESTIMATE) ;
            fftw_free(fmatIn );
            fftw_free(fmatOut);
        }

        template<
        typename T,
        typename ComplexT,
        typename planT,
        planT plan_dft_2d(int, int, ComplexT*, ComplexT*, int, unsigned int),
        void destroy_plan(planT),
        void execute_dft(planT, ComplexT *, ComplexT *)
        >
        fourier_impl<T, ComplexT, planT, plan_dft_2d, destroy_plan, execute_dft>::~fourier_impl()
        {

            std::cout << "Destroying plans: " << m_rows << ", " << m_cols << std::endl; 
            destroy_plan(m_plan_inplace_fft);
            destroy_plan(m_plan_inplace_ifft);
            destroy_plan(m_plan_fft);
            destroy_plan(m_plan_ifft);
        }

        template<
        typename T,
        typename ComplexT,
        typename planT,
        planT plan_dft_2d(int, int, ComplexT*, ComplexT*, int, unsigned int),
        void destroy_plan(planT),
        void execute_dft(planT, ComplexT *, ComplexT *)
        >
        void fourier_impl<T, ComplexT, planT, plan_dft_2d, destroy_plan, execute_dft>::fft(std::complex<T> * data)
        {
            execute_dft( m_plan_inplace_fft,
                         reinterpret_cast<ComplexT *>(data),
                         reinterpret_cast<ComplexT *>(data));
        }

        template class fourier_impl<float, fftwf_complex, fftwf_plan, fftwf_plan_dft_2d, fftwf_destroy_plan, fftwf_execute_dft>;

    }

}
