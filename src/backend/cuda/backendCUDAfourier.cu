/*
 * @file backendCUDAfourier.cu
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

#include "src/backend/cuda/backendCUDAfourier.hpp"
#include "cuAlgo.hpp"

namespace cuda {

    namespace details {

        template<class T>
        class fft_execute { };

        template<>
        class fft_execute<float> {
            public:
            static void execute(cufftHandle plan, cufftComplex *idata, cufftComplex *odata, int direction) {
                cufftExecC2C(plan, (cufftComplex *)idata, (cufftComplex *)odata, direction);
            }
        };

        template<>
        class fft_execute<double> {
            public:
            static void execute(cufftHandle plan, cufftDoubleComplex *idata, cufftDoubleComplex *odata, int direction) {
                cufftExecZ2Z(plan, (cufftDoubleComplex *)idata, (cufftDoubleComplex *)odata, direction);
            }
        };

        template<typename T, typename ComplexT, cufftType type>
        fourier_impl<T, ComplexT, type>::fourier_impl(unsigned int rows, unsigned int cols)
        : m_rows(rows),
          m_cols(cols)
        {

            cufftPlan2d(&m_plan, rows, cols, type);
        }

        template<typename T, typename ComplexT, cufftType type>
        fourier_impl<T, ComplexT, type>::~fourier_impl()
        {

            cufftDestroy(m_plan);
        }

        template<typename T, typename ComplexT, cufftType type>
        void fourier_impl<T, ComplexT, type>::fft(thrust::complex<T> * data) {

            fft_execute<T>::execute(m_plan,
                                    reinterpret_cast<ComplexT *>(data),
                                    reinterpret_cast<ComplexT *>(data),
                                    CUFFT_FORWARD);
        }

        template<typename T, typename ComplexT, cufftType type>
        void fourier_impl<T, ComplexT, type>::ifft(thrust::complex<T> * data) {

            fft_execute<T>::execute(m_plan,
                                    reinterpret_cast<ComplexT *>(data),
                                    reinterpret_cast<ComplexT *>(data),
                                    CUFFT_INVERSE);
        }

        template<typename T, typename ComplexT, cufftType type>
        void fourier_impl<T, ComplexT, type>::fftshift(thrust::complex<T> * data) {

            cuAlgo::fftshift2dMatrix(data, m_rows, m_cols);
        }

        template class fourier_impl<float, cufftComplex, CUFFT_C2C>;

    }

}
