/*
 * @file FourierTransform.hpp
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

#ifndef FOURIERTRANSFORM_HPP_
#define FOURIERTRANSFORM_HPP_

#include "src/backend/cpu/backendCPUfourier.hpp"
#include "src/backend/cuda/backendCUDAfourier.hpp"
#include "src/dataStructure/dataStruct.hpp"

template <typename Tdata, template <class> class  backend,
          template <class> class  backendM>
class FourierTransformImpl
{
public:

    FourierTransformImpl(unsigned int rows, unsigned int cols) {
        m_impl = std::shared_ptr<fft_type>(new fft_type(rows, cols));
    }
    ~FourierTransformImpl() {
        m_impl.reset();
    }

    void fft(DSmatrix<typename backendM<Tdata>::complex, backendM>& inMat) {

        complex_type * data = inMat.data();
        m_impl->fft(data);
    }

    void fftshift(DSmatrix<typename backendM<Tdata>::complex, backendM>& inMat) {

        complex_type * data = inMat.data();
        m_impl->fftshift(data);
    }

private:
    using complex_type = typename backendM<Tdata>::complex;
    using fft_type = typename backend<Tdata>::fourier;
    std::shared_ptr<fft_type> m_impl;
};

template<typename T, typename backend> struct FourierTransform_helper;
template<> struct FourierTransform_helper<float, cpu_impl<float>>  { using type = FourierTransformImpl<float, cpu_fft_impl, cpu_impl>; };
template<> struct FourierTransform_helper<float, cuda_impl<float>>  { using type = FourierTransformImpl<float, cuda_fft_impl, cuda_impl>; };
template<typename Tdata, typename backend>
using FourierTransform = typename FourierTransform_helper<Tdata, backend>::type;

#endif
