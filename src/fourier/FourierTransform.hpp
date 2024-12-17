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

template <typename Tdata,
          template <class> class  backend,
          template <class> class  backendM,
          template <class> class  backendC>
class FourierTransformImpl
{
public:

    using complex_type = typename backendC<Tdata>::complex;

    FourierTransformImpl(unsigned int rows, unsigned int cols) {
        m_impl = std::shared_ptr<fft_type>(new fft_type(rows, cols));
    }
    ~FourierTransformImpl() {
        m_impl.reset();
    }

    void fft(DSmatrix<complex_type, backendM>& inMat) {

        complex_type * data = inMat.data();
        m_impl->fft(data);
    }

    void fftWithShifts(DSmatrix<complex_type, backendM>& inMat) {

        complex_type * data = inMat.data();
        m_impl->ifftshift(data);
        m_impl->fft(data);
        m_impl->fftshift(data);
    }

    void ifft(DSmatrix<complex_type, backendM>& inMat) {

        complex_type * data = inMat.data();
        m_impl->ifft(data);
    }

    void ifftWithShifts(DSmatrix<complex_type, backendM>& inMat) {

        complex_type * data = inMat.data();
        m_impl->ifftshift(data);
        m_impl->ifft(data);
        m_impl->fftshift(data);
    }

    void fftshift(DSmatrix<complex_type, backendM>& inMat) {

        complex_type * data = inMat.data();
        m_impl->fftshift(data);
    }

    void ifftshift(DSmatrix<complex_type, backendM>& inMat) {

        complex_type * data = inMat.data();
        m_impl->ifftshift(data);
    }

    void corrFF2F( const DSmatrix<complex_type, backendM>& A ,
                   const DSmatrix<complex_type, backendM>& B ,
                         DSmatrix<complex_type, backendM>& result) {

        assert(A.size() == B.size());
        assert(A.size() == result.size());
        backendC<Tdata>::op::corrComplex(A.data(), B.data(), result.data(), A.size());
    }

private:
    using fft_type = typename backend<Tdata>::fourier;
    std::shared_ptr<fft_type> m_impl;
};

template<typename T, typename backend> struct FourierTransform_helper;
template<> struct FourierTransform_helper<float, cpu_impl<float>>  { using type = FourierTransformImpl<float, cpu_fft_impl, cpu_impl, cpu_complex_impl>; };
template<> struct FourierTransform_helper<float, cuda_impl<float>>  { using type = FourierTransformImpl<float, cuda_fft_impl, cuda_impl, cuda_complex_impl>; };
template<typename Tdata, typename backend>
using FourierTransform = typename FourierTransform_helper<Tdata, backend>::type;

#endif
