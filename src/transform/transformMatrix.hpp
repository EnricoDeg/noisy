/*
 * @file transformMatrix.hpp
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

#ifndef TRANSFORMMATRIX_HPP_
#define TRANSFORMMATRIX_HPP_

#include <iostream>
#include <cassert>
#include <array>
#include <vector>

#include "src/dataStructure/dataStruct.hpp"
#include "src/backend/cpu/backendCPU.hpp"
#ifdef CUDA
#include "src/backend/cuda/backendCUDA.hpp"
#endif

template <typename Tdata, template <class> class  backend>
inline
t_dims downsample(const DSmatrix<Tdata, backend>& inMat ,
                        unsigned int              dim   ,
                        unsigned int              stride,
                        DSmatrix<Tdata, backend>* outMat = nullptr){

    t_dims dims = inMat.dims();

    if (outMat == nullptr) {

        if (dim == 0) {

            unsigned int count = 0;
            for (unsigned int i = 0; i < dims.rows; i+=stride, ++count);
            return t_dims{.rows = count, .cols = dims.cols};
        } else {

            unsigned int count = 0;
            for (unsigned int i = 0; i < dims.cols; i+=stride, ++count);
            return t_dims{.rows = dims.rows, .cols = count};
        }
    }

    if (dim == 0) {
        unsigned int count = 0;
        for (unsigned int i = 0; i < dims.rows; i+=stride, ++count);
        t_dims dimsOut = outMat->dims();
        assert(dimsOut.rows == count);
        assert(dimsOut.cols == dims.cols);
    } else {
        unsigned int count = 0;
        for (unsigned int i = 0; i < dims.cols; i+=stride, ++count);
        t_dims dimsOut = outMat->dims();
        assert(dimsOut.rows == dims.rows);
        assert(dimsOut.cols == count);
    }

    Tdata * __restrict__ inData = inMat.data();
    Tdata * __restrict__ outData = outMat->data();
    backend<Tdata>::transform::downsample(inData,
                                          outData,
                                          dim,
                                          stride,
                                          dims.rows,
                                          dims.cols);
    return outMat->dims();
}

template <typename Tdata, template <class> class  backend>
t_dims upsample(const DSmatrix<Tdata, backend>& inMat  ,
                    unsigned int              dim    ,
                    unsigned int              nzeros ,
                    DSmatrix<Tdata, backend>* outMat = nullptr) {

    t_dims dims = inMat.dims();

    if (outMat == nullptr) {
        t_dims dimsOutCompute;
        if (dim == 0)
            dimsOutCompute = {.rows = (dims.rows-1)*(nzeros)+dims.rows, .cols = dims.cols};
        else
            dimsOutCompute = {.rows = dims.rows, .cols = (dims.cols-1)*(nzeros)+dims.cols};
        return dimsOutCompute;
    }

    t_dims dimsOut = outMat->dims();
    if (dim == 0) {
        assert(dimsOut.rows == (dims.rows-1)*(nzeros)+dims.rows);
    } else {
        assert(dimsOut.cols == (dims.cols-1)*(nzeros)+dims.cols);
    }

    Tdata * __restrict__ inData = inMat.data();
    Tdata * __restrict__ outData = outMat->data();
    backend<Tdata>::transform::upsample(inData, outData, dim, nzeros, dims.rows, dims.cols);

    return dimsOut;
}

template <typename Tdata, template <class> class  backend>
void pad(const DSmatrix<Tdata, backend>& inMat ,
               DSmatrix<Tdata, backend>& outMat) {

    t_dims dims = inMat.dims();
    t_dims dimsOut = outMat.dims();

    Tdata * __restrict__ inData = inMat.data();
    Tdata * __restrict__ outData = outMat.data();
    backend<Tdata>::transform::pad(inData, outData,
                                   dimsOut.rows, dimsOut.cols,
                                   dims.rows, dims.cols);
}

template <typename Tdata, template <class> class  backend>
void dshear(const DSmatrix<Tdata, backend>& inMat ,
                  DSmatrix<Tdata, backend>& outMat,
                  long int                  k     ,
                  unsigned int              dim   );

template <typename Tdata, template <class> class  backend>
void transpose(const DSmatrix<Tdata, backend>& inMat ,
                     DSmatrix<Tdata, backend>& outMat);

template <typename Tdata, template <class> class  backend>
void normL2(const DSmatrix<Tdata, backend>&  inMat,
                  Tdata                     *out  );

template <typename Tdata, template <class> class  backend>
inline
void matMul(const DSmatrix<Tdata, backend>&  inMatL,
            const DSmatrix<Tdata, backend>&  inMatR,
                  DSmatrix<Tdata, backend>&  outMat) {

    t_dims inMatLDims = inMatL.dims();
    t_dims inMatRDims = inMatR.dims();
    t_dims outMatDims = outMat.dims();

    assert(inMatL.cols == inMatR.rows);
    assert(inMatL.rows == outMat.rows);
    assert(inMatR.cols == outMat.cols);

    backend<Tdata>::transform::matMul(inMatL.data(),
                                  inMatR.data(),
                                  outMat.data(),
                                  inMatLDims.rows,
                                  inMatLDims.cols,
                                  inMatRDims.rows,
                                  inMatRDims.cols);
}

// TODO: fix implementation for CUDA backend
template <typename Tdata, template <class> class  backend,
                          template <class> class  backendC>
struct reduceNmat_impl {

    using complex_type = typename backend<Tdata>::complex;

    static void doit(std::vector<DSmatrix<complex_type, backend>*>& matIn,
                     DSmatrix<Tdata, backend>& outMat) {

        t_dims inDims = matIn[0]->dims();
        t_dims outDims = outMat.dims();
        assert(inDims.rows == outDims.rows);
        assert(inDims.cols == outDims.cols);

        unsigned int numberOfMat = matIn.size();
        unsigned int matSize = matIn[0]->size();
        // pack pointers to data
        complex_type * vecPtr[numberOfMat];
        for (unsigned int i = 0; i < numberOfMat; ++i)
            vecPtr[i] = matIn[i]->data();

        backendC<Tdata>::op::reduceNmat(vecPtr, outMat.data(),
                                        outDims.rows, outDims.cols, numberOfMat);
    }
};

template<typename T, template <class> class  backend> struct reduceNmat_helper;

template<>
struct reduceNmat_helper<float, cpu_impl> {
    using type = reduceNmat_impl<float, cpu_impl, cpu_complex_impl>;
};

template<typename Tdata, template <class> class  backend>
using reduceNmatCaller = typename reduceNmat_helper<Tdata, backend>::type;

template<typename Tdata, template <class> class  backend>
inline
void reduceNmat(std::vector<DSmatrix<typename backend<Tdata>::complex, backend>*>& matIn,
                DSmatrix<Tdata, backend>& outMat) {

    return reduceNmatCaller<Tdata, backend>::doit(matIn, outMat);
}


template <typename Tdata, template <class> class  backend,
                          template <class> class  backendC>
struct real2complex_impl{
    static void doit(const DSmatrix<Tdata, backend>& inMat ,
                           DSmatrix<typename backend<Tdata>::complex, backend>& outMat) {

    t_dims  inDims = inMat.dims();
    t_dims outDims = outMat.dims();
    assert(inDims.rows == outDims.rows);
    assert(inDims.cols == outDims.cols);

    backendC<Tdata>::op::real2complex(inMat.data(), outMat.data(), inDims.rows, inDims.cols);
}
};

template<typename T, template <class> class  backend> struct real2complex_helper;

template<>
struct real2complex_helper<float, cpu_impl> {
    using type = real2complex_impl<float, cpu_impl, cpu_complex_impl>;
};

template<typename Tdata, template <class> class  backend>
using real2complexCaller = typename real2complex_helper<Tdata, backend>::type;

template<typename Tdata, template <class> class  backend>
inline
void real2complex(const DSmatrix<Tdata, backend>&  inMat ,
                        DSmatrix<typename backend<Tdata>::complex, backend>&  outMat) {

    return real2complexCaller<Tdata, backend>::doit(inMat, outMat);
}

template <typename Tdata, template <class> class  backend,
                          template <class> class  backendC>
struct complex2real_impl{
    static void doit(const DSmatrix<typename backend<Tdata>::complex, backend>& inMat ,
                           DSmatrix<Tdata, backend>& outMat) {

    t_dims  inDims = inMat.dims();
    t_dims outDims = outMat.dims();
    assert(inDims.rows == outDims.rows);
    assert(inDims.cols == outDims.cols);

    backendC<Tdata>::op::complex2real(inMat.data(), outMat.data(), inDims.rows, inDims.cols);
}
};

template<typename T, template <class> class  backend> struct complex2real_helper;

template<>
struct complex2real_helper<float, cpu_impl> {
    using type = complex2real_impl<float, cpu_impl, cpu_complex_impl>;
};

template<typename Tdata, template <class> class  backend>
using complex2realCaller = typename complex2real_helper<Tdata, backend>::type;

template<typename Tdata, template <class> class  backend>
inline
void complex2real(const DSmatrix<typename backend<Tdata>::complex, backend>&  inMat ,
                        DSmatrix<Tdata, backend>&  outMat) {

    return complex2realCaller<Tdata, backend>::doit(inMat, outMat);
}

template <typename Tdata, template <class> class  backend,
                          template <class> class  backendC>
struct divComplexByReal_impl{

    static void doit( DSmatrix<typename backend<Tdata>::complex, backend>& complexMat ,
                      DSmatrix<Tdata, backend>& realMat) {

        t_dims complexDims = complexMat.dims();
        t_dims realDims = realMat.dims();
        assert(complexDims.rows == realDims.rows);
        assert(complexDims.cols == realDims.cols);

        backendC<Tdata>::op::divComplexByReal(complexMat.data(),
                                              realMat.data(),
                                              complexDims.rows,
                                              complexDims.cols);
    }
};

template<typename T, template <class> class  backend> struct divComplexByReal_helper;

template<>
struct divComplexByReal_helper<float, cpu_impl> {
    using type = divComplexByReal_impl<float, cpu_impl, cpu_complex_impl>;
};

template<typename Tdata, template <class> class  backend>
using divComplexByRealCaller = typename divComplexByReal_helper<Tdata, backend>::type;

template<typename Tdata, template <class> class  backend>
inline
void divComplexByReal( DSmatrix<typename backend<Tdata>::complex, backend>&  complexMat ,
                       DSmatrix<Tdata, backend>&  realMat) {

    return divComplexByRealCaller<Tdata, backend>::doit(complexMat, realMat);
}

template <typename Tdata, template <class> class  backend,
                          template <class> class  backendC>
struct convolve_impl {

    static t_dims doit(const DSmatrix<Tdata, backend>&  inMat ,
                       const DSmatrix<Tdata, backend>&  filter,
                             DSmatrix<Tdata, backend>*  outMat) {

        t_dims inDims  = inMat.dims();
        t_dims fDims   = filter.dims();
        if (outMat == nullptr) {

            return t_dims{.rows = inDims.rows + fDims.rows - 1,
                        .cols = inDims.cols + fDims.cols - 1};
        }

        t_dims outDims = outMat->dims();
        assert(outDims.rows == inDims.rows + fDims.rows - 1);
        assert(outDims.cols == inDims.cols + fDims.cols - 1);

        DSmatrix<Tdata, backend> inMatPadded(outDims.rows, outDims.cols);
        backendC<Tdata>::op::padMatrix(inMat.data(),
                                    inMatPadded.data(),
                                    inDims.rows,
                                    inDims.cols,
                                    outDims.rows,
                                    outDims.cols);

        DSmatrix<Tdata, backend> filterPadded(outDims.rows, outDims.cols);
        backendC<Tdata>::op::padMatrix(filter.data(),
                                    filterPadded.data(),
                                    fDims.rows,
                                    fDims.cols,
                                    outDims.rows,
                                    outDims.cols);

        backendC<Tdata>::op::convData(inMatPadded.data(),
                                    filterPadded.data(),
                                    outMat->data(),
                                    inDims.rows,
                                    inDims.cols,
                                    fDims.rows,
                                    fDims.cols);
        return outDims;
    }
};

template<typename T, template <class> class  backend> struct convolve_helper;

template<>
struct convolve_helper<float, cpu_impl> {
    using type = convolve_impl<float, cpu_impl, cpu_complex_impl>;
};

template<typename Tdata, template <class> class  backend>
using convolveCaller = typename convolve_helper<Tdata, backend>::type;

template<typename Tdata, template <class> class  backend>
inline
t_dims convolve(const DSmatrix<Tdata, backend>&  inMat ,
                const DSmatrix<Tdata, backend>&  filter,
                      DSmatrix<Tdata, backend>*  outMat = nullptr) {

    return convolveCaller<Tdata, backend>::doit(inMat, filter, outMat);
}

#endif
