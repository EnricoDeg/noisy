/*
 * @file SLfilter.hpp
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

#ifndef SLFILTER_HPP_
#define SLFILTER_HPP_

#include "src/dataStructure/dataStruct.hpp"

#include "src/backend/cpu/backendCPU.hpp"
#ifdef CUDA
#include "src/backend/cuda/backendCUDA.hpp"
#endif

enum SLFilterType {
    SL_SCALING,
    SL_DIRECTIONAL1,
    SL_DIRECTIONAL2,
    SL_DIRECTIONAL3,
    SL_COIFLET,
    SL_WAVELET,
    SL_TEST,
    SL_DIRECTIONAL_TEST
};

template <typename Tdata, template <class> class  backend>
DSmatrix<Tdata, backend> SLfilterMirror(DSmatrix<Tdata, backend>& vecIn);

template <typename Tdata, template <class> class  backend>
DSmatrix<Tdata, backend> SLfilterGenerator(SLFilterType type);

#endif