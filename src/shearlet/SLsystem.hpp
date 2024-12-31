/*
 * @file SLsystem.hpp
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

#ifndef SLSYSTEM_HPP_
#define SLSYSTEM_HPP_

#include <vector>
#include <deque>
#include <map>
#include <cassert>

#include "src/dataStructure/dataStruct.hpp"

#include "src/backend/cpu/backendCPU.hpp"
#ifdef CUDA
#include "src/backend/cuda/backendCUDA.hpp"
#endif

#include "src/fourier/FourierTransform.hpp"

template<typename Tdata, template <class> class  backend>
class SLcoeffs {

public:

    SLcoeffs() {};

    ~SLcoeffs() {
        for (size_t i = 0; i < m_coeffs.size(); ++i)
            delete(m_coeffs[i]);
    };

    void addElement(const DSmatrix<Tdata, backend>& matIn) {
        m_coeffs.push_back( new DSmatrix<Tdata, backend>( matIn ) );
    }

    DSmatrix<Tdata, backend> * getElement(size_t i) {
        return m_coeffs[i];
    }

    void applyThreshold(std::vector<Tdata>& threshold) {

        assert(threshold.size() == m_coeffs.size());

        for (size_t i = 0; i < m_coeffs.size(); ++i) {

            DSmatrix<Tdata, backend>* mat = m_coeffs[i];
            mat->applyThreshold(threshold[i]);
        }
    }

    void muteShearlet(size_t i) {

        assert(i < m_coeffs.size());

        DSmatrix<Tdata, backend>* mat = m_coeffs[i];
        backend<Tdata>::memory::fill(mat->data, mat->size(), Tdata(0));
    }

private:
    std::vector<DSmatrix<Tdata, backend>*> m_coeffs;
};

#endif
