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

#include "src/shearlet/SLfilter.hpp"

template<typename Tdata, template <class> class  backend>
class SLcoeffs {

public:

    SLcoeffs() {};

    ~SLcoeffs() {
        for (unsigned int i = 0; i < m_coeffs.size(); ++i)
            delete(m_coeffs[i]);
    };

    void addElement(const DSmatrix<Tdata, backend>& matIn) {
        m_coeffs.push_back( new DSmatrix<Tdata, backend>( matIn ) );
    }

    DSmatrix<Tdata, backend> * getElement(unsigned int i) {
        return m_coeffs[i];
    }

    void applyThreshold(std::vector<Tdata>& threshold) {

        assert(threshold.size() == m_coeffs.size());

        for (unsigned int i = 0; i < m_coeffs.size(); ++i) {

            DSmatrix<Tdata, backend>* mat = m_coeffs[i];
            mat->applyThreshold(threshold[i]);
        }
    }

    void muteShearlet(unsigned int i) {

        assert(i < m_coeffs.size());

        DSmatrix<Tdata, backend>* mat = m_coeffs[i];
        backend<Tdata>::memory::fill(mat->data(), mat->size(), Tdata(0));
    }

private:
    std::vector<DSmatrix<Tdata, backend>*> m_coeffs;
};

template<typename T, template <class> class  backend>
class SLsystem
{

private:

    using complex_type = typename backend<T>::complex;
    using DSmatrixReal = DSmatrix<T, backend>;
    using DSmatrixComplex = DSmatrix<complex_type, backend>;
    using _SLfilter = SLfilter<T, backend>;

    struct t_FilterDirections {
        using complex_type = typename backend<T>::complex;
        std::deque<DSmatrix<complex_type, backend>*> dir;
    };

    struct t_FiltersWedgeBandLow {

        using complex_type = typename backend<T>::complex;

        std::vector<DSmatrix<complex_type, backend>*> bandpass;
        std::vector<t_FilterDirections*> wedge;
        DSmatrix<complex_type, backend>* lowpass;
    };

    struct t_Filters {

        t_FiltersWedgeBandLow * cone1;
        t_FiltersWedgeBandLow * cone2;
    };

    t_FiltersWedgeBandLow * computeFilters(unsigned int nrows,
                                           unsigned int ncols,
                                           std::vector<int>& shearLevels);

    t_Filters * prepareFilters(unsigned int nrows,
                               unsigned int ncols,
                               std::vector<int>& shearLevels);

    std::vector<int> computeIdxs(std::vector<int>& shearLevels);

    unsigned int m_rows;
    unsigned int m_cols;
    FourierTransform<T, backend> * m_fftOp;
    std::vector<DSmatrixComplex*> m_shearlets;
    DSmatrixReal * m_weights;
    std::map<int, unsigned int> m_shearlevel2index;

public:

    SLsystem(unsigned int rows,
             unsigned int cols,
             unsigned int Nscales);

    ~SLsystem();

    SLcoeffs<complex_type, backend> decode(DSmatrixReal &image);

    DSmatrixReal recover(SLcoeffs<complex_type, backend> &coeffs);
};

template class SLsystem<float, cpu_impl>;

#endif
