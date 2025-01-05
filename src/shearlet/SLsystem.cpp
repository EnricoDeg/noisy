/*
 * @file SLsystem.cpp
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

#include <climits>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <complex>
#include <cassert>
#include <algorithm>
#include <deque>

#include "src/shearlet/SLsystem.hpp"
#include "src/shearlet/SLfilter.hpp"

#include "src/dataStructure/dataStruct.hpp"
#include "src/backend/cpu/backendCPU.hpp"
#ifdef CUDA
#include "src/backend/cuda/backendCUDA.hpp"
#endif

#include "src/fourier/FourierTransform.hpp"
#include "src/transform/transformMatrix.hpp"

template<typename T, template <class> class  backend>
SLsystem<T, backend>::SLsystem(unsigned int rows,
                               unsigned int cols,
                               unsigned int Nscales) :
m_rows(rows), m_cols(cols)
{

    // construct fft operator
    m_fftOp = new FourierTransform<T, backend>(rows, cols);

    // compute shear levels
    std::vector<int> shearLevels(Nscales);
    for (unsigned int i = 1; i <= Nscales; ++i)
        shearLevels[i-1] = (int)ceil((float)i * 0.5);

    // compute filters
    t_Filters * filters = prepareFilters(rows, cols, shearLevels);

    // compute indices
    std::vector<int> shearletIdxs = computeIdxs(shearLevels);

    // compute shearlets and weights
    unsigned int nShearlets = shearletIdxs.size() / 3;
    for (unsigned int i = 0; i < nShearlets; ++i) {
        int cone = shearletIdxs[3*i];
        int scale = shearletIdxs[3*i+1];
        int shearing = shearletIdxs[3*i+2];

        if (cone == 0) {
            m_shearlets.push_back( new DSmatrixComplex( *filters->cone1->lowpass ) );
        } else if (cone == 1) {
            unsigned int shearLevel = shearLevels[scale];
            unsigned int indexLevel = m_shearlevel2index[shearLevel];
            unsigned int direction = -shearing + ( 1 << shearLevel );
            DSmatrixComplex tmp( filters->cone1->wedge[indexLevel]->dir[direction]->dims() );
            m_fftOp->corrFF2F( *filters->cone1->wedge[indexLevel]->dir[direction] ,
                               *filters->cone1->bandpass[scale],
                               tmp);
            m_shearlets.push_back( new DSmatrixComplex( tmp ) );
        } else {
            unsigned int shearLevel = shearLevels[scale];
            unsigned int indexLevel = m_shearlevel2index[shearLevel];
            unsigned int direction = shearing + ( 1 << shearLevel );
            DSmatrixComplex tmp(filters->cone2->wedge[indexLevel]->dir[direction]->dims());
            m_fftOp->corrFF2F( *filters->cone2->wedge[indexLevel]->dir[direction] ,
                               *filters->cone2->bandpass[scale],
                                tmp);
            t_dims tmpDims = tmp.dims();
            DSmatrixComplex tmpTranspose(tmpDims.cols, tmpDims.cols);
            transpose(tmp, tmpTranspose);
            m_shearlets.push_back( new DSmatrixComplex( tmpTranspose ) );
        }
    }

    // compute weights
    DSmatrixReal reductionMat(m_shearlets[0]->dims());
    reduceNmat(m_shearlets, reductionMat);
    m_weights = new DSmatrixReal( reductionMat );

    // clean up
    {
        for (unsigned int i = 0; i < filters->cone1->bandpass.size(); ++i)
            delete filters->cone1->bandpass[i];
        delete filters->cone1->lowpass;
        for (unsigned int i = 0; i < filters->cone1->wedge.size(); ++i ) {
            for (unsigned int j = 0; j < filters->cone1->wedge[i]->dir.size(); ++j)
                delete filters->cone1->wedge[i]->dir[j];
            delete filters->cone1->wedge[i];
        }
        delete filters->cone1;
    }

    if (rows != cols) {
        for (unsigned int i = 0; i < filters->cone2->bandpass.size(); ++i)
            delete filters->cone2->bandpass[i];
        delete filters->cone2->lowpass;
        for (unsigned int i = 0; i < filters->cone2->wedge.size(); ++i ) {
            for (unsigned int j = 0; j < filters->cone2->wedge[i]->dir.size(); ++j)
                delete filters->cone2->wedge[i]->dir[j];
            delete filters->cone2->wedge[i];
        }
        delete filters->cone2;
    }
    delete filters;
}

template<typename T, template <class> class  backend>
SLsystem<T, backend>::~SLsystem()
{
    delete m_fftOp;
    for (unsigned int i = 0; i < m_shearlets.size(); ++i)
        delete m_shearlets[i];
    delete m_weights;
}

template<typename T, template <class> class  backend>
SLsystem<T, backend>::t_Filters * SLsystem<T, backend>::prepareFilters(unsigned int rows,
                                     unsigned int cols,
                                     std::vector<int>& shearLevels) {

    t_Filters * filters = new t_Filters;
    filters->cone1 = computeFilters(rows, cols, shearLevels);
    if (rows == cols)
        filters->cone2 = filters->cone1;
    else
        filters->cone2 = computeFilters(cols, rows, shearLevels);

    return filters;
}

template<typename T, template <class> class  backend>
std::vector<int> SLsystem<T, backend>::computeIdxs(std::vector<int>& shearLevels) {

    std::vector<int> idxs;
    for (int cone = 1; cone <= 2; ++cone) {
        for (int scale = 0; scale < shearLevels.size(); ++scale) {
            for (int shearing = -(1 << shearLevels[scale]); shearing <= (1 << shearLevels[scale]); ++shearing) {
                idxs.push_back(cone);
                idxs.push_back(scale);
                idxs.push_back(shearing);
            }
        }
    }
    idxs.push_back(0);
    idxs.push_back(0);
    idxs.push_back(0);

    return idxs;
}

template<typename T, template <class> class  backend>
SLsystem<T, backend>::t_FiltersWedgeBandLow *
SLsystem<T, backend>::computeFilters(unsigned int rows,
                                     unsigned int cols,
                                     std::vector<int>& shearLevels) {

    t_FiltersWedgeBandLow *filters = new t_FiltersWedgeBandLow;

    unsigned int Nscales = shearLevels.size();
    int maxLevel = *std::max_element(shearLevels.begin(), shearLevels.end()) + 1;

    DSmatrixReal directionalFilter = _SLfilter::generator(SL_DIRECTIONAL1);
    directionalFilter.normalize();

    DSmatrixReal scalingFilter  = _SLfilter::generator(SL_SCALING);
    DSmatrixReal scalingFilter2 = _SLfilter::generator(SL_SCALING);
    DSmatrixReal waveletFilter  = _SLfilter::mirror(scalingFilter);

    std::vector<DSmatrixReal*> filterHigh(Nscales);
    std::vector<DSmatrixReal*> filterLow(Nscales);
    std::vector<DSmatrixReal*> filterLow2(maxLevel);

    filterHigh[Nscales-1]  = new DSmatrixReal(waveletFilter);
    filterLow[Nscales-1]   = new DSmatrixReal(scalingFilter);
    filterLow2[maxLevel-1] = new DSmatrixReal(scalingFilter2);

    for (long int i = Nscales-2; i >= 0; --i) {
        unsigned int nzeros = 1;
        DSmatrixReal tmp2( upsample(*filterLow[i+1], 1, nzeros) );
        upsample(*filterLow[i+1], 1, nzeros, &tmp2);

        DSmatrixReal tmpConvolve2( convolve(*filterLow[Nscales-1], tmp2) );
        convolve(*filterLow[Nscales-1], tmp2, &tmpConvolve2);
        filterLow[i] = new DSmatrix( tmpConvolve2 );

        DSmatrixReal tmp( upsample(*filterHigh[i+1], 1, nzeros) );
        upsample(*filterHigh[i+1], 1, nzeros, &tmp);

        DSmatrixReal tmpConvolve( convolve(*filterLow[Nscales-1], tmp) );
        convolve(*filterLow[Nscales-1], tmp, &tmpConvolve);
        filterHigh[i] = new DSmatrixReal( tmpConvolve );
    }

    for (long int i = maxLevel-2; i >= 0; --i) {

        DSmatrixReal tmp( upsample(*filterLow2[i+1], 1, 1) );
        upsample(*filterLow2[i+1], 1, 1, &tmp);

        DSmatrixReal tmp2( convolve(*filterLow2[maxLevel-1], tmp) );
        convolve(*filterLow2[maxLevel-1], tmp, &tmp2);
        filterLow2[i] = new DSmatrixReal( tmp2 );
    }

    DSmatrixComplex filterPaddedFFT(rows, cols);
    for (unsigned int i = 0; i < Nscales; ++i) {
        DSmatrixComplex filterHighComplex(filterHigh[i]->dims());
        real2complex(*filterHigh[i], filterHighComplex);
        m_fftOp->fftWithShiftsPadded(filterHighComplex, filterPaddedFFT);
        filters->bandpass.push_back( new DSmatrixComplex( filterPaddedFFT ) );
    }

    {
        // transpose
        t_dims filterLowDims = filterLow[0]->dims();
        DSmatrixReal filterLow0Transpose(filterLowDims.cols, filterLowDims.rows);
        transpose(*filterLow[0], filterLow0Transpose);
        // matrix-matrix mult
        DSmatrixReal filterLowMatMul(filterLowDims.rows, filterLowDims.rows);
        matMul(*filterLow[0], filterLow0Transpose, filterLowMatMul);
        // convert DSmatrixReal to DSmatrixComplex
        DSmatrixComplex filterLowComplex(filterLowMatMul.dims());
        real2complex(filterLowMatMul, filterLowComplex);
        // fourier transform
        DSmatrixComplex lowpass(rows, cols);
        m_fftOp->fftWithShiftsPadded(filterLowComplex, lowpass);
        // add to struct
        filters->lowpass = new DSmatrixComplex(lowpass);
    }

    for (unsigned int i = 0; i < Nscales; ++i) {
        delete filterHigh[i];
        delete filterLow[i];
    }

    std::vector<int> shearLevelsUnique = shearLevels;
    std::vector<int>::iterator ip;
 
    // Using std::unique
    ip = std::unique(shearLevelsUnique.begin(), shearLevelsUnique.end());

    // Resizing the vector so as to remove the undefined terms
    shearLevelsUnique.resize(std::distance(shearLevelsUnique.begin(), ip));

    for (unsigned int level = 0; level < shearLevelsUnique.size(); ++level) {

        t_FilterDirections * directions = new t_FilterDirections;
        int shearLevel = shearLevelsUnique[level];

        // mapping between shearlevel and memory index of the wedge filter
        m_shearlevel2index.insert(std::map<int, unsigned int>::value_type(shearLevel, level));

        // upsample the directional filter
        DSmatrixReal directionalFilterUpsampled( upsample(directionalFilter, 0, (1 << (shearLevel+1)) - 1) );
        upsample(directionalFilter, 0, (1 << (shearLevel+1)) - 1, &directionalFilterUpsampled);
        // transpose low2 filter (needed for convolution function)
        t_dims filterLow2Dims = filterLow2[filterLow2.size()-1-shearLevel]->dims();
        DSmatrixReal filterLow2Transpose(filterLow2Dims.cols, filterLow2Dims.rows);
        transpose(*filterLow2[filterLow2.size()-1-shearLevel], filterLow2Transpose);
        // data domain convolution of directionalFilterUpsampled and filterLow2Transpose
        DSmatrixReal wedgeHelp( convolve(directionalFilterUpsampled, filterLow2Transpose) );
        convolve(directionalFilterUpsampled, filterLow2Transpose, &wedgeHelp);
        // pad
        DSmatrixReal wedgeHelpPad(rows, cols);
        pad(wedgeHelp, wedgeHelpPad);
        // upsampled
        DSmatrixReal wedgeHelpUpsampled( upsample(wedgeHelpPad, 1, (1 << shearLevel) - 1) );
        upsample(wedgeHelpPad, 1, (1 << shearLevel) - 1, &wedgeHelpUpsampled);

        // pad low2 filter
        t_dims dimsUpsampled = wedgeHelpUpsampled.dims();
        DSmatrixReal lowpassHelp(dimsUpsampled.rows, dimsUpsampled.cols);
        pad(*filterLow2[maxLevel-1-std::max(shearLevel-1, 0)], lowpassHelp);

        // temporary FFT operator
        FourierTransform<T, backend> FFTOp(dimsUpsampled.rows, dimsUpsampled.cols);

        // convolve lowpassHelp and wedgeHelpUpsampled (from data to data domain)
        DSmatrixComplex lowpassHelpComplex( lowpassHelp.dims() );
        real2complex(lowpassHelp, lowpassHelpComplex);
        DSmatrixComplex wedgeHelpUpsampledComplex( wedgeHelpUpsampled.dims() );
        real2complex(wedgeHelpUpsampled, wedgeHelpUpsampledComplex);
        DSmatrixComplex wedgeConv(dimsUpsampled);
        FFTOp.convDD2D(lowpassHelpComplex, wedgeHelpUpsampledComplex, wedgeConv);

        // flip columns of lowPassHelpComplex
        lowpassHelpComplex.fliplr(1);

        // temporary matrices
        t_dims dimsWedgeConv = wedgeConv.dims();
        DSmatrixComplex wedgeUpsampledSheared(dimsWedgeConv);
        DSmatrixComplex wedgeUpsampledConv(dimsWedgeConv);
        DSmatrixComplex wedgeDownsampledConv(rows, cols);

        // directions
        for (long int k = -(1 << shearLevel); k <= (1 << shearLevel); ++k) {

            // apply dshear operator to wedgeConv
            dshear(wedgeConv, wedgeUpsampledSheared, k, 1);
            // convolve lowpassHelpFlip and wedgeUpsampledSheared (from data to data domain)
            FFTOp.convDD2D(lowpassHelpComplex, wedgeUpsampledSheared, wedgeUpsampledConv);
            // downsample wedgeUpsampledConv to (rows,cols)
            downsample(wedgeUpsampledConv, 1, 1 << shearLevel, &wedgeDownsampledConv);
            // apply scaling factor
            wedgeDownsampledConv *= (T)(1 << shearLevel);
            // FFT of wedgeDownsampledConv
            m_fftOp->fftWithShifts(wedgeDownsampledConv);
            directions->dir.push_front( new DSmatrixComplex( wedgeDownsampledConv ) );
        }
        filters->wedge.push_back(directions);
    }

    for (unsigned int i = 0; i < maxLevel; ++i)
        delete filterLow2[i];

    return filters;
}

template<typename T, template <class> class  backend>
SLcoeffs<typename backend<T>::complex, backend> SLsystem<T, backend>::decode(DSmatrixReal &image) {

    t_dims dims = image.dims();
    assert(dims.rows == m_rows);
    assert(dims.cols == m_cols);

    SLcoeffs<typename backend<T>::complex, backend> coeffs;

    DSmatrixComplex imageComplex(dims);
    real2complex(image, imageComplex);
    m_fftOp->fftWithShifts(imageComplex);

    for (unsigned int i = 0; i < m_shearlets.size(); ++i) {
        DSmatrixComplex coeffsImage(dims);
        m_fftOp->corrFF2D(imageComplex, *m_shearlets[i], coeffsImage);
        coeffs.addElement( coeffsImage );
    }

    return coeffs;
}

template<typename T, template <class> class  backend>
DSmatrix<T, backend> SLsystem<T, backend>::recover(SLcoeffs<typename backend<T>::complex, backend> &coeffs) {

    DSmatrixComplex imageComplex(m_rows, m_cols, 0);
    DSmatrixComplex matConv(m_rows, m_cols);

    for (unsigned int i = 0; i < m_shearlets.size(); ++i) {
        m_fftOp->convDF2F( *(coeffs.getElement(i)) , *m_shearlets[i] , matConv );
        imageComplex +=  matConv;
    }

    divComplexByReal(imageComplex, *m_weights);

    m_fftOp->ifftWithShifts(imageComplex);
    DSmatrixReal resultReal(m_rows, m_cols);
    complex2real(imageComplex, resultReal);

    return resultReal;
}
