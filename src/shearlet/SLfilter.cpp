/*
 * @file SLfilter.cpp
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

#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>
#include <memory>
#include "SLfilter.hpp"

template <typename Tdata, template <class> class  backend>
DSmatrix<Tdata, backend> SLfilterMirror(DSmatrix<Tdata, backend>& vecIn)
{

    size_t N = vecIn.size();
    DSmatrix<Tdata, backend> vecOut(1,N);
    backend<Tdata>::op::mirror(vecIn.data(), vecOut.data(), vecIn.size());
    return vecOut;
}

template <typename Tdata, template <class> class  backend>
DSmatrix<Tdata, backend> SLfilterGenerator(SLFilterType type)
{

    if (type == SL_SCALING) {

        std::vector<Tdata> v = { 0.0104933261758410,
                                -0.0263483047033631,
                                -0.0517766952966370,
                                 0.276348304703363 ,
                                 0.582566738241592 ,
                                 0.276348304703363 ,
                                -0.0517766952966369,
                                -0.0263483047033631,
                                 0.0104933261758408};
        size_t N = v.size();
        DSmatrix<Tdata, backend> vecOut(1, N);
        backend<Tdata>::memory::copy_h2d(vecOut.data(), v.data(), N);
        return vecOut;
    } else if (type == SL_DIRECTIONAL1) {

        std::vector<Tdata> v = {-0.0000000e+00,
                                 0.0000000e+00,
                                -3.0861315e-07,
                                 0.0000000e+00,
                                -3.7033578e-07,
                                 0.0000000e+00,
                                -4.8143652e-07,
                                 0.0000000e+00,
                                -8.3942778e-07,
                                 0.0000000e+00,
                                -4.8143652e-07,
                                 0.0000000e+00,
                                -3.7033578e-07,
                                 0.0000000e+00,
                                -3.0861315e-07,
                                 0.0000000e+00,
                                -0.0000000e+00,
                                -0.0000000e+00,
                                 6.1722631e-07,
                                -0.0000000e+00,
                                -6.0488178e-06,
                                -0.0000000e+00,
                                -7.6782952e-06,
                                -0.0000000e+00,
                                -1.2171703e-05,
                                -0.0000000e+00,
                                -1.2171703e-05,
                                -0.0000000e+00,
                                -7.6782952e-06,
                                -0.0000000e+00,
                                -6.0488178e-06,
                                -0.0000000e+00,
                                 6.1722631e-07,
                                -0.0000000e+00,
                                -3.0861315e-07,
                                 0.0000000e+00,
                                 1.2838307e-05,
                                 0.0000000e+00,
                                -5.9401860e-05,
                                 0.0000000e+00,
                                -8.7893026e-05,
                                 0.0000000e+00,
                                -1.0969346e-04,
                                 0.0000000e+00,
                                -8.7893026e-05,
                                 0.0000000e+00,
                                -5.9401860e-05,
                                 0.0000000e+00,
                                 1.2838307e-05,
                                 0.0000000e+00,
                                -3.0861315e-07,
                                -0.0000000e+00,
                                -6.0488178e-06,
                                -0.0000000e+00,
                                 1.3512318e-04,
                                -0.0000000e+00,
                                -4.3544081e-04,
                                -0.0000000e+00,
                                -5.7848918e-04,
                                -0.0000000e+00,
                                -5.7848918e-04,
                                -0.0000000e+00,
                                -4.3544081e-04,
                                -0.0000000e+00,
                                 1.3512318e-04,
                                -0.0000000e+00,
                                -6.0488178e-06,
                                -0.0000000e+00,
                                -3.7033578e-07,
                                 0.0000000e+00,
                                -5.9401860e-05,
                                 0.0000000e+00,
                                 1.0726899e-03,
                                 6.1035156e-04,
                                -1.9615452e-03,
                                 3.6621094e-04,
                                -2.2362850e-03,
                                 3.6621094e-04,
                                -1.9615452e-03,
                                 6.1035156e-04,
                                 1.0726899e-03,
                                 0.0000000e+00,
                                -5.9401860e-05,
                                 0.0000000e+00,
                                -3.7033578e-07,
                                -0.0000000e+00,
                                -7.6782952e-06,
                                -0.0000000e+00,
                                -4.3544081e-04,
                                -6.1035156e-04,
                                 5.3247620e-03,
                                 6.3476562e-03,
                                -5.3114299e-03,
                                 4.1503906e-03,
                                -5.3114299e-03,
                                 6.3476562e-03,
                                 5.3247620e-03,
                                -6.1035156e-04,
                                -4.3544081e-04,
                                -0.0000000e+00,
                                -7.6782952e-06,
                                -0.0000000e+00,
                                -4.8143652e-07,
                                 0.0000000e+00,
                                -8.7893026e-05,
                                 0.0000000e+00,
                                -1.9615452e-03,
                                -6.3476562e-03,
                                 1.6444291e-02,
                                 3.3691406e-02,
                                -8.7910061e-03,
                                 3.3691406e-02,
                                 1.6444291e-02,
                                -6.3476562e-03,
                                -1.9615452e-03,
                                 0.0000000e+00,
                                -8.7893026e-05,
                                 0.0000000e+00,
                                -4.8143652e-07,
                                -0.0000000e+00,
                                -1.2171703e-05,
                                -0.0000000e+00,
                                -5.7848918e-04,
                                -3.6621094e-04,
                                -5.3114299e-03,
                                -3.3691406e-02,
                                 3.3130363e-02,
                                 1.7749023e-01,
                                 3.3130363e-02,
                                -3.3691406e-02,
                                -5.3114299e-03,
                                -3.6621094e-04,
                                -5.7848918e-04,
                                -0.0000000e+00,
                                -1.2171703e-05,
                                -0.0000000e+00,
                                -8.3942778e-07,
                                 0.0000000e+00,
                                -1.0969346e-04,
                                 0.0000000e+00,
                                -2.2362850e-03,
                                -4.1503906e-03,
                                -8.7910061e-03,
                                -1.7749023e-01,
                                 5.9486541e-01,
                                -1.7749023e-01,
                                -8.7910061e-03,
                                -4.1503906e-03,
                                -2.2362850e-03,
                                 0.0000000e+00,
                                -1.0969346e-04,
                                 0.0000000e+00,
                                -8.3942778e-07,
                                -0.0000000e+00,
                                -1.2171703e-05,
                                -0.0000000e+00,
                                -5.7848918e-04,
                                -3.6621094e-04,
                                -5.3114299e-03,
                                -3.3691406e-02,
                                 3.3130363e-02,
                                 1.7749023e-01,
                                 3.3130363e-02,
                                -3.3691406e-02,
                                -5.3114299e-03,
                                -3.6621094e-04,
                                -5.7848918e-04,
                                -0.0000000e+00,
                                -1.2171703e-05,
                                -0.0000000e+00,
                                -4.8143652e-07,
                                 0.0000000e+00,
                                -8.7893026e-05,
                                 0.0000000e+00,
                                -1.9615452e-03,
                                -6.3476562e-03,
                                 1.6444291e-02,
                                 3.3691406e-02,
                                -8.7910061e-03,
                                 3.3691406e-02,
                                 1.6444291e-02,
                                -6.3476562e-03,
                                -1.9615452e-03,
                                 0.0000000e+00,
                                -8.7893026e-05,
                                 0.0000000e+00,
                                -4.8143652e-07,
                                -0.0000000e+00,
                                -7.6782952e-06,
                                -0.0000000e+00,
                                -4.3544081e-04,
                                -6.1035156e-04,
                                 5.3247620e-03,
                                 6.3476562e-03,
                                -5.3114299e-03,
                                 4.1503906e-03,
                                -5.3114299e-03,
                                 6.3476562e-03,
                                 5.3247620e-03,
                                -6.1035156e-04,
                                -4.3544081e-04,
                                -0.0000000e+00,
                                -7.6782952e-06,
                                -0.0000000e+00,
                                -3.7033578e-07,
                                 0.0000000e+00,
                                -5.9401860e-05,
                                 0.0000000e+00,
                                 1.0726899e-03,
                                 6.1035156e-04,
                                -1.9615452e-03,
                                 3.6621094e-04,
                                -2.2362850e-03,
                                 3.6621094e-04,
                                -1.9615452e-03,
                                 6.1035156e-04,
                                 1.0726899e-03,
                                 0.0000000e+00,
                                -5.9401860e-05,
                                 0.0000000e+00,
                                -3.7033578e-07,
                                -0.0000000e+00,
                                -6.0488178e-06,
                                -0.0000000e+00,
                                 1.3512318e-04,
                                -0.0000000e+00,
                                -4.3544081e-04,
                                -0.0000000e+00,
                                -5.7848918e-04,
                                -0.0000000e+00,
                                -5.7848918e-04,
                                -0.0000000e+00,
                                -4.3544081e-04,
                                -0.0000000e+00,
                                 1.3512318e-04,
                                -0.0000000e+00,
                                -6.0488178e-06,
                                -0.0000000e+00,
                                -3.0861315e-07,
                                 0.0000000e+00,
                                 1.2838307e-05,
                                 0.0000000e+00,
                                -5.9401860e-05,
                                 0.0000000e+00,
                                -8.7893026e-05,
                                 0.0000000e+00,
                                -1.0969346e-04,
                                 0.0000000e+00,
                                -8.7893026e-05,
                                 0.0000000e+00,
                                -5.9401860e-05,
                                 0.0000000e+00,
                                 1.2838307e-05,
                                 0.0000000e+00,
                                -3.0861315e-07,
                                -0.0000000e+00,
                                 6.1722631e-07,
                                -0.0000000e+00,
                                -6.0488178e-06,
                                -0.0000000e+00,
                                -7.6782952e-06,
                                -0.0000000e+00,
                                -1.2171703e-05,
                                -0.0000000e+00,
                                -1.2171703e-05,
                                -0.0000000e+00,
                                -7.6782952e-06,
                                -0.0000000e+00,
                                -6.0488178e-06,
                                -0.0000000e+00,
                                 6.1722631e-07,
                                -0.0000000e+00,
                                -0.0000000e+00,
                                 0.0000000e+00,
                                -3.0861315e-07,
                                 0.0000000e+00
                                -3.7033578e-07,
                                 0.0000000e+00,
                                -4.8143652e-07,
                                 0.0000000e+00,
                                -8.3942778e-07,
                                 0.0000000e+00,
                                -4.8143652e-07,
                                 0.0000000e+00,
                                -3.7033578e-07,
                                 0.0000000e+00,
                                -3.0861315e-07,
                                 0.0000000e+00,
                                -0.0000000e+00};
        size_t N = v.size();
        DSmatrix<Tdata, backend> vecOut(17, 17);
        backend<Tdata>::memory::copy_h2d(vecOut.data(), v.data(), N);
        return vecOut;

    } else if (type == SL_DIRECTIONAL2) {

        std::vector<Tdata> v = {  0.0000000e+00,
                                 -0.0000000e+00,
                                  0.0000000e+00,
                                 -0.0000000e+00,
                                  1.6717973e-03,
                                 -0.0000000e+00,
                                  0.0000000e+00,
                                 -0.0000000e+00,
                                  0.0000000e+00,
                                  0.0000000e+00,
                                 -0.0000000e+00,
                                  0.0000000e+00,
                                 -6.6871894e-03,
                                 -2.1080148e-03,
                                 -6.6871894e-03,
                                  0.0000000e+00,
                                 -0.0000000e+00,
                                  0.0000000e+00,
                                  0.0000000e+00,
                                 -0.0000000e+00,
                                  1.0030784e-02,
                                  6.3240444e-03,
                                 -1.9555817e-02,
                                  6.3240444e-03,
                                  1.0030784e-02,
                                 -0.0000000e+00,
                                  0.0000000e+00,
                                  0.0000000e+00,
                                 -6.6871894e-03,
                                 -6.3240444e-03,
                                  5.2486012e-02,
                                  1.3975610e-01,
                                  5.2486012e-02,
                                 -6.3240444e-03,
                                 -6.6871894e-03,
                                  0.0000000e+00,
                                  1.6717973e-03,
                                  2.1080148e-03,
                                 -1.9555817e-02,
                                 -1.3975610e-01,
                                  6.8785947e-01,
                                 -1.3975610e-01,
                                 -1.9555817e-02,
                                  2.1080148e-03,
                                  1.6717973e-03,
                                  0.0000000e+00,
                                 -6.6871894e-03,
                                 -6.3240444e-03,
                                  5.2486012e-02,
                                  1.3975610e-01,
                                  5.2486012e-02,
                                 -6.3240444e-03,
                                 -6.6871894e-03,
                                  0.0000000e+00,
                                  0.0000000e+00,
                                 -0.0000000e+00,
                                  1.0030784e-02,
                                  6.3240444e-03,
                                 -1.9555817e-02,
                                  6.3240444e-03,
                                  1.0030784e-02,
                                 -0.0000000e+00,
                                  0.0000000e+00,
                                  0.0000000e+00,
                                 -0.0000000e+00,
                                  0.0000000e+00,
                                 -6.6871894e-03,
                                 -2.1080148e-03,
                                 -6.6871894e-03,
                                  0.0000000e+00,
                                 -0.0000000e+00,
                                  0.0000000e+00,
                                  0.0000000e+00,
                                 -0.0000000e+00,
                                  0.0000000e+00,
                                 -0.0000000e+00,
                                  1.6717973e-03,
                                 -0.0000000e+00,
                                  0.0000000e+00,
                                 -0.0000000e+00,
                                  0.0000000e+00};
        size_t N = v.size();
        DSmatrix<Tdata, backend> vecOut(9, 9);
        backend<Tdata>::memory::copy_h2d(vecOut.data(), v.data(), N);
        return vecOut;

    } else if (type == SL_DIRECTIONAL3) {

        std::vector<Tdata> v = { -6.0515365e-02,
                                  0.0000000e+00,
                                  1.2103073e-01,
                                  0.0000000e+00,
                                 -6.0515365e-02,
                                  0.0000000e+00,
                                  4.6875000e-02,
                                  7.8125000e-02,
                                 -4.6875000e-01,
                                  4.6875000e-01,
                                 -7.8125000e-02,
                                 -4.6875000e-02,
                                 -0.0000000e+00,
                                 -6.0515365e-02,
                                 -0.0000000e+00,
                                  1.2103073e-01,
                                 -0.0000000e+00,
                                 -6.0515365e-02 };
        size_t N = v.size();
        DSmatrix<Tdata, backend> vecOut(3, 6);
        backend<Tdata>::memory::copy_h2d(vecOut.data(), v.data(), N);
        return vecOut;

    } else if (type == SL_COIFLET) {

        std::vector<Tdata> v = {  3.8580778e-02,
                                 -1.2696913e-01,
                                 -7.7161555e-02,
                                  6.0749164e-01,
                                  7.4568756e-01,
                                  2.2658427e-01 };
        size_t N = v.size();
        DSmatrix<Tdata, backend> vecOut(1, N);
        backend<Tdata>::memory::copy_h2d(vecOut.data(), v.data(), N);
        return vecOut;

    } else if (type == SL_WAVELET) {

    } else if (type == SL_DIRECTIONAL_TEST) {

        std::vector<Tdata> v = { 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0 };
        size_t N = v.size();
        DSmatrix<Tdata, backend> vecOut(3, 3);
        backend<Tdata>::memory::copy_h2d(vecOut.data(), v.data(), N);
        return vecOut;
    } else {

        std::vector<Tdata> v = { 1.0 ,
                                 2.0 ,
                                 3.0 ,
                                 4.0 ,
                                 5.0 };
        size_t N = v.size();
        DSmatrix<Tdata, backend> vecOut(1, N);
        backend<Tdata>::memory::copy_h2d(vecOut.data(), v.data(), N);
        return vecOut;
    }
}
