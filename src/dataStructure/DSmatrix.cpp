/*
 * @file DSmatrix.cpp
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

#include "src/dataStructure/dataStruct.hpp"

#include "src/dataStructure/cpu/backendCPU.hpp"
#ifdef CUDA
#include "src/dataStructure/cuda/backendCUDA.hpp"
#endif

template <typename Tdata, template <class> class backend>
DSmatrix<Tdata, backend>::DSmatrix(unsigned int rows, unsigned int cols)
: mRows(rows),
  mCols(cols),
  mNeedAlloc(true)
{
    mData = backend<Tdata>::allocate(rows * cols);
}

template <typename Tdata, template <class> class backend>
DSmatrix<Tdata, backend>::DSmatrix(unsigned int rows, unsigned int cols, Tdata value)
: DSmatrix(rows, cols)
{
    backend<Tdata>::fill(mData, mRows*mCols, value);
}

template <typename Tdata, template <class> class backend>
DSmatrix<Tdata, backend>::DSmatrix(unsigned int rows, unsigned int cols, Tdata *ptr)
: mRows(rows),
  mCols(cols),
  mNeedAlloc(false),
  mData(ptr)
{ }

template <typename Tdata, template <class> class backend>
DSmatrix<Tdata, backend>::DSmatrix(DSmatrix& inMat)
: mNeedAlloc(true)
{
    mRows = inMat.mRows;
    mCols = inMat.mCols;
    mData = backend<Tdata>::allocate(rows * cols);
    backend<T>::copy(mData, inMat.mData, mRows*mCols);
}

template <typename Tdata, template <class> class backend>
DSmatrix<Tdata, backend>::~DSmatrix()
{
    if (mNeedAlloc)
        backend<Tdata>::free(mData);
}

template
< typename Tdata,
  template <class> class backend >
inline
Tdata* DSmatrix<Tdata, backend>::data()
{
    return mData;
}

template <typename Tdata, template <class> class backend>
inline
Tdata* DSmatrix<Tdata, backend>::begin()
{
    return mData;
}

template <typename Tdata, template <class> class backend>
inline
Tdata* DSmatrix<Tdata, backend>::end()
{
    return mData + mRows * mCols - 1;
}

template <typename Tdata, template <class> class backend>
t_dims DSmatrix<Tdata, backend>::dims()
{
    return (t_dims){mRows, mCols};
}

template <typename Tdata, template <class> class backend>
inline
unsigned int DSmatrix<Tdata, backend>::size()
{
    return mRows * mCols;
}

template <typename Tdata, template <class> class backend>
void DSmatrix<Tdata, backend>::normalize() {

    backend<Tdata>::normalize(mData, mRows * mCols);
}

template class DSmatrix<float, cpu_impl>;
template class DSmatrix<float, cuda_impl>;

