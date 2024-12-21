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

#include "src/backend/cpu/backendCPU.hpp"
#ifdef CUDA
#include "src/backend/cuda/backendCUDA.hpp"
#endif

// constructors
template <typename Tdata, template <class> class backend>
DSmatrix<Tdata, backend>::DSmatrix(unsigned int rows, unsigned int cols)
: mRows(rows),
  mCols(cols),
  mNeedAlloc(true)
{
    mData = backend<Tdata>::memory::allocate(rows * cols);
}

template <typename Tdata, template <class> class backend>
DSmatrix<Tdata, backend>::DSmatrix(unsigned int rows, unsigned int cols, Tdata value)
: DSmatrix(rows, cols)
{
    backend<Tdata>::memory::fill(mData, mRows*mCols, value);
}

template <typename Tdata, template <class> class backend>
DSmatrix<Tdata, backend>::DSmatrix(unsigned int rows, unsigned int cols, Tdata *ptr)
: mRows(rows),
  mCols(cols),
  mNeedAlloc(false),
  mData(ptr)
{ }

template <typename Tdata, template <class> class backend>
DSmatrix<Tdata, backend>::DSmatrix(const DSmatrix& inMat)
: mNeedAlloc(true)
{
    mRows = inMat.mRows;
    mCols = inMat.mCols;
    mData = backend<Tdata>::memory::allocate(mRows * mCols);
    backend<Tdata>::memory::copy(mData, inMat.mData, mRows*mCols);
}

// destructor
template <typename Tdata, template <class> class backend>
DSmatrix<Tdata, backend>::~DSmatrix()
{
    if (mNeedAlloc)
        backend<Tdata>::memory::free(mData);
}

// operators

template <typename Tdata, template <class> class backend>
inline
Tdata& DSmatrix<Tdata, backend>::operator()(unsigned int i, unsigned int j)
{
    return mData[i * mCols + j];
}

template <typename Tdata, template <class> class backend>
inline
Tdata  DSmatrix<Tdata,backend>::operator()(unsigned int i, unsigned int j) const
{
    return mData[i * mCols + j];
}

template <typename Tdata, template <class> class backend>
DSmatrix<Tdata, backend>& DSmatrix<Tdata, backend>::operator+=(const DSmatrix<Tdata, backend>& B) {

    backend<Tdata>::op::sumInPlace(mData, B.data(), mRows * mCols);

    return *this;
}

template <typename Tdata, template <class> class backend>
DSmatrix<Tdata, backend>& DSmatrix<Tdata, backend>::operator*=(const DSmatrix<Tdata, backend>& B) {

    backend<Tdata>::op::prodInPlace(mData, B.data(), mRows * mCols);

    return *this;
}

// inline info
template
< typename Tdata,
  template <class> class backend >
inline
Tdata* DSmatrix<Tdata, backend>::data() const
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
t_dims DSmatrix<Tdata, backend>::dims() const
{
    return (t_dims){mRows, mCols};
}

template <typename Tdata, template <class> class backend>
inline
unsigned int DSmatrix<Tdata, backend>::size()
{
    return mRows * mCols;
}

// in place operations
template <typename Tdata, template <class> class backend>
void DSmatrix<Tdata, backend>::normalize() {

    backend<Tdata>::op::normalize(mData, mRows * mCols);
}

template <typename Tdata, template <class> class backend>
void DSmatrix<Tdata, backend>::normSize() {

    backend<Tdata>::op::divScalarInPlace(mData, mRows * mCols, Tdata(mRows * mCols));
}

template <typename Tdata, template <class> class backend>
void DSmatrix<Tdata, backend>::fliplr(unsigned int dim) {

    backend<Tdata>::op::fliplr(mData, dim, mRows, mCols);
}

template class DSmatrix<float, cpu_impl>;
template class DSmatrix<std::complex<float>, cpu_impl>;
template class DSmatrix<float, cuda_impl>;
template class DSmatrix<thrust::complex<float>, cuda_impl>;
