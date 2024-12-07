/*
 * @file dataStruct.hpp
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

#ifndef DATASTRUCT_HPP_
#define DATASTRUCT_HPP_

struct t_dims {
    unsigned int rows;
    unsigned int cols;
};
typedef struct t_dims t_dims;

template <typename Tdata, template <class> class  backend>
class DSmatrix
{
public:
    DSmatrix(unsigned int rows, unsigned int cols);
    DSmatrix(unsigned int rows, unsigned int cols, Tdata value);
    DSmatrix(unsigned int rows, unsigned int cols, Tdata *ptr);
    DSmatrix(DSmatrix& inMat);
    ~DSmatrix();
    Tdata * data();
    Tdata * begin();
    Tdata * end();
    t_dims dims();
    unsigned int size();
    void normalize();

private:
    unsigned int mRows;
    unsigned int mCols;
    bool mNeedAlloc;
    Tdata* mData;
};

#endif
