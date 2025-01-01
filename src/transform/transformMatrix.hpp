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

#include "src/dataStructure/dataStruct.hpp"

template <typename Tdata, template <class> class  backend>
void downsample(const DSmatrix<Tdata, backend>& inMat ,
                      unsigned int              dim   ,
                      unsigned int              stride,
                      DSmatrix<Tdata, backend>& outMat);

template <typename Tdata, template <class> class  backend>
t_dims upsample(const DSmatrix<Tdata, backend>& inMat ,
                      unsigned int              dim   ,
                      unsigned int              nzeros,
                      DSmatrix<Tdata, backend>& outMat);

template <typename Tdata, template <class> class  backend>
t_dims upsample(const DSmatrix<Tdata, backend>& inMat ,
                      unsigned int              dim   ,
                      unsigned int              nzeros);

template <typename Tdata, template <class> class  backend>
void pad(const DSmatrix<Tdata, backend>& inMat ,
               DSmatrix<Tdata, backend>& outMat);

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

template <typename Tdata, template <class> class  backend,
                          template <class> class  backendC>
void convolve(const DSmatrix<Tdata, backend>&  inMat ,
              const DSmatrix<Tdata, backend>&  filter,
                    DSmatrix<Tdata, backend>&  outMat);

#endif
