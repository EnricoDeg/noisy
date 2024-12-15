/*
 * @file backendCUDAop.cu
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

#include "src/backend/cuda/backendCUDA.hpp"

#include <cassert>

#include "cuAlgo.hpp"

template<typename T>
__global__ void sumInPlaceKernel(T            * __restrict__ data1,
                                 const T      * __restrict__ data2,
                                 unsigned int                size ) {

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	while (i < size) {

		data1[i] += data2[i];
		i += gridDim.x * blockDim.x;
	}
}

template <typename Tdata>
void cuda_impl<Tdata>::op::normalize(Tdata * __restrict__ data, unsigned int size) {

    cuAlgo::normalizeVector(data, size);
}

template <typename Tdata>
void cuda_impl<Tdata>::op::fliplr(Tdata * __restrict__ data, unsigned int dim,
                                  unsigned int mRows, unsigned int mCols) {

    assert(dim == 0 || dim == 1);
    cuAlgo::fliplr1dMatrix(data, dim, mRows , mCols);
}

template <typename Tdata>
void cuda_impl<Tdata>::op::sumInPlace(Tdata * __restrict__ data1,
                                      const Tdata * __restrict__ data2,
                                      unsigned int size) {

    dim3 threadsPerBlock(THREADS_PER_BLOCK);
    dim3 blocksPerGrid(div_ceil(size, THREADS_PER_BLOCK));
    sumInPlaceKernel<Tdata><<<blocksPerGrid, threadsPerBlock>>>(data1, data2, size);
    check_cuda( cudaStreamSynchronize(0) );
}
