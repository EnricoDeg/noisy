/*
 * @file backendCUDA.cu
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

#include "src/dataStructure/cuda/backendCUDA.hpp"

#include "cuAlgo.hpp"

#define THREADS_PER_BLOCK 1024

__device__ __host__ int div_ceil(int numerator, int denominator)
{

	return (numerator % denominator != 0) ?
	       (numerator / denominator+ 1  ) :
	       (numerator / denominator     ) ;
}

template<typename T>
__global__ void fillKernel(T            * __restrict__ data,
                           unsigned int                size,
                           T                          value) {

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	while (i < size) {

		data[i] = value;
		i += gridDim.x * blockDim.x;
	}
}

template <typename Tdata>
Tdata * cuda_impl<Tdata>::allocate(unsigned int elements) {
    Tdata *p;
    check_cuda( cudaMalloc(&p, elements * sizeof(Tdata)) );
    return p;
}

template <typename Tdata>
void cuda_impl<Tdata>::free(Tdata * data) {
    check_cuda( cudaFree(data) );
}

template <typename Tdata>
void cuda_impl<Tdata>::copy(Tdata *dst, Tdata *src, unsigned int size) {
    check_cuda( cudaMemcpy(dst, src, size*sizeof(Tdata), cudaMemcpyDeviceToDevice) );
}

template <typename Tdata>
void cuda_impl<Tdata>::fill(Tdata * __restrict__ data, unsigned int size, Tdata value) {
    dim3 threadsPerBlock(THREADS_PER_BLOCK);
    dim3 blocksPerGrid(div_ceil(size, THREADS_PER_BLOCK));
    fillKernel<Tdata><<<blocksPerGrid, threadsPerBlock>>>(data, size, value);
    check_cuda( cudaStreamSynchronize(0) );
}

template <typename Tdata>
void cuda_impl<Tdata>::normalize(Tdata * __restrict__ data, unsigned int size) {
    cuAlgo::normalizeVector(data, size);
}

template class cuda_impl<float>;
