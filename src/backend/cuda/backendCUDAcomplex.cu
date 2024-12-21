/*
 * @file backendCUDAcomplex.cu
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

#include "cuAlgo.hpp"

#include <cassert>

template<typename Tdata>
__global__ void corrComplexKernel(thrust::complex<Tdata> * __restrict__ dataIn1,
                                  thrust::complex<Tdata> * __restrict__ dataIn2,
                                  thrust::complex<Tdata> * __restrict__ dataOut,
                                  unsigned int size) {

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	while (i < size) {

		dataOut[i] = dataIn1[i] * thrust::conj(dataIn2[i]);
		i += gridDim.x * blockDim.x;
	}
}

template<typename Tdata>
__global__ void convComplexKernel(thrust::complex<Tdata> * __restrict__ dataIn1,
                                  thrust::complex<Tdata> * __restrict__ dataIn2,
                                  thrust::complex<Tdata> * __restrict__ dataOut,
                                  unsigned int size) {

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	while (i < size) {

		dataOut[i] = dataIn1[i] * dataIn2[i];
		i += gridDim.x * blockDim.x;
	}
}

template <typename Tdata>
void cuda_complex_impl<Tdata>::op::corrComplex(thrust::complex<Tdata> * __restrict__ dataIn1,
                                               thrust::complex<Tdata> * __restrict__ dataIn2,
                                               thrust::complex<Tdata> * __restrict__ dataOut,
                                               unsigned int size) {

    dim3 threadsPerBlock(THREADS_PER_BLOCK);
    dim3 blocksPerGrid(div_ceil(size, THREADS_PER_BLOCK));
    corrComplexKernel<Tdata><<<blocksPerGrid, threadsPerBlock>>>(dataIn1, dataIn2, dataOut, size);
    check_cuda( cudaStreamSynchronize(0) );
}

template <typename Tdata>
void cuda_complex_impl<Tdata>::op::convComplex(thrust::complex<Tdata> * __restrict__ dataIn1,
                                               thrust::complex<Tdata> * __restrict__ dataIn2,
                                               thrust::complex<Tdata> * __restrict__ dataOut,
                                               unsigned int size) {

    dim3 threadsPerBlock(THREADS_PER_BLOCK);
    dim3 blocksPerGrid(div_ceil(size, THREADS_PER_BLOCK));
    convComplexKernel<Tdata><<<blocksPerGrid, threadsPerBlock>>>(dataIn1, dataIn2, dataOut, size);
    check_cuda( cudaStreamSynchronize(0) );
}
