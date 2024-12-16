/*
 * @file ImageLoader.cpp
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

#include<iostream>
#include<cassert>
#include "src/images/ImageLoader.hpp"

DSmatrix<float, cpu_impl> ILload(std::string path)
{

    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    assert( image.data );
    std::cout << image.rows << ", " << image.cols << std::endl;
    cv::Mat imageFloat;
    image.convertTo(imageFloat,CV_32FC1);
    float * data = (float *)imageFloat.data;
    DSmatrix<float, cpu_impl> imgMat(imageFloat.rows, imageFloat.cols);
    float * dataOut = imgMat.data();
    memcpy(dataOut, data, imageFloat.rows * imageFloat.cols * sizeof(float));
    return imgMat;
}

void ILdump(std::string path, DSmatrix<float, cpu_impl>& mat) {

    t_dims dims = mat.dims();
    cv::imwrite(path,  cv::Mat(dims.rows, dims.cols, CV_32FC1, mat.data()));
}

void ILaddNoise(DSmatrix<float, cpu_impl>& image, float noise) {

    t_dims dims = image.dims();

    std::cout << image(0,0) << std::endl;
    for (size_t i = 0; i < dims.rows; ++i)
        for (size_t j = 0; j < dims.cols; ++j)
            image(i,j) *= (1 + noise * 2 * ( (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - 0.5 ));
    std::cout << image(0,0) << std::endl;
}