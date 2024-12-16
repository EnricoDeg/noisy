## Noisy
Description is wip ...

##### Quick Start
Dependencies:
 - CMake
 - CUDA
 - FFTW
 - cuFFT
 - cuAlgo
 - OpenCV

Installation:
```
export CUFFT_ROOT=/path/to/cufft/root
export CUALGO_ROOT=/path/to/cualgo/build
export FFTW_ROOT=/path/to/fftw/root
mkdir build
cd build
cmake ..
make install
```

Run tests:
```
cd build/tests
ctest
```
