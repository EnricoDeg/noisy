cmake_minimum_required (VERSION 3.20)

set(LANG_CXX CXX)
set(LANG_C C)

if(ENABLE_CUDA)
    set(LANG_CUDA CUDA)
endif()

project(Noisy ${LANG_CXX} ${LANG_C} ${LANG_CUDA})

include("CMake/FindFFTW.cmake")
if(ENABLE_CUDA)
    include("CMake/FindcuFFT.cmake")
    include("CMake/FindcuAlgo.cmake")
endif()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(docs)