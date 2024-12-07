cmake_minimum_required (VERSION 3.20)

set(LANG_CXX CXX)
set(LANG_C C)

if(ENABLE_CUDA)
    set(LANG_CUDA CUDA)
endif()

project(Noisy ${LANG_CXX} ${LANG_C} ${LANG_CUDA})

include("CMake/FindFFTW.cmake")
if(ENABLE_CUDA)
    find_package(CUDA QUIET REQUIRED)
    include("CMake/FindcuFFT.cmake")
    include("CMake/FindcuAlgo.cmake")
endif()

# Require C++17 compiler support.
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)

# add_compile_options(-fPIC)

add_subdirectory(src)
add_subdirectory(tests)
