# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/35d0c365609296fa4730d62057c487e3cfa030ff.zip
)
# Do NOT install gtest with the project
option(INSTALL_GTEST "Enable installation of googletest." OFF)
FetchContent_MakeAvailable(googletest)

enable_testing()

if (ENABLE_CUDA)
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -DCUDA ")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDA ")
endif()

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${FFTW_INCLUDES})
if(ENABLE_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
    include_directories(${CUFFT_INCLUDES})
    include_directories(${CUALGO_INCLUDES})
endif()

# DSmatrix
if(ENABLE_CUDA)
    set_source_files_properties(dataStructure/test_DSmatrix.cpp PROPERTIES LANGUAGE CUDA)
endif()
add_executable( test_DSmatrix
                dataStructure/test_DSmatrix.cpp
              )
target_link_libraries(test_DSmatrix noisy)
target_link_libraries(test_DSmatrix ${FFTW_LIBRARIES} ${FFTWF_LIBRARIES})
if (ENABLE_CUDA)
    target_link_libraries(test_DSmatrix ${CUDA_LIBRARIES})
    set_property(TARGET test_DSmatrix PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries (test_DSmatrix ${CUFFT_LIBRARIES} ${CUALGO_LIBRARIES})
endif()
target_link_libraries(test_DSmatrix GTest::gtest_main)

# FourierTransform
if(ENABLE_CUDA)
    set_source_files_properties(fourier/test_FourierTransform.cpp PROPERTIES LANGUAGE CUDA)
endif()
add_executable( test_FourierTransform
                fourier/test_FourierTransform.cpp
              )
target_link_libraries(test_FourierTransform noisy)
target_link_libraries(test_FourierTransform ${FFTW_LIBRARIES} ${FFTWF_LIBRARIES})
if (ENABLE_CUDA)
    target_link_libraries(test_FourierTransform ${CUDA_LIBRARIES})
    set_property(TARGET test_FourierTransform PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries (test_FourierTransform ${CUFFT_LIBRARIES} ${CUALGO_LIBRARIES})
endif()
target_link_libraries(test_FourierTransform GTest::gtest_main)

# Transform
if(ENABLE_CUDA)
    set_source_files_properties(transform/test_transformMatrix.cpp PROPERTIES LANGUAGE CUDA)
endif()
add_executable( test_transformMatrix
                transform/test_transformMatrix.cpp
              )
target_link_libraries(test_transformMatrix noisy)
target_link_libraries(test_transformMatrix ${FFTW_LIBRARIES} ${FFTWF_LIBRARIES})
if (ENABLE_CUDA)
    target_link_libraries(test_transformMatrix ${CUDA_LIBRARIES})
    set_property(TARGET test_transformMatrix PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries (test_transformMatrix ${CUFFT_LIBRARIES} ${CUALGO_LIBRARIES})
endif()
target_link_libraries(test_transformMatrix GTest::gtest_main)

# Shearlet
if(ENABLE_CUDA)
    set_source_files_properties(shearlet/test_SLcoeffs.cpp PROPERTIES LANGUAGE CUDA)
endif()
add_executable( test_SLcoeffs
                shearlet/test_SLcoeffs.cpp
              )
target_link_libraries(test_SLcoeffs noisy)
target_link_libraries(test_SLcoeffs ${FFTW_LIBRARIES} ${FFTWF_LIBRARIES})
if (ENABLE_CUDA)
    target_link_libraries(test_SLcoeffs ${CUDA_LIBRARIES})
    set_property(TARGET test_SLcoeffs PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries (test_SLcoeffs ${CUFFT_LIBRARIES} ${CUALGO_LIBRARIES})
endif()
target_link_libraries(test_SLcoeffs GTest::gtest_main)

# Add all tests to GoogleTest
include(GoogleTest)
gtest_discover_tests(test_DSmatrix)
gtest_discover_tests(test_FourierTransform)
gtest_discover_tests(test_transformMatrix)
gtest_discover_tests(test_SLcoeffs)
