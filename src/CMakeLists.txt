include_directories(${PROJECT_SOURCE_DIR})
include_directories(${FFTW_INCLUDES})
if(ENABLE_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
    include_directories(${CUFFT_INCLUDES})
    include_directories(${CUALGO_INCLUDES})
endif()

if (ENABLE_CUDA)
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -DCUDA ")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDA ")
endif()

set(SOURCE_EXE dataStructure/DSmatrix.cpp)

if (ENABLE_CUDA)
    set(SOURCE_CUDA dataStructure/cuda/backendCUDA.cu)
endif()

add_library(noisy SHARED ${SOURCE_CUDA} ${SOURCE_EXE})
target_link_libraries(noisy ${FFTW_LIBRARIES} ${FFTWF_LIBRARIES})
if(ENABLE_CUDA)
    target_link_libraries(noisy ${CUDA_LIBRARIES})
    set_property(TARGET noisy PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(noisy ${CUFFT_LIBRARIES} ${CUALGO_LIBRARIES})
endif()

# Install the library
install (
  TARGETS noisy
  LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/lib
  ARCHIVE DESTINATION ${PROJECT_SOURCE_DIR}/lib)

# ------------ Header Files	
install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/
  DESTINATION ${PROJECT_SOURCE_DIR}/include
  FILES_MATCHING
  PATTERN "*.hpp"
  )
