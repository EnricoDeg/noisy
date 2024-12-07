# - Find cuFFT
#  Find cuFFT includes and library
#
#  CUFFT_INCLUDES    - where to find cufft.h
#  CUFFT_LIBRARIES   - List of libraries when using cuFFT.
#  CUFFT_FOUND       - True if cuFFT found.

if (CUFFT_INCLUDES)
  # Already in cache, be silent
  set (CUFFT_FIND_QUIETLY TRUE)
endif (CUFFT_INCLUDES)

find_path (CUFFT_INCLUDES cufft.h
  HINTS "${CUFFT_ROOT}/include" "$ENV{CUFFT_ROOT}/include")

string(REGEX REPLACE "/include/?$" "/lib"
  CUFFT_LIB_HINT ${CUFFT_INCLUDES})

find_library (CUFFT_LIBRARIES
  NAMES cufft
  HINTS ${CUFFT_LIB_HINT})

if ((NOT CUFFT_LIBRARIES) OR (NOT CUFFT_INCLUDES))
  message(STATUS "Trying to find cuFFT using LD_LIBRARY_PATH (we're desperate)...")

  file(TO_CMAKE_PATH "$ENV{LD_LIBRARY_PATH}" LD_LIBRARY_PATH)

  find_library(CUFFT_LIBRARIES
    NAMES cufft
    HINTS ${LD_LIBRARY_PATH})

  if (CUFFT_LIBRARIES)
    get_filename_component(CUFFT_LIB_DIR ${CUFFT_LIBRARIES} PATH)
    string(REGEX REPLACE "/lib/?$" "/include"
      CUFFT_H_HINT ${CUFFT_LIB_DIR})

    find_path (CUFFT_INCLUDES cufft.h
      HINTS ${CUFFT_H_HINT}
      DOC "Path to cufft.h")
  endif()
endif()

# handle the QUIETLY and REQUIRED arguments and set CUFFT_FOUND to TRUE if
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (CUFFT DEFAULT_MSG CUFFT_LIBRARIES CUFFT_INCLUDES)

mark_as_advanced (CUFFT_LIBRARIES CUFFT_INCLUDES)
