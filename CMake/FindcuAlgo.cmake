# - Find cuAlgo
#  Find cuAlgo includes and library
#
#  CUALGO_INCLUDES    - where to find cuAlgo.hpp
#  CUALGO_LIBRARIES   - List of libraries when using cuAlgo.
#  CUALGO_FOUND       - True if cuAlgo found.

if (CUALGO_INCLUDES)
  # Already in cache, be silent
  set (CUAGO_FIND_QUIETLY TRUE)
endif (CUALGO_INCLUDES)

find_path (CUALGO_INCLUDES cuAlgo.hpp
  HINTS "${CUALGO_ROOT}/src" "$ENV{CUALGO_ROOT}/src")

string(REGEX REPLACE "/src/?$" "/lib"
  CUALGO_LIB_HINT ${CUALGO_INCLUDES})

find_library (CUALGO_LIBRARIES
  NAMES cuAlgo
  HINTS ${CUALGO_LIB_HINT})

if ((NOT CUALGO_LIBRARIES) OR (NOT CUALGO_INCLUDES))
  message(STATUS "Trying to find cuAlgo using LD_LIBRARY_PATH (we're desperate)...")

  file(TO_CMAKE_PATH "$ENV{LD_LIBRARY_PATH}" LD_LIBRARY_PATH)

  find_library(CUALGO_LIBRARIES
    NAMES cuAlgo
    HINTS ${LD_LIBRARY_PATH})

  if (CUALGO_LIBRARIES)
    get_filename_component(CUALGO_LIB_DIR ${CUALGO_LIBRARIES} PATH)
    string(REGEX REPLACE "/lib/?$" "/include"
      CUALGO_H_HINT ${CUALGO_LIB_DIR})

    find_path (CUALGO_INCLUDES cuAlgo.hpp
      HINTS ${CUALGO_H_HINT}
      DOC "Path to cuAlgo.hpp")
  endif()
endif()

# handle the QUIETLY and REQUIRED arguments and set CUALGO_FOUND to TRUE if
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (CUALGO DEFAULT_MSG CUALGO_LIBRARIES CUALGO_INCLUDES)

mark_as_advanced (CUALGO_LIBRARIES CUALGO_INCLUDES)
