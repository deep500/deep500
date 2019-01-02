cmake_minimum_required(VERSION 3.1)
project(d500_custom_op)

# Force shared library linking
set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)

# Target options 
set(D500_OPNAME "custom_op" CACHE STRING "Name of custom operator")
set(D500_FILES "" CACHE STRING "Host code files")
set(D500_LIBS "" CACHE STRING "Extra libraries")
set(EXTRA_DEFS "" CACHE STRING "Additional macro definitions")

# Input files from other CMakeLists (if exist)
if(NOT DEFINED D500_CPP_FILES)
  set(D500_CPP_FILES)
endif()
if(NOT DEFINED D500_CUDA_FILES)
  set(D500_CUDA_FILES)
  set(D500_ENABLE_CUDA OFF)
else()
    set(D500_ENABLE_CUDA ON)
endif()
if(NOT DEFINED D500_OBJECTS)
  set(D500_OBJECTS)
endif()

# Split list by file type
foreach(D500_FILE ${D500_FILES})
  get_filename_component(D500_FILE_TARGET ${D500_FILE} EXT)
  if(${D500_FILE_TARGET} STREQUAL ".cu")
    set(D500_ENABLE_CUDA ON)
    set(D500_CUDA_FILES ${D500_CUDA_FILES} ${D500_FILE})
  else()
    set(D500_CPP_FILES ${D500_CPP_FILES} ${D500_FILE})
  endif()
endforeach()

# Internal dependencies
include_directories(${CMAKE_SOURCE_DIR}/../../../lv0/operators/include)

# External dependencies 
find_package(Threads REQUIRED)
set(D500_LIBS ${D500_LIBS} ${CMAKE_THREAD_LIBS_INIT})

find_package(OpenMP COMPONENTS CXX)
if (OpenMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(D500_LIBS ${D500_LIBS} ${OpenMP_CXX_LIBRARIES})
endif()

find_package(MPI)
if (MPI_CXX_FOUND)
  include_directories(${MPI_CXX_INCLUDE_DIRS})
  set(D500_LIBS ${D500_LIBS} ${MPI_CXX_LIBRARIES})
endif()

if(D500_ENABLE_CUDA)
  #enable_language(CUDA)
  find_package(CUDA REQUIRED)
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  include_directories(${CUDA_INCLUDE_DIRS})
  set(D500_LIBS ${D500_LIBS} ${CUDA_LIBRARIES})
endif()


# Add extra macros
foreach(DEF ${EXTRA_DEFS})
    add_definitions("-D${DEF}")
endforeach()

# Create CUDA object files
if(D500_ENABLE_CUDA)
  add_definitions(-DD500_ENABLE_CUDA)
  # Get local CUDA architectures
  if (NOT DEFINED LOCAL_CUDA_ARCHITECTURES)
      execute_process(COMMAND "${CUDA_NVCC_EXECUTABLE}" "--run"
                      "${CMAKE_SOURCE_DIR}/../../../lv0/operators/tools/get_cuda_arch.cpp"
                      OUTPUT_VARIABLE _arch_out RESULT_VARIABLE _arch_res
                      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

      if(_arch_res EQUAL 0)
        string(REGEX REPLACE "\n" ";" _arch_out "${_arch_out}")
        list(GET _arch_out -1 _local_arch)
        string(REGEX REPLACE " " ";" _local_arch "${_local_arch}")
        set(LOCAL_CUDA_ARCHITECTURES "${_local_arch}" CACHE STRING "Detected local GPUs for compilation")
        message("-- Local CUDA architectures detected: ${LOCAL_CUDA_ARCHITECTURES}")
      else()
        set(LOCAL_CUDA_ARCHITECTURES "" CACHE STRING "Detected local GPUs for compilation")
        message("-- No local CUDA-capable GPUs found")
      endif()
  endif()

  # Add flags to compile for local CUDA architectures
  foreach(var ${LOCAL_CUDA_ARCHITECTURES})
    list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_${var},code=sm_${var})
  endforeach()

  cuda_include_directories(${CMAKE_SOURCE_DIR}/../../../lv0/operators/include)

  # Create D500 library file 
  cuda_add_library(${D500_OPNAME} SHARED ${D500_CPP_FILES} ${D500_CUDA_FILES})
  #set_property(TARGET ${D500_OPNAME} PROPERTY CUDA_STANDARD 11)
else()
  # Create D500 library file 
  add_library(${D500_OPNAME} SHARED ${D500_CPP_FILES} ${D500_OBJECTS})
endif()

target_link_libraries(${D500_OPNAME} ${D500_LIBS})

# Windows-specific fixes
if (MSVC_IDE)
    # Copy output DLL from the "Debug" and "Release" directories CMake adds
    add_custom_target(CopyDLL ALL
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:${D500_OPNAME}> "${CMAKE_BINARY_DIR}/lib${D500_OPNAME}.dll"
        DEPENDS ${D500_OPNAME}
        COMMENT "Copying binaries" VERBATIM)

    # Replace /MD with /MT so that CUDA links properly
    # https://stackoverflow.com/a/14172871/6489142
    set(CompilerFlags
        CMAKE_CXX_FLAGS
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_C_FLAGS
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_RELEASE
        )
    foreach(CompilerFlag ${CompilerFlags})
      string(REPLACE "/MD" "/MT" ${CompilerFlag} "${${CompilerFlag}}")
    endforeach()
endif()
