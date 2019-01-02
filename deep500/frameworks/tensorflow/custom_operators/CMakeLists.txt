cmake_minimum_required(VERSION 3.1)
project(d500_custom_op)

# Additional variables
set(TENSORFLOW_PATH "" CACHE STRING "Path to tensorflow module")


include_directories(${TENSORFLOW_PATH}/include)
include_directories(${TENSORFLOW_PATH}/include/external/nsync/public)

add_definitions(-DD500_OPNAME=${D500_OPNAME})

# Compiler-specific definitions
if (MSVC_IDE)
    add_definitions(-DCOMPILER_MSVC;-DNOMINMAX)
    link_directories(${TENSORFLOW_PATH}/python)
else()
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
    link_directories(${TENSORFLOW_PATH})
endif()

include(${CMAKE_SOURCE_DIR}/../../reference/custom_operators/CMakeLists.txt)

if (MSVC_IDE)
    target_link_libraries(${D500_OPNAME} pywrap_tensorflow_internal)
else()
    target_link_libraries(${D500_OPNAME} tensorflow_framework)
endif()
