cmake_minimum_required(VERSION 3.1)
project(d500_custom_op)

# Additional variables
set(PYTORCH_PATH "" CACHE STRING "Path to pytorch module")

include_directories(${PYTORCH_PATH}/lib/include)
link_directories(${PYTORCH_PATH}/lib)

add_definitions(-DD500_OPNAME=${D500_OPNAME})

include(${CMAKE_SOURCE_DIR}/../../reference/custom_operators/CMakeLists.txt)

if (MSVC_IDE)
    target_link_libraries(${D500_OPNAME} ATen)
else()
    target_link_libraries(${D500_OPNAME} caffe2 caffe2_gpu)
endif()