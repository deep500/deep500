cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 14)

# 2) find caffe2 package
if (NOT CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
    find_package(Caffe2 REQUIRED)
endif()

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

add_library(caffe2_mpi_operations SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/mpi_operations.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/mpi_operations.h
        )

target_link_libraries(caffe2_mpi_operations caffe2 ${MPI_LIBRARIES})
install(TARGETS caffe2_mpi_operations DESTINATION lib)
