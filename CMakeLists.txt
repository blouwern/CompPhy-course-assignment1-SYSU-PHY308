cmake_minimum_required(VERSION 3.10)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(MPIAssignment)
find_package(MPI REQUIRED)
find_package(BLAS REQUIRED)

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(MPIMatrixMultiply_bsp ${SOURCE_DIR}/MPIMatrixMultiply_bsp.c)
target_link_libraries(MPIMatrixMultiply_bsp MPI::MPI_C
                                            BLAS::BLAS
                                            m)
target_compile_definitions(MPIMatrixMultiply_bsp PRIVATE NON_BLOCKING_COMMUNICATION=1)
target_compile_definitions(MPIMatrixMultiply_bsp PRIVATE MPI_COMPUTATION_USE_BLAS=0)
target_compile_definitions(MPIMatrixMultiply_bsp PRIVATE ADD_RSSEQ_COMPARE=1)
target_compile_definitions(MPIMatrixMultiply_bsp PRIVATE REPORT_DEBUG_INFO=0)

# add_executable(SeqMatrixMultiply ${SOURCE_DIR}/SeqMatrixMultiply.c)
# target_link_libraries(SeqMatrixMultiply BLAS::BLAS)
