cmake_minimum_required(VERSION 3.10)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(MPIAssignment)
find_package(MPI REQUIRED)
find_package(BLAS REQUIRED)

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(comprehensive_demo ${SOURCE_DIR}/comprehensive_demo.c)
target_link_libraries(comprehensive_demo MPI::MPI_C
  BLAS::BLAS
  m)
target_compile_definitions(comprehensive_demo PRIVATE NON_BLOCKING_COMMUNICATION=1)
target_compile_definitions(comprehensive_demo PRIVATE MPI_COMPUTATION_USE_BLAS=1)
target_compile_definitions(comprehensive_demo PRIVATE ADD_RSSEQ_COMPARE=1)
target_compile_definitions(comprehensive_demo PRIVATE REPORT_DEBUG_INFO=1)

add_executable(MM_cblas_mpi ${SOURCE_DIR}/MM_cblas_mpi.c)
target_link_libraries(MM_cblas_mpi MPI::MPI_C
  BLAS::BLAS
  m)
target_compile_definitions(MM_cblas_mpi PRIVATE NON_BLOCKING_COMMUNICATION=1)
target_compile_definitions(MM_cblas_mpi PRIVATE REPORT_DEBUG_INFO=0)

add_executable(MM_RS_mpi ${SOURCE_DIR}/MM_RS_mpi.c)
target_link_libraries(MM_RS_mpi MPI::MPI_C
  BLAS::BLAS
  m)
target_compile_definitions(MM_RS_mpi PRIVATE NON_BLOCKING_COMMUNICATION=1)
target_compile_definitions(MM_RS_mpi PRIVATE REPORT_DEBUG_INFO=0)

add_executable(MM_cblas_seq ${SOURCE_DIR}/MM_cblas_seq.c)
target_link_libraries(MM_cblas_seq BLAS::BLAS
  m)

add_executable(MM_RS_seq ${SOURCE_DIR}/MM_RS_seq.c)
target_link_libraries(MM_RS_seq BLAS::BLAS
  m)

