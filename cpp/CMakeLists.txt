cmake_minimum_required(VERSION 3.15)
project(femsolver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# get Eigen (header-only, make sure to install Eigen package)
find_package(Eigen3 REQUIRED NO_MODULE)

add_library(femsolver SHARED
  fem_solver.cpp
)

target_include_directories(femsolver
  PRIVATE
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(femsolver
  PRIVATE
    Eigen3::Eigen
)

# macOS shared lib name hint
set_target_properties(femsolver PROPERTIES
  OUTPUT_NAME "femsolver"
)