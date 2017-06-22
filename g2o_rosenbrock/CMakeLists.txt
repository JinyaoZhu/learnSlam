cmake_minimum_required(VERSION 2.8)

project(g2o_rosenbrock)

set( CMAKE_CXX_COMPILER "g++" )
set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3" )


list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

#Eigen
include_directories("/usr/include/eigen3")

# G2O
find_package( G2O REQUIRED )
include_directories( ${G2O_INCLUDE_DIRS} )

#CSparse
find_package( CSparse REQUIRED )
include_directories(${CSPARSE_INCLUDE_DIR})

# OpenCV
find_package( OpenCV 3.2 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

#Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

add_executable(g2o_rosenbrock main.cpp)

target_link_libraries(
    g2o_rosenbrock 
    ${OpenCV_LIBS}
    ${CSPARSE_LIBRARY}
    ${CERES_LIBRARIES} 
    g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension)


