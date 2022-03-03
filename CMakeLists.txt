cmake_minimum_required(VERSION 3.8)
project(rgbd_slam)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "-std=c++14 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE  "-std=c++14 -O3 -fopenmp -pthread")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)


## Dependencies


set( DBoW3_INCLUDE_DIRS "/usr/local/include" )
set( DBoW3_LIBS "/usr/local/lib/libDBoW3.so" )

include_directories(${DBoW3_INCLUDE_DIRS})


# Eigen
include_directories("/usr/include/eigen3")

# OpenCV
find_package(OpenCV 4.5 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# Ceres
find_package(Ceres REQUIRED)
include_directories(${Ceres_INCLUDE_DIRS})

# glog
find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

# csparse
find_package(CSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})

# fmt
find_package(fmt REQUIRED)

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS}) 

include_directories(${PROJECT_SOURCE_DIR})

# #${Pangolin_LIBRARIES} GL GLU GLEW glut
set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        pthread
        ${Sophus_LIBRARIES}
        ${CERES_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${CSPARSE_LIBRARY}
        Sophus::Sophus
        ${PCL_LIBRARIES}
        ${Pangolin_LIBRARIES}
        ${DBoW3_LIBS}
        #${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
        )

include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(src)
add_subdirectory(app)
