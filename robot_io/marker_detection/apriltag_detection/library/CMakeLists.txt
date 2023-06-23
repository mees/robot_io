cmake_minimum_required(VERSION 2.6)
project(library)
set (CMAKE_CXX_STANDARD 11)

include_directories("./include/" )

# OpenCV is needed
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})
# message(STATUS "${OpenCV_INCLUDE_DIRS}")
message(STATUS "${OpenCV_LIBS}")

# EIGEN is needed
find_package(Eigen3 REQUIRED)

set ( CPP_FILES ./src/FloatImage.cc ./src/TagDetection.cc ./src/TagDetector.cc
                ./src/TagFamily.cc ./src/Edge.cc ./src/Gaussian.cc ./src/GLine2D.cc
                ./src/GLineSegment2D.cc ./src/GrayModel.cc  ./src/Homography33.cc
                ./src/MathUtil.cc ./src/Quad.cc ./src/Segment.cc ./src/UnionFindSimple.cc)
add_executable(library main.cpp ${CPP_FILES})

add_library(apriltaglib SHARED ${CPP_FILES} )
target_link_libraries(apriltaglib ${OpenCV_LIBS})

target_link_libraries(library ${OpenCV_LIBS} -lpthread)
install(TARGETS library RUNTIME DESTINATION bin)
