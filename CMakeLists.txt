cmake_minimum_required(VERSION 3.5)
cmake_policy(VERSION 3.5)
project(TwoAC)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Gflags REQUIRED)
find_package(Glog REQUIRED)

include_directories(
	third_party/theia
)

###############################################################################
# Theia #######################################################################
###############################################################################

file(GLOB_RECURSE THEIA_HDRS "third_party/theia/*.h") 
file(GLOB_RECURSE THEIA_SRCS "third_party/theia/*.cc") 

add_library(Theia STATIC
	${THEIA_HDRS} ${THEIA_SRCS}
)
target_link_libraries(Theia
	Eigen3::Eigen
	glog::glog
)
###############################################################################
# TwoAC #######################################################################
###############################################################################
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

file(GLOB TWOAC_HDRS TwoAC/*.h TwoAC/*.hpp)
file(GLOB TWOAC_SRCS TwoAC/*.cpp)

add_executable(TwoAC 
	${TWOAC_HDRS} ${TWOAC_SRCS}
)
target_link_libraries(TwoAC 
	Eigen3::Eigen 
	Theia
)
add_dependencies(TwoAC Theia)
