cmake_minimum_required(VERSION 2.8.3)
project(la3dm)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")

find_package(OpenMP)
if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        message("OpenMP enabled")
endif()

## Enable OpenMP
add_definitions(-DOPENMP)

## Enable Debug Info
#add_definitions(-DDEBUG)

# add_definitions(-DPREDICT)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
        geometry_msgs
        visualization_msgs
	pcl_conversions
        nav_msgs
        pcl_ros
        roscpp
	sensor_msgs
        octomap_ros
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependencies might have been
##     pulled in transitively but can be declared for certainty nonetheless:
##     * add a build_depend tag for "message_generation"
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   geometry_msgs
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES la3dm
  #CATKIN_DEPENDS pcl_ros octomap_ros geometry_msgs visualization_msgs roscpp
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
        include/gpoctomap
        include/bgkoctomap
        include/bgkloctomap
	include/bgklvoctomap
        include/common
        ${catkin_INCLUDE_DIRS}
        ${OCTOMAP_INCLUDE_DIRS}
)

## Declare a cpp library
# add_library(bgkoctomap
#   src/${PROJECT_NAME}/bgkoctomap.cpp
# )

# Common source files
set(COMMON_SOURCE_FILES
  src/common/point3f.cpp
  src/common/point6f.cpp)

# Files for BGKL source
set(BGKLV_SOURCE_FILES
  src/bgklvoctomap/bgklvblock.cpp
  src/bgklvoctomap/bgklvoctomap.cpp
  src/bgklvoctomap/bgklvoctree_node.cpp
  src/bgklvoctomap/bgklvoctree.cpp
)

# Files for BGKL source
set(BGKL_SOURCE_FILES
  src/bgkloctomap/bgklblock.cpp
  src/bgkloctomap/bgkloctomap.cpp
  src/bgkloctomap/bgkloctree_node.cpp
  src/bgkloctomap/bgkloctree.cpp
)

# Files for BGK source
set(BGK_SOURCE_FILES
  src/bgkoctomap/bgkblock.cpp
  src/bgkoctomap/bgkoctomap.cpp
  src/bgkoctomap/bgkoctree_node.cpp
  src/bgkoctomap/bgkoctree.cpp
)

# Files for GP source
set(GP_SOURCE_FILES
  src/gpoctomap/gpblock.cpp
  src/gpoctomap/gpoctomap.cpp
  src/gpoctomap/gpoctree_node.cpp
  src/gpoctomap/gpoctree.cpp
)

add_executable(bgklvoctomap_static_node
    ${BGKLV_SOURCE_FILES}
    ${COMMON_SOURCE_FILES}
    src/bgklvoctomap/bgklvoctomap_static_node.cpp
)

add_executable(bgkloctomap_static_node
    ${BGKL_SOURCE_FILES}
    ${COMMON_SOURCE_FILES}
    src/bgkloctomap/bgkloctomap_static_node.cpp
)

add_executable(bgkoctomap_static_node
    ${BGK_SOURCE_FILES}
    ${COMMON_SOURCE_FILES}
    src/bgkoctomap/bgkoctomap_static_node.cpp
)

add_executable(gpoctomap_static_node
    ${GP_SOURCE_FILES}
    ${COMMON_SOURCE_FILES}
    src/gpoctomap/gpoctomap_static_node.cpp
)

add_executable(bgklvoctomap_server
    ${BGKLV_SOURCE_FILES}
    ${COMMON_SOURCE_FILES}
    src/bgklvoctomap/bgklvoctomap_server.cpp
)

add_executable(bgkloctomap_server
    ${BGKL_SOURCE_FILES}
    ${COMMON_SOURCE_FILES}
    src/bgkloctomap/bgkloctomap_server.cpp
)

add_executable(bgkoctomap_server
    ${BGK_SOURCE_FILES}
    ${COMMON_SOURCE_FILES}
    src/bgkoctomap/bgkoctomap_server.cpp
)

add_executable(gpoctomap_server
    ${GP_SOURCE_FILES}
    ${COMMON_SOURCE_FILES}
    src/gpoctomap/gpoctomap_server.cpp
)

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
# add_dependencies(BGKOctoMap_node BGKOctoMap_generate_messages_cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(bgklvoctomap_static_node
        ${catkin_LIBRARIES}
)

target_link_libraries(bgkloctomap_static_node
        ${catkin_LIBRARIES}
)

target_link_libraries(bgkoctomap_static_node
        ${catkin_LIBRARIES}
)

target_link_libraries(gpoctomap_static_node
        ${catkin_LIBRARIES}
)

target_link_libraries(bgklvoctomap_server
        ${catkin_LIBRARIES}
)

target_link_libraries(bgkloctomap_server
        ${catkin_LIBRARIES}
)

target_link_libraries(bgkoctomap_server
        ${catkin_LIBRARIES}
)

target_link_libraries(gpoctomap_server
        ${catkin_LIBRARIES}
)

link_directories(/usr/local/lib)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS bgkoctomap BGKOctoMap_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_BGKOctoMap.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
