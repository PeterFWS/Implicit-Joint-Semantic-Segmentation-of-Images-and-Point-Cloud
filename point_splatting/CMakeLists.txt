cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(render_point_views)

#set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})
#include(CheckCXXCompilerFlag)
#CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
#CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
#if(COMPILER_SUPPORTS_CXX11)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#elseif(COMPILER_SUPPORTS_CXX0X)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
#else()
#        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
#endif()
find_package(PCL REQUIRED )
find_package( OpenCV REQUIRED )

include_directories(${PCL_INCLUDE_DIRS} include)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
set (PROJECT_POINTCLOUD_SOURCE_DIR src)
set (PROJECT_POINTCLOUD_INCLUDE_DIR include)

set(project_pointcloud_src render_views.cpp
		${PROJECT_POINTCLOUD_SOURCE_DIR}/point_cloud_to_depth.cpp
    ${PROJECT_POINTCLOUD_SOURCE_DIR}/cloud_downsampling.cpp
)

#set(project_pointcloud_include ${PROJECT_POINTCLOUD_INCLUDE_DIR}/point_cloud_to_depth.h)

add_executable (render_point_views ${project_pointcloud_src})
target_link_libraries(render_point_views ${PCL_LIBRARIES} ${OpenCV_LIBS})	
