cmake_minimum_required(VERSION 3.10)
set(PROJECT_NAME leet_code)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 11)

find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})

# old study execable comment
#add_executable(${PROJECT_NAME} main.cpp)
#
#target_link_libraries(${PROJECT_NAME}
#        ${GTEST_BOTH_LIBRARIES}
#        pthread )

add_subdirectory(2022)