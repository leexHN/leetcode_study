cmake_minimum_required(VERSION 3.15)
set(PROJECT_NAME leet_code)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 11)

find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME}
        ${GTEST_BOTH_LIBRARIES}
        pthread )