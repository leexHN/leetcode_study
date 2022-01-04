project(${PROJECT_NAME}_2022)

add_executable(SortAlgorithm sort_algorithm.cpp)
target_link_libraries(SortAlgorithm
        ${GTEST_BOTH_LIBRARIES}
        pthread
        )

add_executable(LeetCodeTest leetcode.cpp)
target_link_libraries(LeetCodeTest
        ${GTEST_BOTH_LIBRARIES}
        pthread
        )
