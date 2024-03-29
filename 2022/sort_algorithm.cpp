//
// Created by lix on 1/5/22.
//
#include <iostream>
#include <vector>
#include <algorithm>
#include <gtest/gtest.h>
#include <unordered_map>
#include <map>
#include <string>
#include <climits>
#include <queue>
#include <list>
#include <unordered_set>
#include <random>
#include <chrono>

template<typename T>
bool operator==(const std::vector<T> &lhs, const std::vector<T> &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); i++) {
    if (lhs[i] != rhs[i])
      return false;
  }
  return true;
}

template<typename T>
bool IsSorted(const std::vector<T> vec) {
  if (vec.size() <= 1) {
    return true;
  }
  for (size_t i = 0; i < vec.size() - 1; i++) {
    if (vec[i] > vec[i + 1])
      return false;
  }
  return true;
}

std::vector<int> &RandomVecGenerator(size_t size, std::pair<int, int> range = {0, 100}) {
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> distrib(range.first, range.second);

  static std::vector<int> vec;
  vec.resize(size);

  for (auto &item : vec) {
    item = distrib(gen);
  }
  return vec;
}

namespace quick_sort {
// 分而治之的思想

template<typename T>
void QuickSortUtility(std::vector<T> &vec, size_t l, size_t r) { // 左开右闭
  //1. ready to sort vec [l, r), this vector size is r-l
  if (l + 1 >= r) { // same as r-l <= 1.So we assume vec size is 1 is sorted
    return;
  }
  T pivot = vec[l]; // 左边是空闲位置
  size_t l_bak = l;
  size_t r_bak = r; // using for sub vec [l pivot_idx) [pivot_idx r)
  r = r - 1; //now r is r_idx(the most right index)
  while (l < r) { // when l==r exit l==r==pivot_idx
    // 首先左边空闲，需要在右边找到不合符条件的 r_value<pivot
    while (l < r && vec[r] >= pivot) { // 循环中都是符合条件的
      r--;
    }
    // 现在找到r不符合条件,把它移动到空闲位置l
    vec[l] = vec[r];
    // 现在r的位置空闲,左边找不符合条件的 l_value>pivot
    while (l < r && vec[l] <= pivot) {
      l++;
    }
    vec[r] = vec[l];
  }
  size_t pivot_idx = l; // l==r
  vec[pivot_idx] = pivot;
  // 第一次排序完成 满足pivot左边都小于pivot，pivot右边都大于pivot,注：这里pivot已经排好序了，pivot位置不需要改变了
  // 对子序列做相同方式排序
  QuickSortUtility(vec, l_bak, pivot_idx);
  QuickSortUtility(vec, pivot_idx + 1, r_bak); // 这里必须加pivot+1,保证每次递归子vec的size必须比原始的vec要小，
  // 不然在一边子序列为空时，另一边子序列和原始的完全相同，导致死循环
  // 而且pivot位置已经排好序了，不需要变动
}

template<typename T>
void QuickSort(std::vector<T> &vec) {
  // recursive ways
  QuickSortUtility(vec, 0, vec.size());
}

}

namespace merge_sort {

/**
 * @brief 注意每次合并的两个子集合是连续的
 * @param buffer 在两个子集合归并的时候需要一个buffer数组，用于存储两个数组归并之后的结果,
 * 所以buffer.size()应该等于r-l，最大为该vec的size
 * @param l 两个准备归并的集合的头，即左边集合的头部（含）
 * @param r 两个准备归并的集合的尾，即右边集合的尾部（不含）
 */
template<typename T>
void MergeSortUtility(std::vector<T> &vec, std::vector<T> &buffer, size_t l, size_t r) { //左开右闭
  //1. ready to sort vec [l, r), this vector size is r-l
  if (r - l <= 1) { // we assume vec that size equal to 1 is sorted
    return;
  }
  // divide
  size_t mid = (r - l) / 2 + l;
  MergeSortUtility(vec, buffer, l, mid);
  MergeSortUtility(vec, buffer, mid, r);

  // conquer
  size_t l_idx = l, r_idx = mid;
  //左边集合和右边集合已知已经排好序了，每次取左右两边最小的没有放入buffer的进行比较，较小的放入buffer，即每个集合从左边到右边查找
  size_t buffer_idx = l;
  while (l_idx < mid || r_idx < r) {
    if (r_idx >= r || (vec[l_idx] <= vec[r_idx] && l_idx < mid)) { // 如果右边集合已经全部被选入|| 左边小于右边
      buffer[buffer_idx] = vec[l_idx];
      l_idx++;
      buffer_idx++;
      // above statement can be simplified : buffer[buffer_idx++] = vec[l_idx++];
    } else {
      buffer[buffer_idx] = vec[r_idx];
      r_idx++;
      buffer_idx++;
      // above statement can be simplified : buffer[buffer_idx++] = vec[r_idx++];
    }
  }
  // move buffer value to vec
  for (size_t i = l; i < r; i++) {
    vec[i] = buffer[i];
  }

}

template<typename T>
void MergeSort(std::vector<T> &vec) {
  // recursive ways
  static std::vector<T> buffer_vec;
  buffer_vec.resize(vec.size());
  MergeSortUtility(vec, buffer_vec, 0, vec.size());
}

}// namespace mergesort

namespace insert_sort {

/**
 *  @brief: 将第一待排序序列第一个元素看做一个有序序列，把第二个元素到最后一个元素当成是未排序序列。
从头到尾依次扫描未排序序列，将扫描到的每个元素插入有序序列的适当位置。（如果待插入的元素与有序序列中的某个元素相等，则将待插入元素插入到相等元素的后面。）
 类似打扑克理牌的过程
 */
template<typename T>
void InsertSort(std::vector<T> &vec) {
  // index<=i表示已经排序好的数列
  for (size_t i = 0; i < vec.size(); i++) {
    // 从右向左检测是否插入 j ready to insert index
    for (size_t j = i; j > 0 && vec[j - 1] > vec[j]; j--) {
      std::swap(vec[j - 1], vec[j]);
    }
  }
}

} // namespace insert_sort

namespace bubble_sort {
/**
 * @brief  从左向右变量发现 左边比右边大的就进行交换，一次循环后最后一个元素肯定是最大，
 * 下次循环就可以从左到右以同种方式循环到倒数第二个，即倒数第一和第二都是排好序的了，以此类推
 */
template<typename T>
void BubbleSort(std::vector<T> &vec) {
  if (vec.empty()) {
    return;
  }

  size_t sorted_idx = vec.size(); // 这idx(含)右边，都是排好序的index，初始时index是无效的
  while (sorted_idx > 0) {
    for (size_t i = 1; i < sorted_idx; i++) {
      if (vec[i - 1] > vec[i]) {
        std::swap(vec[i - 1], vec[i]);
      }
    }
    sorted_idx--;
  }

}

} // namespace bubble sort


namespace select_sort {

/**
 * @brief  每次选择最小的放在最左边
 */
template<typename T>
void SelectSort(std::vector<T> &vec) {
  size_t sorted_idx = 0;
  size_t min_idx;
  while (sorted_idx < vec.size()) {
    min_idx = sorted_idx;
    for (size_t i = sorted_idx; i < vec.size(); i++) {
      if (vec[min_idx] > vec[i]) {
        min_idx = i;
      }
    }
    std::swap(vec[sorted_idx], vec[min_idx]);
    sorted_idx++;
  }
}

} // namespace select_sort

TEST(sort_algoritm, corect_test) {
  size_t test_times = 1000;
  size_t vector_size_factor = 5;
  long long sort_time_use = 0;

  for (size_t i = 0; i < test_times; i++) {
    auto &vec = RandomVecGenerator(vector_size_factor * i);
    auto start = std::chrono::high_resolution_clock::now();
    quick_sort::QuickSort(vec);
    auto end = std::chrono::high_resolution_clock::now();
    sort_time_use += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    EXPECT_TRUE(IsSorted(vec));
  }
  std::cout << "QuickSort Time :" << static_cast<double>(sort_time_use) / 1e6 << "ms" << std::endl;

  sort_time_use = 0;
  for (size_t i = 0; i < test_times; i++) {
    auto &vec = RandomVecGenerator(vector_size_factor * i);
    auto start = std::chrono::high_resolution_clock::now();
    merge_sort::MergeSort(vec);
    auto end = std::chrono::high_resolution_clock::now();
    sort_time_use += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    EXPECT_TRUE(IsSorted(vec));
  }
  std::cout << "MergeSort Time :" << static_cast<double>(sort_time_use) / 1e6 << "ms" << std::endl;

  sort_time_use = 0;
  for (size_t i = 0; i < test_times; i++) {
    auto &vec = RandomVecGenerator(vector_size_factor * i, {0, 10});
    auto start = std::chrono::high_resolution_clock::now();
    insert_sort::InsertSort(vec);
    auto end = std::chrono::high_resolution_clock::now();
    sort_time_use += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    EXPECT_TRUE(IsSorted(vec));
  }
  std::cout << "InsertSort Time :" << static_cast<double>(sort_time_use) / 1e6 << "ms" << std::endl;

  sort_time_use = 0;
  for (size_t i = 0; i < test_times; i++) {
    auto &vec = RandomVecGenerator(vector_size_factor * i, {0, 10});
    auto start = std::chrono::high_resolution_clock::now();
    bubble_sort::BubbleSort(vec);
    auto end = std::chrono::high_resolution_clock::now();
    sort_time_use += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    EXPECT_TRUE(IsSorted(vec));
  }
  std::cout << "BubbleSort Time :" << static_cast<double>(sort_time_use) / 1e6 << "ms" << std::endl;

  sort_time_use = 0;
  for (size_t i = 0; i < test_times; i++) {
    auto &vec = RandomVecGenerator(vector_size_factor * i, {0, 10});
    auto start = std::chrono::high_resolution_clock::now();
    select_sort::SelectSort(vec);
    auto end = std::chrono::high_resolution_clock::now();
    sort_time_use += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    EXPECT_TRUE(IsSorted(vec));
  }
  std::cout << "SelectSort Time :" << static_cast<double>(sort_time_use) / 1e6 << "ms" << std::endl;
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
