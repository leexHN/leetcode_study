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
bool operator == (const std::vector<T>& lhs, const std::vector<T>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i=0; i< lhs.size(); i++) {
    if (lhs[i] != rhs[i])
      return false;
  }
  return true;
}

template<typename T>
bool IsSorted(const std::vector<T> vec) {
  if(vec.size() <= 1) {
    return true;
  }
  for (size_t i=0; i< vec.size()-1; i++) {
    if (vec[i] > vec[i+1])
      return false;
  }
  return true;
}

std::vector<int>& RandomVecGenerator(size_t size, std::pair<int,int> range = {0, 100}) {
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> distrib(range.first, range.second);

  static std::vector<int> vec;
  vec.resize(size);

  for(auto& item: vec) {
    item = distrib(gen);
  }
  return vec;
}

namespace quick_sort {

template<typename T>
void QuickSortUtility(std::vector<T> &vec, size_t l, size_t r) { // 左开右闭
  //1. ready to sort vec [l, r), this vector size is r-l
  if (l + 1 >= r) { // same as r-l <= 1.So we assume vec size is 1 is sorted
    return;
  }
  T pivot = vec[l]; // 左边是空闲位置
  size_t l_bak = l;
  size_t r_bak = r; // using for sub vec [l pivot_idx) [pivot_idx r)
  r=r-1; //now r is r_idx(the most right index)
  while (l < r) { // when l==r exit l==r==pivot_idx
    // 首先左边空闲，需要在右边找到不合符条件的 r_value<pivot
    while (l<r && vec[r] >= pivot) { // 循环中都是符合条件的
      r--;
    }
    // 现在找到r不符合条件,把它移动到空闲位置l
    vec[l] = vec[r];
    // 现在r的位置空闲,左边找不符合条件的 l_value>pivot
    while (l<r && vec[l] <= pivot) {
      l++;
    }
    vec[r] = vec[l];
  }
  size_t pivot_idx = l; // l==r
  vec[pivot_idx] = pivot;
  // 第一次排序完成 满足pivot左边都小于pivot，pivot右边都大于pivot,注：这里pivot已经排好序了，pivot位置不需要改变了
  // 对子序列做相同方式排序
  QuickSortUtility(vec, l_bak, pivot_idx);
  QuickSortUtility(vec, pivot_idx+1, r_bak); // 这里必须加pivot+1,保证每次递归子vec的size必须比原始的vec要小，
  // 不然在一边子序列为空时，另一边子序列和原始的完全相同，导致死循环
  // 而且pivot位置已经排好序了，不需要变动
}

template<typename T>
void QuickSort(std::vector<T> &vec) {
  // recursive ways
  QuickSortUtility(vec, 0, vec.size());
}

}


TEST(sort_algoritm, corect_test) {
  size_t test_times = 1000;
  size_t vector_size = 5000;
  long long quick_sort_nano_time = 0;
  while(test_times > 0) {
    auto& vec = RandomVecGenerator(vector_size);
    auto start = std::chrono::high_resolution_clock::now();
    quick_sort::QuickSort(vec);
    auto end = std::chrono::high_resolution_clock::now();
    quick_sort_nano_time+= std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
    EXPECT_TRUE(IsSorted(vec));
    test_times--;
  }

  std::cout << "QuickSort Time :" << static_cast<double>(quick_sort_nano_time) / 1e6 << "ms" << std::endl;
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
