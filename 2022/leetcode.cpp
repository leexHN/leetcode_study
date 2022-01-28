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
#include <numeric>

using namespace std;

namespace greedy_algorithm {

TEST(easy, T455) {
  class Solution {
   public:
    int findContentChildren(vector<int> &g, vector<int> &s) {
      std::sort(g.begin(), g.end());
      std::sort(s.begin(), s.end());
      int num = 0;
      size_t g_idx = 0;
      size_t s_idx = 0;
      while (g_idx < g.size() && s_idx < s.size()) {
        if (g[g_idx] <= s[s_idx]) {
          num++;
          g_idx++;
        }
        s_idx++;
      }
      return num;
    }
  };

  Solution slo;
  std::vector<int> g, s;
  g = {1, 2, 3};
  s = {1, 1};
  EXPECT_EQ(1, slo.findContentChildren(g, s));
  g = {1, 2};
  s = {1, 2, 3};
  EXPECT_EQ(2, slo.findContentChildren(g, s));
}

TEST(hard, T135) { //  需要重做
  /* 错误解法!!!!
  class Solution {
   public:
    // 对每个孩子而言，自己就是局部最优，先初始化每个孩子的糖果为1,每个孩子左边最优+右边最优==全局最优
    // 即先让每个孩子左边最优，基于这个符合左边最优的优化右边最优
    int candy(vector<int>& ratings) {
      if (ratings.empty()) {
        return 0;
      }
      std::vector<int> candy_vec(ratings.size(), 1);
      int num_candy = ratings.size();
      // optimize left
      for (int i = 1; i < ratings.size(); ++i) {
        if (ratings[i] > ratings[i-1] && candy_vec[i] <= candy_vec[i-1]) { // local mismatch 自己糖果不匹配
          candy_vec[i] = candy_vec[i-1]+1;
        }
      }
      // optimize right  右边最优并且不破坏左边最优
      for (int i = 0; i < ratings.size() - 1; i++) {
        if (ratings[i] > ratings[i + 1] && candy_vec[i] <= candy_vec[i + 1] && i == 1) {
          candy_vec[i] = candy_vec[i + 1] + 1;
        } else { // my have left, must keep left optimization
          if (ratings[i] > ratings[i - 1] && candy_vec[i] <= candy_vec[i - 1]) { // keep left optimization
            candy_vec[i] = candy_vec[i - 1] + 1;
          }
          if (ratings[i] > ratings[i + 1] && candy_vec[i] <= candy_vec[i + 1]) { // let right optimization
            candy_vec[i] = candy_vec[i + 1] + 1; // this process may cause left not optimize
          }
        }
      }

      return std::accumulate(candy_vec.begin(), candy_vec.end(), 0);
    }
  };
   */

  class Solution {
   public:
    int candy(vector<int> &ratings) {
      std::vector<int> candy_vec(ratings.size(), 1);
      for (size_t i = 1; i < candy_vec.size(); i++) {
        if (ratings[i] > ratings[i - 1])  //其实这里还有一个条件，candy_vec[i] <= candy_vec[i-1],自动满足了
          candy_vec[i] = candy_vec[i - 1] + 1;
      }
      //左边都最优了，如何利用这一个条件，左边最优最后一个自动满足条件（左右最优），从倒数第二个开始，不改变左边最优，改变自己使得右边最优
      for (int i = candy_vec.size() - 2; i >= 0; i--) {
        if (ratings[i] > ratings[i + 1] && candy_vec[i] <= candy_vec[i + 1]) {
          candy_vec[i] = candy_vec[i + 1] + 1;
        }
      }
      return std::accumulate(candy_vec.begin(), candy_vec.end(), 0);
    }
  };
  Solution slo;
  std::vector<int> ratings;
  ratings = {1, 0, 2};
  EXPECT_EQ(5, slo.candy(ratings));
  ratings = {1, 2, 87, 87, 87, 2, 1};
  EXPECT_EQ(13, slo.candy(ratings));
}

TEST(easy, T605) {
  class Solution {
   public:
//    bool canPlaceFlowers(vector<int>& flowerbed, int n) {
//      int flower_inserted = 0;
//      int idx = 0;
//      if (n==0) {
//        return true;
//      }
//      if (flowerbed.size() <= 2) {
//        for(int i=0; i<flowerbed.size();i++) {
//          if (flowerbed[i]==1) {
//            return false;
//          }
//        }
//        if (n==1)
//          return true;
//        else
//          return false;
//      }
//      while(idx < flowerbed.size()) {
//        if(flowerbed[idx] == 1) {
//          idx+=2;
//          continue;
//        }
//        if (idx+1 < flowerbed.size() && idx-1 >=0){
//          if(flowerbed[idx-1] ==0 && flowerbed[idx+1] == 0) {
//            flowerbed[idx] = 1;
//            flower_inserted++;
//            idx+=2;
//          } else {
//            idx++;
//          }
//        } else if(idx == 0) {
//          if (flowerbed[idx+1] == 1){
//            idx+=2;
//          }else{
//            flower_inserted++;
//            flowerbed[idx] = 1;
//            idx+=2;
//          }
//        } else {
//          if (flowerbed[idx-1] == 1) {
//            idx+=2;
//          } else {
//            flower_inserted++;
//            flowerbed[idx] = 1;
//            idx+=2;
//          }
//        }
//
//        if (flower_inserted >= n) {
//          return true;
//        }
//      }
//      return false;
//    }
    bool canPlaceFlowers(vector<int> &flowerbed, int n) {
      if (n == 0)
        return true;
      int flower_inserted = 0;

      auto get = [&flowerbed](int idx) {
        if (idx < 0 || idx >= flowerbed.size()) {
          return 0;
        }
        return flowerbed[idx];
      };

      int idx = 0;
      while (idx < flowerbed.size()) {
        if (get(idx - 1) == 0 && get(idx) == 0 && get(idx + 1) == 0) {
          flowerbed[idx] = 1;
          flower_inserted++;
          idx += 2;
        } else {
          idx++;
        }
        if (flower_inserted >= n) {
          return true;
        }
      }

      return false;
    }
  };
}

TEST(mid, T452) { // 重做
  class Solution {
   public:
    int findMinArrowShots(vector<vector<int>> &points) {
      auto compare = [](const vector<int> &lhs, const vector<int> &rhs) {
        int l_back = lhs.back();
        int r_back = rhs.back();
        return l_back > r_back;
      };
      int n = 0;
      std::sort(points.begin(), points.end(), compare);
      std::vector<bool> is_selected(points.size());

      std::pair<int, int> range{points[0].front(), points[0].back()};
      n = 1;
      for (size_t i = 1; i < points.size(); i++) {
        if (points[i].back() >= range.first) {
          range = {std::max(range.first, points[i].front()), points[i].back()};
        } else {
          range = {points[i].front(), points[i].back()};
          n++;
        }
      }

      return n;
    }
  };

  Solution slo;
  std::vector<std::vector<int>> points = {{10, 16}, {2, 8}, {1, 6}, {7, 12}};
  EXPECT_EQ(2, slo.findMinArrowShots(points));
  points = {{1, 2}, {4, 5}, {1, 5}};
  EXPECT_EQ(2, slo.findMinArrowShots(points));
  points = {{0, 9}, {1, 8}, {7, 8}, {1, 6}, {9, 16}, {7, 13}, {7, 10}, {6, 11}, {6, 9}, {9, 13}};
  EXPECT_EQ(3, slo.findMinArrowShots(points));
}

TEST(mid, T763) {
  class Solution {
   public:
    vector<int> partitionLabels(const string &s) {
      std::vector<int> slip;
      std::unordered_map<char, std::pair<int, int>> char_num;// 纪录每个字母最小区间
      for (int i = 0; i < s.size(); i++) {
        auto it = char_num.find(s[i]);
        if (it == char_num.end()) {
          char_num[s[i]] = {i, i};
        } else {
          it->second.second = i;
        }
      }

      int index = 0;
      int num = 0;
      int range_end = 0;
      while (index < s.size()) {
        char c = s[index];
        range_end = std::max(char_num[c].second, range_end);
        num++;
        if (range_end == index) {
          slip.push_back(num);
          num = 0;
          range_end = index + 1;
        }
        index++;
      }
      return slip;
    }
  };

  Solution slo;
  std::vector<int> slip;
  slip = {1, 1, 1, 1};
  EXPECT_EQ(slo.partitionLabels("abcd"), slip);
  slip = {9, 7, 8};
  EXPECT_EQ(slo.partitionLabels("ababcbacadefegdehijhklij"), slip);
}

TEST(easy, T122) {
  class Solution {
   public:
    int maxProfit(vector<int> &prices) {
      int profit = 0;
      int buy_price = -1;
      for (int i = 1; i < prices.size(); i++) {
        if (prices[i - 1] < prices[i]) { // price up
          if (buy_price == -1) {
            buy_price = prices[i - 1];
          }
        } else if (prices[i - 1] > prices[i]) { //price down
          if (buy_price != -1) {
            profit += prices[i - 1] - buy_price;
            buy_price = -1;
          }
        }
      }
      if (buy_price != -1) {
        profit += prices.back() - buy_price;
      }
      return profit;
    }
  };
}

TEST(mid, T406) {
  class Solution {
   public:
    vector<vector<int>> reconstructQueue(vector<vector<int>> &people) {
//      std::sort(people.begin(), people.end(), [](const std::vector<int>& lhs, const std::vector<int>& rhs){
//        if (lhs[1] != rhs [1]) {
//          return lhs[1] < rhs[1];
//        }
//        return lhs[0] < rhs[0];
//      });
//      std::vector<bool> is_selected(people.size());
//      std::vector<std::vector<int>> sorted_people(people.size(), std::vector<int>(2));
//      int people_counter = 0;
//      for(size_t i=0; i<people.size(); i++) {
//
//      }

//      std::map<int, std::priority_queue<int>> priority_people; //[之前有多少人][身高]
//      for (size_t i=0; i< people.size(); i++) {
//        priority_people[people[i][1]].push(people[i][0]);
//      }
//      std::vector<std::vector<int>> sorted_people;
//      sorted_people.reserve(people.size());
//
//      for (size_t i=0; i<priority_people[0].size(); i++) {
//
//      }
//
//      while(sorted_people.size() < people.size()) {
//
//      }

      // 先排身高小并且符合条件的
      std::map<int, std::vector<int>> priority_people; //[身高][之前有多少人] 之前有多少人的拍成一组
      std::map<int, int> selected_map; //[身高][前面有多少人大于等于这个身高]
      for (int i = 0; i < people.size(); i++) {
        priority_people[people[i][0]].push_back(people[i][1]);
        selected_map[people[i][0]] = 0;
      }
      for (auto &p : priority_people) {
        std::sort(p.second.begin(), p.second.end(), [](int lhs, int rhs) { return lhs > rhs; });//把能够排最少的放在最后（最优先考虑）
      }

      std::vector<std::vector<int>> sorted_people;
      sorted_people.reserve(people.size());

      auto selected_map_func = [&selected_map](int high) {
        auto it = selected_map.find(high);
        auto set_it = selected_map.begin();
        while (true) {
          set_it->second++;
          if (it == set_it)
            break;
          set_it++;
        }
      };

      while (sorted_people.size() < people.size()) {
        auto it = priority_people.begin();
        while (true) {
          if (it->second.empty()) {
            it++;
            continue;
          }
          int high = it->first;
          int num = it->second.back();
          if (num <= selected_map[high]) {// 找到身高最小且前面人最小的了,并且符合条件
            it->second.pop_back();
            selected_map_func(high);
            sorted_people.push_back({high, num});
            break;
          }
          it++;
        }
      }

      return sorted_people;
    }
  };

  Solution slo;
  std::vector<std::vector<int>> people;
  people = {{7, 0}, {4, 4}, {7, 1}, {5, 0}, {6, 1}, {5, 2}};
  auto re = slo.reconstructQueue(people);
}

TEST(easy, T665) {
  class Solution {
   public:
    bool checkPossibility(vector<int> &nums) {
      if (nums.size() <= 2)
        return true;

      bool is_change_once = false;

      if (nums[0] > nums[1]) {
        nums[0] = nums[1];
        is_change_once = true;
      }
      for (int i = 1; i < nums.size() - 1; ++i) {
        if (nums[i] <= nums[i + 1])
          continue;
        if (is_change_once)
          return false;
        if (nums[i - 1] > nums[i + 1]) {
          nums[i + 1] = nums[i];
        }
        is_change_once = true;
      }
      return true;
    }
  };
}
} // namespace greedy algorithm


namespace two_pointer {

TEST(easy, T167) {
  class Solution {
   public:
    vector<int> twoSum(vector<int> &numbers, int target) {
      // numbers is increased
      int index_l = 0, index_r = numbers.size() - 1;;
      while (index_l <= index_r) {
        int l = numbers[index_l];
        int r = numbers[index_r];
        if (l + r == target)
          return {index_l + 1, index_r + 1};
        else if (l + r > target) {// need dec
          index_r--;
        } else { //need inc
          index_l++;
        }
      }
      return {index_l, index_r};
    }
  };
}

TEST(easy, T88) {
  class Solution {
   public:
    void merge(vector<int> &nums1, int m, vector<int> &nums2, int n) {
      int index1 = n + m - 1; // start from end
      n--;
      m--;
      while (m >= 0 && n >= 0) {
        int n1 = nums1[m], n2 = nums2[n];
        if (n1 >= n2) {
          nums1[index1] = n1;
          m--;
        } else {
          nums1[index1] = n2;
          n--;
        }
        index1--;
      }
      while (n >= 0) {
        nums1[index1] = nums2[n];
        n--;
        index1--;
      }
    }
  };
}

TEST(hard, T76) {
  class Solution {// 重做
   public:
    string minWindow(string s, string t) {
      // 首先统计t中的字符
      std::map<char, int> t_char_map;
      int t_remain = t.size(); // 统计使用t的字符
      for (auto c : t) {
        auto it = t_char_map.find(c);
        if (it == t_char_map.end()) {
          t_char_map[c] = 1;
        } else {
          it->second++;
        }
      }

      int l_idx = 0, r_idx = s.size(); //符合条件的s的idx
      int l_temp = 0;//当前搜索的s的index起点
      for (int r_temp = 0; r_temp < s.size(); ++r_temp) {
        auto it = t_char_map.find(s[r_temp]);
        if (it != t_char_map.end()) {
          it->second--;
          if (it->second >= 0) {
            t_remain--;
          }
        }

        while (t_remain == 0) { // 找到了index_range, 尝试缩小左边的l_temp
          // 移动l_temp
          auto l_it = t_char_map.find(s[l_temp]);
          if (l_it != t_char_map.end()) {
            l_it->second++;
            if (l_it->second > 0) {
              t_remain++;
            }
          }
          // 确定是否比原来的index range小，更新原来的index range
          int origin_range = r_idx - l_idx;
          int new_range = r_temp - l_temp;
          if (new_range < origin_range) {
            l_idx = l_temp;
            r_idx = r_temp;
          }
          l_temp++;
        }
      }
      if (r_idx == s.size()) {
        return {};
      } else {
        return {s.begin() + l_idx, s.begin() + r_idx + 1};
      }
    }
  };

  std::string s, t;
  s = "ADOBECODEBANC", t = "ABC";
  Solution slo;
  EXPECT_EQ(slo.minWindow(s, t), "BANC");
}

} // namespace two pointer

namespace binary_search {

TEST(easy, T68) { // 重做
  class Solution {
   public:
    int mySqrt(int x) {
      int l = 1, r = x;
      if (x == 0)
        return 0;
      while (l < r) {
        int mid = (l + r) / 2;
        int mid_2 = mid * mid;
        int mid_1_2 = (mid + 1) * (mid + 1);
        if (mid_2 <= x && mid_1_2 > x) {
          return mid;
        } else if (mid_1_2 <= x) {
          l = mid + 1;
        } else {
          r = mid - 1;
        }
      }
      return l;
    }
  };
  Solution slo;
  slo.mySqrt(9);
}

TEST(mid, T34) {
  class Solution {
   public:
    vector<int> searchRange(vector<int> &nums, int target) {
      if (nums.empty()) {
        return {-1, -1};
      } else if (nums.size() == 1) {
        if (nums[0] == target) {
          return {0, 0};
        } else {
          return {-1, -1};
        }
      }
      int l = IndexBelowEqual(nums, target);
      if (l == -1)
        return {-1, -1};
      int r = IndexUpper(nums, target);
      return {l, r};
    }
    // 查找第一个等于target的index
    int IndexBelowEqual(std::vector<int> &nums, int target) {
      int l = 0, r = nums.size() - 1;
      while (r - l > 1) {
        int mid = (l + r) / 2;
        if (nums[mid] >= target) {
          r = mid;
        } else {
          l = mid;
        }
      }
      if (nums[l] == target)
        return l;
      else if (nums[r] == target)
        return r;
      else
        return -1;
    }

    // 查找等于target的最后一个index
    int IndexUpper(std::vector<int> &nums, int target) {
      int l = 0, r = nums.size() - 1;
      while (r - l > 1) {
        int mid = (l + r) / 2;
        if (nums[mid] <= target) {
          l = mid;
        } else {
          r = mid;
        }
      }
      if (nums[r] == target) {
        return r;
      } else if (nums[l] == target)
        return l;
      else
        return -1;
    }
  };
}

} // namespace binary_search

namespace sort_algorithm {

TEST(mid, T215) {
  class Solution {
   public:
    int findKthLargest(vector<int> &nums, int k) {
      // 利用快速排序搜索到第nums.size()-k的值
      int idx_need = nums.size() - k;
      int l = 0, r = nums.size() - 1;
      while (true) {
        int now_p_idx = QuitSortUtility(l, r, nums);
        if (now_p_idx == idx_need) {
          return nums[now_p_idx];
        } else if (now_p_idx > idx_need) {
          r = now_p_idx - 1;
        } else {
          l = now_p_idx + 1;
        }
      }
    }

    /**
     * @return 返回pivot所在的index
     */
    int QuitSortUtility(int l, int r, std::vector<int> &nums) {
      int temp_l = l, temp_r = r;
      int pivot = nums[l];
      while (temp_l < temp_r) {
        // 先找右边(因为左边是空闲)
        while (temp_l < temp_r && nums[temp_r] >= pivot) {
          temp_r--;
        }
        // move to empty index
        nums[temp_l] = nums[temp_r];
        // now right empty
        while (temp_l < temp_r && nums[temp_l] <= pivot) {
          temp_l++;
        }
        nums[temp_r] = nums[temp_l];
      }
      nums[temp_l] = pivot;
      return temp_l;
    }
  };
}

TEST(mid, T347) {
  class Solution {
   public:
    vector<int> topKFrequent(vector<int> &nums, int k) {
      std::unordered_map<int, int> data;
      for (auto i : nums) {
        auto it = data.find(i);
        if (it == data.end()) {
          data[i] = 1;
        } else {
          it->second++;
        }
      }
      std::vector<std::pair<const int, int> const *> s_data;
      s_data.reserve(data.size());
      for (const auto &item : data) {
        s_data.push_back(&item);
      }
      std::sort(s_data.begin(), s_data.end(), [](const std::pair<const int, int> *lhs,
                                                 const std::pair<const int, int> *&rhs) {
        return lhs->second > rhs->second;
      });

      std::vector<int> ans;
      ans.reserve(k);
      for(int i=0; i<k;i++) {
        ans.push_back(s_data[i]->first);
      }
      return ans;
    }
  };
}

} // namespace sort algorithm


int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
