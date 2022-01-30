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
      for (int i = 0; i < k; i++) {
        ans.push_back(s_data[i]->first);
      }
      return ans;
    }
  };
}

} // namespace sort algorithm


namespace search {

TEST(mid, T695) {
  class Solution {
   public:
    void SearchUtility(const std::vector<vector<int>> &grid, std::vector<std::vector<bool>> &is_search,
                       int &area, int row, int col,
                       const std::vector<std::vector<int>> &search_direction) {
      int max_row = grid.size();
      int max_col = grid.front().size();
      auto is_valid_grid = [&](int row, int col) {
        if (row < 0 || row >= max_row) {
          return false;
        } else if (col < 0 || col >= max_col) {
          return false;
        }
        if (is_search[row][col]) {
          return false;
        }
        if (grid[row][col] == 0) {
          is_search[row][col] = true;
          return false;
        }
        return true;
      };

      if (!is_valid_grid(row, col)) {
        return;
      }

      is_search[row][col] = true;
      area++;

      for (int i = 0; i < search_direction.size(); i++) {
        int current_row = row + search_direction[i][0];
        int current_col = col + search_direction[i][1];
        SearchUtility(grid, is_search, area, current_row, current_col, search_direction);
      }
    }

    int maxAreaOfIsland(const vector<vector<int>> &grid) {
      const static std::vector<vector<int>> search_direction{{-1, 0}, {1, 0}, {0, 1}, {0, -1}}; // 上下右左
      std::vector<std::vector<bool>> is_search(grid.size(), std::vector<bool>(grid.front().size()));
      int max_area = 0;
      for (int row = 0; row < grid.size(); row++) {
        for (int col = 0; col < grid.front().size(); col++) {
          if (is_search[row][col])
            continue;
          int area = 0;
          SearchUtility(grid, is_search, area, row, col, search_direction);
          max_area = std::max(max_area, area);
        }
      }
      return max_area;
    }
  };

  Solution slo;
  EXPECT_EQ(slo.maxAreaOfIsland({{1, 1}, {1, 0}}), 3);
}

TEST(mid, T417) {// 重做！！！！
  class Solution {
   public:
    typedef enum {
      NONE,
      LEFT_UP,
      RIGHT_DOWN,
      BOTH,
    } POINT_PROPERTY;

    POINT_PROPERTY SearchUtility(const vector<vector<int>> &heights,
                                 std::vector<std::vector<POINT_PROPERTY>> &point_property,
                                 int row,
                                 int col) {
      int max_row = heights.size();
      int max_col = heights.front().size();
      if (row < 0 || col < 0)
        return LEFT_UP;
      if (row >= max_row || col >= max_col) {
        return RIGHT_DOWN;
      }
      if (point_property[row][col] != NONE)
        return point_property[row][col];

      static const std::vector<std::pair<int, int>> search_dir = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
      POINT_PROPERTY cur_p = NONE;

      for (auto dir : search_dir) {
        int cur_row = row + dir.first;
        int cur_col = col + dir.second;
        POINT_PROPERTY next_p = SearchUtility(heights, point_property, cur_row, cur_col);
        if (cur_p == NONE) {
          cur_p = next_p;
        } else if (next_p == BOTH) {
          cur_p = BOTH;
        } else if (next_p != cur_p) {
          cur_p = BOTH;
        }
      }
      point_property[row][col] = cur_p;
      return cur_p;
    }

    vector<vector<int>> pacificAtlantic(const vector<vector<int>> &heights) {
      std::vector<std::vector<POINT_PROPERTY>> point_property(heights.size(),
                                                              std::vector<POINT_PROPERTY>(heights.front().size(),
                                                                                          NONE));
      std::vector<std::vector<int>> ans;
      for (int row = 0; row < heights.size(); row++) {
        for (int col = 0; col < heights.front().size(); col++) {
          auto cur_p = SearchUtility(heights, point_property, row, col);
          if (cur_p == BOTH)
            ans.push_back({row, col});
        }
      }
      return ans;
    }
  };

  Solution slo;
  slo.pacificAtlantic({{1, 2, 2, 3, 5}, {3, 2, 3, 4, 4}, {2, 4, 5, 3, 1}, {6, 7, 1, 4, 5}, {5, 1, 1, 2, 4}});
}

TEST(mid, T46) {
  class Solution {
   public:
    void SearchUtility(const std::vector<int> &nums, std::vector<bool> &is_visited,
                       std::vector<std::vector<int>> &ans, std::vector<int> &temp, int idx) {
      is_visited[idx] = true;
      temp.push_back(nums[idx]);
      if (temp.size() == nums.size()) {
        ans.push_back(temp);
        is_visited[idx] = false;
        temp.pop_back();
        return;
      }

      for (int i = 0; i < nums.size(); i++) {
        if (!is_visited[i]) {
          SearchUtility(nums, is_visited, ans, temp, i);
        }
      }
      temp.pop_back();
      is_visited[idx] = false;
    }

    vector<vector<int>> permute(const vector<int> &nums) {
      if (nums.size() <= 1) {
        return {nums};
      }
      std::vector<std::vector<int>> ans;
      ans.reserve(nums.size() * (nums.size() - 1));
      std::vector<int> temp;
      temp.reserve(nums.size());
      std::vector<bool> is_visited(nums.size(), false);
      for (int i = 0; i < nums.size(); i++) {
        SearchUtility(nums, is_visited, ans, temp, i);
      }
      return ans;
    }
  };

  Solution slo;
  EXPECT_EQ(slo.permute({1, 2, 3}).size(), 6);
}

TEST(mid, T77) {
  class Solution {
   public:
    void Dfs(std::vector<std::vector<int>> &ans,
             std::vector<bool> &is_visited,
             std::vector<int> &temp,
             int n,
             int k,
             int idx) {
      temp.push_back(idx+1);
      if (temp.size() == k) {
        ans.push_back(temp);
        temp.pop_back();
        return;
      }

      is_visited[idx] = true;
      for (int i = idx+1; i <= n; i++) {
        if (is_visited[i-1])
          continue;
        Dfs(ans, is_visited, temp, n, k, i-1);
      }

      is_visited[idx] = false;
      temp.pop_back();
    }

    vector<vector<int>> combine(int n, int k) {
      std::vector<std::vector<int>> ans;
      std::vector<int> temp;
      temp.reserve(k);
      std::vector<bool> is_visited(n, false);
      for (int i=1; i<=n; i++) {
        Dfs(ans, is_visited, temp, n, k, i-1);
      }
      return ans;
    }

  };

  Solution slo;
  slo.combine(4,2);
}

TEST(mid, T79) {
  class Solution {
   public:
    void Dfs(const vector<vector<char>>& board,const string& word,
             std::vector<std::vector<bool>>& is_visited,
             std::string& cur_string,
             bool &is_found, int row, int col) {
      if (is_found)
        return;
      int max_row = board.size();
      int max_col = board.front().size();
      if (row < 0 || col<0 || row >= max_row || col >= max_col)
        return;

      if (is_visited[row][col])
        return;

      cur_string.push_back(board[row][col]);
      is_visited[row][col] = true;

      if (cur_string.size() > word.size()) {
        cur_string.pop_back();
        is_visited[row][col] = false;
        return;
      }
      if (word.substr(0, cur_string.size()) != cur_string) {
        cur_string.pop_back();
        is_visited[row][col] = false;
        return;
      }
      if(cur_string.size() == word.size()) {
        is_found = true;
        cur_string.pop_back();
        is_visited[row][col] = false;
        return;
      }

      is_visited[row][col] = true;
      static const std::vector<std::pair<int,int>> dir = {{-1,0}, {1,0}, {0,1}, {0,-1}};
      for(auto& d: dir) {
        int cur_row = row + d.first;
        int cur_col = col + d.second;
        Dfs(board,word,is_visited,cur_string,is_found, cur_row, cur_col);
      }
      cur_string.pop_back();
      is_visited[row][col] = false;
    }

    bool exist(const vector<vector<char>>& board,const string& word) {
      std::vector<std::vector<bool>> is_visited(board.size(), std::vector<bool>(board.front().size()));
      std::string cur_string;
      bool is_found = false;
      for(int row = 0; row < board.size(); row++) {
        for(int col = 0; col < board.front().size(); col++) {
          Dfs(board,word,is_visited,cur_string, is_found, row, col);
          if (is_found)
            return is_found;
        }
      }
      return false;
    }
  };

  Solution slo;
  slo.exist({{'A','B','C','E'},{'S','F','C','S'},{'A','D','E','E'}}, "ABCCED");
}
} // namespace of search




int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
