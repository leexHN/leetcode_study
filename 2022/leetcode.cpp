//
// Created by lix on 1/5/22.
//
#include <iostream>
#include <utility>
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
#include <queue>
#include <stack>

using namespace std;

struct ListNode {
  int val;
  ListNode *next;
  ListNode() : val(0), next(nullptr) {}
  ListNode(int x) : val(x), next(nullptr) {}
  ListNode(int x, ListNode *next) : val(x), next(next) {}
};

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
      temp.push_back(idx + 1);
      if (temp.size() == k) {
        ans.push_back(temp);
        temp.pop_back();
        return;
      }

      is_visited[idx] = true;
      for (int i = idx + 1; i <= n; i++) {
        if (is_visited[i - 1])
          continue;
        Dfs(ans, is_visited, temp, n, k, i - 1);
      }

      is_visited[idx] = false;
      temp.pop_back();
    }

    vector<vector<int>> combine(int n, int k) {
      std::vector<std::vector<int>> ans;
      std::vector<int> temp;
      temp.reserve(k);
      std::vector<bool> is_visited(n, false);
      for (int i = 1; i <= n; i++) {
        Dfs(ans, is_visited, temp, n, k, i - 1);
      }
      return ans;
    }

  };

  Solution slo;
  slo.combine(4, 2);
}

TEST(mid, T79) {
  class Solution {
   public:
    void Dfs(const vector<vector<char>> &board, const string &word,
             std::vector<std::vector<bool>> &is_visited,
             std::string &cur_string,
             bool &is_found, int row, int col) {
      if (is_found)
        return;
      int max_row = board.size();
      int max_col = board.front().size();
      if (row < 0 || col < 0 || row >= max_row || col >= max_col)
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
      if (cur_string.size() == word.size()) {
        is_found = true;
        cur_string.pop_back();
        is_visited[row][col] = false;
        return;
      }

      is_visited[row][col] = true;
      static const std::vector<std::pair<int, int>> dir = {{-1, 0}, {1, 0}, {0, 1}, {0, -1}};
      for (auto &d : dir) {
        int cur_row = row + d.first;
        int cur_col = col + d.second;
        Dfs(board, word, is_visited, cur_string, is_found, cur_row, cur_col);
      }
      cur_string.pop_back();
      is_visited[row][col] = false;
    }

    bool exist(const vector<vector<char>> &board, const string &word) {
      std::vector<std::vector<bool>> is_visited(board.size(), std::vector<bool>(board.front().size()));
      std::string cur_string;
      bool is_found = false;
      for (int row = 0; row < board.size(); row++) {
        for (int col = 0; col < board.front().size(); col++) {
          Dfs(board, word, is_visited, cur_string, is_found, row, col);
          if (is_found)
            return is_found;
        }
      }
      return false;
    }
  };

  Solution slo;
  slo.exist({{'A', 'B', 'C', 'E'}, {'S', 'F', 'C', 'S'}, {'A', 'D', 'E', 'E'}}, "ABCCED");
}

TEST(hard, T51) {
  class Solution {
   public:
    bool IsValid(const std::vector<std::string> &queen, int row, int col) {
      if (queen[row][col] != 'Q') {
        return true;
      }
      int n = queen.size();
      // row check
      for (int i = 0; i < n; i++) {
        if (i != row && queen[i][col] == 'Q')
          return false;
      }
      // col check
      for (int i = 0; i < n; i++) {
        if (i != col && queen[row][i] == 'Q')
          return false;
      }
      // \dir check
      int cur_row = row - 1;
      int cur_col = col - 1;
      while (cur_row >= 0 && cur_col >= 0) {
        if (queen[cur_row][cur_col] == 'Q')
          return false;
        cur_row--;
        cur_col--;
      }
      cur_row = row + 1;
      cur_col = col + 1;
      while (cur_row < n && cur_col < n) {
        if (queen[cur_row][cur_col] == 'Q')
          return false;
        cur_row++;
        cur_col++;
      }
      // /dir check
      cur_row = row - 1;
      cur_col = col + 1;
      while (cur_row < n && cur_col < n && cur_row >= 0 && cur_col >= 0) {
        if (queen[cur_row][cur_col] == 'Q')
          return false;
        cur_row--;
        cur_col++;
      }
      cur_row = row + 1;
      cur_col = col - 1;
      while (cur_row < n && cur_col < n && cur_row >= 0 && cur_col >= 0) {
        if (queen[cur_row][cur_col] == 'Q')
          return false;
        cur_row++;
        cur_col--;
      }
      return true;
    }

    bool IsValid(const std::vector<std::string> &queen) {
      for (int i = 0; i < queen.size(); i++) {
        for (int j = 0; j < queen.front().size(); j++) {
          if (!IsValid(queen, i, j))
            return false;
        }
      }
      return true;
    }

    void Dfs(std::vector<std::string> &queen,
             int &level) {

      if (level == queen.size()) {
//        if (IsValid(queen)) {
        queens_ptr->push_back(queen);
        return;
//        } else {
//          return;
//        }
      }

      int row = level;

      for (int col = 0; col < queen.size(); col++) {
//        if (col_is_visited[col])
//          continue;
        level++;
        queen[row][col] = 'Q';

        if (!IsValid(queen)) {
          level--;
          queen[row][col] = '.';
          continue;
        }

        Dfs(queen, level);

        level--;
        queen[row][col] = '.';
      }

    }

    vector<vector<string>> solveNQueens(int n) {
      std::vector<std::vector<string>> queens;
      std::vector<std::string> queen(n, std::string(n, '.'));
      queens_ptr = &queens;

      for (int col = 0; col < n; col++) {
        int level = 1;
        int row = 0;
        queen[row][col] = 'Q';

        Dfs(queen, level);

        queen[row][col] = '.';
      }

      return queens;
    }
    std::vector<std::vector<string>> *queens_ptr;
  };
  Solution slo;
  auto ans = slo.solveNQueens(4);
}

TEST(mid, T934) {
  class Solution {
   public:
    struct QueueData {
      QueueData(int _row, int _col, int _dis) : row(_row), col(_col), dis(_dis) {}
      int row;
      int col;
      int dis;
      bool operator<(const QueueData &rhs) const {
        return this->dis > rhs.dis;
      }
    };
    // 相当于求取两个岛屿之间的最短距离
    /*
    // 从岛屿1出发到岛屿2，贪心算法搜索
    void Island0Set(const std::vector<std::vector<int>> &grid,
                    std::vector<std::vector<bool>> &is_land0,
                    std::priority_queue<QueueData> &queue_data) {
      // 先找到一个岛屿
      int row_temp = -1;
      int col_temp = -1;
      for (int row = 0; row < grid.size(); row++) {
        for (int col = 0; col < grid[row].size(); col++) {
          if (grid[row][col] == 1) {
            row_temp = row;
            col_temp = col;
            break;
          }
        }
        if (row_temp >= 0)
          break;
      }
      int max_row = grid.size();
      int max_col = grid.front().size();
      auto is_movable = [&](int row, int col) {
        if (row < 0 || col < 0 || row >= max_row || col >= max_col || grid[row][col] == 0 || is_land0[row][col])
          return false;
        return true;
      };

      // using dfs
      std::stack<std::pair<int,int>> search_stack;
      search_stack.push({row_temp,col_temp});

      while(!search_stack.empty()) {
        int cur_row = search_stack.top().first;
        int cur_col = search_stack.top().second;
        bool is_push = false;

        for(const auto& dir: *search_dir_ptr) {
          row_temp =cur_row+dir.first;
          col_temp =cur_col+dir.second;
          if (is_movable(row_temp, col_temp)) {
            search_stack.push({row_temp, col_temp});
            is_push = true;
          }
        }

        if(is_push)
          continue;

        is_land0[cur_row][cur_col] = true;
        search_stack.pop();
      }

    }
     */
    // 先随机找一个有陆地的地方
    std::pair<int, int> GetLand(const std::vector<std::vector<int>> &grid) {
      // 先找到一个岛屿
      int row_temp = -1;
      int col_temp = -1;
      for (int row = 0; row < grid.size(); row++) {
        for (int col = 0; col < grid[row].size(); col++) {
          if (grid[row][col] == 1) {
            row_temp = row;
            col_temp = col;
            break;
          }
        }
        if (row_temp >= 0)
          break;
      }
      return {row_temp, col_temp};
    }

    int shortestBridge(const vector<vector<int>> &grid) {
      const static std::vector<std::pair<int, int>> search_dir = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
//      std::vector<std::vector<bool>> island0(grid.size(), std::vector<bool>(grid.front().size()));
      std::vector<std::vector<bool>> is_visited(grid.size(), std::vector<bool>(grid.front().size()));
//      search_dir_ptr = &search_dir;

      std::priority_queue<QueueData> queue_data;
      auto land_temp = GetLand(grid);
      queue_data.emplace(land_temp.first, land_temp.second, 0);
      is_visited[land_temp.first][land_temp.second] = true;

      auto point_valid = [&grid](int row, int col) {
        int max_row = grid.size();
        int max_col = grid.front().size();
        if (row < 0 || col < 0 || row >= max_row || col >= max_col)
          return false;
        return true;
      };

      while (!queue_data.empty()) {
        int cur_row = queue_data.top().row;
        int cur_col = queue_data.top().col;
        int cur_dis = queue_data.top().dis;

        queue_data.pop();

        for (const auto dir : search_dir) {
          int row_temp = cur_row + dir.first;
          int col_temp = cur_col + dir.second;
          if (point_valid(row_temp, col_temp) && !is_visited[row_temp][col_temp]) {
            if (cur_dis == 0 && grid[row_temp][col_temp] == 1) { // in land0
              is_visited[row_temp][col_temp] = true;
              queue_data.emplace(row_temp, col_temp, 0);
            } else {
              if (grid[row_temp][col_temp] == 1) {
                return cur_dis;
              } else {
                is_visited[row_temp][col_temp] = true;
                queue_data.emplace(row_temp, col_temp, cur_dis + 1);
              }
            }
          }
        }

      }
      return 0;
    }

//    const std::vector<std::pair<int, int>> *search_dir_ptr;
  };

  Solution slo;
  EXPECT_EQ(slo.shortestBridge({{0, 1, 0}, {0, 0, 0}, {0, 0, 1}}), 2);
}

TEST(hard, T126) {
  // 重做， 目前还没有理解求法
  /*
  class Solution {
   public:

    struct LadderData {
      explicit LadderData(int idx, const std::string& str,const std::vector<bool>& _is_used,const std::vector<std::string>& _path) {
        is_used = _is_used;
        path = _path;
        path.push_back(str);
        if (idx >= 0)
          is_used[idx] = true;
      }

      std::vector<bool> is_used;
      std::vector<std::string> path;
      bool is_valid;
    };

    bool IsValid(int idx1, int idx2, const vector<string> &wordList) {
      const auto &word1 = wordList[idx1];
      const auto &word2 = wordList[idx2];
      return IsValid(word1, word2);
    }

    bool IsValid(const std::string &word1, const std::string &word2) {
      int diff_counter = 0;
      for (int i = 0; i < word1.size(); i++) {
        if (word1[i] != word2[i])
          diff_counter++;
        if (diff_counter > 1)
          return false;
      }
      return diff_counter == 1;
    }

    vector<vector<string>> findLadders(string beginWord, string endWord,const vector<string> &wordList) {
      // 1.find end word
      int end_word_idx = -1;
      for (int i = 0; i < wordList.size(); i++) {
        const auto &word = wordList[i];
        if (word == endWord) {
          end_word_idx = i;
          break;
        }
      }
      if (end_word_idx == -1)
        return {};

      // 2.init bfs data
      std::queue<LadderData> data;
      std::vector<std::vector<std::string>> ans;
      bool is_found = false;
      int level = 0;
      data.emplace(-1, beginWord, std::vector<bool>(wordList.size()), std::vector<std::string>());

      while (level < wordList.size() && !is_found){
        int n_data = data.size();
        std::cout << "Current Search Level " << level << std::endl;
        std::cout << "N_Data = " << n_data << std::endl;
        std::cout << "======================\n";
        while(n_data > 0) {
          const auto & cur_word = data.front().path.back();
          for (int i=0; i< data.front().is_used.size(); i++) {
            if(data.front().is_used[i])
              continue;
            if (!IsValid(cur_word, wordList[i]))
              continue;
            data.emplace(i, wordList[i], data.front().is_used, data.front().path);
            if (i == end_word_idx)
              is_found = true;
          }
          data.pop();
          n_data--;
        }

        level++;
      }

      while(!data.empty() && is_found) {
        if (data.front().path.back() == endWord) {
          ans.emplace_back(std::move(data.front().path));
        }
        data.pop();
      }

      return ans;
    }
  };
  */

}

} // namespace of search

namespace dynamic_program {

TEST(easy, T70) {
  class Solution {
   public:
    int climbStairs(int n) {
      // 因为求取n的解的数量，可以转化为n-1的解加上n-2的解(因为n-1再加上1就可以到达n, 同理n-2加上2)
      // 即状态转移方程可以得到 (n) = (n-1) + (n-2) 括号表示该序列的解
      // 选取边界条件n至少需要大于2
      // n=1 解为1
      // n=2 解为2 1+1 或 2
      if (n <= 2)
        return n;

      // 原始
      std::vector<long> dp(n);
      dp[0] = 1;
      dp[1] = 2;
      for (int i = 2; i < dp.size(); i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
      }
//      return dp.back();

      // 简化
      unsigned int dp_pre_2 = 1; // n-2
      unsigned int dp_pre_1 = 2; // n-1
      unsigned long cur = 0;
      for (int i = 3; i <= n; i++) {
        cur = dp_pre_1 + dp_pre_2;
        dp_pre_2 = dp_pre_1;
        dp_pre_1 = cur;
      }
      return cur;
    }

  };

  Solution slo;
  slo.climbStairs(50);
}

TEST(mid, T198) {
  class Solution {
   public:
    int rob(vector<int> &nums) {
      // 动态规划
      // 假设(k) 表示第k个被选中的情况下的最优解
      // (k) = max[(k-2) + v(k), (k-3) + v(k)] // v(k)表示第k个的价值
      if (nums.size() <= 2) {
        std::sort(nums.begin(), nums.end());
        return nums.back();
      }
      if (nums.size() == 3) {
        return std::max(nums[1], nums[0] + nums[2]);
      }
      std::vector<int> dp(nums.size());
      dp[0] = nums[0];
      dp[1] = nums[1];
      dp[2] = nums[0] + nums[2];

      for (int i = 3; i < dp.size(); i++) {
        dp[i] = std::max(dp[i - 2] + nums[i], dp[i - 3] + nums[i]);
      }
      return std::max(dp[dp.size() - 1], dp[dp.size() - 2]);
    }
  };
}

TEST(mid, T413) {
  // 重做
  class Solution {
   public:
    int numberOfArithmeticSlices(vector<int> &nums) {
      if (nums.size() < 3)
        return 0;
      std::vector<int> dp(nums.size());
      for (int i = 2; i < nums.size(); i++) {
        int counter = 0;
        int delta = nums[i] - nums[i - 1];
        for (int j = i; j > 0; j--) {
          if (nums[j] - nums[j - 1] == delta) {
            counter++;
          } else {
            break;
          }
        }
        dp[i] = counter - 1;
      }
      return std::accumulate(dp.begin(), dp.end(), 0);
    }
  };
}

TEST(mid, T64) {
  class Solution {
   public:
    int minPathSum(const vector<vector<int>> &grid) {
      // 只能向下和向右走， 对于一个点只能从[row-1][col] 或者 [row][col-1]变化而来
      std::vector<std::vector<int>> path(grid.size(), std::vector<int>(grid.front().size()));
      path[0][0] = grid[0][0];
      for (int col = 1; col < grid.front().size(); col++) {
        path[0][col] = path[0][col - 1] + grid[0][col];
      }
      for (int row = 1; row < grid.size(); row++) {
        path[row][0] = path[row - 1][0] + grid[row][0];
      }

      for (int row = 1; row < grid.size(); row++) {
        for (int col = 1; col < grid.front().size(); col++) {
          path[row][col] = std::min(path[row - 1][col], path[row][col - 1]) + grid[row][col];
        }
      }
      return path.back().back();
    }
  };

  Solution slo;
  slo.minPathSum({{1, 3, 1}, {1, 5, 1}, {4, 2, 1}});
}

TEST(mid, T542) {
  class Solution {
   public:
    vector<vector<int>> updateMatrix(const vector<vector<int>> &mat) {
      const int invalid = std::numeric_limits<int>::max();
      const std::vector<std::pair<int, int>> dirs = {{-1, 0}, {1, 0}, {0, 1}, {0, -1}};
      std::vector<std::vector<int>> res(mat.size(), std::vector<int>(mat.front().size(), invalid));
      // 找到所有的0 设置为0, 推入队列
      std::queue<std::pair<int, int>> bfs_queue;
      for (int r = 0; r < mat.size(); r++) {
        for (int j = 0; j < mat.size(); j++) {
          if (mat[r][j] == 0) {
            res[r][j] = 0;
            bfs_queue.emplace(r, j);
          }
        }
      }
      auto is_invalid = [&](const std::pair<int, int> idx) {
        if (idx.first < 0 || idx.first >= mat.size() || idx.second < 0 || idx.second >= mat.front().size()) {
          return false;
        }
        return res[idx.first][idx.second] == invalid;
      };
      while (!bfs_queue.empty()) {
        int n = bfs_queue.size();
        while (n > 0) {
          const auto &idx = bfs_queue.front();
          int dis = res[idx.first][idx.second];
          for (auto &dir : dirs) {
            std::pair<int, int> new_idx = {idx.first - dir.first, idx.second - dir.second};
            if (is_invalid(new_idx)) {
              bfs_queue.push(new_idx);
              res[new_idx.first][new_idx.second] = dis + 1;
            }
          }
          n--;
          bfs_queue.pop();
        }
      }
      return res;
    }
  };

  Solution slo;
  slo.updateMatrix({{0}, {0}, {0}, {0}, {0}});
}

TEST(mid, T221) {
  class Solution {
   public:
    int maximalSquare(vector<vector<char>> &matrix) {
      std::vector<std::vector<int>> dp(matrix.size(), std::vector<int>(matrix.front().size()));
      auto dp_value = [&](int r, int c) {
        if (r < 0 || c < 0) {
          return 0;
        }
        return dp[r][c];
      };
      int max_l = 0;
      for (int i = 0; i < matrix.size(); i++) {
        for (int j = 0; j < matrix.front().size(); j++) {
          if (matrix[i][j] == '1') {
            dp[i][j] = std::min({dp_value(i - 1, j - 1), dp_value(i, j - 1), dp_value(i - 1, j)}) + 1;
            max_l = std::max(dp[i][j], max_l);
          }
        }
      }
      return max_l * max_l;
    }
  };

}

TEST(mid, T279) {
  class Solution {
   public:
    int numSquares(int n) {
      std::vector<int> dp(n + 1);
      dp[0] = 0;
      dp[1] = 1;
      for (int i = 2; i <= n; i++) {
        int counter = std::numeric_limits<int>::max() - 1;
        for (int j = 1; i - j * j >= 0; j++) {
          counter = std::min(counter, dp[i - j * j] + 1);
        }
        dp[i] = counter;
      }
      return dp.back();
    }
  };
  Solution slo;
  slo.numSquares(12);
}

TEST(mid, T91) {
  // 重做 边界条件
  class Solution {
   public:
    int numDecodings(string s) {
      if (s.front() == '0')
        return 0;
      if (s.size() == 1)
        return 1;
      std::vector<int> dp(s.size());
      dp[0] = 1;
      if (s[1] == '0') {
        dp[1] = stoi(s.substr(0, 2)) <= 26;
      } else {
        dp[1] = stoi(s.substr(0, 2)) <= 26 ? 2 : 1;
      }
      for (int i = 2; i < s.size(); i++) {
        int dp_2 = i - 2 >= 0 ? dp[i - 2] : 0;
        if (s[i - 1] != '0') {
          dp[i] = dp_2;
          int two_num = std::stoi(s.substr(i - 1, 2));
          if (s[i] == '0') {
            if (two_num > 26)
              dp[i] = 0;
          } else {
            if (two_num <= 26)
              dp[i]++;
          }
        } else {
          if (s[i] == '0')
            return 0;
          else
            dp[i] = dp[i - 1];
        }
      }
      return dp.back();
    }
  };
  Solution slo;
  slo.numDecodings("226");
}

TEST(mid, T139) {
  class SolutionDfs {
   public:
    bool dfs(const std::string &s, const std::map<int, std::set<std::string *>> &len_word_map,
             int s_idx,
             std::vector<int> &memo) { // 减枝，避免重复计算，记录已经计算的结果
      if (s_idx >= s.size())
        return true;
      if (memo[s_idx] != -1)
        return memo[s_idx];

      for (auto p = len_word_map.rbegin(); p != len_word_map.rend(); p++) {
        int word_len = p->first;
        if (s_idx + word_len > s.size()) {
          continue;
        }
        const auto &word_to_compare = s.substr(s_idx, word_len);

        for (auto word : p->second) {
          if (*word != word_to_compare)
            continue;
          if (dfs(s, len_word_map, s_idx + word_len, memo)) {
            memo[s_idx + word_len] = true;
            return true;
          }
        }
      }
      memo[s_idx] = false;
      return false;
    }

    bool wordBreak(string s, vector<string> &wordDict) {
      std::map<int, std::set<std::string *>> len_word_map;
      std::vector<int> memo(s.size(), -1);
      for (auto &w : wordDict) {
        len_word_map[w.size()].insert(&w);
      }
      return dfs(s, len_word_map, 0, memo);
    }
  };

  class Solution {
   public:
    bool wordBreak(string s, vector<string> &wordDict) {
      std::map<int, std::set<std::string *>> len_word_map;
      for (auto &w : wordDict) {
        len_word_map[w.size()].insert(&w);
      }
      std::vector<bool> dp(s.size() + 1); // dp[i] ,s的前i(不含)个字符是否能够在dir中构成
      dp[0] = true;
      for (int i = 1; i <= s.size(); i++) {
        for (const auto &pair : len_word_map) {
          int word_len = pair.first;
          int idx = i - word_len;
          if (idx < 0)
            break;
          if (!dp[idx]) { // 当前子长度的字符在字典中没有找到也就没必要继续找了,换一个长度
            continue;
          }
          const auto &s_sub = s.substr(idx, word_len);

          for (const auto &word : pair.second) {
            if (*word == s_sub) {
              dp[i] = true;
            }
          }

          if (dp[i])
            break;// 已经找到就没必要继续找了，往下求下一个i的dp
        }
      }
      return dp.back();
    }
  };

  Solution slo;
  vector<string> wordDict{"leet", "code"};
  slo.wordBreak("leetcode", wordDict);
}

TEST(mid, T300) {
  class Solution {
   public:
    int lengthOfLIS(vector<int> &nums) {
      std::vector<int> dp(nums.size());
      dp[0] = 1;
      int max_l = 0;
      for (int i = 1; i < dp.size(); i++) {
        int l = 1;
        for (int j = i - 1; j >= 0; j--) {
          if (nums[i] >= nums[j]) {
            l = std::max(l, dp[j] + 1);
          }
          dp[i] = l;
          max_l = std::max(l, max_l);
        }
      }
      return max_l;
    }
  };
}

namespace bag_problem {
TEST(mid, T416) {//重做
  class Solution {
   public:

    bool canPartition(vector<int> &nums) {
      int sum = std::accumulate(nums.begin(), nums.end(), 0);
      if (sum % 2 != 0)
        return false;
      sum = sum / 2;
      std::vector<std::vector<bool>> dp(nums.size() + 1, std::vector<bool>(sum + 1));
      for (int i = 0; i <= nums.size(); i++) {
        dp[i][0] = true; // 和为0任何i都能达到
      }
      for (int i = 1; i <= nums.size(); i++) {
        for (int j = 0; j <= sum; j++) { // 和为j至少也需要i个数(因为nums[i]>=1)
          if (j - nums[i - 1] >= 0) {
            dp[i][j] = dp[i - 1][j] || dp[i - 1][j - nums[i - 1]];
          } else {
            dp[i][j] = dp[i - 1][j];
          }
        }
      }
      return dp[nums.size()][sum];
    }
  };
  class SolutionSpace {
   public:
    bool canPartition(vector<int> &nums) {
      int sum = std::accumulate(nums.begin(), nums.end(), 0);
      if (sum % 2 != 0)
        return false;
      sum /= 2;
      std::vector<bool> dp(sum + 1);
      dp[0] = true; // 和为0一定能保证
      for (int i = 1; i <= nums.size(); i++) {
        for (int j = sum; j >= 0; j--) { // 0-1背包问题反向递归
          if (j - nums[i - 1] >= 0) {
            dp[j] = dp[j] || dp[j - nums[i - 1]];
          }
        }
      }
      return dp.back();
    }
  };
  std::vector<int> nums{1, 1, 2, 2};
  SolutionSpace slo;
  EXPECT_TRUE(slo.canPartition(nums));
}

TEST(mid, T474) {
  class Solution {
   public:
    std::pair<int, int> Str01(const std::string &str) {
      int zeros = 0, ones = 0;
      for (const auto c : str) {
        if (c == '0')
          zeros++;
        else {
          ones++;
        }
      }
      return {zeros, ones};
    }
    int findMaxForm(vector<string> &strs, int m, int n) {
      std::vector<std::vector<std::vector<int>>>
          dp(strs.size() + 1, std::vector<std::vector<int>>(m + 1, std::vector<int>(n + 1)));
      for (int i = 1; i <= strs.size(); i++) {
        for (int j = 0; j <= m; j++) {
          for (int k = 0; k <= n; k++) {
            auto zero_one = Str01(strs[i - 1]);
            if (j - zero_one.first >= 0 && k - zero_one.second >= 0) {
              dp[i][j][k] = std::max(dp[i - 1][j - zero_one.first][k - zero_one.second] + 1, dp[i - 1][j][k]);
            } else {
              dp[i][j][k] = dp[i - 1][j][k];
            }
          }
        }
      }
      return dp.back().back().back();
    }
  };

  class SolutionSpace {
   public:
    std::pair<int, int> Str01(const std::string &str) {
      int zeros = 0, ones = 0;
      for (const auto c : str) {
        if (c == '0')
          zeros++;
        else {
          ones++;
        }
      }
      return {zeros, ones};
    }

    int findMaxForm(vector<string> &strs, int m, int n) {
      std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));
      for (int i = 1; i <= strs.size(); i++) {
        for (int j = m; j >= 0; j--) {// 0-1问题，反向求解
          for (int k = n; k >= 0; k--) {
            auto zero_one = Str01(strs[i - 1]);
            if (j - zero_one.first >= 0 && k - zero_one.second >= 0)
              dp[j][k] = std::max(dp[j][k], dp[j - zero_one.first][k - zero_one.second] + 1);
          }
        }
      }
      return dp.back().back();
    }
  };
}

TEST(mid, T322) {
  class Solution {
   public:
    int coinChange(vector<int> &coins, int amount) {
      const int invalid = std::numeric_limits<int>::max();
      std::vector<std::vector<int>> dp(coins.size() + 1, std::vector<int>(amount + 1, invalid));
      for (int i = 0; i <= coins.size(); i++) {
        dp[i][0] = 0;
      }
      for (int i = 1; i <= coins.size(); i++) {
        for (int j = 0; j <= amount; j++) {
          int c1 = dp[i - 1][j];
          int c2 = j - coins[i - 1] >= 0 && dp[i][j - coins[i - 1]] < invalid ? dp[i][j - coins[i - 1]] + 1 : invalid;
          dp[i][j] = min(c1, c2);
        }
      }
      if (dp.back().back() == invalid)
        return -1;
      else
        return dp.back().back();
    }
  };

  class SolutionSpace {
   public:
    int coinChange(vector<int> &coins, int amount) {
      const int invalid = std::numeric_limits<int>::max();
      std::vector<int> dp(amount + 1, invalid);
      dp.front() = 0;
      for (int i = 1; i <= coins.size(); i++) {
        for (int j = 0; j <= amount; j++) { // 无限背包问题正向寻找
          int c1 = j - coins[i - 1] >= 0 && dp[j - coins[i - 1]] < invalid ? dp[j - coins[i - 1]] + 1 : invalid;
          dp[j] = std::min(dp[j], c1);
        }
      }
      if (dp.back() == invalid)
        return -1;
      else
        return dp.back();
    }
  };
  Solution slo;
  std::vector<int> nums{1, 2, 5};
  EXPECT_EQ(slo.coinChange(nums, 11), 3);
}

}//namespace of bag problem

namespace string_problem {

TEST(mid, T1143) {
  // 重做
  class Solution {
   public:
    int longestCommonSubsequence(string text1, string text2) {
      std::vector<std::vector<int>> dp(text1.size() + 1, std::vector<int>(text2.size() + 1));
      for (int i = 1; i <= text1.size(); i++) {
        for (int j = 1; j <= text2.size(); j++) {
          dp[i][j] = std::max(dp[i - 1][j - 1] + (text1[i - 1] == text2[j - 1]), dp[i - 1][j]);
          dp[i][j] = std::max(dp[i][j], dp[i][j - 1]);
        }
      }
      return dp.back().back();
    }
  };
}

TEST(hard, T72) {
  // 重做
  class Solution {
   public:
    int minDistance(string word1, string word2) {

    }
  };
}

TEST(mid, T650) {
  class Solution {
   public:
    int minSteps(int n) {
      if (n <= 1) {
        return 0;
      }
      std::vector<pair<int, int>> dp(n + 1);// second 储存copy的缓存A的数量
      dp[2] = {2, 1}; // copy and paste 缓存copy了a的数量为1
      for (int i = 3; i <= n; i++) {
        std::pair<int, int> temp;
        if (i % 2 == 0 && i / 2 >= 2) {
          temp = {dp[i / 2].first + 2, i / 2}; // copy all一半再paste
        } else {
          temp = {std::numeric_limits<int>::max() - 100, 0}; // random big value
        }
        for (int j = 2; j < i; j++) {
          if ((i - j) % dp[j].second == 0) { // 直接paste缓存
            int buffer_paste_count = (i - j) / dp[j].second;
            if (temp.first > (dp[j].first + buffer_paste_count)) {
              temp = {dp[j].first + buffer_paste_count, dp[j].second};
            }
          }
        }
        dp[i] = temp;
      }
      return dp.back().first;
    }
  };
}

TEST(hard, T10) {
  class Solution {
   public:
    enum PROPERTY {
      FIX,
      RANDOM_CHAR,
      REPEAT,
      RANDOM_REPEAT_CHAR
    };
    bool isMatch(string s, string p) {
      std::vector<std::pair<char, PROPERTY>> reg;
      reg.reserve(p.size());
      int idx = 0;
      while (idx < p.size()) {
        if (idx != p.size() - 1 && p[idx + 1] == '*') {
          if (p[idx] == '.') {
            reg.emplace_back(0, RANDOM_REPEAT_CHAR);
          } else {
            reg.emplace_back(p[idx], REPEAT);
          }
          idx += 2;
        } else {
          if (p[idx] == '.') {
            reg.emplace_back(0, RANDOM_CHAR);
          } else {
            reg.emplace_back(p[idx], FIX);
          }
          idx++;
        }
      }

      std::vector<std::vector<char>> dp(s.size() + 1, std::vector<char>(reg.size() + 1));

      auto is_match = [](char c, const std::pair<char, PROPERTY> &property) -> char {
        if (property.second == RANDOM_CHAR)
          return 1;
        else if (c == property.first) {
          if (property.second == REPEAT) {
            return c;
          } else {
            return 1;
          }
        } else if (property.second == RANDOM_REPEAT_CHAR)
          return 3;
        return 0;
      };

      auto is_repeat = [](const std::pair<char, PROPERTY> &property) -> char {
        if ((property.second == RANDOM_REPEAT_CHAR || property.second == REPEAT))
          return 2;
        return 0;
      };

      dp[0][0] = true;
      for (int j = 1; j <= reg.size(); j++) {
        if (dp[0][j - 1] > 0 && is_repeat(reg[j - 1])) {
          dp[0][j] = 2;
        }
      }
      // dp[i][j] 0 不匹配
      // dp[i][j] == 1 正常匹配
      // dp[i][j]==2 通过删除匹配
      // dp[i][j]==3 通过.* 匹配
      // dp[i][j] >= 'a' 通过重复dp[i][j] 进行匹配

      for (int i = 1; i <= s.size(); i++) {
        for (int j = 1; j <= reg.size(); j++) {
          auto s_i = s[i - 1];
          const auto &reg_j = reg[j - 1];
          if (dp[i - 1][j - 1] && is_match(s_i, reg_j)) {
            dp[i][j] = is_match(s_i, reg_j);
            continue;
          } else if (dp[i][j - 1] && is_repeat(reg_j)) {
            if (reg_j.second == REPEAT)
              dp[i][j] = 2; // 通过删除reg_j匹配
            else
              dp[i][j] = 3; // 通过.*匹配
            continue;
          } else if (dp[i - 1][j] && is_repeat(reg_j)) {
            char match_char = dp[i - 1][j];
            if (match_char == s_i) {
              dp[i][j] = s_i;
              continue;
            } else if (match_char == 3) {
              dp[i][j] = 3;
              continue;
            }
          }
          dp[i][j] = false;
        }
      }
      return dp.back().back();
    }
  };
  Solution slo;
  slo.isMatch(std::string("aaabaaaababcbccbaa"), std::string("c*c*.*c*a*..*c*"));
}

} // namespace of string problem

namespace stock_exchange {

TEST(easy, T121) {
  // 重做!!
  class Solution {
   public:
    int maxProfit(const vector<int> &prices) {
      std::vector<int> dp(prices.size());
      int res = 0;
      dp[0] = prices[0];
      for (int i = 1; i < dp.size(); i++) {
        int buy = dp[i - 1];
        int sell = prices[i] - dp[i - 1];
        res = std::max(sell, res);
        if (prices[i] < dp[i - 1]) {
          dp[i] = prices[i];
        } else {
          dp[i] = dp[i - 1];
        }
      }
      return res;
    }
  };
  Solution slo;
  slo.maxProfit({7, 1, 5, 3, 6, 4});
}

TEST(hard, T188) {
  class Solution {
   public:
    int dfs(const std::vector<std::pair<int, int>> &price_ups,
            int k,
            int start_idx) {
      if (k == 0)
        return 0;

      int max_profit = 0;
      for (int i = start_idx; i < price_ups.size(); i++) {
        int new_profit = price_ups[i].second - price_ups[start_idx].first;
        max_profit = std::max(new_profit, max_profit);
        for (int j = i + 1; j < price_ups.size(); j++) {
          int temp_profit = 0;
          if (memo[j].find(k - 1) == memo[j].end()) {
            temp_profit = dfs(price_ups, k - 1, j);
            memo[j][k - 1] = temp_profit;
          } else
            temp_profit = memo[j][k - 1];
          max_profit = std::max(new_profit + temp_profit, max_profit);
        }
      }
      return max_profit;
    }

    int dfs2(const std::vector<std::pair<int, int>> &price_ups,
             int k,
             int start_idx) {
      if (k == 0)
        return 0;

      int max_profit = 0;
      for (int i = start_idx; i < price_ups.size(); i++) {
        int new_profit = price_ups[i].second - price_ups[start_idx].first;
        max_profit = std::max(new_profit, max_profit);
        for (int j = i + 1; j < price_ups.size(); j++) {
          int temp_profit = 0;
          temp_profit = dfs2(price_ups, k - 1, j);
          max_profit = std::max(new_profit + temp_profit, max_profit);
        }
      }
      return max_profit;
    }

    int maxProfit(int k, const vector<int> &prices) {
      if (prices.size() < 2 || k == 0)
        return 0;
      std::vector<int> dp(prices.size()); // 以[i]为最大值之前的最小值
      std::vector<std::pair<int, int>> price_ups; // 价格上涨的区间
      dp[0] = prices[0];
      bool is_last_day_sell = true;
      for (int i = 1; i < prices.size(); i++) {
        if (prices[i] >= prices[i - 1]) {
          dp[i] = dp[i - 1];
          if (i == prices.size() - 1) // 最后一天价格还在涨
            is_last_day_sell = false;
        } else { // 价格下跌了
          if (prices[i - 1] > dp[i - 1])
            price_ups.emplace_back(dp[i - 1], prices[i - 1]); // 找到一个上涨区间
          dp[i] = prices[i];
        }
      }
      if (!is_last_day_sell) {
        price_ups.emplace_back(dp.back(), prices.back());
      }
      memo.clear();
      memo.resize(price_ups.size());

      auto start = std::chrono::high_resolution_clock::now();
      int res_max = 0;
      for (int i = 0; i < price_ups.size(); i++) {
        int temp = dfs(price_ups, k, i);
        res_max = std::max(res_max, temp);
      }
      auto end = std::chrono::high_resolution_clock::now();
      std::cout << "Time Momo : " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e6
                << "ms" << std::endl;

      start = std::chrono::high_resolution_clock::now();
      res_max = 0;
      for (int i = 0; i < price_ups.size(); i++) {
        int temp = dfs2(price_ups, k, i);
        res_max = std::max(res_max, temp);
      }
      end = std::chrono::high_resolution_clock::now();
      std::cout << "Time Without Momo : "
                << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e6 << "ms" << std::endl;

      return res_max;
    }
    std::vector<std::map<int, int>> memo; // [i][j] i-表示从第i个开始 j 表示当前交易次数
  };
  Solution slo;
  slo.maxProfit(9,
                {70, 4, 83, 56, 94, 72, 78, 43, 2, 86, 65, 100, 94, 56, 41, 66, 3, 33, 10, 3, 45, 94, 15, 12, 78, 60,
                 58, 0, 58, 15, 21, 7, 11, 41, 12, 96, 83, 77, 47, 62, 27, 19, 40, 63, 30, 4, 77, 52, 17, 57, 21, 66});
}

} // namespace of stock exchange

}// namespace of dynamic program


namespace divide_conquer {

TEST(mid, T241) {
  // 重做
  class Solution {
    enum OPERATOR {
      ADD,
      SUB,
      PRO,
    };
   public:
    // 左闭右开

    void Utility(int l, int r, const std::vector<int> &nums, const std::vector<OPERATOR> &op) {
      if (l == r) {
        memo[{l, r}] = {nums[l]};
        return;
      }

      std::vector<int> res;
      for (int i = l; i < r; i++) {
        auto cur_op = op[i];
        if (memo.find({l, i}) == memo.end()) {
          Utility(l, i, nums, op);
        }
        if (memo.find({i + 1, r}) == memo.end()) {
          Utility(i + 1, r, nums, op);
        }
        const auto &l_vec = memo[{l, i}];
        const auto &r_vec = memo[{i + 1, r}];
        for (auto l_num : l_vec) {
          for (auto r_num : r_vec) {
            switch (cur_op) {
              case ADD:res.push_back(l_num + r_num);
                break;
              case SUB:res.push_back(l_num - r_num);
                break;
              case PRO:res.push_back(l_num * r_num);
                break;
            }
          }
        }
      }
      memo[{l, r}] = res;
    }

    vector<int> diffWaysToCompute(string expression) {
      std::vector<int> nums;
      std::vector<OPERATOR> op;
      std::string temp;
      for (int i = 0; i < expression.size(); i++) {
        char c = expression[i];
        if (c != '+' && c != '-' && c != '*') {
          temp.push_back(c);
        } else {
          nums.push_back(std::stoi(temp));
          temp.clear();
          if (c == '+') {
            op.push_back(ADD);
          } else if (c == '-') {
            op.push_back(SUB);
          } else {
            op.push_back(PRO);
          }
        }
      }
      nums.push_back(stoi(temp));
      temp.clear();

      Utility(0, nums.size() - 1, nums, op);

      return memo[{0, nums.size() - 1}];
    }
    std::map<std::pair<int, int>, std::vector<int>> memo;
  };
  Solution slo;
  slo.diffWaysToCompute("2*3-4*5");
}

} // namespace divide conquer


namespace data_struct {
TEST(easy, T448) {
  class Solution {
   public:
    vector<int> findDisappearedNumbers(vector<int> &nums) {
      int n = nums.size();
      std::sort(nums.begin(), nums.end());
      std::vector<int> res;
      int cur_num = 0;
      for (int i = 0; i < nums.size(); i++) {
        if (nums[i] != cur_num) {
          cur_num++;
          while (cur_num < nums[i]) {
            res.push_back(cur_num);
            cur_num++;
          }
        }
      }
      if (cur_num < n) {
        cur_num++;
        while (true) {
          res.push_back(cur_num);
          cur_num++;
          if (cur_num > n)
            break;
        }
      }
      return res;
    }
  };
}

TEST(mid, T48) {
  class Solution {
   public:
    void rotate(vector<vector<int>> &matrix) {
      // 重做
    }
  };
}

TEST(mid, T240) {
  class Solution {
   public:
    int RowEnd(const vector<vector<int>> &mat, int r_idx) {
      return mat[r_idx].back();
    }
    int ColEnd(const vector<vector<int>> &mat, int col_idx) {
      return mat.back()[col_idx];
    }

    bool SearchOneRow(const vector<vector<int>> &mat, int row) const {
      int l = row, r = col_size - 1;
      auto get = [&](int i) {
        return mat[row][i];
      };
      while (r >= l) {
        int mid = (l + r) / 2;
        if (get(mid) > target_) {
          r = mid - 1;
        } else if (get(mid) < target_) {
          l = mid + 1;
        } else {
          return true;
        }
      }
      return false;
    }

    bool SearchOneCol(const vector<vector<int>> &mat, int col) const {
      int l = col, r = row_size - 1;
      auto get = [&](int i) {
        return mat[i][col];
      };
      while (r >= l) {
        int mid = (l + r) / 2;
        if (get(mid) > target_) {
          r = mid - 1;
        } else if (get(mid) < target_) {
          l = mid + 1;
        } else {
          return true;
        }
      }
      return false;
    }

    bool searchMatrix(const vector<vector<int>> &matrix, int target) {
      int search_idx = 0;
      row_size = matrix.size();
      col_size = matrix.front().size();
      target_ = target;
      while (search_idx < matrix.size() && search_idx < matrix.front().size()) {
        int row_end = RowEnd(matrix, search_idx);
        if (row_end >= target) { // 这一行可能有值
          if (SearchOneRow(matrix, search_idx))
            return true;
        }

        int col_end = ColEnd(matrix, search_idx);
        if (col_end >= target) { // 这一列可能有值
          if (SearchOneCol(matrix, search_idx))
            return true;
        }

        search_idx++;
      }
      return false;
    }
    int row_size;
    int col_size;
    int target_;
  };
  Solution slo;
  slo.searchMatrix({{1, 4, 7, 11, 15},
                    {2, 5, 8, 12, 19},
                    {3, 6, 9, 16, 22},
                    {10, 13, 14, 17, 24},
                    {18, 21, 23, 26, 30}}, 5);
}

TEST(hard, T23) {
  class Solution {
   public:
    // 左闭右开
    void Merge(vector<ListNode *> &lists, int l, int r) {
      if (r - l <= 1)
        return;
      int mid = (l + r) / 2;
      Merge(lists, l, mid); // exclude mid
      Merge(lists, mid, r); // include mid


      ListNode* list1 = nullptr;
      ListNode *list2 = nullptr;
      for (int i = l; i < r; i++) {
        if (lists[i] != nullptr) {
          if(list1 == nullptr) {
            list1 = lists[i];
            lists[i] = nullptr;
          }else if(list2 == nullptr){
            list2 = lists[i];
            lists[i] = nullptr;
            break;
          }
        }
      }
      if (list2 == nullptr) {
        lists[l] = list1;
        return;
      }
      // 每次都放在最左边->list1
      ListNode *c1 = list1, *c2 = list2;
      ListNode *top = c1->val > c2->val ? c2 : c1;
      ListNode *cur_node = nullptr, *temp;

      if (c1->val > c2->val) {
        cur_node = c2;
        c2 = c2->next;
      }else {
        cur_node = c1;
        c1 = c1->next;
      }

      while (c1 != nullptr && c2 != nullptr) {
        if (c1->val > c2->val) {
          temp = c2->next;
          cur_node->next = c2;
          cur_node = cur_node->next;
          c2 = temp;
        }else{
          temp = c1->next;
          cur_node->next = c1;
          cur_node = cur_node->next;
          c1 = temp;
        }
      }
      if (c1 != nullptr) {
        cur_node->next = c1;
      } else {
        cur_node->next = c2;
      }

      lists[l] = top;
    }
    ListNode *mergeKLists(vector<ListNode *> &lists) {
      Merge(lists, 0, lists.size());
      for(auto list : lists) {
        if(list != nullptr)
          return list;
      }
      return nullptr;
    }
  };
  ListNode* list1 = new ListNode(1);
  list1->next = new ListNode(4);
  list1->next->next = new ListNode(5);
  ListNode* list2 = new ListNode(1);
  list2->next = new ListNode(3);
  list2->next->next = new ListNode(4);
  ListNode* list3 = new ListNode(2);
  list3->next = new ListNode(6);
  std::vector<ListNode*> lists{list1, list2, list3};
  Solution slo;
  slo.mergeKLists(lists);
  lists = {{},new ListNode(1)};
  slo.mergeKLists(lists);
}

} // namespace of data struct

namespace daily_test {
TEST(mid, T1706) {
  class Solution {
   public:
    vector<int> findBall(const vector<vector<int>> &grid) {
      int row_num = grid.size();
      int col_num = grid.front().size();
      if (col_num == 1)
        return {-1};
      auto is_move_able = [&](int row, int col) {
        auto rows = grid[row];
        // left bound
        if (col == 0) {
          if (rows[col] == -1)
            return false;
          else
            return true;
        }
        if (col == col_num - 1) {
          if (rows[col] == 1)
            return false;
          else
            return true;
        }
        int cur_dir = rows[col];
        int next_dir;
        if (cur_dir == 1) {
          next_dir = rows[col + 1];
        } else {
          next_dir = rows[col - 1];
        }
        return cur_dir == next_dir;
      };

      std::vector<int> ans(col_num);
      auto *pre_row = &ans;

      for (int i = 0; i < col_num; i++) {
        int col = i;
        for (int row = 0; row < row_num; row++) {
          if (is_move_able(row, col))
            col += grid[row][col];
          else {
            col = -1;
            break;
          }
        }
        ans[i] = col;
      }
      return ans;
    }
  };
}

}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
