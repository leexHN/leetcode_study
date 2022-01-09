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

using namespace std;

namespace greedy_algorithm {

class Solution {
 public:
   int findContentChildren(vector<int>& g,vector<int>& s) {
    std::sort(g.begin(), g.end());
    std::sort(s.begin(), s.end());
    int num = 0;
    size_t g_idx = 0;
    size_t s_idx = 0;
    while(g_idx < g.size() && s_idx < s.size()){
      if(g[g_idx] <= s[s_idx]){
        num++;
        g_idx++;
      }
      s_idx++;
    }
    return num;
  }
};

TEST(easy, T455) {
  Solution slo;
  std::vector<int> g,s;
  g={1,2,3};
  s={1,1};
  EXPECT_EQ(1, slo.findContentChildren(g,s));
  g={1,2};
  s={1,2,3};
  EXPECT_EQ(2, slo.findContentChildren(g,s));
}

}


int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
