#include <iostream>
#include <vector>
#include <algorithm>
#include <gtest/gtest.h>
#include <unordered_map>
#include <map>
#include <string>
#include <climits>
#include <queue>

using namespace std;

TEST(test,test1){
    ASSERT_TRUE(true);
}

TEST(test,test2){
    ASSERT_TRUE(false);
}

TEST(MID,T221){
    class Solution {
    public:
        typedef std::pair<size_t,size_t> loc_type;
        int maximalSquare(vector<vector<char>>& matrix) {
            if(matrix.empty() || matrix[0].empty())
                return 0;
            mat_ = &matrix;
            num_row = matrix.size();
            num_col = matrix[0].size();
            max_size=0;
            search({0,0});
            return (int)(max_size*max_size);
        }

        void search(loc_type loc){
            size_t row = loc.first;
            size_t col = loc.second;
            if((row + max_size >= num_row && col + max_size>= num_col) || (row>=num_row-1&&col==num_col-1)){
                if(max_size==0&& mat_->at(row)[col]!='0')
                    max_size=1;
                return;
            }
            if(mat_->at(row).at(col)=='0'){
                search(next_point(loc));
                return;
            }
            size_t sz=1;
            for(;;){
                if(row+sz>=num_row||col+sz>=num_col)
                    break;
                bool has_zero = false;

                if(mat_->at(row+sz)[col+sz]=='0')
                    break;
                for(size_t r=row;r < row+sz;r++){
                    if(mat_->at(r)[col+sz]=='0'){
                        has_zero = true;
                        break;
                    }
                }
                if(has_zero)
                    break;

                for(size_t c = col; c< col+sz;c++){
                    if(mat_->at(row+sz)[c]=='0'){
                        has_zero = true;
                        break;
                    }
                }
                if(has_zero)
                    break;

                sz++;
            }
            if(sz>max_size)
                max_size = sz;
            search(next_point(loc));
        }

        loc_type next_point(const loc_type &loc){
            loc_type next_loc;
            size_t row = loc.first;
            size_t col = loc.second;
            if(col < num_col-1){
                col++;
            }else{
                row += 1;
                col = 0;
            }
            return {row,col};
        }

    private:
        size_t num_row,num_col;
        vector<vector<char>> *mat_;
        size_t max_size=0;
    };

    vector<vector<char>> matrix {{'1','0','1','0','0'},
                                 {'1','0','1','1','1'},
                                 {'1','1','1','1','1'},
                                 {'1','0','0','1','0'}};
    Solution slo;
    int area = slo.maximalSquare(matrix);
    EXPECT_EQ(area,4);

    vector<vector<char>> matrix1;
    area = slo.maximalSquare(matrix1);
    EXPECT_EQ(area,0);

    vector<vector<char>> matrix2={{'1'}};
    area = slo.maximalSquare(matrix2);
    EXPECT_EQ(area,1);

    vector<vector<char>> matrix3={{'0','0'}};
    area = slo.maximalSquare(matrix3);
    EXPECT_EQ(area,0);

    vector<vector<char>> matrix4={{'0'}};
    area = slo.maximalSquare(matrix4);
    EXPECT_EQ(area,0);

    vector<vector<char>> matrix5={{'0'},{'1'}};
    area = slo.maximalSquare(matrix5);
    EXPECT_EQ(area,1);

    vector<vector<char>> matrix6={{'0','0','0','1'},
                                  {'1','1','0','1'},
                                  {'1','1','1','1'},
                                  {'0','1','1','1'},
                                  {'0','1','1','1'}};
    area = slo.maximalSquare(matrix6);
    EXPECT_EQ(area,9);

    vector<vector<char>> matrix7={{'0','0','0','1','0','1','1','1'},
                                  {'0','1','1','0','0','1','0','1'},
                                  {'1','0','1','1','1','1','0','1'},
                                  {'0','0','0','1','0','0','0','0'},
                                  {'0','0','1','0','0','0','1','0'},
                                  {'1','1','1','0','0','1','1','1'},
                                  {'1','0','0','1','1','0','0','1'},
                                  {'0','1','0','0','1','1','0','0'},
                                  {'1','0','0','1','0','0','0','0'}};
    area = slo.maximalSquare(matrix7);
    EXPECT_EQ(area,1);
}

TEST(MID,T221_DYNAMIC_PROGRAM){
    class Solution {
    public:
        int maximalSquare(vector<vector<char>>& matrix) {
            if(matrix.empty() || matrix[0].empty())
                return 0;
            int num_row = matrix.size();
            int num_col = matrix[0].size();
            int max_size=0;
            std::vector<std::vector<int>> dp(num_row, std::vector<int>(num_col));
            for(int row = 0; row< num_row; row++){
                for(size_t col=0; col < num_col; col++){
                    if(matrix[row][col]!='0'){
                        if(row==0||col==0){
                            dp[row][col]=1;
                        } else{
                            dp[row][col] = min(min(dp[row - 1][col], dp[row][col - 1]), dp[row - 1][col - 1]) + 1;
                        }
                    }
                    max_size = max(max_size, dp[row][col]);
                }
            }
            return int(max_size*max_size);
        }


    };
    vector<vector<char>> matrix {{'1','0','1','0','0'},
                                 {'1','0','1','1','1'},
                                 {'1','1','1','1','1'},
                                 {'1','0','0','1','0'}};
    Solution slo;
    int area = slo.maximalSquare(matrix);
    EXPECT_EQ(area,4);

    vector<vector<char>> matrix1;
    area = slo.maximalSquare(matrix1);
    EXPECT_EQ(area,0);

    vector<vector<char>> matrix2={{'1'}};
    area = slo.maximalSquare(matrix2);
    EXPECT_EQ(area,1);

    vector<vector<char>> matrix3={{'0','0'}};
    area = slo.maximalSquare(matrix3);
    EXPECT_EQ(area,0);

    vector<vector<char>> matrix4={{'0'}};
    area = slo.maximalSquare(matrix4);
    EXPECT_EQ(area,0);

    vector<vector<char>> matrix5={{'0'},{'1'}};
    area = slo.maximalSquare(matrix5);
    EXPECT_EQ(area,1);

    vector<vector<char>> matrix6={{'0','0','0','1'},
                                  {'1','1','0','1'},
                                  {'1','1','1','1'},
                                  {'0','1','1','1'},
                                  {'0','1','1','1'}};
    area = slo.maximalSquare(matrix6);
    EXPECT_EQ(area,9);

    vector<vector<char>> matrix7={{'0','0','0','1','0','1','1','1'},
                                  {'0','1','1','0','0','1','0','1'},
                                  {'1','0','1','1','1','1','0','1'},
                                  {'0','0','0','1','0','0','0','0'},
                                  {'0','0','1','0','0','0','1','0'},
                                  {'1','1','1','0','0','1','1','1'},
                                  {'1','0','0','1','1','0','0','1'},
                                  {'0','1','0','0','1','1','0','0'},
                                  {'1','0','0','1','0','0','0','0'}};
    area = slo.maximalSquare(matrix7);
    EXPECT_EQ(area,1);
}

TEST(EASY,T69){
    class Solution {
    public:
        int mySqrt(int x) {
//            for(size_t i=1;i<=x; i++){
//                if(i*i<=x && (i+1)*(i+1) > x)
//                    return i;
//            }
            // 二分法
            if(x==1)
                return 1;
            unsigned long long mid=1;
            unsigned long long up = x,low=0;
            unsigned long long ans_1; // store (mid+1)*(mid+1)
            while(mid!=0){
                mid = (up+low)/2;
                ans_1 = (mid+1)*(mid+1);
                if(ans_1 <= x){
                    low = mid;
                }else if((mid)*(mid) > x){
                    up = mid;
                }else{
                    return mid;
                }
            }
            return 0;
        }
    };
    Solution slo;
    EXPECT_EQ(slo.mySqrt(0),0);
    EXPECT_EQ(slo.mySqrt(1),1);
    EXPECT_EQ(slo.mySqrt(4),2);
    EXPECT_EQ(slo.mySqrt(8),2);
    EXPECT_EQ(slo.mySqrt(9),3);
    EXPECT_EQ(slo.mySqrt(2147395599),46339);
}

TEST(MID, T1105){
    class Solution {
    public:
        int minHeightShelves(vector<vector<int>>& books, int shelf_width) {
            size_t num_book = books.size();
            std::vector<int> dp(num_book, 0xfffffff);
            dp[0]=0;
            for(const auto &book:books){

            }
            return 1;
        }
    };
    std::cerr<<"Cant Understand This Question\n";
    EXPECT_TRUE(false);
}

TEST(EASY,T704){
    class Solution {
    public:
        int search(vector<int>& nums, int target) {
            int left,right,mid;
            left = 0;
            right = nums.size() -1;
            while(left <= right){
                mid = (left +right)/2;
                if(nums[mid] == target)
                    return mid;
                else if(nums[mid] > target)
                    right = mid - 1;
                else
                    left = mid + 1;
            }
            return -1;
        }
    };
    Solution slo;
    std::vector<int> a;
    EXPECT_EQ(slo.search(a ,1),-1);

    a={1};
    EXPECT_EQ(slo.search(a,1),0);

    a={-1,0,3,5,9,12};
    EXPECT_EQ(slo.search(a ,9),4);

    a={-1,0,3,5,9,12};
    EXPECT_EQ(slo.search(a ,2),-1);
}

TEST(MID, T236){
    struct TreeNode {
        int val;
        TreeNode *left;
        TreeNode *right;
        TreeNode(int x) : val(x), left(NULL), right(NULL) {}
    };

    class Solution {
    public:
        TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
            p_ = p;
            q_ = q;
            path.clear();
            path_q.clear();
            path_p.clear();
            Dfs(root);
            size_t id=0, max_sz = min(path_p.size(),path_q.size());
            while(id < max_sz - 1){
                if( path_q[id]->val == path_p[id]->val && path_q[id + 1]->val != path_p[id + 1]->val)
                    return path_p[id];
                id++;
            }
            return path_p[max_sz-1];
        }

        void Dfs(TreeNode* cur_node){
            path.push_back(cur_node);
            if(cur_node->val == p_->val){
                path_p = path;
            }else if(cur_node->val == q_->val)
                path_q = path;

            if(!path_p.empty() && !path_q.empty())
                return;

            if(cur_node->left != nullptr){
                Dfs(cur_node->left);
            }
            if(cur_node->right != nullptr){
                Dfs(cur_node->right);
            }
            path.pop_back();
        }

    private:
        TreeNode *p_, *q_;
        std::vector<TreeNode*> path,path_p,path_q;
    };
    EXPECT_TRUE(true);
}


TEST(HARD, T84){
    class Solution {
    public:
        struct Area{
            Area():val(0),is_continue(false),temp_val(0){}
            unsigned int val;
            bool is_continue;
            unsigned int temp_val;
            explicit operator bool() const {
                return is_continue;
            }
            void operator += (unsigned int _val){
                val+=_val;
            }
        };

        int largestRectangleArea(vector<int>& heights) {
            if(heights.empty())
                return 0;
            std::map<int,Area> dp;

            for(auto h:heights){
                // emplace insert
                dp.emplace(std::piecewise_construct,
                           std::forward_as_tuple(h),
                           std::forward_as_tuple());
            }

            for(unsigned int h : heights){
                // emplace insert
                auto it = dp.find(h);
                // set all height bigger than h as false;
                for(auto temp_it = it; temp_it != dp.end(); temp_it++){
                    if(temp_it == it)
                        continue;
                    temp_it->second.is_continue = false;
                }
                // set all height lower than h
                auto temp_it = dp.begin();
                while(true){
                    if(temp_it == it){
                        SetIt(temp_it);
                        break;
                    }
                    SetIt(temp_it);
                    temp_it++;
                }
            }

            int max_area = 0;
            for(const auto& p : dp){
                int area = max(p.second.val,p.second.temp_val);
                if(area>max_area)
                    max_area = area;
            }
            return max_area;
        }

        static void SetIt(std::map<int,Area>::iterator &it){
            if(it->first == 0)
                return;
            if(!(it->second)){
                if(it->second.val > it->second.temp_val)
                    it->second.temp_val = it->second.val;
                it->second.val = it->first;
                it->second.is_continue = true;
            }else{
                (it->second)+=(it->first);
            }
        }
    };
    Solution slo;
    vector<int> h;

    h={2,1,5,6,2,3};
    EXPECT_EQ(slo.largestRectangleArea(h),10);

    h={3,5,5,2,5,5,6,6,4,4,1,1,2,5,5,6,6,4,1,3};
    EXPECT_EQ(slo.largestRectangleArea(h),24);

    h={0,0,0,0,0,0,0,0,2147483647};
    EXPECT_EQ(slo.largestRectangleArea(h),2147483647);

    h.clear();
    for(int i=0;i<=5; i++){
        h.push_back(i);
    }
    EXPECT_EQ(slo.largestRectangleArea(h),9);

    h.clear();
    h.reserve(19999+1);
    for(int i=0;i<=19999; i++){
        h.push_back(i);
    }
    EXPECT_EQ(slo.largestRectangleArea(h),100000000);
}

TEST(HARD, T84_Stack){
    class Solution {
    public:
        int largestRectangleArea(vector<int>& heights) {
            return 0;
        }
    };
    Solution slo;
    vector<int> h;

    h={2,1,5,6,2,3};
    EXPECT_EQ(slo.largestRectangleArea(h),10);

    h={3,5,5,2,5,5,6,6,4,4,1,1,2,5,5,6,6,4,1,3};
    EXPECT_EQ(slo.largestRectangleArea(h),24);

    h={0,0,0,0,0,0,0,0,2147483647};
    EXPECT_EQ(slo.largestRectangleArea(h),2147483647);

    h.clear();
    for(int i=0;i<=5; i++){
        h.push_back(i);
    }
    EXPECT_EQ(slo.largestRectangleArea(h),9);

    h.clear();
    h.reserve(19999+1);
    for(int i=0;i<=19999; i++){
        h.push_back(i);
    }
    EXPECT_EQ(slo.largestRectangleArea(h),100000000);
}

TEST(EASY,T942){
    class Solution {
    public:
        vector<int> diStringMatch(string S) {
            size_t n = S.size();
            if(n==0)
                return vector<int>();
            vector<int> ans(n+1);
            size_t low = 0,up=n;

            for(size_t i=0; i< n; i++){
                if(S[i]=='I')
                    ans[i] = low++;
                else
                    ans[i] = up--;
            }
            ans[n] = low;
            return ans;
        }
    };

    Solution slo;
    std::vector<int> r;
    r = {0,4,1,3,2};
    EXPECT_EQ(slo.diStringMatch("IDID"),r);
    r = {0,1,2,3};
    EXPECT_EQ(slo.diStringMatch("III"),r);
}


TEST(MID, T46){
    class Solution {
    public:
        vector<vector<int>> permute(vector<int>& nums) {
            std::vector<std::vector<int>> ans;
            if(nums.empty())
                return ans;
            if(nums.size()==1){
                ans.push_back(nums);
                return ans;
            }
            std::vector<bool> block(nums.size());
            path.reserve(nums.size());
            std::sort(nums.begin(),nums.end());
            for(size_t i=0; i< nums.size() ; i++){
                dfs(i,ans,block,nums);
            }
            return ans;
        }

        void dfs(size_t cur_id,std::vector<std::vector<int>> &ans,std::vector<bool> &block, vector<int>& nums){
            path.push_back(nums[cur_id]);
            block[cur_id] = true;
            for(size_t i=0; i< nums.size(); i++){
                if(!block[i]){
                    if(path.size() == nums.size() - 1){
                        path.push_back(nums[i]);
                        ans.push_back(path);
                        path.pop_back();
                    }else{
                        dfs(i,ans,block,nums);
                    }
                }
            }
            path.pop_back();
            block[cur_id] = false;
        }

    private:
        std::vector<int> path;
    };

    Solution slo;
    vector<int> nums;
    vector<vector<int>> res;

    nums = {1};
    res={{1}};
    EXPECT_EQ(slo.permute(nums),res);

    nums = {1,2,3};
    res = {
            {1,2,3},
            {1,3,2},
            {2,1,3},
            {2,3,1},
            {3,1,2},
            {3,2,1}
    };
    EXPECT_EQ(slo.permute(nums),res);

}

TEST(MID, T991){
    class Solution {
    public:
        int brokenCalc(int X, int Y) {
            int count=0;
            while (Y>X){
                if(Y%2){
                    Y = (Y+1)/2;
                    count+=2;
                }else{
                    Y = Y /2;
                    count++;
                }
            }
            return X - Y + count;
        }
    };
    Solution slo;

    EXPECT_EQ(slo.brokenCalc(2,3),2);
    EXPECT_EQ(slo.brokenCalc(5,8),2);
    EXPECT_EQ(slo.brokenCalc(3,10),3);
}


TEST(EASY, T155){
    class MinStack {
    public:
        /** initialize your data structure here. */
        MinStack(){
        }

        void push(int x) {
            List* pre = data_;
            data_ = new List(x,pre);
        }

        void pop() {
            List *cur = data_;
            data_ = data_->pre_list;
//            delete cur;
        }

        int top() {
            return data_->val;
        }

        int getMin() {
            return data_->min_val;
        }

    private:
        struct List{
            explicit List(int _val, List *_pre_list):val(_val),pre_list(_pre_list){
                if(pre_list){
                    min_val = min(pre_list->min_val,val);
                }else{
                    min_val = val;
                }
            }
            int min_val;
            int val;
            List *pre_list;
        };
        List *data_ = nullptr;
    };
    EXPECT_TRUE(true);
}

TEST(MID, T102){
    struct TreeNode {
        int val;
        TreeNode *left;
        TreeNode *right;
        TreeNode(int x, TreeNode *_left = nullptr, TreeNode *_right = nullptr) : val(x), left(_left), right(_right) {}
    };

    class Solution {
    public:
        vector<vector<int>> levelOrder(TreeNode* root) {
            struct TreeWithLayer{
                explicit TreeWithLayer(TreeNode* _tr, int _layer):treeNode(_tr),layer(_layer){}
                int NumLayer(){ return layer;}
                TreeNode * treeNode;
            private:
                int layer = 0;
            };
            vector<vector<int>> ans;
            if(!root){
                return ans;
            }
            std::queue<TreeWithLayer> qu;
            qu.emplace(root,0);
            std::vector<int> one_layer;
            int layer_count = 0;

            while(!qu.empty()){
                int cur_layer = qu.front().NumLayer();
                if(qu.front().treeNode->left){
                    qu.emplace(qu.front().treeNode->left, cur_layer+1);
                }
                if(qu.front().treeNode->right){
                    qu.emplace(qu.front().treeNode->right, cur_layer+1);
                }
                if(layer_count != cur_layer){
                    layer_count++;
                    ans.emplace_back(one_layer);
                    one_layer.clear();
                }
                one_layer.push_back(qu.front().treeNode->val);
                qu.pop();
            }
            ans.emplace_back(one_layer);
            return ans;
        }
    };

    TreeNode root(3, new TreeNode(9), new TreeNode(20, new TreeNode(15), new TreeNode(17)));
    Solution slo;
    vector<vector<int>> ans{{3},{9,20},{15,17}};
    EXPECT_EQ(slo.levelOrder(&root),ans);

    TreeNode root2(3);
    ans={{3}};
    EXPECT_EQ(slo.levelOrder(&root2),ans);
}

TEST(MID, T1218){
    class Solution {
    public:
        int longestSubsequence(vector<int>& arr, int difference) {
            if(difference == 0)
                return arr.size();
            if(difference < 0)
                sort(arr.begin(),arr.end(), std::greater<int>());
            else
                sort(arr.begin(),arr.end());

            int count = 0;


        }
    };
    EXPECT_TRUE(false);
}

TEST(EASY, T136){
    class Solution {
    public:
        int singleNumber(vector<int>& nums) {
            int ans = 0;
            for(auto n: nums){
                ans ^= n;
            }
            return ans;
        }
    };
    vector<int> nums;
    Solution slo;
    nums = {2,3,3,2,4,1,4};
    EXPECT_EQ(slo.singleNumber(nums),1);
}

TEST(MID, T137){
    class Solution {
    public:
        int singleNumber(vector<int>& nums) {
            int seen_once =0, seen_twice = 0;

            for(auto n: nums){
                seen_once = (~seen_twice) & (seen_once ^ n);
                seen_twice = (~seen_once) & (seen_twice ^ n);
            }
            return seen_once;
        }
    };
    vector<int> nums;
    Solution slo;
    nums = {2,2,3,2};
    EXPECT_EQ(slo.singleNumber(nums),3);
}

TEST(MID, T318){
    class Solution {
    public:
        int maxProduct(vector<string>& words) {
            int max = 0;
            // pre cal mask before compare
            vector<int> mask_array(words.size());

            for(size_t i=0;i<words.size();i++){
                mask_array[i] = CalStringMask(words[i]);
            }

            for(size_t i=0;i<words.size();i++){
                for (size_t j = 0; j < words.size(); ++j) {
                    if(i!=j&&!bool(mask_array[i]&mask_array[j])){
                        int temp_max = words[i].size() * words[j].size();
                        if( temp_max>max)
                            max = temp_max;
                    }
                }
            }
            return max;
        }

    private:
        inline int CalStringMask(const string &s1){
            int int1 = 0;
            for(const auto ch: s1)
                int1 |= 1<<(ch - 'a');
            return int1;
        }
        inline bool IsHasComm(const string& s1,const string& s2){
            int int1 = 0,int2 = 0;
            for(const auto ch: s1)
                int1 |= 1<<(ch - 'a');
            for(const auto ch : s2)
                int2 |= 1<<(ch - 'a');
            return bool(int1&int2);
        }
    };

    vector<string> words;
    Solution slo;

    words = {"abcw","baz","foo","bar","xtfn","abcdef"};
    EXPECT_EQ(slo.maxProduct(words),16);

    words = {"a","aa","aaa","aaaa"};
    EXPECT_EQ(slo.maxProduct(words),0);
}



int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
