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

TEST(MID, T718){
    class Solution {
    public:
        int findLength(vector<int>& A, vector<int>& B) {
            vector<vector<int>> dp(A.size() + 1, vector<int>(B.size()+1,0));
            int ans = 0;
            for(size_t a =1; a<= A.size(); a++){
                for(size_t b=1; b<= B.size(); b++){
                    if(A[a-1] == B[b - 1]){
                        dp[a][b] = dp[a-1][b-1] + 1;
                        ans = max(ans, dp[a][b]);
                    }
                }
            }
            return ans;
        }
    };

    vector<int> A,B;
    Solution slo;

    A={1,2,3,2,1};
    B= {3,2,1,4,7};
    EXPECT_EQ(slo.findLength(A,B),3);

}

TEST(MID, T718_1D){
    class Solution {
    public:
        int findLength(vector<int>& A, vector<int>& B) {
            vector<int> dp(A.size() + 1, 0);
            int ans = 0;
            for(size_t a = 1; a<= A.size(); a++){
                for(size_t b= B.size(); b>= 1; b--){
                    if(A[a-1] == B[b - 1]){
                        dp[b] = dp[b-1] + 1;
                        ans = max(ans, dp[b]);
                    } else
                        dp[b] = 0;
                }
            }
            return ans;
        }
    };

    vector<int> A,B;
    Solution slo;

    A={1,2,3,2,1};
    B= {3,2,1,4,7};
    EXPECT_EQ(slo.findLength(A,B),3);
    A={2,7,1,5,4};
    B= {4,3,2,7,1};
    EXPECT_EQ(slo.findLength(A,B),3);
}

TEST(MID, T560){
    class Solution {
    public:
        int subarraySum(vector<int>& nums, int k) {
            unordered_map<int, int> k_map;
            int ans = 0;
            int cum_sum = 0;
            for(auto n : nums){
                cum_sum+=n;
                int delta = cum_sum - k;
                auto it = k_map.find(delta);
                if(it != k_map.end())
                    ans += it->second;
                k_map[cum_sum]++;
            }
            ans += k_map[k];
            return ans;
        }
    };

    vector<int> nums;
    Solution slo;

    nums = {1,1,1};
    EXPECT_EQ(slo.subarraySum(nums,2),2);
    nums = {1,-1,1,-1};
    EXPECT_EQ(slo.subarraySum(nums,0),4);
}

TEST(HARD, T25){
    struct ListNode {
        int val;
        ListNode *next;

        ListNode(int x, ListNode* _next = nullptr) : val(x), next(_next) {}
    };

    class Solution {
    public:
        ListNode* reverseKGroup(ListNode* head, int k) {
            if(k == 1)
                return head;
            std::vector<ListNode*> st;
            st.reserve(k+1);
            st.push_back(nullptr);
            ListNode* cur_node = head;
            ListNode* ans = nullptr;
            while (cur_node != nullptr){
                st.push_back(cur_node);
                cur_node = cur_node->next;
                if(st.size() == k+1){
                    auto top = st.back();
                    while(st.size() != 2){
                        auto temp = st.back();
                        st.pop_back();
                        temp->next = st.back();
                    }
                    auto cur_block_end = st.back();
                    st.pop_back();
                    auto pre_block_end = st.back();
                    st.pop_back();
                    if(pre_block_end!= nullptr)
                        pre_block_end->next = top;
                    else
                        ans = top;
                    st.push_back(cur_block_end);
                }
            }
            if(st.size()>1){
                auto pre_block_end = st[0];
                auto top = st[1];
                pre_block_end ->next = top;
            }else{
                st.back()->next = nullptr;
            }
            return ans;
        }
    };

    auto CheckAns = [](ListNode* root, std::vector<int> data){
        ListNode* cur_node = root;
        size_t idx = 0;
        while(cur_node->next!= nullptr){
            if(data[idx] != cur_node->val)
                return false;
            idx++;
            cur_node = cur_node->next;
        }
        return true;
    };

    Solution slo;
    ListNode root(1,new ListNode(2,new ListNode(3,new ListNode(4,new ListNode(5)))));
    std::vector<int> ans;
    ans = {2,1,4,3,5};
    ASSERT_TRUE(CheckAns(slo.reverseKGroup(&root,2),ans));
    ListNode root1(1,new ListNode(2,new ListNode(3,new ListNode(4,new ListNode(5, new ListNode(6))))));
    ans = {2,1,4,3,6,5};
    ASSERT_TRUE(CheckAns(slo.reverseKGroup(&root1,2),ans));
    ListNode root2(1,new ListNode(2,new ListNode(3,new ListNode(4,new ListNode(5, new ListNode(6))))));
    ans ={6,5,4,3,2,1};
    ASSERT_TRUE(CheckAns(slo.reverseKGroup(&root2,6),ans));
}

TEST(MID, T957){
    class Solution {
    public:
        vector<int> prisonAfterNDays(vector<int>& cells, int N) {
            if (!N)
                return cells;
            int n = N % 14;
            if (!n)
                n = 14;
            while (n--) {
                vector<int> tmp = vector<int>(cells);
                for (int i = 1; i < 7; ++i)
                    cells[i] = (tmp[i - 1] ^ !tmp[i + 1]);
                cells[0] = cells[7] = 0;
            }
            return cells;
        }
    };

    Solution slo;
    vector<int> cells(8,1);
    vector<int> ans;
    cells = {0,1,0,1,1,0,0,1};
    ans = {0,0,1,1,0,0,0,0};
    EXPECT_EQ(slo.prisonAfterNDays(cells,7),ans);

    cells = {1,0,0,1,0,0,1,0};
    ans = {0,0,1,1,1,1,1,0};
    EXPECT_EQ(slo.prisonAfterNDays(cells,1000000000),ans);

    cells = {0,0,1,1,1,1,0,0};
    ans = {0,0,0,1,1,0,0,0};
    EXPECT_EQ(slo.prisonAfterNDays(cells,8),ans);
}

TEST(MID, T152){
    class Solution {
    public:
        int maxProduct(vector<int>& nums) {
            int pre_max = 1,pre_min = 1;
            int ans = nums[0];
            for(auto n:nums){
                auto temp = pre_max;
                pre_max = max(pre_max*n, max(pre_min*n,n));
                pre_min = min(pre_min*n, min(n, temp*n));
                ans = max(ans,pre_max);
            }
            return ans;
        }
    };

    Solution slo;
    vector<int> nums = {2,3,-2,4};

    EXPECT_EQ(slo.maxProduct(nums),6);

    nums = {-2,0,-1};
    EXPECT_EQ(slo.maxProduct(nums),0);

    nums = {0,2,0,5,7};
    EXPECT_EQ(slo.maxProduct(nums),35);

    nums = {-4,-3,-2};
    EXPECT_EQ(slo.maxProduct(nums),12);
}

TEST(EASY,T53){
    class Solution {
    public:
        int maxSubArray(vector<int>& nums) {
            int pre = 0, maxAns = nums[0];
            for (const auto &x: nums) {
                pre = max(pre + x, x);
                maxAns = max(maxAns, pre);
            }
            return maxAns;
        }
    };

    Solution slo;
    vector<int> nums = {1,100,-100,100};
    EXPECT_EQ(slo.maxSubArray(nums),101);
}

TEST(MID, T646){
    class Solution {
    public:
        int findLongestChain(vector<vector<int>>& pairs) {
            struct {
                bool operator () (const vector<int>&a, const vector<int>&b) const{
                    return a[1]<b[1];
                }
            }Comp;
            sort(pairs.begin(), pairs.end(), Comp);

            int cur = INT32_MIN, step = 0;
            for(const auto &p: pairs){
                if(p[1] < cur){
                    cur = p[0];
                    step++;
                }
            }
            return step;
        }
    };
    EXPECT_TRUE(true);
}

TEST(EASY, T680){
    class Solution {
    public:
        bool palindrome(const std::string& s, int i, int j)
        {
            for ( ; i < j && s[i] == s[j]; ++i, --j);
            return i >= j;
        }

        bool validPalindrome(string s) {
            int i = 0, j = s.size() - 1;
            for ( ; i < j && s[i] == s[j]; ++i, --j);
            return palindrome(s, i, j - 1) || palindrome(s, i + 1, j);
        }
    };

    Solution slo;
    EXPECT_EQ(slo.validPalindrome("a"),true);
    EXPECT_EQ(slo.validPalindrome("aba"),true);
    EXPECT_EQ(slo.validPalindrome("abca"),true);
    EXPECT_EQ(slo.validPalindrome("abcca"),true);
    EXPECT_EQ(slo.validPalindrome("aebcbca"),false);
}

TEST(EASY, T771){
    class Solution {
    public:
        int numJewelsInStones(string J, string S) {
            set<char> idmal;
            for(auto c:J)
                idmal.insert(c);
            int start = 0;
            for(auto c:S){
                if(idmal.count(c) == 1)
                    start++;
            }
            return start;
        }
    };
    Solution slo;
    EXPECT_EQ(slo.numJewelsInStones("aA","aAAbbbb"),3);
}

TEST(MID, T894){
    struct TreeNode {
        int val;
        TreeNode *left;
        TreeNode *right;
        TreeNode(int x) : val(x), left(NULL), right(NULL) {}
    };

    class Solution {
        unordered_map<int, vector<TreeNode*>> memo;
    public:
        vector<TreeNode*> allPossibleFBT(int N) {
            if(N%2==0)
                return vector<TreeNode*>();

            if(memo.count(N))
                return memo[N];

            if(N==1){
                TreeNode* root = new TreeNode(0);
                memo[N].push_back(root);
                return memo[N];
            }

            for(int left=1; left < N; left+=2){
                int right = N-left-1;
                for(auto& left_node : allPossibleFBT(left)){
                    for(auto&  right_node: allPossibleFBT(right)){
                        TreeNode* root = new TreeNode(0);
                        root->left = left_node;
                        root->right = right_node;
                        memo[N].push_back(root);
                    }
                }

            }

            return memo[N];
        }
    };

    EXPECT_TRUE(true);
}

TEST(MID, T1371){
    class Solution {
    public:
        int findTheLongestSubstring(string s) {
            vector<int> pos(1u<<5u,-1); //a e i o u
            int ans = 0;
            unsigned char status = 0; //0 ~ even 1~odd
            pos[0] = 0; //all element is even minimum id is start
            for(int i=0;i<s.length();i++){
                switch (s[i]){
                    case 'a':
                        status ^= 0b1u<<4u;
                        break;
                    case 'e':
                        status ^= 0b1u<<3u;
                        break;
                    case 'i':
                        status ^= 0b1u<<2u;
                        break;
                    case 'o':
                        status ^= 0b1u<<1u;
                        break;
                    case 'u':
                        status ^= 0b1u;
                        break;
                }
                if(pos[status] == -1){ //current status have not pos id
                    pos[status] = i+1;//only store the minimum id
                }else{ // current status have a identical previous state. odd-odd = even enven-even = even
                    ans = std::max(ans, i+1 - pos[status]);
                }

            }
            return ans;
        }
    };

    Solution slo;

    EXPECT_EQ(slo.findTheLongestSubstring("leetcodeisgreat"),5);
    EXPECT_TRUE(false);
}

TEST(EASY, T205){
    class Solution {
    public:
        bool isIsomorphic(string s, string t) {
            unordered_map<char,char> mm;
            for(size_t i=0; i<s.length();i++){
                if(mm.count(s[i]) == 0 && mm.count(t[i])==0){
                    mm[s[i]] = t[i];
                    mm[t[i]] = s[i];
                }else if(mm.count(s[i]) == 1 && mm.count(t[i])==1){
                    if(!(mm[s[i]] == t[i]&&mm[t[i]] == s[i]))
                        return false;
                } else
                    return false;
            }
            return true;
        }
    };
    Solution slo;
    EXPECT_EQ(slo.isIsomorphic("egg","add"), true);
}

TEST(MID, T54){
    class Solution {
        vector<int> ans;
        int SZ1,SZ2;
    public:
        vector<int> spiralOrder(vector<vector<int>> &matrix) {
            ans.clear();
            if(matrix.empty())
                return ans;
            SZ1 = matrix.size();
            SZ2 = matrix[0].size();
            Recursion(matrix,SZ1,SZ2);
            return ans;
        }

        void Recursion(vector<vector<int>>& matrix, int sz1, int sz2){
            int start_id1 = (SZ1 - sz1)/2;
            int start_id2 = (SZ2 - sz2)/2;
            if(sz1==1){
                for(int i=start_id2;i<start_id2 + sz2;i++)
                    ans.push_back(matrix[start_id1][i]);
                return;
            } else if(sz2==1){
                for(int i=start_id1;i<start_id1 + sz1;i++)
                    ans.push_back(matrix[i][start_id2]);
                return;
            } else if(sz1==0 || sz2 ==0)
                return;

            for(int i=start_id2; i< start_id2+sz2; i++){
                ans.push_back(matrix[start_id1][i]);
            }
            for(int i=start_id1+1; i< start_id1+sz1; i++){
                ans.push_back(matrix[i][start_id2+sz2-1]);
            }
            for(int i=start_id2 + sz2 -2; i>=start_id2; i--){
                ans.push_back(matrix[start_id1+sz1-1][i]);
            }
            for(int i=start_id1+sz1-2;i>start_id1; i--){
                ans.push_back(matrix[i][start_id2]);
            }
            Recursion(matrix,sz1-2,sz2-2);
        }
    };
    Solution slo;
    vector<vector<int>> m {{1,2,3},{4,5,6},{7,8,9}};
    vector<int> a = {1,2,3,6,9,8,7,4,5};
    EXPECT_EQ(slo.spiralOrder(m),a);

    m = {{1,2,3,4},{5,6,7,8},{9,10,11,12}};
    a = {1,2,3,4,8,12,11,10,9,5,6,7};
    EXPECT_EQ(slo.spiralOrder(m),a);
}

TEST(MID, T5){
    class Solution {
    public:
        string longestPalindrome(string s) {
            vector<vector<bool>> dp(s.length(), vector<bool>(s.length()));
            string ans;

            for(size_t j=0; j< s.length();j++){
                for (int i = j; i>=0; i--) {
                    if(i==j)
                        dp[i][j] = true;
                    else if(j-i==1 && s[i]==s[j]){
                        dp[i][j] = true;
                        if(ans.length()<2)
                            ans = string(s.begin()+i, s.begin()+j+1);
                    } else if(j-i>1 && dp[i+1][j-1] && s[i]==s[j]){
                        dp[i][j]=true;
                        if(ans.length() < j-i+1)
                            ans = string(s.begin()+i, s.begin()+j+1);
                    }
                }
            }
            if(ans.empty() && !s.empty())
                ans = s.front();
            return ans;
        }
    };
    Solution slo;
    EXPECT_EQ(slo.longestPalindrome("babad"),"bab");
}

TEST(EASY, T35){
    class Solution {
    public:
        int searchInsert(vector<int> nums, int target) {
            int left =0, right = nums.size()-1;
            while (left <= right){
                int mid = (left+right)/2;
                if(nums[mid]==target)
                    return mid;
                else if(nums[mid] > target)
                    right = mid-1;
                else if(nums[mid] < target)
                    left = mid + 1;
            }
            return left;
        }
    };
    Solution slo;
    EXPECT_EQ(slo.searchInsert({1,3,5,6},5),2);
    EXPECT_EQ(slo.searchInsert({1,3,5,6},2),1);
}

TEST(MID, T1372){
    struct TreeNode {
        int val;
        TreeNode *left;
        TreeNode *right;
        TreeNode(int x) : val(x), left(NULL), right(NULL) {}
    };
    class Solution {
        int path_len_max;
        void dfs(TreeNode* root,int path_len, bool pre_is_left){
            if(pre_is_left){
                if(root->right!= nullptr)
                    dfs(root->right,path_len+1,false);
                else if(path_len_max< path_len)
                    path_len_max = path_len;
                if(root->left != nullptr)
                    dfs(root->left,1,true);
            }else{
                if(root->left!=nullptr)
                    dfs(root->left, path_len+1, true);
                else if(path_len_max< path_len)
                    path_len_max = path_len;
                if(root->right != nullptr)
                    dfs(root->right,1, false);
            }
        }
    public:
        int longestZigZag(TreeNode* root) {
            path_len_max = 0;
            dfs(root, 0, false);
            dfs(root,0, true);
            return path_len_max;
        }

    };
    EXPECT_TRUE(true);
}

TEST(EASY, T463){
    class Solution {
        int perimeter;
        int i_max,j_max;
        void dfs(vector<vector<int>>& grid, int i, int j){
            if(i<0 || j<0 || i>= i_max || j>=j_max){
                perimeter++;
                return;
            }
            if(grid[i][j]==0){
                perimeter++;
                return;
            }else if(grid[i][j] == 2)
                return;
            grid[i][j] = 2;
            dfs(grid,i-1,j);
            dfs(grid,i+1,j);
            dfs(grid,i,j+1);
            dfs(grid,i,j-1);
        }
    public:
        int islandPerimeter(vector<vector<int>>& grid) {
            perimeter = 0;
            i_max = grid.size();
            j_max = grid[0].size();
            for(size_t i=0; i< grid.size(); i++){
                for (size_t j = 0; j < grid[i].size(); ++j) {
                    if(grid[i][j] == 1){
                        dfs(grid,i,j);
                        return perimeter;
                    }
                }
            }
            return perimeter;
        }
    };

    Solution slo;
    vector<vector<int>> grid={{0,1,0,0},{1,1,1,0},{0,1,0,0},{1,1,0,0}};
    EXPECT_EQ(slo.islandPerimeter(grid),16);
}

TEST(MID, T988){
    struct TreeNode {
        int val;
        TreeNode *left;
        TreeNode *right;
        TreeNode(int x, TreeNode *_left = nullptr,TreeNode *_right=nullptr) : val(x), left(_left), right(_right) {}
    };
    class Solution {
    public:
        string smallestFromLeaf(TreeNode* root) {
            struct Path{
                Path(const Path &other,TreeNode* _node):node(_node){
                    path = other.path;
                    path.push_front(node->val + 'a');
                }
                Path() = default;
                bool operator < (const Path &rhs)const {
                    return this->path < rhs.path;
                }
                TreeNode* node;
                list<char> path;
            };
            queue<Path> path_node;
            vector<Path> valid_node;
            int path_min = INT32_MAX;
            Path root_p;
            root_p.node = root;
            root_p.path.push_back(root->val+'a');
            path_node.emplace(root_p);
            while(!path_node.empty()){
                auto temp = path_node.front();
                path_node.pop();
                if(temp.path.size() > path_min)
                    break;
                if(temp.node->left == nullptr && temp.node->right== nullptr){
                    path_min = temp.path.size();
                    valid_node.push_back(temp);
                }
                if(temp.node -> left!= nullptr)
                    path_node.emplace(temp,temp.node->left);
                if(temp.node-> right!= nullptr)
                    path_node.emplace(temp,temp.node->right);
            }

            sort(valid_node.begin(), valid_node.end());
            return string(valid_node.front().path.begin(), valid_node.front().path.end());
        }
    };

    Solution slo;
    auto*  root = new TreeNode(0,
            new TreeNode(1,new TreeNode(3), new TreeNode(4)),
            new TreeNode(2,new TreeNode(3),new TreeNode(4)));
    EXPECT_EQ(slo.smallestFromLeaf(root),"dba");
}

TEST(MID, T105){
    EXPECT_TRUE(false);
}

TEST(MID, T93){
    class Solution {
        std::vector<string> ans;
        void dfs(const string &s, string::iterator it, string &buffer, int sz){
            if(sz>4)
                return;
            if(sz == 4){
                if(it == s.end())
                    ans.emplace_back(buffer.begin(), buffer.end()-1);
                return;
            }

            if( (it+1) <= s.end()){
                auto buffer_end = buffer.end();
                string temp(it,it+1);
                temp +='.';
                buffer += temp;
                dfs(s,it+1,buffer,sz+1);
                buffer.erase(buffer_end, buffer.end());
            }

            if( (it+2) <= s.end()){
                auto buffer_end = buffer.end();
                string temp(it,it+2);
                if(stoi(temp) < 10)
                    return;
                temp +='.';
                buffer += temp;
                dfs(s,it+2,buffer,sz+1);
                buffer.erase(buffer_end, buffer.end());
            }

            if( (it+3) <= s.end() ){
                auto buffer_end = buffer.end();
                string temp(it,it+3);
                if(stoi(temp) > 255 || stoi(temp) < 100)
                    return;
                temp +='.';
                buffer += temp;
                dfs(s,it+3,buffer,sz+1);
                buffer.erase(buffer_end, buffer.end());
            }

        }
    public:
        vector<string> restoreIpAddresses(string s) {
            ans.clear();
            string buffer;
            dfs(s, s.begin(),buffer,0);
            return ans;
        }
    };

    Solution slo;

    vector<string> ans = {"255.255.11.135", "255.255.111.35"};
    EXPECT_EQ(slo.restoreIpAddresses("25525511135"), ans);
}

TEST(MID, T2){
    struct ListNode {
        int val;
        ListNode *next;

        ListNode(int x, ListNode* _next = nullptr) : val(x), next(_next) {}
    };
    class Solution {
        void Copy(ListNode* l1, ListNode* cur,bool is_add){
            while(l1!= nullptr){
                cur->next = new ListNode(l1->val);
                cur = cur->next;
                l1 = l1->next;
                if(is_add){
                    cur->val++;
                    if(cur->val == 10){
                        cur->val = 0;
                        is_add = true;
                    }else
                        is_add = false;
                }
            }
            if(is_add)
                cur->next = new ListNode(1);
        }
    public:
        ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
            bool is_add = false;
            ListNode* ans = nullptr;
            ListNode* cur = nullptr;
            while(l1 != nullptr && l2 != nullptr){
                auto v1 = l1->val, v2 = l2->val;
                auto sum = v1+v2;
                if(is_add)
                    sum++;
                if(sum>= 10){
                    is_add = true;
                    sum-=10;
                } else
                    is_add = false;
                l1 = l1->next;
                l2 = l2->next;
                if(ans == nullptr){
                    ans = new ListNode(sum);
                    cur = ans;
                    continue;
                }
                cur->next = new ListNode(sum);
                cur = cur->next;
            }
            if(l1 == nullptr && l2 == nullptr && is_add)
                cur->next = new ListNode(1);
            else if(l1 != nullptr){
                Copy(l1,cur,is_add);
            } else if(l2 != nullptr){
                Copy(l2, cur,is_add);
            }
            return ans;
        }
    };

    ListNode *pr1 = new ListNode(0), *pr2 = new ListNode(1);
    Solution slo;
    EXPECT_EQ(slo.addTwoNumbers(pr1,pr2)->val,1);
    pr1 = new ListNode(1);
    pr2 = new ListNode(9, new ListNode(9));
    auto ans = slo.addTwoNumbers(pr1,pr2);
}

TEST(MID, T287){
    class Solution {
    public:
        int findDuplicate(vector<int> nums) {
            int l=1, r = nums.size()-1;
            int ans = -1;
            while(l<=r){
                int cnt = 0;
                int mid = (r-l)/2+l;
                for(auto n:nums){
                    cnt += (n<=mid);
                }
                if(cnt <= mid){
                    l = mid + 1;
                }else{
                    ans = mid;
                    r = mid - 1;
                }
            }
            return ans;
        }
    };

    Solution slo;
    EXPECT_EQ(slo.findDuplicate({1,2,3,4,4,5,6,7}),4);
    EXPECT_EQ(slo.findDuplicate({2,3,4,4,4,5,6,7,8}),4);
}

TEST(MID, T974){
    class Solution {
    public:
        int subarraysDivByK(vector<int> A, int K) {
            std::vector<int> dp;
            dp.reserve(A.size());
            int sum=0;
            int ans = 0;
            for(auto a: A){
                sum+=a;
                ans += (sum%K==0);
                dp.push_back(sum);
            }
            for(size_t i=0; i< A.size(); i++){
                for (size_t j = 0; j < i; j++) {
                    ans+= ((dp[i]-dp[j])%K == 0);
                }
            }
            return ans;
        }
    };

    Solution slo;
    EXPECT_EQ(slo.subarraysDivByK({4,5,0,-2,-3,1},5),7);
}

TEST(EASY, T198){
    class Solution {
    public:
        int rob(vector<int> nums) {
            if (nums.empty()) {
                return 0;
            }
            int size = nums.size();
            if (size == 1) {
                return nums[0];
            }
            vector<int> dp = vector<int>(size, 0);
            dp[0] = nums[0];
            dp[1] = max(nums[0], nums[1]);
            for (int i = 2; i < size; i++) {
                dp[i] = max(dp[i - 2] + nums[i], dp[i - 1]);
            }
            return dp[size - 1];
        }
    };

    Solution slo;
    EXPECT_EQ(slo.rob({1,2,3,1}),4);
}

TEST(MID,T238){
    class Solution {
    public:
        vector<int> productExceptSelf(vector<int>& nums) {
            vector<int> ans(nums.size());
            ans[0] = 1;
            for (int i = 1; i < nums.size(); ++i) {
                ans[i] = nums[i - 1] * ans[i - 1];
            }
            int right = 1;
            for(int i= nums.size()-2; i>=0; i--){
                right = nums[i+1] * right;
                ans[i] *= right;
            }
            return ans;
        }
    };
    EXPECT_TRUE(true);
}

TEST(MID,T707){
    class MyLinkedList {
    private:
        struct Node{
            Node *pre,*next;
            int val;
            Node(int _val, Node* _pre= nullptr, Node* _next= nullptr):val(_val),pre(_pre),next(_next){};
        }*head_ = nullptr,*tail_ = nullptr;
        size_t sz_;

        Node* GetNode(int index){
            if(index<0 || index > sz_ -1)
                return nullptr;
            Node* cur_node = head_;
            for(int i=0;i<index;i++)
                cur_node = cur_node->next;
            return cur_node;
        }

        void DeleteHead(){
            if(sz_ == 0){
                head_ = nullptr;
                tail_ = nullptr;
                return;
            }
            auto temp = head_->next;
            head_ = temp;
            if(head_ != nullptr)
                head_->pre = nullptr;
            sz_--;
        }

        void DeleteTail(){
            if(sz_ == 0){
                head_ = nullptr;
                tail_ = nullptr;
                return;
            }
            auto temp = tail_->pre;
            tail_ = temp;
            if(tail_!= nullptr)
                tail_ -> next = nullptr;
            sz_--;
        }
    public:
        /** Initialize your data structure here. */
        MyLinkedList():sz_(0) {

        }

        /** Get the value of the index-th node in the linked list. If the index is invalid, return -1. */
        int get(int index) {
            auto cur_node = GetNode(index);
            if(cur_node == nullptr)
                return -1;
            else
                return cur_node->val;
        }

        /** Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list. */
        void addAtHead(int val) {
            if(head_ == nullptr){
                head_ = new Node(val);
                tail_ = head_;
            }else{
                auto temp = new Node(val, nullptr,head_);
                head_->pre = temp;
                head_ = temp;
            }
            sz_++;
        }

        /** Append a node of value val to the last element of the linked list. */
        void addAtTail(int val) {
            if(head_ == nullptr){
                head_ = new Node(val);
                tail_ = head_;
            }else{
                auto temp = new Node(val, tail_, nullptr);
                tail_->next = temp;
                tail_ = temp;
            }
            sz_++;
        }

        /** Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted. */
        void addAtIndex(int index, int val) {
            if(index > sz_)
                return;
            if(index == sz_){
                addAtTail(val);
                return;
            }
            if(index == 0){
                addAtHead(val);
                return;
            }
            auto cur_node = GetNode(index);
            auto new_node = new Node(val,cur_node->pre,cur_node);
            cur_node->pre->next = new_node;
            cur_node->pre = new_node;
            sz_++;
        }

        /** Delete the index-th node in the linked list, if the index is valid. */
        void deleteAtIndex(int index) {
            auto cur_node = GetNode(index);
            if(cur_node == nullptr)
                return;
            if(cur_node->pre == nullptr||cur_node->next == nullptr){
                if(cur_node->pre == nullptr)
                    DeleteHead();
                if(cur_node->next == nullptr)
                    DeleteTail();
            }else{
                cur_node->pre->next = cur_node-> next;
                cur_node->next->pre = cur_node ->pre;
                sz_--;
            }
            delete cur_node;
        }
    };

/**
 * Your MyLinkedList object will be instantiated and called as such:
 * MyLinkedList* obj = new MyLinkedList();
 * int param_1 = obj->get(index);
 * obj->addAtHead(val);
 * obj->addAtTail(val);
 * obj->addAtIndex(index,val);
 * obj->deleteAtIndex(index);
 */
    MyLinkedList linkedList = *new MyLinkedList();
    linkedList.addAtHead(1);
    linkedList.addAtTail(3);
    linkedList.addAtIndex(1,2);   //链表变为1-> 2-> 3
    int val = linkedList.get(1);            //返回2
    linkedList.deleteAtIndex(1);  //现在链表是1-> 3
    val = linkedList.get(1);            //返回3

    auto l2 = new MyLinkedList();
    l2->addAtHead(1);
    l2->deleteAtIndex(0);

    auto l3 = new MyLinkedList();
    l3->addAtIndex(0,10);
    l3->addAtIndex(0,20);
    l3->addAtIndex(0,30);
    val  = l3->get(0);

    auto l4 = new MyLinkedList();
    l4->addAtHead(5);
    l4->addAtHead(7);
    l4->addAtHead(0);
    l4->get(2);
    l4->get(3);
    l4->addAtIndex(1,3);
    l4->deleteAtIndex(2);
    l4->addAtHead(9);
    l4->addAtIndex(3,9);
    l4->addAtHead(4);

    EXPECT_TRUE(true);
}

TEST(HARD,T126){
    class Solution {
        std::unordered_map<string, int> id_hash_;
        std::vector<std::vector<int>> edge_;
        std::vector<string> id_word_;

        void GenEdge(const string & beginWord,const vector<string>& wordList){
            id_hash_.clear();
            edge_.clear();
            id_word_.clear();
            id_word_.reserve(wordList.size()+1);

            int id = 0;
            id_hash_[beginWord] = id++;
            id_word_.push_back(beginWord);

            for(const auto& w: wordList){
                if(id_hash_.count(w)==0){
                    id_hash_[w] = id++;
                    id_word_.push_back(w);
                }
            }


            edge_ = std::vector<std::vector<int>>(id_word_.size());

            for(int i=0; i<id_word_.size(); i++){
                for(int j=i+1; j<id_word_.size(); j++){
                    if(CheckOneDiff(id_word_[i], id_word_[j])){
                        edge_[i].push_back(j);
                        edge_[j].push_back(i);
                    }
                }
            }

        }

        bool CheckOneDiff(const string &s1,const string &s2){
            int diff_count = 0;
            for(size_t i=0; i<s1.size();i++){
                if(s1[i] != s2[i])
                    diff_count++;
                if(diff_count>1)
                    return false;
            }
            return diff_count==1;
        }

        void GenAns(vector<vector<string>> &ans,const std::vector<int>& path){
            ans.emplace_back();
            for(auto id: path){
                ans.back().push_back(id_word_[id]);
            }
        }
    public:
        vector<vector<string>> findLadders(string beginWord, string endWord, vector<string>& wordList) {
            GenEdge(beginWord,wordList);

            std::queue<std::vector<int>> qu;
            std::vector<int> cost(id_hash_.size(), INT32_MAX);
            vector<vector<string>> ans;

            if(id_hash_.count(beginWord)==0 || id_hash_.count(endWord)==0){
                return ans;
            }

            qu.push({id_hash_[beginWord]});
            int end_id = id_hash_[endWord];
            cost[id_hash_[beginWord]] = 1;

            while(!qu.empty()){
                auto path = qu.front();
                auto cur_end_id = path.back();
                if(cur_end_id == end_id){
                    GenAns(ans, path);
                }
                for(auto next_id: edge_[cur_end_id]){
                    if(cost[cur_end_id] + 1 <= cost[next_id]){
                        qu.push(path);
                        qu.back().push_back(next_id);
                        cost[next_id] = cost[cur_end_id] + 1;
                    }
                }

                qu.pop();
            }

            return ans;
        }
    };

    Solution slo;
    vector<string> wordList;
    vector<vector<string>> res;

    wordList = {"hot","dot","dog","lot","log","cog"};
    res = {{"hit","hot","dot","dog","cog"},
           {"hit","hot","lot","log","cog"}};


    EXPECT_EQ(slo.findLadders("hit", "cog",wordList),res);

}

TEST(HRAD, T297){

    struct TreeNode {
        int val;
        TreeNode *left;
        TreeNode *right;
        TreeNode(int x, TreeNode *_left = nullptr,TreeNode *_right=nullptr) : val(x), left(_left), right(_right) {}
    };
    /// code function : 1 \t 2 \t 3 \t x \t x \t 4 \t 5
    class Codec {
    public:

        // Encodes a tree to a single string.
        string serialize(TreeNode* root) {
            if(root == nullptr)
                return string();
            queue<TreeNode*> qu;
            string ans;
            ans+= to_string(root->val);
            qu.push(root);
            while(!qu.empty()){
                auto cur_node = qu.front();
                qu.pop();

                if(cur_node->left == nullptr)
                    ans += "\tx";
                else{
                    ans += ('\t' + to_string(cur_node->left->val));
                    qu.push(cur_node->left);
                }

                if(cur_node->right == nullptr)
                    ans += "\tx";
                else{
                    ans += ('\t' + to_string(cur_node->right->val));
                    qu.push(cur_node->right);
                }
            }
            while(ans.back() == '\t' || ans.back() == 'x')
                ans.pop_back();
            return ans;
        }

        // Decodes your encoded data to tree.
        TreeNode* deserialize(string data) {
            if(data.empty())
                return nullptr;
            stringstream ss(data);
            string temp;
            ss >> temp;
            auto *root = new TreeNode(stoi(temp));
            queue<TreeNode*> qu;
            qu.push(root);
            int count = 0; // if count == 2 means that the node has been visited;
            while (ss >> temp){
                auto cur_node = qu.front();
                count++;
                if(count == 1){// left
                    if(temp != "x"){
                        cur_node->left = new TreeNode(stoi(temp));
                        qu.push(cur_node->left);
                    }
                } else {
                    count = 0;
                    qu.pop();
                    if(temp != "x"){
                        cur_node->right = new TreeNode(stoi(temp));
                        qu.push(cur_node->right);
                    }
                }
            }
            return root;
        }
    };

    auto root = new TreeNode(1,new TreeNode(2), new TreeNode(3, new TreeNode(4), new TreeNode(5)));
    Codec code;
    EXPECT_EQ(code.serialize(root), string("1\t2\t3\tx\tx\t4\t5"));
    auto new_root = code.deserialize(code.serialize(root));
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
