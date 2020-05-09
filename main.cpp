#include <iostream>
#include <vector>
#include <algorithm>
#include <gtest/gtest.h>

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

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
