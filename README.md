# leetcode_study

## 异或的一些特性(T136)
1. 满足交换律 a^b^c <=> c^b^a 
2. 任何数字与0异或为本身 a^0 <=> a
3. 相同数字异或为0  a^a <=> 0  
利用异或可用于判断不重复数字（详见T136）

## 求取最短路径一般采用广度优先搜索（T126）
1. 采用一个cost数组用于存储起点到该点的路径长度，以广度优先搜索搜素最短路径
