"""
1091. Shortest Path in Binary Matrix
Solved
Medium
Topics
conpanies icon
Companies
Hint
Given an n x n binary matrix grid, return the length of the shortest clear path in the matrix. If there is no clear path, return -1.

A clear path in a binary matrix is a path from the top-left cell (i.e., (0, 0)) to the bottom-right cell (i.e., (n - 1, n - 1)) such that:

All the visited cells of the path are 0.
All the adjacent cells of the path are 8-directionally connected (i.e., they are different and they share an edge or a corner).
The length of a clear path is the number of visited cells of this path.
"""

from collections import deque
class Solution:
    def shortestPathBinaryMatrix(self, grid):
        if grid[0][0] != 0:
            return -1
        
        if grid[-1][-1] != 0:
            return -1

        queue = deque()
        queue.append((0,0,1)) # the starting point is 0,0, and the step count is 1
        directions = [[0,1], [0,-1],
                       [1,0], [-1,0],
                       [1,1], [1,-1],
                       [-1,1], [-1,-1]
                       ]
        visited=set()
        true_path = set()
        mrows, ncols = len(grid), len(grid[0])
        while queue:
            i, j, step = queue.popleft()
            if i==mrows-1 and j==ncols-1:
                true_path.add((mrows-1, ncols-1))
                print(true_path)
                return step
            for direction in directions:
                newi, newj = i+direction[0], j+direction[1]
                if mrows>newi>=0 and ncols>newj>=0 and grid[newi][newj]==0 and (newi, newj) not in visited:
                    # 只有当ij此刻是可以往下走的说明ij是valid的path， 那么添加至set里
                    # 为什么用set， 是因为当ij此刻有多个可以走的路线的时候，它都是valid 就会被重复添加
                    # 那么当此刻的ij所在的node是无法走下去的 则意味着它可能走的下一条路在别的node里已经是被访问过的最优解
                    # 所以它也不是valid path
                    true_path.add((i, j))
                    step_new = step + 1 
                    queue.append((newi, newj, step_new))
                    visited.add((newi, newj))
        print(true_path, step)
        return -1   