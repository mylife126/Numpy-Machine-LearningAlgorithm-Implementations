"""
695. Max Area of Island
Solved
Medium
Topics
conpanies icon
Companies
You are given an m x n binary matrix grid. An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.

The area of an island is the number of cells with a value 1 in the island.

Return the maximum area of an island in grid. If there is no island, return 0.
"""

from collections import deque    
class Solution(object):
    @classmethod
    def bfs(self, grid, i, j, visited):
        directions = [[0,1], [0,-1], [1,0], [-1,0]]
        queue = deque()
        queue.append((i, j))

        mrows, ncols = len(grid), len(grid[0])
        area = 0 
        while queue:
            x, y = queue.popleft()
            for direction in directions:
                new_x = x + direction[0]
                new_y = y + direction[1]
                if 0 <= new_x < mrows and 0 <= new_y < ncols and grid[new_x][new_y] == 1 and (new_x, new_y) not in visited:
                    queue.append((new_x, new_y))
                    visited.add((new_x, new_y))
                    area += 1
        return area

    def maxAreaOfIsland(self, grid):
        if not grid:
            return 0
        
        mrows, ncols = len(grid), len(grid[0])
        visited = set()
        max_area = 0
        for i in range(mrows):
            for j in range(ncols):
                if grid[i][j] == 1 and (i, j) not in visited:
                    area = self.bfs(grid, i, j, visited)
                    max_area = max(max_area, area)
        return max_area