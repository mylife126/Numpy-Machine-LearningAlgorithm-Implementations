"""
695. Max Area of Island

You are given an m x n binary matrix grid. An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.)
You may assume all four edges of the grid are surrounded by water.

The area of an island is the number of cells with a value 1 in the island.

Return the maximum area of an island in grid. If there is no island, return 0.

Input: grid =
[[0,0,1,0,0,0,0,1,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,1,1,0,1,0,0,0,0,0,0,0,0],
 [0,1,0,0,1,1,0,0,1,0,1,0,0],
 [0,1,0,0,1,1,0,0,1,1,1,0,0],
 [0,0,0,0,0,0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,0,0,0,0,0,0,1,1,0,0,0,0]]

Output: 6
Explanation: The answer is not 11, because the island must be connected 4-directionally.

逻辑是bfs 没遇到一个1的时候 就把它当作一个layer 对他的edge 也就是上下左右做搜索， 如果是1 说明也是个岛屿，那么就++ 记录这一片岛屿的大小。

然后动态维护一个max 变量
"""
from collections import deque


class Solution:
    def __init__(self):
        self.directions = [
            [0, 1],  # right
            [0, -1],  # left
            [1, 0],  # up
            [-1, 0]  # down
        ]

        self.max_area = 0
        self.seen = set()

    def bfs(self, grid, start_index):
        # 重点！ 这里的init island 是1 而不是0
        layer_islands = 1
        queue = deque([start_index])
        mrows, ncols = len(grid), len(grid[0])

        grid[start_index[0]][start_index[1]] = 0  # mark the visited to prune
        self.seen.add(tuple(start_index))

        while queue:
            current_node = queue.popleft()

            for direction in self.directions:
                new_x, new_y = current_node[0] + direction[0], current_node[1] + direction[1]
                if 0 <= new_x < mrows and 0 <= new_y < ncols and grid[new_x][new_y] == 1 and (new_x, new_y) not in self.seen:
                    grid[new_x][new_y] = 0
                    queue.append([new_x, new_y])
                    layer_islands += 1
                    self.seen.add((new_x, new_y))
        return layer_islands

    def maxAreaOfIsland(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        if len(grid) == 0 or not grid:
            return 0

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    area_of_island = self.bfs(grid, [i, j])
                    if area_of_island > self.max_area:
                        self.max_area = area_of_island

        return self.max_area