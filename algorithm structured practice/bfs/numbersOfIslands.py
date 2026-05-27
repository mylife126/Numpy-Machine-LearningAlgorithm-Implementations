"""
逻辑是， 每一个node 都可以bfs去上下左右去展开， 那么每一个为1的岛屿就是一个合法的node，则对此开始做bfs搜索，
[1, 1, 0]
[1, 1, 0]
[0, 1, 0]

上例，（0，0）的位置是一个合法node，加入queue， 它开始bfs， 可以向左向下走加入queue， 这个时候 while loop没有结束，因为queue里面有下一个新的合法
nodes 为（0 1） 和 （1 0） 对他们继续bfs

一次bfs结束则意味着找完了一片islands results++
"""

from collections import deque
class Solution(object):
    def __init__(self):
        self.directions = [
            [-1, 0],
            [1, 0],
            [0, -1],
            [0, 1]
        ]

    def bfs(self, grid, i, j):
        mrows, ncols = len(grid), len(grid[0])
        queue = deque()
        queue.append((i,j))
        # prune
        grid[i][j] = "0"

        while queue:
            x, y = queue.popleft()
            for d in self.directions:
                dx, dy = d[0], d[1]
                newx, newy = x + dx, y+dy
                if 0<=newx<mrows and 0<=newy<ncols and grid[newx][newy] == "1":
                    queue.append((newx, newy))
                    # visited pruning
                    grid[newx][newy] = "0"


    def numIslands(self, grid):
        if not grid or len(grid) == 0:
            return 0

        mrows, ncols = len(grid), len(grid[0])
        res = 0
        for i in range(mrows):
            for j in range(ncols):
                if grid[i][j] == "1":
                    self.bfs(grid, i, j)
                    res += 1
        return res