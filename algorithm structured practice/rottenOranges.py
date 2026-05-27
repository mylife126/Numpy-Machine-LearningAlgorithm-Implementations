"""
类似于number of islands 但是区别在于， 每一次bfs展开是 同一层里的所有 rotten orange一起做。 这一批会在同一个时间点里去污染
上下左右的orange， 然后新的一批加进去queue 在下一个时间点同时处理。

所以queue的展开 需要控制边界 也就是说
while queue 下一层需要一个 for loop 一批一批处理每一层layer
    for orange in all_rotten_from_last_layers

如果不加这一层逻辑， 会变成每一个orange 腐蚀一次就时间++， 但是这不是合理的时间模型，因为临近的orange是可以同时一起腐败
例如
[
    [2, 2]
    [2, 1]
]
这个1 是可以同时被2 2 2 一起腐败 或者说他们是同一批次

其次还有一个edge case是， 无法腐败的问题
[
    [2, 2, 1]
    [2, 2, 0]
    [1, 0, 1]
]
可以看到 右下角的1 是不会被腐败的 所以要做一个检测， 检测一开始有多少个好orange 最后看多少个被腐败了。 都腐败了则成功

"""

from collections import deque


class Solution(object):
    def __init__(self):
        self.directions = [
            (-1, 0),  # up
            (1, 0),  # down
            (0, 1),  # right
            (0, -1)  # left
        ]

    def orangesRotting(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        if not grid or len(grid) == 0:
            return -1

        nums_health = 0
        mrows, ncols = len(grid), len(grid[0])

        # record the healthy orange being rotten, x, y
        # seen = set()

        # queue holds the rotten ones
        queue = deque()

        for x in range(mrows):
            for y in range(ncols):
                this_orange = grid[x][y]
                if this_orange == 1:
                    nums_health += 1
                elif this_orange == 2:
                    queue.append((x, y))
                else:
                    continue

        if nums_health == 0:
            return 0

        # 在每层所有rotten orange level， 每展开一次bfs 就是一次rotten as a whole
        time = 0
        while (queue):
            performed_rotting = False

            for _ in range(len(queue)):
                rotten_x, rotten_y = queue.popleft()

                for dx, dy in self.directions:
                    nx, ny = rotten_x + dx, rotten_y + dy
                    if 0 <= nx < mrows and 0 <= ny < ncols and grid[nx][ny] == 1:
                        # seen.add((nx, ny))
                        grid[nx][ny] = 2
                        queue.append((nx, ny))
                        performed_rotting = True
                        nums_health -= 1

            if performed_rotting:
                time += 1

        return time if nums_health == 0 else -1



