from collections import deque


class Solution(object):
    def __init__(self):
        self.directions = [
            (1, 0),  # down
            (-1, 0),  # up
            (0, 1),  # right
            (0, -1)  # left
        ]

    def shortestPath(self, grid, k):
        """
        :type grid: List[List[int]]
        :type k: int
        :rtype: int
        """
        if not grid or len(grid) == 0:
            return -1

        mrows, ncols = len(grid), len(grid[0])

        # x, y, #remaining obstacles can be removed
        seen = set()
        seen.add((0, 0, k))
        queue = deque()

        # x, y, remaining_k, steps
        queue.append((0, 0, k, 0))

        while queue:
            x, y, remaining_k, steps = queue.popleft()

            if (x, y) == (mrows - 1, ncols - 1):
                return steps

            for direction in self.directions:
                nx, ny = x + direction[0], y + direction[1]

                # if it is a valid move
                if 0 <= nx < mrows and 0 <= ny < ncols:

                    # if it is empty land and not reached previously
                    if grid[nx][ny] == 0 and (nx, ny, remaining_k) not in seen:
                        queue.append((nx, ny, remaining_k, steps + 1))
                        seen.add((nx, ny, remaining_k))

                    # if it is an obstacle, and we can still remove it
                    # 重点 ⚠️！ check seen的时候是check remaining k - 1的状态 而不是 nx ny remianingk， 这样会导致无法真的去重复！
                    elif grid[nx][ny] == 1 and remaining_k > 0 and (nx, ny, remaining_k - 1) not in seen:
                        # 注意啊 这里不能直接remaining k -=1 因为这个variable时四个方向共享的！

                        queue.append((nx, ny, remaining_k - 1, steps + 1))

                        seen.add((nx, ny, remaining_k - 1))

        return -1