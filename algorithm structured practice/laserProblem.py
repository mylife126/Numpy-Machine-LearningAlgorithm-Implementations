"""
输入：
        •        一个矩阵 grid：
        •        每个格子可能是：
        •        '.'：空白，激光照过去不变方向；
        •        '/'：45°镜子，会改变激光方向；
        •        '\\'：135°镜子，会改变激光方向；
        •        激光初始从 (sx, sy) 出发，方向为："up", "down", "left", "right"；
        •        边界都是墙（不能出界）；
        •        给定一个目标 (tx, ty)；
        •        问：激光是否最终可以抵达目标格子？

[
    【., ., .],
    [., /, .],
    [., \, .]  <-----
]



bfs 来记录每一个status下激光的下一个state

制定rule, 规定的是输入方向和输出方向：
for "/"
up -> right
down -> left
left -> down
right -> up

similarly, for "\\"

up -> left
down -> right
right -> down
left -> up


then 制定 方向
up : (1, 0),
down : (-1, 0),
left : (0, -1),
right : (0, 1),


加入visited 防止循环
[
   。 ['/', '.'], <----
   。 ['.', '\\']
]
假设我们从 0 2 向左出发会一直在圈出来的小圈圈里无限循环， 而到达不了1， 0 所以得加入visited确保不会一直循环

"""

from collections import deque

class LaserProblem:
    def __init__(self, grid):
        self.grid = grid
        self.directions = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }

    def reflection(self, direction, cell):
        """
        given the current laser direction and the cell style, decide the next moving direction
        """
        if cell == ".":
            return direction

        if cell == "/":
            return {
                "up": "right",
                "down": "left",
                "left": "down",
                "right": "up",
            }[direction]

        if cell == "\\":
            return {
                "up": "left",
                "down": "right",
                "left": "up",
                "right": "down",
            }[direction]

        raise ValueError(f"Invalid cell: {cell}")

    def solve(self, start, direction, end):
        m, n = len(self.grid), len(self.grid[0])
        queue = deque([(start[0], start[1], direction)])
        visited = set()

        while queue:
            x, y, d = queue.popleft()
            print(x,y,d)

            if (x, y, d) in visited:
                continue
            visited.add((x, y, d))

            if [x, y] == end:
                return True

            dx, dy = self.directions[d]
            nx, ny = x + dx, y + dy

            if not (0 <= nx < m and 0 <= ny < n):
                continue

            cell = self.grid[nx][ny]
            if cell == "#":
                continue

            new_d = self.reflection(d, cell)
            queue.append((nx, ny, new_d))

        return False

if __name__ == "__main__":
    grid = """
    [
        [., ., #],
        [., /, .],
        [., \, .] 
    ]
    """

    a = [['.', '.', '#'],
         ['.', '/', '.'],
         ['.', '\\', '.']]

    solution = LaserProblem(a)
    print(solution.solve([2,2], "left", [1,2]))
