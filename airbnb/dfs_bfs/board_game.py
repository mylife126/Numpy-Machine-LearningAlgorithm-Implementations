"""
Description

You are given a 2D board represented as a list of strings.

Each cell is encoded as: <Type><Value>
Where:
	•	Type ∈ {
                W: Water
                L: Land
                T: Tree
                M: Mud
                S: Sand
                R: Rock
                }
	•	Value is an integer score

example:
board = [
  "S0 W1 W1 W0 L2",
  "W0 W0 T0 T0 T0",
  "W0 W1 T0 M2 M1",
  "S0 L0 S1 S0 S0",
  "M0 R2 R0 S1 T0"
]
Task

👉 Find connected regions of LAND cells (L)
	•	Cells are connected 4-directionally (up/down/left/right)

⸻
Score Calculation

For each connected LAND region:
	•	Traverse the region
	•	Add up all scores from cells in that region

⸻

Output

Return the maximum score among all LAND regions

其实这一题等同于LC695 max area of island 唯一的不一样是 记分的规则 这里是string manipulation后的加和
以及确认输入的structure， 这里是 list of string 所以要把它转换成2D grid， 然后建立一个parser机制 这样如果下次感兴趣的是Tree 也可以做
"""
from collections import deque


class Solution:
    def __init__(self):
        self.directions = [
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1]
        ]

        self.max_score = 0
        self.seen = set()
        # self.board_2d = self.parser(board)

    def parser(self, board):
        """
        board: list<string>, each row of string is a row
        """

        grid = []

        for string_row in board:
            row = []
            string_list = string_row.split()

            for item in string_list:
                entity = item[0]
                entity_value = int(item[1])
                row.append((entity, entity_value))

            grid.append(row)

        return grid

    def bfs(self, start_index, grid, target):
        mrows, ncols = len(grid), len(grid[0])
        queue = deque([start_index])
        self.seen.add(start_index)
        layer_score = grid[start_index[0]][start_index[1]][1]
        while queue:
            current_index = queue.popleft()
            current_x, current_y = current_index[0], current_index[1]
            for direction in self.directions:
                next_x, next_y = current_x + direction[0], current_y + direction[1]
                if 0 <= next_x < mrows and 0 <= next_y < ncols and grid[next_x][next_y][0] == target and (next_x,
                                                                                                          next_y) not in self.seen:
                    self.seen.add((next_x, next_y))
                    its_score = grid[next_x][next_y][1]
                    layer_score += its_score

                    queue.append((next_x, next_y))

        return layer_score

    def maxLandScore(self, board, target):

        grid = self.parser(board)

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j][0] == target and (i,j) not in self.seen:
                    layer_score = self.bfs((i,j), grid, target)
                    if layer_score > self.max_score:
                        self.max_score = layer_score

        return self.max_score


if __name__ == "__main__":
    board = [
        "L1 L2",
        "L3 L4"
    ]

    board = [
        "L1 W0 L2",
        "W0 W0 W0",
        "L3 W0 L4"
    ]

    board = [
        "L1 L2 W0",
        "L3 W0 L4",
        "W0 L5 L6"
    ]
    solution = Solution()
    print(solution.maxLandScore(board, "L"))
