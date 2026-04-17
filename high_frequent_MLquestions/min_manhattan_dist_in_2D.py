"""
compared to 1D case, where the math justification is
if a < b < c, then d(a_i, a_j) = |a_i - a_j| , 那么
    •	|b - a| ≤ |b - c|
	•	|c - b| ≤ |c - a|

但是对于一个 二位坐标点
｜
｜  （x, y)
｜             (x', y')
｜
｜———————————————————

🧩 问题定义（2D Manhattan 最近点）

给定 N 个点 (x_i, y_i)，
对于每个点 p_i，找出它到所有其他点的 Manhattan 距离 中的最小值。

即：

\text{res}[i] = \min (i != j) (|x_i - x_j| + |y_i - y_j|)

所以最直接的方法就是对每一个index的数组对其余的数组做 L1 distance calculation 。O(N2)


2D 的情况是， L1 距离成为：
d_M(a,b) = |x_a - x_b| + |y_a - y_b|
if (x₁ ≥ x₂, y₁ ≥ y₂) -> (x₁ - x₂) + (y₁ - y₂) 东北方向 = （x1 + y1） - (x2+ y2)
if (x₁ ≥ x₂, y₁ ≤ y₂) -> (x₁ - x₂) - (y₁ - y₂) 东南方向 = (x1 - y1) -(x2 + y2)
if (x₁ ≤ x₂, y₁ ≥ y₂) -> -(x₁ - x₂) + (y₁ - y₂) 西北方向 = (-x1 + y1) - (x2 + y2)
if (x₁ ≤ x₂, y₁ ≤ y₂ -> -(x₁ - x₂) - (y₁ - y₂) 西南方向 = (-x1 - y1) - (x2 + y2)

可以见的 通过转换后， 其实2D坐标里面的dist 是 x 和 y的投影的差和
🔸 曼哈顿距离的变形

|x_1 - x_2| + |y_1 - y_2| = max((x_1 + y_1) - (x_2 + y_2),
                                (x_1 - y_1) - (x_2 - y_2),
                                (-x_1 + y_1) - (-x_2 + y_2),
                                (-x_1 - y_1) - (-x_2 - y_2))

而这些x 和 y的投影差和可以统一为四个方向即为：
x + y, x- y, -x +y, -x - y

那么我们可以把每一个2D point 都分别在四个方向上做转换， 将其变成一个1D的case， 然后再排序近邻的问题，就等同于1D的case了
等同于一个2D数组会变成4个数组，

假设如下数组
0 0
0 2
2 0
2 2
5 5

-> 在 x+y的方向上 [0, (0, 0,)], [2, [0, 2]], [2, [2,0]], [4, [2,2,]], [10, [5,5]]
然后在这个方向上做排序得到近邻，然后再做点和点的距离计算 更新距离array
然后对四个方向做同样的事情就可

"""

import numpy as np


def manhattan_dist_in_2D(X, Y):
    return abs(X[0] - Y[0]) + abs(X[1] - Y[1])


def brutal_manhattan_dist_in_2D(nums):
    """
    nums is a 2D array, each element is a 2D coordinates
    """
    each_min_dist = [float('inf')] * len(nums)

    for i in range(len(nums)):
        X = nums[i]
        for j in range(len(nums)):
            if i != j:
                Y = nums[j]
                dist = manhattan_dist_in_2D(X, Y)
                its_last_min = each_min_dist[i]
                if dist < its_last_min:
                    each_min_dist[i] = dist

    return each_min_dist


def sorted_manhattan_dist_in_2D(nums):
    """
    Optimized O(n log n) method based on 4-directional projections.
    Transform each 2D point (x,y) into 1D values under:
        x+y, x−y, −x+y, −x−y
    Then in each direction, only adjacent points in sorted order
    can possibly have the minimum Manhattan distance.
    """
    directions = [
        lambda point: point[0] + point[1],
        lambda point: point[0] - point[1],
        lambda point: -point[0] + point[1],
        lambda point: -point[0] - point[1],
    ]
    mins_distance = [float('inf')] * len(nums)  # index at each original point

    for direction in directions:
        this_direction_point = []
        for index, point in enumerate(nums):
            this_direction_point.append([direction(point), index, point])
        this_direction_point = sorted(this_direction_point, key=lambda point: point[0])

        # step 2 compare in this direction
        for i in range(1, len(this_direction_point)):
            point_at_right = this_direction_point[i - 1]
            pivot = this_direction_point[i]

            _, index_1, coordinate_1 = point_at_right[0], point_at_right[1], point_at_right[2]
            _, index_2, coordinate_2 = pivot[0], pivot[1], pivot[2]

            mins_distance[index_1] = min(mins_distance[index_1], manhattan_dist_in_2D(coordinate_1, coordinate_2))
            mins_distance[index_2] = min(mins_distance[index_2], manhattan_dist_in_2D(coordinate_1, coordinate_2))

    return mins_distance

# ==================== TEST ====================
if __name__ == "__main__":
    pts = np.array([
        [0, 0],
        [0, 2],
        [2, 0],
        [2, 2],
        [5, 5]
    ])

    print("Input points:\n", pts)
    result = minimal_manhattan_distance_2D(pts)
    print("Minimal Manhattan distance for each point:\n", result)