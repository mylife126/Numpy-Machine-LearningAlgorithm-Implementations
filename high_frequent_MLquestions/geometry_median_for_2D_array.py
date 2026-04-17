'''
给定 N 个二维点：
P = \{(x_1, y_1), (x_2, y_2), \dots, (x_N, y_N)\}

求一个点 Q = (x, y)，
使得以下式子最小：

f(x, y) = \sum_{i=1}^{N} \sqrt{(x - x_i)^2 + (y - y_i)^2}

这里不能用kmeans的理由是 kmeans之所以成立是因为它计算的优化的是 平方项，也就是 mathematical distance, thus, at M step, we can use

⚙️ 二、K-Means 的更新方式依赖“平方项”

K-Means 的更新步骤是：
c_{\text{new}} = \frac{1}{N} \sum_i x_i

为什么这么简单？
因为对平方距离求导非常容易：

\frac{\partial}{\partial c} \sum_i (x_i - c)^2 = -2 \sum_i (x_i - c)

令其为 0 可得：
c = \frac{1}{N} \sum_i x_i  -> CLOSED Form solution , because the center is independent of itself !!

但是这里是geometric 是sqrt（L2norm（x， y））它的partial derivative is none linear , Thus if do the derivative
对 c 求导，不再是线性的：
\frac{\partial J}{\partial c} = sum(x_i) / {norm2(c)} -> now the C is dependent of c .

所以要计算空间里一组2D 向量里一个点，使得这个点距离每一个点都最近 没法用kmeans来求解。

SOLUTION Weiszfeld’s Algorithm to iterate:

C_ next = sum( x_i /  norm2 (x_i , c_current)) / (1 / sum( norm2 (x_i , c_current)))
c_next == sum (w_i * x_i}/ sum {w_i}, w_i = x_i / norm2(x_i, c_current)

所以操作就是对每一个点进行距离当前c的距离，他们的权重为 1/ d， 逻辑是距离每个点根据它离当前点的距离
越近的点权重越大，越远的点权重越小。
然后计算加权平均。

一直到cnew 不再更新， 那么数学表达为 norm(c_new, c_old) <= eps
'''

import numpy as np


def norm2(x, y):
    """
    x, y are both numpy array
    x: m by n
    y: n,
    """
    d = (x - y) ** 2
    d = np.sum(d, axis=1)  # m sample's norm2'2
    return np.sqrt(d)


def get_closest_point_in_2D_for_an_array(array, stop_eps=1e-5, iteration=1000):
    c_current = np.mean(array, axis=0)  # array : m, n, -> (n,1)

    for i in range(iteration):
        distances = norm2(array, c_current)  # m,

        # avoid division by zero (current exactly equals one of the points)
        if np.any(distances == 0):
            return c_current

        weights = 1 / distances  # m,
        weights_ = weights[:, np.newaxis]  # change the dim to be m by 1

        weighted_points = array * weights_  # m by n * m by 1 = m by n
        weighted_points = np.sum(weighted_points, axis=0)  # n by 1

        factor = np.sum(weights)  # scalar

        c_new = weighted_points / factor  # n by 1

        # decide if the c_new has converged
        if np.linalg.norm(c_new - c_current) < stop_eps:
            break
        c_current = c_new

    return c_current


if __name__ == '__main__':
    points = np.random.rand(100, 2)
    points = np.array([
        [0, 0],
        [0, 2],
        [2, 0],
        [2, 2],
        [5, 5]
    ])

    geometric_points = get_closest_point_in_2D_for_an_array(points)
