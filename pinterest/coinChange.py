"""
还是一个逻辑和bus routes 很像， 你的出发是amount， target是0， 这部分是隐形约束。

那么你有不同的stop可以用那就是coins里面的价格
coins = [1,2,5], amount = 11
例如第一层， 11， 此刻用了一个银币
我可以走1， 或者2， 或者5 分别到达 10， 9， 6 进去queue

那么第二层 开始bfs 10， 9， 6， 此刻再用一个银币
    10可以走1， 2， 5 -》 9， 8， 5 去重得到 8 ， 5
    9还是可以走1， 2 ，5 -》 8， 7， 4 去重复得到 7， 4
    6还是可以走1， 2， 5 -〉 5， 4， 1 去重得到 1

这一层queue 里面只有【8 5 7 4 1】
    可以看到 visited node记录的是每一个价值是否在前一步就已经拿到了， 例如你想要拿到4块钱这个节点，
    其实有两个path

                                            11
                                10           9              6
                        1   2    5     8  7 4         5   4     1

    也就是11 - 9 - 4 或者11 - 6 - 4
    但是我们在11 9 4 就能reach，所以其实不需要重复计算 11 6 4

现在到了第三层bfs， 又可以用一个银币
我们发觉当循环到1这个节点的时候，1 - 1 直接能到0 那说明我们找到了 则此刻用了3个银币

本质上，这就是一个 BFS：每一个金额是一个 node，每次减去一个 coin 是一条边，
我们在找从 amount 到 0 的最短路径。

因为 BFS 是按层扩展的，每一层表示使用了相同数量的 coin，
所以第一次到达 0 时，就是使用 coin 数量最少的解。

"""
from collections import deque


class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        # edge case if the amount is already 0
        if amount == 0:
            return 0

        queue = deque()

        # add the starting node to the queue
        queue.append(amount)

        # get the visited node into memo to prune repeated path
        visited = set()
        visited.add(amount)

        coins_needed = 0

        # bfs to expand on each current value layer and to use all possible edge to find the next value node
        while (queue):
            coins_needed += 1
            # 其实这里就是bfs里的在一层里把每一个边走一遍，参考numbers of island， 他的循环是上下左右
            for _ in range(len(queue)):
                current_value = queue.popleft()

                for coin in coins:
                    next_value = current_value - coin
                    if next_value in visited:
                        continue

                    if next_value == 0:
                        return coins_needed

                    # 重点 如果不佳这个if会有一个 edge case coins =【2】， amount = 3
                    # 你会发觉第一次bfs后，你到达了1， 1  - 2 = -1， 其实是不合法的
                    # 但如果没有这个边界条件，会无限循环negative下去
                    if next_value > 0:
                        queue.append(next_value)
                        visited.add(next_value)

        return -1
