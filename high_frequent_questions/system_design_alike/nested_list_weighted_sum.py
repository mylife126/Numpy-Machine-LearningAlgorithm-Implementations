"""
Step 1｜题目快速解读（Problem Interpretation）

你复述的这题是经典的 Nested List Weight Sum II（反向加权）：
	•	输入：一个嵌套列表 nestedList，元素要么是整数，要么是更深的 list（list of list of …）
	•	depth：整数被包了多少层 list（例如 6 在三层里 depth=3）
	•	weight：maxDepth - depth + 1
	•	越浅权重越大，越深权重越小
	•	输出：所有整数的 value * weight 之和

等价说法（面试很爱）：

“deep integers have smaller weight; shallow integers have larger weight.”

⸻

Step 2｜核心思路讲解（Intuitive Explanation）

要算 weight = maxDepth - depth + 1，你会自然想到两步：

方法 A：两趟遍历（最直观）
	1.	先遍历一次算 maxDepth
	2.	再遍历一次，把每个整数按公式加权累加

方法 B：一趟 BFS（更巧但也直观）

利用一个关键观察：
	•	如果我们按层 BFS，从外到内逐层走：
	•	外层整数会被“累加更多次”
	•	内层整数被累加更少次
	•	具体做法：
	•	每一层把这一层的整数加到 level_sum
	•	把 level_sum 累加进 total
	•	因为 level_sum 会带着之前浅层的整数“贯穿后面每一层”，等价于给浅层更大权重

这个 BFS 解法不需要先算 maxDepth，而且代码短、好讲。

我这里给你面试最常用、最好讲的 BFS 一趟解法。
"""
from collections import deque
class Solution(object):
    def depthSumInverse(self, nestedList):
        """
        :type nestedList: List[NestedInteger]
        :rtype: int
        weight = maxDepth - depth + 1

        given [1, [4, [6]]] -> 1 sum 3 times, 4 2 times, 6 1 time

        level 1 : [1, [4, 6]]
        current_level_sum = 1
        level_sum = 1
        total = 1


        2: [4, [6]]
        current level sum = 4
        level sum = 1 + 4 = 5
        total = 1 + 1 + 4 = 6

        3: [6]
        current level = 6
        level_sum = 5 + 6 = 1 + 4 + 6 = 11
        total = 1 + 1 + 4 + level = 17

        1 3 times, 4 2 time, 6 1 time
        """

        if nestedList is None:
            return 0

        q = deque()
        for item in nestedList:
            q.append(item)

        total = 0
        level_sum = 0

        # BFS by levels
        while q:
            level_size = len(q)
            current_level_sum = 0

            # process one "level"
            for _ in range(level_size):
                item = q.popleft()

                if isinstance(item, int):
                    current_level_sum += item
                else:
                    # item is a list
                    for sub in item:
                        q.append(sub)

            # accumulate running level sum
            level_sum += current_level_sum
            total += level_sum

        return total


from collections import deque

class Solution(object):
    def depthSumInverse(self, nestedList):
        """
        nestedList: List[NestedInteger]
        return: int
        """

        if nestedList is None:
            return 0

        q = deque()
        for ni in nestedList:
            q.append(ni)

        total = 0
        level_sum = 0

        while q:
            level_size = len(q)
            current_level_sum = 0

            for _ in range(level_size):
                ni = q.popleft()

                if ni.isInteger():
                    current_level_sum += ni.getInteger()
                else:
                    sub_list = ni.getList()
                    for child in sub_list:
                        q.append(child)

            level_sum += current_level_sum
            total += level_sum

        return total

if __name__ == "__main__":
    sol = Solution()

    nestedList1 = [[1, 1], 2, [1, 1]]
    print(sol.depthSumInverse(nestedList1))  # expected 8

    nestedList2 = [1, [4, [6]]]
    print(sol.depthSumInverse(nestedList2))  # expected 17