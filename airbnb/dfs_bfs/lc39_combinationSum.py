"""
Medium（Airbnb高频）

⸻

Description

You are given a list of menu items, where each item has a price (floating-point number).

You are also given a target amount of money.

⸻

Task

Find all combinations of menu items such that:
	•	The total price equals the target amount
	•	Each item can be used multiple times
	•	The result should not contain duplicate combinations

⸻

Output

Return all valid combinations of menu items.

Each combination should be:
non-decreasing order（避免重复）

Example 1

Input:
prices = [2.5, 3.0, 4.0]
target = 7.5
Output:
[
 [2.5, 2.5, 2.5],
 [2.5, 5.0],
 [3.0, 4.5] ❌（不存在）
]

核心逻辑：
这道题是要找所有组合，则用dfs， 剪枝叶： 如果当前 sum > target → stop

那么dfs只做两个事情，退出条件：
1. sum 等于 target了， result 记录正确答案
2. sum 超过了 target 。 直接退出

退出后pop上一个state， 因为你要继续往下尝试
"""


class Solution(object):
    def __init__(self):
        self.results = []

    def dfs(self, start_index, current_path, current_sum, target, candidates):
        if current_sum == target:
            # 重点！⚠️！ 因为backtrack里 current path 在python里不是一个固定的pointer 你必须copy 不然返回值为空
            self.results.append(current_path[::])
            return

        if current_sum > target:
            return

        # try each candidate
        for i in range(start_index, len(candidates)):
            current_number = candidates[i]
            if current_sum + current_number > target:
                break

            current_path.append(current_number)
            self.dfs(i, current_path, current_sum + current_number, target, candidates)

            # backtrack to pop out the not working number from the last state
            current_path.pop()

    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        candidates.sort()
        self.results = []

        if len(candidates) == 0 or not candidates:
            return []

        self.dfs(0, [], 0, target, candidates)
        return self.results

