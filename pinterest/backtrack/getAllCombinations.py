class Solution(object):
    def combine(self, nums):
        result = []

        def dfs(start, path):
            # 每一层都可以作为一个结果
            result.append(path[:])

            for i in range(start, len(nums)):
                # choose
                path.append(nums[i])

                # explore
                dfs(i + 1, path)

                # backtrack
                path.pop()

        dfs(0, [])
        return result

