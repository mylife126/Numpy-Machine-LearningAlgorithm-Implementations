"""
nums = [a b c d e]

我们看prefix sum
sum(0) = a
sum(1) = a + b
sum(2) = a + b + c
sum(3) = a + b + c + d

那么我想知道nums[2] to nums[3]的sum该怎么做
可以直接
sum3 - sum2 = a + b + c + d - （a + b + c） = d

那么假设d == k， 则 sum（3） - k = sum（2） 对吧，
所以上述表达式意思是 当我们prefix sum到了index 3的时候 是否存在一个过往index i， 使得 array from i+1到3这个区间等于K。

那么我们只需要用dictionary去记录每一次prefix出现的次数， 只要sum（j） - k存在于dictionary 则说明存在n次一个区间的sum 等于k

"""
from collections import defaultdict


class Solution(object):
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """

        counter = defaultdict(int)
        counter[0] = 1
        prefix_sum = 0

        res = 0

        for num in nums:
            prefix_sum += num

            if (prefix_sum - k) in counter:
                res += counter[(prefix_sum - k)]

            counter[prefix_sum] += 1

        return res