"""
Brutal Force
双循环遍历，以每一个为起始点然后往后不断看乘几是否满足边界
"""

class Solution(object):
    def numSubarrayProductLessThanK(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        if k == 0:
            return 0

        res = 0

        for starting_index in range(len(nums)):
            temp = 1
            for ending_index in range(starting_index, len(nums)):
                temp *= nums[ending_index]

                if temp < k:
                    res += 1
                
                else:
                    break
        return res


"""
更好的方法是滑动窗口。
【10， 5， 2， 6】
想象一开始的窗口只有向右扩张， 那么 【10】 - 【10 5】 - 【10 5 2】
              multiplication    10        50         100
                                                    此刻很明显不能满足需求了， 说明这个区间已经超了， 我们得减少一个element，然而我们是单调向右的，所以1 pass已经见过右边的， 那么删掉2没有意义，则删掉100， window 从5开始
                                                            【5 2】 - 【5 2 6】

逻辑很简单 只有当窗口缩小才有用，而我们是向右扩张的，所以当超了的时候，只需要左边界往右走 直到这个window里的乘机小于k。 这个单调性保证了：一旦 [left, right] 的乘积 < k，那么 [left+1, right]、[left+2, right]... 也一定 < k。不需要回头检查。

现在需要知道给定一个窗口 有多少个subwindow都满足需求，
例如【5 2 6】， 我们有 6， 62， 625， 是以数组的right的结尾的所有合法子集 等于 3 - 1 + 1

"""
class Solution(object):
    def numSubarrayProductLessThanK(self, nums, k):
class Solution(object):
    def numSubarrayProductLessThanK(self, nums, k):
        if k <= 1:
            return 0

        left_index = 0 
        product = 1
        res = 0

        for right_index in range(len(nums)):
            product *= nums[right_index]

            # if the total prodoct in this window is beyond the k, then we need to shrink the window
            while product > k:
                # discard the element at the the current left until the window's product meets the requirement
                product //= nums[left_index]

                # then shrink the window size to test if the product after removing the left number meets the requirement or no
                left_index += 1

            # otherwise this window is a valid one, such as [10], [10 5]
            res += right_index - left_index + 1
        
        return res
