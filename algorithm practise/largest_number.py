"""
179. Largest Number
Solved
Medium
Topics
conpanies icon
Companies
Given a list of non-negative integers nums, arrange them such that they form the largest number and return it.

Since the result may be very large, so you need to return a string instead of an integer.
"""

class Solution(object):
    def largestNumber(self, nums):
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                num1 = nums[i]
                num2 = nums[j]
                if self.compare(num1, num2) == 1:
                    nums[i], nums[j] = nums[j], nums[i]
                else:
                    pass
        num_str = [str(num) for num in nums]
        if num_str[0] == "0":
            return "0"
        return "".join(num_str)
    
    def compare(self, num1, num2):
        return str(num1) + str(num2) > str(num2) + str(num1)