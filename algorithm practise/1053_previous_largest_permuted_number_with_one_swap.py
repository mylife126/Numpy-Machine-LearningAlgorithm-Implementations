"""
1053. Previous Permutation With One Swap
Solved
Medium
Topics
conpanies icon
Companies
Hint
Given an array of positive integers arr (not necessarily distinct), return the lexicographically largest permutation that is smaller than arr, that can be made with exactly one swap. If it cannot be done, then return the same array.

Note that a swap exchanges the positions of two numbers arr[i] and arr[j]

Example 1:

Input: arr = [3,2,1]
Output: [3,1,2]
Explanation: Swapping 2 and 1.
Example 2:

Input: arr = [1,1,5]
Output: [1,1,5]
Explanation: This is already the smallest permutation.
Example 3:

Input: arr = [1,9,4,6,7]
Output: [1,7,4,6,9]
Explanation: Swapping 9 and 7.

"""

# from right to left, find the index where the index - 1's value is larger than the index value, that is the turning point
# then need to find the largest value in the right hand side array but it might be smaller than the turning point's value; becuase
# need to prevent the case where the right most array's largest value equal to the turning point's value such as 3 1 1 3
# then swap it 
class Solution(object):
    def prevPermOpt1(self, digits):
        turning = None
        i = len(digits) - 1
        while i >=0:
            current = digits[i]
            left = digits[i-1]
            if left > current:
                turning = i - 1
                break
            i -= 1
        
        if turning <0 or turning == None:
            return digits
        
        right_array = digits[i::]
        global_max = digits[turning]

        local_max = float("-inf")
        where = None
        for i in range(len(right_array)):
            if right_array[i] > local_max and right_array[i]<global_max:
                local_max = right_array[i]
                where = i
                

        swap_point = len(digits[0:turning+1]) + where
        digits[turning], digits[swap_point] = digits[swap_point], digits[turning]
        return digits