"""
1539. Kth Missing Positive Number
Solved
Easy
Topics
conpanies icon
Companies
Hint
Given an array arr of positive integers sorted in a strictly increasing order, and an integer k.

Return the kth positive integer that is missing from this array.
"""

class Solution(object):
    def findKthPositive(self, arr, k):
        """
        arr: List[int]
        k: int
        return: int
        """
        missing_count = 0
        current_num = 1

        arr_set = set(arr)

        while missing_count != k:
            if current_num not in arr_set:
                missing_count += 1
            current_num += 1

        return current_num - 1
        
            

