"""
56. Merge Intervals
Solved
Medium
Topics
conpanies icon
Companies
Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

 

Example 1:

Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].
Example 2:

Input: intervals = [[1,4],[4,5]]
Output: [[1,5]]
Explanation: Intervals [1,4] and [4,5] are considered overlapping.
Example 3:

Input: intervals = [[4,7],[1,4]]
Output: [[1,7]]
Explanation: Intervals [1,4] and [4,7] are considered overlapping.
"""

class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        if len(intervals) == 1:
            return intervals

        intervals = sorted(intervals, key=lambda x:x[0])

        n = len(intervals)
        merged = [intervals[0]]

        for idx in range(len(intervals)):
            current = intervals[idx]
            last_interval = merged[-1]
            last_interval_right = last_interval[1]
            current_left = current[0]
            if current_left <= last_interval_right:
                last_interval[1] = max(last_interval[1], current[1])
            else:
                merged.append(current)

        return merged





        