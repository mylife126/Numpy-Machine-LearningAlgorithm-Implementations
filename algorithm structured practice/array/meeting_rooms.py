"""
类似于merge intervals
"""

class Solution(object):
    def canAttendMeetings(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: bool
        """
        if not intervals or len(intervals) == 0:
            return True

        intervals = sorted(intervals, key=lambda x:x[0])

        res = True

        current = intervals[0]
        for i in range(1, len(intervals)):
            next_meeting = intervals[i]
            current_end = current[1]
            next_start = next_meeting[0]
            if next_start < current_end:
                return False

            current = next_meeting
        return res
