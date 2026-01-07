"""
223. Rectangle Area
Solved
Medium
Topics
conpanies icon
Companies
Given the coordinates of two rectilinear rectangles in a 2D plane, return the total area covered by the two rectangles.

The first rectangle is defined by its bottom-left corner (ax1, ay1) and its top-right corner (ax2, ay2).

The second rectangle is defined by its bottom-left corner (bx1, by1) and its top-right corner (bx2, by2).
"""

class Solution(object):
    def computeArea(self, ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
        """
        :type ax1: int
        :type ay1: int
        :type ax2: int
        :type ay2: int
        :type bx1: int
        :type by1: int
        :type bx2: int
        :type by2: int
        :rtype: int
        """
        # so get the left most , right min, bottom max, and top min

        left_most = max(ax1, bx1)
        right_min = min(ax2, bx2)
        bottom_most = max(ay1, by1)
        top_min = min(ay2, by2)

        if right_min > left_most and top_min > bottom_most:
            inner_area = (right_min - left_most) * (top_min - bottom_most)
        else:
            inner_area = 0

        rec1 = (ax2 - ax1) * (ay2 - ay1)
        rec2 = (bx2 - bx1) * (by2 - by1)

        total_area = rec1+ rec2 - inner_area
        return total_area