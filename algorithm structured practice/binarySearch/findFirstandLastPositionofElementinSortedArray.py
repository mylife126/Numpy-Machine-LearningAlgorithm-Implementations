class Solution(object):
    def binarySearch(self, nums, target, findLeft):
        left = 0 
        right = len(nums) - 1
        boundary_index = -1
        while left <= right:
            mid = left + (right - left) // 2 # avoid overflow
            if nums[mid] == target:
                boundary_index = mid

                # if you are trying to find the left most boundary, then, you assume there is another same element in the left
                # thus shrink the right bound instead
                if findLeft:
                    right = mid - 1
            
                else:
                    # then you are trying to find the right most boundary, then, you assume there is another same element in the right
                    # thus shrink the left bound instead
                    left = mid + 1
                
            elif nums[mid] < target:
                # means that left bound is too far
                left = mid + 1
            else:
                right = mid - 1
        
        return boundary_index

    def searchRange(self, nums, target):
        """
        Return the first and last position of target in sorted nums.
        """

        # nums_set = set(nums)
        # if target not in nums_set:
        #     return [-1, -1]

        # first let us try to find the left most index
        left_most = self.binarySearch(nums, target, True)
        right_most = self.binarySearch(nums, target, False)

        return [left_most, right_most]