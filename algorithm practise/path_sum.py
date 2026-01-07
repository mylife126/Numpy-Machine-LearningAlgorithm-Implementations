"""
112. Path Sum
Solved
Easy
Topics
conpanies icon
Companies
Given the root of a binary tree and an integer targetSum, return true if the tree has a root-to-leaf path such that adding up all the values along the path equals targetSum.

A leaf is a node with no children.
"""
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def hasPathSum(self, root, targetSum):
        """
        :type root: Optional[TreeNode]
        :type targetSum: int
        :rtype: bool
        """
        from collections import deque
        if not root:
            return False
        que = deque()
        que.append((root, root.val))
        while que:
            node_val = que.popleft()
            node = node_val[0]
            val = node_val[1]
            if val == targetSum and not node.left and not node.right:
                return True
            if node.left:
                left = node.left
                left_val = left.val
                new_val = val + left_val
                if new_val == targetSum and not left.left and not left.right:
                    return True
                que.append((left, new_val))
            
            if node.right:
                right = node.right
                right_val = right.val
                new_val = val + right_val
                if new_val == targetSum and not right.left and not right.right:
                    return True
                que.append((right, new_val))

        return False
