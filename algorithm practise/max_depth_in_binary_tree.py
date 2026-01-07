"""
104. Maximum Depth of Binary Tree
Solved
Easy
Topics
conpanies icon
Companies
Given the root of a binary tree, return its maximum depth.

A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
"""

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
from collections import deque
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: int
        """
        if not root:
            return 0 
            
        que = deque()
        que.append((root, 1))
        parent_seen = set()
        result = None
        while que:
            node_step = que.popleft()
            node = node_step[0]
            step = node_step[1]
            result = step

            if node.left:
                que.append((node.left, step + 1))
            if node.right:
                que.append((node.right, step+1))
        
        return result