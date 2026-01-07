from collections import deque

class Solution(object):
    def rightSideView(self, root):
        """
        BFS approach:
        For each level of the tree, record the rightmost node.
        """
        if not root:
            return []

        result = []
        queue = deque([root])  # queue stores nodes of the current level

        while queue:
            level_size = len(queue)   # number of nodes in this level
            for i in range(level_size):
                node = queue.popleft()

                # enqueue children for next level
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

                # if it's the last node of this level, record it
                if i == level_size - 1:
                    result.append(node.val)

        return result