from collections import deque

class Solution(object):
    def leftSideView(self, root):
        """
        BFS approach for LEFT side view:
        Record the first node at each level.
        """
        if not root:
            return []

        result = []
        queue = deque([root])

        while queue:
            level_size = len(queue)
            for i in range(level_size):
                node = queue.popleft()

                # if this is the first node in this level → leftmost
                if i == 0:
                    result.append(node.val)

                # enqueue children (left first)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

        return result