# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

"""
        1
       / \
      2   3
     /
    4

del 2:

           1
            \
      None   3

    4

return 1 3, 4

1 -> append
  - 1.left =
        2 isroot = False but deleted = True child becomes root
            - left 4 , 4 not in deleted, and is root -> append
                - left, not node return None
            to 4, 4 not deleted return 4 as the node
            - right 5, not in deleted, and is root -> append
                - right, not node, return None
            to 5, 5 not deleted return 5 as the node
        to 2, but 2 deleted return none
    1.left = None
  - 1. right repeat recursion

"""


class Solution(object):
    def __init__(self):
        self.result = []

    def dfs(self, node, is_root, to_be_deleted_set):
        # first, the end of the lead return condition, where
        # neither left or right exists, the end
        if not node:
            return None

        # test if this node needs to be deleted
        deleted = node.val in to_be_deleted_set
        # print(deleted)

        # if it is not deleted and it is also a root, then it can be added
        if is_root and not deleted:
            self.result.append(node)

        # then if it is not deleted, we need to recursively see if its children fall under either of the rule
        # if deleted is False, then child is not root either
        # if deleted if True, then child becomes root
        children_to_be_root = deleted
        node.left = self.dfs(node.left, children_to_be_root, to_be_deleted_set)
        node.right = self.dfs(node.right, children_to_be_root, to_be_deleted_set)

        # finally if it is deleted, then it is none 这一步确保了如果
        # 母节点不删，但是子节点删了，那么它的该子节点为None，或者保持原样
        if deleted:
            return None
        else:
            return node

    def delNodes(self, root, to_delete):
        to_be_deleted_set = set(to_delete)

        self.dfs(root, True, to_be_deleted_set)

        return self.result