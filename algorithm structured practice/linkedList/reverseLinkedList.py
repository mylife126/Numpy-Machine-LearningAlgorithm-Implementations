# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
"""
思路一： 用array 存入每一个node， 然后反向遍历的过程里构建链表。 但是这样是O（N）会很慢。
"""
class Solution(object):
    def reverseList(self, head):
        if not head:
            return None
        array = []
        node = head
        while node:
            array.append(node)
            node = node.next

        new_head = array[-1]
        current = new_head

        # 边界注意， 右边界必须为-1 不然到不了index 0
        for idx in range(len(array) - 2, -1, -1):
            node = array[idx]
            current.next = node
            current = current.next
            
        # 重点 必须给此刻的Current的next 赋值none
        current.next = None
        return new_head


"""
思路二， one pass 遍历链表 然后记录一个previous node
"""
class Solution(object):
    def reverseList(self, head):

        # ----------------------------------------
        # previous_node:
        # head of reversed linked list
        # ----------------------------------------
        previous_node = None

        # ----------------------------------------
        # current_node:
        # node we are currently processing
        # ----------------------------------------
        current_node = head

        # ----------------------------------------
        # iterate through linked list
        # ----------------------------------------
        while current_node:

            # save next node first
            # otherwise linked list will be lost
            next_node = current_node.next

            # reverse pointer
            current_node.next = previous_node

            # move previous forward
            previous_node = current_node

            # move current forward
            current_node = next_node

        # previous_node becomes new head
        return previous_node