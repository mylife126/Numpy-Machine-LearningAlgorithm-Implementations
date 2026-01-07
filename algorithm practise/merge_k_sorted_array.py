# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
"""
思路是用heap来maintain一个nlogn的排序 dynamically。 先对每一个linked list做heap push
然后构造一个dummy node 开始对heap做pop， 每pop一个node 就对它看是否有next， 有的话就push in新的value， node

"""
import heapq
class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[Optional[ListNode]]
        :rtype: Optional[ListNode]
        """
        dummy = ListNode(-1)
        pq = []
        heapq.heapify(pq)
        print(lists)
        for link in lists:
            if link:
                its_value = link.val
                heapq.heappush(pq, [its_value, link])
        
        current = dummy
        while pq:
            val, node = heapq.heappop(pq)
            current.next = node
            current = current.next

            if node.next:
                heapq.heappush(pq, [node.next.val, node.next])
        
        return dummy.next
            
        