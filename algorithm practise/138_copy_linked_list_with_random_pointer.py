
# 正确写法
# 所以连表 其实不仅仅是一个value 而是说它在内存里assign了一个新的node 它的内存里的代表形式是不同的。
# 所以之前的写法 its_new_node.next = the_current.next 虽然长得是一样的， the_current.next是存在另一个内存的表达里面的，这点很重要。 所以构建its_mew_node.next 得找到它自己的new next是哪里
# 那么要找到就是通过old node to new node mapping 因为在第一个循环里已经把所有的node都用mapping存下来了
class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        if not head:
            return None

        old2new_mapping = dict()
        the_current = head

        while the_current:
            old2new_mapping[the_current] = Node(the_current.val)
            the_current = the_current.next
        
        the_current = head
        while the_current:
            its_new_node = old2new_mapping[the_current]

            if the_current.next:
                its_old_next = the_current.next
                its_new_next = old2new_mapping[its_old_next]

                its_new_node.next = its_new_next
            if the_current.random:
                its_old_random = the_current.random
                its_new_random = old2new_mapping[its_old_random]
                its_new_node.random = its_new_random
            the_current = the_current.next
        
        the_new_head = old2new_mapping[head]
        return the_new_head