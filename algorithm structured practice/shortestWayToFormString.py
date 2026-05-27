"""
题目逻辑其实是说有一个target string abcbc
同时有一个source string abc， 我们可以 扫描成 a， 或者ac， 或者abc， 或者bc 这样的substring

问的是如何才能构成target。

用abcbc来看， 我有一个指针在target 从左往右看，我们做的事情是一段段从source里找subsquece去满足target里的每一个指针的位置

例如 a b c bc
    — -  -

可以用source abc 满足

接下来就是bc， 而bc这一层我们发觉 source的a没用， b可以用且满足了bc里面的b， 然后到了c，也满足了bc里面的c。

此时我们两次count。

算法实现这个逻辑则是，
指针i控制target的位置， 然后每一层是一个贪心的向右扫描target 和 source， 而source永远从第一个char开始， 然后这一层里， 只要source[j] == target[i] 则说明有匹配了一个char给target。 举个例子

a b c b c

第一层 count += 1
先从i=0开始， j=0， 匹配成功， j+=1 i+=1，
i =1 ， j = 1， 发觉匹配再度成功， j+=1 i+=1
i=2， j=2， 发觉再度成功，j+=1， i+=1
i=3 = b， j=3 out bound了， 说明这一层没法继续了， 没有subsequence能成为abcb，

那就去下一层，现在要处理的target是bc， count +=1
j重新开始

i=3=b， j=0=a 不匹配， i不改变， j+=1
i=3=b， j=1=b 匹配， i+=1 j+=1
i=4， j=2 匹配成功， i+=1， j+=1
发觉i越界了，则说明找完了。


那么这里当一层循环里 i 完全没改变则说明 不存在这样的subqequence
"""


class Solution(object):
    def shortestWay(self, source, target):
        """
        :type source: str
        :type target: str
        :rtype: int
        """
        chars_needed_from_target = set(target)
        chars_we_have_from_source = set(source)

        for char in chars_needed_from_target:
            if char not in chars_we_have_from_source:
                return -1

        i = 0
        counts = 0

        while i < len(target):
            # for each layer, we count +=1 means that we greedliy find the
            # longest subquence valid to match up with the longest subquence from the target
            counts += 1

            # always construct the subsequence from source from 0
            j = 0

            # set the start pointer for this layer's target
            start = i

            while j < len(source) and i < len(target):
                # if the current char at source is the same from the current char from target, we move to the next to see if we can match
                if source[j] == target[i]:
                    i += 1
                # no matter match or no, we always try the next char from the source
                j += 1

            # now if the i do not move, means that we have scanned every char
            # left to right from source, and we could not even match a single
            # char from the target, we return -1
            if i == start:
                return -1

        return counts