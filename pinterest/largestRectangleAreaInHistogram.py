"""
对于每一个i 认为他自己就是对小的柱子，那么我们目标是不断向左向右扩展直到不能扩展，也就是说找到它左边第一个低于自己的高度的柱子， 然后找到右边第一个低于自己的柱子就是不能扩展了。

左边第一个 < heights[i] 的位置
右边第一个 < heights[i] 的位置

那么如何用one pass解决呢 也就是O（n）

想象我们维护一个单调递增的stack， 每一个element是index， 但是height【1】 < height【2】

例如【1 5 6 2】

我们for 循环的时候只需要先看 height[i] < stack[-1], 此时height i 其实就是stack 【-1】里的右边边界， 而用来计算的height其实就是stack 【-1】 pop的这个height，

例如我们for loop到了2， 此刻6在stack里，而他的左边界在stack下一个 也就是5的位置
所以对于6来说 ：
6 被右边边界2 压死， 所以右边边界是2这个地方，也就是index=4，
而它又被左边界5压死， 所以左边界是stack【-1】 （注意其实是stack【-2】 但是由于已经pop出来一次了所以就是再一次-1） 也就是5的位置=1

得到 width = right_boundary - left_boundary - 1 = 3 - 1 - 1 = 1
而它的最大的面积则是 6 * 1


然后继续看， stack【-2】 = 5 也肯定大于2， 所以5的右边界还是2 index = 3， 而此刻的stack为空， 说明它左边没有比自己小的了，左边界就是自己， 而又边界还是延伸到2之前的6的位置。 则左边界为-1
width = right_boundary - left_boundary - 1 = 3 - (-1) - 1 = 2
      = i - (-1) -1 = i


那么另一个问题则是 如果一个数组是 1 2 3 4 5 6 怎么办， 我们可以在他后面加入一个0
变成 1 2 3 4 5 6 0
那么我们算法会一直在stack里加入 1 2 3 4 5 6 ， 到了0这个位置的时候，
我们开始pop right，
先pop 6， 6 * 1 = 6
在pop 5， 5 * 2 = 10
在pop 4， 4 * 3 = 12
在pop 3， 3 * 4 = 12
在pop 2， 2 * 5 = 10
在pop 1， 1 * 6 = 6

"""


# [2,1,5,6,2,3,0]
class Solution(object):
    def largestRectangleArea(self, heights):
        stack = []
        max_area = 0

        # 🔥 加一个 0 强制清栈
        heights.append(0)

        for i in range(len(heights)):
            while stack and heights[i] < heights[stack[-1]]:
                h = heights[stack.pop()]

                if not stack:
                    w = i - (-1) - 1
                else:
                    w = i - stack[-1] - 1

                max_area = max(max_area, h * w)

            stack.append(i)

        return max_area