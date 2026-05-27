"""
逻辑是， 下一个interval的start 如果小于此刻的end， 则代表可以merge， 那么此刻的merging end则是下一个interval的end

不然此刻的interval 无法继续merge 则加入result， 而由于我们用for loop不断看的是下一个internal所以， 在下一次循环遇到的是下下个interval了，我们需要把merging now 更新成这一次循环里的next 这样循环下一轮的时候 我们就是比对此刻无法merge的 next interval

[[1,3],[2,6],[8,10],[15,18]]

首先 merging start 是 【1 3】
next 是 【2 6】 可以merge， 则更新merging end 为6

继续循环next到 8 10， 不可以merge， 则把现在merge好的interval 【1， 6】加入result， 更新merging start 为8， merging end为10，

继续虚幻next 到了 15 18， 我们此刻的merging start 是8， merging end是10， 发觉无法merge， 则继续添加。merging start 为15， merging end为18

此刻for loop结束，我们还有一个left的interval没有添加 再次添加

⚠️ 这里有一个特别的edge case 就是next interval是完全被当前interval包括的
[[1,4],[2,3]]
所以更新merging end的时候不能直接用next end， 而是用max（merging end， next end）来选取
"""
class Solution(object):
    def merge(self, intervals):
        if not intervals or len(intervals) == 0:
            return 0

        result = []
        intervals = sorted(intervals, key = lambda x:x[0])
        merging_start, merging_end = intervals[0][0], intervals[0][1]

        for next_i in range(1, len(intervals)):
            next_interval = intervals[next_i]
            next_start, next_end = next_interval[0], next_interval[1]


            # ⚠️ 这里有一个特别的edge case 就是next interval是完全被当前interval包括的
            # [[1,4],[2,3]]
            # 所以更新merging end的时候不能直接用next end， 而是用max（merging end， next end）来选取
            if next_start <= merging_end:
                merging_end = max(merging_end, next_end)

            else:
                result.append([merging_start, merging_end])
                merging_start, merging_end = next_start, next_end

        result.append([merging_start, merging_end])
        return result