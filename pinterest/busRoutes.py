"""
stops: [busi, busj, busk...]
那么每一层bfs是看bus level的拓展， 也就是说 看每一个stop能有哪一个bus到达，
然后对于这个bus其余的stops 也加入queue里面。 那么在循环里：
例如 ： routes = [[1,2,7],[3,6,7]], source = 1, target = 6

所以while 循环 queue 也就是bfs展开的时候是按照bus 来展开， 然后within bus layer 去看他每一个next 站台， 用for：
    对于stop1， 目前只有bus1能到， 所以我们将这个stop能到达的buses还能到达的stops都看一遍 也就是next stops，
    那如果有任何一个站台是该目标，则此刻的bus数量就是固定了. 同时记录这个做过的bus， 然后去过的stop 因为有环

    那么如果这个stop对应的buses内的下一站都没有可行的， 则说明这个stop对应的buses到不了了 则需要换乘别的车，则退出for循环，退出到while的bfs去用queue里面的下一站信息去找到接下来可以做的buses

因为我们做了bus visited 检测，哪怕第一层bus里添加了只属于自己的stops， 在下一次while里这个bus不会被访问 除非遇到一个stop他属于几个不同的bus， 例如下面：

    [[1,2,3，7], [3,4,7], [3,6,7]]

    stop 3 属于bus0 属于bus1 也属于bus2
    stop 7 属于bus1 也属于bus2

    所以这个情况里哪怕 stop2在queue里面，但是bus0坐过了 且到不了想去的地方那就去stop3，
    stop3 可以做bus1 也可以bus2， 那么我们进入for 循环去看从stop3开始的buses能去什么地方， 发觉bus2 可以到达target， 则说明找到了，且此刻bus++ = 2

    ⚠️ 同样的逻辑可以用在coin change这个题目 都是逐层遍历！
"""

"""
   [[1,2,3，7], [3,4,7], [3,6,7]]  start1 target6
⸻

其实这题本质是这样的：

我们是在做 BFS，但不是按“站台一层一层扩展”，而是“按坐了几辆 bus 来分层”。

一开始 queue 里只有 source，比如是 stop 1。这个时候还没坐车，所以 num_buses = 0。进入 while 的第一轮，我们做 num_buses += 1，这一层就表示“我现在准备坐第一辆 bus”。

然后这一层的任务是：把当前 queue 里的所有 stop（这些 stop 都是“用 0 辆车能到的”）全部处理一遍。比如 stop 1，它能坐的 bus 是 bus0，那么一旦我们决定坐 bus0，就相当于“我现在用了 1 辆车”。

接下来，从 bus0 出发，它能到的所有 stop（比如 2, 3, 7）就都被加入 queue，这些 stop 都属于“用 1 辆车可以到的范围”。同时这个 bus0 会被标记 visited，这样后面不会重复坐同一辆车。

接下来进入下一轮 while，也就是 num_buses = 2，这一层表示“我现在可以坐第二辆车”。这时候 queue 里的所有 stop（比如 2,3,7）都是“用 1 辆车能到的”，我们要看从这些 stop 出发还能换哪些新的 bus。

比如到 stop 3，这个点很关键，因为它属于多个 bus，比如 bus1 和 bus2。这里不是“先试 bus1 再试 bus2”，而是这一层 BFS 会把 bus1 和 bus2 都一起扩展。也就是说我们同时考虑：如果我在这里换乘 bus1 会去哪，如果换 bus2 会去哪。

然后我们看 bus1 能到哪些站，比如 4；bus2 能到哪些站，比如 6。如果在这个过程中发现某个 next_stop == target（比如 6），那说明我们在这一层找到了目标，而这一层对应的 num_buses 就是最少需要的 bus 数，因为 BFS 是一层一层扩展的，第一次到达一定是最优的。

一个很关键的点是：BFS 不存在“这个 stop 不行就退出”的逻辑。某一个 stop 走不通没关系，queue 里还有别的 stop 会继续被处理。整个过程是“同一层的所有可能一起展开”，而不是一条路径失败再换另一条。

另外 visited_buses 是用来保证一辆 bus 只会被坐一次，否则会在同一条线路上无限循环；visited_stops 是用来防止站台反复加入 queue。

所以总结一下你的这套逻辑可以更精确地理解为：我们在 stop 上做 BFS，但 BFS 的层代表的是“坐了几辆 bus”，每一层会把当前所有 stop 能换乘的所有 bus 一起扩展，再把这些 bus 能到的所有 stop 加入下一层。一旦某一层中出现 target，就说明用这么多 bus 就能到达，而且是最少的。

"""
from collections import defaultdict, deque


class Solution(object):
    def numBusesToDestination(self, routes, source, target):
        """
        :type routes: List[List[int]]
        :type source: int
        :type target: int
        :rtype: int
        """
        # 注意重点边界问题， 当source == target了 直接不用换车了 return 0
        if source == target:
            return 0

        stop_to_bus = defaultdict(list)
        for bus_i in range(len(routes)):
            its_stops = routes[bus_i]
            for stop in its_stops:
                stop_to_bus[stop].append(bus_i)

        # bfs
        queue = deque()
        queue.append(source)
        visited_buses = set()
        visited_stops = set()
        visited_stops.add(source)

        num_buses = 0

        # still bfs on stop but use stop to control the bus layer
        while queue:
            num_buses += 1  # every layer means taking a new bus

            # use the current stop to check the all the buses you can take to reach it
            # finish all the next stops from the previous bus layer to see if the next stop's mapped bus can lead to the right target
            # Caution: must popleft() all the next stops from the last bus layer, first come first serve
            # if you use while, the queue is always expanding not layer by layer

            for _ in range(len(queue)):
                next_stop = queue.popleft()
                buses_can_be_taken = stop_to_bus[next_stop]
                for bus in buses_can_be_taken:
                    if bus in visited_buses:
                        continue

                    visited_buses.add(bus)

                    for new_next_stop in routes[bus]:
                        if new_next_stop in visited_stops:
                            continue

                        if new_next_stop == target:
                            return num_buses

                        visited_stops.add(new_next_stop)
                        queue.append(new_next_stop)

        return -1