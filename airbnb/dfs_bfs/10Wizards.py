"""
🧩 10 Wizards

Medium（Airbnb高频）

⸻

Description

There are n wizards, labeled from 0 to n-1.

You are given a list wizards, where: wizards[i] = list of wizards that wizard i knows
If wizard i knows wizard j, then you can travel from i → j.

Cost Definition

The cost to travel from wizard i to wizard j is: (i - j)²

Task
Given a source wizard and a target wizard, return the minimum cost required to travel from source to target.
If no path exists, return -1.

Example
Input:
wizards = [
 [1, 2],
 [3],
 [3],
 []
]

source = 0
target = 3


Graph:
0 → 1 → 3
 \→ 2 → 3


Cost calculation:
Path1: 0 → 1 → 3
= (0-1)^2 + (1-3)^2
= 1 + 4 = 5

path2:
0 → 2 → 3
= (0-2)^2 + (2-3)^2
= 4 + 1 = 5

思路， 这道题目类似于cheapest flight lc 这道题， 逻辑是因为都是找最短路径 但是每一个edge是有cost的。那么我们就用heapq来维护bfs的堆，
每一个节点逐层遍历， bfs过程里：
1. 首先pop cost 最小的节点
2. 看看节点是不是目标地 是的话return
3. 如果当下这个pop出来的节点的cost 在之前其实已经有访问过且更便宜 continue 所以我们要maintain另一个状态就是每一个节点到达的最低cost

然后对于每一个节点自己的neighbor进行访问 以及计算cost 如果cost 低于 该neighbor的到达cost 则可以放去priority queue里面

"""
import heapq
class Solution:
    def cost(self, node1, node2):
        return (node1 - node2) ** 2

    def minCost(self, wizards, source, target):
        n = len(wizards)
        nodes_dest_cost = [float('inf')] * n

        # the starting point is 0
        nodes_dest_cost[source] = 0

        hq = [] # maintain the cost, node
        heapq.heappush(hq, [0, source])

        while hq:
            current_node_cost, current_node = heapq.heappop(hq)

            if current_node == target:
                return current_node_cost

            if current_node_cost > nodes_dest_cost[current_node]:
                continue

            for neighbor in wizards[current_node]:

                # 这里是重点 我们要track的是总cost 到达一个点的总体cost。 所以要加上 到达current node的cost
                edge_cost = self.cost(current_node, neighbor)
                cost_to_neighbor = current_node_cost + edge_cost

                if cost_to_neighbor < nodes_dest_cost[neighbor]:
                    nodes_dest_cost[neighbor] = cost_to_neighbor
                    heapq.heappush(hq, [cost_to_neighbor, neighbor])

        return -1

if __name__ == '__main__':
    solution = Solution()
    wizards = [
        [1, 2],
        [3],
        [3],
        []
    ]

    print(solution.minCost(wizards, 0, 3))
    # 5

