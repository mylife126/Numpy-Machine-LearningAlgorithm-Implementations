"""
Description

You are given a directed graph representing ski routes on a mountain.

Each edge represents a path between two locations with a cost, and each node (except the starting node) has an associated reward.

The graph is defined by two inputs:
	•	travel: a list of edges where
travel[i] = [from, cost, to]
indicates a directed edge from from to to with cost cost.
	•	points: a list of node rewards where
points[i] = [node, reward]
indicates that arriving at node gives you reward.

You always start from a fixed starting node "start".

There can be multiple end nodes, which are nodes that do not have any outgoing edges.

⸻

Score Definition

The score of a path is defined as:

(total rewards collected from visited nodes) - (total costs of edges traveled)

	•	You do not get any reward for the starting node "start".
	•	You collect the reward of a node when you arrive at it.

⸻

Task

Return the maximum score you can achieve by starting from "start" and ending at any valid end node.

If no path exists, return the minimum possible value (or assume at least one valid path exists).

⸻

Example 1

Input:

travel = [
 ["start","3","A"],
 ["A","4","B"],
 ["B","5","END"]
]

points = [
 ["A","5"],
 ["B","6"],
 ["END","3"]
]


⸻

Output:

4


⸻

Explanation:

There are three possible paths:
	1.	start → A

score = 5 - 3 = 2


⸻

	2.	start → A → B

score = (5 + 6) - (3 + 4)
      = 11 - 7
      = 4


⸻

	3.	start → A → B → END

score = (5 + 6 + 3) - (3 + 4 + 5)
      = 14 - 12
      = 2


⸻

The maximum score is:

4


⸻

Example 2

Input:

travel = [
 ["start","2","A"],
 ["start","4","B"],
 ["A","3","END1"],
 ["B","1","END2"]
]

points = [
 ["A","5"],
 ["B","3"],
 ["END1","4"],
 ["END2","2"]
]


⸻

Output:

4


⸻

Explanation:

Path 1: start → A → END1

score = (5 + 4) - (2 + 3) = 9 - 5 = 4

Path 2: start → B → END2

score = (3 + 2) - (4 + 1) = 5 - 5 = 0


⸻

Maximum score = 4

⸻

Constraints
	•	1 <= number of edges <= 10^4
	•	1 <= number of nodes <= 10^4
	•	Graph is guaranteed to be a Directed Acyclic Graph (DAG)
	•	Costs and rewards are non-negative integers

⸻

Follow-up
	1.	Return the actual path that yields the maximum score.
	2.	If multiple paths have the same maximum score, return any one of them.

⸻
核心疑惑， 为什么要用topological sort来构造处理nodes的顺序？

参考以下可能的滑雪路径
start
  ↙   ↘
 A     C
  ↘   ↙
    B

可见 可能的路径是  start -> A -> B  or
                start -> C -> B

所以可以的做法是遍历所有的路径可能性 然后计算 sum reward - sum cost， 用dfs来做，但是这里的问题是复杂度极高。 因为每一个node 都可能有下一个分支

那么optimal的解法则是用DP来动态的更新到达某一个node的score， topdown来做，那么就有一个dependency 问题， 也就是说 假设路径是 a b c， 想要知道
c的最高分 就得知道b的最高分 想要知道b则需要知道a

那么如何保证用dp来记录state的分数的时候能确保保证这个dependency解决的好呢， 这就需要用topological sort来一个一个把没有dependency的node加进去
上述例子里， 那么start 入度为0， A的入度是1， C是1， 而b是2

那么处理的优先级成为 Start -> A -> C
                            ↘    ↘
                            B     B

也就是说dp[start] = 0
然后处理Dp【A】 = max(dp[start] + reward[A] - cost[A], dp[A]) 这里为什么对比Dp[a] 是因为有可能别的方法能到达A， 这里就能保证最优解
同理 Dp[C] = max(dp[start] + reward[C] - cost[C], dp[C])

now comes to dp[B]
可以看到处理完A后就可以处理B 那么dp[b] = max(dp[A] + reward[B] - cost[b], dp[b])

而处理完C后 也可以在处理一次B， 这个时候的dp[b]是通过路径A 算出来的


indegree 是啥？

👉 “有多少前驱”

⸻

为啥从 0 开始？

👉 “没有依赖，可以先做”

⸻

BFS 在干嘛？

👉 “一层一层消除依赖”

⸻

topo order 是啥？

👉 “正确的计算顺序”

⸻

为啥 DP 要按 topo？

👉 “保证用到的值已经算好”

"""

from collections import defaultdict, deque
class Solution:
    def max_score_ski_path(self, travel, points):
        """
        :type travel: List[List[str]], start, cost, end
        :type points: List[List[str]]
        :rtype: int
        """

        # Step 1: get the graph set up, save the neighbors and its cost and
        # build the indegree graph
        graph = defaultdict(list)
        indegree = defaultdict(int)

        all_nodes = set()

        for t in travel:
            start, cost, end = t[0], int(t[1]), t[2]
            graph[start].append((end, cost))
            indegree[end] += 1

            if start not in indegree:
                indegree[start] = 0

            all_nodes.add(start)
            all_nodes.add(end)

        print(all_nodes)
        # Step 2: track all the rewards
        rewards = defaultdict(int)
        for point in points:
            node, reward = point[0], int(point[1])
            rewards[node] += reward

        # step 3: set up the topological order
        # bfs, starting with those nodes with indegree  with 0
        # then traverse each node's edges , indgree[edge] -=1 , if equals to zero,
        # add into the queue
        queue = deque()

        for node in indegree:
            if indegree[node] == 0:
                queue.append(node)

        topological_order = []
        while queue:
            node = queue.popleft()
            topological_order.append(node)

            for edge in graph[node]:
                edge_node = edge[0]
                indegree[edge_node] -= 1

                if indegree[edge_node] == 0:
                    queue.append(edge_node)

        # step 4 top down DP to track each node's reached state's total score
        dp = defaultdict(int)

        # init the dp with -inf
        for node in all_nodes:
            dp[node] = float('-inf')

        dp["start"] = 0
        for node in topological_order:
            for neighbor in graph[node]:
                neighbor_node, neighbor_cost = neighbor[0], neighbor[1]
                neighbor_reward = rewards[neighbor_node]

                new_score = dp[node] + neighbor_reward - neighbor_cost

                if new_score > dp[neighbor_node]:
                    dp[neighbor_node] = new_score

        max_score = float('-inf')
        for node in all_nodes:
            if node not in graph or len(graph[node]) == 0:
                max_score = max(dp[node], max_score)
        print(dp)
        return max_score

if __name__ == '__main__':
    travel = [
        ["start", "2", "A"],
        ["start", "4", "B"],
        ["A", "3", "END1"],
        ["B", "1", "END2"]
    ]

    points = [
        ["A", "5"],
        ["B", "3"],
        ["END1", "4"],
        ["END2", "2"]
    ]
    solution = Solution()
    print(solution.max_score_ski_path(travel, points))
    # expected: 4