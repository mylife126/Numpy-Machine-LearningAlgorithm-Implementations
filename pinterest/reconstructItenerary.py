"""

⸻

🧠 LeetCode 332 — Intuitive Walkthrough（终极理解版）

⸻

🎯 题目核心（一句话）

👉

用完所有机票（边），并保证字典序最小


⸻

🧩 示例

tickets = [
    ["JFK","A"],
    ["JFK","B"],
    ["JFK","C"],
    ["A","JFK"]
]


⸻

🧠 Step 0：建图（用 heap 保证字典序）

JFK: ["A", "B", "C"]   # A 最小
A: ["JFK"]
B: []
C: []


⸻

🎯 DFS 模板（必须记住）

def dfs(node):
    while graph[node]:
        next = heappop(graph[node])
        dfs(next)
    result.append(node)


⸻

🚀 DFS 执行（逐步）

⸻

Step 1️⃣ dfs(“JFK”)

graph["JFK"] = ["A", "B", "C"]

👉 pop "A"

JFK → A
graph["JFK"] = ["B", "C"]

👉 进入：

dfs("A")


⸻

Step 2️⃣ dfs(“A”)

graph["A"] = ["JFK"]

👉 pop "JFK"

A → JFK
graph["A"] = []

👉 进入：

dfs("JFK")   # 第二层 JFK


⸻

Step 3️⃣ dfs(“JFK”)（第二次）

graph["JFK"] = ["B", "C"]

👉 pop "B"

JFK → B
graph["JFK"] = ["C"]

👉 进入：

dfs("B")


⸻

Step 4️⃣ dfs(“B”)

graph["B"] = []

👉 append：

result = ["B"]


⸻

🔙 回到 JFK（第二次）

👉 ❗关键点：while 没有结束！继续执行

graph["JFK"] = ["C"]

👉 pop "C"

JFK → C
graph["JFK"] = []

👉 进入：

dfs("C")


⸻

Step 5️⃣ dfs(“C”)

graph["C"] = []

👉 append：

result = ["B", "C"]


⸻

🔙 回到 JFK（第二次）

graph["JFK"] = []

👉 append：

result = ["B", "C", "JFK"]


⸻

🔙 回到 A

graph["A"] = []

👉 append：

result = ["B", "C", "JFK", "A"]


⸻

🔙 回到 JFK（第一次）

graph["JFK"] = []

👉 append：

result = ["B", "C", "JFK", "A", "JFK"]


⸻

🎯 最终 reverse

["JFK", "A", "JFK", "C", "B"]


⸻

❗你最容易错的点（重点）

你之前的错误是：

jfk -> b → append
jfk -> c → append

👉 ❌ 错在：

把 while 拆成多个“独立执行”


⸻

🔥 正确理解（最关键）

👉

while 是一个整体


⸻

🧠 正确 mental model（必须记住）

dfs(JFK):
    dfs(A):
        dfs(JFK):
            dfs(B)
            dfs(C)
            append(JFK)
        append(A)
    append(JFK)


⸻

🎯 一句话打通

👉

一个节点只有在“所有 outgoing edges 都用完”之后才 append


⸻

🔥 更深一层理解（面试加分）

👉

heappop = 用掉一张机票
graph 是全局共享的


⸻

⚠️ 关键现象

👉 deeper DFS 会修改 graph：

JFK: ["B", "C"]

可能变成：

JFK: []

👉 当你回到上层时，这些边已经没了！

⸻

🎯 BFS 为什么不行？

❌ BFS：

同时扩展多条路径

👉 无法控制：

哪条路径用了哪张票


⸻

✅ DFS：

一条路径走到底
每条边只用一次（heappop）


⸻

🧠 面试标准表达（英文）

The while loop ensures that all outgoing edges are consumed before appending the node. Even though recursion temporarily pauses the loop, it resumes after returning, until no edges remain.

⸻

🚀 最终总结（直接背）

👉

1. heap → 保证字典序
2. DFS → 一路走到底
3. heappop → 消耗边
4. while → 用完所有边
5. append → 在回溯时记录路径
6. reverse → 得到最终答案


⸻
不是在“找路径”，而是在“消耗所有边”，路径是在回溯时自然形成的

"""
from collections import defaultdict
import heapq


def dfs(node_now, graph, results):
    # if the node_now has its end, then continue to see the end
    while graph[node_now] != []:
        its_end = heapq.heappop(graph[node_now])
        dfs(its_end, graph, results)

    # otherwise we have reached the end of the ticket at this layer
    # put this end into the list
    results.append(node_now)


class Solution(object):
    def findItinerary(self, tickets):
        """
        :type tickets: List[List[str]]
        :rtype: List[str]
        """

        travel_graph = defaultdict(list)

        for start, end in tickets:
            heapq.heappush(travel_graph[start], end)

        results = []
        dfs("JFK", travel_graph, results)
        return results[::-1]


"""
graph = {
    "JFK": ["ATL", "SFO"],   # heap → ATL 小
    "SFO": ["ATL"],
    "ATL": ["JFK", "SFO"]    # heap → JFK 小
}

while jfk -> atl, jfk:[sfo]

while atl -> jfk, atl:[sfo]

while jfk -> sfo, jfk: []

while sfo -> atl, sfo: []

while atl -> sfo, atl: []

whle sfo empty append [sfo] return to while atl -> sfo, atl: []

while atl empty append [sfo, atl], return to while sfo -> atl, sfo: []

while sfo empty append [sfo, atl, sfo], return to while jfk -> sfo, jfk: []

while jfk empty append [sfo, atl, sfo, jfk], return to while atl -> jfk, atl:[sfo] but now the atl 's edge is actually empty

so while atl -> jfk empty append [sfo, atl, sfo, jfk, atl] return to while jfk -> atl, jfk:[sfo] but still jfk now is emtpy too because we poped 

while jfk -> atl empty append [sfo, atl, sfo, jfk, atl, jfk]

so it is [jfk, atl, jfk, sfo, atl, sfo]




tickets = [
    ["JFK","A"],
    ["JFK","B"],
    ["JFK", "C"]
    ["A","JFK"]
]
JFK: ["A", "B", "C"]
A: ["JFK"]

jfk -> A, jfk : b, c
A -> jfk, A: []
jfk -> b, jfk: [c]
b ->[] backtrack , result: [b]

jfk -> c, jfk:[]
c-> [], return to jfk -> c, result: [b, c]

jfk empty return to A -> jfk, result [b, c, jfk]
A empty return to jfk to A, result [b, c, jfk, a, jfk]

jfk a jfk c b 
"""