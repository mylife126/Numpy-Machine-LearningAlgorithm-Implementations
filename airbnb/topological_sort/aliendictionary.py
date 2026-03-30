"""
There is a new alien language that uses the English alphabet. However, the order of the letters is unknown to you.

You are given a list of strings words from the alien language's dictionary. Now it is claimed that the strings in words are sorted lexicographically by the rules of this new language.

If this claim is incorrect, and the given arrangement of string in words cannot correspond to any order of letters, return "".

Otherwise, return a string of the unique letters in the new alien language sorted in lexicographically increasing order by the new language's rules. If there are multiple solutions, return any of them.



Example 1:

Input: words = ["wrt","wrf","er","ett","rftt"]
Output: "wertf"


Topological Sort
“永远添加没有 dependency 的 node”这一点。

我们可以把**拓扑排序（Topological Sort）**形象地理解为**“排课问题”**。

### 1. 你的总结 vs 官方定义

你说的“记录每个 edge 的 prerequisite 的大小”，在算法里专业术语叫 **入度 (Indegree)**。

* **入度 (Indegree)：** 指向这个节点的边的数量。
* **入度 = 0：** 代表这个节点**没有前置条件**（没有先修课），可以直接出发。
* **入度 > 0：** 代表还有“债”没还清，必须等前面的节点处理完。

---

### 2. 拓扑排序的标准流程 (Kahn's Algorithm)

确实如你所说，最常用的实现方式就是 **BFS**：

1. **统计入度：** 遍历所有的依赖关系，算出每个节点的 `indegree`。
2. **找起点：** 把所有 `indegree == 0` 的节点全部扔进 **Queue (BFS 队列)**。
3. **循环处理：**
* 从队列里拿出一个节点（它现在是“自由”的，可以排入最终顺序）。
* **拆掉它的边：** 把它指向的所有邻居（Neighbor）的 `indegree` 都减 1（代表这些邻居的一个前置条件已满足）。
* **释放新节点：** 如果某个邻居的 `indegree` 减到了 0，说明它也变“自由”了，把它也扔进队列。


4. **环检测：** 如果最后排出来的节点数少于总节点数，说明图里有**环**（比如 A 依赖 B，B 又依赖 A，大家都无法变成 0），此时无法排序。

---

### 3. 为什么这道题（Alien Dictionary）必须用它？

因为字母的顺序是**有向的**、**有先后之分的**。

* 当你发现 `w -> e`，`e -> r` 时，你不能简单的用字母表排序。
* 你需要一种方式，能把这些碎片化的“谁在谁前面”的信息，拼成一条完整的线。
* **拓扑排序的作用：** 它能保证在你的最终字符串里，**所有的依赖关系都指向右边**。

---

### 4. 常见的拓扑排序应用场景（帮你触类旁通）

除了外星语字典，只要看到**“先后顺序”**、**“依赖关系”**，就要想到拓扑排序：

| 场景 | 节点 (Node) | 依赖关系 (Edge) |
| --- | --- | --- |
| **排课系统** | 课程 A, B, C | 先修课要求 |
| **软件编译** | 不同的代码模块 | 模块之间的 `import` / `include` |
| **Excel 计算** | 单元格 | 公式里引用了哪个单元格 |
| **并行任务调度** | 任务 1, 2, 3 | 任务必须按顺序执行 |

### 总结

你刚才的直觉：**“记录 prerequisite 大小 + BFS 添加无 dependency 节点”** 已经完全覆盖了拓扑排序的核心逻辑。

> [!TIP]
> **面试小贴士：** > 拓扑排序通常有两种写法：一种是你现在用的 **BFS (Kahn's Algorithm)**，另一种是 **DFS (递归)**。在面试中，BFS 通常更直观，因为它可以直接处理“入度”，逻辑更不容易出错。

**既然你已经掌握了拓扑排序，想不想挑战一下它的“亲兄弟”题目 [Course Schedule II (课程表 II)](https://leetcode.com/problems/course-schedule-ii/)？它的逻辑和你现在的代码几乎一模一样！**
"""

from collections import defaultdict, deque
class Solution:
    def alienOrder(self, words):
        """
        Determine the order of letters in an alien dictionary.
        """
        # set up a adjencency list where tracks each char's edges/ neighbors
        adjacency_list = defaultdict(set)

        # set up a indegree graph, which tracks each char's numbers of prerequisite.
        # for example wrt, wrf , becomes t <- f means f depends on t, then f's indegree is 1 meaning how many char before f
        indegree = {}

        # initialize the indegree for all unique chars, this is because in the later of the bfs
        # the first leading char will not have any indegree, otherwise, there is a loop, and there is no order
        for word in words:
            for char in word:
                if char not in indegree:
                    indegree[char] = 0

                    # build the graph of the adjacency
        for index in range(len(words) - 1):
            first_word = words[index]
            second_word = words[index + 1]

            minimum_length = min(len(first_word), len(second_word))

            # abc ab -> no order
            if len(first_word) > len(second_word) and first_word[:minimum_length] == second_word[:minimum_length]:
                return ""

            for position in range(minimum_length):
                char_from_first_word = first_word[position]
                char_from_second_word = second_word[position]

                # when the char from the later is different than the char from the former, means that this char from later is dependend on the first
                if char_from_first_word != char_from_second_word:
                    if char_from_second_word not in adjacency_list[char_from_first_word]:
                        indegree[char_from_second_word] += 1
                        adjacency_list[char_from_first_word].add(char_from_second_word)

                    # <--- 这个极其重要！后面的字母顺序是不确定的，不能用来建图 也就是说找第一个不一样的dependency
                    """
                    2. 为什么要 break？（反证法）
                    如果不 break，你会建立错误的依赖关系。

                    假设外星语字典里有这两个词：
                    "wrt"
                    "wrf"
                    第 1 位： w == w（跳过）
                    第 2 位： r == r（跳过）
                    第 3 位： t != f。结论：t 优先于 f (t -> f)。

                    此时我们已经知道 t 排在 f 前面了。如果我们不 break 继续往后看（假设单词更长）：
                    比如："ab[c]xyz" 和 "ab[d]abc"

                    我们通过 c 和 d 确定了 c -> d。
                    如果继续看后面的 x 和 a，你会以为 x 也要排在 a 前面。

                    但事实上，x 和 a 的位置完全不能说明它们在外星语里的顺序，因为它们的先后位置已经被前面的 c 和 d 决定好了。
                    """
                    break

        # initialize the queue with the character that has no dependency
        queue = deque()
        for character in indegree:
            if indegree[character] == 0:
                queue.append(character)

        alien_order = []

        while queue:
            current_character = queue.popleft()
            alien_order.append(current_character)

            for its_edge in adjacency_list[current_character]:
                indegree[its_edge] -= 1
                if indegree[its_edge] == 0:
                    queue.append(its_edge)

        if len(alien_order) != len(indegree):
            return ""

        return "".join(alien_order)