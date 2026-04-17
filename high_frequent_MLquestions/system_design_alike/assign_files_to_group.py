"""
从你这段对话里，我抽取到的题目是：
	•	输入：一堆文件，每个文件有：
	•	collection_id（它属于哪个 collection）
	•	file_size（文件大小，整数）
	•	目标：输出 Top K collections，按 该 collection 下所有文件总大小 从大到小排序
	•	常见 tie-break：总大小相同按 collection_id 字典序（我默认加上，面试可一句话说明）
	•	你刚刚那句 “Is that basically the problem?” 是在做正确的澄清 👍

这题本质是：group-by 聚合 + topK。

Step 2｜核心思路讲解（Intuitive Explanation）

Mental model
	1.	先做一次扫一遍文件，把每个 collection 的总大小加出来：

	•	collection2total[collection] += file_size

	2.	得到每个 collection 的总大小后：

	•	用 heap 取 Top K

为什么用 heap？
	•	如果 collection 数量是 M，排序是 O(M log M)
	•	取 Top K 用 heap 是 O(M log K)，当 K << M 更好
"""

# since we maintain the top K only, when k <<m heap is a better choice
import heapq
from collections import defaultdict
# the input is files, [[collection_id, file_size]]
class Solution():
    def topKCollectionsByTotalSize(self, files, k):
        """
        files: List<list>, <collection_id, file_size>
        k: int
        return: list<(collection_id, total_file_size)>
        """

        if files is None or len(files) == 0 or k < 0:
            return []

        # set a variable to track the collection wise size
        collection2total = defaultdict(int)
        for collection_id, file_size in files:
            collection2total[collection_id] += file_size

        # now maintain a min heap to the max size of k
        # heap maintain (total_size, collection_id)
        min_heap = []

        for collection_id, total_file_size in collection2total.items():
            item = (total_file_size, collection_id)
            if len(min_heap) < k:
                heapq.heappush(min_heap, item)

            else:
                # if this item is bigger than the smallest entry in this heap
                smallest_toal, smallest_id = min_heap[0]

                if total_file_size > smallest_toal:
                    heapq.heapreplace(min_heap, item)

                elif total_file_size == smallest_toal and str(collection_id) < str(smallest_id):
                    heapq.heapreplace(min_heap, item)

        result = []
        while min_heap:
            total, cid = heapq.heappop(min_heap)
            result.append((cid, total))

        result = result[::-1]
        return result


"""
Now if the orgnization has the parent, meaning that a->C, or it could even be the case where a ->D meaning DAG, 
a org belongs to multiple parent. How would you do it?

Step 1｜题目快速解读（Problem Interpretation）

这是 follow-up：collection 不再是扁平的，而是有 hierarchy（树/森林），并且“多重从属关系”通常有两种可能含义：
	•	A. 树结构（单亲）：每个 collection 只有一个 parent（最常见、最好实现）
	•	B. DAG（多亲）：一个 collection 可能属于多个 parent（你说的“多重从属”更像这个）

为了不卡住面试，我给一个能覆盖两种的实现思路：
	•	输入：files = [(collection_id, size), ...]（文件归属在某个 leaf / 任意节点）
	•	输入：parents = {child: [parent1, parent2, ...]}（列表长度=1 就是树；>1 就是 DAG）
	•	目标：Top K collections（可以是任意节点，包括父节点），按 该节点“包含的所有后代集合”总文件大小 排序
	•	即：一个文件计入它所属 collection，同时也计入所有祖先 collections

（如果面试官说“只统计直接 children，不含孙子”，那是另一题；这里我按最常见 “roll-up to ancestors” 做。）

⸻

Step 2｜核心思路讲解（Intuitive Explanation）

Mental model
	1.	先把每个 collection 自己“直接拥有”的文件大小加起来：
direct_size[c] += file_size
	2.	然后把 direct_size 往上“汇总”到所有 ancestor：

	•	树（单亲）时：每个节点的 total = 自己 direct + 所有子树 total
	•	DAG（多亲）时：一个节点的 direct 会流向多个父节点（注意：这会导致“重复计数”的语义问题）
	•	默认语义：一个文件属于一个 leaf，但它会被所有祖先都计入（这是可接受的业务定义）

怎么算 total？
	•	这就是一个 图上的 DP / 拓扑 问题：child 的 total 要累加到 parent
	•	我们可以用 Kahn 拓扑排序，从叶子往上推：
	•	先 total[c] = direct_size[c]
	•	计算每个节点还有多少 child 没处理完（out-degree 反向）
	•	child 处理完就把 total 加到 parent

	3.	算完每个 collection 的 total 后，再用你喜欢的 min-heap 取 Top K。
"""
import heapq
from collections import defaultdict, deque
class SolutionV2():
    def topKCollectionsByTotalSize(self, files, parents, k):
        """
        files: list<collection_id, file_size>
        parents:
        dict, child -> List<[parents]>
        if len(parents) == 1 -> tree
        len(parents) >= 1 -> DAG
        """

        if files is None or len(files) == 0 or k < 0:
            return []

        # 1. first we need to get the direct size for each collection directly from the files
        direct=defaultdict(int)
        nodes = set()

        for cid, size in files:
            direct[cid] += size
            nodes.add(cid)

        # now include the nodes from the hierarchy
        children_map = defaultdict(list) # we want to traverse the parent graph into a paren to children graph
        indeg = defaultdict(int)
        # since we will process the add up, by traversing from leap to parent, we will need to track the left over children
        remaining_children = defaultdict(int)

        # now we need to build the graph
        # nodes : has all the nodes including leaf and parent
        # children map now becomes parent -> [leaf node]
        for child in parents:
            nodes.add(child)
            for parent in parents[child]:
                nodes.add(parent)
                children_map[parent].append(child)

        # now record every node's children size
        for node in nodes:
            how_many_children = len(children_map[node])
            remaining_children[node] = how_many_children

        # 2. now we track the total size for every node
        total = defaultdict(int)
        for node in nodes:
            # for now, we only have the size for those are given
            total[node] = direct[node]

        # 3. now we also need to have the child to parents map
        # child -> [parents]
        child2parent = defaultdict(list)
        for child in parents:
            for par in parents[child]:
                child2parent[child].append(par)


        # 4. now we need to start from the leaf and then we traverse all the way up
        q = deque()
        for node in nodes:
            # only when the node has 0 leaf, they are the leaf
            if remaining_children[node] == 0:
                q.append(node)

        # 5. then we pop the finished node, add its total size to the parent
        while q:
            node = q.popleft()

            for parent in child2parent[node]:
                total[parent] += total[node] # this node's upper parent will have its size
                remaining_children[parent] -= 1 # drop the parent's child by one because we finish this round of traversal
                if remaining_children[parent] == 0:
                    q.append(parent)

        # so after this cycle, each parent should have the total size from its leaf, without duplications
        # 6. now same thing, we will keep a minheap to track the topK
        min_heap = []
        for node in nodes:
            item = (total[node], node)
            if len(min_heap) < k:
                heapq.heappush(min_heap, item)

            else:
                if item > min_heap[0]:
                    heapq.heappush(min_heap, item)

        result = []
        while min_heap:
            total, node = heapq.heappop(min_heap)
            result.append((node, total))

        return result[::-1]


if __name__ == '__main__':
    files = [
        ("A", 10),
        ("B", 5),
        ("A", 7),
        ("C", 100),
        ("B", 20),
        ("D", 50),
    ]

    solution = Solution()
    print(solution.topKCollectionsByTotalSize(files, 2))
    # expected : A 17, B 25 D 50 C 100 -> c 100, D 50

    # V2
    files = [
        ("AI", 10),
        ("Infra", 20),
        ("Sales", 5),
    ]

    parents = {
        "AI":["Eng"],
        "Infra":["Eng"],
        "Eng":["Company"],
        "Sales":["Company"],
    }

    solution = SolutionV2()
    print(solution.topKCollectionsByTotalSize(files, parents, 2))
    """
    totals:
    AI = 10, infra = 20, sales =5
    eng = 30,
    com = 35
    
    output will be compny 35, eng 30
    """
