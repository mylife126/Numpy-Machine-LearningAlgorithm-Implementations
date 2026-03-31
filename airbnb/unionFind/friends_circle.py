"""
There are n cities. Some of them are connected,
while some are not.
If city a is connected directly with city b,
and city b is connected directly with city c,
then city a is connected indirectly with city c.

A province is a group of directly or indirectly
connected cities and no other cities outside of the group.

You are given an n x n matrix isConnected where
isConnected[i][j] = 1 if the ith city and the jth city are directly connected,
and isConnected[i][j] = 0 otherwise.

Input: isConnected = [[1,1,0],
                      [1,1,0],
                       [0,0,1]]
Output: 2

"""

'''
Solution 2: Union Find
1. Parent = int[n], we have n people, 我们一开始把每一个人作为一个cluster
2. size = int[n], 同理我们有n个人，我们把每一个人对应的cluster size设为1
3. 当我们遇到 grid[i][j] == 1 且i != j的时候就说明有了connection且不是自己比对自己，这个时候就可以call union
4. 最后找到所有的parent的counts即可
'''


class UF(object):
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.size = [1 for i in range(n)]
        self.count = n

    def getRoot(self, node):
        root = node
        while (root != self.parent[root]):
            self.parent[root] = self.parent[self.parent[root]]
            root = self.parent[root]
        return root

    def union(self, nodeA, nodeB):
        rootA = self.getRoot(nodeA)
        rootB = self.getRoot(nodeB)

        if rootA != rootB:
            if self.size[rootA] >= self.size[rootB]:
                self.parent[rootB] = rootA
                self.size[rootA] += self.size[rootB]
            else:
                self.parent[rootA] = rootB
                self.size[rootB] += self.size[rootA]
            self.count -= 1


class Solution(object):
    def findCircleNum(self, A):
        if not A:
            return 0

        n = len(A)
        unionFind = UF(n)

        for i in range(n):
            for j in range(n):
                if A[i][j] == 1 and i != j:
                    unionFind.union(i, j)

        return unionFind.count


