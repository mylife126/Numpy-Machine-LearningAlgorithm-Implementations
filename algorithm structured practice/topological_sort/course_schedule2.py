"""
每一道题都有一个prerequisite，所以每一个题都有一个入度。

那么只有当一个课程的入度到零了才可以去上课。

思路则用topological sort去维护每一个课程的入度
然后用bfs的展开方式， 我们对于indegree为零的课程开始对其下属课程展开，
而展开的过程里则是对每一个下属课程的入度-1， 如果某一个下属课程的入度为0，
则说明这个课程可以加入Q 去上课了

"""

from collections import defaultdict, deque
class Solution(object):
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """

        adjacency = defaultdict(list)
        indegree = defaultdict(int)

        for course in prerequisites:
            pre = course[1]
            child = course[0]
            indegree[child] += 1
            adjacency[pre].append(child)

        queue = deque()
        for course in range(numCourses):
            if indegree[course] == 0:
                queue.append(course)

        course_order = []
        while queue:
            taken_course = queue.popleft()
            course_order.append(taken_course)
            for child in adjacency[taken_course]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        if len(course_order) != numCourses:
            return []

        return course_order
