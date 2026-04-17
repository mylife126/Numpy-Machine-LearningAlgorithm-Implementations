"""
Given an organizational hierarchy, and your task is to find the lowest common ancestor (LCA) in the group tree, given a set of employees.

This is exactly analogous to finding the lowest common parent in a tree structure, where:
1.  Each group is a node in a tree
2. Each employee is a leaf node associated with one group
3.  Each group may have a parent group (e.g., AI → Engineering)
4. The goal is: Given two or more employees, return the closest (lowest) common group that is a parent of all their groups

中文思路总结（直观版）

这题就是树上的 LCA，只是 employee 不在树里，employee 先映射到它所属的 group。

N 个员工怎么做最稳？

方法 A（最直观，面试最稳）：祖先集合求交
	1.	先拿第一个员工的 group，沿 parent 往上走，把所有祖先存到 set（包含自己）。
	2.	对于剩下每个员工：
	•	从它的 group 往上走，找到第一个出现在 set 里的祖先，这就是“当前答案”。
	•	然后把 set 收缩为“这个答案及其祖先集合”，继续处理下一个员工。
	3.	最终留下的就是所有员工的最低公共祖先。

这个方法的优点：不需要预处理，不需要深度，不需要二叉提升；实现很稳。

复杂度：设树高为 H，员工数为 K
	•	时间：O(K * H)
	•	空间：O(H)

⸻

English explanation (what I’d tell interviewer)

We map each employee to its owning group node. Then the problem reduces to finding the lowest common ancestor of multiple group nodes in a parent-pointer tree.

A simple approach:
	•	Build an ancestor set for the first group node (including itself up to root).
	•	For each subsequent group node, walk up via parent pointers until we hit a node in the ancestor set; that node is the LCA so far.
	•	Optionally shrink the ancestor set to the ancestors of this new LCA to keep it tight for the next iteration.

This runs in O(K * H) where K is the number of employees and H is the tree height.

"""

# approach 1, using node connection
class orgNode():
    def __init__(self, name):
        self.org_name = name
        self.parent = None

# set the org dummy org chart
ai = orgNode("ai")
infra = orgNode("infra")
engr = orgNode("engr")
sales = orgNode("sales")
comp = orgNode("comp")

# connect the org
ai.parent = engr
infra.parent = engr
engr.parent = comp
sales.parent = comp
comp.parent = None

"""
company | -> sales
        | -> engr |
                  |-> ai
                  |-> infra
"""


"""
# build ancestor_set for first org
ancestor_set = ancestors(org0)

lca_so_far = org0

for org in org_list[1:]:
    cur = org
    while cur is not None and cur not in ancestor_set:
        cur = parent[cur]

    if cur is None:
        return None

    lca_so_far = cur
    ancestor_set = ancestors(lca_so_far)

return lca_so_far
"""
class Solution(object):
    def lowestCommonGroupForEmployees(self, employees, employee2org):
        if employees is None or len(employees) == 0:
            return None

        orgs = []
        for employee in employees:
            its_org = employee2org[employee]
            orgs.append(its_org)

        ancestors = set()
        first_org = orgs[0]

        while first_org is not None:
            ancestors.add(first_org)
            first_org = first_org.parent

        common_org = None
        for i in range(1, len(orgs)):
            next_org = orgs[i]
            while next_org is not None and next_org not in ancestors:
                next_org = next_org.parent

            if next_org is None:
                return None

            common_org = next_org
            # otherwise means that the next org is at its parent, which is in the ancestor set, then
            # traverse its parents to shrink its parents
            new_ancestors = set()

            while next_org is not None:
                new_ancestors.add(next_org)
                next_org = next_org.parent

            ancestors = new_ancestors

        return common_org

if __name__ == "__main__":
    employee2org = {
        "jason" : ai,
        "john" : engr,
        "mandy": sales,
        "jeff" : comp,
        "andy" : infra
    }


    tester = ["jason", "andy", "john"]

    solution = Solution()
    result = solution.lowestCommonGroupForEmployees(tester, employee2org).org_name
    print(result)