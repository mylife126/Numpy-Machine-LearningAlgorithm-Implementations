"""
Given an organizational hierarchy, and your task is to find the lowest common ancestor (LCA) in the group tree, given a set of employees.

This is exactly analogous to finding the lowest common parent in a tree structure, where:
1.  Each group is a node in a tree
2. Each employee is a leaf node associated with one group
3.  Each group may have a parent group (e.g., AI → Engineering)
4. The goal is: Given two or more employees, return the closest (lowest) common group that is a parent of all their groups

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