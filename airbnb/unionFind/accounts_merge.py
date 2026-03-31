"""
Given a list of accounts where each element accounts[i] is a list of strings, where the first element accounts[i][0] is a name, and the rest of the elements are emails representing emails of the account.

Now, we would like to merge these accounts. Two accounts definitely belong to the same person if there is some common email to both accounts. Note that even if two accounts have the same name, they may belong to different people as people could have the same name. A person can have any number of accounts initially, but all of their accounts definitely have the same name.

After merging the accounts, return the accounts in the following format: the first element of each account is the name, and the rest of the elements are emails in sorted order. The accounts themselves can be returned in any order.



Example 1:

Input: accounts = [["John","johnsmith@mail.com","john_newyork@mail.com"],["John","johnsmith@mail.com","john00@mail.com"],["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]]
Output: [["John","john00@mail.com","john_newyork@mail.com","johnsmith@mail.com"],["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]]
Explanation:
The first and second John's are the same person as they have the common email "johnsmith@mail.com".
The third John and Mary are different people as none of their email addresses are used by other accounts.
We could return these lists in any order, for example the answer [['Mary', 'mary@mail.com'], ['John', 'johnnybravo@mail.com'],
['John', 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com']] would still be accepted.
"""

"""
这道题目其实要求的是， 给了人的名字和他对应的email 账号， 要求你把属于这个人的所有账号找到并且
放在一起。

accounts = [records1,....recordsn], 其中records = 【name， email1, email2....】

这里有一个核心的地方是， 每一个record 都是 人名 和 已经对应的联通的账号， 那么如果record1 和 record2是同一个人，他们两个record间一定有一个一样的账户当作bridge.

基于这个，其实就是grouping的work，Union Find非常适合。 每次union的时候，都是用每一个record里的emails 做union， 算法自己track了上一个record里 （假设有某一个同样账户）的parent， 则在这一次union里会assign到同一个parent。 

但由于union find的设计，是用array of index来表达每一个node， 所以我们只需要提前设计好email to index， 以及email to user name即可。

for loop 每一个record里面的email 作union，最后traverse每一个email来getRoot 即可。

"""
from collections import defaultdict


class UF():
    def __init__(self, n):
        """
        n is the size of the nodes
        """
        # init the parents to each node to be themselves
        self.parent = [i for i in range(n)]

        # init each parent root's child size to be 1
        self.size = [i for i in range(n)]

    def getRoot(self, node):
        # here we use path compression to assign the root to a node
        while node != self.parent[node]:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]

        return node

    def union(self, a, b):
        rootA = self.getRoot(a)
        rootB = self.getRoot(b)

        if rootA != rootB:
            if self.size[rootA] >= self.size[rootB]:
                self.parent[rootB] = rootA
                self.size[rootA] += self.size[rootB]

            else:
                self.parent[rootA] = rootB
                self.size[rootB] += self.size[rootA]


class Solution(object):
    def accountsMerge(self, accounts):
        # ----------------------------------------
        # Step1: 给每个 email 分配一个 index
        # ----------------------------------------
        email_to_index = {}
        email_to_name = {}
        index_to_email = {}

        # this tracks the total numbers of emails
        index = 0
        for account in accounts:
            name = account[0]
            emails = account[1:]

            for email in emails:
                if email not in email_to_index:
                    email_to_index[email] = index
                    index_to_email[index] = email
                    email_to_name[email] = name
                    index += 1

        # ----------------------------------------
        # Step2: 初始化 Union Find
        # ---------------------------------------
        uf = UF(index)

        # ----------------------------------------
        # Step3: union 同一 account 内的 email
        # ----------------------------------------
        for account in accounts:
            first_email = account[1]
            first_email_index = email_to_index[first_email]

            for other_email in account[2:]:
                uf.union(first_email_index, email_to_index[other_email])

        # ---------------------------------------
        # Step 4: get the groups for every email
        # ---------------------------------------
        groups = defaultdict(list)
        for email in email_to_index:
            its_root_index = uf.getRoot(email_to_index[email])
            root_name = index_to_email[its_root_index]
            groups[root_name].append(email)

        results = []
        for root in groups:
            its_emails = sorted(groups[root])
            name = email_to_name[root]
            results.append([name] + its_emails)

        return results  