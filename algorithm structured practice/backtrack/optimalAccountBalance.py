"""
0 -> 1, 10 == 0: -10
2 -> 0, 5 == 2: -5

so net balance for each one becomes
0: -5
1: 10
2: -5

The core problem is how to make everyone to be 0. Take this debit into consideration

dfs(0), debts = [-3, 3, -2, 2]                dfs(0)
start = 0, debts[0] = -3

└── try j = 1 (3)
    debts[1] = 3 + (-3) = 0
    -> dfs(1), debts = [-3, 0, -2, 2].         1+ dfs(1)

        start = 1
        while skip 0 -> start = 2, debts[2] = -2

        └── try j = 3 (2)
            debts[3] = 2 + (-2) = 0
            -> dfs(3), debts = [-3, 0, -2, 0]， 1 + dfs(3)

                start = 3
                while skip 0 -> start = 4
                start == len(debts)
                return 0

            second level gets: 1 + 0 = 1
            backtrack debts[3] to 2
            perfect match, break           所以dfs(1) @ start = -2 不用再看别的选择了
            return 1

    first level gets: 1 + 1 = 2
    backtrack debts[1] to 3
    perfect match, break 所以 dfs（0) 不需要再看 j=3 (2) 这个选择了
    return 2


"""
from collections import defaultdict


class Solution(object):
    def getDebts(self, transactions):
        # get everyone's balance
        balance = defaultdict(int)
        for trans in transactions:
            from_ = trans[0]
            to_ = trans[1]
            amount = trans[2]

            balance[from_] -= amount
            balance[to_] += amount

        # we just need to handle those are not zero debt or it is already settled
        debts = [v for v in balance.values() if v != 0]
        return debts

    def dfs(self, start, debts):
        # 只处理没有归零的debt
        while start < len(debts) and debts[start] == 0:
            start += 1

        # 如果处理到底了 那就是处理完了
        if start == len(debts):
            return 0

        operations = float("inf")
        # 不然就是对这个当下的debt去看其后面的pair 是否可以互相抵消
        for i in range(start + 1, len(debts)):
            # 只有正负才能抵消 所以判断符号
            if debts[i] * debts[start] >= 0:
                continue

            # 更新debts 的balance
            debts[i] += debts[start]
            result = 1 + self.dfs(start + 1, debts)

            operations = min(operations, result)

            # backtrack
            debts[i] -= debts[start]

            # prune, 如果此刻已经是perfect pair了那对于这个start 不用再去尝试接下去的可能性了
            if debts[i] + debts[start] == 0:
                break

        return operations

    def minTransfers(self, transactions):

        debts = self.getDebts(transactions)
        return self.dfs(0, debts)