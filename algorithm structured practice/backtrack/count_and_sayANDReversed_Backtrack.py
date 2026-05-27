"""
问题本质是理解 如何做run lnegth encoding
它其实就是 1123 读出来这里面就是
两个1，一个2， 一个3，那么就该是211213

而这一题的核心就是for loop循环n次，而每次n里面用while去读之前的sqeuence，
count 永远都是1， 然后到一个char的时候就去看后面的char是不是一样 一样的话对于这个char的count +=1， index+=1 然后将其更新为current string 来提供下一次for loop里使用


"""


class Solution(object):
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """

        # 1 - 1 1 -> n=3 21 -> n4 1 2 + 1 1 -> n = 5 1 1 + 1 2 + 2 1
        # init the base case to be "1"
        current_string = "1"

        for _ in range(n - 1):

            new_current_string = ""
            index = 0

            while index < len(current_string):
                char_count = 1

                # see the chars behind this char if they are equal
                while (index + 1 < len(current_string) and current_string[index + 1] == current_string[index]):
                    char_count += 1
                    index += 1

                # construct the new string for next iteration
                new_current_string += str(char_count) + current_string[index]
                index += 1

            current_string = new_current_string

        return current_string
"""
1️⃣ 核心逻辑解析 (Intuition)

⸻

❌ 先讲一个容易误解的点

你现在这题很多人会第一反应：

“我是不是可以直接反推？”

👉 ❌ 不行，因为：

count-and-say 是“信息压缩”
→ 是多对一映射
→ 反向一定是“多解问题”


⸻

⭐ 核心思维（一句话）

把当前字符串当作“编码结果”，去枚举所有可能的“解码方式”


⸻

🧠 你这题真正本质（非常关键）

⸻

正向：

"2223" → "32 13" → "3213"

👉 encoding 是：

(count, digit)


⸻

⸻

🔥 反向（关键）

现在给：

S = "123"


⸻

👉 你要做的是：

把 S 拆成 (count, digit) 对
然后展开


⸻

⸻

❗核心难点（这题最重要）

⸻

⚠️ S 可以有多种拆法

"123"

可以拆：

(1,2), (3,?) ❌

或者：

(12,3)


⸻

👉 所以：

这是一个 DFS / backtracking 问题


⸻

🔥 正确思路（你必须这样讲）

⸻

Step1：枚举所有 possible split

⸻

每次你要做：

选一个长度作为 count
后面一个 digit


⸻

例如：

S = "123"


⸻

split1：

count = 1
digit = 2

剩下：

"3"

👉 ❌ invalid（剩余长度不够）

⸻

⸻

split2：

count = 12
digit = 3


⸻

👉 生成：

"3" * 12


⸻

⸻

split3（另一种 valid）

(1,2),(1,3)

👉

"2" + "3"


⸻

⸻

⭐ 核心一句话

DFS 枚举所有 (count, digit) 分组方式


⸻

2️⃣ 面试讲解版算法（英文）

The reverse count-and-say is not a one-to-one mapping, so we need to explore all possible ways to split the string into (count, digit) pairs. For each valid split, we expand it and recursively process the remaining string. This naturally leads to a backtracking solution.

⸻

3️⃣ Python 完整实现 (Code)

class Solution(object):
    def reverseCountAndSay(self, s):

        result = []

        # ----------------------------------------
        # DFS function
        # ----------------------------------------
        def dfs(index, current_string_builder):

            # base case
            if index == len(s):
                result.append("".join(current_string_builder))
                return

            # ----------------------------------------
            # try all possible splits for count
            # ----------------------------------------
            for end in range(index + 1, len(s)):

                count_string = s[index:end]
                digit = s[end]

                # skip leading zero
                if count_string[0] == '0':
                    continue

                count = int(count_string)

                # expand
                expanded = digit * count

                # choose
                current_string_builder.append(expanded)

                # recurse
                dfs(end + 1, current_string_builder)

                # backtrack
                current_string_builder.pop()

        dfs(0, [])

        return result


⸻

4️⃣ 算法 Walkthrough（用你例子）

⸻

输入：

s = "123"


⸻

DFS 树

index=0

├── (1,2)
│   └── index=2 → invalid（剩一个字符）
│
└── (12,3)
    └── index=3 → valid
        → "3"*12


⸻

输出：

["333333333333"]


⸻

再看一个更直观例子

⸻


s = "1112"


⸻

DFS 树：

(1,1),(1,1),(1,2) → "112"
(11,1),(2,?) ❌


⸻

👉 输出：

["112"]


⸻

5️⃣ Complexity

⸻

Time Complexity

O(2^n)

👉 因为要枚举所有 split

⸻

Space Complexity

O(n)


⸻

6️⃣ Edge Cases（面试可以跑）

⸻

✅ Case1

s = "11"
# 输出: ["1"]


⸻

✅ Case2

s = "21"
# 输出: ["11"]


⸻

✅ Case3

s = "123"
# 多种可能


⸻

⚠️ Case4（invalid）

s = "1"
# 无法构成 count+digit → []


⸻

⚠️ Case5（leading zero）

s = "01"
# skip


⸻

🎯 最后一行总结（面试）

This is a backtracking problem where we try all possible ways to split the string into count-digit pairs and reconstruct the original strings.

"""

class Solution(object):
    def reverseCountAndSay(self, s):

        result = []

        # ----------------------------------------
        # DFS function
        # ----------------------------------------
        def dfs(index, current_string_builder):

            # base case
            if index == len(s):
                result.append("".join(current_string_builder))
                return

            # ----------------------------------------
            # try all possible splits for count
            # ----------------------------------------
            for end in range(index + 1, len(s)):

                count_string = s[index:end]
                digit = s[end]

                # skip leading zero
                if count_string[0] == '0':
                    continue

                count = int(count_string)

                # expand
                expanded = digit * count

                # choose
                current_string_builder.append(expanded)

                # recurse、
                # 重点 这里对比 accountBalance那道题的传入不一样，这里是end + 1
                # 理由是我们的dfs做的事情是 count = string[start, end], 然后数字本体是string【end】
                # 那么这一层这个end被用掉了，就不能再在下一层被当作count的开头了
                # 123， 你如果选择了1作为count， 2作为数字， 那剩下没处理的就是3 不能是 23！
                # 但是account balence 【1，1， -1， -1】 这里我从1开始 我要尝试和我之后的所有人做比对 所以传入的是 start + 1
                dfs(end + 1, current_string_builder)

                # backtrack
                current_string_builder.pop()

        dfs(0, [])

        return result

if __name__ == '__main__':
    solution = Solution()
    print(solution.reverseCountAndSay("1112"))