"""
这道题目很类似于shortestWayToFormString，都是给了两个string，
A = abc， B=abcab等等，
问你A是否是B的一个subsequence

但是重点区别是， 这一题只是subsequence， 而上一题的约束则是A里面可以multi turn来看是否可以拆出几个substring来组成B，所以
这一题是one pass， 而上一题是multipass

但做法都是双指针一个一个看过去。只是差别在于一个是single while 一个是double while

解说：

你可以把这两类题放在脑子里对比：

⸻

🧠 第一类（你刚刚这题 isSubsequence）

s = "abc"
t = "ahbgdc"

你在做的事情是：

👉 检查 s 能不能嵌在 t 里

⸻

这个过程有个很重要的限制：

t 只能往前走，不能回头


⸻

所以你的思路应该是：

👉 我拿着 t，一路往右扫

每看到一个字符：
    如果是我想要的 s[i] → 消耗掉（i++）


⸻

👉 所以：

t 只能扫一遍

👉 不能写成两层 while

⸻

🚨 如果你写两层 while，会发生什么？

👉 你相当于在说：

我可以反复用 t

👉 但这是不允许的 ❌

⸻

⸻

🧠 第二类（你之前的题：shortestWay）

我们拿这个来看：

source = "abc"
target = "abcbc"


⸻

你在做的事情是：

👉 我可以反复使用 source

⸻

也就是说：

第一轮用 source
第二轮再用 source
第三轮再用 source


⸻

👉 这时候就变成：

外层 while：控制“用第几次 source”
内层 while：扫描一整遍 source


⸻

👉 所以自然就是：

两个 while


⸻

🎯 核心区别（最重要）

你现在要记住这个判断：

⸻

❓这个字符串能不能被“重复使用”？

⸻

✅ 可以重复用（shortestWay）

source 可以用多次

👉 所以你会：

扫一遍 source → 不够 → 再扫一遍

👉 → 两层 while

⸻

❌ 不能重复用（isSubsequence）

t 只能用一次

👉 所以你只能：

一口气扫完 t

👉 → 一个 while

⸻

🔥 用一句话帮你彻底区分

👉

能不能“重来一遍”决定你用几个 while


⸻

🧠 再用你的语言总结一下（建议你记这个）

👉

如果是像 shortestWay，那我每一层都可以重新从 source 开始扫，
所以是 while + while

但如果是 isSubsequence，我只能在 t 上往前走一次，
不能回头，也不能重来，所以只能用一个 while


⸻

🚀 再帮你强化一个对比（非常重要）

⸻

🟢 shortestWay

target: abcbc
source: abc

👉 行为：

扫 abc → 得到 abc
扫 abc → 得到 bc

👉 source 被用多次

⸻

🔴 isSubsequence

s: abc
t: ahbgdc

👉 行为：

只扫一遍 t

👉 t 不能重复用
"""


class Solution(object):
    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        chars_from_s = set(s)
        chars_from_t = set(t)

        for char in chars_from_s:
            if char not in chars_from_t:
                return False

        i = 0
        j = 0
        while i < len(s) and j < len(t):
            char_s = s[i]
            char_t = t[j]
            if char_s == char_t:
                i += 1
            j += 1

        if i == len(s):
            return True

        return False