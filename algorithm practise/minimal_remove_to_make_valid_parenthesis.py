"""
1249. Minimum Remove to Make Valid Parentheses
Solved
Medium
Topics
conpanies icon
Companies
Hint
Given a string s of '(' , ')' and lowercase English characters.

Your task is to remove the minimum number of parentheses ( '(' or ')', in any positions ) so that the resulting parentheses string is valid and return any valid string.

Formally, a parentheses string is valid if and only if:

It is the empty string, contains only lowercase characters, or
It can be written as AB (A concatenated with B), where A and B are valid strings, or
It can be written as (A), where A is a valid string.

"""

"""
思维逻辑是：
首先建造一个循环为了针对于多余right parentheses来做的。 在循环里遇到left的parentheses就添加到stack，
然后遇到right了，要是合法的parentheses就必须得有一个pair的left， 所以首先看有没有left在stack里，没有的话，说明这个是多余的right 要删掉，track他的index，最后删除用

那么如果有left，那就得pop stack，这样使得stack里的left是根据right的情况动态平衡掉，

那么循环后，理论上来说此刻 left paired 应该等于 找到的right， 因为多余的right 都已经存起来了为了删除了。
那么此刻理论上来说如果还有stack不为空，则只有两个情况，
一个是有真的发生配对且多余的right删掉，则这个情况是left多出来了，那么就应该吧多余的left删掉
或者就是 根本没有发生配对 是 ))（（这种情况， 所以只需要维护一个 right found 和 left paired的变量就好了
如果 他们相等 则是第一个情况
不然就是第二个情况 return “”
"""

class Solution(object):
    def minRemoveToMakeValid(self, s):
        """
        :type s: str
        :rtype: str
        """
        not_needed_index = set()
        left_only_stack = []
        result = []
        left_poped = 0
        right_paired = 0
        for i in range(len(s)):
            char = s[i]
            if char == "(":
                # ⚠️这里添加 left parentheses 一开始没有track 他的index
                # 后来添加了index是因为如下所述的第二个错误点
                left_only_stack.append(("(", i))
            # print(char, len(left_only_stack))
            if char == ")" and left_only_stack and len(left_only_stack)!=0:
                left_only_stack.pop()
                left_poped += 1
                right_paired += 1
            # ❌ 的是以下的判断还是用if 来做，就对导致判断两次， 例如"lee(t(c)o)de)" 
            # 到了第二个right parentheses的时候第一个if 满足， pop了left
            # 然后如果下面还是用if来写 就会再次判断， 因为此时两个left都pop出去了， 就导致多添加了一个不该删除的right
            elif char == ")" and not left_only_stack and len(left_only_stack)==0:
                not_needed_index.add(i)

        # ❌点在于 这个题目不仅仅是记录right不合法parentheses，还有一个case是
        # "(a(b(c)d)" left的parentheses多于right的parentheses
        # 一开始的写法是 直接 如果 left parentheses stack不为空，则直接return “” 这个适用于s = "))(("
        # 但是对于"(a(b(c)d)" 来说 以上的循环后 其实第二个开始的left和right都开始匹配了 但是stack里还有一个left
        # 所以我们要判断的第二个地方就是匹配率。 也就是说在上面的循环结束后 如果可以匹配，那么所有的匹配的left right都匹配好了，多出来的right index也添加了
        # 所以如果还有剩下来的left，那他就是多余的left，把他的index添加进去， 所以在第一次针对right的循环时还track了left的index 就是为了此刻删除的
        # 那么第二个情况就是，完全不合法的string， 匹配率不相等却依旧有left剩余，那就是完全不合法的str

        if left_only_stack:
            if left_poped == right_paired:
                for item in left_only_stack:
                    index = item[1]
                    not_needed_index.add(index)
            else:
                return ""

        new_chars = [s[i] for i in range(len(s)) if i not in not_needed_index]
        return "".join(new_chars)