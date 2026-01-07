"""
791. Custom Sort String
Solved
Medium
Topics
conpanies icon
Companies
You are given two strings order and s. All the characters of order are unique and were sorted in some custom order previously.

Permute the characters of s so that they match the order that order was sorted. More specifically, if a character x occurs before a character y in order, then x should occur before y in the permuted string.

Return any permutation of s that satisfies this property."
"""

from collections import Counter

class Solution(object):
    def customSortString(self, order, s):
        """
        order: str  # 排序规则（字符唯一）
        s: str      # 需要被重新排序的字符串
        return: str
        """

        # 1️⃣ 统计 s 中每个字符的出现次数
        char_count = Counter(s)  # {char: freq}

        result = []

        # 2️⃣ 按 order 的顺序输出字符
        for ch in order:
            if ch in char_count:
                while char_count[ch] > 0:
                    result.append(ch)
                    char_count[ch] -= 1

        # 3️⃣ 输出剩余（order 中没出现的字符）
        for ch in char_count:
            while char_count[ch] > 0:
                result.append(ch)
                char_count[ch] -= 1

        return "".join(result)