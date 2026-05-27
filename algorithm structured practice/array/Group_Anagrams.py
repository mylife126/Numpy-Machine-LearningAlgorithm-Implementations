"""
Given an array of strings strs, group the anagrams together. You can return the answer in any order.



Example 1:

Input: strs = ["eat","tea","tan","ate","nat","bat"]

Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

Explanation:

There is no string in strs that can be rearranged to form "bat".
The strings "nat" and "tan" are anagrams as they can be rearranged to form each other.
The strings "ate", "eat", and "tea" are anagrams as they can be rearranged to form each other.

逻辑是， 两个字符如果是 anagram 则代表他们一定有共同的chars， 那么如果将两个字符重新排序后 他们都对应同一个key， 例如 eat 和 tea， 他们sort后肯定都是aet

所以我们只需要1 pass循环这个strs， 每一个str 将其排序后的字符串当作一个key， 如果这个key存在过 则说明已经在之前找到过一个str和他是anagram了。

不然也没事把自己加进去。

所以我们用defualtdict来维护
"""
from collections import defaultdict
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """

        myDict = defaultdict(list)
        for word in strs:
            key = "".join(sorted(word))
            myDict[key].append(word)

        # results = []
        # for key in myDict:
        #     res = myDict[key]
        #     results.append(res)
        # 内部更快调用
        results = list(myDict.values())
        return results