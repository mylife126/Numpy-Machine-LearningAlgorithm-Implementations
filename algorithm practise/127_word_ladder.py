"""
127. Word Ladder
Solved
Hard
Topics
conpanies icon
Companies
A transformation sequence from word beginWord to word endWord using a dictionary wordList is a sequence of words beginWord -> s1 -> s2 -> ... -> sk such that:

Every adjacent pair of words differs by a single letter.
Every si for 1 <= i <= k is in wordList. Note that beginWord does not need to be in wordList.
sk == endWord
Given two words, beginWord and endWord, and a dictionary wordList, return the number of words in the shortest transformation sequence from beginWord to endWord, or 0 if no such sequence exists.""
"""


from collections import deque
class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """

        word_set = set(wordList)
        if endWord not in word_set:
            return 0

        word_set = set(wordList)

        # queue saves each transformed words and their steps
        queue = deque()
        queue.append((beginWord, 1))

        visited = set()
        visited.add(beginWord)

        while queue:
            word, step = queue.popleft()

            if word == endWord:
                return step

            for i in range(len(word)):
                for ch in "abcdefghijklmnopqrstuvwxyz":
                    new_word = word[:i] + ch + word[i+1:]
                    if new_word in word_set and new_word not in visited:
                        visited.add(new_word)
                        queue.append((new_word, step+1))

        return 0