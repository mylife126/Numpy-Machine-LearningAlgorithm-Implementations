"""
125. Valid Palindrome
Solved
Easy
Topics
conpanies icon
Companies
A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.

Given a string s, return true if it is a palindrome, or false otherwise.
"""

class Solution:
    def isPalindrome(self, s):
        filtered = []
        for ch in s:
            if ch.isalnum():
                filtered.append(ch.lower())

        left = 0
        right = len(filtered) - 1
        while left < right:
            its_left = filtered[left]
            its_right = filtered[right]
            if its_left != its_right:
                return False
            left +=1
            right -=1

        return True