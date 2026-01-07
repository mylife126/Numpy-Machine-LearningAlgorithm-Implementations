"""
1704. Determine if String Halves Are Alike
Solved
Easy
Topics
conpanies icon
Companies
Hint
You are given a string s of even length. Split this string into two halves of equal lengths, and let a be the first half and b be the second half.

Two strings are alike if they have the same number of vowels ('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'). Notice that s contains uppercase and lowercase letters.

Return true if a and b are alike. Otherwise, return false.
"""

class Solution(object):
    def halvesAreAlike(self, s):
        """
        :type s: str
        :rtype: bool
        """
        vowels = ["a", "e", "i", "o", "u", "A", "E", "I", "O", "U"]
        vowels = set(vowels)
        half = len(s) // 2

        left = s[0:half]
        right = s[half::]
        # print(left, right)
        count_left = 0 
        count_right = 0
        for char in left:
            if char in vowels:
                count_left +=1 
        
        for char in right:
            if char in vowels:
                count_right += 1

        if count_left == count_right:
            return True
        else:
            return False