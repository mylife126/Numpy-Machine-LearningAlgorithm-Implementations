"""
14. Longest Common Prefix
Solved
Easy
Topics
conpanies icon
Companies
Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".

 

Example 1:

Input: strs = ["flower","flow","flight"]
Output: "fl"
Example 2:

Input: strs = ["dog","racecar","car"]
Output: ""
Explanation: There is no common prefix among the input strings.
"""
class Solution(object):
    def longestCommonPrefix(self, strs):
        if not strs:
            return ""
        
        shortest_string = len(strs[0])
        for str_ in strs:
            if len(str_) < shortest_string:
                shortest_string = len(str_)

        i = 0 
        stopper = False
        while i < shortest_string:
            prefix = ""
            for idx in range(len(strs)):
                the_string = strs[idx]
                if idx == 0:
                    prefix = the_string[i]
                    continue
                comparing_prefix = the_string[i]
                if comparing_prefix != prefix:
                    stopper = True
                    break
                
            if stopper:
                break

            i+=1
        if i == 0:
            return ""
        else:
            return strs[0][0:i]