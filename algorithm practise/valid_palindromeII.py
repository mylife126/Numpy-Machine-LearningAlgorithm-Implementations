"""

680. Valid Palindrome II
Solved
Easy
Topics
conpanies icon
Companies
Given a string s, return true if the s can be palindrome after deleting at most one character from it.

 

"""

"""
逻辑是这样的
对于不需要删除的string aba 这样的clean数据，我们就是按照palindrome来做，
用双指针来看对应前后的char是否一致

判断方法就是init left =0， right = len -1 然后不断递增递减

那么当出现了需要删除char的情况
他要么是删除左边一个， 要么是删除右边一个。

例如acbba, left走到c， right走到b， 那么假设这个原来string是palindrome，且允许删除一个char，我们就assume 跳过c， 去看从c之后的substring，这里的substring其实就是bb， 因为right不变。 为什么不变，或者说不是从 len-1重新看，是因为之前的a 和 a都看过了，只要接下来的所有剩余的substring满足了palindrome，就可以了。 

所以看到bb的时候，也就是 left+1 right不变的情况下， 判断True

同理删除右边的时候
abbca
是一个道理
"""

class Solution(object):
    def is_palindrome(self, left, right, s):
        while left < right:
            left_char = s[left]
            right_char = s[right]
            if left_char != right_char:
                return False
            left += 1
            right -=1
        return True

    def validPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        
        # [acbba]
        # [abbca]
        # [abdbca]

        if s=="" or s==" ":
            return True

        left = 0
        right = len(s) -1 
        while left < right:
            left_char = s[left]
            right_char = s[right]

            if left_char != right_char:
                # [acbba]-> scan b_left,b instead either true or false
                skip_left = self.is_palindrome(left+1, right, s)
                # [abbca] -> scan b,b_right
                skip_right = self.is_palindrome(left, right-1, s)
                return skip_left or skip_right
            
            left+=1
            right-=1
        return True  