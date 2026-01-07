'''
8. String to Integer (atoi)
Solved
Medium
Topics
conpanies icon
Companies
Implement the myAtoi(string s) function, which converts a string to a 32-bit signed integer.

The algorithm for myAtoi(string s) is as follows:

Whitespace: Ignore any leading whitespace (" ").
Signedness: Determine the sign by checking if the next character is '-' or '+', assuming positivity if neither present.
Conversion: Read the integer by skipping leading zeros until a non-digit character is encountered or the end of the string is reached. If no digits were read, then the result is 0.
Rounding: If the integer is out of the 32-bit signed integer range [-231, 231 - 1], then round the integer to remain in the range. Specifically, integers less than -231 should be rounded to -231, and integers greater than 231 - 1 should be rounded to 231 - 1.
Return the integer as the final result.

'''

class Solution(object):
    def myAtoi(self, s):
        i = 0 
        num = 0
        lower = -2**31
        upper = 2**31 - 1
        sign = 1
        
        # Skip leading whitespace
        while i < len(s) and s[i] == " ":
            i += 1
        
        # Check for sign (only once)
        if i < len(s):
            if s[i] == "-":
                sign = -1
                i += 1
            elif s[i] == "+":
                sign = 1
                i += 1

        # Convert digits to number
        for j in range(i, len(s)):
            if s[j].isnumeric():
                num = num * 10 + int(s[j])
            else:
                break

        num *= sign

        if num < lower:
            num = lower
        
        if num > upper:
            num = upper

        return num 

