class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """

        x_str = str(x)
        result = 1
        stack = []
        for char in x_str:
            if char == "-":
                result = -1

            else:
                stack.append(char)

        ans = ""
        while stack:
            ans += stack.pop()

        ans = result * int(ans)
        if ans < -2 ** 31 or ans > 2 ** 31 - 1:
            return 0

        return ans


class Solution(object):
    def reverse(self, x):

        # ----------------------------------------
        # Step1: determine sign
        # ----------------------------------------
        sign = -1 if x < 0 else 1

        # work with positive number
        x = abs(x)

        reversed_number = 0

        # ----------------------------------------
        # Step2: reverse digits
        # ----------------------------------------
        while x > 0:

            # extract last digit
            digit = x % 10

            # remove last digit
            x = x // 10

            # ----------------------------------------
            # Step3: overflow check BEFORE append
            # ----------------------------------------
            if (reversed_number > 214748364 or
                    (reversed_number == 214748364 and digit > 7)):
                return 0

            # append digit
            reversed_number = reversed_number * 10 + digit

        return sign * reversed_number