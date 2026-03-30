"""
🧩 Round Price (Minimize Rounding Error)

Medium / Hard（Airbnb高频）

⸻

Description

You are given a list of floating-point prices.

You need to round each price to either its floor or ceiling integer value.

⸻

Constraints
	•	The sum of the rounded prices must equal:
	round(sum(original prices))

Objective

Return the rounded integer values such that:
1. The total sum constraint is satisfied
2. The total rounding error is minimized

Round error definition
error = abs(rounded_value - original_value)

-------------
Example 1

Input:prices = [1.2, 2.3, 3.4]
Step 1: compute target sum -> sum = 1.2 + 2.3 + 3.4 = 6.9 -> target = round(6.9) = 7
Step 2: possible rounding :
floor: [1,2,3] → sum = 6
ceil : [2,3,4] → sum = 9

Step 3: choose combination

We need sum = 7 -> output to be [1,2,4]

思路是：
首先round original sum 知道target是多少， 接着首先将每一个数字作floor 处理 得到一个floor sum
这样 还缺少的 numbers of number to be ceiled = round(original_sum) - floor_sum

for example target = 7, then, floored sum = 1 + 2 + 3 = 6 so, we need 1 number to be ceiled

那么intuitively 我们只需要sort 原数组 by their floor fraction， 因为 floor error 越大 则这个数字roundup 的时候 error就会越低，
这样优先ceil round up error 低的就好了
"""
class Solution(object):
    def minimizeRoundingError(self, nums):
        """
        nums: List[float]
        return: List[int] (rounded result)
        """

        total_sum = sum(nums)
        total_sum = int(round(total_sum))

        floor_sum = 0
        floor_error = []
        floor_numbers = []
        for i in range(len(nums)):
            this_num = nums[i]
            floor_this_num = int(this_num)
            floor_numbers.append(floor_this_num)

            floor_sum += floor_this_num
            floor_error.append((this_num - floor_this_num, i))

        floor_error = sorted(floor_error, key=lambda x: x[0], reverse=True)
        needed_number = total_sum - floor_sum

        for idx in range(needed_number):
            which_index = floor_error[idx][1]
            floor_numbers[which_index] += 1

        return floor_numbers

if __name__ == '__main__':
    solution = Solution()
    print(solution.minimizeRoundingError([1.0, 2.0, 3.0]))
