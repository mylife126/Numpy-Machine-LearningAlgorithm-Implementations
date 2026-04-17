"""
Given an array of integers nums,
find for each element its minimum Manhattan distance to any other element in the array.

这里的“Manhattan distance”在一维数组中，其实就是：
d(a_i, a_j) = |a_i - a_j|

题目等价于：

对于每个元素，找到与它最接近的其他元素的绝对差值（最小的那个）。

最优解是先做sort 然后对比左右两个相邻的数字的距离大小
数学逻辑是 假设一个数组，它是sorted， a<b<c
那么  	•	|b - a| ≤ |b - c|
	•	|c - b| ≤ |c - a|

所以更快的方案是sorted array nlogn后 做左右对比即可

"""
# brutal force. 对于每一个数字，对其他的数字进行对比，然后找取其最短距离
def manhattan_distance(a, b):
    return abs(a - b)

def min_manhattan_distance(nums):
    dist = []
    for i in range(len(nums)):
        pivot = nums[i]
        its_min = float("inf")
        for j in range(len(nums)):
            if j != i:
                its_dist = manhattan_distance(pivot, nums[j])
                if its_dist < its_min:
                    its_min = its_dist

        dist.append(its_min)

    return dist

def sorted_manhattan_distance(nums):
    nums_ = sorted([(val, idx) for idx, val in enumerate(nums)], key=lambda x: x[0]) # [[val, idx]]
    each_dist = [float("inf")] * len(nums) # each positions' original min-dist, index at idx

    for i in range(len(nums)):
        if i > 0:
            current = nums_[i][0]
            current_original_idx = nums_[i][1]
            its_left = nums_[i - 1][0]
            dist = manhattan_distance(current, its_left)

            each_dist[current_original_idx] = min(dist, each_dist[current_original_idx])

        if i < len(nums) - 1:
            current = nums_[i][0]
            current_original_idx = nums_[i][1]
            its_right = nums_[i + 1][0]
            dist = manhattan_distance(current, its_right)

            each_dist[current_original_idx] = min(dist, each_dist[current_original_idx])

    return each_dist

if __name__ == "__main__":
    nums = [8, 1, 5, 10]
    print(min_manhattan_distance(nums))