## Sliding Window 的本质

Sliding window 本质是对暴力双重循环的优化。当问题满足**单调性**（窗口扩大使某个值变大/变坏，窗口缩小使某个值变小/变好），left 指针就不需要回退，从 O(n²) 降到 O(n)。

## 固定模板

```python
def sliding_window(nums, condition):
    left = 0
    window_state = ...  # 维护窗口内的状态（sum、product、counter等）
    res = ...
    
    for right in range(len(nums)):
        # 1. 扩展：右边元素加入窗口
        update_window(nums[right])
        
        # 2. 收缩：不满足条件时，踢掉左边元素
        while not valid(window_state):
            remove_from_window(nums[left])
            left += 1
        
        # 3. 记录答案
        res = update_answer()
    
    return res
```

三步走：**扩展 → 收缩 → 记录**。

## 什么时候能用 Sliding Window？

关键判断：**窗口的"好坏"是否单调？**

- 乘积 < k → 窗口越大乘积越大 ✓ 单调
- 子数组和 <= target → 窗口越大和越大 ✓ 单调
- 最多包含 k 个不同字符 → 窗口越大字符种类越多 ✓ 单调
- 子数组最大值 - 最小值 <= k → ✓ 单调（但需要额外数据结构）

不能用的情况：数组有负数时求子数组和（扩大窗口可能变小也可能变大，没有单调性）。

## 常见变体

| 题型 | window_state | 记录答案方式 |
|---|---|---|
| 乘积 < k | product | `res += right - left + 1` |
| 和 <= target | sum | `res += right - left + 1` |
| 最长子串无重复 | set 或 dict | `res = max(res, right - left + 1)` |
| 最小长度子数组 >= target | sum | `res = min(res, right - left + 1)` |

## 和 Binary Search 的对比

| | Binary Search | Sliding Window |
|---|---|---|
| 前提条件 | 数据有序 | 窗口状态有单调性 |
| 核心操作 | 每次砍掉一半 | left 只往右走不回退 |
| 时间复杂度 | O(log n) | O(n) |
| 模板结构 | while left <= right + 三分支 | for right + while 收缩 left |
| 解决的问题 | 在有序数据中找目标 | 在连续子数组中找满足条件的 |

两者的共同点：都利用了**单调性**来避免暴力枚举。Binary search 利用有序性每次排除一半，sliding window 利用窗口单调性保证 left 不回退。