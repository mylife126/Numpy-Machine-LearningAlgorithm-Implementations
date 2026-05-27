"""

## 从头讲起

### 什么是二叉树 (Binary Tree)？

```
        1
       / \
      2    3
     / \  / \
    4   5 6   7
```

每个节点最多有两个孩子：左孩子和右孩子。

### 什么是 Preorder（前序遍历）？

遍历顺序：**根 → 左 → 右**

对上面的树：
1. 访问根 `1`
2. 递归访问左子树 `2 → 4 → 5`
3. 递归访问右子树 `3 → 6 → 7`

结果：`[1, 2, 4, 5, 3, 6, 7]`

### 什么是 Postorder（后序遍历）？

遍历顺序：**左 → 右 → 根**

对上面的树：
1. 递归访问左子树 `4 → 5 → 2`
2. 递归访问右子树 `6 → 7 → 3`
3. 最后访问根 `1`

结果：`[4, 5, 2, 6, 7, 3, 1]`

### 两者的关系

| | Preorder | Postorder |
|---|---|---|
| 根的位置 | **最前面** | **最后面** |
| 顺序 | 根→左→右 | 左→右→根 |

---

## 递归怎么理解？

递归的核心思想：**把大问题拆成相同结构的小问题**。

别想"它怎么跑的"，先想"每一层做了什么"。

这道题每一层递归只做一件事：**确定当前的根是谁，然后把数组切成左子树和右子树两部分，交给下一层去处理。**

### 用最简单的例子走一遍

假设树只有 3 个节点：

```
    1
   / \
  2   3
```

preorder = `[1, 2, 3]`，postorder = `[2, 3, 1]`

**第一层递归调用：**
- preorder = `[1, 2, 3]`，postorder = `[2, 3, 1]`
- 根 = preorder[0] = `1`
- 左子树的根 = preorder[1] = `2`
- 在 postorder 中找 `2` → index = 0，所以左子树大小 = 0 + 1 = **1**
- 切分：
  - 左子树：preorder = `[2]`，postorder = `[2]`
  - 右子树：preorder = `[3]`，postorder = `[3]`
- 然后递归处理左右

**第二层递归（左子树）：**
- preorder = `[2]`，只有一个元素
- 创建节点 `2`，没有孩子，直接返回

**第二层递归（右子树）：**
- preorder = `[3]`，只有一个元素
- 创建节点 `3`，没有孩子，直接返回

**回到第一层：**
- `node(1).left = node(2)`
- `node(1).right = node(3)`
- 树建好了！

---

### 递归的思维方式

不要试图在脑子里展开所有层。只需要相信：

> "如果我给递归函数一个正确的 preorder 和 postorder 子数组，它会返回给我一棵正确的子树。"

你只需要关心**当前这一层**做什么：
1. 取出根（preorder 第一个）
2. 算出左子树有多大（用 preorder[1] 在 postorder 中的位置）
3. 切分数组，把左半部分和右半部分分别交给递归

**Base case（递归终止条件）：**
- 数组为空 → 返回 None（没有节点）
- 数组只有 1 个元素 → 返回一个叶子节点

这就是递归的全部。每一层只管自己的事，把子问题交给下一层.
"""

# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

"""
树结构:
        1
       / \
      2    3
     / \  / \
    4   5 6   7

preorder = [1,2,4,5,3,6,7], postorder = [4,5,2,6,7,3,1]

核心规律:
- preorder:  根 → 左子树 → 右子树 (根在最前)
- postorder: 左子树 → 右子树 → 根 (根在最后)

算法步骤:
1. root = preorder[0] (当前根)
2. left_root = preorder[1] (左子树的根)
3. left_size = postorder.index(left_root) + 1 (左子树节点总数)
4. 切分:
   - preorder  for left  = preorder[1 : 1 + left_size]
   - preorder  for right = preorder[1 + left_size :]
   - postorder for left  = postorder[0 : left_size]
   - postorder for right = postorder[left_size : -1]  (注意-1，排除当前根)
5. 递归处理左右子树

示例 - 第一层:
  root = 1, left_root = 2
  left_size = postorder.index(2) + 1 = 3
  left  preorder = [2,4,5],  left  postorder = [4,5,2]
  right preorder = [3,6,7],  right postorder = [6,7,3]

示例 - 第二层 (左子树 [2,4,5]):
  root = 2, left_root = 4
  left_size = postorder.index(4) + 1 = 1
  left  preorder = [4],  left  postorder = [4]
  right preorder = [5],  right postorder = [5]

Base case: len(preorder) == 1 → 叶子节点，直接返回自己
"""

class Solution:
    def constructFromPrePost(self, preorder, postorder):
        
        # to handle the leaf node, their recurssion to find left and right is none
        if not preorder or len(preorder) == 0:
            return None

        root_value = preorder[0]
        root = TreeNode(root_value)

        # return condition the base case
        if len(preorder) == 1:
            return root
        

        # else we need to solve the slice for left and right
        left_tree_root = preorder[1]
        index_where = postorder.index(left_tree_root)
        left_tree_size = index_where + 1

        # slicing
        # starting from index 1 to size +1 so we cover all the elements for left tree
        preorder_for_left = preorder[1 : left_tree_size + 1]
        postorder_for_left = postorder[0 : left_tree_size]
        
        preorder_for_right = preorder[left_tree_size + 1 : ]
        # cannot reach the last one, because the last one in postorder is the original root
        postorder_for_right = postorder[left_tree_size : -1] 
        
        root.left = self.constructFromPrePost(preorder_for_left, postorder_for_left)
        root.right = self.constructFromPrePost(preorder_for_right, postorder_for_right)

        # for the first layer's recurssion return 
        return root