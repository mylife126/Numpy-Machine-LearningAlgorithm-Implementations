"""
[[1,2,3, 4],
[5,6,7,  8],
[9,10,11,12],
[2,3,4,  5]]

1 2 3 4 -> 8 12 5, -> 4 3 2 -> 9 5
repeat 6 7, 11, 10

while matrix
first, append the first row
second, loop each remaining row's last element for row in matrix, append row.pop()
third, append the last row from the end to the first, append matrix[::-1]
forth, trace each remaining row from bottom to up, and only get the first element

then repeat the entire process

"""
class Solution(object):          
    def spiralOrder(self, matrix):
        res = []
        while matrix:
            # step 1, always get the first row, 1 2 3 4
            res += matrix.pop(0)


            # step 2, get the last element from the rest rows
            if matrix and matrix[0]:
                # 8-12-5
                for row in matrix:
                    res.append(row.pop())

            # step 3, get the last row from right to left, 4 3 2
            if matrix:
                res += matrix.pop()[::-1]

            # step 4, trace from bottom to up and get the first elemenet, 9-5
            if matrix and matrix[0]:
                for row in matrix[::-1]:
                    res.append(row.pop(0))

        return res

        