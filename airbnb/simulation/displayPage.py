"""
Description

You are given a list of search results represented as CSV strings.
Each string represents a listing in the format:
"host_id,listing_id,score,city"

Task

You need to display the results in pages.
	•	Each page can contain at most pageSize results (e.g., 12).
	•	You want to avoid showing the same host more than once per page, if possible.
	•	If it is not possible (i.e., not enough unique hosts remain), you may include duplicate hosts.

⸻

Output Requirements
	•	Return the reordered list of results.
	•	Also group them into pages (for printing or visualization).
注意这里面有一个score 也就是每一个list的本身的优先级！ 也就是说在每一个page里 如果遇到了duplicates 且 page listing count不够的时候 不能
把后面的list放进来 而是应该这个时候再把重复的放进去！

⸻

Example

Input:
pageSize = 3

results = [
 "1,28,300.1,San Francisco",
 "4,5,209.1,San Francisco",
 "20,7,208.1,San Francisco",
 "23,8,207.1,San Francisco",
 "16,10,206.1,Oakland",
 "1,16,205.1,San Francisco",
 "6,29,204.1,San Francisco",
 "7,20,203.1,San Francisco",
 "8,21,202.1,San Francisco",
 "2,18,201.1,San Francisco",
 "2,30,200.1,San Francisco",
 "15,27,109.1,Oakland"
]

Output:
Page 1:
1,28,300.1,San Francisco
4,5,209.1,San Francisco
20,7,208.1,San Francisco

Page 2:
23,8,207.1,San Francisco
16,10,206.1,Oakland
1,16,205.1,San Francisco

Page 3:
6,29,204.1,San Francisco
7,20,203.1,San Francisco
8,21,202.1,San Francisco

Page 4:
2,18,201.1,San Francisco
15,27,109.1,Oakland
2,30,200.1,San Francisco

Explanation
	•	Page 1: all unique hosts ✅
	•	Page 2: host 1 appears again (allowed, new page)
	•	Page 4: host 2 appears twice because no alternative hosts left

⸻

Constraints
	•	1 <= results.length <= 10^4
	•	1 <= pageSize <= 100
	•	Each result is a valid CSV string
	•	Input is already sorted by score

思路：
用queue来维护一个remaining 变量， 目标是存储所有剩余的list
同时每一个page loop里维护一个temp的变量 来存储已经重复过的list，当page list counts不足的时候，才从temp里拿出来重复的list。 注意！ 这里不可以
把remaining后面的none duplicate拿进去 因为listing有自己的score 我们还得确保listing的排序性！


"""
from collections import deque

def display_pages(results, page_size):
    # get all the listings into the queue
    remaining = deque(results)
    page_result = []

    while remaining:
        page = []
        temp = deque()
        used_list = set()

        # step 1: fill the unique listing first
        while remaining and len(page) < page_size:
            entry = remaining.popleft()
            host_id = entry.split(",")[0]

            if host_id not in used_list:
                used_list.add(host_id)
                page.append(entry)
            else:
                temp.append(entry)

        # step 2: only fill with the duplicates when the page counts is still less than the page_size
        while len(page) < page_size and temp:
            page.append(temp.popleft())

        # step 3: honor the ranking score by putting the duplicated but top ranked listing back to the remaining
        temp.extend(remaining)
        remaining = temp

        page_result.append(page)

    return page_result

if __name__ == "__main__":
    pageSize = 3

    results = [
        "1,28,300.1,San Francisco",
        "4,5,209.1,San Francisco",
        "20,7,208.1,San Francisco",
        "23,8,207.1,San Francisco",
        "16,10,206.1,Oakland",
        "1,16,205.1,San Francisco",
        "6,29,204.1,San Francisco",
        "7,20,203.1,San Francisco",
        "8,21,202.1,San Francisco",
        "2,18,201.1,San Francisco",
        "2,30,200.1,San Francisco",
        "15,27,109.1,Oakland"
    ]

    print(display_pages(results, pageSize))



