"""
Description

You are given a user’s travel history and a list of other users’ travel histories.

Each user has a set of cities they have visited.

⸻

Definitions
	•	A travel buddy is a user whose travel history has at least 50% overlap with the given user.
	•	The overlap is defined as: (common cities) / (total cities of the given user)

Task
	1.	Identify all travel buddies.
	2.	Sort the travel buddies by number of common cities (descending).
	3.	Recommend cities to the given user:
	•	Cities that the buddy has visited
	•	But the given user has not visited
	4.	Return the recommended cities in priority order (based on buddy ranking)


Example
Input:
user = ["Paris", "London", "NYC", "Dubai"]

friends = {
  "Alice": ["Paris", "London", "Berlin"],
  "Bob": ["London", "NYC", "Dubai", "Tokyo"],
  "Charlie": ["Madrid", "Rome"]
}
Step 1: compute overlap
Alice: 2 / 4 = 0.5  ✅
Bob:   3 / 4 = 0.75 ✅
Charlie: 0 / 4 = 0 ❌


Step 2: sort the buddy
Bob (3 common)
Alice (2 common)

Step 3: recommendations
Bob → Tokyo
Alice → Berlin


Return : ["Tokyo", "Berlin"] Tokyo is from Bob whose priority is higher and Berlin is from Alice whose priority is lower.


思路是：
用set来维护user的cities 和 friend的cities 这样union 的做法很简单就是 setA & setB
然后根据重合率排序friends

接着维护两个变量一个是推荐城市 一个是 seen city
for loop 每一个friend， 将它城市推荐给user 当且当这个city不存在user city set里且没有被之前的人推荐过
"""
class Solution:
    def matched(self, overlapped, user_size):
        """
        这个match的标准是基于原题
        Definitions
        •	A travel buddy is a user whose travel history has at least 50% overlap with the given user.
        •	The overlap is defined as: (common cities) / (total cities of the given user)
        """
        return overlapped >= user_size / 2

    def matched_2(self, overlapped, friend_cities_size):
        """
        This definition changed to be "a travel buddy is when the overlapped cities is over 50% of the friend's visited cities.
        """
        # now returns boolean and the similarity score which is used for ranking
        return overlapped >= friend_cities_size / 2, overlapped / friend_cities_size

    def findRecommendations(self, user, friends):
        user_set = set(user)
        user_size = len(user_set)

        buddies = []

        for name, friend_cities in friends.items():
            friend_set = set(friend_cities)

            overlap = user_set & friend_set
            common_count = len(overlap)

            if self.matched(common_count, user_size):
                buddies.append((name, common_count, friend_set))

        # sort by overlap descending
        buddies.sort(key=lambda x: x[1], reverse=True)

        seen = set()
        recommendations = []

        for name, _, friend_set in buddies:
            for city in friend_set:
                if city not in user_set and city not in seen:
                    recommendations.append(city)
                    seen.add(city)

        return recommendations

if __name__ == '__main__':
    user = ["Paris", "London", "NYC", "Dubai"]

    friends = {
        "Alice": ["Paris", "London", "Berlin"],
        "Bob": ["London", "NYC", "Dubai", "Tokyo"],
        "Charlie": ["Madrid", "Rome"]
    }

    solution = Solution()
    print(solution.findRecommendations(user, friends))
