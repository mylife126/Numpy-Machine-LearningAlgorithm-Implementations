"""
846. Hand of Straights

Alice has some number of cards and she wants to rearrange the cards into groups so that each group is of size groupSize, and consists of groupSize consecutive cards.

Given an integer array hand where hand[i] is the value written on the ith card and an integer groupSize, return true if she can rearrange the cards, or false otherwise."
"""

"""
思路，
so the idea is the sub group has to be consecutive, so for each group, 
always start with the smallest and remaining number. 

Since it is consecutive, the next == current + 1, and if the next is not existed, then false

Create a counter, that counts the occurence of each number. Then sort it, 

Then, while counter is not empty, for loop each group till the next is not existed return false


"""
from collections import Counter
class Solution(object):
    def isNStraightHand(self, hand, groupSize):
        """
        
        :type hand: List[int]
        :type groupSize: int
        :rtype: bool
        """
        
        #means the size cannot even be divided into K
        if len(hand) % groupSize != 0: 
            return False

        # the key is the consecutive 
        myCounter = Counter(hand)
        myCounter_sorted_keys = sorted(myCounter.keys())
        
        # {1:1, 2:2, 3:2, 4:1, 6:1, 7:1, 8:1}
        for card_value in myCounter_sorted_keys:
            while myCounter[card_value] > 0:
                # start with the smallest 
                myCounter[card_value] -= 1

                # iterate the next， range(1,4) ending 4
                for i in range(card_value + 1, card_value + groupSize):
                    next_consecutive = i
                    if myCounter[next_consecutive] <= 0:
                        return False
                    
                    myCounter[next_consecutive] -= 1

        return True
