class Solution(object):
    def dfs(self, memo, price, special, current_needs):
        """ This function uses backtrack to solve the monetization options via two options:
        1. first use the individual price to figure the total for fulfilling the current needs -> cost = sum(price_i * num_i)
        2. then use the offer, only when the offer is valid
        2.1 then find the remaining needs
        3. cost = offer + dfs(new_state)
        """

        # first check if the needed products has been previously
        # calculated.
        if current_needs in memo:
            its_cost = memo[current_needs]
            return its_cost

        # --------------
        # option 1: get all the products using the individual price
        # --------------
        temp_price = 0
        for i in range(len(current_needs)):
            its_price = price[i]
            its_needs_counts = current_needs[i]
            temp_price += its_price * its_needs_counts

        # assume this is the minimal cost
        minimum_cost = temp_price

        # option 2: use the special offer only if when the offer is valid meaning that for loop the bundle_i for the counts and if it is larger than the needs_i, then it is an invalid bundle offer
        for offer in special:
            if_valid = True
            new_current_needs = []
            for i in range(len(current_needs)):
                actual_need_counts = current_needs[i]
                bundle_offer_counts = offer[i]
                if bundle_offer_counts > actual_need_counts:
                    if_valid = False
                    break
                new_current_needs.append(actual_need_counts - bundle_offer_counts)

            if if_valid:
                bundle_offer = offer[-1]
                cost_using_offer = bundle_offer + self.dfs(memo, price, special, tuple(new_current_needs))
                minimum_cost = min(minimum_cost, cost_using_offer)

        memo[current_needs] = minimum_cost
        return minimum_cost

    def shoppingOffers(self, price, special, needs):
        """
        Return minimum cost to satisfy needs using individual purchase or special offers
        """

        # set up the memory to track the minimum cost to reach a state
        # tells you if your current needs is TK, then if it was pre-computed, this directly returns the result
        memo = dict()
        return self.dfs(memo, price, special, tuple(needs))