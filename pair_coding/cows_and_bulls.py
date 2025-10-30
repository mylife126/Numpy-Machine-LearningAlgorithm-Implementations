from collections import Counter
import sys


class CowsAndBullsGame:
    """
    Modularized implementation for the Cows and Bulls game.
    """

    def __init__(self, secret: str, guess: str):
        self.secret = secret
        self.guess = guess

    def compare_digits(self):
        """Counts bulls and cows."""
        bulls = 0
        cows = 0
        secret_unmatched = []
        guess_unmatched = []

        # First pass: find bulls, record unmatched digits
        for s, g in zip(self.secret, self.guess):
            if s == g:
                bulls += 1
            else:
                secret_unmatched.append(s)
                guess_unmatched.append(g)

        # Second pass: count cows using Counter
        secret_counter = Counter(secret_unmatched)
        guess_counter = Counter(guess_unmatched)
        for ch in guess_counter:
            if ch in secret_counter:
                cows += min(secret_counter[ch], guess_counter[ch])

        return bulls, cows

    def format_hint(self, bulls: int, cows: int):
        """Format result like '1A3B'."""
        return f"{bulls}A{cows}B"

    def get_hint(self):
        """Compute final hint."""
        bulls, cows = self.compare_digits()
        return self.format_hint(bulls, cows)


def main():
    print("Please input the secret:")
    secret = input().strip()

    print("Please input the guess:")
    guess = input().strip()

    game = CowsAndBullsGame(secret, guess)
    print(game.get_hint())

    print("Do you want to stop? (y/n)")
    stop = input().strip().lower()
    if stop == "y":
        return False
    return True


if __name__ == "__main__":
    flag = True
    while flag:
        flag = main()
