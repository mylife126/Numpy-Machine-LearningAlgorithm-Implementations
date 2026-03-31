"""
关键技巧是：
split + reverse lookup

核心想法 对于每个 word：
们尝试把它拆成： prefix | suffix
word = "abcd"

"" | abcd
"a" | bcd
"ab" | cd
"abc" | d
"abcd" | ""

然后有两个关键情况：
情况1
如果： prefix 是 palindrome, reverse(suffix)
word = "lls"

prefix = "ll"
suffix = "s"
"ll" 是 palindrome, reverse("s") = "s"
如果 "s" 在 words 中存在：

就可以形成："s" + "lls"

情况2
suffix 是 palindrome
我们找reverse(prefix)
例如：
word = "s"

prefix = ""
suffix = "s"

reverse(prefix) = “”

如果 “” 在字典中

就可以组成："s" + ""
"""
class Solution(object):

    def ispalindrome(self, s):
        left = 0
        right = len(s) - 1

        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1

        return True


    def palindromePairs(self, words):

        word_to_index = {}

        for index, word in enumerate(words):
            word_to_index[word] = index

        result = []

        for index, word in enumerate(words):
            word_length = len(word)

            for split_position in range(word_length + 1):

                prefix = word[:split_position]
                suffix = word[split_position:]



                # --------------------------------------------------
                # Case 1
                #
                # if prefix is palindrome
                # we need reverse(suffix) in front
                #
                # reverse(suffix) + word
                # --------------------------------------------------
                if self.ispalindrome(prefix):

                    reversed_suffix = suffix[::-1]

                    if reversed_suffix in word_to_index and index != word_to_index[reversed_suffix]:
                        result.append([word_to_index[reversed_suffix], index])

                # --------------------------------------------------
                # Case 2
                #
                # if suffix is palindrome
                # we need reverse(prefix) after the word
                #
                # word + reverse(prefix)
                # --------------------------------------------------
                # we avoid duplicate when split_position == word_length
                # duplicate happens in case such as words = ["abcd","dcba"]
                # Case1 已经产生： [1,0] without controlling the duplication prevention
                # 循环到dcba的时候 splitposition == wordlength

                if split_position != word_length and self.ispalindrome(suffix):

                    reversed_prefix = prefix[::-1]

                    if reversed_prefix in word_to_index and word_to_index[reversed_prefix] != index:
                        result.append([index, word_to_index[reversed_prefix]])

        return result


class Solution(object):
    def palindromePairs(self, words):

        # ----------------------------------------
        # Step1: build lookup map
        # ----------------------------------------
        word_to_index = {}

        for i in range(len(words)):
            word_to_index[words[i]] = i

        result = []

        # ----------------------------------------
        # Step2: iterate each word
        # ----------------------------------------
        for i in range(len(words)):
            word = words[i]
            length = len(word)

            # ----------------------------------------
            # Step3: try all splits
            # ----------------------------------------
            for split in range(length + 1):

                prefix = word[:split]
                suffix = word[split:]

                # ----------------------------------------
                # Case1: prefix is palindrome
                # ----------------------------------------
                if self.isPalindrome(prefix):

                    reversed_suffix = suffix[::-1]

                    if reversed_suffix in word_to_index:
                        j = word_to_index[reversed_suffix]

                        if j != i:
                            result.append([j, i])

                # ----------------------------------------
                # Case2: suffix is palindrome
                # ----------------------------------------
                # avoid duplicate when split == length
                if split != length and self.isPalindrome(suffix):

                    reversed_prefix = prefix[::-1]

                    if reversed_prefix in word_to_index:
                        j = word_to_index[reversed_prefix]

                        if j != i:
                            result.append([i, j])

        return result

    def isPalindrome(self, s):
        return s == s[::-1]