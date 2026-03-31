class Solution(object):
    def fullJustify(self, words, maxWidth):

        result = []
        n = len(words)
        index = 0

        while index < n:

            # ----------------------------------------
            # Step1: find how many words fit in one line
            # ----------------------------------------
            line_length = len(words[index])
            last = index + 1

            while last < n:
                if line_length + 1 + len(words[last]) > maxWidth:
                    break
                line_length += 1 + len(words[last])
                last += 1

            # words from index to last-1
            line_words = words[index:last]
            num_words = last - index

            # ----------------------------------------
            # Step2: build the line
            # ----------------------------------------
            line = ""

            # Case1: last line OR single word → left justify
            if last == n or num_words == 1:

                line = " ".join(line_words)

                # pad remaining spaces to right
                spaces_needed = maxWidth - len(line)
                line += " " * spaces_needed

            else:
                # Case2: normal line → distribute spaces

                total_word_length = sum(len(word) for word in line_words)
                total_spaces = maxWidth - total_word_length

                slots = num_words - 1

                spaces_per_slot = total_spaces // slots
                extra_spaces = total_spaces % slots

                for i in range(slots):
                    line += line_words[i]

                    # left slots get extra space
                    spaces_to_add = spaces_per_slot
                    if i < extra_spaces:
                        spaces_to_add += 1

                    line += " " * spaces_to_add

                # add last word
                line += line_words[-1]

            result.append(line)

            index = last

        return result