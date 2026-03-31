class Solution:
    def pourWater(self, heights, volume, pour_index):
        terrain_length = len(heights)
        for drop in range(volume):
            current_pos = pour_index
            scan_pos = current_pos - 1

            best_pos = current_pos
            index_height = heights[pour_index]

            # water will always move to the left until the height before the scan_pos is higher
            while scan_pos >= 0 and heights[scan_pos] <= heights[current_pos]:
                # update the valley only if it is a lower floor
                if heights[scan_pos] < heights[best_pos]:
                    best_pos = scan_pos

                current_pos = scan_pos
                scan_pos -= 1

            # now we want to justify if in the previous left run, it was all flat, meaning that it equals that best pos is the original pour index
            # if so, we need to go to right
            if best_pos != pour_index:
                heights[best_pos] += 1
                continue

            # so, we go to right
            current_pos = pour_index
            best_pos = current_pos

            scan_pos = current_pos + 1
            while scan_pos < terrain_length and heights[scan_pos] <= heights[current_pos]:
                if heights[scan_pos] < heights[best_pos]:
                    best_pos = scan_pos

                current_pos = scan_pos
                scan_pos += 1

            if best_pos != pour_index:
                heights[best_pos] += 1
                continue

            # otherwise, it accumulates at the originally poured index
            heights[pour_index] += 1

        return heights
