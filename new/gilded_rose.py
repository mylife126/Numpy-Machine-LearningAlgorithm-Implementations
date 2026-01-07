# -*- coding: utf-8 -*-

class GildedRose(object):

    def __init__(self, items):
        self.items = items

    def update_quality(self):
        for item in self.items:

            # 🧙‍♂️ Step 1: 判断是否是 Conjured 物品
            is_conjured = "Conjured" in item.name

            if item.name != "Aged Brie" and item.name != "Backstage passes to a TAFKAL80ETC concert":
                if item.quality > 0:
                    if item.name != "Sulfuras, Hand of Ragnaros":
                        # 🧙‍♂️ Step 2: 普通物品降品质；Conjured 降得更快
                        degrade = 2 if is_conjured else 1
                        item.quality = item.quality - degrade
            else:
                if item.quality < 50:
                    item.quality = item.quality + 1
                    if item.name == "Backstage passes to a TAFKAL80ETC concert":
                        if item.sell_in < 11:
                            if item.quality < 50:
                                item.quality = item.quality + 1
                        if item.sell_in < 6:
                            if item.quality < 50:
                                item.quality = item.quality + 1

            if item.name != "Sulfuras, Hand of Ragnaros":
                item.sell_in = item.sell_in - 1

            if item.sell_in < 0:
                if item.name != "Aged Brie":
                    if item.name != "Backstage passes to a TAFKAL80ETC concert":
                        if item.quality > 0:
                            if item.name != "Sulfuras, Hand of Ragnaros":
                                # 🧙‍♂️ Step 3: 过期后再降一次，Conjured 降得更快
                                degrade = 2 if is_conjured else 1
                                item.quality = item.quality - degrade
                    else:
                        item.quality = item.quality - item.quality
                else:
                    if item.quality < 50:
                        item.quality = item.quality + 1

            # 🧙‍♂️ Step 4: 品质不能为负
            if item.quality < 0:
                item.quality = 0


class Item:
    def __init__(self, name, sell_in, quality):
        self.name = name
        self.sell_in = sell_in
        self.quality = quality

    def __repr__(self):
        return "%s, %s, %s" % (self.name, self.sell_in, self.quality)


if __name__ == "__main__":
    items = [
        Item("Conjured Mana Cake", 3, 6),
        Item("+5 Dexterity Vest", 10, 20),
    ]
    shop = GildedRose(items)

    for day in range(4):
        print(f"--- Day {day} ---")
        for item in items:
            print(item)
        shop.update_quality()