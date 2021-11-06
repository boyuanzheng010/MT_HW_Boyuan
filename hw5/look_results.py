import re

with open('MT_Results.txt', 'r', encoding='utf-8') as f:
    for x in f:
        print(re.findall("loss_avg:", x))















