from datetime import date
import numpy as np

events = dict()

with open("./data/combined") as f:
    for line in f.readlines():
        this_event = line.rstrip().split(",")[1]
        this_date = line.rstrip().split(",")[2]
        year = int(float(this_date[0:4]))
        month = int(float(this_date[4:6]))
        day = int(float(this_date[6:8]))

        num = date(year, month, day) - date(2010, 1, 1)

        print num

        if this_event not in events:
            events[this_event] = np.zeros(2038)
        events[this_event][num] += 1

print events







