from datetime import date
import numpy as np

selected_events = []


def get_ranked_event(file_path):
    events = dict()
    event_set = []

    with open(file_path) as f:
        for line in f.readlines():
            this_event = line.rstrip().split(",")[1]
            this_date = line.rstrip().split(",")[2]
            year = int(float(this_date[0:4]))
            month = int(float(this_date[4:6]))
            day = int(float(this_date[6:8]))

            num = date(year, month, day) - date(2010, 1, 1)

            if this_event not in events:
                events[this_event] = np.zeros(2038)
            events[this_event][num.days] += 1

    for key in events.keys():
        # print key
        # print np.sum(events[key])
        event_set.append([key, np.var(events[key])])

    event_set.sort(key=lambda x: x[1], reverse=True)

    return event_set


hae_events = get_ranked_event("./data/hae.csv")
non_hae_events = get_ranked_event("./data/nonhae_sorted.csv")

events_length = len(hae_events) + len(non_hae_events)
print events_length

for event in hae_events[0:events_length/20]:
    event_code = event[0]
    if event_code not in selected_events:
        selected_events.append(event_code)

for event in non_hae_events[0:events_length/20]:
    event_code = event[0]
    if event_code not in selected_events:
        selected_events.append(event_code)

print len(selected_events)




