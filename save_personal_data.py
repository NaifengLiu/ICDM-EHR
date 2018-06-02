import grouping
names = []

for key in grouping.matching.keys():
    names.append(key)
    names.append(grouping.matching[key][0])

with open("./data/combined_filtered", "w") as w:
    with open("./data/combined") as f:
        for line in f.readlines():
            if int(float(line.split(",")[0])) in names:
                w.write(line)

