matching = dict()

with open("./data/combined_matching") as f:
    for line in f:
        if int(float(line.split(",")[0])) not in matching:
            matching[int(float(line.split(",")[0]))] = []
        matching[int(float(line.split(",")[0]))].append(int(float(line.split(",")[1])))

print len(matching.keys())
