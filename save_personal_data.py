import grouping

names = []

for key in grouping.matching.keys():
    names.append(key)
    names.append(grouping.matching[key][0])

print names
print len(names)
