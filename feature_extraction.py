from datetime import date

d0 = date(2010, 1, 1)
d1 = date(2015,07,31)
delta = d1 - d0
print delta.days

with open("./data/combined", "w") as w:
    with open("./data/hae.csv") as f:
        for line in f.readlines():
            w.write(line)
        f.close()
    with open("./data/nonhae_sorted.csv") as f:
        for line in f.readlines():
            w.write(line)
        f.close()






