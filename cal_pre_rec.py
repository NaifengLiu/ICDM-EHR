import numpy as np


def cal_per_result_over_all(method):
    a = 0
    b = 0
    c = 0
    d = 0
    for num in range(5):
        x = np.loadtxt("./result/" + method + "/t" + str(num))
        for i in range(len(x[:, 1])):
            if i < 204:
                if x[:, 1][i] >= 0.5:
                    a += 1
                else:
                    b += 1
            else:
                if x[:, 1][i] >= 0.5:
                    d += 1
                else:
                    c += 1
    return a, b, d, c
