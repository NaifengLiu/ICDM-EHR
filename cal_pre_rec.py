import numpy as np


def cal_per_result(method):
    a = 0
    b = 0
    c = 0
    d = 0
    for num in range(5):
        x = np.loadtxt("./result/" + method + "/t" + str(num))
        print x.shape
        print x[:, 0]
        print x[:, 0].shape

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


print cal_per_result("cnn")
print cal_per_result("random_forest")
