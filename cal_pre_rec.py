import numpy as np


def cal_per_result_over_all(method, v_or_t):
    a = 0
    b = 0
    c = 0
    d = 0
    for num in range(5):
        x = np.loadtxt("./result/" + method + "/" + v_or_t + str(num))
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
    print method
    print "TP: ", a, "FP: ", b, "FN: ", d, "TN: ", c


# cal_per_result_over_all("cnn", "v")
# cal_per_result_over_all("cnn", "t")
# cal_per_result_over_all("random_forest", "v")
# cal_per_result_over_all("random_forest", "t")
# cal_per_result_over_all("lr", "v")
# cal_per_result_over_all("lr", "t")


q = np.array([0.9, 0.7, 0.1, 0.8, 0.5])

print np.argsort(-q)


def cal_pre_rec(method, v_or_t):
    for num in range(5):
        precision = 0
        recall = 0
        tag = []
        x = np.loadtxt("./result/" + method + "/" + v_or_t + str(num))
        tmp = np.argsort(-x[:, 1])
        for i in range(len(tmp)):
            if tmp[i] < 204:
                precision += 1
            if precision >= 204/20*(len(tag) + 1):
                tag.append(round(float(precision) / float(i) * 100, 2))
        print tag


cal_pre_rec("lr", "t")

