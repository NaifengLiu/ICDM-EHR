import numpy as np
import random
import grouping


def get_a_new_file(list_of_names, new_file_name):
    with open("./data/feat/"+new_file_name, "w") as w:
        with open("./data/combined") as f:
            for line in f.readlines():
                if int(float(line.split(",")[0])) in list_of_names:
                    w.write(line)
            f.close()
        w.close()


print("start pre-processing")

matching = grouping.matching
missing = [1, 3, 8, 72, 165, 183, 223, 239, 285, 305, 324, 348, 376, 416, 446, 461, 486, 489, 548, 575, 599, 612,
           632, 682, 683, 718, 721, 757, 777, 792, 799, 816, 819, 998, 1023, 1034, 1075, 1080, 1118, 1123, 1132,
           1146, 1166, 1215]

matching_keys = matching.keys()

x_train_file_names_positive = []
x_test_file_names_positive = []

for i in range(len(matching_keys)):
    if i not in missing:
        if len(x_train_file_names_positive) < 985:
            x_train_file_names_positive.append(matching_keys[i])
        else:
            x_test_file_names_positive.append((matching_keys[i]))

get_a_new_file(x_test_file_names_positive, "x_test_positive")

# edited
x_train_file_names_negative = []
for i in range(200):
    for item in x_train_file_names_positive:
        x_train_file_names_negative.append(matching[item][i])
x_test_file_names_negative = []
for i in range(200):
    for item in x_test_file_names_positive:
        x_test_file_names_negative.append(matching[item][i])
# edited

get_a_new_file(x_test_file_names_negative, "x_test_negative")

count = 0
for fold_num in range(5):
    print("start " + str(fold_num + 1) + " fold")
    tmp_validation_names_positive = x_train_file_names_positive[fold_num * 197:(fold_num + 1) * 197]
    tmp_validation_names_negative = x_train_file_names_negative[fold_num * 39400:(fold_num + 1) * 39400]

    get_a_new_file(tmp_validation_names_positive, "fold_" + str(fold_num) + "_validation_positive")
    get_a_new_file(tmp_validation_names_negative, "fold_" + str(fold_num) + "_validation_negative")

    tmp_training_names_positive = \
        [item for item in x_train_file_names_positive if item not in tmp_validation_names_positive]

    tmp_training_names_negative = []
    for item in tmp_training_names_positive:
        tmp_training_names_negative.append(grouping.matching[item][0])

    get_a_new_file(tmp_training_names_positive, "fold_" + str(fold_num) + "_train_positive")
    get_a_new_file(tmp_training_names_negative, "fold_" + str(fold_num) + "_train_negative")





