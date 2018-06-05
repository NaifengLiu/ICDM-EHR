from gensim.models import word2vec
import numpy as np
import random
from tqdm import tqdm

embedding_model = word2vec.Word2Vec.load(
    "./data/model_without_feature_selection")

positive = []
negative = []
total = []


def read_patient(file_path, group):
    with open(file_path) as r:
        for line in r:
            patient_id = int(float(str(line).split(",")[0]))
            group.append(patient_id)
            total.append(patient_id)


# read_patient("/home/naifeng/Projects/EHR-Deep-Learning/data/hae_dx_new.csv", positive)
# read_patient("/home/naifeng/Projects/EHR-Deep-Learning/data/hae_rx.csv", positive)
read_patient("./data/hae.csv", positive)
# read_patient("/home/naifeng/Projects/EHR-Deep-Learning/data/nonhae_rx.csv", negative)
# read_patient("/home/naifeng/Projects/EHR-Deep-Learning/data/nonhae_rx.csv", negative)
read_patient("./data/nonhae_sorted.csv", negative)

positive = list(set(positive))
negative = list(set(negative))
total = list(set(total))

# prove it works
# print len(positive)
# print len(negative)
# print len(total)


patient_record = dict()


def process_patient_raw_record(file_path):
    with open(file_path) as r:
        for line in r:
            patient_id = int(float(str(line).split(",")[0]))
            if patient_id not in patient_record:
                patient_record[patient_id] = []
            patient_record[patient_id].append([int(float(str(line).split(",")[2])), str(line).split(",")[1]])
            # print patient_record[patient_id]


# process_patient_raw_record("/home/naifeng/Projects/EHR-Deep-Learning/data/hae_dx_new.csv")
# process_patient_raw_record("/home/naifeng/Projects/EHR-Deep-Learning/data/hae_rx.csv")
process_patient_raw_record("./data/hae.csv")
# process_patient_raw_record("/home/naifeng/Projects/EHR-Deep-Learning/data/nonhae_dx.csv")
# process_patient_raw_record("/home/naifeng/Projects/EHR-Deep-Learning/data/nonhae_rx.csv")
process_patient_raw_record("./data/nonhae_sorted.csv")

for key in patient_record.iterkeys():
    patient_record[key] = sorted(patient_record[key])


# prove it works
# print patient_record[positive[0]]

# calc max days
# max_days = 0
# for patient_id in patient_record.iterkeys():
#     days = []
#     for item in patient_record[patient_id]:
#         if item[0] not in days:
#             days.append(item[0])
#     if len(days) > max_days:
#         max_days = len(days)
# print "maxdays: " + str(max_days)


def fill_patient_matrix(this_patient_id):
    matrix = []
    dates = []
    vector = []
    # print patient_record[this_patient_id]
    for item in patient_record[this_patient_id]:
        if item[0] not in dates:
            if dates != []:
                vector = np.array(vector)
                matrix.append(vector.sum(axis=0))
            vector = [embedding_model[item[1]]]
            dates.append(item[0])
            if item[0] == patient_record[this_patient_id][-1][0] and item[1] == patient_record[this_patient_id][-1][1]:
                vector = np.array(vector)
                matrix.append(vector.sum(axis=0))
        else:
            if item[0] == patient_record[this_patient_id][-1][0] and item[1] == patient_record[this_patient_id][-1][1]:
                vector = np.array(vector)
                matrix.append(vector.sum(axis=0))
            else:
                vector.append(embedding_model[item[1]])
    # print len(matrix)
    while len(matrix) != 292:
        matrix.append(np.zeros(50))
    matrix = np.array(matrix)

    return matrix

    # print matrix.shape


# fill_patient_matrix(positive[0])


def get_training_data(positive_sample_num, negative_sample_num):
    hae_patient = random.sample(range(len(positive)), positive_sample_num)
    non_hae_patient = random.sample(range(len(negative)), negative_sample_num)

    all_patient_weight = []
    for num in hae_patient:
        all_patient_weight.append(fill_patient_matrix(positive[num]))
    for num in non_hae_patient:
        all_patient_weight.append(fill_patient_matrix(negative[num]))
    all_patient_weight = np.array(all_patient_weight)
    # print all_patient_weight.shape
    # np.savetxt(str(positive_sample_num)+"+"+str(negative_sample_num), all_patient_weight, fmt='%.5f')
    # print all_patient_weight.shape
    return all_patient_weight, hae_patient, non_hae_patient


patient_matrix = dict()


for i in tqdm(range(len(total))):
    patient_id = total[i]
    if patient_id not in patient_matrix:
        patient_matrix[patient_id] = []
    patient_matrix[patient_id].append(fill_patient_matrix(patient_id))


# x_train_positive_matrix = []
# x_train_negative_matrix = []
# x_test_positive_matrix = []
# x_test_negative_matrix = []
# x_train_positive = positive[0:985]
# x_train_negative = negative[0:197000]
# x_test_positive = positive[985:]
# x_test_negative = negative[197000:]
#
#
# for i in tqdm(range(len(x_train_positive))):
#     person = x_train_positive[i]
#     np.savetxt("./data/x_train_positive/"+str(person), fill_patient_matrix(person))
# for i in tqdm(range(len(x_train_negative))):
#     person = x_train_negative[i]
#     np.savetxt("./data/x_train_negative/"+str(person), fill_patient_matrix(person))
# for i in tqdm(range(len(x_test_positive))):
#     person = x_test_positive[i]
#     np.savetxt("./data/x_test_positive/"+str(person), fill_patient_matrix(person))
# for i in tqdm(range(len(x_test_negative))):
#     person = x_test_negative[i]
#     np.savetxt("./data/x_test_negative/"+str(person), fill_patient_matrix(person))

print "done loading process_raw_data_without_feature_selection.py"
