from datetime import date
import numpy as np
import grouping
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D
import keras
from sklearn.ensemble import RandomForestClassifier

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


# people_1 = []
# people_2 = []
# with open("./data/combined") as f:
#     for line in f.readlines():
#         split = line.split(",")
#         people_1.append(split[0])
# with open("./data/combined_filtered") as f:
#     for line in f.readlines():
#         split = line.split(",")
#         people_2.append(split[0])
# people_1 = list(set(people_1))
# people_2 = list(set(people_2))
# invalid_people = [x for x in people_1 not in people_2]


# hae_events = get_ranked_event("./data/hae.csv")
# non_hae_events = get_ranked_event("./data/nonhae_sorted.csv")
#
# for event in hae_events[0:2901]:
#     event_code = event[0]
#     if event_code not in selected_events:
#         selected_events.append(event_code)
#
# for event in non_hae_events[0:2901]:
#     event_code = event[0]
#     if event_code not in selected_events:
#         selected_events.append(event_code)
#
# print len(selected_events)

with open("./data/selected_events") as f:
    selected_events = f.readline().split(",")

patient_feature_after_extraction = dict()


def get_patient_features():

    with open("./data/combined_filtered") as f:
        for line in f.readlines():
            split = line.rstrip().split(",")
            patient_id = split[0]
            if patient_id not in patient_feature_after_extraction:
                patient_feature_after_extraction[patient_id] = np.zeros((3767, 2038))
            this_date = split[2]
            year = int(float(this_date[0:4]))
            month = int(float(this_date[4:6]))
            day = int(float(this_date[6:8]))
            num = date(year, month, day) - date(2010, 1, 1)
            days_num = num.days
            this_event = selected_events.index(split[1])
            patient_feature_after_extraction[patient_id][this_event][days_num] += 1
        f.close()


def random_forest(x_train, validation_set, test_set):
    y_train = np.concatenate((np.zeros(788) + 1, np.zeros(788)), axis=0)
    x_train = np.array(x_train)

    print(x_train.shape)
    print(y_train.shape)

    nsamples, nx, ny= x_train.shape
    x_train_reshape = x_train.reshape((nsamples, nx * ny))

    nsamples, nx, ny = validation_set.shape
    validation_set_reshape = validation_set.reshape((nsamples, nx * ny))

    nsamples, nx, ny = test_set.shape
    test_set_reshape = test_set.reshape((nsamples, nx * ny))

    print(x_train_reshape.shape)

    clf = RandomForestClassifier()
    clf.fit(x_train_reshape, y_train)

    validation_output = clf.predict(validation_set_reshape)
    test_output = clf.predict(test_set_reshape)
    return validation_output, test_output


def cnn(x_train, validation_set, test_set, batch_size=32, epochs=7, num_classes=2, input_shape=(3767, 2038, 1)):
    x_train = np.array(x_train)
    x_train = x_train.reshape(x_train.shape[0], 292, 50, 1)
    y_train = np.concatenate((np.zeros(788) + 1, np.zeros(788)), axis=0)
    y_train = keras.utils.to_categorical(y_train, num_classes)

    tmp_test_x = np.concatenate((test_set[0:100], test_set[-100:]), axis=0)
    tmp_test_y = np.concatenate((np.zeros(100) + 1, np.zeros(100)), axis=0)
    tmp_test_y = keras.utils.to_categorical(tmp_test_y, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1)

    score, acc = model.evaluate(tmp_test_x, tmp_test_y)
    print('\n', 'Test score:', score)
    print('Test accuracy:', acc)

    validation_output = model.predict(validation_set)
    test_output = model.predict(test_set)
    return validation_output, test_output


def main(method):

    print("start pre-processing")

    matching = grouping.matching
    missing = [1, 3, 8, 72, 165, 183, 223, 239, 285, 305, 324, 348, 376, 416, 446, 461, 486, 489, 548, 575, 599, 612,
               632, 682, 683, 718, 721, 757, 777, 792, 799, 816, 819, 998, 1023, 1034, 1075, 1080, 1118, 1123, 1132,
               1146, 1166, 1215]

    matching_keys = matching.keys()

    get_patient_features()

    x_train_file_names_positive = []
    x_test_file_names_positive = []

    for i in range(len(matching_keys)):
        if i not in missing:
            if len(x_train_file_names_positive) < 985:
                x_train_file_names_positive.append(matching_keys[i])
            else:
                x_test_file_names_positive.append((matching_keys[i]))

    a = 0
    b = 0

    for item in patient_feature_after_extraction.keys():
        if item in matching_keys:
            a += 1
        else:
            b += 1
    print a, b

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

    patient_matrix = patient_feature_after_extraction

    test_matrix = []
    for name in x_test_file_names_positive:
        # test_matrix.append(np.loadtxt("./data/x_test_positive/" + name))
        if name in patient_matrix:
            print patient_matrix[name].shape
            print patient_feature_after_extraction[name].shape
            test_matrix.append(patient_matrix[name])

            print patient_matrix[name].shape
    for name in x_test_file_names_negative:
        # test_matrix.append(np.loadtxt("./data/x_test_negative/" + name))
        test_matrix.append(patient_matrix[name])
    test_matrix = np.array(test_matrix)
    test_matrix = test_matrix.reshape(test_matrix.shape[0], 3767, 2038, 1)

    count = 0
    for fold_num in range(5):
        print("start " + str(fold_num + 1) + " fold")
        tmp_validation_names_positive = x_train_file_names_positive[fold_num * 197:(fold_num + 1) * 197]
        tmp_validation_names_negative = x_train_file_names_negative[fold_num * 39400:(fold_num + 1) * 39400]
        tmp_training_names_positive = \
            [item for item in x_train_file_names_positive if item not in tmp_validation_names_positive]

        print("preparing validation matrix")
        validation_matrix = []
        for name in tmp_validation_names_positive:
            validation_matrix.append(patient_matrix[name])
        for name in tmp_validation_names_negative:
            validation_matrix.append(patient_matrix[name])
        validation_matrix = np.array(validation_matrix)
        validation_matrix = validation_matrix.reshape(validation_matrix.shape[0], 3767, 2038, 1)

        print("preparing training matrix")
        tmp_training = []
        for name in tmp_training_names_positive:
            tmp_training.append(patient_matrix[name])
        for name in tmp_training_names_positive:
            tmp_training.append(patient_matrix[random.choice(matching[name])])
        if method == "cnn":
            v_output, t_output = cnn(tmp_training, validation_matrix, test_matrix, epochs=10)
            np.savetxt("./result/cnn/fv" + str(count), v_output)
            np.savetxt("./result/cnn/ft" + str(count), t_output)
        elif method == "random_forest":
            v_output, t_output = random_forest(tmp_training, validation_matrix, test_matrix)
            np.savetxt("./result/random_forest/fv" + str(count), v_output)
            np.savetxt("./result/random_forest/ft" + str(count), t_output)

        count += 1


# get_patient_features()
main("cnn")
main("random_forest")
