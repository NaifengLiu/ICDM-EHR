from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D
import keras
import numpy as np
from gensim.models import word2vec
import process_raw_data
from tqdm import tqdm
import grouping

# load w2v model
embedding_model = word2vec.Word2Vec.load("./data/50features_1minwords_10context")
# prove it works
# print(embedding_model['V2389'])


batch_size = 32
epochs = 7
num_classes = 2

# generating random word2vec result, a matrix, for per person
# for example, suppose there're x days, and the dimension of
# word2vec vector is 50, so, each patient has a matrix of x * 50
# in this example, we suppose x as 1000
# in this example, we suppose there's 500 patients

# x_train = np.random.rand(500, 1000, 50)
# x_train = x_train.reshape(x_train.shape[0], 1000, 50, 1)
# x_test = np.random.rand(100, 1000, 50)
# x_test = x_test.reshape(x_test.shape[0], 1000, 50, 1)
# y_train = np.random.randint(2, size=500)
# y_test = np.random.randint(2, size=100)
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

# previous test case
# x_train, h_list, nh_list = process_raw_data.get_training_data(800, 800)
# x_train = x_train.reshape(x_train.shape[0], 292, 50, 1)
# y_train = np.concatenate((np.zeros(800) + 1, np.zeros(800)), axis=0)
# y_train = keras.utils.to_categorical(y_train, num_classes)

input_shape = (292, 50, 1)
x_test = []
y_test = []


def toInt(string):
    return int(float(string))


def cnn(x_train, validation_set, test_set):
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
    tmp = model.predict(x_train)
    np.savetxt("../result/tmp", tmp)
    test_output = model.predict(test_set)
    return validation_output, test_output


def cnn_LSTM(x_train, validation_set, test_set):
    x_train = np.array(x_train)
    # x_train = x_train.reshape(x_train.shape[0], 292, 50, 1)

    dataset_size = len(x_train)
    x_train = x_train.reshape(dataset_size, -1)
    print(x_train.shape)

    y_train = np.concatenate((np.zeros(788) + 1, np.zeros(788)), axis=0)
    y_train = keras.utils.to_categorical(y_train, num_classes)

    tmp_test_x = np.concatenate((test_set[0:100], test_set[-100:]), axis=0)
    tmp_test_y = np.concatenate((np.zeros(100) + 1, np.zeros(100)), axis=0)
    tmp_test_y = keras.utils.to_categorical(tmp_test_y, num_classes)
    model = Sequential()
    model.add(Conv1D(32, kernel_size=5,
                     activation='relu',
                     input_shape=(292*50, dataset_size)))
    model.add(MaxPooling1D(pool_size=(2, 2)))
    model.add(LSTM(70))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1)

    tmp_test_x = tmp_test_x.reshape(dataset_size, -1)
    validation_set = validation_set.reshape(dataset_size, -1)
    test_set = test_set.reshape(dataset_size, -1)

    score, acc = model.evaluate(tmp_test_x, tmp_test_y)
    print('\n', 'Test score:', score)
    print('Test accuracy:', acc)

    validation_output = model.predict(validation_set)
    test_output = model.predict(test_set)
    return validation_output, test_output


def NN(x_train, validation_set, test_set):
    x_train = np.array(x_train)
    x_train = x_train.reshape(x_train.shape[0], 292, 50, 1)
    y_train = np.concatenate((np.zeros(788) + 1, np.zeros(788)), axis=0)
    y_train = keras.utils.to_categorical(y_train, num_classes)

    tmp_test_x = np.concatenate((test_set[0:100], test_set[-100:]), axis=0)
    tmp_test_y = np.concatenate((np.zeros(100) + 1, np.zeros(100)), axis=0)
    tmp_test_y = keras.utils.to_categorical(tmp_test_y, num_classes)

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    model.add(Dense(64, kernel_initializer="uniform", activation='sigmoid'))
    model.add(Dense(x_train.shape[0], kernel_initializer="uniform", activation='relu'))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    # model.add(Dense(activation="sigmoid", units=1))

    # compile and fit our input matrix and label
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
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



# now we have 985/197000 training set
# so every time we pop 197/39400 as cross validation set
# then we use the rest 788/157600 as per fold training
# in per fold, we use this 788 as positive and randomly pick 788 from 157600 to train a model
# so, we can train 200 times, then we combine the model, evaluate the validation set and test set
# later we add up 5 fold's result, so we will have 2 matrix, for validation and test set

print("start pre-processing")
# x_train_file_names_positive = np.loadtxt("./data/x_train_positive_patient_list").astype(int)
# x_train_file_names_negative = np.loadtxt("./data/x_train_negative_patient_list").astype(int)
# x_test_file_names_positive = np.loadtxt("./data/x_test_positive_patient_list").astype(int)
# x_test_file_names_negative = np.loadtxt("./data/x_test_negative_patient_list").astype(int)

matching = grouping.matching
missing = [1, 3, 8, 72, 165, 183, 223, 239, 285, 305, 324, 348, 376, 416, 446, 461, 486, 489, 548, 575, 599, 612, 632,
           682, 683, 718, 721, 757, 777, 792, 799, 816, 819, 998, 1023, 1034, 1075, 1080, 1118, 1123, 1132, 1146, 1166,
           1215]

matching_keys = matching.keys()

x_train_file_names_positive = []
x_test_file_names_positive = []

for i in range(len(matching_keys)):
    if i not in missing:
        if len(x_train_file_names_positive) < 985:
            x_train_file_names_positive.append(matching_keys[i])
        else:
            x_test_file_names_positive.append((matching_keys[i]))

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


patient_matrix = process_raw_data.patient_matrix

test_matrix = []
for name in x_test_file_names_positive:
    # test_matrix.append(np.loadtxt("./data/x_test_positive/" + name))
    test_matrix.append(patient_matrix[name])
for name in x_test_file_names_negative:
    # test_matrix.append(np.loadtxt("./data/x_test_negative/" + name))
    test_matrix.append(patient_matrix[name])
test_matrix = np.array(test_matrix)
test_matrix = test_matrix.reshape(test_matrix.shape[0], 292, 50, 1)

count = 0
for fold_num in range(5):
    print("start " + str(fold_num + 1) + " fold")
    tmp_validation_names_positive = x_train_file_names_positive[fold_num * 197:(fold_num + 1) * 197]
    tmp_validation_names_negative = x_train_file_names_negative[fold_num * 39400:(fold_num + 1) * 39400]
    tmp_training_names_positive = \
        [item for item in x_train_file_names_positive if item not in tmp_validation_names_positive]
    tmp_training_names_negative = \
        [item for item in x_train_file_names_negative if item not in tmp_validation_names_negative]

    print("preparing validation matrix")
    validation_matrix = []
    for name in tmp_validation_names_positive:
        # validation_matrix.append(np.loadtxt("./data/x_train_positive/" + name))
        validation_matrix.append(patient_matrix[name])
    for name in tmp_validation_names_negative:
        # validation_matrix.append(np.loadtxt("./data/x_train_negative/" + name))
        validation_matrix.append(patient_matrix[name])
    validation_matrix = np.array(validation_matrix)
    validation_matrix = validation_matrix.reshape(validation_matrix.shape[0], 292, 50, 1)

    for time in tqdm(range(200)):
        tmp_tmp_names_negative = tmp_training_names_negative[time * 788:(time + 1) * 788]
        # then we make the matrix
        print("preparing training matrix")
        tmp_training = []
        for name in tmp_training_names_positive:
            tmp_training.append(patient_matrix[name])
        for name in tmp_tmp_names_negative:
            tmp_training.append(patient_matrix[name])

        v_output, t_output = cnn(tmp_training, validation_matrix, test_matrix)
        np.savetxt("../result/cnn/v" + str(count), v_output)
        np.savetxt("../result/cnn/t" + str(count), t_output)

        # v_output, t_output = cnn_LSTM(tmp_training, validation_matrix, test_matrix)
        # np.savetxt("../result/LSTM/v" + str(count), v_output)
        # np.savetxt("../result/LSTM/t" + str(count), t_output)
        #
        # v_output, t_output = NN(tmp_training, validation_matrix, test_matrix)
        # np.savetxt("../result/NN/v" + str(count), v_output)
        # np.savetxt("../result/NN/t" + str(count), t_output)


        count += 1
