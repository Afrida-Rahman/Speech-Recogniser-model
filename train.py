import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow import optimizers
import h5py

data_path = "data_16c.json"
saved_model_path = "saved_model_100epoch.h5"

learning_rate = 0.0001
epochs = 100
batch_size = 32
num_of_keyword = 16


def load_dataset(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data['MFCCs'])
    y = np.array(data['labels'])
    with h5py.File('metadata.h5', 'w') as hf:
        hf.create_dataset("class", data=y)
    return X, y


def get_data_split(data_path, test_size=.1, test_validation=.1):
    # load dataset
    X, y = load_dataset(data_path)

    # create train,test,validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=test_validation)

    # convert inputs from 2d to 3d arrays
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_test, X_validation, y_train, y_test, y_validation


def build_model(input_shape, learning_rate, loss="sparse_categorical_crossentropy"): #sparse_categorical_crossentropy
    # build network
    model = keras.Sequential()

    # conv1
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape,
                                     kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'))

    # conv2
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu',
                                     kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    # conv3
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu',
                                     kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2,2), padding='same'))

    # flatten the output feed to output layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    # softmax classifier
    model.add(keras.layers.Dense(16, activation='softmax'))

    optimiser = optimizers.Adam(learning_rate=learning_rate)

    # compile model
    model.compile(optimizer=optimiser,
                  loss=loss,
                  metrics=["accuracy"])

    model.summary()

    return model


# def main():
#     # load train,validation,test split
#     X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_split(data_path)
#
#     # build CNN model
#     input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])  # segment (hopfield=512), coefficients 13, 1
#     model = build_model(input_shape, learning_rate)
#
#     # train the model
#     model.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=(X_validation, y_validation))
#
#     # evaluate the model
#     test_error, test_accuracy = model.evaluate(X_test, y_test)
#     print(f'test_err:{test_error},test_acc:{test_accuracy}')
#
#     # save the model
#     model.save(saved_model_path)


# if __name__ == '__main__':
#     main()

X, y = load_dataset(data_path)
