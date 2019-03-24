import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

def prepare_data(filename):
    dataset = pd.read_csv(filename)
    dataset = dataset[dataset.classifs != "\"UNSURE\""]
    dataset.dropna(axis="columns", inplace=True)
    dataset["classifs"] = dataset["classifs"].astype(str)
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1]
    return X, y

def data_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    encoder = LabelEncoder()
    encoder.fit(y)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return X_train, X_test, y_train, y_test

def kfold_y(y):
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    return y

def create_model():
    model = Sequential()
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def kfold(model, X, y):
    estimator = KerasClassifier(build_fn=model, epochs=200, batch_size=5, verbose=1)
    kfold = StratifiedKFold(n_splits=10)
    results = cross_val_score(estimator, X, y, cv=kfold, scoring="accuracy")
    result = results.mean()
    std = results.std()
    results = open("kfold_cnn.txt", "w")
    results.write("mean:" + str(result) + "\n")
    results.write("std:" + str(std) + "\n")

def cnn_results(X_train, X_test, y_train, y_test, estimator):
    estimator.fit(X_train, y_train)
    predictions = estimator.predict(X_test)
    predictions = np_utils.to_categorical(predictions)
    accu_test = np.sum(y_test == predictions)/y_test.size
    results = open("results_cnn.txt", "w")
    results.write("accuracy:" + str(accu_test) + "\n")

def main(file_name):
    X, y = prepare_data(file_name)
    y = kfold_y(y)
    model = create_model()
    kfold(model, X, y)

if __name__ == "__main__":
    main(sys.argv[1])

