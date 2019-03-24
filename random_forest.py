import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def prepare_data(filename):
    dataset = pd.read_csv(sys.argv[1])
    dataset = dataset[dataset.classifs != "\"UNSURE\""]
    dataset.dropna(axis="columns", inplace=True)
    dataset["classifs"] = dataset["classifs"].astype(str)
    X = dataset.iloc[:, 1:-1]
    y = dataset.iloc[:, -1]
    return X, y

def data_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    return X_train, X_test, y_train, y_test

def kfold_y(y):
    le = LabelEncoder()
    y = le.fit_transform(y)
    return y

def random_forest_cycle(X_train, X_test, y_train, y_test):
    results = open("results_forest.txt", "w")
    for estimators in [20, 40, 60, 80, 100]:
        clf = RandomForestClassifier(n_estimators=estimators)
        clf.fit(X_train.values, y_train)
        accu_train = np.sum(clf.predict(X_train.values) == y_train)/y_train.size
        y_score = clf.predict(X_test.values)
        accu_test = np.sum(y_score == y_test)/y_test.size

        results.write("Number of Estimators: " + str(estimators) + "\n")
        results.write("Accuracy on Train: " + str(accu_train) + "\n")
        results.write("Accuracy on Test: " + str(accu_test) + "\n")

def kfold(X, y):
    clf = RandomForestClassifier(n_estimators=100)
    kfold = KFold(n_splits=10)
    results = cross_val_score(clf, X, y, cv=kfold, scoring="accuracy")
    result = results.mean()
    std = results.std()
    results = open("kfold_forest.txt", "w")
    results.write("mean:" + str(result) + "\n")
    results.write("std:" + str(std) + "\n")

def main(file_name):
    X, y = prepare_data(file_name)
    y = kfold_y(y)
    kfold(X, y)

if __name__ == "__main__":
    main(sys.argv[1])