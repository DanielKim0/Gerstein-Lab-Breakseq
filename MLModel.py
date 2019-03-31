from abc import ABCMeta, abstractmethod
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

class MLModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, data_file, results_file, split=0):
        self.results = open(results_file, "w")
        self.X, self.y = self.data_prepare(data_file)
        self.X = scale(self.X)
        if split > 0:
            self.X_train, self.X_test, self.y_train, self.y_test = self.data_train_test(split)
            self.split = True
        else:
            self.split = False
        self.X, self.y = self.data_convert()
        self.kfold = self.kfold_setup()

    @abstractmethod
    def model_build(self):
        pass

    @abstractmethod
    def model_run(self):
        pass

    def model_save(self, model, name):
        pickle.dump(model, open(name + ".p", "wb"))

    def data_prepare(self, data_file):
        dataset = pd.read_csv(data_file)
        dataset = dataset[dataset.classifs != "\"UNSURE\""]
        dataset.dropna(axis="columns", inplace=True)
        dataset["classifs"] = dataset["classifs"].astype(str)
        X = dataset.iloc[:, 1:-1]
        y = dataset.iloc[:, -1]
        return X, y

    def data_train_test(self, split_size):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=split_size, stratify=self.y)
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.transform(y_test)
        return X_train, X_test, y_train, y_test

    def data_convert(self):
        le = LabelEncoder()
        y = le.fit_transform(self.y)
        return self.X, y

    def kfold_setup(self):
        return StratifiedKFold(n_splits=10, shuffle=True)

    def kfold_run(self, model):
        kfold_results = cross_val_score(model, self.X, self.y, cv=self.kfold, scoring="accuracy")
        acc = kfold_results.mean()
        std = kfold_results.std()

        self.results.write("KFold Results\n")
        self.results.write("Scores: " + str(kfold_results) + "\n")
        self.results.write("Accuracy: " + str(acc) + "\n")
        self.results.write("Standard Deviation: " + str(std) + "\n")
