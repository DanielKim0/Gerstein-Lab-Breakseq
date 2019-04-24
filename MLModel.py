from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from scikitplot.metrics import plot_roc, plot_precision_recall
from keras.utils import to_categorical
import matplotlib.pyplot as plt

class MLModel(object):
    """
    An object that serves as the base class for all of the machine learning models used in this project.
    Contains functions that all machine learning models should have to standardize syntax.
    """
    __metaclass__ = ABCMeta

    def __init__(self, data_file, results_file, split=0, lab=False):
        """
        Initializes the model by taking in object and metadata about the model itself, such as train-test split percentage.
        """
        self.results = open(results_file, "w")
        self.X, self.y = self.data_prepare(data_file)
        self.y = self.data_convert(self.y, lab)
        self.X = scale(self.X)
        if split > 0:
            self.X_train, self.X_test, self.y_train, self.y_test = self.data_train_test(split)
            self.split = True
        else:
            self.split = False
        self.kfold = self.kfold_setup()

    @abstractmethod
    def model_build(self):
        """
        An abstract method that describes how the subclass model will be built.
        """
        pass

    @abstractmethod
    def model_run(self):
        """
        An abstract method that describes how the subclass model will be run.
        """
        pass

    def prc(self, predicted_probs, plot_name, file_name):
        """
        A method that plots a precision recall curve for the given model.
        """
        plot_precision_recall(self.results_convert(self.y), predicted_probs, title=plot_name)
        plt.savefig(file_name)

    def roc(self, predicted_probs, plot_name, file_name, change_probs=False):
        """
        A method that plots a receiver operating characteristic curve for the given model.
        """
        y_final = self.results_convert(self.y)
        if change_probs:
            predicted_probs = np.transpose(np.array([predicted_probs[0][:,1], predicted_probs[1][:,1], predicted_probs[2][:,1]]))
        plot_roc(y_final, predicted_probs, title=plot_name)
        plt.savefig(file_name)

    def model_save(self, model, name):
        """
        A method that pickles a given model.
        """
        pickle.dump(model, open(name + ".p", "wb"))

    def data_prepare(self, data_file):
        """
        Prepares the given data file, which should be formatted like the master data file created at the end of data collection.
        """
        dataset = pd.read_csv(data_file)
        dataset = dataset[dataset.classifs != "\"UNSURE\""]
        dataset.dropna(axis="columns", inplace=True)
        dataset["classifs"] = dataset["classifs"].astype(str)
        X = dataset.iloc[:, 1:-1]
        y = dataset.iloc[:, -1]
        return X, y

    def data_train_test(self, split_size):
        """
        Splits data into train and test.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=split_size, stratify=self.y)
        return X_train, X_test, y_train, y_test

    def data_convert(self, y, lab):
        """
        Converts the 'y' values into encoded (and possibly categorical) numerical labels.
        """
        le = LabelEncoder()
        y = le.fit_transform(y)
        if not lab:
            y = to_categorical(y)
        # self.encoder = le
        return y

    def results_convert(self, results):
        """
        Reverses one-hot encoding (to_categorical) of the results of a model's predictions.
        """
        return list(np.argmax(results, axis=1))

    def kfold_setup(self):
        """
        Sets up a kfold object to use in the below kfold_run function.
        """
        return KFold(n_splits=10, shuffle=True)

    def kfold_run(self, model):
        """
        Runs kfold cross-validation using the given model and writes the results to the model's results file.
        """
        kfold_results = cross_val_score(model, self.X, self.y, cv=self.kfold, scoring="accuracy")
        acc = kfold_results.mean()
        std = kfold_results.std()

        self.results.write("KFold Results\n")
        self.results.write("Scores: " + str(kfold_results) + "\n")
        self.results.write("Accuracy: " + str(acc) + "\n")
        self.results.write("Standard Deviation: " + str(std) + "\n")

    @abstractmethod
    def random_param_tune(self):
        """
        An abstract method that ensures subclass models will be able to randomly tune hyperparameters.
        """
        pass

    @abstractmethod
    def param_tune(self):
        """
        An abstract method that ensures subclass models will be able to finely tune hyperparameters.
        """
        pass
