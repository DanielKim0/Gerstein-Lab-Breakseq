import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from MLModel import MLModel
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from random import sample
from itertools import product

class CNNModel(MLModel):
    def __init__(self, data_file):
        super().__init__(data_file, "results_cnn.txt", 0.3)

    def model_build(self, optimizer='adam', init='glorot_uniform'):
        """
        Builds the CNN used to classify the dataset, and then returns it.
        """
        model = Sequential()
        model.add(Dense(16, kernel_initializer=init, activation='relu'))
        model.add(Dense(8, kernel_initializer=init, activation='relu'))
        model.add(Dense(3, kernel_initializer=init, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def model_run(self):
        """
        Runs the CNN model using the above model_build function and fits/predicts the data using it.
        """
        classifier = KerasClassifier(build_fn=self.model_build, epochs=200, batch_size=5)
        classifier.fit(self.X_train, self.y_train)
        predictions = classifier.predict(self.X_test)
        predictions = np_utils.to_categorical(predictions)
        accu_test = np.sum(self.y_test == predictions) / self.y_test.size

        self.results.write("Model Results\n")
        self.results.write("Model Accuracy:" + str(accu_test) + "\n")
        return classifier

    def model_probs(self, classifier=None):
        """
        Inputs a classifer of the type used above in model_run and returns the probabilities that each item in the dataset belongs to each class.
        """
        if not classifier:
            classifier = KerasClassifier(build_fn=self.model_build, epochs=200, batch_size=5)
            classifier.fit(self.X_train, self.y_train)
        predictions = classifier.predict_proba(self.X)
        return predictions

    def kfold_run(self):
        """
        Runs kfold cross-validation using the generated CNN model.
        """
        svscores = []
        for train, test in self.kfold.split(self.X, self.y):
            model = self.model_build()
            model.fit(self.X[train], self.y[train])
            scores = model.evaluate(self.X[test], self.y[test])
            svscores.append(scores[1] * 100)
        self.results.write("KFold Results\n")
        self.results.write("Scores: " + str(svscores) + "\n")
        self.results.write("Accuracy: " + str(np.mean(svscores)) + "\n")
        self.results.write("Standard Deviation: " + str(np.std(svscores)) + "\n")

    def random_param_tune(self):
        """
        Creates, fits, and predicts a model multiple times with random combinations of hyperparameters, given below,
        in an attempt to find the best set of hyperparameters from a wide range of possibilities.
        """
        best_params = ()
        best_acc = 0

        all_comb = list(product([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], # epochs
                       [1, 3, 5, 7, 9], # batch_size
                       ['glorot_uniform', 'normal', 'uniform'], #init
                       ['rmsprop', 'adam'])) #optimizer
        if len(all_comb) > 250:
            all_comb = sample(all_comb, 250)

        for comb in all_comb:
            cnn = KerasClassifier(build_fn=self.model_build, epochs=comb[0], batch_size=comb[1], init=comb[2], optimizer=comb[3])
            cnn.fit(self.X_train, self.y_train)
            predictions = cnn.predict(self.X_test)
            predictions = np_utils.to_categorical(predictions)
            accu_test = np.sum(self.y_test == predictions) / self.y_test.size
            if accu_test > best_acc:
                best_params = comb
                best_acc = accu_test
        self.results.write("Random Param Tune Results\n")
        self.results.write(str(best_params) + "\n")
        self.results.write(str(best_acc) + "\n")

    def param_tune(self):
        """
        Creates, fits, and predicts a model multiple times with every combination of hyperparameters, given below,
        in an attempt to fine-tune the model using more precise possibilities than the random tuning above.
        """
        best_params = ()
        best_acc = 0

        for comb in list(product([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], # epochs
                       [1, 3, 5, 7, 9], # batch_size
                       ['glorot_uniform', 'normal', 'uniform'], #init
                       ['rmsprop', 'adam'])): #optimizer
            cnn = KerasClassifier(build_fn=self.model_build, epochs=comb[0], batch_size=comb[1], init=comb[2], optimizer=comb[3])
            cnn.fit(self.X_train, self.y_train)
            predictions = cnn.predict(self.X_test)
            predictions = np_utils.to_categorical(predictions)
            accu_test = np.sum(self.y_test == predictions) / self.y_test.size
            if accu_test > best_acc:
                best_params = comb
                best_acc = accu_test
        self.results.write("Param Tune Results\n")
        self.results.write(str(best_params) + "\n")
        self.results.write(str(best_acc) + "\n")

if __name__ == "__main__":
    cnn = CNNModel(sys.argv[1])
    classifier = cnn.model_run()
    cnn.kfold_run()

    probs = cnn.model_probs(classifier=classifier)
    cnn.roc(probs, "CNN ROC Graph", "cnn_roc.png")

    cnn.random_param_tune()
