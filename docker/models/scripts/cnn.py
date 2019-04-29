import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from MLModel import MLModel

class CNNModel(MLModel):
    def __init__(self, data_file):
        super().__init__(data_file, "results_cnn.txt", 0.3)

    def model_build(self):
        model = Sequential()
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def model_run(self):
        classifier = KerasClassifier(build_fn=self.model_build, epochs=200, batch_size=5)
        classifier.fit(self.X_train, self.y_train)
        predictions = classifier.predict(self.X_test)
        predictions = np_utils.to_categorical(predictions)
        accu_test = np.sum(self.y_test == predictions) / self.y_test.size

        self.results.write("Model Results\n")
        self.results.write("Model Accuracy:" + str(accu_test) + "\n")
        return classifier

    def model_probs(self, classifier=None):
        if not classifier:
            classifier = KerasClassifier(build_fn=self.model_build, epochs=200, batch_size=5)
            classifier.fit(self.X_train, self.y_train)
        predictions = classifier.predict_proba(self.X_test)
        return predictions

    def kfold_run(self):
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

if __name__ == "__main__":
    cnn = CNNModel(sys.argv[1])
    classifier = cnn.model_run()
    cnn.kfold_run()
    probs = cnn.model_probs(classifier=classifier)
    cnn.roc(probs, "CNN ROC Graph", "cnn_roc.png")
