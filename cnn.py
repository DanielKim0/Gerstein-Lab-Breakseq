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
        model.add(Dense(8, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def model_run(self):
        estimator = KerasClassifier(build_fn=self.model_build, epochs=200, batch_size=5, verbose=0)
        estimator.fit(self.X_train, self.y_train)
        predictions = estimator.predict(self.X_test)
        predictions = np_utils.to_categorical(predictions)
        accu_test = np.sum(self.y_test == predictions) / self.y_test.size

        self.results.write("Model Results\n")
        self.results.write("Model Accuracy:" + str(accu_test) + "\n")

    def kfold_run(self):
        estimator = KerasClassifier(build_fn=self.model_build, epochs=200, batch_size=5, verbose=0)
        super().kfold_run(estimator)

if __name__ == "__main__":
    cnn = CNNModel(sys.argv[1])
    cnn.model_run()
    cnn.kfold_run()