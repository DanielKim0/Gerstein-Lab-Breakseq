from keras.models import Model
import numpy as np
import sys
from keras.models import Sequential, Input
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from MLModel import MLModel


class AutoencoderModel(MLModel):
    def __init__(self, data_file):
        super().__init__(data_file, "results_autoencoder.txt", 0.2)

    def auto_build(self, ncol):
        input_dim = Input(shape=(ncol,))

        encoded1 = Dense(50, activation='relu')(input_dim)
        encoded2 = Dense(40, activation='relu')(encoded1)
        encoded3 = Dense(30, activation='relu')(encoded2)
        encoded4 = Dense(20, activation='relu')(encoded3)
        encoded5 = Dense(10, activation='relu')(encoded4)
        encoded6 = Dense(5, activation='relu')(encoded5)

        decoded1 = Dense(10, activation='relu')(encoded6)
        decoded2 = Dense(20, activation='relu')(decoded1)
        decoded3 = Dense(30, activation='relu')(decoded2)
        decoded4 = Dense(40, activation='relu')(decoded3)
        decoded5 = Dense(50, activation='relu')(decoded4)
        decoded6 = Dense(ncol, activation='sigmoid')(decoded5)

        autoencoder = Model(inputs=input_dim, outputs=decoded6)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        return autoencoder, input_dim, decoded6

    def auto_run(self, autoencoder, input_dim, decoded_layer):
        autoencoder.fit(self.X_train, self.X_train, nb_epoch=10, batch_size=32, shuffle=False,
                        validation_data=(self.X_test, self.X_test))
        encoder = Model(inputs=input_dim, outputs=decoded_layer)

        self.X_test = encoder.predict(self.X_test)
        self.X_train = encoder.predict(self.X_train)

    def model_build(self):
        model = Sequential()
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def model_run(self):
        classifier = KerasClassifier(build_fn=self.model_build, epochs=200, batch_size=5, verbose=0)
        classifier.fit(self.X_train, self.y_train)
        predictions = classifier.predict(self.X_test)
        predictions = np_utils.to_categorical(predictions)
        accu_test = np.sum(self.y_test == predictions) / self.y_test.size

        self.results.write("Model Results\n")
        self.results.write("Model Accuracy:" + str(accu_test) + "\n")
        return classifier

    def model_probs(self, classifier=None):
        if not classifier:
            classifier = KerasClassifier(build_fn=self.model_build, epochs=200, batch_size=5, verbose=0)
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
    auto = AutoencoderModel(sys.argv[1])
    encoder, input_dim, decoded_layer = auto.auto_build(auto.X_train.shape[1])
    auto.auto_run(encoder, input_dim, decoded_layer)
    classifier = auto.model_run()
    auto.kfold_run()
    probs = auto.model_probs(classifier=classifier)
    auto.roc(probs, "Autoencoder ROC Graph", "autoencoder_roc.png")
