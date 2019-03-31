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
        super().__init__(data_file, "results_random_forest.txt", 0.2)

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
        return autoencoder, encoded6

    def auto_run(self, autoencoder, input_dim, encoded_layer):
        autoencoder.fit(self.X_train, self.X_train, nb_epoch=10, batch_size=32, shuffle=False,
                        validation_data=(self.X_test, self.X_test))
        encoder = Model(inputs=input_dim, outputs=encoded_layer)

        self.X_test = encoder.predict(self.X_test)
        self.X_train = encoder.predict(self.X_test)

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

    def kfold_run(self, estimators):
        model = self.model_build()
        super().kfold_run(model)

if __name__ == "__main__":
    auto = AutoencoderModel(sys.argv[1])
    encoder, encoded_layer = auto.auto_build(auto.X.shape[1])
    auto.auto_run(encoder, auto.X.shape[1], encoded_layer)
    auto.model_run()
    auto.kfold_run()